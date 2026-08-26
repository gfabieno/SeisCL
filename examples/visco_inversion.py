"""Viscoelastic FWI driver: misfit, gradient, and an L-BFGS loop.

Kept separate from the notebook so the notebook imports it rather than
duplicating the machinery, and so it can be run headless:

    SEISCL_BIN=<build> python examples/visco_inversion.py

Objective and gradient
----------------------
J = 0.5 * sum (d_mod - d_obs)^2 is computed in Python (`SeisCL.misfit`) and
the residual is fed back with `inputres=1`, so the gradient corresponds to
*this* J by construction. That also sidesteps the engine's own `rms` scalar,
which cannot be reproduced from Python (notes/todo.md item 9).

`back_prop_type=2` (frequency domain) is required: back_prop_type=1 rejects
L>0 outright, because reverse-time reconstruction of a dissipative medium is
unconditionally unstable.

Parameterization
----------------
The engine returns d(J)/d(vp, vs, rho, taup, taus). Inverting for attenuation
means taking the taup/taus components. They are ~1e-17 while the model values
are ~1e-2, so the optimiser sees a badly scaled problem unless the variables
are normalised -- `Inversion.scale` does that, and it matters far more than
the choice of optimiser.
"""
import os

import numpy as np

import visco_crosswell as vc


class Inversion:
    """Least-squares viscoelastic FWI for the attenuation parameters."""

    def __init__(self, workdir, true_params, start_params, gradfreqs,
                 invert=("taup", "taus"), nshot=None, mute=8.0, smooth=0.0):
        self.wd = workdir
        self.true = true_params
        self.start = {k: np.array(v, dtype=np.float64)
                      for k, v in start_params.items()}
        self.gradfreqs = np.asarray(gradfreqs, dtype=float)
        self.invert = tuple(invert)
        self.nshot = nshot
        self.mute = mute
        self.smooth = smooth
        self.names = ["vp", "vs", "rho", "taup", "taus"]
        self.nfev = 0
        self.history = []
        self.mask = None

        os.makedirs(self.wd, exist_ok=True)
        os.chdir(self.wd)          # callcmd resolves file_din against the cwd
        self.din = self._observed()
        # Invert only where the gradient is defined. `cropgrad` zeroes the
        # gradient over the absorbing strip (nab+FDOH cells), but J still
        # depends on those cells -- leaving them in the unknowns makes the
        # problem inconsistent: a direction that perturbs them changes J while
        # the reported directional derivative stays 0. That alone made the
        # calibration constant direction-dependent (3.4x spread over random
        # directions) and stalled the line search.
        self.mask = self._gradient_support() & self._mute_srcrec()
        # Scale each inverted field by its starting magnitude so the optimiser
        # works on O(1) variables.
        self.scale = {k: max(float(np.abs(self.start[k]).max()), 1e-12)
                      for k in self.invert}
        # The engine applies a fixed scaling to the adjoint source (res_scale),
        # so its gradient is correct only up to ONE global constant. J and the
        # gradient are therefore not on a consistent scale and any optimiser
        # that trusts both -- L-BFGS-B does -- stalls immediately, seeing a
        # gradient ~1e-20 against a misfit ~1e-9. Calibrate the constant once
        # by a directional finite difference.
        self.gfactor = 1.0
        # tau must stay non-negative: a negative relaxation level is
        # unphysical and drives the moduli through their pole.
        self.lower = np.zeros_like(self.x0())
        # Normalise the objective to O(1). scipy's L-BFGS-B convergence test
        # divides by max(|f|,1), so an objective of ~1e-11 satisfies the
        # default ftol on the very first step and the optimiser returns
        # without iterating -- a silent no-op that looks like convergence.
        self.Jref = 1.0

    # ---------------------------------------------------------------- setup
    def _seiscl(self, **kw):
        s = vc.make_seiscl(self.wd, **kw)
        if self.nshot is not None:
            keep = s.src_pos_all[3, :] < self.nshot
            s.src_pos_all = s.src_pos_all[:, keep]
            s.rec_pos_all = s.rec_pos_all[:, s.rec_pos_all[3, :] < self.nshot]
        return s

    def _observed(self):
        s = self._seiscl()
        s.set_forward(s.src_pos_all[3, :], self.true, withgrad=False)
        s.execute()
        s.write_data({"p": s.read_data()[0]}, filename="inv_din.mat")
        self.dobs = [np.asarray(s.read_data()[0], dtype=np.float64)]
        return os.path.join(self.wd, "inv_din.mat")

    def _mute_srcrec(self):
        """Drop cells within `mute` cells of a source or receiver.

        The gradient is *wrong*, not merely noisy, in those cells: probing it
        with localized bumps gives FD/<g,v> ~ +1.8e8 consistently across the
        interior but the WRONG SIGN on every bump covering an active shot.
        Near-source contamination is standard in FWI and routinely muted; here
        it is also what made the calibration constant look direction-dependent,
        since a broad direction mixes good interior cells with bad source ones.
        """
        s = self._seiscl()
        nz, nx = self.start[self.invert[0]].shape
        zz, xx = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
        keep = np.ones((nz, nx), dtype=bool)
        if self.mute <= 0:
            return keep
        pts = np.concatenate([s.src_pos_all[[0, 2], :],
                              s.rec_pos_all[[0, 2], :]], axis=1) / vc.DH
        for xp, zp in pts.T:
            keep &= ((zz - zp)**2 + (xx - xp)**2) > self.mute**2
        return keep

    def _gradient_support(self):
        """Boolean mask of the cells cropgrad keeps, read off the gradient."""
        s = self._seiscl(gradout=1, back_prop_type=2, inputres=1,
                         gradfreqs=self.gradfreqs)
        s.file_din = self.din
        s.set_forward(s.src_pos_all[3, :], self.start, withgrad=True)
        d = self.forward(self.start)
        _, res = s.misfit(d, dobs=self.dobs)
        s.set_backward(residuals=res)
        s.execute()
        g = s.read_grad()
        m = np.zeros_like(np.asarray(g[0]), dtype=bool)
        for k in self.invert:
            m |= np.asarray(g[self.names.index(k)]) != 0.0
        return m

    def _params(self, x):
        """Rebuild the full parameter dict from the scaled unknowns."""
        p = {k: np.array(v, dtype=np.float64) for k, v in self.start.items()}
        i = 0
        for k in self.invert:
            n = int(self.mask.sum())
            p[k][self.mask] = x[i:i + n] * self.scale[k]
            i += n
        return p

    def x0(self):
        return np.concatenate([(self.start[k][self.mask] / self.scale[k])
                               for k in self.invert])

    # ------------------------------------------------------- misfit + grad
    def forward(self, params):
        s = self._seiscl(seisout=2)
        s.set_forward(s.src_pos_all[3, :], params, withgrad=False)
        s.execute()
        return s.read_data()

    def Jonly(self, x):
        """Misfit alone -- no adjoint run. Half the cost of fun(), used for
        finite-difference probing where the gradient is not needed."""
        s = self._seiscl(gradout=0)
        s.file_din = self.din
        J, _ = s.misfit(self.forward(self._params(x)), dobs=self.dobs)
        return float(J)

    def fun(self, x):
        """(J, dJ/dx) for scipy's L-BFGS-B."""
        params = self._params(x)
        dmod = self.forward(params)

        s = self._seiscl(gradout=1, back_prop_type=2, inputres=1,
                         gradfreqs=self.gradfreqs)
        s.file_din = self.din
        s.set_forward(s.src_pos_all[3, :], params, withgrad=True)
        J, res = s.misfit(dmod, dobs=self.dobs)
        s.set_backward(residuals=res)
        s.execute()
        g = s.read_grad()

        grad = np.concatenate([
            (np.asarray(g[self.names.index(k)], dtype=np.float64)[self.mask]
             * self.scale[k]) for k in self.invert])
        self.nfev += 1
        self.history.append(float(J))
        return float(J)/self.Jref, (self.gfactor/self.Jref) * grad

    # ------------------------------------------------------------- driver
    def calibrate(self, eps=None, seed=0, ndir=2, verbose=True):
        """Recover the global gradient constant by finite differences.

        Sets self.gfactor so that <gfactor*g, v> matches the directional
        derivative of J. Reports the spread over independent directions: a
        constant ratio means the gradient is right up to that one factor.
        """
        x0 = self.x0()
        rng = np.random.default_rng(seed)
        ratios = []
        for d in range(ndir):
            v = rng.standard_normal(x0.size)
            v /= np.linalg.norm(v)
            e = eps if eps is not None else 0.05 * np.abs(x0).max()
            J, g = self.fun(x0)
            gv = float(g @ v)
            Jp, _ = self.fun(x0 + e*v)
            Jm, _ = self.fun(x0 - e*v)
            fd = (Jp - Jm) / (2*e)
            ratios.append(fd / gv)
            if verbose:
                print("    dir %d: FD=%+.4e  <g,v>=%+.4e  ratio=%+.4e"
                      % (d, fd, gv, ratios[-1]))
        r = np.array(ratios)
        self.gfactor = float(np.mean(r))
        if verbose:
            print("    gradient factor = %.4e   (spread %.3f)"
                  % (self.gfactor, r.max()/r.min() if r.min() != 0 else np.nan))
        self.history.clear(); self.nfev = 0
        return self.gfactor

    def run(self, niter=12, step0=1.0, verbose=True, x=None):
        """Steepest descent with a backtracking line search on J.

        Deliberately *not* scipy's L-BFGS-B on (J, grad): the engine scales the
        adjoint source by a fixed res_scale, so its gradient is right only up
        to one global constant, and any optimiser that trusts J and grad
        together stalls at once (measured: grad ~1e-20 against J ~1e-9).
        Calibrating that constant by finite differences is itself unreliable
        here -- J is ~5e-9 and a 5% model perturbation moves it by ~1e-14, a
        relative change of 2e-6 on float32 seismograms, right at the noise
        floor.

        Using the gradient only as a *direction* and letting a line search on
        J choose the length sidesteps the constant entirely. The sign is taken
        from the data: if the first trial increases J, the descent direction is
        the other one.
        """
        x = self.x0().copy() if x is None else x.copy()
        J, g = self.fun(x)
        J0 = J
        if verbose:
            print("    iter  0   J = %.6e" % J)
        sign = None
        step = step0
        for it in range(1, niter + 1):
            # Normalise by max|g|, NOT the L2 norm: with ~5e3 cells a
            # unit-L2 direction moves each cell by ~1/sqrt(N) of the
            # step, so reaching an O(1) model change would take
            # thousands of iterations. max-normalisation makes a step
            # of 1.0 move the strongest cell by one model unit.
            gp = self.precondition(g)
            d = -gp / max(np.abs(gp).max(), 1e-300)
            if sign is None:
                # settle the sign convention once, from a trial step
                Jt, _ = self._trial(np.maximum(x + step*d, self.lower))
                if not np.isfinite(Jt) or Jt > J:
                    Jt2, _ = self._trial(np.maximum(x - step*d, self.lower))
                    sign = -1.0 if (np.isfinite(Jt2) and Jt2 < J) else +1.0
                else:
                    sign = +1.0
                if verbose:
                    print("    (descent sign: %+d)" % sign)
            d = sign * d
            a, ok = step, False
            for _ in range(8):
                xt = np.maximum(x + a*d, self.lower)
                Jt, gt = self._trial(xt)
                if np.isfinite(Jt) and Jt < J:
                    x, J, g, ok = xt, Jt, gt, True
                    break
                a *= 0.5
            if verbose:
                print("    iter %2d   J = %.6e   step = %.3g%s"
                      % (it, J, a, "" if ok else "   (no decrease)"))
            if not ok:
                break
            step = a * 2.0
        self.history = [J0, J]
        self.x = x
        return self._params(x), J0, J

    def run_multiscale(self, stages=((6.0, 8), (3.0, 8), (1.5, 8)),
                       step0=2.0, verbose=True):
        """Smoothing continuation: coarse structure first, then detail.

        Steepest descent with a *fixed* smoothing stalls once the update it can
        represent is exhausted (measured: 36% misfit reduction, then no
        decrease at any step length). Relaxing the smoothing restarts it on the
        finer scales, which is the usual multiscale FWI strategy applied to the
        preconditioner rather than to frequency.
        """
        x, Jfirst, Jlast = None, None, None
        for smooth, n in stages:
            self.smooth = smooth
            if verbose:
                print("  -- stage: smoothing sigma = %.1f cells" % smooth)
            _, J0, J1 = self.run(niter=n, step0=step0, verbose=verbose, x=x)
            x = self.x
            Jfirst = J0 if Jfirst is None else Jfirst
            Jlast = J1
        return self._params(x), Jfirst, Jlast

    def run_lbfgs(self, niter=15, verbose=True):
        """L-BFGS-B on (J, gfactor*grad), with tau bounded below by 0.

        Requires calibrate() first: without the global constant the optimiser
        sees a gradient ~1e-19 against a misfit ~1e-11 and stops at once.
        """
        from scipy.optimize import minimize
        it = {"n": 0}

        def cb(xk):
            it["n"] += 1
            if verbose:
                print("    iter %2d   J = %.6e" % (it["n"], self.history[-1]))

        x0 = self.x0()
        if self.Jref == 1.0:
            self.Jref = max(self.fun(x0)[0], 1e-300)
            self.history.clear()
            if verbose:
                print("    objective normalised by J0 = %.4e" % self.Jref)
        r = minimize(self.fun, x0, jac=True, method="L-BFGS-B", callback=cb,
                     bounds=[(0.0, None)]*x0.size,
                     options={"maxiter": niter, "maxcor": 10})
        return self._params(r.x), r

    def precondition(self, g):
        """Smooth the gradient before using it as a search direction.

        The raw gradient is spiky: its amplitude is set by illumination, so it
        peaks in a thin band near the wells and carries the source-side
        contamination documented in _mute_srcrec. Gaussian smoothing is the
        standard FWI remedy -- it is a preconditioner, so it changes the
        *path* to the minimum, not the minimum itself.
        """
        if self.smooth <= 0:
            return g
        from scipy.ndimage import gaussian_filter
        out = np.zeros_like(g)
        i = 0
        for k in self.invert:
            n = int(self.mask.sum())
            f = np.zeros(self.mask.shape)
            f[self.mask] = g[i:i + n]
            f = gaussian_filter(f, self.smooth)
            out[i:i + n] = f[self.mask]
            i += n
        return out

    def _trial(self, x):
        """Evaluate without polluting the recorded history."""
        n = len(self.history)
        out = self.fun(x)
        del self.history[n:]
        return out


def circular_tau_model(dtau=0.05, radius=12.0):
    """True model: background plus a circular attenuation anomaly."""
    return vc.circular_inclusion(dtaup=dtau, dtaus=dtau, radius=radius)


if __name__ == "__main__":
    wd = os.environ.get("VISCO_WORKDIR",
                        "/userdata/u/gfabien/claude/visco-work/_wd_inv")
    inv = Inversion(wd, circular_tau_model(), vc.background(),
                    gradfreqs=[40.0, 60.0, 80.0], invert=("taus",), nshot=5)
    J0, g0 = inv.fun(inv.x0())
    print("initial J = %.6e   max|grad| = %.3e" % (J0, np.abs(g0).max()))
    print("  calibrating the global gradient constant:")
    inv.calibrate()

    final, res = inv.run(niter=int(os.environ.get("NITER", 8)))
    Js = inv.history
    print("J: %.4e -> %.4e   (%.1f%% reduction, %d evaluations)"
          % (Js[0], Js[-1], 100*(1 - Js[-1]/Js[0]), len(Js)))

    ts_true, ts_rec = inv.true["taus"], final["taus"]
    ts_start = inv.start["taus"]
    err0 = np.linalg.norm(ts_start - ts_true)
    err1 = np.linalg.norm(ts_rec - ts_true)
    print("model error: %.4e -> %.4e   (%.1f%% reduction)"
          % (err0, err1, 100*(1 - err1/err0)))
    print("recovered taus: min %.4f  max %.4f   (true %.4f / %.4f)"
          % (ts_rec.min(), ts_rec.max(), ts_true.min(), ts_true.max()))
