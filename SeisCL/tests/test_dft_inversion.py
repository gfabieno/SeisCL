"""End-to-end FWI driven by the DFT (back_prop_type=2) gradient.

test_dft_gradient.py checks the gradient against finite differences and against
a float64 host reference. That is necessary but not sufficient: a gradient can
pass a directional check and still be unusable in an inversion, because an
optimizer additionally needs the gradient to be consistent with the objective
over a whole line search. This test closes that gap -- it is the only place the
DFT gradient is actually used for what it exists for.

Two deliberate choices:

* **The objective is computed in Python** (`SeisCL.misfit`) and the residual fed
  back with `inputres=1`, so the gradient corresponds to exactly the J being
  minimised. The engine's own `rms` scalar is not reproducible from Python and,
  for back_prop_type=2, is evaluated at slightly different discrete frequencies
  than the gradient.

* **The optimizer is `slbfgs`, not `scipy.optimize`.** The engine scales the
  adjoint source (`res_scale`), so the gradient it returns is proportional to
  the true gradient rather than equal to it. scipy's L-BFGS-B needs an accurate
  directional derivative to satisfy the strong Wolfe conditions and terminates
  `ABNORMAL_TERMINATION_IN_LNSRCH`; slbfgs takes `success_on_decrease`, which
  applies a step that lowers the cost even when Wolfe fails, and only skips the
  s/y update. Note that slbfgs is *not* invariant to the scaling either -- the
  gradient is normalised below before it is handed over.

Run:  SEISCL_BIN=<build> python test_dft_inversion.py
"""
import os
import sys
import unittest

import numpy as np

import gradient_common as gc
from SeisCL.SeisCL import SeisCLError

try:
    from slbfgs import slbfgs
    HAVE_SLBFGS = True
except ImportError:
    HAVE_SLBFGS = False


class Inversion:
    """Minimal FWI driver: misfit, gradient, and the slbfgs plumbing."""

    def __init__(self, wd, gradfreqs, dvp=300.0, mute=4.0):
        self.wd = wd
        self.gradfreqs = np.asarray(gradfreqs, dtype=float)
        s = gc.make_seiscl(wd)
        self.true = gc.with_anomaly(s, dvp=dvp)
        self.start = gc.homogeneous(s)
        self.shape = self.start["vp"].shape
        # Model the observed data and keep it in memory. Writing it out is
        # still required -- the engine reads file_din to size its residual
        # buffers -- but read_data() cannot read that file back: it looks for
        # "<field>out" keys and the din file stores plain "<field>".
        s.set_forward(s.src_pos_all[3, :], self.true, withgrad=False)
        s.execute()
        dobs = s.read_data()
        s.write_data({"p": dobs[0]}, filename="SeisCL_din.mat")
        self.din = os.path.join(s.workdir, "SeisCL_din.mat")
        self.dobs = [np.asarray(a, dtype=np.float64) for a in dobs]
        # Invert vp only, scaled by its background value so the unknowns are
        # O(1). The gradient is returned per (m/s), so it must be scaled the
        # same way for the two to be consistent.
        self.scale = gc.VP
        self.nfev = 0
        self.gnorm = None
        self.mute = self._mute_mask(s, mute)

    @staticmethod
    def _mute_mask(s, radius):
        """Zero the gradient within `radius` cells of a source or receiver.

        Not cosmetic -- without it the inversion cannot start. The raw DFT
        gradient peaks *exactly* at the source (measured: argmax|g| lands on
        the source cell), so the steepest-descent direction is dominated by the
        source singularity and moving along it raises the misfit by 280% at the
        first trial step while shifting vp inside the anomaly by 0.4 m/s. With
        a 4-cell mute the same direction lowers the misfit by 12%. Muting
        around sources and receivers is standard practice in FWI; it is
        recorded here because nothing else in the repository says the DFT
        gradient needs it.
        """
        if radius <= 0:
            return np.ones((int(s.N[0]), int(s.N[1])), dtype=bool)
        zz, xx = np.indices((int(s.N[0]), int(s.N[1])))
        keep = np.ones(zz.shape, dtype=bool)
        pts = np.concatenate([s.src_pos_all[[2, 0], :],
                              s.rec_pos_all[[2, 0], :]], axis=1) / s.dh
        for zp, xp in pts.T:
            keep &= ((zz - zp)**2 + (xx - xp)**2) > radius**2
        return keep

    # Bound constraints, as any real FWI uses. Without them the line search
    # walks straight out of the stable region on its first trial step and the
    # engine aborts with "Time step too large".
    VP_MIN, VP_MAX = 1400.0, 3000.0

    def params(self, x):
        p = {k: np.array(v) for k, v in self.start.items()}
        p["vp"] = np.clip(x[0].reshape(self.shape) * self.scale,
                          self.VP_MIN, self.VP_MAX)
        return p

    def x0(self):
        return [self.start["vp"].ravel() / self.scale]

    def fun(self, x, withgrad=True):
        """(J, [grad], h0) for slbfgs; J alone when withgrad=False.

        The gradient path runs the engine twice, not three times: the
        `withgrad=True` forward already writes the seismograms *and* the
        boundary checkpoint, so its output serves as d_mod and no separate
        modelling run is needed.
        """
        params = self.params(x)
        self.nfev += 1
        try:
            if not withgrad:
                s = gc.make_seiscl(self.wd, seisout=2)
                s.set_forward(s.src_pos_all[3, :], params, withgrad=False)
                s.execute()
                return s.misfit(s.read_data(), dobs=self.dobs)[0]

            g = gc.make_seiscl(self.wd, gradout=1, back_prop_type=2,
                               inputres=1, gradfreqs=self.gradfreqs)
            g.file_din = self.din
            g.set_forward(g.src_pos_all[3, :], params, withgrad=True)
            g.execute()
            J, res = g.misfit(g.read_data(), dobs=self.dobs)
            g.set_backward(residuals=res)
            g.execute()
        except SeisCLError:
            # An unstable trial model. Report +inf so the line search
            # backtracks instead of the whole inversion dying.
            return np.inf if not withgrad else (np.inf,
                                                [np.zeros(x[0].size)], None)

        grad = np.asarray(g.read_grad()[0], dtype=np.float64) * self.scale
        grad = grad * self.mute
        # res_scale leaves an unknown constant on the gradient; slbfgs does not
        # remove it (its `scaler` normalises the direction, not the Wolfe
        # derivative). Normalising puts J and the gradient on a comparable
        # scale, which is what the line search needs.
        #
        # The constant is computed ONCE, at the first gradient evaluation, and
        # reused. Renormalising per iteration would rescale every gradient by a
        # different factor, so L-BFGS's curvature pairs y = g_{k+1} - g_k would
        # mix inconsistent scalings and the inverse-Hessian estimate would be
        # meaningless.
        if self.gnorm is None:
            gmax = np.abs(grad).max()
            self.gnorm = (J / gmax) if gmax > 0 else 1.0
        grad = grad * self.gnorm
        return J, [grad.ravel()], None


@unittest.skipUnless(HAVE_SLBFGS, "slbfgs not installed (pip install slbfgs)")
class TestDFTInversion(unittest.TestCase):
    """The DFT gradient must be good enough to actually drive an inversion."""

    def test_recovers_velocity_anomaly(self):
        inv = Inversion(gc.workdir("inversion"), gradfreqs=[10.0, 15.0, 20.0])
        x0 = inv.x0()
        J0 = inv.fun(x0, withgrad=False)

        # A small initial step: 1.0 would move vp by its full background
        # value on the first trial and land outside the stable region.
        out = slbfgs(inv.fun, x0, 15, alpha=0.02, verbose=False)
        xk, J1 = out[0], out[5]

        vp = xk[0].reshape(inv.shape) * inv.scale
        vtrue, vstart = inv.true["vp"], inv.start["vp"]
        e0 = np.linalg.norm(vstart - vtrue)
        e1 = np.linalg.norm(vp - vtrue)

        nz, nx = inv.shape
        inside = np.zeros(inv.shape, dtype=bool)
        inside[nz//2-5:nz//2+5, nx//2-5:nx//2+5] = True
        dv = vp - vstart

        print("\n  misfit      %.4e -> %.4e   (%.1f%% reduction, %d evaluations)"
              % (J0, J1, 100*(1 - J1/J0), inv.nfev))
        print("  model error %.4e -> %.4e   (%.1f%% reduction)"
              % (e0, e1, 100*(1 - e1/e0)))
        print("  dvp inside anomaly %+.1f m/s, outside %+.1f m/s  (true %+.1f)"
              % (dv[inside].mean(), dv[~inside].mean(), 300.0))

        # Thresholds are set from measured behaviour with margin, not from
        # what a converged FWI would give. One shot into twelve receivers does
        # not resolve a +300 m/s anomaly: the misfit plateaus near 21% and the
        # recovered update is a few m/s. What is being asserted is that the
        # DFT gradient is a usable descent direction pointing at the right
        # place -- which is exactly what a pure finite-difference check on the
        # gradient cannot tell you.
        #
        # Measured over several runs: 20.7-22.9% misfit reduction, update
        # 11-13x more concentrated inside the anomaly than outside.
        self.assertLess(J1, 0.85 * J0)
        self.assertGreater(dv[inside].mean(), 0.0)
        self.assertGreater(dv[inside].mean(), 5.0 * abs(dv[~inside].mean()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
