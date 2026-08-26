"""Builder for docs/notebooks/Inversion/StochasticLBFGS.ipynb.

Generates the notebook from a script (nbformat), then execute it:

    python docs/builders/build_stochastic_lbfgs.py
    PYTHONPATH=<this clone> PATH=<build>:$PATH CUDA_VISIBLE_DEVICES=<idle gpu> \
        python -m nbclient docs/notebooks/Inversion/StochasticLBFGS.ipynb

Elastic (L=0), so either the CUDA or the OpenCL build works. Runs a full
crosswell inversion with the back-propagation gradient in a few minutes; the
DFT gradient is evaluated once to show it is currently zero at this NT
(see notes/todo.md item 0b), not run through an optimizer.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md(r"""# FWI using Stochastic L-BFGS

This notebook runs a crosswell full-waveform inversion (FWI) with the
stochastic L-BFGS (S-LBFGS) optimizer, using SeisCL's back-propagation
gradient. Section 5 also evaluates the frequency-domain/DFT gradient on the
same problem, which is currently zero at this grid size -- see that section
for what that means and where the DFT gradient does work.

## The stochastic L-BFGS method

L-BFGS speeds up FWI's convergence over plain steepest descent by building an
approximation of the inverse Hessian from a short history of the last $n$
pairs of parameter changes $s_k = m_{k+1}-m_k$ and gradient changes
$y_k = \nabla \chi_{k+1} - \nabla \chi_k$. Applying that approximation to the
current gradient only costs a two-loop recursion over those $n$ vector
pairs, and a Wolfe-condition line search picks the step length, which keeps
the search direction a genuine descent direction.

The stochastic variant additionally draws a random *subset* of the sources
at every iteration, instead of always modeling the full source ensemble --
the survey redundancy in most seismic acquisitions means a subset is often
enough to estimate the gradient, and the forward/adjoint pair per source is
FWI's dominant cost. Mixing subsampling with L-BFGS naively is unstable: if
the source subset changes between the two gradient evaluations that build one
$(s_k, y_k)$ pair, that pair is dominated by sampling noise rather than
curvature. S-LBFGS fixes this by drawing one random source subset per
iteration and using that *same* subset for both gradient evaluations needed
to form the pair, at no extra forward-modeling cost over plain L-BFGS.

Reference:

> Fabien-Ouellet, G., Gloaguen, E., and Giroux, B., 2017, A stochastic L-BFGS
> approach for full waveform inversion: SEG Technical Program Expanded
> Abstracts 2017, p. 1622-1626, doi: 10.1190/segam2017-17783222.1

The `slbfgs` package (https://github.com/gfabieno/slbfgs) implements this
algorithm; install it with `pip install git+https://github.com/gfabieno/slbfgs`
or from a local checkout with `pip install -e .`.""")

code("""import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from SeisCL.SeisCL import SeisCL, SeisCLError
from SeisCL.torch import Config, seiscl_forward
from slbfgs import slbfgs

workdir = os.environ.get("SLBFGS_WORKDIR", os.path.abspath("_nb_slbfgs"))
os.makedirs(workdir, exist_ok=True)""")

md(r"""## 1. The experiment

A crosswell survey: 10 sources down a well on the left, a string of
receivers down a well on the right. The grid is 200x200 cells -- large
enough, relative to the width of the absorbing boundary (`nab`), that the
region actually being imaged is not dominated by boundary effects.""")

code("""NZ, NX = 200, 200
DH = 10.0
DT = 1.5e-3
NT = 1200
NAB = 20
F0 = 10.0

VP0, VS0, RHO0 = 2000.0, 1200.0, 2000.0
VP_MIN, VP_MAX = 1400.0, 3000.0
ANOMALY_RADIUS = 15.0
DVP = 100.0
MUTE_RADIUS = 4.0

N_SOURCES = 10
N_RECEIVERS = 40


def circular_inclusion(dvp=DVP, radius=ANOMALY_RADIUS):
    \"\"\"Background model plus a circular vp anomaly at the centre.\"\"\"
    zz, xx = np.meshgrid(np.arange(NZ), np.arange(NX), indexing="ij")
    r = np.sqrt((zz - NZ / 2.0) ** 2 + (xx - NX / 2.0) ** 2)
    inside = (r <= radius).astype(np.float64)
    return {"vp": np.full((NZ, NX), VP0) + dvp * inside,
            "vs": np.full((NZ, NX), VS0),
            "rho": np.full((NZ, NX), RHO0)}


def homogeneous():
    return circular_inclusion(dvp=0.0)


def make_seiscl(wd, **overrides):
    cfg = dict(N=np.array([NZ, NX]), ND=2, dh=DH, dt=DT, NT=NT, FDORDER=8,
              freesurf=0, abs_type=2, nab=NAB, f0=F0, seisout=1,
              param_type=0)
    cfg.update(overrides)
    s = SeisCL(workdir=wd, **cfg)

    # Sources down the left well.
    xs = (NAB + 10) * DH
    zs = np.linspace((NAB + 15) * DH, (NZ - NAB - 15) * DH, N_SOURCES)
    s.src_pos_all = np.stack([np.full(N_SOURCES, xs), np.zeros(N_SOURCES), zs,
                              np.arange(N_SOURCES, dtype=float),
                              np.full(N_SOURCES, 100.0)])

    # Receivers down the right well, responding to every source.
    xr = (NX - NAB - 10) * DH
    zr = np.linspace((NAB + 10) * DH, (NZ - NAB - 10) * DH, N_RECEIVERS)
    src_ids = np.arange(N_SOURCES, dtype=float)
    xr_all = np.tile(xr, N_SOURCES * N_RECEIVERS)
    zr_all = np.tile(zr, N_SOURCES)
    srcid_all = np.repeat(src_ids, N_RECEIVERS)
    recid_all = np.tile(np.arange(1, N_RECEIVERS + 1, dtype=float), N_SOURCES)
    n = N_SOURCES * N_RECEIVERS
    s.rec_pos_all = np.stack([xr_all, np.zeros(n), zr_all, srcid_all,
                              recid_all, np.zeros(n), np.zeros(n),
                              np.zeros(n)])

    s.src_all = None
    return s


s0 = make_seiscl(workdir)
print("grid %dx%d, dh=%.0f m, dt=%.1e s, NT=%d, %d source(s), %d receiver(s)"
      % (NZ, NX, DH, DT, NT, s0.src_pos_all.shape[1], s0.rec_pos_all.shape[1]))""")

md(r"""## 2. True model, observed data, starting model

Observed data is modeled once, from all 10 sources, on the true model. Each
inversion run below only ever *uses* a random subset of these sources per
iteration, selecting the matching subset of traces from this one dataset.""")

code("""true_model = circular_inclusion()
start_model = homogeneous()

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, m, title in zip(axes, [true_model, start_model], ["true", "starting"]):
    im = ax.imshow(m["vp"], extent=[0, NX * DH, NZ * DH, 0], cmap="viridis")
    ax.plot(s0.src_pos_all[0], s0.src_pos_all[2], "r*", ms=8, label="source")
    ax.plot(s0.rec_pos_all[0, :N_RECEIVERS], s0.rec_pos_all[2, :N_RECEIVERS],
            "kv", ms=3, label="receiver")
    ax.set_title(title + " model")
    ax.set_xlabel("x (m)")
axes[0].set_ylabel("z (m)")
axes[0].legend(loc="upper right", fontsize=8)
fig.colorbar(im, ax=axes, label="vp (m/s)", shrink=0.8)
plt.show()""")

code("""s0.set_forward(s0.src_pos_all[3, :], true_model, withgrad=False)
s0.execute()
dobs_full = [np.asarray(d, dtype=np.float64) for d in s0.read_data()]  # [vx, vz]
recids_full = s0.rec_pos_all[3, :]
print("observed data (all sources):", [d.shape for d in dobs_full])


def dobs_subset(jobids):
    \"\"\"vx/vz traces for a subset of sources, in the order SeisCL returns them.\"\"\"
    mask = np.isin(recids_full, jobids)
    return [d[:, mask] for d in dobs_full]""")

md(r"""Comparing the observed data against data modeled in the starting model,
for a single source, shows how far apart the two are before any inversion
happens. This matters beyond a sanity check: if an event in the residual is
shifted by more than half its dominant period relative to the matching
observed event, gradient-based optimization can lock onto the wrong cycle
instead of closing the true time shift (cycle skipping) -- something to
watch for whenever the source frequency is high, or the starting model far
from the true velocity, relative to the survey's offsets.""")

code("""s0.set_forward(s0.src_pos_all[3, :], start_model, withgrad=False)
s0.execute()
dmod_start = np.asarray(s0.read_data()[0], dtype=np.float64)  # vx only, for this plot

src0 = recids_full == 0
d_obs0, d_mod0 = dobs_full[0][:, src0], dmod_start[:, src0]
residual0 = d_obs0 - d_mod0

t = np.arange(NT) * DT
vmax = np.abs(d_obs0).max()
fig, axes = plt.subplots(1, 3, figsize=(11, 5), sharey=True)
for ax, d, title in zip(axes, [d_obs0, d_mod0, residual0],
                        ["observed (true model)", "modeled (starting model)",
                         "residual"]):
    ax.imshow(d, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax,
             extent=[0, d.shape[1], t[-1], 0])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("receiver")
axes[0].set_ylabel("time (s)")
fig.suptitle("Source 0: observed vs. starting-model data")
plt.tight_layout()
plt.show()""")

md(r"""## 3. The inversion driver

Each iteration draws a random subset of `BATCH` sources and evaluates the
misfit and gradient only for that subset -- both the "before" and "after"
gradient evaluations `slbfgs` needs for one L-BFGS update use the *same*
subset, which is what makes the resulting curvature pairs meaningful (see
the method description above).

The back-propagation gradient (`BACK_PROP_TYPE=1`) runs through
`SeisCL.torch`, the in-memory PyTorch binding, rather than the `SeisCL`
class's usual subprocess/HDF5 file interface -- the whole forward-and-gradient
call stays in one process on the GPU, with no per-call file writes or process
launch, which matters here because the optimizer makes many small calls (one
pair per iteration, per source subset). Section 5 evaluates the DFT gradient
(`BACK_PROP_TYPE=2`) through the `SeisCL` class's file-based engine instead --
at this problem's size it comes back zero regardless of which interface
computes it, so the choice of interface isn't the point there; see section 5.

The gradient is only correct up to an unknown constant (SeisCL scales the
adjoint source internally), so it is normalized once, at the very first
evaluation, and that factor is reused for the rest of the run -- L-BFGS's
curvature pairs would be meaningless if every gradient carried a different
scale.

The gradient is very large right next to a source or receiver, which would
otherwise dominate the search direction rather than the structure the
survey actually illuminates. How wide that exclusion needs to be depends on
how densely the sources and receivers are spaced: with a single source, one
cell was already enough; with 10 sources spaced 13-14 cells apart down the
same well, excluding only the source cell and its direct neighbours still
leaves close to half of the gradient's total energy sitting near the wells
rather than on the model, and an optimizer chasing that produces a *worse*
model even as the misfit goes down. A `MUTE_RADIUS` of 4 cells brings that
down to under a third -- still not zero, but enough for the recovered model
to track the true anomaly rather than the wells.

One layout detail matters for correctness and is easy to get silently
wrong: `SeisCL.torch`'s flat parameter and gradient tensors are ordered
X-slowest, Z-fastest (flat index `x*NZ+z`), not the more familiar
Z-slowest layout used elsewhere in this notebook and in `SeisCL.py`'s HDF5
files. Converting one way is `array.T.reshape(-1)`, and back is
`flat.reshape(NX, NZ).T`.""")

code("""def _make_config(gradfreqs=None):
    cfg = Config()
    cfg.N = [NZ, NX]
    cfg.ND = 2
    cfg.dh = DH
    cfg.dt = DT
    cfg.NT = NT
    cfg.FDORDER = 8
    cfg.FREESURF = 0
    cfg.NAB = NAB
    cfg.ABS_TYPE = 2
    cfg.par_type = 0
    cfg.f0 = F0
    cfg.BACK_PROP_TYPE = 1
    return cfg


class Inversion:
    def __init__(self, wd, method, gradfreqs=None, batch=4, seed=0):
        self.wd = wd
        self.method = method
        self.gradfreqs = gradfreqs
        self.shape = start_model["vp"].shape
        self.scale = VP0
        self.nfev = 0
        self.gnorm = None
        self.mute = self._mute_mask()
        self.batch = batch
        self.rng = np.random.default_rng(seed)

        # Constant (non-inverted) parameters and the source wavelet, built
        # once: SeisCL.torch takes explicit tensors per call rather than a
        # pre-registered acquisition, so per-iteration source subsampling
        # means slicing these ourselves (see _geometry()).
        self.mu0 = torch.from_numpy(
            start_model["vs"].T.reshape(-1).astype(np.float32).copy())
        self.rho0 = torch.from_numpy(
            start_model["rho"].T.reshape(-1).astype(np.float32).copy())
        self.wavelet = torch.from_numpy(s0.ricker_wavelet().astype(np.float32))

    def source_batches(self):
        while True:
            yield self.rng.choice(N_SOURCES, size=self.batch, replace=False)

    @staticmethod
    def _mute_mask():
        \"\"\"Exclude a disc of radius MUTE_RADIUS around every source/receiver.\"\"\"
        zz, xx = np.indices((NZ, NX))
        keep = np.ones(zz.shape, dtype=bool)
        pts = np.concatenate([s0.src_pos_all[[2, 0], :],
                              s0.rec_pos_all[[2, 0], :]], axis=1) / DH
        for zp, xp in pts.T:
            keep &= ((zz - zp) ** 2 + (xx - xp) ** 2) > MUTE_RADIUS ** 2
        return keep

    def _geometry(self, jobids):
        \"\"\"Torch src/src_pos/rec_pos tensors for a subset of sources.

        SeisCL.torch always models exactly the sources/receivers it is
        given -- there is no separate notion of "a subset of a registered
        acquisition" the way SeisCL.py's set_forward(jobids, ...) has, so
        source ids are renumbered to 0..len(jobids)-1 for this sub-problem,
        and src_pos_all/rec_pos_all (SeisCL.py's (field, n) layout) are
        transposed to the (n, field) layout SeisCL.torch expects.
        \"\"\"
        jobids = list(jobids)
        # Sorted native order, not jobids' (random) order -- the engine
        # requires src_pos's shot-id column ascending.
        remap = {j: k for k, j in enumerate(sorted(jobids))}

        src_mask = np.isin(s0.src_pos_all[3, :], jobids)
        src_pos = s0.src_pos_all[:, src_mask].copy()
        src_pos[3, :] = [remap[j] for j in src_pos[3, :]]

        rec_mask = np.isin(s0.rec_pos_all[3, :], jobids)
        rec_pos = s0.rec_pos_all[:, rec_mask].copy()
        rec_pos[3, :] = [remap[j] for j in rec_pos[3, :]]

        src_pos_t = torch.from_numpy(src_pos.T.astype(np.float32).copy())
        rec_pos_t = torch.from_numpy(rec_pos.T.astype(np.float32).copy())
        src_t = self.wavelet.unsqueeze(0).repeat(len(jobids), 1)
        return src_t, src_pos_t, rec_pos_t

    def params(self, x):
        p = {k: np.array(v) for k, v in start_model.items()}
        p["vp"] = np.clip(x[0].reshape(self.shape) * self.scale,
                          VP_MIN, VP_MAX)
        return p

    def x0(self):
        return [start_model["vp"].ravel() / self.scale]

    def fun(self, jobids, x, withgrad=True):
        if self.method == "dft":
            return self._fun_dft(jobids, x, withgrad)
        return self._fun_backprop(jobids, x, withgrad)

    def _fun_backprop(self, jobids, x, withgrad=True):
        jobids = [int(j) for j in jobids]
        params = self.params(x)
        dobs = dobs_subset(jobids)
        self.nfev += 1

        src_t, src_pos_t, rec_pos_t = self._geometry(jobids)
        dobs_t = [torch.from_numpy(d.astype(np.float32)) for d in dobs]
        # X-slowest/Z-fastest flat layout the engine expects (see the note
        # above) -- array.T.reshape(-1).
        M0 = params["vp"].T.reshape(-1).astype(np.float32).copy()
        M = torch.from_numpy(M0)
        M.requires_grad_(withgrad)

        cfg = _make_config()
        try:
            data = seiscl_forward(cfg, {"M": M, "mu": self.mu0, "rho": self.rho0},
                                  src_t, src_pos_t, rec_pos_t,
                                  output_fields=["vx", "vz"])
        except RuntimeError:
            return np.inf if not withgrad else (np.inf,
                                                [np.zeros(x[0].size)], None)

        # SeisCL.torch returns seismograms as (allng, NT) (receiver-major);
        # SeisCL.py's read_data() (used for dobs) is (NT, nrec) -- transpose.
        loss = 0.5 * sum(((data[f].T - d) ** 2).sum()
                         for f, d in zip(["vx", "vz"], dobs_t))
        J = float(loss.detach())
        if not withgrad:
            return J

        loss.backward()
        # Undo the X-slowest/Z-fastest flat layout: reshape(NX, NZ).T.
        grad = M.grad.detach().cpu().numpy().reshape(NX, NZ).T * self.scale
        grad = grad * self.mute
        if self.gnorm is None:
            gmax = np.abs(grad).max()
            self.gnorm = (J / gmax) if gmax > 0 else 1.0
        grad = grad * self.gnorm
        return J, [grad.ravel()], None

    def _fun_dft(self, jobids, x, withgrad=True):
        \"\"\"DFT gradient, through SeisCL's file-based engine.

        set_forward(jobids, ...) subsets a pre-registered acquisition
        directly, unlike SeisCL.torch's per-call explicit geometry -- so
        this needs none of _geometry()'s bookkeeping.
        \"\"\"
        jobids = [int(j) for j in jobids]
        params = self.params(x)
        dobs = dobs_subset(jobids)
        self.nfev += 1

        if not withgrad:
            s = make_seiscl(self.wd, seisout=1)
            s.set_forward(jobids, params, withgrad=False)
            s.execute()
            return s.misfit(s.read_data(), dobs=dobs)[0]

        s = make_seiscl(self.wd, seisout=1)
        s.set_forward(jobids, params, withgrad=False)
        s.execute()
        J, res = s.misfit(s.read_data(), dobs=dobs)

        s.write_data(dict(zip(["vx", "vz"], dobs)), filename="SeisCL_din_batch.mat")
        din = os.path.join(s.workdir, "SeisCL_din_batch.mat")

        try:
            g = make_seiscl(self.wd, gradout=1, back_prop_type=2, inputres=1,
                            gradfreqs=self.gradfreqs, seisout=1)
            g.file_din = din
            g.set_forward(jobids, params, withgrad=True)
            g.set_backward(residuals=res)
            g.execute()
            grad = np.asarray(g.read_grad()[0], dtype=np.float64)
        except SeisCLError:
            return np.inf, [np.zeros(x[0].size)], None

        grad = grad * self.scale * self.mute
        if self.gnorm is None:
            gmax = np.abs(grad).max()
            self.gnorm = (J / gmax) if gmax > 0 else 1.0
        grad = grad * self.gnorm
        return J, [grad.ravel()], None""")

md(r"""## 4. Running S-LBFGS with back-propagation (`back_prop_type=1`)""")

code("""BATCH = 8
NITER = 20

inv_bp = Inversion(workdir, method="backprop", batch=BATCH, seed=0)
x0 = inv_bp.x0()
J0_bp = inv_bp.fun(list(range(N_SOURCES)), x0, withgrad=False)

out_bp = slbfgs(inv_bp.fun, x0, NITER, batch=inv_bp.source_batches(),
               alpha=0.02, verbose=False)
xk_bp, J1_bp = out_bp[0], out_bp[5]

vp_bp = np.clip(xk_bp[0].reshape(inv_bp.shape) * inv_bp.scale, VP_MIN, VP_MAX)
e0 = np.linalg.norm(start_model["vp"] - true_model["vp"])
e1_bp = np.linalg.norm(vp_bp - true_model["vp"])
print("back-propagation: misfit %.4e -> %.4e (%.1f%% reduction, %d evaluations)"
     % (J0_bp, J1_bp, 100 * (1 - J1_bp / J0_bp), inv_bp.nfev))
print("                  model error %.4e -> %.4e (%.1f%% reduction)"
     % (e0, e1_bp, 100 * (1 - e1_bp / e0)))""")

md(r"""## 5. The DFT gradient (`back_prop_type=2`) at this problem size

At this grid size (`NT=1200`), the DFT gradient currently comes back exactly
zero -- a real engine limitation at large `NT`, not a mistake in this
notebook's setup. `ComputingGradient.ipynb` demonstrates the DFT gradient
working correctly at a smaller problem size. The cell below evaluates it
once to show the zero directly, rather than running S-LBFGS against a
gradient that carries no information.""")

code("""GRADFREQS = [6.0, 10.0, 14.0]

inv_dft = Inversion(workdir, method="dft", gradfreqs=GRADFREQS, batch=BATCH, seed=0)
x0 = inv_dft.x0()
J0_dft, (g0_dft,), _ = inv_dft.fun(list(range(N_SOURCES)), x0, withgrad=True)
print("DFT gradient at the starting model: J=%.4e, max|grad|=%.4e"
     % (J0_dft, np.abs(g0_dft).max()))""")

md(r"""## 6. The recovered model""")

code("""fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
vmin, vmax = VP0 - 50, VP0 + DVP + 50
panels = [(true_model["vp"], "true"), (start_model["vp"], "starting"),
         (vp_bp, "recovered (back-prop)")]
for ax, (m, title) in zip(axes, panels):
    im = ax.imshow(m, extent=[0, NX * DH, NZ * DH, 0], cmap="viridis",
                   vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
axes[0].set_ylabel("z (m)")
fig.colorbar(im, ax=axes, label="vp (m/s)", shrink=0.8)
plt.show()

Jhist_bp = out_bp[5] if hasattr(out_bp[5], "__len__") else [J0_bp, J1_bp]
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.semilogy(Jhist_bp, marker="o", label="back-propagation")
ax.set_xlabel("iteration")
ax.set_ylabel("misfit J")
ax.set_title("Misfit history")
ax.legend()
plt.show()""")

md(r"""## Where to go next

This experiment uses a subset of sources per iteration -- enough to
demonstrate the optimizer, not a fully converged inversion. Real use would
add more iterations and, once the DFT gradient works at this problem size,
frequency continuation (start low, add higher frequencies once the
low-frequency structure is recovered).

For a viscoelastic version of the same idea -- recovering an attenuation
anomaly instead of a velocity one -- see the viscoelastic crosswell
inversion notebook.""")

nb['cells'] = c
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                          "name": "python3"}
out = "docs/notebooks/Inversion/StochasticLBFGS.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "-", len(c), "cells")
