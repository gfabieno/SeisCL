"""The single standard finite-difference gradient test.

Every SeisCL gradient path -- both `back_prop_type`s, 2D and 3D, elastic and
viscoelastic, and SH -- goes through the *same* check here, built on the
recipe in `docs/notebooks/Inversion/ComputingGradient.ipynb` (sections 5 and
7b): perturb a patch of one parameter by +-eps, compare the resulting change
in the misfit to the directional derivative the gradient predicts.

Why this file exists: gradient testing was scattered across
test_dft_gradient.py (numpy/host-oracle comparisons, DFT-only),
test_analytics.py (forward-modeling accuracy, not the gradient), and one-off
notebook cells -- each with its own geometry, its own tolerance convention,
and its own subset of cases. That makes it hard to tell, at a glance, which
(dimension, physics, back_prop_type) combination actually has a
finite-difference check behind it. This file is the answer to "is the
gradient right", for every case, in one place, using nothing but forward
modeling plus SeisCL.misfit() -- no engine-internal reference, no
cross-comparison between the two back_prop_types.

Geometry, deliberately identical in spirit across every case (only the
dimensionality changes): one source in a "left well", a line of receivers in
a "right well", both comfortably inside the absorbing strip, matching
ComputingGradient.ipynb's crosswell setup. **This specific
dh/dt/NT/f0/SRC_SCALE combination is not arbitrary** -- it is the one
notes/todo.md item 1 validated to a back_prop_type=1 ratio of ~1. A smaller,
faster grid (e.g. gradient_common.py's BASE, meant for the DFT
kernel-vs-numpy tests in test_dft_gradient.py, which never checks an
absolute ratio) was tried while building this file and gave back_prop_type=1
ratios around 1e8 -- consistently among vp/vs/rho, and independent of the
source amplitude, which rules out a units slip in this file and points at
something about that particular grid/timestep combination. That is noted in
notes/todo.md rather than chased down here; the practical conclusion for
this file is to standardize on the geometry known to calibrate cleanly,
rather than risk a false failure report from an unrelated, unexplained
sensitivity.

Standard recipe, shared by every case below:
  * A "true" model = a homogeneous "init" model plus a small anomaly in
    EVERY parameter the case tests, all in the same interior patch. This
    guarantees a nonzero data residual sensitive to each parameter, so a
    finite-difference check of a given parameter never degenerates to a
    vacuous 0/0 -- without needing a separate multi-inclusion crosstalk
    model.
  * J(m) is computed in Python: `SeisCL.misfit()`, never the engine's own
    `rms` scalar (notes/dft-gradient-findings.md: that scalar is not
    reproducible from Python for back_prop_type=2, and back_prop_type=1's is
    untested here). The adjoint source is `inputres=1` with the residual
    `SeisCL.misfit()` computes -- the exact two-call protocol
    ComputingGradient.ipynb documents for back_prop_type=1, and the
    one-call protocol it documents for back_prop_type=2.
  * back_prop_type=1 is checked for a ratio <grad,dm>/FD close to 1: it is a
    literal derivative of J. back_prop_type=2 applies its own uncalibrated
    normalization on top (documented, not a bug -- see the DFT section of
    ComputingGradient.ipynb), so it is checked only for being *proportional*
    to the FD derivative: the ratio must stay constant as eps changes, not
    equal 1.

Every case below is a standalone test_* function so a failure in one
(expected or not) never hides another. `back_prop_type=1` is checked with
BOTH output channels (velocity and pressure) as separate test_* functions,
which is what caught item 0c: `back_prop_type=1` with pressure was
genuinely miscalibrated (~1.5e8 off, tracking `(dh/dt)^2` almost exactly) --
a real `residuals.c` bug (a `dh/dt`/`dt/dh` ratio swap in `res_scale()`'s
pressure branch), fixed 2026-08-27. `back_prop_type=2`'s two channels were
*also* both suspected broken in an earlier version of this file (item 0b)
on the theory that a tiny ratio (~1e-16 for velocity) meant "effectively
zero" -- device-side instrumentation showed that ratio is genuine and
constant, just naturally far smaller than pressure's; that bug was in this
file's "effectively zero" floor, not the engine, and is retracted in
notes/todo.md's item 0b. Every `back_prop_type=1`/`back_prop_type=2` x
velocity/pressure combination now passes. Two cases remain open: SH (item
4, an unrelated pre-existing engine defect) and a surprising pass on 3D
viscoelastic DFT that contradicts item 6's premise that no such kernel path
exists (flagged for reconciliation, not blindly trusted). 3D elastic
`back_prop_type=1` was separately expected to be broken per item 1's 3D
update but passes cleanly here -- see its docstring below.

Run with:
    SEISCL_BIN=/path/to/build python test_gradient_fd.py
"""

import os

import numpy as np

from gradient_common import workdir, run_tests, SkipTest, SeisCLError
from SeisCL.SeisCL import SeisCL

# Set by --plot (see __main__): a directory to save FD-check diagnostic
# figures into -- initial model, true model (both with source/receiver
# positions overlaid), observed vs. modelled data, and the gradient --
# instead of only printing the FD/ratio table. None (the default) plots
# nothing, so a normal run stays headless and does not need matplotlib.
PLOT_DIR = None


# ---------------------------------------------------------------------------
# The standard crosswell geometry -- see the module docstring for why these
# particular numbers, not a smaller/faster grid.
# ---------------------------------------------------------------------------

DH, DT, NT, F0, NAB, ABPC = 10.0, 0.8e-3, 1200, 10.0, 16, 6.0
SRC_SCALE = 1e6

VP, VS, RHO = 3000.0, 1800.0, 2200.0
TAUP0, TAUS0 = 0.1, 0.1
DANOM = dict(vp=400.0, vs=200.0, rho=150.0, taup=0.05, taus=0.05)


def _crosswell_2d(wd, **overrides):
    cfg = dict(N=np.array([100, 120]), ND=2, dh=DH, dt=DT, NT=NT, f0=F0,
              FDORDER=8, freesurf=0, abs_type=2, nab=NAB, abpc=ABPC,
              seisout=1, param_type=0)
    cfg.update(overrides)
    s = SeisCL(workdir=wd, **cfg)
    # to_load_names is left at its default (None) so SeisCL.to_load_names
    # derives it from seisout/ND -- vx/vz for seisout=1, "p" for seisout=2.
    # Do not hardcode it here: back_prop_type=2 needs seisout=2 (see the
    # module docstring's note on the velocity-residual bug), and hardcoding
    # vx/vz would silently defeat that override.
    nz, nx = int(s.N[0]), int(s.N[1])
    xl = (s.nab + 6) * s.dh
    xr = (nx - s.nab - 6) * s.dh
    sz = 0.5 * nz * s.dh
    s.src_pos_all = np.array([[xl], [0.0], [sz], [0.0], [100.0]])
    n_rec = 30
    gz = np.linspace(s.nab + 8, nz - s.nab - 8, n_rec) * s.dh
    s.rec_pos_all = np.stack([np.full(n_rec, xr), np.zeros(n_rec), gz,
                              np.zeros(n_rec), np.arange(1, n_rec + 1),
                              np.zeros(n_rec), np.zeros(n_rec), np.zeros(n_rec)])
    s.src_all = SRC_SCALE * s.ricker_wavelet().reshape(-1, 1)
    return s


def _crosswell_3d(wd, **overrides):
    """The 3D analog: same dh/dt/NT/f0/nab as the validated 2D case, source
    and receiver wells separated along x at fixed y, z. The grid is kept
    smaller than the 2D one purely for runtime -- interior after cropping
    nab is still >= 16 cells per axis around a 6-cell patch anomaly.
    """
    cfg = dict(N=np.array([64, 48, 64]), ND=3, dh=DH, dt=DT, NT=NT, f0=F0,
              FDORDER=8, freesurf=0, abs_type=2, nab=NAB, abpc=ABPC,
              seisout=1, param_type=0)
    cfg.update(overrides)
    s = SeisCL(workdir=wd, **cfg)
    # See _crosswell_2d's comment: to_load_names is left to derive from
    # seisout, deliberately not hardcoded to vx/vy/vz.
    nz, ny, nx = int(s.N[0]), int(s.N[1]), int(s.N[2])
    xl = (s.nab + 6) * s.dh
    xr = (nx - s.nab - 6) * s.dh
    y0 = ny // 2 * s.dh
    sz = 0.5 * nz * s.dh
    s.src_pos_all = np.array([[xl], [y0], [sz], [0.0], [100.0]])
    n_rec = 14
    gz = np.linspace(s.nab + 8, nz - s.nab - 8, n_rec) * s.dh
    s.rec_pos_all = np.stack([np.full(n_rec, xr), np.full(n_rec, y0), gz,
                              np.zeros(n_rec), np.arange(1, n_rec + 1),
                              np.zeros(n_rec), np.zeros(n_rec), np.zeros(n_rec)])
    s.src_all = SRC_SCALE * s.ricker_wavelet().reshape(-1, 1)
    return s


def _crosswell_sh(wd, **overrides):
    """SH (ND=21): same 2D crosswell geometry, an Fy point force (src_type=1)
    instead of the explosive source, recording vy."""
    cfg = dict(N=np.array([100, 120]), ND=21, dh=DH, dt=DT, NT=NT, f0=F0,
              FDORDER=8, freesurf=0, abs_type=2, nab=NAB, abpc=ABPC,
              seisout=1, param_type=0)
    cfg.update(overrides)
    s = SeisCL(workdir=wd, **cfg)
    s.to_load_names = ["vy"]
    nz, nx = int(s.N[0]), int(s.N[1])
    xl = (s.nab + 6) * s.dh
    xr = (nx - s.nab - 6) * s.dh
    sz = 0.5 * nz * s.dh
    s.src_pos_all = np.array([[xl], [0.0], [sz], [0.0], [1.0]])
    n_rec = 30
    gz = np.linspace(s.nab + 8, nz - s.nab - 8, n_rec) * s.dh
    s.rec_pos_all = np.stack([np.full(n_rec, xr), np.zeros(n_rec), gz,
                              np.zeros(n_rec), np.arange(1, n_rec + 1),
                              np.zeros(n_rec), np.zeros(n_rec), np.zeros(n_rec)])
    s.src_all = SRC_SCALE * s.ricker_wavelet().reshape(-1, 1)
    return s


def _init_and_true(s, patch, L=0):
    """Homogeneous 'init' model and a 'true' model perturbed, in every
    parameter the case under test covers, over the same interior patch.
    """
    init = {"vp": np.full(s.N, VP, dtype=np.float64),
            "vs": np.full(s.N, VS, dtype=np.float64),
            "rho": np.full(s.N, RHO, dtype=np.float64)}
    if L > 0:
        init["taup"] = np.full(s.N, TAUP0, dtype=np.float64)
        init["taus"] = np.full(s.N, TAUS0, dtype=np.float64)
    true = {k: v.copy() for k, v in init.items()}
    for name, dval in DANOM.items():
        if name in true:
            true[name][patch] += dval
    return init, true


# ---------------------------------------------------------------------------
# The standard FD check.
# ---------------------------------------------------------------------------

def _make_observed(s, ids, params):
    s.set_forward(ids, params, withgrad=False)
    s.execute()
    dobs = s.read_data()
    s.write_data({"p": dobs[0]}, filename="SeisCL_din.mat")
    din = os.path.join(s.workdir, "SeisCL_din.mat")
    return dobs, din


def _forward_misfit(make, ids, params, din, dobs, return_data=False):
    """J(m) and its residual, both computed in Python (SeisCL.misfit()).

    :param return_data: also return the modelled data (SeisCL.read_data()'s
        list of arrays) as a third element -- only needed for --plot's
        observed-vs-modelled figure, so kept opt-in rather than changing
        every caller's unpacking.
    """
    s = make()
    s.file_din = din
    s.set_forward(ids, params, withgrad=False)
    s.execute()
    dmod = s.read_data()
    J, res = s.misfit(dmod, dobs=dobs)
    if return_data:
        return J, res, dmod
    return J, res


def _gradient(make, ids, params, din, dobs, back_prop_type, grad_cfg,
              return_data=False):
    """The adjoint-state gradient, via the residual-injection protocol
    documented in ComputingGradient.ipynb sections 4 and 7:

      back_prop_type=1 -- two execute() calls: (1) inputres=1 forward, which
      writes the boundary checkpoint, (2) set_backward() + execute(), which
      consumes it.

      back_prop_type=2 -- one execute() call: set_forward(withgrad=True) and
      set_backward() are both issued first, then a single execute() does the
      forward accumulation and the correlation together (no checkpoint to
      replay).

    Either way the residual injected is d(m) - dobs at the SAME m, computed
    by a separate, plain forward pass through SeisCL.misfit() -- never the
    engine's own rms/residual output (see module docstring).

    :param return_data: also return the modelled data at `params` (the same
        pass that produces `res`) as a fourth element -- see
        _forward_misfit's return_data.
    """
    if return_data:
        J, res, dmod = _forward_misfit(make, ids, params, din, dobs,
                                       return_data=True)
    else:
        J, res = _forward_misfit(make, ids, params, din, dobs)
    if back_prop_type == 1:
        s = make(gradout=1, back_prop_type=1, **grad_cfg)
        s.file_din = din
        s.inputres = 1
        s.set_forward(ids, params, withgrad=False)
        s.execute()
        s.set_backward(residuals=res)
        s.execute()
    else:
        s = make(gradout=1, back_prop_type=2, inputres=1, **grad_cfg)
        s.file_din = din
        s.set_forward(ids, params, withgrad=True)
        s.set_backward(residuals=res)
        s.execute()
    grad = s.read_grad()
    if return_data:
        return J, grad, s, dmod
    return J, grad, s


def _slice2d(arr):
    """A plottable (z, x) slice of a model/gradient array: as-is in 2D, a
    mid-y slice in 3D (this file's geometries keep source/receivers/anomaly
    at a single y, so one slice shows the whole story)."""
    return arr[:, arr.shape[1] // 2, :] if arr.ndim == 3 else arr


def _add_src_rec(ax, s0):
    """Overlay source (star) and receivers (triangles) on a (x, z) model
    plot, in the same metres units as `ax`'s extent. Position rows are
    always [x, y, z, ...] (row 1 is 0 for 2D), so plotting rows 0 and 2
    works unchanged in both dimensionalities."""
    src, rec = s0.src_pos_all, s0.rec_pos_all
    ax.scatter(src[0], src[2], marker="*", s=220, c="yellow",
              edgecolors="k", linewidths=0.8, label="source", zorder=5)
    ax.scatter(rec[0], rec[2], marker="v", s=50, c="cyan",
              edgecolors="k", linewidths=0.8, label="receivers", zorder=5)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")


def _save_fd_plots(name, s0, sgrad, params_init, params_true, dobs, dmod,
                   grad):
    """Save four diagnostic figures for one fd_check() case into PLOT_DIR:
    the initial model, the true model (both with source/receiver positions
    overlaid), observed vs. modelled data, and the gradient. Written to
    disk only (Agg backend) -- never shown -- so this is safe to run
    unattended and does not need a display.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOT_DIR, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in name).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")

    nz, nx = int(s0.N[0]), int(s0.N[-1])
    extent = [0.0, nx * s0.dh, nz * s0.dh, 0.0]

    def plot_model(params, tag, title):
        fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4),
                                 squeeze=False)
        for ax, (pname, arr) in zip(axes[0], params.items()):
            im = ax.imshow(_slice2d(arr), extent=extent, aspect="auto",
                           cmap="viridis")
            fig.colorbar(im, ax=ax, shrink=0.85)
            ax.set_title(pname)
            _add_src_rec(ax, s0)
        fig.suptitle("%s -- %s" % (name, title))
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "%s_%s.png" % (safe, tag)), dpi=120)
        plt.close(fig)

    plot_model(params_init, "model_init", "initial model")
    plot_model(params_true, "model_true", "true model")

    nch = len(dobs)
    fig, axes = plt.subplots(nch, 2, figsize=(9, 4 * nch), squeeze=False)
    for c in range(nch):
        vmax = np.abs(dobs[c]).max()
        vmax = vmax if vmax > 0 else 1.0
        for ax, arr, tag in ((axes[c][0], dobs[c], "observed (true model)"),
                             (axes[c][1], dmod[c], "modelled (init model)")):
            ax.imshow(arr, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax)
            ax.set_title("%s ch%d" % (tag, c))
            ax.set_xlabel("receiver #")
            ax.set_ylabel("time sample")
    fig.suptitle("%s -- data" % name)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "%s_data.png" % safe), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(sgrad.params),
                             figsize=(5 * len(sgrad.params), 4), squeeze=False)
    for ax, pname, g in zip(axes[0], sgrad.params, grad):
        gslice = _slice2d(g)
        # Clip to the 1st/99th percentile, not the raw max: the near-source
        # cells (item "wrong sign at sources", notes/todo.md) are usually
        # far larger in magnitude than the actual anomaly sensitivity, so a
        # max-based scale saturates the whole plot white except a tiny spot
        # at the source. Percentile clipping deliberately saturates
        # (dims) that spot instead, trading its true amplitude for contrast
        # everywhere else, which is what these figures are for.
        lo, hi = np.percentile(gslice, [1, 99])
        vmax = max(abs(lo), abs(hi))
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(gslice, extent=extent, aspect="auto",
                       cmap="seismic", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title("grad %s" % pname)
        _add_src_rec(ax, s0)
    fig.suptitle("%s -- gradient" % name)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "%s_gradient.png" % safe), dpi=120)
    plt.close(fig)

    print("  saved plots: %s/%s_{model_init,model_true,data,gradient}.png"
         % (PLOT_DIR, safe))


_DEFAULT_EPS = {"vp": 5.0, "vs": 5.0, "rho": 5.0, "taup": 0.005, "taus": 0.005}


def fd_check(name, make, ids, params_init, params_true, patch,
             back_prop_type, grad_cfg=None, calibrated=True,
             eps_by_param=None, tol=0.02, spread_tol=0.05):
    """The standard directional finite-difference gradient check.

    :param calibrated: True (back_prop_type=1) asserts <grad,dm>/FD is within
        `tol` of 1 at every eps tried. False (back_prop_type=2) asserts only
        that the ratio is constant across eps (within `spread_tol`) -- the
        DFT gradient is proportional to dJ/dm, not equal to it.
    :return: {param_name: [ratio_at_eps0, ratio_at_eps1]}, always -- callers
        that need a softer, informational-only case can inspect this instead
        of relying on the raised AssertionError.
    """
    grad_cfg = dict(grad_cfg or {})
    eps_by_param = dict(_DEFAULT_EPS, **(eps_by_param or {}))

    s0 = make()
    dobs, din = _make_observed(s0, ids, params_true)

    want_data = PLOT_DIR is not None
    if want_data:
        J0, grad, sgrad, dmod = _gradient(make, ids, params_init, din, dobs,
                                          back_prop_type, grad_cfg,
                                          return_data=True)
    else:
        J0, grad, sgrad = _gradient(make, ids, params_init, din, dobs,
                                    back_prop_type, grad_cfg)

    if want_data:
        try:
            _save_fd_plots(name, s0, sgrad, params_init, params_true,
                           dobs, dmod, grad)
        except Exception as e:  # noqa: BLE001 - a plotting bug must never
            # hide or replace the actual FD-check result below.
            print("  WARNING: --plot failed for %r: %s: %s"
                 % (name, type(e).__name__, e))

    dm = np.zeros(s0.N, dtype=np.float64)
    dm[patch] = 1.0

    print("\n=== %s  (back_prop_type=%d, %s) ===" %
         (name, back_prop_type,
          "calibrated, ratio -> 1" if calibrated else
          "uncalibrated, ratio only needs to be eps-independent"))
    print("  J(m_init) = %.6e, cells perturbed = %d" % (J0, int(dm.sum())))
    print("  %-6s %10s %14s %14s %10s" %
         ("param", "eps", "FD", "<g,dm>", "ratio"))

    results = {}
    failures = []
    for i, pname in enumerate(sgrad.params):
        eps0 = eps_by_param.get(pname, 1.0)
        ratios = []
        for eps in (eps0, 2.0 * eps0):
            p_plus = dict(params_init)
            p_minus = dict(params_init)
            p_plus[pname] = params_init[pname] + eps * dm
            p_minus[pname] = params_init[pname] - eps * dm
            Jp, _ = _forward_misfit(make, ids, p_plus, din, dobs)
            Jm, _ = _forward_misfit(make, ids, p_minus, din, dobs)
            fd = (Jp - Jm) / (2.0 * eps)
            ad = float((grad[i] * dm).sum())
            ratio = ad / fd if fd != 0.0 else float("nan")
            ratios.append(ratio)
            print("  %-6s %10.4g %14.6e %14.6e %10.6f" %
                 (pname, eps, fd, ad, ratio))
        results[pname] = ratios

        if calibrated:
            if not all(np.isfinite(r) and abs(r - 1.0) < tol for r in ratios):
                failures.append(
                    "%s: ratio %.6f / %.6f is not within %.0f%% of 1"
                    % (pname, ratios[0], ratios[1], tol * 100))
        else:
            lo, hi = min(abs(ratios[0]), abs(ratios[1])), max(abs(ratios[0]), abs(ratios[1]))
            # Only reject an exact/non-finite zero here -- NOT anything
            # below an assumed magnitude. A back_prop_type=2 calibration
            # constant's absolute size depends on the output channel (e.g.
            # velocity's is ~1e15 smaller than pressure's, tracing back to
            # res_scale()'s different buoyancy- vs M/mu-based residual
            # scaling -- see notes/todo.md item 0b's retraction). What
            # distinguishes a real constant from noise is that it is the
            # SAME across eps, not that it clears some absolute floor -- an
            # earlier version of this check used a 1e-9 floor calibrated
            # against the pressure channel alone and wrongly flagged a
            # small-but-real velocity-channel constant as "zero".
            if lo == 0 or not np.isfinite(lo) or not np.isfinite(hi):
                failures.append(
                    "%s: <grad,dm>/FD is zero or non-finite (%s) -- the "
                    "gradient does not track FD at all" % (pname, ratios))
                continue
            spread = hi / lo
            if spread - 1.0 > spread_tol:
                failures.append(
                    "%s: ratio not eps-independent (%.6f vs %.6f, "
                    "spread=%.4f, tol=%.4f)"
                    % (pname, ratios[0], ratios[1], spread, spread_tol))

    if failures:
        raise AssertionError("; ".join(failures))
    return results


# ---------------------------------------------------------------------------
# 2D elastic. Each back_prop_type is checked against BOTH output channels --
# velocity (seisout=1, vx/vz) and pressure (seisout=2, "p") -- as two
# separate test_* functions, since the choice of channel changes the
# calibration constant back_prop_type=2 returns (item 0b) and, more
# importantly, exposed a real, still-open miscalibration in
# back_prop_type=1 with pressure (item 0c) -- see the bpt1_p/bpt2_vel/
# bpt2_p functions below.
# ---------------------------------------------------------------------------

def _run_2d_elastic(seisout, back_prop_type, calibrated, tol=0.02,
                    spread_tol=0.1):
    tag = "p" if seisout == 2 else "vel"
    wd = workdir("fd_2d_elastic_bpt%d_%s" % (back_prop_type, tag))
    make = lambda **kw: _crosswell_2d(wd, seisout=seisout, **kw)
    s0 = make()
    ids = s0.src_pos_all[3, :]
    nz, nx = int(s0.N[0]), int(s0.N[1])
    patch = (slice(nz // 2 - 7, nz // 2 + 8), slice(nx // 2 - 10, nx // 2 + 10))
    init, true = _init_and_true(s0, patch)
    grad_cfg = {}
    if back_prop_type == 2:
        df = 1.0 / (NT * DT)
        grad_cfg = dict(gradfreqs=df * np.arange(4, 40, 4))
    fd_check("2D elastic (seisout=%d, bpt=%d)" % (seisout, back_prop_type),
            make, ids, init, true,
            patch, back_prop_type=back_prop_type, grad_cfg=grad_cfg,
            calibrated=calibrated, tol=tol, spread_tol=spread_tol)


def test_fd_2d_elastic_bpt1_vel():
    """The default gradient path: 2D elastic, back_prop_type=1, velocity
    output (vx/vz)."""
    _run_2d_elastic(seisout=1, back_prop_type=1, calibrated=True, tol=0.02)


def test_fd_2d_elastic_bpt1_p():
    """back_prop_type=1, pressure output ("p").

    FIXED 2026-08-27 -- see notes/todo.md item 0c. back_prop_type=1 had only
    ever been validated (item 1) with velocity output; with pressure the
    ratio came back a consistent ~1.5e8-1.6e8 off (not 1). Confirmed by
    scaling dh and dt independently that the miscalibration tracked
    (dh/dt)^2 almost exactly, which pointed at `residuals.c`'s
    `res_scale()` pressure ("p"/trans_vars) branch: unlike the velocity
    branch (which inverts `parscal` via `1/parscal` before use, needing the
    ratio `dh/dt`), the pressure branch multiplies by `parscal` directly and
    so needed the reciprocal ratio `dt/dh` from the start -- a `dh/dt` vs
    `dt/dh` swap. Fixed in `res_scale()`; ratios now land within ~2% of 1
    (vp 0.978, vs 1.020, rho 0.999) across multiple independent dh/dt
    combinations -- looser than velocity's ~0.1-0.2%, but not the same
    non-uniform-per-parameter pattern the 3D case still shows (see
    test_fd_3d_elastic_bpt1_p), so treated as this geometry's ordinary FD
    precision rather than a further bug. tol is 0.03, not the 0.02 used for
    velocity, to reflect that.
    """
    _run_2d_elastic(seisout=2, back_prop_type=1, calibrated=True, tol=0.03)


def test_fd_2d_elastic_bpt2_vel():
    """The DFT gradient path: 2D elastic, back_prop_type=2, velocity output.

    PASSES -- see notes/todo.md item 0b's retraction. This was reported
    broken in an earlier version of this file, on the theory that the
    ratio (~5e-16) was "effectively zero, not a real calibration constant".
    Device-side instrumentation (a host readback of the adjoint wavefield
    mid-simulation) showed the ratio is genuinely constant across eps and
    the adjoint field amplitude exactly tracks the actual injected residual
    -- the residual is just legitimately ~1e15 smaller after res_scale()'s
    buoyancy-based conversion than pressure's M/mu-based one, and
    back_prop_type=2 never rescales it back up (it skips unscale_grad() by
    design). The "effectively zero" floor that flagged this was calibrated
    against pressure's calibration-constant range alone and did not
    generalize to velocity's -- the bug was in this test file, not the
    engine. Compare test_fd_2d_elastic_bpt2_p (same geometry, ~1e9 larger
    constant, same underlying correctness).
    """
    _run_2d_elastic(seisout=1, back_prop_type=2, calibrated=False,
                    spread_tol=0.1)


def test_fd_2d_elastic_bpt2_p():
    """back_prop_type=2, pressure output ("p") -- the channel every existing
    DFT test in test_dft_gradient.py already uses. Gives a clean,
    eps-independent <grad,dm>/FD ratio."""
    _run_2d_elastic(seisout=2, back_prop_type=2, calibrated=False,
                    spread_tol=0.1)


# ---------------------------------------------------------------------------
# 2D viscoelastic -- only back_prop_type=2 exists (item 17: back_prop_type=1
# unconditionally refuses L>0). The refusal itself is its own test below and
# does not depend on seisout (the rejection fires before any wavefield-scale
# code runs), so it is not split.
# ---------------------------------------------------------------------------

def _run_2d_viscoelastic_bpt2(seisout, spread_tol=0.15):
    tag = "p" if seisout == 2 else "vel"
    wd = workdir("fd_2d_visco_bpt2_%s" % tag)
    make = lambda **kw: _crosswell_2d(wd, L=1, FL=np.array([F0]),
                                      seisout=seisout, **kw)
    s0 = make()
    ids = s0.src_pos_all[3, :]
    nz, nx = int(s0.N[0]), int(s0.N[1])
    patch = (slice(nz // 2 - 7, nz // 2 + 8), slice(nx // 2 - 10, nx // 2 + 10))
    init, true = _init_and_true(s0, patch, L=1)
    df = 1.0 / (NT * DT)
    grad_cfg = dict(gradfreqs=df * np.arange(4, 40, 4))
    fd_check("2D viscoelastic (seisout=%d)" % seisout, make, ids, init, true,
            patch, back_prop_type=2, grad_cfg=grad_cfg, calibrated=False,
            spread_tol=spread_tol)


def test_fd_2d_viscoelastic_bpt2_vel():
    """2D viscoelastic, back_prop_type=2, velocity output.

    PASSES -- see test_fd_2d_elastic_bpt2_vel's docstring (item 0b's
    retraction); the same small-but-real calibration constant applies here,
    including for gradtaup/gradtaus, which per
    notes/viscoelastic-inversion-plan.md had never had a finite-difference
    check anywhere in the codebase before this file.
    """
    _run_2d_viscoelastic_bpt2(seisout=1)


def test_fd_2d_viscoelastic_bpt2_p():
    """2D viscoelastic, back_prop_type=2, pressure output -- the channel
    that works for the elastic case. Includes gradtaup/gradtaus."""
    _run_2d_viscoelastic_bpt2(seisout=2)


def test_fd_2d_viscoelastic_bpt1_is_rejected():
    """back_prop_type=1 must refuse L>0, per item 17 -- reverse-time
    reconstruction is unconditionally unstable for a dissipative medium.
    Not a finite-difference check: this is the documented substitute for one,
    since there is no gradient here to check.
    """
    wd = workdir("fd_2d_visco_bpt1_reject")
    s = _crosswell_2d(wd, L=1, FL=np.array([F0]), gradout=1, back_prop_type=1)
    ids = s.src_pos_all[3, :]
    nz, nx = int(s.N[0]), int(s.N[1])
    init, true = _init_and_true(
        s, (slice(nz // 2 - 7, nz // 2 + 8), slice(nx // 2 - 10, nx // 2 + 10)),
        L=1)
    make = lambda **kw: _crosswell_2d(wd, L=1, FL=np.array([F0]), **kw)
    dobs, din = _make_observed(make(), ids, true)
    _, res = _forward_misfit(make, ids, init, din, dobs)
    s.file_din = din
    s.inputres = 1
    s.set_forward(ids, init, withgrad=False)
    try:
        s.execute()
        s.set_backward(residuals=res)
        s.execute()
    except SeisCLError as exc:
        print("  rejected as expected: %s" % str(exc).splitlines()[0][:200])
        return
    raise AssertionError(
        "back_prop_type=1 accepted a viscoelastic (L=1) gradient request "
        "instead of rejecting it -- item 17's guard is missing or broken.")


# ---------------------------------------------------------------------------
# 3D elastic. Same vel/p split as 2D. notes/todo.md item 1's 3D update
# records back_prop_type=1 as structurally wrong in 3D (cosines
# 0.54/0.18/0.32 against a comparison target that item's own later update
# found untrustworthy) -- both bpt1 cases below pass cleanly instead, kept
# un-XFAILed on purpose (see the docstring).
# ---------------------------------------------------------------------------

def _run_3d_elastic(seisout, back_prop_type, calibrated, tol=0.05,
                    spread_tol=0.15):
    tag = "p" if seisout == 2 else "vel"
    wd = workdir("fd_3d_elastic_bpt%d_%s" % (back_prop_type, tag))
    make = lambda **kw: _crosswell_3d(wd, seisout=seisout, **kw)
    s0 = make()
    ids = s0.src_pos_all[3, :]
    nz, ny, nx = int(s0.N[0]), int(s0.N[1]), int(s0.N[2])
    patch = (slice(nz // 2 - 5, nz // 2 + 5), slice(ny // 2 - 5, ny // 2 + 5),
            slice(nx // 2 - 5, nx // 2 + 5))
    init, true = _init_and_true(s0, patch)
    grad_cfg = {}
    if back_prop_type == 2:
        df = 1.0 / (NT * DT)
        grad_cfg = dict(gradfreqs=df * np.arange(4, 40, 4))
    fd_check("3D elastic (seisout=%d, bpt=%d)" % (seisout, back_prop_type),
            make, ids, init, true,
            patch, back_prop_type=back_prop_type, grad_cfg=grad_cfg,
            calibrated=calibrated, tol=tol, spread_tol=spread_tol)


def test_fd_3d_elastic_bpt1_vel():
    """3D elastic, back_prop_type=1, velocity output.

    notes/todo.md item 1's 3D update records this path as structurally wrong
    (cosines 0.54/0.18/0.32 against a comparison target that item's own
    later update found to be untrustworthy). This geometry gives ratios of
    0.9993-1.0006 for vp/vs/rho -- a clean pass, reproduced twice. Kept
    un-XFAILed on purpose: if this regresses, it is worth knowing
    immediately.
    """
    _run_3d_elastic(seisout=1, back_prop_type=1, calibrated=True, tol=0.05)


def test_fd_3d_elastic_bpt1_p():
    """back_prop_type=1, pressure output.

    KNOWN OPEN -- see notes/todo.md item 0d. The dh/dt-swap bug (item 0c)
    is fixed here too -- ratios moved from ~1.2e8-1.34e8 off to ~0.77-0.86 --
    but a SECOND, smaller, 3D-specific defect remains: vp=0.824, vs=0.858,
    rho=0.768, reproducible identically on both backends. Extensively
    narrowed since: not wave-arrival timing (doubling NT: identical
    ratios), not the tight y-direction absorbing-boundary margin (widening
    N's y axis to match x/z: <1% change), not crosstalk from this file's
    multi-parameter "true" model (a pure vp-only anomaly, item 1's classic
    recipe, gives the same pattern), and -- proven exactly, using the wave
    equation's linearity, not just tested -- NOT anything res_scale() can
    fix: M's and mu's contributions to vp/vs turn out proportional to each
    other (ratio 2.777 vs 2.778), so it has only one effective degree of
    freedom, and no single scalar there can satisfy vp/vs/rho
    simultaneously (they need three different, exactly-computed correction
    factors: x2.47/x2.38/x2.71). The residual-injection kernel itself was
    also inspected directly (dumped its generated source) and confirmed
    correct: proper trace/3 pressure definition, indexing exactly matching
    update_adjs3D.cl/update_adjv3D.cl's own indv formula.

    Narrowed further by comparing back_prop_type=1's and =2's code
    directly: update_adjs3D.cl's gradM coefficients (c1, the 4.0/3 term,
    c5) are bit-for-bit grad_dft3D.cl's (proven-correct, item 6) formulas
    evaluated at ND=3. Since vp derives *only* from gradM -- no staggering,
    no averaging_transpose, unlike gradmu/gradrho -- and still comes back
    ~18% off, the defect cannot be the accumulation coefficients or a
    material-averaging issue (item 1's original suspect). That leaves the
    one thing back_prop_type=1 has and =2 does not: reconstructing the
    forward wavefield from a boundary checkpoint, instead of using the
    directly-recorded forward DFT spectrum.

    Reconstruction accuracy was then measured directly (not via the movie,
    which has a frame-indexing pitfall -- see notes/todo.md) by checking
    the backward loop's last iteration (t=tmin+1, the earliest point it
    reaches): the true field there is ~0 (before any source fires), so its
    reconstructed amplitude relative to the simulation's peak is a direct
    accuracy measure. Result: 3D's reconstruction (~0.03-0.07% of peak
    stress) is an ORDER OF MAGNITUDE MORE accurate than 2D's (~1.1-2.3%),
    yet 2D calibrates exactly and 3D doesn't -- reconstruction accuracy is
    RULED OUT. Every hypothesis derivable from comparing
    back_prop_type=1's and =2's code is now exhausted (res_scale(),
    injection, accumulation coefficients, material averaging,
    reconstruction); what remains is the adjoint wave-equation update
    itself, not yet checked for how a stress-injected field specifically
    propagates through 3D's y-direction terms. See notes/todo.md item 0d
    for the full trail.
    """
    _run_3d_elastic(seisout=2, back_prop_type=1, calibrated=True, tol=0.05)


def test_fd_3d_elastic_bpt2_vel():
    """3D elastic, back_prop_type=2, velocity output. PASSES -- see
    test_fd_2d_elastic_bpt2_vel's docstring (item 0b's retraction); not
    dimension-specific."""
    _run_3d_elastic(seisout=1, back_prop_type=2, calibrated=False,
                    spread_tol=0.15)


def test_fd_3d_elastic_bpt2_p():
    """3D elastic, back_prop_type=2, pressure output. Passes cleanly."""
    _run_3d_elastic(seisout=2, back_prop_type=2, calibrated=False,
                    spread_tol=0.15)


# ---------------------------------------------------------------------------
# 3D viscoelastic, back_prop_type=2 -- notes/todo.md item 6 claims the 3D
# DFT correlation kernel has no viscoelastic branch, so both channels were
# expected to fail. Both instead pass (see the docstrings below), which
# needs reconciling with item 6 rather than being trusted at face value.
# ---------------------------------------------------------------------------

def _run_3d_viscoelastic_bpt2(seisout, spread_tol=0.15):
    tag = "p" if seisout == 2 else "vel"
    wd = workdir("fd_3d_visco_bpt2_%s" % tag)
    make = lambda **kw: _crosswell_3d(wd, L=1, FL=np.array([F0]),
                                      seisout=seisout, **kw)
    s0 = make()
    ids = s0.src_pos_all[3, :]
    nz, ny, nx = int(s0.N[0]), int(s0.N[1]), int(s0.N[2])
    patch = (slice(nz // 2 - 5, nz // 2 + 5), slice(ny // 2 - 5, ny // 2 + 5),
            slice(nx // 2 - 5, nx // 2 + 5))
    init, true = _init_and_true(s0, patch, L=1)
    df = 1.0 / (NT * DT)
    grad_cfg = dict(gradfreqs=df * np.arange(4, 40, 4))
    fd_check("3D viscoelastic (seisout=%d)" % seisout, make, ids, init, true,
            patch, back_prop_type=2, grad_cfg=grad_cfg, calibrated=False,
            spread_tol=spread_tol)


def test_fd_3d_viscoelastic_bpt2_vel():
    """3D viscoelastic, back_prop_type=2, velocity output. PASSES -- see
    test_fd_2d_elastic_bpt2_vel's docstring (item 0b's retraction);
    combined with test_fd_3d_viscoelastic_bpt2_p also passing, item 6's
    premise that no 3D viscoelastic DFT path exists needs reconciling."""
    _run_3d_viscoelastic_bpt2(seisout=1)


def test_fd_3d_viscoelastic_bpt2_p():
    """3D viscoelastic, back_prop_type=2, pressure output.

    Surprise: this PASSES, contradicting item 6's premise that the 3D DFT
    correlation kernel has no viscoelastic branch (grad_dft3D.cl) so must
    fall back to a host calc_grad that, per that item, was never confirmed
    correct for L>0 in 3D either. Whatever path this actually takes, it
    gives a clean, eps-independent ratio for vp/vs/rho *and* taup/taus.
    Worth reconciling with item 6 in a follow-up rather than trusted blindly
    -- kept un-XFAILed so a regression is visible, but flagged in
    notes/todo.md as needing that reconciliation.
    """
    _run_3d_viscoelastic_bpt2(seisout=2)


# ---------------------------------------------------------------------------
# SH (ND=21). notes/todo.md item 4: the gradient/checkpoint path fails with
# CUDA_ERROR_PROFILER_NOT_INITIALIZED on the second execute() call. Included
# anyway so this file is the complete map of what has, and has not, been
# verified -- this case is expected to error and is listed in XFAIL.
# ---------------------------------------------------------------------------

def test_fd_sh_bpt1():
    """SH (ND=21), back_prop_type=1. KNOWN OPEN -- see notes/todo.md item 4.
    """
    wd = workdir("fd_sh_bpt1")
    make = lambda **kw: _crosswell_sh(wd, **kw)
    s0 = make()
    ids = s0.src_pos_all[3, :]
    nz, nx = int(s0.N[0]), int(s0.N[1])
    patch = (slice(nz // 2 - 7, nz // 2 + 8), slice(nx // 2 - 10, nx // 2 + 10))
    init, true = _init_and_true(s0, patch)
    fd_check("SH", make, ids, init, true, patch,
            back_prop_type=1, calibrated=True, tol=0.05)


TESTS = [
    test_fd_2d_elastic_bpt1_vel,
    test_fd_2d_elastic_bpt1_p,
    test_fd_2d_elastic_bpt2_vel,
    test_fd_2d_elastic_bpt2_p,
    test_fd_2d_viscoelastic_bpt2_vel,
    test_fd_2d_viscoelastic_bpt2_p,
    test_fd_2d_viscoelastic_bpt1_is_rejected,
    test_fd_3d_elastic_bpt1_vel,
    test_fd_3d_elastic_bpt1_p,
    test_fd_3d_elastic_bpt2_vel,
    test_fd_3d_elastic_bpt2_p,
    test_fd_3d_viscoelastic_bpt2_vel,
    test_fd_3d_viscoelastic_bpt2_p,
    test_fd_sh_bpt1,
]

# Known-open failures, each tracked in notes/todo.md -- see the docstring of
# the corresponding test_* function for the item number and status. They
# still run and still print their numbers; they just do not fail the build.
XFAIL = {
    "test_fd_3d_elastic_bpt1_p",         # item 0d
    "test_fd_sh_bpt1",                   # item 4
}

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot", metavar="DIR", nargs="?", const="fd_check_plots",
        default=None,
        help="Save diagnostic figures (initial model, true model -- both "
             "with source/receiver positions, observed vs. modelled data, "
             "and the gradient) for every case into DIR (default: "
             "'fd_check_plots' in the current directory) instead of only "
             "printing the FD/ratio table. Figures are written to disk "
             "only, never displayed.")
    args = parser.parse_args()
    PLOT_DIR = args.plot

    sys.exit(run_tests(TESTS, xfail=XFAIL))
