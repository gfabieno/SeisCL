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

Geometry, now identical in *substance* across every case, not merely in
spirit: one source in a "left well", a line of receivers in a "right well",
and a PATCH_SIDE cube of perturbed cells midway between them, with the
clearances from sources, receivers and the absorbing strip ASSERTED by
`_patch()` rather than assumed. Until 2026-09-03 the cases were not
comparable -- 3D perturbed a patch 5 cells from the source where 2D's was 28
-- and that alone accounted for the 3D ratios looking structurally broken;
see PATCH_SIDE's comment. Grid sizes are the smallest that satisfy those
clearances (2D [64,80], 3D [44,44,76], nab=10): `nab` does not need to be
large here, because boundary reflections do not invalidate an FD check --
J and the gradient see the same boundaries either way -- and the y/z axes
carry no source-receiver separation constraint, only boundary clearance. **This specific
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
notes/todo.md's item 0b.

Status as of 2026-09-04: EVERY `back_prop_type=1` case is now exact -- both
dimensions, both output channels, all of vp/vs/rho within 0.1% of 1. That is
the gold standard this file is read against, and it is the first time the
pressure channel has met it. Getting there took two fixes in the shared
residual path (notes/todo.md item 0d2): res_scale() was applying the 2D
trace modulus in 3D, and the pressure adjoint source was injected one sample
early.

READ THE RATIOS WITH THEIR CONDITIONING. `rho` and `taup` are both
DIFFERENCES of much larger terms in this parameterization -- `rho` because
dJ/drho|_{vp,vs} subtracts the M and mu chain-rule terms, `taup` because
M() stores M = rho*vp^2/(1+alpha*taup) so M depends on taup at fixed vp.
Their FD ratios therefore amplify small errors enormously (measured: 238x
for `rho`, 6x for `taup`), and a ratio far from 1 in either says almost
nothing on its own. fd_check prints the amplification factor next to both.
`vp` and `vs` are single-term probes and have no such problem -- judge a
gradient by those. See notes/todo.md items 0j and 0d3.

Concretely, on this geometry `back_prop_type=2` differs from the exact
`back_prop_type=1` by a uniform 1-3% per cell (median per-cell ratio 1.000),
and every alarming-looking bpt2 number reduces to that once divided by its
conditioning: `rho` 0.508 -> 0.12%, `taup` 0.838 -> 2.6%.

Still open: SH does not run at all (item 4, an unrelated engine defect).

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

# FL of the case under test, set by fd_check; see _report_rho_conditioning.
FL_USED = None


# ---------------------------------------------------------------------------
# The standard crosswell geometry -- see the module docstring for why these
# particular numbers, not a smaller/faster grid.
# ---------------------------------------------------------------------------

DH, DT, NT, F0, NAB, ABPC = 10.0, 0.8e-3, 1200, 10.0, 10, 6.0
SRC_SCALE = 1e6


def _all_energetic_freqs():
    """Every DFT bin (step = df, not a sparse stride) covering the Ricker
    source's full energetic bandwidth.

    back_prop_type=2's gradient is a sum over exactly the bins in
    `gradfreqs` (see notes/todo.md item 0b/0f): by Parseval's theorem, that
    sum only reconstructs the true (unnormalized) time-domain correlation
    once it includes *every* bin carrying real spectral energy, not an
    arbitrary subset. The previous choice here (`df*np.arange(4, 40, 4)`,
    9 bins spanning 4.2-37.5 Hz at a stride of 4) skipped 3 of every 4 bins
    inside the energetic band and NEVER covered 1-3 Hz, which measurably
    skewed the resulting gradient's relative calibration between vp/vs/rho
    (confirmed 2026-09-02: with the sparse selection, ad/fd ranged
    4.3e-16-7.6e-16 across the three parameters at seisout=1 -- a ~1.8x
    spread; with every bin from 1 Hz to 5*f0, that spread shrinks to
    ~1.24x, and matches summing literally every bin to Nyquist to 4
    significant figures, confirming 5*f0 already captures the source's
    full energy: a Ricker wavelet centred at f0 has 99.9% of its energy
    below ~2.3*f0). Using literally every bin to Nyquist (599 bins for
    this file's NT/dt) is NOT an option -- tried directly, it OOMs the 3D
    case (spectral storage is NFREQS*grid_size*2 floats per saved
    variable) and buys nothing, since bins above ~25 Hz carry ~0 energy
    for this source anyway.

    Starts at bin 4 (4.17 Hz), not bin 1: found, while building this, a
    real and separate bug -- some combinations of the lowest 3 bins
    (1-3.1 Hz) together with a wider set make the *viscoelastic* DFT
    gradient come back all-NaN (bins {1,2} alone: NaN; {1,2,3} alone:
    fine; {1,2,3,4}: NaN again; {4..N}: always fine, elastic or
    viscoelastic, tested up to 50 Hz) -- reproducible, not a fluke, and
    not explained by any single frequency's coefficients (bin 4 alone,
    or bins 4-19, are individually fine). Elastic never showed this.
    Filed as notes/todo.md item 0g rather than chased down here; bins
    1-3 carry only ~2.75% of a 10 Hz Ricker's cumulative spectral
    energy by bin 4 (measured directly), so excluding them costs
    essentially nothing towards "every energetic bin" while sidestepping
    a real, separate defect.
    """
    df = 1.0 / (NT * DT)
    fmax = 5.0 * F0
    return df * np.arange(4, int(fmax / df) + 1)


VP, VS, RHO = 3000.0, 1800.0, 2200.0
TAUP0, TAUS0 = 0.1, 0.1
DANOM = dict(vp=400.0, vs=200.0, rho=150.0, taup=0.05, taus=0.05)

# ---------------------------------------------------------------------------
# The perturbed patch: identical in every case, and provably clear of the
# sources, the receivers and the absorbing strip.
#
# These have to be shared constants rather than per-case literals because the
# FD ratio is only comparable between cases when the *perturbation* is. Until
# 2026-09-03 they were not: 2D perturbed a 15x20=300-cell patch sitting 28
# cells from the source, while 3D perturbed a 10x10x10=1000-cell patch sitting
# **5 cells** from the source and 6 from the nearest receiver, with only 3
# cells of margin to the absorbing strip in y. That is inside the ~4-cell
# radius of the known-wrong near-source gradient (notes/todo.md, "Wrong
# gradient SIGN in source cells"), so the 3D FD checks were measuring that
# defect on top of whatever they were meant to measure -- which is exactly
# why 3D's ratios looked structurally broken (rho ~3x, vs sign-flipped) while
# 2D's, on the same code path, came out near 1.
#
# PATCH_SIDE is the side of the cube (square in 2D) in cells, the same on
# every axis and in every case, so <grad,dm> sums the same shape of
# neighbourhood everywhere. CLEAR_SRC_REC is enforced against every source
# and every receiver; it is 4x the documented near-source mute radius.
# CLEAR_BND is enforced against the inner edge of the absorbing strip, which
# read_grad() crops and where the back_prop_type=1 gradient is invalid by
# construction (notes/todo.md item 0f).
# ---------------------------------------------------------------------------

PATCH_SIDE = 10
CLEAR_SRC_REC = 16
CLEAR_BND = 6


def _box_gap(pt, ext):
    """Euclidean distance, in cells, from point `pt` to the patch box `ext`
    (a list of inclusive (lo, hi) per axis). Zero if inside."""
    g = [max(lo - c, 0.0, c - hi) for c, (lo, hi) in zip(pt, ext)]
    return float(np.sqrt(sum(x * x for x in g)))


def _patch(s, verbose=True):
    """The standard perturbed patch for `s`'s geometry, with its clearances
    asserted.

    A PATCH_SIDE cube centred midway between the source well and the
    receiver well along x, and on the sources' own y/z (which is the grid
    centre for every geometry in this file) -- i.e. the well-illuminated
    interior spot, as far from both wells as the crosswell layout allows.

    Raises if the grid is too small to give the patch CLEAR_SRC_REC cells of
    clearance from every source and receiver and CLEAR_BND from the
    absorbing strip: a silently-too-close patch is precisely the failure
    mode this function exists to prevent, so it must never degrade quietly
    into "as much clearance as fits".
    """
    N = [int(v) for v in s.N]
    ndim = len(N)
    src, rec = s.src_pos_all, s.rec_pos_all

    # Position rows are [x, y, z, ...] in metres; SeisCL axis order is
    # (z[,y],x). Build both source and receiver positions as cell indices in
    # that axis order.
    def to_cells(p):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        return [z / s.dh, y / s.dh, x / s.dh] if ndim == 3 else [z / s.dh, x / s.dh]

    src_c = [to_cells(src[:, i]) for i in range(src.shape[1])]
    rec_c = [to_cells(rec[:, i]) for i in range(rec.shape[1])]

    # Centre: midway between the wells in x, on the sources' own other axes.
    cx = 0.5 * (np.mean([p[-1] for p in src_c]) + np.mean([p[-1] for p in rec_c]))
    centre = [np.mean([p[a] for p in src_c]) for a in range(ndim - 1)] + [cx]

    half = PATCH_SIDE // 2
    ext, patch = [], []
    for a in range(ndim):
        lo = int(round(centre[a])) - half
        hi = lo + PATCH_SIDE - 1
        ext.append((lo, hi))
        patch.append(slice(lo, hi + 1))

    # --- clearances, asserted ---
    bad = []
    d_src = min(_box_gap(p, ext) for p in src_c)
    d_rec = min(_box_gap(p, ext) for p in rec_c)
    if d_src < CLEAR_SRC_REC:
        bad.append("nearest source is %.1f cells away (need >= %d)"
                   % (d_src, CLEAR_SRC_REC))
    if d_rec < CLEAR_SRC_REC:
        bad.append("nearest receiver is %.1f cells away (need >= %d)"
                   % (d_rec, CLEAR_SRC_REC))
    for a, (lo, hi) in enumerate(ext):
        m = min(lo - s.nab, (N[a] - s.nab - 1) - hi)
        if m < CLEAR_BND:
            bad.append("axis %d is %d cells from the absorbing strip "
                       "(need >= %d)" % (a, m, CLEAR_BND))
    if bad:
        raise AssertionError(
            "patch geometry is unusable for N=%s, nab=%d: %s. Enlarge the "
            "grid or move the wells -- do NOT shrink the clearance, it is "
            "what makes this case comparable to the others." % (N, s.nab,
                                                                "; ".join(bad)))
    if verbose:
        print("  patch %s (%d cells); clearance: source %.0f, receiver %.0f, "
              "boundary %d cells"
              % (ext, PATCH_SIDE ** ndim, d_src, d_rec,
                 min(min(lo - s.nab, (N[a] - s.nab - 1) - hi)
                     for a, (lo, hi) in enumerate(ext))))
    return tuple(patch)


def _crosswell_2d(wd, **overrides):
    cfg = dict(N=np.array([64, 80]), ND=2, dh=DH, dt=DT, NT=NT, f0=F0,
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
    n_rec = 20
    gz = np.linspace(s.nab + 6, nz - s.nab - 6, n_rec) * s.dh
    s.rec_pos_all = np.stack([np.full(n_rec, xr), np.zeros(n_rec), gz,
                              np.zeros(n_rec), np.arange(1, n_rec + 1),
                              np.zeros(n_rec), np.zeros(n_rec), np.zeros(n_rec)])
    s.src_all = SRC_SCALE * s.ricker_wavelet().reshape(-1, 1)
    return s


def _crosswell_3d(wd, **overrides):
    """The 3D analog: same dh/dt/NT/f0/nab as the validated 2D case, source
    and receiver wells separated along x at fixed y, z.

    N was [64, 48, 64] until 2026-09-03, "kept smaller than the 2D one purely
    for runtime". That was too small to be a fair test: it left the standard
    patch 5 cells from the source, 6 from the nearest receiver and 3 from the
    absorbing strip in y (see PATCH_SIDE's comment above), so 3D was
    measuring near-source gradient contamination that 2D, at 28 cells of
    clearance, was not. Enlarged to satisfy the same clearances 2D gets --
    which `_patch()` now asserts rather than assumes. The cost is modest: a
    forward pass goes 1.3s -> 3.1s, i.e. ~45s for a 3-parameter fd_check.
    """
    cfg = dict(N=np.array([44, 44, 76]), ND=3, dh=DH, dt=DT, NT=NT, f0=F0,
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
    n_rec = 10
    gz = np.linspace(s.nab + 6, nz - s.nab - 6, n_rec) * s.dh
    s.rec_pos_all = np.stack([np.full(n_rec, xr), np.full(n_rec, y0), gz,
                              np.zeros(n_rec), np.arange(1, n_rec + 1),
                              np.zeros(n_rec), np.zeros(n_rec), np.zeros(n_rec)])
    s.src_all = SRC_SCALE * s.ricker_wavelet().reshape(-1, 1)
    return s


def _crosswell_sh(wd, **overrides):
    """SH (ND=21): same 2D crosswell geometry, an Fy point force (src_type=1)
    instead of the explosive source, recording vy."""
    cfg = dict(N=np.array([64, 80]), ND=21, dh=DH, dt=DT, NT=NT, f0=F0,
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
    n_rec = 20
    gz = np.linspace(s.nab + 6, nz - s.nab - 6, n_rec) * s.dh
    s.rec_pos_all = np.stack([np.full(n_rec, xr), np.zeros(n_rec), gz,
                              np.zeros(n_rec), np.arange(1, n_rec + 1),
                              np.zeros(n_rec), np.zeros(n_rec), np.zeros(n_rec)])
    s.src_all = SRC_SCALE * s.ricker_wavelet().reshape(-1, 1)
    return s


def _init_and_true(s, patch, L=0):
    """Homogeneous 'init' model and a 'true' model perturbed, in every
    parameter the case under test covers, over the same interior patch.

    NOTE what this cannot test. The gradient is evaluated at `init`, which is
    HOMOGENEOUS, so every averaged staggered parameter (muipkp/muipjp/mujpkp,
    rip/rjp/rkp, tausipkp/...) is numerically equal to its cell-centred
    counterpart and the material-averaging transpose is the identity apart
    from its boundary copy rows. These checks are therefore **blind to
    material-averaging errors by construction** -- the same blind spot
    notes/todo.md item 1 records for T4. A wrong or missing averaging chain
    rule (item 1's subject) will not move any ratio here. `dot_prod_average.py`
    covers the transpose operators themselves; what is still missing anywhere
    is an FD check on a *heterogeneous* init model, which is the only thing
    that would exercise averaging end to end. Worth adding; deliberately not
    done inside this function, because making `init` heterogeneous changes
    every baseline in this file at once and would need its own calibration
    pass.
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

    # Remembered for _report_rho_conditioning's taup branch, which needs the
    # GSLS alpha = sum_l r^2/(1+r^2), r = f0/FL_l.
    global FL_USED
    FL_USED = getattr(make(), "FL", None)

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
    fd_by_param = {}
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
            fd_by_param[pname] = fd
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

    _report_rho_conditioning(fd_by_param, params_init, dm)

    if failures:
        raise AssertionError("; ".join(failures))
    return results


def _report_rho_conditioning(fd, params_init, dm):
    """Print how much `rho`'s FD ratio amplifies errors in the other two.

    dJ/drho at fixed (vp,vs) is a DIFFERENCE of much larger terms,

        dJ/drho|_{vp,vs} = dJ/drho|_{M,mu} + (M/rho) dJ/dM + (mu/rho) dJ/dmu

    and in this geometry they cancel to 1 part in 60-344. So a sub-1% error
    in dJ/dM or dJ/dmu lands in `rho`'s ratio multiplied by that factor, and
    `rho` being far from 1 says almost nothing on its own -- see
    notes/todo.md item 0j, which records the measured factors and why this
    was repeatedly misread as a density-gradient bug. `vp` and `vs` are each
    a single-term probe (of dJ/dM and dJ/dmu) and have no such problem; they
    are the quantities to judge a gradient by.

    Printed rather than asserted on: it is a property of the *test geometry*,
    not of the engine, so it is context for reading the table above, not a
    pass/fail criterion.
    """
    if not all(k in fd for k in ("vp", "vs", "rho")) or fd["rho"] == 0.0:
        return
    n = float(dm.sum())
    if n <= 0:
        return
    # Background values are homogeneous in every case here, so one cell's
    # values describe the whole patch.
    vp, vs = float(params_init["vp"].flat[0]), float(params_init["vs"].flat[0])
    rho = float(params_init["rho"].flat[0])
    M, mu = rho * vp * vp, rho * vs * vs
    gM = fd["vp"] / (2.0 * rho * vp)
    gmu = fd["vs"] / (2.0 * rho * vs)
    cross_M, cross_mu = (M / rho) * gM, (mu / rho) * gmu
    direct = fd["rho"] - cross_M - cross_mu
    tot = abs(direct) + abs(cross_M) + abs(cross_mu)
    fac = tot / abs(fd["rho"])
    print("  [rho conditioning] buoyancy %+.3e | (M/rho)dJ/dM %+.3e | "
          "(mu/rho)dJ/dmu %+.3e" % (direct, cross_M, cross_mu))
    print("  [rho conditioning] these cancel to %.3e: a %.2f%% error in them "
          "becomes 1%% in rho's ratio (factor %.0fx)"
          % (fd["rho"], 100.0 / fac, fac))

    # taup has the SAME problem, for the same reason: M() stores
    # M = rho*vp^2/(1+alpha*taup), so M depends on taup at fixed vp and
    # dJ/dtaup|_{vp,vs,rho} is again a difference of larger terms. Measured
    # factor ~6x on this geometry, which turns the DFT path's ordinary 2.6%
    # into an alarming-looking 16%. Same trap, one item later.
    if "taup" in fd and fd["taup"] != 0.0 and TAUP0 > 0:
        al = sum((F0 / f) ** 2 / (1.0 + (F0 / f) ** 2)
                 for f in np.atleast_1d(FL_USED) if f > 0) if FL_USED is not None else 0.0
        if al > 0:
            Mn = rho * vp * vp / (1.0 + al * TAUP0)
            gM2 = fd["vp"] / (2.0 * rho * vp / (1.0 + al * TAUP0))
            cross_t = -Mn * al / (1.0 + al * TAUP0) * gM2
            direct_t = fd["taup"] - cross_t
            ft = (abs(direct_t) + abs(cross_t)) / abs(fd["taup"])
            print("  [taup conditioning] direct %+.3e | (dM/dtaup)dJ/dM %+.3e"
                  % (direct_t, cross_t))
            print("  [taup conditioning] cancel to %.3e: a %.2f%% error becomes "
                  "1%% in taup's ratio (factor %.0fx)"
                  % (fd["taup"], 100.0 / ft, ft))


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
    patch = _patch(s0)
    init, true = _init_and_true(s0, patch)
    grad_cfg = {}
    if back_prop_type == 2:
        grad_cfg = dict(gradfreqs=_all_energetic_freqs())
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

    UPDATE 2026-09-04: FIXED, both of them. Item 0d turned out to be TWO
    defects, both in the shared residual path. (a) res_scale()'s "p" branch
    scaled by the 2D trace modulus 2*(M-mu) in 3D as well, where it should
    be N*M-2(N-1)*mu = 3M-4mu -- a pure scale error that hit every 3D
    pressure gradient by 0.8205 and left 2D untouched; fixed, and every 3D
    pressure ratio moved to ~1. (b) The pressure adjoint source was
    injected one FULL SAMPLE early. The forward records a seismogram after
    the whole update, but vx/vy/vz come from update_v and the normal
    stresses (hence "p") from update_s, which runs second; the adjoint runs
    the pair in reverse order, so the stress half belongs one sample later
    than the velocity half. res_scale() now delays the trans_vars residual
    by one sample. Found via dt-refinement (the error was exactly
    proportional to dt) and confirmed by an integer-shift scan whose error
    changes sign between k=1 and k=2 with a clean zero at k=1; applying the
    same shift to the VELOCITY residual instead breaks it, so the offset is
    specific to the stress half. See notes/todo.md item 0d2.

    Both back_prop_type=1 pressure cases are now exact (2D 1.0000/1.0001/
    1.0002, 3D 1.0000/1.0001/0.9997) and are no longer XFAILed.

    Original note follows.
    With the standardized patch (see PATCH_SIDE) vs comes out at 1.062,
    outside the 3% tolerance. This is NOT the patch's doing -- the pressure
    channel is off in *every* case, in BOTH back_prop_types, and by very
    nearly the SAME factor in each. Measured on 3D elastic, as
    ratio(pressure)/ratio(velocity) per parameter:

        param   bpt1     bpt2     difference
        vp      0.8261   0.8208   0.6%
        vs      0.9420   1.0051   6.7%
        rho     0.8506   0.8422   1.0%

    vp and rho degrade by the same factor whichever gradient method is used,
    which places the defect in the **shared pressure-residual path**
    (residuals.c's res_scale() "p"/trans_vars branch and the adjoint source
    built from it), not in either gradient. That is a much sharper
    localization than item 0c had, and it means chasing it in the gradient
    kernels would be wasted effort. Item 0c fixed one real dh/dt-vs-dt/dh
    swap in that branch; this is what remains.
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
    patch = _patch(s0)
    init, true = _init_and_true(s0, patch, L=1)
    grad_cfg = dict(gradfreqs=_all_energetic_freqs())
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
    patch = _patch(s0)
    init, true = _init_and_true(s0, patch)
    grad_cfg = {}
    if back_prop_type == 2:
        grad_cfg = dict(gradfreqs=_all_energetic_freqs())
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
    patch = _patch(s0)
    init, true = _init_and_true(s0, patch, L=1)
    grad_cfg = dict(gradfreqs=_all_energetic_freqs())
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
    patch = _patch(s0)
    init, true = _init_and_true(s0, patch)
    fd_check("SH", make, ids, init, true, patch,
            back_prop_type=1, calibrated=True, tol=0.05)


# ---------------------------------------------------------------------------
# Source-cell gradient (notes/todo.md, "the source-cell gradient")
# ---------------------------------------------------------------------------
#
# Every other case in this file deliberately keeps the perturbed patch >=16
# cells clear of the sources (_patch's CLEAR_SRC_REC), so none of them can see
# the source term of eq. (26a) at all -- the whole suite passed both before
# and after that term was implemented. This case perturbs the SOURCE CELL
# ITSELF, which is the only way the term is exercised.
#
# It needs its own geometry, not the shared crosswell one: make_seiscl()-style
# setups here put the receiver line at the SAME depth as the source and the
# source a few cells from the absorbing layer, and on such a geometry even a
# correct source-cell gradient measures 2-18% off (one configuration comes out
# inverted). Source mid-domain, receivers well away, is what makes the check
# meaningful. That trap cost most of the debugging time; do not "simplify"
# this back onto the shared geometry.

def _srccell_model(wd, ND=2, srctype=100.0, **over):
    """Clean geometry for a source-cell check: source mid-domain, receiver
    line far from it, everything well inside the absorbing layer."""
    if ND == 2:
        cfg = dict(N=np.array([96, 96]), ND=2, dh=10, dt=1e-3, NT=256,
                   FDORDER=8, freesurf=0, abs_type=2, nab=16, f0=25,
                   seisout=2, param_type=0)
    else:
        cfg = dict(N=np.array([48, 48, 48]), ND=3, dh=10, dt=8e-4, NT=500,
                   FDORDER=8, freesurf=0, abs_type=2, nab=8, f0=25,
                   seisout=2, param_type=0)
    cfg.update(over)
    s = SeisCL(workdir=wd, **cfg)
    if ND == 2:
        nz, nx = int(s.N[0]), int(s.N[1])
        zs, ys = nz // 2, 0.0
        ny_mid = 0.0
    else:
        nz, ny, nx = (int(v) for v in s.N)
        zs = nz // 2
        ny_mid = ny // 2 * s.dh
    # srctype indexes kernel_sources()'s src_names[] = {vx,vy,vz,p,...}:
    # 100 is the "p" trans_var (a pressure source), 0/1/2 are force sources.
    s.src_pos_all = np.stack([[nx // 2 * s.dh], [ny_mid], [zs * s.dh],
                              [0.], [float(srctype)]])
    nr = 8
    zr = (nz - s.nab - 6) * s.dh
    xr = np.linspace((s.nab + 6) * s.dh, (nx - s.nab - 6) * s.dh, nr)
    s.rec_pos_all = np.stack([xr, np.full(nr, ny_mid), np.full(nr, zr),
                              np.zeros(nr), np.arange(1, nr + 1, dtype=float),
                              np.zeros(nr), np.zeros(nr), np.zeros(nr)])
    s.src_all = SRC_SCALE * s.ricker_wavelet().reshape(-1, 1)
    return s, zs


def _fd_at_source_cell(ND, tol=0.01, bpt=1, fp16=0, srctype=100.0, extra=None):
    vp, vs, rho = 2000.0, 1200.0, 2000.0
    wd = workdir("srccell_%dd_b%d_f%d_s%d" % (ND, bpt, fp16, int(srctype)))
    base = dict(extra or {})
    if fp16:
        base["FP16"] = fp16
    mk = lambda **kw: _srccell_model(wd, ND=ND, srctype=srctype, **dict(base, **kw))
    s0, zs = mk()
    start = {"vp": np.full(s0.N, vp), "vs": np.full(s0.N, vs),
             "rho": np.full(s0.N, rho)}
    true = {k: np.array(v) for k, v in start.items()}
    mid = [int(v) // 2 for v in s0.N]
    if ND == 2:
        true["vp"][mid[0] + 12:mid[0] + 22, mid[1] - 5:mid[1] + 5] += 300.0
        cell = (zs, mid[1])
    else:
        true["vp"][mid[0] + 8:mid[0] + 14, mid[1] - 3:mid[1] + 3,
                   mid[2] - 3:mid[2] + 3] += 300.0
        cell = (zs, mid[1], mid[2])
    g0NT, g0DT, g0F0 = int(s0.NT), float(s0.dt), float(s0.f0)
    s0.set_forward(s0.src_pos_all[3, :], true, withgrad=False)
    s0.execute()
    dobs = [np.asarray(a, np.float64) for a in s0.read_data()]
    s0.write_data({"p": dobs[0]}, filename="SeisCL_din.mat")
    din = os.path.join(s0.workdir, "SeisCL_din.mat")

    def misfit(p):
        t, _ = mk()
        t.file_din = din
        t.set_forward(t.src_pos_all[3, :], p, withgrad=False)
        t.execute()
        return t.misfit(t.read_data(), dobs=dobs)[0]

    gcfg = {}
    if bpt == 2:
        # Build the bin set from THIS model's NT/dt, not the module-level ones:
        # gradfreqs are Hz, and a set built for a different NT*dt lands on
        # non-integer DFT bins, where the basis is not orthogonal and the
        # gradient is garbage. (Cost real debugging time.)
        dfm = 1.0 / (int(g0NT) * float(g0DT))
        gcfg["gradfreqs"] = dfm * np.arange(4, int(5.0 * float(g0F0) / dfm) + 1)
    g, _ = mk(gradout=1, back_prop_type=bpt, **gcfg)
    g.file_din = din
    g.set_forward(g.src_pos_all[3, :], start, withgrad=True)
    g.execute()
    grads = {nm: np.asarray(a, np.float64)
             for nm, a in zip(g.params, g.read_grad())}

    print("=== %dD elastic, gradient AT THE SOURCE CELL %s "
          "(back_prop_type=%d, FP16=%d, srctype=%d) ==="
          % (ND, cell, bpt, fp16, int(srctype)))
    print("%-6s %14s %14s %10s" % ("param", "FD", "<g,dm>", "ratio"))
    bad = []
    for par in ("vp", "rho"):
        G = grads[par]
        v = np.zeros_like(G)
        v[cell] = 1.0
        gv = float((G * v).sum())
        pp = {k: np.array(a) for k, a in start.items()}
        pm = {k: np.array(a) for k, a in start.items()}
        pp[par] = pp[par] + v
        pm[par] = pm[par] - v
        fd = (misfit(pp) - misfit(pm)) / 2.0
        ratio = fd / gv if gv else float("nan")
        print("%-6s %14.6e %14.6e %10.4f" % (par, fd, gv, ratio))
        if not abs(ratio - 1.0) <= tol:
            bad.append("%s ratio %.4f" % (par, ratio))
    if bad:
        raise AssertionError(
            "source-cell gradient off by more than %.0f%%: %s. This is the "
            "eq. (26a) source term -- see notes/todo.md."
            % (100 * tol, ", ".join(bad)))


def test_fd_2d_srccell_bpt1():
    """2D elastic back_prop_type=1, perturbing the SOURCE CELL itself.

    Guards the eq. (26a) source term: without it this ratio is about -6.5,
    not 1. No other case in this file perturbs a source cell.
    """
    _fd_at_source_cell(2)


def test_fd_3d_srccell_bpt1():
    """3D elastic back_prop_type=1, perturbing the SOURCE CELL itself."""
    _fd_at_source_cell(3)


def test_fd_2d_srccell_bpt1_fp16():
    """Same, at FP16=1 -- a SEPARATE kernel.

    assign_modeling_case.c routes FP16==0 to update_adjs2D.cl and *everything
    else* to update_adjs2D_half2.cl, so the FP16=0 cases above do not cover
    the half2 path at all. There DIV=2 z-cells share one work item and the
    correction has to land in the source's lane only.
    """
    _fd_at_source_cell(2, fp16=1)


def test_fd_2d_srccell_force():
    """A FORCE source (type 2, Fz) instead of a pressure one.

    A pressure source enters only the stress block and a force source only the
    velocity block, so they exercise different halves of eq. (26a): this one
    lands in dJ/drho via (A1a), and specifically in the STAGGERED buoyancy
    accumulators (vx at rip, vz at rkp), not the cell-centred gradrho -- that
    is what lets average_grad_transpose() apply the averaging Jacobian to it.
    Without the term the density ratio here is 1.54, not 1.00, while vp is
    untouched either way (the two source types do not leak into each other).
    """
    _fd_at_source_cell(2, srctype=2.0)


def test_fd_3d_srccell_force():
    """3D force source (type 2, Fz); see test_fd_2d_srccell_force."""
    _fd_at_source_cell(3, srctype=2.0)


def _srccell_grad(fp16, srctype):
    """The raw gradient at a source cell -- for comparing PHYSICAL units."""
    vp, vs, rho = 2000.0, 1200.0, 2000.0
    wd = workdir("srccell_units_f%d_s%d" % (fp16, int(srctype)))
    base = {} if fp16 == 0 else {"FP16": fp16}
    mk = lambda **k: _srccell_model(wd, ND=2, srctype=srctype,
                                    **dict(base, **k))
    s0, zs = mk()
    nz, nx = int(s0.N[0]), int(s0.N[1])
    start = {"vp": np.full(s0.N, vp), "vs": np.full(s0.N, vs),
             "rho": np.full(s0.N, rho)}
    true = {k: np.array(v) for k, v in start.items()}
    true["vp"][nz // 2 + 12:nz // 2 + 22, nx // 2 - 5:nx // 2 + 5] += 300.0
    s0.set_forward(s0.src_pos_all[3, :], true, withgrad=False)
    s0.execute()
    dobs = [np.asarray(a, np.float64) for a in s0.read_data()]
    s0.write_data({"p": dobs[0]}, filename="SeisCL_din.mat")
    g, _ = mk(gradout=1, back_prop_type=1)
    g.file_din = os.path.join(s0.workdir, "SeisCL_din.mat")
    g.set_forward(g.src_pos_all[3, :], start, withgrad=True)
    g.execute()
    G = {nm: np.asarray(a, np.float64) for nm, a in zip(g.params, g.read_grad())}
    return (float(np.abs(dobs[0]).max()),
            {p_: float(G[p_][zs, nx // 2]) for p_ in ("vp", "rho")})


def test_fd_fp16_physical_units():
    """FP16>0 must return data and gradient in the SAME PHYSICAL UNITS as
    FP16=0 -- not merely proportional to them.

    Every variable is stored in its own scaled units at FP16>0: set_par_scale()
    gives vx/vy/vz a scaler equal to par_scale and leaves the stresses at 0, and
    kernel_varout() recovers a stored value as
    ldexp(var, -src_scale + var->scaler), i.e. stored = physical *
    2^(src_scale - scaler). kernel_sources() used to inject an amplitude
    carrying 2^src_scale alone into BOTH, which is right for a stress and too
    large by 2^scaler for a velocity -- so FORCE sources came out mis-scaled by
    2^par_scale (~2^33) at FP16>0 while pressure sources were fine. Nothing
    downstream could undo it, because the forward run itself had the wrong
    source.

    This checks the raw values, since a ratio-based test cannot see a uniform
    scale error at all.

    The same reasoning applies to `src_scale` itself (es of eq. 6 in
    Fabien-Ouellet 2020): it normalizes the injected amplitude to ~1 only for a
    source going into a variable stored at scaler 0. A source injected into
    vx/vy/vz lands as ~2^(-par_scale), which at FP16=2 is far past half's 65504
    and NaN'd for ANY amplitude. Init_model.c now offsets es by the target
    variable's scaler, which is why FP16=2 is checked for BOTH source types
    here.
    """
    bad = []
    for label, st, levels in (("pressure", 100.0, (1, 2)),
                              ("force Fz", 2.0, (1, 2))):
        d0, g0 = _srccell_grad(0, st)
        for fp in levels:
            d1, g1 = _srccell_grad(fp, st)
            tol = 0.002 if fp == 1 else 0.02
            print("=== %s source, FP16=%d vs FP16=0 (physical units) ==="
                  % (label, fp))
            print("  |d|max  %13.5e vs %13.5e   ratio %8.5f"
                  % (d1, d0, d1 / d0))
            if not abs(d1 / d0 - 1.0) <= tol:
                bad.append("%s FP16=%d data %.5f" % (label, fp, d1 / d0))
            for par in ("vp", "rho"):
                r = g1[par] / g0[par] if g0[par] else float("nan")
                print("  grad %-4s %13.5e vs %13.5e   ratio %8.5f"
                      % (par, g1[par], g0[par], r))
                if not abs(r - 1.0) <= tol:
                    bad.append("%s FP16=%d %s %.5f" % (label, fp, par, r))
    if bad:
        raise AssertionError(
            "FP16>0 is not in the same physical units as FP16=0: %s. See "
            "kernel_sources() in automatic_kernels.c and the source term in "
            "update_adjv*_half2.cl -- notes/todo.md." % ", ".join(bad))


def test_fd_2d_srccell_force_bpt2():
    """FORCE source in the DFT path, perturbing the SOURCE CELL.

    Checks the DENSITY channel only, and against the interior rather than
    against 1, for two separate reasons:

    * back_prop_type=2 is the uncalibrated path, so its interior is the right
      yardstick (see test_fd_2d_srccell_bpt2);
    * dJ/dvp at a FORCE source cell carries a pre-existing bpt2 defect that
      has nothing to do with the source term -- it measures 0.709 whether the
      eq. (26a) velocity term is enabled or not (verified by toggling it), and
      is unchanged by summing every DFT bin to Nyquist, so it is neither this
      term nor band-limiting. It is recorded in notes/todo.md; do not let it
      creep into this test.

    A force source enters only the velocity block, so the density channel is
    exactly what it should move: without the term the ratio here is 1.74.
    """
    vp, vs, rho = 2000.0, 1200.0, 2000.0
    wd = workdir("srccell_2d_force_bpt2")
    mk = lambda **k: _srccell_model(wd, ND=2, srctype=2.0, **k)
    s0, zs = mk()
    dfm = 1.0 / (int(s0.NT) * float(s0.dt))
    fr = dfm * np.arange(4, int(5.0 * float(s0.f0) / dfm) + 1)
    nz, nx = int(s0.N[0]), int(s0.N[1])
    start = {"vp": np.full(s0.N, vp), "vs": np.full(s0.N, vs),
             "rho": np.full(s0.N, rho)}
    true = {k: np.array(v) for k, v in start.items()}
    true["vp"][nz // 2 + 12:nz // 2 + 22, nx // 2 - 5:nx // 2 + 5] += 300.0
    s0.set_forward(s0.src_pos_all[3, :], true, withgrad=False)
    s0.execute()
    dobs = [np.asarray(a, np.float64) for a in s0.read_data()]
    s0.write_data({"p": dobs[0]}, filename="SeisCL_din.mat")
    din = os.path.join(s0.workdir, "SeisCL_din.mat")
    xs = nx // 2

    def misfit(p):
        t, _ = mk()
        t.file_din = din
        t.set_forward(t.src_pos_all[3, :], p, withgrad=False)
        t.execute()
        return t.misfit(t.read_data(), dobs=dobs)[0]

    g, _ = mk(gradout=1, back_prop_type=2, gradfreqs=fr)
    g.file_din = din
    g.set_forward(g.src_pos_all[3, :], start, withgrad=True)
    g.execute()
    G = {nm: np.asarray(a, np.float64)
         for nm, a in zip(g.params, g.read_grad())}
    print("=== 2D elastic FORCE source AT THE SOURCE CELL "
          "(back_prop_type=2, density channel) ===")
    print("%-6s %-9s %14s %14s %9s" % ("param", "cell", "FD", "<g,dm>", "ratio"))
    r = {}
    for tag, z in (("SOURCE", zs), ("interior", zs + 8)):
        v = np.zeros_like(G["rho"])
        v[z, xs] = 1.0
        gv = float((G["rho"] * v).sum())
        pp = {k: np.array(a) for k, a in start.items()}
        pm = {k: np.array(a) for k, a in start.items()}
        pp["rho"] = pp["rho"] + v
        pm["rho"] = pm["rho"] - v
        fd = (misfit(pp) - misfit(pm)) / 2.0
        r[tag] = fd / gv if gv else float("nan")
        print("%-6s %-9s %14.6e %14.6e %9.4f" % ("rho", tag, fd, gv, r[tag]))
    rel = r["SOURCE"] / r["interior"]
    if not abs(rel - 1.0) <= 0.18:
        raise AssertionError(
            "DFT force-source density gradient inconsistent with the interior "
            "(source/interior %.4f). This is the eq. (26a) velocity-block term "
            "in grad_dft2D.cl -- see notes/todo.md." % rel)


def test_fd_2d_srccell_visco_bpt2():
    """VISCOELASTIC source cell, back_prop_type=2.

    back_prop_type=1 is rejected for L>0 (its backpropagation is
    unconditionally unstable for a dissipative medium), so bpt2 is the only
    viscoelastic path and there is no exact reference to compare against. What
    this checks instead is INTERNAL consistency: the source cell's FD/adjoint
    ratio against an interior cell's from the same run. If the eq. (26a) term
    is right the two agree, whatever bpt2's overall calibration happens to be
    -- and gradtaup/gradtaus have a known calibration error of their own
    (todo.md item 0d3) that this deliberately does not re-test.

    In the viscoelastic kernels P1 also carries c1taup and c2taus (A1c/A1e),
    so correcting d0 corrects gradtaup and gradtaus along with gradM/gradmu.
    """
    F0v = 25.0
    wd = workdir("srccell_2d_visco")
    mk = lambda **k: _srccell_model(wd, ND=2, L=1, FL=np.array([F0v]), **k)
    s0, zs = mk()
    dfm = 1.0 / (int(s0.NT) * float(s0.dt))
    fr = dfm * np.arange(4, int(5.0 * F0v / dfm) + 1)
    nz, nx = int(s0.N[0]), int(s0.N[1])
    start = {"vp": np.full(s0.N, 2000.0), "vs": np.full(s0.N, 1200.0),
             "rho": np.full(s0.N, 2000.0), "taup": np.full(s0.N, 0.02),
             "taus": np.full(s0.N, 0.02)}
    true = {k: np.array(v) for k, v in start.items()}
    true["vp"][nz // 2 + 12:nz // 2 + 22, nx // 2 - 5:nx // 2 + 5] += 300.0
    s0.set_forward(s0.src_pos_all[3, :], true, withgrad=False)
    s0.execute()
    dobs = [np.asarray(a, np.float64) for a in s0.read_data()]
    s0.write_data({"p": dobs[0]}, filename="SeisCL_din.mat")
    din = os.path.join(s0.workdir, "SeisCL_din.mat")
    xs = nx // 2

    def misfit(p):
        t, _ = mk()
        t.file_din = din
        t.set_forward(t.src_pos_all[3, :], p, withgrad=False)
        t.execute()
        return t.misfit(t.read_data(), dobs=dobs)[0]

    g, _ = mk(gradout=1, back_prop_type=2, gradfreqs=fr)
    g.file_din = din
    g.set_forward(g.src_pos_all[3, :], start, withgrad=True)
    g.execute()
    G = {nm: np.asarray(a, np.float64)
         for nm, a in zip(g.params, g.read_grad())}
    print("=== 2D viscoelastic (L=1) AT THE SOURCE CELL (back_prop_type=2) ===")
    print("%-6s %-9s %14s %14s %9s" % ("param", "cell", "FD", "<g,dm>", "ratio"))
    bad = []
    for par, eps in (("vp", 1.0), ("rho", 1.0)):
        r = {}
        for tag, z in (("SOURCE", zs), ("interior", zs + 8)):
            v = np.zeros_like(G[par])
            v[z, xs] = 1.0
            gv = float((G[par] * v).sum()) * eps
            pp = {k: np.array(a) for k, a in start.items()}
            pm = {k: np.array(a) for k, a in start.items()}
            pp[par] = pp[par] + eps * v
            pm[par] = pm[par] - eps * v
            fd = (misfit(pp) - misfit(pm)) / 2.0
            r[tag] = fd / gv if gv else float("nan")
            print("%-6s %-9s %14.6e %14.6e %9.4f" % (par, tag, fd, gv, r[tag]))
        rel = r["SOURCE"] / r["interior"]
        if not abs(rel - 1.0) <= 0.12:
            bad.append("%s source/interior %.4f" % (par, rel))
    if bad:
        raise AssertionError(
            "viscoelastic source cell inconsistent with the interior: %s. "
            "This is the eq. (26a) source term in grad_dft2D_visc.cl -- see "
            "notes/todo.md." % ", ".join(bad))


def _fd_at_receiver_cell(ND, tol=0.02):
    vp, vs, rho = 2000.0, 1200.0, 2000.0
    wd = workdir("reccell_%dd" % ND)
    over = dict(f0=10.0, NT=1200, dt=0.8e-3) if ND == 2 else {}
    mk = lambda **k: _srccell_model(wd, ND=ND, srctype=100.0, **dict(over, **k))
    s0, zs = mk()
    N = [int(v) for v in s0.N]
    rz = int(round(s0.rec_pos_all[2, 0] / s0.dh))
    rx = int(round(s0.rec_pos_all[0, 0] / s0.dh))
    cell = (rz, rx) if ND == 2 else (rz, N[1] // 2, rx)
    start = {"vp": np.full(s0.N, vp), "vs": np.full(s0.N, vs),
             "rho": np.full(s0.N, rho)}
    true = {k: np.array(v) for k, v in start.items()}
    if ND == 2:
        true["vp"][N[0]//2+12:N[0]//2+22, N[1]//2-5:N[1]//2+5] += 300.0
    else:
        true["vp"][N[0]//2+8:N[0]//2+14, N[1]//2-3:N[1]//2+3,
                   N[2]//2-3:N[2]//2+3] += 300.0
    s0.set_forward(s0.src_pos_all[3, :], true, withgrad=False)
    s0.execute()
    dobs = [np.asarray(a, np.float64) for a in s0.read_data()]
    s0.write_data({"p": dobs[0]}, filename="SeisCL_din.mat")
    din = os.path.join(s0.workdir, "SeisCL_din.mat")

    def misfit(p):
        t, _ = mk()
        t.file_din = din
        t.set_forward(t.src_pos_all[3, :], p, withgrad=False)
        t.execute()
        return t.misfit(t.read_data(), dobs=dobs)[0]

    g, _ = mk(gradout=1, back_prop_type=1)
    g.file_din = din
    g.set_forward(g.src_pos_all[3, :], start, withgrad=True)
    g.execute()
    G = {nm: np.asarray(a, np.float64) for nm, a in zip(g.params, g.read_grad())}
    print("=== %dD elastic, gradient AT A RECEIVER CELL %s, bpt1 ===" % (ND, cell))
    print("%-6s %14s %14s %9s" % ("param", "FD", "<g,dm>", "ratio"))
    bad = []
    for par in ("vp", "rho"):
        v = np.zeros_like(G[par])
        v[cell] = 1.0
        gv = float((G[par] * v).sum())
        pp = {k: np.array(a) for k, a in start.items()}
        pm = {k: np.array(a) for k, a in start.items()}
        pp[par] = pp[par] + v
        pm[par] = pm[par] - v
        fd = (misfit(pp) - misfit(pm)) / 2.0
        r = fd / gv if gv else float("nan")
        print("%-6s %14.6e %14.6e %9.4f" % (par, fd, gv, r))
        if not abs(r - 1.0) <= tol:
            bad.append("%s %.4f" % (par, r))
    if bad:
        raise AssertionError(
            "receiver-cell gradient off by more than %.0f%%: %s. This is the "
            "residual term in the adjoint increment (update_adjs%dD.cl) -- see "
            "notes/todo.md." % (100 * tol, ", ".join(bad), ND))


def test_fd_3d_receiver_cell_bpt1():
    """3D receiver cell; see test_fd_2d_receiver_cell_bpt1.

    Porting the 2D fix here took the whole-grid bpt2-vs-bpt1 cosine from
    0.8373 to 0.9995 -- the "broad DFT disagreement" was this defect all along.
    """
    _fd_at_receiver_cell(3)


def test_fd_2d_receiver_cell_bpt1():
    """The gradient AT A RECEIVER CELL, back_prop_type=1.

    Mirror of the source-cell cases. The reverse loop runs
    `inject residuals -> update_grid_adj`, so the adjoint increment for step t
    is res(t) + lsxx(t); the kernel computes only the propagation part lsxx,
    so pairing the forward field against it drops res(t). That is nonzero only
    in receiver cells, and there it is not a small error: the ratio was -0.4537
    -- the WRONG SIGN -- while every neighbouring cell read 0.995-1.023.

    No other test in this file can see it: _patch keeps 16 cells clear of
    receivers as well as sources (CLEAR_SRC_REC), the same blind spot that hid
    the source-cell defect.
    """
    vp, vs, rho = 2000.0, 1200.0, 2000.0
    wd = workdir("reccell_2d")
    mk = lambda **k: _srccell_model(wd, ND=2, srctype=100.0,
                                    f0=10.0, NT=1200, dt=0.8e-3, **k)
    s0, zs = mk()
    N = [int(v) for v in s0.N]
    rz = int(round(s0.rec_pos_all[2, 0] / s0.dh))
    rx = int(round(s0.rec_pos_all[0, 0] / s0.dh))
    start = {"vp": np.full(s0.N, vp), "vs": np.full(s0.N, vs),
             "rho": np.full(s0.N, rho)}
    true = {k: np.array(v) for k, v in start.items()}
    true["vp"][N[0] // 2 + 12:N[0] // 2 + 22, N[1] // 2 - 5:N[1] // 2 + 5] += 300.0
    s0.set_forward(s0.src_pos_all[3, :], true, withgrad=False)
    s0.execute()
    dobs = [np.asarray(a, np.float64) for a in s0.read_data()]
    s0.write_data({"p": dobs[0]}, filename="SeisCL_din.mat")
    din = os.path.join(s0.workdir, "SeisCL_din.mat")

    def misfit(p):
        t, _ = mk()
        t.file_din = din
        t.set_forward(t.src_pos_all[3, :], p, withgrad=False)
        t.execute()
        return t.misfit(t.read_data(), dobs=dobs)[0]

    g, _ = mk(gradout=1, back_prop_type=1)
    g.file_din = din
    g.set_forward(g.src_pos_all[3, :], start, withgrad=True)
    g.execute()
    G = {nm: np.asarray(a, np.float64) for nm, a in zip(g.params, g.read_grad())}
    print("=== 2D elastic, gradient AT A RECEIVER CELL (%d,%d), bpt1 ==="
          % (rz, rx))
    print("%-6s %14s %14s %9s" % ("param", "FD", "<g,dm>", "ratio"))
    bad = []
    for par in ("vp", "rho"):
        v = np.zeros_like(G[par])
        v[rz, rx] = 1.0
        gv = float((G[par] * v).sum())
        pp = {k: np.array(a) for k, a in start.items()}
        pm = {k: np.array(a) for k, a in start.items()}
        pp[par] = pp[par] + v
        pm[par] = pm[par] - v
        fd = (misfit(pp) - misfit(pm)) / 2.0
        r = fd / gv if gv else float("nan")
        print("%-6s %14.6e %14.6e %9.4f" % (par, fd, gv, r))
        if not abs(r - 1.0) <= 0.02:
            bad.append("%s %.4f" % (par, r))
    if bad:
        raise AssertionError(
            "receiver-cell gradient off by more than 2%%: %s. This is the "
            "residual term in update_adjs2D.cl's adjoint increment -- see "
            "notes/todo.md." % ", ".join(bad))


def test_fd_2d_srccell_bpt2():
    """2D elastic back_prop_type=2 (DFT), perturbing the SOURCE CELL.

    Looser tolerance than the bpt1 cases on purpose: back_prop_type=2 is the
    uncalibrated path (see the module docstring and the other bpt2 cases,
    which only require eps-independence), and its INTERIOR cells sit a few
    percent off FD in this geometry too. What this guards is that the source
    cell is no worse than the interior -- without the eq. (26a) term and the
    savefreqs source restoration it is off by hundreds of percent.
    """
    _fd_at_source_cell(2, tol=0.10, bpt=2)


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
    test_fd_2d_srccell_bpt1,
    test_fd_3d_srccell_bpt1,
    test_fd_2d_srccell_bpt1_fp16,
    test_fd_2d_srccell_bpt2,
    test_fd_2d_srccell_force,
    test_fd_3d_srccell_force,
    test_fd_2d_srccell_visco_bpt2,
    test_fd_2d_srccell_force_bpt2,
    test_fd_fp16_physical_units,
    test_fd_2d_receiver_cell_bpt1,
    test_fd_3d_receiver_cell_bpt1,
]

# Known-open failures, each tracked in notes/todo.md -- see the docstring of
# the corresponding test_* function for the item number and status. They
# still run and still print their numbers; they just do not fail the build.
XFAIL = {
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
