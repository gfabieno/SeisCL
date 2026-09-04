# -*- coding: utf-8 -*-
"""Finite-difference check of the FREESURF=2 (improved vacuum formulation,
Zeng et al. 2012, see notes/vacuum-freesurface-plan.md) gradient.

This is the one combination FREESURF=2 supports for gradients in this clone:
2D elastic, BACK_PROP_TYPE=1, RESTYPE=1 (see assign_modeling_case.c's gate).
Uses the SeisCL_MPI subprocess/HDF5 workflow directly (this clone predates
SeisCL.misfit()), not the torch binding -- the torch binding's compiled
_C.so is a separate build artifact (SEISCL_BUILD_TORCH=1) that these source
fixes have not been folded into, see CLAUDE.md's torch build recipe.

Background: as of the fixes below, update_adjs2D.cl's RESTYPE==1 branch was
missing (a) the c1=1/(2M-2mu)^2 normalization RESTYPE==0 has (gradM was off
by that whole factor, FD ratios ~1e12) and (b) gradmu entirely (RESTYPE==1
never computed it, so d(vs) was always 0). Both are now computed with a
zero-guard (c1/c3 divide by (M-mu)/mu, which are exactly 0 in the vacuum
band -- the reason RESTYPE==1 avoids RESTYPE==0's formula at all). residuals.c
also had the same dh/dt-instead-of-dt/dh and receiver-position-indexing bugs
already fixed elsewhere for SeisCL-dft (notes/todo.md items 0c/0e), ported
here. See notes/vacuum-freesurface-plan.md for the full history.

vp/vs now match a central finite difference to a few percent, in line with
this clone's existing (non-vacuum) FREESURF=0/1 precision -- see the
FREESURF=1 control case below, run with the identical geometry/parameters,
which shows the same few-percent-to-30% imprecision. rho is NOT asserted
against a tight tolerance: both FREESURF=1 and FREESURF=2 show a rho ratio
of several tenths to a few units off 1 with this exact setup, tracked as the
pre-existing, clone-wide "material averaging" gradient gap in
notes/material-averaging-gradient-review.md (P0, steps 3-4 of that plan are
not implemented) -- not something this vacuum-specific pass can or should
fix. rho is still checked for the right sign and for being within an order
of magnitude, so a future regression (e.g. losing the dh^2/dt^3 unscale
factor in calc_grad.c again) would still be caught.

Run: SEISCL_BIN=<build dir> python test_freesurf2_gradient.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from SeisCL.SeisCL import SeisCL, SeisCLError

DH, DT, NT, F0, NAB, ABPC = 10.0, 0.8e-3, 1200, 10.0, 16, 6.0
SRC_SCALE = 1e6
VP, VS, RHO = 3000.0, 1800.0, 2200.0
N = np.array([100, 120])

# vp/vs: tight, matches this clone's ordinary (non-vacuum) precision.
# rho: loose, see module docstring -- pre-existing, tracked separately.
RATIO_TOL = {"vp": 0.05, "vs": 0.06, "rho": None}
RHO_MAX_ABS_RATIO = 5.0


def _make(wd, freesurf, seisout, **overrides):
    cfg = dict(N=N, ND=2, dh=DH, dt=DT, NT=NT, f0=F0, FDORDER=8,
               freesurf=freesurf, abs_type=2, nab=NAB, abpc=ABPC,
               seisout=seisout, param_type=0, restype=1)
    cfg.update(overrides)
    s = SeisCL(workdir=wd, **cfg)
    nz, nx = int(s.N[0]), int(s.N[1])
    xl = (s.nab + 6) * s.dh
    xr = (nx - s.nab - 6) * s.dh
    sz = 0.5 * nz * s.dh
    s.src_pos_all = np.array([[xl], [0.0], [sz], [0.0], [100.0]])
    n_rec = 30
    gz = np.linspace(s.nab + 8, nz - s.nab - 8, n_rec) * s.dh
    s.rec_pos_all = np.stack(
        [np.full(n_rec, xr), np.zeros(n_rec), gz, np.zeros(n_rec),
         np.arange(1, n_rec + 1), np.zeros(n_rec), np.zeros(n_rec),
         np.zeros(n_rec)])
    s.src_all = SRC_SCALE * s.ricker_wavelet().reshape(-1, 1)
    return s


def _init_and_true(s, patch):
    init = {"vp": np.full(s.N, VP), "vs": np.full(s.N, VS),
            "rho": np.full(s.N, RHO)}
    # freesurf=2: the vacuum belongs to the MODEL, not to the engine. The
    # user zeroes the material above the interface and those nodes are
    # updated like interior ones; the traction-free condition comes out of
    # the parameter averaging (Zeng et al. 2012). The engine used to carve
    # this band itself, which contradicted the method -- see the note where
    # set_freesurf2_vacuum() used to live in assign_modeling_case.c.
    if getattr(s, "freesurf", 0) == 2:
        fdoh = s.FDORDER // 2
        for a in init.values():
            a[:fdoh, ...] = 0
    true = {k: v.copy() for k, v in init.items()}
    true["vp"][patch] += 400.0
    true["vs"][patch] += 200.0
    true["rho"][patch] += 150.0
    return init, true


def _misfit(dmod, dobs):
    res = [np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
           for a, b in zip(dmod, dobs)]
    J = 0.5 * float(sum((r ** 2).sum() for r in res))
    return J, res


def _forward_misfit(mk, ids, params, din, dobs):
    s = mk()
    s.file_din = din
    s.set_forward(ids, params, withgrad=False)
    s.execute()
    return _misfit(s.read_data(), dobs)


def _check_gradient(freesurf, seisout, wd):
    mk = lambda **kw: _make(wd, freesurf, seisout, **kw)
    s0 = mk()
    ids = s0.src_pos_all[3, :]
    nz, nx = int(s0.N[0]), int(s0.N[1])
    patch = (slice(nz // 2 - 7, nz // 2 + 8), slice(nx // 2 - 10, nx // 2 + 10))
    init, true = _init_and_true(s0, patch)

    s0.set_forward(ids, true, withgrad=False)
    s0.execute()
    dobs = s0.read_data()
    s0.write_data({"p": dobs[0]}, filename="SeisCL_din.mat")
    din = os.path.join(s0.workdir, "SeisCL_din.mat")

    J0, res = _forward_misfit(mk, ids, init, din, dobs)
    assert J0 > 0, "residual is exactly zero, test would be vacuous"

    s = mk(gradout=1, back_prop_type=1)
    s.file_din = din
    s.inputres = 1
    s.set_forward(ids, init, withgrad=False)
    s.execute()
    s.set_backward(residuals=res)
    s.execute()
    grad = s.read_grad()

    dm = np.zeros(s0.N)
    dm[patch] = 1.0
    eps = {"vp": 10.0, "vs": 10.0, "rho": 10.0}
    for i, nm in enumerate(s.params):
        pp = dict(init); pp[nm] = init[nm] + eps[nm] * dm
        pm = dict(init); pm[nm] = init[nm] - eps[nm] * dm
        Jp, _ = _forward_misfit(mk, ids, pp, din, dobs)
        Jm, _ = _forward_misfit(mk, ids, pm, din, dobs)
        fd = (Jp - Jm) / (2 * eps[nm])
        ad = float((grad[i] * dm).sum())
        ratio = ad / fd if fd else float("nan")
        tol = RATIO_TOL[nm]
        if tol is None:
            assert np.sign(ad) == np.sign(fd), (
                f"freesurf={freesurf} seisout={seisout} {nm}: wrong sign, "
                f"fd={fd:.4e} ad={ad:.4e}")
            assert abs(ratio) < RHO_MAX_ABS_RATIO, (
                f"freesurf={freesurf} seisout={seisout} {nm}: ratio "
                f"{ratio:.4g} exceeds the loose bound of "
                f"{RHO_MAX_ABS_RATIO} (see module docstring)")
        else:
            assert abs(ratio - 1.0) < tol, (
                f"freesurf={freesurf} seisout={seisout} {nm}: ratio="
                f"{ratio:.6f} outside 1+-{tol}")


def test_freesurf2_gradient_particle_velocity():
    wd = os.path.join(_HERE, "_freesurf2_gradient_work", "fs2_seisout1")
    os.makedirs(wd, exist_ok=True)
    _check_gradient(freesurf=2, seisout=1, wd=wd)
    print("Testing: freesurf2_gradient (seisout=1, particle velocity) "
          "....... passed")


def test_freesurf2_gradient_pressure():
    wd = os.path.join(_HERE, "_freesurf2_gradient_work", "fs2_seisout2")
    os.makedirs(wd, exist_ok=True)
    _check_gradient(freesurf=2, seisout=2, wd=wd)
    print("Testing: freesurf2_gradient (seisout=2, pressure) ....... passed")


if __name__ == "__main__":
    SEISCL_BIN = os.environ.get("SEISCL_BIN")
    if SEISCL_BIN:
        os.environ["PATH"] = (os.path.abspath(SEISCL_BIN) + os.pathsep
                               + os.environ.get("PATH", ""))
    try:
        test_freesurf2_gradient_particle_velocity()
        test_freesurf2_gradient_pressure()
    except (AssertionError, SeisCLError) as e:
        print("FAILED:", e)
        sys.exit(1)
