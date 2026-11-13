"""Finite-difference checks of the gradient in the top rows under a free surface, and at a density contrast.

* freesurf=1 (stress image): the rows 0..FDOH get their gradient in the direct form (surface_grad in
  surface2D_adj.cl / surface3D_adj.cl), because the adjoint of the surface changes the adjoint fields there and the
  summation-by-parts form of the adjoint kernels fails. Before: ratios -2.6 to 0.9 in rows 0-3.
* freesurf=2 (vacuum): the density gradient of the first row under the vacuum missed the term of the vertical
  velocity above it (the averaging transpose dropped a vacuum neighbour). Before: 1.29 (3.34 with pressure data).
* Density contrast: the averaging transpose used the buoyancy Jacobian (rho_j/rho_avg)^2/2 instead of 1/2.
  Before: 1.021 at the interface of a 1500/2500 contrast.

Source and receivers are put just under the surface (the usual acquisition), so the top rows matter. For each row,
a random perturbation of that row only (away from the absorbing layers), and the ratio of the two-sided finite
difference of the misfit to <gradient, perturbation>. FD step: 0.125 % of the mean of the parameter.

    python test_gradient_surface.py          (SeisCL_MPI from PATH or SEISCL_BIN)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradient_common import workdir, run_tests
from test_gradient_fd import _crosswell_2d, _crosswell_3d

TOL = 0.015      # on |FD - adjoint| / max|FD| over the rows (see _check)


def _check(name, make, rows, params=("vp", "vs", "rho"), sz=None, gz=None, src=2.0, rhojump=None, **over):
    wd = workdir("surface-" + name)
    s = make(wd, back_prop_type=1, **over)
    s.file_din = os.path.join(wd, "din.mat")
    shape = tuple(int(v) for v in s.N)
    nab, fdoh = int(s.nab), int(s.FDORDER) // 2
    if sz is not None:                      # surface acquisition: source at depth sz, receiver line at depth gz
        s.src_pos_all[2, 0] = sz * s.dh
        s.src_pos_all[4, 0] = src
        nx = shape[-1]
        gx = np.arange(nab + 10, nx - nab - 2, 2 if len(shape) == 2 else 3) * s.dh
        y0 = s.src_pos_all[1, 0]
        s.rec_pos_all = np.stack([gx, 0 * gx + y0, 0 * gx + gz * s.dh, 0 * gx, np.arange(1, len(gx) + 1),
                                  0 * gx, 0 * gx, 0 * gx])
    init = {"vp": np.full(shape, 3500.0), "vs": np.full(shape, 2000.0), "rho": np.full(shape, 2000.0)}
    if rhojump is not None:
        init["rho"][: shape[0] // 2] = 1500.0
        init["rho"][shape[0] // 2:] = rhojump
    if s.freesurf == 2:
        for k in init:
            init[k][:fdoh] = 0.0
    idx = np.indices(shape)
    cen = [n // 2 for n in shape]
    r2 = sum((idx[a] - cen[a]) ** 2 for a in range(len(shape)))
    true = {k: (v * (1 + 0.05 * np.exp(-r2 / 36.0))).astype(np.float32) for k, v in init.items()}
    s.set_forward([0], true, withgrad=False); s.execute()
    dobs = [np.asarray(d, np.float64) for d in s.read_data()]
    s.write_data(dict(zip(s.to_load_names, dobs)), filename=s.file_din)
    s.set_forward([0], init, withgrad=True); s.execute()
    grads = dict(zip(["vp", "vs", "rho"], [np.asarray(v, np.float64) for v in s.read_grad()]))

    def misfit(p):
        s.set_forward([0], p, withgrad=False); s.execute()
        return s.misfit(s.read_data(), dobs=dobs)[0]

    cols = np.ones(shape, bool)
    for ax in range(1, len(shape)):
        cols &= (idx[ax] >= nab + fdoh) & (idx[ax] < shape[ax] - nab - fdoh)
    noise = np.random.default_rng(3).standard_normal(shape)
    bad = []
    for k in params:
        assert np.isfinite(grads[k]).all(), "%s: gradient of %s not finite" % (name, k)
        res = []
        for z in rows:
            dm = noise * cols * (idx[0] == z)
            eps = 0.5 * 0.0025 * init[k][init[k] > 0].mean()
            pp = dict(init); pm = dict(init)
            pp[k] = (init[k] + eps * dm).astype(np.float32); pm[k] = (init[k] - eps * dm).astype(np.float32)
            fd = (misfit(pp) - misfit(pm)) / 2.0
            ad = float((grads[k] * eps * dm).sum())
            res.append((z, fd, ad))
        # A row whose random perturbation nearly cancels has a tiny FD signal, and its ratio is FD noise (row 3
        # of the FD-order-4 case: 0.64 with this seed, 0.98-1.001 with three others). So each row's error is
        # judged against the largest FD signal of this parameter in the test, as check.py judges "all".
        scale = max(abs(fd) for _, fd, _ in res)
        for z, fd, ad in res:
            ratio = fd / ad if ad else np.nan
            err = abs(fd - ad) / scale
            print("      %s %s row %2d  FD %.4e adjoint %.4e ratio %.4f  error/max|FD| %.4f"
                  % (name, k, z, fd, ad, ratio, err))
            if not err <= TOL:
                bad.append("%s row %d ratio %.4f, error/max|FD| %.4f" % (k, z, ratio, err))
    assert not bad, "%s: %s" % (name, "; ".join(bad))


def test_fs1_2d_surface_acquisition():
    _check("fs1-2d", _crosswell_2d, rows=range(0, 6), sz=10, gz=1, freesurf=1)


def test_fs1_2d_surface_fdorders():
    """The direct-form rows are 0..FDORDER/2: FD orders 4 and 12."""
    for fdo in (4, 12):
        _check("fs1-2d-f%d" % fdo, _crosswell_2d, rows=range(0, fdo // 2 + 2), sz=10, gz=1, freesurf=1, FDORDER=fdo)


def test_fs1_3d_surface_acquisition():
    _check("fs1-3d", _crosswell_3d, rows=range(0, 5), sz=8, gz=1, freesurf=1)


def test_fs2_2d_first_row_density():
    _check("fs2-2d", _crosswell_2d, rows=range(4, 8), params=("rho",), sz=6, gz=5, freesurf=2)


def test_density_contrast_2d():
    _check("rhojump-2d", _crosswell_2d, rows=range(30, 34), params=("rho",), rhojump=2500.0, src=100.0)


if __name__ == "__main__":
    sys.exit(run_tests([test_fs1_2d_surface_acquisition, test_fs1_2d_surface_fdorders, test_fs1_3d_surface_acquisition,
                        test_fs2_2d_first_row_density, test_density_contrast_2d]) > 0)
