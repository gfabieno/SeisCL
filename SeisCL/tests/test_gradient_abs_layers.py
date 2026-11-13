"""Finite-difference checks of the gradient inside and near the absorbing layers.

With back_prop_type=1 the forward wavefield is rebuilt backwards in time; the
absorption cannot be undone, so the rebuilt field is wrong in the layers.
gradabs=1 saves the forward field in the whole layers at every time step, and
the CPML (abs_type=1) adjoint filters the adjoint fields before their spatial
derivatives (cpml_adj2D.cl, cpml_adj3D.cl). Theory and numbers:
docs/notebooks/Inversion/AbsorbingLayerGradient.ipynb.

For each case, a random perturbation restricted to one region -- the layers,
the band of FDOH cells inside them, the rest of the model -- and the ratio of
the two-sided finite difference of the misfit to <gradient, perturbation>.

    python test_gradient_abs_layers.py          (SeisCL_MPI from PATH or SEISCL_BIN)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradient_common import workdir, run_tests
from test_gradient_fd import _crosswell_2d, _crosswell_3d

TOL = 0.01          # measured: 0.9964-1.0010 in the layers and the band (notebook, section 5)
TOL_INTERIOR = 0.02  # the interior is covered by test_gradient_fd; here it guards against the old CPML adjoint (0.0000)


def _regions(s, shape):
    nab, fdoh = int(s.nab), int(s.FDORDER) // 2
    idx = np.indices(shape)
    layer = np.zeros(shape, bool)
    inner = np.ones(shape, bool)
    for ax, n in enumerate(shape):
        i = idx[ax]
        if not (ax == 0 and s.freesurf):
            layer |= i < nab
            inner &= i >= nab + fdoh
        layer |= i >= n - nab
        inner &= i < n - nab - fdoh
    return {"layer": layer, "band": ~layer & ~inner, "interior": inner}


def _check(name, make, params=("vp", "vs", "rho"), regions=("layer", "band", "interior"), **over):
    wd = workdir("abslayers-" + name)
    s = make(wd, back_prop_type=1, **over)
    s.file_din = os.path.join(wd, "din.mat")
    shape = tuple(int(v) for v in s.N)
    idx = np.indices(shape)
    cen = [n // 2 for n in shape]
    r2 = sum((idx[a] - cen[a]) ** 2 for a in range(len(shape)))
    init = {"vp": np.full(shape, 3500.0), "vs": np.full(shape, 2000.0), "rho": np.full(shape, 2000.0)}
    true = {k: (v * (1 + 0.05 * np.exp(-r2 / 36.0))).astype(np.float32) for k, v in init.items()}
    s.set_forward([0], true, withgrad=False); s.execute()
    dobs = [np.asarray(d, np.float64) for d in s.read_data()]
    s.write_data(dict(zip(s.to_load_names, dobs)), filename=s.file_din)
    s.set_forward([0], init, withgrad=True); s.execute()
    grads = dict(zip(["vp", "vs", "rho"], [np.asarray(v, np.float64) for v in s.read_grad()]))

    def misfit(p):
        s.set_forward([0], p, withgrad=False); s.execute()
        return s.misfit(s.read_data(), dobs=dobs)[0]

    noise = np.random.default_rng(3).standard_normal(shape)
    masks = _regions(s, shape)
    bad = []
    for k in params:
        assert np.isfinite(grads[k]).all(), "%s: gradient of %s not finite" % (name, k)
        for r in regions:
            dm = noise * masks[r]
            eps = 0.0025 * init[k].mean()
            pp = dict(init); pm = dict(init)
            pp[k] = (init[k] + eps * dm).astype(np.float32); pm[k] = (init[k] - eps * dm).astype(np.float32)
            fd = (misfit(pp) - misfit(pm)) / 2.0
            ad = float((grads[k] * eps * dm).sum())
            ratio = fd / ad if ad else np.nan
            tol = TOL_INTERIOR if r == "interior" else TOL
            print("      %s %s %-8s FD %.4e adjoint %.4e ratio %.4f" % (name, k, r, fd, ad, ratio))
            if not abs(ratio - 1) <= tol:
                bad.append("%s %s ratio %.4f" % (k, r, ratio))
    assert not bad, "%s: %s" % (name, "; ".join(bad))


def test_cerjan_2d_gradabs():
    _check("cerjan2d", _crosswell_2d, abs_type=2, gradabs=1)


def test_cpml_2d_gradabs():
    _check("cpml2d", _crosswell_2d, abs_type=1, gradabs=1)


def test_cpml_2d_interior():
    """Without gradabs: the CPML adjoint itself (the interior was 0.0000 before)."""
    _check("cpml2d-g0", _crosswell_2d, regions=("interior",), abs_type=1, gradabs=0)


def test_band_without_gradabs_2d():
    """Without gradabs, the FDOH band inside the layers (vs, rho were 0.76 / -0.43 to -1.16): the last layer cell is
    now stored too (savebnd2D.cl, FR_OL), for FD orders 4, 8, 12 and both absorbing types."""
    for fdo in (4, 8, 12):
        for abs_type in (2, 1):
            _check("band-f%d-a%d" % (fdo, abs_type), _crosswell_2d, regions=("band", "interior"),
                   abs_type=abs_type, gradabs=0, FDORDER=fdo)


def test_band_without_gradabs_3d():
    for abs_type in (2, 1):
        _check("band3d-a%d" % abs_type, _crosswell_3d, params=("vs", "rho"), regions=("band",),
               abs_type=abs_type, gradabs=0, FDORDER=4)


def test_cerjan_3d_gradabs():
    _check("cerjan3d", _crosswell_3d, params=("vs", "rho"), regions=("layer", "band"), abs_type=2, gradabs=1)


def test_cpml_3d_gradabs():
    _check("cpml3d", _crosswell_3d, params=("vs", "rho"), regions=("layer", "band"), abs_type=1, gradabs=1)


if __name__ == "__main__":
    sys.exit(run_tests([test_cerjan_2d_gradabs, test_cpml_2d_gradabs, test_cpml_2d_interior,
                        test_band_without_gradabs_2d, test_band_without_gradabs_3d,
                        test_cerjan_3d_gradabs, test_cpml_3d_gradabs]) > 0)
