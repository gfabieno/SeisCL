# -*- coding: utf-8 -*-
""" Verification tests for the SeisCL/torch binding (SeisCL/torch/): forward
modeling parity against the plain SeisCL_MPI subprocess/HDF5 workflow, and
gradient correctness via finite differences.

Requires the `torch` install extra and a CUDA-capable GPU
(`pip install -e .[torch]`, see CMakeLists.txt's BUILD_TORCH_CORE option) --
skipped with a message if the compiled extension isn't importable.

Run: python test_torch_binding.py
"""
import numpy as np

from SeisCL.SeisCL import SeisCL

try:
    import torch
    import SeisCL.torch as seiscl_torch
    from SeisCL.torch import (Config, seiscl_forward, clear_engine_cache,
                              engine_cache_size, set_engine_cache_size)
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

N = [80, 80]
DH = 10.0
DT = 0.001
NT = 200
VP, VS, RHO = 2000.0, 1200.0, 2000.0


def _make_config():
    cfg = Config()
    cfg.N = N
    cfg.ND = 2
    cfg.dh = DH
    cfg.dt = DT
    cfg.NT = NT
    cfg.FDORDER = 8
    cfg.FREESURF = 0
    cfg.NAB = 10
    cfg.ABS_TYPE = 2
    cfg.par_type = 0
    cfg.f0 = 25.0
    return cfg


def _homogeneous_params():
    # Parameter dict keys are always the engine's internal names "M"/"mu"/
    # "rho" (assign_modeling_case.c's M()/mu() transform functions), never
    # the physical name implied by par_type -- for par_type=0 the values
    # stored under "M"/"mu" are vp/vs (m/s), squared into Lame parameters
    # in place by Init_model().
    nz, nx = N
    return {
        "M": torch.full((nz * nx,), VP, dtype=torch.float32),
        "mu": torch.full((nz * nx,), VS, dtype=torch.float32),
        "rho": torch.full((nz * nx,), RHO, dtype=torch.float32),
    }


def test_forward_smoke():
    """Forward modeling runs end-to-end and produces finite, nonzero output."""
    cfg = _make_config()
    nx = N[1]
    sx, sz = nx // 2 * DH, N[0] // 2 * DH
    src_pos = torch.tensor([[sx, 0.0, sz, 0.0, 0.0]], dtype=torch.float32)
    nrec = 10
    rec_x = np.linspace(20 * DH, (nx - 20) * DH, nrec)
    rec_pos = torch.zeros((nrec, 8), dtype=torch.float32)
    rec_pos[:, 0] = torch.from_numpy(rec_x).float()
    rec_pos[:, 2] = sz
    t = np.arange(NT) * DT
    f0 = 25.0
    t0 = 1.0 / f0
    ricker = ((1 - 2 * (np.pi * f0 * (t - t0)) ** 2)
              * np.exp(-((np.pi * f0 * (t - t0)) ** 2)))
    src = torch.from_numpy(ricker).float().reshape(1, NT)

    data = seiscl_forward(cfg, _homogeneous_params(), src, src_pos, rec_pos)
    for name, d in data.items():
        assert torch.isfinite(d).all(), f"{name} has non-finite values"
    assert data["vx"].abs().max() > 0, "vx is identically zero"
    print("Testing: torch_forward_smoke ....... passed")


def test_forward_parity():
    """Forward output matches the plain SeisCL_MPI subprocess/HDF5 path."""
    seis = SeisCL()
    seis.ND = 2
    seis.N = np.array(N)
    seis.dh = DH
    seis.dt = DT
    seis.NT = NT
    seis.FDORDER = 8
    seis.freesurf = 0
    seis.nab = 10
    seis.abs_type = 2
    seis.f0 = 25.0
    # seisout=1 -> vx/vz (ND=2); SeisCL.py's default (seisout=2) is pressure
    # ("p"), which would silently compare the wrong field below.
    seis.seisout = 1

    vp = np.full(N, VP, dtype=np.float32)
    vs = np.full(N, VS, dtype=np.float32)
    rho = np.full(N, RHO, dtype=np.float32)
    # src_type=0 (inject directly into vx): surface_acquisition_2d's default
    # (src_type=100) is a "backward compatibility" path
    # (automatic_kernels.c:437) that injects into trans_vars[0] instead.
    seis.surface_acquisition_2d(ds=1000, src_type=0)
    seis.set_forward([0], {"vp": vp, "vs": vs, "rho": rho}, withgrad=False)
    seis.execute()
    vx_mpi = seis.read_data()[0]

    cfg = _make_config()
    src = torch.from_numpy(seis.src_all).float().T.contiguous()
    src_pos = torch.from_numpy(seis.src_pos_all).float().T.contiguous()
    rec_pos = torch.from_numpy(seis.rec_pos_all).float().T.contiguous()
    data = seiscl_forward(cfg, _homogeneous_params(), src, src_pos, rec_pos,
                           output_fields=["vx"])
    vx_torch = data["vx"].numpy().T

    # Traces with a meaningful signal-to-noise ratio should line up with
    # zero lag and near-1 correlation; far-offset traces decay into FP
    # roundoff noise by t=NT and aren't meaningful to compare.
    peak = np.abs(vx_mpi).max(axis=0)
    significant = peak > 1e-3 * peak.max()
    for tr in np.nonzero(significant)[0]:
        a, b = vx_mpi[:, tr], vx_torch[:, tr]
        xcorr = np.correlate(a, b, mode="full")
        lag = xcorr.argmax() - (NT - 1)
        corr = xcorr[xcorr.argmax()] / (np.linalg.norm(a) * np.linalg.norm(b))
        assert lag == 0, f"trace {tr}: nonzero lag {lag}"
        assert corr > 0.98, f"trace {tr}: correlation {corr:.4f} too low"
    print("Testing: torch_forward_parity ....... passed "
          f"({int(significant.sum())} traces checked)")


def test_gradient_finite_difference():
    """Adjoint gradient (backward()) matches a central finite difference."""
    torch.manual_seed(0)
    cfg = _make_config()
    nz, nx = N
    M0 = torch.full((nz * nx,), VP, dtype=torch.float32)
    mu0 = torch.full((nz * nx,), VS, dtype=torch.float32)
    rho = torch.full((nz * nx,), RHO, dtype=torch.float32)

    src_pos = torch.tensor([[400.0, 0.0, 400.0, 0.0, 0.0]], dtype=torch.float32)
    rec_pos = torch.tensor(
        [[420.0, 0.0, 400.0, 0.0, 1.0, 0.0, 0.0, 0.0],
         [440.0, 0.0, 400.0, 0.0, 2.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    t = torch.arange(NT, dtype=torch.float32) * DT
    f0 = 30.0
    t0 = 1.0 / f0
    src = ((1 - 2 * (torch.pi * f0 * (t - t0)) ** 2)
           * torch.exp(-((torch.pi * f0 * (t - t0)) ** 2))).reshape(1, NT)

    def loss(M_, mu_):
        data = seiscl_forward(cfg, {"M": M_, "mu": mu_, "rho": rho},
                               src, src_pos, rec_pos)
        return 0.5 * (data["vx"] ** 2).sum()

    M = M0.clone().requires_grad_(True)
    mu = mu0.clone().requires_grad_(True)
    loss(M, mu).backward()
    grad_M = M.grad

    # Central finite difference on a few entries near the source, where
    # sensitivity is largest and easiest to resolve in float32.
    idx = [39 * nx + 39, 39 * nx + 41, 41 * nx + 39]
    eps = 20.0  # m/s, ~1% perturbation
    for i in idx:
        Mp, Mm = M0.clone(), M0.clone()
        Mp[i] += eps
        Mm[i] -= eps
        with torch.no_grad():
            fd = float((loss(Mp, mu0) - loss(Mm, mu0)) / (2 * eps))
        analytic = float(grad_M[i])
        rel_diff = abs(fd - analytic) / (abs(analytic) + 1e-30)
        assert rel_diff < 0.01, (f"param {i}: analytic={analytic:.6g} "
                                  f"fd={fd:.6g} rel_diff={rel_diff:.4f}")
    print("Testing: torch_gradient_finite_difference ....... passed")


def _simple_geometry(nrec=10):
    """Shared source/receiver/wavelet setup for the engine-reuse tests."""
    nx = N[1]
    sx, sz = nx // 2 * DH, N[0] // 2 * DH
    src_pos = torch.tensor([[sx, 0.0, sz, 0.0, 0.0]], dtype=torch.float32)
    rec_x = np.linspace(20 * DH, (nx - 20) * DH, nrec)
    rec_pos = torch.zeros((nrec, 8), dtype=torch.float32)
    rec_pos[:, 0] = torch.from_numpy(rec_x).float()
    rec_pos[:, 2] = sz
    rec_pos[:, 4] = torch.arange(1, nrec + 1)
    t = np.arange(NT) * DT
    f0 = 25.0
    t0 = 1.0 / f0
    ricker = ((1 - 2 * (np.pi * f0 * (t - t0)) ** 2)
              * np.exp(-((np.pi * f0 * (t - t0)) ** 2)))
    src = torch.from_numpy(ricker).float().reshape(1, NT)
    return src, src_pos, rec_pos


def _forward_with_vp(vp, nrec=10):
    cfg = _make_config()
    src, src_pos, rec_pos = _simple_geometry(nrec)
    nz, nx = N
    params = {
        "M": torch.full((nz * nx,), vp, dtype=torch.float32),
        "mu": torch.full((nz * nx,), VS, dtype=torch.float32),
        "rho": torch.full((nz * nx,), RHO, dtype=torch.float32),
    }
    return seiscl_forward(cfg, params, src, src_pos, rec_pos,
                          output_fields=["vx"])["vx"]


def test_cuda_input_rejected():
    """CUDA tensors are refused, rather than memcpy'd from a device pointer."""
    if not torch.cuda.is_available():
        print("Testing: torch_cuda_input_rejected ....... skipped (no CUDA)")
        return
    cfg = _make_config()
    src, src_pos, rec_pos = _simple_geometry()
    params = _homogeneous_params()
    params["M"] = params["M"].cuda()
    try:
        seiscl_forward(cfg, params, src, src_pos, rec_pos,
                       output_fields=["vx"])
    except (ValueError, RuntimeError) as e:
        assert "CPU tensor" in str(e), f"unexpected error message: {e}"
    else:
        raise AssertionError("a CUDA parameter tensor was silently accepted")
    print("Testing: torch_cuda_input_rejected ....... passed")


def test_cache_hit_matches_fresh_build():
    """A reused engine gives the same answer as a freshly built one.

    Guards the core assumption behind engine reuse: that nothing carries
    over between calls on a shared handle. reduce_seis() accumulates into
    gl_varout with +=, so a missing reset here shows up immediately as a
    doubled seismogram.
    """
    clear_engine_cache()
    a_fresh = _forward_with_vp(2000.0)
    clear_engine_cache()
    b_fresh = _forward_with_vp(2200.0)

    # Same two runs again, now back-to-back so the second reuses the first's
    # engine rather than building its own.
    clear_engine_cache()
    a_reused = _forward_with_vp(2000.0)
    b_reused = _forward_with_vp(2200.0)

    assert not torch.equal(a_fresh, b_fresh), \
        "different vp gave identical output -- the test is not sensitive"
    assert torch.equal(a_fresh, a_reused), "first call changed under reuse"
    assert torch.equal(b_fresh, b_reused), \
        "cache-hit call differs from a fresh build"
    print("Testing: torch_cache_hit_matches_fresh_build ....... passed")


def test_cache_shape_change_and_back():
    """Interleaving a different shape leaves the original still correct."""
    clear_engine_cache()
    ref = _forward_with_vp(2000.0, nrec=10)

    clear_engine_cache()
    a = _forward_with_vp(2000.0, nrec=10)
    other = _forward_with_vp(2000.0, nrec=7)     # different geometry -> new key
    back = _forward_with_vp(2000.0, nrec=10)     # back to the first shape

    assert other.shape[0] == 7, "second shape did not take effect"
    assert torch.equal(ref, a)
    assert torch.equal(ref, back), \
        "returning to an earlier shape gave a different result"
    print("Testing: torch_cache_shape_change_and_back ....... passed")


def test_cache_eviction_correctness():
    """Results stay correct when the cache is too small to hold every shape."""
    clear_engine_cache()
    refs = {n: _forward_with_vp(2000.0, nrec=n) for n in (8, 9, 10)}

    clear_engine_cache()
    set_engine_cache_size(1)
    try:
        for _ in range(2):
            for n in (8, 9, 10):
                got = _forward_with_vp(2000.0, nrec=n)
                assert torch.equal(got, refs[n]), \
                    f"nrec={n} wrong after eviction cycling"
    finally:
        set_engine_cache_size(2)
        clear_engine_cache()
    print("Testing: torch_cache_eviction_correctness ....... passed")


def test_cache_is_actually_used():
    """A repeat call reuses its engine instead of rebuilding."""
    clear_engine_cache()
    assert engine_cache_size() == 0
    _forward_with_vp(2000.0)
    assert engine_cache_size() == 1
    _forward_with_vp(2100.0)
    assert engine_cache_size() == 1, \
        "a same-shape repeat call built a second engine instead of reusing"
    clear_engine_cache()
    assert engine_cache_size() == 0
    print("Testing: torch_cache_is_actually_used ....... passed")


if __name__ == "__main__":
    if not _TORCH_AVAILABLE:
        print("SeisCL.torch not importable (torch extra not installed) "
              "-- skipping SeisCL/torch binding tests")
    else:
        test_forward_smoke()
        test_forward_parity()
        test_gradient_finite_difference()
        test_cuda_input_rejected()
        test_cache_is_actually_used()
        test_cache_hit_matches_fresh_build()
        test_cache_shape_change_and_back()
        test_cache_eviction_correctness()
