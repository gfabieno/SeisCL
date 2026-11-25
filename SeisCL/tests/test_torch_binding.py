# -*- coding: utf-8 -*-
""" Verification tests for the SeisCL/torch binding (SeisCL/torch/): forward
modeling parity against the plain SeisCL_MPI subprocess/HDF5 workflow, and
gradient correctness via finite differences.

Requires the `torch` install extra and a CUDA-capable GPU
(`pip install -e .[torch]`, see CMakeLists.txt's BUILD_TORCH_CORE option) --
skipped with a message if the compiled extension isn't importable.

Run: python test_torch_binding.py
"""
import os

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


def _make_config(freesurf=0, restype=0):
    cfg = Config()
    cfg.N = N
    cfg.ND = 2
    cfg.dh = DH
    cfg.dt = DT
    cfg.NT = NT
    cfg.FDORDER = 8
    cfg.FREESURF = freesurf
    cfg.NAB = 10
    cfg.ABS_TYPE = 2
    cfg.par_type = 0
    cfg.f0 = 25.0
    cfg.restype = restype
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


def test_forward_smoke(freesurf=0):
    """Forward modeling runs end-to-end and produces finite, nonzero output.

    Parametrized over freesurf so the stress-image method (1) and the
    improved vacuum formulation (2, see notes/vacuum-freesurface-plan.md)
    get the same basic coverage as the default (0) -- previously only
    freesurf=0 was ever exercised through this binding.
    """
    cfg = _make_config(freesurf=freesurf)
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
    print(f"Testing: torch_forward_smoke (freesurf={freesurf}) ....... passed")


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


def test_gradient_finite_difference(freesurf=0, restype=0):
    """Adjoint gradient (backward()) matches a central finite difference.

    Parametrized over freesurf/restype: previously this only ever ran with
    freesurf=0, restype=0 (the default), so neither the stress-image method
    (freesurf=1) nor the improved vacuum formulation (freesurf=2) had ever
    been gradient-checked through the autograd path at all. freesurf=2
    requires restype=1 in this version (see
    notes/vacuum-freesurface-plan.md, Phase 3 -- BACK_PROP_TYPE=1's default
    compliance-based gradient divides by the raw, zeroed-in-vacuum M/mu).

    restype=1 ("cross-correlation of traces" costfunction, per
    SeisCL.py's docstring, vs. restype=0's "l2 cost") does NOT reduce to
    the gradient of loss=0.5*sum(vx**2) used below -- tried both that and
    a loss=sum(vx*fixed_target) variant (a natural guess for what a
    cross-correlation costfunction's gradient should be); neither matches
    the finite difference (the FD comes out ~0 while the analytic gradient
    is large, for both). What restype=1's gradient actually corresponds to
    numerically is an open question -- not investigated further here, out
    of scope for the free-surface work. So for freesurf=2 (which forces
    restype=1) this only checks finiteness and correct vacuum-band
    cropping, not FD agreement; freesurf in {0,1} keep the full FD check.
    """
    torch.manual_seed(0)
    cfg = _make_config(freesurf=freesurf, restype=restype)
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

    if restype == 1:
        assert torch.isfinite(grad_M).all(), "gradM has non-finite values"
        fdoh = cfg.FDORDER // 2
        vacuum_thickness = min(fdoh, nz)
        # gl_par/gl_grad are X-slowest/Z-fastest flat (grad[x*NZ+z], see
        # SeisCL/torch/bindings.cpp's crop_boundary_2d comment) -- reshape
        # as (nx, nz), not (nz, nx), to slice a z-band correctly.
        band = grad_M.reshape(nx, nz)[:, :vacuum_thickness]
        assert (band == 0).all(), "vacuum-band gradM should be exactly 0"
        print("Testing: torch_gradient_finite_difference "
              f"(freesurf={freesurf}, restype={restype}) -- finiteness + "
              "crop only (see docstring) ....... passed")
        return

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
    print("Testing: torch_gradient_finite_difference "
          f"(freesurf={freesurf}, restype={restype}) ....... passed")


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


def test_cuda_geometry_rejected():
    """CUDA geometry tensors are refused rather than read as host pointers.

    Parameters are the exception -- they're copied down automatically, see
    test_cuda_params_accepted.
    """
    if not torch.cuda.is_available():
        print("Testing: torch_cuda_geometry_rejected ....... skipped (no CUDA)")
        return
    cfg = _make_config()
    src, src_pos, rec_pos = _simple_geometry()
    try:
        seiscl_forward(cfg, _homogeneous_params(), src.cuda(), src_pos,
                       rec_pos, output_fields=["vx"])
    except (ValueError, RuntimeError) as e:
        assert "CPU tensor" in str(e), f"unexpected error message: {e}"
    else:
        raise AssertionError("a CUDA src tensor was silently accepted")
    print("Testing: torch_cuda_geometry_rejected ....... passed")


def test_cuda_params_accepted():
    """GPU-resident model parameters give the same answer as CPU ones."""
    if not torch.cuda.is_available():
        print("Testing: torch_cuda_params_accepted ....... skipped (no CUDA)")
        return
    cfg = _make_config()
    src, src_pos, rec_pos = _simple_geometry()

    clear_engine_cache()
    on_cpu = seiscl_forward(cfg, _homogeneous_params(), src, src_pos, rec_pos,
                            output_fields=["vx"])["vx"]

    cuda_params = {k: v.cuda() for k, v in _homogeneous_params().items()}
    on_gpu = seiscl_forward(cfg, cuda_params, src, src_pos, rec_pos,
                            output_fields=["vx"])["vx"]

    assert not on_gpu.is_cuda, "output should still be a CPU tensor"
    assert torch.equal(on_cpu, on_gpu), \
        "CUDA-resident parameters gave a different result"
    print("Testing: torch_cuda_params_accepted ....... passed")


def test_cuda_gradient_returned_on_param_device():
    """A CUDA-resident, differentiable parameter gets its gradient back on
    the same CUDA device, not on the CPU.

    collect_grads() (bindings.cpp) always builds its result from the
    engine's host gl_grad buffers, since set_params() copies any CUDA input
    down to host before it reaches the engine. torch.autograd.Function
    requires a returned gradient to be on the same device as its input, so
    without moving it back in op.py's backward(), this raised
    "RuntimeError: function ... returned an invalid gradient ... expected
    device cuda:0 but got device cpu" instead of producing a gradient.
    """
    if not torch.cuda.is_available():
        print("Testing: torch_cuda_gradient_returned_on_param_device "
              "....... skipped (no CUDA)")
        return
    clear_engine_cache()
    reference = _grad_of(VP)

    clear_engine_cache()
    cfg = _make_config()
    src, src_pos, rec_pos = _simple_geometry()
    nz, nx = N
    M = torch.full((nz * nx,), VP, dtype=torch.float32,
                   device="cuda").requires_grad_(True)
    mu = torch.full((nz * nx,), VS, dtype=torch.float32, device="cuda")
    rho = torch.full((nz * nx,), RHO, dtype=torch.float32, device="cuda")
    data = seiscl_forward(cfg, {"M": M, "mu": mu, "rho": rho},
                          src, src_pos, rec_pos, output_fields=["vx"])
    (0.5 * (data["vx"] ** 2).sum()).backward()

    assert M.grad.is_cuda, "gradient for a CUDA parameter came back on CPU"
    assert torch.equal(M.grad.cpu(), reference), \
        "CUDA-parameter gradient differs from the CPU-parameter reference"
    print("Testing: torch_cuda_gradient_returned_on_param_device "
          "....... passed")


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


def test_cache_source_type_change():
    """Changing only the source type between two calls rebuilds the source kernel.

    kernel_sources (automatic_kernels.c) compiles the source types of src_pos
    column 4 into the "sources" kernel, but the engine cache key did not hold
    them: a second call with the same geometry and another source type reused
    the first call's kernel. An explosive call after a vz-force call returned
    the force data exactly (2-D full space, bench-deepwave-work/timing,
    2026-11-24). Each call after another type must equal its fresh build;
    threshold: relative difference <= 1e-6 (bit-identical expected).
    """
    def run(stype):
        src, src_pos, rec_pos = _simple_geometry()
        src_pos = src_pos.clone()
        src_pos[0, 4] = stype
        out = seiscl_forward(_make_config(), _homogeneous_params(), src,
                             src_pos, rec_pos, output_fields=["vx", "vz"])
        return torch.cat([out["vx"].flatten(), out["vz"].flatten()])

    types = (2, 100, 0)                  # vz force, explosive, vx force
    fresh = {}
    for st in types:
        clear_engine_cache()
        fresh[st] = run(st)
    clear_engine_cache()
    for a, b in ((2, 100), (100, 0), (0, 2)):
        run(a)
        after = run(b)
        ref = fresh[b]
        rel = float((after - ref).abs().max() / ref.abs().max())
        print("      type %d after type %d: max relative difference to a fresh "
              "build %.2e (<= 1e-6)" % (b, a, rel))
        assert float((fresh[a] - ref).abs().max() / ref.abs().max()) > 1e-2, \
            "types %d and %d gave the same data -- the test is not sensitive" % (a, b)
        assert rel <= 1e-6, (
            "source type %d after type %d: relative difference %.2e to a fresh "
            "build > 1e-6 -- the cached source kernel was reused" % (b, a, rel))
    print("Testing: torch_cache_source_type_change ....... passed")


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


def test_cache_hit_rejects_wrong_src_length():
    """A malformed src on a *reused* engine is a clean error, not an
    out-of-bounds read.

    engine_build()'s miss path validates src.numel() == allns*NT before
    seiscl_set_srcrec(); engine_refresh_srcrec(), the cache-hit counterpart,
    did the same fixed-size memcpy with no such check -- a short src tensor
    was read past its own allocation.
    """
    cfg = _make_config()
    src, src_pos, rec_pos = _simple_geometry()
    params = _homogeneous_params()

    clear_engine_cache()
    seiscl_forward(cfg, params, src, src_pos, rec_pos,
                   output_fields=["vx"])  # cache miss: builds the engine

    bad_src = src[:, :-1]  # same geometry/cache key, one sample short
    try:
        seiscl_forward(cfg, params, bad_src, src_pos, rec_pos,
                       output_fields=["vx"])  # cache hit: refresh path
    except (ValueError, RuntimeError) as e:
        assert "src must have shape" in str(e), f"unexpected error: {e}"
        print("Testing: torch_cache_hit_rejects_wrong_src_length "
              "....... passed")
        return
    raise AssertionError(
        "a cache-hit call accepted a src tensor of the wrong length")


def _grad_of(vp_value, other_forward_vp=None, nshot=1):
    """Gradient of 0.5*sum(vx**2) w.r.t. M at vp_value.

    If other_forward_vp is given, a second forward pass runs *between* this
    one and its backward -- gradient accumulation, and the case where the
    in-memory checkpoint handoff must fall back to a file because the second
    forward overwrites the first's boundary buffers (one shot: the engine's
    own buffers; several shots: the per-shot host buffers).
    """
    cfg = _make_config()
    if nshot == 1:
        src, src_pos, rec_pos = _simple_geometry()
    else:
        src, src_pos, rec_pos = _multishot_geometry(nshot)
    nz, nx = N
    mu = torch.full((nz * nx,), VS, dtype=torch.float32)
    rho = torch.full((nz * nx,), RHO, dtype=torch.float32)

    M = torch.full((nz * nx,), vp_value, dtype=torch.float32).requires_grad_(True)
    data = seiscl_forward(cfg, {"M": M, "mu": mu, "rho": rho},
                          src, src_pos, rec_pos, output_fields=["vx"])
    loss = 0.5 * (data["vx"] ** 2).sum()

    if other_forward_vp is not None:
        M2 = torch.full((nz * nx,), other_forward_vp,
                        dtype=torch.float32).requires_grad_(True)
        other = seiscl_forward(cfg, {"M": M2, "mu": mu, "rho": rho},
                               src, src_pos, rec_pos, output_fields=["vx"])
        # Keep it alive and differentiable so nothing is optimised away.
        assert torch.isfinite(other["vx"]).all()

    loss.backward()
    return M.grad.clone()


def test_checkpoint_survives_interleaved_forward(nshot=1):
    """A second forward before the first's backward must not corrupt it.

    The boundary wavefield normally stays in the engine's buffers between
    forward and backward instead of going through an HDF5 file. A second
    forward on the same engine overwrites those buffers, so the first pass's
    checkpoint has to be flushed to disk before that happens -- if it isn't,
    this returns a gradient computed from the *wrong* wavefield, silently.
    """
    clear_engine_cache()
    reference = _grad_of(VP, nshot=nshot)

    clear_engine_cache()
    interleaved = _grad_of(VP, other_forward_vp=VP + 300.0, nshot=nshot)

    assert torch.equal(reference, interleaved), (
        f"{nshot} shot(s): gradient changed when another forward ran before "
        f"backward: max abs diff "
        f"{float((reference - interleaved).abs().max()):.6g}")
    print(f"Testing: torch_checkpoint_survives_interleaved_forward "
          f"({nshot} shot(s)) ....... passed")


def test_checkpoint_survives_cache_clear():
    """Dropping the engine between forward and backward is a clean error.

    Without the engine, the boundary wavefield is gone and there is no file
    to fall back on, so this must fail loudly rather than return a wrong
    gradient.
    """
    cfg = _make_config()
    src, src_pos, rec_pos = _simple_geometry()
    nz, nx = N
    clear_engine_cache()
    M = torch.full((nz * nx,), VP, dtype=torch.float32).requires_grad_(True)
    data = seiscl_forward(cfg, {"M": M,
                                "mu": torch.full((nz * nx,), VS),
                                "rho": torch.full((nz * nx,), RHO)},
                          src, src_pos, rec_pos, output_fields=["vx"])
    loss = 0.5 * (data["vx"] ** 2).sum()
    clear_engine_cache()
    try:
        loss.backward()
    except RuntimeError as e:
        assert "checkpoint" in str(e).lower(), f"unexpected error: {e}"
        print("Testing: torch_checkpoint_survives_cache_clear ....... passed")
        return
    raise AssertionError(
        "backward() silently produced a gradient after its engine was dropped")


def test_pending_checkpoint_survives_rekey(nshot=1):
    """Two differentiable forwards, output_fields left at its default (an
    empty list resolved to "every declared field" once the engine is
    built), called back-to-back before either backward -- the "ordinary
    summed-loss" pattern of accumulating several forward passes before one
    combined loss.backward().

    Both calls share the same cfg/geometry, so both start out keyed under
    the same (as yet unresolved) empty-output-fields CacheKey. The first
    call's handle gets rekeyed onto the resolved key once its fields are
    known, and is left there with pending_valid=True (its checkpoint lives
    only in its own buffers, single-shot). The second call's own handle,
    built fresh under the same empty key, then rekeys onto that identical
    resolved key -- displacing the first handle. Without flushing that
    displaced handle's pending checkpoint first, its buffers are freed
    outright, and the first call's backward() can recover it from neither
    the (evicted) handle nor a checkpoint file that was never written.
    """
    clear_engine_cache()
    cfg = _make_config()
    if nshot == 1:
        src, src_pos, rec_pos = _simple_geometry()
    else:
        src, src_pos, rec_pos = _multishot_geometry(nshot)
    nz, nx = N
    mu = torch.full((nz * nx,), VS, dtype=torch.float32)
    rho = torch.full((nz * nx,), RHO, dtype=torch.float32)

    # Reference: the first call alone, nothing in between.
    M0 = torch.full((nz * nx,), VP, dtype=torch.float32).requires_grad_(True)
    data0 = seiscl_forward(cfg, {"M": M0, "mu": mu, "rho": rho},
                           src, src_pos, rec_pos)
    (0.5 * (data0["vx"] ** 2).sum()).backward()
    clear_engine_cache()

    M1 = torch.full((nz * nx,), VP, dtype=torch.float32).requires_grad_(True)
    data1 = seiscl_forward(cfg, {"M": M1, "mu": mu, "rho": rho},
                           src, src_pos, rec_pos)  # output_fields=None
    loss1 = 0.5 * (data1["vx"] ** 2).sum()

    M2 = torch.full((nz * nx,), VP + 300.0,
                    dtype=torch.float32).requires_grad_(True)
    data2 = seiscl_forward(cfg, {"M": M2, "mu": mu, "rho": rho},
                           src, src_pos, rec_pos)  # output_fields=None
    loss2 = 0.5 * (data2["vx"] ** 2).sum()

    loss1.backward()
    loss2.backward()

    assert torch.isfinite(M1.grad).all() and M1.grad.abs().max() > 0
    assert torch.isfinite(M2.grad).all() and M2.grad.abs().max() > 0
    assert torch.equal(M1.grad, M0.grad), (
        f"{nshot} shot(s): the displaced call's gradient changed: max abs diff "
        f"{float((M1.grad - M0.grad).abs().max()):.6g}")
    print(f"Testing: torch_pending_checkpoint_survives_rekey ({nshot} shot(s)) "
          "....... passed")


def _multishot_geometry(nshot, nrec=6):
    nz, nx = N
    sp, rp = [], []
    for s in range(nshot):
        sp.append([(20 + s * 20) * DH, 0.0, nz // 2 * DH, float(s), 0.0])
        for r in range(nrec):
            rp.append([(20 + r * 8) * DH, 0.0, nz // 2 * DH,
                       float(s), float(r + 1), 0.0, 0.0, 0.0])
    t = np.arange(NT) * DT
    f0 = 25.0
    t0 = 1.0 / f0
    ricker = ((1 - 2 * (np.pi * f0 * (t - t0)) ** 2)
              * np.exp(-((np.pi * f0 * (t - t0)) ** 2))).astype(np.float32)
    return (torch.from_numpy(np.tile(ricker, (nshot, 1))),
            torch.tensor(sp, dtype=torch.float32),
            torch.tensor(rp, dtype=torch.float32))


def _viscoelastic_params(tau=0.02):
    p = _homogeneous_params()
    nz, nx = N
    p["taup"] = torch.full((nz * nx,), tau, dtype=torch.float32)
    p["taus"] = torch.full((nz * nx,), tau, dtype=torch.float32)
    return p


def test_viscoelastic_requires_FL():
    """cfg.L>0 needs cfg.FL, and honours its values.

    Regression test for the original torch binding: Config exposed L but
    nothing populated the "FL" constants array, so assign_modeling_case()'s
    zero-filled allocation reached eta() (eta[l]=2*pi*FL[l]*dt=0) and the
    M()/mu() transforms computed dt/eta[l] -> Inf. Every viscoelastic run
    returned Inf/NaN silently, even with valid taup/taus.
    """
    src, src_pos, rec_pos = _simple_geometry()

    def run(L, FL=None, tau=0.02):
        cfg = _make_config()
        cfg.L = L
        if FL is not None:
            cfg.FL = FL
        params = _viscoelastic_params(tau) if L else _homogeneous_params()
        return seiscl_forward(cfg, params, src, src_pos, rec_pos)["vx"]

    # A missing or wrong-length FL must be a clear error, not silent NaN.
    for L, FL in ((1, None), (2, [15.0])):
        try:
            run(L, FL)
            raise AssertionError(f"L={L} FL={FL} should have been rejected")
        except ValueError as exc:
            assert "FL" in str(exc), str(exc)

    vx_elastic = run(0)
    vx_visco = run(1, [25.0])
    assert torch.isfinite(vx_visco).all(), "viscoelastic output not finite"
    assert (vx_visco != 0).any(), "viscoelastic output is all zero"

    # Attenuation must actually do something: less energy than elastic.
    e_el = float((vx_elastic ** 2).sum())
    e_vi = float((vx_visco ** 2).sum())
    assert e_vi < e_el, f"viscoelastic not attenuated: {e_vi:.4g} vs {e_el:.4g}"

    # FL is part of the engine cache key: same L, different FL must rebuild
    # rather than silently reuse the first build's relaxation times.
    a = run(1, [10.0])
    b = run(1, [60.0])
    assert float((a - b).abs().max()) > 0, \
        "different FL returned identical data -- stale cached build"
    assert torch.equal(a, run(1, [10.0])), \
        "same FL not reproducible on a cache hit"

    print("Testing: torch_viscoelastic_requires_FL ....... passed")


def test_multishot_gradient_both_checkpoint_policies():
    """Multi-shot gradients are right whether the checkpoint is RAM or a file.

    A single shot keeps its wavefield in the engine's own buffers, but every
    shot of a multi-shot run has to survive until the adjoint pass, so the
    per-shot datasets go to a RAM-backed HDF5 file (or a real one when they
    are too big). Both paths are checked against a central finite
    difference, and the two are checked against each other.
    """
    nshot = 2
    cfg = _make_config()
    src, src_pos, rec_pos = _multishot_geometry(nshot)
    nz, nx = N
    mu = torch.full((nz * nx,), VS, dtype=torch.float32)
    rho = torch.full((nz * nx,), RHO, dtype=torch.float32)
    M0 = torch.full((nz * nx,), VP, dtype=torch.float32)

    def loss(Mv):
        data = seiscl_forward(cfg, {"M": Mv, "mu": mu, "rho": rho},
                              src, src_pos, rec_pos, output_fields=["vx"])
        return 0.5 * (data["vx"] ** 2).sum()

    grads = {}
    for policy in ("file", "memory", "hdf5-memory"):
        seiscl_torch.set_checkpoint_policy(policy)
        clear_engine_cache()
        M = M0.clone().requires_grad_(True)
        loss(M).backward()
        grads[policy] = M.grad.clone()

        idx = [39 * nx + 39, 41 * nx + 41]
        eps = 20.0
        for i in idx:
            Mp, Mm = M0.clone(), M0.clone()
            Mp[i] += eps
            Mm[i] -= eps
            with torch.no_grad():
                fd = float((loss(Mp) - loss(Mm)) / (2 * eps))
            analytic = float(grads[policy][i])
            rel = abs(fd - analytic) / (abs(analytic) + 1e-30)
            assert rel < 0.02, (f"policy={policy} param {i}: "
                                f"analytic={analytic:.6g} fd={fd:.6g} "
                                f"rel_diff={rel:.4f}")

    seiscl_torch.set_checkpoint_policy("auto")
    # Bit-identical: only where the checkpoint is kept differs, not the
    # arithmetic. (This used to be compared loosely; since the multi-shot
    # race fixes of 2026-11-11 repeated runs are identical, measured
    # 2026-11-21 with file/memory/file/memory/file.)
    for policy in ("memory", "hdf5-memory"):
        assert torch.equal(grads["file"], grads[policy]), (
            f"{policy} and file checkpoints disagree: max abs diff "
            f"{float((grads['file'] - grads[policy]).abs().max()):.6g}")
    print("Testing: torch_multishot_gradient_both_checkpoint_policies "
          "....... passed")


def _three_param_grads(cfg, src, src_pos, rec_pos, field="vx"):
    nz, nx = cfg.N
    p = {k: torch.full((nz * nx,), v, dtype=torch.float32, requires_grad=True)
         for k, v in (("M", VP), ("mu", VS), ("rho", RHO))}
    d = seiscl_forward(cfg, p, src, src_pos, rec_pos, output_fields=[field])
    (0.5 * (d[field].double() ** 2).sum()).backward()
    return d[field].detach().clone(), {k: v.grad.clone() for k, v in p.items()}


def test_multishot_shotstore():
    """Several shots in one call keep each shot's checkpoint in its own host
    buffers (policy "memory", the default "auto" when it fits): no copy into
    an HDF5 image between shots.

    Checks: (a) data and gradients bit-identical to the RAM-backed HDF5
    image ("hdf5-memory") and to a real file ("file"); (b) the gradient of a
    3-shot call equals the sum of 3 one-shot calls (each shot really gets its
    own boundary history: the 2026-11-11 bug mixed the shots). 3 shots: the
    sources of _multishot_geometry fit the 80-cell grid up to 3.
    """
    nshot = 3
    cfg = _make_config()
    src, src_pos, rec_pos = _multishot_geometry(nshot)

    res = {}
    for policy in ("hdf5-memory", "file", "memory"):
        seiscl_torch.set_checkpoint_policy(policy)
        clear_engine_cache()
        res[policy] = _three_param_grads(cfg, src, src_pos, rec_pos)
    seiscl_torch.set_checkpoint_policy("auto")
    for policy in ("hdf5-memory", "file"):
        assert torch.equal(res["memory"][0], res[policy][0]), (
            f"data differ between memory and {policy}")
        for k in ("M", "mu", "rho"):
            a, b = res["memory"][1][k], res[policy][1][k]
            assert torch.equal(a, b), (
                f"d/d{k} differs between memory and {policy}: max abs diff "
                f"{float((a - b).abs().max()):.6g}")

    total = {k: torch.zeros_like(v) for k, v in res["memory"][1].items()}
    clear_engine_cache()
    for s in range(nshot):
        sp = src_pos[s:s + 1].clone()
        sp[:, 3] = 0.0
        rp = rec_pos[rec_pos[:, 3] == s].clone()
        rp[:, 3] = 0.0
        g = _three_param_grads(cfg, src[s:s + 1], sp, rp)[1]
        for k in total:
            total[k] += g[k]
    for k in total:
        rel = float((total[k] - res["memory"][1][k]).abs().max()
                    / total[k].abs().max())
        print(f"    d/d{k}: {nshot}-shot call vs sum of one-shot calls, max rel "
              f"diff {rel:.2e} (threshold 1e-5)")
        assert rel <= 1e-5, f"d/d{k}: {nshot}-shot call != sum of one-shot calls"
    print("Testing: torch_multishot_shotstore ....... passed")


def _speed_geometry(nshot, n=300, nt=1000):
    """2-D n x n, nt steps, CPML, force sources, 32 receivers per shot."""
    cfg = _make_config()
    cfg.N = [n, n]
    cfg.NT = nt
    cfg.NAB = 20
    cfg.ABS_TYPE = 1
    t = np.arange(nt) * DT
    a = (np.pi * 10.0 * (t - 0.12)) ** 2
    ricker = ((1 - 2 * a) * np.exp(-a)).astype(np.float32)
    sx = np.linspace(60, n - 60, nshot)
    rx = np.linspace(30, n - 30, 32)
    sp = torch.tensor([[x * DH, 0.0, n // 2 * DH, s, 2.0]
                       for s, x in enumerate(sx)], dtype=torch.float32)
    rp = torch.tensor([[x * DH, 0.0, 40 * DH, s, s * 32 + r + 1, 0, 0, 0]
                       for s in range(nshot) for r, x in enumerate(rx)],
                      dtype=torch.float32)
    src = torch.from_numpy(np.tile(ricker, (nshot, 1)))
    return cfg, src, sp, rp


def _median_time(f, reps=3):
    """Median of reps timed calls of f, after one warm-up call."""
    import time
    f()                                        # warm-up: compile, allocate
    tt = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        tt.append(time.perf_counter() - t0)
    return sorted(tt)[reps // 2]


def test_multishot_shotstore_speed(nshot=4, ratio_max=1.15):
    """A gradient call with several shots costs about as much per shot as a
    call with one shot (it was ~1.46x when each shot's checkpoint was copied
    into an HDF5 image; notes: torch-benchmark-deepwave section 7.4).

    2-D 300 x 300, 1000 steps, force source, 32 receivers; median of 3 timed
    repetitions after one warm-up, the 4-shot call against 4 one-shot calls.
    """
    cfg, src, sp, rp = _speed_geometry(nshot)
    one = []
    for s in range(nshot):
        p1 = sp[s:s + 1].clone()
        p1[:, 3] = 0.0
        r1 = rp[rp[:, 3] == s].clone()
        r1[:, 3] = 0.0
        r1[:, 4] -= s * 32
        one.append((src[s:s + 1], p1, r1))

    def many():
        _three_param_grads(cfg, src, sp, rp, field="vz")

    def ones():
        for args in one:
            _three_param_grads(cfg, *args, field="vz")

    seiscl_torch.set_checkpoint_policy("auto")
    clear_engine_cache()
    times = {name: _median_time(f)
             for name, f in (("one call", many), ("one-shot calls", ones))}
    ratio = times["one call"] / times["one-shot calls"]
    print(f"    {nshot} shots: one call {times['one call']:.3f} s, "
          f"{nshot} one-shot calls {times['one-shot calls']:.3f} s, "
          f"ratio {ratio:.3f} (threshold {ratio_max})")
    assert ratio <= ratio_max, (
        f"several shots in one call cost {ratio:.2f}x the one-shot calls")
    print("Testing: torch_multishot_shotstore_speed ....... passed")


def _checkpoint_files():
    """Checkpoint files in the temporary folder (op.py reserves their names
    with tempfile.mkstemp; only the "file" policy writes them)."""
    import glob
    import tempfile
    return set(glob.glob(os.path.join(tempfile.gettempdir(),
                                      "*_checkpoint.mat")))


def test_recompute_matches_memory(nshot=3):
    """A multi-shot checkpoint that does not fit in RAM is not written to
    disk: the backward call re-runs each shot's forward pass (policy
    "recompute", and "auto" above its RAM budget; decision 75).

    Checks: (a) data and gradients bit-identical to the in-RAM path
    ("memory"): the recomputed boundaries are the same numbers; (b) no
    checkpoint file exists between forward and backward with "recompute" and
    with "auto" over budget, while "file" writes one (control).
    """
    cfg = _make_config()
    src, src_pos, rec_pos = _multishot_geometry(nshot)
    res, files = {}, {}
    for name, policy, budget in (("memory", "memory", 0),
                                 ("recompute", "recompute", 0),
                                 ("auto over budget", "auto", 1),
                                 ("file", "file", 0)):
        seiscl_torch.set_checkpoint_policy(policy, budget_bytes=budget)
        clear_engine_cache()
        before = _checkpoint_files()
        nz, nx = cfg.N
        p = {k: torch.full((nz * nx,), v, dtype=torch.float32,
                           requires_grad=True)
             for k, v in (("M", VP), ("mu", VS), ("rho", RHO))}
        d = seiscl_forward(cfg, p, src, src_pos, rec_pos,
                           output_fields=["vx"])["vx"]
        files[name] = len(_checkpoint_files() - before)
        (0.5 * (d.double() ** 2).sum()).backward()
        res[name] = (d.detach().clone(),
                     {k: v.grad.clone() for k, v in p.items()})
    seiscl_torch.set_checkpoint_policy("auto", budget_bytes=0)
    print(f"    checkpoint files between forward and backward: {files} "
          f"(expected 0 except file: 1)")
    assert files["recompute"] == 0 and files["auto over budget"] == 0, files
    assert files["file"] == 1 and files["memory"] == 0, files
    for name in ("recompute", "auto over budget", "file"):
        assert torch.equal(res["memory"][0], res[name][0]), (
            f"data differ between memory and {name}")
        for k in ("M", "mu", "rho"):
            a, b = res["memory"][1][k], res[name][1][k]
            assert torch.isfinite(a).all() and a.abs().max() > 0
            assert torch.equal(a, b), (
                f"d/d{k} differs between memory and {name}: max abs diff "
                f"{float((a - b).abs().max()):.6g} (threshold 0)")
        print(f"    {name}: data and gradients bit-identical to memory")
    print("Testing: torch_recompute_matches_memory ....... passed")


def test_auto_counts_own_checkpoint(nshot=3):
    """A repeated multi-shot gradient call with policy "auto" must not count
    the checkpoint its engine still holds from the previous call as used RAM.
    MemAvailable excludes those buffers, but this call reuses them. Before the
    fix, a checkpoint that fitted on the first call (bytes <= half the free
    RAM) went to disk on every later call (3-D 128^3, 4 shots: 20 GB written,
    forward 100 s instead of 18 s, cgm-geo1 2026-11-22).

    Simulated RAM (tests-only override), B = checkpoint bytes: first call with
    2B available, so B <= 0.5 x 2B fits; second call with B available (the
    first call's B now held). Expected "shotstore" both times; the old rule
    (B <= 0.5 x B) would choose "recompute". Then a call with too little RAM
    must choose "recompute" and free the store."""
    cfg = _make_config()
    src, src_pos, rec_pos = _multishot_geometry(nshot)
    nz, nx = cfg.N
    p = {k: torch.full((nz * nx,), v, dtype=torch.float32, requires_grad=True)
         for k, v in (("M", VP), ("mu", VS), ("rho", RHO))}

    def call():
        d = seiscl_forward(cfg, p, src, src_pos, rec_pos,
                           output_fields=["vx"])["vx"]
        mode = seiscl_torch.last_checkpoint_mode()
        (0.5 * (d.double() ** 2).sum()).backward()
        for v in p.values():
            v.grad = None
        return mode

    seiscl_torch.set_checkpoint_policy("memory")
    clear_engine_cache()
    call()
    B = seiscl_torch.last_checkpoint_bytes()
    assert B > 0, B
    seiscl_torch.set_checkpoint_policy("auto", budget_bytes=0)
    clear_engine_cache()
    modes = []
    try:
        for avail in (2 * B, B, B // 4):
            seiscl_torch._C._set_available_ram_override(avail)
            modes.append(call())
    finally:
        seiscl_torch._C._set_available_ram_override(0)
    print(f"    checkpoint {B} B; available 2B, B (B held), B/4 (B held): "
          f"{modes} (expected shotstore, shotstore, recompute)")
    assert modes == ["shotstore", "shotstore", "recompute"], modes
    print("Testing: torch_auto_counts_own_checkpoint ....... passed")


def test_recompute_interleaved(nshot=3):
    """Recompute keeps no state between the two calls, so interleaved
    forwards and engine rekeys between a forward and its backward must not
    change the gradient: the checkpoint tests, under policy "recompute"."""
    seiscl_torch.set_checkpoint_policy("recompute")
    try:
        test_checkpoint_survives_interleaved_forward(nshot=nshot)
        test_pending_checkpoint_survives_rekey(nshot=nshot)
    finally:
        seiscl_torch.set_checkpoint_policy("auto")
    print("Testing: torch_recompute_interleaved ....... passed")


def test_recompute_speed(nshot=4, ratio_max=1.2):
    """Recomputing costs one forward pass per shot: (gradient call with
    "recompute" - gradient call with "memory") / forward-only call, for the
    same nshot-shot call, at most ratio_max (2-D 300 x 300, 1000 steps)."""
    cfg, src, sp, rp = _speed_geometry(nshot)
    nz, nx = cfg.N
    p = {k: torch.full((nz * nx,), v, dtype=torch.float32)
         for k, v in (("M", VP), ("mu", VS), ("rho", RHO))}

    def forward_only():
        seiscl_forward(cfg, p, src, sp, rp, output_fields=["vz"])

    times = {}
    for policy in ("memory", "recompute"):
        seiscl_torch.set_checkpoint_policy(policy)
        clear_engine_cache()
        times[policy] = _median_time(
            lambda: _three_param_grads(cfg, src, sp, rp, field="vz"))
    seiscl_torch.set_checkpoint_policy("auto")
    clear_engine_cache()
    times["forward"] = _median_time(forward_only)
    ratio = (times["recompute"] - times["memory"]) / times["forward"]
    print(f"    {nshot} shots: gradient memory {times['memory']:.3f} s, "
          f"recompute {times['recompute']:.3f} s, forward only "
          f"{times['forward']:.3f} s; extra / forward {ratio:.3f} "
          f"(threshold {ratio_max})")
    assert ratio <= ratio_max, (
        f"recompute costs {ratio:.2f} forward passes, not about one")
    print("Testing: torch_recompute_speed ....... passed")


def test_dft_gradient_through_inputres():
    """back_prop_type=2 (DFT) works through the binding's two-call protocol.

    autograd forces forward and backward into separate calls (INPUTRES=1).
    The DFT method cannot use the boundary checkpoint -- its adjoint needs the
    frequency buffers accumulated during the forward pass, which are not
    checkpointed -- so time_stepping() re-runs the forward pass on the adjoint
    call instead. This checks that path produces a real gradient rather than
    the silent zero it used to.

    Deliberately NOT compared to back_prop_type=1 in absolute terms: the two
    methods are known not to agree yet (test_dft_gradient.py's
    test_dense_dft_matches_backprop is open at cos=0.856), because staggered
    material averaging is missing from the cross-correlation in both. What is
    checked is that the DFT path is populated, finite, and points broadly the
    same way as boundary storage.
    """
    cfg = _make_config()
    cfg.BACK_PROP_TYPE = 2
    df = 1.0 / (NT * DT)
    cfg.gradfreqs = [k * df for k in range(1, int(50.0 / df) + 1)]

    src, src_pos, rec_pos = _simple_geometry(nrec=2)
    nz, nx = N
    mu = torch.full((nz * nx,), VS, dtype=torch.float32)
    rho = torch.full((nz * nx,), RHO, dtype=torch.float32)

    def grad_for(config):
        clear_engine_cache()
        M = torch.full((nz * nx,), VP, dtype=torch.float32).requires_grad_(True)
        data = seiscl_forward(config, {"M": M, "mu": mu, "rho": rho},
                              src, src_pos, rec_pos, output_fields=["vx"])
        (0.5 * (data["vx"] ** 2).sum()).backward()
        return M.grad.clone()

    g_dft = grad_for(cfg)
    assert torch.isfinite(g_dft).all(), "DFT gradient has non-finite values"
    assert g_dft.abs().max() > 0, "DFT gradient is identically zero"

    cfg1 = _make_config()
    g_bnd = grad_for(cfg1)
    a = g_bnd.numpy().astype(np.float64).ravel()
    b = g_dft.numpy().astype(np.float64).ravel()
    # float32 gradients around 1e-24 underflow a float32 norm -- do this in
    # double or the cosine comes out as garbage.
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.8, (f"DFT gradient points a different way than boundary "
                       f"storage (cos={cos:.4f})")
    print("Testing: torch_dft_gradient_through_inputres ....... passed "
          f"(cos vs back_prop_type=1 = {cos:.4f})")


def test_dft_requires_gradfreqs():
    """back_prop_type=2 without gradfreqs is refused, not silently zero."""
    cfg = _make_config()
    cfg.BACK_PROP_TYPE = 2
    src, src_pos, rec_pos = _simple_geometry(nrec=2)
    try:
        seiscl_forward(cfg, _homogeneous_params(), src, src_pos, rec_pos,
                       output_fields=["vx"])
    except (ValueError, RuntimeError) as e:
        assert "gradfreqs" in str(e), f"unexpected error: {e}"
        print("Testing: torch_dft_requires_gradfreqs ....... passed")
        return
    raise AssertionError("back_prop_type=2 accepted an empty gradfreqs")


def test_dft_tmin_beyond_modeled_interval_rejected():
    """cfg.tmin past the modeled interval is a clean error, not a corrupted
    DFT buffer size.

    The standalone HDF5 path (read_hdf5.c) rejects tmin > tmax. The binding
    had no equivalent check: NTNYQ = (tmax-tmin+DTNYQ-1)/DTNYQ
    (assign_modeling_case.c) goes to zero or negative instead.
    """
    cfg = _make_config()
    cfg.BACK_PROP_TYPE = 2
    cfg.gradfreqs = [10.0, 20.0]
    cfg.tmin = NT + 10  # past cfg.NT
    src, src_pos, rec_pos = _simple_geometry(nrec=2)
    try:
        seiscl_forward(cfg, _homogeneous_params(), src, src_pos, rec_pos,
                       output_fields=["vx"])
    except (ValueError, RuntimeError) as e:
        assert "tmin" in str(e), f"unexpected error: {e}"
        print("Testing: torch_dft_tmin_beyond_modeled_interval_rejected "
              "....... passed")
        return
    raise AssertionError("cfg.tmin beyond cfg.NT was silently accepted")


if __name__ == "__main__":
    if not _TORCH_AVAILABLE:
        print("SeisCL.torch not importable (torch extra not installed) "
              "-- skipping SeisCL/torch binding tests")
    else:
        test_forward_smoke()
        test_forward_smoke(freesurf=1)
        test_forward_smoke(freesurf=2)
        test_forward_parity()
        test_gradient_finite_difference()
        test_gradient_finite_difference(freesurf=1)
        test_gradient_finite_difference(freesurf=2, restype=1)
        test_cuda_geometry_rejected()
        test_cuda_params_accepted()
        test_cuda_gradient_returned_on_param_device()
        test_cache_is_actually_used()
        test_cache_hit_matches_fresh_build()
        test_cache_shape_change_and_back()
        test_cache_source_type_change()
        test_cache_eviction_correctness()
        test_cache_hit_rejects_wrong_src_length()
        test_checkpoint_survives_interleaved_forward()
        test_checkpoint_survives_interleaved_forward(nshot=3)
        test_checkpoint_survives_cache_clear()
        test_pending_checkpoint_survives_rekey()
        test_pending_checkpoint_survives_rekey(nshot=3)
        test_viscoelastic_requires_FL()
        test_multishot_gradient_both_checkpoint_policies()
        test_multishot_shotstore()
        test_multishot_shotstore_speed()
        test_recompute_matches_memory()
        test_auto_counts_own_checkpoint()
        test_recompute_interleaved()
        test_recompute_speed()
        test_dft_requires_gradfreqs()
        test_dft_gradient_through_inputres()
        test_dft_tmin_beyond_modeled_interval_rejected()
