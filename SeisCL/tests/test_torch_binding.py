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


def _grad_of(vp_value, other_forward_vp=None):
    """Gradient of 0.5*sum(vx**2) w.r.t. M at vp_value.

    If other_forward_vp is given, a second forward pass runs *between* this
    one and its backward -- gradient accumulation, and the case where the
    in-memory checkpoint handoff must fall back to a file because the second
    forward overwrites the first's boundary buffers.
    """
    cfg = _make_config()
    src, src_pos, rec_pos = _simple_geometry()
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


def test_checkpoint_survives_interleaved_forward():
    """A second forward before the first's backward must not corrupt it.

    The boundary wavefield normally stays in the engine's buffers between
    forward and backward instead of going through an HDF5 file. A second
    forward on the same engine overwrites those buffers, so the first pass's
    checkpoint has to be flushed to disk before that happens -- if it isn't,
    this returns a gradient computed from the *wrong* wavefield, silently.
    """
    clear_engine_cache()
    reference = _grad_of(VP)

    clear_engine_cache()
    interleaved = _grad_of(VP, other_forward_vp=VP + 300.0)

    assert torch.equal(reference, interleaved), (
        "gradient changed when another forward ran before backward: "
        f"max abs diff {float((reference - interleaved).abs().max()):.6g}")
    print("Testing: torch_checkpoint_survives_interleaved_forward ....... passed")


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


def test_pending_checkpoint_survives_rekey():
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
    src, src_pos, rec_pos = _simple_geometry()
    nz, nx = N
    mu = torch.full((nz * nx,), VS, dtype=torch.float32)
    rho = torch.full((nz * nx,), RHO, dtype=torch.float32)

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
    print("Testing: torch_pending_checkpoint_survives_rekey ....... passed")


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
    for policy in ("file", "memory"):
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
    # Not bit-identical: the engine itself is not run-to-run reproducible for
    # multi-shot gradients (file-vs-file differs too), so compare loosely.
    scale = float(grads["file"].abs().max())
    drift = float((grads["file"] - grads["memory"]).abs().max()) / scale
    assert drift < 0.02, f"RAM and file checkpoints disagree by {drift:.4f}"
    print("Testing: torch_multishot_gradient_both_checkpoint_policies "
          "....... passed")


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
        test_forward_parity()
        test_gradient_finite_difference()
        test_cuda_geometry_rejected()
        test_cuda_params_accepted()
        test_cuda_gradient_returned_on_param_device()
        test_cache_is_actually_used()
        test_cache_hit_matches_fresh_build()
        test_cache_shape_change_and_back()
        test_cache_eviction_correctness()
        test_cache_hit_rejects_wrong_src_length()
        test_checkpoint_survives_interleaved_forward()
        test_checkpoint_survives_cache_clear()
        test_pending_checkpoint_survives_rekey()
        test_multishot_gradient_both_checkpoint_policies()
        test_dft_requires_gradfreqs()
        test_dft_gradient_through_inputres()
        test_dft_tmin_beyond_modeled_interval_rejected()
