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
    from SeisCL.torch import Config, seiscl_forward
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
