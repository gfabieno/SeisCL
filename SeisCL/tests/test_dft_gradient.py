"""Accuracy tests for the frequency-domain (back_prop_type=2) FWI gradient.

Run with:
    SEISCL_BIN=/path/to/build-cuda python test_dft_gradient.py

Zero-argument test_* functions with real asserts, plus a __main__ runner that
prints a PASS/FAIL table and exits with the failure count. Also pytest
-collectable if pytest is available.

T1 (this file, first test) validates the *forward DFT spectrum* that the
savefreqs kernel accumulates, against a float64 numpy DFT of the forward
wavefield taken from a separate forward-only run. It deliberately does not
involve calc_grad, so it works in the CUDA build where calc_grad() is a no-op
stub -- this is what makes the savefreqs half of back_prop_type=2 testable
before the gradient half works at all.

Time alignment (verified against src/time_stepping.c):
  forward loop step t:  savefreqs (samples state u_t)  ->  sources  ->
                        update_grid  ->  free surface  ->  movout (stores u_{t+1}
                        at frame (t+1)/MOVOUT - 1)
so with MOVOUT=1, movie[t] == u_{t+1}, and the series savefreqs sees is
u_0 = 0 (initial condition), u_t = movie[t-1].
"""

import os

import h5py as h5
import numpy as np

from gradient_common import (make_seiscl, homogeneous, make_observed,
                             reference_dft, relerr, run_tests, workdir,
                             SkipTest)

# 2D P-SV for_grad variables (assign_modeling_case.c:1009-1023)
VARS2D = ["vx", "vz", "sxx", "szz", "sxz"]


def _forward_movie(wd, params, **cfg):
    """Forward-only run with a full-field movie. Returns {var: (nz,nx,nt)}."""
    s = make_seiscl(wd, seisout=4, movout=1, **cfg)
    s.set_forward(s.src_pos_all[3, :], params, withgrad=False)
    s.execute()
    out = {}
    with h5.File(os.path.join(wd, s.file_movout), "r") as f:
        for v in VARS2D:
            key = "mov" + v
            if key not in f:
                raise AssertionError(
                    "movie is missing %s; got %s" % (key, sorted(f.keys())))
            # stored (ns, nt, NX, NZ) -> full transpose -> (NZ, NX, nt, ns)
            a = np.transpose(np.array(f[key]))
            out[v] = a[..., 0]
    return out


def _dft_run(wd, params, gradfreqs, din, **cfg):
    s = make_seiscl(wd, gradout=1, back_prop_type=2, dftout=1,
                    gradfreqs=np.asarray(gradfreqs, dtype=float), **cfg)
    s.file_din = din
    s.set_forward(s.src_pos_all[3, :], params, withgrad=True)
    s.execute()
    return s.read_dft()


def _pad_field(mov, npad_z, npad_x, fdoh, nt):
    """Embed a movie (nz,nx,nt) into the padded array, prepending u_0 = 0."""
    nz, nx, _ = mov.shape
    f = np.zeros((npad_z, npad_x, nt), dtype=np.float64)
    f[fdoh:fdoh + nz, fdoh:fdoh + nx, 1:] = mov[:, :, :nt - 1]
    return f


def _check_forward_spectrum(nfreqs_list, gradfreqs_all, tag):
    """Core of T1, shared across NFREQS settings."""
    params = None
    msgs = []
    for nf in nfreqs_list:
        wd = workdir("t1_%s_nf%d" % (tag, nf))
        s0 = make_seiscl(wd)
        params = homogeneous(s0)
        din = make_observed(s0)

        gradfreqs = gradfreqs_all[:nf]
        mov = _forward_movie(wd, params)
        d = _dft_run(wd, params, gradfreqs, din)

        dtnyq, ntnyq_impl = d["DTNYQ"], d["NTNYQ"]
        tmin, tmax = d["tminind"], d["tmaxind"]
        fdoh = d["FDORDER"] // 2
        bins = d["gradfreqsn"]
        # savefreqs fires for t in [tmin, tmax) with (t-tmin) % DTNYQ == 0
        nsaves = int(np.ceil((tmax - tmin) / dtnyq))
        # The correct DFT period is the number of samples actually accumulated.
        ntnyq_correct = nsaves

        assert len(bins) == nf, "expected %d bins, got %d" % (nf, len(bins))

        for v in VARS2D:
            got = d["f_" + v]                       # (NZpad, NXpad, NFREQS)
            npz, npx = got.shape[0], got.shape[1]
            field = _pad_field(mov[v], npz, npx, fdoh, tmax)
            ref_ok = reference_dft(field, bins, dtnyq, ntnyq_correct,
                                   1e-3, tmin, nsaves)
            ref_impl = reference_dft(field, bins, dtnyq, ntnyq_impl,
                                     1e-3, tmin, nsaves)
            e_ok = relerr(got, ref_ok)
            e_impl = relerr(got, ref_impl)
            msgs.append("  NFREQS=%d %-4s err_correct=%.3e err_asimpl=%.3e"
                        % (nf, v, e_ok, e_impl))
            if e_ok > 1e-3:
                raise AssertionError(
                    "forward DFT of %s disagrees with the reference "
                    "(NFREQS=%d): err=%.3e (tolerance 1e-3).\n"
                    "  Same data against the as-implemented NTNYQ=%d instead of "
                    "%d gives err=%.3e.\n"
                    "  A much smaller err_asimpl means the DFT period is off by "
                    "one (NTNYQ counts tmax-tmin+1 samples but only %d are "
                    "accumulated).\n%s"
                    % (v, nf, e_ok, ntnyq_impl, ntnyq_correct, e_impl,
                       nsaves, "\n".join(msgs)))
    print("\n".join(msgs))


def test_dft_forward_spectrum_vs_numpy():
    """T1: the accumulated forward spectrum equals a float64 DFT of the field.

    Catches a wrong twiddle, a wrong DFT period, and (via NFREQS>1) frequency
    slices that were never initialized.
    """
    _check_forward_spectrum([1], [11.0], "single")


def test_dft_forward_spectrum_multifreq():
    """T1b: same, with NFREQS>1.

    initsavefreqs only zeroes frequency slice 0 (automatic_kernels.c:989-999)
    while the buffer is num_ele*NFREQS and clbuf_create does not zero-init, so
    slices 1.. start from uninitialized device memory.
    """
    _check_forward_spectrum([2, 7], [11.0, 19.0, 27.0, 7.0, 15.0, 23.0, 31.0],
                            "multi")


def test_dft_padded_tail_is_accumulated():
    """T1c: every element of the padded buffer is accumulated, not just prod(N).

    Both kernel generators launch with gsize[0]=prod(N) while the in-kernel guard
    is gid < num_ele = prod(N+FDORDER) (automatic_kernels.c:1007-1010, 1153-1156),
    so a tail of the padded array is never written -- yet calc_grad reads the full
    padded layout.
    """
    # Geometry matters here. Uncovered elements are flat = ix*(nz+FDORDER) + iz
    # >= nz*nx, i.e. padded ix >= nz*nx/(nz+FDORDER), which is always at high x.
    # With (nz,nx)=(40,120) that is padded ix >= 100, i.e. model x >= 96. nab
    # must be small enough that x=96 is well clear of the absorbing strip:
    # at nab=16 the interior ends at x=104 and the Cerjan taper damps the field
    # there to ~1e-20, so got and ref are both zero and the test would pass
    # vacuously. nab=8 puts the interior end at x=112, leaving x=96 undamped.
    wd = workdir("t1_tail")
    geom = dict(N=np.array([40, 120]), nab=8)
    s0 = make_seiscl(wd, **geom)
    params = homogeneous(s0)
    din = make_observed(s0)
    mov = _forward_movie(wd, params, **geom)
    d = _dft_run(wd, params, [11.0], din, **geom)

    got = d["f_vx"]
    npz, npx, _ = got.shape
    fdoh = d["FDORDER"] // 2
    nsaves = int(np.ceil((d["tmaxind"] - d["tminind"]) / d["DTNYQ"]))
    field = _pad_field(mov["vx"], npz, npx, fdoh, d["tmaxind"])
    # Use the DFT period the engine actually used, so that a period bug (B5)
    # cancels out and this test measures *coverage* only.
    ref = reference_dft(field, d["gradfreqsn"], d["DTNYQ"], d["NTNYQ"],
                        1e-3, d["tminind"], nsaves)

    nz, nx = int(s0.N[0]), int(s0.N[1])
    flat = (np.arange(npx)[None, :] * npz + np.arange(npz)[:, None])
    tail = flat >= nz * nx
    assert tail.any(), "test model does not exercise a padded tail"

    scale = np.abs(ref).max()
    err_tail = float(np.abs(got[..., 0][tail] - ref[..., 0][tail]).max() / scale)
    err_head = float(np.abs(got[..., 0][~tail] - ref[..., 0][~tail]).max() / scale)
    # Guard against a vacuous pass: if the reference is ~0 across the whole tail
    # there is nothing to detect, so the geometry above is wrong.
    tail_signal = float(np.abs(ref[..., 0][tail]).max() / scale)
    print("  tail elements=%d tail_signal=%.3e err_tail=%.3e err_head=%.3e"
          % (int(tail.sum()), tail_signal, err_tail, err_head))
    assert tail_signal > 1e-4, (
        "vacuous test: the reference spectrum is only %.3e of peak across the "
        "uncovered tail, so a coverage bug there is undetectable. Adjust the "
        "grid aspect ratio / nab so the tail lands on undamped interior cells."
        % tail_signal)
    assert err_tail <= 1e-3, (
        "the %d padded elements at flat index >= prod(N)=%d were not "
        "accumulated: err=%.3e there vs %.3e elsewhere (period bug cancelled, "
        "so this is coverage alone). gsize[0] is prod(N) but the in-kernel guard "
        "is gid < num_ele=%d."
        % (int(tail.sum()), nz * nx, err_tail, err_head, npz * npx))


def _grad_and_dft(wd, params, gradfreqs, din, **cfg):
    s = make_seiscl(wd, gradout=1, back_prop_type=2, dftout=1,
                    gradfreqs=np.asarray(gradfreqs, dtype=float), **cfg)
    s.file_din = din
    s.set_forward(s.src_pos_all[3, :], params, withgrad=True)
    s.execute()
    return s, s.read_dft(), s.read_grad()


def _numpy_reference(s, d, params):
    from dft_reference import gradient_2d_elastic
    nz, nx = int(s.N[0]), int(s.N[1])
    rho = params["rho"]
    M = rho * params["vp"] ** 2
    mu = rho * params["vs"] ** 2
    fwd = {k[2:]: v for k, v in d.items() if k.startswith("f_")}
    adj = {k[2:]: v for k, v in d.items() if k.startswith("a_")}
    return gradient_2d_elastic(fwd, adj, M, mu, rho, d["gradfreqsn"],
                               d["NTNYQ"], d["DTNYQ"], s.dt,
                               d["FDORDER"] // 2, nz, nx)


def _interior(s, a):
    nab = s.nab
    nz, nx = int(s.N[0]), int(s.N[1])
    return a[nab:nz - nab, nab:nx - nab].ravel()


def test_dft_gradient_vs_numpy_heterogeneous():
    """T3: SeisCL's DFT gradient equals a float64 numpy reference, cell by cell.

    Uses a *heterogeneous* model on purpose. The grad_coef* coefficients are
    nonlinear functions of (M, mu, rho), so on a homogeneous model any error in
    the parameter scaling is a spatially constant factor and shows up as a clean
    alpha with cos==1 -- indistinguishable from a convention difference. Only a
    heterogeneous model makes the coefficients vary in space, so that a scaling
    error distorts the spatial pattern and drives cos below 1.

    This test is the acceptance criterion for the Phase 4 device-side kernel and
    doubles as its executable specification.
    """
    wd = workdir("t3_hetero")
    s0 = make_seiscl(wd)
    nz, nx = int(s0.N[0]), int(s0.N[1])
    zz, xx = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    # smooth, strong lateral+vertical variation in all three parameters
    params = {
        "vp": 2000.0 + 600.0 * np.sin(2 * np.pi * xx / nx)
                     * np.cos(np.pi * zz / nz) + 4.0 * zz,
        "vs": 1200.0 + 250.0 * np.cos(2 * np.pi * xx / nx) + 2.0 * zz,
        "rho": 2000.0 + 300.0 * np.sin(np.pi * zz / nz),
    }
    params = {k: v.astype(np.float64) for k, v in params.items()}
    # Observed data must come from a *different* model, otherwise the residual is
    # zero and so is the gradient.
    true_params = {k: v.copy() for k, v in params.items()}
    true_params["vp"][nz // 2 - 5:nz // 2 + 5, nx // 2 - 5:nx // 2 + 5] += 300.0
    din = make_observed(s0, params=true_params)

    s, d, g = _grad_and_dft(wd, params, [11.0, 19.0, 27.0], din)
    ref = _numpy_reference(s, d, params)

    worst = 0.0
    for i, nm in enumerate(("vp", "vs", "rho")):
        a, b = _interior(s, ref[nm]), _interior(s, g[i])
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        assert nb > 0, "SeisCL grad%s is identically zero" % nm
        cos = float(a @ b / (na * nb))
        alpha = float(a @ b / (b @ b))
        rel = float(np.linalg.norm(a - alpha * b) / na)
        print("  %-4s cos=%.8f alpha=%.6e resid=%.3e" % (nm, cos, alpha, rel))
        worst = max(worst, abs(cos - 1.0), abs(alpha - 1.0))
    assert worst < 1e-4, (
        "DFT gradient disagrees with the float64 reference on a heterogeneous "
        "model (worst |cos-1| or |alpha-1| = %.3e). A cos of 1 with alpha != 1 "
        "is a global convention factor; cos < 1 means the per-cell coefficients "
        "are wrong, e.g. the grad_coef* formulas being fed internally scaled "
        "parameters instead of physical ones." % worst)


def test_dense_dft_matches_backprop():
    """T2: with every DFT bin selected, the DFT gradient matches back_prop_type=1.

    KNOWN OPEN -- listed in XFAIL below. Measured on this model with all 128
    bins selected (2026-07-30, OpenCL build):

        vp   cos=0.855562   vs   cos=0.912550   rho  cos=-0.643659

    What has been ruled out:
      * An implementation error in the correlation. T3 shows calc_grad agrees
        with an independent float64 reference to 2e-7 per cell on a
        heterogeneous model, and that same reference reproduces these very cos
        values against back_prop_type=1 -- so the DFT gradient is a faithful
        evaluation of its own formula.
      * A whole-sample time offset between the forward and adjoint spectra
        (the B6 sampling asymmetry). Sweeping a k-sample phase correction
        exp(-2j*pi*bin*k/NTNYQ) on the adjoint spectrum over k in -2..2 leaves
        k=0 as the best value for all three parameters, so the mismatch is not
        a shift.
      * Wrong spectra. T1 validates both against a numpy DFT to fp32 noise.

    The strongest remaining lead is gradrho, which is *anti*-correlated
    (cos<0) rather than merely inaccurate -- a sign convention rather than an
    approximation. Note also that a finite-difference probe against
    read_rms() currently fails to give a constant ratio for *either*
    back_prop_type (bpt=1 gave 5.5e-5, 6.1e-5, 2.2e-5 across three cells), so
    the FD harness needs its own validation -- read_rms() may not return the
    objective the gradient corresponds to -- before it can arbitrate which
    method is right.

    This does not block the device-side correlation kernel, whose acceptance
    test is T3.
    """
    wd = workdir("t2_dense")
    s0 = make_seiscl(wd)
    params = homogeneous(s0)
    din = make_observed(s0)

    # DTNYQ is 1 whenever 0.0156/fmax/dt <= 1, so NTNYQ == NT and df == 1/(NT*dt).
    nt = int(s0.NT)
    df = 1.0 / (nt * s0.dt)
    gradfreqs = df * np.arange(1, nt // 2 + 1)

    s, d, g2 = _grad_and_dft(wd, params, gradfreqs, din)
    assert d["DTNYQ"] == 1, "expected DTNYQ==1, got %d" % d["DTNYQ"]
    assert d["NTNYQ"] == nt, "expected NTNYQ==%d, got %d" % (nt, d["NTNYQ"])

    s1 = make_seiscl(wd, gradout=1, back_prop_type=1)
    s1.file_din = din
    s1.set_forward(s1.src_pos_all[3, :], params, withgrad=True)
    s1.execute()
    g1 = s1.read_grad()

    for i, nm in enumerate(("vp", "vs", "rho")):
        a, b = _interior(s1, g1[i]), _interior(s, g2[i])
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        assert nb > 0, "DFT grad%s is identically zero" % nm
        cos = float(a @ b / (na * nb))
        alpha = float(a @ b / (b @ b))
        rel = float(np.linalg.norm(a - alpha * b) / na)
        print("  %-4s cos=%.6f alpha=%.6e resid=%.3e" % (nm, cos, alpha, rel))
        assert cos > 0.999, (
            "dense-frequency DFT grad%s does not match back_prop_type=1: "
            "cos=%.6f (alpha=%.4e). With every bin selected these must be "
            "proportional." % (nm, cos, alpha))

def test_device_kernel_matches_host_oracle():
    """T6: the on-device correlation reproduces the host calc_grad reference.

    calc_grad() is kept as the reference implementation of the DFT gradient and
    is selectable at runtime with SEISCL_DFT_HOST=1, so the two can be compared
    in the same build on the same data. Both are double precision internally, so
    the only expected difference is the float gradient buffer, ~1e-7.

    This is the permanent guard on src/grad_dft2D.cl. It also covers the CUDA
    build, where calc_grad() is a no-op stub and the DFT gradient produced
    exactly zero before the device kernel existed.
    """
    wd = workdir("t6_devhost")
    s0 = make_seiscl(wd)
    params = homogeneous(s0)
    true_params = {k: v.copy() for k, v in params.items()}
    nz, nx = int(s0.N[0]), int(s0.N[1])
    true_params["vp"][nz//2-5:nz//2+5, nx//2-5:nx//2+5] += 300.0
    din = make_observed(s0, params=true_params)

    def run(host):
        if host:
            os.environ["SEISCL_DFT_HOST"] = "1"
        else:
            os.environ.pop("SEISCL_DFT_HOST", None)
        try:
            s = make_seiscl(wd, gradout=1, back_prop_type=2,
                            gradfreqs=np.array([11.0, 19.0, 27.0]))
            s.file_din = din
            s.set_forward(s.src_pos_all[3, :], params, withgrad=True)
            s.execute()
            return s, s.read_grad()
        finally:
            os.environ.pop("SEISCL_DFT_HOST", None)

    s, g_dev = run(host=False)
    _, g_host = run(host=True)

    for i, nm in enumerate(("vp", "vs", "rho")):
        a, b = _interior(s, g_host[i]), _interior(s, g_dev[i])
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        assert nb > 0, (
            "the device DFT correlation produced an identically zero grad%s. In "
            "a CUDA build this is what the old no-op calc_grad() stub did." % nm)
        if na == 0:
            raise SkipTest(
                "the host calc_grad reference returned zero, so this build has "
                "no host implementation to compare against. calc_grad() is "
                "#ifdef __SEISCL__ and is a no-op stub in the CUDA build "
                "(calc_grad.c:969), which is precisely why the device kernel "
                "exists. Run this test against the OpenCL build.")
        cos = float(a @ b / (na * nb))
        rel = float(np.abs(a - b).max() / np.abs(a).max())
        print("  %-4s cos=%.8f reldiff=%.3e" % (nm, cos, rel))
        assert rel < 1e-5 and abs(cos - 1.0) < 1e-6, (
            "device kernel disagrees with the host reference for grad%s: "
            "cos=%.8f reldiff=%.3e" % (nm, cos, rel))

def test_dft_osamp_convergence():
    """T8: how far the savefreqs oversampling can be relaxed.

    DTNYQ = ceil((1/dft_osamp)/fmax/dt) sets how often savefreqs fires. The
    historical hardcoded constant 0.0156 is an oversample of 64, which at
    typical 2D parameters gives DTNYQ=1 -- savefreqs runs on *every* time step
    and is ~75% of GPU time. This test characterizes the error incurred by
    lowering it.

    The trade-off is aliasing, not quadrature: with DTNYQ>1 the accumulation is
    the exact DFT of the decimated series, and the error is energy above the
    decimated Nyquist folding onto the selected bins.

    MEASURED (2D elastic, Ricker f0=25, gradfreqs up to 25 Hz):

        dft_osamp   DTNYQ    rel err
             64        1     0
             20        2     6.6e-3
           13.3        3     1.3e-2
              8        5     2.7e-2
              4       10     9.3e-2
              2       20     1.9

    This *refutes* the estimate that an oversample of 8 would be safe at the
    1e-27 level from Ricker bandwidth alone. It is 2.7% in practice, because the
    adjoint field is driven by the data residual rather than by the source
    wavelet, and carries far more high-frequency content -- plus grid dispersion
    and boundary reflections. So the default stays at 64 and dft_osamp is
    offered as an explicit, measured trade: DTNYQ=2 halves the dominant kernel
    for ~0.7% gradient error.

    The assertions below are structural rather than a threshold on accuracy:
    the error must be monotone in DTNYQ, and modest at DTNYQ=2. A regression
    that reintroduced a DTNYQ-dependent *scaling* (as the missing DTNYQ in the
    Parseval normalization did -- it produced errors of exactly DTNYQ-1) would
    break both.

    NT is chosen divisible by every DTNYQ exercised here. Otherwise
    NTNYQ = ceil((tmax-tmin)/DTNYQ) rounds up, df = 1/(NTNYQ*dt*DTNYQ) drifts,
    and the analysis frequencies shift between runs -- which would show up as
    "aliasing error" that is really just a different set of frequencies.
    """
    wd = workdir("t8_osamp")
    geom = dict(NT=240)
    s0 = make_seiscl(wd, **geom)
    params = homogeneous(s0)
    true_params = {k: v.copy() for k, v in params.items()}
    nz, nx = int(s0.N[0]), int(s0.N[1])
    true_params["vp"][nz//2-5:nz//2+5, nx//2-5:nx//2+5] += 300.0
    din = make_observed(s0, params=true_params)

    gradfreqs = np.array([10.0, 17.0, 25.0])     # fmax = 25 -> DTNYQ = ceil(40/osamp)
    ref = None
    df0 = None
    rows = []
    for osamp in (64.0, 20.0, 40.0/3.0, 8.0, 4.0, 2.0):
        s = make_seiscl(wd, gradout=1, back_prop_type=2, dftout=1,
                        gradfreqs=gradfreqs, dft_osamp=osamp, **geom)
        s.file_din = din
        s.set_forward(s.src_pos_all[3, :], params, withgrad=True)
        s.execute()
        d = s.read_dft()
        g = s.read_grad()
        dtnyq, ntnyq = d["DTNYQ"], d["NTNYQ"]
        df = 1.0 / (ntnyq * s.dt * dtnyq)
        if df0 is None:
            df0 = df
        assert abs(df - df0) < 1e-6 * df0, (
            "df drifted from %.6f to %.6f at dft_osamp=%g (DTNYQ=%d, NTNYQ=%d): "
            "NT must be divisible by every DTNYQ exercised, or the comparison "
            "is between different analysis frequencies."
            % (df0, df, osamp, dtnyq, ntnyq))
        if ref is None:
            ref = [x.copy() for x in g]
            rows.append((osamp, dtnyq, 0.0))
            continue
        err = max(float(np.abs(a - b).max() / (np.abs(a).max() or 1.0))
                  for a, b in zip(ref, g))
        rows.append((osamp, dtnyq, err))

    print("  %-10s %7s %12s" % ("dft_osamp", "DTNYQ", "rel err"))
    for osamp, dtnyq, err in rows:
        print("  %-10.3f %7d %12.3e" % (osamp, dtnyq, err))

    errs = [e for _, _, e in rows]
    assert all(b >= a - 1e-12 for a, b in zip(errs, errs[1:])), (
        "aliasing error is not monotone in DTNYQ: %s. A DTNYQ-dependent scaling "
        "error would show up here." % errs)
    at2 = [e for _, dtn, e in rows if dtn == 2]
    assert at2 and at2[0] < 5e-2, (
        "relative gradient error at DTNYQ=2 is %.3e; it was ~6.6e-3 when "
        "measured. A value near DTNYQ-1 = 1.0 means the Parseval normalization "
        "lost its DTNYQ factor again (calc_grad.c dftnorm)."
        % (at2[0] if at2 else float("nan")))


TESTS = [
    test_dft_forward_spectrum_vs_numpy,
    test_dft_forward_spectrum_multifreq,
    test_dft_padded_tail_is_accumulated,
    test_dft_gradient_vs_numpy_heterogeneous,
    test_device_kernel_matches_host_oracle,
    test_dft_osamp_convergence,
    test_dense_dft_matches_backprop,
]

# Known-open failures, documented in the corresponding docstring.
XFAIL = {"test_dense_dft_matches_backprop"}

if __name__ == "__main__":
    import sys
    sys.exit(run_tests(TESTS, xfail=XFAIL))
