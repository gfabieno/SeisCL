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
                             reference_dft, relerr, run_tests, workdir)

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


TESTS = [
    test_dft_forward_spectrum_vs_numpy,
    test_dft_forward_spectrum_multifreq,
    test_dft_padded_tail_is_accumulated,
]

if __name__ == "__main__":
    import sys
    sys.exit(run_tests(TESTS))
