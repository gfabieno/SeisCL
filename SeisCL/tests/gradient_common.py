"""Shared helpers for the DFT (back_prop_type=2) gradient tests.

Deliberately self-contained: it does not import test_consistency.py or
test_analytics.py. The former has no asserts (it catches SeisCLError and prints
"failed", so it always exits 0) and the latter calls matplotlib.use('TkAgg') at
import time, which breaks headless runs.
"""

import os
import sys

import numpy as np

# Never rely on PATH to find the binary: on a dev box SeisCL_MPI often resolves
# to an unrelated build tree. SEISCL_BIN must point at the directory holding the
# binary to test.
SEISCL_BIN = os.environ.get("SEISCL_BIN")
if SEISCL_BIN:
    os.environ["PATH"] = os.path.abspath(SEISCL_BIN) + os.pathsep \
                         + os.environ.get("PATH", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from SeisCL.SeisCL import SeisCL, SeisCLError            # noqa: E402


# Base model. Deliberately the same recipe as test_torch_binding.py's
# _make_config(), the one existing assert-based numeric test, so failures here
# are comparable with failures there.
BASE = dict(N=np.array([60, 60]), ND=2, dh=10, dt=1e-3, NT=256, FDORDER=8,
            freesurf=0, abs_type=2, nab=16, f0=25, seisout=2, param_type=0)

VP, VS, RHO = 2000.0, 1200.0, 2000.0


def workdir(name):
    d = os.path.join(os.environ.get("SEISCL_TEST_WORKDIR",
                                    os.path.join(_HERE, "_dftwork")), name)
    os.makedirs(d, exist_ok=True)
    return d


def make_seiscl(wd, **overrides):
    """A configured SeisCL with one source at the centre and a receiver line."""
    cfg = dict(BASE)
    cfg.update(overrides)
    s = SeisCL(workdir=wd, **cfg)
    nz, nx = int(s.N[0]), int(s.N[1])
    z0 = (s.nab + 5) * s.dh
    s.src_pos_all = np.stack([[nx // 2 * s.dh], [0.], [z0], [0.], [100.]])
    # Keep well inside the absorbing strip: valid x is [nab*dh, (nx-nab)*dh].
    nr = 12
    xlo = (s.nab + 4) * s.dh
    xhi = (nx - s.nab - 4) * s.dh
    xr = np.linspace(xlo, xhi, nr)
    s.rec_pos_all = np.stack([xr, np.zeros(nr), np.full(nr, z0),
                              np.zeros(nr), np.arange(1, nr + 1, dtype=float),
                              np.zeros(nr), np.zeros(nr), np.zeros(nr)])
    s.src_all = None
    return s


def homogeneous(s):
    return {"vp": np.full(s.N, VP), "vs": np.full(s.N, VS),
            "rho": np.full(s.N, RHO)}


def with_anomaly(s, dvp=300.0):
    p = homogeneous(s)
    nz, nx = int(s.N[0]), int(s.N[1])
    p["vp"][nz // 2 - 5:nz // 2 + 5, nx // 2 - 5:nx // 2 + 5] += dvp
    return p


def make_observed(s, params=None):
    """Model 'observed' data into <workdir>/SeisCL_din.mat and return its path.

    Mirrors the pattern at test_consistency.py:210-221. Note file_din must
    include the workdir: callcmd() abspaths file_din but only joins workdir for
    the base file name.
    """
    if params is None:
        params = with_anomaly(s)
    s.set_forward(s.src_pos_all[3, :], params, withgrad=False)
    s.execute()
    s.write_data({"p": s.read_data()[0]}, filename="SeisCL_din.mat")
    return os.path.join(s.workdir, "SeisCL_din.mat")


def dft_params(gradfreqs, dt, tmin_ind, tmax_ind, osamp=64.0):
    """Reimplementation of gradfreqsn() at assign_modeling_case.c:506-539.

    Returns (DTNYQ, NTNYQ, df, gradfreqsn). Kept here so a test can predict what
    the engine will do; tests should still cross-check against the values the
    engine reports via read_dft(), which is the authoritative source.
    """
    fmax = float(np.max(gradfreqs))
    dtnyq = int(np.ceil((1.0 / osamp) / fmax / dt))
    ntnyq = int((tmax_ind - tmin_ind) / dtnyq + 1)
    df = 1.0 / ntnyq / dt / dtnyq
    return dtnyq, ntnyq, df, np.floor(np.asarray(gradfreqs) / df).astype(int)


def reference_dft(field, bins, dtnyq, ntnyq, dt, tmin_ind, nsaves):
    """Reference DFT matching what savefreqs accumulates, in float64.

    savefreqs is launched at t = tmin, tmin+DTNYQ, ... with the decimated index
    nt = (t-tmin)/DTNYQ, and accumulates
        fvar += DTNYQ*dt * exp(-2j*pi*bin*nt/NTNYQ) * field(t)
    (a rectangle-rule DFT of the decimated series).

    :param field: array (..., nt_frames) sampled at every time step
    :param bins:  integer DFT bin indices (gradfreqsn)
    :param nsaves: number of accumulated samples
    :return: complex array (..., len(bins))
    """
    n = np.arange(nsaves)
    tind = tmin_ind + n * dtnyq
    x = field[..., tind].astype(np.float64)
    out = np.zeros(field.shape[:-1] + (len(bins),), dtype=np.complex128)
    for j, b in enumerate(bins):
        w = dtnyq * dt * np.exp(-2j * np.pi * float(b) * n / float(ntnyq))
        out[..., j] = x @ w
    return out


def relerr(a, b):
    """Error normalized by the peak of the reference, not per element.

    Per-element relative error is meaningless where the reference is ~0, which
    is most of a padded wavefield array.
    """
    scale = np.abs(b).max()
    if scale == 0:
        return np.abs(a).max()
    return float(np.abs(a - b).max() / scale)


class SkipTest(Exception):
    """Raised by a test that cannot run in this build (e.g. the host calc_grad
    reference, which exists only in the OpenCL build)."""


def run_tests(tests, xfail=()):
    """Run zero-argument test callables, print a table, return the failure count.

    Works under plain `python test_x.py` and is also pytest-collectable.

    :param xfail: names of tests that are known to fail for a documented,
                  unresolved reason. They still run and still print their
                  diagnostics, but they do not contribute to the exit status --
                  so a genuine regression elsewhere is not masked by a red test
                  that everyone has learned to ignore. An xfail test that starts
                  passing is reported as XPASS and should be un-listed.
    """
    nfail = 0
    nskip = 0
    for fn in tests:
        name = fn.__name__
        expected_fail = name in xfail
        try:
            fn()
            if expected_fail:
                print("XPASS %s  (listed as xfail but passed -- un-list it)"
                      % name)
            else:
                print("PASS  %s" % name)
        except SkipTest as e:
            print("SKIP  %s\n      %s" % (name, e))
            nskip += 1
            continue
        except AssertionError as e:
            msg = str(e).replace("\n", "\n      ")
            if expected_fail:
                print("XFAIL %s  (known open)\n      %s" % (name, msg))
            else:
                nfail += 1
                print("FAIL  %s\n      %s" % (name, msg))
        except Exception as e:  # noqa: BLE001 - report, do not mask
            nfail += 1
            print("ERROR %s\n      %s: %s"
                  % (name, type(e).__name__,
                     str(e).replace("\n", "\n      ")[:400]))
    nx = sum(1 for fn in tests if fn.__name__ in xfail)
    print("\n%d/%d passed (%d known-open, %d skipped)"
          % (len(tests) - nfail - nx - nskip, len(tests) - nx - nskip,
             nx, nskip))
    return nfail
