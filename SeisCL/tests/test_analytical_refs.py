# -*- coding: utf-8 -*-
"""Checks of the analytical reference solutions themselves (SeisCL/analytical), without any FD run.

garvin2.compute_shot and lamb3D.compute_shot convolve the Green's function with the source wavelet. When the
convolution is circular (an FFT without padding), whatever arrives after the end of the record wraps around to
t = 0: with a record that ends during the Rayleigh wave, the reference has energy before the P wave can arrive,
and every FD solution compared with it looks wrong. Two properties a correct (linear) convolution must have:

  causality   nothing before the P arrival:  max |v(t < r / vp)| / max |v|  <= 1e-3
  window      a record twice as long gives the same first samples:
              max |v_short - v_long[:nt]| / max |v_long|  <= 1e-6

The geometry puts the Rayleigh wave at the end of the record (offset 400 m, record 0.3 s), as in a comparison whose
record is just long enough for the farthest receiver.
Usage: python SeisCL/tests/test_analytical_refs.py   (from the repository root, PYTHONPATH=.)
"""
import sys
import numpy as np

from SeisCL.analytical import garvin2, lamb3D
from SeisCL.tests.gradient_common import run_tests

VP, VS, RHO, F0 = 3500.0, 2000.0, 2000.0, 20.0
DT, RESAMP, NT = 0.25e-3, 10, 1200           # record 0.3 s
OFFSET = 400.0                                # Rayleigh wave (~0.92 vs) peaks at 1.5/f0 + 400/1840 = 0.292 s
CAUSAL_TOL, WINDOW_TOL = 1e-3, 1e-6


def ricker(nt, dt):
    t = -1.5 / F0 + np.arange(nt) * dt
    return (1 - 2 * (np.pi * F0 * t) ** 2) * np.exp(-(np.pi * F0 * t) ** 2)


def check(shot):
    """shot(nt_fine, src) -> trace sampled at DT / RESAMP. Returns (causality leak, window difference)."""
    short = shot(RESAMP * NT - 1)[::RESAMP][:NT]
    long_ = shot(2 * RESAMP * NT - 1)[::RESAMP][:NT]
    t = np.arange(NT) * DT
    leak = np.max(np.abs(short[t < OFFSET / VP])) / np.max(np.abs(short))
    window = np.max(np.abs(short - long_)) / np.max(np.abs(long_))
    return leak, window


def assert_ok(name, leak, window):
    print(f"      {name}: before the P arrival {leak:.2e} (<= {CAUSAL_TOL:.0e}), "
          f"record x2 difference {window:.2e} (<= {WINDOW_TOL:.0e})")
    assert leak <= CAUSAL_TOL and window <= WINDOW_TOL, (
        f"{name}: energy before the P arrival {leak:.2e} > {CAUSAL_TOL:.0e} or record x2 difference "
        f"{window:.2e} > {WINDOW_TOL:.0e}: the convolution wraps late arrivals around to t = 0")


def test_garvin2_no_wraparound():
    for rectype in ("x", "z"):
        leak, window = check(lambda n: garvin2.compute_shot(
            np.array([OFFSET]), VP, VS, RHO, DT / RESAMP, ricker(n, DT / RESAMP), rectype=rectype,
            zsrc=20.0, zrec=20.0)[:, 0])
        assert_ok(f"garvin2 v{rectype}", leak, window)


def test_lamb3D_no_wraparound():
    for rectype in ("x", "z"):
        leak, window = check(lambda n: lamb3D.compute_shot(
            np.array([OFFSET]), VP, VS, RHO, DT / RESAMP, ricker(n, DT / RESAMP), srctype="x",
            rectype=rectype, linedir="x")[:, 0])
        assert_ok(f"lamb3D v{rectype} (x force)", leak, window)


if __name__ == "__main__":
    sys.exit(run_tests([test_garvin2_no_wraparound, test_lamb3D_no_wraparound]) > 0)
