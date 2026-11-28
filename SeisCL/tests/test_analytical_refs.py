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

Absolute amplitude and geometry, against closed-form time-domain solutions computed here independently:

  viscoelastic_3D (elastic)  Stokes' solution for a point force (Aki & Richards 2002, eq. 4.23)
  viscoelastic_2D (elastic)  the same, integrated along the line y (a line force is a line of point forces)
  viscoelastic_2D_SH         u = 1/(2 pi mu) int H(tau - r/vs) / sqrt(tau^2 - r^2/vs^2) f(t - tau) dtau
  garvin2, source and receiver at different depths: before the PP reflection arrives, the direct P wave equals the
             full-space line explosion, u_r = d/dr 1/(2 pi mu) int H(tau - r/vp) / sqrt(tau^2 - r^2/vp^2) f(t - tau)
             dtau, along the source-receiver direction (no transverse motion)

  relative RMS error <= 1e-4 (viscoelastic) and 1e-2 (garvin2, limited by its own time sampling)

garvin2 with a high vp/vs (3.5): the record must contain the Rayleigh wave, the largest arrival on vz at the
surface for a shallow source: the largest peak lies within 1 / f0 of offset / cR + 1.5 / f0 (cR from the Rayleigh
equation; peak there / peak of the trace >= 0.5). Garvin2 used to stop at 3 tPS, before the Rayleigh wave.
Usage: python SeisCL/tests/test_analytical_refs.py   (from the repository root, PYTHONPATH=.)
"""
import sys
import numpy as np

from SeisCL.analytical.viscoelastic import viscoelastic_2D, viscoelastic_3D, viscoelastic_2D_SH

from SeisCL.analytical import garvin2, lamb3D
from SeisCL.tests.gradient_common import run_tests

VP, VS, RHO, F0 = 3500.0, 2000.0, 2000.0, 20.0
DT, RESAMP, NT = 0.25e-3, 10, 1200           # record 0.3 s
OFFSET = 400.0                                # Rayleigh wave (~0.92 vs) peaks at 1.5/f0 + 400/1840 = 0.292 s
CAUSAL_TOL, WINDOW_TOL = 1e-3, 1e-6
AMP_TOL, GARVIN_TOL = 1e-4, 1e-2


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


# ---- closed-form time-domain references (Ricker of centre frequency f0, peak at tc = 1.5 / f0) ----
def rick(t, f0):
    s = np.pi * f0 * (t - 1.5 / f0)
    return (1 - 2 * s ** 2) * np.exp(-s ** 2)


def rick_int(t, f0):        # its integral from -infinity
    s = np.pi * f0 * (t - 1.5 / f0)
    return s * np.exp(-s ** 2) / (np.pi * f0)


def rick_dt(t, f0):         # its time derivative
    s = np.pi * f0 * (t - 1.5 / f0)
    return np.pi * f0 * np.exp(-s ** 2) * (4 * s ** 3 - 6 * s)


def stokes_velocity(x, y, z, t, f0):
    """Velocity (vx, vz) of a z point force f(t) = rick (N) in a full space, Aki & Richards eq. 4.23 applied to df/dt."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    a, b = r / VP, r / VS
    gx, gz = x / r, z / r
    # int_a^b tau f'(t - tau) dtau, by parts
    near = a * rick(t - a, f0) - b * rick(t - b, f0) + rick_int(t - a, f0) - rick_int(t - b, f0)
    c = 1 / (4 * np.pi * RHO)
    vx = c * (3 * gx * gz / r ** 3 * near + gx * gz / (VP ** 2 * r) * rick_dt(t - a, f0)
              - gx * gz / (VS ** 2 * r) * rick_dt(t - b, f0))
    vz = c * ((3 * gz * gz - 1) / r ** 3 * near + gz * gz / (VP ** 2 * r) * rick_dt(t - a, f0)
              - (gz * gz - 1) / (VS ** 2 * r) * rick_dt(t - b, f0))
    return vx, vz


def line_force_velocity(x, z, t, f0, dy=0.5):
    """2-D line force (N/m): the point-force solution integrated over y (trapezoid rule, y >= 0 counted twice)."""
    y = np.arange(0, VP * t[-1] + 10, dy)[:, None]
    weight = np.full(y.shape, 2 * dy)
    weight[0] = dy
    vx, vz = np.zeros_like(t), np.zeros_like(t)
    for chunk in np.array_split(np.arange(len(y)), 50):
        ax, az = stokes_velocity(x, y[chunk], z, t[None, :], f0)
        vx += np.sum(weight[chunk] * ax, 0)
        vz += np.sum(weight[chunk] * az, 0)
    return vx, vz


def line_integral(fun, r, v, t, nw=4000):
    """int_{r/v}^t fun(t - tau) / sqrt(tau^2 - r^2/v^2) dtau, with tau = (r/v) cosh(w) to remove the singularity."""
    out = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti * v > r:
            w = np.linspace(0, np.arccosh(ti * v / r), nw)
            out[i] = np.trapezoid(fun(ti - r / v * np.cosh(w)), w)
    return out


def rms(a, b):
    return np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2))


def test_viscoelastic_absolute_amplitude():
    f0, dt, nt = 10.0, 1e-3, 1000
    t = np.arange(nt) * dt
    src = rick(np.arange(4 * nt) * dt, f0)          # long record: the 2-D tail does not wrap around
    recs = [[300.0, 0.0, 150.0], [0.0, 0.0, 500.0]]
    vx2, vz2 = viscoelastic_2D(VP, VS, RHO, 0, 0, f0, np.array([f0]), dt, recs, src)
    _, _, vz3 = viscoelastic_3D(VP, VS, RHO, 0, 0, f0, np.array([f0]), dt, recs, src)
    vy = viscoelastic_2D_SH(VS, RHO, 0, f0, [f0], dt, recs, src)
    worst = 0
    for j, (x, _, z) in enumerate(recs):
        ex3 = stokes_velocity(x, 0.0, z, t, f0)[1]
        ex2x, ex2z = line_force_velocity(x, z, t, f0)
        exsh = line_integral(lambda s: rick_dt(s, f0), np.hypot(x, z), VS, t) / (2 * np.pi * RHO * VS ** 2)
        pairs = [("3-D vz", np.real(vz3[:nt, j]), ex3), ("2-D vz", np.real(vz2[:nt, j]), ex2z),
                 ("2-D SH vy", vy[:nt, j], exsh)]
        if x and z:
            pairs.append(("2-D vx", np.real(vx2[:nt, j]), ex2x))
        for name, num, ex in pairs:
            err = rms(num, ex)
            worst = max(worst, err)
            print(f"      receiver ({x:.0f}, {z:.0f}) {name}: RMS error {err:.2e} (<= {AMP_TOL:.0e}), "
                  f"amplitude ratio {np.dot(num, ex) / np.dot(ex, ex):.4f}")
    assert worst <= AMP_TOL, f"viscoelastic analytical solution: RMS error {worst:.2e} > {AMP_TOL:.0e}"


def test_garvin2_source_receiver_depths():
    x, zsrc, zrec = 400.0, 1000.0, 500.0
    r1, r2 = np.hypot(x, zrec - zsrc), np.hypot(x, zrec + zsrc)
    mu = VS ** 2 * RHO
    nt = 2000
    src = ricker(RESAMP * nt - 1, DT / RESAMP)
    shot = [garvin2.compute_shot(np.array([x]), VP, VS, RHO, DT / RESAMP, src, rectype=c, zsrc=zsrc,
                                 zrec=zrec)[::RESAMP, 0][:nt] for c in ("x", "z")]
    radial = (shot[0] * x + shot[1] * (zrec - zsrc)) / r1
    transverse = (shot[0] * (zrec - zsrc) - shot[1] * x) / r1
    t = np.arange(nt) * DT
    phi = lambda r: line_integral(lambda s: rick(s, F0), r, VP, t)
    exact = (phi(r1 + 0.05) - phi(r1 - 0.05)) / 0.1 / (2 * np.pi * mu)
    w = t < r2 / VP                                  # before the PP reflection
    err = rms(radial[w], exact[w])
    tr = np.max(np.abs(transverse[w])) / np.max(np.abs(exact[w]))
    print(f"      direct P, source {zsrc:.0f} m, receiver {zrec:.0f} m deep, offset {x:.0f} m: radial RMS error "
          f"{err:.2e} (<= {GARVIN_TOL:.0e}), transverse / radial {tr:.2e} (<= {GARVIN_TOL:.0e})")
    assert err <= GARVIN_TOL and tr <= GARVIN_TOL, (
        f"garvin2 with zsrc != zrec: radial error {err:.2e} or transverse {tr:.2e} > {GARVIN_TOL:.0e}")


def test_garvin2_rayleigh_high_vpvs():
    vp, x, zsrc, zrec = 3.5 * VS, 1200.0, 50.0, 5.0
    ratio = vp / VS
    rayleigh = lambda c: (2 - c ** 2) ** 2 - 4 * np.sqrt(1 - c ** 2) * np.sqrt(1 - c ** 2 / ratio ** 2)
    lo, hi = 0.5, 0.999                              # bisection for cR / vs
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rayleigh(lo) * rayleigh(mid) > 0 else (lo, mid)
    cr = 0.5 * (lo + hi) * VS
    nt = 4000                                        # 1 s
    src = ricker(RESAMP * nt - 1, DT / RESAMP)
    vz = garvin2.compute_shot(np.array([x]), vp, VS, RHO, DT / RESAMP, src, rectype="z", zsrc=zsrc,
                              zrec=zrec)[::RESAMP, 0][:nt]
    t = np.arange(nt) * DT
    window = np.abs(t - (x / cr + 1.5 / F0)) < 1.0 / F0
    share = np.max(np.abs(vz[window])) / np.max(np.abs(vz))
    print(f"      vp/vs {ratio:g}, source {zsrc:.0f} m, receiver {zrec:.0f} m deep, offset {x:.0f} m: "
          f"peak in the Rayleigh window / peak of the trace {share:.3f} (>= 0.5)")
    assert share >= 0.5, f"garvin2: no Rayleigh wave at vp/vs {ratio:g} ({share:.3f} < 0.5)"

if __name__ == "__main__":
    sys.exit(run_tests([test_garvin2_no_wraparound, test_lamb3D_no_wraparound,
                        test_viscoelastic_absolute_amplitude, test_garvin2_source_receiver_depths,
                        test_garvin2_rayleigh_high_vpvs]) > 0)
