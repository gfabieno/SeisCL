"""Adjoint (transpose) test for the material-parameter averaging operators.

SeisCL evaluates the FD physics at staggered, averaged material parameters
(rip/rjp/rkp from the buoyancy, muipkp/muipjp/mujpkp from mu, tausipkp/... from
taus) but accumulates the FWI gradient into the cell-centred slots. Mapping a
gradient from the averaged parameters back to the cell-centred ones requires
the transpose of the averaging operator's Jacobian -- see
notes/material-averaging-gradient-review.md.

This file is the executable specification of that transpose, in the style of
dot_prod_surface.py: numpy reimplementations of the three averaging routines in
src/assign_modeling_case.c, their analytic Jacobian-transposes, and a dot test

    <J(p) db, y> == <db, J(p)^T y>

with J(p) db obtained by a central finite difference of the forward operator,
so the two sides share no code. Run directly:

    python dot_prod_average.py
"""

import numpy as np

# ---------------------------------------------------------------------------
# Forward operators: transcribed from src/assign_modeling_case.c.
#
# Index convention there is ind = i*NY*NZ + j*NZ + k with (NZ,NY,NX) =
# (N[0],N[1],N[2]) in 3D and NY==1 in 2D, i.e. a C-order (NX,NY,NZ) array.
# `dir` entries are offsets in (k,j,i) = (z,y,x) order.
# ---------------------------------------------------------------------------


def _shape(N):
    """(NX, NY, NZ) for the C indexing above, from SeisCL's N = (NZ[,NY],NX)."""
    if len(N) == 3:
        return N[2], N[1], N[0]
    return N[1], 1, N[0]


def _roll_view(a, d):
    """a shifted by dir offset d=(dz,dy,dx), as a[i+dx, j+dy, k+dz]."""
    dz, dy, dx = d[0], d[1], d[2]
    NX, NY, NZ = a.shape
    out = np.zeros_like(a)
    out[:NX - dx, :NY - dy, :NZ - dz] = a[dx:NX or None, dy:NY or None,
                                          dz:NZ or None][
        :NX - dx, :NY - dy, :NZ - dz]
    return out


def _interior_mask(shape, dirs):
    """True where the averaging formula applies (the C loops' bounds)."""
    NX, NY, NZ = shape
    sz = sum(d[0] for d in dirs)
    sy = sum(d[1] for d in dirs)
    sx = sum(d[2] for d in dirs)
    m = np.zeros(shape, dtype=bool)
    m[:NX - sx, :NY - sy, :NZ - sz] = True
    return m


def ave_arithmetic_rho(pin, N, d):
    """2-point harmonic mean of the buoyancy = arithmetic mean of density.

    pout = 2/(1/p1 + 1/p2); the trailing row/column in the averaged direction
    is copied through unchanged.
    """
    a = pin.reshape(_shape(N))
    p2 = _roll_view(a, d)
    m = _interior_mask(a.shape, [d])
    out = a.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = 2.0 / (1.0 / a + 1.0 / p2)
    out[m] = avg[m]
    return out.ravel()


def ave_harmonic_mu(pin, N, dirs):
    """4-point harmonic mean of mu, zeroed if any contributor is vacuum."""
    a = pin.reshape(_shape(N))
    d0, d1 = dirs
    dsum = [d0[i] + d1[i] for i in range(3)]
    p2, p3, p4 = _roll_view(a, d0), _roll_view(a, d1), _roll_view(a, dsum)
    m = _interior_mask(a.shape, dirs)
    out = a.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = 4.0 / (1.0 / a + 1.0 / p2 + 1.0 / p3 + 1.0 / p4)
    avg = np.where((a == 0) | (p2 == 0) | (p3 == 0) | (p4 == 0), 0.0, avg)
    out[m] = avg[m]
    return out.ravel()


def ave_arithmetic_tau(pin, N, dirs):
    """4-point arithmetic mean."""
    a = pin.reshape(_shape(N))
    d0, d1 = dirs
    dsum = [d0[i] + d1[i] for i in range(3)]
    p2, p3, p4 = _roll_view(a, d0), _roll_view(a, d1), _roll_view(a, dsum)
    m = _interior_mask(a.shape, dirs)
    out = a.copy()
    avg = 0.25 * (a + p2 + p3 + p4)
    out[m] = avg[m]
    return out.ravel()


# ---------------------------------------------------------------------------
# Jacobian transposes. y lives on the averaged (staggered) grid; the result
# lives on the cell-centred grid. Written as a scatter, which is how the
# derivation reads; the engine will do the equivalent gather.
# ---------------------------------------------------------------------------


def _scatter(acc, contrib, d):
    """acc[i+dx, j+dy, k+dz] += contrib, over the valid region."""
    dz, dy, dx = d[0], d[1], d[2]
    NX, NY, NZ = acc.shape
    acc[dx:, dy:, dz:] += contrib[:NX - dx, :NY - dy, :NZ - dz]


def ave_arithmetic_rho_T(y, pin, N, d):
    """d(rip)/d(b_j) = (rip^2/2)/b_j^2, and 1 on the copied trailing rows."""
    a = pin.reshape(_shape(N))
    yy = y.reshape(_shape(N))
    p2 = _roll_view(a, d)
    m = _interior_mask(a.shape, [d])
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = 2.0 / (1.0 / a + 1.0 / p2)
        j1 = np.where(m, 0.5 * avg ** 2 / a ** 2, 0.0)
        j2 = np.where(m, 0.5 * avg ** 2 / p2 ** 2, 0.0)
    out = np.zeros_like(a)
    out += np.where(m, yy * j1, 0.0)          # contribution to the cell itself
    _scatter(out, yy * j2, d)                 # ... and to its partner
    out += np.where(m, 0.0, yy)               # copied region: Jacobian 1
    return out.ravel()


def ave_harmonic_mu_T(y, pin, N, dirs):
    """d(muipkp)/d(mu_j) = (muipkp^2/4)/mu_j^2, all four zero in a vacuum."""
    a = pin.reshape(_shape(N))
    yy = y.reshape(_shape(N))
    d0, d1 = dirs
    dsum = [d0[i] + d1[i] for i in range(3)]
    ps = [a, _roll_view(a, d0), _roll_view(a, d1), _roll_view(a, dsum)]
    m = _interior_mask(a.shape, dirs)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = 4.0 / sum(1.0 / p for p in ps)
    vac = (ps[0] == 0) | (ps[1] == 0) | (ps[2] == 0) | (ps[3] == 0)
    live = m & ~vac
    out = np.zeros_like(a)
    for p, d in zip(ps, [[0, 0, 0], d0, d1, dsum]):
        with np.errstate(divide="ignore", invalid="ignore"):
            j = np.where(live, 0.25 * avg ** 2 / p ** 2, 0.0)
        if d == [0, 0, 0]:
            out += yy * j
        else:
            _scatter(out, yy * j, d)
    out += np.where(m, 0.0, yy)               # copied region
    # A vacuum cell inside the averaged region contributes nothing, but the
    # averaged value is still defined (zero) -- no gradient flows either way.
    return out.ravel()


def ave_arithmetic_tau_T(y, pin, N, dirs):
    """d(tausipkp)/d(taus_j) = 0.25."""
    a = pin.reshape(_shape(N))
    yy = y.reshape(_shape(N))
    d0, d1 = dirs
    dsum = [d0[i] + d1[i] for i in range(3)]
    m = _interior_mask(a.shape, dirs)
    out = np.zeros_like(a)
    for d in [[0, 0, 0], d0, d1, dsum]:
        c = np.where(m, 0.25 * yy, 0.0)
        if d == [0, 0, 0]:
            out += c
        else:
            _scatter(out, c, d)
    out += np.where(m, 0.0, yy)
    return out.ravel()


# ---------------------------------------------------------------------------
# Dot test
# ---------------------------------------------------------------------------


def dot_test(fwd, transpose, p, db, y, eps=1e-6):
    """<J db, y> vs <db, J^T y>, with J db by central difference of fwd."""
    jdb = (fwd(p + eps * db) - fwd(p - eps * db)) / (2.0 * eps)
    lhs = float(jdb @ y)
    rhs = float(db @ transpose(y, p))
    denom = max(abs(lhs), abs(rhs), 1e-300)
    return lhs, rhs, abs(lhs - rhs) / denom


CASES = []


def _case(name, fwd, tr, N, dirs, vacuum=False, seed=0):
    rng = np.random.default_rng(seed)
    n = int(np.prod(N))
    p = 1.0 + rng.random(n)                 # keep away from 0 for 1/p
    if vacuum:
        p[n // 3:n // 3 + 5] = 0.0          # a vacuum patch
    db = rng.standard_normal(n)
    if vacuum:
        db[p == 0] = 0.0                    # do not perturb off the kink
    y = rng.standard_normal(n)
    CASES.append((name, lambda q: fwd(q, N, dirs), lambda yy, q:
                  tr(yy, q, N, dirs), p, db, y))


# 2D, N = (NZ, NX)
N2 = (7, 5)
_case("rip   (2D, x)", ave_arithmetic_rho, ave_arithmetic_rho_T, N2, [0, 0, 1])
_case("rkp   (2D, z)", ave_arithmetic_rho, ave_arithmetic_rho_T, N2, [1, 0, 0])
_case("muipkp(2D)", ave_harmonic_mu, ave_harmonic_mu_T, N2,
      [[0, 0, 1], [1, 0, 0]])
_case("muipkp(2D, vacuum)", ave_harmonic_mu, ave_harmonic_mu_T, N2,
      [[0, 0, 1], [1, 0, 0]], vacuum=True)
_case("tausipkp(2D)", ave_arithmetic_tau, ave_arithmetic_tau_T, N2,
      [[0, 0, 1], [1, 0, 0]])

# 3D, N = (NZ, NY, NX)
N3 = (5, 4, 3)
_case("rip   (3D, x)", ave_arithmetic_rho, ave_arithmetic_rho_T, N3, [0, 0, 1])
_case("rjp   (3D, y)", ave_arithmetic_rho, ave_arithmetic_rho_T, N3, [0, 1, 0])
_case("rkp   (3D, z)", ave_arithmetic_rho, ave_arithmetic_rho_T, N3, [1, 0, 0])
_case("muipkp(3D)", ave_harmonic_mu, ave_harmonic_mu_T, N3,
      [[0, 0, 1], [1, 0, 0]])
_case("muipjp(3D)", ave_harmonic_mu, ave_harmonic_mu_T, N3,
      [[0, 0, 1], [0, 1, 0]])
_case("mujpkp(3D)", ave_harmonic_mu, ave_harmonic_mu_T, N3,
      [[0, 1, 0], [1, 0, 0]])


def _literal_arithmetic_rho(pin, N, d):
    """Loop-for-loop transcription of ave_arithmetic_rho, for cross-checking.

    A dot test is self-consistent: it would pass just as happily against a
    mis-transcribed forward operator. This deliberately mirrors the C control
    flow (including the trailing copy loop's NX0/NY0/NZ0 logic) rather than
    the vectorized form above, so the two can only agree if the indexing is
    right.
    """
    NX, NY, NZ = _shape(N)
    a = pin.reshape(NX, NY, NZ)
    out = np.zeros_like(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(NZ - d[0]):
            for j in range(NY - d[1]):
                for i in range(NX - d[2]):
                    # 1/0 -> inf, so a vacuum contributor drives the mean to
                    # 0, matching the C (which has no explicit guard here).
                    out[i, j, k] = 2.0 / (
                        1.0 / a[i, j, k]
                        + 1.0 / a[i + d[2], j + d[1], k + d[0]])
    NX0 = NY0 = NZ0 = 0
    if d[2] == 1:
        NX0 = NX - 1
    elif d[1] == 1:
        NY0 = NY - 1
    if d[0] == 1:
        NZ0 = NZ - 1
    for k in range(NZ0, NZ):
        for j in range(NY0, NY):
            for i in range(NX0, NX):
                out[i, j, k] = a[i, j, k]
    return out.ravel()


def _literal_harmonic_mu(pin, N, dirs):
    """Loop-for-loop transcription of ave_harmonic_mu."""
    NX, NY, NZ = _shape(N)
    a = pin.reshape(NX, NY, NZ)
    d0, d1 = dirs
    out = np.zeros_like(a)
    for k in range(NZ - d0[0] - d1[0]):
        for j in range(NY - d0[1] - d1[1]):
            for i in range(NX - d0[2] - d1[2]):
                q = [a[i, j, k],
                     a[i + d0[2], j + d0[1], k + d0[0]],
                     a[i + d1[2], j + d1[1], k + d1[0]],
                     a[i + d0[2] + d1[2], j + d0[1] + d1[1], k + d0[0] + d1[0]]]
                if min(q) == 0.0:
                    out[i, j, k] = 0.0
                else:
                    out[i, j, k] = 4.0 / sum(1.0 / v for v in q)
    for d in (d0, d1):
        NX0 = NY0 = NZ0 = 0
        if d[2] == 1:
            NX0 = NX - 1
        elif d[1] == 1:
            NY0 = NY - 1
        if d[0] == 1:
            NZ0 = NZ - 1
        for k in range(NZ0, NZ):
            for j in range(NY0, NY):
                for i in range(NX0, NX):
                    out[i, j, k] = a[i, j, k]
    return out.ravel()


def test_forward_matches_literal_transcription():
    """The vectorized operators equal a literal loop transcription of the C."""
    rng = np.random.default_rng(7)
    worst = 0.0
    checks = [
        ("rip 2D", ave_arithmetic_rho, _literal_arithmetic_rho, N2, [0, 0, 1]),
        ("rkp 2D", ave_arithmetic_rho, _literal_arithmetic_rho, N2, [1, 0, 0]),
        ("rip 3D", ave_arithmetic_rho, _literal_arithmetic_rho, N3, [0, 0, 1]),
        ("rjp 3D", ave_arithmetic_rho, _literal_arithmetic_rho, N3, [0, 1, 0]),
        ("rkp 3D", ave_arithmetic_rho, _literal_arithmetic_rho, N3, [1, 0, 0]),
        ("muipkp 2D", ave_harmonic_mu, _literal_harmonic_mu, N2,
         [[0, 0, 1], [1, 0, 0]]),
        ("muipkp 3D", ave_harmonic_mu, _literal_harmonic_mu, N3,
         [[0, 0, 1], [1, 0, 0]]),
        ("muipjp 3D", ave_harmonic_mu, _literal_harmonic_mu, N3,
         [[0, 0, 1], [0, 1, 0]]),
        ("mujpkp 3D", ave_harmonic_mu, _literal_harmonic_mu, N3,
         [[0, 1, 0], [1, 0, 0]]),
    ]
    for name, vec, lit, N, dirs in checks:
        n = int(np.prod(N))
        p = 1.0 + rng.random(n)
        p[n // 3] = 0.0                      # exercise the vacuum branch too
        e = float(np.max(np.abs(vec(p, N, dirs) - lit(p, N, dirs))))
        print("  %-12s max|vectorized - literal| = %.3e" % (name, e))
        worst = max(worst, e)
    assert worst == 0.0, (
        "vectorized averaging disagrees with the literal transcription "
        "(worst %.3e) -- the roll/scatter indexing is wrong, and the dot "
        "test above would not have caught it." % worst)


def test_averaging_adjoint():
    """Every averaging operator's transpose passes the dot test."""
    worst = 0.0
    for name, fwd, tr, p, db, y in CASES:
        lhs, rhs, rel = dot_test(fwd, tr, p, db, y)
        print("  %-20s <Jdb,y>=% .8e  <db,J^Ty>=% .8e  rel=%.2e"
              % (name, lhs, rhs, rel))
        worst = max(worst, rel)
    assert worst < 1e-6, (
        "averaging transpose failed the dot test (worst rel=%.3e). A failure "
        "confined to one operator is its Jacobian; a failure in every case at "
        "the same magnitude is usually the trailing copied row/column, whose "
        "Jacobian is 1 rather than the averaging formula." % worst)
    print("  worst relative error: %.3e" % worst)


if __name__ == "__main__":
    test_forward_matches_literal_transcription()
    test_averaging_adjoint()
    print("dot_prod_average: PASS")
