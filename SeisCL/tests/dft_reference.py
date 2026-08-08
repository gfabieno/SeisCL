"""Float64 numpy reference for the frequency-domain (back_prop_type=2) gradient.

Transcribed from src/calc_grad.c: grad_coefelast_0() (:154-169) for the
coefficients and the ND==2 correlation block (:745-860) for the dot products.
Elastic only (L==0), 2D P-SV, param_type=0.

This is both the acceptance test for the device-side correlation kernel and its
executable specification.

Conventions worth stating explicitly, because they are easy to get wrong:

  cl_itreal(a, b) = a.im*b.re - a.re*b.im  (calc_grad.c:31-36)
                  = Im(a * conj(b))

  freq = 2*pi*df*gradfreqsn[f],  df = 1/(NTNYQ*dt*DTNYQ)   (calc_grad.c:405)

  Every dot product carries a 1/NTNYQ factor.
"""

import numpy as np


def grad_coef_elast_0(M, mu, rho, ND=2.0):
    """grad_coefelast_0 (calc_grad.c:154-169), vectorized.

    M, mu, rho are the *stiffnesses*, i.e. M = rho*vp**2 and mu = rho*vs**2.
    Returns the 8 coefficients that are nonzero for L==0, param_type=0. The c
    indices are kept as in the C code so the two can be diffed side by side.
    """
    M = np.asarray(M, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    den = (ND * M - 2.0 * (ND - 1.0) * mu) ** 2

    c = {}
    c[0] = 2.0 * np.sqrt(rho * M) / den
    c[2] = 2.0 * np.sqrt(rho * mu) / (mu * mu)
    c[3] = 2.0 * np.sqrt(rho * mu) * (ND + 1.0) / 3.0 / den
    c[4] = 2.0 * np.sqrt(rho * mu) / (2.0 * ND * mu * mu)
    c[16] = M / rho / den
    c[18] = mu / rho / (mu * mu)
    c[19] = mu / rho * (ND + 1.0) / 3.0 / den
    c[20] = mu / rho / (2.0 * ND * mu * mu)
    return c


def _itreal(a, b):
    """cl_itreal: Im(a * conj(b))."""
    return np.imag(a * np.conj(b))


def gradient_2d_elastic(fwd, adj, M, mu, rho, bins, ntnyq, dtnyq, dt,
                        fdoh, nz, nx, ND=2.0):
    """Reference (gradvp, gradvs, gradrho) from the dumped DFT spectra.

    :param fwd: dict var -> (NZpad, NXpad, NFREQS) forward spectrum
    :param adj: dict var -> same, adjoint spectrum
    :param M, mu, rho: physical stiffness/density arrays, shape (nz, nx)
    :param bins: gradfreqsn, integer DFT bin indices
    :return: dict with 'vp', 'vs', 'rho', each (nz, nx)
    """
    df = 1.0 / ntnyq / dt / dtnyq
    # Parseval factor is 1/(NTNYQ*DTNYQ), not 1/NTNYQ -- see calc_grad.c.
    dftnorm = float(ntnyq) * float(dtnyq)

    # The correlation is evaluated at the parameter the physics actually uses:
    # sxz is driven by muipkp and the two velocity components by rip/rkp
    # (update_s2D.cl / update_v2D.cl). The averaging and its transpose come
    # from dot_prod_average, which dot-tests them independently of anything
    # here -- that is the part of this reference that is *not* a transcription
    # of the kernel. Both operators are scale invariant (a harmonic mean and
    # the ratio muipkp^2/mu_j^2), so running them on physical values rather
    # than the engine's internally scaled ones gives the same answer.
    from dot_prod_average import (ave_harmonic_mu, ave_arithmetic_rho,
                                  ave_harmonic_mu_T, ave_arithmetic_rho_T)
    N2 = (nz, nx)
    DIR_IPKP = [[0, 0, 1], [1, 0, 0]]
    DIR_IP = [0, 0, 1]
    DIR_KP = [1, 0, 0]

    def _fwd(op, arr, dirs):
        return op(np.asarray(arr, dtype=np.float64).T.ravel(),
                  N2, dirs).reshape(nx, nz).T

    def _T(op, y, arr, dirs):
        return op(np.asarray(y, dtype=np.float64).T.ravel(),
                  np.asarray(arr, dtype=np.float64).T.ravel(),
                  N2, dirs).reshape(nx, nz).T

    muipkp = _fwd(ave_harmonic_mu, mu, DIR_IPKP)
    buoy = 1.0 / rho
    with np.errstate(divide="ignore", invalid="ignore"):
        imuipkp2 = np.where(muipkp > 0, 1.0 / (muipkp * muipkp), 0.0)

    den = (ND * M - 2.0 * (ND - 1.0) * mu) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        iden = np.where(den > 0, 1.0 / den, 0.0)
        imu2 = np.where(mu > 0, 1.0 / (mu * mu), 0.0)
    i3den = (ND + 1.0) / 3.0 * iden
    i2ndmu2 = imu2 / (2.0 * ND)

    sl = (slice(fdoh, fdoh + nz), slice(fdoh, fdoh + nx))

    def F(v):
        return fwd[v][sl].astype(np.complex128)

    def A(v):
        return adj[v][sl].astype(np.complex128)

    gM = np.zeros((nz, nx), dtype=np.float64)
    gmu = np.zeros((nz, nx), dtype=np.float64)
    gmuipkp = np.zeros((nz, nx), dtype=np.float64)
    grip = np.zeros((nz, nx), dtype=np.float64)
    grkp = np.zeros((nz, nx), dtype=np.float64)

    for j, b in enumerate(bins):
        w = 2.0 * np.pi * df * float(b)

        fsxx, fszz, fsxz = F("sxx")[..., j], F("szz")[..., j], F("sxz")[..., j]
        asxx, aszz, asxz = A("sxx")[..., j], A("szz")[..., j], A("sxz")[..., j]
        fvx, fvz = F("vx")[..., j], F("vz")[..., j]
        avx, avz = A("vx")[..., j], A("vz")[..., j]

        sxxzz = fsxx + fszz
        sxxzzr = asxx + aszz
        sxx_mzz = fsxx - fszz
        szz_mxx = fszz - fsxx

        d0 = w * _itreal(sxxzzr, sxxzz) / dftnorm
        d2 = w * _itreal(asxz, fsxz) / dftnorm
        d3 = d0
        d4 = w * (_itreal(asxx, sxx_mzz) + _itreal(aszz, szz_mxx)) / dftnorm
        # vx sits at the rip position, vz at rkp: different parameters, so the
        # two must not be summed before the correlation is stored.
        d8x = w * _itreal(avx, fvx) / dftnorm
        d8z = w * _itreal(avz, fvz) / dftnorm

        gM += -d0 * iden
        gmu += d3 * i3den - d4 * i2ndmu2      # sxx/szz: cell-centred mu
        gmuipkp += -d2 * imuipkp2             # sxz: the averaged mu
        grip += -d8x
        grkp += -d8z

    # Averaging transpose: fold the staggered gradients back onto the
    # cell-centred parameters. Density enters the physics only through
    # rip/rkp, so grho has no term of its own.
    gmu = gmu + _T(ave_harmonic_mu_T, gmuipkp, mu, DIR_IPKP)
    grho = (_T(ave_arithmetic_rho_T, grip, buoy, DIR_IP)
            + _T(ave_arithmetic_rho_T, grkp, buoy, DIR_KP))

    # Parameterization chain rule, (M, mu, rho) -> (vp, vs, rho), as
    # chain_rule_par_type does on the host.
    with np.errstate(divide="ignore", invalid="ignore"):
        irho = np.where(rho > 0, 1.0 / rho, 0.0)
    gvp = 2.0 * np.sqrt(rho * M) * gM
    gvs = 2.0 * np.sqrt(rho * mu) * gmu
    gvrho = grho + M * irho * gM + mu * irho * gmu
    return {"vp": gvp, "vs": gvs, "rho": gvrho}


def gradient_3d_elastic(fwd, adj, M, mu, rho, bins, ntnyq, dtnyq, dt,
                        fdoh, nz, ny, nx, ND=3.0):
    """3D extension of gradient_2d_elastic. Same coefficients (already generic
    in ND), dot products extended to the extra field components (vy, syy,
    sxy, syz), transcribed from src/grad_dft3D.cl / calc_grad.c's ND==3
    branch. cl_diff(a,b,c) = a-b-c (calc_grad.c:46): each of
    sxx_myyzz/syy_mxxzz/szz_mxxyy drops its own component from the sum of all
    three.

    :param fwd: dict var -> (NZpad, NYpad, NXpad, NFREQS) forward spectrum
    :param adj: dict var -> same, adjoint spectrum
    :param M, mu, rho: physical stiffness/density arrays, shape (nz, ny, nx)
    :param bins: gradfreqsn, integer DFT bin indices
    :return: dict with 'vp', 'vs', 'rho', each (nz, ny, nx)
    """
    df = 1.0 / ntnyq / dt / dtnyq
    dftnorm = float(ntnyq) * float(dtnyq)
    # Internal (M, mu, rho) coefficients: grad_coef_elast_0 with the
    # parameterization chain rule factored out, matching grad_dft3D.cl and
    # calc_grad.c's _1 family. The chain rule is applied once at the end.
    den = (ND * M - 2.0 * (ND - 1.0) * mu) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        iden = np.where(den > 0, 1.0 / den, 0.0)
        imu2 = np.where(mu > 0, 1.0 / (mu * mu), 0.0)
    i3den = (ND + 1.0) / 3.0 * iden
    i2ndmu2 = imu2 / (2.0 * ND)

    sl = (slice(fdoh, fdoh + nz), slice(fdoh, fdoh + ny), slice(fdoh, fdoh + nx))

    def F(v):
        return fwd[v][sl].astype(np.complex128)

    def A(v):
        return adj[v][sl].astype(np.complex128)

    gM = np.zeros((nz, ny, nx), dtype=np.float64)
    gmu = np.zeros((nz, ny, nx), dtype=np.float64)
    grho = np.zeros((nz, ny, nx), dtype=np.float64)

    for j, b in enumerate(bins):
        w = 2.0 * np.pi * df * float(b)

        fsxx, fsyy, fszz = F("sxx")[..., j], F("syy")[..., j], F("szz")[..., j]
        fsxy, fsxz, fsyz = F("sxy")[..., j], F("sxz")[..., j], F("syz")[..., j]
        asxx, asyy, aszz = A("sxx")[..., j], A("syy")[..., j], A("szz")[..., j]
        asxy, asxz, asyz = A("sxy")[..., j], A("sxz")[..., j], A("syz")[..., j]
        fvx, fvy, fvz = F("vx")[..., j], F("vy")[..., j], F("vz")[..., j]
        avx, avy, avz = A("vx")[..., j], A("vy")[..., j], A("vz")[..., j]

        sppp = fsxx + fsyy + fszz
        spppr = asxx + asyy + aszz
        sxx_myyzz = fsxx - fsyy - fszz
        syy_mxxzz = fsyy - fsxx - fszz
        szz_mxxyy = fszz - fsxx - fsyy

        d0 = w * _itreal(spppr, sppp) / dftnorm
        d2 = w * (_itreal(asxy, fsxy) + _itreal(asxz, fsxz)
                  + _itreal(asyz, fsyz)) / dftnorm
        d3 = d0
        d4 = w * (_itreal(asxx, sxx_myyzz) + _itreal(asyy, syy_mxxzz)
                  + _itreal(aszz, szz_mxxyy)) / dftnorm
        d8 = w * (_itreal(avx, fvx) + _itreal(avy, fvy)
                  + _itreal(avz, fvz)) / dftnorm

        gM += -d0 * iden
        gmu += -d2 * imu2 + d3 * i3den - d4 * i2ndmu2
        grho += -d8

    # Parameterization chain rule, (M, mu, rho) -> (vp, vs, rho), as
    # chain_rule_par_type does on the host.
    with np.errstate(divide="ignore", invalid="ignore"):
        irho = np.where(rho > 0, 1.0 / rho, 0.0)
    return {"vp": 2.0 * np.sqrt(rho * M) * gM,
            "vs": 2.0 * np.sqrt(rho * mu) * gmu,
            "rho": grho + M * irho * gM + mu * irho * gmu}
