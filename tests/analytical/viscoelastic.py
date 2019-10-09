# -*- coding: utf-8 -*-
"""Analytical solution to a (visco)elastic infinite isotropic space in 2D or 3D.

This module contains 3 functions:
    -complex_modulus: Routine to compute the complexe modulus with the GSLS
    -viscoelastic_2D: Solution to the 2D problem
    -viscoelastic_3D: Solution to the 3D problem
"""

import numpy as np
from scipy.special import hankel2


def complex_modulus(vp, vs, rho, taup, taus, omegaL, omega, omega0):
    """
     Complex modulus given by the Generalized standard linear solid.
     As implemented in:

         Fabien-Ouellet, G., E. Gloaguen and B. Giroux (2017).
         "Time-domain seismic modeling in viscoelastic media for full waveform
         inversion on heterogeneous computing platforms with OpenCL."
         Computers & Geosciences 100: 142-155.

    Further description can be found in:

     Bohlen, T. (2002). "Parallel 3-D viscoelastic finite difference seismic
     modelling." Computers & Geosciences 28(8): 887-899.

     Blanch, J. O., J. O. A. Robertsson and W. W. Symes (1995).
     "Modeling of a constantQ: Methodology and algorithm for an efficient and
      optimally inexpensive viscoelastic technique." Geophysics 60(1): 176-184.

    Args:
        vp (float): Vp velocity in m/s
        vs (float): Vs velocity in m/s
        rho (float): Density in kg/m3
        taup (float): P-wave attenuation level
        taus (float):  S-wave attenuation level
        omegaL (list): Center frequencies of each Maxwell body
        omega: (float): Frequency at which the modulus are computed
        omega0 (float): Center frequency of the source

    Returns:
        nu (float) Complex P-wave modulus
        mu (float) Complex shear wave modulus
    """

    sumi = 0.0
    sumr = 0.0
    for l in range(len(omegaL)):
        deni = 1.0 + 1.0j * omega / omegaL[l]
        denr = 1.0 + omega0 ** 2 / omegaL[l] ** 2
        sumi += 1.0j * omega / omegaL[l] / deni
        sumr += omega0 ** 2 / omegaL[l] ** 2 / denr

    nu = vp ** 2 * rho * (1.0 + sumi * taup) / (1.0 + sumr * taup)
    mu = vs ** 2 * rho * (1.0 + sumi * taus) / (1.0 + sumr * taus)

    return nu, mu

def viscoelastic_2D(vp, vs, rho, taup, taus, f0, fl, dt, rec_pos, src):
    """
    Analytic solution for a point force in the z direction in an infinite 2D
    homogeneous space for the viscoelastic half space.

    The analytic solution can be found in:

        Carcione, J. M. (2014). Wave Fields in Real Media:
        Wave Propagation in Anisotropic, Anelastic,
        Porous and Electromagnetic Media, Elsevier Science.

    The formulation for the Generalized Standard linear solide rheology can
    be found in:


    Args:
         vp (float): P-wave velocity
         vs (float): S-wave velocity
         rho (float): density
         taup (float): relaxation time for P-waves
         taus (float): relaxation time for S-waves
         f0 (float): The source center frequency
         fl (list): List of relaxation frequencies of each Maxwell body
         dt (float): time step size
         rec_pos (list): a list of [ [x,y,z] ] for each receiver position
         src (np.array): the src signal

    Returns:
        vx, vz (np.array): The particle velocities in x an z for this shot.

    """

    nt = src.shape[0]
    nrec = len(rec_pos)
    F = np.fft.fft(src)
    omega = 2*np.pi*np.fft.fftfreq(F.shape[0], dt)

    Vx = np.zeros([nt, nrec], dtype=np.complex128)
    Vz = np.zeros([nt, nrec], dtype=np.complex128)

    omegaL = 2 * np.pi * fl
    omega0 = 2 * np.pi * f0

    for ii in range(1, nt//2):

        nu, mu = complex_modulus(vp, vs, rho, taup, taus,
                                 omegaL, omega[ii], omega0)
        kp = omega[ii] / np.sqrt(nu / rho)
        ks = omega[ii] / np.sqrt(mu / rho)
        vpc = np.sqrt(nu/rho)
        vsc = np.sqrt(mu/rho)

        for jj in range(nrec):
            x, _, z = rec_pos[jj]

            r = np.sqrt(x ** 2 + z ** 2)

            G1 = -1.0j * np.pi / 2.0 * (1 / vpc ** 2 * hankel2(0, kp * r) +
                      1.0 / (omega[ii] * r * vsc) * hankel2(1, ks * r) -
                      1.0 / (omega[ii] * r * vpc) * hankel2(1, kp * r))
            G2 =  1.0j * np.pi / 2.0 * (1 / vsc ** 2 * hankel2(0, ks * r ) -
                      1.0 / (omega[ii] * r * vsc) * hankel2(1, ks * r ) +
                      1.0 / (omega[ii] * r * vpc) * hankel2(1, kp * r ))

            Vx[ii, jj] = (F[ii] / (2 * np.pi * rho) * (x * z / r ** 2)
                         * (G1 + G2) * 1j * omega[ii])
            Vz[ii, jj] = (F[ii] / (2 * np.pi * rho) * (1 / r ** 2)
                         * (z ** 2 * G1 - x ** 2 * G2) * 1j * omega[ii])

    for ii in range(1, nt//2):
        Vx[-ii, :] = np.conj(Vx[ii, :])
        Vz[-ii, :] = np.conj(Vz[ii, :])

    vx = nt * np.fft.ifft(Vx, axis=0)
    vz = nt * np.fft.ifft(Vz, axis=0)
    vx = np.abs(vx) * np.sign(np.real(vx))
    vz = np.abs(vz) * np.sign(np.real(vz))

    return vx, vz

def viscoelastic_3D(vp, vs, rho, taup, taus, f0, fl, dt, rec_pos, src):
    """
    Analytic solution for a point force in the z direction in an infinite 3D
    homogeneous space

    The analytic solution can be found in:

         Gosselin-Cliche, B., & Giroux, B. (2014).
         3D frequency-domain finite-difference viscoelastic-wave modeling
         using weighted average 27-point operators with optimal coefficients.
         Geophysics, 79(3), T169-T188. doi: 10.1190/geo2013-0368.1

    Args:
         vp (float): P-wave velocity
         vs (float): S-wave velocity
         rho (float): density
         taup (float): relaxation time for P-waves
         taus (float): relaxation time for S-waves
         f0 (float): The source center frequency
         fl (list): List of relaxation frequencies of each Maxwell body
         dt (float): time step size
         rec_pos (list): a list of [ [x,y,z] ] for each receiver position
         src (np.array): the src signal

    Returns:
        vx, vz (np.array): The particle velocities in x an z for this shot.

    """

    nt = src.shape[0]
    nrec = len(rec_pos)
    F = np.fft.fft(src)
    omega = 2*np.pi*np.fft.fftfreq(F.shape[0], dt)

    Vx = np.zeros([nt, nrec], dtype=np.complex128)
    Vy = np.zeros([nt, nrec], dtype=np.complex128)
    Vz = np.zeros([nt, nrec], dtype=np.complex128)

    omegaL = 2 * np.pi * fl
    omega0 = 2 * np.pi * f0

    for ii in range(1, nt//2):

        nu, mu = complex_modulus(vp, vs, rho, taup, taus,
                                 omegaL, omega[ii], omega0)

        kp = omega[ii] / np.sqrt(nu / rho)
        ks = omega[ii] / np.sqrt(mu / rho)

        for jj in range(nrec):
            x, y, z = rec_pos[jj]

            R = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            amp = F[ii] / (4.0 * np.pi * rho * R ** 5 * omega[ii] ** 2)

            Vx[ii, jj] = amp * x * z * (
                                        (R ** 2 * kp ** 2
                                         - 3.0 - 3.0 * 1j * R * kp)
                                        * np.exp(-1j * kp * R)

                                        + (3.0 + 3.0 * 1j * R * ks
                                           - R ** 2 * ks ** 2)
                                        * np.exp(-1j * ks * R)
                                        ) * 1j * omega[ii]

            Vy[ii, jj] = amp * y * z * (
                                        (R ** 2 * kp ** 2
                                         - 3 - 3 * 1j * R * kp)
                                        * np.exp(-1j * kp * R) +

                                        (3 + 3 * 1j * R * ks
                                         - R ** 2 * ks ** 2)
                                        * np.exp(-1j * ks * R)
                                        ) * 1j * omega[ii]

            Vz[ii, jj] = amp * (
                                (x ** 2 + y ** 2 - 2.0 * z ** 2)
                                 * (np.exp(-1j * kp * R) - np.exp(-1j * ks * R))

                                + (z ** 2 * R ** 2 * kp ** 2
                                    + 1j * (x ** 2 + y ** 2 - 2.0 * z ** 2)
                                    * R * kp) * np.exp(-1j * kp * R)

                                + ((x ** 2 + y ** 2) * R ** 2 * ks ** 2
                                    - 1j * (x ** 2 + y ** 2 - 2.0 * z ** 2)
                                    * R * ks) * np.exp(-1j * ks * R)
                                ) * 1j * omega[ii]


    for ii in range(1, nt // 2):
        Vx[-ii, :] = np.conj(Vx[ii, :])
        Vy[-ii, :] = np.conj(Vy[ii, :])
        Vz[-ii, :] = np.conj(Vz[ii, :])

    vx = nt * np.fft.ifft(Vx, axis=0)
    vy = nt * np.fft.ifft(Vy, axis=0)
    vz = nt * np.fft.ifft(Vz, axis=0)
    vx = np.abs(vx) * np.sign(np.real(vx))
    vy = np.abs(vy) * np.sign(np.real(vy))
    vz = np.abs(vz) * np.sign(np.real(vz))

    return vx, vy, vz