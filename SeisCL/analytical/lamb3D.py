# -*- coding: utf-8 -*-
"""Analytical solution to Lamb's (3D elastic half-space)

This script is a translation from Matlab code privided with he following paper:

     Kausel, E., Lamb's problem at its simplest
     Proceeding Royal Society of London, Series A, 2012

This module contains several functions:
    -Lamb_3D: The python version of the Matlab script
    -ellipint3: The python version of the Matlab function used in Lamb_3D
    -to_cartesian: Converts the solution from the cylindrical to cartesian
    -resample: Resample the provided solution to an constant time step
    -get_source_response: Obtain particle velocities from Lamb_3D solution
    -compute_shot: Main interface. Computes the solution for a shot
    -main: Calls shot to plot an example of the solution
"""

import numpy as np
import matplotlib.pyplot as plt

def Lamb_3D(pois=0.25, NoPlot=0, mu=1, rho=1, r=1, tmax=2):
    """
    Transcripted from Matlab to Python by Gabriel Fabien-Ouellet.

    ___________________________________________________________________
     Point loads suddenly applied onto the surface of a lower elastic halfspace.
     Time variation of load is a unit step function (Heaviside), which is
     applied at the origin on the surface (x=0, z=0).
     Both horizontal and vertical loads are considered.

     **********************************************************
           Version 3.3, September 6, 2012
           Copyleft by Eduardo Kausel
           MIT, Room 1-271, Cambridge, MA 02139
           kausel@mit.edu
     **********************************************************

      Input arguments:
            pois   = Poisson's ratio
            NoPlot = plots omitted if true (1). Default is false (0)
      Output arguments (cylindrical coordinates)
            T    = Dimensionless time vector, tau = t*Cs/r
                   Correction: It is the physical time
            Urx  = Radial displacement caused by a horizontal load
                   Varies as  cos(theta) with azimuth
            Utx  = Tangential displacement caused by a horizontal load
                   Varies as  -sin(theta) with azimuth
            Uzz  = Vertical displacement caused by a vertical load
                   (no variation with azimuth)
            Urz  = Radial displacement caused by a vertical load
                   (no variation with azimuth)
            Uzx  = Vertical displacement caused by a horizontal load
                   Not returned, because  Uzx = -Urz (reciprocity),
                   except that Uzx varies as  cos(theta)
      Note: All displacements are normalized by the shear modulus (mu) and
            by the range (r), i.e. Urx = urx*r*mu and so forth.
            Also, the time returned is dimensionless, i.e. T = t*Cs/r, where
            Cs is the shear wave velocity. Both of these normalizations
            are equivalent to assuming that mu=1, Cs=1, r=1.
            Observe that the first point of all arrays returned is for time
            t=0, and thus not equally spaced with the other points, which
            begin with the arrival of the P wave.

     Sign convention:
     Vertical load and vertical displacements at the surface point up
     Response given in terms of dimensionless time tau=t*Cs/r

     References:
        a) Vertical loads:
    			Eringen and Suhubi, Elastodynamics, Vol. II, 748-750, pois<0.26
     			Mooney, 1974, BSSA, V. 64, No.2, pp. 473-491, pois > 0.2631,
               but vertical component only
        b) Horizontal loads: (pois = 0.25 only)
            Chao, C.C., Dynamical response of an elastic halfspace to
            tangential surface loading, Journal of Applied Mechanics,
            Vol 27, September 1960, pp 559-567
        c) Generalization of all of the above to arbitrary Poisson's ratio:
            Kausel, E., Lamb's problem at its simplest
            Proceeding Royal Society of London, Series A, 2012 (in print)
    """
    # Default data & input parameters
    pois = np.double(pois)
    mu = np.double(mu)             # shear modulus
    r = np.double(r)             # range (= radial distance)
    rho = np.double(rho)            # mass density
    Cs = np.sqrt(mu/rho)  # Shear wave velocity
    #tmax = 2			# Maximum time for plotting (t=1 => arrival of S waves)
    Nt = 1000          # Number of time steps between tp and ts

    # Roots of Rayleigh function
    a2 = (1-2*pois)/(2 - 2*pois)  # (Cs/Cp)**2
    b2 = 1-a2
    p = [-16*(1-a2), 8*(3-2*a2), -8, 1]    # Characteristic polynomial
    x = np.sort(np.roots(p)) # find and sort roots
    x1 = x[0]  # First false root
    x2 = x[1]  # Second false root
    x3 = np.real(x[2])  # True Rayleigh root = (Cs/Cr)**2

    # Dimensionless arrival times and time arrays
    tp = np.sqrt(a2)		# Arrival time of P waves
    ts = 1				#    "      "   " S   "
    tr = np.sqrt(x3)		#    "      "   " R   "
    dt = (ts-tp)/Nt    # Time step
    T0 = [0, tp]        # Time before arrival of P wave
    U0 = [0, 0]         # Quiescent phase (during T0)
    T1 = np.arange(tp+dt, ts, dt, dtype=np.double)     # Time from P to S wave arrival
    T2 = np.arange(ts+dt, tr-dt, dt, dtype=np.double)  # Time from S to before R wave arrival
    T3 = [np.double(tr)]              # Arrival of R wave
    T4 = np.arange(tr+dt, tmax, dt, dtype=np.double)   # Time after passage of R wave
    T = np.concatenate([T0, T1, T2, T3, T4]) # Total time array
    T = T*r/Cs             # actual (physical) time
    T12 = T1**2
    T22 = T2**2
    S11 = np.sqrt(T12-x1)
    S21 = np.sqrt(T12-x2)
    S31 = np.sqrt(x3-T12)
    S12 = np.sqrt(T22-x1)
    S22 = np.sqrt(T22-x2)
    S32 = np.sqrt(x3-T22)

    # I.- VERTICAL LOAD
    # Vertical displacements due to vertical step load
    f = (1-pois)/(2*np.pi*mu*r)
    if np.imag(x1) == 0:
        A1 = (x1-0.5)**2*np.sqrt(a2-x1)/((x1-x2)*(x1-x3))
        A2 = (x2-0.5)**2*np.sqrt(a2-x2)/((x2-x1)*(x2-x3))
        A3 = (x3-0.5)**2*np.sqrt(x3-a2)/((x3-x1)*(x3-x2))
        U1 = 0.5*f*(1-A1/S11-A2/S21-A3/S31)
    else:
        A1 = (x1-0.5)**2*np.sqrt(a2-x1)/((x1-x2)*(x1-x3))
        A3 = (x3-0.5)**2*np.sqrt(x3-a2)/np.real((x3-x1)*(x3-x2))
        U1 = 0.5*f*(1-2*np.real(A1/S11)-A3/S31)

    U2 = f*(1-A3/S32)
    U3 = [U2[-1]]#[-np.sign(A3)*np.inf]
    U4 = f*np.ones(len(T4))
    U4[0] = U2[-1]
    Uzz = np.concatenate([U0, U1, U2, U3, U4])

    # Radial displacements due to vertical load
    n1 = b2/(a2-x1)
    if pois==0:
        x2 = a2
        n2 = np.inf
    else:
        n2 = b2/(a2-x2)
    n3 = b2/(a2-x3)
    k2 = (T12-a2)/b2	# = k**2
    if np.imag(x1)==0:
        B1 = ellipint3(90,n1*k2,k2)*(1-2*x1)*(1-x1)/(x1-x2)/(x1-x3)
        B2 = ellipint3(90,n2*k2,k2)*(1-2*x2)*(1-x2)/(x2-x1)/(x2-x3)
        B3 = ellipint3(90,n3*k2,k2)*(1-2*x3)*(1-x3)/(x3-x1)/(x3-x2)
        U1 = 2*ellipint3(90,0,k2)-B1-B2-B3
    else:
        B1 = 2*np.real(ellipint3(90,n1*k2,k2)*(1-2*x1)*(1-x1)/(x1-x2)/(x1-x3))
        tmp = np.real((x3-x1)*(x3-x2))
        B3 = ellipint3(90,n3*k2,k2)*(1-2*x3)*(1-x3)/tmp
        U1 = 2*ellipint3(90,0,k2)-B1-B3

    fac = 1/(8*np.pi**2*mu*r)
    f = fac/np.sqrt(b2**3)
    U1 = f*U1*T1
    t2 = T2**2
    k2 = b2/(t2-a2)	# inverse of k**2
    if np.imag(x1)==0:
        B1 = ellipint3(90,n1,k2)*(1-2*x1)*(1-x1)/(x1-x2)/(x1-x3)
        B2 = ellipint3(90,n2,k2)*(1-2*x2)*(1-x2)/(x2-x1)/(x2-x3)
        B3 = ellipint3(90,n3,k2)*(1-2*x3)*(1-x3)/(x3-x1)/(x3-x2)
        U2 = 2*ellipint3(90,0,k2) - B1 - B2 - B3
    else:
        B1 = 2*np.real(ellipint3(90,n1,k2)*(1-2*x1)*(1-x1)/(x1-x2)/(x1-x3))
        tmp = np.real((x3-x1)*(x3-x2))
        B3 = ellipint3(90,n3,k2)*(1-2*x3)*(1-x3)/tmp
        U2 = 2*ellipint3(90,0,k2) - B1 - B3

    C = (2*x3-1)**3/(1-4*x3+8*b2*x3**3)
    U2 = f*U2*T2*np.sqrt(k2)
    U3 = [U2[-1]]
    U4 = 2*np.pi*fac*C*T4/np.sqrt(T4**2-x3)
    Urz = np.concatenate([U0, U1, U2, U3, U4])

    # II.- HORIZONTAL LOAD
    # Radial displacements due to horizontal load
    f = 1/(2*np.pi*mu*r)
    fac = (1-pois)*f
    C1 = (1-x1)*np.sqrt(a2-x1)/((x1-x2)*(x1-x3))
    if np.imag(x1)==0:
        C2 = (1-x2)*np.sqrt(a2-x2)/((x2-x1)*(x2-x3))
        C3 = (1-x3)*np.sqrt(x3-a2)/((x3-x1)*(x3-x2))
        U1 = 0.5*fac*T12*(C1/S11+C2/S21+C3/S31)
    else:
        C3 = (1-x3)*np.sqrt(x3-a2)/np.real((x3-x1)*(x3-x2))
        U1 = 0.5*fac*T12*(2*np.real(C1/S11)+C3/S31)

    U2 = f+fac*C3*T22/S32
    U3 = [f]
    U4 = f*np.ones(np.size(T4))
    Urx = np.concatenate([U0, U1, U2, U3, U4])

    # Tangential displacements due to horizontal load
    if np.imag(x1)==0:
        U1 = 0.5*fac*(1-C1*S11-C2*S21+C3*S31)
    else:
        U1 = 0.5*fac*(1-2*np.real(C1*S11)+C3*S31)

    U2 = fac*(1+C3*S32)
    U3 = [fac]
    U4 = fac*np.ones(np.size(T4))
    Utx = np.concatenate([U0, U1, U2, U3, U4])

    # Displacements on epicentral axis
    # ********************************
    z = 1  # depth at which displacements are computed
    t = np.concatenate([T2,T3,T4]) # time after arrival of S wave
    t2 = t**2
    D1 = (16*(1-a2)*t2+8*(3-8*a2+6*a2**2))*t2
    D1 = (D1+8*(1-6*a2+10*a2**2-6*a2**3))*t2+(1-2*a2)**4
    D2 = (16*(1-a2)*t2-8*(3-4*a2))*t2
    D2 = (D2+8*(1-2*a2))*t2+1
    S1 = 0.5*(1+np.sqrt(1+(1-a2)/t2))*D1
    S2 = 0.5*(1+np.sqrt(1-(1-a2)/t2))*D2

    # a) Horizontal load
    fac = 0.25/np.pi/mu/z
    U1 = np.sqrt(T12-a2+1)
    U1 = 2*T1*(T12-a2)*U1
    U1 = U1/(U1-(2*T12-2*a2+1)**2)
    U2 = ((128*(1-a2)*t2-64*(1+4*a2-6*a2**2))*t2-16*(3-15*a2-4*a2**2+24*a2**3))*t2
    U2 = ((U2+16*a2*(4-17*a2+10*a2**2+8*a2**3))*t2+16*a2*(1-3*a2+7*a2**2-6*a2**3))*t2
    U2 = (U2-(1-10*a2+40*a2**2-48*a2**3+16*a2**4))*t2/D1/D2+1
    U2 = U2-(1-a2)*((t2-a2)*(2*t2-2*a2+1)**2/S1+2*t2*(2*t2-1)*(t2-1)/S2)
    Vxx = fac*np.concatenate([U0,U1,U2])   # Horiz. displacement at depth due to horiz. load

    # b) Vertical load
    fac = 0.5/np.pi/mu/z
    U1 = 2*T12-2*a2+1
    U1 = T12*U1/(U1**2-4*T1*(T12-a2)*np.sqrt(T12-a2+1))
    U2 = ((128*(1-a2)*t2+64*(1-2*a2)*(2-4*a2+a2**2))*t2-16*(21-37*a2+4*a2**2+36*a2**3-16*a2**4))*t2
    U2 = ((U2+16*(3+26*a2-78*a2**2+70*a2**3-8*a2**4-8*a2**5))*t2+4*(15-87*a2+116*a2**2+24*a2**3-136*a2**4+64*a2**5))*t2
    U2 = (U2+(11-28*a2+16*a2**2)*(1-2*a2)**3)*t2/D1/D2
    U2 = U2+(1-a2)*( 2*t2*(t2-a2)*(2*t2-2*a2+1)/S1 + (2*t2-1)**2*(t2-1)/S2)
    Vzz = fac*np.concatenate([U0,U1,U2]) # Vert. displacement at depth due to horiz. load

    if NoPlot:
      return T, Urx, Utx, Uzz, Urz

    # Plot response functions
    # ***********************

    plt.plot(T, Uzz)
    plt.grid()
    plt.axis([0, tmax, -1, 0.4])
    tit = 'Vertical displacement due to vertical point (step) load, \\nu=%5.3f' % pois
    plt.title(tit)
    titx = 'Dimensionless time'
    plt.xlabel(titx)
    plt.show()

    plt.plot(T,Urz)
    plt.grid()
    plt.axis([0, tmax, -0.2, 0.6])
    tit = 'Radial displacement due to vertical point (step) load, \\nu=%5.3f' % pois
    plt.title(tit)
    plt.xlabel(titx)
    plt.show()

    plt.plot(T,Urx)
    plt.grid()
    tit ='Radial displacement due to horizontal point (step) load, \\nu=%5.3f' % pois
    plt.title (tit)
    plt.xlabel(titx)
    plt.show()

    plt.plot(T,Utx)
    plt.grid()
    tit ='Tangential displacement due to horizontal point (step) load, \\nu=%5.3f' % pois
    plt.title (tit)
    plt.xlabel(titx)
    plt.show()

    plt.plot(T,Vzz)
    plt.grid()
    tit = 'Vertical displacement under load at epicentral axis, \\nu=%5.3f' % pois
    plt.title(tit)
    plt.xlabel(titx)
    plt.show()

    plt.plot(T,Vxx)
    plt.grid()
    tit = 'Horizontal displacement under load at epicentral axis, \\nu=%5.3f' % pois
    plt.title(tit)
    plt.xlabel(titx)
    plt.show()

    return T, Urx, Utx, Uzz, Urz


def ellipint3(phi, N, M):
    #    [EL3] = ELLIPINT3 (phi,N,M) returns the elliptic integral of
    #            the third kind, evaluated for each value of N, M
    #            Can also be used to obtain the elliptic integral
    #            of the first kind by setting N=0.
    #    Arguments :  phi   -- Upper limit of integration (in degrees)
    #                 M=[m] -- Modulus   (some authors use k=sqrt(m))
    #                          M can be a scalar or vector
    #                 N=[n] -- Parameter (some authors use c=-n)
    #                          N can be also be scalar or vector, but if
    #                          the latter, it must agree in size with M
    #    Definition: If n, m are elements of N, M, then
    #
    #                phi
    #    el3 = integ  |   [dt/((1+n*sin**2(t))*sqrt(1-m*sin**2(t)))
    #                 0
    #
    #    Observe that m = k**2 is the square of the argument
    #    used by some authors for the elliptic integrals
    #    Method:  10-point Gauss-Legendre quadrature
    if (phi < 0):
        'Error, first argument in ellipint3 cannot be negative'
        el3 = 0
        return 0
    try:
        len(N)
    except TypeError:
        N = N*np.ones(len(M), dtype=np.double)
    # if type(N) != list or type(N) != np.array:
    #     N = N*np.ones(M.shape)
    # elif len(N) != len(M):
    #     raise TypeError('Error, wrong size of second argument in ellipint3.'
    #                     'Should be size(N)=1 or size(N)=size(M)')

    tol = 1.e-16
    ang = phi*np.pi/180
    psi = ang/2
    t = np.array([.9931285991850949, .9639719272779138, .9122344282513259,
                  .8391169718222188, .7463319064601508, .6360536807265150,
                  .5108670019508271, .3737060887154195, .2277858511416451,
                  .07652652113349734], dtype=np.double)
    w = np.array([.01761400713915212, .04060142980038694, .06267204833410907,
                  .08327674157670475, .1019301198172404,  .1181945319615184,
                  .1316886384491766,  .1420961093183820,  .1491729864726037,
                  .1527533871307258], dtype=np.double)
    t1 = psi*(1+t)
    t2 = psi*(1-t)
    s1 = np.sin(t1)**2
    s2 = np.sin(t2)**2
    el3 = np.zeros(M.shape, dtype=np.complex128)
    s = np.sin(ang)**2
    for j in range(0, len(M)):
        k2 = M[j]
        n = N[j]
        # assuming phi is in degrees here
        if ((phi<=90 and abs(1+n*s)>tol and abs(1-k2*s)>tol) or
            (phi>90  and abs(1+n)  >tol and abs(1-k2)  >tol)):
            f1 = 1/( (1+n*s1)*np.sqrt(1-k2*s1) )
            f2 = 1/( (1+n*s2)*np.sqrt(1-k2*s2) )
            el3[j] = np.sum((f1+f2)*w)
        else:
            el3[j] = np.inf
            print("output inf")
    el3 = psi*el3
    return el3

def to_cartesian(ur, ut, uz, l, theta=0):
    """
    Transform the solution to Lamb's problem to cartesian, positive z down!

    Args:
        ur (np.array): Radial component of displacement
        ut (np.array): Tangential component of displacement
        uz (np.array): Z component of displacement
        l (int): 0: force in z, 1, force in x
        theta (float): Azimuth in radian

    Returns:
        ux, uy, uz (np.array): The X, Y and Z components of displacement
    """

    ux = ur * np.cos(l * theta) * np.cos(theta) + ut * np.sin(l * theta) * np.sin(theta)
    uy = ur * np.cos(l * theta) * np.sin(theta) - ut * np.sin(l * theta) * np.cos(theta)
    uz = -uz * np.cos(l * theta)

    return ux, uy, uz

def resample(u, t, dt, tmin=0, tmax=None):
    """
    The solution from Lamb_3D is not evenly sampled in time.
    Resample the solution using linear interpolation

    Args:
        u (np.array): Computed displacement u
        t (np.array): Time vector of this displacement
        dt (float): desired time step
        tmin (float): minimum time (default 0)
        tmax (float): maximum time (max(t))

    Returns:
        ui (np.array): Interpolated displacement u
        ti (np.array): Time vector of the interpolation
    """
    if tmax is None:
        tmax = np.max(t)
    ti = np.arange(tmin, tmax+dt, dt)
    return np.interp(ti, t, u), ti

def get_source_response(u, s):
    """
    Converts lamb_3D original solution which is for the displacement due to
    a heaviside function to the particle velocities for a specified source
    function.

    Args:
        u: (np.array): Displacement solution
        s: (np.array): Source signature

    Returns:
        v (np.array) Particule velocities
    """

    #Green functions for particle velocities is the second derivative in time
    #of the displacement solution to a heaviside source
    u[1:-1] = u[2:]-2*u[1:-1]+u[:-2]
    u[0] = u[-1] = 0
    #Convole the Green function with the wavelet.
    U = np.fft.fft(u)
    S = np.fft.fft(s)
    v = np.real(np.fft.ifft(U * S))

    return v

def compute_shot(offsets, vp, vs, rho, dt, src,
                 srctype = "x", rectype = "x", linedir = "x"):
    """
    Main interface to obtain the analytical to Lamb's problem for a shot.

    Args:
        offsets (np.array): Desired offsets of the receivers
        vp (float): Vp velocity in m/s
        vs (float): Vs velocity in m/s
        rho (float): Density in kg/m3
        dt (float):  Time step
        src np.array): Source wavelet
        srctype (str): Direction of the source (force in "x", "y" or "z")
        rectype (str): Direction of the receivers ("x", "y" or "z"):
        linedir (str): Direction of the line ("x" or "y")

    Returns:
        traces (np.array): The seismogram for this shot.
    """

    pois = 0.5 * (vp ** 2 - 2.0 * vs ** 2) / (vp ** 2 - vs ** 2)
    mu = vs ** 2 * rho
    tmax = (src.shape[0]-1) * dt
    if linedir == "x":
        theta = 0
    elif linedir == "y":
        theta = np.pi/2
    else:
        raise ValueError("linedir must be either \"x\" or \"y\"")
    
    traces = []
    for r in offsets:
        T, Urx, Utx, Uzz, Urz = Lamb_3D(pois=pois, NoPlot=1,
                                        r=r, mu=mu, rho=rho)
        t = T

        if srctype == "x":
            uxx, uyx, uzx = to_cartesian(Urx, Utx, -Urz * np.cos(theta), l=1,
                                         theta=theta)
            if rectype == "x":
                u, ti = resample(uxx, t, dt, tmax=tmax)
            elif rectype == "y":
                u, ti = resample(uyx, t, dt, tmax=tmax)
            elif rectype == "z":
                u, ti = resample(uzx, t, dt, tmax=tmax)
            else:
                raise ValueError("rectype must be either \"x\", \"y\" or \"z\"")

        elif srctype == "z":
            uxz, uyz, uzz = to_cartesian(Urz, Urz * 0, Uzz, l=0, theta=theta)
            if rectype == "x":
                u, ti = resample(uxz, t, dt, tmax=tmax)
            elif rectype == "y":
                u, ti = resample(uyz, t, dt, tmax=tmax)
            elif rectype == "z":
                u, ti = resample(uzz, t, dt, tmax=tmax)
            else:
                raise ValueError("rectype must be either \"x\", \"y\" or \"z\"")
        else:
            raise ValueError("srctype must be either \"x\" or \"z\" ")

        v = get_source_response(u, src)
        traces.append(np.real(v))

    return np.transpose(np.array(traces))

def ricker_wavelet(f0, NT, dt ):
    """
    Computes a Ricker wavelet

    Args:
        f0 (float): Center frequency of the wavelet in Hz
        NT (int): Number of timesteps
        dt (float): Time interval in seconds

    Returns:
        ricker (np.array): An array containing the wavelet samples

    """
    tmin = -1.5 / f0
    t = np.linspace(tmin, (NT-1) * dt + tmin, num=NT)
    ricker = ((1.0 - 2.0 * (np.pi ** 2) * (f0 ** 2) * (t ** 2))
              * np.exp(-(np.pi ** 2) * (f0 ** 2) * (t ** 2)))

    return ricker

if __name__ == "__main__":
    """Example to compute a shot"""

    offsets = np.arange(10, 600, 10)
    dt = 0.0001
    vp = 1400
    vs = 200
    rho = 1800
    f0 = 5
    tmax = 4

    src = ricker_wavelet(f0, int(tmax//dt), dt)
    shot = compute_shot(offsets, vp, vs, rho, dt, src,
                        srctype="z", rectype="x", linedir="x")
    clip = 0.1
    vmax = np.max(shot) * clip
    vmin = -vmax
    plt.imshow(shot, aspect='auto', vmax=vmax, vmin=vmin)
    plt.show()
