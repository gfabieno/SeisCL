# -*- coding: utf-8 -*-
"""Analytical solution to Garvin's problem (2D elastic half-space)

This script is a translation from Matlab code privided with he following paper:

     Sánchez-Sesma, F. J., Iturrarán-Viveros, U., & Kausel, E. (2013).
     Garvin’s generalized problem revisited.
     Soil Dynamics and Earthquake Engineering, 47, 4-15.

This module contains several functions:
    -Garvin2: The python version of the Matlab script
    -t_PS: The python version of the Matlab function used in Garvin2
    -resample: Resample the provided solution to an constant time step
    -get_source_response: Obtain particle velocities from Lamb_3D solution
    -compute_shot: Main interface. Computes the solution for a shot
    -main: Calls shot to plot an example of the solution
"""

import numpy as np
import matplotlib.pyplot as plt

def Garvin2(x,z,pois=0.25,h=1,Cs=1,rho=1, plot=1):
    """
     Solves the generalized Garvin problem of a line blast
     source applied at depth h within a homogeneous half-space.
     The response is sought within that same half-space
     at a receiver at range x and depth z.

     Written by Eduardo Kausel, MIT, Room 1-271, Cambridge, MA
     Version 1, July 19, 2011

     Input arguments:
     x = range of receiver >0
     z = depth of receiver >=0
     pois = Poisson's ratio Defaults to 0.25 if not given
     h = Depth of source > 0 " " 1 " " "
     Cs = Shear wave velocity " " 1 " " "
     rho = mass density " " 1 " " "

     Sign convention:
     x from left to right, z=0 at the surface, z points down
     Displacements are positive down and to the right.

     References:
     W.W. Garvin, Exact transient solution of the buried line source
     problem, Proceedings of the Royal Society of London,
     Series A, Vol. 234, No. 1199, March 1956, 528-541
     Z.S. Alterman and D. Loewenthal, Algebraic Expressions for the
     impulsive motion of an elastic half-space,
     Israel Journal of Technology, Vol. 7, No. 6,
     1969, pp. 495-504
     default data
    """
    N = 1000 # number of time steps to arrival of PS waves
    mu = rho*Cs**2 # shear modulus
    r1 = np.sqrt(x**2+(z-h)**2) # source-receiver distance
    r2 = np.sqrt(x**2+(z+h)**2) # image source-receiver distance
    s1 = x/r1 # sin(theta1)
    c1 = (z-h)/r1 # cos(theta1)
    s2 = x/r2 # sin(theta2)
    c2 = (h+z)/r2 # cos(theta2)
    a2 = (0.5-pois)/(1-pois)
    a = np.sqrt(a2) # Cs/Cp
    Cp = Cs/a # P-wave velocity
    tS = r1/Cs # S-wave arrival (none here)
    tP = r1/Cp # time of arrival of direct P waves
    tPP = r2/Cp # time of arrival of PP waves
    tPS = t_PS(x,z,h,Cs,Cp) # time of arrival or PS waves
    dt = (tPS-tP)/N # time step
    t1 = np.arange((tP+dt),(tPP-dt), dt) # time before reflections
    t2 = np.arange((tPP+dt), tPS, dt) # time from PP to PS reflection
    t3 = np.arange((tPS+dt),3*tPS, dt) # time after arrival of PS waves
    # Find q3(tau) from tau(q3) by solving quartic
    X=x/r2; Z=z/r2; H=h/r2;
    A = ((H+Z)**2+X**2)*((H-Z)**2+X**2)
    B1 = X*(X**2+H**2+Z**2)
    C1 = X**2*(a2*H**2+Z**2)+(H**2-Z**2)*(a2*H**2-Z**2)
    C2 = 3*X**2+H**2+Z**2
    D1 = X*(a2*H**2+Z**2)
    E1 = (a*H+Z)**2
    E2 = (a*H-Z)**2
    tau = t3*Cs/r2 # dimensionless time for PS waves
    q3 = []
    for j in range(0,len(t3)):
        tau2 = tau[j]**2
        B = tau[j]*B1
        C = tau2*C2-C1
        D = tau[j]*(tau2*X-D1)
        E = (tau2-E1)*(tau2-E2)
        q = np.roots([A,-4*1j*B,-2*C,4*1j*D,E]) # in lieu of Ferrari
        q = q[(np.real(q)>=0) & (np.imag(q)>=0)] # discard negative roots
        q1 = np.min(np.imag(q)) # find position of true root
        I = np.where(np.imag(q)==q1)
        q3.append(q[I]) # choose that root
    q3 = np.array(q3)[:,0]

    # Compare exact vs. approximate
    if plot:
        # Sánchez-Sesma approximation:
        # *****************************
        R = (h + z / a) / (h + z)  # r_eq/r2
        r3 = R * r2  # equivalent radius
        tapp = tau / R
        T = np.conj(np.sqrt(tapp ** 2 - a2))  # conj --> T must have neg. imag part
        q3app = R * (c2 * T + 1j * tapp * s2)
        plt.plot(t3,np.real(q3))
        plt.plot(t3,np.imag(q3),'r')
        plt.plot(t3,np.real(q3app),'--')
        plt.plot(t3,np.imag(q3app),'r--')
        plt.grid()
        plt.title('q3 --> exact vs. Sánchez-Sesma''s approximation')
        plt.xlabel('Time')
        plt.show()

    # Find and plot the time histories
    # ****************************
    # a) From tP to tPP
    T1 = np.sqrt(t1**2-tP**2)
    f1 = (0.5/r1)*t1/T1
    u1 = f1*s1
    w1 = f1*c1
    # b) From tPP to tPS
    T1 = np.sqrt(t2**2-tP**2)
    T2 = np.sqrt(t2**2-tPP**2)
    f1 = (0.5/r1)*t2/T1
    f2 = (0.5/r2)*t2/T2
    q2 = (c2*T2+1j*s2*t2)*Cs/r2
    dq2 = c2*t2/T2+1j*s2 # derivative
    Q2 = q2**2
    Q2S = np.sqrt(Q2+1)
    Q2P = np.sqrt(Q2+a2)
    S2 = (1+2*Q2)**2
    D2 = S2-4*Q2*Q2S*Q2P # Rayleigh function
    u2 = f1*s1-f2*s2-(4/r2)*np.imag(q2**3.*Q2S*dq2/D2)
    w2 = f1*c1+f2*c2-(1/r2)*np.real(S2*dq2/D2)
    # c) From tPS and on
    T1 = np.sqrt(t3**2-tP**2)
    T2 = np.sqrt(t3**2-tPP**2)
    f1 = (0.5/r1)*t3/T1
    f2 = (0.5/r2)*t3/T2
    # Contribution of PP waves
    q2 = (c2*T2+1j*s2*t3)*Cs/r2
    dq2 = c2*t3/T2+1j*s2 # derivative
    Q2 = q2**2
    Q2S = np.sqrt(Q2+1)
    Q2P = np.sqrt(Q2+a2)
    S2 = (1+2*Q2)**2
    D2 = S2-4*Q2*Q2S*Q2P # Rayleigh function
    f3 = (4/r2)*np.imag(q2**3.*Q2S*dq2/D2)
    f5 = (1/r2)*np.real(S2*dq2/D2)
    # Contribution of PS waves
    Q3 = q3**2
    Q3S = np.sqrt(Q3+1)
    Q3P = np.sqrt(Q3+a2)
    S = 1+2*Q3
    S3 = S**2
    D3 = S3-4*Q3*Q3S*Q3P # Rayleigh function
    dq3 = 1./((h/r2/Q3P+z/r2/Q3S)*q3-1j*x/r2)
    f4 = (2/r2)*np.imag(q3*S*Q3S*dq3/D3)
    f6 = (2/r2)*np.real(Q3*S*dq3/D3)
    u3 = f1*s1-f2*s2-f3+f4
    w3 = f1*c1+f2*c2-f5+f6
    # Combine the results and plot
    #time = [[0,tP],t1,t2,t3]*Cs/r1
    time = np.concatenate([[0], [tP], t1, t2, t3])*Cs/r1
    u = np.concatenate([[0],[0],u1,u2,u3])*(r1/np.pi)
    w = np.concatenate([[0],[0],w1,w2,w3])*(r1/np.pi)

    if plot:
        plt.plot(time,u)
        tit = 'Horizontal displacements at x =%f z =%f' % (x, z)
        plt.title(tit)
        plt.xlabel('t*\beta/r1')
        plt.ylabel('Ux*r1*\mu')
        plt.grid()
        plt.show()

        plt.plot(time,w)
        tit = 'Vertical displacements at x =%f z =%f' % (x, z)
        plt.title(tit)
        plt.xlabel('t*\beta/r1')
        plt.ylabel('Uz*r1*\mu')
        plt.grid()
        plt.show()
    return u, w, time

#--------------------------------------------------------------
def t_PS(x,z,h,Cs,Cp):
    """
     Determines the total travel time of the PS reflection
     Arguments
     *********
     x = range of receiver
     z = depth of receiver
     h = depth of source
     Cs = S-wave velocity
     Cp = P-wave velocity
    """
    if z==0:
        tPS=np.sqrt(x**2+h**2)/Cp
        return tPS
    a = Cs/Cp
    # Bracket the S point
    xP = x*h/(h+z) # point of reflection of PP ray
    ang1 = np.arctan(xP/h) # minimum angle of incident ray
    ang2 = np.arctan(x/h) # maximum angle
    # Find the S point by search within bracket
    dang = (ang2-ang1)/10
    TOL = 1.e-8
    TRUE = 1
    while TRUE:
        angP = ang1+dang
        angS = np.arcsin(a*np.sin(angP))
        L = h*np.tan(angP)+z*np.tan(angS)
        if L>x:
            if L-x<TOL*x:
                break
            else:
                dang = dang/10
        else:
            ang1 = angP
    tPS = h/np.cos(angP)/Cp+z/np.cos(angS)/Cs

    xS = h*np.tan(angP) # point of reflection of PS ray
    return tPS
#--------------------------------------------------------------


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
    Converts garvin2 original solution which is for the displacement due to
    a heaviside function to the particle velocities for a specified source
    function.

    Args:
        u: (np.array): Displacement solution
        s: (np.array): Source signature

    Returns:
        v (np.array) Particle velocities
    """
    #Green functions for particle velocities is the second derivative in time
    #of the displacement solution to a heaviside source. But source shape for a
    # pressure source is the integral of the source wavelet. So we use the first
    #derivative here.
    u[1:-1] = 0.5 * u[2:] - 0.5 * u[:-2]
    u[0] = u[-1] = 0
    # Convole the Green function with the wavelet.
    U = np.fft.fft(u)
    S = np.fft.fft(s)
    v = -np.real(np.fft.ifft(U * S))

    return v

def compute_shot(offsets, vp, vs, rho, dt, src, rectype="x", zsrc=1, zrec=1):
    """
    Main interface to obtain the analytical to Garvin's problem for a shot.

    Args:
        offsets (np.array): Desired offsets of the receivers
        vp (float): Vp velocity in m/s
        vs (float): Vs velocity in m/s
        rho (float): Density in kg/m3
        dt (float):  Time step
        src np.array): Source wavelet
        rectype (str): Direction of the receivers ("x" or "z"):
        zsrc (float): Position in z of the source (>0)
        zrec (float): position in z of the receiver (>0)

    Returns:
        traces (np.array): The seismogram for this shot.
    """
    pois = 0.5 * (vp ** 2 - 2.0 * vs ** 2) / (vp ** 2 - vs ** 2)
    mu = vs ** 2 * rho
    tmax = (src.shape[0]-1) * dt

    traces = [[] for _ in offsets]
    for ii, r in enumerate(offsets):
        ux, uz, t = Garvin2(r, zrec, pois=pois, h=zsrc, Cs=vs, rho=rho, plot=0)
        t = t * r / vs
        ux = ux / r / mu
        uz = uz / r / mu

        if rectype == "x":
            u, ti = resample(ux, t, dt, tmax=tmax)
        elif rectype == "z":
            u, ti = resample(uz, t, dt, tmax=tmax)
        else:
            raise ValueError("rectype must be either \"x\" or \"z\"")

        v = get_source_response(u, src, dt)
        traces[ii] = np.real(v)

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

    offsets = np.arange(10, 400, 10)
    dt = 0.0001
    vp = 1400
    vs = 200
    rho = 1800
    f0 = 5
    tmax = 4

    src = ricker_wavelet(f0, int(tmax // dt), dt)
    shot = compute_shot(offsets, vp, vs, rho, dt, src, rectype="x")
    clip = 0.1
    vmax = np.max(shot) * clip
    vmin = -vmax
    plt.imshow(np.array(shot), aspect='auto', vmax=vmax, vmin=vmin)
    plt.show()