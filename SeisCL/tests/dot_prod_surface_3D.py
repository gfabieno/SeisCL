import numpy as np
import copy


FDOH = 2
hc = [1.1382, - 0.046414]
slices = tuple([slice(FDOH, -FDOH) for _ in range(3)])

"""
    Adjoint of the surface kernel. With the symmetrization strategy, we need to
    transform the adjoint kernel according to:
    
    F_s = S F
    
    where S is the surface kernel and F the seismic modeling kernel. The
    transformed operator is:
    
    F_s' = L S F T
    
    where L and T are the transform matrices. Taking the adjoint:
    
    F_s'^* = ( L S L^-1 L F T )^*
    
           = L F T L^-1 S^* L
           
    Performing the back transformation:
    
    L^-1 F_s'^* T = F (T L^-1) S^* (L T)
    
    To apply the free surface kernel, we must then apply the transformation
    L T, then apply the free surface kernel S^* and do the back transformation.
    
    For the isotropic elastic wave equation, the transform matrices are:
    
    T = [1  0     L = [rho   0
         0 -1]         0 C^-1]
         
    where C is the rigidity tensor, given by:
    
    C [   M   M-2mu M-2mu 0   0   0
        M-2mu   M   M-2mu 0   0   0
        M-2mu M-2mu   M   0   0   0
          0     0     0   mu  0   0
          0     0     0   0   mu  0
          0     0     0   0   0   mu]
    
    C^-1 [ a [2M-2mu -M+2mu -M+2mu   0   0     0
             -M+2mu 2M-2mu -M+2mu    0   0     0
             -M+2mu -M+2mu 2M-2mu]   0   0     0
             0     0     0         1/mu  0     0
             0     0     0           0   1/mu  0
             0     0     0           0   0   1/mu]
             
    C^-1 [ a [M -M+2mu  0
             -M+2mu  M  0
               0     0  1/mu ]   in 2D
             
    a = 1/( 6Mmu - 8mu^2 )  in 3D
      
    a = 1/( 4Mmu - 4mu^2 )       in 2D
    
    In 2D, the transformation is thus:
    
    C^-1 sigma = [ a (M sxx - (M-2mu) szz),
                   a (M szz - (M-2mu) sxx),
                   1/mu sxy ]
"""


def Dpx(var):
    return hc[0] * (var[2:-2, 2:-2, 3:-1] - var[2:-2, 2:-2, 2:-2]) + hc[1] * (var[2:-2, 2:-2, 4:] - var[2:-2, 2:-2, 1:-3])


def Dmx(var):
    return hc[0] * (var[2:-2, 2:-2, 2:-2] - var[2:-2, 2:-2, 1:-3]) + hc[1] * (var[2:-2, 2:-2, 3:-1] - var[2:-2, 2:-2, 0:-4])

def Dpy(var):
    return hc[0] * (var[2:-2, 3:-1, 2:-2] - var[2:-2, 2:-2, 2:-2]) + hc[1] * (var[2:-2, 4:, 2:-2] - var[2:-2, 1:-3, 2:-2])

def Dmy(var):
    return hc[0] * (var[2:-2, 2:-2, 2:-2] - var[2:-2, 1:-3, 2:-2]) + hc[1] * (var[2:-2, 3:-1, 2:-2] - var[2:-2, 0:-4, 2:-2])

def Dpz(var):
    return hc[0] * (var[3:-1, 2:-2, 2:-2] - var[2:-2, 2:-2, 2:-2]) + hc[1] * (var[4:, 2:-2, 2:-2] - var[1:-3, 2:-2, 2:-2])


def Dmz(var):
    return hc[0] * (var[2:-2, 2:-2, 2:-2] - var[1:-3, 2:-2, 2:-2]) + hc[1] * (var[3:-1, 2:-2, 2:-2] - var[0:-4, 2:-2, 2:-2])



def apply_T(vxi, vyi, vzi, sxxi, syyi, szzi, sxzi, sxyi, syzi, rho, M, mu):
    
    vx = copy.copy(vxi)
    vy = copy.copy(vyi)
    vz = copy.copy(vzi)
    sxx = copy.copy(sxxi)
    syy = copy.copy(syyi)
    szz = copy.copy(szzi)
    sxz = copy.copy(sxzi)
    sxy = copy.copy(sxyi)
    syz = copy.copy(syzi)

    return vx, vy, vz, -sxx, -syy, -szz, -sxz, -sxy, -syz


def apply_L(vxi, vyi, vzi, sxxi, syyi, szzi, sxzi, sxyi, syzi, rho, M, mu):

    vx = copy.copy(vxi)
    vy = copy.copy(vyi)
    vz = copy.copy(vzi)
    sxx = copy.copy(sxxi)
    syy = copy.copy(syyi)
    szz = copy.copy(szzi)
    sxz = copy.copy(sxzi)
    sxy = copy.copy(sxyi)
    syz = copy.copy(syzi)
    
    a = 1.0 / ( 6.0 * M * mu - 8.0 * mu**2 )
    vx[slices] = vx[slices] / rho
    vy[slices] = vy[slices] / rho
    vz[slices] = vz[slices] / rho
    sxx[slices] = a *( 2.0*(M-mu) * sxxi[slices] - (M-2.0*mu) * (syyi[slices] + szzi[slices]))
    syy[slices] = a *( 2.0*(M-mu) * syyi[slices] - (M-2.0*mu) * (sxxi[slices] + szzi[slices]))
    szz[slices] = a *( 2.0*(M-mu) * szzi[slices] - (M-2.0*mu) * (syyi[slices] + sxxi[slices]))
    sxz[slices] = sxzi[slices] / mu
    sxy[slices] = sxyi[slices] / mu
    syz[slices] = syzi[slices] / mu

    return vx, vy, vz, sxx, syy, szz, sxz, sxy, syz

def apply_Lm(vxi, vyi, vzi, sxxi, syyi, szzi, sxzi, sxyi, syzi, rho, M, mu):

    vx = copy.copy(vxi)
    vy = copy.copy(vyi)
    vz = copy.copy(vzi)
    sxx = copy.copy(sxxi)
    syy = copy.copy(syyi)
    szz = copy.copy(szzi)
    sxz = copy.copy(sxzi)
    sxy = copy.copy(sxyi)
    syz = copy.copy(syzi)
    
    vx[slices] = vx[slices] * rho
    vy[slices] = vy[slices] * rho
    vz[slices] = vz[slices] * rho
    sxx[slices] = ( M * sxxi[slices] + (M-2.0*mu) * (syyi[slices] + szzi[slices]))
    syy[slices] = ( M * syyi[slices] + (M-2.0*mu) * (sxxi[slices] + szzi[slices]))
    szz[slices] = ( M * szzi[slices] + (M-2.0*mu) * (syyi[slices] + sxxi[slices]))
    sxz[slices] = sxzi[slices] * mu
    sxy[slices] = sxyi[slices] * mu
    syz[slices] = syzi[slices] * mu

    return vx, vy, vz, sxx, syy, szz, sxz, sxy, syz

def update_v(vxi, vyi, vzi, sxxi, syyi, szzi, sxzi, sxyi, syzi, rho, M, mu):

    vx = copy.copy(vxi)
    vy = copy.copy(vyi)
    vz = copy.copy(vzi)
    sxx = copy.copy(sxxi)
    syy = copy.copy(syyi)
    szz = copy.copy(szzi)
    sxz = copy.copy(sxzi)
    sxy = copy.copy(sxyi)
    syz = copy.copy(syzi)
    
    sxx_x = Dpx(sxx)
    syy_y = Dpy(syy)
    szz_z = Dpz(szz)
    sxz_x = Dmx(sxz)
    sxz_z = Dmz(sxz)
    sxy_x = Dmx(sxy)
    sxy_y = Dmy(sxy)
    syz_y = Dmy(syz)
    syz_z = Dmz(syz)
    
    vx[slices] += (sxx_x + sxy_y + sxz_z) * rho
    vy[slices] += (syy_y + sxy_x + syz_z) * rho
    vz[slices] += (szz_z + sxz_x + syz_y) * rho

    return vx, vy, vz, sxx, syy, szz, sxz, sxy, syz

def update_s(vxi, vyi, vzi, sxxi, syyi, szzi, sxzi, sxyi, syzi, rho, M, mu):
    vx = copy.copy(vxi)
    vy = copy.copy(vyi)
    vz = copy.copy(vzi)
    sxx = copy.copy(sxxi)
    syy = copy.copy(syyi)
    szz = copy.copy(szzi)
    sxz = copy.copy(sxzi)
    sxy = copy.copy(sxyi)
    syz = copy.copy(syzi)
    
    vz_x = Dpx(vz)
    vz_y = Dpy(vz)
    vz_z = Dmz(vz)
    vy_x = Dpx(vy)
    vy_y = Dmy(vy)
    vy_z = Dpz(vy)
    vx_x = Dmx(vx)
    vx_y = Dpy(vx)
    vx_z = Dpz(vx)
    
    sxz[slices] += mu * (vx_z + vz_x)
    sxy[slices] += mu * (vx_y + vy_x)
    syz[slices] += mu * (vy_z + vz_y)
    sxx[slices] += M * (vx_x + vy_y + vz_z) - 2.0 * mu * (vy_y + vz_z)
    syy[slices] += M * (vx_x + vy_y + vz_z) - 2.0 * mu * (vx_x + vz_z)
    szz[slices] += M * (vx_x + vy_y + vz_z) - 2.0 * mu * (vy_y + vx_x)

    return vx, vy, vz, sxx, syy, szz, sxz, sxy, syz

def surface(vxi, vyi, vzi, sxxi, syyi, szzi, sxzi, sxyi, syzi, rho, M, mu):
    vx = copy.copy(vxi)
    vy = copy.copy(vyi)
    vz = copy.copy(vzi)
    sxx = copy.copy(sxxi)
    syy = copy.copy(syyi)
    szz = copy.copy(szzi)
    sxz = copy.copy(sxzi)
    sxy = copy.copy(sxyi)
    syz = copy.copy(syzi)

    for ii in range(vx.shape[2] - 2*FDOH):
        for jj in range(vx.shape[1] - 2*FDOH):
            szz[FDOH, jj+FDOH, ii+FDOH] = 0.0


            for kk in range(1,FDOH+1):
                szz[FDOH - kk, jj+FDOH, ii+FDOH] = -szz[FDOH+kk, jj+FDOH, ii+FDOH]
                sxz[FDOH - kk, jj+FDOH, ii+FDOH] = -sxz[FDOH+kk-1, jj+FDOH, ii+FDOH]
                syz[FDOH - kk, jj+FDOH, ii+FDOH] = -syz[FDOH+kk-1, jj+FDOH, ii+FDOH]

            vxx = 0
            vyy = 0
            vzz = 0
            for kk in range(FDOH):
                vxx += hc[kk] * (vx[FDOH, jj + FDOH, ii + FDOH + kk] - vx[FDOH, jj + FDOH, ii + FDOH - (kk + 1)])
                vyy += hc[kk] * (vy[FDOH, jj + FDOH + kk, ii + FDOH] - vy[FDOH, jj + FDOH - (kk + 1), ii + FDOH])
                vzz += hc[kk] * (vz[FDOH + kk, jj + FDOH, ii + FDOH] - vz[FDOH - (kk + 1), jj + FDOH, ii + FDOH])

            f = mu[0, jj, ii] * 2.0
            g = M[0, jj, ii]
            h = -((g-f)*(g-f)*(vxx+vyy)/g)-((g-f)*vzz)
            #h *= taper[FDOH, ii+FDOH]

            sxx[FDOH, jj+FDOH, ii+FDOH] += h
            syy[FDOH, jj+FDOH, ii+FDOH] += h

    return vx, vy, vz, sxx, syy, szz, sxz, sxy, syz

def surface_adj(vxi, vyi, vzi, sxxi, syyi, szzi, sxzi, sxyi, syzi, rho, M, mu):
    vx = copy.copy(vxi)
    vy = copy.copy(vyi)
    vz = copy.copy(vzi)
    sxx = copy.copy(sxxi)
    syy = copy.copy(syyi)
    szz = copy.copy(szzi)
    sxz = copy.copy(sxzi)
    sxy = copy.copy(sxyi)
    syz = copy.copy(syzi)
    
    for ii in range(vx.shape[2] - 2*FDOH):
        for jj in range(vx.shape[1] - 2*FDOH):
            szz[FDOH, jj+FDOH, ii+FDOH] = 0.0

            f = mu[0, jj, ii] * 2.0
            g = M[0, jj, ii]
            hx = -((g - f) * (g - f) / g)
            hz = -(g - f)

            for kk in range(FDOH):
                if jj + kk < vx.shape[1] - 2*FDOH:
                    vy[FDOH, jj + FDOH + kk, ii + FDOH] += hc[kk] * hx * (sxx[FDOH, jj + FDOH, ii + FDOH] + syy[FDOH, jj + FDOH, ii + FDOH])
                if jj + FDOH - (kk + 1) > FDOH-1:
                    vy[FDOH, jj + FDOH - (kk + 1), ii + FDOH] += - hc[kk] * hx * (sxx[FDOH, jj + FDOH, ii + FDOH] + syy[FDOH, jj + FDOH, ii + FDOH])
                if ii + kk < vx.shape[2] - 2*FDOH:
                    vx[FDOH, jj + FDOH, ii + FDOH + kk] += hc[kk] * hx * (sxx[FDOH, jj + FDOH, ii + FDOH] + syy[FDOH, jj + FDOH, ii + FDOH])
                if ii + FDOH - (kk + 1) > FDOH-1:
                    vx[FDOH, jj + FDOH, ii + FDOH - (kk + 1)] += - hc[kk] * hx * (sxx[FDOH, jj + FDOH, ii + FDOH] + syy[FDOH, jj + FDOH, ii + FDOH])

                vz[FDOH + kk, jj + FDOH, ii + FDOH] += hc[kk] * hz * (sxx[FDOH, jj + FDOH, ii + FDOH] + syy[FDOH, jj + FDOH, ii + FDOH])

    return vx, vy, vz, sxx, syy, szz, sxz, sxy, syz


if __name__ == "__main__":
    
    
    vx = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    vx[slices] = np.random.rand(10, 10, 10)
    vy = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    vy[slices] = np.random.rand(10, 10, 10)
    vz = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    vz[slices] = np.random.rand(10, 10, 10)
    sxx = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    sxx[slices] = np.random.rand(10, 10, 10)
    syy = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    syy[slices] = np.random.rand(10, 10, 10)
    szz = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    szz[slices] = np.random.rand(10, 10, 10)
    sxz = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    sxz[slices] = np.random.rand(10, 10, 10)
    sxy = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    sxy[slices] = np.random.rand(10, 10, 10)
    syz = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    syz[slices] = np.random.rand(10, 10, 10)
    
    vx_a = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    vx_a[slices] = np.random.rand(10, 10, 10)
    vy_a = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    vy_a[slices] = np.random.rand(10, 10, 10)
    vz_a = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    vz_a[slices] = np.random.rand(10, 10, 10)
    sxx_a = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    sxx_a[slices] = np.random.rand(10, 10, 10)
    syy_a = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    syy_a[slices] = np.random.rand(10, 10, 10)
    szz_a = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    szz_a[slices] = np.random.rand(10, 10, 10)
    sxz_a = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    sxz_a[slices] = np.random.rand(10, 10, 10)
    sxy_a = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    sxy_a[slices] = np.random.rand(10, 10, 10)
    syz_a = np.zeros([2 * FDOH + 10, 2 * FDOH + 10, 2 * FDOH + 10], np.float64)
    syz_a[slices] = np.random.rand(10, 10, 10)
    
    
    M = np.float64(np.random.rand(10, 10, 10))
    mu = np.float64(np.random.rand(10, 10, 10))
    rho = np.float64(np.random.rand(10, 10, 10))
    
    print("Dot product for the surface kernel")
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = surface(vx, vy, vz, sxx, syy, szz, sxz, sxy, syz, rho, M, mu)
    (Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a) = surface_adj(vx_a, vy_a, vz_a, sxx_a, syy_a, szz_a, sxz_a, sxy_a, syz_a, rho, M, mu)

    prod1 = (  np.sum(vx_a*Fvx)
             + np.sum(vy_a*Fvy)
             + np.sum(vz_a*Fvz)
             + np.sum(sxx_a*Fsxx)
             + np.sum(syy_a*Fsyy)
             + np.sum(szz_a*Fszz)
             + np.sum(sxz_a * Fsxz)
             + np.sum(sxy_a * Fsxy)
             + np.sum(syz_a * Fsyz)
    )
    prod2 = (   np.sum(Fvx_a*vx)
             + np.sum(Fvy_a*vy)
             + np.sum(Fvz_a*vz)
             + np.sum(Fsxx_a*sxx)
             + np.sum(Fsyy_a*syy)
             + np.sum(Fszz_a*szz)
             + np.sum(Fsxz_a * sxz)
             + np.sum(Fsxy_a * sxy)
             + np.sum(Fsyz_a * syz)
    )

    print(prod2-prod1)

    print("Testing L is inverse of L^-1")
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = apply_L(vx, vy, vz, sxx, syy, szz, sxz, sxy, syz, rho, M, mu)
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = apply_Lm(Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz, rho, M, mu)
    diff = vx-Fvx + vy-Fvy + vz-Fvz + sxz-Fsxz + sxy-Fsxy + syz-Fsyz + sxx-Fsxx + syy-Fsyy + szz-Fszz
    print(np.max(diff))

    print("Testing T is inverse of T")
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = apply_T(vx, vy, vz, sxx, syy, szz, sxz, sxy, syz, rho, M, mu)
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = apply_T(Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz, rho, M, mu)
    diff = vx-Fvx + vy-Fvy + vz-Fvz + sxz-Fsxz + sxy-Fsxy + syz-Fsyz + sxx-Fsxx + syy-Fsyy + szz-Fszz
    print(np.max(diff))

    print("Dot product for LFT")
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = apply_T(vx, vy, vz, sxx, syy, szz, sxz, sxy, syz, rho, M, mu)
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = update_v(Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz, rho, M, mu)
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = update_s(Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz, rho, M, mu)
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = apply_L(Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz, rho, M, mu)
    
    (Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a) = apply_T(vx_a, vy_a, vz_a, sxx_a, syy_a, szz_a, sxz_a, sxy_a, syz_a, rho, M, mu)
    (Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a) = update_v(Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a, rho, M, mu)
    (Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a) = update_s(Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a, rho, M, mu)
    (Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a) = apply_L(Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a, rho, M, mu)

    prod1 = (  np.sum(vx_a*Fvx)
             + np.sum(vy_a*Fvy)
             + np.sum(vz_a*Fvz)
             + np.sum(sxx_a*Fsxx)
             + np.sum(syy_a*Fsyy)
             + np.sum(szz_a*Fszz)
             + np.sum(sxz_a * Fsxz)
             + np.sum(sxy_a * Fsxy)
             + np.sum(syz_a * Fsyz)
    )
    prod2 = (   np.sum(Fvx_a*vx)
             + np.sum(Fvy_a*vy)
             + np.sum(Fvz_a*vz)
             + np.sum(Fsxx_a*sxx)
             + np.sum(Fsyy_a*syy)
             + np.sum(Fszz_a*szz)
             + np.sum(Fsxz_a * sxz)
             + np.sum(Fsxy_a * sxy)
             + np.sum(Fsyz_a * syz)
    )

    print(prod2-prod1)

    print("Dot product for F_s' = LSFT and F_s'^* = L F T L^-1 S^* L")
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = apply_T(vx, vy, vz, sxx, syy, szz, sxz, sxy, syz, rho, M, mu)
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = update_v(Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz, rho, M, mu)
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = update_s(Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz, rho, M, mu)
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = surface(Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz, rho, M, mu)
    (Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz) = apply_L(Fvx, Fvy, Fvz, Fsxx, Fsyy, Fszz, Fsxz, Fsxy, Fsyz, rho, M, mu)
    
    
    (Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a) = apply_L(vx_a, vy_a, vz_a, sxx_a, syy_a, szz_a, sxz_a, sxy_a, syz_a, rho, M, mu)
    (Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a) = surface_adj(Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a, rho, M, mu)
    (Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a) = apply_Lm(Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a, rho, M, mu)
    (Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a) = apply_T(Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a, rho, M, mu)
    (Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a) = update_v(Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a, rho, M, mu)
    (Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a) = update_s(Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a, rho, M, mu)
    (Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a) = apply_L(Fvx_a, Fvy_a, Fvz_a, Fsxx_a, Fsyy_a, Fszz_a, Fsxz_a, Fsxy_a, Fsyz_a, rho, M, mu)

    prod1 = (  np.sum(vx_a*Fvx)
             + np.sum(vy_a*Fvy)
             + np.sum(vz_a*Fvz)
             + np.sum(sxx_a*Fsxx)
             + np.sum(syy_a*Fsyy)
             + np.sum(szz_a*Fszz)
             + np.sum(sxz_a * Fsxz)
             + np.sum(sxy_a * Fsxy)
             + np.sum(syz_a * Fsyz)
    )
    prod2 = (   np.sum(Fvx_a*vx)
             + np.sum(Fvy_a*vy)
             + np.sum(Fvz_a*vz)
             + np.sum(Fsxx_a*sxx)
             + np.sum(Fsyy_a*syy)
             + np.sum(Fszz_a*szz)
             + np.sum(Fsxz_a * sxz)
             + np.sum(Fsxy_a * sxy)
             + np.sum(Fsyz_a * syz)
    )

    print(prod2-prod1)










