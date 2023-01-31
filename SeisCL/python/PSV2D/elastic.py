from SeisCL.python.pycl_backend import (GridCL,
                                        FunctionGPU,
                                        ReversibleFunctionCL,
                                        ComputeRessource,
                                        SequenceCL,
                                        PropagatorCL,
                                        )
from SeisCL.python.FDstencils import get_pos_header, FDCoefficients, CUDACL_header
from SeisCL.python.seis2D import ReversibleFunction, Sequence, Propagator, ricker, Cerjan
from SeisCL.python.common.sources import Source
from SeisCL.python.common.receivers import Receiver
import numpy as np
from copy import copy
from pyopencl.array import max
from SeisCL.python.common.vel2lame import Velocity2LameCL
from SeisCL.python.common.scaling import ScaledParameters
from SeisCL.python.common.averaging import ArithmeticAveraging, HarmonicAveraging
import matplotlib.pyplot as plt

#TODO interface with backward compatibility with SeisCL
#TODO abs cerjan
#TODO PML



class UpdateStress(ReversibleFunctionCL):
    forward_src = """
FUNDEF void UpdateStress(grid pos,
                         GLOBARG float *vx,         GLOBARG float *vz,
                         GLOBARG float *sxx,        GLOBARG float *szz,
                         GLOBARG float *sxz,        GLOBARG float *M,
                         GLOBARG float *mu,         GLOBARG float *muipkp,
                         int backpropagate)
{
    
    get_pos(&pos);
    LOCID float lvar[__LSIZE__];
    
    float vxx, vzz, vzx, vxz;
    int ind0 = indg(pos,0,0,0);
    
// Calculation of the velocity spatial derivatives
#if LOCAL_OFF==0
    load_local_in(pos, vx, lvar);
    load_local_haloz(pos, vx, lvar);
    load_local_halox(pos, vx, lvar);
    BARRIER
#endif
    vxx = Dxm(pos, lvar);
    vxz = Dzp(pos, lvar);
    
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, vz, lvar);
    load_local_haloz(pos, vz, lvar);
    load_local_halox(pos, vz, lvar);
    BARRIER
#endif
    vzz = Dzm(pos, lvar);
    vzx = Dxp(pos, lvar);
    
    gridstop(pos);

// Update the stresses
    int sign = -2*backpropagate+1; 
    sxz[ind0]+= sign * (muipkp[ind0]*(vxz+vzx));
    sxx[ind0]+= sign * ((M[ind0]*(vxx+vzz))-(2.0*mu[ind0]*vzz));
    szz[ind0]+= sign * ((M[ind0]*(vxx+vzz))-(2.0*mu[ind0]*vxx));
}
"""

    linear_src = """
FUNDEF void UpdateStress_lin(grid pos,
                       GLOBARG float *vx,         GLOBARG float *vz,
                       GLOBARG float *sxx,        GLOBARG float *szz,
                       GLOBARG float *sxz,        GLOBARG float *M,
                       GLOBARG float *mu,         GLOBARG float *muipkp,
                       GLOBARG float *vx_lin,     GLOBARG float *vz_lin,
                       GLOBARG float *sxx_lin,    GLOBARG float *szz_lin,
                       GLOBARG float *sxz_lin,    GLOBARG float *M_lin,
                       GLOBARG float *mu_lin,     GLOBARG float *muipkp_lin)
{
    
    get_pos(&pos);
    LOCID float lvar[__LSIZE__];
    
    float vxx, vzz, vzx, vxz;
    float vxx_lin, vzz_lin, vzx_lin, vxz_lin;
    int ind0 = indg(pos,0,0,0);
    
// Calculation of the velocity spatial derivatives
#if LOCAL_OFF==0
    load_local_in(pos, vx, lvar);
    load_local_haloz(pos, vx, lvar);
    load_local_halox(pos, vx, lvar);
    BARRIER
#endif
    vxx = Dxm(pos, lvar);
    vxz = Dzp(pos, lvar);
    
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, vx_lin, lvar);
    load_local_haloz(pos, vx_lin, lvar);
    load_local_halox(pos, vx_lin, lvar);
    BARRIER
#endif
    vxx_lin = Dxm(pos, lvar);
    vxz_lin = Dzp(pos, lvar);
    
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, vz, lvar);
    load_local_haloz(pos, vz, lvar);
    load_local_halox(pos, vz, lvar);
    BARRIER
#endif
    vzz = Dzm(pos, lvar);
    vzx = Dxp(pos, lvar);
    
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, vz_lin, lvar);
    load_local_haloz(pos, vz_lin, lvar);
    load_local_halox(pos, vz_lin, lvar);
    BARRIER
#endif
    vzz_lin = Dzm(pos, lvar);
    vzx_lin = Dxp(pos, lvar);
    gridstop(pos);
    
    sxz_lin[ind0]+= muipkp[ind0]*(vxz_lin+vzx_lin)\
                   +muipkp_lin[ind0]*(vxz+vzx);
    sxx_lin[ind0]+= M[ind0]*(vxx_lin+vzz_lin)-(2.0*mu[ind0]*vzz_lin) \
                   +M_lin[ind0]*(vxx+vzz)-(2.0*mu_lin[ind0]*vzz);
    szz_lin[ind0]+= M[ind0]*(vxx_lin+vzz_lin)-(2.0*mu[ind0]*vxx_lin) \
                   +M_lin[ind0]*(vxx+vzz)-(2.0*mu_lin[ind0]*vxx);
}
"""

    adjoint_src = """
FUNDEF void UpdateStress_adj(grid pos,
                       GLOBARG float *vx,         GLOBARG float *vz,
                       GLOBARG float *sxx,        GLOBARG float *szz,
                       GLOBARG float *sxz,        GLOBARG float *M,
                       GLOBARG float *mu,         GLOBARG float *muipkp,
                       GLOBARG float *vx_adj,     GLOBARG float *vz_adj,
                       GLOBARG float *sxx_adj,    GLOBARG float *szz_adj,
                       GLOBARG float *sxz_adj,    GLOBARG float *M_adj,
                       GLOBARG float *mu_adj,     GLOBARG float *muipkp_adj)
{
    
    get_pos(&pos);
    LOCID float lvar[__LSIZE__];
    
    float vxx, vzz, vzx, vxz;
    float sxx_x_adj, sxx_z_adj, szz_x_adj, szz_z_adj, sxz_x_adj, sxz_z_adj;
    int ind0 = indg(pos,0,0,0);
    
// Calculation of the velocity spatial derivatives
#if LOCAL_OFF==0
    load_local_in(pos, vx, lvar);
    load_local_haloz(pos, vx, lvar);
    load_local_halox(pos, vx, lvar);
    BARRIER
#endif
    vxx = Dxm(pos, lvar);
    vxz = Dzp(pos, lvar);
    
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, vz, lvar);
    load_local_haloz(pos, vz, lvar);
    load_local_halox(pos, vz, lvar);
    BARRIER
#endif
    vzz = Dzm(pos, lvar);
    vzx = Dxp(pos, lvar);
    
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, sxz_adj, lvar);
    mul_local_in(pos, muipkp, lvar);
    load_local_haloz(pos, sxz_adj, lvar);
    mul_local_haloz(pos, muipkp, lvar);
    load_local_halox(pos, sxz_adj, lvar);
    mul_local_halox(pos, muipkp, lvar);
    BARRIER
#endif
    sxz_x_adj = -Dxm(pos, lvar);
    sxz_z_adj = -Dzm(pos, lvar);

#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, sxx_adj, lvar);
    mul_local_in(pos, M, lvar);
    load_local_haloz(pos, sxx_adj, lvar);
    mul_local_haloz(pos, M, lvar);
    load_local_halox(pos, sxx_adj, lvar);
    mul_local_halox(pos, M, lvar);
    BARRIER
#endif
    sxx_x_adj = -Dxp(pos, lvar);
    sxx_z_adj = -Dzp(pos, lvar);

#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, sxx_adj, lvar);
    mul_local_in(pos, mu, lvar);
    load_local_haloz(pos, sxx_adj, lvar);
    mul_local_haloz(pos, mu, lvar);
    BARRIER
#endif
    sxx_z_adj += 2.0*Dzp(pos, lvar);

#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, szz_adj, lvar);
    mul_local_in(pos, M, lvar);
    load_local_haloz(pos, szz_adj, lvar);
    mul_local_haloz(pos, M, lvar);
    load_local_halox(pos, szz_adj, lvar);
    mul_local_halox(pos, M, lvar);
    BARRIER
#endif
    szz_x_adj = -Dxp(pos, lvar);
    szz_z_adj = -Dzp(pos, lvar);

#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, szz_adj, lvar);
    mul_local_in(pos, mu, lvar);
    load_local_halox(pos, szz_adj, lvar);
    mul_local_halox(pos, mu, lvar);
    BARRIER
#endif
    szz_x_adj += 2.0*Dxp(pos, lvar);
      
    gridstop(pos);
    
    vx_adj[ind0] += sxx_x_adj + szz_x_adj + sxz_z_adj;
    vz_adj[ind0] += sxx_z_adj + szz_z_adj + sxz_x_adj;
    
    M_adj[ind0] += (vxx + vzz) * (sxx_adj[ind0] + szz_adj[ind0]);
    mu_adj[ind0] += - 2.0 * (vzz * sxx_adj[ind0] + vxx * szz_adj[ind0]);
    muipkp_adj[ind0] += (vxz + vzx) * sxz_adj[ind0];
}
"""

    def __init__(self, grids=None, computegrid=None, fdcoefs=None,
                 local_size=(16, 16), local_off=0, **kwargs):
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "M", "mu", "muipkp"]
        self.updated_states = ["sxx", "szz", "sxz"]
        self.default_grids = {"vx": "gridfd",
                              "vz": "gridfd",
                              "sxx": "gridfd",
                              "szz": "gridfd",
                              "sxz": "gridfd",
                              "M": "gridpar",
                              "mu": "gridpar",
                              "muipkp": "gridpar"}
        if fdcoefs is None:
            fdcoefs = FDCoefficients(order=grids["gridfd"].pad*2,
                                     local_off=local_off)
        if fdcoefs.order//2 != grids["gridfd"].pad:
            raise ValueError("The grid padding should be equal to half the "
                             "fd stencil width")
        self.headers = fdcoefs.header()
        options = fdcoefs.options
        super().__init__(grids=grids, computegrid=computegrid,
                         local_size=local_size, options=options, **kwargs)


class UpdateVelocity(ReversibleFunctionCL):
    forward_src = """

FUNDEF void UpdateVelocity(grid pos,
                     GLOBARG float *vx,      GLOBARG float *vz,
                     GLOBARG float *sxx,     GLOBARG float *szz,     
                     GLOBARG float *sxz,     GLOBARG float *rip,
                     GLOBARG float *rkp,     int backpropagate)
{

    get_pos(&pos);
    LOCID float lvar[__LSIZE__];
    float sxx_x;
    float szz_z;
    float sxz_x;
    float sxz_z;
    
    int ind0 = indg(pos,0,0,0);
    
    // Calculation of the stresses spatial derivatives
#if LOCAL_OFF==0
    load_local_in(pos, sxx, lvar);
    load_local_halox(pos, sxx, lvar);
    BARRIER
#endif
    sxx_x = Dxp(pos, lvar);
        
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, szz, lvar);
    load_local_haloz(pos, szz, lvar);
    BARRIER
#endif
    szz_z = Dzp(pos, lvar);
        
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, sxz, lvar);
    load_local_haloz(pos, sxz, lvar);
    load_local_halox(pos, sxz, lvar);
    BARRIER
#endif
    sxz_z = Dzm(pos, lvar);
    sxz_x = Dxm(pos, lvar);

    gridstop(pos);
    
    // Update the velocities
    int sign = -2*backpropagate+1; 
    vx[ind0]+= sign * ((sxx_x + sxz_z)*rip[ind0]);
    vz[ind0]+= sign * ((szz_z + sxz_x)*rkp[ind0]);
}
"""

    linear_src = """

FUNDEF void UpdateVelocity_lin(grid pos,
                               GLOBARG float *vx,      GLOBARG float *vz,
                               GLOBARG float *sxx,     GLOBARG float *szz,     
                               GLOBARG float *sxz,     GLOBARG float *rip,
                               GLOBARG float *rkp,
                               GLOBARG float *vx_lin,  GLOBARG float *vz_lin,
                               GLOBARG float *sxx_lin, GLOBARG float *szz_lin,     
                               GLOBARG float *sxz_lin, GLOBARG float *rip_lin,
                               GLOBARG float *rkp_lin)
{

    get_pos(&pos);
    LOCID float lvar[__LSIZE__];
    float sxx_x, szz_z, sxz_x, sxz_z;
    float sxx_x_lin, szz_z_lin, sxz_x_lin, sxz_z_lin;
    
    int ind0 = indg(pos,0,0,0);
    
    // Calculation of the stresses spatial derivatives
#if LOCAL_OFF==0
    load_local_in(pos, sxx, lvar);
    load_local_halox(pos, sxx, lvar);
    BARRIER
#endif
    sxx_x = Dxp(pos, lvar);
        
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, szz, lvar);
    load_local_haloz(pos, szz, lvar);
    BARRIER
#endif
    szz_z = Dzp(pos, lvar);
        
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, sxz, lvar);
    load_local_haloz(pos, sxz, lvar);
    load_local_halox(pos, sxz, lvar);
    BARRIER
#endif
    sxz_z = Dzm(pos, lvar);
    sxz_x = Dxm(pos, lvar);

#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, sxx_lin, lvar);
    load_local_halox(pos, sxx_lin, lvar);
    BARRIER
#endif
    sxx_x_lin = Dxp(pos, lvar);
        
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, szz_lin, lvar);
    load_local_haloz(pos, szz_lin, lvar);
    BARRIER
#endif
    szz_z_lin = Dzp(pos, lvar);
        
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, sxz_lin, lvar);
    load_local_haloz(pos, sxz_lin, lvar);
    load_local_halox(pos, sxz_lin, lvar);
    BARRIER
#endif
    sxz_z_lin = Dzm(pos, lvar);
    sxz_x_lin = Dxm(pos, lvar);

    gridstop(pos);
    
    // Update the velocities
    vx_lin[ind0]+= (sxx_x_lin + sxz_z_lin)*rip[ind0] \
                   +(sxx_x + sxz_z)*rip_lin[ind0];
    vz_lin[ind0]+= (szz_z_lin + sxz_x_lin)*rkp[ind0] \
                   +(szz_z + sxz_x)*rkp_lin[ind0];
}
"""

    adjoint_src = """

FUNDEF void UpdateVelocity_adj(grid pos,
                               GLOBARG float *vx,      GLOBARG float *vz,
                               GLOBARG float *sxx,     GLOBARG float *szz,     
                               GLOBARG float *sxz,     GLOBARG float *rip,
                               GLOBARG float *rkp,
                               GLOBARG float *vx_adj,  GLOBARG float *vz_adj,
                               GLOBARG float *sxx_adj, GLOBARG float *szz_adj,     
                               GLOBARG float *sxz_adj, GLOBARG float *rip_adj,
                               GLOBARG float *rkp_adj)
{

    get_pos(&pos);
    LOCID float lvar[__LSIZE__];
    float sxx_x, szz_z, sxz_x, sxz_z;
    float vxx_adj, vxz_adj, vzz_adj, vzx_adj;
    
    int ind0 = indg(pos,0,0,0);
    
    // Calculation of the stresses spatial derivatives
#if LOCAL_OFF==0
    load_local_in(pos, sxx, lvar);
    load_local_halox(pos, sxx, lvar);
    BARRIER
#endif
    sxx_x = Dxp(pos, lvar);
        
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, szz, lvar);
    load_local_haloz(pos, szz, lvar);
    BARRIER
#endif
    szz_z = Dzp(pos, lvar);
        
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, sxz, lvar);
    load_local_haloz(pos, sxz, lvar);
    load_local_halox(pos, sxz, lvar);
    BARRIER
#endif
    sxz_z = Dzm(pos, lvar);
    sxz_x = Dxm(pos, lvar);

#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, vx_adj, lvar);
    mul_local_in(pos, rip, lvar);
    load_local_haloz(pos, vx_adj, lvar);
    mul_local_haloz(pos, rip, lvar);
    load_local_halox(pos, vx_adj, lvar);
    mul_local_halox(pos, rip, lvar);
    BARRIER
#endif
    vxx_adj = -Dxm(pos, lvar);
    vxz_adj = -Dzp(pos, lvar);
    
#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, vz_adj, lvar);
    mul_local_in(pos, rkp, lvar);
    load_local_haloz(pos, vz_adj, lvar);
    mul_local_haloz(pos, rkp, lvar);
    load_local_halox(pos, vz_adj, lvar);
    mul_local_halox(pos, rkp, lvar);
    BARRIER
#endif
    vzx_adj = -Dxp(pos, lvar);
    vzz_adj = -Dzm(pos, lvar);
    
    gridstop(pos);
    
    // Update the velocities
    sxx_adj[ind0] += vxx_adj;
    szz_adj[ind0] += vzz_adj;
    sxz_adj[ind0] += vxz_adj + vzx_adj;
    
    rip_adj[ind0] += (sxx_x + sxz_z) * vx_adj[ind0];
    rkp_adj[ind0] += (szz_z + sxz_x) * vz_adj[ind0];
}
"""

    def __init__(self, grids=None, computegrid=None, fdcoefs=None,
                 local_size=(16, 16), local_off=0, **kwargs):
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "rip", "rkp"]
        self.updated_states = ["vx", "vz"]
        self.default_grids = {"vx": "gridfd",
                              "vz": "gridfd",
                              "sxx": "gridfd",
                              "szz": "gridfd",
                              "sxz": "gridfd",
                              "rip": "gridpar",
                              "rkp": "gridpar"}
        if fdcoefs is None:
            fdcoefs = FDCoefficients(order=grids["gridfd"].pad*2,
                                     local_off=local_off)
        self.headers = fdcoefs.header()
        options = fdcoefs.options
        super().__init__(grids=grids, computegrid=computegrid, options=options,
                         local_size=local_size, **kwargs)


class FreeSurface(ReversibleFunctionCL):

    forward_src = """
FUNDEF void FreeSurface1(grid pos,
                         GLOBARG float *vx,         GLOBARG float *vz,
                         GLOBARG float *sxx,        GLOBARG float *szz,
                         GLOBARG float *sxz,        GLOBARG float *M,
                         GLOBARG float *mu,         GLOBARG float *rkp,
                         GLOBARG float *rip,        int backpropagate)
{
    get_pos(&pos);
    float vxx, vzz;
    float g, f;
    int ind0 = indg(pos,0,0,0);
    int sign = -2*backpropagate+1; 
        
    vxx = Dxm(pos, vx);
    vzz = Dzm(pos, vz);
    //int i, j;
    //vxx = vzz = 0;
    //for (i=0; i<__FDOH__; i++){
    //    vxx += hc[i] * (vx[indg(pos, 0, 0, i)] - vx[indg(pos, 0, 0, -i-1)]);
    //    vzz += hc[i] * (vz[indg(pos, i, 0, 0)] - vz[indg(pos, -i-1, 0, 0)]);
    //}
    f = mu[ind0] * 2.0;
    g = M[ind0];
    sxx[ind0] += sign * (-((g - f) * (g - f) * vxx / g) - ((g - f) * vzz));
    //szz[ind0] = 0;
    szz[ind0]+= sign * -((g*(vxx+vzz))-(f*vxx));
}
FUNDEF void FreeSurface2(grid pos,
                         GLOBARG float *vx,         GLOBARG float *vz,
                         GLOBARG float *sxx,        GLOBARG float *szz,
                         GLOBARG float *sxz,        GLOBARG float *M,
                         GLOBARG float *mu,         GLOBARG float *rkp,
                         GLOBARG float *rip,        int backpropagate)
{
    get_pos(&pos);
    float szz_z, sxz_z;
    int i, j;
    int indi;
    
    int sign = -2*backpropagate+1; 
    for (i=0; i<__FDOH__; i++){
        sxz_z = szz_z = 0;
        for (j=i+1; j<__FDOH__; j++){
            indi = indg(pos,j-i,0,0);
            szz_z += hc[j] * szz[indi];
        }
        for (j=i; j<__FDOH__; j++){
            indi = indg(pos,j-i+1,0,0);
            sxz_z += hc[j] * sxz[indi];
        }
        indi = indg(pos,i,0,0);
        vx[indi] += sign * sxz_z * rip[indi];
        vz[indi] += sign * szz_z * rkp[indi];
    }
}
    """

    linear_src = """
FUNDEF void FreeSurface_lin1(grid pos,
                             GLOBARG float *vx,         GLOBARG float *vz,
                             GLOBARG float *sxx,        GLOBARG float *szz,
                             GLOBARG float *sxz,        GLOBARG float *M,
                             GLOBARG float *mu,         GLOBARG float *rkp,
                             GLOBARG float *rip,
                             GLOBARG float *vx_lin,     GLOBARG float *vz_lin,
                             GLOBARG float *sxx_lin,    GLOBARG float *szz_lin,
                             GLOBARG float *sxz_lin,    GLOBARG float *M_lin,
                             GLOBARG float *mu_lin,     GLOBARG float *rkp_lin,
                             GLOBARG float *rip_lin)
{
    get_pos(&pos);
    float vxx, vzz;
    float g, f, dg, df;
    int ind0 = indg(pos,0,0,0);
    
    vxx = Dxm(pos, vx_lin);
    vzz = Dzm(pos, vz_lin);
    f = mu[ind0] * 2.0;
    g = M[ind0];
    sxx_lin[ind0] += -((g - f) * (g - f) * vxx / g) - ((g - f) * vzz);
    //szz_lin[ind0] = 0;
    szz_lin[ind0]+= -((g*(vxx+vzz))-(f*vxx));
    
    vxx = Dxm(pos, vx);
    vzz = Dzm(pos, vz);

    df = mu_lin[ind0] * 2.0;
    dg = M_lin[ind0];
    sxx_lin[ind0] += (2.0 * (g - f) * vxx / g + vzz) * df +\
                     (-2.0 * (g - f) * vxx / g 
                     + (g - f)*(g - f) / g / g * vxx - vzz) * dg;
    szz_lin[ind0] += -((dg*(vxx+vzz))-(df*vxx));
}
FUNDEF void FreeSurface_lin2(grid pos,
                             GLOBARG float *vx,         GLOBARG float *vz,
                             GLOBARG float *sxx,        GLOBARG float *szz,
                             GLOBARG float *sxz,        GLOBARG float *M,
                             GLOBARG float *mu,         GLOBARG float *rkp,
                             GLOBARG float *rip,
                             GLOBARG float *vx_lin,     GLOBARG float *vz_lin,
                             GLOBARG float *sxx_lin,    GLOBARG float *szz_lin,
                             GLOBARG float *sxz_lin,    GLOBARG float *M_lin,
                             GLOBARG float *mu_lin,     GLOBARG float *rkp_lin,
                             GLOBARG float *rip_lin)
{
    get_pos(&pos);
    float szz_z, sxz_z;
    int i, j;
    int indi;

    for (i=0; i<__FDOH__; i++){
        sxz_z = szz_z = 0;
        for (j=i+1; j<__FDOH__; j++){
            indi = indg(pos,j-i,0,0);
            szz_z += hc[j] * szz_lin[indi];
        }
        for (j=i; j<__FDOH__; j++){
            indi = indg(pos,j-i+1,0,0);
            sxz_z += hc[j] * sxz_lin[indi];
        }
        indi = indg(pos,i,0,0);
        vx_lin[indi] += sxz_z * rip[indi];
        vz_lin[indi] += szz_z * rkp[indi];
    }
    
    for (i=0; i<__FDOH__; i++){
        sxz_z = szz_z = 0;
        for (j=i+1; j<__FDOH__; j++){
            indi = indg(pos,j-i,0,0);
            szz_z += hc[j] * szz[indi];
        }
        for (j=i; j<__FDOH__; j++){
            indi = indg(pos,j-i+1,0,0);
            sxz_z += hc[j] * sxz[indi];
        }
        indi = indg(pos,i,0,0);
        vx_lin[indi] += sxz_z * rip_lin[indi];
        vz_lin[indi] += szz_z * rkp_lin[indi];
    }
}
"""

    adjoint_src = """
    FUNDEF void FreeSurface_adj1(grid pos,
                         GLOBARG float *vx,         GLOBARG float *vz,
                         GLOBARG float *sxx,        GLOBARG float *szz,
                         GLOBARG float *sxz,        GLOBARG float *rkp,
                         GLOBARG float *rip,        GLOBARG float *M,
                         GLOBARG float *mu, 
                         GLOBARG float *vx_adj,     GLOBARG float *vz_adj,
                         GLOBARG float *sxx_adj,    GLOBARG float *szz_adj,
                         GLOBARG float *sxz_adj,    GLOBARG float *rkp_adj,
                         GLOBARG float *rip_adj,    GLOBARG float *M_adj,
                         GLOBARG float *mu_adj)
{
    get_pos(&pos);
    float szz_z, sxz_z;
    int i, j;
    int indi, ind0;

    for (i=0; i<__FDOH__; i++){
        ind0 = indg(pos,i,0,0);
        for (j=i+1; j<__FDOH__; j++){
            indi = indg(pos,j-i,0,0);
            szz_adj[indi] += rkp[ind0] * hc[j] * vz_adj[ind0];
        }
        for (j=i; j<__FDOH__; j++){
            indi = indg(pos,j-i+1,0,0);
            sxz_adj[indi] += rip[ind0] * hc[j] * vx_adj[ind0];
        }
    }
    
    for (i=0; i<__FDOH__; i++){
        sxz_z = szz_z = 0;
        for (j=i+1; j<__FDOH__; j++){
            indi = indg(pos,j-i,0,0);
            szz_z += hc[j] * szz[indi];
        }
        for (j=i; j<__FDOH__; j++){
            indi = indg(pos,j-i+1,0,0);
            sxz_z += hc[j] * sxz[indi];
        }
        indi = indg(pos,i,0,0);
        rip_adj[indi] += sxz_z * vx_adj[indi];
        rkp_adj[indi] += szz_z * vz_adj[indi];
    }
}
FUNDEF void FreeSurface_adj2(grid pos,
                             GLOBARG float *vx,         GLOBARG float *vz,
                             GLOBARG float *sxx,        GLOBARG float *szz,
                             GLOBARG float *sxz,        GLOBARG float *rkp,
                             GLOBARG float *rip,        GLOBARG float *M,
                             GLOBARG float *mu, 
                             GLOBARG float *vx_adj,     GLOBARG float *vz_adj,
                             GLOBARG float *sxx_adj,    GLOBARG float *szz_adj,
                             GLOBARG float *sxz_adj,    GLOBARG float *rkp_adj,
                             GLOBARG float *rip_adj,    GLOBARG float *M_adj,
                             GLOBARG float *mu_adj)
{
    get_pos(&pos);
    float vxx, vzz;
    float g, f, g_adj, f_adj;
    int ind0 = indg(pos,0,0,0);
    int i;
    float hx, hz;
    

    f = mu[ind0] * 2.0;
    g = M[ind0];
    hx = -((g - f) * (g - f) / g);
    hz = - (g - f);
    for (i=0; i<__FDOH__; i++){
        vx_adj[indg(pos, 0, 0, i)] += hc[i] * hx * sxx_adj[ind0];
        vx_adj[indg(pos, 0, 0, -i-1)] += - hc[i] * hx * sxx_adj[ind0];
        vz_adj[indg(pos, i, 0, 0)] += hc[i] * hz * sxx_adj[ind0];
        vz_adj[indg(pos, -i-1, 0, 0)] += - hc[i] * hz * sxx_adj[ind0];
        
        vx_adj[indg(pos, 0, 0, i)] += hc[i] * hz * szz_adj[ind0];
        vx_adj[indg(pos, 0, 0, -i-1)] += - hc[i] * hz * szz_adj[ind0];
        vz_adj[indg(pos, i, 0, 0)] += hc[i] * (-g) * szz_adj[ind0];
        vz_adj[indg(pos, -i-1, 0, 0)] += - hc[i] * (-g) * szz_adj[ind0];
    }
    
    //szz_adj[ind0] = 0;
    
    vxx = Dxm(pos, vx);
    vzz = Dzm(pos, vz);

    f_adj = mu_adj[ind0] * 2.0;
    g_adj = M_adj[ind0];
    M_adj[ind0] += (-2.0 * (g - f) * vxx / g 
                  + (g - f)*(g - f) / g / g * vxx - vzz) * sxx_adj[ind0];
    mu_adj[ind0] += 2.0 * (2.0 * (g - f) * vxx / g + vzz) * sxx_adj[ind0];
    
    M_adj[ind0] += -(vxx + vzz) *  szz_adj[ind0];
    mu_adj[ind0] += 2.0 * vxx *  szz_adj[ind0];
}
"""

    def __init__(self, grids=None, fdcoefs=None, **kwargs):
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "mu", "M",
                                "rip", "rkp"]
        self.updated_states = ["sxx", "szz", "vx", "vz"]
        self.default_grids = {"vx": "gridfd",
                              "vz": "gridfd",
                              "sxx": "gridfd",
                              "szz": "gridfd",
                              "sxz": "gridfd",
                              "M": "gridpar",
                              "mu": "gridpar",
                              "muipkp": "gridpar",
                              "rip": "gridpar",
                              "rkp": "gridpar"}
        if fdcoefs is None:
            fdcoefs = FDCoefficients(order=grids["gridfd"].pad*2,
                                     local_off=1)
        if fdcoefs.order//2 != grids["gridfd"].pad:
            raise ValueError("The grid padding should be equal to half the "
                             "fd stencil width")
        self.headers = fdcoefs.header()
        computegrid = copy(grids["gridfd"])
        options = fdcoefs.options
        super().__init__(grids=grids, computegrid=computegrid, options=options,
                         **kwargs)

    def global_size_fw(self):
        gsize = [np.int32(s - 2 * self.computegrid.pad)
                 for s in self.computegrid.shape]
        gsize[0] = 1
        return gsize


class ScaledParameters(ReversibleFunction):

    def __init__(self, grids=None, dt=1, dh=1, **kwargs):
        super().__init__(grids, **kwargs)
        self.required_states = ["M", "mu", "muipkp", "rho", "rip", "rkp"]
        self.updated_states = ["M", "mu", "muipkp", "rho", "rip", "rkp"]
        self.sc = 1.0
        self.dt = dt
        self.dh = dh
        self.default_grids = {el: "gridpar" for el in self.required_states}

    @staticmethod
    def scale(M, dt, dx):
        return int(np.log2(max(M).get() * dt / dx))

    def forward(self, states, **kwargs):

        dt = self.dt
        dh = self.dh
        self.sc = sc = self.scale(states["M"], dt, dh)
        states["rho"] = 2 ** sc * dt / dh / states["rho"]
        states["rip"] = 2 ** sc * dt / dh / states["rip"]
        states["rkp"] = 2 ** sc * dt / dh / states["rkp"]
        states["M"] = 2 ** -sc * dt / dh * states["M"]
        states["mu"] = 2 ** -sc * dt / dh * states["mu"]
        states["muipkp"] = 2 ** -sc * dt / dh * states["muipkp"]

        return states

    def linear(self, dstates, states, **kwargs):

        sc = self.sc
        dt = self.dt
        dh = self.dh
        dstates["rho"] = -2 ** sc * dt / dh / states["rho"] ** 2 * dstates["rho"]
        dstates["rip"] = -2 ** sc * dt / dh / states["rip"] ** 2 * dstates["rip"]
        dstates["rkp"] = -2 ** sc * dt / dh / states["rkp"] ** 2 * dstates["rkp"]
        dstates["M"] = 2 ** -sc * dt / dh * dstates["M"]
        dstates["mu"] = 2 ** -sc * dt / dh * dstates["mu"]
        dstates["muipkp"] = 2 ** -sc * dt / dh * dstates["muipkp"]

        return dstates

    def adjoint(self, adj_states, states,  **kwargs):

        sc = self.sc
        dt = self.dt
        dh = self.dh
        adj_states["rho"] = -2 ** sc * dt / dh / states["rho"] ** 2 * adj_states["rho"]
        adj_states["rip"] = -2 ** sc * dt / dh / states["rip"] ** 2 * adj_states["rip"]
        adj_states["rkp"] = -2 ** sc * dt / dh / states["rkp"] ** 2 * adj_states["rkp"]
        adj_states["M"] = 2 ** -sc * dt / dh * adj_states["M"]
        adj_states["mu"] = 2 ** -sc * dt / dh * adj_states["mu"]
        adj_states["muipkp"] = 2 ** -sc * dt / dh * adj_states["muipkp"]

        return adj_states

    def backward(self, states, **kwargs):

        sc = self.sc
        dt = self.dt
        dh = self.dh
        states["rho"] = 2 ** sc * dt / dh / states["rho"]
        states["rip"] = 2 ** sc * dt / dh / states["rip"]
        states["rkp"] = 2 ** sc * dt / dh / states["rkp"]
        states["M"] = 2 ** sc / dt * dh * states["M"]
        states["mu"] = 2 ** sc / dt * dh * states["mu"]
        states["muipkp"] = 2 ** sc / dt * dh * states["muipkp"]

        return states


class Cerjan(FunctionGPU):

    def __init__(self, grids=None, freesurf=False, abpc=4.0, nab=2,
                 required_states=(), **kwargs):
        self.abpc = abpc
        self.nab = nab
        self.required_states = required_states
        self.updated_states = required_states
        self.taper = np.exp(np.log(1.0-abpc/100)/nab**2 * np.arange(nab) **2)
        self.taper = np.expand_dims(self.taper, -1)
        self.freesurf = freesurf
        self.default_grids = {el: "gridfd" for el in self.required_states}
        super().__init__(grids, **kwargs)

    @property
    def updated_regions(self):
        regions = []
        pad = self.grids[self.updated_states[0]].pad
        ndim = len(self.grids[self.updated_states[0]].shape)
        b = self.nab + pad
        for dim in range(ndim):
            region = [Ellipsis for _ in range(ndim)]
            region[dim] = slice(pad, b)
            if dim != 0 or not self.freesurf:
                regions.append(region)
            region = [Ellipsis for _ in range(ndim)]
            region[dim] = slice(-b, -pad)
            regions.append(tuple(region))
        return {el: regions for el in self.updated_states}

    def forward(self, states, **kwargs):
        pad = self.grids[self.updated_states[0]].pad
        for el in self.required_states:
            if not self.freesurf:
                states[el][pad:self.nab+pad, :] *= self.taper[::-1]
            states[el][-self.nab-pad:-pad, :] *= self.taper

            tapert = np.transpose(self.taper)
            states[el][:, pad:self.nab+pad] *= tapert[:, ::-1]
            states[el][:, -self.nab-pad:-pad] *= tapert

        return states

    def adjoint(self, adj_states, states, **kwargs):

        return self.forward(adj_states, **kwargs)


def elastic2d(grid2D, gridout, gridsrc, nab):

    gridpar = copy(grid2D)
    gridpar.zero_boundary = False
    gridpar.pad = 0
    defs = {"gridfd": grid2D, "gridpar": gridpar, "gridout": gridout,
            "gridsrc": gridsrc}

    stepper = SequenceCL([Source(required_states=["vz"], grids=defs),
                          UpdateVelocity(grids=defs),
                          Cerjan(required_states=["vx", "vz"], freesurf=1, nab=nab),
                          Receiver(required_states=["vz"],
                                   updated_states=["vzout"],
                                   grids=defs),
                          UpdateStress(grids=defs),
                          FreeSurface(grids=defs),
                          Cerjan(required_states=["sxx", "szz", "sxz"],
                                 freesurf=1, nab=nab),
                          ])
    psv2D = SequenceCL([Velocity2LameCL(grids=defs),
                        ArithmeticAveraging(grids=defs,
                                            required_states=["rho", "rip"],
                                            updated_states=["rip"], dx=1),
                        ArithmeticAveraging(grids=defs,
                                            required_states=["rho", "rkp"],
                                            updated_states=["rkp"], dz=1),
                        HarmonicAveraging(grids=defs,
                                          required_states=["mu", "muipkp"],
                                          updated_states=["muipkp"],
                                          dx1=1, dz2=1),
                        ScaledParameters(grids=defs,
                                         dt=gridsrc.dt,
                                         dh=grid2D.dh),
                        PropagatorCL(stepper, gridsrc.nt)],
                       grids=defs)
    return psv2D


if __name__ == '__main__':

    resc = ComputeRessource()

    nx = 24
    nz = 24
    nrec = 1
    nt = 3
    nab = 2
    dh = 1.0
    dt = 0.0001

    grid2D = GridCL(resc.queues[0], shape=(nz, nx), pad=4, type=np.float32,
                    zero_boundary=True)
    src_linpos = grid2D.xyz2lin([0], [15]).astype(np.int32)
    xrec = np.arange(10, 15)
    zrec = xrec*0
    rec_linpos = grid2D.xyz2lin(zrec, xrec).astype(np.int32)
    gridsrc = GridCL(resc.queues[0], shape=(1,), pad=0, type=np.float32,
                     dt=dt, nt=nt)
    gridout = GridCL(resc.queues[0], shape=(nt, nrec), type=np.float32)
    psv2D = psv2D = elastic2d(grid2D, gridout, gridsrc, nab)

    # psv2D.backward_test(reclinpos=rec_linpos,
    #                     srclinpos=src_linpos)
    # psv2D.linear_test(reclinpos=rec_linpos,
    #                   srclinpos=src_linpos)
    # psv2D.dot_test(reclinpos=rec_linpos,
    #                srclinpos=src_linpos)

    nrec = 1
    nt = 7500
    nab = 16
    dh = 1.0
    dt = 0.0001

    grid2D = GridCL(resc.queues[0], shape=(160, 300), type=np.float32,
                    zero_boundary=True, dh=dh, pad=2)

    src_linpos = grid2D.xyz2lin([0], [50]).astype(np.int32)
    xrec = np.arange(50, 250)
    zrec = xrec*0
    rec_linpos = grid2D.xyz2lin(zrec, xrec).astype(np.int32)
    gridout = GridCL(resc.queues[0], shape=(nt, xrec.shape[0]), pad=0,
                     type=np.float32, dt=dt, nt=nt, nfddim=1)
    gridsrc = GridCL(resc.queues[0], shape=(nt, 1), pad=0, type=np.float32,
                     dt=dt, nt=nt, nfddim=1)
    psv2D = elastic2d(grid2D, gridout, gridsrc, nab)

    vs = np.full(grid2D.shape, 300.0)
    rho = np.full(grid2D.shape, 1800.0)
    vp = np.full(grid2D.shape, 1500.0)
    vs[80:, :] = 600
    rho[80:, :] = 2000
    vp[80:, :] = 2000
    vs0 = vs.copy()
    vs[5:10, 145:155] *= 1.05

    states = psv2D({"vs": vs,
                    "vp": vp,
                    "rho": rho,
                    "signal": ricker(10, dt, nt)},
                    reclinpos=rec_linpos,
                    srclinpos=src_linpos)
    plt.imshow(states["vx"].get())
    plt.show()
    #
    vzobs = states["vzout"].get()
    clip = 0.01
    vmin = np.min(vzobs) * 0.1
    vmax=-vmin
    plt.imshow(vzobs, aspect="auto", vmin=vmin, vmax=vmax)
    plt.show()

