import pyopencl.array

from SeisCL.python.pycl_backend import (GridCL,
                                        StateKernelGPU,
                                        ReversibleKernelCL,
                                        ComputeRessource,
                                        State,
                                        SequenceCL,
                                        PropagatorCL,
                                        )
from SeisCL.python.seis2D import ReversibleKernel, ricker
from SeisCL.python.FDstencils import FDCoefficients
from SeisCL.python.common.vel2lame import Velocity2LameCL
from SeisCL.python.PSV2D.elastic import ScaledParameters
from SeisCL.python.common.averaging import (ArithmeticAveraging,
                                            HarmonicAveraging,
                                            ArithmeticAveraging2)
from SeisCL.python.common.sources import Source
from SeisCL.python.common.receivers import Receiver
from SeisCL.python.PSV2D.elastic import UpdateVelocity
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from SeisCL.python.PSV2D.elastic import UpdateStress as UpdateStresselas
from SeisCL.python.PSV2D.elastic import elastic2d


class UpdateStress(StateKernelGPU):
    forward_src = """
FUNDEF void UpdateStress(grid pos,
                         GLOBARG float *vx,         GLOBARG float *vz,
                         GLOBARG float *sxx,        GLOBARG float *szz,
                         GLOBARG float *sxz,        GLOBARG float *Me,
                         GLOBARG float *lame,        GLOBARG float *mueipkp,   
                         GLOBARG float *rxx,        GLOBARG float *rzz, 
                         GLOBARG float *rxz,        GLOBARG float *Mv,
                         GLOBARG float *lamv,         GLOBARG float *muvipkp,
                         GLOBARG float *eta,        int backpropagate)
{
    
    get_pos(&pos);
    LOCID float lvar[__LSIZE__];
    
    float vxx, vzz, vzx, vxz;
    int l;
    float sumrxz, sumrxx, sumrzz;
    float b,c;
    float lsxx, lszz, lsxz;
    int indr;
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
    
    /* computing sums of the old memory variables */
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<__L__;l++){
        indr = ind0 + l*pos.NX*pos.NZ;
        sumrxz+=rxz[indr];
        sumrxx+=rxx[indr];
        sumrzz+=rzz[indr];
    }

    /* updating components of the stress tensor, partially */
    lsxz=(mueipkp[ind0]*(vxz+vzx)) + (pos.dt/2.0*sumrxz);
    lsxx= Me[ind0]*vxx + lame[ind0]*vzz + (pos.dt/2.0*sumrxx);
    lszz= Me[ind0]*vzz + lame[ind0]*vxx + (pos.dt/2.0*sumrzz);
    
    /* now updating the memory-variables and sum them up*/
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<__L__;l++){
        b=1.0/(1.0+(eta[l]*0.5));
        c=1.0-(eta[l]*0.5);
        indr = ind0 + l*pos.NX*pos.NZ;
        rxz[indr]=b*(rxz[indr]*c-eta[l]*(muvipkp[ind0]*(vxz+vzx)));
        rxx[indr]=b*(rxx[indr]*c-eta[l]*(Mv[ind0]*vxx + lamv[ind0]*vzz));
        rzz[indr]=b*(rzz[indr]*c-eta[l]*(Mv[ind0]*vzz + lamv[ind0]*vxx));

        sumrxz+=rxz[indr];
        sumrxx+=rxx[indr];
        sumrzz+=rzz[indr];
    }
    
    /* and now the components of the stress tensor are
     completely updated */
    sxz[ind0]+= lsxz + (pos.dt/2.0*sumrxz);
    sxx[ind0]+= lsxx + (pos.dt/2.0*sumrxx);
    szz[ind0]+= lszz + (pos.dt/2.0*sumrzz);
}
"""

    linear_src = """
FUNDEF void UpdateStress_lin(grid pos,
                         GLOBARG float *vx,         GLOBARG float *vz,
                         GLOBARG float *sxx,        GLOBARG float *szz,
                         GLOBARG float *sxz,        GLOBARG float *Me,
                         GLOBARG float *lame,        GLOBARG float *mueipkp,   
                         GLOBARG float *rxx,        GLOBARG float *rzz, 
                         GLOBARG float *rxz,        GLOBARG float *Mv,
                         GLOBARG float *lamv,         GLOBARG float *muvipkp,
                         GLOBARG float *vx_lin,     GLOBARG float *vz_lin,
                         GLOBARG float *sxx_lin,    GLOBARG float *szz_lin,
                         GLOBARG float *sxz_lin,    GLOBARG float *Me_lin,
                         GLOBARG float *lame_lin,    GLOBARG float *mueipkp_lin,   
                         GLOBARG float *rxx_lin,    GLOBARG float *rzz_lin, 
                         GLOBARG float *rxz_lin,    GLOBARG float *Mv_lin,
                         GLOBARG float *lamv_lin,    GLOBARG float *muvipkp_lin,
                         GLOBARG float *eta)
{
    
    get_pos(&pos);
    LOCID float lvar[__LSIZE__];
    
    float vxx, vzz, vzx, vxz;
    float vxx_lin, vzz_lin, vzx_lin, vxz_lin;
    int l;
    float sumrxz, sumrxx, sumrzz;
    float b,c;
    float lsxx, lszz, lsxz;
    int indr;
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
    
    /* computing sums of the old memory variables */
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<__L__;l++){
        indr = ind0 + l*pos.NX*pos.NZ;
        sumrxz+=rxz_lin[indr];
        sumrxx+=rxx_lin[indr];
        sumrzz+=rzz_lin[indr];
    }

    /* updating components of the stress tensor, partially */
    lsxz= (mueipkp[ind0]*(vxz_lin+vzx_lin))\
         +(mueipkp_lin[ind0]*(vxz+vzx))\
         +(pos.dt/2.0*sumrxz);
    lsxx= Me[ind0]*vxx_lin + lame[ind0]*vzz_lin\
          + Me_lin[ind0]*vxx + lame_lin[ind0]*vzz\
          +(pos.dt/2.0*sumrxx);
    lszz=Me[ind0]*vzz_lin + lame[ind0]*vxx_lin\
         +Me_lin[ind0]*vzz + lame_lin[ind0]*vxx\
         +(pos.dt/2.0*sumrzz);
    
    /* now updating the memory-variables and sum them up*/
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<__L__;l++){
        b=1.0/(1.0+(eta[l]*0.5));
        c=1.0-(eta[l]*0.5);
        indr = ind0 + l*pos.NX*pos.NZ;
        rxz_lin[indr]=b*(rxz_lin[indr]*c
                         -eta[l]*(muvipkp[ind0]*(vxz_lin+vzx_lin))
                         -eta[l]*(muvipkp_lin[ind0]*(vxz+vzx)));
        rxx_lin[indr]=b*(rxx_lin[indr]*c
                         -eta[l]*(Mv[ind0]*vxx_lin + lamv[ind0]*vzz_lin)
                         -eta[l]*(Mv_lin[ind0]*vxx + lamv_lin[ind0]*vzz));
        rzz_lin[indr]=b*(rzz_lin[indr]*c
                         -eta[l]*(Mv[ind0]*vzz_lin + lamv[ind0]*vxx_lin)
                         -eta[l]*(Mv_lin[ind0]*vzz + lamv_lin[ind0]*vxx));
        
        sumrxz+=rxz_lin[indr];
        sumrxx+=rxx_lin[indr];
        sumrzz+=rzz_lin[indr];
    }
    
    /* and now the components of the stress tensor are
     completely updated */
    sxz_lin[ind0]+= lsxz + (pos.dt/2.0*sumrxz);
    sxx_lin[ind0]+= lsxx + (pos.dt/2.0*sumrxx);
    szz_lin[ind0]+= lszz + (pos.dt/2.0*sumrzz);
}   
"""

    adjoint_src = """
FUNDEF void UpdateStress_adj1(grid pos,
                         GLOBARG float *vx,         GLOBARG float *vz,
                         GLOBARG float *sxx,        GLOBARG float *szz,
                         GLOBARG float *sxz,        GLOBARG float *Me,
                         GLOBARG float *lame,        GLOBARG float *mueipkp,   
                         GLOBARG float *rxx,        GLOBARG float *rzz, 
                         GLOBARG float *rxz,        GLOBARG float *Mv,
                         GLOBARG float *lamv,        GLOBARG float *muvipkp,
                         GLOBARG float *vx_adj,     GLOBARG float *vz_adj,
                         GLOBARG float *sxx_adj,    GLOBARG float *szz_adj,
                         GLOBARG float *sxz_adj,    GLOBARG float *Me_adj,
                         GLOBARG float *lame_adj,    GLOBARG float *mueipkp_adj,   
                         GLOBARG float *rxx_adj,    GLOBARG float *rzz_adj, 
                         GLOBARG float *rxz_adj,    GLOBARG float *Mv_adj,
                         GLOBARG float *lamv_adj,    GLOBARG float *muvipkp_adj,
                         GLOBARG float *eta)
{
    float b,c;
    int l;
    int indr;
    get_pos(&pos);
    int ind0 = indg(pos,0,0,0);
    
    for (l=0;l<__L__;l++){
        b=1.0/(1.0+(eta[l]*0.5));
        c=1.0-(eta[l]*0.5);
        
        indr = ind0 + l*pos.NX*pos.NZ;
        rxz_adj[indr] += pos.dt/2.0*sxz_adj[ind0];
        rxx_adj[indr] += pos.dt/2.0*sxx_adj[ind0];
        rzz_adj[indr] += pos.dt/2.0*szz_adj[ind0];
    }
}

FUNDEF void UpdateStress_adj2(grid pos,
                         GLOBARG float *vx,         GLOBARG float *vz,
                         GLOBARG float *sxx,        GLOBARG float *szz,
                         GLOBARG float *sxz,        GLOBARG float *Me,
                         GLOBARG float *lame,       GLOBARG float *mueipkp,   
                         GLOBARG float *rxx,        GLOBARG float *rzz, 
                         GLOBARG float *rxz,        GLOBARG float *Mv,
                         GLOBARG float *lamv,       GLOBARG float *muvipkp,
                         GLOBARG float *vx_adj,     GLOBARG float *vz_adj,
                         GLOBARG float *sxx_adj,    GLOBARG float *szz_adj,
                         GLOBARG float *sxz_adj,    GLOBARG float *Me_adj,
                         GLOBARG float *lame_adj,   GLOBARG float *mueipkp_adj,   
                         GLOBARG float *rxx_adj,    GLOBARG float *rzz_adj, 
                         GLOBARG float *rxz_adj,    GLOBARG float *Mv_adj,
                         GLOBARG float *lamv_adj,   GLOBARG float *muvipkp_adj,
                         GLOBARG float *eta)
{
    
    get_pos(&pos);
    LOCID float lvar[__LSIZE__];
    
    float sxx_z_adj, sxx_x_adj, szz_z_adj, szz_x_adj, sxz_x_adj, sxz_z_adj;
    float rxx_z_adj[__L__], rxx_x_adj[__L__], rzz_z_adj[__L__]; 
    float rzz_x_adj[__L__], rxz_x_adj[__L__], rxz_z_adj[__L__];
    GLOBARG float *lrxx_adj, *lrzz_adj, *lrxz_adj;
    float lvx, lvz;
    float vxx, vxz, vzz, vzx;
    int l;
    float sumrxz, sumrxx, sumrzz;
    float b,c;

    int indr;
    int ind0 = indg(pos,0,0,0);
    
// Calculation of the spatial derivatives 

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
    load_local_in(pos, mueipkp, lvar);
    mul_local_in(pos, sxz_adj, lvar);
    load_local_haloz(pos, mueipkp, lvar);
    mul_local_haloz(pos, sxz_adj, lvar);
    load_local_halox(pos, mueipkp, lvar);
    mul_local_halox(pos, sxz_adj, lvar);
    BARRIER
#endif
    sxz_x_adj = -Dxm(pos, lvar);
    sxz_z_adj = -Dzm(pos, lvar);

#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, Me, lvar);
    mul_local_in(pos, sxx_adj, lvar);
    load_local_halox(pos, Me, lvar);
    mul_local_halox(pos, sxx_adj, lvar);
    BARRIER
#endif
    sxx_x_adj = -Dxp(pos, lvar);

#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, lame, lvar);
    mul_local_in(pos, sxx_adj, lvar);
    load_local_haloz(pos, lame, lvar);
    mul_local_haloz(pos, sxx_adj, lvar);
    BARRIER
#endif
    sxx_z_adj = -Dzp(pos, lvar);

#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, Me, lvar);
    mul_local_in(pos, szz_adj, lvar);
    load_local_haloz(pos, Me, lvar);
    mul_local_haloz(pos, szz_adj, lvar);
    BARRIER
#endif

    szz_z_adj = -Dzp(pos, lvar);

#if LOCAL_OFF==0
    BARRIER
    load_local_in(pos, lame, lvar);
    mul_local_in(pos, szz_adj, lvar);
    load_local_halox(pos, lame, lvar);
    mul_local_halox(pos, szz_adj, lvar);
    BARRIER
#endif
    szz_x_adj = -Dxp(pos, lvar);
    
    for (l=0;l<__L__;l++){
    
        lrxx_adj= &rxx_adj[l*pos.NX*pos.NZ];
        lrzz_adj= &rzz_adj[l*pos.NX*pos.NZ];
        lrxz_adj= &rxz_adj[l*pos.NX*pos.NZ];
        
        #if LOCAL_OFF==0
        BARRIER
        load_local_in(pos, muvipkp, lvar);
        mul_local_in(pos, lrxz_adj, lvar);
        load_local_haloz(pos, muvipkp, lvar);
        mul_local_haloz(pos, lrxz_adj, lvar);
        load_local_halox(pos, muvipkp, lvar);
        mul_local_halox(pos, lrxz_adj, lvar);
        BARRIER
        #endif
        rxz_x_adj[l] = -Dxm(pos, lvar);
        rxz_z_adj[l] = -Dzm(pos, lvar);
    
        #if LOCAL_OFF==0
        BARRIER
        load_local_in(pos, Mv, lvar);
        mul_local_in(pos, lrxx_adj, lvar);
        load_local_halox(pos, Mv, lvar);
        mul_local_halox(pos, lrxx_adj, lvar);
        BARRIER
        #endif
        rxx_x_adj[l] = -Dxp(pos, lvar);
    
        #if LOCAL_OFF==0
        BARRIER
        load_local_in(pos, lamv, lvar);
        mul_local_in(pos, lrxx_adj, lvar);
        load_local_haloz(pos, lamv, lvar);
        mul_local_haloz(pos, lrxx_adj, lvar);
        BARRIER
        #endif
        rxx_z_adj[l] = -Dzp(pos, lvar);
    
        #if LOCAL_OFF==0
        BARRIER
        load_local_in(pos, Mv, lvar);
        mul_local_in(pos, lrzz_adj, lvar);
        load_local_haloz(pos, Mv, lvar);
        mul_local_haloz(pos, lrzz_adj, lvar);
        BARRIER
        #endif
    
        rzz_z_adj[l] = -Dzp(pos, lvar);
    
        #if LOCAL_OFF==0
        BARRIER
        load_local_in(pos, lamv, lvar);
        mul_local_in(pos, lrzz_adj, lvar);
        load_local_halox(pos, lamv, lvar);
        mul_local_halox(pos, lrzz_adj, lvar);
        BARRIER
        #endif
        rzz_x_adj[l] = -Dxp(pos, lvar);
    }
    gridstop(pos);
    
    lvx=lvz=0;
    for (l=0;l<__L__;l++){
        b=1.0/(1.0+(eta[l]*0.5));
        c=1.0-(eta[l]*0.5);
        indr = ind0 + l*pos.NX*pos.NZ;
        
        lvx += -b*eta[l] * (rxz_z_adj[l] + rxx_x_adj[l] + rzz_x_adj[l]);
        lvz += -b*eta[l] * (rxz_x_adj[l] + rxx_z_adj[l] + rzz_z_adj[l]);
        muvipkp_adj[ind0] += -b*eta[l] * (vxz+vzx) * rxz_adj[indr];
        Mv_adj[ind0] += -b*eta[l] * (vxx * rxx_adj[indr] + vzz * rzz_adj[indr]);
        lamv_adj[ind0] += -b*eta[l] * (vzz * rxx_adj[indr] + vxx * rzz_adj[indr]);
    }
    
    lvx += sxz_z_adj + sxx_x_adj + szz_x_adj;
    lvz += sxz_x_adj + sxx_z_adj + szz_z_adj;
    vx_adj[ind0] += lvx;
    vz_adj[ind0] += lvz;
    mueipkp_adj[ind0] += (vxz+vzx) * sxz_adj[ind0];
    Me_adj[ind0] += vxx * sxx_adj[ind0] + vzz * szz_adj[ind0];
    lame_adj[ind0] += vzz * sxx_adj[ind0] + vxx * szz_adj[ind0];
}   

FUNDEF void UpdateStress_adj3(grid pos,
                         GLOBARG float *vx,         GLOBARG float *vz,
                         GLOBARG float *sxx,        GLOBARG float *szz,
                         GLOBARG float *sxz,        GLOBARG float *Me,
                         GLOBARG float *lame,        GLOBARG float *mueipkp,   
                         GLOBARG float *rxx,        GLOBARG float *rzz, 
                         GLOBARG float *rxz,        GLOBARG float *Mv,
                         GLOBARG float *lamv,        GLOBARG float *muvipkp,
                         GLOBARG float *vx_adj,     GLOBARG float *vz_adj,
                         GLOBARG float *sxx_adj,    GLOBARG float *szz_adj,
                         GLOBARG float *sxz_adj,    GLOBARG float *Me_adj,
                         GLOBARG float *lame_adj,    GLOBARG float *mueipkp_adj,   
                         GLOBARG float *rxx_adj,    GLOBARG float *rzz_adj, 
                         GLOBARG float *rxz_adj,    GLOBARG float *Mv_adj,
                         GLOBARG float *lamv_adj,    GLOBARG float *muvipkp_adj,
                         GLOBARG float *eta)
{
    float b,c;
    int l;
    int indr;
    get_pos(&pos);
    int ind0 = indg(pos,0,0,0);
    
    for (l=0;l<__L__;l++){
        b=1.0/(1.0+(eta[l]*0.5));
        c=1.0-(eta[l]*0.5);
        
        indr = ind0 + l*pos.NX*pos.NZ;
        rxz_adj[indr]=b*rxz_adj[indr]*c + pos.dt/2.0*sxz_adj[ind0];
        rxx_adj[indr]=b*rxx_adj[indr]*c + pos.dt/2.0*sxx_adj[ind0];
        rzz_adj[indr]=b*rzz_adj[indr]*c + pos.dt/2.0*szz_adj[ind0];
    }
}
"""

    def __init__(self, grids=None, computegrid=None, fdcoefs=FDCoefficients(),
                 local_size=(16, 16), L=1, FL=(10,), dt=1, **kwargs):
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz",
                                "Me", "lame", "mueipkp", "rxx", "rzz", "rxz",
                                "Mv", "lamv", "muvipkp"]
        self.updated_states = ["sxx", "szz", "sxz", "rxx", "rzz", "rxz"]
        self.default_grids = {"vx": "gridfd",
                              "vz": "gridfd",
                              "sxx": "gridfd",
                              "szz": "gridfd",
                              "sxz": "gridfd",
                              "Me": "gridpar",
                              "lame": "gridpar",
                              "mueipkp": "gridpar",
                              "rxx": "gridmemvar",
                              "rzz": "gridmemvar",
                              "rxz": "gridmemvar",
                              "Mv": "gridpar",
                              "lamv": "gridpar",
                              "muvipkp": "gridpar",}
        self.headers = fdcoefs.header()
        self.L = L
        #TODO correct options
        options = fdcoefs.options
        options += ["-D __L__=%d" % L]
        super().__init__(grids=grids,
                         computegrid=computegrid,
                         local_size=local_size,
                         default_args={"eta": 2 * np.pi * np.array(FL) * dt},
                         options=options, **kwargs)


class FreeSurface(StateKernelGPU):

    forward_src = """
FUNDEF void FreeSurface(grid pos,
                         GLOBARG float *vx,         GLOBARG float *vz,
                         GLOBARG float *sxx,        GLOBARG float *szz,
                         GLOBARG float *sxz,        GLOBARG float *rxx,
                         GLOBARG float *rzz,        GLOBARG float *rxz,
                         GLOBARG float *Me,         GLOBARG float *lame,
                         GLOBARG float *Mv,         GLOBARG float *lamv,
                         GLOBARG float *eta)
{
    get_pos(&pos);
    float vxx, vzz;
    float h, sump, b;
    int l, indr;
    int ind0 = indg(pos,0,0,0);
    
    szz[ind0] = 0; 
    for (l=0; l<__L__; l++){
        indr = ind0 + l*pos.NX*pos.NZ;
        rzz[indr] = 0;
    }
        
    vxx = Dxm(pos, vx);
    vzz = Dzm(pos, vz);
    h =  (-(lame[ind0] * lame[ind0] * vxx / Me[ind0])  -(lame[ind0] * vzz));
    sump=0;
    for (l=0;l<__L__;l++){
        indr = ind0 + l*pos.NX*pos.NZ;
        sump+=rxx[indr];
    }
    h+=-sump * pos.dt / 2.0;
    sump=0;
    for (l=0;l<__L__;l++){
        indr = ind0 + l*pos.NX*pos.NZ;
        b=1.0/(1.0+(eta[l]*0.5));
        rxx[indr] += b*(lamv[indr]*(((Me[ind0]-lame[ind0])/Me[ind0])-1.0)*vxx
                       -lamv[indr]*vzz);
        sump+=rxx[indr];
    }
    h+=sump * pos.dt / 2.0;
    sxx[ind0] += h;
}
    """

    linear_src = """
FUNDEF void FreeSurface_lin(grid pos,
                             GLOBARG float *vx,         GLOBARG float *vz,
                             GLOBARG float *sxx,        GLOBARG float *szz,
                             GLOBARG float *sxz,        GLOBARG float *rxx,
                             GLOBARG float *rzz,        GLOBARG float *rxz,
                             GLOBARG float *Me,         GLOBARG float *lame,
                             GLOBARG float *Mv,         GLOBARG float *lamv,
                             GLOBARG float *vx_lin,     GLOBARG float *vz_lin,
                             GLOBARG float *sxx_lin,    GLOBARG float *szz_lin,
                             GLOBARG float *sxz_lin,    GLOBARG float *rxx_lin,
                             GLOBARG float *rzz_lin,    GLOBARG float *rxz_lin,
                             GLOBARG float *Me_lin,     GLOBARG float *lame_lin,
                             GLOBARG float *Mv_lin,     GLOBARG float *lamv_lin,
                             GLOBARG float *eta)
{
    get_pos(&pos);
    float vxx, vzz, vxx_lin, vzz_lin;
    float b, h_lin, sump_lin;
    int l, indr;
    int ind0 = indg(pos,0,0,0);
    
    
    szz_lin[ind0] = 0;
    for (l=0;l<__L__;l++){
        indr = ind0 + l*pos.NX*pos.NZ;
        rzz_lin[indr] = 0;
    }
    
    vxx = Dxm(pos, vx);
    vzz = Dzm(pos, vz);
    vxx_lin = Dxm(pos, vx_lin);
    vzz_lin = Dzm(pos, vz_lin);
    h_lin =  ((-2*lame[ind0] * vxx / Me[ind0]  - vzz) * lame_lin[ind0] +
              lame[ind0] * lame[ind0] * vxx / Me[ind0]/Me[ind0]*Me_lin[ind0]+
              -(lame[ind0] * lame[ind0] / Me[ind0]) * vxx_lin +
              -lame[ind0] * vzz_lin);
    sump_lin=0;
    for (l=0;l<__L__;l++){
        indr = ind0 + l*pos.NX*pos.NZ;
        sump_lin+=rxx_lin[indr];
    }
    h_lin+=-sump_lin * pos.dt / 2.0;
    sump_lin=0;
    for (l=0;l<__L__;l++){
        indr = ind0 + l*pos.NX*pos.NZ;
        b=1.0/(1.0+(eta[l]*0.5));
        rxx_lin[indr] += b*( ((((Me[ind0]-lame[ind0])/Me[ind0])-1.0)*vxx - vzz)*lamv_lin[indr] +
                          -lamv[indr]/Me[ind0]*vxx * lame_lin[ind0] +
                          lamv[indr]*lame[ind0]/Me[ind0]/Me[ind0]*vxx * Me_lin[ind0] +
                          lamv[indr]*(((Me[ind0]-lame[ind0])/Me[ind0])-1.0)*vxx_lin +
                          -lamv[ind0]*vzz_lin);  
        sump_lin+=rxx_lin[indr];
    }
    h_lin+=sump_lin * pos.dt / 2.0;
    sxx_lin[ind0] += h_lin;
}
"""

    adjoint_src = """
FUNDEF void FreeSurface_adj(grid pos,
                            GLOBARG float *vx,         GLOBARG float *vz,
                            GLOBARG float *sxx,        GLOBARG float *szz,
                            GLOBARG float *sxz,        GLOBARG float *rxx,
                            GLOBARG float *rzz,        GLOBARG float *rxz,
                            GLOBARG float *Me,         GLOBARG float *lame,
                            GLOBARG float *Mv,         GLOBARG float *lamv,
                            GLOBARG float *vx_adj,     GLOBARG float *vz_adj,
                            GLOBARG float *sxx_adj,    GLOBARG float *szz_adj,
                            GLOBARG float *sxz_adj,    GLOBARG float *rxx_adj,
                            GLOBARG float *rzz_adj,    GLOBARG float *rxz_adj,
                            GLOBARG float *Me_adj,     GLOBARG float *lame_adj,
                            GLOBARG float *Mv_adj,     GLOBARG float *lamv_adj,
                            GLOBARG float *eta)
{
    get_pos(&pos);
    float vxx, vzz, b;
    int ind0 = indg(pos,0,0,0);
    int i, l, indr;
    float vxx_adj, vzz_adj, lame_loc, Me_loc, temp;
    
    vxx = Dxm(pos, vx);
    vzz = Dzm(pos, vz);
    szz_adj[ind0] = 0;
        
    lame_loc = (-2*lame[ind0] * vxx / Me[ind0]  - vzz) * sxx_adj[ind0];
    Me_loc = lame[ind0] * lame[ind0] * vxx / Me[ind0]/Me[ind0] * sxx_adj[ind0];
    vxx_adj = -(lame[ind0] * lame[ind0] / Me[ind0]) * sxx_adj[ind0];
    vzz_adj = -lame[ind0] * sxx_adj[ind0];
    
    for (l=0;l<__L__;l++){
        indr = ind0 + l*pos.NX*pos.NZ;
        b=1.0/(1.0+(eta[l]*0.5));
        temp = rxx_adj[indr] + pos.dt / 2.0 * sxx_adj[ind0];
        lamv_adj[indr] += b*((((Me[ind0]-lame[ind0])/Me[ind0])-1.0)*vxx-vzz)*temp;
        lame_loc += -b*lamv[indr]/Me[ind0]*vxx* temp;
        Me_loc += b*lamv[indr]*lame[ind0]/Me[ind0]/Me[ind0]*vxx*temp;
        vxx_adj += b*lamv[indr]*(((Me[ind0]-lame[ind0])/Me[ind0])-1.0)*temp;
        vzz_adj += -b*lamv[indr]*temp;
        rzz_adj[indr] = 0;
    }
    for (i=0; i<__FDOH__; i++){
        vx_adj[indg(pos, 0, 0, i)] += hc[i] * vxx_adj;
        vx_adj[indg(pos, 0, 0, -i-1)] += - hc[i] * vxx_adj;
        vz_adj[indg(pos, i, 0, 0)] += hc[i] * vzz_adj;
        vz_adj[indg(pos, -i-1, 0, 0)] += - hc[i] * vzz_adj;
    }
    Me_adj[ind0] += Me_loc;
    lame_adj[ind0] += lame_loc;
}
"""

    def __init__(self, grids=None, fdcoefs=None, L=1, FL=(10,), dt=1, **kwargs):
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "Me", "lame",
                                "rxx", "rzz", "rxz", "Mv", "lamv"]
        self.updated_states = ["sxx", "szz", "rxx", "rzz"]
        self.default_grids = {"vx": "gridfd",
                              "vz": "gridfd",
                              "sxx": "gridfd",
                              "szz": "gridfd",
                              "Me": "gridpar",
                              "lame": "gridpar",
                              "rxx": "gridmemvar",
                              "rzz": "gridmemvar",
                              "Mv": "gridpar",
                              "lamv": "gridpar"}
        if fdcoefs is None:
            fdcoefs = FDCoefficients(order=grids["gridfd"].pad*2,
                                     local_off=1)
        if fdcoefs.order//2 != grids["gridfd"].pad:
            raise ValueError("The grid padding should be equal to half the "
                             "fd stencil width")
        self.headers = fdcoefs.header()
        computegrid = copy(grids["gridfd"])
        options = fdcoefs.options
        self.L = L
        #TODO correct options
        options = fdcoefs.options
        options += ["-D __L__=%d" % L]
        super().__init__(grids=grids, computegrid=computegrid, options=options,
                         default_args={"eta": 2 * np.pi * np.array(FL) * dt},
                         **kwargs)

    def global_size_fw(self):
        gsize = [np.int32(s - 2 * self.computegrid.pad)
                 for s in self.computegrid.shape]
        gsize[0] = 1
        return gsize


class FreeSurface2(ReversibleKernelCL):

    forward_src = """
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
FUNDEF void FreeSurface2_lin(grid pos,
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
    FUNDEF void FreeSurface2_adj(grid pos,
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
"""

    def __init__(self, grids=None, fdcoefs=None, **kwargs):
        self.required_states = ["vx", "vz", "sxx", "szz", "sxz", "mu", "M",
                                "rip", "rkp"]
        self.updated_states = ["vx", "vz"]
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


class ViscoChangePar(ReversibleKernelCL):

    forward_src = """
    FUNDEF void ViscoChangePar(grid pos,
                                GLOBARG float *M,
                                GLOBARG float *mu,         
                                GLOBARG float *muipkp,
                                GLOBARG float *taus,  
                                GLOBARG float *tausipkp,   
                                GLOBARG float *taup,
                                int backpropagate){
    
    get_pos(&pos);
    int ind0 = indg(pos, 0, 0, 0);
    float lM, lmu, lmuipkp, ltaup, ltaus, ltausipkp;
    
    lM = M[ind0];
    lmu = mu[ind0];
    lmuipkp = muipkp[ind0];
    ltaup = taup[ind0];
    ltaus = taus[ind0];
    ltausipkp = tausipkp[ind0];
    
    if (backpropagate){
        lmu = lM - lmu;
        ltaus = ltaup -ltaus;        
        tausipkp[ind0]= 1.0 / (lmuipkp/ltausipkp/pos.dt - (float)__L__);
        muipkp[ind0]= lmuipkp / (1.0+ (float)__L__* tausipkp[ind0]);
        taup[ind0]= 1.0 / (lM/ltaup/pos.dt - (float)__L__);
        M[ind0]= lM / (1.0+ (float)__L__* taup[ind0]);
        taus[ind0]= 1.0 / (lmu/ltaus/pos.dt - (float)__L__);
        mu[ind0]= lmu / (1.0+ (float)__L__* taus[ind0]) / 2.0;
    }
    else{
        muipkp[ind0]=lmuipkp*(1.0+ (float)__L__*ltausipkp);
        M[ind0]=lM*(1.0+(float)__L__*ltaup);
        mu[ind0]=M[ind0]-2.0*lmu*(1.0+(float)__L__*ltaus);
        tausipkp[ind0]=lmuipkp*ltausipkp/pos.dt;
        taup[ind0]=lM*ltaup/pos.dt;
        taus[ind0]= taup[ind0]-2.0*lmu*ltaus/pos.dt;
    }
}
    """

    linear_src = """
    FUNDEF void ViscoChangePar_lin(grid pos,
                                GLOBARG float *M,
                                GLOBARG float *mu,         
                                GLOBARG float *muipkp,
                                GLOBARG float *taus,  
                                GLOBARG float *tausipkp,   
                                GLOBARG float *taup,
                                GLOBARG float *M_lin,
                                GLOBARG float *mu_lin,         
                                GLOBARG float *muipkp_lin,
                                GLOBARG float *taus_lin,  
                                GLOBARG float *tausipkp_lin,   
                                GLOBARG float *taup_lin){
    
    get_pos(&pos);
    int ind0 = indg(pos, 0, 0, 0);
    float lM_lin, lmu_lin, lmuipkp_lin, ltaup_lin, ltaus_lin, ltausipkp_lin;
    
    lM_lin = M_lin[ind0];
    lmu_lin = mu_lin[ind0];
    lmuipkp_lin = muipkp_lin[ind0];
    ltaup_lin = taup_lin[ind0];
    ltaus_lin = taus_lin[ind0];
    ltausipkp_lin = tausipkp_lin[ind0];
    
    muipkp_lin[ind0]=lmuipkp_lin*(1.0+ (float)__L__*tausipkp[ind0])\
                    +muipkp[ind0]*(float)__L__*ltausipkp_lin;
    M_lin[ind0]=lM_lin*(1.0+(float)__L__*taup[ind0])\
                +M[ind0]*(float)__L__*ltaup_lin;
    mu_lin[ind0]=2.0*lmu_lin*(1.0+(float)__L__*taus[ind0])\
             +2.0*mu[ind0]*(float)__L__*ltaus_lin;
    mu_lin[ind0] = M_lin[ind0] - mu_lin[ind0];
    tausipkp_lin[ind0]=muipkp[ind0]*ltausipkp_lin/pos.dt\
                   + lmuipkp_lin*tausipkp[ind0]/pos.dt ;
    taup_lin[ind0]=M[ind0]*ltaup_lin/pos.dt + lM_lin*taup[ind0]/pos.dt;
    taus_lin[ind0]=2.0*mu[ind0]*ltaus_lin/pos.dt + 2.0*lmu_lin*taus[ind0]/pos.dt;
    taus_lin[ind0] = taup_lin[ind0] - taus_lin[ind0];
}
    """

    adjoint_src = """
    FUNDEF void ViscoChangePar_adj(grid pos,
                                GLOBARG float *M,
                                GLOBARG float *mu,         
                                GLOBARG float *muipkp,
                                GLOBARG float *taus,  
                                GLOBARG float *tausipkp,   
                                GLOBARG float *taup,
                                GLOBARG float *M_adj,
                                GLOBARG float *mu_adj,         
                                GLOBARG float *muipkp_adj,
                                GLOBARG float *taus_adj,  
                                GLOBARG float *tausipkp_adj,   
                                GLOBARG float *taup_adj){
    
    get_pos(&pos);
    int ind0 = indg(pos, 0, 0, 0);
    float lM_adj, lmu_adj, lmuipkp_adj, ltaup_adj, ltaus_adj, ltausipkp_adj;
    
    lM_adj = M_adj[ind0];
    lmu_adj = mu_adj[ind0];
    lmuipkp_adj = muipkp_adj[ind0];
    ltaup_adj = taup_adj[ind0];
    ltaus_adj = taus_adj[ind0];
    ltausipkp_adj = tausipkp_adj[ind0];

        
    ltaup_adj += ltaus_adj;
    ltaus_adj = -ltaus_adj;
    lM_adj += lmu_adj;
    lmu_adj = -lmu_adj;

    muipkp_adj[ind0]= lmuipkp_adj * (1.0+ (float)__L__*tausipkp[ind0]) \
                      + ltausipkp_adj * tausipkp[ind0]/pos.dt;
    M_adj[ind0]=lM_adj * (1.0+(float)__L__*taup[ind0]) \
                      + ltaup_adj * taup[ind0]/pos.dt;
    mu_adj[ind0]=2.0 * lmu_adj *(1.0+(float)__L__*taus[ind0]) \
               + 2.0 * ltaus_adj * taus[ind0]/pos.dt;
    tausipkp_adj[ind0]=lmuipkp_adj *(float)__L__*muipkp[ind0] \
                      + ltausipkp_adj * muipkp[ind0]/pos.dt;
    taus_adj[ind0]=2.0 * lmu_adj *(float)__L__*mu[ind0]\
                   + 2.0 * ltaus_adj * mu[ind0]/pos.dt;
    taup_adj[ind0]= lM_adj *(float)__L__*M[ind0] \
                      + ltaup_adj * M[ind0]/pos.dt;
}
    """

    def __init__(self, grids=None, computegrid=None, L=1, **kwargs):
        self.required_states = ["M", "mu", "muipkp", "taup", "taus", "tausipkp"]
        self.updated_states = ["M", "mu", "muipkp", "taup", "taus", "tausipkp"]
        self.default_grids = {el: "gridpar" for el in self.required_states}
        self.copy_states = {"Me": "M", "lame": "mu", "mueipkp": "muipkp",
                            "Mv": "taup", "lamv": "taus", "muvipkp": "tausipkp"}
        options = ["-D __L__=%d" % L]
        super().__init__(grids=grids,
                         computegrid=computegrid,
                         options=options,
                         **kwargs)


def viscoelastic(grid2D, gridout, gridsrc, nab, nl=1):

    gridpar = copy(grid2D)
    gridpar.zero_boundary = False
    gridpar.pad = 0
    gridmemvar = copy(grid2D)
    gridmemvar.shape = tuple(list(gridmemvar.shape) + [nl])
    gridmemvar.nfddim = 2
    grideta = copy(grid2D)
    grideta.shape = (nl,)
    grideta.nfddim = 1

    defs = {"gridfd": grid2D, "gridmemvar": gridmemvar, "gridpar": gridpar,
            "gridout": gridout, "gridsrc": gridsrc, "grideta": grideta}

    stepper = SequenceCL([Source(required_states=["vz"], grids=defs),
                          UpdateVelocity(grids=defs),
                          Receiver(required_states=["vz"],
                                   updated_states=["vzout"],
                                   grids=defs),
                          UpdateStress(grids=defs),
                          FreeSurface(grids=defs),
                          FreeSurface2(grids=defs),
                          ])
    psv2D = SequenceCL([#Eta(grids=defs, dt=gridsrc.dt),
                        Velocity2LameCL(grids=defs),
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
                        ArithmeticAveraging2(grids=defs,
                                          required_states=["taus", "tausipkp"],
                                          updated_states=["tausipkp"],
                                          dx1=1, dz2=1),
                        ScaledParameters(grids=defs,
                                         dt=gridsrc.dt,
                                         dh=grid2D.dh),
                        ViscoChangePar(grids=defs),
                        PropagatorCL(stepper, gridsrc.nt)
                        ],
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
    psv2D = viscoelastic(grid2D, gridout, gridsrc, nab)

    psv2D.backward_test(reclinpos=rec_linpos,
                        srclinpos=src_linpos)
    psv2D.linear_test(reclinpos=rec_linpos,
                      srclinpos=src_linpos)
    psv2D.dot_test(reclinpos=rec_linpos,
                   srclinpos=src_linpos)

    nrec = 1
    nt = 7500
    nab = 16
    dh = 1.0
    dt = 0.0001

    grid2D = GridCL(resc.queues[0], shape=(160, 300), type=np.float32,
                    zero_boundary=True, dh=dh, pad=2, dt=dt)

    src_linpos = grid2D.xyz2lin([0], [50]).astype(np.int32)
    xrec = np.arange(50, 250)
    zrec = xrec*0
    rec_linpos = grid2D.xyz2lin(zrec, xrec).astype(np.int32)
    gridout = GridCL(resc.queues[0], shape=(nt, xrec.shape[0]), pad=0,
                     type=np.float32, dt=dt, nt=nt, nfddim=1)
    gridsrc = GridCL(resc.queues[0], shape=(nt, 1), pad=0, type=np.float32,
                     dt=dt, nt=nt, nfddim=1)
    psv2D = viscoelastic(grid2D, gridout, gridsrc, nab)

    vs = np.full(grid2D.shape, 300.0)
    rho = np.full(grid2D.shape, 1800.0)
    vp = np.full(grid2D.shape, 1500.0)
    taup = np.full(grid2D.shape, 0.02)
    taus = np.full(grid2D.shape, 0.02)
    vs[80:, :] = 600
    rho[80:, :] = 2000
    vp[80:, :] = 2000
    vs0 = vs.copy()
    vs[5:10, 145:155] *= 1.05

    states = psv2D({"vs": vs,
                    "vp": vp,
                    "rho": rho,
                    "taup": taup,
                    "taus": taus,
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