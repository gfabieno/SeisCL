/*------------------------------------------------------------------------
 * Copyright (C) 2016 For the list of authors, see file AUTHORS.
 *
 * This file is part of SeisCL.
 *
 * SeisCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.0 of the License only.
 *
 * SeisCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SeisCL. See file COPYING and/or
 * <http://www.gnu.org/licenses/gpl-3.0.html>.
 --------------------------------------------------------------------------*/

/*Update of the stresses in 2D SV*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#define lbnd (FDOH+NAB)

#define rho(z,x)    rho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,x)    rip[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rjp(z,x)    rjp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,x)    rkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define muipkp(z,x) muipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define mu(z,x)        mu[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define M(z,x)      M[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradrho(z,x)  gradrho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradM(z,x)  gradM[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradmu(z,x)  gradmu[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaup(z,x)  gradtaup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaus(z,x)  gradtaus[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define taus(z,x)        taus[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausipkp(z,x) tausipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define taup(z,x)        taup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define vx(z,x)  vx[(x)*(NZ)+(z)]
#define vz(z,x)  vz[(x)*(NZ)+(z)]
#define sxx(z,x) sxx[(x)*(NZ)+(z)]
#define szz(z,x) szz[(x)*(NZ)+(z)]
#define sxz(z,x) sxz[(x)*(NZ)+(z)]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_vx_x(z,x) psi_vx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_vz_x(z,x) psi_vz_x[(x)*(NZ-2*FDOH)+(z)]

#define psi_vx_z(z,x) psi_vx_z[(x)*(2*NAB)+(z)]
#define psi_vz_z(z,x) psi_vz_z[(x)*(2*NAB)+(z)]


#if LOCAL_OFF==0

#define lvar(z,x)  lvar[(x)*lsizez+(z)]

#endif


#define vxout(y,x) vxout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#define PI (3.141592653589793238462643383279502884197169)
#define signals(y,x) signals[(y)*NT+(x)]

#define __h2f0(x) __half2float(*((half*)&(x)))
#define __h2f1(x) __half2float(*((half*)&(x)+1))

extern "C" __global__ void update_s(int offcomm,
                                    half2 *vx,         half2 *vz,
                                    half2 *sxx,        half2 *szz,        half2 *sxz,
                                    float2 *M,         float2 *mu,          float2 *muipkp,
                                    half2 *rxx,        half2 *rzz,        half2 *rxz,
                                    float2 *taus,       float2 *tausipkp,   float2 *taup,
                                    float *eta,        float *taper,
                                    float2 *K_x,        float2 *a_x,          float2 *b_x,
                                    float2 *K_x_half,   float2 *a_x_half,     float2 *b_x_half,
                                    float2 *K_z,        float2 *a_z,          float2 *b_z,
                                    float2 *K_z_half,   float2 *a_z_half,     float2 *b_z_half,
                                    half2 *psi_vx_x,    half2 *psi_vx_z,
                                    half2 *psi_vz_x,    half2 *psi_vz_z)
{
    
    extern __shared__ half2 lvar[];
    
    float2 vxx, vzz, vzx, vxz;
    int i,k,l,ind;
    float2 sumrxz, sumrxx, sumrzz;
    float2 b,c,e,g,d,f,fipkp,dipkp;
    float leta[LVE];
    float2 lM, lmu, lmuipkp, ltaup, ltaus, ltausipkp;
    float2 lsxx, lszz, lsxz;
    
    // If we use local memory
#if LOCAL_OFF==0
    int lsizez = blockDim.x+2*FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
    
#define lvx lvar
#define lvz lvar
    
    // If local memory is turned off
#elif LOCAL_OFF==1
    
    int lsizez = blockDim.x+2*FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
    
    
#define lvx vx
#define lvz vz
#define lidx gidx
#define lidz gidz
    
#endif
    
    // Calculation of the velocity spatial derivatives
    {
#if LOCAL_OFF==0
        lvx(lidz,lidx)=vx(gidz, gidx);
        if (lidx<2*FDOH)
            lvx(lidz,lidx-FDOH)=vx(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvx(lidz,lidx+lsizex-3*FDOH)=vx(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvx(lidz,lidx+FDOH)=vx(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvx(lidz,lidx-lsizex+3*FDOH)=vx(gidz,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvx(lidz-FDOH,lidx)=vx(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvx(lidz+FDOH,lidx)=vx(gidz+FDOH,gidx);
        
        __syncthreads();
#endif
        
#if   FDOH==1
        vxx.x = HC1*(__h2f0(lvx(lidz, lidx))  -__h2f0(lvx(lidz, lidx-1)))/DH;
        vxx.y = HC1*(__h2f1(lvx(lidz, lidx))  -__h2f1(lvx(lidz, lidx-1)))/DH;
        vxz.x = HC1*(__h2f0(lvx(lidz+1, lidx))-__h2f0(lvx(lidz, lidx)))/DH;
        vxz.y = HC1*(__h2f1(lvx(lidz+1, lidx))-__h2f1(lvx(lidz, lidx)))/DH;
#elif FDOH==2
        vxx.x = (  HC1*(__h2f0(lvx(lidz, lidx))  -__h2f0(lvx(lidz, lidx-1)))
               + HC2*(__h2f0(lvx(lidz, lidx+1))-__h2f0(lvx(lidz, lidx-2)))
               )/DH;
        vxx.y = (  HC1*(__h2f1(lvx(lidz, lidx))  -__h2f1(lvx(lidz, lidx-1)))
               + HC2*(__h2f1(lvx(lidz, lidx+1))-__h2f1(lvx(lidz, lidx-2)))
               )/DH;
        vxz.x = (  HC1*(__h2f0(lvx(lidz+1, lidx))-__h2f0(lvx(lidz, lidx)))
               + HC2*(__h2f0(lvx(lidz+2, lidx))-__h2f0(lvx(lidz-1, lidx)))
               )/DH;
        vxz.y = (  HC1*(__h2f1(lvx(lidz+1, lidx))-__h2f1(lvx(lidz, lidx)))
               + HC2*(__h2f1(lvx(lidz+2, lidx))-__h2f1(lvx(lidz-1, lidx)))
               )/DH;
#elif FDOH==3
        vxx.x = (  HC1*(__h2f0(lvx(lidz, lidx))  -__h2f0(lvx(lidz, lidx-1)))
               + HC2*(__h2f0(lvx(lidz, lidx+1))-__h2f0(lvx(lidz, lidx-2)))
               + HC3*(__h2f0(lvx(lidz, lidx+2))-__h2f0(lvx(lidz, lidx-3)))
               )/DH;
        vxx.y = (  HC1*(__h2f1(lvx(lidz, lidx))  -__h2f1(lvx(lidz, lidx-1)))
               + HC2*(__h2f1(lvx(lidz, lidx+1))-__h2f1(lvx(lidz, lidx-2)))
               + HC3*(__h2f1(lvx(lidz, lidx+2))-__h2f1(lvx(lidz, lidx-3)))
               )/DH;
        vxz.x = (  HC1*(__h2f0(lvx(lidz+1, lidx))-__h2f0(lvx(lidz, lidx)))
               + HC2*(__h2f0(lvx(lidz+2, lidx))-__h2f0(lvx(lidz-1, lidx)))
               + HC3*(__h2f0(lvx(lidz+3, lidx))-__h2f0(lvx(lidz-2, lidx)))
               )/DH;
        vxz.y = (  HC1*(__h2f1(lvx(lidz+1, lidx))-__h2f1(lvx(lidz, lidx)))
               + HC2*(__h2f1(lvx(lidz+2, lidx))-__h2f1(lvx(lidz-1, lidx)))
               + HC3*(__h2f1(lvx(lidz+3, lidx))-__h2f1(lvx(lidz-2, lidx)))
               )/DH;
#elif FDOH==4
        vxx.x = (   HC1*(__h2f0(lvx(lidz, lidx))  -__h2f0(lvx(lidz, lidx-1)))
               + HC2*(__h2f0(lvx(lidz, lidx+1))-__h2f0(lvx(lidz, lidx-2)))
               + HC3*(__h2f0(lvx(lidz, lidx+2))-__h2f0(lvx(lidz, lidx-3)))
               + HC4*(__h2f0(lvx(lidz, lidx+3))-__h2f0(lvx(lidz, lidx-4)))
               )/DH;
        vxx.y = (   HC1*(__h2f1(lvx(lidz, lidx))  -__h2f1(lvx(lidz, lidx-1)))
               + HC2*(__h2f1(lvx(lidz, lidx+1))-__h2f1(lvx(lidz, lidx-2)))
               + HC3*(__h2f1(lvx(lidz, lidx+2))-__h2f1(lvx(lidz, lidx-3)))
               + HC4*(__h2f1(lvx(lidz, lidx+3))-__h2f1(lvx(lidz, lidx-4)))
               )/DH;
        vxz.x = (  HC1*(__h2f0(lvx(lidz+1, lidx))-__h2f0(lvx(lidz, lidx)))
               + HC2*(__h2f0(lvx(lidz+2, lidx))-__h2f0(lvx(lidz-1, lidx)))
               + HC3*(__h2f0(lvx(lidz+3, lidx))-__h2f0(lvx(lidz-2, lidx)))
               + HC4*(__h2f0(lvx(lidz+4, lidx))-__h2f0(lvx(lidz-3, lidx)))
               )/DH;
        vxz.y = (  HC1*(__h2f1(lvx(lidz+1, lidx))-__h2f1(lvx(lidz, lidx)))
               + HC2*(__h2f1(lvx(lidz+2, lidx))-__h2f1(lvx(lidz-1, lidx)))
               + HC3*(__h2f1(lvx(lidz+3, lidx))-__h2f1(lvx(lidz-2, lidx)))
               + HC4*(__h2f1(lvx(lidz+4, lidx))-__h2f1(lvx(lidz-3, lidx)))
               )/DH;
#elif FDOH==5
        vxx.x = (  HC1*(__h2f0(lvx(lidz, lidx))  -__h2f0(lvx(lidz, lidx-1)))
               + HC2*(__h2f0(lvx(lidz, lidx+1))-__h2f0(lvx(lidz, lidx-2)))
               + HC3*(__h2f0(lvx(lidz, lidx+2))-__h2f0(lvx(lidz, lidx-3)))
               + HC4*(__h2f0(lvx(lidz, lidx+3))-__h2f0(lvx(lidz, lidx-4)))
               + HC5*(__h2f0(lvx(lidz, lidx+4))-__h2f0(lvx(lidz, lidx-5)))
               )/DH;
        vxx.y = (  HC1*(__h2f1(lvx(lidz, lidx))  -__h2f1(lvx(lidz, lidx-1)))
               + HC2*(__h2f1(lvx(lidz, lidx+1))-__h2f1(lvx(lidz, lidx-2)))
               + HC3*(__h2f1(lvx(lidz, lidx+2))-__h2f1(lvx(lidz, lidx-3)))
               + HC4*(__h2f1(lvx(lidz, lidx+3))-__h2f1(lvx(lidz, lidx-4)))
               + HC5*(__h2f1(lvx(lidz, lidx+4))-__h2f1(lvx(lidz, lidx-5)))
               )/DH;
        vxz.x = (  HC1*(__h2f0(lvx(lidz+1, lidx))-__h2f0(lvx(lidz, lidx)))
               + HC2*(__h2f0(lvx(lidz+2, lidx))-__h2f0(lvx(lidz-1, lidx)))
               + HC3*(__h2f0(lvx(lidz+3, lidx))-__h2f0(lvx(lidz-2, lidx)))
               + HC4*(__h2f0(lvx(lidz+4, lidx))-__h2f0(lvx(lidz-3, lidx)))
               + HC5*(__h2f0(lvx(lidz+5, lidx))-__h2f0(lvx(lidz-4, lidx)))
               )/DH;
        vxz.y = (  HC1*(__h2f1(lvx(lidz+1, lidx))-__h2f1(lvx(lidz, lidx)))
               + HC2*(__h2f1(lvx(lidz+2, lidx))-__h2f1(lvx(lidz-1, lidx)))
               + HC3*(__h2f1(lvx(lidz+3, lidx))-__h2f1(lvx(lidz-2, lidx)))
               + HC4*(__h2f1(lvx(lidz+4, lidx))-__h2f1(lvx(lidz-3, lidx)))
               + HC5*(__h2f1(lvx(lidz+5, lidx))-__h2f1(lvx(lidz-4, lidx)))
               )/DH;
#elif FDOH==6
        vxx.x = (  HC1*(__h2f0(lvx(lidz, lidx))  -__h2f0(lvx(lidz, lidx-1)))
               + HC2*(__h2f0(lvx(lidz, lidx+1))-__h2f0(lvx(lidz, lidx-2)))
               + HC3*(__h2f0(lvx(lidz, lidx+2))-__h2f0(lvx(lidz, lidx-3)))
               + HC4*(__h2f0(lvx(lidz, lidx+3))-__h2f0(lvx(lidz, lidx-4)))
               + HC5*(__h2f0(lvx(lidz, lidx+4))-__h2f0(lvx(lidz, lidx-5)))
               + HC6*(__h2f0(lvx(lidz, lidx+5))-__h2f0(lvx(lidz, lidx-6)))
               )/DH;
        vxx.y = (  HC1*(__h2f1(lvx(lidz, lidx))  -__h2f1(lvx(lidz, lidx-1)))
               + HC2*(__h2f1(lvx(lidz, lidx+1))-__h2f1(lvx(lidz, lidx-2)))
               + HC3*(__h2f1(lvx(lidz, lidx+2))-__h2f1(lvx(lidz, lidx-3)))
               + HC4*(__h2f1(lvx(lidz, lidx+3))-__h2f1(lvx(lidz, lidx-4)))
               + HC5*(__h2f1(lvx(lidz, lidx+4))-__h2f1(lvx(lidz, lidx-5)))
               + HC6*(__h2f1(lvx(lidz, lidx+5))-__h2f1(lvx(lidz, lidx-6)))
               )/DH;
        vxz.x = (  HC1*(__h2f0(lvx(lidz+1, lidx))-__h2f0(lvx(lidz, lidx)))
               + HC2*(__h2f0(lvx(lidz+2, lidx))-__h2f0(lvx(lidz-1, lidx)))
               + HC3*(__h2f0(lvx(lidz+3, lidx))-__h2f0(lvx(lidz-2, lidx)))
               + HC4*(__h2f0(lvx(lidz+4, lidx))-__h2f0(lvx(lidz-3, lidx)))
               + HC5*(__h2f0(lvx(lidz+5, lidx))-__h2f0(lvx(lidz-4, lidx)))
               + HC6*(__h2f0(lvx(lidz+6, lidx))-__h2f0(lvx(lidz-5, lidx)))
               )/DH;
        vxz.y = (  HC1*(__h2f1(lvx(lidz+1, lidx))-__h2f1(lvx(lidz, lidx)))
               + HC2*(__h2f1(lvx(lidz+2, lidx))-__h2f1(lvx(lidz-1, lidx)))
               + HC3*(__h2f1(lvx(lidz+3, lidx))-__h2f1(lvx(lidz-2, lidx)))
               + HC4*(__h2f1(lvx(lidz+4, lidx))-__h2f1(lvx(lidz-3, lidx)))
               + HC5*(__h2f1(lvx(lidz+5, lidx))-__h2f1(lvx(lidz-4, lidx)))
               + HC6*(__h2f1(lvx(lidz+6, lidx))-__h2f1(lvx(lidz-5, lidx)))
               )/DH;
#endif
        
        
#if LOCAL_OFF==0
        __syncthreads();
        lvz(lidz,lidx)=vz(gidz, gidx);
        if (lidx<2*FDOH)
            lvz(lidz,lidx-FDOH)=vz(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvz(lidz,lidx+lsizex-3*FDOH)=vz(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvz(lidz,lidx+FDOH)=vz(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvz(lidz,lidx-lsizex+3*FDOH)=vz(gidz,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvz(lidz-FDOH,lidx)=vz(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvz(lidz+FDOH,lidx)=vz(gidz+FDOH,gidx);
        __syncthreads();
#endif
        
#if   FDOH==1
        vzz.x = HC1*(__h2f0(lvz(lidz, lidx))  -__h2f0(lvz(lidz-1, lidx)))/DH;
        vzz.y = HC1*(__h2f1(lvz(lidz, lidx))  -__h2f1(lvz(lidz-1, lidx)))/DH;
        vzx.x = HC1*(__h2f0(lvz(lidz, lidx+1))-__h2f0(lvz(lidz, lidx)))/DH;
        vzx.y = HC1*(__h2f1(lvz(lidz, lidx+1))-__h2f1(lvz(lidz, lidx)))/DH;
#elif FDOH==2
        vzz.x = (  HC1*(__h2f0(lvz(lidz, lidx))  -__h2f0(lvz(lidz-1, lidx)))
               + HC2*(__h2f0(lvz(lidz+1, lidx))-__h2f0(lvz(lidz-2, lidx)))
               )/DH;
        vzz.y = (  HC1*(__h2f1(lvz(lidz, lidx))  -__h2f1(lvz(lidz-1, lidx)))
               + HC2*(__h2f1(lvz(lidz+1, lidx))-__h2f1(lvz(lidz-2, lidx)))
               )/DH;
        vzx.x = (  HC1*(__h2f0(lvz(lidz, lidx+1))-__h2f0(lvz(lidz, lidx)))
               + HC2*(__h2f0(lvz(lidz, lidx+2))-__h2f0(lvz(lidz, lidx-1)))
               )/DH;
        vzx.y = (  HC1*(__h2f1(lvz(lidz, lidx+1))-__h2f1(lvz(lidz, lidx)))
               + HC2*(__h2f1(lvz(lidz, lidx+2))-__h2f1(lvz(lidz, lidx-1)))
               )/DH;
#elif FDOH==3
        vzz.x = (  HC1*(__h2f0(lvz(lidz, lidx))  -__h2f0(lvz(lidz-1, lidx)))
               + HC2*(__h2f0(lvz(lidz+1, lidx))-__h2f0(lvz(lidz-2, lidx)))
               + HC3*(__h2f0(lvz(lidz+2, lidx))-__h2f0(lvz(lidz-3, lidx)))
               )/DH;
        vzz.y = (  HC1*(__h2f1(lvz(lidz, lidx))  -__h2f1(lvz(lidz-1, lidx)))
               + HC2*(__h2f1(lvz(lidz+1, lidx))-__h2f1(lvz(lidz-2, lidx)))
               + HC3*(__h2f1(lvz(lidz+2, lidx))-__h2f1(lvz(lidz-3, lidx)))
               )/DH;
        vzx.x = (  HC1*(__h2f0(lvz(lidz, lidx+1))-__h2f0(lvz(lidz, lidx)))
               + HC2*(__h2f0(lvz(lidz, lidx+2))-__h2f0(lvz(lidz, lidx-1)))
               + HC3*(__h2f0(lvz(lidz, lidx+3))-__h2f0(lvz(lidz, lidx-2)))
               )/DH;
        vzx.y = (  HC1*(__h2f1(lvz(lidz, lidx+1))-__h2f1(lvz(lidz, lidx)))
               + HC2*(__h2f1(lvz(lidz, lidx+2))-__h2f1(lvz(lidz, lidx-1)))
               + HC3*(__h2f1(lvz(lidz, lidx+3))-__h2f1(lvz(lidz, lidx-2)))
               )/DH;
#elif FDOH==4
        vzz.x = (  HC1*(__h2f0(lvz(lidz, lidx))  -__h2f0(lvz(lidz-1, lidx)))
               + HC2*(__h2f0(lvz(lidz+1, lidx))-__h2f0(lvz(lidz-2, lidx)))
               + HC3*(__h2f0(lvz(lidz+2, lidx))-__h2f0(lvz(lidz-3, lidx)))
               + HC4*(__h2f0(lvz(lidz+3, lidx))-__h2f0(lvz(lidz-4, lidx)))
               )/DH;
        vzz.y = (  HC1*(__h2f1(lvz(lidz, lidx))  -__h2f1(lvz(lidz-1, lidx)))
               + HC2*(__h2f1(lvz(lidz+1, lidx))-__h2f1(lvz(lidz-2, lidx)))
               + HC3*(__h2f1(lvz(lidz+2, lidx))-__h2f1(lvz(lidz-3, lidx)))
               + HC4*(__h2f1(lvz(lidz+3, lidx))-__h2f1(lvz(lidz-4, lidx)))
               )/DH;
        vzx.x = (  HC1*(__h2f0(lvz(lidz, lidx+1))-__h2f0(lvz(lidz, lidx)))
               + HC2*(__h2f0(lvz(lidz, lidx+2))-__h2f0(lvz(lidz, lidx-1)))
               + HC3*(__h2f0(lvz(lidz, lidx+3))-__h2f0(lvz(lidz, lidx-2)))
               + HC4*(__h2f0(lvz(lidz, lidx+4))-__h2f0(lvz(lidz, lidx-3)))
               )/DH;
        vzx.y = (  HC1*(__h2f1(lvz(lidz, lidx+1))-__h2f1(lvz(lidz, lidx)))
               + HC2*(__h2f1(lvz(lidz, lidx+2))-__h2f1(lvz(lidz, lidx-1)))
               + HC3*(__h2f1(lvz(lidz, lidx+3))-__h2f1(lvz(lidz, lidx-2)))
               + HC4*(__h2f1(lvz(lidz, lidx+4))-__h2f1(lvz(lidz, lidx-3)))
               )/DH;
#elif FDOH==5
        vzz.x = (  HC1*(__h2f0(lvz(lidz, lidx))  -__h2f0(lvz(lidz-1, lidx)))
               + HC2*(__h2f0(lvz(lidz+1, lidx))-__h2f0(lvz(lidz-2, lidx)))
               + HC3*(__h2f0(lvz(lidz+2, lidx))-__h2f0(lvz(lidz-3, lidx)))
               + HC4*(__h2f0(lvz(lidz+3, lidx))-__h2f0(lvz(lidz-4, lidx)))
               + HC5*(__h2f0(lvz(lidz+4, lidx))-__h2f0(lvz(lidz-5, lidx)))
               )/DH;
        vzz.y = (  HC1*(__h2f1(lvz(lidz, lidx))  -__h2f1(lvz(lidz-1, lidx)))
               + HC2*(__h2f1(lvz(lidz+1, lidx))-__h2f1(lvz(lidz-2, lidx)))
               + HC3*(__h2f1(lvz(lidz+2, lidx))-__h2f1(lvz(lidz-3, lidx)))
               + HC4*(__h2f1(lvz(lidz+3, lidx))-__h2f1(lvz(lidz-4, lidx)))
               + HC5*(__h2f1(lvz(lidz+4, lidx))-__h2f1(lvz(lidz-5, lidx)))
               )/DH;
        vzx.x = (  HC1*(__h2f0(lvz(lidz, lidx+1))-__h2f0(lvz(lidz, lidx)))
               + HC2*(__h2f0(lvz(lidz, lidx+2))-__h2f0(lvz(lidz, lidx-1)))
               + HC3*(__h2f0(lvz(lidz, lidx+3))-__h2f0(lvz(lidz, lidx-2)))
               + HC4*(__h2f0(lvz(lidz, lidx+4))-__h2f0(lvz(lidz, lidx-3)))
               + HC5*(__h2f0(lvz(lidz, lidx+5))-__h2f0(lvz(lidz, lidx-4)))
               )/DH;
        vzx.y = (  HC1*(__h2f1(lvz(lidz, lidx+1))-__h2f1(lvz(lidz, lidx)))
               + HC2*(__h2f1(lvz(lidz, lidx+2))-__h2f1(lvz(lidz, lidx-1)))
               + HC3*(__h2f1(lvz(lidz, lidx+3))-__h2f1(lvz(lidz, lidx-2)))
               + HC4*(__h2f1(lvz(lidz, lidx+4))-__h2f1(lvz(lidz, lidx-3)))
               + HC5*(__h2f1(lvz(lidz, lidx+5))-__h2f1(lvz(lidz, lidx-4)))
               )/DH;
#elif FDOH==6
        vzz.x = (  HC1*(__h2f0(lvz(lidz, lidx))  -__h2f0(lvz(lidz-1, lidx)))
               + HC2*(__h2f0(lvz(lidz+1, lidx))-__h2f0(lvz(lidz-2, lidx)))
               + HC3*(__h2f0(lvz(lidz+2, lidx))-__h2f0(lvz(lidz-3, lidx)))
               + HC4*(__h2f0(lvz(lidz+3, lidx))-__h2f0(lvz(lidz-4, lidx)))
               + HC5*(__h2f0(lvz(lidz+4, lidx))-__h2f0(lvz(lidz-5, lidx)))
               + HC6*(__h2f0(lvz(lidz+5, lidx))-__h2f0(lvz(lidz-6, lidx)))
               )/DH;
        vzz.y = (  HC1*(__h2f1(lvz(lidz, lidx))  -__h2f1(lvz(lidz-1, lidx)))
               + HC2*(__h2f1(lvz(lidz+1, lidx))-__h2f1(lvz(lidz-2, lidx)))
               + HC3*(__h2f1(lvz(lidz+2, lidx))-__h2f1(lvz(lidz-3, lidx)))
               + HC4*(__h2f1(lvz(lidz+3, lidx))-__h2f1(lvz(lidz-4, lidx)))
               + HC5*(__h2f1(lvz(lidz+4, lidx))-__h2f1(lvz(lidz-5, lidx)))
               + HC6*(__h2f1(lvz(lidz+5, lidx))-__h2f1(lvz(lidz-6, lidx)))
               )/DH;
        vzx.x = (  HC1*(__h2f0(lvz(lidz, lidx+1))-__h2f0(lvz(lidz, lidx)))
               + HC2*(__h2f0(lvz(lidz, lidx+2))-__h2f0(lvz(lidz, lidx-1)))
               + HC3*(__h2f0(lvz(lidz, lidx+3))-__h2f0(lvz(lidz, lidx-2)))
               + HC4*(__h2f0(lvz(lidz, lidx+4))-__h2f0(lvz(lidz, lidx-3)))
               + HC5*(__h2f0(lvz(lidz, lidx+5))-__h2f0(lvz(lidz, lidx-4)))
               + HC6*(__h2f0(lvz(lidz, lidx+6))-__h2f0(lvz(lidz, lidx-5)))
               )/DH;
        vzx.y = (  HC1*(__h2f1(lvz(lidz, lidx+1))-__h2f1(lvz(lidz, lidx)))
               + HC2*(__h2f1(lvz(lidz, lidx+2))-__h2f1(lvz(lidz, lidx-1)))
               + HC3*(__h2f1(lvz(lidz, lidx+3))-__h2f1(lvz(lidz, lidx-2)))
               + HC4*(__h2f1(lvz(lidz, lidx+4))-__h2f1(lvz(lidz, lidx-3)))
               + HC5*(__h2f1(lvz(lidz, lidx+5))-__h2f1(lvz(lidz, lidx-4)))
               + HC6*(__h2f1(lvz(lidz, lidx+6))-__h2f1(lvz(lidz, lidx-5)))
               )/DH;
#endif
    }
    
    
    // To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
#if LOCAL_OFF==0
#if COMM12==0
    if (gidz>(NZ-FDOH-1) || (gidx-offcomm)>(NX-FDOH-1-LCOMM) ){
        return;
    }
#else
    if (gidz>(NZ-FDOH-1) ){
        return;
    }
#endif
#endif
    
    
    // Correct spatial derivatives to implement CPML
    
//#if ABS_TYPE==1
//    {
//        
//        if (gidz>NZ-NAB-FDOH-1){
//            
//            i =gidx-FDOH;
//            k =gidz - NZ+NAB+FDOH+NAB;
//            ind=2*NAB-1-k;
//            
//            psi_vx_z(k,i) = b_z_half[ind] * psi_vx_z(k,i) + a_z_half[ind] * vxz;
//            vxz = vxz / K_z_half[ind] + psi_vx_z(k,i);
//            psi_vz_z(k,i) = b_z[ind+1] * psi_vz_z(k,i) + a_z[ind+1] * vzz;
//            vzz = vzz / K_z[ind+1] + psi_vz_z(k,i);
//            
//        }
//        
//#if FREESURF==0
//        else if (gidz-FDOH<NAB){
//            
//            i =gidx-FDOH;
//            k =gidz-FDOH;
//            
//            
//            psi_vx_z(k,i) = b_z_half[k] * psi_vx_z(k,i) + a_z_half[k] * vxz;
//            vxz = vxz / K_z_half[k] + psi_vx_z(k,i);
//            psi_vz_z(k,i) = b_z[k] * psi_vz_z(k,i) + a_z[k] * vzz;
//            vzz = vzz / K_z[k] + psi_vz_z(k,i);
//            
//            
//        }
//#endif
//        
//#if DEVID==0 & MYLOCALID==0
//        if (gidx-FDOH<NAB){
//            
//            i =gidx-FDOH;
//            k =gidz-FDOH;
//            
//            psi_vx_x(k,i) = b_x[i] * psi_vx_x(k,i) + a_x[i] * vxx;
//            vxx = vxx / K_x[i] + psi_vx_x(k,i);
//            psi_vz_x(k,i) = b_x_half[i] * psi_vz_x(k,i) + a_x_half[i] * vzx;
//            vzx = vzx / K_x_half[i] + psi_vz_x(k,i);
//            
//        }
//#endif
//        
//#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//        if (gidx>NX-NAB-FDOH-1){
//            
//            i =gidx - NX+NAB+FDOH+NAB;
//            k =gidz-FDOH;
//            ind=2*NAB-1-i;
//            
//            
//            psi_vx_x(k,i) = b_x[ind+1] * psi_vx_x(k,i) + a_x[ind+1] * vxx;
//            vxx = vxx /K_x[ind+1] + psi_vx_x(k,i);
//            psi_vz_x(k,i) = b_x_half[ind] * psi_vz_x(k,i) + a_x_half[ind] * vzx;
//            vzx = vzx / K_x_half[ind]  +psi_vz_x(k,i);
//            
//            
//        }
//#endif
//    }
//#endif
    
    
//    // Read model parameters into local memory
//    {
//#if LVE==0
//        fipkp=muipkp(gidz, gidx)*DT;
//        f=2.0*mu(gidz, gidx)*DT;
//        g=M(gidz, gidx)*DT;
//        
//#else
//        
//        lM=M(gidz,gidx);
//        lmu=mu(gidz,gidx);
//        lmuipkp=muipkp(gidz,gidx);
//        ltaup=taup(gidz,gidx);
//        ltaus=taus(gidz,gidx);
//        ltausipkp=tausipkp(gidz,gidx);
//        
//        for (l=0;l<LVE;l++){
//            leta[l]=eta[l];
//        }
//        
//        fipkp=lmuipkp*DT*(1.0+ (float)LVE*ltausipkp);
//        g=lM*(1.0+(float)LVE*ltaup)*DT;
//        f=2.0*lmu*(1.0+(float)LVE*ltaus)*DT;
//        dipkp=lmuipkp*ltausipkp;
//        d=2.0*lmu*ltaus;
//        e=lM*ltaup;
//        
//#endif
//    }
//    
//    // Update the stresses
//    {
//#if LVE==0
//        
//        sxz(gidz, gidx)+=(fipkp*(vxz+vzx));
//        sxx(gidz, gidx)+=(g*(vxx+vzz))-(f*vzz);
//        szz(gidz, gidx)+=(g*(vxx+vzz))-(f*vxx);
//        
//        
//#else
//        
//        
//        /* computing sums of the old memory variables */
//        sumrxz=sumrxx=sumrzz=0;
//        for (l=0;l<LVE;l++){
//            sumrxz+=rxz(gidz,gidx,l);
//            sumrxx+=rxx(gidz,gidx,l);
//            sumrzz+=rzz(gidz,gidx,l);
//        }
//        
//        
//        /* updating components of the stress tensor, partially */
//        lsxz=(fipkp*(vxz+vzx))+(DT2*sumrxz);
//        lsxx=((g*(vxx+vzz))-(f*vzz))+(DT2*sumrxx);
//        lszz=((g*(vxx+vzz))-(f*vxx))+(DT2*sumrzz);
//        
//        
//        /* now updating the memory-variables and sum them up*/
//        sumrxz=sumrxx=sumrzz=0;
//        for (l=0;l<LVE;l++){
//            b=1.0/(1.0+(leta[l]*0.5));
//            c=1.0-(leta[l]*0.5);
//            
//            rxz(gidz,gidx,l)=b*(rxz(gidz,gidx,l)*c-leta[l]*(dipkp*(vxz+vzx)));
//            rxx(gidz,gidx,l)=b*(rxx(gidz,gidx,l)*c-leta[l]*((e*(vxx+vzz))-(d*vzz)));
//            rzz(gidz,gidx,l)=b*(rzz(gidz,gidx,l)*c-leta[l]*((e*(vxx+vzz))-(d*vxx)));
//            
//            sumrxz+=rxz(gidz,gidx,l);
//            sumrxx+=rxx(gidz,gidx,l);
//            sumrzz+=rzz(gidz,gidx,l);
//        }
//        
//        
//        /* and now the components of the stress tensor are
//         completely updated */
//        sxz(gidz, gidx)+= lsxz + (DT2*sumrxz);
//        sxx(gidz, gidx)+= lsxx + (DT2*sumrxx) ;
//        szz(gidz, gidx)+= lszz + (DT2*sumrzz) ;
//        
//#endif
//    }
//    
//    // Absorbing boundary
//#if ABS_TYPE==2
//    {
//        if (gidz-FDOH<NAB){
//            sxx(gidz,gidx)*=taper[gidz-FDOH];
//            szz(gidz,gidx)*=taper[gidz-FDOH];
//            sxz(gidz,gidx)*=taper[gidz-FDOH];
//        }
//        
//        if (gidz>NZ-NAB-FDOH-1){
//            sxx(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
//            szz(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
//            sxz(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
//        }
//        
//#if DEVID==0 & MYLOCALID==0
//        if (gidx-FDOH<NAB){
//            sxx(gidz,gidx)*=taper[gidx-FDOH];
//            szz(gidz,gidx)*=taper[gidx-FDOH];
//            sxz(gidz,gidx)*=taper[gidx-FDOH];
//        }
//#endif
//        
//#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//        if (gidx>NX-NAB-FDOH-1){
//            sxx(gidz,gidx)*=taper[NX-FDOH-gidx-1];
//            szz(gidz,gidx)*=taper[NX-FDOH-gidx-1];
//            sxz(gidz,gidx)*=taper[NX-FDOH-gidx-1];
//        }
//#endif
//    }
//#endif
    
}

