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

/*Update of the velocity in 2D SV*/

//Define useful macros to be able to write a matrix formulation in 2D with OpenCl
#define lbnd (FDOH+NAB)

#define rho(z,x)    rho[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define rip(z,x)    rip[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define rjp(z,x)    rjp[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define rkp(z,x)    rkp[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define muipkp(z,x) muipkp[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define mu(z,x)        mu[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define M(z,x)      M[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define gradrho(z,x)  gradrho[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define gradM(z,x)  gradM[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define gradmu(z,x)  gradmu[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define gradtaup(z,x)  gradtaup[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define gradtaus(z,x)  gradtaus[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]

#define taus(z,x)        taus[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define tausipkp(z,x) tausipkp[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define taup(z,x)        taup[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]

#define vx(z,x)  vx[(x)*(NZ)+(z)]
#define vy(z,x)  vy[(x)*(NZ)+(z)]
#define vz(z,x)  vz[(x)*(NZ)+(z)]
#define sxx(z,x) sxx[(x)*(NZ)+(z)]
#define szz(z,x) szz[(x)*(NZ)+(z)]
#define sxz(z,x) sxz[(x)*(NZ)+(z)]
#define sxy(z,x) sxy[(x)*(NZ)+(z)]
#define syz(z,x) syz[(x)*(NZ)+(z)]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxy(z,x,l) rxy[(l)*NX*NZ+(x)*NZ+(z)]
#define ryz(z,x,l) ryz[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_sxx_x(z,x) psi_sxx_x[(x)*(NZ-FDOH)+(z)]
#define psi_sxz_x(z,x) psi_sxz_x[(x)*(NZ-FDOH)+(z)]
#define psi_sxz_z(z,x) psi_sxz_z[(x)*(2*NAB)+(z)]
#define psi_szz_z(z,x) psi_szz_z[(x)*(2*NAB)+(z)]
#define psi_sxy_x(z,x) psi_sxy_x[(x)*(NZ-FDOH)+(z)]
#define psi_syz_z(z,x) psi_syz_z[(x)*(2*NAB)+(z)]

#if LOCAL_OFF==0

#define lvar2(z,x)  lvar2[(x)*lsizez+(z)]
#define lvar(z,x)  lvar[(x)*lsizez*2+(z)]

#endif


#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vy0(y,x) vy0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define ry(y,x) ry[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#define PI (3.141592653589793238462643383279502884197169)
#define signals(y,x) signals[(y)*NT+(x)]

#define __h2f(x) __half2float((x))

extern "C" __global__ void update_v(int offcomm,
                                    half2 *vx,      half2 *vz,
                                    half2 *sxx,     half2 *szz,     half2 *sxz,
                                    float2 *rip,     float2 *rkp,
                                    float *taper,
                                    float *K_z,        float *a_z,          float *b_z,
                                    float *K_z_half,   float *a_z_half,     float *b_z_half,
                                    float *K_x,        float *a_x,          float *b_x,
                                    float *K_x_half,   float *a_x_half,     float *b_x_half,
                                    half2 *psi_sxx_x,  half2 *psi_sxz_x,
                                    half2 *psi_sxz_z,  half2 *psi_szz_z)
{
    
    extern __shared__ half2 lvar2[];
    half * lvar=(half*)lvar2;
    
    float2 sxx_x;
    float2 szz_z;
    float2 sxz_x;
    float2 sxz_z;
    
    // If we use local memory
#if LOCAL_OFF==0
    int lsizez = blockDim.x+FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH/2;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH/2;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
    
#define lsxx2 lvar2
#define lszz2 lvar2
#define lsxz2 lvar2
#define lsxx lvar
#define lszz lvar
#define lsxz lvar
    
    // If local memory is turned off
#elif LOCAL_OFF==1
    
    int lsizez = blockDim.x+FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH/2;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH/2;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
    
#define lsxx sxx
#define lszz szz
#define lsxz sxz
#define lidx gidx
#define lidz gidz
    
#endif
    
    // Calculation of the stresses spatial derivatives
    {
#if LOCAL_OFF==0
        lsxx2(lidz,lidx)=sxx(gidz, gidx);
        if (lidx<2*FDOH)
            lsxx2(lidz,lidx-FDOH)=sxx(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxx2(lidz,lidx+lsizex-3*FDOH)=sxx(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxx2(lidz,lidx+FDOH)=sxx(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxx2(lidz,lidx-lsizex+3*FDOH)=sxx(gidz,gidx-lsizex+3*FDOH);
        
        __syncthreads();
#endif
        

#if   FDOH ==1
        sxx_x.x = DTDH*HC1*(__h2f(lsxx((2*lidz),lidx+1)) - (__h2f(lsxx((2*lidz),lidx))));
        sxx_x.y = DTDH*HC1*(__h2f(lsxx((2*lidz+1),lidx+1)) - (__h2f(lsxx((2*lidz+1),lidx))));
#elif FDOH ==2
        sxx_x.x = DTDH*(HC1*(__h2f(lsxx((2*lidz),lidx+1)) - __h2f(lsxx((2*lidz),lidx)))
                      +HC2*(__h2f(lsxx((2*lidz),lidx+2)) - __h2f(lsxx((2*lidz),lidx-1))));
        sxx_x.y = DTDH*(HC1*(__h2f(lsxx((2*lidz+1),lidx+1)) - __h2f(lsxx((2*lidz+1),lidx)))
                      +HC2*(__h2f(lsxx((2*lidz+1),lidx+2)) - __h2f(lsxx((2*lidz+1),lidx-1))));
#elif FDOH ==3
        sxx_x.x = DTDH*(HC1*(__h2f(lsxx((2*lidz),lidx+1))-__h2f(lsxx((2*lidz),lidx)))+
                      HC2*(__h2f(lsxx((2*lidz),lidx+2))-__h2f(lsxx((2*lidz),lidx-1)))+
                      HC3*(__h2f(lsxx((2*lidz),lidx+3))-__h2f(lsxx((2*lidz),lidx-2))));
        sxx_x.y = DTDH*(HC1*(__h2f(lsxx((2*lidz+1),lidx+1))-__h2f(lsxx((2*lidz+1),lidx)))+
                      HC2*(__h2f(lsxx((2*lidz+1),lidx+2))-__h2f(lsxx((2*lidz+1),lidx-1)))+
                      HC3*(__h2f(lsxx((2*lidz+1),lidx+3))-__h2f(lsxx((2*lidz+1),lidx-2))));
#elif FDOH ==4
        sxx_x.x = DTDH*(HC1*(__h2f(lsxx((2*lidz),lidx+1))-__h2f(lsxx((2*lidz),lidx)))+
                      HC2*(__h2f(lsxx((2*lidz),lidx+2))-__h2f(lsxx((2*lidz),lidx-1)))+
                      HC3*(__h2f(lsxx((2*lidz),lidx+3))-__h2f(lsxx((2*lidz),lidx-2)))+
                      HC4*(__h2f(lsxx((2*lidz),lidx+4))-__h2f(lsxx((2*lidz),lidx-3))));
        sxx_x.y = DTDH*(HC1*(__h2f(lsxx((2*lidz+1),lidx+1))-__h2f(lsxx((2*lidz+1),lidx)))+
                      HC2*(__h2f(lsxx((2*lidz+1),lidx+2))-__h2f(lsxx((2*lidz+1),lidx-1)))+
                      HC3*(__h2f(lsxx((2*lidz+1),lidx+3))-__h2f(lsxx((2*lidz+1),lidx-2)))+
                      HC4*(__h2f(lsxx((2*lidz+1),lidx+4))-__h2f(lsxx((2*lidz+1),lidx-3))));
#elif FDOH ==5
        sxx_x.x = DTDH*(HC1*(__h2f(lsxx((2*lidz),lidx+1))-__h2f(lsxx((2*lidz),lidx)))+
                      HC2*(__h2f(lsxx((2*lidz),lidx+2))-__h2f(lsxx((2*lidz),lidx-1)))+
                      HC3*(__h2f(lsxx((2*lidz),lidx+3))-__h2f(lsxx((2*lidz),lidx-2)))+
                      HC4*(__h2f(lsxx((2*lidz),lidx+4))-__h2f(lsxx((2*lidz),lidx-3)))+
                      HC5*(__h2f(lsxx((2*lidz),lidx+5))-__h2f(lsxx((2*lidz),lidx-4))));
        sxx_x.y = DTDH*(HC1*(__h2f(lsxx((2*lidz+1),lidx+1))-__h2f(lsxx((2*lidz+1),lidx)))+
                      HC2*(__h2f(lsxx((2*lidz+1),lidx+2))-__h2f(lsxx((2*lidz+1),lidx-1)))+
                      HC3*(__h2f(lsxx((2*lidz+1),lidx+3))-__h2f(lsxx((2*lidz+1),lidx-2)))+
                      HC4*(__h2f(lsxx((2*lidz+1),lidx+4))-__h2f(lsxx((2*lidz+1),lidx-3)))+
                      HC5*(__h2f(lsxx((2*lidz+1),lidx+5))-__h2f(lsxx((2*lidz+1),lidx-4))));
#elif FDOH ==6
        sxx_x.x = DTDH*(HC1*(__h2f(lsxx((2*lidz),lidx+1))-__h2f(lsxx((2*lidz),lidx)))+
                      HC2*(__h2f(lsxx((2*lidz),lidx+2))-__h2f(lsxx((2*lidz),lidx-1)))+
                      HC3*(__h2f(lsxx((2*lidz),lidx+3))-__h2f(lsxx((2*lidz),lidx-2)))+
                      HC4*(__h2f(lsxx((2*lidz),lidx+4))-__h2f(lsxx((2*lidz),lidx-3)))+
                      HC5*(__h2f(lsxx((2*lidz),lidx+5))-__h2f(lsxx((2*lidz),lidx-4)))+
                      HC6*(__h2f(lsxx((2*lidz),lidx+6))-__h2f(lsxx((2*lidz),lidx-5))));
        sxx_x.y = DTDH*(HC1*(__h2f(lsxx((2*lidz+1),lidx+1))-__h2f(lsxx((2*lidz+1),lidx)))+
                      HC2*(__h2f(lsxx((2*lidz+1),lidx+2))-__h2f(lsxx((2*lidz+1),lidx-1)))+
                      HC3*(__h2f(lsxx((2*lidz+1),lidx+3))-__h2f(lsxx((2*lidz+1),lidx-2)))+
                      HC4*(__h2f(lsxx((2*lidz+1),lidx+4))-__h2f(lsxx((2*lidz+1),lidx-3)))+
                      HC5*(__h2f(lsxx((2*lidz+1),lidx+5))-__h2f(lsxx((2*lidz+1),lidx-4)))+
                      HC6*(__h2f(lsxx((2*lidz+1),lidx+6))-__h2f(lsxx((2*lidz+1),lidx-5))));
#endif
        

#if LOCAL_OFF==0
        __syncthreads();
        lszz2(lidz,lidx)=szz(gidz, gidx);
        if (lidz<FDOH)
            lszz2(lidz-FDOH/2,lidx)=szz(gidz-FDOH/2,gidx);
        if (lidz>(lsizez-FDOH-1))
            lszz2(lidz+FDOH/2,lidx)=szz(gidz+FDOH/2,gidx);
        __syncthreads();
#endif
        
#if   FDOH ==1
        szz_z.x = DTDH*HC1*(__h2f(lszz((2*lidz)+1,lidx)) - __h2f(lszz((2*lidz),lidx)));
        szz_z.y = DTDH*HC1*(__h2f(lszz((2*lidz+1)+1,lidx)) - __h2f(lszz((2*lidz+1),lidx)));
#elif FDOH ==2
        szz_z.x = DTDH*(HC1*(__h2f(lszz((2*lidz)+1,lidx)) - __h2f(lszz((2*lidz),lidx)))
                      +HC2*(__h2f(lszz((2*lidz)+2,lidx)) - __h2f(lszz((2*lidz)-1,lidx))));
        szz_z.y = DTDH*(HC1*(__h2f(lszz((2*lidz+1)+1,lidx)) - __h2f(lszz((2*lidz+1),lidx)))
                      +HC2*(__h2f(lszz((2*lidz+1)+2,lidx)) - __h2f(lszz((2*lidz+1)-1,lidx))));
#elif FDOH ==3
        szz_z.x = DTDH*(HC1*(__h2f(lszz((2*lidz)+1,lidx))-__h2f(lszz((2*lidz),lidx)))+
                      HC2*(__h2f(lszz((2*lidz)+2,lidx))-__h2f(lszz((2*lidz)-1,lidx)))+
                      HC3*(__h2f(lszz((2*lidz)+3,lidx))-__h2f(lszz((2*lidz)-2,lidx))));
        szz_z.y = DTDH*(HC1*(__h2f(lszz((2*lidz+1)+1,lidx))-__h2f(lszz((2*lidz+1),lidx)))+
                      HC2*(__h2f(lszz((2*lidz+1)+2,lidx))-__h2f(lszz((2*lidz+1)-1,lidx)))+
                      HC3*(__h2f(lszz((2*lidz+1)+3,lidx))-__h2f(lszz((2*lidz+1)-2,lidx))));
#elif FDOH ==4
        szz_z.x = DTDH*(HC1*(__h2f(lszz((2*lidz)+1,lidx))-__h2f(lszz((2*lidz),lidx)))+
                      HC2*(__h2f(lszz((2*lidz)+2,lidx))-__h2f(lszz((2*lidz)-1,lidx)))+
                      HC3*(__h2f(lszz((2*lidz)+3,lidx))-__h2f(lszz((2*lidz)-2,lidx)))+
                      HC4*(__h2f(lszz((2*lidz)+4,lidx))-__h2f(lszz((2*lidz)-3,lidx))));
        szz_z.y = DTDH*(HC1*(__h2f(lszz((2*lidz+1)+1,lidx))-__h2f(lszz((2*lidz+1),lidx)))+
                      HC2*(__h2f(lszz((2*lidz+1)+2,lidx))-__h2f(lszz((2*lidz+1)-1,lidx)))+
                      HC3*(__h2f(lszz((2*lidz+1)+3,lidx))-__h2f(lszz((2*lidz+1)-2,lidx)))+
                      HC4*(__h2f(lszz((2*lidz+1)+4,lidx))-__h2f(lszz((2*lidz+1)-3,lidx))));
#elif FDOH ==5
        szz_z.x = DTDH*(HC1*(__h2f(lszz((2*lidz)+1,lidx))-__h2f(lszz((2*lidz),lidx)))+
                      HC2*(__h2f(lszz((2*lidz)+2,lidx))-__h2f(lszz((2*lidz)-1,lidx)))+
                      HC3*(__h2f(lszz((2*lidz)+3,lidx))-__h2f(lszz((2*lidz)-2,lidx)))+
                      HC4*(__h2f(lszz((2*lidz)+4,lidx))-__h2f(lszz((2*lidz)-3,lidx)))+
                      HC5*(__h2f(lszz((2*lidz)+5,lidx))-__h2f(lszz((2*lidz)-4,lidx))));
        szz_z.y = DTDH*(HC1*(__h2f(lszz((2*lidz+1)+1,lidx))-__h2f(lszz((2*lidz+1),lidx)))+
                      HC2*(__h2f(lszz((2*lidz+1)+2,lidx))-__h2f(lszz((2*lidz+1)-1,lidx)))+
                      HC3*(__h2f(lszz((2*lidz+1)+3,lidx))-__h2f(lszz((2*lidz+1)-2,lidx)))+
                      HC4*(__h2f(lszz((2*lidz+1)+4,lidx))-__h2f(lszz((2*lidz+1)-3,lidx)))+
                      HC5*(__h2f(lszz((2*lidz+1)+5,lidx))-__h2f(lszz((2*lidz+1)-4,lidx))));
#elif FDOH ==6
        szz_z.x = DTDH*(HC1*(__h2f(lszz((2*lidz)+1,lidx))-__h2f(lszz((2*lidz),lidx)))+
                      HC2*(__h2f(lszz((2*lidz)+2,lidx))-__h2f(lszz((2*lidz)-1,lidx)))+
                      HC3*(__h2f(lszz((2*lidz)+3,lidx))-__h2f(lszz((2*lidz)-2,lidx)))+
                      HC4*(__h2f(lszz((2*lidz)+4,lidx))-__h2f(lszz((2*lidz)-3,lidx)))+
                      HC5*(__h2f(lszz((2*lidz)+5,lidx))-__h2f(lszz((2*lidz)-4,lidx)))+
                      HC6*(__h2f(lszz((2*lidz)+6,lidx))-__h2f(lszz((2*lidz)-5,lidx))));
        szz_z.y = DTDH*(HC1*(__h2f(lszz((2*lidz+1)+1,lidx))-__h2f(lszz((2*lidz+1),lidx)))+
                      HC2*(__h2f(lszz((2*lidz+1)+2,lidx))-__h2f(lszz((2*lidz+1)-1,lidx)))+
                      HC3*(__h2f(lszz((2*lidz+1)+3,lidx))-__h2f(lszz((2*lidz+1)-2,lidx)))+
                      HC4*(__h2f(lszz((2*lidz+1)+4,lidx))-__h2f(lszz((2*lidz+1)-3,lidx)))+
                      HC5*(__h2f(lszz((2*lidz+1)+5,lidx))-__h2f(lszz((2*lidz+1)-4,lidx)))+
                      HC6*(__h2f(lszz((2*lidz+1)+6,lidx))-__h2f(lszz((2*lidz+1)-5,lidx))));
#endif
        
#if LOCAL_OFF==0
        __syncthreads();
        lsxz2(lidz,lidx)=sxz(gidz, gidx);
        
        if (lidx<2*FDOH)
            lsxz2(lidz,lidx-FDOH)=sxz(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxz2(lidz,lidx+lsizex-3*FDOH)=sxz(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxz2(lidz,lidx+FDOH)=sxz(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxz2(lidz,lidx-lsizex+3*FDOH)=sxz(gidz,gidx-lsizex+3*FDOH);
        if (lidz<FDOH)
            lsxz2(lidz-FDOH/2,lidx)=sxz(gidz-FDOH/2,gidx);
        if (lidz>(lsizez-FDOH-1))
            lsxz2(lidz+FDOH/2,lidx)=sxz(gidz+FDOH/2,gidx);
        __syncthreads();
#endif
        
#if   FDOH ==1
        sxz_z.x = DTDH*HC1*(__h2f(lsxz((2*lidz),lidx))   - __h2f(lsxz((2*lidz)-1,lidx)));
        sxz_z.y = DTDH*HC1*(__h2f(lsxz((2*lidz+1),lidx))   - __h2f(lsxz((2*lidz+1)-1,lidx)));
        sxz_x.x = DTDH*HC1*(__h2f(lsxz((2*lidz),lidx))   - __h2f(lsxz((2*lidz),lidx-1)));
        sxz_x.y = DTDH*HC1*(__h2f(lsxz((2*lidz+1),lidx))   - __h2f(lsxz((2*lidz+1),lidx-1)));
#elif FDOH ==2
        sxz_z.x = DTDH*(HC1*(__h2f(lsxz((2*lidz),lidx))   - __h2f(lsxz((2*lidz)-1,lidx)))
                      +HC2*(__h2f(lsxz((2*lidz)+1,lidx)) - __h2f(lsxz((2*lidz)-2,lidx))));
        sxz_z.y = DTDH*(HC1*(__h2f(lsxz((2*lidz+1),lidx))   - __h2f(lsxz((2*lidz+1)-1,lidx)))
                      +HC2*(__h2f(lsxz((2*lidz+1)+1,lidx)) - __h2f(lsxz((2*lidz+1)-2,lidx))));
        sxz_x.x = DTDH*(HC1*(__h2f(lsxz((2*lidz),lidx))   - __h2f(lsxz((2*lidz),lidx-1)))
                      +HC2*(__h2f(lsxz((2*lidz),lidx+1)) - __h2f(lsxz((2*lidz),lidx-2))));
        sxz_x.y = DTDH*(HC1*(__h2f(lsxz((2*lidz+1),lidx))   - __h2f(lsxz((2*lidz+1),lidx-1)))
                      +HC2*(__h2f(lsxz((2*lidz+1),lidx+1)) - __h2f(lsxz((2*lidz+1),lidx-2))));
#elif FDOH ==3
        sxz_z.x = DTDH*(HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz)-1,lidx)))+
                      HC2*(__h2f(lsxz((2*lidz)+1,lidx))-__h2f(lsxz((2*lidz)-2,lidx)))+
                      HC3*(__h2f(lsxz((2*lidz)+2,lidx))-__h2f(lsxz((2*lidz)-3,lidx))));
        sxz_z.y = DTDH*(HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1)-1,lidx)))+
                      HC2*(__h2f(lsxz((2*lidz+1)+1,lidx))-__h2f(lsxz((2*lidz+1)-2,lidx)))+
                      HC3*(__h2f(lsxz((2*lidz+1)+2,lidx))-__h2f(lsxz((2*lidz+1)-3,lidx))));
        
        sxz_x.x = DTDH*(HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz),lidx-1)))+
                      HC2*(__h2f(lsxz((2*lidz),lidx+1))-__h2f(lsxz((2*lidz),lidx-2)))+
                      HC3*(__h2f(lsxz((2*lidz),lidx+2))-__h2f(lsxz((2*lidz),lidx-3))));
        sxz_x.y = DTDH*(HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1),lidx-1)))+
                      HC2*(__h2f(lsxz((2*lidz+1),lidx+1))-__h2f(lsxz((2*lidz+1),lidx-2)))+
                      HC3*(__h2f(lsxz((2*lidz+1),lidx+2))-__h2f(lsxz((2*lidz+1),lidx-3))));
#elif FDOH ==4
        sxz_z.x = DTDH*(HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz)-1,lidx)))+
                      HC2*(__h2f(lsxz((2*lidz)+1,lidx))-__h2f(lsxz((2*lidz)-2,lidx)))+
                      HC3*(__h2f(lsxz((2*lidz)+2,lidx))-__h2f(lsxz((2*lidz)-3,lidx)))+
                      HC4*(__h2f(lsxz((2*lidz)+3,lidx))-__h2f(lsxz((2*lidz)-4,lidx))));
        sxz_z.y = DTDH*(HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1)-1,lidx)))+
                      HC2*(__h2f(lsxz((2*lidz+1)+1,lidx))-__h2f(lsxz((2*lidz+1)-2,lidx)))+
                      HC3*(__h2f(lsxz((2*lidz+1)+2,lidx))-__h2f(lsxz((2*lidz+1)-3,lidx)))+
                      HC4*(__h2f(lsxz((2*lidz+1)+3,lidx))-__h2f(lsxz((2*lidz+1)-4,lidx))));
        
        sxz_x.x = DTDH*(HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz),lidx-1)))+
                      HC2*(__h2f(lsxz((2*lidz),lidx+1))-__h2f(lsxz((2*lidz),lidx-2)))+
                      HC3*(__h2f(lsxz((2*lidz),lidx+2))-__h2f(lsxz((2*lidz),lidx-3)))+
                      HC4*(__h2f(lsxz((2*lidz),lidx+3))-__h2f(lsxz((2*lidz),lidx-4))));
        sxz_x.y = DTDH*(HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1),lidx-1)))+
                      HC2*(__h2f(lsxz((2*lidz+1),lidx+1))-__h2f(lsxz((2*lidz+1),lidx-2)))+
                      HC3*(__h2f(lsxz((2*lidz+1),lidx+2))-__h2f(lsxz((2*lidz+1),lidx-3)))+
                      HC4*(__h2f(lsxz((2*lidz+1),lidx+3))-__h2f(lsxz((2*lidz+1),lidx-4))));
#elif FDOH ==5
        sxz_z.x = DTDH*(HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz)-1,lidx)))+
                      HC2*(__h2f(lsxz((2*lidz)+1,lidx))-__h2f(lsxz((2*lidz)-2,lidx)))+
                      HC3*(__h2f(lsxz((2*lidz)+2,lidx))-__h2f(lsxz((2*lidz)-3,lidx)))+
                      HC4*(__h2f(lsxz((2*lidz)+3,lidx))-__h2f(lsxz((2*lidz)-4,lidx)))+
                      HC5*(__h2f(lsxz((2*lidz)+4,lidx))-__h2f(lsxz((2*lidz)-5,lidx))));
        sxz_z.y = DTDH*(HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1)-1,lidx)))+
                      HC2*(__h2f(lsxz((2*lidz+1)+1,lidx))-__h2f(lsxz((2*lidz+1)-2,lidx)))+
                      HC3*(__h2f(lsxz((2*lidz+1)+2,lidx))-__h2f(lsxz((2*lidz+1)-3,lidx)))+
                      HC4*(__h2f(lsxz((2*lidz+1)+3,lidx))-__h2f(lsxz((2*lidz+1)-4,lidx)))+
                      HC5*(__h2f(lsxz((2*lidz+1)+4,lidx))-__h2f(lsxz((2*lidz+1)-5,lidx))));
        
        sxz_x.x = DTDH*(HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz),lidx-1)))+
                      HC2*(__h2f(lsxz((2*lidz),lidx+1))-__h2f(lsxz((2*lidz),lidx-2)))+
                      HC3*(__h2f(lsxz((2*lidz),lidx+2))-__h2f(lsxz((2*lidz),lidx-3)))+
                      HC4*(__h2f(lsxz((2*lidz),lidx+3))-__h2f(lsxz((2*lidz),lidx-4)))+
                      HC5*(__h2f(lsxz((2*lidz),lidx+4))-__h2f(lsxz((2*lidz),lidx-5))));
        sxz_x.y = DTDH*(HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1),lidx-1)))+
                      HC2*(__h2f(lsxz((2*lidz+1),lidx+1))-__h2f(lsxz((2*lidz+1),lidx-2)))+
                      HC3*(__h2f(lsxz((2*lidz+1),lidx+2))-__h2f(lsxz((2*lidz+1),lidx-3)))+
                      HC4*(__h2f(lsxz((2*lidz+1),lidx+3))-__h2f(lsxz((2*lidz+1),lidx-4)))+
                      HC5*(__h2f(lsxz((2*lidz+1),lidx+4))-__h2f(lsxz((2*lidz+1),lidx-5))));
#elif FDOH ==6
        
        sxz_z.x = DTDH*(HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz)-1,lidx)))+
                      HC2*(__h2f(lsxz((2*lidz)+1,lidx))-__h2f(lsxz((2*lidz)-2,lidx)))+
                      HC3*(__h2f(lsxz((2*lidz)+2,lidx))-__h2f(lsxz((2*lidz)-3,lidx)))+
                      HC4*(__h2f(lsxz((2*lidz)+3,lidx))-__h2f(lsxz((2*lidz)-4,lidx)))+
                      HC5*(__h2f(lsxz((2*lidz)+4,lidx))-__h2f(lsxz((2*lidz)-5,lidx)))+
                      HC6*(__h2f(lsxz((2*lidz)+5,lidx))-__h2f(lsxz((2*lidz)-6,lidx))));
        sxz_z.y = DTDH*(HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1)-1,lidx)))+
                      HC2*(__h2f(lsxz((2*lidz+1)+1,lidx))-__h2f(lsxz((2*lidz+1)-2,lidx)))+
                      HC3*(__h2f(lsxz((2*lidz+1)+2,lidx))-__h2f(lsxz((2*lidz+1)-3,lidx)))+
                      HC4*(__h2f(lsxz((2*lidz+1)+3,lidx))-__h2f(lsxz((2*lidz+1)-4,lidx)))+
                      HC5*(__h2f(lsxz((2*lidz+1)+4,lidx))-__h2f(lsxz((2*lidz+1)-5,lidx)))+
                      HC6*(__h2f(lsxz((2*lidz+1)+5,lidx))-__h2f(lsxz((2*lidz+1)-6,lidx))));
        
        sxz_x.x = DTDH*(HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz),lidx-1)))+
                      HC2*(__h2f(lsxz((2*lidz),lidx+1))-__h2f(lsxz((2*lidz),lidx-2)))+
                      HC3*(__h2f(lsxz((2*lidz),lidx+2))-__h2f(lsxz((2*lidz),lidx-3)))+
                      HC4*(__h2f(lsxz((2*lidz),lidx+3))-__h2f(lsxz((2*lidz),lidx-4)))+
                      HC5*(__h2f(lsxz((2*lidz),lidx+4))-__h2f(lsxz((2*lidz),lidx-5)))+
                      HC6*(__h2f(lsxz((2*lidz),lidx+5))-__h2f(lsxz((2*lidz),lidx-6))));
        sxz_x.y = DTDH*(HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1),lidx-1)))+
                      HC2*(__h2f(lsxz((2*lidz+1),lidx+1))-__h2f(lsxz((2*lidz+1),lidx-2)))+
                      HC3*(__h2f(lsxz((2*lidz+1),lidx+2))-__h2f(lsxz((2*lidz+1),lidx-3)))+
                      HC4*(__h2f(lsxz((2*lidz+1),lidx+3))-__h2f(lsxz((2*lidz+1),lidx-4)))+
                      HC5*(__h2f(lsxz((2*lidz+1),lidx+4))-__h2f(lsxz((2*lidz+1),lidx-5)))+
                      HC6*(__h2f(lsxz((2*lidz+1),lidx+5))-__h2f(lsxz((2*lidz+1),lidx-6))));
#endif
    }

    // To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
#if LOCAL_OFF==0
#if COMM12==0
    if (gidz>(NZ-FDOH/2-1) || (gidx-offcomm)>(NX-FDOH-1-LCOMM) ){
        return;
    }
    
#else
    if (gidz>(NZ-FDOH/2-1) ){
        return;
    }
#endif
#endif
    
    
    
//    // Correct spatial derivatives to implement CPML
//#if ABS_TYPE==1
//    {
//        int i,k,ind;
////        float2 lpsi;
//        
//        if (gidz>NZ-NAB-FDOH-1){
//            
//            i =gidx-FDOH;
//            k =gidz - NZ+NAB+FDOH+NAB;
//            ind=2*NAB-1-k;
//            
////            lpsi = __half22float2(psi_sxz_z(k,i));
//            psi_sxz_z(k,i) = b_z[ind+1] * psi_sxz_z(k,i) + a_z[ind+1] * sxz_z;
//            sxz_z = sxz_z / K_z[ind+1] + psi_sxz_z(k,i);
//            psi_szz_z(k,i) = b_z_half[ind] * psi_szz_z(k,i) + a_z_half[ind] * szz_z;
//            szz_z = szz_z / K_z_half[ind] + psi_szz_z(k,i);
//            
//        }
//        
//#if FREESURF==0
//        else if (gidz-FDOH<NAB){
//            
//            i =gidx-FDOH;
//            k =gidz-FDOH;
//            
//            psi_sxz_z(k,i) = b_z[k] * psi_sxz_z(k,i) + a_z[k] * sxz_z;
//            sxz_z = sxz_z / K_z[k] + psi_sxz_z(k,i);
//            psi_szz_z(k,i) = b_z_half[k] * psi_szz_z(k,i) + a_z_half[k] * szz_z;
//            szz_z = szz_z / K_z_half[k] + psi_szz_z(k,i);
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
//            psi_sxx_x(k,i) = b_x_half[i] * psi_sxx_x(k,i) + a_x_half[i] * sxx_x;
//            sxx_x = sxx_x / K_x_half[i] + psi_sxx_x(k,i);
//            psi_sxz_x(k,i) = b_x[i] * psi_sxz_x(k,i) + a_x[i] * sxz_x;
//            sxz_x = sxz_x / K_x[i] + psi_sxz_x(k,i);
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
//            psi_sxx_x(k,i) = b_x_half[ind] * psi_sxx_x(k,i) + a_x_half[ind] * sxx_x;
//            sxx_x = sxx_x / K_x_half[ind] + psi_sxx_x(k,i);
//            psi_sxz_x(k,i) = b_x[ind+1] * psi_sxz_x(k,i) + a_x[ind+1] * sxz_x;
//            sxz_x = sxz_x / K_x[ind+1] + psi_sxz_x(k,i);
//            
//        }
//#endif
//    }
//#endif
//
    // Update the velocities
    {
        float2 lvx = __half22float2(vx(gidz,gidx));
        float2 lvz = __half22float2(vz(gidz,gidx));
//        float2 lrip = __half22float2(rip(gidz,gidx));
//        float2 lrkp = __half22float2(rkp(gidz,gidx));
        float2 lrip = (rip(gidz,gidx));
        float2 lrkp = (rkp(gidz,gidx));
        lvx.x += ((sxx_x.x + sxz_z.x)/lrip.x)*10000000000.0;
        lvx.y += ((sxx_x.y + sxz_z.y)/lrip.y)*10000000000.0;
        lvz.x += ((szz_z.x + sxz_x.x)/lrkp.x)*10000000000.0;
        lvz.y += ((szz_z.y + sxz_x.y)/lrkp.y)*10000000000.0;
        
//        lvx.x=2.0*gidz;
//        lvx.y=2.0*gidz+1.0;
//        lvz.x=2.0*gidz;
//        lvz.y=2.0*gidz+1.0;
        
        vx(gidz,gidx)= __float22half2_rn(lvx);
        vz(gidz,gidx)= __float22half2_rn(lvz);
        
    }
    
//    // Absorbing boundary
//#if ABS_TYPE==2
//    {
//        if (gidz-FDOH<NAB){
//            vx(gidz,gidx)*=taper[gidz-FDOH];
//            vz(gidz,gidx)*=taper[gidz-FDOH];
//        }
//        
//        if (gidz>NZ-NAB-FDOH-1){
//            vx(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
//            vz(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
//        }
//        
//#if DEVID==0 & MYLOCALID==0
//        if (gidx-FDOH<NAB){
//            vx(gidz,gidx)*=taper[gidx-FDOH];
//            vz(gidz,gidx)*=taper[gidx-FDOH];
//        }
//#endif
//        
//#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//        if (gidx>NX-NAB-FDOH-1){
//            vx(gidz,gidx)*=taper[NX-FDOH-gidx-1];
//            vz(gidz,gidx)*=taper[NX-FDOH-gidx-1];
//        }
//#endif 
//    }
//#endif
    
}

