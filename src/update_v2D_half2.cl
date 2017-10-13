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

#define psi_sxx_x(z,x) psi_sxx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_x(z,x) psi_sxz_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_z(z,x) psi_sxz_z[(x)*(2*NAB)+(z)]
#define psi_szz_z(z,x) psi_szz_z[(x)*(2*NAB)+(z)]
#define psi_sxy_x(z,x) psi_sxy_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_syz_z(z,x) psi_syz_z[(x)*(2*NAB)+(z)]

#if LOCAL_OFF==0

#define lvar(z,x)  lvar[(x)*lsizez+(z)]

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

#define __h2f0(x) __half2float(*((half*)&(x)))
#define __h2f1(x) __half2float(*((half*)&(x)+1))

extern "C" __global__ void update_v(int offcomm,
                                    half2 *vx,      half2 *vz,
                                    half2 *sxx,     half2 *szz,     half2 *sxz,
                                    float *rip,     float *rkp,
                                    float *taper,
                                    float *K_z,        float *a_z,          float *b_z,
                                    float *K_z_half,   float *a_z_half,     float *b_z_half,
                                    float *K_x,        float *a_x,          float *b_x,
                                    float *K_x_half,   float *a_x_half,     float *b_x_half,
                                    half2 *psi_sxx_x,  half2 *psi_sxz_x,
                                    half2 *psi_sxz_z,  half2 *psi_szz_z)
{
    
    extern __shared__ half2 lvar[];
    
    
    float2 sxx_x;
    float2 szz_z;
    float2 sxz_x;
    float2 sxz_z;
    
    // If we use local memory
#if LOCAL_OFF==0
    int lsizez = blockDim.x+2*FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
    
#define lsxx lvar
#define lszz lvar
#define lsxz lvar
    
    // If local memory is turned off
#elif LOCAL_OFF==1
    
    int lsizez = blockDim.x+2*FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
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
        lsxx(lidz,lidx)=sxx(gidz, gidx);
        if (lidx<2*FDOH)
            lsxx(lidz,lidx-FDOH)=sxx(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxx(lidz,lidx+lsizex-3*FDOH)=sxx(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxx(lidz,lidx+FDOH)=sxx(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxx(lidz,lidx-lsizex+3*FDOH)=sxx(gidz,gidx-lsizex+3*FDOH);
        
        __syncthreads();
#endif
        
        float a = __h2f0(lsxx(lidz,lidx+1));
//#if   FDOH ==1
//        sxx_x.x = DTDH*HC1*(__half2float(lsxx(lidz,lidx+1).x) - (__half2float(lsxx(lidz,lidx).x)));
//        sxx_x.y = DTDH*HC1*(__half2float(lsxx(lidz,lidx+1).y) - (__half2float(lsxx(lidz,lidx).y)));
//#elif FDOH ==2
//        sxx_x.x = DTDH*(HC1*(__half2float(lsxx(lidz,lidx+1).x) - __half2float(__half2float(lsxx(lidz,lidx).x)))
//                      +HC2*(__half2float(lsxx(lidz,lidx+2).x) - __half2float(lsxx(lidz,lidx-1).x)));
//        sxx_x.y = DTDH*(HC1*(__half2float(lsxx(lidz,lidx+1).y) - __half2float(__half2float(lsxx(lidz,lidx).y)))
//                      +HC2*(__half2float(lsxx(lidz,lidx+2).y) - __half2float(lsxx(lidz,lidx-1).y)));
//#elif FDOH ==3
//        sxx_x.x = DTDH*(HC1*(__half2float(lsxx(lidz,lidx+1).x)-__half2float(lsxx(lidz,lidx).x))+
//                      HC2*(__half2float(lsxx(lidz,lidx+2).x)-__half2float(lsxx(lidz,lidx-1).x))+
//                      HC3*(__half2float(lsxx(lidz,lidx+3).x)-__half2float(lsxx(lidz,lidx-2).x)));
//        sxx_x.y = DTDH*(HC1*(__half2float(lsxx(lidz,lidx+1).y)-__half2float(lsxx(lidz,lidx).y))+
//                      HC2*(__half2float(lsxx(lidz,lidx+2).y)-__half2float(lsxx(lidz,lidx-1).y))+
//                      HC3*(__half2float(lsxx(lidz,lidx+3).y)-__half2float(lsxx(lidz,lidx-2).y)));
//#elif FDOH ==4
//        sxx_x.x = DTDH*(HC1*(__half2float(lsxx(lidz,lidx+1).x)-__half2float(lsxx(lidz,lidx).x))+
//                      HC2*(__half2float(lsxx(lidz,lidx+2).x)-__half2float(lsxx(lidz,lidx-1).x))+
//                      HC3*(__half2float(lsxx(lidz,lidx+3).x)-__half2float(lsxx(lidz,lidx-2).x))+
//                      HC4*(__half2float(lsxx(lidz,lidx+4).x)-__half2float(lsxx(lidz,lidx-3).x)));
//        sxx_x.y = DTDH*(HC1*(__half2float(lsxx(lidz,lidx+1).y)-__half2float(lsxx(lidz,lidx).y))+
//                      HC2*(__half2float(lsxx(lidz,lidx+2).y)-__half2float(lsxx(lidz,lidx-1).y))+
//                      HC3*(__half2float(lsxx(lidz,lidx+3).y)-__half2float(lsxx(lidz,lidx-2).y))+
//                      HC4*(__half2float(lsxx(lidz,lidx+4).y)-__half2float(lsxx(lidz,lidx-3).y)));
//#elif FDOH ==5
//        sxx_x.x = DTDH*(HC1*(__half2float(lsxx(lidz,lidx+1).x)-__half2float(lsxx(lidz,lidx).x))+
//                      HC2*(__half2float(lsxx(lidz,lidx+2).x)-__half2float(lsxx(lidz,lidx-1).x))+
//                      HC3*(__half2float(lsxx(lidz,lidx+3).x)-__half2float(lsxx(lidz,lidx-2).x))+
//                      HC4*(__half2float(lsxx(lidz,lidx+4).x)-__half2float(lsxx(lidz,lidx-3).x))+
//                      HC5*(__half2float(lsxx(lidz,lidx+5).x)-__half2float(lsxx(lidz,lidx-4).x)));
//        sxx_x.y = DTDH*(HC1*(__half2float(lsxx(lidz,lidx+1).y)-__half2float(lsxx(lidz,lidx).y))+
//                      HC2*(__half2float(lsxx(lidz,lidx+2).y)-__half2float(lsxx(lidz,lidx-1).y))+
//                      HC3*(__half2float(lsxx(lidz,lidx+3).y)-__half2float(lsxx(lidz,lidx-2).y))+
//                      HC4*(__half2float(lsxx(lidz,lidx+4).y)-__half2float(lsxx(lidz,lidx-3).y))+
//                      HC5*(__half2float(lsxx(lidz,lidx+5).y)-__half2float(lsxx(lidz,lidx-4).y)));
//#elif FDOH ==6
//        sxx_x.x = DTDH*(HC1*(__half2float(lsxx(lidz,lidx+1).x)-__half2float(lsxx(lidz,lidx).x))+
//                      HC2*(__half2float(lsxx(lidz,lidx+2).x)-__half2float(lsxx(lidz,lidx-1).x))+
//                      HC3*(__half2float(lsxx(lidz,lidx+3).x)-__half2float(lsxx(lidz,lidx-2).x))+
//                      HC4*(__half2float(lsxx(lidz,lidx+4).x)-__half2float(lsxx(lidz,lidx-3).x))+
//                      HC5*(__half2float(lsxx(lidz,lidx+5).x)-__half2float(lsxx(lidz,lidx-4).x))+
//                      HC6*(__half2float(lsxx(lidz,lidx+6).x)-__half2float(lsxx(lidz,lidx-5).x)));
//        sxx_x.y = DTDH*(HC1*(__half2float(lsxx(lidz,lidx+1).y)-__half2float(lsxx(lidz,lidx).y))+
//                      HC2*(__half2float(lsxx(lidz,lidx+2).y)-__half2float(lsxx(lidz,lidx-1).y))+
//                      HC3*(__half2float(lsxx(lidz,lidx+3).y)-__half2float(lsxx(lidz,lidx-2).y))+
//                      HC4*(__half2float(lsxx(lidz,lidx+4).y)-__half2float(lsxx(lidz,lidx-3).y))+
//                      HC5*(__half2float(lsxx(lidz,lidx+5).y)-__half2float(lsxx(lidz,lidx-4).y))+
//                      HC6*(__half2float(lsxx(lidz,lidx+6).y)-__half2float(lsxx(lidz,lidx-5).y)));
//#endif
        
//        
//#if LOCAL_OFF==0
//        __syncthreads();
//        lszz(lidz,lidx)=szz(gidz, gidx);
//        if (lidz<2*FDOH)
//            lszz(lidz-FDOH,lidx)=szz(gidz-FDOH,gidx);
//        if (lidz>(lsizez-2*FDOH-1))
//            lszz(lidz+FDOH,lidx)=szz(gidz+FDOH,gidx);
//        __syncthreads();
//#endif
//        
//#if   FDOH ==1
//        szz_z = DTDH*HC1*(__half2float(lszz(lidz+1,lidx)) - __half2float(lszz(lidz,lidx)));
//#elif FDOH ==2
//        szz_z = DTDH*(HC1*(__half2float(lszz(lidz+1,lidx)) - __half2float(lszz(lidz,lidx)))
//                      +HC2*(__half2float(lszz(lidz+2,lidx)) - __half2float(lszz(lidz-1,lidx))));
//#elif FDOH ==3
//        szz_z = DTDH*(HC1*(__half2float(lszz(lidz+1,lidx))-__half2float(lszz(lidz,lidx)))+
//                      HC2*(__half2float(lszz(lidz+2,lidx))-__half2float(lszz(lidz-1,lidx)))+
//                      HC3*(__half2float(lszz(lidz+3,lidx))-__half2float(lszz(lidz-2,lidx))));
//#elif FDOH ==4
//        szz_z = DTDH*(HC1*(__half2float(lszz(lidz+1,lidx))-__half2float(lszz(lidz,lidx)))+
//                      HC2*(__half2float(lszz(lidz+2,lidx))-__half2float(lszz(lidz-1,lidx)))+
//                      HC3*(__half2float(lszz(lidz+3,lidx))-__half2float(lszz(lidz-2,lidx)))+
//                      HC4*(__half2float(lszz(lidz+4,lidx))-__half2float(lszz(lidz-3,lidx))));
//#elif FDOH ==5
//        szz_z = DTDH*(HC1*(__half2float(lszz(lidz+1,lidx))-__half2float(lszz(lidz,lidx)))+
//                      HC2*(__half2float(lszz(lidz+2,lidx))-__half2float(lszz(lidz-1,lidx)))+
//                      HC3*(__half2float(lszz(lidz+3,lidx))-__half2float(lszz(lidz-2,lidx)))+
//                      HC4*(__half2float(lszz(lidz+4,lidx))-__half2float(lszz(lidz-3,lidx)))+
//                      HC5*(__half2float(lszz(lidz+5,lidx))-__half2float(lszz(lidz-4,lidx))));
//#elif FDOH ==6
//        szz_z = DTDH*(HC1*(__half2float(lszz(lidz+1,lidx))-__half2float(lszz(lidz,lidx)))+
//                      HC2*(__half2float(lszz(lidz+2,lidx))-__half2float(lszz(lidz-1,lidx)))+
//                      HC3*(__half2float(lszz(lidz+3,lidx))-__half2float(lszz(lidz-2,lidx)))+
//                      HC4*(__half2float(lszz(lidz+4,lidx))-__half2float(lszz(lidz-3,lidx)))+
//                      HC5*(__half2float(lszz(lidz+5,lidx))-__half2float(lszz(lidz-4,lidx)))+
//                      HC6*(__half2float(lszz(lidz+6,lidx))-__half2float(lszz(lidz-5,lidx))));
//#endif
//        
//#if LOCAL_OFF==0
//        __syncthreads();
//        lsxz(lidz,lidx)=sxz(gidz, gidx);
//        
//        if (lidx<2*FDOH)
//            lsxz(lidz,lidx-FDOH)=sxz(gidz,gidx-FDOH);
//        if (lidx+lsizex-3*FDOH<FDOH)
//            lsxz(lidz,lidx+lsizex-3*FDOH)=sxz(gidz,gidx+lsizex-3*FDOH);
//        if (lidx>(lsizex-2*FDOH-1))
//            lsxz(lidz,lidx+FDOH)=sxz(gidz,gidx+FDOH);
//        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
//            lsxz(lidz,lidx-lsizex+3*FDOH)=sxz(gidz,gidx-lsizex+3*FDOH);
//        if (lidz<2*FDOH)
//            lsxz(lidz-FDOH,lidx)=sxz(gidz-FDOH,gidx);
//        if (lidz>(lsizez-2*FDOH-1))
//            lsxz(lidz+FDOH,lidx)=sxz(gidz+FDOH,gidx);
//        __syncthreads();
//#endif
//        
//#if   FDOH ==1
//        sxz_z = DTDH*HC1*(__half2float(lsxz(lidz,lidx))   - __half2float(lsxz(lidz-1,lidx)));
//        sxz_x = DTDH*HC1*(__half2float(lsxz(lidz,lidx))   - __half2float(lsxz(lidz,lidx-1)));
//#elif FDOH ==2
//        sxz_z = DTDH*(HC1*(__half2float(lsxz(lidz,lidx))   - __half2float(lsxz(lidz-1,lidx)))
//                      +HC2*(__half2float(lsxz(lidz+1,lidx)) - __half2float(lsxz(lidz-2,lidx))));
//        sxz_x = DTDH*(HC1*(__half2float(lsxz(lidz,lidx))   - __half2float(lsxz(lidz,lidx-1)))
//                      +HC2*(__half2float(lsxz(lidz,lidx+1)) - __half2float(lsxz(lidz,lidx-2))));
//#elif FDOH ==3
//        sxz_z = DTDH*(HC1*(__half2float(lsxz(lidz,lidx))  -__half2float(lsxz(lidz-1,lidx)))+
//                      HC2*(__half2float(lsxz(lidz+1,lidx))-__half2float(lsxz(lidz-2,lidx)))+
//                      HC3*(__half2float(lsxz(lidz+2,lidx))-__half2float(lsxz(lidz-3,lidx))));
//        
//        sxz_x = DTDH*(HC1*(__half2float(lsxz(lidz,lidx))  -__half2float(lsxz(lidz,lidx-1)))+
//                      HC2*(__half2float(lsxz(lidz,lidx+1))-__half2float(lsxz(lidz,lidx-2)))+
//                      HC3*(__half2float(lsxz(lidz,lidx+2))-__half2float(lsxz(lidz,lidx-3))));
//#elif FDOH ==4
//        sxz_z = DTDH*(HC1*(__half2float(lsxz(lidz,lidx))  -__half2float(lsxz(lidz-1,lidx)))+
//                      HC2*(__half2float(lsxz(lidz+1,lidx))-__half2float(lsxz(lidz-2,lidx)))+
//                      HC3*(__half2float(lsxz(lidz+2,lidx))-__half2float(lsxz(lidz-3,lidx)))+
//                      HC4*(__half2float(lsxz(lidz+3,lidx))-__half2float(lsxz(lidz-4,lidx))));
//        
//        sxz_x = DTDH*(HC1*(__half2float(lsxz(lidz,lidx))  -__half2float(lsxz(lidz,lidx-1)))+
//                      HC2*(__half2float(lsxz(lidz,lidx+1))-__half2float(lsxz(lidz,lidx-2)))+
//                      HC3*(__half2float(lsxz(lidz,lidx+2))-__half2float(lsxz(lidz,lidx-3)))+
//                      HC4*(__half2float(lsxz(lidz,lidx+3))-__half2float(lsxz(lidz,lidx-4))));
//#elif FDOH ==5
//        sxz_z = DTDH*(HC1*(__half2float(lsxz(lidz,lidx))  -__half2float(lsxz(lidz-1,lidx)))+
//                      HC2*(__half2float(lsxz(lidz+1,lidx))-__half2float(lsxz(lidz-2,lidx)))+
//                      HC3*(__half2float(lsxz(lidz+2,lidx))-__half2float(lsxz(lidz-3,lidx)))+
//                      HC4*(__half2float(lsxz(lidz+3,lidx))-__half2float(lsxz(lidz-4,lidx)))+
//                      HC5*(__half2float(lsxz(lidz+4,lidx))-__half2float(lsxz(lidz-5,lidx))));
//        
//        sxz_x = DTDH*(HC1*(__half2float(lsxz(lidz,lidx))  -__half2float(lsxz(lidz,lidx-1)))+
//                      HC2*(__half2float(lsxz(lidz,lidx+1))-__half2float(lsxz(lidz,lidx-2)))+
//                      HC3*(__half2float(lsxz(lidz,lidx+2))-__half2float(lsxz(lidz,lidx-3)))+
//                      HC4*(__half2float(lsxz(lidz,lidx+3))-__half2float(lsxz(lidz,lidx-4)))+
//                      HC5*(__half2float(lsxz(lidz,lidx+4))-__half2float(lsxz(lidz,lidx-5))));
//#elif FDOH ==6
//        
//        sxz_z = DTDH*(HC1*(__half2float(lsxz(lidz,lidx))  -__half2float(lsxz(lidz-1,lidx)))+
//                      HC2*(__half2float(lsxz(lidz+1,lidx))-__half2float(lsxz(lidz-2,lidx)))+
//                      HC3*(__half2float(lsxz(lidz+2,lidx))-__half2float(lsxz(lidz-3,lidx)))+
//                      HC4*(__half2float(lsxz(lidz+3,lidx))-__half2float(lsxz(lidz-4,lidx)))+
//                      HC5*(__half2float(lsxz(lidz+4,lidx))-__half2float(lsxz(lidz-5,lidx)))+
//                      HC6*(__half2float(lsxz(lidz+5,lidx))-__half2float(lsxz(lidz-6,lidx))));
//        
//        sxz_x = DTDH*(HC1*(__half2float(lsxz(lidz,lidx))  -__half2float(lsxz(lidz,lidx-1)))+
//                      HC2*(__half2float(lsxz(lidz,lidx+1))-__half2float(lsxz(lidz,lidx-2)))+
//                      HC3*(__half2float(lsxz(lidz,lidx+2))-__half2float(lsxz(lidz,lidx-3)))+
//                      HC4*(__half2float(lsxz(lidz,lidx+3))-__half2float(lsxz(lidz,lidx-4)))+
//                      HC5*(__half2float(lsxz(lidz,lidx+4))-__half2float(lsxz(lidz,lidx-5)))+
//                      HC6*(__half2float(lsxz(lidz,lidx+5))-__half2float(lsxz(lidz,lidx-6))));
//#endif
    }
//
//    // To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
//#if LOCAL_OFF==0
//#if COMM12==0
//    if (gidz>(NZ-FDOH-1) || (gidx-offcomm)>(NX-FDOH-1-LCOMM) ){
//        return;
//    }
//    
//#else
//    if (gidz>(NZ-FDOH-1) ){
//        return;
//    }
//#endif
//#endif
//    
//    
//    
//    // Correct spatial derivatives to implement CPML
//#if ABS_TYPE==1
//    {
//        int i,k,ind;
//        
//        if (gidz>NZ-NAB-FDOH-1){
//            
//            i =gidx-FDOH;
//            k =gidz - NZ+NAB+FDOH+NAB;
//            ind=2*NAB-1-k;
//            
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
//    // Update the velocities
//    {
//        vx(gidz,gidx)+= ((sxx_x + sxz_z)/rip(gidz,gidx));
//        vz(gidz,gidx)+= ((szz_z + sxz_x)/rkp(gidz,gidx));
//    }
//    
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

