///*------------------------------------------------------------------------
// * Copyright (C) 2016 For the list of authors, see file AUTHORS.
// *
// * This file is part of SeisCL.
// *
// * SeisCL is free software: you can redistribute it and/or modify
// * it under the terms of the GNU General Public License as published by
// * the Free Software Foundation, version 3.0 of the License only.
// *
// * SeisCL is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// * GNU General Public License for more details.
// *
// * You should have received a copy of the GNU General Public License
// * along with SeisCL. See file COPYING and/or
// * <http://www.gnu.org/licenses/gpl-3.0.html>.
// --------------------------------------------------------------------------*/
//
///*Update of the stresses in 2D SV*/
//
///*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
//#define lbnd (FDOH+NAB)
//
//#define rho(z,x)    rho[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define rip(z,x)    rip[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define rjp(z,x)    rjp[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define rkp(z,x)    rkp[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define muipkp(z,x) muipkp[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define mu(z,x)        mu[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define M(z,x)      M[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define gradrho(z,x)  gradrho[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define gradM(z,x)  gradM[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define gradmu(z,x)  gradmu[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define gradtaup(z,x)  gradtaup[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define gradtaus(z,x)  gradtaus[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//
//#define taus(z,x)         taus[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define tausipkp(z,x) tausipkp[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define taup(z,x)         taup[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//
//
//#define vx(z,x)  vx[(x)*(NZ)+(z)]
//#define vz(z,x)  vz[(x)*(NZ)+(z)]
//#define sxx(z,x) sxx[(x)*(NZ)+(z)]
//#define szz(z,x) szz[(x)*(NZ)+(z)]
//#define sxz(z,x) sxz[(x)*(NZ)+(z)]
//
//#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
//#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
//#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]
//
//#define psi_vx_x(z,x) psi_vx_x[(x)*(NZ-FDOH/2)+(z)]
//#define psi_vz_x(z,x) psi_vz_x[(x)*(NZ-FDOH/2)+(z)]
//
//#define psi_vx_z(z,x) psi_vx_z[(x)*(NAB)+(z)]
//#define psi_vz_z(z,x) psi_vz_z[(x)*(NAB)+(z)]
//
//
//#if LOCAL_OFF==0
//
//#define lvar2(z,x)  lvar2[(x)*lsizez+(z)]
//#define lvar(z,x)  lvar[(x)*lsizez*2+(z)]
//
//#endif
//
//
//#define vxout(y,x) vxout[(y)*NT+(x)]
//#define vzout(y,x) vzout[(y)*NT+(x)]
//#define vx0(y,x) vx0[(y)*NT+(x)]
//#define vz0(y,x) vz0[(y)*NT+(x)]
//#define rx(y,x) rx[(y)*NT+(x)]
//#define rz(y,x) rz[(y)*NT+(x)]
//
//#define PI (3.141592653589793238462643383279502884197169)
//#define signals(y,x) signals[(y)*NT+(x)]
//
//#if FP16==1
//
//#define __h2f(x) __half2float((x))
//#define __h22f2(x) __half22float2((x))
//#define __f22h2(x) __float22half2_rn((x))
//#define __prec half
//#define __prec2 half2
//
//#else
//
//#define __h2f(x) (x)
//#define __h22f2(x) (x)
//#define __f22h2(x) (x)
//#define __prec float
//#define __prec2 float2
//
//#endif
//
//extern "C" __global__ void update_s(int offcomm,
//                                    __prec2 *vx,         __prec2 *vz,
//                                    __prec2 *sxx,        __prec2 *szz,        __prec2 *sxz,
//                                    float2 *M,         float2 *mu,          float2 *muipkp,
//                                    __prec2 *rxx,        __prec2 *rzz,        __prec2 *rxz,
//                                    float2 *taus,       float2 *tausipkp,   float2 *taup,
//                                    float *eta,         float *taper,
//                                    float *K_x,        float *a_x,          float *b_x,
//                                    float *K_x_half,   float *a_x_half,     float *b_x_half,
//                                    float *K_z,        float *a_z,          float *b_z,
//                                    float *K_z_half,   float *a_z_half,     float *b_z_half,
//                                    __prec2 *psi_vx_x,    __prec2 *psi_vx_z,
//                                    __prec2 *psi_vz_x,    __prec2 *psi_vz_z,
//                                    int scaler_sxx)
//{
//
//    extern __shared__ __prec2 lvar2[];
//    __prec * lvar = (__prec *)lvar2;
//
//    float2 vxx, vzz, vzx, vxz;
//    int i,k,l,ind;
//    float2 sumrxz, sumrxx, sumrzz;
//    float2 e,g,d,f,fipkp,dipkp;
//    float b,c;
//#if LVE>0
//    float leta[LVE];
//    float2 lrxx[LVE], lrzz[LVE], lrxz[LVE];
//#endif
//    float2 lM, lmu, lmuipkp, ltaup, ltaus, ltausipkp;
//    float2 lsxx, lszz, lsxz;
//
//
//    // If we use local memory
//#if LOCAL_OFF==0
//    int lsizez = blockDim.x+FDOH;
//    int lsizex = blockDim.y+2*FDOH;
//    int lidz = threadIdx.x+FDOH/2;
//    int lidx = threadIdx.y+FDOH;
//    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH/2;
//    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
//
//#define lvx2 lvar2
//#define lvz2 lvar2
//#define lvx lvar
//#define lvz lvar
//
//    // If local memory is turned off
//#elif LOCAL_OFF==1
//
//    int lsizez = blockDim.x+FDOH;
//    int lsizex = blockDim.y+2*FDOH;
//    int lidz = threadIdx.x+FDOH/2;
//    int lidx = threadIdx.y+FDOH;
//    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH/2;
//    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
//
//
//#define lvx vx
//#define lvz vz
//#define lidx gidx
//#define lidz gidz
//
//#endif
//
//    // Calculation of the velocity spatial derivatives
//    {
//#if LOCAL_OFF==0
//        lvx2(lidz,lidx)=vx(gidz, gidx);
//        if (lidx<2*FDOH)
//            lvx2(lidz,lidx-FDOH)=vx(gidz,gidx-FDOH);
//        if (lidx+lsizex-3*FDOH<FDOH)
//            lvx2(lidz,lidx+lsizex-3*FDOH)=vx(gidz,gidx+lsizex-3*FDOH);
//        if (lidx>(lsizex-2*FDOH-1))
//            lvx2(lidz,lidx+FDOH)=vx(gidz,gidx+FDOH);
//        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
//            lvx2(lidz,lidx-lsizex+3*FDOH)=vx(gidz,gidx-lsizex+3*FDOH);
//        if (lidz<FDOH)
//            lvx2(lidz-FDOH/2,lidx)=vx(gidz-FDOH/2,gidx);
//        if (lidz>(lsizez-FDOH-1))
//            lvx2(lidz+FDOH/2,lidx)=vx(gidz+FDOH/2,gidx);
//
//        __syncthreads();
//#endif
//
//#if   FDOH==1
//        vxx.x = HC1*(__h2f(lvx((2*lidz), lidx))  -__h2f(lvx((2*lidz), lidx-1)));
//        vxx.y = HC1*(__h2f(lvx((2*lidz+1), lidx))  -__h2f(lvx((2*lidz+1), lidx-1)));
//        vxz.x = HC1*(__h2f(lvx((2*lidz)+1, lidx))-__h2f(lvx((2*lidz), lidx)));
//        vxz.y = HC1*(__h2f(lvx((2*lidz+1)+1, lidx))-__h2f(lvx((2*lidz+1), lidx)));
//#elif FDOH==2
//        vxx.x = (  HC1*(__h2f(lvx((2*lidz), lidx))  -__h2f(lvx((2*lidz), lidx-1)))
//               + HC2*(__h2f(lvx((2*lidz), lidx+1))-__h2f(lvx((2*lidz), lidx-2)))
//               );
//        vxx.y = (  HC1*(__h2f(lvx((2*lidz+1), lidx))  -__h2f(lvx((2*lidz+1), lidx-1)))
//               + HC2*(__h2f(lvx((2*lidz+1), lidx+1))-__h2f(lvx((2*lidz+1), lidx-2)))
//               );
//        vxz.x = (  HC1*(__h2f(lvx((2*lidz)+1, lidx))-__h2f(lvx((2*lidz), lidx)))
//               + HC2*(__h2f(lvx((2*lidz)+2, lidx))-__h2f(lvx((2*lidz)-1, lidx)))
//               );
//        vxz.y = (  HC1*(__h2f(lvx((2*lidz+1)+1, lidx))-__h2f(lvx((2*lidz+1), lidx)))
//               + HC2*(__h2f(lvx((2*lidz+1)+2, lidx))-__h2f(lvx((2*lidz+1)-1, lidx)))
//               );
//#elif FDOH==3
//        vxx.x = (  HC1*(__h2f(lvx((2*lidz), lidx))  -__h2f(lvx((2*lidz), lidx-1)))
//               + HC2*(__h2f(lvx((2*lidz), lidx+1))-__h2f(lvx((2*lidz), lidx-2)))
//               + HC3*(__h2f(lvx((2*lidz), lidx+2))-__h2f(lvx((2*lidz), lidx-3)))
//               );
//        vxx.y = (  HC1*(__h2f(lvx((2*lidz+1), lidx))  -__h2f(lvx((2*lidz+1), lidx-1)))
//               + HC2*(__h2f(lvx((2*lidz+1), lidx+1))-__h2f(lvx((2*lidz+1), lidx-2)))
//               + HC3*(__h2f(lvx((2*lidz+1), lidx+2))-__h2f(lvx((2*lidz+1), lidx-3)))
//               );
//        vxz.x = (  HC1*(__h2f(lvx((2*lidz)+1, lidx))-__h2f(lvx((2*lidz), lidx)))
//               + HC2*(__h2f(lvx((2*lidz)+2, lidx))-__h2f(lvx((2*lidz)-1, lidx)))
//               + HC3*(__h2f(lvx((2*lidz)+3, lidx))-__h2f(lvx((2*lidz)-2, lidx)))
//               );
//        vxz.y = (  HC1*(__h2f(lvx((2*lidz+1)+1, lidx))-__h2f(lvx((2*lidz+1), lidx)))
//               + HC2*(__h2f(lvx((2*lidz+1)+2, lidx))-__h2f(lvx((2*lidz+1)-1, lidx)))
//               + HC3*(__h2f(lvx((2*lidz+1)+3, lidx))-__h2f(lvx((2*lidz+1)-2, lidx)))
//               );
//#elif FDOH==4
//        vxx.x = (   HC1*(__h2f(lvx((2*lidz), lidx))  -__h2f(lvx((2*lidz), lidx-1)))
//               + HC2*(__h2f(lvx((2*lidz), lidx+1))-__h2f(lvx((2*lidz), lidx-2)))
//               + HC3*(__h2f(lvx((2*lidz), lidx+2))-__h2f(lvx((2*lidz), lidx-3)))
//               + HC4*(__h2f(lvx((2*lidz), lidx+3))-__h2f(lvx((2*lidz), lidx-4)))
//               );
//        vxx.y = (   HC1*(__h2f(lvx((2*lidz+1), lidx))  -__h2f(lvx((2*lidz+1), lidx-1)))
//               + HC2*(__h2f(lvx((2*lidz+1), lidx+1))-__h2f(lvx((2*lidz+1), lidx-2)))
//               + HC3*(__h2f(lvx((2*lidz+1), lidx+2))-__h2f(lvx((2*lidz+1), lidx-3)))
//               + HC4*(__h2f(lvx((2*lidz+1), lidx+3))-__h2f(lvx((2*lidz+1), lidx-4)))
//               );
//        vxz.x = (  HC1*(__h2f(lvx((2*lidz)+1, lidx))-__h2f(lvx((2*lidz), lidx)))
//               + HC2*(__h2f(lvx((2*lidz)+2, lidx))-__h2f(lvx((2*lidz)-1, lidx)))
//               + HC3*(__h2f(lvx((2*lidz)+3, lidx))-__h2f(lvx((2*lidz)-2, lidx)))
//               + HC4*(__h2f(lvx((2*lidz)+4, lidx))-__h2f(lvx((2*lidz)-3, lidx)))
//               );
//        vxz.y = (  HC1*(__h2f(lvx((2*lidz+1)+1, lidx))-__h2f(lvx((2*lidz+1), lidx)))
//               + HC2*(__h2f(lvx((2*lidz+1)+2, lidx))-__h2f(lvx((2*lidz+1)-1, lidx)))
//               + HC3*(__h2f(lvx((2*lidz+1)+3, lidx))-__h2f(lvx((2*lidz+1)-2, lidx)))
//               + HC4*(__h2f(lvx((2*lidz+1)+4, lidx))-__h2f(lvx((2*lidz+1)-3, lidx)))
//               );
//#elif FDOH==5
//        vxx.x = (  HC1*(__h2f(lvx((2*lidz), lidx))  -__h2f(lvx((2*lidz), lidx-1)))
//               + HC2*(__h2f(lvx((2*lidz), lidx+1))-__h2f(lvx((2*lidz), lidx-2)))
//               + HC3*(__h2f(lvx((2*lidz), lidx+2))-__h2f(lvx((2*lidz), lidx-3)))
//               + HC4*(__h2f(lvx((2*lidz), lidx+3))-__h2f(lvx((2*lidz), lidx-4)))
//               + HC5*(__h2f(lvx((2*lidz), lidx+4))-__h2f(lvx((2*lidz), lidx-5)))
//               );
//        vxx.y = (  HC1*(__h2f(lvx((2*lidz+1), lidx))  -__h2f(lvx((2*lidz+1), lidx-1)))
//               + HC2*(__h2f(lvx((2*lidz+1), lidx+1))-__h2f(lvx((2*lidz+1), lidx-2)))
//               + HC3*(__h2f(lvx((2*lidz+1), lidx+2))-__h2f(lvx((2*lidz+1), lidx-3)))
//               + HC4*(__h2f(lvx((2*lidz+1), lidx+3))-__h2f(lvx((2*lidz+1), lidx-4)))
//               + HC5*(__h2f(lvx((2*lidz+1), lidx+4))-__h2f(lvx((2*lidz+1), lidx-5)))
//               );
//        vxz.x = (  HC1*(__h2f(lvx((2*lidz)+1, lidx))-__h2f(lvx((2*lidz), lidx)))
//               + HC2*(__h2f(lvx((2*lidz)+2, lidx))-__h2f(lvx((2*lidz)-1, lidx)))
//               + HC3*(__h2f(lvx((2*lidz)+3, lidx))-__h2f(lvx((2*lidz)-2, lidx)))
//               + HC4*(__h2f(lvx((2*lidz)+4, lidx))-__h2f(lvx((2*lidz)-3, lidx)))
//               + HC5*(__h2f(lvx((2*lidz)+5, lidx))-__h2f(lvx((2*lidz)-4, lidx)))
//               );
//        vxz.y = (  HC1*(__h2f(lvx((2*lidz+1)+1, lidx))-__h2f(lvx((2*lidz+1), lidx)))
//               + HC2*(__h2f(lvx((2*lidz+1)+2, lidx))-__h2f(lvx((2*lidz+1)-1, lidx)))
//               + HC3*(__h2f(lvx((2*lidz+1)+3, lidx))-__h2f(lvx((2*lidz+1)-2, lidx)))
//               + HC4*(__h2f(lvx((2*lidz+1)+4, lidx))-__h2f(lvx((2*lidz+1)-3, lidx)))
//               + HC5*(__h2f(lvx((2*lidz+1)+5, lidx))-__h2f(lvx((2*lidz+1)-4, lidx)))
//               );
//#elif FDOH==6
//        vxx.x = (  HC1*(__h2f(lvx((2*lidz), lidx))  -__h2f(lvx((2*lidz), lidx-1)))
//               + HC2*(__h2f(lvx((2*lidz), lidx+1))-__h2f(lvx((2*lidz), lidx-2)))
//               + HC3*(__h2f(lvx((2*lidz), lidx+2))-__h2f(lvx((2*lidz), lidx-3)))
//               + HC4*(__h2f(lvx((2*lidz), lidx+3))-__h2f(lvx((2*lidz), lidx-4)))
//               + HC5*(__h2f(lvx((2*lidz), lidx+4))-__h2f(lvx((2*lidz), lidx-5)))
//               + HC6*(__h2f(lvx((2*lidz), lidx+5))-__h2f(lvx((2*lidz), lidx-6)))
//               );
//        vxx.y = (  HC1*(__h2f(lvx((2*lidz+1), lidx))  -__h2f(lvx((2*lidz+1), lidx-1)))
//               + HC2*(__h2f(lvx((2*lidz+1), lidx+1))-__h2f(lvx((2*lidz+1), lidx-2)))
//               + HC3*(__h2f(lvx((2*lidz+1), lidx+2))-__h2f(lvx((2*lidz+1), lidx-3)))
//               + HC4*(__h2f(lvx((2*lidz+1), lidx+3))-__h2f(lvx((2*lidz+1), lidx-4)))
//               + HC5*(__h2f(lvx((2*lidz+1), lidx+4))-__h2f(lvx((2*lidz+1), lidx-5)))
//               + HC6*(__h2f(lvx((2*lidz+1), lidx+5))-__h2f(lvx((2*lidz+1), lidx-6)))
//               );
//        vxz.x = (  HC1*(__h2f(lvx((2*lidz)+1, lidx))-__h2f(lvx((2*lidz), lidx)))
//               + HC2*(__h2f(lvx((2*lidz)+2, lidx))-__h2f(lvx((2*lidz)-1, lidx)))
//               + HC3*(__h2f(lvx((2*lidz)+3, lidx))-__h2f(lvx((2*lidz)-2, lidx)))
//               + HC4*(__h2f(lvx((2*lidz)+4, lidx))-__h2f(lvx((2*lidz)-3, lidx)))
//               + HC5*(__h2f(lvx((2*lidz)+5, lidx))-__h2f(lvx((2*lidz)-4, lidx)))
//               + HC6*(__h2f(lvx((2*lidz)+6, lidx))-__h2f(lvx((2*lidz)-5, lidx)))
//               );
//        vxz.y = (  HC1*(__h2f(lvx((2*lidz+1)+1, lidx))-__h2f(lvx((2*lidz+1), lidx)))
//               + HC2*(__h2f(lvx((2*lidz+1)+2, lidx))-__h2f(lvx((2*lidz+1)-1, lidx)))
//               + HC3*(__h2f(lvx((2*lidz+1)+3, lidx))-__h2f(lvx((2*lidz+1)-2, lidx)))
//               + HC4*(__h2f(lvx((2*lidz+1)+4, lidx))-__h2f(lvx((2*lidz+1)-3, lidx)))
//               + HC5*(__h2f(lvx((2*lidz+1)+5, lidx))-__h2f(lvx((2*lidz+1)-4, lidx)))
//               + HC6*(__h2f(lvx((2*lidz+1)+6, lidx))-__h2f(lvx((2*lidz+1)-5, lidx)))
//               );
//#endif
//
//
//#if LOCAL_OFF==0
//        __syncthreads();
//        lvz2(lidz,lidx)=vz(gidz, gidx);
//        if (lidx<2*FDOH)
//            lvz2(lidz,lidx-FDOH)=vz(gidz,gidx-FDOH);
//        if (lidx+lsizex-3*FDOH<FDOH)
//            lvz2(lidz,lidx+lsizex-3*FDOH)=vz(gidz,gidx+lsizex-3*FDOH);
//        if (lidx>(lsizex-2*FDOH-1))
//            lvz2(lidz,lidx+FDOH)=vz(gidz,gidx+FDOH);
//        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
//            lvz2(lidz,lidx-lsizex+3*FDOH)=vz(gidz,gidx-lsizex+3*FDOH);
//        if (lidz<FDOH)
//            lvz2(lidz-FDOH/2,lidx)=vz(gidz-FDOH/2,gidx);
//        if (lidz>(lsizez-FDOH-1))
//            lvz2(lidz+FDOH/2,lidx)=vz(gidz+FDOH/2,gidx);
//        __syncthreads();
//#endif
//
//#if   FDOH==1
//        vzz.x = HC1*(__h2f(lvz((2*lidz), lidx))  -__h2f(lvz((2*lidz)-1, lidx)));
//        vzz.y = HC1*(__h2f(lvz((2*lidz+1), lidx))  -__h2f(lvz((2*lidz+1)-1, lidx)));
//        vzx.x = HC1*(__h2f(lvz((2*lidz), lidx+1))-__h2f(lvz((2*lidz), lidx)));
//        vzx.y = HC1*(__h2f(lvz((2*lidz+1), lidx+1))-__h2f(lvz((2*lidz+1), lidx)));
//#elif FDOH==2
//        vzz.x = (  HC1*(__h2f(lvz((2*lidz), lidx))  -__h2f(lvz((2*lidz)-1, lidx)))
//               + HC2*(__h2f(lvz((2*lidz)+1, lidx))-__h2f(lvz((2*lidz)-2, lidx)))
//               );
//        vzz.y = (  HC1*(__h2f(lvz((2*lidz+1), lidx))  -__h2f(lvz((2*lidz+1)-1, lidx)))
//               + HC2*(__h2f(lvz((2*lidz+1)+1, lidx))-__h2f(lvz((2*lidz+1)-2, lidx)))
//               );
//        vzx.x = (  HC1*(__h2f(lvz((2*lidz), lidx+1))-__h2f(lvz((2*lidz), lidx)))
//               + HC2*(__h2f(lvz((2*lidz), lidx+2))-__h2f(lvz((2*lidz), lidx-1)))
//               );
//        vzx.y = (  HC1*(__h2f(lvz((2*lidz+1), lidx+1))-__h2f(lvz((2*lidz+1), lidx)))
//               + HC2*(__h2f(lvz((2*lidz+1), lidx+2))-__h2f(lvz((2*lidz+1), lidx-1)))
//               );
//#elif FDOH==3
//        vzz.x = (  HC1*(__h2f(lvz((2*lidz), lidx))  -__h2f(lvz((2*lidz)-1, lidx)))
//               + HC2*(__h2f(lvz((2*lidz)+1, lidx))-__h2f(lvz((2*lidz)-2, lidx)))
//               + HC3*(__h2f(lvz((2*lidz)+2, lidx))-__h2f(lvz((2*lidz)-3, lidx)))
//               );
//        vzz.y = (  HC1*(__h2f(lvz((2*lidz+1), lidx))  -__h2f(lvz((2*lidz+1)-1, lidx)))
//               + HC2*(__h2f(lvz((2*lidz+1)+1, lidx))-__h2f(lvz((2*lidz+1)-2, lidx)))
//               + HC3*(__h2f(lvz((2*lidz+1)+2, lidx))-__h2f(lvz((2*lidz+1)-3, lidx)))
//               );
//        vzx.x = (  HC1*(__h2f(lvz((2*lidz), lidx+1))-__h2f(lvz((2*lidz), lidx)))
//               + HC2*(__h2f(lvz((2*lidz), lidx+2))-__h2f(lvz((2*lidz), lidx-1)))
//               + HC3*(__h2f(lvz((2*lidz), lidx+3))-__h2f(lvz((2*lidz), lidx-2)))
//               );
//        vzx.y = (  HC1*(__h2f(lvz((2*lidz+1), lidx+1))-__h2f(lvz((2*lidz+1), lidx)))
//               + HC2*(__h2f(lvz((2*lidz+1), lidx+2))-__h2f(lvz((2*lidz+1), lidx-1)))
//               + HC3*(__h2f(lvz((2*lidz+1), lidx+3))-__h2f(lvz((2*lidz+1), lidx-2)))
//               );
//#elif FDOH==4
//        vzz.x = (  HC1*(__h2f(lvz((2*lidz), lidx))  -__h2f(lvz((2*lidz)-1, lidx)))
//               + HC2*(__h2f(lvz((2*lidz)+1, lidx))-__h2f(lvz((2*lidz)-2, lidx)))
//               + HC3*(__h2f(lvz((2*lidz)+2, lidx))-__h2f(lvz((2*lidz)-3, lidx)))
//               + HC4*(__h2f(lvz((2*lidz)+3, lidx))-__h2f(lvz((2*lidz)-4, lidx)))
//               );
//        vzz.y = (  HC1*(__h2f(lvz((2*lidz+1), lidx))  -__h2f(lvz((2*lidz+1)-1, lidx)))
//               + HC2*(__h2f(lvz((2*lidz+1)+1, lidx))-__h2f(lvz((2*lidz+1)-2, lidx)))
//               + HC3*(__h2f(lvz((2*lidz+1)+2, lidx))-__h2f(lvz((2*lidz+1)-3, lidx)))
//               + HC4*(__h2f(lvz((2*lidz+1)+3, lidx))-__h2f(lvz((2*lidz+1)-4, lidx)))
//               );
//        vzx.x = (  HC1*(__h2f(lvz((2*lidz), lidx+1))-__h2f(lvz((2*lidz), lidx)))
//               + HC2*(__h2f(lvz((2*lidz), lidx+2))-__h2f(lvz((2*lidz), lidx-1)))
//               + HC3*(__h2f(lvz((2*lidz), lidx+3))-__h2f(lvz((2*lidz), lidx-2)))
//               + HC4*(__h2f(lvz((2*lidz), lidx+4))-__h2f(lvz((2*lidz), lidx-3)))
//               );
//        vzx.y = (  HC1*(__h2f(lvz((2*lidz+1), lidx+1))-__h2f(lvz((2*lidz+1), lidx)))
//               + HC2*(__h2f(lvz((2*lidz+1), lidx+2))-__h2f(lvz((2*lidz+1), lidx-1)))
//               + HC3*(__h2f(lvz((2*lidz+1), lidx+3))-__h2f(lvz((2*lidz+1), lidx-2)))
//               + HC4*(__h2f(lvz((2*lidz+1), lidx+4))-__h2f(lvz((2*lidz+1), lidx-3)))
//               );
//#elif FDOH==5
//        vzz.x = (  HC1*(__h2f(lvz((2*lidz), lidx))  -__h2f(lvz((2*lidz)-1, lidx)))
//               + HC2*(__h2f(lvz((2*lidz)+1, lidx))-__h2f(lvz((2*lidz)-2, lidx)))
//               + HC3*(__h2f(lvz((2*lidz)+2, lidx))-__h2f(lvz((2*lidz)-3, lidx)))
//               + HC4*(__h2f(lvz((2*lidz)+3, lidx))-__h2f(lvz((2*lidz)-4, lidx)))
//               + HC5*(__h2f(lvz((2*lidz)+4, lidx))-__h2f(lvz((2*lidz)-5, lidx)))
//               );
//        vzz.y = (  HC1*(__h2f(lvz((2*lidz+1), lidx))  -__h2f(lvz((2*lidz+1)-1, lidx)))
//               + HC2*(__h2f(lvz((2*lidz+1)+1, lidx))-__h2f(lvz((2*lidz+1)-2, lidx)))
//               + HC3*(__h2f(lvz((2*lidz+1)+2, lidx))-__h2f(lvz((2*lidz+1)-3, lidx)))
//               + HC4*(__h2f(lvz((2*lidz+1)+3, lidx))-__h2f(lvz((2*lidz+1)-4, lidx)))
//               + HC5*(__h2f(lvz((2*lidz+1)+4, lidx))-__h2f(lvz((2*lidz+1)-5, lidx)))
//               );
//        vzx.x = (  HC1*(__h2f(lvz((2*(2*lidz)), lidx+1))-__h2f(lvz((2*lidz), lidx)))
//               + HC2*(__h2f(lvz((2*lidz), lidx+2))-__h2f(lvz((2*lidz), lidx-1)))
//               + HC3*(__h2f(lvz((2*lidz), lidx+3))-__h2f(lvz((2*lidz), lidx-2)))
//               + HC4*(__h2f(lvz((2*lidz), lidx+4))-__h2f(lvz((2*lidz), lidx-3)))
//               + HC5*(__h2f(lvz((2*lidz), lidx+5))-__h2f(lvz((2*lidz), lidx-4)))
//               );
//        vzx.y = (  HC1*(__h2f(lvz((2*lidz+1), lidx+1))-__h2f(lvz((2*lidz+1), lidx)))
//               + HC2*(__h2f(lvz((2*lidz+1), lidx+2))-__h2f(lvz((2*lidz+1), lidx-1)))
//               + HC3*(__h2f(lvz((2*lidz+1), lidx+3))-__h2f(lvz((2*lidz+1), lidx-2)))
//               + HC4*(__h2f(lvz((2*lidz+1), lidx+4))-__h2f(lvz((2*lidz+1), lidx-3)))
//               + HC5*(__h2f(lvz((2*lidz+1), lidx+5))-__h2f(lvz((2*lidz+1), lidx-4)))
//               );
//#elif FDOH==6
//        vzz.x = (  HC1*(__h2f(lvz((2*lidz), lidx))  -__h2f(lvz((2*lidz)-1, lidx)))
//               + HC2*(__h2f(lvz((2*lidz)+1, lidx))-__h2f(lvz((2*lidz)-2, lidx)))
//               + HC3*(__h2f(lvz((2*lidz)+2, lidx))-__h2f(lvz((2*lidz)-3, lidx)))
//               + HC4*(__h2f(lvz((2*lidz)+3, lidx))-__h2f(lvz((2*lidz)-4, lidx)))
//               + HC5*(__h2f(lvz((2*lidz)+4, lidx))-__h2f(lvz((2*lidz)-5, lidx)))
//               + HC6*(__h2f(lvz((2*lidz)+5, lidx))-__h2f(lvz((2*lidz)-6, lidx)))
//               );
//        vzz.y = (  HC1*(__h2f(lvz((2*lidz+1), lidx))  -__h2f(lvz((2*lidz+1)-1, lidx)))
//               + HC2*(__h2f(lvz((2*lidz+1)+1, lidx))-__h2f(lvz((2*lidz+1)-2, lidx)))
//               + HC3*(__h2f(lvz((2*lidz+1)+2, lidx))-__h2f(lvz((2*lidz+1)-3, lidx)))
//               + HC4*(__h2f(lvz((2*lidz+1)+3, lidx))-__h2f(lvz((2*lidz+1)-4, lidx)))
//               + HC5*(__h2f(lvz((2*lidz+1)+4, lidx))-__h2f(lvz((2*lidz+1)-5, lidx)))
//               + HC6*(__h2f(lvz((2*lidz+1)+5, lidx))-__h2f(lvz((2*lidz+1)-6, lidx)))
//               );
//        vzx.x = (  HC1*(__h2f(lvz((2*lidz), lidx+1))-__h2f(lvz((2*lidz), lidx)))
//               + HC2*(__h2f(lvz((2*lidz), lidx+2))-__h2f(lvz((2*lidz), lidx-1)))
//               + HC3*(__h2f(lvz((2*lidz), lidx+3))-__h2f(lvz((2*lidz), lidx-2)))
//               + HC4*(__h2f(lvz((2*lidz), lidx+4))-__h2f(lvz((2*lidz), lidx-3)))
//               + HC5*(__h2f(lvz((2*lidz), lidx+5))-__h2f(lvz((2*lidz), lidx-4)))
//               + HC6*(__h2f(lvz((2*lidz), lidx+6))-__h2f(lvz((2*lidz), lidx-5)))
//               );
//        vzx.y = (  HC1*(__h2f(lvz((2*lidz+1), lidx+1))-__h2f(lvz((2*lidz+1), lidx)))
//               + HC2*(__h2f(lvz((2*lidz+1), lidx+2))-__h2f(lvz((2*lidz+1), lidx-1)))
//               + HC3*(__h2f(lvz((2*lidz+1), lidx+3))-__h2f(lvz((2*lidz+1), lidx-2)))
//               + HC4*(__h2f(lvz((2*lidz+1), lidx+4))-__h2f(lvz((2*lidz+1), lidx-3)))
//               + HC5*(__h2f(lvz((2*lidz+1), lidx+5))-__h2f(lvz((2*lidz+1), lidx-4)))
//               + HC6*(__h2f(lvz((2*lidz+1), lidx+6))-__h2f(lvz((2*lidz+1), lidx-5)))
//               );
//#endif
//
//    }
//
//
//    // To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
//#if LOCAL_OFF==0
//#if COMM12==0
//    if (gidz>(NZ-FDOH/2-1) || (gidx-offcomm)>(NX-FDOH-1-LCOMM) ){
//        return;
//    }
//#else
//    if (gidz>(NZ-FDOH/2-1) ){
//        return;
//    }
//#endif
//#endif
//
//
//    // Correct spatial derivatives to implement CPML
//
//#if ABS_TYPE==1
//    {
//        float2 lpsi_vx_x, lpsi_vx_z, lpsi_vz_x, lpsi_vz_z;
//
//        if (gidz>NZ-NAB/2-FDOH/2-1){
//
//            i =gidx-FDOH;
//            k =gidz - NZ+NAB/2+FDOH/2+NAB/2;
//            ind=2*NAB-1-2*k;
//
//            lpsi_vx_z = __h22f2(psi_vx_z(k,i));
//            lpsi_vz_z = __h22f2(psi_vz_z(k,i));
//
//            lpsi_vx_z.x = b_z_half[ind  ] * lpsi_vx_z.x + a_z_half[ind  ] * vxz.x;
//            lpsi_vx_z.y = b_z_half[ind-1] * lpsi_vx_z.y + a_z_half[ind-1] * vxz.y;
//            vxz.x = vxz.x / K_z_half[ind  ] + lpsi_vx_z.x;
//            vxz.y = vxz.y / K_z_half[ind-1] + lpsi_vx_z.y;
//
//            lpsi_vz_z.x = b_z[ind+1] * lpsi_vz_z.x + a_z[ind+1] * vzz.x;
//            lpsi_vz_z.y = b_z[ind  ] * lpsi_vz_z.y + a_z[ind  ] * vzz.y;
//            vzz.x = vzz.x / K_z[ind+1] + lpsi_vz_z.x;
//            vzz.y = vzz.y / K_z[ind  ] + lpsi_vz_z.y;
//
//            psi_vx_z(k,i)=__f22h2(lpsi_vx_z);
//            psi_vz_z(k,i)=__f22h2(lpsi_vz_z);
//
//        }
//
//#if FREESURF==0
//        else if (gidz-FDOH/2<NAB/2){
//
//            i =gidx-FDOH;
//            k =gidz-FDOH/2;
//
//            lpsi_vx_z = __h22f2(psi_vx_z(k,i));
//            lpsi_vz_z = __h22f2(psi_vz_z(k,i));
//
//            lpsi_vx_z.x = b_z_half[2*k  ] * lpsi_vx_z.x + a_z_half[2*k  ] * vxz.x;
//            lpsi_vx_z.y = b_z_half[2*k+1] * lpsi_vx_z.y + a_z_half[2*k+1] * vxz.y;
//            vxz.x = vxz.x / K_z_half[2*k  ] + lpsi_vx_z.x;
//            vxz.y = vxz.y / K_z_half[2*k+1] + lpsi_vx_z.y;
//
//            lpsi_vz_z.x = b_z[2*k  ] * lpsi_vz_z.x + a_z[2*k  ] * vzz.x;
//            lpsi_vz_z.y = b_z[2*k+1] * lpsi_vz_z.y + a_z[2*k+1] * vzz.y;
//            vzz.x = vzz.x / K_z[2*k  ] + lpsi_vz_z.x;
//            vzz.y = vzz.y / K_z[2*k+1] + lpsi_vz_z.y;
//
//            psi_vz_z(k,i)=__f22h2(lpsi_vz_z);
//            psi_vx_z(k,i)=__f22h2(lpsi_vx_z);
//        }
//#endif
//
//#if DEVID==0 & MYLOCALID==0
//        if (gidx-FDOH<NAB){
//
//            i =gidx-FDOH;
//            k =gidz-FDOH/2;
//
//            lpsi_vz_x = __h22f2(psi_vz_x(k,i));
//            lpsi_vx_x = __h22f2(psi_vx_x(k,i));
//
//            lpsi_vz_x.x = b_x_half[i] * lpsi_vz_x.x + a_x_half[i] * vzx.x;
//            lpsi_vz_x.y = b_x_half[i] * lpsi_vz_x.y + a_x_half[i] * vzx.y;
//            vzx.x = vzx.x / K_x_half[i] + lpsi_vz_x.x;
//            vzx.y = vzx.y / K_x_half[i] + lpsi_vz_x.y;
//
//            lpsi_vx_x.x = b_x[i] * lpsi_vx_x.x + a_x[i] * vxx.x;
//            lpsi_vx_x.y = b_x[i] * lpsi_vx_x.y + a_x[i] * vxx.y;
//            vxx.x = vxx.x / K_x[i] + lpsi_vx_x.x;
//            vxx.y = vxx.y / K_x[i] + lpsi_vx_x.y;
//
//            psi_vz_x(k,i)=__f22h2(lpsi_vz_x);
//            psi_vx_x(k,i)=__f22h2(lpsi_vx_x);
//
//        }
//#endif
//
//#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//        if (gidx>NX-NAB-FDOH-1){
//
//            i =gidx - NX+NAB+FDOH+NAB;
//            k =gidz-FDOH/2;
//            ind=2*NAB-1-i;
//
//            lpsi_vz_x = __h22f2(psi_vz_x(k,i));
//            lpsi_vx_x = __h22f2(psi_vx_x(k,i));
//
//            lpsi_vz_x.x = b_x_half[ind] * lpsi_vz_x.x + a_x_half[ind] * vzx.x;
//            lpsi_vz_x.y = b_x_half[ind] * lpsi_vz_x.y + a_x_half[ind] * vzx.y;
//            vzx.x = vzx.x / K_x_half[ind] + lpsi_vz_x.x;
//            vzx.y = vzx.y / K_x_half[ind] + lpsi_vz_x.y;
//
//            lpsi_vx_x.x = b_x[ind+1] * lpsi_vx_x.x + a_x[ind+1] * vxx.x;
//            lpsi_vx_x.y = b_x[ind+1] * lpsi_vx_x.y + a_x[ind+1] * vxx.y;
//            vxx.x = vxx.x / K_x[ind+1] + lpsi_vx_x.x;
//            vxx.y = vxx.y / K_x[ind+1] + lpsi_vx_x.y;
//
//            psi_vz_x(k,i)=__f22h2(lpsi_vz_x);
//            psi_vx_x(k,i)=__f22h2(lpsi_vx_x);
//        }
//#endif
//    }
//#endif
//
//
//    // Read model parameters into local memory
//    {
//#if LVE==0
//
//        lmuipkp=muipkp(gidz, gidx);
//        lmu=mu(gidz, gidx);
//        lM=M(gidz, gidx);
//
//#else
//
////        lM=     (     M(gidz,gidx));
////        lmu=    (    mu(gidz,gidx));
////        lmuipkp=(muipkp(gidz,gidx));
////        ltaup=  (  taup(gidz,gidx));
////        ltaus=    (    taus(gidz,gidx));
////        ltausipkp=(tausipkp(gidz,gidx));
////
////
////        for (l=0;l<LVE;l++){
////            leta[l]=eta[l];
////        }
////
////        fipkp.x=scalbnf(lmuipkp.x*DTDH*(1.0+ (float)LVE*ltausipkp.x),scaler_sxx);
////        fipkp.y=scalbnf(lmuipkp.y*DTDH*(1.0+ (float)LVE*ltausipkp.y),scaler_sxx);
////        g.x=scalbnf(lM.x*(1.0+(float)LVE*ltaup.x)*DTDH,scaler_sxx);
////        g.y=scalbnf(lM.y*(1.0+(float)LVE*ltaup.y)*DTDH,scaler_sxx);
////        f.x=scalbnf(2.0*lmu.x*(1.0+(float)LVE*ltaus.x)*DTDH,scaler_sxx);
////        f.y=scalbnf(2.0*lmu.y*(1.0+(float)LVE*ltaus.y)*DTDH,scaler_sxx);
////        dipkp.x=scalbnf(lmuipkp.x*ltausipkp.x/DH,scaler_sxx);
////        dipkp.y=scalbnf(lmuipkp.y*ltausipkp.y/DH,scaler_sxx);
////        d.x=scalbnf(2.0*lmu.x*ltaus.x/DH,scaler_sxx);
////        d.y=scalbnf(2.0*lmu.y*ltaus.y/DH,scaler_sxx);
////        e.x=scalbnf(lM.x*ltaup.x/DH,scaler_sxx);
////        e.y=scalbnf(lM.y*ltaup.y/DH,scaler_sxx);
//
//#endif
//    }
//
//    // Update the stresses
//    {
//#if LVE==0
//        lsxx = __h22f2(sxx(gidz, gidx));
//        lszz = __h22f2(szz(gidz, gidx));
//        lsxz = __h22f2(sxz(gidz, gidx));
//
//        lsxz.x+=(lmuipkp.x*(vxz.x+vzx.x));
//        lsxz.y+=(lmuipkp.y*(vxz.y+vzx.y));
//        lsxx.x+=(lM.x*(vxx.x+vzz.x))-(2.0*lmu.x*vzz.x);
//        lsxx.y+=(lM.y*(vxx.y+vzz.y))-(2.0*lmu.y*vzz.y);
//        lszz.x+=(lM.x*(vxx.x+vzz.x))-(2.0*lmu.x*vxx.x);
//        lszz.y+=(lM.y*(vxx.y+vzz.y))-(2.0*lmu.y*vxx.y);
//
//#else
////        /* computing sums of the old memory variables */
////        sumrxz.x=sumrxx.x=sumrzz.x=0;
////        sumrxz.y=sumrxx.y=sumrzz.y=0;
////        for (l=0;l<LVE;l++){
////            lrxx[l] = __h22f2(rxx(gidz,gidx,l));
////            lrzz[l] = __h22f2(rzz(gidz,gidx,l));
////            lrxz[l] = __h22f2(rxz(gidz,gidx,l));
////            sumrxz.x+=lrxz[l].x;
////            sumrxz.y+=lrxz[l].y;
////            sumrxx.x+=lrxx[l].x;
////            sumrxx.y+=lrxx[l].y;
////            sumrzz.x+=lrzz[l].x;
////            sumrzz.y+=lrzz[l].y;
////        }
////
////
////        /* updating components of the stress tensor, partially */
////        lsxx = __h22f2(sxx(gidz, gidx));
////        lszz = __h22f2(szz(gidz, gidx));
////        lsxz = __h22f2(sxz(gidz, gidx));
////
////        lsxz.x+=(fipkp.x*(vxz.x+vzx.x))+(DT2*sumrxz.x);
////        lsxz.y+=(fipkp.y*(vxz.y+vzx.y))+(DT2*sumrxz.y);
////        lsxx.x+=((g.x*(vxx.x+vzz.x))-(f.x*vzz.x))+(DT2*sumrxx.x);
////        lsxx.y+=((g.y*(vxx.y+vzz.y))-(f.y*vzz.y))+(DT2*sumrxx.y);
////        lszz.x+=((g.x*(vxx.x+vzz.x))-(f.x*vxx.x))+(DT2*sumrzz.x);
////        lszz.y+=((g.y*(vxx.y+vzz.y))-(f.y*vxx.y))+(DT2*sumrzz.y);
////
////
////        /* now updating the memory-variables and sum them up*/
////        sumrxz.x=sumrxx.x=sumrzz.x=0;
////        sumrxz.y=sumrxx.y=sumrzz.y=0;
////        for (l=0;l<LVE;l++){
////            b=1.0/(1.0+(leta[l]*0.5));
////            c=1.0-(leta[l]*0.5);
////
////            lrxz[l].x=b*(lrxz[l].x*c-leta[l]*(dipkp.x*(vxz.x+vzx.x)));
////            lrxz[l].y=b*(lrxz[l].y*c-leta[l]*(dipkp.y*(vxz.y+vzx.y)));
////            lrxx[l].x=b*(lrxx[l].x*c-leta[l]*((e.x*(vxx.x+vzz.x))-(d.x*vzz.x)));
////            lrxx[l].y=b*(lrxx[l].y*c-leta[l]*((e.y*(vxx.y+vzz.y))-(d.y*vzz.y)));
////            lrzz[l].x=b*(lrzz[l].x*c-leta[l]*((e.x*(vxx.x+vzz.x))-(d.x*vxx.x)));
////            lrzz[l].y=b*(lrzz[l].y*c-leta[l]*((e.y*(vxx.y+vzz.y))-(d.y*vxx.y)));
////
////            sumrxz.x+=lrxz[l].x;
////            sumrxz.y+=lrxz[l].y;
////            sumrxx.x+=lrxx[l].x;
////            sumrxx.y+=lrxx[l].y;
////            sumrzz.x+=lrzz[l].x;
////            sumrzz.y+=lrzz[l].y;
////
////            rxx(gidz,gidx,l)=__f22h2(lrxx[l]);
////            rzz(gidz,gidx,l)=__f22h2(lrzz[l]);
////            rxz(gidz,gidx,l)=__f22h2(lrxz[l]);
////        }
////
////
////        /* and now the components of the stress tensor are
////         completely updated */
////        lsxz.x+=  (DT2*sumrxz.x);
////        lsxz.y+=  (DT2*sumrxz.y);
////        lsxx.x+=  (DT2*sumrxx.x);
////        lsxx.y+=  (DT2*sumrxx.y);
////        lszz.x+=  (DT2*sumrzz.x);
////        lszz.y+=  (DT2*sumrzz.y);
//
//
//#endif
//    }
//
////    // Absorbing boundary
////#if ABS_TYPE==2
////    {
////        if (2*gidz-FDOH<NAB){
////            lsxx.x*=taper[2*gidz-FDOH];
////            lsxx.y*=taper[2*gidz+1-FDOH];
////            lszz.x*=taper[2*gidz-FDOH];
////            lszz.y*=taper[2*gidz+1-FDOH];
////            lsxz.x*=taper[2*gidz-FDOH];
////            lsxz.y*=taper[2*gidz+1-FDOH];
////        }
////
////        if (2*gidz>2*NZ-NAB-FDOH-1){
////            lsxx.x*=taper[2*NZ-FDOH-2*gidz-1];
////            lsxx.y*=taper[2*NZ-FDOH-2*gidz-1-1];
////            lszz.x*=taper[2*NZ-FDOH-2*gidz-1];
////            lszz.y*=taper[2*NZ-FDOH-2*gidz-1-1];
////            lsxz.x*=taper[2*NZ-FDOH-2*gidz-1];
////            lsxz.y*=taper[2*NZ-FDOH-2*gidz-1-1];
////        }
////
////#if DEVID==0 & MYLOCALID==0
////        if (gidx-FDOH<NAB){
////            lsxx.x*=taper[gidx-FDOH];
////            lsxx.y*=taper[gidx-FDOH];
////            lszz.x*=taper[gidx-FDOH];
////            lszz.y*=taper[gidx-FDOH];
////            lsxz.x*=taper[gidx-FDOH];
////            lsxz.y*=taper[gidx-FDOH];
////        }
////#endif
////
////#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
////        if (gidx>NX-NAB-FDOH-1){
////            lsxx.x*=taper[NX-FDOH-gidx-1];
////            lsxx.y*=taper[NX-FDOH-gidx-1];
////            lszz.x*=taper[NX-FDOH-gidx-1];
////            lszz.y*=taper[NX-FDOH-gidx-1];
////            lsxz.x*=taper[NX-FDOH-gidx-1];
////            lsxz.y*=taper[NX-FDOH-gidx-1];
////        }
////#endif
////    }
////#endif
//
//    sxz(gidz, gidx)=__f22h2(lsxz);
//    sxx(gidz, gidx)=__f22h2(lsxx);
//    szz(gidz, gidx)=__f22h2(lszz);
//}

#define rip(z,x) rip[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH/2)]
#define rkp(z,x) rkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH/2)]
#define muipkp(z,x) muipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH/2)]
#define M(z,x) M[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH/2)]
#define mu(z,x) mu[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH/2)]

#define sxx(z,x) sxx[(x)*NZ+(z)]
#define sxz(z,x) sxz[(x)*NZ+(z)]
#define szz(z,x) szz[(x)*NZ+(z)]
#define vx(z,x) vx[(x)*NZ+(z)]
#define vz(z,x) vz[(x)*NZ+(z)]

#if LOCAL_OFF==0
#define lvar(z,x) lvar[(x)*2*lsizez+(z)]
#define lvar2(z,x) lvar2[(x)*lsizez+(z)]
#endif



#if FP16==1

#define __h2f(x) __half2float((x))
#define __h22f2(x) __half22float2((x))
#define __f22h2(x) __float22half2_rn((x))

#else

#define __h2f(x) (x)
#define __h22f2(x) (x)
#define __f22h2(x) (x)

#endif

#if FP16==0

#define __prec float
#define __prec2 float2

#else

#define __prec half
#define __prec2 half2

#endif


#if FP16!=2

#define __cprec float2
#define __f22h2c(x) (x)

extern "C" __device__ float2 add2(float2 a, float2 b ){
    
    float2 output;
    output.x = a.x+b.x;
    output.y = a.y+b.y;
    return output;
}
extern "C" __device__ float2 mul2(float2 a, float2 b ){
    
    float2 output;
    output.x = a.x*b.x;
    output.y = a.y*b.y;
    return output;
}
extern "C" __device__ float2 sub2(float2 a, float2 b ){
    
    float2 output;
    output.x = a.x-b.x;
    output.y = a.y-b.y;
    return output;
}
extern "C" __device__ float2 f2h2(float a){
    
    float2 output={a,a};
    return output;
}

#else

#define __cprec half2
#define add2 __hadd2
#define mul2 __hmul2
#define sub2 __hsub2
#define f2h2 __float2half2_rn
#define __f22h2c(x) __float22half2_rn((x))

#endif

extern "C" __device__ __prec2 __hp(__prec *a ){
    
    __prec2 output;
    *((__prec *)&output) = *a;
    *((__prec *)&output+1) = *(a+1);
    return output;
}



extern "C" __global__ void update_s(int offcomm,
                                    __prec2 *muipkp, __prec2 *M, __prec2 *mu,
                                    __prec2 *sxx,__prec2 *sxz,__prec2 *szz,
                                    __prec2 *vx,__prec2 *vz
                                    )

{
//    //Local memory
//    extern __shared__ __prec2 lvar2[];
//    __prec * lvar=(__prec *)lvar2;
    
    //Grid position
    int lsizez = blockDim.x+FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH/2;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/2;
    int gidx = blockIdx.y*blockDim.y+threadIdx.y+FDOH+offcomm;
    
    //Define and load private parameters and variables
    __prec2 lsxx ;//= __h22f2(sxx(gidz,gidx));
    __prec2 lsxz ;//= __h22f2(sxz(gidz,gidx));
    __prec2 lszz ;//= __h22f2(szz(gidz,gidx));
//    __cprec lM = __f22h2c(M(gidz,gidx));
//    __cprec lmu = __f22h2c(mu(gidz,gidx));
//    __cprec lmuipkp = __f22h2c(muipkp(gidz,gidx));
    __prec2 lM = (M(gidz,gidx));
    __prec2 lmu = (mu(gidz,gidx));
    __prec2 lmuipkp = (muipkp(gidz,gidx));
    
//    //Define private derivatives
//    __cprec vx_x2;
//    __cprec vx_z1;
//    __cprec vz_x1;
//    __cprec vz_z2;
    
    //Local memory definitions if local is used
#if LOCAL_OFF==0
#define lvx lvar
#define lvz lvar
#define lvx2 lvar2
#define lvz2 lvar2
    
    //Local memory definitions if local is not used
#elif LOCAL_OFF==1
    
#define lvx vx
#define lvz vz
#define lidz gidz
#define lidx gidx
    
#endif
    
//    //Calculation of the spatial derivatives
//    {
//#if LOCAL_OFF==0
//        __syncthreads();
//        lvx2(lidz,lidx)=vx(gidz,gidx);
//        if (lidz<FDOH)
//            lvx2(lidz-FDOH/2,lidx)=vx(gidz-FDOH/2,gidx);
//        if (lidz>(lsizez-FDOH-1))
//            lvx2(lidz+FDOH/2,lidx)=vx(gidz+FDOH/2,gidx);
//        if (lidx<2*FDOH)
//            lvx2(lidz,lidx-FDOH)=vx(gidz,gidx-FDOH);
//        if (lidx+lsizex-3*FDOH<FDOH)
//            lvx2(lidz,lidx+lsizex-3*FDOH)=vx(gidz,gidx+lsizex-3*FDOH);
//        if (lidx>(lsizex-2*FDOH-1))
//            lvx2(lidz,lidx+FDOH)=vx(gidz,gidx+FDOH);
//        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
//            lvx2(lidz,lidx-lsizex+3*FDOH)=vx(gidz,gidx-lsizex+3*FDOH);
//        __syncthreads();
//#endif
//
//#if   FDOH == 1
//        vx_x2=mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1))));
//#elif FDOH == 2
//        vx_x2=add2(
//                   mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
//                   mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2)))));
//#elif FDOH == 3
//        vx_x2=add2(add2(
//                        mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
//                        mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2))))),
//                   mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidx+2)), __h22f2(lvx2(lidz,lidx-3)))));
//#elif FDOH == 4
//        vx_x2=add2(add2(add2(
//                             mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
//                             mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2))))),
//                        mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidx+2)), __h22f2(lvx2(lidz,lidx-3))))),
//                   mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidx+3)), __h22f2(lvx2(lidz,lidx-4)))));
//#elif FDOH == 5
//        vx_x2=add2(add2(add2(add2(
//                                  mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
//                                  mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2))))),
//                             mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidx+2)), __h22f2(lvx2(lidz,lidx-3))))),
//                        mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidx+3)), __h22f2(lvx2(lidz,lidx-4))))),
//                   mul2( f2h2(HC5), sub2(__h22f2(lvx2(lidz,lidx+4)), __h22f2(lvx2(lidz,lidx-5)))));
//#elif FDOH == 6
//        vx_x2=add2(add2(add2(add2(add2(
//                                       mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
//                                       mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2))))),
//                                  mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidx+2)), __h22f2(lvx2(lidz,lidx-3))))),
//                             mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidx+3)), __h22f2(lvx2(lidz,lidx-4))))),
//                        mul2( f2h2(HC5), sub2(__h22f2(lvx2(lidz,lidx+4)), __h22f2(lvx2(lidz,lidx-5))))),
//                   mul2( f2h2(HC6), sub2(__h22f2(lvx2(lidz,lidx+5)), __h22f2(lvx2(lidz,lidx-6)))));
//#endif
//
//#if   FDOH == 1
//        vx_z1=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx)))));
//#elif FDOH == 2
//        vx_z1=add2(
//                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
//                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx))))));
//#elif FDOH == 3
//        vx_z1=add2(add2(
//                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
//                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx)))))),
//                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidx))))));
//#elif FDOH == 4
//        vx_z1=add2(add2(add2(
//                             mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
//                             mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx)))))),
//                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidx)))))),
//                   mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvx(2*lidz+4,lidx))), __h22f2(__hp(&lvx(2*lidz-3,lidx))))));
//#elif FDOH == 5
//        vx_z1=add2(add2(add2(add2(
//                                  mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
//                                  mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx)))))),
//                             mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidx)))))),
//                        mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvx(2*lidz+4,lidx))), __h22f2(__hp(&lvx(2*lidz-3,lidx)))))),
//                   mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvx(2*lidz+5,lidx))), __h22f2(__hp(&lvx(2*lidz-4,lidx))))));
//#elif FDOH == 6
//        vx_z1=add2(add2(add2(add2(add2(
//                                       mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
//                                       mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx)))))),
//                                  mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidx)))))),
//                             mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvx(2*lidz+4,lidx))), __h22f2(__hp(&lvx(2*lidz-3,lidx)))))),
//                        mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvx(2*lidz+5,lidx))), __h22f2(__hp(&lvx(2*lidz-4,lidx)))))),
//                   mul2( f2h2(HC6), sub2(__h22f2(__hp(&lvx(2*lidz+6,lidx))), __h22f2(__hp(&lvx(2*lidz-5,lidx))))));
//#endif
//
//#if LOCAL_OFF==0
//        __syncthreads();
//        lvz2(lidz,lidx)=vz(gidz,gidx);
//        if (lidz<FDOH)
//            lvz2(lidz-FDOH/2,lidx)=vz(gidz-FDOH/2,gidx);
//        if (lidz>(lsizez-FDOH-1))
//            lvz2(lidz+FDOH/2,lidx)=vz(gidz+FDOH/2,gidx);
//        if (lidx<2*FDOH)
//            lvz2(lidz,lidx-FDOH)=vz(gidz,gidx-FDOH);
//        if (lidx+lsizex-3*FDOH<FDOH)
//            lvz2(lidz,lidx+lsizex-3*FDOH)=vz(gidz,gidx+lsizex-3*FDOH);
//        if (lidx>(lsizex-2*FDOH-1))
//            lvz2(lidz,lidx+FDOH)=vz(gidz,gidx+FDOH);
//        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
//            lvz2(lidz,lidx-lsizex+3*FDOH)=vz(gidz,gidx-lsizex+3*FDOH);
//        __syncthreads();
//#endif
//
//#if   FDOH == 1
//        vz_x1=mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx))));
//#elif FDOH == 2
//        vz_x1=add2(
//                   mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
//                   mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1)))));
//#elif FDOH == 3
//        vz_x1=add2(add2(
//                        mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
//                        mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1))))),
//                   mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidx+3)), __h22f2(lvz2(lidz,lidx-2)))));
//#elif FDOH == 4
//        vz_x1=add2(add2(add2(
//                             mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
//                             mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1))))),
//                        mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidx+3)), __h22f2(lvz2(lidz,lidx-2))))),
//                   mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidx+4)), __h22f2(lvz2(lidz,lidx-3)))));
//#elif FDOH == 5
//        vz_x1=add2(add2(add2(add2(
//                                  mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
//                                  mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1))))),
//                             mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidx+3)), __h22f2(lvz2(lidz,lidx-2))))),
//                        mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidx+4)), __h22f2(lvz2(lidz,lidx-3))))),
//                   mul2( f2h2(HC5), sub2(__h22f2(lvz2(lidz,lidx+5)), __h22f2(lvz2(lidz,lidx-4)))));
//#elif FDOH == 6
//        vz_x1=add2(add2(add2(add2(add2(
//                                       mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
//                                       mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1))))),
//                                  mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidx+3)), __h22f2(lvz2(lidz,lidx-2))))),
//                             mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidx+4)), __h22f2(lvz2(lidz,lidx-3))))),
//                        mul2( f2h2(HC5), sub2(__h22f2(lvz2(lidz,lidx+5)), __h22f2(lvz2(lidz,lidx-4))))),
//                   mul2( f2h2(HC6), sub2(__h22f2(lvz2(lidz,lidx+6)), __h22f2(lvz2(lidz,lidx-5)))));
//#endif
//
//#if   FDOH == 1
//        vz_z2=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx)))));
//#elif FDOH == 2
//        vz_z2=add2(
//                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
//                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx))))));
//#elif FDOH == 3
//        vz_z2=add2(add2(
//                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
//                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx)))))),
//                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidx))))));
//#elif FDOH == 4
//        vz_z2=add2(add2(add2(
//                             mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
//                             mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx)))))),
//                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidx)))))),
//                   mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvz(2*lidz+3,lidx))), __h22f2(__hp(&lvz(2*lidz-4,lidx))))));
//#elif FDOH == 5
//        vz_z2=add2(add2(add2(add2(
//                                  mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
//                                  mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx)))))),
//                             mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidx)))))),
//                        mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvz(2*lidz+3,lidx))), __h22f2(__hp(&lvz(2*lidz-4,lidx)))))),
//                   mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvz(2*lidz+4,lidx))), __h22f2(__hp(&lvz(2*lidz-5,lidx))))));
//#elif FDOH == 6
//        vz_z2=add2(add2(add2(add2(add2(
//                                       mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
//                                       mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx)))))),
//                                  mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidx)))))),
//                             mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvz(2*lidz+3,lidx))), __h22f2(__hp(&lvz(2*lidz-4,lidx)))))),
//                        mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvz(2*lidz+4,lidx))), __h22f2(__hp(&lvz(2*lidz-5,lidx)))))),
//                   mul2( f2h2(HC6), sub2(__h22f2(__hp(&lvz(2*lidz+5,lidx))), __h22f2(__hp(&lvz(2*lidz-6,lidx))))));
//#endif
//
//    }
    // To stop updating if we are outside the model (global id must be amultiple of local id in OpenCL, hence we stop if we have a global idoutside the grid)
#if  LOCAL_OFF==0
#if COMM12==0
    if ( gidz>(NZ-FDOH/2-1) ||  (gidx-offcomm)>(NX-FDOH-1-LCOMM) )
        return;
#else
    if ( gidz>(NZ-FDOH/2-1)  )
        return;
#endif
#endif
    
    // Update the variables
//    lsxz=add2(lsxz,mul2(lmuipkp,add2(vx_z1,vz_x1)));
//    lsxx=sub2(add2(lsxx,mul2(lM,add2(vx_x2,vz_z2))),mul2(mul2(f2h2(2.0),lmu),vz_z2));
//    lszz=sub2(add2(lszz,mul2(lM,add2(vx_x2,vz_z2))),mul2(mul2(f2h2(2.0),lmu),vx_x2));
//    //Write updated values to global memory
//    sxx(gidz,gidx) = __f22h2(lsxx);
//    sxz(gidz,gidx) = __f22h2(lsxz);
//    szz(gidz,gidx) = __f22h2(lszz);
    muipkp(gidz,gidx) =(lsxx);
    mu(gidz,gidx) = (lsxz);
    M(gidz,gidx) = (lszz);
    
    
}
