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
///*Update of the velocity in 2D SV*/
//
////Define useful macros to be able to write a matrix formulation in 2D with OpenCl
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
//#define taus(z,x)        taus[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define tausipkp(z,x) tausipkp[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define taup(z,x)        taup[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//
//#define vx(z,x)  vx[(x)*(NZ)+(z)]
//#define vy(z,x)  vy[(x)*(NZ)+(z)]
//#define vz(z,x)  vz[(x)*(NZ)+(z)]
//#define sxx(z,x) sxx[(x)*(NZ)+(z)]
//#define szz(z,x) szz[(x)*(NZ)+(z)]
//#define sxz(z,x) sxz[(x)*(NZ)+(z)]
//#define sxy(z,x) sxy[(x)*(NZ)+(z)]
//#define syz(z,x) syz[(x)*(NZ)+(z)]
//
//#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
//#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
//#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]
//#define rxy(z,x,l) rxy[(l)*NX*NZ+(x)*NZ+(z)]
//#define ryz(z,x,l) ryz[(l)*NX*NZ+(x)*NZ+(z)]
//
//#define psi_sxx_x(z,x) psi_sxx_x[(x)*(NZ-FDOH/2)+(z)]
//#define psi_sxz_x(z,x) psi_sxz_x[(x)*(NZ-FDOH/2)+(z)]
//#define psi_sxz_z(z,x) psi_sxz_z[(x)*(NAB)+(z)]
//#define psi_szz_z(z,x) psi_szz_z[(x)*(NAB)+(z)]
//#define psi_sxy_x(z,x) psi_sxy_x[(x)*(NZ-FDOH/2)+(z)]
//#define psi_syz_z(z,x) psi_syz_z[(x)*(NAB)+(z)]
//
//#if LOCAL_OFF==0
//
//#define lvar2(z,x)  lvar2[(x)*lsizez+(z)]
//#define lvar(z,x)  lvar[(x)*lsizez*2+(z)]
//
//#endif
//
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
//
//extern "C" __device__ float2 add2(float2 a, float2 b ){
//
//    float2 output;
//    output.x = a.x+b.x;
//    output.y = a.y+b.y;
//    return output;
//}
//extern "C" __device__ float2 mul2(float2 a, float2 b ){
//
//    float2 output;
//    output.x = a.x*b.x;
//    output.y = a.y*b.y;
//    return output;
//}
//extern "C" __device__ float2 sub2(float2 a, float2 b ){
//
//    float2 output;
//    output.x = a.x-b.x;
//    output.y = a.y-b.y;
//    return output;
//}
//
//
//extern "C" __global__ void update_v(int offcomm,
//                                    __prec2 *vx,      __prec2 *vz,
//                                    __prec2 *sxx,     __prec2 *szz,     __prec2 *sxz,
//                                    float2 *rip,     float2 *rkp,
//                                    float *taper,
//                                    float *K_z,        float *a_z,          float *b_z,
//                                    float *K_z_half,   float *a_z_half,     float *b_z_half,
//                                    float *K_x,        float *a_x,          float *b_x,
//                                    float *K_x_half,   float *a_x_half,     float *b_x_half,
//                                    __prec2 *psi_sxx_x,  __prec2 *psi_sxz_x,
//                                    __prec2 *psi_sxz_z,  __prec2 *psi_szz_z,
//                                    int scaler_sxx)
//{
//
//    extern __shared__ __prec2 lvar2[];
//    __prec * lvar=(__prec *)lvar2;
//
//    float2 sxx_x;
//    float2 szz_z;
//    float2 sxz_x;
//    float2 sxz_z;
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
//#define lsxx2 lvar2
//#define lszz2 lvar2
//#define lsxz2 lvar2
//#define lsxx lvar
//#define lszz lvar
//#define lsxz lvar
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
//#define lsxx sxx
//#define lszz szz
//#define lsxz sxz
//#define lidx gidx
//#define lidz gidz
//
//#endif
//
//    // Calculation of the stresses spatial derivatives
//    {
//#if LOCAL_OFF==0
//        lsxx2(lidz,lidx)=sxx(gidz, gidx);
//        if (lidx<2*FDOH)
//            lsxx2(lidz,lidx-FDOH)=sxx(gidz,gidx-FDOH);
//        if (lidx+lsizex-3*FDOH<FDOH)
//            lsxx2(lidz,lidx+lsizex-3*FDOH)=sxx(gidz,gidx+lsizex-3*FDOH);
//        if (lidx>(lsizex-2*FDOH-1))
//            lsxx2(lidz,lidx+FDOH)=sxx(gidz,gidx+FDOH);
//        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
//            lsxx2(lidz,lidx-lsizex+3*FDOH)=sxx(gidz,gidx-lsizex+3*FDOH);
//
//        __syncthreads();
//#endif
//
//
//#if   FDOH ==1
//        sxx_x.x =  HC1*(__h2f(lsxx((2*lidz),lidx+1)) - (__h2f(lsxx((2*lidz),lidx))));
//        sxx_x.y =  HC1*(__h2f(lsxx((2*lidz+1),lidx+1)) - (__h2f(lsxx((2*lidz+1),lidx))));
//#elif FDOH ==2
//        sxx_x.x =  (HC1*(__h2f(lsxx((2*lidz),lidx+1)) - __h2f(lsxx((2*lidz),lidx)))
//                      +HC2*(__h2f(lsxx((2*lidz),lidx+2)) - __h2f(lsxx((2*lidz),lidx-1))));
//        sxx_x.y =  (HC1*(__h2f(lsxx((2*lidz+1),lidx+1)) - __h2f(lsxx((2*lidz+1),lidx)))
//                      +HC2*(__h2f(lsxx((2*lidz+1),lidx+2)) - __h2f(lsxx((2*lidz+1),lidx-1))));
//#elif FDOH ==3
//        sxx_x.x =  (HC1*(__h2f(lsxx((2*lidz),lidx+1))-__h2f(lsxx((2*lidz),lidx)))+
//                      HC2*(__h2f(lsxx((2*lidz),lidx+2))-__h2f(lsxx((2*lidz),lidx-1)))+
//                      HC3*(__h2f(lsxx((2*lidz),lidx+3))-__h2f(lsxx((2*lidz),lidx-2))));
//        sxx_x.y =  (HC1*(__h2f(lsxx((2*lidz+1),lidx+1))-__h2f(lsxx((2*lidz+1),lidx)))+
//                      HC2*(__h2f(lsxx((2*lidz+1),lidx+2))-__h2f(lsxx((2*lidz+1),lidx-1)))+
//                      HC3*(__h2f(lsxx((2*lidz+1),lidx+3))-__h2f(lsxx((2*lidz+1),lidx-2))));
//#elif FDOH ==4
//
//        sxx_x.x =  (HC1*(__h2f(lsxx((2*lidz),lidx+1))-__h2f(lsxx((2*lidz),lidx)))+
//                      HC2*(__h2f(lsxx((2*lidz),lidx+2))-__h2f(lsxx((2*lidz),lidx-1)))+
//                      HC3*(__h2f(lsxx((2*lidz),lidx+3))-__h2f(lsxx((2*lidz),lidx-2)))+
//                      HC4*(__h2f(lsxx((2*lidz),lidx+4))-__h2f(lsxx((2*lidz),lidx-3))));
//        sxx_x.y =  (HC1*(__h2f(lsxx((2*lidz+1),lidx+1))-__h2f(lsxx((2*lidz+1),lidx)))+
//                      HC2*(__h2f(lsxx((2*lidz+1),lidx+2))-__h2f(lsxx((2*lidz+1),lidx-1)))+
//                      HC3*(__h2f(lsxx((2*lidz+1),lidx+3))-__h2f(lsxx((2*lidz+1),lidx-2)))+
//                      HC4*(__h2f(lsxx((2*lidz+1),lidx+4))-__h2f(lsxx((2*lidz+1),lidx-3))));
//#elif FDOH ==5
//        sxx_x.x =  (HC1*(__h2f(lsxx((2*lidz),lidx+1))-__h2f(lsxx((2*lidz),lidx)))+
//                      HC2*(__h2f(lsxx((2*lidz),lidx+2))-__h2f(lsxx((2*lidz),lidx-1)))+
//                      HC3*(__h2f(lsxx((2*lidz),lidx+3))-__h2f(lsxx((2*lidz),lidx-2)))+
//                      HC4*(__h2f(lsxx((2*lidz),lidx+4))-__h2f(lsxx((2*lidz),lidx-3)))+
//                      HC5*(__h2f(lsxx((2*lidz),lidx+5))-__h2f(lsxx((2*lidz),lidx-4))));
//        sxx_x.y =  (HC1*(__h2f(lsxx((2*lidz+1),lidx+1))-__h2f(lsxx((2*lidz+1),lidx)))+
//                      HC2*(__h2f(lsxx((2*lidz+1),lidx+2))-__h2f(lsxx((2*lidz+1),lidx-1)))+
//                      HC3*(__h2f(lsxx((2*lidz+1),lidx+3))-__h2f(lsxx((2*lidz+1),lidx-2)))+
//                      HC4*(__h2f(lsxx((2*lidz+1),lidx+4))-__h2f(lsxx((2*lidz+1),lidx-3)))+
//                      HC5*(__h2f(lsxx((2*lidz+1),lidx+5))-__h2f(lsxx((2*lidz+1),lidx-4))));
//#elif FDOH ==6
//        sxx_x.x =  (HC1*(__h2f(lsxx((2*lidz),lidx+1))-__h2f(lsxx((2*lidz),lidx)))+
//                      HC2*(__h2f(lsxx((2*lidz),lidx+2))-__h2f(lsxx((2*lidz),lidx-1)))+
//                      HC3*(__h2f(lsxx((2*lidz),lidx+3))-__h2f(lsxx((2*lidz),lidx-2)))+
//                      HC4*(__h2f(lsxx((2*lidz),lidx+4))-__h2f(lsxx((2*lidz),lidx-3)))+
//                      HC5*(__h2f(lsxx((2*lidz),lidx+5))-__h2f(lsxx((2*lidz),lidx-4)))+
//                      HC6*(__h2f(lsxx((2*lidz),lidx+6))-__h2f(lsxx((2*lidz),lidx-5))));
//        sxx_x.y =  (HC1*(__h2f(lsxx((2*lidz+1),lidx+1))-__h2f(lsxx((2*lidz+1),lidx)))+
//                      HC2*(__h2f(lsxx((2*lidz+1),lidx+2))-__h2f(lsxx((2*lidz+1),lidx-1)))+
//                      HC3*(__h2f(lsxx((2*lidz+1),lidx+3))-__h2f(lsxx((2*lidz+1),lidx-2)))+
//                      HC4*(__h2f(lsxx((2*lidz+1),lidx+4))-__h2f(lsxx((2*lidz+1),lidx-3)))+
//                      HC5*(__h2f(lsxx((2*lidz+1),lidx+5))-__h2f(lsxx((2*lidz+1),lidx-4)))+
//                      HC6*(__h2f(lsxx((2*lidz+1),lidx+6))-__h2f(lsxx((2*lidz+1),lidx-5))));
//#endif
//
//
//#if LOCAL_OFF==0
//        __syncthreads();
//        lszz2(lidz,lidx)=szz(gidz, gidx);
//        if (lidz<FDOH)
//            lszz2(lidz-FDOH/2,lidx)=szz(gidz-FDOH/2,gidx);
//        if (lidz>(lsizez-FDOH-1))
//            lszz2(lidz+FDOH/2,lidx)=szz(gidz+FDOH/2,gidx);
//        __syncthreads();
//#endif
//
//#if   FDOH ==1
//        szz_z.x =  HC1*(__h2f(lszz((2*lidz)+1,lidx)) - __h2f(lszz((2*lidz),lidx)));
//        szz_z.y =  HC1*(__h2f(lszz((2*lidz+1)+1,lidx)) - __h2f(lszz((2*lidz+1),lidx)));
//#elif FDOH ==2
//        szz_z.x =  (HC1*(__h2f(lszz((2*lidz)+1,lidx)) - __h2f(lszz((2*lidz),lidx)))
//                      +HC2*(__h2f(lszz((2*lidz)+2,lidx)) - __h2f(lszz((2*lidz)-1,lidx))));
//        szz_z.y =  (HC1*(__h2f(lszz((2*lidz+1)+1,lidx)) - __h2f(lszz((2*lidz+1),lidx)))
//                      +HC2*(__h2f(lszz((2*lidz+1)+2,lidx)) - __h2f(lszz((2*lidz+1)-1,lidx))));
//#elif FDOH ==3
//        szz_z.x =  (HC1*(__h2f(lszz((2*lidz)+1,lidx))-__h2f(lszz((2*lidz),lidx)))+
//                      HC2*(__h2f(lszz((2*lidz)+2,lidx))-__h2f(lszz((2*lidz)-1,lidx)))+
//                      HC3*(__h2f(lszz((2*lidz)+3,lidx))-__h2f(lszz((2*lidz)-2,lidx))));
//        szz_z.y =  (HC1*(__h2f(lszz((2*lidz+1)+1,lidx))-__h2f(lszz((2*lidz+1),lidx)))+
//                      HC2*(__h2f(lszz((2*lidz+1)+2,lidx))-__h2f(lszz((2*lidz+1)-1,lidx)))+
//                      HC3*(__h2f(lszz((2*lidz+1)+3,lidx))-__h2f(lszz((2*lidz+1)-2,lidx))));
//#elif FDOH ==4
//        szz_z.x =  (HC1*(__h2f(lszz((2*lidz)+1,lidx))-__h2f(lszz((2*lidz),lidx)))+
//                      HC2*(__h2f(lszz((2*lidz)+2,lidx))-__h2f(lszz((2*lidz)-1,lidx)))+
//                      HC3*(__h2f(lszz((2*lidz)+3,lidx))-__h2f(lszz((2*lidz)-2,lidx)))+
//                      HC4*(__h2f(lszz((2*lidz)+4,lidx))-__h2f(lszz((2*lidz)-3,lidx))));
//        szz_z.y =  (HC1*(__h2f(lszz((2*lidz+1)+1,lidx))-__h2f(lszz((2*lidz+1),lidx)))+
//                      HC2*(__h2f(lszz((2*lidz+1)+2,lidx))-__h2f(lszz((2*lidz+1)-1,lidx)))+
//                      HC3*(__h2f(lszz((2*lidz+1)+3,lidx))-__h2f(lszz((2*lidz+1)-2,lidx)))+
//                      HC4*(__h2f(lszz((2*lidz+1)+4,lidx))-__h2f(lszz((2*lidz+1)-3,lidx))));
//#elif FDOH ==5
//        szz_z.x =  (HC1*(__h2f(lszz((2*lidz)+1,lidx))-__h2f(lszz((2*lidz),lidx)))+
//                      HC2*(__h2f(lszz((2*lidz)+2,lidx))-__h2f(lszz((2*lidz)-1,lidx)))+
//                      HC3*(__h2f(lszz((2*lidz)+3,lidx))-__h2f(lszz((2*lidz)-2,lidx)))+
//                      HC4*(__h2f(lszz((2*lidz)+4,lidx))-__h2f(lszz((2*lidz)-3,lidx)))+
//                      HC5*(__h2f(lszz((2*lidz)+5,lidx))-__h2f(lszz((2*lidz)-4,lidx))));
//        szz_z.y =  (HC1*(__h2f(lszz((2*lidz+1)+1,lidx))-__h2f(lszz((2*lidz+1),lidx)))+
//                      HC2*(__h2f(lszz((2*lidz+1)+2,lidx))-__h2f(lszz((2*lidz+1)-1,lidx)))+
//                      HC3*(__h2f(lszz((2*lidz+1)+3,lidx))-__h2f(lszz((2*lidz+1)-2,lidx)))+
//                      HC4*(__h2f(lszz((2*lidz+1)+4,lidx))-__h2f(lszz((2*lidz+1)-3,lidx)))+
//                      HC5*(__h2f(lszz((2*lidz+1)+5,lidx))-__h2f(lszz((2*lidz+1)-4,lidx))));
//#elif FDOH ==6
//        szz_z.x =  (HC1*(__h2f(lszz((2*lidz)+1,lidx))-__h2f(lszz((2*lidz),lidx)))+
//                      HC2*(__h2f(lszz((2*lidz)+2,lidx))-__h2f(lszz((2*lidz)-1,lidx)))+
//                      HC3*(__h2f(lszz((2*lidz)+3,lidx))-__h2f(lszz((2*lidz)-2,lidx)))+
//                      HC4*(__h2f(lszz((2*lidz)+4,lidx))-__h2f(lszz((2*lidz)-3,lidx)))+
//                      HC5*(__h2f(lszz((2*lidz)+5,lidx))-__h2f(lszz((2*lidz)-4,lidx)))+
//                      HC6*(__h2f(lszz((2*lidz)+6,lidx))-__h2f(lszz((2*lidz)-5,lidx))));
//        szz_z.y =  (HC1*(__h2f(lszz((2*lidz+1)+1,lidx))-__h2f(lszz((2*lidz+1),lidx)))+
//                      HC2*(__h2f(lszz((2*lidz+1)+2,lidx))-__h2f(lszz((2*lidz+1)-1,lidx)))+
//                      HC3*(__h2f(lszz((2*lidz+1)+3,lidx))-__h2f(lszz((2*lidz+1)-2,lidx)))+
//                      HC4*(__h2f(lszz((2*lidz+1)+4,lidx))-__h2f(lszz((2*lidz+1)-3,lidx)))+
//                      HC5*(__h2f(lszz((2*lidz+1)+5,lidx))-__h2f(lszz((2*lidz+1)-4,lidx)))+
//                      HC6*(__h2f(lszz((2*lidz+1)+6,lidx))-__h2f(lszz((2*lidz+1)-5,lidx))));
//#endif
//
//#if LOCAL_OFF==0
//        __syncthreads();
//        lsxz2(lidz,lidx)=sxz(gidz, gidx);
//
//        if (lidx<2*FDOH)
//            lsxz2(lidz,lidx-FDOH)=sxz(gidz,gidx-FDOH);
//        if (lidx+lsizex-3*FDOH<FDOH)
//            lsxz2(lidz,lidx+lsizex-3*FDOH)=sxz(gidz,gidx+lsizex-3*FDOH);
//        if (lidx>(lsizex-2*FDOH-1))
//            lsxz2(lidz,lidx+FDOH)=sxz(gidz,gidx+FDOH);
//        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
//            lsxz2(lidz,lidx-lsizex+3*FDOH)=sxz(gidz,gidx-lsizex+3*FDOH);
//        if (lidz<FDOH)
//            lsxz2(lidz-FDOH/2,lidx)=sxz(gidz-FDOH/2,gidx);
//        if (lidz>(lsizez-FDOH-1))
//            lsxz2(lidz+FDOH/2,lidx)=sxz(gidz+FDOH/2,gidx);
//        __syncthreads();
//#endif
//
//#if   FDOH ==1
//        sxz_z.x =  HC1*(__h2f(lsxz((2*lidz),lidx))   - __h2f(lsxz((2*lidz)-1,lidx)));
//        sxz_z.y =  HC1*(__h2f(lsxz((2*lidz+1),lidx))   - __h2f(lsxz((2*lidz+1)-1,lidx)));
//        sxz_x.x =  HC1*(__h2f(lsxz((2*lidz),lidx))   - __h2f(lsxz((2*lidz),lidx-1)));
//        sxz_x.y =  HC1*(__h2f(lsxz((2*lidz+1),lidx))   - __h2f(lsxz((2*lidz+1),lidx-1)));
//#elif FDOH ==2
//        sxz_z.x =  (HC1*(__h2f(lsxz((2*lidz),lidx))   - __h2f(lsxz((2*lidz)-1,lidx)))
//                      +HC2*(__h2f(lsxz((2*lidz)+1,lidx)) - __h2f(lsxz((2*lidz)-2,lidx))));
//        sxz_z.y =  (HC1*(__h2f(lsxz((2*lidz+1),lidx))   - __h2f(lsxz((2*lidz+1)-1,lidx)))
//                      +HC2*(__h2f(lsxz((2*lidz+1)+1,lidx)) - __h2f(lsxz((2*lidz+1)-2,lidx))));
//        sxz_x.x =  (HC1*(__h2f(lsxz((2*lidz),lidx))   - __h2f(lsxz((2*lidz),lidx-1)))
//                      +HC2*(__h2f(lsxz((2*lidz),lidx+1)) - __h2f(lsxz((2*lidz),lidx-2))));
//        sxz_x.y =  (HC1*(__h2f(lsxz((2*lidz+1),lidx))   - __h2f(lsxz((2*lidz+1),lidx-1)))
//                      +HC2*(__h2f(lsxz((2*lidz+1),lidx+1)) - __h2f(lsxz((2*lidz+1),lidx-2))));
//#elif FDOH ==3
//        sxz_z.x =  (HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz)-1,lidx)))+
//                      HC2*(__h2f(lsxz((2*lidz)+1,lidx))-__h2f(lsxz((2*lidz)-2,lidx)))+
//                      HC3*(__h2f(lsxz((2*lidz)+2,lidx))-__h2f(lsxz((2*lidz)-3,lidx))));
//        sxz_z.y =  (HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1)-1,lidx)))+
//                      HC2*(__h2f(lsxz((2*lidz+1)+1,lidx))-__h2f(lsxz((2*lidz+1)-2,lidx)))+
//                      HC3*(__h2f(lsxz((2*lidz+1)+2,lidx))-__h2f(lsxz((2*lidz+1)-3,lidx))));
//
//        sxz_x.x =  (HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz),lidx-1)))+
//                      HC2*(__h2f(lsxz((2*lidz),lidx+1))-__h2f(lsxz((2*lidz),lidx-2)))+
//                      HC3*(__h2f(lsxz((2*lidz),lidx+2))-__h2f(lsxz((2*lidz),lidx-3))));
//        sxz_x.y =  (HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1),lidx-1)))+
//                      HC2*(__h2f(lsxz((2*lidz+1),lidx+1))-__h2f(lsxz((2*lidz+1),lidx-2)))+
//                      HC3*(__h2f(lsxz((2*lidz+1),lidx+2))-__h2f(lsxz((2*lidz+1),lidx-3))));
//#elif FDOH ==4
//        sxz_z.x =  (HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz)-1,lidx)))+
//                      HC2*(__h2f(lsxz((2*lidz)+1,lidx))-__h2f(lsxz((2*lidz)-2,lidx)))+
//                      HC3*(__h2f(lsxz((2*lidz)+2,lidx))-__h2f(lsxz((2*lidz)-3,lidx)))+
//                      HC4*(__h2f(lsxz((2*lidz)+3,lidx))-__h2f(lsxz((2*lidz)-4,lidx))));
//        sxz_z.y =  (HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1)-1,lidx)))+
//                      HC2*(__h2f(lsxz((2*lidz+1)+1,lidx))-__h2f(lsxz((2*lidz+1)-2,lidx)))+
//                      HC3*(__h2f(lsxz((2*lidz+1)+2,lidx))-__h2f(lsxz((2*lidz+1)-3,lidx)))+
//                      HC4*(__h2f(lsxz((2*lidz+1)+3,lidx))-__h2f(lsxz((2*lidz+1)-4,lidx))));
//
//        sxz_x.x =  (HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz),lidx-1)))+
//                      HC2*(__h2f(lsxz((2*lidz),lidx+1))-__h2f(lsxz((2*lidz),lidx-2)))+
//                      HC3*(__h2f(lsxz((2*lidz),lidx+2))-__h2f(lsxz((2*lidz),lidx-3)))+
//                      HC4*(__h2f(lsxz((2*lidz),lidx+3))-__h2f(lsxz((2*lidz),lidx-4))));
//        sxz_x.y =  (HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1),lidx-1)))+
//                      HC2*(__h2f(lsxz((2*lidz+1),lidx+1))-__h2f(lsxz((2*lidz+1),lidx-2)))+
//                      HC3*(__h2f(lsxz((2*lidz+1),lidx+2))-__h2f(lsxz((2*lidz+1),lidx-3)))+
//                      HC4*(__h2f(lsxz((2*lidz+1),lidx+3))-__h2f(lsxz((2*lidz+1),lidx-4))));
//#elif FDOH ==5
//        sxz_z.x =  (HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz)-1,lidx)))+
//                      HC2*(__h2f(lsxz((2*lidz)+1,lidx))-__h2f(lsxz((2*lidz)-2,lidx)))+
//                      HC3*(__h2f(lsxz((2*lidz)+2,lidx))-__h2f(lsxz((2*lidz)-3,lidx)))+
//                      HC4*(__h2f(lsxz((2*lidz)+3,lidx))-__h2f(lsxz((2*lidz)-4,lidx)))+
//                      HC5*(__h2f(lsxz((2*lidz)+4,lidx))-__h2f(lsxz((2*lidz)-5,lidx))));
//        sxz_z.y =  (HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1)-1,lidx)))+
//                      HC2*(__h2f(lsxz((2*lidz+1)+1,lidx))-__h2f(lsxz((2*lidz+1)-2,lidx)))+
//                      HC3*(__h2f(lsxz((2*lidz+1)+2,lidx))-__h2f(lsxz((2*lidz+1)-3,lidx)))+
//                      HC4*(__h2f(lsxz((2*lidz+1)+3,lidx))-__h2f(lsxz((2*lidz+1)-4,lidx)))+
//                      HC5*(__h2f(lsxz((2*lidz+1)+4,lidx))-__h2f(lsxz((2*lidz+1)-5,lidx))));
//
//        sxz_x.x =  (HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz),lidx-1)))+
//                      HC2*(__h2f(lsxz((2*lidz),lidx+1))-__h2f(lsxz((2*lidz),lidx-2)))+
//                      HC3*(__h2f(lsxz((2*lidz),lidx+2))-__h2f(lsxz((2*lidz),lidx-3)))+
//                      HC4*(__h2f(lsxz((2*lidz),lidx+3))-__h2f(lsxz((2*lidz),lidx-4)))+
//                      HC5*(__h2f(lsxz((2*lidz),lidx+4))-__h2f(lsxz((2*lidz),lidx-5))));
//        sxz_x.y =  (HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1),lidx-1)))+
//                      HC2*(__h2f(lsxz((2*lidz+1),lidx+1))-__h2f(lsxz((2*lidz+1),lidx-2)))+
//                      HC3*(__h2f(lsxz((2*lidz+1),lidx+2))-__h2f(lsxz((2*lidz+1),lidx-3)))+
//                      HC4*(__h2f(lsxz((2*lidz+1),lidx+3))-__h2f(lsxz((2*lidz+1),lidx-4)))+
//                      HC5*(__h2f(lsxz((2*lidz+1),lidx+4))-__h2f(lsxz((2*lidz+1),lidx-5))));
//#elif FDOH ==6
//
//        sxz_z.x =  (HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz)-1,lidx)))+
//                      HC2*(__h2f(lsxz((2*lidz)+1,lidx))-__h2f(lsxz((2*lidz)-2,lidx)))+
//                      HC3*(__h2f(lsxz((2*lidz)+2,lidx))-__h2f(lsxz((2*lidz)-3,lidx)))+
//                      HC4*(__h2f(lsxz((2*lidz)+3,lidx))-__h2f(lsxz((2*lidz)-4,lidx)))+
//                      HC5*(__h2f(lsxz((2*lidz)+4,lidx))-__h2f(lsxz((2*lidz)-5,lidx)))+
//                      HC6*(__h2f(lsxz((2*lidz)+5,lidx))-__h2f(lsxz((2*lidz)-6,lidx))));
//        sxz_z.y =  (HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1)-1,lidx)))+
//                      HC2*(__h2f(lsxz((2*lidz+1)+1,lidx))-__h2f(lsxz((2*lidz+1)-2,lidx)))+
//                      HC3*(__h2f(lsxz((2*lidz+1)+2,lidx))-__h2f(lsxz((2*lidz+1)-3,lidx)))+
//                      HC4*(__h2f(lsxz((2*lidz+1)+3,lidx))-__h2f(lsxz((2*lidz+1)-4,lidx)))+
//                      HC5*(__h2f(lsxz((2*lidz+1)+4,lidx))-__h2f(lsxz((2*lidz+1)-5,lidx)))+
//                      HC6*(__h2f(lsxz((2*lidz+1)+5,lidx))-__h2f(lsxz((2*lidz+1)-6,lidx))));
//
//        sxz_x.x =  (HC1*(__h2f(lsxz((2*lidz),lidx))  -__h2f(lsxz((2*lidz),lidx-1)))+
//                      HC2*(__h2f(lsxz((2*lidz),lidx+1))-__h2f(lsxz((2*lidz),lidx-2)))+
//                      HC3*(__h2f(lsxz((2*lidz),lidx+2))-__h2f(lsxz((2*lidz),lidx-3)))+
//                      HC4*(__h2f(lsxz((2*lidz),lidx+3))-__h2f(lsxz((2*lidz),lidx-4)))+
//                      HC5*(__h2f(lsxz((2*lidz),lidx+4))-__h2f(lsxz((2*lidz),lidx-5)))+
//                      HC6*(__h2f(lsxz((2*lidz),lidx+5))-__h2f(lsxz((2*lidz),lidx-6))));
//        sxz_x.y =  (HC1*(__h2f(lsxz((2*lidz+1),lidx))  -__h2f(lsxz((2*lidz+1),lidx-1)))+
//                      HC2*(__h2f(lsxz((2*lidz+1),lidx+1))-__h2f(lsxz((2*lidz+1),lidx-2)))+
//                      HC3*(__h2f(lsxz((2*lidz+1),lidx+2))-__h2f(lsxz((2*lidz+1),lidx-3)))+
//                      HC4*(__h2f(lsxz((2*lidz+1),lidx+3))-__h2f(lsxz((2*lidz+1),lidx-4)))+
//                      HC5*(__h2f(lsxz((2*lidz+1),lidx+4))-__h2f(lsxz((2*lidz+1),lidx-5)))+
//                      HC6*(__h2f(lsxz((2*lidz+1),lidx+5))-__h2f(lsxz((2*lidz+1),lidx-6))));
//#endif
//    }
//
//    // To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
//#if LOCAL_OFF==0
//#if COMM12==0
//    if (gidz>(NZ-FDOH/2-1) || (gidx-offcomm)>(NX-FDOH-1-LCOMM) ){
//        return;
//    }
//
//#else
//    if (gidz>(NZ-FDOH/2-1) ){
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
//        float2 lpsi_sxz_z, lpsi_szz_z, lpsi_sxx_x, lpsi_sxz_x;
//
//        if (gidz>NZ-NAB/2-FDOH/2-1){
//
//            i =gidx-FDOH;
//            k =gidz - NZ+NAB/2+FDOH/2+NAB/2;
//            ind=2*NAB-1-2*k;
//
//            lpsi_sxz_z = __h22f2(psi_sxz_z(k,i));
//            lpsi_szz_z = __h22f2(psi_szz_z(k,i));
//
//            lpsi_sxz_z.x = b_z[ind+1] * lpsi_sxz_z.x + a_z[ind+1] * sxz_z.x;
//            lpsi_sxz_z.y = b_z[ind  ] * lpsi_sxz_z.y + a_z[ind  ] * sxz_z.y;
//            sxz_z.x = sxz_z.x / K_z[ind+1] + lpsi_sxz_z.x;
//            sxz_z.y = sxz_z.y / K_z[ind  ] + lpsi_sxz_z.y;
//
//            lpsi_szz_z.x = b_z_half[ind  ] * lpsi_szz_z.x + a_z_half[ind  ] * szz_z.x;
//            lpsi_szz_z.y = b_z_half[ind-1] * lpsi_szz_z.y + a_z_half[ind-1] * szz_z.y;
//            szz_z.x = szz_z.x / K_z_half[ind  ] + lpsi_szz_z.x;
//            szz_z.y = szz_z.y / K_z_half[ind-1] + lpsi_szz_z.y;
//
//            psi_sxz_z(k,i)=__f22h2(lpsi_sxz_z);
//            psi_szz_z(k,i)=__f22h2(lpsi_szz_z);
//
//        }
//
//#if FREESURF==0
//        else if (gidz-FDOH/2<NAB/2){
//
//            i =gidx-FDOH;
//            k =gidz-FDOH/2;
//
//            lpsi_sxz_z = __h22f2(psi_sxz_z(k,i));
//            lpsi_szz_z = __h22f2(psi_szz_z(k,i));
//
//            lpsi_sxz_z.x = b_z[2*k  ] * lpsi_sxz_z.x + a_z[2*k  ] * sxz_z.x;
//            lpsi_sxz_z.y = b_z[2*k+1] * lpsi_sxz_z.y + a_z[2*k+1] * sxz_z.y;
//            sxz_z.x = sxz_z.x / K_z[2*k  ] + lpsi_sxz_z.x;
//            sxz_z.y = sxz_z.y / K_z[2*k+1] + lpsi_sxz_z.y;
//
//            lpsi_szz_z.x = b_z_half[2*k  ] * lpsi_szz_z.x + a_z_half[2*k  ] * szz_z.x;
//            lpsi_szz_z.y = b_z_half[2*k+1] * lpsi_szz_z.y + a_z_half[2*k+1] * szz_z.y;
//            szz_z.x = szz_z.x / K_z_half[2*k  ] + lpsi_szz_z.x;
//            szz_z.y = szz_z.y / K_z_half[2*k+1] + lpsi_szz_z.y;
//
//            psi_sxz_z(k,i)=__f22h2(lpsi_sxz_z);
//            psi_szz_z(k,i)=__f22h2(lpsi_szz_z);
//
//        }
//#endif
//
//#if DEVID==0 & MYLOCALID==0
//        if (gidx-FDOH<NAB){
//
//            i =gidx-FDOH;
//            k =gidz-FDOH/2;
//
//            lpsi_sxx_x = __h22f2(psi_sxx_x(k,i));
//            lpsi_sxz_x = __h22f2(psi_sxz_x(k,i));
//
//            lpsi_sxx_x.x = b_x_half[i] * lpsi_sxx_x.x + a_x_half[i] * sxx_x.x;
//            lpsi_sxx_x.y = b_x_half[i] * lpsi_sxx_x.y + a_x_half[i] * sxx_x.y;
//            sxx_x.x = sxx_x.x / K_x_half[i] + lpsi_sxx_x.x;
//            sxx_x.y = sxx_x.y / K_x_half[i] + lpsi_sxx_x.y;
//
//            lpsi_sxz_x.x = b_x[i] * lpsi_sxz_x.x + a_x[i] * sxz_x.x;
//            lpsi_sxz_x.y = b_x[i] * lpsi_sxz_x.y + a_x[i] * sxz_x.y;
//            sxz_x.x = sxz_x.x / K_x[i] + lpsi_sxz_x.x;
//            sxz_x.y = sxz_x.y / K_x[i] + lpsi_sxz_x.y;
//
//            psi_sxx_x(k,i)=__f22h2(lpsi_sxx_x);
//            psi_sxz_x(k,i)=__f22h2(lpsi_sxz_x);
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
//            lpsi_sxx_x = __h22f2(psi_sxx_x(k,i));
//            lpsi_sxz_x = __h22f2(psi_sxz_x(k,i));
//
//            lpsi_sxx_x.x = b_x_half[ind] * lpsi_sxx_x.x + a_x_half[ind] * sxx_x.x;
//            lpsi_sxx_x.y = b_x_half[ind] * lpsi_sxx_x.y + a_x_half[ind] * sxx_x.y;
//            sxx_x.x = sxx_x.x / K_x_half[ind] + lpsi_sxx_x.x;
//            sxx_x.y = sxx_x.y / K_x_half[ind] + lpsi_sxx_x.y;
//
//            lpsi_sxz_x.x = b_x[ind+1] * lpsi_sxz_x.x + a_x[ind+1] * sxz_x.x;
//            lpsi_sxz_x.y = b_x[ind+1] * lpsi_sxz_x.y + a_x[ind+1] * sxz_x.y;
//            sxz_x.x = sxz_x.x / K_x[ind+1] + lpsi_sxz_x.x;
//            sxz_x.y = sxz_x.y / K_x[ind+1] + lpsi_sxz_x.y;
//
//            psi_sxx_x(k,i)=__f22h2(lpsi_sxx_x);
//            psi_sxz_x(k,i)=__f22h2(lpsi_sxz_x);
//
//        }
//#endif
//    }
//#endif
//
//    // Update the velocities
//    {
//        float2 lvx = __h22f2(vx(gidz,gidx));
//        float2 lvz = __h22f2(vz(gidz,gidx));
//        float2 lrip = (rip(gidz,gidx));
//        float2 lrkp = (rkp(gidz,gidx));
//
//        lvx.x += (sxx_x.x + sxz_z.x)*lrip.x;
//        lvx.y += (sxx_x.y + sxz_z.y)*lrip.y;
//        lvz.x += (szz_z.x + sxz_x.x)*lrkp.x;
//        lvz.y += (szz_z.y + sxz_x.y)*lrkp.y;
//
//
//        // Absorbing boundary
//#if ABS_TYPE==2
//        {
//            if (2*gidz-FDOH<NAB){
//                lvx.x*=taper[2*gidz-FDOH];
//                lvx.y*=taper[2*gidz+1-FDOH];
//                lvz.x*=taper[2*gidz-FDOH];
//                lvz.y*=taper[2*gidz+1-FDOH];
//            }
//
//            if (2*gidz>2*NZ-NAB-FDOH-1){
//                lvx.x*=taper[2*NZ-FDOH-2*gidz-1];
//                lvx.y*=taper[2*NZ-FDOH-2*gidz-1-1];
//                lvz.x*=taper[2*NZ-FDOH-2*gidz-1];
//                lvz.y*=taper[2*NZ-FDOH-2*gidz-1-1];
//            }
//
//#if DEVID==0 & MYLOCALID==0
//            if (gidx-FDOH<NAB){
//                lvx.x*=taper[gidx-FDOH];
//                lvx.y*=taper[gidx-FDOH];
//                lvz.x*=taper[gidx-FDOH];
//                lvz.y*=taper[gidx-FDOH];
//            }
//#endif
//
//#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//            if (gidx>NX-NAB-FDOH-1){
//                lvx.x*=taper[NX-FDOH-gidx-1];
//                lvx.y*=taper[NX-FDOH-gidx-1];
//                lvz.x*=taper[NX-FDOH-gidx-1];
//                lvz.y*=taper[NX-FDOH-gidx-1];
//            }
//#endif
//        }
//#endif
//
//        vx(gidz,gidx)= __f22h2(lvx);
//        vz(gidz,gidx)= __f22h2(lvz);
//    }
//
//}

//Define useful macros to be able to write a matrix formulation in 2D with OpenCl
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



#if FP16==1 || FP16==2

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


#if FP16<3

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

#if FP16==2 || FP16==4

#define __pprec half2

#else

#define __pprec float2

#endif

#if FP16==2

#define __pconv(x) __half22float2((x))

#elif FP==3

#define __pconv(x) __float22half2_rn((x))

#else

#define __pconv(x) (x)

#endif


extern "C" __global__ void update_v(int offcomm,
                                    __pprec *rip, __pprec *rkp,__prec2 *sxx,__prec2 *sxz,__prec2 *szz,
                                    __prec2 *vx,__prec2 *vz
                                    )

{
    //Local memory
    extern __shared__ __prec2 lvar2[];
    __prec * lvar=(__prec *)lvar2;
    
    //Grid position
    int lsizez = blockDim.x+FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH/2;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/2;
    int gidx = blockIdx.y*blockDim.y+threadIdx.y+FDOH+offcomm;
    
    //Define and load private parameters and variables
    __cprec lvx = __h22f2(vx(gidz,gidx));
    __cprec lvz = __h22f2(vz(gidz,gidx));
    __pprec lrip = __pconv(rip(gidz,gidx));
    __pprec lrkp = __pconv(rkp(gidz,gidx));
    
    //Define private derivatives
    __cprec sxx_x1;
    __cprec sxz_x2;
    __cprec sxz_z2;
    __cprec szz_z1;
    
    //Local memory definitions if local is used
#if LOCAL_OFF==0
#define lsxx lvar
#define lszz lvar
#define lsxz lvar
#define lsxx2 lvar2
#define lszz2 lvar2
#define lsxz2 lvar2
    
    //Local memory definitions if local is not used
#elif LOCAL_OFF==1
    
#define lsxx sxx
#define lszz szz
#define lsxz sxz
#define lidz gidz
#define lidx gidx
    
#endif
    
    //Calculation of the spatial derivatives
    {
#if LOCAL_OFF==0
        lsxx2(lidz,lidx)=sxx(gidz,gidx);
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
        
#if   FDOH == 1
        sxx_x1=mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidx+1)), __h22f2(lsxx2(lidz,lidx))));
#elif FDOH == 2
        sxx_x1=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidx+1)), __h22f2(lsxx2(lidz,lidx)))),
                    mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidx+2)), __h22f2(lsxx2(lidz,lidx-1)))));
#elif FDOH == 3
        sxx_x1=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidx+1)), __h22f2(lsxx2(lidz,lidx)))),
                         mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidx+2)), __h22f2(lsxx2(lidz,lidx-1))))),
                    mul2( f2h2(HC3), sub2(__h22f2(lsxx2(lidz,lidx+3)), __h22f2(lsxx2(lidz,lidx-2)))));
#elif FDOH == 4
        sxx_x1=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidx+1)), __h22f2(lsxx2(lidz,lidx)))),
                              mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidx+2)), __h22f2(lsxx2(lidz,lidx-1))))),
                         mul2( f2h2(HC3), sub2(__h22f2(lsxx2(lidz,lidx+3)), __h22f2(lsxx2(lidz,lidx-2))))),
                    mul2( f2h2(HC4), sub2(__h22f2(lsxx2(lidz,lidx+4)), __h22f2(lsxx2(lidz,lidx-3)))));
#elif FDOH == 5
        sxx_x1=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidx+1)), __h22f2(lsxx2(lidz,lidx)))),
                                   mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidx+2)), __h22f2(lsxx2(lidz,lidx-1))))),
                              mul2( f2h2(HC3), sub2(__h22f2(lsxx2(lidz,lidx+3)), __h22f2(lsxx2(lidz,lidx-2))))),
                         mul2( f2h2(HC4), sub2(__h22f2(lsxx2(lidz,lidx+4)), __h22f2(lsxx2(lidz,lidx-3))))),
                    mul2( f2h2(HC5), sub2(__h22f2(lsxx2(lidz,lidx+5)), __h22f2(lsxx2(lidz,lidx-4)))));
#elif FDOH == 6
        sxx_x1=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidx+1)), __h22f2(lsxx2(lidz,lidx)))),
                                        mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidx+2)), __h22f2(lsxx2(lidz,lidx-1))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(lsxx2(lidz,lidx+3)), __h22f2(lsxx2(lidz,lidx-2))))),
                              mul2( f2h2(HC4), sub2(__h22f2(lsxx2(lidz,lidx+4)), __h22f2(lsxx2(lidz,lidx-3))))),
                         mul2( f2h2(HC5), sub2(__h22f2(lsxx2(lidz,lidx+5)), __h22f2(lsxx2(lidz,lidx-4))))),
                    mul2( f2h2(HC6), sub2(__h22f2(lsxx2(lidz,lidx+6)), __h22f2(lsxx2(lidz,lidx-5)))));
#endif
        
#if LOCAL_OFF==0
        __syncthreads();
        lszz2(lidz,lidx)=szz(gidz,gidx);
        if (lidz<FDOH)
            lszz2(lidz-FDOH/2,lidx)=szz(gidz-FDOH/2,gidx);
        if (lidz>(lsizez-FDOH-1))
            lszz2(lidz+FDOH/2,lidx)=szz(gidz+FDOH/2,gidx);
        __syncthreads();
#endif
        
#if   FDOH == 1
        szz_z1=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidx))), __h22f2(__hp(&lszz(2*lidz,lidx)))));
#elif FDOH == 2
        szz_z1=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidx))), __h22f2(__hp(&lszz(2*lidz,lidx))))),
                    mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidx))))));
#elif FDOH == 3
        szz_z1=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidx))), __h22f2(__hp(&lszz(2*lidz,lidx))))),
                         mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidx)))))),
                    mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszz(2*lidz+3,lidx))), __h22f2(__hp(&lszz(2*lidz-2,lidx))))));
#elif FDOH == 4
        szz_z1=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidx))), __h22f2(__hp(&lszz(2*lidz,lidx))))),
                              mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidx)))))),
                         mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszz(2*lidz+3,lidx))), __h22f2(__hp(&lszz(2*lidz-2,lidx)))))),
                    mul2( f2h2(HC4), sub2(__h22f2(__hp(&lszz(2*lidz+4,lidx))), __h22f2(__hp(&lszz(2*lidz-3,lidx))))));
#elif FDOH == 5
        szz_z1=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidx))), __h22f2(__hp(&lszz(2*lidz,lidx))))),
                                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidx)))))),
                              mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszz(2*lidz+3,lidx))), __h22f2(__hp(&lszz(2*lidz-2,lidx)))))),
                         mul2( f2h2(HC4), sub2(__h22f2(__hp(&lszz(2*lidz+4,lidx))), __h22f2(__hp(&lszz(2*lidz-3,lidx)))))),
                    mul2( f2h2(HC5), sub2(__h22f2(__hp(&lszz(2*lidz+5,lidx))), __h22f2(__hp(&lszz(2*lidz-4,lidx))))));
#elif FDOH == 6
        szz_z1=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidx))), __h22f2(__hp(&lszz(2*lidz,lidx))))),
                                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidx)))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszz(2*lidz+3,lidx))), __h22f2(__hp(&lszz(2*lidz-2,lidx)))))),
                              mul2( f2h2(HC4), sub2(__h22f2(__hp(&lszz(2*lidz+4,lidx))), __h22f2(__hp(&lszz(2*lidz-3,lidx)))))),
                         mul2( f2h2(HC5), sub2(__h22f2(__hp(&lszz(2*lidz+5,lidx))), __h22f2(__hp(&lszz(2*lidz-4,lidx)))))),
                    mul2( f2h2(HC6), sub2(__h22f2(__hp(&lszz(2*lidz+6,lidx))), __h22f2(__hp(&lszz(2*lidz-5,lidx))))));
#endif
        
#if LOCAL_OFF==0
        __syncthreads();
        lsxz2(lidz,lidx)=sxz(gidz,gidx);
        if (lidz<FDOH)
            lsxz2(lidz-FDOH/2,lidx)=sxz(gidz-FDOH/2,gidx);
        if (lidz>(lsizez-FDOH-1))
            lsxz2(lidz+FDOH/2,lidx)=sxz(gidz+FDOH/2,gidx);
        if (lidx<2*FDOH)
            lsxz2(lidz,lidx-FDOH)=sxz(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxz2(lidz,lidx+lsizex-3*FDOH)=sxz(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxz2(lidz,lidx+FDOH)=sxz(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxz2(lidz,lidx-lsizex+3*FDOH)=sxz(gidz,gidx-lsizex+3*FDOH);
        __syncthreads();
#endif
        
#if   FDOH == 1
        sxz_x2=mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidx)), __h22f2(lsxz2(lidz,lidx-1))));
#elif FDOH == 2
        sxz_x2=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidx)), __h22f2(lsxz2(lidz,lidx-1)))),
                    mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidx+1)), __h22f2(lsxz2(lidz,lidx-2)))));
#elif FDOH == 3
        sxz_x2=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidx)), __h22f2(lsxz2(lidz,lidx-1)))),
                         mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidx+1)), __h22f2(lsxz2(lidz,lidx-2))))),
                    mul2( f2h2(HC3), sub2(__h22f2(lsxz2(lidz,lidx+2)), __h22f2(lsxz2(lidz,lidx-3)))));
#elif FDOH == 4
        sxz_x2=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidx)), __h22f2(lsxz2(lidz,lidx-1)))),
                              mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidx+1)), __h22f2(lsxz2(lidz,lidx-2))))),
                         mul2( f2h2(HC3), sub2(__h22f2(lsxz2(lidz,lidx+2)), __h22f2(lsxz2(lidz,lidx-3))))),
                    mul2( f2h2(HC4), sub2(__h22f2(lsxz2(lidz,lidx+3)), __h22f2(lsxz2(lidz,lidx-4)))));
#elif FDOH == 5
        sxz_x2=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidx)), __h22f2(lsxz2(lidz,lidx-1)))),
                                   mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidx+1)), __h22f2(lsxz2(lidz,lidx-2))))),
                              mul2( f2h2(HC3), sub2(__h22f2(lsxz2(lidz,lidx+2)), __h22f2(lsxz2(lidz,lidx-3))))),
                         mul2( f2h2(HC4), sub2(__h22f2(lsxz2(lidz,lidx+3)), __h22f2(lsxz2(lidz,lidx-4))))),
                    mul2( f2h2(HC5), sub2(__h22f2(lsxz2(lidz,lidx+4)), __h22f2(lsxz2(lidz,lidx-5)))));
#elif FDOH == 6
        sxz_x2=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidx)), __h22f2(lsxz2(lidz,lidx-1)))),
                                        mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidx+1)), __h22f2(lsxz2(lidz,lidx-2))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(lsxz2(lidz,lidx+2)), __h22f2(lsxz2(lidz,lidx-3))))),
                              mul2( f2h2(HC4), sub2(__h22f2(lsxz2(lidz,lidx+3)), __h22f2(lsxz2(lidz,lidx-4))))),
                         mul2( f2h2(HC5), sub2(__h22f2(lsxz2(lidz,lidx+4)), __h22f2(lsxz2(lidz,lidx-5))))),
                    mul2( f2h2(HC6), sub2(__h22f2(lsxz2(lidz,lidx+5)), __h22f2(lsxz2(lidz,lidx-6)))));
#endif
        
#if   FDOH == 1
        sxz_z2=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidx)))));
#elif FDOH == 2
        sxz_z2=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidx))))),
                    mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidx))))));
#elif FDOH == 3
        sxz_z2=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidx))))),
                         mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidx)))))),
                    mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxz(2*lidz+2,lidx))), __h22f2(__hp(&lsxz(2*lidz-3,lidx))))));
#elif FDOH == 4
        sxz_z2=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidx))))),
                              mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidx)))))),
                         mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxz(2*lidz+2,lidx))), __h22f2(__hp(&lsxz(2*lidz-3,lidx)))))),
                    mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsxz(2*lidz+3,lidx))), __h22f2(__hp(&lsxz(2*lidz-4,lidx))))));
#elif FDOH == 5
        sxz_z2=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidx))))),
                                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidx)))))),
                              mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxz(2*lidz+2,lidx))), __h22f2(__hp(&lsxz(2*lidz-3,lidx)))))),
                         mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsxz(2*lidz+3,lidx))), __h22f2(__hp(&lsxz(2*lidz-4,lidx)))))),
                    mul2( f2h2(HC5), sub2(__h22f2(__hp(&lsxz(2*lidz+4,lidx))), __h22f2(__hp(&lsxz(2*lidz-5,lidx))))));
#elif FDOH == 6
        sxz_z2=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidx))))),
                                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidx)))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxz(2*lidz+2,lidx))), __h22f2(__hp(&lsxz(2*lidz-3,lidx)))))),
                              mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsxz(2*lidz+3,lidx))), __h22f2(__hp(&lsxz(2*lidz-4,lidx)))))),
                         mul2( f2h2(HC5), sub2(__h22f2(__hp(&lsxz(2*lidz+4,lidx))), __h22f2(__hp(&lsxz(2*lidz-5,lidx)))))),
                    mul2( f2h2(HC6), sub2(__h22f2(__hp(&lsxz(2*lidz+5,lidx))), __h22f2(__hp(&lsxz(2*lidz-6,lidx))))));
#endif
        
    }
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
    lvx=add2(lvx,mul2(add2(sxx_x1,sxz_z2),lrip));
    lvz=add2(lvz,mul2(add2(szz_z1,sxz_x2),lrkp));
    //Write updated values to global memory
    vx(gidz,gidx) = __f22h2(lvx);
    vz(gidz,gidx) = __f22h2(lvz);

    
}
