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
///*Update of the velocity in 3D*/
//
///*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
//
//
//#define rho(z,y,x)     rho[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define rip(z,y,x)     rip[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define rjp(z,y,x)     rjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define rkp(z,y,x)     rkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define uipjp(z,y,x) uipjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define ujpkp(z,y,x) ujpkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define uipkp(z,y,x) uipkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define u(z,y,x)         u[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define pi(z,y,x)       pi[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define gradrho(z,y,x)   gradrho[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define gradM(z,y,x)   gradM[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define gradmu(z,y,x)   gradmu[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define gradtaup(z,y,x)   gradtaup[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define gradtaus(z,y,x)   gradtaus[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//
//#define taus(z,y,x)         taus[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define tausipjp(z,y,x) tausipjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define tausjpkp(z,y,x) tausjpkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define tausipkp(z,y,x) tausipkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define taup(z,y,x)         taup[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//
//#define vx(z,y,x)   vx[(x)*NY*(NZ)+(y)*(NZ)+(z)]
//#define vy(z,y,x)   vy[(x)*NY*(NZ)+(y)*(NZ)+(z)]
//#define vz(z,y,x)   vz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
//#define sxx(z,y,x) sxx[(x)*NY*(NZ)+(y)*(NZ)+(z)]
//#define syy(z,y,x) syy[(x)*NY*(NZ)+(y)*(NZ)+(z)]
//#define szz(z,y,x) szz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
//#define sxy(z,y,x) sxy[(x)*NY*(NZ)+(y)*(NZ)+(z)]
//#define syz(z,y,x) syz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
//#define sxz(z,y,x) sxz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
//
//#define rxx(z,y,x,l) rxx[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
//#define ryy(z,y,x,l) ryy[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
//#define rzz(z,y,x,l) rzz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
//#define rxy(z,y,x,l) rxy[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
//#define ryz(z,y,x,l) ryz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
//#define rxz(z,y,x,l) rxz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
//
//#define psi_sxx_x(z,y,x) psi_sxx_x[(x)*(NY-2*FDOH)*(NZ-FDOH)+(y)*(NZ-FDOH)+(z)]
//#define psi_sxy_x(z,y,x) psi_sxy_x[(x)*(NY-2*FDOH)*(NZ-FDOH)+(y)*(NZ-FDOH)+(z)]
//#define psi_sxz_x(z,y,x) psi_sxz_x[(x)*(NY-2*FDOH)*(NZ-FDOH)+(y)*(NZ-FDOH)+(z)]
//#define psi_sxy_y(z,y,x) psi_sxy_y[(x)*(2*NAB)*(NZ-FDOH)+(y)*(NZ-FDOH)+(z)]
//#define psi_syy_y(z,y,x) psi_syy_y[(x)*(2*NAB)*(NZ-FDOH)+(y)*(NZ-FDOH)+(z)]
//#define psi_syz_y(z,y,x) psi_syz_y[(x)*(2*NAB)*(NZ-FDOH)+(y)*(NZ-FDOH)+(z)]
//#define psi_sxz_z(z,y,x) psi_sxz_z[(x)*(NY-2*FDOH)*(NAB)+(y)*(NAB)+(z)]
//#define psi_syz_z(z,y,x) psi_syz_z[(x)*(NY-2*FDOH)*(NAB)+(y)*(NAB)+(z)]
//#define psi_szz_z(z,y,x) psi_szz_z[(x)*(NY-2*FDOH)*(NAB)+(y)*(NAB)+(z)]
//
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
//#define vyout(y,x) vyout[(y)*NT+(x)]
//#define vzout(y,x) vzout[(y)*NT+(x)]
//#define vx0(y,x) vx0[(y)*NT+(x)]
//#define vy0(y,x) vy0[(y)*NT+(x)]
//#define vz0(y,x) vz0[(y)*NT+(x)]
//#define rx(y,x) rx[(y)*NT+(x)]
//#define ry(y,x) ry[(y)*NT+(x)]
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
//extern "C" __global__ void update_v(int offcomm,
//                       __prec2 *vx,         __prec2 *vy,           __prec2 *vz,
//                       __prec2 *sxx,        __prec2 *syy,          __prec2 *szz,
//                       __prec2 *sxy,        __prec2 *syz,          __prec2 *sxz,
//                       float *rip,        float *rjp,          float *rkp,
//                       float *taper,
//                       float *K_x,        float *a_x,          float *b_x,
//                       float *K_x_half,   float *a_x_half,     float *b_x_half,
//                       float *K_y,        float *a_y,          float *b_y,
//                       float *K_y_half,   float *a_y_half,     float *b_y_half,
//                       float *K_z,        float *a_z,          float *b_z,
//                       float *K_z_half,   float *a_z_half,     float *b_z_half,
//                       __prec2 *psi_sxx_x,  __prec2 *psi_sxy_x, __prec2 *psi_sxy_y,
//                       __prec2 *psi_sxz_x,  __prec2 *psi_sxz_z, __prec2 *psi_syy_y,
//                       __prec2 *psi_syz_y,  __prec2 *psi_syz_z, __prec2 *psi_szz_z,
//                       int scaler_sxx)
//{
//
//    float2 sxx_x;
//    float2 syy_y;
//    float2 szz_z;
//    float2 sxy_y;
//    float2 sxy_x;
//    float2 syz_y;
//    float2 syz_z;
//    float2 sxz_x;
//    float2 sxz_z;
//
//// If we use local memory
//#if LOCAL_OFF==0
//
//    int lsizez = blockDim.x+FDOH;
//    int lsizey = blockDim.y+2*FDOH;
//    int lsizex = blockDim.z+2*FDOH;
//    int lidz = threadIdx.x+FDOH/2;
//    int lidy = threadIdx.y+FDOH;
//    int lidx = threadIdx.z+FDOH;
//    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH/2;
//    int gidy = blockIdx.y*blockDim.y + threadIdx.y+FDOH;
//    int gidx = blockIdx.z*blockDim.z + threadIdx.z+FDOH+offcomm;
//
//#define lsxx lvar
//#define lsyy lvar
//#define lszz lvar
//#define lsxy lvar
//#define lsyz lvar
//#define lsxz lvar
//
//#define lsxx2 lvar2
//#define lsyy2 lvar2
//#define lszz2 lvar2
//#define lsxy2 lvar2
//#define lsyz2 lvar2
//#define lsxz2 lvar2
//
//// If local memory is turned off
//#elif LOCAL_OFF==1
//
//    int lsizez = blockDim.x+FDOH;
//    int lsizey = blockDim.y+2*FDOH;
//    int lsizex = blockDim.z+2*FDOH;
//    int lidz = threadIdx.x+FDOH/2;
//    int lidy = threadIdx.y+FDOH;
//    int lidx = threadIdx.z+FDOH;
//    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH/2;
//    int gidy = blockIdx.y*blockDim.y + threadIdx.y+FDOH;
//    int gidx = blockIdx.z*blockDim.z + threadIdx.z+FDOH+offcomm;
//
//#define lsxx sxx
//#define lsyy syy
//#define lszz szz
//#define lsxy sxy
//#define lsyz syz
//#define lsxz sxz
//#define lidx gidx
//#define lidy gidy
//#define lidz gidz
//
//#endif
//
//// Calculation of the stresses spatial derivatives
//    {
//#if LOCAL_OFF==0
//        lsxx2(lidz,lidy,lidx)=sxx(gidz,gidy,gidx);
//        if (lidx<2*FDOH)
//            lsxx2(lidz,lidy,lidx-FDOH)=sxx(gidz,gidy,gidx-FDOH);
//        if (lidx+lsizex-3*FDOH<FDOH)
//            lsxx2(lidz,lidy,lidx+lsizex-3*FDOH)=sxx(gidz,gidy,gidx+lsizex-3*FDOH);
//        if (lidx>(lsizex-2*FDOH-1))
//            lsxx2(lidz,lidy,lidx+FDOH)=sxx(gidz,gidy,gidx+FDOH);
//        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
//            lsxx2(lidz,lidy,lidx-lsizex+3*FDOH)=sxx(gidz,gidy,gidx-lsizex+3*FDOH);
//
//        barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//
//#if   FDOH ==1
//        sxx_x.x = HC1*(__h2f(lsxx(2*lidz,lidy,lidx+1)) - __h2f(lsxx(2*lidz,lidy,lidx)));
//        sxx_x.y = HC1*(__h2f(lsxx(2*lidz+1,lidy,lidx+1)) - __h2f(lsxx(2*lidz+1,lidy,lidx)));
//#elif FDOH ==2
//        sxx_x.x = (HC1*(__h2f(lsxx(2*lidz,lidy,lidx+1)) - __h2f(lsxx(2*lidz,lidy,lidx)))
//                  +HC2*(__h2f(lsxx(2*lidz,lidy,lidx+2)) - __h2f(lsxx(2*lidz,lidy,lidx-1))));
//        sxx_x.y = (HC1*(__h2f(lsxx(2*lidz+1,lidy,lidx+1)) - __h2f(lsxx(2*lidz+1,lidy,lidx)))
//                  +HC2*(__h2f(lsxx(2*lidz+1,lidy,lidx+2)) - __h2f(lsxx(2*lidz+1,lidy,lidx-1))));
//#elif FDOH ==3
//        sxx_x.x = (HC1*(__h2f(lsxx(2*lidz,lidy,lidx+1))-__h2f(lsxx(2*lidz,lidy,lidx)))+
//                      HC2*(__h2f(lsxx(2*lidz,lidy,lidx+2))-__h2f(lsxx(2*lidz,lidy,lidx-1)))+
//                      HC3*(__h2f(lsxx(2*lidz,lidy,lidx+3))-__h2f(lsxx(2*lidz,lidy,lidx-2))));
//sxx_x.y = (HC1*(__h2f(lsxx(2*lidz+1,lidy,lidx+1))-__h2f(lsxx(2*lidz+1,lidy,lidx)))+
//                      HC2*(__h2f(lsxx(2*lidz+1,lidy,lidx+2))-__h2f(lsxx(2*lidz+1,lidy,lidx-1)))+
//                      HC3*(__h2f(lsxx(2*lidz+1,lidy,lidx+3))-__h2f(lsxx(2*lidz+1,lidy,lidx-2))));
//#elif FDOH ==4
//        sxx_x.x = (HC1*(__h2f(lsxx(2*lidz,lidy,lidx+1))-__h2f(lsxx(2*lidz,lidy,lidx)))+
//                      HC2*(__h2f(lsxx(2*lidz,lidy,lidx+2))-__h2f(lsxx(2*lidz,lidy,lidx-1)))+
//                      HC3*(__h2f(lsxx(2*lidz,lidy,lidx+3))-__h2f(lsxx(2*lidz,lidy,lidx-2)))+
//                      HC4*(__h2f(lsxx(2*lidz,lidy,lidx+4))-__h2f(lsxx(2*lidz,lidy,lidx-3))));
//sxx_x.y = (HC1*(__h2f(lsxx(2*lidz+1,lidy,lidx+1))-__h2f(lsxx(2*lidz+1,lidy,lidx)))+
//                      HC2*(__h2f(lsxx(2*lidz+1,lidy,lidx+2))-__h2f(lsxx(2*lidz+1,lidy,lidx-1)))+
//                      HC3*(__h2f(lsxx(2*lidz+1,lidy,lidx+3))-__h2f(lsxx(2*lidz+1,lidy,lidx-2)))+
//                      HC4*(__h2f(lsxx(2*lidz+1,lidy,lidx+4))-__h2f(lsxx(2*lidz+1,lidy,lidx-3))));
//#elif FDOH ==5
//        sxx_x.x = (HC1*(__h2f(lsxx(2*lidz,lidy,lidx+1))-__h2f(lsxx(2*lidz,lidy,lidx)))+
//                      HC2*(__h2f(lsxx(2*lidz,lidy,lidx+2))-__h2f(lsxx(2*lidz,lidy,lidx-1)))+
//                      HC3*(__h2f(lsxx(2*lidz,lidy,lidx+3))-__h2f(lsxx(2*lidz,lidy,lidx-2)))+
//                      HC4*(__h2f(lsxx(2*lidz,lidy,lidx+4))-__h2f(lsxx(2*lidz,lidy,lidx-3)))+
//                      HC5*(__h2f(lsxx(2*lidz,lidy,lidx+5))-__h2f(lsxx(2*lidz,lidy,lidx-4))));
//sxx_x.y = (HC1*(__h2f(lsxx(2*lidz+1,lidy,lidx+1))-__h2f(lsxx(2*lidz+1,lidy,lidx)))+
//                      HC2*(__h2f(lsxx(2*lidz+1,lidy,lidx+2))-__h2f(lsxx(2*lidz+1,lidy,lidx-1)))+
//                      HC3*(__h2f(lsxx(2*lidz+1,lidy,lidx+3))-__h2f(lsxx(2*lidz+1,lidy,lidx-2)))+
//                      HC4*(__h2f(lsxx(2*lidz+1,lidy,lidx+4))-__h2f(lsxx(2*lidz+1,lidy,lidx-3)))+
//                      HC5*(__h2f(lsxx(2*lidz+1,lidy,lidx+5))-__h2f(lsxx(2*lidz+1,lidy,lidx-4))));
//#elif FDOH ==6
//        sxx_x.x = (HC1*(__h2f(lsxx(2*lidz,lidy,lidx+1))-__h2f(lsxx(2*lidz,lidy,lidx)))+
//                      HC2*(__h2f(lsxx(2*lidz,lidy,lidx+2))-__h2f(lsxx(2*lidz,lidy,lidx-1)))+
//                      HC3*(__h2f(lsxx(2*lidz,lidy,lidx+3))-__h2f(lsxx(2*lidz,lidy,lidx-2)))+
//                      HC4*(__h2f(lsxx(2*lidz,lidy,lidx+4))-__h2f(lsxx(2*lidz,lidy,lidx-3)))+
//                      HC5*(__h2f(lsxx(2*lidz,lidy,lidx+5))-__h2f(lsxx(2*lidz,lidy,lidx-4)))+
//                      HC6*(__h2f(lsxx(2*lidz,lidy,lidx+6))-__h2f(lsxx(2*lidz,lidy,lidx-5))));
//sxx_x.y = (HC1*(__h2f(lsxx(2*lidz+1,lidy,lidx+1))-__h2f(lsxx(2*lidz+1,lidy,lidx)))+
//                      HC2*(__h2f(lsxx(2*lidz+1,lidy,lidx+2))-__h2f(lsxx(2*lidz+1,lidy,lidx-1)))+
//                      HC3*(__h2f(lsxx(2*lidz+1,lidy,lidx+3))-__h2f(lsxx(2*lidz+1,lidy,lidx-2)))+
//                      HC4*(__h2f(lsxx(2*lidz+1,lidy,lidx+4))-__h2f(lsxx(2*lidz+1,lidy,lidx-3)))+
//                      HC5*(__h2f(lsxx(2*lidz+1,lidy,lidx+5))-__h2f(lsxx(2*lidz+1,lidy,lidx-4)))+
//                      HC6*(__h2f(lsxx(2*lidz+1,lidy,lidx+6))-__h2f(lsxx(2*lidz+1,lidy,lidx-5))));
//#endif
//
//
//#if LOCAL_OFF==0
//        barrier(CLK_LOCAL_MEM_FENCE);
//        lsyy2(lidz,lidy,lidx)=syy(gidz,gidy,gidx);
//        if (lidy<2*FDOH)
//            lsyy2(lidz,lidy-FDOH,lidx)=syy(gidz,gidy-FDOH,gidx);
//        if (lidy+lsizey-3*FDOH<FDOH)
//            lsyy2(lidz,lidy+lsizey-3*FDOH,lidx)=syy(gidz,gidy+lsizey-3*FDOH,gidx);
//        if (lidy>(lsizey-2*FDOH-1))
//            lsyy2(lidz,lidy+FDOH,lidx)=syy(gidz,gidy+FDOH,gidx);
//        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
//            lsyy2(lidz,lidy-lsizey+3*FDOH,lidx)=syy(gidz,gidy-lsizey+3*FDOH,gidx);
//        barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//
//#if   FDOH ==1
//        syy_y.x = HC1*(__h2f(lsyy(2*lidz,lidy+1,lidx)) - __h2f(lsyy(2*lidz,lidy,lidx)));
//        syy_y.y = HC1*(__h2f(lsyy(2*lidz+1,lidy+1,lidx)) - __h2f(lsyy(2*lidz+1,lidy,lidx)));
//#elif FDOH ==2
//        syy_y.x = (HC1*(__h2f(lsyy(2*lidz,lidy+1,lidx)) - __h2f(lsyy(2*lidz,lidy,lidx)))
//                  +HC2*(__h2f(lsyy(2*lidz,lidy+2,lidx)) - __h2f(lsyy(2*lidz,lidy-1,lidx))));
//        syy_y.y = (HC1*(__h2f(lsyy(2*lidz+1,lidy+1,lidx)) - __h2f(lsyy(2*lidz+1,lidy,lidx)))
//                  +HC2*(__h2f(lsyy(2*lidz+1,lidy+2,lidx)) - __h2f(lsyy(2*lidz+1,lidy-1,lidx))));
//#elif FDOH ==3
//        syy_y.x = (HC1*(__h2f(lsyy(2*lidz,lidy+1,lidx))-__h2f(lsyy(2*lidz,lidy,lidx)))+
//                      HC2*(__h2f(lsyy(2*lidz,lidy+2,lidx))-__h2f(lsyy(2*lidz,lidy-1,lidx)))+
//                      HC3*(__h2f(lsyy(2*lidz,lidy+3,lidx))-__h2f(lsyy(2*lidz,lidy-2,lidx))));
//syy_y.y = (HC1*(__h2f(lsyy(2*lidz+1,lidy+1,lidx))-__h2f(lsyy(2*lidz+1,lidy,lidx)))+
//                      HC2*(__h2f(lsyy(2*lidz+1,lidy+2,lidx))-__h2f(lsyy(2*lidz+1,lidy-1,lidx)))+
//                      HC3*(__h2f(lsyy(2*lidz+1,lidy+3,lidx))-__h2f(lsyy(2*lidz+1,lidy-2,lidx))));
//#elif FDOH ==4
//        syy_y.x = (HC1*(__h2f(lsyy(2*lidz,lidy+1,lidx))-__h2f(lsyy(2*lidz,lidy,lidx)))+
//                      HC2*(__h2f(lsyy(2*lidz,lidy+2,lidx))-__h2f(lsyy(2*lidz,lidy-1,lidx)))+
//                      HC3*(__h2f(lsyy(2*lidz,lidy+3,lidx))-__h2f(lsyy(2*lidz,lidy-2,lidx)))+
//                      HC4*(__h2f(lsyy(2*lidz,lidy+4,lidx))-__h2f(lsyy(2*lidz,lidy-3,lidx))));
//syy_y.y = (HC1*(__h2f(lsyy(2*lidz+1,lidy+1,lidx))-__h2f(lsyy(2*lidz+1,lidy,lidx)))+
//                      HC2*(__h2f(lsyy(2*lidz+1,lidy+2,lidx))-__h2f(lsyy(2*lidz+1,lidy-1,lidx)))+
//                      HC3*(__h2f(lsyy(2*lidz+1,lidy+3,lidx))-__h2f(lsyy(2*lidz+1,lidy-2,lidx)))+
//                      HC4*(__h2f(lsyy(2*lidz+1,lidy+4,lidx))-__h2f(lsyy(2*lidz+1,lidy-3,lidx))));
//#elif FDOH ==5
//        syy_y.x = (HC1*(__h2f(lsyy(2*lidz,lidy+1,lidx))-__h2f(lsyy(2*lidz,lidy,lidx)))+
//                      HC2*(__h2f(lsyy(2*lidz,lidy+2,lidx))-__h2f(lsyy(2*lidz,lidy-1,lidx)))+
//                      HC3*(__h2f(lsyy(2*lidz,lidy+3,lidx))-__h2f(lsyy(2*lidz,lidy-2,lidx)))+
//                      HC4*(__h2f(lsyy(2*lidz,lidy+4,lidx))-__h2f(lsyy(2*lidz,lidy-3,lidx)))+
//                      HC5*(__h2f(lsyy(2*lidz,lidy+5,lidx))-__h2f(lsyy(2*lidz,lidy-4,lidx))));
//syy_y.y = (HC1*(__h2f(lsyy(2*lidz+1,lidy+1,lidx))-__h2f(lsyy(2*lidz+1,lidy,lidx)))+
//                      HC2*(__h2f(lsyy(2*lidz+1,lidy+2,lidx))-__h2f(lsyy(2*lidz+1,lidy-1,lidx)))+
//                      HC3*(__h2f(lsyy(2*lidz+1,lidy+3,lidx))-__h2f(lsyy(2*lidz+1,lidy-2,lidx)))+
//                      HC4*(__h2f(lsyy(2*lidz+1,lidy+4,lidx))-__h2f(lsyy(2*lidz+1,lidy-3,lidx)))+
//                      HC5*(__h2f(lsyy(2*lidz+1,lidy+5,lidx))-__h2f(lsyy(2*lidz+1,lidy-4,lidx))));
//#elif FDOH ==6
//        syy_y.x = (HC1*(__h2f(lsyy(2*lidz,lidy+1,lidx))-__h2f(lsyy(2*lidz,lidy,lidx)))+
//                      HC2*(__h2f(lsyy(2*lidz,lidy+2,lidx))-__h2f(lsyy(2*lidz,lidy-1,lidx)))+
//                      HC3*(__h2f(lsyy(2*lidz,lidy+3,lidx))-__h2f(lsyy(2*lidz,lidy-2,lidx)))+
//                      HC4*(__h2f(lsyy(2*lidz,lidy+4,lidx))-__h2f(lsyy(2*lidz,lidy-3,lidx)))+
//                      HC5*(__h2f(lsyy(2*lidz,lidy+5,lidx))-__h2f(lsyy(2*lidz,lidy-4,lidx)))+
//                      HC6*(__h2f(lsyy(2*lidz,lidy+6,lidx))-__h2f(lsyy(2*lidz,lidy-5,lidx))));
//syy_y.y = (HC1*(__h2f(lsyy(2*lidz+1,lidy+1,lidx))-__h2f(lsyy(2*lidz+1,lidy,lidx)))+
//                      HC2*(__h2f(lsyy(2*lidz+1,lidy+2,lidx))-__h2f(lsyy(2*lidz+1,lidy-1,lidx)))+
//                      HC3*(__h2f(lsyy(2*lidz+1,lidy+3,lidx))-__h2f(lsyy(2*lidz+1,lidy-2,lidx)))+
//                      HC4*(__h2f(lsyy(2*lidz+1,lidy+4,lidx))-__h2f(lsyy(2*lidz+1,lidy-3,lidx)))+
//                      HC5*(__h2f(lsyy(2*lidz+1,lidy+5,lidx))-__h2f(lsyy(2*lidz+1,lidy-4,lidx)))+
//                      HC6*(__h2f(lsyy(2*lidz+1,lidy+6,lidx))-__h2f(lsyy(2*lidz+1,lidy-5,lidx))));
//#endif
//
//#if LOCAL_OFF==0
//        barrier(CLK_LOCAL_MEM_FENCE);
//        lszz2(lidz,lidy,lidx)=szz(gidz,gidy,gidx);
//        if (lidz<FDOH)
//            lszz2(lidz-FDOH,lidy,lidx)=szz(gidz-FDOH,gidy,gidx);
//        if (lidz>(lsizez-FDOH-1))
//            lszz2(lidz+FDOH,lidy,lidx)=szz(gidz+FDOH,gidy,gidx);
//        barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//
//#if   FDOH ==1
//        szz_z.x = HC1*(__h2f(lszz(2*lidz+1,lidy,lidx)) - __h2f(lszz(2*lidz,lidy,lidx)));
//        szz_z.y = HC1*(__h2f(lszz(2*lidz+1+1,lidy,lidx)) - __h2f(lszz(2*lidz+1,lidy,lidx)));
//#elif FDOH ==2
//        szz_z.x = (HC1*(__h2f(lszz(2*lidz+1,lidy,lidx)) - __h2f(lszz(2*lidz,lidy,lidx)))
//                  +HC2*(__h2f(lszz(2*lidz+2,lidy,lidx)) - __h2f(lszz(2*lidz-1,lidy,lidx))));
//        szz_z.y = (HC1*(__h2f(lszz(2*lidz+1+1,lidy,lidx)) - __h2f(lszz(2*lidz+1,lidy,lidx)))
//                  +HC2*(__h2f(lszz(2*lidz+2+1,lidy,lidx)) - __h2f(lszz(2*lidz-1+1,lidy,lidx))));
//#elif FDOH ==3
//        szz_z.x = (HC1*(__h2f(lszz(2*lidz+1,lidy,lidx))-__h2f(lszz(2*lidz,lidy,lidx)))+
//                      HC2*(__h2f(lszz(2*lidz+2,lidy,lidx))-__h2f(lszz(2*lidz-1,lidy,lidx)))+
//                      HC3*(__h2f(lszz(2*lidz+3,lidy,lidx))-__h2f(lszz(2*lidz-2,lidy,lidx))));
//szz_z.y = (HC1*(__h2f(lszz(2*lidz+1+1,lidy,lidx))-__h2f(lszz(2*lidz+1,lidy,lidx)))+
//                      HC2*(__h2f(lszz(2*lidz+1+2,lidy,lidx))-__h2f(lszz(2*lidz+1-1,lidy,lidx)))+
//                      HC3*(__h2f(lszz(2*lidz+1+3,lidy,lidx))-__h2f(lszz(2*lidz+1-2,lidy,lidx))));
//#elif FDOH ==4
//        szz_z.x = (HC1*(__h2f(lszz(2*lidz+1,lidy,lidx))-__h2f(lszz(2*lidz,lidy,lidx)))+
//                      HC2*(__h2f(lszz(2*lidz+2,lidy,lidx))-__h2f(lszz(2*lidz-1,lidy,lidx)))+
//                      HC3*(__h2f(lszz(2*lidz+3,lidy,lidx))-__h2f(lszz(2*lidz-2,lidy,lidx)))+
//                      HC4*(__h2f(lszz(2*lidz+4,lidy,lidx))-__h2f(lszz(2*lidz-3,lidy,lidx))));
//szz_z.y = (HC1*(__h2f(lszz(2*lidz+1+1,lidy,lidx))-__h2f(lszz(2*lidz+1,lidy,lidx)))+
//                      HC2*(__h2f(lszz(2*lidz+1+2,lidy,lidx))-__h2f(lszz(2*lidz+1-1,lidy,lidx)))+
//                      HC3*(__h2f(lszz(2*lidz+1+3,lidy,lidx))-__h2f(lszz(2*lidz+1-2,lidy,lidx)))+
//                      HC4*(__h2f(lszz(2*lidz+1+4,lidy,lidx))-__h2f(lszz(2*lidz+1-3,lidy,lidx))));
//#elif FDOH ==5
//        szz_z.x = (HC1*(__h2f(lszz(2*lidz+1,lidy,lidx))-__h2f(lszz(2*lidz,lidy,lidx)))+
//                      HC2*(__h2f(lszz(2*lidz+2,lidy,lidx))-__h2f(lszz(2*lidz-1,lidy,lidx)))+
//                      HC3*(__h2f(lszz(2*lidz+3,lidy,lidx))-__h2f(lszz(2*lidz-2,lidy,lidx)))+
//                      HC4*(__h2f(lszz(2*lidz+4,lidy,lidx))-__h2f(lszz(2*lidz-3,lidy,lidx)))+
//                      HC5*(__h2f(lszz(2*lidz+5,lidy,lidx))-__h2f(lszz(2*lidz-4,lidy,lidx))));
//szz_z.y = (HC1*(__h2f(lszz(2*lidz+1+1,lidy,lidx))-__h2f(lszz(2*lidz+1,lidy,lidx)))+
//                      HC2*(__h2f(lszz(2*lidz+1+2,lidy,lidx))-__h2f(lszz(2*lidz+1-1,lidy,lidx)))+
//                      HC3*(__h2f(lszz(2*lidz+1+3,lidy,lidx))-__h2f(lszz(2*lidz+1-2,lidy,lidx)))+
//                      HC4*(__h2f(lszz(2*lidz+1+4,lidy,lidx))-__h2f(lszz(2*lidz+1-3,lidy,lidx)))+
//                      HC5*(__h2f(lszz(2*lidz+1+5,lidy,lidx))-__h2f(lszz(2*lidz+1-4,lidy,lidx))));
//#elif FDOH ==6
//        szz_z.x = (HC1*(__h2f(lszz(2*lidz+1,lidy,lidx))-__h2f(lszz(2*lidz,lidy,lidx)))+
//                      HC2*(__h2f(lszz(2*lidz+2,lidy,lidx))-__h2f(lszz(2*lidz-1,lidy,lidx)))+
//                      HC3*(__h2f(lszz(2*lidz+3,lidy,lidx))-__h2f(lszz(2*lidz-2,lidy,lidx)))+
//                      HC4*(__h2f(lszz(2*lidz+4,lidy,lidx))-__h2f(lszz(2*lidz-3,lidy,lidx)))+
//                      HC5*(__h2f(lszz(2*lidz+5,lidy,lidx))-__h2f(lszz(2*lidz-4,lidy,lidx)))+
//                      HC6*(__h2f(lszz(2*lidz+6,lidy,lidx))-__h2f(lszz(2*lidz-5,lidy,lidx))));
//szz_z.y = (HC1*(__h2f(lszz(2*lidz+1+1,lidy,lidx))-__h2f(lszz(2*lidz+1,lidy,lidx)))+
//                      HC2*(__h2f(lszz(2*lidz+1+2,lidy,lidx))-__h2f(lszz(2*lidz+1-1,lidy,lidx)))+
//                      HC3*(__h2f(lszz(2*lidz+1+3,lidy,lidx))-__h2f(lszz(2*lidz+1-2,lidy,lidx)))+
//                      HC4*(__h2f(lszz(2*lidz+1+4,lidy,lidx))-__h2f(lszz(2*lidz+1-3,lidy,lidx)))+
//                      HC5*(__h2f(lszz(2*lidz+1+5,lidy,lidx))-__h2f(lszz(2*lidz+1-4,lidy,lidx)))+
//                      HC6*(__h2f(lszz(2*lidz+1+6,lidy,lidx))-__h2f(lszz(2*lidz+1-5,lidy,lidx))));
//#endif
//
//
//#if LOCAL_OFF==0
//        barrier(CLK_LOCAL_MEM_FENCE);
//        lsxy2(lidz,lidy,lidx)=sxy(gidz,gidy,gidx);
//        if (lidy<2*FDOH)
//            lsxy2(lidz,lidy-FDOH,lidx)=sxy(gidz,gidy-FDOH,gidx);
//        if (lidy+lsizey-3*FDOH<FDOH)
//            lsxy2(lidz,lidy+lsizey-3*FDOH,lidx)=sxy(gidz,gidy+lsizey-3*FDOH,gidx);
//        if (lidy>(lsizey-2*FDOH-1))
//            lsxy2(lidz,lidy+FDOH,lidx)=sxy(gidz,gidy+FDOH,gidx);
//        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
//            lsxy2(lidz,lidy-lsizey+3*FDOH,lidx)=sxy(gidz,gidy-lsizey+3*FDOH,gidx);
//        if (lidx<2*FDOH)
//            lsxy2(lidz,lidy,lidx-FDOH)=sxy(gidz,gidy,gidx-FDOH);
//        if (lidx+lsizex-3*FDOH<FDOH)
//            lsxy2(lidz,lidy,lidx+lsizex-3*FDOH)=sxy(gidz,gidy,gidx+lsizex-3*FDOH);
//        if (lidx>(lsizex-2*FDOH-1))
//            lsxy2(lidz,lidy,lidx+FDOH)=sxy(gidz,gidy,gidx+FDOH);
//        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
//            lsxy2(lidz,lidy,lidx-lsizex+3*FDOH)=sxy(gidz,gidy,gidx-lsizex+3*FDOH);
//        barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//
//#if   FDOH ==1
//        sxy_y.x = HC1*(__h2f(lsxy(2*lidz,lidy,lidx))   - __h2f(lsxy(2*lidz,lidy-1,lidx)));
//        sxy_y.y = HC1*(__h2f(lsxy(2*lidz+1,lidy,lidx))   - __h2f(lsxy(2*lidz+1,lidy-1,lidx)));
//        sxy_x.x = HC1*(__h2f(lsxy(2*lidz,lidy,lidx))   - __h2f(lsxy(2*lidz,lidy,lidx-1)));
//        sxy_x.y = HC1*(__h2f(lsxy(2*lidz+1,lidy,lidx))   - __h2f(lsxy(2*lidz+1,lidy,lidx-1)));
//#elif FDOH ==2
//        sxy_y.x = (HC1*(__h2f(lsxy(2*lidz,lidy,lidx))   - __h2f(lsxy(2*lidz,lidy-1,lidx)))
//                  +HC2*(__h2f(lsxy(2*lidz,lidy+1,lidx)) - __h2f(lsxy(2*lidz,lidy-2,lidx))));
//        sxy_y.y = (HC1*(__h2f(lsxy(2*lidz+1,lidy,lidx))   - __h2f(lsxy(2*lidz+1,lidy-1,lidx)))
//                  +HC2*(__h2f(lsxy(2*lidz+1,lidy+1,lidx)) - __h2f(lsxy(2*lidz+1,lidy-2,lidx))));
//        sxy_x.x = (HC1*(__h2f(lsxy(2*lidz,lidy,lidx))   - __h2f(lsxy(2*lidz,lidy,lidx-1)))
//                  +HC2*(__h2f(lsxy(2*lidz,lidy,lidx+1)) - __h2f(lsxy(2*lidz,lidy,lidx-2))));
//        sxy_x.y = (HC1*(__h2f(lsxy(2*lidz+1,lidy,lidx))   - __h2f(lsxy(2*lidz+1,lidy,lidx-1)))
//                  +HC2*(__h2f(lsxy(2*lidz+1,lidy,lidx+1)) - __h2f(lsxy(2*lidz+1,lidy,lidx-2))));
//#elif FDOH ==3
//        sxy_y.x = (HC1*(__h2f(lsxy(2*lidz,lidy,lidx))  -__h2f(lsxy(2*lidz,lidy-1,lidx)))+
//                      HC2*(__h2f(lsxy(2*lidz,lidy+1,lidx))-__h2f(lsxy(2*lidz,lidy-2,lidx)))+
//                      HC3*(__h2f(lsxy(2*lidz,lidy+2,lidx))-__h2f(lsxy(2*lidz,lidy-3,lidx))));
//sxy_y.y = (HC1*(__h2f(lsxy(2*lidz+1,lidy,lidx))  -__h2f(lsxy(2*lidz+1,lidy-1,lidx)))+
//                      HC2*(__h2f(lsxy(2*lidz+1,lidy+1,lidx))-__h2f(lsxy(2*lidz+1,lidy-2,lidx)))+
//                      HC3*(__h2f(lsxy(2*lidz+1,lidy+2,lidx))-__h2f(lsxy(2*lidz+1,lidy-3,lidx))));
//
//        sxy_x.x = (HC1*(__h2f(lsxy(2*lidz,lidy,lidx))  -__h2f(lsxy(2*lidz,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxy(2*lidz,lidy,lidx+1))-__h2f(lsxy(2*lidz,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxy(2*lidz,lidy,lidx+2))-__h2f(lsxy(2*lidz,lidy,lidx-3))));
//sxy_x.y = (HC1*(__h2f(lsxy(2*lidz+1,lidy,lidx))  -__h2f(lsxy(2*lidz+1,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxy(2*lidz+1,lidy,lidx+1))-__h2f(lsxy(2*lidz+1,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxy(2*lidz+1,lidy,lidx+2))-__h2f(lsxy(2*lidz+1,lidy,lidx-3))));
//#elif FDOH ==4
//        sxy_y.x = (HC1*(__h2f(lsxy(2*lidz,lidy,lidx))  -__h2f(lsxy(2*lidz,lidy-1,lidx)))+
//                      HC2*(__h2f(lsxy(2*lidz,lidy+1,lidx))-__h2f(lsxy(2*lidz,lidy-2,lidx)))+
//                      HC3*(__h2f(lsxy(2*lidz,lidy+2,lidx))-__h2f(lsxy(2*lidz,lidy-3,lidx)))+
//                      HC4*(__h2f(lsxy(2*lidz,lidy+3,lidx))-__h2f(lsxy(2*lidz,lidy-4,lidx))));
//sxy_y.y = (HC1*(__h2f(lsxy(2*lidz+1,lidy,lidx))  -__h2f(lsxy(2*lidz+1,lidy-1,lidx)))+
//                      HC2*(__h2f(lsxy(2*lidz+1,lidy+1,lidx))-__h2f(lsxy(2*lidz+1,lidy-2,lidx)))+
//                      HC3*(__h2f(lsxy(2*lidz+1,lidy+2,lidx))-__h2f(lsxy(2*lidz+1,lidy-3,lidx)))+
//                      HC4*(__h2f(lsxy(2*lidz+1,lidy+3,lidx))-__h2f(lsxy(2*lidz+1,lidy-4,lidx))));
//
//        sxy_x.x = (HC1*(__h2f(lsxy(2*lidz,lidy,lidx))  -__h2f(lsxy(2*lidz,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxy(2*lidz,lidy,lidx+1))-__h2f(lsxy(2*lidz,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxy(2*lidz,lidy,lidx+2))-__h2f(lsxy(2*lidz,lidy,lidx-3)))+
//                      HC4*(__h2f(lsxy(2*lidz,lidy,lidx+3))-__h2f(lsxy(2*lidz,lidy,lidx-4))));
//sxy_x.y = (HC1*(__h2f(lsxy(2*lidz+1,lidy,lidx))  -__h2f(lsxy(2*lidz+1,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxy(2*lidz+1,lidy,lidx+1))-__h2f(lsxy(2*lidz+1,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxy(2*lidz+1,lidy,lidx+2))-__h2f(lsxy(2*lidz+1,lidy,lidx-3)))+
//                      HC4*(__h2f(lsxy(2*lidz+1,lidy,lidx+3))-__h2f(lsxy(2*lidz+1,lidy,lidx-4))));
//#elif FDOH ==5
//        sxy_y.x = (HC1*(__h2f(lsxy(2*lidz,lidy,lidx))  -__h2f(lsxy(2*lidz,lidy-1,lidx)))+
//                      HC2*(__h2f(lsxy(2*lidz,lidy+1,lidx))-__h2f(lsxy(2*lidz,lidy-2,lidx)))+
//                      HC3*(__h2f(lsxy(2*lidz,lidy+2,lidx))-__h2f(lsxy(2*lidz,lidy-3,lidx)))+
//                      HC4*(__h2f(lsxy(2*lidz,lidy+3,lidx))-__h2f(lsxy(2*lidz,lidy-4,lidx)))+
//                      HC5*(__h2f(lsxy(2*lidz,lidy+4,lidx))-__h2f(lsxy(2*lidz,lidy-5,lidx))));
//sxy_y.y = (HC1*(__h2f(lsxy(2*lidz+1,lidy,lidx))  -__h2f(lsxy(2*lidz+1,lidy-1,lidx)))+
//                      HC2*(__h2f(lsxy(2*lidz+1,lidy+1,lidx))-__h2f(lsxy(2*lidz+1,lidy-2,lidx)))+
//                      HC3*(__h2f(lsxy(2*lidz+1,lidy+2,lidx))-__h2f(lsxy(2*lidz+1,lidy-3,lidx)))+
//                      HC4*(__h2f(lsxy(2*lidz+1,lidy+3,lidx))-__h2f(lsxy(2*lidz+1,lidy-4,lidx)))+
//                      HC5*(__h2f(lsxy(2*lidz+1,lidy+4,lidx))-__h2f(lsxy(2*lidz+1,lidy-5,lidx))));
//
//        sxy_x.x = (HC1*(__h2f(lsxy(2*lidz,lidy,lidx))  -__h2f(lsxy(2*lidz,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxy(2*lidz,lidy,lidx+1))-__h2f(lsxy(2*lidz,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxy(2*lidz,lidy,lidx+2))-__h2f(lsxy(2*lidz,lidy,lidx-3)))+
//                      HC4*(__h2f(lsxy(2*lidz,lidy,lidx+3))-__h2f(lsxy(2*lidz,lidy,lidx-4)))+
//                      HC5*(__h2f(lsxy(2*lidz,lidy,lidx+4))-__h2f(lsxy(2*lidz,lidy,lidx-5))));
//sxy_x.y = (HC1*(__h2f(lsxy(2*lidz+1,lidy,lidx))  -__h2f(lsxy(2*lidz+1,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxy(2*lidz+1,lidy,lidx+1))-__h2f(lsxy(2*lidz+1,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxy(2*lidz+1,lidy,lidx+2))-__h2f(lsxy(2*lidz+1,lidy,lidx-3)))+
//                      HC4*(__h2f(lsxy(2*lidz+1,lidy,lidx+3))-__h2f(lsxy(2*lidz+1,lidy,lidx-4)))+
//                      HC5*(__h2f(lsxy(2*lidz+1,lidy,lidx+4))-__h2f(lsxy(2*lidz+1,lidy,lidx-5))));
//
//#elif FDOH ==6
//
//        sxy_y.x = (HC1*(__h2f(lsxy(2*lidz,lidy,lidx))  -__h2f(lsxy(2*lidz,lidy-1,lidx)))+
//                      HC2*(__h2f(lsxy(2*lidz,lidy+1,lidx))-__h2f(lsxy(2*lidz,lidy-2,lidx)))+
//                      HC3*(__h2f(lsxy(2*lidz,lidy+2,lidx))-__h2f(lsxy(2*lidz,lidy-3,lidx)))+
//                      HC4*(__h2f(lsxy(2*lidz,lidy+3,lidx))-__h2f(lsxy(2*lidz,lidy-4,lidx)))+
//                      HC5*(__h2f(lsxy(2*lidz,lidy+4,lidx))-__h2f(lsxy(2*lidz,lidy-5,lidx)))+
//                      HC6*(__h2f(lsxy(2*lidz,lidy+5,lidx))-__h2f(lsxy(2*lidz,lidy-6,lidx))));
//sxy_y.y = (HC1*(__h2f(lsxy(2*lidz+1,lidy,lidx))  -__h2f(lsxy(2*lidz+1,lidy-1,lidx)))+
//                      HC2*(__h2f(lsxy(2*lidz+1,lidy+1,lidx))-__h2f(lsxy(2*lidz+1,lidy-2,lidx)))+
//                      HC3*(__h2f(lsxy(2*lidz+1,lidy+2,lidx))-__h2f(lsxy(2*lidz+1,lidy-3,lidx)))+
//                      HC4*(__h2f(lsxy(2*lidz+1,lidy+3,lidx))-__h2f(lsxy(2*lidz+1,lidy-4,lidx)))+
//                      HC5*(__h2f(lsxy(2*lidz+1,lidy+4,lidx))-__h2f(lsxy(2*lidz+1,lidy-5,lidx)))+
//                      HC6*(__h2f(lsxy(2*lidz+1,lidy+5,lidx))-__h2f(lsxy(2*lidz+1,lidy-6,lidx))));
//
//        sxy_x.x = (HC1*(__h2f(lsxy(2*lidz,lidy,lidx))  -__h2f(lsxy(2*lidz,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxy(2*lidz,lidy,lidx+1))-__h2f(lsxy(2*lidz,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxy(2*lidz,lidy,lidx+2))-__h2f(lsxy(2*lidz,lidy,lidx-3)))+
//                      HC4*(__h2f(lsxy(2*lidz,lidy,lidx+3))-__h2f(lsxy(2*lidz,lidy,lidx-4)))+
//                      HC5*(__h2f(lsxy(2*lidz,lidy,lidx+4))-__h2f(lsxy(2*lidz,lidy,lidx-5)))+
//                      HC6*(__h2f(lsxy(2*lidz,lidy,lidx+5))-__h2f(lsxy(2*lidz,lidy,lidx-6))));
//sxy_x.y = (HC1*(__h2f(lsxy(2*lidz+1,lidy,lidx))  -__h2f(lsxy(2*lidz+1,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxy(2*lidz+1,lidy,lidx+1))-__h2f(lsxy(2*lidz+1,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxy(2*lidz+1,lidy,lidx+2))-__h2f(lsxy(2*lidz+1,lidy,lidx-3)))+
//                      HC4*(__h2f(lsxy(2*lidz+1,lidy,lidx+3))-__h2f(lsxy(2*lidz+1,lidy,lidx-4)))+
//                      HC5*(__h2f(lsxy(2*lidz+1,lidy,lidx+4))-__h2f(lsxy(2*lidz+1,lidy,lidx-5)))+
//                      HC6*(__h2f(lsxy(2*lidz+1,lidy,lidx+5))-__h2f(lsxy(2*lidz+1,lidy,lidx-6))));
//#endif
//
//#if LOCAL_OFF==0
//        barrier(CLK_LOCAL_MEM_FENCE);
//        lsyz2(lidz,lidy,lidx)=syz(gidz,gidy,gidx);
//        if (lidy<2*FDOH)
//            lsyz2(lidz,lidy-FDOH,lidx)=syz(gidz,gidy-FDOH,gidx);
//        if (lidy+lsizey-3*FDOH<FDOH)
//            lsyz2(lidz,lidy+lsizey-3*FDOH,lidx)=syz(gidz,gidy+lsizey-3*FDOH,gidx);
//        if (lidy>(lsizey-2*FDOH-1))
//            lsyz2(lidz,lidy+FDOH,lidx)=syz(gidz,gidy+FDOH,gidx);
//        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
//            lsyz2(lidz,lidy-lsizey+3*FDOH,lidx)=syz(gidz,gidy-lsizey+3*FDOH,gidx);
//        if (lidz<FDOH)
//            lsyz2(lidz-FDOH,lidy,lidx)=syz(gidz-FDOH,gidy,gidx);
//        if (lidz>(lsizez-FDOH-1))
//            lsyz2(lidz+FDOH,lidy,lidx)=syz(gidz+FDOH,gidy,gidx);
//        barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//
//#if   FDOH ==1
//        syz_z.x = HC1*(__h2f(lsyz(2*lidz,lidy,lidx))   - __h2f(lsyz(2*lidz-1,lidy,lidx)));
//        syz_z.y = HC1*(__h2f(lsyz(2*lidz+1,lidy,lidx))   - __h2f(lsyz(2*lidz-1+1,lidy,lidx)));
//        syz_y.x = HC1*(__h2f(lsyz(2*lidz,lidy,lidx))   - __h2f(lsyz(2*lidz,lidy-1,lidx)));
//        syz_y.y = HC1*(__h2f(lsyz(2*lidz+1,lidy,lidx))   - __h2f(lsyz(2*lidz+1,lidy-1,lidx)));
//#elif FDOH ==2
//        syz_z.x = (HC1*(__h2f(lsyz(2*lidz,lidy,lidx))   - __h2f(lsyz(2*lidz-1,lidy,lidx)))
//                  +HC2*(__h2f(lsyz(2*lidz+1,lidy,lidx)) - __h2f(lsyz(2*lidz-2,lidy,lidx))));
//        syz_z.y = (HC1*(__h2f(lsyz(2*lidz+1,lidy,lidx))   - __h2f(lsyz(2*lidz-1+1,lidy,lidx)))
//                  +HC2*(__h2f(lsyz(2*lidz+1+1,lidy,lidx)) - __h2f(lsyz(2*lidz-2+1,lidy,lidx))));
//        syz_y.x = (HC1*(__h2f(lsyz(2*lidz,lidy,lidx))   - __h2f(lsyz(2*lidz,lidy-1,lidx)))
//                  +HC2*(__h2f(lsyz(2*lidz,lidy+1,lidx)) - __h2f(lsyz(2*lidz,lidy-2,lidx))));
//        syz_y.y = (HC1*(__h2f(lsyz(2*lidz+1,lidy,lidx))   - __h2f(lsyz(2*lidz+1,lidy-1,lidx)))
//                  +HC2*(__h2f(lsyz(2*lidz+1,lidy+1,lidx)) - __h2f(lsyz(2*lidz+1,lidy-2,lidx))));
//#elif FDOH ==3
//        syz_z.x = (HC1*(__h2f(lsyz(2*lidz,lidy,lidx))  -__h2f(lsyz(2*lidz-1,lidy,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz+1,lidy,lidx))-__h2f(lsyz(2*lidz-2,lidy,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz+2,lidy,lidx))-__h2f(lsyz(2*lidz-3,lidy,lidx))));
//syz_z.y = (HC1*(__h2f(lsyz(2*lidz+1,lidy,lidx))  -__h2f(lsyz(2*lidz+1-1,lidy,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz+1+1,lidy,lidx))-__h2f(lsyz(2*lidz+1-2,lidy,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz+1+2,lidy,lidx))-__h2f(lsyz(2*lidz+1-3,lidy,lidx))));
//
//        syz_y.x = (HC1*(__h2f(lsyz(2*lidz,lidy,lidx))  -__h2f(lsyz(2*lidz,lidy-1,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz,lidy+1,lidx))-__h2f(lsyz(2*lidz,lidy-2,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz,lidy+2,lidx))-__h2f(lsyz(2*lidz,lidy-3,lidx))));
//syz_y.y = (HC1*(__h2f(lsyz(2*lidz+1,lidy,lidx))  -__h2f(lsyz(2*lidz+1,lidy-1,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz+1,lidy+1,lidx))-__h2f(lsyz(2*lidz+1,lidy-2,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz+1,lidy+2,lidx))-__h2f(lsyz(2*lidz+1,lidy-3,lidx))));
//#elif FDOH ==4
//        syz_z.x = (HC1*(__h2f(lsyz(2*lidz,lidy,lidx))  -__h2f(lsyz(2*lidz-1,lidy,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz+1,lidy,lidx))-__h2f(lsyz(2*lidz-2,lidy,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz+2,lidy,lidx))-__h2f(lsyz(2*lidz-3,lidy,lidx)))+
//                      HC4*(__h2f(lsyz(2*lidz+3,lidy,lidx))-__h2f(lsyz(2*lidz-4,lidy,lidx))));
//syz_z.y = (HC1*(__h2f(lsyz(2*lidz+1,lidy,lidx))  -__h2f(lsyz(2*lidz+1-1,lidy,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz+1+1,lidy,lidx))-__h2f(lsyz(2*lidz+1-2,lidy,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz+1+2,lidy,lidx))-__h2f(lsyz(2*lidz+1-3,lidy,lidx)))+
//                      HC4*(__h2f(lsyz(2*lidz+1+3,lidy,lidx))-__h2f(lsyz(2*lidz+1-4,lidy,lidx))));
//
//        syz_y.x = (HC1*(__h2f(lsyz(2*lidz,lidy,lidx))-__h2f(lsyz(2*lidz,lidy-1,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz,lidy+1,lidx))-__h2f(lsyz(2*lidz,lidy-2,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz,lidy+2,lidx))-__h2f(lsyz(2*lidz,lidy-3,lidx)))+
//                      HC4*(__h2f(lsyz(2*lidz,lidy+3,lidx))-__h2f(lsyz(2*lidz,lidy-4,lidx))));
//syz_y.y = (HC1*(__h2f(lsyz(2*lidz+1,lidy,lidx))-__h2f(lsyz(2*lidz+1,lidy-1,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz+1,lidy+1,lidx))-__h2f(lsyz(2*lidz+1,lidy-2,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz+1,lidy+2,lidx))-__h2f(lsyz(2*lidz+1,lidy-3,lidx)))+
//                      HC4*(__h2f(lsyz(2*lidz+1,lidy+3,lidx))-__h2f(lsyz(2*lidz+1,lidy-4,lidx))));
//#elif FDOH ==5
//        syz_z.x = (HC1*(__h2f(lsyz(2*lidz,lidy,lidx))  -__h2f(lsyz(2*lidz-1,lidy,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz+1,lidy,lidx))-__h2f(lsyz(2*lidz-2,lidy,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz+2,lidy,lidx))-__h2f(lsyz(2*lidz-3,lidy,lidx)))+
//                      HC4*(__h2f(lsyz(2*lidz+3,lidy,lidx))-__h2f(lsyz(2*lidz-4,lidy,lidx)))+
//                      HC5*(__h2f(lsyz(2*lidz+4,lidy,lidx))-__h2f(lsyz(2*lidz-5,lidy,lidx))));
//syz_z.y = (HC1*(__h2f(lsyz(2*lidz+1,lidy,lidx))  -__h2f(lsyz(2*lidz+1-1,lidy,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz+1+1,lidy,lidx))-__h2f(lsyz(2*lidz+1-2,lidy,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz+1+2,lidy,lidx))-__h2f(lsyz(2*lidz+1-3,lidy,lidx)))+
//                      HC4*(__h2f(lsyz(2*lidz+1+3,lidy,lidx))-__h2f(lsyz(2*lidz+1-4,lidy,lidx)))+
//                      HC5*(__h2f(lsyz(2*lidz+1+4,lidy,lidx))-__h2f(lsyz(2*lidz+1-5,lidy,lidx))));
//
//        syz_y.x = (HC1*(__h2f(lsyz(2*lidz,lidy,lidx))  -__h2f(lsyz(2*lidz,lidy-1,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz,lidy+1,lidx))-__h2f(lsyz(2*lidz,lidy-2,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz,lidy+2,lidx))-__h2f(lsyz(2*lidz,lidy-3,lidx)))+
//                      HC4*(__h2f(lsyz(2*lidz,lidy+3,lidx))-__h2f(lsyz(2*lidz,lidy-4,lidx)))+
//                      HC5*(__h2f(lsyz(2*lidz,lidy+4,lidx))-__h2f(lsyz(2*lidz,lidy-5,lidx))));
//syz_y.y = (HC1*(__h2f(lsyz(2*lidz+1,lidy,lidx))  -__h2f(lsyz(2*lidz+1,lidy-1,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz+1,lidy+1,lidx))-__h2f(lsyz(2*lidz+1,lidy-2,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz+1,lidy+2,lidx))-__h2f(lsyz(2*lidz+1,lidy-3,lidx)))+
//                      HC4*(__h2f(lsyz(2*lidz+1,lidy+3,lidx))-__h2f(lsyz(2*lidz+1,lidy-4,lidx)))+
//                      HC5*(__h2f(lsyz(2*lidz+1,lidy+4,lidx))-__h2f(lsyz(2*lidz+1,lidy-5,lidx))));
//#elif FDOH ==6
//        syz_z.x = (HC1*(__h2f(lsyz(2*lidz,lidy,lidx))  -__h2f(lsyz(2*lidz-1,lidy,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz+1,lidy,lidx))-__h2f(lsyz(2*lidz-2,lidy,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz+2,lidy,lidx))-__h2f(lsyz(2*lidz-3,lidy,lidx)))+
//                      HC4*(__h2f(lsyz(2*lidz+3,lidy,lidx))-__h2f(lsyz(2*lidz-4,lidy,lidx)))+
//                      HC5*(__h2f(lsyz(2*lidz+4,lidy,lidx))-__h2f(lsyz(2*lidz-5,lidy,lidx)))+
//                      HC6*(__h2f(lsyz(2*lidz+5,lidy,lidx))-__h2f(lsyz(2*lidz-6,lidy,lidx))));
//syz_z.y = (HC1*(__h2f(lsyz(2*lidz+1,lidy,lidx))  -__h2f(lsyz(2*lidz+1-1,lidy,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz+1+1,lidy,lidx))-__h2f(lsyz(2*lidz+1-2,lidy,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz+1+2,lidy,lidx))-__h2f(lsyz(2*lidz+1-3,lidy,lidx)))+
//                      HC4*(__h2f(lsyz(2*lidz+1+3,lidy,lidx))-__h2f(lsyz(2*lidz+1-4,lidy,lidx)))+
//                      HC5*(__h2f(lsyz(2*lidz+1+4,lidy,lidx))-__h2f(lsyz(2*lidz+1-5,lidy,lidx)))+
//                      HC6*(__h2f(lsyz(2*lidz+1+5,lidy,lidx))-__h2f(lsyz(2*lidz+1-6,lidy,lidx))));
//
//        syz_y.x = (HC1*(__h2f(lsyz(2*lidz,lidy,lidx))  -__h2f(lsyz(2*lidz,lidy-1,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz,lidy+1,lidx))-__h2f(lsyz(2*lidz,lidy-2,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz,lidy+2,lidx))-__h2f(lsyz(2*lidz,lidy-3,lidx)))+
//                      HC4*(__h2f(lsyz(2*lidz,lidy+3,lidx))-__h2f(lsyz(2*lidz,lidy-4,lidx)))+
//                      HC5*(__h2f(lsyz(2*lidz,lidy+4,lidx))-__h2f(lsyz(2*lidz,lidy-5,lidx)))+
//                      HC6*(__h2f(lsyz(2*lidz,lidy+5,lidx))-__h2f(lsyz(2*lidz,lidy-6,lidx))));
//syz_y.y = (HC1*(__h2f(lsyz(2*lidz+1,lidy,lidx))  -__h2f(lsyz(2*lidz+1,lidy-1,lidx)))+
//                      HC2*(__h2f(lsyz(2*lidz+1,lidy+1,lidx))-__h2f(lsyz(2*lidz+1,lidy-2,lidx)))+
//                      HC3*(__h2f(lsyz(2*lidz+1,lidy+2,lidx))-__h2f(lsyz(2*lidz+1,lidy-3,lidx)))+
//                      HC4*(__h2f(lsyz(2*lidz+1,lidy+3,lidx))-__h2f(lsyz(2*lidz+1,lidy-4,lidx)))+
//                      HC5*(__h2f(lsyz(2*lidz+1,lidy+4,lidx))-__h2f(lsyz(2*lidz+1,lidy-5,lidx)))+
//                      HC6*(__h2f(lsyz(2*lidz+1,lidy+5,lidx))-__h2f(lsyz(2*lidz+1,lidy-6,lidx))));
//#endif
//
//#if LOCAL_OFF==0
//        barrier(CLK_LOCAL_MEM_FENCE);
//        lsxz2(lidz,lidy,lidx)=sxz(gidz,gidy,gidx);
//
//        if (lidx<2*FDOH)
//            lsxz2(lidz,lidy,lidx-FDOH)=sxz(gidz,gidy,gidx-FDOH);
//        if (lidx+lsizex-3*FDOH<FDOH)
//            lsxz2(lidz,lidy,lidx+lsizex-3*FDOH)=sxz(gidz,gidy,gidx+lsizex-3*FDOH);
//        if (lidx>(lsizex-2*FDOH-1))
//            lsxz2(lidz,lidy,lidx+FDOH)=sxz(gidz,gidy,gidx+FDOH);
//        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
//            lsxz2(lidz,lidy,lidx-lsizex+3*FDOH)=sxz(gidz,gidy,gidx-lsizex+3*FDOH);
//        if (lidz<FDOH)
//            lsxz2(lidz-FDOH,lidy,lidx)=sxz(gidz-FDOH,gidy,gidx);
//        if (lidz>(lsizez-FDOH-1))
//            lsxz2(lidz+FDOH,lidy,lidx)=sxz(gidz+FDOH,gidy,gidx);
//        barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//
//#if   FDOH ==1
//        sxz_z.x = HC1*(__h2f(lsxz(2*lidz,lidy,lidx))   - __h2f(lsxz(2*lidz-1,lidy,lidx)));
//        sxz_z.y = HC1*(__h2f(lsxz(2*lidz+1,lidy,lidx))   - __h2f(lsxz(2*lidz-1+1,lidy,lidx)));
//        sxz_x.x = HC1*(__h2f(lsxz(2*lidz,lidy,lidx))   - __h2f(lsxz(2*lidz,lidy,lidx-1)));
//        sxz_x.y = HC1*(__h2f(lsxz(2*lidz+1,lidy,lidx))   - __h2f(lsxz(2*lidz+1,lidy,lidx-1)));
//#elif FDOH ==2
//        sxz_z.x = (HC1*(__h2f(lsxz(2*lidz,lidy,lidx))   - __h2f(lsxz(2*lidz-1,lidy,lidx)))
//                  +HC2*(__h2f(lsxz(2*lidz+1,lidy,lidx)) - __h2f(lsxz(2*lidz-2,lidy,lidx))));
//        sxz_z.y = (HC1*(__h2f(lsxz(2*lidz+1,lidy,lidx))   - __h2f(lsxz(2*lidz-1+1,lidy,lidx)))
//                  +HC2*(__h2f(lsxz(2*lidz+1+1,lidy,lidx)) - __h2f(lsxz(2*lidz-2+1,lidy,lidx))));
//        sxz_x.x = (HC1*(__h2f(lsxz(2*lidz,lidy,lidx))   - __h2f(lsxz(2*lidz,lidy,lidx-1)))
//                  +HC2*(__h2f(lsxz(2*lidz,lidy,lidx+1)) - __h2f(lsxz(2*lidz,lidy,lidx-2))));
//        sxz_x.y = (HC1*(__h2f(lsxz(2*lidz+1,lidy,lidx))   - __h2f(lsxz(2*lidz+1,lidy,lidx-1)))
//                  +HC2*(__h2f(lsxz(2*lidz+1,lidy,lidx+1)) - __h2f(lsxz(2*lidz+1,lidy,lidx-2))));
//
//#elif FDOH ==3
//        sxz_z.x = (HC1*(__h2f(lsxz(2*lidz,lidy,lidx))  -__h2f(lsxz(2*lidz-1,lidy,lidx)))+
//                      HC2*(__h2f(lsxz(2*lidz+1,lidy,lidx))-__h2f(lsxz(2*lidz-2,lidy,lidx)))+
//                      HC3*(__h2f(lsxz(2*lidz+2,lidy,lidx))-__h2f(lsxz(2*lidz-3,lidy,lidx))));
//sxz_z.y = (HC1*(__h2f(lsxz(2*lidz+1,lidy,lidx))  -__h2f(lsxz(2*lidz+1-1,lidy,lidx)))+
//                      HC2*(__h2f(lsxz(2*lidz+1+1,lidy,lidx))-__h2f(lsxz(2*lidz+1-2,lidy,lidx)))+
//                      HC3*(__h2f(lsxz(2*lidz+1+2,lidy,lidx))-__h2f(lsxz(2*lidz+1-3,lidy,lidx))));
//
//        sxz_x.x = (HC1*(__h2f(lsxz(2*lidz,lidy,lidx))  -__h2f(lsxz(2*lidz,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxz(2*lidz,lidy,lidx+1))-__h2f(lsxz(2*lidz,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxz(2*lidz,lidy,lidx+2))-__h2f(lsxz(2*lidz,lidy,lidx-3))));
//sxz_x.y = (HC1*(__h2f(lsxz(2*lidz+1,lidy,lidx))  -__h2f(lsxz(2*lidz+1,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxz(2*lidz+1,lidy,lidx+1))-__h2f(lsxz(2*lidz+1,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxz(2*lidz+1,lidy,lidx+2))-__h2f(lsxz(2*lidz+1,lidy,lidx-3))));
//#elif FDOH ==4
//        sxz_z.x = (HC1*(__h2f(lsxz(2*lidz,lidy,lidx))  -__h2f(lsxz(2*lidz-1,lidy,lidx)))+
//                      HC2*(__h2f(lsxz(2*lidz+1,lidy,lidx))-__h2f(lsxz(2*lidz-2,lidy,lidx)))+
//                      HC3*(__h2f(lsxz(2*lidz+2,lidy,lidx))-__h2f(lsxz(2*lidz-3,lidy,lidx)))+
//                      HC4*(__h2f(lsxz(2*lidz+3,lidy,lidx))-__h2f(lsxz(2*lidz-4,lidy,lidx))));
//sxz_z.y = (HC1*(__h2f(lsxz(2*lidz+1,lidy,lidx))  -__h2f(lsxz(2*lidz+1-1,lidy,lidx)))+
//                      HC2*(__h2f(lsxz(2*lidz+1+1,lidy,lidx))-__h2f(lsxz(2*lidz+1-2,lidy,lidx)))+
//                      HC3*(__h2f(lsxz(2*lidz+1+2,lidy,lidx))-__h2f(lsxz(2*lidz+1-3,lidy,lidx)))+
//                      HC4*(__h2f(lsxz(2*lidz+1+3,lidy,lidx))-__h2f(lsxz(2*lidz+1-4,lidy,lidx))));
//
//        sxz_x.x = (HC1*(__h2f(lsxz(2*lidz,lidy,lidx))  -__h2f(lsxz(2*lidz,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxz(2*lidz,lidy,lidx+1))-__h2f(lsxz(2*lidz,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxz(2*lidz,lidy,lidx+2))-__h2f(lsxz(2*lidz,lidy,lidx-3)))+
//                      HC4*(__h2f(lsxz(2*lidz,lidy,lidx+3))-__h2f(lsxz(2*lidz,lidy,lidx-4))));
//sxz_x.y = (HC1*(__h2f(lsxz(2*lidz+1,lidy,lidx))  -__h2f(lsxz(2*lidz+1,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxz(2*lidz+1,lidy,lidx+1))-__h2f(lsxz(2*lidz+1,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxz(2*lidz+1,lidy,lidx+2))-__h2f(lsxz(2*lidz+1,lidy,lidx-3)))+
//                      HC4*(__h2f(lsxz(2*lidz+1,lidy,lidx+3))-__h2f(lsxz(2*lidz+1,lidy,lidx-4))));
//#elif FDOH ==5
//        sxz_z.x = (HC1*(__h2f(lsxz(2*lidz,lidy,lidx))  -__h2f(lsxz(2*lidz-1,lidy,lidx)))+
//                      HC2*(__h2f(lsxz(2*lidz+1,lidy,lidx))-__h2f(lsxz(2*lidz-2,lidy,lidx)))+
//                      HC3*(__h2f(lsxz(2*lidz+2,lidy,lidx))-__h2f(lsxz(2*lidz-3,lidy,lidx)))+
//                      HC4*(__h2f(lsxz(2*lidz+3,lidy,lidx))-__h2f(lsxz(2*lidz-4,lidy,lidx)))+
//                      HC5*(__h2f(lsxz(2*lidz+4,lidy,lidx))-__h2f(lsxz(2*lidz-5,lidy,lidx))));
//sxz_z.y = (HC1*(__h2f(lsxz(2*lidz+1,lidy,lidx))  -__h2f(lsxz(2*lidz+1-1,lidy,lidx)))+
//                      HC2*(__h2f(lsxz(2*lidz+1+1,lidy,lidx))-__h2f(lsxz(2*lidz+1-2,lidy,lidx)))+
//                      HC3*(__h2f(lsxz(2*lidz+1+2,lidy,lidx))-__h2f(lsxz(2*lidz+1-3,lidy,lidx)))+
//                      HC4*(__h2f(lsxz(2*lidz+1+3,lidy,lidx))-__h2f(lsxz(2*lidz+1-4,lidy,lidx)))+
//                      HC5*(__h2f(lsxz(2*lidz+1+4,lidy,lidx))-__h2f(lsxz(2*lidz+1-5,lidy,lidx))));
//
//        sxz_x.x = (HC1*(__h2f(lsxz(2*lidz,lidy,lidx))  -__h2f(lsxz(2*lidz,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxz(2*lidz,lidy,lidx+1))-__h2f(lsxz(2*lidz,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxz(2*lidz,lidy,lidx+2))-__h2f(lsxz(2*lidz,lidy,lidx-3)))+
//                      HC4*(__h2f(lsxz(2*lidz,lidy,lidx+3))-__h2f(lsxz(2*lidz,lidy,lidx-4)))+
//                      HC5*(__h2f(lsxz(2*lidz,lidy,lidx+4))-__h2f(lsxz(2*lidz,lidy,lidx-5))));
//sxz_x.y = (HC1*(__h2f(lsxz(2*lidz+1,lidy,lidx))  -__h2f(lsxz(2*lidz+1,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxz(2*lidz+1,lidy,lidx+1))-__h2f(lsxz(2*lidz+1,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxz(2*lidz+1,lidy,lidx+2))-__h2f(lsxz(2*lidz+1,lidy,lidx-3)))+
//                      HC4*(__h2f(lsxz(2*lidz+1,lidy,lidx+3))-__h2f(lsxz(2*lidz+1,lidy,lidx-4)))+
//                      HC5*(__h2f(lsxz(2*lidz+1,lidy,lidx+4))-__h2f(lsxz(2*lidz+1,lidy,lidx-5))));
//#elif FDOH ==6
//        sxz_z.x = (HC1*(__h2f(lsxz(2*lidz,lidy,lidx))  -__h2f(lsxz(2*lidz-1,lidy,lidx)))+
//                      HC2*(__h2f(lsxz(2*lidz+1,lidy,lidx))-__h2f(lsxz(2*lidz-2,lidy,lidx)))+
//                      HC3*(__h2f(lsxz(2*lidz+2,lidy,lidx))-__h2f(lsxz(2*lidz-3,lidy,lidx)))+
//                      HC4*(__h2f(lsxz(2*lidz+3,lidy,lidx))-__h2f(lsxz(2*lidz-4,lidy,lidx)))+
//                      HC5*(__h2f(lsxz(2*lidz+4,lidy,lidx))-__h2f(lsxz(2*lidz-5,lidy,lidx)))+
//                      HC6*(__h2f(lsxz(2*lidz+5,lidy,lidx))-__h2f(lsxz(2*lidz-6,lidy,lidx))));
//sxz_z.y = (HC1*(__h2f(lsxz(2*lidz+1,lidy,lidx))  -__h2f(lsxz(2*lidz+1-1,lidy,lidx)))+
//                      HC2*(__h2f(lsxz(2*lidz+1+1,lidy,lidx))-__h2f(lsxz(2*lidz+1-2,lidy,lidx)))+
//                      HC3*(__h2f(lsxz(2*lidz+1+2,lidy,lidx))-__h2f(lsxz(2*lidz+1-3,lidy,lidx)))+
//                      HC4*(__h2f(lsxz(2*lidz+1+3,lidy,lidx))-__h2f(lsxz(2*lidz+1-4,lidy,lidx)))+
//                      HC5*(__h2f(lsxz(2*lidz+1+4,lidy,lidx))-__h2f(lsxz(2*lidz+1-5,lidy,lidx)))+
//                      HC6*(__h2f(lsxz(2*lidz+1+5,lidy,lidx))-__h2f(lsxz(2*lidz+1-6,lidy,lidx))));
//
//
//        sxz_x.x = (HC1*(__h2f(lsxz(2*lidz,lidy,lidx))  -__h2f(lsxz(2*lidz,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxz(2*lidz,lidy,lidx+1))-__h2f(lsxz(2*lidz,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxz(2*lidz,lidy,lidx+2))-__h2f(lsxz(2*lidz,lidy,lidx-3)))+
//                      HC4*(__h2f(lsxz(2*lidz,lidy,lidx+3))-__h2f(lsxz(2*lidz,lidy,lidx-4)))+
//                      HC5*(__h2f(lsxz(2*lidz,lidy,lidx+4))-__h2f(lsxz(2*lidz,lidy,lidx-5)))+
//                      HC6*(__h2f(lsxz(2*lidz,lidy,lidx+5))-__h2f(lsxz(2*lidz,lidy,lidx-6))));
//sxz_x.y = (HC1*(__h2f(lsxz(2*lidz+1,lidy,lidx))  -__h2f(lsxz(2*lidz+1,lidy,lidx-1)))+
//                      HC2*(__h2f(lsxz(2*lidz+1,lidy,lidx+1))-__h2f(lsxz(2*lidz+1,lidy,lidx-2)))+
//                      HC3*(__h2f(lsxz(2*lidz+1,lidy,lidx+2))-__h2f(lsxz(2*lidz+1,lidy,lidx-3)))+
//                      HC4*(__h2f(lsxz(2*lidz+1,lidy,lidx+3))-__h2f(lsxz(2*lidz+1,lidy,lidx-4)))+
//                      HC5*(__h2f(lsxz(2*lidz+1,lidy,lidx+4))-__h2f(lsxz(2*lidz+1,lidy,lidx-5)))+
//                      HC6*(__h2f(lsxz(2*lidz+1,lidy,lidx+5))-__h2f(lsxz(2*lidz+1,lidy,lidx-6))));
//
//#endif
//    }
//
//// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
//#if LOCAL_OFF==0
//#if COMM12==0
//    if ( gidy>(NY-FDOH-1) ||gidz>(NZ-FDOH/2-1) || (gidx-offcomm)>(NX-FDOH-1-LCOMM) ){
//        return;
//    }
//
//#else
//    if ( gidy>(NY-FDOH-1) || gidz>(NZ-FDOH/2-1) ){
//        return;
//    }
//#endif
//#endif
//
//
//
//
//// Correct spatial derivatives to implement CPML
//#if ABS_TYPE==1
//    {
//        int i,j,k, ind;
//
//        if (gidz>NZ-NAB-FDOH-1){
//
//            i =gidx-FDOH;
//            j =gidy-FDOH;
//            k =gidz - NZ+NAB+FDOH+NAB;
//            ind=2*NAB-1-k;
//
//            psi_sxz_z(k,j,i) = b_z[ind+1] * psi_sxz_z(k,j,i) + a_z[ind+1] * sxz_z;
//            sxz_z = sxz_z / K_z[ind+1] + psi_sxz_z(k,j,i);
//            psi_syz_z(k,j,i) = b_z[ind+1] * psi_syz_z(k,j,i) + a_z[ind+1] * syz_z;
//            syz_z = syz_z / K_z[ind+1] + psi_syz_z(k,j,i);
//            psi_szz_z(k,j,i) = b_z_half[ind] * psi_szz_z(k,j,i) + a_z_half[ind] * szz_z;
//            szz_z = szz_z / K_z_half[ind] + psi_szz_z(k,j,i);
//
//        }
//
//#if FREESURF==0
//        else if (gidz-FDOH<NAB){
//
//            i =gidx-FDOH;
//            j =gidy-FDOH;
//            k =gidz-FDOH;
//
//            psi_sxz_z(k,j,i) = b_z[k] * psi_sxz_z(k,j,i) + a_z[k] * sxz_z;
//            sxz_z = sxz_z / K_z[k] + psi_sxz_z(k,j,i);
//            psi_syz_z(k,j,i) = b_z[k] * psi_syz_z(k,j,i) + a_z[k] * syz_z;
//            syz_z = syz_z / K_z[k] + psi_syz_z(k,j,i);
//            psi_szz_z(k,j,i) = b_z_half[k] * psi_szz_z(k,j,i) + a_z_half[k] * szz_z;
//            szz_z = szz_z / K_z_half[k] + psi_szz_z(k,j,i);
//
//        }
//#endif
//
//        if (gidy-FDOH<NAB){
//            i =gidx-FDOH;
//            j =gidy-FDOH;
//            k =gidz-FDOH;
//
//            psi_sxy_y(k,j,i) = b_y[j] * psi_sxy_y(k,j,i) + a_y[j] * sxy_y;
//            sxy_y = sxy_y / K_y[j] + psi_sxy_y(k,j,i);
//            psi_syy_y(k,j,i) = b_y_half[j] * psi_syy_y(k,j,i) + a_y_half[j] * syy_y;
//            syy_y = syy_y / K_y_half[j] + psi_syy_y(k,j,i);
//            psi_syz_y(k,j,i) = b_y[j] * psi_syz_y(k,j,i) + a_y[j] * syz_y;
//            syz_y = syz_y / K_y[j] + psi_syz_y(k,j,i);
//
//        }
//
//        else if (gidy>NY-NAB-FDOH-1){
//
//            i =gidx-FDOH;
//            j =gidy - NY+NAB+FDOH+NAB;
//            k =gidz-FDOH;
//            ind=2*NAB-1-j;
//
//            psi_sxy_y(k,j,i) = b_y[ind+1] * psi_sxy_y(k,j,i) + a_y[ind+1] * sxy_y;
//            sxy_y = sxy_y / K_y[ind+1] + psi_sxy_y(k,j,i);
//            psi_syy_y(k,j,i) = b_y_half[ind] * psi_syy_y(k,j,i) + a_y_half[ind] * syy_y;
//            syy_y = syy_y / K_y_half[ind] + psi_syy_y(k,j,i);
//            psi_syz_y(k,j,i) = b_y[ind+1] * psi_syz_y(k,j,i) + a_y[ind+1] * syz_y;
//            syz_y = syz_y / K_y[ind+1] + psi_syz_y(k,j,i);
//
//
//        }
//#if DEVID==0 & MYLOCALID==0
//        if (gidx-FDOH<NAB){
//
//            i =gidx-FDOH;
//            j =gidy-FDOH;
//            k =gidz-FDOH;
//
//            psi_sxx_x(k,j,i) = b_x_half[i] * psi_sxx_x(k,j,i) + a_x_half[i] * sxx_x;
//            sxx_x = sxx_x / K_x_half[i] + psi_sxx_x(k,j,i);
//            psi_sxy_x(k,j,i) = b_x[i] * psi_sxy_x(k,j,i) + a_x[i] * sxy_x;
//            sxy_x = sxy_x / K_x[i] + psi_sxy_x(k,j,i);
//            psi_sxz_x(k,j,i) = b_x[i] * psi_sxz_x(k,j,i) + a_x[i] * sxz_x;
//            sxz_x = sxz_x / K_x[i] + psi_sxz_x(k,j,i);
//
//        }
//#endif
//
//#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//        if (gidx>NX-NAB-FDOH-1){
//
//            i =gidx - NX+NAB+FDOH+NAB;
//            j =gidy-FDOH;
//            k =gidz-FDOH;
//            ind=2*NAB-1-i;
//
//            psi_sxx_x(k,j,i) = b_x_half[ind] * psi_sxx_x(k,j,i) + a_x_half[ind] * sxx_x;
//            sxx_x = sxx_x / K_x_half[ind] + psi_sxx_x(k,j,i);
//            psi_sxy_x(k,j,i) = b_x[ind+1] * psi_sxy_x(k,j,i) + a_x[ind+1] * sxy_x;
//            sxy_x = sxy_x / K_x[ind+1] + psi_sxy_x(k,j,i);
//            psi_sxz_x(k,j,i) = b_x[ind+1] * psi_sxz_x(k,j,i) + a_x[ind+1] * sxz_x;
//            sxz_x = sxz_x / K_x[ind+1] + psi_sxz_x(k,j,i);
//
//
//
//        }
//#endif
//    }
//#endif
//
//// Update the velocities
//    {
//        float2 lvx = __h22f2(vx(gidz,gidy,gidx));
//        float2 lvy = __h22f2(vy(gidz,gidy,gidx));
//        float2 lvz = __h22f2(vz(gidz,gidy,gidx));
//        float2 lrip = (rip(gidz,gidy,gidx));
//        float2 lrip = (rjp(gidz,gidy,gidx));
//        float2 lrkp = (rkp(gidz,gidy,gidx));
//
//        lvx.x+= (sxx_x.x + sxy_y.x + sxz_z.x)*scalbnf(DTDH/lrip.x, -scaler_sxx);
//        lvx.y+= (sxx_x.y + sxy_y.y + sxz_z.y)*scalbnf(DTDH/lrip.y, -scaler_sxx);
//        lvy.x+= (syy_y.x + sxy_x.x + syz_z.x)*scalbnf(DTDH/lrjp.x, -scaler_sxx);
//        lvy.y+= (syy_y.y + sxy_x.y + syz_z.y)*scalbnf(DTDH/lrjp.y, -scaler_sxx);
//        lvz.x+= (szz_z.x + sxz_x.x + syz_y.x)*scalbnf(DTDH/lrkp.x, -scaler_sxx);
//        lvz.y+= (szz_z.y + sxz_x.y + syz_y.y)*scalbnf(DTDH/lrkp.y, -scaler_sxx);
//    }
//
//// Absorbing boundary
//#if ABS_TYPE==2
//    {
//#if FREESURF==0
//        if (2*gidz-FDOH<NAB){
//            lvx.x*=taper[2*gidz-FDOH];
//            lvx.y*=taper[2*gidz-FDOH];
//            lvy.x*=taper[2*gidz-FDOH];
//            lvy.y*=taper[2*gidz-FDOH];
//            lvz.x*=taper[2*gidz-FDOH];
//            lvz.y*=taper[2*gidz-FDOH];
//        }
//#endif
//
//        if (2*gidz>NZ-NAB-FDOH-1){
//            lvx.x*=taper[2*NZ-FDOH-2*gidz-1];
//            lvx.y*=taper[2*NZ-FDOH-2*gidz-1];
//            lvy.x*=taper[2*NZ-FDOH-2*gidz-1];
//            lvy.y*=taper[2*NZ-FDOH-2*gidz-1];
//            lvz.x*=taper[2*NZ-FDOH-2*gidz-1];
//            lvz.y*=taper[2*NZ-FDOH-2*gidz-1];
//        }
//
//        if (gidy-FDOH<NAB){
//            lvx.x*=taper[gidy-FDOH];
//            lvx.y*=taper[gidy-FDOH];
//            lvy.x*=taper[gidy-FDOH];
//            lvy.y*=taper[gidy-FDOH];
//            lvz.x*=taper[gidy-FDOH];
//            lvz.y*=taper[gidy-FDOH];
//        }
//
//        if (gidy>NY-NAB-FDOH-1){
//            lvx.x*=taper[NY-FDOH-gidy-1];
//            lvx.y*=taper[NY-FDOH-gidy-1];
//            lvy.x*=taper[NY-FDOH-gidy-1];
//            lvy.y*=taper[NY-FDOH-gidy-1];
//            lvz.x*=taper[NY-FDOH-gidy-1];
//            lvz.y*=taper[NY-FDOH-gidy-1];
//        }
//#if DEVID==0 & MYLOCALID==0
//        if (gidx-FDOH<NAB){
//            lvx.x*=taper[gidx-FDOH];
//            lvx.y*=taper[gidx-FDOH];
//            lvy.x*=taper[gidx-FDOH];
//            lvy.y*=taper[gidx-FDOH];
//            lvz.x*=taper[gidx-FDOH];
//            lvz.y*=taper[gidx-FDOH];
//        }
//#endif
//
//#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//        if (gidx>NX-NAB-FDOH-1){
//            lvx.x*=taper[NX-FDOH-gidx-1];
//            lvx.y*=taper[NX-FDOH-gidx-1];
//            lvy.x*=taper[NX-FDOH-gidx-1];
//            lvy.y*=taper[NX-FDOH-gidx-1];
//            lvz.x*=taper[NX-FDOH-gidx-1];
//            lvz.y*=taper[NX-FDOH-gidx-1];
//        }
//#endif
//    }
//#endif
//
//    vx(gidz,gidy,gidx)= __f22h2(lvx);
//    vy(gidz,gidy,gidx)= __f22h2(lvy);
//    vz(gidz,gidy,gidx)= __f22h2(lvz);
//}



#define rip(z,y,x) rip[((x)-FDOH)*(NZ-FDOH)*(NY-2*FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define rjp(z,y,x) rjp[((x)-FDOH)*(NZ-FDOH)*(NY-2*FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define rkp(z,y,x) rkp[((x)-FDOH)*(NZ-FDOH)*(NY-2*FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define muipjp(z,y,x) muipjp[((x)-FDOH)*(NZ-FDOH)*(NY-2*FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define mujpkp(z,y,x) mujpkp[((x)-FDOH)*(NZ-FDOH)*(NY-2*FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define muipkp(z,y,x) muipkp[((x)-FDOH)*(NZ-FDOH)*(NY-2*FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define M(z,y,x) M[((x)-FDOH)*(NZ-FDOH)*(NY-2*FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define mu(z,y,x) mu[((x)-FDOH)*(NZ-FDOH)*(NY-2*FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]

#define sxx(z,y,x) sxx[(x)*NZ*NY+(y)*NZ+(z)]
#define sxy(z,y,x) sxy[(x)*NZ*NY+(y)*NZ+(z)]
#define sxz(z,y,x) sxz[(x)*NZ*NY+(y)*NZ+(z)]
#define syy(z,y,x) syy[(x)*NZ*NY+(y)*NZ+(z)]
#define syz(z,y,x) syz[(x)*NZ*NY+(y)*NZ+(z)]
#define szz(z,y,x) szz[(x)*NZ*NY+(y)*NZ+(z)]
#define vx(z,y,x) vx[(x)*NZ*NY+(y)*NZ+(z)]
#define vy(z,y,x) vy[(x)*NZ*NY+(y)*NZ+(z)]
#define vz(z,y,x) vz[(x)*NZ*NY+(y)*NZ+(z)]

#if LOCAL_OFF==0
#define lvar(z,y,x) lvar[(x)*2*lsizez*lsizey+(y)*2*lsizez+(z)]
#define lvar2(z,y,x) lvar2[(x)*lsizez*lsizey+(y)*lsizez+(z)]
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
#define __h22f2c(x) (x)

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
extern "C" __device__ float2 div2(float2 a, float2 b ){
    
    float2 output;
    output.x = a.x/b.x;
    output.y = a.y/b.y;
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
#define div2 __h2div
#define sub2 __hsub2
#define f2h2 __float2half2_rn
#define __f22h2c(x) __float22half2_rn((x))
#define __h22f2c(x) __half22float2((x))

#endif

extern "C" __device__ __prec2 __hp(__prec *a ){
    
    __prec2 output;
    *((__prec *)&output) = *a;
    *((__prec *)&output+1) = *(a+1);
    return output;
}
extern "C" __device__ float2 scalbnf2(float2 a, int scaler ){
    
    float2 output;
    output.x  = scalbnf(a.x, scaler);
    output.y  = scalbnf(a.y, scaler);
    return output;
}

#if FP16==2 || FP16==4

#define __pprec half2

#else

#define __pprec float2

#endif

#if FP16==2

#define __pconv(x) __half22float2((x))

#elif FP16==3

#define __pconv(x) __float22half2_rn((x))

#else

#define __pconv(x) (x)

#endif



extern "C" __global__ void update_v(int offcomm,
                                    __pprec *rip, __pprec *rjp, __pprec *rkp,
                                    __prec2 *sxx,__prec2 *sxy,__prec2 *sxz,
                                    __prec2 *syy,__prec2 *syz,__prec2 *szz,
                                    __prec2 *vx,__prec2 *vy,__prec2 *vz
                                    )

{
    //Local memory
    extern __shared__ __prec2 lvar2[];
    __prec * lvar=(__prec *)lvar2;
    
    //Grid position
    int lsizez = blockDim.x+FDOH;
    int lsizey = blockDim.y+2*FDOH;
    int lsizex = blockDim.z+2*FDOH;
    int lidz = threadIdx.x+FDOH/2;
    int lidy = threadIdx.y+FDOH;
    int lidx = threadIdx.z+FDOH;
    int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/2;
    int gidy = blockIdx.y*blockDim.y+threadIdx.y+FDOH;
    int gidx = blockIdx.z*blockDim.z+threadIdx.z+FDOH+offcomm;
    

    
    //Define private derivatives
    __cprec sxx_x1;
    __cprec sxy_x2;
    __cprec sxy_y2;
    __cprec sxz_x2;
    __cprec sxz_z2;
    __cprec syy_y1;
    __cprec syz_y2;
    __cprec syz_z2;
    __cprec szz_z1;
    
    //Local memory definitions if local is used
#if LOCAL_OFF==0
#define lszz lvar
#define lsxx lvar
#define lsxz lvar
#define lsyz lvar
#define lsyy lvar
#define lsxy lvar
#define lszz2 lvar2
#define lsxx2 lvar2
#define lsxz2 lvar2
#define lsyz2 lvar2
#define lsyy2 lvar2
#define lsxy2 lvar2
    
    //Local memory definitions if local is not used
#elif LOCAL_OFF==1
    
#define lszz szz
#define lsxx sxx
#define lsxz sxz
#define lsyz syz
#define lsyy syy
#define lsxy sxy
#define lidz gidz
#define lidy gidy
#define lidx gidx
    
#endif
    
    //Calculation of the spatial derivatives
    {
#if LOCAL_OFF==0
//        __syncthreads();
        lszz2(lidz,lidy,lidx)=szz(gidz,gidy,gidx);
        if (lidz<FDOH)
            lszz2(lidz-FDOH/2,lidy,lidx)=szz(gidz-FDOH/2,gidy,gidx);
        if (lidz>(lsizez-FDOH-1))
            lszz2(lidz+FDOH/2,lidy,lidx)=szz(gidz+FDOH/2,gidy,gidx);
//        __syncthreads();
#endif
        
#if   FDOH == 1
        szz_z1=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz,lidy,lidx)))));
#elif FDOH == 2
        szz_z1=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz,lidy,lidx))))),
                    mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidy,lidx))))));
#elif FDOH == 3
        szz_z1=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz,lidy,lidx))))),
                         mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidy,lidx)))))),
                    mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszz(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-2,lidy,lidx))))));
#elif FDOH == 4
        szz_z1=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz,lidy,lidx))))),
                              mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidy,lidx)))))),
                         mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszz(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-2,lidy,lidx)))))),
                    mul2( f2h2(HC4), sub2(__h22f2(__hp(&lszz(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-3,lidy,lidx))))));
#elif FDOH == 5
        szz_z1=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz,lidy,lidx))))),
                                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidy,lidx)))))),
                              mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszz(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-2,lidy,lidx)))))),
                         mul2( f2h2(HC4), sub2(__h22f2(__hp(&lszz(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-3,lidy,lidx)))))),
                    mul2( f2h2(HC5), sub2(__h22f2(__hp(&lszz(2*lidz+5,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-4,lidy,lidx))))));
#elif FDOH == 6
        szz_z1=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lszz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz,lidy,lidx))))),
                                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lszz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-1,lidy,lidx)))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lszz(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-2,lidy,lidx)))))),
                              mul2( f2h2(HC4), sub2(__h22f2(__hp(&lszz(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-3,lidy,lidx)))))),
                         mul2( f2h2(HC5), sub2(__h22f2(__hp(&lszz(2*lidz+5,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-4,lidy,lidx)))))),
                    mul2( f2h2(HC6), sub2(__h22f2(__hp(&lszz(2*lidz+6,lidy,lidx))), __h22f2(__hp(&lszz(2*lidz-5,lidy,lidx))))));
#endif
        
#if LOCAL_OFF==0
//        __syncthreads();
        lsxx2(lidz,lidy,lidx)=sxx(gidz,gidy,gidx);
        if (lidx<2*FDOH)
            lsxx2(lidz,lidy,lidx-FDOH)=sxx(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxx2(lidz,lidy,lidx+lsizex-3*FDOH)=sxx(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxx2(lidz,lidy,lidx+FDOH)=sxx(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxx2(lidz,lidy,lidx-lsizex+3*FDOH)=sxx(gidz,gidy,gidx-lsizex+3*FDOH);
//        __syncthreads();
#endif
        
#if   FDOH == 1
        sxx_x1=mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidy,lidx+1)), __h22f2(lsxx2(lidz,lidy,lidx))));
#elif FDOH == 2
        sxx_x1=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidy,lidx+1)), __h22f2(lsxx2(lidz,lidy,lidx)))),
                    mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidy,lidx+2)), __h22f2(lsxx2(lidz,lidy,lidx-1)))));
#elif FDOH == 3
        sxx_x1=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidy,lidx+1)), __h22f2(lsxx2(lidz,lidy,lidx)))),
                         mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidy,lidx+2)), __h22f2(lsxx2(lidz,lidy,lidx-1))))),
                    mul2( f2h2(HC3), sub2(__h22f2(lsxx2(lidz,lidy,lidx+3)), __h22f2(lsxx2(lidz,lidy,lidx-2)))));
#elif FDOH == 4
        sxx_x1=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidy,lidx+1)), __h22f2(lsxx2(lidz,lidy,lidx)))),
                              mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidy,lidx+2)), __h22f2(lsxx2(lidz,lidy,lidx-1))))),
                         mul2( f2h2(HC3), sub2(__h22f2(lsxx2(lidz,lidy,lidx+3)), __h22f2(lsxx2(lidz,lidy,lidx-2))))),
                    mul2( f2h2(HC4), sub2(__h22f2(lsxx2(lidz,lidy,lidx+4)), __h22f2(lsxx2(lidz,lidy,lidx-3)))));
#elif FDOH == 5
        sxx_x1=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidy,lidx+1)), __h22f2(lsxx2(lidz,lidy,lidx)))),
                                   mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidy,lidx+2)), __h22f2(lsxx2(lidz,lidy,lidx-1))))),
                              mul2( f2h2(HC3), sub2(__h22f2(lsxx2(lidz,lidy,lidx+3)), __h22f2(lsxx2(lidz,lidy,lidx-2))))),
                         mul2( f2h2(HC4), sub2(__h22f2(lsxx2(lidz,lidy,lidx+4)), __h22f2(lsxx2(lidz,lidy,lidx-3))))),
                    mul2( f2h2(HC5), sub2(__h22f2(lsxx2(lidz,lidy,lidx+5)), __h22f2(lsxx2(lidz,lidy,lidx-4)))));
#elif FDOH == 6
        sxx_x1=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(lsxx2(lidz,lidy,lidx+1)), __h22f2(lsxx2(lidz,lidy,lidx)))),
                                        mul2( f2h2(HC2), sub2(__h22f2(lsxx2(lidz,lidy,lidx+2)), __h22f2(lsxx2(lidz,lidy,lidx-1))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(lsxx2(lidz,lidy,lidx+3)), __h22f2(lsxx2(lidz,lidy,lidx-2))))),
                              mul2( f2h2(HC4), sub2(__h22f2(lsxx2(lidz,lidy,lidx+4)), __h22f2(lsxx2(lidz,lidy,lidx-3))))),
                         mul2( f2h2(HC5), sub2(__h22f2(lsxx2(lidz,lidy,lidx+5)), __h22f2(lsxx2(lidz,lidy,lidx-4))))),
                    mul2( f2h2(HC6), sub2(__h22f2(lsxx2(lidz,lidy,lidx+6)), __h22f2(lsxx2(lidz,lidy,lidx-5)))));
#endif
        
#if LOCAL_OFF==0
//        __syncthreads();
        lsxz2(lidz,lidy,lidx)=sxz(gidz,gidy,gidx);
        if (lidz<FDOH)
            lsxz2(lidz-FDOH/2,lidy,lidx)=sxz(gidz-FDOH/2,gidy,gidx);
        if (lidz>(lsizez-FDOH-1))
            lsxz2(lidz+FDOH/2,lidy,lidx)=sxz(gidz+FDOH/2,gidy,gidx);
        if (lidx<2*FDOH)
            lsxz2(lidz,lidy,lidx-FDOH)=sxz(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxz2(lidz,lidy,lidx+lsizex-3*FDOH)=sxz(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxz2(lidz,lidy,lidx+FDOH)=sxz(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxz2(lidz,lidy,lidx-lsizex+3*FDOH)=sxz(gidz,gidy,gidx-lsizex+3*FDOH);
//        __syncthreads();
#endif
        
#if   FDOH == 1
        sxz_x2=mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidy,lidx)), __h22f2(lsxz2(lidz,lidy,lidx-1))));
#elif FDOH == 2
        sxz_x2=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidy,lidx)), __h22f2(lsxz2(lidz,lidy,lidx-1)))),
                    mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidy,lidx+1)), __h22f2(lsxz2(lidz,lidy,lidx-2)))));
#elif FDOH == 3
        sxz_x2=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidy,lidx)), __h22f2(lsxz2(lidz,lidy,lidx-1)))),
                         mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidy,lidx+1)), __h22f2(lsxz2(lidz,lidy,lidx-2))))),
                    mul2( f2h2(HC3), sub2(__h22f2(lsxz2(lidz,lidy,lidx+2)), __h22f2(lsxz2(lidz,lidy,lidx-3)))));
#elif FDOH == 4
        sxz_x2=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidy,lidx)), __h22f2(lsxz2(lidz,lidy,lidx-1)))),
                              mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidy,lidx+1)), __h22f2(lsxz2(lidz,lidy,lidx-2))))),
                         mul2( f2h2(HC3), sub2(__h22f2(lsxz2(lidz,lidy,lidx+2)), __h22f2(lsxz2(lidz,lidy,lidx-3))))),
                    mul2( f2h2(HC4), sub2(__h22f2(lsxz2(lidz,lidy,lidx+3)), __h22f2(lsxz2(lidz,lidy,lidx-4)))));
#elif FDOH == 5
        sxz_x2=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidy,lidx)), __h22f2(lsxz2(lidz,lidy,lidx-1)))),
                                   mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidy,lidx+1)), __h22f2(lsxz2(lidz,lidy,lidx-2))))),
                              mul2( f2h2(HC3), sub2(__h22f2(lsxz2(lidz,lidy,lidx+2)), __h22f2(lsxz2(lidz,lidy,lidx-3))))),
                         mul2( f2h2(HC4), sub2(__h22f2(lsxz2(lidz,lidy,lidx+3)), __h22f2(lsxz2(lidz,lidy,lidx-4))))),
                    mul2( f2h2(HC5), sub2(__h22f2(lsxz2(lidz,lidy,lidx+4)), __h22f2(lsxz2(lidz,lidy,lidx-5)))));
#elif FDOH == 6
        sxz_x2=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(lsxz2(lidz,lidy,lidx)), __h22f2(lsxz2(lidz,lidy,lidx-1)))),
                                        mul2( f2h2(HC2), sub2(__h22f2(lsxz2(lidz,lidy,lidx+1)), __h22f2(lsxz2(lidz,lidy,lidx-2))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(lsxz2(lidz,lidy,lidx+2)), __h22f2(lsxz2(lidz,lidy,lidx-3))))),
                              mul2( f2h2(HC4), sub2(__h22f2(lsxz2(lidz,lidy,lidx+3)), __h22f2(lsxz2(lidz,lidy,lidx-4))))),
                         mul2( f2h2(HC5), sub2(__h22f2(lsxz2(lidz,lidy,lidx+4)), __h22f2(lsxz2(lidz,lidy,lidx-5))))),
                    mul2( f2h2(HC6), sub2(__h22f2(lsxz2(lidz,lidy,lidx+5)), __h22f2(lsxz2(lidz,lidy,lidx-6)))));
#endif
        
#if   FDOH == 1
        sxz_z2=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidy,lidx)))));
#elif FDOH == 2
        sxz_z2=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidy,lidx))))),
                    mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidy,lidx))))));
#elif FDOH == 3
        sxz_z2=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidy,lidx))))),
                         mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidy,lidx)))))),
                    mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-3,lidy,lidx))))));
#elif FDOH == 4
        sxz_z2=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidy,lidx))))),
                              mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidy,lidx)))))),
                         mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-3,lidy,lidx)))))),
                    mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsxz(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-4,lidy,lidx))))));
#elif FDOH == 5
        sxz_z2=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidy,lidx))))),
                                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidy,lidx)))))),
                              mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-3,lidy,lidx)))))),
                         mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsxz(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-4,lidy,lidx)))))),
                    mul2( f2h2(HC5), sub2(__h22f2(__hp(&lsxz(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-5,lidy,lidx))))));
#elif FDOH == 6
        sxz_z2=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsxz(2*lidz,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-1,lidy,lidx))))),
                                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsxz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-2,lidy,lidx)))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsxz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-3,lidy,lidx)))))),
                              mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsxz(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-4,lidy,lidx)))))),
                         mul2( f2h2(HC5), sub2(__h22f2(__hp(&lsxz(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-5,lidy,lidx)))))),
                    mul2( f2h2(HC6), sub2(__h22f2(__hp(&lsxz(2*lidz+5,lidy,lidx))), __h22f2(__hp(&lsxz(2*lidz-6,lidy,lidx))))));
#endif
        
#if LOCAL_OFF==0
//        __syncthreads();
        lsyz2(lidz,lidy,lidx)=syz(gidz,gidy,gidx);
        if (lidz<FDOH)
            lsyz2(lidz-FDOH/2,lidy,lidx)=syz(gidz-FDOH/2,gidy,gidx);
        if (lidz>(lsizez-FDOH-1))
            lsyz2(lidz+FDOH/2,lidy,lidx)=syz(gidz+FDOH/2,gidy,gidx);
        if (lidy<2*FDOH)
            lsyz2(lidz,lidy-FDOH,lidx)=syz(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lsyz2(lidz,lidy+lsizey-3*FDOH,lidx)=syz(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lsyz2(lidz,lidy+FDOH,lidx)=syz(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lsyz2(lidz,lidy-lsizey+3*FDOH,lidx)=syz(gidz,gidy-lsizey+3*FDOH,gidx);
//        __syncthreads();
#endif
        
#if   FDOH == 1
        syz_y2=mul2( f2h2(HC1), sub2(__h22f2(lsyz2(lidz,lidy,lidx)), __h22f2(lsyz2(lidz,lidy-1,lidx))));
#elif FDOH == 2
        syz_y2=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(lsyz2(lidz,lidy,lidx)), __h22f2(lsyz2(lidz,lidy-1,lidx)))),
                    mul2( f2h2(HC2), sub2(__h22f2(lsyz2(lidz,lidy+1,lidx)), __h22f2(lsyz2(lidz,lidy-2,lidx)))));
#elif FDOH == 3
        syz_y2=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(lsyz2(lidz,lidy,lidx)), __h22f2(lsyz2(lidz,lidy-1,lidx)))),
                         mul2( f2h2(HC2), sub2(__h22f2(lsyz2(lidz,lidy+1,lidx)), __h22f2(lsyz2(lidz,lidy-2,lidx))))),
                    mul2( f2h2(HC3), sub2(__h22f2(lsyz2(lidz,lidy+2,lidx)), __h22f2(lsyz2(lidz,lidy-3,lidx)))));
#elif FDOH == 4
        syz_y2=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(lsyz2(lidz,lidy,lidx)), __h22f2(lsyz2(lidz,lidy-1,lidx)))),
                              mul2( f2h2(HC2), sub2(__h22f2(lsyz2(lidz,lidy+1,lidx)), __h22f2(lsyz2(lidz,lidy-2,lidx))))),
                         mul2( f2h2(HC3), sub2(__h22f2(lsyz2(lidz,lidy+2,lidx)), __h22f2(lsyz2(lidz,lidy-3,lidx))))),
                    mul2( f2h2(HC4), sub2(__h22f2(lsyz2(lidz,lidy+3,lidx)), __h22f2(lsyz2(lidz,lidy-4,lidx)))));
#elif FDOH == 5
        syz_y2=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(lsyz2(lidz,lidy,lidx)), __h22f2(lsyz2(lidz,lidy-1,lidx)))),
                                   mul2( f2h2(HC2), sub2(__h22f2(lsyz2(lidz,lidy+1,lidx)), __h22f2(lsyz2(lidz,lidy-2,lidx))))),
                              mul2( f2h2(HC3), sub2(__h22f2(lsyz2(lidz,lidy+2,lidx)), __h22f2(lsyz2(lidz,lidy-3,lidx))))),
                         mul2( f2h2(HC4), sub2(__h22f2(lsyz2(lidz,lidy+3,lidx)), __h22f2(lsyz2(lidz,lidy-4,lidx))))),
                    mul2( f2h2(HC5), sub2(__h22f2(lsyz2(lidz,lidy+4,lidx)), __h22f2(lsyz2(lidz,lidy-5,lidx)))));
#elif FDOH == 6
        syz_y2=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(lsyz2(lidz,lidy,lidx)), __h22f2(lsyz2(lidz,lidy-1,lidx)))),
                                        mul2( f2h2(HC2), sub2(__h22f2(lsyz2(lidz,lidy+1,lidx)), __h22f2(lsyz2(lidz,lidy-2,lidx))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(lsyz2(lidz,lidy+2,lidx)), __h22f2(lsyz2(lidz,lidy-3,lidx))))),
                              mul2( f2h2(HC4), sub2(__h22f2(lsyz2(lidz,lidy+3,lidx)), __h22f2(lsyz2(lidz,lidy-4,lidx))))),
                         mul2( f2h2(HC5), sub2(__h22f2(lsyz2(lidz,lidy+4,lidx)), __h22f2(lsyz2(lidz,lidy-5,lidx))))),
                    mul2( f2h2(HC6), sub2(__h22f2(lsyz2(lidz,lidy+5,lidx)), __h22f2(lsyz2(lidz,lidy-6,lidx)))));
#endif
        
#if   FDOH == 1
        syz_z2=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsyz(2*lidz,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-1,lidy,lidx)))));
#elif FDOH == 2
        syz_z2=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsyz(2*lidz,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-1,lidy,lidx))))),
                    mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsyz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-2,lidy,lidx))))));
#elif FDOH == 3
        syz_z2=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsyz(2*lidz,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-1,lidy,lidx))))),
                         mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsyz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-2,lidy,lidx)))))),
                    mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsyz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-3,lidy,lidx))))));
#elif FDOH == 4
        syz_z2=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsyz(2*lidz,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-1,lidy,lidx))))),
                              mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsyz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-2,lidy,lidx)))))),
                         mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsyz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-3,lidy,lidx)))))),
                    mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsyz(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-4,lidy,lidx))))));
#elif FDOH == 5
        syz_z2=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsyz(2*lidz,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-1,lidy,lidx))))),
                                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsyz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-2,lidy,lidx)))))),
                              mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsyz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-3,lidy,lidx)))))),
                         mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsyz(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-4,lidy,lidx)))))),
                    mul2( f2h2(HC5), sub2(__h22f2(__hp(&lsyz(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-5,lidy,lidx))))));
#elif FDOH == 6
        syz_z2=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lsyz(2*lidz,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-1,lidy,lidx))))),
                                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lsyz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-2,lidy,lidx)))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lsyz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-3,lidy,lidx)))))),
                              mul2( f2h2(HC4), sub2(__h22f2(__hp(&lsyz(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-4,lidy,lidx)))))),
                         mul2( f2h2(HC5), sub2(__h22f2(__hp(&lsyz(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-5,lidy,lidx)))))),
                    mul2( f2h2(HC6), sub2(__h22f2(__hp(&lsyz(2*lidz+5,lidy,lidx))), __h22f2(__hp(&lsyz(2*lidz-6,lidy,lidx))))));
#endif
        
#if LOCAL_OFF==0
//        __syncthreads();
        lsyy2(lidz,lidy,lidx)=syy(gidz,gidy,gidx);
        if (lidy<2*FDOH)
            lsyy2(lidz,lidy-FDOH,lidx)=syy(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lsyy2(lidz,lidy+lsizey-3*FDOH,lidx)=syy(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lsyy2(lidz,lidy+FDOH,lidx)=syy(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lsyy2(lidz,lidy-lsizey+3*FDOH,lidx)=syy(gidz,gidy-lsizey+3*FDOH,gidx);
//        __syncthreads();
#endif
        
#if   FDOH == 1
        syy_y1=mul2( f2h2(HC1), sub2(__h22f2(lsyy2(lidz,lidy+1,lidx)), __h22f2(lsyy2(lidz,lidy,lidx))));
#elif FDOH == 2
        syy_y1=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(lsyy2(lidz,lidy+1,lidx)), __h22f2(lsyy2(lidz,lidy,lidx)))),
                    mul2( f2h2(HC2), sub2(__h22f2(lsyy2(lidz,lidy+2,lidx)), __h22f2(lsyy2(lidz,lidy-1,lidx)))));
#elif FDOH == 3
        syy_y1=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(lsyy2(lidz,lidy+1,lidx)), __h22f2(lsyy2(lidz,lidy,lidx)))),
                         mul2( f2h2(HC2), sub2(__h22f2(lsyy2(lidz,lidy+2,lidx)), __h22f2(lsyy2(lidz,lidy-1,lidx))))),
                    mul2( f2h2(HC3), sub2(__h22f2(lsyy2(lidz,lidy+3,lidx)), __h22f2(lsyy2(lidz,lidy-2,lidx)))));
#elif FDOH == 4
        syy_y1=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(lsyy2(lidz,lidy+1,lidx)), __h22f2(lsyy2(lidz,lidy,lidx)))),
                              mul2( f2h2(HC2), sub2(__h22f2(lsyy2(lidz,lidy+2,lidx)), __h22f2(lsyy2(lidz,lidy-1,lidx))))),
                         mul2( f2h2(HC3), sub2(__h22f2(lsyy2(lidz,lidy+3,lidx)), __h22f2(lsyy2(lidz,lidy-2,lidx))))),
                    mul2( f2h2(HC4), sub2(__h22f2(lsyy2(lidz,lidy+4,lidx)), __h22f2(lsyy2(lidz,lidy-3,lidx)))));
#elif FDOH == 5
        syy_y1=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(lsyy2(lidz,lidy+1,lidx)), __h22f2(lsyy2(lidz,lidy,lidx)))),
                                   mul2( f2h2(HC2), sub2(__h22f2(lsyy2(lidz,lidy+2,lidx)), __h22f2(lsyy2(lidz,lidy-1,lidx))))),
                              mul2( f2h2(HC3), sub2(__h22f2(lsyy2(lidz,lidy+3,lidx)), __h22f2(lsyy2(lidz,lidy-2,lidx))))),
                         mul2( f2h2(HC4), sub2(__h22f2(lsyy2(lidz,lidy+4,lidx)), __h22f2(lsyy2(lidz,lidy-3,lidx))))),
                    mul2( f2h2(HC5), sub2(__h22f2(lsyy2(lidz,lidy+5,lidx)), __h22f2(lsyy2(lidz,lidy-4,lidx)))));
#elif FDOH == 6
        syy_y1=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(lsyy2(lidz,lidy+1,lidx)), __h22f2(lsyy2(lidz,lidy,lidx)))),
                                        mul2( f2h2(HC2), sub2(__h22f2(lsyy2(lidz,lidy+2,lidx)), __h22f2(lsyy2(lidz,lidy-1,lidx))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(lsyy2(lidz,lidy+3,lidx)), __h22f2(lsyy2(lidz,lidy-2,lidx))))),
                              mul2( f2h2(HC4), sub2(__h22f2(lsyy2(lidz,lidy+4,lidx)), __h22f2(lsyy2(lidz,lidy-3,lidx))))),
                         mul2( f2h2(HC5), sub2(__h22f2(lsyy2(lidz,lidy+5,lidx)), __h22f2(lsyy2(lidz,lidy-4,lidx))))),
                    mul2( f2h2(HC6), sub2(__h22f2(lsyy2(lidz,lidy+6,lidx)), __h22f2(lsyy2(lidz,lidy-5,lidx)))));
#endif
        
#if LOCAL_OFF==0
//        __syncthreads();
        lsxy2(lidz,lidy,lidx)=sxy(gidz,gidy,gidx);
        if (lidy<2*FDOH)
            lsxy2(lidz,lidy-FDOH,lidx)=sxy(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lsxy2(lidz,lidy+lsizey-3*FDOH,lidx)=sxy(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lsxy2(lidz,lidy+FDOH,lidx)=sxy(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lsxy2(lidz,lidy-lsizey+3*FDOH,lidx)=sxy(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lsxy2(lidz,lidy,lidx-FDOH)=sxy(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxy2(lidz,lidy,lidx+lsizex-3*FDOH)=sxy(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxy2(lidz,lidy,lidx+FDOH)=sxy(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxy2(lidz,lidy,lidx-lsizex+3*FDOH)=sxy(gidz,gidy,gidx-lsizex+3*FDOH);
//        __syncthreads();
#endif
        
#if   FDOH == 1
        sxy_x2=mul2( f2h2(HC1), sub2(__h22f2(lsxy2(lidz,lidy,lidx)), __h22f2(lsxy2(lidz,lidy,lidx-1))));
#elif FDOH == 2
        sxy_x2=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(lsxy2(lidz,lidy,lidx)), __h22f2(lsxy2(lidz,lidy,lidx-1)))),
                    mul2( f2h2(HC2), sub2(__h22f2(lsxy2(lidz,lidy,lidx+1)), __h22f2(lsxy2(lidz,lidy,lidx-2)))));
#elif FDOH == 3
        sxy_x2=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(lsxy2(lidz,lidy,lidx)), __h22f2(lsxy2(lidz,lidy,lidx-1)))),
                         mul2( f2h2(HC2), sub2(__h22f2(lsxy2(lidz,lidy,lidx+1)), __h22f2(lsxy2(lidz,lidy,lidx-2))))),
                    mul2( f2h2(HC3), sub2(__h22f2(lsxy2(lidz,lidy,lidx+2)), __h22f2(lsxy2(lidz,lidy,lidx-3)))));
#elif FDOH == 4
        sxy_x2=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(lsxy2(lidz,lidy,lidx)), __h22f2(lsxy2(lidz,lidy,lidx-1)))),
                              mul2( f2h2(HC2), sub2(__h22f2(lsxy2(lidz,lidy,lidx+1)), __h22f2(lsxy2(lidz,lidy,lidx-2))))),
                         mul2( f2h2(HC3), sub2(__h22f2(lsxy2(lidz,lidy,lidx+2)), __h22f2(lsxy2(lidz,lidy,lidx-3))))),
                    mul2( f2h2(HC4), sub2(__h22f2(lsxy2(lidz,lidy,lidx+3)), __h22f2(lsxy2(lidz,lidy,lidx-4)))));
#elif FDOH == 5
        sxy_x2=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(lsxy2(lidz,lidy,lidx)), __h22f2(lsxy2(lidz,lidy,lidx-1)))),
                                   mul2( f2h2(HC2), sub2(__h22f2(lsxy2(lidz,lidy,lidx+1)), __h22f2(lsxy2(lidz,lidy,lidx-2))))),
                              mul2( f2h2(HC3), sub2(__h22f2(lsxy2(lidz,lidy,lidx+2)), __h22f2(lsxy2(lidz,lidy,lidx-3))))),
                         mul2( f2h2(HC4), sub2(__h22f2(lsxy2(lidz,lidy,lidx+3)), __h22f2(lsxy2(lidz,lidy,lidx-4))))),
                    mul2( f2h2(HC5), sub2(__h22f2(lsxy2(lidz,lidy,lidx+4)), __h22f2(lsxy2(lidz,lidy,lidx-5)))));
#elif FDOH == 6
        sxy_x2=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(lsxy2(lidz,lidy,lidx)), __h22f2(lsxy2(lidz,lidy,lidx-1)))),
                                        mul2( f2h2(HC2), sub2(__h22f2(lsxy2(lidz,lidy,lidx+1)), __h22f2(lsxy2(lidz,lidy,lidx-2))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(lsxy2(lidz,lidy,lidx+2)), __h22f2(lsxy2(lidz,lidy,lidx-3))))),
                              mul2( f2h2(HC4), sub2(__h22f2(lsxy2(lidz,lidy,lidx+3)), __h22f2(lsxy2(lidz,lidy,lidx-4))))),
                         mul2( f2h2(HC5), sub2(__h22f2(lsxy2(lidz,lidy,lidx+4)), __h22f2(lsxy2(lidz,lidy,lidx-5))))),
                    mul2( f2h2(HC6), sub2(__h22f2(lsxy2(lidz,lidy,lidx+5)), __h22f2(lsxy2(lidz,lidy,lidx-6)))));
#endif
        
#if   FDOH == 1
        sxy_y2=mul2( f2h2(HC1), sub2(__h22f2(lsxy2(lidz,lidy,lidx)), __h22f2(lsxy2(lidz,lidy-1,lidx))));
#elif FDOH == 2
        sxy_y2=add2(
                    mul2( f2h2(HC1), sub2(__h22f2(lsxy2(lidz,lidy,lidx)), __h22f2(lsxy2(lidz,lidy-1,lidx)))),
                    mul2( f2h2(HC2), sub2(__h22f2(lsxy2(lidz,lidy+1,lidx)), __h22f2(lsxy2(lidz,lidy-2,lidx)))));
#elif FDOH == 3
        sxy_y2=add2(add2(
                         mul2( f2h2(HC1), sub2(__h22f2(lsxy2(lidz,lidy,lidx)), __h22f2(lsxy2(lidz,lidy-1,lidx)))),
                         mul2( f2h2(HC2), sub2(__h22f2(lsxy2(lidz,lidy+1,lidx)), __h22f2(lsxy2(lidz,lidy-2,lidx))))),
                    mul2( f2h2(HC3), sub2(__h22f2(lsxy2(lidz,lidy+2,lidx)), __h22f2(lsxy2(lidz,lidy-3,lidx)))));
#elif FDOH == 4
        sxy_y2=add2(add2(add2(
                              mul2( f2h2(HC1), sub2(__h22f2(lsxy2(lidz,lidy,lidx)), __h22f2(lsxy2(lidz,lidy-1,lidx)))),
                              mul2( f2h2(HC2), sub2(__h22f2(lsxy2(lidz,lidy+1,lidx)), __h22f2(lsxy2(lidz,lidy-2,lidx))))),
                         mul2( f2h2(HC3), sub2(__h22f2(lsxy2(lidz,lidy+2,lidx)), __h22f2(lsxy2(lidz,lidy-3,lidx))))),
                    mul2( f2h2(HC4), sub2(__h22f2(lsxy2(lidz,lidy+3,lidx)), __h22f2(lsxy2(lidz,lidy-4,lidx)))));
#elif FDOH == 5
        sxy_y2=add2(add2(add2(add2(
                                   mul2( f2h2(HC1), sub2(__h22f2(lsxy2(lidz,lidy,lidx)), __h22f2(lsxy2(lidz,lidy-1,lidx)))),
                                   mul2( f2h2(HC2), sub2(__h22f2(lsxy2(lidz,lidy+1,lidx)), __h22f2(lsxy2(lidz,lidy-2,lidx))))),
                              mul2( f2h2(HC3), sub2(__h22f2(lsxy2(lidz,lidy+2,lidx)), __h22f2(lsxy2(lidz,lidy-3,lidx))))),
                         mul2( f2h2(HC4), sub2(__h22f2(lsxy2(lidz,lidy+3,lidx)), __h22f2(lsxy2(lidz,lidy-4,lidx))))),
                    mul2( f2h2(HC5), sub2(__h22f2(lsxy2(lidz,lidy+4,lidx)), __h22f2(lsxy2(lidz,lidy-5,lidx)))));
#elif FDOH == 6
        sxy_y2=add2(add2(add2(add2(add2(
                                        mul2( f2h2(HC1), sub2(__h22f2(lsxy2(lidz,lidy,lidx)), __h22f2(lsxy2(lidz,lidy-1,lidx)))),
                                        mul2( f2h2(HC2), sub2(__h22f2(lsxy2(lidz,lidy+1,lidx)), __h22f2(lsxy2(lidz,lidy-2,lidx))))),
                                   mul2( f2h2(HC3), sub2(__h22f2(lsxy2(lidz,lidy+2,lidx)), __h22f2(lsxy2(lidz,lidy-3,lidx))))),
                              mul2( f2h2(HC4), sub2(__h22f2(lsxy2(lidz,lidy+3,lidx)), __h22f2(lsxy2(lidz,lidy-4,lidx))))),
                         mul2( f2h2(HC5), sub2(__h22f2(lsxy2(lidz,lidy+4,lidx)), __h22f2(lsxy2(lidz,lidy-5,lidx))))),
                    mul2( f2h2(HC6), sub2(__h22f2(lsxy2(lidz,lidy+5,lidx)), __h22f2(lsxy2(lidz,lidy-6,lidx)))));
#endif
        
    }
    // To stop updating if we are outside the model (global id must be amultiple of local id in OpenCL, hence we stop if we have a global idoutside the grid)
#if  LOCAL_OFF==0
#if COMM12==0
    if ( gidz>(NZ-FDOH/2-1) ||  gidy>(NY-FDOH-1) ||  (gidx-offcomm)>(NX-FDOH-1-LCOMM) )
        return;
#else
    if ( gidz>(NZ-FDOH/2-1) ||  gidy>(NY-FDOH-1)  )
        return;
#endif
#endif
    
    //Define and load private parameters and variables
    __cprec lvx = __h22f2(vx(gidz,gidy,gidx));
    __cprec lvy = __h22f2(vy(gidz,gidy,gidx));
    __cprec lvz = __h22f2(vz(gidz,gidy,gidx));
    __cprec lrip = __pconv(rip(gidz,gidy,gidx));
    __cprec lrjp = __pconv(rjp(gidz,gidy,gidx));
    __cprec lrkp = __pconv(rkp(gidz,gidy,gidx));
    
    // Update the variables
    lvx=add2(lvx,mul2(add2(add2(sxx_x1,sxy_y2),sxz_z2),lrip));
    lvy=add2(lvy,mul2(add2(add2(syy_y1,sxy_x2),syz_z2),lrjp));
    lvz=add2(lvz,mul2(add2(add2(szz_z1,sxz_x2),syz_y2),lrkp));
    //Write updated values to global memory
    vx(gidz,gidy,gidx) = __f22h2(lvx);
    vy(gidz,gidy,gidx) = __f22h2(lvy);
    vz(gidz,gidy,gidx) = __f22h2(lvz);
    
    
}



