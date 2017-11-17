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
///*Update of the stresses in 3D*/
//
///*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
//#define rho(z,y,x)     rho[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define rip(z,y,x)     rip[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define rjp(z,y,x)     rjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define rkp(z,y,x)     rkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define muipjp(z,y,x) muipjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define mujpkp(z,y,x) mujpkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define muipkp(z,y,x) muipkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define mu(z,y,x)         mu[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
//#define M(z,y,x)       M[((x)-FDOH)*(NY-2*FDOH)*(NZ-FDOH)+((y)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
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
//#define psi_vxx(z,y,x) psi_vxx[(x)*(NY-2*FDOH)*(NZ-FDOH)+(y)*(NZ-FDOH)+(z)]
//#define psi_vyx(z,y,x) psi_vyx[(x)*(NY-2*FDOH)*(NZ-FDOH)+(y)*(NZ-FDOH)+(z)]
//#define psi_vzx(z,y,x) psi_vzx[(x)*(NY-2*FDOH)*(NZ-FDOH)+(y)*(NZ-FDOH)+(z)]
//
//#define psi_vxy(z,y,x) psi_vxy[(x)*(2*NAB)*(NZ-FDOH)+(y)*(NZ-FDOH)+(z)]
//#define psi_vyy(z,y,x) psi_vyy[(x)*(2*NAB)*(NZ-FDOH)+(y)*(NZ-FDOH)+(z)]
//#define psi_vzy(z,y,x) psi_vzy[(x)*(2*NAB)*(NZ-FDOH)+(y)*(NZ-FDOH)+(z)]
//
//#define psi_vxz(z,y,x) psi_vxz[(x)*(NY-2*FDOH)*(NAB)+(y)*(NAB)+(z)]
//#define psi_vyz(z,y,x) psi_vyz[(x)*(NY-2*FDOH)*(NAB)+(y)*(NAB)+(z)]
//#define psi_vzz(z,y,x) psi_vzz[(x)*(NY-2*FDOH)*(NAB)+(y)*(NAB)+(z)]
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
//extern "C" __global__ update_s(int offcomm,
//                       __prec2*vx,         __prec2*vy,            __prec2*vz,
//                       __prec2*sxx,        __prec2*syy,           __prec2*szz,
//                       __prec2*sxy,        __prec2*syz,           __prec2*sxz,
//                       float *M,         float *mu,             float *muipjp,
//                       float *mujpkp,      float *muipkp,
//                       __prec2*rxx,        __prec2*ryy,           __prec2*rzz,
//                       __prec2*rxy,        __prec2*ryz,           __prec2*rxz,
//                       float *taus,       float *tausipjp,      float *tausjpkp,
//                       float *tausipkp,   float *taup,          float *eta,
//                       float *taper,
//                       float *K_x,        float *a_x,          float *b_x,
//                       float *K_x_half,   float *a_x_half,     float *b_x_half,
//                       float *K_y,        float *a_y,          float *b_y,
//                       float *K_y_half,   float *a_y_half,     float *b_y_half,
//                       float *K_z,        float *a_z,          float *b_z,
//                       float *K_z_half,   float *a_z_half,     float *b_z_half,
//                       __prec2*psi_vxx,    __prec2*psi_vxy,       __prec2*psi_vxz,
//                       __prec2*psi_vyx,    __prec2*psi_vyy,       __prec2*psi_vyz,
//                       __prec2*psi_vzx,    __prec2*psi_vzy,       __prec2*psi_vzz,
//                       int scaler_sxx)
//
//{
//
//    /* Standard staggered grid kernel, finite difference order of 4.  */
//
//    int i,j,k,l,ind;
//    float2 fipjp, fjpkp, fipkp, f, g;
//    float2 sumrxy,sumryz,sumrxz,sumrxx,sumryy,sumrzz;
//    float2 e,d,dipjp,djpkp,dipkp;
//    float b,c;
//    float lM, lmu, lmuipjp, lmuipkp, lmujpkp, ltaup, ltaus, ltausipjp, ltausipkp, ltausjpkp;
//#if LVE>0
//    float leta[LVE];
//#endif
//    float2 vxx,vxy,vxz,vyx,vyy,vyz,vzx,vzy,vzz;
//    float2 vxyyx,vyzzy,vxzzx,vxxyyzz,vyyzz,vxxzz,vxxyy;
//
//    float2 lsxx, lsyy, lszz, lsxy, lsxz, lsyz;
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
//#define lvx lvar
//#define lvy lvar
//#define lvz lvar
//
//#define lvx2 lvar2
//#define lvy2 lvar2
//#define lvz2 lvar2
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
//#define lvx vx
//#define lvy vy
//#define lvz vz
//#define lidx gidx
//#define lidy gidy
//#define lidz gidz
//
//#endif
//
//// Calculation of the velocity spatial derivatives
//    {
//#if LOCAL_OFF==0
//        lvx2(lidz,lidy,lidx)=vx(gidz, gidy, gidx);
//        if (lidy<2*FDOH)
//            lvx2(lidz,lidy-FDOH,lidx)=vx(gidz,gidy-FDOH,gidx);
//        if (lidy+lsizey-3*FDOH<FDOH)
//            lvx2(lidz,lidy+lsizey-3*FDOH,lidx)=vx(gidz,gidy+lsizey-3*FDOH,gidx);
//        if (lidy>(lsizey-2*FDOH-1))
//            lvx2(lidz,lidy+FDOH,lidx)=vx(gidz,gidy+FDOH,gidx);
//        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
//            lvx2(lidz,lidy-lsizey+3*FDOH,lidx)=vx(gidz,gidy-lsizey+3*FDOH,gidx);
//        if (lidx<2*FDOH)
//            lvx2(lidz,lidy,lidx-FDOH)=vx(gidz,gidy,gidx-FDOH);
//        if (lidx+lsizex-3*FDOH<FDOH)
//            lvx2(lidz,lidy,lidx+lsizex-3*FDOH)=vx(gidz,gidy,gidx+lsizex-3*FDOH);
//        if (lidx>(lsizex-2*FDOH-1))
//            lvx2(lidz,lidy,lidx+FDOH)=vx(gidz,gidy,gidx+FDOH);
//        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
//            lvx2(lidz,lidy,lidx-lsizex+3*FDOH)=vx(gidz,gidy,gidx-lsizex+3*FDOH);
//        if (lidz<FDOH)
//            lvx2(lidz-FDOH,lidy,lidx)=vx(gidz-FDOH,gidy,gidx);
//        if (lidz>(lsizez-FDOH-1))
//            lvx2(lidz+FDOH,lidy,lidx)=vx(gidz+FDOH,gidy,gidx);
//        barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//
//#if   FDOH==1
//        vxx.x = (__h2f(lvx(2*lidz,lidy,lidx))-__h2f(lvx(2*lidz,lidy,lidx-1)));
//        vxx.y = (__h2f(lvx(2*lidz+1,lidy,lidx))-__h2f(lvx(2*lidz+1,lidy,lidx-1)));
//        vxy.x = (__h2f(lvx(2*lidz,lidy+1,lidx))-__h2f(lvx(2*lidz,lidy,lidx)));
//        vxy.y = (__h2f(lvx(2*lidz+1,lidy+1,lidx))-__h2f(lvx(2*lidz+1,lidy,lidx)));
//        vxz.x = (__h2f(lvx(2*lidz+1,lidy,lidx))-__h2f(lvx(2*lidz,lidy,lidx)));
//        vxz.y = (__h2f(lvx(2*lidz+1+1,lidy,lidx))-__h2f(lvx(2*lidz+1,lidy,lidx)));
//#elif FDOH==2
//        vxx.x = (HC1*(__h2f(lvx(2*lidz,lidy,lidx))  -__h2f(lvx(2*lidz,lidy,lidx-1)))+
//               HC2*(__h2f(lvx(2*lidz,lidy,lidx+1))-__h2f(lvx(2*lidz,lidy,lidx-2))));
//        vxx.y = (HC1*(__h2f(lvx(2*lidz+1,lidy,lidx))  -__h2f(lvx(2*lidz+1,lidy,lidx-1)))+
//               HC2*(__h2f(lvx(2*lidz+1,lidy,lidx+1))-__h2f(lvx(2*lidz+1,lidy,lidx-2))));
//
//        vxy.x = (HC1*(__h2f(lvx(2*lidz,lidy+1,lidx))-__h2f(lvx(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz,lidy+2,lidx))-__h2f(lvx(2*lidz,lidy-1,lidx))));
//        vxy.y = (HC1*(__h2f(lvx(2*lidz+1,lidy+1,lidx))-__h2f(lvx(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+1,lidy+2,lidx))-__h2f(lvx(2*lidz+1,lidy-1,lidx))));
//
//        vxz.x = (HC1*(__h2f(lvx(2*lidz+1,lidy,lidx))-__h2f(lvx(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+2,lidy,lidx))-__h2f(lvx(2*lidz-1,lidy,lidx))));
//        vxz.y = (HC1*(__h2f(lvx(2*lidz+1+1,lidy,lidx))-__h2f(lvx(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+2+1,lidy,lidx))-__h2f(lvx(2*lidz-1+1,lidy,lidx))));
//#elif FDOH==3
//        vxx.x = (HC1*(__h2f(lvx(2*lidz,lidy,lidx))  -__h2f(lvx(2*lidz,lidy,lidx-1)))+
//               HC2*(__h2f(lvx(2*lidz,lidy,lidx+1))-__h2f(lvx(2*lidz,lidy,lidx-2)))+
//               HC3*(__h2f(lvx(2*lidz,lidy,lidx+2))-__h2f(lvx(2*lidz,lidy,lidx-3))));
//        vxx.y = (HC1*(__h2f(lvx(2*lidz+1,lidy,lidx))  -__h2f(lvx(2*lidz+1,lidy,lidx-1)))+
//               HC2*(__h2f(lvx(2*lidz+1,lidy,lidx+1))-__h2f(lvx(2*lidz+1,lidy,lidx-2)))+
//               HC3*(__h2f(lvx(2*lidz+1,lidy,lidx+2))-__h2f(lvx(2*lidz+1,lidy,lidx-3))));
//
//        vxy.x = (HC1*(__h2f(lvx(2*lidz,lidy+1,lidx))-__h2f(lvx(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz,lidy+2,lidx))-__h2f(lvx(2*lidz,lidy-1,lidx)))+
//               HC3*(__h2f(lvx(2*lidz,lidy+3,lidx))-__h2f(lvx(2*lidz,lidy-2,lidx))));
//        vxy.y = (HC1*(__h2f(lvx(2*lidz+1,lidy+1,lidx))-__h2f(lvx(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+1,lidy+2,lidx))-__h2f(lvx(2*lidz+1,lidy-1,lidx)))+
//               HC3*(__h2f(lvx(2*lidz+1,lidy+3,lidx))-__h2f(lvx(2*lidz+1,lidy-2,lidx))));
//
//        vxz.x = (HC1*(__h2f(lvx(2*lidz+1,lidy,lidx))-__h2f(lvx(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+2,lidy,lidx))-__h2f(lvx(2*lidz-1,lidy,lidx)))+
//               HC3*(__h2f(lvx(2*lidz+3,lidy,lidx))-__h2f(lvx(2*lidz-2,lidy,lidx))));
//        vxz.y = (HC1*(__h2f(lvx(2*lidz+1+1,lidy,lidx))-__h2f(lvx(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+2+1,lidy,lidx))-__h2f(lvx(2*lidz-1+1,lidy,lidx)))+
//               HC3*(__h2f(lvx(2*lidz+3+1,lidy,lidx))-__h2f(lvx(2*lidz-2+1,lidy,lidx))));
//#elif FDOH==4
//        vxx.x = (HC1*(__h2f(lvx(2*lidz,lidy,lidx))  -__h2f(lvx(2*lidz,lidy,lidx-1)))+
//               HC2*(__h2f(lvx(2*lidz,lidy,lidx+1))-__h2f(lvx(2*lidz,lidy,lidx-2)))+
//               HC3*(__h2f(lvx(2*lidz,lidy,lidx+2))-__h2f(lvx(2*lidz,lidy,lidx-3)))+
//               HC4*(__h2f(lvx(2*lidz,lidy,lidx+3))-__h2f(lvx(2*lidz,lidy,lidx-4))));
//        vxx.y = (HC1*(__h2f(lvx(2*lidz+1,lidy,lidx))  -__h2f(lvx(2*lidz+1,lidy,lidx-1)))+
//               HC2*(__h2f(lvx(2*lidz+1,lidy,lidx+1))-__h2f(lvx(2*lidz+1,lidy,lidx-2)))+
//               HC3*(__h2f(lvx(2*lidz+1,lidy,lidx+2))-__h2f(lvx(2*lidz+1,lidy,lidx-3)))+
//               HC4*(__h2f(lvx(2*lidz+1,lidy,lidx+3))-__h2f(lvx(2*lidz+1,lidy,lidx-4))));
//
//        vxy.x = (HC1*(__h2f(lvx(2*lidz,lidy+1,lidx))-__h2f(lvx(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz,lidy+2,lidx))-__h2f(lvx(2*lidz,lidy-1,lidx)))+
//               HC3*(__h2f(lvx(2*lidz,lidy+3,lidx))-__h2f(lvx(2*lidz,lidy-2,lidx)))+
//               HC4*(__h2f(lvx(2*lidz,lidy+4,lidx))-__h2f(lvx(2*lidz,lidy-3,lidx))));
//        vxy.y = (HC1*(__h2f(lvx(2*lidz+1,lidy+1,lidx))-__h2f(lvx(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+1,lidy+2,lidx))-__h2f(lvx(2*lidz+1,lidy-1,lidx)))+
//               HC3*(__h2f(lvx(2*lidz+1,lidy+3,lidx))-__h2f(lvx(2*lidz+1,lidy-2,lidx)))+
//               HC4*(__h2f(lvx(2*lidz+1,lidy+4,lidx))-__h2f(lvx(2*lidz+1,lidy-3,lidx))));
//
//        vxz.x = (HC1*(__h2f(lvx(2*lidz+1,lidy,lidx))-__h2f(lvx(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+2,lidy,lidx))-__h2f(lvx(2*lidz-1,lidy,lidx)))+
//               HC3*(__h2f(lvx(2*lidz+3,lidy,lidx))-__h2f(lvx(2*lidz-2,lidy,lidx)))+
//               HC4*(__h2f(lvx(2*lidz+4,lidy,lidx))-__h2f(lvx(2*lidz-3,lidy,lidx))));
//        vxz.y = (HC1*(__h2f(lvx(2*lidz+1+1,lidy,lidx))-__h2f(lvx(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+2+1,lidy,lidx))-__h2f(lvx(2*lidz-1+1,lidy,lidx)))+
//               HC3*(__h2f(lvx(2*lidz+3+1,lidy,lidx))-__h2f(lvx(2*lidz-2+1,lidy,lidx)))+
//               HC4*(__h2f(lvx(2*lidz+4+1,lidy,lidx))-__h2f(lvx(2*lidz-3+1,lidy,lidx))));
//#elif FDOH==5
//        vxx.x = (HC1*(__h2f(lvx(2*lidz,lidy,lidx))  -__h2f(lvx(2*lidz,lidy,lidx-1)))+
//               HC2*(__h2f(lvx(2*lidz,lidy,lidx+1))-__h2f(lvx(2*lidz,lidy,lidx-2)))+
//               HC3*(__h2f(lvx(2*lidz,lidy,lidx+2))-__h2f(lvx(2*lidz,lidy,lidx-3)))+
//               HC4*(__h2f(lvx(2*lidz,lidy,lidx+3))-__h2f(lvx(2*lidz,lidy,lidx-4)))+
//               HC5*(__h2f(lvx(2*lidz,lidy,lidx+4))-__h2f(lvx(2*lidz,lidy,lidx-5))));
//        vxx.y = (HC1*(__h2f(lvx(2*lidz,lidy,lidx))  -__h2f(lvx(2*lidz,lidy,lidx-1)))+
//               HC2*(__h2f(lvx(2*lidz+1,lidy,lidx+1))-__h2f(lvx(2*lidz+1,lidy,lidx-2)))+
//               HC3*(__h2f(lvx(2*lidz+1,lidy,lidx+2))-__h2f(lvx(2*lidz+1,lidy,lidx-3)))+
//               HC4*(__h2f(lvx(2*lidz+1,lidy,lidx+3))-__h2f(lvx(2*lidz+1,lidy,lidx-4)))+
//               HC5*(__h2f(lvx(2*lidz+1,lidy,lidx+4))-__h2f(lvx(2*lidz+1,lidy,lidx-5))));
//
//        vxy.x = (HC1*(__h2f(lvx(2*lidz,lidy+1,lidx))-__h2f(lvx(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz,lidy+2,lidx))-__h2f(lvx(2*lidz,lidy-1,lidx)))+
//               HC3*(__h2f(lvx(2*lidz,lidy+3,lidx))-__h2f(lvx(2*lidz,lidy-2,lidx)))+
//               HC4*(__h2f(lvx(2*lidz,lidy+4,lidx))-__h2f(lvx(2*lidz,lidy-3,lidx)))+
//               HC5*(__h2f(lvx(2*lidz,lidy+5,lidx))-__h2f(lvx(2*lidz,lidy-4,lidx))));
//        vxy.x = (HC1*(__h2f(lvx(2*lidz+1,lidy+1,lidx))-__h2f(lvx(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+1,lidy+2,lidx))-__h2f(lvx(2*lidz+1,lidy-1,lidx)))+
//               HC3*(__h2f(lvx(2*lidz+1,lidy+3,lidx))-__h2f(lvx(2*lidz+1,lidy-2,lidx)))+
//               HC4*(__h2f(lvx(2*lidz+1,lidy+4,lidx))-__h2f(lvx(2*lidz+1,lidy-3,lidx)))+
//               HC5*(__h2f(lvx(2*lidz+1,lidy+5,lidx))-__h2f(lvx(2*lidz+1,lidy-4,lidx))));
//
//        vxz.x = (HC1*(__h2f(lvx(2*lidz+1,lidy,lidx))-__h2f(lvx(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+2,lidy,lidx))-__h2f(lvx(2*lidz-1,lidy,lidx)))+
//               HC3*(__h2f(lvx(2*lidz+3,lidy,lidx))-__h2f(lvx(2*lidz-2,lidy,lidx)))+
//               HC4*(__h2f(lvx(2*lidz+4,lidy,lidx))-__h2f(lvx(2*lidz-3,lidy,lidx)))+
//               HC5*(__h2f(lvx(2*lidz+5,lidy,lidx))-__h2f(lvx(2*lidz-4,lidy,lidx))));
//        vxz.y = (HC1*(__h2f(lvx(2*lidz+1+1,lidy,lidx))-__h2f(lvx(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+2+1,lidy,lidx))-__h2f(lvx(2*lidz-1+1,lidy,lidx)))+
//               HC3*(__h2f(lvx(2*lidz+3+1,lidy,lidx))-__h2f(lvx(2*lidz-2+1,lidy,lidx)))+
//               HC4*(__h2f(lvx(2*lidz+4+1,lidy,lidx))-__h2f(lvx(2*lidz-3+1,lidy,lidx)))+
//               HC5*(__h2f(lvx(2*lidz+5+1,lidy,lidx))-__h2f(lvx(2*lidz-4+1,lidy,lidx))));
//#elif FDOH==6
//        vxx.x = (HC1*(__h2f(lvx(2*lidz,lidy,lidx))  -__h2f(lvx(2*lidz,lidy,lidx-1)))+
//               HC2*(__h2f(lvx(2*lidz,lidy,lidx+1))-__h2f(lvx(2*lidz,lidy,lidx-2)))+
//               HC3*(__h2f(lvx(2*lidz,lidy,lidx+2))-__h2f(lvx(2*lidz,lidy,lidx-3)))+
//               HC4*(__h2f(lvx(2*lidz,lidy,lidx+3))-__h2f(lvx(2*lidz,lidy,lidx-4)))+
//               HC5*(__h2f(lvx(2*lidz,lidy,lidx+4))-__h2f(lvx(2*lidz,lidy,lidx-5)))+
//               HC6*(__h2f(lvx(2*lidz,lidy,lidx+5))-__h2f(lvx(2*lidz,lidy,lidx-6))));
//        vxx.y = (HC1*(__h2f(lvx(2*lidz+1,lidy,lidx))  -__h2f(lvx(2*lidz+1,lidy,lidx-1)))+
//               HC2*(__h2f(lvx(2*lidz+1,lidy,lidx+1))-__h2f(lvx(2*lidz+1,lidy,lidx-2)))+
//               HC3*(__h2f(lvx(2*lidz+1,lidy,lidx+2))-__h2f(lvx(2*lidz+1,lidy,lidx-3)))+
//               HC4*(__h2f(lvx(2*lidz+1,lidy,lidx+3))-__h2f(lvx(2*lidz+1,lidy,lidx-4)))+
//               HC5*(__h2f(lvx(2*lidz+1,lidy,lidx+4))-__h2f(lvx(2*lidz+1,lidy,lidx-5)))+
//               HC6*(__h2f(lvx(2*lidz+1,lidy,lidx+5))-__h2f(lvx(2*lidz+1,lidy,lidx-6))));
//
//        vxy.x = (HC1*(__h2f(lvx(2*lidz,lidy+1,lidx))-__h2f(lvx(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz,lidy+2,lidx))-__h2f(lvx(2*lidz,lidy-1,lidx)))+
//               HC3*(__h2f(lvx(2*lidz,lidy+3,lidx))-__h2f(lvx(2*lidz,lidy-2,lidx)))+
//               HC4*(__h2f(lvx(2*lidz,lidy+4,lidx))-__h2f(lvx(2*lidz,lidy-3,lidx)))+
//               HC5*(__h2f(lvx(2*lidz,lidy+5,lidx))-__h2f(lvx(2*lidz,lidy-4,lidx)))+
//               HC6*(__h2f(lvx(2*lidz,lidy+6,lidx))-__h2f(lvx(2*lidz,lidy-5,lidx))));
//        vxy.y = (HC1*(__h2f(lvx(2*lidz+1,lidy+1,lidx))-__h2f(lvx(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+1,lidy+2,lidx))-__h2f(lvx(2*lidz+1,lidy-1,lidx)))+
//               HC3*(__h2f(lvx(2*lidz+1,lidy+3,lidx))-__h2f(lvx(2*lidz+1,lidy-2,lidx)))+
//               HC4*(__h2f(lvx(2*lidz+1,lidy+4,lidx))-__h2f(lvx(2*lidz+1,lidy-3,lidx)))+
//               HC5*(__h2f(lvx(2*lidz+1,lidy+5,lidx))-__h2f(lvx(2*lidz+1,lidy-4,lidx)))+
//               HC6*(__h2f(lvx(2*lidz+1,lidy+6,lidx))-__h2f(lvx(2*lidz+1,lidy-5,lidx))));
//
//        vxz.x = (HC1*(__h2f(lvx(2*lidz+1,lidy,lidx))-__h2f(lvx(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+2,lidy,lidx))-__h2f(lvx(2*lidz-1,lidy,lidx)))+
//               HC3*(__h2f(lvx(2*lidz+3,lidy,lidx))-__h2f(lvx(2*lidz-2,lidy,lidx)))+
//               HC4*(__h2f(lvx(2*lidz+4,lidy,lidx))-__h2f(lvx(2*lidz-3,lidy,lidx)))+
//               HC5*(__h2f(lvx(2*lidz+5,lidy,lidx))-__h2f(lvx(2*lidz-4,lidy,lidx)))+
//               HC6*(__h2f(lvx(2*lidz+6,lidy,lidx))-__h2f(lvx(2*lidz-5,lidy,lidx))));
//        vxz.y = (HC1*(__h2f(lvx(2*lidz+1+1,lidy,lidx))-__h2f(lvx(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvx(2*lidz+2+1,lidy,lidx))-__h2f(lvx(2*lidz-1+1,lidy,lidx)))+
//               HC3*(__h2f(lvx(2*lidz+3+1,lidy,lidx))-__h2f(lvx(2*lidz-2+1,lidy,lidx)))+
//               HC4*(__h2f(lvx(2*lidz+4+1,lidy,lidx))-__h2f(lvx(2*lidz-3+1,lidy,lidx)))+
//               HC5*(__h2f(lvx(2*lidz+5+1,lidy,lidx))-__h2f(lvx(2*lidz-4+1,lidy,lidx)))+
//               HC6*(__h2f(lvx(2*lidz+6+1,lidy,lidx))-__h2f(lvx(2*lidz-5+1,lidy,lidx))));
//#endif
//
//
//#if LOCAL_OFF==0
//        barrier(CLK_LOCAL_MEM_FENCE);
//        lvy2(lidz,lidy,lidx)=vy(gidz, gidy, gidx);
//        if (lidy<2*FDOH)
//            lvy2(lidz,lidy-FDOH,lidx)=vy(gidz,gidy-FDOH,gidx);
//        if (lidy+lsizey-3*FDOH<FDOH)
//            lvy2(lidz,lidy+lsizey-3*FDOH,lidx)=vy(gidz,gidy+lsizey-3*FDOH,gidx);
//        if (lidy>(lsizey-2*FDOH-1))
//            lvy2(lidz,lidy+FDOH,lidx)=vy(gidz,gidy+FDOH,gidx);
//        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
//            lvy2(lidz,lidy-lsizey+3*FDOH,lidx)=vy(gidz,gidy-lsizey+3*FDOH,gidx);
//        if (lidx<2*FDOH)
//            lvy2(lidz,lidy,lidx-FDOH)=vy(gidz,gidy,gidx-FDOH);
//        if (lidx+lsizex-3*FDOH<FDOH)
//            lvy2(lidz,lidy,lidx+lsizex-3*FDOH)=vy(gidz,gidy,gidx+lsizex-3*FDOH);
//        if (lidx>(lsizex-2*FDOH-1))
//            lvy2(lidz,lidy,lidx+FDOH)=vy(gidz,gidy,gidx+FDOH);
//        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
//            lvy2(lidz,lidy,lidx-lsizex+3*FDOH)=vy(gidz,gidy,gidx-lsizex+3*FDOH);
//        if (lidz<FDOH)
//            lvy2(lidz-FDOH,lidy,lidx)=vy(gidz-FDOH,gidy,gidx);
//        if (lidz>(lsizez-FDOH-1))
//            lvy2(lidz+FDOH,lidy,lidx)=vy(gidz+FDOH,gidy,gidx);
//        barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//
//#if   FDOH==1
//        vyx.x = (__h2f(lvy(2*lidz,lidy,lidx+1))-__h2f(lvy(2*lidz,lidy,lidx)));
//        vyx.y = (__h2f(lvy(2*lidz+1,lidy,lidx+1))-__h2f(lvy(2*lidz+1,lidy,lidx)));
//        vyy.x = (__h2f(lvy(2*lidz,lidy,lidx))-__h2f(lvy(2*lidz,lidy-1,lidx)));
//        vyy.y = (__h2f(lvy(2*lidz+1,lidy,lidx))-__h2f(lvy(2*lidz+1,lidy-1,lidx)));
//        vyz.x = (__h2f(lvy(2*lidz+1,lidy,lidx))-__h2f(lvy(2*lidz,lidy,lidx)));
//        vyz.y = (__h2f(lvy(2*lidz+1+1,lidy,lidx))-__h2f(lvy(2*lidz+1,lidy,lidx)));
//#elif FDOH==2
//        vyx.x = (HC1*(__h2f(lvy(2*lidz,lidy,lidx+1))-__h2f(lvy(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz,lidy,lidx+2))-__h2f(lvy(2*lidz,lidy,lidx-1))));
//        vyx.y = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx+1))-__h2f(lvy(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+1,lidy,lidx+2))-__h2f(lvy(2*lidz+1,lidy,lidx-1))));
//
//        vyy.x = (HC1*(__h2f(lvy(2*lidz,lidy,lidx))  -__h2f(lvy(2*lidz,lidy-1,lidx)))+
//               HC2*(__h2f(lvy(2*lidz,lidy+1,lidx))-__h2f(lvy(2*lidz,lidy-2,lidx))));
//        vyy.y = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx))  -__h2f(lvy(2*lidz+1,lidy-1,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+1,lidy+1,lidx))-__h2f(lvy(2*lidz+1,lidy-2,lidx))));
//
//        vyz.x = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx))-__h2f(lvy(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+2,lidy,lidx))-__h2f(lvy(2*lidz-1,lidy,lidx))));
//        vyz.y = (HC1*(__h2f(lvy(2*lidz+1+1,lidy,lidx))-__h2f(lvy(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+2+1,lidy,lidx))-__h2f(lvy(2*lidz-1+1,lidy,lidx))));
//#elif FDOH==3
//        vyx.x = (HC1*(__h2f(lvy(2*lidz,lidy,lidx+1))-__h2f(lvy(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz,lidy,lidx+2))-__h2f(lvy(2*lidz,lidy,lidx-1)))+
//               HC3*(__h2f(lvy(2*lidz,lidy,lidx+3))-__h2f(lvy(2*lidz,lidy,lidx-2))));
//        vyx.y = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx+1))-__h2f(lvy(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+1,lidy,lidx+2))-__h2f(lvy(2*lidz+1,lidy,lidx-1)))+
//               HC3*(__h2f(lvy(2*lidz+1,lidy,lidx+3))-__h2f(lvy(2*lidz+1,lidy,lidx-2))));
//
//        vyy.x = (HC1*(__h2f(lvy(2*lidz,lidy,lidx))-__h2f(lvy(2*lidz,lidy-1,lidx)))+
//               HC2*(__h2f(lvy(2*lidz,lidy+1,lidx))-__h2f(lvy(2*lidz,lidy-2,lidx)))+
//               HC3*(__h2f(lvy(2*lidz,lidy+2,lidx))-__h2f(lvy(2*lidz,lidy-3,lidx))));
//        vyy.y = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx))-__h2f(lvy(2*lidz+1,lidy-1,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+1,lidy+1,lidx))-__h2f(lvy(2*lidz+1,lidy-2,lidx)))+
//               HC3*(__h2f(lvy(2*lidz+1,lidy+2,lidx))-__h2f(lvy(2*lidz+1,lidy-3,lidx))));
//
//        vyz.x = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx))-__h2f(lvy(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+2,lidy,lidx))-__h2f(lvy(2*lidz-1,lidy,lidx)))+
//               HC3*(__h2f(lvy(2*lidz+3,lidy,lidx))-__h2f(lvy(2*lidz-2,lidy,lidx))));
//        vyz.y = (HC1*(__h2f(lvy(2*lidz+1+1,lidy,lidx))-__h2f(lvy(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+2+1,lidy,lidx))-__h2f(lvy(2*lidz-1+1,lidy,lidx)))+
//               HC3*(__h2f(lvy(2*lidz+3+1,lidy,lidx))-__h2f(lvy(2*lidz-2+1,lidy,lidx))));
//#elif FDOH==4
//        vyx.x = (HC1*(__h2f(lvy(2*lidz,lidy,lidx+1))-__h2f(lvy(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz,lidy,lidx+2))-__h2f(lvy(2*lidz,lidy,lidx-1)))+
//               HC3*(__h2f(lvy(2*lidz,lidy,lidx+3))-__h2f(lvy(2*lidz,lidy,lidx-2)))+
//               HC4*(__h2f(lvy(2*lidz,lidy,lidx+4))-__h2f(lvy(2*lidz,lidy,lidx-3))));
//        vyx.y = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx+1))-__h2f(lvy(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+1,lidy,lidx+2))-__h2f(lvy(2*lidz+1,lidy,lidx-1)))+
//               HC3*(__h2f(lvy(2*lidz+1,lidy,lidx+3))-__h2f(lvy(2*lidz+1,lidy,lidx-2)))+
//               HC4*(__h2f(lvy(2*lidz+1,lidy,lidx+4))-__h2f(lvy(2*lidz+1,lidy,lidx-3))));
//
//        vyy.x = (HC1*(__h2f(lvy(2*lidz,lidy,lidx))  -__h2f(lvy(2*lidz,lidy-1,lidx)))+
//               HC2*(__h2f(lvy(2*lidz,lidy+1,lidx))-__h2f(lvy(2*lidz,lidy-2,lidx)))+
//               HC3*(__h2f(lvy(2*lidz,lidy+2,lidx))-__h2f(lvy(2*lidz,lidy-3,lidx)))+
//               HC4*(__h2f(lvy(2*lidz,lidy+3,lidx))-__h2f(lvy(2*lidz,lidy-4,lidx))));
//        vyy.y = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx))  -__h2f(lvy(2*lidz+1,lidy-1,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+1,lidy+1,lidx))-__h2f(lvy(2*lidz+1,lidy-2,lidx)))+
//               HC3*(__h2f(lvy(2*lidz+1,lidy+2,lidx))-__h2f(lvy(2*lidz+1,lidy-3,lidx)))+
//               HC4*(__h2f(lvy(2*lidz+1,lidy+3,lidx))-__h2f(lvy(2*lidz+1,lidy-4,lidx))));
//
//        vyz.x = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx))-__h2f(lvy(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+2,lidy,lidx))-__h2f(lvy(2*lidz-1,lidy,lidx)))+
//               HC3*(__h2f(lvy(2*lidz+3,lidy,lidx))-__h2f(lvy(2*lidz-2,lidy,lidx)))+
//               HC4*(__h2f(lvy(2*lidz+4,lidy,lidx))-__h2f(lvy(2*lidz-3,lidy,lidx))));
//        vyz.y = (HC1*(__h2f(lvy(2*lidz+1+1,lidy,lidx))-__h2f(lvy(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+2+1,lidy,lidx))-__h2f(lvy(2*lidz-1+1,lidy,lidx)))+
//               HC3*(__h2f(lvy(2*lidz+3+1,lidy,lidx))-__h2f(lvy(2*lidz-2+1,lidy,lidx)))+
//               HC4*(__h2f(lvy(2*lidz+4+1,lidy,lidx))-__h2f(lvy(2*lidz-3+1,lidy,lidx))));
//#elif FDOH==5
//        vyx.x = (HC1*(__h2f(lvy(2*lidz,lidy,lidx+1))-__h2f(lvy(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz,lidy,lidx+2))-__h2f(lvy(2*lidz,lidy,lidx-1)))+
//               HC3*(__h2f(lvy(2*lidz,lidy,lidx+3))-__h2f(lvy(2*lidz,lidy,lidx-2)))+
//               HC4*(__h2f(lvy(2*lidz,lidy,lidx+4))-__h2f(lvy(2*lidz,lidy,lidx-3)))+
//               HC5*(__h2f(lvy(2*lidz,lidy,lidx+5))-__h2f(lvy(2*lidz,lidy,lidx-4))));
//        vyx.y = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx+1))-__h2f(lvy(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+1,lidy,lidx+2))-__h2f(lvy(2*lidz+1,lidy,lidx-1)))+
//               HC3*(__h2f(lvy(2*lidz+1,lidy,lidx+3))-__h2f(lvy(2*lidz+1,lidy,lidx-2)))+
//               HC4*(__h2f(lvy(2*lidz+1,lidy,lidx+4))-__h2f(lvy(2*lidz+1,lidy,lidx-3)))+
//               HC5*(__h2f(lvy(2*lidz+1,lidy,lidx+5))-__h2f(lvy(2*lidz+1,lidy,lidx-4))));
//
//        vyy.x = (HC1*(__h2f(lvy(2*lidz,lidy,lidx))  -__h2f(lvy(2*lidz,lidy-1,lidx)))+
//               HC2*(__h2f(lvy(2*lidz,lidy+1,lidx))-__h2f(lvy(2*lidz,lidy-2,lidx)))+
//               HC3*(__h2f(lvy(2*lidz,lidy+2,lidx))-__h2f(lvy(2*lidz,lidy-3,lidx)))+
//               HC4*(__h2f(lvy(2*lidz,lidy+3,lidx))-__h2f(lvy(2*lidz,lidy-4,lidx)))+
//               HC5*(__h2f(lvy(2*lidz,lidy+4,lidx))-__h2f(lvy(2*lidz,lidy-5,lidx))));
//        vyy.y = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx))  -__h2f(lvy(2*lidz+1,lidy-1,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+1,lidy+1,lidx))-__h2f(lvy(2*lidz+1,lidy-2,lidx)))+
//               HC3*(__h2f(lvy(2*lidz+1,lidy+2,lidx))-__h2f(lvy(2*lidz+1,lidy-3,lidx)))+
//               HC4*(__h2f(lvy(2*lidz+1,lidy+3,lidx))-__h2f(lvy(2*lidz+1,lidy-4,lidx)))+
//               HC5*(__h2f(lvy(2*lidz+1,lidy+4,lidx))-__h2f(lvy(2*lidz+1,lidy-5,lidx))));
//
//        vyz.x = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx))-__h2f(lvy(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+2,lidy,lidx))-__h2f(lvy(2*lidz-1,lidy,lidx)))+
//               HC3*(__h2f(lvy(2*lidz+3,lidy,lidx))-__h2f(lvy(2*lidz-2,lidy,lidx)))+
//               HC4*(__h2f(lvy(2*lidz+4,lidy,lidx))-__h2f(lvy(2*lidz-3,lidy,lidx)))+
//               HC5*(__h2f(lvy(2*lidz+5,lidy,lidx))-__h2f(lvy(2*lidz-4,lidy,lidx))));
//        vyz.y = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx))-__h2f(lvy(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+2+1,lidy,lidx))-__h2f(lvy(2*lidz-1+1,lidy,lidx)))+
//               HC3*(__h2f(lvy(2*lidz+3+1,lidy,lidx))-__h2f(lvy(2*lidz-2+1,lidy,lidx)))+
//               HC4*(__h2f(lvy(2*lidz+4+1,lidy,lidx))-__h2f(lvy(2*lidz-3+1,lidy,lidx)))+
//               HC5*(__h2f(lvy(2*lidz+5+1,lidy,lidx))-__h2f(lvy(2*lidz-4+1,lidy,lidx))));
//#elif FDOH==6
//        vyx.x = (HC1*(__h2f(lvy(2*lidz,lidy,lidx+1))-__h2f(lvy(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz,lidy,lidx+2))-__h2f(lvy(2*lidz,lidy,lidx-1)))+
//               HC3*(__h2f(lvy(2*lidz,lidy,lidx+3))-__h2f(lvy(2*lidz,lidy,lidx-2)))+
//               HC4*(__h2f(lvy(2*lidz,lidy,lidx+4))-__h2f(lvy(2*lidz,lidy,lidx-3)))+
//               HC5*(__h2f(lvy(2*lidz,lidy,lidx+5))-__h2f(lvy(2*lidz,lidy,lidx-4)))+
//               HC6*(__h2f(lvy(2*lidz,lidy,lidx+6))-__h2f(lvy(2*lidz,lidy,lidx-5))));
//        vyx.y = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx+1))-__h2f(lvy(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+1,lidy,lidx+2))-__h2f(lvy(2*lidz+1,lidy,lidx-1)))+
//               HC3*(__h2f(lvy(2*lidz+1,lidy,lidx+3))-__h2f(lvy(2*lidz+1,lidy,lidx-2)))+
//               HC4*(__h2f(lvy(2*lidz+1,lidy,lidx+4))-__h2f(lvy(2*lidz+1,lidy,lidx-3)))+
//               HC5*(__h2f(lvy(2*lidz+1,lidy,lidx+5))-__h2f(lvy(2*lidz+1,lidy,lidx-4)))+
//               HC6*(__h2f(lvy(2*lidz+1,lidy,lidx+6))-__h2f(lvy(2*lidz+1,lidy,lidx-5))));
//
//        vyy.x = (HC1*(__h2f(lvy(2*lidz,lidy,lidx))  -__h2f(lvy(2*lidz,lidy-1,lidx)))+
//               HC2*(__h2f(lvy(2*lidz,lidy+1,lidx))-__h2f(lvy(2*lidz,lidy-2,lidx)))+
//               HC3*(__h2f(lvy(2*lidz,lidy+2,lidx))-__h2f(lvy(2*lidz,lidy-3,lidx)))+
//               HC4*(__h2f(lvy(2*lidz,lidy+3,lidx))-__h2f(lvy(2*lidz,lidy-4,lidx)))+
//               HC5*(__h2f(lvy(2*lidz,lidy+4,lidx))-__h2f(lvy(2*lidz,lidy-5,lidx)))+
//               HC6*(__h2f(lvy(2*lidz,lidy+5,lidx))-__h2f(lvy(2*lidz,lidy-6,lidx))));
//        vyy.y = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx))  -__h2f(lvy(2*lidz+1,lidy-1,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+1,lidy+1,lidx))-__h2f(lvy(2*lidz+1,lidy-2,lidx)))+
//               HC3*(__h2f(lvy(2*lidz+1,lidy+2,lidx))-__h2f(lvy(2*lidz+1,lidy-3,lidx)))+
//               HC4*(__h2f(lvy(2*lidz+1,lidy+3,lidx))-__h2f(lvy(2*lidz+1,lidy-4,lidx)))+
//               HC5*(__h2f(lvy(2*lidz+1,lidy+4,lidx))-__h2f(lvy(2*lidz+1,lidy-5,lidx)))+
//               HC6*(__h2f(lvy(2*lidz+1,lidy+5,lidx))-__h2f(lvy(2*lidz+1,lidy-6,lidx))));
//
//        vyz.x = (HC1*(__h2f(lvy(2*lidz+1,lidy,lidx))-__h2f(lvy(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+2,lidy,lidx))-__h2f(lvy(2*lidz-1,lidy,lidx)))+
//               HC3*(__h2f(lvy(2*lidz+3,lidy,lidx))-__h2f(lvy(2*lidz-2,lidy,lidx)))+
//               HC4*(__h2f(lvy(2*lidz+4,lidy,lidx))-__h2f(lvy(2*lidz-3,lidy,lidx)))+
//               HC5*(__h2f(lvy(2*lidz+5,lidy,lidx))-__h2f(lvy(2*lidz-4,lidy,lidx)))+
//               HC6*(__h2f(lvy(2*lidz+6,lidy,lidx))-__h2f(lvy(2*lidz-5,lidy,lidx))));
//        vyz.y = (HC1*(__h2f(lvy(2*lidz+1+1,lidy,lidx))-__h2f(lvy(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvy(2*lidz+2+1,lidy,lidx))-__h2f(lvy(2*lidz-1+1,lidy,lidx)))+
//               HC3*(__h2f(lvy(2*lidz+3+1,lidy,lidx))-__h2f(lvy(2*lidz-2+1,lidy,lidx)))+
//               HC4*(__h2f(lvy(2*lidz+4+1,lidy,lidx))-__h2f(lvy(2*lidz-3+1,lidy,lidx)))+
//               HC5*(__h2f(lvy(2*lidz+5+1,lidy,lidx))-__h2f(lvy(2*lidz-4+1,lidy,lidx)))+
//               HC6*(__h2f(lvy(2*lidz+6+1,lidy,lidx))-__h2f(lvy(2*lidz-5+1,lidy,lidx))));
//#endif
//
//
//
//
//#if LOCAL_OFF==0
//        barrier(CLK_LOCAL_MEM_FENCE);
//        lvz2(lidz,lidy,lidx)=vz(gidz, gidy, gidx);
//        if (lidy<2*FDOH)
//            lvz2(lidz,lidy-FDOH,lidx)=vz(gidz,gidy-FDOH,gidx);
//        if (lidy+lsizey-3*FDOH<FDOH)
//            lvz2(lidz,lidy+lsizey-3*FDOH,lidx)=vz(gidz,gidy+lsizey-3*FDOH,gidx);
//        if (lidy>(lsizey-2*FDOH-1))
//            lvz2(lidz,lidy+FDOH,lidx)=vz(gidz,gidy+FDOH,gidx);
//        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
//            lvz2(lidz,lidy-lsizey+3*FDOH,lidx)=vz(gidz,gidy-lsizey+3*FDOH,gidx);
//        if (lidx<2*FDOH)
//            lvz2(lidz,lidy,lidx-FDOH)=vz(gidz,gidy,gidx-FDOH);
//        if (lidx+lsizex-3*FDOH<FDOH)
//            lvz2(lidz,lidy,lidx+lsizex-3*FDOH)=vz(gidz,gidy,gidx+lsizex-3*FDOH);
//        if (lidx>(lsizex-2*FDOH-1))
//            lvz2(lidz,lidy,lidx+FDOH)=vz(gidz,gidy,gidx+FDOH);
//        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
//            lvz2(lidz,lidy,lidx-lsizex+3*FDOH)=vz(gidz,gidy,gidx-lsizex+3*FDOH);
//        if (lidz<FDOH)
//            lvz2(lidz-FDOH,lidy,lidx)=vz(gidz-FDOH,gidy,gidx);
//        if (lidz>(lsizez-FDOH-1))
//            lvz2(lidz+FDOH,lidy,lidx)=vz(gidz+FDOH,gidy,gidx);
//        barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//
//
//#if   FDOH==1
//        vzx.x = (__h2f(lvz(2*lidz,lidy,lidx+1))-__h2f(lvz(2*lidz,lidy,lidx)));
//        vzx.y = (__h2f(lvz(2*lidz+1,lidy,lidx+1))-__h2f(lvz(2*lidz+1,lidy,lidx)));
//        vzy.x = (__h2f(lvz(2*lidz,lidy+1,lidx))-__h2f(lvz(2*lidz,lidy,lidx)));
//        vzy.y = (__h2f(lvz(2*lidz+1,lidy+1,lidx))-__h2f(lvz(2*lidz+1,lidy,lidx)));
//        vzz.x = (__h2f(lvz(2*lidz,lidy,lidx))-__h2f(lvz(2*lidz-1,lidy,lidx)));
//        vzz.y = (__h2f(lvz(2*lidz+1,lidy,lidx))-__h2f(lvz(2*lidz-1+1,lidy,lidx)));
//#elif FDOH==2
//
//        vzx.x = (HC1*(__h2f(lvz(2*lidz,lidy,lidx+1))-__h2f(lvz(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz,lidy,lidx+2))-__h2f(lvz(2*lidz,lidy,lidx-1))));
//        vzx.y = (HC1*(__h2f(lvz(2*lidz+1,lidy,lidx+1))-__h2f(lvz(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy,lidx+2))-__h2f(lvz(2*lidz+1,lidy,lidx-1))));
//
//        vzy.x = (HC1*(__h2f(lvz(2*lidz,lidy+1,lidx))-__h2f(lvz(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz,lidy+2,lidx))-__h2f(lvz(2*lidz,lidy-1,lidx))));
//        vzy.y = (HC1*(__h2f(lvz(2*lidz+1,lidy+1,lidx))-__h2f(lvz(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy+2,lidx))-__h2f(lvz(2*lidz+1,lidy-1,lidx))));
//
//        vzz.x = (HC1*(__h2f(lvz(2*lidz,lidy,lidx))  -__h2f(lvz(2*lidz-1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy,lidx))-__h2f(lvz(2*lidz-2,lidy,lidx))));
//        vzz.y = (HC1*(__h2f(lvz(2*lidz+1,lidy,lidx))  -__h2f(lvz(2*lidz-1+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1+1,lidy,lidx))-__h2f(lvz(2*lidz-2+1,lidy,lidx))));
//#elif FDOH==3
//        vzx.x = (HC1*(__h2f(lvz(2*lidz,lidy,lidx+1))-__h2f(lvz(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz,lidy,lidx+2))-__h2f(lvz(2*lidz,lidy,lidx-1)))+
//               HC3*(__h2f(lvz(2*lidz,lidy,lidx+3))-__h2f(lvz(2*lidz,lidy,lidx-2))));
//        vzx.y = (HC1*(__h2f(lvz(2*lidz+1,lidy,lidx+1))-__h2f(lvz(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy,lidx+2))-__h2f(lvz(2*lidz+1,lidy,lidx-1)))+
//               HC3*(__h2f(lvz(2*lidz+1,lidy,lidx+3))-__h2f(lvz(2*lidz+1,lidy,lidx-2))));
//
//        vzy.x = (HC1*(__h2f(lvz(2*lidz,lidy+1,lidx))-__h2f(lvz(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz,lidy+2,lidx))-__h2f(lvz(2*lidz,lidy-1,lidx)))+
//               HC3*(__h2f(lvz(2*lidz,lidy+3,lidx))-__h2f(lvz(2*lidz,lidy-2,lidx))));
//        vzy.y = (HC1*(__h2f(lvz(2*lidz+1,lidy+1,lidx))-__h2f(lvz(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy+2,lidx))-__h2f(lvz(2*lidz+1,lidy-1,lidx)))+
//               HC3*(__h2f(lvz(2*lidz+1,lidy+3,lidx))-__h2f(lvz(2*lidz+1,lidy-2,lidx))));
//
//        vzz.x = (HC1*(__h2f(lvz(2*lidz,lidy,lidx))-__h2f(lvz(2*lidz-1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy,lidx))-__h2f(lvz(2*lidz-2,lidy,lidx)))+
//               HC3*(__h2f(lvz(2*lidz+2,lidy,lidx))-__h2f(lvz(2*lidz-3,lidy,lidx))));
//        vzz.y = (HC1*(__h2f(lvz(2*lidz+1,lidy,lidx))-__h2f(lvz(2*lidz-1+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1+1,lidy,lidx))-__h2f(lvz(2*lidz-2+1,lidy,lidx)))+
//               HC3*(__h2f(lvz(2*lidz+2+1,lidy,lidx))-__h2f(lvz(2*lidz-3+1,lidy,lidx))));
//#elif FDOH==4
//        vzx.x = (HC1*(__h2f(lvz(2*lidz,lidy,lidx+1))-__h2f(lvz(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz,lidy,lidx+2))-__h2f(lvz(2*lidz,lidy,lidx-1)))+
//               HC3*(__h2f(lvz(2*lidz,lidy,lidx+3))-__h2f(lvz(2*lidz,lidy,lidx-2)))+
//               HC4*(__h2f(lvz(2*lidz,lidy,lidx+4))-__h2f(lvz(2*lidz,lidy,lidx-3))));
//        vzx.y = (HC1*(__h2f(lvz(2*lidz+1,lidy,lidx+1))-__h2f(lvz(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy,lidx+2))-__h2f(lvz(2*lidz+1,lidy,lidx-1)))+
//               HC3*(__h2f(lvz(2*lidz+1,lidy,lidx+3))-__h2f(lvz(2*lidz+1,lidy,lidx-2)))+
//               HC4*(__h2f(lvz(2*lidz+1,lidy,lidx+4))-__h2f(lvz(2*lidz+1,lidy,lidx-3))));
//
//        vzy.x = (HC1*(__h2f(lvz(2*lidz,lidy+1,lidx))-__h2f(lvz(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz,lidy+2,lidx))-__h2f(lvz(2*lidz,lidy-1,lidx)))+
//               HC3*(__h2f(lvz(2*lidz,lidy+3,lidx))-__h2f(lvz(2*lidz,lidy-2,lidx)))+
//               HC4*(__h2f(lvz(2*lidz,lidy+4,lidx))-__h2f(lvz(2*lidz,lidy-3,lidx))));
//        vzy.y = (HC1*(__h2f(lvz(2*lidz+1,lidy+1,lidx))-__h2f(lvz(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy+2,lidx))-__h2f(lvz(2*lidz+1,lidy-1,lidx)))+
//               HC3*(__h2f(lvz(2*lidz+1,lidy+3,lidx))-__h2f(lvz(2*lidz+1,lidy-2,lidx)))+
//               HC4*(__h2f(lvz(2*lidz+1,lidy+4,lidx))-__h2f(lvz(2*lidz+1,lidy-3,lidx))));
//
//        vzz.x = (HC1*(__h2f(lvz(2*lidz,lidy,lidx))  -__h2f(lvz(2*lidz-1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy,lidx))-__h2f(lvz(2*lidz-2,lidy,lidx)))+
//               HC3*(__h2f(lvz(2*lidz+2,lidy,lidx))-__h2f(lvz(2*lidz-3,lidy,lidx)))+
//               HC4*(__h2f(lvz(2*lidz+3,lidy,lidx))-__h2f(lvz(2*lidz-4,lidy,lidx))));
//        vzz.y = (HC1*(__h2f(lvz(2*lidz+1,lidy,lidx))  -__h2f(lvz(2*lidz-1+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1+1,lidy,lidx))-__h2f(lvz(2*lidz-2+1,lidy,lidx)))+
//               HC3*(__h2f(lvz(2*lidz+2+1,lidy,lidx))-__h2f(lvz(2*lidz-3+1,lidy,lidx)))+
//               HC4*(__h2f(lvz(2*lidz+3+1,lidy,lidx))-__h2f(lvz(2*lidz-4+1,lidy,lidx))));
//#elif FDOH==5
//        vzx.x = (HC1*(__h2f(lvz(2*lidz,lidy,lidx+1))-__h2f(lvz(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz,lidy,lidx+2))-__h2f(lvz(2*lidz,lidy,lidx-1)))+
//               HC3*(__h2f(lvz(2*lidz,lidy,lidx+3))-__h2f(lvz(2*lidz,lidy,lidx-2)))+
//               HC4*(__h2f(lvz(2*lidz,lidy,lidx+4))-__h2f(lvz(2*lidz,lidy,lidx-3)))+
//               HC5*(__h2f(lvz(2*lidz,lidy,lidx+5))-__h2f(lvz(2*lidz,lidy,lidx-4))));
//        vzx.y = (HC1*(__h2f(lvz(2*lidz+1,lidy,lidx+1))-__h2f(lvz(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy,lidx+2))-__h2f(lvz(2*lidz+1,lidy,lidx-1)))+
//               HC3*(__h2f(lvz(2*lidz+1,lidy,lidx+3))-__h2f(lvz(2*lidz+1,lidy,lidx-2)))+
//               HC4*(__h2f(lvz(2*lidz+1,lidy,lidx+4))-__h2f(lvz(2*lidz+1,lidy,lidx-3)))+
//               HC5*(__h2f(lvz(2*lidz+1,lidy,lidx+5))-__h2f(lvz(2*lidz+1,lidy,lidx-4))));
//
//        vzy.x = (HC1*(__h2f(lvz(2*lidz,lidy+1,lidx))-__h2f(lvz(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz,lidy+2,lidx))-__h2f(lvz(2*lidz,lidy-1,lidx)))+
//               HC3*(__h2f(lvz(2*lidz,lidy+3,lidx))-__h2f(lvz(2*lidz,lidy-2,lidx)))+
//               HC4*(__h2f(lvz(2*lidz,lidy+4,lidx))-__h2f(lvz(2*lidz,lidy-3,lidx)))+
//               HC5*(__h2f(lvz(2*lidz,lidy+5,lidx))-__h2f(lvz(2*lidz,lidy-4,lidx))));
//        vzy.y = (HC1*(__h2f(lvz(2*lidz+1,lidy+1,lidx))-__h2f(lvz(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy+2,lidx))-__h2f(lvz(2*lidz+1,lidy-1,lidx)))+
//               HC3*(__h2f(lvz(2*lidz+1,lidy+3,lidx))-__h2f(lvz(2*lidz+1,lidy-2,lidx)))+
//               HC4*(__h2f(lvz(2*lidz+1,lidy+4,lidx))-__h2f(lvz(2*lidz+1,lidy-3,lidx)))+
//               HC5*(__h2f(lvz(2*lidz+1,lidy+5,lidx))-__h2f(lvz(2*lidz+1,lidy-4,lidx))));
//
//        vzz.x = (HC1*(__h2f(lvz(2*lidz,lidy,lidx))  -__h2f(lvz(2*lidz-1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy,lidx))-__h2f(lvz(2*lidz-2,lidy,lidx)))+
//               HC3*(__h2f(lvz(2*lidz+2,lidy,lidx))-__h2f(lvz(2*lidz-3,lidy,lidx)))+
//               HC4*(__h2f(lvz(2*lidz+3,lidy,lidx))-__h2f(lvz(2*lidz-4,lidy,lidx)))+
//               HC5*(__h2f(lvz(2*lidz+4,lidy,lidx))-__h2f(lvz(2*lidz-5,lidy,lidx))));
//        vzz.y = (HC1*(__h2f(lvz(2*lidz+1,lidy,lidx))  -__h2f(lvz(2*lidz-1+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1+1,lidy,lidx))-__h2f(lvz(2*lidz-2+1,lidy,lidx)))+
//               HC3*(__h2f(lvz(2*lidz+2+1,lidy,lidx))-__h2f(lvz(2*lidz-3+1,lidy,lidx)))+
//               HC4*(__h2f(lvz(2*lidz+3+1,lidy,lidx))-__h2f(lvz(2*lidz-4+1,lidy,lidx)))+
//               HC5*(__h2f(lvz(2*lidz+4+1,lidy,lidx))-__h2f(lvz(2*lidz-5+1,lidy,lidx))));
//#elif FDOH==6
//        vzx.x = (HC1*(__h2f(lvz(2*lidz,lidy,lidx+1))-__h2f(lvz(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz,lidy,lidx+2))-__h2f(lvz(2*lidz,lidy,lidx-1)))+
//               HC3*(__h2f(lvz(2*lidz,lidy,lidx+3))-__h2f(lvz(2*lidz,lidy,lidx-2)))+
//               HC4*(__h2f(lvz(2*lidz,lidy,lidx+4))-__h2f(lvz(2*lidz,lidy,lidx-3)))+
//               HC5*(__h2f(lvz(2*lidz,lidy,lidx+5))-__h2f(lvz(2*lidz,lidy,lidx-4)))+
//               HC6*(__h2f(lvz(2*lidz,lidy,lidx+6))-__h2f(lvz(2*lidz,lidy,lidx-5))));
//        vzx.y = (HC1*(__h2f(lvz(2*lidz+1,lidy,lidx+1))-__h2f(lvz(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy,lidx+2))-__h2f(lvz(2*lidz+1,lidy,lidx-1)))+
//               HC3*(__h2f(lvz(2*lidz+1,lidy,lidx+3))-__h2f(lvz(2*lidz+1,lidy,lidx-2)))+
//               HC4*(__h2f(lvz(2*lidz+1,lidy,lidx+4))-__h2f(lvz(2*lidz+1,lidy,lidx-3)))+
//               HC5*(__h2f(lvz(2*lidz+1,lidy,lidx+5))-__h2f(lvz(2*lidz+1,lidy,lidx-4)))+
//               HC6*(__h2f(lvz(2*lidz+1,lidy,lidx+6))-__h2f(lvz(2*lidz+1,lidy,lidx-5))));
//
//        vzy.x = (HC1*(__h2f(lvz(2*lidz,lidy+1,lidx))-__h2f(lvz(2*lidz,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz,lidy+2,lidx))-__h2f(lvz(2*lidz,lidy-1,lidx)))+
//               HC3*(__h2f(lvz(2*lidz,lidy+3,lidx))-__h2f(lvz(2*lidz,lidy-2,lidx)))+
//               HC4*(__h2f(lvz(2*lidz,lidy+4,lidx))-__h2f(lvz(2*lidz,lidy-3,lidx)))+
//               HC5*(__h2f(lvz(2*lidz,lidy+5,lidx))-__h2f(lvz(2*lidz,lidy-4,lidx)))+
//               HC6*(__h2f(lvz(2*lidz,lidy+6,lidx))-__h2f(lvz(2*lidz,lidy-5,lidx))));
//        vzy.y = (HC1*(__h2f(lvz(2*lidz+1,lidy+1,lidx))-__h2f(lvz(2*lidz+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy+2,lidx))-__h2f(lvz(2*lidz+1,lidy-1,lidx)))+
//               HC3*(__h2f(lvz(2*lidz+1,lidy+3,lidx))-__h2f(lvz(2*lidz+1,lidy-2,lidx)))+
//               HC4*(__h2f(lvz(2*lidz+1,lidy+4,lidx))-__h2f(lvz(2*lidz+1,lidy-3,lidx)))+
//               HC5*(__h2f(lvz(2*lidz+1,lidy+5,lidx))-__h2f(lvz(2*lidz+1,lidy-4,lidx)))+
//               HC6*(__h2f(lvz(2*lidz+1,lidy+6,lidx))-__h2f(lvz(2*lidz+1,lidy-5,lidx))));
//
//        vzz.x = (HC1*(__h2f(lvz(2*lidz,lidy,lidx))  -__h2f(lvz(2*lidz-1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1,lidy,lidx))-__h2f(lvz(2*lidz-2,lidy,lidx)))+
//               HC3*(__h2f(lvz(2*lidz+2,lidy,lidx))-__h2f(lvz(2*lidz-3,lidy,lidx)))+
//               HC4*(__h2f(lvz(2*lidz+3,lidy,lidx))-__h2f(lvz(2*lidz-4,lidy,lidx)))+
//               HC5*(__h2f(lvz(2*lidz+4,lidy,lidx))-__h2f(lvz(2*lidz-5,lidy,lidx)))+
//               HC6*(__h2f(lvz(2*lidz+5,lidy,lidx))-__h2f(lvz(2*lidz-6,lidy,lidx))));
//        vzz.y = (HC1*(__h2f(lvz(2*lidz+1,lidy,lidx))  -__h2f(lvz(2*lidz-1+1,lidy,lidx)))+
//               HC2*(__h2f(lvz(2*lidz+1+1,lidy,lidx))-__h2f(lvz(2*lidz-2+1,lidy,lidx)))+
//               HC3*(__h2f(lvz(2*lidz+2+1,lidy,lidx))-__h2f(lvz(2*lidz-3+1,lidy,lidx)))+
//               HC4*(__h2f(lvz(2*lidz+3+1,lidy,lidx))-__h2f(lvz(2*lidz-4+1,lidy,lidx)))+
//               HC5*(__h2f(lvz(2*lidz+4+1,lidy,lidx))-__h2f(lvz(2*lidz-5+1,lidy,lidx)))+
//               HC6*(__h2f(lvz(2*lidz+5+1,lidy,lidx))-__h2f(lvz(2*lidz-6+1,lidy,lidx))));
//#endif
//    }
//
//    // To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
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
//// Correct spatial derivatives to implement CPML
//#if ABS_TYPE==1
//    {
//
//        if (gidz>NZ-NAB-FDOH-1){
//
//            i =gidx-FDOH;
//            j =gidy-FDOH;
//            k =gidz - NZ+NAB+FDOH+NAB;
//            ind=2*NAB-1-k;
//
//            psi_vxz(k,j,i) = b_z_half[ind] * psi_vxz(k,j,i) + a_z_half[ind] * vxz;
//            vxz = vxz / K_z_half[ind] + psi_vxz(k,j,i);
//            psi_vyz(k,j,i) = b_z_half[ind] * psi_vyz(k,j,i) + a_z_half[ind] * vyz;
//            vyz = vyz / K_z_half[ind] + psi_vyz(k,j,i);
//            psi_vzz(k,j,i) = b_z[ind+1] * psi_vzz(k,j,i) + a_z[ind+1] * vzz;
//            vzz = vzz / K_z[ind+1] + psi_vzz(k,j,i);
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
//
//            psi_vxz(k,j,i) = b_z_half[k] * psi_vxz(k,j,i) + a_z_half[k] * vxz;
//            vxz = vxz / K_z_half[k] + psi_vxz(k,j,i);
//            psi_vyz(k,j,i) = b_z_half[k] * psi_vyz(k,j,i) + a_z_half[k] * vyz;
//            vyz = vyz / K_z_half[k] + psi_vyz(k,j,i);
//            psi_vzz(k,j,i) = b_z[k] * psi_vzz(k,j,i) + a_z[k] * vzz;
//            vzz = vzz / K_z[k] + psi_vzz(k,j,i);
//
//
//        }
//#endif
//
//        if (gidy-FDOH<NAB){
//            i =gidx-FDOH;
//            j =gidy-FDOH;
//            k =gidz-FDOH;
//
//            psi_vxy(k,j,i) = b_y_half[j] * psi_vxy(k,j,i) + a_y_half[j] * vxy;
//            vxy = vxy / K_y_half[j] + psi_vxy(k,j,i);
//            psi_vyy(k,j,i) = b_y[j] * psi_vyy(k,j,i) + a_y[j] * vyy;
//            vyy = vyy / K_y[j] + psi_vyy(k,j,i);
//            psi_vzy(k,j,i) = b_y_half[j] * psi_vzy(k,j,i) + a_y_half[j] * vzy;
//            vzy = vzy / K_y_half[j] + psi_vzy(k,j,i);
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
//
//            psi_vxy(k,j,i) = b_y_half[ind] * psi_vxy(k,j,i) + a_y_half[ind] * vxy;
//            vxy = vxy / K_y_half[ind] + psi_vxy(k,j,i);
//            psi_vyy(k,j,i) = b_y[ind+1] * psi_vyy(k,j,i) + a_y[ind+1] * vyy;
//            vyy = vyy / K_y[ind+1] + psi_vyy(k,j,i);
//            psi_vzy(k,j,i) = b_y_half[ind] * psi_vzy(k,j,i) + a_y_half[ind] * vzy;
//            vzy = vzy / K_y_half[ind] + psi_vzy(k,j,i);
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
//            psi_vxx(k,j,i) = b_x[i] * psi_vxx(k,j,i) + a_x[i] * vxx;
//            vxx = vxx / K_x[i] + psi_vxx(k,j,i);
//            psi_vyx(k,j,i) = b_x_half[i] * psi_vyx(k,j,i) + a_x_half[i] * vyx;
//            vyx = vyx / K_x_half[i] + psi_vyx(k,j,i);
//            psi_vzx(k,j,i) = b_x_half[i] * psi_vzx(k,j,i) + a_x_half[i] * vzx;
//            vzx = vzx / K_x_half[i] + psi_vzx(k,j,i);
//
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
//
//            psi_vxx(k,j,i) = b_x[ind+1] * psi_vxx(k,j,i) + a_x[ind+1] * vxx;
//            vxx = vxx /K_x[ind+1] + psi_vxx(k,j,i);
//            psi_vyx(k,j,i) = b_x_half[ind] * psi_vyx(k,j,i) + a_x_half[ind] * vyx;
//            vyx = vyx  /K_x_half[ind] + psi_vyx(k,j,i);
//            psi_vzx(k,j,i) = b_x_half[ind] * psi_vzx(k,j,i) + a_x_half[ind] * vzx;
//            vzx = vzx / K_x_half[ind]  +psi_vzx(k,j,i);
//
//
//        }
//#endif
//    }
//#endif
//
//// Read model parameters into local memory
//    {
//#if LVE==0
//
//        fipkp=muipkp(gidz,gidy,gidx);
//        fipkp.x=scalbnf(fipkp.x*DTDH,scaler_sxx);
//        fipkp.y=scalbnf(fipkp.y*DTDH,scaler_sxx);
//        fjpkp=mujpkp(gidz,gidy,gidx);
//        fjpkp.x=scalbnf(fjpkp.x*DTDH,scaler_sxx);
//        fjpkp.y=scalbnf(fjpkp.y*DTDH,scaler_sxx);
//        fipjp=muipjp(gidz,gidy,gidx);
//        fipjp.x=scalbnf(fipjp.x*DTDH,scaler_sxx);
//        fipjp.y=scalbnf(fipjp.y*DTDH,scaler_sxx);
//        f=mu(gidz,gidy,gidx);
//        f.x=scalbnf(f.x*2.0*DTDH,scaler_sxx);
//        f.y=scalbnf(f.y*2.0*DTDH,scaler_sxx);
//        g=M(gidz,gidy,gidx);
//        g.x=scalbnf(g.x*DTDH,scaler_sxx);
//        g.y=scalbnf(g.y*DTDH,scaler_sxx);
//
//#else
//
//        lM=M(gidz,gidy,gidx);
//        lmu=mu(gidz,gidy,gidx);
//        lmuipkp=muipkp(gidz,gidy,gidx);
//        lmuipjp=muipjp(gidz,gidy,gidx);
//        lmujpkp=mujpkp(gidz,gidy,gidx);
//        ltaup=taup(gidz,gidy,gidx);
//        ltaus=taus(gidz,gidy,gidx);
//        ltausipkp=tausipkp(gidz,gidy,gidx);
//        ltausipjp=tausipjp(gidz,gidy,gidx);
//        ltausjpkp=tausjpkp(gidz,gidy,gidx);
//
//        for (l=0;l<LVE;l++){
//            leta[l]=eta[l];
//        }
//
//
//        fipkp.x=scalbnf(lmuipkp.x*DTDH*(1.0+ (float)LVE*ltausipkp.x),scaler_sxx);
//        fipkp.y=scalbnf(lmuipkp.y*DTDH*(1.0+ (float)LVE*ltausipkp.y),scaler_sxx);
//        fjpkp.x=scalbnf(lmujpkp.x*DTDH*(1.0+ (float)LVE*ltausjpkp.x),scaler_sxx);
//        fjpkp.y=scalbnf(lmujpkp.y*DTDH*(1.0+ (float)LVE*ltausjpkp.y),scaler_sxx);
//        fipjp.x=scalbnf(lmuipjp.x*DTDH*(1.0+ (float)LVE*ltausipjp.x),scaler_sxx);
//        fipjp.y=scalbnf(lmuipjp.y*DTDH*(1.0+ (float)LVE*ltausipjp.y),scaler_sxx);
//        g.x=scalbnf(lM.x*(1.0+(float)LVE*ltaup.x)*DTDH,scaler_sxx);
//        g.y=scalbnf(lM.y*(1.0+(float)LVE*ltaup.y)*DTDH,scaler_sxx);
//        f.x=scalbnf(2.0*lmu.x*(1.0+(float)LVE*ltaus.x)*DTDH,scaler_sxx);
//        f.y=scalbnf(2.0*lmu.y*(1.0+(float)LVE*ltaus.y)*DTDH,scaler_sxx);
//        dipkp.x=scalbnf(lmuipkp.x*ltausipkp.x/DH,scaler_sxx);
//        dipkp.y=scalbnf(lmuipkp.y*ltausipkp.y/DH,scaler_sxx);
//        djpkp.x=scalbnf(lmujpkp.x*ltausjpkp.x/DH,scaler_sxx);
//        djpkp.y=scalbnf(lmujpkp.y*ltausjpkp.y/DH,scaler_sxx);
//        dipjp.x=scalbnf(lmuipjp.x*ltausipjp.x/DH,scaler_sxx);
//        dipjp.y=scalbnf(lmuipjp.y*ltausipjp.y/DH,scaler_sxx);
//        d.x=scalbnf(2.0*lmu.x*ltaus.x/DH,scaler_sxx);
//        d.y=scalbnf(2.0*lmu.y*ltaus.y/DH,scaler_sxx);
//        e.x=scalbnf(lM.x*ltaup.x/DH,scaler_sxx);
//        e.y=scalbnf(lM.y*ltaup.y/DH,scaler_sxx);
//
//#endif
//    }
//
//// Update the stresses
//    {
//#if LVE==0
//
//        lsxx = __h22f2(sxx(gidz,gidy,gidx));
//        lsyy = __h22f2(syy(gidz,gidy,gidx));
//        lszz = __h22f2(szz(gidz,gidy,gidx));
//        lsxz = __h22f2(sxz(gidz,gidy,gidx));
//        lsxy = __h22f2(sxy(gidz,gidy,gidx));
//        lsyz = __h22f2(syz(gidz,gidy,gidx));
//
//        vxyyx.x=vxy.x+vyx.x;
//        vxyyx.y=vxy.y+vyx.y;
//        vyzzy.x=vyz.x+vzy.x;
//        vyzzy.y=vyz.y+vzy.y;
//        vxzzx.x=vxz.x+vzx.x;
//        vxzzx.y=vxz.y+vzx.y;
//        vxxyyzz.x=vxx.x+vyy.x+vzz.x;
//        vxxyyzz.y=vxx.y+vyy.y+vzz.y;
//        vyyzz.x=vyy.x+vzz.x;
//        vyyzz.y=vyy.y+vzz.y;
//        vxxzz.x=vxx.x+vzz.x;
//        vxxzz.y=vxx.y+vzz.y;
//        vxxyy.x=vxx.x+vyy.x;
//        vxxyy.y=vxx.y+vyy.y;
//
//        lsxy.x+=(fipjp.x*vxyyx.x);
//        lsxy.y+=(fipjp.y*vxyyx.y);
//        lsyz.x+=(fjpkp.x*vyzzy.x);
//        lsyz.y+=(fjpkp.y*vyzzy.y);
//        lsxz.x+=(fipkp.x*vxzzx.x);
//        lsxz.y+=(fipkp.y*vxzzx.y);
//
//        lsxx.x+=((g.x*vxxyyzz.x)-(f.x*vyyzz.x)) ;
//        lsxx.y+=((g.y*vxxyyzz.y)-(f.y*vyyzz.y)) ;
//        lsyy.x+=((g.x*vxxyyzz.x)-(f.x*vxxzz.x)) ;
//        lsyy.y+=((g.y*vxxyyzz.y)-(f.y*vxxzz.y)) ;
//        lszz.x+=((g.x*vxxyyzz.x)-(f.x*vxxyy.x)) ;
//        lszz.y+=((g.y*vxxyyzz.y)-(f.y*vxxyy.y)) ;
//
//
//#else
//
//        /* computing sums of the old memory variables */
//        sumrxy.x=sumryz.x=sumrxz.x=sumrxx.x=sumryy.x=sumrzz.x=0;
//        sumrxy.y=sumryz.y=sumrxz.y=sumrxx.y=sumryy.y=sumrzz.y=0;
//        for (l=0;l<LVE;l++){
//
//            lrxx[l] = __h22f2(rxx(gidz,gidy,gidx,l));
//            lryy[l] = __h22f2(ryy(gidz,gidy,gidx,l));
//            lrzz[l] = __h22f2(rzz(gidz,gidy,gidx,l));
//            lrxz[l] = __h22f2(rxz(gidz,gidy,gidx,l));
//            lrxy[l] = __h22f2(rxy(gidz,gidy,gidx,l));
//            lryz[l] = __h22f2(ryz(gidz,gidy,gidx,l));
//
//            sumrxy.x+=lrxy[l].x;
//            sumrxy.y+=lrxy[l].y;
//            sumryz.x+=lryz[l].x;
//            sumryz.y+=lryz[l].y;
//            sumrxz.x+=lrxz[l].x;
//            sumrxz.y+=lrxz[l].y;
//            sumrxx.x+=lrxx[l].x;
//            sumrxx.y+=lrxx[l].y;
//            sumryy.x+=lryy[l].x;
//            sumryy.y+=lryy[l].y;
//            sumrzz.x+=lrzz[l].x;
//            sumrzz.y+=lrzz[l].y;
//        }
//
//        /* updating components of the stress tensor, partially */
//
//        lsxx = __h22f2(sxx(gidz,gidy,gidx));
//        lsyy = __h22f2(syy(gidz,gidy,gidx));
//        lszz = __h22f2(szz(gidz,gidy,gidx));
//        lsxz = __h22f2(sxz(gidz,gidy,gidx));
//        lsxy = __h22f2(sxy(gidz,gidy,gidx));
//        lsyz = __h22f2(syz(gidz,gidy,gidx));
//
//        vxyyx.x=vxy.x+vyx.x;
//        vxyyx.y=vxy.y+vyx.y;
//        vyzzy.x=vyz.x+vzy.x;
//        vyzzy.y=vyz.y+vzy.y;
//        vxzzx.x=vxz.x+vzx.x;
//        vxzzx.y=vxz.y+vzx.y;
//        vxxyyzz.x=vxx.x+vyy.x+vzz.x;
//        vxxyyzz.y=vxx.y+vyy.y+vzz.y;
//        vyyzz.x=vyy.x+vzz.x;
//        vyyzz.y=vyy.y+vzz.y;
//        vxxzz.x=vxx.x+vzz.x;
//        vxxzz.y=vxx.y+vzz.y;
//        vxxyy.x=vxx.x+vyy.x;
//        vxxyy.y=vxx.y+vyy.y;
//
//        lsxy.x+=(fipjp.x*vxyyx.x)+(DT2*sumrxy.x);
//        lsxy.y+=(fipjp.y*vxyyx.y)+(DT2*sumrxy.y);
//        lsyz.x+=(fjpkp.x*vyzzy.x)+(DT2*sumryz.x);
//        lsyz.y+=(fjpkp.y*vyzzy.y)+(DT2*sumryz.y);
//        lsxz.x+=(fipkp.x*vxzzx.x)+(DT2*sumrxz.x);
//        lsxz.y+=(fipkp.y*vxzzx.y)+(DT2*sumrxz.y);
//        lsxx.x+=((g.x*vxxyyzz.x)-(f.x*vyyzz.x))+(DT2*sumrxx.x);
//        lsxx.y+=((g.y*vxxyyzz.y)-(f.y*vyyzz.y))+(DT2*sumrxx.y);
//        lsyy.x+=((g.x*vxxyyzz.x)-(f.x*vxxzz.x))+(DT2*sumryy.x);
//        lsyy.y+=((g.y*vxxyyzz.y)-(f.y*vxxzz.y))+(DT2*sumryy.y);
//        lszz.x+=((g.x*vxxyyzz.x)-(f.x*vxxyy.x))+(DT2*sumrzz.x);
//        lszz.y+=((g.y*vxxyyzz.y)-(f.y*vxxyy.y))+(DT2*sumrzz.y);
//
//
//        sumrxy.x=sumryz.x=sumrxz.x=sumrxx.x=sumryy.x=sumrzz.x=0;
//        sumrxy.y=sumryz.y=sumrxz.y=sumrxx.y=sumryy.y=sumrzz.y=0;
//        for (l=0;l<LVE;l++){
//            b=1.0/(1.0+(leta[l]*0.5));
//            c=1.0-(leta[l]*0.5);
//
//            lrxy[l].x=b*(lrxy[l].x*c-leta[l]*(dipjp.x*vxyyx.x));
//            lrxy[l].y=b*(lrxy[l].y*c-leta[l]*(dipjp.y*vxyyx.y));
//            lryz[l].x=b*(lryz[l].x*c-leta[l]*(djpkp.x*vyzzy.x));
//            lryz[l].y=b*(lryz[l].y*c-leta[l]*(djpkp.y*vyzzy.y));
//            lrxz[l].x=b*(lrxz[l].x*c-leta[l]*(dipkp.x*vxzzx.x));
//            lrxz[l].y=b*(lrxz[l].y*c-leta[l]*(dipkp.y*vxzzx.y));
//            lrxx[l].x=b*(lrxx[l].x*c-leta[l]*((e.x*vxxyyzz.x)-(d.x*vyyzz.x)));
//            lrxx[l].y=b*(lrxx[l].y*c-leta[l]*((e.y*vxxyyzz.y)-(d.y*vyyzz.y)));
//            lryy[l].x=b*(lryy[l].x*c-leta[l]*((e.x*vxxyyzz.x)-(d.x*vxxzz.x)));
//            lryy[l].y=b*(lryy[l].y*c-leta[l]*((e.y*vxxyyzz.y)-(d.y*vxxzz.y)));
//            lrzz[l].x=b*(lrzz[l].x*c-leta[l]*((e.x*vxxyyzz.x)-(d.x*vxxyy.x)));
//            lrzz[l].y=b*(lrzz[l].y*c-leta[l]*((e.y*vxxyyzz.y)-(d.y*vxxyy.y)));
//
//            sumrxy.x+=lrxy[l].x;
//            sumrxy.y+=lrxy[l].y;
//            sumryz.x+=lryz[l].x;
//            sumryz.y+=lryz[l].y;
//            sumrxz.x+=lrxz[l].x;
//            sumrxz.y+=lrxz[l].y;
//            sumrxx.x+=lrxx[l].x;
//            sumrxx.y+=lrxx[l].y;
//            sumryy.x+=lryy[l].x;
//            sumryy.y+=lryy[l].y;
//            sumrzz.x+=lrzz[l].x;
//            sumrzz.y+=lrzz[l].y;
//
//            rxx(gidz,gidy,gidx,l)=__f22h2(lrxx[l]);
//            ryy(gidz,gidy,gidx,l)=__f22h2(lryy[l]);
//            rzz(gidz,gidy,gidx,l)=__f22h2(lrzz[l]);
//            rxz(gidz,gidy,gidx,l)=__f22h2(lrxz[l]);
//            rxy(gidz,gidy,gidx,l)=__f22h2(lrxy[l]);
//            ryz(gidz,gidy,gidx,l)=__f22h2(lryz[l]);
//        }
//
//        /* and now the components of the stress tensor are
//         completely updated */
//        lsxy.x+=(DT2*sumrxy.x);
//        lsxy.y+=(DT2*sumrxy.y);
//        lsyz.x+=(DT2*sumryz.x);
//        lsyz.y+=(DT2*sumryz.y);
//        lsxz.x+=(DT2*sumrxz.x);
//        lsxz.y+=(DT2*sumrxz.y);
//        lsxx.x+=(DT2*sumrxx.x);
//        lsxx.y+=(DT2*sumrxx.y);
//        lsyy.x+=(DT2*sumryy.x);
//        lsyy.y+=(DT2*sumryy.y);
//        lszz.x+=(DT2*sumrzz.x);
//        lszz.y+=(DT2*sumrzz.y);
//
//#endif
//    }
//
//// Absorbing boundary
//#if abstype==2
//    {
//
//#if FREESURF==0
//        if (2*gidz-FDOH<NAB){
//            lsxy.x*=taper[2*gidz-FDOH];
//            lsxy.y*=taper[2*gidz-FDOH];
//            lsyz.x*=taper[2*gidz-FDOH];
//            lsyz.y*=taper[2*gidz-FDOH];
//            lsxz.x*=taper[2*gidz-FDOH];
//            lsxz.y*=taper[2*gidz-FDOH];
//            lsxx.x*=taper[2*gidz-FDOH];
//            lsxx.y*=taper[2*gidz-FDOH];
//            lsyy.x*=taper[2*gidz-FDOH];
//            lsyy.y*=taper[2*gidz-FDOH];
//            lszz.x*=taper[2*gidz-FDOH];
//            lszz.y*=taper[2*gidz-FDOH];
//        }
//#endif
//
//        if (2*gidz>2*NZ-NAB-FDOH-1){
//            lsxy.x*=taper[2*NZ-FDOH-2*gidz-1];
//            lsxy.y*=taper[2*NZ-FDOH-2*gidz-1];
//            lsyz.x*=taper[2*NZ-FDOH-2*gidz-1];
//            lsyz.y*=taper[2*NZ-FDOH-2*gidz-1];
//            lsxz.x*=taper[2*NZ-FDOH-2*gidz-1];
//            lsxz.y*=taper[2*NZ-FDOH-2*gidz-1];
//            lsxx.x*=taper[2*NZ-FDOH-2*gidz-1];
//            lsxx.y*=taper[2*NZ-FDOH-2*gidz-1];
//            lsyy.x*=taper[2*NZ-FDOH-2*gidz-1];
//            lsyy.y*=taper[2*NZ-FDOH-2*gidz-1];
//            lszz.x*=taper[2*NZ-FDOH-2*gidz-1];
//            lszz.y*=taper[2*NZ-FDOH-2*gidz-1];
//        }
//        if (gidy-FDOH<NAB){
//            lsxy.x*=taper[gidy-FDOH];
//            lsxy.y*=taper[gidy-FDOH];
//            lsyz.x*=taper[gidy-FDOH];
//            lsyz.y*=taper[gidy-FDOH];
//            lsxz.x*=taper[gidy-FDOH];
//            lsxz.y*=taper[gidy-FDOH];
//            lsxx.x*=taper[gidy-FDOH];
//            lsxx.y*=taper[gidy-FDOH];
//            lsyy.x*=taper[gidy-FDOH];
//            lsyy.y*=taper[gidy-FDOH];
//            lszz.x*=taper[gidy-FDOH];
//            lszz.y*=taper[gidy-FDOH];
//        }
//
//        if (gidy>NY-NAB-FDOH-1){
//            lsxy.x*=taper[NY-FDOH-gidy-1];
//            lsxy.y*=taper[NY-FDOH-gidy-1];
//            lsyz.x*=taper[NY-FDOH-gidy-1];
//            lsyz.y*=taper[NY-FDOH-gidy-1];
//            lsxz.x*=taper[NY-FDOH-gidy-1];
//            lsxz.y*=taper[NY-FDOH-gidy-1];
//            lsxx.x*=taper[NY-FDOH-gidy-1];
//            lsxx.y*=taper[NY-FDOH-gidy-1];
//            lsyy.x*=taper[NY-FDOH-gidy-1];
//            lsyy.y*=taper[NY-FDOH-gidy-1];
//            lszz.x*=taper[NY-FDOH-gidy-1];
//            lszz.y*=taper[NY-FDOH-gidy-1];
//        }
//
//#if DEVID==0 & MYLOCALID==0
//        if (gidx-FDOH<NAB){
//            lsxy.x*=taper[gidx-FDOH];
//            lsxy.y*=taper[gidx-FDOH];
//            lsyz.x*=taper[gidx-FDOH];
//            lsyz.y*=taper[gidx-FDOH];
//            lsxz.x*=taper[gidx-FDOH];
//            lsxz.y*=taper[gidx-FDOH];
//            lsxx.x*=taper[gidx-FDOH];
//            lsxx.y*=taper[gidx-FDOH];
//            lsyy.x*=taper[gidx-FDOH];
//            lsyy.y*=taper[gidx-FDOH];
//            lszz.x*=taper[gidx-FDOH];
//            lszz.y*=taper[gidx-FDOH];
//        }
//#endif
//
//#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//        if (gidx>NX-NAB-FDOH-1){
//            lsxy.x*=taper[NX-FDOH-gidx-1];
//            lsxy.y*=taper[NX-FDOH-gidx-1];
//            lsyz.x*=taper[NX-FDOH-gidx-1];
//            lsyz.y*=taper[NX-FDOH-gidx-1];
//            lsxz.x*=taper[NX-FDOH-gidx-1];
//            lsxz.y*=taper[NX-FDOH-gidx-1];
//            lsxx.x*=taper[NX-FDOH-gidx-1];
//            lsxx.y*=taper[NX-FDOH-gidx-1];
//            lsyy.x*=taper[NX-FDOH-gidx-1];
//            lsyy.y*=taper[NX-FDOH-gidx-1];
//            lszz.x*=taper[NX-FDOH-gidx-1];
//            lszz.y*=taper[NX-FDOH-gidx-1];
//        }
//#endif
//    }
//#endif
//
//
//    sxx(gidz, gidy, gidx)=__f22h2(lsxx);
//    syy(gidz, gidy, gidx)=__f22h2(lsyy);
//    szz(gidz, gidy, gidx)=__f22h2(lszz);
//    sxz(gidz, gidy, gidx)=__f22h2(lsxz);
//    sxy(gidz, gidy, gidx)=__f22h2(lsxy);
//    syz(gidz, gidy, gidx)=__f22h2(lsyz);
//
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
#define mul2 __h2div
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
extern "C" __device__ float2 scalbnf2 (float2 a ){
    
    float2 output;
    output.x  = scalbnf(a.x);
    output.y  = scalbnf(a.y);
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



extern "C" __global__ void update_s(int offcomm,
                                    __pprec *muipjp, __pprec *mujpkp, __pprec *muipkp,
                                    __pprec *M, __pprec *mu,__prec2 *sxx,__prec2 *sxy,__prec2 *sxz,
                                    __prec2 *syy,__prec2 *syz,__prec2 *szz,
                                    __prec2 *vx,__prec2 *vy,__prec2 *vz, float *taper
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
    
    //Define and load private parameters and variables
    __cprec lsxx = __h22f2(sxx(gidz,gidy,gidx));
    __cprec lsxy = __h22f2(sxy(gidz,gidy,gidx));
    __cprec lsxz = __h22f2(sxz(gidz,gidy,gidx));
    __cprec lsyy = __h22f2(syy(gidz,gidy,gidx));
    __cprec lsyz = __h22f2(syz(gidz,gidy,gidx));
    __cprec lszz = __h22f2(szz(gidz,gidy,gidx));
    __cprec lM = __pconv(M(gidz,gidy,gidx));
    __cprec lmu = __pconv(mu(gidz,gidy,gidx));
    __cprec lmuipjp = __pconv(muipjp(gidz,gidy,gidx));
    __cprec lmuipkp = __pconv(muipkp(gidz,gidy,gidx));
    __cprec lmujpkp = __pconv(mujpkp(gidz,gidy,gidx));
    
    //Define private derivatives
    __cprec vx_x2;
    __cprec vx_y1;
    __cprec vx_z1;
    __cprec vy_x1;
    __cprec vy_y2;
    __cprec vy_z1;
    __cprec vz_x1;
    __cprec vz_y1;
    __cprec vz_z2;
    
    //Local memory definitions if local is used
#if LOCAL_OFF==0
#define lvz lvar
#define lvy lvar
#define lvx lvar
#define lvz2 lvar2
#define lvy2 lvar2
#define lvx2 lvar2
    
    //Local memory definitions if local is not used
#elif LOCAL_OFF==1
    
#define lvz vz
#define lvy vy
#define lvx vx
#define lidz gidz
#define lidy gidy
#define lidx gidx
    
#endif
    
    //Calculation of the spatial derivatives
    {
#if LOCAL_OFF==0
        lvz2(lidz,lidy,lidx)=vz(gidz,gidy,gidx);
        if (lidz<FDOH)
            lvz2(lidz-FDOH/2,lidy,lidx)=vz(gidz-FDOH/2,gidy,gidx);
        if (lidz>(lsizez-FDOH-1))
            lvz2(lidz+FDOH/2,lidy,lidx)=vz(gidz+FDOH/2,gidy,gidx);
        if (lidy<2*FDOH)
            lvz2(lidz,lidy-FDOH,lidx)=vz(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lvz2(lidz,lidy+lsizey-3*FDOH,lidx)=vz(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lvz2(lidz,lidy+FDOH,lidx)=vz(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lvz2(lidz,lidy-lsizey+3*FDOH,lidx)=vz(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lvz2(lidz,lidy,lidx-FDOH)=vz(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvz2(lidz,lidy,lidx+lsizex-3*FDOH)=vz(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvz2(lidz,lidy,lidx+FDOH)=vz(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvz2(lidz,lidy,lidx-lsizex+3*FDOH)=vz(gidz,gidy,gidx-lsizex+3*FDOH);
        __syncthreads();
#endif
        
#if   FDOH == 1
        vz_x1=mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidy,lidx+1)), __h22f2(lvz2(lidz,lidy,lidx))));
#elif FDOH == 2
        vz_x1=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidy,lidx+1)), __h22f2(lvz2(lidz,lidy,lidx)))),
                   mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidy,lidx+2)), __h22f2(lvz2(lidz,lidy,lidx-1)))));
#elif FDOH == 3
        vz_x1=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidy,lidx+1)), __h22f2(lvz2(lidz,lidy,lidx)))),
                        mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidy,lidx+2)), __h22f2(lvz2(lidz,lidy,lidx-1))))),
                   mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidy,lidx+3)), __h22f2(lvz2(lidz,lidy,lidx-2)))));
#elif FDOH == 4
        vz_x1=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidy,lidx+1)), __h22f2(lvz2(lidz,lidy,lidx)))),
                             mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidy,lidx+2)), __h22f2(lvz2(lidz,lidy,lidx-1))))),
                        mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidy,lidx+3)), __h22f2(lvz2(lidz,lidy,lidx-2))))),
                   mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidy,lidx+4)), __h22f2(lvz2(lidz,lidy,lidx-3)))));
#elif FDOH == 5
        vz_x1=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidy,lidx+1)), __h22f2(lvz2(lidz,lidy,lidx)))),
                                  mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidy,lidx+2)), __h22f2(lvz2(lidz,lidy,lidx-1))))),
                             mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidy,lidx+3)), __h22f2(lvz2(lidz,lidy,lidx-2))))),
                        mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidy,lidx+4)), __h22f2(lvz2(lidz,lidy,lidx-3))))),
                   mul2( f2h2(HC5), sub2(__h22f2(lvz2(lidz,lidy,lidx+5)), __h22f2(lvz2(lidz,lidy,lidx-4)))));
#elif FDOH == 6
        vz_x1=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidy,lidx+1)), __h22f2(lvz2(lidz,lidy,lidx)))),
                                       mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidy,lidx+2)), __h22f2(lvz2(lidz,lidy,lidx-1))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidy,lidx+3)), __h22f2(lvz2(lidz,lidy,lidx-2))))),
                             mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidy,lidx+4)), __h22f2(lvz2(lidz,lidy,lidx-3))))),
                        mul2( f2h2(HC5), sub2(__h22f2(lvz2(lidz,lidy,lidx+5)), __h22f2(lvz2(lidz,lidy,lidx-4))))),
                   mul2( f2h2(HC6), sub2(__h22f2(lvz2(lidz,lidy,lidx+6)), __h22f2(lvz2(lidz,lidy,lidx-5)))));
#endif
        
#if   FDOH == 1
        vz_y1=mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidy+1,lidx)), __h22f2(lvz2(lidz,lidy,lidx))));
#elif FDOH == 2
        vz_y1=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidy+1,lidx)), __h22f2(lvz2(lidz,lidy,lidx)))),
                   mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidy+2,lidx)), __h22f2(lvz2(lidz,lidy-1,lidx)))));
#elif FDOH == 3
        vz_y1=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidy+1,lidx)), __h22f2(lvz2(lidz,lidy,lidx)))),
                        mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidy+2,lidx)), __h22f2(lvz2(lidz,lidy-1,lidx))))),
                   mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidy+3,lidx)), __h22f2(lvz2(lidz,lidy-2,lidx)))));
#elif FDOH == 4
        vz_y1=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidy+1,lidx)), __h22f2(lvz2(lidz,lidy,lidx)))),
                             mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidy+2,lidx)), __h22f2(lvz2(lidz,lidy-1,lidx))))),
                        mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidy+3,lidx)), __h22f2(lvz2(lidz,lidy-2,lidx))))),
                   mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidy+4,lidx)), __h22f2(lvz2(lidz,lidy-3,lidx)))));
#elif FDOH == 5
        vz_y1=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidy+1,lidx)), __h22f2(lvz2(lidz,lidy,lidx)))),
                                  mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidy+2,lidx)), __h22f2(lvz2(lidz,lidy-1,lidx))))),
                             mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidy+3,lidx)), __h22f2(lvz2(lidz,lidy-2,lidx))))),
                        mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidy+4,lidx)), __h22f2(lvz2(lidz,lidy-3,lidx))))),
                   mul2( f2h2(HC5), sub2(__h22f2(lvz2(lidz,lidy+5,lidx)), __h22f2(lvz2(lidz,lidy-4,lidx)))));
#elif FDOH == 6
        vz_y1=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidy+1,lidx)), __h22f2(lvz2(lidz,lidy,lidx)))),
                                       mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidy+2,lidx)), __h22f2(lvz2(lidz,lidy-1,lidx))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidy+3,lidx)), __h22f2(lvz2(lidz,lidy-2,lidx))))),
                             mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidy+4,lidx)), __h22f2(lvz2(lidz,lidy-3,lidx))))),
                        mul2( f2h2(HC5), sub2(__h22f2(lvz2(lidz,lidy+5,lidx)), __h22f2(lvz2(lidz,lidy-4,lidx))))),
                   mul2( f2h2(HC6), sub2(__h22f2(lvz2(lidz,lidy+6,lidx)), __h22f2(lvz2(lidz,lidy-5,lidx)))));
#endif
        
#if   FDOH == 1
        vz_z2=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidy,lidx)))));
#elif FDOH == 2
        vz_z2=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidy,lidx))))),
                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidy,lidx))))));
#elif FDOH == 3
        vz_z2=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidy,lidx))))),
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidy,lidx)))))),
                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidy,lidx))))));
#elif FDOH == 4
        vz_z2=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidy,lidx))))),
                             mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidy,lidx)))))),
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidy,lidx)))))),
                   mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvz(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-4,lidy,lidx))))));
#elif FDOH == 5
        vz_z2=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidy,lidx))))),
                                  mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidy,lidx)))))),
                             mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidy,lidx)))))),
                        mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvz(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-4,lidy,lidx)))))),
                   mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvz(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-5,lidy,lidx))))));
#elif FDOH == 6
        vz_z2=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidy,lidx))))),
                                       mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidy,lidx)))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidy,lidx)))))),
                             mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvz(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-4,lidy,lidx)))))),
                        mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvz(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-5,lidy,lidx)))))),
                   mul2( f2h2(HC6), sub2(__h22f2(__hp(&lvz(2*lidz+5,lidy,lidx))), __h22f2(__hp(&lvz(2*lidz-6,lidy,lidx))))));
#endif
        
#if LOCAL_OFF==0
        __syncthreads();
        lvy2(lidz,lidy,lidx)=vy(gidz,gidy,gidx);
        if (lidz<FDOH)
            lvy2(lidz-FDOH/2,lidy,lidx)=vy(gidz-FDOH/2,gidy,gidx);
        if (lidz>(lsizez-FDOH-1))
            lvy2(lidz+FDOH/2,lidy,lidx)=vy(gidz+FDOH/2,gidy,gidx);
        if (lidy<2*FDOH)
            lvy2(lidz,lidy-FDOH,lidx)=vy(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lvy2(lidz,lidy+lsizey-3*FDOH,lidx)=vy(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lvy2(lidz,lidy+FDOH,lidx)=vy(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lvy2(lidz,lidy-lsizey+3*FDOH,lidx)=vy(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lvy2(lidz,lidy,lidx-FDOH)=vy(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvy2(lidz,lidy,lidx+lsizex-3*FDOH)=vy(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvy2(lidz,lidy,lidx+FDOH)=vy(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvy2(lidz,lidy,lidx-lsizex+3*FDOH)=vy(gidz,gidy,gidx-lsizex+3*FDOH);
        __syncthreads();
#endif
        
#if   FDOH == 1
        vy_x1=mul2( f2h2(HC1), sub2(__h22f2(lvy2(lidz,lidy,lidx+1)), __h22f2(lvy2(lidz,lidy,lidx))));
#elif FDOH == 2
        vy_x1=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(lvy2(lidz,lidy,lidx+1)), __h22f2(lvy2(lidz,lidy,lidx)))),
                   mul2( f2h2(HC2), sub2(__h22f2(lvy2(lidz,lidy,lidx+2)), __h22f2(lvy2(lidz,lidy,lidx-1)))));
#elif FDOH == 3
        vy_x1=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(lvy2(lidz,lidy,lidx+1)), __h22f2(lvy2(lidz,lidy,lidx)))),
                        mul2( f2h2(HC2), sub2(__h22f2(lvy2(lidz,lidy,lidx+2)), __h22f2(lvy2(lidz,lidy,lidx-1))))),
                   mul2( f2h2(HC3), sub2(__h22f2(lvy2(lidz,lidy,lidx+3)), __h22f2(lvy2(lidz,lidy,lidx-2)))));
#elif FDOH == 4
        vy_x1=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(lvy2(lidz,lidy,lidx+1)), __h22f2(lvy2(lidz,lidy,lidx)))),
                             mul2( f2h2(HC2), sub2(__h22f2(lvy2(lidz,lidy,lidx+2)), __h22f2(lvy2(lidz,lidy,lidx-1))))),
                        mul2( f2h2(HC3), sub2(__h22f2(lvy2(lidz,lidy,lidx+3)), __h22f2(lvy2(lidz,lidy,lidx-2))))),
                   mul2( f2h2(HC4), sub2(__h22f2(lvy2(lidz,lidy,lidx+4)), __h22f2(lvy2(lidz,lidy,lidx-3)))));
#elif FDOH == 5
        vy_x1=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(lvy2(lidz,lidy,lidx+1)), __h22f2(lvy2(lidz,lidy,lidx)))),
                                  mul2( f2h2(HC2), sub2(__h22f2(lvy2(lidz,lidy,lidx+2)), __h22f2(lvy2(lidz,lidy,lidx-1))))),
                             mul2( f2h2(HC3), sub2(__h22f2(lvy2(lidz,lidy,lidx+3)), __h22f2(lvy2(lidz,lidy,lidx-2))))),
                        mul2( f2h2(HC4), sub2(__h22f2(lvy2(lidz,lidy,lidx+4)), __h22f2(lvy2(lidz,lidy,lidx-3))))),
                   mul2( f2h2(HC5), sub2(__h22f2(lvy2(lidz,lidy,lidx+5)), __h22f2(lvy2(lidz,lidy,lidx-4)))));
#elif FDOH == 6
        vy_x1=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(lvy2(lidz,lidy,lidx+1)), __h22f2(lvy2(lidz,lidy,lidx)))),
                                       mul2( f2h2(HC2), sub2(__h22f2(lvy2(lidz,lidy,lidx+2)), __h22f2(lvy2(lidz,lidy,lidx-1))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(lvy2(lidz,lidy,lidx+3)), __h22f2(lvy2(lidz,lidy,lidx-2))))),
                             mul2( f2h2(HC4), sub2(__h22f2(lvy2(lidz,lidy,lidx+4)), __h22f2(lvy2(lidz,lidy,lidx-3))))),
                        mul2( f2h2(HC5), sub2(__h22f2(lvy2(lidz,lidy,lidx+5)), __h22f2(lvy2(lidz,lidy,lidx-4))))),
                   mul2( f2h2(HC6), sub2(__h22f2(lvy2(lidz,lidy,lidx+6)), __h22f2(lvy2(lidz,lidy,lidx-5)))));
#endif
        
#if   FDOH == 1
        vy_y2=mul2( f2h2(HC1), sub2(__h22f2(lvy2(lidz,lidy,lidx)), __h22f2(lvy2(lidz,lidy-1,lidx))));
#elif FDOH == 2
        vy_y2=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(lvy2(lidz,lidy,lidx)), __h22f2(lvy2(lidz,lidy-1,lidx)))),
                   mul2( f2h2(HC2), sub2(__h22f2(lvy2(lidz,lidy+1,lidx)), __h22f2(lvy2(lidz,lidy-2,lidx)))));
#elif FDOH == 3
        vy_y2=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(lvy2(lidz,lidy,lidx)), __h22f2(lvy2(lidz,lidy-1,lidx)))),
                        mul2( f2h2(HC2), sub2(__h22f2(lvy2(lidz,lidy+1,lidx)), __h22f2(lvy2(lidz,lidy-2,lidx))))),
                   mul2( f2h2(HC3), sub2(__h22f2(lvy2(lidz,lidy+2,lidx)), __h22f2(lvy2(lidz,lidy-3,lidx)))));
#elif FDOH == 4
        vy_y2=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(lvy2(lidz,lidy,lidx)), __h22f2(lvy2(lidz,lidy-1,lidx)))),
                             mul2( f2h2(HC2), sub2(__h22f2(lvy2(lidz,lidy+1,lidx)), __h22f2(lvy2(lidz,lidy-2,lidx))))),
                        mul2( f2h2(HC3), sub2(__h22f2(lvy2(lidz,lidy+2,lidx)), __h22f2(lvy2(lidz,lidy-3,lidx))))),
                   mul2( f2h2(HC4), sub2(__h22f2(lvy2(lidz,lidy+3,lidx)), __h22f2(lvy2(lidz,lidy-4,lidx)))));
#elif FDOH == 5
        vy_y2=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(lvy2(lidz,lidy,lidx)), __h22f2(lvy2(lidz,lidy-1,lidx)))),
                                  mul2( f2h2(HC2), sub2(__h22f2(lvy2(lidz,lidy+1,lidx)), __h22f2(lvy2(lidz,lidy-2,lidx))))),
                             mul2( f2h2(HC3), sub2(__h22f2(lvy2(lidz,lidy+2,lidx)), __h22f2(lvy2(lidz,lidy-3,lidx))))),
                        mul2( f2h2(HC4), sub2(__h22f2(lvy2(lidz,lidy+3,lidx)), __h22f2(lvy2(lidz,lidy-4,lidx))))),
                   mul2( f2h2(HC5), sub2(__h22f2(lvy2(lidz,lidy+4,lidx)), __h22f2(lvy2(lidz,lidy-5,lidx)))));
#elif FDOH == 6
        vy_y2=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(lvy2(lidz,lidy,lidx)), __h22f2(lvy2(lidz,lidy-1,lidx)))),
                                       mul2( f2h2(HC2), sub2(__h22f2(lvy2(lidz,lidy+1,lidx)), __h22f2(lvy2(lidz,lidy-2,lidx))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(lvy2(lidz,lidy+2,lidx)), __h22f2(lvy2(lidz,lidy-3,lidx))))),
                             mul2( f2h2(HC4), sub2(__h22f2(lvy2(lidz,lidy+3,lidx)), __h22f2(lvy2(lidz,lidy-4,lidx))))),
                        mul2( f2h2(HC5), sub2(__h22f2(lvy2(lidz,lidy+4,lidx)), __h22f2(lvy2(lidz,lidy-5,lidx))))),
                   mul2( f2h2(HC6), sub2(__h22f2(lvy2(lidz,lidy+5,lidx)), __h22f2(lvy2(lidz,lidy-6,lidx)))));
#endif
        
#if   FDOH == 1
        vy_z1=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvy(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz,lidy,lidx)))));
#elif FDOH == 2
        vy_z1=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvy(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz,lidy,lidx))))),
                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvy(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-1,lidy,lidx))))));
#elif FDOH == 3
        vy_z1=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvy(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz,lidy,lidx))))),
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvy(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-1,lidy,lidx)))))),
                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvy(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-2,lidy,lidx))))));
#elif FDOH == 4
        vy_z1=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvy(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz,lidy,lidx))))),
                             mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvy(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-1,lidy,lidx)))))),
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvy(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-2,lidy,lidx)))))),
                   mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvy(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-3,lidy,lidx))))));
#elif FDOH == 5
        vy_z1=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvy(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz,lidy,lidx))))),
                                  mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvy(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-1,lidy,lidx)))))),
                             mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvy(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-2,lidy,lidx)))))),
                        mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvy(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-3,lidy,lidx)))))),
                   mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvy(2*lidz+5,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-4,lidy,lidx))))));
#elif FDOH == 6
        vy_z1=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvy(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz,lidy,lidx))))),
                                       mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvy(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-1,lidy,lidx)))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvy(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-2,lidy,lidx)))))),
                             mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvy(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-3,lidy,lidx)))))),
                        mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvy(2*lidz+5,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-4,lidy,lidx)))))),
                   mul2( f2h2(HC6), sub2(__h22f2(__hp(&lvy(2*lidz+6,lidy,lidx))), __h22f2(__hp(&lvy(2*lidz-5,lidy,lidx))))));
#endif
        
#if LOCAL_OFF==0
        __syncthreads();
        lvx2(lidz,lidy,lidx)=vx(gidz,gidy,gidx);
        if (lidz<FDOH)
            lvx2(lidz-FDOH/2,lidy,lidx)=vx(gidz-FDOH/2,gidy,gidx);
        if (lidz>(lsizez-FDOH-1))
            lvx2(lidz+FDOH/2,lidy,lidx)=vx(gidz+FDOH/2,gidy,gidx);
        if (lidy<2*FDOH)
            lvx2(lidz,lidy-FDOH,lidx)=vx(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lvx2(lidz,lidy+lsizey-3*FDOH,lidx)=vx(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lvx2(lidz,lidy+FDOH,lidx)=vx(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lvx2(lidz,lidy-lsizey+3*FDOH,lidx)=vx(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lvx2(lidz,lidy,lidx-FDOH)=vx(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvx2(lidz,lidy,lidx+lsizex-3*FDOH)=vx(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvx2(lidz,lidy,lidx+FDOH)=vx(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvx2(lidz,lidy,lidx-lsizex+3*FDOH)=vx(gidz,gidy,gidx-lsizex+3*FDOH);
        __syncthreads();
#endif
        
#if   FDOH == 1
        vx_x2=mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidy,lidx)), __h22f2(lvx2(lidz,lidy,lidx-1))));
#elif FDOH == 2
        vx_x2=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidy,lidx)), __h22f2(lvx2(lidz,lidy,lidx-1)))),
                   mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidy,lidx+1)), __h22f2(lvx2(lidz,lidy,lidx-2)))));
#elif FDOH == 3
        vx_x2=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidy,lidx)), __h22f2(lvx2(lidz,lidy,lidx-1)))),
                        mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidy,lidx+1)), __h22f2(lvx2(lidz,lidy,lidx-2))))),
                   mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidy,lidx+2)), __h22f2(lvx2(lidz,lidy,lidx-3)))));
#elif FDOH == 4
        vx_x2=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidy,lidx)), __h22f2(lvx2(lidz,lidy,lidx-1)))),
                             mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidy,lidx+1)), __h22f2(lvx2(lidz,lidy,lidx-2))))),
                        mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidy,lidx+2)), __h22f2(lvx2(lidz,lidy,lidx-3))))),
                   mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidy,lidx+3)), __h22f2(lvx2(lidz,lidy,lidx-4)))));
#elif FDOH == 5
        vx_x2=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidy,lidx)), __h22f2(lvx2(lidz,lidy,lidx-1)))),
                                  mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidy,lidx+1)), __h22f2(lvx2(lidz,lidy,lidx-2))))),
                             mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidy,lidx+2)), __h22f2(lvx2(lidz,lidy,lidx-3))))),
                        mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidy,lidx+3)), __h22f2(lvx2(lidz,lidy,lidx-4))))),
                   mul2( f2h2(HC5), sub2(__h22f2(lvx2(lidz,lidy,lidx+4)), __h22f2(lvx2(lidz,lidy,lidx-5)))));
#elif FDOH == 6
        vx_x2=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidy,lidx)), __h22f2(lvx2(lidz,lidy,lidx-1)))),
                                       mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidy,lidx+1)), __h22f2(lvx2(lidz,lidy,lidx-2))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidy,lidx+2)), __h22f2(lvx2(lidz,lidy,lidx-3))))),
                             mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidy,lidx+3)), __h22f2(lvx2(lidz,lidy,lidx-4))))),
                        mul2( f2h2(HC5), sub2(__h22f2(lvx2(lidz,lidy,lidx+4)), __h22f2(lvx2(lidz,lidy,lidx-5))))),
                   mul2( f2h2(HC6), sub2(__h22f2(lvx2(lidz,lidy,lidx+5)), __h22f2(lvx2(lidz,lidy,lidx-6)))));
#endif
        
#if   FDOH == 1
        vx_y1=mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidy+1,lidx)), __h22f2(lvx2(lidz,lidy,lidx))));
#elif FDOH == 2
        vx_y1=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidy+1,lidx)), __h22f2(lvx2(lidz,lidy,lidx)))),
                   mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidy+2,lidx)), __h22f2(lvx2(lidz,lidy-1,lidx)))));
#elif FDOH == 3
        vx_y1=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidy+1,lidx)), __h22f2(lvx2(lidz,lidy,lidx)))),
                        mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidy+2,lidx)), __h22f2(lvx2(lidz,lidy-1,lidx))))),
                   mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidy+3,lidx)), __h22f2(lvx2(lidz,lidy-2,lidx)))));
#elif FDOH == 4
        vx_y1=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidy+1,lidx)), __h22f2(lvx2(lidz,lidy,lidx)))),
                             mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidy+2,lidx)), __h22f2(lvx2(lidz,lidy-1,lidx))))),
                        mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidy+3,lidx)), __h22f2(lvx2(lidz,lidy-2,lidx))))),
                   mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidy+4,lidx)), __h22f2(lvx2(lidz,lidy-3,lidx)))));
#elif FDOH == 5
        vx_y1=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidy+1,lidx)), __h22f2(lvx2(lidz,lidy,lidx)))),
                                  mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidy+2,lidx)), __h22f2(lvx2(lidz,lidy-1,lidx))))),
                             mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidy+3,lidx)), __h22f2(lvx2(lidz,lidy-2,lidx))))),
                        mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidy+4,lidx)), __h22f2(lvx2(lidz,lidy-3,lidx))))),
                   mul2( f2h2(HC5), sub2(__h22f2(lvx2(lidz,lidy+5,lidx)), __h22f2(lvx2(lidz,lidy-4,lidx)))));
#elif FDOH == 6
        vx_y1=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidy+1,lidx)), __h22f2(lvx2(lidz,lidy,lidx)))),
                                       mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidy+2,lidx)), __h22f2(lvx2(lidz,lidy-1,lidx))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidy+3,lidx)), __h22f2(lvx2(lidz,lidy-2,lidx))))),
                             mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidy+4,lidx)), __h22f2(lvx2(lidz,lidy-3,lidx))))),
                        mul2( f2h2(HC5), sub2(__h22f2(lvx2(lidz,lidy+5,lidx)), __h22f2(lvx2(lidz,lidy-4,lidx))))),
                   mul2( f2h2(HC6), sub2(__h22f2(lvx2(lidz,lidy+6,lidx)), __h22f2(lvx2(lidz,lidy-5,lidx)))));
#endif
        
#if   FDOH == 1
        vx_z1=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz,lidy,lidx)))));
#elif FDOH == 2
        vx_z1=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz,lidy,lidx))))),
                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidy,lidx))))));
#elif FDOH == 3
        vx_z1=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz,lidy,lidx))))),
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidy,lidx)))))),
                   mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidy,lidx))))));
#elif FDOH == 4
        vx_z1=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz,lidy,lidx))))),
                             mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidy,lidx)))))),
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidy,lidx)))))),
                   mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvx(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-3,lidy,lidx))))));
#elif FDOH == 5
        vx_z1=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz,lidy,lidx))))),
                                  mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidy,lidx)))))),
                             mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidy,lidx)))))),
                        mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvx(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-3,lidy,lidx)))))),
                   mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvx(2*lidz+5,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-4,lidy,lidx))))));
#elif FDOH == 6
        vx_z1=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz,lidy,lidx))))),
                                       mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidy,lidx)))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidy,lidx)))))),
                             mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvx(2*lidz+4,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-3,lidy,lidx)))))),
                        mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvx(2*lidz+5,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-4,lidy,lidx)))))),
                   mul2( f2h2(HC6), sub2(__h22f2(__hp(&lvx(2*lidz+6,lidy,lidx))), __h22f2(__hp(&lvx(2*lidz-5,lidy,lidx))))));
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
    
    // Update the variables
    lsxy=add2(lsxy,mul2(lmuipjp,add2(vx_y1,vy_x1)));
    lsyz=add2(lsyz,mul2(lmujpkp,add2(vy_z1,vz_y1)));
    lsxz=add2(lsxz,mul2(lmuipkp,add2(vx_z1,vz_x1)));
    lsxx=sub2(add2(lsxx,mul2(lM,add2(add2(vx_x2,vy_y2),vz_z2))),mul2(mul2(f2h2(2.0),lmu),add2(vy_y2,vz_z2)));
    lsyy=sub2(add2(lsyy,mul2(lM,add2(add2(vx_x2,vy_y2),vz_z2))),mul2(mul2(f2h2(2.0),lmu),add2(vx_x2,vz_z2)));
    lszz=sub2(add2(lszz,mul2(lM,add2(add2(vx_x2,vy_y2),vz_z2))),mul2(mul2(f2h2(2.0),lmu),add2(vx_x2,vy_y2)));
    //Write updated values to global memory
    sxx(gidz,gidy,gidx) = __f22h2(lsxx);
    sxy(gidz,gidy,gidx) = __f22h2(lsxy);
    sxz(gidz,gidy,gidx) = __f22h2(lsxz);
    syy(gidz,gidy,gidx) = __f22h2(lsyy);
    syz(gidz,gidy,gidx) = __f22h2(lsyz);
    szz(gidz,gidy,gidx) = __f22h2(lszz);
    
    
}
