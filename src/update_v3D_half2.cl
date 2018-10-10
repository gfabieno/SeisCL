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

/*Update of the velocity in 3D, The variable FP16 is
 used to control how FP16 is used: 1: FP32, 2: FP16 IO only, 3: FP16 IO and
 arithmetics*/



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

//Define useful macros to be able to write a matrix formulation in 2D with OpenCl
extern "C" __device__ float2 add2f(float2 a, float2 b ){
    
    float2 output;
    output.x = a.x+b.x;
    output.y = a.y+b.y;
    return output;
}
extern "C" __device__ float2 mul2f(float2 a, float2 b ){
    
    float2 output;
    output.x = a.x*b.x;
    output.y = a.y*b.y;
    return output;
}
extern "C" __device__ float2 div2f(float2 a, float2 b ){
    
    float2 output;
    output.x = a.x/b.x;
    output.y = a.y/b.y;
    return output;
}
extern "C" __device__ float2 sub2f(float2 a, float2 b ){
    
    float2 output;
    output.x = a.x-b.x;
    output.y = a.y-b.y;
    return output;
}
extern "C" __device__ float2 f2h2f(float a){
    
    float2 output={a,a};
    return output;
}


#if FP16==2

#define __h2f(x) __half2float((x))
#define __h22f2(x) __half22float2((x))
#define __f22h2(x) __float22half2_rn((x))

#else

#define __h2f(x) (x)
#define __h22f2(x) (x)
#define __f22h2(x) (x)

#endif

#if FP16==1

#define __prec float
#define __prec2 float2

#else

#define __prec half
#define __prec2 half2

#endif


#if FP16!=3

#define __cprec float2
#define __f22h2c(x) (x)
#define __h22f2c(x) (x)

#define add2 add2f
#define mul2 mul2f
#define div2 div2f
#define sub2 sub2f
#define f2h2 f2h2f

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

#if FP16>1

#define __pprec half2

#else

#define __pprec float2

#endif

#if FP16==2

#define __pconv(x) __half22float2((x))

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
        lszz2(lidz,lidy,lidx)=szz(gidz,gidy,gidx);
        if (lidz<FDOH)
            lszz2(lidz-FDOH/2,lidy,lidx)=szz(gidz-FDOH/2,gidy,gidx);
        if (lidz>(lsizez-FDOH-1))
            lszz2(lidz+FDOH/2,lidy,lidx)=szz(gidz+FDOH/2,gidy,gidx);
        __syncthreads();
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
        __syncthreads();
        lsxx2(lidz,lidy,lidx)=sxx(gidz,gidy,gidx);
        if (lidx<2*FDOH)
            lsxx2(lidz,lidy,lidx-FDOH)=sxx(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxx2(lidz,lidy,lidx+lsizex-3*FDOH)=sxx(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxx2(lidz,lidy,lidx+FDOH)=sxx(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxx2(lidz,lidy,lidx-lsizex+3*FDOH)=sxx(gidz,gidy,gidx-lsizex+3*FDOH);
        __syncthreads();
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
        __syncthreads();
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
        __syncthreads();
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
        __syncthreads();
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
        __syncthreads();
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
        __syncthreads();
        lsyy2(lidz,lidy,lidx)=syy(gidz,gidy,gidx);
        if (lidy<2*FDOH)
            lsyy2(lidz,lidy-FDOH,lidx)=syy(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lsyy2(lidz,lidy+lsizey-3*FDOH,lidx)=syy(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lsyy2(lidz,lidy+FDOH,lidx)=syy(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lsyy2(lidz,lidy-lsizey+3*FDOH,lidx)=syy(gidz,gidy-lsizey+3*FDOH,gidx);
        __syncthreads();
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
        __syncthreads();
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
        __syncthreads();
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



