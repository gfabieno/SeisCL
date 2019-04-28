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

/*Update of the stresses in 2D SV. The variable FP16 is
 used to control how FP16 is used: 1: FP32, 2: FP16 IO only, 3: FP16 IO and
 arithmetics*/


//Define useful macros to be able to write a matrix formulation in 2D with OpenCl
#define rip(z,x) rip[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define rkp(z,x) rkp[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define muipkp(z,x) muipkp[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define M(z,x) M[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]
#define mu(z,x) mu[((x)-FDOH)*(NZ-FDOH)+((z)-FDOH/2)]

#define sxx(z,x) sxx[(x)*NZ+(z)]
#define sxz(z,x) sxz[(x)*NZ+(z)]
#define szz(z,x) szz[(x)*NZ+(z)]
#define vx(z,x) vx[(x)*NZ+(z)]
#define vz(z,x) vz[(x)*NZ+(z)]

#if LOCAL_OFF==0
#define lvar(z,x) lvar[(x)*2*lsizez+(z)]
#define lvar2(z,x) lvar2[(x)*lsizez+(z)]
#endif

/*Define functions and macros to be able to change operations types only with
 preprossor directives, that is with different values of FP16. Those functions
 are basic arithmetic operations and conversion between half2 and float2.*/
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



extern "C" __global__ void update_s(int offcomm,
                                    __pprec *muipkp, __pprec *M, __pprec *mu,
                                    __prec2 *sxx,__prec2 *sxz,__prec2 *szz,
                                    __prec2 *vx,__prec2 *vz, float *taper
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
    

    
    //Define private derivatives
    __cprec vx_x2;
    __cprec vx_z1;
    __cprec vz_x1;
    __cprec vz_z2;

    //Define and load private parameters and variables
    __cprec lsxx = __h22f2(sxx(gidz,gidx));
    __cprec lsxz = __h22f2(sxz(gidz,gidx));
    __cprec lszz = __h22f2(szz(gidz,gidx));
    __cprec lM = __pconv(M(gidz,gidx));
    __cprec lmu = __pconv(mu(gidz,gidx));
    __cprec lmuipkp = __pconv(muipkp(gidz,gidx));
    
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
    
    //Calculation of the spatial derivatives
    {
#if LOCAL_OFF==0
        __syncthreads();
        lvx2(lidz,lidx)=vx(gidz,gidx);
        if (lidz<FDOH)
            lvx2(lidz-FDOH/2,lidx)=vx(gidz-FDOH/2,gidx);
        if (lidz>(lsizez-FDOH-1))
            lvx2(lidz+FDOH/2,lidx)=vx(gidz+FDOH/2,gidx);
        if (lidx<2*FDOH)
            lvx2(lidz,lidx-FDOH)=vx(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvx2(lidz,lidx+lsizex-3*FDOH)=vx(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvx2(lidz,lidx+FDOH)=vx(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvx2(lidz,lidx-lsizex+3*FDOH)=vx(gidz,gidx-lsizex+3*FDOH);
        __syncthreads();
#endif

#if   FDOH == 1
        vx_x2=mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1))));
#elif FDOH == 2
        vx_x2=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
                   mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2)))));
#elif FDOH == 3
        vx_x2=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
                        mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2))))),
                        mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidx+2)), __h22f2(lvx2(lidz,lidx-3)))));
#elif FDOH == 4
        vx_x2=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
                             mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2))))),
                             mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidx+2)), __h22f2(lvx2(lidz,lidx-3))))),
                             mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidx+3)), __h22f2(lvx2(lidz,lidx-4)))));
#elif FDOH == 5
        vx_x2=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
                                  mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidx+2)), __h22f2(lvx2(lidz,lidx-3))))),
                                  mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidx+3)), __h22f2(lvx2(lidz,lidx-4))))),
                                  mul2( f2h2(HC5), sub2(__h22f2(lvx2(lidz,lidx+4)), __h22f2(lvx2(lidz,lidx-5)))));
#elif FDOH == 6
        vx_x2=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(lvx2(lidz,lidx)), __h22f2(lvx2(lidz,lidx-1)))),
                                       mul2( f2h2(HC2), sub2(__h22f2(lvx2(lidz,lidx+1)), __h22f2(lvx2(lidz,lidx-2))))),
                                       mul2( f2h2(HC3), sub2(__h22f2(lvx2(lidz,lidx+2)), __h22f2(lvx2(lidz,lidx-3))))),
                                       mul2( f2h2(HC4), sub2(__h22f2(lvx2(lidz,lidx+3)), __h22f2(lvx2(lidz,lidx-4))))),
                                       mul2( f2h2(HC5), sub2(__h22f2(lvx2(lidz,lidx+4)), __h22f2(lvx2(lidz,lidx-5))))),
                                       mul2( f2h2(HC6), sub2(__h22f2(lvx2(lidz,lidx+5)), __h22f2(lvx2(lidz,lidx-6)))));
#endif

#if   FDOH == 1
        vx_z1=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx)))));
#elif FDOH == 2
        vx_z1=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx))))));
#elif FDOH == 3
        vx_z1=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx)))))),
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidx))))));
#elif FDOH == 4
        vx_z1=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
                             mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx)))))),
                             mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidx)))))),
                             mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvx(2*lidz+4,lidx))), __h22f2(__hp(&lvx(2*lidz-3,lidx))))));
#elif FDOH == 5
        vx_z1=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
                                  mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx)))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidx)))))),
                                  mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvx(2*lidz+4,lidx))), __h22f2(__hp(&lvx(2*lidz-3,lidx)))))),
                                  mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvx(2*lidz+5,lidx))), __h22f2(__hp(&lvx(2*lidz-4,lidx))))));
#elif FDOH == 6
        vx_z1=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvx(2*lidz+1,lidx))), __h22f2(__hp(&lvx(2*lidz,lidx))))),
                                       mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvx(2*lidz+2,lidx))), __h22f2(__hp(&lvx(2*lidz-1,lidx)))))),
                                       mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvx(2*lidz+3,lidx))), __h22f2(__hp(&lvx(2*lidz-2,lidx)))))),
                                       mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvx(2*lidz+4,lidx))), __h22f2(__hp(&lvx(2*lidz-3,lidx)))))),
                                       mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvx(2*lidz+5,lidx))), __h22f2(__hp(&lvx(2*lidz-4,lidx)))))),
                                       mul2( f2h2(HC6), sub2(__h22f2(__hp(&lvx(2*lidz+6,lidx))), __h22f2(__hp(&lvx(2*lidz-5,lidx))))));
#endif

#if LOCAL_OFF==0
        __syncthreads();
        lvz2(lidz,lidx)=vz(gidz,gidx);
        if (lidz<FDOH)
            lvz2(lidz-FDOH/2,lidx)=vz(gidz-FDOH/2,gidx);
        if (lidz>(lsizez-FDOH-1))
            lvz2(lidz+FDOH/2,lidx)=vz(gidz+FDOH/2,gidx);
        if (lidx<2*FDOH)
            lvz2(lidz,lidx-FDOH)=vz(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvz2(lidz,lidx+lsizex-3*FDOH)=vz(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvz2(lidz,lidx+FDOH)=vz(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvz2(lidz,lidx-lsizex+3*FDOH)=vz(gidz,gidx-lsizex+3*FDOH);
        __syncthreads();
#endif

#if   FDOH == 1
        vz_x1=mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx))));
#elif FDOH == 2
        vz_x1=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
                   mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1)))));
#elif FDOH == 3
        vz_x1=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
                        mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1))))),
                        mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidx+3)), __h22f2(lvz2(lidz,lidx-2)))));
#elif FDOH == 4
        vz_x1=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
                             mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1))))),
                             mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidx+3)), __h22f2(lvz2(lidz,lidx-2))))),
                             mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidx+4)), __h22f2(lvz2(lidz,lidx-3)))));
#elif FDOH == 5
        vz_x1=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
                                  mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidx+3)), __h22f2(lvz2(lidz,lidx-2))))),
                                  mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidx+4)), __h22f2(lvz2(lidz,lidx-3))))),
                                  mul2( f2h2(HC5), sub2(__h22f2(lvz2(lidz,lidx+5)), __h22f2(lvz2(lidz,lidx-4)))));
#elif FDOH == 6
        vz_x1=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(lvz2(lidz,lidx+1)), __h22f2(lvz2(lidz,lidx)))),
                                       mul2( f2h2(HC2), sub2(__h22f2(lvz2(lidz,lidx+2)), __h22f2(lvz2(lidz,lidx-1))))),
                                       mul2( f2h2(HC3), sub2(__h22f2(lvz2(lidz,lidx+3)), __h22f2(lvz2(lidz,lidx-2))))),
                                       mul2( f2h2(HC4), sub2(__h22f2(lvz2(lidz,lidx+4)), __h22f2(lvz2(lidz,lidx-3))))),
                                       mul2( f2h2(HC5), sub2(__h22f2(lvz2(lidz,lidx+5)), __h22f2(lvz2(lidz,lidx-4))))),
                                       mul2( f2h2(HC6), sub2(__h22f2(lvz2(lidz,lidx+6)), __h22f2(lvz2(lidz,lidx-5)))));
#endif

#if   FDOH == 1
        vz_z2=mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx)))));
#elif FDOH == 2
        vz_z2=add2(
                   mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
                   mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx))))));
#elif FDOH == 3
        vz_z2=add2(add2(
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx)))))),
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidx))))));
#elif FDOH == 4
        vz_z2=add2(add2(add2(
                             mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
                             mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx)))))),
                             mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidx)))))),
                             mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvz(2*lidz+3,lidx))), __h22f2(__hp(&lvz(2*lidz-4,lidx))))));
#elif FDOH == 5
        vz_z2=add2(add2(add2(add2(
                                  mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
                                  mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx)))))),
                                  mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidx)))))),
                                  mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvz(2*lidz+3,lidx))), __h22f2(__hp(&lvz(2*lidz-4,lidx)))))),
                                  mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvz(2*lidz+4,lidx))), __h22f2(__hp(&lvz(2*lidz-5,lidx))))));
#elif FDOH == 6
        vz_z2=add2(add2(add2(add2(add2(
                                       mul2( f2h2(HC1), sub2(__h22f2(__hp(&lvz(2*lidz,lidx))), __h22f2(__hp(&lvz(2*lidz-1,lidx))))),
                                       mul2( f2h2(HC2), sub2(__h22f2(__hp(&lvz(2*lidz+1,lidx))), __h22f2(__hp(&lvz(2*lidz-2,lidx)))))),
                                       mul2( f2h2(HC3), sub2(__h22f2(__hp(&lvz(2*lidz+2,lidx))), __h22f2(__hp(&lvz(2*lidz-3,lidx)))))),
                                       mul2( f2h2(HC4), sub2(__h22f2(__hp(&lvz(2*lidz+3,lidx))), __h22f2(__hp(&lvz(2*lidz-4,lidx)))))),
                                       mul2( f2h2(HC5), sub2(__h22f2(__hp(&lvz(2*lidz+4,lidx))), __h22f2(__hp(&lvz(2*lidz-5,lidx)))))),
                                       mul2( f2h2(HC6), sub2(__h22f2(__hp(&lvz(2*lidz+5,lidx))), __h22f2(__hp(&lvz(2*lidz-6,lidx))))));
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
    lsxz=add2(lsxz,mul2(lmuipkp,add2(vx_z1,vz_x1)));
    lsxx=sub2(add2(lsxx,mul2(lM,add2(vx_x2,vz_z2))),mul2(mul2(f2h2(2.0),lmu),vz_z2));
    lszz=sub2(add2(lszz,mul2(lM,add2(vx_x2,vz_z2))),mul2(mul2(f2h2(2.0),lmu),vx_x2));
    
#if ABS_TYPE==2
    {
#if FREESURF==0
        if (2*gidz-FDOH<NAB){
            lsxx.x*=taper[2*gidz-FDOH];
            lsxx.y*=taper[2*gidz+1-FDOH];
            lszz.x*=taper[2*gidz-FDOH];
            lszz.y*=taper[2*gidz+1-FDOH];
            lsxz.x*=taper[2*gidz-FDOH];
            lsxz.y*=taper[2*gidz+1-FDOH];
        }
#endif
        if (2*gidz>2*NZ-NAB-FDOH-1){
            lsxx.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsxx.y*=taper[2*NZ-FDOH-2*gidz-1-1];
            lszz.x*=taper[2*NZ-FDOH-2*gidz-1];
            lszz.y*=taper[2*NZ-FDOH-2*gidz-1-1];
            lsxz.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsxz.y*=taper[2*NZ-FDOH-2*gidz-1-1];
        }
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            lsxx.x*=taper[gidx-FDOH];
            lsxx.y*=taper[gidx-FDOH];
            lszz.x*=taper[gidx-FDOH];
            lszz.y*=taper[gidx-FDOH];
            lsxz.x*=taper[gidx-FDOH];
            lsxz.y*=taper[gidx-FDOH];
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            lsxx.x*=taper[NX-FDOH-gidx-1];
            lsxx.y*=taper[NX-FDOH-gidx-1];
            lszz.x*=taper[NX-FDOH-gidx-1];
            lszz.y*=taper[NX-FDOH-gidx-1];
            lsxz.x*=taper[NX-FDOH-gidx-1];
            lsxz.y*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif

    //Write updated values to global memory
    sxx(gidz,gidx) = __f22h2(lsxx);
    sxz(gidz,gidx) = __f22h2(lsxz);
    szz(gidz,gidx) = __f22h2(lszz);

}

