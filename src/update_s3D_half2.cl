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

/*Update of the stresses in 3D*/



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
