/*Macros for FD difference stencils up to order 12 on GPU in 2D
 
 The macros assume the following variables are defined in the kernel:
 -lvar: A float array in local memory
 -gidx, gidz: The grid indices in the global memory
 -lidx, lidz: The grid indices in the local memory
 -FDOH: Half with of the final difference stencil
 -NZ: Global grid size in Z
 -lsizex: Local grid size in x
 -lsizez: Local grid size in z
 -LOCAL_OFF: If 0, uses local memory grid, else uses global memory grid
 -FP16: Type of FP16 computation: 1: use float2 for everything
                                  2: Read half2, compute in float2, write half2
                                  3: Read half2, compute half2, write half2
 */

#if LOCAL_OFF==0
    #define ind1(z,x)   (x)*2*lsizez+(z)
    #define ind2(z,x)  (x)*lsizez+(z)
    #define indg(z,x)  (x)*NZ+(z)
#else
    #define lidx gidx
    #define lidz gidz
    #define ind2(z,x)   (x)*(NZ)+(z)
#endif

/*Define functions and macros to be able to change operations types only with
 preprossor directives, that is with different values of FP16. Those functions
 are basic arithmetic operations and conversion between half2 and float2.*/

// precision of variables (__prec) and parameters (__pprec)
#if FP16==1
    #define __prec float
    #define __prec2 float2
    #define __pprec float2
#else
    #define __prec half
    #define __prec2 half2
    #define __pprec half2
#endif

// conversion functions from reading/writing and computations
#if FP16==2
    #define __h2f(x) __half2float((x))
    #define __h22f2(x) __half22float2((x))
    #define __f22h2(x) __float22half2_rn((x))
    #define __pconv(x) __half22float2((x))
#else
    #define __h2f(x) (x)
    #define __h22f2(x) (x)
    #define __f22h2(x) (x)
    #define __pconv(x) (x)
#endif

// functions to compute with half2 or float2 (only FP16=3 computes with half2)
#if FP16==3
    #define __cprec half2
    #define __f22h2c(x) __float22half2_rn((x))
    #define __h22f2c(x) __half22float2((x))
    #define add2 __hadd2
    #define mul2 __hmul2
    #define div2 __h2div
    #define sub2 __hsub2
    #define f2h2 __float2half2_rn
#else
    #define __cprec float2
    #define __f22h2c(x) (x)
    #define __h22f2c(x) (x)
    #define add2 add2f
    #define mul2 mul2f
    #define div2 div2f
    #define sub2 sub2f
    #define f2h2 f2h2f
#endif

// functions implementation for float2 operations
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

//Load in local memory with the halo for FD in different directions
#define load_local_x(v) \
do {\
        lvar2[ind2(lidz,lidx)]=v[indg(gidz, gidx)];\
        if (lidx<2*FDOH)\
            lvar2[ind2(lidz,lidx-FDOH)]=v[indg(gidz,gidx-FDOH)];\
        if (lidx+lsizex-3*FDOH<FDOH)\
            lvar2[ind2(lidz,lidx+lsizex-3*FDOH)]=v[indg(gidz,gidx+lsizex-3*FDOH)];\
        if (lidx>(lsizex-2*FDOH-1))\
            lvar2[ind2(lidz,lidx+FDOH)]=v[indg(gidz,gidx+FDOH)];\
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))\
            lvar2[ind2(lidz,lidx-lsizex+3*FDOH)]=v[indg(gidz,gidx-lsizex+3*FDOH)];\
} while(0)


#define load_local_z(v) \
do {\
        lvar2[ind2(lidz,lidx)]=v[indg(gidz,gidx)];\
        if (lidz<FDOH)\
            lvar2[ind2(lidz-FDOH/2,lidx)]=v[indg(gidz-FDOH/2,gidx)];\
        if (lidz>(lsizez-FDOH-1))\
            lvar2[ind2(lidz+FDOH/2,lidx)]=v[indg(gidz+FDOH/2,gidx)];\
} while(0)

#define load_local_xz(v) \
do {\
        lvar2[ind2(lidz,lidx)]=v[indg(gidz,gidx)];\
        if (lidz<FDOH)\
            lvar2[ind2(lidz-FDOH/2,lidx)]=v[indg(gidz-FDOH/2,gidx)];\
        if (lidz>(lsizez-FDOH-1))\
            lvar2[ind2(lidz+FDOH/2,lidx)]=v[indg(gidz+FDOH/2,gidx)];\
        if (lidx<2*FDOH)\
            lvar2[ind2(lidz,lidx-FDOH)]=v[indg(gidz,gidx-FDOH)];\
        if (lidx+lsizex-3*FDOH<FDOH)\
            lvar2[ind2(lidz,lidx+lsizex-3*FDOH)]=v[indg(gidz,gidx+lsizex-3*FDOH)];\
        if (lidx>(lsizex-2*FDOH-1))\
            lvar2[ind2(lidz,lidx+FDOH)]=v[indg(gidz,gidx+FDOH)];\
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))\
            lvar2[ind2(lidz,lidx-lsizex+3*FDOH)]=v[indg(gidz,gidx-lsizex+3*FDOH)];\
} while(0)


//Forward stencil in x
#if   FDOH == 1
        #define Dxp(v)  mul2( f2h2(HC1), sub2(__h22f2(v[ind2(lidz,lidx+1)]), __h22f2(v[ind2(lidz,lidx)])))
#elif FDOH == 2
        #define Dxp(v) (\
                        mul2( f2h2(HC1), sub2(__h22f2(v[ind2(lidz,lidx+1)]), __h22f2(v[ind2(lidz,lidx)]))),\
                        mul2( f2h2(HC2), sub2(__h22f2(v[ind2(lidz,lidx+2)]), __h22f2(v[ind2(lidz,lidx-1)]))));
#elif FDOH == 3
        #define Dxp(v) add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(v[ind2(lidz,lidx+1)]), __h22f2(v[ind2(lidz,lidx)]))),\
                        mul2( f2h2(HC2), sub2(__h22f2(v[ind2(lidz,lidx+2)]), __h22f2(v[ind2(lidz,lidx-1)])))),\
                        mul2( f2h2(HC3), sub2(__h22f2(v[ind2(lidz,lidx+3)]), __h22f2(v[ind2(lidz,lidx-2)]))));
#elif FDOH == 4
        #define Dxp(v) add2(add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(v[ind2(lidz,lidx+1)]), __h22f2(v[ind2(lidz,lidx)]))),\
                        mul2( f2h2(HC2), sub2(__h22f2(v[ind2(lidz,lidx+2)]), __h22f2(v[ind2(lidz,lidx-1)])))),\
                        mul2( f2h2(HC3), sub2(__h22f2(v[ind2(lidz,lidx+3)]), __h22f2(v[ind2(lidz,lidx-2)])))),\
                        mul2( f2h2(HC4), sub2(__h22f2(v[ind2(lidz,lidx+4)]), __h22f2(v[ind2(lidz,lidx-3)]))));
#elif FDOH == 5
        #define Dxp(v) dd2(add2(add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(v[ind2(lidz,lidx+1)]), __h22f2(v[ind2(lidz,lidx)]))),\
                        mul2( f2h2(HC2), sub2(__h22f2(v[ind2(lidz,lidx+2)]), __h22f2(v[ind2(lidz,lidx-1)])))),\
                        mul2( f2h2(HC3), sub2(__h22f2(v[ind2(lidz,lidx+3)]), __h22f2(v[ind2(lidz,lidx-2)])))),\
                        mul2( f2h2(HC4), sub2(__h22f2(v[ind2(lidz,lidx+4)]), __h22f2(v[ind2(lidz,lidx-3)])))),\
                        mul2( f2h2(HC5), sub2(__h22f2(v[ind2(lidz,lidx+5)]), __h22f2(v[ind2(lidz,lidx-4)]))));
#elif FDOH == 6
        #define Dxp(v) add2(add2(add2(add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(v[ind2(lidz,lidx+1)]), __h22f2(v[ind2(lidz,lidx)]))),\
                        mul2( f2h2(HC2), sub2(__h22f2(v[ind2(lidz,lidx+2)]), __h22f2(v[ind2(lidz,lidx-1)])))),\
                        mul2( f2h2(HC3), sub2(__h22f2(v[ind2(lidz,lidx+3)]), __h22f2(v[ind2(lidz,lidx-2)])))),\
                        mul2( f2h2(HC4), sub2(__h22f2(v[ind2(lidz,lidx+4)]), __h22f2(v[ind2(lidz,lidx-3)])))),\
                        mul2( f2h2(HC5), sub2(__h22f2(v[ind2(lidz,lidx+5)]), __h22f2(v[ind2(lidz,lidx-4)])))),\
                        mul2( f2h2(HC6), sub2(__h22f2(v[ind2(lidz,lidx+6)]), __h22f2(v[ind2(lidz,lidx-5)]))));
#endif


//Forward stencil in x
#if   FDOH == 1
    #define Dzp(v)      mul2( f2h2(HC1), sub2(__h22f2(__hp(&v[ind1(2*lidz+1,lidx)])), __h22f2(__hp(&v[ind1(2*lidz,lidx)]))));
#elif FDOH == 2
    #define Dzp(v) add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&v[ind1(2*lidz+1,lidx)])), __h22f2(__hp(&v[ind1(2*lidz,lidx)])))),\
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&v[ind1(2*lidz+2,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-1,lidx)])))));
#elif FDOH == 3
    #define Dzp(v) add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&v[ind1(2*lidz+1,lidx)])), __h22f2(__hp(&v[ind1(2*lidz,lidx)])))),\
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&v[ind1(2*lidz+2,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-1,lidx)]))))),\
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&v[ind1(2*lidz+3,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-2,lidx)])))));
#elif FDOH == 4
    #define Dzp(v) add2(add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&v[ind1(2*lidz+1,lidx)])), __h22f2(__hp(&v[ind1(2*lidz,lidx)])))),\
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&v[ind1(2*lidz+2,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-1,lidx)]))))),\
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&v[ind1(2*lidz+3,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-2,lidx)]))))),\
                        mul2( f2h2(HC4), sub2(__h22f2(__hp(&v[ind1(2*lidz+4,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-3,lidx)])))));
#elif FDOH == 5
    #define Dzp(v) add2(add2(add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&v[ind1(2*lidz+1,lidx)])), __h22f2(__hp(&v[ind1(2*lidz,lidx)])))),\
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&v[ind1(2*lidz+2,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-1,lidx)]))))),\
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&v[ind1(2*lidz+3,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-2,lidx)]))))),\
                        mul2( f2h2(HC4), sub2(__h22f2(__hp(&v[ind1(2*lidz+4,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-3,lidx)]))))),\
                        mul2( f2h2(HC5), sub2(__h22f2(__hp(&v[ind1(2*lidz+5,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-4,lidx)])))));
#elif FDOH == 6
    #define Dzp(v) add2(add2(add2(add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&v[ind1(2*lidz+1,lidx)])), __h22f2(__hp(&v[ind1(2*lidz,lidx)])))),\
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&v[ind1(2*lidz+2,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-1,lidx)]))))),\
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&v[ind1(2*lidz+3,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-2,lidx)]))))),\
                        mul2( f2h2(HC4), sub2(__h22f2(__hp(&v[ind1(2*lidz+4,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-3,lidx)]))))),\
                        mul2( f2h2(HC5), sub2(__h22f2(__hp(&v[ind1(2*lidz+5,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-4,lidx)]))))),\
                        mul2( f2h2(HC6), sub2(__h22f2(__hp(&v[ind1(2*lidz+6,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-5,lidx)])))));
#endif

//Backward stencil in x
#if   FDOH == 1
    #define Dxm(v)      mul2( f2h2(HC1), sub2(__h22f2(v[ind2(lidz,lidx)]), __h22f2(v[ind2(lidz,lidx-1)])));
#elif FDOH == 2
    #define Dxm(v) add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(v[ind2(lidz,lidx)]), __h22f2(v[ind2(lidz,lidx-1)]))),\
                        mul2( f2h2(HC2), sub2(__h22f2(v[ind2(lidz,lidx+1)]), __h22f2(v[ind2(lidz,lidx-2)]))));
#elif FDOH == 3
    #define Dxm(v) add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(v[ind2(lidz,lidx)]), __h22f2(v[ind2(lidz,lidx-1)]))),\
                        mul2( f2h2(HC2), sub2(__h22f2(v[ind2(lidz,lidx+1)]), __h22f2(v[ind2(lidz,lidx-2)])))),\
                        mul2( f2h2(HC3), sub2(__h22f2(v[ind2(lidz,lidx+2)]), __h22f2(v[ind2(lidz,lidx-3)]))));
#elif FDOH == 4
    #define Dxm(v) add2(add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(v[ind2(lidz,lidx)]), __h22f2(v[ind2(lidz,lidx-1)]))),\
                        mul2( f2h2(HC2), sub2(__h22f2(v[ind2(lidz,lidx+1)]), __h22f2(v[ind2(lidz,lidx-2)])))),\
                        mul2( f2h2(HC3), sub2(__h22f2(v[ind2(lidz,lidx+2)]), __h22f2(v[ind2(lidz,lidx-3)])))),\
                        mul2( f2h2(HC4), sub2(__h22f2(v[ind2(lidz,lidx+3)]), __h22f2(v[ind2(lidz,lidx-4)]))));
#elif FDOH == 5
    #define Dxm(v) add2(add2(add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(v[ind2(lidz,lidx)]), __h22f2(v[ind2(lidz,lidx-1)]))),\
                        mul2( f2h2(HC2), sub2(__h22f2(v[ind2(lidz,lidx+1)]), __h22f2(v[ind2(lidz,lidx-2)])))),\
                        mul2( f2h2(HC3), sub2(__h22f2(v[ind2(lidz,lidx+2)]), __h22f2(v[ind2(lidz,lidx-3)])))),\
                        mul2( f2h2(HC4), sub2(__h22f2(v[ind2(lidz,lidx+3)]), __h22f2(v[ind2(lidz,lidx-4)])))),\
                        mul2( f2h2(HC5), sub2(__h22f2(v[ind2(lidz,lidx+4)]), __h22f2(v[ind2(lidz,lidx-5)]))));
#elif FDOH == 6
    #define Dxm(v) add2(add2(add2(add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(v[ind2(lidz,lidx)]), __h22f2(v[ind2(lidz,lidx-1)]))),\
                        mul2( f2h2(HC2), sub2(__h22f2(v[ind2(lidz,lidx+1)]), __h22f2(v[ind2(lidz,lidx-2)])))),\
                        mul2( f2h2(HC3), sub2(__h22f2(v[ind2(lidz,lidx+2)]), __h22f2(v[ind2(lidz,lidx-3)])))),\
                        mul2( f2h2(HC4), sub2(__h22f2(v[ind2(lidz,lidx+3)]), __h22f2(v[ind2(lidz,lidx-4)])))),\
                        mul2( f2h2(HC5), sub2(__h22f2(v[ind2(lidz,lidx+4)]), __h22f2(v[ind2(lidz,lidx-5)])))),\
                        mul2( f2h2(HC6), sub2(__h22f2(v[ind2(lidz,lidx+5)]), __h22f2(v[ind2(lidz,lidx-6)]))));
#endif

//Backward stencil in z
#if   FDOH == 1
    #define Dzm(v) mul2( f2h2(HC1), sub2(__h22f2(__hp(&v[ind1(2*lidz,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-1,lidx)]))));
#elif FDOH == 2
    #define Dzm(v) add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&v[ind1(2*lidz,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-1,lidx)])))),\
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&v[ind1(2*lidz+1,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-2,lidx)])))));\
#elif FDOH == 3
    #define Dzm(v) add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&v[ind1(2*lidz,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-1,lidx)])))),\
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&v[ind1(2*lidz+1,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-2,lidx)]))))),\
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&v[ind1(2*lidz+2,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-3,lidx)])))));
#elif FDOH == 4
    #define Dzm(v) add2(add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&v[ind1(2*lidz,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-1,lidx)])))),\
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&v[ind1(2*lidz+1,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-2,lidx)]))))),\
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&v[ind1(2*lidz+2,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-3,lidx)]))))),\
                        mul2( f2h2(HC4), sub2(__h22f2(__hp(&v[ind1(2*lidz+3,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-4,lidx)])))));
#elif FDOH == 5
    #define Dzm(v) add2(add2(add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&v[ind1(2*lidz,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-1,lidx)])))),\
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&v[ind1(2*lidz+1,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-2,lidx)]))))),\
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&v[ind1(2*lidz+2,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-3,lidx)]))))),\
                        mul2( f2h2(HC4), sub2(__h22f2(__hp(&v[ind1(2*lidz+3,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-4,lidx)]))))),\
                        mul2( f2h2(HC5), sub2(__h22f2(__hp(&v[ind1(2*lidz+4,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-5,lidx)])))));
#elif FDOH == 6
    #define Dzm(v) add2(add2(add2(add2(add2(\
                        mul2( f2h2(HC1), sub2(__h22f2(__hp(&v[ind1(2*lidz,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-1,lidx)])))),\
                        mul2( f2h2(HC2), sub2(__h22f2(__hp(&v[ind1(2*lidz+1,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-2,lidx)]))))),\
                        mul2( f2h2(HC3), sub2(__h22f2(__hp(&v[ind1(2*lidz+2,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-3,lidx)]))))),\
                        mul2( f2h2(HC4), sub2(__h22f2(__hp(&v[ind1(2*lidz+3,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-4,lidx)]))))),\
                        mul2( f2h2(HC5), sub2(__h22f2(__hp(&v[ind1(2*lidz+4,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-5,lidx)]))))),\
                        mul2( f2h2(HC6), sub2(__h22f2(__hp(&v[ind1(2*lidz+5,lidx)])), __h22f2(__hp(&v[ind1(2*lidz-6,lidx)])))));
#endif

