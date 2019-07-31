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
 */

#if LOCAL_OFF==0
    #define lvar(z,x) lvar[(x)*2*lsizez+(z)]
    #define lvar2(z,x) lvar2[(x)*lsizez+(z)]
    #define ind(z,x)   (x)*2*lsizez+(z)
    #define ind2(z,x)  (x)*lsizez+(z)
    #define indg(z,x)  (x)*NZ+(z)
#else
    #define lvar(z,x)  lvar[(x)*(NZ)+(z)]
    #define lidx gidx
    #define lidz gidz
    #define ind(z,x)   (x)*(NZ)+(z)
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

#define load_local_x(v) ({\
        lvar2[ind(lidz,lidx)]=v[indg(gidz, gidx)];\
        if (lidx<2*FDOH)\
            lvar2[ind(lidz,lidx-FDOH)]=v[indg(gidz,gidx-FDOH)];\
        if (lidx+lsizex-3*FDOH<FDOH)\
            lvar2[ind(lidz,lidx+lsizex-3*FDOH)]=v[indg(gidz,gidx+lsizex-3*FDOH)];\
        if (lidx>(lsizex-2*FDOH-1))\
            lvar2[ind(lidz,lidx+FDOH)]=v[indg(gidz,gidx+FDOH)];\
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))\
            lvar2[ind(lidz,lidx-lsizex+3*FDOH)]=v[indg(gidz,gidx-lsizex+3*FDOH)];\
        })


