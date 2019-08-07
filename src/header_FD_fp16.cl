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


/*Define functions and macros to be able to change operations types only with
 preprossor directives, that is with different values of FP16. Those functions
 are basic arithmetic operations and conversion between half2 and float2.*/

// DIV allow to change from vector (type2) type to scalar type
#if FP16==0
    #define DIV 1
    #define __gprec float
#else
    #define DIV 2
    #define __gprec float2
#endif

// precision of variables (__prec) and parameters (__pprec) in global memory
#if FP16==0
    #define __prec float
    #define __prec2 float
    #define __pprec float
#elif FP16==1
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
    #define __h22f2(x) __half22float2((x))
    #define __f22h2(x) __float22half2_rn((x))
    #define __pconv(x) __half22float2((x))
#else
    #define __h22f2(x) (x)
    #define __f22h2(x) (x)
    #define __pconv(x) (x)
#endif

// functions to compute with half2 or float2 (only FP16=3 computes with half2)
#if FP16==3
    #define __cprec half2
    #define __cprec0 {0.0f,0.0f}
#elif FP16==0
    #define __cprec float
    #define __cprec0 0.0f
#else
    #define __cprec float2
    #define __cprec0 {0.0f,0.0f}
#endif

#if FP16==3
    #define __f22h2c(x) __float22half2_rn((x))
    #define __h22f2c(x) __half22float2((x))
#else
    #define __f22h2c(x) (x)
    #define __h22f2c(x) (x)
#endif

// functions to scale parameters

#ifdef __OPENCL_VERSION__
    #if FP16==0
    LFUNDEF float scalefun(float a, int scaler ){
        return ldexp(a, scaler);
    }
    #else
    LFUNDEF float2 scalefun(float2 a, int scaler ){
        
        float2 output;
        output.x  = ldexp(a.x, scaler);
        output.y  = ldexp(a.y, scaler);
        return output;
    }
    #endif
#else
    #if FP16==0
    LFUNDEF float scalefun(float a, int scaler ){
        return scalbnf(a, scaler);
    }
    #else
    LFUNDEF float2 scalefun(float2 a, int scaler ){
        
        float2 output;
        output.x  = scalbnf(a.x, scaler);
        output.y  = scalbnf(a.y, scaler);
        return output;
    }
    #endif
#endif

// functions to handle FD stencils on length 2 vectors
#if FP16==0
LFUNDEF float __hp(float *a ){
    return *a;
}
LFUNDEF float __hpi(float *a ){
    return *a;
}
#else
LFUNDEF __prec2 __hp(LOCID __prec *a ){
    
    __prec2 output;
    *((__prec *)&output) = *a;
    *((__prec *)&output+1) = *(a+1);
    return output;
}
LFUNDEF __prec2 __hpi(LOCID __prec *a ){
    
    __prec2 output;
    *((__prec *)&output) = *a;
    *((__prec *)&output+1) = *(a-1);
    return output;
}
LFUNDEF __prec2 __hpg(GLOBARG float *a ){
    
    __prec2 output;
    *((__prec *)&output) = *a;
    *((__prec *)&output+1) = *(a+1);
    return output;
}
LFUNDEF __prec2 __hpgi(GLOBARG float *a ){
    
    __prec2 output;
    *((__prec *)&output) = *a;
    *((__prec *)&output+1) = *(a-1);
    return output;
}
#endif


#ifndef __OPENCL_VERSION__
//Operators definition for float2 and half2 operations//
__device__ __inline__ float2 operator-(const float2 a) {

    float2 output;
    output.x = -a.x;
    output.y = -a.y;
    return output;

};

__device__ __inline__ float2 operator+(const float2 a, const float2 b) {

    float2 output;
    output.x = a.x+b.x;
    output.y = a.y+b.y;
    return output;

};

__device__ __inline__ float2 operator-(const float2 a, const float2 b) {

    float2 output;
    output.x = a.x-b.x;
    output.y = a.y-b.y;
    return output;

};

__device__ __inline__ float2 operator*(const float2 a, const float2 b) {

    float2 output;
    output.x = a.x*b.x;
    output.y = a.y*b.y;
    return output;

};

__device__ __inline__ float2 operator/(const float2 a, const float2 b) {

    float2 output;
    output.x = a.x/b.x;
    output.y = a.y/b.y;
    return output;

};

__device__ __inline__ float2 operator+(const float a, const float2 b) {

    float2 output;
    output.x = a + b.x;
    output.y = a + b.y;
    return output;

};
__device__ __inline__ float2 operator+(const float2 b, const float a) {

    float2 output;
    output.x = a + b.x;
    output.y = a + b.y;
    return output;

};

__device__ __inline__ float2 operator-(const float a, const float2 b) {

    float2 output;
    output.x = a - b.x;
    output.y = a - b.y;
    return output;

};
__device__ __inline__ float2 operator-(const float2 b, const float a) {

    float2 output;
    output.x = b.x - a;
    output.y = b.y - a;
    return output;

};

__device__ __inline__ float2 operator*(const float a, const float2 b) {

    float2 output;
    output.x = a * b.x;
    output.y = a * b.y;
    return output;

};
__device__ __inline__ float2 operator*(const float2 b, const float a) {

    float2 output;
    output.x = b.x * a;
    output.y = b.y * a;
    return output;

};

__device__ __inline__ float2 operator/(const float a, const float2 b) {

    float2 output;
    output.x = a / b.x;
    output.y = a / b.y;
    return output;

};
__device__ __inline__ float2 operator/(const float2 b, const float a) {

    float2 output;
    output.x = b.x / a;
    output.y = b.y / a;
    return output;

};


__device__ __inline__ half2 operator+(const float a, const half2 b) {

    half2 output;
    output.x = __float2half_rn(a) + b.x;
    output.y = __float2half_rn(a) + b.y;
    return output;

};
__device__ __inline__ half2 operator+(const half2 b, const float a) {

    half2 output;
    output.x = __float2half_rn(a) + b.x;
    output.y = __float2half_rn(a) + b.y;
    return output;

};

__device__ __inline__ half2 operator-(const float a, const half2 b) {

    half2 output;
    output.x = __float2half_rn(a) - b.x;
    output.y = __float2half_rn(a) - b.y;
    return output;

};
__device__ __inline__ half2 operator-(const half2 b, const float a) {

    half2 output;
    output.x = b.x - __float2half_rn(a);
    output.y = b.y - __float2half_rn(a);
    return output;

};

__device__ __inline__ half2 operator*(const float a, const half2 b) {

    half2 output;
    output.x = __float2half_rn(a) * b.x;
    output.y = __float2half_rn(a) * b.y;
    return output;

};
__device__ __inline__ half2 operator*(const half2 b, const float a) {

    half2 output;
    output.x = b.x * __float2half_rn(a);
    output.y = b.y * __float2half_rn(a);
    return output;

};

__device__ __inline__ half2 operator/(const float a, const half2 b) {

    half2 output;
    output.x = __float2half_rn(a) / b.x;
    output.y = __float2half_rn(a) / b.y;
    return output;

};
__device__ __inline__ half2 operator/(const half2 b, const float a) {

    half2 output;
    output.x = b.x / __float2half_rn(a);
    output.y = b.y / __float2half_rn(a);
    return output;

};


__device__ __inline__ float2 operator+(const float2 a, const half2 b) {
    return a + __half22float2(b);
};
__device__ __inline__ float2 operator+(const half2 b, const float2 a) {
    return a + __half22float2(b);
};
__device__ __inline__ float2 operator-(const float2 a, const half2 b) {
    return a - __half22float2(b);
};
__device__ __inline__ float2 operator-(const half2 b, const float2 a) {
    return __half22float2(b) - a;
};
__device__ __inline__ float2 operator*(const float2 a, const half2 b) {
    return a * __half22float2(b);
};
__device__ __inline__ float2 operator*(const half2 b, const float2 a) {
    return __half22float2(b) * a;
};
__device__ __inline__ float2 operator/(const float2 a, const half2 b) {
    return a / __half22float2(b);
};
__device__ __inline__ float2 operator/(const half2 b, const float2 a) {
    return __half22float2(b) / a;
};


//__device__ half2 operator+(half2 a, half2 b) {
//    return __hadd2(a,b);
//};
//__device__ half2 operator-(half2 a, half2 b) {
//    return __hsub2(a,b);
//};
//__device__ half2 operator*(half2 a, half2 b) {
//    return __hmul2(a,b);
//};
//__device__ half2 operator/(half2 a, half2 b) {
//    return __h2div(a,b);
//};
#endif

//Indices for FD stencils//

#if ND==3
    #if LOCAL_OFF==0
        #define ind1(z,y,x)   (x)*DIV*lsizey*lsizez+(y)*DIV*lsizez+(z)
        #define ind2(z,y,x)   (x)*lsizey*lsizez+(y)*lsizez+(z)
        #define indg(z,y,x)   (x)*NY*(NZ)+(y)*(NZ)+(z)
    #else
        #define lidx gidx
        #define lidy gidy
        #define lidz gidz
        #define ind1(z,y,x)   (x)*DIV*(NY)*(NZ)+(y)*DIV*(NZ)+(z)
        #define ind2(z,y,x)   (x)*(NY)*(NZ)+(y)*(NZ)+(z)
    #endif
#else
    #define lidy 0
    #define gidy 0
    #if LOCAL_OFF==0
        #define ind1(z,y,x)  (x)*DIV*lsizez+(z)
        #define ind2(z,y,x)  (x)*lsizez+(z)
        #define indg(z,y,x)  (x)*NZ+(z)
    #else
        #define lidx gidx
        #define lidz gidz
        #define ind1(z,y,x)   (x)*DIV*(NZ)+(z)
        #define ind2(z,y,x)   (x)*(NZ)+(z)
    #endif
#endif


//Load in local memory with the halo for FD in different directions
#define load_local_in(v) lvar2[ind2(lidz,lidy,lidx)]=v[indg(gidz,gidy,gidx)]

#define load_local_halox(v) \
do {\
        if (lidx<2*FDOH)\
            lvar2[ind2(lidz,lidy,lidx-FDOH)]=v[indg(gidz,gidy,gidx-FDOH)];\
        if (lidx+lsizex-3*FDOH<FDOH)\
            lvar2[ind2(lidz,lidy,lidx+lsizex-3*FDOH)]=v[indg(gidz,gidy,gidx+lsizex-3*FDOH)];\
        if (lidx>(lsizex-2*FDOH-1))\
            lvar2[ind2(lidz,lidy,lidx+FDOH)]=v[indg(gidz,gidy,gidx+FDOH)];\
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))\
            lvar2[ind2(lidz,lidy,lidx-lsizex+3*FDOH)]=v[indg(gidz,gidy,gidx-lsizex+3*FDOH)];\
} while(0)

#define load_local_haloy(v) \
do {\
        if (lidy<2*FDOH)\
            lvar2[ind2(lidz,lidy-FDOH,lidx)]=v[indg(gidz,gidy-FDOH,gidx)];\
        if (lidy+lsizey-3*FDOH<FDOH)\
            lvar2[ind2(lidz,lidy+lsizey-3*FDOH,lidx)]=v[indg(gidz,gidy+lsizey-3*FDOH,gidx)];\
        if (lidy>(lsizey-2*FDOH-1))\
            lvar2[ind2(lidz,lidy+FDOH,lidx)]=v[indg(gidz,gidy+FDOH,gidx)];\
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))\
            lvar2[ind2(lidz,lidy-lsizey+3*FDOH,lidx)]=v[indg(gidz,gidy-lsizey+3*FDOH,gidx)];\
} while(0)

#define load_local_haloz(v) \
do {\
        if (lidz<2*FDOH/DIV)\
            lvar2[ind2(lidz-FDOH/DIV,lidy,lidx)]=v[indg(gidz-FDOH/DIV,gidy,gidx)];\
        if (lidz>(lsizez-2*FDOH/DIV-1))\
            lvar2[ind2(lidz+FDOH/DIV,lidy,lidx)]=v[indg(gidz+FDOH/DIV,gidy,gidx)];\
} while(0)


//Forward stencil in x
#if   FDOH ==1
    #define Dxp(v)  HC1*(__h22f2(v[ind2(lidz,lidy,lidx+1)]) - __h22f2(v[ind2(lidz,lidy,lidx)]))
#elif FDOH ==2
    #define Dxp(v)  (HC1*(__h22f2(v[ind2(lidz,lidy,lidx+1)]) - __h22f2(v[ind2(lidz,lidy,lidx)]))+\
                        HC2*(__h22f2(v[ind2(lidz,lidy,lidx+2)]) - __h22f2(v[ind2(lidz,lidy,lidx-1)])))
#elif FDOH ==3
    #define Dxp(v)  (HC1*(__h22f2(v[ind2(lidz,lidy,lidx+1)])-__h22f2(v[ind2(lidz,lidy,lidx)]))+\
                        HC2*(__h22f2(v[ind2(lidz,lidy,lidx+2)])-__h22f2(v[ind2(lidz,lidy,lidx-1)]))+\
                        HC3*(__h22f2(v[ind2(lidz,lidy,lidx+3)])-__h22f2(v[ind2(lidz,lidy,lidx-2)])))
#elif FDOH ==4
    #define Dxp(v)  (HC1*(__h22f2(v[ind2(lidz,lidy,lidx+1)])-__h22f2(v[ind2(lidz,lidy,lidx)]))+\
                        HC2*(__h22f2(v[ind2(lidz,lidy,lidx+2)])-__h22f2(v[ind2(lidz,lidy,lidx-1)]))+\
                        HC3*(__h22f2(v[ind2(lidz,lidy,lidx+3)])-__h22f2(v[ind2(lidz,lidy,lidx-2)]))+\
                        HC4*(__h22f2(v[ind2(lidz,lidy,lidx+4)])-__h22f2(v[ind2(lidz,lidy,lidx-3)])))
#elif FDOH ==5
    #define Dxp(v)  (HC1*(__h22f2(v[ind2(lidz,lidy,lidx+1)])-__h22f2(v[ind2(lidz,lidy,lidx)]))+\
                        HC2*(__h22f2(v[ind2(lidz,lidy,lidx+2)])-__h22f2(v[ind2(lidz,lidy,lidx-1)]))+\
                        HC3*(__h22f2(v[ind2(lidz,lidy,lidx+3)])-__h22f2(v[ind2(lidz,lidy,lidx-2)]))+\
                        HC4*(__h22f2(v[ind2(lidz,lidy,lidx+4)])-__h22f2(v[ind2(lidz,lidy,lidx-3)]))+\
                        HC5*(__h22f2(v[ind2(lidz,lidy,lidx+5)])-__h22f2(v[ind2(lidz,lidy,lidx-4)])))
#elif FDOH ==6
    #define Dxp(v)  (HC1*(__h22f2(v[ind2(lidz,lidy,lidx+1)])-__h22f2(v[ind2(lidz,lidy,lidx)]))+\
                        HC2*(__h22f2(v[ind2(lidz,lidy,lidx+2)])-__h22f2(v[ind2(lidz,lidy,lidx-1)]))+\
                        HC3*(__h22f2(v[ind2(lidz,lidy,lidx+3)])-__h22f2(v[ind2(lidz,lidy,lidx-2)]))+\
                        HC4*(__h22f2(v[ind2(lidz,lidy,lidx+4)])-__h22f2(v[ind2(lidz,lidy,lidx-3)]))+\
                        HC5*(__h22f2(v[ind2(lidz,lidy,lidx+5)])-__h22f2(v[ind2(lidz,lidy,lidx-4)]))+\
                        HC6*(__h22f2(v[ind2(lidz,lidy,lidx+6)])-__h22f2(v[ind2(lidz,lidy,lidx-5)])))
#endif

//Backward stencil in x
#if   FDOH ==1
    #define Dxm(v) HC1*(__h22f2(v[ind2(lidz,lidy,lidx)])   - __h22f2(v[ind2(lidz,lidy,lidx-1)]))
#elif FDOH ==2
    #define Dxm(v) (HC1*(__h22f2(v[ind2(lidz,lidy,lidx)])   - __h22f2(v[ind2(lidz,lidy,lidx-1)]))\
                      +HC2*(__h22f2(v[ind2(lidz,lidy,lidx+1)]) - __h22f2(v[ind2(lidz,lidy,lidx-2)])))
#elif FDOH ==3
    #define Dxm(v) (HC1*(__h22f2(v[ind2(lidz,lidy,lidx)])  -__h22f2(v[ind2(lidz,lidy,lidx-1)]))+\
                       HC2*(__h22f2(v[ind2(lidz,lidy,lidx+1)])-__h22f2(v[ind2(lidz,lidy,lidx-2)]))+\
                       HC3*(__h22f2(v[ind2(lidz,lidy,lidx+2)])-__h22f2(v[ind2(lidz,lidy,lidx-3)])))
#elif FDOH ==4
    #define Dxm(v) (HC1*(__h22f2(v[ind2(lidz,lidy,lidx)])  -__h22f2(v[ind2(lidz,lidy,lidx-1)]))+\
                       HC2*(__h22f2(v[ind2(lidz,lidy,lidx+1)])-__h22f2(v[ind2(lidz,lidy,lidx-2)]))+\
                       HC3*(__h22f2(v[ind2(lidz,lidy,lidx+2)])-__h22f2(v[ind2(lidz,lidy,lidx-3)]))+\
                       HC4*(__h22f2(v[ind2(lidz,lidy,lidx+3)])-__h22f2(v[ind2(lidz,lidy,lidx-4)])))
#elif FDOH ==5
    #define Dxm(v) (HC1*(__h22f2(v[ind2(lidz,lidy,lidx)])  -__h22f2(v[ind2(lidz,lidy,lidx-1)]))+\
                       HC2*(__h22f2(v[ind2(lidz,lidy,lidx+1)])-__h22f2(v[ind2(lidz,lidy,lidx-2)]))+\
                       HC3*(__h22f2(v[ind2(lidz,lidy,lidx+2)])-__h22f2(v[ind2(lidz,lidy,lidx-3)]))+\
                       HC4*(__h22f2(v[ind2(lidz,lidy,lidx+3)])-__h22f2(v[ind2(lidz,lidy,lidx-4)]))+\
                       HC5*(__h22f2(v[ind2(lidz,lidy,lidx+4)])-__h22f2(v[ind2(lidz,lidy,lidx-5)])))
#elif FDOH ==6
    #define Dxm(v) (HC1*(__h22f2(v[ind2(lidz,lidy,lidx)])  -__h22f2(v[ind2(lidz,lidy,lidx-1)]))+\
                       HC2*(__h22f2(v[ind2(lidz,lidy,lidx+1)])-__h22f2(v[ind2(lidz,lidy,lidx-2)]))+\
                       HC3*(__h22f2(v[ind2(lidz,lidy,lidx+2)])-__h22f2(v[ind2(lidz,lidy,lidx-3)]))+\
                       HC4*(__h22f2(v[ind2(lidz,lidy,lidx+3)])-__h22f2(v[ind2(lidz,lidy,lidx-4)]))+\
                       HC5*(__h22f2(v[ind2(lidz,lidy,lidx+4)])-__h22f2(v[ind2(lidz,lidy,lidx-5)]))+\
                       HC6*(__h22f2(v[ind2(lidz,lidy,lidx+5)])-__h22f2(v[ind2(lidz,lidy,lidx-6)])))
#endif

//Forward stencil in y
#if   FDOH ==1
    #define Dyp(v) HC1*(__h22f2(v[ind2(lidz,lidy+1,lidx)])-__h22f2(v[ind2(lidz,lidy,lidx)]))
#elif FDOH ==2
    #define Dyp(v) (HC1*(__h22f2(v[ind2(lidz,lidy+1,lidx)])-__h22f2(v[ind2(lidz,lidy,lidx)]))\
                      +HC2*(__h22f2(v[ind2(lidz,lidy+2,lidx)])-__h22f2(v[ind2(lidz,lidy-1,lidx)])))
#elif FDOH ==3
    #define Dyp(v) (HC1*(__h22f2(v[ind2(lidz,lidy+1,lidx)])-__h22f2(v[ind2(lidz,lidy,lidx)]))+\
                       HC2*(__h22f2(v[ind2(lidz,lidy+2,lidx)])-__h22f2(v[ind2(lidz,lidy-1,lidx)]))+\
                       HC3*(__h22f2(v[ind2(lidz,lidy+3,lidx)])-__h22f2(v[ind2(lidz,lidy-2,lidx)])))
#elif FDOH ==4
    #define Dyp(v) (HC1*(__h22f2(v[ind2(lidz,lidy+1,lidx)])-__h22f2(v[ind2(lidz,lidy,lidx)]))+\
                       HC2*(__h22f2(v[ind2(lidz,lidy+2,lidx)])-__h22f2(v[ind2(lidz,lidy-1,lidx)]))+\
                       HC3*(__h22f2(v[ind2(lidz,lidy+3,lidx)])-__h22f2(v[ind2(lidz,lidy-2,lidx)]))+\
                       HC4*(__h22f2(v[ind2(lidz,lidy+4,lidx)])-__h22f2(v[ind2(lidz,lidy-3,lidx)])))
#elif FDOH ==5
    #define Dyp(v) (HC1*(__h22f2(v[ind2(lidz,lidy+1,lidx)])-__h22f2(v[ind2(lidz,lidy,lidx)]))+\
                       HC2*(__h22f2(v[ind2(lidz,lidy+2,lidx)])-__h22f2(v[ind2(lidz,lidy-1,lidx)]))+\
                       HC3*(__h22f2(v[ind2(lidz,lidy+3,lidx)])-__h22f2(v[ind2(lidz,lidy-2,lidx)]))+\
                       HC4*(__h22f2(v[ind2(lidz,lidy+4,lidx)])-__h22f2(v[ind2(lidz,lidy-3,lidx)]))+\
                       HC5*(__h22f2(v[ind2(lidz,lidy+5,lidx)])-__h22f2(v[ind2(lidz,lidy-4,lidx)])))
#elif FDOH ==6
    #define Dyp(v) (HC1*(__h22f2(v[ind2(lidz,lidy+1,lidx)])-__h22f2(v[ind2(lidz,lidy,lidx)]))+\
                       HC2*(__h22f2(v[ind2(lidz,lidy+2,lidx)])-__h22f2(v[ind2(lidz,lidy-1,lidx)]))+\
                       HC3*(__h22f2(v[ind2(lidz,lidy+3,lidx)])-__h22f2(v[ind2(lidz,lidy-2,lidx)]))+\
                       HC4*(__h22f2(v[ind2(lidz,lidy+4,lidx)])-__h22f2(v[ind2(lidz,lidy-3,lidx)]))+\
                       HC5*(__h22f2(v[ind2(lidz,lidy+5,lidx)])-__h22f2(v[ind2(lidz,lidy-4,lidx)]))+\
                       HC6*(__h22f2(v[ind2(lidz,lidy+6,lidx)])-__h22f2(v[ind2(lidz,lidy-5,lidx)])))
#endif

//Backward stencil in y
#if   FDOH ==1
    #define Dym(v) HC1*(__h22f2(v[ind2(lidz,lidy,lidx)])-__h22f2(v[ind2(lidz,lidy-1,lidx)]))
#elif FDOH ==2
    #define Dym(v) (HC1*(__h22f2(v[ind2(lidz,lidy,lidx)])-__h22f2(v[ind2(lidz,lidy-1,lidx)]))\
                      +HC2*(__h22f2(v[ind2(lidz,lidy+1,lidx)])-__h22f2(v[ind2(lidz,lidy-2,lidx)])))
#elif FDOH ==3
    #define Dym(v) (HC1*(__h22f2(v[ind2(lidz,lidy,lidx)])-__h22f2(v[ind2(lidz,lidy-1,lidx)]))+\
                       HC2*(__h22f2(v[ind2(lidz,lidy+1,lidx)])-__h22f2(v[ind2(lidz,lidy-2,lidx)]))+\
                       HC3*(__h22f2(v[ind2(lidz,lidy+2,lidx)])-__h22f2(v[ind2(lidz,lidy-3,lidx)])))
#elif FDOH ==4
    #define Dym(v) (HC1*(__h22f2(v[ind2(lidz,lidy,lidx)])-__h22f2(v[ind2(lidz,lidy-1,lidx)]))+\
                       HC2*(__h22f2(v[ind2(lidz,lidy+1,lidx)])-__h22f2(v[ind2(lidz,lidy-2,lidx)]))+\
                       HC3*(__h22f2(v[ind2(lidz,lidy+2,lidx)])-__h22f2(v[ind2(lidz,lidy-3,lidx)]))+\
                       HC4*(__h22f2(v[ind2(lidz,lidy+3,lidx)])-__h22f2(v[ind2(lidz,lidy-4,lidx)])))
#elif FDOH ==5
    #define Dym(v) (HC1*(__h22f2(v[ind2(lidz,lidy,lidx)])-__h22f2(v[ind2(lidz,lidy-1,lidx)]))+\
                       HC2*(__h22f2(v[ind2(lidz,lidy+1,lidx)])-__h22f2(v[ind2(lidz,lidy-2,lidx)]))+\
                       HC3*(__h22f2(v[ind2(lidz,lidy+2,lidx)])-__h22f2(v[ind2(lidz,lidy-3,lidx)]))+\
                       HC4*(__h22f2(v[ind2(lidz,lidy+3,lidx)])-__h22f2(v[ind2(lidz,lidy-4,lidx)]))+\
                       HC5*(__h22f2(v[ind2(lidz,lidy+4,lidx)])-__h22f2(v[ind2(lidz,lidy-5,lidx)])))
#elif FDOH ==6
    #define Dym(v) (HC1*(__h22f2(v[ind2(lidz,lidy,lidx)])-__h22f2(v[ind2(lidz,lidy-1,lidx)]))+\
                       HC2*(__h22f2(v[ind2(lidz,lidy+1,lidx)])-__h22f2(v[ind2(lidz,lidy-2,lidx)]))+\
                       HC3*(__h22f2(v[ind2(lidz,lidy+2,lidx)])-__h22f2(v[ind2(lidz,lidy-3,lidx)]))+\
                       HC4*(__h22f2(v[ind2(lidz,lidy+3,lidx)])-__h22f2(v[ind2(lidz,lidy-4,lidx)]))+\
                       HC5*(__h22f2(v[ind2(lidz,lidy+4,lidx)])-__h22f2(v[ind2(lidz,lidy-5,lidx)]))+\
                       HC6*(__h22f2(v[ind2(lidz,lidy+5,lidx)])-__h22f2(v[ind2(lidz,lidy-6,lidx)])))
#endif

//Forward stencil in z
#if   FDOH ==1
    #define Dzp(v) HC1*(__h22f2(__hp(&v[ind1(DIV*lidz+1,lidy,lidx)])) - __h22f2(__hp(&v[ind1(DIV*lidz,lidy,lidx)])))
#elif FDOH ==2
    #define Dzp(v) (HC1*(__h22f2(__hp(&v[ind1(DIV*lidz+1,lidy,lidx)])) - __h22f2(__hp(&v[ind1(DIV*lidz,lidy,lidx)])))\
                      +HC2*(__h22f2(__hp(&v[ind1(DIV*lidz+2,lidy,lidx)])) - __h22f2(__hp(&v[ind1(DIV*lidz-1,lidy,lidx)]))))
#elif FDOH ==3
    #define Dzp(v) (HC1*(__h22f2(__hp(&v[ind1(DIV*lidz+1,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz,lidy,lidx)])))+\
                       HC2*(__h22f2(__hp(&v[ind1(DIV*lidz+2,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-1,lidy,lidx)])))+\
                       HC3*(__h22f2(__hp(&v[ind1(DIV*lidz+3,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-2,lidy,lidx)]))))
#elif FDOH ==4
    #define Dzp(v) (HC1*(__h22f2(__hp(&v[ind1(DIV*lidz+1,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz,lidy,lidx)])))+\
                       HC2*(__h22f2(__hp(&v[ind1(DIV*lidz+2,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-1,lidy,lidx)])))+\
                       HC3*(__h22f2(__hp(&v[ind1(DIV*lidz+3,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-2,lidy,lidx)])))+\
                       HC4*(__h22f2(__hp(&v[ind1(DIV*lidz+4,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-3,lidy,lidx)]))))
#elif FDOH ==5
    #define Dzp(v) (HC1*(__h22f2(__hp(&v[ind1(DIV*lidz+1,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz,lidy,lidx)])))+\
                       HC2*(__h22f2(__hp(&v[ind1(DIV*lidz+2,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-1,lidy,lidx)])))+\
                       HC3*(__h22f2(__hp(&v[ind1(DIV*lidz+3,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-2,lidy,lidx)])))+\
                       HC4*(__h22f2(__hp(&v[ind1(DIV*lidz+4,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-3,lidy,lidx)])))+\
                       HC5*(__h22f2(__hp(&v[ind1(DIV*lidz+5,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-4,lidy,lidx)]))))
#elif FDOH ==6
    #define Dzp(v) (HC1*(__h22f2(__hp(&v[ind1(DIV*lidz+1,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz,lidy,lidx)])))+\
                       HC2*(__h22f2(__hp(&v[ind1(DIV*lidz+2,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-1,lidy,lidx)])))+\
                       HC3*(__h22f2(__hp(&v[ind1(DIV*lidz+3,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-2,lidy,lidx)])))+\
                       HC4*(__h22f2(__hp(&v[ind1(DIV*lidz+4,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-3,lidy,lidx)])))+\
                       HC5*(__h22f2(__hp(&v[ind1(DIV*lidz+5,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-4,lidy,lidx)])))+\
                       HC6*(__h22f2(__hp(&v[ind1(DIV*lidz+6,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-5,lidy,lidx)]))))
#endif


//Backward stencil in z
#if   FDOH ==1
    #define Dzm(v) HC1*(__h22f2(__hp(&v[ind1(DIV*lidz,lidy,lidx)]))   - __h22f2(__hp(&v[ind1(DIV*lidz-1,lidy,lidx)])))
#elif FDOH ==2
    #define Dzm(v) (HC1*(__h22f2(__hp(&v[ind1(DIV*lidz,lidy,lidx)]))   - __h22f2(__hp(&v[ind1(DIV*lidz-1,lidy,lidx)])))\
                      +HC2*(__h22f2(__hp(&v[ind1(DIV*lidz+1,lidy,lidx)])) - __h22f2(__hp(&v[ind1(DIV*lidz-2,lidy,lidx)]))))
#elif FDOH ==3
    #define Dzm(v) (HC1*(__h22f2(__hp(&v[ind1(DIV*lidz,lidy,lidx)]))  -__h22f2(__hp(&v[ind1(DIV*lidz-1,lidy,lidx)])))+\
                       HC2*(__h22f2(__hp(&v[ind1(DIV*lidz+1,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-2,lidy,lidx)])))+\
                       HC3*(__h22f2(__hp(&v[ind1(DIV*lidz+2,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-3,lidy,lidx)]))))
#elif FDOH ==4
    #define Dzm(v) (HC1*(__h22f2(__hp(&v[ind1(DIV*lidz,lidy,lidx)]))  -__h22f2(__hp(&v[ind1(DIV*lidz-1,lidy,lidx)])))+\
                       HC2*(__h22f2(__hp(&v[ind1(DIV*lidz+1,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-2,lidy,lidx)])))+\
                       HC3*(__h22f2(__hp(&v[ind1(DIV*lidz+2,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-3,lidy,lidx)])))+\
                       HC4*(__h22f2(__hp(&v[ind1(DIV*lidz+3,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-4,lidy,lidx)]))))
#elif FDOH ==5
    #define Dzm(v) (HC1*(__h22f2(__hp(&v[ind1(DIV*lidz,lidy,lidx)]))  -__h22f2(__hp(&v[ind1(DIV*lidz-1,lidy,lidx)])))+\
                       HC2*(__h22f2(__hp(&v[ind1(DIV*lidz+1,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-2,lidy,lidx)])))+\
                       HC3*(__h22f2(__hp(&v[ind1(DIV*lidz+2,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-3,lidy,lidx)])))+\
                       HC4*(__h22f2(__hp(&v[ind1(DIV*lidz+3,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-4,lidy,lidx)])))+\
                       HC5*(__h22f2(__hp(&v[ind1(DIV*lidz+4,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-5,lidy,lidx)]))))
#elif FDOH ==6
    #define Dzm(v) (HC1*(__h22f2(__hp(&v[ind1(DIV*lidz,lidy,lidx)]))  -__h22f2(__hp(&v[ind1(DIV*lidz-1,lidy,lidx)])))+\
                       HC2*(__h22f2(__hp(&v[ind1(DIV*lidz+1,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-2,lidy,lidx)])))+\
                       HC3*(__h22f2(__hp(&v[ind1(DIV*lidz+2,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-3,lidy,lidx)])))+\
                       HC4*(__h22f2(__hp(&v[ind1(DIV*lidz+3,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-4,lidy,lidx)])))+\
                       HC5*(__h22f2(__hp(&v[ind1(DIV*lidz+4,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-5,lidy,lidx)])))+\
                       HC6*(__h22f2(__hp(&v[ind1(DIV*lidz+5,lidy,lidx)]))-__h22f2(__hp(&v[ind1(DIV*lidz-6,lidy,lidx)]))))
#endif

