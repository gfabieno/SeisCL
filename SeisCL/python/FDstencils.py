from SeisCL.python.pycl_backend import StateKernelGPU


FDstencils = """
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

#define DIV 1
#define __gprec float
#define __prec float
#define __prec2 float
#define __pprec float
#define __cprec1 float
#define __h22f2(x) (x)
#define __f22h2(x) (x)
#define __pconv(x) (x)
#define __f22h2c(x) (x)
#define __h22f2c(x) (x)
#define __cprec float2
#define initc0(v) x=0.0f

#if ND==3
    #if LOCAL_OFF==0
        #define ind(z,y,x)   (x)*lsizey*lsizez+(y)*lsizez+(z)
        #define indg(z,y,x)  (x)*NY*(NZ)+(y)*(NZ)+(z)
    #else
        #define lidx gidx
        #define lidy gidy
        #define lidz gidz
        #define ind(z,y,x)   (x)*NY*(NZ)+(y)*(NZ)+(z)
    #endif
#else
    #define lidy 0
    #define gidy 0
    #if LOCAL_OFF==0
        #define ind(z,y,x)   (x)*lsizez+(z)
        #define indg(z,y,x)  (x)*(NZ)+(z)
    #else
        #define lidx gidx
        #define lidz gidz
        #define ind(z,y,x)   (x)*(NZ)+(z)
    #endif
#endif

//Load in local memory with the halo for FD in different directions
#define load_local_in(v) lvar[ind(lidz,lidy,lidx)]=v[indg(gidz,gidy,gidx)]

#define load_local_halox(v) \
    do{\
        if (lidx<2*FDOH)\
            lvar[ind(lidz,lidy,lidx-FDOH)]=v[indg(gidz,gidy,gidx-FDOH)];\
        if (lidx+lsizex-3*FDOH<FDOH)\
            lvar[ind(lidz,lidy,lidx+lsizex-3*FDOH)]=v[indg(gidz,gidy,gidx+lsizex-3*FDOH)];\
        if (lidx>(lsizex-2*FDOH-1))\
            lvar[ind(lidz,lidy,lidx+FDOH)]=v[indg(gidz,gidy,gidx+FDOH)];\
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))\
            lvar[ind(lidz,lidy,lidx-lsizex+3*FDOH)]=v[indg(gidz,gidy,gidx-lsizex+3*FDOH)];\
        } while(0)

#define load_local_haloy(v) \
    do{\
        if (lidy<2*FDOH)\
            lvar[ind(lidz,lidy-FDOH,lidx)]=v[indg(gidz,gidy-FDOH,gidx)];\
        if (lidy+lsizey-3*FDOH<FDOH)\
            lvar[ind(lidz,lidy+lsizey-3*FDOH,lidx)]=v[indg(gidz,gidy+lsizey-3*FDOH,gidx)];\
        if (lidy>(lsizey-2*FDOH-1))\
            lvar[ind(lidz,lidy+FDOH,lidx)]=v[indg(gidz,gidy+FDOH,gidx)];\
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))\
            lvar[ind(lidz,lidy-lsizey+3*FDOH,lidx)]=v[indg(gidz,gidy-lsizey+3*FDOH,gidx)];\
     } while(0)

#define load_local_haloz(v) \
    do{\
        if (lidz<2*FDOH)\
            lvar[ind(lidz-FDOH,lidy,lidx)]=v[indg(gidz-FDOH,gidy,gidx)];\
        if (lidz>(lsizez-2*FDOH-1))\
            lvar[ind(lidz+FDOH,lidy,lidx)]=v[indg(gidz+FDOH,gidy,gidx)];\
    } while(0)


//Forward stencil in x
#if   FDOH ==1
    #define Dxp(v)  HC1*(v[ind(lidz,lidy,lidx+1)] - v[ind(lidz,lidy,lidx)])
#elif FDOH ==2
    #define Dxp(v)  (HC1*(v[ind(lidz,lidy,lidx+1)] - v[ind(lidz,lidy,lidx)])+\
                        HC2*(v[ind(lidz,lidy,lidx+2)] - v[ind(lidz,lidy,lidx-1)]))
#elif FDOH ==3
    #define Dxp(v)  (HC1*(v[ind(lidz,lidy,lidx+1)]-v[ind(lidz,lidy,lidx)])+\
                        HC2*(v[ind(lidz,lidy,lidx+2)]-v[ind(lidz,lidy,lidx-1)])+\
                        HC3*(v[ind(lidz,lidy,lidx+3)]-v[ind(lidz,lidy,lidx-2)]))
#elif FDOH ==4
    #define Dxp(v)  (HC1*(v[ind(lidz,lidy,lidx+1)]-v[ind(lidz,lidy,lidx)])+\
                        HC2*(v[ind(lidz,lidy,lidx+2)]-v[ind(lidz,lidy,lidx-1)])+\
                        HC3*(v[ind(lidz,lidy,lidx+3)]-v[ind(lidz,lidy,lidx-2)])+\
                        HC4*(v[ind(lidz,lidy,lidx+4)]-v[ind(lidz,lidy,lidx-3)]))
#elif FDOH ==5
    #define Dxp(v)  (HC1*(v[ind(lidz,lidy,lidx+1)]-v[ind(lidz,lidy,lidx)])+\
                        HC2*(v[ind(lidz,lidy,lidx+2)]-v[ind(lidz,lidy,lidx-1)])+\
                        HC3*(v[ind(lidz,lidy,lidx+3)]-v[ind(lidz,lidy,lidx-2)])+\
                        HC4*(v[ind(lidz,lidy,lidx+4)]-v[ind(lidz,lidy,lidx-3)])+\
                        HC5*(v[ind(lidz,lidy,lidx+5)]-v[ind(lidz,lidy,lidx-4)]))
#elif FDOH ==6
    #define Dxp(v)  (HC1*(v[ind(lidz,lidy,lidx+1)]-v[ind(lidz,lidy,lidx)])+\
                        HC2*(v[ind(lidz,lidy,lidx+2)]-v[ind(lidz,lidy,lidx-1)])+\
                        HC3*(v[ind(lidz,lidy,lidx+3)]-v[ind(lidz,lidy,lidx-2)])+\
                        HC4*(v[ind(lidz,lidy,lidx+4)]-v[ind(lidz,lidy,lidx-3)])+\
                        HC5*(v[ind(lidz,lidy,lidx+5)]-v[ind(lidz,lidy,lidx-4)])+\
                        HC6*(v[ind(lidz,lidy,lidx+6)]-v[ind(lidz,lidy,lidx-5)]))
#endif

//Backward stencil in x
#if   FDOH ==1
    #define Dxm(v) HC1*(v[ind(lidz,lidy,lidx)]   - v[ind(lidz,lidy,lidx-1)])
#elif FDOH ==2
    #define Dxm(v) (HC1*(v[ind(lidz,lidy,lidx)]   - v[ind(lidz,lidy,lidx-1)])\
                      +HC2*(v[ind(lidz,lidy,lidx+1)] - v[ind(lidz,lidy,lidx-2)]))
#elif FDOH ==3
    #define Dxm(v) (HC1*(v[ind(lidz,lidy,lidx)]  -v[ind(lidz,lidy,lidx-1)])+\
                       HC2*(v[ind(lidz,lidy,lidx+1)]-v[ind(lidz,lidy,lidx-2)])+\
                       HC3*(v[ind(lidz,lidy,lidx+2)]-v[ind(lidz,lidy,lidx-3)]))
#elif FDOH ==4
    #define Dxm(v) (HC1*(v[ind(lidz,lidy,lidx)]  -v[ind(lidz,lidy,lidx-1)])+\
                       HC2*(v[ind(lidz,lidy,lidx+1)]-v[ind(lidz,lidy,lidx-2)])+\
                       HC3*(v[ind(lidz,lidy,lidx+2)]-v[ind(lidz,lidy,lidx-3)])+\
                       HC4*(v[ind(lidz,lidy,lidx+3)]-v[ind(lidz,lidy,lidx-4)]))
#elif FDOH ==5
    #define Dxm(v) (HC1*(v[ind(lidz,lidy,lidx)]  -v[ind(lidz,lidy,lidx-1)])+\
                       HC2*(v[ind(lidz,lidy,lidx+1)]-v[ind(lidz,lidy,lidx-2)])+\
                       HC3*(v[ind(lidz,lidy,lidx+2)]-v[ind(lidz,lidy,lidx-3)])+\
                       HC4*(v[ind(lidz,lidy,lidx+3)]-v[ind(lidz,lidy,lidx-4)])+\
                       HC5*(v[ind(lidz,lidy,lidx+4)]-v[ind(lidz,lidy,lidx-5)]))
#elif FDOH ==6
    #define Dxm(v) (HC1*(v[ind(lidz,lidy,lidx)]  -v[ind(lidz,lidy,lidx-1)])+\
                       HC2*(v[ind(lidz,lidy,lidx+1)]-v[ind(lidz,lidy,lidx-2)])+\
                       HC3*(v[ind(lidz,lidy,lidx+2)]-v[ind(lidz,lidy,lidx-3)])+\
                       HC4*(v[ind(lidz,lidy,lidx+3)]-v[ind(lidz,lidy,lidx-4)])+\
                       HC5*(v[ind(lidz,lidy,lidx+4)]-v[ind(lidz,lidy,lidx-5)])+\
                       HC6*(v[ind(lidz,lidy,lidx+5)]-v[ind(lidz,lidy,lidx-6)]))
#endif

//Forward stencil in y
#if   FDOH ==1
    #define Dyp(v) HC1*(v[ind(lidz,lidy+1,lidx)]-v[ind(lidz,lidy,lidx)])
#elif FDOH ==2
    #define Dyp(v) (HC1*(v[ind(lidz,lidy+1,lidx)]-v[ind(lidz,lidy,lidx)])\
                      +HC2*(v[ind(lidz,lidy+2,lidx)]-v[ind(lidz,lidy-1,lidx)]))
#elif FDOH ==3
    #define Dyp(v) (HC1*(v[ind(lidz,lidy+1,lidx)]-v[ind(lidz,lidy,lidx)])+\
                       HC2*(v[ind(lidz,lidy+2,lidx)]-v[ind(lidz,lidy-1,lidx)])+\
                       HC3*(v[ind(lidz,lidy+3,lidx)]-v[ind(lidz,lidy-2,lidx)]))
#elif FDOH ==4
    #define Dyp(v) (HC1*(v[ind(lidz,lidy+1,lidx)]-v[ind(lidz,lidy,lidx)])+\
                       HC2*(v[ind(lidz,lidy+2,lidx)]-v[ind(lidz,lidy-1,lidx)])+\
                       HC3*(v[ind(lidz,lidy+3,lidx)]-v[ind(lidz,lidy-2,lidx)])+\
                       HC4*(v[ind(lidz,lidy+4,lidx)]-v[ind(lidz,lidy-3,lidx)]))
#elif FDOH ==5
    #define Dyp(v) (HC1*(v[ind(lidz,lidy+1,lidx)]-v[ind(lidz,lidy,lidx)])+\
                       HC2*(v[ind(lidz,lidy+2,lidx)]-v[ind(lidz,lidy-1,lidx)])+\
                       HC3*(v[ind(lidz,lidy+3,lidx)]-v[ind(lidz,lidy-2,lidx)])+\
                       HC4*(v[ind(lidz,lidy+4,lidx)]-v[ind(lidz,lidy-3,lidx)])+\
                       HC5*(v[ind(lidz,lidy+5,lidx)]-v[ind(lidz,lidy-4,lidx)]))
#elif FDOH ==6
    #define Dyp(v) (HC1*(v[ind(lidz,lidy+1,lidx)]-v[ind(lidz,lidy,lidx)])+\
                       HC2*(v[ind(lidz,lidy+2,lidx)]-v[ind(lidz,lidy-1,lidx)])+\
                       HC3*(v[ind(lidz,lidy+3,lidx)]-v[ind(lidz,lidy-2,lidx)])+\
                       HC4*(v[ind(lidz,lidy+4,lidx)]-v[ind(lidz,lidy-3,lidx)])+\
                       HC5*(v[ind(lidz,lidy+5,lidx)]-v[ind(lidz,lidy-4,lidx)])+\
                       HC6*(v[ind(lidz,lidy+6,lidx)]-v[ind(lidz,lidy-5,lidx)]))
#endif

//Backward stencil in y
#if   FDOH ==1
    #define Dym(v) HC1*(v[ind(lidz,lidy,lidx)]-v[ind(lidz,lidy-1,lidx)])
#elif FDOH ==2
    #define Dym(v) (HC1*(v[ind(lidz,lidy,lidx)]-v[ind(lidz,lidy-1,lidx)])\
                      +HC2*(v[ind(lidz,lidy+1,lidx)]-v[ind(lidz,lidy-2,lidx)]))
#elif FDOH ==3
    #define Dym(v) (HC1*(v[ind(lidz,lidy,lidx)]-v[ind(lidz,lidy-1,lidx)])+\
                       HC2*(v[ind(lidz,lidy+1,lidx)]-v[ind(lidz,lidy-2,lidx)])+\
                       HC3*(v[ind(lidz,lidy+2,lidx)]-v[ind(lidz,lidy-3,lidx)]))
#elif FDOH ==4
    #define Dym(v) (HC1*(v[ind(lidz,lidy,lidx)]-v[ind(lidz,lidy-1,lidx)])+\
                       HC2*(v[ind(lidz,lidy+1,lidx)]-v[ind(lidz,lidy-2,lidx)])+\
                       HC3*(v[ind(lidz,lidy+2,lidx)]-v[ind(lidz,lidy-3,lidx)])+\
                       HC4*(v[ind(lidz,lidy+3,lidx)]-v[ind(lidz,lidy-4,lidx)]))
#elif FDOH ==5
    #define Dym(v) (HC1*(v[ind(lidz,lidy,lidx)]-v[ind(lidz,lidy-1,lidx)])+\
                       HC2*(v[ind(lidz,lidy+1,lidx)]-v[ind(lidz,lidy-2,lidx)])+\
                       HC3*(v[ind(lidz,lidy+2,lidx)]-v[ind(lidz,lidy-3,lidx)])+\
                       HC4*(v[ind(lidz,lidy+3,lidx)]-v[ind(lidz,lidy-4,lidx)])+\
                       HC5*(v[ind(lidz,lidy+4,lidx)]-v[ind(lidz,lidy-5,lidx)]))
#elif FDOH ==6
    #define Dym(v) (HC1*(v[ind(lidz,lidy,lidx)]-v[ind(lidz,lidy-1,lidx)])+\
                       HC2*(v[ind(lidz,lidy+1,lidx)]-v[ind(lidz,lidy-2,lidx)])+\
                       HC3*(v[ind(lidz,lidy+2,lidx)]-v[ind(lidz,lidy-3,lidx)])+\
                       HC4*(v[ind(lidz,lidy+3,lidx)]-v[ind(lidz,lidy-4,lidx)])+\
                       HC5*(v[ind(lidz,lidy+4,lidx)]-v[ind(lidz,lidy-5,lidx)])+\
                       HC6*(v[ind(lidz,lidy+5,lidx)]-v[ind(lidz,lidy-6,lidx)]))
#endif

//Forward stencil in z
#if   FDOH ==1
    #define Dzp(v) HC1*(v[ind(lidz+1,lidy,lidx)] - v[ind(lidz,lidy,lidx)])
#elif FDOH ==2
    #define Dzp(v) (HC1*(v[ind(lidz+1,lidy,lidx)] - v[ind(lidz,lidy,lidx)])\
                      +HC2*(v[ind(lidz+2,lidy,lidx)] - v[ind(lidz-1,lidy,lidx)]))
#elif FDOH ==3
    #define Dzp(v) (HC1*(v[ind(lidz+1,lidy,lidx)]-v[ind(lidz,lidy,lidx)])+\
                       HC2*(v[ind(lidz+2,lidy,lidx)]-v[ind(lidz-1,lidy,lidx)])+\
                       HC3*(v[ind(lidz+3,lidy,lidx)]-v[ind(lidz-2,lidy,lidx)]))
#elif FDOH ==4
    #define Dzp(v) (HC1*(v[ind(lidz+1,lidy,lidx)]-v[ind(lidz,lidy,lidx)])+\
                       HC2*(v[ind(lidz+2,lidy,lidx)]-v[ind(lidz-1,lidy,lidx)])+\
                       HC3*(v[ind(lidz+3,lidy,lidx)]-v[ind(lidz-2,lidy,lidx)])+\
                       HC4*(v[ind(lidz+4,lidy,lidx)]-v[ind(lidz-3,lidy,lidx)]))
#elif FDOH ==5
    #define Dzp(v) (HC1*(v[ind(lidz+1,lidy,lidx)]-v[ind(lidz,lidy,lidx)])+\
                       HC2*(v[ind(lidz+2,lidy,lidx)]-v[ind(lidz-1,lidy,lidx)])+\
                       HC3*(v[ind(lidz+3,lidy,lidx)]-v[ind(lidz-2,lidy,lidx)])+\
                       HC4*(v[ind(lidz+4,lidy,lidx)]-v[ind(lidz-3,lidy,lidx)])+\
                       HC5*(v[ind(lidz+5,lidy,lidx)]-v[ind(lidz-4,lidy,lidx)]))
#elif FDOH ==6
    #define Dzp(v) (HC1*(v[ind(lidz+1,lidy,lidx)]-v[ind(lidz,lidy,lidx)])+\
                       HC2*(v[ind(lidz+2,lidy,lidx)]-v[ind(lidz-1,lidy,lidx)])+\
                       HC3*(v[ind(lidz+3,lidy,lidx)]-v[ind(lidz-2,lidy,lidx)])+\
                       HC4*(v[ind(lidz+4,lidy,lidx)]-v[ind(lidz-3,lidy,lidx)])+\
                       HC5*(v[ind(lidz+5,lidy,lidx)]-v[ind(lidz-4,lidy,lidx)])+\
                       HC6*(v[ind(lidz+6,lidy,lidx)]-v[ind(lidz-5,lidy,lidx)]))
#endif

//Backward stencil in z
#if   FDOH ==1
    #define Dzm(v) HC1*(v[ind(lidz,lidy,lidx)]   - v[ind(lidz-1,lidy,lidx)])
#elif FDOH ==2
    #define Dzm(v) (HC1*(v[ind(lidz,lidy,lidx)]   - v[ind(lidz-1,lidy,lidx)])\
                      +HC2*(v[ind(lidz+1,lidy,lidx)] - v[ind(lidz-2,lidy,lidx)]))
#elif FDOH ==3
    #define Dzm(v) (HC1*(v[ind(lidz,lidy,lidx)]  -v[ind(lidz-1,lidy,lidx)])+\
                       HC2*(v[ind(lidz+1,lidy,lidx)]-v[ind(lidz-2,lidy,lidx)])+\
                       HC3*(v[ind(lidz+2,lidy,lidx)]-v[ind(lidz-3,lidy,lidx)]))
#elif FDOH ==4
    #define Dzm(v) (HC1*(v[ind(lidz,lidy,lidx)]  -v[ind(lidz-1,lidy,lidx)])+\
                       HC2*(v[ind(lidz+1,lidy,lidx)]-v[ind(lidz-2,lidy,lidx)])+\
                       HC3*(v[ind(lidz+2,lidy,lidx)]-v[ind(lidz-3,lidy,lidx)])+\
                       HC4*(v[ind(lidz+3,lidy,lidx)]-v[ind(lidz-4,lidy,lidx)]))
#elif FDOH ==5
    #define Dzm(v) (HC1*(v[ind(lidz,lidy,lidx)]  -v[ind(lidz-1,lidy,lidx)])+\
                       HC2*(v[ind(lidz+1,lidy,lidx)]-v[ind(lidz-2,lidy,lidx)])+\
                       HC3*(v[ind(lidz+2,lidy,lidx)]-v[ind(lidz-3,lidy,lidx)])+\
                       HC4*(v[ind(lidz+3,lidy,lidx)]-v[ind(lidz-4,lidy,lidx)])+\
                       HC5*(v[ind(lidz+4,lidy,lidx)]-v[ind(lidz-5,lidy,lidx)]))
#elif FDOH ==6
    #define Dzm(v) (HC1*(v[ind(lidz,lidy,lidx)]  -v[ind(lidz-1,lidy,lidx)])+\
                       HC2*(v[ind(lidz+1,lidy,lidx)]-v[ind(lidz-2,lidy,lidx)])+\
                       HC3*(v[ind(lidz+2,lidy,lidx)]-v[ind(lidz-3,lidy,lidx)])+\
                       HC4*(v[ind(lidz+3,lidy,lidx)]-v[ind(lidz-4,lidy,lidx)])+\
                       HC5*(v[ind(lidz+4,lidy,lidx)]-v[ind(lidz-5,lidy,lidx)])+\
                       HC6*(v[ind(lidz+5,lidy,lidx)]-v[ind(lidz-6,lidy,lidx)]))
#endif
"""


class Derivative(StateKernelGPU):
    forward_src = """
__kernel void Dx
(
    __global const float *a, __local float *lvar, )
{
  int gidx = get_global_id(0);
  int gidz = get_global_id(0);
  int lidx = get_gl
  res[gid] = a[gid] + b[gid];
}
"""


