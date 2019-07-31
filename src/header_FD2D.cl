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
    #define lvar(z,x)  lvar[(x)*lsizez+(z)]
    #define ind(z,x)   (x)*lsizez+(z)
    #define indg(z,x)   (x)*(NZ)+(z)
#else
    #define lvar(z,x)  lvar[(x)*(NZ)+(z)]
    #define lidx gidx
    #define lidz gidz
    #define ind(z,x)   (x)*(NZ)+(z)
#endif

//Load in local memory with the halo for FD in different directions
#define load_local_x(v) ({\
        lvar[ind(lidz,lidx)]=v[indg(gidz, gidx)];\
        if (lidx<2*FDOH)\
            lvar[ind(lidz,lidx-FDOH)]=v[indg(gidz,gidx-FDOH)];\
        if (lidx+lsizex-3*FDOH<FDOH)\
            lvar[ind(lidz,lidx+lsizex-3*FDOH)]=v[indg(gidz,gidx+lsizex-3*FDOH)];\
        if (lidx>(lsizex-2*FDOH-1))\
            lvar[ind(lidz,lidx+FDOH)]=v[indg(gidz,gidx+FDOH)];\
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))\
            lvar[ind(lidz,lidx-lsizex+3*FDOH)]=v[indg(gidz,gidx-lsizex+3*FDOH)];\
        })

#define load_local_z(v) ({\
        lvar[ind(lidz,lidx)]=v[indg(gidz, gidx)];\
        if (lidz<2*FDOH)\
            lvar[ind(lidz-FDOH,lidx)]=v[indg(gidz-FDOH,gidx)];\
        if (lidz>(lsizez-2*FDOH-1))\
            lvar[ind(lidz+FDOH,lidx)]=v[indg(gidz+FDOH,gidx)];\
})

#define load_local_xz(v) ({\
        lvar[ind(lidz,lidx)]=v[indg(gidz, gidx)];\
        if (lidx<2*FDOH)\
            lvar[ind(lidz,lidx-FDOH)]=v[indg(gidz,gidx-FDOH)];\
        if (lidx+lsizex-3*FDOH<FDOH)\
            lvar[ind(lidz,lidx+lsizex-3*FDOH)]=v[indg(gidz,gidx+lsizex-3*FDOH)];\
        if (lidx>(lsizex-2*FDOH-1))\
            lvar[ind(lidz,lidx+FDOH)]=v[indg(gidz,gidx+FDOH)];\
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))\
            lvar[ind(lidz,lidx-lsizex+3*FDOH)]=v[indg(gidz,gidx-lsizex+3*FDOH)];\
        if (lidz<2*FDOH)\
            lvar[ind(lidz-FDOH,lidx)]=v[indg(gidz-FDOH,gidx)];\
        if (lidz>(lsizez-2*FDOH-1))\
            lvar[ind(lidz+FDOH,lidx)]=v[indg(gidz+FDOH,gidx)];\
})



//Forward stencil in x
#if   FDOH ==1
    #define Dxp(v)  HC1*(v[ind(lidz,lidx+1)] - v[ind(lidz,lidx)])
#elif FDOH ==2
    #define Dxp(v)  (HC1*(v[ind(lidz,lidx+1)] - v[ind(lidz,lidx)])+\
                        HC2*(v[ind(lidz,lidx+2)] - v[ind(lidz,lidx-1)]))
#elif FDOH ==3
    #define Dxp(v)  (HC1*(v[ind(lidz,lidx+1)]-v[ind(lidz,lidx)])+\
                        HC2*(v[ind(lidz,lidx+2)]-v[ind(lidz,lidx-1)])+\
                        HC3*(v[ind(lidz,lidx+3)]-v[ind(lidz,lidx-2)]))
#elif FDOH ==4
    #define Dxp(v)  (HC1*(v[ind(lidz,lidx+1)]-v[ind(lidz,lidx)])+\
                        HC2*(v[ind(lidz,lidx+2)]-v[ind(lidz,lidx-1)])+\
                        HC3*(v[ind(lidz,lidx+3)]-v[ind(lidz,lidx-2)])+\
                        HC4*(v[ind(lidz,lidx+4)]-v[ind(lidz,lidx-3)]))
#elif FDOH ==5
    #define Dxp(v)  (HC1*(v[ind(lidz,lidx+1)]-v[ind(lidz,lidx)])+\
                        HC2*(v[ind(lidz,lidx+2)]-v[ind(lidz,lidx-1)])+\
                        HC3*(v[ind(lidz,lidx+3)]-v[ind(lidz,lidx-2)])+\
                        HC4*(v[ind(lidz,lidx+4)]-v[ind(lidz,lidx-3)])+\
                        HC5*(v[ind(lidz,lidx+5)]-v[ind(lidz,lidx-4)]))
#elif FDOH ==6
    #define Dxp(v)  (HC1*(v[ind(lidz,lidx+1)]-v[ind(lidz,lidx)])+\
                        HC2*(v[ind(lidz,lidx+2)]-v[ind(lidz,lidx-1)])+\
                        HC3*(v[ind(lidz,lidx+3)]-v[ind(lidz,lidx-2)])+\
                        HC4*(v[ind(lidz,lidx+4)]-v[ind(lidz,lidx-3)])+\
                        HC5*(v[ind(lidz,lidx+5)]-v[ind(lidz,lidx-4)])+\
                        HC6*(v[ind(lidz,lidx+6)]-v[ind(lidz,lidx-5)]))
#endif

//Backward stencil in x
#if   FDOH ==1
    #define Dxm(v) HC1*(v[ind(lidz,lidx)]   - v[ind(lidz,lidx-1)])
#elif FDOH ==2
    #define Dxm(v) (HC1*(v[ind(lidz,lidx)]   - v[ind(lidz,lidx-1)])\
                      +HC2*(v[ind(lidz,lidx+1)] - v[ind(lidz,lidx-2)]))
#elif FDOH ==3
    #define Dxm(v) (HC1*(v[ind(lidz,lidx)]  -v[ind(lidz,lidx-1)])+\
                       HC2*(v[ind(lidz,lidx+1)]-v[ind(lidz,lidx-2)])+\
                       HC3*(v[ind(lidz,lidx+2)]-v[ind(lidz,lidx-3)]))
#elif FDOH ==4
    #define Dxm(v) (HC1*(v[ind(lidz,lidx)]  -v[ind(lidz,lidx-1)])+\
                       HC2*(v[ind(lidz,lidx+1)]-v[ind(lidz,lidx-2)])+\
                       HC3*(v[ind(lidz,lidx+2)]-v[ind(lidz,lidx-3)])+\
                       HC4*(v[ind(lidz,lidx+3)]-v[ind(lidz,lidx-4)]))
#elif FDOH ==5
    #define Dxm(v) (HC1*(v[ind(lidz,lidx)]  -v[ind(lidz,lidx-1)])+\
                       HC2*(v[ind(lidz,lidx+1)]-v[ind(lidz,lidx-2)])+\
                       HC3*(v[ind(lidz,lidx+2)]-v[ind(lidz,lidx-3)])+\
                       HC4*(v[ind(lidz,lidx+3)]-v[ind(lidz,lidx-4)])+\
                       HC5*(v[ind(lidz,lidx+4)]-v[ind(lidz,lidx-5)]))
#elif FDOH ==6
    #define Dxm(v) (HC1*(v[ind(lidz,lidx)]  -v[ind(lidz,lidx-1)])+\
                       HC2*(v[ind(lidz,lidx+1)]-v[ind(lidz,lidx-2)])+\
                       HC3*(v[ind(lidz,lidx+2)]-v[ind(lidz,lidx-3)])+\
                       HC4*(v[ind(lidz,lidx+3)]-v[ind(lidz,lidx-4)])+\
                       HC5*(v[ind(lidz,lidx+4)]-v[ind(lidz,lidx-5)])+\
                       HC6*(v[ind(lidz,lidx+5)]-v[ind(lidz,lidx-6)]))
#endif

//Forward stencil in x
#if   FDOH ==1
    #define Dzp(v) HC1*(v[ind(lidz+1,lidx)] - v[ind(lidz,lidx)])
#elif FDOH ==2
    #define Dzp(v) (HC1*(v[ind(lidz+1,lidx)] - v[ind(lidz,lidx)])\
                      +HC2*(v[ind(lidz+2,lidx)] - v[ind(lidz-1,lidx)]))
#elif FDOH ==3
    #define Dzp(v) (HC1*(v[ind(lidz+1,lidx)]-v[ind(lidz,lidx)])+\
                       HC2*(v[ind(lidz+2,lidx)]-v[ind(lidz-1,lidx)])+\
                       HC3*(v[ind(lidz+3,lidx)]-v[ind(lidz-2,lidx)]))
#elif FDOH ==4
    #define Dzp(v) (HC1*(v[ind(lidz+1,lidx)]-v[ind(lidz,lidx)])+\
                       HC2*(v[ind(lidz+2,lidx)]-v[ind(lidz-1,lidx)])+\
                       HC3*(v[ind(lidz+3,lidx)]-v[ind(lidz-2,lidx)])+\
                       HC4*(v[ind(lidz+4,lidx)]-v[ind(lidz-3,lidx)]))
#elif FDOH ==5
    #define Dzp(v) (HC1*(v[ind(lidz+1,lidx)]-v[ind(lidz,lidx)])+\
                       HC2*(v[ind(lidz+2,lidx)]-v[ind(lidz-1,lidx)])+\
                       HC3*(v[ind(lidz+3,lidx)]-v[ind(lidz-2,lidx)])+\
                       HC4*(v[ind(lidz+4,lidx)]-v[ind(lidz-3,lidx)])+\
                       HC5*(v[ind(lidz+5,lidx)]-v[ind(lidz-4,lidx)]))
#elif FDOH ==6
    #define Dzp(v) (HC1*(v[ind(lidz+1,lidx)]-v[ind(lidz,lidx)])+\
                       HC2*(v[ind(lidz+2,lidx)]-v[ind(lidz-1,lidx)])+\
                       HC3*(v[ind(lidz+3,lidx)]-v[ind(lidz-2,lidx)])+\
                       HC4*(v[ind(lidz+4,lidx)]-v[ind(lidz-3,lidx)])+\
                       HC5*(v[ind(lidz+5,lidx)]-v[ind(lidz-4,lidx)])+\
                       HC6*(v[ind(lidz+6,lidx)]-v[ind(lidz-5,lidx)]))
#endif

//Backward stencil in z
#if   FDOH ==1
    #define Dzm(v) HC1*(v[ind(lidz,lidx)]   - v[ind(lidz-1,lidx)])
#elif FDOH ==2
    #define Dzm(v) (HC1*(v[ind(lidz,lidx)]   - v[ind(lidz-1,lidx)])\
                      +HC2*(v[ind(lidz+1,lidx)] - v[ind(lidz-2,lidx)]))
#elif FDOH ==3
    #define Dzm(v) (HC1*(v[ind(lidz,lidx)]  -v[ind(lidz-1,lidx)])+\
                       HC2*(v[ind(lidz+1,lidx)]-v[ind(lidz-2,lidx)])+\
                       HC3*(v[ind(lidz+2,lidx)]-v[ind(lidz-3,lidx)]))
#elif FDOH ==4
    #define Dzm(v) (HC1*(v[ind(lidz,lidx)]  -v[ind(lidz-1,lidx)])+\
                       HC2*(v[ind(lidz+1,lidx)]-v[ind(lidz-2,lidx)])+\
                       HC3*(v[ind(lidz+2,lidx)]-v[ind(lidz-3,lidx)])+\
                       HC4*(v[ind(lidz+3,lidx)]-v[ind(lidz-4,lidx)]))
#elif FDOH ==5
    #define Dzm(v) (HC1*(v[ind(lidz,lidx)]  -v[ind(lidz-1,lidx)])+\
                       HC2*(v[ind(lidz+1,lidx)]-v[ind(lidz-2,lidx)])+\
                       HC3*(v[ind(lidz+2,lidx)]-v[ind(lidz-3,lidx)])+\
                       HC4*(v[ind(lidz+3,lidx)]-v[ind(lidz-4,lidx)])+\
                       HC5*(v[ind(lidz+4,lidx)]-v[ind(lidz-5,lidx)]))
#elif FDOH ==6
    #define Dzm(v) (HC1*(v[ind(lidz,lidx)]  -v[ind(lidz-1,lidx)])+\
                       HC2*(v[ind(lidz+1,lidx)]-v[ind(lidz-2,lidx)])+\
                       HC3*(v[ind(lidz+2,lidx)]-v[ind(lidz-3,lidx)])+\
                       HC4*(v[ind(lidz+3,lidx)]-v[ind(lidz-4,lidx)])+\
                       HC5*(v[ind(lidz+4,lidx)]-v[ind(lidz-5,lidx)])+\
                       HC6*(v[ind(lidz+5,lidx)]-v[ind(lidz-6,lidx)]))
#endif

