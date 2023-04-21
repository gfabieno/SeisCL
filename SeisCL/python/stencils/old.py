#Replaced in stencils.headers
FDstencils = """
/*Macros for FD difference stencils up to order 12 on GPU in 2D
 
 The macros assume the following variables are defined in the kernel:
 -FDOH: Half with of the final difference stencil
 -__LOCAL_OFF__: If 0, uses local memory grid, else uses global memory grid
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


#if __ND__==3
    #define ig(p,dz,dy,dx)  ((p).gx+(dx))*(p).NY*((p).NZ)\
                              +((p).gy+(dy))*((p).NZ)\
                              +(p).gz+(dz)
    #if __LOCAL_OFF__==0
        #define il(p,dz,dy,dx)   ((p).lx+(dx))*(p).nly*(p).nlz\
                                 +((p).ly+(dy))*(p).nlz\
                                 +(p).lz+(dz)
    #else
        #define il(p,dz,dy,dx)   ((p).gx+(dx))*(p).NY*((p).NZ)\
                                 +((p).gy+(dy))*((p).NZ)\
                                 +(p).gz+(dz)
    #endif
#else
    #define ig(p,dz,dy,dx) ((p).gx+(dx))*((p).NZ)+(p).gz+(dz)
    #if __LOCAL_OFF__==0
        #define il(p,dz,dy,dx)   ((p).lx+(dx))*(p).nlz+(p).lz+(dz)
    #else
        #define il(p,dz,dy,dx)  ((p).gx+(dx))*((p).NZ)+(p).gz+(dz)
    #endif
#endif

//Load in local memory with the halo for FD in different directions
#define load_local_in(p, v, lv) lv[il((p), 0, 0, 0)]=v[ig((p), 0, 0, 0)]
#define mul_local_in(p, v, lv) lv[il((p), 0, 0, 0)]*=v[ig((p), 0, 0, 0)]

#define load_local_halox(p, v, lv) \
    do{\
        if ((p).lx<2*FDOH)\
            lv[il((p), 0, 0, -FDOH)]=v[ig((p), 0, 0, -FDOH)];\
        if ((p).lx+(p).nlx-3*FDOH<FDOH)\
            lv[il((p), 0, 0, (p).nlx-3*FDOH)]=v[ig((p), 0, 0, (p).nlx-3*FDOH)];\
        if ((p).lx>((p).nlx-2*FDOH-1))\
            lv[il((p), 0, 0, FDOH)]=v[ig((p), 0, 0, FDOH)];\
        if ((p).lx-(p).nlx+3*FDOH>((p).nlx-FDOH-1))\
            lv[il((p), 0, 0, -(p).nlx+3*FDOH)]=v[ig((p), 0, 0, -(p).nlx+3*FDOH)];\
        } while(0)

#define mul_local_halox(p, v, lv) \
    do{\
        if ((p).lx<2*FDOH)\
            lv[il((p), 0, 0, -FDOH)]*=v[ig((p), 0, 0, -FDOH)];\
        if ((p).lx+(p).nlx-3*FDOH<FDOH)\
            lv[il((p), 0, 0, (p).nlx-3*FDOH)]*=v[ig((p), 0, 0, (p).nlx-3*FDOH)];\
        if ((p).lx>((p).nlx-2*FDOH-1))\
            lv[il((p), 0, 0, FDOH)]*=v[ig((p), 0, 0, FDOH)];\
        if ((p).lx-(p).nlx+3*FDOH>((p).nlx-FDOH-1))\
            lv[il((p), 0, 0, -(p).nlx+3*FDOH)]*=v[ig((p), 0, 0, -(p).nlx+3*FDOH)];\
        } while(0)
        
#define load_local_haloy(p, v, lv) \
    do{\
        if ((p).ly<2*FDOH)\
            lv[il((p), 0, -FDOH, 0)]=v[ig((p), 0, -FDOH, 0)];\
        if ((p).ly+(p).nly-3*FDOH<FDOH)\
            lv[il((p), 0 , (p).nly-3*FDOH, 0)]=v[ig((p), 0, (p).nly-3*FDOH, 0)];\
        if ((p).ly>((p).nly-2*FDOH-1))\
            lv[il((p), 0, FDOH, 0)]=v[ig((p), 0, FDOH, 0)];\
        if ((p).ly-(p).nly+3*FDOH>((p).nly-FDOH-1))\
            lv[il((p), 0, -(p).nly+3*FDOH, 0)]=v[ig((p), 0, -(p).nly+3*FDOH, 0)];\
     } while(0)

#define mul_local_haloy(p, v, lv) \
    do{\
        if ((p).ly<2*FDOH)\
            lv[il((p), 0, -FDOH, 0)]*=v[ig((p), 0, -FDOH, 0)];\
        if ((p).ly+(p).nly-3*FDOH<FDOH)\
            lv[il((p), 0 , (p).nly-3*FDOH, 0)]*=v[ig((p), 0, (p).nly-3*FDOH, 0)];\
        if ((p).ly>((p).nly-2*FDOH-1))\
            lv[il((p), 0, FDOH, 0)]*=v[ig((p), 0, FDOH, 0)];\
        if ((p).ly-(p).nly+3*FDOH>((p).nly-FDOH-1))\
            lv[il((p), 0, -(p).nly+3*FDOH, 0)]*=v[ig((p), 0, -(p).nly+3*FDOH, 0)];\
     } while(0)
     
#define load_local_haloz(p, v, lv) \
    do{\
        if ((p).lz<2*FDOH)\
            lv[il((p), -FDOH, 0, 0)]=v[ig((p), -FDOH, 0, 0)];\
        if ((p).lz>((p).nlz-2*FDOH-1))\
            lv[il((p), FDOH, 0, 0)]=v[ig((p), FDOH, 0, 0)];\
    } while(0)
    
#define mul_local_haloz(p, v, lv) \
    do{\
        if ((p).lz<2*FDOH)\
            lv[il((p), -FDOH, 0, 0)]*=v[ig((p), -FDOH, 0, 0)];\
        if ((p).lz>((p).nlz-2*FDOH-1))\
            lv[il((p), FDOH, 0, 0)]*=v[ig((p), FDOH, 0, 0)];\
    } while(0)


//Forward stencil in x
#if   FDOH ==1
    #define Dxp(p, v)  HC1*(v[il((p), 0, 0, 1)]-v[il((p), 0, 0, 0)])
#elif FDOH ==2
    #define Dxp(p, v)  (HC1*(v[il((p), 0, 0, 1)]-v[il((p), 0, 0, 0)])+\
                        HC2*(v[il((p), 0, 0, 2)]-v[il((p), 0, 0, -1)]))
#elif FDOH ==3
    #define Dxp(p, v)  (HC1*(v[il((p), 0, 0, 1)]-v[il((p), 0, 0, 0)])+\
                        HC2*(v[il((p), 0, 0, 2)]-v[il((p), 0, 0, -1)])+\
                        HC3*(v[il((p), 0, 0, 3)]-v[il((p), 0, 0, -2)]))
#elif FDOH ==4
    #define Dxp(p, v)  (HC1*(v[il((p), 0, 0, 1)]-v[il((p), 0, 0, 0)])+\
                        HC2*(v[il((p), 0, 0, 2)]-v[il((p), 0, 0, -1)])+\
                        HC3*(v[il((p), 0, 0, 3)]-v[il((p), 0, 0, -2)])+\
                        HC4*(v[il((p), 0, 0, 4)]-v[il((p), 0, 0, -3)]))
#elif FDOH ==5
    #define Dxp(p, v)  (HC1*(v[il((p), 0, 0, 1)]-v[il((p), 0, 0, 0)])+\
                        HC2*(v[il((p), 0, 0, 2)]-v[il((p), 0, 0, -1)])+\
                        HC3*(v[il((p), 0, 0, 3)]-v[il((p), 0, 0, -2)])+\
                        HC4*(v[il((p), 0, 0, 4)]-v[il((p), 0, 0, -3)])+\
                        HC5*(v[il((p), 0, 0, 5)]-v[il((p), 0, 0, -4)]))
#elif FDOH ==6
    #define Dxp(p, v)  (HC1*(v[il((p), 0, 0, 1)]-v[il((p), 0, 0, 0)])+\
                        HC2*(v[il((p), 0, 0, 2)]-v[il((p), 0, 0, -1)])+\
                        HC3*(v[il((p), 0, 0, 3)]-v[il((p), 0, 0, -2)])+\
                        HC4*(v[il((p), 0, 0, 4)]-v[il((p), 0, 0, -3)])+\
                        HC5*(v[il((p), 0, 0, 5)]-v[il((p), 0, 0, -4)])+\
                        HC6*(v[il((p), 0, 0, 6)]-v[il((p), 0, 0, -5)]))
#endif

//Backward stencil in x
#if   FDOH ==1
    #define Dxm(p, v) HC1*(v[il((p), 0, 0, 0)]-v[il((p), 0, 0, -1)])
#elif FDOH ==2
    #define Dxm(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), 0, 0, -1)])\
                      +HC2*(v[il((p), 0, 0, 1)]-v[il((p), 0, 0, -2)]))
#elif FDOH ==3
    #define Dxm(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), 0, 0, -1)])+\
                       HC2*(v[il((p), 0, 0, 1)]-v[il((p), 0, 0, -2)])+\
                       HC3*(v[il((p), 0, 0, 2)]-v[il((p), 0, 0, -3)]))
#elif FDOH ==4
    #define Dxm(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), 0, 0, -1)])+\
                       HC2*(v[il((p), 0, 0, 1)]-v[il((p), 0, 0, -2)])+\
                       HC3*(v[il((p), 0, 0, 2)]-v[il((p), 0, 0, -3)])+\
                       HC4*(v[il((p), 0, 0, 3)]-v[il((p), 0, 0, -4)]))
#elif FDOH ==5
    #define Dxm(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), 0, 0, -1)])+\
                       HC2*(v[il((p), 0, 0, 1)]-v[il((p), 0, 0, -2)])+\
                       HC3*(v[il((p), 0, 0, 2)]-v[il((p), 0, 0, -3)])+\
                       HC4*(v[il((p), 0, 0, 3)]-v[il((p), 0, 0, -4)])+\
                       HC5*(v[il((p), 0, 0, 4)]-v[il((p), 0, 0, -5)]))
#elif FDOH ==6
    #define Dxm(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), 0, 0, -1)])+\
                       HC2*(v[il((p), 0, 0, 1)]-v[il((p), 0, 0, -2)])+\
                       HC3*(v[il((p), 0, 0, 2)]-v[il((p), 0, 0, -3)])+\
                       HC4*(v[il((p), 0, 0, 3)]-v[il((p), 0, 0, -4)])+\
                       HC5*(v[il((p), 0, 0, 4)]-v[il((p), 0, 0, -5)])+\
                       HC6*(v[il((p), 0, 0, 5)]-v[il((p), 0, 0, -6)]))
#endif

//Forward stencil in y
#if   FDOH ==1
    #define Dyp(p, v) HC1*(v[il((p), 0, 1, 0)]-v[il((p), 0, 0, 0)])
#elif FDOH ==2
    #define Dyp(p, v) (HC1*(v[il((p), 0, 1, 0)]-v[il((p), 0, 0, 0)])\
                      +HC2*(v[il((p), 0, 2, 0)]-v[il((p), 0, -1, 0)]))
#elif FDOH ==3
    #define Dyp(p, v) (HC1*(v[il((p), 0, 1, 0)]-v[il((p), 0, 0, 0)])+\
                       HC2*(v[il((p), 0, 2, 0)]-v[il((p), 0, -1, 0)])+\
                       HC3*(v[il((p), 0, 3, 0)]-v[il((p), 0, -2, 0)]))
#elif FDOH ==4
    #define Dyp(p, v) (HC1*(v[il((p), 0, 1, 0)]-v[il((p), 0, 0, 0)])+\
                       HC2*(v[il((p), 0, 2, 0)]-v[il((p), 0, -1, 0)])+\
                       HC3*(v[il((p), 0, 3, 0)]-v[il((p), 0, -2, 0)])+\
                       HC4*(v[il((p), 0, 4, 0)]-v[il((p), 0, -3, 0)]))
#elif FDOH ==5
    #define Dyp(p, v) (HC1*(v[il((p), 0, 1, 0)]-v[il((p), 0, 0, 0)])+\
                       HC2*(v[il((p), 0, 2, 0)]-v[il((p), 0, -1, 0)])+\
                       HC3*(v[il((p), 0, 3, 0)]-v[il((p), 0, -2, 0)])+\
                       HC4*(v[il((p), 0, 4, 0)]-v[il((p), 0, -3, 0)])+\
                       HC5*(v[il((p), 0, 5, 0)]-v[il((p), 0, -4, 0)]))
#elif FDOH ==6
    #define Dyp(p, v) (HC1*(v[il((p), 0, 1, 0)]-v[il((p), 0, 0, 0)])+\
                       HC2*(v[il((p), 0, 2, 0)]-v[il((p), 0, -1, 0)])+\
                       HC3*(v[il((p), 0, 3, 0)]-v[il((p), 0, -2, 0)])+\
                       HC4*(v[il((p), 0, 4, 0)]-v[il((p), 0, -3, 0)])+\
                       HC5*(v[il((p), 0, 5, 0)]-v[il((p), 0, -4, 0)])+\
                       HC6*(v[il((p), 0, 6, 0)]-v[il((p), 0, -5, 0)]))
#endif

//Backward stencil in y
#if   FDOH ==1
    #define Dym(p, v) HC1*(v[il((p), 0, 0, 0)]-v[il((p), 0, -1, 0)])
#elif FDOH ==2
    #define Dym(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), 0, -1, 0)])\
                      +HC2*(v[il((p), 0, 1, 0)]-v[il((p), 0, -2, 0)]))
#elif FDOH ==3
    #define Dym(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), 0, -1, 0)])+\
                       HC2*(v[il((p), 0, 1, 0)]-v[il((p), 0, -2, 0)])+\
                       HC3*(v[il((p), 0, 2, 0)]-v[il((p), 0, -3, 0)]))
#elif FDOH ==4
    #define Dym(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), 0, -1, 0)])+\
                       HC2*(v[il((p), 0, 1, 0)]-v[il((p), 0, -2, 0)])+\
                       HC3*(v[il((p), 0, 2, 0)]-v[il((p), 0, -3, 0)])+\
                       HC4*(v[il((p), 0, 3, 0)]-v[il((p), 0, -4, 0)]))
#elif FDOH ==5
    #define Dym(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), 0, -1, 0)])+\
                       HC2*(v[il((p), 0, 1, 0)]-v[il((p), 0, -2, 0)])+\
                       HC3*(v[il((p), 0, 2, 0)]-v[il((p), 0, -3, 0)])+\
                       HC4*(v[il((p), 0, 3, 0)]-v[il((p), 0, -4, 0)])+\
                       HC5*(v[il((p), 0, 4, 0)]-v[il((p), 0, -5, 0)]))
#elif FDOH ==6
    #define Dym(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), 0, -1, 0)])+\
                       HC2*(v[il((p), 0, 1, 0)]-v[il((p), 0, -2, 0)])+\
                       HC3*(v[il((p), 0, 2, 0)]-v[il((p), 0, -3, 0)])+\
                       HC4*(v[il((p), 0, 3, 0)]-v[il((p), 0, -4, 0)])+\
                       HC5*(v[il((p), 0, 4, 0)]-v[il((p), 0, -5, 0)])+\
                       HC6*(v[il((p), 0, 5, 0)]-v[il((p), 0, -6, 0)]))
#endif

//Forward stencil in z
#if   FDOH ==1
    #define Dzp(p, v) HC1*(v[il((p), 1, 0, 0)] - v[il((p), 0, 0, 0)])
#elif FDOH ==2
    #define Dzp(p, v) (HC1*(v[il((p), 1, 0, 0)]-v[il((p), 0, 0, 0)])\
                      +HC2*(v[il((p), 2, 0, 0)]-v[il((p), -1, 0, 0)]))
#elif FDOH ==3
    #define Dzp(p, v) (HC1*(v[il((p), 1, 0, 0)]-v[il((p), 0, 0, 0)])+\
                       HC2*(v[il((p), 2, 0, 0)]-v[il((p), -1, 0, 0)])+\
                       HC3*(v[il((p), 3, 0, 0)]-v[il((p), -2, 0, 0)]))
#elif FDOH ==4
    #define Dzp(p, v) (HC1*(v[il((p), 1, 0, 0)]-v[il((p), 0, 0, 0)])+\
                       HC2*(v[il((p), 2, 0, 0)]-v[il((p), -1, 0, 0)])+\
                       HC3*(v[il((p), 3, 0, 0)]-v[il((p), -2, 0, 0)])+\
                       HC4*(v[il((p), 4, 0, 0)]-v[il((p), -3, 0, 0)]))
#elif FDOH ==5
    #define Dzp(p, v) (HC1*(v[il((p), 1, 0, 0)]-v[il((p), 0, 0, 0)])+\
                       HC2*(v[il((p), 2, 0, 0)]-v[il((p), -1, 0, 0)])+\
                       HC3*(v[il((p), 3, 0, 0)]-v[il((p), -2, 0, 0)])+\
                       HC4*(v[il((p), 4, 0, 0)]-v[il((p), -3, 0, 0)])+\
                       HC5*(v[il((p), 5, 0, 0)]-v[il((p), -4, 0, 0)]))
#elif FDOH ==6
    #define Dzp(p, v) (HC1*(v[il((p), 1, 0, 0)]-v[il((p), 0, 0, 0)])+\
                       HC2*(v[il((p), 2, 0, 0)]-v[il((p), -1, 0, 0)])+\
                       HC3*(v[il((p), 3, 0, 0)]-v[il((p), -2, 0, 0)])+\
                       HC4*(v[il((p), 4, 0, 0)]-v[il((p), -3, 0, 0)])+\
                       HC5*(v[il((p), 5, 0, 0)]-v[il((p), -4, 0, 0)])+\
                       HC6*(v[il((p), 6, 0, 0)]-v[il((p), -5, 0, 0)]))
#endif

//Backward stencil in z
#if   FDOH ==1
    #define Dzm(p, v) HC1*(v[il((p), 0, 0, 0)]- v[il((p), -1, 0, 0)])
#elif FDOH ==2
    #define Dzm(p, v) (HC1*(v[il((p), 0, 0, 0)]- v[il((p), -1, 0, 0)])\
                      +HC2*(v[il((p), 1, 0, 0)]- v[il((p), -2, 0, 0)]))
#elif FDOH ==3
    #define Dzm(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), -1, 0, 0)])+\
                       HC2*(v[il((p), 1, 0, 0)]-v[il((p), -2, 0, 0)])+\
                       HC3*(v[il((p), 2, 0, 0)]-v[il((p), -3, 0, 0)]))
#elif FDOH ==4
    #define Dzm(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), -1, 0, 0)])+\
                       HC2*(v[il((p), 1, 0, 0)]-v[il((p), -2, 0, 0)])+\
                       HC3*(v[il((p), 2, 0, 0)]-v[il((p), -3, 0, 0)])+\
                       HC4*(v[il((p), 3, 0, 0)]-v[il((p), -4, 0, 0)]))
#elif FDOH ==5
    #define Dzm(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), -1, 0, 0)])+\
                       HC2*(v[il((p), 1, 0, 0)]-v[il((p), -2, 0, 0)])+\
                       HC3*(v[il((p), 2, 0, 0)]-v[il((p), -3, 0, 0)])+\
                       HC4*(v[il((p), 3, 0, 0)]-v[il((p), -4, 0, 0)])+\
                       HC5*(v[il((p), 4, 0, 0)]-v[il((p), -5, 0, 0)]))
#elif FDOH ==6
    #define Dzm(p, v) (HC1*(v[il((p), 0, 0, 0)]-v[il((p), -1, 0, 0)])+\
                       HC2*(v[il((p), 1, 0, 0)]-v[il((p), -2, 0, 0)])+\
                       HC3*(v[il((p), 2, 0, 0)]-v[il((p), -3, 0, 0)])+\
                       HC4*(v[il((p), 3, 0, 0)]-v[il((p), -4, 0, 0)])+\
                       HC5*(v[il((p), 4, 0, 0)]-v[il((p), -5, 0, 0)])+\
                       HC6*(v[il((p), 5, 0, 0)]-v[il((p), -6, 0, 0)]))
#endif
"""

#Replaced in function/kernel.py
positional_header = """
void get_pos(grid * g);
void get_pos(grid * g){

#if __ND__==3
    #ifdef __OPENCL_VERSION__
        g->nlz = get_local_size(0)+2*FDOH;
        g->nly = get_local_size(1)+2*FDOH;
        g->nlx = get_local_size(2)+2*FDOH;
        g->lz = get_local_id(0)+FDOH;
        g->ly = get_local_id(1)+FDOH;
        g->lx = get_local_id(2)+FDOH;
        g->gz = get_global_id(0)+FDOH;
        g->gy = get_global_id(1)+FDOH;
        g->gx = get_global_id(2)+FDOH; //+g->offset;
    #else
        g->nlz = blockDim.x+2*FDOH;
        g->nly = blockDim.y+2*FDOH;
        g->nlx = blockDim.z+2*FDOH;
        g->lz = threadIdx.x+FDOH;
        g->ly = threadIdx.y+FDOH;
        g->lx = threadIdx.z+FDOH;
        g->gz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
        g->gy = blockIdx.y*blockDim.y + threadIdx.y+FDOH;
        g->gx = blockIdx.y*blockDim.z + threadIdx.z+FDOH; //+g->offset;
    #endif
#else
    #ifdef __OPENCL_VERSION__
        g->nlz = get_local_size(0)+2*FDOH;
        g->nlx = get_local_size(1)+2*FDOH;
        g->lz = get_local_id(0)+FDOH;
        g->lx = get_local_id(1)+FDOH;
        g->gz = get_global_id(0)+FDOH;
        g->gx = get_global_id(1)+FDOH+g->offset;
    #else
        g->nlz = blockDim.x+2*FDOH;
        g->nlx = blockDim.y+2*FDOH;
        g->lz = threadIdx.x+FDOH;
        g->lx = threadIdx.y+FDOH;
        g->gz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
        g->gx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+g->offset;
    #endif
#endif    
} 
"""

#Replaced in function/kernel.py
CUDACL_header = """
/*Macros for writing kernels compatible with CUDA and OpenCL */

#ifdef __OPENCL_VERSION__
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    #define FUNDEF __kernel
    #define LFUNDEF
    #define GLOBARG __global
    #define LOCID __local
    #define BARRIER barrier(CLK_LOCAL_MEM_FENCE);
#else
    #define FUNDEF extern "C" __global__
    #define LFUNDEF __device__ __inline__
    #define GLOBARG
    #define LOCID __shared__
    #define BARRIER __syncthreads();
#endif
"""


grid_stop_header="""
#if LOCAL_OFF==0
    #if COMM12==0
    #define gridstop(p)\
    do{\
            if ((p).gz>((p).NZ-FDOH-1) || ((p).gx-(p).offset)>((p).NX-FDOH-1-LCOMM) ){\
        return;}\
        } while(0)
    #else
    #define gridstop(p)\
    do{\
        if ((p).gz>((p).NZ-FDOH-1) ){\
            return;}\
        } while(0)
    #endif
#else
    #define gridstop(p) 
#endif
"""