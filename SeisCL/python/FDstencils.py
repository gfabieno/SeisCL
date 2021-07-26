
class FDCoefficients:
    """-------------------------------------------------------------
    Adapted from SOFI2D
    * Holberg coefficients for a certain FD order and a margin of error E
    * (MAXRELERROR)
    *
    * MAXRELERROR = 0 -> Taylor-coeff.
    * MAXRELERROR = 1 -> Holberg-coeff.: E = 0.1 %
    * MAXRELERROR = 2 ->                 E = 0.5 %
    * MAXRELERROR = 3 ->                 E = 1.0 %
    * MAXRELERROR = 4 ->                 E = 3.0 %
    *
    * hc: column 0 = minimum number of grid points per shortest wavelength
    *     columns 1-6 = Holberg coefficients
    *
    * ------------------------------------------------------------- """

    definitions = [
    [
        [23.0, 1.0],
        [8.0, 9.0/8.0, -1.0/24.0],
        [6.0, 75.0/64.0, -25.0/384.0, 3.0/640.0],
        [5.0, 1225.0/1024.0, -245.0/3072.0, 49.0/5120.0, -5.0/7168.0],
        [5.0, 19845.0/16384.0, -735.0/8192.0, 567.0/40960.0, -405.0/229376.0,
         35.0/294912.0],
        [4.0, 160083.0/131072.0, -12705.0/131072.0, 22869.0/1310720.0,
         -5445.0/1835008.0, 847.0/2359296.0, -63.0/2883584.0]
    ],
    [
        [49.7, 1.0010],
        [8.32, 1.1382, -0.046414],
        [4.77, 1.1965, -0.078804, 0.0081781],
        [3.69, 1.2257, -0.099537, 0.018063,  -0.0026274],
        [3.19, 1.2415, -0.11231,  0.026191,  -0.0064682, 0.001191],
        [2.91, 1.2508, -0.12034,  0.032131,  -0.010142,  0.0029857, -0.00066667]
    ],
    [
        [22.2, 1.0050],
        [5.65, 1.1534, -0.052806],
        [3.74, 1.2111, -0.088313, 0.011768],
        [3.11, 1.2367, -0.10815,  0.023113,  -0.0046905],
        [2.80, 1.2496, -0.11921,  0.031130,  -0.0093272, 0.0025161],
        [2.62, 1.2568, -0.12573,  0.036423,  -0.013132,  0.0047484, -0.0015979]
    ],
    [
        [15.8,  1.0100, ],
        [4.80, 1.1640, -0.057991],
        [3.39, 1.2192, -0.094070, 0.014608],
        [2.90, 1.2422, -0.11269,  0.026140,  -0.0064054],
        [2.65, 1.2534, -0.12257,  0.033755,  -0.011081,  0.0036784],
        [2.51, 1.2596, -0.12825,  0.038550,  -0.014763,  0.0058619, -0.0024538]
    ],
    [
        [9.16, 1.0300],
        [3.47, 1.1876, -0.072518],
        [2.91, 1.2341, -0.10569,  0.022589],
        [2.61, 1.2516, -0.12085,  0.032236,  -0.011459],
        [2.45, 1.2596, -0.12829,  0.038533,  -0.014681,  0.0072580],
        [2.36, 1.2640, -0.13239,  0.042217,  -0.017803,  0.0081959, -0.0051848]
    ],
    ]

    def __init__(self, order=8, maxrerror=1, local_off=0):
        coefs = self.definitions[maxrerror][order//2-1]
        self.grid_per_wavelength = coefs[0]
        self.coefs = coefs[1:]
        self.order = order
        self.local_off = local_off

    def header(self):
        header = "".join(["#define HC%d %f \n" % (ii+1, hc)
                          for ii, hc in enumerate(self.coefs)])
        header += "#define __FDOH__ %d\n" % (self.order//2)
        header += "#define __LOCAL_OFF__ %d\n" % self.local_off
        header += FDstencils
        return header

FDstencils = """
/*Macros for FD difference stencils up to order 12 on GPU in 2D
 
 The macros assume the following variables are defined in the kernel:
 -__FDOH__: Half with of the final difference stencil
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

//Load in local memory with the halo for FD in different directions
#define load_local_in(p, v, lv) lv[ind((p), 0, 0, 0)]=v[indg((p), 0, 0, 0)]

#define load_local_halox(p, v, lv) \
    do{\
        if ((p).lidx<2*__FDOH__)\
            lv[ind((p), 0, 0, -__FDOH__)]=v[indg((p), 0, 0, -__FDOH__)];\
        if ((p).lidx+(p).lsizex-3*__FDOH__<__FDOH__)\
            lv[ind((p), 0, 0, (p).lsizex-3*__FDOH__)]=v[indg((p), 0, 0, (p).lsizex-3*__FDOH__)];\
        if ((p).lidx>((p).lsizex-2*__FDOH__-1))\
            lv[ind((p), 0, 0, __FDOH__)]=v[indg((p), 0, 0, __FDOH__)];\
        if ((p).lidx-(p).lsizex+3*__FDOH__>((p).lsizex-__FDOH__-1))\
            lv[ind((p), 0, 0, -(p).lsizex+3*__FDOH__)]=v[indg((p), 0, 0, -(p).lsizex+3*__FDOH__)];\
        } while(0)

#define load_local_haloy(p, v, lv) \
    do{\
        if ((p).lidy<2*__FDOH__)\
            lv[ind((p), 0, -__FDOH__, 0)]=v[indg((p), 0, -__FDOH__, 0)];\
        if ((p).lidy+(p).lsizey-3*__FDOH__<__FDOH__)\
            lv[ind((p), 0 , (p).lsizey-3*__FDOH__, 0)]=v[indg((p), 0, (p).lsizey-3*__FDOH__, 0)];\
        if ((p).lidy>((p).lsizey-2*__FDOH__-1))\
            lv[ind((p), 0, __FDOH__, 0)]=v[indg((p), 0, __FDOH__, 0)];\
        if ((p).lidy-(p).lsizey+3*__FDOH__>((p).lsizey-__FDOH__-1))\
            lv[ind((p), 0, -(p).lsizey+3*__FDOH__, 0)]=v[indg((p), 0, -(p).lsizey+3*__FDOH__, 0)];\
     } while(0)

#define load_local_haloz(p, v, lv) \
    do{\
        if ((p).lidz<2*__FDOH__)\
            lv[ind((p), -__FDOH__, 0, 0)]=v[indg((p), -__FDOH__, 0, 0)];\
        if ((p).lidz>((p).lsizez-2*__FDOH__-1))\
            lv[ind((p), __FDOH__, 0, 0)]=v[indg((p), __FDOH__, 0, 0)];\
    } while(0)


//Forward stencil in x
#if   __FDOH__ ==1
    #define Dxp(p, v)  HC1*(v[ind((p), 0, 0, 1)]-v[ind((p), 0, 0, 0)])
#elif __FDOH__ ==2
    #define Dxp(p, v)  (HC1*(v[ind((p), 0, 0, 1)]-v[ind((p), 0, 0, 0)])+\
                        HC2*(v[ind((p), 0, 0, 2)]-v[ind((p), 0, 0, -1)]))
#elif __FDOH__ ==3
    #define Dxp(p, v)  (HC1*(v[ind((p), 0, 0, 1)]-v[ind((p), 0, 0, 0)])+\
                        HC2*(v[ind((p), 0, 0, 2)]-v[ind((p), 0, 0, -1)])+\
                        HC3*(v[ind((p), 0, 0, 3)]-v[ind((p), 0, 0, -2)]))
#elif __FDOH__ ==4
    #define Dxp(p, v)  (HC1*(v[ind((p), 0, 0, 1)]-v[ind((p), 0, 0, 0)])+\
                        HC2*(v[ind((p), 0, 0, 2)]-v[ind((p), 0, 0, -1)])+\
                        HC3*(v[ind((p), 0, 0, 3)]-v[ind((p), 0, 0, -2)])+\
                        HC4*(v[ind((p), 0, 0, 4)]-v[ind((p), 0, 0, -3)]))
#elif __FDOH__ ==5
    #define Dxp(p, v)  (HC1*(v[ind((p), 0, 0, 1)]-v[ind((p), 0, 0, 0)])+\
                        HC2*(v[ind((p), 0, 0, 2)]-v[ind((p), 0, 0, -1)])+\
                        HC3*(v[ind((p), 0, 0, 3)]-v[ind((p), 0, 0, -2)])+\
                        HC4*(v[ind((p), 0, 0, 4)]-v[ind((p), 0, 0, -3)])+\
                        HC5*(v[ind((p), 0, 0, 5)]-v[ind((p), 0, 0, -4)]))
#elif __FDOH__ ==6
    #define Dxp(p, v)  (HC1*(v[ind((p), 0, 0, 1)]-v[ind((p), 0, 0, 0)])+\
                        HC2*(v[ind((p), 0, 0, 2)]-v[ind((p), 0, 0, -1)])+\
                        HC3*(v[ind((p), 0, 0, 3)]-v[ind((p), 0, 0, -2)])+\
                        HC4*(v[ind((p), 0, 0, 4)]-v[ind((p), 0, 0, -3)])+\
                        HC5*(v[ind((p), 0, 0, 5)]-v[ind((p), 0, 0, -4)])+\
                        HC6*(v[ind((p), 0, 0, 6)]-v[ind((p), 0, 0, -5)]))
#endif

//Backward stencil in x
#if   __FDOH__ ==1
    #define Dxm(p, v) HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), 0, 0, -1)])
#elif __FDOH__ ==2
    #define Dxm(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), 0, 0, -1)])\
                      +HC2*(v[ind((p), 0, 0, 1)]-v[ind((p), 0, 0, -2)]))
#elif __FDOH__ ==3
    #define Dxm(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), 0, 0, -1)])+\
                       HC2*(v[ind((p), 0, 0, 1)]-v[ind((p), 0, 0, -2)])+\
                       HC3*(v[ind((p), 0, 0, 2)]-v[ind((p), 0, 0, -3)]))
#elif __FDOH__ ==4
    #define Dxm(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), 0, 0, -1)])+\
                       HC2*(v[ind((p), 0, 0, 1)]-v[ind((p), 0, 0, -2)])+\
                       HC3*(v[ind((p), 0, 0, 2)]-v[ind((p), 0, 0, -3)])+\
                       HC4*(v[ind((p), 0, 0, 3)]-v[ind((p), 0, 0, -4)]))
#elif __FDOH__ ==5
    #define Dxm(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), 0, 0, -1)])+\
                       HC2*(v[ind((p), 0, 0, 1)]-v[ind((p), 0, 0, -2)])+\
                       HC3*(v[ind((p), 0, 0, 2)]-v[ind((p), 0, 0, -3)])+\
                       HC4*(v[ind((p), 0, 0, 3)]-v[ind((p), 0, 0, -4)])+\
                       HC5*(v[ind((p), 0, 0, 4)]-v[ind((p), 0, 0, -5)]))
#elif __FDOH__ ==6
    #define Dxm(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), 0, 0, -1)])+\
                       HC2*(v[ind((p), 0, 0, 1)]-v[ind((p), 0, 0, -2)])+\
                       HC3*(v[ind((p), 0, 0, 2)]-v[ind((p), 0, 0, -3)])+\
                       HC4*(v[ind((p), 0, 0, 3)]-v[ind((p), 0, 0, -4)])+\
                       HC5*(v[ind((p), 0, 0, 4)]-v[ind((p), 0, 0, -5)])+\
                       HC6*(v[ind((p), 0, 0, 5)]-v[ind((p), 0, 0, -6)]))
#endif

//Forward stencil in y
#if   __FDOH__ ==1
    #define Dyp(p, v) HC1*(v[ind((p), 0, 1, 0)]-v[ind((p), 0, 0, 0)])
#elif __FDOH__ ==2
    #define Dyp(p, v) (HC1*(v[ind((p), 0, 1, 0)]-v[ind((p), 0, 0, 0)])\
                      +HC2*(v[ind((p), 0, 2, 0)]-v[ind((p), 0, -1, 0)]))
#elif __FDOH__ ==3
    #define Dyp(p, v) (HC1*(v[ind((p), 0, 1, 0)]-v[ind((p), 0, 0, 0)])+\
                       HC2*(v[ind((p), 0, 2, 0)]-v[ind((p), 0, -1, 0)])+\
                       HC3*(v[ind((p), 0, 3, 0)]-v[ind((p), 0, -2, 0)]))
#elif __FDOH__ ==4
    #define Dyp(p, v) (HC1*(v[ind((p), 0, 1, 0)]-v[ind((p), 0, 0, 0)])+\
                       HC2*(v[ind((p), 0, 2, 0)]-v[ind((p), 0, -1, 0)])+\
                       HC3*(v[ind((p), 0, 3, 0)]-v[ind((p), 0, -2, 0)])+\
                       HC4*(v[ind((p), 0, 4, 0)]-v[ind((p), 0, -3, 0)]))
#elif __FDOH__ ==5
    #define Dyp(p, v) (HC1*(v[ind((p), 0, 1, 0)]-v[ind((p), 0, 0, 0)])+\
                       HC2*(v[ind((p), 0, 2, 0)]-v[ind((p), 0, -1, 0)])+\
                       HC3*(v[ind((p), 0, 3, 0)]-v[ind((p), 0, -2, 0)])+\
                       HC4*(v[ind((p), 0, 4, 0)]-v[ind((p), 0, -3, 0)])+\
                       HC5*(v[ind((p), 0, 5, 0)]-v[ind((p), 0, -4, 0)]))
#elif __FDOH__ ==6
    #define Dyp(p, v) (HC1*(v[ind((p), 0, 1, 0)]-v[ind((p), 0, 0, 0)])+\
                       HC2*(v[ind((p), 0, 2, 0)]-v[ind((p), 0, -1, 0)])+\
                       HC3*(v[ind((p), 0, 3, 0)]-v[ind((p), 0, -2, 0)])+\
                       HC4*(v[ind((p), 0, 4, 0)]-v[ind((p), 0, -3, 0)])+\
                       HC5*(v[ind((p), 0, 5, 0)]-v[ind((p), 0, -4, 0)])+\
                       HC6*(v[ind((p), 0, 6, 0)]-v[ind((p), 0, -5, 0)]))
#endif

//Backward stencil in y
#if   __FDOH__ ==1
    #define Dym(p, v) HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), 0, -1, 0)])
#elif __FDOH__ ==2
    #define Dym(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), 0, -1, 0)])\
                      +HC2*(v[ind((p), 0, 1, 0)]-v[ind((p), 0, -2, 0)]))
#elif __FDOH__ ==3
    #define Dym(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), 0, -1, 0)])+\
                       HC2*(v[ind((p), 0, 1, 0)]-v[ind((p), 0, -2, 0)])+\
                       HC3*(v[ind((p), 0, 2, 0)]-v[ind((p), 0, -3, 0)]))
#elif __FDOH__ ==4
    #define Dym(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), 0, -1, 0)])+\
                       HC2*(v[ind((p), 0, 1, 0)]-v[ind((p), 0, -2, 0)])+\
                       HC3*(v[ind((p), 0, 2, 0)]-v[ind((p), 0, -3, 0)])+\
                       HC4*(v[ind((p), 0, 3, 0)]-v[ind((p), 0, -4, 0)]))
#elif __FDOH__ ==5
    #define Dym(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), 0, -1, 0)])+\
                       HC2*(v[ind((p), 0, 1, 0)]-v[ind((p), 0, -2, 0)])+\
                       HC3*(v[ind((p), 0, 2, 0)]-v[ind((p), 0, -3, 0)])+\
                       HC4*(v[ind((p), 0, 3, 0)]-v[ind((p), 0, -4, 0)])+\
                       HC5*(v[ind((p), 0, 4, 0)]-v[ind((p), 0, -5, 0)]))
#elif __FDOH__ ==6
    #define Dym(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), 0, -1, 0)])+\
                       HC2*(v[ind((p), 0, 1, 0)]-v[ind((p), 0, -2, 0)])+\
                       HC3*(v[ind((p), 0, 2, 0)]-v[ind((p), 0, -3, 0)])+\
                       HC4*(v[ind((p), 0, 3, 0)]-v[ind((p), 0, -4, 0)])+\
                       HC5*(v[ind((p), 0, 4, 0)]-v[ind((p), 0, -5, 0)])+\
                       HC6*(v[ind((p), 0, 5, 0)]-v[ind((p), 0, -6, 0)]))
#endif

//Forward stencil in z
#if   __FDOH__ ==1
    #define Dzp(p, v) HC1*(v[ind((p), 1, 0, 0)] - v[ind((p), 0, 0, 0)])
#elif __FDOH__ ==2
    #define Dzp(p, v) (HC1*(v[ind((p), 1, 0, 0)]-v[ind((p), 0, 0, 0)])\
                      +HC2*(v[ind((p), 2, 0, 0)]-v[ind((p), -1, 0, 0)]))
#elif __FDOH__ ==3
    #define Dzp(p, v) (HC1*(v[ind((p), 1, 0, 0)]-v[ind((p), 0, 0, 0)])+\
                       HC2*(v[ind((p), 2, 0, 0)]-v[ind((p), -1, 0, 0)])+\
                       HC3*(v[ind((p), 3, 0, 0)]-v[ind((p), -2, 0, 0)]))
#elif __FDOH__ ==4
    #define Dzp(p, v) (HC1*(v[ind((p), 1, 0, 0)]-v[ind((p), 0, 0, 0)])+\
                       HC2*(v[ind((p), 2, 0, 0)]-v[ind((p), -1, 0, 0)])+\
                       HC3*(v[ind((p), 3, 0, 0)]-v[ind((p), -2, 0, 0)])+\
                       HC4*(v[ind((p), 4, 0, 0)]-v[ind((p), -3, 0, 0)]))
#elif __FDOH__ ==5
    #define Dzp(p, v) (HC1*(v[ind((p), 1, 0, 0)]-v[ind((p), 0, 0, 0)])+\
                       HC2*(v[ind((p), 2, 0, 0)]-v[ind((p), -1, 0, 0)])+\
                       HC3*(v[ind((p), 3, 0, 0)]-v[ind((p), -2, 0, 0)])+\
                       HC4*(v[ind((p), 4, 0, 0)]-v[ind((p), -3, 0, 0)])+\
                       HC5*(v[ind((p), 5, 0, 0)]-v[ind((p), -4, 0, 0)]))
#elif __FDOH__ ==6
    #define Dzp(p, v) (HC1*(v[ind((p), 1, 0, 0)]-v[ind((p), 0, 0, 0)])+\
                       HC2*(v[ind((p), 2, 0, 0)]-v[ind((p), -1, 0, 0)])+\
                       HC3*(v[ind((p), 3, 0, 0)]-v[ind((p), -2, 0, 0)])+\
                       HC4*(v[ind((p), 4, 0, 0)]-v[ind((p), -3, 0, 0)])+\
                       HC5*(v[ind((p), 5, 0, 0)]-v[ind((p), -4, 0, 0)])+\
                       HC6*(v[ind((p), 6, 0, 0)]-v[ind((p), -5, 0, 0)]))
#endif

//Backward stencil in z
#if   __FDOH__ ==1
    #define Dzm(p, v) HC1*(v[ind((p), 0, 0, 0)]- v[ind((p), -1, 0, 0)])
#elif __FDOH__ ==2
    #define Dzm(p, v) (HC1*(v[ind((p), 0, 0, 0)]- v[ind((p), -1, 0, 0)])\
                      +HC2*(v[ind((p), 1, 0, 0)]- v[ind((p), -2, 0, 0)]))
#elif __FDOH__ ==3
    #define Dzm(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), -1, 0, 0)])+\
                       HC2*(v[ind((p), 1, 0, 0)]-v[ind((p), -2, 0, 0)])+\
                       HC3*(v[ind((p), 2, 0, 0)]-v[ind((p), -3, 0, 0)]))
#elif __FDOH__ ==4
    #define Dzm(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), -1, 0, 0)])+\
                       HC2*(v[ind((p), 1, 0, 0)]-v[ind((p), -2, 0, 0)])+\
                       HC3*(v[ind((p), 2, 0, 0)]-v[ind((p), -3, 0, 0)])+\
                       HC4*(v[ind((p), 3, 0, 0)]-v[ind((p), -4, 0, 0)]))
#elif __FDOH__ ==5
    #define Dzm(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), -1, 0, 0)])+\
                       HC2*(v[ind((p), 1, 0, 0)]-v[ind((p), -2, 0, 0)])+\
                       HC3*(v[ind((p), 2, 0, 0)]-v[ind((p), -3, 0, 0)])+\
                       HC4*(v[ind((p), 3, 0, 0)]-v[ind((p), -4, 0, 0)])+\
                       HC5*(v[ind((p), 4, 0, 0)]-v[ind((p), -5, 0, 0)]))
#elif __FDOH__ ==6
    #define Dzm(p, v) (HC1*(v[ind((p), 0, 0, 0)]-v[ind((p), -1, 0, 0)])+\
                       HC2*(v[ind((p), 1, 0, 0)]-v[ind((p), -2, 0, 0)])+\
                       HC3*(v[ind((p), 2, 0, 0)]-v[ind((p), -3, 0, 0)])+\
                       HC4*(v[ind((p), 3, 0, 0)]-v[ind((p), -4, 0, 0)])+\
                       HC5*(v[ind((p), 4, 0, 0)]-v[ind((p), -5, 0, 0)])+\
                       HC6*(v[ind((p), 5, 0, 0)]-v[ind((p), -6, 0, 0)]))
#endif
"""

CUDACL_header = """
/*Macros for writing kernels compatible with CUDA and OpenCL */

#ifdef __OPENCL_VERSION__
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    #define FUNDEF __kernel
    #define LFUNDEF
    #define GLOBARG __global
    #define LOCARG __local float *lvar
    #define LOCARG2 __local __prec2 *lvar2
    #define LOCID __local
    #define LOCDEF
    #define BARRIER barrier(CLK_LOCAL_MEM_FENCE);
#else
    #define FUNDEF extern "C" __global__
    #define LFUNDEF __device__ __inline__
    #define GLOBARG
    #define LOCARG float *nullarg
    #define LOCARG2 __prec2 *nullarg
    #define LOCDEF extern __shared__ float lvar[];
    #define LOCID
    #define BARRIER __syncthreads();
#endif
"""
get_pos_header = """

#if __ND__==3
    #define indg(p,dz,dy,dx)  ((p).gidx+(dx))*(p).NY*((p).NZ)\
                              +((p).gidy+(dy))*((p).NZ)\
                              +(p).gidz+(dz)
    #if __LOCAL_OFF__==0
        #define ind(p,dz,dy,dx)   ((p).lidx+(dx))*(p).lsizey*(p).lsizez\
                                 +((p).lidy+(dy))*(p).lsizez\
                                 +(p).lidz+(dz)
    #else
        #define ind(p,dz,dy,dx)   ((p).gidx+(dx))*(p).NY*((p).NZ)\
                                 +((p).gidy+(dy))*((p).NZ)\
                                 +(p).gidz+(dz)
    #endif
#else
    #define indg(p,dz,dy,dx)  ((p).gidx+(dx))*((p).NZ)+(p).gidz+(dz)
    #if __LOCAL_OFF__==0
        #define ind(p,dz,dy,dx)   ((p).lidx+(dx))*(p).lsizez+(p).lidz+(dz)
    #else
        #define ind(p,dz,dy,dx)  ((p).gidx+(dx))*((p).NZ)+(p).gidz+(dz)
    #endif
#endif

void get_pos(grid * g);
void get_pos(grid * g){

#if __LOCAL_OFF__==0
#ifdef __OPENCL_VERSION__
    g->lsizez = get_local_size(0)+2*__FDOH__;
    g->lsizex = get_local_size(1)+2*__FDOH__;
    g->lidz = get_local_id(0)+__FDOH__;
    g->lidx = get_local_id(1)+__FDOH__;
    g->gidz = get_global_id(0)+__FDOH__;
    g->gidx = get_global_id(1)+__FDOH__+g->offset;
#else
    g->lsizez = blockDim.x+2*__FDOH__;
    g->lsizex = blockDim.y+2*__FDOH__;
    g->lidz = threadIdx.x+__FDOH__;
    g->lidx = threadIdx.y+__FDOH__;
    g->gidz = blockIdx.x*blockDim.x + threadIdx.x+__FDOH__;
    g->gidx = blockIdx.y*blockDim.y + threadIdx.y+__FDOH__+g->offset;
#endif
    
// If local memory is turned off
#elif __LOCAL_OFF__==1
    
#ifdef __OPENCL_VERSION__
    g->gid = get_global_id(0);
    g->glsizez = (g.NZ-2*__FDOH__);
    g->gidz = g->gid%g->lsizez+__FDOH__;
    g->gidx = (g->gid/g->lsizez)+__FDOH__+g->offset;
#else
    g->lsizez = blockDim.x+2*__FDOH__;
    g->lsizex = blockDim.y+2*__FDOH__;
    g->lidz = threadIdx.x+__FDOH__;
    g->lidx = threadIdx.y+__FDOH__;
    g->gidz = blockIdx.x*blockDim.x + threadIdx.x+__FDOH__;
    g->gidx = blockIdx.y*blockDim.y + threadIdx.y+__FDOH__+offset;
#endif
#endif
} 

"""



