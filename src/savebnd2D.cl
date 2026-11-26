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

/*Kernels to save boundary wavefield in 2D if backpropagation is used in the computation of the gradient */

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#if FP16==0

#define __prec float
#define __prec2 float
#define DIV 1

#elif FP16==1

#define __prec float
#define __prec2 float2
#define DIV 2

#else

#define __prec half
#define __prec2 half2
#define DIV 2

#endif

#ifndef GRADABS
#define GRADABS 0
#endif
/* Frame of stored forward values (Init_OpenCL.c, NBND), one layout for every case:
   the cells between an outer box (FR_OL cells inside the grid edge; FR_OT on top)
   and an inner box FR_W cells further in (FR_WT on top), stored as scalar cells
   (FP16 1 and 2 included).
   GRADABS 1: the whole absorbing layers + the FDOH band (FR_OL = 0).
   GRADABS 0: the FDOH band + the last layer cell (FR_OL = NAB-1): the gradient
   of the averaged parameters (rip, rkp, muipkp, ...) stored at that cell
   straddles the layer and the band, and the averaging transpose spreads it
   into the first band cell; its forward field must be exact.
   Free surface (FREESURF != 0): no layer on top, FDOH rows from the surface.
   Domain decomposition along x: each subdomain stores the global frame clipped
   to its own x range. Only the first subdomain has the low x face (FR_OX0,
   FR_WX0) and only the last one the high x face (FR_OX1, FR_WX1); the z (and
   y) faces span the whole local x interior of a middle subdomain. */
#ifndef NZS
#define NZS (NZ*DIV)
#endif
#ifndef FR_OL
#if GRADABS==1
#define FR_OL 0
#else
#define FR_OL ((NAB)>0 ? (NAB)-1 : 0)
#endif
#define FR_W (NAB+FDOH-FR_OL)
#if FREESURF==0
#define FR_OT FR_OL
#define FR_WT FR_W
#else
#define FR_OT 0
#define FR_WT FDOH
#endif
#define FR_X_FIRST (DEVID==0 && MYLOCALID==0)
#define FR_X_LAST (DEVID==NUM_DEVICES-1 && MYLOCALID==NLOCALP-1)
#define FR_OX0 (FR_X_FIRST ? FR_OL : 0)
#define FR_OX1 (FR_X_LAST ? FR_OL : 0)
#define FR_WX0 (FR_X_FIRST ? FR_W : 0)
#define FR_WX1 (FR_X_LAST ? FR_W : 0)
#endif

FUNDEF void savebnd(GLOBARG __prec2 *sxx, GLOBARG __prec2 *sxz, GLOBARG __prec2 *szz,
                                   GLOBARG __prec2 *vx, GLOBARG __prec2 *vz,
                                   GLOBARG __prec2 *sxxbnd, GLOBARG __prec2 *sxzbnd, GLOBARG __prec2 *szzbnd,
                                   GLOBARG __prec2 *vxbnd, GLOBARG __prec2 *vzbnd)
{
    int i,k,indv,gidf;
    #ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
    #else
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    #endif
    int NZo = NZS-2*FDOH-FR_OT-FR_OL, NXo = NX-2*FDOH-FR_OX0-FR_OX1;
    int W = FR_W, Wt = FR_WT, W0 = FR_WX0, W1 = FR_WX1;
    int A0 = W0*NZo, A1 = W1*NZo, NXm = NXo-W0-W1;
    int i0 = FDOH+FR_OX0, k0 = FDOH+FR_OT;
    if (gid<A0){                         /* low x face */
        i = i0+gid/NZo;                 k = k0+gid%NZo;
    }
    else if (gid<A0+A1){                 /* high x face */
        gidf = gid-A0;
        i = i0+NXo-W1+gidf/NZo;         k = k0+gidf%NZo;
    }
    else if (gid<A0+A1+NXm*Wt){          /* top */
        gidf = gid-A0-A1;
        i = i0+W0+gidf/Wt;              k = k0+gidf%Wt;
    }
    else if (gid<A0+A1+NXm*(Wt+W)){      /* bottom */
        gidf = gid-A0-A1-NXm*Wt;
        i = i0+W0+gidf/W;               k = k0+NZo-W+gidf%W;
    }
    else{
        return;
    }
    /* Copy the raw stored values: float, or the 16 bits of a half (FP16 2). */
    #if FP16==2
        #ifdef __OPENCL_VERSION__
        #define RAWT ushort
        #else
        #define RAWT unsigned short
        #endif
    #else
        #define RAWT float
    #endif
    indv = i*NZS+k;
    ((GLOBARG RAWT*)vxbnd)[gid]  = ((GLOBARG RAWT*)vx)[indv];
    ((GLOBARG RAWT*)vzbnd)[gid]  = ((GLOBARG RAWT*)vz)[indv];
    ((GLOBARG RAWT*)sxxbnd)[gid] = ((GLOBARG RAWT*)sxx)[indv];
    ((GLOBARG RAWT*)szzbnd)[gid] = ((GLOBARG RAWT*)szz)[indv];
    ((GLOBARG RAWT*)sxzbnd)[gid] = ((GLOBARG RAWT*)sxz)[indv];
}
