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

/* Stress-image free surface (FREESURF 1) in 2D SH (Levander, 1988), in the format
   of surface2D.cl (P-SV). One thread per column. vy lies on the surface row
   z = FDOH; syz (at z+1/2) is mirrored with opposite sign about it, so that
   syz = 0 at the surface: syz(FDOH-m) = -syz(FDOH+m-1), m = 1..FDOH. sxy needs no
   condition. The old kernel (named "surface", OpenCL-only syntax) did the same but
   could not be found by the host, which asks for "freesurface". */

#define INDV(z,x) ((x)*NZS+(z))

FUNDEF void freesurface(GLOBARG STO *syz, int pdir)
{
    #ifdef __OPENCL_VERSION__
    int gidx = get_global_id(0) + FDOH;
    #else
    int gidx = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    #endif
    int gidz = FDOH;
    if (gidx > NX-FDOH-1) return;               // the grid is rounded up to the work-group size
    for (int m=1; m<=FDOH; m++){
        stf(syz, INDV(gidz-m, gidx), -ldf(syz, INDV(gidz+m-1, gidx)));
    }
}

/* Adjoint of the SH free surface, back_prop_type 2: the same image condition on
   the adjoint stress (time_stepping.c runs it after the adjoint stress update,
   where the forward runs freesurface after the stress update). In SH the image
   acts on syz alone, which the transform of thesis Ch. 3 does not mix with any
   other component (G is diagonal), so no transform bracket as in
   surface2D_adj.cl (P-SV) should be needed (notes sh-gradient.md, 2.6): checked
   by finite differences with freesurf = 1. Under back_prop_type 2, syzr is bound
   to the syz buffer. */
FUNDEF void surface_adj(GLOBARG STO *syzr, int pdir)
{
    #ifdef __OPENCL_VERSION__
    int gidx = get_global_id(0) + FDOH;
    #else
    int gidx = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    #endif
    int gidz = FDOH;
    if (gidx > NX-FDOH-1) return;
    for (int m=1; m<=FDOH; m++){
        stf(syzr, INDV(gidz-m, gidx), -ldf(syzr, INDV(gidz+m-1, gidx)));
    }
}
