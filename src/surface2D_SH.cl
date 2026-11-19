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
