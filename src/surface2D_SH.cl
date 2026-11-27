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

/* Adjoint of the SH free surface (back_prop_type 1 and 2), surface2D_adj.cl's
   surface_adj_fold (P-SV, sxz part) for syz. time_stepping.c runs it after the
   adjoint stress update, where the forward runs freesurface after the stress update.
   In the forward step, update_v reads the syz images above the surface through Dzm:
   vy(j) gets -hc[m] syz(j-m-1) with j-m-1 = FDOH-k an image row (k = 1..FDOH,
   m >= k-1), and the image is -syz(FDOH+k-1). The transpose folds the adjoint
   velocities back, with a minus sign, into the rows below the surface:
       img(k) = -sum_{m=k-1}^{FDOH-1} hc[m] vyr(FDOH-k+m+1),
       syzr(FDOH+k-1) += mujpkp(FDOH+k-1) img(k),
   and the image rows of the adjoint stress stay zero. Until 2026-11-19 this kernel
   applied the forward image condition to syzr instead: right below the top
   2 FDOH rows, wrong in them (receiver 3 rows under the surface: mu 1.078, rho
   1.016 back_prop_type 2). The fold is part of the stress update in mu, so with
   back_prop_type 1 its increment mujpkp img enters the summation-by-parts
   gradient, with the forward syz that update_adjs has just rebuilt.
   One thread per column, writes its own column only. Under back_prop_type 2,
   syzr is bound to the syz buffer and vyr to vy. */
FUNDEF void surface_adj(GLOBARG STO *vyr,     GLOBARG STO *syzr,
                        GLOBARG STO *syz,     GLOBARG STO *mujpkp,
                        GLOBARG float *gradmujpkp,
                        int src_scale,        int res_scale,
                        int pdir, int par_scale)
{
    #ifdef __OPENCL_VERSION__
    int gidx = get_global_id(0) + FDOH;
    #else
    int gidx = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    #endif
    if (gidx > NX-FDOH-1) return;
    float hc[6] = {HC1, HC2, HC3, HC4, HC5, HC6};
    for (int k=1; k<=FDOH; k++){
        float img = 0.0f;
        for (int m=k-1; m<FDOH; m++){
            img -= hc[m]*ldf(vyr, INDV(FDOH-k+m+1, gidx));
        }
        int z = FDOH+k-1;
        int ip = (gidx-FDOH)*(NZS-2*FDOH) + (z-FDOH);
        float lmu = ldf(mujpkp, ip);
        stf(syzr, INDV(z, gidx), ldf(syzr, INDV(z, gidx)) + lmu*img);
#if BACK_PROP_TYPE==1 && GRADOUT==1
        if (lmu != 0.0f)
            gradmujpkp[ip] += GSCALE(-ldf(syz, INDV(z, gidx))*img/lmu);
#endif
    }
    for (int m=1; m<=FDOH; m++){
        stf(syzr, INDV(FDOH-m, gidx), 0.0f);
    }
}
