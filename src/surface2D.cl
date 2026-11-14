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

/* Stress-image free surface (FREESURF 1) in 2D (Levander, 1988), for every
   precision mode (FP16 0, 1, 2; storage from header_vec.cl). One thread per
   column: szz = 0 on the surface row, the stresses mirrored above it, and the
   sxx correction h = -(M-2mu)^2/M vxx - (M-2mu) vzz that makes the update of
   the surface row consistent with szz = 0. */

#define INDV(z,x) ((x)*NZS+(z))
#define PSIX(z,x) ((x)*(NZS-2*FDOH)+(z))

FUNDEF void freesurface(GLOBARG STO *vx,    GLOBARG STO *vz,
                        GLOBARG STO *sxx,   GLOBARG STO *szz,
                        GLOBARG STO *sxz,   GLOBARG STO *M,
                        GLOBARG STO *mu,    GLOBARG STO *rxx,
                        GLOBARG STO *rzz,   GLOBARG STO *taus,
                        GLOBARG STO *taup,  GLOBARG float *eta,
                        GLOBARG float *K_x, GLOBARG float *psi_vx_x,
                        GLOBARG float *taper, int pdir)
{
    #ifdef __OPENCL_VERSION__
    int gidx = get_global_id(0) + FDOH;
    #else
    int gidx = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    #endif
    int gidz = FDOH;
    if (gidx > NX-FDOH-1) return;               // the grid is rounded up to the work-group size
    const float hc[6] = HCV;
    int indp = (gidx-FDOH)*(NZS-2*FDOH) + (gidz-FDOH);

    /* Mirror the stresses: a stress-free surface by the method of images */
    stf(szz, INDV(gidz, gidx), 0.0f);
#if LVE>0
    for (int l=0; l<LVE; l++) stf(rzz, l*NX*NZS + INDV(gidz, gidx), 0.0f);
#endif
    for (int m=1; m<=FDOH; m++){
        stf(szz, INDV(gidz-m, gidx), -ldf(szz, INDV(gidz+m, gidx)));
        stf(sxz, INDV(gidz-m, gidx), -ldf(sxz, INDV(gidz+m-1, gidx)));
    }

    /* Velocity derivatives on the surface row (the rows above it are zero) */
    float vxx = 0.0f, vzz = 0.0f;
    for (int m=0; m<FDOH; m++){
        vxx += hc[m]*(ldf(vx, INDV(gidz, gidx+m)) - ldf(vx, INDV(gidz, gidx-m-1)));
        vzz += hc[m]*(ldf(vz, INDV(gidz+m, gidx)) - ldf(vz, INDV(gidz-m-1, gidx)));
    }
#if ABS_TYPE==1
#if DEVID==0 & MYLOCALID==0
    if (gidx-FDOH<NAB){
        int i = gidx-FDOH, k = gidz-FDOH;
        vxx = vxx/K_x[i] + psi_vx_x[PSIX(k,i)];
    }
#endif
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-NAB-FDOH-1){
        int i = gidx-NX+NAB+FDOH+NAB, k = gidz-FDOH;
        int ind = 2*NAB-1-i;
        vxx = vxx/K_x[ind+1] + psi_vx_x[PSIX(k,i)];
    }
#endif
#endif

#if LVE==0
    float f = 2.0f*ldf(mu, indp);
    float g = ldf(M, indp);
    float h = -((g-f)*(g-f)*vxx/g) - ((g-f)*vzz);
    stf(sxx, INDV(gidz, gidx), ldf(sxx, INDV(gidz, gidx)) + pdir*h);
#else
    float f = 2.0f*ldf(mu, indp)*(1.0f + LVE*ldf(taus, indp));
    float g = ldf(M, indp)*(1.0f + LVE*ldf(taup, indp));
    float h = -((g-f)*(g-f)*vxx/g) - ((g-f)*vzz);
    float sump = 0.0f;
    for (int l=0; l<LVE; l++) sump += ldf(rxx, l*NX*NZS + INDV(gidz, gidx));
    h += -DT2*sump;
    float lsxx = ldf(sxx, INDV(gidz, gidx)) + pdir*h;
    /* memory variable rxx on the surface */
    float d = 2.0f*ldf(mu, indp)*ldf(taus, indp)/DT;
    float e = ldf(M, indp)*ldf(taup, indp)/DT;
    sump = 0.0f;
    for (int l=0; l<LVE; l++){
        float b = eta[l]/(1.0f + eta[l]*0.5f);
        float hr = b*(((d-e)*((f/g)-1.0f)*vxx) - ((d-e)*vzz));
        float r = ldf(rxx, l*NX*NZS + INDV(gidz, gidx)) + pdir*hr;
        stf(rxx, l*NX*NZS + INDV(gidz, gidx), r);
        sump += r;
    }
    lsxx += pdir*(DT2*sump);
    stf(sxx, INDV(gidz, gidx), lsxx);
#endif
}
