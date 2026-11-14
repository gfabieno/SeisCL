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

/* Residual terms of the gradient at the receiver cells (BACK_PROP_TYPE 1, FP16 0).

   The adjoint kernels accumulate the gradient in the summation-by-parts form,
   forward field x the adjoint increment; at a receiver cell the increment also
   holds the residual injected this step, which their own increment misses. These
   terms used to be computed in update_adjv / update_adjs by a loop over all the
   receivers IN EVERY CELL at every time step: with 196 receivers in 2-D, 115 s
   for update_adjv against 1 s without. Here: one work-item loops over the
   receivers (no write race when two receivers share a cell).

   rec_grad_v runs at the start of the adjoint step, before update_adjs: the
   forward velocity is still v(t+1) (the velocity recorded by the residual of this
   step), as (vx+fvx) was in update_adjv:   gradrip += -v(t+1) * res.
   rec_grad_s runs after update_adjs, which has just rebuilt the forward stresses
   to s(t), as in update_adjs:  dM = c1 * trace(s) * res; gradM -= dM; gradmu += a dM.
   The rows under a stress-image free surface computed in the direct form by
   surface_grad (which already holds the residual) are skipped. */

#ifndef SURFROWS
#define SURFROWS (FDOH+1)
#endif

#if ND==3
#define INDV(z,y,x) ((x)*NY*NZ+(y)*NZ+(z))
#define INDP(z,y,x) (((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH))
#else
#define INDV(z,y,x) ((x)*NZ+(z))
#define INDP(z,y,x) (((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH))
#endif

FUNDEF void rec_grad_v(GLOBARG float *vx,      GLOBARG float *vy,      GLOBARG float *vz,
                       GLOBARG float *vxout,   GLOBARG float *vyout,   GLOBARG float *vzout,
                       GLOBARG float *rec_pos, int nrec,               int nt,
                       GLOBARG float *taper,
                       GLOBARG float *gradrip, GLOBARG float *gradrjp, GLOBARG float *gradrkp)
{
#if GRADOUT==1 && VELOUT==1
    #ifdef __OPENCL_VERSION__
    if (get_global_id(0) != 0) return;
    #else
    if (blockIdx.x*blockDim.x + threadIdx.x != 0) return;
    #endif
    for (int g=0; g<nrec; g++){
        int i = (int)(rec_pos[0+8*g]/DH)+FDOH;
        int j = (int)(rec_pos[1+8*g]/DH)+FDOH;
        int k = (int)(rec_pos[2+8*g]/DH)+FDOH;
        #if ND!=3
        j = 0;
        #endif
        #if FREESURF==1
        if (k-FDOH < FDOH) continue;          /* direct form (surface_grad) */
        #endif
        /* v(t+1) without the taper of this step: the residual measures the
           recorded velocity, the update before the taper is what the
           summation form pairs (v_old / taper in update_adjv). */
        float gt = 1.0f;
        #if ABS_TYPE==2
        #if FREESURF==0
        if (k-FDOH<NAB) gt*=taper[k-FDOH];
        #endif
        if (k>NZ-NAB-FDOH-1) gt*=taper[NZ-FDOH-k-1];
        #if ND==3
        if (j-FDOH<NAB) gt*=taper[j-FDOH];
        if (j>NY-NAB-FDOH-1) gt*=taper[NY-FDOH-j-1];
        #endif
        if (i-FDOH<NAB) gt*=taper[i-FDOH];
        if (i>NX-NAB-FDOH-1) gt*=taper[NX-FDOH-i-1];
        #endif
        int iv = INDV(k,j,i), ip = INDP(k,j,i);
        #if VXOUT==1
        gradrip[ip] += -vx[iv]/gt * vxout[NT*g+nt];
        #endif
        #if VYOUT==1 && ND==3
        gradrjp[ip] += -vy[iv]/gt * vyout[NT*g+nt];
        #endif
        #if VZOUT==1
        gradrkp[ip] += -vz[iv]/gt * vzout[NT*g+nt];
        #endif
    }
#endif
}

FUNDEF void rec_grad_s(GLOBARG float *sxx,     GLOBARG float *syy,     GLOBARG float *szz,
                       GLOBARG float *pout,    GLOBARG float *rec_pos, int nrec,
                       int nt,                 GLOBARG float *M,       GLOBARG float *mu,
                       GLOBARG float *gradM,   GLOBARG float *gradmu)
{
#if GRADOUT==1 && PRESOUT==1
    #ifdef __OPENCL_VERSION__
    if (get_global_id(0) != 0) return;
    #else
    if (blockIdx.x*blockDim.x + threadIdx.x != 0) return;
    #endif
    for (int g=0; g<nrec; g++){
        int i = (int)(rec_pos[0+8*g]/DH)+FDOH;
        int j = (int)(rec_pos[1+8*g]/DH)+FDOH;
        int k = (int)(rec_pos[2+8*g]/DH)+FDOH;
        #if ND!=3
        j = 0;
        #endif
        #if FREESURF==1
        if (k-FDOH < SURFROWS) continue;      /* direct form (surface_grad) */
        #endif
        int iv = INDV(k,j,i), ip = INDP(k,j,i);
        float lM = M[ip], lmu = mu[ip];
        float res = pout[NT*g+nt];
        #if ND==3
        float den = 3.0f*lM-4.0f*lmu;
        float c1 = (den!=0.0f) ? 1.0f/(den*den) : 0.0f;
        float dM = c1*(sxx[iv]+syy[iv]+szz[iv])*res;
        gradM[ip]  += -dM;
        gradmu[ip] += 4.0f/3.0f*dM;
        #else
        float den = 2.0f*lM-2.0f*lmu;
        float c1 = (den!=0.0f) ? 1.0f/(den*den) : 0.0f;
        float dM = c1*(sxx[iv]+szz[iv])*res;
        gradM[ip]  += -dM;
        gradmu[ip] += dM;
        #endif
    }
#endif
}
