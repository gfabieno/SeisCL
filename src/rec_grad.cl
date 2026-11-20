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

/* Residual terms of the gradient at the receiver cells (BACK_PROP_TYPE 1; storage from header_vec.cl).

   The adjoint kernels accumulate the gradient in the summation-by-parts form,
   forward field x the adjoint increment; at a receiver cell the increment also
   holds the residual injected this step, which their own increment misses. These
   terms used to be computed in update_adjv / update_adjs by a loop over all the
   receivers IN EVERY CELL at every time step: with 196 receivers in 2-D, 115 s
   for update_adjv against 1 s without. Here: one work-item per receiver. Two
   receivers can share a cell; to avoid a write race without atomics, only the
   first receiver of a cell (lowest index) does work, and it adds the terms of
   every receiver of that cell in increasing index order, the order of the
   former one-work-item loop: the result is bit-identical to it.

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
#define INDV(z,y,x) ((x)*NY*NZS+(y)*NZS+(z))
#define INDP(z,y,x) (((x)-FDOH)*(NY-2*FDOH)*(NZS-2*FDOH)+((y)-FDOH)*(NZS-2*FDOH)+((z)-FDOH))
#else
#define INDV(z,y,x) ((x)*NZS+(z))
#define INDP(z,y,x) (((x)-FDOH)*(NZS-2*FDOH)+((z)-FDOH))
#endif


/* Cell (k, j, i) of receiver g, padded indices. */
#if ND==3
#define REC_CELL(g, k, j, i) do{ i = (int)(rec_pos[0+8*(g)]/DH)+FDOH; \
                                 j = (int)(rec_pos[1+8*(g)]/DH)+FDOH; \
                                 k = (int)(rec_pos[2+8*(g)]/DH)+FDOH; } while(0)
#else
#define REC_CELL(g, k, j, i) do{ i = (int)(rec_pos[0+8*(g)]/DH)+FDOH; j = 0; \
                                 k = (int)(rec_pos[2+8*(g)]/DH)+FDOH; } while(0)
#endif

/* This work-item's receiver g0, and whether it owns its cell: it returns if g0
   is past the last receiver or if a receiver of lower index is in the same cell.
   The owner then loops over g = g0 .. nrec-1 and skips the other cells. */
#ifdef __OPENCL_VERSION__
#define REC_GID get_global_id(0)
#else
#define REC_GID (blockIdx.x*blockDim.x + threadIdx.x)
#endif
#define REC_OWNER(g0, k0, j0, i0) \
    int g0 = REC_GID; \
    if (g0 >= nrec) return; \
    int k0, j0, i0; \
    REC_CELL(g0, k0, j0, i0); \
    for (int h=0; h<g0; h++){ int kh, jh, ih; REC_CELL(h, kh, jh, ih); \
        if (kh==k0 && jh==j0 && ih==i0) return; }

FUNDEF void rec_grad_v(GLOBARG STO *vx,        GLOBARG STO *vy,        GLOBARG STO *vz,
                       GLOBARG float *vxout,   GLOBARG float *vyout,   GLOBARG float *vzout,
                       GLOBARG float *rec_pos, int nrec,               int nt,
                       GLOBARG float *taper,
                       GLOBARG float *gradrip, GLOBARG float *gradrjp, GLOBARG float *gradrkp,
                       int src_scale,          int res_scale)
{
#if GRADOUT==1 && VELOUT==1
    REC_OWNER(g0, k0, j0, i0)
    for (int g=g0; g<nrec; g++){
        int k, j, i;
        REC_CELL(g, k, j, i);
        if (k!=k0 || j!=j0 || i!=i0) continue;
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
        if (k>NZS-NAB-FDOH-1) gt*=taper[NZS-FDOH-k-1];
        #if ND==3
        if (j-FDOH<NAB) gt*=taper[j-FDOH];
        if (j>NY-NAB-FDOH-1) gt*=taper[NY-FDOH-j-1];
        #endif
        if (i-FDOH<NAB) gt*=taper[i-FDOH];
        if (i>NX-NAB-FDOH-1) gt*=taper[NX-FDOH-i-1];
        #endif
        int iv = INDV(k,j,i), ip = INDP(k,j,i);
        #if VXOUT==1
        gradrip[ip] += GSCALE(-ldf(vx, iv)/gt * pow2s(vxout[NT*g+nt], res_scale));
        #endif
        #if VYOUT==1 && ND==3
        gradrjp[ip] += GSCALE(-ldf(vy, iv)/gt * pow2s(vyout[NT*g+nt], res_scale));
        #endif
        #if VZOUT==1
        gradrkp[ip] += GSCALE(-ldf(vz, iv)/gt * pow2s(vzout[NT*g+nt], res_scale));
        #endif
    }
#endif
}

FUNDEF void rec_grad_s(GLOBARG STO *sxx,       GLOBARG STO *syy,       GLOBARG STO *szz,
                       GLOBARG float *pout,    GLOBARG float *rec_pos, int nrec,
                       int nt,                 GLOBARG STO *M,         GLOBARG STO *mu,
                       GLOBARG float *gradM,   GLOBARG float *gradmu,
                       int src_scale,          int res_scale)
{
#if GRADOUT==1 && PRESOUT==1
    REC_OWNER(g0, k0, j0, i0)
    for (int g=g0; g<nrec; g++){
        int k, j, i;
        REC_CELL(g, k, j, i);
        if (k!=k0 || j!=j0 || i!=i0) continue;
        #if FREESURF==1
        if (k-FDOH < SURFROWS) continue;      /* direct form (surface_grad) */
        #endif
        int iv = INDV(k,j,i), ip = INDP(k,j,i);
        float lM = ldf(M, ip), lmu = ldf(mu, ip);
        float res = pow2s(pout[NT*g+nt], res_scale);
        #if ND==3
        float den = 3.0f*lM-4.0f*lmu;
        float c1 = (den!=0.0f) ? 1.0f/(den*den) : 0.0f;
        float dM = c1*(ldf(sxx, iv)+ldf(syy, iv)+ldf(szz, iv))*res;
        gradM[ip]  += GSCALE(-dM);
        gradmu[ip] += GSCALE(4.0f/3.0f*dM);
        #else
        float den = 2.0f*lM-2.0f*lmu;
        float c1 = (den!=0.0f) ? 1.0f/(den*den) : 0.0f;
        float dM = c1*(ldf(sxx, iv)+ldf(szz, iv))*res;
        gradM[ip]  += GSCALE(-dM);
        gradmu[ip] += GSCALE(dM);
        #endif
    }
#endif
}
