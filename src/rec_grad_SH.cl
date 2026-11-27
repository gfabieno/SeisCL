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

/* Residual term of the gradient at the receiver cells, SH (ND 21), BACK_PROP_TYPE 1; rec_grad.cl's rec_grad_v
   with vy alone (SH gradients need velocity receivers). vy sits at rjp = rho, so the term goes to gradrho.
   rec_grad_v runs at the start of the adjoint step: the forward velocity is still vy(t+1), the velocity
   recorded by the residual of this step:   gradrho += -vy(t+1) * res.
   A separate file because clprogram.c binds the arguments by name, from the first signature in the source. */

#define INDV(z,x) ((x)*NZS+(z))
#define INDP(z,x) (((x)-FDOH)*(NZS-2*FDOH)+((z)-FDOH))

FUNDEF void rec_grad_v(GLOBARG STO *vy,        GLOBARG float *vyout,
                       GLOBARG float *rec_pos, int nrec,               int nt,
                       GLOBARG float *taper,   GLOBARG float *gradrho,
                       int src_scale,          int res_scale, int par_scale)
{
#if GRADOUT==1 && VELOUT==1 && VYOUT==1
    #ifdef __OPENCL_VERSION__
    if (get_global_id(0) != 0) return;
    #else
    if (blockIdx.x*blockDim.x + threadIdx.x != 0) return;
    #endif
    for (int g=0; g<nrec; g++){
        int i = (int)(rec_pos[0+8*g]/DH)+FDOH-OFFSET;   /* this subdomain */
        int k = (int)(rec_pos[2+8*g]/DH)+FDOH;
        if (i<FDOH || i>NX-FDOH-1) continue;
        /* vy(t+1) without the taper of this step (see rec_grad.cl) */
        float gt = 1.0f;
        #if ABS_TYPE==2
        #if FREESURF==0
        if (k-FDOH<NAB) gt*=taper[k-FDOH];
        #endif
        if (k>NZS-NAB-FDOH-1) gt*=taper[NZS-FDOH-k-1];
        if (DEVID==0 && MYLOCALID==0 && i-FDOH<NAB) gt*=taper[i-FDOH];
        if (DEVID==NUM_DEVICES-1 && MYLOCALID==NLOCALP-1 && i>NX-NAB-FDOH-1) gt*=taper[NX-FDOH-i-1];
        #endif
        gradrho[INDP(k,i)] += GSCALE(-ldf(vy, INDV(k,i))/gt * pow2s(vyout[NT*g+nt], res_scale));
    }
#endif
}
