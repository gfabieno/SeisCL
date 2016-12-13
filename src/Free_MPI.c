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
// Free all buffers on the host

#include "F.h"

int Free_MPI(struct modcsts * m)  {
    
    GFree(m->rip)
    GFree(m->rjp)
    GFree(m->rkp)
    GFree(m->uipjp)
    GFree(m->ujpkp)
    GFree(m->uipkp)
    GFree(m->taper)
    GFree(m->tausipjp)
    GFree(m->tausjpkp)
    GFree(m->tausipkp)
    GFree(m->eta)
    
    GFree(m->gradfreqsn)

    GFree(m->K_x)
    GFree(m->a_x)
    GFree(m->b_x)
    GFree(m->K_x_half)
    GFree(m->a_x_half)
    GFree(m->b_x_half)
    
    GFree(m->K_y)
    GFree(m->a_y)
    GFree(m->b_y)
    GFree(m->K_y_half)
    GFree(m->a_y_half)
    GFree(m->b_y_half)
    
    GFree(m->K_z)
    GFree(m->a_z)
    GFree(m->b_z)
    GFree(m->K_z_half)
    GFree(m->a_z_half)
    GFree(m->b_z_half)
    
    
    GFree(m->rho)
    GFree(m->u)
    GFree(m->pi)
    GFree(m->taus)
    GFree(m->taup)
    GFree(m->FL)

    GFree(m->gradrho)
    GFree(m->gradM)
    GFree(m->gradmu)
    GFree(m->gradtaup)
    GFree(m->gradtaus)
    if  (m->gradsrc){
        GFree(m->gradsrc[0])
    }
    GFree(m->gradsrc)
    
    GFree(m->topo);
    GFree(m->sinccoef);
    
    GFree(m->H.pp);
    GFree(m->H.mp);
    GFree(m->H.up);
    GFree(m->H.tpp);
    GFree(m->H.tsp);
    
    GFree(m->H.mm);
    GFree(m->H.um);
    GFree(m->H.tpm);
    GFree(m->H.tsm);
    
    GFree(m->H.uu);
    GFree(m->H.tpu);
    GFree(m->H.tsu);
    
    GFree(m->H.tptp);
    GFree(m->H.tstp);
    
    GFree(m->H.tsts);
    

    GFree(m->no_use_GPUs)

    GFree(m->nsrc)
    GFree(m->nrec)
    
    if  (m->src_pos){
        GFree(m->src_pos[0])
    }
    if  (m->rec_pos){
        GFree(m->rec_pos[0])
    }
    GFree(m->src_pos)
    GFree(m->src)
    GFree(m->rec_pos)
    
    
    if  (m->vxout){
        GFree(m->vxout[0])
    }
    if  (m->vyout){
        GFree(m->vyout[0])
    }
    if  (m->vzout){
        GFree(m->vzout[0])
    }
    GFree(m->vxout)
    GFree(m->vyout)
    GFree(m->vzout)

    GFree(m->gradfreqs)
    
    if  (m->vx0){
        GFree(m->vx0[0])
    }
    if  (m->vy0){
        GFree(m->vy0[0])
    }
    if  (m->vz0){
        GFree(m->vz0[0])
    }
    GFree(m->vx0)
    GFree(m->vy0)
    GFree(m->vz0)
    
    if  (m->rx){
        GFree(m->rx[0])
    }
    if  (m->ry){
        GFree(m->ry[0])
    }
    if  (m->rz){
        GFree(m->rz[0])
    }
    GFree(m->rx)
    GFree(m->ry)
    GFree(m->rz)
    
    if  (m->mute){
        GFree(m->mute[0])
    }
    GFree(m->mute)
    if  (m->weight){
        GFree(m->weight[0])
    }
    GFree(m->weight)
    
    return 0;
    
    
}