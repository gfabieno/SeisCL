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

#include "F.h"

int alloc_seismo(float *** var, int ns, int allng, int NT, int * nrec ){
    int state=0;
    int i;
    
    GMALLOC(*var,sizeof(float*)*ns)
    if (!state) memset((void*)*var, 0, sizeof(float*)*ns);
    GMALLOC((*var)[0],sizeof(float)*allng*NT)
    if (!state) memset((void*)(*var)[0], 0, sizeof(float)*allng*NT);
    if (!state){
        for ( i=1; i<ns; i++){
            (*var)[i]=(*var)[i-1]+nrec[i-1]*NT;
        }
    }

    return state;
    
}


int Init_MPI(struct modcsts * m) {

    int state=0;
    int i;


    if (!state){
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast( &m->NX, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->NY, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->NZ, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->NT, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->FDORDER, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->nab, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->MAXRELERROR, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->gradout, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->ns, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->L, 1, MPI_INT, 0, MPI_COMM_WORLD );
        
        
        MPI_Bcast( &m->TAU, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->dt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->dh, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->abpc, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->pref_device_type, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->nmax_dev, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->n_no_use_GPUs, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->MPI_NPROC_SHOT, 1, MPI_INT, 0, MPI_COMM_WORLD );
        
        MPI_Bcast( &m->freesurf, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->ND, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->abs_type, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->VPPML, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->FPML, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->NPOWER, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->K_MAX_CPML, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->f0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->allng, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->allns, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->bcastvx, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->bcastvy, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->bcastvz, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->bcastsxx, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->bcastsyy, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->bcastszz, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->bcastsxy, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->bcastsxz, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->bcastsyz, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->bcastp, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->gradsrcout, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->Hout, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->seisout, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->movout, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->resout, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->rmsout, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->Hout, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->tmin, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->tmax, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->param_type, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->fmin, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->fmax, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->restype, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->vpmax, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->vsmin, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->topowidth, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->scalerms, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->scaleshot, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->scalermsnorm, 1, MPI_INT, 0, MPI_COMM_WORLD );
        
        MPI_Barrier(MPI_COMM_WORLD);
    }

    m->NGROUP=m->NP/m->MPI_NPROC_SHOT;
    if (m->NGROUP<1){
        m->NGROUP=1;
        m->MPI_NPROC_SHOT=m->NP;
    }
    
    m->MYGROUPID=m->MYID/m->MPI_NPROC_SHOT;
    
    if (m->NP>m->MPI_NPROC_SHOT){
        if (m->MYGROUPID>m->NGROUP-1){
            m->NLOCALP=m->MPI_NPROC_SHOT;
        }
        else{
            m->NLOCALP=m->MPI_NPROC_SHOT+m->NP%m->MPI_NPROC_SHOT;
        }
        m->MYLOCALID=m->MYID-m->MYGROUPID*m->MPI_NPROC_SHOT;
    }
    else{
        m->NLOCALP=m->NP;
        m->MYLOCALID=m->MYID;
    }
    
    if (m->n_no_use_GPUs>0){
        if (m->MYID!=0){
            GMALLOC(m->no_use_GPUs,sizeof(int)*m->n_no_use_GPUs)
        }
        
        if (!state){
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast( m->no_use_GPUs, m->n_no_use_GPUs, MPI_INT, 0, MPI_COMM_WORLD );
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    if (m->MYID!=0){
        GMALLOC(m->nsrc,sizeof(int)*m->ns)
        GMALLOC(m->src_pos,sizeof(float*)*m->ns)
        if (!state) memset((void*)m->src_pos, 0, sizeof(float*)*m->ns);
        GMALLOC(m->src_pos[0],sizeof(float)*m->allns*5)
        
        GMALLOC(m->src,sizeof(float*)*m->ns)
        if (!state) memset((void*)m->src, 0, sizeof(float*)*m->ns);
        GMALLOC(m->src[0],sizeof(float)*m->allns*m->NT)
        
        GMALLOC(m->nrec,sizeof(int)*m->ns)
        GMALLOC(m->rec_pos,sizeof(float*)*m->ns)
        if (!state) memset((void*)m->rec_pos, 0, sizeof(float*)*m->ns);
        GMALLOC(m->rec_pos[0],sizeof(float)*m->allng*8)
    }

    if (!state){
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast( m->nsrc, m->ns, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( m->src_pos[0], m->allns*5, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( m->src[0], m->allns*m->NT,  MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( m->nrec, m->ns, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( m->rec_pos[0], m->allng*8,  MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (m->MYID!=0){
        for (i=1; i<m->ns; i++){
            m->src_pos[i]=m->src_pos[i-1]+m->nsrc[i-1]*5;
            m->src[i]=m->src[i-1]+m->nsrc[i-1]*m->NT;
            m->rec_pos[i]=m->rec_pos[i-1]+m->nrec[i-1]*8;
        }
    }

    
    if (m->MYID!=0){
        GMALLOC(m->rho,sizeof(float)*m->NX*m->NY*m->NZ)
        GMALLOC(m->u,sizeof(float)*m->NX*m->NY*m->NZ)
        if (m->ND!=21)
            GMALLOC(m->pi,sizeof(float)*m->NX*m->NY*m->NZ)
    }
    
    if (!state){
        MPI_Bcast( m->rho, m->NX*m->NY*m->NZ, MPI_FLOAT, 0, MPI_COMM_WORLD );
        MPI_Bcast( m->u, m->NX*m->NY*m->NZ, MPI_FLOAT, 0, MPI_COMM_WORLD );
        if (m->ND!=21)
            MPI_Bcast( m->pi, m->NX*m->NY*m->NZ, MPI_FLOAT, 0, MPI_COMM_WORLD );
    }

    if (m->L>0){
        if (m->MYID!=0){
            GMALLOC(m->FL,sizeof(float)*m->L)
            if (m->ND!=21)
                GMALLOC(m->taup,sizeof(float)*m->NX*m->NY*m->NZ)
            GMALLOC(m->taus,sizeof(float)*m->NX*m->NY*m->NZ)
        }
        
        if (!state){
            MPI_Bcast( m->FL, m->L, MPI_FLOAT, 0, MPI_COMM_WORLD );
            if (m->ND!=21)
                MPI_Bcast( m->taup, m->NX*m->NY*m->NZ, MPI_FLOAT, 0, MPI_COMM_WORLD );
            MPI_Bcast( m->taus, m->NX*m->NY*m->NZ, MPI_FLOAT, 0, MPI_COMM_WORLD );
        }
    }

    if (m->topo){
        if (m->MYID!=0){
            GMALLOC(m->topo,sizeof(float)*m->NX*m->NY)
        }
        if (!state){
            MPI_Bcast( m->topo, m->NX*m->NY, MPI_FLOAT, 0, MPI_COMM_WORLD );
        }
    }
    // Allocate memory for the seismograms
    if (m->seisout && m->seisout!=2){
        if (m->ND!=21){
            alloc_seismo(&m->vxout, m->ns, m->allng, m->NT, m->nrec);
            m->bcastvx=1;
            alloc_seismo(&m->vzout, m->ns, m->allng, m->NT, m->nrec);
            m->bcastvz=1;
        }
        if (m->ND==3 || m->ND==21){
            alloc_seismo(&m->vyout, m->ns, m->allng, m->NT, m->nrec);
            m->bcastvy=1;
        }
    }
    if (m->seisout && m->seisout!=1){
        alloc_seismo(&m->pout, m->ns, m->allng, m->NT, m->nrec);
        m->bcastp=1;
    }
    if (m->seisout && m->seisout==4){
        
        if (m->ND!=21){
            alloc_seismo(&m->sxxout, m->ns, m->allng, m->NT, m->nrec);
            m->bcastsxx=1;
            alloc_seismo(&m->szzout, m->ns, m->allng, m->NT, m->nrec);
             m->bcastszz=1;
            alloc_seismo(&m->sxzout, m->ns, m->allng, m->NT, m->nrec);
             m->bcastsxz=1;
        }
        if (m->ND==3 || m->ND==21){
            alloc_seismo(&m->sxyout, m->ns, m->allng, m->NT, m->nrec);
             m->bcastsxy=1;
            alloc_seismo(&m->syzout, m->ns, m->allng, m->NT, m->nrec);
             m->bcastsyz=1;
        }
        if (m->ND==3 ){
            alloc_seismo(&m->syyout, m->ns, m->allng, m->NT, m->nrec);
             m->bcastsyy=1;
        }
    }
    
    // Allocate memory for the movie
    if (m->movout>0){
        if (m->ND!=21){
            GMALLOC(m->movvx,sizeof(float)*m->ns*m->NX*m->NY*m->NZ*m->NT/m->movout)
            GMALLOC(m->movvz,sizeof(float)*m->ns*m->NX*m->NY*m->NZ*m->NT/m->movout)
            
            if (!state) memset((void*)m->movvx, 0, sizeof(float)*m->ns*m->NX*m->NY*m->NZ*m->NT/m->movout);
            if (!state) memset((void*)m->movvz, 0, sizeof(float)*m->ns*m->NX*m->NY*m->NZ*m->NT/m->movout);
            
        }
        if (m->ND==3 || m->ND==21){
            GMALLOC(m->movvy,sizeof(float)*m->ns*m->NX*m->NY*m->NZ*m->NT/m->movout)
            if (!state) memset((void*)m->movvy, 0, sizeof(float)*m->ns*m->NX*m->NY*m->NZ*m->NT/m->movout);
        }

    }
    

    if ( (m->rmsout==1 || m->resout) && m->bcastvx){
        
        // Allocate memory for the residuals seismograms
        alloc_seismo(&m->rx, m->ns, m->allng, m->NT, m->nrec);
        if (m->MYID!=0){
            alloc_seismo(&m->vx0, m->ns, m->allng, m->NT, m->nrec);
        }
        if (!state){
            MPI_Bcast( m->vx0[0], m->allng*m->NT, MPI_FLOAT, 0, MPI_COMM_WORLD );
        }
    }
    if ( (m->rmsout==1 || m->resout) && m->bcastvy){
        
        // Allocate memory for the resiudals seismograms
        alloc_seismo(&m->ry, m->ns, m->allng, m->NT, m->nrec);
        if (m->MYID!=0){
            alloc_seismo(&m->vy0, m->ns, m->allng, m->NT, m->nrec);
        }
        if (!state){
            MPI_Bcast( m->vy0[0], m->allng*m->NT, MPI_FLOAT, 0, MPI_COMM_WORLD );
        }
    }
    if ( (m->rmsout==1 || m->resout) && m->bcastvz){
        
        // Allocate memory for the resiudals seismograms
        alloc_seismo(&m->rz, m->ns, m->allng, m->NT, m->nrec);
        if (m->MYID!=0){
            alloc_seismo(&m->vz0, m->ns, m->allng, m->NT, m->nrec);
        }
        if (!state){
            MPI_Bcast( m->vz0[0], m->allng*m->NT, MPI_FLOAT, 0, MPI_COMM_WORLD );
        }
    }
//    if ( (m->rmsout==1 || m->resout) && m->bcastp){
//        
//        // Allocate memory for the resiudals seismograms
//        alloc_seismo(&m->rp, m->ns, m->allng, m->NT, m->nrec);
//        if (m->MYID!=0){
//            alloc_seismo(&m->p0, m->ns, m->allng, m->NT, m->nrec);
//        }
//        if (!state){
//            MPI_Bcast( m->p0[0], m->allng*m->NT, MPI_FLOAT, 0, MPI_COMM_WORLD );
//        }
//    }
    

    if (m->gradout==1 ){
        
        MPI_Bcast( &m->back_prop_type, 1, MPI_INT, 0, MPI_COMM_WORLD );
        if (m->back_prop_type==2){
            MPI_Bcast( &m->nfreqs, 1, MPI_INT, 0, MPI_COMM_WORLD );
            
            if (m->MYID!=0){
                GMALLOC(m->gradfreqs,sizeof(float*)*m->nfreqs)
            }
            
            MPI_Bcast( m->gradfreqs, m->nfreqs, MPI_FLOAT, 0, MPI_COMM_WORLD );
        }
    }
    
    
    MPI_Barrier(MPI_COMM_WORLD);

    return state;

}
