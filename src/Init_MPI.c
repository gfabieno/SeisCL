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
        MPI_Bcast( &m->bcastvx, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->bcastvy, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->bcastvz, 1, MPI_INT, 0, MPI_COMM_WORLD );
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
    
    
    if (m->MYID!=0){
        GMALLOC(m->no_use_GPUs,sizeof(int)*m->n_no_use_GPUs)
    }
    
    if (!state){
        MPI_Bcast( m->no_use_GPUs, m->n_no_use_GPUs, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Barrier(MPI_COMM_WORLD);
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
    if (m->ND!=21){
        GMALLOC(m->vxout,sizeof(float*)*m->ns)
        GMALLOC(m->vzout,sizeof(float*)*m->ns)
        
        if (!state) memset((void*)m->vxout, 0, sizeof(float*)*m->ns);
        if (!state) memset((void*)m->vzout, 0, sizeof(float*)*m->ns);
        
        GMALLOC(m->vxout[0],sizeof(float)*m->allng*m->NT)
        GMALLOC(m->vzout[0],sizeof(float)*m->allng*m->NT)
        
        if (!state) memset((void*)m->vxout[0], 0, sizeof(float)*m->allng*m->NT);
        if (!state) memset((void*)m->vzout[0], 0, sizeof(float)*m->allng*m->NT);
        
        if (!state){
            for (i=1; i<m->ns; i++){
                m->vxout[i]=m->vxout[i-1]+m->nrec[i-1]*m->NT;
                m->vzout[i]=m->vzout[i-1]+m->nrec[i-1]*m->NT;
            }
        }
        
    }
    if (m->ND==3 || m->ND==21){
        GMALLOC(m->vyout,sizeof(float*)*m->ns)
        if (!state) memset((void*)m->vyout, 0, sizeof(float*)*m->ns);
        GMALLOC(m->vyout[0],sizeof(float)*m->allng*m->NT)
        if (!state) memset((void*)m->vyout[0], 0, sizeof(float)*m->allng*m->NT);
        if (!state){
            for (i=1; i<m->ns; i++){
                m->vyout[i]=m->vyout[i-1]+m->nrec[i-1]*m->NT;
            }
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
    
    
    
    
    
    
    if (m->bcastvx){
        
        // Allocate memory for the resiudals seismograms
        GMALLOC(m->rx,sizeof(float*)*m->ns)
        if (!state) memset((void*)m->rx, 0, sizeof(float*)*m->ns);
        GMALLOC(m->rx[0],sizeof(float)*m->allng*m->NT)
        if (!state) memset((void*)m->rx[0], 0, sizeof(float)*m->allng*m->NT);
        if (!state){
            for (i=1; i<m->ns; i++){
                m->rx[i]=m->rx[i-1]+m->nrec[i-1]*m->NT;
            }
        }

        
        if (m->MYID!=0){
            
            GMALLOC(m->vx0,sizeof(float*)*m->ns)
            if (!state) memset((void*)m->vx0, 0, sizeof(float*)*m->ns);
            GMALLOC(m->vx0[0],sizeof(float)*m->allng*m->NT)
            if (!state) memset((void*)m->vx0[0], 0, sizeof(float)*m->allng*m->NT);
            if (!state){
                for (i=1; i<m->ns; i++){
                    m->vx0[i]=m->vx0[i-1]+m->nrec[i-1]*m->NT;
                }
            }
        }
        
        if (!state){
            MPI_Bcast( m->vx0[0], m->allng*m->NT, MPI_FLOAT, 0, MPI_COMM_WORLD );
        }
    }
    if (m->bcastvy){
        
        // Allocate memory for the resiudals seismograms
        GMALLOC(m->ry,sizeof(float*)*m->ns)
        if (!state) memset((void*)m->ry, 0, sizeof(float*)*m->ns);
        GMALLOC(m->ry[0],sizeof(float)*m->allng*m->NT)
        if (!state) memset((void*)m->ry[0], 0, sizeof(float)*m->allng*m->NT);
        if (!state){
            for (i=1; i<m->ns; i++){
                m->ry[i]=m->ry[i-1]+m->nrec[i-1]*m->NT;
            }
        }

        if (m->MYID!=0){
            
            GMALLOC(m->vy0,sizeof(float*)*m->ns)
            if (!state) memset((void*)m->vy0, 0, sizeof(float*)*m->ns);
            GMALLOC(m->vy0[0],sizeof(float)*m->allng*m->NT)
            if (!state) memset((void*)m->vy0[0], 0, sizeof(float)*m->allng*m->NT);
            if (!state){
                for (i=1; i<m->ns; i++){
                    m->vy0[i]=m->vy0[i-1]+m->nrec[i-1]*m->NT;
                }
            }
        }
        
        if (!state){
            MPI_Bcast( m->vy0[0], m->allng*m->NT, MPI_FLOAT, 0, MPI_COMM_WORLD );
        }
    }
    if (m->bcastvz){
        
        // Allocate memory for the resiudals seismograms
        GMALLOC(m->rz,sizeof(float*)*m->ns)
        if (!state) memset((void*)m->rz, 0, sizeof(float*)*m->ns);
        GMALLOC(m->rz[0],sizeof(float)*m->allng*m->NT)
        if (!state) memset((void*)m->rz[0], 0, sizeof(float)*m->allng*m->NT);
        if (!state){
            for (i=1; i<m->ns; i++){
                m->rz[i]=m->rz[i-1]+m->nrec[i-1]*m->NT;
            }
        }
        
        if (m->MYID!=0){
            
            GMALLOC(m->vz0,sizeof(float*)*m->ns)
            if (!state) memset((void*)m->vz0, 0, sizeof(float*)*m->ns);
            GMALLOC(m->vz0[0],sizeof(float)*m->allng*m->NT)
            if (!state) memset((void*)m->vz0[0], 0, sizeof(float)*m->allng*m->NT);
            if (!state){
                for (i=1; i<m->ns; i++){
                    m->vz0[i]=m->vz0[i-1]+m->nrec[i-1]*m->NT;
                }
            }
        }
        
        if (!state){
            MPI_Bcast( m->vz0[0], m->allng*m->NT, MPI_FLOAT, 0, MPI_COMM_WORLD );
        }
    }
    

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