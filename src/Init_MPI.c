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
    
    GMALLOC(*var,sizeof(float*)*ns)
    if (!state) memset((void*)*var, 0, sizeof(float*)*ns);
    GMALLOC((*var)[0],sizeof(float)*allng*NT)
    if (!state) memset((void*)(*var)[0], 0, sizeof(float)*allng*NT);
    if (!state){
        for (int i=1; i<ns; i++){
            (*var)[i]=(*var)[i-1]+nrec[i-1]*NT;
        }
    }

    return state;
    
}


int Init_MPI(struct modcsts * m) {

    int state=0;
    int i;

    //Communicate constants
    if (!state){
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast( &m->numdim, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast( m->N, m->numdim, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->NT, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->FDORDER, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->fdoh, 1, MPI_INT, 0, MPI_COMM_WORLD );
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
        MPI_Bcast( &m->scalerms, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->scaleshot, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->scalermsnorm, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->back_prop_type, 1, MPI_INT, 0, MPI_COMM_WORLD );
        MPI_Bcast( &m->nfreqs, 1, MPI_INT, 0, MPI_COMM_WORLD );
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    //Communicate vectors
    {
        if (m->n_no_use_GPUs>0){
            if (m->MYID!=0){
                GMALLOC(m->no_use_GPUs,sizeof(int)*m->n_no_use_GPUs)
            }
            
            if (!state){
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Bcast( m->no_use_GPUs, m->n_no_use_GPUs, MPI_INT, 0, MPI_COMM_WORLD );
            }
        }
        
        
        if (m->MYID!=0){
            GMALLOC(m->src_recs.nsrc,sizeof(int)*m->ns)
            GMALLOC(m->src_recs.src_pos,sizeof(float*)*m->ns)
            if (!state) memset((void*)m->src_recs.src_pos, 0, sizeof(float*)*m->ns);
            GMALLOC(m->src_recs.src_pos[0],sizeof(float)*m->allns*5)
            
            GMALLOC(m->src_recs.src,sizeof(float*)*m->ns)
            if (!state) memset((void*)m->src_recs.src, 0, sizeof(float*)*m->ns);
            GMALLOC(m->src_recs.src[0],sizeof(float)*m->allns*m->NT)
            
            GMALLOC(m->src_recs.nrec,sizeof(int)*m->ns)
            GMALLOC(m->src_recs.rec_pos,sizeof(float*)*m->ns)
            if (!state) memset((void*)m->src_recs.rec_pos, 0, sizeof(float*)*m->ns);
            GMALLOC(m->src_recs.rec_pos[0],sizeof(float)*m->allng*8)
        }
        
        if (!state){
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast( m->src_recs.nsrc, m->ns, MPI_INT, 0, MPI_COMM_WORLD );
            MPI_Bcast( m->src_recs.src_pos[0], m->allns*5, MPI_FLOAT, 0, MPI_COMM_WORLD );
            MPI_Bcast( m->src_recs.src[0], m->allns*m->NT,  MPI_FLOAT, 0, MPI_COMM_WORLD );
            MPI_Bcast( m->src_recs.nrec, m->ns, MPI_INT, 0, MPI_COMM_WORLD );
            MPI_Bcast( m->src_recs.rec_pos[0], m->allng*8,  MPI_FLOAT, 0, MPI_COMM_WORLD );
            MPI_Barrier(MPI_COMM_WORLD);
        }
        
        if (m->MYID!=0){
            for (i=1; i<m->ns; i++){
                m->src_recs.src_pos[i]=m->src_recs.src_pos[i-1]+m->src_recs.nsrc[i-1]*5;
                m->src_recs.src[i]=m->src_recs.src[i-1]+m->src_recs.nsrc[i-1]*m->NT;
                m->src_recs.rec_pos[i]=m->src_recs.rec_pos[i-1]+m->src_recs.nrec[i-1]*8;
            }
        }
        
    }
    
    //Communicate parameters and variables
    if (m->MYID!=0){
        __GUARD assign_modeling_case(m);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (!state){
        for (i=0;i<m->nvars;i++){
            MPI_Bcast( &m->vars[i].to_output, 1, MPI_INT, 0, MPI_COMM_WORLD );
        }
        for (i=0;i<m->ntvars;i++){
            MPI_Bcast( &m->trans_vars[i].to_output, 1, MPI_INT, 0, MPI_COMM_WORLD );
        }
        
        for (i=0;i<m->nparams;i++){
            MPI_Bcast( m->params[i].gl_param, m->params[i].num_ele, MPI_FLOAT, 0, MPI_COMM_WORLD );
        }
        
        for (i=0;i<m->ncsts;i++){
            if (m->csts[i].active)
                MPI_Bcast( m->csts[i].gl_cst, m->csts[i].num_ele, MPI_FLOAT, 0, MPI_COMM_WORLD );
        }
        
    }
    

    //Allocate and broadcast the data in
    if (m->rmsout==1 || m->resout==1 || m->gradout==1){
        for (i=0;i<m->nvars;i++){
            if (m->vars[i].to_output){
                alloc_seismo(&m->vars[i].gl_var_res, m->ns, m->allng, m->NT, m->src_recs.nrec);
                if (m->MYID!=0){
                    alloc_seismo(&m->vars[i].gl_varin, m->ns, m->allng, m->NT, m->src_recs.nrec);
                }
                if (!state){
                    MPI_Bcast( m->vars[i].gl_varin[0], m->allng*m->NT, MPI_FLOAT, 0, MPI_COMM_WORLD );
                }
            }
        }
        for (i=0;i<m->ntvars;i++){
            if (m->trans_vars[i].to_output){
                alloc_seismo(&m->trans_vars[i].gl_var_res, m->ns, m->allng, m->NT, m->src_recs.nrec);
                if (m->MYID!=0){
                    alloc_seismo(&m->trans_vars[i].gl_varin, m->ns, m->allng, m->NT, m->src_recs.nrec);
                }
                if (!state){
                    MPI_Bcast( m->trans_vars[i].gl_varin[0], m->allng*m->NT, MPI_FLOAT, 0, MPI_COMM_WORLD );
                }
            }
        }
    }

    
    //Assign a group within wich domain decomposition is performed.
    //Different groupes are assigned different sources
    {
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
    }
    
    
    MPI_Barrier(MPI_COMM_WORLD);

    return state;

}
