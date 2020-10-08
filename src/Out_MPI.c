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


/* Write output files */

#include "F.h"
int buf_reduce_float(float * var, int size, int MPI_ID){
    int state=0;
    if (MPI_ID==0){
        __GUARD MPI_Reduce(MPI_IN_PLACE,
                           var,
                           size,
                           MPI_FLOAT,
                           MPI_SUM,
                           0,
                           MPI_COMM_WORLD);
    }
    else{
        __GUARD MPI_Reduce(var,
                           var,
                           size,
                           MPI_FLOAT,
                           MPI_SUM,
                           0,
                           MPI_COMM_WORLD);
    }
    return state;
}

int buf_reduce_double(float * var, int size, int MPI_ID){
    int state=0;
    if (MPI_ID==0){
        __GUARD MPI_Reduce(MPI_IN_PLACE,
                           var,
                           size,
                           MPI_DOUBLE,
                           MPI_SUM,
                           0,
                           MPI_COMM_WORLD);
    }
    else{
        __GUARD MPI_Reduce(var,
                           var,
                           size,
                           MPI_DOUBLE,
                           MPI_SUM,
                           0,
                           MPI_COMM_WORLD);
    }
    return state;
}


int Out_MPI(model * m)  {

    int state=0;
    int i;

    // Gather the output variables from all processing elements
    if (m->VARSOUT){
        for (i=0;i<m->nvars;i++){
            if (m->vars[i].to_output){
                __GUARD buf_reduce_float(m->vars[i].gl_varout[0],
                                         m->src_recs.allng*m->NT,
                                         m->GID);
            }
        }
        for (i=0;i<m->ntvars;i++){
            if (m->trans_vars[i].to_output){
                __GUARD buf_reduce_float(m->trans_vars[i].gl_varout[0],
                                         m->src_recs.allng*m->NT,
                                         m->GID);
            }
        }
    }
    // Add the rms value of all processing elements
    if (m->RMSOUT){
        __GUARD buf_reduce_float(&m->rms,1,m->GID);
        __GUARD buf_reduce_float(&m->rmsnorm,1,m->GID);
    }
    
    // Gather the residuals from all processing elements
    if (m->RESOUT){
        for (i=0;i<m->nvars;i++){
            if (m->vars[i].to_output){
                __GUARD buf_reduce_float(m->vars[i].gl_var_res[0],
                                         m->src_recs.allng*m->NT,
                                         m->GID);
            }
        }
        for (i=0;i<m->ntvars;i++){
            if (m->trans_vars[i].to_output){
                __GUARD buf_reduce_float(m->trans_vars[i].gl_var_res[0],
                                        m->src_recs.allng*m->NT,
                                         m->GID);
            }
        }
    }

    // Gather the movie from all processing elements
    if (m->MOVOUT){
        for (i=0;i<m->nvars;i++){
            if (m->vars[i].to_output){
                __GUARD buf_reduce_float(m->vars[i].gl_mov,
                                         m->vars[i].num_ele*
                                                 m->src_recs.ns*m->NT/m->MOVOUT,
                                         m->GID);
            }
        }
    }
    
    // Gather the gradient from all processing elements
    if (m->GRADOUT==1){
        
        for (i=0;i<m->npars;i++){
            if (m->pars[i].to_grad){
                __GUARD buf_reduce_float(m->pars[i].gl_grad,
                                         m->pars[i].num_ele,
                                         m->GID);
            }
        }
        if (m->GRADSRCOUT==1){
            __GUARD buf_reduce_double(m->src_recs.gradsrc[0],
                                      m->src_recs.allns*m->NT,
                                      m->GID);
        }
        if (m->HOUT==1){
            for (i=0;i<m->npars;i++){
                if (m->pars[i].to_grad){
                    __GUARD buf_reduce_float(m->pars[i].gl_H,
                                              m->pars[i].num_ele,
                                              m->GID);
                }
            }
        }
    }

    MPI_Comm_free(&m->mpigroupcomm);
    MPI_Barrier(MPI_COMM_WORLD);

    return state;

}
