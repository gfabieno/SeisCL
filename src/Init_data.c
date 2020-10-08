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

int var_alloc_out(float *** var, model *m ){
    int state=0;
    int i;
    
    GMALLOC(*var,sizeof(float*)*m->src_recs.ns);
    GMALLOC((*var)[0],sizeof(float)*m->src_recs.allng*m->NT);
    if (!state){
        for (i=1; i<m->src_recs.ns; i++){
            (*var)[i]=(*var)[i-1]+m->src_recs.nrec[i-1]*m->NT;
        }
    }
    
    
    return state;
    
}

int Init_data(model * m) {
    
    int state=0, i;
    
    if (m->GRADOUT==1){
        GMALLOC(m->src_recs.res_scales, sizeof(int)*m->src_recs.ns);
    }
    if (m->restype==0){
        m->res_calc = &var_res_raw;
        m->res_scale = &res_scale;
    }
    else if (m->restype==1){
        m->res_calc = &rtm_res;
        m->res_scale = &res_scale;
    }
    else{
        fprintf(stderr, "Error: Unknown restype\n");
        return 1;
    }
    
    //Allocate memory of variable outputs
    for (i=0;i<m->nvars;i++){
        if (m->vars[i].to_output){
            var_alloc_out(&m->vars[i].gl_varout, m);
            if (m->MOVOUT>0){
                GMALLOC(m->vars[i].gl_mov,sizeof(float)*
                        m->src_recs.ns*m->vars[i].num_ele*m->NT/m->MOVOUT);
            }
        }
    }

    for (i=0;i<m->ntvars;i++){
        if (m->trans_vars[i].to_output){
            var_alloc_out(&m->trans_vars[i].gl_varout, m);
        }
    }
    
    if (m->GRADSRCOUT==1){
        GMALLOC(m->src_recs.gradsrc,sizeof(float*)*m->src_recs.ns);
        GMALLOC(m->src_recs.gradsrc[0],sizeof(float)*m->src_recs.allns*m->NT);
        for (i=1;i<m->src_recs.ns;i++){
            m->src_recs.gradsrc[i]=m->src_recs.gradsrc[i-1]
            +m->src_recs.nsrc[i-1]*m->NT;
        }
    }
    
    
    return state;
}
