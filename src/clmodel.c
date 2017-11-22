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



int append_cst(model * m,
               const char * name,
               const char * to_read,
               int num_ele,
               void (*transform)(void *, void *, int)){
    
    int state=0;
    int ind = m->ncsts;
    m->csts[ind].name=name;
    m->csts[ind].to_read=to_read;
    m->csts[ind].num_ele=num_ele;
    m->csts[ind].transform=transform;
    
    
    //Allocate memory of parameters
    GMALLOC(m->csts[ind].gl_cst, sizeof(float)*m->csts[ind].num_ele);
    
    m->ncsts+=1;
    
    return state;
}
int append_par(model * m,
               int *ind,
               const char * name,
               const char * to_read,
               void (*transform)(void *)){
    
    int state=0;
    m->pars[*ind].name=name;
    m->pars[*ind].to_read=to_read;
    m->pars[*ind].transform=transform;
    int i;
    int sizepars=1;
    for (i=0;i<m->NDIM;i++){
        sizepars*=m->N[i];
    }
    m->pars[*ind].num_ele=sizepars;
    
    //Allocate memory of parameters
    GMALLOC(m->pars[*ind].gl_par, sizeof(float)*m->pars[*ind].num_ele);
    if (m->GRADOUT && m->pars[*ind].to_read){
        m->pars[*ind].to_grad=1;
        GMALLOC(m->pars[*ind].gl_grad, sizeof(double)*m->pars[*ind].num_ele);
        if (m->HOUT==1){
            GMALLOC(m->pars[*ind].gl_H, sizeof(double)*m->pars[*ind].num_ele);
        }
    }
    
    *ind+=1;
    
    return state;
}
int append_var(model * m,
               int *ind,
               const char * name,
               int for_grad,
               int to_comm,
               void (*set_size)(int* , void *, void *)){
    int state=0;
    m->vars[*ind].name=name;
    m->vars[*ind].for_grad=for_grad;
    m->vars[*ind].to_comm=to_comm;
    m->vars[*ind].set_size=set_size;
    
    //Assign the number of elements of the parameters
    m->vars[*ind].set_size(m->N, (void*) m, &m->vars[*ind]);
    
    *ind+=1;
    
    return state;
}
int append_update(update * up, int * ind, char * name, const char * source){
    int state =0;
    __GUARD prog_source(&up[*ind].center, name, source);
    __GUARD prog_source(&up[*ind].com1, name, source);
    __GUARD prog_source(&up[*ind].com2, name, source);
    *ind+=1;
    return state;
}

constants * get_cst(constants * csts, int ncsts, const char * name){
    
    int i;
    constants zerocst;
    memset(&zerocst, 0, sizeof(constants));
    constants * outptr=&zerocst;
    for (i=0;i<ncsts;i++){
        if (strcmp(csts[i].name,name)==0){
            outptr=&csts[i];
        }
    }
    
    return outptr;
    
}
variable * get_var(variable * vars, int nvars, const char * name){
    
    int i;
    variable zerovar;
    memset(&zerovar, 0, sizeof(variable));
    variable * outptr=&zerovar;
    for (i=0;i<nvars;i++){
        if (strcmp(vars[i].name,name)==0){
            outptr=&vars[i];
        }
    }
    
    return outptr;
    
}
parameter * get_par(parameter * pars, int npars, const char * name){
    
    int i;
    parameter zeropar;
    memset(&zeropar, 0, sizeof(parameter));
    parameter * outptr=&zeropar;
    for (i=0;i<npars;i++){
        if (strcmp(pars[i].name,name)==0){
            outptr=&pars[i];
        }
    }
    
    return outptr;
    
}
