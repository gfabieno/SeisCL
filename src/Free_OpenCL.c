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

void clbuf_free(clbuf *buf){

    if (buf->mem) cuMemFree(buf->mem);
//    if (buf->pin) cuMemFreeHost(buf->pin);
//    else if (buf->free_host){
//         GFree(buf->host);
//    }

}

void clprogram_freeCL(clprogram * prog){
    
    if (prog->module) cuModuleUnload(prog->module);
    if (prog->prog) free(prog->prog);
    
}

void clprogram_freeGL(clprogram * prog){
    int i;
    if (prog->input_list){
        for (i=0;i<prog->ninputs;i++){
            GFree(prog->input_list[i]);
        }
        GFree(prog->input_list);
    }
}

void variable_freeCL(variable * var){
    
    fprintf(stdout,"freeing %s \n", var->name );
    clbuf_free(&var->cl_var);
    clbuf_free(&var->cl_varout);
    clbuf_free(&var->cl_varbnd);
    clbuf_free(&var->cl_fvar);
    clbuf_free(&var->cl_fvar_adj);
    clbuf_free(&var->cl_buf1);
    clbuf_free(&var->cl_buf2);
    clbuf_free(&var->cl_var_res);
    
}

void variable_freeGL(variable * var){


    GFree(var->gl_varout);
    if (var->gl_varin){
        GFree(var->gl_varin[0]);
    }
    GFree(var->gl_varin);
    if (var->gl_var_res){
        GFree(var->gl_var_res[0]);
    }
    GFree(var->gl_var_res);
    GFree(var->gl_mov);
    GFree(var->var2ave);
    
}

void parameter_freeCL(parameter * par){
    
    clbuf_free(&par->cl_par);
    clbuf_free(&par->cl_grad);
    clbuf_free(&par->cl_H);
}

void parameter_freeGL(parameter * par){
    
    GFree(par->gl_par);
    GFree(par->gl_grad);
    GFree(par->gl_H);
}

void constants_freeCL(constants *cst){
    
    clbuf_free(&cst->cl_cst);
}

void constants_freeGL(constants *cst){
    
    GFree(cst->gl_cst);
}

void sources_records_freeCL(sources_records * src_rec){
    
    clbuf_free(&src_rec->cl_src);
    clbuf_free(&src_rec->cl_src_pos);
    clbuf_free(&src_rec->cl_rec_pos);
    clbuf_free(&src_rec->cl_grad_src);
    
    clprogram_freeCL(&src_rec->sources);
    clprogram_freeCL(&src_rec->varsout);
    clprogram_freeCL(&src_rec->varsoutinit);
    clprogram_freeCL(&src_rec->residuals);
    clprogram_freeCL(&src_rec->init_gradsrc);
    
}

void sources_records_freeGL(sources_records * src_rec){
    
   
    clprogram_freeGL(&src_rec->sources);
    clprogram_freeGL(&src_rec->varsout);
    clprogram_freeGL(&src_rec->varsoutinit);
    clprogram_freeGL(&src_rec->residuals);
    clprogram_freeGL(&src_rec->init_gradsrc);
    
    GFree(src_rec->nsrc);
    GFree(src_rec->nrec);
    if  (src_rec->src){
        GFree(src_rec->src[0]);
    }
    GFree(src_rec->src);
    if  (src_rec->gradsrc){
        GFree(src_rec->gradsrc[0]);
    }
    GFree(src_rec->gradsrc);
    if  (src_rec->src_pos){
        GFree(src_rec->src_pos[0]);
    }
    GFree(src_rec->src_pos);
    if  (src_rec->rec_pos){
        GFree(src_rec->rec_pos[0]);
    }
    GFree(src_rec->rec_pos);
    GFree(src_rec->src_scales);
    GFree(src_rec->res_scales);
    
}

void update_freeCL(update * up){
    
    clprogram_freeCL(&up->center);
    clprogram_freeCL(&up->com1);
    clprogram_freeCL(&up->com2);
    clprogram_freeCL(&up->fcom1_out);
    clprogram_freeCL(&up->fcom2_out);
    clprogram_freeCL(&up->fcom1_in);
    clprogram_freeCL(&up->fcom2_out);
    GFree(up->v2com)
}

void update_freeGL(update * up){
    
    clprogram_freeGL(&up->center);
    clprogram_freeGL(&up->com1);
    clprogram_freeGL(&up->com2);
    clprogram_freeGL(&up->fcom1_out);
    clprogram_freeGL(&up->fcom2_out);
    clprogram_freeGL(&up->fcom1_in);
    clprogram_freeGL(&up->fcom2_out);
    GFree(up->v2com)
}

void boundary_conditions_freeCL(boundary_conditions * bnd){
    
    clprogram_freeCL(&bnd->surf);
    clprogram_freeCL(&bnd->init_f);
    clprogram_freeCL(&bnd->init_adj);

}

void boundary_conditions_freeGL(boundary_conditions * bnd){
    
    clprogram_freeGL(&bnd->surf);
    clprogram_freeGL(&bnd->init_f);
    clprogram_freeGL(&bnd->init_adj);
    
}

void gradients_freeCL(gradients * grad){
    
    clprogram_freeCL(&grad->init);
    clprogram_freeCL(&grad->savefreqs);
    clprogram_freeCL(&grad->initsavefreqs);
    clprogram_freeCL(&grad->savebnd);
    
}

void gradients_freeGL(gradients * grad){
    
    clprogram_freeGL(&grad->init);
    clprogram_freeGL(&grad->savefreqs);
    clprogram_freeGL(&grad->initsavefreqs);
    clprogram_freeGL(&grad->savebnd);
    
}

void device_free(device * dev){
    int i;
    
    if (dev->vars){
        for (i=0;i<dev->nvars;i++){
            variable_freeCL(&(dev->vars[i]));
        }
        GFree(dev->vars);
    }
    if (dev->vars_adj){
        for (i=0;i<dev->nvars;i++){
            variable_freeCL(&dev->vars_adj[i]);
        }
        GFree(dev->vars_adj);
    }
    if (dev->pars){
        for (i=0;i<dev->npars;i++){
            parameter_freeCL(&dev->pars[i]);
        }
        GFree(dev->pars);
    }
    if (dev->csts){
        for (i=0;i<dev->ncsts;i++){
            constants_freeCL(&dev->csts[i]);
        }
        GFree(dev->csts);
    }
    if (dev->trans_vars){
        for (i=0;i<dev->ntvars;i++){
            variable_freeCL(&dev->trans_vars[i]);
        }
        GFree(dev->trans_vars);
    }
    if (dev->ups_f){
        for (i=0;i<dev->nupdates;i++){
            update_freeCL(&dev->ups_f[i]);
        }
        GFree(dev->ups_f);
    }
    if (dev->ups_adj){
        for (i=0;i<dev->nupdates;i++){
            update_freeCL(&dev->ups_adj[i]);
        }
        GFree(dev->ups_adj);
    }
    
    sources_records_freeCL(&dev->src_recs);
    gradients_freeCL(&dev->grads);
    boundary_conditions_freeCL(&dev->bnd_cnds);
    
    if (dev->cuda_null) cuMemFree(dev->cuda_null);
    if (dev->queue) cuStreamDestroy(dev->queue);
    if (dev->queuecomm) cuStreamDestroy(dev->queuecomm);
    if (dev->context) cuCtxDestroy(dev->context);
    
}

void model_free(model * m){
    int i;
    device * dev = NULL;
    if (m->vars){
        for (i=0;i<m->nvars;i++){
            variable_freeGL(&m->vars[i]);
        }
        GFree(m->vars);
    }
    if (m->vars_adj){
        for (i=0;i<m->nvars;i++){
            variable_freeGL(&m->vars_adj[i]);
        }
        GFree(m->vars_adj);
    }
    if (m->pars){
        for (i=0;i<m->npars;i++){
            parameter_freeGL(&m->pars[i]);
        }
        GFree(m->pars);
    }
    if (m->csts){
        for (i=0;i<m->ncsts;i++){
            constants_freeGL(&m->csts[i]);
        }
        GFree(m->csts);
    }
    if (m->trans_vars){
        for (i=0;i<m->ntvars;i++){
            variable_freeGL(&m->trans_vars[i]);
        }
        GFree(m->trans_vars);
    }
    if (m->ups_f){
        for (i=0;i<m->nupdates;i++){
            update_freeGL(&m->ups_f[i]);
        }
        GFree(m->ups_f);
    }
    if (m->ups_adj){
        for (i=0;i<m->nupdates;i++){
            update_freeGL(&m->ups_adj[i]);
        }
        GFree(m->ups_adj);
    }

    sources_records_freeGL(&m->src_recs);
    gradients_freeGL(&m->grads);
    boundary_conditions_freeGL(&m->bnd_cnds);
    
    GFree(m->no_use_GPUs);
    
//    if (m->context) clReleaseContext(m->context);
    
}

int Free_OpenCL(model * m, device ** dev)  {
    // Free all memory contained in all structures listed in F.h
    int d;
    if (*dev){
        for (d=0;d<m->NUM_DEVICES;d++){
            device_free(dev[d]);
        }
    }
    GFree(*dev);
    model_free(m);
    
    return 0;
}
