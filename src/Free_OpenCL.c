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

void clbuf_free(device *dev, clbuf *buf){
    
    if (buf->pin) clEnqueueUnmapMemObject(dev->queuecomm,
                                          buf->pin,
                                          buf->host,
                                          0,
                                          NULL,
                                          NULL);
    if (buf->pin) clReleaseMemObject(buf->pin);
    if (buf->mem) clReleaseMemObject(buf->mem);
}

void clprogram_free(clprogram * prog){
    int i;
    if (prog->kernel ) clReleaseKernel(prog->kernel);
    if (prog->prog) clReleaseProgram(prog->prog);
    if (prog->input_list){
        for (i=0;i<prog->ninputs;i++){
            GFree(prog->input_list[i]);
        }
        free(prog->input_list);
    }
}

void variable_free(device *dev, variable * var){
    
    clbuf_free(dev, &var->cl_var);
    clbuf_free(dev, &var->cl_varout);
    clbuf_free(dev, &var->cl_varbnd);
    clbuf_free(dev, &var->cl_fvar);
    clbuf_free(dev, &var->cl_buf1);
    clbuf_free(dev, &var->cl_buf2);
    clbuf_free(dev, &var->cl_var_res);
    
    GFree(var->gl_fvar);
    if  (var->gl_varout) GFree(var->gl_varout[0]);
    GFree(var->gl_varout);
    if  (var->gl_varin) GFree(var->gl_varin[0]);
    GFree(var->gl_varin);
    GFree(var->gl_var_res);
    if  (var->gl_var_res) GFree(var->gl_var_res[0]);
    GFree(var->gl_mov);
    
}

void parameter_free(device *dev, parameter * par){
    
    clbuf_free(dev, &par->cl_par);
    clbuf_free(dev, &par->cl_grad);
    clbuf_free(dev, &par->cl_H);
    GFree(par->gl_par);
    GFree(par->gl_grad);
    GFree(par->gl_H);
}

void constants_free(device *dev, constants *cst){
    
    clbuf_free(dev, &cst->cl_cst);
    GFree(cst->gl_cst);
}

void sources_records_free(device *dev, sources_records * src_rec){
    
    clbuf_free(dev, &src_rec->cl_src);
    clbuf_free(dev, &src_rec->cl_src_pos);
    clbuf_free(dev, &src_rec->cl_rec_pos);
    clbuf_free(dev, &src_rec->cl_grad_src);
    
    clprogram_free(&src_rec->sources);
    clprogram_free(&src_rec->varsout);
    clprogram_free(&src_rec->varsoutinit);
    clprogram_free(&src_rec->residuals);
    clprogram_free(&src_rec->init_gradsrc);
    
    GFree(src_rec->nsrc);
    GFree(src_rec->nrec);
    if  (src_rec->src) GFree(src_rec->src[0]);
    GFree(src_rec->src);
    if  (src_rec->gradsrc) GFree(src_rec->gradsrc[0]);
    GFree(src_rec->gradsrc);
    if  (src_rec->src_pos) GFree(src_rec->src_pos[0]);
    GFree(src_rec->src_pos);
    if  (src_rec->rec_pos) GFree(src_rec->rec_pos[0]);
    GFree(src_rec->rec_pos);
    
}

void update_free(update * up){
    
    clprogram_free(&up->center);
    clprogram_free(&up->com1);
    clprogram_free(&up->com2);
    clprogram_free(&up->fcom1_out);
    clprogram_free(&up->fcom2_out);
    clprogram_free(&up->fcom1_in);
    clprogram_free(&up->fcom2_out);
    GFree(up->v2com)
}

void boundary_conditions_free(boundary_conditions * bnd){
    
    clprogram_free(&bnd->surf);
    clprogram_free(&bnd->init_f);
    clprogram_free(&bnd->init_adj);

}

void gradients_free(gradients * grad){
    
    clprogram_free(&grad->init);
    clprogram_free(&grad->savefreqs);
    clprogram_free(&grad->initsavefreqs);
    clprogram_free(&grad->savebnd);
    
}

void varcl_free(device * dev){
    int i;
    
    if (dev->vars){
        for (i=0;i<dev->nvars;i++){
            variable_free(dev, &dev->vars[i]);
        }
        GFree(dev->vars);
    }
    if (dev->vars_adj){
        for (i=0;i<dev->nvars;i++){
            variable_free(dev, &dev->vars_adj[i]);
        }
        GFree(dev->vars_adj);
    }
    if (dev->pars){
        for (i=0;i<dev->npars;i++){
            parameter_free(dev, &dev->pars[i]);
        }
        GFree(dev->pars);
    }
    if (dev->csts){
        for (i=0;i<dev->ncsts;i++){
            constants_free(dev, &dev->csts[i]);
        }
        GFree(dev->csts);
    }
    if (dev->trans_vars){
        for (i=0;i<dev->ntvars;i++){
            variable_free(dev, &dev->trans_vars[i]);
        }
        GFree(dev->trans_vars);
    }
    if (dev->ups_f){
        for (i=0;i<dev->nupdates;i++){
            update_free(&dev->ups_f[i]);
        }
        GFree(dev->ups_f);
    }
    if (dev->ups_adj){
        for (i=0;i<dev->nupdates;i++){
            update_free(&dev->ups_adj[i]);
        }
        GFree(dev->ups_adj);
    }
    
    sources_records_free(dev, &dev->src_recs);
    gradients_free(&dev->grads);
    boundary_conditions_free(&dev->bnd_cnds);
    
    if (dev->queue) clReleaseCommandQueue(dev->queue);
    if (dev->queuecomm) clReleaseCommandQueue(dev->queuecomm);
    
}

void modcsts_free(model * m){
    int i;
    device * dev = NULL;
    if (m->vars){
        for (i=0;i<m->nvars;i++){
            variable_free(dev, &m->vars[i]);
        }
        GFree(m->vars);
    }
    if (m->vars_adj){
        for (i=0;i<m->nvars;i++){
            variable_free(dev, &m->vars_adj[i]);
        }
        GFree(m->vars_adj);
    }
    if (m->pars){
        for (i=0;i<m->npars;i++){
            parameter_free(dev, &m->pars[i]);
        }
        GFree(m->pars);
    }
    if (m->csts){
        for (i=0;i<m->ncsts;i++){
            constants_free(dev, &m->csts[i]);
        }
        GFree(m->csts);
    }
    if (m->trans_vars){
        for (i=0;i<m->ntvars;i++){
            variable_free(dev, &m->trans_vars[i]);
        }
        GFree(m->trans_vars);
    }
    if (m->ups_f){
        for (i=0;i<m->nupdates;i++){
            update_free(&m->ups_f[i]);
        }
        GFree(m->ups_f);
    }
    if (m->ups_adj){
        for (i=0;i<m->nupdates;i++){
            update_free(&m->ups_adj[i]);
        }
        GFree(m->ups_adj);
    }

    sources_records_free(dev, &m->src_recs);
    gradients_free(&m->grads);
    boundary_conditions_free(&m->bnd_cnds);
    
    GFree(m->no_use_GPUs);
    
    if (m->context) clReleaseContext(m->context);
    
}

int Free_OpenCL(model * m, device ** dev)  {
    // Free all memory contained in all structures listed in F.h
    int d;
    if (*dev){
        for (d=0;d<m->NUM_DEVICES;d++){
            varcl_free(dev[d]);
        }
    }
    GFree(*dev);
    modcsts_free(m);
    
    return 0;
}
