//
//  clprogram.c
//  SeisCL
//
//  Created by Gabriel Fabien-Ouellet on 17-07-10.
//  Copyright © 2017 Gabriel Fabien-Ouellet. All rights reserved.
//

#include "clprogram.h"


int assign_prog_source(struct clprogram * prog, char* name, const char * source){
    int state =0;
    (*prog).src=source;
    (*prog).name=name;
    state =extract_args(source, (char *)(*prog).name , &(*prog).input_list, &(*prog).ninputs);

    return state;
}

char *get_build_options(struct varcl *vcl, struct modcsts *m, int LCOMM, int comm, int DIRPROP)
{
    int i;
    static char build_options [6000]={0};
    char src[50];
    
    build_options[0]=0;
    if (m->N_names[0]){
        for (i=0;i<m->NDIM;i++){
            sprintf(src,"-D N%s=%d ",m->N_names[i],(*vcl).N[i]+m->FDORDER);
            strcat(build_options,src);
            
        }
    }
    else{
        for (i=0;i<m->NDIM;i++){
            sprintf(src,"-D N%d=%d ",i,(*vcl).N[i]+m->FDORDER);
            strcat(build_options,src);
            
        }
    }
    for (i=0;i<m->FDOH;i++){
        sprintf(src,"-D hc%d=%9.9f ",i+1,m->hc[i]);
        strcat(build_options,src);
        
    }

    char src2[2000];
    sprintf(src2, "-D NDIM=%d -D OFFSET=%d -D FDOH=%d -D DTDH=%9.9f -D DH=%9.9f -D DT=%9.9f -D DT2=%9.9f -D NT=%d -D NAB=%d -D NBND=%d -D LOCAL_OFF=%d -D LVE=%d -D DEV=%d -D NUM_DEVICES=%d -D ND=%d -D ABS_TYPE=%d -D FREESURF=%d -D LCOMM=%d -D MYLOCALID=%d -D NLOCALP=%d -D NFREQS=%d -D BACK_PROP_TYPE=%d -D COMM12=%d -D NTNYQ=%d -D DTNYQ=%d -D SEISOUT=%d -D RESOUT=%d  -D RMSOUT=%d -D MOVOUT=%d -D GRADOUT=%d -D HOUT=%d -D GRADSRCOUT=%d -D DIRPROP=%d", (*m).NDIM, (*vcl).NX0, (*m).FDOH, (*m).dt/(*m).dh, (*m).dh, (*m).dt, (*m).dt/2.0, (*m).NT, (*m).NAB, (*vcl).NBND, (*vcl).LOCAL_OFF, (*m).L, (*vcl).DEV, (*m).NUM_DEVICES,(*m).ND, (*m).ABS_TYPE, (*m).FREESURF, LCOMM, (*m).MYLOCALID, (*m).NLOCALP, (*m).NFREQS, (*m).BACK_PROP_TYPE, comm, (*m).NTNYQ, (*m).DTNYQ, (*m).SEISOUT, (*m).RESOUT, (*m).RMSOUT, (*m).MOVOUT, (*m).GRADOUT, (*m).HOUT, (*m).GRADSRCOUT, DIRPROP  );
    strcat(build_options,src2);
    
    //Make it all uppercase
    char *s = build_options;
    while (*s) {
        *s = toupper((unsigned char) *s);
        s++;
    }
    
    return build_options;
}

int gpu_initialize_kernel(struct modcsts * m, struct varcl * vcl,  struct clprogram * prog, int offcomm, int LCOMM, int comm, int DIRPROP){
    
    cl_int cl_err = 0;
    int i,j, argfound;
    size_t shared_size=sizeof(float);
    char adjvar[50];
    
    const char * build_options = get_build_options(vcl, m, LCOMM, comm, DIRPROP);
    cl_err = create_gpu_kernel_from_string( (*prog).src, &(*prog).prog, &m->context, &(*prog).kernel, (*prog).name, build_options);
    
    /*Define the size of the local variables of the compute device*/
    if (   (*prog).local==1 ){
        for (i=0;i<m->NDIM;i++){
            shared_size*=(*prog).lsize[i]+m->FDORDER;
        }
    }
    
    /*Define the arguments for this kernel */
    
    for (i=0;i<(*prog).ninputs;i++){
        argfound=0;
        //        printf("%s\n",(*prog).input_list[i]);
        for (j=0;j<m->nparams;j++){
            if (strcmp((*vcl).params[j].name,(*prog).input_list[i])==0){
                cl_err = clSetKernelArg((*prog).kernel,  i, sizeof(cl_mem), &(*vcl).params[j].cl_param.mem);
                argfound=1;

                break;
            }
        }
        if (!argfound){
            for (j=0;j<m->nvars;j++){
                if (strcmp((*vcl).vars[j].name,(*prog).input_list[i])==0){
                    cl_err = clSetKernelArg((*prog).kernel,  i, sizeof(cl_mem), &(*vcl).vars[j].cl_var.mem);
                    argfound=1;

                    break;
                }
            }
        }
        if (!argfound && m->vars_adj){
            for (j=0;j<m->nvars;j++){
                sprintf(adjvar,"%s_r",(*vcl).vars_adj[j].name);
                if (strcmp(adjvar,(*prog).input_list[i])==0){
                    cl_err = clSetKernelArg((*prog).kernel,  i, sizeof(cl_mem), &(*vcl).vars_adj[j].cl_var.mem);
                    argfound=1;

                    break;
                }
            }
        }
        if (!argfound){
            for (j=0;j<m->ncsts;j++){
                if (strcmp((*vcl).csts[j].name,(*prog).input_list[i])==0){
                    cl_err = clSetKernelArg((*prog).kernel,  i, sizeof(cl_mem), &(*vcl).csts[j].cl_cst.mem);
                    argfound=1;

                    break;
                }
            }
        }
        if (!argfound){
            if (strcmp("offcomm",(*prog).input_list[i])==0){
                cl_err = clSetKernelArg((*prog).kernel,  i, sizeof(int), &offcomm);
                //                printf("%s\n",(*prog).input_list[i]);
            }
        }
        
    }
    
    return cl_err;
}
