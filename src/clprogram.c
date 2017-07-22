//
//  clprogram.c
//  SeisCL
//
//  Created by Gabriel Fabien-Ouellet on 17-07-10.
//  Copyright © 2017 Gabriel Fabien-Ouellet. All rights reserved.
//

#include "clprogram.h"


int prog_source(struct clprogram * prog, char* name, const char * source){
    int state =0;
    (*prog).src=source;
    (*prog).name=name;
    state =extract_args(source,
                        (char *)(*prog).name ,
                        &(*prog).input_list,
                        &(*prog).ninputs);

    return state;
}

char *get_build_options(struct varcl *vcl,
                        struct modcsts *m,
                        int LCOMM,
                        int comm,
                        int DIRPROP)
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
    sprintf(src2,"-D NDIM=%d -D OFFSET=%d -D FDOH=%d -D DTDH=%9.9f -D DH=%9.9f "
                 "-D DT=%9.9f -D DT2=%9.9f -D NT=%d -D NAB=%d -D NBND=%d "
                 "-D LOCAL_OFF=%d -D LVE=%d -D DEV=%d -D NUM_DEVICES=%d "
                 "-D ND=%d -D ABS_TYPE=%d -D FREESURF=%d -D LCOMM=%d "
                 "-D MYLOCALID=%d -D NLOCALP=%d -D NFREQS=%d "
                 "-D BACK_PROP_TYPE=%d -D COMM12=%d -D NTNYQ=%d -D DTNYQ=%d "
                 "-D VARSOUT=%d -D RESOUT=%d  -D RMSOUT=%d -D MOVOUT=%d "
                 "-D GRADOUT=%d -D HOUT=%d -D GRADSRCOUT=%d -D DIRPROP=%d",
                 (*m).NDIM, (*vcl).NX0, (*m).FDOH, (*m).dt/(*m).dh, (*m).dh,
                 (*m).dt, (*m).dt/2.0, (*m).NT, (*m).NAB, (*vcl).NBND,
                 (*vcl).LOCAL_OFF, (*m).L, (*vcl).DEV, (*m).NUM_DEVICES,
                 (*m).ND, (*m).ABS_TYPE, (*m).FREESURF, LCOMM,
                 (*m).MYLOCALID, (*m).NLOCALP, (*m).NFREQS,
                 (*m).BACK_PROP_TYPE, comm, (*m).NTNYQ, (*m).DTNYQ,
                 (*m).VARSOUT, (*m).RESOUT, (*m).RMSOUT, (*m).MOVOUT,
                 (*m).GRADOUT, (*m).HOUT, (*m).GRADSRCOUT, DIRPROP  );
    
    strcat(build_options,src2);
    
    //Make it all uppercase
    char *s = build_options;
    while (*s) {
        *s = toupper((unsigned char) *s);
        s++;
    }
    
    return build_options;
}

int create_kernel(struct modcsts * m,
                  struct varcl * vcl,
                  struct clprogram * prog){
    
    cl_int cl_err = 0;
    int i,j, argfound;
    size_t shared_size=sizeof(float);
    char str2comp[50];
    
    const char * build_options = get_build_options(vcl,
                                                   m,
                                                   prog->LCOMM,
                                                   prog->COMM,
                                                   prog->DIRPROP);
    cl_err = create_gpu_kernel_from_string( (*prog).src,
                                           &(*prog).prog,
                                           &m->context,
                                           &(*prog).kernel,
                                           (*prog).name,
                                           build_options);
    if (cl_err){
        printf("Error: Could not build kernel %s\n", (*prog).name);
        
        return cl_err;
    }
    
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
        for (j=0;j<m->npars;j++){
            if (strcmp((*vcl).pars[j].name,(*prog).input_list[i])==0){
                cl_err = clSetKernelArg((*prog).kernel,
                                        i, sizeof(cl_mem),
                                        &(*vcl).pars[j].cl_par.mem);
                argfound=1;

                break;
            }
        }
        if (!argfound){
            for (j=0;j<m->nvars;j++){
                if (strcmp((*vcl).vars[j].name,(*prog).input_list[i])==0){
                    cl_err = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*vcl).vars[j].cl_var.mem);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%sout",(*vcl).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    cl_err = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*vcl).vars[j].cl_varout.mem);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%sbnd",(*vcl).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    cl_err = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*vcl).vars[j].cl_varbnd.mem);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"f%s",(*vcl).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    cl_err = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*vcl).vars[j].cl_fvar.mem);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%s_buf1",(*vcl).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    cl_err = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*vcl).vars[j].cl_buf1.mem);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%s_buf2",(*vcl).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    cl_err = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*vcl).vars[j].cl_buf2.mem);
                    argfound=1;
                    break;
                }
            }
        }
        if (!argfound && m->vars_adj){
            for (j=0;j<m->nvars;j++){
                sprintf(str2comp,"%s_r",(*vcl).vars_adj[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    cl_err = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*vcl).vars_adj[j].cl_var.mem);
                    argfound=1;

                    break;
                }
            }
        }
        if (!argfound){
            for (j=0;j<m->ncsts;j++){
                if (strcmp((*vcl).csts[j].name,(*prog).input_list[i])==0){
                    cl_err = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*vcl).csts[j].cl_cst.mem);
                    argfound=1;

                    break;
                }
            }
        }
        if (!argfound){
            if (strcmp("src" ,(*prog).input_list[i])==0){
                cl_err = clSetKernelArg((*prog).kernel,
                                        i,
                                        sizeof(cl_mem),
                                        &(*vcl).src_recs.cl_src.mem);
                argfound=1;
            }
            else if (strcmp("src_pos" ,(*prog).input_list[i])==0){
                cl_err = clSetKernelArg((*prog).kernel,
                                        i,
                                        sizeof(cl_mem),
                                        &(*vcl).src_recs.cl_src_pos.mem);
                argfound=1;
            }
            else if (strcmp("rec_pos" ,(*prog).input_list[i])==0){
                cl_err = clSetKernelArg((*prog).kernel,
                                        i,
                                        sizeof(cl_mem),
                                        &(*vcl).src_recs.cl_rec_pos.mem);
                argfound=1;
            }
            else if (strcmp("grad_src" ,(*prog).input_list[i])==0){
                cl_err = clSetKernelArg((*prog).kernel,
                                        i,
                                        sizeof(cl_mem),
                                        &(*vcl).src_recs.cl_grad_src.mem);
                argfound=1;
            }
        }
        
        
        
        if (!argfound){
            if (strcmp("offcomm",(*prog).input_list[i])==0){
                cl_err = clSetKernelArg((*prog).kernel,
                                        i,
                                        sizeof(int),
                                        &prog->OFFCOMM);
                argfound=1;
                //                printf("%s\n",(*prog).input_list[i]);
            }
        }
        if (strcmp("nt"  ,(*prog).input_list[i])==0)
            prog->tinput=i+1;
        
        if (!argfound
            && strcmp("lvar",(*prog).input_list[i])!=0
            && strcmp("nt"  ,(*prog).input_list[i])!=0){
            printf("Error: input %s undefined for kernel %s\n\n",
                             (*prog).input_list[i], (*prog).name);
            printf("Input list: \n\n");
            for (j=0;j<(*prog).ninputs;j++){
                printf("%s\n",(*prog).input_list[j]);
            }
            printf("\n\nKernel: \n\n");
            printf("%s\n",(*prog).src);
            cl_err=1;
            
        }

    }
    

    vcl->progs[vcl->nprogs]=prog;
    vcl->nprogs++;
    
    return cl_err;
}
