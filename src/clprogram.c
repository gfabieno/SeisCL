//
//  clprogram.c
//  SeisCL
//
//  Created by Gabriel Fabien-Ouellet on 17-07-10.
//  Copyright © 2017 Gabriel Fabien-Ouellet. All rights reserved.
//

#include "F.h"


int split (const char *str, char c, char ***arr)
{
    int count = 1;
    int token_len = 1;
    int i = 0;
    char *p;
    char *t;
    
    p = (char*)str;
    while (*p != '\0')
    {
        if (*p == c)
            count++;
        p++;
    }
    
    *arr = (char**) malloc(sizeof(char*) * count);
    if (*arr == NULL)
        exit(1);
    
    p = (char*)str;
    while (*p != '\0')
    {
        if (*p == c)
        {
            (*arr)[i] = (char*) malloc( sizeof(char) * token_len );
            if ((*arr)[i] == NULL)
                exit(1);
            
            token_len = 0;
            i++;
        }
        p++;
        token_len++;
    }
    (*arr)[i] = (char*) malloc( sizeof(char) * token_len );
    if ((*arr)[i] == NULL)
        exit(1);
    
    i = 0;
    p = (char*)str;
    t = ((*arr)[i]);
    while (*p != '\0')
    {
        if (*p != c && *p != '\0')
        {
            *t = *p;
            t++;
        }
        else
        {
            *t = '\0';
            i++;
            t = ((*arr)[i]);
        }
        p++;
    }
    *t='\0';
    
    return count;
}

char * extract_name(const char *str){
    
    int state=0;
    
    char * output=NULL;
    
    int len=(int)strlen(str);
    
    char *p1=(char*)str+len-1;
    
    while (  !(isalnum(*p1) || *p1=='_') && p1>=str){
        
        p1--;
    }
    
    char *p2=p1;
    
    while (  (isalnum(*p2) || *p2=='_') && p2>=str){
        
        p2--;
    }
    p2++;
    
    GMALLOC(output, sizeof(str)*(p1-p2+1) );
    
    sprintf(output,"%.*s", (int)(p1-p2+1), p2);
    
    return output;
}

int set_args_list(const char *str, char *name, char *** argnames, int * ninputs){
    
    int state=0;
    int c = 0;
    char **arr = NULL;
    char * args = NULL;
    char del2[2] = ")";
    
    
    char del1[100];
    
    sprintf(del1,"__kernel void %s(", name);
    
    char * strbeg = strstr(str, del1);
    
    if (!strbeg){
        free(del1);
        fprintf(stderr, "Could not extract kernel arguments of %s\n",name);
        return 1;
    }
    
    char * strend = strstr(strbeg, del2);
    
    if (!strend){
        free(del1);
        fprintf(stderr, "Could not extract kernel arguments of %s\n",name);
        return 1;
    }
    
    
    args = malloc(sizeof(str)*(strend- strbeg-strlen(del1)+1 ));
    
    sprintf(args,"%.*s", (int)(strend- strbeg-strlen(del1)), strbeg + strlen(del1));
    
    
    
    c = split(args, ',', &arr);
    
    GMALLOC((*argnames), c*sizeof(char *) );
    
    
    for (int i = 0; i < c; i++){
        (*argnames)[i]=extract_name(arr[i]);
        free(arr[i]);
    }
    *ninputs=c;
    free(arr);
    free(args);
    
    return state;
}

char *get_build_options(device *dev,
                        model *m,
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
            sprintf(src,"-D N%s=%d ",m->N_names[i],(*dev).N[i]+m->FDORDER);
            strcat(build_options,src);
            
        }
    }
    else{
        for (i=0;i<m->NDIM;i++){
            sprintf(src,"-D N%d=%d ",i,(*dev).N[i]+m->FDORDER);
            strcat(build_options,src);
            
        }
    }
    for (i=0;i<m->FDOH;i++){
        sprintf(src,"-D hc%d=%9.9f ",i+1,m->hc[i+1]);
        strcat(build_options,src);
        
    }
    
    char src2[2000];
    sprintf(src2,"-D NDIM=%d -D OFFSET=%d -D FDOH=%d -D DTDH=%9.9f -D DH=%9.9f "
            "-D DT=%9.9f -D DT2=%9.9f -D NT=%d -D NAB=%d -D NBND=%d "
            "-D LOCAL_OFF=%d -D LVE=%d -D DEVID=%d -D NUM_DEVICES=%d "
            "-D ND=%d -D ABS_TYPE=%d -D FREESURF=%d -D LCOMM=%d "
            "-D MYLOCALID=%d -D NLOCALP=%d -D NFREQS=%d "
            "-D BACK_PROP_TYPE=%d -D COMM12=%d -D NTNYQ=%d -D DTNYQ=%d "
            "-D VARSOUT=%d -D RESOUT=%d  -D RMSOUT=%d -D MOVOUT=%d "
            "-D GRADOUT=%d -D HOUT=%d -D GRADSRCOUT=%d -D DIRPROP=%d",
            (*m).NDIM, (*dev).NX0, (*m).FDOH, (*m).dt/(*m).dh, (*m).dh,
            (*m).dt, (*m).dt/2.0, (*m).NT, (*m).NAB, (*dev).NBND,
            (*dev).LOCAL_OFF, (*m).L, (*dev).DEVID, (*m).NUM_DEVICES,
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

int compile(const char *program_source,
                    cl_program *program,
                    cl_context *context,
                    cl_kernel *kernel,
                    const char * program_name,
                    const char * build_options)
{
    /* Routine to build a kernel from the source file contained in a c string*/
    
    int state = 0;
    
    if (!*program){
        *program = clCreateProgramWithSource(*context,
                                             1,
                                             &program_source,
                                             NULL,
                                             &state);
        if (state !=CL_SUCCESS) fprintf(stderr,"%s\n",clerrors(state));
        
        state = clBuildProgram(*program, 0, NULL, build_options, NULL, NULL);
        if (state !=CL_SUCCESS) fprintf(stderr,"%s\n",clerrors(state));
    }
    // Now create the kernel "objects"
    *kernel = clCreateKernel(*program, program_name, &state);
    if (state !=CL_SUCCESS) fprintf(stderr,"%s\n",clerrors(state));
    
    
    return state;
    
}

int prog_source(clprogram * prog, char* name, const char * source){
    int state =0;
    (*prog).src=source;
    (*prog).name=name;
    state =set_args_list(source,
                        (char *)(*prog).name ,
                        &(*prog).input_list,
                        &(*prog).ninputs);
    
    return state;
}

int prog_create(model * m,
                device * dev,
                clprogram * prog){
    
    int state = 0;
    int i,j, argfound;
    size_t shared_size=sizeof(float);
    char str2comp[50];
    
    const char * build_options = get_build_options(dev,
                                                   m,
                                                   prog->LCOMM,
                                                   prog->COMM,
                                                   prog->DIRPROP);
    state = compile( (*prog).src,
                     &(*prog).prog,
                     &m->context,
                     &(*prog).kernel,
                     (*prog).name,
                     build_options);
    if (state){
        printf("Error: Could not build kernel %s\n", (*prog).name);
        
        return state;
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
            if (strcmp((*dev).pars[j].name,(*prog).input_list[i])==0){
                state = clSetKernelArg((*prog).kernel,
                                        i, sizeof(cl_mem),
                                        &(*dev).pars[j].cl_par.mem);
                argfound=1;

                break;
            }
        }
        if (!argfound){
            for (j=0;j<m->nvars;j++){
                if (strcmp((*dev).vars[j].name,(*prog).input_list[i])==0){
                    state = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*dev).vars[j].cl_var.mem);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%sout",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    state = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*dev).vars[j].cl_varout.mem);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%sbnd",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    state = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*dev).vars[j].cl_varbnd.mem);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"f%s",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    state = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*dev).vars[j].cl_fvar.mem);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%s_buf1",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    state = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*dev).vars[j].cl_buf1.mem);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%s_buf2",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    state = clSetKernelArg((*prog).kernel,
                                            i,
                                            sizeof(cl_mem),
                                            &(*dev).vars[j].cl_buf2.mem);
                    argfound=1;
                    break;
                }
            }
        }
        if (!argfound){
            for (j=0;j<m->nvars;j++){
                sprintf(str2comp,"%s_r",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    //TODO this is a hack for how adj kernels are written,
                    //     think of a better way
                    if (m->BACK_PROP_TYPE==1){
                        state = clSetKernelArg((*prog).kernel,
                                               i,
                                               sizeof(cl_mem),
                                               &(*dev).vars_adj[j].cl_var.mem);
                    }
                    else if (m->BACK_PROP_TYPE==2){
                        state = clSetKernelArg((*prog).kernel,
                                               i,
                                               sizeof(cl_mem),
                                               &(*dev).vars[j].cl_var.mem);
                    }
                    argfound=1;

                    break;
                }
            }
        }
        if (!argfound){
            for (j=0;j<m->ncsts;j++){
                if (strcmp((*dev).csts[j].name,(*prog).input_list[i])==0){
                    state = clSetKernelArg((*prog).kernel,
                                           i,
                                           sizeof(cl_mem),
                                           &(*dev).csts[j].cl_cst.mem);
                    argfound=1;

                    break;
                }
            }
        }
        if (!argfound){
            if (strcmp("src" ,(*prog).input_list[i])==0){
                state = clSetKernelArg((*prog).kernel,
                                        i,
                                        sizeof(cl_mem),
                                        &(*dev).src_recs.cl_src.mem);
                argfound=1;
            }
            else if (strcmp("src_pos" ,(*prog).input_list[i])==0){
                state = clSetKernelArg((*prog).kernel,
                                        i,
                                        sizeof(cl_mem),
                                        &(*dev).src_recs.cl_src_pos.mem);
                argfound=1;
            }
            else if (strcmp("rec_pos" ,(*prog).input_list[i])==0){
                state = clSetKernelArg((*prog).kernel,
                                        i,
                                        sizeof(cl_mem),
                                        &(*dev).src_recs.cl_rec_pos.mem);
                argfound=1;
            }
            else if (strcmp("grad_src" ,(*prog).input_list[i])==0){
                state = clSetKernelArg((*prog).kernel,
                                        i,
                                        sizeof(cl_mem),
                                        &(*dev).src_recs.cl_grad_src.mem);
                argfound=1;
            }
        }

        if (!argfound){
            if (strcmp("offcomm",(*prog).input_list[i])==0){
                state = clSetKernelArg((*prog).kernel,
                                        i,
                                        sizeof(int),
                                        &prog->OFFCOMM);
                argfound=1;
                //                printf("%s\n",(*prog).input_list[i]);
            }
        }
        
        if (!argfound){
            if (strcmp("lvar",(*prog).input_list[i])==0){
                state = clSetKernelArg((*prog).kernel,
                                       i,
                                       shared_size,
                                       NULL);
                argfound=1;
                //                printf("%s\n",(*prog).input_list[i]);
            }
        }
        
        if (!argfound){
            if (strcmp("nt"  ,(*prog).input_list[i])==0){
                prog->tinput=i+1;
                argfound=1;
            }
        }
        if (!argfound){
            if (strcmp("pdir"  ,(*prog).input_list[i])==0){
                prog->pdir=i+1;
                argfound=1;
            }
        }
        
        if (!argfound){
            fprintf(stdout,"Warning: input %s undefined for kernel %s\n",
                             (*prog).input_list[i], (*prog).name);
            state = clSetKernelArg((*prog).kernel,
                                   i,
                                   sizeof(cl_mem),
                                   NULL);
        }

    }
    

    dev->progs[dev->nprogs]=prog;
    dev->nprogs++;
    
    return state;
}

int prog_launch( cl_command_queue *inqueue, clprogram * prog){
    
    /*Launch a kernel and check for errors */
    int state = 0;
    cl_event * event=NULL;
    size_t * lsize=NULL;
    if (prog->outevent)
        event=&prog->event;
    if (prog->lsize[0]!=0)
        lsize=prog->lsize;
    
    state = clEnqueueNDRangeKernel(*inqueue,
                                   prog->kernel,
                                   prog->wdim,
                                   NULL,
                                   prog->gsize,
                                   lsize,
                                   prog->nwait,
                                   prog->waits,
                                   event);
    
    if (state !=CL_SUCCESS) fprintf(stderr,"%s\n",clerrors(state));
    
    return state;
    
}
