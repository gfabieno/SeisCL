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
    int i;
    
    
    char del1[100];
    
    sprintf(del1,"__global__ void %s(", name);
    
    char * strbeg = strstr(str, del1);
    
    if (!strbeg){
        fprintf(stderr, "Could not extract kernel arguments of %s\n",name);
        return 1;
    }
    
    char * strend = strstr(strbeg, del2);
    
    if (!strend){
        fprintf(stderr, "Could not extract kernel arguments of %s\n",name);
        return 1;
    }
    
    
    args = malloc(sizeof(str)*(strend- strbeg-strlen(del1)+1 ));
    
    sprintf(args,"%.*s", (int)(strend- strbeg-strlen(del1)), strbeg + strlen(del1));
    
    
    
    c = split(args, ',', &arr);
    
    GMALLOC((*argnames), c*sizeof(char *) );
    
    
    for ( i = 0; i < c; i++){
        (*argnames)[i]=extract_name(arr[i]);
        free(arr[i]);
    }
    *ninputs=c;
    free(arr);
    free(args);
    
    return state;
}

int add_option( char ** build_options, int * n){
    int state=0;
    GMALLOC(build_options[*n], sizeof(char)*30);
    *n+=1;
    return state;
    
}

int get_build_options(device *dev,
                        model *m,
                        char **build_options,
                        int *n,
                        int LCOMM,
                        int comm,
                        int DIRPROP)
{
    int state=0;
    int i;
    
    *n=0;
    if (m->N_names[0]){
        for (i=0;i<m->NDIM;i++){
            *n+=1;
            sprintf(build_options[*n-1],"-D N%s=%d ",m->N_names[i],(*dev).N[i]+m->FDORDER);
        }
    }
    else{
        for (i=0;i<m->NDIM;i++){
            *n+=1;
            sprintf(build_options[*n-1],"-D N%d=%d ",i,(*dev).N[i]+m->FDORDER);
        }
    }
    for (i=0;i<m->FDOH;i++){
        *n+=1;
        sprintf(build_options[*n-1],"-D hc%d=%9.9f ",i+1,m->hc[i+1]);
    }
    
    *n+=1;
    sprintf(build_options[*n-1],"-D NDIM=%d ",(*m).NDIM);
    *n+=1;
    sprintf(build_options[*n-1],"-D OFFSET=%d",(*dev).NX0);
    *n+=1;
    sprintf(build_options[*n-1],"-D FDOH=%d",(*m).FDOH);
    *n+=1;
    sprintf(build_options[*n-1],"-D DTDH=%9.9f",(*m).dt/(*m).dh);
    *n+=1;
    sprintf(build_options[*n-1],"-D DH=%9.9f",(*m).dh);
    *n+=1;
    sprintf(build_options[*n-1],"-D DT=%9.9f",(*m).dt);
    *n+=1;
    sprintf(build_options[*n-1],"-D DT2=%9.9f",(*m).dt/2.0);
    *n+=1;
    sprintf(build_options[*n-1],"-D NT=%d",(*m).NT);
    *n+=1;
    sprintf(build_options[*n-1],"-D NAB=%d",(*m).NAB);
    *n+=1;
    sprintf(build_options[*n-1],"-D NBND=%d",(*dev).NBND);
    *n+=1;
    sprintf(build_options[*n-1],"-D LOCAL_OFF=%d",(*dev).LOCAL_OFF);
    *n+=1;
    sprintf(build_options[*n-1],"-D LVE=%d",(*m).L);
    *n+=1;
    sprintf(build_options[*n-1],"-D DEVID=%d", (*dev).DEVID);
    *n+=1;
    sprintf(build_options[*n-1],"-D NUM_DEVICES=%d",  (*m).NUM_DEVICES);
    *n+=1;
    sprintf(build_options[*n-1],"-D ND=%d",  (*m).ND);
    *n+=1;
    sprintf(build_options[*n-1],"-D ABS_TYPE=%d",  (*m).ABS_TYPE);
    *n+=1;
    sprintf(build_options[*n-1],"-D FREESURF=%d",  (*m).FREESURF);
    *n+=1;
    sprintf(build_options[*n-1],"-D LCOMM=%d",  LCOMM);
    *n+=1;
    sprintf(build_options[*n-1],"-D MYLOCALID=%d",  (*m).MYLOCALID);
    *n+=1;
    sprintf(build_options[*n-1],"-D NLOCALP=%d",  (*m).NLOCALP);
    *n+=1;
    sprintf(build_options[*n-1],"-D NFREQS=%d",  (*m).NFREQS);
    *n+=1;
    sprintf(build_options[*n-1],"-D BACK_PROP_TYPE=%d",  (*m).BACK_PROP_TYPE);
    *n+=1;
    sprintf(build_options[*n-1],"-D COMM12=%d",  comm);
    *n+=1;
    sprintf(build_options[*n-1],"-D NTNYQ=%d", (*m).NTNYQ);
    *n+=1;
    sprintf(build_options[*n-1],"-D VARSOUT=%d",(*m).VARSOUT);
    *n+=1;
    sprintf(build_options[*n-1],"-D RESOUT=%d",(*m).RESOUT);
    *n+=1;
    sprintf(build_options[*n-1],"-D RMSOUT=%d",(*m).RMSOUT);
    *n+=1;
    sprintf(build_options[*n-1],"-D MOVOUT=%d",(*m).MOVOUT);
    *n+=1;
    sprintf(build_options[*n-1],"-D GRADOUT=%d ",(*m).GRADOUT);
    *n+=1;
    sprintf(build_options[*n-1],"-D HOUT=%d",(*m).HOUT);
    *n+=1;
    sprintf(build_options[*n-1],"-D GRADSRCOUT=%d",(*m).GRADSRCOUT);
    *n+=1;
    sprintf(build_options[*n-1],"-D DIRPROP=%d",DIRPROP);
    
    return state;
}

int compile(const char *program_source,
                    char * program,
                    CUmodule *module,
                    CUfunction *kernel,
                    const char * program_name,
                    char ** build_options,
                    int noptions)
{
    /* Routine to build a kernel from the source file contained in a c string*/
    
    int state = 0;
    size_t ptxSize=0;
    nvrtcProgram cuprog;
    if (!program){
        __GUARD nvrtcCreateProgram(&cuprog,
                           program_source,
                           program_name,
                           0,
                           NULL,
                           NULL);
        if (state !=CUDA_SUCCESS) fprintf(stderr,"%s\n",clerrors(state));
        __GUARD nvrtcCompileProgram(cuprog,noptions,build_options);
        if (state !=NVRTC_SUCCESS) fprintf(stderr,"%s\n",clerrors(state));
        __GUARD nvrtcGetPTXSize(cuprog, &ptxSize);
        GMALLOC(program, sizeof(char)*ptxSize);
        __GUARD nvrtcGetPTX(cuprog, program);
        __GUARD nvrtcDestroyProgram(&cuprog);
        __GUARD cuModuleLoadDataEx(module, program, 0, 0, 0);
    }
    
    // Now create the kernel "objects"
    __GUARD cuModuleGetFunction(kernel, *module, program_name);
    if (state !=CUDA_SUCCESS) fprintf(stderr,"%s\n",clerrors(state));
    
    
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
    
    int noptions=0;
    char ** build_options=NULL;
        GMALLOC(build_options, sizeof(char*)*50);
    for (i=0;i<50;i++){
        GMALLOC(build_options[i], sizeof(char)*30);
    }
    
    state= get_build_options(dev,
                              m,
                              build_options,
                              &noptions,
                              prog->LCOMM,
                              prog->COMM,
                              prog->DIRPROP);
    fprintf(stdout,"compiling=%s\n",(*prog).name);
    state = compile( (*prog).src,
                     (*prog).prog,
                     &(*prog).module,
                     &(*prog).kernel,
                     (*prog).name,
                     build_options,
                     noptions);
    if (build_options){
        for (i=0;i<noptions;i++){
            GFree(build_options[i]);
        }
        GFree(build_options);
    }
    
    
    if (state){
        printf("Error: Could not build kernel %s\n", (*prog).name);
        
        return state;
    }
    
    /*Define the size of the local variables of the compute device*/
    if (   dev->LOCAL_OFF==0 && (*prog).lsize[0]>0){
        for (i=0;i<m->NDIM;i++){
            shared_size*=(*prog).lsize[i]+m->FDORDER;
        }
        (*prog).shared_size=shared_size;
    }
    

    /*Define the arguments for this kernel */
    
    for (i=0;i<(*prog).ninputs;i++){
        argfound=0;
        //        printf("%s\n",(*prog).input_list[i]);
        for (j=0;j<m->npars;j++){
            if (strcmp((*dev).pars[j].name,(*prog).input_list[i])==0){
                (*prog).inputs[i]=(void*)(*dev).pars[j].cl_par.mem;
                argfound=1;

                break;
            }
            sprintf(str2comp,"grad%s",(*dev).pars[j].name);
            if (strcmp(str2comp,(*prog).input_list[i])==0){
                (*prog).inputs[i]=(void*)(*dev).pars[j].cl_grad.mem;
                argfound=1;
                
                break;
            }
            sprintf(str2comp,"H%s",(*dev).pars[j].name);
            if (strcmp(str2comp,(*prog).input_list[i])==0){
                (*prog).inputs[i]=(void*)(*dev).pars[j].cl_H.mem;
                argfound=1;
                
                break;
            }
        }
        if (!argfound){
            for (j=0;j<m->nvars;j++){
                if (strcmp((*dev).vars[j].name,(*prog).input_list[i])==0){
                    (*prog).inputs[i]=(void*)(*dev).vars[j].cl_var.mem;
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%sout",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    (*prog).inputs[i]=(void*)(*dev).vars[j].cl_varout.mem;
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%sbnd",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    (*prog).inputs[i]=(void*)(*dev).vars[j].cl_varbnd.mem;
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"f%s",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    (*prog).inputs[i]=(void*)(*dev).vars[j].cl_fvar.mem;
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%s_buf1",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    (*prog).inputs[i]=(void*)(*dev).vars[j].cl_buf1.mem;
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%s_buf2",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    (*prog).inputs[i]=(void*)(*dev).vars[j].cl_buf2.mem;
                    argfound=1;
                    break;
                }
            }
        }
        if (!argfound){
            for (j=0;j<m->ntvars;j++){
                sprintf(str2comp,"%sout",(*dev).trans_vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    (*prog).inputs[i]=(void*)(*dev).trans_vars[j].cl_varout.mem;
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
                        (*prog).inputs[i]=(void*)(*dev).vars_adj[j].cl_var.mem;
                    }
                    else if (m->BACK_PROP_TYPE==2){
                        (*prog).inputs[i]=(void*)(*dev).vars[j].cl_var.mem;
                    }
                    argfound=1;

                    break;
                }
            }
        }
        if (!argfound){
            for (j=0;j<m->ncsts;j++){
                if (strcmp((*dev).csts[j].name,(*prog).input_list[i])==0){
                    (*prog).inputs[i]=(void*)(*dev).csts[j].cl_cst.mem;
                    argfound=1;

                    break;
                }
            }
        }
        if (!argfound){
            if (strcmp("src" ,(*prog).input_list[i])==0){
                (*prog).inputs[i]=(void*)(*dev).src_recs.cl_src.mem;
                argfound=1;
            }
            else if (strcmp("src_pos" ,(*prog).input_list[i])==0){
                (*prog).inputs[i]=(void*)(*dev).src_recs.cl_src_pos.mem;
                argfound=1;
            }
            else if (strcmp("rec_pos" ,(*prog).input_list[i])==0){
                (*prog).inputs[i]=(void*)(*dev).src_recs.cl_rec_pos.mem;
                argfound=1;
            }
            else if (strcmp("grad_src" ,(*prog).input_list[i])==0){
                (*prog).inputs[i]=(void*)(*dev).src_recs.cl_grad_src.mem;
                argfound=1;
            }
        }

        if (!argfound){
            if (strcmp("offcomm",(*prog).input_list[i])==0){
                (*prog).inputs[i]=&prog->OFFCOMM;
                argfound=1;
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
            (*prog).inputs[i]=NULL;
        }

    }
    

    dev->progs[dev->nprogs]=prog;
    dev->nprogs++;
    
    return state;
}

int prog_launch( CUstream *inqueue, clprogram * prog){
    
    /*Launch a kernel and check for errors */
    int state = 0;

    size_t * lsize=NULL;

    if (prog->lsize[0]!=0)
        lsize=prog->lsize;

    state = cuLaunchKernel (prog->kernel,
                            (unsigned int)prog->bsize[0],
                            (unsigned int)prog->bsize[1],
                            (unsigned int)prog->bsize[2],
                            (unsigned int)prog->lsize[0],
                            (unsigned int)prog->lsize[1],
                            (unsigned int)prog->lsize[2],
                            (unsigned int)prog->shared_size,
                            *inqueue,
                            prog->inputs,
                            NULL );

    
    if (state !=CUDA_SUCCESS) fprintf(stderr,"%s\n",clerrors(state));
    
    return state;
    
}
