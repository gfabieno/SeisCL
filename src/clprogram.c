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

char **get_build_options(device *dev,
                        model *m,
                        int LCOMM,
                        int comm,
                        int DIRPROP)
{
    int state=0;
    int i;
    char **build_options=NULL;
    char src[50];
    
    GMALLOC(build_options, sizeof(char*)*2);
    
    if (m->N_names[0]){
        for (i=0;i<m->NDIM-1;i++){
            GMALLOC(build_options[i], sizeof(char)*30);
            sprintf(build_options[i],"-D N%s=%d ",m->N_names[i],(*dev).N[i]+m->FDORDER);
//            strcat(build_options[i],src);
            
        }
    }
//    else{
//        for (i=0;i<m->NDIM;i++){
//            sprintf(src,"-D N%d=%d ",i,(*dev).N[i]+m->FDORDER);
//            strcat(build_options,src);
//            
//        }
//    }
//    for (i=0;i<m->FDOH;i++){
//        sprintf(src,"-D hc%d=%9.9f ",i+1,m->hc[i+1]);
//        strcat(build_options,src);
//        
//    }
//    
//    char src2[2000];
//    sprintf(src2,"-D NDIM=%d -D OFFSET=%d -D FDOH=%d -D DTDH=%9.9f -D DH=%9.9f "
//            "-D DT=%9.9f -D DT2=%9.9f -D NT=%d -D NAB=%d -D NBND=%d "
//            "-D LOCAL_OFF=%d -D LVE=%d -D DEVID=%d -D NUM_DEVICES=%d "
//            "-D ND=%d -D ABS_TYPE=%d -D FREESURF=%d -D LCOMM=%d "
//            "-D MYLOCALID=%d -D NLOCALP=%d -D NFREQS=%d "
//            "-D BACK_PROP_TYPE=%d -D COMM12=%d -D NTNYQ=%d -D DTNYQ=%d "
//            "-D VARSOUT=%d -D RESOUT=%d  -D RMSOUT=%d -D MOVOUT=%d "
//            "-D GRADOUT=%d -D HOUT=%d -D GRADSRCOUT=%d -D DIRPROP=%d",
//            (*m).NDIM, (*dev).NX0, (*m).FDOH, (*m).dt/(*m).dh, (*m).dh,
//            (*m).dt, (*m).dt/2.0, (*m).NT, (*m).NAB, (*dev).NBND,
//            (*dev).LOCAL_OFF, (*m).L, (*dev).DEVID, (*m).NUM_DEVICES,
//            (*m).ND, (*m).ABS_TYPE, (*m).FREESURF, LCOMM,
//            (*m).MYLOCALID, (*m).NLOCALP, (*m).NFREQS,
//            (*m).BACK_PROP_TYPE, comm, (*m).NTNYQ, (*m).DTNYQ,
//            (*m).VARSOUT, (*m).RESOUT, (*m).RMSOUT, (*m).MOVOUT,
//            (*m).GRADOUT, (*m).HOUT, (*m).GRADSRCOUT, DIRPROP  );
//    
//    strcat(build_options,src2);
    
    //Make it all uppercase
//    char *s = build_options;
//    while (*s) {
//        *s = toupper((unsigned char) *s);
//        s++;
//    }
    
    return build_options;
}

int compile(const char *program_source,
                    char * program,
                    CUmodule *module,
                    CUfunction *kernel,
                    const char * program_name,
                    char ** build_options)
{
    /* Routine to build a kernel from the source file contained in a c string*/
    
    int state = 0;
    size_t ptxSize=0;
    nvrtcProgram cuprog;
    fprintf(stdout,"%s\n",build_options[0]);
    fprintf(stdout,"%s\n",build_options[1]);
    if (!program){
        __GUARD nvrtcCreateProgram(&cuprog,
                           program_source,
                           program_name,
                           0,
                           NULL,
                           NULL);
        if (state !=CUDA_SUCCESS) fprintf(stderr,"%s\n",clerrors(state));
        __GUARD nvrtcCompileProgram(cuprog,2,build_options);
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
    
    char ** build_options = get_build_options(dev,
                                                   m,
                                                   prog->LCOMM,
                                                   prog->COMM,
                                                   prog->DIRPROP);
    fprintf(stdout,"compiling=%s\n",(*prog).name);
    state = compile( (*prog).src,
                     (*prog).prog,
                     &(*prog).module,
                     &(*prog).kernel,
                     (*prog).name,
                     build_options);
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
