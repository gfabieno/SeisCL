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

int prog_args_list(const char *str, char *name, char *** argnames, int * ninputs){
    
    int state=0;
    int c = 0;
    char **arr = NULL;
    char * args = NULL;
    char del2[2] = ")";
    int i;
    
    
    char del1[100];
    sprintf(del1,"void %s(", name);
    
    char * strbeg = strstr(str, del1);
    char * strchk = strbeg;
    int cmt=1;
    while (cmt==1 && strbeg){
        cmt=0;
        while ( *strchk !='\n' && cmt!=1 && strchk!=str){
            if (*strchk=='/' && *(strchk-1)=='/'){
                cmt=1;
            }
            strchk-=1;
        }
        if (cmt==1){
            strbeg = strstr(strbeg+strlen(del1), del1);
            strchk=strbeg;
        }

    }
    
    if (!strbeg){
        fprintf(stderr, "Error: Could not extract kernel arguments of %s\n",name);
        return 1;
    }
    
    char * strend = strstr(strbeg, del2);
    
    if (!strend){
        fprintf(stderr, "Error: Could not extract kernel arguments of %s\n",name);
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
#ifdef __SEISCL__
int prog_read_file(char **output, size_t *size, const char *name) {
    FILE *fp = fopen(name, "rb");
    if (!fp) {
        return -1;
    }
    
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    *output = (char *)malloc(*size);
    if (!*output) {
        fclose(fp);
        return -1;
    }
    
    fread(*output, *size, 1, fp);
    fclose(fp);
    return 0;
}

int prog_write_file(const char *name, const unsigned char *content, size_t size) {
    FILE *fp = fopen(name, "wb+");
    if (!fp) {
        return -1;
    }
    fwrite(content, size, 1, fp);
    fclose(fp);
    return 0;
}
int prog_write_src(const char *name, char * content) {
    FILE *fp = fopen(name, "w");
    if (!fp) {
        return -1;
    }
    fprintf(fp, "%s", content);
    fclose(fp);
    return 0;
}

cl_int prog_compare(char * filename_src,
                    char * prog_src){
    
    int i;
    char src_cache[MAX_KERN_STR];
    long length;
    FILE * f = fopen (filename_src, "rb");
    if (f){
        fseek (f, 0, SEEK_END);
        length = ftell (f);
        fseek (f, 0, SEEK_SET);
        if (length<MAX_KERN_STR){
        fread (src_cache, 1, length, f);
        }
        else{
            fprintf(stderr,"Error: cached kernel length too long\n");
            return 1;
        }
        fclose (f);
    }
    else{
        return 0;
    }

    for (i=0;i<length;i++){
        if (prog_src[i]!=src_cache[i]){
            return 0;
        }
    }
    

    return 1;

}

cl_int prog_write_binaries(cl_program *program,
                           char * filename_bin,
                           char * filename_src,
                           char * prog_src) {
    cl_int state = CL_SUCCESS;
    size_t *binaries_size = NULL;
    unsigned char **binaries_ptr = NULL;
    
    // Read the binaries size
    size_t binaries_size_alloc_size = sizeof(size_t);
    GMALLOC(binaries_size, binaries_size_alloc_size);
    
    __GUARD clGetProgramInfo(*program, CL_PROGRAM_BINARY_SIZES,
                             binaries_size_alloc_size, binaries_size, NULL);
    
    
    // Read the binaries
    size_t binaries_ptr_alloc_size = sizeof(unsigned char *);
    GMALLOC(binaries_ptr, binaries_ptr_alloc_size);
    GMALLOC(binaries_ptr[0],binaries_size[0]);
    __GUARD clGetProgramInfo(*program, CL_PROGRAM_BINARIES,
                             binaries_ptr_alloc_size,
                             binaries_ptr, NULL);
    
    // Write the binary ans src to the output file
    prog_write_file(filename_bin, binaries_ptr[0], binaries_size[0]);
    prog_write_src(filename_src, prog_src);
    
    if (binaries_ptr && binaries_ptr[0]){
        free(binaries_ptr[0]);
    }
    if (binaries_ptr){
        free(binaries_ptr);
    }
    if (binaries_ptr){
        free(binaries_size);
    }
    
    return state;
}
#endif

#ifdef __SEISCL__
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
        sprintf(src,"-D HC%d=%9.9f ",i+1,m->hc[i+1]);
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
            cl_device_id device,
            cl_kernel *kernel,
            const char * program_name,
            const char * build_options,
            const char * cache_dir,
            int devid,
            int ctxid)
{
    /* Routine to build a kernel from the source file contained in a c string*/
    
    int state = 0;
    size_t program_size = 0;
    
    // Write the binaries to file
    // Create output file name
    char filename_bin[PATH_MAX];
    snprintf(filename_bin, sizeof(filename_bin), "%s/%s-%d-%d.bin",
             cache_dir, program_name, ctxid, devid);
    char filename_src[PATH_MAX];
    snprintf(filename_src, sizeof(filename_src), "%s/%s-%d-%d.src",
             cache_dir, program_name, ctxid, devid);
    
    int same =  prog_compare(filename_src,
                             (char *)program_source);
    if (!*program){
        if (same!=1){
            *program = clCreateProgramWithSource(*context,
                                                 1,
                                                 &program_source,
                                                 NULL,
                                                 &state);
            if (state !=CL_SUCCESS) fprintf(stderr,"Error: %s\n",clerrors(state));
        }
        else{
            unsigned char* program_file = NULL;
            prog_read_file((char **)&program_file,
                           &program_size,
                           filename_bin);
            *program = clCreateProgramWithBinary(*context,
                                      1,
                                      &device,
                                      &program_size,
                                      (const unsigned char **)&program_file,
                                      NULL,
                                      &state);
            GFree(program_file);
            if (state !=CL_SUCCESS) fprintf(stderr,"Error: %s\n",clerrors(state));
            
        }
        state = clBuildProgram(*program, 1, &device, build_options, NULL, NULL);
        if (state !=CL_SUCCESS) fprintf(stderr,"Error: %s\n",clerrors(state));
        
        if (same!=1){
            __GUARD prog_write_binaries(program,
                                        (char *)filename_bin,
                                        (char *)filename_src,
                                        (char *)program_source);
        }
    }
    // Now create the kernel "objects"
    *kernel = clCreateKernel(*program, program_name, &state);
    if (state !=CL_SUCCESS) fprintf(stderr,"Error: %s\n",clerrors(state));
    
    
    return state;
    
}
#else
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
    
    char * value = getenv("CUDA_PATH");
    *n+=1;
    sprintf(build_options[*n-1],"--include-path=%s/include",value);
    *n+=1;
    sprintf(build_options[*n-1],"--pre-include=cuda_fp16.h");
    *n+=1;
    sprintf(build_options[*n-1],"--gpu-architecture=compute_%d%d",
                        dev->cuda_arc[0], dev->cuda_arc[1]);

    if (m->N_names[0]){
        for (i=0;i<m->NDIM;i++){
            *n+=1;
            if (i==0 && m->FP16>0){
                sprintf(build_options[*n-1],"-D N%s=%d ",
                        m->N_names[i],((*dev).N[i]+m->FDORDER)/2);
            }
            else{
                sprintf(build_options[*n-1],"-D N%s=%d ",
                        m->N_names[i],(*dev).N[i]+m->FDORDER);
            }
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
        sprintf(build_options[*n-1],"-D HC%d=%9.9f ",i+1,m->hc[i+1]);
    }
    
    *n+=1;
    sprintf(build_options[*n-1],"-D FP16=%d ",(*m).FP16);
    
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
        if (state !=CUDA_SUCCESS) fprintf(stderr,"Error: %s\n",clerrors(state));
        //WARNING $CUDA_PATH/lib64 has to be in LD_LIBRARY_PATH for compilation
        __GUARD nvrtcCompileProgram(cuprog,noptions,(const char * const*)build_options);
        if (state !=NVRTC_SUCCESS) fprintf(stderr,"Error: %s\n",clerrors(state));
        size_t logSize;
        nvrtcGetProgramLogSize(cuprog, &logSize);
        if (state){
            char *log = malloc(logSize);
            state = nvrtcGetProgramLog(cuprog, log);
            fprintf(stdout,"Compilation of %s:\n",program_name);
            fprintf(stdout,"%s",log);
            free(log);
        }
        
        if (state !=NVRTC_SUCCESS) fprintf(stderr,"Error: %s\n",clerrors(state));
        __GUARD nvrtcGetPTXSize(cuprog, &ptxSize);
        GMALLOC(program, sizeof(char)*ptxSize);
        __GUARD nvrtcGetPTX(cuprog, program);
        __GUARD nvrtcDestroyProgram(&cuprog);
        __GUARD cuModuleLoadDataEx(module, program, 0, 0, 0);
    }
    
    // Now create the kernel "objects"
    __GUARD cuModuleGetFunction(kernel, *module, program_name);
    if (state !=CUDA_SUCCESS) fprintf(stderr,"Error: %s\n",clerrors(state));
    
    
    return state;
    
}
#endif


int prog_source(clprogram * prog, char* name, const char * source){
    int state =0;
    snprintf((*prog).src, sizeof((*prog).src), "%s", source);
    (*prog).name=name;
    state =prog_args_list(source,
                        (char *)(*prog).name ,
                        &(*prog).input_list,
                        &(*prog).ninputs);
    
    return state;
}
int prog_arg(clprogram * prog, int i, void * mem, int size){
    int state =0;
    
    #ifdef __SEISCL__
    state = clSetKernelArg((*prog).kernel, i, size, mem);
    #else
    (*prog).inputs[i]=mem;
    #endif
    return state;
}

int prog_create(model * m,
                device * dev,
                clprogram * prog){
    
    int state = 0;
    int i,j, argfound;
    char str2comp[50];
    int shared_size = 0;
    int memsize = 0;
    int noptions=0;
    #ifdef __SEISCL__
    const char * build_options = get_build_options(dev,
                                                   m,
                                                   prog->LCOMM,
                                                   prog->COMM,
                                                   prog->DIRPROP);
    state = compile( (*prog).src,
                    &(*prog).prog,
                    &m->context,
                    dev->cudev,
                    &(*prog).kernel,
                    (*prog).name,
                    build_options,
                    m->cache_dir,
                    dev->DEVID,
                    dev->ctx_id);
    
    
    #else
    char ** build_options=NULL;
        GMALLOC(build_options, sizeof(char*)*50);
    for (i=0;i<50;i++){
        GMALLOC(build_options[i], sizeof(char)*500);
    }
    state= get_build_options(dev,
                              m,
                              build_options,
                              &noptions,
                              prog->LCOMM,
                              prog->COMM,
                              prog->DIRPROP);
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
    #endif
    
    if (state){
        printf("Error: Could not build kernel %s\n", (*prog).name);
        
        return state;
    }

    if (m->FP16==1){
        shared_size=2*sizeof(float);
    }
    else{
        shared_size=sizeof(float);
    }

    /*Define the size of the local variables of the compute device*/
    if (   dev->LOCAL_OFF==0 && (*prog).lsize[0]>0){
        for (i=0;i<m->NDIM;i++){
            shared_size*=(*prog).lsize[i]+m->FDORDER;
        }
        (*prog).shared_size=shared_size;
    }
    #ifdef __SEISCL__
    memsize = sizeof(cl_mem);
    #endif

    /*Define the arguments for this kernel */
    
    for (i=0;i<(*prog).ninputs;i++){
        argfound=0;
        //        printf("%s\n",(*prog).input_list[i]);
        for (j=0;j<m->npars;j++){
            if (strcmp((*dev).pars[j].name,(*prog).input_list[i])==0){
                prog_arg(prog, i, &(*dev).pars[j].cl_par.mem, memsize);
                argfound=1;

                break;
            }
            sprintf(str2comp,"grad%s",(*dev).pars[j].name);
            if (strcmp(str2comp,(*prog).input_list[i])==0){
                prog_arg(prog, i, &(*dev).pars[j].cl_grad.mem, memsize);
                argfound=1;
                
                break;
            }
            sprintf(str2comp,"H%s",(*dev).pars[j].name);
            if (strcmp(str2comp,(*prog).input_list[i])==0 && m->HOUT==1){
                prog_arg(prog, i, &(*dev).pars[j].cl_H.mem, memsize);
                argfound=1;
                
                break;
            }
        }
        if (!argfound){
            for (j=0;j<m->nvars;j++){
                if (strcmp((*dev).vars[j].name,(*prog).input_list[i])==0){
                    prog_arg(prog, i, &(*dev).vars[j].cl_var.mem, memsize);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%sout",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    prog_arg(prog, i, &(*dev).vars[j].cl_varout.mem, memsize);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%sbnd",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    prog_arg(prog, i, &(*dev).vars[j].cl_varbnd.mem, memsize);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"f%s",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    prog_arg(prog, i, &(*dev).vars[j].cl_fvar.mem, memsize);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%s_buf1",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    prog_arg(prog, i, &(*dev).vars[j].cl_buf1.mem, memsize);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%s_buf2",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    prog_arg(prog, i, &(*dev).vars[j].cl_buf2.mem, memsize);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%sr_buf1",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    prog_arg(prog, i, &(*dev).vars_adj[j].cl_buf1.mem, memsize);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"%sr_buf2",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    prog_arg(prog, i, &(*dev).vars_adj[j].cl_buf2.mem, memsize);
                    argfound=1;
                    break;
                }
                sprintf(str2comp,"scaler_%s",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    prog_arg(prog, i, &(*dev).vars[j].scaler, sizeof(int));
                    argfound=1;
                    break;
                }
            }
        }
        if (!argfound){
            for (j=0;j<m->ntvars;j++){
                sprintf(str2comp,"%sout",(*dev).trans_vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    prog_arg(prog, i, &(*dev).trans_vars[j].cl_varout.mem, memsize);
                    argfound=1;
                    break;
                }
               
            }
        }
        if (!argfound){
            for (j=0;j<m->nvars;j++){
                sprintf(str2comp,"%sr",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    //TODO this is a hack for how adj kernels are written,
                    //     think of a better way
                    if (m->BACK_PROP_TYPE==1){
                        prog_arg(prog, i, &(*dev).vars_adj[j].cl_var.mem, memsize);
                    }
                    else if (m->BACK_PROP_TYPE==2){
                        prog_arg(prog, i, &(*dev).vars[j].cl_var.mem, memsize);
                    }
                    argfound=1;

                    break;
                }
            }
        }
        if (!argfound){
            for (j=0;j<m->ncsts;j++){
                if (strcmp((*dev).csts[j].name,(*prog).input_list[i])==0){
                    prog_arg(prog, i, &(*dev).csts[j].cl_cst.mem, memsize);
                    argfound=1;

                    break;
                }
            }
        }
        if (!argfound){
            if (strcmp("src" ,(*prog).input_list[i])==0){
                prog_arg(prog, i, &(*dev).src_recs.cl_src.mem, memsize);
                argfound=1;
            }
            else if (strcmp("src_pos" ,(*prog).input_list[i])==0){
                prog_arg(prog, i, &(*dev).src_recs.cl_src_pos.mem, memsize);
                argfound=1;
            }
            else if (strcmp("rec_pos" ,(*prog).input_list[i])==0){
                prog_arg(prog, i, &(*dev).src_recs.cl_rec_pos.mem, memsize);
                argfound=1;
            }
            else if (strcmp("grad_src" ,(*prog).input_list[i])==0){
                prog_arg(prog, i, &(*dev).src_recs.cl_grad_src.mem, memsize);
                argfound=1;
            }
        }

        if (!argfound){
            if (strcmp("offcomm",(*prog).input_list[i])==0){
                prog_arg(prog, i, &prog->OFFCOMM, sizeof(int));
                argfound=1;
            }
        }
        
        if (!argfound){
            if (strcmp("LOCARG",(*prog).input_list[i])==0){
                #ifdef __SEISCL__
                prog_arg(prog, i, NULL, shared_size);
                #else
                prog_arg(prog, i, &dev->cuda_null, memsize);
                #endif
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
            if (strcmp("nsrc"  ,(*prog).input_list[i])==0){
                prog->nsinput=i+1;
                argfound=1;
            }
        }
        if (!argfound){
            if (strcmp("nrec"  ,(*prog).input_list[i])==0){
                prog->nrinput=i+1;
                argfound=1;
            }
        }
        if (!argfound){
            if (strcmp("src_scale"  ,(*prog).input_list[i])==0){
                prog->scinput=i+1;
                argfound=1;
            }
        }
        if (!argfound){
            if (strcmp("res_scale"  ,(*prog).input_list[i])==0){
                prog->rcinput=i+1;
                argfound=1;
            }
        }
        
        if (!argfound){
            #ifdef __DEBUGGING__
            fprintf(stdout,"Warning: input %s undefined for kernel %s\n",
                             (*prog).input_list[i], (*prog).name);
            #endif
            prog_arg(prog, i, &dev->cuda_null, memsize);
        }

    }
    

    dev->progs[dev->nprogs]=prog;
    dev->nprogs++;
    
    return state;
}

    
int prog_launch( QUEUE *inqueue, clprogram * prog){
    
    /*Launch a kernel and check for errors */
    int state = 0;
    
    #ifdef __SEISCL__
    cl_event * event=NULL;
    size_t * lsize=NULL;
    if (prog->outevent){
        if (prog->event){
            state =  clReleaseEvent(prog->event);
        }
        event=&prog->event;
    }
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
    #else
    int i;
    unsigned int bsize[] ={1,1,1};
    unsigned int tsize[] ={BLOCK_SIZE,1,1};
    if (prog->wdim<1){
        prog->wdim=1;
    }
    for (i=0;i<prog->wdim;i++){
        if (prog->lsize[i]>0){
            tsize[i]=(unsigned int)prog->lsize[i];
        }
        bsize[i]=(unsigned int)(prog->gsize[i]+tsize[i]-1)/tsize[i];
    }
    
    state = cuLaunchKernel (prog->kernel,
                            bsize[0],
                            bsize[1],
                            bsize[2],
                            tsize[0],
                            tsize[1],
                            tsize[2],
                            (unsigned int)prog->shared_size,
                            *inqueue,
                            prog->inputs,
                            NULL );
    #endif
    

    if (state !=CUCL_SUCCESS) {fprintf(stderr,"Error launching %s: %s\n",
                                       prog->name,clerrors(state));    }
    
    return state;
}
