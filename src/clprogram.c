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
int prog_read_file(char **output, size_t *size, const char *name) {
    FILE *fp = fopen(name, "rb");
    if (!fp) {
        return -1;
    }
    
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (*size > MAX_KERN_BIN){
        fprintf(stderr,"Error: cached kernel binary too long, raise MAX_KERN_BIN in F.h\n");
        return -1;
    }
    
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

CL_INT prog_compare(char * filename_src,
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

    /* the whole source: a longer source with the same beginning is not a hit */
    if ((size_t)length != strlen(prog_src)){
        return 0;
    }
    for (i=0;i<length;i++){
        if (prog_src[i]!=src_cache[i]){
            return 0;
        }
    }
    

    return 1;

}

#ifdef __SEISCL__
CL_INT prog_write_binaries(PROGRAM *program,
                           char * filename_bin,
                           char * filename_src,
                           char * prog_src,
                           char * filename_opt,
                           char * options) {
    CL_INT state = 0;
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
    prog_write_src(filename_opt, options);
    
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


char *get_build_options(device *dev,
                        model *m,
                        int LCOMM,
                        int comm,
                        int DIRPROP)
{
    int i;
    static char build_options [6000]={0};
    char src[50];
    int parny, parnx;

    build_options[0]=0;
    if (m->N_names[0]){
        for (i=0;i<m->NDIM;i++){
            if (i==0 && m->FP16>0){
                sprintf(src,"-D N%s=%d ",m->N_names[i],((*dev).N[i]+m->FDORDER)/2);
            }
            else{
                sprintf(src,"-D N%s=%d ",m->N_names[i],(*dev).N[i]+m->FDORDER);
            }
            strcat(build_options,src);

        }
        /* Scalar (unvectorized) padded extents. N<dim> above is halved on the
           fastest axis whenever FP16>0, because the update kernels address the
           wavefield as float2/half2. Kernels that address it one scalar element
           at a time -- the DFT gradient correlation, whose spectra are one
           float2 per *scalar* element (Init_OpenCL.c's cl_fvar sizing) -- need
           the true extent, and cannot recover it as 2*N<dim> when
           N[i]+FDORDER is odd. */
        for (i=0;i<m->NDIM;i++){
            sprintf(src,"-D N%sS=%d ",m->N_names[i],(*dev).N[i]+m->FDORDER);
            strcat(build_options,src);
        }
    }
    else{
        for (i=0;i<m->NDIM;i++){
            sprintf(src,"-D N%d=%d ",i,(*dev).N[i]+m->FDORDER);
            strcat(build_options,src);
            
        }
    }
    for (i=0;i<MAX_FD_ORDER/2;i++){
        sprintf(src,"-D HC%d=%9.9ff ",i+1,m->hc[i+1]);
        strcat(build_options,src);
        
    }
    
    // Raw (unpadded) per-dimension grid size for src/average_params.cl (see
    // notes/vacuum-freesurface-plan.md, Phase 8) -- distinct from the N%d/N%s
    // macros above, which are FDORDER-padded wavefield sizes. dev->N is
    // [NZ,NX] for 2D and [NZ,NY,NX] for 3D; PARNY is unused (left at 1) for
    // 2D's own kernels but still emitted unconditionally to keep this call
    // simple.
    /* Which variable the RESIDUALS are injected into, so the gradient kernels
       can read the right `*out` buffer -- reading `pout` when the "p" trans_var
       is not an output binds to a placeholder and faults. */
    int presout = 0, velout = 0;
    for (i=0;i<(*dev).ntvars;i++){
        if (strcmp((*dev).trans_vars[i].name,"p")==0 && (*dev).trans_vars[i].to_output)
            presout = 1;
    }
    /* One flag per component as well: an output set may hold vx without vz
       (the PyTorch engine accepts any subset), and the adjoint kernels then
       must not read the unallocated vzout (a NULL read, update_adjv). */
    int vout[3] = {0, 0, 0};
    for (i=0;i<(*dev).nvars;i++){
        if (!(*dev).vars[i].to_output) continue;
        if (strcmp((*dev).vars[i].name,"vx")==0) vout[0] = 1;
        if (strcmp((*dev).vars[i].name,"vy")==0) vout[1] = 1;
        if (strcmp((*dev).vars[i].name,"vz")==0) vout[2] = 1;
    }
    velout = vout[0] || vout[1] || vout[2];

    parny = (m->NDIM==3) ? (*dev).N[1] : 1;
    parnx = (m->NDIM==3) ? (*dev).N[2] : (*dev).N[1];

    char src2[2000];
    sprintf(src2,"-I ./ -D NDIM=%d -D OFFSET=%d -D FDOH=%d -D DTDH=%9.9f -D DH=%9.9ff "
            "-D DT=%9.9ff -D DT2=%9.9ff -D NT=%d -D NAB=%d -D NBND=%d "
            "-D LOCAL_OFF=%d -D LVE=%d -D DEVID=%d -D NUM_DEVICES=%d "
            "-D ND=%d -D ABS_TYPE=%d -D FREESURF=%d -D LCOMM=%d "
            "-D MYLOCALID=%d -D NLOCALP=%d -D NFREQS=%d "
            "-D BACK_PROP_TYPE=%d -D COMM12=%d -D NTNYQ=%d -D DTNYQ=%d "
            "-D VARSOUT=%d -D RESOUT=%d  -D RMSOUT=%d -D MOVOUT=%d "
            "-D GRADOUT=%d -D HOUT=%d -D GRADSRCOUT=%d -D DIRPROP=%d "
            "-D RESTYPE=%d -D FP16=%d -D PARSCALE=%d -D FREQ0=%9.9ff "
            "-D PARNZ=%d -D PARNY=%d -D PARNX=%d -D TMIN=%d "
            "-D PRESOUT=%d -D VELOUT=%d -D VXOUT=%d -D VYOUT=%d -D VZOUT=%d -D GRADABS=%d",
            (*m).NDIM, (*dev).OFFSET, (*m).FDOH, (*m).dt/(*m).dh, (*m).dh,
            (*m).dt, (*m).dt/2.0, (*m).NT, (*m).NAB, (*dev).NBND,
            (*dev).LOCAL_OFF, (*m).L, (*dev).DEVID, (*m).NUM_DEVICES,
            (*m).ND, (*m).ABS_TYPE, (*m).FREESURF, LCOMM,
            (*m).MYLOCALID, (*m).NLOCALP, (*m).NFREQS,
            (*m).BACK_PROP_TYPE, comm, (*m).NTNYQ, (*m).DTNYQ,
            (*m).VARSOUT, (*m).RESOUT, (*m).RMSOUT, (*m).MOVOUT,
            (*m).GRADOUT, (*m).HOUT, (*m).GRADSRCOUT, DIRPROP, (*m).restype,
            (*m).FP16, (*m).par_scale, (*m).f0,
            (*dev).N[0], parny, parnx, (*m).tmin, presout, velout, vout[0], vout[1], vout[2], (*m).GRADABS) ;


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
            int procid)
{
    /* Routine to build a kernel from the source file contained in a c string*/
    
    int state = 0;
    size_t program_size = 0;
    
    // Write the binaries to file
    // Create output file name
    char filename_bin[PATH_MAX];
    snprintf(filename_bin, sizeof(filename_bin), "%s/%s-%d-%d.bin",
             cache_dir, program_name, procid, devid);
    char filename_src[PATH_MAX];
    snprintf(filename_src, sizeof(filename_src), "%s/%s-%d-%d.src",
             cache_dir, program_name, procid, devid);
    char filename_opt[PATH_MAX];
    snprintf(filename_opt, sizeof(filename_opt), "%s/%s-%d-%d.opt",
             cache_dir, program_name, procid, devid);
    
    int same;
    same =  prog_compare(filename_opt, (char *)build_options);
    if (same==1){
        same =  prog_compare(filename_src, (char *)program_source);
    }

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
        if (state !=CL_SUCCESS){
            /* Print the compiler diagnostics. Without this a kernel that fails
             * to compile reports only "CL_BUILD_PROGRAM_FAILURE" and the count
             * of errors, with no indication of what or where. */
            size_t logsize=0;
            char * log=NULL;
            fprintf(stderr,"Error: %s\n",clerrors(state));
            clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG,
                                  0, NULL, &logsize);
            if (logsize>1){
                log = malloc(logsize+1);
                if (log){
                    clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG,
                                          logsize, log, NULL);
                    log[logsize]='\0';
                    fprintf(stderr,"Build log for %s:\n%s\n",
                            program_name, log);
                    free(log);
                }
            }
        }

        if (same!=1){
            __GUARD prog_write_binaries(program,
                                        (char *)filename_bin,
                                        (char *)filename_src,
                                        (char *)program_source,
                                        (char *)filename_opt,
                                        (char *)build_options);
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
    /* See the same computation in the OpenCL option string above: which
       variable the residuals are injected into, so the gradient kernels read a
       live `*out` buffer. */
    int presout = 0, velout = 0;
    for (i=0;i<(*dev).ntvars;i++){
        if (strcmp((*dev).trans_vars[i].name,"p")==0 && (*dev).trans_vars[i].to_output)
            presout = 1;
    }
    /* One flag per component as well: an output set may hold vx without vz
       (the PyTorch engine accepts any subset), and the adjoint kernels then
       must not read the unallocated vzout (a NULL read, update_adjv). */
    int vout[3] = {0, 0, 0};
    for (i=0;i<(*dev).nvars;i++){
        if (!(*dev).vars[i].to_output) continue;
        if (strcmp((*dev).vars[i].name,"vx")==0) vout[0] = 1;
        if (strcmp((*dev).vars[i].name,"vy")==0) vout[1] = 1;
        if (strcmp((*dev).vars[i].name,"vz")==0) vout[2] = 1;
    }
    velout = vout[0] || vout[1] || vout[2];
    
    *n=0;
    
//    char * value = getenv("CUDA_PATH");
//    *n+=1;
//    sprintf(build_options[*n-1],"--include-path=%s/include",value);
//    char * value = __CUDA_INCLUDE__;
    *n+=1;
    sprintf(build_options[*n-1],"--include-path=%s",__CUDA_INCLUDE__);
    *n+=1;
    sprintf(build_options[*n-1],"--pre-include=cuda_fp16.h");
    *n+=1;
    sprintf(build_options[*n-1],"--gpu-architecture=compute_%d%d",
                        dev->cuda_arc[0], dev->cuda_arc[1]);
    /* SEISCL_LINEINFO=1: compile the kernels with line information, so that
       compute-sanitizer reports the source line of a bad access (debugging
       only; see SeisCL_cache/<kernel>.src for the source). */
    {
        const char * v = getenv("SEISCL_LINEINFO");
        if (v && v[0]=='1'){
            *n+=1;
            sprintf(build_options[*n-1],"--generate-line-info");
        }
    }

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
        /* Scalar (unvectorized) padded extents -- see the OpenCL branch. */
        for (i=0;i<m->NDIM;i++){
            *n+=1;
            sprintf(build_options[*n-1],"-D N%sS=%d ",
                    m->N_names[i],(*dev).N[i]+m->FDORDER);
        }
    }
    else{
        for (i=0;i<m->NDIM;i++){
            *n+=1;
            sprintf(build_options[*n-1],"-D N%d=%d ",i,(*dev).N[i]+m->FDORDER);
        }
    }
    for (i=0;i<MAX_FD_ORDER/2;i++){
        *n+=1;
        sprintf(build_options[*n-1],"-D HC%d=%9.9ff ",i+1,m->hc[i+1]);
    }
    
    *n+=1;
    sprintf(build_options[*n-1],"-D FP16=%d ",(*m).FP16);
    GMALLOC(build_options[*n], sizeof(char)*30); *n+=1;
    sprintf(build_options[*n-1],"-D PARSCALE=%d ",(*m).par_scale);
    /* f0: the viscoelastic gradient needs it to rebuild the
       phase-velocity factor alpha -- see grad_dft2D_visc.cl. */
    GMALLOC(build_options[*n], sizeof(char)*30); *n+=1;
    sprintf(build_options[*n-1],"-D FREQ0=%9.9ff ",(*m).f0);
    
    *n+=1;
    sprintf(build_options[*n-1],"-D NDIM=%d ",(*m).NDIM);
    *n+=1;
    sprintf(build_options[*n-1],"-D OFFSET=%d",(*dev).OFFSET);
    *n+=1;
    sprintf(build_options[*n-1],"-D FDOH=%d",(*m).FDOH);
    *n+=1;
    sprintf(build_options[*n-1],"-D DTDH=%9.9ff",(*m).dt/(*m).dh);
    *n+=1;
    sprintf(build_options[*n-1],"-D DH=%9.9ff",(*m).dh);
    *n+=1;
    sprintf(build_options[*n-1],"-D DT=%9.9ff",(*m).dt);
    *n+=1;
    sprintf(build_options[*n-1],"-D DT2=%9.9ff",(*m).dt/2.0);
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
    sprintf(build_options[*n-1],"-D DTNYQ=%d", (*m).DTNYQ);
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
    *n+=1;
    sprintf(build_options[*n-1],"-D RESTYPE=%d",(*m).restype);
    *n+=1;
    /* First time step of the gradient window. savefreqs phases its DFT
       against the SAMPLED index (t-tmin)/DTNYQ, so any kernel that has to
       build a spectrum of its own on the same convention -- calc_grad_dft's
       source term, eq. (26a) -- needs tmin. */
    sprintf(build_options[*n-1],"-D TMIN=%d",(*m).tmin);
    *n+=1;
    sprintf(build_options[*n-1],"-D PRESOUT=%d",presout);
    *n+=1;
    sprintf(build_options[*n-1],"-D VELOUT=%d",velout);
    *n+=1;
    sprintf(build_options[*n-1],"-D VXOUT=%d",vout[0]);
    *n+=1;
    sprintf(build_options[*n-1],"-D VYOUT=%d",vout[1]);
    *n+=1;
    sprintf(build_options[*n-1],"-D VZOUT=%d",vout[2]);
    *n+=1;
    sprintf(build_options[*n-1],"-D GRADABS=%d",(*m).GRADABS);
    // Raw (unpadded) per-dimension grid size, matching the material
    // parameter arrays' actual layout (num_ele = prod(N), no FDORDER) --
    // distinct from the NZ/NX/etc. macros above, which are sized for the
    // FDORDER-padded wavefield arrays and would be wrong here. Used by
    // src/average_params.cl (see notes/vacuum-freesurface-plan.md, Phase 8).
    // dev->N is [NZ,NX] for 2D and [NZ,NY,NX] for 3D; PARNY is unused (left
    // at 1) for 2D's own kernels.
    *n+=1;
    sprintf(build_options[*n-1],"-D PARNZ=%d",(*dev).N[0]);
    *n+=1;
    sprintf(build_options[*n-1],"-D PARNY=%d",(m->NDIM==3) ? (*dev).N[1] : 1);
    *n+=1;
    sprintf(build_options[*n-1],"-D PARNX=%d",
            (m->NDIM==3) ? (*dev).N[2] : (*dev).N[1]);

    return state;
}

/* Kernel cache (CUDA). An entry is named by a 64-bit FNV-1a hash of the
   source, the options and the architecture: <cache_dir>/<kernel>-<hash>.bin.
   A new configuration adds entries instead of overwriting the previous ones,
   and an entry is written to a temporary file then renamed, so that processes
   sharing a cache directory never read a partial file. An empty cache_dir
   disables the cache. The exact source and options are also written to
   <kernel>-<procid>-<devid>.src/.opt (debugging: memcheck line numbers). */
static unsigned long long fnv1a(unsigned long long h, const char * s, size_t n){
    size_t i;
    for (i=0;i<n;i++){ h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}

/* NVRTC emits a CUBIN for the device (--gpu-architecture=sm_XY): no driver JIT
   at load. If this NVRTC does not know the architecture (a device newer than
   the toolkit), it falls back to PTX (compute_XY), JIT-compiled by the driver. */
static int nvrtc_build(const char * program_source, const char * program_name,
                       char ** build_options, int noptions,
                       char ** binary, size_t * size, int cubin){
    int state = 0, i;
    nvrtcProgram cuprog;
    state = nvrtcCreateProgram(&cuprog, program_source, program_name, 0, NULL, NULL);
    if (state != NVRTC_SUCCESS){
        fprintf(stderr,"Error: %s\n",clerrors(state));
        return state;
    }
    /* the architecture option written by get_build_options():
       --gpu-architecture=compute_XY, replaced by sm_XY for a CUBIN */
    const char * pre = "--gpu-architecture=compute_";
    int iarch = -1;
    char saved[500];
    for (i=0;i<noptions;i++){
        if (strncmp(build_options[i], pre, strlen(pre))==0) iarch = i;
    }
    if (cubin && iarch >= 0){
        snprintf(saved, sizeof(saved), "%s", build_options[iarch]);
        snprintf(build_options[iarch], 500, "--gpu-architecture=sm_%s",
                 saved + strlen(pre));
    }
    state = nvrtcCompileProgram(cuprog, noptions, (const char * const*)build_options);
    if (cubin && iarch >= 0){
        snprintf(build_options[iarch], 500, "%s", saved);
    }
    if (state != NVRTC_SUCCESS){
        size_t logSize = 0;
        nvrtcGetProgramLogSize(cuprog, &logSize);
        char *log = malloc(logSize+1);
        if (log){
            nvrtcGetProgramLog(cuprog, log);
            log[logSize] = 0;
            if (!cubin){
                fprintf(stdout,"Compilation of %s:\n%s", program_name, log);
            }
            free(log);
        }
        nvrtcDestroyProgram(&cuprog);
        return state;
    }
    if (cubin){
        state = nvrtcGetCUBINSize(cuprog, size);
        if (!state) *binary = malloc(*size);
        if (!state && !*binary) state = 1;
        if (!state) state = nvrtcGetCUBIN(cuprog, *binary);
    }
    else{
        state = nvrtcGetPTXSize(cuprog, size);
        if (!state) *binary = malloc(*size);
        if (!state && !*binary) state = 1;
        if (!state) state = nvrtcGetPTX(cuprog, *binary);
    }
    nvrtcDestroyProgram(&cuprog);
    return state;
}

int compile(const char *program_source,
            char * program,
            CUmodule *module,
            CUfunction *kernel,
            const char * program_name,
            char ** build_options,
            int noptions,
            const char * cache_dir,
            int devid,
            int procid)
{
    int state = 0;
    int i;
    char * binary = NULL;
    size_t size = 0;
    int use_cache = cache_dir && cache_dir[0];
    char filename_bin[PATH_MAX];
    (void)program;

    unsigned long long h = 14695981039346656037ULL;
    h = fnv1a(h, program_source, strlen(program_source) + 1);
    for (i=0;i<noptions;i++){
        h = fnv1a(h, build_options[i], strlen(build_options[i]) + 1);
    }
    if (use_cache){
        snprintf(filename_bin, sizeof(filename_bin), "%s/%s-%016llx.bin",
                 cache_dir, program_name, h);
        if (prog_read_file(&binary, &size, filename_bin) != 0){
            binary = NULL;
        }
    }
    if (!binary){
        state = nvrtc_build(program_source, program_name, build_options,
                            noptions, &binary, &size, 1);
        if (state){
            free(binary);
            binary = NULL;
            state = nvrtc_build(program_source, program_name, build_options,
                                noptions, &binary, &size, 0);
        }
        if (state) fprintf(stderr,"Error: %s\n",clerrors(state));
        if (!state && use_cache){
            char tmp[PATH_MAX+32];
            snprintf(tmp, sizeof(tmp), "%s.%d.tmp", filename_bin, (int)getpid());
            if (prog_write_file(tmp, (unsigned char *)binary, size) == 0){
                rename(tmp, filename_bin);
            }
            /* debugging copies of the source and options */
            char filename_src[PATH_MAX], filename_opt[PATH_MAX];
            char concat_options[5000] = {0};
            for (i=0;i<noptions;i++){
                strncat(concat_options, build_options[i],
                        sizeof(concat_options) - strlen(concat_options) - 1);
            }
            snprintf(filename_src, sizeof(filename_src), "%s/%s-%d-%d.src",
                     cache_dir, program_name, procid, devid);
            snprintf(filename_opt, sizeof(filename_opt), "%s/%s-%d-%d.opt",
                     cache_dir, program_name, procid, devid);
            prog_write_src(filename_src, (char *)program_source);
            prog_write_src(filename_opt, concat_options);
        }
    }
    __GUARD cuModuleLoadDataEx(module, binary, 0, 0, 0);
    free(binary);

    // Now create the kernel "objects"
    __GUARD cuModuleGetFunction(kernel, *module, program_name);
    if (state !=CUDA_SUCCESS) fprintf(stderr,"Error: %s\n",clerrors(state));

    return state;
}
#endif


int prog_source(clprogram * prog, char* name, const char * source, int nheaders, const char ** headers){
    int state =0;
    int i;
    for (i=0; i<nheaders; i++){
        strcat((*prog).src, headers[i]);
    }
    strcat((*prog).src, source);
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

/* Kernel settings measured fastest on Pascal and Ampere GPUs (notes: perf-shared-memory-pipeline).
   Each one is on by default where it applies; SEISCL_<NAME>=0 turns it off.
   * NOTILE: update and adjoint kernels without shared-memory tiles (header_vec.cl). CUDA and
     OpenCL, every FP16, one or several devices and processes per shot (domain decomposition:
     the com1 / com2 / center launches of time_stepping.c take the same offsets). The exact CPML
     adjoint then reads the field itself where the stencil box of a work-item is outside the
     layers, the filtered copy elsewhere (NT_RAW*, NT_SKIP*).
   * CPMLINL: the exact CPML adjoint is computed inside update_adjs / update_adjv (header_vec.cl)
     instead of cpml_adj_v / cpml_adj_s; the adjoint memories psi_*r are double buffered
     (Init_OpenCL.c) and cpml_adj_* is not launched (time_stepping.c). 3D, NOTILE, FP16 0,
     back_prop_type 1, one device and one process per shot. The doubled buffers must be covered
     by the vars_init launch ((N+FDORDER) per dimension); otherwise it is off.
   * MAXRREG_ADJ: under CPMLINL, at most 56 registers for update_adjv / update_adjs on compute
     capability 6.x (Pascal, CUDA); SEISCL_MAXRREG_ADJ=n[,m] sets n for update_adjv, m for update_adjs. */
static int env_on(const char * name){
    const char * v = getenv(name);
    return !(v && v[0]=='0');
}
int notile_on(model * m){
    static int env = -1;
    if (env < 0) env = env_on("SEISCL_NOTILE");
    return env;
}
int cpml_inline(model * m){
    static int env = -1;
    if (env < 0) env = env_on("SEISCL_CPMLINL");
    if (!env || !notile_on(m) || m->NUM_DEVICES!=1 || m->NLOCALP!=1 || m->ND!=3 || m->ABS_TYPE!=1 || m->BACK_PROP_TYPE!=1 || !m->GRADOUT
        || m->FP16!=0 || m->NAB<1)
        return 0;
    long a = (long)m->N[0]*m->N[1], b = (long)m->N[0]*m->N[2], c = (long)m->N[1]*m->N[2];
    long big = a > b ? a : b; big = big > c ? big : c;
    long cover = (long)(m->N[0]+m->FDORDER)*(m->N[1]+m->FDORDER)*(m->N[2]+m->FDORDER);
    return 2*2*m->NAB*big <= cover;
}

/* The options of the kernel settings above for this kernel, in opt[0..n-1] (n returned):
   -D NOTILE=1, -D CPMLINL=1, the register cap. *notile: compiled without tiles. */
static int perf_options(model * m, device * dev, clprogram * prog, char opt[3][64], int * notile){
    int n = 0;
    int upd = strcmp(prog->name, "update_v")==0 || strcmp(prog->name, "update_s")==0;
    int adj = strcmp(prog->name, "update_adjv")==0 || strcmp(prog->name, "update_adjs")==0;
    int cpa = strcmp(prog->name, "cpml_adj_v")==0 || strcmp(prog->name, "cpml_adj_s")==0;
    *notile = 0;
    if (notile_on(m) && !dev->LOCAL_OFF && (upd || adj || cpa)){
        sprintf(opt[n++], "-D NOTILE=1");
        *notile = 1;
        if (cpml_inline(m) && adj)
            sprintf(opt[n++], "-D CPMLINL=1");
    }
    /* cpml_adj_* is not launched under CPMLINL: the adjoint updates must have it */
    if (cpml_inline(m) && adj && !*notile){
        fprintf(stderr, "Error: SEISCL_CPMLINL could not be compiled into %s\n", prog->name);
        return -1;
    }
    const char * v = getenv("SEISCL_MAXRREG_ADJ");
    int nv = 0, ns = 0;
    if (v){
        nv = ns = atoi(v);
        const char * comma = strchr(v, ',');
        if (comma) ns = atoi(comma + 1);
    }
    #ifndef __SEISCL__
    else if (cpml_inline(m) && dev->cuda_arc[0] == 6)
        nv = ns = 56;
    #endif
    int cap = strcmp(prog->name, "update_adjv")==0 ? nv : strcmp(prog->name, "update_adjs")==0 ? ns : 0;
    /* A work group must fit in the 64K registers of a multiprocessor: without a cap the compiler
       may give a work-item more than 65536 / (work-group size) registers, and the launch fails
       (OpenCL: CL_OUT_OF_RESOURCES; measured: update_adjs of a boundary strip, NOTILE, FP16 1,
       3-D, 512 work-items; the interior variant uses 126 registers, the strip variant more).
       Under NOTILE only: the tiled kernels are unchanged. */
    if (*notile && prog->lsize[0] > 0){
        size_t threads = 1;
        for (int k = 0; k < prog->wdim && k < MAX_DIMS; k++) threads *= prog->lsize[k];
        int fit = (int)(65536 / threads) & ~7;
        if (fit < 255 && (cap == 0 || fit < cap)) cap = fit;
    }
    if (cap > 0){
        #ifdef __SEISCL__
        sprintf(opt[n++], "-cl-nv-maxrregcount=%d", cap);
        #else
        sprintf(opt[n++], "--maxrregcount=%d", cap);
        #endif
    }
    return n;
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
    int notile = 0;
    
    prog->context = &dev->context;
    #ifdef __SEISCL__
    const char * build_options0 = get_build_options(dev,
                                                   m,
                                                   prog->LCOMM,
                                                   prog->COMM,
                                                   prog->DIRPROP);
    /* No shared tiles, the CPML adjoint inside the adjoint updates and the register cap: perf_options
       (appended after the upper-casing of get_build_options) */
    char perf[3][64];
    int nperf = perf_options(m, dev, prog, perf, &notile);
    if (nperf < 0) return 1;
    char build_options[6400];
    snprintf(build_options, sizeof(build_options), "%s", build_options0);
    for (i=0; i<nperf; i++){
        strncat(build_options, " ", sizeof(build_options)-strlen(build_options)-1);
        strncat(build_options, perf[i], sizeof(build_options)-strlen(build_options)-1);
    }
    state = compile( (*prog).src,
                    &(*prog).prog,
                    &m->context,
                    dev->cudev,
                    &(*prog).kernel,
                    (*prog).name,
                    build_options,
                    m->cache_dir,
                    dev->DEVID,
                    m->GID);
    
    
    #else
    // 64, not 50: a typical 2D elastic config already uses 46 of the
    // previous 50 slots (measured), leaving too little headroom for any
    // future -D option addition -- get_build_options has no bounds check
    // on *n, so overflowing this silently corrupts heap memory.
    char ** build_options=NULL;
    /* 64 was already close to the ~50 options get_build_options() emits (more
       in 3D, which adds per-dimension ones), and the comment above says an
       overflow corrupts the heap silently. Widened when TMIN was added. */
    /* 128 slots; about 75 are used in 3-D (VXOUT/VYOUT/VZOUT added). The
       check after the call turns an overflow into an error: the writes
       themselves are not bounded, so keep a wide margin. */
    #define SEISCL_MAX_BUILD_OPTIONS 128
    GMALLOC(build_options, sizeof(char*)*SEISCL_MAX_BUILD_OPTIONS);
    for (i=0;i<SEISCL_MAX_BUILD_OPTIONS;i++){
        GMALLOC(build_options[i], sizeof(char)*500);
    }
    state= get_build_options(dev,
                              m,
                              build_options,
                              &noptions,
                              prog->LCOMM,
                              prog->COMM,
                              prog->DIRPROP);
    if (noptions > SEISCL_MAX_BUILD_OPTIONS){
        fprintf(stderr, "Error: %d build options for %s, more than the %d "
                        "slots (clprogram.c)\n", noptions, prog->name,
                SEISCL_MAX_BUILD_OPTIONS);
        return 1;
    }
    /* No shared tiles, the CPML adjoint inside the adjoint updates and the register cap: perf_options */
    char perf[3][64];
    int nperf = perf_options(m, dev, prog, perf, &notile);
    if (nperf < 0) return 1;
    for (i=0; i<nperf; i++){
        if (noptions >= SEISCL_MAX_BUILD_OPTIONS){
            fprintf(stderr, "Error: no build option slot left for %s (clprogram.c)\n", prog->name);
            return 1;
        }
        strcpy(build_options[noptions++], perf[i]);
    }
    
    state = compile((*prog).src,
                    (*prog).prog,
                    &(*prog).module,
                    &(*prog).kernel,
                    (*prog).name,
                    build_options,
                    noptions,
                    m->cache_dir,
                    dev->DEVID,
                    m->GID);
    if (build_options){
        /* All the slots, not only the noptions used: the others leaked
           (about 27 kB per kernel compile, at every engine build). */
        for (i=0;i<SEISCL_MAX_BUILD_OPTIONS;i++){
            GFree(build_options[i]);
        }
        GFree(build_options);
    }
    #endif
    
    if (state){
        printf("Error: Could not build kernel %s\n", (*prog).name);
        
        return state;
    }

    /* FP16 1 and 2: two cells per thread along z; the unified kernels keep
       float tiles (header_vec.cl) whatever the storage. */
    if (m->FP16==1 || m->FP16==2){
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
    /* no tiles: no shared memory (OpenCL: a 4-byte __local argument, unused) */
    if (notile){
        (*prog).shared_size = 0;
        shared_size = sizeof(float);
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
                /* f<var>_f is the resident forward DFT spectrum; f<var> alone
                 * stays bound to the adjoint one, so savefreqs and everything
                 * else keep their existing binding. */
                sprintf(str2comp,"f%s_f",(*dev).vars[j].name);
                if (strcmp(str2comp,(*prog).input_list[i])==0){
                    prog_arg(prog, i, &(*dev).vars[j].cl_fvar_f.mem, memsize);
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
            /* The kernels (init_gradsrc, update_adj*) call it "gradsrc"; only
               "grad_src" was matched, so the source gradient was bound to the
               4-byte placeholder and every write went out of bounds
               (notes/gpu-memory-audit.md, B2). */
            else if (strcmp("gradsrc" ,(*prog).input_list[i])==0
                     || strcmp("grad_src" ,(*prog).input_list[i])==0){
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
            if (strcmp("LOCARG2",(*prog).input_list[i])==0){
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
            if (strcmp("par_scale"  ,(*prog).input_list[i])==0){
                prog_arg(prog, i, &m->par_scale, sizeof(int));
                argfound=1;
            }
        }
        
        if (!argfound){
            /* SEISCL_WARN_ARGS=1 reports these. An unmatched argument is
             * bound to a placeholder that is not a valid memory object, so the
             * kernel either launches with garbage or fails with
             * CL_INVALID_KERNEL_ARGS far from the cause -- which is what made
             * the ND=21 gradient path hard to diagnose. Off by default because
             * kernels declare every possible argument by convention (the CPML
             * coefficients, for instance, are legitimately unmatched whenever
             * ABS_TYPE != 1), so it is noisy rather than wrong. */
            {
                static int warn=-1;
                if (warn<0){
                    const char * v = getenv("SEISCL_WARN_ARGS");
                    warn = (v && v[0] && v[0]!='0') ? 1 : 0;
                }
                if (warn)
                    fprintf(stderr,"Warning: argument %s of kernel %s matched "
                                   "nothing and is bound to a placeholder\n",
                                   (*prog).input_list[i], (*prog).name);
            }
            prog_arg(prog, i, &dev->cuda_null, memsize);
        }

    }
    

    dev->progs[dev->nprogs]=prog;
    dev->nprogs++;
    
    return state;
}

    
int prog_launch(QUEUE *inqueue, clprogram * prog){
    
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
    if (state != CL_SUCCESS){
        /* what the launch asked against what the kernel allows: a CL_OUT_OF_RESOURCES is either
           a work group too large for this build of the kernel or an earlier kernel's fault.
           NVIDIA's CL_KERNEL_WORK_GROUP_SIZE is conservative (256 for kernels that run 512), so it
           is printed, not used to decide. */
        size_t wgmax = 0; cl_ulong lmem = 0, pmem = 0; cl_device_id dv;
        clGetCommandQueueInfo(*inqueue, CL_QUEUE_DEVICE, sizeof(dv), &dv, NULL);
        clGetKernelWorkGroupInfo(prog->kernel, dv, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgmax), &wgmax, NULL);
        clGetKernelWorkGroupInfo(prog->kernel, dv, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, NULL);
        clGetKernelWorkGroupInfo(prog->kernel, dv, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(pmem), &pmem, NULL);
        fprintf(stderr, "    %s: global %zu %zu %zu, local %zu %zu %zu (wdim %d); kernel allows a work group "
                "of %zu, local memory %llu B, private %llu B\n", prog->name,
                prog->gsize[0], prog->wdim>1 ? prog->gsize[1] : 1, prog->wdim>2 ? prog->gsize[2] : 1,
                lsize ? lsize[0] : 0, lsize && prog->wdim>1 ? lsize[1] : 0, lsize && prog->wdim>2 ? lsize[2] : 0,
                (int)prog->wdim, wgmax, (unsigned long long)lmem, (unsigned long long)pmem);
    }
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
    /* Wait on all nwait events, not just waits[0]: the OpenCL branch above
     * honours nwait, so silently dropping the rest here would desynchronize
     * the CUDA path for any kernel with more than one dependency. */
    for (i=0;i<prog->nwait;i++){
        state = cuStreamWaitEvent(*inqueue, prog->waits[i], 0);
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
    if (prog->outevent){
        state  = cuCtxSetCurrent(*prog->context);
        if (!prog->event){
            state =  cuEventCreate(&prog->event, CU_EVENT_DISABLE_TIMING);
        }
        state = cuEventRecord( prog->event, *inqueue);
    }
    #endif
    

    if (state !=CUCL_SUCCESS) {fprintf(stderr,"Error launching %s: %s\n",
                                       prog->name,clerrors(state));    }
    
    return state;
}
