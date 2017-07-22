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

/* This is a collection of utility functions for OpenCL */



#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
//#include <cmath>

//#include <libc.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
//#include <mach/mach_time.h>

#include "kiss_fft.h"
#include "kiss_fftr.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <mpi.h>
#include <hdf5.h>


#define STRING_SIZE 256
#define PI (3.141592653589793238462643383279502884197169)

#define GMALLOC(x,y) ({\
            if (!state) if (!((x)=malloc((y)))) {state=1;fprintf(stderr,"malloc failed at line %d in %s()\n",__LINE__,__func__);};\
            if (!state) memset((x),0,(y));\
            })

#define GFree(x) if ((x)) free( (x) );(x)=NULL;

#define __GUARD if (!state) state=
#define CLGUARD(x) if (!state) if (!(state = (x) )) {fprintf(stderr,"OpenCL function failed at line %d in %s()\n",__LINE__,__func__);};

#define MAX_DIMS 10
#define MAX_KERNELS 100
#define MAX_KERN_STR 10000

struct filenames {
    char model[1024];
    char csts[1024];
    char dout[1024];
    char din[1024];
    char gout[1024];
    char RMSOUT[1024];
    char MOVOUT[1024];
};


struct clbuf {
    
    cl_mem mem;
    size_t size;
    
    cl_mem pin;
    float * host;
    
    int outevent_r;
    int outevent_s;
    cl_event event_r;
    cl_event event_s;
    
    int nwait_r;
    cl_event * waits_r;
    int nwait_s;
    cl_event * waits_s;
    
};


struct clprogram {
    
    const char * name;
    const char * src;
    cl_program prog;
    cl_kernel kernel;
    char ** input_list;
    int ninputs;
    int tinput;
    size_t lsize[MAX_DIMS];
    size_t gsize[MAX_DIMS];
    int local;
    int NDIM;
    
    int OFFCOMM;
    int LCOMM;
    int COMM;
    int DIRPROP;

    int outevent;
    cl_event event;
    
    int nwait;
    cl_event * waits;
    
};

struct variable{
    
    const char * name;
    
    struct clbuf cl_var;
    struct clbuf cl_varout;
    struct clbuf cl_varbnd;
    struct clbuf cl_fvar;
    struct clbuf cl_buf1;
    struct clbuf cl_buf2;
    struct clbuf cl_var_res;

    cl_float2 * gl_fvar;
    float **    gl_varout;
    float **    gl_varin;
    float   *   gl_mov;
    float **    gl_var_res;
    int       to_output;
    int       for_grad;
    int  to_comm;
    int num_ele;
    
};

struct parameter{
    
    const char * name;
    
    struct clbuf   cl_par;
    struct clbuf   cl_grad;
    struct clbuf   cl_H;
    float * gl_par;
    float * gl_grad;
    float * gl_H;
    int num_ele;
    
    const char * to_read;
    int to_grad;
    int (*transform)(float *);

};

struct constants{
    
    const char * name;
    
    struct clbuf   cl_cst;
    float  * gl_cst;
    int num_ele;
    const char * to_read;
    int active;

    int (*transform)(float *);
    
};


struct sources_records{

    struct clbuf cl_src;
    struct clbuf cl_src_pos;
    struct clbuf cl_rec_pos;
    struct clbuf cl_grad_src;;
    
    struct clprogram sources;
    struct clprogram varsout;
    struct clprogram varsoutinit;
    struct clprogram residuals;
    struct clprogram init_gradsrc;

    int ns;
    int nsmax;
    int ngmax;
    int allng;
    int allns;
    int smin;
    int smax;
    int *nsrc;
    int *nrec;
    float **src;
    float **gradsrc;
    float **src_pos;
    float **rec_pos;

};

struct update{

    const char * name;

    struct clprogram center;
    struct clprogram com1;
    struct clprogram com2;
    struct clprogram fcom1_out;
    struct clprogram fcom2_out;
    struct clprogram fcom1_in;
    struct clprogram fcom2_in;
    
    int nvcom;
    struct variable ** v2com;
    
    cl_event event_readMPI1[6];
    cl_event event_readMPI2[6];
    
    cl_event event_read1;
    cl_event event_read2;
    cl_event event_write1;
    cl_event event_write2;
    cl_event event_update_com1;
    cl_event event_update_com2;

};

struct boundary_conditions{
    
    struct clprogram surf;
    struct clprogram init_f;
    struct clprogram init_adj;

};

struct gradients {

    struct clprogram init;
    struct clprogram savefreqs;
    struct clprogram initsavefreqs;
    struct clprogram savebnd;

};

struct varcl {
    
    cl_command_queue queue;
    cl_command_queue queuecomm;

    int workdim;
    int NDIM;
    int N[MAX_DIMS];
    char * N_names[MAX_DIMS];
    int NX0;
    int OFFSET;
    int OFFSETfd;
    int DEV;
    int NBND;
    
    int LOCAL_OFF;
    
    struct clprogram * progs[MAX_KERNELS];
    int nprogs;

    struct variable * vars;
    struct variable * vars_adj;
    int nvars;
    struct parameter * pars;
    int npars;
    struct constants * csts;
    int ncsts;
    
    struct variable * trans_vars;
    int ntvars;
    
    struct update * ups_f;
    struct update * ups_adj;
    int nupdates;
    
    struct sources_records src_recs;
    struct gradients grads;
    struct boundary_conditions bnd_cnds;

};


// Structure containing all seismic parameters
struct modcsts {
    
    struct variable * vars;
    struct variable * vars_adj;
    int nvars;
    struct parameter * pars;
    int npars;
    struct constants * csts;
    int ncsts;
    
    struct variable * trans_vars;
    int ntvars;
    
    struct update * ups_f;
    struct update * ups_adj;
    int nupdates;
    
    struct sources_records src_recs;
    struct gradients grads;
    struct boundary_conditions bnd_cnds;

    int NXP;
    int NT;
    int FDORDER;
    int FDOH;
    int MAXRELERROR;
    int GRADOUT;
    int GRADSRCOUT;
    int HOUT;
    int VARSOUT;
    int MOVOUT;
    int RESOUT;
    int RMSOUT;
    int L;
    int MYID;
    int NP;


    int ND;
    int tmax;
    int tmin;
    int NTNYQ;
    int DTNYQ;
    int NDIM;
    
    int NGROUP;
    int MYGROUPID;
    int MYLOCALID;
    int MPI_NPROC_SHOT;
    int NLOCALP;
    int MPI_INIT;
    
    int BACK_PROP_TYPE;
    int par_type;
    int NFREQS;

    float rms;
    float rmsnorm;
    float fmin, fmax;
    
    int scalerms;
    int scaleshot;
    int scalermsnorm;
    
    float TAU;
    float f0;
    
    float vpmax;
    float vsmin;

    float dt;
    float dh;

    int NAB;
    int FREESURF;
    int ABS_TYPE;
    float VPPML;
    float FPML;
    float NPOWER;
    float K_MAX_CPML;
    float abpc;
    
    double hc[7];

    int restype;
    
    int N[MAX_DIMS];
    char * N_names[MAX_DIMS];
   
    
    int nmax_dev;
    int *no_use_GPUs;
    int n_no_use_GPUs;
    cl_device_type pref_device_type;
    cl_device_type device_type;
    cl_uint NUM_DEVICES;
    cl_context context;
    size_t buffer_size_comm;

    int (*res_calc)(struct modcsts * , int );

};

// SeisCL function definition

int toMPI(struct modcsts * mptr);

int Init_cst(struct modcsts * m);

int Init_model(struct modcsts * m);

int Init_OpenCL(struct modcsts * m, struct varcl ** vcl);

int Free_OpenCL(struct modcsts * m, struct varcl ** vcl) ;

int time_stepping(struct modcsts * m, struct varcl ** vcl);

int readhdf5(struct filenames files, struct modcsts * m);

int Init_MPI(struct modcsts * m);

int writehdf5(struct filenames file, struct modcsts * m);

int Free_MPI(struct modcsts * m) ;

int Out_MPI(struct filenames file, struct modcsts * m);

int assign_modeling_case(struct modcsts * m);

int assign_var_size(int* N, int NDIM, int FDORDER, int numvar, int L, struct variable * vars);


cl_int GetPlatformID( cl_device_type * pref_device_type, cl_device_type * device_type, cl_platform_id* clsel_plat_id, cl_uint  *outnum_devices, int n_no_use_GPUs, int * no_use_GPUs);

cl_int connect_allgpus(struct varcl ** vcl, cl_context *incontext, cl_device_type * device_type, cl_platform_id* clsel_plat_id, int n_no_use_GPUs, int * no_use_GPUs, int nmax_dev);

cl_int get_device_num(cl_uint * NUM_DEVICES);

cl_int create_gpu_kernel(const char * filename, cl_program *program, cl_context *context, cl_kernel *kernel, const char * program_name, const char * build_options);

cl_int create_gpu_kernel_from_string(const char *program_source, cl_program *program, cl_context *context, cl_kernel *kernel, const char * program_name, const char * build_options);

cl_int clbuf_send(cl_command_queue *inqueue, struct clbuf * buf);

cl_int clbuf_sendpin( cl_command_queue *inqueue,
                     struct clbuf * buf,
                     struct clbuf * bufpin,
                     int offset);

cl_int clbuf_read(cl_command_queue *inqueue, struct clbuf * buf);

cl_int clbuf_readpin( cl_command_queue *inqueue,
                     struct clbuf * buf,
                     struct clbuf * bufpin,
                     int offset);

cl_int clbuf_create(cl_context *incontext, struct clbuf * buf);

cl_int clbuf_create_cst(cl_context *incontext, struct clbuf * buf);

cl_int clbuf_create_pin(cl_context *incontext, cl_command_queue *inqueue,
                        struct clbuf * buf);

cl_int prog_launch( cl_command_queue *inqueue, struct clprogram * prog);

double machcore(uint64_t endTime, uint64_t startTime);

char *cl_err_code(cl_int err);

int calc_grad(struct modcsts* m);

int calc_Hessian(struct modcsts* mglob);

int butterworth(float * data, float fcl, float fch, float dt, int NT, int tmax, int ntrace, int order);

int res_raw(struct modcsts * mptr, int s);
int res_amp(struct modcsts * mptr, int s);

int alloc_seismo(float *** var, struct modcsts *m );

int split (const char *str, char c, char ***arr);

int extract_args(const char *str, char *name, char *** argnames, int * ninputs);

int create_kernel(struct modcsts * m, struct varcl * vcl,  struct clprogram * prog);

int prog_source(struct clprogram * prog,
                char* name,
                const char * source);

int kernel_varout(struct varcl * vcl,
                  struct variable * vars,
                  struct clprogram * prog);

int kernel_varoutinit(struct varcl * vcl,
                      struct variable * vars,
                      struct clprogram * prog);

int kernel_varinit(struct varcl * vcl,
                   struct variable * vars,
                   struct clprogram * prog);

int kernel_residuals(struct varcl * vcl,
                     struct variable * vars,
                     struct clprogram * prog);

int kernel_gradinit(struct varcl * vcl,
                    struct parameter * pars,
                    struct clprogram * prog);

int kernel_initsavefreqs(struct varcl * vcl,
                         struct variable * vars,
                         struct clprogram * prog);

int kernel_savefreqs(struct varcl * vcl,
                     struct variable * vars,
                     struct clprogram * prog);

int kernel_init_gradsrc(struct clprogram * prog);

int kernel_fcom_out(struct varcl * vcl,
                    struct variable * vars,
                    struct clprogram * prog,
                    int upid,
                    int buff12);

int kernel_fcom_in(struct varcl * vcl,
                   struct variable * vars,
                   struct clprogram * prog,
                   int upid,
                   int buff12);

int kernel_sources(struct varcl * vcl,
                   struct variable * vars,
                   struct clprogram * prog);

int event_dependency( struct modcsts * m,  struct varcl ** vcl, int adj);

int comm(struct modcsts * m, struct varcl ** vcl, int adj, int ui);
