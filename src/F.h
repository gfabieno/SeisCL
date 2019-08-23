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
#include <stdint.h>
//#include <cmath>

//#include <libc.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pwd.h>
//#include <mach/mach_time.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>

#include "third_party/KISS_FFT/kiss_fft.h"
#include "third_party/KISS_FFT/kiss_fftr.h"

#include "CUDA_CL.h"

#include <mpi.h>
#include <hdf5.h>

#define STRING_SIZE 256
#define PI (3.141592653589793238462643383279502884197169)

#define GMALLOC(x,y) ({\
            if (!state) if (!((x)=malloc((y)))) {state=1;fprintf(stderr,"Error: malloc failed at line %d in %s()\n",__LINE__,__func__);};\
            if (!state) memset((x),0,(y));\
            })

#define GFree(x) if ((x)) free( (x) );(x)=NULL;

#define __GUARD if (state) {return state;} else state=
#define CLGUARD(x) if (!state) if (!(state = (x) )) {fprintf(stderr,"Error: OpenCL function failed at line %d in %s()\n",__LINE__,__func__);};



#define MAX_DIMS 10
#define MAX_KERNELS 100
#define MAX_KERN_STR 200000
#define BLOCK_SIZE 256
#define MAX_FD_ORDER 12
//#define __DEBUGGING__



struct device;

struct model;

struct filenames {
    char model[1024];
    char csts[1024];
    char dout[1024];
    char din[1024];
    char gout[1024];
    char rmsout[1024];
    char movout[1024];
    char res[1024];
};


/* _____________Structure to intereact with OpenCL memory buffers ____________*/
typedef struct clbuf {
    
    MEM mem;
    size_t size;
    
    MEM pin;
    size_t sizepin;
    float * host;
    int free_host;
    int free_pin;
    
    CONTEXT * context;
    int outevent_r;
    int outevent_s;
    EVENT event_r;
    EVENT event_s;
    
    int nwait_r;
    EVENT * waits_r;
    int nwait_s;
    EVENT * waits_s;
    
} clbuf;

CL_INT clbuf_send(QUEUE *inqueue,  clbuf * buf);

CL_INT clbuf_sendfrom(QUEUE *inqueue, clbuf * buf, void * ptr);

CL_INT clbuf_read(QUEUE *inqueue, clbuf * buf);

CL_INT clbuf_readto(QUEUE *inqueue,
                 clbuf * buf,
                 void * ptr);

CL_INT clbuf_create(CONTEXT *incontext, clbuf * buf);

CL_INT clbuf_create_pin(CONTEXT *incontext, QUEUE *inqueue,clbuf * buf);


/* ____________________Structure to execute OpenCL kernels____________________*/


typedef struct clprogram {
    
    const char * name;
    char src[MAX_KERN_STR];
    PROGRAM prog;
    MODULE module;
    KERNEL kernel;
    CONTEXT * context;
    char ** input_list;
    int ninputs;
    void * inputs[2000];
    int tinput;
    int pdir;
    int nsinput;
    int nrinput;
    int scinput;
    int rcinput;
    size_t lsize[MAX_DIMS];
    size_t gsize[MAX_DIMS];
    size_t bsize[MAX_DIMS];
    size_t shared_size;
    int wdim;
    
    int OFFCOMM;
    int LCOMM;
    int COMM;
    int DIRPROP;

    int outevent;
    EVENT event;
    
    int nwait;
    EVENT * waits;
    
} clprogram;


int prog_source(clprogram * prog,
                char* name,
                const char * source,
                int nheaders,
                const char ** headers);

int prog_launch( QUEUE *inqueue, clprogram * prog);

int prog_create(struct model * m, struct device * dev,clprogram * prog);

int prog_arg(clprogram * prog, int i, void * mem, int size);


/* ___________Structure for variables, or what is to be modelled______________*/
typedef struct variable{
    
    const char * name;
    
    clbuf cl_var;
    clbuf cl_varout;
    clbuf cl_varbnd;
    clbuf cl_fvar;
    clbuf cl_fvar_adj;
    clbuf cl_buf1;
    clbuf cl_buf2;
    clbuf cl_var_res;

    float **    gl_varout;
    float **    gl_varin;
    float   *   gl_mov;
    float **    gl_var_res;
    
    int       to_output;
    int       for_grad;
    int  to_comm;
    int num_ele;
    int active;
    int scaler;
    
    int n2ave;
    const char ** var2ave;
    
    void (*set_size)(int* , void *, void *);
    
} variable;

int var_alloc_out(float *** var, struct model *m );
int var_res_raw(struct model * m, int s);
int rtm_res(struct model * m, int s);
int res_scale(struct model * m, int s);

/* _____________Structure for parameters, or what can be inverted_____________*/
typedef struct parameter{
    
    const char * name;
    
    clbuf   cl_par;
    clbuf   cl_grad;
    clbuf   cl_H;
    float * gl_par;
    float * gl_grad;
    float * gl_H;
    int num_ele;
    int active;
    float scaler;
    
    const char * to_read;
    int to_grad;
    void (*transform)(void *);


} parameter;

int calc_grad(struct model * m, struct device * dev);
int transf_grad(struct model * m);

/* ____Structure for constants, which vectors broadcasted to all devices______*/
typedef struct constants{
    
    const char * name;
    
    clbuf   cl_cst;
    float  * gl_cst;
    int num_ele;
    const char * to_read;

    void (*transform)(void *, void *, int);
    
} constants;

/* ______________Structure that control sources and receivers ________________*/
typedef struct sources_records{

    clbuf cl_src;
    clbuf cl_src_pos;
    clbuf cl_rec_pos;
    clbuf cl_grad_src;
    
    clprogram sources;
    clprogram varsout;
    clprogram varsoutinit;
    clprogram residuals;
    clprogram init_gradsrc;

    int ns;
    int nsmax;
    int ngmax;
    int allng;
    int allns;
    int smin;
    int smax;
    int *nsrc;
    int *nrec;
    int * src_scales;
    int * res_scales;
    float **src;
    float **gradsrc;
    float **src_pos;
    float **rec_pos;

} sources_records;

/* ________________Structure that defines an update step _____________________*/
typedef struct update{

    const char * name;

    clprogram center;
    clprogram com1;
    clprogram com2;
    clprogram fcom1_out;
    clprogram fcom2_out;
    clprogram fcom1_in;
    clprogram fcom2_in;
    
    int nvcom;
    variable ** v2com;
    

} update;

/* _____________Structure that defines the boundary conditions _______________*/
typedef struct boundary_conditions{
    
    clprogram surf;
    clprogram surf_adj;
    clprogram init_f;
    clprogram init_adj;

} boundary_conditions;

/* _____________Structure that defines the gradient_______________*/
typedef struct gradients {

    clprogram init;
    clprogram savefreqs;
    clprogram initsavefreqs;
    clprogram savebnd;

} gradients;


/* _____________Structure that holds all information of a device _____________*/
typedef struct device {
    
    QUEUE queue;
    QUEUE queuecomm;
    MEM cuda_null;

    int workdim;
    int NDIM;
    int N[MAX_DIMS];
    char * N_names[MAX_DIMS];
    int OFFSET;
    int DEVID;
    int NBND;
    int par_scale;
    
    int LOCAL_OFF;
    int FP16;
    int cuda_arc[2];
    
    clprogram * progs[MAX_KERNELS];
    int nprogs;

    variable * vars;
    variable * vars_adj;
    int nvars;
    parameter * pars;
    int npars;
    constants * csts;
    int ncsts;
    
    variable * trans_vars;
    int ntvars;
    
    update * ups_f;
    update * ups_adj;
    int nupdates;
    
    sources_records src_recs;
    gradients grads;
    boundary_conditions bnd_cnds;
    
    CONTEXT context;
    CONTEXT * context_ptr;
    DEVICE cudev;

} device;

/* _____________Structure that holds all information of a MPI process_________*/
typedef struct model {
    
    char cache_dir[PATH_MAX];

    variable * vars;
    variable * vars_adj;
    int nvars;
    parameter * pars;
    int npars;
    constants * csts;
    int ncsts;
    
    variable * trans_vars;
    int ntvars;
    
    update * ups_f;
    update * ups_adj;
    int nupdates;
    
    sources_records src_recs;
    gradients grads;
    boundary_conditions bnd_cnds;

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
    int INPUTRES;
    int L;

    int ND;
    int tmax;
    int tmin;
    int NTNYQ;
    int DTNYQ;
    int NDIM;
    
    int GID; //The global MPI Process ID
    int GNP; //The global number of MPI Processes
    int LID; //The local MPI Process ID for this node
    int LNP;   //The local number MPI Processes for this node
    int NGROUP;
    int MYGROUPID;
    int MYLOCALID;
    int MPI_NPROC_SHOT;
    int NLOCALP;
    int MPI_INIT;
    MPI_Comm mpigroupcomm;
    
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
    DEVICE_TYPE pref_device_type;
    DEVICE_TYPE device_type;
    CL_UINT NUM_DEVICES;
    CONTEXT context;
    int FP16;
    int halfpar;
    int par_scale;

    int (*res_calc)(struct model * , int );
    int (*res_scale)(struct model * , int );
    int (*check_stability)(void *);
    int (*set_par_scale)(void *);

} model;

int append_update(update * up, int * ind, char * name, const char * source,
                  int nheaders, const char ** headers);
int append_var(model * m,
               int *ind,
               const char * name,
               int for_grad,
               int to_comm,
               void (*set_size)(int* , void *, void *));
int append_par(model * m,
               int *ind,
               const char * name,
               const char * to_read,
               void (*transform)(void *));
int append_cst(model * m,
               const char * name,
               const char * to_read,
               int num_ele,
               void (*transform)(void *, void *, int));
constants * get_cst(constants * csts, int ncsts, const char * name);
parameter * get_par(parameter * pars, int npars, const char * name);
variable * get_var(variable * vars, int nvars, const char * name);

/* __________________________SeisCL functions________________________________*/

int readhdf5(struct filenames files, model * m);

int assign_modeling_case(model * m);


int Init_cst(model * m);

int Init_model(model * m);

int Init_MPI(model * m);

int Init_CUDA(model * m, device ** dev);

int event_dependency( model * m,  device ** dev, int adj);

int time_stepping(model * m, device ** dev);

int comm(model * m, device ** dev, int adj, int ui);

int Out_MPI(model * m);

int writehdf5(struct filenames file, model * m);

int Free_OpenCL(model * m, device * dev) ;

const char *clerrors(int err);


/* __________________________Data Processing________________________________*/

int butterworth(float * data,
                float fcl,
                float fch,
                float dt,
                int NT,
                int tmax,
                int ntrace,
                int order);

/* ______________________Automatic kernels functions__________________________*/
int kernel_varout(device * dev,
                  clprogram * prog);

int kernel_varoutinit(device * dev,
                      clprogram * prog);

int kernel_varinit(device * dev,
                   model * m,
                   variable * vars,
                   clprogram * prog,
                   int BACK_PROP_TYPE);

int kernel_residuals(device * dev,
                     clprogram * prog,
                     int BACK_PROP_TYPE);

int kernel_gradinit(device * dev,
                    parameter * pars,
                    clprogram * prog);

int kernel_initsavefreqs(device * dev,
                         variable * vars,
                         clprogram * prog);

int kernel_savefreqs(device * dev,
                     variable * vars,
                     clprogram * prog);

int kernel_init_gradsrc(clprogram * prog);

int kernel_fcom_out(device * dev,
                    variable * vars,
                    clprogram * prog,
                    int upid,
                    int buff12,
                    int adj);

int kernel_fcom_in(device * dev,
                   variable * vars,
                   clprogram * prog,
                   int upid,
                   int buff12,
                   int adj);

int kernel_sources(model * m,
                   device * dev,
                   clprogram * prog);




