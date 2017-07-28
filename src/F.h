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
};


/* _____________Structure to intereact with OpenCL memory buffers ____________*/
typedef struct clbuf {
    
    cl_mem mem;
    size_t size;
    
    cl_mem pin;
    float * host;
    int free_host;
    
    int outevent_r;
    int outevent_s;
    cl_event event_r;
    cl_event event_s;
    
    int nwait_r;
    cl_event * waits_r;
    int nwait_s;
    cl_event * waits_s;
    
} clbuf;

cl_int clbuf_send(cl_command_queue *inqueue, clbuf * buf);

cl_int clbuf_sendpin( cl_command_queue *inqueue,
                     clbuf * buf,
                     clbuf * bufpin,
                     int offset);

cl_int clbuf_read(cl_command_queue *inqueue, clbuf * buf);

cl_int clbuf_readpin( cl_command_queue *inqueue,
                     clbuf * buf,
                     clbuf * bufpin,
                     int offset);

cl_int clbuf_create(cl_context *incontext, clbuf * buf);

cl_int clbuf_create_cst(cl_context *incontext, clbuf * buf);

cl_int clbuf_create_pin(cl_context *incontext, cl_command_queue *inqueue,
                        clbuf * buf);


/* ____________________Structure to execute OpenCL kernels____________________*/


typedef struct clprogram {
    
    const char * name;
    const char * src;
    cl_program prog;
    cl_kernel kernel;
    char ** input_list;
    int ninputs;
    int tinput;
    int pdir;
    size_t lsize[MAX_DIMS];
    size_t gsize[MAX_DIMS];
    int local;
    int wdim;
    
    int OFFCOMM;
    int LCOMM;
    int COMM;
    int DIRPROP;

    int outevent;
    cl_event event;
    
    int nwait;
    cl_event * waits;
    
} clprogram;

int prog_source(clprogram * prog,
                char* name,
                const char * source);

int prog_launch( cl_command_queue *inqueue, clprogram * prog);

int prog_create(struct model * m, struct device * dev,clprogram * prog);


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
    
} variable;

int var_alloc_out(float *** var, struct model *m );
int var_res_raw(struct model * m, int s);

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
    
    const char * to_read;
    int to_grad;
    int (*transform)(float *);

} parameter;

int par_calc_grad(struct model * m, struct device * dev);


/* ____Structure for constants, which vectors broadcasted to all devices______*/
typedef struct constants{
    
    const char * name;
    
    clbuf   cl_cst;
    float  * gl_cst;
    int num_ele;
    const char * to_read;
    int active;

    int (*transform)(float *);
    
} constants;

/* ______________Structure that control sources and receivers ________________*/
typedef struct sources_records{

    clbuf cl_src;
    clbuf cl_src_pos;
    clbuf cl_rec_pos;
    clbuf cl_grad_src;;
    
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
    
    cl_command_queue queue;
    cl_command_queue queuecomm;

    int workdim;
    int NDIM;
    int N[MAX_DIMS];
    char * N_names[MAX_DIMS];
    int NX0;
    int OFFSET;
    int OFFSETfd;
    int DEVID;
    int NBND;
    
    int LOCAL_OFF;
    
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

} device;

/* _____________Structure that holds all information of a MPI process_________*/
typedef struct model {
    
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

    int (*res_calc)(struct model * , int );

} model;


/* __________________________SeisCL functions________________________________*/

int readhdf5(struct filenames files, model * m);

int assign_modeling_case(model * m);

int assign_var_size(int* N,
                    int nab,
                    int NDIM,
                    int FDORDER,
                    int numvar,
                    int L, variable * vars);

int Init_cst(model * m);

int Init_model(model * m);

int Init_MPI(model * m);

int Init_OpenCL(model * m, device ** dev);

int event_dependency( model * m,  device ** dev, int adj);

int time_stepping(model * m, device ** dev);

int comm(model * m, device ** dev, int adj, int ui);

int Out_MPI(model * m);

int writehdf5(struct filenames file, model * m);

int Free_OpenCL(model * m, device ** dev) ;

char *clerrors(int err);


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
                  variable * vars,
                  clprogram * prog);

int kernel_varoutinit(device * dev,
                      variable * vars,
                      clprogram * prog);

int kernel_varinit(device * dev,
                   variable * vars,
                   clprogram * prog);

int kernel_residuals(device * dev,
                     variable * vars,
                     clprogram * prog);

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
                    int buff12);

int kernel_fcom_in(device * dev,
                   variable * vars,
                   clprogram * prog,
                   int upid,
                   int buff12);

int kernel_sources(device * dev,
                   variable * vars,
                   clprogram * prog);




