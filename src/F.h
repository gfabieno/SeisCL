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
#include <unistd.h>
#include <errno.h>
//#include <cmath>

//#include <libc.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
//#include <sys/sysctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pwd.h>
//#include <mach/mach_time.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>
#include <sys/time.h>

#include "third_party/KISS_FFT/kiss_fft.h"
#include "third_party/KISS_FFT/kiss_fftr.h"

#include "CUDA_CL.h"

#ifndef __NOMPI__
#include <mpi.h>
#endif

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

#ifdef __llvm__
#pragma GCC diagnostic ignored "-Wdangling-else"
#endif

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
    char checkpoint[1024];
    char dftout[1024];
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
CL_INT clbuf_copy(QUEUE *inqueue, clbuf * src, clbuf * dst);

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
    clbuf cl_fvar;      /* adjoint DFT spectrum (device + host mirror) */
    clbuf cl_fvar_f;    /* forward DFT spectrum, device side (BACK_PROP_TYPE==2
                         * with the on-device correlation). Kept resident so the
                         * correlation can run on the device instead of round
                         * tripping both spectra over PCIe. */
    clbuf cl_fvar_adj;  /* host-only mirror of the adjoint spectrum */
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
/* transf_grad's three jobs, separately: storage format, units, and
   parameterization. See calc_grad.c for why they are not one function. */
int unpack_par_fp16(struct model * m);
int unscale_par(struct model * m);
int unscale_grad(struct model * m);
int chain_rule_par_type(struct model * m);
/* Transpose of the material-parameter averaging: folds the staggered
   gradients (rip/rkp/muipkp/...) back onto the cell-centred ones. Runs before
   unscale_par, whose output its Jacobians would otherwise be evaluated at. */
int average_grad_transpose(struct model * m);

/* ____Structure for constant vectors broadcasted to all devices______*/
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
    clprogram calc_grad;   /* on-device DFT correlation (BACK_PROP_TYPE==2) */

} gradients;

/* _____GPU port of the staggered-grid material-parameter averaging_______
   (src/average_params.cl -- ave_rip/ave_rkp/ave_muipkp), replacing the CPU
   loop in assign_modeling_case.c's ave_arithmetic_rho()/ave_harmonic_mu()
   for the 2D elastic case. See notes/vacuum-freesurface-plan.md, Phase 8.*/
typedef struct param_avg {

    clprogram rip;
    clprogram rjp;
    clprogram rkp;
    clprogram muipkp;
    clprogram muipjp;
    clprogram mujpkp;

} param_avg;


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
    param_avg par_avg;
    
    CONTEXT context;
    CONTEXT * context_ptr;
    DEVICE cudev;

} device;

/* _____________Structure that holds all information of a MPI process_________*/
typedef struct model {
    
    char cache_dir[PATH_MAX];
    int printkernels;
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
    param_avg par_avg;

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
    /* Debug: dump the raw forward and adjoint DFT wavefield buffers
     * (BACK_PROP_TYPE==2 only) so they can be checked against a reference DFT
     * independently of the gradient correlation. Single device, single shot. */
    int DFTOUT;
    /* Oversampling factor of max(gradfreqs) at which savefreqs accumulates.
     * Default 64, which is what the hardcoded 0.0156 used to give. */
    float dft_osamp;
    int INPUTRES;
    /* Skip the HDF5 boundary checkpoint entirely, on both the writing
     * (GRADOUT=0) and reading (GRADOUT=1) leg of the INPUTRES=1 two-call
     * gradient protocol. Only valid when both legs run against the same
     * model/device state, so that the boundary wavefield the adjoint pass
     * needs is already resident in the buffers the checkpoint would have
     * round-tripped through -- see SeisCL/torch/bindings.cpp, which is the
     * only caller able to guarantee that. Defaults to 0 (write and read the
     * file, the standalone SeisCL_MPI behaviour) via the memset of model. */
    int SKIP_CHECKPOINT_FILE;
    /* Keep the boundary checkpoint in a RAM-backed HDF5 file instead of one
     * on disk. Unlike SKIP_CHECKPOINT_FILE this works for any number of
     * shots, since the per-shot datasets are still written -- just not to
     * disk. CKPT_FILE_ID holds that file across the two time_stepping()
     * calls of the gradient protocol and is owned by the caller, which must
     * close it (SeisCL/torch/engine_handle.cpp does so when the engine is
     * freed). Both default to 0, i.e. the ordinary on-disk behaviour. */
    int CKPT_IN_MEMORY;
    hid_t CKPT_FILE_ID;
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
    #ifndef __NOMPI__
    MPI_Comm mpigroupcomm;
    #endif
    
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
void set_freesurf2_vacuum(void * mptr);

/* __________________________SeisCL functions________________________________*/

int readhdf5(struct filenames files, model * m);
int readvar(hid_t file_id, hid_t memtype, const char * invar, void * varptr);
int checkexists(hid_t file_id, const char * invar);

/* In-memory counterpart of read_srcrec() (src/read_hdf5.c), used by the
 * PyTorch binding (src/seiscl_api.c) instead of reading HDF5 files. */
int seiscl_set_srcrec(model * m,
                      const float * src_pos, int allns,
                      const float * src,
                      const float * rec_pos, int allng);

int assign_modeling_case(model * m);


int Init_cst(model * m);

int Init_model(model * m);

/* Allocation-free part of Init_model(): recomputes host parameter values
 * (scaling, transforms, stability check, FP16 conversion) from freshly
 * supplied raw values. Used to refresh a reused engine build. */
int Init_model_values(model * m);

int Init_data(model * m);

#ifndef __NOMPI__
int Init_MPI(model * m);
#endif

int Init_CUDA(model * m, device ** dev);

int event_dependency( model * m,  device ** dev, int adj);

int time_stepping(model * m, device ** dev, struct filenames files);

int comm(model * m, device ** dev, int adj, int ui);

#ifndef __NOMPI__
int Out_MPI(model * m);
#endif
int writehdf5(struct filenames file, model * m);
hid_t create_file(const char *filename);

/* HDF5 file backed by RAM instead of disk (core driver, no backing store),
 * and a way to spill such a file out to disk if it must be persisted. */
hid_t create_file_core(const char *name);
int checkpoint_image_to_disk(hid_t file_id, const char *filename);

/* Write the boundary checkpoint for the single resident shot, for a caller
 * that ran the forward pass with SKIP_CHECKPOINT_FILE and now needs the file
 * after all. Returns nonzero if there isn't exactly one shot in flight. */
int checkpoint_flush(model * m, device ** dev, const char * path);
/* Uncompressed variant of writetomat(), for transient data (the boundary
 * checkpoint) where gzip costs far more than the space it saves. */
void writetomat_nocomp(hid_t* file_id,
                       const char *var,
                       float * varptr,
                       int NDIMs,
                       hsize_t dims[] );

void writetomat(hid_t* file_id,
                const char *var,
                float * varptr,
                int NDIMs,
                hsize_t dims[]);

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

int scale_sources(struct model * m);

