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

#define GMALLOC(x,y) if (!state) if (!((x)=malloc((y)))) {state=1;fprintf(stderr,"malloc failed at line %d in %s()\n",__LINE__,__func__);};
#define GFree(x) if ((x)) free( (x) );(x)=NULL;

#define __GUARD if (!state) state=
#define CLGUARD(x) if (!state) if (!(state = (x) )) {fprintf(stderr,"OpenCL function failed at line %d in %s()\n",__LINE__,__func__);};

#define CLPERR(err) fprintf(stderr,"Function %s at line %d: %s\n",__func__, __LINE__,gpu_error_code((err)))


struct filenames {
    char model[1024];
    char csts[1024];
    char dout[1024];
    char din[1024];
    char gout[1024];
    char rmsout[1024];
    char movout[1024];
};

// Structure containing all OpenCL variables
struct varcl {
    
    cl_command_queue cmd_queue;
    cl_command_queue cmd_queuecomm;
    
    size_t buffer_size_model;
    size_t buffer_size_modelc;
    size_t buffer_size_fd;
    size_t buffer_size_bnd;
    size_t buffer_size_surf_ref;

    
    size_t buffer_size_CPML_NX;
    size_t buffer_size_CPML_NY;
    size_t buffer_size_CPML_NZ;
    
    int numdim;

    cl_mem sxx;
    cl_mem syy;
    cl_mem szz;
    cl_mem sxy;
    cl_mem syz;
    cl_mem sxz;
    cl_mem vx;
    cl_mem vy;
    cl_mem vz;
    
    
    cl_mem rip;
    cl_mem rjp;
    cl_mem rkp;
    cl_mem u;
    cl_mem pi;
    cl_mem uipjp;
    cl_mem ujpkp;
    cl_mem uipkp;
    
    cl_mem taup;
    cl_mem taus;
    cl_mem tausipjp;
    cl_mem tausjpkp;
    cl_mem tausipkp;
    cl_mem eta;
    cl_mem rxx;
    cl_mem ryy;
    cl_mem rzz;
    cl_mem rxy;
    cl_mem ryz;
    cl_mem rxz;
    
    cl_mem taper;
    
    
    cl_mem K_x;
    cl_mem a_x;
    cl_mem b_x;
    cl_mem K_x_half;
    cl_mem a_x_half;
    cl_mem b_x_half;
    
    cl_mem K_y;
    cl_mem a_y;
    cl_mem b_y;
    cl_mem K_y_half;
    cl_mem a_y_half;
    cl_mem b_y_half;
    
    cl_mem K_z;
    cl_mem a_z;
    cl_mem b_z;
    cl_mem K_z_half;
    cl_mem a_z_half;
    cl_mem b_z_half;

    cl_mem psi_sxx_x;
    cl_mem psi_sxy_x;
    cl_mem psi_sxz_x;
    cl_mem psi_syy_y;
    cl_mem psi_sxy_y;
    cl_mem psi_syz_y;
    cl_mem psi_szz_z;
    cl_mem psi_sxz_z;
    cl_mem psi_syz_z;
    
    
    cl_mem psi_vxx;
    cl_mem psi_vyy;
    cl_mem psi_vzz;
    cl_mem psi_vxy;
    cl_mem psi_vxz;
    cl_mem psi_vyx;
    cl_mem psi_vyz;
    cl_mem psi_vzx;
    cl_mem psi_vzy;
    
    
    cl_mem src;
    cl_mem src_pos;
    cl_mem rec_pos;
    cl_mem vxout;
    cl_mem vyout;
    cl_mem vzout;
    cl_mem sxxout;
    cl_mem syyout;
    cl_mem szzout;
    cl_mem sxyout;
    cl_mem sxzout;
    cl_mem syzout;
    cl_mem pout;
    
    cl_mem sxx_r;
    cl_mem syy_r;
    cl_mem szz_r;
    cl_mem sxy_r;
    cl_mem syz_r;
    cl_mem sxz_r;
    cl_mem vx_r;
    cl_mem vy_r;
    cl_mem vz_r;
    
    cl_mem rxx_r;
    cl_mem ryy_r;
    cl_mem rzz_r;
    cl_mem rxy_r;
    cl_mem ryz_r;
    cl_mem rxz_r;

    cl_mem sxxbnd;
    cl_mem syybnd;
    cl_mem szzbnd;
    cl_mem sxybnd;
    cl_mem syzbnd;
    cl_mem sxzbnd;
    cl_mem vxbnd;
    cl_mem vybnd;
    cl_mem vzbnd;
    
    cl_mem sxxbnd_pin;
    cl_mem syybnd_pin;
    cl_mem szzbnd_pin;
    cl_mem sxybnd_pin;
    cl_mem syzbnd_pin;
    cl_mem sxzbnd_pin;
    cl_mem vxbnd_pin;
    cl_mem vybnd_pin;
    cl_mem vzbnd_pin;
    
    cl_mem rx;
    cl_mem ry;
    cl_mem rz;
    cl_mem rp;
    cl_mem gradrho;
    cl_mem gradM;
    cl_mem gradmu;
    cl_mem gradtaup;
    cl_mem gradtaus;
    cl_mem gradsrc;
    
    cl_mem gradfreqs;
    cl_mem gradfreqsn;
    
    cl_mem Hrho;
    cl_mem HM;
    cl_mem Hmu;
    cl_mem Htaup;
    cl_mem Htaus;
    
    cl_mem f_sxx;
    cl_mem f_syy;
    cl_mem f_szz;
    cl_mem f_sxy;
    cl_mem f_syz;
    cl_mem f_sxz;
    cl_mem f_vx;
    cl_mem f_vy;
    cl_mem f_vz;
    
    cl_mem f_rxx;
    cl_mem f_ryy;
    cl_mem f_rzz;
    cl_mem f_rxy;
    cl_mem f_ryz;
    cl_mem f_rxz;
    
    cl_mem sxx_sub1;
    cl_mem syy_sub1;
    cl_mem szz_sub1;
    cl_mem sxy_sub1;
    cl_mem syz_sub1;
    cl_mem sxz_sub1;
    cl_mem vx_sub1;
    cl_mem vy_sub1;
    cl_mem vz_sub1;
    
    cl_mem sxx_sub1_dev;
    cl_mem syy_sub1_dev;
    cl_mem szz_sub1_dev;
    cl_mem sxy_sub1_dev;
    cl_mem syz_sub1_dev;
    cl_mem sxz_sub1_dev;
    cl_mem vx_sub1_dev;
    cl_mem vy_sub1_dev;
    cl_mem vz_sub1_dev;
    
    cl_mem sxx_sub2;
    cl_mem syy_sub2;
    cl_mem szz_sub2;
    cl_mem sxy_sub2;
    cl_mem syz_sub2;
    cl_mem sxz_sub2;
    cl_mem vx_sub2;
    cl_mem vy_sub2;
    cl_mem vz_sub2;
    
    cl_mem sxx_sub2_dev;
    cl_mem syy_sub2_dev;
    cl_mem szz_sub2_dev;
    cl_mem sxy_sub2_dev;
    cl_mem syz_sub2_dev;
    cl_mem sxz_sub2_dev;
    cl_mem vx_sub2_dev;
    cl_mem vy_sub2_dev;
    cl_mem vz_sub2_dev;
    
    cl_mem sxx_r_sub1;
    cl_mem syy_r_sub1;
    cl_mem szz_r_sub1;
    cl_mem sxy_r_sub1;
    cl_mem syz_r_sub1;
    cl_mem sxz_r_sub1;
    cl_mem vx_r_sub1;
    cl_mem vy_r_sub1;
    cl_mem vz_r_sub1;
    
    cl_mem sxx_r_sub1_dev;
    cl_mem syy_r_sub1_dev;
    cl_mem szz_r_sub1_dev;
    cl_mem sxy_r_sub1_dev;
    cl_mem syz_r_sub1_dev;
    cl_mem sxz_r_sub1_dev;
    cl_mem vx_r_sub1_dev;
    cl_mem vy_r_sub1_dev;
    cl_mem vz_r_sub1_dev;
    
    cl_mem sxx_r_sub2;
    cl_mem syy_r_sub2;
    cl_mem szz_r_sub2;
    cl_mem sxy_r_sub2;
    cl_mem syz_r_sub2;
    cl_mem sxz_r_sub2;
    cl_mem vx_r_sub2;
    cl_mem vy_r_sub2;
    cl_mem vz_r_sub2;
    
    cl_mem sxx_r_sub2_dev;
    cl_mem syy_r_sub2_dev;
    cl_mem szz_r_sub2_dev;
    cl_mem sxy_r_sub2_dev;
    cl_mem syz_r_sub2_dev;
    cl_mem sxz_r_sub2_dev;
    cl_mem vx_r_sub2_dev;
    cl_mem vy_r_sub2_dev;
    cl_mem vz_r_sub2_dev;
    
    cl_kernel kernel_v;
    cl_kernel kernel_vcomm1;
    cl_kernel kernel_vcomm2;
    cl_kernel kernel_fill_transfer_buff1_v_out;
    cl_kernel kernel_fill_transfer_buff2_v_out;
    cl_kernel kernel_fill_transfer_buff1_v_in;
    cl_kernel kernel_fill_transfer_buff2_v_in;
    cl_kernel kernel_s;
    cl_kernel kernel_scomm1;
    cl_kernel kernel_scomm2;
    cl_kernel kernel_fill_transfer_buff1_s_out;
    cl_kernel kernel_fill_transfer_buff2_s_out;
    cl_kernel kernel_fill_transfer_buff1_s_in;
    cl_kernel kernel_fill_transfer_buff2_s_in;
    cl_kernel kernel_surf;
    cl_kernel kernel_initseis;
    cl_kernel kernel_initseis_r;
    cl_kernel kernel_initgrad;
    cl_kernel kernel_seisout;
    cl_kernel kernel_seisoutinit;
    cl_kernel kernel_residuals;
    cl_kernel kernel_adjv;
    cl_kernel kernel_adjvcomm1;
    cl_kernel kernel_adjvcomm2;
    cl_kernel kernel_adj_fill_transfer_buff1_v_out;
    cl_kernel kernel_adj_fill_transfer_buff2_v_out;
    cl_kernel kernel_adj_fill_transfer_buff1_v_in;
    cl_kernel kernel_adj_fill_transfer_buff2_v_in;
    cl_kernel kernel_adjs;
    cl_kernel kernel_adjscomm1;
    cl_kernel kernel_adjscomm2;
    cl_kernel kernel_adj_fill_transfer_buff1_s_out;
    cl_kernel kernel_adj_fill_transfer_buff2_s_out;
    cl_kernel kernel_adj_fill_transfer_buff1_s_in;
    cl_kernel kernel_adj_fill_transfer_buff2_s_in;
    cl_kernel kernel_bnd;
    cl_kernel kernel_savefreqs;
    cl_kernel kernel_initsavefreqs;
    cl_kernel kernel_initialize_gradsrc;
    cl_kernel kernel_surfgrid_coarse2fine;
    cl_kernel kernel_surfgrid_fine2coarse;
    cl_kernel kernel_sources;
    
    cl_program program_v;
    cl_program program_s;
    cl_program program_vcomm1;
    cl_program program_vcomm2;
    cl_program program_scomm1;
    cl_program program_scomm2;
    cl_program program_fill_transfer_buff_v;
    cl_program program_fill_transfer_buff_s;
    cl_program program_surf;
    cl_program program_initseis;
    cl_program program_initseis_r;
    cl_program program_initgrad;
    cl_program program_seisout;
    cl_program program_seisoutinit;
    cl_program program_residuals;
    cl_program program_adjv;
    cl_program program_adjs;
    cl_program program_adjvcomm1;
    cl_program program_adjvcomm2;
    cl_program program_adjscomm1;
    cl_program program_adjscomm2;
    cl_program program_bnd;
    cl_program program_savefreqs;
    cl_program program_initsavefreqs;
    cl_program program_initialize_gradsrc;
    cl_program program_surfgrid_coarse2fine;
    cl_program program_surfgrid_fine2coarse;
    cl_program program_sources;
    
    cl_event event_readMPI1[6];
    cl_event event_readMPI2[6];

    cl_event event_readv1;
    cl_event event_readv2;
    cl_event event_reads1;
    cl_event event_reads2;
    
    cl_event event_writev1;
    cl_event event_writev2;
    
    cl_event event_writes1;
    cl_event event_writes2;
    
    cl_event event_updatev_comm1;
    cl_event event_updatev_comm2;
    cl_event event_updates_comm1;
    cl_event event_updates_comm2;

    cl_event event_bndsave;
    cl_event event_bndtransf;
    cl_event event_bndsave2;
    cl_event event_bndtransf2;

    
};

// Structure containing all seismic parameters
struct modcsts {
    
    int NY;
    int NX;
    int NZ;
    int NXP;
    int NT;
    int FDORDER;
    int fdo;
    int fdoh;
    int nab;
    int MAXRELERROR;
    int gradout;
    int gradsrcout;
    int Hout;
    int seisout;
    int movout;
    int resout;
    int rmsout;
    int ns;
    int nsmax;
    int ngmax;
    int L;
    int MYID;
    int NP;
    int allng;
    int allns;
    int smin;
    int smax;
    int freesurf;
    int ND;
    int tmax;
    int tmin;
    int NTnyq;
    int dtnyq;
    
    int NGROUP;
    int MYGROUPID;
    int MYLOCALID;
    int MPI_NPROC_SHOT;
    int NLOCALP;
    int MPI_INIT;
    
    int back_prop_type;
    int param_type;
    int nfreqs;
    float * gradfreqs;
    float * gradfreqsn;
    float rms;
    float rmsnorm;
    float fmin, fmax;
    

    
    int restype;
    int (*res_calc)(struct modcsts * , int );
    
    
    cl_float2 * f_sxx;
    cl_float2 * f_syy;
    cl_float2 * f_szz;
    cl_float2 * f_sxy;
    cl_float2 * f_syz;
    cl_float2 * f_sxz;
    cl_float2 * f_vx;
    cl_float2 * f_vy;
    cl_float2 * f_vz;
    cl_float2 * f_rxx;
    cl_float2 * f_ryy;
    cl_float2 * f_rzz;
    cl_float2 * f_rxy;
    cl_float2 * f_ryz;
    cl_float2 * f_rxz;
    
    cl_float2 * f_sxxr;
    cl_float2 * f_syyr;
    cl_float2 * f_szzr;
    cl_float2 * f_sxyr;
    cl_float2 * f_syzr;
    cl_float2 * f_sxzr;
    cl_float2 * f_vxr;
    cl_float2 * f_vyr;
    cl_float2 * f_vzr;
    cl_float2 * f_rxxr;
    cl_float2 * f_ryyr;
    cl_float2 * f_rzzr;
    cl_float2 * f_rxyr;
    cl_float2 * f_ryzr;
    cl_float2 * f_rxzr;

    float vpmax;
    float vsmin;
    
    int abs_type;
    float VPPML;
    float FPML;
    float NPOWER;
    float K_MAX_CPML;
    
    
    float * K_x;
    float * a_x;
    float * b_x;
    float * K_x_half;
    float * a_x_half;
    float * b_x_half;
    float * K_y;
    float * a_y;
    float * b_y;
    float * K_y_half;
    float * a_y_half;
    float * b_y_half;
    float * K_z;
    float * a_z;
    float * b_z;
    float * K_z_half;
    float * a_z_half;
    float * b_z_half;
    
    int nmax_dev;
    int *no_use_GPUs;
    int n_no_use_GPUs;
    cl_device_type pref_device_type;
    cl_device_type device_type;
    cl_uint num_devices;
    cl_context context;
    size_t buffer_size_comm;
    
    float dt;
    float dh;
    float dhi;
    float abpc;
    
    double hc[7];
    
    float *rho;
    float *rip;
    float *rjp;
    float *rkp;
    float *u;
    float *pi;
    float *uipjp;
    float *ujpkp;
    float *uipkp;
    float *taper;
    double *gradrho, *gradM, *gradmu, *gradtaup, *gradtaus;
    double *Hrho;
    double *HM;
    double *Hmu;
    double *Htaup;
    double *Htaus;
    float **gradsrc;
    float **rx, **ry, **rz, **rp;
    float *taus;
    float *tausipjp;
    float *tausjpkp;
    float *tausipkp;
    float *taup;
    float *FL;
    float *eta;
    float TAU;
    float f0;
    float *topo;
    int topowidth;
    
    int *nsrc;
    int *nrec;
    float **src;
    float **src_pos;
    float **rec_pos;
    float **vxout;
    float **vyout;
    float **vzout;
    float **sxxout;
    float **syyout;
    float **szzout;
    float **sxyout;
    float **sxzout;
    float **syzout;
    float **pout;
    float **vx0;
    float **vy0;
    float **vz0;
    float **p0;
    int bcastvx;
    int bcastvy;
    int bcastvz;
    int bcastp;
    int bcastsxx;
    int bcastsyy;
    int bcastszz;
    int bcastsxy;
    int bcastsxz;
    int bcastsyz;
    
    float *movvx;
    float *movvy;
    float *movvz;
    
    
    
    float **weight;
    float **mute;
    int weightlength;
    int scalerms;
    int scaleshot;
    int scalermsnorm;

  
};

// Structure containing all seismic paramters local to the processing element
struct modcstsloc {
    
    int NY;
    int NX;
    int NZ;
    int NX0;
    int offset;
    int offsetfd;
    int dev;
    int num_devices;
    int Nbnd;
    int NZ_al16;
    int NZ_al0;
    
    size_t local_work_size[3];
    size_t global_work_size[3];
    size_t global_work_sizecomm2[3];
    size_t global_work_sizecomm1[3];
    size_t global_work_size_fillcomm[3];
    size_t global_work_size_surf[2];
    size_t global_work_size_initfd;
    size_t global_work_size_init;
    size_t global_work_size_f;
    size_t global_work_size_bnd;
    size_t global_work_size_gradsrc;
    size_t global_work_size_surfgrid[3];
    size_t global_work_size_sources;
    
    int local_off;
    cl_ulong required_global_mem_size;
    
    float *rho;
    float *rip;
    float *rjp;
    float *rkp;
    float *u;
    float *pi;
    float *uipjp;
    float *ujpkp;
    float *uipkp;
    
    float *movvx;
    float *movvy;
    float *movvz;
    float *buffermovvx;
    float *buffermovvy;
    float *buffermovvz;
    
    
    double *gradrho, *gradM, *gradmu, *gradtaup, *gradtaus;
    double *Hrho;
    double *HM;
    double *Hmu;
    double *Htaup;
    double *Htaus;

    float *taus;
    float *tausipjp;
    float *tausjpkp;
    float *tausipkp;
    float *taup;
    float **vxout;
    float **vyout;
    float **vzout;
    float **sxxout;
    float **syyout;
    float **szzout;
    float **sxyout;
    float **sxzout;
    float **syzout;
    float **pout;
    
    float * sxx_sub1;
    float * syy_sub1;
    float * szz_sub1;
    float * sxy_sub1;
    float * syz_sub1;
    float * sxz_sub1;
    float * vx_sub1;
    float * vy_sub1;
    float * vz_sub1;
    
    float * sxx_sub2;
    float * syy_sub2;
    float * szz_sub2;
    float * sxy_sub2;
    float * syz_sub2;
    float * sxz_sub2;
    float * vx_sub2;
    float * vy_sub2;
    float * vz_sub2;
    
    float * sxx_r_sub1;
    float * syy_r_sub1;
    float * szz_r_sub1;
    float * sxy_r_sub1;
    float * syz_r_sub1;
    float * sxz_r_sub1;
    float * vx_r_sub1;
    float * vy_r_sub1;
    float * vz_r_sub1;
    
    float * sxx_r_sub2;
    float * syy_r_sub2;
    float * szz_r_sub2;
    float * sxy_r_sub2;
    float * syz_r_sub2;
    float * sxz_r_sub2;
    float * vx_r_sub2;
    float * vy_r_sub2;
    float * vz_r_sub2;

    float * sxxbnd;
    float * syybnd;
    float * szzbnd;
    float * sxybnd;
    float * syzbnd;
    float * sxzbnd;
    float * vxbnd;
    float * vybnd;
    float * vzbnd;
    
    cl_float2 * f_sxx;
    cl_float2 * f_syy;
    cl_float2 * f_szz;
    cl_float2 * f_sxy;
    cl_float2 * f_syz;
    cl_float2 * f_sxz;
    cl_float2 * f_vx;
    cl_float2 * f_vy;
    cl_float2 * f_vz;
    cl_float2 * f_rxx;
    cl_float2 * f_ryy;
    cl_float2 * f_rzz;
    cl_float2 * f_rxy;
    cl_float2 * f_ryz;
    cl_float2 * f_rxz;
    
    cl_float2 * f_sxxr;
    cl_float2 * f_syyr;
    cl_float2 * f_szzr;
    cl_float2 * f_sxyr;
    cl_float2 * f_syzr;
    cl_float2 * f_sxzr;
    cl_float2 * f_vxr;
    cl_float2 * f_vyr;
    cl_float2 * f_vzr;
    cl_float2 * f_rxxr;
    cl_float2 * f_ryyr;
    cl_float2 * f_rzzr;
    cl_float2 * f_rxyr;
    cl_float2 * f_ryzr;
    cl_float2 * f_rxzr;
    


};



// SeisCL function definition

int toMPI(struct modcsts * mptr);

int Init_cst(struct modcsts * m);

int Init_model(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc);

int Init_OpenCL(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc);

int Free_OpenCL(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc) ;

int time_stepping(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc);

int comm_v(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep);

int comm_s(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep);

int readhdf5(struct filenames files, struct modcsts * m);

int Init_MPI(struct modcsts * m);

int writehdf5(struct filenames file, struct modcsts * m);

int Free_MPI(struct modcsts * m) ;

int Out_MPI(struct filenames file, struct modcsts * m);


cl_int GetPlatformID( cl_device_type * pref_device_type, cl_device_type * device_type, cl_platform_id* clsel_plat_id, cl_uint  *outnum_devices, int n_no_use_GPUs, int * no_use_GPUs);

cl_int connect_allgpus(struct varcl ** vcl, cl_context *incontext, cl_device_type * device_type, cl_platform_id* clsel_plat_id, int n_no_use_GPUs, int * no_use_GPUs, int nmax_dev);

cl_int get_device_num(cl_uint * num_devices);

cl_int create_gpu_kernel(const char * filename, cl_program *program, cl_context *context, cl_kernel *kernel, const char * program_name, const char * build_options);

cl_int create_gpu_kernel_from_string(const char *program_source, cl_program *program, cl_context *context, cl_kernel *kernel, const char * program_name, const char * build_options);

cl_int transfer_gpu_memory( cl_command_queue *inqueue, size_t buffer_size, cl_mem *var_mem, float *var);

cl_int read_gpu_memory( cl_command_queue *inqueue, size_t buffer_size, cl_mem *var_mem, void *var);

cl_int create_gpu_memory_buffer(cl_context *incontext, size_t buffer_size, cl_mem *var_mem);

cl_int create_gpu_memory_buffer_cst(cl_context *incontext, size_t buffer_size, cl_mem *var_mem);

cl_int create_pinned_memory_buffer(cl_context *incontext, cl_command_queue *inqueue, size_t buffer_size, cl_mem *var_mem, float **var_buf);

cl_int create_gpu_subbuffer(cl_mem *var_mem, cl_mem *sub_mem, cl_buffer_region * region);

cl_int launch_gpu_kernel( cl_command_queue *inqueue, cl_kernel *kernel, int ndim, size_t global_work_size[2], size_t local_work_size[2], int numevent, cl_event * waitlist, cl_event * eventout);

double machcore(uint64_t endTime, uint64_t startTime);

char *gpu_error_code(cl_int err);


int gpu_intialize_seis(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc );
int gpu_intialize_seis_r(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm , struct modcstsloc *inmloc );
int gpu_intialize_grad(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm , struct modcstsloc *inmloc );

int gpu_initialize_update_v(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc, int bndoff, int lcomm, int comm);
int gpu_initialize_update_s(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc, int bndoff, int lcomm , int comm);
int gpu_initialize_surface(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc );

int gpu_intialize_seisout(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc );
int gpu_intialize_seisoutinit(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc );
int gpu_intialize_residuals(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc );

int gpu_initialize_update_adjv(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm , struct modcstsloc *inmloc, int bndoff, int lcomm, int comm );
int gpu_initialize_update_adjs(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm , struct modcstsloc *inmloc, int bndoff , int lcomm, int comm);
int gpu_initialize_savebnd(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc  );
int holbergcoeff(struct modcsts *inm);
int gpu_initialize_savefreqs(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc, int dirprop );
int gpu_initialize_initsavefreqs(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc );
int gpu_initialize_gradsrc(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc );

int gpu_intialize_surfgrid_coarse2fine(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, struct varcl *inmem_c, struct modcsts *inm_c, struct modcstsloc *inmloc_c ,struct varcl *inmem_f, struct modcsts *inm_f, struct modcstsloc *inmloc_f,size_t *local_work_size );
int gpu_intialize_surfgrid_fine2coarse(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, struct varcl *inmem_c, struct modcsts *inm_c, struct modcstsloc *inmloc_c ,struct varcl *inmem_f, struct modcsts *inm_f, struct modcstsloc *inmloc_f );

int gpu_intialize_fill_transfer_buff_s(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc, int out, int comm, int adj );
int gpu_intialize_fill_transfer_buff_v(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc, int out, int comm, int adj );

int gpu_intialize_sources(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc );

void CPML_coeff(struct modcsts * m);

int calc_grad(struct modcsts* m, struct modcstsloc * mloc);

int calc_Hessian(struct modcsts* mglob, struct modcstsloc * m);

int butterworth(float * data, float fcl, float fch, float dt, int NT, int tmax, int ntrace, int order);

int res_raw(struct modcsts * mptr, int s);
int res_amp(struct modcsts * mptr, int s);

int Init_surfgrid(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, struct modcsts * m_s, struct varcl ** vcl_s, struct modcstsloc ** mloc_s);

int alloc_seismo(float *** var, int ns, int allng, int NT, int * nrec );


