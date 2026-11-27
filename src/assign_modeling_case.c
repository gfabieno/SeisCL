//
//  assign_modeling_case.c
//  SeisCL
//
//  Created by Gabriel Fabien-Ouellet on 17-07-06.
//  Copyright © 2017 Gabriel Fabien-Ouellet. All rights reserved.
//

#include "F.h"
#include "third_party/SOFI3D/CPML_COEFF.h"

/*Loading files autmatically created by the makefile that contain the *.cl kernels in a c string.
 This way, no .cl file need to be read and there is no need to be in the executable directory to execute SeisCL.*/
#include "grad_dft2D.hcl"
#include "grad_dft2D_SH.hcl"
#include "grad_dft2D_visc.hcl"
#include "grad_dft2D_SH_visc.hcl"
#include "grad_dft3D.hcl"
#include "grad_dft3D_visc.hcl"
#include "average_params.hcl"
#include "prep_params.hcl"
#include "dev_io.hcl"
#include "grad_post.hcl"
#include "savebnd2D.hcl"
#include "savebnd3D.hcl"
#include "savebnd2D_SH.hcl"
#include "surface2D.hcl"
#include "surface2D_adj.hcl"
#include "cpml_adj2D.hcl"
#include "cpml_adj3D.hcl"
#include "cpml_adj2D_SH.hcl"
#include "rec_grad.hcl"
#include "rec_grad_SH.hcl"
#include "header_vec.hcl"
#include "surface2D_SH.hcl"
#include "surface3D.hcl"
#include "surface3D_adj.hcl"
#include "update_adjs2D.hcl"
#include "update_adjs2D_SH.hcl"
#include "update_adjs3D.hcl"
#include "update_adjv2D.hcl"
#include "update_adjv2D_SH.hcl"
#include "update_adjv3D.hcl"
#include "update_s2D.hcl"
#include "update_s2D_SH.hcl"
#include "update_s2D_acc.hcl"
#include "update_s3D.hcl"
#include "update_v2D.hcl"
#include "update_v2D_SH.hcl"
#include "update_v2D_acc.hcl"
#include "update_v3D.hcl"
#include "header_FD.hcl"
#include "header_FD_fp16.hcl"
#include "header_CUDACL.hcl"
#include "header_injectbnd.hcl"

void ave_arithmetic_rho(float * pin, float * pout, int * N, int ndim, int  *dir){
    
    int i,j,k;
    int NX, NY, NZ;
    int NX0=0, NY0=0, NZ0=0;
    int ind1, ind2;
    if (ndim==3){
        NX=N[2];
        NY=N[1];
        NZ=N[0];
    }
    else
    {
        NX=N[1];
        NY=1;
        NZ=N[0];
    }

    // implementation of improved vacuum formulation eq. 6-7
    // converted rho to buoyancy (pin=b here), so arithmetic averaging of rho
    // is now: rhoij = 0.5*(rho1+rho2) -> bij = 2 / (1/pin1 + 1/pin2)
    for (k=0;k<NZ-dir[0];k++){
        for (j=0;j<NY-dir[1];j++){
            for (i=0;i<NX-dir[2];i++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                ind2 = (i+dir[2])*NY*NZ+(j+dir[1])*NZ+(k+dir[0]);
                // eq. 6, piecewise: both nodes vacuum (buoyancy 0) -> 0;
                // exactly one vacuum -> twice the solid neighbor's
                // buoyancy (2/(0+1/pin)=2*pin); both solid -> the usual
                // harmonic-of-density combination. Written this way
                // (rather than relying on 1/0 -> +inf -> 1/inf -> 0
                // round-tripping back to the right answer) so the vacuum
                // region (FREESURF==2) is handled explicitly and doesn't
                // depend on IEEE inf/nan semantics surviving compiler
                // flags such as -ffast-math.
                if (pin[ind1]==0 && pin[ind2]==0){
                    pout[ind1] = 0;
                }
                else if (pin[ind1]==0){
                    pout[ind1] = 2*pin[ind2];
                }
                else if (pin[ind2]==0){
                    pout[ind1] = 2*pin[ind1];
                }
                else{
                    pout[ind1]=2/(1.0/pin[ind1] + 1.0/pin[ind2]);
                }
            }
        }
    }
    
    if (dir[2]==1){
        NX0=NX-1;
    }
    else if (dir[1]==1){
        NY0=NY-1;
    }
    if (dir[0]==1){
        NZ0=NZ-1;
    }
    
    for (k=NZ0;k<NZ;k++){
        for (j=NY0;j<NY;j++){
            for (i=NX0;i<NX;i++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                pout[ind1]= pin[ind1];
            }
        }
    }
    
    
}
void ave_arithmetic_tau(float *pin, float *pout, int *N, int ndim, int dir[][3]) {
    
    int i,j,k;
    int NX, NY, NZ;
    int NX0=0, NY0=0, NZ0=0;
    int ind1, ind2, ind3, ind4;
    if (ndim==3){
        NX=N[2];
        NY=N[1];
        NZ=N[0];
    }
    else
    {
        NX=N[1];
        NY=1;
        NZ=N[0];
    }
    
    
    for (k=0;k<NZ-dir[0][0]-dir[1][0];k++){
        for (j=0;j<NY-dir[0][1]-dir[1][1];j++){
            for (i=0;i<NX-dir[0][2]-dir[1][2];i++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                ind2 = (i+dir[0][2])*NY*NZ+(j+dir[0][1])*NZ+(k+dir[0][0]);
                ind3 = (i+dir[1][2])*NY*NZ+(j+dir[1][1])*NZ+(k+dir[1][0]);
                ind4 = (i+dir[0][2]+dir[1][2])*NY*NZ
                +(j+dir[0][1]+dir[1][1])*NZ
                +(k+dir[0][0]+dir[1][0]);
                
                pout[ind1]=0.25*( pin[ind1]+pin[ind2]+pin[ind3]+pin[ind4]);
            }
        }
    }
    
    if (dir[0][2]==1){
        NX0=NX-1;
    }
    else if (dir[0][1]==1){
        NY0=NY-1;
    }
    if (dir[0][0]==1){
        NZ0=NZ-1;
    }
    
    for (k=NZ0;k<NZ;k++){
        for (j=NY0;j<NY;j++){
            for (i=NX0;i<NX;i++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                pout[ind1]= pin[ind1];
            }
        }
    }
    
    /* The last plane of the second direction keeps the cell value too
       (NX0/NY0/NZ0 reset: without it only the corner was copied and the rest
       of the plane stayed 0). A zero direction (SH) has no plane. */
    NX0=0; NY0=0; NZ0=0;
    if (dir[1][2]==1){
        NX0=NX-1;
    }
    else if (dir[1][1]==1){
        NY0=NY-1;
    }
    if (dir[1][0]==1){
        NZ0=NZ-1;
    }
    
    if (dir[1][0] || dir[1][1] || dir[1][2]){
    for (k=NZ0;k<NZ;k++){
        for (j=NY0;j<NY;j++){
            for (i=NX0;i<NX;i++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                pout[ind1]= pin[ind1];
            }
        }
    }
    }
    
    
}

void ave_harmonic_mu(float * pin, float * pout, int * N, int ndim, int dir[][3]) {
    
    int i,j,k;
    int NX, NY, NZ;
    int NX0=0, NY0=0, NZ0=0;
    int ind1, ind2, ind3, ind4;
    if (ndim==3){
        NX=N[2];
        NY=N[1];
        NZ=N[0];
    }
    else
    {
        NX=N[1];
        NY=1;
        NZ=N[0];
    }
    
    
    for (k=0;k<NZ-dir[0][0]-dir[1][0];k++){
        for (j=0;j<NY-dir[0][1]-dir[1][1];j++){
            for (i=0;i<NX-dir[0][2]-dir[1][2];i++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                ind2 = (i+dir[0][2])*NY*NZ+(j+dir[0][1])*NZ+(k+dir[0][0]);
                ind3 = (i+dir[1][2])*NY*NZ+(j+dir[1][1])*NZ+(k+dir[1][0]);
                ind4 = (i+dir[0][2]+dir[1][2])*NY*NZ
                      +(j+dir[0][1]+dir[1][1])*NZ
                      +(k+dir[0][0]+dir[1][0]);
                
                pout[ind1]=4.0/( 1.0/pin[ind1]
                                +1.0/pin[ind2]
                                +1.0/pin[ind3]
                                +1.0/pin[ind4]);
                // implementation of improved vacuum formulation eq. 8
                if (pin[ind1]==0 || pin[ind2]==0 || pin[ind3]==0 || pin[ind4]==0)
                {
                    pout[ind1] = 0;
                }
            }
        }
    }
    
    if (dir[0][2]==1){
        NX0=NX-1;
    }
    else if (dir[0][1]==1){
        NY0=NY-1;
    }
    if (dir[0][0]==1){
        NZ0=NZ-1;
    }
    
    for (k=NZ0;k<NZ;k++){
        for (j=NY0;j<NY;j++){
            for (i=NX0;i<NX;i++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                pout[ind1]= pin[ind1];
            }
        }
    }
    
    /* The last plane of the second direction keeps the cell value too
       (NX0/NY0/NZ0 reset: without it only the corner was copied and the rest
       of the plane stayed 0). A zero direction (SH) has no plane. */
    NX0=0; NY0=0; NZ0=0;
    if (dir[1][2]==1){
        NX0=NX-1;
    }
    else if (dir[1][1]==1){
        NY0=NY-1;
    }
    if (dir[1][0]==1){
        NZ0=NZ-1;
    }
    
    if (dir[1][0] || dir[1][1] || dir[1][2]){
    for (k=NZ0;k<NZ;k++){
        for (j=NY0;j<NY;j++){
            for (i=NX0;i<NX;i++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                pout[ind1]= pin[ind1];
            }
        }
    }
    }
    
    
}

/* NOTE: SeisCL does not create a free surface itself.

   There used to be a set_freesurf2_vacuum() here that zeroed M/mu/rho in the
   top FDOH-thick band whenever FREESURF==2, i.e. the engine manufactured the
   vacuum. That is not what the improved vacuum formulation says
   (Zeng et al. 2012, doi:10.1190/geo2011-0067.1 --
   papers/pdfs/zeng_2012_geophysics-improved-vacuum-freesurface.pdf): the
   whole point of the method is that a free surface needs NO special-cased
   boundary code at all. The user supplies a model with vp=vs=rho=0 wherever
   the vacuum is, those nodes are then updated exactly like interior nodes,
   and the traction-free condition falls out of the parameter averaging
   (ave_arithmetic_rho / ave_harmonic_mu above, the paper's eq. 6-8). Baking a
   fixed flat band into the engine also threw away the method's main
   advantage, which is that arbitrary topography and internal discontinuities
   need no extra handling -- they are just more zeros in the input model.

   FREESURF==2 therefore has exactly one job left, and it needs no code here:
   disabling the absorbing boundary on the TOP edge so the vacuum is not
   damped. That is already done at compile time by the `#if FREESURF==0`
   guards around both the CPML (ABS_TYPE==1) and Cerjan (ABS_TYPE==2)
   top-edge blocks in update_v*.cl / update_s*.cl.

   What DOES belong in the engine, and stays: the handling of zero-material
   cells wherever they occur -- the averaging above, and the vacuum guards in
   calc_grad.c. Those work on the user's zeros, whatever shape they take. */

/*Functions to define the transformations requires for each parameter
  and constant */
void mu(void * mptr){
    
    int i,l, state=0;
    model * m = (model *) mptr;
    float *mu = get_par(m->pars, m->npars, "mu")->gl_par;
    int num_ele = get_par(m->pars, m->npars, "mu")->num_ele;
    float *rho = get_par(m->pars, m->npars, "rho")->gl_par;
    float *taus = get_par(m->pars, m->npars, "taus")->gl_par;
    float thisvs, thistaus;
    if (m->par_type==0){  //vp, vs, rho
        for (i=0;i<num_ele;i++){
            mu[i]=powf(mu[i],2)*rho[i]*m->dt/m->dh;
        }
    }
    else if (m->par_type==1){//M, mu, rho
        for (i=0;i<num_ele;i++){
            mu[i]=mu[i]*m->dt/m->dh;
        }
    }
    else if (m->par_type==2){
        for (i=0;i<num_ele;i++){//Ip, Is, rho
            mu[i]=powf(mu[i]/rho[i],2)*rho[i]*m->dt/m->dh;
        }
    }
    else if (m->par_type==3){
        for (i=0;i<num_ele;i++){
            
            if (mu[i]>0){
                thisvs=mu[i]-taus[i];
                thistaus=taus[i]/(mu[i]-taus[i]);
            }
            else{
                thistaus=0;
                thisvs=0;
            }
            mu[i]=powf(thisvs,2)*rho[i]*m->dt/m->dh;
            taus[i]=thistaus;
        }
        
    }
    //Correct the phase velocity for viscoelastic modeling
    if (m->L>0){
        float *pts=NULL;
        /* vector for maxwellbodies */
        GMALLOC(pts,m->L*sizeof(float));
        float * eta = get_cst( m->csts,m->ncsts, "eta")->gl_cst;
        for (l=0;l<m->L;l++) {
            pts[l]=m->dt/eta[l];
        }
        
        float ws=2.0*PI*m->f0;
        float sumu=0.0;
        for (l=0;l<m->L;l++){
            sumu+= ((ws*ws*pts[l]*pts[l])
                    /(1.0+ws*ws*pts[l]*pts[l]));
        }
        for (i=0;i<num_ele;i++){
            mu[i]=mu[i]/(1.0+sumu*taus[i]);
        }
        free(pts);
    }
    
    int scaler = 0;
    if (m->FP16>0){
        if (m->par_scale == 0){
            m->set_par_scale( (void*) m);
        }
        scaler = m->par_scale;
        /* 2^scaler once, not one powf per cell: gcc does not hoist powf out
           of the loop (errno), and SeisCL.torch re-runs this at every call
           (0.3 s per call at 1000^2, 2026-11-25). Exact power of two: the
           same values as powf(2, scaler). */
        const float sc = ldexpf(1.0f, scaler);
        for (i=0;i<num_ele;i++){
            mu[i]*=sc;
        }
    }
    
}
void M(void * mptr){
    
    int i,l, state=0;
    model * m = (model *) mptr;
    float *M = get_par(m->pars, m->npars, "M")->gl_par;
    int num_ele = get_par(m->pars, m->npars, "M")->num_ele;
    float *rho = get_par(m->pars, m->npars, "rho")->gl_par;
    float *taup = get_par(m->pars, m->npars, "taup")->gl_par;
    float thisvp, thistaup;
    if (m->par_type==0){
        for (i=0;i<num_ele;i++){
            M[i]=powf(M[i],2)*rho[i]*m->dt/m->dh;
        }
    }
    if (m->par_type==1){
        for (i=0;i<num_ele;i++){
            M[i]=M[i]*m->dt/m->dh;
        }
    }
    else if (m->par_type==2){
        for (i=0;i<num_ele;i++){
            M[i]=powf(M[i]/rho[i],2)*rho[i]*m->dt/m->dh;
        }
    }
    else if (m->par_type==3){
        for (i=0;i<num_ele;i++){
            
            thisvp=M[i]-taup[i];
            thistaup=taup[i]/(M[i]-taup[i]);
            M[i]=powf(thisvp,2)*rho[i]*m->dt/m->dh;
            taup[i]=thistaup;
        }
        
    }
    //Correct the phase velocity for viscoelastic modeling
    if (m->L>0){
        float *pts=NULL;
        /* vector for maxwellbodies */
        GMALLOC(pts,m->L*sizeof(float));
        float * eta = get_cst(m->csts,m->ncsts, "eta")->gl_cst;
        for (l=0;l<m->L;l++) {
            pts[l]=m->dt/eta[l];
        }
        
        float ws=2.0*PI*m->f0;
        float sumpi=0.0;
        
        for (i=0;i<num_ele;i++){
            sumpi=0.0;
            for (l=0;l<m->L;l++){
                sumpi+= ((ws*ws*pts[l]*pts[l]*taup[i])
                         /(1.0+ws*ws*pts[l]*pts[l]));
            }
            M[i]=M[i]/(1.0+sumpi);
        }
        
        free(pts);
    }
    
    int scaler=0;
    if (m->FP16>0){
        m->set_par_scale( (void*) m);
        scaler = m->par_scale;
        const float sc = ldexpf(1.0f, scaler);   /* once, see mu() */
        for (i=0;i<num_ele;i++){
            M[i]*=sc;
        }
    }
    
}

void rho(void * mptr){
    
    model * m = (model *) mptr;
    float *rho = get_par(m->pars, m->npars, "rho")->gl_par;
    int scaler=m->par_scale;

    int i;
    int num_ele = get_par(m->pars, m->npars, "rho")->num_ele;
    const float sc = ldexpf(1.0f, -scaler);   /* = powf(2,-scaler), once, see mu() */
    for (i=0;i<num_ele;i++){
        /*A zero density marks a vacuum cell (improved vacuum free-surface
          formulation, FREESURF==2): keep its buoyancy exactly zero instead
          of 1/0, so the arithmetic averaging in ave_arithmetic_rho below
          can rely on seeing true zeros rather than +inf.*/
        if (rho[i]==0){
            rho[i]=0;
        }
        else{
            rho[i]=1.0/rho[i]*m->dt/m->dh*sc;
        }
    }
}

void rip(void * mptr){
    
    model * m = (model *) mptr;
    float *rho = get_par(m->pars, m->npars, "rho")->gl_par;
    float *rip = get_par(m->pars, m->npars, "rip")->gl_par;
    int dir[3]={0,0,1};
    ave_arithmetic_rho(rho, rip, m->N, m->NDIM, dir);
}
void rjp(void * mptr){
    
    model * m = (model *) mptr;
    float *rho = get_par(m->pars, m->npars, "rho")->gl_par;
    float *rjp = get_par(m->pars, m->npars, "rjp")->gl_par;
    int dir[3]={0,1,0};
    ave_arithmetic_rho(rho, rjp, m->N, m->NDIM, dir);
}
void rkp(void * mptr){
    
    model * m = (model *) mptr;
    float *rho = get_par(m->pars, m->npars, "rho")->gl_par;
    float *rkp = get_par(m->pars, m->npars, "rkp")->gl_par;
    int dir[3]={1,0,0};
    ave_arithmetic_rho(rho, rkp, m->N, m->NDIM, dir);
}
void muipkp(void * mptr){
    
    model * m = (model *) mptr;
    float *mu = get_par(m->pars, m->npars, "mu")->gl_par;
    float *muipkp = get_par(m->pars, m->npars, "muipkp")->gl_par;
    int dir[2][3]={{0,0,1},{1,0,0}};
    ave_harmonic_mu(mu, muipkp, m->N, m->NDIM, dir);
}
void mujpkp(void * mptr){
    
    model * m = (model *) mptr;
    float *mu = get_par(m->pars, m->npars, "mu")->gl_par;
    float *mujpkp = get_par(m->pars, m->npars, "mujpkp")->gl_par;
    int dir[2][3]={{0,1,0},{1,0,0}};
    /* SH (ND 21): the 3-D staggering with nothing varying along y, i.e. the
       average along z only (second direction 0: two-point mean). */
    int dsh[2][3]={{1,0,0},{0,0,0}};
    ave_harmonic_mu(mu, mujpkp, m->N, m->NDIM, m->ND==21 ? dsh : dir);
}
void muipjp(void * mptr){
    
    model * m = (model *) mptr;
    float *mu = get_par(m->pars, m->npars, "mu")->gl_par;
    float *muipjp = get_par(m->pars, m->npars, "muipjp")->gl_par;
    int dir[2][3]={{0,0,1},{0,1,0}};
    /* SH (ND 21): the 3-D staggering with nothing varying along y, i.e. the
       average along x only (second direction 0: two-point mean). */
    int dsh[2][3]={{0,0,1},{0,0,0}};
    ave_harmonic_mu(mu, muipjp, m->N, m->NDIM, m->ND==21 ? dsh : dir);
}
void tausipkp(void * mptr){
    
    model * m = (model *) mptr;
    float *taus = get_par(m->pars, m->npars, "taus")->gl_par;
    float *tausipkp = get_par(m->pars, m->npars, "tausipkp")->gl_par;
    int dir[2][3]={{0,0,1},{1,0,0}};
    ave_arithmetic_tau(taus, tausipkp, m->N, m->NDIM, dir);
}
void tausjpkp(void * mptr){
    
    model * m = (model *) mptr;
    float *taus = get_par(m->pars, m->npars, "taus")->gl_par;
    float *tausjpkp = get_par(m->pars, m->npars, "tausjpkp")->gl_par;
    int dir[2][3]={{0,1,0},{1,0,0}};
    /* SH (ND 21): the 3-D staggering with nothing varying along y, i.e. the
       average along z only (second direction 0: two-point mean). */
    int dsh[2][3]={{1,0,0},{0,0,0}};
    ave_arithmetic_tau(taus, tausjpkp, m->N, m->NDIM, m->ND==21 ? dsh : dir);
}
void tausipjp(void * mptr){
    
    model * m = (model *) mptr;
    float *taus = get_par(m->pars, m->npars, "taus")->gl_par;
    float *tausipjp = get_par(m->pars, m->npars, "tausipjp")->gl_par;
    int dir[2][3]={{0,0,1},{0,1,0}};
    /* SH (ND 21): the 3-D staggering with nothing varying along y, i.e. the
       average along x only (second direction 0: two-point mean). */
    int dsh[2][3]={{0,0,1},{0,0,0}};
    ave_arithmetic_tau(taus, tausipjp, m->N, m->NDIM, m->ND==21 ? dsh : dir);
}
void eta( void *mptr, void *cstptr, int ncst){
    //Viscoelastic constants initialization
    int l;
    model * m = (model *) mptr;
    float * eta=get_cst(m->csts,m->ncsts, "eta")->gl_cst;
    float * FL =get_cst(m->csts,m->ncsts, "FL" )->gl_cst;
    if (m->L>0){
        for (l=0;l<m->L;l++) {
            eta[l]=(2.0*PI*FL[l])*m->dt;
        }
    }
    
}
void gradfreqsn( void *mptr, void *cstptr, int ncst){
    //Viscoelastic constants initialization
    int j;
    model * m = (model *) mptr;
    float * gradfreqs=get_cst(m->csts,m->ncsts, "gradfreqs")->gl_cst;
    float * gradfreqsn =get_cst(m->csts,m->ncsts, "gradfreqsn" )->gl_cst;
    
    
    float fmaxout=0;
    for (j=0;j<m->NFREQS;j++){
        if (gradfreqs[j]>fmaxout)
        fmaxout=gradfreqs[j];
    }
    float df;
    if (!(fmaxout>0)){
        /* Every gradfreqs entry is zero or negative. Without this guard the
         * DTNYQ expression below divides by zero and every downstream size is
         * inf/nan. assign_modeling_case() cannot catch this because the cst
         * values are not loaded yet when it runs. */
        fprintf(stderr,"Error: back_prop_type=2 requires at least one strictly "
                       "positive entry in gradfreqs (max was %g)\n", fmaxout);
        m->DTNYQ=1;
        m->NTNYQ=1;
        for (j=0;j<m->NFREQS;j++) gradfreqsn[j]=0;
        return;
    }
    /* Oversampling of the highest requested gradient frequency. The DFT only
     * needs the Nyquist rate of fmax, but savefreqs was sampling at
     * 1/0.0156 = 64x that, which for typical 2D parameters (fmax=25 Hz,
     * dt=1e-3) gives DTNYQ=1 -- i.e. it fires on *every* time step, and it is
     * ~75% of GPU time. dft_osamp exposes the factor; the default of 64
     * preserves the previous behaviour exactly.
     *
     * Lowering it is an aliasing trade-off, not a quadrature one: with
     * DTNYQ>1 the accumulation is the *exact* DFT of the decimated series, and
     * the error is energy above the decimated Nyquist folding onto the selected
     * bins. For a Ricker at f0 the spectrum is ~e^-64 at 8*f0, so with
     * fmax ~ 2*f0 an oversample of 8 puts the decimated Nyquist at 8*f0. */
    /* dft_step > 0 sets the step directly (read_hdf5.c and the torch
     * binding refuse a step whose Nyquist frequency is not above fmax). */
    if (m->dft_step>0)
        m->DTNYQ=m->dft_step;
    else
        m->DTNYQ=ceil((1.0/m->dft_osamp)/fmaxout/m->dt);
    if (m->DTNYQ<1) m->DTNYQ=1;
    /* NTNYQ is the DFT period: it must equal the number of samples savefreqs
     * actually accumulates, otherwise the basis is not orthogonal and every bin
     * picks up a phase drift. The forward loop runs t = 0 .. tmax-1 and fires
     * when t >= tmin && (t-tmin) % DTNYQ == 0 (time_stepping.c:838, :850-853),
     * i.e. ceil((tmax-tmin)/DTNYQ) samples. The previous
     * (tmax-tmin)/DTNYQ + 1 overcounted by one whenever DTNYQ divides
     * tmax-tmin, giving e.g. 257 for 256 accumulated samples. */
    m->NTNYQ=(m->tmax-m->tmin+m->DTNYQ-1)/m->DTNYQ;
    df=1.0/m->NTNYQ/m->dt/m->DTNYQ;
    /* A requested frequency is used at the bin below it (floor). The DFT
       gradient is the gradient of the misfit restricted to these bins: it
       equals back_prop_type=1 only when every bin to the Nyquist frequency
       of the record is used. With fewer bins vs and rho can be wrong by
       factors, sign included, with force sources or velocity receivers
       (measurements in SeisCL.py, gradfreqs; pitfall A13 of the group skill). */
    for (j=0;j<m->NFREQS;j++){
        gradfreqsn[j]=floor(gradfreqs[j]/df);
    }
    
}
void taper( void *mptr, void *cstptr, int ncst){
    //Viscoelastic constants initialization
    int i;
    model * m = (model *) mptr;
    float * taper=get_cst(m->csts,m->ncsts, "taper")->gl_cst;
    
    
    float amp=1-m->abpc/100;
    float a=sqrt(-log(amp)/((m->NAB-1)*(m->NAB-1)));
    for (i=1; i<=m->NAB; i++) {
        taper[i-1]=exp(-(a*a*(m->NAB-i)*(m->NAB-i)));
    }
    
}

void CPML_z( void *mptr, void *cstptr, int ncst){
    //Viscoelastic constants initialization
    model *m  = (model*)mptr;
    float * K_i=get_cst(m->csts,m->ncsts, "K_z")->gl_cst;
    float * a_i=get_cst(m->csts,m->ncsts, "a_z")->gl_cst;
    float * b_i=get_cst(m->csts,m->ncsts, "b_z")->gl_cst;
    float * K_i_half=get_cst(m->csts,m->ncsts, "K_z_half")->gl_cst;
    float * a_i_half=get_cst(m->csts,m->ncsts, "a_z_half")->gl_cst;
    float * b_i_half=get_cst(m->csts,m->ncsts, "b_z_half")->gl_cst;

    CPML_coeff(m->NPOWER,
               m->K_MAX_CPML,
               m->FPML,
               m->VPPML,
               m->dh,
               m->dt,
               m->NAB,
               K_i,
               b_i,
               a_i,
               K_i_half,
               b_i_half,
               a_i_half);
    
}
void CPML_x( void *mptr, void *cstptr, int ncst){
    //Viscoelastic constants initialization
    model * m = (model *) mptr;
    float * K_i=get_cst(m->csts,m->ncsts, "K_x")->gl_cst;
    float * a_i=get_cst(m->csts,m->ncsts, "a_x")->gl_cst;
    float * b_i=get_cst(m->csts,m->ncsts, "b_x")->gl_cst;
    float * K_i_half=get_cst(m->csts,m->ncsts, "K_x_half")->gl_cst;
    float * a_i_half=get_cst(m->csts,m->ncsts, "a_x_half")->gl_cst;
    float * b_i_half=get_cst(m->csts,m->ncsts, "b_x_half")->gl_cst;
    
    CPML_coeff(m->NPOWER,
               m->K_MAX_CPML,
               m->FPML,
               m->VPPML,
               m->dh,
               m->dt,
               m->NAB,
               K_i,
               b_i,
               a_i,
               K_i_half,
               b_i_half,
               a_i_half);
    
}
void CPML_y( void *mptr, void *cstptr, int ncst){
    //Viscoelastic constants initialization
    model * m = (model *) mptr;
    float * K_i=get_cst(m->csts,m->ncsts, "K_y")->gl_cst;
    float * a_i=get_cst(m->csts,m->ncsts, "a_y")->gl_cst;
    float * b_i=get_cst(m->csts,m->ncsts, "b_y")->gl_cst;
    float * K_i_half=get_cst(m->csts,m->ncsts, "K_y_half")->gl_cst;
    float * a_i_half=get_cst(m->csts,m->ncsts, "a_y_half")->gl_cst;
    float * b_i_half=get_cst(m->csts,m->ncsts, "b_y_half")->gl_cst;
    
    CPML_coeff(m->NPOWER,
               m->K_MAX_CPML,
               m->FPML,
               m->VPPML,
               m->dh,
               m->dt,
               m->NAB,
               K_i,
               b_i,
               a_i,
               K_i_half,
               b_i_half,
               a_i_half);
    
}
void CPML_ka( void *mptr, void *cstptr, int ncst){
    /* Experiment (CPMLINL): Ka = 1/K + a, so the inline adjoint filter
       e/K + (b*psi + a*e) is one fma: e*Ka + b*psi (header_vec.cl, cpi_val). */
    model * m = (model *) mptr;
    const char * ax[3] = {"x", "y", "z"};
    char nk[16], na[16], nka[16];
    for (int d = 0; d < 3; d++){
        for (int h = 0; h < 2; h++){
            sprintf(nk, "K_%s%s", ax[d], h ? "_half" : "");
            sprintf(na, "a_%s%s", ax[d], h ? "_half" : "");
            sprintf(nka, "Ka_%s%s", ax[d], h ? "_half" : "");
            if (!get_cst(m->csts, m->ncsts, nka)->gl_cst) continue;
            float * K = get_cst(m->csts, m->ncsts, nk)->gl_cst;
            float * a = get_cst(m->csts, m->ncsts, na)->gl_cst;
            float * ka = get_cst(m->csts, m->ncsts, nka)->gl_cst;
            for (int i = 0; i < 2*m->NAB; i++) ka[i] = 1.0f/K[i] + a[i];
        }
    }
}
void size_varseis(int* N, void *mptr, void *varptr){
    
    model * m = (model *) mptr;
    variable * var = (variable *) varptr;
    int i;
    int sizevars=1;
    for (i=0;i<m->NDIM;i++){
        sizevars*=N[i]+m->FDORDER;
    }
    var->num_ele=sizevars;
}
void size_varmem(int* N, void *mptr, void *varptr){
    
    model * m = (model *) mptr;
    variable * var = (variable *) varptr;
    int i;
    int sizevars=1;
    for (i=0;i<m->NDIM;i++){
        sizevars*=N[i]+m->FDORDER;
    }
    var->num_ele=sizevars*m->L;
}
void size_varcpmlx(int* N, void *mptr, void *varptr){
    
    model * m = (model *) mptr;
    variable * var = (variable *) varptr;
    int i,j;
    int sizebnd[10]={0};
    for (i=0;i<m->NDIM;i++){
        sizebnd[i]=2*m->NAB;
        for (j=0;j<m->NDIM;j++){
            if (i!=j)
            sizebnd[i]*=N[j];
        }
    }
    var->num_ele=sizebnd[m->NDIM-1];
}
void size_varcpmly(int* N, void *mptr, void *varptr){
    
    model * m = (model *) mptr;
    variable * var = (variable *) varptr;
    int i,j;
    int sizebnd[10]={0};
    for (i=0;i<m->NDIM;i++){
        sizebnd[i]=2*m->NAB;
        for (j=0;j<m->NDIM;j++){
            if (i!=j)
            sizebnd[i]*=N[j];
        }
    }
    var->num_ele=sizebnd[1];
}
void size_varcpmlz(int* N, void *mptr, void *varptr){
    
    model * m = (model *) mptr;
    variable * var = (variable *) varptr;
    int i,j;
    int sizebnd[10]={0};
    for (i=0;i<m->NDIM;i++){
        sizebnd[i]=2*m->NAB;
        for (j=0;j<m->NDIM;j++){
            if (i!=j)
            sizebnd[i]*=N[j];
        }
    }
    var->num_ele=sizebnd[0];
}


int check_stability( void *mptr){
    
    int i;
    int state=0;
    float vpmax=0;
    float vsmax=0;
    float vpmin=99999;
    float vsmin=99999;
    float thisvs, thisvp;
    
    model * m = (model *) mptr;
    float *mu = get_par(m->pars, m->npars, "mu")->gl_par;
    int num_ele;
    float *M = get_par(m->pars, m->npars, "M")->gl_par;
    float *rho = get_par(m->pars, m->npars, "rho")->gl_par;
    
    if (mu){
        num_ele = get_par(m->pars, m->npars, "mu")->num_ele;
        for (i=0;i<num_ele;i++){
            thisvs=sqrt(mu[i]*rho[i])/m->dt*m->dh;
            if (vsmax<thisvs) vsmax=thisvs;
            if (vsmin>thisvs && thisvs>0.1) vsmin=thisvs;
        }
    }
    if (M){
        num_ele = get_par(m->pars, m->npars, "M")->num_ele;
        for (i=0;i<num_ele;i++){
            thisvp=sqrt(M[i]*rho[i])/m->dt*m->dh;
            if (vpmax<thisvp) vpmax=thisvp;
            if (vpmin>thisvp && thisvp>0.1) vpmin=thisvp;
        }
    }
    return check_stability_values(m, vpmax, vpmin, vsmax, vsmin);
}

/* The checks of check_stability() on the velocity extrema (vsmin, vpmin: of the
   values > 0.1; 99999 if none), computed by the host loop above or on the
   device (prep_params.cl, prep_reduce). */
int check_stability_values(model * m, float vpmax, float vpmin, float vsmax, float vsmin){
    int state=0;
    float vmin, vmax;
    float g, gamma, dtstable;
    if (vsmin==99999){
        vmin=vpmin;
    }
    else{
        vmin=vsmin;
    }
    if (vpmax==0){
        vmax=vsmax;
    }
    else{
        vmax=vpmax;
    }
    
    switch (m->FDORDER){
        case 2: g=12.0;
            break;
        case 4: g=8.32;
            break;
        case 6: g=4.77;
            break;
        case 8: g=3.69;
            break;
        case 10: g=3.19;
            break;
        case 12: g=2.91;
            break;
        default: g=0;
            break;
    }
    if ((m->dh>(vmin/m->fmax/g))){
        fprintf(stdout,"Warning: Grid spacing too large, ");
        fprintf(stdout,"dispersion will affect the solution\n");
    }
    /*  gamma for stability estimation (Holberg) */
    switch (m->FDORDER){
        case 2: gamma=1.0;
            break;
        case 4: gamma=1.184614;
            break;
        case 6: gamma=1.283482;
            break;
        case 8: gamma=1.345927;
            break;
        case 10: gamma=1.38766;
            break;
        case 12: gamma=1.417065;
            break;
        default: gamma=1.0;
            break;
    }
    if (m->ND==3)
    dtstable=m->dh/(gamma*sqrt(3.0)*vmax);
    else
    dtstable=m->dh/(gamma*sqrt(2.0)*vmax);
    
    if (m->dt>dtstable){
        state=1;
        fprintf(stderr, "Error: Time step too large, to be stable, ");
        fprintf(stderr, "set dt<%f\n", dtstable);
    }
    
    
    return state;
}

int set_par_scale( void *mptr){
    
    int state=0;
    int i;
    
    model * m = (model *) mptr;
    float *mu = get_par(m->pars, m->npars, "mu")->gl_par;
    int num_ele;
    float *M = get_par(m->pars, m->npars, "M")->gl_par;
    float Mmax=0;
    int scaler = 0;
    
    if (m->FP16>0){
        if (M){
            num_ele = get_par(m->pars, m->npars, "M")->num_ele;
            for (i=0;i<num_ele;i++){
                if (Mmax<M[i]) Mmax=M[i];
            }
        }
        else if (mu){
            num_ele = get_par(m->pars, m->npars, "mu")->num_ele;
            for (i=0;i<num_ele;i++){
                if (Mmax<mu[i]) Mmax=mu[i];
            }
        }
    }
    return set_par_scale_value(m, Mmax);
}

/* par_scale from the maximum of M (FP16 > 0; 0 otherwise), and the scaler of
   the velocities. Mmax from the host loop above or from the device
   (prep_params.cl, prep_reduce). */
int set_par_scale_value(model * m, float Mmax){
    int state=0;
    int scaler = 0;
    if (m->FP16>0){
        //TODO review scaler constant
        scaler = -log2(Mmax*1.0);
    }

    m->par_scale = scaler;
    variable * var;
    var = get_var(m->vars,m->nvars, "vx");
    if (var) var->scaler = scaler;
    var = get_var(m->vars,m->nvars, "vy");
    if (var) var->scaler = scaler;
    var = get_var(m->vars,m->nvars, "vz");
    if (var) var->scaler = scaler;
    
    return state;
}

//Assign parameters list depending on which case of modeling is desired
/* Whether the material averages (rip, rjp, rkp, muipkp, muipjp, mujpkp) are
   computed on the device by average_params.cl. For every FP16 value: with
   FP16>1 the kernels read and write half (average_params.cl, APAR). The
   environment variable SEISCL_HOST_AVERAGE=1 forces the host path, so that
   tests can compare both. */
int use_gpu_average(model * m){
    const char * v = getenv("SEISCL_HOST_AVERAGE");
    if (v && v[0]=='1') return 0;
    return (m->ND==2 || m->ND==3);
}

/* Whether the parameters are prepared on the device (prep_params.cl): the
   transforms, scaling, stability check, tau averages and half conversion of
   the host (Init_model_values) run as kernels on the raw values. Where the
   material averages run on the device (ND 2 and 3). SEISCL_HOST_PREP=1 keeps
   the host path, so that tests can compare both. */
int use_gpu_prep(model * m){
    const char * v = getenv("SEISCL_HOST_PREP");
    if (v && v[0]=='1') return 0;
    return use_gpu_average(m);
}

int assign_modeling_case(model * m){
    
    int i;
    int state =0;
    int ind = 0;
    int nheaders = 0;
    
    // Check Stability function
    m->check_stability=&check_stability;
    m->set_par_scale=&set_par_scale;

    /* Reject the BACK_PROP_TYPE==2 (DFT gradient) configurations that are not
     * supported. Each of these used to run to completion and emit a silently
     * wrong or empty gradient. */
    if (m->GRADOUT==1 && m->BACK_PROP_TYPE==2){
        if (m->NFREQS<1){
            state=1;
            fprintf(stderr,"Error: back_prop_type=2 requires a non-empty "
                           "gradfreqs\n");
        }
        /* All FP16 levels are supported. savefreqs types the wavefield as
           half when FP16>1 and converts on load; the correlation kernels do
           the same for the parameter buffers (Init_OpenCL.c halves cl_par.size
           too), index with the scalar extents NZS/NXS rather than the
           float2-halved NZ/NX, and undo the wavefield scaling with
           src_scale/res_scale/par_scale. Everything downstream of the load is
           float/double -- half is a storage format here, never a compute one.
           A small accuracy cost at FP16>1 is expected from the half storage
           itself. */

    }
    /* Only par_type=0, (vp, vs, rho), is supported by the engine. The other
     * parameterizations are a chain rule on the model grid -- pointwise, and
     * negligible next to propagation -- so they belong in the caller, where
     * they are one expression instead of twelve hand-derived coefficient
     * variants (grad_coef{visc,elast}_{0,1,2,3} plus the _SH twins). Keeping
     * them here duplicated the chain rule between transf_grad (used by
     * BACK_PROP_TYPE==1) and the grad_coef* families (used by
     * BACK_PROP_TYPE==2), which is what let gradrho's chain-rule group sit
     * sign-flipped without being noticed. par_type=1 was in fact unreachable:
     * it was silently reset to 0 and then failed asking for /vp.
     *
     * The SeisCL Python wrapper still accepts param_type 0, 1 and 2 and does
     * the conversion itself, so it is unaffected. This error exists so that
     * other wrappers are told rather than silently given a gradient in the
     * wrong parameterization. */
    if (m->par_type!=0){
        state=1;
        fprintf(stderr,"Error: par_type=%d is no longer supported by the "
                       "engine; only par_type=0 (vp, vs, rho). Convert the "
                       "model and apply the chain rule to the gradient in the "
                       "caller. The SeisCL Python wrapper does this for you.\n",
                m->par_type);
    }
    /* N[0] even: the kernels address the wavefield as float2/half2 along z
       (clprogram.c halves NZ when FP16>0). read_hdf5.c refuses an odd N[0]
       for SeisCL_MPI, for every FP16 value; the PyTorch engine does not read
       an HDF5 file, so the same rule is applied here, for both. */
    if (m->N[0]%2!=0){
        state=1;
        fprintf(stderr,"Error: N[0] (NZ=%d) must be a multiple of 2\n",
                m->N[0]);
    }
    /* SH (ND=21) gradients: velocity receivers only. The SH adjoint and
       gradient kernels (2026-11-19) are the 3-D ones with the terms that
       vanish by symmetry dropped (notes sh-gradient.md); back_prop_type 1
       (elastic) follows update_adjv2D/adjs2D.cl. Stress receivers would need
       the adjoint memory-variable source of the P-SV pressure fix (thesis eq.
       3.29) and an SH rec_grad_s, not written. */
    if (m->GRADOUT==1 && m->ND==21 && m->HOUT==1){
        state=1;
        fprintf(stderr,"Error: Hout=1 is not implemented for ND=21 (SH).\n");
    }
    if (m->GRADOUT==1 && m->ND==21 && m->VARSOUT!=1){
        state=1;
        fprintf(stderr,"Error: gradients in SH (ND=21) need velocity "
                       "receivers (seisout=1).\n");
    }
    /* GRADABS: the full-layer storage is written for the Cerjan taper and the
       float kernels; the CPML adjoint inside the layers is not derived yet. */
    if (m->GRADABS==1 && m->BACK_PROP_TYPE==1 && (m->GRADOUT || m->INPUTRES)){
        if (m->ABS_TYPE!=1 && m->ABS_TYPE!=2){
            state=1;
            fprintf(stderr,"Error: gradabs=1 needs absorbing layers "
                           "(abs_type 1 or 2)\n");
        }
        if (m->ND!=2 && m->ND!=3){
            state=1;
            fprintf(stderr,"Error: gradabs=1 requires ND=2 or ND=3\n");
        }
    }
    /* FP16: 0 float, 1 float with two cells per thread, 2 half storage with
       float arithmetic (header_vec.cl). Half arithmetic (FP16=3) gave no
       practical speed-up and was removed on 2026-11-14 (supervisor). */
    if (m->FP16<0 || m->FP16>2){
        state=1;
        fprintf(stderr,"Error: FP16=%d is not available: use 0, 1 or 2. Half-"
                       "precision arithmetic (FP16=3) is kept in the tags "
                       "v1.0.0 to v1.1.1 of SeisCL only.\n", m->FP16);
    }
#ifdef __SEISCL__
    /* OpenCL: the generated source, residual and output kernels
       (automatic_kernels.c) convert half with CUDA functions. */
    if (m->FP16==2){
        state=1;
        fprintf(stderr,"Error: FP16=2 needs the CUDA build of SeisCL\n");
    }
#endif
    if (m->GRADOUT==1 && m->ND==22){
        state=1;
        fprintf(stderr,"Error: gradient computation is not implemented for "
                       "ND=22 (acoustic): no adjoint or savebnd kernel is "
                       "assigned\n");
    }

    /* BACK_PROP_TYPE==1 reconstructs the forward wavefield by re-running the
     * simulation backward in time from the saved domain-boundary values. For
     * a viscoelastic (L>0) medium the forward equations are dissipative, so
     * running them backward in time is not merely inaccurate but
     * unconditionally unstable: the attenuation that damps the forward
     * simulation amplifies the reconstruction error at the same rate running
     * in reverse, and it blows up regardless of time step or model size. This
     * is a property of the method, not a bug to fix here. It also means
     * gradtaup/gradtaus were never implemented for BACK_PROP_TYPE==1 (in
     * either 2D or 3D: both update_adjs2D.cl and update_adjs3D.cl declare
     * them as read-only kernel arguments, and transf_grad() in calc_grad.c
     * has no taup/taus branch) -- there was no point wiring up gradients that
     * the reconstructed wavefield feeding them cannot be trusted to produce.
     * BACK_PROP_TYPE==2 does not reconstruct the wavefield this way (it
     * correlates spectra accumulated during the single forward pass), so it
     * is unaffected once it supports L>0 (todo item 6). */
    if (m->GRADOUT==1 && m->BACK_PROP_TYPE==1 && m->L>0){
        state=1;
        fprintf(stderr,"Error: back_prop_type=1 cannot compute a gradient for "
                       "a viscoelastic model (L=%d>0): it reconstructs the "
                       "forward wavefield by backpropagating in time, which "
                       "is unconditionally unstable for a dissipative "
                       "(attenuating) medium -- the reconstruction error "
                       "grows exponentially, independent of the vp/vs/rho "
                       "gradient's own correctness. Use back_prop_type=2 "
                       "once it supports L>0, or a fully-checkpointed "
                       "(non-reconstructing) gradient method.\n", m->L);
    }

    /* Definition of each seismic modeling case that has been implemented */
    const char * updatev = NULL;
    const char * updates = NULL;
    const char * updatev_adj = NULL;
    const char * updates_adj = NULL;
    const char * surface = NULL;
    const char * surface_adj = NULL;
    const char * savebnd = NULL;
    const char * headers[3];
    headers[0] = header_CUDACL_source;
    
    if (m->FP16==0){
        headers[1] = header_FD_source;
    }
    else {
        headers[1] = header_FD_fp16_source;
    }
    headers[2] = header_injectbnd_source;
    
    //TODO surface kernels with FP16, Adjoint kernels in 3D with FP16
    /* ND 2 and 3: update kernels are the unified ones (append_update below) */
    if (m->ND==3){
        surface = surface3D_source;
        surface_adj = surface3D_adj_source;
        savebnd = savebnd3D_source;
    }
    else if (m->ND==2){
        surface = surface2D_source;
        surface_adj = surface2D_adj_source;
        savebnd = savebnd2D_source;
    }
     else if (m->ND==21){
         if (m->FP16==0){
             updatev = update_v2D_SH_source;
             updates = update_s2D_SH_source;
             updatev_adj = update_adjv2D_SH_source;
             updates_adj = update_adjs2D_SH_source;
             surface = surface2D_SH_source;
             surface_adj = surface2D_SH_source;   /* kernel surface_adj in the same file */
             savebnd = savebnd2D_SH_source;
         }
         else{
             state = 1;
             fprintf(stderr,"Error: Only FP16=0 is supported for ND=21 \n");
         }
     }
    else if (m->ND==22){
        if (m->FP16==0){
            updatev = update_v2D_acc_source;
            updates = update_s2D_acc_source;
//            updatev_adj = update_adjv2D_acc_source;
//            updates_adj = update_adjs2D_acc_source;
//            surface = surface2D_acc_source;
//            savebnd = savebnd2D_source;
        }
        else{
            state = 1;
            fprintf(stderr,"Error: Only FP16=0 is supported for ND=22 \n");
        }
    }
    m->nupdates=2;
    GMALLOC(m->ups_f, m->nupdates*sizeof(update));
    ind=0;
    /* Unified kernels (header_vec.cl): one source for FP16 0, 1 and 2, float
       arithmetic, storage and cells per thread set at compile time. */
    const char * vheaders[3] = {headers[0], headers[1], header_vec_source};
    if (m->ND==2 || m->ND==3 || m->ND==21){
        /* SH (ND 21) uses the same format and headers as P-SV since 2026-11-19 */
        __GUARD append_update(m->ups_f, &ind, "update_v",
                              m->ND==3 ? update_v3D_source : (m->ND==21 ? updatev : update_v2D_source),
                              3, vheaders);
    }
    else {
        __GUARD append_update(m->ups_f, &ind, "update_v", updatev, 2, headers);
    }
    if (m->ND==2 || m->ND==3 || m->ND==21){
        /* SH (ND 21) uses the same format and headers as P-SV since 2026-11-19 */
        __GUARD append_update(m->ups_f, &ind, "update_s",
                              m->ND==3 ? update_s3D_source : (m->ND==21 ? updates : update_s2D_source),
                              3, vheaders);
    }
    else {
        __GUARD append_update(m->ups_f, &ind, "update_s", updates, 2, headers);
    }
    if (m->GRADOUT){
        GMALLOC(m->ups_adj, m->nupdates*sizeof(update));
        ind=0;
        if (m->BACK_PROP_TYPE==1){
            nheaders=3;
        }
        else{
            nheaders=2;
        }
        if (m->ND==2 || m->ND==3 || m->ND==21){
            /* SH: same format and headers as P-SV since 2026-11-19 */
            const char * aheaders[4] = {headers[0], headers[1],
                                        header_injectbnd_source, header_vec_source};
            __GUARD append_update(m->ups_adj, &ind, "update_adjv",
                                  m->ND==3 ? update_adjv3D_source :
                                  (m->ND==21 ? updatev_adj : update_adjv2D_source),
                                  4, aheaders);
            __GUARD append_update(m->ups_adj, &ind, "update_adjs",
                                  m->ND==3 ? update_adjs3D_source :
                                  (m->ND==21 ? updates_adj : update_adjs2D_source),
                                  4, aheaders);
        }
        else {
            __GUARD append_update(m->ups_adj, &ind, "update_adjv", updatev_adj, nheaders, headers);
            __GUARD append_update(m->ups_adj, &ind, "update_adjs", updates_adj, nheaders, headers);
        }
    }
    /* Residual terms of the gradient at the receiver cells (rec_grad.cl). */
    if (m->GRADOUT && m->BACK_PROP_TYPE==1 && (m->ND==2 || m->ND==3)){
        __GUARD prog_source(&m->bnd_cnds.rec_grad_v, "rec_grad_v",
                            rec_grad_source, 3, vheaders);
        __GUARD prog_source(&m->bnd_cnds.rec_grad_s, "rec_grad_s",
                            rec_grad_source, 3, vheaders);
    }
    /* SH: velocity receivers only (refused otherwise), vy -> gradrho */
    if (m->GRADOUT && m->BACK_PROP_TYPE==1 && m->ND==21){
        __GUARD prog_source(&m->bnd_cnds.rec_grad_v, "rec_grad_v",
                            rec_grad_SH_source, 3, vheaders);
    }
    /* Exact CPML adjoint (BACK_PROP_TYPE 1): filters run before each adjoint
       update (time_stepping.c, update_grid_adj). */
    if (m->ABS_TYPE==1 && m->GRADOUT && m->BACK_PROP_TYPE==1){
        const char * cpmladj = m->ND==3 ? cpml_adj3D_source :
                               (m->ND==21 ? cpml_adj2D_SH_source : cpml_adj2D_source);
        __GUARD prog_source(&m->bnd_cnds.cpml_adj_v, "cpml_adj_v",
                            cpmladj, 3, vheaders);
        __GUARD prog_source(&m->bnd_cnds.cpml_adj_s, "cpml_adj_s",
                            cpmladj, 3, vheaders);
    }
    if (m->FREESURF==1){
        /* ND 2, 3: the unified kernels (header_vec.cl storage); SH: its own */
        int uni2 = (m->ND==2 || m->ND==3);
        const char * gheaders[4] = {headers[0], headers[1],
                                    header_injectbnd_source, header_vec_source};
        if (uni2 || m->ND==21){
            /* SH: surface2D_SH.cl, same format and headers since 2026-11-19 */
            __GUARD prog_source(&m->bnd_cnds.surf, "freesurface",
                                m->ND==3 ? surface3D_source : (m->ND==21 ? surface : surface2D_source),
                                3, vheaders);
        }
        else {
            __GUARD prog_source(&m->bnd_cnds.surf, "freesurface", surface, 2, headers);
        }
        if (m->GRADOUT){
            __GUARD prog_source(&m->bnd_cnds.surf_adj, "surface_adj", surface_adj,
                                (uni2 || m->ND==21) ? 3 : 2,
                                (uni2 || m->ND==21) ? vheaders : headers);
            if (uni2){
                __GUARD prog_source(&m->bnd_cnds.surf_adj_fold,
                                    "surface_adj_fold", surface_adj,
                                    uni2 ? 3 : 2, uni2 ? vheaders : headers);
            }
            /* Direct-form gradient in the top rows (surface2D/3D_adj.cl) */
            if (uni2 && m->BACK_PROP_TYPE==1
                && m->restype==0){
                if (uni2){
                    __GUARD prog_source(&m->bnd_cnds.surf_grad, "surface_grad",
                                        surface_adj, 4, gheaders);
                }
                else {
                    __GUARD prog_source(&m->bnd_cnds.surf_grad, "surface_grad",
                                        surface_adj, 3, headers);
                }
            }
        }
    }
    /* DFT gradient correlation, on the device: 2D P-SV, SH and 3D, elastic
       and viscoelastic. There is no other implementation (the host
       calc_grad() was removed on 2026-11-14), so a case without a kernel is
       refused. */
    if (m->GRADOUT && m->BACK_PROP_TYPE==2){
        const char * graddft = NULL;
        if (m->L==0){
            if (m->ND==2)       graddft = grad_dft2D_source;
            else if (m->ND==21) graddft = grad_dft2D_SH_source;
            else if (m->ND==3)  graddft = grad_dft3D_source;
        }
        else {
            /* Viscoelastic */
            if (m->ND==2)       graddft = grad_dft2D_visc_source;
            else if (m->ND==21) graddft = grad_dft2D_SH_visc_source;
            else if (m->ND==3)  graddft = grad_dft3D_visc_source;
        }
        if (graddft){
            if (m->ND==21 && m->HOUT){
                state=1;
                fprintf(stderr,"Error: Hout=1 is not implemented for ND=21 "
                               "(SH).\n");
            }
            __GUARD prog_source(&m->grads.calc_grad, "calc_grad_dft",
                                (char*)graddft, 2, headers);
        }
        else {
            state=1;
            fprintf(stderr,"Error: back_prop_type=2 has no gradient kernel "
                           "for ND=%d\n", m->ND);
        }
    }
    if ((m->GRADOUT || m->INPUTRES) && m->BACK_PROP_TYPE==1){
        __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd, 1, headers);
    }
    /*GPU port of the staggered-grid material-parameter averaging (elastic
      only -- see src/average_params.cl and
      notes/vacuum-freesurface-plan.md, Phase 8). Replaces the CPU
      ave_arithmetic_rho()/ave_harmonic_mu() computation for "rip"/"rkp"/
      "muipkp" (ND==2 and ND==3) and "rjp"/"muipjp"/"mujpkp" (ND==3 only)
      below (their transform is NULL, so Init_model()'s transform loop
      skips them and this kernel is the only thing that fills them). Only
      needs header_CUDACL_source (headers count 1, like savebnd above) --
      no FD stencil or CPML macros.*/
    /* FP16>1 stores the parameters as half on the device (Init_OpenCL.c
       halves cl_par.size); average_params.cl then reads and writes half
       (it used to write float, twice past the end of each buffer: Xid 31 on
       cgm-geo1). With SEISCL_HOST_AVERAGE=1 the averages are computed on the
       host instead, by the transforms below. */
    int gpu_avg = use_gpu_average(m);
    m->gpu_prep = use_gpu_prep(m);
    if (m->gpu_prep){
        __GUARD prog_source(&m->par_avg.prep_reduce, "prep_reduce", prep_params_source,
                            1, headers);
        __GUARD prog_source(&m->par_avg.prep_store, "prep_store", prep_params_source,
                            1, headers);
        if (m->L>0 && (m->ND==2 || m->ND==3)){
            __GUARD prog_source(&m->par_avg.tausipkp, "prep_tausipkp", prep_params_source,
                                1, headers);
        }
        if (m->L>0 && m->ND==3){
            __GUARD prog_source(&m->par_avg.tausipjp, "prep_tausipjp", prep_params_source,
                                1, headers);
            __GUARD prog_source(&m->par_avg.tausjpkp, "prep_tausjpkp", prep_params_source,
                                1, headers);
        }
    }
    /* Residual scaling on the device (dev_io.cl): CUDA, with the device
       preparation of the parameters, which it reads there. */
    #ifndef __SEISCL__
    m->dev_res = m->gpu_prep;
    #else
    m->dev_res = 0;
    #endif
    if (m->dev_res){
        __GUARD prog_source(&m->dio.gather, "dio_rec_gather", dev_io_source, 1, headers);
        __GUARD prog_source(&m->dio.res_scale, "dio_res_scale", dev_io_source, 1, headers);
        __GUARD prog_source(&m->dio.recgrad, "dio_recgrad", dev_io_source, 1, headers);
        __GUARD prog_source(&m->dio.recgrad_add, "dio_recgrad_add", dev_io_source, 1, headers);
        __GUARD prog_source(&m->dio.add_out, "dio_add_out", dev_io_source, 1, headers);
        __GUARD prog_source(&m->dio.recmod, "dio_recmod", dev_io_source, 1, headers);
    }
    /* Gradient post-processing on the device (grad_post.cl): with dev_res,
       par_type 0 (the only one the engine runs), without the Hessian. */
    m->dev_grad = m->dev_res && m->GRADOUT && !m->HOUT && m->par_type==0
                  && (m->BACK_PROP_TYPE==1 || m->BACK_PROP_TYPE==2);
    {   /* SEISCL_HOST_GRAD=1: the host post-processing (tests) */
        const char * v = getenv("SEISCL_HOST_GRAD");
        if (v && v[0]=='1') m->dev_grad = 0;
    }
    if (m->dev_grad){
        clprogram * gp[12] = {&m->gp.cvt, &m->gp.addrec, &m->gp.scale, &m->gp.unbuoy,
                              &m->gp.trho, &m->gp.tmu_w, &m->gp.tmu_pass, &m->gp.ttau_pass,
                              &m->gp.edge, &m->gp.sn_pre, &m->gp.sn_post, &m->gp.chain};
        const char * gn[12] = {"gp_cvt", "gp_addrec", "gp_scale", "gp_unbuoy", "gp_trho",
                               "gp_tmu_w", "gp_tmu_pass", "gp_ttau_pass", "gp_edge",
                               "gp_sn_pre", "gp_sn_post", "gp_chain"};
        for (int k=0;k<12;k++){
            __GUARD prog_source(gp[k], (char *)gn[k], grad_post_source, 1, headers);
        }
    }
    if (gpu_avg){
        __GUARD prog_source(&m->par_avg.rip, "ave_rip", average_params_source,
                            1, headers);
        __GUARD prog_source(&m->par_avg.rkp, "ave_rkp", average_params_source,
                            1, headers);
        __GUARD prog_source(&m->par_avg.muipkp, "ave_muipkp",
                            average_params_source, 1, headers);
    }
    if (gpu_avg && m->ND==3){
        __GUARD prog_source(&m->par_avg.rjp, "ave_rjp", average_params_source,
                            1, headers);
        __GUARD prog_source(&m->par_avg.muipjp, "ave_muipjp",
                            average_params_source, 1, headers);
        __GUARD prog_source(&m->par_avg.mujpkp, "ave_mujpkp",
                            average_params_source, 1, headers);
    }
    /* There is deliberately no FREESURF==2 gradient restriction here.

       There used to be one rejecting FREESURF==2 + GRADOUT unless
       (BACK_PROP_TYPE==1 && restype==1), because the vacuum band the engine
       itself used to zero produced NaN gradients three different ways. Now
       that the engine no longer creates that band, keying a restriction on
       FREESURF==2 tests the wrong thing in both directions: it would reject
       a perfectly ordinary FREESURF==2 run on a fully solid model (nothing
       is zero, nothing can divide by zero), while still letting through a
       FREESURF==0 run on a model the user zeroed, which carries exactly the
       same risk.

       The real condition is "the model contains zero-material cells", and
       that is a pre-existing SeisCL property independent of FREESURF -- it
       already applies to any acoustic/water layer with mu==0. The
       divide-by-zero sites it used to guard have since been given their own
       guards where they belong: transf_grad()'s 1/rho (see
       notes/back-prop-type1-zero-material-nan.md) and calc_grad.c's
       coefficient routines, which skip cells with rho==0 or
       (ND*M-2(ND-1)mu)^2==0 rather than evaluating 0/0. What remains
       unguarded is BACK_PROP_TYPE==1's boundary-reconstruction shell
       (header_injectbnd.cl), which the old comment itself recorded as NOT
       specific to FREESURF==2 -- tracked in notes/vacuum-freesurface-plan.md
       rather than papered over with a flag check that does not correspond to
       it. */

    /*___________________Assign material parameters__________________________ */
    
    m->npars=14;
    GMALLOC(m->pars, sizeof(parameter)*m->npars);
    ind=0;
    if (m->ND!=21){
        __GUARD append_par(m, &ind, "M", "/M", &M);
    }
    if (m->ND!=22){
        __GUARD append_par(m, &ind, "mu", "/mu", &mu);
    }
    __GUARD append_par(m, &ind, "rho", "/rho", &rho);
    if (m->L>0){
        __GUARD append_par(m, &ind, "taup", "/taup", NULL);
        if (m->ND!=22){
            __GUARD append_par(m, &ind, "taus", "/taus", NULL);
        }
    }
    if (m->ND!=21){
        // ND==2 and ND==3: rip/rjp/rkp/muipkp/muipjp/mujpkp are computed
        // on-device instead (see the par_avg kernel registration above) --
        // transform=NULL so Init_model()'s transform loop leaves their raw
        // host gl_par alone; Init_OpenCL.c reads the GPU-computed values
        // back to host right after launching the kernel, since res_scale()
        // (residuals.c) reads rip/rkp's *host* gl_par directly.
        // transform NULL when the GPU computes the average (gpu_avg).
        __GUARD append_par(m, &ind, "rip", NULL, gpu_avg ? NULL : &rip);
        if (m->ND==3){
            __GUARD append_par(m, &ind, "rjp", NULL, gpu_avg ? NULL : &rjp);
        }
        __GUARD append_par(m, &ind, "rkp", NULL, gpu_avg ? NULL : &rkp);
    }
    if (m->ND==2 || m->ND==3){
        __GUARD append_par(m, &ind, "muipkp", NULL, gpu_avg ? NULL : &muipkp);
    }
    if (m->ND==3){
        __GUARD append_par(m, &ind, "muipjp", NULL, gpu_avg ? NULL : &muipjp);
        __GUARD append_par(m, &ind, "mujpkp", NULL, gpu_avg ? NULL : &mujpkp);
    }
    if (m->ND==21){
        /* SH stresses: sxy at (x+1/2, z) uses muipjp, syz at (x, z+1/2) mujpkp,
           as the same stresses in 3-D (host averages; SH has no GPU path). */
        __GUARD append_par(m, &ind, "muipjp", NULL, &muipjp);
        __GUARD append_par(m, &ind, "mujpkp", NULL, &mujpkp);
    }
    if (m->L>0){
        /* gpu_prep: computed on the device (prep_tau*), no host transform */
        if (m->ND==2 || m->ND==3){
            __GUARD append_par(m, &ind, "tausipkp", NULL, m->gpu_prep ? NULL : &tausipkp);
        }
        if (m->ND==3 || m->ND==21){
            __GUARD append_par(m, &ind, "tausipjp", NULL, m->gpu_prep ? NULL : &tausipjp);
            __GUARD append_par(m, &ind, "tausjpkp", NULL, m->gpu_prep ? NULL : &tausjpkp);
        }
    }
    m->npars = ind;
    
    
    /*___________________Assign seismic variables__________________________ */
    m->nvars=15;
    if (m->NAB<1) {m->ABS_TYPE=0;}
    if (m->ABS_TYPE==1) {m->nvars+=18+9;}   /* + cpml_adj scratch (cpfa...) */
    GMALLOC(m->vars, sizeof(variable)*m->nvars);
    
    ind=0;
    
    if (m->ND!=21){
        __GUARD append_var(m, &ind, "vx", 1, 1, &size_varseis);
    }
    
    if (m->ND==21 || m->ND==3){
        __GUARD append_var(m, &ind, "vy", 1, 1, &size_varseis);
    }
    
    if (m->ND!=21){
        __GUARD append_var(m, &ind, "vz", 1, 1, &size_varseis);
    }

    if (m->ND==2 || m->ND==3){
        __GUARD append_var(m, &ind, "sxx", 1, 2, &size_varseis);
        __GUARD append_var(m, &ind, "szz", 1, 2, &size_varseis);
        __GUARD append_var(m, &ind, "sxz", 1, 2, &size_varseis);
    }
    
    if (m->ND==21 || m->ND==3){
        
        __GUARD append_var(m, &ind, "sxy", 1, 2, &size_varseis);
        __GUARD append_var(m, &ind, "syz", 1, 2, &size_varseis);
    }
    if (m->ND==3){
        __GUARD append_var(m, &ind, "syy", 1, 2, &size_varseis);
    }
    if (m->ND==22){
        __GUARD append_var(m, &ind, "p", 1, 2, &size_varseis);
    }
    if (m->L>0){
        if (m->ND==2 || m->ND==3){
            __GUARD append_var(m, &ind, "rxx", 1, 0, &size_varmem);
            __GUARD append_var(m, &ind, "rzz", 1, 0, &size_varmem);
            __GUARD append_var(m, &ind, "rxz", 1, 0, &size_varmem);
        }
        if (m->ND==21 || m->ND==3){
            __GUARD append_var(m, &ind, "rxy", 1, 0, &size_varmem);
            __GUARD append_var(m, &ind, "ryz", 1, 0, &size_varmem);
        }
        if (m->ND==3){
            __GUARD append_var(m, &ind, "ryy", 1, 0, &size_varmem);
        }
        if (m->ND==22){
            __GUARD append_var(m, &ind, "rp", 1, 0, &size_varmem);
        }
    }
    
    if (m->ABS_TYPE==1){
        if (m->ND!=21){
            __GUARD append_var(m, &ind, "psi_vx_x", 0, 0, &size_varcpmlx);
            __GUARD append_var(m, &ind, "psi_vz_z", 0, 0, &size_varcpmlz);
        }
        if (m->ND==2 || m->ND==3){
            __GUARD append_var(m, &ind, "psi_sxx_x", 0, 0, &size_varcpmlx);
            __GUARD append_var(m, &ind, "psi_sxz_x", 0, 0, &size_varcpmlx);
            __GUARD append_var(m, &ind, "psi_szz_z", 0, 0, &size_varcpmlz);
            __GUARD append_var(m, &ind, "psi_sxz_z", 0, 0, &size_varcpmlz);
            __GUARD append_var(m, &ind, "psi_vz_x", 0, 0, &size_varcpmlx);
            __GUARD append_var(m, &ind, "psi_vx_z", 0, 0, &size_varcpmlz);
        }
        if (m->ND==21 || m->ND==3){
            __GUARD append_var(m, &ind, "psi_sxy_x", 0, 0, &size_varcpmlx);
            __GUARD append_var(m, &ind, "psi_syz_z", 0, 0, &size_varcpmlz);
            __GUARD append_var(m, &ind, "psi_vy_x", 0, 0, &size_varcpmlx);
            __GUARD append_var(m, &ind, "psi_vy_z", 0, 0, &size_varcpmlz);
        }
        if (m->ND==3){
            __GUARD append_var(m, &ind, "psi_syy_y", 0, 0, &size_varcpmly);
            __GUARD append_var(m, &ind, "psi_sxy_y", 0, 0, &size_varcpmly);
            __GUARD append_var(m, &ind, "psi_syz_y", 0, 0, &size_varcpmly);
            __GUARD append_var(m, &ind, "psi_vx_y", 0, 0, &size_varcpmly);
            __GUARD append_var(m, &ind, "psi_vy_y", 0, 0, &size_varcpmly);
            __GUARD append_var(m, &ind, "psi_vz_y", 0, 0, &size_varcpmly);
        }
        if (m->ND==22){
            __GUARD append_var(m, &ind, "psi_p_x", 0, 0, &size_varcpmlx);
            __GUARD append_var(m, &ind, "psi_p_z", 0, 0, &size_varcpmlz);
        }
        /* Scratch fields of the exact CPML adjoint (cpml_adj2D.cl): the
           adjoint fields filtered before their spatial derivatives. Also
           created by the forward leg of the INPUTRES two-call protocol, whose
           checkpoint must hold the same variables as the gradient leg. */
        if ((m->GRADOUT || m->INPUTRES) && m->BACK_PROP_TYPE==1 && m->ND==2){
            __GUARD append_var(m, &ind, "cpfa", 0, 0, &size_varseis);
            __GUARD append_var(m, &ind, "cpfb", 0, 0, &size_varseis);
            __GUARD append_var(m, &ind, "cpfc", 0, 0, &size_varseis);
            __GUARD append_var(m, &ind, "cpfd", 0, 0, &size_varseis);
        }
        if ((m->GRADOUT || m->INPUTRES) && m->BACK_PROP_TYPE==1 && m->ND==21){
            __GUARD append_var(m, &ind, "cpfa", 0, 0, &size_varseis);   /* cpml_adj2D_SH.cl */
            __GUARD append_var(m, &ind, "cpfb", 0, 0, &size_varseis);
        }
        if ((m->GRADOUT || m->INPUTRES) && m->BACK_PROP_TYPE==1 && m->ND==3){
            __GUARD append_var(m, &ind, "cpf1", 0, 0, &size_varseis);
            __GUARD append_var(m, &ind, "cpf2", 0, 0, &size_varseis);
            __GUARD append_var(m, &ind, "cpf3", 0, 0, &size_varseis);
            __GUARD append_var(m, &ind, "cpf4", 0, 0, &size_varseis);
            __GUARD append_var(m, &ind, "cpf5", 0, 0, &size_varseis);
            __GUARD append_var(m, &ind, "cpf6", 0, 0, &size_varseis);
            __GUARD append_var(m, &ind, "cpf7", 0, 0, &size_varseis);
            __GUARD append_var(m, &ind, "cpf8", 0, 0, &size_varseis);
            __GUARD append_var(m, &ind, "cpf9", 0, 0, &size_varseis);
        }

    }
    m->nvars = ind;
    
    
    /*___Create variables that are transformed from updated variables_______ */
    if (m->ND==2 || m->ND==3){
        m->ntvars=1;
        GMALLOC(m->trans_vars, sizeof(variable)*m->ntvars);
        m->trans_vars[0].name="p";
        m->trans_vars[0].n2ave=m->ND;
        GMALLOC(m->trans_vars[0].var2ave, sizeof(char *)*m->ND);
        m->trans_vars[0].var2ave[0]="sxx";
        m->trans_vars[0].var2ave[1]="szz";
        if (m->ND==3){
            m->trans_vars[0].var2ave[2]="syy";
        }
    }

    /*___Create variables that are transformed from updated variables_______ */
    if (m->GRADOUT && m->BACK_PROP_TYPE==1){
        GMALLOC(m->vars_adj, sizeof(variable)*m->nvars);
        for (i=0;i<m->nvars;i++){
            m->vars_adj[i]=m->vars[i];
        }
    }
    
    /* Arrays of constants size on all devices. The most constants we have
     here is 23 */
    GMALLOC(m->csts, sizeof(constants)*29);
    if (m->ABS_TYPE==2)
    __GUARD append_cst(m,"taper",NULL,m->NAB,&taper);
    if (m->ABS_TYPE==1){
        __GUARD append_cst(m,"K_z",NULL,2*m->NAB,&CPML_z);
        __GUARD append_cst(m,"a_z",NULL,2*m->NAB,NULL);
        __GUARD append_cst(m,"b_z",NULL,2*m->NAB,NULL);
        __GUARD append_cst(m,"K_z_half",NULL,2*m->NAB,NULL);
        __GUARD append_cst(m,"a_z_half",NULL,2*m->NAB,NULL);
        __GUARD append_cst(m,"b_z_half",NULL,2*m->NAB,NULL);
        __GUARD append_cst(m,"K_x",NULL,2*m->NAB,&CPML_x);
        __GUARD append_cst(m,"a_x",NULL,2*m->NAB,NULL);
        __GUARD append_cst(m,"b_x",NULL,2*m->NAB,NULL);
        __GUARD append_cst(m,"K_x_half",NULL,2*m->NAB,NULL);
        __GUARD append_cst(m,"a_x_half",NULL,2*m->NAB,NULL);
        __GUARD append_cst(m,"b_x_half",NULL,2*m->NAB,NULL);
        if (m->ND==3){
            __GUARD append_cst(m,"K_y",NULL,2*m->NAB,&CPML_y);
            __GUARD append_cst(m,"a_y",NULL,2*m->NAB,NULL);
            __GUARD append_cst(m,"b_y",NULL,2*m->NAB,NULL);
            __GUARD append_cst(m,"K_y_half",NULL,2*m->NAB,NULL);
            __GUARD append_cst(m,"a_y_half",NULL,2*m->NAB,NULL);
            __GUARD append_cst(m,"b_y_half",NULL,2*m->NAB,NULL);
            /* experiment, CPMLINL: 1/K + a (CPML_ka) */
            __GUARD append_cst(m,"Ka_x",NULL,2*m->NAB,NULL);
            __GUARD append_cst(m,"Ka_x_half",NULL,2*m->NAB,NULL);
            __GUARD append_cst(m,"Ka_y",NULL,2*m->NAB,NULL);
            __GUARD append_cst(m,"Ka_y_half",NULL,2*m->NAB,NULL);
            __GUARD append_cst(m,"Ka_z",NULL,2*m->NAB,NULL);
            __GUARD append_cst(m,"Ka_z_half",NULL,2*m->NAB,&CPML_ka);
        }
        
    }
    if (m->L>0){
        __GUARD append_cst(m,"FL","/FL",m->L,NULL);
        __GUARD append_cst(m,"eta",NULL,m->L,&eta);
    }
    if (m->GRADOUT && m->BACK_PROP_TYPE==2){
        __GUARD append_cst(m,"gradfreqs","/gradfreqs",m->NFREQS,NULL);
        __GUARD append_cst(m,"gradfreqsn",NULL,m->NFREQS,&gradfreqsn);
    }
    
    //Assign dimensions name
    if (m->ND==3){
        m->N_names[0]="Z";
        m->N_names[1]="Y";
        m->N_names[2]="X";
    }
    else{
        m->N_names[0]="Z";
        m->N_names[1]="X";
    }

    //Check the name of variable to read depending on the chosen parametrization
    if (!state){

        if (m->par_type==2){
            for (i=0;i<m->npars;i++){
                if (strcmp(m->pars[i].name,"M")==0)
                    m->pars[i].to_read="/Ip";
                if (strcmp(m->pars[i].name,"mu")==0)
                    m->pars[i].to_read="/Is";
            }
        }
        else if (m->par_type==3){
            if (m->L==0) {
                state=1;
                fprintf(stderr,"Error: par_type=3 requires Viscoelastic modeling \n");
            }
            for (i=0;i<m->npars;i++){
                if (strcmp(m->pars[i].name,"M")==0)
                    m->pars[i].to_read="/vpR";
                if (strcmp(m->pars[i].name,"mu")==0)
                    m->pars[i].to_read="/vsR";
                if (strcmp(m->pars[i].name,"taup")==0)
                    m->pars[i].to_read="/vpI";
                if (strcmp(m->pars[i].name,"taus")==0)
                    m->pars[i].to_read="/vsI";
            }
        }
        else if (m->par_type==1){
            /* (M, mu, rho). This branch used to be absent: par_type==1 fell
             * into the else below, which forcibly reset par_type to 0 and then
             * asked for /vp and /vs. So par_type=1 was silently downgraded to
             * par_type=0 and the run died with "Variable /vp is not defined",
             * while the par_type==1 arms of the parameter transforms
             * (assign_modeling_case.c:276-280, :351-354) and the
             * grad_coefelast_1 / grad_coefvisc_1 coefficient sets were all
             * unreachable. */
            for (i=0;i<m->npars;i++){
                if (strcmp(m->pars[i].name,"M")==0)
                    m->pars[i].to_read="/M";
                if (strcmp(m->pars[i].name,"mu")==0)
                    m->pars[i].to_read="/mu";
            }
        }
        else {
            m->par_type=0;
            for (i=0;i<m->npars;i++){
                if (strcmp(m->pars[i].name,"M")==0)
                    m->pars[i].to_read="/vp";
                if (strcmp(m->pars[i].name,"mu")==0)
                    m->pars[i].to_read="/vs";
            }
        }
    }

    //Flag variables to output
    if (!state && m->GRADOUT==0){
        if (m->VARSOUT==1){
            for (i=0;i<m->nvars;i++){
                if (strcmp(m->vars[i].name,"vx")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"vy")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"vz")==0)
                    m->vars[i].to_output=1;
            }
        }
        if (m->VARSOUT==2){
            if (m->ND==21){
                fprintf(stderr,"Error: Cannot output pressure"
                                " for SH modeling \n");
                return 1;
            }
            if (m->ND==2 || m->ND==3){
                for (i=0;i<m->ntvars;i++){
                    if (strcmp(m->trans_vars[i].name,"p")==0)
                        m->trans_vars[i].to_output=1;
                }
            }
            if (m->ND==22){
                for (i=0;i<m->nvars;i++){
                    if (strcmp(m->vars[i].name,"p")==0)
                        m->vars[i].to_output=1;
                }
            }
        }
        if (m->VARSOUT==3){
            for (i=0;i<m->nvars;i++){
                if (strcmp(m->vars[i].name,"sxx")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"syy")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"szz")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"sxz")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"syz")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"sxy")==0)
                    m->vars[i].to_output=1;
            }
        }
        if (m->VARSOUT==4){
            for (i=0;i<m->nvars;i++){
                if (strcmp(m->vars[i].name,"vx")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"vy")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"vz")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"sxx")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"syy")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"szz")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"sxz")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"syz")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"sxy")==0)
                    m->vars[i].to_output=1;
            }
            if (m->ND==2 || m->ND==3){
                for (i=0;i<m->ntvars;i++){
                    if (strcmp(m->trans_vars[i].name,"p")==0)
                    m->trans_vars[i].to_output=1;
                }
            }
            if (m->ND==22){
                for (i=0;i<m->nvars;i++){
                    if (strcmp(m->vars[i].name,"p")==0)
                    m->vars[i].to_output=1;
                }
            }
        }
    }

    return state;
}


