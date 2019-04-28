//
//  assign_modeling_case.c
//  SeisCL
//
//  Created by Gabriel Fabien-Ouellet on 17-07-06.
//  Copyright © 2017 Gabriel Fabien-Ouellet. All rights reserved.
//

#include "F.h"

/*Loading files autmatically created by the makefile that contain the *.cl kernels in a c string.
 This way, no .cl file need to be read and there is no need to be in the executable directory to execute SeisCL.*/
#include "savebnd2D.hcl"
#include "savebnd3D.hcl"
#include "surface2D.hcl"
#include "surface2D_adj.hcl"
#include "surface2D_SH.hcl"
#include "surface3D.hcl"
#include "update_adjs2D.hcl"
#include "update_adjs2D_half2.hcl"
#include "update_adjs2D_SH.hcl"
#include "update_adjs3D.hcl"
#include "update_adjv2D.hcl"
#include "update_adjv2D_half2.hcl"
#include "update_adjv2D_SH.hcl"
#include "update_adjv3D.hcl"
#include "update_s2D.hcl"
#include "update_s2D_half2.hcl"
#include "update_s2D_SH.hcl"
#include "update_s2D_acc.hcl"
#include "update_s3D.hcl"
#include "update_s3D_half2.hcl"
#include "update_v2D.hcl"
#include "update_v2D_half2.hcl"
#include "update_v2D_SH.hcl"
#include "update_v2D_acc.hcl"
#include "update_v3D.hcl"
#include "update_v3D_half2.hcl"


void ave_arithmetic1(float * pin, float * pout, int * N, int ndim, int  dir[3]){
    
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
    
    
    for (k=0;k<NZ-dir[0];k++){
        for (j=0;j<NY-dir[1];j++){
            for (i=0;i<NX-dir[2];i++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                ind2 = (i+dir[2])*NY*NZ+(j+dir[1])*NZ+(k+dir[0]);
                pout[ind1]=0.5*( pin[ind1]+pin[ind2]);
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
void ave_arithmetic2(float *pin, float *pout, int *N, int ndim, int dir[2][3]) {
    
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
    
    if (dir[1][2]==1){
        NX0=NX-1;
    }
    else if (dir[1][1]==1){
        NY0=NY-1;
    }
    if (dir[1][0]==1){
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
void ave_harmonic1(float * pin, float * pout, int * N, int ndim, int  dir[3]){
    
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
    
    
    for (k=0;k<NZ-dir[0];k++){
        for (j=0;j<NY-dir[1];j++){
            for (i=0;i<NX-dir[2];i++){
                ind1 = (i  )*NY*NZ+(j)*NZ+(k);
                ind2 = (i+dir[2])*NY*NZ+(j+dir[1])*NZ+(k+dir[0]);
                pout[ind1]=2.0*( 1.0/pin[ind1]+1.0/pin[ind2]);
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

void ave_harmonic(float * pin, float * pout, int * N, int ndim, int dir[2][3]) {
    
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
    
    if (dir[1][2]==1){
        NX0=NX-1;
    }
    else if (dir[1][1]==1){
        NY0=NY-1;
    }
    if (dir[1][0]==1){
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
        
        for (i=0;i<num_ele;i++){
            sumu=0.0;
            for (l=0;l<m->L;l++){
                sumu+= ((ws*ws*pts[l]*pts[l]*taus[i])
                        /(1.0+ws*ws*pts[l]*pts[l]));
            }
            mu[i]=mu[i]/(1.0+sumu);
        }
        
        free(pts);
    }
    
    int scaler = 0;
    if (m->FP16>0){
        if (m->par_scale == 0){
            m->set_par_scale( (void*) m);
        }
        scaler = m->par_scale;
        
        for (i=0;i<num_ele;i++){
            mu[i]*=powf(2, scaler);
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
        
        for (i=0;i<num_ele;i++){
            M[i]*=powf(2,scaler);
        }
    }
    
}

void rho(void * mptr){
    
    model * m = (model *) mptr;
    float *rho = get_par(m->pars, m->npars, "rho")->gl_par;
    int scaler=m->par_scale;

    int i;
    int num_ele = get_par(m->pars, m->npars, "rho")->num_ele;
    for (i=0;i<num_ele;i++){
        rho[i]=1.0/rho[i]*m->dt/m->dh*powf(2,-scaler);
    }
}

void rip(void * mptr){
    
    model * m = (model *) mptr;
    float *rho = get_par(m->pars, m->npars, "rho")->gl_par;
    float *rip = get_par(m->pars, m->npars, "rip")->gl_par;
    int dir[3]={0,0,1};
    ave_arithmetic1(rho, rip, m->N, m->NDIM, dir);
}
void rjp(void * mptr){
    
    model * m = (model *) mptr;
    float *rho = get_par(m->pars, m->npars, "rho")->gl_par;
    float *rjp = get_par(m->pars, m->npars, "rjp")->gl_par;
    int dir[3]={0,1,0};
    ave_arithmetic1(rho, rjp, m->N, m->NDIM, dir);
}
void rkp(void * mptr){
    
    model * m = (model *) mptr;
    float *rho = get_par(m->pars, m->npars, "rho")->gl_par;
    float *rkp = get_par(m->pars, m->npars, "rkp")->gl_par;
    int dir[3]={1,0,0};
    ave_arithmetic1(rho, rkp, m->N, m->NDIM, dir);
}
void muipkp(void * mptr){
    
    model * m = (model *) mptr;
    float *mu = get_par(m->pars, m->npars, "mu")->gl_par;
    float *muipkp = get_par(m->pars, m->npars, "muipkp")->gl_par;
    int dir[2][3]={{0,0,1},{1,0,0}};
    ave_harmonic(mu, muipkp, m->N, m->NDIM, dir);
}
void mujpkp(void * mptr){
    
    model * m = (model *) mptr;
    float *mu = get_par(m->pars, m->npars, "mu")->gl_par;
    float *mujpkp = get_par(m->pars, m->npars, "mujpkp")->gl_par;
    int dir[2][3]={{0,1,0},{1,0,0}};
    ave_harmonic(mu, mujpkp, m->N, m->NDIM, dir);
}
void muipjp(void * mptr){
    
    model * m = (model *) mptr;
    float *mu = get_par(m->pars, m->npars, "mu")->gl_par;
    float *muipjp = get_par(m->pars, m->npars, "muipjp")->gl_par;
    int dir[2][3]={{0,0,1},{0,1,0}};
    ave_harmonic(mu, muipjp, m->N, m->NDIM, dir);
}
void tausipkp(void * mptr){
    
    model * m = (model *) mptr;
    float *taus = get_par(m->pars, m->npars, "taus")->gl_par;
    float *tausipkp = get_par(m->pars, m->npars, "tausipkp")->gl_par;
    int dir[2][3]={{0,0,1},{1,0,0}};
    ave_arithmetic2(taus, tausipkp, m->N, m->NDIM, dir);
}
void tausjpkp(void * mptr){
    
    model * m = (model *) mptr;
    float *taus = get_par(m->pars, m->npars, "taus")->gl_par;
    float *tausjpkp = get_par(m->pars, m->npars, "tausjpkp")->gl_par;
    int dir[2][3]={{0,1,0},{1,0,0}};
    ave_arithmetic2(taus, tausjpkp, m->N, m->NDIM, dir);
}
void tausipjp(void * mptr){
    
    model * m = (model *) mptr;
    float *taus = get_par(m->pars, m->npars, "taus")->gl_par;
    float *tausipjp = get_par(m->pars, m->npars, "tausipjp")->gl_par;
    int dir[2][3]={{0,0,1},{0,1,0}};
    ave_arithmetic2(taus, tausipjp, m->N, m->NDIM, dir);
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
    m->DTNYQ=ceil(0.0156/fmaxout/m->dt);
    m->NTNYQ=(m->tmax-m->tmin)/m->DTNYQ+1;
    df=1.0/m->NTNYQ/m->dt/m->DTNYQ;
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
    float vmin, vmax;
    float g, gamma, dtstable;
    
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
        //TODO review scaler constant
        scaler = -log2(Mmax*100);
    }

    m->par_scale = scaler;
    variable * var;
    var = get_var(m->vars,m->nvars, "vx");
    if (var) var->scaler = -2*scaler;
    var = get_var(m->vars,m->nvars, "vy");
    if (var) var->scaler = -2*scaler;
    var = get_var(m->vars,m->nvars, "vz");
    if (var) var->scaler = -2*scaler;
    
    return state;
}

//Assign parameters list depending on which case of modeling is desired
int assign_modeling_case(model * m){
    
    int i;
    int state =0;
    int ind = 0;
    
    // Check Stability function
    m->check_stability=&check_stability;
    m->set_par_scale=&set_par_scale;
    
    /* Arrays of constants size on all devices. The most constants we have 
       here is 23 */
    GMALLOC(m->csts, sizeof(constants)*23);
    
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

    /* Definition of each seismic modeling case that has been implemented */
    const char * updatev;
    const char * updates;
    const char * updatev_adj;
    const char * updates_adj;
    const char * surface;
    const char * surface_adj;
    const char * savebnd;
    //TODO surface kernels with FP16, Adjoint kernels in 3D with FP16
    if (m->ND==3 ){
        if (m->FP16==0){
            updatev = update_v3D_source;
            updates = update_s3D_source;
            updatev_adj = update_adjv3D_source;
            updates_adj = update_adjs2D_source;
            surface = surface3D_source;
            savebnd = savebnd3D_source;
        }
        else{
            updatev = update_v3D_half2_source;
            updates = update_s3D_half2_source;
//            updatev_adj = update_adjv3D_half2_source;
//            updates_adj = update_adjs2D_half2_source;
//            surface = surface3D_source;
//            savebnd = savebnd3D_source;
            fprintf(stdout,"Warning: Only forward modeling is implemeted in 3D "
                            "when FP16 is not 0 \n");
        }
    }
    else if (m->ND==2){
        if (m->FP16==0){
            updatev = update_v2D_source;
            updates = update_s2D_source;
            updatev_adj = update_adjv2D_source;
            updates_adj = update_adjs2D_source;
            surface = surface2D_source;
            surface_adj = surface2D_adj_source;
            savebnd = savebnd2D_source;
        }
        else{
            updatev = update_v2D_half2_source;
            updates = update_s2D_half2_source;
            updatev_adj = update_adjv2D_half2_source;
            updates_adj = update_adjs2D_half2_source;
            surface = surface2D_source;
            savebnd = savebnd2D_source;
        }
    }
     else if (m->ND==21){
         if (m->FP16==0){
             updatev = update_v2D_SH_source;
             updates = update_s2D_SH_source;
             updatev_adj = update_adjv2D_SH_source;
             updates_adj = update_adjs2D_SH_source;
             surface = surface2D_SH_source;
             savebnd = savebnd2D_source;
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
            fprintf(stderr,"Error: Only FP16=0 is supported for ND=21 \n");
        }
    }
    m->nupdates=2;
    GMALLOC(m->ups_f, m->nupdates*sizeof(update));
    ind=0;
    __GUARD append_update(m->ups_f, &ind, "update_v", updatev);
    __GUARD append_update(m->ups_f, &ind, "update_s", updates);
    if (m->GRADOUT){
        GMALLOC(m->ups_adj, m->nupdates*sizeof(update));
        ind=0;
        __GUARD append_update(m->ups_adj, &ind, "update_adjv", updatev_adj);
        __GUARD append_update(m->ups_adj, &ind, "update_adjs", updates_adj);
    }
    if (m->FREESURF){
        __GUARD prog_source(&m->bnd_cnds.surf, "surface", surface);
        if (m->GRADOUT){
            __GUARD prog_source(&m->bnd_cnds.surf_adj,
                                "surface_adj", surface_adj);
        }
    }
    if (m->GRADOUT && m->BACK_PROP_TYPE==1){
        __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd);
    }
    
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
        __GUARD append_par(m, &ind, "rip", NULL, &rip);
        if (m->ND==3){
            __GUARD append_par(m, &ind, "rjp", NULL, &rjp);
        }
        __GUARD append_par(m, &ind, "rkp", NULL, &rkp);
    }
    if (m->ND==2 || m->ND==3){
        __GUARD append_par(m, &ind, "muipkp", NULL, &muipkp);
    }
    if (m->ND==3){
        __GUARD append_par(m, &ind, "muipjp", NULL, &muipjp);
        __GUARD append_par(m, &ind, "mujpkp", NULL, &mujpkp);
    }
    if (m->L>0){
        if (m->ND==2 || m->ND==3){
            __GUARD append_par(m, &ind, "tausipkp", NULL, &tausipkp);
        }
        if (m->ND==3){
            __GUARD append_par(m, &ind, "tausipjp", NULL, &tausipjp);
            __GUARD append_par(m, &ind, "tausjpkp", NULL, &tausjpkp);
        }
    }
    m->npars = ind;
    
    m->nvars=15;
    if (m->ABS_TYPE==1)
    m->nvars+=18;
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

    }
    m->nvars = ind;
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

    //Create adjoint variables if necessary
    if (m->GRADOUT && m->BACK_PROP_TYPE==1){
        GMALLOC(m->vars_adj, sizeof(variable)*m->nvars);
        for (i=0;i<m->nvars;i++){
            m->vars_adj[i]=m->vars[i];
        }
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
    if (!state){
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


