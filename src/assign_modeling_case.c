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
#include "surface2D_SH.hcl"
#include "surface3D.hcl"
#include "update_adjs2D.hcl"
#include "update_adjs2D_SH.hcl"
#include "update_adjs3D.hcl"
#include "update_adjv2D.hcl"
#include "update_adjv2D_SH.hcl"
#include "update_adjv3D.hcl"
#include "update_s2D.hcl"
#include "update_s2D_SH.hcl"
#include "update_s3D.hcl"
#include "update_v2D.hcl"
#include "update_v2D_SH.hcl"
#include "update_v3D.hcl"


float * get_par(parameter * pars, int npars, const char * name){
    
    int i;
    float * outptr=NULL;
    for (i=0;i<npars;i++){
        if (strcmp(pars[i].name,name)==0){
            outptr=pars[i].gl_par;
        }
    }
    
    return outptr;
 
}
int get_num_ele(parameter * pars, int npars, const char * name){
    
    int i;
    int output=0;
    for (i=0;i<npars;i++){
        if (strcmp(pars[i].name,name)==0){
            output=pars[i].num_ele;
        }
    }
    
    return output;
    
}
float * get_cst(constants* csts, int ncsts, const char * name){
    
    int i;
    float * outptr=NULL;
    for (i=0;i<ncsts;i++){
        if (strcmp(csts[i].name,name)==0){
            outptr=csts[i].gl_cst;
        }
    }
    
    return outptr;
    
}
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


void mu(void * mptr){
    
    int i,l, state=0;
    model * m = (model *) mptr;
    float *mu = get_par(m->pars, m->npars, "mu");
    int num_ele = get_num_ele(m->pars, m->npars, "mu");
    float *rho = get_par(m->pars, m->npars, "rho");
    float *taus = get_par(m->pars, m->npars, "taus");
    float thisvs, thistaus;
    if (m->par_type==0){
        for (i=0;i<num_ele;i++){
            mu[i]=powf(mu[i],2)*rho[i];
        }
    }
    else if (m->par_type==2){
        for (i=0;i<num_ele;i++){
            mu[i]=powf(mu[i]/rho[i],2)*rho[i];
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
            mu[i]=powf(thisvs,2)*rho[i];
            taus[i]=thistaus;
        }
        
    }
    //Correct the phase velocity for viscoelastic modeling
    if (m->L>0){
        float *pts=NULL;
        /* vector for maxwellbodies */
        GMALLOC(pts,m->L*sizeof(float));
        
        for (l=0;l<m->L;l++) {
            pts[l]=m->dt/m->csts[20].gl_cst[l];
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
    
}
void M(void * mptr){
    
    int i,l, state=0;
    model * m = (model *) mptr;
    float *M = get_par(m->pars, m->npars, "M");
    int num_ele = get_num_ele(m->pars, m->npars, "M");
    float *rho = get_par(m->pars, m->npars, "rho");
    float *taup = get_par(m->pars, m->npars, "taup");
    float thisvp, thistaup;
    if (m->par_type==0){
        for (i=0;i<num_ele;i++){
            M[i]=powf(M[i],2)*rho[i];
        }
    }
    else if (m->par_type==2){
        for (i=0;i<num_ele;i++){
            M[i]=powf(M[i]/rho[i],2)*rho[i];
        }
    }
    else if (m->par_type==3){
        for (i=0;i<num_ele;i++){
            
            thisvp=M[i]-taup[i];
            thistaup=taup[i]/(M[i]-taup[i]);
            M[i]=powf(thisvp,2)*rho[i];
            taup[i]=thistaup;
        }
        
    }
    //Correct the phase velocity for viscoelastic modeling
    if (m->L>0){
        float *pts=NULL;
        /* vector for maxwellbodies */
        GMALLOC(pts,m->L*sizeof(float));
        
        for (l=0;l<m->L;l++) {
            pts[l]=m->dt/m->csts[20].gl_cst[l];
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
    
}
void rip(void * mptr){
    
    model * m = (model *) mptr;
    float *rho = get_par(m->pars, m->npars, "rho");
    float *rip = get_par(m->pars, m->npars, "rip");
    int dir[3]={0,0,1};
    ave_arithmetic1(rho, rip, m->N, m->NDIM, dir);
}
void rjp(void * mptr){
    
    model * m = (model *) mptr;
    float *rho = get_par(m->pars, m->npars, "rho");
    float *rjp = get_par(m->pars, m->npars, "rjp");
    int dir[3]={0,1,0};
    ave_arithmetic1(rho, rjp, m->N, m->NDIM, dir);
}
void rkp(void * mptr){
    
    model * m = (model *) mptr;
    float *rho = get_par(m->pars, m->npars, "rho");
    float *rkp = get_par(m->pars, m->npars, "rkp");
    int dir[3]={1,0,0};
    ave_arithmetic1(rho, rkp, m->N, m->NDIM, dir);
}
void muipkp(void * mptr){
    
    model * m = (model *) mptr;
    float *mu = get_par(m->pars, m->npars, "mu");
    float *muipkp = get_par(m->pars, m->npars, "muipkp");
    int dir[2][3]={{0,0,1},{1,0,0}};
    ave_harmonic(mu, muipkp, m->N, m->NDIM, dir);
}
void mujpkp(void * mptr){
    
    model * m = (model *) mptr;
    float *mu = get_par(m->pars, m->npars, "mu");
    float *mujpkp = get_par(m->pars, m->npars, "mujpkp");
    int dir[2][3]={{0,1,0},{1,0,0}};
    ave_harmonic(mu, mujpkp, m->N, m->NDIM, dir);
}
void muipjp(void * mptr){
    
    model * m = (model *) mptr;
    float *mu = get_par(m->pars, m->npars, "mu");
    float *muipjp = get_par(m->pars, m->npars, "muipjp");
    int dir[2][3]={{0,0,1},{0,1,0}};
    ave_harmonic(mu, muipjp, m->N, m->NDIM, dir);
}
void tausipkp(void * mptr){
    
    model * m = (model *) mptr;
    float *taus = get_par(m->pars, m->npars, "taus");
    float *tausipkp = get_par(m->pars, m->npars, "tausipkp");
    int dir[2][3]={{0,0,1},{1,0,0}};
    ave_arithmetic2(taus, tausipkp, m->N, m->NDIM, dir);
}
void tausjpkp(void * mptr){
    
    model * m = (model *) mptr;
    float *taus = get_par(m->pars, m->npars, "taus");
    float *tausjpkp = get_par(m->pars, m->npars, "tausjpkp");
    int dir[2][3]={{0,1,0},{1,0,0}};
    ave_arithmetic2(taus, tausjpkp, m->N, m->NDIM, dir);
}
void tausipjp(void * mptr){
    
    model * m = (model *) mptr;
    float *taus = get_par(m->pars, m->npars, "taus");
    float *tausipjp = get_par(m->pars, m->npars, "tausipjp");
    int dir[2][3]={{0,0,1},{0,1,0}};
    ave_arithmetic2(taus, tausipjp, m->N, m->NDIM, dir);
}
void eta( void *mptr, void *cstptr, int ncst){
    //Viscoelastic constants initialization
    int l;
    float * eta=get_cst((constants*) cstptr, ncst, "eta");
    float * FL =get_cst((constants*) cstptr, ncst, "FL" );
    model * m = (model *) mptr;
    if (m->L>0){
        for (l=0;l<m->L;l++) {
            eta[l]=(2.0*PI*FL[l])*m->dt;
        }
    }
    
}
void gradfreqsn( void *mptr, void *cstptr, int ncst){
    //Viscoelastic constants initialization
    int j;
    float * gradfreqs=get_cst((constants*) cstptr, ncst, "gradfreqs");
    float * gradfreqsn =get_cst((constants*) cstptr, ncst, "gradfreqsn" );
    model * m = (model *) mptr;
    
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
    float * taper=get_cst((constants*) cstptr, ncst, "taper");
    model * m = (model *) mptr;
    
    float amp=1-m->abpc/100;
    float a=sqrt(-log(amp)/((m->NAB-1)*(m->NAB-1)));
    for (i=1; i<=m->NAB; i++) {
        taper[i-1]=exp(-(a*a*(m->NAB-i)*(m->NAB-i)));
    }
    
}

void CPML_z( void *mptr, void *cstptr, int ncst){
    //Viscoelastic constants initialization
    float * K_i=get_cst((constants*) cstptr, ncst, "K_z");
    float * a_i=get_cst((constants*) cstptr, ncst, "a_z");
    float * b_i=get_cst((constants*) cstptr, ncst, "b_z");
    float * K_i_half=get_cst((constants*) cstptr, ncst, "K_z_half");
    float * a_i_half=get_cst((constants*) cstptr, ncst, "a_z_half");
    float * b_i_half=get_cst((constants*) cstptr, ncst, "b_z_half");
    model * m = (model *) mptr;
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
    float * K_i=get_cst((constants*) cstptr, ncst, "K_x");
    float * a_i=get_cst((constants*) cstptr, ncst, "a_x");
    float * b_i=get_cst((constants*) cstptr, ncst, "b_x");
    float * K_i_half=get_cst((constants*) cstptr, ncst, "K_x_half");
    float * a_i_half=get_cst((constants*) cstptr, ncst, "a_x_half");
    float * b_i_half=get_cst((constants*) cstptr, ncst, "b_x_half");
    model * m = (model *) mptr;
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
    float * K_i=get_cst((constants*) cstptr, ncst, "K_y");
    float * a_i=get_cst((constants*) cstptr, ncst, "a_y");
    float * b_i=get_cst((constants*) cstptr, ncst, "b_y");
    float * K_i_half=get_cst((constants*) cstptr, ncst, "K_y_half");
    float * a_i_half=get_cst((constants*) cstptr, ncst, "a_y_half");
    float * b_i_half=get_cst((constants*) cstptr, ncst, "b_y_half");
    model * m = (model *) mptr;
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
    float *mu = get_par(m->pars, m->npars, "mu");
    int num_ele = get_num_ele(m->pars, m->npars, "mu");
    float *M = get_par(m->pars, m->npars, "M");
    float *rho = get_par(m->pars, m->npars, "rho");
    
    if (mu){
        for (i=0;i<num_ele;i++){
            thisvs=sqrt(mu[i]/rho[i]);
            if (vsmax<thisvs) vsmax=thisvs;
            if (vsmin>thisvs && thisvs>0.1) vsmin=thisvs;
        }
    }
    if (M){
        for (i=0;i<num_ele;i++){
            thisvp=sqrt(M[i]/rho[i]);
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

//Assign parameters list depending on which case of modeling is desired
int assign_modeling_case(model * m){
    
    int i,j;
    int state =0;
    
    int sizepars=1;
    int sizevars=1;
    int sizebnd[10];
    for (i=0;i<m->NDIM;i++){
        sizepars*=m->N[i];
        sizevars*=m->N[i]+m->FDORDER;
    }
    for (i=0;i<m->NDIM;i++){
        sizebnd[i]=1;
        for (j=0;j<m->NDIM;j++){
            if (i!=j)
                sizebnd[i]*=m->N[j];
        }
    }
    
    // Check Stability function
    m->check_stability=&check_stability;
    
    //Arrays of constants size on all devices
    {
        m->ncsts=23;
        GMALLOC(m->csts, sizeof(constants)*m->ncsts);
        m->csts[0].name="taper";   m->csts[0].num_ele=m->NAB; m->csts[0].transform=&taper;
        m->csts[1].name="K_z";      m->csts[1].num_ele=2*m->NAB; m->csts[1].transform=&CPML_z;
        m->csts[2].name="a_z";      m->csts[2].num_ele=2*m->NAB;
        m->csts[3].name="b_z";      m->csts[3].num_ele=2*m->NAB;
        m->csts[4].name="K_z_half"; m->csts[4].num_ele=2*m->NAB;
        m->csts[5].name="a_z_half"; m->csts[5].num_ele=2*m->NAB;
        m->csts[6].name="b_z_half"; m->csts[6].num_ele=2*m->NAB;
        
        m->csts[7].name="K_x";      m->csts[7].num_ele=2*m->NAB; m->csts[7].transform=&CPML_x;
        m->csts[8].name="a_x";      m->csts[8].num_ele=2*m->NAB;
        m->csts[9].name="b_x";      m->csts[9].num_ele=2*m->NAB;
        m->csts[10].name="K_x_half"; m->csts[10].num_ele=2*m->NAB;
        m->csts[11].name="a_x_half"; m->csts[11].num_ele=2*m->NAB;
        m->csts[12].name="b_x_half"; m->csts[12].num_ele=2*m->NAB;
        
        m->csts[13].name="K_y";      m->csts[13].num_ele=2*m->NAB; m->csts[13].transform=&CPML_y;
        m->csts[14].name="a_y";      m->csts[14].num_ele=2*m->NAB;
        m->csts[15].name="b_y";      m->csts[15].num_ele=2*m->NAB;
        m->csts[16].name="K_y_half"; m->csts[16].num_ele=2*m->NAB;
        m->csts[17].name="a_y_half"; m->csts[17].num_ele=2*m->NAB;
        m->csts[18].name="b_y_half"; m->csts[18].num_ele=2*m->NAB;
        m->csts[19].name="FL"; m->csts[19].num_ele=m->L; m->csts[19].to_read="/FL";
        m->csts[20].name="eta"; m->csts[20].num_ele=m->L; m->csts[20].transform=&eta;
        m->csts[21].name="gradfreqs";  m->csts[21].num_ele=m->NFREQS; m->csts[21].to_read="/gradfreqs";
        m->csts[22].name="gradfreqsn"; m->csts[22].num_ele=m->NFREQS;  m->csts[22].transform=&gradfreqsn;
        
        if (m->L>0){
            m->csts[19].active=1;
            m->csts[20].active=1;
        }
        if (m->GRADOUT && m->BACK_PROP_TYPE==2){
            m->csts[21].active=1;
            m->csts[22].active=1;
        }
        if (m->ABS_TYPE==2){
            m->csts[0].active=1;
        }
        else if (m->ABS_TYPE==1){
            for (i=1;i<19;i++){
                m->csts[i].active=1;
            }
            if (m->ND!=3){
                for (i=13;i<19;i++){
                    m->csts[i].active=0;
                }
            }
        }
    }
    
    //Define the update kernels
    m->nupdates=2;
    GMALLOC(m->ups_f, m->nupdates*sizeof(update));
    m->ups_f[0].name="update_v";
    m->ups_f[1].name="update_s";

    
    if (m->GRADOUT){
        GMALLOC(m->ups_adj, m->nupdates*sizeof(update));
        m->ups_adj[0].name="update_adjv";
        m->ups_adj[1].name="update_adjs";
    }



    

    //Define parameters and variables
    {
        if (m->ND==3 && m->L>0){

            __GUARD prog_source(&m->ups_f[0].center, "update_v", update_v3D_source);
            __GUARD prog_source(&m->ups_f[0].com1, "update_v", update_v3D_source);
            __GUARD prog_source(&m->ups_f[0].com2, "update_v", update_v3D_source);
            __GUARD prog_source(&m->ups_f[1].center, "update_s", update_s3D_source);
            __GUARD prog_source(&m->ups_f[1].com1, "update_s", update_s3D_source);
            __GUARD prog_source(&m->ups_f[1].com2, "update_s", update_s3D_source);
            
            
            if (m->GRADOUT){
                __GUARD prog_source(&m->ups_adj[0].center, "update_adjv", update_adjv3D_source);
                __GUARD prog_source(&m->ups_adj[0].com1, "update_adjv", update_adjv3D_source);
                __GUARD prog_source(&m->ups_adj[0].com2, "update_adjv", update_adjv3D_source);
                __GUARD prog_source(&m->ups_adj[1].center, "update_adjs", update_adjs3D_source);
                __GUARD prog_source(&m->ups_adj[1].com1, "update_adjs", update_adjs3D_source);
                __GUARD prog_source(&m->ups_adj[1].com2, "update_adjs", update_adjs3D_source);
            }
            if (m->FREESURF){
                __GUARD prog_source(&m->bnd_cnds.surf, "surface", surface3D_source);
            }
            if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd3D_source);
            }
            
            m->npars=14;
            
            GMALLOC(m->pars, sizeof(parameter)*m->npars);
            if (!state){
            m->pars[0].name="mu";        m->pars[0].to_read="/mu";
            m->pars[1].name="M";         m->pars[1].to_read="/M";
            m->pars[2].name="rho";       m->pars[2].to_read="/rho";
            m->pars[3].name="taup";      m->pars[3].to_read="/taup";
            m->pars[4].name="taus";      m->pars[4].to_read="/taus";
            m->pars[5].name="rip";
            m->pars[6].name="rjp";
            m->pars[7].name="rkp";
            m->pars[8].name="muipjp";
            m->pars[9].name="mujpkp";
            m->pars[10].name="muipkp";
            m->pars[11].name="tausipjp";
            m->pars[12].name="tausjpkp";
            m->pars[13].name="tausipkp";
            }
            
            m->nvars=15;
            if (m->ABS_TYPE==1)
                m->nvars+=18;
            GMALLOC(m->vars, sizeof(variable)*m->nvars);
            
            
            if (!state){
            m->vars[0].name="vx"; m->vars[0].for_grad=1; m->vars[0].to_comm=1;
            m->vars[1].name="vy"; m->vars[1].for_grad=1; m->vars[1].to_comm=1;
            m->vars[2].name="vz"; m->vars[2].for_grad=1; m->vars[2].to_comm=1;
            m->vars[3].name="sxx"; m->vars[3].for_grad=1; m->vars[3].to_comm=2;
            m->vars[4].name="syy"; m->vars[4].for_grad=1; m->vars[4].to_comm=2;
            m->vars[5].name="szz"; m->vars[5].for_grad=1; m->vars[5].to_comm=2;
            m->vars[6].name="sxy"; m->vars[6].for_grad=1; m->vars[6].to_comm=2;
            m->vars[7].name="sxz"; m->vars[7].for_grad=1; m->vars[7].to_comm=2;
            m->vars[8].name="syz"; m->vars[8].for_grad=1; m->vars[8].to_comm=2;
            m->vars[9].name="rxx"; m->vars[9].for_grad=1; 
            m->vars[10].name="ryy"; m->vars[10].for_grad=1; 
            m->vars[11].name="rzz"; m->vars[11].for_grad=1; 
            m->vars[12].name="rxy"; m->vars[12].for_grad=1; 
            m->vars[13].name="rxz"; m->vars[13].for_grad=1; 
            m->vars[14].name="ryz"; m->vars[14].for_grad=1; 
            
            if (m->ABS_TYPE==1){
                m->vars[15].name="psi_sxx_x"; m->vars[15].for_grad=0; 
                m->vars[16].name="psi_sxy_x"; m->vars[16].for_grad=0; 
                m->vars[17].name="psi_sxz_x"; m->vars[17].for_grad=0; 
                m->vars[18].name="psi_syy_y"; m->vars[18].for_grad=0; 
                m->vars[19].name="psi_sxy_y"; m->vars[19].for_grad=0; 
                m->vars[20].name="psi_syz_y"; m->vars[20].for_grad=0; 
                m->vars[21].name="psi_szz_z"; m->vars[21].for_grad=0; 
                m->vars[22].name="psi_sxz_z"; m->vars[22].for_grad=0; 
                m->vars[23].name="psi_syz_z"; m->vars[23].for_grad=0; 
                m->vars[24].name="psi_vx_x"; m->vars[24].for_grad=0; 
                m->vars[25].name="psi_vy_x"; m->vars[25].for_grad=0; 
                m->vars[26].name="psi_vz_x"; m->vars[26].for_grad=0; 
                m->vars[27].name="psi_vx_y"; m->vars[27].for_grad=0; 
                m->vars[28].name="psi_vy_y"; m->vars[28].for_grad=0; 
                m->vars[29].name="psi_vz_y"; m->vars[29].for_grad=0; 
                m->vars[30].name="psi_vx_z"; m->vars[30].for_grad=0; 
                m->vars[31].name="psi_vy_z"; m->vars[31].for_grad=0; 
                m->vars[32].name="psi_vz_z"; m->vars[32].for_grad=0; 
                
            }}
            
           
            
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(variable)*m->ntvars);
            m->trans_vars[0].name="p";
            m->trans_vars[0].n2ave=3;
            GMALLOC(m->trans_vars[0].var2ave, sizeof(char *)*3);
            m->trans_vars[0].var2ave[0]="sxx";
            m->trans_vars[0].var2ave[1]="syy";
            m->trans_vars[0].var2ave[2]="szz";
            
            
        }
        else if (m->ND==3 && m->L==0){
            
            __GUARD prog_source(&m->ups_f[0].center, "update_v", update_v3D_source);
            __GUARD prog_source(&m->ups_f[0].com1, "update_v", update_v3D_source);
            __GUARD prog_source(&m->ups_f[0].com2, "update_v", update_v3D_source);
            __GUARD prog_source(&m->ups_f[1].center, "update_s", update_s3D_source);
            __GUARD prog_source(&m->ups_f[1].com1, "update_s", update_s3D_source);
            __GUARD prog_source(&m->ups_f[1].com2, "update_s", update_s3D_source);
            if (m->GRADOUT){
                __GUARD prog_source(&m->ups_adj[0].center, "update_adjv", update_adjv3D_source);
                __GUARD prog_source(&m->ups_adj[0].com1, "update_adjv", update_adjv3D_source);
                __GUARD prog_source(&m->ups_adj[0].com2, "update_adjv", update_adjv3D_source);
                __GUARD prog_source(&m->ups_adj[1].center, "update_adjs", update_adjs3D_source);
                __GUARD prog_source(&m->ups_adj[1].com1, "update_adjs", update_adjs3D_source);
                __GUARD prog_source(&m->ups_adj[1].com2, "update_adjs", update_adjs3D_source);
            }
            if (m->FREESURF){
                __GUARD prog_source(&m->bnd_cnds.surf, "surface", surface3D_source);
            }
            if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd3D_source);
            }
            
            
            m->npars=9;
            GMALLOC(m->pars, sizeof(parameter)*m->npars);
            if (!state){
            m->pars[0].name="mu";    m->pars[0].to_read="/mu";
            m->pars[1].name="M";     m->pars[1].to_read="/M";
            m->pars[2].name="rho";   m->pars[2].to_read="/rho";
            m->pars[3].name="rip";
            m->pars[4].name="rjp";
            m->pars[5].name="rkp";
            m->pars[6].name="muipjp";
            m->pars[7].name="mujpkp";
            m->pars[8].name="muipkp";
            }
            
            m->nvars=9;
            if (m->ABS_TYPE==1)
                m->nvars+=18;
            GMALLOC(m->vars, sizeof(variable)*m->nvars);
            if (!state){
            m->vars[0].name="vx"; m->vars[0].for_grad=1; m->vars[0].to_comm=1;
            m->vars[1].name="vy"; m->vars[1].for_grad=1; m->vars[1].to_comm=1;
            m->vars[2].name="vz"; m->vars[2].for_grad=1; m->vars[2].to_comm=1;
            m->vars[3].name="sxx"; m->vars[3].for_grad=1; m->vars[3].to_comm=2;
            m->vars[4].name="syy"; m->vars[4].for_grad=1; m->vars[4].to_comm=2;
            m->vars[5].name="szz"; m->vars[5].for_grad=1; m->vars[5].to_comm=2;
            m->vars[6].name="sxy"; m->vars[6].for_grad=1; m->vars[6].to_comm=2;
            m->vars[7].name="sxz"; m->vars[7].for_grad=1; m->vars[7].to_comm=2;
            m->vars[8].name="syz"; m->vars[8].for_grad=1; m->vars[8].to_comm=2;
            
            if (m->ABS_TYPE==1){
                m->vars[9].name="psi_sxx_x"; m->vars[9].for_grad=0; 
                m->vars[10].name="psi_sxy_x"; m->vars[10].for_grad=0; 
                m->vars[11].name="psi_sxz_x"; m->vars[11].for_grad=0; 
                m->vars[12].name="psi_syy_y"; m->vars[12].for_grad=0; 
                m->vars[13].name="psi_sxy_y"; m->vars[13].for_grad=0; 
                m->vars[14].name="psi_syz_y"; m->vars[14].for_grad=0; 
                m->vars[15].name="psi_szz_z"; m->vars[15].for_grad=0; 
                m->vars[16].name="psi_sxz_z"; m->vars[16].for_grad=0; 
                m->vars[17].name="psi_syz_z"; m->vars[17].for_grad=0; 
                m->vars[18].name="psi_vx_x"; m->vars[18].for_grad=0; 
                m->vars[19].name="psi_vy_x"; m->vars[19].for_grad=0; 
                m->vars[20].name="psi_vz_x"; m->vars[20].for_grad=0; 
                m->vars[21].name="psi_vx_y"; m->vars[21].for_grad=0; 
                m->vars[22].name="psi_vy_y"; m->vars[22].for_grad=0; 
                m->vars[23].name="psi_vz_y"; m->vars[23].for_grad=0; 
                m->vars[24].name="psi_vx_z"; m->vars[24].for_grad=0; 
                m->vars[25].name="psi_vy_z"; m->vars[25].for_grad=0; 
                m->vars[26].name="psi_vz_z"; m->vars[26].for_grad=0; 
                
            }}
            
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(variable)*m->ntvars);
            m->trans_vars[0].name="p";
            m->trans_vars[0].n2ave=3;
            GMALLOC(m->trans_vars[0].var2ave, sizeof(char *)*3);
            m->trans_vars[0].var2ave[0]="sxx";
            m->trans_vars[0].var2ave[1]="syy";
            m->trans_vars[0].var2ave[2]="szz";
        }
        else if (m->ND==2 && m->L>0){
            
            __GUARD prog_source(&m->ups_f[0].center, "update_v", update_v2D_source);
            __GUARD prog_source(&m->ups_f[0].com1, "update_v", update_v2D_source);
            __GUARD prog_source(&m->ups_f[0].com2, "update_v", update_v2D_source);
            __GUARD prog_source(&m->ups_f[1].center, "update_s", update_s2D_source);
            __GUARD prog_source(&m->ups_f[1].com1, "update_s", update_s2D_source);
            __GUARD prog_source(&m->ups_f[1].com2, "update_s", update_s2D_source);
            if (m->GRADOUT){
                __GUARD prog_source(&m->ups_adj[0].center, "update_adjv", update_adjv2D_source);
                __GUARD prog_source(&m->ups_adj[0].com1, "update_adjv", update_adjv2D_source);
                __GUARD prog_source(&m->ups_adj[0].com2, "update_adjv", update_adjv2D_source);
                __GUARD prog_source(&m->ups_adj[1].center, "update_adjs", update_adjs2D_source);
                __GUARD prog_source(&m->ups_adj[1].com1, "update_adjs", update_adjs2D_source);
                __GUARD prog_source(&m->ups_adj[1].com2, "update_adjs", update_adjs2D_source);
            }
            if (m->FREESURF){
                __GUARD prog_source(&m->bnd_cnds.surf, "surface", surface2D_source);
            }
            if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd2D_source);
            }
            
            m->npars=9;
            GMALLOC(m->pars, sizeof(parameter)*m->npars);
            if (!state){
            m->pars[0].name="mu";       m->pars[0].to_read="/mu"; m->pars[0].transform=&mu;
            m->pars[1].name="M";        m->pars[1].to_read="/M";  m->pars[1].transform=&M;
            m->pars[2].name="rho";      m->pars[2].to_read="/rho";
            m->pars[3].name="taup";     m->pars[3].to_read="/taup";
            m->pars[4].name="taus";     m->pars[4].to_read="/taus";
            m->pars[5].name="rip";      m->pars[5].transform=&rip;
            m->pars[6].name="rkp";      m->pars[6].transform=&rkp;
            m->pars[7].name="muipkp";   m->pars[7].transform=&muipkp;
            m->pars[8].name="tausipkp"; m->pars[8].transform=&tausipkp;
            }
            
            m->nvars=8;
            if (m->ABS_TYPE==1)
                m->nvars+=8;
            GMALLOC(m->vars, sizeof(variable)*m->nvars);
            if (!state){
            m->vars[0].name="vx"; m->vars[0].for_grad=1; m->vars[0].to_comm=1, m->vars[0].set_size=&size_varseis;
            m->vars[1].name="vz"; m->vars[1].for_grad=1; m->vars[1].to_comm=1; m->vars[1].set_size=&size_varseis;
            m->vars[2].name="sxx"; m->vars[2].for_grad=1; m->vars[2].to_comm=2; m->vars[2].set_size=&size_varseis;
            m->vars[3].name="szz"; m->vars[3].for_grad=1; m->vars[3].to_comm=2; m->vars[3].set_size=&size_varseis;
            m->vars[4].name="sxz"; m->vars[4].for_grad=1; m->vars[4].to_comm=2; m->vars[4].set_size=&size_varseis;
            m->vars[5].name="rxx"; m->vars[5].for_grad=1; m->vars[5].set_size=&size_varmem;
            m->vars[6].name="rzz"; m->vars[6].for_grad=1; m->vars[6].set_size=&size_varmem;
            m->vars[7].name="rxz"; m->vars[7].for_grad=1; m->vars[7].set_size=&size_varmem;
            
            if (m->ABS_TYPE==1){
                m->vars[8].name="psi_sxx_x"; m->vars[8].for_grad=0; m->vars[8].set_size=&size_varcpmlx;
                m->vars[9].name="psi_sxz_x"; m->vars[9].for_grad=0; m->vars[9].set_size=&size_varcpmlx;
                m->vars[10].name="psi_szz_z"; m->vars[10].for_grad=0; m->vars[10].set_size=&size_varcpmlz;
                m->vars[11].name="psi_sxz_z"; m->vars[11].for_grad=0; m->vars[11].set_size=&size_varcpmlz;
                m->vars[12].name="psi_vx_x"; m->vars[12].for_grad=0; m->vars[12].set_size=&size_varcpmlx;
                m->vars[13].name="psi_vz_x"; m->vars[13].for_grad=0; m->vars[13].set_size=&size_varcpmlx;
                m->vars[14].name="psi_vx_z"; m->vars[14].for_grad=0; m->vars[14].set_size=&size_varcpmlz;
                m->vars[15].name="psi_vz_z"; m->vars[15].for_grad=0; m->vars[15].set_size=&size_varcpmlz;
                
            }}
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(variable)*m->ntvars);
            if (!state){
                m->trans_vars[0].name="p";
                m->trans_vars[0].n2ave=2;
                GMALLOC(m->trans_vars[0].var2ave, sizeof(char *)*2);
                m->trans_vars[0].var2ave[0]="sxx";
                m->trans_vars[0].var2ave[1]="szz";
            }
        }
        else if (m->ND==2 && m->L==0){
            
            __GUARD prog_source(&m->ups_f[0].center, "update_v", update_v2D_source);
            __GUARD prog_source(&m->ups_f[0].com1, "update_v", update_v2D_source);
            __GUARD prog_source(&m->ups_f[0].com2, "update_v", update_v2D_source);
            __GUARD prog_source(&m->ups_f[1].center, "update_s", update_s2D_source);
            __GUARD prog_source(&m->ups_f[1].com1, "update_s", update_s2D_source);
            __GUARD prog_source(&m->ups_f[1].com2, "update_s", update_s2D_source);
            if (m->GRADOUT){
                __GUARD prog_source(&m->ups_adj[0].center, "update_adjv", update_adjv2D_source);
                __GUARD prog_source(&m->ups_adj[0].com1, "update_adjv", update_adjv2D_source);
                __GUARD prog_source(&m->ups_adj[0].com2, "update_adjv", update_adjv2D_source);
                __GUARD prog_source(&m->ups_adj[1].center, "update_adjs", update_adjs2D_source);
                __GUARD prog_source(&m->ups_adj[1].com1, "update_adjs", update_adjs2D_source);
                __GUARD prog_source(&m->ups_adj[1].com2, "update_adjs", update_adjs2D_source);
            }
            if (m->FREESURF){
                __GUARD prog_source(&m->bnd_cnds.surf, "surface", surface2D_source);
            }
            if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd2D_source);
            }
            
            m->npars=6;
            
            GMALLOC(m->pars, sizeof(parameter)*m->npars);
            if (!state){
            m->pars[0].name="mu";    m->pars[0].to_read="/mu";
            m->pars[1].name="M";     m->pars[1].to_read="/M";
            m->pars[2].name="rho";   m->pars[2].to_read="/rho";
            m->pars[3].name="rip";
            m->pars[4].name="rkp";
            m->pars[5].name="muipkp";
            }
            
            m->nvars=5;
            if (m->ABS_TYPE==1)
                m->nvars+=8;
            GMALLOC(m->vars, sizeof(variable)*m->nvars);
            if (!state){
            m->vars[0].name="vx"; m->vars[0].for_grad=1; m->vars[0].to_comm=1;
            m->vars[1].name="vz"; m->vars[1].for_grad=1; m->vars[1].to_comm=1;
            m->vars[2].name="sxx"; m->vars[2].for_grad=1; m->vars[2].to_comm=2;
            m->vars[3].name="szz"; m->vars[3].for_grad=1; m->vars[3].to_comm=2;
            m->vars[4].name="sxz"; m->vars[4].for_grad=1; m->vars[4].to_comm=2;
            
            
            if (m->ABS_TYPE==1){
                m->vars[5].name="psi_sxx_x"; m->vars[5].for_grad=0; 
                m->vars[6].name="psi_sxz_x"; m->vars[6].for_grad=0; 
                m->vars[7].name="psi_szz_z"; m->vars[7].for_grad=0; 
                m->vars[8].name="psi_sxz_z"; m->vars[8].for_grad=0; 
                m->vars[9].name="psi_vx_x"; m->vars[9].for_grad=0; 
                m->vars[10].name="psi_vz_x"; m->vars[10].for_grad=0; 
                m->vars[11].name="psi_vx_z"; m->vars[11].for_grad=0; 
                m->vars[12].name="psi_vz_z"; m->vars[12].for_grad=0; 
                
            }}
            
            m->ntvars=1;
            GMALLOC(m->trans_vars, sizeof(variable)*m->ntvars);
            if (!state){
                m->trans_vars[0].name="p";
                m->trans_vars[0].n2ave=2;
                GMALLOC(m->trans_vars[0].var2ave, sizeof(char *)*2);
                m->trans_vars[0].var2ave[0]="sxx";
                m->trans_vars[0].var2ave[1]="szz";
            }
            
        }
        else if (m->ND==21 && m->L>0){
            
            __GUARD prog_source(&m->ups_f[0].center, "update_v", update_v2D_SH_source);
            __GUARD prog_source(&m->ups_f[0].com1, "update_v", update_v2D_SH_source);
            __GUARD prog_source(&m->ups_f[0].com2, "update_v", update_v2D_SH_source);
            __GUARD prog_source(&m->ups_f[1].center, "update_s", update_s2D_SH_source);
            __GUARD prog_source(&m->ups_f[1].com1, "update_s", update_s2D_SH_source);
            __GUARD prog_source(&m->ups_f[1].com2, "update_s", update_s2D_SH_source);
            if (m->GRADOUT){
                __GUARD prog_source(&m->ups_adj[0].center, "update_adjv", update_adjv2D_SH_source);
                __GUARD prog_source(&m->ups_adj[0].com1, "update_adjv", update_adjv2D_SH_source);
                __GUARD prog_source(&m->ups_adj[0].com2, "update_adjv", update_adjv2D_SH_source);
                __GUARD prog_source(&m->ups_adj[1].center, "update_adjs", update_adjs2D_SH_source);
                __GUARD prog_source(&m->ups_adj[1].com1, "update_adjs", update_adjs2D_SH_source);
                __GUARD prog_source(&m->ups_adj[1].com2, "update_adjs", update_adjs2D_SH_source);
            }
            if (m->FREESURF){
                __GUARD prog_source(&m->bnd_cnds.surf, "surface", surface2D_source);
            }
            if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd2D_source);
            }
            
            m->npars=7;
            
            GMALLOC(m->pars, sizeof(parameter)*m->npars);
            if (!state){
            m->pars[0].name="mu";        m->pars[0].to_read="/mu";
            m->pars[1].name="rho";      m->pars[1].to_read="/rho";
            m->pars[2].name="taus";     m->pars[2].to_read="/taus";
            m->pars[3].name="rip";
            m->pars[4].name="rkp";
            m->pars[5].name="muipkp";
            m->pars[6].name="tausipkp";
            }
            
            m->nvars=5;
            if (m->ABS_TYPE==1)
                m->vars+=4;
            GMALLOC(m->vars, sizeof(variable)*m->nvars);
            if (!state){
            m->vars[0].name="vy"; m->vars[0].for_grad=1; m->vars[0].to_comm=1;
            m->vars[1].name="sxy"; m->vars[1].for_grad=1; m->vars[1].to_comm=2;
            m->vars[2].name="syz"; m->vars[2].for_grad=1; m->vars[2].to_comm=2;
            m->vars[3].name="rxy"; m->vars[3].for_grad=1; 
            m->vars[4].name="ryz"; m->vars[4].for_grad=1; 
            
            if (m->ABS_TYPE==1){
                m->vars[5].name="psi_sxy_x"; m->vars[5].for_grad=0; 
                m->vars[6].name="psi_sxy_z"; m->vars[6].for_grad=0; 
                m->vars[7].name="psi_vy_x"; m->vars[7].for_grad=0; 
                m->vars[8].name="psi_vy_z"; m->vars[8].for_grad=0; 
                
            }}
            
            
        }
        else if (m->ND==21 && m->L==0){
            
            __GUARD prog_source(&m->ups_f[0].center, "update_v", update_v2D_SH_source);
            __GUARD prog_source(&m->ups_f[0].com1, "update_v", update_v2D_SH_source);
            __GUARD prog_source(&m->ups_f[0].com2, "update_v", update_v2D_SH_source);
            __GUARD prog_source(&m->ups_f[1].center, "update_s", update_s2D_SH_source);
            __GUARD prog_source(&m->ups_f[1].com1, "update_s", update_s2D_SH_source);
            __GUARD prog_source(&m->ups_f[1].com2, "update_s", update_s2D_SH_source);

            if (m->GRADOUT){
                __GUARD prog_source(&m->ups_adj[0].center, "update_adjv", update_adjv2D_SH_source);
                __GUARD prog_source(&m->ups_adj[0].com1, "update_adjv", update_adjv2D_SH_source);
                __GUARD prog_source(&m->ups_adj[0].com2, "update_adjv", update_adjv2D_SH_source);
                __GUARD prog_source(&m->ups_adj[1].center, "update_adjs", update_adjs2D_SH_source);
                __GUARD prog_source(&m->ups_adj[1].com1, "update_adjs", update_adjs2D_SH_source);
                __GUARD prog_source(&m->ups_adj[1].com2, "update_adjs", update_adjs2D_SH_source);

            }
            if (m->FREESURF){
                __GUARD prog_source(&m->bnd_cnds.surf, "surface", surface2D_source);
            }
            if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                __GUARD prog_source(&m->grads.savebnd, "savebnd", savebnd2D_source);
            }
            
            m->npars=5;
            
            GMALLOC(m->pars, sizeof(parameter)*m->npars);
            if (!state){
            m->pars[0].name="mu";     m->pars[0].to_read="/mu";
            m->pars[1].name="rho";   m->pars[1].to_read="/rho";
            m->pars[2].name="rip";
            m->pars[3].name="rkp";
            m->pars[4].name="muipkp";
            }
            
            m->nvars=3;
            if (m->ABS_TYPE==1)
                m->nvars+=4;
            GMALLOC(m->vars, sizeof(variable)*m->nvars);
            if (!state){
            m->vars[0].name="vy"; m->vars[0].for_grad=1; m->vars[0].to_comm=1;
            m->vars[1].name="sxy"; m->vars[1].for_grad=1; m->vars[1].to_comm=2;
            m->vars[2].name="syz"; m->vars[2].for_grad=1; m->vars[2].to_comm=2;
            
            if (m->ABS_TYPE==1){
                m->vars[3].name="psi_sxy_x"; m->vars[3].for_grad=0; 
                m->vars[4].name="psi_sxy_z"; m->vars[4].for_grad=0;     
                m->vars[5].name="psi_vy_x"; m->vars[5].for_grad=0;      
                m->vars[6].name="psi_vy_z"; m->vars[6].for_grad=0;      
                
            }}
                
            
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

    //Assign the number of elements of the parameters
    if (!state){
    for (i=0;i<m->npars;i++){
        m->pars[i].num_ele=sizepars;
    }}
    if (!state){
        for (i=0;i<m->nvars;i++){
            m->vars[i].set_size(m->N, (void*) m, &m->vars[i]);
    }}

    
    //Check the name of variable to be read depending on the chosen paretrization
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
            if (m->L==0) {state=1;fprintf(stderr, "Viscoelastic modeling is required for par_type 3\n");};
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
    
    //Flags variables to output
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
            for (i=0;i<m->ntvars;i++){
                if (strcmp(m->trans_vars[i].name,"p")==0)
                    m->trans_vars[i].to_output=1;
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
                if (strcmp(m->vars[i].name,"sxx")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"syy")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"szz")==0)
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
                if (strcmp(m->vars[i].name,"sxx")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"syy")==0)
                    m->vars[i].to_output=1;
                if (strcmp(m->vars[i].name,"szz")==0)
                    m->vars[i].to_output=1;
            }
            for (i=0;i<m->ntvars;i++){
                if (strcmp(m->trans_vars[i].name,"p")==0)
                    m->trans_vars[i].to_output=1;
            }
        }
    }

    //Allocate memory of constants
    for (i=0;i<m->ncsts;i++){
        if (m->csts[i].active)
            GMALLOC(m->csts[i].gl_cst, sizeof(float)*m->csts[i].num_ele);
    }
    
    //Allocate memory of parameters
    for (i=0;i<m->npars;i++){
        GMALLOC(m->pars[i].gl_par, sizeof(float)*m->pars[i].num_ele);
        if (m->GRADOUT && m->pars[i].to_read){
            m->pars[i].to_grad=1;
            GMALLOC(m->pars[i].gl_grad, sizeof(double)*m->pars[i].num_ele);
            if (m->HOUT){
                GMALLOC(m->pars[i].gl_H, sizeof(double)*m->pars[i].num_ele);
            }
        }
    }
    
    //Allocate memory of variables
    for (i=0;i<m->nvars;i++){
        if (m->vars[i].to_output){
            var_alloc_out(&m->vars[i].gl_varout, m);
            if (m->MOVOUT>0){
                GMALLOC(m->vars[i].gl_mov,sizeof(float)*
                             m->src_recs.ns*m->vars[i].num_ele*m->NT/m->MOVOUT);
            }
        }
    }
    for (i=0;i<m->ntvars;i++){
        if (m->trans_vars[i].to_output){
            var_alloc_out(&m->trans_vars[i].gl_varout, m);
        }
    }
    
    if (m->GRADSRCOUT==1){
        GMALLOC(m->src_recs.gradsrc,sizeof(float*)*m->src_recs.ns);
        GMALLOC(m->src_recs.gradsrc[0],sizeof(float)*m->src_recs.allns*m->NT);
        for (i=1;i<m->src_recs.ns;i++){
            m->src_recs.gradsrc[i]=m->src_recs.gradsrc[i-1]
                                  +m->src_recs.nsrc[i-1]*m->NT;
        }
    }
    
    

    
    return state;
}


