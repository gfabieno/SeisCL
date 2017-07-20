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

#include "F.h"

#define rho(z,y,x) rho[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define rip(z,y,x) rip[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define rjp(z,y,x) rjp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define rkp(z,y,x) rkp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define uipjp(z,y,x) uipjp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define ujpkp(z,y,x) ujpkp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define uipkp(z,y,x) uipkp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define u(z,y,x) u[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define pi(z,y,x) pi[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define grad(z,y,x) grad[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define grads(z,y,x) grads[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define amp1(z,y,x) amp1[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define amp2(z,y,x) amp2[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]

#define vxout(y,x) vxout[(y)*m->NT+(x)]
#define vyout(y,x) vyout[(y)*m->NT+(x)]
#define vzout(y,x) vzout[(y)*m->NT+(x)]
#define vx0(y,x) vx0[(y)*m->NT+(x)]
#define vy0(y,x) vy0[(y)*m->NT+(x)]
#define vz0(y,x) vz0[(y)*m->NT+(x)]
#define rx(y,x) rx[(y)*m->NT+(x)]
#define ry(y,x) ry[(y)*m->NT+(x)]
#define rz(y,x) rz[(y)*m->NT+(x)]

#define vxcum(y,x) vxcum[(y)*m->NT+(x)]
#define vycum(y,x) vycum[(y)*m->NT+(x)]

#define u_in(z,y,x) u_in[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define pi_in(z,y,x) pi_in[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define uL(z,y,x) uL[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define piL(z,y,x) piL[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define taus(z,y,x) taus[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define tausipjp(z,y,x) tausipjp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define tausjpkp(z,y,x) tausjpkp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define tausipkp(z,y,x) tausipkp[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]
#define taup(z,y,x) taup[(x)*m->NY*m->NZ+(y)*m->NZ+(z)]


#define PI (3.141592653589793238462643383279502884197169)



int Init_model(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc) {

    int state=0;
    int i,j,k,l;
    float ws=0,sumu=0,sumpi=0, thisvp=0, thisvs=0, thistaup=0, thistaus=0;
    float *pts=NULL;
    float vpmax, vpmin, vsmin, vsmax, vmin, vmax;
    float gamma=0, g=0, dtstable;
    
    
    //Transform variables into modulus
    if (!state){
        
        if (m->ND!=21){
            if (m->param_type==0){
                for (i=0;i<m->NX*m->NY*m->NZ;i++){
                    m->u[i]=powf(m->u[i],2)*m->rho[i];
                    m->pi[i]=powf(m->pi[i],2)*m->rho[i];
                }
            }
            if (m->param_type==2){
                for (i=0;i<m->NX*m->NY*m->NZ;i++){
                    m->u[i]=powf(m->u[i]/m->rho[i],2)*m->rho[i];
                    m->pi[i]=powf(m->pi[i]/m->rho[i],2)*m->rho[i];
                }
            }
            if (m->param_type==3){
                for (i=0;i<m->NX*m->NY*m->NZ;i++){
                    
                    thisvp=m->pi[i]-m->taup[i];
                    thistaup=m->taup[i]/(m->pi[i]-m->taup[i]);
                    if (m->u[i]>0){
                        thisvs=m->u[i]-m->taus[i];
                        thistaus=m->taus[i]/(m->u[i]-m->taus[i]);
                    }
                    else{
                        thistaus=0;
                        thisvs=0;
                    }
                    m->u[i]=powf(thisvs,2)*m->rho[i];
                    m->pi[i]=powf(thisvp,2)*m->rho[i];
                    m->taup[i]=thistaup;
                    m->taus[i]=thistaus;
                }
                
            }
        }
        else {
            if (m->param_type==0){
                for (i=0;i<m->NX*m->NY*m->NZ;i++){
                    m->u[i]=powf(m->u[i],2)*m->rho[i];
                }
            }
            if (m->param_type==2){
                for (i=0;i<m->NX*m->NY*m->NZ;i++){
                    m->u[i]=powf(m->u[i]/m->rho[i],2)*m->rho[i];
                }
            }
            if (m->param_type==3){
                for (i=0;i<m->NX*m->NY*m->NZ;i++){
                    if (m->u[i]>0){
                        thisvs=m->u[i]-m->taus[i];
                        thistaus=m->taus[i]/(m->u[i]-m->taus[i]);
                    }
                    else{
                        thistaus=0;
                        thisvs=0;
                    }
                    m->u[i]=powf(thisvs,2)*m->rho[i];
                    m->taus[i]=thistaus;
                }
                
            }
        }
        
    }
    
    if (m->L>0){

        /* vector for maxwellbodies */
        if (!state) if (!(pts =malloc(m->L*sizeof(float))))              {state=1; fprintf(stderr,"could not allocate eta\n");};

        for (l=0;l<m->L;l++) {
            pts[l]=1.0/(2.0*PI*m->FL[l]);
            m->eta[l]=m->dt/pts[l];
        }
        
        ws=2.0*PI*m->f0;
        sumu=0.0;
        sumpi=0.0;

        /* loop over global grid */
        if (m->pi){
            for (k=0;k<m->NZ;k++){
                for (j=0;j<m->NY;j++){
                    for (i=0;i<m->NX;i++){
                        sumpi=0.0;
                        for (l=0;l<m->L;l++){
                            sumpi+= ((ws*ws*pts[l]*pts[l]*m->taup(k,j,i))/(1.0+ws*ws*pts[l]*pts[l]));
                        }
                        m->pi(k,j,i)=m->pi(k,j,i)/(1.0+sumpi);
                    }
                }
            }
        }
        if (m->u){
            for (k=0;k<m->NZ;k++){
                for (j=0;j<m->NY;j++){
                    for (i=0;i<m->NX;i++){
                        sumu=0.0;
                        for (l=0;l<m->L;l++){
                            sumu+=  ((ws*ws*pts[l]*pts[l]*m->taus(k,j,i))/(1.0+ws*ws*pts[l]*pts[l]));
                        }
                        m->u(k,j,i)=m->u(k,j,i)/(1.0+sumu);
                    }
                }
            }
        }
        
        free(pts);
        
    }
    
    /* Check stability and dispersion */
    vpmax=0;
    vsmax=0;
    vpmin=99999;
    vsmin=99999;
    for (i=0;i<m->NX*m->NY*m->NZ;i++){
        
        thisvp=sqrt(m->pi[i]/m->rho[i]);
        thisvs=sqrt(m->u[i]/m->rho[i]);
        
        if (vpmax<thisvp) vpmax=thisvp;
        if (vsmax<thisvs) vsmax=thisvs;
        if (vsmin>thisvs && thisvs>0.1) vsmin=thisvs;
        if (vpmin>thisvp && thisvp>0.1) vpmin=thisvp;
        
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
    if ((m->dh>(vmin/m->fmax/g)))
        fprintf(stdout,"Warning: Grid spacing too large, dispersion will affect the solution\n");
    
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
        fprintf(stderr, "Error: Time step too large, to be stable, set dt<%f\n", dtstable);
    }
    
    /* harmonic averaging of shear modulus */
    if (m->uipjp){
        for (k=0;k<m->NZ;k++){
            for (j=0;j<m->NY;j++){
                for (i=0;i<m->NX;i++){
                    if (i<m->NX-1 && j<m->NY-1)
                        m->uipjp(k,j,i)=4.0/((1.0/m->u(k,j,i))+(1.0/m->u(k,j,i+1))+(1.0/m->u(k,j+1,i+1))+(1.0/m->u(k,j+1,i)));
                    else
                        m->uipjp(k,j,i)=m->u(k,j,i);
                }
            }
        }
    }
    if (m->ujpkp){
        for (k=0;k<m->NZ;k++){
            for (j=0;j<m->NY;j++){
                for (i=0;i<m->NX;i++){
                    if (k<m->NZ-1 && j<m->NY-1)
                        m->ujpkp(k,j,i)=4.0/((1.0/m->u(k,j,i))+(1.0/m->u(k+1,j,i))+(1.0/m->u(k+1,j+1,i))+(1.0/m->u(k,j+1,i)));
                    else
                        m->ujpkp(k,j,i)=m->u(k,j,i);
                }
            }
        }
    }
    if (m->uipkp){
        for (k=0;k<m->NZ;k++){
            for (j=0;j<m->NY;j++){
                for (i=0;i<m->NX;i++){
                    if (k<m->NZ-1 && i<m->NX-1)
                        m->uipkp(k,j,i)=4.0/((1.0/m->u(k,j,i))+(1.0/m->u(k+1,j,i))+(1.0/m->u(k+1,j,i+1))+(1.0/m->u(k,j,i+1)));
                    else
                        m->uipkp(k,j,i)=m->u(k,j,i);
                     }
            }
        }
    }
    /* arithmetic averaging of TAU for S-waves and density */
    if (m->tausipjp){
        for (k=0;k<m->NZ;k++){
            for (j=0;j<m->NY;j++){
                for (i=0;i<m->NX;i++){
                    if (i<m->NX-1 && j<m->NY-1)
                        m->tausipjp(k,j,i)=0.25*(m->taus(k,j,i)+m->taus(k,j,i+1)+m->taus(k,j+1,i+1)+m->taus(k,j+1,i));
                    else
                        m->tausipjp(k,j,i)=m->taus(k,j,i);                }
            }
        }
    }

    if (m->tausjpkp){
        for (k=0;k<m->NZ;k++){
            for (j=0;j<m->NY;j++){
                for (i=0;i<m->NX;i++){
                    if (k<m->NZ-1 && j<m->NY-1)
                        m->tausjpkp(k,j,i)=0.25*(m->taus(k,j,i)+m->taus(k,j+1,i)+m->taus(k+1,j+1,i)+m->taus(k+1,j,i));
                    else
                        m->tausjpkp(k,j,i)=m->taus(k,j,i);
                }
            }
        }
    }
    
    if (m->tausipkp){
        for (k=0;k<m->NZ;k++){
            for (j=0;j<m->NY;j++){
                for (i=0;i<m->NX;i++){
                    if (k<m->NZ-1 && i<m->NX-1)
                        m->tausipkp(k,j,i)=0.25*(m->taus(k,j,i)+m->taus(k,j,i+1)+m->taus(k+1,j,i+1)+m->taus(k+1,j,i));
                    else
                        m->tausipkp(k,j,i)=m->taus(k,j,i);

                }
            }
        }
    }
    if (m->rjp){
        for (k=0;k<m->NZ;k++){
            for (j=0;j<m->NY;j++){
                for (i=0;i<m->NX;i++){
                    if (j<m->NY-1)
                        m->rjp(k,j,i)=0.5*(m->rho(k,j,i)+m->rho(k,j+1,i));
                    else
                        m->rjp(k,j,i)=m->rho(k,j,i);
                }
            }
        }
    }
    if (m->rip){
        for (k=0;k<m->NZ;k++){
            for (j=0;j<m->NY;j++){
                for (i=0;i<m->NX;i++){
                    if (i<m->NX-1)
                        m->rip(k,j,i)=0.5*(m->rho(k,j,i)+m->rho(k,j,i+1));
                    else
                        m->rip(k,j,i)=m->rho(k,j,i);
                }
            }
        }
    }

    if (m->rkp){
        for (k=0;k<m->NZ;k++){
            for (j=0;j<m->NY;j++){
                for (i=0;i<m->NX;i++){
                    if (k<m->NZ-1)
                        m->rkp(k,j,i)=0.5*(m->rho(k,j,i)+m->rho(k+1,j,i));
                    else
                        m->rkp(k,j,i)=m->rho(k,j,i);
                }
            }
        }
    }

 

    //Initialize the gradient
    if (m->gradout==1 ){

        if (m->gradrho) memset (m->gradrho, 0, m->NX*m->NY*m->NZ*sizeof(double));
        if (m->gradM) memset (m->gradM, 0, m->NX*m->NY*m->NZ*sizeof(double));
        if (m->gradmu) memset (m->gradmu, 0, m->NX*m->NY*m->NZ*sizeof(double));
        if (m->gradtaup) memset (m->gradtaup, 0, m->NX*m->NY*m->NZ*sizeof(double));
        if (m->gradtaus) memset (m->gradtaus, 0, m->NX*m->NY*m->NZ*sizeof(double));
        
        if (m->back_prop_type==2){
            float fmaxout=0;
            for (j=0;j<m->nfreqs;j++){
                if (m->gradfreqs[j]>fmaxout)
                    fmaxout=m->gradfreqs[j];
            }
            float df;
            m->dtnyq=ceil(0.0156/fmaxout/m->dt);
            m->NTnyq=(m->tmax-m->tmin)/m->dtnyq+1;
            df=1.0/m->NTnyq/m->dt/m->dtnyq;
            for (j=0;j<m->nfreqs;j++){
                m->gradfreqsn[j]=floor(m->gradfreqs[j]/df);
            }
        }
        
        //Initialize the amplitude output (approximate Hessian)
        if (m->Hout==1 ){
            
            if (m->Hrho) memset (m->Hrho, 0, m->NX*m->NY*m->NZ*sizeof(double));
            if (m->HM) memset (m->HM, 0, m->NX*m->NY*m->NZ*sizeof(double));
            if (m->Hmu) memset (m->Hmu, 0, m->NX*m->NY*m->NZ*sizeof(double));
            if (m->Htaup) memset (m->Htaup, 0, m->NX*m->NY*m->NZ*sizeof(double));
            if (m->Htaus) memset (m->Htaus, 0, m->NX*m->NY*m->NZ*sizeof(double));
            
        }
        
    }
    


    if (state && m->MPI_INIT==1) MPI_Bcast( &state, 1, MPI_INT, m->MYID, MPI_COMM_WORLD );
    
    return state;
    
    
    
}
