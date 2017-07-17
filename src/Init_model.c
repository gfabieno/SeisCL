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

#define rho(z,y,x) rho[(x)*NY*NZ+(y)*NZ+(z)]
#define rip(z,y,x) rip[(x)*NY*NZ+(y)*NZ+(z)]
#define rjp(z,y,x) rjp[(x)*NY*NZ+(y)*NZ+(z)]
#define rkp(z,y,x) rkp[(x)*NY*NZ+(y)*NZ+(z)]
#define muipjp(z,y,x) muipjp[(x)*NY*NZ+(y)*NZ+(z)]
#define mujpkp(z,y,x) mujpkp[(x)*NY*NZ+(y)*NZ+(z)]
#define muipkp(z,y,x) muipkp[(x)*NY*NZ+(y)*NZ+(z)]
#define mu(z,y,x) mu[(x)*NY*NZ+(y)*NZ+(z)]
#define pi(z,y,x) pi[(x)*NY*NZ+(y)*NZ+(z)]
#define taus(z,y,x) taus[(x)*NY*NZ+(y)*NZ+(z)]
#define tausipjp(z,y,x) tausipjp[(x)*NY*NZ+(y)*NZ+(z)]
#define tausjpkp(z,y,x) tausjpkp[(x)*NY*NZ+(y)*NZ+(z)]
#define tausipkp(z,y,x) tausipkp[(x)*NY*NZ+(y)*NZ+(z)]
#define taup(z,y,x) taup[(x)*NY*NZ+(y)*NZ+(z)]



int Init_model(struct modcsts * m) {

    int state=0;
    int i,j,k,l;
    int NX, NY, NZ;
    float ws=0,sumu=0,sumpi=0, thisvp=0, thisvs=0, thistaup=0, thistaus=0;
    float *pts=NULL;
    float vpmax, vpmin, vsmin, vsmax, vmin, vmax;
    float gamma=0, g=0, dtstable;
    
    
    float * mu=NULL;
    float * M=NULL;
    float * rho=NULL;
    float * taup=NULL;
    float * taus=NULL;
    
    float * rip=NULL;
    float * rjp=NULL;
    float * rkp=NULL;
    
    float * muipjp=NULL;
    float * mujpkp=NULL;
    float * muipkp=NULL;
    
    float * tausipjp=NULL;
    float * tausjpkp=NULL;
    float * tausipkp=NULL;
    
    int num_ele=0;

    for (i=0;i<m->npars;i++){
        if (strcmp(m->pars[i].name,"rho")==0){
            rho= m->pars[i].gl_par;
            num_ele=m->pars[i].num_ele;
        }
        else if (strcmp(m->pars[i].name,"M")==0){
            M= m->pars[i].gl_par;            
        }
        else if (strcmp(m->pars[i].name,"mu")==0){
            mu= m->pars[i].gl_par;
        }
        else if (strcmp(m->pars[i].name,"taup")==0){
            taup= m->pars[i].gl_par;
        }
        else if (strcmp(m->pars[i].name,"taus")==0){
            taus= m->pars[i].gl_par;
        }
        else if (strcmp(m->pars[i].name,"rip")==0){
            rip= m->pars[i].gl_par;
        }
        else if (strcmp(m->pars[i].name,"rjp")==0){
            rjp= m->pars[i].gl_par;
        }
        else if (strcmp(m->pars[i].name,"rkp")==0){
            rkp= m->pars[i].gl_par;
        }
        else if (strcmp(m->pars[i].name,"muipjp")==0){
            muipjp= m->pars[i].gl_par;
        }
        else if (strcmp(m->pars[i].name,"mujpkp")==0){
            mujpkp= m->pars[i].gl_par;
        }
        else if (strcmp(m->pars[i].name,"muipkp")==0){
            muipkp= m->pars[i].gl_par;
        }
        else if (strcmp(m->pars[i].name,"tausipjp")==0){
            tausipjp= m->pars[i].gl_par;
        }
        else if (strcmp(m->pars[i].name,"tausjpkp")==0){
            tausjpkp= m->pars[i].gl_par;
        }
        else if (strcmp(m->pars[i].name,"tausipkp")==0){
            tausipkp= m->pars[i].gl_par;
        }
    }
    
    //Transform variables into modulus
    if (!state){
        if (m->par_type==0){
            if (M){
                for (i=0;i<num_ele;i++){
                    M[i]=powf(M[i],2)*rho[i];
                }
            }
            if (mu){
                for (i=0;i<num_ele;i++){
                    mu[i]=powf(mu[i],2)*rho[i];
                }
            }
        }
        else if (m->par_type==2){
            if (M){
                for (i=0;i<num_ele;i++){
                    M[i]=powf(M[i]/rho[i],2)*rho[i];
                }
            }
            if (mu){
                for (i=0;i<num_ele;i++){
                    mu[i]=powf(mu[i]/rho[i],2)*rho[i];
                }
            }
        }
        if (m->par_type==3){
            if (M && mu && taup && taus){
                for (i=0;i<num_ele;i++){
                    
                    thisvp=M[i]-taup[i];
                    thistaup=taup[i]/(M[i]-taup[i]);
                    if (mu[i]>0){
                        thisvs=mu[i]-taus[i];
                        thistaus=taus[i]/(mu[i]-taus[i]);
                    }
                    else{
                        thistaus=0;
                        thisvs=0;
                    }
                    mu[i]=powf(thisvs,2)*rho[i];
                    M[i]=powf(thisvp,2)*rho[i];
                    taup[i]=thistaup;
                    taus[i]=thistaus;
                }
            }
            
        }
    }

    //Correct the phase velocity for viscoelastic modeling
    if (m->L>0){

        /* vector for maxwellbodies */
        GMALLOC(pts,m->L*sizeof(float));

        for (l=0;l<m->L;l++) {
            pts[l]=m->dt/m->csts[20].gl_cst[l];
        }
        
        ws=2.0*PI*m->f0;
        sumu=0.0;
        sumpi=0.0;

        if (M){
            for (i=0;i<num_ele;i++){
                sumpi=0.0;
                for (l=0;l<m->L;l++){
                    sumpi+= ((ws*ws*pts[l]*pts[l]*taup[i])
                               /(1.0+ws*ws*pts[l]*pts[l]));
                }
                M[i]=M[i]/(1.0+sumpi);
            }
        }
        if (mu){
            for (i=0;i<num_ele;i++){
                sumu=0.0;
                for (l=0;l<m->L;l++){
                    sumu+= ((ws*ws*pts[l]*pts[l]*taus[i])
                              /(1.0+ws*ws*pts[l]*pts[l]));
                }
                mu[i]=mu[i]/(1.0+sumu);
            }
        }

        free(pts);
    }
    
    /* Check stability and dispersion */
    vpmax=0;
    vsmax=0;
    vpmin=99999;
    vsmin=99999;
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
    
    //Create averaged properties
    if (m->NDIM==3){
        NX=m->N[2];
        NY=m->N[1];
        NZ=m->N[0];
    }
    else
    {
        NX=m->N[1];
        NY=0;
        NZ=m->N[0];
    }
    
    /* harmonic averaging of shear modulus */
    if (muipjp){
        for (k=0;k<NZ;k++){
            for (j=0;j<NY;j++){
                for (i=0;i<NX;i++){
                    if (i<NX-1 && j<NY-1)
                        muipjp(k,j,i)=4.0/( (1.0/mu(k,j,i))
                                           +(1.0/mu(k,j,i+1))
                                           +(1.0/mu(k,j+1,i+1))
                                           +(1.0/mu(k,j+1,i)));
                    else
                        muipjp(k,j,i)=mu(k,j,i);
                }
            }
        }
    }
    if (mujpkp){
        for (k=0;k<NZ;k++){
            for (j=0;j<NY;j++){
                for (i=0;i<NX;i++){
                    if (k<NZ-1 && j<NY-1)
                        mujpkp(k,j,i)=4.0/( (1.0/mu(k,j,i))
                                           +(1.0/mu(k+1,j,i))
                                           +(1.0/mu(k+1,j+1,i))
                                           +(1.0/mu(k,j+1,i)));
                    else
                        mujpkp(k,j,i)=mu(k,j,i);
                }
            }
        }
    }
    if (muipkp){
        for (k=0;k<NZ;k++){
            for (j=0;j<NY;j++){
                for (i=0;i<NX;i++){
                    if (k<NZ-1 && i<NX-1)
                        muipkp(k,j,i)=4.0/( (1.0/mu(k,j,i))
                                           +(1.0/mu(k+1,j,i))
                                           +(1.0/mu(k+1,j,i+1))
                                           +(1.0/mu(k,j,i+1)));
                    else
                        muipkp(k,j,i)=mu(k,j,i);
                     }
            }
        }
    }
    /* arithmetic averaging of TAU for S-waves and density */
    if (tausipjp){
        for (k=0;k<NZ;k++){
            for (j=0;j<NY;j++){
                for (i=0;i<NX;i++){
                    if (i<NX-1 && j<NY-1)
                        tausipjp(k,j,i)=0.25*( taus(k,j,i)
                                              +taus(k,j,i+1)
                                              +taus(k,j+1,i+1)
                                              +taus(k,j+1,i));
                    else
                        tausipjp(k,j,i)=taus(k,j,i);
                }
            }
        }
    }

    if (tausjpkp){
        for (k=0;k<NZ;k++){
            for (j=0;j<NY;j++){
                for (i=0;i<NX;i++){
                    if (k<NZ-1 && j<NY-1)
                        tausjpkp(k,j,i)=0.25*( taus(k,j,i)
                                              +taus(k,j+1,i)
                                              +taus(k+1,j+1,i)
                                              +taus(k+1,j,i));
                    else
                        tausjpkp(k,j,i)=taus(k,j,i);
                }
            }
        }
    }
    
    if (tausipkp){
        for (k=0;k<NZ;k++){
            for (j=0;j<NY;j++){
                for (i=0;i<NX;i++){
                    if (k<NZ-1 && i<NX-1)
                        tausipkp(k,j,i)=0.25*( taus(k,j,i)
                                              +taus(k,j,i+1)
                                              +taus(k+1,j,i+1)
                                              +taus(k+1,j,i));
                    else
                        tausipkp(k,j,i)=taus(k,j,i);

                }
            }
        }
    }
    if (rjp){
        for (k=0;k<NZ;k++){
            for (j=0;j<NY;j++){
                for (i=0;i<NX;i++){
                    if (j<NY-1)
                        rjp(k,j,i)=0.5*(rho(k,j,i)+rho(k,j+1,i));
                    else
                        rjp(k,j,i)=rho(k,j,i);
                }
            }
        }
    }
    if (rip){
        for (k=0;k<NZ;k++){
            for (j=0;j<NY;j++){
                for (i=0;i<NX;i++){
                    if (i<NX-1)
                        rip(k,j,i)=0.5*(rho(k,j,i)+rho(k,j,i+1));
                    else
                        rip(k,j,i)=rho(k,j,i);
                }
            }
        }
    }

    if (rkp){
        for (k=0;k<NZ;k++){
            for (j=0;j<NY;j++){
                for (i=0;i<NX;i++){
                    if (k<NZ-1)
                        rkp(k,j,i)=0.5*(rho(k,j,i)+rho(k+1,j,i));
                    else
                        rkp(k,j,i)=rho(k,j,i);
                }
            }
        }
    }

    
    if (state && m->MPI_INIT==1)
        MPI_Bcast( &state, 1, MPI_INT, m->MYID, MPI_COMM_WORLD );
    
    return state;
    
    
    
}
