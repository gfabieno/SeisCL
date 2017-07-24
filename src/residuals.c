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

// Calculation of the residuals. The only really working function is res_raw, where raw traces are compared.

#include "F.h"

#define rho(z,y,x) rho[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define rip(z,y,x) rip[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define rjp(z,y,x) rjp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define rkp(z,y,x) rkp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define uipjp(z,y,x) uipjp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define ujpkp(z,y,x) ujpkp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define uipkp(z,y,x) uipkp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define u(z,y,x) u[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define pi(z,y,x) pi[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define grad(z,y,x) grad[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define grads(z,y,x) grads[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define amp1(z,y,x) amp1[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define amp2(z,y,x) amp2[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]

#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vy0(y,x) vy0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define ry(y,x) ry[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]
#define mute(y,x) mute[(y)*5+(x)]
#define weight(y,x) weight[(y)*NT+(x)]

#define vxcum(y,x) vxcum[(y)*NT+(x)]
#define vycum(y,x) vycum[(y)*NT+(x)]

#define u_in(z,y,x) u_in[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define pi_in(z,y,x) pi_in[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define uL(z,y,x) uL[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define piL(z,y,x) piL[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define taus(z,y,x) taus[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define tausipjp(z,y,x) tausipjp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define tausjpkp(z,y,x) tausjpkp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define tausipkp(z,y,x) tausipkp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define taup(z,y,x) taup[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]


#define PI (3.141592653589793238462643383279502884197169)

#define norm(x) sqrt( pow( (x).r,2) + pow( (x).i,2) )

static void ampdiff(float * data, float * data0, float *res, kiss_fft_cpx *trace, kiss_fft_cpx * trace0,  kiss_fft_cpx * Trace, kiss_fft_cpx * Trace0, int tmax, float dt, int nfft, int nfmax, int nfmin, kiss_fft_cfg *stf, kiss_fft_cfg *sti, int BACK_PROP_TYPE, float * rms, float * rmsnorm, int NFREQS, float *gradfreqs){
    int t,n, freqn;
    float normT, normT0;
    
    
    memset(trace,0,sizeof(kiss_fft_cpx)*nfft);
    memset(trace0,0,sizeof(kiss_fft_cpx)*nfft);
    memset(Trace,0,sizeof(kiss_fft_cpx)*nfft);
    memset(Trace0,0,sizeof(kiss_fft_cpx)*nfft);
    
    for (t=0;t<tmax;t++){
        trace[t].r=data[t];
        trace0[t].r=data0[t];
    }
    for (t=tmax;t<nfft;t++){
        trace[t].r=data[t-tmax];
        trace0[t].r=data0[t-tmax];
    }
    kiss_fft( *stf , (kiss_fft_cpx*)trace, (kiss_fft_cpx*)Trace );
    kiss_fft( *stf , (kiss_fft_cpx*)trace0, (kiss_fft_cpx*)Trace0 );
    
    

    for (n=0;n<nfft;n++){
        normT=norm(Trace[n]);
        normT0=norm(Trace0[n]);
        Trace[n].r*=(normT-normT0)/(normT+FLT_MIN*FLT_EPSILON);
        Trace[n].i*=(normT-normT0)/(normT+FLT_MIN*FLT_EPSILON);
    }
    
    kiss_fft( *sti , (kiss_fft_cpx*)Trace, (kiss_fft_cpx*)trace);
    
    for (t=0;t<tmax;t++){
        res[t]=-trace[t].r;
    }
    
    //    for (t=0;t<tmax;t++){
    //        res[t]=0;
    //        for (n=nfmin;n<nfmax;n++){
    //            res[t]+= -(norm(Trace[n])-norm(Trace0[n]))/norm(Trace[n])*(Trace[n].r*cos(2*PI*n*t/nfft)-Trace[n].i*sin(2*PI*n*t/nfft));
    //        }
    //    }
    
    
    if (BACK_PROP_TYPE==1){
        for (n=nfmin;n<nfmax;n++){
            *rms+=0.5*pow(norm(Trace[n])-norm(Trace0[n]),2);
            *rmsnorm+=0.5*pow(norm(Trace0[n]),2);
        }
    }
    else if (BACK_PROP_TYPE==2){
        for (n=0;n<NFREQS;n++){
            freqn=gradfreqs[n]*nfft*dt;
            *rms+=0.5*pow(norm(Trace[freqn])-norm(Trace0[freqn]),2);
            *rmsnorm+=0.5*pow(norm(Trace0[freqn]),2);
        }
    }
    
}

static void crossvar(float * data, float * data0, float *res, float *trace, float * trace0,  kiss_fft_cpx * Trace, kiss_fft_cpx * Trace0, int tmax, float dt, int nfft, int nfmax, int nfmin, kiss_fftr_cfg *stf, int BACK_PROP_TYPE, float * rms, float * rmsnorm, int NFREQS, float *gradfreqs){
    int t;
    
//    for (t=0;t<tmax;t++){
//        trace[t]=data[t];
//        trace0[t]=data0[t];
//    }
//    for (t=tmax;t<nfft;t++){
//        trace[t]=data[t-tmax];
//        trace0[t]=data0[t-tmax];
//    }
//    kiss_fftr( *stf , trace, (kiss_fft_cpx*)Trace );
//    kiss_fftr( *stf , trace0, (kiss_fft_cpx*)Trace0 );
    
    
    for (t=0;t<tmax;t++){
        res[t]=2.0*data[t]-data0[t];
    }
    
//    if (BACK_PROP_TYPE==1){
//        for (n=nfmin;n<nfmax;n++){
//            *rms+=0.5*pow(norm(Trace[n])-norm(Trace0[n]),2);
//            *rmsnorm+=0.5*pow(norm(Trace0[n]),2);
//            
//        }
//    }
//    else if (BACK_PROP_TYPE==2){
//        for (n=0;n<NFREQS;n++){
//            freqn=gradfreqs[n]*nfft*dt;
//            *rms+=0.5*pow(norm(Trace[freqn])-norm(Trace0[freqn]),2);
//            *rmsnorm+=0.5*pow(norm(Trace0[freqn]),2);
//        }
//    }
    
}


int res_raw(model * mptr, int s)
{
    
    int t,g,n, thisfreq;
    int state=0;
    float *vxout=NULL, *vzout=NULL, *vyout=NULL, *vx0=NULL, *vy0=NULL, *vz0=NULL, *rx=NULL, *ry=NULL, *rz=NULL;
    float *mute=NULL, *weight=NULL, *rms_scaling=NULL, *rms_scaling0=NULL;
    float *rmsnorm_scaling=NULL;
    float mw=0, ws=0;
    int Nw=0;
    float ts=0;
    int t0=0, nfft=0;
    float *temp_vx=NULL, *temp_vy=NULL, *temp_vz=NULL,*temp_vx0=NULL, *temp_vy0=NULL, *temp_vz0=NULL;
    kiss_fftr_cfg stf=NULL;
    kiss_fft_cpx *temp_vx_out=NULL, *temp_vy_out=NULL, *temp_vz_out=NULL,*temp_vx0_out=NULL, *temp_vy0_out=NULL, *temp_vz0_out=NULL;
    
    
    int NT=mptr->NT;
    int tmax=mptr->tmax;
    int tmin=mptr->tmin;
    float dt=mptr->dt;
    int nrec=(mptr->src_recs.nrec[s]);
    
    // For a lighter notation, we use local pointers to the data
//    if (mptr->vx0){
//        vxout=mptr->vxout[s];
//        vx0=mptr->vx0[s];
//    }
//    if (mptr->vy0){
//        vyout=mptr->vyout[s];
//        vy0=mptr->vy0[s];
//    }
//    if (mptr->vz0){
//        vzout=mptr->vzout[s];
//        vz0=mptr->vz0[s];
//    }
//    if (mptr->rx)
//        rx=mptr->rx[s];
//    if (mptr->ry)
//        ry=mptr->ry[s];
//    if (mptr->rz)
//        rz=mptr->rz[s];
//    
//    if (mptr->mute)
//        mute=mptr->mute[s];
//    if (mptr->weight)
//        weight=mptr->weight[s];
    
    
    //  The data is filtered between the maximum and minimum frequencies
    if ( (mptr->fmax>0 | mptr->fmin>0) ){
        if (vx0){
            butterworth(vx0, mptr->fmin, mptr->fmax, mptr->dt, mptr->NT, mptr->tmax, nrec, 6);
            butterworth(vxout, mptr->fmin, mptr->fmax, mptr->dt, mptr->NT, mptr->tmax,nrec, 6);
        }
        if (vy0){
            butterworth(vy0, mptr->fmin, mptr->fmax, mptr->dt, mptr->NT, mptr->tmax,nrec, 6);
            butterworth(vyout, mptr->fmin, mptr->fmax, mptr->dt, mptr->NT, mptr->tmax,nrec, 6);
        }
        if (vz0){
            butterworth(vz0, mptr->fmin, mptr->fmax, mptr->dt, mptr->NT, mptr->tmax,nrec, 6);
            butterworth(vzout, mptr->fmin, mptr->fmax, mptr->dt, mptr->NT, mptr->tmax,nrec, 6);
        }
    }
    

    
    
    // We calculate the rms value, if data is to be scaled with it
    rms_scaling = malloc(nrec*sizeof(float));
    rms_scaling0 = malloc(nrec*sizeof(float));
    
    
    if (mptr->scalermsnorm || mptr->scalerms){
        rmsnorm_scaling = malloc(nrec*sizeof(float));
    }
    
    if ( mptr->scalerms || mptr->scaleshot || mptr->scalermsnorm){
        for (g=0;g<nrec;g++){
            rms_scaling[g]=0;
            rms_scaling0[g]=0;
            if (vx0){
                for (t=tmin;t<tmax;t++){
                    rms_scaling[g]+=pow(vxout(g,t),2);
                    rms_scaling0[g]+=pow(vx0(g,t),2);
                }
            }
            if (vy0){
                for (t=tmin;t<tmax;t++){
                    rms_scaling[g]+=pow(vyout(g,t),2);
                    rms_scaling0[g]+=pow(vy0(g,t),2);
                }
            }
            if (vz0){
                for (t=tmin;t<tmax;t++){
                    rms_scaling[g]+=pow(vzout(g,t),2);
                    rms_scaling0[g]+=pow(vz0(g,t),2);
                }
            }
            
            rms_scaling[g]=sqrt(1./rms_scaling[g]);
            rms_scaling0[g]=sqrt(1./rms_scaling0[g]);
        }
    }
    if (mptr->scaleshot){
        ws=0;
        for (g=0;g<nrec;g++){
            ws+=1/pow(rms_scaling0[g],2);
        }
        ws=1/sqrt(ws);
        for (g=0;g<nrec;g++){
            rms_scaling[g]=ws;
            rms_scaling0[g]=ws;
        }
        
    }
    if (mptr->scalerms){
        ws=0;
        for (g=0;g<nrec;g++){
            ws+=1/pow(rms_scaling0[g],2);
        }
        ws=1/sqrt(ws);
        for (g=0;g<nrec;g++){
            rms_scaling[g]*=ws/rms_scaling0[g];
            rms_scaling0[g]=ws;
        }
        
    }
    
    
    

    
    //Apply the rms scaling to the data
    if (mptr->scalerms || mptr->scaleshot || mptr->scalermsnorm){
        for (g=0;g<nrec;g++){
            
            if(vx0){
                for (t=tmin;t<tmax;t++){
                    vxout(g,t)*= rms_scaling[g];
                    vx0(g,t)*= rms_scaling0[g];
                }
            }
            if(vy0){
                for (t=tmin;t<tmax;t++){
                    vyout(g,t)*= rms_scaling[g];
                    vy0(g,t)*= rms_scaling0[g];
                }
            }
            if(vz0){
                for (t=tmin;t<tmax;t++){
                    vzout(g,t)*= rms_scaling[g];
                    vz0(g,t)*= rms_scaling0[g];
                }
            }
            
        }
    }
    else {
        for (g=0;g<nrec;g++){
            rms_scaling[g]=1.;
            rms_scaling0[g]=1.;
        }
    }
    
    if (mptr->scalerms || mptr->scalermsnorm){
        
        for (g=0;g<nrec;g++){
            rmsnorm_scaling[g]=0;
            
            if(vx0){
                for (t=tmin;t<tmax;t++){
                    rmsnorm_scaling[g]+=vxout(g,t)*(vxout(g,t)-vx0(g,t));
                }
            }
            if(vy0){
                for (t=tmin;t<tmax;t++){
                    rmsnorm_scaling[g]+=vyout(g,t)*(vyout(g,t)-vy0(g,t));
                }
            }
            if(vz0){
                for (t=tmin;t<tmax;t++){
                    rmsnorm_scaling[g]+=vzout(g,t)*(vzout(g,t)-vz0(g,t));
                }
            }
            if (mptr->scalerms){
                rmsnorm_scaling[g]*=pow(ws/rms_scaling0[g]/ws,2);
            }
            rmsnorm_scaling[g]=1.0-rmsnorm_scaling[g];
            
        }
    }
    
    // Define some intermediate buffers needed we use frequency domain gradient
    if (mptr->BACK_PROP_TYPE==2 )  {
        nfft=kiss_fft_next_fast_size(mptr->tmax);
        if (vx0){
            GMALLOC(temp_vx,sizeof(float)*nfft*2);
            GMALLOC(temp_vx_out,sizeof(float)*nfft*2);
            GMALLOC(temp_vx0,sizeof(float)*nfft*2);
            GMALLOC(temp_vx0_out,sizeof(float)*nfft*2);

        }
        if (vy0){
            GMALLOC(temp_vy,sizeof(float)*nfft*2);
            GMALLOC(temp_vy_out,sizeof(float)*nfft*2);
            GMALLOC(temp_vy0,sizeof(float)*nfft*2);
            GMALLOC(temp_vy0_out,sizeof(float)*nfft*2);

        }
        if (vz0){
            GMALLOC(temp_vz,sizeof(float)*nfft*2);
            GMALLOC(temp_vz_out,sizeof(float)*nfft*2);
            GMALLOC(temp_vz0,sizeof(float)*nfft*2);
            GMALLOC(temp_vz0_out,sizeof(float)*nfft*2);

        }

        stf = kiss_fftr_alloc( nfft ,0 ,0,0);
    }
    
    
    // Main loop to calculate residuals
    for (g=0;g<nrec;g++){

        for (t=0;t<tmax;t++){
            mw=1.0;
            
           
            
            // Calculate the rms value
            if (mptr->BACK_PROP_TYPE==1){
                if (vx0){
                    mptr->rms+=pow(-vxout(g,t)+vx0(g,t),2);
                    mptr->rmsnorm+=pow(vx0(g,t),2);
                }
                if (vy0){
                    mptr->rms+=pow(-vyout(g,t)+vy0(g,t),2);
                    mptr->rmsnorm+=pow(vy0(g,t),2);
                }
                if (vz0){
                    mptr->rms+=pow(-vzout(g,t)+vz0(g,t),2);
                    mptr->rmsnorm+=pow(vz0(g,t),2);
                }
            }
            else if (mptr->BACK_PROP_TYPE==2){
                if (vx0){
                    temp_vx[t]=(-vxout(g,t)+vx0(g,t));
                    temp_vx0[t]=vx0(g,t);
                }
                if (vy0){
                    temp_vy[t]=(-vyout(g,t)+vy0(g,t));
                    temp_vy0[t]=vy0(g,t);
                }
                if (vz0){
                    temp_vz[t]=(-vzout(g,t)+vz0(g,t));
                    temp_vz0[t]=vz0(g,t);
                }
            }
            
            //Calculate the adjoint sources
            if (mptr->scalermsnorm || mptr->scalerms){
                if (rx)
                    rx(g,t)= -mw*rms_scaling[g]*( (1.0-rmsnorm_scaling[g])*vxout(g,t)-vx0(g,t));
                if (ry)
                    ry(g,t)= -mw*rms_scaling[g]*( (1.0-rmsnorm_scaling[g])*vyout(g,t)-vy0(g,t));
                if (rz)
                    rz(g,t)= -mw*rms_scaling[g]*( (1.0-rmsnorm_scaling[g])*vzout(g,t)-vz0(g,t));
            }
            else {
                if (rx)
                rx(g,t)= -mw*rms_scaling[g]*(vxout(g,t)-vx0(g,t));
                if (ry)
                    ry(g,t)= -mw*rms_scaling[g]*(vyout(g,t)-vy0(g,t));
                if (rz)
                    rz(g,t)= -mw*rms_scaling[g]*(vzout(g,t)-vz0(g,t));
            }
            
        }
        
        //Calculate the rms for the selected frequencies only for frequency gradient
        if (mptr->BACK_PROP_TYPE==2){
            if (vx0){
                kiss_fftr( stf , temp_vx, (kiss_fft_cpx*)temp_vx_out );
                kiss_fftr( stf , temp_vx0, (kiss_fft_cpx*)temp_vx0_out );
            }
            if (vy0){
                kiss_fftr( stf , temp_vy, (kiss_fft_cpx*)temp_vy_out );
                kiss_fftr( stf , temp_vy0, (kiss_fft_cpx*)temp_vy0_out );
            }
            if (vz0){
                kiss_fftr( stf , temp_vz, (kiss_fft_cpx*)temp_vz_out );
                kiss_fftr( stf , temp_vz0, (kiss_fft_cpx*)temp_vz0_out );
            }

            for (n=0;n<mptr->NFREQS;n++){
                thisfreq=mptr->csts[21].gl_cst[n]*nfft*mptr->dt+1;
                if (vx0){
                    mptr->rms+=powf(temp_vx_out[thisfreq].i ,2)+powf(temp_vx_out[thisfreq].r, 2);
                    mptr->rmsnorm+=powf(temp_vx0_out[thisfreq].i,2)+powf(temp_vx0_out[thisfreq].r,2);
                }
                if (vy0){
                    mptr->rms+=powf(temp_vy_out[thisfreq].i,2)+powf(temp_vy_out[thisfreq].r,2);
                    mptr->rmsnorm+=powf(temp_vy0_out[thisfreq].i,2)+powf(temp_vy0_out[thisfreq].r,2);
                }
                if (vz0){
                    mptr->rms+=powf(temp_vz_out[thisfreq].i,2)+powf(temp_vz_out[thisfreq].r,2);
                    mptr->rmsnorm+=powf(temp_vz0_out[thisfreq].i,2)+powf(temp_vz0_out[thisfreq].r,2);
                }
            }
        }
        
        
        
    }
    
    if (mptr->BACK_PROP_TYPE==2){
        free(stf);
        if (temp_vx) free(temp_vx);
        if (temp_vx0) free(temp_vx0);
        if (temp_vx_out) free(temp_vx_out);
        if (temp_vx0_out) free(temp_vx0_out);
        if (temp_vy) free(temp_vy);
        if (temp_vy0) free(temp_vy0);
        if (temp_vy_out) free(temp_vy_out);
        if (temp_vy0_out) free(temp_vy0_out);
        if (temp_vz) free(temp_vz);
        if (temp_vz0) free(temp_vz0);
        if (temp_vz_out) free(temp_vz_out);
        if (temp_vz0_out) free(temp_vz0_out);
        kiss_fft_cleanup();
    }
    
    free(rms_scaling);
    free(rms_scaling0);
    if (mptr->scalerms || mptr->scalermsnorm){
        free(rmsnorm_scaling);
    }
    
    //Check if we have infinite or NaN values
    if (mptr->rms != mptr->rms){
        state=1;
        fprintf(stderr,"Simulation has become unstable, stopping\n");
    }
    
    return state;
}

int res_amp(model * m, int s){
    int state=0;
    
    int g;
    float *vxout=NULL, *vzout=NULL, *vyout=NULL, *vx0=NULL, *vy0=NULL, *vz0=NULL, *rx=NULL, *ry=NULL, *rz=NULL;
    int  nfft=0;
    int nfmin=0, nfmax=0;
    kiss_fft_cfg stf=NULL, sti=NULL;
   
    
    int NT=m->NT;
    int tmax=m->tmax;
    float dt=m->dt;
    int nrec=(m->src_recs.nrec[s]);
//    vxout=m->vxout[s];
//    vzout=m->vzout[s];
//    vx0=m->vx0[s];
//    vz0=m->vz0[s];
//    
//    
//    if (m->ND==3){
//        vyout=m->vyout[s];
//        vy0=m->vy0[s];
//    }
//    if (m->rx){
//        rx=m->rx[s];
//        rz=m->rz[s];
//        if (m->ND==3){
//            ry=m->ry[s];
//        }
//    }

    
    kiss_fft_cpx *trace=NULL, *trace0=NULL;
    kiss_fft_cpx *Trace=NULL, *Trace0=NULL;
    nfft=kiss_fft_next_fast_size(tmax);
    GMALLOC(trace,sizeof(kiss_fft_cpx)*nfft);
    GMALLOC(trace0,sizeof(kiss_fft_cpx)*nfft);
    GMALLOC(Trace,sizeof(kiss_fft_cpx)*nfft);
    GMALLOC(Trace0,sizeof(kiss_fft_cpx)*nfft);
    if (!(stf = kiss_fft_alloc( nfft ,0 ,0,0))) {state=1;fprintf(stderr,"residual failed\n");};
    if (!(sti = kiss_fft_alloc( nfft ,1 ,0,0))) {state=1;fprintf(stderr,"residual failed\n");};
    nfmin=m->fmin/(nfft*m->dt);
    nfmax=m->fmax/(nfft*m->dt);
    
    
    for (g=0;g<nrec;g++){
        
        
        ampdiff( &vxout(g,0), &vx0(g,0), &rx(g,0), trace, trace0,  Trace, Trace0, tmax, dt, nfft, nfmax, nfmin, &stf, &sti, m->BACK_PROP_TYPE, &m->rms, &m->rmsnorm, m->NFREQS, m->csts[21].gl_cst);
        ampdiff( &vzout(g,0), &vz0(g,0), &rz(g,0), trace, trace0,  Trace, Trace0, tmax, dt, nfft, nfmax, nfmin, &stf, &sti, m->BACK_PROP_TYPE, &m->rms, &m->rmsnorm, m->NFREQS, m->csts[21].gl_cst);
        if (m->ND==3){
            ampdiff( &vyout(g,0), &vy0(g,0), &ry(g,0), trace, trace0,  Trace, Trace0, tmax, dt, nfft, nfmax, nfmin, &stf, &sti, m->BACK_PROP_TYPE, &m->rms, &m->rmsnorm, m->NFREQS, m->csts[21].gl_cst);
        }

        
    }
    
    GFree(trace);
    GFree(trace0);
    GFree(Trace);
    GFree(Trace0);
    GFree(stf);
    GFree(sti);
    kiss_fft_cleanup();
    
    return state;
}


