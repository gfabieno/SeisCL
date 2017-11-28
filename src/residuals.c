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


int var_res_raw(model * m, int s)
{
    
    int t,g,n,i,f, thisfreq, j;
    int x,y,z, pos=0;
    int state=0;
    float *rms_scaling=NULL, *rms_scaling0=NULL;
    float *rmsnorm_scaling=NULL;
    float ws=0;
    int nfft=0;
    kiss_fftr_cfg stf=NULL;
    
    float **temp=NULL;
    kiss_fft_cpx **temp_out=NULL;
    float **temp0=NULL;
    kiss_fft_cpx **temp0_out=NULL;
   
    
    int NT=m->NT;
    int tmax=m->tmax;
    int tmin=m->tmin;
    int nrec=(m->src_recs.nrec[s]);
    float * par=NULL;
    float * par2 = NULL;
    float * gradfreqs;
    float parscal;
    float resmax;
    
    
    //  The data is filtered between the maximum and minimum frequencies
    if ( (m->fmax>0 | m->fmin>0) ){
        for (i=0;i<m->nvars;i++){
            if (m->vars[i].to_output){
                butterworth(m->vars[i].gl_varout[s],
                            m->fmin,
                            m->fmax,
                            m->dt,
                            m->NT,
                            m->tmax,
                            nrec, 6);
                butterworth(m->vars[i].gl_varin[s],
                            m->fmin,
                            m->fmax,
                            m->dt,
                            m->NT,
                            m->tmax,
                            nrec, 6);
            }
            
        }
        for (i=0;i<m->ntvars;i++){
            if (m->trans_vars[i].to_output){
                butterworth(m->trans_vars[i].gl_varout[s],
                            m->fmin,
                            m->fmax,
                            m->dt,
                            m->NT,
                            m->tmax,
                            nrec, 6);
                butterworth(m->trans_vars[i].gl_varin[s],
                            m->fmin,
                            m->fmax,
                            m->dt,
                            m->NT,
                            m->tmax,
                            nrec, 6);
            }
            
        }
    }

    // We compute the rms value, if data is to be scaled with it
    rms_scaling = malloc(nrec*sizeof(float));
    rms_scaling0 = malloc(nrec*sizeof(float));
    
    if (m->scalermsnorm || m->scalerms){
        rmsnorm_scaling = malloc(nrec*sizeof(float));
    }
    
    if ( m->scalerms || m->scaleshot || m->scalermsnorm){
        for (g=0;g<nrec;g++){
            rms_scaling[g]=0;
            rms_scaling0[g]=0;
            for (i=0;i<m->nvars;i++){
                if (m->vars[i].to_output){
                    for (t=tmin;t<tmax;t++){
                        rms_scaling[g]+=pow(m->vars[i].gl_varout[s][g*NT+t],2);
                        rms_scaling0[g]+=pow(m->vars[i].gl_varin[s][g*NT+t],2);
                    }
                }
            }
            for (i=0;i<m->ntvars;i++){
                if (m->trans_vars[i].to_output){
                    for (t=tmin;t<tmax;t++){
                        rms_scaling[g]+=pow(m->trans_vars[i].gl_varout[s][g*NT+t],2);
                        rms_scaling0[g]+=pow(m->trans_vars[i].gl_varin[s][g*NT+t],2);
                    }
                }
            }
            rms_scaling[g]=sqrt(1./rms_scaling[g]);
            rms_scaling0[g]=sqrt(1./rms_scaling0[g]);
        }
    }
    if (m->scaleshot){
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
    if (m->scalerms){
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
    if (m->scalerms || m->scaleshot || m->scalermsnorm){
        for (g=0;g<nrec;g++){
            for (i=0;i<m->nvars;i++){
                if (m->vars[i].to_output){
                    for (t=tmin;t<tmax;t++){
                        m->vars[i].gl_varout[s][g*NT+t]*=rms_scaling[g];
                        m->vars[i].gl_varin[s][g*NT+t]*=rms_scaling0[g];
                    }
                }
            }
            for (i=0;i<m->ntvars;i++){
                if (m->trans_vars[i].to_output){
                    for (t=tmin;t<tmax;t++){
                        m->trans_vars[i].gl_varout[s][g*NT+t]*=rms_scaling[g];
                        m->trans_vars[i].gl_varin[s][g*NT+t]*=rms_scaling0[g];
                    }
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
    
    if (m->scalerms || m->scalermsnorm){
        
        for (g=0;g<nrec;g++){
            rmsnorm_scaling[g]=0;
            
            for (i=0;i<m->nvars;i++){
                if (m->vars[i].to_output){
                    for (t=tmin;t<tmax;t++){
                        rmsnorm_scaling[g]+= m->vars[i].gl_varout[s][g*NT+t]*
                                            (m->vars[i].gl_varout[s][g*NT+t]
                                            -m->vars[i].gl_varin[s][g*NT+t]);
                    }
                }
            }
            for (i=0;i<m->ntvars;i++){
                if (m->trans_vars[i].to_output){
                    for (t=tmin;t<tmax;t++){
                        rmsnorm_scaling[g]+= m->trans_vars[i].gl_varout[s][g*NT+t]*
                        (m->trans_vars[i].gl_varout[s][g*NT+t]
                         -m->trans_vars[i].gl_varin[s][g*NT+t]);
                    }
                }
            }
            if (m->scalerms){
                rmsnorm_scaling[g]*=pow(ws/rms_scaling0[g]/ws,2);
            }
            rmsnorm_scaling[g]=1.0-rmsnorm_scaling[g];
            
        }
    }
    
    // Define some intermediate buffers needed we use frequency domain gradient
    int nout=0;
    for (i=0;i<m->nvars;i++){
        if (m->vars[i].to_output){
            nout++;
        }
    }
    for (i=0;i<m->ntvars;i++){
        if (m->trans_vars[i].to_output){
            nout++;
        }
    }


    if (m->BACK_PROP_TYPE==2 )  {
        GMALLOC(temp, sizeof(float *)*nout);
        GMALLOC(temp_out, sizeof(kiss_fft_cpx *)*nout);
        GMALLOC(temp0, sizeof(float *)*nout);
        GMALLOC(temp0_out, sizeof(kiss_fft_cpx *)*nout);
        
        nfft=kiss_fft_next_fast_size(m->tmax);
        n=0;
        for (i=0;i<m->nvars;i++){
            if (m->vars[i].to_output){
                GMALLOC(temp[n], sizeof(float)*nfft*2);
                GMALLOC(temp_out[n], sizeof(kiss_fft_cpx)*nfft*2);
                GMALLOC(temp0[n], sizeof(float)*nfft*2);
                GMALLOC(temp0_out[n], sizeof(kiss_fft_cpx)*nfft*2);
                n++;
            }
        }
        for (i=0;i<m->ntvars;i++){
            if (m->trans_vars[i].to_output){
                GMALLOC(temp[n], sizeof(float)*nfft*2);
                GMALLOC(temp_out[n], sizeof(kiss_fft_cpx)*nfft*2);
                GMALLOC(temp0[n], sizeof(float)*nfft*2);
                GMALLOC(temp0_out[n], sizeof(kiss_fft_cpx)*nfft*2);
                n++;
            }
        }
        stf = kiss_fftr_alloc( nfft ,0 ,0,0);
    }
    
    
    // Main loop to calculate residuals
    for (g=0;g<nrec;g++){

        for (t=0;t<tmax;t++){

            // Calculate the rms value
            if (m->BACK_PROP_TYPE==1){
                for (i=0;i<m->nvars;i++){
                    if (m->vars[i].to_output){
                       m->rms+=pow(-m->vars[i].gl_varout[s][g*NT+t]
                                   +m->vars[i].gl_varin[s][g*NT+t],2);
                        m->rmsnorm+=pow(m->vars[i].gl_varin[s][g*NT+t],2);
                    }
                }
                for (i=0;i<m->ntvars;i++){
                    if (m->trans_vars[i].to_output){
                        m->rms+=pow(-m->trans_vars[i].gl_varout[s][g*NT+t]
                                    +m->trans_vars[i].gl_varin[s][g*NT+t],2);
                        m->rmsnorm+=pow(m->trans_vars[i].gl_varin[s][g*NT+t],2);
                    }
                }
            }
            else if (m->BACK_PROP_TYPE==2){
                n=0;
                for (i=0;i<m->nvars;i++){
                    if (m->vars[i].to_output){
                        temp[n][t]=-m->vars[i].gl_varout[s][g*NT+t]
                                   +m->vars[i].gl_varin[s][g*NT+t];
                        temp0[n][t]=m->vars[i].gl_varin[s][g*NT+t];
                        n++;
                    }
                }
                for (i=0;i<m->ntvars;i++){
                    if (m->trans_vars[i].to_output){
                        temp[n][t]=-m->trans_vars[i].gl_varout[s][g*NT+t]
                        +m->trans_vars[i].gl_varin[s][g*NT+t];
                        temp0[n][t]=m->trans_vars[i].gl_varin[s][g*NT+t];
                        n++;
                    }
                }
            }
            
            //Calculate the adjoint sources
            if (m->scalermsnorm || m->scalerms){
                for (i=0;i<m->nvars;i++){
                    if (m->vars[i].to_output){
                        m->vars[i].gl_var_res[s][g*NT+t]=rms_scaling[g]*(
                                            (1.0-rmsnorm_scaling[g])
                                            *m->vars[i].gl_varout[s][g*NT+t]
                                            -m->vars[i].gl_varin[s][g*NT+t]);
                    }
                }
                for (i=0;i<m->ntvars;i++){
                    if (m->trans_vars[i].to_output){
                        m->trans_vars[i].gl_var_res[s][g*NT+t]=rms_scaling[g]*(
                                        (1.0-rmsnorm_scaling[g])
                                        *m->trans_vars[i].gl_varout[s][g*NT+t]
                                        -m->trans_vars[i].gl_varin[s][g*NT+t]);
                    }
                }
            }
            else {
                for (i=0;i<m->nvars;i++){
                    if (m->vars[i].to_output){
                        m->vars[i].gl_var_res[s][g*NT+t]=rms_scaling[g]*(
                                                m->vars[i].gl_varout[s][g*NT+t]
                                               -m->vars[i].gl_varin[s][g*NT+t]);
                    }
                }
                for (i=0;i<m->ntvars;i++){
                    if (m->trans_vars[i].to_output){
                        m->trans_vars[i].gl_var_res[s][g*NT+t]=rms_scaling[g]*(
                                         m->trans_vars[i].gl_varout[s][g*NT+t]
                                        -m->trans_vars[i].gl_varin[s][g*NT+t]);
                    }
                }
            }
            
        }
        
        //Compute the rms for the selected frequencies only
        if (m->BACK_PROP_TYPE==2){
            n=0;
            for (i=0;i<m->nvars;i++){
                if (m->vars[i].to_output){
                    kiss_fftr( stf , temp[n], (kiss_fft_cpx*)temp_out[n] );
                    kiss_fftr( stf , temp0[n], (kiss_fft_cpx*)temp0_out[n] );
                    n++;
                }
            }
            for (i=0;i<m->ntvars;i++){
                if (m->trans_vars[i].to_output){
                    kiss_fftr( stf , temp[n], (kiss_fft_cpx*)temp_out[n] );
                    kiss_fftr( stf , temp0[n], (kiss_fft_cpx*)temp0_out[n] );
                    n++;
                }
            }
            
            gradfreqs = get_cst( m->csts, m->ncsts, "gradfreqs")->gl_cst;
            for (f=0;f<m->NFREQS;f++){
                thisfreq=gradfreqs[f]*nfft*m->dt+1;
                n=0;
                for (i=0;i<m->nvars;i++){
                    if (m->vars[i].to_output){
                        m->rms+=powf(temp_out[n][thisfreq].i ,2)
                               +powf(temp_out[n][thisfreq].r, 2);
                        m->rmsnorm+=powf(temp0_out[n][thisfreq].i ,2)
                                   +powf(temp0_out[n][thisfreq].r, 2);
                        n++;
                    }
                }
                for (i=0;i<m->ntvars;i++){
                    if (m->trans_vars[i].to_output){
                        m->rms+=powf(temp_out[n][thisfreq].i ,2)
                        +powf(temp_out[n][thisfreq].r, 2);
                        m->rmsnorm+=powf(temp0_out[n][thisfreq].i ,2)
                        +powf(temp0_out[n][thisfreq].r, 2);
                        n++;
                    }
                }
            }
        }
        
        
        
    }
    
    
    // Scale by the material parameters
    int scaler=0;
    variable * var;
    var = get_var(m->vars,m->nvars, "sxx");
    if (var) scaler = var->scaler;
    var = get_var(m->vars,m->nvars, "sxz");
    if (var) scaler = var->scaler;
    
    resmax=0;
    for (i=0;i<m->nvars;i++){
        if (m->vars[i].to_output){
            if (strcmp(m->vars[i].name,"vx")==0 ||
                strcmp(m->vars[i].name,"vy")==0 ||
                strcmp(m->vars[i].name,"vz")==0 ){
                
                if (strcmp(m->vars[i].name,"vx")==0){
                    par = get_par(m->pars, m->npars, "rip")->gl_par;
                }
                else if (strcmp(m->vars[i].name,"vy")==0){
                    par = get_par(m->pars, m->npars, "rjp")->gl_par;
                }
                else {
                    par = get_par(m->pars, m->npars, "rkp")->gl_par;
                }
                for (g=0;g<nrec;g++){
                    
                    x = m->src_recs.rec_pos[s][0+8*g]/m->dh;
                    y = m->src_recs.rec_pos[s][0+8*g]/m->dh;
                    z = m->src_recs.rec_pos[s][0+8*g]/m->dh;
                    if (m->NDIM==2){
                        pos = x*m->N[0]+z;
                    }
                    else {
                        pos = x*m->N[0]*m->N[1]+y*m->N[0]+z;
                    }
                    if (m->FP16!=2 || m->FP16!=4){
                        parscal = half_to_float( ((half*)par)[pos] );
                        parscal = 1.0/parscal*m->dh/m->dt*powf(2,scaler);
                    }
                    else{
                        parscal = 1.0/par[pos]*m->dh/m->dt*powf(2,scaler);
                    }
                    for (t=0;t<tmax;t++){
                        m->vars[i].gl_var_res[s][g*NT+t]*=1.0/parscal*m->dt ;
                        if (resmax<fabsf(m->vars[i].gl_var_res[s][g*NT+t])){
                            resmax=fabsf(m->vars[i].gl_var_res[s][g*NT+t]);
                        }
                    }
                }
            }
            
        }
    }
    for (i=0;i<m->ntvars;i++){
        if (m->trans_vars[i].to_output){
            if (strcmp(m->trans_vars[i].name,"p")==0){
                par = get_par(m->pars, m->npars, "M")->gl_par;
                par2 = get_par(m->pars, m->npars, "mu")->gl_par;

                for (g=0;g<nrec;g++){
                    
                    x = m->src_recs.rec_pos[s][0+8*g]/m->dh;
                    y = m->src_recs.rec_pos[s][1+8*g]/m->dh;
                    z = m->src_recs.rec_pos[s][2+8*g]/m->dh;
                    if (m->NDIM==2){
                        pos = x*m->N[0]+z;
                    }
                    else {
                        pos = x*m->N[0]*m->N[1]+y*m->N[0]+z;
                    }
                    if (m->FP16!=2 || m->FP16!=4){
                        parscal = half_to_float( ((half*)par)[pos] )
                                 -half_to_float( ((half*)par2)[pos] );
                        parscal = -2.0*(parscal)*m->dh/m->dt*powf(2,-scaler);
                    }
                    else{
                        parscal = -2.0*(par[pos]-par2[pos])
                                                   *m->dh/m->dt*powf(2,-scaler);
                    }
                    for (t=0;t<tmax;t++){
                        m->trans_vars[i].gl_var_res[s][g*NT+t]*=parscal*m->dt;
                        if (resmax<fabsf(m->trans_vars[i].gl_var_res[s][g*NT+t])){
                            resmax=fabsf(m->trans_vars[i].gl_var_res[s][g*NT+t]);
                        }
                    }
                }
            }
        }
    }
    m->src_recs.res_scales[s]=-log2(resmax/10);
    // Free memory of FFTs
    if (m->BACK_PROP_TYPE==2){
        free(stf);
        n=0;
        for (i=0;i<m->nvars;i++){
            if (m->vars[i].to_output){
                GFree(temp[n]);
                GFree(temp0[n]);
                GFree(temp_out[n]);
                GFree(temp0_out[n]);
                n++;
            }
        }
        for (i=0;i<m->ntvars;i++){
            if (m->trans_vars[i].to_output){
                GFree(temp[n]);
                GFree(temp0[n]);
                GFree(temp_out[n]);
                GFree(temp0_out[n]);
                n++;
            }
        }
        kiss_fft_cleanup();
    }
    
    free(rms_scaling);
    free(rms_scaling0);
    if (m->scalerms || m->scalermsnorm){
        free(rmsnorm_scaling);
    }
    
    //Check if we have infinite or NaN values
    if (m->rms != m->rms){
        state=1;
        fprintf(stderr,"Simulation has become unstable, stopping\n");
    }
    
    return state;
}



