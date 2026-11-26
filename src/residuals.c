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
#include "third_party/NVIDIA_FP16/fp16_conversion.h"


int rtm_res(model * m, int s)
{
    int state=0;
    int g, t, i;
    int tmax=m->tmax;
    int nrec=(m->src_recs.nrec[s]);
    int NT=m->NT;
    
    // Main loop to calculate residuals
    for (g=0;g<nrec;g++){
        
        for (t=0;t<tmax;t++){
            
           
            for (i=0;i<m->nvars;i++){
                if (m->vars[i].to_output){
                    m->vars[i].gl_var_res[s][g*NT+t]=m->vars[i].gl_varin[s][g*NT+t];
                }
            }
            for (i=0;i<m->ntvars;i++){
                if (m->trans_vars[i].to_output){
                    m->trans_vars[i].gl_var_res[s][g*NT+t]=m->trans_vars[i].gl_varin[s][g*NT+t];
                }
            }
            
        }
    }
    
    
    return state;
}

int var_res_raw(model * m, int s)
{
    
    int t,g,n,i,f, thisfreq;
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
    float * gradfreqs;
    
    
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
        fprintf(stderr,"Error: Simulation has become unstable, stopping\n");
    }
    
    return state;
}
int res_scale(model * m, int s)
{
    // Scale by the material parameters
    int scaler=0;
    int i, g, x, y, z, pos, t;
    float * par=NULL;
    float * par2 = NULL;
    float parscal;
    float resmax;
    int nrec=(m->src_recs.nrec[s]);
    int NT=m->NT;
    int tmax=m->tmax;
    
    if (m->FP16>0){
        scaler = m->par_scale;
    }
    
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
                    /* 2D SH has no staggered buoyancy at all: rip/rjp/rkp are
                     * appended only for ND!=21 (assign_modeling_case.c:1073),
                     * and update_v2D_SH takes plain rho. Looking up "rjp" here
                     * returned a zeroed struct -- get_par returns the address
                     * of a local when the name is unknown -- so gl_par was
                     * NULL and any SH run with gradout, rmsout or resout
                     * segfaulted in the line below, for either
                     * back_prop_type. */
                    par = get_par(m->pars, m->npars,
                                  m->ND==21 ? "rho" : "rjp")->gl_par;
                }
                else {
                    par = get_par(m->pars, m->npars, "rkp")->gl_par;
                }
                for (g=0;g<nrec;g++){

                    /* Was rec_pos[s][0+8*g] for all three of x/y/z (copied
                     * from the same index) -- correct only for a receiver
                     * exactly on the x axis at y=z=0. The trans_vars ("p")
                     * branch below has always used the correct 0/1/2
                     * indices; this branch didn't match it. Currently masked
                     * in test_gradient_fd.py because that test's background
                     * model is homogeneous at every receiver depth, so
                     * sampling the wrong (but still-uniform) cell reads the
                     * same buoyancy value regardless. Independently found
                     * and fixed the same way in SeisCL-freesurface. */
                    x = m->src_recs.rec_pos[s][0+8*g]/m->dh;
                    y = m->src_recs.rec_pos[s][1+8*g]/m->dh;
                    z = m->src_recs.rec_pos[s][2+8*g]/m->dh;
                    if (m->NDIM==2){
                        pos = x*m->N[0]+z;
                    }
                    else {
                        pos = x*m->N[0]*m->N[1]+y*m->N[0]+z;
                    }
                    if (m->FP16>1){
                        parscal = half_to_float( ((half*)par)[pos] );
                        parscal = 1.0/parscal*m->dh/m->dt*powf(2,scaler);
                    }
                    else{
                        if (!par){
                            fprintf(stderr,"Error: res_scale: no buoyancy "
                                           "parameter for variable %s\n",
                                    m->vars[i].name);
                            return 1;
                        }
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
                    /* dt/dh here, not dh/dt: unlike the vx/vy/vz branch above
                     * (which inverts parscal via 1/parscal before use, so it
                     * needs the reciprocal ratio dh/dt), this branch
                     * multiplies by parscal directly, so it needs dt/dh from
                     * the start. Using dh/dt here left every pressure-
                     * residual back_prop_type=1 gradient miscalibrated by
                     * (dh/dt)^2 -- confirmed by scaling dh and dt
                     * independently and observing the miscalibration scale
                     * as (dh/dt)^2 exactly (notes/todo.md item 0c). */
                    /* Exponent on the FP16 par_scale term is
                     * back_prop_type-dependent -- confirmed empirically,
                     * not derived from first principles (notes/todo.md
                     * item 0e). back_prop_type=1 needs -2*scaler: its
                     * unscale_grad() later multiplies the whole gradient by
                     * a *single* extra powf(2,par_scale), so the adjoint
                     * source itself must carry the other factor of
                     * 2^-par_scale for the round trip to cancel -- measured
                     * directly (FP16=1 ratio 0.978/1.021/0.999, matching
                     * FP16=0 to 4 digits, vs ~1e-6 or ~1e-19 for the two
                     * single-power alternatives tried first).
                     * back_prop_type=2 never calls unscale_grad(), and
                     * changing this from the original single -scaler broke
                     * test_dft_gradient_every_fp16_level (which needs
                     * FP16=1 proportional to FP16=0, not just internally
                     * consistent) -- so it keeps the single power.
                     * Independently found and fixed the same way in
                     * SeisCL-freesurface. */
                    /* 2026-11-14: one convention for both back_prop_types.
                     * The single power above made back_prop_type=2 right for
                     * pressure receivers only: with velocity receivers its
                     * gradient was 2^-par_scale too large, because
                     * unscale_grad_dft() lacked unscale_grad()'s 2^par_scale.
                     * unscale_grad_dft() now applies it, so the pressure
                     * adjoint source takes the same -2*scaler as
                     * back_prop_type=1. Checked: FP16=1 equals FP16=0 to
                     * ~1e-6 for both gradient methods, velocity and pressure
                     * receivers, force and pressure sources (2-D, 3-D). */
                    float scaler2 = 2.0f*scaler;
                    /* The trace-of-stress modulus, N*M - 2(N-1)*mu.
                     *
                     * "p" is the average of the N normal stresses, and the
                     * stress update (update_s{2D,3D}.cl, elastic g=M, f=2mu)
                     * gives their sum directly:
                     *   2D  sxx+szz         = (2M - 2mu)*theta
                     *   3D  sxx+syy+szz     = (3M - 4mu)*theta
                     * i.e. (N*M - 2(N-1)*mu)*theta -- the same combination
                     * that appears as `den` throughout the gradient
                     * coefficients (grad_coefelast_1's
                     * 1/pow(ND*M-2*(ND-1)*mu,2)).
                     *
                     * This was hardcoded as -2.0*(M-mu), which IS
                     * N*M-2(N-1)*mu at N=2 but not at N=3 -- the 2D relation
                     * used in 3D. Every 3D pressure-output gradient was
                     * therefore scaled by 2(M-mu)/(3M-4mu); at this test
                     * suite's vp/vs/rho that is 0.82051, against a measured
                     * pressure/velocity ratio of 0.82077 (back_prop_type=2)
                     * and 0.82610 (back_prop_type=1) -- see notes/todo.md
                     * item 0d. Being in the shared residual path, it hit
                     * BOTH back_prop_types by the same factor, which is what
                     * localized it here rather than in either gradient.
                     * 2D is algebraically unchanged. */
                    float NDf = (float)m->NDIM;
                    float trmod;
                    /* The adjoint stress source is Psi^0 applied to the misfit
                     * derivative (thesis eq. 3.33), i.e. the UNRELAXED trace
                     * modulus. M/mu as stored are the RELAXED ones: the update
                     * kernels form g = M*(1+L*taup) and f = 2*mu*(1+L*taus)
                     * before using them. Using the stored values here left the
                     * viscoelastic adjoint source short by (1+L*tau) -- exactly
                     * 1 when L==0, which is why every elastic case was correct
                     * and only the viscoelastic gradient drifted with tau. */
                    float ltp = 1.0f, lts = 1.0f;
                    if (m->L > 0){
                        float * taup_a = get_par(m->pars,m->npars,"taup")->gl_par;
                        float * taus_a = get_par(m->pars,m->npars,"taus")->gl_par;
                        if (taup_a) ltp = 1.0f + (float)m->L*taup_a[pos];
                        if (taus_a) lts = 1.0f + (float)m->L*taus_a[pos];
                    }
                    if (m->FP16>1){
                        trmod = NDf*half_to_float( ((half*)par)[pos] )*ltp
                        -2.0f*(NDf-1.0f)*half_to_float( ((half*)par2)[pos] )*lts;
                    }
                    else{
                        trmod = NDf*par[pos]*ltp - 2.0f*(NDf-1.0f)*par2[pos]*lts;
                    }
                    parscal = -trmod*m->dt/m->dh*powf(2,-scaler2);

                    /* No time shift here: since 2026-11-18 the pressure (and
                     * every stress output) is recorded at the state before
                     * the step's stress update (kernel_varout), so its
                     * residual belongs to the same adjoint iteration as the
                     * velocity residual. The one-sample delay that was here
                     * (2026-09-04, todo 0d2) gave the right timing but
                     * dropped the last sample, which the adjoint loop never
                     * reaches: an O(dt) error that depended on how much
                     * signal was left at the end of the record (measured:
                     * 4.8 % on a 256-sample 2-D record with a force source,
                     * 0 with a record twice as long). */
                    for (t=0;t<tmax;t++){
                        m->trans_vars[i].gl_var_res[s][g*NT+t]*=parscal*m->dt;
                    }
                    
                    if (m->FP16>0){
                        for (t=0;t<tmax;t++){
                            if (resmax<fabsf(m->trans_vars[i].gl_var_res[s][g*NT+t])){
                                resmax=fabsf(m->trans_vars[i].gl_var_res[s][g*NT+t]);
                            }
                        }
                    }
                    
                }
            }
        }
    }
    if (m->FP16>0){
        m->src_recs.res_scales[s]=-log2(resmax/10.0);
        #ifndef __NOMPI__
        MPI_Allreduce(&m->src_recs.res_scales[s],
                      &m->src_recs.res_scales[s],
                      1,
                      MPI_INT,
                      MPI_MIN,
                      m->mpigroupcomm);
        #endif
    }
    
    return 0;
}




/* Receiver-cell term of the back_prop_type 2 density gradient (velocity receivers).

   back_prop_type 1 pairs the residual injected at adjoint step t with the forward
   velocity v(t+1), the recorded sample of index t (rec_grad.cl, rec_grad_v:
   gradrip += -v(t+1) res(t)). The DFT correlation of back_prop_type 2 pairs the
   whole adjoint increment of a step, residual included, with v(t): at a receiver
   cell it lacks  -sum_t res(t) [v(t+1) - v(t)] = -sum_t res(t) [out(t) - out(t-1)],
   with out the recorded trace. Measured before this term (FD / <g,dm>, bump on a
   receiver, 0.25 % step): rho 1.18 (P-SV, force source), 1.20 (explosive), 1.13
   (SH); vp, vs exact; back_prop_type 1 exact.
   Units: rec_grad_v adds GSCALE(-v_dev res_dev), with v_dev = out 2^(src_scale -
   par_scale) (kernel_varout) and res_dev = res 2^res_scale, i.e. -out res 2^par_scale;
   back_prop_type 1 then multiplies by dh^2/dt^3 2^par_scale (unscale_grad),
   back_prop_type 2 by 2 dh^2/dt^4 2^par_scale (unscale_grad_dft). So the term is
   stored as -2^par_scale (dt/2) sum_t res(t) [out(t) - out(t-1)], added to the
   device gradient after its readback (add_rec_grad_dft), before unscale_grad_dft.
   Called once per shot after res_scale. Receivers inside a Cerjan layer: the taper
   factor of rec_grad.cl is ignored. */
int rec_grad_dft(model * m, int s)
{
    int state=0;
    int i, g, t, pos, x, y, z;
    int NT=m->NT;
    int nrec=m->src_recs.nrec[s];
    float sc = powf(2.0f, (float)m->par_scale)*m->dt/2.0f;
    if (!(m->GRADOUT==1 && m->BACK_PROP_TYPE==2)) return state;
    for (i=0;i<m->nvars;i++){
        const char * pname = NULL;
        if (!m->vars[i].to_output) continue;
        if (strcmp(m->vars[i].name,"vx")==0) pname = "rip";
        else if (strcmp(m->vars[i].name,"vy")==0) pname = (m->ND==21) ? "rho" : "rjp";
        else if (strcmp(m->vars[i].name,"vz")==0) pname = "rkp";
        else continue;
        parameter * p = get_par(m->pars, m->npars, pname);
        if (!p || !p->to_grad || !p->gl_grad) continue;
        if (!p->gl_gradrec){
            GMALLOC(p->gl_gradrec, sizeof(float)*p->num_ele);
            if (state) return state;
        }
        for (g=0;g<nrec;g++){
            x = m->src_recs.rec_pos[s][0+8*g]/m->dh;
            y = m->src_recs.rec_pos[s][1+8*g]/m->dh;
            z = m->src_recs.rec_pos[s][2+8*g]/m->dh;
            if (m->NDIM==2) pos = x*m->N[0]+z;
            else            pos = x*m->N[0]*m->N[1]+y*m->N[0]+z;
            float * out = &m->vars[i].gl_varout[s][g*NT];
            float * res = &m->vars[i].gl_var_res[s][g*NT];
            double acc = 0.0;
            for (t=m->tmin+1;t<m->tmax;t++){
                acc += (double)res[t]*((double)out[t]-(double)out[t-1]);
            }
            p->gl_gradrec[pos] += (float)(-sc*acc);
        }
    }
    return state;
}

/* Add the receiver-cell terms (rec_grad_dft) to the gradient read back from the devices. */
int add_rec_grad_dft(model * m)
{
    int i, j;
    for (i=0;i<m->npars;i++){
        parameter * p = &m->pars[i];
        if (!p->gl_gradrec || !p->gl_grad) continue;
        for (j=0;j<p->num_ele;j++){
            p->gl_grad[j] += p->gl_gradrec[j];
            p->gl_gradrec[j] = 0.0f;
        }
    }
    return 0;
}


/* ---- Residual scaling on the device (dev_io.cl) ----------------------------
   m->dev_res (CUDA, parameters prepared on the device): res_scale() and
   rec_grad_dft() run as kernels on the first device, reading the parameters
   at the receivers on the devices that hold them. m->dev_data (SeisCL.torch):
   the seismograms and residuals never reach the host either (cl_out_all,
   cl_res_all of the first device). Otherwise (SeisCL_MPI) the host keeps its
   seismograms (written to the output file, used by res_calc) and the residual
   of each shot is uploaded for its scaling. Same results as the host
   functions (bit-identical). */
#ifndef __SEISCL__
static int dio_buf(device * di, clbuf * b, size_t size, int host){
    int state = 0;
    b->size = size;
    if (host){
        GMALLOC(b->host, size);
        b->free_host = 1;
    }
    __GUARD clbuf_create(di->context_ptr, b);
    if (!state) state = cuMemsetD8(b->mem, 0, size);
    return state;
}
static int dio_mkprog(model * m, device * di, clprogram * p, clprogram * mp, size_t gsize, size_t lsize){
    int state = 0;
    *p = *mp;
    __GUARD prog_create(m, di, p);
    p->wdim = 1;
    p->lsize[0] = lsize;
    p->gsize[0] = ((gsize + lsize - 1)/lsize)*lsize;
    return state;
}
/* bind argument name of prog to the device pointer held at mem */
static int dio_arg(clprogram * p, const char * name, void * mem){
    int i;
    for (i=0;i<p->ninputs;i++){
        if (strcmp(p->input_list[i], name)==0){
            prog_arg(p, i, mem, sizeof(MEM));
            return 0;
        }
    }
    fprintf(stderr, "Error: no argument %s in kernel %s\n", name, p->name);
    return 1;
}
static void dio_seti(float * c, int k, int v){ memcpy(&c[k], &v, 4); }
static int dio_row(model * m, int s){
    int r = 0, i;
    for (i=0;i<s;i++) r += m->src_recs.nrec[i];
    return r;
}
static float dio_float(unsigned int u){ float v; memcpy(&v, &u, 4); return v; }
/* the receiver-cell gradient parameter of output variable name (rec_grad_dft) */
static const char * dio_pname(model * m, const char * name){
    if (strcmp(name,"vx")==0) return "rip";
    if (strcmp(name,"vy")==0) return (m->ND==21) ? "rho" : "rjp";
    if (strcmp(name,"vz")==0) return "rkp";
    return NULL;
}
#endif

int dio_create(model * m, device * di){
    int state = 0;
    #ifndef __SEISCL__
    int i;
    int ng = m->src_recs.ngmax;
    size_t nout = (size_t)m->src_recs.allng*m->NT;
    __GUARD dio_buf(di, &di->dio.recpar, sizeof(float)*ng*DIO_NP, 0);
    __GUARD dio_buf(di, &di->dio.cst, sizeof(float)*PREP_NCST, 1);
    __GUARD dio_mkprog(m, di, &di->dio.gather, &m->dio.gather, ng, 64);
    if (di->DEVID!=0) return state;
    __GUARD dio_buf(di, &di->dio.recpar_all, sizeof(float)*ng*DIO_NP*m->NUM_DEVICES, 0);
    __GUARD dio_buf(di, &di->dio.stage, sizeof(float)*ng*m->NT, 0);
    __GUARD dio_buf(di, &di->dio.red, sizeof(unsigned int)*8, 1);
    __GUARD dio_buf(di, &di->dio.acc, sizeof(double)*ng, 0);
    __GUARD dio_mkprog(m, di, &di->dio.res_scale, &m->dio.res_scale, ng, 64);
    __GUARD dio_mkprog(m, di, &di->dio.recgrad, &m->dio.recgrad, ng, 64);
    __GUARD dio_mkprog(m, di, &di->dio.recgrad_add, &m->dio.recgrad_add, 1, 1);
    __GUARD dio_mkprog(m, di, &di->dio.add_out, &m->dio.add_out, ng, 64);
    __GUARD gpost_create(m, di);
    for (i=0;i<m->nvars;i++){
        if (!di->vars[i].to_output) continue;
        __GUARD dio_buf(di, &di->vars[i].cl_out_all, sizeof(float)*nout, 0);
        __GUARD dio_buf(di, &di->vars[i].cl_res_all, sizeof(float)*nout, 0);
    }
    for (i=0;i<m->ntvars;i++){
        if (!di->trans_vars[i].to_output) continue;
        __GUARD dio_buf(di, &di->trans_vars[i].cl_out_all, sizeof(float)*nout, 0);
        __GUARD dio_buf(di, &di->trans_vars[i].cl_res_all, sizeof(float)*nout, 0);
    }
    if (m->GRADOUT && m->BACK_PROP_TYPE==2){
        for (i=0;i<m->nvars;i++){
            const char * pn = di->vars[i].to_output ? dio_pname(m, di->vars[i].name) : NULL;
            if (!pn) continue;
            parameter * p = get_par(di->pars, di->npars, pn);
            if (!p || !p->to_grad || p->cl_gradrec.mem) continue;
            __GUARD dio_buf(di, &p->cl_gradrec,
                            sizeof(float)*get_par(m->pars, m->npars, pn)->num_ele, 0);
        }
    }
    #endif
    return state;
}

/* Zero the seismograms of every shot before a call (reduce_seis adds) */
int dio_reset(model * m, device ** dev){
    int state = 0;
    #ifndef __SEISCL__
    int i;
    if (!m->dev_res) return 0;
    __GUARD cuCtxSetCurrent((*dev)[0].context);
    for (i=0;i<m->nvars;i++){
        if ((*dev)[0].vars[i].cl_out_all.mem)
            __GUARD cuMemsetD8((*dev)[0].vars[i].cl_out_all.mem, 0, (*dev)[0].vars[i].cl_out_all.size);
    }
    for (i=0;i<m->ntvars;i++){
        if ((*dev)[0].trans_vars[i].cl_out_all.mem)
            __GUARD cuMemsetD8((*dev)[0].trans_vars[i].cl_out_all.mem, 0, (*dev)[0].trans_vars[i].cl_out_all.size);
    }
    #endif
    return state;
}

#ifndef __SEISCL__
/* reduce_seis for one variable list (vars or trans_vars) */
static int dio_reduce_list(model * m, device ** dev, int s, int trans){
    int state = 0;
    int d, k;
    device * d0 = &(*dev)[0];
    int nrec = m->src_recs.nrec[s];
    int nv = trans ? m->ntvars : m->nvars;
    float * c = d0->dio.cst.host;
    for (d=0;d<m->NUM_DEVICES;d++){
        device * dd = &(*dev)[d];
        for (k=0;k<nv;k++){
            variable * v = trans ? &dd->trans_vars[k] : &dd->vars[k];
            variable * v0 = trans ? &d0->trans_vars[k] : &d0->vars[k];
            if (!v->to_output) continue;
            void * src = &v->cl_varout.mem;
            if (d>0){
                __GUARD cuCtxSetCurrent(dd->context);
                __GUARD cuMemcpyAsync(d0->dio.stage.mem, v->cl_varout.mem,
                                      sizeof(float)*nrec*m->NT, dd->queue);
                __GUARD cuStreamSynchronize(dd->queue);
                src = &d0->dio.stage.mem;
            }
            memset(c, 0, sizeof(float)*PREP_NCST);
            c[0] = m->dh;
            dio_seti(c, 2, nrec);
            dio_seti(c, 3, m->NT);
            dio_seti(c, 10, dio_row(m, s));
            dio_seti(c, 11, dd->OFFSET);
            dio_seti(c, 12, dd->N[dd->NDIM-1]);
            __GUARD clbuf_send(&d0->queue, &d0->dio.cst);
            __GUARD dio_arg(&d0->dio.add_out, "out", &v0->cl_out_all.mem);
            __GUARD dio_arg(&d0->dio.add_out, "varout", src);
            __GUARD prog_launch(&d0->queue, &d0->dio.add_out);
            if (d>0) __GUARD cuStreamSynchronize(d0->queue);   /* stage is reused */
        }
    }
    return state;
}
#endif

int dio_reduce_seis(model * m, device ** dev, int s){
    int state = 0;
    #ifndef __SEISCL__
    __GUARD dio_reduce_list(m, dev, s, 0);
    __GUARD dio_reduce_list(m, dev, s, 1);
    #endif
    return state;
}

/* res_scale() on the device: residuals of shot s scaled in cl_res_all */
int dio_res_scale(model * m, device ** dev, int s){
    int state = 0;
    #ifndef __SEISCL__
    int d, i;
    device * d0 = &(*dev)[0];
    int nrec = m->src_recs.nrec[s];
    int row = dio_row(m, s);
    int ng = m->src_recs.ngmax;
    size_t bytes = sizeof(float)*nrec*m->NT;
    int scaler = (m->FP16>0) ? m->par_scale : 0;

    /* SeisCL_MPI: the residuals of res_calc (host) to the device */
    if (!m->dev_data){
        __GUARD cuCtxSetCurrent(d0->context);
        for (i=0;i<m->nvars;i++){
            if (!d0->vars[i].to_output) continue;
            __GUARD cuMemcpyHtoD(d0->vars[i].cl_res_all.mem + sizeof(float)*row*m->NT,
                                 m->vars[i].gl_var_res[s], bytes);
        }
        for (i=0;i<m->ntvars;i++){
            if (!d0->trans_vars[i].to_output) continue;
            __GUARD cuMemcpyHtoD(d0->trans_vars[i].cl_res_all.mem + sizeof(float)*row*m->NT,
                                 m->trans_vars[i].gl_var_res[s], bytes);
        }
    }
    /* the parameters at the receivers, from the device that holds each one */
    for (d=0;d<m->NUM_DEVICES;d++){
        device * dd = &(*dev)[d];
        float * c = dd->dio.cst.host;
        memset(c, 0, sizeof(float)*PREP_NCST);
        c[0] = m->dh;
        dio_seti(c, 2, nrec);
        dio_seti(c, 8, ng);
        dio_seti(c, 11, dd->OFFSET);
        dio_seti(c, 12, dd->N[dd->NDIM-1]);
        dio_seti(c, 13, m->N[0]);
        dio_seti(c, 14, m->NDIM==3 ? m->N[1] : 1);
        dio_seti(c, 15, 0);
        __GUARD clbuf_send(&dd->queue, &dd->dio.cst);
        __GUARD prog_launch(&dd->queue, &dd->dio.gather);
        __GUARD cuCtxSetCurrent(dd->context);
        __GUARD cuMemcpyAsync(d0->dio.recpar_all.mem + sizeof(float)*d*ng*DIO_NP,
                              dd->dio.recpar.mem, sizeof(float)*ng*DIO_NP, dd->queue);
        __GUARD cuStreamSynchronize(dd->queue);
    }
    unsigned int * r = (unsigned int *)d0->dio.red.host;
    memset(r, 0, d0->dio.red.size);
    __GUARD clbuf_send(&d0->queue, &d0->dio.red);
    float * c = d0->dio.cst.host;
    for (i=0;i<m->nvars+m->ntvars;i++){
        variable * v0 = (i<m->nvars) ? &d0->vars[i] : &d0->trans_vars[i-m->nvars];
        if (!v0->to_output) continue;
        int kind = -1;
        if (i<m->nvars){
            if (strcmp(v0->name,"vx")==0) kind = 0;
            else if (strcmp(v0->name,"vy")==0) kind = 1;
            else if (strcmp(v0->name,"vz")==0) kind = 2;
        }
        else if (strcmp(v0->name,"p")==0) kind = 3;
        if (kind<0) continue;          /* res_scale leaves the other residuals as they are */
        memset(c, 0, sizeof(float)*PREP_NCST);
        c[0] = m->dh;
        c[1] = m->dt;
        dio_seti(c, 2, nrec);
        dio_seti(c, 3, m->NT);
        dio_seti(c, 4, m->tmax);
        dio_seti(c, 5, scaler);
        dio_seti(c, 6, kind);
        dio_seti(c, 7, m->NUM_DEVICES);
        dio_seti(c, 8, ng);
        dio_seti(c, 9, m->L);
        dio_seti(c, 10, row);
        __GUARD clbuf_send(&d0->queue, &d0->dio.cst);
        __GUARD dio_arg(&d0->dio.res_scale, "res", &v0->cl_res_all.mem);
        __GUARD prog_launch(&d0->queue, &d0->dio.res_scale);
    }
    __GUARD clbuf_read(&d0->queue, &d0->dio.red);
    __GUARD WAITQUEUE(d0->queue);
    if (m->FP16>0){
        float resmax = dio_float(r[0]);
        m->src_recs.res_scales[s]=-log2(resmax/10.0);
        #ifndef __NOMPI__
        MPI_Allreduce(&m->src_recs.res_scales[s],
                      &m->src_recs.res_scales[s],
                      1,
                      MPI_INT,
                      MPI_MIN,
                      m->mpigroupcomm);
        #endif
    }
    /* SeisCL_MPI: the host keeps the scaled residuals, as res_scale() left them */
    if (!m->dev_data){
        __GUARD cuCtxSetCurrent(d0->context);
        for (i=0;i<m->nvars;i++){
            if (!d0->vars[i].to_output) continue;
            __GUARD cuMemcpyDtoH(m->vars[i].gl_var_res[s],
                                 d0->vars[i].cl_res_all.mem + sizeof(float)*row*m->NT, bytes);
        }
        for (i=0;i<m->ntvars;i++){
            if (!d0->trans_vars[i].to_output) continue;
            __GUARD cuMemcpyDtoH(m->trans_vars[i].gl_var_res[s],
                                 d0->trans_vars[i].cl_res_all.mem + sizeof(float)*row*m->NT, bytes);
        }
    }
    #endif
    return state;
}

/* rec_grad_dft() on the device, into cl_gradrec of the first device */
int dio_rec_grad_dft(model * m, device ** dev, int s){
    int state = 0;
    #ifndef __SEISCL__
    int i;
    device * d0 = &(*dev)[0];
    if (!(m->GRADOUT==1 && m->BACK_PROP_TYPE==2)) return state;
    float * c = d0->dio.cst.host;
    for (i=0;i<m->nvars;i++){
        variable * v0 = &d0->vars[i];
        const char * pn = v0->to_output ? dio_pname(m, v0->name) : NULL;
        if (!pn) continue;
        parameter * p = get_par(d0->pars, d0->npars, pn);
        if (!p || !p->to_grad || !p->cl_gradrec.mem) continue;
        memset(c, 0, sizeof(float)*PREP_NCST);
        c[0] = m->dh;
        dio_seti(c, 2, m->src_recs.nrec[s]);
        dio_seti(c, 3, m->NT);
        dio_seti(c, 4, m->tmax);
        dio_seti(c, 10, dio_row(m, s));
        dio_seti(c, 13, m->N[0]);
        dio_seti(c, 14, m->NDIM==3 ? m->N[1] : 1);
        dio_seti(c, 16, m->tmin);
        c[17] = powf(2.0f, (float)m->par_scale)*m->dt/2.0f;
        __GUARD clbuf_send(&d0->queue, &d0->dio.cst);
        __GUARD dio_arg(&d0->dio.recgrad, "out", &v0->cl_out_all.mem);
        __GUARD dio_arg(&d0->dio.recgrad, "res", &v0->cl_res_all.mem);
        __GUARD prog_launch(&d0->queue, &d0->dio.recgrad);
        __GUARD dio_arg(&d0->dio.recgrad_add, "gradrec", &p->cl_gradrec.mem);
        __GUARD prog_launch(&d0->queue, &d0->dio.recgrad_add);
    }
    __GUARD WAITQUEUE(d0->queue);
    #endif
    return state;
}

/* initialize_adj: the scaled residuals of shot s become the adjoint sources */
int dio_res_to_adj(model * m, device ** dev, int s){
    int state = 0;
    #ifndef __SEISCL__
    int d, i;
    device * d0 = &(*dev)[0];
    size_t row = dio_row(m, s);
    size_t bytes = sizeof(float)*m->src_recs.nrec[s]*m->NT;
    __GUARD WAITQUEUE(d0->queue);
    for (d=0;d<m->NUM_DEVICES;d++){
        device * dd = &(*dev)[d];
        __GUARD cuCtxSetCurrent(dd->context);
        for (i=0;i<m->nvars;i++){
            if (!dd->vars[i].to_output) continue;
            __GUARD cuMemcpyAsync(dd->vars[i].cl_varout.mem,
                                  d0->vars[i].cl_res_all.mem + sizeof(float)*row*m->NT,
                                  bytes, dd->queue);
        }
        for (i=0;i<m->ntvars;i++){
            if (!dd->trans_vars[i].to_output) continue;
            __GUARD cuMemcpyAsync(dd->trans_vars[i].cl_varout.mem,
                                  d0->trans_vars[i].cl_res_all.mem + sizeof(float)*row*m->NT,
                                  bytes, dd->queue);
        }
    }
    #endif
    return state;
}

/* Until the gradient post-processing runs on the device (plan
   gpu-param-pipeline, step 3): the receiver-cell terms to the host, where
   add_rec_grad_dft adds them. */
int dio_gradrec_readback(model * m, device ** dev){
    int state = 0;
    #ifndef __SEISCL__
    int i;
    device * d0 = &(*dev)[0];
    __GUARD cuCtxSetCurrent(d0->context);
    for (i=0;i<d0->npars;i++){
        parameter * p = &d0->pars[i];
        if (!p->cl_gradrec.mem) continue;
        parameter * pm = &m->pars[i];
        if (!pm->gl_gradrec){
            GMALLOC(pm->gl_gradrec, sizeof(float)*pm->num_ele);
            if (state) return state;
        }
        __GUARD cuMemcpyDtoH(pm->gl_gradrec, p->cl_gradrec.mem, p->cl_gradrec.size);
        __GUARD cuMemsetD8(p->cl_gradrec.mem, 0, p->cl_gradrec.size);
    }
    #endif
    return state;
}
