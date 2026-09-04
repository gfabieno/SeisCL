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
                    float scaler2 = (m->BACK_PROP_TYPE==1) ? 2.0f*scaler : (float)scaler;
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
                    if (m->FP16>1){
                        trmod = NDf*half_to_float( ((half*)par)[pos] )
                        -2.0f*(NDf-1.0f)*half_to_float( ((half*)par2)[pos] );
                    }
                    else{
                        trmod = NDf*par[pos] - 2.0f*(NDf-1.0f)*par2[pos];
                    }
                    parscal = -trmod*m->dt/m->dh*powf(2,-scaler2);

                    /* Delay the pressure adjoint source by one sample.
                     *
                     * The forward loop records a seismogram AFTER the whole
                     * update, but the two outputs are produced at different
                     * points of it: vx/vy/vz by update_v and the normal
                     * stresses (hence "p") by update_s, which runs second and
                     * consumes the velocities update_v just wrote. The
                     * adjoint runs the pair in the reverse order, so the
                     * stress half of the adjoint source belongs one sample
                     * later than the velocity half. It was injected at the
                     * same index as velocity, i.e. one sample early.
                     *
                     * Measured on the 2D elastic back_prop_type=1 FD check
                     * (which must give exactly 1), shifting the pressure
                     * residual by k samples:
                     *     k=0    vp 0.99316  vs 1.06221  rho 1.01561
                     *     k=1    vp 0.99998  vs 1.00010  rho 1.00021
                     *     k=2    vp 1.00122  vs 0.93256  rho 0.97999
                     * i.e. the error changes sign between k=1 and k=2 and is
                     * a clean zero at k=1, landing on the velocity channel's
                     * own accuracy. Applying the same shift to the VELOCITY
                     * residual instead breaks it (vs 0.99994 -> 1.03563),
                     * confirming the offset is specific to the stress half.
                     * Before this, the defect showed up as an error
                     * proportional to dt in every pressure-output gradient
                     * (vs: 0.062/0.033/0.017 at dt = 0.8/0.4/0.2 ms) -- see
                     * notes/todo.md item 0d2.
                     *
                     * Done here rather than in the injection kernel because
                     * it is a property of the residual's time indexing, not
                     * of where the kernel is launched: injecting at a
                     * different point of the adjoint iteration provably
                     * changes nothing, since update_adjs only accumulates
                     * into the adjoint stress and never reads it. */
                    for (t=tmax-1;t>0;t--){
                        m->trans_vars[i].gl_var_res[s][g*NT+t] =
                            m->trans_vars[i].gl_var_res[s][g*NT+t-1];
                    }
                    m->trans_vars[i].gl_var_res[s][g*NT] = 0.0f;

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


