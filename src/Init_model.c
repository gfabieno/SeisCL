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
#include "third_party/NVIDIA_FP16/fp16_conversion.h"

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



/* Recompute every host-side parameter value that depends on the raw values
 * the caller supplied: scaling, the per-parameter transforms (e.g. squaring
 * vp/vs into Lame parameters for par_type=0), the stability check, and the
 * FP16 conversions. Allocation-free and idempotent with respect to a fresh
 * set of raw values, so the PyTorch binding (SeisCL/torch/engine_handle.cpp)
 * can call it on its own to refresh a reused engine without going through
 * Init_model()'s allocation. Split out of Init_model() so both callers share
 * one implementation.
 */
int Init_model_values(model * m) {

    int state=0;
    int i,j,t;
    half * hpar;


    __GUARD m->set_par_scale( (void*) m);
    for (i=0;i<m->npars;i++){
        if (m->pars[i].transform !=NULL){
            m->pars[i].transform( (void*) m);
        }
    }
    __GUARD m->check_stability( (void*) m);

    float srcmax;
    if (m->FP16!=0){
        for (i=0;i<m->src_recs.ns;i++){
            srcmax=0;
            for (t=0;t<m->NT*m->src_recs.nsrc[i];t++){
                if (srcmax<fabsf(m->src_recs.src[i][t])){
                    srcmax=fabsf(m->src_recs.src[i][t]);
                }
            }
            /* es of eq. (6) in Fabien-Ouellet (2020), Geophysics 85(3) F65:
               es = -log2(dt*max(s)), which normalizes the injected amplitude
               to ~1 -- but only for a source that goes into a variable stored
               at scaler 0, i.e. a stress. Every variable is stored as
               physical*2^(src_scale - scaler) (see kernel_varout), so a source
               injected into vx/vy/vz, whose scaler is par_scale = ev, lands as
               ~2^(-ev) instead of ~1. With ev ~ -33 that is ~2^33, far past
               half's 65504, and FP16=2 NaN'd for ANY force-source amplitude.
               Offsetting es by the target variable's scaler makes the stored
               increment ~1 for both kinds. varout and the gradient descaling
               (2*par_scale - src_scale - res_scale, eq. 11) both read the same
               src_scale, so they stay consistent.
               A shot mixing stress and velocity sources cannot satisfy both
               with one scaler; the velocity one is taken, since overflowing is
               fatal and the stress merely loses precision. */
            int vsc = 0;
            for (j=0;j<m->src_recs.nsrc[i];j++){
                int st = (int)m->src_recs.src_pos[i][4+5*j];
                const char * vn = (st==0) ? "vx" : (st==1) ? "vy"
                                          : (st==2) ? "vz" : NULL;
                if (vn){
                    variable * sv = get_var(m->vars, m->nvars, vn);
                    if (sv && abs(sv->scaler)>abs(vsc)) vsc = sv->scaler;
                }
            }
            m->src_recs.src_scales[i]=-log2(srcmax*m->dt*1.0) + vsc;
        }
    }
    
    if (m->FP16>1){
        for (i=0;i<m->npars;i++){
            hpar = (half*)m->pars[i].gl_par;
            for (j=0;j<m->pars[i].num_ele;j++){
                hpar[j] =float_to_half_full_rtne(m->pars[i].gl_par[j]);
            }

        }
    }
    
    if (m->FP16==1 && m->halfpar>0){
        for (i=0;i<m->npars;i++){
            for (j=0;j<m->pars[i].num_ele;j++){
                m->pars[i].gl_par[j] =half_to_float(float_to_half_full_rtne(m->pars[i].gl_par[j]));
            }

        }
    }

    return state;

}

int Init_model(model * m) {

    int state=0;

    GMALLOC(m->src_recs.src_scales, sizeof(int)*m->src_recs.ns);
    if (state) return state;

    state = Init_model_values(m);

    #ifndef __NOMPI__
    if (state && m->MPI_INIT==1)
        MPI_Bcast( &state, 1, MPI_INT, m->GID, MPI_COMM_WORLD );
    #endif

    return state;

}
