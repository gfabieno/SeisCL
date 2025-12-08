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

static size_t idx3d(int x, int y, int z, int ny, int nz, int fdoh) {
    return (size_t)(x - fdoh) * (size_t)(ny - 2 * fdoh) * (size_t)(nz - 2 * fdoh)
         + (size_t)(y - fdoh) * (size_t)(nz - 2 * fdoh)
         + (size_t)(z - fdoh);
}

static size_t idx2d(int x, int z, int nz, int fdoh) {
    return (size_t)(x - fdoh) * (size_t)(nz - 2 * fdoh) + (size_t)(z - fdoh);
}

int scale_sources(model *m) {

    int state = 0;
    int s, src, t;
    int fdoh = m->FDOH;
    float dh = m->dh;
    float dt = m->dt;
    int nd = m->NDIM;
    int nz = m->N[0];
    int ny = nd == 3 ? m->N[1] : 1;
    float vol_scale = powf(dh, nd - 1);

    parameter *par = NULL;
    float *rip = NULL, *rjp = NULL, *rkp = NULL, *mu = NULL, *M = NULL;

    par = get_par(m->pars, m->npars, "rip");
    if (par)
        rip = par->gl_par;
    par = get_par(m->pars, m->npars, "rjp");
    if (par)
        rjp = par->gl_par;
    par = get_par(m->pars, m->npars, "rkp");
    if (par)
        rkp = par->gl_par;
    par = get_par(m->pars, m->npars, "mu");
    if (par)
        mu = par->gl_par;
    par = get_par(m->pars, m->npars, "M");
    if (par)
        M = par->gl_par;

    int original_total = m->src_recs.allns;
    int total = 0;
    int *new_nsrc = NULL;
    int *expanded_counts = NULL;
    size_t *pos_offsets = NULL;
    size_t *src_offsets = NULL;

    GMALLOC(new_nsrc, sizeof(int) * m->src_recs.ns);
    GMALLOC(expanded_counts, sizeof(int) * original_total);
    GMALLOC(pos_offsets, sizeof(size_t) * m->src_recs.ns);
    GMALLOC(src_offsets, sizeof(size_t) * m->src_recs.ns);

    for (s = 0; s < m->src_recs.ns; s++) {
        pos_offsets[s] = (size_t)(m->src_recs.src_pos[s] - m->src_recs.src_pos[0]);
        src_offsets[s] = (size_t)(m->src_recs.src[s] - m->src_recs.src[0]);
        new_nsrc[s] = 0;
    }

    int global_idx = 0;
    for (s = 0; s < m->src_recs.ns; s++) {
        for (src = 0; src < m->src_recs.nsrc[s]; src++) {

            float *pos = m->src_recs.src_pos[0] + pos_offsets[s] + src * 5;
            int type = (int)pos[4];

            int count = 1;
            if (type == 10 || type == 11) {
                count = nd == 3 ? 3 : 2;
                if (mu == NULL || M == NULL)
                    count = 1;
            }
            else if (type == 12) {
                if (mu == NULL || M == NULL)
                    count = 1;
            }

            expanded_counts[global_idx++] = count;
            new_nsrc[s] += count;
            total += count;
        }
    }

    int capacity = original_total;
    if (total > capacity)
        capacity = total;

    float *pos_out = m->src_recs.src_pos[0];
    float *src_out = m->src_recs.src[0];

    if (capacity != original_total) {
        pos_out = realloc(pos_out, sizeof(float) * capacity * 5);
        src_out = realloc(src_out, sizeof(float) * capacity * m->NT);
    }

    int write_index = total;
    global_idx = original_total;
    for (s = m->src_recs.ns - 1; s >= 0; s--) {
        for (src = m->src_recs.nsrc[s] - 1; src >= 0; src--) {

            float *pos = pos_out + pos_offsets[s] + src * 5;
            float *signal = src_out + src_offsets[s] + src * m->NT;
            int type = (int)pos[4];
            int count = expanded_counts[--global_idx];
            write_index -= count;
            float *pos_dst = pos_out + write_index * 5;
            float *src_dst = src_out + write_index * m->NT;

            int i = (int)(pos[0] / dh) + fdoh;
            int j = nd == 3 ? (int)(pos[1] / dh) + fdoh : fdoh;
            int k = (int)(pos[2] / dh) + fdoh;

            size_t idx = nd == 3 ? idx3d(i, j, k, ny, nz, fdoh) : idx2d(i, k, nz, fdoh);

            if (type >= 0 && type <= 2) {
                float inv_rho = 0.0f;
                if (type == 0 && rip)
                    inv_rho = rip[idx];
                else if (type == 1 && nd == 3 && rjp)
                    inv_rho = rjp[idx];
                else if (type == 2 && rkp)
                    inv_rho = rkp[idx];

                if (inv_rho == 0.0f && rip)
                    inv_rho = rip[idx];

                float factor = inv_rho / (dt * vol_scale);

                memcpy(pos_dst, pos, sizeof(float) * 5);
                pos_dst[4] = (float)type;

                for (t = 0; t < m->NT; t++) {
                    src_dst[t] = signal[t] * factor;
                }
            }
            else if (type == 10 || type == 11 || type == 12) {
                if (mu == NULL || M == NULL) {
                    int fallback_type = type == 10 ? 4 : (type == 11 ? 6 : 7);
                    memcpy(pos_dst, pos, sizeof(float) * 5);
                    pos_dst[4] = (float)fallback_type;
                    memcpy(src_dst, signal, sizeof(float) * m->NT);
                    continue;
                }

                float mu_val = mu[idx];
                float lambda = M[idx] - 2.0f * mu_val;

                if (type == 10 || type == 11) {
                    float *primary = src_dst;
                    float *coupled = src_dst + m->NT;
                    float *syy = nd == 3 ? src_dst + 2 * m->NT : NULL;

                    for (t = 0; t < m->NT; t++) {
                        if (type == 10) {
                            primary[t] = signal[t] * M[idx];
                            coupled[t] = signal[t] * lambda;
                        } else {
                            primary[t] = signal[t] * lambda;
                            coupled[t] = signal[t] * M[idx];
                        }
                        if (nd == 3)
                            syy[t] = signal[t] * lambda;
                    }

                    memcpy(pos_dst, pos, sizeof(float) * 5);
                    pos_dst[4] = (float)(type == 10 ? 4 : 6);

                    memcpy(pos_dst + 5, pos, sizeof(float) * 5);
                    pos_dst[9] = (float)(type == 10 ? 6 : 4);

                    if (nd == 3) {
                        memcpy(pos_dst + 10, pos, sizeof(float) * 5);
                        pos_dst[14] = 5.0f;
                    }
                }
                else if (type == 12) {
                    for (t = 0; t < m->NT; t++) {
                        src_dst[t] = signal[t] * mu_val;
                    }

                    memcpy(pos_dst, pos, sizeof(float) * 5);
                    pos_dst[4] = 7.0f;
                }
            }
            else {
                memcpy(pos_dst, pos, sizeof(float) * 5);
                pos_dst[4] = (float)type;
                memcpy(src_dst, signal, sizeof(float) * m->NT);
            }

        }
    }

    m->src_recs.allns = total;
    m->src_recs.src_pos[0] = pos_out;
    m->src_recs.src[0] = src_out;

    size_t offset = 0;
    for (s = 0; s < m->src_recs.ns; s++) {
        m->src_recs.nsrc[s] = new_nsrc[s];
        m->src_recs.src_pos[s] = m->src_recs.src_pos[0] + offset * 5;
        m->src_recs.src[s] = m->src_recs.src[0] + offset * m->NT;
        offset += new_nsrc[s];
    }

    free(new_nsrc);
    free(expanded_counts);
    free(pos_offsets);
    free(src_offsets);

    return state;
}

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



int Init_model(model * m) {

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

    /*
     * Scale and expand the sources so that user-provided amplitudes correspond
     * to physical quantities. Point-force signals are interpreted as Newtons
     * and are internally converted to acceleration by dividing by
     * (rho * cell_volume). Moment-tensor entries (provided as strain-rates)
     * are converted to stress-rate contributions using the local elastic
     * moduli.
     */
    __GUARD scale_sources(m);

    GMALLOC(m->src_recs.src_scales, sizeof(int)*m->src_recs.ns);
    float srcmax;
    if (m->FP16!=0){
        //TODO review scaler constant
        for (i=0;i<m->src_recs.ns;i++){
            srcmax=0;
                for (t=0;t<m->NT*m->src_recs.nsrc[i];t++){
                    if (srcmax<fabsf(m->src_recs.src[i][t])){
                        srcmax=fabsf(m->src_recs.src[i][t]);
                    }
                    m->src_recs.src_scales[i]=-log2(srcmax*m->dt*1.0);
            }
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
    
   
    #ifndef __NOMPI__
    if (state && m->MPI_INIT==1)
        MPI_Bcast( &state, 1, MPI_INT, m->GID, MPI_COMM_WORLD );
    #endif
    
    return state;

}
