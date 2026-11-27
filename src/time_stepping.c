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

/*Main part of the program. Perform FD time stepping */

#include "F.h"
#include "third_party/NVIDIA_FP16/fp16_conversion.h"

int reduce_seis(model * m, device ** dev, int s){
    // Transfer the variables to output to host and reduce in global buffer
    int state=0;
    int posx, i, j, k, d;
    
    // dev_res: the seismograms of the shot on the first device (residuals.c);
    // dev_data: only there
    if (m->dev_res){
        __GUARD dio_reduce_seis(m, dev, s);
        if (m->dev_data) return state;
    }
    // Transfer the seismogram from GPUs to host
    for (d=0;d<m->NUM_DEVICES;d++){
        for (i=0;i<m->nvars;i++){
            if ( (*dev)[d].vars[i].to_output){
                (*dev)[d].vars[i].cl_varout.size=sizeof(float)
                * m->NT * m->src_recs.nrec[s];
                __GUARD clbuf_read( &(*dev)[d].queue,
                                   &(*dev)[d].vars[i].cl_varout);
            }
        }
        for (i=0;i<m->ntvars;i++){
            if ( (*dev)[d].trans_vars[i].to_output){
                (*dev)[d].trans_vars[i].cl_varout.size=sizeof(float)
                * m->NT * m->src_recs.nrec[s];
                __GUARD clbuf_read( &(*dev)[d].queue,
                                   &(*dev)[d].trans_vars[i].cl_varout);
            }
            
        }
    }
    
    // Put them in the global buffer that collect all sources and receivers data
    // from all devices. For all MPI processes, it is reduced at the end of
    // program
    for (d=0;d<m->NUM_DEVICES;d++){
        __GUARD WAITQUEUE((*dev)[d].queue);
        for (k=0;k<(*dev)[d].nvars;k++){
            if ((*dev)[d].vars[k].to_output){
                for ( i=0;i<(*dev)[d].src_recs.nrec[s];i++){
                    posx=(int)floor((*dev)[d].src_recs.rec_pos[s][8*i]/m->dh);
                    if (posx>=(*dev)[d].OFFSET
                        && posx<((*dev)[d].OFFSET+(*dev)[d].N[(*dev)[d].NDIM-1])){
                        
                        for (j=0;j<m->NT;j++){
                            (*dev)[d].vars[k].gl_varout[s][i*m->NT+j]+=
                            (*dev)[d].vars[k].cl_varout.host[i*m->NT+j];
                        }
                    }
                }
            }
        }
        for (k=0;k<(*dev)[d].ntvars;k++){
            if ((*dev)[d].trans_vars[k].to_output){
                for ( i=0;i<(*dev)[d].src_recs.nrec[s];i++){
                    posx=(int)floor((*dev)[d].src_recs.rec_pos[s][8*i]/m->dh);
                    if (posx>=(*dev)[d].OFFSET
                        && posx<((*dev)[d].OFFSET+(*dev)[d].N[(*dev)[d].NDIM-1])){
                        
                        for (j=0;j<m->NT;j++){
                            (*dev)[d].trans_vars[k].gl_varout[s][i*m->NT+j]+=
                            (*dev)[d].trans_vars[k].cl_varout.host[i*m->NT+j];
                        }
                    }
                }
            }
        }
    }
    
    return state;
    
}

/* Debug facility (DFTOUT): dump the raw forward and adjoint DFT wavefield
 * buffers that savefreqs accumulated, plus the derived DFT parameters, so the
 * spectra can be validated against a reference DFT without involving the
 * gradient correlation at all. This is what makes the savefreqs half of
 * BACK_PROP_TYPE==2 testable independently.
 *
 * Layout written, C order: (NFREQS, NX+FDORDER, NZ+FDORDER, 2) in 2D and
 * (NFREQS, NX+FDORDER, NY+FDORDER, NZ+FDORDER, 2) in 3D, matching the
 * f*num_ele + padded-flat indexing that savefreqs and grad_dft*.cl use. The
 * trailing 2 is the interleaved (real, imag) float2 pair.
 *
 * Single device, single shot only: the buffers are per device and are reset
 * between shots, so anything else would need a reduction that would defeat the
 * purpose of dumping raw values. */
static int dump_dft(model * m, device ** dev, struct filenames files){

    int state=0;
    int i, nd;
    char name[100];
    hsize_t dims[MAX_DIMS];
    hid_t file_id=0;
    device * di = &(*dev)[0];

    if (m->NUM_DEVICES!=1 || m->NLOCALP!=1){
        fprintf(stderr,"Error: dftout requires a single device and a single "
                       "MPI process per shot (debug facility)\n");
        return 1;
    }

    file_id = create_file(files.dftout);
    if (!file_id) return 1;

    /* Derived DFT parameters, so a test cannot silently drift from the C code
     * that produced the spectra. */
    dims[0]=1; dims[1]=1;
    float tmp;
    tmp=(float)m->DTNYQ;  writetomat(&file_id, "/DTNYQ",  &tmp, 2, dims);
    tmp=(float)m->NTNYQ;  writetomat(&file_id, "/NTNYQ",  &tmp, 2, dims);
    tmp=(float)m->NFREQS; writetomat(&file_id, "/NFREQS", &tmp, 2, dims);
    tmp=(float)m->FDORDER;writetomat(&file_id, "/FDORDER",&tmp, 2, dims);
    tmp=(float)m->tmin;   writetomat(&file_id, "/tminind",&tmp, 2, dims);
    tmp=(float)m->tmax;   writetomat(&file_id, "/tmaxind",&tmp, 2, dims);
    dims[0]=1; dims[1]=m->NFREQS;
    writetomat(&file_id, "/gradfreqsn",
               get_cst(m->csts, m->ncsts, "gradfreqsn")->gl_cst, 2, dims);

    /* The spectra. nd counts the padded spatial axes plus the complex pair. */
    int npad=1, a;
    for (a=0;a<m->NDIM;a++) npad*=di->N[a]+m->FDORDER;
    for (i=0;i<di->nvars;i++){
        if (!di->vars[i].for_grad) continue;
        /* Wavefields have num_ele==npad; the viscoelastic memory variables are
         * sized npad*L, so give them an extra leading axis rather than
         * mis-shaping them. */
        int nrep=di->vars[i].num_ele/npad;
        if (nrep<1 || nrep*npad!=di->vars[i].num_ele){
            fprintf(stderr,"Error: dftout: unexpected num_ele %d for %s "
                           "(npad %d)\n",
                    di->vars[i].num_ele, di->vars[i].name, npad);
            state=1;
            break;
        }
        nd=0;
        dims[nd++]=m->NFREQS;
        if (nrep>1) dims[nd++]=nrep;
        /* di->N is (NZ, NX) in 2D and (NZ, NY, NX) in 3D; the padded flat
         * index runs slowest on the last axis, so emit them reversed. */
        for (a=m->NDIM-1;a>=0;a--){
            dims[nd++]=di->N[a]+m->FDORDER;
        }
        dims[nd++]=2;
        sprintf(name, "/dft_f_%s", di->vars[i].name);
        writetomat(&file_id, name, di->vars[i].cl_fvar.host, nd, dims);
        sprintf(name, "/dft_a_%s", di->vars[i].name);
        writetomat(&file_id, name, di->vars[i].cl_fvar_adj.host, nd, dims);
    }

    if (file_id) H5Fclose(file_id);

    return state;
}

/* Per-shot store (CKPT_SHOTSTORE), defined below checkpoint_flush(). */
static int shotstore_slot(model * m, int s, int i);
static void shotstore_swap_bnd(model * m, device ** dev, int s);
static void shotstore_swap_var(model * m, device ** dev, int s);

int checkpoint_d2h(model * m, device ** dev, hid_t file_id, int s){
    int state=0;
    int d, i, j;
    char name[100];
    hsize_t dims[MAX_DIMS];

    for (d=0;d<m->NUM_DEVICES;d++){
        WAITQUEUE((*dev)[d].queue);
        /* The boundary of each time step is copied to the pinned host buffer
           cl_varbnd.host on the second queue (save_bnd: clbuf_readto on
           queuecomm). Wait for those copies before writing the buffer below:
           otherwise the last time steps of a shot can be written before they
           arrive, and the checkpoint keeps the previous shot's values there.
           Only the last shot survived that (its copies complete before the
           adjoint pass reads the buffer), so every other shot of a
           back_prop_type 1 gradient was wrong (seen in 3-D, 2026-11-11). */
        if ((*dev)[d].queuecomm) WAITQUEUE((*dev)[d].queuecomm);
        for (i=0;i<m->nvars;i++){
            if ((*dev)[d].vars[i].cl_buf1.host){
                dims[0] = (*dev)[d].vars[i].cl_buf1.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf1h", s, d, (*dev)[d].vars[i].name);
                if (!m->SKIP_CHECKPOINT_FILE)
                    writetomat_nocomp(&file_id, name, (*dev)[d].vars[i].cl_buf1.host,
                           1, dims);
            }
            if ((*dev)[d].vars[i].cl_buf2.host){
                dims[0] = (*dev)[d].vars[i].cl_buf2.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf2h", s, d, (*dev)[d].vars[i].name);
                if (!m->SKIP_CHECKPOINT_FILE)
                    writetomat_nocomp(&file_id, name, (*dev)[d].vars[i].cl_buf2.host,
                           1, dims);
            }
        }
    }

    /* CKPT_SHOTSTORE: read the final wavefield straight into shot s's own
       buffer (no file is written: SKIP_CHECKPOINT_FILE is set with it). */
    if (m->CKPT_SHOTSTORE) shotstore_swap_var(m, dev, s);

    // Transfer all variables to output to host for this time step
    for (d=0;d<m->NUM_DEVICES;d++){
        for (i=0;i<m->nvars;i++){
            __GUARD clbuf_read(&(*dev)[d].queue, &(*dev)[d].vars[i].cl_var);
            if ((*dev)[d].vars[i].cl_buf1.host){
                __GUARD clbuf_read(&(*dev)[d].queue,
                                   &(*dev)[d].vars[i].cl_buf1);
            }
            if ((*dev)[d].vars[i].cl_buf2.host){
                __GUARD clbuf_read(&(*dev)[d].queue,
                                   &(*dev)[d].vars[i].cl_buf2);
            }
        }
    }

    for (d=0;d<m->NUM_DEVICES;d++){
        WAITQUEUE((*dev)[d].queue);
        for (i=0;i<m->nvars;i++){

            if ((*dev)[d].vars[i].cl_var.host){
                dims[0] = (*dev)[d].vars[i].num_ele;
                sprintf(name, "src%d_dev%d_%s", s, d, (*dev)[d].vars[i].name);
                if (!m->SKIP_CHECKPOINT_FILE)
                    writetomat_nocomp(&file_id, name, (*dev)[d].vars[i].cl_var.host,
                           1, dims);
            }

            if ((*dev)[d].vars[i].cl_varbnd.host) {
                dims[0] = (*dev)[d].vars[i].cl_varbnd.sizepin / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_bnd", s, d,
                        (*dev)[d].vars[i].name);
                if (!m->SKIP_CHECKPOINT_FILE)
                    writetomat_nocomp(&file_id, name, (*dev)[d].vars[i].cl_varbnd.host,
                           1, dims);
            }

            if ((*dev)[d].vars[i].cl_buf1.host){
                dims[0] = (*dev)[d].vars[i].cl_buf1.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf1d", s, d,
                        (*dev)[d].vars[i].name);
                if (!m->SKIP_CHECKPOINT_FILE)
                    writetomat_nocomp(&file_id, name, (*dev)[d].vars[i].cl_buf1.host,
                           1, dims);
            }
            if ((*dev)[d].vars[i].cl_buf2.host){
                dims[0] = (*dev)[d].vars[i].cl_buf2.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf2d", s, d,
                        (*dev)[d].vars[i].name);
                if (!m->SKIP_CHECKPOINT_FILE)
                    writetomat_nocomp(&file_id, name, (*dev)[d].vars[i].cl_buf2.host,
                           1, dims);
            }
        }
    }

    /* The reads above were waited for (WAITQUEUE): give the engine its own
       buffer back. */
    if (m->CKPT_SHOTSTORE) shotstore_swap_var(m, dev, s);

    return state;
}

/* Write the boundary checkpoint that a SKIP_CHECKPOINT_FILE forward pass
 * deliberately did not write, for the single shot currently resident in the
 * device/host buffers. Lets a caller that skipped the file on the assumption
 * its own backward pass would follow immediately change its mind -- e.g.
 * because another forward is about to overwrite those buffers -- without
 * having to redo the forward pass. Single-shot only, for the same reason the
 * skip itself is: checkpoint_d2h() runs per shot inside the shot loop, so
 * only the last shot's wavefield is ever in memory.
 */
static int checkpoint_flush_store(model * m, device ** dev, const char * path) {
    int s, i, k;
    char name[100];
    hsize_t dims[MAX_DIMS];
    hid_t file_id;

    remove(path);
    file_id = create_file(path);
    if (file_id < 0) return 1;
    for (s = m->src_recs.smin; s < m->src_recs.smax; s++) {
        for (i = 0; i < m->nvars; i++) {
            k = shotstore_slot(m, s, i);
            if (m->CKPT_STORE_VAR[k]) {
                dims[0] = (*dev)[0].vars[i].num_ele;
                sprintf(name, "src%d_dev%d_%s", s, 0, (*dev)[0].vars[i].name);
                writetomat_nocomp(&file_id, name, m->CKPT_STORE_VAR[k], 1, dims);
            }
            if (m->CKPT_STORE_BND[k]) {
                dims[0] = (*dev)[0].vars[i].cl_varbnd.sizepin / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_bnd", s, 0, (*dev)[0].vars[i].name);
                writetomat_nocomp(&file_id, name, m->CKPT_STORE_BND[k], 1, dims);
            }
        }
    }
    H5Fclose(file_id);
    return 0;
}

int checkpoint_flush(model * m, device ** dev, const char * path) {

    int state = 0;
    int skip;
    hid_t file_id;

    /* Several shots in the per-shot store: write each one from its own
       buffers, under the names checkpoint_d2h() uses, without touching the
       device (it holds the last shot only). */
    if (m->CKPT_SHOTSTORE && m->CKPT_STORE_VAR)
        return checkpoint_flush_store(m, dev, path);

    if (m->src_recs.smax - m->src_recs.smin != 1) return 1;

    remove(path);
    file_id = create_file(path);
    if (file_id < 0) return 1;

    /* checkpoint_d2h() honours SKIP_CHECKPOINT_FILE, which is still set from
     * the forward pass that chose to keep the checkpoint in memory. Clear it
     * for the duration, or we would create the file and write nothing to it. */
    skip = m->SKIP_CHECKPOINT_FILE;
    m->SKIP_CHECKPOINT_FILE = 0;
    state = checkpoint_d2h(m, dev, file_id, m->src_recs.smin);
    m->SKIP_CHECKPOINT_FILE = skip;

    H5Fclose(file_id);

    return state;
}

/* ---- Per-shot host buffers (CKPT_SHOTSTORE, see F.h) --------------------
 * Several shots in one SeisCL.torch gradient: the forward call and the
 * backward call are two time_stepping() runs, and every shot's final
 * wavefield and boundary history must survive from the first to the second.
 * The engine's host buffers hold one shot, so they used to be copied into an
 * HDF5 file after each shot and back before each adjoint shot (~1.5 GB/s, the
 * GPU idle meanwhile). With the store, shot s simply uses its own host
 * buffers: shotstore_swap() exchanges the host pointer of cl_varbnd (for the
 * whole shot) or cl_var (inside checkpoint_d2h/h2d) with shot s's slot.
 * A swap is its own inverse, so the same call puts the engine's own buffer
 * back. Only pointers move: the asynchronous copies already queued keep the
 * address they were given. One device (NUM_DEVICES == 1, enforced by the
 * caller), so dev[0] only. */
static int shotstore_slot(model * m, int s, int i) {
    return (s - m->src_recs.smin) * m->nvars + i;
}

static void shotstore_swap_bnd(model * m, device ** dev, int s) {
    int i;
    float * tmp;
    if (!m->CKPT_STORE_BND) return;      /* allocation failed: state is set */
    for (i = 0; i < m->nvars; i++) {
        float ** slot = &m->CKPT_STORE_BND[shotstore_slot(m, s, i)];
        if (!*slot) continue;
        tmp = (*dev)[0].vars[i].cl_varbnd.host;
        (*dev)[0].vars[i].cl_varbnd.host = *slot;
        *slot = tmp;
    }
}

static void shotstore_swap_var(model * m, device ** dev, int s) {
    int i;
    float * tmp;
    if (!m->CKPT_STORE_VAR) return;      /* allocation failed: state is set */
    for (i = 0; i < m->nvars; i++) {
        float ** slot = &m->CKPT_STORE_VAR[shotstore_slot(m, s, i)];
        if (!*slot) continue;
        tmp = (*dev)[0].vars[i].cl_var.host;
        (*dev)[0].vars[i].cl_var.host = *slot;
        *slot = tmp;
    }
}

void checkpoint_store_free(model * m, device * dev) {
    int k, n;
    if (!m->CKPT_STORE_VAR && !m->CKPT_STORE_BND) return;
    n = m->CKPT_STORE_NS * m->nvars;
    #ifndef __SEISCL__
    /* Pinned memory is freed in the context that allocated it. */
    if (dev && dev[0].context) cuCtxSetCurrent(dev[0].context);
    #endif
    for (k = 0; k < n; k++) {
        if (m->CKPT_STORE_VAR && m->CKPT_STORE_VAR[k]) free(m->CKPT_STORE_VAR[k]);
        if (m->CKPT_STORE_BND && m->CKPT_STORE_BND[k]) {
            #ifdef __SEISCL__
            free(m->CKPT_STORE_BND[k]);
            #else
            cuMemFreeHost(m->CKPT_STORE_BND[k]);
            #endif
        }
    }
    free(m->CKPT_STORE_VAR);
    free(m->CKPT_STORE_BND);
    m->CKPT_STORE_VAR = NULL;
    m->CKPT_STORE_BND = NULL;
    m->CKPT_STORE_NS = 0;
}

int checkpoint_store_alloc(model * m, device * dev) {
    int ns = m->src_recs.smax - m->src_recs.smin;
    int k, i, n;
    if (m->CKPT_STORE_VAR && m->CKPT_STORE_NS == ns) return 0;   /* kept from an earlier call */
    checkpoint_store_free(m, dev);
    if (m->NUM_DEVICES != 1 || ns < 1) return 1;
    #ifndef __SEISCL__
    if (dev[0].context) cuCtxSetCurrent(dev[0].context);
    #endif
    n = ns * m->nvars;
    m->CKPT_STORE_VAR = calloc(n, sizeof(float *));
    m->CKPT_STORE_BND = calloc(n, sizeof(float *));
    m->CKPT_STORE_NS = ns;
    if (!m->CKPT_STORE_VAR || !m->CKPT_STORE_BND) {
        checkpoint_store_free(m, dev);
        return 1;
    }
    for (k = 0; k < ns; k++) {
        for (i = 0; i < m->nvars; i++) {
            variable * v = &dev[0].vars[i];
            /* buf1/buf2 exist only with several devices: not handled here. */
            if (v->cl_buf1.host || v->cl_buf2.host) {
                checkpoint_store_free(m, dev);
                return 1;
            }
            if (v->cl_var.host) {
                /* checkpoint_d2h/h2d move cl_var.size bytes (half that with
                   FP16 storage); the engine's own buffer is num_ele floats. */
                m->CKPT_STORE_VAR[k * m->nvars + i] = malloc(sizeof(float) * v->num_ele);
                if (!m->CKPT_STORE_VAR[k * m->nvars + i]) {
                    checkpoint_store_free(m, dev);
                    return 1;
                }
            }
            if (v->cl_varbnd.host) {
                void * p = NULL;
                #ifdef __SEISCL__
                p = malloc(v->cl_varbnd.sizepin);
                #else
                if (cuMemAllocHost(&p, v->cl_varbnd.sizepin) != CUDA_SUCCESS) p = NULL;
                #endif
                if (!p) {
                    checkpoint_store_free(m, dev);
                    return 1;
                }
                m->CKPT_STORE_BND[k * m->nvars + i] = p;
            }
        }
    }
    return 0;
}

int checkpoint_h2d(model * m, device ** dev, hid_t file_id, int s) {
    int state = 0;
    int d, i;
    char name[100];
    hsize_t dims[MAX_DIMS];
    int nwait;
    EVENT * event;

    /* The host buffers read below (cl_var.host, and above all the pinned
       boundary buffer cl_varbnd.host) are the sources of asynchronous copies
       queued by the previous shot's adjoint pass (inject_bnd: clbuf_sendfrom
       on queue). Overwriting them before those copies have run gave the last
       time steps of every shot but the last the boundaries of the NEXT shot:
       a wrong back_prop_type 1 gradient for multi-shot runs, depending on the
       timing (large in 3-D; 2026-11-11). Wait for both queues first. */
    for (d = 0; d < m->NUM_DEVICES; d++) {
        WAITQUEUE((*dev)[d].queue);
        if ((*dev)[d].queuecomm) WAITQUEUE((*dev)[d].queuecomm);
    }
    /* CKPT_SHOTSTORE: send shot s's final wavefield from its own buffer
       (nothing is read from a file: SKIP_CHECKPOINT_FILE is set with it). */
    if (m->CKPT_SHOTSTORE) shotstore_swap_var(m, dev, s);

    // Transfer all variables to output to host for this time step
    for (d = 0; d < m->NUM_DEVICES; d++) {
        for (i = 0; i < m->nvars; i++) {

            if ((*dev)[d].vars[i].cl_var.host){
                sprintf(name, "src%d_dev%d_%s", s, d, (*dev)[d].vars[i].name);
                dims[0] = (*dev)[d].vars[i].num_ele;
                if (!m->SKIP_CHECKPOINT_FILE)
                    __GUARD checkexists(file_id, name);
                if (!m->SKIP_CHECKPOINT_FILE)
                    __GUARD readvar(file_id,
                                H5T_NATIVE_FLOAT,
                                name,
                                (*dev)[d].vars[i].cl_var.host);
                __GUARD clbuf_send(&(*dev)[d].queue, &(*dev)[d].vars[i].cl_var);
            }

            if ((*dev)[d].vars[i].cl_varbnd.host){
                sprintf(name, "src%d_dev%d_%s_bnd", s, d,
                        (*dev)[d].vars[i].name);
                dims[0] = (*dev)[d].vars[i].cl_varbnd.sizepin / sizeof(float);
                if (!m->SKIP_CHECKPOINT_FILE)
                    __GUARD readvar(file_id,
                                H5T_NATIVE_FLOAT,
                                name,
                                (*dev)[d].vars[i].cl_varbnd.host);
            }

            if ((*dev)[d].vars[i].cl_buf1.host) {
                dims[0] = (*dev)[d].vars[i].cl_buf1.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf1d", s, d,
                        (*dev)[d].vars[i].name);
                if (!m->SKIP_CHECKPOINT_FILE)
                    __GUARD readvar(file_id,
                                H5T_NATIVE_FLOAT,
                                name,
                                (*dev)[d].vars[i].cl_buf1.host);
                nwait = (*dev)[d].vars[i].cl_buf1.nwait_s;
                event = (*dev)[d].vars[i].cl_buf1.waits_s;
                (*dev)[d].vars[i].cl_buf1.nwait_s = 0;
                (*dev)[d].vars[i].cl_buf1.waits_s = NULL;
                __GUARD clbuf_send(&(*dev)[d].queue, &(*dev)[d].vars[i].cl_buf1);
                (*dev)[d].vars[i].cl_buf1.nwait_s = nwait;
                (*dev)[d].vars[i].cl_buf1.waits_s = event;
            }

            if ((*dev)[d].vars[i].cl_buf2.host) {
                dims[0] = (*dev)[d].vars[i].cl_buf2.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf2d", s, d,
                        (*dev)[d].vars[i].name);
                if (!m->SKIP_CHECKPOINT_FILE)
                    __GUARD readvar(file_id,
                                H5T_NATIVE_FLOAT,
                                name,
                                (*dev)[d].vars[i].cl_buf2.host);
                nwait = (*dev)[d].vars[i].cl_buf2.nwait_s;
                event = (*dev)[d].vars[i].cl_buf2.waits_s;
                (*dev)[d].vars[i].cl_buf2.nwait_s = 0;
                (*dev)[d].vars[i].cl_buf2.waits_s = NULL;
                __GUARD clbuf_send(&(*dev)[d].queue, &(*dev)[d].vars[i].cl_buf2);
                (*dev)[d].vars[i].cl_buf2.nwait_s = nwait;
                (*dev)[d].vars[i].cl_buf2.waits_s = event;
            }
        }
    }
    for (d = 0; d < m->NUM_DEVICES; d++) {
        WAITQUEUE((*dev)[d].queue);
        for (i = 0; i < m->nvars; i++) {

            if ((*dev)[d].vars[i].cl_buf1.host) {
                dims[0] = (*dev)[d].vars[i].cl_buf1.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf1h", s, d,
                        (*dev)[d].vars[i].name);
                if (!m->SKIP_CHECKPOINT_FILE)
                    __GUARD readvar(file_id,
                                H5T_NATIVE_FLOAT,
                                name,
                                (*dev)[d].vars[i].cl_buf1.host);
            }

            if ((*dev)[d].vars[i].cl_buf2.host) {
                dims[0] = (*dev)[d].vars[i].cl_buf2.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf2h", s, d,
                        (*dev)[d].vars[i].name);
                if (!m->SKIP_CHECKPOINT_FILE)
                    __GUARD readvar(file_id,
                                H5T_NATIVE_FLOAT,
                                name,
                                (*dev)[d].vars[i].cl_buf2.host);
            }
        }
    }

    for (d=0;d<m->NUM_DEVICES;d++) {
        WAITQUEUE((*dev)[d].queue);
    }
    /* The sends above have completed: give the engine its own buffer back. */
    if (m->CKPT_SHOTSTORE) shotstore_swap_var(m, dev, s);

    return state;
}

int movout(model * m, device ** dev, int t, int s){
    // Collect the buffers for movie creation
    int state=0;
    int d, i, j, elm, elfd;
    int k,l;
    int Nel=1;
    int Nelg;
    int Nm[MAX_DIMS];
    int Nfd[MAX_DIMS];
    half * hptr;
    float * fptr;

    // Tranfer all variables to ouput to host for this time step
    for (d=0;d<m->NUM_DEVICES;d++){
        for (i=0;i<m->nvars;i++){
            if ((*dev)[d].vars[i].to_output){
                __GUARD clbuf_read(&(*dev)[d].queue,&(*dev)[d].vars[i].cl_var);
            }
        }
    }

    Nelg=1;
    for (j=0;j<m->NDIM;j++){
        Nelg*=m->N[j];
    }

    // Aggregate in a global buffers all variables from all devices.
    // Local and global variables don't have the same size, the first being
    // padded by FDORDER/2 on all sides, so we need to transform coordinates
    for (d=0;d<m->NUM_DEVICES;d++){
        WAITQUEUE((*dev)[d].queue);
        //Number of elements mapped from local to global buffer
        Nel=1;
        for (j=0;j<m->NDIM;j++){
            Nel*=(*dev)[d].N[j];
        }
        for (i=0;i<m->nvars;i++){
            if ((*dev)[d].vars[i].to_output){
                if (m->FP16<2){
                    fptr = (*dev)[d].vars[i].cl_var.host;
                }
                else{
                    hptr = (half*)(*dev)[d].vars[i].cl_var.host;
                }
                for (j=0;j<Nel;j++){
                    //Linear indice in global buffer of this element
                    elm=s*m->NT/m->MOVOUT*Nelg
                       +((t+1)/m->MOVOUT-1)*Nelg
                        +j ;
                    // Indices for each dimensions for global Nm and local Nfd
                    for (k=0;k<m->NDIM;k++){
                        Nm[k]=j;
                        for (l=0;l<k;l++){
                            Nm[k]=Nm[k]/(*dev)[d].N[l];
                        }
                        Nm[k]=Nm[k]%(*dev)[d].N[k];
                        Nfd[k]=Nm[k]+m->FDOH;
                        for (l=0;l<k;l++){
                            Nfd[k]*=(*dev)[d].N[l]+m->FDORDER;
                        }
                    }
                    // Linear indice for local buffer
                    elfd=0;
                    for (k=0;k<m->NDIM;k++){
                        elfd+=Nfd[k];
                    }
                    if (m->FP16<2){
                        (*dev)[d].vars[i].gl_mov[elm]=fptr[elfd];
                    }
                    else{
                        (*dev)[d].vars[i].gl_mov[elm]=half_to_float(hptr[elfd]);
                    }
                }
            }
        }
    }
    
    return state;
}

int save_bnd(model * m, device ** dev, int t){
    
    int state=0;
    int d,i;
    int lv=-1;
    int l0=-1;
    int offset;
    
    for (d=0;d<m->NUM_DEVICES;d++){
        
        (*dev)[d].grads.savebnd.outevent=1;
        __GUARD prog_launch(&(*dev)[d].queue, &(*dev)[d].grads.savebnd);
        lv=-1;
        l0=-1;
        for (i=0;i<m->nvars;i++){
            if ((*dev)[d].vars[i].to_comm){
                if (l0<0){
                    l0=i;
                }
                lv=i;
            }
        }
        (*dev)[d].vars[lv].cl_varbnd.outevent_r=1;
        (*dev)[d].vars[l0].cl_varbnd.nwait_r=1;
        (*dev)[d].vars[l0].cl_varbnd.waits_r=&(*dev)[d].grads.savebnd.event;
        if (m->FP16>1){
            offset =(*dev)[d].NBND*t/2;
        }
        else{
            offset =(*dev)[d].NBND*t;
        }
        for (i=0;i<m->nvars;i++){
            if ((*dev)[d].vars[i].to_comm){
                __GUARD clbuf_readto(&(*dev)[d].queuecomm,
                                     &(*dev)[d].vars[i].cl_varbnd,
                                     &(*dev)[d].vars[i].cl_varbnd.host[offset]);
            }
        }
        (*dev)[d].grads.savebnd.nwait=1;
        (*dev)[d].grads.savebnd.waits=&(*dev)[d].vars[lv].cl_varbnd.event_r;
    }

    return state;
}

#ifndef __SEISCL__
/* Asynchronous upload of the saved boundary in the adjoint pass (CUDA, see
   BND_ASYNC in F.h). inject_bnd used to copy every variable's boundary from the
   host on the compute queue at every step, in series with the kernels (8-10 %
   of a step). Here the device buffer cl_varbnd has two halves (Init_OpenCL.c):
   at step t the kernels read half t&1 while queuecomm uploads the boundary of
   step t-1 into the other half, after the kernels of step t+1 have finished
   with it. The kernels take &cl_varbnd.mem (prog_arg), so pointing .mem at a
   half re-points them; .mem is restored to the start of the buffer at the end. */
struct bnd_async {
    CUevent up[2];        /* upload of a half done (queuecomm) */
    CUevent done[2];      /* kernels done with a half (queue) */
    CUdeviceptr * base;   /* start of each variable's buffer */
};

static int bnd_offset(model * m, device * dv, int t){
    return m->FP16>1 ? dv->NBND*(t-1)/2 : dv->NBND*(t-1);
}

/* Queue the upload of the boundary of step t into half t&1. */
static int bnd_async_upload(model * m, device * dv, struct bnd_async * b, int t){
    int state = 0, i, h = t & 1, offset = bnd_offset(m, dv, t);
    __GUARD cuStreamWaitEvent(dv->queuecomm, b->done[h], 0);
    for (i=0;i<m->nvars;i++){
        if (dv->vars[i].to_comm){
            __GUARD cuMemcpyHtoDAsync(b->base[i] + h*dv->vars[i].cl_varbnd.size,
                                      &dv->vars[i].cl_varbnd.host[offset],
                                      dv->vars[i].cl_varbnd.size,
                                      dv->queuecomm);
        }
    }
    __GUARD cuEventRecord(b->up[h], dv->queuecomm);
    return state;
}

static int bnd_async_begin(model * m, device * dv, struct bnd_async * b, int t){
    int state = 0, i, h;
    GMALLOC(b->base, sizeof(CUdeviceptr)*m->nvars);
    for (i=0;i<m->nvars;i++) b->base[i] = dv->vars[i].cl_varbnd.mem;
    for (h=0;h<2;h++){
        __GUARD cuEventCreate(&b->up[h], CU_EVENT_DISABLE_TIMING);
        __GUARD cuEventCreate(&b->done[h], CU_EVENT_DISABLE_TIMING);
    }
    /* the copies of the forward pass (save_bnd) into the same host buffer and
       the previous shot's uploads are on queuecomm too: in order */
    __GUARD bnd_async_upload(m, dv, b, t);
    return state;
}

/* Before the kernels of step t: wait for its half and point the buffers at it,
   then queue the upload of step t-1 (if any). */
static int bnd_async_step(model * m, device * dv, struct bnd_async * b, int t){
    int state = 0, i, h = t & 1;
    __GUARD cuStreamWaitEvent(dv->queue, b->up[h], 0);
    for (i=0;i<m->nvars;i++){
        if (dv->vars[i].to_comm)
            dv->vars[i].cl_varbnd.mem = b->base[i] + h*dv->vars[i].cl_varbnd.size;
    }
    if (t-1 > m->tmin) __GUARD bnd_async_upload(m, dv, b, t-1);
    return state;
}

/* After the kernels of step t. */
static int bnd_async_step_done(device * dv, struct bnd_async * b, int t){
    return cuEventRecord(b->done[t & 1], dv->queue);
}

/* Always run, even after an error: restores the buffers and frees the events. */
static int bnd_async_end(model * m, device * dv, struct bnd_async * b){
    int state = 0, i, h;
    cuStreamSynchronize(dv->queuecomm);
    if (b->base){
        for (i=0;i<m->nvars;i++) dv->vars[i].cl_varbnd.mem = b->base[i];
        free(b->base);
        b->base = NULL;
    }
    for (h=0;h<2;h++){
        if (b->up[h]) cuEventDestroy(b->up[h]);
        if (b->done[h]) cuEventDestroy(b->done[h]);
        b->up[h] = b->done[h] = 0;
    }
    return state;
}
#endif

int inject_bnd(model * m, device ** dev, int t){
//TODO overlap comm and a kernel to inject the wavefield.
// Must create a new kernel for injecting boundaries to do so.
    int state=0;
    int d,i;
    int offset;
    
    for (d=0;d<m->NUM_DEVICES;d++){

        
        if (m->FP16>1){
            offset =(*dev)[d].NBND*(t-1)/2;
        }
        else{
            offset =(*dev)[d].NBND*(t-1);
        }

        for (i=0;i<m->nvars;i++){
            if ((*dev)[d].vars[i].to_comm){
                __GUARD clbuf_sendfrom(&(*dev)[d].queue,
                                       &(*dev)[d].vars[i].cl_varbnd,
                                       &(*dev)[d].vars[i].cl_varbnd.host[offset]);
            }
        }
    }
    
    return state;
}

int update_grid(model * m, device ** dev, int docomm){
    /*Update operations of one iteration */
    int state=0;
    int d, i;


    for (i=0;i<m->nupdates;i++){
        
        // Updating the variables
        for (d=0;d<m->NUM_DEVICES;d++){
            // Launch the kernel on the outside grid needing communication only
            // if a neighbouring device or processing elelement exist
            if (d>0 || m->MYLOCALID>0){
                __GUARD prog_launch( &(*dev)[d].queue,
                                       &(*dev)[d].ups_f[i].com1);
                __GUARD prog_launch( &(*dev)[d].queue,
                                       &(*dev)[d].ups_f[i].fcom1_out);
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                __GUARD prog_launch( &(*dev)[d].queue,
                                      &(*dev)[d].ups_f[i].com2);
                __GUARD prog_launch( &(*dev)[d].queue,
                                      &(*dev)[d].ups_f[i].fcom2_out);
            }

            //Launch kernel on the interior elements
            __GUARD prog_launch( &(*dev)[d].queue,
                                  &(*dev)[d].ups_f[i].center);

        }
        
        if (docomm==1){
            // Communication between devices and MPI processes
            if (m->NUM_DEVICES>1 || m->NLOCALP>1)
                __GUARD comm(m, dev, 0, i);

            // Transfer memory in communication buffers to variables' buffers
            for (d=0;d<m->NUM_DEVICES;d++){

                if (d>0 || m->MYLOCALID>0){
                    __GUARD prog_launch(   &(*dev)[d].queue,
                                           &(*dev)[d].ups_f[i].fcom1_in);
                }
                if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                    __GUARD prog_launch(   &(*dev)[d].queue,
                                           &(*dev)[d].ups_f[i].fcom2_in);
                }
            }
        }
    }


    return state;
}

int update_grid_adj(model * m, device ** dev){
    /*Update operations for one iteration */
    int state=0;
    int d, i;
    
    /* Direct-form gradient in the top rows under a stress-image free surface
       (surface2D_adj.cl): needs the adjoint fields of the start of the step,
       before the stress update and the adjoint of the surface. */
    for (d=0;d<m->NUM_DEVICES;d++){
        if ((*dev)[d].bnd_cnds.surf_grad.name){
            __GUARD prog_launch(&(*dev)[d].queue, &(*dev)[d].bnd_cnds.surf_grad);
        }
        /* velocity receivers: the forward velocity is still v(t+1) */
        if ((*dev)[d].bnd_cnds.rec_grad_v.name){
            __GUARD prog_launch(&(*dev)[d].queue, &(*dev)[d].bnd_cnds.rec_grad_v);
        }
    }
    // Perform the updates in backward order for adjoint simulation
    for (i=m->nupdates-1;i>=0;i--){
        /* Exact CPML adjoint: filter the adjoint fields that this update
           differentiates (velocities before update_adjs, i = 1; stresses
           before update_adjv, i = 0). See cpml_adj2D.cl. */
        for (d=0;d<m->NUM_DEVICES;d++){
            if ((*dev)[d].bnd_cnds.cpml_adj_v.name && !cpml_inline(m)){
                __GUARD prog_launch(&(*dev)[d].queue, i==1 ?
                                    &(*dev)[d].bnd_cnds.cpml_adj_v :
                                    &(*dev)[d].bnd_cnds.cpml_adj_s);
            }
        }
        // Updating the variables
        for (d=0;d<m->NUM_DEVICES;d++){
            // Launch the kernel on the outside grid needing communication only
            // if a neighbouring device or processing elelement exist
            if (d>0 || m->MYLOCALID>0){
                __GUARD prog_launch( &(*dev)[d].queue,
                                    &(*dev)[d].ups_adj[i].com1);
                __GUARD prog_launch( &(*dev)[d].queue,
                                    &(*dev)[d].ups_f[i].fcom1_out);
                if (m->BACK_PROP_TYPE==1){
                    __GUARD prog_launch( &(*dev)[d].queue,
                                        &(*dev)[d].ups_adj[i].fcom1_out);
                }
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                __GUARD prog_launch( &(*dev)[d].queue,
                                    &(*dev)[d].ups_adj[i].com2);
                __GUARD prog_launch( &(*dev)[d].queue,
                                    &(*dev)[d].ups_f[i].fcom2_out);
                if (m->BACK_PROP_TYPE==1){
                    __GUARD prog_launch( &(*dev)[d].queue,
                                        &(*dev)[d].ups_adj[i].fcom2_out);
                }
            }
            
            //Launch kernel on the interior elements
            __GUARD prog_launch( &(*dev)[d].queue,
                                &(*dev)[d].ups_adj[i].center);
            /* pressure receivers: update_adjs (i = 1) has just rebuilt the
               forward stresses to s(t) */
            if (i==1 && (*dev)[d].bnd_cnds.rec_grad_s.name){
                __GUARD prog_launch(&(*dev)[d].queue,
                                    &(*dev)[d].bnd_cnds.rec_grad_s);
            }
//            cuStreamSynchronize((*dev)[d].queue);
//            cuStreamSynchronize((*dev)[d].queuecomm);
        }
        
        // Communication between devices and MPI processes
        if (m->NUM_DEVICES>1 || m->NLOCALP>1)
            __GUARD comm(m, dev, 1, i);
        
//        for (d=0;d<m->NUM_DEVICES;d++){
//            cuStreamSynchronize((*dev)[d].queue);
//            cuStreamSynchronize((*dev)[d].queuecomm);
//        }
        // Transfer memory in communication buffers to variables' buffers
        for (d=0;d<m->NUM_DEVICES;d++){
            if (d>0 || m->MYLOCALID>0){
                __GUARD prog_launch(&(*dev)[d].queue,
                                    &(*dev)[d].ups_f[i].fcom1_in);
                if (m->BACK_PROP_TYPE==1){
                    __GUARD prog_launch(&(*dev)[d].queue,
                                        &(*dev)[d].ups_adj[i].fcom1_in);
                }
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                __GUARD prog_launch(&(*dev)[d].queue,
                                    &(*dev)[d].ups_f[i].fcom2_in);
                if (m->BACK_PROP_TYPE==1){
                    __GUARD prog_launch(&(*dev)[d].queue,
                                        &(*dev)[d].ups_adj[i].fcom2_in);
                }
            }
        }

        // Adjoint of the free surface: in the forward step the surface kernel
        // runs after the stress update and before the next velocity update;
        // its adjoint must sit at the same place, between the adjoint stress
        // update (i = 1, run first) and the adjoint velocity update (i = 0).
        if (m->FREESURF==1 && i==1){
            for (d=0;d<m->NUM_DEVICES;d++){
                if ((*dev)[d].bnd_cnds.surf_adj_fold.name){
                    __GUARD prog_launch(&(*dev)[d].queue,
                                        &(*dev)[d].bnd_cnds.surf_adj_fold);
                }
                __GUARD prog_launch(&(*dev)[d].queue,
                                    &(*dev)[d].bnd_cnds.surf_adj);
            }
        }
    }

    return state;
}

int initialize_forward(model * m, device ** dev, int s, int * pdir){
    /*Initialize the buffers to 0 before the first time step of each shot*/
    int state=0;
    int d, i, ind;
    
    // Initialization of the seismic variables
    for (d=0;d<m->NUM_DEVICES;d++){
        
        // Source and receivers position are transfered at the beginning of each
        // simulation
        (*dev)[d].src_recs.cl_src.size=sizeof(float)
                                           * m->NT * (*dev)[d].src_recs.nsrc[s];
        (*dev)[d].src_recs.cl_src.host=(*dev)[d].src_recs.src[s];
        (*dev)[d].src_recs.cl_src_pos.size= sizeof(float)
                                               * 5 * (*dev)[d].src_recs.nsrc[s];
        (*dev)[d].src_recs.cl_src_pos.host=(*dev)[d].src_recs.src_pos[s];
        (*dev)[d].src_recs.cl_rec_pos.size= sizeof(float)
                                               * 8 * (*dev)[d].src_recs.nrec[s];
        (*dev)[d].src_recs.cl_rec_pos.host=(*dev)[d].src_recs.rec_pos[s];
        /* The source gradient is read back into this shot's slice of the
           host array, with this shot's size: the host pointer was never set
           (read into NULL), and the allocation size (nsmax) would overflow
           the slice of a shot with fewer sources (gpu-memory-audit.md, D1). */
        if (m->GRADSRCOUT){
            (*dev)[d].src_recs.cl_grad_src.size= sizeof(float)
                                       * m->NT * (*dev)[d].src_recs.nsrc[s];
            (*dev)[d].src_recs.cl_grad_src.host=(*dev)[d].src_recs.gradsrc[s];
        }
        
        #ifndef __SEISCL__
        if (m->src_recs.src_dev){
            /* the wavelets are on a device (SeisCL.torch): device to device */
            size_t off = sizeof(float)*(size_t)(m->src_recs.src[s]-m->src_recs.src[0]);
            __GUARD cuMemcpyAsync((*dev)[d].src_recs.cl_src.mem,
                                  (CUdeviceptr)(m->src_recs.src_dev+off),
                                  (*dev)[d].src_recs.cl_src.size, (*dev)[d].queue);
        }
        else
        #endif
        __GUARD clbuf_send(&(*dev)[d].queue, &(*dev)[d].src_recs.cl_src);
        __GUARD clbuf_send(&(*dev)[d].queue, &(*dev)[d].src_recs.cl_src_pos);
        __GUARD clbuf_send(&(*dev)[d].queue, &(*dev)[d].src_recs.cl_rec_pos);
        
        // Assign work sizes to kernels
        (*dev)[d].src_recs.sources.gsize[0]=(*dev)[d].src_recs.nsrc[s];
        (*dev)[d].src_recs.varsout.gsize[0]=(*dev)[d].src_recs.nrec[s];
        (*dev)[d].src_recs.varsoutinit.gsize[0]=(*dev)[d].src_recs.nrec[s]*m->NT;
        (*dev)[d].src_recs.residuals.gsize[0]=(*dev)[d].src_recs.nrec[s];
        /* rec_grad.cl: one work-item per receiver of this shot */
        (*dev)[d].bnd_cnds.rec_grad_v.gsize[0]=(*dev)[d].src_recs.nrec[s];
        (*dev)[d].bnd_cnds.rec_grad_s.gsize[0]=(*dev)[d].src_recs.nrec[s];
        (*dev)[d].src_recs.init_gradsrc.gsize[0]=(*dev)[d].src_recs.nsrc[s]
                                                                        * m->NT;
        //Assign some arg to kernels
        for (i=0;i<(*dev)[d].nprogs;i++){
            if ((*dev)[d].progs[i]->pdir>0){
                ind =(*dev)[d].progs[i]->pdir-1;
                __GUARD prog_arg((*dev)[d].progs[i], ind, pdir, sizeof(int));
            }
            if ((*dev)[d].progs[i]->nsinput>0){
                ind =(*dev)[d].progs[i]->nsinput-1;
                __GUARD prog_arg((*dev)[d].progs[i], ind, &m->src_recs.nsrc[s], sizeof(int));
            }
            if ((*dev)[d].progs[i]->nrinput>0){
                ind=(*dev)[d].progs[i]->nrinput-1;
                __GUARD prog_arg((*dev)[d].progs[i], ind, &m->src_recs.nrec[s], sizeof(int));
            }
            if ((*dev)[d].progs[i]->scinput>0){
                ind=(*dev)[d].progs[i]->scinput-1;
                __GUARD prog_arg((*dev)[d].progs[i], ind, &m->src_recs.src_scales[s], sizeof(int));
            }
            if ((*dev)[d].progs[i]->psinput>0){
                ind=(*dev)[d].progs[i]->psinput-1;
                __GUARD prog_arg((*dev)[d].progs[i], ind, &m->par_scale, sizeof(int));
            }
        }
        
        // Implement initial conditions
        __GUARD prog_launch( &(*dev)[d].queue, &(*dev)[d].bnd_cnds.init_f);
        if (m->VARSOUT>0 || m->GRADOUT || m->RMSOUT || m->RESOUT){
            __GUARD prog_launch(&(*dev)[d].queue,
                                &(*dev)[d].src_recs.varsoutinit);
        }

        if (m->GRADOUT==1 && m->BACK_PROP_TYPE==2){
            __GUARD prog_launch( &(*dev)[d].queue,
                                 &(*dev)[d].grads.initsavefreqs);
        }
        

        
    }
    
    return state;
}

int initialize_adj(model * m, device ** dev, int s, int * pdir){
    /*Initialize the buffers to 0 before the first time step of each shot*/
    int state=0;
    int d, i, ind;
    
    // Initialize the backpropagation and gradient.
    // and transfer the residual to GPUs
    for (d=0;d<m->NUM_DEVICES;d++){
        
        /* stress receivers: the parameters at the receivers (residuals.c;
           dev_res: written on the devices by dio_res_scale) */
        if (rec_mod_needed(m) && !m->dev_res){
            __GUARD clbuf_sendfrom(&(*dev)[d].queue, &(*dev)[d].src_recs.cl_recmod,
                                   m->src_recs.recmod);
        }
        // Transfer the residuals to the gpus (dev_res: from the first device)
        if (m->dev_res){
            if (d==0) __GUARD dio_res_to_adj(m, dev, s);
        }
        else
        for (i=0;i<m->nvars;i++){
            if ( (*dev)[d].vars[i].to_output){
                __GUARD clbuf_sendfrom(&(*dev)[d].queue,
                                       &(*dev)[d].vars[i].cl_varout,
                                       (*dev)[d].vars[i].gl_var_res[s]);
            }
        }
        if (!m->dev_res)
        for (i=0;i<m->ntvars;i++){
            if ( (*dev)[d].trans_vars[i].to_output){
                __GUARD clbuf_sendfrom(&(*dev)[d].queue,
                                       &(*dev)[d].trans_vars[i].cl_varout,
                                       (*dev)[d].trans_vars[i].gl_var_res[s]);
            }
        }
        // Initialize the backpropagation of the forward variables
        if (m->BACK_PROP_TYPE==1){
            __GUARD prog_launch( &(*dev)[d].queue,
                                &(*dev)[d].bnd_cnds.init_adj);
        }
        // Initialized the source gradient
        if (m->GRADSRCOUT==1){
            __GUARD prog_launch( &(*dev)[d].queue,
                                &(*dev)[d].src_recs.init_gradsrc);
        }
        // Transfer to host the forward variable frequencies obtained
        // obtained by DFT. The same buffers on the devices are reused
        // for the adjoint variables.
        if (m->BACK_PROP_TYPE==2){
            for (i=0;i<(*dev)[d].nvars;i++){
                if ((*dev)[d].vars[i].for_grad){
                    /* Stash the forward spectrum in a second device buffer, so
                     * savefreqs can reuse cl_fvar for the adjoint pass and the
                     * correlation still has both spectra on the device. Only
                     * pull it to the host when something actually reads it
                     * there: the dftout dump. */
                    __GUARD clbuf_copy(&(*dev)[d].queue,
                                       &(*dev)[d].vars[i].cl_fvar,
                                       &(*dev)[d].vars[i].cl_fvar_f);
                    /* Viscoelastic: keep the forward final state for the
                     * end-of-record terms of the correlation, before init_f
                     * zeroes cl_var for the adjoint run. */
                    if ((*dev)[d].vars[i].cl_var_e.size>0){
                        __GUARD clbuf_copy(&(*dev)[d].queue,
                                           &(*dev)[d].vars[i].cl_var,
                                           &(*dev)[d].vars[i].cl_var_e);
                    }
                    if (m->DFTOUT){
                        __GUARD clbuf_read(&(*dev)[d].queue,
                                           &(*dev)[d].vars[i].cl_fvar);
                    }
                }
            }
            // Inialize to 0 the frequency buffers, and the adjoint
            // variable buffers (forward buffers are reused).
            __GUARD prog_launch( &(*dev)[d].queue,
                                &(*dev)[d].grads.initsavefreqs);
            __GUARD prog_launch( &(*dev)[d].queue,
                                &(*dev)[d].bnd_cnds.init_f);
        }
        
        //Assign the propagation direction to kernels
        for (i=0;i<(*dev)[d].nprogs;i++){
            if ((*dev)[d].progs[i]->pdir>0){
                ind=(*dev)[d].progs[i]->pdir-1;
                __GUARD prog_arg((*dev)[d].progs[i], ind, pdir, sizeof(int));
            }
            if ((*dev)[d].progs[i]->rcinput>0){
                ind=(*dev)[d].progs[i]->rcinput-1;
                __GUARD prog_arg((*dev)[d].progs[i], ind, &m->src_recs.res_scales[s], sizeof(int));
            }
        }
        
    }
    
    return state;
}

int time_stepping(model * m, device ** dev, struct filenames files) {
    // Performs forward and adjoint modeling for each source point assigned to
    // this group of nodes and devices.

    int state=0;
    
    int t,s,i,d, thist;
    int ind;
    int pdir = 1;
    hid_t file_id=0;
   

    // Calculate what shots belong to the group this processing element
    m->src_recs.smin=0;
    m->src_recs.smax=0;
    
    for (i=0;i<m->MYGROUPID;i++){
        if (i<m->src_recs.ns%m->NGROUP){
            m->src_recs.smin+=(m->src_recs.ns/m->NGROUP+1);
        }
        else{
            m->src_recs.smin+=(m->src_recs.ns/m->NGROUP);
        }
        
    }
    if (m->MYGROUPID<m->src_recs.ns%m->NGROUP){
        m->src_recs.smax=m->src_recs.smin+(m->src_recs.ns/m->NGROUP+1);
    }
    else{
        m->src_recs.smax=m->src_recs.smin+(m->src_recs.ns/m->NGROUP);
    }
    
    // Initialize the gradient buffers before time stepping (both methods:
    // the DFT correlation accumulates into the same buffers across shots).
    if (m->GRADOUT==1){
        for (d=0;d<m->NUM_DEVICES;d++){
            __GUARD prog_launch( &(*dev)[d].queue, &(*dev)[d].grads.init);
        }
    }

    /* The checkpoint protocol stores the wavefield so the second call can skip
     * the forward pass. It cannot serve BACK_PROP_TYPE==2, whose adjoint needs
     * the *frequency* buffers accumulated during the forward pass, and those
     * are not checkpointed. For the DFT path the forward pass is simply re-run
     * below: one extra propagation, but obviously correct and no new state. */
    //Initialize checkpoint file
    /* BACK_PROP_TYPE==1 only: the DFT path has no checkpoint (see above).
     * The in-memory image and the SKIP_CHECKPOINT_FILE flag come from devel. */
    if (m->INPUTRES==1 && m->GRADOUT==0 && m->BACK_PROP_TYPE==1
        && !m->SKIP_CHECKPOINT_FILE){
        if (m->CKPT_IN_MEMORY){
            /* Discard any image left by an earlier forward pass and start a
             * fresh one; it stays open for the adjoint pass to read. */
            if (m->CKPT_FILE_ID > 0){
                H5Fclose(m->CKPT_FILE_ID);
                m->CKPT_FILE_ID = 0;
            }
            m->CKPT_FILE_ID = create_file_core(files.checkpoint);
            file_id = m->CKPT_FILE_ID;
        }
        else {
            state = remove(files.checkpoint);
            if (errno == ENOENT)
                state = 0;
            else if (state){
                perror("Could not delete checkpoint file");
                }
            if (!state){
                file_id = create_file(files.checkpoint);
            }
        }
    }
    if (m->INPUTRES==1 && m->GRADOUT==1 && m->BACK_PROP_TYPE==1
        && !m->SKIP_CHECKPOINT_FILE){
        if (m->CKPT_IN_MEMORY){
            file_id = m->CKPT_FILE_ID;
        }
        else {
            file_id = H5Fopen(files.checkpoint, H5F_ACC_RDWR, H5P_DEFAULT);
        }
    }
    /* Per-shot host buffers (CKPT_SHOTSTORE, F.h): allocated on the first
     * call, kept by the caller for the matching backward call. */
    if (m->CKPT_SHOTSTORE && !state && checkpoint_store_alloc(m, *dev)){
        fprintf(stderr, "Error: could not allocate the per-shot checkpoint "
                        "buffers (%d shots)\n",
                m->src_recs.smax - m->src_recs.smin);
        state = 1;
    }
    // Main loop over shots of this group
    for (s= m->src_recs.smin;s< m->src_recs.smax;s++){

        /* CKPT_SHOTSTORE: shot s saves (forward) or reads (adjoint) its
           boundary history in its own pinned buffer. Swapped back at the end
           of the shot. */
        if (m->CKPT_SHOTSTORE) shotstore_swap_bnd(m, dev, s);

        // Initialization of the seismic variables
        pdir=1;
        __GUARD initialize_forward(m, dev, s, &pdir);

        // Loop for forward time stepping
        /* Skipped in the backward call of the two-call protocol, which
           restores the checkpoint instead -- unless CKPT_RECOMPUTE (F.h):
           then this loop runs again and saves the boundaries itself. */
        if (!(m->INPUTRES && m->GRADOUT && m->BACK_PROP_TYPE==1
              && !m->CKPT_RECOMPUTE)) {
            for (t = 0; t < m->tmax; t++) {
                //Assign the time step value to kernels
                for (d = 0; d < m->NUM_DEVICES; d++) {
                    for (i = 0; i < (*dev)[d].nprogs; i++) {
                        if ((*dev)[d].progs[i]->tinput > 0) {
                            ind = (*dev)[d].progs[i]->tinput - 1;
                            __GUARD prog_arg((*dev)[d].progs[i], ind, &t,
                                             sizeof(int));
                        }
                    }
                }

                //Save the selected frequency if the gradient is obtained by DFT
                if (m->GRADOUT == 1
                    && m->BACK_PROP_TYPE == 2
                    && t >= m->tmin
                    && (t - m->tmin) % m->DTNYQ == 0) {

                    for (d = 0; d < m->NUM_DEVICES; d++) {
                        thist = (t - m->tmin) / m->DTNYQ;
                        ind = (*dev)[d].grads.savefreqs.tinput - 1;
                        __GUARD prog_arg(&(*dev)[d].grads.savefreqs, ind,
                                         &thist, sizeof(int));
                        __GUARD prog_launch(&(*dev)[d].queue,
                                            &(*dev)[d].grads.savefreqs);
                    }

                }

                // Inject the sources
                for (d = 0; d < m->NUM_DEVICES; d++) {
                    __GUARD prog_launch(&(*dev)[d].queue,
                                        &(*dev)[d].src_recs.sources);
                }

                // Apply all updates
                if (t < (m->tmax - 1)) {
                    __GUARD update_grid(m, dev, 1);
                } else {
                    __GUARD update_grid(m, dev, 0);
                }


                // Computing the free surface
                if (m->FREESURF == 1) {
                    for (d = 0; d < m->NUM_DEVICES; d++) {
                        __GUARD prog_launch(&(*dev)[d].queue,
                                            &(*dev)[d].bnd_cnds.surf);
                    }
                }

                // Save the boundaries. After the free surface: the adjoint
                // pass injects them as the state at the end of this step,
                // which the next step uses, i.e. with szz = 0 and the
                // corrected sxx on the surface row. Saving them before the
                // surface kernel made the rebuilt forward field wrong in the
                // FDOH rows below the surface (gradient ratios 0.1-0.9).
                // CKPT_RECOMPUTE: the forward call keeps no boundaries; the
                // backward call saves them when it re-runs this loop.
                if ((m->GRADOUT == 1 || (m->INPUTRES && !m->CKPT_RECOMPUTE))
                    && m->BACK_PROP_TYPE == 1)
                    __GUARD save_bnd(m, dev, t);

                // Outputting seismograms
                if (m->VARSOUT > 0 || m->GRADOUT || m->RMSOUT || m->RESOUT) {
                    for (d = 0; d < m->NUM_DEVICES; d++) {
                        __GUARD prog_launch(&(*dev)[d].queue,
                                            &(*dev)[d].src_recs.varsout);
                    }
                }

                // Outputting the movie
                if (m->MOVOUT > 0 &&
                    !(m->BACK_PROP_TYPE == 1 && m->GRADOUT == 1)
                    && (t + 1) % m->MOVOUT == 0 && state == 0) {
                    movout(m, dev, t, s);
                }

#ifdef __SEISCL__
                // Flush all the previous commands to the computing device
                for (d = 0; d < m->NUM_DEVICES; d++) {
                    if (d > 0 || d < m->NUM_DEVICES - 1) {
                        __GUARD clFlush((*dev)[d].queuecomm);
                    }
                    __GUARD clFlush((*dev)[d].queue);
                }
#endif

            }

            // Aggregate the seismograms in the output variable
            if (m->VARSOUT>0 || m->GRADOUT || m->RMSOUT || m->RESOUT){
                __GUARD reduce_seis(m, dev, s);
            }

            //Calculate the residuals
            if ((m->GRADOUT || m->RMSOUT || m->RESOUT) && m->INPUTRES==0){
                __GUARD m->res_calc(m,s);
            }
            /* res_scale turns the residual into the adjoint source. In the
             * original protocol it ran only in the else branch below, the one
             * taken when the forward loop is skipped. BACK_PROP_TYPE==2 now
             * runs the forward loop even under INPUTRES, so it has to run here
             * too -- otherwise a supplied residual is never made into an
             * adjoint source and the gradient comes out identically zero. */
            if ((m->GRADOUT || m->RMSOUT || m->RESOUT)
                && (!m->INPUTRES || m->GRADOUT)){
                if (m->dev_res){
                    __GUARD dio_res_scale(m, dev, s);
                    /* back_prop_type 2: receiver-cell density term (residuals.c) */
                    __GUARD dio_rec_grad_dft(m, dev, s);
                }
                else {
                    __GUARD m->res_scale(m,s);
                    /* back_prop_type 2: receiver-cell density term (residuals.c) */
                    __GUARD rec_grad_dft(m, s);
                }
            }

            // Save the checkpoints. BACK_PROP_TYPE==2 keeps none: its
            // adjoint call re-runs this forward pass instead (see the note
            // above the checkpoint file creation), so no file was opened and
            // file_id is not a valid location.
            if (m->INPUTRES && m->GRADOUT==0 && m->BACK_PROP_TYPE==1
                && !m->CKPT_RECOMPUTE){
                __GUARD checkpoint_d2h(m, dev, file_id, s);

            }
        }
        else {
            checkpoint_h2d(m, dev, file_id, s);
            if (m->dev_res){
                __GUARD dio_res_scale(m, dev, s);
            }
            else {
                __GUARD m->res_scale(m,s);
            }
        }

        // Calculation of the gradient for this shot, if required
        if (m->GRADOUT==1){

            // Initialize adjoint time stepping
            pdir=-1;
            __GUARD initialize_adj(m, dev, s, &pdir);
            #ifndef __SEISCL__
            struct bnd_async bnda;
            memset(&bnda, 0, sizeof(bnda));
            int bnd_async = BND_ASYNC(m) && m->tmax-1 > m->tmin;
            if (bnd_async) __GUARD bnd_async_begin(m, &(*dev)[0], &bnda, m->tmax-1);
            #endif

            // Inverse time stepping
            for (t=m->tmax-1;t>m->tmin; t--){

                //Assign the time step value to kernels
                for (d=0;d<m->NUM_DEVICES;d++){
                    for (i=0;i<(*dev)[d].nprogs;i++){
                        if ((*dev)[d].progs[i]->tinput>0){
                            ind = (*dev)[d].progs[i]->tinput-1;
                            __GUARD prog_arg((*dev)[d].progs[i], ind, &t,
                                             sizeof(int));
                        }
                    }
                }

                // Inject the forward variables boundaries
                #ifndef __SEISCL__
                if (bnd_async){
                    __GUARD bnd_async_step(m, &(*dev)[0], &bnda, t);
                }
                else
                #endif
                if (m->BACK_PROP_TYPE==1){
                    __GUARD inject_bnd(m, dev, t);
                }

                // Inject the residuals
                for (d=0;d<m->NUM_DEVICES;d++){
                    __GUARD prog_launch( &(*dev)[d].queue,
                                         &(*dev)[d].src_recs.residuals);
                }

                // Update the adjoint wavefield and perform back-propagation of
                // forward wavefield
                __GUARD update_grid_adj(m, dev);
                // Inject the sources with negative sign
                //TODO not right if source is inside saved boundary
                if (m->BACK_PROP_TYPE==1){
                    for (d=0;d<m->NUM_DEVICES;d++){
                        __GUARD prog_launch( &(*dev)[d].queue,
                                            &(*dev)[d].src_recs.sources);
                    }
                }

                //Save the selected frequency if the gradient is obtained by DFT
                if (m->BACK_PROP_TYPE==2 && (t-m->tmin)%m->DTNYQ==0){

                    for (d=0;d<m->NUM_DEVICES;d++){
                        thist=(t-m->tmin)/m->DTNYQ;
                        ind = (*dev)[d].grads.savefreqs.tinput-1;
                        __GUARD prog_arg(&(*dev)[d].grads.savefreqs, ind,
                                         &thist, sizeof(int));
                        __GUARD prog_launch(&(*dev)[d].queue,
                                            &(*dev)[d].grads.savefreqs);
                    }

                }
                // Outputting the movie.
                // NOTE: left gated on BACK_PROP_TYPE==1 deliberately. Enabling
                // it for BACK_PROP_TYPE==2 would dump the adjoint field (which
                // reuses vars[i].cl_var in that mode), but into the same gl_mov
                // buffer and the same frame index the forward loop above already
                // wrote, so the two passes would partly overwrite each other.
                // An adjoint movie needs its own buffer; not needed here because
                // T1 validates the forward spectrum against a separate
                // forward-only run, and the adjoint sampling is covered by the
                // gradient-level tests.
                if (m->MOVOUT>0 && m->BACK_PROP_TYPE==1
                    && (t)%m->MOVOUT==0 && state==0)
                    movout(m, dev, t, s);

                #ifdef __SEISCL__
                for (d=0;d<m->NUM_DEVICES;d++){
                    if (d>0 || d<m->NUM_DEVICES-1){
                        __GUARD clFlush((*dev)[d].queuecomm);
                    }
                    __GUARD clFlush((*dev)[d].queue);
                }
                #else
                if (bnd_async) __GUARD bnd_async_step_done(&(*dev)[0], &bnda, t);
                #endif

            }
            #ifndef __SEISCL__
            if (bnd_async) bnd_async_end(m, &(*dev)[0], &bnda);
            #endif

            // Transfer  the source gradient to the host
            if (m->GRADSRCOUT==1){
                for (d=0;d<m->NUM_DEVICES;d++){
                    __GUARD clbuf_read( &(*dev)[d].queue,
                                        &(*dev)[d].src_recs.cl_grad_src);
                }
            }

            // Transfer the adjoint frequencies to the host, calculate the
            // gradient by the crosscorrelation of forward and adjoint
            // frequencies and intialize frequencies and forward buffers to 0
            // for the forward modeling of the next source.
            if (m->BACK_PROP_TYPE==2 && !state){
                for (d=0;d<m->NUM_DEVICES;d++){
                    /* The on-device correlation reads both spectra straight
                     * from device memory. Only the dftout dump needs the
                     * adjoint spectrum on the host. */
                    if (m->DFTOUT){
                        for (i=0;i<(*dev)[d].nvars;i++){
                            if ((*dev)[d].vars[i].for_grad){
                                __GUARD clbuf_readto(&(*dev)[d].queue,
                                                 &(*dev)[d].vars[i].cl_fvar,
                                                 (*dev)[d].vars[i].cl_fvar_adj.host);
                            }
                        }
                    }
                    __GUARD prog_launch(&(*dev)[d].queue,
                                        &(*dev)[d].grads.calc_grad);

                    __GUARD prog_launch(&(*dev)[d].queue,
                                        &(*dev)[d].grads.initsavefreqs);
                    __GUARD prog_launch( &(*dev)[d].queue,
                                        &(*dev)[d].bnd_cnds.init_f);
                }
                for (d=0;d<m->NUM_DEVICES;d++){
                    __GUARD WAITQUEUE((*dev)[d].queue);
                }
                if (m->DFTOUT && !state){
                    __GUARD dump_dft(m, dev, files);
                }
            }

        }

        if (m->CKPT_SHOTSTORE) shotstore_swap_bnd(m, dev, s);

    }
    // Using back-propagation, the gradient is computed on the devices. After
    // all sources positions have been modeled, transfer back the gradient.
    /* dev_grad: the readback and the host post-processing below run on the
       device (calc_grad.c, gpost_run) */
    if (m->GRADOUT==1 && m->dev_grad){
        __GUARD gpost_run(m, dev);
    }
    else if (m->GRADOUT==1){
        /* BACK_PROP_TYPE==2 with the on-device correlation also accumulates
         * into cl_grad, so the readback is shared. It does *not* get
         * transf_grad: the DFT coefficients already carry the par_type chain
         * rule, so applying it again would double it. */
        int devgrad = (m->BACK_PROP_TYPE==1 || m->BACK_PROP_TYPE==2);
        if (devgrad){
            for (d=0;d<m->NUM_DEVICES;d++){
                for (i=0;i<m->npars;i++){
                    if ((*dev)[d].pars[i].to_grad){
                        __GUARD clbuf_read(&(*dev)[d].queue,
                                           &(*dev)[d].pars[i].cl_grad);
                    }
                    if (m->HOUT==1 && (*dev)[d].pars[i].to_grad){
                        __GUARD clbuf_read(&(*dev)[d].queue,
                                           &(*dev)[d].pars[i].cl_H);
                    }
                }

            }

            for (d=0;d<m->NUM_DEVICES;d++){
                __GUARD WAITQUEUE((*dev)[d].queue);
            }
        }

        if (m->BACK_PROP_TYPE==1){
            __GUARD transf_grad(m);
        }
        else if (m->BACK_PROP_TYPE==2){
            /* Same steps as transf_grad, but unscale_grad_dft() in place of
               unscale_grad(): the frequency-domain correlation's own missing
               unit-conversion factor is *not* the same as the time-domain
               boundary-storage gradient's (derived in unscale_grad_dft()'s
               own docstring, calc_grad.c -- notes/todo.md item 0h). Applied
               before average_grad_transpose(): it is a uniform per-cell
               scalar, so it commutes with the (linear) averaging operator.
               The parameters still have to be unscaled, because
               chain_rule_par_type evaluates its Jacobians at physical
               values. */
            __GUARD unpack_par_fp16(m);
            if (m->dev_res){
                __GUARD dio_gradrec_readback(m, dev);
            }
            __GUARD add_rec_grad_dft(m);
            __GUARD unscale_grad_dft(m);
            __GUARD average_grad_transpose(m);
            __GUARD unscale_par(m);
            __GUARD chain_rule_par_type(m);
        }
    }

    /* A RAM-backed checkpoint has to outlive this call so the adjoint pass
     * can read it, so it is left open and owned by the caller. */
    if (file_id && !(m->CKPT_IN_MEMORY && file_id == m->CKPT_FILE_ID)){
        H5Fclose(file_id);
    }

    #ifndef __NOMPI__
    if (state && m->MPI_INIT==1)
        MPI_Bcast( &state, 1, MPI_INT, m->GID, MPI_COMM_WORLD );
    #endif
    
    return state;
}

