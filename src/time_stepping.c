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

int checkpoint_d2h(model * m, device ** dev, struct filenames files, int s){
    int state=0;
    int d, i, j;
    char name[100];
    hid_t file_id=0;
    file_id = -1;
    hsize_t dims[MAX_DIMS];
    struct stat info;

    if (stat(files.checkpoint, &info ) != 0) {
        file_id = create_file(files.checkpoint);
    }
    else{
        file_id = H5Fopen(files.checkpoint, H5F_ACC_RDWR, H5P_DEFAULT);
    }
    if (file_id<0){
        state=1;
        fprintf(stderr,"Error: Could not open the checkpoint file %s",
                files.checkpoint);
    }

    for (d=0;d<m->NUM_DEVICES;d++){
        WAITQUEUE((*dev)[d].queue);
        for (i=0;i<m->nvars;i++){
            if ((*dev)[d].vars[i].cl_buf1.host){
                dims[0] = (*dev)[d].vars[i].cl_buf1.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf1h", s, d, (*dev)[d].vars[i].name);
                writetomat(&file_id, name, (*dev)[d].vars[i].cl_buf1.host,
                           1, dims);
            }
            if ((*dev)[d].vars[i].cl_buf2.host){
                dims[0] = (*dev)[d].vars[i].cl_buf2.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf2h", s, d, (*dev)[d].vars[i].name);
                writetomat(&file_id, name, (*dev)[d].vars[i].cl_buf2.host,
                           1, dims);
            }
        }
    }

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

            dims[0] = (*dev)[d].vars[i].num_ele;
            sprintf(name, "src%d_dev%d_%s", s, d, (*dev)[d].vars[i].name);
            writetomat(&file_id, name, (*dev)[d].vars[i].cl_var.host,
                       1, dims);

            dims[0] = (*dev)[d].vars[i].cl_varbnd.sizepin / sizeof(float);
            sprintf(name, "src%d_dev%d_%s_bnd", s, d, (*dev)[d].vars[i].name);
            writetomat(&file_id, name, (*dev)[d].vars[i].cl_varbnd.host,
                       1, dims);

            if ((*dev)[d].vars[i].cl_buf1.host){
                dims[0] = (*dev)[d].vars[i].cl_buf1.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf1d", s, d, (*dev)[d].vars[i].name);
                writetomat(&file_id, name, (*dev)[d].vars[i].cl_buf1.host,
                           1, dims);
            }
            if ((*dev)[d].vars[i].cl_buf2.host){
                dims[0] = (*dev)[d].vars[i].cl_buf2.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf2d", s, d, (*dev)[d].vars[i].name);
                writetomat(&file_id, name, (*dev)[d].vars[i].cl_buf2.host,
                           1, dims);
            }
        }
    }

    if (file_id) H5Fclose(file_id);

    return state;
}

int checkpoint_h2d(model * m, device ** dev, struct filenames files, int s) {
    int state = 0;
    int d, i;
    char name[100];
    hid_t file_id = 0;
    file_id = -1;
    hsize_t dims[MAX_DIMS];
    int nwait;
    EVENT * event;

    file_id = H5Fopen(files.checkpoint, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error: Could not open the checkpoint file %s",
                files.checkpoint);
    }

    // Transfer all variables to output to host for this time step
    for (d = 0; d < m->NUM_DEVICES; d++) {
        for (i = 0; i < m->nvars; i++) {
            sprintf(name, "src%d_dev%d_%s", s, d, (*dev)[d].vars[i].name);
            dims[0] = (*dev)[d].vars[i].num_ele;
            __GUARD checkexists(file_id, name);
            __GUARD readvar(file_id,
                            H5T_NATIVE_FLOAT,
                            name,
                            (*dev)[d].vars[i].cl_var.host);
            __GUARD clbuf_send(&(*dev)[d].queue, &(*dev)[d].vars[i].cl_var);

            sprintf(name, "src%d_dev%d_%s_bnd", s, d, (*dev)[d].vars[i].name);
            dims[0] = (*dev)[d].vars[i].cl_varbnd.sizepin / sizeof(float);
            __GUARD readvar(file_id,
                            H5T_NATIVE_FLOAT,
                            name,
                            (*dev)[d].vars[i].cl_varbnd.host);

            if ((*dev)[d].vars[i].cl_buf1.host) {
                dims[0] = (*dev)[d].vars[i].cl_buf1.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf1d", s, d,
                        (*dev)[d].vars[i].name);
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
                __GUARD readvar(file_id,
                                H5T_NATIVE_FLOAT,
                                name,
                                (*dev)[d].vars[i].cl_buf1.host);
            }

            if ((*dev)[d].vars[i].cl_buf2.host) {
                dims[0] = (*dev)[d].vars[i].cl_buf2.size / sizeof(float);
                sprintf(name, "src%d_dev%d_%s_buf2h", s, d,
                        (*dev)[d].vars[i].name);
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

    if (file_id) H5Fclose(file_id);

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
    
    // Perform the updates in backward order for adjoint simulation
    for (i=m->nupdates-1;i>=0;i--){
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
        
        __GUARD clbuf_send(&(*dev)[d].queue, &(*dev)[d].src_recs.cl_src);
        __GUARD clbuf_send(&(*dev)[d].queue, &(*dev)[d].src_recs.cl_src_pos);
        __GUARD clbuf_send(&(*dev)[d].queue, &(*dev)[d].src_recs.cl_rec_pos);
        
        // Assign work sizes to kernels
        (*dev)[d].src_recs.sources.gsize[0]=(*dev)[d].src_recs.nsrc[s];
        (*dev)[d].src_recs.varsout.gsize[0]=(*dev)[d].src_recs.nrec[s];
        (*dev)[d].src_recs.varsoutinit.gsize[0]=(*dev)[d].src_recs.nrec[s]*m->NT;
        (*dev)[d].src_recs.residuals.gsize[0]=(*dev)[d].src_recs.nrec[s];
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
        
        // Transfer the residuals to the gpus
        for (i=0;i<m->nvars;i++){
            if ( (*dev)[d].vars[i].to_output){
                __GUARD clbuf_sendfrom(&(*dev)[d].queue,
                                       &(*dev)[d].vars[i].cl_varout,
                                       (*dev)[d].vars[i].gl_var_res[s]);
            }
        }
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
                    __GUARD clbuf_read(&(*dev)[d].queue,
                                       &(*dev)[d].vars[i].cl_fvar);
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
    
    // Initialize the gradient buffers before time stepping
    if (m->GRADOUT==1 && m->BACK_PROP_TYPE==1){
        for (d=0;d<m->NUM_DEVICES;d++){
            __GUARD prog_launch( &(*dev)[d].queue, &(*dev)[d].grads.init);
        }
    }

    //Intialize checkpoint file
    if (m->INPUTRES==1 && m->GRADOUT==0){
        state = remove(files.checkpoint);
        if (state)
            fprintf(stderr, "Could not delete checkpoint file %s\n", files.res);
        file_id = create_file(files.checkpoint);
        H5Fclose(file_id);
    }

    // Main loop over shots of this group
    for (s= m->src_recs.smin;s< m->src_recs.smax;s++){

        // Initialization of the seismic variables
        pdir=1;
        __GUARD initialize_forward(m, dev, s, &pdir);

        // Loop for forward time stepping
        if (!(m->INPUTRES && m->GRADOUT)) {
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


                // Save the boundaries
                if ((m->GRADOUT == 1 || m->INPUTRES) && m->BACK_PROP_TYPE == 1)
                    __GUARD save_bnd(m, dev, t);

                // Computing the free surface
                if (m->FREESURF == 1) {
                    for (d = 0; d < m->NUM_DEVICES; d++) {
                        __GUARD prog_launch(&(*dev)[d].queue,
                                            &(*dev)[d].bnd_cnds.surf);
                    }
                }

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
            if (!m->INPUTRES && (m->GRADOUT || m->RMSOUT || m->RESOUT)){
                __GUARD m->res_scale(m,s);
            }

            // Save the checkpoints
            if (m->INPUTRES && m->GRADOUT==0){
                __GUARD checkpoint_d2h(m, dev, files, s);
            }
        }
        else {
            checkpoint_h2d(m, dev, files, s);
        }

        // Calculation of the gradient for this shot, if required
        if (m->GRADOUT==1){

            // Initialize adjoint time stepping
            pdir=-1;
            __GUARD initialize_adj(m, dev, s, &pdir);

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
                if (m->BACK_PROP_TYPE==1){
                    __GUARD inject_bnd(m, dev, t);
                }

                // Computing the free surface
                if (m->FREESURF==1){
                    for (d=0;d<m->NUM_DEVICES;d++){
                        __GUARD prog_launch(&(*dev)[d].queue,
                                            &(*dev)[d].bnd_cnds.surf_adj);
                    }
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
                // Outputting the movie
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
                #endif

            }

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
                    for (i=0;i<(*dev)[d].nvars;i++){
                        if ((*dev)[d].vars[i].for_grad){
                            __GUARD clbuf_readto(&(*dev)[d].queue,
                                                 &(*dev)[d].vars[i].cl_fvar,
                                                 (*dev)[d].vars[i].cl_fvar_adj.host);
                        }

                    }

                    __GUARD prog_launch(&(*dev)[d].queue,
                                        &(*dev)[d].grads.initsavefreqs);
                    __GUARD prog_launch( &(*dev)[d].queue,
                                        &(*dev)[d].bnd_cnds.init_f);
                }
                for (d=0;d<m->NUM_DEVICES;d++){
                    __GUARD WAITQUEUE((*dev)[d].queue);
                    __GUARD calc_grad(m, &(*dev)[d]);
                }
            }

        }
        

    }
    // Using back-propagation, the gradient is computed on the devices. After
    // all sources positions have been modeled, transfer back the gradient.
    if (m->GRADOUT==1 && m->BACK_PROP_TYPE==1){
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

        __GUARD transf_grad(m);
    }

    #ifndef __NOMPI__
    if (state && m->MPI_INIT==1)
        MPI_Bcast( &state, 1, MPI_INT, m->GID, MPI_COMM_WORLD );
    #endif
    
    return state;
}

