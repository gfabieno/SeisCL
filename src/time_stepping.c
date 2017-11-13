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
        __GUARD cuStreamSynchronize((*dev)[d].queue);
        for (k=0;k<(*dev)[d].nvars;k++){
            if ((*dev)[d].vars[k].to_output){
                for ( i=0;i<(*dev)[d].src_recs.nrec[s];i++){
                    posx=(int)floor((*dev)[d].src_recs.rec_pos[s][8*i]/m->dh);
                    if (posx>=(*dev)[d].NX0
                        && posx<((*dev)[d].NX0+(*dev)[d].N[(*dev)[d].NDIM-1])){
                        
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
                    if (posx>=(*dev)[d].NX0
                        && posx<((*dev)[d].NX0+(*dev)[d].N[(*dev)[d].NDIM-1])){
                        
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

int movout(model * m, device ** dev, int t, int s){
    // Collect the buffers for movie creation
    int state=0;
    int d, i, j, elm, elfd;
    int k,l;
    int Nel=1;
    int Nm[MAX_DIMS];
    int Nfd[MAX_DIMS];

    // Tranfer all variables to ouput to host for this time step
    for (d=0;d<m->NUM_DEVICES;d++){
        for (i=0;i<m->nvars;i++){
            if ((*dev)[d].vars[i].to_output){
                __GUARD clbuf_read(&(*dev)[d].queue,&(*dev)[d].vars[i].cl_var);
            }
        }
        
    }
    
    // Aggregate in a global buffers all variables from all devices.
    // Local and global variables don't have the same size, the first being
    // padded by FDORDER/2 on all sides, so we need to transform coordinates
    for (d=0;d<m->NUM_DEVICES;d++){
        cuStreamSynchronize((*dev)[d].queue);
        for (i=0;i<m->nvars;i++){
            if ((*dev)[d].vars[i].to_output){
                
                //Number of elements mapped from local to global buffer
                Nel=1;
                for (j=0;j<m->NDIM;j++){
                    Nel*=(*dev)[d].N[j];
                }
                for (j=0;j<Nel;j++){
                    //Linear indice in global buffer of this element
                    elm=s*m->NT/m->MOVOUT*Nel
                       +((t+1)/m->MOVOUT-1)*Nel
                       +j;
                    // Indices for each dimensions for global Nm and local Nfd
                    for (k=0;k<m->NDIM;k++){
                        Nm[k]=j;
                        for (l=0;l<k;l++){
                            Nm[k]=Nm[k]/(*dev)[d].N[l];
                        }
                        Nm[k]=Nm[k]%(*dev)[d].N[l];
                        Nfd[k]=Nm[k]+m->FDOH;
                        for (l=0;l<k;l++){
                            Nm[k]*=(*dev)[d].N[l]+m->FDORDER;
                        }
                    }
                    // Linear indice for local buffer
                    elfd=1;
                    for (k=0;k<m->NDIM;k++){
                        for (l=0;l<k;l++){
                            elfd+=Nfd[k];
                        }
                    }
                    (*dev)[d].vars[i].gl_mov[elm]=(*dev)[d].vars[i].gl_mov[elfd];
                    
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
//        if ((*dev)[d].grads.savebnd.waits)
//            __GUARD clReleaseEvent(*(*dev)[d].grads.savebnd.waits);
        
        for (i=0;i<m->nvars;i++){
            if ((*dev)[d].vars[i].to_comm){
                if (l0<0){
                    l0=i;
                }
                lv=i;
            }
        }

//        (*dev)[d].vars[lv].cl_varbnd.outevent_r=1;
//        (*dev)[d].vars[l0].cl_varbnd.nwait_r=1;
//        (*dev)[d].vars[l0].cl_varbnd.waits_r=&(*dev)[d].grads.savebnd.event;
        
        if (m->FP16>0){
            offset =(*dev)[d].NBND*t/2;
        }
        else{
            offset =(*dev)[d].NBND*t;
        }
        int j;
        for (i=0;i<m->nvars;i++){
            if ((*dev)[d].vars[i].to_comm){
                __GUARD clbuf_readpin(&(*dev)[d].queue,
                                      &(*dev)[d].vars[i].cl_varbnd,
                                      &(*dev)[d].vars[i].cl_varbnd,
                                      offset);
                for (j=0;j<(*dev)[d].NBND;j++){
                    (*dev)[d].vars[i].cl_varbnd.pin[offset+j]=1;
                }
            }
        }
//        (*dev)[d].grads.savebnd.nwait=1;
//        (*dev)[d].grads.savebnd.waits=&(*dev)[d].vars[lv].cl_varbnd.event_r;
//        __GUARD clReleaseEvent((*dev)[d].grads.savebnd.event);
        
    }

    
    return state;
}

int inject_bnd(model * m, device ** dev, int t){
//TODO overlapped comm and a kernel to inject the wavefield
    int state=0;
    int d,i;
    int offset;
    
    for (d=0;d<m->NUM_DEVICES;d++){
        
        if (m->FP16>0){
            offset =(*dev)[d].NBND*t/2;
        }
        else{
            offset =(*dev)[d].NBND*t;
        }

        for (i=0;i<m->nvars;i++){
            if ((*dev)[d].vars[i].to_comm){
                __GUARD clbuf_sendpin(&(*dev)[d].queue,
                                      &(*dev)[d].vars[i].cl_varbnd,
                                      &(*dev)[d].vars[i].cl_varbnd,
                                      offset);
            }
        }
        
        
    }
    
    
    return state;
}

int update_grid(model * m, device ** dev){
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
        
        // Communication between devices and MPI processes
//        if (m->NUM_DEVICES>1 || m->NLOCALP>1)
//            __GUARD comm(m, dev, 0, i);

        
        // Transfer memory in communication buffers to variables' buffers
        for (d=0;d<m->NUM_DEVICES;d++){
            
            if (d>0 || m->MYLOCALID>0){
                __GUARD prog_launch(   &(*dev)[d].queue,
                                       &(*dev)[d].ups_f[i].fcom1_in);
//                __GUARD clReleaseEvent(*(*dev)[d].ups_f[i].fcom1_in.waits);
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                __GUARD prog_launch(   &(*dev)[d].queue,
                                       &(*dev)[d].ups_f[i].fcom2_in);
//                __GUARD clReleaseEvent(*(*dev)[d].ups_f[i].fcom2_in.waits);
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
                                    &(*dev)[d].ups_adj[i].fcom1_out);
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                __GUARD prog_launch( &(*dev)[d].queue,
                                    &(*dev)[d].ups_adj[i].com2);
                __GUARD prog_launch( &(*dev)[d].queue,
                                    &(*dev)[d].ups_adj[i].fcom2_out);
            }
            
            //Launch kernel on the interior elements
            __GUARD prog_launch( &(*dev)[d].queue,
                                &(*dev)[d].ups_adj[i].center);
            
        }
        
        // Communication between devices and MPI processes
//        if (m->NUM_DEVICES>1 || m->NLOCALP>1)
//        __GUARD comm(m, dev, 1, i);
        
        
        // Transfer memory in communication buffers to variables' buffers
        for (d=0;d<m->NUM_DEVICES;d++){
            
            if (d>0 || m->MYLOCALID>0){
                __GUARD prog_launch(   &(*dev)[d].queue,
                                    &(*dev)[d].ups_adj[i].fcom1_in);
//                __GUARD clReleaseEvent(*(*dev)[d].ups_adj[i].fcom1_in.waits);
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                __GUARD prog_launch(   &(*dev)[d].queue,
                                    &(*dev)[d].ups_adj[i].fcom2_in);
//                __GUARD clReleaseEvent(*(*dev)[d].ups_adj[i].fcom2_in.waits);
            }
        }
        
    }

    return state;
}

int initialize_grid(model * m, device ** dev, int s){
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
        //Assign the some arg to kernels
        int pdir=1;
        for (i=0;i<(*dev)[d].nprogs;i++){
            if ((*dev)[d].progs[i]->pdir>0){
                ind =(*dev)[d].progs[i]->pdir-1;
                (*dev)[d].progs[i]->inputs[ind]=&pdir;
            }
            if ((*dev)[d].progs[i]->nsinput>0){
                ind =(*dev)[d].progs[i]->nsinput-1;
                (*dev)[d].progs[i]->inputs[ind]=&m->src_recs.nsrc[s];
            }
            if ((*dev)[d].progs[i]->nrinput>0){
                ind=(*dev)[d].progs[i]->nrinput-1;
                (*dev)[d].progs[i]->inputs[ind]=&m->src_recs.nrec[s];
            }
            if ((*dev)[d].progs[i]->scinput>0){
                ind=(*dev)[d].progs[i]->scinput-1;
                (*dev)[d].progs[i]->inputs[ind]=&m->src_recs.src_scales[s];
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

int time_stepping(model * m, device ** dev) {
    // Performs forward and adjoint modeling for each source point assigned to
    // this group of nodes and devices.

    int state=0;
    
    int t,s,i,d, thist;
   

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
    
    // Main loop over shots of this group
    for (s= m->src_recs.smin;s< m->src_recs.smax;s++){

        // Initialization of the seismic variables
        __GUARD initialize_grid(m, dev, s);
        
        // Loop for forward time stepping
        for (t=0;t<m->tmax; t++){
            
            //Assign the time step value to kernels
            for (d=0;d<m->NUM_DEVICES;d++){
                for (i=0;i<(*dev)[d].nprogs;i++){
                    if ((*dev)[d].progs[i]->tinput>0){
                        (*dev)[d].progs[i]->inputs[(*dev)[d].progs[i]->tinput-1]=&t;
                    }
                }
            }
            
            //Save the selected frequency if the gradient is obtained by DFT
            if (m->GRADOUT==1
                && m->BACK_PROP_TYPE==2
                && t>=m->tmin
                && (t-m->tmin)%m->DTNYQ==0){

                for (d=0;d<m->NUM_DEVICES;d++){
                    thist=(t-m->tmin)/m->DTNYQ;
                    (*dev)[d].grads.savefreqs.inputs[(*dev)[d].grads.savefreqs.tinput-1]=&thist;
                    __GUARD prog_launch( &(*dev)[d].queue,
                                         &(*dev)[d].grads.savefreqs);
                }

            }
            
            // Inject the sources
            for (d=0;d<m->NUM_DEVICES;d++){
                __GUARD prog_launch( &(*dev)[d].queue,
                                    &(*dev)[d].src_recs.sources);
            }

            // Apply all updates
            __GUARD update_grid(m, dev);
            
            
            // Computing the free surface
            if (m->FREESURF==1){
                for (d=0;d<m->NUM_DEVICES;d++){
                    __GUARD prog_launch( &(*dev)[d].queue,
                                        &(*dev)[d].bnd_cnds.surf);
                }
            }

            // Save the boundaries
            if (m->GRADOUT==1 && m->BACK_PROP_TYPE==1){
                __GUARD save_bnd( m, dev, t);
            }

            // Outputting seismograms
            if (m->VARSOUT>0 || m->GRADOUT || m->RMSOUT || m->RESOUT){
                for (d=0;d<m->NUM_DEVICES;d++){
                    __GUARD prog_launch( &(*dev)[d].queue,
                                        &(*dev)[d].src_recs.varsout);
                }
            }

            // Outputting the movie
            if (m->MOVOUT>0 && (t+1)%m->MOVOUT==0 && state==0)
                movout( m, dev, t, s);


            
        }
        

//        //Realease events that have not been released
//        for (d=0;d<m->NUM_DEVICES;d++){
//            if (m->GRADOUT==1 && m->BACK_PROP_TYPE==1){
//                __GUARD clReleaseEvent(*(*dev)[d].grads.savebnd.waits);
//            }
//        }

        // Aggregate the seismograms in the output variable
        if (m->VARSOUT>0 || m->GRADOUT || m->RMSOUT || m->RESOUT){
            __GUARD reduce_seis(m, dev, s);
        }

        //Calculate the residuals
        if (m->GRADOUT || m->RMSOUT || m->RESOUT){
            __GUARD m->res_calc(m,s);
        }

        // Calculation of the gradient for this shot, if required
        if (m->GRADOUT==1){
            
            // Initialize the backpropagation and gradient.
            // and transfer the residual to GPUs
            for (d=0;d<m->NUM_DEVICES;d++){
               
                // Transfer the residuals to the gpus
                for (i=0;i<m->nvars;i++){
                    if ( (*dev)[d].vars[i].to_output){
                        (*dev)[d].vars[i].cl_var_res.size=sizeof(float)
                                                  * m->NT * m->src_recs.nrec[s];
                        (*dev)[d].vars[i].cl_var_res.pin=
                                                (*dev)[d].vars[i].gl_var_res[s];
                        __GUARD clbuf_sendpin(&(*dev)[d].queue,
                                              &(*dev)[d].vars[i].cl_varout,
                                              &(*dev)[d].vars[i].cl_var_res,
                                              0);
                    }
                }
                for (i=0;i<m->ntvars;i++){
                    if ( (*dev)[d].trans_vars[i].to_output){
                        (*dev)[d].trans_vars[i].cl_var_res.size=sizeof(float)
                        * m->NT * m->src_recs.nrec[s];
                        (*dev)[d].trans_vars[i].cl_var_res.pin=
                        (*dev)[d].trans_vars[i].gl_var_res[s];
                        __GUARD clbuf_sendpin(&(*dev)[d].queue,
                                              &(*dev)[d].trans_vars[i].cl_varout,
                                              &(*dev)[d].trans_vars[i].cl_var_res,
                                              0);
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
                int pdir=-1;
                for (d=0;d<m->NUM_DEVICES;d++){
                    for (i=0;i<(*dev)[d].nprogs;i++){
                        if ((*dev)[d].progs[i]->pdir>0){
                            (*dev)[d].progs[i]->inputs[(*dev)[d].progs[i]->pdir-1]=&pdir;
                        }
                    }
                }

            }

            // Inverse time stepping
            for (t=m->tmax-1;t>=m->tmin; t--){

                //Assign the time step value to kernels
                for (d=0;d<m->NUM_DEVICES;d++){
                    for (i=0;i<(*dev)[d].nprogs;i++){
                        if ((*dev)[d].progs[i]->tinput>0){
                            (*dev)[d].progs[i]->inputs[(*dev)[d].progs[i]->tinput-1]=&t;
                        }
                    }
                }
                
                // Inject the forward variables boundaries
                if (m->BACK_PROP_TYPE==1){
                    __GUARD inject_bnd( m, dev, t);
                }
                
                // Inject the sources with negative sign
                if (m->BACK_PROP_TYPE==1){
                    for (d=0;d<m->NUM_DEVICES;d++){
                        __GUARD prog_launch( &(*dev)[d].queue,
                                            &(*dev)[d].src_recs.sources);
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
                
                //Save the selected frequency if the gradient is obtained by DFT
                if (m->BACK_PROP_TYPE==2 && (t-m->tmin)%m->DTNYQ==0){
                    
                    for (d=0;d<m->NUM_DEVICES;d++){
                        thist=(t-m->tmin)/m->DTNYQ;
                        (*dev)[d].grads.savefreqs.inputs[(*dev)[d].grads.savefreqs.tinput-1]=&thist;
                        __GUARD prog_launch( &(*dev)[d].queue,
                                            &(*dev)[d].grads.savefreqs);
                    }
                    
                }

//                for (d=0;d<m->NUM_DEVICES;d++){
//                    if (d>0 || d<m->NUM_DEVICES-1)
//                        __GUARD clFlush((*dev)[d].queuecomm);
//                    __GUARD clFlush((*dev)[d].queue);
//                }
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
                            __GUARD clbuf_readpin(&(*dev)[d].queue,
                                                  &(*dev)[d].vars[i].cl_fvar,
                                                  &(*dev)[d].vars[i].cl_fvar_adj,
                                                  0);
                        }
                        
                    }
                    
                    __GUARD prog_launch(&(*dev)[d].queue,
                                        &(*dev)[d].grads.initsavefreqs);
                    __GUARD prog_launch( &(*dev)[d].queue,
                                        &(*dev)[d].bnd_cnds.init_f);
                }
                for (d=0;d<m->NUM_DEVICES;d++){
                    __GUARD cuStreamSynchronize((*dev)[d].queue);
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
                    __GUARD clbuf_read( &(*dev)[d].queue,
                                       &(*dev)[d].pars[i].cl_grad);
                }
                if (m->HOUT==1 && (*dev)[d].pars[i].to_grad){
                    __GUARD clbuf_read( &(*dev)[d].queue,
                                       &(*dev)[d].pars[i].cl_H);
                }
            }
            
        }
        
        for (d=0;d<m->NUM_DEVICES;d++){
            __GUARD cuStreamSynchronize((*dev)[d].queue);
        }
        
//        __GUARD transf_grad(m);
    }
    
    if (state && m->MPI_INIT==1)
        MPI_Bcast( &state, 1, MPI_INT, m->MYID, MPI_COMM_WORLD );
    
    
    return state;
}

