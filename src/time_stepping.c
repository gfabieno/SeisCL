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

int reduce_seis(struct modcsts * m, struct varcl ** vcl, int s){
    // Transfer the variables to output to host and reduce in global buffer
    int state=0;
    int posx, i, j, k, d;
    
    // Transfer the seismogram from GPUs to host
    for (d=0;d<m->NUM_DEVICES;d++){
        for (i=0;i<m->nvars;i++){
            if ( (*vcl)[d].vars[i].to_output){
                (*vcl)[d].vars[i].cl_var.size=sizeof(float)
                * m->NT * m->src_recs.nrec[s];
                __GUARD clbuf_read( &(*vcl)[d].queue,
                                   &(*vcl)[d].vars[i].cl_var);
            }
            
        }
    }
    
    // Put them in the global buffer that collect all sources and receivers data
    // from all devices. For all MPI processes, it is reduced at the end of
    // program
    for (d=0;d<m->NUM_DEVICES;d++){
        __GUARD clFinish((*vcl)[d].queue);
        for (k=0;k<(*vcl)[d].nvars;k++){
            if ((*vcl)[d].vars[k].to_output){
                for ( i=0;i<(*vcl)[d].src_recs.nrec[s];i++){
                    posx=(int)floor((*vcl)[d].src_recs.rec_pos[s][8*i]/m->dh);
                    if (posx>=(*vcl)[d].NX0
                        && posx<((*vcl)[d].NX0+(*vcl)[d].N[(*vcl)[d].NDIM-1])){
                        
                        for (j=0;j<m->NT;j++){
                            (*vcl)[d].vars[k].gl_varout[s][i*m->NT+j]=
                            (*vcl)[d].vars[k].cl_varout.host[i*m->NT+j];
                        }
                    }
                }
            }
        }
    }
    
    return state;
    
}

int movout(struct modcsts * m, struct varcl ** vcl, int t, int s){
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
            if ((*vcl)[d].vars[i].to_output){
                __GUARD clbuf_read(&(*vcl)[d].queue,&(*vcl)[d].vars[i].cl_var);
            }
        }
        
    }
    
    // Aggregate in a global buffers all variables from all devices.
    // Local and global variables don't have the same size, the first being
    // padded by FDORDER/2 on all sides, so we need to transform coordinates
    for (d=0;d<m->NUM_DEVICES;d++){
        clFinish((*vcl)[d].queue);
        for (i=0;i<m->nvars;i++){
            if ((*vcl)[d].vars[i].to_output){
                
                //Number of elements mapped from local to global buffer
                Nel=1;
                for (j=0;j<m->NDIM;j++){
                    Nel*=(*vcl)[d].N[j];
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
                            Nm[k]=Nm[k]/(*vcl)[d].N[l];
                        }
                        Nm[k]=Nm[k]%(*vcl)[d].N[l];
                        Nfd[k]=Nm[k]+m->FDOH;
                        for (l=0;l<k;l++){
                            Nm[k]*=(*vcl)[d].N[l]+m->FDORDER;
                        }
                    }
                    // Linear indice for local buffer
                    elfd=1;
                    for (k=0;k<m->NDIM;k++){
                        for (l=0;l<k;l++){
                            elfd+=Nfd[k];
                        }
                    }
                    (*vcl)[d].vars[i].gl_mov[elm]=(*vcl)[d].vars[i].gl_mov[elfd];
                    
                }
            }
        }
        
    }
    
    return state;
}

int save_bnd(struct modcsts * m, struct varcl ** vcl, int t){
    
    int state=0;
    int d,i;
    int lv=-1;

    
    for (d=0;d<m->NUM_DEVICES;d++){
        (*vcl)[d].grads.savebnd.outevent=1;
        __GUARD prog_launch(&(*vcl)[d].queue, &(*vcl)[d].grads.savebnd);
        if (t>0)
            __GUARD clReleaseEvent(*(*vcl)[d].grads.savebnd.waits);
        
        for (i=0;i<m->nvars;i++){
            if ((*vcl)[d].vars[i].to_comm){
                lv=i;
            }
        }

        (*vcl)[d].vars[lv].cl_varbnd.outevent_r=1;
        (*vcl)[d].vars[0].cl_varbnd.nwait_r=1;
        (*vcl)[d].vars[0].cl_varbnd.waits_r=&(*vcl)[d].grads.savebnd.event;
        (*vcl)[d].grads.savebnd.nwait=1;
        (*vcl)[d].grads.savebnd.waits=
        &(*vcl)[d].vars[lv].cl_varbnd.event_r;
        for (i=0;i<m->nvars;i++){
            if ((*vcl)[d].vars[i].to_comm){
                __GUARD clbuf_readpin(&(*vcl)[d].queuecomm,
                                      &(*vcl)[d].vars[i].cl_varbnd,
                                      &(*vcl)[d].vars[i].cl_varbnd,
                                      (*vcl)[d].NBND*t);
            }
        }
        __GUARD clReleaseEvent((*vcl)[d].grads.savebnd.event);
        
    }

    
    return state;
}

int inject_bnd(struct modcsts * m, struct varcl ** vcl, int t){
//TODO overlapped comm and a kernel to inject the wavefield
    int state=0;
    int d,i;
   
    
    for (d=0;d<m->NUM_DEVICES;d++){
        

        for (i=0;i<m->nvars;i++){
            if ((*vcl)[d].vars[i].to_comm){
                __GUARD clbuf_sendpin(&(*vcl)[d].queue,
                                      &(*vcl)[d].vars[i].cl_varbnd,
                                      &(*vcl)[d].vars[i].cl_varbnd,
                                      (*vcl)[d].NBND*t);
            }
        }
        
        
    }
    
    
    return state;
}

int update_grid(struct modcsts * m, struct varcl ** vcl){
    /*Update operations of one iteration */
    int state=0;
    int d, i;
    
    
    for (i=0;i<m->nupdates;i++){
        
        // Updating the variables
        for (d=0;d<m->NUM_DEVICES;d++){
            // Launch the kernel on the outside grid needing communication only
            // if a neighbouring device or processing elelement exist
            if (d>0 || m->MYLOCALID>0){
                __GUARD prog_launch( &(*vcl)[d].queue,
                                       &(*vcl)[d].ups_f[i].com1);
                __GUARD prog_launch( &(*vcl)[d].queue,
                                       &(*vcl)[d].ups_f[i].fcom1_out);
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                __GUARD prog_launch( &(*vcl)[d].queue,
                                      &(*vcl)[d].ups_f[i].com2);
                __GUARD prog_launch( &(*vcl)[d].queue,
                                      &(*vcl)[d].ups_f[i].fcom2_out);
            }
            
            //Launch kernel on the interior elements
            __GUARD prog_launch( &(*vcl)[d].queue,
                                  &(*vcl)[d].ups_f[i].center);

        }
        
        // Communication between devices and MPI processes
        if (m->NUM_DEVICES>1 || m->NLOCALP>1)
            __GUARD comm(m, vcl, 0, i);

        
        // Transfer memory in communication buffers to variables' buffers
        for (d=0;d<m->NUM_DEVICES;d++){
            
            if (d>0 || m->MYLOCALID>0){
                __GUARD prog_launch(   &(*vcl)[d].queue,
                                       &(*vcl)[d].ups_f[i].fcom1_in);
                __GUARD clReleaseEvent(*(*vcl)[d].ups_f[i].fcom1_in.waits);
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                __GUARD prog_launch(   &(*vcl)[d].queue,
                                       &(*vcl)[d].ups_f[i].fcom2_in);
                __GUARD clReleaseEvent(*(*vcl)[d].ups_f[i].fcom2_in.waits);
            }
        }
        
    }


    return state;
}

int update_grid_adj(struct modcsts * m, struct varcl ** vcl){
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
                __GUARD prog_launch( &(*vcl)[d].queue,
                                    &(*vcl)[d].ups_adj[i].com1);
                __GUARD prog_launch( &(*vcl)[d].queue,
                                    &(*vcl)[d].ups_adj[i].fcom1_out);
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                __GUARD prog_launch( &(*vcl)[d].queue,
                                    &(*vcl)[d].ups_adj[i].com2);
                __GUARD prog_launch( &(*vcl)[d].queue,
                                    &(*vcl)[d].ups_adj[i].fcom2_out);
            }
            
            //Launch kernel on the interior elements
            __GUARD prog_launch( &(*vcl)[d].queue,
                                &(*vcl)[d].ups_adj[i].center);
            
        }
        
        // Communication between devices and MPI processes
        if (m->NUM_DEVICES>1 || m->NLOCALP>1)
        __GUARD comm(m, vcl, 1, i);
        
        
        // Transfer memory in communication buffers to variables' buffers
        for (d=0;d<m->NUM_DEVICES;d++){
            
            if (d>0 || m->MYLOCALID>0){
                __GUARD prog_launch(   &(*vcl)[d].queue,
                                    &(*vcl)[d].ups_adj[i].fcom1_in);
                __GUARD clReleaseEvent(*(*vcl)[d].ups_adj[i].fcom1_in.waits);
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                __GUARD prog_launch(   &(*vcl)[d].queue,
                                    &(*vcl)[d].ups_adj[i].fcom2_in);
                __GUARD clReleaseEvent(*(*vcl)[d].ups_adj[i].fcom2_in.waits);
            }
        }
        
    }

    return state;
}

int initialize_grid(struct modcsts * m, struct varcl ** vcl, int s){
    /*Initialize the buffers to 0 before the first time step of each shot*/
    int state=0;
    int d;
    

    // Initialization of the seismic variables
    for (d=0;d<m->NUM_DEVICES;d++){
        
        // Source and receivers position are transfered at the beginning of each
        // simulation
        (*vcl)[d].src_recs.cl_src.size=sizeof(float)
                                           * m->NT * (*vcl)[d].src_recs.nsrc[s];
        (*vcl)[d].src_recs.cl_src_pos.size= sizeof(float)
                                               * 5 * (*vcl)[d].src_recs.nsrc[s];
        (*vcl)[d].src_recs.cl_rec_pos.size= sizeof(float)
                                               * 8 * (*vcl)[d].src_recs.nrec[s];
        
        __GUARD clbuf_send(&(*vcl)[d].queue, &(*vcl)[d].src_recs.cl_src);
        __GUARD clbuf_send(&(*vcl)[d].queue, &(*vcl)[d].src_recs.cl_src_pos);
        __GUARD clbuf_send(&(*vcl)[d].queue, &(*vcl)[d].src_recs.cl_rec_pos);
        
        // Implent initial conditions
        __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].bnd_cnds.init_f);
        (*vcl)[d].src_recs.varsoutinit.gsize[0]=m->NT*(*vcl)[d].src_recs.nrec[s];
        
        __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].src_recs.varsoutinit);

        
        // Buffer size for this shot
        (*vcl)[d].src_recs.varsout.gsize[0]=(*vcl)[d].src_recs.nrec[s];
        (*vcl)[d].src_recs.sources.gsize[0]=(*vcl)[d].src_recs.nsrc[s];

        if (m->GRADOUT==1 && m->BACK_PROP_TYPE==2){
            __GUARD prog_launch( &(*vcl)[d].queue,
                                 &(*vcl)[d].grads.initsavefreqs);
        }
        
        
    }
    
    return state;
}

int time_stepping(struct modcsts * m, struct varcl ** vcl) {
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
            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].grads.init);
        }
    }
    
    // Main loop over shots of this group
    for (s= m->src_recs.smin;s< m->src_recs.smax;s++){

        // Initialization of the seismic variables
        __GUARD initialize_grid(m, vcl, s);
        
        // Loop for forward time stepping
        for (t=0;t<m->tmax; t++){
            
            //Assign the time step value to kernels
            for (d=0;d<m->NUM_DEVICES;d++){
                for (i=0;i<(*vcl)[d].nprogs;i++){
                    if ((*vcl)[d].progs[i]->tinput>0)
                        __GUARD clSetKernelArg((*vcl)[d].progs[i]->kernel,
                                               (*vcl)[d].progs[i]->tinput-1,
                                               sizeof(int), &t);
                }
            }
            
            //Save the selected frequency if the gradient is obtained by DFT
            if (m->GRADOUT==1
                && m->BACK_PROP_TYPE==2
                && t>=m->tmin
                && (t-m->tmin)%m->DTNYQ==0){
                
                for (d=0;d<m->NUM_DEVICES;d++){
                    thist=(t-m->tmin)/m->DTNYQ;
                    __GUARD clSetKernelArg((*vcl)[d].grads.savefreqs.kernel,
                                           (*vcl)[d].grads.savefreqs.tinput-1,
                                           sizeof(int), &thist);
                    __GUARD prog_launch( &(*vcl)[d].queue,
                                         &(*vcl)[d].grads.savefreqs);
                }
                
            }
            
            // Apply all updates
            update_grid(m, vcl);
            
            // Computing the free surface
            if (m->FREESURF==1){
                for (d=0;d<m->NUM_DEVICES;d++){
                    __GUARD prog_launch( &(*vcl)[d].queue,
                                        &(*vcl)[d].bnd_cnds.surf);
                }
            }
            
            // Save the boundaries
            if (m->GRADOUT==1 && m->BACK_PROP_TYPE==1)
                __GUARD save_bnd( m, vcl, t);
            
            
            // Outputting seismograms
            for (d=0;d<m->NUM_DEVICES;d++){
                __GUARD prog_launch( &(*vcl)[d].queue,
                                     &(*vcl)[d].src_recs.varsout);
            }

            // Outputting the movie
            if (m->MOVOUT>0 && (t+1)%m->MOVOUT==0 && state==0)
                movout( m, vcl, t, s);
            

            // Flush all the previous commands to the computing device
            for (d=0;d<m->NUM_DEVICES;d++){
                if (d>0 || d<m->NUM_DEVICES-1)
                    __GUARD clFlush((*vcl)[d].queuecomm);
                __GUARD clFlush((*vcl)[d].queue);
            }
            
        }
        

        //Realease events that have not been released
        for (d=0;d<m->NUM_DEVICES;d++){
            if (m->GRADOUT==1 && m->BACK_PROP_TYPE==1){
                __GUARD clReleaseEvent((*vcl)[d].grads.savebnd.event);
            }
        }

        // Aggregate the seismograms in the output variable
        __GUARD reduce_seis(m, vcl, s);

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
                    if ( (*vcl)[d].vars[i].to_output){
                        (*vcl)[d].vars[i].cl_var_res.size=sizeof(float)
                                                  * m->NT * m->src_recs.nrec[s];
                        (*vcl)[d].vars[i].cl_var_res.host=
                                                (*vcl)[d].vars[i].gl_var_res[s];
                        __GUARD clbuf_send(&(*vcl)[d].queue,
                                           &(*vcl)[d].vars[i].cl_var_res);
                    }
                }
                // Initialize the backpropagation of the forward variables
                if (m->BACK_PROP_TYPE==1){
                    __GUARD prog_launch( &(*vcl)[d].queue,
                                         &(*vcl)[d].bnd_cnds.init_adj);
                }
                // Initialized the source gradient
                if (m->GRADSRCOUT==1){
                    __GUARD prog_launch( &(*vcl)[d].queue,
                                        &(*vcl)[d].src_recs.init_gradsrc);
                }
                // Transfer to host the forward variable frequencies obtained
                // obtained by DFT. The same buffers on the devices are reused
                // for the adjoint variables.
                if (m->BACK_PROP_TYPE==2){
                    for (i=0;i<(*vcl)[d].nvars;i++){
                        if ((*vcl)[d].vars[i].for_grad){
                            __GUARD clbuf_read(&(*vcl)[d].queue,
                                               &(*vcl)[d].vars[i].cl_fvar);
                        }
                    }
                    // Inialize to 9 the frequency buffers, and the adjoint
                    // variable buffers (forward buffers are reused).
                    __GUARD prog_launch( &(*vcl)[d].queue,
                                         &(*vcl)[d].grads.initsavefreqs);
                    __GUARD prog_launch( &(*vcl)[d].queue,
                                         &(*vcl)[d].bnd_cnds.init_f);
                }

            }

            // Inverse time stepping
            for (t=m->tmax-1;t>=m->tmin; t--){

                // Injecct the forward variables boundaries
                if (m->BACK_PROP_TYPE==1)
                    __GUARD inject_bnd( m, vcl, t);
                
                // Update the adjoint wavefield and perform back-propagation of
                // forward wavefield
                __GUARD update_grid_adj(m, vcl);
                
                //Save the selected frequency if the gradient is obtained by DFT
                if (m->BACK_PROP_TYPE==2 && (t-m->tmin)%m->DTNYQ==0){
                    
                    for (d=0;d<m->NUM_DEVICES;d++){
                        thist=(t-m->tmin)/m->DTNYQ;
                        __GUARD clSetKernelArg((*vcl)[d].grads.savefreqs.kernel,
                                               (*vcl)[d].grads.savefreqs.tinput-1,
                                               sizeof(int), &thist);
                        __GUARD prog_launch( &(*vcl)[d].queue,
                                            &(*vcl)[d].grads.savefreqs);
                    }
                    
                }

                for (d=0;d<m->NUM_DEVICES;d++){
                    if (d>0 || d<m->NUM_DEVICES-1)
                        __GUARD clFlush((*vcl)[d].queuecomm);
                    __GUARD clFlush((*vcl)[d].queue);
                }
            }
            
            // Transfer  the source gradient to the host
            if (m->GRADSRCOUT==1){
                for (d=0;d<m->NUM_DEVICES;d++){
                    __GUARD clbuf_read( &(*vcl)[d].queue,
                                        &(*vcl)[d].src_recs.cl_grad_src);
                }
            }
            
            // Transfer the adjoint frequencies to the host, calculate the
            // gradient by the crosscorrelation of forward and adjoint
            // frequencies and intialize frequencies and forward buffers to 0
            // for the forward modeling of the next source.
            if (m->BACK_PROP_TYPE==2){
                for (i=0;i<(*vcl)[d].nvars;i++){
                    if ((*vcl)[d].vars_adj[i].for_grad){
                        __GUARD clbuf_read(&(*vcl)[d].queue,
                                           &(*vcl)[d].vars_adj[i].cl_fvar);
                    }
                }
                
                __GUARD prog_launch( &(*vcl)[d].queue,
                                    &(*vcl)[d].grads.initsavefreqs);
                __GUARD prog_launch( &(*vcl)[d].queue,
                                    &(*vcl)[d].bnd_cnds.init_f);
                
                for (d=0;d<m->NUM_DEVICES;d++){
                    __GUARD clFinish((*vcl)[d].queue);
                    if (!state) calc_grad(m);
                }
            }

        }
        

    }
    // Using back-propagation, the gradient is computed on the devices. After
    // all sources positions have been modeled, transfer back the gradient.
    if (m->GRADOUT==1 && m->BACK_PROP_TYPE==1){
        for (d=0;d<m->NUM_DEVICES;d++){
            for (i=0;i<m->npars;i++){
                if ((*vcl)[d].pars[i].to_grad){
                    __GUARD clbuf_read( &(*vcl)[d].queue,
                                       &(*vcl)[d].pars[i].cl_grad);
                }
                if (m->HOUT==1 && (*vcl)[d].pars[i].to_grad){
                    __GUARD clbuf_read( &(*vcl)[d].queue,
                                       &(*vcl)[d].pars[i].cl_H);
                }
            }
            
        }
        
        for (d=0;d<m->NUM_DEVICES;d++){
            __GUARD clFinish((*vcl)[d].queue);
        }
    }
    
    if (state && m->MPI_INIT==1)
        MPI_Bcast( &state, 1, MPI_INT, m->MYID, MPI_COMM_WORLD );
    
    
    return state;
}

