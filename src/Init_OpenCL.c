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

int Init_OpenCL(struct modcsts * m, struct varcl ** vcl)  {
    /* Function that intialize model decomposition on multiple devices on one
       host. All OpenCL buffers and kernels are created and are contained in a
       varcl structure (one for each devices)
     */
    int state=0;
    int dimbool;
    int i,j,d;
    int parsize=0;
    int fdsize=0;
    int slicesize=0;
    int slicesizefd=0;
    cl_platform_id sel_plat_id=0;
    cl_device_id device=0;
    cl_ulong local_mem_size=0;
    cl_ulong required_local_mem_size=0;
    int required_work_size=0;
    size_t workitem_size[MAX_DIMS];
    size_t workgroup_size=0;
    int lsize[MAX_DIMS];
    int gsize[MAX_DIMS];
    int gsize_com1[MAX_DIMS];
    int gsize_com2[MAX_DIMS];
    int gsize_fcom[MAX_DIMS];
    int LCOMM=0;
    int offcom1=0;
    int offcom2=0;
    struct varcl * vd;
    
    // Find a platform where the prefered device type can be found
    __GUARD GetPlatformID( &m->pref_device_type, &m->device_type, &sel_plat_id,
                           &m->NUM_DEVICES, m->n_no_use_GPUs, m->no_use_GPUs);
    if (m->NUM_DEVICES>m->nmax_dev)
        m->NUM_DEVICES=m->nmax_dev;

    //For each GPU, allocate the memory structures
    GMALLOC((*vcl),sizeof(struct varcl)*m->NUM_DEVICES);

    //Connect all GPUs
    __GUARD connect_allgpus(vcl, &m->context, &m->device_type, &sel_plat_id,
                            m->n_no_use_GPUs, m->no_use_GPUs,m->nmax_dev);
    
    
    //For each device, create the memory buffers and programs on the GPU
    for (d=0; d<m->NUM_DEVICES; d++) {
        
        vd=&(*vcl)[d];
        /* The domain of the FD simulation is decomposed between the devices and
        *  MPI processes along the X direction. We must compute the subdomain 
        *  size, and its location in the domain*/
        if (!state){
            vd->DEV=d;
            vd->NDIM=m->NDIM;
            //Only the last dimensions changes due to domain decomposition
            for (i=0;i<m->NDIM-1;i++){
                vd->N[i]=m->N[i];
                vd->N_names[i]=m->N_names[i];
            }
            //We first decompose between MPI processess
            if (m->MYLOCALID<m->N[m->NDIM-1]%m->NLOCALP){
                vd->N[m->NDIM-1]=m->N[m->NDIM-1]/m->NLOCALP+1;
                m->NXP=m->N[m->NDIM-1]/m->NLOCALP+1;
            }
            else{
                vd->N[m->NDIM-1]=m->N[m->NDIM-1]/m->NLOCALP;
                m->NXP=m->N[m->NDIM-1]/m->NLOCALP;
            }
            //Then decompose between devices
            if (d<vd->N[m->NDIM-1]%m->NUM_DEVICES)
                vd->N[m->NDIM-1]= vd->N[m->NDIM-1]/m->NUM_DEVICES+1;
            else
                vd->N[m->NDIM-1]= vd->N[m->NDIM-1]/m->NUM_DEVICES;

            vd->OFFSET=0;
            vd->NX0=0;
            vd->OFFSETfd=0;

            //Some useful sizes can now be computed for this device
            parsize=1;
            fdsize=1;
            slicesize=1;
            slicesizefd=1;
            for (i=0;i<m->NDIM;i++){
                fdsize*=vd->N[i]+m->FDORDER;
                parsize*=vd->N[i];
            }
            for (i=0;i<m->NDIM-1;i++){
                slicesizefd*=vd->N[i]+m->FDORDER;
                slicesize*=vd->N[i];
            }
            
            //Offset is the location in the model for this device from address 0
            // NX0 is the x location of the first element for this device
            for (i=0;i<m->MYLOCALID;i++){
                if (i<m->N[m->NDIM-1]%m->NLOCALP){
                    vd->OFFSET+=(m->N[m->NDIM-1]/m->NLOCALP+1)*slicesize;
                    vd->NX0+=(m->N[m->NDIM-1]/m->NLOCALP+1);
                }
                else{
                    vd->OFFSET+=(m->N[m->NDIM-1]/m->NLOCALP)*slicesize;
                    vd->NX0+=(m->N[m->NDIM-1]/m->NLOCALP);
                }
                
            }
            for (i=0;i<d;i++){
                vd->NX0+=(*vcl)[i].N[m->NDIM-1];
                vd->OFFSET+=(*vcl)[i].N[m->NDIM-1]*slicesize;
                vd->OFFSETfd+=((*vcl)[i].N[m->NDIM-1]+m->FDORDER)*slicesizefd;
            }
            
        }

        // Get some properties of the device, to define the work sizes
        {
            __GUARD clGetCommandQueueInfo( vd->queue, CL_QUEUE_DEVICE,
                                           sizeof(cl_device_id), &device, NULL);
            __GUARD clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                    sizeof(workitem_size), &workitem_size,NULL);
            __GUARD clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                    sizeof(workgroup_size),&workgroup_size,NULL);
            __GUARD clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                                    sizeof(local_mem_size),&local_mem_size,NULL);
            if (state !=CL_SUCCESS) fprintf(stderr,"%s\n",cl_err_code(state));
            
            // Intel SDK doesn't give the right max work_group_size our kernels,
            // we force it here!
            if (!state && m->pref_device_type==CL_DEVICE_TYPE_CPU)
                workgroup_size= workgroup_size>1024 ? 1024:workgroup_size;
            if (!state && m->pref_device_type==CL_DEVICE_TYPE_ACCELERATOR)
                workgroup_size= workgroup_size>1024 ? 1024:workgroup_size;
        }
        
        // Define the local work size of the update kernels.
        //By default, it is 32 elements long to have coalesced memory in cuda
        //Local memory usage must fit the size of local memory of the device
        if (!state){
            dimbool=0;
            for (i=0;i<m->NDIM;i++){
                if (workitem_size[i]<2)
                    dimbool=1;
            }
            if (workitem_size[0]<m->FDORDER || dimbool){
                fprintf(stdout,"Maximum device work item size of device"
                               "%d doesn't support 3D local memory\n"
                               "Switching off local memory optimization\n", d);
                vd->LOCAL_OFF = 1;
                
            }
            else {

                lsize[0]=32;
                for (i=1;i<m->NDIM;i++){
                    lsize[i]=16;
                }
                required_local_mem_size =sizeof(float);
                for (i=0;i<m->NDIM;i++){
                    required_local_mem_size *= (lsize[i]+m->FDORDER);
                }
                while ( (lsize[1]>(m->FDORDER)/2
                         &&  required_local_mem_size>local_mem_size)
                         || required_work_size>workgroup_size ){
                    required_local_mem_size =sizeof(float);
                    required_work_size=1;
                    for (i=0;i<m->NDIM;i++){
                        if (i>0)
                            lsize[i]-=2;
                        required_local_mem_size *= (lsize[i]+m->FDORDER);
                        required_work_size*=lsize[i];
                    }
                }
                for (j=0;j<m->NDIM;j++){
                    if (required_local_mem_size>local_mem_size){
                        while ( (lsize[j]>(m->FDORDER)/4
                                 &&  required_local_mem_size>local_mem_size)
                                 || required_work_size>workgroup_size ){
                            required_local_mem_size =sizeof(float);
                            required_work_size=1;
                            for (i=0;i<m->NDIM;i++){
                                lsize[j]-=2;
                                required_local_mem_size*=(lsize[i]+m->FDORDER);
                                required_work_size*=lsize[i];
                            }
                        }
                    }
                }
                
                //Check if too many GPUS are used in the domain decomposition
                if  (vd->N[m->NDIM-1]<3*lsize[m->NDIM-1]){
                    vd->LOCAL_OFF = 1;
                    fprintf(stderr,"Too many GPUs for domain decompositon,"
                                   "Switching off local memory optimization\n");
                }
                //Check if the local memory is big enough, else turn it off
                if (required_local_mem_size>local_mem_size){
                    fprintf(stderr,"Local memory needed to perform seismic "
                                   "modeling (%llu bits) exceeds the local "
                                   "memory capacity of device %d (%llu bits)\n"
                                   "Switching off local memory optimization\n",
                                   required_local_mem_size, d, local_mem_size );
                    vd->LOCAL_OFF = 1;
                }
                
            }

            if (vd->LOCAL_OFF==1)
                lsize[0] = 1;
            

        }
        
        // Define the global work size of the update kernels.
        // To overlap computations and communications, we have 3 kernels:
        // One for the interior (no communications) and 2 for the front and back
        // We define here de sizes of those region (offcom1, offcom2)
        if (!state){
            if (vd->LOCAL_OFF==1){
                // If we turn off local memory usage, use one dimension worksize
                LCOMM=0;
                offcom1=0;
                offcom2=0;
                if (d>0 || m->MYLOCALID>0){
                    gsize_com1[0] = slicesize*m->FDOH;
                    LCOMM+=m->FDOH;
                    offcom1=m->FDOH;
                    
                }
                if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                    gsize_com2[0] = slicesize*m->FDOH;
                    LCOMM+=m->FDOH;
                    offcom2=vd->N[m->NDIM-1]-m->FDOH;
                }
                if (d>0 || m->MYLOCALID>0
                        || d<m->NUM_DEVICES-1
                        || m->MYLOCALID<m->NLOCALP-1){
                    gsize_fcom[0] = slicesize*m->FDOH;
                }
                
                gsize[0] = (vd->N[m->NDIM-1]-LCOMM)*slicesize;
                
                vd->workdim=1;
                
            }
            else{

                // When using local work sizes in OpenCL,
                // global work size must be a multiple of local work size
                for (i=0;i<m->NDIM-1;i++){
                    gsize[i]=vd->N[i]
                            +(lsize[i]-vd->N[i]%lsize[i])%lsize[i];
                }
                LCOMM=0;
                offcom1=0;
                offcom2=0;
                if (d>0 || m->MYLOCALID>0){
                    for (i=0;i<m->NDIM-1;i++){
                        gsize_com1[i] = gsize[i];
                    }
                    gsize_com1[m->NDIM-1] = lsize[m->NDIM-1];
                    LCOMM+=lsize[m->NDIM-1];
                    offcom1=(int)lsize[m->NDIM-1];
                    
                }
                if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                    for (i=0;i<m->NDIM-1;i++){
                        gsize_com2[i] = gsize[i];
                    }
                    gsize_com2[m->NDIM-1] = lsize[m->NDIM-1];
                    LCOMM+=lsize[m->NDIM-1];
                    offcom2=vd->N[m->NDIM-1]-(int)lsize[m->NDIM-1];
                }
                if (d>0 || m->MYLOCALID>0
                        || d<m->NUM_DEVICES-1
                        || m->MYLOCALID<m->NLOCALP-1){
                    
                    for (i=0;i<m->NDIM-1;i++){
                        gsize_fcom[i] = vd->N[i];
                    }
                    gsize_fcom[m->NDIM-1] = m->FDOH;
                    
                }

                gsize[m->NDIM-1] = vd->N[m->NDIM-1]-LCOMM
                               +(lsize[m->NDIM-1]-(vd->N[m->NDIM-1]-LCOMM)
                                  %lsize[m->NDIM-1])%lsize[m->NDIM-1];
                
                vd->workdim=m->NDIM;

            }
        }
        
        //Create the required updates struct and assign the working size
        {
            //Struct for the forward modeling
            GMALLOC(vd->ups_f, m->nupdates*sizeof(struct update));
            for (i=0;i<m->nupdates;i++){
                vd->ups_f[i]=m->ups_f[i];
            }
            for (i=0;i<m->nupdates;i++){
                for (j=0;j<vd->workdim;j++){
                    vd->ups_f[i].center.gsize[j]=gsize[j];
                    vd->ups_f[i].center.lsize[j]=lsize[j];
                    vd->ups_f[i].com1.gsize[j]=gsize_com1[j];
                    vd->ups_f[i].com1.lsize[j]=lsize[j];
                    vd->ups_f[i].com2.gsize[j]=gsize_com2[j];
                    vd->ups_f[i].com1.lsize[j]=lsize[j];
                    vd->ups_f[i].fcom1_in.gsize[j]=gsize_fcom[j];
                    vd->ups_f[i].fcom2_in.gsize[j]=gsize_fcom[j];
                    vd->ups_f[i].fcom1_out.gsize[j]=gsize_fcom[j];
                    vd->ups_f[i].fcom2_out.gsize[j]=gsize_fcom[j];
                }
            }
            //Struct for the adjoint modeling
            if (m->GRADOUT){
                GMALLOC(vd->ups_adj, m->nupdates*sizeof(struct update));
                for (i=0;i<m->nupdates;i++){
                    vd->ups_adj[i]=m->ups_adj[i];
                }
                for (i=0;i<m->nupdates;i++){
                    for (j=0;j<vd->workdim;j++){
                        vd->ups_adj[i].center.gsize[j]=gsize[j];
                        vd->ups_adj[i].center.lsize[j]=lsize[j];
                        vd->ups_adj[i].com1.gsize[j]=gsize_com1[j];
                        vd->ups_adj[i].com1.lsize[j]=lsize[j];
                        vd->ups_adj[i].com2.gsize[j]=gsize_com2[j];
                        vd->ups_adj[i].com1.lsize[j]=lsize[j];
                        vd->ups_adj[i].fcom1_in.gsize[j]=gsize_fcom[j];
                        vd->ups_adj[i].fcom2_in.gsize[j]=gsize_fcom[j];
                        vd->ups_adj[i].fcom1_out.gsize[j]=gsize_fcom[j];
                        vd->ups_adj[i].fcom2_out.gsize[j]=gsize_fcom[j];
                    }
                }
            }
        }

        // Create parameter structure for this device, its buffers and transfer
        {
            vd->npars=m->npars;
            GMALLOC(vd->pars, sizeof(struct parameter)*m->npars);

            // Create a pointer at the right position (OFFSET) of the decomposed
            // model and assign memory buffers on the host side for each device
            if (!state){
                for (i=0;i<m->npars;i++){
                    vd->pars[i].gl_par=&m->pars[i].gl_par[vd->OFFSET];
                    vd->pars[i].num_ele=parsize;
                    if (m->pars[i].to_grad){
                        vd->pars[i].gl_grad=&m->pars[i].gl_grad[vd->OFFSET];
                        if (m->pars[i].gl_H)
                            vd->pars[i].gl_H=&m->pars[i].gl_H[vd->OFFSET];
                    }
                }
                
            }
            
            // Create buffers for all parameters and their gradient,
            // and transfer to device parameters
            if (!state){
                for (i=0;i<m->npars;i++){
                    vd->pars[i]=m->pars[i];
                    vd->pars[i].num_ele=parsize;
                    vd->pars[i].cl_par.size=sizeof(float)*parsize;
                    __GUARD create_clbuf(&m->context,&vd->pars[i].cl_par);
                    __GUARD transf_clbuf(&vd->queue,
                                         &vd->pars[i].cl_par,
                                          vd->pars[i].gl_par);
                    if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                        vd->pars[i].cl_grad.size=sizeof(float)*parsize;
                        __GUARD create_clbuf(&m->context, &vd->pars[i].cl_grad);
                    }
                    if (m->HOUT && m->BACK_PROP_TYPE==1){
                        vd->pars[i].cl_H.size=sizeof(float)*parsize;
                        __GUARD create_clbuf( &m->context, &vd->pars[i].cl_H);
                    }
                    
                }
            }
            
        }
        
        //Allocate the variables structure for this device and create buffers
        vd->nvars=m->nvars;
        GMALLOC(vd->vars, sizeof(struct variable)*m->nvars);
        for (i=0;i<m->nvars;i++){
            vd->vars[i]=m->vars[i];
        }
        //TODO: This is model specific, find a better way
        assign_var_size(vd->N,m->NDIM,m->FDORDER, m->nvars, m->L, vd->vars);
        
        //Create OpenCL buffers with the right size
        for (i=0;i<m->nvars;i++){
            
            //Create variable buffers for the interior domain
            vd->vars[i].cl_var.size=sizeof(float)*vd->vars[i].num_ele;
            __GUARD create_clbuf( &m->context, &vd->vars[i].cl_var);
            
            //Create variable buffers for the boundary of the domain
            if ( vd->vars[i].to_comm
                && (d>0 || m->MYLOCALID>0
                    || d<m->NUM_DEVICES-1
                    || m->MYLOCALID<m->NLOCALP-1)){
                    
                    //On the device side
                    vd->vars[i].cl_buf1.size=sizeof(float)*m->FDOH*slicesize;
                    __GUARD create_clbuf_pin(&m->context,
                                             &vd->queuecomm,
                                             &vd->vars[i].cl_buf1,
                                             &vd->vars[i].de_buf1);
                    vd->vars[i].cl_buf2.size=sizeof(float)*m->FDOH*slicesize;
                    
                    //On the host, overlapped transfers need pinned memory
                    __GUARD create_clbuf_pin(&m->context,
                                             &vd->queuecomm,
                                             &vd->vars[i].cl_buf2,
                                             &vd->vars[i].de_buf2);
                    __GUARD create_clbuf(&m->context, &vd->vars[i].cl_buf1_dev);
                    vd->vars[i].cl_buf2_dev.size=sizeof(float)
                                                 * m->FDOH*slicesize;
                    __GUARD create_clbuf(&m->context, &vd->vars[i].cl_buf2_dev);
                }
            
            // Create the buffers to output variables at receivers locations
            if (vd->vars[i].to_output){
                
                //Memory for recordings for this device on host side
                alloc_seismo(&vd->vars[i].de_varout, m);
                
                vd->vars[i].cl_varout.size=sizeof(float)
                                          * m->NT * m->src_recs.ngmax;
                __GUARD create_clbuf( &m->context, &vd->vars[i].cl_varout);
                
                //Create also a buffer for the residuals
                if (m->vars[i].gl_var_res){
                    vd->vars[i].cl_var_res.size=sizeof(float)
                                               * m->NT * m->src_recs.ngmax;
                    __GUARD create_clbuf( &m->context, &vd->vars[i].cl_var_res);
                }
            }
            
            // If we use the DFT for gradient computation,
            // we create the buffers to hold each frequency
            if (m->GRADOUT
                && m->BACK_PROP_TYPE==2
                && vd->vars[i].for_grad){
                
                GMALLOC(vd->vars[i].de_fvar,
                        sizeof(cl_float2)*vd->vars[i].num_ele * m->NFREQS);
                vd->vars[i].cl_fvar.size=sizeof(cl_float2)
                                        * vd->vars[i].num_ele * m->NFREQS;
                __GUARD create_clbuf(&m->context,&vd->vars[i].cl_fvar);
            }
            
            // If we want the movie, allocate memory for variables
            if (m->MOVOUT){
                if (m->vars[i].to_output){
                    vd->vars[i].gl_mov=&m->vars[i].gl_mov[vd->OFFSET];
                    GMALLOC(vd->vars[i].de_mov,
                            vd->vars[i].num_ele*sizeof(cl_float));
                }
            }
            
        }

        
        //Create constants structure and buffers, transfer to device
        vd->ncsts=m->ncsts;
        GMALLOC(vd->csts, sizeof(struct constants)*m->ncsts);
        for (i=0;i<m->ncsts;i++){
            vd->csts[i]=m->csts[i];
            //Size of constants does not depend of domain decomposition
            if (vd->csts[i].active){
                vd->csts[i].cl_cst.size=sizeof(float)*vd->csts[i].num_ele;
                __GUARD create_clbuf( &m->context, &vd->csts[i].cl_cst);
                __GUARD transf_clbuf( &vd->queue, &vd->csts[i].cl_cst,
                                                    m->csts[i].gl_cst);
            }
            
        }
        
        //Set the sources and receivers structure for this device
        {
            vd->src_recs=m->src_recs;
            vd->src_recs.cl_src_pos.size=sizeof(float) * 5 * m->src_recs.nsmax;
            vd->src_recs.cl_rec_pos.size=sizeof(float) * 8 * m->src_recs.ngmax;
            vd->src_recs.cl_src.size=sizeof(float) * m->NT * m->src_recs.nsmax;
            __GUARD create_clbuf_cst( &m->context, &vd->src_recs.cl_src_pos);
            __GUARD create_clbuf_cst( &m->context, &vd->src_recs.cl_rec_pos);
            __GUARD create_clbuf_cst( &m->context, &vd->src_recs.cl_src);
            if (m->GRADSRCOUT){
                vd->src_recs.cl_grad_src.size=sizeof(float)
                                             * m->NT * m->src_recs.nsmax;
                __GUARD create_clbuf_cst(&m->context,&vd->src_recs.cl_grad_src);
            }
        }
        
        // Determine the size of the outside boundary used for the back
        // propagation of the seismic wavefield
        // TODO: this is ugly and model specific, find a better way
        if (m->BACK_PROP_TYPE==1 && m->GRADOUT){
            if (m->ND==3){// For 3D
                if (m->NUM_DEVICES==1 && m->NLOCALP==1)
                    vd->NBND=(vd->N[2]-2*m->NAB)*(vd->N[1]-2*m->NAB)*2*m->FDOH
                    +(vd->N[2]-2*m->NAB)*(vd->N[0]-2*m->NAB-2*m->FDOH)*2*m->FDOH
                    +(vd->N[1]-2*m->NAB-2*m->FDOH)*(vd->N[0]-2*m->NAB-2*m->FDOH)
                    *2*m->FDOH;
                
                else if ( (d==0 && m->MYGROUPID==0)
                         || (d==m->NUM_DEVICES-1 && m->MYGROUPID==m->NLOCALP-1))
                    vd->NBND=(vd->N[2]-m->NAB)*(vd->N[1]-2*m->NAB)*2*m->FDOH
                    +(vd->N[2]-m->NAB)*(vd->N[0]-2*m->NAB-2*m->FDOH)*2*m->FDOH
                    +(vd->N[1]-2*m->NAB-2*m->FDOH)*(vd->N[0]-2*m->NAB-2*m->FDOH)
                    *m->FDOH;
                
                else
                    vd->NBND=vd->N[2]*(vd->N[1]-2*m->NAB)*2*m->FDOH+
                    vd->N[2]*(vd->N[0]-2*m->NAB-2*m->FDOH)*2*m->FDOH;
            }
            else{
                if (m->NUM_DEVICES==1 && m->NLOCALP==1)
                    vd->NBND=(vd->N[1]-2*m->NAB)*2*m->FDOH+
                    (vd->N[0]-2*m->NAB-2*m->FDOH)*2*m->FDOH;
                
                else if ( (d==0 && m->MYGROUPID==0)
                         || (d==m->NUM_DEVICES-1 && m->MYGROUPID==m->NLOCALP-1))
                    vd->NBND=(vd->N[1]-m->NAB)*m->FDOH+
                    (vd->N[0]-2*m->NAB-m->FDOH)*2*m->FDOH;
                
                else
                    vd->NBND= (vd->N[1])*2*m->FDOH;
            }
            
            vd->grads.savebnd.gsize[0]=vd->NBND;

            for (i=0;i<m->nvars;i++){
                if (vd->vars[i].to_comm){
                    vd->vars[i].cl_varbnd.size=sizeof(float) * vd->NBND;
                    __GUARD create_clbuf( &m->context, &vd->vars[i].cl_varbnd);
                    vd->vars[i].cl_varbnd_pin.size=sizeof(float) * vd->NBND;
                    __GUARD create_clbuf_pin(&m->context, &vd->queuecomm,
                                             &vd->vars[i].cl_varbnd_pin,
                                             &vd->vars[i].de_varbnd);
                }
            }
            
            
            GMALLOC(vd->vars_adj, sizeof(struct variable)*m->nvars);
            for (i=0;i<m->nvars;i++){
                vd->vars_adj[i]=vd->vars[i];
            }
            
            for (i=0;i<m->nvars;i++){
                __GUARD create_clbuf( &m->context, &vd->vars_adj[i].cl_var);
                if (vd->vars[i].to_comm && (d>0
                                            || m->MYLOCALID>0
                                            || d<m->NUM_DEVICES-1
                                            || m->MYLOCALID<m->NLOCALP-1)){
                    
                    __GUARD create_clbuf_pin(&m->context, &vd->queuecomm,
                                             &vd->vars_adj[i].cl_buf1,
                                             &vd->vars_adj[i].de_buf1);
                    __GUARD create_clbuf_pin(&m->context, &vd->queuecomm,
                                             &vd->vars_adj[i].cl_buf2,
                                             &vd->vars_adj[i].de_buf2);
                    __GUARD create_clbuf( &m->context,
                                          &vd->vars_adj[i].cl_buf1_dev);
                    __GUARD create_clbuf( &m->context,
                                          &vd->vars_adj[i].cl_buf2_dev);
                }
            }

        }
        
        // Create the update kernels
        vd->nupdates=m->nupdates;
        for (i=0;i<m->nupdates;i++){
            vd->ups_f[i].center.OFFCOMM=offcom1;
            vd->ups_f[i].center.LCOMM=LCOMM;
            __GUARD create_kernel(m, vd,  &vd->ups_f[i].center);
            if (d>0 || m->MYLOCALID>0){
                vd->ups_f[i].com1.OFFCOMM=0;
                vd->ups_f[i].com1.LCOMM=LCOMM;
                vd->ups_f[i].com1.COMM=1;
                __GUARD create_kernel(m, vd,  &vd->ups_f[i].com1);
                
                __GUARD kernel_fcom_out( vd ,vd->vars,
                                        &vd->ups_f[i].fcom1_out, i+1, 1);
                __GUARD create_kernel(m, vd,  &vd->ups_f[i].fcom1_out);
                __GUARD kernel_fcom_in( vd ,vd->vars,
                                       &vd->ups_f[i].fcom1_in, i+1, 1);
                __GUARD create_kernel(m, vd,  &vd->ups_f[i].fcom1_in);
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                vd->ups_f[i].com2.OFFCOMM=offcom2;
                vd->ups_f[i].com2.LCOMM=LCOMM;
                vd->ups_f[i].com2.COMM=1;
                __GUARD create_kernel(m, vd,  &vd->ups_f[i].com2);
                
                __GUARD kernel_fcom_out( vd, vd->vars,
                                     &vd->ups_f[i].fcom2_out, i+1, 2);
                __GUARD create_kernel(m, vd,  &vd->ups_f[i].fcom2_out);
                __GUARD kernel_fcom_in(vd ,vd->vars,
                                   &vd->ups_f[i].fcom2_in, i+1, 2);
                __GUARD create_kernel(m, vd,  &vd->ups_f[i].fcom2_in);
            }
        }


        
        
        //Create automaticly kernels for gradient, variable inti, sources ...
        __GUARD kernel_varout(vd, vd->vars, &vd->src_recs.varsout);
        __GUARD create_kernel(m, vd,  &vd->src_recs.varsout);
        
        __GUARD kernel_varoutinit(vd, vd->vars, &vd->src_recs.varsoutinit);
        __GUARD create_kernel(m, vd,  &vd->src_recs.varsoutinit);
        
        __GUARD kernel_varinit(vd, vd->vars, &vd->bnd_cnds.init_f);
        __GUARD create_kernel(m, vd,  &vd->bnd_cnds.init_f);
        
        
        if (m->GRADOUT){
            __GUARD kernel_residuals(vd, vd->vars, &vd->src_recs.residuals);
            __GUARD create_kernel(m, vd,  &vd->src_recs.residuals);
            
            if (m->BACK_PROP_TYPE==1){
                __GUARD kernel_varinit(vd,vd->vars_adj, &vd->bnd_cnds.init_adj);
                __GUARD create_kernel(m, vd,  &vd->bnd_cnds.init_adj);
                
                __GUARD kernel_gradinit(vd, vd->pars, &vd->grads.init);
                __GUARD create_kernel(m, vd,  &vd->grads.init);
            }
            else if(m->BACK_PROP_TYPE==2){
                __GUARD kernel_initsavefreqs(vd, vd->vars, &vd->grads.initsavefreqs);
                __GUARD create_kernel(m, vd,  &vd->grads.initsavefreqs);
                
                kernel_savefreqs(vd, vd->vars, &vd->grads.savefreqs);
                __GUARD create_kernel(m, vd,  &vd->grads.savefreqs);
            }
            
            if (m->GRADSRCOUT){
                __GUARD kernel_init_gradsrc( &vd->src_recs.init_gradsrc);
                __GUARD create_kernel(m, vd,  &vd->src_recs.init_gradsrc);
            }
        }
    
        
        //TODO Boundary conditions should be included in the update kernel
        //TODO Adjoint free surface
        if (m->FREESURF){
            __GUARD create_kernel(m, vd,  &vd->bnd_cnds.surf);
        }
        
        //TODO Create automatically the kernel for saving boundary
        //TODO Implement random boundaries instead
        if (m->GRADOUT && m->BACK_PROP_TYPE==1){
            __GUARD create_kernel(m, vd,  &vd->grads.savebnd);
        }
        
        
    }

    if (state && m->MPI_INIT==1)
        MPI_Bcast( &state, 1, MPI_INT, m->MYID, MPI_COMM_WORLD );
    
    return state;


}
