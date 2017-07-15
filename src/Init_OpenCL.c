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

    int state=0;
    int dimbool;
    int i,j,d;
    int paramsize=0;
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
    int gsize_comm1[MAX_DIMS];
    int gsize_comm2[MAX_DIMS];
    int gsize_fillcomm[MAX_DIMS];
    int LCOMM=0;
    int offcomm1=0;
    int offcomm2=0;
    
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
        

        /* The domain of the FD simulation is decomposed between the devices and
        *  MPI processes along the X direction. We must compute the subdomain 
        *  size, and its location in the domain*/
        if (!state){
            (*vcl)[d].DEV=d;
            //Only the last dimensions changes due to domain decomposition
            for (i=0;i<m->NDIM-1;i++){
                (*vcl)[d].N[i]=m->N[i];
            }
            //We first decompose between MPI processess
            if (m->MYLOCALID<m->N[m->NDIM-1]%m->NLOCALP){
                (*vcl)[d].N[m->NDIM-1]=m->N[m->NDIM-1]/m->NLOCALP+1;
                m->NXP=m->N[m->NDIM-1]/m->NLOCALP+1;
            }
            else{
                (*vcl)[d].N[m->NDIM-1]=m->N[m->NDIM-1]/m->NLOCALP;
                m->NXP=m->N[m->NDIM-1]/m->NLOCALP;
            }
            //Then decompose between devices
            if (d<(*vcl)[d].N[m->NDIM-1]%m->NUM_DEVICES)
                (*vcl)[d].N[m->NDIM-1]= (*vcl)[d].N[m->NDIM-1]/m->NUM_DEVICES+1;
            else
                (*vcl)[d].N[m->NDIM-1]= (*vcl)[d].N[m->NDIM-1]/m->NUM_DEVICES;

            (*vcl)[d].OFFSET=0;
            (*vcl)[d].NX0=0;
            (*vcl)[d].OFFSETfd=0;

            //Some useful sizes can now be computed for this device
            paramsize=1;
            fdsize=1;
            slicesize=1;
            slicesizefd=1;
            for (i=0;i<m->NDIM;i++){
                fdsize*=(*vcl)[d].N[i]+m->FDORDER;
                paramsize*=(*vcl)[d].N[i];
            }
            for (i=0;i<m->NDIM-1;i++){
                slicesizefd*=(*vcl)[d].N[i]+m->FDORDER;
                slicesize*=(*vcl)[d].N[i];
            }
            
            //Offset is the location in the model for this device from address 0
            // NX0 is the x location of the first element for this device
            for (i=0;i<m->MYLOCALID;i++){
                if (i<m->N[m->NDIM-1]%m->NLOCALP){
                    (*vcl)[d].OFFSET+=(m->N[m->NDIM-1]/m->NLOCALP+1)*slicesize;
                    (*vcl)[d].NX0+=(m->N[m->NDIM-1]/m->NLOCALP+1);
                }
                else{
                    (*vcl)[d].OFFSET+=(m->N[m->NDIM-1]/m->NLOCALP)*slicesize;
                    (*vcl)[d].NX0+=(m->N[m->NDIM-1]/m->NLOCALP);
                }
                
            }
            for (i=0;i<d;i++){
                (*vcl)[d].NX0+=(*vcl)[i].N[m->NDIM-1];
                (*vcl)[d].OFFSET+=(*vcl)[i].N[m->NDIM-1]*slicesize;
                (*vcl)[d].OFFSETfd+=((*vcl)[i].N[m->NDIM-1]+m->FDORDER)*slicesizefd;
            }
            
        }

        // Get some properties of the device, to define the work sizes
        {
            __GUARD clGetCommandQueueInfo(	(*vcl)[d].cmd_queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL);
            __GUARD clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
            __GUARD clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
            __GUARD clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
            if (state !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(state));
            
            //Intel SDK does not give the right max work_group_size our kernels, we force it here!
            if (!state && m->pref_device_type==CL_DEVICE_TYPE_CPU) workgroup_size= workgroup_size>1024 ? 1024:workgroup_size;
            if (!state && m->pref_device_type==CL_DEVICE_TYPE_ACCELERATOR) workgroup_size= workgroup_size>1024 ? 1024:workgroup_size;
        }
        
        // Define the local work size of the update kernels.
        //By default, it is 32 elements long to have coalesced memory in cuda
        //Local memory usage must fit with the size of local memory of the device
        if (!state){
            dimbool=0;
            for (i=0;i<m->NDIM;i++){
                if (workitem_size[i]<2)
                    dimbool=1;
            }
            if (workitem_size[0]<m->FDORDER || dimbool){
                fprintf(stdout,"Maximum device work item size of device %d doesn't support 3D local memory\n", d);
                fprintf(stdout,"Switching off local memory optimization\n");
                (*vcl)[d].LOCAL_OFF = 1;
                
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
                                required_local_mem_size *= (lsize[i]+m->FDORDER);
                                required_work_size*=lsize[i];
                            }
                        }
                    }
                }
                
                //Check if too many GPUS are used in the domain decomposition
                if  ((*vcl)[d].N[m->NDIM-1]<3*lsize[m->NDIM-1]){
                    (*vcl)[d].LOCAL_OFF = 1;
                    fprintf(stderr,"Too many GPUs for domain decompositon, Switching off local memory optimization\n");
                }
                //Check if the local memory is big enough, else turn off local memory usage
                if (required_local_mem_size>local_mem_size){
                    fprintf(stderr,"Local memory needed to perform seismic modeling (%llu bits) exceeds the local memory capacity of device %d (%llu bits)\n", required_local_mem_size, d, local_mem_size );
                    fprintf(stderr,"Switching off local memory optimization\n");
                    (*vcl)[d].LOCAL_OFF = 1;
                }
                
            }

            if ((*vcl)[d].LOCAL_OFF==1)
                lsize[0] = 1;
            

        }
        
        // Define the global work size of the update kernels.
        // To overlap computations and communications, we have 3 kernels:
        // One for the interior (no communications) and 2 for the front and back boundaries in x
        // We define here de sizes of those region (offcomm1, offcomm2)
        if (!state){
            if ((*vcl)[d].LOCAL_OFF==1){
                // If we turn off local memory usage, we only use a work size with one dimension.
                LCOMM=0;
                offcomm1=0;
                offcomm2=0;
                if (d>0 || m->MYLOCALID>0){
                    gsize_comm1[0] = slicesize*m->FDOH;
                    LCOMM+=m->FDOH;
                    offcomm1=m->FDOH;
                    
                }
                if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                    gsize_comm2[0] = slicesize*m->FDOH;
                    LCOMM+=m->FDOH;
                    offcomm2=(*vcl)[d].N[m->NDIM-1]-m->FDOH;
                }
                if (d>0 || m->MYLOCALID>0 || d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                    gsize_fillcomm[0] = slicesize*m->FDOH;
                }
                
                gsize[0] = ((*vcl)[d].N[m->NDIM-1]-LCOMM)*slicesize;
                
                (*vcl)[d].NDIM=1;
                
            }
            else{

                //When using local work sizes in Opecn, global work size must be a multiple of local work size
                for (i=0;i<m->NDIM-1;i++){
                    gsize[i]=(*vcl)[d].N[i]+ (lsize[i]-(*vcl)[d].N[i]%lsize[i])%lsize[i];
                }
                LCOMM=0;
                offcomm1=0;
                offcomm2=0;
                if (d>0 || m->MYLOCALID>0){
                    for (i=0;i<m->NDIM-1;i++){
                        gsize_comm1[i] = gsize[i];
                    }
                    gsize_comm1[m->NDIM-1] = lsize[m->NDIM-1];
                    LCOMM+=lsize[m->NDIM-1];
                    offcomm1=(int)lsize[m->NDIM-1];
                    
                }
                if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                    for (i=0;i<m->NDIM-1;i++){
                        gsize_comm2[i] = gsize[i];
                    }
                    gsize_comm2[m->NDIM-1] = lsize[m->NDIM-1];
                    LCOMM+=lsize[m->NDIM-1];
                    offcomm2=(*vcl)[d].N[m->NDIM-1]-(int)lsize[m->NDIM-1];
                }
                if (d>0 || m->MYLOCALID>0 || d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                    for (i=0;i<m->NDIM-1;i++){
                        gsize_fillcomm[i] = (*vcl)[d].N[i];
                    }
                    gsize_fillcomm[m->NDIM-1] = m->FDOH;
                    
                }

                gsize[m->NDIM-1] = (*vcl)[d].N[m->NDIM-1]-LCOMM
                +(lsize[m->NDIM-1]-((*vcl)[d].N[m->NDIM-1]-LCOMM)%lsize[m->NDIM-1])%lsize[m->NDIM-1];
                
                (*vcl)[d].NDIM=m->NDIM;

            }
        }
        
        //Create the required updates struct and assign the working size
        {
            //Struct for the forward modeling
            GMALLOC((*vcl)[d].updates_f, m->nupdates*sizeof(struct update));
            for (i=0;i<m->nupdates;i++){
                (*vcl)[d].updates_f[i]=m->updates_f[i];
            }
            for (i=0;i<m->nupdates;i++){
                for (j=0;j<(*vcl)[d].NDIM;j++){
                    (*vcl)[d].updates_f[i].center.gsize[j]=gsize[j];
                    (*vcl)[d].updates_f[i].center.lsize[j]=lsize[j];
                    (*vcl)[d].updates_f[i].comm1.gsize[j]=gsize_comm1[j];
                    (*vcl)[d].updates_f[i].comm1.lsize[j]=lsize[j];
                    (*vcl)[d].updates_f[i].comm2.gsize[j]=gsize_comm2[j];
                    (*vcl)[d].updates_f[i].comm1.lsize[j]=lsize[j];
                    (*vcl)[d].updates_f[i].fill_buff1_in.gsize[j]=gsize_fillcomm[j];
                    (*vcl)[d].updates_f[i].fill_buff2_in.gsize[j]=gsize_fillcomm[j];
                    (*vcl)[d].updates_f[i].fill_buff1_out.gsize[j]=gsize_fillcomm[j];
                    (*vcl)[d].updates_f[i].fill_buff2_out.gsize[j]=gsize_fillcomm[j];
                }
            }
            //Struct for the adjoint modeling
            if (m->GRADOUT){
                GMALLOC((*vcl)[d].updates_adj, m->nupdates*sizeof(struct update));
                for (i=0;i<m->nupdates;i++){
                    (*vcl)[d].updates_adj[i]=m->updates_adj[i];
                }
                for (i=0;i<m->nupdates;i++){
                    for (j=0;j<(*vcl)[d].NDIM;j++){
                        (*vcl)[d].updates_adj[i].center.gsize[j]=gsize[j];
                        (*vcl)[d].updates_adj[i].center.lsize[j]=lsize[j];
                        (*vcl)[d].updates_adj[i].comm1.gsize[j]=gsize_comm1[j];
                        (*vcl)[d].updates_adj[i].comm1.lsize[j]=lsize[j];
                        (*vcl)[d].updates_adj[i].comm2.gsize[j]=gsize_comm2[j];
                        (*vcl)[d].updates_adj[i].comm1.lsize[j]=lsize[j];
                        (*vcl)[d].updates_adj[i].fill_buff1_in.gsize[j]=gsize_fillcomm[j];
                        (*vcl)[d].updates_adj[i].fill_buff2_in.gsize[j]=gsize_fillcomm[j];
                        (*vcl)[d].updates_adj[i].fill_buff1_out.gsize[j]=gsize_fillcomm[j];
                        (*vcl)[d].updates_adj[i].fill_buff2_out.gsize[j]=gsize_fillcomm[j];
                    }
                }
            }
        }

        //Allocate the parameter structure for this device, create OpenCL buffers, transfer to device
        {
            GMALLOC((*vcl)[d].params, sizeof(struct parameter)*m->nparams);

            // Create a pointer at the right position (OFFSET) of the decomposed model
            // and assign memory buffers on the host side for each device
            if (!state){
                for (i=0;i<m->nparams;i++){
                    (*vcl)[d].params[i].gl_param=&m->params[i].gl_param[(*vcl)[d].OFFSET];
                    (*vcl)[d].params[i].num_ele=paramsize;
                    if (m->params[i].to_grad){
                        (*vcl)[d].params[i].gl_grad=&m->params[i].gl_grad[(*vcl)[d].OFFSET];
                        if (m->params[i].gl_H)
                            (*vcl)[d].params[i].gl_H=&m->params[i].gl_H[(*vcl)[d].OFFSET];
                    }
                }
                
            }
            
            //Create the OpenCL buffers for all parameters and their gradient, and transfer to device parameters
            if (!state){
                for (i=0;i<m->nparams;i++){
                    (*vcl)[d].params[i]=m->params[i];
                    (*vcl)[d].params[i].num_ele=paramsize;
                    (*vcl)[d].params[i].cl_param.size=sizeof(float)*paramsize;
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].params[i].cl_param.size,    &(*vcl)[d].params[i].cl_param.mem);
                    __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].params[i].cl_param.size,    &(*vcl)[d].params[i].cl_param.mem, (*vcl)[d].params[i].gl_param);
                    if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                        (*vcl)[d].params[i].cl_grad.size=sizeof(float)*paramsize;
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].params[i].cl_grad.size,    &(*vcl)[d].params[i].cl_grad.mem);
                    }
                    if (m->HOUT && m->BACK_PROP_TYPE==1){
                        (*vcl)[d].params[i].cl_H.size=sizeof(float)*paramsize;
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].params[i].cl_H.size,    &(*vcl)[d].params[i].cl_H.mem);
                    }
                    
                }
            }
            
        }
        
        //Allocate the variables structure for this device and create OpenCL buffers
        {
            GMALLOC((*vcl)[d].vars, sizeof(struct variable)*m->nvars);
            for (i=0;i<m->nvars;i++){
                (*vcl)[d].vars[i]=m->vars[i];
            }
            //TODO: This is model specific, find a better way
            assign_var_size( (*vcl)[d].N, m->NDIM, m->FDORDER, m->nvars, m->L, (*vcl)[d].vars);
            
            //Create OpenCL buffers with the right size
            for (i=0;i<m->nvars;i++){
                
                //Create variable buffers for the interior domain
                (*vcl)[d].vars[i].cl_var.size=sizeof(float)*(*vcl)[d].vars[i].num_ele;
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].vars[i].cl_var.size,    &(*vcl)[d].vars[i].cl_var.mem);
                
                //Create variable buffers for the boundary of the domain
                if ((*vcl)[d].vars[i].to_comm && (d>0 || m->MYLOCALID>0 || d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1)){
                    //On the device side
                    (*vcl)[d].vars[i].cl_var_sub1.size=sizeof(float) * m->FDOH*slicesize;
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].vars[i].cl_var_sub1.size, &(*vcl)[d].vars[i].cl_var_sub1.mem, &(*vcl)[d].vars[i].de_var_sub1);
                    (*vcl)[d].vars[i].cl_var_sub2.size=sizeof(float) * m->FDOH*slicesize;
                    
                    //On the host side, memory should be pinned for overlapped transfers
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].vars[i].cl_var_sub2.size, &(*vcl)[d].vars[i].cl_var_sub2.mem, &(*vcl)[d].vars[i].de_var_sub2);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].vars[i].cl_var_sub1_dev.size,    &(*vcl)[d].vars[i].cl_var_sub1_dev.mem);
                    (*vcl)[d].vars[i].cl_var_sub2_dev.size=sizeof(float) * m->FDOH*slicesize;
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].vars[i].cl_var_sub2_dev.size,    &(*vcl)[d].vars[i].cl_var_sub2_dev.mem);
                }
                
                // Create the buffers to output variables at receivers locations
                if ((*vcl)[d].vars[i].to_output){
                    
                    //Memory to hold all the shots records for this device on host side
                    alloc_seismo(&(*vcl)[d].vars[i].de_varout, m->ns, m->allng, m->NT, m->src_recs.nrec);
                    
                    (*vcl)[d].vars[i].cl_varout.size=sizeof(float) * m->NT * m->src_recs.ngmax;
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].vars[i].cl_varout.size, &(*vcl)[d].vars[i].cl_varout.mem);
                    
                    //Create also a buffer for the residuals
                    if (m->vars[i].gl_var_res){
                        (*vcl)[d].vars[i].cl_var_res.size=sizeof(float) * m->NT * m->src_recs.ngmax;
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].vars[i].cl_var_res.size, &(*vcl)[d].vars[i].cl_var_res.mem);
                    }
                }
                
                // If we use the DFT for gradient computation, we create the buffers to hold each frequency
                if (m->GRADOUT && m->BACK_PROP_TYPE==2 && (*vcl)[d].vars[i].for_grad){
                    GMALLOC( (*vcl)[d].vars[i].de_fvar, sizeof(cl_float2)*(*vcl)[d].vars[i].num_ele* m->NFREQS);
                    (*vcl)[d].vars[i].cl_fvar.size=sizeof(cl_float2)*(*vcl)[d].vars[i].num_ele* m->NFREQS;
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].vars[i].cl_fvar.size, &(*vcl)[d].vars[i].cl_fvar.mem);
                }
                
                // If we want the movie, allocate memory for variables for this device on host
                if (m->MOVOUT){
                    if (m->vars[i].to_output){
                        (*vcl)[d].vars[i].gl_mov=&m->vars[i].gl_mov[(*vcl)[d].OFFSET];
                        GMALLOC((*vcl)[d].vars[i].de_mov,(*vcl)[d].vars[i].num_ele*sizeof(cl_float));
                    }
                }
                
            }
        }

        //Allocate the constants structure for this device, create OpenCL buffers, transfer to device
        {
            GMALLOC((*vcl)[d].csts, sizeof(struct constants)*m->ncsts);
            for (i=0;i<m->ncsts;i++){
                (*vcl)[d].csts[i]=m->csts[i];
                //Size of constants does not depend of domain decomposition (fixed across all devices)
                if ((*vcl)[d].csts[i].active){
                    (*vcl)[d].csts[i].cl_cst.size=sizeof(float)*(*vcl)[d].csts[i].num_ele;
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].csts[i].cl_cst.size,    &(*vcl)[d].csts[i].cl_cst.mem);
                    __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].csts[i].cl_cst.size,    &(*vcl)[d].csts[i].cl_cst.mem, m->csts[i].gl_cst);
                }
                
            }
        }
        
        //Set the sources and receivers structure for this device and create OpenCL buffers
        {
            (*vcl)[d].src_recs=m->src_recs;
            (*vcl)[d].src_recs.cl_src_pos.size=sizeof(float) * 5 * m->src_recs.nsmax;
            (*vcl)[d].src_recs.cl_rec_pos.size=sizeof(float) * 8 * m->src_recs.ngmax;
            (*vcl)[d].src_recs.cl_src.size=sizeof(float) * m->NT * m->src_recs.nsmax;
            __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].src_recs.cl_src_pos.size,     &(*vcl)[d].src_recs.cl_src_pos.mem);
            __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].src_recs.cl_rec_pos.size,     &(*vcl)[d].src_recs.cl_rec_pos.mem);
            __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].src_recs.cl_src.size, &(*vcl)[d].src_recs.cl_src.mem);
            if (m->GRADSRCOUT){
                (*vcl)[d].src_recs.cl_grad_src.size=sizeof(float) * m->NT * m->src_recs.nsmax;
                __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].src_recs.cl_grad_src.size, &(*vcl)[d].src_recs.cl_grad_src.mem);
            }
        }
        
        // Determine the size of the outside boundary used for the back propagation of the seismic wavefield
        // TODO: this is ugly and model specific, find a better way
        if (m->BACK_PROP_TYPE==1 && m->GRADOUT){
            if (m->ND==3){// For 3D
                if (m->NUM_DEVICES==1 && m->NLOCALP==1)
                    (*vcl)[d].NBND=((*vcl)[d].N[2]-2*m->NAB)*((*vcl)[d].N[1]-2*m->NAB)*2*m->FDOH+
                    ((*vcl)[d].N[2]-2*m->NAB)*((*vcl)[d].N[0]-2*m->NAB-2*m->FDOH)*2*m->FDOH+
                    ((*vcl)[d].N[1]-2*m->NAB-2*m->FDOH)*((*vcl)[d].N[0]-2*m->NAB-2*m->FDOH)*2*m->FDOH;
                
                else if ( (d==0 && m->MYGROUPID==0) || (d==m->NUM_DEVICES-1 && m->MYGROUPID==m->NLOCALP-1) )
                    (*vcl)[d].NBND=((*vcl)[d].N[2]-m->NAB)*((*vcl)[d].N[1]-2*m->NAB)*2*m->FDOH+
                    ((*vcl)[d].N[2]-m->NAB)*((*vcl)[d].N[0]-2*m->NAB-2*m->FDOH)*2*m->FDOH+
                    ((*vcl)[d].N[1]-2*m->NAB-2*m->FDOH)*((*vcl)[d].N[0]-2*m->NAB-2*m->FDOH)*m->FDOH;
                
                else
                    (*vcl)[d].NBND=(*vcl)[d].N[2]*((*vcl)[d].N[1]-2*m->NAB)*2*m->FDOH+
                    (*vcl)[d].N[2]*((*vcl)[d].N[0]-2*m->NAB-2*m->FDOH)*2*m->FDOH;
            }
            else{
                if (m->NUM_DEVICES==1 && m->NLOCALP==1)
                    (*vcl)[d].NBND=((*vcl)[d].N[1]-2*m->NAB)*2*m->FDOH+
                    ((*vcl)[d].N[0]-2*m->NAB-2*m->FDOH)*2*m->FDOH;
                
                else if ( (d==0 && m->MYGROUPID==0) || (d==m->NUM_DEVICES-1 && m->MYGROUPID==m->NLOCALP-1) )
                    (*vcl)[d].NBND=((*vcl)[d].N[1]-m->NAB)*m->FDOH+
                    ((*vcl)[d].N[0]-2*m->NAB-m->FDOH)*2*m->FDOH;
                
                else
                    (*vcl)[d].NBND= ((*vcl)[d].N[1])*2*m->FDOH;
            }
            
            (*vcl)[d].grads.savebnd.gsize[0]=(*vcl)[d].NBND;

            for (i=0;i<m->nvars;i++){
                if ((*vcl)[d].vars[i].to_comm){
                    (*vcl)[d].vars[i].cl_varbnd.size=sizeof(float) * (*vcl)[d].NBND;
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].vars[i].cl_varbnd.size,    &(*vcl)[d].vars[i].cl_varbnd.mem);
                    (*vcl)[d].vars[i].cl_varbnd_pinned.size=sizeof(float) * (*vcl)[d].NBND;
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].vars[i].cl_varbnd_pinned.size, &(*vcl)[d].vars[i].cl_varbnd_pinned.mem, &(*vcl)[d].vars[i].de_varbnd);
                }
            }
            
            
            GMALLOC((*vcl)[d].vars_adj, sizeof(struct variable)*m->nvars);
            for (i=0;i<m->nvars;i++){
                (*vcl)[d].vars_adj[i]=(*vcl)[d].vars[i];
            }
            
            for (i=0;i<m->nvars;i++){
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].vars_adj[i].cl_var.size,    &(*vcl)[d].vars_adj[i].cl_var.mem);
                if ((*vcl)[d].vars[i].to_comm && (d>0 || m->MYLOCALID>0 || d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1)){
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].vars_adj[i].cl_var_sub1.size, &(*vcl)[d].vars_adj[i].cl_var_sub1.mem, &(*vcl)[d].vars_adj[i].de_var_sub1);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].vars_adj[i].cl_var_sub2.size, &(*vcl)[d].vars_adj[i].cl_var_sub2.mem, &(*vcl)[d].vars_adj[i].de_var_sub2);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].vars_adj[i].cl_var_sub1_dev.size,    &(*vcl)[d].vars_adj[i].cl_var_sub1_dev.mem);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].vars_adj[i].cl_var_sub2_dev.size,    &(*vcl)[d].vars_adj[i].cl_var_sub2_dev.mem);
                }
            }

        }
        
        // Create the update kernels
        for (i=0;i<m->nupdates;i++){
            __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].updates_f[i].center, offcomm1, LCOMM, 0, 0);
            if (d>0 || m->MYLOCALID>0){
                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].updates_f[i].comm1, 0, LCOMM, 1, 0 );
                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].updates_f[i].fill_buff1_in, 0, 0, 0, 0 );
                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].updates_f[i].fill_buff1_out, 0, 0, 0, 0 );
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].updates_f[i].comm1, offcomm2, LCOMM, 1, 0 );
                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].updates_f[i].fill_buff2_in, 0, 0, 0, 0 );
                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].updates_f[i].fill_buff2_out, 0, 0, 0, 0 );
            }
        }

        
        
        
        kernel_varout(m->NDIM, m->nvars, (*vcl)[d].vars, &(*vcl)[d].src_recs.seisout.src);
        __GUARD assign_prog_source(&(*vcl)[d].src_recs.seisout, "seisout", (*vcl)[d].src_recs.seisout.src);
        __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].src_recs.seisout, 0, 0, 0, 0);
        kernel_varoutinit(m->NDIM, m->nvars, (*vcl)[d].vars, &(*vcl)[d].src_recs.seisoutinit.src);
        __GUARD assign_prog_source(&(*vcl)[d].src_recs.seisoutinit, "seisoutinit", (*vcl)[d].src_recs.seisoutinit.src);
        __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].src_recs.seisoutinit, 0, 0, 0, 0);
        
         kernel_varinit(m->NDIM, m->nvars, (*vcl)[d].vars, &(*vcl)[d].bnd_cnds.init_f.src);
        __GUARD assign_prog_source(&(*vcl)[d].bnd_cnds.init_f, "vars_init", (*vcl)[d].bnd_cnds.init_f.src);
        __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].bnd_cnds.init_f, 0, 0, 0, 0);
        
        
        if (m->GRADOUT){
            kernel_residuals(m->NDIM, m->nvars, (*vcl)[d].vars, &(*vcl)[d].src_recs.residuals.src);
            __GUARD assign_prog_source(&(*vcl)[d].src_recs.residuals, "residuals", (*vcl)[d].src_recs.residuals.src);
            __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].src_recs.residuals, 0, 0, 0, 0);
            
            if (m->BACK_PROP_TYPE==1){
                kernel_varinit(m->NDIM, m->nvars, (*vcl)[d].vars_adj, &(*vcl)[d].bnd_cnds.init_adj.src);
                __GUARD assign_prog_source(&(*vcl)[d].bnd_cnds.init_adj, "vars_init", (*vcl)[d].bnd_cnds.init_adj.src);
                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].bnd_cnds.init_adj, 0, 0, 0, 0);
                
                kernel_gradinit(m->NDIM, m->nparams, (*vcl)[d].params, &(*vcl)[d].grads.init.src);
                __GUARD assign_prog_source(&(*vcl)[d].grads.init, "gradinit", (*vcl)[d].grads.init.src);
                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].grads.init, 0, 0, 0, 0);
            }
            else if(m->BACK_PROP_TYPE==2){
                kernel_initsavefreqs(m->NDIM, m->nvars, (*vcl)[d].vars, &(*vcl)[d].grads.initsavefreqs.src);
                __GUARD assign_prog_source(&(*vcl)[d].grads.initsavefreqs, "initsavefreqs", (*vcl)[d].grads.initsavefreqs.src);
                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].grads.initsavefreqs, 0, 0, 0, 0);
                
                kernel_savefreqs(m->NDIM, m->nvars, (*vcl)[d].vars, &(*vcl)[d].grads.savefreqs.src);
                __GUARD assign_prog_source(&(*vcl)[d].grads.savefreqs, "savefreqs", (*vcl)[d].grads.savefreqs.src);
                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].grads.savefreqs, 0, 0, 0, 0);
            }
            
            if (m->GRADSRCOUT){
                kernel_init_gradsrc(m->NDIM, &(*vcl)[d].src_recs.init_gradsrc.src);
                __GUARD assign_prog_source(&(*vcl)[d].src_recs.init_gradsrc, "init_gradsrc", (*vcl)[d].src_recs.init_gradsrc.src);
                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].src_recs.init_gradsrc, 0, 0, 0, 0);
            }
        }
        

        kernel_fillbuff( m->NDIM, m->N_names, (*vcl)[d].LOCAL_OFF, m->nvars ,(*vcl)[d].vars,&(*vcl)[d].updates_f[0].fill_buff1_in.src, 1, 0, 1);
        
        //Define other kernels
        //TODO Most of these kernel could be automatically generated
        //    __GUARD assign_prog_source(&m->updates_f[0].fill_buff1_in, "fill_transfer_buff_v_in", fill_transfer_buff_v_source);
        //    __GUARD assign_prog_source(&m->updates_f[0].fill_buff1_out, "fill_transfer_buff_v_out", fill_transfer_buff_v_source);
        //    __GUARD assign_prog_source(&m->updates_f[0].fill_buff2_in, "fill_transfer_buff_v_in", fill_transfer_buff_v_source);
        //    __GUARD assign_prog_source(&m->updates_f[0].fill_buff2_out, "fill_transfer_buff_v_out", fill_transfer_buff_v_source);
        //    __GUARD assign_prog_source(&m->updates_f[1].fill_buff1_in, "fill_transfer_buff_s_in", fill_transfer_buff_s_source);
        //    __GUARD assign_prog_source(&m->updates_f[1].fill_buff1_out, "fill_transfer_buff_s_out", fill_transfer_buff_s_source);
        //    __GUARD assign_prog_source(&m->updates_f[1].fill_buff2_in, "fill_transfer_buff_s_in", fill_transfer_buff_s_source);
        //    __GUARD assign_prog_source(&m->updates_f[1].fill_buff2_out, "fill_transfer_buff_s_out", fill_transfer_buff_s_source);
        //

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
//        __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].bnd_cnds.init_f, 0, 0, 0, 0);
//        
//      
//        
//        if (m->GRADOUT){
//            for (i=0;i<m->nupdates;i++){
//                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].updates_adj[i].center, offcomm1, LCOMM, 0, 0);
//                if (d>0 || m->MYLOCALID>0){
//                    __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].updates_adj[i].comm1, 0, LCOMM, 1, 0 );
//                }
//                if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
//                    __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].updates_adj[i].comm1, offcomm2, LCOMM, 1, 0 );
//                }
//            }
//            
//            __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].src_recs.residuals, 0, 0, 0, 0);
//            
//            if (m->FREESURF){
//                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].bnd_cnds.surf, 0, 0, 0, 0);
//            }
//
//            if (m->GRADSRCOUT){
//                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].src_recs.init_gradsrc, 0, 0, 0, 0);
//            }
//            if (m->BACK_PROP_TYPE==1){
//                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].grads.savebnd, 0, 0, 0, 0);
//                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].bnd_cnds.init_adj, 0, 0, 0, 0);
//                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].grads.init, 0, 0, 0, 0);
//            }
//            else if(m->BACK_PROP_TYPE==2){
//                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].grads.savefreqs, 0, 0, 0, 0);
//                __GUARD gpu_initialize_kernel(m, &(*vcl)[d],  &(*vcl)[d].grads.initsavefreqs, 0, 0, 0, 0);
//            }
//
//        }
        
        



//        if (d>0 || m->MYLOCALID>0 || d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
//            __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_fill_transfer_buff1_v_in, &(*vcl)[d], m, &(*mloc)[d],0,1,0);
//            __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_fill_transfer_buff2_v_in, &(*vcl)[d], m, &(*mloc)[d],0,2,0);
//            __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_fill_transfer_buff1_v_out, &(*vcl)[d], m, &(*mloc)[d],1,1,0);
//            __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_fill_transfer_buff2_v_out, &(*vcl)[d], m, &(*mloc)[d],1,2,0);
//            __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_fill_transfer_buff1_s_in, &(*vcl)[d], m, &(*mloc)[d],0,1,0);
//            __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_fill_transfer_buff2_s_in, &(*vcl)[d], m, &(*mloc)[d],0,2,0);
//            __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_fill_transfer_buff1_s_out, &(*vcl)[d], m, &(*mloc)[d],1,1,0);
//            __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_fill_transfer_buff2_s_out, &(*vcl)[d], m, &(*mloc)[d],1,2,0);
//        }

        if (state !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(state));
//
//
//        //If we want the gradient by the adjoint model method, we create the variables
//        if (m->GRADOUT==1 ){
//
//            //Create the kernels for the backpropagation and gradient computation
//            __GUARD gpu_initialize_update_adjv(&m->context, &(*vcl)[d].program_adjv, &(*vcl)[d].kernel_adjv, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], offcomm1, LCOMM, 0  );
//            __GUARD gpu_initialize_update_adjs(&m->context, &(*vcl)[d].program_adjs, &(*vcl)[d].kernel_adjs, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], offcomm1, LCOMM, 0 );
//            
//            if (d>0 || m->MYLOCALID>0){
//                __GUARD gpu_initialize_update_adjv(&m->context, &(*vcl)[d].program_adjvcomm1, &(*vcl)[d].kernel_adjvcomm1, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], 0 , LCOMM, 1);
//                __GUARD gpu_initialize_update_adjs(&m->context, &(*vcl)[d].program_adjscomm1, &(*vcl)[d].kernel_adjscomm1, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], 0 , LCOMM, 1);
//            }
//            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1 ){
//                __GUARD gpu_initialize_update_adjv(&m->context, &(*vcl)[d].program_adjvcomm2, &(*vcl)[d].kernel_adjvcomm2, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], offcomm2, LCOMM, 1 );
//                __GUARD gpu_initialize_update_adjs(&m->context, &(*vcl)[d].program_adjscomm2, &(*vcl)[d].kernel_adjscomm2, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], offcomm2, LCOMM, 1 );
//            }
//            if ( (d>0 || m->MYLOCALID>0 || d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1) && m->BACK_PROP_TYPE==1 ){
//                __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_adj_fill_transfer_buff1_v_in, &(*vcl)[d], m, &(*mloc)[d],0,1,1);
//                __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_adj_fill_transfer_buff2_v_in, &(*vcl)[d], m, &(*mloc)[d],0,1,1);
//                __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_adj_fill_transfer_buff1_v_out, &(*vcl)[d], m, &(*mloc)[d],1,1,1);
//                __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_adj_fill_transfer_buff2_v_out, &(*vcl)[d], m, &(*mloc)[d],1,1,1);
//                __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_adj_fill_transfer_buff1_s_in, &(*vcl)[d], m, &(*mloc)[d],0,1,1);
//                __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_adj_fill_transfer_buff2_s_in, &(*vcl)[d], m, &(*mloc)[d],0,1,1);
//                __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_adj_fill_transfer_buff1_s_out, &(*vcl)[d], m, &(*mloc)[d],1,1,1);
//                __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_adj_fill_transfer_buff2_s_out, &(*vcl)[d], m, &(*mloc)[d],1,1,1);
//            }
//
//            __GUARD gpu_intialize_residuals(&m->context, &(*vcl)[d].program_residuals, &(*vcl)[d].kernel_residuals, NULL, &(*vcl)[d], m, &(*mloc)[d]);
//            __GUARD gpu_intialize_grad(&m->context, &(*vcl)[d].program_initgrad, &(*vcl)[d].kernel_initgrad, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d]);
//            __GUARD gpu_intialize_seis_r(&m->context, &(*vcl)[d].program_initseis_r, &(*vcl)[d].kernel_initseis_r, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d]);
//
//            
//            if (m->BACK_PROP_TYPE==1){
//                __GUARD gpu_initialize_savebnd(&m->context, &(*vcl)[d].program_bnd, &(*vcl)[d].kernel_bnd, NULL, &(*vcl)[d], m, &(*mloc)[d]);
//            }
//            if (m->BACK_PROP_TYPE==2){
//                __GUARD gpu_initialize_savefreqs(&m->context, &(*vcl)[d].program_savefreqs, &(*vcl)[d].kernel_savefreqs, NULL, &(*vcl)[d], m, &(*mloc)[d], 0);
//                __GUARD gpu_initialize_initsavefreqs(&m->context, &(*vcl)[d].program_initsavefreqs, &(*vcl)[d].kernel_initsavefreqs, NULL, &(*vcl)[d], m, &(*mloc)[d]);
//            }
//            if (m->GRADSRCOUT==1){
//                __GUARD gpu_initialize_gradsrc(&m->context, &(*vcl)[d].program_initialize_gradsrc, &(*vcl)[d].kernel_initialize_gradsrc, NULL, &(*vcl)[d], m, &(*mloc)[d]);
//                
//            }
//            if (state !=CL_SUCCESS) fprintf(stderr,"%s\n",gpu_error_code(state));
//        }

        
    }

    if (state && m->MPI_INIT==1) MPI_Bcast( &state, 1, MPI_INT, m->MYID, MPI_COMM_WORLD );
    
    return state;


}
