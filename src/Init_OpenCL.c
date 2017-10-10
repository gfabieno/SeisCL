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
//
//int get_platform( model * m, cl_platform_id*  clplateform)
//{
//    /* Find all platforms , and select the first with the desired device type */
//    
//    char chBuffer[1024];
//    cl_uint num_platforms=0;
//    cl_platform_id* clPlatformIDs=NULL;
//    int state=0;
//    int i,j,k;
//    cl_int device_found;
//    
//    //Check that a suitable device type was given
//    if (m->pref_device_type!=CL_DEVICE_TYPE_GPU &&
//        m->pref_device_type!=CL_DEVICE_TYPE_CPU &&
//        m->pref_device_type!=CL_DEVICE_TYPE_ACCELERATOR ){
//        fprintf(stdout," Warning: invalid prefered device type,"
//                       " defaulting to GPU\n");
//        m->pref_device_type=CL_DEVICE_TYPE_GPU;
//    }
//    
//    // Get OpenCL platform count
//    __GUARD clGetPlatformIDs (0, NULL, &num_platforms);
//    if(num_platforms == 0){
//        fprintf(stderr,"No OpenCL platform found!\n\n");
//        return 1;
//    }
//    
//
//    fprintf(stdout,"Found %u OpenCL platforms:\n", num_platforms);
//    // Allocate the number of platforms
//    GMALLOC(clPlatformIDs,num_platforms * sizeof(cl_platform_id));
//    
//    // Get platform info for each platform and the platform containing the
//    // prefered device type if found
//    __GUARD clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
//
//    // Try to find a platform with the prefered device type first
//    for(i = 0; i < num_platforms; ++i){
//        device_found = clGetDeviceIDs(clPlatformIDs[i],
//                                      m->pref_device_type,
//                                      0,
//                                      NULL,
//                                      &m->NUM_DEVICES);
//        
//        if(device_found==0 && m->NUM_DEVICES>0){
//            //Check if any GPU is flagged not to be used
//            for (j=0;j<m->NUM_DEVICES;j++){
//                for (k=0;k<m->n_no_use_GPUs;k++){
//                    if (m->no_use_GPUs[k]==j)
//                        m->NUM_DEVICES-=1;
//                }
//            }
//            
//            if (m->NUM_DEVICES<1){
//                fprintf(stdout,"Warning: No allowed devices could be found");
//                return 1;
//            }
//            else{
//                *clplateform = clPlatformIDs[i];
//                m->device_type=m->pref_device_type;
//                break;
//            }
//        }
//    }
//    // Then if not found, fall back on a platform with GPUs
//    if(*clplateform == NULL){
//        for(i = 0; i < num_platforms; ++i){
//            device_found = clGetDeviceIDs(clPlatformIDs[i],
//                                          CL_DEVICE_TYPE_GPU,
//                                          0,
//                                          NULL,
//                                          &m->NUM_DEVICES);
//            if(device_found == 0 && m->NUM_DEVICES>0){
//                *clplateform = clPlatformIDs[i];
//                m->device_type=CL_DEVICE_TYPE_GPU;
//                break;
//            }
//        }
//    }
//    // Then if not found, fall back on a platform with accelerators
//    if(*clplateform == NULL){
//        for(i = 0; i < num_platforms; ++i){
//            device_found = clGetDeviceIDs(clPlatformIDs[i],
//                                          CL_DEVICE_TYPE_ACCELERATOR,
//                                          0,
//                                          NULL,
//                                          &m->NUM_DEVICES);
//            if(device_found == 0 && m->NUM_DEVICES>0){
//                *clplateform = clPlatformIDs[i];
//                m->device_type=CL_DEVICE_TYPE_ACCELERATOR;
//                break;
//            }
//        }
//    }
//    // Finally, fall back on a platform with CPUs
//    if(*clplateform == NULL){
//        for(i = 0; i < num_platforms; ++i){
//            device_found = clGetDeviceIDs(clPlatformIDs[i],
//                                          CL_DEVICE_TYPE_CPU,
//                                          0,
//                                          NULL,
//                                          &m->NUM_DEVICES);
//            if(device_found == 0 && m->NUM_DEVICES>0){
//                *clplateform = clPlatformIDs[i];
//                m->device_type=CL_DEVICE_TYPE_CPU;
//                break;
//            }
//        }
//    }
//    if(*clplateform == NULL)
//    {
//        fprintf(stderr,"Error: No usable platforms could be found\n\n");
//        state=1;
//    }
//    else{
//        __GUARD clGetPlatformInfo (*clplateform,
//                                   CL_PLATFORM_NAME,
//                                   1024,
//                                   &chBuffer,
//                                   NULL);
//        fprintf(stdout,"Connection to platform %d: %s\n", i, chBuffer);
//        
//    }
//    
//    //Verify that we do not use more than the allowed number of devices
//    m->NUM_DEVICES=m->NUM_DEVICES>m->nmax_dev ? m->nmax_dev : m->NUM_DEVICES;
//    
//    GFree(clPlatformIDs);
//    if (state !=CUDA_SUCCESS) fprintf(stderr,"%s\n",clerrors(state));
//    
//    return state;
//}
//
int connect_devices(device ** dev, model * m)
{
    /*Routine to connect all computing devices, create context and queues*/
    
    int state = 0;
    int nalldevices=0;
    int *allow_devs=NULL;
    char vendor_name[1024] = {0};
    char device_name[1024] = {0};
    int i,j,n;
    int allowed;

    if (m->nmax_dev<1){
        fprintf(stdout,"Warning: maximum number of devices too small,"
                       "default to 1\n");
        m->nmax_dev=1;
    }
    
    __GUARD cuInit(0);
    __GUARD cuDeviceGetCount ( &nalldevices );

    
    
    
    // Find the number of prefered devices
    GMALLOC(allow_devs,sizeof(int)*nalldevices);
    //Collect all allowed devices
    n=0;
    if (!state){
        for (i=0;i<nalldevices;i++){
            allowed=1;
            for (j=0;j<m->n_no_use_GPUs;j++){
                if (m->no_use_GPUs[j]!=i){
                    allowed=0;
                }
            }
            if (allowed){
                allow_devs[n]=i;
                n++;
            }
            
        }
    }
    m->NUM_DEVICES = n;
    GMALLOC(*dev, sizeof(device)*m->NUM_DEVICES);
    
    
    // Print some information about the returned devices
    fprintf(stdout,"Connecting to %d devices: \n",m->NUM_DEVICES);
    // Create command queues for each devices
    for (i=0;i<m->NUM_DEVICES;i++){
        // Create a context with the specified devices
        __GUARD cuDeviceGet(&(*dev)[i].cudev, allow_devs[i]);
        __GUARD cuCtxCreate(&(*dev)[i].context,
                            0,
                            (*dev)[i].cudev);
        __GUARD cuDeviceGetName(device_name,
                                sizeof(vendor_name),
                                (*dev)[i].cudev);
        fprintf(stdout,"-Device %d: %s \n",allow_devs[i], device_name);
        __GUARD cuStreamCreate( &(*dev)[i].queue, CU_STREAM_NON_BLOCKING );
        __GUARD cuStreamCreate( &(*dev)[i].queuecomm, CU_STREAM_NON_BLOCKING );
        
    }

    if (state !=CUDA_SUCCESS) fprintf(stderr,
                                      "Devices connection failed: %s\n",
                                      clerrors(state));
    GFree(allow_devs);
    
    return state;
    
}


int Init_CUDA(model * m, device ** dev)  {
    /* Function that intialize model decomposition on multiple devices on one
       host. All OpenCL buffers and kernels are created and are contained in a
       device structure (one for each devices)
     */
    int state=0;
    int i,j,d;
    int parsize=0;
    int fdsize=0;
    int slicesize=0;
    int slicesizefd=0;
    int local_mem_size=0;
    int required_local_mem_size=0;
    int required_work_size=0;
    int workgroup_size=0;
    int workdim = 0;
    int lsize[MAX_DIMS];
    int gsize[MAX_DIMS];
    int gsize_com1[MAX_DIMS];
    int gsize_com2[MAX_DIMS];
    int gsize_fcom[MAX_DIMS];
    int LCOMM=0;
    int offcom1=0;
    int offcom2=0;
    struct device * di;
    
    // Connect all OpenCL devices that can be used in a single context
    __GUARD connect_devices(dev, m);
    fprintf(stdout,"connected\n");

    
    //For each device, create the memory buffers and programs on the GPU
    for (d=0; d<m->NUM_DEVICES; d++) {
        
        di=&(*dev)[d];
        __GUARD cuCtxSetCurrent ( di->context );
        /* The domain of the FD simulation is decomposed between the devices and
        *  MPI processes along the X direction. We must compute the subdomain 
        *  size, and its location in the domain*/
        if (!state){
            di->DEVID=d;
            di->NDIM=m->NDIM;
            //Only the last dimensions changes due to domain decomposition
            for (i=0;i<m->NDIM-1;i++){
                di->N[i]=m->N[i];
            }
            for (i=0;i<m->NDIM;i++){
                di->N_names[i]=m->N_names[i];
            }
            //We first decompose between MPI processess
            if (m->MYLOCALID<m->N[m->NDIM-1]%m->NLOCALP){
                di->N[m->NDIM-1]=m->N[m->NDIM-1]/m->NLOCALP+1;
                m->NXP=m->N[m->NDIM-1]/m->NLOCALP+1;
            }
            else{
                di->N[m->NDIM-1]=m->N[m->NDIM-1]/m->NLOCALP;
                m->NXP=m->N[m->NDIM-1]/m->NLOCALP;
            }
            //Then decompose between devices
            if (d<di->N[m->NDIM-1]%m->NUM_DEVICES)
                di->N[m->NDIM-1]= di->N[m->NDIM-1]/m->NUM_DEVICES+1;
            else
                di->N[m->NDIM-1]= di->N[m->NDIM-1]/m->NUM_DEVICES;

            di->OFFSET=0;
            di->NX0=0;
            di->OFFSETfd=0;

            //Some useful sizes can now be computed for this device
            parsize=1;
            fdsize=1;
            slicesize=1;
            slicesizefd=1;
            for (i=0;i<m->NDIM;i++){
                fdsize*=di->N[i]+m->FDORDER;
                parsize*=di->N[i];
            }
            for (i=0;i<m->NDIM-1;i++){
                slicesizefd*=di->N[i]+m->FDORDER;
                slicesize*=di->N[i];
            }
            
            //Offset is the location in the model for this device from address 0
            // NX0 is the x location of the first element for this device
            for (i=0;i<m->MYLOCALID;i++){
                if (i<m->N[m->NDIM-1]%m->NLOCALP){
                    di->OFFSET+=(m->N[m->NDIM-1]/m->NLOCALP+1)*slicesize;
                    di->NX0+=(m->N[m->NDIM-1]/m->NLOCALP+1);
                }
                else{
                    di->OFFSET+=(m->N[m->NDIM-1]/m->NLOCALP)*slicesize;
                    di->NX0+=(m->N[m->NDIM-1]/m->NLOCALP);
                }
                
            }
            for (i=0;i<d;i++){
                di->NX0+=(*dev)[i].N[m->NDIM-1];
                di->OFFSET+=(*dev)[i].N[m->NDIM-1]*slicesize;
                di->OFFSETfd+=((*dev)[i].N[m->NDIM-1]+m->FDORDER)*slicesizefd;
            }
            
        }

        // Get some properties of the device, to define the work sizes
        {
            __GUARD  cuDeviceGetAttribute(&workgroup_size,
                                  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                  di->cudev );
            __GUARD  cuDeviceGetAttribute(&local_mem_size,
                                          CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                                          di->cudev );

            if (state !=CUDA_SUCCESS) fprintf(stderr,"%s\n",clerrors(state));
            
        }
        fprintf(stdout,"properties\n");
        // Define the local work size of the update kernels.
        //By default, it is 32 elements long to have coalesced memory in cuda
        //Local memory usage must fit the size of local memory of the device
        if (!state){


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
//                }
                
                //Check if too many GPUS are used in the domain decomposition
                if  (di->N[m->NDIM-1]<3*lsize[m->NDIM-1]){
                    di->LOCAL_OFF = 1;
                    fprintf(stderr,"Too many GPUs for domain decompositon,"
                                   "Switching off local memory optimization\n");
                }
                //Check if the local memory is big enough, else turn it off
                if (required_local_mem_size>local_mem_size){
                    fprintf(stderr,"Local memory needed to perform seismic "
                                   "modeling (%d bits) exceeds the local "
                                   "memory capacity of device %d (%d bits)\n"
                                   "Switching off local memory optimization\n",
                                   required_local_mem_size, d, local_mem_size );
                    di->LOCAL_OFF = 1;
                }
                
            }

            if (di->LOCAL_OFF==1)
                lsize[0] = 1;
        }
        
        // Define the global work size of the update kernels.
        // To overlap computations and communications, we have 3 kernels:
        // One for the interior (no communications) and 2 for the front and back
        // We define here de sizes of those region (offcom1, offcom2)
        if (!state){
            if (di->LOCAL_OFF==1){
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
                    offcom2=di->N[m->NDIM-1]-m->FDOH;
                }
                if (d>0 || m->MYLOCALID>0
                        || d<m->NUM_DEVICES-1
                        || m->MYLOCALID<m->NLOCALP-1){
                    gsize_fcom[0] = slicesize*m->FDOH;
                }
                
                gsize[0] = (di->N[m->NDIM-1]-LCOMM)*slicesize;
                
                workdim=1;
                
            }
            else{

                // When using local work sizes in OpenCL,
                // global work size must be a multiple of local work size
                for (i=0;i<m->NDIM-1;i++){
                    gsize[i]=di->N[i]
                            +(lsize[i]-di->N[i]%lsize[i])%lsize[i];
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
                    offcom2=di->N[m->NDIM-1]-(int)lsize[m->NDIM-1];
                }
                if (d>0 || m->MYLOCALID>0
                        || d<m->NUM_DEVICES-1
                        || m->MYLOCALID<m->NLOCALP-1){
                    
                    for (i=0;i<m->NDIM-1;i++){
                        gsize_fcom[i] = di->N[i];
                    }
                    gsize_fcom[m->NDIM-1] = m->FDOH;
                    
                }

                gsize[m->NDIM-1] = di->N[m->NDIM-1]-LCOMM
                               +(lsize[m->NDIM-1]-(di->N[m->NDIM-1]-LCOMM)
                                  %lsize[m->NDIM-1])%lsize[m->NDIM-1];
                
                workdim=m->NDIM;

            }
        }
        
        //Create the required updates struct and assign the working size
        {
            //Struct for the forward modeling
            GMALLOC(di->ups_f, m->nupdates*sizeof(update));
            for (i=0;i<m->nupdates;i++){
                di->ups_f[i]=m->ups_f[i];
                GMALLOC(di->ups_f[i].v2com,m->nvars*sizeof(variable*));
            }
            
            for (i=0;i<m->nupdates;i++){
                for (j=0;j<workdim;j++){
                    di->ups_f[i].center.wdim=workdim;
                    di->ups_f[i].center.gsize[j]=gsize[j];
                    di->ups_f[i].center.lsize[j]=lsize[j];
                    di->ups_f[i].com1.wdim=workdim;
                    di->ups_f[i].com1.gsize[j]=gsize_com1[j];
                    di->ups_f[i].com1.lsize[j]=lsize[j];
                    di->ups_f[i].com2.wdim=workdim;
                    di->ups_f[i].com2.gsize[j]=gsize_com2[j];
                    di->ups_f[i].com2.lsize[j]=lsize[j];
                    di->ups_f[i].fcom1_in.wdim=workdim;
                    di->ups_f[i].fcom1_in.gsize[j]=gsize_fcom[j];
                    di->ups_f[i].fcom2_in.wdim=workdim;
                    di->ups_f[i].fcom2_in.gsize[j]=gsize_fcom[j];
                    di->ups_f[i].fcom1_out.wdim=workdim;
                    di->ups_f[i].fcom1_out.gsize[j]=gsize_fcom[j];
                    di->ups_f[i].fcom2_out.wdim=workdim;
                    di->ups_f[i].fcom2_out.gsize[j]=gsize_fcom[j];
                }
            }
            //Struct for the adjoint modeling
            if (m->GRADOUT){
                GMALLOC(di->ups_adj, m->nupdates*sizeof(update));
                for (i=0;i<m->nupdates;i++){
                    di->ups_adj[i]=m->ups_adj[i];
                    GMALLOC(di->ups_adj[i].v2com,m->nvars*sizeof(variable*));
                }
                for (i=0;i<m->nupdates;i++){
                    for (j=0;j<workdim;j++){
                        di->ups_adj[i].center.wdim=workdim;
                        di->ups_adj[i].center.gsize[j]=gsize[j];
                        di->ups_adj[i].center.lsize[j]=lsize[j];
                        di->ups_adj[i].com1.wdim=workdim;
                        di->ups_adj[i].com1.gsize[j]=gsize_com1[j];
                        di->ups_adj[i].com1.lsize[j]=lsize[j];
                        di->ups_adj[i].com2.wdim=workdim;
                        di->ups_adj[i].com2.gsize[j]=gsize_com2[j];
                        di->ups_adj[i].com2.lsize[j]=lsize[j];
                        di->ups_adj[i].fcom1_in.wdim=workdim;
                        di->ups_adj[i].fcom1_in.gsize[j]=gsize_fcom[j];
                        di->ups_adj[i].fcom2_in.wdim=workdim;
                        di->ups_adj[i].fcom2_in.gsize[j]=gsize_fcom[j];
                        di->ups_adj[i].fcom1_out.wdim=workdim;
                        di->ups_adj[i].fcom1_out.gsize[j]=gsize_fcom[j];
                        di->ups_adj[i].fcom2_out.wdim=workdim;
                        di->ups_adj[i].fcom2_out.gsize[j]=gsize_fcom[j];
                    }
                }
            }
        }

        // Create parameter structure for this device, its buffers and transfer
        {
            di->npars=m->npars;
            GMALLOC(di->pars, sizeof(parameter)*m->npars);

            // Create a pointer at the right position (OFFSET) of the decomposed
            // model and assign memory buffers on the host side for each device
            if (!state){
                for (i=0;i<m->npars;i++){
                    di->pars[i]=m->pars[i];
                    di->pars[i].cl_par.host=&m->pars[i].gl_par[di->OFFSET];
                    di->pars[i].num_ele=parsize;
                    if (m->pars[i].to_grad){
                        di->pars[i].cl_grad.host=&m->pars[i].gl_grad[di->OFFSET];
                        if (m->pars[i].gl_H)
                            di->pars[i].cl_H.host=&m->pars[i].gl_H[di->OFFSET];
                    }
                }
                
            }
            
            // Create buffers for all parameters and their gradient,
            // and transfer to device parameters
            if (!state){
                for (i=0;i<m->npars;i++){
                    di->pars[i].num_ele=parsize;
                    di->pars[i].cl_par.size=sizeof(float)*parsize;
                    __GUARD clbuf_create(&di->pars[i].cl_par);
                    __GUARD clbuf_send(&di->queue,&di->pars[i].cl_par);
                    if (m->GRADOUT && m->BACK_PROP_TYPE==1){
                        di->pars[i].cl_grad.size=sizeof(float)*parsize;
                        __GUARD clbuf_create(&di->pars[i].cl_grad);
                    }
                    if (m->HOUT && m->BACK_PROP_TYPE==1){
                        di->pars[i].cl_H.size=sizeof(float)*parsize;
                        __GUARD clbuf_create( &di->pars[i].cl_H);
                    }
                    
                }
            }
            
        }
        
        //Set the sources and receivers structure for this device
        {
            di->src_recs=m->src_recs;
            di->src_recs.cl_src_pos.size=sizeof(float) * 5 * m->src_recs.nsmax;
            di->src_recs.cl_rec_pos.size=sizeof(float) * 8 * m->src_recs.ngmax;
            di->src_recs.cl_src.size=sizeof(float) * m->NT * m->src_recs.nsmax;
            __GUARD clbuf_create( &di->src_recs.cl_src_pos);
            __GUARD clbuf_create( &di->src_recs.cl_rec_pos);
            __GUARD clbuf_create( &di->src_recs.cl_src);
            if (m->GRADSRCOUT){
                di->src_recs.cl_grad_src.size=sizeof(float)
                * m->NT * m->src_recs.nsmax;
                __GUARD clbuf_create(&di->src_recs.cl_grad_src);
            }
        }
        
        //Allocate the variables structure for this device and create buffers
        di->nvars=m->nvars;
        di->ntvars=m->ntvars;
        GMALLOC(di->vars, sizeof(variable)*m->nvars);
        if (di->ntvars){
            GMALLOC(di->trans_vars, sizeof(variable)*m->nvars);
        }
        for (i=0;i<m->nvars;i++){
            di->vars[i]=m->vars[i];
            //Set the size of the local variables
            di->vars[i].set_size(di->N, (void*) m, &di->vars[i]);
        }
        for (i=0;i<m->ntvars;i++){
            di->trans_vars[i]=m->trans_vars[i];
        }

        //Create OpenCL buffers with the right size
        for (i=0;i<m->nvars;i++){
            
            //Add the variable to variables to communicate during update
            if (di->vars[i].to_comm>0){
                j=di->vars[i].to_comm-1;
                di->ups_f[j].v2com[di->ups_f[j].nvcom]=&di->vars[i];
                di->ups_f[j].nvcom++;
            }
            
            //Create variable buffers for the interior domain
            di->vars[i].cl_var.size=sizeof(float)*di->vars[i].num_ele;
            __GUARD clbuf_create( &di->vars[i].cl_var);
            
            //Create variable buffers for the boundary of the domain
            if ( di->vars[i].to_comm
                && (d>0 || m->MYLOCALID>0
                        || d<m->NUM_DEVICES-1
                        || m->MYLOCALID<m->NLOCALP-1)){
                    
                    //On the device side
                    di->vars[i].cl_buf1.size=sizeof(float)*m->FDOH*slicesize;
                    __GUARD clbuf_create_pin(&di->vars[i].cl_buf1);
                    di->vars[i].cl_buf2.size=sizeof(float)*m->FDOH*slicesize;
                    __GUARD clbuf_create_pin(&di->vars[i].cl_buf2);
                }
            
            // Create the buffers to output variables at receivers locations
            if (di->vars[i].to_output){
                
                //Memory for recordings for this device on host side
                di->vars[i].cl_varout.size=sizeof(float)
                                          * m->NT * m->src_recs.ngmax;
                __GUARD clbuf_create(&di->vars[i].cl_varout);
                GMALLOC(di->vars[i].cl_varout.host, di->vars[i].cl_varout.size);
                di->vars[i].cl_varout.free_host=1;
                
                //Create also a buffer for the residuals
                if (m->vars[i].gl_var_res){
                    di->vars[i].cl_var_res.size=sizeof(float)
                                               * m->NT * m->src_recs.ngmax;
                    __GUARD clbuf_create( &di->vars[i].cl_var_res);
                }
            }
            
            // If we use the DFT for gradient computation,
            // we create the buffers to hold each frequency
            if (m->GRADOUT
                && m->BACK_PROP_TYPE==2
                && di->vars[i].for_grad){
                

                di->vars[i].cl_fvar.size=2*sizeof(float)
                                        * di->vars[i].num_ele * m->NFREQS;
                __GUARD clbuf_create(&di->vars[i].cl_fvar);
                GMALLOC(di->vars[i].cl_fvar.host,di->vars[i].cl_fvar.size);
                di->vars[i].cl_fvar.free_host=1;
                di->vars[i].cl_fvar_adj.size= di->vars[i].cl_fvar.size;
                GMALLOC(di->vars[i].cl_fvar_adj.host,di->vars[i].cl_fvar.size);
                di->vars[i].cl_fvar_adj.free_host=1;
            }
            
            // If we want the movie, allocate memory for variables
            if (m->MOVOUT){
                if (m->vars[i].to_output){
                    di->vars[i].gl_mov=&m->vars[i].gl_mov[di->OFFSET];
                    GMALLOC(di->vars[i].cl_var.host,
                            di->vars[i].num_ele*sizeof(float));
                    di->vars[i].cl_var.free_host=1;
                }
            }
            
        }
        fprintf(stdout,"variable\n");
        //Create OpenCL buffers for transformed varibales
        for (i=0;i<m->ntvars;i++){
            // Create the buffers to output variables at receivers locations
            if (di->trans_vars[i].to_output){
                
                //Memory for recordings for this device on host side
                di->trans_vars[i].cl_varout.size=sizeof(float)
                * m->NT * m->src_recs.ngmax;
                __GUARD clbuf_create( &di->trans_vars[i].cl_varout);
                GMALLOC(di->trans_vars[i].cl_varout.host, di->trans_vars[i].cl_varout.size);
                di->trans_vars[i].cl_varout.free_host=1;
                
                //Create also a buffer for the residuals
                if (m->trans_vars[i].gl_var_res){
                    di->trans_vars[i].cl_var_res.size=sizeof(float)
                    * m->NT * m->src_recs.ngmax;
                    __GUARD clbuf_create(  &di->trans_vars[i].cl_var_res);
                }

            }
        }
        fprintf(stdout,"transformed\n");
        //Create constants structure and buffers, transfer to device
        di->ncsts=m->ncsts;
        GMALLOC(di->csts, sizeof(constants)*m->ncsts);
        for (i=0;i<m->ncsts;i++){
            di->csts[i]=m->csts[i];
            //Size of constants does not depend of domain decomposition
            di->csts[i].cl_cst.size=sizeof(float)*di->csts[i].num_ele;
            di->csts[i].cl_cst.host=m->csts[i].gl_cst;
            __GUARD clbuf_create(  &di->csts[i].cl_cst);
            __GUARD clbuf_send( &di->queue, &di->csts[i].cl_cst);
            
        }
        fprintf(stdout,"csts\n");
        
        // Determine the size of the outside boundary used for the back
        // propagation of the seismic wavefield
        // TODO: this is ugly and model specific, find a better way
        if (m->BACK_PROP_TYPE==1 && m->GRADOUT){
            
            GMALLOC(di->vars_adj, sizeof(variable)*m->nvars);
            for (i=0;i<m->nvars;i++){
                di->vars_adj[i]=di->vars[i];
            }
            
            for (i=0;i<m->nvars;i++){
                //Add the variable to variables to communicate during update
                if (di->vars_adj[i].to_comm>0){
                    j=di->vars_adj[i].to_comm-1;
                    di->ups_adj[j].v2com[di->ups_adj[j].nvcom]=&di->vars_adj[i];
                    di->ups_adj[j].nvcom++;
                }
                __GUARD clbuf_create(  &di->vars_adj[i].cl_var);
                if (di->vars[i].to_comm && (d>0
                                            || m->MYLOCALID>0
                                            || d<m->NUM_DEVICES-1
                                            || m->MYLOCALID<m->NLOCALP-1)){
                    
                    __GUARD clbuf_create_pin(&di->vars_adj[i].cl_buf1);
                    __GUARD clbuf_create_pin(&di->vars_adj[i].cl_buf2);
                }
            }
            
            if (m->ND==3){// For 3D
                if (m->NUM_DEVICES==1 && m->NLOCALP==1)
                    di->NBND=(di->N[2]-2*m->NAB)*(di->N[1]-2*m->NAB)*2*m->FDOH
                    +(di->N[2]-2*m->NAB)*(di->N[0]-2*m->NAB-2*m->FDOH)*2*m->FDOH
                    +(di->N[1]-2*m->NAB-2*m->FDOH)*(di->N[0]-2*m->NAB-2*m->FDOH)
                    *2*m->FDOH;
                
                else if ( (d==0 && m->MYGROUPID==0)
                         || (d==m->NUM_DEVICES-1 && m->MYGROUPID==m->NLOCALP-1))
                    di->NBND=(di->N[2]-m->NAB)*(di->N[1]-2*m->NAB)*2*m->FDOH
                    +(di->N[2]-m->NAB)*(di->N[0]-2*m->NAB-2*m->FDOH)*2*m->FDOH
                    +(di->N[1]-2*m->NAB-2*m->FDOH)*(di->N[0]-2*m->NAB-2*m->FDOH)
                    *m->FDOH;
                
                else
                    di->NBND=di->N[2]*(di->N[1]-2*m->NAB)*2*m->FDOH+
                    di->N[2]*(di->N[0]-2*m->NAB-2*m->FDOH)*2*m->FDOH;
            }
            else{
                if (m->NUM_DEVICES==1 && m->NLOCALP==1)
                    di->NBND=(di->N[1]-2*m->NAB)*2*m->FDOH+
                    (di->N[0]-2*m->NAB-2*m->FDOH)*2*m->FDOH;
                
                else if ( (d==0 && m->MYGROUPID==0)
                         || (d==m->NUM_DEVICES-1 && m->MYGROUPID==m->NLOCALP-1))
                    di->NBND=(di->N[1]-m->NAB)*m->FDOH+
                    (di->N[0]-2*m->NAB-m->FDOH)*2*m->FDOH;
                
                else
                    di->NBND= (di->N[1])*2*m->FDOH;
            }
            
            di->grads.savebnd.gsize[0]=di->NBND;

            for (i=0;i<m->nvars;i++){
                if (di->vars[i].to_comm){
                    di->vars[i].cl_varbnd.size=sizeof(float) * di->NBND;
                    di->vars[i].cl_varbnd.sizepin=sizeof(float) *di->NBND*m->NT;
                    __GUARD clbuf_create_pin( &di->vars[i].cl_varbnd);
                }
            }
            
            


        }
        
        fprintf(stdout,"memory\n");
        // Create the update kernels
        di->nupdates=m->nupdates;
        for (i=0;i<m->nupdates;i++){
            di->ups_f[i].center.OFFCOMM=offcom1;
            di->ups_f[i].center.LCOMM=LCOMM;
            __GUARD prog_create(m, di,  &di->ups_f[i].center);
            if (d>0 || m->MYLOCALID>0){
                di->ups_f[i].com1.OFFCOMM=0;
                di->ups_f[i].com1.LCOMM=LCOMM;
                di->ups_f[i].com1.COMM=1;
                __GUARD prog_create(m, di,  &di->ups_f[i].com1);
                
                __GUARD kernel_fcom_out( di ,di->vars,
                                        &di->ups_f[i].fcom1_out, i+1, 1);
                __GUARD prog_create(m, di,  &di->ups_f[i].fcom1_out);
                __GUARD kernel_fcom_in( di ,di->vars,
                                       &di->ups_f[i].fcom1_in, i+1, 1);
                __GUARD prog_create(m, di,  &di->ups_f[i].fcom1_in);
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                di->ups_f[i].com2.OFFCOMM=offcom2;
                di->ups_f[i].com2.LCOMM=LCOMM;
                di->ups_f[i].com2.COMM=1;
                __GUARD prog_create(m, di,  &di->ups_f[i].com2);
                
                __GUARD kernel_fcom_out( di, di->vars,
                                     &di->ups_f[i].fcom2_out, i+1, 2);
                __GUARD prog_create(m, di,  &di->ups_f[i].fcom2_out);
                __GUARD kernel_fcom_in(di ,di->vars,
                                   &di->ups_f[i].fcom2_in, i+1, 2);
                __GUARD prog_create(m, di,  &di->ups_f[i].fcom2_in);
            }
        }
//        if (m->GRADOUT){
//            for (i=0;i<m->nupdates;i++){
//                di->ups_adj[i].center.OFFCOMM=offcom1;
//                di->ups_adj[i].center.LCOMM=LCOMM;
//                __GUARD prog_create(m, di,  &di->ups_adj[i].center);
//                if (d>0 || m->MYLOCALID>0){
//                    di->ups_adj[i].com1.OFFCOMM=0;
//                    di->ups_adj[i].com1.LCOMM=LCOMM;
//                    di->ups_adj[i].com1.COMM=1;
//                    __GUARD prog_create(m, di,  &di->ups_adj[i].com1);
//                    
//                    __GUARD kernel_fcom_out( di ,di->vars,
//                                            &di->ups_adj[i].fcom1_out, i+1, 1);
//                    __GUARD prog_create(m, di,  &di->ups_adj[i].fcom1_out);
//                    __GUARD kernel_fcom_in( di ,di->vars,
//                                           &di->ups_adj[i].fcom1_in, i+1, 1);
//                    __GUARD prog_create(m, di,  &di->ups_adj[i].fcom1_in);
//                }
//                if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
//                    di->ups_adj[i].com2.OFFCOMM=offcom2;
//                    di->ups_adj[i].com2.LCOMM=LCOMM;
//                    di->ups_adj[i].com2.COMM=1;
//                    __GUARD prog_create(m, di,  &di->ups_adj[i].com2);
//                    
//                    __GUARD kernel_fcom_out( di, di->vars,
//                                            &di->ups_adj[i].fcom2_out, i+1, 2);
//                    __GUARD prog_create(m, di,  &di->ups_adj[i].fcom2_out);
//                    __GUARD kernel_fcom_in(di ,di->vars,
//                                           &di->ups_adj[i].fcom2_in, i+1, 2);
//                    __GUARD prog_create(m, di,  &di->ups_adj[i].fcom2_in);
//                }
//            }
//        }
//        
//        //Create automaticly kernels for gradient, variable inti, sources ...
//        __GUARD kernel_sources(di,  &di->src_recs.sources);
//        __GUARD prog_create(m, di,  &di->src_recs.sources);
//        
//        __GUARD kernel_varout(di, &di->src_recs.varsout);
//        __GUARD prog_create(m, di,  &di->src_recs.varsout);
//        
//        __GUARD kernel_varoutinit(di, &di->src_recs.varsoutinit);
//        __GUARD prog_create(m, di,  &di->src_recs.varsoutinit);
//        
//        __GUARD kernel_varinit(di, di->vars, &di->bnd_cnds.init_f);
//        __GUARD prog_create(m, di,  &di->bnd_cnds.init_f);
//        di->bnd_cnds.init_f.gsize[0]=fdsize;
//        
//        
//        if (m->GRADOUT){
//            __GUARD kernel_residuals(di,
//                                     &di->src_recs.residuals,
//                                     m->BACK_PROP_TYPE);
//            __GUARD prog_create(m, di,  &di->src_recs.residuals);
//            
//            if (m->BACK_PROP_TYPE==1){
//                __GUARD kernel_varinit(di,di->vars_adj, &di->bnd_cnds.init_adj);
//                __GUARD prog_create(m, di,  &di->bnd_cnds.init_adj);
//                di->bnd_cnds.init_adj.gsize[0]=fdsize;
//                
//                __GUARD kernel_gradinit(di, di->pars, &di->grads.init);
//                __GUARD prog_create(m, di,  &di->grads.init);
//                di->grads.init.gsize[0]=parsize;
//            }
//            else if(m->BACK_PROP_TYPE==2){
//                __GUARD kernel_initsavefreqs(di, di->vars,
//                                                      &di->grads.initsavefreqs);
//                __GUARD prog_create(m, di,  &di->grads.initsavefreqs);
//                di->grads.initsavefreqs.gsize[0]=fdsize;
//                
//                kernel_savefreqs(di, di->vars, &di->grads.savefreqs);
//                __GUARD prog_create(m, di,  &di->grads.savefreqs);
//                di->grads.savefreqs.gsize[0]=fdsize;
//            }
//            
//            if (m->GRADSRCOUT){
//                __GUARD kernel_init_gradsrc( &di->src_recs.init_gradsrc);
//                __GUARD prog_create(m, di,  &di->src_recs.init_gradsrc);
//            }
//        }
//    
//        
//        //TODO Boundary conditions should be included in the update kernel
//        //TODO Adjoint free surface
//        if (m->FREESURF){
//            di->bnd_cnds.surf=m->bnd_cnds.surf;
//            __GUARD prog_create(m, di,  &di->bnd_cnds.surf);
//            di->bnd_cnds.surf.wdim=m->NDIM-1;
//            for (i=1;i<m->NDIM;i++){
//                di->bnd_cnds.surf.gsize[i-1]=di->N[i];
//            }
//        }
//        
//        //TODO Create automatically the kernel for saving boundary
//        //TODO Implement random boundaries instead
//        if (m->GRADOUT && m->BACK_PROP_TYPE==1){
//            di->grads.savebnd=m->grads.savebnd;
//            __GUARD prog_create(m, di,  &di->grads.savebnd);
//            di->grads.savebnd.wdim=1;
//            di->grads.savebnd.gsize[0]=di->NBND;
//            
//        }
        
        
    }
    
    
    

    if (state && m->MPI_INIT==1)
        MPI_Bcast( &state, 1, MPI_INT, m->MYID, MPI_COMM_WORLD );
    
    return state;


}
