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

int Init_OpenCL(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc)  {

    int state=0;
    int i,s,d;
    size_t buffer_size_s=0;
    size_t buffer_size_ns=0;
    size_t buffer_size_ng=0;
    size_t buffer_size_taper=0;
    size_t buffer_size_seisout=0;
    size_t buffer_size_L=0;
    size_t thissize=0;
    cl_platform_id sel_plat_id=0;
    
    cl_device_id device=0;
    cl_ulong global_mem_size=0;
    cl_ulong local_mem_size=0;
    cl_ulong required_local_mem_size=0;
    size_t workitem_size[3];
    size_t workgroup_size=0;
    int lsizez=0;
    int lsizex=0;
    int lsizey=0;
    int lcomm=0;
    int offcomm1=0;
    int offcomm2=0;
    
    // Find a platform where the prefered device type can be found
    __GUARD GetPlatformID( &m->pref_device_type, &m->device_type, &sel_plat_id, &m->num_devices, m->n_no_use_GPUs, m->no_use_GPUs);
    if (m->num_devices>m->nmax_dev)
        m->num_devices=m->nmax_dev;

    //For each GPU, allocate the memory structures
    GMALLOC((*vcl),sizeof(struct varcl)*m->num_devices)
    if (!state) memset ((void*)(*vcl), 0, sizeof(struct varcl)*m->num_devices);
    GMALLOC((*mloc),sizeof(struct modcstsloc)*m->num_devices)
    if (!state) memset ((void*)(*mloc), 0, sizeof(struct modcstsloc)*m->num_devices);
    
    //Connect all GPUs
    
    __GUARD connect_allgpus( vcl, &m->context, &m->device_type, &sel_plat_id, m->n_no_use_GPUs, m->no_use_GPUs,m->nmax_dev);
    
    
    //For each device, create the memory buffers and programs on the GPU
    for (d=0; d<m->num_devices; d++) {
        
        //The domain of the seismic simulation is decomposed between the devices
        //along the X direction.
        if (!state){
            (*mloc)[d].dev=d;
            (*mloc)[d].num_devices=m->num_devices;
            (*mloc)[d].NY=m->NY;
            (*mloc)[d].NZ=m->NZ;
            
            
            (*mloc)[d].NZ_al0=m->fdoh%16;
            (*mloc)[d].NZ_al16= ((*mloc)[d].NZ+m->FDORDER)%16;
            
            (*mloc)[d].NZ_al0=0;
            (*mloc)[d].NZ_al16= 0;
            
            
            if (m->MYLOCALID<m->NX%m->NLOCALP){
                (*mloc)[d].NX=m->NX/m->NLOCALP+1;
                m->NXP=m->NX/m->NLOCALP+1;
            }
            else{
                (*mloc)[d].NX=m->NX/m->NLOCALP;
                m->NXP=m->NX/m->NLOCALP;
            }
            if (d<(*mloc)[d].NX%m->num_devices)
                (*mloc)[d].NX= (*mloc)[d].NX/m->num_devices+1;
            else
                (*mloc)[d].NX= (*mloc)[d].NX/m->num_devices;

            (*mloc)[d].offset=0;
            (*mloc)[d].NX0=0;
            (*mloc)[d].offsetfd=0;
            
            for (i=0;i<m->MYLOCALID;i++){
                if (i<m->NX%m->NLOCALP){
                    (*mloc)[d].offset+=(m->NX/m->NLOCALP+1)*m->NY*m->NZ;
                    (*mloc)[d].NX0+=(m->NX/m->NLOCALP+1);
                }
                else{
                    (*mloc)[d].offset+=(m->NX/m->NLOCALP)*m->NY*m->NZ;
                    (*mloc)[d].NX0+=(m->NX/m->NLOCALP);
                }
                
            }


            for (i=0;i<d;i++){
                (*mloc)[d].NX0+=(*mloc)[i].NX;
                (*mloc)[d].offset+=(*mloc)[i].NX*(*mloc)[i].NY*(*mloc)[i].NZ;
                if (m->ND==3){
                    (*mloc)[d].offsetfd+=((*mloc)[i].NX+m->FDORDER)*((*mloc)[i].NY+m->FDORDER)*((*mloc)[i].NZ+m->FDORDER);
                }
                else {
                    (*mloc)[d].offsetfd+=((*mloc)[i].NX+m->FDORDER)*((*mloc)[i].NZ+m->FDORDER);
                }
            }
        }

        // Create a pointer at the right position (offset) of the decomposed model
        if (!state){
            if (m->rho)      (*mloc)[d].rho      =&m->rho[(*mloc)[d].offset];
            if (m->rip)      (*mloc)[d].rip      =&m->rip[(*mloc)[d].offset];
            if (m->rkp)      (*mloc)[d].rkp      =&m->rkp[(*mloc)[d].offset];
            if (m->u)        (*mloc)[d].u        =&m->u[(*mloc)[d].offset];
            if (m->pi)       (*mloc)[d].pi       =&m->pi[(*mloc)[d].offset];
            if (m->uipkp)    (*mloc)[d].uipkp    =&m->uipkp[(*mloc)[d].offset];
            if (m->taus)     (*mloc)[d].taus     =&m->taus[(*mloc)[d].offset];
            if (m->tausipkp) (*mloc)[d].tausipkp =&m->tausipkp[(*mloc)[d].offset];
            if (m->taup)     (*mloc)[d].taup     =&m->taup[(*mloc)[d].offset];
            if (m->rjp)      (*mloc)[d].rjp      =&m->rjp[(*mloc)[d].offset];
            if (m->uipjp)    (*mloc)[d].uipjp    =&m->uipjp[(*mloc)[d].offset];
            if (m->ujpkp)    (*mloc)[d].ujpkp    =&m->ujpkp[(*mloc)[d].offset];
            if (m->tausipjp) (*mloc)[d].tausipjp =&m->tausipjp[(*mloc)[d].offset];
            if (m->tausjpkp) (*mloc)[d].tausjpkp =&m->tausjpkp[(*mloc)[d].offset];
            
            if (m->gradrho)  (*mloc)[d].gradrho  =&m->gradrho[(*mloc)[d].offset];
            if (m->gradM)    (*mloc)[d].gradM    =&m->gradM[(*mloc)[d].offset];
            if (m->gradmu)   (*mloc)[d].gradmu   =&m->gradmu[(*mloc)[d].offset];
            if (m->gradtaup) (*mloc)[d].gradtaup =&m->gradtaup[(*mloc)[d].offset];
            if (m->gradtaus) (*mloc)[d].gradtaus =&m->gradtaus[(*mloc)[d].offset];
            
            if (m->Hrho)  (*mloc)[d].Hrho  =&m->Hrho[(*mloc)[d].offset];
            if (m->HM)    (*mloc)[d].HM    =&m->HM[(*mloc)[d].offset];
            if (m->Hmu)   (*mloc)[d].Hmu   =&m->Hmu[(*mloc)[d].offset];
            if (m->Htaup) (*mloc)[d].Htaup =&m->Htaup[(*mloc)[d].offset];
            if (m->Htaus) (*mloc)[d].Htaus =&m->Htaus[(*mloc)[d].offset];
            
            
            if (m->movvx)  (*mloc)[d].movvx  =&m->movvx[(*mloc)[d].offset];
            if (m->movvy)  (*mloc)[d].movvy  =&m->movvy[(*mloc)[d].offset];
            if (m->movvz)  (*mloc)[d].movvz  =&m->movvz[(*mloc)[d].offset];
            if (m->movout>0){
                if (m->ND==3){
                    thissize=((*mloc)[d].NX+m->FDORDER)*((*mloc)[d].NY+m->FDORDER)*((*mloc)[d].NZ+m->FDORDER)*sizeof(cl_float);
                }
                else{
                    thissize=((*mloc)[d].NX+m->FDORDER)*((*mloc)[d].NZ+m->FDORDER)*sizeof(cl_float);
                }
                if (m->ND!=21){
                    GMALLOC((*mloc)[d].buffermovvx,thissize);
                    GMALLOC((*mloc)[d].buffermovvz,thissize);
                }
                if (m->ND==3 || m->ND==21){
                    GMALLOC((*mloc)[d].buffermovvy,thissize);
                }
                
            }
            
            
            if (m->gradout==1){

                
                if (m->back_prop_type==2){
                    if (m->ND==3){
                        thissize=m->nfreqs*((*mloc)[d].NX+m->FDORDER)*((*mloc)[d].NY+m->FDORDER)*((*mloc)[d].NZ+m->FDORDER)*sizeof(cl_float2);
                    }
                    else{
                        thissize=m->nfreqs*((*mloc)[d].NX+m->FDORDER)*((*mloc)[d].NZ+m->FDORDER)*sizeof(cl_float2);
                    }
                    
                    if (m->ND!=21){
                        GMALLOC((*mloc)[d].f_vx,thissize);
                        GMALLOC((*mloc)[d].f_vz,thissize);
                        GMALLOC((*mloc)[d].f_sxx,thissize);
                        GMALLOC((*mloc)[d].f_szz,thissize);
                        GMALLOC((*mloc)[d].f_sxz,thissize);
                        GMALLOC((*mloc)[d].f_vxr,thissize);
                        GMALLOC((*mloc)[d].f_vzr,thissize);
                        GMALLOC((*mloc)[d].f_sxxr,thissize);
                        GMALLOC((*mloc)[d].f_szzr,thissize);
                        GMALLOC((*mloc)[d].f_sxzr,thissize);
                    }
                    
                    if (m->ND==3 || m->ND==21){
                        GMALLOC((*mloc)[d].f_vy,thissize);
                        GMALLOC((*mloc)[d].f_sxy,thissize);
                        GMALLOC((*mloc)[d].f_syz,thissize);
                        GMALLOC((*mloc)[d].f_vyr,thissize);
                        GMALLOC((*mloc)[d].f_sxyr,thissize);
                        GMALLOC((*mloc)[d].f_syzr,thissize);
                    }
                    if (m->ND==3 ){
                        GMALLOC((*mloc)[d].f_syy,thissize);
                        GMALLOC((*mloc)[d].f_syyr,thissize);
                    }
                    
                    if (m->L>0){
                        if (m->ND!=21){
                            GMALLOC((*mloc)[d].f_rxx,thissize*m->L);
                            GMALLOC((*mloc)[d].f_rzz,thissize*m->L);
                            GMALLOC((*mloc)[d].f_rxz,thissize*m->L);
                            GMALLOC((*mloc)[d].f_rxxr,thissize*m->L);
                            GMALLOC((*mloc)[d].f_rzzr,thissize*m->L);
                            GMALLOC((*mloc)[d].f_rxzr,thissize*m->L);
                        }
                        
                        if (m->ND==3 || m->ND==21){
                            GMALLOC((*mloc)[d].f_rxy,thissize*m->L);
                            GMALLOC((*mloc)[d].f_ryz,thissize*m->L);
                            GMALLOC((*mloc)[d].f_rxyr,thissize*m->L);
                            GMALLOC((*mloc)[d].f_ryzr,thissize*m->L);
                        }
                        if (m->ND==3){
                            GMALLOC((*mloc)[d].f_ryy,thissize*m->L);
                            GMALLOC((*mloc)[d].f_ryyr,thissize*m->L);
                        }
                        
                        
                    }
                    
                }
            }

        }

        
        // Create the memory to read the seismograms for each shot
        if (m->vxout && !state){
            alloc_seismo(&(*mloc)[d].vxout, m->ns, m->allng, m->NT, m->nrec);
        }
        if (m->vyout && !state){
            alloc_seismo(&(*mloc)[d].vyout, m->ns, m->allng, m->NT, m->nrec);
        }
        if (m->vzout && !state){
            alloc_seismo(&(*mloc)[d].vzout, m->ns, m->allng, m->NT, m->nrec);
        }
        if (m->sxxout && !state){
            alloc_seismo(&(*mloc)[d].sxxout, m->ns, m->allng, m->NT, m->nrec);
        }
        if (m->syyout && !state){
            alloc_seismo(&(*mloc)[d].syyout, m->ns, m->allng, m->NT, m->nrec);
        }
        if (m->szzout && !state){
            alloc_seismo(&(*mloc)[d].szzout, m->ns, m->allng, m->NT, m->nrec);
        }
        if (m->sxyout && !state){
            alloc_seismo(&(*mloc)[d].sxyout, m->ns, m->allng, m->NT, m->nrec);
        }
        if (m->sxzout && !state){
            alloc_seismo(&(*mloc)[d].sxzout, m->ns, m->allng, m->NT, m->nrec);
        }
        if (m->syzout && !state){
            alloc_seismo(&(*mloc)[d].syzout, m->ns, m->allng, m->NT, m->nrec);
        }
        if (m->pout && !state){
            alloc_seismo(&(*mloc)[d].pout, m->ns, m->allng, m->NT, m->nrec);
        }
        
        
        
        
        // Get some properties of the device
        __GUARD  clGetCommandQueueInfo(	(*vcl)[d].cmd_queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL);
        if (state !=CL_SUCCESS) CLPERR(state);
        
        __GUARD clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
        __GUARD clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
        __GUARD clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
        if (state !=CL_SUCCESS) CLPERR(state);
        
        //Intel SDK does not give the right max work_group_size our kernels, we force it here!
        if (!state && m->pref_device_type==CL_DEVICE_TYPE_CPU) workgroup_size= workgroup_size>1024 ? 1024:workgroup_size;
        if (!state && m->pref_device_type==CL_DEVICE_TYPE_ACCELERATOR) workgroup_size= workgroup_size>1024 ? 1024:workgroup_size;
        
        // Define the local work size and global work size of the device.
        if (!state){
            if (workitem_size[0]<m->FDORDER || workitem_size[1]<2|| workitem_size[2]<2){
                fprintf(stdout,"Maximum device work item size of device %d doesn't support 3D local memory\n", d);
                fprintf(stdout,"Switching off local memory optimization\n");
                (*mloc)[d].local_off = 1;
                
            }
            else {

                lsizez=32;
                lsizex=16;
                lsizey=16;
                if (m->ND==3){// For 3D
                    required_local_mem_size = (lsizez+m->FDORDER)*(lsizex+m->FDORDER)*(lsizey+m->FDORDER)*sizeof(float);
                    while ( (lsizex>(m->FDORDER)/2 && lsizey>(m->FDORDER)/2 &&  required_local_mem_size>local_mem_size) || lsizex*lsizey*lsizez>workgroup_size ){
                        lsizex-=2;
                        lsizey-=2;
                        required_local_mem_size = (lsizez+m->FDORDER)*(lsizex+m->FDORDER)*(lsizey+m->FDORDER)*sizeof(float);
                    }
                    if (required_local_mem_size>local_mem_size){
                        while ( (lsizex>(m->FDORDER)/4 &&  required_local_mem_size>local_mem_size) || lsizex*lsizey*lsizez>workgroup_size ){
                            lsizex-=2;
                            required_local_mem_size = (lsizez+m->FDORDER)*(lsizex+m->FDORDER)*(lsizey+m->FDORDER)*sizeof(float);
                        }
                    }
                    if (required_local_mem_size>local_mem_size){
                        while ( (lsizey>(m->FDORDER)/4 &&  required_local_mem_size>local_mem_size) || lsizex*lsizey*lsizez>workgroup_size ){
                            lsizey-=2;
                            required_local_mem_size = (lsizez+m->FDORDER)*(lsizex+m->FDORDER)*(lsizey+m->FDORDER)*sizeof(float);
                        }
                    }
  
                }
                if (m->ND==2){// For 2D
                    required_local_mem_size = (lsizez+m->FDORDER)*(lsizex+m->FDORDER)*sizeof(float);
                    while ( (lsizex>(m->FDORDER)/4  &&  required_local_mem_size>local_mem_size) || lsizex*lsizez>workgroup_size ){
                        lsizex-=2;
                        required_local_mem_size = (lsizez+m->FDORDER)*(lsizex+m->FDORDER)*sizeof(float);
                    }
                    
                }
                
                if (required_local_mem_size>0.9*local_mem_size && required_local_mem_size<local_mem_size){
                    fprintf(stderr,"Warning: local memory needed to perform seismic modeling (%llu bits) exceeds 90%% of the local memory capacity of device %d (%llu bits)\n", required_local_mem_size, d, local_mem_size);
                }
                else if (required_local_mem_size>local_mem_size){
                    
                    fprintf(stderr,"Local memory needed to perform seismic modeling (%llu bits) exceeds the local memory capacity of device %d (%llu bits)\n", required_local_mem_size, d, local_mem_size );
                    fprintf(stderr,"Switching off local memory optimization\n");
                    (*mloc)[d].local_off = 1;
                }
                
            }
            
        }
        if (!state){
            (*mloc)[d].local_work_size[0] = lsizez;
            (*mloc)[d].local_work_size[1] = lsizex;
            (*mloc)[d].local_work_size[2] = lsizey;
            
            
            /* Global sizes for the kernels */
            if ((*mloc)[d].local_off==1){
                (*mloc)[d].local_work_size[0] = 1;
                
                lcomm=0;
                offcomm1=0;
                offcomm2=0;
                if (d>0 || m->MYLOCALID>0){
                    (*mloc)[d].global_work_sizecomm1[0] = (*mloc)[d].NY*m->fdoh*(*mloc)[d].NZ;
                    lcomm+=m->fdoh;
                    offcomm1=m->fdoh;
                    
                }
                if (d<m->num_devices-1 || m->MYLOCALID<m->NLOCALP-1){
                    (*mloc)[d].global_work_sizecomm2[0] = (*mloc)[d].NY*m->fdoh*(*mloc)[d].NZ;
                    lcomm+=m->fdoh;
                    offcomm2=(*mloc)[d].NX-m->fdoh;
                }
                if (d>0 || m->MYLOCALID>0 || d<m->num_devices-1 || m->MYLOCALID<m->NLOCALP-1){
                    (*mloc)[d].global_work_size_fillcomm[0] = (*mloc)[d].NY*m->fdoh*(*mloc)[d].NZ;
                }
                
                (*mloc)[d].global_work_size[0] = ((*mloc)[d].NX-lcomm)*(*mloc)[d].NY*(*mloc)[d].NZ;
                
                (*vcl)[d].numdim=1;
                
            }
            else{
                //Check if too many GPUS are used in the domain decomposition
                if  ((*mloc)[d].NX<3*(*mloc)[d].local_work_size[1]){
                    state=1;
                    fprintf(stderr,"Too many GPUs for domain decompositon\n");
                }
                
                
                //Global work size must be a multiple of local work size
                if (m->ND==3){// For 3D
                    
                    (*mloc)[d].global_work_size[0] = (*mloc)[d].NZ
                    + ((*mloc)[d].local_work_size[0]-(*mloc)[d].NZ%(*mloc)[d].local_work_size[0])%(*mloc)[d].local_work_size[0];
                    (*mloc)[d].global_work_size[1] = (*mloc)[d].NY
                    + ((*mloc)[d].local_work_size[1]-(*mloc)[d].NY%(*mloc)[d].local_work_size[1])%(*mloc)[d].local_work_size[1];
                    
                    lcomm=0;
                    offcomm1=0;
                    offcomm2=0;
                    if (d>0 || m->MYLOCALID>0){
                        (*mloc)[d].global_work_sizecomm1[0] = (*mloc)[d].global_work_size[0];
                        (*mloc)[d].global_work_sizecomm1[1] = (*mloc)[d].global_work_size[1];
                        (*mloc)[d].global_work_sizecomm1[2] = (*mloc)[d].local_work_size[2];
                        lcomm+=(*mloc)[d].local_work_size[2];
                        offcomm1=(int)(*mloc)[d].local_work_size[2];
                        
                    }
                    if (d<m->num_devices-1 || m->MYLOCALID<m->NLOCALP-1){
                        (*mloc)[d].global_work_sizecomm2[0] = (*mloc)[d].global_work_size[0];
                        (*mloc)[d].global_work_sizecomm2[1] = (*mloc)[d].global_work_size[1];
                        (*mloc)[d].global_work_sizecomm2[2] = (*mloc)[d].local_work_size[2];
                        lcomm+=(*mloc)[d].local_work_size[2];
                        offcomm2=(*mloc)[d].NX-(int)(*mloc)[d].local_work_size[2];
                    }
                    if (d>0 || m->MYLOCALID>0 || d<m->num_devices-1 || m->MYLOCALID<m->NLOCALP-1){
                        (*mloc)[d].global_work_size_fillcomm[0] = (*mloc)[d].NZ;
                        (*mloc)[d].global_work_size_fillcomm[1] = (*mloc)[d].NY;
                        (*mloc)[d].global_work_size_fillcomm[2] = m->fdoh;
                    }
                    
                    
                    (*mloc)[d].global_work_size[2] = (*mloc)[d].NX-lcomm
                    + ((*mloc)[d].local_work_size[2]-((*mloc)[d].NX-lcomm)%(*mloc)[d].local_work_size[2])%(*mloc)[d].local_work_size[2];
                    
                    (*vcl)[d].numdim=3;
                    
                }
                else{
                    (*mloc)[d].global_work_size[0] = (*mloc)[d].NZ
                    + ((*mloc)[d].local_work_size[0]-(*mloc)[d].NZ%(*mloc)[d].local_work_size[0])%(*mloc)[d].local_work_size[0];
                    
                    lcomm=0;
                    offcomm1=0;
                    offcomm2=0;
                    if (d>0 || m->MYLOCALID>0){
                        (*mloc)[d].global_work_sizecomm1[0] = (*mloc)[d].global_work_size[0];
                        (*mloc)[d].global_work_sizecomm1[1] = (*mloc)[d].local_work_size[1];
                        lcomm+=(*mloc)[d].local_work_size[1];
                        offcomm1=(int)(*mloc)[d].local_work_size[1];
                    }
                    if (d<m->num_devices-1 || m->MYLOCALID<m->NLOCALP-1){
                        (*mloc)[d].global_work_sizecomm2[0] = (*mloc)[d].global_work_size[0];
                        (*mloc)[d].global_work_sizecomm2[1] = (*mloc)[d].local_work_size[1];
                        lcomm+=(*mloc)[d].local_work_size[1];
                        offcomm2=(*mloc)[d].NX-(int)(*mloc)[d].local_work_size[1];
                    }
                    if (d>0 || m->MYLOCALID>0 || d<m->num_devices-1 || m->MYLOCALID<m->NLOCALP-1){
                        (*mloc)[d].global_work_size_fillcomm[0] = (*mloc)[d].NZ;
                        (*mloc)[d].global_work_size_fillcomm[1] = m->fdoh;
                    }
                    
                    (*mloc)[d].global_work_size[1] = (*mloc)[d].NX-lcomm
                    + ((*mloc)[d].local_work_size[1]-((*mloc)[d].NX-lcomm)%(*mloc)[d].local_work_size[1])%(*mloc)[d].local_work_size[1];
                    
                    (*vcl)[d].numdim=2;
                    
                }
            }
            

        }
        
        
        // Calculate the dimension of the buffers needed to transfer memory from/to the GPU
        if (!state){
            
            if (m->ND==3){// For 3D
                (*vcl)[d].buffer_size_fd    = sizeof(float)
                * ((*mloc)[d].NX+m->FDORDER)
                * ((*mloc)[d].NY+m->FDORDER)
                * ((*mloc)[d].NZ+m->FDORDER);
            }
            else{// For 2D
                (*vcl)[d].buffer_size_fd    = sizeof(float)
                * ((*mloc)[d].NX+m->FDORDER)
                * ((*mloc)[d].NZ+m->FDORDER);
            }

            (*vcl)[d].buffer_size_model = sizeof(float) * (*mloc)[d].NX * (*mloc)[d].NY * (*mloc)[d].NZ;
            if (m->ND==3){
                (*vcl)[d].buffer_size_modelc = sizeof(cl_float2)*( (*mloc)[d].NX+m->FDORDER ) *( (*mloc)[d].NY+m->FDORDER )*( (*mloc)[d].NZ+m->FDORDER )* m->nfreqs;
            }
            else{
                (*vcl)[d].buffer_size_modelc = sizeof(cl_float2)*( (*mloc)[d].NX+m->FDORDER )*( (*mloc)[d].NZ+m->FDORDER )* m->nfreqs;
            }
            buffer_size_s        = sizeof(float) * m->NT * m->nsmax;
            buffer_size_ns       = sizeof(float) * 5 * m->nsmax;
            buffer_size_ng       = sizeof(float) * 8 * m->ngmax;
            buffer_size_taper    = sizeof(float) * m->nab;
            buffer_size_seisout     = sizeof(float) * m->NT * m->ngmax;
            buffer_size_L        = sizeof(float) * m->L;
            m->buffer_size_comm     = sizeof(float) * m->fdoh*(*mloc)[d].NY*(*mloc)[d].NZ;
            
            if (m->abs_type==1){
                (*vcl)[d].buffer_size_CPML_NX=sizeof(float) * 2*m->nab * (*mloc)[d].NY * (*mloc)[d].NZ;
                if (m->ND==3){// For 3D
                    (*vcl)[d].buffer_size_CPML_NY=sizeof(float) * 2*m->nab * (*mloc)[d].NX * (*mloc)[d].NZ;
                }
                (*vcl)[d].buffer_size_CPML_NZ=sizeof(float) * 2*m->nab * (*mloc)[d].NY * (*mloc)[d].NX;
            }
            
            // Determine the size of the outside boundary used for the back propagation of the seismic wavefield
            if (m->back_prop_type==1){
            if (m->ND==3){// For 3D
                if (m->num_devices==1 && m->NLOCALP==1)
                    (*mloc)[d].Nbnd=((*mloc)[d].NX-2*m->nab)*((*mloc)[d].NY-2*m->nab)*2*m->fdoh+
                    ((*mloc)[d].NX-2*m->nab)*((*mloc)[d].NZ-2*m->nab-2*m->fdoh)*2*m->fdoh+
                    ((*mloc)[d].NY-2*m->nab-2*m->fdoh)*((*mloc)[d].NZ-2*m->nab-2*m->fdoh)*2*m->fdoh;
                
                else if ( (d==0 && m->MYGROUPID==0) || (d==m->num_devices-1 && m->MYGROUPID==m->NLOCALP-1) )
                    (*mloc)[d].Nbnd=((*mloc)[d].NX-m->nab)*((*mloc)[d].NY-2*m->nab)*2*m->fdoh+
                    ((*mloc)[d].NX-m->nab)*((*mloc)[d].NZ-2*m->nab-2*m->fdoh)*2*m->fdoh+
                    ((*mloc)[d].NY-2*m->nab-2*m->fdoh)*((*mloc)[d].NZ-2*m->nab-2*m->fdoh)*m->fdoh;
                
                else
                    (*mloc)[d].Nbnd=(*mloc)[d].NX*((*mloc)[d].NY-2*m->nab)*2*m->fdoh+
                    (*mloc)[d].NX*((*mloc)[d].NZ-2*m->nab-2*m->fdoh)*2*m->fdoh;
            }
            else{
                if (m->num_devices==1 && m->NLOCALP==1)
                    (*mloc)[d].Nbnd=((*mloc)[d].NX-2*m->nab)*2*m->fdoh+
                    ((*mloc)[d].NZ-2*m->nab-2*m->fdoh)*2*m->fdoh;
                
                else if ( (d==0 && m->MYGROUPID==0) || (d==m->num_devices-1 && m->MYGROUPID==m->NLOCALP-1) )
                    (*mloc)[d].Nbnd=((*mloc)[d].NX-m->nab)*m->fdoh+
                    ((*mloc)[d].NZ-2*m->nab-m->fdoh)*2*m->fdoh;
                
                else
                    (*mloc)[d].Nbnd= ((*mloc)[d].NX)*2*m->fdoh;
            }
            
            (*mloc)[d].global_work_size_bnd =(*mloc)[d].Nbnd ;
                
            
            
            //During backpropagation of the seismic field, we inject at the positions given by the absorbing boundary
                (*vcl)[d].buffer_size_bnd = sizeof(float)*(*mloc)[d].Nbnd;
            }
            
            if (m->ND==3){// For 3D
                (*mloc)[d].global_work_size_surf[0] = (*mloc)[d].NY;
                (*mloc)[d].global_work_size_surf[1] = (*mloc)[d].NX;
                (*mloc)[d].global_work_size_initfd=  (*vcl)[d].buffer_size_fd/sizeof(float);
                (*mloc)[d].global_work_size_f = ((*mloc)[d].NY+2*m->fdoh)*((*mloc)[d].NX+2*m->fdoh)*((*mloc)[d].NZ+2*m->fdoh);
                
                (*mloc)[d].global_work_size_surfgrid[0] = (*mloc)[d].NZ;
                (*mloc)[d].global_work_size_surfgrid[1] = (*mloc)[d].NY;
                (*mloc)[d].global_work_size_surfgrid[2] = (*mloc)[d].NZ;
                
            }
            else{
                (*mloc)[d].global_work_size_surf[0] = (*mloc)[d].NX;
                (*mloc)[d].global_work_size_initfd= (*vcl)[d].buffer_size_fd/sizeof(float);
                (*mloc)[d].global_work_size_f = ((*mloc)[d].NX+2*m->fdoh)*((*mloc)[d].NZ+2*m->fdoh);
                
                (*mloc)[d].global_work_size_surfgrid[0] = (*mloc)[d].NZ;
                (*mloc)[d].global_work_size_surfgrid[1] = (*mloc)[d].NX;
                
            }
            (*mloc)[d].global_work_size_init=  (*mloc)[d].NX*(*mloc)[d].NY*(*mloc)[d].NZ;
            (*mloc)[d].global_work_size_gradsrc = m->NT*m->nsmax ;
            
        }

        
        // Check global memory is sufficient
        __GUARD clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
        if (state !=CL_SUCCESS) CLPERR(state);
 
        if (!state){
            if (m->ND==3){// For 3D
                (*mloc)[d].required_global_mem_size += 9*(*vcl)[d].buffer_size_fd + 8*(*vcl)[d].buffer_size_model + buffer_size_s + buffer_size_ns + buffer_size_ng;
                
                (*mloc)[d].required_global_mem_size +=  3*buffer_size_seisout;
                
                if (m->abs_type==1){
                    (*mloc)[d].required_global_mem_size+= 36*buffer_size_taper + 6*(*vcl)[d].buffer_size_CPML_NX+ 6*(*vcl)[d].buffer_size_CPML_NY+ 6*(*vcl)[d].buffer_size_CPML_NZ;
                }
                else if (m->abs_type==2){
                    (*mloc)[d].required_global_mem_size+= buffer_size_taper;
                }
                
                if (m->L>0){
                    (*mloc)[d].required_global_mem_size+= 6*(*vcl)[d].buffer_size_fd*m->L + 5*(*vcl)[d].buffer_size_model;
                }
                
                if (m->gradout==1 && m->back_prop_type==1 ){
                    
                    (*mloc)[d].required_global_mem_size+= 9*(*vcl)[d].buffer_size_fd+ 9*(*vcl)[d].buffer_size_bnd + 3*(*vcl)[d].buffer_size_model;
                    if (m->L>0){
                        (*mloc)[d].required_global_mem_size+= 6*(*vcl)[d].buffer_size_fd*m->L + 2*(*vcl)[d].buffer_size_model;
                    }
                }
                
                if (m->gradout==1 && m->back_prop_type==2){
                    
                    (*mloc)[d].required_global_mem_size+= 9*(*vcl)[d].buffer_size_modelc;
                    if (m->L>0){
                        (*mloc)[d].required_global_mem_size+= m->L*6*(*vcl)[d].buffer_size_modelc;
                    }
                }
                
            }
            if (m->ND==2){// For 2D
                (*mloc)[d].required_global_mem_size += 6*(*vcl)[d].buffer_size_fd + 5*(*vcl)[d].buffer_size_model + buffer_size_s + buffer_size_ns + buffer_size_ng;
                
                (*mloc)[d].required_global_mem_size +=  3*buffer_size_seisout;
                
                if (m->abs_type==1){
                    (*mloc)[d].required_global_mem_size+= 16*buffer_size_taper + 4*(*vcl)[d].buffer_size_CPML_NX+ 4*(*vcl)[d].buffer_size_CPML_NZ;
                }
                
                else if (m->abs_type==2){
                    (*mloc)[d].required_global_mem_size+=buffer_size_taper;
                }
                
                if (m->L>0){
                    (*mloc)[d].required_global_mem_size+= 3*(*vcl)[d].buffer_size_fd*m->L + 3*(*vcl)[d].buffer_size_model;
                }
                
                if (m->gradout==1 && m->back_prop_type==1 ){
                    
                    (*mloc)[d].required_global_mem_size+= 6*(*vcl)[d].buffer_size_fd+ 5*(*vcl)[d].buffer_size_bnd + 3*(*vcl)[d].buffer_size_model;
                    if (m->L>0){
                        (*mloc)[d].required_global_mem_size+= 3*(*vcl)[d].buffer_size_fd*m->L + 2*(*vcl)[d].buffer_size_model;
                    }
                }
                
                if (m->gradout==1 && m->back_prop_type==2){
                    
                    (*mloc)[d].required_global_mem_size+= 6*(*vcl)[d].buffer_size_modelc;
                    if (m->L>0){
                        (*mloc)[d].required_global_mem_size+= m->L*3*(*vcl)[d].buffer_size_modelc;
                    }
                }
                
            }
            if (m->ND==21){// For 2D
                (*mloc)[d].required_global_mem_size += 3*(*vcl)[d].buffer_size_fd + 3*(*vcl)[d].buffer_size_model + buffer_size_s +buffer_size_ns + buffer_size_ng;
                
                (*mloc)[d].required_global_mem_size +=   3*buffer_size_seisout;
                
                if (m->abs_type==1){
                    (*mloc)[d].required_global_mem_size+= 12*buffer_size_taper + 2*(*vcl)[d].buffer_size_CPML_NX+ 2*(*vcl)[d].buffer_size_CPML_NZ;
                }
                else if (m->abs_type==2){
                    (*mloc)[d].required_global_mem_size+=buffer_size_taper;
                }
                
                if (m->L>0){
                    (*mloc)[d].required_global_mem_size+= 2*(*vcl)[d].buffer_size_fd*m->L + 2*(*vcl)[d].buffer_size_model;
                }
                
                if (m->gradout==1 && m->back_prop_type==1){
                    
                    (*mloc)[d].required_global_mem_size+= 3*(*vcl)[d].buffer_size_fd+ 3*(*vcl)[d].buffer_size_bnd + 2*(*vcl)[d].buffer_size_model;
                    if (m->L>0){
                        (*mloc)[d].required_global_mem_size+= 2*(*vcl)[d].buffer_size_fd*m->L + 1*(*vcl)[d].buffer_size_model;
                    }
                }
                
                if (m->gradout==1 && m->back_prop_type==2){
                    
                    (*mloc)[d].required_global_mem_size+= 3*(*vcl)[d].buffer_size_modelc;
                    if (m->L>0){
                        (*mloc)[d].required_global_mem_size+= m->L*2*(*vcl)[d].buffer_size_modelc;
                    }
                }
                
            }
            

            if ((*mloc)[d].required_global_mem_size>0.9*global_mem_size && (*mloc)[d].required_global_mem_size<global_mem_size){
                fprintf(stderr,"Warning: memory needed to perform seismic modeling (%llu bits) exceeds 90%% of the memory capacity of device %d (%llu bits)\n", (*mloc)[d].required_global_mem_size, d, global_mem_size);
            }
            else if ((*mloc)[d].required_global_mem_size>global_mem_size){
                
                fprintf(stderr,"Memory needed to perform seismic modeling (%llu bits) exceeds the memory capacity of device %d (%llu bits)\nTerminating\n", (*mloc)[d].required_global_mem_size, d, global_mem_size );
                state=1;
            }
        }
        
        
       
        // Create the memory buffers for the seismic variables of the GPU
        __GUARD create_gpu_memory_buffer_cst( &m->context, buffer_size_ns,              &(*vcl)[d].src_pos);
        __GUARD create_gpu_memory_buffer_cst( &m->context, buffer_size_ng,              &(*vcl)[d].rec_pos);
        __GUARD create_gpu_memory_buffer_cst( &m->context, buffer_size_s,               &(*vcl)[d].src);
        // Allocate memory for the seismograms
        if ( m->bcastvx){
            __GUARD create_gpu_memory_buffer( &m->context, buffer_size_seisout,            &(*vcl)[d].vxout);
        }
        if ( m->bcastvy){
            __GUARD create_gpu_memory_buffer( &m->context, buffer_size_seisout,            &(*vcl)[d].vyout);
        }
        if ( m->bcastvz){
            __GUARD create_gpu_memory_buffer( &m->context, buffer_size_seisout,            &(*vcl)[d].vzout);
        }
        if ( m->bcastp){
            __GUARD create_gpu_memory_buffer( &m->context, buffer_size_seisout,            &(*vcl)[d].pout);
        }
        if ( m->bcastsxx){
            __GUARD create_gpu_memory_buffer( &m->context, buffer_size_seisout,            &(*vcl)[d].sxxout);
        }
        if ( m->bcastsyy){
            __GUARD create_gpu_memory_buffer( &m->context, buffer_size_seisout,            &(*vcl)[d].syyout);
        }
        if ( m->bcastszz){
            __GUARD create_gpu_memory_buffer( &m->context, buffer_size_seisout,            &(*vcl)[d].szzout);
        }
        if ( m->bcastsxy){
            __GUARD create_gpu_memory_buffer( &m->context, buffer_size_seisout,            &(*vcl)[d].sxyout);
        }
        if ( m->bcastsxz){
            __GUARD create_gpu_memory_buffer( &m->context, buffer_size_seisout,            &(*vcl)[d].sxzout);
        }
        if ( m->bcastsyz){
            __GUARD create_gpu_memory_buffer( &m->context, buffer_size_seisout,            &(*vcl)[d].syzout);
        }

        if (m->ND!=21){
            __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].sxx);
            __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].szz);
            __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].sxz);
            __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].vx);
            __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].vz);
            __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].rip);
            __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].rkp);
            __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].u);
            __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].pi);
            __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].uipkp);

        }
        if (m->ND==3 || m->ND==21){
            __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].syy);
            __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].sxy);
            __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].syz);
            __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].vy);
            __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].rjp);
            __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].uipjp);
            __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].ujpkp);
        }
        if (m->ND==3){// For 3D
            __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].syy);
        }
        // Create the visco-elastic variables if visco-elastic is demanded
        if (m->L>0){
            __GUARD create_gpu_memory_buffer_cst( &m->context, buffer_size_L,                 &(*vcl)[d].eta);
            if (m->ND!=21){
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd*m->L, &(*vcl)[d].rxx);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd*m->L, &(*vcl)[d].rzz);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd*m->L, &(*vcl)[d].rxz);
                __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].buffer_size_model,   &(*vcl)[d].taup);
                __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].buffer_size_model,   &(*vcl)[d].taus);
                __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].buffer_size_model,   &(*vcl)[d].tausipkp);
            }
            if (m->ND==3 || m->ND==21){
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd*m->L, &(*vcl)[d].ryy);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd*m->L, &(*vcl)[d].rxy);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd*m->L, &(*vcl)[d].ryz);
                __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].buffer_size_model,   &(*vcl)[d].tausipjp);
                __GUARD create_gpu_memory_buffer_cst( &m->context, (*vcl)[d].buffer_size_model,   &(*vcl)[d].tausjpkp);
            }
            if (m->ND==3){// For 3D
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd*m->L, &(*vcl)[d].ryy);
            }
        }
        if (m->abs_type==1){
            if (m->ND!=21){
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NX,    &(*vcl)[d].psi_sxx_x);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NX,    &(*vcl)[d].psi_sxz_x);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NZ,    &(*vcl)[d].psi_szz_z);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NZ,    &(*vcl)[d].psi_sxz_z);
                
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NX,    &(*vcl)[d].psi_vxx);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NZ,    &(*vcl)[d].psi_vzz);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NZ,    &(*vcl)[d].psi_vxz);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NX,    &(*vcl)[d].psi_vzx);
            }
            
            __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].K_x);
            __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].a_x);
            __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].b_x);
            __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].K_x_half);
            __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].a_x_half);
            __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].b_x_half);
            
            __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].K_z);
            __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].a_z);
            __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].b_z);
            __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].K_z_half);
            __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].a_z_half);
            __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].b_z_half);

            if (m->ND==3 || m->ND==21){
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NX,    &(*vcl)[d].psi_sxy_x);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NZ,    &(*vcl)[d].psi_syz_z);
                
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NX,    &(*vcl)[d].psi_vyx);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NZ,    &(*vcl)[d].psi_vyz);

            }
            if (m->ND==3 ){// For 3D
                
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NY,    &(*vcl)[d].psi_syy_y);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NY,    &(*vcl)[d].psi_sxy_y);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NY,    &(*vcl)[d].psi_syz_y);
                
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NY,    &(*vcl)[d].psi_vyy);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NY,    &(*vcl)[d].psi_vxy);
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_CPML_NY,    &(*vcl)[d].psi_vzy);
                
                __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].K_y);
                __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].a_y);
                __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].b_y);
                __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].K_y_half);
                __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].a_y_half);
                __GUARD create_gpu_memory_buffer_cst( &m->context, 2*buffer_size_taper,    &(*vcl)[d].b_y_half);
            }
            
        }
        else if (m->abs_type==2) {
            __GUARD create_gpu_memory_buffer( &m->context, buffer_size_taper, &(*vcl)[d].taper);
        }
        if (state !=CL_SUCCESS) CLPERR(state);
        // Create the sub-buffers that will transfer seismic variables between the GPUs at each time step
        if (m->ND!=21){
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].sxx_sub1, &(*mloc)[d].sxx_sub1);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].szz_sub1, &(*mloc)[d].szz_sub1);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].sxz_sub1, &(*mloc)[d].sxz_sub1);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].vx_sub1, &(*mloc)[d].vx_sub1);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].vz_sub1, &(*mloc)[d].vz_sub1);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].sxx_sub2, &(*mloc)[d].sxx_sub2);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].szz_sub2, &(*mloc)[d].szz_sub2);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].sxz_sub2, &(*mloc)[d].sxz_sub2);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].vx_sub2, &(*mloc)[d].vx_sub2);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].vz_sub2, &(*mloc)[d].vz_sub2);
            
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].sxx_sub1_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].szz_sub1_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].sxz_sub1_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].vx_sub1_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].vz_sub1_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].sxx_sub2_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].szz_sub2_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].sxz_sub2_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].vx_sub2_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].vz_sub2_dev);
        }
        if (m->ND==3 || m->ND==21){
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].sxy_sub1, &(*mloc)[d].sxy_sub1);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].syz_sub1, &(*mloc)[d].syz_sub1);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].vy_sub1, &(*mloc)[d].vy_sub1);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].sxy_sub2, &(*mloc)[d].sxy_sub2);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].syz_sub2, &(*mloc)[d].syz_sub2);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].vy_sub2, &(*mloc)[d].vy_sub2);
            
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].sxy_sub1_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].syz_sub1_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].vy_sub1_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].sxy_sub2_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].syz_sub2_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].vy_sub2_dev);
            
        }
        if (m->ND==3){// For 3D
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].syy_sub1, &(*mloc)[d].syy_sub1);
            __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].syy_sub2, &(*mloc)[d].syy_sub2);
            
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].syy_sub1_dev);
            __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].syy_sub2_dev);
        }

        // Create the kernels of the devices
        __GUARD gpu_initialize_update_v(&m->context, &(*vcl)[d].program_v, &(*vcl)[d].kernel_v, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], offcomm1, lcomm, 0 );
        __GUARD gpu_initialize_update_s(&m->context, &(*vcl)[d].program_s, &(*vcl)[d].kernel_s, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], offcomm1, lcomm, 0  );
        __GUARD gpu_intialize_seis(&m->context, &(*vcl)[d].program_initseis, &(*vcl)[d].kernel_initseis, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d]);
        __GUARD gpu_intialize_seisout(&m->context, &(*vcl)[d].program_seisout, &(*vcl)[d].kernel_seisout, NULL, &(*vcl)[d], m, &(*mloc)[d]);
        __GUARD gpu_intialize_seisoutinit(&m->context, &(*vcl)[d].program_seisoutinit, &(*vcl)[d].kernel_seisoutinit, NULL, &(*vcl)[d], m, &(*mloc)[d]);
        if (m->freesurf==1){
            __GUARD gpu_initialize_surface(&m->context, &(*vcl)[d].program_surf, &(*vcl)[d].kernel_surf, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d] );
        }
        
        if (d>0 || m->MYLOCALID>0){
            __GUARD gpu_initialize_update_v(&m->context, &(*vcl)[d].program_vcomm1, &(*vcl)[d].kernel_vcomm1, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], 0 , lcomm, 1);
            __GUARD gpu_initialize_update_s(&m->context, &(*vcl)[d].program_scomm1, &(*vcl)[d].kernel_scomm1, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], 0 , lcomm, 1);
        }
        if (d<m->num_devices-1 || m->MYLOCALID<m->NLOCALP-1){
            __GUARD gpu_initialize_update_v(&m->context, &(*vcl)[d].program_vcomm2, &(*vcl)[d].kernel_vcomm2, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], offcomm2, lcomm, 1 );
            __GUARD gpu_initialize_update_s(&m->context, &(*vcl)[d].program_scomm2, &(*vcl)[d].kernel_scomm2, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], offcomm2, lcomm, 1 );
        }
        if (d>0 || m->MYLOCALID>0 || d<m->num_devices-1 || m->MYLOCALID<m->NLOCALP-1){
            __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_fill_transfer_buff1_v_in, &(*vcl)[d], m, &(*mloc)[d],0,1,0);
            __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_fill_transfer_buff2_v_in, &(*vcl)[d], m, &(*mloc)[d],0,2,0);
            __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_fill_transfer_buff1_v_out, &(*vcl)[d], m, &(*mloc)[d],1,1,0);
            __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_fill_transfer_buff2_v_out, &(*vcl)[d], m, &(*mloc)[d],1,2,0);
            __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_fill_transfer_buff1_s_in, &(*vcl)[d], m, &(*mloc)[d],0,1,0);
            __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_fill_transfer_buff2_s_in, &(*vcl)[d], m, &(*mloc)[d],0,2,0);
            __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_fill_transfer_buff1_s_out, &(*vcl)[d], m, &(*mloc)[d],1,1,0);
            __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_fill_transfer_buff2_s_out, &(*vcl)[d], m, &(*mloc)[d],1,2,0);
        }
        
        __GUARD gpu_intialize_sources(&m->context, &(*vcl)[d].program_sources, &(*vcl)[d].kernel_sources, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d]);
        
        
        if (state !=CL_SUCCESS) CLPERR(state);


        //If we want the gradient by the adjoint model method, we create the variables
        if (m->gradout==1 ){
            
            //Residual back propgation memory allocation whith direct field backpropagation
            if (m->back_prop_type==1){
                

                
                //Allocate memory for the varibales that keeps the seismic wavefiled at the boundary. We keep a band of the width of fd order
                if (m->ND!=21){
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_bnd,    &(*vcl)[d].sxxbnd);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_bnd,    &(*vcl)[d].szzbnd);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_bnd,    &(*vcl)[d].sxzbnd);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_bnd,    &(*vcl)[d].vxbnd);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_bnd,    &(*vcl)[d].vzbnd);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].buffer_size_bnd*m->NT, &(*vcl)[d].sxxbnd_pin, &(*mloc)[d].sxxbnd);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].buffer_size_bnd*m->NT, &(*vcl)[d].szzbnd_pin, &(*mloc)[d].szzbnd);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].buffer_size_bnd*m->NT, &(*vcl)[d].sxzbnd_pin, &(*mloc)[d].sxzbnd);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].buffer_size_bnd*m->NT, &(*vcl)[d].vxbnd_pin, &(*mloc)[d].vxbnd);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].buffer_size_bnd*m->NT, &(*vcl)[d].vzbnd_pin, &(*mloc)[d].vzbnd);
                }
                if (m->ND==3 || m->ND==21){// For 3D
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_bnd,    &(*vcl)[d].sxybnd);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_bnd,    &(*vcl)[d].syzbnd);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_bnd,    &(*vcl)[d].vybnd);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].buffer_size_bnd*m->NT, &(*vcl)[d].sxybnd_pin, &(*mloc)[d].sxybnd);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].buffer_size_bnd*m->NT, &(*vcl)[d].syzbnd_pin, &(*mloc)[d].syzbnd);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].buffer_size_bnd*m->NT, &(*vcl)[d].vybnd_pin, &(*mloc)[d].vybnd);
                }
                if (m->ND==3){// For 3D
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_bnd,    &(*vcl)[d].syybnd);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, (*vcl)[d].buffer_size_bnd*m->NT, &(*vcl)[d].syybnd_pin, &(*mloc)[d].syybnd);
                }

                
                
                //Allocate the memory for the variables for the residual wavefield
                if (m->ND!=21){
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].sxx_r);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].szz_r);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].sxz_r);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].vx_r);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].vz_r);
                    if (m->L>0){
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd*m->L,    &(*vcl)[d].rxx_r);
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd*m->L,    &(*vcl)[d].rzz_r);
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd*m->L,    &(*vcl)[d].rxz_r);
                    }
                }
                if (m->ND==3 || m->ND==21){
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].sxy_r);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].syz_r);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].vy_r);
                    if (m->L>0){
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd*m->L,    &(*vcl)[d].rxy_r);
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd*m->L,    &(*vcl)[d].ryz_r);
                    }
                }
                if (m->ND==3){// For 3D
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd,    &(*vcl)[d].syy_r);
                    if (m->L>0){
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_fd*m->L,    &(*vcl)[d].ryy_r);
                    }
                }

                
                //Allocate the gradient output
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].gradrho);
                if (m->ND!=21){
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].gradM);
                }
                __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].gradmu);
                if (m->L>0){
                    if (m->ND!=21){
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].gradtaup);
                    }
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].gradtaus);
                }
                
                
                //Allocate H output
                if (m->Hout==1){
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].Hrho);
                    if (m->ND!=21){
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].HM);
                    }
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].Hmu);
                    if (m->L>0){
                        if (m->ND!=21){
                            __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].Htaup);
                        }
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_model, &(*vcl)[d].Htaus);
                    }
                }
                
                if (state !=CL_SUCCESS) CLPERR(state);
                
                //Create buffers and memory for GPU communication of the residual wavefield
                if (m->ND!=21){
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].sxx_r_sub1, &(*mloc)[d].sxx_r_sub1);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].szz_r_sub1, &(*mloc)[d].szz_r_sub1);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].sxz_r_sub1, &(*mloc)[d].sxz_r_sub1);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].vx_r_sub1, &(*mloc)[d].vx_r_sub1);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].vz_r_sub1, &(*mloc)[d].vz_r_sub1);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].sxx_r_sub2, &(*mloc)[d].sxx_r_sub2);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].szz_r_sub2, &(*mloc)[d].szz_r_sub2);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].sxz_r_sub2, &(*mloc)[d].sxz_r_sub2);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].vx_r_sub2, &(*mloc)[d].vx_r_sub2);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].vz_r_sub2, &(*mloc)[d].vz_r_sub2);
                    
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].sxx_r_sub1_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].szz_r_sub1_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].sxz_r_sub1_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].vx_r_sub1_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].vz_r_sub1_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].sxx_r_sub2_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].szz_r_sub2_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].sxz_r_sub2_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].vx_r_sub2_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].vz_r_sub2_dev);
                }
                if (m->ND==3 || m->ND==21){
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].sxy_r_sub1, &(*mloc)[d].sxy_r_sub1);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].syz_r_sub1, &(*mloc)[d].syz_r_sub1);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].vy_r_sub1, &(*mloc)[d].vy_r_sub1);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].sxy_r_sub2, &(*mloc)[d].sxy_r_sub2);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].syz_r_sub2, &(*mloc)[d].syz_r_sub2);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].vy_r_sub2, &(*mloc)[d].vy_r_sub2);
                    
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].sxy_r_sub1_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].syz_r_sub1_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].vy_r_sub1_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].sxy_r_sub2_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].syz_r_sub2_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].vy_r_sub2_dev);
                    
                }
                if (m->ND==3){// For 3D
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].syy_r_sub1, &(*mloc)[d].syy_r_sub1);
                    __GUARD create_pinned_memory_buffer(&m->context, &(*vcl)[d].cmd_queuecomm, m->buffer_size_comm, &(*vcl)[d].syy_r_sub2, &(*mloc)[d].syy_r_sub2);
                    
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].syy_r_sub1_dev);
                    __GUARD create_gpu_memory_buffer( &m->context, m->buffer_size_comm,    &(*vcl)[d].syy_r_sub2_dev);
                }
                

            }
            
            else if (m->back_prop_type==2){
                
                __GUARD create_gpu_memory_buffer( &m->context, sizeof(float)*m->nfreqs, &(*vcl)[d].gradfreqsn);
                if (m->ND!=21){
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_vx);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_vz);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_sxx);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_szz);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_sxz);
                    if (m->L>0){
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_rxx);
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_rzz);
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_rxz);
                    }
                }

                if (m->ND==3 || m->ND==21){
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_vy);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_sxy);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_syz);
                    if (m->L>0){
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_rxy);
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_ryz);
                    }
                }
                if (m->ND==3){// For 3D
                    __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_syy);
                    if (m->L>0){
                        __GUARD create_gpu_memory_buffer( &m->context, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_ryy);
                    }
                }
                if (state !=CL_SUCCESS) CLPERR(state);
   
            }
            
            if (m->gradsrcout ){
                __GUARD create_gpu_memory_buffer( &m->context, buffer_size_s, &(*vcl)[d].gradsrc);
            }
            
            
            //Create the kernels for the backpropagation and gradient computation
            __GUARD gpu_initialize_update_adjv(&m->context, &(*vcl)[d].program_adjv, &(*vcl)[d].kernel_adjv, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], offcomm1, lcomm, 0  );
            __GUARD gpu_initialize_update_adjs(&m->context, &(*vcl)[d].program_adjs, &(*vcl)[d].kernel_adjs, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], offcomm1, lcomm, 0 );
            
            if (d>0 || m->MYLOCALID>0){
                __GUARD gpu_initialize_update_adjv(&m->context, &(*vcl)[d].program_adjvcomm1, &(*vcl)[d].kernel_adjvcomm1, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], 0 , lcomm, 1);
                __GUARD gpu_initialize_update_adjs(&m->context, &(*vcl)[d].program_adjscomm1, &(*vcl)[d].kernel_adjscomm1, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], 0 , lcomm, 1);
            }
            if (d<m->num_devices-1 || m->MYLOCALID<m->NLOCALP-1 ){
                __GUARD gpu_initialize_update_adjv(&m->context, &(*vcl)[d].program_adjvcomm2, &(*vcl)[d].kernel_adjvcomm2, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], offcomm2, lcomm, 1 );
                __GUARD gpu_initialize_update_adjs(&m->context, &(*vcl)[d].program_adjscomm2, &(*vcl)[d].kernel_adjscomm2, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d], offcomm2, lcomm, 1 );
            }
            if ( (d>0 || m->MYLOCALID>0 || d<m->num_devices-1 || m->MYLOCALID<m->NLOCALP-1) && m->back_prop_type==1 ){
                __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_adj_fill_transfer_buff1_v_in, &(*vcl)[d], m, &(*mloc)[d],0,1,1);
                __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_adj_fill_transfer_buff2_v_in, &(*vcl)[d], m, &(*mloc)[d],0,1,1);
                __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_adj_fill_transfer_buff1_v_out, &(*vcl)[d], m, &(*mloc)[d],1,1,1);
                __GUARD gpu_intialize_fill_transfer_buff_v(&m->context, &(*vcl)[d].program_fill_transfer_buff_v, &(*vcl)[d].kernel_adj_fill_transfer_buff2_v_out, &(*vcl)[d], m, &(*mloc)[d],1,1,1);
                __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_adj_fill_transfer_buff1_s_in, &(*vcl)[d], m, &(*mloc)[d],0,1,1);
                __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_adj_fill_transfer_buff2_s_in, &(*vcl)[d], m, &(*mloc)[d],0,1,1);
                __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_adj_fill_transfer_buff1_s_out, &(*vcl)[d], m, &(*mloc)[d],1,1,1);
                __GUARD gpu_intialize_fill_transfer_buff_s(&m->context, &(*vcl)[d].program_fill_transfer_buff_s, &(*vcl)[d].kernel_adj_fill_transfer_buff2_s_out, &(*vcl)[d], m, &(*mloc)[d],1,1,1);
            }

            __GUARD gpu_intialize_residuals(&m->context, &(*vcl)[d].program_residuals, &(*vcl)[d].kernel_residuals, NULL, &(*vcl)[d], m, &(*mloc)[d]);
            __GUARD gpu_intialize_grad(&m->context, &(*vcl)[d].program_initgrad, &(*vcl)[d].kernel_initgrad, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d]);
            __GUARD gpu_intialize_seis_r(&m->context, &(*vcl)[d].program_initseis_r, &(*vcl)[d].kernel_initseis_r, (*mloc)[d].local_work_size, &(*vcl)[d], m, &(*mloc)[d]);

            
            if (m->back_prop_type==1){
                __GUARD gpu_initialize_savebnd(&m->context, &(*vcl)[d].program_bnd, &(*vcl)[d].kernel_bnd, NULL, &(*vcl)[d], m, &(*mloc)[d]);
            }
            if (m->back_prop_type==2){
                __GUARD gpu_initialize_savefreqs(&m->context, &(*vcl)[d].program_savefreqs, &(*vcl)[d].kernel_savefreqs, NULL, &(*vcl)[d], m, &(*mloc)[d], 0);
                __GUARD gpu_initialize_initsavefreqs(&m->context, &(*vcl)[d].program_initsavefreqs, &(*vcl)[d].kernel_initsavefreqs, NULL, &(*vcl)[d], m, &(*mloc)[d]);
            }
            if (m->gradsrcout==1){
                __GUARD gpu_initialize_gradsrc(&m->context, &(*vcl)[d].program_initialize_gradsrc, &(*vcl)[d].kernel_initialize_gradsrc, NULL, &(*vcl)[d], m, &(*mloc)[d]);
                
            }
            if (state !=CL_SUCCESS) CLPERR(state);
        }

        /*Transfer memory from host to the device*/
        if ((*m).abs_type==1){
            
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].K_x, m->K_x );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].a_x, m->a_x );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].b_x, m->b_x );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].K_x_half, m->K_x_half );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].a_x_half, m->a_x_half );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].b_x_half, m->b_x_half );
            
            if (m->ND==3){// For 3D
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].K_y, m->K_y );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].a_y, m->a_y );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].b_y, m->b_y );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].K_y_half, m->K_y_half );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].a_y_half, m->a_y_half );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].b_y_half, m->b_y_half );
            }
            
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].K_z, m->K_z );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].a_z, m->a_z );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].b_z, m->b_z );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].K_z_half, m->K_z_half );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].a_z_half, m->a_z_half );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, 2*buffer_size_taper,    &(*vcl)[d].b_z_half, m->b_z_half );
            
            
        }
        else if ((*m).abs_type==2){
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, buffer_size_taper,    &(*vcl)[d].taper, m->taper );
        }
        
        if (m->ND!=21){
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].rip,   (*mloc)[d].rip );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].rkp,   (*mloc)[d].rkp );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].uipkp, (*mloc)[d].uipkp );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].u,     (*mloc)[d].u );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].pi,    (*mloc)[d].pi );
            if (m->L>0){
                __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].taup, (*mloc)[d].taup );
                __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].taus, (*mloc)[d].taus );
                __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].tausipkp, (*mloc)[d].tausipkp );
            }
        }
        
        if (m->ND==3 || m->ND==21){
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].rjp,   (*mloc)[d].rjp );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].uipjp, (*mloc)[d].uipjp );
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].ujpkp, (*mloc)[d].ujpkp );
            if (m->L>0){
                    __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].tausipjp, (*mloc)[d].tausipjp );
                    __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].tausjpkp, (*mloc)[d].tausjpkp );
            }
        }
        
        if (m->L>0){
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, sizeof(float)*m->L,        &(*vcl)[d].eta,  m->eta );
        }
        
        if (m->gradout==1 && m->back_prop_type==2 ){
            __GUARD transfer_gpu_memory( &(*vcl)[d].cmd_queue, sizeof(float)*m->nfreqs, &(*vcl)[d].gradfreqsn,   m->gradfreqsn );
        }
        
        if (state !=CL_SUCCESS) CLPERR(state);
        
    }

    if (state && m->MPI_INIT==1) MPI_Bcast( &state, 1, MPI_INT, m->MYID, MPI_COMM_WORLD );
    
    return state;


}
