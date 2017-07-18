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


int comm(struct modcsts * m, struct varcl ** vcl, int adj, int upind){
    /* Communication for domain decompositon for MPI (between processes)
       and OpenCL (between devices) */
    
    int state = 0;
    int d,i, lastd;

    
    
    //Read buffers for comunnication between devices
    for (d=0;d<m->NUM_DEVICES;d++){
        
        //TODO event dependance for adj
        if (d>0){
            if (adj && m->BACK_PROP_TYPE==1){
                for (i=0;i<m->nvars;i++){
                    if (m->vars_adj[i].to_comm-1==upind){
                        __GUARD clbuf_readpin(&(*vcl)[d-1].queuecomm,
                                              &(*vcl)[d-1].vars_adj[i].cl_buf2,
                                              &(*vcl)[d  ].vars_adj[i].cl_buf1_pin);
                    }
                }
            }
            for (i=0;i<m->nvars;i++){
                if (m->vars[i].to_comm-1==upind)
                    __GUARD clbuf_readpin(&(*vcl)[d-1].queuecomm,
                                          &(*vcl)[d-1].vars[i].cl_buf2,
                                          &(*vcl)[d  ].vars[i].cl_buf1_pin);
            }
            if (adj){
                __GUARD clReleaseEvent((*vcl)[d-1].ups_adj[i].com2.event);
            }
            else{
                __GUARD clReleaseEvent((*vcl)[d-1].ups_f[i].com2.event);
            }
        }

        if (d<m->NUM_DEVICES-1){
            
            if (adj && m->BACK_PROP_TYPE==1){
                for (i=0;i<m->nvars;i++){
                    if (m->vars_adj[i].to_comm-1==upind)
                        __GUARD clbuf_readpin(&(*vcl)[d+1].queuecomm,
                                              &(*vcl)[d+1].vars_adj[i].cl_buf1,
                                              &(*vcl)[d  ].vars_adj[i].cl_buf2_pin);
                }
            }
            for (i=0;i<m->nvars;i++){
                if (m->vars[i].to_comm-1==upind)
                    __GUARD clbuf_readpin(&(*vcl)[d+1].queuecomm,
                                          &(*vcl)[d+1].vars[i].cl_buf1,
                                          &(*vcl)[d  ].vars[i].cl_buf2_pin);
            }
            if (adj){
                __GUARD clReleaseEvent((*vcl)[d-1].ups_adj[i].com1.event);
            }
            else{
                __GUARD clReleaseEvent((*vcl)[d-1].ups_f[i].com1.event);
            }
            
        }

    }
    
    //Read buffers for comunnication between MPI processes sharing this shot
    if (m->MYLOCALID>0){
        if (adj && m->BACK_PROP_TYPE==1){
            for (i=0;i<m->nvars;i++){
                if (m->vars_adj[i].to_comm-1==upind){
                    __GUARD clbuf_readpin(&(*vcl)[0].queuecomm,
                                          &(*vcl)[0].vars_adj[i].cl_buf1,
                                          &(*vcl)[0].vars_adj[i].cl_buf1_pin);
                }
            }
        }
        for (i=0;i<m->nvars;i++){
            if (m->vars[i].to_comm-1==upind)
                __GUARD clbuf_readpin(&(*vcl)[0].queuecomm,
                                      &(*vcl)[0].vars[i].cl_buf1,
                                      &(*vcl)[0].vars[i].cl_buf1_pin);
        }
        if (adj){
            __GUARD clReleaseEvent((*vcl)[0].ups_adj[i].com1.event);
        }
        else{
            __GUARD clReleaseEvent((*vcl)[0].ups_f[i].com1.event);
        }
    }
    if (m->MYLOCALID<m->NLOCALP-1){
        lastd=m->NUM_DEVICES-1;
        if (adj && m->BACK_PROP_TYPE==1){
            for (i=0;i<m->nvars;i++){
                if (m->vars_adj[i].to_comm-1==upind){
                    __GUARD clbuf_readpin(&(*vcl)[lastd].queuecomm,
                                          &(*vcl)[lastd].vars_adj[i].cl_buf2,
                                          &(*vcl)[lastd].vars_adj[i].cl_buf2_pin);
                }
            }
        }
        for (i=0;i<m->nvars;i++){
            if (m->vars[i].to_comm-1==upind)
                __GUARD clbuf_readpin(&(*vcl)[lastd].queuecomm,
                                      &(*vcl)[lastd].vars[i].cl_buf2,
                                      &(*vcl)[lastd].vars[i].cl_buf2_pin);
        }
        if (adj){
            __GUARD clReleaseEvent((*vcl)[lastd].ups_adj[i].com2.event);
        }
        else{
            __GUARD clReleaseEvent((*vcl)[lastd].ups_f[i].com2.event);
        }
    }



int comm1_v_MPI(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep){
    int state=0;
    if (m->ND==21){
        __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[0]);
        __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[0]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vy_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_r_sub1, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vy_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_sub1, 0, NULL, &(*vcl)[0].event_writev1);
    }
    else{
        __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[0]);
        __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[0]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vx_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vx_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vx_r_sub1, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vx_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vx_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vx_sub1, 0, NULL, NULL);
        
        if (m->ND==3){// For 3D
            __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[1]);
            __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[1]);
            if (bstep && m->back_prop_type==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vy_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 2, m->MYID-1, 2, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_r_sub1, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vy_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 2, m->MYID-1, 2, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_sub1, 0, NULL, NULL);
        }
        
        __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[2]);
        __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[2]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vz_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 3, m->MYID-1, 3, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vz_r_sub1, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vz_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 3, m->MYID-1, 3, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vz_sub1, 0, NULL, &(*vcl)[0].event_writev1);
        
    }
    
    return state;
    
}

int comm2_v_MPI(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep){
    int state=0;
    
    if (m->ND==21){
        __GUARD clWaitForEvents(	1, &(*vcl)[m->num_devices-1].event_readMPI2[0]);
        __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_readMPI2[0]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].vy_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vy_r_sub2, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].vy_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vy_sub2, 0, NULL, &(*vcl)[m->num_devices-1].event_writev2);
    }
    else{
        __GUARD clWaitForEvents(	1, &(*vcl)[m->num_devices-1].event_readMPI2[0]);
        __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_readMPI2[0]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].vx_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vx_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vx_r_sub2, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].vx_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vx_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vx_sub2, 0, NULL, NULL);
        
        if (m->ND==3){// For 3D
            __GUARD clWaitForEvents(	1, &(*vcl)[m->num_devices-1].event_readMPI2[1]);
            __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_readMPI2[1]);
            if (bstep && m->back_prop_type==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].vy_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 2, m->MYID+1, 2, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vy_r_sub2, 0, NULL, NULL);
                
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].vy_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 2, m->MYID+1, 2, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vy_sub2, 0, NULL, NULL);
        }
        
        __GUARD clWaitForEvents(	1, &(*vcl)[m->num_devices-1].event_readMPI2[2]);
        __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_readMPI2[2]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].vz_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 3, m->MYID+1, 3, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vz_r_sub2, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].vz_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 3, m->MYID+1, 3, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vz_sub2, 0, NULL, &(*vcl)[m->num_devices-1].event_writev2);
        
    }
    
    return state;
    
}

int comm1_s_MPI(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep){
    int state=0;
    
    if (m->ND==21){
        
        __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[0]);
        __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[0]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxy_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_r_sub1, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxy_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_sub1, 0, NULL, NULL);
        
        __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[1]);
        __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[1]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].syz_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 6, m->MYID-1, 6, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].syz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_r_sub1, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].syz_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 6, m->MYID-1, 6, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].syz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_sub1, 0, NULL, &(*vcl)[0].event_writes1);
    }
    else{
        __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[0]);
        __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[0]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxx_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxx_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxx_r_sub1, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxx_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxx_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxx_sub1, 0, NULL, NULL);
        
        if (m->ND==3){// For 3D
            __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[1]);
            __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[1]);
            if (bstep && m->back_prop_type==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].syy_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 2, m->MYID-1, 2, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].syy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syy_r_sub1, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].syy_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 2, m->MYID-1, 2, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].syy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syy_sub1, 0, NULL, NULL);
            
            __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[2]);
            __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[2]);
            if (bstep && m->back_prop_type==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxy_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 3, m->MYID-1, 3, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_r_sub1, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxy_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 3, m->MYID-1, 3, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_sub1, 0, NULL, NULL);
            
            __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[3]);
            __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[3]);
            if (bstep && m->back_prop_type==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].syz_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 4, m->MYID-1, 4, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].syz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_r_sub1, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].syz_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 4, m->MYID-1, 4, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].syz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_sub1, 0, NULL, NULL);
            
        }
        
        __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[4]);
        __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[4]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].szz_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 5, m->MYID-1, 5, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].szz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].szz_r_sub1, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].szz_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 5, m->MYID-1, 5, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].szz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].szz_sub1, 0, NULL, NULL);
        
        __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[5]);
        __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[5]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxz_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 6, m->MYID-1, 6, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxz_r_sub1, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxz_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 6, m->MYID-1, 6, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxz_sub1, 0, NULL, &(*vcl)[0].event_writes1);
    }
    
    return state;
    
}

int comm2_s_MPI(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep){
    int state=0;
    
    if (m->ND==21){
        __GUARD clWaitForEvents(	1, &(*vcl)[m->num_devices-1].event_readMPI2[0]);
        __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_readMPI2[0]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].sxy_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxy_r_sub2, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].sxy_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxy_sub2, 0, NULL, NULL);
        
        
        __GUARD clWaitForEvents(	1, &(*vcl)[m->num_devices-1].event_readMPI2[1]);
        __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_readMPI2[1]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].syz_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 6, m->MYID+1, 6, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].syz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].syz_r_sub2, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].syz_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 6, m->MYID+1, 6, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].syz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].syz_sub2, 0, NULL, &(*vcl)[m->num_devices-1].event_writes2);
        
    }
    else{
        __GUARD clWaitForEvents(	1, &(*vcl)[m->num_devices-1].event_readMPI2[0]);
        __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_readMPI2[0]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].sxx_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxx_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxx_r_sub2, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].sxx_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxx_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxx_sub2, 0, NULL, NULL);
        
        if (m->ND==3){// For 3D
            __GUARD clWaitForEvents(	1, &(*vcl)[m->num_devices-1].event_readMPI2[1]);
            __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_readMPI2[1]);
            if (bstep && m->back_prop_type==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].syy_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 2, m->MYID+1, 2, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].syy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].syy_r_sub2, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].syy_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 2, m->MYID+1, 2, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].syy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].syy_sub2, 0, NULL, NULL);
            
            __GUARD clWaitForEvents(	1, &(*vcl)[m->num_devices-1].event_readMPI2[2]);
            __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_readMPI2[2]);
            if (bstep && m->back_prop_type==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].sxy_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 3, m->MYID+1, 3, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxy_r_sub2, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].sxy_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 3, m->MYID+1, 3, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxy_sub2, 0, NULL, NULL);
            
            __GUARD clWaitForEvents(	1, &(*vcl)[m->num_devices-1].event_readMPI2[3]);
            __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_readMPI2[3]);
            if (bstep && m->back_prop_type==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].syz_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 4, m->MYID+1, 4, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].syz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].syz_r_sub2, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].syz_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 4, m->MYID+1, 4, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].syz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].syz_sub2, 0, NULL, NULL);
        }
        
        __GUARD clWaitForEvents(	1, &(*vcl)[m->num_devices-1].event_readMPI2[4]);
        __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_readMPI2[4]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].szz_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 5, m->MYID+1, 5, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].szz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].szz_r_sub2, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].szz_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 5, m->MYID+1, 5, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].szz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].szz_sub2, 0, NULL, NULL);
        
        __GUARD clWaitForEvents(	1, &(*vcl)[m->num_devices-1].event_readMPI2[5]);
        __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_readMPI2[5]);
        if (bstep && m->back_prop_type==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].sxz_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 6, m->MYID+1, 6, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxz_r_sub2, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->num_devices-1].sxz_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 6, m->MYID+1, 6, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxz_sub2, 0, NULL, &(*vcl)[m->num_devices-1].event_writes2);
    }
    
    return state;
    
}

int comm_v(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep){
    /*Communication for domain decompositon for MPI (between processing elements) and OpenCL (between devices) */
    int state = 0;
    int d;
    
    
    //Read buffers for comunnication between devices
    for (d=0;d<m->num_devices;d++){
        
        
        if (d>0){
            if (m->ND==21){
                
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].vy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub1, 1, &(*vcl)[d-1].event_updatev_comm2, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].vy_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub1, 0, NULL, &(*vcl)[d].event_readv1);
                }
                else{
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].vy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub1, 1, &(*vcl)[d-1].event_updatev_comm2, &(*vcl)[d].event_readv1);
                }
                __GUARD clReleaseEvent((*vcl)[d-1].event_updatev_comm2);
            }
            else{
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].vx_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vx_sub1, 1, &(*vcl)[d-1].event_updatev_comm2, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].vy_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub1, 0, NULL, NULL);
                }
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].vx_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vx_r_sub1, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].vy_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub1, 0, NULL, NULL);
                    }
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].vz_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vz_r_sub1, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].vz_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vz_sub1, 0, NULL, &(*vcl)[d].event_readv1);
                __GUARD clReleaseEvent((*vcl)[d-1].event_updatev_comm2);
            }
        }
        
        if (d<m->num_devices-1){
            if (m->ND==21){
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].vy_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub2, 1, &(*vcl)[d+1].event_updatev_comm1, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].vy_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub2, 0, NULL, &(*vcl)[d].event_readv2);
                }
                else{
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].vy_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub2, 1, &(*vcl)[d+1].event_updatev_comm1, &(*vcl)[d].event_readv2);
                }
                __GUARD clReleaseEvent((*vcl)[d+1].event_updatev_comm1);
            }
            else{
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].vx_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vx_sub2, 1, &(*vcl)[d+1].event_updatev_comm1, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].vy_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub2, 0, NULL, NULL);
                }
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].vx_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vx_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].vz_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vz_r_sub2, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].vy_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub2, 0, NULL, NULL);
                    }
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].vz_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vz_sub2, 0, NULL, &(*vcl)[d].event_readv2);
                __GUARD clReleaseEvent((*vcl)[d+1].event_updatev_comm1);
            }
        }
        
    }
    
    //Read buffers for comunnication between MPI processes sharing this shot
    if (m->MYLOCALID>0){
        
        if (m->ND==21){
            if (bstep && m->back_prop_type==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_r_sub1, 1, &(*vcl)[0].event_updatev_comm1, NULL);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_sub1, 1, &(*vcl)[0].event_updatev_comm1, &(*vcl)[0].event_readMPI1[0]);
            __GUARD clReleaseEvent((*vcl)[0].event_updatev_comm1);
        }
        else{
            if (bstep && m->back_prop_type==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vx_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vx_r_sub1, 1, &(*vcl)[0].event_updatev_comm1, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_r_sub1, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vz_r_sub1, 0, NULL, NULL);
            }
            
            __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vx_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vx_sub1, 1, &(*vcl)[0].event_updatev_comm1, &(*vcl)[0].event_readMPI1[0]);
            if (m->ND==3){// For 3D
                __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[1]);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].vz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vz_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[2]);
            __GUARD clReleaseEvent((*vcl)[0].event_updatev_comm1);
        }
        
    }
    if (m->MYLOCALID<m->NLOCALP-1){
        if (m->ND==21){
            if (bstep && m->back_prop_type==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vy_r_sub2, 1, &(*vcl)[m->num_devices-1].event_updatev_comm2, NULL);
            }
            
            __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vy_sub2, 1, &(*vcl)[m->num_devices-1].event_updatev_comm2, &(*vcl)[m->num_devices-1].event_readMPI2[0]);
            __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_updatev_comm2);
        }
        else{
            if (bstep && m->back_prop_type==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vx_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vx_r_sub2, 1, &(*vcl)[m->num_devices-1].event_updatev_comm2, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vy_r_sub2, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vz_r_sub2, 0, NULL, NULL);
            }
            
            __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vx_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vx_sub2, 1, &(*vcl)[m->num_devices-1].event_updatev_comm2, &(*vcl)[m->num_devices-1].event_readMPI2[0]);
            if (m->ND==3){// For 3D
                __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vy_sub2, 0, NULL, &(*vcl)[m->num_devices-1].event_readMPI2[1]);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].vz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].vz_sub2, 0, NULL, &(*vcl)[m->num_devices-1].event_readMPI2[2]);
            __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_updatev_comm2);
        }
    }
    
    for (d=0;d<m->num_devices;d++){
        clFlush((*vcl)[d].cmd_queue);
        clFlush((*vcl)[d].cmd_queuecomm);
    }
    
    //Write buffers for comunnication between devices
    for (d=0;d<m->num_devices;d++){
        if (d>0){
            if (m->ND==21){
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vy_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub1, 1, &(*vcl)[d].event_readv1, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vy_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub1, 0, NULL, &(*vcl)[d].event_writev1);
                }
                else{
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vy_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub1, 1, &(*vcl)[d].event_readv1, &(*vcl)[d].event_writev1);
                }
                __GUARD clReleaseEvent((*vcl)[d].event_readv1);
            }
            else{
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vx_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vx_sub1, 1, &(*vcl)[d].event_readv1, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vy_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub1, 0, NULL, NULL);
                }
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vx_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vx_r_sub1, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vy_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub1, 0, NULL, NULL);
                    }
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vz_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vz_r_sub1, 0, NULL, NULL);
                }
                
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vz_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vz_sub1, 0, NULL, &(*vcl)[d].event_writev1);
                __GUARD clReleaseEvent((*vcl)[d].event_readv1);
            }
        }
        
        if (d<m->num_devices-1){
            if (m->ND==21){
                
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vy_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub2, 1, &(*vcl)[d].event_readv2, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vy_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub2, 0, NULL, &(*vcl)[d].event_writev2);
                }
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vy_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub2, 1, &(*vcl)[d].event_readv2, &(*vcl)[d].event_writev2);
                __GUARD clReleaseEvent((*vcl)[d].event_readv2);
            }
            else{
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vx_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vx_sub2, 1, &(*vcl)[d].event_readv2, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vy_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub2, 0, NULL, NULL);
                }
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vx_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vx_r_sub2, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vy_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub2, 0, NULL, NULL);
                    }
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vz_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vz_r_sub2, 0, NULL, NULL);
                }
                
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].vz_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vz_sub2, 0, NULL, &(*vcl)[d].event_writev2);
                __GUARD clReleaseEvent((*vcl)[d].event_readv2);
            }
        }
        
    }
    
    for (d=0;d<m->num_devices;d++){
        clFlush((*vcl)[d].cmd_queuecomm);
    }
    
    //Wait for Opencl buffers to be read, send MPI bufers and write to devices
    //Processess with even ID in the group send and receive buffers 1 first, and then buffers 2, vice versa for odd IDs
    if (m->MYLOCALID % 2 != 0 && m->MYLOCALID>0){
        __GUARD comm1_v_MPI(m, vcl, mloc, bstep);
    }
    
    if (m->MYLOCALID % 2 == 0 && m->MYLOCALID<m->NLOCALP-1){
        __GUARD comm2_v_MPI(m, vcl, mloc, bstep);
    }
    
    if (m->MYLOCALID % 2 == 0 && m->MYLOCALID>0){
        __GUARD comm1_v_MPI(m, vcl, mloc, bstep);
    }
    
    if (m->MYLOCALID % 2 != 0 && m->MYLOCALID<m->NLOCALP-1){
        __GUARD comm2_v_MPI(m, vcl, mloc, bstep);
    }
    
    return state;
    
}




int comm_s(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep){
    
    int state = 0;
    int d;
    
    
    //Read buffers for comunnication between devices
    for (d=0;d<m->num_devices;d++){
        
        
        if (d>0){
            if (m->ND==21){
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].sxy_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub1, 1, &(*vcl)[d-1].event_updates_comm2, NULL);
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].sxy_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].syz_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub1, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].syz_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub1, 0, NULL, &(*vcl)[d].event_reads1);
                __GUARD clReleaseEvent((*vcl)[d-1].event_updates_comm2);
            }
            else{
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].sxx_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_sub1, 1, &(*vcl)[d-1].event_updates_comm2, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].syy_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syy_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].sxy_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].syz_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub1, 0, NULL, NULL);
                }
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].sxx_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_r_sub1, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].syy_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syy_r_sub1, 0, NULL, NULL);
                        __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].sxy_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub1, 0, NULL, NULL);
                        __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].syz_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub1, 0, NULL, NULL);
                        
                    }
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].szz_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].szz_r_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].sxz_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_r_sub1, 0, NULL, NULL);
                    
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].szz_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].szz_sub1, 0, NULL, NULL);
                
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].cmd_queuecomm, (*vcl)[d-1].sxz_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_sub1, 0, NULL, &(*vcl)[d].event_reads1);
                __GUARD clReleaseEvent((*vcl)[d-1].event_updates_comm2);
            }
            
        }
        
        if (d<m->num_devices-1){
            if (m->ND==21){
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].sxy_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub2, 1, &(*vcl)[d+1].event_updates_comm1, NULL);
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].sxy_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].syz_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub2, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].syz_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub2, 0, NULL, &(*vcl)[d].event_reads2);
                __GUARD clReleaseEvent((*vcl)[d+1].event_updates_comm1);
            }
            else{
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].sxx_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_sub2, 1, &(*vcl)[d+1].event_updates_comm1, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].syy_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syy_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].sxy_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].syz_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub2, 0, NULL, NULL);
                }
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].sxx_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_r_sub2, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].syy_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syy_r_sub2, 0, NULL, NULL);
                        __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].sxy_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub2, 0, NULL, NULL);
                        __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].syz_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub2, 0, NULL, NULL);
                    }
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].szz_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].szz_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].sxz_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_r_sub2, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].szz_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].szz_sub2, 0, NULL, NULL);
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].cmd_queuecomm, (*vcl)[d+1].sxz_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_sub2, 0, NULL, &(*vcl)[d].event_reads2);
                __GUARD clReleaseEvent((*vcl)[d+1].event_updates_comm1);
            }
        }
        
    }
    
    
    //Read buffers for comunnication between MPI processes sharing this shot
    if (m->MYLOCALID>0){
        if (m->ND==21){
            if (bstep && m->back_prop_type==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_r_sub1, 1, &(*vcl)[0].event_updates_comm1, NULL);
                __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].syz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_r_sub1, 0, NULL, NULL);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_sub1, 1, &(*vcl)[0].event_updates_comm1, &(*vcl)[0].event_readMPI1[0]);
            __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].syz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[1]);
            __GUARD clReleaseEvent((*vcl)[0].event_updates_comm1);
        }
        else{
            if (bstep && m->back_prop_type==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxx_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxx_r_sub1, 1, &(*vcl)[0].event_updates_comm1, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].syy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syy_r_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_r_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].syz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_r_sub1, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].szz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].szz_r_sub1, 0, NULL, NULL);
                __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxz_r_sub1, 0, NULL, NULL);
            }
            
            __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxx_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxx_sub1, 1, &(*vcl)[0].event_updates_comm1, &(*vcl)[0].event_readMPI1[0]);
            if (m->ND==3){// For 3D
                __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].syy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syy_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[1]);
                __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[2]);
                __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].syz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[3]);
            }
            
            __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].szz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].szz_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[4]);
            __GUARD clEnqueueReadBuffer( (*vcl)[0].cmd_queuecomm, (*vcl)[0].sxz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxz_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[5]);
            __GUARD clReleaseEvent((*vcl)[0].event_updates_comm1);
        }
    }
    if (m->MYLOCALID<m->NLOCALP-1){
        if (m->ND==21){
            if (bstep && m->back_prop_type==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxy_r_sub2, 1, &(*vcl)[m->num_devices-1].event_updates_comm2, NULL);
                __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].syz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].syz_r_sub2, 0, NULL, NULL);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxy_sub2, 1, &(*vcl)[m->num_devices-1].event_updates_comm2, &(*vcl)[m->num_devices-1].event_readMPI2[0]);
            __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].syz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].syz_sub2, 0, NULL, &(*vcl)[m->num_devices-1].event_readMPI2[1]);
            __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_updates_comm2);
        }
        else{
            if (bstep && m->back_prop_type==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxx_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxx_r_sub2, 1, &(*vcl)[m->num_devices-1].event_updates_comm2, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].syy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].syy_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxy_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].syz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].syz_r_sub2, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].szz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].szz_r_sub2, 0, NULL, NULL);
                __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxz_r_sub2, 0, NULL, NULL);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxx_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxx_sub2, 1, &(*vcl)[m->num_devices-1].event_updates_comm2, &(*vcl)[m->num_devices-1].event_readMPI2[0]);
            
            if (m->ND==3){// For 3D
                __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].syy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].syy_sub2, 0, NULL, &(*vcl)[m->num_devices-1].event_readMPI2[1]);
                __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxy_sub2, 0, NULL, &(*vcl)[m->num_devices-1].event_readMPI2[2]);
                __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].syz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].syz_sub2, 0, NULL, &(*vcl)[m->num_devices-1].event_readMPI2[3]);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].szz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].szz_sub2, 0, NULL, &(*vcl)[m->num_devices-1].event_readMPI2[4]);
            __GUARD clEnqueueReadBuffer( (*vcl)[m->num_devices-1].cmd_queuecomm, (*vcl)[m->num_devices-1].sxz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->num_devices-1].sxz_sub2, 0, NULL, &(*vcl)[m->num_devices-1].event_readMPI2[5]);
            __GUARD clReleaseEvent((*vcl)[m->num_devices-1].event_updates_comm2);
        }
    }
    
    for (d=0;d<m->num_devices;d++){
        clFlush((*vcl)[d].cmd_queue);
        clFlush((*vcl)[d].cmd_queuecomm);
    }
    
    //Write buffers for comunnication between devices
    for (d=0;d<m->num_devices;d++){
        if (d>0){
            if (m->ND==21){
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxy_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub1, 1, &(*vcl)[d].event_reads1, NULL);
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxy_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].syz_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub1, 0, NULL, NULL);
                }
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].syz_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub1, 0, NULL, &(*vcl)[d].event_writes1);
                __GUARD clReleaseEvent((*vcl)[d].event_reads1);
            }
            else{
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxx_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_sub1, 1, &(*vcl)[d].event_reads1, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].syy_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syy_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxy_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].syz_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub1, 0, NULL, NULL);
                }
                
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxx_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_r_sub1, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].syy_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syy_r_sub1, 0, NULL, NULL);
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxy_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub1, 0, NULL, NULL);
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].syz_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub1, 0, NULL, NULL);
                        
                    }
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].szz_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].szz_r_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxz_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_r_sub1, 0, NULL, NULL);
                }
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].szz_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].szz_sub1, 0, NULL, NULL);
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxz_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_sub1, 0, NULL, &(*vcl)[d].event_writes1);
                __GUARD clReleaseEvent((*vcl)[d].event_reads1);
            }
        }
        
        if (d<m->num_devices-1){
            if (m->ND==21){
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxy_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub2, 1, &(*vcl)[d].event_reads2, NULL);
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxy_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].syz_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub2, 0, NULL, NULL);
                }
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].syz_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub2, 0, NULL, &(*vcl)[d].event_writes2);
                __GUARD clReleaseEvent((*vcl)[d].event_reads2);
            }
            else{
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxx_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_sub2, 1, &(*vcl)[d].event_reads2, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].syy_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syy_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxy_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].syz_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub2, 0, NULL, NULL);
                }
                if (bstep && m->back_prop_type==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxx_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_r_sub2, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].syy_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syy_r_sub2, 0, NULL, NULL);
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxy_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub2, 0, NULL, NULL);
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].syz_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub2, 0, NULL, NULL);
                        
                    }
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].szz_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].szz_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxz_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_r_sub2, 0, NULL, NULL);
                }
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].szz_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].szz_sub2, 0, NULL, NULL);
                __GUARD clEnqueueWriteBuffer((*vcl)[d].cmd_queuecomm,   (*vcl)[d].sxz_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_sub2, 0, NULL, &(*vcl)[d].event_writes2);
                __GUARD clReleaseEvent((*vcl)[d].event_reads2);
            }
        }
        
    }
    
    for (d=0;d<m->num_devices;d++){
        clFlush((*vcl)[d].cmd_queuecomm);
    }
    
    //Wait for Opencl buffers to be read, send MPI bufers and write to devices
    //Processess with even ID in the group send and receive buffers 1 first, and then buffers 2, vice versa for odd IDs
    if (m->MYLOCALID % 2 != 0 && m->MYLOCALID>0){
        __GUARD comm1_s_MPI(m, vcl, mloc, bstep);
    }
    
    if (m->MYLOCALID % 2 == 0 && m->MYLOCALID<m->NLOCALP-1){
        __GUARD comm2_s_MPI(m, vcl, mloc, bstep);
    }
    
    if (m->MYLOCALID % 2 == 0 && m->MYLOCALID>0){
        __GUARD comm1_s_MPI(m, vcl, mloc, bstep);
    }
    
    if (m->MYLOCALID % 2 != 0 && m->MYLOCALID<m->NLOCALP-1){
        __GUARD comm2_s_MPI(m, vcl, mloc, bstep);
    }
    
    
    return state;
    
}



