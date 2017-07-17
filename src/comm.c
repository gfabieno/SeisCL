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

int com_v_MPI(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep){
    int state=0;
    if (m->ND==21){
        __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[0]);
        __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[0]);
        if (bstep && m->BACK_PROP_TYPE==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vy_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_r_sub1, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vy_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_sub1, 0, NULL, &(*vcl)[0].event_writev1);
    }
    else{
        __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[0]);
        __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[0]);
        if (bstep && m->BACK_PROP_TYPE==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vx_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vx_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vx_r_sub1, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vx_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vx_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vx_sub1, 0, NULL, NULL);
        
        if (m->ND==3){// For 3D
            __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[1]);
            __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[1]);
            if (bstep && m->BACK_PROP_TYPE==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vy_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 2, m->MYID-1, 2, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_r_sub1, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vy_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 2, m->MYID-1, 2, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_sub1, 0, NULL, NULL);
        }
        
        __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[2]);
        __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[2]);
        if (bstep && m->BACK_PROP_TYPE==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vz_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 3, m->MYID-1, 3, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vz_r_sub1, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].vz_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 3, m->MYID-1, 3, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vz_sub1, 0, NULL, &(*vcl)[0].event_writev1);
        
    }
    
    return state;
    
}

int com_v_MPI(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep){
    int state=0;
    
    if (m->ND==21){
        __GUARD clWaitForEvents(	1, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[0]);
        __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_readMPI2[0]);
        if (bstep && m->BACK_PROP_TYPE==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].vy_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vy_r_sub2, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].vy_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vy_sub2, 0, NULL, &(*vcl)[m->NUM_DEVICES-1].event_writev2);
    }
    else{
        __GUARD clWaitForEvents(	1, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[0]);
        __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_readMPI2[0]);
        if (bstep && m->BACK_PROP_TYPE==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].vx_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vx_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vx_r_sub2, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].vx_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vx_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vx_sub2, 0, NULL, NULL);
        
        if (m->ND==3){// For 3D
            __GUARD clWaitForEvents(	1, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[1]);
            __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_readMPI2[1]);
            if (bstep && m->BACK_PROP_TYPE==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].vy_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 2, m->MYID+1, 2, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vy_r_sub2, 0, NULL, NULL);
                
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].vy_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 2, m->MYID+1, 2, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vy_sub2, 0, NULL, NULL);
        }
        
        __GUARD clWaitForEvents(	1, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[2]);
        __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_readMPI2[2]);
        if (bstep && m->BACK_PROP_TYPE==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].vz_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 3, m->MYID+1, 3, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vz_r_sub2, 0, NULL, NULL);
        }
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].vz_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 3, m->MYID+1, 3, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vz_sub2, 0, NULL, &(*vcl)[m->NUM_DEVICES-1].event_writev2);
        
    }
    
    return state;
    
}

int com_s_MPI(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep){
    int state=0;
    
if (m->ND==21){
    
    __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[0]);
    __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[0]);
    if (bstep && m->BACK_PROP_TYPE==1){
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxy_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_r_sub1, 0, NULL, NULL);
    }
    if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxy_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_sub1, 0, NULL, NULL);
        
        __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[1]);
        __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[1]);
        if (bstep && m->BACK_PROP_TYPE==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].syz_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 6, m->MYID-1, 6, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].syz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_r_sub1, 0, NULL, NULL);
        }
    if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].syz_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 6, m->MYID-1, 6, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].syz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_sub1, 0, NULL, &(*vcl)[0].event_writes1);
        }
else{
    __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[0]);
    __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[0]);
    if (bstep && m->BACK_PROP_TYPE==1){
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxx_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxx_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxx_r_sub1, 0, NULL, NULL);
    }
    if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxx_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 1, m->MYID-1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxx_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxx_sub1, 0, NULL, NULL);
        
        if (m->ND==3){// For 3D
            __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[1]);
            __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[1]);
            if (bstep && m->BACK_PROP_TYPE==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].syy_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 2, m->MYID-1, 2, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].syy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syy_r_sub1, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].syy_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 2, m->MYID-1, 2, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].syy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syy_sub1, 0, NULL, NULL);
            
            __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[2]);
            __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[2]);
            if (bstep && m->BACK_PROP_TYPE==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxy_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 3, m->MYID-1, 3, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_r_sub1, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxy_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 3, m->MYID-1, 3, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_sub1, 0, NULL, NULL);
            
            __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[3]);
            __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[3]);
            if (bstep && m->BACK_PROP_TYPE==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].syz_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 4, m->MYID-1, 4, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].syz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_r_sub1, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].syz_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 4, m->MYID-1, 4, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].syz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_sub1, 0, NULL, NULL);
            
        }
    
    __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[4]);
    __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[4]);
    if (bstep && m->BACK_PROP_TYPE==1){
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].szz_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 5, m->MYID-1, 5, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].szz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].szz_r_sub1, 0, NULL, NULL);
    }
    if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].szz_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 5, m->MYID-1, 5, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].szz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].szz_sub1, 0, NULL, NULL);
        
        __GUARD clWaitForEvents(	1, &(*vcl)[0].event_readMPI1[5]);
        __GUARD clReleaseEvent((*vcl)[0].event_readMPI1[5]);
        if (bstep && m->BACK_PROP_TYPE==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxz_r_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 6, m->MYID-1, 6, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxz_r_sub1, 0, NULL, NULL);
        }
    if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[0].sxz_sub1, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID-1, 6, m->MYID-1, 6, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxz_sub1, 0, NULL, &(*vcl)[0].event_writes1);
        }

    return state;
    
}

int com_s_MPI(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep){
    int state=0;
    
if (m->ND==21){
    __GUARD clWaitForEvents(	1, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[0]);
    __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_readMPI2[0]);
    if (bstep && m->BACK_PROP_TYPE==1){
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].sxy_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxy_r_sub2, 0, NULL, NULL);
    }
    if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].sxy_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxy_sub2, 0, NULL, NULL);
        
        
        __GUARD clWaitForEvents(	1, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[1]);
        __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_readMPI2[1]);
        if (bstep && m->BACK_PROP_TYPE==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].syz_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 6, m->MYID+1, 6, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].syz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].syz_r_sub2, 0, NULL, NULL);
        }
    if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].syz_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 6, m->MYID+1, 6, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].syz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].syz_sub2, 0, NULL, &(*vcl)[m->NUM_DEVICES-1].event_writes2);
        
        }
else{
    __GUARD clWaitForEvents(	1, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[0]);
    __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_readMPI2[0]);
    if (bstep && m->BACK_PROP_TYPE==1){
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].sxx_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxx_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxx_r_sub2, 0, NULL, NULL);
    }
    if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].sxx_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 1, m->MYID+1, 1, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxx_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxx_sub2, 0, NULL, NULL);
        
        if (m->ND==3){// For 3D
            __GUARD clWaitForEvents(	1, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[1]);
            __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_readMPI2[1]);
            if (bstep && m->BACK_PROP_TYPE==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].syy_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 2, m->MYID+1, 2, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].syy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].syy_r_sub2, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].syy_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 2, m->MYID+1, 2, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].syy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].syy_sub2, 0, NULL, NULL);
            
            __GUARD clWaitForEvents(	1, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[2]);
            __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_readMPI2[2]);
            if (bstep && m->BACK_PROP_TYPE==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].sxy_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 3, m->MYID+1, 3, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxy_r_sub2, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].sxy_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 3, m->MYID+1, 3, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxy_sub2, 0, NULL, NULL);
            
            __GUARD clWaitForEvents(	1, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[3]);
            __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_readMPI2[3]);
            if (bstep && m->BACK_PROP_TYPE==1){
                if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].syz_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 4, m->MYID+1, 4, MPI_COMM_WORLD, NULL);
                __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].syz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].syz_r_sub2, 0, NULL, NULL);
            }
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].syz_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 4, m->MYID+1, 4, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].syz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].syz_sub2, 0, NULL, NULL);
        }
    
    __GUARD clWaitForEvents(	1, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[4]);
    __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_readMPI2[4]);
    if (bstep && m->BACK_PROP_TYPE==1){
        if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].szz_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 5, m->MYID+1, 5, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].szz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].szz_r_sub2, 0, NULL, NULL);
    }
    if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].szz_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 5, m->MYID+1, 5, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].szz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].szz_sub2, 0, NULL, NULL);
        
        __GUARD clWaitForEvents(	1, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[5]);
        __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_readMPI2[5]);
        if (bstep && m->BACK_PROP_TYPE==1){
            if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].sxz_r_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 6, m->MYID+1, 6, MPI_COMM_WORLD, NULL);
            __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxz_r_sub2, 0, NULL, NULL);
        }
    if (!state) state= MPI_Sendrecv_replace( (void*)(*mloc)[m->NUM_DEVICES-1].sxz_sub2, (int)m->buffer_size_comm/sizeof(float), MPI_FLOAT,m->MYID+1, 6, m->MYID+1, 6, MPI_COMM_WORLD, NULL);
        __GUARD clEnqueueWriteBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxz_sub2, 0, NULL, &(*vcl)[m->NUM_DEVICES-1].event_writes2);
        }

    return state;
    
}

int comm_v(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep){
    /*Communication for domain decompositon for MPI (between processing elements) and OpenCL (between devices) */
    int state = 0;
    int d;
    
    
    //Read buffers for comunnication between devices
    for (d=0;d<m->NUM_DEVICES;d++){

        
        if (d>0){
            if (m->ND==21){

                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].vy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub1, 1, &(*vcl)[d-1].event_updatev_com, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].vy_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub1, 0, NULL, &(*vcl)[d].event_readv1);
                }
                else{
                   __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].vy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub1, 1, &(*vcl)[d-1].event_updatev_com, &(*vcl)[d].event_readv1);
                }
                __GUARD clReleaseEvent((*vcl)[d-1].event_updatev_com);
            }
            else{
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].vx_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vx_sub1, 1, &(*vcl)[d-1].event_updatev_com, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].vy_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub1, 0, NULL, NULL);
                }
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].vx_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vx_r_sub1, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].vy_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub1, 0, NULL, NULL);
                    }
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].vz_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vz_r_sub1, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].vz_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vz_sub1, 0, NULL, &(*vcl)[d].event_readv1);
                __GUARD clReleaseEvent((*vcl)[d-1].event_updatev_com);
            }
        }
        
        if (d<m->NUM_DEVICES-1){
            if (m->ND==21){
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].vy_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub2, 1, &(*vcl)[d+1].event_updatev_com, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].vy_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub2, 0, NULL, &(*vcl)[d].event_readv2);
                }
                else{
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].vy_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub2, 1, &(*vcl)[d+1].event_updatev_com, &(*vcl)[d].event_readv2);
                }
                __GUARD clReleaseEvent((*vcl)[d+1].event_updatev_com);
            }
            else{
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].vx_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vx_sub2, 1, &(*vcl)[d+1].event_updatev_com, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].vy_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub2, 0, NULL, NULL);
                }
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].vx_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vx_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].vz_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vz_r_sub2, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].vy_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub2, 0, NULL, NULL);
                    }
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].vz_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].vz_sub2, 0, NULL, &(*vcl)[d].event_readv2);
                __GUARD clReleaseEvent((*vcl)[d+1].event_updatev_com);
            }
        }
        
    }
    
    //Read buffers for comunnication between MPI processes sharing this shot
    if (m->MYLOCALID>0){
        
        if (m->ND==21){
            if (bstep && m->BACK_PROP_TYPE==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_r_sub1, 1, &(*vcl)[0].event_updatev_com, NULL);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_sub1, 1, &(*vcl)[0].event_updatev_com, &(*vcl)[0].event_readMPI1[0]);
            __GUARD clReleaseEvent((*vcl)[0].event_updatev_com);
        }
        else{
            if (bstep && m->BACK_PROP_TYPE==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vx_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vx_r_sub1, 1, &(*vcl)[0].event_updatev_com, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_r_sub1, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vz_r_sub1, 0, NULL, NULL);
            }
            
            __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vx_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vx_sub1, 1, &(*vcl)[0].event_updatev_com, &(*vcl)[0].event_readMPI1[0]);
            if (m->ND==3){// For 3D
                __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vy_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[1]);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].vz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].vz_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[2]);
            __GUARD clReleaseEvent((*vcl)[0].event_updatev_com);
        }
        
    }
    if (m->MYLOCALID<m->NLOCALP-1){
        if (m->ND==21){
            if (bstep && m->BACK_PROP_TYPE==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vy_r_sub2, 1, &(*vcl)[m->NUM_DEVICES-1].event_updatev_com, NULL);
            }
            
            __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vy_sub2, 1, &(*vcl)[m->NUM_DEVICES-1].event_updatev_com, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[0]);
            __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_updatev_com);
        }
        else{
            if (bstep && m->BACK_PROP_TYPE==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vx_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vx_r_sub2, 1, &(*vcl)[m->NUM_DEVICES-1].event_updatev_com, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vy_r_sub2, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vz_r_sub2, 0, NULL, NULL);
            }
            
            __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vx_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vx_sub2, 1, &(*vcl)[m->NUM_DEVICES-1].event_updatev_com, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[0]);
            if (m->ND==3){// For 3D
                __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vy_sub2, 0, NULL, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[1]);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].vz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].vz_sub2, 0, NULL, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[2]);
            __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_updatev_com);
        }
    }
    
    for (d=0;d<m->NUM_DEVICES;d++){
        clFlush((*vcl)[d].queue);
        clFlush((*vcl)[d].queuecomm);
    }

    //Write buffers for comunnication between devices
    for (d=0;d<m->NUM_DEVICES;d++){
        if (d>0){
            if (m->ND==21){
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vy_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub1, 1, &(*vcl)[d].event_readv1, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vy_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub1, 0, NULL, &(*vcl)[d].event_writev1);
                }
                else{
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vy_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub1, 1, &(*vcl)[d].event_readv1, &(*vcl)[d].event_writev1);
                }
                __GUARD clReleaseEvent((*vcl)[d].event_readv1);
            }
            else{
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vx_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vx_sub1, 1, &(*vcl)[d].event_readv1, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vy_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub1, 0, NULL, NULL);
                }
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vx_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vx_r_sub1, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vy_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub1, 0, NULL, NULL);
                    }
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vz_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vz_r_sub1, 0, NULL, NULL);
                }
                
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vz_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vz_sub1, 0, NULL, &(*vcl)[d].event_writev1);
                __GUARD clReleaseEvent((*vcl)[d].event_readv1);
            }
        }
        
        if (d<m->NUM_DEVICES-1){
            if (m->ND==21){

                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vy_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub2, 1, &(*vcl)[d].event_readv2, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vy_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub2, 0, NULL, &(*vcl)[d].event_writev2);
                }
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vy_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub2, 1, &(*vcl)[d].event_readv2, &(*vcl)[d].event_writev2);
                __GUARD clReleaseEvent((*vcl)[d].event_readv2);
            }
            else{
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vx_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vx_sub2, 1, &(*vcl)[d].event_readv2, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vy_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_sub2, 0, NULL, NULL);
                }
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vx_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vx_r_sub2, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vy_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vy_r_sub2, 0, NULL, NULL);
                    }
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vz_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vz_r_sub2, 0, NULL, NULL);
                }
                
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].vz_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].vz_sub2, 0, NULL, &(*vcl)[d].event_writev2);
                __GUARD clReleaseEvent((*vcl)[d].event_readv2);
            }
        }
        
    }
    
    for (d=0;d<m->NUM_DEVICES;d++){
            clFlush((*vcl)[d].queuecomm);
    }
    
    //Wait for Opencl buffers to be read, send MPI bufers and write to devices
    //Processess with even ID in the group send and receive buffers 1 first, and then buffers 2, vice versa for odd IDs
    if (m->MYLOCALID % 2 != 0 && m->MYLOCALID>0){
        __GUARD com_v_MPI(m, vcl, mloc, bstep);
    }
    
    if (m->MYLOCALID % 2 == 0 && m->MYLOCALID<m->NLOCALP-1){
        __GUARD com_v_MPI(m, vcl, mloc, bstep);
    }
    
    if (m->MYLOCALID % 2 == 0 && m->MYLOCALID>0){
        __GUARD com_v_MPI(m, vcl, mloc, bstep);
    }
    
    if (m->MYLOCALID % 2 != 0 && m->MYLOCALID<m->NLOCALP-1){
        __GUARD com_v_MPI(m, vcl, mloc, bstep);
    }
    
    return state;
    
}




int comm_s(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int bstep){
    
    int state = 0;
    int d;
    
    
    //Read buffers for comunnication between devices
    for (d=0;d<m->NUM_DEVICES;d++){
        
        
        if (d>0){
            if (m->ND==21){
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].sxy_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub1, 1, &(*vcl)[d-1].event_updates_com, NULL);
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].sxy_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].syz_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub1, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].syz_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub1, 0, NULL, &(*vcl)[d].event_reads1);
                __GUARD clReleaseEvent((*vcl)[d-1].event_updates_com);
            }
            else{
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].sxx_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_sub1, 1, &(*vcl)[d-1].event_updates_com, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].syy_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syy_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].sxy_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].syz_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub1, 0, NULL, NULL);
                }
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].sxx_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_r_sub1, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].syy_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syy_r_sub1, 0, NULL, NULL);
                        __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].sxy_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub1, 0, NULL, NULL);
                        __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].syz_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub1, 0, NULL, NULL);
                        
                    }
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].szz_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].szz_r_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].sxz_r_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_r_sub1, 0, NULL, NULL);
                    
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].szz_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].szz_sub1, 0, NULL, NULL);
                
                __GUARD clEnqueueReadBuffer( (*vcl)[d-1].queuecomm, (*vcl)[d-1].sxz_sub2_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_sub1, 0, NULL, &(*vcl)[d].event_reads1);
                __GUARD clReleaseEvent((*vcl)[d-1].event_updates_com);
            }
            
        }
        
        if (d<m->NUM_DEVICES-1){
            if (m->ND==21){
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].sxy_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub2, 1, &(*vcl)[d+1].event_updates_com, NULL);
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].sxy_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].syz_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub2, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].syz_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub2, 0, NULL, &(*vcl)[d].event_reads2);
                __GUARD clReleaseEvent((*vcl)[d+1].event_updates_com);
            }
            else{
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].sxx_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_sub2, 1, &(*vcl)[d+1].event_updates_com, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].syy_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syy_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].sxy_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].syz_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub2, 0, NULL, NULL);
                }
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].sxx_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_r_sub2, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].syy_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syy_r_sub2, 0, NULL, NULL);
                        __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].sxy_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub2, 0, NULL, NULL);
                        __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].syz_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub2, 0, NULL, NULL);
                    }
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].szz_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].szz_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].sxz_r_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_r_sub2, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].szz_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].szz_sub2, 0, NULL, NULL);
                __GUARD clEnqueueReadBuffer( (*vcl)[d+1].queuecomm, (*vcl)[d+1].sxz_sub1_dev, CL_FALSE,0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_sub2, 0, NULL, &(*vcl)[d].event_reads2);
                __GUARD clReleaseEvent((*vcl)[d+1].event_updates_com);
            }
        }
        
    }
    
    
    //Read buffers for comunnication between MPI processes sharing this shot
    if (m->MYLOCALID>0){
        if (m->ND==21){
            if (bstep && m->BACK_PROP_TYPE==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_r_sub1, 1, &(*vcl)[0].event_updates_com, NULL);
                __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].syz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_r_sub1, 0, NULL, NULL);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_sub1, 1, &(*vcl)[0].event_updates_com, &(*vcl)[0].event_readMPI1[0]);
            __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].syz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[1]);
            __GUARD clReleaseEvent((*vcl)[0].event_updates_com);
        }
        else{
            if (bstep && m->BACK_PROP_TYPE==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxx_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxx_r_sub1, 1, &(*vcl)[0].event_updates_com, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].syy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syy_r_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxy_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_r_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].syz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_r_sub1, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].szz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].szz_r_sub1, 0, NULL, NULL);
                __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxz_r_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxz_r_sub1, 0, NULL, NULL);
            }
            
            __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxx_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxx_sub1, 1, &(*vcl)[0].event_updates_com, &(*vcl)[0].event_readMPI1[0]);
            if (m->ND==3){// For 3D
                __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].syy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syy_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[1]);
                __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxy_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxy_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[2]);
                __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].syz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].syz_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[3]);
            }
            
            __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].szz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].szz_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[4]);
            __GUARD clEnqueueReadBuffer( (*vcl)[0].queuecomm, (*vcl)[0].sxz_sub1_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[0].sxz_sub1, 0, NULL, &(*vcl)[0].event_readMPI1[5]);
            __GUARD clReleaseEvent((*vcl)[0].event_updates_com);
        }
    }
    if (m->MYLOCALID<m->NLOCALP-1){
        if (m->ND==21){
            if (bstep && m->BACK_PROP_TYPE==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxy_r_sub2, 1, &(*vcl)[m->NUM_DEVICES-1].event_updates_com, NULL);
                __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].syz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].syz_r_sub2, 0, NULL, NULL);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxy_sub2, 1, &(*vcl)[m->NUM_DEVICES-1].event_updates_com, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[0]);
            __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].syz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].syz_sub2, 0, NULL, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[1]);
            __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_updates_com);
        }
        else{
            if (bstep && m->BACK_PROP_TYPE==1){
                __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxx_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxx_r_sub2, 1, &(*vcl)[m->NUM_DEVICES-1].event_updates_com, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].syy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].syy_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxy_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxy_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].syz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].syz_r_sub2, 0, NULL, NULL);
                }
                __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].szz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].szz_r_sub2, 0, NULL, NULL);
                __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxz_r_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxz_r_sub2, 0, NULL, NULL);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxx_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxx_sub2, 1, &(*vcl)[m->NUM_DEVICES-1].event_updates_com, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[0]);
            
            if (m->ND==3){// For 3D
                __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].syy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].syy_sub2, 0, NULL, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[1]);
                __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxy_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxy_sub2, 0, NULL, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[2]);
                __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].syz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].syz_sub2, 0, NULL, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[3]);
            }
            __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].szz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].szz_sub2, 0, NULL, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[4]);
            __GUARD clEnqueueReadBuffer( (*vcl)[m->NUM_DEVICES-1].queuecomm, (*vcl)[m->NUM_DEVICES-1].sxz_sub2_dev, CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[m->NUM_DEVICES-1].sxz_sub2, 0, NULL, &(*vcl)[m->NUM_DEVICES-1].event_readMPI2[5]);
            __GUARD clReleaseEvent((*vcl)[m->NUM_DEVICES-1].event_updates_com);
        }
    }
    
    for (d=0;d<m->NUM_DEVICES;d++){
        clFlush((*vcl)[d].queue);
        clFlush((*vcl)[d].queuecomm);
    }
    
    //Write buffers for comunnication between devices
    for (d=0;d<m->NUM_DEVICES;d++){
        if (d>0){
            if (m->ND==21){
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxy_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub1, 1, &(*vcl)[d].event_reads1, NULL);
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxy_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].syz_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub1, 0, NULL, NULL);
                }
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].syz_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub1, 0, NULL, &(*vcl)[d].event_writes1);
                __GUARD clReleaseEvent((*vcl)[d].event_reads1);
            }
            else{
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxx_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_sub1, 1, &(*vcl)[d].event_reads1, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].syy_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syy_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxy_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].syz_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub1, 0, NULL, NULL);
                }
                
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxx_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_r_sub1, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].syy_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syy_r_sub1, 0, NULL, NULL);
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxy_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub1, 0, NULL, NULL);
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].syz_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub1, 0, NULL, NULL);
                        
                    }
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].szz_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].szz_r_sub1, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxz_r_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_r_sub1, 0, NULL, NULL);
                }
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].szz_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].szz_sub1, 0, NULL, NULL);
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxz_sub1_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_sub1, 0, NULL, &(*vcl)[d].event_writes1);
                __GUARD clReleaseEvent((*vcl)[d].event_reads1);
            }
        }
        
        if (d<m->NUM_DEVICES-1){
            if (m->ND==21){
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxy_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub2, 1, &(*vcl)[d].event_reads2, NULL);
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxy_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].syz_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub2, 0, NULL, NULL);
                }
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].syz_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub2, 0, NULL, &(*vcl)[d].event_writes2);
                __GUARD clReleaseEvent((*vcl)[d].event_reads2);
            }
            else{
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxx_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_sub2, 1, &(*vcl)[d].event_reads2, NULL);
                if (m->ND==3){// For 3D
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].syy_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syy_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxy_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].syz_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_sub2, 0, NULL, NULL);
                }
                if (bstep && m->BACK_PROP_TYPE==1){
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxx_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxx_r_sub2, 0, NULL, NULL);
                    if (m->ND==3){// For 3D
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].syy_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syy_r_sub2, 0, NULL, NULL);
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxy_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxy_r_sub2, 0, NULL, NULL);
                        __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].syz_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].syz_r_sub2, 0, NULL, NULL);
                        
                    }
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].szz_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].szz_r_sub2, 0, NULL, NULL);
                    __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxz_r_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_r_sub2, 0, NULL, NULL);
                }
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].szz_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].szz_sub2, 0, NULL, NULL);
                __GUARD clEnqueueWriteBuffer((*vcl)[d].queuecomm,   (*vcl)[d].sxz_sub2_dev,   CL_FALSE, 0, m->buffer_size_comm, (void*)(*mloc)[d].sxz_sub2, 0, NULL, &(*vcl)[d].event_writes2);
                __GUARD clReleaseEvent((*vcl)[d].event_reads2);
            }
        }
        
    }
    
    for (d=0;d<m->NUM_DEVICES;d++){
        clFlush((*vcl)[d].queuecomm);
    }
    
    //Wait for Opencl buffers to be read, send MPI bufers and write to devices
    //Processess with even ID in the group send and receive buffers 1 first, and then buffers 2, vice versa for odd IDs
    if (m->MYLOCALID % 2 != 0 && m->MYLOCALID>0){
        __GUARD com_s_MPI(m, vcl, mloc, bstep);
    }
    
    if (m->MYLOCALID % 2 == 0 && m->MYLOCALID<m->NLOCALP-1){
        __GUARD com_s_MPI(m, vcl, mloc, bstep);
    }
    
    if (m->MYLOCALID % 2 == 0 && m->MYLOCALID>0){
        __GUARD com_s_MPI(m, vcl, mloc, bstep);
    }
    
    if (m->MYLOCALID % 2 != 0 && m->MYLOCALID<m->NLOCALP-1){
        __GUARD com_s_MPI(m, vcl, mloc, bstep);
    }

    
    return state;
    
}


