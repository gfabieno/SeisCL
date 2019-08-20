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

CL_INT clbuf_send(QUEUE *inqueue, clbuf * buf)
{
    /*Routine to allocate memory buffers to the device*/
    
    CL_INT state = 0;
    #ifdef __SEISCL__
    cl_event * event=NULL;
    if (buf->outevent_s){
        if (buf->event_s){
            state = clReleaseEvent(buf->event_s);
        }
        event=&buf->event_s;
    }
    /*Transfer memory from host to the device*/
    state = clEnqueueWriteBuffer(*inqueue, buf->mem,
                               CL_TRUE,
                               0,
                               buf->size,
                               (void*)buf->host,
                               buf->nwait_s,
                               buf->waits_s,
                               event);
    #else
    if (buf->nwait_s >0){
        state = cuStreamWaitEvent(*inqueue, *buf->waits_s, 0);
    }
    state = cuMemcpyHtoDAsync( buf->mem, (void*)buf->host, buf->size, *inqueue );
    if (buf->outevent_s){
        if (!buf->event_s){
            state =  cuEventCreate(&buf->event_s, CU_EVENT_DISABLE_TIMING);
        }
        state = cuEventRecord(buf->event_s, *inqueue);
    }
    #endif
    if (state !=CUCL_SUCCESS)
        fprintf(stderr,"Error: clbuf_send: %s\n", clerrors(state));
    
    return state;
}

CL_INT clbuf_sendfrom(QUEUE *inqueue,
                   clbuf * buf,
                   void * ptr)
{
    /*Routine to allocate memory buffers to the device*/
    
    CL_INT state = 0;
    /*Transfer memory from host to the device*/
    #ifdef __SEISCL__
    cl_event * event=NULL;
    if (buf->outevent_s){
        if (buf->event_s){
            state = clReleaseEvent(buf->event_s);
        }
        event=&buf->event_s;
    }
    /*Transfer memory from host to the device*/
    state = clEnqueueWriteBuffer(*inqueue, buf->mem,
                               CL_TRUE,
                               0,
                               buf->size,
                               ptr,
                               buf->nwait_s,
                               buf->waits_s,
                               event);
    #else
    if (buf->nwait_s >0){
        state = cuStreamWaitEvent(*inqueue, *buf->waits_s, 0);
    }
    state = cuMemcpyHtoDAsync (buf->mem,
                             ptr,
                             buf->size,
                             *inqueue );
    if (buf->outevent_s){
        if (!buf->event_s){
            state =  cuEventCreate(&buf->event_s, CU_EVENT_DISABLE_TIMING);
        }
        state = cuEventRecord(buf->event_s, *inqueue);
    }
    #endif
    if (state !=CUCL_SUCCESS) fprintf(stderr,
                                    "Error: clbuf_sendfrom: %s\n",
                                    clerrors(state));
    
    return state;
}

CL_INT clbuf_read(QUEUE *inqueue, clbuf * buf)
{
    /*Routine to read memory buffers from the device*/
    
    CL_INT state = 0;
    
    /*Read memory from device to the host*/
    #ifdef __SEISCL__
    cl_event * event=NULL;
    if (buf->outevent_r){
        if (buf->event_r){
            state = clReleaseEvent(buf->event_r);
        }
        event=&buf->event_r;
    }
    /*Read memory from device to the host*/
    state = clEnqueueReadBuffer(*inqueue,
                              buf->mem,
                              CL_FALSE,
                              0,
                              buf->size,
                              buf->host,
                              buf->nwait_r,
                              buf->waits_r,
                              event);
    #else
    if (buf->nwait_r >0){
        state = cuStreamWaitEvent(*inqueue, *buf->waits_r, 0);
    }
    state= cuMemcpyDtoHAsync ( buf->host, buf->mem, buf->size, *inqueue );
    if (buf->outevent_r){
        if (!buf->event_r){
            state =  cuEventCreate(&buf->event_r, CU_EVENT_DISABLE_TIMING);
        }
        state = cuEventRecord(buf->event_r, *inqueue);
    }
    #endif
    if (state !=CUCL_SUCCESS) fprintf(stderr,
                                    "Error: clbuf_read: %s\n",
                                    clerrors(state));
    
    return state;
}

CL_INT clbuf_readto(QUEUE *inqueue,
                    clbuf * buf,
                    void * ptr)
{
    /*Routine to read memory buffers from the device*/
    
    CL_INT state = 0;
    
    /*Read memory from device to the host*/
    #ifdef __SEISCL__
    cl_event * event=NULL;
    if (buf->outevent_r){
        if (buf->event_r){
            state = clReleaseEvent(buf->event_r);
        }
        event=&buf->event_r;
    }

    /*Read memory from device to the host*/
    state = clEnqueueReadBuffer(*inqueue,
                              buf->mem,
                              CL_FALSE,
                              0,
                              buf->size,
                              (float *)ptr,
                              buf->nwait_r,
                              buf->waits_r,
                              event);
    
    #else
    if (buf->nwait_r >0){
        state = cuStreamWaitEvent(*inqueue, *buf->waits_r, 0);
    }
    state= cuMemcpyDtoHAsync(ptr, buf->mem, buf->size, *inqueue);
    if (buf->outevent_r){
        if (!buf->event_r){
            state =  cuEventCreate(&buf->event_r, CU_EVENT_DISABLE_TIMING);
        }
        state = cuEventRecord(buf->event_r, *inqueue);
    }
    #endif
    if (state !=CUCL_SUCCESS)
        fprintf(stderr, "Error: clbuf_readto: %s\n", clerrors(state));
    
    return state;
}

CL_INT clbuf_create(CONTEXT *incontext, clbuf * buf)
{
    /*Create the buffer on the device */
    CL_INT state = 0;
    #ifdef __SEISCL__
    (*buf).mem = clCreateBuffer(*incontext,
                                CL_MEM_READ_WRITE,
                                (*buf).size, NULL, &state);
    #else
    state = cuMemAlloc( &(*buf).mem , (*buf).size);
    #endif
    if (state !=CUCL_SUCCESS) fprintf(stderr,
                                    "Error: clbuf_create: %s\n",
                                    clerrors(state));
    
    return state;
    
}

CL_INT clbuf_create_pin(CONTEXT *incontext, QUEUE *inqueue,
                        clbuf * buf)
{
    size_t sizepin;
    /*Create pinned memory */
    CL_INT state = 0;
    #ifdef __SEISCL__
    (*buf).mem = clCreateBuffer(*incontext,
                                CL_MEM_READ_WRITE,
                                (*buf).size,
                                NULL,
                                &state);
    #else
    state = cuMemAlloc( &(*buf).mem , (*buf).size);
    #endif
    if ((*buf).sizepin>0){
        sizepin=(*buf).sizepin;
    }
    else{
        sizepin=(*buf).size;
    }
    #ifdef __SEISCL__
    (*buf).pin= clCreateBuffer(*incontext,
                               CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               sizepin,
                               NULL,
                               &state);
    
    (*buf).host = (float*)clEnqueueMapBuffer(*inqueue,
                                             (*buf).pin,
                                             CL_TRUE,
                                             CL_MAP_WRITE | CL_MAP_READ,
                                             0,
                                             sizepin,
                                             0,
                                             NULL,
                                             NULL,
                                             &state);
    if (state==CL_MEM_OBJECT_ALLOCATION_FAILURE){
        (*buf).host = malloc(sizepin);
        fprintf(stdout, "Warning: could not allocate pinned memory\n");
        (*buf).free_host = 1;
        state = 0;
    }
    else{
        (*buf).free_pin = 1;
    }
    #else
    state = cuMemAllocHost((void**)&(*buf).host, sizepin);
    #endif
    
    if (state !=CUCL_SUCCESS) fprintf(stderr,
                                    "Error: clbuf_create_pin: %s\n",
                                    clerrors(state));
    
    return state;
    
}



