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

cl_int clbuf_send( cl_command_queue *inqueue, clbuf * buf)
{
    /*Routine to allocate memory buffers to the device*/
    
    cl_int cl_err = 0;
    cl_event * event=NULL;
    if (buf->outevent_s)
        event=&buf->event_s;
    /*Transfer memory from host to the device*/
    cl_err = clEnqueueWriteBuffer(*inqueue, buf->mem,
                                  CL_TRUE,
                                  0,
                                  buf->size,
                                  (void*)buf->host,
                                  buf->nwait_s,
                                  buf->waits_s,
                                  event);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",clerrors(cl_err));
    
    return cl_err;
}

cl_int clbuf_sendpin( cl_command_queue *inqueue,
                     clbuf * buf,
                     clbuf * bufpin,
                     int offset)
{
    /*Routine to allocate memory buffers to the device*/
    
    cl_int cl_err = 0;
    cl_event * event=NULL;
    if (buf->outevent_s)
        event=&buf->event_s;
    /*Transfer memory from host to the device*/
    cl_err = clEnqueueWriteBuffer(*inqueue, buf->mem,
                                  CL_TRUE,
                                  0,
                                  buf->size,
                                  (void*)&bufpin->host[offset],
                                  buf->nwait_s,
                                  buf->waits_s,
                                  event);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",clerrors(cl_err));
    
    return cl_err;
}



cl_int clbuf_read( cl_command_queue *inqueue, clbuf * buf)
{
    /*Routine to read memory buffers from the device*/
    
    cl_int cl_err = 0;
    cl_event * event=NULL;
    if (buf->outevent_r)
        event=&buf->event_r;
    
    /*Read memory from device to the host*/
    cl_err = clEnqueueReadBuffer(*inqueue,
                                 buf->mem,
                                 CL_FALSE,
                                 0,
                                 buf->size,
                                 buf->host,
                                 buf->nwait_r,
                                 buf->waits_r,
                                 event);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",clerrors(cl_err));
    
    return cl_err;
}

cl_int clbuf_readpin( cl_command_queue *inqueue,
                     clbuf * buf,
                     clbuf * bufpin,
                     int offset)
{
    /*Routine to read memory buffers from the device*/
    
    cl_int cl_err = 0;
    cl_event * event=NULL;
    if (buf->outevent_r)
        event=&buf->event_r;
    
    /*Read memory from device to the host*/
    cl_err = clEnqueueReadBuffer(*inqueue,
                                 buf->mem,
                                 CL_FALSE,
                                 0,
                                 buf->size,
                                 &bufpin->host[offset],
                                 buf->nwait_r,
                                 buf->waits_r,
                                 event);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",clerrors(cl_err));
    
    return cl_err;
}

cl_int clbuf_create(cl_context *incontext, clbuf * buf)
{
    /*Create the buffer on the device */
    cl_int cl_err = 0;
    (*buf).mem = clCreateBuffer(*incontext, CL_MEM_READ_WRITE, (*buf).size, NULL, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",clerrors(cl_err));
    
    return cl_err;
    
}

cl_int clbuf_create_pin(cl_context *incontext, cl_command_queue *inqueue,
                        clbuf * buf)
{
    /*Create pinned memory */
    cl_int cl_err = 0;
    (*buf).mem = clCreateBuffer(*incontext,
                                CL_MEM_READ_WRITE,
                                (*buf).size,
                                NULL,
                                &cl_err);
    (*buf).pin= clCreateBuffer(*incontext,
                               CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               (*buf).size,
                               NULL,
                               &cl_err);
    
    (*buf).host = (float*)clEnqueueMapBuffer(*inqueue,
                                             (*buf).mem,
                                             CL_TRUE,
                                             CL_MAP_WRITE | CL_MAP_READ,
                                             0,
                                             (*buf).size,
                                             0,
                                             NULL,
                                             NULL,
                                             &cl_err);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",clerrors(cl_err));
    
    return cl_err;
    
}

cl_int clbuf_create_cst(cl_context *incontext, clbuf * buf)
{
    /*Create read only memory */
    cl_int cl_err = 0;
    (*buf).mem = clCreateBuffer(*incontext, CL_MEM_READ_ONLY, (*buf).size, NULL, &cl_err);
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s\n",clerrors(cl_err));
    
    return cl_err;
    
}

