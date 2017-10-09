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

int clbuf_send(CUstream *inqueue, clbuf * buf)
{
    /*Routine to allocate memory buffers to the device*/
    
    int err = 0;

    /*Transfer memory from host to the device*/
    err = cuMemcpyHtoDAsync ( buf->mem, (void*)buf->host, buf->size, *inqueue );
    if (err !=0) fprintf(stderr,"%s\n",clerrors(err));
    
    return err;
}

int clbuf_sendpin(CUstream *inqueue,
                  clbuf * buf,
                  clbuf * bufpin,
                  int offset)
{
    /*Routine to allocate memory buffers to the device*/
    
    int err = 0;
    /*Transfer memory from host to the device*/
    err = cuMemcpyHtoDAsync ( buf->mem,
                             (void*)&buf->host[offset],
                             buf->size,
                             *inqueue );
    if (err !=0) fprintf(stderr,"%s\n",clerrors(err));
    
    return err;
}

int clbuf_read(CUstream *inqueue, clbuf * buf)
{
    /*Routine to read memory buffers from the device*/
    
    int err = 0;
    
    /*Read memory from device to the host*/
    err= cuMemcpyDtoHAsync ( buf->host, buf->mem, buf->size, *inqueue );
    
    if (err !=0) fprintf(stderr,"%s\n",clerrors(err));
    
    return err;
}

int clbuf_readpin(CUstream *inqueue,
                  clbuf * buf,
                  clbuf * bufpin,
                  int offset)
{
    /*Routine to read memory buffers from the device*/
    
    int err = 0;
    
    /*Read memory from device to the host*/
    err= cuMemcpyDtoHAsync(&bufpin->host[offset], buf->mem,buf->size, *inqueue);
    
    if (err !=0) fprintf(stderr,"%s\n",clerrors(err));
    
    return err;
}

int clbuf_create(clbuf * buf)
{
    /*Create the buffer on the device */
    int err = 0;
    err = cuMemAlloc( &(*buf).mem , (*buf).size);
    if (err !=0) fprintf(stderr,"%s\n",clerrors(err));
    
    return err;
    
}

int clbuf_create_pin(clbuf * buf)
{
    size_t sizepin;
    /*Create pinned memory */
    int err = 0;
    err = cuMemAlloc( &(*buf).mem , (*buf).size);

    if ((*buf).sizepin>0){
        sizepin=(*buf).sizepin;
    }
    else{
        sizepin=(*buf).size;
    }
    err = cuMemAllocHost((void**)&(*buf).pin, sizepin);
        
    if (err !=0) fprintf(stderr,"%s\n",clerrors(err));
    
    return err;
    
}



