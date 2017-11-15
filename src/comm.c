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
//int comm1_MPI(model * m, device ** dev, int adj, int ui){
//    int state=0;
//    int i;
//
//    //For buffer to transmit, wait for the event telling that reading has
//    //occured, then release the event. We exchange the buffer with the previous
//    //processing element, sending buf1 et receiving buf2.
//    //Then send to device buf1. The last buf_1 in the list to transmit outputs
//    // an event for fcom1_in.
//    if (adj && m->BACK_PROP_TYPE==1){
//        for (i=0;i<m->ups_adj[ui].nvcom;i++){
//            __GUARD clWaitForEvents(1,
//                               &(*dev)[0].ups_adj[i].v2com[i]->cl_buf1.event_r);
//            __GUARD clReleaseEvent(
//                                (*dev)[0].ups_adj[i].v2com[i]->cl_buf1.event_r);
//            
//            MPI_Sendrecv_replace(
//             (void*)(*dev)[0].ups_adj[i].v2com[i]->cl_buf1.host,
//             (int)(*dev)[0].ups_adj[i].v2com[i]->cl_buf1.size/sizeof(float),
//                                 MPI_FLOAT,
//                                 m->MYID-1,
//                                 i,
//                                 m->MYID-1,
//                                 i,
//                                 MPI_COMM_WORLD,
//                                 NULL);
//            __GUARD clbuf_send(
//                               &(*dev)[0].queuecomm,
//                               &(*dev)[0].ups_adj[ui].v2com[i]->cl_buf1);
//        }
//    }
//    for (i=0;i<m->ups_f[ui].nvcom;i++){
//        __GUARD clWaitForEvents(1,
//                                &(*dev)[0].ups_f[i].v2com[i]->cl_buf1.event_r);
//        __GUARD clReleaseEvent(  (*dev)[0].ups_f[i].v2com[i]->cl_buf1.event_r);
//        
//        MPI_Sendrecv_replace(
//                    (void*)(*dev)[0].ups_f[i].v2com[i]->cl_buf1.host,
//                    (int)(*dev)[0].ups_f[i].v2com[i]->cl_buf1.size/sizeof(float),
//                    MPI_FLOAT,
//                    m->MYID-1,
//                    i,
//                    m->MYID-1,
//                    i,
//                    MPI_COMM_WORLD,
//                    NULL);
//        
//        __GUARD clbuf_send(
//                           &(*dev)[0].queuecomm,
//                           &(*dev)[0].ups_f[ui].v2com[i]->cl_buf1);
//    }
//
//    return state;
//    
//}
//
//int comm2_MPI(model * m, device ** dev, int adj, int ui){
//    int state=0;
//    int i, ld;
//    
//    ld=m->NUM_DEVICES-1;
//    
//    //For buffer to transmit, wait for the event telling that reading has
//    //occured, then release the event. We exchange the buffer with the next
//    //processing element, sending buf2 et receiving buf1.
//    //Then send to device buf2. The last buf_2 in the list to transmit outputs
//    // an event for fcom2_in.
//    if (adj && m->BACK_PROP_TYPE==1){
//        for (i=0;i<m->ups_adj[ui].nvcom;i++){
//            __GUARD clWaitForEvents(1,
//                              &(*dev)[ld].ups_adj[i].v2com[i]->cl_buf2.event_r);
//            __GUARD clReleaseEvent(
//                               (*dev)[ld].ups_adj[i].v2com[i]->cl_buf2.event_r);
//            
//            MPI_Sendrecv_replace(
//                (void*)(*dev)[ld].ups_adj[i].v2com[i]->cl_buf2.host,
//                (int)(*dev)[ld].ups_adj[i].v2com[i]->cl_buf2.size/sizeof(float),
//                MPI_FLOAT,
//                m->MYID-1,
//                i,
//                m->MYID-1,
//                i,
//                MPI_COMM_WORLD,
//                NULL);
//            __GUARD clbuf_send(&(*dev)[ld].queuecomm,
//                               &(*dev)[ld].ups_adj[ui].v2com[i]->cl_buf2);
//        }
//    }
//    for (i=0;i<m->ups_f[ui].nvcom;i++){
//        __GUARD clWaitForEvents(1,
//                                &(*dev)[ld].ups_f[i].v2com[i]->cl_buf2.event_r);
//        __GUARD clReleaseEvent(  (*dev)[ld].ups_f[i].v2com[i]->cl_buf2.event_r);
//        
//        MPI_Sendrecv_replace(
//                  (void*)(*dev)[ld].ups_f[i].v2com[i]->cl_buf2.host,
//                  (int)(*dev)[ld].ups_f[i].v2com[i]->cl_buf2.size/sizeof(float),
//                  MPI_FLOAT,
//                  m->MYID+1,
//                  m->ups_f[ui].nvcom+i,
//                  m->MYID+1,
//                  m->ups_f[ui].nvcom+i,
//                  MPI_COMM_WORLD,
//                  NULL);
//        __GUARD clbuf_send(&(*dev)[ld].queuecomm,
//                           &(*dev)[ld].ups_f[ui].v2com[i]->cl_buf2);
//    }
//    
//    return state;
//    
//}
//
//
//int comm(model * m, device ** dev, int adj, int ui){
//    /* Communication for domain decompositon for MPI (between processes)
//     and OpenCL (between devices) */
//    
//    int state = 0;
//    int d,i, ld;
//    
//    //Read buffers for comunnication between MPI processes sharing this shot
//    if (m->MYLOCALID>0){
//        //For all MPI processes except the first, com1 must occur on the firt
//        //device
//        
//        
//        //The first buffer cl_buf1 in the list must wait on the kernel fcom1
//        //All buffers reading must output an event for MPI communications
//        if (adj && m->BACK_PROP_TYPE==1){
//            for (i=0;i<m->ups_adj[ui].nvcom;i++){
//                __GUARD clbuf_readto(&(*dev)[0].queuecomm,
//                                      &(*dev)[0].ups_adj[ui].v2com[i]->cl_buf1,
//                                      &(*dev)[0].ups_adj[ui].v2com[i]->cl_buf1.pin);
//            }
//        }
//        for (i=0;i<m->ups_f[ui].nvcom;i++){
//            __GUARD clbuf_readto(&(*dev)[0].queuecomm,
//                                  &(*dev)[0].ups_f[ui].v2com[i]->cl_buf1,
//                                  &(*dev)[0].ups_f[ui].v2com[i]->cl_buf1.pin);
//        }
//        //We can realease the fcom1 event, it is no longer needed.
//        if (adj){
//            __GUARD clReleaseEvent((*dev)[0].ups_adj[i].fcom1_out.event);
//        }
//        else{
//            __GUARD clReleaseEvent((*dev)[0].ups_f[i].fcom1_out.event);
//        }
//        
//    }
//    if (m->MYLOCALID<m->NLOCALP-1){
//        //For all MPI processes except the last, com2 must occur on the last
//        //device
//        
//        ld=m->NUM_DEVICES-1;
//        
//        //The first buffer cl_buf2 in the list must wait on the kernel fcom2
//        //All buffers reading must output an event for MPI communications
//        if (adj && m->BACK_PROP_TYPE==1){
//            for (i=0;i<m->ups_adj[ui].nvcom;i++){
//                __GUARD clbuf_readto(&(*dev)[ld].queuecomm,
//                                      &(*dev)[ld].ups_adj[ui].v2com[i]->cl_buf2,
//                                      &(*dev)[ld].ups_adj[ui].v2com[i]->cl_buf2.pin);
//            }
//        }
//        for (i=0;i<m->ups_f[ui].nvcom;i++){
//            __GUARD clbuf_readto( &(*dev)[ld].queuecomm,
//                                  &(*dev)[ld].ups_f[ui].v2com[i]->cl_buf2,
//                                  &(*dev)[ld].ups_f[ui].v2com[i]->cl_buf2.pin);
//        }
//        //We can realease the fcom2 event, it is no longer needed.
//        if (adj){
//            __GUARD clReleaseEvent((*dev)[ld].ups_adj[i].fcom2_out.event);
//        }
//        else{
//            __GUARD clReleaseEvent((*dev)[ld].ups_f[i].fcom2_out.event);
//        }
//        
//    }
//    
//    //Read buffers for comunnication between devices
//    for (d=0;d<m->NUM_DEVICES;d++){
//
//        if (d>0){
//            //For all devices except the first, com1 must occur
//            
//            //The first buffer cl_buf2 in the list must wait on the kernel
//            //fcom2 of the previous device
//            //The last buf2 in the list must output an event
//            if (adj && m->BACK_PROP_TYPE==1){
//                for (i=0;i<m->ups_adj[ui].nvcom;i++){
//                    __GUARD clbuf_readto(
//                                    &(*dev)[d-1].queuecomm,
//                                    &(*dev)[d-1].ups_adj[ui].v2com[i]->cl_buf2,
//                                          &(*dev)[d  ].ups_adj[ui].v2com[i]->cl_buf1.pin);
//                }
//            }
//            for (i=0;i<m->ups_f[ui].nvcom;i++){
//                __GUARD clbuf_readto(&(*dev)[d-1].queuecomm,
//                                      &(*dev)[d-1].ups_f[ui].v2com[i]->cl_buf2,
//                                      &(*dev)[d  ].ups_f[ui].v2com[i]->cl_buf1.pin);
//            }
//            //We can realease the fcom2 event, it is no longer needed.
//            if (adj){
//                __GUARD clReleaseEvent((*dev)[d-1].ups_adj[i].fcom2_out.event);
//            }
//            else{
//                __GUARD clReleaseEvent((*dev)[d-1].ups_f[i].fcom2_out.event);
//            }
//        }
//        
//        if (d<m->NUM_DEVICES-1){
//            //For all devices except the last, com2 must occur
//            
//            //The first buffer cl_buf1 in the list must wait on the kernel
//            //fcom1 of the next device
//            //The last buf1 in the list must output an event
//            
//            if (adj && m->BACK_PROP_TYPE==1){
//                for (i=0;i<m->ups_adj[ui].nvcom;i++){
//                    __GUARD clbuf_readto(
//                                    &(*dev)[d+1].queuecomm,
//                                    &(*dev)[d+1].ups_adj[ui].v2com[i]->cl_buf1,
//                                    &(*dev)[d  ].ups_adj[ui].v2com[i]->cl_buf2.pin);
//                }
//            }
//            for (i=0;i<m->ups_f[ui].nvcom;i++){
//                __GUARD clbuf_readto(&(*dev)[d+1].queuecomm,
//                                      &(*dev)[d+1].ups_f[ui].v2com[i]->cl_buf1,
//                                      &(*dev)[d  ].ups_f[ui].v2com[i]->cl_buf2.pin);
//            }
//            //We can realease the fcom1 event, it is no longer needed.
//            if (adj){
//                __GUARD clReleaseEvent((*dev)[d+1].ups_adj[i].fcom1_out.event);
//            }
//            else{
//                __GUARD clReleaseEvent((*dev)[d+1].ups_f[i].fcom1_out.event);
//            }
//            
//        }
//        
//    }
//    
//    //Write buffers for comunnication between devices
//    for (d=0;d<m->NUM_DEVICES;d++){
//        
//        if (d>0){
//            
//            //We must transfer buf1 from the host to the device. The first buf1
//            //transfer must wait on the buf2 receive of the previous device
//            //The last buf1 on the list must output an event for fcom1_in
//            if (adj && m->BACK_PROP_TYPE==1){
//                for (i=0;i<m->ups_adj[ui].nvcom;i++){
//                   __GUARD clbuf_send(&(*dev)[d].queuecomm,
//                                      &(*dev)[d].ups_adj[ui].v2com[i]->cl_buf1);
//                }
//            }
//            for (i=0;i<m->ups_f[ui].nvcom;i++){
//                __GUARD clbuf_send(
//                                   &(*dev)[d].queuecomm,
//                                   &(*dev)[d].ups_f[ui].v2com[i]->cl_buf1);
//            }
//            __GUARD clReleaseEvent(
//              (*dev)[d-1].ups_f[i].v2com[m->ups_f[i].nvcom-1]->cl_buf2.event_r);
//            
//        }
//        
//        if (d<m->NUM_DEVICES-1){
//            
//            //We must transfer buf2 from the host to the device. The first buf2
//            //transfer must wait on the buf1 receive of the previous device
//            if (adj && m->BACK_PROP_TYPE==1){
//                for (i=0;i<m->ups_adj[ui].nvcom;i++){
//                   __GUARD clbuf_send(&(*dev)[d].queuecomm,
//                                      &(*dev)[d].ups_adj[ui].v2com[i]->cl_buf2);
//                }
//            }
//            for (i=0;i<m->ups_f[ui].nvcom;i++){
//                __GUARD clbuf_send(
//                                   &(*dev)[d].queuecomm,
//                                   &(*dev)[d].ups_f[ui].v2com[i]->cl_buf2);
//            }
//            __GUARD clReleaseEvent(
//              (*dev)[d+1].ups_f[i].v2com[m->ups_f[i].nvcom-1]->cl_buf1.event_r);
//        }
//        
//    }
//    
//    //Sends those commands to the compute devices
//    for (d=0;d<m->NUM_DEVICES;d++){
//        clFlush( (*dev)[d].queuecomm);
//        clFlush( (*dev)[d].queue);
//    }
//    
//    // Wait for Opencl buffers to be read, send MPI bufers and write to devices
//    // Processess with even ID in the group send and receive buffers 1 first,
//    // and then buffers 2, vice versa for odd IDs
//    if (m->MYLOCALID % 2 != 0 && m->MYLOCALID>0){
//        __GUARD comm1_MPI(m, dev, adj, ui);
//    }
//    
//    if (m->MYLOCALID % 2 == 0 && m->MYLOCALID<m->NLOCALP-1){
//        __GUARD comm2_MPI(m, dev, adj, ui);
//    }
//    
//    if (m->MYLOCALID % 2 == 0 && m->MYLOCALID>0){
//        __GUARD comm1_MPI(m, dev, adj, ui);
//    }
//    
//    if (m->MYLOCALID % 2 != 0 && m->MYLOCALID<m->NLOCALP-1){
//        __GUARD comm2_MPI(m, dev, adj, ui);
//    }
//
//    return state;
//    
//}
//


