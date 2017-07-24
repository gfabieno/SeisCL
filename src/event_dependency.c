//
//  event_dependency.c
//  SeisCL
//
//  Created by Gabriel Fabien-Ouellet on 17-07-20.
//  Copyright © 2017 Gabriel Fabien-Ouellet. All rights reserved.
//

#include "F.h"



int event_dependency( model * m,  device ** dev, int adj){
//Assign kernels and memory transfer event dependencies

    int state=0;
    int i,j,d, ld, lu;
    
    
    //Event dependencies for MPI communication
    for (i=0;i<m->nupdates;i++){
        lu=m->ups_f[i].nvcom-1;
        if (m->MYLOCALID>0){
            
            //The first buffer cl_buf1 in the list must wait on the kernel fcom1
            if (adj){
                (*dev)[0].ups_adj[i].fcom1_out.outevent=1;
            }
            else{
                (*dev)[0].ups_f[i].fcom1_out.outevent=1;
            }
            if (adj && m->BACK_PROP_TYPE==1){
                (*dev)[0].ups_adj[i].v2com[0]->cl_buf1.nwait_r=1;
                (*dev)[0].ups_adj[i].v2com[0]->cl_buf1.waits_r=
                &(*dev)[0].ups_adj[i].fcom1_out.event;
            }
            else{
                (*dev)[0].ups_f[i].v2com[0]->cl_buf1.nwait_r=1;
                (*dev)[0].ups_f[i].v2com[0]->cl_buf1.waits_r=
                &(*dev)[0].ups_f[i].fcom1_out.event;
            }
            
            //All buffers reading must output an event for MPI communications
            for (j=0;j<m->ups_f[i].nvcom;j++){
                if (adj && m->BACK_PROP_TYPE==1){
                    (*dev)[0].ups_adj[i].v2com[j]->cl_buf1.outevent_r=1;
                }
                (*dev)[0].ups_f[i].v2com[j]->cl_buf1.outevent_r=1;
            }
            
            //The last buf1 outputs an event for fcom1_in.
            (*dev)[0].ups_f[i].v2com[lu]->cl_buf1.outevent_s=1;
            
            if (adj){
                (*dev)[0].ups_adj[i].fcom1_in.nwait=1;
                (*dev)[0].ups_adj[i].fcom1_in.waits=
                &(*dev)[0].ups_f[i].v2com[lu]->cl_buf1.event_s;
                
            }
            else{
                (*dev)[0].ups_f[i].fcom1_in.nwait=1;
                (*dev)[0].ups_f[i].fcom1_in.waits=
                &(*dev)[0].ups_f[i].v2com[lu]->cl_buf1.event_s;
                
            }
            
        }
        if (m->MYLOCALID<m->NLOCALP-1){
            ld=m->NUM_DEVICES-1;
            
            //The first buffer cl_buf2 in the list must wait on the kernel fcom2
            if (adj){
                (*dev)[ld].ups_adj[i].fcom2_out.outevent=1;
            }
            else{
                (*dev)[ld].ups_f[i].fcom2_out.outevent=1;
            }
            if (adj && m->BACK_PROP_TYPE==1){
                (*dev)[ld].ups_adj[i].v2com[0]->cl_buf2.nwait_r=1;
                (*dev)[ld].ups_adj[i].v2com[0]->cl_buf2.waits_r=
                &(*dev)[ld].ups_adj[i].fcom2_out.event;
            }
            else{
                (*dev)[ld].ups_f[i].v2com[0]->cl_buf2.nwait_r=1;
                (*dev)[ld].ups_f[i].v2com[0]->cl_buf2.waits_r=
                &(*dev)[ld].ups_f[i].fcom2_out.event;
            }
            
            //All buffers reading must output an event for MPI communications
            for (j=0;j<m->ups_f[i].nvcom;j++){
                if (adj && m->BACK_PROP_TYPE==1){
                    (*dev)[ld].ups_adj[i].v2com[j]->cl_buf2.outevent_r=1;
                }
                (*dev)[ld].ups_f[i].v2com[j]->cl_buf2.outevent_r=1;
            }
            
            //The last buf2 outputs an event for fcom2_in.
            (*dev)[ld].ups_f[i].v2com[lu]->cl_buf2.outevent_s=1;
            
            if (adj){
                (*dev)[ld].ups_adj[i].fcom2_in.nwait=1;
                (*dev)[ld].ups_adj[i].fcom2_in.waits=
                &(*dev)[ld].ups_f[i].v2com[lu]->cl_buf2.event_s;
                
            }
            else{
                (*dev)[ld].ups_f[i].fcom2_in.nwait=1;
                (*dev)[ld].ups_f[i].fcom2_in.waits=
                &(*dev)[ld].ups_f[i].v2com[lu]->cl_buf2.event_s;
                
            }
            
        }
    }
    
    //Assign events to outut and event to wait for for each devices
    for (d=0;d<m->NUM_DEVICES;d++){
        for (i=0;i<m->nupdates;i++){
            lu=m->ups_f[i].nvcom-1;
            if (d>0 ){
                //For all devices except the first, com1 must occur
                
                //The first buffer cl_buf1 in the list must wait on the kernel
                //fcom1
                if (adj){
                    (*dev)[d].ups_adj[i].fcom1_out.outevent=1;
                }
                else{
                    (*dev)[d].ups_f[i].fcom1_out.outevent=1;
                }
                if (adj && m->BACK_PROP_TYPE==1){
                    (*dev)[d].ups_adj[i].v2com[0]->cl_buf1.nwait_r=1;
                    (*dev)[d].ups_adj[i].v2com[0]->cl_buf1.waits_r=
                    &(*dev)[d].ups_adj[i].fcom1_out.event;
                }
                else{
                    (*dev)[d].ups_f[i].v2com[0]->cl_buf1.nwait_r=1;
                    (*dev)[d].ups_f[i].v2com[0]->cl_buf1.waits_r=
                    &(*dev)[d].ups_f[i].fcom1_out.event;
                }
                
                //The last buf1 in the list must output an event
                (*dev)[d].ups_f[i].v2com[lu]->cl_buf1.outevent_r=1;
                
                //The first buf1 transfer must wait on the buf2 receive of the
                //previous device
                if (adj && m->BACK_PROP_TYPE==1){
                    (*dev)[d].ups_adj[i].v2com[0]->cl_buf1.nwait_s=1;
                    (*dev)[d].ups_adj[i].v2com[0]->cl_buf1.waits_s=
                    &(*dev)[d-1].ups_f[i].v2com[lu]->cl_buf2.event_r;
                }
                else{
                    (*dev)[d].ups_f[i].v2com[0]->cl_buf1.nwait_s=1;
                    (*dev)[d].ups_f[i].v2com[0]->cl_buf1.waits_s=
                    &(*dev)[d-1].ups_f[i].v2com[lu]->cl_buf2.event_r;
                }
                //The last buf1 on the list must output an event for fcom1_in
                (*dev)[d].ups_f[i].v2com[lu]->cl_buf1.outevent_s=1;
                
                if (adj){
                    (*dev)[d].ups_adj[i].fcom1_in.nwait=1;
                    (*dev)[d].ups_adj[i].fcom1_in.waits=
                    &(*dev)[d].ups_f[i].v2com[lu]->cl_buf1.event_s;
                    
                }
                else{
                    (*dev)[d].ups_f[i].fcom1_in.nwait=1;
                    (*dev)[d].ups_f[i].fcom1_in.waits=
                    &(*dev)[d].ups_f[i].v2com[lu]->cl_buf1.event_s;
                    
                }
                
                
            }
            if (d<m->NUM_DEVICES-1 ){
                //For all devices except the last, com2 must occur
                
                //The first buffer cl_buf2 in the list must wait on the kernel
                //fcom2
                if (adj){
                    (*dev)[d].ups_adj[i].fcom2_out.outevent=1;
                }
                else{
                    (*dev)[d].ups_f[i].fcom2_out.outevent=1;
                }
                if (adj && m->BACK_PROP_TYPE==1){
                    (*dev)[d].ups_adj[i].v2com[0]->cl_buf2.nwait_r=1;
                    (*dev)[d].ups_adj[i].v2com[0]->cl_buf2.waits_r=
                    &(*dev)[d].ups_adj[i].fcom2_out.event;
                }
                else{
                    (*dev)[d].ups_f[i].v2com[0]->cl_buf2.nwait_r=1;
                    (*dev)[d].ups_f[i].v2com[0]->cl_buf2.waits_r=
                    &(*dev)[d].ups_f[i].fcom2_out.event;
                }
                
                //The last buf2 in the list must output an event
                (*dev)[d].ups_f[i].v2com[lu]->cl_buf2.outevent_r=1;
                
                //The first buf2 transfer must wait on the buf1 receive of the
                //next device
                if (adj && m->BACK_PROP_TYPE==1){
                    (*dev)[d].ups_adj[i].v2com[0]->cl_buf2.nwait_s=1;
                    (*dev)[d].ups_adj[i].v2com[0]->cl_buf2.waits_s=
                    &(*dev)[d+1].ups_f[i].v2com[lu]->cl_buf1.event_r;
                }
                else{
                    (*dev)[d].ups_f[i].v2com[0]->cl_buf2.nwait_s=1;
                    (*dev)[d].ups_f[i].v2com[0]->cl_buf2.waits_s=
                    &(*dev)[d+1].ups_f[i].v2com[lu]->cl_buf1.event_r;
                }
                
                //The last buf2 on the list must output an event for fcom2_in
                (*dev)[d].ups_f[i].v2com[lu]->cl_buf2.outevent_s=1;
                
                if (adj){
                    (*dev)[d].ups_adj[i].fcom2_in.nwait=1;
                    (*dev)[d].ups_adj[i].fcom2_in.waits=
                    &(*dev)[d].ups_f[i].v2com[lu]->cl_buf2.event_s;
                    
                }
                else{
                    (*dev)[d].ups_f[i].fcom2_in.nwait=1;
                    (*dev)[d].ups_f[i].fcom2_in.waits=
                    &(*dev)[d].ups_f[i].v2com[lu]->cl_buf2.event_s;
                    
                }
                
            }
        }
    }
    

    

   
    
    return state;
}
