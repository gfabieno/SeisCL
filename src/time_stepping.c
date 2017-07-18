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

/*Main part of the program. Perform FD time stepping */

#include "F.h"

#define rho(z,y,x) rho[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define rip(z,y,x) rip[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define rjp(z,y,x) rjp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define rkp(z,y,x) rkp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define uipjp(z,y,x) uipjp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define ujpkp(z,y,x) ujpkp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define uipkp(z,y,x) uipkp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define u(z,y,x) u[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define pi(z,y,x) pi[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define grad(z,y,x) grad[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define grads(z,y,x) grads[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define amp1(z,y,x) amp1[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define amp2(z,y,x) amp2[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]

#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vy0(y,x) vy0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define ry(y,x) ry[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]
#define mute(y,x) mute[(y)*5+(x)]
#define weight(y,x) weight[(y)*NT+(x)]

#define vxcum(y,x) vxcum[(y)*NT+(x)]
#define vycum(y,x) vycum[(y)*NT+(x)]

#define u_in(z,y,x) u_in[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define pi_in(z,y,x) pi_in[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define uL(z,y,x) uL[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define piL(z,y,x) piL[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define taus(z,y,x) taus[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define tausipjp(z,y,x) tausipjp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define tausjpkp(z,y,x) tausjpkp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define tausipkp(z,y,x) tausipkp[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]
#define taup(z,y,x) taup[(x)*m.NY*m.NZ+(y)*m.NZ+(z)]


#define PI (3.141592653589793238462643383279502884197169)

void reduce_seis(float * varloc, float * varglob, int nrec, float * rec_pos, float dh, int NX0, int NX, int NT ){
    
    int posx, i, j;
    for ( i=0;i<nrec;i++){
        posx=(int)floor(rec_pos[8*i]/dh-0.5);
        if (posx>=NX0 && posx<( NX0+NX) ){
            for (j=0;j<NT;j++){
                varglob[i*NT+j]+=varloc[i*NT+j];
            }
        }
    }
    
}
int update_grid(struct modcsts * m, struct varcl ** vcl){
    /*Update operations of one iteration */
    int state=0;
    int d, i;
    
    
    for (i=0;i<m->nupdates;i++){
        
        // Updating the variables
        for (d=0;d<m->NUM_DEVICES;d++){
            // Launch the kernel on the outside grid needing communication only
            // if a neighbouring device or processing elelement exist
            if (d>0 || m->MYLOCALID>0){
                __GUARD prog_launch( &(*vcl)[d].queue,
                                       &(*vcl)[d].ups_f[i].com1);
                __GUARD prog_launch( &(*vcl)[d].queue,
                                       &(*vcl)[d].ups_f[i].fcom1_out);
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                __GUARD prog_launch( &(*vcl)[d].queue,
                                      &(*vcl)[d].ups_f[i].com2);
                __GUARD prog_launch( &(*vcl)[d].queue,
                                      &(*vcl)[d].ups_f[i].fcom2_out);
            }
            
            //Launch kernel on the interior elements
            __GUARD prog_launch( &(*vcl)[d].queue,
                                  &(*vcl)[d].ups_f[i].center);

        }
        
        //            // Communication between devices and MPI processes
        //            if (m->NUM_DEVICES>1 || m->NLOCALP>1)
        //                __GUARD comm_v(m, vcl, mloc, 0);
        
        // Send the communcated buffers to the devices
        for (d=0;d<m->NUM_DEVICES;d++){
            
            if (d>0 || m->MYLOCALID>0){
                __GUARD prog_launch( &(*vcl)[d].queue,
                                      &(*vcl)[d].ups_f[i].fcom1_in);
                __GUARD clReleaseEvent((*vcl)[d].ups_f[i].event_write1);
            }
            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                __GUARD prog_launch( &(*vcl)[d].queue,
                                      &(*vcl)[d].ups_f[i].fcom2_in);
                __GUARD clReleaseEvent((*vcl)[d].ups_f[i].event_write2);
            }
        }
        
    }

    
//    //Save the boundary if the gradient output is required
//    if (m->GRADOUT==1 && m->BACK_PROP_TYPE==1){
//        for (d=0;d<m->NUM_DEVICES;d++){
//            if (t==0){
//                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_bnd, 1, &(*mloc)[d].global_work_size_bnd, NULL, 0, NULL, &(*vcl)[d].event_bndsave);
//            }
//            else{
//                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_bnd, 1, &(*mloc)[d].global_work_size_bnd, NULL, 1, &(*vcl)[d].event_bndtransf, &(*vcl)[d].event_bndsave);
//                __GUARD clReleaseEvent((*vcl)[d].event_bndtransf);
//            }
//            
//            if (m->ND==21){
//                __GUARD clEnqueueReadBuffer( (*vcl)[d].queuecomm, (*vcl)[d].sxybnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, &(*mloc)[d].sxybnd[(*mloc)[d].NBND*t], 1, &(*vcl)[d].event_bndsave, NULL);
//                __GUARD clEnqueueReadBuffer( (*vcl)[d].queuecomm, (*vcl)[d].syzbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, &(*mloc)[d].syzbnd[(*mloc)[d].NBND*t], 0, NULL, NULL);
//                __GUARD clEnqueueReadBuffer( (*vcl)[d].queuecomm, (*vcl)[d].vybnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, &(*mloc)[d].vybnd[(*mloc)[d].NBND*t], 0, NULL, &(*vcl)[d].event_bndtransf);
//                __GUARD clReleaseEvent((*vcl)[d].event_bndsave);
//                
//            }
//            else{
//                
//                __GUARD clEnqueueReadBuffer( (*vcl)[d].queuecomm, (*vcl)[d].sxxbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, &(*mloc)[d].sxxbnd[(*mloc)[d].NBND*t], 1, &(*vcl)[d].event_bndsave, NULL);
//                __GUARD clEnqueueReadBuffer( (*vcl)[d].queuecomm, (*vcl)[d].vxbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, &(*mloc)[d].vxbnd[(*mloc)[d].NBND*t], 0, NULL, NULL);
//                if (m->ND==3){// For 3D
//                    __GUARD clEnqueueReadBuffer( (*vcl)[d].queuecomm, (*vcl)[d].syybnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, &(*mloc)[d].syybnd[(*mloc)[d].NBND*t], 0, NULL, NULL);
//                    __GUARD clEnqueueReadBuffer( (*vcl)[d].queuecomm, (*vcl)[d].sxybnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, &(*mloc)[d].sxybnd[(*mloc)[d].NBND*t], 0, NULL, NULL);
//                    __GUARD clEnqueueReadBuffer( (*vcl)[d].queuecomm, (*vcl)[d].syzbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, &(*mloc)[d].syzbnd[(*mloc)[d].NBND*t], 0, NULL, NULL);
//                    __GUARD clEnqueueReadBuffer( (*vcl)[d].queuecomm, (*vcl)[d].vybnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, &(*mloc)[d].vybnd[(*mloc)[d].NBND*t], 0, NULL, NULL);
//                }
//                __GUARD clEnqueueReadBuffer( (*vcl)[d].queuecomm, (*vcl)[d].szzbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, &(*mloc)[d].szzbnd[(*mloc)[d].NBND*t], 0, NULL, NULL);
//                __GUARD clEnqueueReadBuffer( (*vcl)[d].queuecomm, (*vcl)[d].sxzbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, &(*mloc)[d].sxzbnd[(*mloc)[d].NBND*t], 0, NULL, NULL);
//                __GUARD clEnqueueReadBuffer( (*vcl)[d].queuecomm, (*vcl)[d].vzbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, &(*mloc)[d].vzbnd[(*mloc)[d].NBND*t], 0, NULL, &(*vcl)[d].event_bndtransf);
//                __GUARD clReleaseEvent((*vcl)[d].event_bndsave);
//            }
//            
//        }
//    }
    


    return state;
}

//int update_grid_adj(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int t, size_t global_work_size_varsout){
//    /*Update operations of one iteration */
//    int state=0;
//    int d, thist;
//    
//    //Adjoint time stepping of the stress variable
//    for (d=0;d<m->NUM_DEVICES;d++){
//        
//        //Set kernel arguments for this timestep
//        {
//            __GUARD clSetKernelArg((*vcl)[d].kernel_adjs,  2, sizeof(int), &t);
//            __GUARD clSetKernelArg((*vcl)[d].kernel_adjv,  3, sizeof(int), &t);
//            if (d>0 || m->MYLOCALID>0){
//                __GUARD clSetKernelArg((*vcl)[d].kernel_adjscom1,  2, sizeof(int), &t);
//                __GUARD clSetKernelArg((*vcl)[d].kernel_adjvcom1,  3, sizeof(int), &t);
//            }
//            
//            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
//                __GUARD clSetKernelArg((*vcl)[d].kernel_adjscom2,  2, sizeof(int), &t);
//                __GUARD clSetKernelArg((*vcl)[d].kernel_adjvcom2,  3, sizeof(int), &t);
//            }
//            
//            __GUARD clSetKernelArg((*vcl)[d].kernel_residuals,  10, sizeof(int), &t);
//        }
//        
//        
//        //Update the stresses
//        {
//            if (d>0 || m->MYLOCALID>0){
//                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adjscom1, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_sizecom1, (*mloc)[d].local_work_size, 0, NULL, NULL);
//                if (m->BACK_PROP_TYPE==1){
//                    __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adj_fill_transfer_buff1_s_out, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 0, NULL, NULL);
//                }
//                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_fill_transfer_buff1_s_out, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 0, NULL, &(*vcl)[d].event_updates_com1);
//            }
//            if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
//                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adjscom2, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_sizecom2, (*mloc)[d].local_work_size, 0, NULL, NULL);
//                if (m->BACK_PROP_TYPE==1){
//                    __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adj_fill_transfer_buff2_s_out, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 0, NULL, NULL);
//                }
//                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_fill_transfer_buff2_s_out, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 0, NULL, &(*vcl)[d].event_updates_com1);
//            }
//            
//            if (m->BACK_PROP_TYPE==1){
//                if (t==m->tmax-1){
//                    __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adjs, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size, (*mloc)[d].local_work_size, 0, NULL, &(*vcl)[d].event_bndsave);
//                }
//                else{
//                    __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adjs, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size, (*mloc)[d].local_work_size, 1, &(*vcl)[d].event_bndtransf2, &(*vcl)[d].event_bndsave);
//                    if (!state) clReleaseEvent((*vcl)[d].event_bndtransf2);
//                }
//            }
//            else{
//                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adjs, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size, (*mloc)[d].local_work_size, 0, NULL, NULL);
//            }
//        }
//        
//    }
//    
//    //Transfer the boundary variables at time step t
//    if (m->BACK_PROP_TYPE==1){
//        for (d=0;d<m->NUM_DEVICES;d++){
//            
//            
//            if (m->ND==21){
//                if (t==m->tmax-1){
//                    __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].vybnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].vybnd[(*mloc)[d].NBND*(t-1)], 0, NULL, &(*vcl)[d].event_bndtransf);
//                }
//                else {
//                    __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].vybnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].vybnd[(*mloc)[d].NBND*(t-1)], 1, &(*vcl)[d].event_bndsave2, &(*vcl)[d].event_bndtransf);
//                }
//                if (t!=m->tmax-1){
//                    if (!state) clReleaseEvent((*vcl)[d].event_bndsave2);
//                }
//                
//                __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].sxybnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].sxybnd[(*mloc)[d].NBND*(t-1)], 1, &(*vcl)[d].event_bndsave, NULL);
//                __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].syzbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].syzbnd[(*mloc)[d].NBND*(t-1)], 0, NULL, &(*vcl)[d].event_bndtransf2);
//                if (!state) clReleaseEvent((*vcl)[d].event_bndsave);
//            }
//            else{
//                
//                
//                
//                if (t==m->tmax-1){
//                    __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].vxbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].vxbnd[(*mloc)[d].NBND*(t-1)], 0, NULL, NULL);
//                }
//                else {
//                    __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].vxbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].vxbnd[(*mloc)[d].NBND*(t-1)], 1, &(*vcl)[d].event_bndsave2, NULL);
//                }
//                if (m->ND==3){// For 3D
//                    __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].vybnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].vybnd[(*mloc)[d].NBND*(t-1)], 0, NULL, NULL);
//                }
//                __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].vzbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].vzbnd[(*mloc)[d].NBND*(t-1)], 0, NULL, &(*vcl)[d].event_bndtransf);
//                if (t!=m->tmax-1){
//                    if (!state) clReleaseEvent((*vcl)[d].event_bndsave2);
//                }
//                
//                __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].sxxbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].sxxbnd[(*mloc)[d].NBND*(t-1)], 1, &(*vcl)[d].event_bndsave, NULL);
//                if (m->ND==3){// For 3D
//                    __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].syybnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].syybnd[(*mloc)[d].NBND*(t-1)], 0, NULL, NULL);
//                    __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].sxybnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].sxybnd[(*mloc)[d].NBND*(t-1)], 0, NULL, NULL);
//                    __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].syzbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].syzbnd[(*mloc)[d].NBND*(t-1)], 0, NULL, NULL);
//                    
//                }
//                __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].szzbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].szzbnd[(*mloc)[d].NBND*(t-1)], 0, NULL, NULL);
//                __GUARD clEnqueueWriteBuffer( (*vcl)[d].queuecomm, (*vcl)[d].sxzbnd, CL_FALSE, 0, (*vcl)[d].buffer_size_bnd, (void*)&(*mloc)[d].sxzbnd[(*mloc)[d].NBND*(t-1)], 0, NULL, &(*vcl)[d].event_bndtransf2);
//                if (!state) clReleaseEvent((*vcl)[d].event_bndsave);
//                
//            }
//            
//        }
//    }
//    
//    // Communicating the stress variables between GPUs
//    if (m->NUM_DEVICES>1 || m->NLOCALP>1)
//        __GUARD comm_s(m, vcl, mloc, 1);
//    
//    for (d=0;d<m->NUM_DEVICES;d++){
//        if (d>0 || m->MYLOCALID>0){
//            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_fill_transfer_buff1_s_in, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 1, &(*vcl)[d].event_writes1, NULL);
//            __GUARD clReleaseEvent((*vcl)[d].event_writes1);
//            if (m->BACK_PROP_TYPE==1){
//                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adj_fill_transfer_buff1_s_in, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 0, NULL, NULL);
//            }
//        }
//        if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
//            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_fill_transfer_buff2_s_in, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 1, &(*vcl)[d].event_writes2, NULL);
//            __GUARD clReleaseEvent((*vcl)[d].event_writes2);
//            if (m->BACK_PROP_TYPE==1){
//                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adj_fill_transfer_buff2_s_in, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 0, NULL, NULL);
//            }
//        }
//    }
//    
//    
//    // Adjoint time stepping of the velocity variables
//    for (d=0;d<m->NUM_DEVICES;d++){
//        
//        if (d>0 || m->MYLOCALID>0){
//            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adjvcom1, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_sizecom1, (*mloc)[d].local_work_size, 0, NULL, NULL);
//            if (m->BACK_PROP_TYPE==1){
//                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adj_fill_transfer_buff1_v_out, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 0, NULL, NULL);
//            }
//            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_fill_transfer_buff1_v_out, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 0, NULL, &(*vcl)[d].event_updatev_com1);
//        }
//        if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
//            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adjvcom2, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_sizecom2, (*mloc)[d].local_work_size, 0, NULL, NULL);
//            if (m->BACK_PROP_TYPE==1){
//                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adj_fill_transfer_buff2_v_out, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 0, NULL, NULL);
//            }
//            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_fill_transfer_buff2_v_out, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 0, NULL, &(*vcl)[d].event_updatev_com1);
//        }
//        
//        if (m->BACK_PROP_TYPE==1){
//            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adjv, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size, (*mloc)[d].local_work_size, 1, &(*vcl)[d].event_bndtransf, &(*vcl)[d].event_bndsave2);
//            if (!state) clReleaseEvent((*vcl)[d].event_bndtransf);
//            
//        }
//        else{
//            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adjv, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size, (*mloc)[d].local_work_size, 0, NULL, NULL);
//        }
//        __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_residuals, 1, &global_work_size_varsout, NULL, 0, NULL, NULL);
//    }
//    
//    // Communicating the velocity variables between GPUs
//    if (m->NUM_DEVICES>1 || m->NLOCALP>1)
//        __GUARD comm_v(m, vcl, mloc, 1);
//    for (d=0;d<m->NUM_DEVICES;d++){
//        if (d>0 || m->MYLOCALID>0){
//            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_fill_transfer_buff1_v_in, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 1, &(*vcl)[d].event_writev1, NULL);
//            __GUARD clReleaseEvent((*vcl)[d].event_writev1);
//            if (m->BACK_PROP_TYPE==1){
//                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adj_fill_transfer_buff1_v_in, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 0, NULL, NULL);
//            }
//        }
//        if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
//            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_fill_transfer_buff2_v_in, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 1, &(*vcl)[d].event_writev2, NULL);
//            __GUARD clReleaseEvent((*vcl)[d].event_writev2);
//            if (m->BACK_PROP_TYPE==1){
//                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_adj_fill_transfer_buff2_v_in, (*vcl)[d].dims.workdim, (*mloc)[d].global_work_size_fcom, NULL, 0, NULL, NULL);
//            }
//        }
//    }
//    
//    //Save the selected residual wavefield frequencies
//    if (m->BACK_PROP_TYPE==2 && (t-m->tmin)%m->DTNYQ==0){
//        for (d=0;d<m->NUM_DEVICES;d++){
//            thist=(t-m->tmin)/m->DTNYQ;
//            __GUARD clSetKernelArg((*vcl)[d].kernel_savefreqs,  31, sizeof(int), &thist);
//            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_savefreqs, 1, &(*mloc)[d].global_work_size_f, NULL, 0, NULL, NULL);
//        }
//    }
//
//    
//    return state;
//}

int initialize_grid(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc, int s){
    /*Initialize the buffers to 0 before the first time step of each shot*/
    int state=0;
    int d;
    size_t buffer_size_thiss=0;
    size_t buffer_size_thisns=0;
    size_t buffer_size_nrec=0;
    size_t global_work_size_varsout=0;
    
    // Buffer size for this shot
    if (!state){
        buffer_size_thiss  = sizeof(float) * m->NT * m->nsrc[s];
        buffer_size_thisns = sizeof(float) * 5 * m->nsrc[s];
        buffer_size_nrec = sizeof(float) * 8 * m->nrec[s];
    }
    // Initialization of the seismic variables
    for (d=0;d<m->NUM_DEVICES;d++){
        
        
        __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_initseis, 1, &(*mloc)[d].global_work_size_initfd, NULL, 0, NULL, NULL);
        if (!state) global_work_size_varsout= m->NT*m->nrec[s];
        __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_varsoutinit, 1, &global_work_size_varsout , NULL, 0, NULL, NULL);
        if (!state) global_work_size_varsout= m->nrec[s];
        /* Global size for the varsout kernel */
        
        //if (m->fmax || m->fmin) butterworth(m->src[s], m->fmin, m->fmax, m->dt, m->NT, 1, 6);
        __GUARD clbuf_send(&(*vcl)[d].queue,  buffer_size_thiss,  &(*vcl)[d].src,     m->src[s]);
        __GUARD clbuf_send(&(*vcl)[d].queue,  buffer_size_thisns, &(*vcl)[d].src_pos, m->src_pos[s]);
        __GUARD clbuf_send(&(*vcl)[d].queue,  buffer_size_nrec, &(*vcl)[d].rec_pos,   m->rec_pos[s]);
        
        
        __GUARD clSetKernelArg((*vcl)[d].kernel_s,  1, sizeof(int), &m->nsrc[s]);
        __GUARD clSetKernelArg((*vcl)[d].kernel_v,  1, sizeof(int), &m->nsrc[s]);
        if (d>0 || m->MYLOCALID>0){
            __GUARD clSetKernelArg((*vcl)[d].kernel_scom1,  1, sizeof(int), &m->nsrc[s]);
            __GUARD clSetKernelArg((*vcl)[d].kernel_vcom1,  1, sizeof(int), &m->nsrc[s]);
        }
        if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
            __GUARD clSetKernelArg((*vcl)[d].kernel_scom2,  1, sizeof(int), &m->nsrc[s]);
            __GUARD clSetKernelArg((*vcl)[d].kernel_vcom2,  1, sizeof(int), &m->nsrc[s]);
        }
        
        if (m->GRADOUT==1 && m->BACK_PROP_TYPE==2){
            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_initsavefreqs, 1, &(*mloc)[d].global_work_size_f, NULL, 0, NULL, NULL);
        }
        
        
    }
    
    return state;
}


int time_stepping(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc) {
    
    int state=0;

    size_t buffer_size_thiss=0;
    size_t buffer_size_thisns=0;
    size_t buffer_size_nrec=0;
    size_t buffer_size_thisvarsout=0;
    size_t global_work_size_varsout=0;


    
    int t,s,i,d,j, thist,k;
    int posx;
    

    // Calculate what shots belong to the group this processing element belongs to
    if (!state){
        m->smin=0;
        m->smax=0;
        
        for (i=0;i<m->MYGROUPID;i++){
            if (i<m->ns%m->NGROUP){
                m->smin+=(m->ns/m->NGROUP+1);
            }
            else{
                m->smin+=(m->ns/m->NGROUP);
            }
            
        }
        if (m->MYGROUPID<m->ns%m->NGROUP){
            m->smax=m->smin+(m->ns/m->NGROUP+1);
        }
        else{
            m->smax=m->smin+(m->ns/m->NGROUP);
        }
    }
    
    // Initialize the gradient buffers before time stepping
    if (m->GRADOUT==1 && m->BACK_PROP_TYPE==1){
        for (d=0;d<m->NUM_DEVICES;d++){
            __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_initgrad, 1, &(*mloc)[d].global_work_size_init, NULL, 0, NULL, NULL);
        }
    }
    
    // Main loop over shots of this group
    for (s=m->smin;s<m->smax;s++){
        
        // Buffer size for this shot
        if (!state){
            buffer_size_thiss  = sizeof(float) * m->NT * m->nsrc[s];
            buffer_size_thisns = sizeof(float) * 5 * m->nsrc[s];
            buffer_size_nrec = sizeof(float) * 8 * m->nrec[s];
            global_work_size_varsout= m->nrec[s];
        }
        
        // Initialization of the seismic variables
        __GUARD initialize_grid(m, vcl, mloc, s);
        
        // Loop for seismic propagation
        for (t=0;t<m->tmax; t++){
            
            //Assign the time step to kernels
            for (d=0;d<m->NUM_DEVICES;d++){
                for (i=0;i<(*vcl)[d].nprogs;i++){
                    if ((*vcl)[d].progs[i].tinput>0)
                        __GUARD clSetKernelArg((*vcl)[d].progs[i].kernel,
                                               (*vcl)[d].progs[i].tinput-1,
                                               sizeof(int), &t);
                }
            }
            
            //Save the selected frequency if the gradient is to be obtained by DFT
            if (m->GRADOUT==1 && m->BACK_PROP_TYPE==2 && t>=m->tmin && (t-m->tmin)%m->DTNYQ==0){
                for (d=0;d<m->NUM_DEVICES;d++){
                    thist=(t-m->tmin)/m->DTNYQ;
                    __GUARD clSetKernelArg((*vcl)[d].kernel_savefreqs,  31, sizeof(int), &(thist));
                    __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_savefreqs, 1, &(*mloc)[d].global_work_size_f, NULL, 0, NULL, NULL);
                }
            }
            
            update_grid(m, vcl, mloc, t);
            
            // Computing the free surface
            if (m->FREESURF==1){
                for (d=0;d<m->NUM_DEVICES;d++){
                    if (m->ND==3){// For 3D
                        __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_surf, 2, (*mloc)[d].global_work_size_surf, NULL, 0, NULL, NULL);
                    }
                    else{
                        __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_surf, 1, (*mloc)[d].global_work_size_surf, NULL, 0, NULL, NULL);
                    }
                }
            }
            
            // Outputting seismograms
            for (d=0;d<m->NUM_DEVICES;d++){
                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_varsout, 1, &global_work_size_varsout, NULL, 0, NULL, NULL);
            }

            // Outputting the movie
            if (m->MOVOUT>0 && (t+1)%m->MOVOUT==0 && state==0){
                for (d=0;d<m->NUM_DEVICES;d++){
                    if (m->ND!=21){
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_fd, &(*vcl)[d].vx, (*mloc)[d].buffermovvx);
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_fd, &(*vcl)[d].vz, (*mloc)[d].buffermovvz);
                    }
                    if (m->ND==3 || m->ND==21){
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_fd, &(*vcl)[d].vy, (*mloc)[d].buffermovvy);
                    }
                }
                
                for (d=0;d<m->NUM_DEVICES;d++){
                    clFinish((*vcl)[d].queue);
                    

                    if (m->ND==3){
                        for (i=0;i<(*mloc)[d].NX;i++){
                            for (j=0;j<(*mloc)[d].NY;j++){
                                for (k=0;k<(*mloc)[d].NZ;k++){
                                    (*mloc)[d].movvx[s*m->NT/m->MOVOUT*m->NX*m->NY*m->NZ+ ((t+1)/m->MOVOUT-1)*m->NX*m->NY*m->NZ+ i*m->NY*m->NZ+ j*m->NZ +k]=(*mloc)[d].buffermovvx[ (i+m->FDOH)*(m->NY+m->FDORDER)*(m->NZ+m->FDORDER)+ (j+m->FDOH)*(m->NZ+m->FDORDER) +(k+m->FDOH)];
                                }
                            }
                        }
                        for (i=0;i<(*mloc)[d].NX;i++){
                            for (j=0;j<(*mloc)[d].NY;j++){
                                for (k=0;k<(*mloc)[d].NZ;k++){
                                    (*mloc)[d].movvz[s*m->NT/m->MOVOUT*m->NX*m->NY*m->NZ+ ((t+1)/m->MOVOUT-1)*m->NX*m->NY*m->NZ+ i*m->NY*m->NZ+ j*m->NZ +k]=(*mloc)[d].buffermovvz[ (i+m->FDOH)*(m->NY+m->FDORDER)*(m->NZ+m->FDORDER)+ (j+m->FDOH)*(m->NZ+m->FDORDER) +(k+m->FDOH)];
                                }
                            }
                        }
                        for (i=0;i<(*mloc)[d].NX;i++){
                            for (j=0;j<(*mloc)[d].NY;j++){
                                for (k=0;k<(*mloc)[d].NZ;k++){
                                    (*mloc)[d].movvy[s*m->NT/m->MOVOUT*m->NX*m->NY*m->NZ+ ((t+1)/m->MOVOUT-1)*m->NX*m->NY*m->NZ+ i*m->NY*m->NZ+ j*m->NZ +k]=(*mloc)[d].buffermovvy[ (i+m->FDOH)*(m->NY+m->FDORDER)*(m->NZ+m->FDORDER)+ (j+m->FDOH)*(m->NZ+m->FDORDER) +(k+m->FDOH)];
                                }
                            }
                        }

                    }
                    else if (m->ND==2){
                        for (i=0;i<(*mloc)[d].NX;i++){
                                for (k=0;k<(*mloc)[d].NZ;k++){
                                    (*mloc)[d].movvx[s*m->NT/m->MOVOUT*m->NX*m->NZ+ ((t+1)/m->MOVOUT-1)*m->NX*m->NZ+ i*m->NZ+ k]=(*mloc)[d].buffermovvx[ (i+m->FDOH)*(m->NZ+m->FDORDER)+(k+m->FDOH)];
                                }
                        }
                        for (i=0;i<(*mloc)[d].NX;i++){
                            for (k=0;k<(*mloc)[d].NZ;k++){
                                (*mloc)[d].movvz[s*m->NT/m->MOVOUT*m->NX*m->NZ+ ((t+1)/m->MOVOUT-1)*m->NX*m->NZ+ i*m->NZ+ k]=(*mloc)[d].buffermovvz[ (i+m->FDOH)*(m->NZ+m->FDORDER)+(k+m->FDOH)];
                            }
                        }
                    }
                    else if (m->ND==21 ){
                        for (i=0;i<(*mloc)[d].NX;i++){
                            for (k=0;k<(*mloc)[d].NZ;k++){
                                (*mloc)[d].movvy[s*m->NT/m->MOVOUT*m->NX*m->NZ+ ((t+1)/m->MOVOUT-1)*m->NX*m->NZ+ i*m->NZ+ k]=(*mloc)[d].buffermovvy[ (i+m->FDOH)*(m->NZ+m->FDORDER)+(k+m->FDOH)];
                            }
                        }
                    }
                    
                }
                
                
                
                
            }

            // Flush all the previous command to the computing device
            for (d=0;d<m->NUM_DEVICES;d++){
                if (d>0 || d<m->NUM_DEVICES-1)
                    __GUARD clFlush((*vcl)[d].queuecomm);
                __GUARD clFlush((*vcl)[d].queue);
            }
            
        }
        

        //Realease events that have not been released
        for (d=0;d<m->NUM_DEVICES;d++){

            if (m->GRADOUT==1 && m->BACK_PROP_TYPE==1){
                __GUARD clReleaseEvent((*vcl)[d].event_bndtransf);
            }
        }
        
        // Transfer the seismogram from GPUs to host
        if (!state) buffer_size_thisvarsout = sizeof(float) * m->NT * m->nrec[s];
        for (d=0;d<m->NUM_DEVICES;d++){
            if (m->bcastvx) {
                __GUARD clbuf_read( &(*vcl)[d].queue, buffer_size_thisvarsout, &(*vcl)[d].vxout, (*mloc)[d].vxout[s]);
            }
            if (m->bcastvy) {
                __GUARD clbuf_read( &(*vcl)[d].queue, buffer_size_thisvarsout, &(*vcl)[d].vyout, (*mloc)[d].vyout[s]);
            }
            if (m->bcastvz) {
                __GUARD clbuf_read( &(*vcl)[d].queue, buffer_size_thisvarsout, &(*vcl)[d].vzout, (*mloc)[d].vzout[s]);
            }
            if (m->bcastsxx) {
                __GUARD clbuf_read( &(*vcl)[d].queue, buffer_size_thisvarsout, &(*vcl)[d].sxxout, (*mloc)[d].sxxout[s]);
            }
            if (m->bcastsyy) {
                __GUARD clbuf_read( &(*vcl)[d].queue, buffer_size_thisvarsout, &(*vcl)[d].syyout, (*mloc)[d].syyout[s]);
            }
            if (m->bcastszz) {
                __GUARD clbuf_read( &(*vcl)[d].queue, buffer_size_thisvarsout, &(*vcl)[d].szzout, (*mloc)[d].szzout[s]);
            }
            if (m->bcastsxy) {
                __GUARD clbuf_read( &(*vcl)[d].queue, buffer_size_thisvarsout, &(*vcl)[d].sxyout, (*mloc)[d].sxyout[s]);
            }
            if (m->bcastsxz) {
                __GUARD clbuf_read( &(*vcl)[d].queue, buffer_size_thisvarsout, &(*vcl)[d].sxzout, (*mloc)[d].sxzout[s]);
            }
            if (m->bcastsyz) {
                __GUARD clbuf_read( &(*vcl)[d].queue, buffer_size_thisvarsout, &(*vcl)[d].syzout, (*mloc)[d].syzout[s]);
            }
            if (m->bcastp) {
                __GUARD clbuf_read( &(*vcl)[d].queue, buffer_size_thisvarsout, &(*vcl)[d].pout, (*mloc)[d].pout[s]);
            }
        }
        
        // Aggregate the seismograms in the output variable
        if (!state){
            for (d=0;d<m->NUM_DEVICES;d++){
                __GUARD clFinish((*vcl)[d].queue);
                if (m->bcastvx){
                    reduce_seis((*mloc)[d].vxout[s], m->vxout[s], m->nrec[s], m->rec_pos[s], m->dh, (*mloc)[d].NX0, (*mloc)[d].NX, m->NT);
                }
                if (m->bcastvy){
                    reduce_seis((*mloc)[d].vyout[s], m->vyout[s], m->nrec[s], m->rec_pos[s], m->dh, (*mloc)[d].NX0, (*mloc)[d].NX, m->NT);
                }
                if (m->bcastvz){
                    reduce_seis((*mloc)[d].vzout[s], m->vzout[s], m->nrec[s], m->rec_pos[s], m->dh, (*mloc)[d].NX0, (*mloc)[d].NX, m->NT);
                }
                if (m->bcastsxx){
                    reduce_seis((*mloc)[d].sxxout[s], m->sxxout[s], m->nrec[s], m->rec_pos[s], m->dh, (*mloc)[d].NX0, (*mloc)[d].NX, m->NT);
                }
                if (m->bcastsyy){
                    reduce_seis((*mloc)[d].syyout[s], m->syyout[s], m->nrec[s], m->rec_pos[s], m->dh, (*mloc)[d].NX0, (*mloc)[d].NX, m->NT);
                }
                if (m->bcastszz){
                    reduce_seis((*mloc)[d].szzout[s], m->szzout[s], m->nrec[s], m->rec_pos[s], m->dh, (*mloc)[d].NX0, (*mloc)[d].NX, m->NT);
                }
                if (m->bcastsxy){
                    reduce_seis((*mloc)[d].sxyout[s], m->sxyout[s], m->nrec[s], m->rec_pos[s], m->dh, (*mloc)[d].NX0, (*mloc)[d].NX, m->NT);
                }
                if (m->bcastsxz){
                    reduce_seis((*mloc)[d].sxzout[s], m->sxzout[s], m->nrec[s], m->rec_pos[s], m->dh, (*mloc)[d].NX0, (*mloc)[d].NX, m->NT);
                }
                if (m->bcastsyz){
                    reduce_seis((*mloc)[d].syzout[s], m->syzout[s], m->nrec[s], m->rec_pos[s], m->dh, (*mloc)[d].NX0, (*mloc)[d].NX, m->NT);
                }
                if (m->bcastp){
                    reduce_seis((*mloc)[d].pout[s], m->pout[s], m->nrec[s], m->rec_pos[s], m->dh, (*mloc)[d].NX0, (*mloc)[d].NX, m->NT);
                }
            }
        }

        
        //Calculate the residual
        if (m->GRADOUT || m->RMSOUT || m->RESOUT){
            __GUARD m->res_calc(m,s);
        }
        

        
        // Calculation of the gradient for this shot
        if (m->GRADOUT==1){
            
            // Initialize the backpropagation and gradient. Transfer the residual to GPUs
            for (d=0;d<m->NUM_DEVICES;d++){
               
                if (!state) if (m->vx0) state = clbuf_send(&(*vcl)[d].queue,  buffer_size_thisvarsout, &(*vcl)[d].vxout, m->rx[s]);
                if (!state) if (m->vz0) state = clbuf_send(&(*vcl)[d].queue,  buffer_size_thisvarsout, &(*vcl)[d].vzout, m->rz[s]);
                if (!state) if (m->vy0) state = clbuf_send(&(*vcl)[d].queue,  buffer_size_thisvarsout, &(*vcl)[d].vyout, m->ry[s]);
                
                __GUARD clSetKernelArg((*vcl)[d].kernel_adjs,  1, sizeof(int), &m->nsrc[s]);
                __GUARD clSetKernelArg((*vcl)[d].kernel_adjv,  1, sizeof(int), &m->nsrc[s]);
                __GUARD clSetKernelArg((*vcl)[d].kernel_adjv,  2, sizeof(int), &m->nrec[s]);
                
                if (d>0 || m->MYLOCALID>0){
                    __GUARD clSetKernelArg((*vcl)[d].kernel_adjscom1,  1, sizeof(int), &m->nsrc[s]);
                    __GUARD clSetKernelArg((*vcl)[d].kernel_adjvcom1,  1, sizeof(int), &m->nsrc[s]);
                    __GUARD clSetKernelArg((*vcl)[d].kernel_adjvcom1,  2, sizeof(int), &m->nrec[s]);
                }
                if (d<m->NUM_DEVICES-1 || m->MYLOCALID<m->NLOCALP-1){
                    __GUARD clSetKernelArg((*vcl)[d].kernel_adjscom2,  1, sizeof(int), &m->nsrc[s]);
                    __GUARD clSetKernelArg((*vcl)[d].kernel_adjvcom2,  1, sizeof(int), &m->nsrc[s]);
                    __GUARD clSetKernelArg((*vcl)[d].kernel_adjvcom2,  2, sizeof(int), &m->nrec[s]);
                }
                
                
                __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_initseis_r, 1, &(*mloc)[d].global_work_size_initfd, NULL, 0, NULL, NULL);
                
                if (m->GRADSRCOUT==1){
                    __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_initialize_gradsrc, 1, &(*mloc)[d].global_work_size_gradsrc, NULL, 0, NULL, NULL);
                }
                
                if (m->BACK_PROP_TYPE==2){
                    if (m->ND!=21){
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_vx, (*mloc)[d].f_vx);
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_vz, (*mloc)[d].f_vz);
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_sxx, (*mloc)[d].f_sxx);
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_szz, (*mloc)[d].f_szz);
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_sxz, (*mloc)[d].f_sxz);
                    }
                
                    if (m->ND==3 || m->ND==21){
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_vy, (*mloc)[d].f_vy);
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_sxy, (*mloc)[d].f_sxy);
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_syz, (*mloc)[d].f_syz);
                    }
                    if (m->ND==3){
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_syy, (*mloc)[d].f_syy);
                    }
                    
                    if (m->L>0){
                        if (m->ND!=21){
                            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_rxx, (*mloc)[d].f_rxx);
                            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_rzz, (*mloc)[d].f_rzz);
                            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_rxz, (*mloc)[d].f_rxz);
                        }
                        
                        if (m->ND==3 || m->ND==21){
                            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_rxy, (*mloc)[d].f_rxy);
                            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_ryz, (*mloc)[d].f_ryz);
                        }
                        if (m->ND==3){
                            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_ryy, (*mloc)[d].f_ryy);
                        }
                        
                    }
                    
                    __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_initsavefreqs, 1, &(*mloc)[d].global_work_size_f, NULL, 0, NULL, NULL);
                    __GUARD prog_launch( &(*vcl)[d].queue, &(*vcl)[d].kernel_initseis, 1, &(*mloc)[d].global_work_size_initfd, NULL, 0, NULL, NULL);
                }
                
                
                
            }

            // Inverse time stepping
            for (t=m->tmax-1;t>=m->tmin; t--){

                __GUARD update_grid_adj(m, vcl, mloc, t, global_work_size_varsout);
                
                for (d=0;d<m->NUM_DEVICES;d++){
                    if (d>0 || d<m->NUM_DEVICES-1)
                        __GUARD clFlush((*vcl)[d].queuecomm);
                    __GUARD clFlush((*vcl)[d].queue);
                }
            }
            
            
            //Realease events that have not been released
            for (d=0;d<m->NUM_DEVICES;d++){
                if (d>0){
                    __GUARD clReleaseEvent((*vcl)[d].event_writev1);
                }
                if (d<m->NUM_DEVICES-1){
                    __GUARD clReleaseEvent((*vcl)[d].event_writev2);
                }
                if (m->GRADOUT==1 && m->BACK_PROP_TYPE==1){
                    __GUARD clReleaseEvent((*vcl)[d].event_bndtransf2);
                    __GUARD clReleaseEvent((*vcl)[d].event_bndsave2);
                }
            }
            
            if (m->GRADSRCOUT==1){
                for (d=0;d<m->NUM_DEVICES;d++){
                    __GUARD clbuf_read( &(*vcl)[d].queue, buffer_size_thiss, &(*vcl)[d].gradsrc, m->gradsrc[s]);
                }
            }
            // Transfer the gradient from GPUs to host
            if (m->BACK_PROP_TYPE==2){
                for (d=0;d<m->NUM_DEVICES;d++){
                    
                    if (m->ND!=21){
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_vx, (*mloc)[d].f_vxr);
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_vz, (*mloc)[d].f_vzr);
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_sxx, (*mloc)[d].f_sxxr);
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_szz, (*mloc)[d].f_szzr);
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_sxz, (*mloc)[d].f_sxzr);
                    }
                    
                    if (m->ND==3 || m->ND==21){
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_vy, (*mloc)[d].f_vyr);
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_sxy, (*mloc)[d].f_sxyr);
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_syz, (*mloc)[d].f_syzr);
                    }
                    if (m->ND==3){
                        __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc, &(*vcl)[d].f_syy, (*mloc)[d].f_syyr);
                    }
                    if (m->L>0){
                        if (m->ND!=21){
                            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_rxx, (*mloc)[d].f_rxxr);
                            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_rzz, (*mloc)[d].f_rzzr);
                            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_rxz, (*mloc)[d].f_rxzr);
                        }
                        
                        if (m->ND==3 || m->ND==21){
                            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_rxy, (*mloc)[d].f_rxyr);
                            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_ryz, (*mloc)[d].f_ryzr);
                        }
                        
                        if (m->ND==3){
                            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_modelc*m->L, &(*vcl)[d].f_ryy, (*mloc)[d].f_ryyr);
                        }
                    }
                }

                for (d=0;d<m->NUM_DEVICES;d++){
                    __GUARD clFinish((*vcl)[d].queue);
                    if (!state) calc_grad(m, &(*mloc)[d]);
                }
            }
            
            
        }
        

    }
    
    if (m->GRADOUT==1 && m->BACK_PROP_TYPE==1){
        for (d=0;d<m->NUM_DEVICES;d++){
            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].gradrho, (*mloc)[d].gradrho);
            if (m->ND!=21){
                __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].gradM, (*mloc)[d].gradM);
            }
            __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].gradmu, (*mloc)[d].gradmu);
            
        }
        
        if (m->HOUT==1){
            for (d=0;d<m->NUM_DEVICES;d++){
                __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].Hrho, (*mloc)[d].Hrho);
                if (m->ND!=21){
                    __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].HM, (*mloc)[d].HM);
                }
                __GUARD clbuf_read( &(*vcl)[d].queue, (*vcl)[d].buffer_size_model, &(*vcl)[d].Hmu, (*mloc)[d].Hmu);
                
            }
        }
        
        for (d=0;d<m->NUM_DEVICES;d++){
            __GUARD clFinish((*vcl)[d].queue);
        }
    }
    
    
    if (state) fprintf(stderr,"%s\n",cl_err_code(state));
    if (state && m->MPI_INIT==1) MPI_Bcast( &state, 1, MPI_INT, m->MYID, MPI_COMM_WORLD );
    
    
    return state;
}

