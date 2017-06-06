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

// Free all memory for local grids and OpenCL devices
#include "F.h"



int Free_OpenCL(struct modcsts * m, struct varcl ** vcl, struct modcstsloc ** mloc)  {
    
    int d;
    // Memory freeing for GPU buffers and host memory
    for (d=0;d<m->num_devices;d++){
        
        if ((*mloc)){
            
            
            if  ((*mloc)[d].vxout){
                GFree((*mloc)[d].vxout[0])
            }
            if  ((*mloc)[d].vyout){
                GFree((*mloc)[d].vyout[0])
            }
            if  ((*mloc)[d].vzout){
                GFree((*mloc)[d].vzout[0])
            }
            if ((*mloc)[d].vxout) free((*mloc)[d].vxout);
            if ((*mloc)[d].vyout) free((*mloc)[d].vyout);
            if ((*mloc)[d].vzout) free((*mloc)[d].vzout);
            
            GFree((*mloc)[d].f_vx);
            GFree((*mloc)[d].f_vy);
            GFree((*mloc)[d].f_vz);
            GFree((*mloc)[d].f_sxx);
            GFree((*mloc)[d].f_syy);
            GFree((*mloc)[d].f_szz);
            GFree((*mloc)[d].f_sxy);
            GFree((*mloc)[d].f_sxz);
            GFree((*mloc)[d].f_syz);
            GFree((*mloc)[d].f_rxx);
            GFree((*mloc)[d].f_ryy);
            GFree((*mloc)[d].f_rzz);
            GFree((*mloc)[d].f_rxy);
            GFree((*mloc)[d].f_rxz);
            GFree((*mloc)[d].f_ryz);
            GFree((*mloc)[d].f_vxr);
            GFree((*mloc)[d].f_vyr);
            GFree((*mloc)[d].f_vzr);
            GFree((*mloc)[d].f_sxxr);
            GFree((*mloc)[d].f_syyr);
            GFree((*mloc)[d].f_szzr);
            GFree((*mloc)[d].f_sxyr);
            GFree((*mloc)[d].f_sxzr);
            GFree((*mloc)[d].f_syzr);
            GFree((*mloc)[d].f_rxxr);
            GFree((*mloc)[d].f_ryyr);
            GFree((*mloc)[d].f_rzzr);
            GFree((*mloc)[d].f_rxyr);
            GFree((*mloc)[d].f_rxzr);
            GFree((*mloc)[d].f_ryzr);
            
            GFree((*mloc)[d].buffermovvx);
            GFree((*mloc)[d].buffermovvy);
            GFree((*mloc)[d].buffermovvz);
            
            
            if ((*mloc)[d].sxx_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxx_sub1, (*mloc)[d].sxx_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].syy_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].syy_sub1, (*mloc)[d].syy_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].szz_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].szz_sub1, (*mloc)[d].szz_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].sxy_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxy_sub1, (*mloc)[d].sxy_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].syz_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].syz_sub1, (*mloc)[d].syz_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].sxz_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxz_sub1, (*mloc)[d].sxz_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].vx_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vx_sub1, (*mloc)[d].vx_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].vy_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vy_sub1, (*mloc)[d].vy_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].vz_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vz_sub1, (*mloc)[d].vz_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].sxx_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxx_sub2, (*mloc)[d].sxx_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].syy_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].syy_sub2, (*mloc)[d].syy_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].szz_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].szz_sub2, (*mloc)[d].szz_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].sxy_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxy_sub2, (*mloc)[d].sxy_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].syz_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].syz_sub2, (*mloc)[d].syz_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].sxz_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxz_sub2, (*mloc)[d].sxz_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].vx_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vx_sub2, (*mloc)[d].vx_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].vy_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vy_sub2, (*mloc)[d].vy_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].vz_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vz_sub2, (*mloc)[d].vz_sub2, 0 , NULL , NULL );
            
            if ((*mloc)[d].sxx_r_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxx_r_sub1, (*mloc)[d].sxx_r_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].syy_r_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].syy_r_sub1, (*mloc)[d].syy_r_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].szz_r_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].szz_r_sub1, (*mloc)[d].szz_r_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].sxy_r_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxy_r_sub1, (*mloc)[d].sxy_r_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].syz_r_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].syz_r_sub1, (*mloc)[d].syz_r_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].sxz_r_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxz_r_sub1, (*mloc)[d].sxz_r_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].vx_r_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vx_r_sub1, (*mloc)[d].vx_r_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].vy_r_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vy_r_sub1, (*mloc)[d].vy_r_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].vz_r_sub1) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vz_r_sub1, (*mloc)[d].vz_r_sub1, 0 , NULL , NULL );
            if ((*mloc)[d].sxx_r_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxx_r_sub2, (*mloc)[d].sxx_r_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].syy_r_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].syy_r_sub2, (*mloc)[d].syy_r_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].szz_r_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].szz_r_sub2, (*mloc)[d].szz_r_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].sxy_r_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxy_r_sub2, (*mloc)[d].sxy_r_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].syz_r_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].syz_r_sub2, (*mloc)[d].syz_r_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].sxz_r_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxz_r_sub2, (*mloc)[d].sxz_r_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].vx_r_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vx_r_sub2, (*mloc)[d].vx_r_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].vy_r_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vy_r_sub2, (*mloc)[d].vy_r_sub2, 0 , NULL , NULL );
            if ((*mloc)[d].vz_r_sub2) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vz_r_sub2, (*mloc)[d].vz_r_sub2, 0 , NULL , NULL );
            
            
            
            if ((*mloc)[d].sxxbnd) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxxbnd_pin, (*mloc)[d].sxxbnd, 0 , NULL , NULL );
            if ((*mloc)[d].syybnd) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].syybnd_pin, (*mloc)[d].syybnd, 0 , NULL , NULL );
            if ((*mloc)[d].szzbnd) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].szzbnd_pin, (*mloc)[d].szzbnd, 0 , NULL , NULL );
            if ((*mloc)[d].sxybnd) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxybnd_pin, (*mloc)[d].sxybnd, 0 , NULL , NULL );
            if ((*mloc)[d].syzbnd) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].syzbnd_pin, (*mloc)[d].syzbnd, 0 , NULL , NULL );
            if ((*mloc)[d].sxzbnd) clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].sxzbnd_pin, (*mloc)[d].sxzbnd, 0 , NULL , NULL );
            if ((*mloc)[d].vxbnd)  clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vxbnd_pin,  (*mloc)[d].vxbnd, 0 , NULL , NULL );
            if ((*mloc)[d].vybnd)  clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vybnd_pin,  (*mloc)[d].vybnd, 0 , NULL , NULL );
            if ((*mloc)[d].vzbnd)  clEnqueueUnmapMemObject( (*vcl)[d].cmd_queuecomm, (*vcl)[d].vzbnd_pin,  (*mloc)[d].vzbnd, 0 , NULL , NULL );
            
            
        }
        
        if ((*vcl)){
            
            if ((*vcl)[d].sxx) clReleaseMemObject((*vcl)[d].sxx);
            if ((*vcl)[d].syy) clReleaseMemObject((*vcl)[d].syy);
            if ((*vcl)[d].szz) clReleaseMemObject((*vcl)[d].szz);
            if ((*vcl)[d].sxy) clReleaseMemObject((*vcl)[d].sxy);
            if ((*vcl)[d].syz) clReleaseMemObject((*vcl)[d].syz);
            if ((*vcl)[d].sxz) clReleaseMemObject((*vcl)[d].sxz);
            if ((*vcl)[d].vx) clReleaseMemObject((*vcl)[d].vx);
            if ((*vcl)[d].vy) clReleaseMemObject((*vcl)[d].vy);
            if ((*vcl)[d].vz) clReleaseMemObject((*vcl)[d].vz);
            if ((*vcl)[d].rip) clReleaseMemObject((*vcl)[d].rip);
            if ((*vcl)[d].rjp) clReleaseMemObject((*vcl)[d].rjp);
            if ((*vcl)[d].rkp) clReleaseMemObject((*vcl)[d].rkp);
            if ((*vcl)[d].uipjp) clReleaseMemObject((*vcl)[d].uipjp);
            if ((*vcl)[d].ujpkp) clReleaseMemObject((*vcl)[d].ujpkp);
            if ((*vcl)[d].uipkp) clReleaseMemObject((*vcl)[d].uipkp);
            if ((*vcl)[d].u) clReleaseMemObject((*vcl)[d].u);
            if ((*vcl)[d].pi) clReleaseMemObject((*vcl)[d].pi);
            if ((*vcl)[d].src) clReleaseMemObject((*vcl)[d].src);
            if ((*vcl)[d].src_pos) clReleaseMemObject((*vcl)[d].src_pos);
            if ((*vcl)[d].rec_pos) clReleaseMemObject((*vcl)[d].rec_pos);
            if ((*vcl)[d].taper) clReleaseMemObject((*vcl)[d].taper);
            if ((*vcl)[d].vxout) clReleaseMemObject((*vcl)[d].vxout);
            if ((*vcl)[d].vyout) clReleaseMemObject((*vcl)[d].vyout);
            if ((*vcl)[d].vzout) clReleaseMemObject((*vcl)[d].vzout);
            
            if ((*vcl)[d].taup) clReleaseMemObject((*vcl)[d].taup);
            if ((*vcl)[d].taus) clReleaseMemObject((*vcl)[d].taus);
            if ((*vcl)[d].tausipjp) clReleaseMemObject((*vcl)[d].tausipjp);
            if ((*vcl)[d].tausjpkp) clReleaseMemObject((*vcl)[d].tausjpkp);
            if ((*vcl)[d].tausipkp) clReleaseMemObject((*vcl)[d].tausipkp);
            if ((*vcl)[d].eta) clReleaseMemObject((*vcl)[d].eta);
            if ((*vcl)[d].rxx) clReleaseMemObject((*vcl)[d].rxx);
            if ((*vcl)[d].ryy) clReleaseMemObject((*vcl)[d].ryy);
            if ((*vcl)[d].rzz) clReleaseMemObject((*vcl)[d].rzz);
            if ((*vcl)[d].rxy) clReleaseMemObject((*vcl)[d].rxy);
            if ((*vcl)[d].ryz) clReleaseMemObject((*vcl)[d].ryz);
            if ((*vcl)[d].rxz) clReleaseMemObject((*vcl)[d].rxz);

            if ((*vcl)[d].psi_sxx_x) clReleaseMemObject((*vcl)[d].psi_sxx_x);
            if ((*vcl)[d].psi_sxy_x) clReleaseMemObject((*vcl)[d].psi_sxy_x);
            if ((*vcl)[d].psi_sxz_x) clReleaseMemObject((*vcl)[d].psi_sxz_x);
            if ((*vcl)[d].psi_syy_y) clReleaseMemObject((*vcl)[d].psi_syy_y);
            if ((*vcl)[d].psi_sxy_y) clReleaseMemObject((*vcl)[d].psi_sxy_y);
            if ((*vcl)[d].psi_syz_y) clReleaseMemObject((*vcl)[d].psi_syz_y);
            if ((*vcl)[d].psi_szz_z) clReleaseMemObject((*vcl)[d].psi_szz_z);
            if ((*vcl)[d].psi_sxz_z) clReleaseMemObject((*vcl)[d].psi_sxz_z);
            if ((*vcl)[d].psi_syz_z) clReleaseMemObject((*vcl)[d].psi_syz_z);
            
            if ((*vcl)[d].psi_vxx) clReleaseMemObject((*vcl)[d].psi_vxx);
            if ((*vcl)[d].psi_vyy) clReleaseMemObject((*vcl)[d].psi_vyy);
            if ((*vcl)[d].psi_vzz) clReleaseMemObject((*vcl)[d].psi_vzz);
            if ((*vcl)[d].psi_vxy) clReleaseMemObject((*vcl)[d].psi_vxy);
            if ((*vcl)[d].psi_vxz) clReleaseMemObject((*vcl)[d].psi_vxz);
            if ((*vcl)[d].psi_vyx) clReleaseMemObject((*vcl)[d].psi_vyx);
            if ((*vcl)[d].psi_vyz) clReleaseMemObject((*vcl)[d].psi_vyz);
            if ((*vcl)[d].psi_vzx) clReleaseMemObject((*vcl)[d].psi_vzx);
            if ((*vcl)[d].psi_vzy) clReleaseMemObject((*vcl)[d].psi_vzy);
            
            if ((*vcl)[d].K_x) clReleaseMemObject((*vcl)[d].K_x);
            if ((*vcl)[d].a_x) clReleaseMemObject((*vcl)[d].a_x);
            if ((*vcl)[d].b_x) clReleaseMemObject((*vcl)[d].b_x);
            if ((*vcl)[d].K_x_half) clReleaseMemObject((*vcl)[d].K_x_half);
            if ((*vcl)[d].a_x_half) clReleaseMemObject((*vcl)[d].a_x_half);
            if ((*vcl)[d].b_x_half) clReleaseMemObject((*vcl)[d].b_x_half);
            
            if ((*vcl)[d].K_y) clReleaseMemObject((*vcl)[d].K_y);
            if ((*vcl)[d].a_y) clReleaseMemObject((*vcl)[d].a_y);
            if ((*vcl)[d].b_y) clReleaseMemObject((*vcl)[d].b_y);
            if ((*vcl)[d].K_y_half) clReleaseMemObject((*vcl)[d].K_y_half);
            if ((*vcl)[d].a_y_half) clReleaseMemObject((*vcl)[d].a_y_half);
            if ((*vcl)[d].b_y_half) clReleaseMemObject((*vcl)[d].b_y_half);
            
            if ((*vcl)[d].K_z) clReleaseMemObject((*vcl)[d].K_z);
            if ((*vcl)[d].a_z) clReleaseMemObject((*vcl)[d].a_z);
            if ((*vcl)[d].b_z) clReleaseMemObject((*vcl)[d].b_z);
            if ((*vcl)[d].K_z_half) clReleaseMemObject((*vcl)[d].K_z_half);
            if ((*vcl)[d].a_z_half) clReleaseMemObject((*vcl)[d].a_z_half);
            if ((*vcl)[d].b_z_half) clReleaseMemObject((*vcl)[d].b_z_half);
            
            if ((*vcl)[d].gradsrc) clReleaseMemObject((*vcl)[d].gradsrc);
            
            
            if ((*vcl)[d].sxxbnd) clReleaseMemObject((*vcl)[d].sxxbnd);
            if ((*vcl)[d].syybnd) clReleaseMemObject((*vcl)[d].syybnd);
            if ((*vcl)[d].szzbnd) clReleaseMemObject((*vcl)[d].szzbnd);
            if ((*vcl)[d].sxybnd) clReleaseMemObject((*vcl)[d].sxybnd);
            if ((*vcl)[d].syzbnd) clReleaseMemObject((*vcl)[d].syzbnd);
            if ((*vcl)[d].sxzbnd) clReleaseMemObject((*vcl)[d].sxzbnd);
            if ((*vcl)[d].vxbnd) clReleaseMemObject((*vcl)[d].vxbnd);
            if ((*vcl)[d].vybnd) clReleaseMemObject((*vcl)[d].vybnd);
            if ((*vcl)[d].vzbnd) clReleaseMemObject((*vcl)[d].vzbnd);
            
            if ((*vcl)[d].sxx_r) clReleaseMemObject((*vcl)[d].sxx_r);
            if ((*vcl)[d].syy_r) clReleaseMemObject((*vcl)[d].syy_r);
            if ((*vcl)[d].szz_r) clReleaseMemObject((*vcl)[d].szz_r);
            if ((*vcl)[d].sxy_r) clReleaseMemObject((*vcl)[d].sxy_r);
            if ((*vcl)[d].syz_r) clReleaseMemObject((*vcl)[d].syz_r);
            if ((*vcl)[d].sxz_r) clReleaseMemObject((*vcl)[d].sxz_r);
            
            if ((*vcl)[d].vx_r) clReleaseMemObject((*vcl)[d].vx_r);
            if ((*vcl)[d].vy_r) clReleaseMemObject((*vcl)[d].vy_r);
            if ((*vcl)[d].vz_r) clReleaseMemObject((*vcl)[d].vz_r);
            
            if ((*vcl)[d].rxx_r) clReleaseMemObject((*vcl)[d].rxx_r);
            if ((*vcl)[d].ryy_r) clReleaseMemObject((*vcl)[d].ryy_r);
            if ((*vcl)[d].rzz_r) clReleaseMemObject((*vcl)[d].rzz_r);
            if ((*vcl)[d].rxy_r) clReleaseMemObject((*vcl)[d].rxy_r);
            if ((*vcl)[d].ryz_r) clReleaseMemObject((*vcl)[d].ryz_r);
            if ((*vcl)[d].rxz_r) clReleaseMemObject((*vcl)[d].rxz_r);
            
            if ((*vcl)[d].gradtaup) clReleaseMemObject((*vcl)[d].gradtaup);
            if ((*vcl)[d].gradtaus) clReleaseMemObject((*vcl)[d].gradtaus);
            if ((*vcl)[d].gradrho) clReleaseMemObject((*vcl)[d].gradrho);
            if ((*vcl)[d].gradM) clReleaseMemObject((*vcl)[d].gradM);
            if ((*vcl)[d].gradmu) clReleaseMemObject((*vcl)[d].gradmu);
            
            if ((*vcl)[d].Hrho) clReleaseMemObject((*vcl)[d].Hrho);
            if ((*vcl)[d].HM) clReleaseMemObject((*vcl)[d].HM);
            if ((*vcl)[d].Hmu) clReleaseMemObject((*vcl)[d].Hmu);
            if ((*vcl)[d].Htaup) clReleaseMemObject((*vcl)[d].Htaup);
            if ((*vcl)[d].Htaus) clReleaseMemObject((*vcl)[d].Htaus);
            
            if ((*vcl)[d].f_sxx) clReleaseMemObject((*vcl)[d].f_sxx);
            if ((*vcl)[d].f_syy) clReleaseMemObject((*vcl)[d].f_syy);
            if ((*vcl)[d].f_szz) clReleaseMemObject((*vcl)[d].f_szz);
            if ((*vcl)[d].f_sxy) clReleaseMemObject((*vcl)[d].f_sxy);
            if ((*vcl)[d].f_syz) clReleaseMemObject((*vcl)[d].f_syz);
            if ((*vcl)[d].f_sxz) clReleaseMemObject((*vcl)[d].f_sxz);
            if ((*vcl)[d].f_vx) clReleaseMemObject((*vcl)[d].f_vx);
            if ((*vcl)[d].f_vy) clReleaseMemObject((*vcl)[d].f_vy);
            if ((*vcl)[d].f_vz) clReleaseMemObject((*vcl)[d].f_vz);
            
            if ((*vcl)[d].f_rxx) clReleaseMemObject((*vcl)[d].f_rxx);
            if ((*vcl)[d].f_ryy) clReleaseMemObject((*vcl)[d].f_ryy);
            if ((*vcl)[d].f_rzz) clReleaseMemObject((*vcl)[d].f_rzz);
            if ((*vcl)[d].f_rxy) clReleaseMemObject((*vcl)[d].f_rxy);
            if ((*vcl)[d].f_ryz) clReleaseMemObject((*vcl)[d].f_ryz);
            if ((*vcl)[d].f_rxz) clReleaseMemObject((*vcl)[d].f_rxz);
            
            if ((*vcl)[d].gradfreqsn) clReleaseMemObject((*vcl)[d].gradfreqsn);
            
            if ((*vcl)[d].sxx_sub1) clReleaseMemObject((*vcl)[d].sxx_sub1);
            if ((*vcl)[d].syy_sub1) clReleaseMemObject((*vcl)[d].syy_sub1);
            if ((*vcl)[d].szz_sub1) clReleaseMemObject((*vcl)[d].szz_sub1);
            if ((*vcl)[d].sxy_sub1) clReleaseMemObject((*vcl)[d].sxy_sub1);
            if ((*vcl)[d].syz_sub1) clReleaseMemObject((*vcl)[d].syz_sub1);
            if ((*vcl)[d].sxz_sub1) clReleaseMemObject((*vcl)[d].sxz_sub1);
            if ((*vcl)[d].vx_sub1) clReleaseMemObject((*vcl)[d].vx_sub1);
            if ((*vcl)[d].vy_sub1) clReleaseMemObject((*vcl)[d].vy_sub1);
            if ((*vcl)[d].vz_sub1) clReleaseMemObject((*vcl)[d].vz_sub1);
            if ((*vcl)[d].sxx_sub2) clReleaseMemObject((*vcl)[d].sxx_sub2);
            if ((*vcl)[d].syy_sub2) clReleaseMemObject((*vcl)[d].syy_sub2);
            if ((*vcl)[d].szz_sub2) clReleaseMemObject((*vcl)[d].szz_sub2);
            if ((*vcl)[d].sxy_sub2) clReleaseMemObject((*vcl)[d].sxy_sub2);
            if ((*vcl)[d].syz_sub2) clReleaseMemObject((*vcl)[d].syz_sub2);
            if ((*vcl)[d].sxz_sub2) clReleaseMemObject((*vcl)[d].sxz_sub2);
            if ((*vcl)[d].vx_sub2) clReleaseMemObject((*vcl)[d].vx_sub2);
            if ((*vcl)[d].vy_sub2) clReleaseMemObject((*vcl)[d].vy_sub2);
            if ((*vcl)[d].vz_sub2) clReleaseMemObject((*vcl)[d].vz_sub2);
            
            if ((*vcl)[d].sxx_r_sub1) clReleaseMemObject((*vcl)[d].sxx_r_sub1);
            if ((*vcl)[d].syy_r_sub1) clReleaseMemObject((*vcl)[d].syy_r_sub1);
            if ((*vcl)[d].szz_r_sub1) clReleaseMemObject((*vcl)[d].szz_r_sub1);
            if ((*vcl)[d].sxy_r_sub1) clReleaseMemObject((*vcl)[d].sxy_r_sub1);
            if ((*vcl)[d].syz_r_sub1) clReleaseMemObject((*vcl)[d].syz_r_sub1);
            if ((*vcl)[d].sxz_r_sub1) clReleaseMemObject((*vcl)[d].sxz_r_sub1);
            if ((*vcl)[d].vx_r_sub1) clReleaseMemObject((*vcl)[d].vx_r_sub1);
            if ((*vcl)[d].vy_r_sub1) clReleaseMemObject((*vcl)[d].vy_r_sub1);
            if ((*vcl)[d].vz_r_sub1) clReleaseMemObject((*vcl)[d].vz_r_sub1);
            if ((*vcl)[d].sxx_r_sub2) clReleaseMemObject((*vcl)[d].sxx_r_sub2);
            if ((*vcl)[d].syy_r_sub2) clReleaseMemObject((*vcl)[d].syy_r_sub2);
            if ((*vcl)[d].szz_r_sub2) clReleaseMemObject((*vcl)[d].szz_r_sub2);
            if ((*vcl)[d].sxy_r_sub2) clReleaseMemObject((*vcl)[d].sxy_r_sub2);
            if ((*vcl)[d].syz_r_sub2) clReleaseMemObject((*vcl)[d].syz_r_sub2);
            if ((*vcl)[d].sxz_r_sub2) clReleaseMemObject((*vcl)[d].sxz_r_sub2);
            if ((*vcl)[d].vx_r_sub2) clReleaseMemObject((*vcl)[d].vx_r_sub2);
            if ((*vcl)[d].vy_r_sub2) clReleaseMemObject((*vcl)[d].vy_r_sub2);
            if ((*vcl)[d].vz_r_sub2) clReleaseMemObject((*vcl)[d].vz_r_sub2);
            
            if ((*vcl)[d].sxxbnd_pin) clReleaseMemObject((*vcl)[d].sxxbnd_pin);
            if ((*vcl)[d].syybnd_pin) clReleaseMemObject((*vcl)[d].syybnd_pin);
            if ((*vcl)[d].szzbnd_pin) clReleaseMemObject((*vcl)[d].szzbnd_pin);
            if ((*vcl)[d].sxybnd_pin) clReleaseMemObject((*vcl)[d].sxybnd_pin);
            if ((*vcl)[d].syzbnd_pin) clReleaseMemObject((*vcl)[d].syzbnd_pin);
            if ((*vcl)[d].sxzbnd_pin) clReleaseMemObject((*vcl)[d].sxzbnd_pin);
            if ((*vcl)[d].vxbnd_pin) clReleaseMemObject((*vcl)[d].vxbnd_pin);
            if ((*vcl)[d].vybnd_pin) clReleaseMemObject((*vcl)[d].vybnd_pin);
            if ((*vcl)[d].vzbnd_pin) clReleaseMemObject((*vcl)[d].vzbnd_pin);
            
            if ((*vcl)[d].kernel_v) clReleaseKernel((*vcl)[d].kernel_v);
            if ((*vcl)[d].kernel_s) clReleaseKernel((*vcl)[d].kernel_s);
            if ((*vcl)[d].kernel_surf) clReleaseKernel((*vcl)[d].kernel_surf);
            if ((*vcl)[d].kernel_initseis) clReleaseKernel((*vcl)[d].kernel_initseis);
            if ((*vcl)[d].kernel_vout) clReleaseKernel((*vcl)[d].kernel_vout);
            if ((*vcl)[d].kernel_voutinit) clReleaseKernel((*vcl)[d].kernel_voutinit);
            if ((*vcl)[d].kernel_vcomm1) clReleaseKernel((*vcl)[d].kernel_vcomm1);
            if ((*vcl)[d].kernel_scomm1) clReleaseKernel((*vcl)[d].kernel_scomm1);
            if ((*vcl)[d].kernel_vcomm2) clReleaseKernel((*vcl)[d].kernel_vcomm2);
            if ((*vcl)[d].kernel_scomm2) clReleaseKernel((*vcl)[d].kernel_scomm2);
            if ((*vcl)[d].kernel_adjv) clReleaseKernel((*vcl)[d].kernel_adjv);
            if ((*vcl)[d].kernel_adjs) clReleaseKernel((*vcl)[d].kernel_adjs);
            if ((*vcl)[d].kernel_initseis_r) clReleaseKernel((*vcl)[d].kernel_initseis_r);
            if ((*vcl)[d].kernel_residuals ) clReleaseKernel((*vcl)[d].kernel_residuals );
            if ((*vcl)[d].kernel_initgrad) clReleaseKernel((*vcl)[d].kernel_initgrad);
            if ((*vcl)[d].kernel_bnd) clReleaseKernel((*vcl)[d].kernel_bnd);
            if ((*vcl)[d].kernel_savefreqs) clReleaseKernel((*vcl)[d].kernel_savefreqs);
            if ((*vcl)[d].kernel_initsavefreqs) clReleaseKernel((*vcl)[d].kernel_initsavefreqs);
            if ((*vcl)[d].kernel_initialize_gradsrc) clReleaseKernel((*vcl)[d].kernel_initialize_gradsrc);
            
            
            if ((*vcl)[d].program_v) clReleaseProgram((*vcl)[d].program_v);
            if ((*vcl)[d].program_s) clReleaseProgram((*vcl)[d].program_s);
            if ((*vcl)[d].program_surf) clReleaseProgram((*vcl)[d].program_surf);
            if ((*vcl)[d].program_initseis) clReleaseProgram((*vcl)[d].program_initseis);
            if ((*vcl)[d].program_vout) clReleaseProgram((*vcl)[d].program_vout);
            if ((*vcl)[d].program_voutinit) clReleaseProgram((*vcl)[d].program_voutinit);
            if ((*vcl)[d].program_adjv) clReleaseProgram((*vcl)[d].program_adjv);
            if ((*vcl)[d].program_adjs) clReleaseProgram((*vcl)[d].program_adjs);
            if ((*vcl)[d].program_initseis_r) clReleaseProgram((*vcl)[d].program_initseis_r);
            if ((*vcl)[d].program_residuals) clReleaseProgram((*vcl)[d].program_residuals);
            if ((*vcl)[d].program_initgrad) clReleaseProgram((*vcl)[d].program_initgrad);
            if ((*vcl)[d].program_bnd) clReleaseProgram((*vcl)[d].program_bnd);
            if ((*vcl)[d].program_savefreqs) clReleaseProgram((*vcl)[d].program_savefreqs);
            if ((*vcl)[d].program_initsavefreqs) clReleaseProgram((*vcl)[d].program_initsavefreqs);
            if ((*vcl)[d].program_initialize_gradsrc) clReleaseProgram((*vcl)[d].program_initialize_gradsrc);
            

            
            if ((*vcl)[d].cmd_queue) clReleaseCommandQueue((*vcl)[d].cmd_queue);
            if ((*vcl)[d].cmd_queuecomm) clReleaseCommandQueue((*vcl)[d].cmd_queuecomm);
        }

    }

    if (m->context) clReleaseContext(m->context);
    if ((*vcl)) free((*vcl));
    if ((*mloc)) free((*mloc));
    
    return 0;


}
