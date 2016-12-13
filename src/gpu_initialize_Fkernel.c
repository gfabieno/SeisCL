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

/*Initalize the kernel that updates the velocity components*/

#include "F.h"

char *get_build_options(struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc, int lcomm, int comm, int dirprop){
    
    static char option_DH [2000];
    sprintf(option_DH, "-D NX=%d -D NY=%d -D NZ=%d -D offset=%d -D fdo=%d -D fdoh=%d -D hc0=%9.9f -D hc1=%9.9f -D hc2=%9.9f -D hc3=%9.9f -D hc4=%9.9f -D hc5=%9.9f -D hc6=%9.9f -D dhi=%9.9f -D dhi2=%9.9f -D dtdh=%9.9f -D dtdh2=%9.9f -D DH=%9.9f -D DT=%9.9f -D dt2=%9.9f -D NT=%d -D nab=%d -D Nbnd=%d -D local_off=%d -D Lve=%d -D dev=%d -D num_devices=%d -D ND=%d -D abs_type=%d -D freesurf=%d -D lcomm=%d -D MYLOCALID=%d -D NLOCALP=%d -D nfreqs=%d -D back_prop_type=%d -D comm12=%d -D locsizexy=%d -D NZ_al16=%d -D NZ_al0=%d -D NTnyq=%d -D dtnyq=%d -D gradsrcout=%d -D bcastvx=%d -D bcastvy=%d -D bcastvz=%d -D dirprop=%d",(*inmloc).NX+(*inm).FDORDER, (*inmloc).NY+(*inm).FDORDER, (*inmloc).NZ+(*inm).FDORDER, (*inmloc).NX0, (*inm).fdo, (*inm).fdoh, (*inm).hc[0], (*inm).hc[1], (*inm).hc[2], (*inm).hc[3], (*inm).hc[4], (*inm).hc[5], (*inm).hc[6], (*inm).dhi, (*inm).dhi/2.0, (*inm).dt/(*inm).dh, (*inm).dt/(*inm).dh/2.0, (*inm).dh, (*inm).dt, (*inm).dt/2.0, (*inm).NT, (*inm).nab, (*inmloc).Nbnd, (*inmloc).local_off, (*inm).L, (*inmloc).dev, (*inmloc).num_devices,(*inm).ND, (*inm).abs_type, (*inm).freesurf, lcomm, (*inm).MYLOCALID, (*inm).NLOCALP, (*inm).nfreqs, (*inm).back_prop_type, comm, (int)(*inmloc).local_work_size[1], (*inmloc).NZ_al16, (*inmloc).NZ_al0, (*inm).NTnyq,(*inm).dtnyq, (*inm).gradsrcout, (*inm).bcastvx, (*inm).bcastvy, (*inm).bcastvz, dirprop  );
    
    return option_DH;
}

char *get_build_options_surf(struct varcl *inmem_c, struct modcsts *inm_c, struct modcstsloc *inmloc_c, struct varcl *inmem_f, struct modcsts *inm_f, struct modcstsloc *inmloc_f){
    
    static char option_DH [2000];
    sprintf(option_DH, "-D NX_c=%d -D NY_c=%d -D NZ_c=%d -D offset_c=%d -D fdo_c=%d -D fdoh_c=%d -D NT_c=%d -D nab_c=%d -D Nbnd_c=%d -D Lve_c=%d -D dev_c=%d -D num_devices_c=%d -D ND_c=%d -D Nr_c=%d -D abs_type_c=%d -D freesurf_c=%d -D MYLOCALID_c=%d -D NLOCALP_c=%d -D NZ_al16_c=%d -D NZ_al0_c=%d -D NX_f=%d -D NY_f=%d -D NZ_f=%d -D offset_f=%d -D fdo_f=%d -D fdoh_f=%d -D NT_f=%d -D nab_f=%d -D Nbnd_f=%d -D Lve_f=%d -D dev_f=%d -D num_devices_f=%d -D ND_f=%d -D abs_type_f=%d -D freesurf_f=%d -D MYLOCALID_f=%d -D NLOCALP_f=%d -D NZ_al16_f=%d -D NZ_al0_f=%d",(*inmloc_c).NX+(*inm_c).FDORDER, (*inmloc_c).NY+(*inm_c).FDORDER, (*inmloc_c).NZ+(*inm_c).FDORDER, (*inmloc_c).NX0, (*inm_c).fdo, (*inm_c).fdoh,  (*inm_c).NT, (*inm_c).nab, (*inmloc_c).Nbnd, (*inm_c).L, (*inmloc_c).dev, (*inmloc_c).num_devices,(*inm_c).ND, (int)pow(2,(*inm_c).Nr), (*inm_c).abs_type, (*inm_c).freesurf, (*inm_c).MYLOCALID, (*inm_c).NLOCALP,  (*inmloc_c).NZ_al16, (*inmloc_c).NZ_al0, (*inmloc_f).NX+(*inm_f).FDORDER, (*inmloc_f).NY+(*inm_f).FDORDER, (*inmloc_f).NZ+(*inm_f).FDORDER, (*inmloc_f).NX0, (*inm_f).fdo, (*inm_f).fdoh,  (*inm_f).NT, (*inm_f).nab, (*inmloc_f).Nbnd, (*inm_f).L, (*inmloc_f).dev, (*inmloc_f).num_devices,(*inm_f).ND, (*inm_f).abs_type, (*inm_f).freesurf, (*inm_f).MYLOCALID, (*inm_f).NLOCALP,  (*inmloc_f).NZ_al16, (*inmloc_f).NZ_al0  );
    
    return option_DH;
}


int gpu_initialize_update_v(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc, int bndoff, int lcomm, int comm )
{

	cl_int cl_err = 0;
    size_t shared_size=sizeof(float);
    
    char filename[20];
    if ((*inm).ND==3){
      sprintf(filename,"./update_v3D.cl");
    }
    else if ((*inm).ND==2){
      sprintf(filename,"./update_v2D.cl");
    }
    else if ((*inm).ND==21){
        sprintf(filename,"./update_v2D_SH.cl");
    }

    const char * build_options=get_build_options(inmem, inm, inmloc, lcomm, comm, 0);
    
    /*Create the kernel, ther kernel version depends on the finite difference order*/
    const char * program_name = "update_v";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    

    /*Define the size of the local variables of the compute device*/
    if ((*inmloc).local_off==0){
        if ((*inm).ND==3){
            shared_size = ((local_work_size[0]+(*inm).FDORDER)*(local_work_size[1]+(*inm).FDORDER)*(local_work_size[2]+(*inm).FDORDER)) * sizeof(float);
        }
        else if ((*inm).ND==2 || (*inm).ND==21){
            shared_size = ((local_work_size[0]+(*inm).FDORDER)*(local_work_size[1]+(*inm).FDORDER)) * sizeof(float);
        }
    }
    
    /*Define the arguments for this kernel */
    cl_err = clSetKernelArg(*pkernel,  0, sizeof(int), &bndoff);
    if ((*inm).ND==3){
        cl_err = clSetKernelArg(*pkernel,  3, sizeof(cl_mem), &inmem->vx);
        cl_err = clSetKernelArg(*pkernel,  4, sizeof(cl_mem), &inmem->vy);
        cl_err = clSetKernelArg(*pkernel,  5, sizeof(cl_mem), &inmem->vz);
        cl_err = clSetKernelArg(*pkernel,  6, sizeof(cl_mem), &inmem->sxx);
        cl_err = clSetKernelArg(*pkernel,  7, sizeof(cl_mem), &inmem->syy);
        cl_err = clSetKernelArg(*pkernel,  8, sizeof(cl_mem), &inmem->szz);
        cl_err = clSetKernelArg(*pkernel,  9, sizeof(cl_mem), &inmem->sxy);
        cl_err = clSetKernelArg(*pkernel,  10, sizeof(cl_mem), &inmem->syz);
        cl_err = clSetKernelArg(*pkernel,  11, sizeof(cl_mem), &inmem->sxz);
        cl_err = clSetKernelArg(*pkernel,  12, sizeof(cl_mem), &inmem->rip);
        cl_err = clSetKernelArg(*pkernel,  13, sizeof(cl_mem), &inmem->rjp);
        cl_err = clSetKernelArg(*pkernel,  14, sizeof(cl_mem), &inmem->rkp);
        cl_err = clSetKernelArg(*pkernel,  15, sizeof(cl_mem), &inmem->src_pos);
        cl_err = clSetKernelArg(*pkernel,  16, sizeof(cl_mem), &inmem->src);
        cl_err = clSetKernelArg(*pkernel,  17, sizeof(cl_mem), &inmem->rec_pos);
        
        cl_err = clSetKernelArg(*pkernel,  18, sizeof(cl_mem), &inmem->taper);
        
        cl_err = clSetKernelArg(*pkernel,  19, sizeof(cl_mem), &inmem->K_x);
        cl_err = clSetKernelArg(*pkernel,  20, sizeof(cl_mem), &inmem->a_x);
        cl_err = clSetKernelArg(*pkernel,  21, sizeof(cl_mem), &inmem->b_x);
        cl_err = clSetKernelArg(*pkernel,  22, sizeof(cl_mem), &inmem->K_x_half);
        cl_err = clSetKernelArg(*pkernel,  23, sizeof(cl_mem), &inmem->a_x_half);
        cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem->b_x_half);
        
        cl_err = clSetKernelArg(*pkernel,  25, sizeof(cl_mem), &inmem->K_y);
        cl_err = clSetKernelArg(*pkernel,  26, sizeof(cl_mem), &inmem->a_y);
        cl_err = clSetKernelArg(*pkernel,  27, sizeof(cl_mem), &inmem->b_y);
        cl_err = clSetKernelArg(*pkernel,  28, sizeof(cl_mem), &inmem->K_y_half);
        cl_err = clSetKernelArg(*pkernel,  29, sizeof(cl_mem), &inmem->a_y_half);
        cl_err = clSetKernelArg(*pkernel,  30, sizeof(cl_mem), &inmem->b_y_half);
        
        cl_err = clSetKernelArg(*pkernel,  31, sizeof(cl_mem), &inmem->K_z);
        cl_err = clSetKernelArg(*pkernel,  32, sizeof(cl_mem), &inmem->a_z);
        cl_err = clSetKernelArg(*pkernel,  33, sizeof(cl_mem), &inmem->b_z);
        cl_err = clSetKernelArg(*pkernel,  34, sizeof(cl_mem), &inmem->K_z_half);
        cl_err = clSetKernelArg(*pkernel,  35, sizeof(cl_mem), &inmem->a_z_half);
        cl_err = clSetKernelArg(*pkernel,  36, sizeof(cl_mem), &inmem->b_z_half);
        
        cl_err = clSetKernelArg(*pkernel,  37, sizeof(cl_mem), &inmem->psi_sxx_x);
        cl_err = clSetKernelArg(*pkernel,  38, sizeof(cl_mem), &inmem->psi_sxy_x);
        cl_err = clSetKernelArg(*pkernel,  39, sizeof(cl_mem), &inmem->psi_sxy_y);
        cl_err = clSetKernelArg(*pkernel,  40, sizeof(cl_mem), &inmem->psi_sxz_x);
        cl_err = clSetKernelArg(*pkernel,  41, sizeof(cl_mem), &inmem->psi_sxz_z);
        cl_err = clSetKernelArg(*pkernel,  42, sizeof(cl_mem), &inmem->psi_syy_y);
        cl_err = clSetKernelArg(*pkernel,  43, sizeof(cl_mem), &inmem->psi_syz_y);
        cl_err = clSetKernelArg(*pkernel,  44, sizeof(cl_mem), &inmem->psi_syz_z);
        cl_err = clSetKernelArg(*pkernel,  45, sizeof(cl_mem), &inmem->psi_szz_z);

        
        cl_err = clSetKernelArg(*pkernel,  46, shared_size, NULL);

        
    }
    else if ((*inm).ND==2){
        cl_err = clSetKernelArg(*pkernel,  3, sizeof(cl_mem), &inmem->vx);
        cl_err = clSetKernelArg(*pkernel,  4, sizeof(cl_mem), &inmem->vz);
        cl_err = clSetKernelArg(*pkernel,  5, sizeof(cl_mem), &inmem->sxx);
        cl_err = clSetKernelArg(*pkernel,  6, sizeof(cl_mem), &inmem->szz);
        cl_err = clSetKernelArg(*pkernel,  7, sizeof(cl_mem), &inmem->sxz);
        cl_err = clSetKernelArg(*pkernel,  8, sizeof(cl_mem), &inmem->rip);
        cl_err = clSetKernelArg(*pkernel,  9, sizeof(cl_mem), &inmem->rkp);
        cl_err = clSetKernelArg(*pkernel,  10, sizeof(cl_mem), &inmem->src_pos);
        cl_err = clSetKernelArg(*pkernel,  11, sizeof(cl_mem), &inmem->src);
        cl_err = clSetKernelArg(*pkernel,  12, sizeof(cl_mem), &inmem->rec_pos);
        cl_err = clSetKernelArg(*pkernel,  13, sizeof(cl_mem), &inmem->taper);
        
        cl_err = clSetKernelArg(*pkernel,  14, sizeof(cl_mem), &inmem->K_x);
        cl_err = clSetKernelArg(*pkernel,  15, sizeof(cl_mem), &inmem->a_x);
        cl_err = clSetKernelArg(*pkernel,  16, sizeof(cl_mem), &inmem->b_x);
        cl_err = clSetKernelArg(*pkernel,  17, sizeof(cl_mem), &inmem->K_x_half);
        cl_err = clSetKernelArg(*pkernel,  18, sizeof(cl_mem), &inmem->a_x_half);
        cl_err = clSetKernelArg(*pkernel,  19, sizeof(cl_mem), &inmem->b_x_half);
    
        cl_err = clSetKernelArg(*pkernel,  20, sizeof(cl_mem), &inmem->K_z);
        cl_err = clSetKernelArg(*pkernel,  21, sizeof(cl_mem), &inmem->a_z);
        cl_err = clSetKernelArg(*pkernel,  22, sizeof(cl_mem), &inmem->b_z);
        cl_err = clSetKernelArg(*pkernel,  23, sizeof(cl_mem), &inmem->K_z_half);
        cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem->a_z_half);
        cl_err = clSetKernelArg(*pkernel,  25, sizeof(cl_mem), &inmem->b_z_half);
        
        cl_err = clSetKernelArg(*pkernel,  26, sizeof(cl_mem), &inmem->psi_sxx_x);
        cl_err = clSetKernelArg(*pkernel,  27, sizeof(cl_mem), &inmem->psi_sxz_x);
        cl_err = clSetKernelArg(*pkernel,  28, sizeof(cl_mem), &inmem->psi_sxz_z);
        cl_err = clSetKernelArg(*pkernel,  29, sizeof(cl_mem), &inmem->psi_szz_z);

        
        cl_err = clSetKernelArg(*pkernel,  30, shared_size, NULL);


    }
    if ((*inm).ND==21){
        cl_err = clSetKernelArg(*pkernel,  3, sizeof(cl_mem), &inmem->vy);
        cl_err = clSetKernelArg(*pkernel,  4, sizeof(cl_mem), &inmem->sxy);
        cl_err = clSetKernelArg(*pkernel,  5, sizeof(cl_mem), &inmem->syz);
        cl_err = clSetKernelArg(*pkernel,  6, sizeof(cl_mem), &inmem->rjp);
        cl_err = clSetKernelArg(*pkernel,  7, sizeof(cl_mem), &inmem->src_pos);
        cl_err = clSetKernelArg(*pkernel,  8, sizeof(cl_mem), &inmem->src);
        cl_err = clSetKernelArg(*pkernel,  9, sizeof(cl_mem), &inmem->rec_pos);
        
        cl_err = clSetKernelArg(*pkernel,  10, sizeof(cl_mem), &inmem->taper);
        
        cl_err = clSetKernelArg(*pkernel,  11, sizeof(cl_mem), &inmem->K_x);
        cl_err = clSetKernelArg(*pkernel,  12, sizeof(cl_mem), &inmem->a_x);
        cl_err = clSetKernelArg(*pkernel,  13, sizeof(cl_mem), &inmem->b_x);
        cl_err = clSetKernelArg(*pkernel,  14, sizeof(cl_mem), &inmem->K_x_half);
        cl_err = clSetKernelArg(*pkernel,  15, sizeof(cl_mem), &inmem->a_x_half);
        cl_err = clSetKernelArg(*pkernel,  16, sizeof(cl_mem), &inmem->b_x_half);
        
        cl_err = clSetKernelArg(*pkernel,  17, sizeof(cl_mem), &inmem->K_z);
        cl_err = clSetKernelArg(*pkernel,  18, sizeof(cl_mem), &inmem->a_z);
        cl_err = clSetKernelArg(*pkernel,  19, sizeof(cl_mem), &inmem->b_z);
        cl_err = clSetKernelArg(*pkernel,  20, sizeof(cl_mem), &inmem->K_z_half);
        cl_err = clSetKernelArg(*pkernel,  21, sizeof(cl_mem), &inmem->a_z_half);
        cl_err = clSetKernelArg(*pkernel,  22, sizeof(cl_mem), &inmem->b_z_half);
        
        cl_err = clSetKernelArg(*pkernel,  23, sizeof(cl_mem), &inmem->psi_sxy_x);
        cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem->psi_syz_z);

        cl_err = clSetKernelArg(*pkernel,  25, shared_size, NULL);
        
        
    }

    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
	return cl_err;
}

int gpu_initialize_update_s(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc, int bndoff, int lcomm , int comm )


{
	cl_int cl_err = 0;
    size_t shared_size=sizeof(float);
    
    char filename[20];
    if ((*inm).ND==3){
        sprintf(filename,"./update_s3D.cl");
    }
    else if ((*inm).ND==2){
        sprintf(filename,"./update_s2D.cl");
    }
    else if ((*inm).ND==21){
        sprintf(filename,"./update_s2D_SH.cl");
    }
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, lcomm, comm, 0);
    
    /*Create the kernel*/
    const char * program_name = "update_s";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    /*Define the size of the local variables of the compute device*/
    if ((*inmloc).local_off==0){
        if ((*inm).ND==3){
            shared_size = ((local_work_size[0]+(*inm).FDORDER)*(local_work_size[1]+(*inm).FDORDER)*(local_work_size[2]+(*inm).FDORDER)) * sizeof(float);
        }
        else if ((*inm).ND==2 || (*inm).ND==21){
            shared_size = ((local_work_size[0]+(*inm).FDORDER)*(local_work_size[1]+(*inm).FDORDER)) * sizeof(float);
        }
    }
    
    /*Define the arguments for this kernel */
    cl_err = clSetKernelArg(*pkernel,  0, sizeof(int), &bndoff);
    if ((*inm).ND==3){
        cl_err = clSetKernelArg(*pkernel,  3, sizeof(cl_mem), &inmem->vx);
        cl_err = clSetKernelArg(*pkernel,  4, sizeof(cl_mem), &inmem->vy);
        cl_err = clSetKernelArg(*pkernel,  5, sizeof(cl_mem), &inmem->vz);
        cl_err = clSetKernelArg(*pkernel,  6, sizeof(cl_mem), &inmem->sxx);
        cl_err = clSetKernelArg(*pkernel,  7, sizeof(cl_mem), &inmem->syy);
        cl_err = clSetKernelArg(*pkernel,  8, sizeof(cl_mem), &inmem->szz);
        cl_err = clSetKernelArg(*pkernel,  9, sizeof(cl_mem), &inmem->sxy);
        cl_err = clSetKernelArg(*pkernel,  10, sizeof(cl_mem), &inmem->syz);
        cl_err = clSetKernelArg(*pkernel,  11, sizeof(cl_mem), &inmem->sxz);
        
        
        cl_err = clSetKernelArg(*pkernel,  12, sizeof(cl_mem), &inmem->pi);
        cl_err = clSetKernelArg(*pkernel,  13, sizeof(cl_mem), &inmem->u);
        cl_err = clSetKernelArg(*pkernel,  14, sizeof(cl_mem), &inmem->uipjp);
        cl_err = clSetKernelArg(*pkernel,  15, sizeof(cl_mem), &inmem->ujpkp);
        cl_err = clSetKernelArg(*pkernel,  16, sizeof(cl_mem), &inmem->uipkp);
    
        cl_err = clSetKernelArg(*pkernel,  17, sizeof(cl_mem), &inmem->rxx);
        cl_err = clSetKernelArg(*pkernel,  18, sizeof(cl_mem), &inmem->ryy);
        cl_err = clSetKernelArg(*pkernel,  19, sizeof(cl_mem), &inmem->rzz);
        cl_err = clSetKernelArg(*pkernel,  20, sizeof(cl_mem), &inmem->rxy);
        cl_err = clSetKernelArg(*pkernel,  21, sizeof(cl_mem), &inmem->ryz);
        cl_err = clSetKernelArg(*pkernel,  22, sizeof(cl_mem), &inmem->rxz);
        cl_err = clSetKernelArg(*pkernel,  23, sizeof(cl_mem), &inmem->taus);
        cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem->tausipjp);
        cl_err = clSetKernelArg(*pkernel,  25, sizeof(cl_mem), &inmem->tausjpkp);
        cl_err = clSetKernelArg(*pkernel,  26, sizeof(cl_mem), &inmem->tausipkp);
        cl_err = clSetKernelArg(*pkernel,  27, sizeof(cl_mem), &inmem->taup);
        cl_err = clSetKernelArg(*pkernel,  28, sizeof(cl_mem), &inmem->eta);
        
        
        cl_err = clSetKernelArg(*pkernel,  29, sizeof(cl_mem), &inmem->src_pos);
        cl_err = clSetKernelArg(*pkernel,  30, sizeof(cl_mem), &inmem->src);
        
        cl_err = clSetKernelArg(*pkernel,  31, sizeof(cl_mem), &inmem->taper);
        
        cl_err = clSetKernelArg(*pkernel,  32, sizeof(cl_mem), &inmem->K_x);
        cl_err = clSetKernelArg(*pkernel,  33, sizeof(cl_mem), &inmem->a_x);
        cl_err = clSetKernelArg(*pkernel,  34, sizeof(cl_mem), &inmem->b_x);
        cl_err = clSetKernelArg(*pkernel,  35, sizeof(cl_mem), &inmem->K_x_half);
        cl_err = clSetKernelArg(*pkernel,  36, sizeof(cl_mem), &inmem->a_x_half);
        cl_err = clSetKernelArg(*pkernel,  37, sizeof(cl_mem), &inmem->b_x_half);

        cl_err = clSetKernelArg(*pkernel,  38, sizeof(cl_mem), &inmem->K_y);
        cl_err = clSetKernelArg(*pkernel,  39, sizeof(cl_mem), &inmem->a_y);
        cl_err = clSetKernelArg(*pkernel,  40, sizeof(cl_mem), &inmem->b_y);
        cl_err = clSetKernelArg(*pkernel,  41, sizeof(cl_mem), &inmem->K_y_half);
        cl_err = clSetKernelArg(*pkernel,  42, sizeof(cl_mem), &inmem->a_y_half);
        cl_err = clSetKernelArg(*pkernel,  43, sizeof(cl_mem), &inmem->b_y_half);
        
        cl_err = clSetKernelArg(*pkernel,  44, sizeof(cl_mem), &inmem->K_z);
        cl_err = clSetKernelArg(*pkernel,  45, sizeof(cl_mem), &inmem->a_z);
        cl_err = clSetKernelArg(*pkernel,  46, sizeof(cl_mem), &inmem->b_z);
        cl_err = clSetKernelArg(*pkernel,  47, sizeof(cl_mem), &inmem->K_z_half);
        cl_err = clSetKernelArg(*pkernel,  48, sizeof(cl_mem), &inmem->a_z_half);
        cl_err = clSetKernelArg(*pkernel,  49, sizeof(cl_mem), &inmem->b_z_half);
        
        cl_err = clSetKernelArg(*pkernel,  50, sizeof(cl_mem), &inmem->psi_vxx);
        cl_err = clSetKernelArg(*pkernel,  51, sizeof(cl_mem), &inmem->psi_vxy);
        cl_err = clSetKernelArg(*pkernel,  52, sizeof(cl_mem), &inmem->psi_vxz);
        cl_err = clSetKernelArg(*pkernel,  53, sizeof(cl_mem), &inmem->psi_vyx);
        cl_err = clSetKernelArg(*pkernel,  54, sizeof(cl_mem), &inmem->psi_vyy);
        cl_err = clSetKernelArg(*pkernel,  55, sizeof(cl_mem), &inmem->psi_vyz);
        cl_err = clSetKernelArg(*pkernel,  56, sizeof(cl_mem), &inmem->psi_vzx);
        cl_err = clSetKernelArg(*pkernel,  57, sizeof(cl_mem), &inmem->psi_vzy);
        cl_err = clSetKernelArg(*pkernel,  58, sizeof(cl_mem), &inmem->psi_vzz);
        
        cl_err = clSetKernelArg(*pkernel,  59, shared_size, NULL);
        
    }
    else if ((*inm).ND==2){
        cl_err = clSetKernelArg(*pkernel,  3, sizeof(cl_mem), &inmem->vx);
        cl_err = clSetKernelArg(*pkernel,  4, sizeof(cl_mem), &inmem->vz);
        cl_err = clSetKernelArg(*pkernel,  5, sizeof(cl_mem), &inmem->sxx);
        cl_err = clSetKernelArg(*pkernel,  6, sizeof(cl_mem), &inmem->szz);
        cl_err = clSetKernelArg(*pkernel,  7, sizeof(cl_mem), &inmem->sxz);
        
        
        cl_err = clSetKernelArg(*pkernel,  8, sizeof(cl_mem), &inmem->pi);
        cl_err = clSetKernelArg(*pkernel,  9, sizeof(cl_mem), &inmem->u);
        cl_err = clSetKernelArg(*pkernel,  10, sizeof(cl_mem), &inmem->uipkp);
        
        cl_err = clSetKernelArg(*pkernel,  11, sizeof(cl_mem), &inmem->rxx);
        cl_err = clSetKernelArg(*pkernel,  12, sizeof(cl_mem), &inmem->rzz);
        cl_err = clSetKernelArg(*pkernel,  13, sizeof(cl_mem), &inmem->rxz);
        cl_err = clSetKernelArg(*pkernel,  14, sizeof(cl_mem), &inmem->taus);
        cl_err = clSetKernelArg(*pkernel,  15, sizeof(cl_mem), &inmem->tausipkp);
        cl_err = clSetKernelArg(*pkernel,  16, sizeof(cl_mem), &inmem->taup);
        cl_err = clSetKernelArg(*pkernel,  17, sizeof(cl_mem), &inmem->eta);
        
        
        cl_err = clSetKernelArg(*pkernel,  18, sizeof(cl_mem), &inmem->src_pos);
        cl_err = clSetKernelArg(*pkernel,  19, sizeof(cl_mem), &inmem->src);
        cl_err = clSetKernelArg(*pkernel,  20, sizeof(cl_mem), &inmem->taper);
        
        cl_err = clSetKernelArg(*pkernel,  21, sizeof(cl_mem), &inmem->K_x);
        cl_err = clSetKernelArg(*pkernel,  22, sizeof(cl_mem), &inmem->a_x);
        cl_err = clSetKernelArg(*pkernel,  23, sizeof(cl_mem), &inmem->b_x);
        cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem->K_x_half);
        cl_err = clSetKernelArg(*pkernel,  25, sizeof(cl_mem), &inmem->a_x_half);
        cl_err = clSetKernelArg(*pkernel,  26, sizeof(cl_mem), &inmem->b_x_half);
    
        cl_err = clSetKernelArg(*pkernel,  27, sizeof(cl_mem), &inmem->K_z);
        cl_err = clSetKernelArg(*pkernel,  28, sizeof(cl_mem), &inmem->a_z);
        cl_err = clSetKernelArg(*pkernel,  29, sizeof(cl_mem), &inmem->b_z);
        cl_err = clSetKernelArg(*pkernel,  30, sizeof(cl_mem), &inmem->K_z_half);
        cl_err = clSetKernelArg(*pkernel,  31, sizeof(cl_mem), &inmem->a_z_half);
        cl_err = clSetKernelArg(*pkernel,  32, sizeof(cl_mem), &inmem->b_z_half);
        
        cl_err = clSetKernelArg(*pkernel,  33, sizeof(cl_mem), &inmem->psi_vxx);
        cl_err = clSetKernelArg(*pkernel,  34, sizeof(cl_mem), &inmem->psi_vxz);
        cl_err = clSetKernelArg(*pkernel,  35, sizeof(cl_mem), &inmem->psi_vzx);
        cl_err = clSetKernelArg(*pkernel,  36, sizeof(cl_mem), &inmem->psi_vzz);
        
        cl_err = clSetKernelArg(*pkernel,  37, shared_size, NULL);
        
    }
    else if ((*inm).ND==21){
        cl_err = clSetKernelArg(*pkernel,  3, sizeof(cl_mem), &inmem->vy);
        cl_err = clSetKernelArg(*pkernel,  4, sizeof(cl_mem), &inmem->sxy);
        cl_err = clSetKernelArg(*pkernel,  5, sizeof(cl_mem), &inmem->syz);

        cl_err = clSetKernelArg(*pkernel,  6, sizeof(cl_mem), &inmem->uipjp);
        cl_err = clSetKernelArg(*pkernel,  7, sizeof(cl_mem), &inmem->ujpkp);
        
        cl_err = clSetKernelArg(*pkernel,  8, sizeof(cl_mem), &inmem->rxy);
        cl_err = clSetKernelArg(*pkernel,  9, sizeof(cl_mem), &inmem->ryz);
        
        cl_err = clSetKernelArg(*pkernel,  10, sizeof(cl_mem), &inmem->tausipjp);
        cl_err = clSetKernelArg(*pkernel,  11, sizeof(cl_mem), &inmem->tausjpkp);

        cl_err = clSetKernelArg(*pkernel,  12, sizeof(cl_mem), &inmem->eta);
        
        
        cl_err = clSetKernelArg(*pkernel,  13, sizeof(cl_mem), &inmem->src_pos);
        cl_err = clSetKernelArg(*pkernel,  14, sizeof(cl_mem), &inmem->src);
        cl_err = clSetKernelArg(*pkernel,  15, sizeof(cl_mem), &inmem->taper);
        
        cl_err = clSetKernelArg(*pkernel,  16, sizeof(cl_mem), &inmem->K_x);
        cl_err = clSetKernelArg(*pkernel,  17, sizeof(cl_mem), &inmem->a_x);
        cl_err = clSetKernelArg(*pkernel,  18, sizeof(cl_mem), &inmem->b_x);
        cl_err = clSetKernelArg(*pkernel,  19, sizeof(cl_mem), &inmem->K_x_half);
        cl_err = clSetKernelArg(*pkernel,  20, sizeof(cl_mem), &inmem->a_x_half);
        cl_err = clSetKernelArg(*pkernel,  21, sizeof(cl_mem), &inmem->b_x_half);
        
        cl_err = clSetKernelArg(*pkernel,  22, sizeof(cl_mem), &inmem->K_z);
        cl_err = clSetKernelArg(*pkernel,  23, sizeof(cl_mem), &inmem->a_z);
        cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem->b_z);
        cl_err = clSetKernelArg(*pkernel,  25, sizeof(cl_mem), &inmem->K_z_half);
        cl_err = clSetKernelArg(*pkernel,  26, sizeof(cl_mem), &inmem->a_z_half);
        cl_err = clSetKernelArg(*pkernel,  27, sizeof(cl_mem), &inmem->b_z_half);
        
        cl_err = clSetKernelArg(*pkernel,  28, sizeof(cl_mem), &inmem->psi_vyx);
        cl_err = clSetKernelArg(*pkernel,  29, sizeof(cl_mem), &inmem->psi_vyz);
        
        cl_err = clSetKernelArg(*pkernel,  30, shared_size, NULL);
        
    }
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
	return cl_err;
}

int gpu_initialize_surface(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc )


{
    cl_int cl_err = 0;
    
    
    char filename[20];
    if ((*inm).ND==3){
        sprintf(filename,"./surface3D.cl");
    }
    else if ((*inm).ND==2){
        sprintf(filename,"./surface2D.cl");
    }
    else if ((*inm).ND==21){
        sprintf(filename,"./surface2D_SH.cl");
    }
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, 0, 0, 0);

    /*Create the kernel*/
    const char * program_name = "surface";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    /*Define the arguments for this kernel */
    if ((*inm).ND==3){
        cl_err = clSetKernelArg(*pkernel,  0, sizeof(cl_mem), &inmem->vx);
        cl_err = clSetKernelArg(*pkernel,  1, sizeof(cl_mem), &inmem->vy);
        cl_err = clSetKernelArg(*pkernel,  2, sizeof(cl_mem), &inmem->vz);
        cl_err = clSetKernelArg(*pkernel,  3, sizeof(cl_mem), &inmem->sxx);
        cl_err = clSetKernelArg(*pkernel,  4, sizeof(cl_mem), &inmem->syy);
        cl_err = clSetKernelArg(*pkernel,  5, sizeof(cl_mem), &inmem->szz);
        cl_err = clSetKernelArg(*pkernel,  6, sizeof(cl_mem), &inmem->sxy);
        cl_err = clSetKernelArg(*pkernel,  7, sizeof(cl_mem), &inmem->syz);
        cl_err = clSetKernelArg(*pkernel,  8, sizeof(cl_mem), &inmem->sxz);
        
        
        cl_err = clSetKernelArg(*pkernel,  9, sizeof(cl_mem), &inmem->pi);
        cl_err = clSetKernelArg(*pkernel,  10, sizeof(cl_mem), &inmem->u);
        
        cl_err = clSetKernelArg(*pkernel,  11, sizeof(cl_mem), &inmem->rxx);
        cl_err = clSetKernelArg(*pkernel,  12, sizeof(cl_mem), &inmem->ryy);
        cl_err = clSetKernelArg(*pkernel,  13, sizeof(cl_mem), &inmem->rzz);
        cl_err = clSetKernelArg(*pkernel,  14, sizeof(cl_mem), &inmem->taus);
        cl_err = clSetKernelArg(*pkernel,  15, sizeof(cl_mem), &inmem->taup);
        cl_err = clSetKernelArg(*pkernel,  16, sizeof(cl_mem), &inmem->eta);
    }
    else if ((*inm).ND==2){
        cl_err = clSetKernelArg(*pkernel,  0, sizeof(cl_mem), &inmem->vx);
        cl_err = clSetKernelArg(*pkernel,  1, sizeof(cl_mem), &inmem->vz);
        cl_err = clSetKernelArg(*pkernel,  2, sizeof(cl_mem), &inmem->sxx);
        cl_err = clSetKernelArg(*pkernel,  3, sizeof(cl_mem), &inmem->szz);
        cl_err = clSetKernelArg(*pkernel,  4, sizeof(cl_mem), &inmem->sxz);
        
        
        cl_err = clSetKernelArg(*pkernel,  5, sizeof(cl_mem), &inmem->pi);
        cl_err = clSetKernelArg(*pkernel,  6, sizeof(cl_mem), &inmem->u);
        
        cl_err = clSetKernelArg(*pkernel,  7, sizeof(cl_mem), &inmem->rxx);
        cl_err = clSetKernelArg(*pkernel,  8, sizeof(cl_mem), &inmem->rzz);
        cl_err = clSetKernelArg(*pkernel,  9, sizeof(cl_mem), &inmem->taus);
        cl_err = clSetKernelArg(*pkernel,  10, sizeof(cl_mem), &inmem->taup);
        cl_err = clSetKernelArg(*pkernel,  11, sizeof(cl_mem), &inmem->eta);
        
    }
    else if ((*inm).ND==21){
        cl_err = clSetKernelArg(*pkernel,  0, sizeof(cl_mem), &inmem->syz);
    }
 
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
    return cl_err;
}




int gpu_intialize_seis(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc ){
    
    
	cl_int cl_err = 0;
    
    
    
    const char * filename = "./initialize.cl";
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, 0, 0, 0);

    /*Create the kernel*/
    const char * program_name = "initialize_seis";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    
    /*Define the arguments for this kernel */
    cl_err = clSetKernelArg(*pkernel,  0,  sizeof(cl_mem), &inmem->vx);
    cl_err = clSetKernelArg(*pkernel,  1,  sizeof(cl_mem), &inmem->vy);
    cl_err = clSetKernelArg(*pkernel,  2,  sizeof(cl_mem), &inmem->vz);
    cl_err = clSetKernelArg(*pkernel,  3,  sizeof(cl_mem), &inmem->sxx);
    cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem->syy);
    cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem->szz);
    cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem->sxy);
    cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem->syz);
    cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem->sxz);
    cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem->rxx);
    cl_err = clSetKernelArg(*pkernel,  10, sizeof(cl_mem), &inmem->ryy);
    cl_err = clSetKernelArg(*pkernel,  11, sizeof(cl_mem), &inmem->rzz);
    cl_err = clSetKernelArg(*pkernel,  12, sizeof(cl_mem), &inmem->rxy);
    cl_err = clSetKernelArg(*pkernel,  13, sizeof(cl_mem), &inmem->ryz);
    cl_err = clSetKernelArg(*pkernel,  14, sizeof(cl_mem), &inmem->rxz);
    cl_err = clSetKernelArg(*pkernel,  15, sizeof(cl_mem), &inmem->psi_sxx_x);
    cl_err = clSetKernelArg(*pkernel,  16, sizeof(cl_mem), &inmem->psi_sxy_x);
    cl_err = clSetKernelArg(*pkernel,  17, sizeof(cl_mem), &inmem->psi_sxy_y);
    cl_err = clSetKernelArg(*pkernel,  18, sizeof(cl_mem), &inmem->psi_sxz_x);
    cl_err = clSetKernelArg(*pkernel,  19, sizeof(cl_mem), &inmem->psi_sxz_z);
    cl_err = clSetKernelArg(*pkernel,  20, sizeof(cl_mem), &inmem->psi_syy_y);
    cl_err = clSetKernelArg(*pkernel,  21, sizeof(cl_mem), &inmem->psi_syz_y);
    cl_err = clSetKernelArg(*pkernel,  22, sizeof(cl_mem), &inmem->psi_syz_z);
    cl_err = clSetKernelArg(*pkernel,  23, sizeof(cl_mem), &inmem->psi_szz_z);
    cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem->psi_vxx);
    cl_err = clSetKernelArg(*pkernel,  25, sizeof(cl_mem), &inmem->psi_vxy);
    cl_err = clSetKernelArg(*pkernel,  26, sizeof(cl_mem), &inmem->psi_vxz);
    cl_err = clSetKernelArg(*pkernel,  27, sizeof(cl_mem), &inmem->psi_vyx);
    cl_err = clSetKernelArg(*pkernel,  28, sizeof(cl_mem), &inmem->psi_vyy);
    cl_err = clSetKernelArg(*pkernel,  29, sizeof(cl_mem), &inmem->psi_vyz);
    cl_err = clSetKernelArg(*pkernel,  30, sizeof(cl_mem), &inmem->psi_vzx);
    cl_err = clSetKernelArg(*pkernel,  31, sizeof(cl_mem), &inmem->psi_vzy);
    cl_err = clSetKernelArg(*pkernel,  32, sizeof(cl_mem), &inmem->psi_vzz);
    
    
    
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
    return cl_err;
    
}

int gpu_intialize_seis_r(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc ){
    
    
	cl_int cl_err = 0;
    
    
    
    const char * filename = "./initialize.cl";
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, 0, 0, 0);
    
    
    
    /*Create the kernel, ther kernel version depends on the finite difference order*/
    const char * program_name = "initialize_seis";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    
    /*Define the arguments for this kernel */
    if (inm->back_prop_type==1){
        cl_err = clSetKernelArg(*pkernel,  0,  sizeof(cl_mem), &inmem->vx_r);
        cl_err = clSetKernelArg(*pkernel,  1,  sizeof(cl_mem), &inmem->vy_r);
        cl_err = clSetKernelArg(*pkernel,  2,  sizeof(cl_mem), &inmem->vz_r);
        cl_err = clSetKernelArg(*pkernel,  3,  sizeof(cl_mem), &inmem->sxx_r);
        cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem->syy_r);
        cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem->szz_r);
        cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem->sxy_r);
        cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem->syz_r);
        cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem->sxz_r);
        cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem->rxx_r);
        cl_err = clSetKernelArg(*pkernel,  10, sizeof(cl_mem), &inmem->ryy_r);
        cl_err = clSetKernelArg(*pkernel,  11, sizeof(cl_mem), &inmem->rzz_r);
        cl_err = clSetKernelArg(*pkernel,  12, sizeof(cl_mem), &inmem->rxy_r);
        cl_err = clSetKernelArg(*pkernel,  13, sizeof(cl_mem), &inmem->ryz_r);
        cl_err = clSetKernelArg(*pkernel,  14, sizeof(cl_mem), &inmem->rxz_r);
    }
    else if (inm->back_prop_type==2){
        cl_err = clSetKernelArg(*pkernel,  0,  sizeof(cl_mem), &inmem->vx);
        cl_err = clSetKernelArg(*pkernel,  1,  sizeof(cl_mem), &inmem->vy);
        cl_err = clSetKernelArg(*pkernel,  2,  sizeof(cl_mem), &inmem->vz);
        cl_err = clSetKernelArg(*pkernel,  3,  sizeof(cl_mem), &inmem->sxx);
        cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem->syy);
        cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem->szz);
        cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem->sxy);
        cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem->syz);
        cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem->sxz);
        cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem->rxx);
        cl_err = clSetKernelArg(*pkernel,  10, sizeof(cl_mem), &inmem->ryy);
        cl_err = clSetKernelArg(*pkernel,  11, sizeof(cl_mem), &inmem->rzz);
        cl_err = clSetKernelArg(*pkernel,  12, sizeof(cl_mem), &inmem->rxy);
        cl_err = clSetKernelArg(*pkernel,  13, sizeof(cl_mem), &inmem->ryz);
        cl_err = clSetKernelArg(*pkernel,  14, sizeof(cl_mem), &inmem->rxz);
        
    }
    cl_err = clSetKernelArg(*pkernel,  15, sizeof(cl_mem), &inmem->psi_sxx_x);
    cl_err = clSetKernelArg(*pkernel,  16, sizeof(cl_mem), &inmem->psi_sxy_x);
    cl_err = clSetKernelArg(*pkernel,  17, sizeof(cl_mem), &inmem->psi_sxy_y);
    cl_err = clSetKernelArg(*pkernel,  18, sizeof(cl_mem), &inmem->psi_sxz_x);
    cl_err = clSetKernelArg(*pkernel,  19, sizeof(cl_mem), &inmem->psi_sxz_z);
    cl_err = clSetKernelArg(*pkernel,  20, sizeof(cl_mem), &inmem->psi_syy_y);
    cl_err = clSetKernelArg(*pkernel,  21, sizeof(cl_mem), &inmem->psi_syz_y);
    cl_err = clSetKernelArg(*pkernel,  22, sizeof(cl_mem), &inmem->psi_syz_z);
    cl_err = clSetKernelArg(*pkernel,  23, sizeof(cl_mem), &inmem->psi_szz_z);
    cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem->psi_vxx);
    cl_err = clSetKernelArg(*pkernel,  25, sizeof(cl_mem), &inmem->psi_vxy);
    cl_err = clSetKernelArg(*pkernel,  26, sizeof(cl_mem), &inmem->psi_vxz);
    cl_err = clSetKernelArg(*pkernel,  27, sizeof(cl_mem), &inmem->psi_vyx);
    cl_err = clSetKernelArg(*pkernel,  28, sizeof(cl_mem), &inmem->psi_vyy);
    cl_err = clSetKernelArg(*pkernel,  29, sizeof(cl_mem), &inmem->psi_vyz);
    cl_err = clSetKernelArg(*pkernel,  30, sizeof(cl_mem), &inmem->psi_vzx);
    cl_err = clSetKernelArg(*pkernel,  31, sizeof(cl_mem), &inmem->psi_vzy);
    cl_err = clSetKernelArg(*pkernel,  32, sizeof(cl_mem), &inmem->psi_vzz);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
    return cl_err;
    
}

int gpu_intialize_grad(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc  ){
    
    
	cl_int cl_err = 0;
    
    
    
    const char * filename = "./initialize.cl";
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, 0, 0, 1);
    
    
    
    /*Create the kernel, ther kernel version depends on the finite difference order*/
    const char * program_name = "initialize_grad";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    
    /*Define the arguments for this kernel */
    cl_err = clSetKernelArg(*pkernel,  0, sizeof(cl_mem), &inmem->gradrho);
    cl_err = clSetKernelArg(*pkernel,  1, sizeof(cl_mem), &inmem->gradM);
    cl_err = clSetKernelArg(*pkernel,  2, sizeof(cl_mem), &inmem->gradmu);
    cl_err = clSetKernelArg(*pkernel,  3, sizeof(cl_mem), &inmem->gradtaup);
    cl_err = clSetKernelArg(*pkernel,  4, sizeof(cl_mem), &inmem->gradtaus);
    

    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
    return cl_err;
    
}


int gpu_intialize_vout(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc ){
    
    
	cl_int cl_err = 0;
    
    
    
    const char * filename = "./vout.cl";
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, 0, 0, 0);
    
    
    
    /*Create the kernel, ther kernel version depends on the finite difference order*/
    const char * program_name = "vout";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    
    /*Define the arguments for this kernel */
    cl_err = clSetKernelArg(*pkernel,  0, sizeof(cl_mem), &inmem->vx);
    cl_err = clSetKernelArg(*pkernel,  1, sizeof(cl_mem), &inmem->vy);
    cl_err = clSetKernelArg(*pkernel,  2, sizeof(cl_mem), &inmem->vz);
    cl_err = clSetKernelArg(*pkernel,  3, sizeof(cl_mem), &inmem->vxout);
    cl_err = clSetKernelArg(*pkernel,  4, sizeof(cl_mem), &inmem->vyout);
    cl_err = clSetKernelArg(*pkernel,  5, sizeof(cl_mem), &inmem->vzout);
    cl_err = clSetKernelArg(*pkernel,  6, sizeof(cl_mem), &inmem->rec_pos);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
    return cl_err;
    
}

int gpu_intialize_voutinit(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc ){
    
    
    cl_int cl_err = 0;
    
    
    
    const char * filename = "./initialize.cl";
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, 0, 0, 0);
    
    
    
    /*Create the kernel, ther kernel version depends on the finite difference order*/
    const char * program_name = "voutinit";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    
    /*Define the arguments for this kernel */
    cl_err = clSetKernelArg(*pkernel,  0, sizeof(cl_mem), &inmem->vxout);
    cl_err = clSetKernelArg(*pkernel,  1, sizeof(cl_mem), &inmem->vyout);
    cl_err = clSetKernelArg(*pkernel,  2, sizeof(cl_mem), &inmem->vzout);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
    return cl_err;
    
}



int gpu_intialize_residuals(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc ){
    
    
    cl_int cl_err = 0;
    
    
    
    const char * filename = "./residuals.cl";
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, 0, 0, 0);
    
    
    
    /*Create the kernel, ther kernel version depends on the finite difference order*/
    const char * program_name = "residuals";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    
    /*Define the arguments for this kernel */
    if (inm->back_prop_type==1){
        cl_err = clSetKernelArg(*pkernel,  0, sizeof(cl_mem), &inmem->vx_r);
        cl_err = clSetKernelArg(*pkernel,  1, sizeof(cl_mem), &inmem->vy_r);
        cl_err = clSetKernelArg(*pkernel,  2, sizeof(cl_mem), &inmem->vz_r);
    }
    else if (inm->back_prop_type==2){
        
        cl_err = clSetKernelArg(*pkernel,  0, sizeof(cl_mem), &inmem->vx);
        cl_err = clSetKernelArg(*pkernel,  1, sizeof(cl_mem), &inmem->vy);
        cl_err = clSetKernelArg(*pkernel,  2, sizeof(cl_mem), &inmem->vz);
    }
    cl_err = clSetKernelArg(*pkernel,  3, sizeof(cl_mem), &inmem->vxout);
    cl_err = clSetKernelArg(*pkernel,  4, sizeof(cl_mem), &inmem->vyout);
    cl_err = clSetKernelArg(*pkernel,  5, sizeof(cl_mem), &inmem->vzout);
    cl_err = clSetKernelArg(*pkernel,  6, sizeof(cl_mem), &inmem->rip);
    cl_err = clSetKernelArg(*pkernel,  7, sizeof(cl_mem), &inmem->rjp);
    cl_err = clSetKernelArg(*pkernel,  8, sizeof(cl_mem), &inmem->rkp);
    cl_err = clSetKernelArg(*pkernel,  9, sizeof(cl_mem), &inmem->rec_pos);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
    return cl_err;
    
}


int gpu_initialize_update_adjv(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc, int bndoff, int lcomm, int comm  )
{
	
	cl_int cl_err = 0;
    size_t shared_size=sizeof(float);
    
    char filename[50];
    if ((*inm).ND==3){
        sprintf(filename,"./update_adjv3D.cl");
    }
    else if ((*inm).ND==2){
        sprintf(filename,"./update_adjv2D.cl");
    }
    else if ((*inm).ND==21){
        sprintf(filename,"./update_adjv2D_SH.cl");
    }
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, lcomm, comm, 1);
    
    /*Create the kernel, ther kernel version depends on the finite difference order*/
    const char * program_name = "update_adjv";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    /*Define the size of the local variables of the compute device*/
    if ((*inmloc).local_off==0){
        if ((*inm).ND==3){//For 3D
            shared_size = ((local_work_size[0]+(*inm).FDORDER)*(local_work_size[1]+(*inm).FDORDER)*(local_work_size[2]+(*inm).FDORDER)) * sizeof(float);
        }
        else if ((*inm).ND==2 || (*inm).ND==21){//For 2D
            shared_size = ((local_work_size[0]+(*inm).FDORDER)*(local_work_size[1]+(*inm).FDORDER)) * sizeof(float);
        }
    }
    
    /*Define the arguments for this kernel */
    cl_err = clSetKernelArg(*pkernel,  0, sizeof(int), &bndoff);
    if ((*inm).ND==3){//For 3D
        cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem->vx);
        cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem->vy);
        cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem->vz);
        cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem->sxx);
        cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem->syy);
        cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem->szz);
        cl_err = clSetKernelArg(*pkernel,  10,  sizeof(cl_mem), &inmem->sxy);
        cl_err = clSetKernelArg(*pkernel,  11,  sizeof(cl_mem), &inmem->syz);
        cl_err = clSetKernelArg(*pkernel,  12,  sizeof(cl_mem), &inmem->sxz);
        
        cl_err = clSetKernelArg(*pkernel,  13,  sizeof(cl_mem), &inmem->vxbnd);
        cl_err = clSetKernelArg(*pkernel,  14,  sizeof(cl_mem), &inmem->vybnd);
        cl_err = clSetKernelArg(*pkernel,  15,  sizeof(cl_mem), &inmem->vzbnd);
        cl_err = clSetKernelArg(*pkernel,  16,  sizeof(cl_mem), &inmem->sxxbnd);
        cl_err = clSetKernelArg(*pkernel,  17,  sizeof(cl_mem), &inmem->syybnd);
        cl_err = clSetKernelArg(*pkernel,  18,  sizeof(cl_mem), &inmem->szzbnd);
        cl_err = clSetKernelArg(*pkernel,  19,  sizeof(cl_mem), &inmem->sxybnd);
        cl_err = clSetKernelArg(*pkernel,  20,  sizeof(cl_mem), &inmem->syzbnd);
        cl_err = clSetKernelArg(*pkernel,  21,  sizeof(cl_mem), &inmem->sxzbnd);
        
        if (inm->back_prop_type==1){
            cl_err = clSetKernelArg(*pkernel,  22,  sizeof(cl_mem), &inmem->vx_r);
            cl_err = clSetKernelArg(*pkernel,  23,  sizeof(cl_mem), &inmem->vy_r);
            cl_err = clSetKernelArg(*pkernel,  24,  sizeof(cl_mem), &inmem->vz_r);
            cl_err = clSetKernelArg(*pkernel,  25,  sizeof(cl_mem), &inmem->sxx_r);
            cl_err = clSetKernelArg(*pkernel,  26,  sizeof(cl_mem), &inmem->syy_r);
            cl_err = clSetKernelArg(*pkernel,  27,  sizeof(cl_mem), &inmem->szz_r);
            cl_err = clSetKernelArg(*pkernel,  28,  sizeof(cl_mem), &inmem->sxy_r);
            cl_err = clSetKernelArg(*pkernel,  29,  sizeof(cl_mem), &inmem->syz_r);
            cl_err = clSetKernelArg(*pkernel,  30,  sizeof(cl_mem), &inmem->sxz_r);
        }
        else if (inm->back_prop_type==2){
            cl_err = clSetKernelArg(*pkernel,  22,  sizeof(cl_mem), &inmem->vx);
            cl_err = clSetKernelArg(*pkernel,  23,  sizeof(cl_mem), &inmem->vy);
            cl_err = clSetKernelArg(*pkernel,  24,  sizeof(cl_mem), &inmem->vz);
            cl_err = clSetKernelArg(*pkernel,  25,  sizeof(cl_mem), &inmem->sxx);
            cl_err = clSetKernelArg(*pkernel,  26,  sizeof(cl_mem), &inmem->syy);
            cl_err = clSetKernelArg(*pkernel,  27,  sizeof(cl_mem), &inmem->szz);
            cl_err = clSetKernelArg(*pkernel,  28,  sizeof(cl_mem), &inmem->sxy);
            cl_err = clSetKernelArg(*pkernel,  29,  sizeof(cl_mem), &inmem->syz);
            cl_err = clSetKernelArg(*pkernel,  30,  sizeof(cl_mem), &inmem->sxz);
        }
        
        cl_err = clSetKernelArg(*pkernel,  31, sizeof(cl_mem), &inmem->vxout);
        cl_err = clSetKernelArg(*pkernel,  32, sizeof(cl_mem), &inmem->vyout);
        cl_err = clSetKernelArg(*pkernel,  33, sizeof(cl_mem), &inmem->vzout);
        
        cl_err = clSetKernelArg(*pkernel,  34, sizeof(cl_mem), &inmem->rip);
        cl_err = clSetKernelArg(*pkernel,  35, sizeof(cl_mem), &inmem->rjp);
        cl_err = clSetKernelArg(*pkernel,  36, sizeof(cl_mem), &inmem->rkp);
        cl_err = clSetKernelArg(*pkernel,  37, sizeof(cl_mem), &inmem->src_pos);
        cl_err = clSetKernelArg(*pkernel,  38, sizeof(cl_mem), &inmem->src);
        cl_err = clSetKernelArg(*pkernel,  39, sizeof(cl_mem), &inmem->rec_pos);
        cl_err = clSetKernelArg(*pkernel,  40, sizeof(cl_mem), &inmem->taper);
        
        cl_err = clSetKernelArg(*pkernel,  41, sizeof(cl_mem), &inmem->K_x);
        cl_err = clSetKernelArg(*pkernel,  42, sizeof(cl_mem), &inmem->a_x);
        cl_err = clSetKernelArg(*pkernel,  43, sizeof(cl_mem), &inmem->b_x);
        cl_err = clSetKernelArg(*pkernel,  44, sizeof(cl_mem), &inmem->K_x_half);
        cl_err = clSetKernelArg(*pkernel,  45, sizeof(cl_mem), &inmem->a_x_half);
        cl_err = clSetKernelArg(*pkernel,  46, sizeof(cl_mem), &inmem->b_x_half);
        
        cl_err = clSetKernelArg(*pkernel,  47, sizeof(cl_mem), &inmem->K_y);
        cl_err = clSetKernelArg(*pkernel,  48, sizeof(cl_mem), &inmem->a_y);
        cl_err = clSetKernelArg(*pkernel,  49, sizeof(cl_mem), &inmem->b_y);
        cl_err = clSetKernelArg(*pkernel,  50, sizeof(cl_mem), &inmem->K_y_half);
        cl_err = clSetKernelArg(*pkernel,  51, sizeof(cl_mem), &inmem->a_y_half);
        cl_err = clSetKernelArg(*pkernel,  52, sizeof(cl_mem), &inmem->b_y_half);
        
        cl_err = clSetKernelArg(*pkernel,  53, sizeof(cl_mem), &inmem->K_z);
        cl_err = clSetKernelArg(*pkernel,  54, sizeof(cl_mem), &inmem->a_z);
        cl_err = clSetKernelArg(*pkernel,  55, sizeof(cl_mem), &inmem->b_z);
        cl_err = clSetKernelArg(*pkernel,  56, sizeof(cl_mem), &inmem->K_z_half);
        cl_err = clSetKernelArg(*pkernel,  57, sizeof(cl_mem), &inmem->a_z_half);
        cl_err = clSetKernelArg(*pkernel,  58, sizeof(cl_mem), &inmem->b_z_half);
        
        cl_err = clSetKernelArg(*pkernel,  59, sizeof(cl_mem), &inmem->psi_sxx_x);
        cl_err = clSetKernelArg(*pkernel,  60, sizeof(cl_mem), &inmem->psi_sxy_x);
        cl_err = clSetKernelArg(*pkernel,  61, sizeof(cl_mem), &inmem->psi_sxy_y);
        cl_err = clSetKernelArg(*pkernel,  62, sizeof(cl_mem), &inmem->psi_sxz_x);
        cl_err = clSetKernelArg(*pkernel,  63, sizeof(cl_mem), &inmem->psi_sxz_z);
        cl_err = clSetKernelArg(*pkernel,  64, sizeof(cl_mem), &inmem->psi_syy_y);
        cl_err = clSetKernelArg(*pkernel,  65, sizeof(cl_mem), &inmem->psi_syz_y);
        cl_err = clSetKernelArg(*pkernel,  66, sizeof(cl_mem), &inmem->psi_syz_z);
        cl_err = clSetKernelArg(*pkernel,  67, sizeof(cl_mem), &inmem->psi_szz_z);
        
        cl_err = clSetKernelArg(*pkernel,  68, shared_size, NULL);
        
        cl_err = clSetKernelArg(*pkernel,  69, sizeof(cl_mem), &inmem->gradrho);
        cl_err = clSetKernelArg(*pkernel,  70, sizeof(cl_mem), &inmem->gradsrc);

    }
    else if ((*inm).ND==2){//For 2D
        cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem->vx);
        cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem->vz);
        cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem->sxx);
        cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem->szz);
        cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem->sxz);
        
        cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem->vxbnd);
        cl_err = clSetKernelArg(*pkernel,  10,  sizeof(cl_mem), &inmem->vzbnd);
        cl_err = clSetKernelArg(*pkernel,  11,  sizeof(cl_mem), &inmem->sxxbnd);
        cl_err = clSetKernelArg(*pkernel,  12,  sizeof(cl_mem), &inmem->szzbnd);
        cl_err = clSetKernelArg(*pkernel,  13,  sizeof(cl_mem), &inmem->sxzbnd);
        
        if (inm->back_prop_type==1){
            cl_err = clSetKernelArg(*pkernel,  14,  sizeof(cl_mem), &inmem->vx_r);
            cl_err = clSetKernelArg(*pkernel,  15,  sizeof(cl_mem), &inmem->vz_r);
            cl_err = clSetKernelArg(*pkernel,  16,  sizeof(cl_mem), &inmem->sxx_r);
            cl_err = clSetKernelArg(*pkernel,  17,  sizeof(cl_mem), &inmem->szz_r);
            cl_err = clSetKernelArg(*pkernel,  18,  sizeof(cl_mem), &inmem->sxz_r);
        }
        else if (inm->back_prop_type==2){
            cl_err = clSetKernelArg(*pkernel,  14,  sizeof(cl_mem), &inmem->vx);
            cl_err = clSetKernelArg(*pkernel,  15,  sizeof(cl_mem), &inmem->vz);
            cl_err = clSetKernelArg(*pkernel,  16,  sizeof(cl_mem), &inmem->sxx);
            cl_err = clSetKernelArg(*pkernel,  17,  sizeof(cl_mem), &inmem->szz);
            cl_err = clSetKernelArg(*pkernel,  18,  sizeof(cl_mem), &inmem->sxz);
        }
        
        cl_err = clSetKernelArg(*pkernel,  19, sizeof(cl_mem), &inmem->vxout);
        cl_err = clSetKernelArg(*pkernel,  20, sizeof(cl_mem), &inmem->vzout);
        
        cl_err = clSetKernelArg(*pkernel,  21, sizeof(cl_mem), &inmem->rip);
        cl_err = clSetKernelArg(*pkernel,  22, sizeof(cl_mem), &inmem->rkp);
        cl_err = clSetKernelArg(*pkernel,  23, sizeof(cl_mem), &inmem->src_pos);
        cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem->src);
        cl_err = clSetKernelArg(*pkernel,  25, sizeof(cl_mem), &inmem->rec_pos);
        cl_err = clSetKernelArg(*pkernel,  26, sizeof(cl_mem), &inmem->taper);
        
        cl_err = clSetKernelArg(*pkernel,  27, sizeof(cl_mem), &inmem->K_x);
        cl_err = clSetKernelArg(*pkernel,  28, sizeof(cl_mem), &inmem->a_x);
        cl_err = clSetKernelArg(*pkernel,  29, sizeof(cl_mem), &inmem->b_x);
        cl_err = clSetKernelArg(*pkernel,  30, sizeof(cl_mem), &inmem->K_x_half);
        cl_err = clSetKernelArg(*pkernel,  31, sizeof(cl_mem), &inmem->a_x_half);
        cl_err = clSetKernelArg(*pkernel,  32, sizeof(cl_mem), &inmem->b_x_half);
        
        cl_err = clSetKernelArg(*pkernel,  33, sizeof(cl_mem), &inmem->K_z);
        cl_err = clSetKernelArg(*pkernel,  34, sizeof(cl_mem), &inmem->a_z);
        cl_err = clSetKernelArg(*pkernel,  35, sizeof(cl_mem), &inmem->b_z);
        cl_err = clSetKernelArg(*pkernel,  36, sizeof(cl_mem), &inmem->K_z_half);
        cl_err = clSetKernelArg(*pkernel,  37, sizeof(cl_mem), &inmem->a_z_half);
        cl_err = clSetKernelArg(*pkernel,  38, sizeof(cl_mem), &inmem->b_z_half);
        
        cl_err = clSetKernelArg(*pkernel,  39, sizeof(cl_mem), &inmem->psi_sxx_x);
        cl_err = clSetKernelArg(*pkernel,  40, sizeof(cl_mem), &inmem->psi_sxz_x);
        cl_err = clSetKernelArg(*pkernel,  41, sizeof(cl_mem), &inmem->psi_sxz_z);
        cl_err = clSetKernelArg(*pkernel,  42, sizeof(cl_mem), &inmem->psi_szz_z);
        
        cl_err = clSetKernelArg(*pkernel,  43, shared_size, NULL);
        
        cl_err = clSetKernelArg(*pkernel,  44, sizeof(cl_mem), &inmem->gradrho);
        cl_err = clSetKernelArg(*pkernel,  45, sizeof(cl_mem), &inmem->gradsrc);

    }
    else if ((*inm).ND==21){//For 2D SH
        cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem->vy);
        cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem->sxy);
        cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem->syz);
        cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem->vybnd);
        cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem->sxybnd);
        cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem->syzbnd);
        if (inm->back_prop_type==1){
            cl_err = clSetKernelArg(*pkernel,  10,  sizeof(cl_mem), &inmem->vy_r);
            cl_err = clSetKernelArg(*pkernel,  11,  sizeof(cl_mem), &inmem->sxy_r);
            cl_err = clSetKernelArg(*pkernel,  12,  sizeof(cl_mem), &inmem->syz_r);
        }
        else if (inm->back_prop_type==2){
            cl_err = clSetKernelArg(*pkernel,  10,  sizeof(cl_mem), &inmem->vy);
            cl_err = clSetKernelArg(*pkernel,  11,  sizeof(cl_mem), &inmem->sxy);
            cl_err = clSetKernelArg(*pkernel,  12,  sizeof(cl_mem), &inmem->syz);
        }
        

        cl_err = clSetKernelArg(*pkernel,  13,  sizeof(cl_mem), &inmem->vyout);
        
        cl_err = clSetKernelArg(*pkernel,  14, sizeof(cl_mem), &inmem->rjp);

        cl_err = clSetKernelArg(*pkernel,  15, sizeof(cl_mem), &inmem->src_pos);
        cl_err = clSetKernelArg(*pkernel,  16, sizeof(cl_mem), &inmem->src);
        cl_err = clSetKernelArg(*pkernel,  17, sizeof(cl_mem), &inmem->rec_pos);
        cl_err = clSetKernelArg(*pkernel,  18, sizeof(cl_mem), &inmem->taper);
        
        cl_err = clSetKernelArg(*pkernel,  19, sizeof(cl_mem), &inmem->K_x);
        cl_err = clSetKernelArg(*pkernel,  20, sizeof(cl_mem), &inmem->a_x);
        cl_err = clSetKernelArg(*pkernel,  21, sizeof(cl_mem), &inmem->b_x);
        cl_err = clSetKernelArg(*pkernel,  22, sizeof(cl_mem), &inmem->K_x_half);
        cl_err = clSetKernelArg(*pkernel,  23, sizeof(cl_mem), &inmem->a_x_half);
        cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem->b_x_half);
        
        cl_err = clSetKernelArg(*pkernel,  25, sizeof(cl_mem), &inmem->K_z);
        cl_err = clSetKernelArg(*pkernel,  26, sizeof(cl_mem), &inmem->a_z);
        cl_err = clSetKernelArg(*pkernel,  27, sizeof(cl_mem), &inmem->b_z);
        cl_err = clSetKernelArg(*pkernel,  28, sizeof(cl_mem), &inmem->K_z_half);
        cl_err = clSetKernelArg(*pkernel,  29, sizeof(cl_mem), &inmem->a_z_half);
        cl_err = clSetKernelArg(*pkernel,  30, sizeof(cl_mem), &inmem->b_z_half);
        
        cl_err = clSetKernelArg(*pkernel,  31, sizeof(cl_mem), &inmem->psi_sxy_x);
        cl_err = clSetKernelArg(*pkernel,  32, sizeof(cl_mem), &inmem->psi_syz_z);

        
        cl_err = clSetKernelArg(*pkernel,  33, shared_size, NULL);
        
        cl_err = clSetKernelArg(*pkernel,  34, sizeof(cl_mem), &inmem->gradrho);
        cl_err = clSetKernelArg(*pkernel,  35, sizeof(cl_mem), &inmem->gradsrc);
        
    }
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
	return cl_err;
}

int gpu_initialize_update_adjs(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc, int bndoff, int lcomm, int comm  )


{
	
	cl_int cl_err = 0;
    size_t shared_size=sizeof(float);
    
    

    char filename[50];
    if ((*inm).ND==3){
        sprintf(filename,"./update_adjs3D.cl");
    }
    else if ((*inm).ND==2){
        sprintf(filename,"./update_adjs2D.cl");
    }
    else if ((*inm).ND==21){
        sprintf(filename,"./update_adjs2D_SH.cl");
    }
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, lcomm, comm, 1);
    
    
    
    /*Create the kernel, ther kernel version depends on the finite difference order*/
    const char * program_name = "update_adjs";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    /*Define the size of the local variables of the compute device*/
    if ((*inmloc).local_off==0){
        if ((*inm).ND==3){//For 3D
            shared_size = ((local_work_size[0]+(*inm).FDORDER)*(local_work_size[1]+(*inm).FDORDER)*(local_work_size[2]+(*inm).FDORDER)) * sizeof(float);
        }
        else if ((*inm).ND==2 || (*inm).ND==21){//For 2D
            shared_size = ((local_work_size[0]+(*inm).FDORDER)*(local_work_size[1]+(*inm).FDORDER)) * sizeof(float);
        }
    }
    
    /*Define the arguments for this kernel */
    cl_err = clSetKernelArg(*pkernel,  0, sizeof(int), &bndoff);
    if ((*inm).ND==3){//For 3D
        cl_err = clSetKernelArg(*pkernel,  3,  sizeof(cl_mem), &inmem->vx);
        cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem->vy);
        cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem->vz);
        cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem->sxx);
        cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem->syy);
        cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem->szz);
        cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem->sxy);
        cl_err = clSetKernelArg(*pkernel,  10,  sizeof(cl_mem), &inmem->syz);
        cl_err = clSetKernelArg(*pkernel,  11,  sizeof(cl_mem), &inmem->sxz);
        
        cl_err = clSetKernelArg(*pkernel,  12,  sizeof(cl_mem), &inmem->vxbnd);
        cl_err = clSetKernelArg(*pkernel,  13,  sizeof(cl_mem), &inmem->vybnd);
        cl_err = clSetKernelArg(*pkernel,  14,  sizeof(cl_mem), &inmem->vzbnd);
        cl_err = clSetKernelArg(*pkernel,  15,  sizeof(cl_mem), &inmem->sxxbnd);
        cl_err = clSetKernelArg(*pkernel,  16,  sizeof(cl_mem), &inmem->syybnd);
        cl_err = clSetKernelArg(*pkernel,  17,  sizeof(cl_mem), &inmem->szzbnd);
        cl_err = clSetKernelArg(*pkernel,  18,  sizeof(cl_mem), &inmem->sxybnd);
        cl_err = clSetKernelArg(*pkernel,  19,  sizeof(cl_mem), &inmem->syzbnd);
        cl_err = clSetKernelArg(*pkernel,  20,  sizeof(cl_mem), &inmem->sxzbnd);
        
        if (inm->back_prop_type==1){
            cl_err = clSetKernelArg(*pkernel,  21,  sizeof(cl_mem), &inmem->vx_r);
            cl_err = clSetKernelArg(*pkernel,  22,  sizeof(cl_mem), &inmem->vy_r);
            cl_err = clSetKernelArg(*pkernel,  23,  sizeof(cl_mem), &inmem->vz_r);
            cl_err = clSetKernelArg(*pkernel,  24,  sizeof(cl_mem), &inmem->sxx_r);
            cl_err = clSetKernelArg(*pkernel,  25,  sizeof(cl_mem), &inmem->syy_r);
            cl_err = clSetKernelArg(*pkernel,  26,  sizeof(cl_mem), &inmem->szz_r);
            cl_err = clSetKernelArg(*pkernel,  27,  sizeof(cl_mem), &inmem->sxy_r);
            cl_err = clSetKernelArg(*pkernel,  28,  sizeof(cl_mem), &inmem->syz_r);
            cl_err = clSetKernelArg(*pkernel,  29,  sizeof(cl_mem), &inmem->sxz_r);
        }
        else if (inm->back_prop_type==2){
            cl_err = clSetKernelArg(*pkernel,  21,  sizeof(cl_mem), &inmem->vx);
            cl_err = clSetKernelArg(*pkernel,  22,  sizeof(cl_mem), &inmem->vy);
            cl_err = clSetKernelArg(*pkernel,  23,  sizeof(cl_mem), &inmem->vz);
            cl_err = clSetKernelArg(*pkernel,  24,  sizeof(cl_mem), &inmem->sxx);
            cl_err = clSetKernelArg(*pkernel,  25,  sizeof(cl_mem), &inmem->syy);
            cl_err = clSetKernelArg(*pkernel,  26,  sizeof(cl_mem), &inmem->szz);
            cl_err = clSetKernelArg(*pkernel,  27,  sizeof(cl_mem), &inmem->sxy);
            cl_err = clSetKernelArg(*pkernel,  28,  sizeof(cl_mem), &inmem->syz);
            cl_err = clSetKernelArg(*pkernel,  29,  sizeof(cl_mem), &inmem->sxz);
        }
        
        cl_err = clSetKernelArg(*pkernel,  30,  sizeof(cl_mem), &inmem->rxx);
        cl_err = clSetKernelArg(*pkernel,  31,  sizeof(cl_mem), &inmem->ryy);
        cl_err = clSetKernelArg(*pkernel,  32,  sizeof(cl_mem), &inmem->rzz);
        cl_err = clSetKernelArg(*pkernel,  33,  sizeof(cl_mem), &inmem->rxy);
        cl_err = clSetKernelArg(*pkernel,  34,  sizeof(cl_mem), &inmem->ryz);
        cl_err = clSetKernelArg(*pkernel,  35,  sizeof(cl_mem), &inmem->rxz);
        
        if (inm->back_prop_type==1){
            cl_err = clSetKernelArg(*pkernel,  36,  sizeof(cl_mem), &inmem->rxx_r);
            cl_err = clSetKernelArg(*pkernel,  37,  sizeof(cl_mem), &inmem->ryy_r);
            cl_err = clSetKernelArg(*pkernel,  38,  sizeof(cl_mem), &inmem->rzz_r);
            cl_err = clSetKernelArg(*pkernel,  39,  sizeof(cl_mem), &inmem->rxy_r);
            cl_err = clSetKernelArg(*pkernel,  40,  sizeof(cl_mem), &inmem->ryz_r);
            cl_err = clSetKernelArg(*pkernel,  41,  sizeof(cl_mem), &inmem->rxz_r);
        }
        else if (inm->back_prop_type==2){
            cl_err = clSetKernelArg(*pkernel,  36,  sizeof(cl_mem), &inmem->rxx);
            cl_err = clSetKernelArg(*pkernel,  37,  sizeof(cl_mem), &inmem->ryy);
            cl_err = clSetKernelArg(*pkernel,  38,  sizeof(cl_mem), &inmem->rzz);
            cl_err = clSetKernelArg(*pkernel,  39,  sizeof(cl_mem), &inmem->rxy);
            cl_err = clSetKernelArg(*pkernel,  40,  sizeof(cl_mem), &inmem->ryz);
            cl_err = clSetKernelArg(*pkernel,  41,  sizeof(cl_mem), &inmem->rxz);
            
        }
        
        cl_err = clSetKernelArg(*pkernel,  42, sizeof(cl_mem), &inmem->pi);
        cl_err = clSetKernelArg(*pkernel,  43, sizeof(cl_mem), &inmem->u);
        cl_err = clSetKernelArg(*pkernel,  44, sizeof(cl_mem), &inmem->uipjp);
        cl_err = clSetKernelArg(*pkernel,  45, sizeof(cl_mem), &inmem->ujpkp);
        cl_err = clSetKernelArg(*pkernel,  46, sizeof(cl_mem), &inmem->uipkp);
        cl_err = clSetKernelArg(*pkernel,  47, sizeof(cl_mem), &inmem->taus);
        cl_err = clSetKernelArg(*pkernel,  48, sizeof(cl_mem), &inmem->tausipjp);
        cl_err = clSetKernelArg(*pkernel,  49, sizeof(cl_mem), &inmem->tausjpkp);
        cl_err = clSetKernelArg(*pkernel,  50, sizeof(cl_mem), &inmem->tausipkp);
        cl_err = clSetKernelArg(*pkernel,  51, sizeof(cl_mem), &inmem->taup);
        cl_err = clSetKernelArg(*pkernel,  52, sizeof(cl_mem), &inmem->eta);
        
        cl_err = clSetKernelArg(*pkernel,  53, sizeof(cl_mem), &inmem->src_pos);
        cl_err = clSetKernelArg(*pkernel,  54, sizeof(cl_mem), &inmem->src);
        cl_err = clSetKernelArg(*pkernel,  55, sizeof(cl_mem), &inmem->taper);
        
        cl_err = clSetKernelArg(*pkernel,  56, sizeof(cl_mem), &inmem->K_x);
        cl_err = clSetKernelArg(*pkernel,  57, sizeof(cl_mem), &inmem->a_x);
        cl_err = clSetKernelArg(*pkernel,  58, sizeof(cl_mem), &inmem->b_x);
        cl_err = clSetKernelArg(*pkernel,  59, sizeof(cl_mem), &inmem->K_x_half);
        cl_err = clSetKernelArg(*pkernel,  60, sizeof(cl_mem), &inmem->a_x_half);
        cl_err = clSetKernelArg(*pkernel,  61, sizeof(cl_mem), &inmem->b_x_half);
        
        cl_err = clSetKernelArg(*pkernel,  62, sizeof(cl_mem), &inmem->K_y);
        cl_err = clSetKernelArg(*pkernel,  63, sizeof(cl_mem), &inmem->a_y);
        cl_err = clSetKernelArg(*pkernel,  64, sizeof(cl_mem), &inmem->b_y);
        cl_err = clSetKernelArg(*pkernel,  65, sizeof(cl_mem), &inmem->K_y_half);
        cl_err = clSetKernelArg(*pkernel,  66, sizeof(cl_mem), &inmem->a_y_half);
        cl_err = clSetKernelArg(*pkernel,  67, sizeof(cl_mem), &inmem->b_y_half);
        
        cl_err = clSetKernelArg(*pkernel,  68, sizeof(cl_mem), &inmem->K_z);
        cl_err = clSetKernelArg(*pkernel,  69, sizeof(cl_mem), &inmem->a_z);
        cl_err = clSetKernelArg(*pkernel,  70, sizeof(cl_mem), &inmem->b_z);
        cl_err = clSetKernelArg(*pkernel,  71, sizeof(cl_mem), &inmem->K_z_half);
        cl_err = clSetKernelArg(*pkernel,  72, sizeof(cl_mem), &inmem->a_z_half);
        cl_err = clSetKernelArg(*pkernel,  73, sizeof(cl_mem), &inmem->b_z_half);
        
        cl_err = clSetKernelArg(*pkernel,  74, sizeof(cl_mem), &inmem->psi_vxx);
        cl_err = clSetKernelArg(*pkernel,  75, sizeof(cl_mem), &inmem->psi_vxy);
        cl_err = clSetKernelArg(*pkernel,  76, sizeof(cl_mem), &inmem->psi_vxz);
        cl_err = clSetKernelArg(*pkernel,  77, sizeof(cl_mem), &inmem->psi_vyx);
        cl_err = clSetKernelArg(*pkernel,  78, sizeof(cl_mem), &inmem->psi_vyy);
        cl_err = clSetKernelArg(*pkernel,  79, sizeof(cl_mem), &inmem->psi_vyz);
        cl_err = clSetKernelArg(*pkernel,  80, sizeof(cl_mem), &inmem->psi_vzx);
        cl_err = clSetKernelArg(*pkernel,  81, sizeof(cl_mem), &inmem->psi_vzy);
        cl_err = clSetKernelArg(*pkernel,  82, sizeof(cl_mem), &inmem->psi_vzz);
        
        cl_err = clSetKernelArg(*pkernel,  83, sizeof(cl_mem), &inmem->gradrho);
        cl_err = clSetKernelArg(*pkernel,  84, sizeof(cl_mem), &inmem->gradM);
        cl_err = clSetKernelArg(*pkernel,  85, sizeof(cl_mem), &inmem->gradmu);
        cl_err = clSetKernelArg(*pkernel,  86, sizeof(cl_mem), &inmem->gradtaup);
        cl_err = clSetKernelArg(*pkernel,  87, sizeof(cl_mem), &inmem->gradtaus);
        cl_err = clSetKernelArg(*pkernel,  88, sizeof(cl_mem), &inmem->gradsrc);
        
        cl_err = clSetKernelArg(*pkernel,  89, shared_size, NULL);
        
    }
    else if ((*inm).ND==2){//For 3D
        
        cl_err = clSetKernelArg(*pkernel,  3,  sizeof(cl_mem), &inmem->vx);
        cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem->vz);
        cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem->sxx);
        cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem->szz);
        cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem->sxz);
        
        cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem->vxbnd);
        cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem->vzbnd);
        cl_err = clSetKernelArg(*pkernel,  10,  sizeof(cl_mem), &inmem->sxxbnd);
        cl_err = clSetKernelArg(*pkernel,  11,  sizeof(cl_mem), &inmem->szzbnd);
        cl_err = clSetKernelArg(*pkernel,  12,  sizeof(cl_mem), &inmem->sxzbnd);
        
        if (inm->back_prop_type==1){
            cl_err = clSetKernelArg(*pkernel,  13,  sizeof(cl_mem), &inmem->vx_r);
            cl_err = clSetKernelArg(*pkernel,  14,  sizeof(cl_mem), &inmem->vz_r);
            cl_err = clSetKernelArg(*pkernel,  15,  sizeof(cl_mem), &inmem->sxx_r);
            cl_err = clSetKernelArg(*pkernel,  16,  sizeof(cl_mem), &inmem->szz_r);
            cl_err = clSetKernelArg(*pkernel,  17,  sizeof(cl_mem), &inmem->sxz_r);
        }
        else if (inm->back_prop_type==2){
            cl_err = clSetKernelArg(*pkernel,  13,  sizeof(cl_mem), &inmem->vx);
            cl_err = clSetKernelArg(*pkernel,  14,  sizeof(cl_mem), &inmem->vz);
            cl_err = clSetKernelArg(*pkernel,  15,  sizeof(cl_mem), &inmem->sxx);
            cl_err = clSetKernelArg(*pkernel,  16,  sizeof(cl_mem), &inmem->szz);
            cl_err = clSetKernelArg(*pkernel,  17,  sizeof(cl_mem), &inmem->sxz);
            
        }
        cl_err = clSetKernelArg(*pkernel,  18,  sizeof(cl_mem), &inmem->rxx);
        cl_err = clSetKernelArg(*pkernel,  19,  sizeof(cl_mem), &inmem->rzz);
        cl_err = clSetKernelArg(*pkernel,  20,  sizeof(cl_mem), &inmem->rxz);
        
        if (inm->back_prop_type==1){
            cl_err = clSetKernelArg(*pkernel,  21,  sizeof(cl_mem), &inmem->rxx_r);
            cl_err = clSetKernelArg(*pkernel,  22,  sizeof(cl_mem), &inmem->rzz_r);
            cl_err = clSetKernelArg(*pkernel,  23,  sizeof(cl_mem), &inmem->rxz_r);
        }
        else if (inm->back_prop_type==2){
            cl_err = clSetKernelArg(*pkernel,  21,  sizeof(cl_mem), &inmem->rxx);
            cl_err = clSetKernelArg(*pkernel,  22,  sizeof(cl_mem), &inmem->rzz);
            cl_err = clSetKernelArg(*pkernel,  23,  sizeof(cl_mem), &inmem->rxz);
        }
        
        cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem->pi);
        cl_err = clSetKernelArg(*pkernel,  25, sizeof(cl_mem), &inmem->u);
        cl_err = clSetKernelArg(*pkernel,  26, sizeof(cl_mem), &inmem->uipkp);
        cl_err = clSetKernelArg(*pkernel,  27, sizeof(cl_mem), &inmem->taus);
        cl_err = clSetKernelArg(*pkernel,  28, sizeof(cl_mem), &inmem->tausipkp);
        cl_err = clSetKernelArg(*pkernel,  29, sizeof(cl_mem), &inmem->taup);
        cl_err = clSetKernelArg(*pkernel,  30, sizeof(cl_mem), &inmem->eta);
        
        
        cl_err = clSetKernelArg(*pkernel,  31, sizeof(cl_mem), &inmem->src_pos);
        cl_err = clSetKernelArg(*pkernel,  32, sizeof(cl_mem), &inmem->src);
        cl_err = clSetKernelArg(*pkernel,  33, sizeof(cl_mem), &inmem->taper);
        
        
        cl_err = clSetKernelArg(*pkernel,  34, sizeof(cl_mem), &inmem->K_x);
        cl_err = clSetKernelArg(*pkernel,  35, sizeof(cl_mem), &inmem->a_x);
        cl_err = clSetKernelArg(*pkernel,  36, sizeof(cl_mem), &inmem->b_x);
        cl_err = clSetKernelArg(*pkernel,  37, sizeof(cl_mem), &inmem->K_x_half);
        cl_err = clSetKernelArg(*pkernel,  38, sizeof(cl_mem), &inmem->a_x_half);
        cl_err = clSetKernelArg(*pkernel,  39, sizeof(cl_mem), &inmem->b_x_half);

        
        cl_err = clSetKernelArg(*pkernel,  40, sizeof(cl_mem), &inmem->K_z);
        cl_err = clSetKernelArg(*pkernel,  41, sizeof(cl_mem), &inmem->a_z);
        cl_err = clSetKernelArg(*pkernel,  42, sizeof(cl_mem), &inmem->b_z);
        cl_err = clSetKernelArg(*pkernel,  43, sizeof(cl_mem), &inmem->K_z_half);
        cl_err = clSetKernelArg(*pkernel,  44, sizeof(cl_mem), &inmem->a_z_half);
        cl_err = clSetKernelArg(*pkernel,  45, sizeof(cl_mem), &inmem->b_z_half);
        
        cl_err = clSetKernelArg(*pkernel,  46, sizeof(cl_mem), &inmem->psi_vxx);
        cl_err = clSetKernelArg(*pkernel,  47, sizeof(cl_mem), &inmem->psi_vxz);
        cl_err = clSetKernelArg(*pkernel,  48, sizeof(cl_mem), &inmem->psi_vzx);
        cl_err = clSetKernelArg(*pkernel,  49, sizeof(cl_mem), &inmem->psi_vzz);
        
        cl_err = clSetKernelArg(*pkernel,  50, sizeof(cl_mem), &inmem->gradrho);
        cl_err = clSetKernelArg(*pkernel,  51, sizeof(cl_mem), &inmem->gradM);
        cl_err = clSetKernelArg(*pkernel,  52, sizeof(cl_mem), &inmem->gradmu);
        cl_err = clSetKernelArg(*pkernel,  53, sizeof(cl_mem), &inmem->gradtaup);
        cl_err = clSetKernelArg(*pkernel,  54, sizeof(cl_mem), &inmem->gradtaus);
        cl_err = clSetKernelArg(*pkernel,  55, sizeof(cl_mem), &inmem->gradsrc);
        
        cl_err = clSetKernelArg(*pkernel,  56, shared_size, NULL);
        
    }
    else if ((*inm).ND==21){//For 2D SH
        
        cl_err = clSetKernelArg(*pkernel,  3,  sizeof(cl_mem), &inmem->vy);
        cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem->sxy);
        cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem->syz);
        cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem->vybnd);
        cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem->sxybnd);
        cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem->syzbnd);
        if (inm->back_prop_type==1){
            cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem->vy_r);
            cl_err = clSetKernelArg(*pkernel,  10,  sizeof(cl_mem), &inmem->sxy_r);
            cl_err = clSetKernelArg(*pkernel,  11,  sizeof(cl_mem), &inmem->syz_r);
        }
        else if (inm->back_prop_type==2){
            cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem->vy);
            cl_err = clSetKernelArg(*pkernel,  10,  sizeof(cl_mem), &inmem->sxy);
            cl_err = clSetKernelArg(*pkernel,  11,  sizeof(cl_mem), &inmem->syz);
        }

        cl_err = clSetKernelArg(*pkernel,  12,  sizeof(cl_mem), &inmem->rxy);
        cl_err = clSetKernelArg(*pkernel,  13,  sizeof(cl_mem), &inmem->ryz);
        
        if (inm->back_prop_type==1){
            cl_err = clSetKernelArg(*pkernel,  14,  sizeof(cl_mem), &inmem->rxy_r);
            cl_err = clSetKernelArg(*pkernel,  15,  sizeof(cl_mem), &inmem->ryz_r);
        }
        else if (inm->back_prop_type==2){
            cl_err = clSetKernelArg(*pkernel,  14,  sizeof(cl_mem), &inmem->rxy);
            cl_err = clSetKernelArg(*pkernel,  15,  sizeof(cl_mem), &inmem->ryz);
        }
 
        
        cl_err = clSetKernelArg(*pkernel,  16, sizeof(cl_mem), &inmem->uipjp);
        cl_err = clSetKernelArg(*pkernel,  17, sizeof(cl_mem), &inmem->ujpkp);
        cl_err = clSetKernelArg(*pkernel,  18, sizeof(cl_mem), &inmem->tausipjp);
        cl_err = clSetKernelArg(*pkernel,  19, sizeof(cl_mem), &inmem->tausjpkp);
        cl_err = clSetKernelArg(*pkernel,  20, sizeof(cl_mem), &inmem->eta);
        
        
        cl_err = clSetKernelArg(*pkernel,  21, sizeof(cl_mem), &inmem->src_pos);
        cl_err = clSetKernelArg(*pkernel,  22, sizeof(cl_mem), &inmem->src);
        cl_err = clSetKernelArg(*pkernel,  23, sizeof(cl_mem), &inmem->taper);
        
        
        cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem->K_x);
        cl_err = clSetKernelArg(*pkernel,  25, sizeof(cl_mem), &inmem->a_x);
        cl_err = clSetKernelArg(*pkernel,  26, sizeof(cl_mem), &inmem->b_x);
        cl_err = clSetKernelArg(*pkernel,  27, sizeof(cl_mem), &inmem->K_x_half);
        cl_err = clSetKernelArg(*pkernel,  28, sizeof(cl_mem), &inmem->a_x_half);
        cl_err = clSetKernelArg(*pkernel,  29, sizeof(cl_mem), &inmem->b_x_half);
        
        
        cl_err = clSetKernelArg(*pkernel,  30, sizeof(cl_mem), &inmem->K_z);
        cl_err = clSetKernelArg(*pkernel,  31, sizeof(cl_mem), &inmem->a_z);
        cl_err = clSetKernelArg(*pkernel,  32, sizeof(cl_mem), &inmem->b_z);
        cl_err = clSetKernelArg(*pkernel,  33, sizeof(cl_mem), &inmem->K_z_half);
        cl_err = clSetKernelArg(*pkernel,  34, sizeof(cl_mem), &inmem->a_z_half);
        cl_err = clSetKernelArg(*pkernel,  35, sizeof(cl_mem), &inmem->b_z_half);
        
        cl_err = clSetKernelArg(*pkernel,  36, sizeof(cl_mem), &inmem->psi_vyx);
        cl_err = clSetKernelArg(*pkernel,  37, sizeof(cl_mem), &inmem->psi_vyz);

        
        cl_err = clSetKernelArg(*pkernel,  38, sizeof(cl_mem), &inmem->gradrho);
        cl_err = clSetKernelArg(*pkernel,  39, sizeof(cl_mem), &inmem->gradmu);
        cl_err = clSetKernelArg(*pkernel,  40, sizeof(cl_mem), &inmem->gradsrc);
        
        cl_err = clSetKernelArg(*pkernel,  41, shared_size, NULL);
        
    }
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
	return cl_err;
}


int gpu_initialize_savebnd(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc )


{
	
	cl_int cl_err = 0;
    
    
    char filename[20];
    if ((*inm).ND==3){
        sprintf(filename,"./savebnd3D.cl");
    }
    else if ((*inm).ND==2){
        sprintf(filename,"./savebnd2D.cl");
    }
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, 0, 0, 0);
    
    const char * program_name = "savebnd";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    
    /*Define the arguments for this kernel */
    cl_err = clSetKernelArg(*pkernel,  0,  sizeof(cl_mem), &inmem->vx);
    cl_err = clSetKernelArg(*pkernel,  1,  sizeof(cl_mem), &inmem->vy);
    cl_err = clSetKernelArg(*pkernel,  2,  sizeof(cl_mem), &inmem->vz);
    cl_err = clSetKernelArg(*pkernel,  3,  sizeof(cl_mem), &inmem->sxx);
    cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem->syy);
    cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem->szz);
    cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem->sxy);
    cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem->syz);
    cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem->sxz);
    
    cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem->vxbnd);
    cl_err = clSetKernelArg(*pkernel,  10,  sizeof(cl_mem), &inmem->vybnd);
    cl_err = clSetKernelArg(*pkernel,  11,  sizeof(cl_mem), &inmem->vzbnd);
    cl_err = clSetKernelArg(*pkernel,  12,  sizeof(cl_mem), &inmem->sxxbnd);
    cl_err = clSetKernelArg(*pkernel,  13,  sizeof(cl_mem), &inmem->syybnd);
    cl_err = clSetKernelArg(*pkernel,  14,  sizeof(cl_mem), &inmem->szzbnd);
    cl_err = clSetKernelArg(*pkernel,  15,  sizeof(cl_mem), &inmem->sxybnd);
    cl_err = clSetKernelArg(*pkernel,  16,  sizeof(cl_mem), &inmem->syzbnd);
    cl_err = clSetKernelArg(*pkernel,  17,  sizeof(cl_mem), &inmem->sxzbnd);
        

    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
	return cl_err;
}
int gpu_initialize_savefreqs(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc, int dirprop )


{
    
    cl_int cl_err = 0;
    
    
    char filename[20];
    sprintf(filename,"./savefreqs.cl");
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, 0, 0, dirprop);
    
    const char * program_name = "savefreqs";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    
    /*Define the arguments for this kernel */

        cl_err = clSetKernelArg(*pkernel,  0,  sizeof(cl_mem), &inmem->vx);
        cl_err = clSetKernelArg(*pkernel,  1,  sizeof(cl_mem), &inmem->vy);
        cl_err = clSetKernelArg(*pkernel,  2,  sizeof(cl_mem), &inmem->vz);
        cl_err = clSetKernelArg(*pkernel,  3,  sizeof(cl_mem), &inmem->sxx);
        cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem->syy);
        cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem->szz);
        cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem->sxy);
        cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem->syz);
        cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem->sxz);
        
        cl_err = clSetKernelArg(*pkernel,  9,   sizeof(cl_mem), &inmem->f_vx);
        cl_err = clSetKernelArg(*pkernel,  10,  sizeof(cl_mem), &inmem->f_vy);
        cl_err = clSetKernelArg(*pkernel,  11,  sizeof(cl_mem), &inmem->f_vz);
        cl_err = clSetKernelArg(*pkernel,  12,  sizeof(cl_mem), &inmem->f_sxx);
        cl_err = clSetKernelArg(*pkernel,  13,  sizeof(cl_mem), &inmem->f_syy);
        cl_err = clSetKernelArg(*pkernel,  14,  sizeof(cl_mem), &inmem->f_szz);
        cl_err = clSetKernelArg(*pkernel,  15,  sizeof(cl_mem), &inmem->f_sxy);
        cl_err = clSetKernelArg(*pkernel,  16,  sizeof(cl_mem), &inmem->f_syz);
        cl_err = clSetKernelArg(*pkernel,  17,  sizeof(cl_mem), &inmem->f_sxz);
        
        cl_err = clSetKernelArg(*pkernel,  18,  sizeof(cl_mem), &inmem->rxx);
        cl_err = clSetKernelArg(*pkernel,  19,  sizeof(cl_mem), &inmem->ryy);
        cl_err = clSetKernelArg(*pkernel,  20,  sizeof(cl_mem), &inmem->rzz);
        cl_err = clSetKernelArg(*pkernel,  21,  sizeof(cl_mem), &inmem->rxy);
        cl_err = clSetKernelArg(*pkernel,  22,  sizeof(cl_mem), &inmem->ryz);
        cl_err = clSetKernelArg(*pkernel,  23,  sizeof(cl_mem), &inmem->rxz);
        
        cl_err = clSetKernelArg(*pkernel,  24,  sizeof(cl_mem), &inmem->f_rxx);
        cl_err = clSetKernelArg(*pkernel,  25,  sizeof(cl_mem), &inmem->f_ryy);
        cl_err = clSetKernelArg(*pkernel,  26,  sizeof(cl_mem), &inmem->f_rzz);
        cl_err = clSetKernelArg(*pkernel,  27,  sizeof(cl_mem), &inmem->f_rxy);
        cl_err = clSetKernelArg(*pkernel,  28,  sizeof(cl_mem), &inmem->f_ryz);
        cl_err = clSetKernelArg(*pkernel,  29,  sizeof(cl_mem), &inmem->f_rxz);
    
        cl_err = clSetKernelArg(*pkernel,  30,  sizeof(cl_mem), &inmem->gradfreqsn);
    
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
    return cl_err;
}

int gpu_initialize_initsavefreqs(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc )


{
    
    cl_int cl_err = 0;
    
    
    char filename[20];
    sprintf(filename,"./initialize.cl");
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, 0, 0, 0);
    
    const char * program_name = "initialize_savefreqs";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    
    /*Define the arguments for this kernel */
    
    
    cl_err = clSetKernelArg(*pkernel,  0,  sizeof(cl_mem), &inmem->f_vx);
    cl_err = clSetKernelArg(*pkernel,  1,  sizeof(cl_mem), &inmem->f_vy);
    cl_err = clSetKernelArg(*pkernel,  2,  sizeof(cl_mem), &inmem->f_vz);
    cl_err = clSetKernelArg(*pkernel,  3,  sizeof(cl_mem), &inmem->f_sxx);
    cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem->f_syy);
    cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem->f_szz);
    cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem->f_sxy);
    cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem->f_syz);
    cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem->f_sxz);
    
    cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem->f_rxx);
    cl_err = clSetKernelArg(*pkernel,  10,  sizeof(cl_mem), &inmem->f_ryy);
    cl_err = clSetKernelArg(*pkernel,  11,  sizeof(cl_mem), &inmem->f_rzz);
    cl_err = clSetKernelArg(*pkernel,  12,  sizeof(cl_mem), &inmem->f_rxy);
    cl_err = clSetKernelArg(*pkernel,  13,  sizeof(cl_mem), &inmem->f_ryz);
    cl_err = clSetKernelArg(*pkernel,  14,  sizeof(cl_mem), &inmem->f_rxz);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
    return cl_err;
}

int gpu_initialize_gradsrc(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, size_t *local_work_size, struct varcl *inmem, struct modcsts *inm, struct modcstsloc *inmloc)
{
    
    cl_int cl_err = 0;
    
    
    char filename[20];
    sprintf(filename,"./initialize.cl");
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options(inmem, inm, inmloc, 0, 0, 0);
    
    const char * program_name = "initialize_gradsrc";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    
    /*Define the arguments for this kernel */
    cl_err = clSetKernelArg(*pkernel,  0,  sizeof(cl_mem), &inmem->gradsrc);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
    return cl_err;
}


int gpu_intialize_surfgrid_fine2coarse(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, struct varcl *inmem_c, struct modcsts *inm_c, struct modcstsloc *inmloc_c ,struct varcl *inmem_f, struct modcsts *inm_f, struct modcstsloc *inmloc_f ){
    
    
    cl_int cl_err = 0;
    
    
    
    const char * filename = "./surfgrid_fine2coarse.cl";
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options_surf(inmem_c, inm_c, inmloc_c, inmem_f, inm_f, inmloc_f);
    
    /*Create the kernel*/
    const char * program_name = "surfgrid_fine2coarse";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    
    /*Define the arguments for this kernel */
    cl_err = clSetKernelArg(*pkernel,  0,  sizeof(cl_mem), &inmem_c->vx);
    cl_err = clSetKernelArg(*pkernel,  1,  sizeof(cl_mem), &inmem_c->vy);
    cl_err = clSetKernelArg(*pkernel,  2,  sizeof(cl_mem), &inmem_c->vz);
    cl_err = clSetKernelArg(*pkernel,  3,  sizeof(cl_mem), &inmem_c->sxx);
    cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem_c->syy);
    cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem_c->szz);
    cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem_c->sxy);
    cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem_c->syz);
    cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem_c->sxz);
    cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem_c->rxx);
    cl_err = clSetKernelArg(*pkernel,  10, sizeof(cl_mem), &inmem_c->ryy);
    cl_err = clSetKernelArg(*pkernel,  11, sizeof(cl_mem), &inmem_c->rzz);
    cl_err = clSetKernelArg(*pkernel,  12, sizeof(cl_mem), &inmem_c->rxy);
    cl_err = clSetKernelArg(*pkernel,  13, sizeof(cl_mem), &inmem_c->ryz);
    cl_err = clSetKernelArg(*pkernel,  14, sizeof(cl_mem), &inmem_c->rxz);
    cl_err = clSetKernelArg(*pkernel,  15, sizeof(cl_mem), &inmem_c->psi_vxx);
    cl_err = clSetKernelArg(*pkernel,  16, sizeof(cl_mem), &inmem_c->psi_vyx);
    cl_err = clSetKernelArg(*pkernel,  17, sizeof(cl_mem), &inmem_c->psi_vzx);
    cl_err = clSetKernelArg(*pkernel,  18, sizeof(cl_mem), &inmem_c->psi_vxy);
    cl_err = clSetKernelArg(*pkernel,  19, sizeof(cl_mem), &inmem_c->psi_vyy);
    cl_err = clSetKernelArg(*pkernel,  20, sizeof(cl_mem), &inmem_c->psi_vzy);
    cl_err = clSetKernelArg(*pkernel,  21, sizeof(cl_mem), &inmem_c->psi_sxx_x);
    cl_err = clSetKernelArg(*pkernel,  22, sizeof(cl_mem), &inmem_c->psi_sxy_x);
    cl_err = clSetKernelArg(*pkernel,  23, sizeof(cl_mem), &inmem_c->psi_sxz_x);
    cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem_c->psi_sxy_y);
    cl_err = clSetKernelArg(*pkernel,  25, sizeof(cl_mem), &inmem_c->psi_syy_y);
    cl_err = clSetKernelArg(*pkernel,  26, sizeof(cl_mem), &inmem_c->psi_syz_y);
    
    cl_err = clSetKernelArg(*pkernel,  27,  sizeof(cl_mem), &inmem_f->vx);
    cl_err = clSetKernelArg(*pkernel,  28,  sizeof(cl_mem), &inmem_f->vy);
    cl_err = clSetKernelArg(*pkernel,  29,  sizeof(cl_mem), &inmem_f->vz);
    cl_err = clSetKernelArg(*pkernel,  30,  sizeof(cl_mem), &inmem_f->sxx);
    cl_err = clSetKernelArg(*pkernel,  31,  sizeof(cl_mem), &inmem_f->syy);
    cl_err = clSetKernelArg(*pkernel,  32,  sizeof(cl_mem), &inmem_f->szz);
    cl_err = clSetKernelArg(*pkernel,  33,  sizeof(cl_mem), &inmem_f->sxy);
    cl_err = clSetKernelArg(*pkernel,  34,  sizeof(cl_mem), &inmem_f->syz);
    cl_err = clSetKernelArg(*pkernel,  35,  sizeof(cl_mem), &inmem_f->sxz);
    cl_err = clSetKernelArg(*pkernel,  36,  sizeof(cl_mem), &inmem_f->rxx);
    cl_err = clSetKernelArg(*pkernel,  37, sizeof(cl_mem), &inmem_f->ryy);
    cl_err = clSetKernelArg(*pkernel,  38, sizeof(cl_mem), &inmem_f->rzz);
    cl_err = clSetKernelArg(*pkernel,  39, sizeof(cl_mem), &inmem_f->rxy);
    cl_err = clSetKernelArg(*pkernel,  40, sizeof(cl_mem), &inmem_f->ryz);
    cl_err = clSetKernelArg(*pkernel,  41, sizeof(cl_mem), &inmem_f->rxz);
    cl_err = clSetKernelArg(*pkernel,  42, sizeof(cl_mem), &inmem_f->psi_vxx);
    cl_err = clSetKernelArg(*pkernel,  43, sizeof(cl_mem), &inmem_f->psi_vyx);
    cl_err = clSetKernelArg(*pkernel,  44, sizeof(cl_mem), &inmem_f->psi_vzx);
    cl_err = clSetKernelArg(*pkernel,  45, sizeof(cl_mem), &inmem_f->psi_vxy);
    cl_err = clSetKernelArg(*pkernel,  46, sizeof(cl_mem), &inmem_f->psi_vyy);
    cl_err = clSetKernelArg(*pkernel,  47, sizeof(cl_mem), &inmem_f->psi_vzy);
    cl_err = clSetKernelArg(*pkernel,  48, sizeof(cl_mem), &inmem_f->psi_sxx_x);
    cl_err = clSetKernelArg(*pkernel,  49, sizeof(cl_mem), &inmem_f->psi_sxy_x);
    cl_err = clSetKernelArg(*pkernel,  50, sizeof(cl_mem), &inmem_f->psi_sxz_x);
    cl_err = clSetKernelArg(*pkernel,  51, sizeof(cl_mem), &inmem_f->psi_sxy_y);
    cl_err = clSetKernelArg(*pkernel,  52, sizeof(cl_mem), &inmem_f->psi_syy_y);
    cl_err = clSetKernelArg(*pkernel,  53, sizeof(cl_mem), &inmem_f->psi_syz_y);


    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
    return cl_err;
    
}

int gpu_intialize_surfgrid_coarse2fine(cl_context  * pcontext, cl_program  * program, cl_kernel * pkernel, struct varcl *inmem_c, struct modcsts *inm_c, struct modcstsloc *inmloc_c ,struct varcl *inmem_f, struct modcsts *inm_f, struct modcstsloc *inmloc_f, size_t *local_work_size){
    
    
    cl_int cl_err = 0;
    size_t shared_size=sizeof(float);
    
    
    
    const char * filename = "./surfgrid_coarse2fine_cubic.cl";
    
    /* Pass some constant arguments as build options */
    const char * build_options=get_build_options_surf(inmem_c, inm_c, inmloc_c, inmem_f, inm_f, inmloc_f);
    
    /*Create the kernel*/
    const char * program_name = "surfgrid_coarse2fine";
    cl_err = create_gpu_kernel( filename, program, pcontext, pkernel, program_name, build_options);
    
    /*Define the size of the local variables of the compute device*/
    if ((*inmloc_c).local_off==0){
        if ((*inm_c).ND==3){//For 3D
            shared_size = ((4+3)*(local_work_size[1]+3)*(local_work_size[2]+3)) * sizeof(float);
        }
        else if ((*inm_c).ND==2 || (*inm_c).ND==21){//For 2D
            shared_size = ((4+3)*(local_work_size[1]+3)) * sizeof(float);
        }
    }
    
    /*Define the arguments for this kernel */
    cl_err = clSetKernelArg(*pkernel,  0,  sizeof(cl_mem), &inmem_c->vx);
    cl_err = clSetKernelArg(*pkernel,  1,  sizeof(cl_mem), &inmem_c->vy);
    cl_err = clSetKernelArg(*pkernel,  2,  sizeof(cl_mem), &inmem_c->vz);
    cl_err = clSetKernelArg(*pkernel,  3,  sizeof(cl_mem), &inmem_c->sxx);
    cl_err = clSetKernelArg(*pkernel,  4,  sizeof(cl_mem), &inmem_c->syy);
    cl_err = clSetKernelArg(*pkernel,  5,  sizeof(cl_mem), &inmem_c->szz);
    cl_err = clSetKernelArg(*pkernel,  6,  sizeof(cl_mem), &inmem_c->sxy);
    cl_err = clSetKernelArg(*pkernel,  7,  sizeof(cl_mem), &inmem_c->syz);
    cl_err = clSetKernelArg(*pkernel,  8,  sizeof(cl_mem), &inmem_c->sxz);
    cl_err = clSetKernelArg(*pkernel,  9,  sizeof(cl_mem), &inmem_c->rxx);
    cl_err = clSetKernelArg(*pkernel,  10, sizeof(cl_mem), &inmem_c->ryy);
    cl_err = clSetKernelArg(*pkernel,  11, sizeof(cl_mem), &inmem_c->rzz);
    cl_err = clSetKernelArg(*pkernel,  12, sizeof(cl_mem), &inmem_c->rxy);
    cl_err = clSetKernelArg(*pkernel,  13, sizeof(cl_mem), &inmem_c->ryz);
    cl_err = clSetKernelArg(*pkernel,  14, sizeof(cl_mem), &inmem_c->rxz);
    cl_err = clSetKernelArg(*pkernel,  15, sizeof(cl_mem), &inmem_c->psi_vxx);
    cl_err = clSetKernelArg(*pkernel,  16, sizeof(cl_mem), &inmem_c->psi_vyx);
    cl_err = clSetKernelArg(*pkernel,  17, sizeof(cl_mem), &inmem_c->psi_vzx);
    cl_err = clSetKernelArg(*pkernel,  18, sizeof(cl_mem), &inmem_c->psi_vxy);
    cl_err = clSetKernelArg(*pkernel,  19, sizeof(cl_mem), &inmem_c->psi_vyy);
    cl_err = clSetKernelArg(*pkernel,  20, sizeof(cl_mem), &inmem_c->psi_vzy);
    cl_err = clSetKernelArg(*pkernel,  21, sizeof(cl_mem), &inmem_c->psi_sxx_x);
    cl_err = clSetKernelArg(*pkernel,  22, sizeof(cl_mem), &inmem_c->psi_sxy_x);
    cl_err = clSetKernelArg(*pkernel,  23, sizeof(cl_mem), &inmem_c->psi_sxz_x);
    cl_err = clSetKernelArg(*pkernel,  24, sizeof(cl_mem), &inmem_c->psi_sxy_y);
    cl_err = clSetKernelArg(*pkernel,  25, sizeof(cl_mem), &inmem_c->psi_syy_y);
    cl_err = clSetKernelArg(*pkernel,  26, sizeof(cl_mem), &inmem_c->psi_syz_y);
    
    cl_err = clSetKernelArg(*pkernel,  27,  sizeof(cl_mem), &inmem_f->vx);
    cl_err = clSetKernelArg(*pkernel,  28,  sizeof(cl_mem), &inmem_f->vy);
    cl_err = clSetKernelArg(*pkernel,  29,  sizeof(cl_mem), &inmem_f->vz);
    cl_err = clSetKernelArg(*pkernel,  30,  sizeof(cl_mem), &inmem_f->sxx);
    cl_err = clSetKernelArg(*pkernel,  31,  sizeof(cl_mem), &inmem_f->syy);
    cl_err = clSetKernelArg(*pkernel,  32,  sizeof(cl_mem), &inmem_f->szz);
    cl_err = clSetKernelArg(*pkernel,  33,  sizeof(cl_mem), &inmem_f->sxy);
    cl_err = clSetKernelArg(*pkernel,  34,  sizeof(cl_mem), &inmem_f->syz);
    cl_err = clSetKernelArg(*pkernel,  35,  sizeof(cl_mem), &inmem_f->sxz);
    cl_err = clSetKernelArg(*pkernel,  36,  sizeof(cl_mem), &inmem_f->rxx);
    cl_err = clSetKernelArg(*pkernel,  37, sizeof(cl_mem), &inmem_f->ryy);
    cl_err = clSetKernelArg(*pkernel,  38, sizeof(cl_mem), &inmem_f->rzz);
    cl_err = clSetKernelArg(*pkernel,  39, sizeof(cl_mem), &inmem_f->rxy);
    cl_err = clSetKernelArg(*pkernel,  40, sizeof(cl_mem), &inmem_f->ryz);
    cl_err = clSetKernelArg(*pkernel,  41, sizeof(cl_mem), &inmem_f->rxz);
    cl_err = clSetKernelArg(*pkernel,  42, sizeof(cl_mem), &inmem_f->psi_vxx);
    cl_err = clSetKernelArg(*pkernel,  43, sizeof(cl_mem), &inmem_f->psi_vyx);
    cl_err = clSetKernelArg(*pkernel,  44, sizeof(cl_mem), &inmem_f->psi_vzx);
    cl_err = clSetKernelArg(*pkernel,  45, sizeof(cl_mem), &inmem_f->psi_vxy);
    cl_err = clSetKernelArg(*pkernel,  46, sizeof(cl_mem), &inmem_f->psi_vyy);
    cl_err = clSetKernelArg(*pkernel,  47, sizeof(cl_mem), &inmem_f->psi_vzy);
    cl_err = clSetKernelArg(*pkernel,  48, sizeof(cl_mem), &inmem_f->psi_sxx_x);
    cl_err = clSetKernelArg(*pkernel,  49, sizeof(cl_mem), &inmem_f->psi_sxy_x);
    cl_err = clSetKernelArg(*pkernel,  50, sizeof(cl_mem), &inmem_f->psi_sxz_x);
    cl_err = clSetKernelArg(*pkernel,  51, sizeof(cl_mem), &inmem_f->psi_sxy_y);
    cl_err = clSetKernelArg(*pkernel,  52, sizeof(cl_mem), &inmem_f->psi_syy_y);
    cl_err = clSetKernelArg(*pkernel,  53, sizeof(cl_mem), &inmem_f->psi_syz_y);

    cl_err = clSetKernelArg(*pkernel,  54, shared_size, NULL);
    cl_err = clSetKernelArg(*pkernel,  55, shared_size, NULL);
    cl_err = clSetKernelArg(*pkernel,  56, shared_size, NULL);
    
    if (cl_err !=CL_SUCCESS) fprintf(stderr,"%s",gpu_error_code(cl_err));
    
    return cl_err;
    
}