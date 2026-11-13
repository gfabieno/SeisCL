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

/* Exact adjoint of the 3D CPML (ABS_TYPE 1, BACK_PROP_TYPE 1): the 3D twin of
 * cpml_adj2D.cl (see there for the derivation). Each adjoint field is filtered,
 * before its spatial derivative, with the coefficients of the forward term it
 * feeds; the memories are the adjoint copies psi_*r of the forward memories.
 * cpml_adj_v (before update_adjs): cpf1..9 = F[sxx_x](vxr), F[sxy_y](vxr),
 *   F[sxz_z](vxr), F[sxy_x](vyr), F[syy_y](vyr), F[syz_z](vyr), F[sxz_x](vzr),
 *   F[syz_y](vzr), F[szz_z](vzr).
 * cpml_adj_s (before update_adjv): cpf1..9 = F[vx_x](sxxr), F[vx_y](sxyr),
 *   F[vx_z](sxzr), F[vy_x](sxyr), F[vy_y](syyr), F[vy_z](syzr), F[vz_x](sxzr),
 *   F[vz_y](syzr), F[vz_z](szzr).
 * Single device only (Init_OpenCL.c). */

#define psi_sxz_zr(z,y,x) psi_sxz_zr[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_syz_zr(z,y,x) psi_syz_zr[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_szz_zr(z,y,x) psi_szz_zr[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_sxy_yr(z,y,x) psi_sxy_yr[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_syy_yr(z,y,x) psi_syy_yr[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_syz_yr(z,y,x) psi_syz_yr[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_sxx_xr(z,y,x) psi_sxx_xr[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_sxy_xr(z,y,x) psi_sxy_xr[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_sxz_xr(z,y,x) psi_sxz_xr[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vx_zr(z,y,x) psi_vx_zr[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_vy_zr(z,y,x) psi_vy_zr[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_vz_zr(z,y,x) psi_vz_zr[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_vx_yr(z,y,x) psi_vx_yr[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vy_yr(z,y,x) psi_vy_yr[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vz_yr(z,y,x) psi_vz_yr[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vx_xr(z,y,x) psi_vx_xr[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vy_xr(z,y,x) psi_vy_xr[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vz_xr(z,y,x) psi_vz_xr[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]

FUNDEF void cpml_adj_v(GLOBARG const float * RESTRICT vxr, GLOBARG const float * RESTRICT vyr, GLOBARG const float * RESTRICT vzr,
                       GLOBARG const float * RESTRICT K_x,  GLOBARG const float * RESTRICT a_x,  GLOBARG const float * RESTRICT b_x,
                       GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                       GLOBARG const float * RESTRICT K_y,  GLOBARG const float * RESTRICT a_y,  GLOBARG const float * RESTRICT b_y,
                       GLOBARG const float * RESTRICT K_y_half,  GLOBARG const float * RESTRICT a_y_half,  GLOBARG const float * RESTRICT b_y_half,
                       GLOBARG const float * RESTRICT K_z,  GLOBARG const float * RESTRICT a_z,  GLOBARG const float * RESTRICT b_z,
                       GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                       GLOBARG float * RESTRICT psi_sxz_zr,
                       GLOBARG float * RESTRICT psi_syz_zr,
                       GLOBARG float * RESTRICT psi_szz_zr,
                       GLOBARG float * RESTRICT psi_sxy_yr,
                       GLOBARG float * RESTRICT psi_syy_yr,
                       GLOBARG float * RESTRICT psi_syz_yr,
                       GLOBARG float * RESTRICT psi_sxx_xr,
                       GLOBARG float * RESTRICT psi_sxy_xr,
                       GLOBARG float * RESTRICT psi_sxz_xr,
                       GLOBARG float * RESTRICT cpf1,
                       GLOBARG float * RESTRICT cpf2,
                       GLOBARG float * RESTRICT cpf3,
                       GLOBARG float * RESTRICT cpf4,
                       GLOBARG float * RESTRICT cpf5,
                       GLOBARG float * RESTRICT cpf6,
                       GLOBARG float * RESTRICT cpf7,
                       GLOBARG float * RESTRICT cpf8,
                       GLOBARG float * RESTRICT cpf9)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0)+FDOH;
    int gidy = get_global_id(1)+FDOH;
    int gidx = get_global_id(2)+FDOH;
#else
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidy = blockIdx.y*blockDim.y + threadIdx.y+FDOH;
    int gidx = blockIdx.z*blockDim.z + threadIdx.z+FDOH;
#endif
    if (gidz>NZ-FDOH-1 || gidy>NY-FDOH-1 || gidx>NX-FDOH-1){
        return;
    }
    int indv = gidx*NY*NZ+gidy*NZ+gidz;
    int i,j,k,indm;
    float e_vxr = vxr[indv];
    float e_vyr = vyr[indv];
    float e_vzr = vzr[indv];
    float f3 = e_vxr;
    float f6 = e_vyr;
    float f9 = e_vzr;
    float f2 = e_vxr;
    float f5 = e_vyr;
    float f8 = e_vzr;
    float f1 = e_vxr;
    float f4 = e_vyr;
    float f7 = e_vzr;
    if (gidz>NZ-NAB-FDOH-1){
        i =gidx-FDOH;
        j =gidy-FDOH;
        k =gidz - NZ+NAB+FDOH+NAB;
        indm=2*NAB-1-k;
        psi_sxz_zr(k,j,i) = b_z[indm+1] * psi_sxz_zr(k,j,i) + a_z[indm+1] * e_vxr;
        f3 = e_vxr / K_z[indm+1] + psi_sxz_zr(k,j,i);
        psi_syz_zr(k,j,i) = b_z[indm+1] * psi_syz_zr(k,j,i) + a_z[indm+1] * e_vyr;
        f6 = e_vyr / K_z[indm+1] + psi_syz_zr(k,j,i);
        psi_szz_zr(k,j,i) = b_z_half[indm] * psi_szz_zr(k,j,i) + a_z_half[indm] * e_vzr;
        f9 = e_vzr / K_z_half[indm] + psi_szz_zr(k,j,i);
    }
#if FREESURF==0
    else if (gidz-FDOH<NAB){
        i =gidx-FDOH;
        j =gidy-FDOH;
        k =gidz-FDOH;
        psi_sxz_zr(k,j,i) = b_z[k] * psi_sxz_zr(k,j,i) + a_z[k] * e_vxr;
        f3 = e_vxr / K_z[k] + psi_sxz_zr(k,j,i);
        psi_syz_zr(k,j,i) = b_z[k] * psi_syz_zr(k,j,i) + a_z[k] * e_vyr;
        f6 = e_vyr / K_z[k] + psi_syz_zr(k,j,i);
        psi_szz_zr(k,j,i) = b_z_half[k] * psi_szz_zr(k,j,i) + a_z_half[k] * e_vzr;
        f9 = e_vzr / K_z_half[k] + psi_szz_zr(k,j,i);
    }
#endif
    if (gidy-FDOH<NAB){
        i =gidx-FDOH;
        j =gidy-FDOH;
        k =gidz-FDOH;
        psi_sxy_yr(k,j,i) = b_y[j] * psi_sxy_yr(k,j,i) + a_y[j] * e_vxr;
        f2 = e_vxr / K_y[j] + psi_sxy_yr(k,j,i);
        psi_syy_yr(k,j,i) = b_y_half[j] * psi_syy_yr(k,j,i) + a_y_half[j] * e_vyr;
        f5 = e_vyr / K_y_half[j] + psi_syy_yr(k,j,i);
        psi_syz_yr(k,j,i) = b_y[j] * psi_syz_yr(k,j,i) + a_y[j] * e_vzr;
        f8 = e_vzr / K_y[j] + psi_syz_yr(k,j,i);
    }
    else if (gidy>NY-NAB-FDOH-1){
        i =gidx-FDOH;
        j =gidy - NY+NAB+FDOH+NAB;
        k =gidz-FDOH;
        indm=2*NAB-1-j;
        psi_sxy_yr(k,j,i) = b_y[indm+1] * psi_sxy_yr(k,j,i) + a_y[indm+1] * e_vxr;
        f2 = e_vxr / K_y[indm+1] + psi_sxy_yr(k,j,i);
        psi_syy_yr(k,j,i) = b_y_half[indm] * psi_syy_yr(k,j,i) + a_y_half[indm] * e_vyr;
        f5 = e_vyr / K_y_half[indm] + psi_syy_yr(k,j,i);
        psi_syz_yr(k,j,i) = b_y[indm+1] * psi_syz_yr(k,j,i) + a_y[indm+1] * e_vzr;
        f8 = e_vzr / K_y[indm+1] + psi_syz_yr(k,j,i);
    }
    if (gidx-FDOH<NAB){
        i =gidx-FDOH;
        j =gidy-FDOH;
        k =gidz-FDOH;
        psi_sxx_xr(k,j,i) = b_x_half[i] * psi_sxx_xr(k,j,i) + a_x_half[i] * e_vxr;
        f1 = e_vxr / K_x_half[i] + psi_sxx_xr(k,j,i);
        psi_sxy_xr(k,j,i) = b_x[i] * psi_sxy_xr(k,j,i) + a_x[i] * e_vyr;
        f4 = e_vyr / K_x[i] + psi_sxy_xr(k,j,i);
        psi_sxz_xr(k,j,i) = b_x[i] * psi_sxz_xr(k,j,i) + a_x[i] * e_vzr;
        f7 = e_vzr / K_x[i] + psi_sxz_xr(k,j,i);
    }
    if (gidx>NX-NAB-FDOH-1){
        i =gidx - NX+NAB+FDOH+NAB;
        j =gidy-FDOH;
        k =gidz-FDOH;
        indm=2*NAB-1-i;
        psi_sxx_xr(k,j,i) = b_x_half[indm] * psi_sxx_xr(k,j,i) + a_x_half[indm] * e_vxr;
        f1 = e_vxr / K_x_half[indm] + psi_sxx_xr(k,j,i);
        psi_sxy_xr(k,j,i) = b_x[indm+1] * psi_sxy_xr(k,j,i) + a_x[indm+1] * e_vyr;
        f4 = e_vyr / K_x[indm+1] + psi_sxy_xr(k,j,i);
        psi_sxz_xr(k,j,i) = b_x[indm+1] * psi_sxz_xr(k,j,i) + a_x[indm+1] * e_vzr;
        f7 = e_vzr / K_x[indm+1] + psi_sxz_xr(k,j,i);
    }
    cpf1[indv] = f1;
    cpf2[indv] = f2;
    cpf3[indv] = f3;
    cpf4[indv] = f4;
    cpf5[indv] = f5;
    cpf6[indv] = f6;
    cpf7[indv] = f7;
    cpf8[indv] = f8;
    cpf9[indv] = f9;
}

FUNDEF void cpml_adj_s(GLOBARG const float * RESTRICT sxxr, GLOBARG const float * RESTRICT syyr, GLOBARG const float * RESTRICT szzr, GLOBARG const float * RESTRICT sxyr, GLOBARG const float * RESTRICT syzr, GLOBARG const float * RESTRICT sxzr,
                       GLOBARG const float * RESTRICT K_x,  GLOBARG const float * RESTRICT a_x,  GLOBARG const float * RESTRICT b_x,
                       GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                       GLOBARG const float * RESTRICT K_y,  GLOBARG const float * RESTRICT a_y,  GLOBARG const float * RESTRICT b_y,
                       GLOBARG const float * RESTRICT K_y_half,  GLOBARG const float * RESTRICT a_y_half,  GLOBARG const float * RESTRICT b_y_half,
                       GLOBARG const float * RESTRICT K_z,  GLOBARG const float * RESTRICT a_z,  GLOBARG const float * RESTRICT b_z,
                       GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                       GLOBARG float * RESTRICT psi_vx_zr,
                       GLOBARG float * RESTRICT psi_vy_zr,
                       GLOBARG float * RESTRICT psi_vz_zr,
                       GLOBARG float * RESTRICT psi_vx_yr,
                       GLOBARG float * RESTRICT psi_vy_yr,
                       GLOBARG float * RESTRICT psi_vz_yr,
                       GLOBARG float * RESTRICT psi_vx_xr,
                       GLOBARG float * RESTRICT psi_vy_xr,
                       GLOBARG float * RESTRICT psi_vz_xr,
                       GLOBARG float * RESTRICT cpf1,
                       GLOBARG float * RESTRICT cpf2,
                       GLOBARG float * RESTRICT cpf3,
                       GLOBARG float * RESTRICT cpf4,
                       GLOBARG float * RESTRICT cpf5,
                       GLOBARG float * RESTRICT cpf6,
                       GLOBARG float * RESTRICT cpf7,
                       GLOBARG float * RESTRICT cpf8,
                       GLOBARG float * RESTRICT cpf9)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0)+FDOH;
    int gidy = get_global_id(1)+FDOH;
    int gidx = get_global_id(2)+FDOH;
#else
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidy = blockIdx.y*blockDim.y + threadIdx.y+FDOH;
    int gidx = blockIdx.z*blockDim.z + threadIdx.z+FDOH;
#endif
    if (gidz>NZ-FDOH-1 || gidy>NY-FDOH-1 || gidx>NX-FDOH-1){
        return;
    }
    int indv = gidx*NY*NZ+gidy*NZ+gidz;
    int i,j,k,indm;
    float e_sxxr = sxxr[indv];
    float e_syyr = syyr[indv];
    float e_szzr = szzr[indv];
    float e_sxyr = sxyr[indv];
    float e_syzr = syzr[indv];
    float e_sxzr = sxzr[indv];
    float f3 = e_sxzr;
    float f6 = e_syzr;
    float f9 = e_szzr;
    float f2 = e_sxyr;
    float f5 = e_syyr;
    float f8 = e_syzr;
    float f1 = e_sxxr;
    float f4 = e_sxyr;
    float f7 = e_sxzr;
    if (gidz>NZ-NAB-FDOH-1){
        i =gidx-FDOH;
        j =gidy-FDOH;
        k =gidz - NZ+NAB+FDOH+NAB;
        indm=2*NAB-1-k;
        psi_vx_zr(k,j,i) = b_z_half[indm] * psi_vx_zr(k,j,i) + a_z_half[indm] * e_sxzr;
        f3 = e_sxzr / K_z_half[indm] + psi_vx_zr(k,j,i);
        psi_vy_zr(k,j,i) = b_z_half[indm] * psi_vy_zr(k,j,i) + a_z_half[indm] * e_syzr;
        f6 = e_syzr / K_z_half[indm] + psi_vy_zr(k,j,i);
        psi_vz_zr(k,j,i) = b_z[indm+1] * psi_vz_zr(k,j,i) + a_z[indm+1] * e_szzr;
        f9 = e_szzr / K_z[indm+1] + psi_vz_zr(k,j,i);
    }
#if FREESURF==0
    else if (gidz-FDOH<NAB){
        i =gidx-FDOH;
        j =gidy-FDOH;
        k =gidz-FDOH;
        psi_vx_zr(k,j,i) = b_z_half[k] * psi_vx_zr(k,j,i) + a_z_half[k] * e_sxzr;
        f3 = e_sxzr / K_z_half[k] + psi_vx_zr(k,j,i);
        psi_vy_zr(k,j,i) = b_z_half[k] * psi_vy_zr(k,j,i) + a_z_half[k] * e_syzr;
        f6 = e_syzr / K_z_half[k] + psi_vy_zr(k,j,i);
        psi_vz_zr(k,j,i) = b_z[k] * psi_vz_zr(k,j,i) + a_z[k] * e_szzr;
        f9 = e_szzr / K_z[k] + psi_vz_zr(k,j,i);
    }
#endif
    if (gidy-FDOH<NAB){
        i =gidx-FDOH;
        j =gidy-FDOH;
        k =gidz-FDOH;
        psi_vx_yr(k,j,i) = b_y_half[j] * psi_vx_yr(k,j,i) + a_y_half[j] * e_sxyr;
        f2 = e_sxyr / K_y_half[j] + psi_vx_yr(k,j,i);
        psi_vy_yr(k,j,i) = b_y[j] * psi_vy_yr(k,j,i) + a_y[j] * e_syyr;
        f5 = e_syyr / K_y[j] + psi_vy_yr(k,j,i);
        psi_vz_yr(k,j,i) = b_y_half[j] * psi_vz_yr(k,j,i) + a_y_half[j] * e_syzr;
        f8 = e_syzr / K_y_half[j] + psi_vz_yr(k,j,i);
    }
    else if (gidy>NY-NAB-FDOH-1){
        i =gidx-FDOH;
        j =gidy - NY+NAB+FDOH+NAB;
        k =gidz-FDOH;
        indm=2*NAB-1-j;
        psi_vx_yr(k,j,i) = b_y_half[indm] * psi_vx_yr(k,j,i) + a_y_half[indm] * e_sxyr;
        f2 = e_sxyr / K_y_half[indm] + psi_vx_yr(k,j,i);
        psi_vy_yr(k,j,i) = b_y[indm+1] * psi_vy_yr(k,j,i) + a_y[indm+1] * e_syyr;
        f5 = e_syyr / K_y[indm+1] + psi_vy_yr(k,j,i);
        psi_vz_yr(k,j,i) = b_y_half[indm] * psi_vz_yr(k,j,i) + a_y_half[indm] * e_syzr;
        f8 = e_syzr / K_y_half[indm] + psi_vz_yr(k,j,i);
    }
    if (gidx-FDOH<NAB){
        i =gidx-FDOH;
        j =gidy-FDOH;
        k =gidz-FDOH;
        psi_vx_xr(k,j,i) = b_x[i] * psi_vx_xr(k,j,i) + a_x[i] * e_sxxr;
        f1 = e_sxxr / K_x[i] + psi_vx_xr(k,j,i);
        psi_vy_xr(k,j,i) = b_x_half[i] * psi_vy_xr(k,j,i) + a_x_half[i] * e_sxyr;
        f4 = e_sxyr / K_x_half[i] + psi_vy_xr(k,j,i);
        psi_vz_xr(k,j,i) = b_x_half[i] * psi_vz_xr(k,j,i) + a_x_half[i] * e_sxzr;
        f7 = e_sxzr / K_x_half[i] + psi_vz_xr(k,j,i);
    }
    if (gidx>NX-NAB-FDOH-1){
        i =gidx - NX+NAB+FDOH+NAB;
        j =gidy-FDOH;
        k =gidz-FDOH;
        indm=2*NAB-1-i;
        psi_vx_xr(k,j,i) = b_x[indm+1] * psi_vx_xr(k,j,i) + a_x[indm+1] * e_sxxr;
        f1 = e_sxxr / K_x[indm+1] + psi_vx_xr(k,j,i);
        psi_vy_xr(k,j,i) = b_x_half[indm] * psi_vy_xr(k,j,i) + a_x_half[indm] * e_sxyr;
        f4 = e_sxyr / K_x_half[indm] + psi_vy_xr(k,j,i);
        psi_vz_xr(k,j,i) = b_x_half[indm] * psi_vz_xr(k,j,i) + a_x_half[indm] * e_sxzr;
        f7 = e_sxzr / K_x_half[indm] + psi_vz_xr(k,j,i);
    }
    cpf1[indv] = f1;
    cpf2[indv] = f2;
    cpf3[indv] = f3;
    cpf4[indv] = f4;
    cpf5[indv] = f5;
    cpf6[indv] = f6;
    cpf7[indv] = f7;
    cpf8[indv] = f8;
    cpf9[indv] = f9;
}

