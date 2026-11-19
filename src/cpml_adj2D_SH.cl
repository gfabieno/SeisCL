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

/* Exact adjoint of the 2D SH CPML (ABS_TYPE 1, BACK_PROP_TYPE 1): cpml_adj2D.cl (P-SV) for the SH fields.
 * Each adjoint field is filtered, before its spatial derivative, with the coefficients of the forward term it
 * feeds (psit = b psit + a e;  e~ = e/K + psit):
 *   cpml_adj_v (before update_adjs): vyr feeds the forward velocity update through d(sxy)/dx and d(syz)/dz
 *     (update_v2D_SH.cl: integer-node coefficients a_x, b_x, K_x and a_z, b_z, K_z, ind+1 on the far side):
 *     cpfa = F[sxy_x](vyr), cpfb = F[syz_z](vyr);
 *   cpml_adj_s (before update_adjv): sxyr and syzr feed the forward stress update through d(vy)/dx and d(vy)/dz
 *     (update_s2D_SH.cl: half-node coefficients):  cpfa = F[vy_x](sxyr), cpfb = F[vy_z](syzr).
 * The memories are the adjoint copies psi_*r of the forward memories (same layout). Outside the layers cpfa and
 * cpfb are copies of the fields. A separate file because clprogram.c binds the arguments by name. */

#define PSIX(z,x) ((x)*(NZS-2*FDOH)+(z))
#define PSIZ(z,x) ((x)*(2*NAB)+(z))

FUNDEF void cpml_adj_v(GLOBARG const STO * RESTRICT vyr,
                       GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                       GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                       GLOBARG float * RESTRICT psi_sxy_xr,      GLOBARG float * RESTRICT psi_syz_zr,
                       GLOBARG STO * RESTRICT cpfa,            GLOBARG STO * RESTRICT cpfb)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0)+FDOH;
    int gidx = get_global_id(1)+FDOH;
#else
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH;
#endif
    if (gidz>NZS-FDOH-1 || gidx>NX-FDOH-1){
        return;
    }
    int indv = gidx*NZS+gidz;
    int i,k,ind;
    float e = ldf(vyr, indv);
    float fa = e, fb = e;
    float ps;

    if (gidz>NZS-NAB-FDOH-1){
        i = gidx-FDOH;
        k = gidz-NZS+NAB+FDOH+NAB;
        ind = 2*NAB-1-k;
        ps = b_z[ind+1]*psi_syz_zr[PSIZ(k,i)] + a_z[ind+1]*e;
        psi_syz_zr[PSIZ(k,i)] = ps;
        fb = e/K_z[ind+1] + ps;
    }
#if FREESURF==0
    else if (gidz-FDOH<NAB){
        i = gidx-FDOH;
        k = gidz-FDOH;
        ps = b_z[k]*psi_syz_zr[PSIZ(k,i)] + a_z[k]*e;
        psi_syz_zr[PSIZ(k,i)] = ps;
        fb = e/K_z[k] + ps;
    }
#endif
    if (gidx-FDOH<NAB){
        i = gidx-FDOH;
        k = gidz-FDOH;
        ps = b_x[i]*psi_sxy_xr[PSIX(k,i)] + a_x[i]*e;
        psi_sxy_xr[PSIX(k,i)] = ps;
        fa = e/K_x[i] + ps;
    }
    if (gidx>NX-NAB-FDOH-1){
        i = gidx-NX+NAB+FDOH+NAB;
        k = gidz-FDOH;
        ind = 2*NAB-1-i;
        ps = b_x[ind+1]*psi_sxy_xr[PSIX(k,i)] + a_x[ind+1]*e;
        psi_sxy_xr[PSIX(k,i)] = ps;
        fa = e/K_x[ind+1] + ps;
    }
    stf(cpfa, indv, fa);
    stf(cpfb, indv, fb);
}

FUNDEF void cpml_adj_s(GLOBARG const STO * RESTRICT sxyr, GLOBARG const STO * RESTRICT syzr,
                       GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                       GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                       GLOBARG float * RESTRICT psi_vy_xr,       GLOBARG float * RESTRICT psi_vy_zr,
                       GLOBARG STO * RESTRICT cpfa,            GLOBARG STO * RESTRICT cpfb)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0)+FDOH;
    int gidx = get_global_id(1)+FDOH;
#else
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH;
#endif
    if (gidz>NZS-FDOH-1 || gidx>NX-FDOH-1){
        return;
    }
    int indv = gidx*NZS+gidz;
    int i,k,ind;
    float exy = ldf(sxyr, indv), eyz = ldf(syzr, indv);
    float fa = exy, fb = eyz;
    float ps;

    if (gidz>NZS-NAB-FDOH-1){
        i = gidx-FDOH;
        k = gidz-NZS+NAB+FDOH+NAB;
        ind = 2*NAB-1-k;
        ps = b_z_half[ind]*psi_vy_zr[PSIZ(k,i)] + a_z_half[ind]*eyz;
        psi_vy_zr[PSIZ(k,i)] = ps;
        fb = eyz/K_z_half[ind] + ps;
    }
#if FREESURF==0
    else if (gidz-FDOH<NAB){
        i = gidx-FDOH;
        k = gidz-FDOH;
        ps = b_z_half[k]*psi_vy_zr[PSIZ(k,i)] + a_z_half[k]*eyz;
        psi_vy_zr[PSIZ(k,i)] = ps;
        fb = eyz/K_z_half[k] + ps;
    }
#endif
    if (gidx-FDOH<NAB){
        i = gidx-FDOH;
        k = gidz-FDOH;
        ps = b_x_half[i]*psi_vy_xr[PSIX(k,i)] + a_x_half[i]*exy;
        psi_vy_xr[PSIX(k,i)] = ps;
        fa = exy/K_x_half[i] + ps;
    }
    if (gidx>NX-NAB-FDOH-1){
        i = gidx-NX+NAB+FDOH+NAB;
        k = gidz-FDOH;
        ind = 2*NAB-1-i;
        ps = b_x_half[ind]*psi_vy_xr[PSIX(k,i)] + a_x_half[ind]*exy;
        psi_vy_xr[PSIX(k,i)] = ps;
        fa = exy/K_x_half[ind] + ps;
    }
    stf(cpfa, indv, fa);
    stf(cpfb, indv, fb);
}
