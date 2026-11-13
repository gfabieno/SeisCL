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

/* Exact adjoint of the 2D CPML (ABS_TYPE 1, BACK_PROP_TYPE 1).
 *
 * Forward (update_v2D.cl, update_s2D.cl): each spatial derivative d of a field,
 * in the layer of its direction, is filtered at the cell where it is used:
 *      psi = b psi + a d ;     d~ = d/K + psi.
 * Adjoint: the transpose applies the same recursion to the ADJOINT FIELD
 * before the spatial derivative, with the coefficients of the forward term
 * that the field feeds (change of variables psit = a lambda_psi / b):
 *      psit = b psit + a e ;   e~ = e/K + psit ;   then derivative of e~.
 * cpml_adj_v filters the adjoint velocities (coefficients of the forward
 * velocity update, run before update_adjs); cpml_adj_s filters the adjoint
 * stresses (coefficients of the forward stress update, run before
 * update_adjv). The memories psit are the adjoint copies psi_*r of the
 * forward memories (same layout). The filtered fields go to the scratch
 * buffers cpfa..cpfd, which update_adjs2D.cl / update_adjv2D.cl
 * differentiate. Outside the layers they are copies of the fields.
 * Checked against the exact matrix transpose of the time step (3.8e-16),
 * notes: torch-robust/cpmladj/ref.py. Single device only (assign_modeling_case.c).
 */

#define psi_sxx_xr(z,x) psi_sxx_xr[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_xr(z,x) psi_sxz_xr[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_zr(z,x) psi_sxz_zr[(x)*(2*NAB)+(z)]
#define psi_szz_zr(z,x) psi_szz_zr[(x)*(2*NAB)+(z)]
#define psi_vx_xr(z,x) psi_vx_xr[(x)*(NZ-2*FDOH)+(z)]
#define psi_vz_xr(z,x) psi_vz_xr[(x)*(NZ-2*FDOH)+(z)]
#define psi_vx_zr(z,x) psi_vx_zr[(x)*(2*NAB)+(z)]
#define psi_vz_zr(z,x) psi_vz_zr[(x)*(2*NAB)+(z)]

/* Velocities: cpfa = F[sxx_x](vxr), cpfb = F[sxz_z](vxr),
               cpfc = F[szz_z](vzr), cpfd = F[sxz_x](vzr). */
FUNDEF void cpml_adj_v(GLOBARG const float * RESTRICT vxr,  GLOBARG const float * RESTRICT vzr,
                       GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                       GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                       GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                       GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                       GLOBARG float * RESTRICT psi_sxx_xr,      GLOBARG float * RESTRICT psi_sxz_xr,
                       GLOBARG float * RESTRICT psi_sxz_zr,      GLOBARG float * RESTRICT psi_szz_zr,
                       GLOBARG float * RESTRICT cpfa,            GLOBARG float * RESTRICT cpfb,
                       GLOBARG float * RESTRICT cpfc,            GLOBARG float * RESTRICT cpfd)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0)+FDOH;
    int gidx = get_global_id(1)+FDOH;
#else
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH;
#endif
    if (gidz>NZ-FDOH-1 || gidx>NX-FDOH-1){
        return;
    }
    int indv = gidx*NZ+gidz;
    int i,k,ind;
    float ex = vxr[indv], ez = vzr[indv];
    float fa = ex, fb = ex, fc = ez, fd = ez;

    if (gidz>NZ-NAB-FDOH-1){
        i =gidx-FDOH;
        k =gidz - NZ+NAB+FDOH+NAB;
        ind=2*NAB-1-k;
        psi_sxz_zr(k,i) = b_z[ind+1] * psi_sxz_zr(k,i) + a_z[ind+1] * ex;
        fb = ex / K_z[ind+1] + psi_sxz_zr(k,i);
        psi_szz_zr(k,i) = b_z_half[ind] * psi_szz_zr(k,i) + a_z_half[ind] * ez;
        fc = ez / K_z_half[ind] + psi_szz_zr(k,i);
    }
#if FREESURF==0
    else if (gidz-FDOH<NAB){
        i =gidx-FDOH;
        k =gidz-FDOH;
        psi_sxz_zr(k,i) = b_z[k] * psi_sxz_zr(k,i) + a_z[k] * ex;
        fb = ex / K_z[k] + psi_sxz_zr(k,i);
        psi_szz_zr(k,i) = b_z_half[k] * psi_szz_zr(k,i) + a_z_half[k] * ez;
        fc = ez / K_z_half[k] + psi_szz_zr(k,i);
    }
#endif
    if (gidx-FDOH<NAB){
        i =gidx-FDOH;
        k =gidz-FDOH;
        psi_sxx_xr(k,i) = b_x_half[i] * psi_sxx_xr(k,i) + a_x_half[i] * ex;
        fa = ex / K_x_half[i] + psi_sxx_xr(k,i);
        psi_sxz_xr(k,i) = b_x[i] * psi_sxz_xr(k,i) + a_x[i] * ez;
        fd = ez / K_x[i] + psi_sxz_xr(k,i);
    }
    if (gidx>NX-NAB-FDOH-1){
        i =gidx - NX+NAB+FDOH+NAB;
        k =gidz-FDOH;
        ind=2*NAB-1-i;
        psi_sxx_xr(k,i) = b_x_half[ind] * psi_sxx_xr(k,i) + a_x_half[ind] * ex;
        fa = ex / K_x_half[ind] + psi_sxx_xr(k,i);
        psi_sxz_xr(k,i) = b_x[ind+1] * psi_sxz_xr(k,i) + a_x[ind+1] * ez;
        fd = ez / K_x[ind+1] + psi_sxz_xr(k,i);
    }
    cpfa[indv] = fa;
    cpfb[indv] = fb;
    cpfc[indv] = fc;
    cpfd[indv] = fd;
}

/* Stresses: cpfa = F[vx_x](sxxr), cpfb = F[vx_z](sxzr),
             cpfc = F[vz_z](szzr), cpfd = F[vz_x](sxzr). */
FUNDEF void cpml_adj_s(GLOBARG const float * RESTRICT sxxr, GLOBARG const float * RESTRICT szzr,
                       GLOBARG const float * RESTRICT sxzr,
                       GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                       GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                       GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                       GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                       GLOBARG float * RESTRICT psi_vx_xr,       GLOBARG float * RESTRICT psi_vz_xr,
                       GLOBARG float * RESTRICT psi_vx_zr,       GLOBARG float * RESTRICT psi_vz_zr,
                       GLOBARG float * RESTRICT cpfa,            GLOBARG float * RESTRICT cpfb,
                       GLOBARG float * RESTRICT cpfc,            GLOBARG float * RESTRICT cpfd)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0)+FDOH;
    int gidx = get_global_id(1)+FDOH;
#else
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH;
#endif
    if (gidz>NZ-FDOH-1 || gidx>NX-FDOH-1){
        return;
    }
    int indv = gidx*NZ+gidz;
    int i,k,ind;
    float exx = sxxr[indv], ezz = szzr[indv], exz = sxzr[indv];
    float fa = exx, fb = exz, fc = ezz, fd = exz;

    if (gidz>NZ-NAB-FDOH-1){
        i =gidx-FDOH;
        k =gidz - NZ+NAB+FDOH+NAB;
        ind=2*NAB-1-k;
        psi_vx_zr(k,i) = b_z_half[ind] * psi_vx_zr(k,i) + a_z_half[ind] * exz;
        fb = exz / K_z_half[ind] + psi_vx_zr(k,i);
        psi_vz_zr(k,i) = b_z[ind+1] * psi_vz_zr(k,i) + a_z[ind+1] * ezz;
        fc = ezz / K_z[ind+1] + psi_vz_zr(k,i);
    }
#if FREESURF==0
    else if (gidz-FDOH<NAB){
        i =gidx-FDOH;
        k =gidz-FDOH;
        psi_vx_zr(k,i) = b_z_half[k] * psi_vx_zr(k,i) + a_z_half[k] * exz;
        fb = exz / K_z_half[k] + psi_vx_zr(k,i);
        psi_vz_zr(k,i) = b_z[k] * psi_vz_zr(k,i) + a_z[k] * ezz;
        fc = ezz / K_z[k] + psi_vz_zr(k,i);
    }
#endif
    if (gidx-FDOH<NAB){
        i =gidx-FDOH;
        k =gidz-FDOH;
        psi_vx_xr(k,i) = b_x[i] * psi_vx_xr(k,i) + a_x[i] * exx;
        fa = exx / K_x[i] + psi_vx_xr(k,i);
        psi_vz_xr(k,i) = b_x_half[i] * psi_vz_xr(k,i) + a_x_half[i] * exz;
        fd = exz / K_x_half[i] + psi_vz_xr(k,i);
    }
    if (gidx>NX-NAB-FDOH-1){
        i =gidx - NX+NAB+FDOH+NAB;
        k =gidz-FDOH;
        ind=2*NAB-1-i;
        psi_vx_xr(k,i) = b_x[ind+1] * psi_vx_xr(k,i) + a_x[ind+1] * exx;
        fa = exx / K_x[ind+1] + psi_vx_xr(k,i);
        psi_vz_xr(k,i) = b_x_half[ind] * psi_vz_xr(k,i) + a_x_half[ind] * exz;
        fd = exz / K_x_half[ind] + psi_vz_xr(k,i);
    }
    cpfa[indv] = fa;
    cpfb[indv] = fb;
    cpfc[indv] = fc;
    cpfd[indv] = fd;
}
