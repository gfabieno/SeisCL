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
 * differentiate. Outside the layers they are not written: the adjoint updates read the
 * fields there instead (header_vec.cl, load_tile*_cpf). Without tiles (NOTILE) they are
 * written in the layers and in a band next to them (NT_SKIP2).
 * Checked against the exact matrix transpose of the time step (3.8e-16),
 * notes: torch-robust/cpmladj/ref.py. With domain decomposition along x, only
 * the first and last subdomains have x layers; an inner cut gets identity copies
 * over its halo (CPML_X, HALO_X in header_vec.cl).
 */

#define psi_sxx_xr_I(z,x) ((x)*(NZS-2*FDOH)+(z))
#define psi_sxz_xr_I(z,x) ((x)*(NZS-2*FDOH)+(z))
#define psi_sxz_zr_I(z,x) ((x)*(2*NAB)+(z))
#define psi_szz_zr_I(z,x) ((x)*(2*NAB)+(z))
#define psi_vx_xr_I(z,x) ((x)*(NZS-2*FDOH)+(z))
#define psi_vz_xr_I(z,x) ((x)*(NZS-2*FDOH)+(z))
#define psi_vx_zr_I(z,x) ((x)*(2*NAB)+(z))
#define psi_vz_zr_I(z,x) ((x)*(2*NAB)+(z))

/* Velocities: cpfa = F[sxx_x](vxr), cpfb = F[sxz_z](vxr),
               cpfc = F[szz_z](vzr), cpfd = F[sxz_x](vzr). */
FUNDEF void cpml_adj_v(GLOBARG const STO * RESTRICT vxr,  GLOBARG const STO * RESTRICT vzr,
                       GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                       GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                       GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                       GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                       GLOBARG float * RESTRICT psi_sxx_xr,      GLOBARG float * RESTRICT psi_sxz_xr,
                       GLOBARG float * RESTRICT psi_sxz_zr,      GLOBARG float * RESTRICT psi_szz_zr,
                       GLOBARG STO * RESTRICT cpfa,            GLOBARG STO * RESTRICT cpfb,
                       GLOBARG STO * RESTRICT cpfc,            GLOBARG STO * RESTRICT cpfd)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0)+FDOH;
    int gidx = get_global_id(1)+FDOH-CPML_XH0;
#else
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH-CPML_XH0;
#endif
    if (gidz>NZS-FDOH-1 || gidx>NX-FDOH-1+CPML_XH1){
        return;
    }
    int hx = CPML_HALOX(gidx);   /* x halo of an inner cut: identity copies */
#if NOTILE
    if (NT_SKIP2(gidz, gidx)) return;      /* no adjoint update reads the copy here (header_vec.cl) */
#else
    if (!CPML_CELL2(gidz, gidx)) return;   /* identity filter: read in place (header_vec.cl) */
#endif
    int indv = gidx*NZS+gidz;
    int i,k,ind;
    float ex = ldf(vxr, indv), ez = ldf(vzr, indv);
    float fa = ex, fb = ex, fc = ez, fd = ez;

    if (!hx && gidz>NZS-NAB-FDOH-1){
        i =gidx-FDOH;
        k =gidz - NZS+NAB+FDOH+NAB;
        ind=2*NAB-1-k;
        { float ps = b_z[ind+1] * psi_sxz_zr[psi_sxz_zr_I(k,i)] + a_z[ind+1] * ex; psi_sxz_zr[psi_sxz_zr_I(k,i)] = ps; }
        fb = ex / K_z[ind+1] + psi_sxz_zr[psi_sxz_zr_I(k,i)];
        { float ps = b_z_half[ind] * psi_szz_zr[psi_szz_zr_I(k,i)] + a_z_half[ind] * ez; psi_szz_zr[psi_szz_zr_I(k,i)] = ps; }
        fc = ez / K_z_half[ind] + psi_szz_zr[psi_szz_zr_I(k,i)];
    }
#if FREESURF==0
    else if (!hx && gidz-FDOH<NAB){
        i =gidx-FDOH;
        k =gidz-FDOH;
        { float ps = b_z[k] * psi_sxz_zr[psi_sxz_zr_I(k,i)] + a_z[k] * ex; psi_sxz_zr[psi_sxz_zr_I(k,i)] = ps; }
        fb = ex / K_z[k] + psi_sxz_zr[psi_sxz_zr_I(k,i)];
        { float ps = b_z_half[k] * psi_szz_zr[psi_szz_zr_I(k,i)] + a_z_half[k] * ez; psi_szz_zr[psi_szz_zr_I(k,i)] = ps; }
        fc = ez / K_z_half[k] + psi_szz_zr[psi_szz_zr_I(k,i)];
    }
#endif
    if (X_FIRST && gidx-FDOH<NAB){
        i =gidx-FDOH;
        k =gidz-FDOH;
        { float ps = b_x_half[i] * psi_sxx_xr[psi_sxx_xr_I(k,i)] + a_x_half[i] * ex; psi_sxx_xr[psi_sxx_xr_I(k,i)] = ps; }
        fa = ex / K_x_half[i] + psi_sxx_xr[psi_sxx_xr_I(k,i)];
        { float ps = b_x[i] * psi_sxz_xr[psi_sxz_xr_I(k,i)] + a_x[i] * ez; psi_sxz_xr[psi_sxz_xr_I(k,i)] = ps; }
        fd = ez / K_x[i] + psi_sxz_xr[psi_sxz_xr_I(k,i)];
    }
    if (X_LAST && gidx>NX-NAB-FDOH-1){
        i =gidx - NX+NAB+FDOH+NAB;
        k =gidz-FDOH;
        ind=2*NAB-1-i;
        { float ps = b_x_half[ind] * psi_sxx_xr[psi_sxx_xr_I(k,i)] + a_x_half[ind] * ex; psi_sxx_xr[psi_sxx_xr_I(k,i)] = ps; }
        fa = ex / K_x_half[ind] + psi_sxx_xr[psi_sxx_xr_I(k,i)];
        { float ps = b_x[ind+1] * psi_sxz_xr[psi_sxz_xr_I(k,i)] + a_x[ind+1] * ez; psi_sxz_xr[psi_sxz_xr_I(k,i)] = ps; }
        fd = ez / K_x[ind+1] + psi_sxz_xr[psi_sxz_xr_I(k,i)];
    }
    stf(cpfa, indv, fa);
    stf(cpfb, indv, fb);
    stf(cpfc, indv, fc);
    stf(cpfd, indv, fd);
}

/* Stresses: cpfa = F[vx_x](sxxr), cpfb = F[vx_z](sxzr),
             cpfc = F[vz_z](szzr), cpfd = F[vz_x](sxzr). */
FUNDEF void cpml_adj_s(GLOBARG const STO * RESTRICT sxxr, GLOBARG const STO * RESTRICT szzr,
                       GLOBARG const STO * RESTRICT sxzr,
                       GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                       GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                       GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                       GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                       GLOBARG float * RESTRICT psi_vx_xr,       GLOBARG float * RESTRICT psi_vz_xr,
                       GLOBARG float * RESTRICT psi_vx_zr,       GLOBARG float * RESTRICT psi_vz_zr,
                       GLOBARG STO * RESTRICT cpfa,            GLOBARG STO * RESTRICT cpfb,
                       GLOBARG STO * RESTRICT cpfc,            GLOBARG STO * RESTRICT cpfd)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0)+FDOH;
    int gidx = get_global_id(1)+FDOH-CPML_XH0;
#else
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH-CPML_XH0;
#endif
    if (gidz>NZS-FDOH-1 || gidx>NX-FDOH-1+CPML_XH1){
        return;
    }
    int hx = CPML_HALOX(gidx);   /* x halo of an inner cut: identity copies */
#if NOTILE
    if (NT_SKIP2(gidz, gidx)) return;      /* no adjoint update reads the copy here (header_vec.cl) */
#else
    if (!CPML_CELL2(gidz, gidx)) return;   /* identity filter: read in place (header_vec.cl) */
#endif
    int indv = gidx*NZS+gidz;
    int i,k,ind;
    float exx = ldf(sxxr, indv), ezz = ldf(szzr, indv), exz = ldf(sxzr, indv);
    float fa = exx, fb = exz, fc = ezz, fd = exz;

    if (!hx && gidz>NZS-NAB-FDOH-1){
        i =gidx-FDOH;
        k =gidz - NZS+NAB+FDOH+NAB;
        ind=2*NAB-1-k;
        { float ps = b_z_half[ind] * psi_vx_zr[psi_vx_zr_I(k,i)] + a_z_half[ind] * exz; psi_vx_zr[psi_vx_zr_I(k,i)] = ps; }
        fb = exz / K_z_half[ind] + psi_vx_zr[psi_vx_zr_I(k,i)];
        { float ps = b_z[ind+1] * psi_vz_zr[psi_vz_zr_I(k,i)] + a_z[ind+1] * ezz; psi_vz_zr[psi_vz_zr_I(k,i)] = ps; }
        fc = ezz / K_z[ind+1] + psi_vz_zr[psi_vz_zr_I(k,i)];
    }
#if FREESURF==0
    else if (!hx && gidz-FDOH<NAB){
        i =gidx-FDOH;
        k =gidz-FDOH;
        { float ps = b_z_half[k] * psi_vx_zr[psi_vx_zr_I(k,i)] + a_z_half[k] * exz; psi_vx_zr[psi_vx_zr_I(k,i)] = ps; }
        fb = exz / K_z_half[k] + psi_vx_zr[psi_vx_zr_I(k,i)];
        { float ps = b_z[k] * psi_vz_zr[psi_vz_zr_I(k,i)] + a_z[k] * ezz; psi_vz_zr[psi_vz_zr_I(k,i)] = ps; }
        fc = ezz / K_z[k] + psi_vz_zr[psi_vz_zr_I(k,i)];
    }
#endif
    if (X_FIRST && gidx-FDOH<NAB){
        i =gidx-FDOH;
        k =gidz-FDOH;
        { float ps = b_x[i] * psi_vx_xr[psi_vx_xr_I(k,i)] + a_x[i] * exx; psi_vx_xr[psi_vx_xr_I(k,i)] = ps; }
        fa = exx / K_x[i] + psi_vx_xr[psi_vx_xr_I(k,i)];
        { float ps = b_x_half[i] * psi_vz_xr[psi_vz_xr_I(k,i)] + a_x_half[i] * exz; psi_vz_xr[psi_vz_xr_I(k,i)] = ps; }
        fd = exz / K_x_half[i] + psi_vz_xr[psi_vz_xr_I(k,i)];
    }
    if (X_LAST && gidx>NX-NAB-FDOH-1){
        i =gidx - NX+NAB+FDOH+NAB;
        k =gidz-FDOH;
        ind=2*NAB-1-i;
        { float ps = b_x[ind+1] * psi_vx_xr[psi_vx_xr_I(k,i)] + a_x[ind+1] * exx; psi_vx_xr[psi_vx_xr_I(k,i)] = ps; }
        fa = exx / K_x[ind+1] + psi_vx_xr[psi_vx_xr_I(k,i)];
        { float ps = b_x_half[ind] * psi_vz_xr[psi_vz_xr_I(k,i)] + a_x_half[ind] * exz; psi_vz_xr[psi_vz_xr_I(k,i)] = ps; }
        fd = exz / K_x_half[ind] + psi_vz_xr[psi_vz_xr_I(k,i)];
    }
    stf(cpfa, indv, fa);
    stf(cpfb, indv, fb);
    stf(cpfc, indv, fc);
    stf(cpfd, indv, fd);
}
