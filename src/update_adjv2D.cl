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

/* Adjoint update of the velocities in 2D P-SV, for every precision mode
   (FP16 0, 1, 2): float arithmetic, one cell at a time; storage and cells per
   thread come from header_vec.cl.
   BACK_PROP_TYPE 1: also rebuilds the forward velocities one step back (with the
   stored boundary values) and accumulates the density gradient in the
   summation-by-parts form, forward velocity x adjoint increment.
   Receiver cells: the residual term is added by rec_grad.cl. Free surface
   (FREESURF 1): the FDOH rows the adjoint surface changes get their density
   gradient from surface_grad (surface2D_adj.cl), in the direct form.
   CPML with BACK_PROP_TYPE 1: the derivatives of the adjoint stresses are taken
   of the fields filtered by cpml_adj_s (cpml_adj2D.cl), the exact adjoint. */

#define PSIX(z,x) ((x)*(NZS-2*FDOH)+(z))
#define PSIZ(z,x) ((x)*(2*NAB)+(z))

FUNDEF void update_adjv(int offcomm,
                        GLOBARG STO * RESTRICT vx,              GLOBARG STO * RESTRICT vz,
                        GLOBARG const STO * RESTRICT sxx,       GLOBARG const STO * RESTRICT szz,
                        GLOBARG const STO * RESTRICT sxz,
                        GLOBARG const STO * RESTRICT vxbnd,     GLOBARG const STO * RESTRICT vzbnd,
                        GLOBARG STO * RESTRICT vxr,             GLOBARG STO * RESTRICT vzr,
                        GLOBARG const STO * RESTRICT sxxr,      GLOBARG const STO * RESTRICT szzr,
                        GLOBARG const STO * RESTRICT sxzr,
                        GLOBARG const STO * RESTRICT rip,       GLOBARG const STO * RESTRICT rkp,
                        GLOBARG const float * RESTRICT taper,
                        GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                        GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                        GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                        GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                        GLOBARG float * RESTRICT psi_sxx_x,       GLOBARG float * RESTRICT psi_sxz_x,
                        GLOBARG float * RESTRICT psi_sxz_z,       GLOBARG float * RESTRICT psi_szz_z,
                        GLOBARG const STO * RESTRICT cpfa,      GLOBARG const STO * RESTRICT cpfb,
                        GLOBARG const STO * RESTRICT cpfc,      GLOBARG const STO * RESTRICT cpfd,
                        GLOBARG float * RESTRICT gradrip,       GLOBARG float * RESTRICT gradrkp,
                        GLOBARG float * RESTRICT Hrho,
                        GLOBARG const float * RESTRICT src,     GLOBARG const float * RESTRICT src_pos,
                        int nsrc,                               int nt,
                        int src_scale,                          int res_scale,
                        LOCARG)
{
    LOCDEF
    int gidx = GRP1*LSZ1 + LID1 + FDOH + offcomm;
    int zt = Z0 + NV*(GRP0*LSZ0 + LID0);
    int tz = NV*LSZ0 + 2*ZH, tx = LSZ1 + 2*FDOH;
    int tile_z0 = Z0 + NV*GRP0*LSZ0 - ZH, tile_x0 = GRP1*LSZ1 + offcomm;
    int lx = LID1 + FDOH;
    int lz0 = ZH + NV*LID0;
#if NOTILE
    /* SEISCL_NOTILE=1 (experiment): no shared tiles, the derivatives read the
       arrays in place; threads past the grid return before reading. */
#if COMM12==0
    if (zt > NZS-FDOH-1 || (gidx-offcomm) > (NX-FDOH-1-LCOMM)) return;
#else
    if (zt > NZS-FDOH-1) return;
#endif
    tz = NZS; lx = gidx; lz0 = zt;
#endif

    // Spatial derivatives of the forward stresses (to rebuild the forward velocities)
#if BACK_PROP_TYPE==1
    float sxx_x[NV], szz_z[NV], sxz_x[NV], sxz_z[NV];
    load_tile2(lvar, sxx, tile_z0, tile_x0, tz, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) sxx_x[c] = tdxp(lvar, tz, lz0+c, lx);
    BARRIER
    load_tile2(lvar, szz, tile_z0, tile_x0, tz, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) szz_z[c] = tdzp(lvar, tz, lz0+c, lx);
    BARRIER
    load_tile2(lvar, sxz, tile_z0, tile_x0, tz, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++){
        sxz_z[c] = tdzm(lvar, tz, lz0+c, lx);
        sxz_x[c] = tdxm(lvar, tz, lz0+c, lx);
    }
    BARRIER
#endif

    // Spatial derivatives of the adjoint stresses
    float sxx_xr[NV], szz_zr[NV], sxz_xr[NV], sxz_zr[NV];
#if ABS_TYPE==1 && BACK_PROP_TYPE==1
    /* exact CPML adjoint: the fields filtered by cpml_adj_s */
    load_tile2_cpf(lvar, cpfa, sxxr, tile_z0, tile_x0, tz, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) sxx_xr[c] = tdxp(lvar, tz, lz0+c, lx);
    BARRIER
    load_tile2_cpf(lvar, cpfb, sxzr, tile_z0, tile_x0, tz, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) sxz_zr[c] = tdzm(lvar, tz, lz0+c, lx);
    BARRIER
    load_tile2_cpf(lvar, cpfc, szzr, tile_z0, tile_x0, tz, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) szz_zr[c] = tdzp(lvar, tz, lz0+c, lx);
    BARRIER
    load_tile2_cpf(lvar, cpfd, sxzr, tile_z0, tile_x0, tz, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) sxz_xr[c] = tdxm(lvar, tz, lz0+c, lx);
#else
    load_tile2(lvar, sxxr, tile_z0, tile_x0, tz, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) sxx_xr[c] = tdxp(lvar, tz, lz0+c, lx);
    BARRIER
    load_tile2(lvar, szzr, tile_z0, tile_x0, tz, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) szz_zr[c] = tdzp(lvar, tz, lz0+c, lx);
    BARRIER
    load_tile2(lvar, sxzr, tile_z0, tile_x0, tz, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++){
        sxz_zr[c] = tdzm(lvar, tz, lz0+c, lx);
        sxz_xr[c] = tdxm(lvar, tz, lz0+c, lx);
    }
#endif

#if COMM12==0
    if (zt > NZS-FDOH-1 || (gidx-offcomm) > (NX-FDOH-1-LCOMM)) return;
#else
    if (zt > NZS-FDOH-1) return;
#endif

    int indv = gidx*NZS + zt;
    float lvxr[NV], lvzr[NV], lrip[NV], lrkp[NV];
    ldn(vxr, indv, lvxr);
    ldn(vzr, indv, lvzr);
    ldp2(rip, gidx, zt, lrip);
    ldp2(rkp, gidx, zt, lrkp);
#if BACK_PROP_TYPE==1
    float lvx[NV], lvz[NV];
    ldn(vx, indv, lvx);
    ldn(vz, indv, lvz);
#endif

    #pragma unroll
    for (int c=0; c<NV; c++){
        int gidz = zt + c;
        if (!INTERIOR_Z(gidz)) continue;              // halo cell of an aligned pair
        int indp = (gidx-FDOH)*(NZS-2*FDOH) + (gidz-FDOH);

        // Rebuild the forward velocities one step back; stored cells get their stored value
#if BACK_PROP_TYPE==1
        lvx[c] -= (sxx_x[c] + sxz_z[c])*lrip[c];
        lvz[c] -= (szz_z[c] + sxz_x[c])*lrkp[c];
        {
            int m = inject_ind(gidz, gidx);
            if (m != -1){
                lvx[c] = ldf(vxbnd, m);
                lvz[c] = ldf(vzbnd, m);
            }
        }
#endif

        float dsxx = sxx_xr[c], dszz = szz_zr[c], dsxzz = sxz_zr[c], dsxzx = sxz_xr[c];

        // CPML, BACK_PROP_TYPE 2: the adjoint run as a forward simulation
#if ABS_TYPE==1 && BACK_PROP_TYPE!=1
        {
            int i, k, ind;
            float psi;
            if (gidz>NZS-NAB-FDOH-1){
                i = gidx-FDOH;
                k = gidz-NZS+NAB+FDOH+NAB;
                ind = 2*NAB-1-k;
                psi = b_z[ind+1]*psi_sxz_z[PSIZ(k,i)] + a_z[ind+1]*dsxzz;
                psi_sxz_z[PSIZ(k,i)] = psi;
                dsxzz = dsxzz/K_z[ind+1] + psi;
                psi = b_z_half[ind]*psi_szz_z[PSIZ(k,i)] + a_z_half[ind]*dszz;
                psi_szz_z[PSIZ(k,i)] = psi;
                dszz = dszz/K_z_half[ind] + psi;
            }
#if FREESURF==0
            else if (gidz-FDOH<NAB){
                i = gidx-FDOH;
                k = gidz-FDOH;
                psi = b_z[k]*psi_sxz_z[PSIZ(k,i)] + a_z[k]*dsxzz;
                psi_sxz_z[PSIZ(k,i)] = psi;
                dsxzz = dsxzz/K_z[k] + psi;
                psi = b_z_half[k]*psi_szz_z[PSIZ(k,i)] + a_z_half[k]*dszz;
                psi_szz_z[PSIZ(k,i)] = psi;
                dszz = dszz/K_z_half[k] + psi;
            }
#endif
#if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB){
                i = gidx-FDOH;
                k = gidz-FDOH;
                psi = b_x_half[i]*psi_sxx_x[PSIX(k,i)] + a_x_half[i]*dsxx;
                psi_sxx_x[PSIX(k,i)] = psi;
                dsxx = dsxx/K_x_half[i] + psi;
                psi = b_x[i]*psi_sxz_x[PSIX(k,i)] + a_x[i]*dsxzx;
                psi_sxz_x[PSIX(k,i)] = psi;
                dsxzx = dsxzx/K_x[i] + psi;
            }
#endif
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1){
                i = gidx-NX+NAB+FDOH+NAB;
                k = gidz-FDOH;
                ind = 2*NAB-1-i;
                psi = b_x_half[ind]*psi_sxx_x[PSIX(k,i)] + a_x_half[ind]*dsxx;
                psi_sxx_x[PSIX(k,i)] = psi;
                dsxx = dsxx/K_x_half[ind] + psi;
                psi = b_x[ind+1]*psi_sxz_x[PSIX(k,i)] + a_x[ind+1]*dsxzx;
                psi_sxz_x[PSIX(k,i)] = psi;
                dsxzx = dsxzx/K_x[ind+1] + psi;
            }
#endif
        }
#endif

        // Update the adjoint velocities
        float ivx = (dsxx + dsxzz)*lrip[c];
        float ivz = (dszz + dsxzx)*lrkp[c];
        lvxr[c] += ivx;
        lvzr[c] += ivz;

        // Absorbing boundary (Cerjan)
#if ABS_TYPE==2
        {
            float tp = 1.0f;
#if FREESURF==0
            if (gidz-FDOH<NAB) tp *= taper[gidz-FDOH];
#endif
            if (gidz>NZS-NAB-FDOH-1) tp *= taper[NZS-FDOH-gidz-1];
#if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB) tp *= taper[gidx-FDOH];
#endif
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1) tp *= taper[NX-FDOH-gidx-1];
#endif
            lvxr[c] *= tp;
            lvzr[c] *= tp;
        }
#endif

        // Density gradient: forward velocity x adjoint increment
#if BACK_PROP_TYPE==1
        int sumform = 1;
#if FREESURF==1 && GRADOUT==1
        if (gidz-FDOH < FDOH) sumform = 0;            // surface_grad (direct form)
#endif
        if (sumform){
            gradrip[indp] += GSCALE(-lvx[c]*ivx);
            gradrkp[indp] += GSCALE(-lvz[c]*ivz);
        }
        /* Source term of the misfit gradient -- GJI 2017 eq. (26a) -- for a
           force source (codes 0 = Fx, 2 = Fz), in the buoyancy accumulator of
           its own staggered position: adjoint velocity field x DT*src. The
           velocities are stored scaled by 2^(src_scale - PARSCALE). */
#if GRADOUT==1
        if (nsrc>0 && sumform){
            for (int srci=0; srci<nsrc; srci++){
                int st = (int)src_pos[4+5*srci];
                if (st==0 || st==2){
                    int si = (int)(src_pos[0+5*srci]/DH)+FDOH;
                    int sk = (int)(src_pos[2+5*srci]/DH)+FDOH;
                    if (si==gidx && sk==gidz){
                        float samp = pow2s(DT*src[srci*NT+nt], src_scale - PARSCALE);
                        if (st==0) gradrip[indp] += GSCALE(lvxr[c]*samp);
                        else       gradrkp[indp] += GSCALE(lvzr[c]*samp);
                    }
                }
            }
        }
#endif
#endif
    }
    stn(vxr, indv, lvxr);
    stn(vzr, indv, lvzr);
#if BACK_PROP_TYPE==1
    stn(vx, indv, lvx);
    stn(vz, indv, lvz);
#endif
}
