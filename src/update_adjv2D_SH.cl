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

/* Adjoint update of the velocity in 2D SH, back_prop_type 2, in the format of
   update_adjv2D.cl (P-SV).
   Obtained from update_adjv3D.cl by dropping the terms that are zero by
   symmetry (nothing varies along y; vx = vz = 0, sxx = syy = szz = sxz = 0):
   3-D  vyr += (syy_yr + sxy_xr + syz_zr) * rjp  ->  SH  vyr += (sxy_xr + syz_zr) * rho,
   rjp being rho when nothing varies along y. The CPML and Cerjan treatment are
   those of the 3-D back_prop_type 2 path ("the adjoint run as a forward
   simulation"), i.e. those of update_v2D_SH.cl. Thesis eq. 3.33 (SH form in
   notes sh-gradient.md). Under back_prop_type 2 every <var>r argument is bound
   to the same buffer as <var> (clprogram.c).
   BACK_PROP_TYPE 1 (elastic), as update_adjv2D.cl: also rebuilds the forward
   velocity one step back (stored frame values injected) and accumulates the
   density gradient in the summation-by-parts form, forward velocity x adjoint
   increment, in gradrho (vy sits at rjp = rho: no average). Receiver cells: the
   residual term is added by rec_grad_SH.cl. CPML: the derivatives of the adjoint
   stresses are taken of the fields filtered by cpml_adj_s (cpml_adj2D_SH.cl),
   the exact adjoint. */

#define PSIX(z,x) ((x)*(NZS-2*FDOH)+(z))
#define PSIZ(z,x) ((x)*(2*NAB)+(z))

FUNDEF void update_adjv(int offcomm,
                     GLOBARG STO * RESTRICT vy,
                     GLOBARG const STO * RESTRICT sxy,       GLOBARG const STO * RESTRICT syz,
                     GLOBARG const STO * RESTRICT vybnd,
                     GLOBARG STO * RESTRICT vyr,
                     GLOBARG const STO * RESTRICT sxyr,      GLOBARG const STO * RESTRICT syzr,
                     GLOBARG const STO * RESTRICT rho,
                     GLOBARG const float * RESTRICT taper,
                     GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                     GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                     GLOBARG float * RESTRICT psi_sxy_x,       GLOBARG float * RESTRICT psi_syz_z,
                     GLOBARG const STO * RESTRICT cpfa,      GLOBARG const STO * RESTRICT cpfb,
                     GLOBARG float * RESTRICT gradrho,
                     GLOBARG const float * RESTRICT src,     GLOBARG const float * RESTRICT src_pos,
                     int nsrc,                               int nt,
                     int src_scale,                          int res_scale,
                     int par_scale, LOCARG)
{
    LOCDEF
    /* This thread: column gidx, cells z = zt .. zt+NV-1 (padded indices). */
    int gidx = GRP1*LSZ1 + LID1 + FDOH + offcomm;
    int zt = Z0 + NV*(GRP0*LSZ0 + LID0);
    /* Shared tile: columns gidx-FDOH .. of the work-group, z from zt0-ZH. */
    int tz = NV*LSZ0 + 2*ZH, tx = LSZ1 + 2*FDOH;
    int tile_z0 = Z0 + NV*GRP0*LSZ0 - ZH, tile_x0 = GRP1*LSZ1 + offcomm;
    int lx = LID1 + FDOH;
    int lz0 = ZH + NV*LID0;
#if NOTILE
    /* no shared tiles (header_vec.cl): the derivatives read the arrays in place, with the global
       indices; threads past the grid return before reading. */
#if COMM12==0
    if (zt > NZS-FDOH-1 || (gidx-offcomm) > (NX-FDOH-1-LCOMM)) return;
#else
    if (zt > NZS-FDOH-1) return;
#endif
    tz = NZS; lx = gidx; lz0 = zt;
#endif

    float sxy_x[NV], syz_z[NV];

    // Spatial derivatives of the forward stresses (to rebuild the forward velocity)
#if BACK_PROP_TYPE==1
    float fsxy_x[NV], fsyz_z[NV];
    load_tile2(lvar, sxy, tile_z0, tile_x0, tz, tx);
    BARRIER
    tdmn(lvar, lx*tz + lz0, tz, fsxy_x);
    BARRIER
    load_tile2(lvar, syz, tile_z0, tile_x0, tz, tx);
    BARRIER
    tdmn(lvar, lx*tz + lz0, 1, fsyz_z);
    BARRIER
#endif

    // Spatial derivatives of the adjoint stresses, from shared tiles
#if ABS_TYPE==1 && BACK_PROP_TYPE==1
    /* exact CPML adjoint: the fields filtered by cpml_adj_s */
    load_tile2(lvar, cpfa, tile_z0, tile_x0, tz, tx);
    BARRIER
    tdmn(lvar, lx*tz + lz0, tz, sxy_x);
    BARRIER
    load_tile2(lvar, cpfb, tile_z0, tile_x0, tz, tx);
    BARRIER
    tdmn(lvar, lx*tz + lz0, 1, syz_z);
#else
    load_tile2(lvar, sxyr, tile_z0, tile_x0, tz, tx);
    BARRIER
    tdmn(lvar, lx*tz + lz0, tz, sxy_x);
    BARRIER
    load_tile2(lvar, syzr, tile_z0, tile_x0, tz, tx);
    BARRIER
    tdmn(lvar, lx*tz + lz0, 1, syz_z);
#endif

    // Threads outside the grid stop here (the grid is rounded up to the work-group size)
#if COMM12==0
    if (zt > NZS-FDOH-1 || (gidx-offcomm) > (NX-FDOH-1-LCOMM)) return;
#else
    if (zt > NZS-FDOH-1) return;
#endif

    int indv = gidx*NZS + zt;
    float lvy[NV], lrho[NV];
    ldn(vyr, indv, lvy);
    ldp2(rho, gidx, zt, lrho);
#if BACK_PROP_TYPE==1
    float lvyf[NV];
    ldn(vy, indv, lvyf);
#endif

    #pragma unroll
    for (int c=0; c<NV; c++){
        int gidz = zt + c;
        if (!INTERIOR_Z(gidz)) continue;              // halo cell of an aligned pair
        float dsxy = sxy_x[c], dsyz = syz_z[c];

        // Rebuild the forward velocity one step back; stored cells get their stored value
#if BACK_PROP_TYPE==1
        int indp = (gidx-FDOH)*(NZS-2*FDOH) + (gidz-FDOH);
        lvyf[c] -= (fsxy_x[c] + fsyz_z[c])*lrho[c];
        {
            int m = inject_ind(gidz, gidx);
            if (m != -1) lvyf[c] = ldf(vybnd, m);
        }
#endif

        // CPML, BACK_PROP_TYPE 2: the adjoint run as a forward simulation
#if ABS_TYPE==1 && BACK_PROP_TYPE!=1
        {
            int i, k, ind;
            float psi;
            if (gidz>NZS-NAB-FDOH-1){
                i = gidx-FDOH;
                k = gidz-NZS+NAB+FDOH+NAB;
                ind = 2*NAB-1-k;
                psi = b_z[ind+1]*psi_syz_z[PSIZ(k,i)] + a_z[ind+1]*dsyz;
                psi_syz_z[PSIZ(k,i)] = psi;
                dsyz = dsyz/K_z[ind+1] + psi;
            }
#if FREESURF==0
            else if (gidz-FDOH<NAB){
                i = gidx-FDOH;
                k = gidz-FDOH;
                psi = b_z[k]*psi_syz_z[PSIZ(k,i)] + a_z[k]*dsyz;
                psi_syz_z[PSIZ(k,i)] = psi;
                dsyz = dsyz/K_z[k] + psi;
            }
#endif
#if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB){
                i = gidx-FDOH;
                k = gidz-FDOH;
                psi = b_x[i]*psi_sxy_x[PSIX(k,i)] + a_x[i]*dsxy;
                psi_sxy_x[PSIX(k,i)] = psi;
                dsxy = dsxy/K_x[i] + psi;
            }
#endif
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1){
                i = gidx-NX+NAB+FDOH+NAB;
                k = gidz-FDOH;
                ind = 2*NAB-1-i;
                /* ind+1, as P-SV for the same (integer-node) x derivative (update_v2D.cl,
                   dsxzx); the old SH kernel used ind (SH-D3, fixed 2026-11-19). */
                psi = b_x[ind+1]*psi_sxy_x[PSIX(k,i)] + a_x[ind+1]*dsxy;
                psi_sxy_x[PSIX(k,i)] = psi;
                dsxy = dsxy/K_x[ind+1] + psi;
            }
#endif
        }
#endif

        // Update the adjoint velocity
        float ivy = (dsxy + dsyz)*lrho[c];
        lvy[c] += ivy;

        // Absorbing boundary (Cerjan)
#if ABS_TYPE==2
#if FREESURF==0
        if (gidz-FDOH<NAB) lvy[c] *= taper[gidz-FDOH];
#endif
        if (gidz>NZS-NAB-FDOH-1) lvy[c] *= taper[NZS-FDOH-gidz-1];
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB) lvy[c] *= taper[gidx-FDOH];
#endif
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1) lvy[c] *= taper[NX-FDOH-gidx-1];
#endif
#endif

        // Density gradient: forward velocity x adjoint increment (update_adjv2D.cl)
#if BACK_PROP_TYPE==1
        gradrho[indp] += GSCALE(-lvyf[c]*ivy);
        /* Source term of the misfit gradient, GJI 2017 eq. (26a), for a force
           source Fy (code 1): adjoint velocity x DT*src. */
#if GRADOUT==1
        for (int srci=0; srci<nsrc; srci++){
            if ((int)src_pos[4+5*srci]==1){
                int si = (int)(src_pos[0+5*srci]/DH)+FDOH-OFFSET;   /* local x (domain decomposition) */
                int sk = (int)(src_pos[2+5*srci]/DH)+FDOH;
                if (si==gidx && sk==gidz){
                    float samp = pow2s(DT*src[srci*NT+nt], src_scale - PARSCALE);
                    gradrho[indp] += GSCALE(lvy[c]*samp);
                }
            }
        }
#endif
#endif
    }
    stn(vyr, indv, lvy);
#if BACK_PROP_TYPE==1
    stn(vy, indv, lvyf);
#endif
}
