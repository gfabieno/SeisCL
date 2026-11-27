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

/* Update of the velocity in 2D SH (vy, out of the x-z plane), in the format of
   update_v2D.cl (P-SV): float arithmetic, storage and cells per thread from
   header_vec.cl (included before this file).
   Grid: vy at (x, z); sxy at (x+1/2, z); syz at (x, z+1/2).
   vy += rho * (d sxy/dx + d syz/dz), rho = dt/(dh rho) (assign_modeling_case.c rho()). */

/* CPML memory variables: interior z index k, x index i (x layers: 0..2NAB-1). */
#define PSIX(z,x) ((x)*(NZS-2*FDOH)+(z))
#define PSIZ(z,x) ((x)*(2*NAB)+(z))

FUNDEF void update_v(int offcomm,
                     GLOBARG STO * RESTRICT vy,
                     GLOBARG const STO * RESTRICT sxy,       GLOBARG const STO * RESTRICT syz,
                     GLOBARG const STO * RESTRICT rho,
                     GLOBARG const float * RESTRICT taper,
                     GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                     GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                     GLOBARG float * RESTRICT psi_sxy_x,       GLOBARG float * RESTRICT psi_syz_z,
                     LOCARG)
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

    // Spatial derivatives of the stresses, from shared tiles
    load_tile2(lvar, sxy, tile_z0, tile_x0, tz, tx);
    BARRIER
    tdmn(lvar, lx*tz + lz0, tz, sxy_x);
    BARRIER
    load_tile2(lvar, syz, tile_z0, tile_x0, tz, tx);
    BARRIER
    tdmn(lvar, lx*tz + lz0, 1, syz_z);

    // Threads outside the grid stop here (the grid is rounded up to the work-group size)
#if COMM12==0
    if (zt > NZS-FDOH-1 || (gidx-offcomm) > (NX-FDOH-1-LCOMM)) return;
#else
    if (zt > NZS-FDOH-1) return;
#endif

    int indv = gidx*NZS + zt;
    float lvy[NV], lrho[NV];
    ldn(vy, indv, lvy);
    ldp2(rho, gidx, zt, lrho);

    for (int c=0; c<NV; c++){
        int gidz = zt + c;
        if (!INTERIOR_Z(gidz)) continue;              // halo cell of an aligned pair
        float dsxy = sxy_x[c], dsyz = syz_z[c];

        // CPML: filter the derivatives in the layers
#if ABS_TYPE==1
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

        // Update the velocity
        lvy[c] += (dsxy + dsyz)*lrho[c];

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
    }
    stn(vy, indv, lvy);
}
