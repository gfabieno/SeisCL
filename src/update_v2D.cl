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

/* Update of the velocities in 2D P-SV, for every precision mode (FP16 0, 1, 2):
   float arithmetic, one cell at a time; storage and cells per thread come from
   header_vec.cl (included before this file). */

/* CPML memory variables: interior z index k, x index i (x layers: 0..2NAB-1). */
#define PSIX(z,x) ((x)*(NZS-2*FDOH)+(z))
#define PSIZ(z,x) ((x)*(2*NAB)+(z))

FUNDEF void update_v(int offcomm,
                     GLOBARG STO * RESTRICT vx,              GLOBARG STO * RESTRICT vz,
                     GLOBARG const STO * RESTRICT sxx,       GLOBARG const STO * RESTRICT szz,
                     GLOBARG const STO * RESTRICT sxz,
                     GLOBARG const STO * RESTRICT rip,       GLOBARG const STO * RESTRICT rkp,
                     GLOBARG const float * RESTRICT taper,
                     GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                     GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                     GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                     GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                     GLOBARG float * RESTRICT psi_sxx_x,       GLOBARG float * RESTRICT psi_sxz_x,
                     GLOBARG float * RESTRICT psi_sxz_z,       GLOBARG float * RESTRICT psi_szz_z,
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

    float sxx_x[NV], szz_z[NV], sxz_x[NV], sxz_z[NV];

    // Spatial derivatives of the stresses, from shared tiles
    load_tile2(lvar, sxx, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++) sxx_x[c] = tdxp(lvar, tz, ZH + NV*LID0 + c, lx);
    BARRIER
    load_tile2(lvar, szz, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++) szz_z[c] = tdzp(lvar, tz, ZH + NV*LID0 + c, lx);
    BARRIER
    load_tile2(lvar, sxz, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++){
        sxz_z[c] = tdzm(lvar, tz, ZH + NV*LID0 + c, lx);
        sxz_x[c] = tdxm(lvar, tz, ZH + NV*LID0 + c, lx);
    }

    // Threads outside the grid stop here (the grid is rounded up to the work-group size)
#if COMM12==0
    if (zt > NZS-FDOH-1 || (gidx-offcomm) > (NX-FDOH-1-LCOMM)) return;
#else
    if (zt > NZS-FDOH-1) return;
#endif

    int indv = gidx*NZS + zt;
    float lvx[NV], lvz[NV], lrip[NV], lrkp[NV];
    ldn(vx, indv, lvx);
    ldn(vz, indv, lvz);
    ldp2(rip, gidx, zt, lrip);
    ldp2(rkp, gidx, zt, lrkp);

    for (int c=0; c<NV; c++){
        int gidz = zt + c;
        if (!INTERIOR_Z(gidz)) continue;              // halo cell of an aligned pair
        float dsxx = sxx_x[c], dszz = szz_z[c], dsxzz = sxz_z[c], dsxzx = sxz_x[c];

        // CPML: filter the derivatives in the layers
#if ABS_TYPE==1
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

        // Update the velocities
        lvx[c] += (dsxx + dsxzz)*lrip[c];
        lvz[c] += (dszz + dsxzx)*lrkp[c];

        // Absorbing boundary (Cerjan)
#if ABS_TYPE==2
#if FREESURF==0
        if (gidz-FDOH<NAB){
            lvx[c] *= taper[gidz-FDOH];
            lvz[c] *= taper[gidz-FDOH];
        }
#endif
        if (gidz>NZS-NAB-FDOH-1){
            lvx[c] *= taper[NZS-FDOH-gidz-1];
            lvz[c] *= taper[NZS-FDOH-gidz-1];
        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            lvx[c] *= taper[gidx-FDOH];
            lvz[c] *= taper[gidx-FDOH];
        }
#endif
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            lvx[c] *= taper[NX-FDOH-gidx-1];
            lvz[c] *= taper[NX-FDOH-gidx-1];
        }
#endif
#endif
    }
    stn(vx, indv, lvx);
    stn(vz, indv, lvz);
}
