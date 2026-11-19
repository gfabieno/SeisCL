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

/* Update of the stresses in 2D SH, elastic or viscoelastic (LVE mechanisms), in
   the format of update_s2D.cl (P-SV): float arithmetic, storage and cells per
   thread from header_vec.cl.
   Grid: vy at (x, z); sxy at (x+1/2, z); syz at (x, z+1/2).
   sxy += f d vy/dx, syz += f d vy/dz, with f = mu (elastic) or mu (1 + L taus),
   mu = rho vs^2 dt/dh (assign_modeling_case.c mu()).
   SH-D1: both stresses use mu and taus at the vy node (x, z); P-SV and 3-D use
   harmonic averages at the stress nodes (muipkp; muipjp, mujpkp). Kept as in the
   old SH kernel until fixed. */

#define PSIX(z,x) ((x)*(NZS-2*FDOH)+(z))
#define PSIZ(z,x) ((x)*(2*NAB)+(z))

FUNDEF void update_s(int offcomm,
                     GLOBARG const STO * RESTRICT vy,
                     GLOBARG STO * RESTRICT sxy,             GLOBARG STO * RESTRICT syz,
                     GLOBARG const STO * RESTRICT mu,
                     GLOBARG STO * RESTRICT rxy,             GLOBARG STO * RESTRICT ryz,
                     GLOBARG const STO * RESTRICT taus,
                     GLOBARG const float * RESTRICT eta,     GLOBARG const float * RESTRICT taper,
                     GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                     GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                     GLOBARG float * RESTRICT psi_vy_x,        GLOBARG float * RESTRICT psi_vy_z,
                     LOCARG)
{
    LOCDEF
    int gidx = GRP1*LSZ1 + LID1 + FDOH + offcomm;
    int zt = Z0 + NV*(GRP0*LSZ0 + LID0);
    int tz = NV*LSZ0 + 2*ZH, tx = LSZ1 + 2*FDOH;
    int tile_z0 = Z0 + NV*GRP0*LSZ0 - ZH, tile_x0 = GRP1*LSZ1 + offcomm;
    int lx = LID1 + FDOH;

    float vyx[NV], vyz[NV];

    // Spatial derivatives of the velocity, from a shared tile
    load_tile2(lvar, vy, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++){
        vyx[c] = tdxp(lvar, tz, ZH + NV*LID0 + c, lx);
        vyz[c] = tdzp(lvar, tz, ZH + NV*LID0 + c, lx);
    }

#if COMM12==0
    if (zt > NZS-FDOH-1 || (gidx-offcomm) > (NX-FDOH-1-LCOMM)) return;
#else
    if (zt > NZS-FDOH-1) return;
#endif

    int indv = gidx*NZS + zt;
    float lsxy[NV], lsyz[NV], lmu[NV];
    ldn(sxy, indv, lsxy);
    ldn(syz, indv, lsyz);
    ldp2(mu, gidx, zt, lmu);
#if LVE>0
    float ltaus[NV];
    ldp2(taus, gidx, zt, ltaus);
#endif

    for (int c=0; c<NV; c++){
        int gidz = zt + c;
        if (!INTERIOR_Z(gidz)) continue;              // halo cell of an aligned pair
        float dvyx = vyx[c], dvyz = vyz[c];

        // CPML: filter the derivatives in the layers
#if ABS_TYPE==1
        {
            int i, k, ind;
            float psi;
            if (gidz>NZS-NAB-FDOH-1){
                i = gidx-FDOH;
                k = gidz-NZS+NAB+FDOH+NAB;
                ind = 2*NAB-1-k;
                psi = b_z_half[ind]*psi_vy_z[PSIZ(k,i)] + a_z_half[ind]*dvyz;
                psi_vy_z[PSIZ(k,i)] = psi;
                dvyz = dvyz/K_z_half[ind] + psi;
            }
#if FREESURF==0
            else if (gidz-FDOH<NAB){
                i = gidx-FDOH;
                k = gidz-FDOH;
                psi = b_z_half[k]*psi_vy_z[PSIZ(k,i)] + a_z_half[k]*dvyz;
                psi_vy_z[PSIZ(k,i)] = psi;
                dvyz = dvyz/K_z_half[k] + psi;
            }
#endif
#if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB){
                i = gidx-FDOH;
                k = gidz-FDOH;
                psi = b_x_half[i]*psi_vy_x[PSIX(k,i)] + a_x_half[i]*dvyx;
                psi_vy_x[PSIX(k,i)] = psi;
                dvyx = dvyx/K_x_half[i] + psi;
            }
#endif
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1){
                i = gidx-NX+NAB+FDOH+NAB;
                k = gidz-FDOH;
                ind = 2*NAB-1-i;
                psi = b_x_half[ind]*psi_vy_x[PSIX(k,i)] + a_x_half[ind]*dvyx;
                psi_vy_x[PSIX(k,i)] = psi;
                dvyx = dvyx/K_x_half[ind] + psi;
            }
#endif
        }
#endif

        // Update the stresses
#if LVE==0
        float f = lmu[c];
        lsxy[c] += f*dvyx;
        lsyz[c] += f*dvyz;
#else
        float f = lmu[c]*(1.0f + (float)LVE*ltaus[c]);
        float d = lmu[c]*ltaus[c]/DT;
        /* sums of the old memory variables */
        float sumrxy = 0.0f, sumryz = 0.0f;
        for (int l=0; l<LVE; l++){
            int indr = l*NX*NZS + gidx*NZS + gidz;
            sumrxy += ldf(rxy, indr);
            sumryz += ldf(ryz, indr);
        }
        /* stresses, partially */
        float usxy = f*dvyx + DT2*sumrxy;
        float usyz = f*dvyz + DT2*sumryz;
        /* memory variables, and their new sums */
        sumrxy = sumryz = 0.0f;
        for (int l=0; l<LVE; l++){
            float b = 1.0f/(1.0f + eta[l]*0.5f);
            float cc = 1.0f - eta[l]*0.5f;
            int indr = l*NX*NZS + gidx*NZS + gidz;
            float r;
            r = b*(ldf(rxy, indr)*cc - eta[l]*(d*dvyx));
            stf(rxy, indr, r); sumrxy += r;
            r = b*(ldf(ryz, indr)*cc - eta[l]*(d*dvyz));
            stf(ryz, indr, r); sumryz += r;
        }
        lsxy[c] += usxy + DT2*sumrxy;
        lsyz[c] += usyz + DT2*sumryz;
#endif

        /* SH-D2: no Cerjan taper on the stresses. The old SH kernel tested
           "#if abstype==2" (lower case, never defined), so only vy was tapered;
           P-SV tapers velocities and stresses. Kept as in the old kernel until fixed. */
    }
    stn(sxy, indv, lsxy);
    stn(syz, indv, lsyz);
}
