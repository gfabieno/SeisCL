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

/* Adjoint update of the stresses in 2D SH, back_prop_type 2, elastic or
   viscoelastic, in the format of update_adjs2D.cl (P-SV).
   Obtained from update_adjs3D.cl's back_prop_type 2 path by dropping the terms
   that are zero by symmetry (nothing varies along y; vx = vz = 0):
   3-D  vxyyxr = vxyr + vyxr -> vyxr,  vyzzyr = vyzr + vzyr -> vyzr; the sxyr and
   syzr updates, with fipjp = muipjp(1 + L tausipjp), fjpkp = mujpkp(1 + L
   tausjpkp), dipjp = muipjp tausipjp/DT, djpkp = mujpkp tausjpkp/DT, are kept
   unchanged; sxxr, syyr, szzr, sxzr and their memory variables stay 0.
   The adjoint memory variables rxyr, ryzr are read and written through ONE
   pointer and summed from registers (pitfall A23: 3-D read the new value
   through an aliased const RESTRICT pointer). Thesis eq. 3.33 (SH form in
   notes sh-gradient.md).
   back_prop_type 1 is not implemented for SH (refused by assign_modeling_case.c). */

#if BACK_PROP_TYPE==1
#error "SH (ND 21): back_prop_type 1 adjoint not implemented"
#endif

#define PSIX(z,x) ((x)*(NZS-2*FDOH)+(z))
#define PSIZ(z,x) ((x)*(2*NAB)+(z))

FUNDEF void update_adjs(int offcomm,
                     GLOBARG const STO * RESTRICT vyr,
                     GLOBARG STO * RESTRICT sxyr,            GLOBARG STO * RESTRICT syzr,
                     GLOBARG const STO * RESTRICT muipjp,    GLOBARG const STO * RESTRICT mujpkp,
                     GLOBARG STO * RESTRICT rxyr,            GLOBARG STO * RESTRICT ryzr,
                     GLOBARG const STO * RESTRICT tausipjp,  GLOBARG const STO * RESTRICT tausjpkp,
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
    load_tile2(lvar, vyr, tile_z0, tile_x0, tz, tx);
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
    float lsxy[NV], lsyz[NV], lmuipjp[NV], lmujpkp[NV];
    ldn(sxyr, indv, lsxy);
    ldn(syzr, indv, lsyz);
    ldp2(muipjp, gidx, zt, lmuipjp);
    ldp2(mujpkp, gidx, zt, lmujpkp);
#if LVE>0
    float ltausipjp[NV], ltausjpkp[NV];
    ldp2(tausipjp, gidx, zt, ltausipjp);
    ldp2(tausjpkp, gidx, zt, ltausjpkp);
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

        // Update the adjoint stresses
#if LVE==0
        lsxy[c] += lmuipjp[c]*dvyx;
        lsyz[c] += lmujpkp[c]*dvyz;
#else
        float fipjp = lmuipjp[c]*(1.0f + (float)LVE*ltausipjp[c]);
        float fjpkp = lmujpkp[c]*(1.0f + (float)LVE*ltausjpkp[c]);
        float dipjp = lmuipjp[c]*ltausipjp[c]/DT;
        float djpkp = lmujpkp[c]*ltausjpkp[c]/DT;
        /* sums of the old memory variables */
        float sumrxy = 0.0f, sumryz = 0.0f;
        for (int l=0; l<LVE; l++){
            int indr = l*NX*NZS + gidx*NZS + gidz;
            sumrxy += ldf(rxyr, indr);
            sumryz += ldf(ryzr, indr);
        }
        /* stresses, partially */
        float usxy = fipjp*dvyx + DT2*sumrxy;
        float usyz = fjpkp*dvyz + DT2*sumryz;
        /* memory variables, and their new sums */
        sumrxy = sumryz = 0.0f;
        for (int l=0; l<LVE; l++){
            float b = 1.0f/(1.0f + eta[l]*0.5f);
            float cc = 1.0f - eta[l]*0.5f;
            int indr = l*NX*NZS + gidx*NZS + gidz;
            float r;
            r = b*(ldf(rxyr, indr)*cc - eta[l]*(dipjp*dvyx));
            stf(rxyr, indr, r); sumrxy += r;
            r = b*(ldf(ryzr, indr)*cc - eta[l]*(djpkp*dvyz));
            stf(ryzr, indr, r); sumryz += r;
        }
        lsxy[c] += usxy + DT2*sumrxy;
        lsyz[c] += usyz + DT2*sumryz;
#endif

        // Absorbing boundary (Cerjan). Until 2026-11-19 the SH stresses were not
        // tapered (the old kernel tested "#if abstype==2", never defined; SH-D2).
#if ABS_TYPE==2
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
        lsxy[c] *= tp;
        lsyz[c] *= tp;
#endif
    }
    stn(sxyr, indv, lsxy);
    stn(syzr, indv, lsyz);
}
