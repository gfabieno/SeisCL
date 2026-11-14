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

/* Update of the stresses in 2D P-SV, elastic or viscoelastic (LVE mechanisms),
   for every precision mode (FP16 0, 1, 2): float arithmetic, one cell at a time;
   storage and cells per thread come from header_vec.cl. */

#define PSIX(z,x) ((x)*(NZS-2*FDOH)+(z))
#define PSIZ(z,x) ((x)*(2*NAB)+(z))

FUNDEF void update_s(int offcomm,
                     GLOBARG const STO * RESTRICT vx,        GLOBARG const STO * RESTRICT vz,
                     GLOBARG STO * RESTRICT sxx,             GLOBARG STO * RESTRICT szz,             GLOBARG STO * RESTRICT sxz,
                     GLOBARG const STO * RESTRICT M,         GLOBARG const STO * RESTRICT mu,        GLOBARG const STO * RESTRICT muipkp,
                     GLOBARG STO * RESTRICT rxx,             GLOBARG STO * RESTRICT rzz,             GLOBARG STO * RESTRICT rxz,
                     GLOBARG const STO * RESTRICT taus,      GLOBARG const STO * RESTRICT tausipkp,  GLOBARG const STO * RESTRICT taup,
                     GLOBARG const float * RESTRICT eta,     GLOBARG const float * RESTRICT taper,
                     GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                     GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                     GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                     GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                     GLOBARG STO * RESTRICT psi_vx_x,        GLOBARG STO * RESTRICT psi_vx_z,
                     GLOBARG STO * RESTRICT psi_vz_x,        GLOBARG STO * RESTRICT psi_vz_z,
                     LOCARG)
{
    LOCDEF
    int gidx = GRP1*LSZ1 + LID1 + FDOH + offcomm;
    int zt = Z0 + NV*(GRP0*LSZ0 + LID0);
    int tz = NV*LSZ0 + 2*ZH, tx = LSZ1 + 2*FDOH;
    int tile_z0 = Z0 + NV*GRP0*LSZ0 - ZH, tile_x0 = GRP1*LSZ1 + offcomm;
    int lx = LID1 + FDOH;

    float vxx[NV], vxz[NV], vzz[NV], vzx[NV];

    // Spatial derivatives of the velocities, from shared tiles
    load_tile2(lvar, vx, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++){
        vxx[c] = tdxm(lvar, tz, ZH + NV*LID0 + c, lx);
        vxz[c] = tdzp(lvar, tz, ZH + NV*LID0 + c, lx);
    }
    BARRIER
    load_tile2(lvar, vz, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++){
        vzz[c] = tdzm(lvar, tz, ZH + NV*LID0 + c, lx);
        vzx[c] = tdxp(lvar, tz, ZH + NV*LID0 + c, lx);
    }

#if COMM12==0
    if (zt > NZS-FDOH-1 || (gidx-offcomm) > (NX-FDOH-1-LCOMM)) return;
#else
    if (zt > NZS-FDOH-1) return;
#endif

    int indv = gidx*NZS + zt;
    float lsxx[NV], lszz[NV], lsxz[NV];
    float lM[NV], lmu[NV], lmuipkp[NV];
    ldn(sxx, indv, lsxx);
    ldn(szz, indv, lszz);
    ldn(sxz, indv, lsxz);
    ldp2(M, gidx, zt, lM);
    ldp2(mu, gidx, zt, lmu);
    ldp2(muipkp, gidx, zt, lmuipkp);
#if LVE>0
    float ltaup[NV], ltaus[NV], ltausipkp[NV];
    ldp2(taup, gidx, zt, ltaup);
    ldp2(taus, gidx, zt, ltaus);
    ldp2(tausipkp, gidx, zt, ltausipkp);
#endif

    for (int c=0; c<NV; c++){
        int gidz = zt + c;
        if (!INTERIOR_Z(gidz)) continue;              // halo cell of an aligned pair
        float dvxx = vxx[c], dvxz = vxz[c], dvzz = vzz[c], dvzx = vzx[c];

        // CPML: filter the derivatives in the layers
#if ABS_TYPE==1
        {
            int i, k, ind;
            float psi;
            if (gidz>NZS-NAB-FDOH-1){
                i = gidx-FDOH;
                k = gidz-NZS+NAB+FDOH+NAB;
                ind = 2*NAB-1-k;
                psi = b_z_half[ind]*ldf(psi_vx_z, PSIZ(k,i)) + a_z_half[ind]*dvxz;
                stf(psi_vx_z, PSIZ(k,i), psi);
                dvxz = dvxz/K_z_half[ind] + psi;
                psi = b_z[ind+1]*ldf(psi_vz_z, PSIZ(k,i)) + a_z[ind+1]*dvzz;
                stf(psi_vz_z, PSIZ(k,i), psi);
                dvzz = dvzz/K_z[ind+1] + psi;
            }
#if FREESURF==0
            else if (gidz-FDOH<NAB){
                i = gidx-FDOH;
                k = gidz-FDOH;
                psi = b_z_half[k]*ldf(psi_vx_z, PSIZ(k,i)) + a_z_half[k]*dvxz;
                stf(psi_vx_z, PSIZ(k,i), psi);
                dvxz = dvxz/K_z_half[k] + psi;
                psi = b_z[k]*ldf(psi_vz_z, PSIZ(k,i)) + a_z[k]*dvzz;
                stf(psi_vz_z, PSIZ(k,i), psi);
                dvzz = dvzz/K_z[k] + psi;
            }
#endif
#if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB){
                i = gidx-FDOH;
                k = gidz-FDOH;
                psi = b_x[i]*ldf(psi_vx_x, PSIX(k,i)) + a_x[i]*dvxx;
                stf(psi_vx_x, PSIX(k,i), psi);
                dvxx = dvxx/K_x[i] + psi;
                psi = b_x_half[i]*ldf(psi_vz_x, PSIX(k,i)) + a_x_half[i]*dvzx;
                stf(psi_vz_x, PSIX(k,i), psi);
                dvzx = dvzx/K_x_half[i] + psi;
            }
#endif
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1){
                i = gidx-NX+NAB+FDOH+NAB;
                k = gidz-FDOH;
                ind = 2*NAB-1-i;
                psi = b_x[ind+1]*ldf(psi_vx_x, PSIX(k,i)) + a_x[ind+1]*dvxx;
                stf(psi_vx_x, PSIX(k,i), psi);
                dvxx = dvxx/K_x[ind+1] + psi;
                psi = b_x_half[ind]*ldf(psi_vz_x, PSIX(k,i)) + a_x_half[ind]*dvzx;
                stf(psi_vz_x, PSIX(k,i), psi);
                dvzx = dvzx/K_x_half[ind] + psi;
            }
#endif
        }
#endif

        // Update the stresses
#if LVE==0
        float fipkp = lmuipkp[c];
        float f = 2.0f*lmu[c];
        float g = lM[c];
        lsxz[c] += fipkp*(dvxz+dvzx);
        lsxx[c] += g*(dvxx+dvzz) - f*dvzz;
        lszz[c] += g*(dvxx+dvzz) - f*dvxx;
#else
        float fipkp = lmuipkp[c]*(1.0f + (float)LVE*ltausipkp[c]);
        float g = lM[c]*(1.0f + (float)LVE*ltaup[c]);
        float f = 2.0f*lmu[c]*(1.0f + (float)LVE*ltaus[c]);
        float dipkp = lmuipkp[c]*ltausipkp[c]/DT;
        float d = 2.0f*lmu[c]*ltaus[c]/DT;
        float e = lM[c]*ltaup[c]/DT;
        /* sums of the old memory variables */
        float sumrxz = 0.0f, sumrxx = 0.0f, sumrzz = 0.0f;
        for (int l=0; l<LVE; l++){
            int indr = l*NX*NZS + gidx*NZS + gidz;
            sumrxz += ldf(rxz, indr);
            sumrxx += ldf(rxx, indr);
            sumrzz += ldf(rzz, indr);
        }
        /* stresses, partially */
        float usxz = fipkp*(dvxz+dvzx) + DT2*sumrxz;
        float usxx = g*(dvxx+dvzz) - f*dvzz + DT2*sumrxx;
        float uszz = g*(dvxx+dvzz) - f*dvxx + DT2*sumrzz;
        /* memory variables, and their new sums */
        sumrxz = sumrxx = sumrzz = 0.0f;
        for (int l=0; l<LVE; l++){
            float b = 1.0f/(1.0f + eta[l]*0.5f);
            float cc = 1.0f - eta[l]*0.5f;
            int indr = l*NX*NZS + gidx*NZS + gidz;
            float r;
            r = b*(ldf(rxz, indr)*cc - eta[l]*(dipkp*(dvxz+dvzx)));
            stf(rxz, indr, r); sumrxz += r;
            r = b*(ldf(rxx, indr)*cc - eta[l]*(e*(dvxx+dvzz) - d*dvzz));
            stf(rxx, indr, r); sumrxx += r;
            r = b*(ldf(rzz, indr)*cc - eta[l]*(e*(dvxx+dvzz) - d*dvxx));
            stf(rzz, indr, r); sumrzz += r;
        }
        lsxz[c] += usxz + DT2*sumrxz;
        lsxx[c] += usxx + DT2*sumrxx;
        lszz[c] += uszz + DT2*sumrzz;
#endif

        // Absorbing boundary (Cerjan)
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
        lsxx[c] *= tp;
        lszz[c] *= tp;
        lsxz[c] *= tp;
#endif
    }
    stn(sxx, indv, lsxx);
    stn(szz, indv, lszz);
    stn(sxz, indv, lsxz);
}
