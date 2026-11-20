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

/* Adjoint update of the stresses in 2D P-SV, for every precision mode (FP16 0,
   1, 2): float arithmetic, one cell at a time; storage and cells per thread come
   from header_vec.cl.
   BACK_PROP_TYPE 1 (elastic only): also rebuilds the forward stresses one step
   back (with the stored boundary values) and accumulates the moduli gradient in
   the summation-by-parts form, forward stress x adjoint increment.
   BACK_PROP_TYPE 2: the adjoint run as a forward simulation (viscoelastic
   allowed: adjoint memory variables rxxr, ...).
   Receiver cells: the residual term is added by rec_grad.cl. Free surface
   (FREESURF 1): the rows 0..SURFROWS-1 get their gradient from surface_grad
   (surface2D_adj.cl), in the direct form.
   CPML with BACK_PROP_TYPE 1: derivatives of the adjoint velocities filtered by
   cpml_adj_v (cpml_adj2D.cl), the exact adjoint. */

#ifndef SURFROWS
#define SURFROWS (FDOH+1)
#endif
#define PSIX(z,x) ((x)*(NZS-2*FDOH)+(z))
#define PSIZ(z,x) ((x)*(2*NAB)+(z))

FUNDEF void update_adjs(int offcomm,
                        GLOBARG const STO * RESTRICT vx,         GLOBARG const STO * RESTRICT vz,
                        GLOBARG STO * RESTRICT sxx,              GLOBARG STO * RESTRICT szz,           GLOBARG STO * RESTRICT sxz,
                        GLOBARG const STO * RESTRICT sxxbnd,     GLOBARG const STO * RESTRICT szzbnd,  GLOBARG const STO * RESTRICT sxzbnd,
                        GLOBARG const STO * RESTRICT vxr,        GLOBARG const STO * RESTRICT vzr,
                        GLOBARG STO * RESTRICT sxxr,             GLOBARG STO * RESTRICT szzr,          GLOBARG STO * RESTRICT sxzr,
                        GLOBARG STO * RESTRICT rxxr,             GLOBARG STO * RESTRICT rzzr,          GLOBARG STO * RESTRICT rxzr,
                        GLOBARG const STO * RESTRICT M,          GLOBARG const STO * RESTRICT mu,      GLOBARG const STO * RESTRICT muipkp,
                        GLOBARG const STO * RESTRICT taus,       GLOBARG const STO * RESTRICT tausipkp, GLOBARG const STO * RESTRICT taup,
                        GLOBARG const float * RESTRICT eta,      GLOBARG const float * RESTRICT taper,
                        GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                        GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                        GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                        GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                        GLOBARG float * RESTRICT psi_vx_x,         GLOBARG float * RESTRICT psi_vx_z,
                        GLOBARG float * RESTRICT psi_vz_x,         GLOBARG float * RESTRICT psi_vz_z,
                        GLOBARG const STO * RESTRICT cpfa,       GLOBARG const STO * RESTRICT cpfb,
                        GLOBARG const STO * RESTRICT cpfc,       GLOBARG const STO * RESTRICT cpfd,
                        GLOBARG float * RESTRICT gradM,          GLOBARG float * RESTRICT gradmu,
                        GLOBARG float * RESTRICT gradmuipkp,
                        GLOBARG float * RESTRICT HM,             GLOBARG float * RESTRICT Hmu,
                        GLOBARG const float * RESTRICT src,      GLOBARG const float * RESTRICT src_pos,
                        int nsrc,                                int nt,
                        int src_scale,                           int res_scale,
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

    // Spatial derivatives of the forward velocities (to rebuild the forward stresses)
#if BACK_PROP_TYPE==1
    float vxx[NV], vxz[NV], vzz[NV], vzx[NV];
    load_tile2(lvar, vx, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++){
        vxx[c] = tdxm(lvar, tz, lz0+c, lx);
        vxz[c] = tdzp(lvar, tz, lz0+c, lx);
    }
    BARRIER
    load_tile2(lvar, vz, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++){
        vzz[c] = tdzm(lvar, tz, lz0+c, lx);
        vzx[c] = tdxp(lvar, tz, lz0+c, lx);
    }
    BARRIER
#endif

    // Spatial derivatives of the adjoint velocities
    float vxxr[NV], vxzr[NV], vzzr[NV], vzxr[NV];
#if ABS_TYPE==1 && BACK_PROP_TYPE==1
    /* exact CPML adjoint: the fields filtered by cpml_adj_v */
    load_tile2_cpf(lvar, cpfa, vxr, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++) vxxr[c] = tdxm(lvar, tz, lz0+c, lx);
    BARRIER
    load_tile2_cpf(lvar, cpfb, vxr, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++) vxzr[c] = tdzp(lvar, tz, lz0+c, lx);
    BARRIER
    load_tile2_cpf(lvar, cpfc, vzr, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++) vzzr[c] = tdzm(lvar, tz, lz0+c, lx);
    BARRIER
    load_tile2_cpf(lvar, cpfd, vzr, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++) vzxr[c] = tdxp(lvar, tz, lz0+c, lx);
#else
    load_tile2(lvar, vxr, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++){
        vxxr[c] = tdxm(lvar, tz, lz0+c, lx);
        vxzr[c] = tdzp(lvar, tz, lz0+c, lx);
    }
    BARRIER
    load_tile2(lvar, vzr, tile_z0, tile_x0, tz, tx);
    BARRIER
    for (int c=0; c<NV; c++){
        vzzr[c] = tdzm(lvar, tz, lz0+c, lx);
        vzxr[c] = tdxp(lvar, tz, lz0+c, lx);
    }
#endif

#if COMM12==0
    if (zt > NZS-FDOH-1 || (gidx-offcomm) > (NX-FDOH-1-LCOMM)) return;
#else
    if (zt > NZS-FDOH-1) return;
#endif

    int indv = gidx*NZS + zt;
    float lsxxr[NV], lszzr[NV], lsxzr[NV];
    float lM[NV], lmu[NV], lmuipkp[NV];
    ldn(sxxr, indv, lsxxr);
    ldn(szzr, indv, lszzr);
    ldn(sxzr, indv, lsxzr);
    ldp2(M, gidx, zt, lM);
    ldp2(mu, gidx, zt, lmu);
    ldp2(muipkp, gidx, zt, lmuipkp);
#if LVE>0
    float ltaup[NV], ltaus[NV], ltausipkp[NV];
    ldp2(taup, gidx, zt, ltaup);
    ldp2(taus, gidx, zt, ltaus);
    ldp2(tausipkp, gidx, zt, ltausipkp);
#endif
#if BACK_PROP_TYPE==1
    float lsxx[NV], lszz[NV], lsxz[NV];
    ldn(sxx, indv, lsxx);
    ldn(szz, indv, lszz);
    ldn(sxz, indv, lsxz);
#endif

    for (int c=0; c<NV; c++){
        int gidz = zt + c;
        if (!INTERIOR_Z(gidz)) continue;              // halo cell of an aligned pair
        int indp = (gidx-FDOH)*(NZS-2*FDOH) + (gidz-FDOH);

#if LVE==0
        float fipkp = lmuipkp[c];
        float f = 2.0f*lmu[c];
        float g = lM[c];
#else
        float fipkp = lmuipkp[c]*(1.0f + (float)LVE*ltausipkp[c]);
        float g = lM[c]*(1.0f + (float)LVE*ltaup[c]);
        float f = 2.0f*lmu[c]*(1.0f + (float)LVE*ltaus[c]);
        float dipkp = lmuipkp[c]*ltausipkp[c]/DT;
        float d = 2.0f*lmu[c]*ltaus[c]/DT;
        float e = lM[c]*ltaup[c]/DT;
#endif

        // Rebuild the forward stresses one step back (elastic); stored cells get their stored value
#if BACK_PROP_TYPE==1
        lsxz[c] -= fipkp*(vxz[c]+vzx[c]);
        lsxx[c] -= g*(vxx[c]+vzz[c]) - f*vzz[c];
        lszz[c] -= g*(vxx[c]+vzz[c]) - f*vxx[c];
        {
            int m = inject_ind(gidz, gidx);
            if (m != -1){
                lsxx[c] = ldf(sxxbnd, m);
                lszz[c] = ldf(szzbnd, m);
                lsxz[c] = ldf(sxzbnd, m);
            }
        }
#endif

        float dvxx = vxxr[c], dvxz = vxzr[c], dvzz = vzzr[c], dvzx = vzxr[c];

        // CPML, BACK_PROP_TYPE 2: the adjoint run as a forward simulation
#if ABS_TYPE==1 && BACK_PROP_TYPE!=1
        {
            int i, k, ind;
            float psi;
            if (gidz>NZS-NAB-FDOH-1){
                i = gidx-FDOH;
                k = gidz-NZS+NAB+FDOH+NAB;
                ind = 2*NAB-1-k;
                psi = b_z_half[ind]*psi_vx_z[PSIZ(k,i)] + a_z_half[ind]*dvxz;
                psi_vx_z[PSIZ(k,i)] = psi;
                dvxz = dvxz/K_z_half[ind] + psi;
                psi = b_z[ind+1]*psi_vz_z[PSIZ(k,i)] + a_z[ind+1]*dvzz;
                psi_vz_z[PSIZ(k,i)] = psi;
                dvzz = dvzz/K_z[ind+1] + psi;
            }
#if FREESURF==0
            else if (gidz-FDOH<NAB){
                i = gidx-FDOH;
                k = gidz-FDOH;
                psi = b_z_half[k]*psi_vx_z[PSIZ(k,i)] + a_z_half[k]*dvxz;
                psi_vx_z[PSIZ(k,i)] = psi;
                dvxz = dvxz/K_z_half[k] + psi;
                psi = b_z[k]*psi_vz_z[PSIZ(k,i)] + a_z[k]*dvzz;
                psi_vz_z[PSIZ(k,i)] = psi;
                dvzz = dvzz/K_z[k] + psi;
            }
#endif
#if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB){
                i = gidx-FDOH;
                k = gidz-FDOH;
                psi = b_x[i]*psi_vx_x[PSIX(k,i)] + a_x[i]*dvxx;
                psi_vx_x[PSIX(k,i)] = psi;
                dvxx = dvxx/K_x[i] + psi;
                psi = b_x_half[i]*psi_vz_x[PSIX(k,i)] + a_x_half[i]*dvzx;
                psi_vz_x[PSIX(k,i)] = psi;
                dvzx = dvzx/K_x_half[i] + psi;
            }
#endif
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1){
                i = gidx-NX+NAB+FDOH+NAB;
                k = gidz-FDOH;
                ind = 2*NAB-1-i;
                psi = b_x[ind+1]*psi_vx_x[PSIX(k,i)] + a_x[ind+1]*dvxx;
                psi_vx_x[PSIX(k,i)] = psi;
                dvxx = dvxx/K_x[ind+1] + psi;
                psi = b_x_half[ind]*psi_vz_x[PSIX(k,i)] + a_x_half[ind]*dvzx;
                psi_vz_x[PSIX(k,i)] = psi;
                dvzx = dvzx/K_x_half[ind] + psi;
            }
#endif
        }
#endif

        // Update the adjoint stresses
#if LVE==0
        float isxz = fipkp*(dvxz+dvzx);
        float isxx = g*(dvxx+dvzz) - f*dvzz;
        float iszz = g*(dvxx+dvzz) - f*dvxx;
#else
        /* sums of the old adjoint memory variables */
        float sumrxz = 0.0f, sumrxx = 0.0f, sumrzz = 0.0f;
        for (int l=0; l<LVE; l++){
            int indr = l*NX*NZS + gidx*NZS + gidz;
            sumrxz += ldf(rxzr, indr);
            sumrxx += ldf(rxxr, indr);
            sumrzz += ldf(rzzr, indr);
        }
        float isxz = fipkp*(dvxz+dvzx) + DT2*sumrxz;
        float isxx = g*(dvxx+dvzz) - f*dvzz + DT2*sumrxx;
        float iszz = g*(dvxx+dvzz) - f*dvxx + DT2*sumrzz;
        /* memory variables (their sign changes in reverse time), new sums */
        sumrxz = sumrxx = sumrzz = 0.0f;
        for (int l=0; l<LVE; l++){
            float b = 1.0f/(1.0f + eta[l]*0.5f);
            float cc = 1.0f - eta[l]*0.5f;
            int indr = l*NX*NZS + gidx*NZS + gidz;
            float r;
            r = b*(ldf(rxzr, indr)*cc - eta[l]*(dipkp*(dvxz+dvzx)));
            stf(rxzr, indr, r); sumrxz += r;
            r = b*(ldf(rxxr, indr)*cc - eta[l]*(e*(dvxx+dvzz) - d*dvzz));
            stf(rxxr, indr, r); sumrxx += r;
            r = b*(ldf(rzzr, indr)*cc - eta[l]*(e*(dvxx+dvzz) - d*dvxx));
            stf(rzzr, indr, r); sumrzz += r;
        }
        isxz += DT2*sumrxz;
        isxx += DT2*sumrxx;
        iszz += DT2*sumrzz;
#endif
        lsxzr[c] += isxz;
        lsxxr[c] += isxx;
        lszzr[c] += iszz;

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
            lsxzr[c] *= tp;
            lsxxr[c] *= tp;
            lszzr[c] *= tp;
        }
#endif

        // Moduli gradient: forward stress x adjoint increment
#if BACK_PROP_TYPE==1
#if RESTYPE==0
        {
            float c1 = 1.0f/((2.0f*lM[c]-2.0f*lmu[c])*(2.0f*lM[c]-2.0f*lmu[c]));
            /* the shear term at the muipkp (staggered) position, kept in its
               own accumulator for average_grad_transpose() */
            float c3 = 1.0f/(fipkp*fipkp);
            float c5 = 0.25f/(lmu[c]*lmu[c]);
            float dM = c1*(lsxx[c]+lszz[c])*(isxx+iszz);
            int sumform = 1;
#if FREESURF==1
            if (gidz-FDOH < SURFROWS) sumform = 0;    // surface_grad (direct form)
#endif
            if (sumform){
                gradM[indp]      += GSCALE(-dM);
                gradmuipkp[indp] += GSCALE(-c3*(lsxz[c]*isxz));
                gradmu[indp]     += GSCALE(dM - c5*((lsxx[c]-lszz[c])*(isxx-iszz)));
            }
            /* Source term of the misfit gradient -- GJI 2017 eq. (26a) -- for a
               pressure source (code 100): adjoint stress field x DT*src. The
               stresses are stored scaled by 2^src_scale. */
#if GRADOUT==1
            if (nsrc>0 && sumform){
                for (int srci=0; srci<nsrc; srci++){
                    if ((int)src_pos[4+5*srci]==100){
                        int si = (int)(src_pos[0+5*srci]/DH)+FDOH;
                        int sk = (int)(src_pos[2+5*srci]/DH)+FDOH;
                        if (si==gidx && sk==gidz){
                            float samp = pow2s(DT*src[srci*NT+nt], src_scale);
                            float Csrc = c1*(lsxxr[c]+lszzr[c])*samp;
                            gradM[indp]  += GSCALE(Csrc);
                            gradmu[indp] += GSCALE(-Csrc);
                        }
                    }
                }
            }
#endif
#if HOUT==1
            float dMH = c1*(lsxx[c]+lszz[c])*(lsxx[c]+lszz[c]);
            HM[indp]  += GSCALE(dMH);
            Hmu[indp] += GSCALE(c3*lsxz[c]*lsxz[c] - dM + c5*(lsxx[c]-lszz[c])*(lsxx[c]-lszz[c]));
#endif
        }
#endif
#if RESTYPE==1
        {
            float fMmu = 2.0f*lM[c]-2.0f*lmu[c];
            float c1 = (fMmu!=0.0f) ? 1.0f/(fMmu*fMmu) : 0.0f;
            float dM = c1*(lsxx[c]+lszz[c])*(isxx+iszz);
            gradM[indp] += GSCALE(-dM);
            float c3 = (lmu[c]!=0.0f) ? 1.0f/(lmu[c]*lmu[c]) : 0.0f;
            float c5 = 0.25f*c3;
            gradmu[indp] += GSCALE(-c3*(lsxz[c]*isxz) + dM - c5*((lsxx[c]-lszz[c])*(isxx-iszz)));
#if HOUT==1
            HM[indp]  += GSCALE(c1*(lsxx[c]+lszz[c])*(lsxx[c]+lszz[c]));
            Hmu[indp] += GSCALE(c3*lsxz[c]*lsxz[c] - dM + c5*(lsxx[c]-lszz[c])*(lsxx[c]-lszz[c]));
#endif
        }
#endif
#endif
    }
    stn(sxxr, indv, lsxxr);
    stn(szzr, indv, lszzr);
    stn(sxzr, indv, lsxzr);
#if BACK_PROP_TYPE==1
    stn(sxx, indv, lsxx);
    stn(szz, indv, lszz);
    stn(sxz, indv, lsxz);
#endif
}
