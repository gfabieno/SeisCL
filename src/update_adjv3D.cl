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

/*Adjoint update of the velocities in 3D SV*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */

/* Generated from the FP32 3-D kernel by torch-robust/unify/gen3d.py: unified storage (header_vec.cl),
   per-cell physics unchanged. */
#define I_psi_sxx_x(z,y,x) ((x)*(NYS-2*FDOH)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_sxy_x(z,y,x) ((x)*(NYS-2*FDOH)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_sxz_x(z,y,x) ((x)*(NYS-2*FDOH)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_sxy_y(z,y,x) ((x)*(2*NAB)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_syy_y(z,y,x) ((x)*(2*NAB)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_syz_y(z,y,x) ((x)*(2*NAB)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_sxz_z(z,y,x) ((x)*(NYS-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z))
#define I_psi_syz_z(z,y,x) ((x)*(NYS-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z))
#define I_psi_szz_z(z,y,x) ((x)*(NYS-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z))

FUNDEF void update_adjv(int offcomm,
                          GLOBARG STO * RESTRICT vx,              GLOBARG STO * RESTRICT vy,              GLOBARG STO * RESTRICT vz,
                          GLOBARG const STO * RESTRICT sxx,       GLOBARG const STO * RESTRICT syy,       GLOBARG const STO * RESTRICT szz,
                          GLOBARG const STO * RESTRICT sxy,       GLOBARG const STO * RESTRICT syz,       GLOBARG const STO * RESTRICT sxz,
                          GLOBARG const STO * RESTRICT vxbnd,     GLOBARG const STO * RESTRICT vybnd,     GLOBARG const STO * RESTRICT vzbnd,
                          GLOBARG const STO * RESTRICT sxxbnd,    GLOBARG const STO * RESTRICT syybnd,    GLOBARG const STO * RESTRICT szzbnd,
                          GLOBARG const STO * RESTRICT sxybnd,    GLOBARG const STO * RESTRICT syzbnd,    GLOBARG const STO * RESTRICT sxzbnd,
                          GLOBARG STO * RESTRICT vxr,             GLOBARG STO * RESTRICT vyr,             GLOBARG STO * RESTRICT vzr,
                          GLOBARG const STO * RESTRICT sxxr,      GLOBARG const STO * RESTRICT syyr,      GLOBARG const STO * RESTRICT szzr,
                          GLOBARG const STO * RESTRICT sxyr,      GLOBARG const STO * RESTRICT syzr,      GLOBARG const STO * RESTRICT sxzr,
                          GLOBARG const STO * RESTRICT rip,       GLOBARG const STO * RESTRICT rjp,       GLOBARG const STO * RESTRICT rkp,
                          GLOBARG const float * RESTRICT taper,
                          GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                          GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                          GLOBARG const float * RESTRICT K_y,       GLOBARG const float * RESTRICT a_y,       GLOBARG const float * RESTRICT b_y,
                          GLOBARG const float * RESTRICT K_y_half,  GLOBARG const float * RESTRICT a_y_half,  GLOBARG const float * RESTRICT b_y_half,
                          GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                          GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                          GLOBARG STO * RESTRICT psi_sxx_x,       GLOBARG STO * RESTRICT psi_sxy_x,       GLOBARG STO * RESTRICT psi_sxy_y,
                          GLOBARG STO * RESTRICT psi_sxz_x,       GLOBARG STO * RESTRICT psi_sxz_z,       GLOBARG STO * RESTRICT psi_syy_y,
                          GLOBARG STO * RESTRICT psi_syz_y,       GLOBARG STO * RESTRICT psi_syz_z,       GLOBARG STO * RESTRICT psi_szz_z,
                          GLOBARG const STO * RESTRICT cpf1,
                          GLOBARG const STO * RESTRICT cpf2,
                          GLOBARG const STO * RESTRICT cpf3,
                          GLOBARG const STO * RESTRICT cpf4,
                          GLOBARG const STO * RESTRICT cpf5,
                          GLOBARG const STO * RESTRICT cpf6,
                          GLOBARG const STO * RESTRICT cpf7,
                          GLOBARG const STO * RESTRICT cpf8,
                          GLOBARG const STO * RESTRICT cpf9,
                          int src_scale, int res_scale,
                LOCARG,                                   GLOBARG float * RESTRICT gradrho,         GLOBARG const float * RESTRICT gradsrc,
                          GLOBARG float * RESTRICT gradrip,         GLOBARG float * RESTRICT gradrjp,         GLOBARG float * RESTRICT gradrkp,
                          GLOBARG const float * RESTRICT Hrho,      GLOBARG const float * RESTRICT Hsrc,
                          GLOBARG const float * RESTRICT src,       GLOBARG const float * RESTRICT src_pos,
                          int nsrc,                                 int nt,
                          GLOBARG const float * RESTRICT vxout,     GLOBARG const float * RESTRICT vyout,
                          GLOBARG const float * RESTRICT vzout,     GLOBARG const float * RESTRICT rec_pos,
                          int nrec)
{
    LOCDEF

    
    int m;
    float lvx, lvy, lvz;
    float fvx = 0.0f, fvy = 0.0f, fvz = 0.0f;   // forward velocity increment of this step (BACK_PROP_TYPE 1)

// If we use local memory

    int gidy = GRP1*LSZ1 + LID1 + FDOH;
    int gidx = GRP2*LSZ2 + LID2 + FDOH + offcomm;
    int zt = Z0 + NV*(GRP0*LSZ0 + LID0);
    int tz = NV*LSZ0 + 2*ZH, ty = LSZ1 + 2*FDOH, tx = LSZ2 + 2*FDOH;
    int tile_z0 = Z0 + NV*GRP0*LSZ0 - ZH, tile_y0 = GRP1*LSZ1, tile_x0 = GRP2*LSZ2 + offcomm;
    int it0 = ((LID2+FDOH)*ty + (LID1+FDOH))*tz + ZH + NV*LID0;
    int sy = tz, sx = tz*ty;
    float d_sxx_x[NV], d_sxx_xr[NV], d_sxy_x[NV], d_sxy_xr[NV], d_sxy_y[NV], d_sxy_yr[NV], d_sxz_x[NV], d_sxz_xr[NV], d_sxz_z[NV], d_sxz_zr[NV], d_syy_y[NV], d_syy_yr[NV], d_syz_y[NV], d_syz_yr[NV], d_syz_z[NV], d_syz_zr[NV], d_szz_z[NV], d_szz_zr[NV];
#if BACK_PROP_TYPE==1
    load_tile3(lvar, szz, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_szz_z[c] = tdp(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, sxx, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_sxx_x[c] = tdp(lvar, it0+c, sx);
    BARRIER
    load_tile3(lvar, sxz, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_sxz_x[c] = tdm(lvar, it0+c, sx);
    for (int c=0; c<NV; c++) d_sxz_z[c] = tdm(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, syz, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_syz_y[c] = tdm(lvar, it0+c, sy);
    for (int c=0; c<NV; c++) d_syz_z[c] = tdm(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, syy, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_syy_y[c] = tdp(lvar, it0+c, sy);
    BARRIER
    load_tile3(lvar, sxy, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_sxy_x[c] = tdm(lvar, it0+c, sx);
    for (int c=0; c<NV; c++) d_sxy_y[c] = tdm(lvar, it0+c, sy);
    BARRIER
#endif
#if ABS_TYPE==1 && BACK_PROP_TYPE==1
    load_tile3(lvar, cpf1, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_sxx_xr[c] = tdp(lvar, it0+c, sx);
    BARRIER
    load_tile3(lvar, cpf2, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_sxy_yr[c] = tdm(lvar, it0+c, sy);
    BARRIER
    load_tile3(lvar, cpf3, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_sxz_zr[c] = tdm(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, cpf4, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_sxy_xr[c] = tdm(lvar, it0+c, sx);
    BARRIER
    load_tile3(lvar, cpf5, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_syy_yr[c] = tdp(lvar, it0+c, sy);
    BARRIER
    load_tile3(lvar, cpf6, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_syz_zr[c] = tdm(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, cpf7, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_sxz_xr[c] = tdm(lvar, it0+c, sx);
    BARRIER
    load_tile3(lvar, cpf8, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_syz_yr[c] = tdm(lvar, it0+c, sy);
    BARRIER
    load_tile3(lvar, cpf9, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_szz_zr[c] = tdp(lvar, it0+c, 1);
    BARRIER
#else
    load_tile3(lvar, szzr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_szz_zr[c] = tdp(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, sxxr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_sxx_xr[c] = tdp(lvar, it0+c, sx);
    BARRIER
    load_tile3(lvar, sxzr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_sxz_xr[c] = tdm(lvar, it0+c, sx);
    for (int c=0; c<NV; c++) d_sxz_zr[c] = tdm(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, syzr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_syz_yr[c] = tdm(lvar, it0+c, sy);
    for (int c=0; c<NV; c++) d_syz_zr[c] = tdm(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, syyr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_syy_yr[c] = tdp(lvar, it0+c, sy);
    BARRIER
    load_tile3(lvar, sxyr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_sxy_xr[c] = tdm(lvar, it0+c, sx);
    for (int c=0; c<NV; c++) d_sxy_yr[c] = tdm(lvar, it0+c, sy);
    BARRIER
#endif

#if COMM12==0
    if (zt > NZS-FDOH-1 || gidy > NYS-FDOH-1 || (gidx-offcomm) > (NX-FDOH-1-LCOMM)) return;
#else
    if (zt > NZS-FDOH-1 || gidy > NYS-FDOH-1) return;
#endif

    for (int c=0; c<NV; c++){
        int gidz = zt + c;
        if (!INTERIOR_Z(gidz)) continue;              // halo cell of an aligned pair
        int indp = ((gidx-FDOH)*(NYS-2*FDOH) + (gidy-FDOH))*(NZS-2*FDOH) + (gidz-FDOH);
        int indv = (gidx*NYS + gidy)*NZS + gidz;
        float sxx_x = d_sxx_x[c];
        float sxx_xr = d_sxx_xr[c];
        float sxy_x = d_sxy_x[c];
        float sxy_xr = d_sxy_xr[c];
        float sxy_y = d_sxy_y[c];
        float sxy_yr = d_sxy_yr[c];
        float sxz_x = d_sxz_x[c];
        float sxz_xr = d_sxz_xr[c];
        float sxz_z = d_sxz_z[c];
        float sxz_zr = d_sxz_zr[c];
        float syy_y = d_syy_y[c];
        float syy_yr = d_syy_yr[c];
        float syz_y = d_syz_y[c];
        float syz_yr = d_syz_yr[c];
        float syz_z = d_syz_z[c];
        float syz_zr = d_syz_zr[c];
        float szz_z = d_szz_z[c];
        float szz_zr = d_szz_zr[c];


// Backpropagate the forward velocity
#if BACK_PROP_TYPE==1
    {
        float vxold = ldf(vx, indv), vyold = ldf(vy, indv), vzold = ldf(vz, indv);
        lvx=((sxx_x + sxy_y + sxz_z)*ldf(rip, indp));
        lvy=((syy_y + sxy_x + syz_z)*ldf(rjp, indp));
        lvz=((szz_z + sxz_x + syz_y)*ldf(rkp, indp));

        stf(vx, indv, ldf(vx, indv) - (lvx));
        stf(vy, indv, ldf(vy, indv) - (lvy));
        stf(vz, indv, ldf(vz, indv) - (lvz));
        fvx = lvx;
        fvy = lvy;
        fvz = lvz;

        // Inject the boundary values
        m=inject_ind(gidz, gidy, gidx);
        if (m!=-1){
            stf(vx, indv, ldf(vxbnd, m));
            stf(vy, indv, ldf(vybnd, m));
            stf(vz, indv, ldf(vzbnd, m));
            /* Exact forward increment from the two stored states (see
               update_adjv2D.cl). */
            float gt = 1.0f;
            #if ABS_TYPE==2
            #if FREESURF==0
            if (gidz-FDOH<NAB) gt*=taper[gidz-FDOH];
            #endif
            if (gidz>NZS-NAB-FDOH-1) gt*=taper[NZS-FDOH-gidz-1];
            if (gidy-FDOH<NAB) gt*=taper[gidy-FDOH];
            if (gidy>NYS-NAB-FDOH-1) gt*=taper[NYS-FDOH-gidy-1];
            #if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB) gt*=taper[gidx-FDOH];
            #endif
            #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1) gt*=taper[NX-FDOH-gidx-1];
            #endif
            #endif
            fvx = vxold/gt - ldf(vx, indv);
            fvy = vyold/gt - ldf(vy, indv);
            fvz = vzold/gt - ldf(vz, indv);
        }
    }
#endif

// Correct adjoint spatial derivatives to implement CPML (BACK_PROP_TYPE 2;
// BACK_PROP_TYPE 1 uses the exact adjoint of cpml_adj3D.cl instead)
#if ABS_TYPE==1 && BACK_PROP_TYPE!=1
    {
        int ind;
        int i,j,k;
        if (gidz>NZS-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz - NZS+NAB+FDOH+NAB;
            ind=2*NAB-1-k;

            stf(psi_sxz_z, I_psi_sxz_z(k,j,i), b_z[ind+1] * ldf(psi_sxz_z, I_psi_sxz_z(k,j,i)) + a_z[ind+1] * sxz_zr);
            sxz_zr = sxz_zr / K_z[ind+1] + ldf(psi_sxz_z, I_psi_sxz_z(k,j,i));
            stf(psi_syz_z, I_psi_syz_z(k,j,i), b_z[ind+1] * ldf(psi_syz_z, I_psi_syz_z(k,j,i)) + a_z[ind+1] * syz_zr);
            syz_zr = syz_zr / K_z[ind+1] + ldf(psi_syz_z, I_psi_syz_z(k,j,i));
            stf(psi_szz_z, I_psi_szz_z(k,j,i), b_z_half[ind] * ldf(psi_szz_z, I_psi_szz_z(k,j,i)) + a_z_half[ind] * szz_zr);
            szz_zr = szz_zr / K_z_half[ind] + ldf(psi_szz_z, I_psi_szz_z(k,j,i));

        }

#if FREESURF==0
        else if (gidz-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            stf(psi_sxz_z, I_psi_sxz_z(k,j,i), b_z[k] * ldf(psi_sxz_z, I_psi_sxz_z(k,j,i)) + a_z[k] * sxz_zr);
            sxz_zr = sxz_zr / K_z[k] + ldf(psi_sxz_z, I_psi_sxz_z(k,j,i));
            stf(psi_syz_z, I_psi_syz_z(k,j,i), b_z[k] * ldf(psi_syz_z, I_psi_syz_z(k,j,i)) + a_z[k] * syz_zr);
            syz_zr = syz_zr / K_z[k] + ldf(psi_syz_z, I_psi_syz_z(k,j,i));
            stf(psi_szz_z, I_psi_szz_z(k,j,i), b_z_half[k] * ldf(psi_szz_z, I_psi_szz_z(k,j,i)) + a_z_half[k] * szz_zr);
            szz_zr = szz_zr / K_z_half[k] + ldf(psi_szz_z, I_psi_szz_z(k,j,i));

        }
#endif

        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            stf(psi_sxy_y, I_psi_sxy_y(k,j,i), b_y[j] * ldf(psi_sxy_y, I_psi_sxy_y(k,j,i)) + a_y[j] * sxy_yr);
            sxy_yr = sxy_yr / K_y[j] + ldf(psi_sxy_y, I_psi_sxy_y(k,j,i));
            stf(psi_syy_y, I_psi_syy_y(k,j,i), b_y_half[j] * ldf(psi_syy_y, I_psi_syy_y(k,j,i)) + a_y_half[j] * syy_yr);
            syy_yr = syy_yr / K_y_half[j] + ldf(psi_syy_y, I_psi_syy_y(k,j,i));
            stf(psi_syz_y, I_psi_syz_y(k,j,i), b_y[j] * ldf(psi_syz_y, I_psi_syz_y(k,j,i)) + a_y[j] * syz_yr);
            syz_yr = syz_yr / K_y[j] + ldf(psi_syz_y, I_psi_syz_y(k,j,i));

        }

        else if (gidy>NYS-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy - NYS+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-j;

            stf(psi_sxy_y, I_psi_sxy_y(k,j,i), b_y[ind+1] * ldf(psi_sxy_y, I_psi_sxy_y(k,j,i)) + a_y[ind+1] * sxy_yr);
            sxy_yr = sxy_yr / K_y[ind+1] + ldf(psi_sxy_y, I_psi_sxy_y(k,j,i));
            stf(psi_syy_y, I_psi_syy_y(k,j,i), b_y_half[ind] * ldf(psi_syy_y, I_psi_syy_y(k,j,i)) + a_y_half[ind] * syy_yr);
            syy_yr = syy_yr / K_y_half[ind] + ldf(psi_syy_y, I_psi_syy_y(k,j,i));
            stf(psi_syz_y, I_psi_syz_y(k,j,i), b_y[ind+1] * ldf(psi_syz_y, I_psi_syz_y(k,j,i)) + a_y[ind+1] * syz_yr);
            syz_yr = syz_yr / K_y[ind+1] + ldf(psi_syz_y, I_psi_syz_y(k,j,i));


        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            stf(psi_sxx_x, I_psi_sxx_x(k,j,i), b_x_half[i] * ldf(psi_sxx_x, I_psi_sxx_x(k,j,i)) + a_x_half[i] * sxx_xr);
            sxx_xr = sxx_xr / K_x_half[i] + ldf(psi_sxx_x, I_psi_sxx_x(k,j,i));
            stf(psi_sxy_x, I_psi_sxy_x(k,j,i), b_x[i] * ldf(psi_sxy_x, I_psi_sxy_x(k,j,i)) + a_x[i] * sxy_xr);
            sxy_xr = sxy_xr / K_x[i] + ldf(psi_sxy_x, I_psi_sxy_x(k,j,i));
            stf(psi_sxz_x, I_psi_sxz_x(k,j,i), b_x[i] * ldf(psi_sxz_x, I_psi_sxz_x(k,j,i)) + a_x[i] * sxz_xr);
            sxz_xr = sxz_xr / K_x[i] + ldf(psi_sxz_x, I_psi_sxz_x(k,j,i));

        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){

            i =gidx - NX+NAB+FDOH+NAB;
            j =gidy-FDOH;
            k =gidz-FDOH;
            ind=2*NAB-1-i;

            stf(psi_sxx_x, I_psi_sxx_x(k,j,i), b_x_half[ind] * ldf(psi_sxx_x, I_psi_sxx_x(k,j,i)) + a_x_half[ind] * sxx_xr);
            sxx_xr = sxx_xr / K_x_half[ind] + ldf(psi_sxx_x, I_psi_sxx_x(k,j,i));
            stf(psi_sxy_x, I_psi_sxy_x(k,j,i), b_x[ind+1] * ldf(psi_sxy_x, I_psi_sxy_x(k,j,i)) + a_x[ind+1] * sxy_xr);
            sxy_xr = sxy_xr / K_x[ind+1] + ldf(psi_sxy_x, I_psi_sxy_x(k,j,i));
            stf(psi_sxz_x, I_psi_sxz_x(k,j,i), b_x[ind+1] * ldf(psi_sxz_x, I_psi_sxz_x(k,j,i)) + a_x[ind+1] * sxz_xr);
            sxz_xr = sxz_xr / K_x[ind+1] + ldf(psi_sxz_x, I_psi_sxz_x(k,j,i));



        }
#endif
    }
#endif

    // Update adjoint velocities
    lvx=((sxx_xr + sxy_yr + sxz_zr)*ldf(rip, indp));
    lvy=((syy_yr + sxy_xr + syz_zr)*ldf(rjp, indp));
    lvz=((szz_zr + sxz_xr + syz_yr)*ldf(rkp, indp));
    stf(vxr, indv, ldf(vxr, indv) + (lvx));
    stf(vyr, indv, ldf(vyr, indv) + (lvy));
    stf(vzr, indv, ldf(vzr, indv) + (lvz));


// Absorbing boundary
#if ABS_TYPE==2
    {
        #if FREESURF==0
        if (gidz-FDOH<NAB){
            stf(vxr, indv, ldf(vxr, indv) * (taper[gidz-FDOH]));
            stf(vyr, indv, ldf(vyr, indv) * (taper[gidz-FDOH]));
            stf(vzr, indv, ldf(vzr, indv) * (taper[gidz-FDOH]));
        }
        #endif

        if (gidz>NZS-NAB-FDOH-1){
            stf(vxr, indv, ldf(vxr, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(vyr, indv, ldf(vyr, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(vzr, indv, ldf(vzr, indv) * (taper[NZS-FDOH-gidz-1]));
        }

        if (gidy-FDOH<NAB){
            stf(vxr, indv, ldf(vxr, indv) * (taper[gidy-FDOH]));
            stf(vyr, indv, ldf(vyr, indv) * (taper[gidy-FDOH]));
            stf(vzr, indv, ldf(vzr, indv) * (taper[gidy-FDOH]));
        }

        if (gidy>NYS-NAB-FDOH-1){
            stf(vxr, indv, ldf(vxr, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(vyr, indv, ldf(vyr, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(vzr, indv, ldf(vzr, indv) * (taper[NYS-FDOH-gidy-1]));
        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            stf(vxr, indv, ldf(vxr, indv) * (taper[gidx-FDOH]));
            stf(vyr, indv, ldf(vyr, indv) * (taper[gidx-FDOH]));
            stf(vzr, indv, ldf(vzr, indv) * (taper[gidx-FDOH]));
        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            stf(vxr, indv, ldf(vxr, indv) * (taper[NX-FDOH-gidx-1]));
            stf(vyr, indv, ldf(vyr, indv) * (taper[NX-FDOH-gidx-1]));
            stf(vzr, indv, ldf(vzr, indv) * (taper[NX-FDOH-gidx-1]));
        }
#endif
    }
#endif

// Density gradient calculation on the fly
#if BACK_PROP_TYPE==1
    /* Residual term at receiver cells for VELOCITY receivers -- the mirror of
     * update_adjs3D.cl's, landing in each component's own STAGGERED buoyancy
     * accumulator. See update_adjv2D.cl. */
    /* Receiver cells: the residual term of the gradient is added by
       rec_grad.cl (see there). */
    /* The residual of step nt measures the velocity recorded at the end of
       that step, v^{nt+1} = v + fv (v was just stepped back to v^{nt}). */
    /* Under a stress-image free surface the FDOH rows that the adjoint
       surface gather changes get their density gradient from surface_grad
       (surface3D_adj.cl), in the direct form. */
    int sumform = 1;
    #if FREESURF==1 && GRADOUT==1
    if (gidz-FDOH < FDOH) sumform = 0;
    #endif
    if (sumform){
    gradrip[indp] += GSCALE(-ldf(vx, indv)*lvx);
    gradrjp[indp] += GSCALE(-ldf(vy, indv)*lvy);
    gradrkp[indp] += GSCALE(-ldf(vz, indv)*lvz);
    }

    /* Source term of the misfit gradient -- GJI 2017 eq. (26a) -- for a FORCE
     * source; see update_adjv2D.cl for the derivation. Each component lands in
     * its OWN staggered buoyancy accumulator (vx at rip, vy at rjp, vz at rkp)
     * so average_grad_transpose() can apply the averaging Jacobian; writing it
     * into the cell-centred gradrho would skip that and place the sensitivity
     * at the wrong point. Coefficient 1 -- (A1a) has no c-factor.
     * Types index kernel_sources()'s src_names[] = {vx,vy,vz,p,...}.
     * No src_scale: this file is compiled only at FP16==0. */
    #if GRADOUT==1
    if (nsrc>0 && sumform){
        for (int srci=0; srci<nsrc; srci++){
            int st = (int)src_pos[4+5*srci];
            if (st>=0 && st<=2){
                int si=(int)(src_pos[0+5*srci]/DH)+FDOH;
                int sj=(int)(src_pos[1+5*srci]/DH)+FDOH;
                int sk=(int)(src_pos[2+5*srci]/DH)+FDOH;
                if (si==gidx && sj==gidy && sk==gidz){
                    float samp = pow2s(DT*src[srci*NT+nt], src_scale - PARSCALE);
                    if      (st==0) gradrip[indp] += GSCALE(ldf(vxr, indv)*samp);
                    else if (st==1) gradrjp[indp] += GSCALE(ldf(vyr, indv)*samp);
                    else            gradrkp[indp] += GSCALE(ldf(vzr, indv)*samp);
                }
            }
        }
    }
    #endif

#if HOUT==1
    Hrho[inp])+= pown(ldf(vx, indv),2)+pown(ldf(vy, indv),2)+pown(ldf(vz, indv),2);
#endif

#endif

#if GRADSRCOUT==1
    //TODO
//    if (nsrc>0){
//
//
//        for (int srci=0; srci<nsrc; srci++){
//
//
//            int i=(int)(srcpos_loc(0,srci)/DH-0.5)+FDOH;
//            int j=(int)(srcpos_loc(1,srci)/DH-0.5)+FDOH;
//            int k=(int)(srcpos_loc(2,srci)/DH-0.5)+FDOH;
//
//            if (i==gidx && j==gidy && k==gidz){
//
//                int SOURCE_TYPE= (int)srcpos_loc(4,srci);
//
//                if (SOURCE_TYPE==2){
//                    /* single force in x */
//                    gradsrc(srci,nt)+= ldf(vxr, indv)/ldf(rip, inp)/(DH*DH*DH);
//                }
//                else if (SOURCE_TYPE==3){
//                    /* single force in y */
//
//                    gradsrc(srci,nt)+= ldf(vyr, indv)/ldf(rip, inp)/(DH*DH*DH);
//                }
//                else if (SOURCE_TYPE==4){
//                    /* single force in z */
//
//                    gradsrc(srci,nt)+= ldf(vzr, indv)/ldf(rip, inp)/(DH*DH*DH);
//                }
//
//            }
//        }
//
//
//    }
#endif
    

    }
}
