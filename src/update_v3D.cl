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

/*Update of the velocity in 3D*/

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

FUNDEF void update_v(int offcomm,
                       GLOBARG STO * RESTRICT vx,              GLOBARG STO * RESTRICT vy,              GLOBARG STO * RESTRICT vz,
                       GLOBARG const STO * RESTRICT sxx,       GLOBARG const STO * RESTRICT syy,       GLOBARG const STO * RESTRICT szz,
                       GLOBARG const STO * RESTRICT sxy,       GLOBARG const STO * RESTRICT syz,       GLOBARG const STO * RESTRICT sxz,
                       GLOBARG const STO * RESTRICT rip,       GLOBARG const STO * RESTRICT rjp,       GLOBARG const STO * RESTRICT rkp,
                       GLOBARG const float * RESTRICT taper,
                       GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                       GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                       GLOBARG const float * RESTRICT K_y,       GLOBARG const float * RESTRICT a_y,       GLOBARG const float * RESTRICT b_y,
                       GLOBARG const float * RESTRICT K_y_half,  GLOBARG const float * RESTRICT a_y_half,  GLOBARG const float * RESTRICT b_y_half,
                       GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                       GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                       GLOBARG float * RESTRICT psi_sxx_x,       GLOBARG float * RESTRICT psi_sxy_x,       GLOBARG float * RESTRICT psi_sxy_y,
                       GLOBARG float * RESTRICT psi_sxz_x,       GLOBARG float * RESTRICT psi_sxz_z,       GLOBARG float * RESTRICT psi_syy_y,
                       GLOBARG float * RESTRICT psi_syz_y,       GLOBARG float * RESTRICT psi_syz_z,       GLOBARG float * RESTRICT psi_szz_z,
                       LOCARG)
{
    LOCDEF

    
    
// If we use local memory

    int gidy = GRP1*LSZ1 + LID1 + FDOH;
    int gidx = GRP2*LSZ2 + LID2 + FDOH + offcomm;
    int zt = Z0 + NV*(GRP0*LSZ0 + LID0);
    int tz = NV*LSZ0 + 2*ZH, ty = LSZ1 + 2*FDOH, tx = LSZ2 + 2*FDOH;
    int tile_z0 = Z0 + NV*GRP0*LSZ0 - ZH, tile_y0 = GRP1*LSZ1, tile_x0 = GRP2*LSZ2 + offcomm;
    int it0 = ((LID2+FDOH)*ty + (LID1+FDOH))*tz + ZH + NV*LID0;
    int sy = tz, sx = tz*ty;
    float d_sxx_x[NV], d_sxy_x[NV], d_sxy_y[NV], d_sxz_x[NV], d_sxz_z[NV], d_syy_y[NV], d_syz_y[NV], d_syz_z[NV], d_szz_z[NV];
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
        float sxy_x = d_sxy_x[c];
        float sxy_y = d_sxy_y[c];
        float sxz_x = d_sxz_x[c];
        float sxz_z = d_sxz_z[c];
        float syy_y = d_syy_y[c];
        float syz_y = d_syz_y[c];
        float syz_z = d_syz_z[c];
        float szz_z = d_szz_z[c];


    
    
    
// Correct spatial derivatives to implement CPML
#if ABS_TYPE==1
    {
        int i,j,k, indm;
        
        if (gidz>NZS-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz - NZS+NAB+FDOH+NAB;
            indm=2*NAB-1-k;

            psi_sxz_z[I_psi_sxz_z(k,j,i)] = b_z[indm+1] * psi_sxz_z[I_psi_sxz_z(k,j,i)] + a_z[indm+1] * sxz_z;
            sxz_z = sxz_z / K_z[indm+1] + psi_sxz_z[I_psi_sxz_z(k,j,i)];
            psi_syz_z[I_psi_syz_z(k,j,i)] = b_z[indm+1] * psi_syz_z[I_psi_syz_z(k,j,i)] + a_z[indm+1] * syz_z;
            syz_z = syz_z / K_z[indm+1] + psi_syz_z[I_psi_syz_z(k,j,i)];
            psi_szz_z[I_psi_szz_z(k,j,i)] = b_z_half[indm] * psi_szz_z[I_psi_szz_z(k,j,i)] + a_z_half[indm] * szz_z;
            szz_z = szz_z / K_z_half[indm] + psi_szz_z[I_psi_szz_z(k,j,i)];

        }

#if FREESURF==0
        else if (gidz-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_sxz_z[I_psi_sxz_z(k,j,i)] = b_z[k] * psi_sxz_z[I_psi_sxz_z(k,j,i)] + a_z[k] * sxz_z;
            sxz_z = sxz_z / K_z[k] + psi_sxz_z[I_psi_sxz_z(k,j,i)];
            psi_syz_z[I_psi_syz_z(k,j,i)] = b_z[k] * psi_syz_z[I_psi_syz_z(k,j,i)] + a_z[k] * syz_z;
            syz_z = syz_z / K_z[k] + psi_syz_z[I_psi_syz_z(k,j,i)];
            psi_szz_z[I_psi_szz_z(k,j,i)] = b_z_half[k] * psi_szz_z[I_psi_szz_z(k,j,i)] + a_z_half[k] * szz_z;
            szz_z = szz_z / K_z_half[k] + psi_szz_z[I_psi_szz_z(k,j,i)];

        }
#endif

        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_sxy_y[I_psi_sxy_y(k,j,i)] = b_y[j] * psi_sxy_y[I_psi_sxy_y(k,j,i)] + a_y[j] * sxy_y;
            sxy_y = sxy_y / K_y[j] + psi_sxy_y[I_psi_sxy_y(k,j,i)];
            psi_syy_y[I_psi_syy_y(k,j,i)] = b_y_half[j] * psi_syy_y[I_psi_syy_y(k,j,i)] + a_y_half[j] * syy_y;
            syy_y = syy_y / K_y_half[j] + psi_syy_y[I_psi_syy_y(k,j,i)];
            psi_syz_y[I_psi_syz_y(k,j,i)] = b_y[j] * psi_syz_y[I_psi_syz_y(k,j,i)] + a_y[j] * syz_y;
            syz_y = syz_y / K_y[j] + psi_syz_y[I_psi_syz_y(k,j,i)];

        }

        else if (gidy>NYS-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy - NYS+NAB+FDOH+NAB;
            k =gidz-FDOH;
            indm=2*NAB-1-j;

            psi_sxy_y[I_psi_sxy_y(k,j,i)] = b_y[indm+1] * psi_sxy_y[I_psi_sxy_y(k,j,i)] + a_y[indm+1] * sxy_y;
            sxy_y = sxy_y / K_y[indm+1] + psi_sxy_y[I_psi_sxy_y(k,j,i)];
            psi_syy_y[I_psi_syy_y(k,j,i)] = b_y_half[indm] * psi_syy_y[I_psi_syy_y(k,j,i)] + a_y_half[indm] * syy_y;
            syy_y = syy_y / K_y_half[indm] + psi_syy_y[I_psi_syy_y(k,j,i)];
            psi_syz_y[I_psi_syz_y(k,j,i)] = b_y[indm+1] * psi_syz_y[I_psi_syz_y(k,j,i)] + a_y[indm+1] * syz_y;
            syz_y = syz_y / K_y[indm+1] + psi_syz_y[I_psi_syz_y(k,j,i)];


        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_sxx_x[I_psi_sxx_x(k,j,i)] = b_x_half[i] * psi_sxx_x[I_psi_sxx_x(k,j,i)] + a_x_half[i] * sxx_x;
            sxx_x = sxx_x / K_x_half[i] + psi_sxx_x[I_psi_sxx_x(k,j,i)];
            psi_sxy_x[I_psi_sxy_x(k,j,i)] = b_x[i] * psi_sxy_x[I_psi_sxy_x(k,j,i)] + a_x[i] * sxy_x;
            sxy_x = sxy_x / K_x[i] + psi_sxy_x[I_psi_sxy_x(k,j,i)];
            psi_sxz_x[I_psi_sxz_x(k,j,i)] = b_x[i] * psi_sxz_x[I_psi_sxz_x(k,j,i)] + a_x[i] * sxz_x;
            sxz_x = sxz_x / K_x[i] + psi_sxz_x[I_psi_sxz_x(k,j,i)];

        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){

            i =gidx - NX+NAB+FDOH+NAB;
            j =gidy-FDOH;
            k =gidz-FDOH;
            indm=2*NAB-1-i;

            psi_sxx_x[I_psi_sxx_x(k,j,i)] = b_x_half[indm] * psi_sxx_x[I_psi_sxx_x(k,j,i)] + a_x_half[indm] * sxx_x;
            sxx_x = sxx_x / K_x_half[indm] + psi_sxx_x[I_psi_sxx_x(k,j,i)];
            psi_sxy_x[I_psi_sxy_x(k,j,i)] = b_x[indm+1] * psi_sxy_x[I_psi_sxy_x(k,j,i)] + a_x[indm+1] * sxy_x;
            sxy_x = sxy_x / K_x[indm+1] + psi_sxy_x[I_psi_sxy_x(k,j,i)];
            psi_sxz_x[I_psi_sxz_x(k,j,i)] = b_x[indm+1] * psi_sxz_x[I_psi_sxz_x(k,j,i)] + a_x[indm+1] * sxz_x;
            sxz_x = sxz_x / K_x[indm+1] + psi_sxz_x[I_psi_sxz_x(k,j,i)];



        }
#endif
    }
#endif

// Update the velocities
    {
        stf(vx, indv, ldf(vx, indv) + (((sxx_x + sxy_y + sxz_z)*ldf(rip, indp))));
        stf(vy, indv, ldf(vy, indv) + (((syy_y + sxy_x + syz_z)*ldf(rjp, indp))));
        stf(vz, indv, ldf(vz, indv) + (((szz_z + sxz_x + syz_y)*ldf(rkp, indp))));
    }
    
// Absorbing boundary
#if ABS_TYPE==2
    {
#if FREESURF==0
        if (gidz-FDOH<NAB){
            stf(vx, indv, ldf(vx, indv) * (taper[gidz-FDOH]));
            stf(vy, indv, ldf(vy, indv) * (taper[gidz-FDOH]));
            stf(vz, indv, ldf(vz, indv) * (taper[gidz-FDOH]));
        }
#endif
        
        if (gidz>NZS-NAB-FDOH-1){
            stf(vx, indv, ldf(vx, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(vy, indv, ldf(vy, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(vz, indv, ldf(vz, indv) * (taper[NZS-FDOH-gidz-1]));
        }
        
        if (gidy-FDOH<NAB){
            stf(vx, indv, ldf(vx, indv) * (taper[gidy-FDOH]));
            stf(vy, indv, ldf(vy, indv) * (taper[gidy-FDOH]));
            stf(vz, indv, ldf(vz, indv) * (taper[gidy-FDOH]));
        }
        
        if (gidy>NYS-NAB-FDOH-1){
            stf(vx, indv, ldf(vx, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(vy, indv, ldf(vy, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(vz, indv, ldf(vz, indv) * (taper[NYS-FDOH-gidy-1]));
        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            stf(vx, indv, ldf(vx, indv) * (taper[gidx-FDOH]));
            stf(vy, indv, ldf(vy, indv) * (taper[gidx-FDOH]));
            stf(vz, indv, ldf(vz, indv) * (taper[gidx-FDOH]));
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            stf(vx, indv, ldf(vx, indv) * (taper[NX-FDOH-gidx-1]));
            stf(vy, indv, ldf(vy, indv) * (taper[NX-FDOH-gidx-1]));
            stf(vz, indv, ldf(vz, indv) * (taper[NX-FDOH-gidx-1]));
        }
#endif
    }
#endif


    }
}
