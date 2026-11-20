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

/*Update of the stresses in 3D*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */

/* Generated from the FP32 3-D kernel by torch-robust/unify/gen3d.py: unified storage (header_vec.cl),
   per-cell physics unchanged. */
#define I_psi_vx_x(z,y,x) ((x)*(NYS-2*FDOH)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_vy_x(z,y,x) ((x)*(NYS-2*FDOH)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_vz_x(z,y,x) ((x)*(NYS-2*FDOH)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_vx_y(z,y,x) ((x)*(2*NAB)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_vy_y(z,y,x) ((x)*(2*NAB)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_vz_y(z,y,x) ((x)*(2*NAB)*(NZS-2*FDOH)+(y)*(NZS-2*FDOH)+(z))
#define I_psi_vx_z(z,y,x) ((x)*(NYS-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z))
#define I_psi_vy_z(z,y,x) ((x)*(NYS-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z))
#define I_psi_vz_z(z,y,x) ((x)*(NYS-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z))

FUNDEF void update_s(int offcomm,  int nt,
                       GLOBARG const STO * RESTRICT vx,          GLOBARG const STO * RESTRICT vy,        GLOBARG const STO * RESTRICT vz,
                       GLOBARG STO * RESTRICT sxx,               GLOBARG STO * RESTRICT syy,             GLOBARG STO * RESTRICT szz,
                       GLOBARG STO * RESTRICT sxy,               GLOBARG STO * RESTRICT syz,             GLOBARG STO * RESTRICT sxz,
                       GLOBARG const STO * RESTRICT M,           GLOBARG const STO * RESTRICT mu,        GLOBARG const STO * RESTRICT muipjp,
                       GLOBARG const STO * RESTRICT mujpkp,      GLOBARG const STO * RESTRICT muipkp,
                       GLOBARG STO * RESTRICT rxx,               GLOBARG STO * RESTRICT ryy,             GLOBARG STO * RESTRICT rzz,
                       GLOBARG STO * RESTRICT rxy,               GLOBARG STO * RESTRICT ryz,             GLOBARG STO * RESTRICT rxz,
                       GLOBARG const STO * RESTRICT taus,        GLOBARG const STO * RESTRICT tausipjp,  GLOBARG const STO * RESTRICT tausjpkp,
                       GLOBARG const STO * RESTRICT tausipkp,    GLOBARG const STO * RESTRICT taup,      GLOBARG const float * RESTRICT eta,
                       GLOBARG const STO * RESTRICT srcpos_loc,  GLOBARG const STO * RESTRICT signals,   GLOBARG const float * RESTRICT taper,
                       GLOBARG const float * RESTRICT K_x,         GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                       GLOBARG const float * RESTRICT K_x_half,    GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                       GLOBARG const float * RESTRICT K_y,         GLOBARG const float * RESTRICT a_y,       GLOBARG const float * RESTRICT b_y,
                       GLOBARG const float * RESTRICT K_y_half,    GLOBARG const float * RESTRICT a_y_half,  GLOBARG const float * RESTRICT b_y_half,
                       GLOBARG const float * RESTRICT K_z,         GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                       GLOBARG const float * RESTRICT K_z_half,    GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                       GLOBARG float * RESTRICT psi_vx_x,          GLOBARG float * RESTRICT psi_vx_y,        GLOBARG float * RESTRICT psi_vx_z,
                       GLOBARG float * RESTRICT psi_vy_x,          GLOBARG float * RESTRICT psi_vy_y,        GLOBARG float * RESTRICT psi_vy_z,
                       GLOBARG float * RESTRICT psi_vz_x,          GLOBARG float * RESTRICT psi_vz_y,        GLOBARG float * RESTRICT psi_vz_z,
                       LOCARG)
{
    LOCDEF

    
    float fipjp, fjpkp, fipkp, f, g;
#if LVE>0
    float leta[LVE];
    int l,indr;
    float sumrxy,sumryz,sumrxz,sumrxx,sumryy,sumrzz;
    float b,cv,e,d,dipjp,djpkp,dipkp;
    float lmuipjp, lmuipkp, lmujpkp, ltaup, ltaus, ltausipjp, ltausipkp, ltausjpkp;
    float lsxx, lsyy, lszz, lsxy, lsxz, lsyz;
    float lM, lmu;
#endif
    float vxyyx,vyzzy,vxzzx,vxxyyzz,vyyzz,vxxzz,vxxyy;
    
    
    
// If we use local memory

    int gidy = GRP1*LSZ1 + LID1 + FDOH;
    int gidx = GRP2*LSZ2 + LID2 + FDOH + offcomm;
    int zt = Z0 + NV*(GRP0*LSZ0 + LID0);
    int tz = NV*LSZ0 + 2*ZH, ty = LSZ1 + 2*FDOH, tx = LSZ2 + 2*FDOH;
    int tile_z0 = Z0 + NV*GRP0*LSZ0 - ZH, tile_y0 = GRP1*LSZ1, tile_x0 = GRP2*LSZ2 + offcomm;
    int it0 = ((LID2+FDOH)*ty + (LID1+FDOH))*tz + ZH + NV*LID0;
    int sy = tz, sx = tz*ty;
#if NOTILE
    /* SEISCL_NOTILE=1 (experiment): no shared tiles, the derivatives read the
       arrays in place; threads past the grid return before reading. */
#if COMM12==0
    if (zt > NZS-FDOH-1 || gidy > NYS-FDOH-1 || (gidx-offcomm) > (NX-FDOH-1-LCOMM)) return;
#else
    if (zt > NZS-FDOH-1 || gidy > NYS-FDOH-1) return;
#endif
    it0 = (gidx*NYS + gidy)*NZS + zt; sy = NZS; sx = NZS*NYS;
#endif
    float d_vxx[NV], d_vxy[NV], d_vxz[NV], d_vyx[NV], d_vyy[NV], d_vyz[NV], d_vzx[NV], d_vzy[NV], d_vzz[NV];
    load_tile3(lvar, vz, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_vzx[c] = tdp(lvar, it0+c, sx);
    for (int c=0; c<NV; c++) d_vzy[c] = tdp(lvar, it0+c, sy);
    for (int c=0; c<NV; c++) d_vzz[c] = tdm(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, vy, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_vyx[c] = tdp(lvar, it0+c, sx);
    for (int c=0; c<NV; c++) d_vyy[c] = tdm(lvar, it0+c, sy);
    for (int c=0; c<NV; c++) d_vyz[c] = tdp(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, vx, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    for (int c=0; c<NV; c++) d_vxx[c] = tdm(lvar, it0+c, sx);
    for (int c=0; c<NV; c++) d_vxy[c] = tdp(lvar, it0+c, sy);
    for (int c=0; c<NV; c++) d_vxz[c] = tdp(lvar, it0+c, 1);
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
        float vxx = d_vxx[c];
        float vxy = d_vxy[c];
        float vxz = d_vxz[c];
        float vyx = d_vyx[c];
        float vyy = d_vyy[c];
        float vyz = d_vyz[c];
        float vzx = d_vzx[c];
        float vzy = d_vzy[c];
        float vzz = d_vzz[c];


 
// Correct spatial derivatives to implement CPML
#if ABS_TYPE==1
    {
        int i,j,k,indm;
        if (gidz>NZS-NAB-FDOH-1){
            
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz - NZS+NAB+FDOH+NAB;
            indm=2*NAB-1-k;

            psi_vx_z[I_psi_vx_z(k,j,i)] = b_z_half[indm] * psi_vx_z[I_psi_vx_z(k,j,i)] + a_z_half[indm] * vxz;
            vxz = vxz / K_z_half[indm] + psi_vx_z[I_psi_vx_z(k,j,i)];
            psi_vy_z[I_psi_vy_z(k,j,i)] = b_z_half[indm] * psi_vy_z[I_psi_vy_z(k,j,i)] + a_z_half[indm] * vyz;
            vyz = vyz / K_z_half[indm] + psi_vy_z[I_psi_vy_z(k,j,i)];
            psi_vz_z[I_psi_vz_z(k,j,i)] = b_z[indm+1] * psi_vz_z[I_psi_vz_z(k,j,i)] + a_z[indm+1] * vzz;
            vzz = vzz / K_z[indm+1] + psi_vz_z[I_psi_vz_z(k,j,i)];

        }

#if FREESURF==0
        if (gidz-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;


            psi_vx_z[I_psi_vx_z(k,j,i)] = b_z_half[k] * psi_vx_z[I_psi_vx_z(k,j,i)] + a_z_half[k] * vxz;
            vxz = vxz / K_z_half[k] + psi_vx_z[I_psi_vx_z(k,j,i)];
            psi_vy_z[I_psi_vy_z(k,j,i)] = b_z_half[k] * psi_vy_z[I_psi_vy_z(k,j,i)] + a_z_half[k] * vyz;
            vyz = vyz / K_z_half[k] + psi_vy_z[I_psi_vy_z(k,j,i)];
            psi_vz_z[I_psi_vz_z(k,j,i)] = b_z[k] * psi_vz_z[I_psi_vz_z(k,j,i)] + a_z[k] * vzz;
            vzz = vzz / K_z[k] + psi_vz_z[I_psi_vz_z(k,j,i)];


        }
#endif

        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_vx_y[I_psi_vx_y(k,j,i)] = b_y_half[j] * psi_vx_y[I_psi_vx_y(k,j,i)] + a_y_half[j] * vxy;
            vxy = vxy / K_y_half[j] + psi_vx_y[I_psi_vx_y(k,j,i)];
            psi_vy_y[I_psi_vy_y(k,j,i)] = b_y[j] * psi_vy_y[I_psi_vy_y(k,j,i)] + a_y[j] * vyy;
            vyy = vyy / K_y[j] + psi_vy_y[I_psi_vy_y(k,j,i)];
            psi_vz_y[I_psi_vz_y(k,j,i)] = b_y_half[j] * psi_vz_y[I_psi_vz_y(k,j,i)] + a_y_half[j] * vzy;
            vzy = vzy / K_y_half[j] + psi_vz_y[I_psi_vz_y(k,j,i)];

        }

        if (gidy>NYS-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy - NYS+NAB+FDOH+NAB;
            k =gidz-FDOH;
            indm=2*NAB-1-j;


            psi_vx_y[I_psi_vx_y(k,j,i)] = b_y_half[indm] * psi_vx_y[I_psi_vx_y(k,j,i)] + a_y_half[indm] * vxy;
            vxy = vxy / K_y_half[indm] + psi_vx_y[I_psi_vx_y(k,j,i)];
            psi_vy_y[I_psi_vy_y(k,j,i)] = b_y[indm+1] * psi_vy_y[I_psi_vy_y(k,j,i)] + a_y[indm+1] * vyy;
            vyy = vyy / K_y[indm+1] + psi_vy_y[I_psi_vy_y(k,j,i)];
            psi_vz_y[I_psi_vz_y(k,j,i)] = b_y_half[indm] * psi_vz_y[I_psi_vz_y(k,j,i)] + a_y_half[indm] * vzy;
            vzy = vzy / K_y_half[indm] + psi_vz_y[I_psi_vz_y(k,j,i)];


        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_vx_x[I_psi_vx_x(k,j,i)] = b_x[i] * psi_vx_x[I_psi_vx_x(k,j,i)] + a_x[i] * vxx;
            vxx = vxx / K_x[i] + psi_vx_x[I_psi_vx_x(k,j,i)];
            psi_vy_x[I_psi_vy_x(k,j,i)] = b_x_half[i] * psi_vy_x[I_psi_vy_x(k,j,i)] + a_x_half[i] * vyx;
            vyx = vyx / K_x_half[i] + psi_vy_x[I_psi_vy_x(k,j,i)];
            psi_vz_x[I_psi_vz_x(k,j,i)] = b_x_half[i] * psi_vz_x[I_psi_vz_x(k,j,i)] + a_x_half[i] * vzx;
            vzx = vzx / K_x_half[i] + psi_vz_x[I_psi_vz_x(k,j,i)];


        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){

            i =gidx - NX+NAB+FDOH+NAB;
            j =gidy-FDOH;
            k =gidz-FDOH;
            indm=2*NAB-1-i;


            psi_vx_x[I_psi_vx_x(k,j,i)] = b_x[indm+1] * psi_vx_x[I_psi_vx_x(k,j,i)] + a_x[indm+1] * vxx;
            vxx = vxx /K_x[indm+1] + psi_vx_x[I_psi_vx_x(k,j,i)];
            psi_vy_x[I_psi_vy_x(k,j,i)] = b_x_half[indm] * psi_vy_x[I_psi_vy_x(k,j,i)] + a_x_half[indm] * vyx;
            vyx = vyx  /K_x_half[indm] + psi_vy_x[I_psi_vy_x(k,j,i)];
            psi_vz_x[I_psi_vz_x(k,j,i)] = b_x_half[indm] * psi_vz_x[I_psi_vz_x(k,j,i)] + a_x_half[indm] * vzx;
            vzx = vzx / K_x_half[indm]  +psi_vz_x[I_psi_vz_x(k,j,i)];


        }
#endif
    }
#endif

// Read model parameters into local memory
    {
#if LVE==0
        
        fipjp=ldf(muipjp, indp);
        fjpkp=ldf(mujpkp, indp);
        fipkp=ldf(muipkp, indp);
        g=ldf(M, indp);
        f=2.0*ldf(mu, indp);
        
#else
        
        lM=ldf(M, indp);
        lmu=ldf(mu, indp);
        lmuipkp=ldf(muipkp, indp);
        lmuipjp=ldf(muipjp, indp);
        lmujpkp=ldf(mujpkp, indp);
        ltaup=ldf(taup, indp);
        ltaus=ldf(taus, indp);
        ltausipkp=ldf(tausipkp, indp);
        ltausipjp=ldf(tausipjp, indp);
        ltausjpkp=ldf(tausjpkp, indp);
        
        for (l=0;l<LVE;l++){
            leta[l]=eta[l];
        }
        
        fipjp=lmuipjp*(1.0+ (float)LVE*ltausipjp);
        fjpkp=lmujpkp*(1.0+ (float)LVE*ltausjpkp);
        fipkp=lmuipkp*(1.0+ (float)LVE*ltausipkp);
        g=lM*(1.0+(float)LVE*ltaup);
        f=2.0*lmu*(1.0+(float)LVE*ltaus);
        dipjp=lmuipjp*ltausipjp/DT;
        djpkp=lmujpkp*ltausjpkp/DT;
        dipkp=lmuipkp*ltausipkp/DT;
        d=2.0*lmu*ltaus/DT;
        e=lM*ltaup/DT;
        
        
#endif
    }

// Update the stresses
    {
#if LVE==0
        
        vxyyx=vxy+vyx;
        vyzzy=vyz+vzy;
        vxzzx=vxz+vzx;
        vxxyyzz=vxx+vyy+vzz;
        vyyzz=vyy+vzz;
        vxxzz=vxx+vzz;
        vxxyy=vxx+vyy;

        stf(sxy, indv, ldf(sxy, indv) + ((fipjp*vxyyx)));
        stf(syz, indv, ldf(syz, indv) + ((fjpkp*vyzzy)));
        stf(sxz, indv, ldf(sxz, indv) + ((fipkp*vxzzx)));
        stf(sxx, indv, ldf(sxx, indv) + (((g*vxxyyzz)-(f*vyyzz))));
        stf(syy, indv, ldf(syy, indv) + (((g*vxxyyzz)-(f*vxxzz))));
        stf(szz, indv, ldf(szz, indv) + (((g*vxxyyzz)-(f*vxxyy))));
        
#else
        
        /* computing sums of the old memory variables */
        sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
        for (l=0;l<LVE;l++){
            indr = l*NX*NYS*NZS + gidx*NYS*NZS + gidy*NZS +gidz;
            sumrxy+=ldf(rxy, indr);
            sumryz+=ldf(ryz, indr);
            sumrxz+=ldf(rxz, indr);
            sumrxx+=ldf(rxx, indr);
            sumryy+=ldf(ryy, indr);
            sumrzz+=ldf(rzz, indr);
        }
        
        /* updating components of the stress tensor, partially */
        
        vxyyx=vxy+vyx;
        vyzzy=vyz+vzy;
        vxzzx=vxz+vzx;
        vxxyyzz=vxx+vyy+vzz;
        vyyzz=vyy+vzz;
        vxxzz=vxx+vzz;
        vxxyy=vxx+vyy;
        
        lsxy=(fipjp*vxyyx)+(DT2*sumrxy);
        lsyz=(fjpkp*vyzzy)+(DT2*sumryz);
        lsxz=(fipkp*vxzzx)+(DT2*sumrxz);
        lsxx=((g*vxxyyzz)-(f*vyyzz))+(DT2*sumrxx);
        lsyy=((g*vxxyyzz)-(f*vxxzz))+(DT2*sumryy);
        lszz=((g*vxxyyzz)-(f*vxxyy))+(DT2*sumrzz);
        
        
        sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
        for (l=0;l<LVE;l++){
            b=1.0/(1.0+(leta[l]*0.5));
            cv=1.0-(leta[l]*0.5);
            indr = l*NX*NYS*NZS + gidx*NYS*NZS + gidy*NZS +gidz;
            stf(rxy, indr, b*(ldf(rxy, indr)*cv-leta[l]*(dipjp*vxyyx)));
            stf(ryz, indr, b*(ldf(ryz, indr)*cv-leta[l]*(djpkp*vyzzy)));
            stf(rxz, indr, b*(ldf(rxz, indr)*cv-leta[l]*(dipkp*vxzzx)));
            stf(rxx, indr, b*(ldf(rxx, indr)*cv-leta[l]*((e*vxxyyzz)-(d*vyyzz))));
            stf(ryy, indr, b*(ldf(ryy, indr)*cv-leta[l]*((e*vxxyyzz)-(d*vxxzz))));
            stf(rzz, indr, b*(ldf(rzz, indr)*cv-leta[l]*((e*vxxyyzz)-(d*vxxyy))));

            sumrxy+=ldf(rxy, indr);
            sumryz+=ldf(ryz, indr);
            sumrxz+=ldf(rxz, indr);
            sumrxx+=ldf(rxx, indr);
            sumryy+=ldf(ryy, indr);
            sumrzz+=ldf(rzz, indr);
        }

        /* and now the components of the stress tensor are
         completely updated */
        stf(sxy, indv, ldf(sxy, indv) + (lsxy+(DT2*sumrxy)));
        stf(syz, indv, ldf(syz, indv) + (lsyz+(DT2*sumryz)));
        stf(sxz, indv, ldf(sxz, indv) + (lsxz+(DT2*sumrxz)));
        stf(sxx, indv, ldf(sxx, indv) + (lsxx+(DT2*sumrxx)));
        stf(syy, indv, ldf(syy, indv) + (lsyy+(DT2*sumryy)));
        stf(szz, indv, ldf(szz, indv) + (lszz+(DT2*sumrzz)));

#endif
    }

// Absorbing boundary    
#if ABS_TYPE==2
    {
        
#if FREESURF==0
        if (gidz-FDOH<NAB){
            stf(sxy, indv, ldf(sxy, indv) * (taper[gidz-FDOH]));
            stf(syz, indv, ldf(syz, indv) * (taper[gidz-FDOH]));
            stf(sxz, indv, ldf(sxz, indv) * (taper[gidz-FDOH]));
            stf(sxx, indv, ldf(sxx, indv) * (taper[gidz-FDOH]));
            stf(syy, indv, ldf(syy, indv) * (taper[gidz-FDOH]));
            stf(szz, indv, ldf(szz, indv) * (taper[gidz-FDOH]));
        }
#endif
        
        if (gidz>NZS-NAB-FDOH-1){
            stf(sxy, indv, ldf(sxy, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(syz, indv, ldf(syz, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(sxz, indv, ldf(sxz, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(sxx, indv, ldf(sxx, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(syy, indv, ldf(syy, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(szz, indv, ldf(szz, indv) * (taper[NZS-FDOH-gidz-1]));
        }
        if (gidy-FDOH<NAB){
            stf(sxy, indv, ldf(sxy, indv) * (taper[gidy-FDOH]));
            stf(syz, indv, ldf(syz, indv) * (taper[gidy-FDOH]));
            stf(sxz, indv, ldf(sxz, indv) * (taper[gidy-FDOH]));
            stf(sxx, indv, ldf(sxx, indv) * (taper[gidy-FDOH]));
            stf(syy, indv, ldf(syy, indv) * (taper[gidy-FDOH]));
            stf(szz, indv, ldf(szz, indv) * (taper[gidy-FDOH]));
        }
        
        if (gidy>NYS-NAB-FDOH-1){
            stf(sxy, indv, ldf(sxy, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(syz, indv, ldf(syz, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(sxz, indv, ldf(sxz, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(sxx, indv, ldf(sxx, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(syy, indv, ldf(syy, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(szz, indv, ldf(szz, indv) * (taper[NYS-FDOH-gidy-1]));
        }
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            stf(sxy, indv, ldf(sxy, indv) * (taper[gidx-FDOH]));
            stf(syz, indv, ldf(syz, indv) * (taper[gidx-FDOH]));
            stf(sxz, indv, ldf(sxz, indv) * (taper[gidx-FDOH]));
            stf(sxx, indv, ldf(sxx, indv) * (taper[gidx-FDOH]));
            stf(syy, indv, ldf(syy, indv) * (taper[gidx-FDOH]));
            stf(szz, indv, ldf(szz, indv) * (taper[gidx-FDOH]));
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            stf(sxy, indv, ldf(sxy, indv) * (taper[NX-FDOH-gidx-1]));
            stf(syz, indv, ldf(syz, indv) * (taper[NX-FDOH-gidx-1]));
            stf(sxz, indv, ldf(sxz, indv) * (taper[NX-FDOH-gidx-1]));
            stf(sxx, indv, ldf(sxx, indv) * (taper[NX-FDOH-gidx-1]));
            stf(syy, indv, ldf(syy, indv) * (taper[NX-FDOH-gidx-1]));
            stf(szz, indv, ldf(szz, indv) * (taper[NX-FDOH-gidx-1]));
        }
#endif
    }
#endif

    }
}
