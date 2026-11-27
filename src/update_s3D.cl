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
    tdpn(lvar, it0, sx, d_vzx);
    tdpn(lvar, it0, sy, d_vzy);
    tdmn(lvar, it0, 1, d_vzz);
    BARRIER
    load_tile3(lvar, vy, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    tdpn(lvar, it0, sx, d_vyx);
    tdmn(lvar, it0, sy, d_vyy);
    tdpn(lvar, it0, 1, d_vyz);
    BARRIER
    load_tile3(lvar, vx, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    tdmn(lvar, it0, sx, d_vxx);
    tdpn(lvar, it0, sy, d_vxy);
    tdpn(lvar, it0, 1, d_vxz);
    BARRIER

#if COMM12==0
    if (zt > NZS-FDOH-1 || gidy > NYS-FDOH-1 || (gidx-offcomm) > (NX-FDOH-1-LCOMM)) return;
#else
    if (zt > NZS-FDOH-1 || gidy > NYS-FDOH-1) return;
#endif

    /* own cells and parameters: one paired load and store per array (ldn/stn, ldp3), as in 2D;
       rnd rounds where a store and reload did (FP16 2), so the results keep their bits */
    int indv0 = (gidx*NYS + gidy)*NZS + zt;
#if 1
    float l_sxx[NV], l_syy[NV], l_szz[NV], l_sxy[NV], l_syz[NV], l_sxz[NV];
    ldn(sxx, indv0, l_sxx);
    ldn(syy, indv0, l_syy);
    ldn(szz, indv0, l_szz);
    ldn(sxy, indv0, l_sxy);
    ldn(syz, indv0, l_syz);
    ldn(sxz, indv0, l_sxz);
#endif
#if 1
    float p_M[NV], p_mu[NV], p_muipjp[NV], p_mujpkp[NV], p_muipkp[NV];
    ldp3(M, gidx, gidy, zt, p_M);
    ldp3(mu, gidx, gidy, zt, p_mu);
    ldp3(muipjp, gidx, gidy, zt, p_muipjp);
    ldp3(mujpkp, gidx, gidy, zt, p_mujpkp);
    ldp3(muipkp, gidx, gidy, zt, p_muipkp);
#endif
#if LVE>0
    float p_taup[NV], p_taus[NV], p_tausipjp[NV], p_tausipkp[NV], p_tausjpkp[NV];
    ldp3(taup, gidx, gidy, zt, p_taup);
    ldp3(taus, gidx, gidy, zt, p_taus);
    ldp3(tausipjp, gidx, gidy, zt, p_tausipjp);
    ldp3(tausipkp, gidx, gidy, zt, p_tausipkp);
    ldp3(tausjpkp, gidx, gidy, zt, p_tausjpkp);
#endif
#if LVE>0
    float l_rxx[LVE][NV], l_ryy[LVE][NV], l_rzz[LVE][NV], l_rxy[LVE][NV], l_ryz[LVE][NV], l_rxz[LVE][NV];
    for (int lp=0; lp<LVE; lp++) ldn(rxx, lp*NX*NYS*NZS + indv0, l_rxx[lp]);
    for (int lp=0; lp<LVE; lp++) ldn(ryy, lp*NX*NYS*NZS + indv0, l_ryy[lp]);
    for (int lp=0; lp<LVE; lp++) ldn(rzz, lp*NX*NYS*NZS + indv0, l_rzz[lp]);
    for (int lp=0; lp<LVE; lp++) ldn(rxy, lp*NX*NYS*NZS + indv0, l_rxy[lp]);
    for (int lp=0; lp<LVE; lp++) ldn(ryz, lp*NX*NYS*NZS + indv0, l_ryz[lp]);
    for (int lp=0; lp<LVE; lp++) ldn(rxz, lp*NX*NYS*NZS + indv0, l_rxz[lp]);
#endif
#if LVE>0
    /* Unrolled with memory variables: rolled, the 9 derivative arrays went to the stack at FP16 1/2
       (REG 145 STACK 72). A6000, 3-D 192^3 x 1000 forward: FP16 2 3.87 to 3.38 s, FP16 1 6.85 to 6.67 s.
       Elastic (LVE 0) stays rolled: unrolled it was slower (FP16 1 2.45 to 2.98 s). */
    #pragma unroll
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
        
        fipjp=p_muipjp[c];
        fjpkp=p_mujpkp[c];
        fipkp=p_muipkp[c];
        g=p_M[c];
        f=2.0*p_mu[c];
        
#else
        
        lM=p_M[c];
        lmu=p_mu[c];
        lmuipkp=p_muipkp[c];
        lmuipjp=p_muipjp[c];
        lmujpkp=p_mujpkp[c];
        ltaup=p_taup[c];
        ltaus=p_taus[c];
        ltausipkp=p_tausipkp[c];
        ltausipjp=p_tausipjp[c];
        ltausjpkp=p_tausjpkp[c];
        
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

        l_sxy[c] = rnd(l_sxy[c] + ((fipjp*vxyyx)));
        l_syz[c] = rnd(l_syz[c] + ((fjpkp*vyzzy)));
        l_sxz[c] = rnd(l_sxz[c] + ((fipkp*vxzzx)));
        l_sxx[c] = rnd(l_sxx[c] + (((g*vxxyyzz)-(f*vyyzz))));
        l_syy[c] = rnd(l_syy[c] + (((g*vxxyyzz)-(f*vxxzz))));
        l_szz[c] = rnd(l_szz[c] + (((g*vxxyyzz)-(f*vxxyy))));
        
#else
        
        /* computing sums of the old memory variables */
        sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
        for (l=0;l<LVE;l++){
            indr = l*NX*NYS*NZS + gidx*NYS*NZS + gidy*NZS +gidz;
            sumrxy+=l_rxy[l][c];
            sumryz+=l_ryz[l][c];
            sumrxz+=l_rxz[l][c];
            sumrxx+=l_rxx[l][c];
            sumryy+=l_ryy[l][c];
            sumrzz+=l_rzz[l][c];
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
            l_rxy[l][c] = rnd(b*(l_rxy[l][c]*cv-leta[l]*(dipjp*vxyyx)));
            l_ryz[l][c] = rnd(b*(l_ryz[l][c]*cv-leta[l]*(djpkp*vyzzy)));
            l_rxz[l][c] = rnd(b*(l_rxz[l][c]*cv-leta[l]*(dipkp*vxzzx)));
            l_rxx[l][c] = rnd(b*(l_rxx[l][c]*cv-leta[l]*((e*vxxyyzz)-(d*vyyzz))));
            l_ryy[l][c] = rnd(b*(l_ryy[l][c]*cv-leta[l]*((e*vxxyyzz)-(d*vxxzz))));
            l_rzz[l][c] = rnd(b*(l_rzz[l][c]*cv-leta[l]*((e*vxxyyzz)-(d*vxxyy))));

            sumrxy+=l_rxy[l][c];
            sumryz+=l_ryz[l][c];
            sumrxz+=l_rxz[l][c];
            sumrxx+=l_rxx[l][c];
            sumryy+=l_ryy[l][c];
            sumrzz+=l_rzz[l][c];
        }

        /* and now the components of the stress tensor are
         completely updated */
        l_sxy[c] = rnd(l_sxy[c] + (lsxy+(DT2*sumrxy)));
        l_syz[c] = rnd(l_syz[c] + (lsyz+(DT2*sumryz)));
        l_sxz[c] = rnd(l_sxz[c] + (lsxz+(DT2*sumrxz)));
        l_sxx[c] = rnd(l_sxx[c] + (lsxx+(DT2*sumrxx)));
        l_syy[c] = rnd(l_syy[c] + (lsyy+(DT2*sumryy)));
        l_szz[c] = rnd(l_szz[c] + (lszz+(DT2*sumrzz)));

#endif
    }

// Absorbing boundary    
#if ABS_TYPE==2
    {
        
#if FREESURF==0
        if (gidz-FDOH<NAB){
            l_sxy[c] = rnd(l_sxy[c] * (taper[gidz-FDOH]));
            l_syz[c] = rnd(l_syz[c] * (taper[gidz-FDOH]));
            l_sxz[c] = rnd(l_sxz[c] * (taper[gidz-FDOH]));
            l_sxx[c] = rnd(l_sxx[c] * (taper[gidz-FDOH]));
            l_syy[c] = rnd(l_syy[c] * (taper[gidz-FDOH]));
            l_szz[c] = rnd(l_szz[c] * (taper[gidz-FDOH]));
        }
#endif
        
        if (gidz>NZS-NAB-FDOH-1){
            l_sxy[c] = rnd(l_sxy[c] * (taper[NZS-FDOH-gidz-1]));
            l_syz[c] = rnd(l_syz[c] * (taper[NZS-FDOH-gidz-1]));
            l_sxz[c] = rnd(l_sxz[c] * (taper[NZS-FDOH-gidz-1]));
            l_sxx[c] = rnd(l_sxx[c] * (taper[NZS-FDOH-gidz-1]));
            l_syy[c] = rnd(l_syy[c] * (taper[NZS-FDOH-gidz-1]));
            l_szz[c] = rnd(l_szz[c] * (taper[NZS-FDOH-gidz-1]));
        }
        if (gidy-FDOH<NAB){
            l_sxy[c] = rnd(l_sxy[c] * (taper[gidy-FDOH]));
            l_syz[c] = rnd(l_syz[c] * (taper[gidy-FDOH]));
            l_sxz[c] = rnd(l_sxz[c] * (taper[gidy-FDOH]));
            l_sxx[c] = rnd(l_sxx[c] * (taper[gidy-FDOH]));
            l_syy[c] = rnd(l_syy[c] * (taper[gidy-FDOH]));
            l_szz[c] = rnd(l_szz[c] * (taper[gidy-FDOH]));
        }
        
        if (gidy>NYS-NAB-FDOH-1){
            l_sxy[c] = rnd(l_sxy[c] * (taper[NYS-FDOH-gidy-1]));
            l_syz[c] = rnd(l_syz[c] * (taper[NYS-FDOH-gidy-1]));
            l_sxz[c] = rnd(l_sxz[c] * (taper[NYS-FDOH-gidy-1]));
            l_sxx[c] = rnd(l_sxx[c] * (taper[NYS-FDOH-gidy-1]));
            l_syy[c] = rnd(l_syy[c] * (taper[NYS-FDOH-gidy-1]));
            l_szz[c] = rnd(l_szz[c] * (taper[NYS-FDOH-gidy-1]));
        }
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            l_sxy[c] = rnd(l_sxy[c] * (taper[gidx-FDOH]));
            l_syz[c] = rnd(l_syz[c] * (taper[gidx-FDOH]));
            l_sxz[c] = rnd(l_sxz[c] * (taper[gidx-FDOH]));
            l_sxx[c] = rnd(l_sxx[c] * (taper[gidx-FDOH]));
            l_syy[c] = rnd(l_syy[c] * (taper[gidx-FDOH]));
            l_szz[c] = rnd(l_szz[c] * (taper[gidx-FDOH]));
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            l_sxy[c] = rnd(l_sxy[c] * (taper[NX-FDOH-gidx-1]));
            l_syz[c] = rnd(l_syz[c] * (taper[NX-FDOH-gidx-1]));
            l_sxz[c] = rnd(l_sxz[c] * (taper[NX-FDOH-gidx-1]));
            l_sxx[c] = rnd(l_sxx[c] * (taper[NX-FDOH-gidx-1]));
            l_syy[c] = rnd(l_syy[c] * (taper[NX-FDOH-gidx-1]));
            l_szz[c] = rnd(l_szz[c] * (taper[NX-FDOH-gidx-1]));
        }
#endif
    }
#endif

    }
#if 1
    stn(sxx, indv0, l_sxx);
    stn(syy, indv0, l_syy);
    stn(szz, indv0, l_szz);
    stn(sxy, indv0, l_sxy);
    stn(syz, indv0, l_syz);
    stn(sxz, indv0, l_sxz);
#endif
#if LVE>0
    for (int lp=0; lp<LVE; lp++) stn(rxx, lp*NX*NYS*NZS + indv0, l_rxx[lp]);
    for (int lp=0; lp<LVE; lp++) stn(ryy, lp*NX*NYS*NZS + indv0, l_ryy[lp]);
    for (int lp=0; lp<LVE; lp++) stn(rzz, lp*NX*NYS*NZS + indv0, l_rzz[lp]);
    for (int lp=0; lp<LVE; lp++) stn(rxy, lp*NX*NYS*NZS + indv0, l_rxy[lp]);
    for (int lp=0; lp<LVE; lp++) stn(ryz, lp*NX*NYS*NZS + indv0, l_ryz[lp]);
    for (int lp=0; lp<LVE; lp++) stn(rxz, lp*NX*NYS*NZS + indv0, l_rxz[lp]);
#endif
}
