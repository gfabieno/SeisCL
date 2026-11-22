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

/*Adjoint update of the stresses in 3D*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */





#ifndef SURFROWS
/* Rows below a stress-image free surface (FREESURF 1) whose moduli gradient
   is computed in the direct form by surface_grad (surface3D_adj.cl). */
#define SURFROWS (FDOH+1)
#endif

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

FUNDEF void update_adjs(int offcomm,
                          GLOBARG const STO * RESTRICT vx,        GLOBARG const STO * RESTRICT vy,        GLOBARG const STO * RESTRICT vz,
                          GLOBARG STO * RESTRICT sxx,             GLOBARG STO * RESTRICT syy,             GLOBARG STO * RESTRICT szz,
                          GLOBARG STO * RESTRICT sxy,             GLOBARG STO * RESTRICT syz,             GLOBARG STO * RESTRICT sxz,
                          GLOBARG const STO * RESTRICT vxbnd,     GLOBARG const STO * RESTRICT vybnd,     GLOBARG const STO * RESTRICT vzbnd,
                          GLOBARG const STO * RESTRICT sxxbnd,    GLOBARG const STO * RESTRICT syybnd,    GLOBARG const STO * RESTRICT szzbnd,
                          GLOBARG const STO * RESTRICT sxybnd,    GLOBARG const STO * RESTRICT syzbnd,    GLOBARG const STO * RESTRICT sxzbnd,
                          GLOBARG const STO * RESTRICT vxr,       GLOBARG const STO * RESTRICT vyr,       GLOBARG const STO * RESTRICT vzr,
                          GLOBARG STO * RESTRICT sxxr,            GLOBARG STO * RESTRICT syyr,            GLOBARG STO * RESTRICT szzr,
                          GLOBARG STO * RESTRICT sxyr,            GLOBARG STO * RESTRICT syzr,            GLOBARG STO * RESTRICT sxzr,
                          GLOBARG STO * RESTRICT rxx,             GLOBARG STO * RESTRICT ryy,             GLOBARG STO * RESTRICT rzz,
                          GLOBARG STO * RESTRICT rxy,             GLOBARG STO * RESTRICT ryz,             GLOBARG STO * RESTRICT rxz,
                          GLOBARG const STO * RESTRICT rxxr,      GLOBARG const STO * RESTRICT ryyr,      GLOBARG const STO * RESTRICT rzzr,
                          GLOBARG const STO * RESTRICT rxyr,      GLOBARG const STO * RESTRICT ryzr,      GLOBARG const STO * RESTRICT rxzr,
                          GLOBARG const STO * RESTRICT M,         GLOBARG const STO * RESTRICT mu,        GLOBARG const STO * RESTRICT muipjp,
                          GLOBARG const STO * RESTRICT mujpkp,    GLOBARG const STO * RESTRICT muipkp,
                          GLOBARG const STO * RESTRICT taus,      GLOBARG const STO * RESTRICT tausipjp,  GLOBARG const STO * RESTRICT tausjpkp,
                          GLOBARG const STO * RESTRICT tausipkp,  GLOBARG const STO * RESTRICT taup,      GLOBARG const float * RESTRICT eta,
                          GLOBARG const float * RESTRICT taper,
                          GLOBARG const float * RESTRICT K_x,       GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                          GLOBARG const float * RESTRICT K_x_half,  GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                          GLOBARG const float * RESTRICT K_y,       GLOBARG const float * RESTRICT a_y,       GLOBARG const float * RESTRICT b_y,
                          GLOBARG const float * RESTRICT K_y_half,  GLOBARG const float * RESTRICT a_y_half,  GLOBARG const float * RESTRICT b_y_half,
                          GLOBARG const float * RESTRICT K_z,       GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                          GLOBARG const float * RESTRICT K_z_half,  GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                          GLOBARG const float * RESTRICT Ka_x,      GLOBARG const float * RESTRICT Ka_x_half,
                          GLOBARG const float * RESTRICT Ka_y,      GLOBARG const float * RESTRICT Ka_y_half,
                          GLOBARG const float * RESTRICT Ka_z,      GLOBARG const float * RESTRICT Ka_z_half,
                          GLOBARG float * RESTRICT psi_vx_x,        GLOBARG float * RESTRICT psi_vx_y,        GLOBARG float * RESTRICT psi_vx_z,
                          GLOBARG float * RESTRICT psi_vy_x,        GLOBARG float * RESTRICT psi_vy_y,        GLOBARG float * RESTRICT psi_vy_z,
                          GLOBARG float * RESTRICT psi_vz_x,        GLOBARG float * RESTRICT psi_vz_y,        GLOBARG float * RESTRICT psi_vz_z,
                          GLOBARG float * RESTRICT psi_sxx_xr,      GLOBARG float * RESTRICT psi_sxy_xr,      GLOBARG float * RESTRICT psi_sxz_xr,
                          GLOBARG float * RESTRICT psi_sxy_yr,      GLOBARG float * RESTRICT psi_syy_yr,      GLOBARG float * RESTRICT psi_syz_yr,
                          GLOBARG float * RESTRICT psi_sxz_zr,      GLOBARG float * RESTRICT psi_syz_zr,      GLOBARG float * RESTRICT psi_szz_zr,
                          GLOBARG const STO * RESTRICT cpf1,
                          GLOBARG const STO * RESTRICT cpf2,
                          GLOBARG const STO * RESTRICT cpf3,
                          GLOBARG const STO * RESTRICT cpf4,
                          GLOBARG const STO * RESTRICT cpf5,
                          GLOBARG const STO * RESTRICT cpf6,
                          GLOBARG const STO * RESTRICT cpf7,
                          GLOBARG const STO * RESTRICT cpf8,
                          GLOBARG const STO * RESTRICT cpf9,
                          GLOBARG const float * RESTRICT gradrho,   GLOBARG float * RESTRICT gradM,           GLOBARG float * RESTRICT gradmu,
                          GLOBARG float * RESTRICT gradmuipkp,      GLOBARG float * RESTRICT gradmuipjp,      GLOBARG float * RESTRICT gradmujpkp,
                          GLOBARG const float * RESTRICT gradtaup,  GLOBARG const float * RESTRICT gradtaus,  GLOBARG const float * RESTRICT gradsrc,
                          GLOBARG const float * RESTRICT Hrho,      GLOBARG float * RESTRICT HM,              GLOBARG float * RESTRICT Hmu,
                          GLOBARG const float * RESTRICT Htaup,     GLOBARG const float * RESTRICT Htaus,     GLOBARG const float * RESTRICT Hsrc,
                          GLOBARG const float * RESTRICT src,       GLOBARG const float * RESTRICT src_pos,
                          int nsrc,                                 int nt,
                          int src_scale,
                          GLOBARG const float * RESTRICT pout,      GLOBARG const float * RESTRICT rec_pos,
                          int nrec,                                 int res_scale,
                          LOCARG)
{
    LOCDEF

    
    int m;
    float vxyyx,vyzzy,vxzzx,vxxyyzz,vyyzz,vxxzz,vxxyy;
    float vxyyxr,vyzzyr,vxzzxr,vxxyyzzr,vyyzzr,vxxzzr,vxxyyr;
    float fipjp, fjpkp, fipkp, f, g;
    float lM, lmu;
    #if LVE>0
    float sumrxy,sumryz,sumrxz,sumrxx,sumryy,sumrzz;
    float b,cv,e,d,dipjp,djpkp,dipkp;
    int l, indr;;
    float ltaup, ltaus, ltausipjp, ltausipkp, ltausjpkp, lmuipjp, lmuipkp, lmujpkp;
    float leta[LVE];
    #endif
    float lsxx, lsyy, lszz, lsxy, lsxz, lsyz;
    
    
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
    float d_vxx[NV], d_vxxr[NV], d_vxy[NV], d_vxyr[NV], d_vxz[NV], d_vxzr[NV], d_vyx[NV], d_vyxr[NV], d_vyy[NV], d_vyyr[NV], d_vyz[NV], d_vyzr[NV], d_vzx[NV], d_vzxr[NV], d_vzy[NV], d_vzyr[NV], d_vzz[NV], d_vzzr[NV];
#if BACK_PROP_TYPE==1 && !(CPMLINL && LVE==0)
    load_tile3(lvar, vz, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vzx[c] = tdp(lvar, it0+c, sx);
    #pragma unroll
    for (int c=0; c<NV; c++) d_vzy[c] = tdp(lvar, it0+c, sy);
    #pragma unroll
    for (int c=0; c<NV; c++) d_vzz[c] = tdm(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, vy, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vyx[c] = tdp(lvar, it0+c, sx);
    #pragma unroll
    for (int c=0; c<NV; c++) d_vyy[c] = tdm(lvar, it0+c, sy);
    #pragma unroll
    for (int c=0; c<NV; c++) d_vyz[c] = tdp(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, vx, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vxx[c] = tdm(lvar, it0+c, sx);
    #pragma unroll
    for (int c=0; c<NV; c++) d_vxy[c] = tdp(lvar, it0+c, sy);
    #pragma unroll
    for (int c=0; c<NV; c++) d_vxz[c] = tdp(lvar, it0+c, 1);
    BARRIER
#endif
#if CPMLINL
    /* the adjoint derivatives are computed in the body (CPI_AXES) */
#elif ABS_TYPE==1 && BACK_PROP_TYPE==1
    load_tile3_cpf(lvar, cpf1, vxr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vxxr[c] = tdm(lvar, it0+c, sx);
    BARRIER
    load_tile3_cpf(lvar, cpf2, vxr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vxyr[c] = tdp(lvar, it0+c, sy);
    BARRIER
    load_tile3_cpf(lvar, cpf3, vxr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vxzr[c] = tdp(lvar, it0+c, 1);
    BARRIER
    load_tile3_cpf(lvar, cpf4, vyr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vyxr[c] = tdp(lvar, it0+c, sx);
    BARRIER
    load_tile3_cpf(lvar, cpf5, vyr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vyyr[c] = tdm(lvar, it0+c, sy);
    BARRIER
    load_tile3_cpf(lvar, cpf6, vyr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vyzr[c] = tdp(lvar, it0+c, 1);
    BARRIER
    load_tile3_cpf(lvar, cpf7, vzr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vzxr[c] = tdp(lvar, it0+c, sx);
    BARRIER
    load_tile3_cpf(lvar, cpf8, vzr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vzyr[c] = tdp(lvar, it0+c, sy);
    BARRIER
    load_tile3_cpf(lvar, cpf9, vzr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vzzr[c] = tdm(lvar, it0+c, 1);
    BARRIER
#else
    load_tile3(lvar, vzr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vzxr[c] = tdp(lvar, it0+c, sx);
    #pragma unroll
    for (int c=0; c<NV; c++) d_vzyr[c] = tdp(lvar, it0+c, sy);
    #pragma unroll
    for (int c=0; c<NV; c++) d_vzzr[c] = tdm(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, vyr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vyxr[c] = tdp(lvar, it0+c, sx);
    #pragma unroll
    for (int c=0; c<NV; c++) d_vyyr[c] = tdm(lvar, it0+c, sy);
    #pragma unroll
    for (int c=0; c<NV; c++) d_vyzr[c] = tdp(lvar, it0+c, 1);
    BARRIER
    load_tile3(lvar, vxr, tile_z0, tile_y0, tile_x0, tz, ty, tx);
    BARRIER
    #pragma unroll
    for (int c=0; c<NV; c++) d_vxxr[c] = tdm(lvar, it0+c, sx);
    #pragma unroll
    for (int c=0; c<NV; c++) d_vxyr[c] = tdp(lvar, it0+c, sy);
    #pragma unroll
    for (int c=0; c<NV; c++) d_vxzr[c] = tdp(lvar, it0+c, 1);
    BARRIER
#endif

#if COMM12==0
    if (zt > NZS-FDOH-1 || gidy > NYS-FDOH-1 || (gidx-offcomm) > (NX-FDOH-1-LCOMM)) return;
#else
    if (zt > NZS-FDOH-1 || gidy > NYS-FDOH-1) return;
#endif

    #pragma unroll
    for (int c=0; c<NV; c++){
        int gidz = zt + c;
        if (!INTERIOR_Z(gidz)) continue;              // halo cell of an aligned pair
        int indp = ((gidx-FDOH)*(NYS-2*FDOH) + (gidy-FDOH))*(NZS-2*FDOH) + (gidz-FDOH);
        int indv = (gidx*NYS + gidy)*NZS + gidz;
#if CPMLINL
        CPI_AXES
        #define FL(f) ((TPTR)(f))
#endif
#if !CPMLINL
        float vxx = d_vxx[c];
        float vxxr = d_vxxr[c];
        float vxy = d_vxy[c];
        float vxyr = d_vxyr[c];
        float vxz = d_vxz[c];
        float vxzr = d_vxzr[c];
        float vyx = d_vyx[c];
        float vyxr = d_vyxr[c];
        float vyy = d_vyy[c];
        float vyyr = d_vyyr[c];
        float vyz = d_vyz[c];
        float vyzr = d_vyzr[c];
        float vzx = d_vzx[c];
        float vzxr = d_vzxr[c];
        float vzy = d_vzy[c];
        float vzyr = d_vzyr[c];
        float vzz = d_vzz[c];
        float vzzr = d_vzzr[c];
#elif LVE>0
        float vxx = d_vxx[c];
        float vxy = d_vxy[c];
        float vxz = d_vxz[c];
        float vyx = d_vyx[c];
        float vyy = d_vyy[c];
        float vyz = d_vyz[c];
        float vzx = d_vzx[c];
        float vzy = d_vzy[c];
        float vzz = d_vzz[c];
        /* exact CPML adjoint of the velocity derivatives, in place (header_vec.cl, CPMLINL;
           replaces cpml_adj_v and cpf1..9 of cpml_adj3D.cl, same filters) */
        float vxxr = cpi_tdm(vxr, it0, sx, CPI_X(psi_sxx_xr), 0, Ka_x_half, b_x_half);
        float vxyr = cpi_tdp(vxr, it0, sy, CPI_Y(psi_sxy_yr), 1, Ka_y, b_y);
        float vxzr = cpi_tdp(vxr, it0, 1,  CPI_Z(psi_sxz_zr), 1, Ka_z, b_z);
        float vyxr = cpi_tdp(vyr, it0, sx, CPI_X(psi_sxy_xr), 1, Ka_x, b_x);
        float vyyr = cpi_tdm(vyr, it0, sy, CPI_Y(psi_syy_yr), 0, Ka_y_half, b_y_half);
        float vyzr = cpi_tdp(vyr, it0, 1,  CPI_Z(psi_syz_zr), 1, Ka_z, b_z);
        float vzxr = cpi_tdp(vzr, it0, sx, CPI_X(psi_sxz_xr), 1, Ka_x, b_x);
        float vzyr = cpi_tdp(vzr, it0, sy, CPI_Y(psi_syz_yr), 1, Ka_y, b_y);
        float vzzr = cpi_tdm(vzr, it0, 1,  CPI_Z(psi_szz_zr), 0, Ka_z_half, b_z_half);
#endif




// Read model parameters into local memory
#if LVE==0
    lM=ldf(M, indp);
    lmu=ldf(mu, indp);
    fipjp=ldf(muipjp, indp);
    fjpkp=ldf(mujpkp, indp);
    fipkp=ldf(muipkp, indp);
    g=lM;
    f=2.0*lmu;


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


// Backpropagate the forward stresses
#if BACK_PROP_TYPE==1
    {
#if LVE==0

#if CPMLINL
        /* derivatives computed where they are used, to keep few of them live */
        vxyyx=tdp(FL(vx), it0, sy)+tdp(FL(vy), it0, sx);
        stf(sxy, indv, ldf(sxy, indv) - ((fipjp*vxyyx)));
        vyzzy=tdp(FL(vy), it0, 1)+tdp(FL(vz), it0, sy);
        stf(syz, indv, ldf(syz, indv) - ((fjpkp*vyzzy)));
        vxzzx=tdp(FL(vx), it0, 1)+tdp(FL(vz), it0, sx);
        stf(sxz, indv, ldf(sxz, indv) - ((fipkp*vxzzx)));
        {
        float vxx = tdm(FL(vx), it0, sx), vyy = tdm(FL(vy), it0, sy), vzz = tdm(FL(vz), it0, 1);
        vxxyyzz=vxx+vyy+vzz;
        vyyzz=vyy+vzz;
        vxxzz=vxx+vzz;
        vxxyy=vxx+vyy;
        }
        stf(sxx, indv, ldf(sxx, indv) - (((g*vxxyyzz)-(f*vyyzz)) ));
        stf(syy, indv, ldf(syy, indv) - (((g*vxxyyzz)-(f*vxxzz)) ));
        stf(szz, indv, ldf(szz, indv) - (((g*vxxyyzz)-(f*vxxyy)) ));
#else
        vxyyx=vxy+vyx;
        vyzzy=vyz+vzy;
        vxzzx=vxz+vzx;
        vxxyyzz=vxx+vyy+vzz;
        vyyzz=vyy+vzz;
        vxxzz=vxx+vzz;
        vxxyy=vxx+vyy;

        stf(sxy, indv, ldf(sxy, indv) - ((fipjp*vxyyx)));
        stf(syz, indv, ldf(syz, indv) - ((fjpkp*vyzzy)));
        stf(sxz, indv, ldf(sxz, indv) - ((fipkp*vxzzx)));
        stf(sxx, indv, ldf(sxx, indv) - (((g*vxxyyzz)-(f*vyyzz)) ));
        stf(syy, indv, ldf(syy, indv) - (((g*vxxyyzz)-(f*vxxzz)) ));
        stf(szz, indv, ldf(szz, indv) - (((g*vxxyyzz)-(f*vxxyy)) ));
#endif

// Backpropagation is not stable for viscoelastic wave equation
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
            //Those variables change sign for reverse time
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

        stf(sxy, indv, ldf(sxy, indv) - (lsxy+(DT2*sumrxy)));
        stf(syz, indv, ldf(syz, indv) - (lsyz+(DT2*sumryz)));
        stf(sxz, indv, ldf(sxz, indv) - (lsxz+(DT2*sumrxz)));
        stf(sxx, indv, ldf(sxx, indv) - (lsxx+(DT2*sumrxx)));
        stf(syy, indv, ldf(syy, indv) - (lsyy+(DT2*sumryy)));
        stf(szz, indv, ldf(szz, indv) - (lszz+(DT2*sumrzz)));

#endif

        m=inject_ind(gidz, gidy, gidx);
        if (m!=-1){
            stf(sxx, indv, ldf(sxxbnd, m));
            stf(syy, indv, ldf(syybnd, m));
            stf(szz, indv, ldf(szzbnd, m));
            stf(sxy, indv, ldf(sxybnd, m));
            stf(syz, indv, ldf(syzbnd, m));
            stf(sxz, indv, ldf(sxzbnd, m));
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

            psi_vx_z[I_psi_vx_z(k,j,i)] = b_z_half[ind] * psi_vx_z[I_psi_vx_z(k,j,i)] + a_z_half[ind] * vxzr;
            vxzr = vxzr / K_z_half[ind] + psi_vx_z[I_psi_vx_z(k,j,i)];
            psi_vy_z[I_psi_vy_z(k,j,i)] = b_z_half[ind] * psi_vy_z[I_psi_vy_z(k,j,i)] + a_z_half[ind] * vyzr;
            vyzr = vyzr / K_z_half[ind] + psi_vy_z[I_psi_vy_z(k,j,i)];
            psi_vz_z[I_psi_vz_z(k,j,i)] = b_z[ind+1] * psi_vz_z[I_psi_vz_z(k,j,i)] + a_z[ind+1] * vzzr;
            vzzr = vzzr / K_z[ind+1] + psi_vz_z[I_psi_vz_z(k,j,i)];

        }

    #if FREESURF==0
        else if (gidz-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;


            psi_vx_z[I_psi_vx_z(k,j,i)] = b_z_half[k] * psi_vx_z[I_psi_vx_z(k,j,i)] + a_z_half[k] * vxzr;
            vxzr = vxzr / K_z_half[k] + psi_vx_z[I_psi_vx_z(k,j,i)];
            psi_vy_z[I_psi_vy_z(k,j,i)] = b_z_half[k] * psi_vy_z[I_psi_vy_z(k,j,i)] + a_z_half[k] * vyzr;
            vyzr = vyzr / K_z_half[k] + psi_vy_z[I_psi_vy_z(k,j,i)];
            psi_vz_z[I_psi_vz_z(k,j,i)] = b_z[k] * psi_vz_z[I_psi_vz_z(k,j,i)] + a_z[k] * vzzr;
            vzzr = vzzr / K_z[k] + psi_vz_z[I_psi_vz_z(k,j,i)];


        }
    #endif

        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_vx_y[I_psi_vx_y(k,j,i)] = b_y_half[j] * psi_vx_y[I_psi_vx_y(k,j,i)] + a_y_half[j] * vxyr;
            vxyr = vxyr / K_y_half[j] + psi_vx_y[I_psi_vx_y(k,j,i)];
            psi_vy_y[I_psi_vy_y(k,j,i)] = b_y[j] * psi_vy_y[I_psi_vy_y(k,j,i)] + a_y[j] * vyyr;
            vyyr = vyyr / K_y[j] + psi_vy_y[I_psi_vy_y(k,j,i)];
            psi_vz_y[I_psi_vz_y(k,j,i)] = b_y_half[j] * psi_vz_y[I_psi_vz_y(k,j,i)] + a_y_half[j] * vzyr;
            vzyr = vzyr / K_y_half[j] + psi_vz_y[I_psi_vz_y(k,j,i)];

        }

        else if (gidy>NYS-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy - NYS+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-j;


            psi_vx_y[I_psi_vx_y(k,j,i)] = b_y_half[ind] * psi_vx_y[I_psi_vx_y(k,j,i)] + a_y_half[ind] * vxyr;
            vxyr = vxyr / K_y_half[ind] + psi_vx_y[I_psi_vx_y(k,j,i)];
            psi_vy_y[I_psi_vy_y(k,j,i)] = b_y[ind+1] * psi_vy_y[I_psi_vy_y(k,j,i)] + a_y[ind+1] * vyyr;
            vyyr = vyyr / K_y[ind+1] + psi_vy_y[I_psi_vy_y(k,j,i)];
            psi_vz_y[I_psi_vz_y(k,j,i)] = b_y_half[ind] * psi_vz_y[I_psi_vz_y(k,j,i)] + a_y_half[ind] * vzyr;
            vzyr = vzyr / K_y_half[ind] + psi_vz_y[I_psi_vz_y(k,j,i)];


        }
    #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_vx_x[I_psi_vx_x(k,j,i)] = b_x[i] * psi_vx_x[I_psi_vx_x(k,j,i)] + a_x[i] * vxxr;
            vxxr = vxxr / K_x[i] + psi_vx_x[I_psi_vx_x(k,j,i)];
            psi_vy_x[I_psi_vy_x(k,j,i)] = b_x_half[i] * psi_vy_x[I_psi_vy_x(k,j,i)] + a_x_half[i] * vyxr;
            vyxr = vyxr / K_x_half[i] + psi_vy_x[I_psi_vy_x(k,j,i)];
            psi_vz_x[I_psi_vz_x(k,j,i)] = b_x_half[i] * psi_vz_x[I_psi_vz_x(k,j,i)] + a_x_half[i] * vzxr;
            vzxr = vzxr / K_x_half[i] + psi_vz_x[I_psi_vz_x(k,j,i)];


        }
    #endif

    #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){

            i =gidx - NX+NAB+FDOH+NAB;
            j =gidy-FDOH;
            k =gidz-FDOH;
            ind=2*NAB-1-i;


            psi_vx_x[I_psi_vx_x(k,j,i)] = b_x[ind+1] * psi_vx_x[I_psi_vx_x(k,j,i)] + a_x[ind+1] * vxxr;
            vxxr = vxxr /K_x[ind+1] + psi_vx_x[I_psi_vx_x(k,j,i)];
            psi_vy_x[I_psi_vy_x(k,j,i)] = b_x_half[ind] * psi_vy_x[I_psi_vy_x(k,j,i)] + a_x_half[ind] * vyxr;
            vyxr = vyxr  /K_x_half[ind] + psi_vy_x[I_psi_vy_x(k,j,i)];
            psi_vz_x[I_psi_vz_x(k,j,i)] = b_x_half[ind] * psi_vz_x[I_psi_vz_x(k,j,i)] + a_x_half[ind] * vzxr;
            vzxr = vzxr / K_x_half[ind]  +psi_vz_x[I_psi_vz_x(k,j,i)];


        }
    #endif

    }
#endif

// Update adjoint stresses
    {
#if LVE==0

#if CPMLINL
    /* exact CPML adjoint of the velocity derivatives, in place (header_vec.cl, CPMLINL;
       replaces cpml_adj_v and cpf1..9 of cpml_adj3D.cl, same filters), computed where used */
    vxyyxr=cpi_tdp(vxr, it0, sy, CPI_Y(psi_sxy_yr), 1, Ka_y, b_y)
          +cpi_tdp(vyr, it0, sx, CPI_X(psi_sxy_xr), 1, Ka_x, b_x);
    vyzzyr=cpi_tdp(vyr, it0, 1,  CPI_Z(psi_syz_zr), 1, Ka_z, b_z)
          +cpi_tdp(vzr, it0, sy, CPI_Y(psi_syz_yr), 1, Ka_y, b_y);
    vxzzxr=cpi_tdp(vxr, it0, 1,  CPI_Z(psi_sxz_zr), 1, Ka_z, b_z)
          +cpi_tdp(vzr, it0, sx, CPI_X(psi_sxz_xr), 1, Ka_x, b_x);
    {
    float vxxr = cpi_tdm(vxr, it0, sx, CPI_X(psi_sxx_xr), 0, Ka_x_half, b_x_half);
    float vyyr = cpi_tdm(vyr, it0, sy, CPI_Y(psi_syy_yr), 0, Ka_y_half, b_y_half);
    float vzzr = cpi_tdm(vzr, it0, 1,  CPI_Z(psi_szz_zr), 0, Ka_z_half, b_z_half);
    vxxyyzzr=vxxr+vyyr+vzzr;
    vyyzzr=vyyr+vzzr;
    vxxzzr=vxxr+vzzr;
    vxxyyr=vxxr+vyyr;
    }
#else
    vxyyxr=vxyr+vyxr;
    vyzzyr=vyzr+vzyr;
    vxzzxr=vxzr+vzxr;
    vxxyyzzr=vxxr+vyyr+vzzr;
    vyyzzr=vyyr+vzzr;
    vxxzzr=vxxr+vzzr;
    vxxyyr=vxxr+vyyr;
#endif

    lsxy=(fipjp*vxyyxr);
    lsyz=(fjpkp*vyzzyr);
    lsxz=(fipkp*vxzzxr);
    lsxx=((g*vxxyyzzr)-(f*vyyzzr));
    lsyy=((g*vxxyyzzr)-(f*vxxzzr));
    lszz=((g*vxxyyzzr)-(f*vxxyyr));

    stf(sxyr, indv, ldf(sxyr, indv) + (lsxy));
    stf(syzr, indv, ldf(syzr, indv) + (lsyz));
    stf(sxzr, indv, ldf(sxzr, indv) + (lsxz));
    stf(sxxr, indv, ldf(sxxr, indv) + (lsxx));
    stf(syyr, indv, ldf(syyr, indv) + (lsyy));
    stf(szzr, indv, ldf(szzr, indv) + (lszz));


#else

    /* computing sums of the old memory variables */
    sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
    for (l=0;l<LVE;l++){
        indr = l*NX*NYS*NZS + gidx*NYS*NZS + gidy*NZS +gidz;
        sumrxy+=ldf(rxyr, indr);
        sumryz+=ldf(ryzr, indr);
        sumrxz+=ldf(rxzr, indr);
        sumrxx+=ldf(rxxr, indr);
        sumryy+=ldf(ryyr, indr);
        sumrzz+=ldf(rzzr, indr);
    }

    vxyyxr=vxyr+vyxr;
    vyzzyr=vyzr+vzyr;
    vxzzxr=vxzr+vzxr;
    vxxyyzzr=vxxr+vyyr+vzzr;
    vyyzzr=vyyr+vzzr;
    vxxzzr=vxxr+vzzr;
    vxxyyr=vxxr+vyyr;

    lsxy=(fipjp*vxyyxr)+(DT2*sumrxy);
    lsyz=(fjpkp*vyzzyr)+(DT2*sumryz);
    lsxz=(fipkp*vxzzxr)+(DT2*sumrxz);
    lsxx=((g*vxxyyzzr)-(f*vyyzzr))+(DT2*sumrxx);
    lsyy=((g*vxxyyzzr)-(f*vxxzzr))+(DT2*sumryy);
    lszz=((g*vxxyyzzr)-(f*vxxyyr))+(DT2*sumrzz);

    sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
    for (l=0;l<LVE;l++){
        b=1.0/(1.0+(leta[l]*0.5));
        cv=1.0-(leta[l]*0.5);
        indr = l*NX*NYS*NZS + gidx*NYS*NZS + gidy*NZS +gidz;

        /* Under back_prop_type 2, rxy and rxyr are the SAME buffer (clprogram.c binds
           every <var>r argument to the forward buffer). rxyr is const RESTRICT, so a
           read of rxyr after the write to rxy may return the OLD value (CUDA read-only
           cache): the stress then got DT2*(sum_old+sum_old) instead of
           DT2*(sum_old+sum_new), an O(dt) adjoint attenuation error (3-D visco 3-14 %).
           Keep the new value in a register, as update_adjs2D.cl does. */
        float r;
        r = b*(ldf(rxyr, indr)*cv-leta[l]*(dipjp*vxyyxr)); stf(rxy, indr, r); sumrxy += r;
        r = b*(ldf(ryzr, indr)*cv-leta[l]*(djpkp*vyzzyr)); stf(ryz, indr, r); sumryz += r;
        r = b*(ldf(rxzr, indr)*cv-leta[l]*(dipkp*vxzzxr)); stf(rxz, indr, r); sumrxz += r;
        r = b*(ldf(rxxr, indr)*cv-leta[l]*((e*vxxyyzzr)-(d*vyyzzr))); stf(rxx, indr, r); sumrxx += r;
        r = b*(ldf(ryyr, indr)*cv-leta[l]*((e*vxxyyzzr)-(d*vxxzzr))); stf(ryy, indr, r); sumryy += r;
        r = b*(ldf(rzzr, indr)*cv-leta[l]*((e*vxxyyzzr)-(d*vxxyyr))); stf(rzz, indr, r); sumrzz += r;
    }

    /* and now the components of the stress tensor are
     completely updated */
    stf(sxyr, indv, ldf(sxyr, indv) + (lsxy+(DT2*sumrxy)));
    stf(syzr, indv, ldf(syzr, indv) + (lsyz+(DT2*sumryz)));
    stf(sxzr, indv, ldf(sxzr, indv) + (lsxz+(DT2*sumrxz)));
    stf(sxxr, indv, ldf(sxxr, indv) + (lsxx+(DT2*sumrxx)));
    stf(syyr, indv, ldf(syyr, indv) + (lsyy+(DT2*sumryy)));
    stf(szzr, indv, ldf(szzr, indv) + (lszz+(DT2*sumrzz)));

#endif
}


// Absorbing boundary
#if ABS_TYPE==2
    {
        #if FREESURF==0
        if (gidz-FDOH<NAB){
            stf(sxyr, indv, ldf(sxyr, indv) * (taper[gidz-FDOH]));
            stf(syzr, indv, ldf(syzr, indv) * (taper[gidz-FDOH]));
            stf(sxzr, indv, ldf(sxzr, indv) * (taper[gidz-FDOH]));
            stf(sxxr, indv, ldf(sxxr, indv) * (taper[gidz-FDOH]));
            stf(syyr, indv, ldf(syyr, indv) * (taper[gidz-FDOH]));
            stf(szzr, indv, ldf(szzr, indv) * (taper[gidz-FDOH]));
        }
        #endif

        if (gidz>NZS-NAB-FDOH-1){
            stf(sxyr, indv, ldf(sxyr, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(syzr, indv, ldf(syzr, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(sxzr, indv, ldf(sxzr, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(sxxr, indv, ldf(sxxr, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(syyr, indv, ldf(syyr, indv) * (taper[NZS-FDOH-gidz-1]));
            stf(szzr, indv, ldf(szzr, indv) * (taper[NZS-FDOH-gidz-1]));

        }
        if (gidy-FDOH<NAB){
            stf(sxyr, indv, ldf(sxyr, indv) * (taper[gidy-FDOH]));
            stf(syzr, indv, ldf(syzr, indv) * (taper[gidy-FDOH]));
            stf(sxzr, indv, ldf(sxzr, indv) * (taper[gidy-FDOH]));
            stf(sxxr, indv, ldf(sxxr, indv) * (taper[gidy-FDOH]));
            stf(syyr, indv, ldf(syyr, indv) * (taper[gidy-FDOH]));
            stf(szzr, indv, ldf(szzr, indv) * (taper[gidy-FDOH]));
        }

        if (gidy>NYS-NAB-FDOH-1){
            stf(sxyr, indv, ldf(sxyr, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(syzr, indv, ldf(syzr, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(sxzr, indv, ldf(sxzr, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(sxxr, indv, ldf(sxxr, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(syyr, indv, ldf(syyr, indv) * (taper[NYS-FDOH-gidy-1]));
            stf(szzr, indv, ldf(szzr, indv) * (taper[NYS-FDOH-gidy-1]));
        }

#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            stf(sxyr, indv, ldf(sxyr, indv) * (taper[gidx-FDOH]));
            stf(syzr, indv, ldf(syzr, indv) * (taper[gidx-FDOH]));
            stf(sxzr, indv, ldf(sxzr, indv) * (taper[gidx-FDOH]));
            stf(sxxr, indv, ldf(sxxr, indv) * (taper[gidx-FDOH]));
            stf(syyr, indv, ldf(syyr, indv) * (taper[gidx-FDOH]));
            stf(szzr, indv, ldf(szzr, indv) * (taper[gidx-FDOH]));
        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            stf(sxyr, indv, ldf(sxyr, indv) * (taper[NX-FDOH-gidx-1]));
            stf(syzr, indv, ldf(syzr, indv) * (taper[NX-FDOH-gidx-1]));
            stf(sxzr, indv, ldf(sxzr, indv) * (taper[NX-FDOH-gidx-1]));
            stf(sxxr, indv, ldf(sxxr, indv) * (taper[NX-FDOH-gidx-1]));
            stf(syyr, indv, ldf(syyr, indv) * (taper[NX-FDOH-gidx-1]));
            stf(szzr, indv, ldf(szzr, indv) * (taper[NX-FDOH-gidx-1]));
        }
#endif
    }
#endif

//Shear wave modulus and P-wave modulus gradient calculation on the fly
#if BACK_PROP_TYPE==1
    #if RESTYPE==0
    float c1=1.0/(3.0*lM-4.0*lmu)/(3.0*lM-4.0*lmu);
    // Each shear term is evaluated at its own staggered mu position
    // (fipkp/fipjp/fjpkp), not the cell-centred lmu the diagonal (c5) term
    // uses -- matching grad_dft2D.cl's Gmu/Gmuipkp split, generalized to
    // the three 3D shear planes. Kept in their own gradmu{ipkp,ipjp,jpkp}
    // accumulators so average_grad_transpose() (calc_grad.c) can apply the
    // harmonic-mean averaging Jacobian to each separately.
    // Read the staggered mu values directly from the parameter buffers
    // rather than reusing fipkp/fipjp/fjpkp -- those locals are only
    // (re)assigned inside the forward-reconstruction branch above, so
    // reading the raw buffer here is the safer choice regardless. This did
    // NOT fix the NaN gradmuipkp/gradmuipjp/gradmujpkp observed via
    // instrumentation (average_grad_transpose() dump, host-side
    // ldf(muipkp, indp) is confirmed nonzero/sane at this same index) -- the
    // real cause is still open, see notes/todo.md item 1's 3D section.
    // Kept because it removes one real risk (stale locals) even though it
    // is not the fix; do not assume this is done.
    float gfipkp=ldf(muipkp, indp);
    float gfipjp=ldf(muipjp, indp);
    float gfjpkp=ldf(mujpkp, indp);
    // Fluid/padding cells (staggered mu==0, e.g. muipkp/muipjp/mujpkp can
    // legitimately be 0 at domain-edge ghost positions even where the
    // cell-centred lmu is not) must drop the shear coefficient entirely,
    // with a guard mu>0 (not mu>=1: with FP16>0 mu is stored scaled by
    // 2^PARSCALE and can be below 1, which dropped the shear terms) -- 1/0 gives Inf, and
    // Inf*0 (whenever the correlated stress happens to be exactly 0 at
    // that same cell) gives NaN, which silently poisons the whole
    // gradmuipkp/gradmuipjp/gradmujpkp buffer through the += accumulation.
    // Confirmed via instrumentation: ldf(muipkp, indp) is 0 at 4032/262144
    // cells for this test model.
    float c3xz = (gfipkp>0.0f) ? 1.0f/gfipkp/gfipkp : 0.0f;
    float c3xy = (gfipjp>0.0f) ? 1.0f/gfipjp/gfipjp : 0.0f;
    float c3yz = (gfjpkp>0.0f) ? 1.0f/gfjpkp/gfjpkp : 0.0f;
    float c5=1.0/6.0/lmu/lmu;

    /* Receiver cells: the residual term of the gradient is added by
       rec_grad.cl (see there). */

    float dM=c1*( ldf(sxx, indv)+ldf(syy, indv)+ldf(szz, indv) )*( lsxx+lsyy+lszz );

    int sumform = 1;
    #if FREESURF==1
    if (gidz-FDOH < SURFROWS) sumform = 0;
    #endif
    if (sumform){
    gradM[indp] += GSCALE(-dM);
    gradmuipkp[indp] += GSCALE(-c3xz*(ldf(sxz, indv)*lsxz));
    gradmuipjp[indp] += GSCALE(-c3xy*(ldf(sxy, indv)*lsxy));
    gradmujpkp[indp] += GSCALE(-c3yz*(ldf(syz, indv)*lsyz));
    gradmu[indp] += GSCALE(4.0/3*dM-c5*(lsxx*(2.0*ldf(sxx, indv)- ldf(syy, indv)-ldf(szz, indv) )
                      +lsyy*(2.0*ldf(syy, indv)- ldf(sxx, indv)-ldf(szz, indv) )
                      +lszz*(2.0*ldf(szz, indv)- ldf(sxx, indv)-ldf(syy, indv) )));
    }

    /* Source term of the misfit gradient -- GJI 2017 eq. (26a), thesis
     * eq. (3.51).  See update_adjs2D.cl for the full derivation; this is the
     * same term with N=3.  The accumulation above is the (A phi' + B phi)
     * part only, so the "- s" of the bracket survives as a purely local
     *     +c1M * <sigma~_kk , s_kk>
     * that is nonzero only in cells containing a source.
     *
     * Coefficients follow the same appendix rows the code already uses:
     * c1M = b1^2 = c1 (eq. A4a at L=0) for M, and c2mu = (N+1)/3 * b1^2 for
     * mu -- which is exactly the 4.0/3*dM factor above, so the mu side of
     * the correction carries the same 4/3.
     *
     * Only P1 is affected: an isotropic source injects amp/n2ave into each
     * of sxx,syy,szz, so the deviatoric combination
     * (N-1)s_ii - sum_{j!=i} s_jj vanishes and the c5 term above is
     * untouched, as are the shear planes and dJ/drho. */
    #if GRADOUT==1
    if (nsrc>0 && sumform){
        for (int srci=0; srci<nsrc; srci++){
            if ((int)src_pos[4+5*srci]==100){
                int si=(int)(src_pos[0+5*srci]/DH)+FDOH-OFFSET;   /* local x (domain decomposition) */
                int sj=(int)(src_pos[1+5*srci]/DH)+FDOH;
                int sk=(int)(src_pos[2+5*srci]/DH)+FDOH;
                if (si==gidx && sj==gidy && sk==gidz){
                    float samp = pow2s(DT*src[srci*NT+nt], src_scale);
                    float Csrc = c1*( ldf(sxxr, indv)+ldf(syyr, indv)+ldf(szzr, indv) )*samp;
                    gradM[indp] += GSCALE(Csrc);
                    gradmu[indp] += GSCALE(-4.0/3*Csrc);
                }
            }
        }
    }
    #endif
    #if HOUT==1
    float dMH=c1*(ldf(sxx, indv)+ldf(syy, indv)+ldf(szz, indv))*(ldf(sxx, indv)+ldf(syy, indv)+ldf(szz, indv));
    HM[indp] += GSCALE(dMH);
    Hmu[indp] += GSCALE(c3xz*ldf(sxz, indv)*ldf(sxz, indv)+c3xy*ldf(sxy, indv)*ldf(sxy, indv)+c3yz*ldf(syz, indv)*ldf(syz, indv)
                    - 4.0/3*dM
                    +c5*( (2.0*ldf(sxx, indv)- ldf(syy, indv)-ldf(szz, indv))*(2.0*ldf(sxx, indv)- ldf(syy, indv)-ldf(szz, indv)))
                         +(2.0*ldf(syy, indv)- ldf(sxx, indv)-ldf(szz, indv))*(2.0*ldf(syy, indv)- ldf(sxx, indv)-ldf(szz, indv))
                         +(2.0*ldf(szz, indv)- ldf(sxx, indv)-ldf(syy, indv))*(2.0*ldf(szz, indv)- ldf(sxx, indv)-ldf(syy, indv))));
    #endif
    #endif

    #if RESTYPE==1
    float dM=(ldf(sxx, indv)+ldf(syy, indv)+ldf(szz, indv) )*( lsxx+lsyy+lszz );

    gradM[indp] += GSCALE(-dM);

    #if HOUT==1
    float dMH= (ldf(sxx, indv)+ldf(syy, indv)+ldf(szz, indv))*(ldf(sxx, indv)+ldf(syy, indv)+ldf(szz, indv));
    HM[indp] += GSCALE(dMH);

    #endif
    #endif

#endif

#if CPMLINL
    {   /* the memory of the cell's own values: psi_new = b*psi_old + a*e */
        cpi_upd3(it0, gidx, NX, 1, PSI_HX, cpi_o, cpi_pbx, cpi_psx, vxr, psi_sxx_xr, 0, vyr, psi_sxy_xr, 1,
                 vzr, psi_sxz_xr, 1, a_x, b_x, a_x_half, b_x_half);
        cpi_upd3(it0, gidy, NYS, 1, PSI_HY, cpi_o, cpi_pby, cpi_psy, vxr, psi_sxy_yr, 1, vyr, psi_syy_yr, 0,
                 vzr, psi_syz_yr, 1, a_y, b_y, a_y_half, b_y_half);
        cpi_upd3(it0, zt, NZS, CPI_LOWZ, PSI_HZ, cpi_o, cpi_pbz, cpi_psz, vxr, psi_sxz_zr, 1, vyr, psi_syz_zr, 1,
                 vzr, psi_szz_zr, 0, a_z, b_z, a_z_half, b_z_half);
    }
#endif
#if GRADSRCOUT==1
    //TODO
//    float pressure;
//    if (nsrc>0){
//
//        for (int srci=0; srci<nsrc; srci++){
//
//            int SOURCE_TYPE= (int)srcpos_loc(4,srci);
//
//            if (SOURCE_TYPE==1){
//                int i=(int)(srcpos_loc(0,srci)-0.5)+FDOH;
//                int j=(int)(srcpos_loc(1,srci)-0.5)+FDOH;
//                int k=(int)(srcpos_loc(2,srci)-0.5)+FDOH;
//
//
//                if (i==gidx && j==gidy && k==gidz){
//
//                    pressure=(ldf(sxxr, indv)+ldf(syyr, indv)+ldf(szzr, indv) )/(2.0*DH*DH*DH);
//                    if ( (nt>0) && (nt< NT ) ){
//                        gradsrc(srci,nt+1)+=pressure;
//                        gradsrc(srci,nt-1)-=pressure;
//                    }
//                    else if (nt==0)
//                        gradsrc(srci,nt+1)+=pressure;
//                    else if (nt==NT)
//                        gradsrc(srci,nt-1)-=pressure;
//
//                }
//            }
//        }
//    }

#endif



    }
}
