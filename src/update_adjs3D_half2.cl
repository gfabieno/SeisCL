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

FUNDEF void update_adjs(int offcomm,
                        GLOBARG __pprec *muipjp, GLOBARG __pprec *mujpkp,
                        GLOBARG __pprec *muipkp, GLOBARG __pprec *M,
                        GLOBARG __pprec *mu,
                        GLOBARG __prec2 *sxx, GLOBARG __prec2 *sxy,
                        GLOBARG __prec2 *sxz, GLOBARG __prec2 *syy,
                        GLOBARG __prec2 *syz, GLOBARG __prec2 *szz,
                        GLOBARG __prec2 *vx,  GLOBARG __prec2 *vy,
                        GLOBARG __prec2 *vz,
                        GLOBARG __prec2 *sxxbnd, GLOBARG __prec2 *sxybnd,
                        GLOBARG __prec2 *sxzbnd, GLOBARG __prec2 *syybnd,
                        GLOBARG __prec2 *syzbnd, GLOBARG __prec2 *szzbnd,
                        GLOBARG __prec2 *sxxr, GLOBARG __prec2 *sxyr,
                        GLOBARG __prec2 *sxzr, GLOBARG __prec2 *syyr,
                        GLOBARG __prec2 *syzr, GLOBARG __prec2 *szzr,
                        GLOBARG __prec2 *vxr,  GLOBARG __prec2 *vyr,
                        GLOBARG __prec2 *vzr,
                        GLOBARG float *taper,
                        GLOBARG float *K_x,     GLOBARG float *a_x,
                        GLOBARG float *b_x,     GLOBARG float *K_x_half,
                        GLOBARG float *a_x_half,GLOBARG float *b_x_half,
                        GLOBARG float *K_y,     GLOBARG float *a_y,
                        GLOBARG float *b_y,     GLOBARG float *K_y_half,
                        GLOBARG float *a_y_half,GLOBARG float *b_y_half,
                        GLOBARG float *K_z,     GLOBARG float *a_z,
                        GLOBARG float *b_z,     GLOBARG float *K_z_half,
                        GLOBARG float *a_z_half,GLOBARG float *b_z_half,
                        GLOBARG __prec2 *psi_vx_x, GLOBARG __prec2 *psi_vx_y,
                        GLOBARG __prec2 *psi_vx_z, GLOBARG __prec2 *psi_vy_x,
                        GLOBARG __prec2 *psi_vy_y, GLOBARG __prec2 *psi_vy_z,
                        GLOBARG __prec2 *psi_vz_x, GLOBARG __prec2 *psi_vz_y,
                        GLOBARG __prec2 *psi_vz_z,
                        GLOBARG __gprec *gradM,  GLOBARG __gprec *gradmu,
                        GLOBARG __gprec *HM,     GLOBARG __gprec *Hmu,
                        int res_scale, int src_scale, int par_scale,
                        LOCARG2)
{

    //Local memory
    #ifdef __OPENCL_VERSION__
    __local __prec * lvar=lvar2;
    #else
    extern __shared__ __prec2 lvar2[];
    __prec * lvar=(__prec *)lvar2;
    #endif
    
    //Grid position
    #if LOCAL_OFF==0

        #ifdef __OPENCL_VERSION__
            int lsizez = get_local_size(0)+2*FDOH/DIV;
            int lsizey = get_local_size(1)+2*FDOH;
            int lsizex = get_local_size(2)+2*FDOH;
            int lidz = get_local_id(0)+FDOH/DIV;
            int lidy = get_local_id(1)+FDOH;
            int lidx = get_local_id(2)+FDOH;
            int gidz = get_global_id(0)+FDOH/DIV;
            int gidy = get_global_id(1)+FDOH;
            int gidx = get_global_id(2)+FDOH+offcomm;
        #else
            int lsizez = blockDim.x+2*FDOH/DIV;
            int lsizey = blockDim.y+2*FDOH;
            int lsizex = blockDim.z+2*FDOH;
            int lidz = threadIdx.x+FDOH/DIV;
            int lidy = threadIdx.y+FDOH;
            int lidx = threadIdx.z+FDOH;
            int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/DIV;
            int gidy = blockIdx.y*blockDim.y+threadIdx.y+FDOH;
            int gidx = blockIdx.z*blockDim.z+threadIdx.z+FDOH+offcomm;
        #endif
    
    // If local memory is turned off
    #elif LOCAL_OFF==1
    
        #ifdef __OPENCL_VERSION__
            int gid = get_global_id(0);
            int glsizez = (NZ-2*FDOH/DIV);
            int glsizey = (NY-2*FDOH);
            int gidz = gid%glsizez+FDOH/DIV;
            int gidy = (gid/glsizez)%glsizey+FDOH;
            int gidx = gid/(glsizez*glsizey)+FDOH+offcomm;
        #else
            int lsizez = blockDim.x+2*FDOH/DIV;
            int lsizey = blockDim.y+2*FDOH;
            int lsizex = blockDim.z+2*FDOH;
            int lidz = threadIdx.x+FDOH/DIV;
            int lidy = threadIdx.y+FDOH;
            int lidx = threadIdx.z+FDOH;
            int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/DIV;
            int gidy = blockIdx.y*blockDim.y+threadIdx.y+FDOH;
            int gidx = blockIdx.z*blockDim.z+threadIdx.z+FDOH+offcomm;
        #endif
    
    #endif
    
    int indp = ((gidx)-FDOH)*(NZ-2*FDOH/DIV)*(NY-2*FDOH)+((gidy)-FDOH)*(NZ-2*FDOH/DIV)+((gidz)-FDOH/DIV);
    int indv = (gidx)*NZ*NY+(gidy)*NZ+(gidz);

    //Define private derivatives
    __cprec vx_x2;
    __cprec vx_y1;
    __cprec vx_z1;
    __cprec vy_x1;
    __cprec vy_y2;
    __cprec vy_z1;
    __cprec vz_x1;
    __cprec vz_y1;
    __cprec vz_z2;

    __cprec vxr_x2;
    __cprec vxr_y1;
    __cprec vxr_z1;
    __cprec vyr_x1;
    __cprec vyr_y2;
    __cprec vyr_z1;
    __cprec vzr_x1;
    __cprec vzr_y1;
    __cprec vzr_z2;

    //Local memory definitions if local is used
    #if LOCAL_OFF==0
        #define lvz lvar
        #define lvy lvar
        #define lvx lvar
        #define lvz2 lvar2
        #define lvy2 lvar2
        #define lvx2 lvar2

        #define lvzr lvar
        #define lvyr lvar
        #define lvxr lvar
        #define lvzr2 lvar2
        #define lvyr2 lvar2
        #define lvxr2 lvar2

        //Local memory definitions if local is not used
    #elif LOCAL_OFF==1
        #define lvz vz
        #define lvy vy
        #define lvx vx
        #define lidz gidz
        #define lidy gidy
        #define lidx gidx

    #endif

    #if BACK_PROP_TYPE==1
    //Calculation of the spatial derivatives
    {
        #if LOCAL_OFF==0
            load_local_in(vz);
            load_local_haloz(vz);
            load_local_haloy(vz);
            load_local_halox(vz);
            BARRIER
        #endif
        vz_x1 = Dxp(lvz2);
        vz_y1 = Dyp(lvz2);
        vz_z2 = Dzm(lvz);
        
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(vy);
            load_local_haloz(vy);
            load_local_haloy(vy);
            load_local_halox(vy);
            BARRIER
        #endif
        vy_x1 = Dxp(lvy2);
        vy_y2 = Dym(lvy2);
        vy_z1 = Dzp(lvy);
        
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(vx);
            load_local_haloz(vx);
            load_local_haloy(vx);
            load_local_halox(vx);
            BARRIER
        #endif
        vx_x2 = Dxm(lvx2);
        vx_y1 = Dyp(lvx2);
        vx_z1 = Dzp(lvx);

        BARRIER
            }
    #endif

    //Calculation of the spatial derivatives
    {
    #if LOCAL_OFF==0
        load_local_in(vzr);
        load_local_haloz(vzr);
        load_local_haloy(vzr);
        load_local_halox(vzr);
        BARRIER
    #endif
    vzr_x1 = Dxp(lvzr2);
    vzr_y1 = Dyp(lvzr2);
    vzr_z2 = Dzm(lvzr);
        
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(vyr);
        load_local_haloz(vyr);
        load_local_haloy(vyr);
        load_local_halox(vyr);
        BARRIER
    #endif
    vyr_x1 = Dxp(lvyr2);
    vyr_y2 = Dym(lvyr2);
    vyr_z1 = Dzp(lvyr);
        
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(vxr);
        load_local_haloz(vxr);
        load_local_haloy(vxr);
        load_local_halox(vxr);
        BARRIER
    #endif
    vxr_x2 = Dxm(lvxr2);
    vxr_y1 = Dyp(lvxr2);
    vxr_z1 = Dzp(lvxr);
        }
    // To stop updating if we are outside the model (global id must be amultiple
    // of local id in OpenCL, hence we stop if we have a global idoutside the grid)
    #if  LOCAL_OFF==0
    #if COMM12==0
    if ( gidz>(NZ-FDOH/DIV-1) ||  gidy>(NY-FDOH-1) ||  (gidx-offcomm)>(NX-FDOH-1-LCOMM) )
        return;
    #else
    if ( gidz>(NZ-FDOH/DIV-1) ||  gidy>(NY-FDOH-1)  )
        return;
    #endif
    #endif
    
    //Define and load private parameters and variables
    __cprec lsxxr = __h22f2(sxxr[indv]);
    __cprec lsxyr = __h22f2(sxyr[indv]);
    __cprec lsxzr = __h22f2(sxzr[indv]);
    __cprec lsyyr = __h22f2(syyr[indv]);
    __cprec lsyzr = __h22f2(syzr[indv]);
    __cprec lszzr = __h22f2(szzr[indv]);
    __cprec lM = __pconv(M[indp]);
    __cprec lmu = __pconv(mu[indp]);
    __cprec lmuipjp = __pconv(muipjp[indp]);
    __cprec lmuipkp = __pconv(muipkp[indp]);
    __cprec lmujpkp = __pconv(mujpkp[indp]);
    
    // Backpropagate the forward stresses
    #if BACK_PROP_TYPE==1
    __cprec lsxx = __h22f2(sxx[indv]);
    __cprec lsxy = __h22f2(sxy[indv]);
    __cprec lsxz = __h22f2(sxz[indv]);
    __cprec lsyy = __h22f2(syy[indv]);
    __cprec lsyz = __h22f2(syz[indv]);
    __cprec lszz = __h22f2(szz[indv]);
    {
        
        lsxy=lsxy - lmuipjp*(vx_y1+vy_x1);
        lsyz=lsyz - lmujpkp*(vy_z1+vz_y1);
        lsxz=lsxz - lmuipkp*(vx_z1+vz_x1);
        lsxx=lsxx - lM*(vx_x2+vy_y2+vz_z2) + 2.0f * lmu*(vy_y2+vz_z2);
        lsyy=lsyy - lM*(vx_x2+vy_y2+vz_z2) + 2.0f * lmu*(vx_x2+vz_z2);
        lszz=lszz - lM*(vx_x2+vy_y2+vz_z2) + 2.0f * lmu*(vx_x2+vy_y2);
        
        int m=inject_ind(gidz, gidy, gidx);
        if (m!=-1){
            lsxx= __h22f2(sxxbnd[m]);
            lsyy= __h22f2(syybnd[m]);
            lszz= __h22f2(szzbnd[m]);
            lsxy= __h22f2(sxybnd[m]);
            lsxz= __h22f2(sxzbnd[m]);
            lsyz= __h22f2(syzbnd[m]);
        }

        //Write updated values to global memory
        sxx[indv] = __f22h2(lsxx);
        sxy[indv] = __f22h2(lsxy);
        sxz[indv] = __f22h2(lsxz);
        syy[indv] = __f22h2(lsyy);
        syz[indv] = __f22h2(lsyz);
        szz[indv] = __f22h2(lszz);
    }
    #endif
    
    // Correct spatial derivatives to implement CPML for adjoint variables
    #if ABS_TYPE==1
    {
        int i,j,k,indm, indn;
        if (DIV*gidz>DIV*NZ-NAB-FDOH-1){
            
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz - NZ + 2*NAB/DIV + FDOH/DIV;
            indm=2*NAB - 1 - k*DIV;
            indn = (i)*(NY-2*FDOH)*(2*NAB/DIV)+(j)*(2*NAB/DIV)+(k);
            
            psi_vx_z[indn] = __f22h2(__hpgi(&b_z_half[indm]) * psi_vx_z[indn]
                                    + __hpgi(&a_z_half[indm]) * vxr_z1);
            vxr_z1 = vxr_z1 / __hpgi(&K_z_half[indm]) + psi_vx_z[indn];
            psi_vy_z[indn] = __f22h2(__hpgi(&b_z_half[indm]) * psi_vy_z[indn]
                                    + __hpgi(&a_z_half[indm]) * vyr_z1);
            vyr_z1 = vyr_z1 / __hpgi(&K_z_half[indm]) + psi_vy_z[indn];
            psi_vz_z[indn] = __f22h2(__hpgi(&b_z[indm+1]) * psi_vz_z[indn]
                                    + __hpgi(&a_z[indm+1]) * vzr_z2);
            vzr_z2 = vzr_z2 / __hpgi(&K_z[indm+1]) + psi_vz_z[indn];
        }
        
        #if FREESURF==0
        else if (DIV*gidz-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =DIV*gidz-FDOH;
            indn = (i)*(NY-2*FDOH)*(2*NAB/DIV)+(j)*(2*NAB/DIV)+(k/DIV);

            psi_vx_z[indn] = __f22h2(__hpg(&b_z_half[k]) * psi_vx_z[indn]
                                    + __hpg(&a_z_half[k]) * vxr_z1);
            vxr_z1 = vxr_z1 / __hpg(&K_z_half[k]) + psi_vx_z[indn];
            psi_vy_z[indn] = __f22h2(__hpg(&b_z_half[k]) * psi_vy_z[indn]
                                    + __hpg(&a_z_half[k]) * vyr_z1);
            vyr_z1 = vyr_z1 / __hpg(&K_z_half[k]) + psi_vy_z[indn];
            psi_vz_z[indn] = __f22h2(__hpg(&b_z[k]) * psi_vz_z[indn]
                                    + __hpg(&a_z[k]) * vzr_z2);
            vzr_z2 = vzr_z2 / __hpg(&K_z[k]) + psi_vz_z[indn];
        }
        #endif

        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH/DIV;
            indn = (i)*(2*NAB)*(NZ-2*FDOH/DIV)+(j)*(NZ-2*FDOH/DIV)+(k);

            psi_vx_y[indn] = __f22h2(b_y_half[j] * psi_vx_y[indn]
                                    + a_y_half[j] * vxr_y1);
            vxr_y1 = vxr_y1 / K_y_half[j] + psi_vx_y[indn];
            psi_vy_y[indn] = __f22h2(b_y[j] * psi_vy_y[indn]
                                    + a_y[j] * vyr_y2);
            vyr_y2 = vyr_y2 / K_y[j] + psi_vy_y[indn];
            psi_vz_y[indn] = __f22h2(b_y_half[j] * psi_vz_y[indn]
                                    + a_y_half[j] * vzr_y1);
            vzr_y1 = vzr_y1 / K_y_half[j] + psi_vz_y[indn];
        }

        else if (gidy>NY-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy - NY+NAB+FDOH+NAB;
            k =gidz-FDOH/DIV;
            indm=2*NAB-1-j;
            indn = (i)*(2*NAB)*(NZ-2*FDOH/DIV)+(j)*(NZ-2*FDOH/DIV)+(k);

            psi_vx_y[indn] = __f22h2(b_y_half[indm] * psi_vx_y[indn]
                                    + a_y_half[indm] * vxr_y1);
            vxr_y1 = vxr_y1 / K_y_half[indm] + psi_vx_y[indn];
            psi_vy_y[indn] = __f22h2(b_y[indm+1] * psi_vy_y[indn]
                                    + a_y[indm+1] * vyr_y2);
            vyr_y2 = vyr_y2 / K_y[indm+1] + psi_vy_y[indn];
            psi_vz_y[indn] = __f22h2(b_y_half[indm] * psi_vz_y[indn]
                                    + a_y_half[indm] * vzr_y1);
            vzr_y1 = vzr_y1 / K_y_half[indm] + psi_vz_y[indn];
        }
        #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH/DIV;
            indn = (i)*(NY-2*FDOH)*(NZ-2*FDOH/DIV)+(j)*(NZ-2*FDOH/DIV)+(k);

            psi_vx_x[indn] = __f22h2(b_x[i] * psi_vx_x[indn]
                                    + a_x[i] * vxr_x2);
            vxr_x2 = vxr_x2 / K_x[i] + psi_vx_x[indn];
            psi_vy_x[indn] = __f22h2(b_x_half[i] * psi_vy_x[indn]
                                    + a_x_half[i] * vyr_x1);
            vyr_x1 = vyr_x1 / K_x_half[i] + psi_vy_x[indn];
            psi_vz_x[indn] = __f22h2(b_x_half[i] * psi_vz_x[indn]
                                    + a_x_half[i] * vzr_x1);
            vzr_x1 = vzr_x1 / K_x_half[i] + psi_vz_x[indn];
        }
        #endif

        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){

            i =gidx - NX+NAB+FDOH+NAB;
            j =gidy-FDOH;
            k =gidz-FDOH/DIV;
            indm=2*NAB-1-i;
            indn = (i)*(NY-2*FDOH)*(NZ-2*FDOH/DIV)+(j)*(NZ-2*FDOH/DIV)+(k);

            psi_vx_x[indn] = __f22h2(b_x[indm+1] * psi_vx_x[indn]
                                    + a_x[indm+1] * vxr_x2);
            vxr_x2 = vxr_x2 /K_x[indm+1] + psi_vx_x[indn];
            psi_vy_x[indn] = __f22h2(b_x_half[indm] * psi_vy_x[indn]
                                    + a_x_half[indm] * vyr_x1);
            vyr_x1 = vyr_x1  /K_x_half[indm] + psi_vy_x[indn];
            psi_vz_x[indn] = __f22h2(b_x_half[indm] * psi_vz_x[indn]
                                    + a_x_half[indm] * vzr_x1);
            vzr_x1 = vzr_x1 / K_x_half[indm]  +psi_vz_x[indn];
        }
        #endif
    }
    #endif
    
    // Update adjoint stresses
    {
        // Update the variables
        lsxyr=lsxyr + lmuipjp*(vxr_y1+vyr_x1);
        lsyzr=lsyzr + lmujpkp*(vyr_z1+vzr_y1);
        lsxzr=lsxzr + lmuipkp*(vxr_z1+vzr_x1);
        lsxxr=lsxxr + lM*(vxr_x2+vyr_y2+vzr_z2) - 2.0f * lmu*(vyr_y2+vzr_z2);
        lsyyr=lsyyr + lM*(vxr_x2+vyr_y2+vzr_z2) - 2.0f * lmu*(vxr_x2+vzr_z2);
        lszzr=lszzr + lM*(vxr_x2+vyr_y2+vzr_z2) - 2.0f * lmu*(vxr_x2+vyr_y2);
    
     // Absorbing boundary
    #if ABS_TYPE==2
        {
        #if FREESURF==0
        if (DIV*gidz-FDOH<NAB){
            lsxyr = lsxyr * __hpg(&taper[DIV*gidz-FDOH]);
            lsyzr = lsyzr * __hpg(&taper[DIV*gidz-FDOH]);
            lsxzr = lsxzr * __hpg(&taper[DIV*gidz-FDOH]);
            lsxxr = lsxxr * __hpg(&taper[DIV*gidz-FDOH]);
            lsyyr = lsyyr * __hpg(&taper[DIV*gidz-FDOH]);
            lszzr = lszzr * __hpg(&taper[DIV*gidz-FDOH]);
        }
        #endif

        if (DIV*gidz>DIV*NZ-NAB-FDOH-1){
            lsxyr = lsxyr * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lsyzr = lsyzr * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lsxzr = lsxzr * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lsxxr = lsxxr * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lsyyr = lsyyr * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lszzr = lszzr * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
        }
        if (gidy-FDOH<NAB){
            lsxyr = lsxyr * taper[gidy-FDOH];
            lsyzr = lsyzr * taper[gidy-FDOH];
            lsxzr = lsxzr * taper[gidy-FDOH];
            lsxxr = lsxxr * taper[gidy-FDOH];
            lsyyr = lsyyr * taper[gidy-FDOH];
            lszzr = lszzr * taper[gidy-FDOH];
        }

        if (gidy>NY-NAB-FDOH-1){
            lsxyr = lsxyr * taper[NY-FDOH-gidy-1];
            lsyzr = lsyzr * taper[NY-FDOH-gidy-1];
            lsxzr = lsxzr * taper[NY-FDOH-gidy-1];
            lsxxr = lsxxr * taper[NY-FDOH-gidy-1];
            lsyyr = lsyyr * taper[NY-FDOH-gidy-1];
            lszzr = lszzr * taper[NY-FDOH-gidy-1];
        }
        #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            lsxyr = lsxyr * taper[gidx-FDOH];
            lsyzr = lsyzr * taper[gidx-FDOH];
            lsxzr = lsxzr * taper[gidx-FDOH];
            lsxxr = lsxxr * taper[gidx-FDOH];
            lsyyr = lsyyr * taper[gidx-FDOH];
            lszzr = lszzr * taper[gidx-FDOH];
        }
        #endif

        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            lsxyr = lsxyr * taper[NX-FDOH-gidx-1];
            lsyzr = lsyzr * taper[NX-FDOH-gidx-1];
            lsxzr = lsxzr * taper[NX-FDOH-gidx-1];
            lsxxr = lsxxr * taper[NX-FDOH-gidx-1];
            lsyyr = lsyyr * taper[NX-FDOH-gidx-1];
            lszzr = lszzr * taper[NX-FDOH-gidx-1];

        }
        #endif
        }
    #endif
    
    //Write updated values to global memory
    sxxr[indv] = __f22h2(lsxxr);
    sxyr[indv] = __f22h2(lsxyr);
    sxzr[indv] = __f22h2(lsxzr);
    syyr[indv] = __f22h2(lsyyr);
    syzr[indv] = __f22h2(lsyzr);
    szzr[indv] = __f22h2(lszzr);
        
    }
    
    //Shear wave modulus and P-wave modulus gradient calculation on the fly
    #if BACK_PROP_TYPE==1
    lsxyr=lmuipjp*(vxr_y1+vyr_x1);
    lsyzr=lmujpkp*(vyr_z1+vzr_y1);
    lsxzr=lmuipkp*(vxr_z1+vzr_x1);
    lsxxr=lM*(vxr_x2+vyr_y2+vzr_z2) - 2.0f * lmu*(vyr_y2+vzr_z2);
    lsyyr=lM*(vxr_x2+vyr_y2+vzr_z2) - 2.0f * lmu*(vxr_x2+vzr_z2);
    lszzr=lM*(vxr_x2+vyr_y2+vzr_z2) - 2.0f * lmu*(vxr_x2+vyr_y2);

    #if RESTYPE==0
    __gprec c1=__h22f2c(1.0f/(3.0f*lM-4.0f*lmu)/(3.0f*lM-4.0f*lmu));
    __gprec c3=__h22f2c(1.0f/lmu/lmu);
    __gprec c5=1.0f/6.0f*c3;
    
    __gprec dM=c1*( lsxx+lsyy+lszz )*( lsxxr+lsyyr+lszzr );
    gradM[indp] = gradM[indp] - scalefun(dM, 2*par_scale-src_scale - res_scale);
    
    gradmu[indp]=gradmu[indp] \
                 + scalefun(-c3*(lsxz*lsxzr +lsxy*lsxyr +lsyz*lsyzr)
                            + 4.0f/3.0f*dM
                            -c5*(lsxxr*(2.0f*lsxx-lsyy-lszz)
                                +lsyyr*(2.0f*lsyy-lsxx-lszz)
                                +lszzr*(2.0f*lszz-lsxx-lsyy)),
                            2*par_scale-src_scale - res_scale);

    #if HOUT==1
    dM=c1*( lsxx+lsyy+lszz )*( lsxx+lsyy+lszz );
    HM[indp] = HM[indp] + scalefun(dM, 2*par_scale-src_scale - res_scale);

    Hmu[indp]=Hmu[indp] + scalefun(-c3*(lsxz*lsxz +lsxy*lsxy +lsyz*lsyz)
                                   + 4.0f/3.0f*dM
                                   -c5*(lsxx*(2.0f*lsxx-lsyy-lszz)
                                        +lsyy*(2.0f*lsyy-lsxx-lszz)
                                        +lszz*(2.0f*lszz-lsxx-lsyy)),
                                   2*par_scale-src_scale - res_scale);
    #endif
    #endif

    #if RESTYPE==1
    __gprec dM=_( lsxx+lsyy+lszz )*( lsxxr+lsyyr+lszzr );

    gradM[indp] = gradM[indp] - scalefun(dM, 2*par_scale-src_scale - res_scale);

    #if HOUT==1
    dM= ( lsxx+lsyy+lszz )*( lsxx+lsyy+lszz );
    HM[indp] = HM[indp] - scalefun(dM, 2*par_scale-src_scale - res_scale);
    #endif
    #endif

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
    //                    pressure=(lsxxr+lsyyr+lszzr )/(2.0*DH*DH*DH);
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

