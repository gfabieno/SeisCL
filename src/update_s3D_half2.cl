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


FUNDEF void update_s(int offcomm,
                     GLOBARG __pprec *muipjp, GLOBARG __pprec *mujpkp,
                     GLOBARG __pprec *muipkp, GLOBARG __pprec *M,
                     GLOBARG __pprec *mu, GLOBARG __pprec *taus,
                     GLOBARG __pprec *tausipjp, GLOBARG __pprec *tausjpkp,
                     GLOBARG __pprec *tausipkp, GLOBARG __pprec *taup,
                     GLOBARG float *eta,
                     GLOBARG __prec2 *sxx, GLOBARG __prec2 *sxy,
                     GLOBARG __prec2 *sxz, GLOBARG __prec2 *syy,
                     GLOBARG __prec2 *syz, GLOBARG __prec2 *szz,
                     GLOBARG __prec2 *rxx, GLOBARG __prec2 *ryy,
                     GLOBARG __prec2 *rzz, GLOBARG __prec2 *rxy,
                     GLOBARG __prec2 *ryz, GLOBARG __prec2 *rxz,
                     GLOBARG __prec2 *vx,  GLOBARG __prec2 *vy,
                     GLOBARG __prec2 *vz,
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
                     GLOBARG float *taper,
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

    //Local memory definitions if local is used
    #if LOCAL_OFF==0
        #define lvz lvar
        #define lvy lvar
        #define lvx lvar
        #define lvz2 lvar2
        #define lvy2 lvar2
        #define lvx2 lvar2

    //Local memory definitions if local is not used
    #elif LOCAL_OFF==1

        #define lvz vz
        #define lvy vy
        #define lvx vx
        #define lidz gidz
        #define lidy gidy
        #define lidx gidx

    #endif

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

    // Correct spatial derivatives to implement CPML
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
                                    + __hpgi(&a_z_half[indm]) * vx_z1);
            vx_z1 = vx_z1 / __hpgi(&K_z_half[indm]) + psi_vx_z[indn];
            psi_vy_z[indn] = __f22h2(__hpgi(&b_z_half[indm]) * psi_vy_z[indn]
                                    + __hpgi(&a_z_half[indm]) * vy_z1);
            vy_z1 = vy_z1 / __hpgi(&K_z_half[indm]) + psi_vy_z[indn];
            psi_vz_z[indn] = __f22h2(__hpgi(&b_z[indm+1]) * psi_vz_z[indn]
                                    + __hpgi(&a_z[indm+1]) * vz_z2);
            vz_z2 = vz_z2 / __hpgi(&K_z[indm+1]) + psi_vz_z[indn];
        }
        
        #if FREESURF==0
        else if (DIV*gidz-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =DIV*gidz-FDOH;
            indn = (i)*(NY-2*FDOH)*(2*NAB/DIV)+(j)*(2*NAB/DIV)+(k/DIV);

            psi_vx_z[indn] = __f22h2(__hpg(&b_z_half[k]) * psi_vx_z[indn]
                                    + __hpg(&a_z_half[k]) * vx_z1);
            vx_z1 = vx_z1 / __hpg(&K_z_half[k]) + psi_vx_z[indn];
            psi_vy_z[indn] = __f22h2(__hpg(&b_z_half[k]) * psi_vy_z[indn]
                                    + __hpg(&a_z_half[k]) * vy_z1);
            vy_z1 = vy_z1 / __hpg(&K_z_half[k]) + psi_vy_z[indn];
            psi_vz_z[indn] = __f22h2(__hpg(&b_z[k]) * psi_vz_z[indn]
                                    + __hpg(&a_z[k]) * vz_z2);
            vz_z2 = vz_z2 / __hpg(&K_z[k]) + psi_vz_z[indn];
        }
        #endif

        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH/DIV;
            indn = (i)*(2*NAB)*(NZ-2*FDOH/DIV)+(j)*(NZ-2*FDOH/DIV)+(k);

            psi_vx_y[indn] = __f22h2(b_y_half[j] * psi_vx_y[indn]
                                    + a_y_half[j] * vx_y1);
            vx_y1 = vx_y1 / K_y_half[j] + psi_vx_y[indn];
            psi_vy_y[indn] = __f22h2(b_y[j] * psi_vy_y[indn]
                                    + a_y[j] * vy_y2);
            vy_y2 = vy_y2 / K_y[j] + psi_vy_y[indn];
            psi_vz_y[indn] = __f22h2(b_y_half[j] * psi_vz_y[indn]
                                    + a_y_half[j] * vz_y1);
            vz_y1 = vz_y1 / K_y_half[j] + psi_vz_y[indn];
        }

        else if (gidy>NY-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy - NY+NAB+FDOH+NAB;
            k =gidz-FDOH/DIV;
            indm=2*NAB-1-j;
            indn = (i)*(2*NAB)*(NZ-2*FDOH/DIV)+(j)*(NZ-2*FDOH/DIV)+(k);

            psi_vx_y[indn] = __f22h2(b_y_half[indm] * psi_vx_y[indn]
                                    + a_y_half[indm] * vx_y1);
            vx_y1 = vx_y1 / K_y_half[indm] + psi_vx_y[indn];
            psi_vy_y[indn] = __f22h2(b_y[indm+1] * psi_vy_y[indn]
                                    + a_y[indm+1] * vy_y2);
            vy_y2 = vy_y2 / K_y[indm+1] + psi_vy_y[indn];
            psi_vz_y[indn] = __f22h2(b_y_half[indm] * psi_vz_y[indn]
                                    + a_y_half[indm] * vz_y1);
            vz_y1 = vz_y1 / K_y_half[indm] + psi_vz_y[indn];
        }
        #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH/DIV;
            indn = (i)*(NY-2*FDOH)*(NZ-2*FDOH/DIV)+(j)*(NZ-2*FDOH/DIV)+(k);

            psi_vx_x[indn] = __f22h2(b_x[i] * psi_vx_x[indn]
                                    + a_x[i] * vx_x2);
            vx_x2 = vx_x2 / K_x[i] + psi_vx_x[indn];
            psi_vy_x[indn] = __f22h2(b_x_half[i] * psi_vy_x[indn]
                                    + a_x_half[i] * vy_x1);
            vy_x1 = vy_x1 / K_x_half[i] + psi_vy_x[indn];
            psi_vz_x[indn] = __f22h2(b_x_half[i] * psi_vz_x[indn]
                                    + a_x_half[i] * vz_x1);
            vz_x1 = vz_x1 / K_x_half[i] + psi_vz_x[indn];
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
                                    + a_x[indm+1] * vx_x2);
            vx_x2 = vx_x2 /K_x[indm+1] + psi_vx_x[indn];
            psi_vy_x[indn] = __f22h2(b_x_half[indm] * psi_vy_x[indn]
                                    + a_x_half[indm] * vy_x1);
            vy_x1 = vy_x1  /K_x_half[indm] + psi_vy_x[indn];
            psi_vz_x[indn] = __f22h2(b_x_half[indm] * psi_vz_x[indn]
                                    + a_x_half[indm] * vz_x1);
            vz_x1 = vz_x1 / K_x_half[indm]  +psi_vz_x[indn];
        }
        #endif
    }
    #endif
    //Define and load private parameters and variables
    __cprec lsxx = __h22f2(sxx[indv]);
    __cprec lsxy = __h22f2(sxy[indv]);
    __cprec lsxz = __h22f2(sxz[indv]);
    __cprec lsyy = __h22f2(syy[indv]);
    __cprec lsyz = __h22f2(syz[indv]);
    __cprec lszz = __h22f2(szz[indv]);


    #if LVE==0
    {
        __cprec lM  = __pconv(M[indp]);
        __cprec lmu = __pconv(mu[indp]);

        // Update the variables
        lsxy=lsxy + __pconv(muipjp[indp])*(vx_y1+vy_x1);
        lsyz=lsyz + __pconv(mujpkp[indp])*(vy_z1+vz_y1);
        lsxz=lsxz + __pconv(muipkp[indp])*(vx_z1+vz_x1);
        lsxx=lsxx + lM*(vx_x2+vy_y2+vz_z2) - 2.0f * lmu*(vy_y2+vz_z2);
        lsyy=lsyy + lM*(vx_x2+vy_y2+vz_z2) - 2.0f * lmu*(vx_x2+vz_z2);
        lszz=lszz + lM*(vx_x2+vy_y2+vz_z2) - 2.0f * lmu*(vx_x2+vy_y2);
    }
    #else
    {
        int indr, l;
        __cprec lM=__pconv(M[indp]);
        __cprec lmu=__pconv(mu[indp]);
        __cprec lmuipkp=__pconv(muipkp[indp]);
        __cprec lmuipjp=__pconv(muipjp[indp]);
        __cprec lmujpkp=__pconv(mujpkp[indp]);
        __cprec ltaup=__pconv(taup[indp]);
        __cprec ltaus=__pconv(taus[indp]);
        __cprec ltausipkp=__pconv(tausipkp[indp]);
        __cprec ltausipjp=__pconv(tausipjp[indp]);
        __cprec ltausjpkp=__pconv(tausjpkp[indp]);

        __cprec fipjp=lmuipjp*(1.0f+ (float)LVE*ltausipjp);
        __cprec fjpkp=lmujpkp*(1.0f+ (float)LVE*ltausjpkp);
        __cprec fipkp=lmuipkp*(1.0f+ (float)LVE*ltausipkp);
        __cprec g=lM*(1.0f+(float)LVE*ltaup);
        __cprec f=2.0f*lmu*(1.0f+(float)LVE*ltaus);
        __cprec dipjp=lmuipjp*ltausipjp/DT;
        __cprec djpkp=lmujpkp*ltausjpkp/DT;
        __cprec dipkp=lmuipkp*ltausipkp/DT;
        __cprec d=2.0f*lmu*ltaus/DT;
        __cprec e=lM*ltaup/DT;

        float leta[LVE];
        for (l=0;l<LVE;l++){
            leta[l]=eta[l];
        }

        /* computing sums of the old memory variables */
        __cprec sumrxy;
        __cprec sumryz;
        __cprec sumrxz;
        __cprec sumrxx;
        __cprec sumryy;
        __cprec sumrzz;
        initc0(sumrxy);initc0(sumryz);initc0(sumrxz);
        initc0(sumrxx);initc0(sumryy);initc0(sumrzz);
        for (l=0;l<LVE;l++){
            indr = l*NX*NY*NZ + gidx*NY*NZ + gidy*NZ +gidz;
            sumrxy= sumrxy + rxy[indr];
            sumryz= sumryz + ryz[indr];
            sumrxz= sumrxz + rxz[indr];
            sumrxx= sumrxx + rxx[indr];
            sumryy= sumryy + ryy[indr];
            sumrzz= sumrzz + rzz[indr];
        }

        /* updating components of the stress tensor, partially */
        lsxy=lsxy + fipjp*(vx_y1+vy_x1)+(DT2*sumrxy);
        lsyz=lsyz + fjpkp*(vy_z1+vz_y1)+(DT2*sumryz);
        lsxz=lsxz + fipkp*(vx_z1+vz_x1)+(DT2*sumrxz);
        lsxx=lsxx + g*(vx_x2+vy_y2+vz_z2) - f*(vy_y2+vz_z2)+(DT2*sumrxx);
        lsyy=lsyy + g*(vx_x2+vy_y2+vz_z2) - f*(vx_x2+vz_z2)+(DT2*sumryy);
        lszz=lszz + g*(vx_x2+vy_y2+vz_z2) - f*(vx_x2+vy_y2)+(DT2*sumrzz);

        initc0(sumrxy);initc0(sumryz);initc0(sumrxz);
        initc0(sumrxx);initc0(sumryy);initc0(sumrzz);
        float b,c;
        for (l=0;l<LVE;l++){
            b=1.0f/(1.0f+(leta[l]*0.5f));
            c=1.0f-(leta[l]*0.5f);
            indr = l*NX*NY*NZ + gidx*NY*NZ + gidy*NZ +gidz;
            rxy[indr]=__f22h2(b*(rxy[indr]*c-leta[l]*(dipjp*(vx_y1+vy_x1))));
            ryz[indr]=__f22h2(b*(ryz[indr]*c-leta[l]*(djpkp*(vy_z1+vz_y1))));
            rxz[indr]=__f22h2(b*(rxz[indr]*c-leta[l]*(dipkp*(vx_z1+vz_x1))));
            rxx[indr]=__f22h2(b*(rxx[indr]*c-leta[l]*((e*(vx_x2+vy_y2+vz_z2))-(d*(vy_y2+vz_z2)))));
            ryy[indr]=__f22h2(b*(ryy[indr]*c-leta[l]*((e*(vx_x2+vy_y2+vz_z2))-(d*(vx_x2+vz_z2)))));
            rzz[indr]=__f22h2(b*(rzz[indr]*c-leta[l]*((e*(vx_x2+vy_y2+vz_z2))-(d*(vx_x2+vy_y2)))));

            sumrxy= sumrxy + rxy[indr];
            sumryz= sumryz + ryz[indr];
            sumrxz= sumrxz + rxz[indr];
            sumrxx= sumrxx + rxx[indr];
            sumryy= sumryy + ryy[indr];
            sumrzz= sumrzz + rzz[indr];
        }

        /* and now the components of the stress tensor are
         completely updated */
        lsxy=lsxy+(DT2*sumrxy);
        lsyz=lsyz+(DT2*sumryz);
        lsxz=lsxz+(DT2*sumrxz);
        lsxx=lsxx+(DT2*sumrxx);
        lsyy=lsyy+(DT2*sumryy);
        lszz=lszz+(DT2*sumrzz);
    }
    #endif

    // Absorbing boundary
    #if ABS_TYPE==2
        {
        #if FREESURF==0
        if (DIV*gidz-FDOH<NAB){
            lsxy = lsxy * __hpg(&taper[DIV*gidz-FDOH]);
            lsyz = lsyz * __hpg(&taper[DIV*gidz-FDOH]);
            lsxz = lsxz * __hpg(&taper[DIV*gidz-FDOH]);
            lsxx = lsxx * __hpg(&taper[DIV*gidz-FDOH]);
            lsyy = lsyy * __hpg(&taper[DIV*gidz-FDOH]);
            lszz = lszz * __hpg(&taper[DIV*gidz-FDOH]);
        }
        #endif

        if (DIV*gidz>DIV*NZ-NAB-FDOH-1){
            lsxy = lsxy * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lsyz = lsyz * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lsxz = lsxz * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lsxx = lsxx * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lsyy = lsyy * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lszz = lszz * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
        }
        if (gidy-FDOH<NAB){
            lsxy = lsxy * taper[gidy-FDOH];
            lsyz = lsyz * taper[gidy-FDOH];
            lsxz = lsxz * taper[gidy-FDOH];
            lsxx = lsxx * taper[gidy-FDOH];
            lsyy = lsyy * taper[gidy-FDOH];
            lszz = lszz * taper[gidy-FDOH];
        }

        if (gidy>NY-NAB-FDOH-1){
            lsxy = lsxy * taper[NY-FDOH-gidy-1];
            lsyz = lsyz * taper[NY-FDOH-gidy-1];
            lsxz = lsxz * taper[NY-FDOH-gidy-1];
            lsxx = lsxx * taper[NY-FDOH-gidy-1];
            lsyy = lsyy * taper[NY-FDOH-gidy-1];
            lszz = lszz * taper[NY-FDOH-gidy-1];
        }
        #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            lsxy = lsxy * taper[gidx-FDOH];
            lsyz = lsyz * taper[gidx-FDOH];
            lsxz = lsxz * taper[gidx-FDOH];
            lsxx = lsxx * taper[gidx-FDOH];
            lsyy = lsyy * taper[gidx-FDOH];
            lszz = lszz * taper[gidx-FDOH];
        }
        #endif

        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            lsxy = lsxy * taper[NX-FDOH-gidx-1];
            lsyz = lsyz * taper[NX-FDOH-gidx-1];
            lsxz = lsxz * taper[NX-FDOH-gidx-1];
            lsxx = lsxx * taper[NX-FDOH-gidx-1];
            lsyy = lsyy * taper[NX-FDOH-gidx-1];
            lszz = lszz * taper[NX-FDOH-gidx-1];

        }
        #endif
        }
    #endif

    //Write updated values to global memory
    sxx[indv] = __f22h2(lsxx);
    sxy[indv] = __f22h2(lsxy);
    sxz[indv] = __f22h2(lsxz);
    syy[indv] = __f22h2(lsyy);
    syz[indv] = __f22h2(lsyz);
    szz[indv] = __f22h2(lszz);

}
