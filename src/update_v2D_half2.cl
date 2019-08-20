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

/*Update of the velocity in 2D SV using half precision. The variable FP16 is
 used to control how FP16 is used: 1: FP32, 2: FP16 IO only, 3: FP16 IO and
 arithmetics*/

FUNDEF void update_v(int offcomm,
                     GLOBARG __pprec *rip,       GLOBARG __pprec *rkp,
                     GLOBARG __prec2 *sxx,       GLOBARG __prec2 *sxz,
                     GLOBARG __prec2 *szz,       GLOBARG __prec2 *vx,
                     GLOBARG __prec2 *vz,        GLOBARG float *taper,
                     GLOBARG float *K_z,         GLOBARG float *a_z,
                     GLOBARG float *b_z,         GLOBARG float *K_z_half,
                     GLOBARG float *a_z_half,    GLOBARG float *b_z_half,
                     GLOBARG float *K_x,         GLOBARG float *a_x,
                     GLOBARG float *b_x,         GLOBARG float *K_x_half,
                     GLOBARG float *a_x_half,    GLOBARG float *b_x_half,
                     GLOBARG __prec2 *psi_sxx_x, GLOBARG __prec2 *psi_sxz_x,
                     GLOBARG __prec2 *psi_sxz_z, GLOBARG __prec2 *psi_szz_z,
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
    // If we use local memory
    #if LOCAL_OFF==0
        #ifdef __OPENCL_VERSION__
        int lsizez = get_local_size(0)+2*FDOH/DIV;
        int lsizex = get_local_size(1)+2*FDOH;
        int lidz = get_local_id(0)+FDOH/DIV;
        int lidx = get_local_id(1)+FDOH;
        int gidz = get_global_id(0)+FDOH/DIV;
        int gidx = get_global_id(1)+FDOH+offcomm;
        #else
        int lsizez = blockDim.x+2*FDOH/DIV;
        int lsizex = blockDim.y+2*FDOH;
        int lidz = threadIdx.x+FDOH/DIV;
        int lidx = threadIdx.y+FDOH;
        int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/DIV;
        int gidx = blockIdx.y*blockDim.y+threadIdx.y+FDOH+offcomm;
        #endif

    // If local memory is turned off
    #elif LOCAL_OFF==1
        #ifdef __OPENCL_VERSION__
        int gid = get_global_id(0);
        int glsizez = (NZ-2*FDOH/DIV);
        int gidz = gid%glsizez+FDOH/DIV;
        int gidx = (gid/glsizez)+FDOH+offcomm;
        #else
        int lsizez = blockDim.x+2*FDOH/DIV;
        int lsizex = blockDim.y+2*FDOH;
        int lidz = threadIdx.x+FDOH/DIV;
        int lidx = threadIdx.y+FDOH;
        int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/DIV;
        int gidx = blockIdx.y*blockDim.y+threadIdx.y+FDOH+offcomm;
        #endif

    #endif

    int indp = ((gidx)-FDOH)*(NZ-2*FDOH/DIV)+((gidz)-FDOH/DIV);
    int indv = gidx*NZ+gidz;


    //Define private derivatives
    __cprec sxx_x1;
    __cprec sxz_x2;
    __cprec sxz_z2;
    __cprec szz_z1;

    //Local memory definitions if local is used
    #if LOCAL_OFF==0
        #define lsxx lvar
        #define lszz lvar
        #define lsxz lvar
        #define lsxx2 lvar2
        #define lszz2 lvar2
        #define lsxz2 lvar2

    //Local memory definitions if local is not used
    #elif LOCAL_OFF==1

        #define lsxx sxx
        #define lszz szz
        #define lsxz sxz
        #define lidz gidz
        #define lidx gidx

    #endif

    //Calculation of the spatial derivatives
    {
    #if LOCAL_OFF==0
        load_local_in(sxx);
        load_local_halox(sxx);
        BARRIER
    #endif
        sxx_x1 = Dxp(lsxx2);

    #if LOCAL_OFF==0
        BARRIER
        load_local_in(szz);
        load_local_haloz(szz);
        BARRIER
    #endif
        szz_z1 = Dzp(lszz);

    #if LOCAL_OFF==0
        BARRIER
        load_local_in(sxz);
        load_local_halox(sxz);
        load_local_haloz(sxz);
        BARRIER
    #endif
        sxz_x2 = Dxm(lsxz2);
        sxz_z2 = Dzm(lsxz);

    }

    // To stop updating if we are outside the model (global id must be a
    //multiple of local id in OpenCL, hence we stop if we have a global id
    //outside the grid)
    #if  LOCAL_OFF==0
    #if COMM12==0
    if ( gidz>(NZ-FDOH/DIV-1) ||  (gidx-offcomm)>(NX-FDOH-1-LCOMM) )
        return;
    #else
    if ( gidz>(NZ-FDOH/DIV-1)  )
        return;
    #endif
    #endif

     // Correct spatial derivatives to implement CPML
    #if ABS_TYPE==1
    {
        int i,k,indm,indn;
        if (DIV*gidz>DIV*NZ-NAB-FDOH-1){
            i =gidx - FDOH;
            k =gidz - NZ + 2*NAB/DIV + FDOH/DIV;
            indm=2*NAB - 1 - k*DIV;
            indn = (i)*(2*NAB/DIV)+(k);

            psi_sxz_z[indn] = __f22h2(__hpgi(&b_z[indm+1]) * psi_sxz_z[indn]
                                      + __hpgi(&a_z[indm+1]) * sxz_z2);
            sxz_z2 = sxz_z2 / __hpgi(&K_z[indm+1]) + psi_sxz_z[indn];
            psi_szz_z[indn] = __f22h2(__hpgi(&b_z_half[indm]) * psi_szz_z[indn]
                                      + __hpgi(&a_z_half[indm]) * szz_z1);
            szz_z1 = szz_z1 / __hpgi(&K_z_half[indm]) + psi_szz_z[indn];
        }

        #if FREESURF==0
        if (DIV*gidz-FDOH<NAB){
            i =gidx-FDOH;
            k =gidz*DIV-FDOH;
            indn = (i)*(2*NAB/DIV)+(k/DIV);

            psi_sxz_z[indn] = __f22h2(__hpg(&b_z[k]) * psi_sxz_z[indn]
                                      + __hpg(&a_z[k]) * sxz_z2);
            sxz_z2 = sxz_z2 / __hpg(&K_z[k]) + psi_sxz_z[indn];
            psi_szz_z[indn] = __f22h2(__hpg(&b_z_half[k]) * psi_szz_z[indn]
                                      + __hpg(&a_z_half[k]) * szz_z1);
            szz_z1 = szz_z1 / __hpg(&K_z_half[k]) + psi_szz_z[indn];
        }
        #endif

        #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            i =gidx-FDOH;
            k =gidz-FDOH/DIV;
            indn = (i)*(NZ-2*FDOH/DIV)+(k);

            psi_sxx_x[indn] = __f22h2(b_x_half[i] * psi_sxx_x[indn]
                                      + a_x_half[i] * sxx_x1);
            sxx_x1 = sxx_x1 / K_x_half[i] + psi_sxx_x[indn];
            psi_sxz_x[indn] = __f22h2(b_x[i] * psi_sxz_x[indn]
                                      + a_x[i] * sxz_x2);
            sxz_x2 = sxz_x2 / K_x[i] + psi_sxz_x[indn];
        }
        #endif

        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            i =gidx - NX+NAB+FDOH+NAB;
            k =gidz-FDOH/DIV;
            indm=2*NAB-1-i;
            indn = (i)*(NZ-2*FDOH/DIV)+(k);

            psi_sxx_x[indn] = __f22h2(b_x_half[indm] * psi_sxx_x[indn]
                                      + a_x_half[indm] * sxx_x1);
            sxx_x1 = sxx_x1 / K_x_half[indm] + psi_sxx_x[indn];
            psi_sxz_x[indn] = __f22h2(b_x[indm+1] * psi_sxz_x[indn]
                                      + a_x[indm+1] * sxz_x2);
            sxz_x2 = sxz_x2 / K_x[indm+1] + psi_sxz_x[indn];
        }
        #endif
       }
    #endif

    //Define and load private parameters and variables
    __cprec lvx = __h22f2(vx[indv]);
    __cprec lvz = __h22f2(vz[indv]);
    __cprec lrip = __pconv(rip[indp]);
    __cprec lrkp = __pconv(rkp[indp]);

    // Update the variables
    lvx=lvx+(sxx_x1+sxz_z2)*lrip;
    lvz=lvz+(szz_z1+sxz_x2)*lrkp;
    #if ABS_TYPE==2
    {
    #if FREESURF==0
        if (DIV*gidz-FDOH<NAB){
            lvx = lvx * __hpg(&taper[DIV*gidz-FDOH]);
            lvz = lvz * __hpg(&taper[DIV*gidz-FDOH]);
        }
    #endif
        if (DIV*gidz>DIV*NZ-NAB-FDOH-1){
            lvx = lvx * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lvz = lvz * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
        }

    #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            lvx = lvx * taper[gidx-FDOH];
            lvz = lvz * taper[gidx-FDOH];
        }
    #endif

    #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            lvx = lvx * taper[NX-FDOH-gidx-1];
            lvz = lvz * taper[NX-FDOH-gidx-1];
        }
    #endif
    }
    #endif

    //Write updated values to global memory
    vx[indv] = __f22h2(lvx);
    vz[indv] = __f22h2(lvz);
}
