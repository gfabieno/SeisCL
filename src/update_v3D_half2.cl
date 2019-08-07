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

/*Update of the velocity in 3D, The variable FP16 is
 used to control how FP16 is used: 1: FP32, 2: FP16 IO only, 3: FP16 IO and
 arithmetics*/

FUNDEF void update_v(int offcomm,
                     GLOBARG __pprec *rip, GLOBARG __pprec *rjp,
                     GLOBARG __pprec *rkp,
                     GLOBARG __prec2 *sxx, GLOBARG __prec2 *sxy,
                     GLOBARG __prec2 *sxz, GLOBARG __prec2 *syy,
                     GLOBARG __prec2 *syz, GLOBARG __prec2 *szz,
                     GLOBARG __prec2 *vx,  GLOBARG __prec2 *vy,
                     GLOBARG __prec2 *vz, float *taper)
{
    //Local memory
    extern __shared__ __prec2 lvar2[];
    __prec * lvar=(__prec *)lvar2;
    
    //Grid position
    int lsizez = blockDim.x+2*FDOH/DIV;
    int lsizey = blockDim.y+2*FDOH;
    int lsizex = blockDim.z+2*FDOH;
    int lidz = threadIdx.x+FDOH/DIV;
    int lidy = threadIdx.y+FDOH;
    int lidx = threadIdx.z+FDOH;
    int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/DIV;
    int gidy = blockIdx.y*blockDim.y+threadIdx.y+FDOH;
    int gidx = blockIdx.z*blockDim.z+threadIdx.z+FDOH+offcomm;
    
    int indp = ((gidx)-FDOH)*(NZ-2*FDOH/DIV)*(NY-2*FDOH)+((gidy)-FDOH)*(NZ-2*FDOH/DIV)+((gidz)-FDOH/DIV);
    int indv = (gidx)*NZ*NY+(gidy)*NZ+(gidz);
    
    //Define private derivatives
    __cprec sxx_x1;
    __cprec sxy_x2;
    __cprec sxy_y2;
    __cprec sxz_x2;
    __cprec sxz_z2;
    __cprec syy_y1;
    __cprec syz_y2;
    __cprec syz_z2;
    __cprec szz_z1;
    
    //Local memory definitions if local is used
    #if LOCAL_OFF==0
        #define lszz lvar
        #define lsxx lvar
        #define lsxz lvar
        #define lsyz lvar
        #define lsyy lvar
        #define lsxy lvar
        #define lszz2 lvar2
        #define lsxx2 lvar2
        #define lsxz2 lvar2
        #define lsyz2 lvar2
        #define lsyy2 lvar2
        #define lsxy2 lvar2
    
    //Local memory definitions if local is not used
    #elif LOCAL_OFF==1
    
        #define lszz szz
        #define lsxx sxx
        #define lsxz sxz
        #define lsyz syz
        #define lsyy syy
        #define lsxy sxy
        #define lidz gidz
        #define lidy gidy
        #define lidx gidx
    
    #endif
    
    //Calculation of the spatial derivatives
    {
#if LOCAL_OFF==0
        load_local_in(szz);
        load_local_haloz(szz);
        BARRIER
#endif
        szz_z1 = Dzp(lszz);

        
#if LOCAL_OFF==0
        BARRIER
        load_local_in(sxx);
        load_local_halox(sxx);
        BARRIER
#endif
        sxx_x1 = Dxp(lsxx2);

        
#if LOCAL_OFF==0
        BARRIER
        load_local_in(sxz);
        load_local_haloz(sxz);
        load_local_halox(sxz);
        BARRIER
#endif
        sxz_x2 = Dxm(lsxz2);
        sxz_z2 = Dzm(lsxz);
        
#if LOCAL_OFF==0
        BARRIER
        load_local_in(syz);
        load_local_haloz(syz);
        load_local_haloy(syz);
        BARRIER
#endif
        syz_y2 = Dym(lsyz2);
        syz_z2 = Dzm(lsyz);
        
#if LOCAL_OFF==0
        BARRIER
        load_local_in(syy);
        load_local_haloy(syy);
        BARRIER
#endif
        syy_y1 = Dyp(lsyy2);

        
#if LOCAL_OFF==0
        BARRIER
        load_local_in(sxy);
        load_local_halox(sxy);
        load_local_haloy(sxy);
        BARRIER
#endif
        sxy_x2 = Dxm(lsxy2);
        sxy_y2 = Dym(lsxy2);
  
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
    __cprec lvx = __h22f2(vx[indv]);
    __cprec lvy = __h22f2(vy[indv]);
    __cprec lvz = __h22f2(vz[indv]);
    
    // Update the variables
    lvx = lvx + (sxx_x1 + sxy_y2 + sxz_z2) * __pconv(rip[indp]);
    lvy = lvy + (syy_y1 + sxy_x2 + syz_z2) * __pconv(rjp[indp]);
    lvz = lvz + (szz_z1 + sxz_x2 + syz_y2) * __pconv(rkp[indp]);
    
    // Absorbing boundary
    #if ABS_TYPE==2
        {
        #if FREESURF==0
        if (DIV*gidz-FDOH<NAB){
            lvx = lvx * __hp(&taper[DIV*gidz-FDOH]);
            lvy = lvy * __hp(&taper[DIV*gidz-FDOH]);
            lvz = lvz * __hp(&taper[DIV*gidz-FDOH]);
        }
        #endif

        if (DIV*gidz>DIV*NZ-NAB-FDOH-1){
            lvx = lvx * __hpi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lvy = lvy * __hpi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lvz = lvz * __hpi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
        }
        if (gidy-FDOH<NAB){
            lvx = lvx * taper[gidy-FDOH];
            lvy = lvy * taper[gidy-FDOH];
            lvz = lvz * taper[gidy-FDOH];
        }

        if (gidy>NY-NAB-FDOH-1){
            lvx = lvx * taper[NY-FDOH-gidy-1];
            lvy = lvy * taper[NY-FDOH-gidy-1];
            lvz = lvz * taper[NY-FDOH-gidy-1];
        }
        #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            lvx = lvx * taper[gidx-FDOH];
            lvy = lvy * taper[gidx-FDOH];
            lvz = lvz * taper[gidx-FDOH];
        }
        #endif

        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            lvx = lvx * taper[NX-FDOH-gidx-1];
            lvy = lvy * taper[NX-FDOH-gidx-1];
            lvz = lvz * taper[NX-FDOH-gidx-1];
        }
        #endif
        }
    #endif

    //Write updated values to global memory
    vx[indv] = __f22h2(lvx);
    vy[indv] = __f22h2(lvy);
    vz[indv] = __f22h2(lvz);
}



