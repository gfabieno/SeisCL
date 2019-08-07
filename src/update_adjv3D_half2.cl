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

FUNDEF void update_adjv(int offcomm,
                        GLOBARG __pprec *rip, GLOBARG __pprec *rjp,
                        GLOBARG __pprec *rkp,
                        GLOBARG __prec2 *sxx, GLOBARG __prec2 *sxy,
                        GLOBARG __prec2 *sxz, GLOBARG __prec2 *syy,
                        GLOBARG __prec2 *syz, GLOBARG __prec2 *szz,
                        GLOBARG __prec2 *vx,  GLOBARG __prec2 *vy,
                        GLOBARG __prec2 *vz,
                        GLOBARG __prec2 *vxbnd, GLOBARG __prec2 *vybnd,
                        GLOBARG __prec2 *vzbnd,
                        GLOBARG __prec2 *sxxr, GLOBARG __prec2 *sxyr,
                        GLOBARG __prec2 *sxzr, GLOBARG __prec2 *syyr,
                        GLOBARG __prec2 *syzr, GLOBARG __prec2 *szzr,
                        GLOBARG __prec2 *vxr,  GLOBARG __prec2 *vyr,
                        GLOBARG __prec2 *vzr,
                        GLOBARG float *taper,
                        GLOBARG __gprec *gradrho, GLOBARG __gprec *Hrho,
                        int res_scale, int src_scale, int par_scale,
                        LOCARG2)
{
    //Local memory
    LOCDEF2
    #ifdef __OPENCL_VERSION__
    __local __prec * lvar=lvar2;
    #else
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
    __cprec sxx_x1;
    __cprec sxy_x2;
    __cprec sxy_y2;
    __cprec sxz_x2;
    __cprec sxz_z2;
    __cprec syy_y1;
    __cprec syz_y2;
    __cprec syz_z2;
    __cprec szz_z1;

    __cprec sxxr_x1;
    __cprec sxyr_x2;
    __cprec sxyr_y2;
    __cprec sxzr_x2;
    __cprec sxzr_z2;
    __cprec syyr_y1;
    __cprec syzr_y2;
    __cprec syzr_z2;
    __cprec szzr_z1;

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

        #define lszzr lvar
        #define lsxxr lvar
        #define lsxzr lvar
        #define lsyzr lvar
        #define lsyyr lvar
        #define lsxyr lvar
        #define lszzr2 lvar2
        #define lsxxr2 lvar2
        #define lsxzr2 lvar2
        #define lsyzr2 lvar2
        #define lsyyr2 lvar2
        #define lsxyr2 lvar2
    //Local memory definitions if local is not used
    #elif LOCAL_OFF==1

        #define lszz szz
        #define lsxx sxx
        #define lsxz sxz
        #define lsyz syz
        #define lsyy syy
        #define lsxy sxy

        #define lszzr szzr
        #define lsxxr sxxr
        #define lsxzr sxzr
        #define lsyzr syzr
        #define lsyyr syyr
        #define lsxyr sxyr

        #define lidz gidz
        #define lidy gidy
        #define lidx gidx

    #endif
    
#if BACK_PROP_TYPE==1
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
        BARRIER
    }
#endif


    //Calculation of the spatial derivatives
    {
    #if LOCAL_OFF==0
        load_local_in(szzr);
        load_local_haloz(szzr);
        BARRIER
    #endif
        szzr_z1 = Dzp(lszzr);


    #if LOCAL_OFF==0
        BARRIER
        load_local_in(sxxr);
        load_local_halox(sxxr);
        BARRIER
    #endif
        sxxr_x1 = Dxp(lsxxr2);


    #if LOCAL_OFF==0
        BARRIER
        load_local_in(sxzr);
        load_local_haloz(sxzr);
        load_local_halox(sxzr);
        BARRIER
    #endif
        sxzr_x2 = Dxm(lsxzr2);
        sxzr_z2 = Dzm(lsxzr);

    #if LOCAL_OFF==0
        BARRIER
        load_local_in(syzr);
        load_local_haloz(syzr);
        load_local_haloy(syzr);
        BARRIER
    #endif
        syzr_y2 = Dym(lsyzr2);
        syzr_z2 = Dzm(lsyzr);

    #if LOCAL_OFF==0
        BARRIER
        load_local_in(syyr);
        load_local_haloy(syyr);
        BARRIER
    #endif
        syyr_y1 = Dyp(lsyyr2);


    #if LOCAL_OFF==0
        BARRIER
        load_local_in(sxyr);
        load_local_halox(sxyr);
        load_local_haloy(sxyr);
        BARRIER
    #endif
        sxyr_x2 = Dxm(lsxyr2);
        sxyr_y2 = Dym(lsxyr2);

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
    
    __cprec lvxr = __h22f2(vxr[indv]);
    __cprec lvyr = __h22f2(vyr[indv]);
    __cprec lvzr = __h22f2(vzr[indv]);

    __cprec lrip = __pconv(rip[indp]);
    __cprec lrjp = __pconv(rjp[indp]);
    __cprec lrkp = __pconv(rkp[indp]);
    
    // Backpropagate the forward velocities
    #if BACK_PROP_TYPE==1
    __cprec lvx = __h22f2(vx[indv]);
    __cprec lvy = __h22f2(vy[indv]);
    __cprec lvz = __h22f2(vz[indv]);
    {

        // Update the variables
        lvx = lvx - (sxx_x1 + sxy_y2 + sxz_z2) * lrip;
        lvy = lvy - (syy_y1 + sxy_x2 + syz_z2) * lrjp;
        lvz = lvz - (szz_z1 + sxz_x2 + syz_y2) * lrkp;

        int m=inject_ind(gidz, gidy, gidx);
        if (m!=-1){
            lvx= __h22f2(vxbnd[m]);
            lvy= __h22f2(vybnd[m]);
            lvz= __h22f2(vzbnd[m]);
        }

        //Write updated values to global memory
        vx[indv] = __f22h2(lvx);
        vy[indv] = __f22h2(lvy);
        vz[indv] = __f22h2(lvz);

    }
    #endif
    // Update adjoint velocties
    {
        // Update the variables
        lvxr = lvxr + (sxxr_x1 + sxyr_y2 + sxzr_z2) * lrip;
        lvyr = lvyr + (syyr_y1 + sxyr_x2 + syzr_z2) * lrjp;
        lvzr = lvzr + (szzr_z1 + sxzr_x2 + syzr_y2) * lrkp;

    // Absorbing boundary
    #if ABS_TYPE==2
        {
        #if FREESURF==0
        if (DIV*gidz-FDOH<NAB){
            lvxr = lvxr * __hpg(&taper[DIV*gidz-FDOH]);
            lvyr = lvyr * __hpg(&taper[DIV*gidz-FDOH]);
            lvzr = lvzr * __hpg(&taper[DIV*gidz-FDOH]);
        }
        #endif

        if (DIV*gidz>DIV*NZ-NAB-FDOH-1){
            lvxr = lvxr * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lvyr = lvyr * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lvzr = lvzr * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
        }
        if (gidy-FDOH<NAB){
            lvxr = lvxr * taper[gidy-FDOH];
            lvyr = lvyr * taper[gidy-FDOH];
            lvzr = lvzr * taper[gidy-FDOH];
        }

        if (gidy>NY-NAB-FDOH-1){
            lvxr = lvxr * taper[NY-FDOH-gidy-1];
            lvyr = lvyr * taper[NY-FDOH-gidy-1];
            lvzr = lvzr * taper[NY-FDOH-gidy-1];
        }
        #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            lvxr = lvxr * taper[gidx-FDOH];
            lvyr = lvyr * taper[gidx-FDOH];
            lvzr = lvzr * taper[gidx-FDOH];
        }
        #endif

        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            lvxr = lvxr * taper[NX-FDOH-gidx-1];
            lvyr = lvyr * taper[NX-FDOH-gidx-1];
            lvzr = lvzr * taper[NX-FDOH-gidx-1];
        }
        #endif
        }
    #endif

        //Write updated values to global memory
        vxr[indv] = __f22h2(lvxr);
        vyr[indv] = __f22h2(lvyr);
        vzr[indv] = __f22h2(lvzr);
    }

    // Density gradient calculation on the fly
    #if BACK_PROP_TYPE==1
    lvxr = (sxxr_x1 + sxyr_y2 + sxzr_z2) * lrip;
    lvyr = (syyr_y1 + sxyr_x2 + syzr_z2) * lrjp;
    lvzr = (szzr_z1 + sxzr_x2 + syzr_y2) * lrkp;

    gradrho[indp]=gradrho[indp] - scalefun(lvx*lvxr + lvy*lvyr + lvz*lvzr,
                                           2*par_scale -src_scale - res_scale);
    #if HOUT==1
    Hrho[indp]=Hrho[indp] + scalefun(lvx*lvx + lvy*lvy + lvz*lvz,
                                     2*par_scale - 2*src_scale);
    #endif

    #endif
}



