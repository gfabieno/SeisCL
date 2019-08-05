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
                        __pprec *rip, __pprec *rjp, __pprec *rkp,
                        __prec2 *sxx,__prec2 *sxy,__prec2 *sxz,
                        __prec2 *syy,__prec2 *syz,__prec2 *szz,
                        __prec2 *vx,__prec2 *vy,__prec2 *vz,
                        __prec2 *vxbnd,__prec2 *vybnd,__prec2 *vzbnd,
                        __prec2 *sxxr,__prec2 *sxyr,__prec2 *sxzr,
                        __prec2 *syyr,__prec2 *syzr,__prec2 *szzr,
                        __prec2 *vxr,__prec2 *vyr,__prec2 *vzr,
                        float *taper,  float2 *gradrho, float2 *Hrho,
                        int res_scale, int src_scale, int par_scale
                     )

{
    //Local memory
    extern __shared__ __prec2 lvar2[];
    __prec * lvar=(__prec *)lvar2;
    
    //Grid position
    int lsizez = blockDim.x+FDOH;
    int lsizey = blockDim.y+2*FDOH;
    int lsizex = blockDim.z+2*FDOH;
    int lidz = threadIdx.x+FDOH/2;
    int lidy = threadIdx.y+FDOH;
    int lidx = threadIdx.z+FDOH;
    int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/2;
    int gidy = blockIdx.y*blockDim.y+threadIdx.y+FDOH;
    int gidx = blockIdx.z*blockDim.z+threadIdx.z+FDOH+offcomm;
    
    int indp = ((gidx)-FDOH)*(NZ-FDOH)*(NY-2*FDOH)+((gidy)-FDOH)*(NZ-FDOH)+((gidz)-FDOH/2);
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
    
    
    // To stop updating if we are outside the model (global id must be amultiple of local id in OpenCL, hence we stop if we have a global idoutside the grid)
#if  LOCAL_OFF==0
#if COMM12==0
    if ( gidz>(NZ-FDOH/2-1) ||  gidy>(NY-FDOH-1) ||  (gidx-offcomm)>(NX-FDOH-1-LCOMM) )
        return;
#else
    if ( gidz>(NZ-FDOH/2-1) ||  gidy>(NY-FDOH-1)  )
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
    {
        __cprec lvx = __h22f2(vx[indv]);
        __cprec lvy = __h22f2(vy[indv]);
        __cprec lvz = __h22f2(vz[indv]);
        
        // Update the variables
        lvx=sub2(lvx,mul2(add2(add2(sxx_x1,sxy_y2),sxz_z2),lrip));
        lvy=sub2(lvy,mul2(add2(add2(syy_y1,sxy_x2),syz_z2),lrjp));
        lvz=sub2(lvz,mul2(add2(add2(szz_z1,sxz_x2),syz_y2),lrkp));
        
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
        lvxr=add2(lvxr,mul2(add2(add2(sxxr_x1,sxyr_y2),sxzr_z2),lrip));
        lvyr=add2(lvyr,mul2(add2(add2(syyr_y1,sxyr_x2),syzr_z2),lrjp));
        lvzr=add2(lvzr,mul2(add2(add2(szzr_z1,sxzr_x2),syzr_y2),lrkp));
        
        
    // Absorbing boundary
    #if ABS_TYPE==2
        {
        #if FREESURF==0
            if (2*gidz-FDOH<NAB){
                lvxr.x*=taper[2*gidz-FDOH];
                lvxr.y*=taper[2*gidz-FDOH+1];
                lvyr.x*=taper[2*gidz-FDOH];
                lvyr.y*=taper[2*gidz-FDOH+1];
                lvzr.x*=taper[2*gidz-FDOH];
                lvzr.y*=taper[2*gidz-FDOH+1];
            }
        #endif
        
            if (2*gidz>2*NZ-NAB-FDOH-1){
                lvxr.x*=taper[2*NZ-FDOH-2*gidz-1];
                lvxr.y*=taper[2*NZ-FDOH-2*gidz-2];
                lvyr.x*=taper[2*NZ-FDOH-2*gidz-1];
                lvyr.y*=taper[2*NZ-FDOH-2*gidz-2];
                lvzr.x*=taper[2*NZ-FDOH-2*gidz-1];
                lvzr.y*=taper[2*NZ-FDOH-2*gidz-2];
            }
        
            if (gidy-FDOH<NAB){
                lvxr.x*=taper[gidy-FDOH];
                lvxr.y*=taper[gidy-FDOH];
                lvyr.x*=taper[gidy-FDOH];
                lvyr.y*=taper[gidy-FDOH];
                lvzr.x*=taper[gidy-FDOH];
                lvzr.y*=taper[gidy-FDOH];
            }
        
            if (gidy>NY-NAB-FDOH-1){
                lvxr.x*=taper[NY-FDOH-gidy-1];
                lvxr.y*=taper[NY-FDOH-gidy-1];
                lvyr.x*=taper[NY-FDOH-gidy-1];
                lvyr.y*=taper[NY-FDOH-gidy-1];
                lvzr.x*=taper[NY-FDOH-gidy-1];
                lvzr.y*=taper[NY-FDOH-gidy-1];
            }
        #if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB){
                lvxr.x*=taper[gidx-FDOH];
                lvxr.y*=taper[gidx-FDOH];
                lvyr.x*=taper[gidx-FDOH];
                lvyr.y*=taper[gidx-FDOH];
                lvzr.x*=taper[gidx-FDOH];
                lvzr.y*=taper[gidx-FDOH];
            }
        #endif
        
        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1){
                lvxr.x*=taper[NX-FDOH-gidx-1];
                lvxr.y*=taper[NX-FDOH-gidx-1];
                lvyr.x*=taper[NX-FDOH-gidx-1];
                lvyr.y*=taper[NX-FDOH-gidx-1];
                lvzr.x*=taper[NX-FDOH-gidx-1];
                lvzr.y*=taper[NX-FDOH-gidx-1];
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
    lvxr=mul2(add2(add2(sxxr_x1,sxyr_y2),sxzr_z2),lrip);
    lvyr=mul2(add2(add2(syyr_y1,sxyr_x2),syzr_z2),lrjp);
    lvzr=mul2(add2(add2(szzr_z1,sxzr_x2),syzr_y2),lrkp);
    
    gradrho[indp]=sub2f( gradrho[indp], scalbnf2(add2f(add2f( mul2f( __h22f2c(lvx), __h22f2c(lvxr)), mul2f( __h22f2c(lvy), __h22f2c(lvyr))), mul2f( __h22f2c(lvz), __h22f2c(lvzr))), 2*par_scale -src_scale - res_scale) );
    #if HOUT==1
    Hrho[indp]= sub2f( Hrho[indp], scalbnf2(add2f( add2f(mul2f( __h22f2c(lvx), __h22f2c(lvx)), mul2f( __h22f2c(lvy), __h22f2c(lvy))), mul2f( __h22f2c(lvz), __h22f2c(lvz))), 2*par_scale-2*src_scale));
    #endif
    
    #endif
}



