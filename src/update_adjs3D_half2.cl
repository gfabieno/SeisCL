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
                        __pprec *muipjp, __pprec *mujpkp, __pprec *muipkp,
                        __pprec *M, __pprec *mu,
                        __prec2 *sxx,__prec2 *sxy,__prec2 *sxz,
                        __prec2 *syy,__prec2 *syz,__prec2 *szz,
                        __prec2 *vx,__prec2 *vy,__prec2 *vz,
                        __prec2 *sxxbnd,__prec2 *sxybnd,__prec2 *sxzbnd,
                        __prec2 *syybnd,__prec2 *syzbnd,__prec2 *szzbnd,
                        __prec2 *sxxr,__prec2 *sxyr,__prec2 *sxzr,
                        __prec2 *syyr,__prec2 *syzr,__prec2 *szzr,
                        __prec2 *vxr,__prec2 *vyr,__prec2 *vzr,
                        float *taper)

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
    {
        __cprec lsxx = __h22f2(sxx[indv]);
        __cprec lsxy = __h22f2(sxy[indv]);
        __cprec lsxz = __h22f2(sxz[indv]);
        __cprec lsyy = __h22f2(syy[indv]);
        __cprec lsyz = __h22f2(syz[indv]);
        __cprec lszz = __h22f2(szz[indv]);
        
        lsxy=sub2(lsxy,mul2(lmuipjp,add2(vx_y1,vy_x1)));
        lsyz=sub2(lsyz,mul2(lmujpkp,add2(vy_z1,vz_y1)));
        lsxz=sub2(lsxz,mul2(lmuipkp,add2(vx_z1,vz_x1)));
        lsxx=add2(sub2(lsxx,mul2(lM,add2(add2(vx_x2,vy_y2),vz_z2))),mul2(mul2(f2h2(2.0),lmu),add2(vy_y2,vz_z2)));
        lsyy=add2(sub2(lsyy,mul2(lM,add2(add2(vx_x2,vy_y2),vz_z2))),mul2(mul2(f2h2(2.0),lmu),add2(vx_x2,vz_z2)));
        lszz=add2(sub2(lszz,mul2(lM,add2(add2(vx_x2,vy_y2),vz_z2))),mul2(mul2(f2h2(2.0),lmu),add2(vx_x2,vy_y2)));
        
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
    
    
    // Update adjoint stresses
    {
        // Update the variables
        lsxyr=add2(lsxyr,mul2(lmuipjp,add2(vxr_y1,vyr_x1)));
        lsyzr=add2(lsyzr,mul2(lmujpkp,add2(vyr_z1,vzr_y1)));
        lsxzr=add2(lsxzr,mul2(lmuipkp,add2(vxr_z1,vzr_x1)));
        lsxxr=sub2(add2(lsxxr,mul2(lM,add2(add2(vxr_x2,vyr_y2),vzr_z2))),mul2(mul2(f2h2(2.0),lmu),add2(vyr_y2,vzr_z2)));
        lsyyr=sub2(add2(lsyyr,mul2(lM,add2(add2(vxr_x2,vyr_y2),vzr_z2))),mul2(mul2(f2h2(2.0),lmu),add2(vxr_x2,vzr_z2)));
        lszzr=sub2(add2(lszzr,mul2(lM,add2(add2(vxr_x2,vyr_y2),vzr_z2))),mul2(mul2(f2h2(2.0),lmu),add2(vxr_x2,vyr_y2)));
    
    // Absorbing boundary
    #if ABS_TYPE==2
        {
        #if FREESURF==0
        if (2*gidz-FDOH<NAB){
            lsxyr.x*=taper[2*gidz-FDOH];
            lsxyr.y*=taper[2*gidz+1-FDOH];
            lsyzr.x*=taper[2*gidz-FDOH];
            lsyzr.y*=taper[2*gidz+1-FDOH];
            lsxzr.x*=taper[2*gidz-FDOH];
            lsxzr.y*=taper[2*gidz+1-FDOH];
            lsxxr.x*=taper[2*gidz-FDOH];
            lsxxr.y*=taper[2*gidz+1-FDOH];
            lsyyr.x*=taper[2*gidz-FDOH];
            lsyyr.y*=taper[2*gidz+1-FDOH];
            lszzr.x*=taper[2*gidz-FDOH];
            lszzr.y*=taper[2*gidz+1-FDOH];
        }
        #endif
        
        if (2*gidz>2*NZ-NAB-FDOH-1){
            lsxyr.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsxyr.y*=taper[2*NZ-FDOH-2*gidz-2];
            lsyzr.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsyzr.y*=taper[2*NZ-FDOH-2*gidz-2];
            lsxzr.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsxzr.y*=taper[2*NZ-FDOH-2*gidz-2];
            lsxxr.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsxxr.y*=taper[2*NZ-FDOH-2*gidz-2];
            lsyyr.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsyyr.y*=taper[2*NZ-FDOH-2*gidz-2];
            lszzr.x*=taper[2*NZ-FDOH-2*gidz-1];
            lszzr.y*=taper[2*NZ-FDOH-2*gidz-2];
        }
        if (gidy-FDOH<NAB){
            lsxyr.x*=taper[gidy-FDOH];
            lsxyr.y*=taper[gidy-FDOH];
            lsyzr.x*=taper[gidy-FDOH];
            lsyzr.y*=taper[gidy-FDOH];
            lsxzr.x*=taper[gidy-FDOH];
            lsxzr.y*=taper[gidy-FDOH];
            lsxxr.x*=taper[gidy-FDOH];
            lsxxr.y*=taper[gidy-FDOH];
            lsyyr.x*=taper[gidy-FDOH];
            lsyyr.y*=taper[gidy-FDOH];
            lszzr.x*=taper[gidy-FDOH];
            lszzr.y*=taper[gidy-FDOH];
        }
        
        if (gidy>NY-NAB-FDOH-1){
            lsxyr.x*=taper[NY-FDOH-gidy-1];
            lsxyr.y*=taper[NY-FDOH-gidy-1];
            lsyzr.x*=taper[NY-FDOH-gidy-1];
            lsyzr.y*=taper[NY-FDOH-gidy-1];
            lsxzr.x*=taper[NY-FDOH-gidy-1];
            lsxzr.y*=taper[NY-FDOH-gidy-1];
            lsxxr.x*=taper[NY-FDOH-gidy-1];
            lsxxr.y*=taper[NY-FDOH-gidy-1];
            lsyyr.x*=taper[NY-FDOH-gidy-1];
            lsyyr.y*=taper[NY-FDOH-gidy-1];
            lszzr.x*=taper[NY-FDOH-gidy-1];
        }
        #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            lsxyr.x*=taper[gidx-FDOH];
            lsxyr.y*=taper[gidx-FDOH];
            lsyzr.x*=taper[gidx-FDOH];
            lsyzr.y*=taper[gidx-FDOH];
            lsxzr.x*=taper[gidx-FDOH];
            lsxzr.y*=taper[gidx-FDOH];
            lsxxr.x*=taper[gidx-FDOH];
            lsxxr.y*=taper[gidx-FDOH];
            lsyyr.x*=taper[gidx-FDOH];
            lsyyr.y*=taper[gidx-FDOH];
            lszzr.x*=taper[gidx-FDOH];
            lszzr.y*=taper[gidx-FDOH];
        }
        #endif
        
        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            lsxyr.x*=taper[NX-FDOH-gidx-1];
            lsxyr.y*=taper[NX-FDOH-gidx-1];
            lsyzr.x*=taper[NX-FDOH-gidx-1];
            lsyzr.y*=taper[NX-FDOH-gidx-1];
            lsxzr.x*=taper[NX-FDOH-gidx-1];
            lsxzr.y*=taper[NX-FDOH-gidx-1];
            lsxxr.x*=taper[NX-FDOH-gidx-1];
            lsxxr.y*=taper[NX-FDOH-gidx-1];
            lsyyr.x*=taper[NX-FDOH-gidx-1];
            lsyyr.y*=taper[NX-FDOH-gidx-1];
            lszzr.x*=taper[NX-FDOH-gidx-1];
            lszzr.y*=taper[NX-FDOH-gidx-1];
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
    #if RESTYPE==0
    float2 c1=div2f(div2f(f2h2f(1.0),sub2f(mul2(f2h2f(3.0),__h22f2c(lM)),mul2f(f2h2f(4.0),__h22f2c(lmu)))),sub2f(mul2(f2h2f(3.0),__h22f2c(lM)),mul2f(f2h2f(4.0),__h22f2c(lmu))));
    float2 c3=div2f(div2f(f2h2f(1.0),__h22f2c(lmu)),__h22f2c(lmu));
    float2 c5=mul2f(div2f(f2h2f(1.0),f2h2f(6.0)),c3);
   
    
    float2 dM=mul2f(mul2f(c1,__h22f2c(add2(add2(sxx[indv],syy[indv]),szz[indv]))),__h22f2c(add2(add2(lsxx,lsyy),lszz)));
    
    gradM[indp]=sub2f(gradM[indp],scalbnf2(dM, 2*par_scale-src_scale - res_scale));
    
    gradmu[indp] = sub2f(add2f(sub2f(gradmu[indp],mul2f(c3,__h22f2c(add2(add2(mul2(sxz[indv],lsxz),mul2(sxy[indv],lsxy)),mul2(syz[indv],lsyz))))),mul2(div2(f2h2f(4.0),f2h2f(3.0)),dM)),mul2f(c5,__h22f2c(add2(add2(mul2(lsxx,sub2(sub2(mul2(f2h2(2.0),sxx[indv]),syy[indv]),szz[indv])),mul2(lsyy,sub2(sub2(mul2(f2h2(2.0),syy[indv]),sxx[indv]),szz[indv]))),mul2(lszz,sub2(sub2(mul2(f2h2(2.0),szz[indv]),sxx[indv]),syy[indv]))))));
    
    #if HOUT==1
    float2 dMH=mul2f(mul2f(c1,__h22f2c(add2(add2(sxx[indv],syy[indv]),szz[indv]))),__h22f2c(add2(add2(sxx[indv],syy[indv]),szz[indv])));
    HM[indp]= add2f(HM[indp],dMH);
    Hmu[indp]= add2f(sub2f(add2f(Hmu[indp],mul2f(c3,__h22f2c(add2(add2(mul2(sxz[indv],sxz[indv]),mul2(sxy[indv],sxy[indv])),mul2(syz[indv],syz[indv]))))),mul2f(div2f(f2h2(4.0),f2h2(3.0)),dM)),mul2f(c5,__h22f2c(add2(add2(mul2(sub2(sub2(mul2(f2h2(2.0),sxx[indv]),syy[indv]),szz[indv]),sub2(sub2(mul2(f2h2(2.0),sxx[indv]),syy[indv]),szz[indv])),mul2(sub2(sub2(mul2(f2h2(2.0),syy[indv]),sxx[indv]),szz[indv]),sub2(sub2(mul2(f2h2(2.0),syy[indv]),sxx[indv]),szz[indv]))),mul2(sub2(sub2(mul2(f2h2(2.0),szz[indv]),sxx[indv]),syy[indv]),sub2(sub2(mul2(f2h2(2.0),szz[indv]),sxx[indv]),syy[indv]))))));
    #endif
    #endif
    
    #if RESTYPE==1
    float2 dM=__h22f2c(mul2(add2(add2(sxx[indv],syy[indv]),szz[indv]),add2(add2(lsxx,lsyy),lszz)));
    
    gradM[indp]=sub2f(gradM[indp],dM);
    
    #if HOUT==1
    float2 dMH= __h22f2c(mul2(add2(add2(sxx[indv],syy[indv]),szz[indv]),add2(add2(sxx[indv],syy[indv]),szz[indv])));
    HM[indp]= add2f(HM[indp],dMH);
    
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
    //                    pressure=(sxxr[indv]+syyr[indv]+szzr[indv] )/(2.0*DH*DH*DH);
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

