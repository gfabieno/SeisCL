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
                                    __pprec *muipjp, __pprec *mujpkp, __pprec *muipkp,
                                    __pprec *M, __pprec *mu,__prec2 *sxx,__prec2 *sxy,__prec2 *sxz,
                                    __prec2 *syy,__prec2 *syz,__prec2 *szz,
                                    __prec2 *vx,__prec2 *vy,__prec2 *vz, float *taper
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
    __cprec lsxx = __h22f2(sxx[indv]);
    __cprec lsxy = __h22f2(sxy[indv]);
    __cprec lsxz = __h22f2(sxz[indv]);
    __cprec lsyy = __h22f2(syy[indv]);
    __cprec lsyz = __h22f2(syz[indv]);
    __cprec lszz = __h22f2(szz[indv]);
    __cprec lM = __pconv(M[indp]);
    __cprec lmu = __pconv(mu[indp]);
    __cprec lmuipjp = __pconv(muipjp[indp]);
    __cprec lmuipkp = __pconv(muipkp[indp]);
    __cprec lmujpkp = __pconv(mujpkp[indp]);
    
    // Update the variables
    lsxy=add2(lsxy,mul2(lmuipjp,add2(vx_y1,vy_x1)));
    lsyz=add2(lsyz,mul2(lmujpkp,add2(vy_z1,vz_y1)));
    lsxz=add2(lsxz,mul2(lmuipkp,add2(vx_z1,vz_x1)));
    lsxx=sub2(add2(lsxx,mul2(lM,add2(add2(vx_x2,vy_y2),vz_z2))),mul2(mul2(f2h2(2.0),lmu),add2(vy_y2,vz_z2)));
    lsyy=sub2(add2(lsyy,mul2(lM,add2(add2(vx_x2,vy_y2),vz_z2))),mul2(mul2(f2h2(2.0),lmu),add2(vx_x2,vz_z2)));
    lszz=sub2(add2(lszz,mul2(lM,add2(add2(vx_x2,vy_y2),vz_z2))),mul2(mul2(f2h2(2.0),lmu),add2(vx_x2,vy_y2)));
    
    // Absorbing boundary
#if ABS_TYPE==2
    {
#if FREESURF==0
        if (2*gidz-FDOH<NAB){
            lsxy.x*=taper[2*gidz-FDOH];
            lsxy.y*=taper[2*gidz+1-FDOH];
            lsyz.x*=taper[2*gidz-FDOH];
            lsyz.y*=taper[2*gidz+1-FDOH];
            lsxz.x*=taper[2*gidz-FDOH];
            lsxz.y*=taper[2*gidz+1-FDOH];
            lsxx.x*=taper[2*gidz-FDOH];
            lsxx.y*=taper[2*gidz+1-FDOH];
            lsyy.x*=taper[2*gidz-FDOH];
            lsyy.y*=taper[2*gidz+1-FDOH];
            lszz.x*=taper[2*gidz-FDOH];
            lszz.y*=taper[2*gidz+1-FDOH];
        }
#endif

        if (2*gidz>2*NZ-NAB-FDOH-1){
            lsxy.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsxy.y*=taper[2*NZ-FDOH-2*gidz-2];
            lsyz.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsyz.y*=taper[2*NZ-FDOH-2*gidz-2];
            lsxz.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsxz.y*=taper[2*NZ-FDOH-2*gidz-2];
            lsxx.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsxx.y*=taper[2*NZ-FDOH-2*gidz-2];
            lsyy.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsyy.y*=taper[2*NZ-FDOH-2*gidz-2];
            lszz.x*=taper[2*NZ-FDOH-2*gidz-1];
            lszz.y*=taper[2*NZ-FDOH-2*gidz-2];
        }
        if (gidy-FDOH<NAB){
            lsxy.x*=taper[gidy-FDOH];
            lsxy.y*=taper[gidy-FDOH];
            lsyz.x*=taper[gidy-FDOH];
            lsyz.y*=taper[gidy-FDOH];
            lsxz.x*=taper[gidy-FDOH];
            lsxz.y*=taper[gidy-FDOH];
            lsxx.x*=taper[gidy-FDOH];
            lsxx.y*=taper[gidy-FDOH];
            lsyy.x*=taper[gidy-FDOH];
            lsyy.y*=taper[gidy-FDOH];
            lszz.x*=taper[gidy-FDOH];
            lszz.y*=taper[gidy-FDOH];
        }

        if (gidy>NY-NAB-FDOH-1){
            lsxy.x*=taper[NY-FDOH-gidy-1];
            lsxy.y*=taper[NY-FDOH-gidy-1];
            lsyz.x*=taper[NY-FDOH-gidy-1];
            lsyz.y*=taper[NY-FDOH-gidy-1];
            lsxz.x*=taper[NY-FDOH-gidy-1];
            lsxz.y*=taper[NY-FDOH-gidy-1];
            lsxx.x*=taper[NY-FDOH-gidy-1];
            lsxx.y*=taper[NY-FDOH-gidy-1];
            lsyy.x*=taper[NY-FDOH-gidy-1];
            lsyy.y*=taper[NY-FDOH-gidy-1];
            lszz.x*=taper[NY-FDOH-gidy-1];
        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            lsxy.x*=taper[gidx-FDOH];
            lsxy.y*=taper[gidx-FDOH];
            lsyz.x*=taper[gidx-FDOH];
            lsyz.y*=taper[gidx-FDOH];
            lsxz.x*=taper[gidx-FDOH];
            lsxz.y*=taper[gidx-FDOH];
            lsxx.x*=taper[gidx-FDOH];
            lsxx.y*=taper[gidx-FDOH];
            lsyy.x*=taper[gidx-FDOH];
            lsyy.y*=taper[gidx-FDOH];
            lszz.x*=taper[gidx-FDOH];
            lszz.y*=taper[gidx-FDOH];
        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            lsxy.x*=taper[NX-FDOH-gidx-1];
            lsxy.y*=taper[NX-FDOH-gidx-1];
            lsyz.x*=taper[NX-FDOH-gidx-1];
            lsyz.y*=taper[NX-FDOH-gidx-1];
            lsxz.x*=taper[NX-FDOH-gidx-1];
            lsxz.y*=taper[NX-FDOH-gidx-1];
            lsxx.x*=taper[NX-FDOH-gidx-1];
            lsxx.y*=taper[NX-FDOH-gidx-1];
            lsyy.x*=taper[NX-FDOH-gidx-1];
            lsyy.y*=taper[NX-FDOH-gidx-1];
            lszz.x*=taper[NX-FDOH-gidx-1];
            lszz.y*=taper[NX-FDOH-gidx-1];
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
