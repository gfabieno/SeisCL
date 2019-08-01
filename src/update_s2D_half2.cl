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

/*Update of the stresses in 2D SV. The variable FP16 is
 used to control how FP16 is used: 1: FP32, 2: FP16 IO only, 3: FP16 IO and
 arithmetics*/


FUNDEF void update_s(int offcomm,
                                    __pprec *muipkp, __pprec *M, __pprec *mu,
                                    __prec2 *sxx,__prec2 *sxz,__prec2 *szz,
                                    __prec2 *vx,__prec2 *vz, float *taper
                                    )

{
    //Local memory
    extern __shared__ __prec2 lvar2[];
    __prec * lvar=(__prec *)lvar2;
    
    //Grid position
    int lsizez = blockDim.x+FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH/2;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/2;
    int gidx = blockIdx.y*blockDim.y+threadIdx.y+FDOH+offcomm;
    
    int indp = ((gidx)-FDOH)*(NZ-FDOH)+((gidz)-FDOH/2);
    int indv = gidx*NZ+gidz;
    
    //Define private derivatives
    __cprec vx_x2;
    __cprec vx_z1;
    __cprec vz_x1;
    __cprec vz_z2;
   
    //Local memory definitions if local is used
    #if LOCAL_OFF==0
        #define lvx lvar
        #define lvz lvar
        #define lvx2 lvar2
        #define lvz2 lvar2
    
    //Local memory definitions if local is not used
    #elif LOCAL_OFF==1
    
        #define lvx vx
        #define lvz vz
        #define lidz gidz
        #define lidx gidx
    
    #endif
    
    //Calculation of the spatial derivatives
    {
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(vx);
        load_local_haloz(vx);
        load_local_halox(vx);
        BARRIER
    #endif
        vx_x2 = Dxm(lvx2);
        vx_z1 = Dzp(lvx);

    #if LOCAL_OFF==0
        BARRIER
        load_local_in(vz);
        load_local_haloz(vz);
        load_local_halox(vz);
        BARRIER
    #endif
        vz_x1 = Dxp(lvz2);
        vz_z2 = Dzm(lvz);
        
    }
    
    // To stop updating if we are outside the model (global id must be a
    //multiple of local id in OpenCL, hence we stop if we have a global id
    //outside the grid)
    #if  LOCAL_OFF==0
    #if COMM12==0
    if ( gidz>(NZ-FDOH/2-1) ||  (gidx-offcomm)>(NX-FDOH-1-LCOMM) )
        return;
    #else
    if ( gidz>(NZ-FDOH/2-1)  )
        return;
    #endif
    #endif
   
    //Define and load private parameters and variables
    __cprec lsxx = __h22f2(sxx[indv]);
    __cprec lsxz = __h22f2(sxz[indv]);
    __cprec lszz = __h22f2(szz[indv]);
    __cprec lM = __pconv(M[indp]);
    __cprec lmu = __pconv(mu[indp]);
    __cprec lmuipkp = __pconv(muipkp[indp]);
    
    // Update the variables
    lsxz=add2(lsxz,mul2(lmuipkp,add2(vx_z1,vz_x1)));
    lsxx=sub2(add2(lsxx,mul2(lM,add2(vx_x2,vz_z2))),mul2(mul2(f2h2(2.0),lmu),vz_z2));
    lszz=sub2(add2(lszz,mul2(lM,add2(vx_x2,vz_z2))),mul2(mul2(f2h2(2.0),lmu),vx_x2));
    
    #if ABS_TYPE==2
    {
    #if FREESURF==0
        if (2*gidz-FDOH<NAB){
            lsxx.x*=taper[2*gidz-FDOH];
            lsxx.y*=taper[2*gidz+1-FDOH];
            lszz.x*=taper[2*gidz-FDOH];
            lszz.y*=taper[2*gidz+1-FDOH];
            lsxz.x*=taper[2*gidz-FDOH];
            lsxz.y*=taper[2*gidz+1-FDOH];
        }
    #endif
        if (2*gidz>2*NZ-NAB-FDOH-1){
            lsxx.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsxx.y*=taper[2*NZ-FDOH-2*gidz-1-1];
            lszz.x*=taper[2*NZ-FDOH-2*gidz-1];
            lszz.y*=taper[2*NZ-FDOH-2*gidz-1-1];
            lsxz.x*=taper[2*NZ-FDOH-2*gidz-1];
            lsxz.y*=taper[2*NZ-FDOH-2*gidz-1-1];
        }
        
    #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            lsxx.x*=taper[gidx-FDOH];
            lsxx.y*=taper[gidx-FDOH];
            lszz.x*=taper[gidx-FDOH];
            lszz.y*=taper[gidx-FDOH];
            lsxz.x*=taper[gidx-FDOH];
            lsxz.y*=taper[gidx-FDOH];
        }
    #endif
        
    #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            lsxx.x*=taper[NX-FDOH-gidx-1];
            lsxx.y*=taper[NX-FDOH-gidx-1];
            lszz.x*=taper[NX-FDOH-gidx-1];
            lszz.y*=taper[NX-FDOH-gidx-1];
            lsxz.x*=taper[NX-FDOH-gidx-1];
            lsxz.y*=taper[NX-FDOH-gidx-1];
        }
    #endif
    }
    #endif

    //Write updated values to global memory
    sxx[indv] = __f22h2(lsxx);
    sxz[indv] = __f22h2(lsxz);
    szz[indv] = __f22h2(lszz);

}

