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

/*Adjoint update of the stresses in 2D SV*/

FUNDEF __global__ void update_adjs(int offcomm,
                           __pprec *muipkp, __pprec *M, __pprec *mu,
                           __prec2 *sxx,__prec2 *sxz,__prec2 *szz,
                           __prec2 *vx,__prec2 *vz,
                           __prec2 *sxxbnd,__prec2 *sxzbnd,__prec2 *szzbnd,
                           __prec2 *vxbnd,__prec2 *vzbnd,
                           __prec2 *sxxr,__prec2 *sxzr,__prec2 *szzr,
                           __prec2 *vxr,__prec2 *vzr, float *taper,
                          float2 *gradrho,    float2 *gradM,     float2 *gradmu,
                           float2 *Hrho,    float2 *HM,     float2 *Hmu,
                           int res_scale, int src_scale, int par_scale)
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
    __cprec vxr_x2;
    __cprec vxr_z1;
    __cprec vzr_x1;
    __cprec vzr_z2;

    
    
// If we use local memory
#if LOCAL_OFF==0
#define lvx lvar
#define lvz lvar
#define lvx2 lvar2
#define lvz2 lvar2
    
#define lvxr lvar
#define lvzr lvar
#define lvxr2 lvar2
#define lvzr2 lvar2

//// If local memory is turned off
//#elif LOCAL_OFF==1
//
//#define lvx_r vx_r
//#define lvz_r vz_r
//#define lvx vx
//#define lvz vz
//#define lidx gidx
//#define lidz gidz
//
//#define lsizez NZ
//#define lsizex NX
    
#endif
    
// Calculation of the velocity spatial derivatives of the forward wavefield if backpropagation is used
#if BACK_PROP_TYPE==1
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
        BARRIER
    }
#endif
    
// Calculation of the velocity spatial derivatives of the adjoint wavefield
    {
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(vxr);
        load_local_haloz(vxr);
        load_local_halox(vxr);
        BARRIER
    #endif
        vxr_x2 = Dxm(lvxr2);
        vxr_z1 = Dzp(lvxr);

    #if LOCAL_OFF==0
        BARRIER
        load_local_in(vzr);
        load_local_haloz(vzr);
        load_local_halox(vzr);
        BARRIER
    #endif
        vzr_x1 = Dxp(lvzr2);
        vzr_z2 = Dzm(lvzr);
    }
    
// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
    // To stop updating if we are outside the model (global id must be amultiple of local id in OpenCL, hence we stop if we have a global idoutside the grid)
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
    __cprec lsxxr = __h22f2(sxxr[indv]);
    __cprec lsxzr = __h22f2(sxzr[indv]);
    __cprec lszzr = __h22f2(szzr[indv]);
    __cprec lM = __pconv(M[indp]);
    __cprec lmu = __pconv(mu[indp]);
    __cprec lmuipkp = __pconv(muipkp[indp]);
    
    // Backpropagate the forward stresses
    #if BACK_PROP_TYPE==1
    {
        // Update the variables
        lsxz=sub2(lsxz,mul2(lmuipkp,add2(vx_z1,vz_x1)));
        lsxx=add2(sub2(lsxx,mul2(lM,add2(vx_x2,vz_z2))),mul2(mul2(f2h2(2.0),lmu),vz_z2));
        lszz=add2(sub2(lszz,mul2(lM,add2(vx_x2,vz_z2))),mul2(mul2(f2h2(2.0),lmu),vx_x2));

        int m=inject_ind(gidz, gidx);
        if (m!=-1){
            lsxx= __h22f2(sxxbnd[m]);
            lszz= __h22f2(szzbnd[m]);
            lsxz= __h22f2(sxzbnd[m]);
        }
        
        //Write updated values to global memory
        sxx[indv] = __f22h2(lsxx);
        sxz[indv] = __f22h2(lsxz);
        szz[indv] = __f22h2(lszz);
        
    }
    #endif


    
    
// Update adjoint stresses
    {
        // Update the variables
        lsxzr=add2(lsxzr,mul2(lmuipkp,add2(vxr_z1,vzr_x1)));
        lsxxr=sub2(add2(lsxxr,mul2(lM,add2(vxr_x2,vzr_z2))),mul2(mul2(f2h2(2.0),lmu),vzr_z2));
        lszzr=sub2(add2(lszzr,mul2(lM,add2(vxr_x2,vzr_z2))),mul2(mul2(f2h2(2.0),lmu),vxr_x2));
        
#if ABS_TYPE==2
        {
            if (2*gidz-FDOH<NAB){
                lsxxr.x*=taper[2*gidz-FDOH];
                lsxxr.y*=taper[2*gidz+1-FDOH];
                lszzr.x*=taper[2*gidz-FDOH];
                lszzr.y*=taper[2*gidz+1-FDOH];
                lsxzr.x*=taper[2*gidz-FDOH];
                lsxzr.y*=taper[2*gidz+1-FDOH];
            }
            
            if (2*gidz>2*NZ-NAB-FDOH-1){
                lsxxr.x*=taper[2*NZ-FDOH-2*gidz-1];
                lsxxr.y*=taper[2*NZ-FDOH-2*gidz-1-1];
                lszzr.x*=taper[2*NZ-FDOH-2*gidz-1];
                lszzr.y*=taper[2*NZ-FDOH-2*gidz-1-1];
                lsxzr.x*=taper[2*NZ-FDOH-2*gidz-1];
                lsxzr.y*=taper[2*NZ-FDOH-2*gidz-1-1];
            }
            
#if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB){
                lsxxr.x*=taper[gidx-FDOH];
                lsxxr.y*=taper[gidx-FDOH];
                lszzr.x*=taper[gidx-FDOH];
                lszzr.y*=taper[gidx-FDOH];
                lsxzr.x*=taper[gidx-FDOH];
                lsxzr.y*=taper[gidx-FDOH];
            }
#endif
            
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1){
                lsxxr.x*=taper[NX-FDOH-gidx-1];
                lsxxr.y*=taper[NX-FDOH-gidx-1];
                lszzr.x*=taper[NX-FDOH-gidx-1];
                lszzr.y*=taper[NX-FDOH-gidx-1];
                lsxzr.x*=taper[NX-FDOH-gidx-1];
                lsxzr.y*=taper[NX-FDOH-gidx-1];
            }
#endif
        }
#endif
        
        
        //Write updated values to global memory
        sxxr[indv] = __f22h2(lsxxr);
        sxzr[indv] = __f22h2(lsxzr);
        szzr[indv] = __f22h2(lszzr);
    }

    // Shear wave modulus and P-wave modulus gradient calculation on the fly
#if BACK_PROP_TYPE==1
    #if RESTYPE==0
    //TODO review scaling
    float2 c1= div2f(f2h2f(1.0), mul2f(mul2f(f2h2f(2.0), sub2f(__h22f2c(lM),__h22f2c(lmu))),mul2f(f2h2f(2.0), sub2f(__h22f2c(lM),__h22f2c(lmu)))));
    float2 c3=div2f(f2h2f(1.0), mul2f(__h22f2c(lmu),__h22f2c(lmu)));
    float2 c5=mul2f(f2h2f(0.25), c3);
    
    
    lsxzr=mul2(lmuipkp,add2(vxr_z1,vzr_x1));
    lsxxr=sub2(mul2(lM,add2(vxr_x2,vzr_z2)),mul2(mul2(f2h2(2.0),lmu),vzr_z2));
    lszzr=sub2(mul2(lM,add2(vxr_x2,vzr_z2)),mul2(mul2(f2h2(2.0),lmu),vxr_x2));

    float2 dM=mul2f(c1,mul2f(__h22f2c(add2(lsxx,lszz)), __h22f2c(add2(lsxxr,lszzr)) ) );

    gradM[indp]=sub2f(gradM[indp], scalbnf2(dM, 2*par_scale-src_scale - res_scale));
    gradmu[indp]=add2f(gradmu[indp],
                           scalbnf2(sub2f(sub2f( dM, mul2f(c3, mul2f(__h22f2c(lsxz),__h22f2c(lsxzr)))), mul2f(c5,mul2f( sub2f(__h22f2c(lsxx),__h22f2c(lszz)), sub2f(__h22f2c(lsxxr),__h22f2c(lszzr))))),
                                2*par_scale-src_scale-res_scale));
    
        #if HOUT==1
    float2 dMH=mul2f(c1,mul2f(__h22f2c(add2(lsxx,lszz)), __h22f2c(add2(lsxx,lszz) )) );
    HM[indp]=add2f(HM[indp], scalbnf2(dMH, -2.0*src_scale));
    Hmu[indp]=add2f(Hmu[indp],
                        scalbnf2(add2f(sub2f( mul2f(c3, mul2f(__h22f2c(lsxz),__h22f2c(lsxz))), dMH ), mul2f(c5, mul2f( __h22f2c(sub2(lsxx,lszz)), __h22f2c(sub2(lsxx,lszz))))),
                                 2*par_scale-2.0*src_scale));
        #endif
    #endif
    
    #if RESTYPE==1
    float2 dM=mul2f(__h22f2c(add2(lsxx,lszz)), __h22f2c(add2(lsxxr,lszzr)) );
    
    gradM[indp]=sub2f(gradM[indp], scalbnf2(dM, -src_scale - res_scale));
    
    #if HOUT==1
    float2 dMH=mul2f(__h22f2c(add2(lsxx,lszz)), __h22f2c(add2(lsxx,lszz)) );
    HM[indp]=add2f(HM[indp], scalbnf2(dMH, -2.0*src_scale));

    #endif
    #endif
    
#endif


}

