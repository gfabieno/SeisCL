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


FUNDEF void update_adjs(int offcomm,
                           GLOBARG __pprec *muipkp, GLOBARG __pprec *M, GLOBARG __pprec *mu,
                           GLOBARG __prec2 *sxx,GLOBARG __prec2 *sxz,GLOBARG __prec2 *szz,
                           GLOBARG __prec2 *vx,GLOBARG __prec2 *vz,
                           GLOBARG __prec2 *sxxbnd,GLOBARG __prec2 *sxzbnd,GLOBARG __prec2 *szzbnd,
                           GLOBARG __prec2 *vxbnd,GLOBARG __prec2 *vzbnd,
                           GLOBARG __prec2 *sxxr,GLOBARG __prec2 *sxzr,GLOBARG __prec2 *szzr,
                           GLOBARG __prec2 *vxr,GLOBARG __prec2 *vzr, GLOBARG float *taper,
                          GLOBARG float2 *gradrho,    GLOBARG float2 *gradM,     GLOBARG float2 *gradmu,
                           GLOBARG float2 *Hrho,    GLOBARG float2 *HM,    GLOBARG  float2 *Hmu,
                           int res_scale, int src_scale, int par_scale)
{

    //Local memory
    extern __shared__ __prec2 lvar2[];
    __prec * lvar=(__prec *)lvar2;
    
    //Grid position
    int lsizez = blockDim.x+2*FDOH/DIV;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH/DIV;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/DIV;
    int gidx = blockIdx.y*blockDim.y+threadIdx.y+FDOH+offcomm;
    
    int indp = ((gidx)-FDOH)*(NZ-FDOH)+((gidz)-FDOH/DIV);
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
        // Update the variables
        lsxz=lsxz - lmuipkp * (vx_z1+vz_x1);
        lsxx=lsxx - lM*(vx_x2+vz_z2) + 2.0 * lmu * vz_z2;
        lszz=lszz - lM*(vx_x2+vz_z2) + 2.0 * lmu * vx_x2;

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
        lsxzr=lsxzr + lmuipkp * (vxr_z1+vzr_x1);
        lsxxr=lsxxr + lM*(vxr_x2+vzr_z2) - 2.0 * lmu * vzr_z2;
        lszzr=lszzr + lM*(vxr_x2+vzr_z2) - 2.0 * lmu * vxr_x2;

        
        #if ABS_TYPE==2
        {
        #if FREESURF==0
            if (DIV*gidz-FDOH<NAB){
                lsxxr = lsxxr * __hp(&taper[DIV*gidz-FDOH]);
                lszzr = lszzr * __hp(&taper[DIV*gidz-FDOH]);
                lsxzr = lsxzr * __hp(&taper[DIV*gidz-FDOH]);
            }
        #endif
            if (DIV*gidz>DIV*NZ-NAB-FDOH-1){
                lsxxr =lsxxr * __hpi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
                lszzr =lszzr * __hpi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
                lsxzr =lsxzr * __hpi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            }
            
        #if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB){
                lsxxr = lsxxr * taper[gidx-FDOH];
                lszzr = lszzr * taper[gidx-FDOH];
                lsxzr = lsxzr * taper[gidx-FDOH];
            }
        #endif
            
        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1){
                lsxxr = lsxxr * taper[NX-FDOH-gidx-1];
                lszzr = lszzr * taper[NX-FDOH-gidx-1];
                lsxzr = lsxzr * taper[NX-FDOH-gidx-1];
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

