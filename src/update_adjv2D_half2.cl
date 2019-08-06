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

/*Adjoint update of the velocities in 2D SV*/



FUNDEF void update_adjv(int offcomm,
                           __pprec *rip, __pprec *rkp,
                           __prec2 *sxx,__prec2 *sxz,__prec2 *szz,
                           __prec2 *vx,__prec2 *vz,
                           __prec2 *sxxbnd,__prec2 *sxzbnd,__prec2 *szzbnd,
                           __prec2 *vxbnd,__prec2 *vzbnd,
                           __prec2 *sxxr,__prec2 *sxzr,__prec2 *szzr,
                           __prec2 *vxr,__prec2 *vzr, float *taper,
                           float2 *gradrho, float2 *Hrho, int res_scale,
                           int src_scale, int par_scale)
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
    
    //Define private derivatives
    __cprec sxx_x1;
    __cprec sxz_x2;
    __cprec sxz_z2;
    __cprec szz_z1;
    __cprec sxxr_x1;
    __cprec sxzr_x2;
    __cprec sxzr_z2;
    __cprec szzr_z1;
    
// If we use local memory
#if LOCAL_OFF==0

#define lsxx lvar
#define lszz lvar
#define lsxz lvar
#define lsxx2 lvar2
#define lszz2 lvar2
#define lsxz2 lvar2
    
#define lsxxr lvar
#define lszzr lvar
#define lsxzr lvar
#define lsxxr2 lvar2
#define lszzr2 lvar2
#define lsxzr2 lvar2
    
#endif
    
    int indp = (gidx-FDOH)*(NZ-2*FDOH)+(gidz-FDOH);
    int indv = gidx*NZ+gidz;
    
// Calculation of the stress spatial derivatives of the forward wavefield if backpropagation is used
#if BACK_PROP_TYPE==1
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
        BARRIER
}
#endif

    #if LOCAL_OFF==0
        load_local_in(sxxr);
        load_local_halox(sxxr);
        BARRIER
    #endif
        sxxr_x1 = Dxp(lsxxr2);
    
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(szzr);
        load_local_haloz(szzr);
        BARRIER
    #endif
        szzr_z1 = Dzp(lszzr);

    #if LOCAL_OFF==0
        BARRIER
        load_local_in(sxzr);
        load_local_halox(sxzr);
        load_local_haloz(sxzr);
        BARRIER
    #endif
        sxzr_x2 = Dxm(lsxzr2);
        sxzr_z2 = Dzm(lsxzr);


    
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
    __cprec lvxr = __h22f2(vxr[indv]);
    __cprec lvzr = __h22f2(vzr[indv]);
    __cprec lrip = __pconv(rip[indp]);
    __cprec lrkp = __pconv(rkp[indp]);

// Backpropagate the forward velocity
#if BACK_PROP_TYPE==1
    __cprec lvx = __h22f2(vx[indv]);
    __cprec lvz = __h22f2(vz[indv]);
    {
        // Update the variables
        lvx=sub2(lvx,mul2(add2(sxx_x1,sxz_z2),lrip));
        lvz=sub2(lvz,mul2(add2(szz_z1,sxz_x2),lrkp));
        
        // Inject the boundary values
        int m=inject_ind(gidz, gidx);
        if (m!=-1){
            lvx= __h22f2(vxbnd[m]);
            lvz= __h22f2(vzbnd[m]);
        }

        
        //Write updated values to global memory
        vx[indv] = __f22h2(lvx);
        vz[indv] = __f22h2(lvz);
    }
#endif

    // Update the variables
    lvxr=add2(lvxr,mul2(add2(sxxr_x1,sxzr_z2),lrip));
    lvzr=add2(lvzr,mul2(add2(szzr_z1,sxzr_x2),lrkp));
    
#if ABS_TYPE==2
    {
        if (2*gidz-FDOH<NAB){
            lvxr.x*=taper[2*gidz-FDOH];
            lvxr.y*=taper[2*gidz+1-FDOH];
            lvzr.x*=taper[2*gidz-FDOH];
            lvzr.y*=taper[2*gidz+1-FDOH];
        }
        
        if (2*gidz>2*NZ-NAB-FDOH-1){
            lvxr.x*=taper[2*NZ-FDOH-2*gidz-1];
            lvxr.y*=taper[2*NZ-FDOH-2*gidz-1-1];
            lvzr.x*=taper[2*NZ-FDOH-2*gidz-1];
            lvzr.y*=taper[2*NZ-FDOH-2*gidz-1-1];
        }
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            lvxr.x*=taper[gidx-FDOH];
            lvxr.y*=taper[gidx-FDOH];
            lvzr.x*=taper[gidx-FDOH];
            lvzr.y*=taper[gidx-FDOH];
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            lvxr.x*=taper[NX-FDOH-gidx-1];
            lvxr.y*=taper[NX-FDOH-gidx-1];
            lvzr.x*=taper[NX-FDOH-gidx-1];
            lvzr.y*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif
    
    //Write updated values to global memory
    vxr[indv] = __f22h2(lvxr);
    vzr[indv] = __f22h2(lvzr);
    
    
// Density gradient calculation on the fly
#if BACK_PROP_TYPE==1
    lvxr=mul2(add2(sxxr_x1,sxzr_z2),lrip);
    lvzr=mul2(add2(szzr_z1,sxzr_x2),lrkp);

    gradrho[indp]=sub2f( gradrho[indp], scalbnf2(add2f( mul2f( __h22f2c(lvx), __h22f2c(lvxr)), mul2f( __h22f2c(lvz), __h22f2c(lvzr)) ), 2*par_scale -src_scale - res_scale) );
    #if HOUT==1
        Hrho[indp]= sub2f( Hrho[indp], scalbnf2(add2f( mul2f( __h22f2c(lvx), __h22f2c(lvx)), mul2f( __h22f2c(lvz), __h22f2c(lvz)) ), 2*par_scale-2*src_scale) );
    #endif
    
#endif

}

