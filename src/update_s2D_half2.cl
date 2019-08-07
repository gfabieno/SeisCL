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
                     GLOBARG __pprec *muipkp, GLOBARG __pprec *M,
                     GLOBARG __pprec *mu,
                     GLOBARG __prec2 *sxx, GLOBARG __prec2 *sxz,
                     GLOBARG __prec2 *szz, GLOBARG __prec2 *vx,
                     GLOBARG __prec2 *vz,
                     GLOBARG __prec2 *rxx, GLOBARG __prec2 *rzz,
                     GLOBARG __prec2 *rxz, GLOBARG __pprec *taus,
                     GLOBARG __pprec *tausipkp,   GLOBARG __pprec *taup,
                     GLOBARG float *eta, GLOBARG float *taper, LOCARG2)

{
    //Local memory
    LOCDEF2
    #ifdef __OPENCL_VERSION__
    __local __prec * lvar=lvar2;
    #else
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
    
    // Read model parameters into local memory
    #if LVE==0
    {
        __cprec fipkp=__pconv(muipkp[indp]);
        __cprec f=2.0f*__pconv(mu[indp]);
        __cprec g=__pconv(M[indp]);
    
        // Update the variables
        lsxz=lsxz + fipkp * (vx_z1+vz_x1);
        lsxx=lsxx + g*(vx_x2+vz_z2) - f * vz_z2;
        lszz=lszz + g*(vx_x2+vz_z2) - f * vx_x2;
    }
    #else
    {
        int indr, l;
        
        __cprec lM=__pconv(M[indp]);
        __cprec lmu=__pconv(mu[indp]);
        __cprec lmuipkp=__pconv(muipkp[indp]);
        __cprec ltaup=__pconv(taup[indp]);
        __cprec ltaus=__pconv(taus[indp]);
        __cprec ltausipkp=__pconv(tausipkp[indp]);
    
        float leta[LVE];
        for (l=0;l<LVE;l++){
            leta[l]=eta[l];
        }
        
        __cprec fipkp=lmuipkp*(1.0f+ (float)LVE*ltausipkp);
        __cprec g=lM*(1.0f+(float)LVE*ltaup);
        __cprec f=2.0f*lmu*(1.0f+(float)LVE*ltaus);
        __cprec dipkp=lmuipkp*ltausipkp/DT;
        __cprec d=2.0f*lmu*ltaus/DT;
        __cprec e=lM*ltaup/DT;
        
        /* computing sums of the old memory variables */
        __cprec sumrxz=__cprec0;
        __cprec sumrxx=__cprec0;
        __cprec sumrzz=__cprec0;
        for (l=0;l<LVE;l++){
            indr = l*NX*NZ + gidx*NZ+gidz;
            sumrxz=sumrxz + rxz[indr];
            sumrxx=sumrxx + rxx[indr];
            sumrzz=sumrzz + rzz[indr];
        }

        /* updating components of the stress tensor, partially */
        lsxz=lsxz +(fipkp*(vx_z1+vz_x1))+(DT2*sumrxz);
        lsxx=lsxx +((g*(vx_x2+vz_z2))-(f*vz_z2))+(DT2*sumrxx);
        lszz=lszz +((g*(vx_x2+vz_z2))-(f*vx_x2))+(DT2*sumrzz);

        /* now updating the memory-variables and sum them up*/
        sumrxz=sumrxx=sumrzz=sumrxz * 0.0f;
        float b,c;
        for (l=0;l<LVE;l++){
            b=1.0f/(1.0f+(leta[l]*0.5f));
            c=1.0f-(leta[l]*0.5f);
            indr = l*NX*NZ + gidx*NZ+gidz;
            rxz[indr]=__f22h2(b*(rxz[indr]*c-leta[l]*(dipkp*(vx_z1+vz_x1))));
            rxx[indr]=__f22h2(b*(rxx[indr]*c-leta[l]*((e*(vx_x2+vz_z2))-(d*vz_z2))));
            rzz[indr]=__f22h2(b*(rzz[indr]*c-leta[l]*((e*(vx_x2+vz_z2))-(d*vx_x2))));

            sumrxz=sumrxz + rxz[indr];
            sumrxx=sumrxx + rxx[indr];
            sumrzz=sumrzz + rzz[indr];
        }

        /* and now the components of the stress tensor are
         completely updated */
        lsxz= lsxz + (DT2*sumrxz);
        lsxx= lsxx + (DT2*sumrxx) ;
        lszz= lszz + (DT2*sumrzz) ;
    }
    #endif
  
    // Absorbing boundary
    #if ABS_TYPE==2
    {
    #if FREESURF==0
        if (DIV*gidz-FDOH<NAB){
            lsxx = lsxx * __hpg(&taper[DIV*gidz-FDOH]);
            lszz = lszz * __hpg(&taper[DIV*gidz-FDOH]);
            lsxz = lsxz * __hpg(&taper[DIV*gidz-FDOH]);
        }
    #endif
        if (DIV*gidz>DIV*NZ-NAB-FDOH-1){
            lsxx =lsxx * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lszz =lszz * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lsxz =lsxz * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
        }
        
    #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            lsxx = lsxx * taper[gidx-FDOH];
            lszz = lszz * taper[gidx-FDOH];
            lsxz = lsxz * taper[gidx-FDOH];
        }
    #endif
        
    #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            lsxx = lsxx * taper[NX-FDOH-gidx-1];
            lszz = lszz * taper[NX-FDOH-gidx-1];
            lsxz = lsxz * taper[NX-FDOH-gidx-1];
        }
    #endif
    }
    #endif

    //Write updated values to global memory
    sxx[indv] = __f22h2(lsxx);
    sxz[indv] = __f22h2(lsxz);
    szz[indv] = __f22h2(lszz);

}

