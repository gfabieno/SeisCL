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

/*Adjoint update of the velocities in 3D SV*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */


#define psi_sxx_x(z,y,x) psi_sxx_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_sxy_x(z,y,x) psi_sxy_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_sxz_x(z,y,x) psi_sxz_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_sxy_y(z,y,x) psi_sxy_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_syy_y(z,y,x) psi_syy_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_syz_y(z,y,x) psi_syz_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_sxz_z(z,y,x) psi_sxz_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_syz_z(z,y,x) psi_syz_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_szz_z(z,y,x) psi_szz_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]




FUNDEF void update_adjv(int offcomm,
                          GLOBARG float *vx,         GLOBARG float *vy,      GLOBARG float *vz,
                          GLOBARG float *sxx,        GLOBARG float *syy,     GLOBARG float *szz,
                          GLOBARG float *sxy,        GLOBARG float *syz,     GLOBARG float *sxz,
                          GLOBARG float *vxbnd,      GLOBARG float *vybnd,   GLOBARG float *vzbnd,
                          GLOBARG float *sxxbnd,     GLOBARG float *syybnd,  GLOBARG float *szzbnd,
                          GLOBARG float *sxybnd,     GLOBARG float *syzbnd,  GLOBARG float *sxzbnd,
                          GLOBARG float *vxr,       GLOBARG float *vyr,    GLOBARG float *vzr,
                          GLOBARG float *sxxr,      GLOBARG float *syyr,   GLOBARG float *szzr,
                          GLOBARG float *sxyr,      GLOBARG float *syzr,   GLOBARG float *sxzr,
                          GLOBARG float *rip,        GLOBARG float *rjp,     GLOBARG float *rkp,
                          GLOBARG float *taper,
                          GLOBARG float *K_x,        GLOBARG float *a_x,          GLOBARG float *b_x,
                          GLOBARG float *K_x_half,   GLOBARG float *a_x_half,     GLOBARG float *b_x_half,
                          GLOBARG float *K_y,        GLOBARG float *a_y,          GLOBARG float *b_y,
                          GLOBARG float *K_y_half,   GLOBARG float *a_y_half,     GLOBARG float *b_y_half,
                          GLOBARG float *K_z,        GLOBARG float *a_z,          GLOBARG float *b_z,
                          GLOBARG float *K_z_half,   GLOBARG float *a_z_half,     GLOBARG float *b_z_half,
                          GLOBARG float *psi_sxx_x,  GLOBARG float *psi_sxy_x,     GLOBARG float *psi_sxy_y,
                          GLOBARG float *psi_sxz_x,  GLOBARG float *psi_sxz_z,     GLOBARG float *psi_syy_y,
                          GLOBARG float *psi_syz_y,  GLOBARG float *psi_syz_z,     GLOBARG float *psi_szz_z,
                          LOCARG, GLOBARG float *gradrho, GLOBARG float *gradsrc,
                          GLOBARG float *Hrho, GLOBARG float *Hsrc)
{
    LOCDEF
    
    int m;
    float sxx_x, syy_y, szz_z, sxy_y, sxy_x, syz_y, syz_z, sxz_x, sxz_z;
    float sxx_xr, syy_yr, szz_zr, sxy_yr, sxy_xr;
    float syz_yr, syz_zr, sxz_xr, sxz_zr;
    float lvx, lvy, lvz;

// If we use local memory
#if LOCAL_OFF==0

    #ifdef __OPENCL_VERSION__
    int lsizez = get_local_size(0)+2*FDOH;
    int lsizey = get_local_size(1)+2*FDOH;
    int lsizex = get_local_size(2)+2*FDOH;
    int lidz = get_local_id(0)+FDOH;
    int lidy = get_local_id(1)+FDOH;
    int lidx = get_local_id(2)+FDOH;
    int gidz = get_global_id(0)+FDOH;
    int gidy = get_global_id(1)+FDOH;
    int gidx = get_global_id(2)+FDOH+offcomm;
    #else
    int lsizez = blockDim.x+2*FDOH;
    int lsizey = blockDim.y+2*FDOH;
    int lsizex = blockDim.z+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidy = threadIdx.y+FDOH;
    int lidx = threadIdx.z+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidy = blockIdx.y*blockDim.y + threadIdx.y+FDOH;
    int gidx = blockIdx.z*blockDim.z + threadIdx.z+FDOH+offcomm;
    #endif

    #define lsxx lvar
    #define lsyy lvar
    #define lszz lvar
    #define lsxy lvar
    #define lsyz lvar
    #define lsxz lvar

    #define lsxxr lvar
    #define lsyyr lvar
    #define lszzr lvar
    #define lsxyr lvar
    #define lsyzr lvar
    #define lsxzr lvar

// If local memory is turned off
#elif LOCAL_OFF==1

#ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
    int glsizez = (NZ-2*FDOH);
    int glsizey = (NY-2*FDOH);
    int gidz = gid%glsizez+FDOH;
    int gidy = (gid/glsizez)%glsizey+FDOH;
    int gidx = gid/(glsizez*glsizey)+FDOH+offcomm;
#else
    int lsizez = blockDim.x+2*FDOH;
    int lsizey = blockDim.y+2*FDOH;
    int lsizex = blockDim.z+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidy = threadIdx.y+FDOH;
    int lidx = threadIdx.z+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidy = blockIdx.y*blockDim.y + threadIdx.y+FDOH;
    int gidx = blockIdx.z*blockDim.z + threadIdx.z+FDOH+offcomm;
#endif

    #define lsxx sxx
    #define lsyy syy
    #define lszz szz
    #define lsxy sxy
    #define lsyz syz
    #define lsxz sxz

    #define lsxxr sxxr
    #define lsyyr syyr
    #define lszzr szzr
    #define lsxyr sxyr
    #define lsyzr syzr
    #define lsxzr sxzr

    #define lidx gidx
    #define lidy gidy
    #define lidz gidz

    #define lsizez NZ
    #define lsizey NY
    #define lsizex NX

#endif
    int indp = (gidx-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+(gidy-FDOH)*(NZ-2*FDOH)+(gidz-FDOH);
    int indv = (gidx)*NZ*NY+(gidy)*NZ+(gidz);

    
// Calculation of the stress spatial derivatives of the forward wavefield if backpropagation is used
#if BACK_PROP_TYPE==1
    {
        #if LOCAL_OFF==0
        load_local_in(szz);
        load_local_haloz(szz);
        BARRIER
        #endif

        szz_z = Dzp(lszz);

        #if LOCAL_OFF==0
        BARRIER
        load_local_in(sxx);
        load_local_halox(sxx);
        BARRIER
        #endif

        sxx_x = Dxp(lsxx);

        #if LOCAL_OFF==0
        BARRIER
        load_local_in(sxz);
        load_local_haloz(sxz);
        load_local_halox(sxz);
        BARRIER
        #endif

        sxz_x = Dxm(lsxz);
        sxz_z = Dzm(lsxz);

        #if LOCAL_OFF==0
        BARRIER
        load_local_in(syz);
        load_local_haloz(syz);
        load_local_haloy(syz);
        BARRIER
        #endif

        syz_y = Dym(lsyz);
        syz_z = Dzm(lsyz);

        #if LOCAL_OFF==0
        BARRIER
        load_local_in(syy);
        load_local_haloy(syy);
        BARRIER
        #endif

        syy_y = Dyp(lsyy);

        #if LOCAL_OFF==0
        BARRIER
        load_local_in(sxy);
        load_local_halox(sxy);
        load_local_haloy(sxy);
        BARRIER
        #endif

        sxy_x = Dxm(lsxy);
        sxy_y = Dym(lsxy);
        BARRIER
    }
#endif

// Calculation of the stress spatial derivatives of the adjoint wavefield
    {
        #if LOCAL_OFF==0
        load_local_in(szzr);
        load_local_haloz(szzr);
        BARRIER
        #endif

        szz_zr = Dzp(lszzr);

        #if LOCAL_OFF==0
        BARRIER
        load_local_in(sxxr);
        load_local_halox(sxxr);
        BARRIER
        #endif

        sxx_xr = Dxp(lsxxr);

        #if LOCAL_OFF==0
        BARRIER
        load_local_in(sxzr);
        load_local_haloz(sxzr);
        load_local_halox(sxzr);
        BARRIER
        #endif

        sxz_xr = Dxm(lsxzr);
        sxz_zr = Dzm(lsxzr);

        #if LOCAL_OFF==0
        BARRIER
        load_local_in(syzr);
        load_local_haloz(syzr);
        load_local_haloy(syzr);
        BARRIER
        #endif

        syz_yr = Dym(lsyzr);
        syz_zr = Dzm(lsyzr);

        #if LOCAL_OFF==0
        BARRIER
        load_local_in(syyr);
        load_local_haloy(syyr);
        BARRIER
        #endif

        syy_yr = Dyp(lsyyr);

        #if LOCAL_OFF==0
        BARRIER
        load_local_in(sxyr);
        load_local_halox(sxyr);
        load_local_haloy(sxyr);
        BARRIER
        #endif

        sxy_xr = Dxm(lsxyr);
        sxy_yr = Dym(lsxyr);
        BARRIER

    }

// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
#if LOCAL_OFF==0
#if COMM12==0
    if (gidy>(NY-FDOH-1) || gidz>(NZ-FDOH-1) || (gidx-offcomm)>(NX-FDOH-1-LCOMM) ){
        return;
    }

#else
    if (gidy>(NY-FDOH-1) || gidz>(NZ-FDOH-1) ){
        return;
    }
#endif
#endif

// Backpropagate the forward velocity
#if BACK_PROP_TYPE==1
    {
        lvx=((sxx_x + sxy_y + sxz_z)*rip[indp]);
        lvy=((syy_y + sxy_x + syz_z)*rjp[indp]);
        lvz=((szz_z + sxz_x + syz_y)*rkp[indp]);

        vx[indv]-= lvx;
        vy[indv]-= lvy;
        vz[indv]-= lvz;

        // Inject the boundary values
        m=inject_ind(gidz, gidy, gidx);
        if (m!=-1){
            vx[indv]= vxbnd[m];
            vy[indv]= vybnd[m];
            vz[indv]= vzbnd[m];
        }
    }
#endif

// Correct adjoint spatial derivatives to implement CPML
#if ABS_TYPE==1
    {
        int ind;
        int i,j,k;
        if (gidz>NZ-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz - NZ+NAB+FDOH+NAB;
            ind=2*NAB-1-k;

            psi_sxz_z(k,j,i) = b_z[ind+1] * psi_sxz_z(k,j,i) + a_z[ind+1] * sxz_zr;
            sxz_zr = sxz_zr / K_z[ind+1] + psi_sxz_z(k,j,i);
            psi_syz_z(k,j,i) = b_z[ind+1] * psi_syz_z(k,j,i) + a_z[ind+1] * syz_zr;
            syz_zr = syz_zr / K_z[ind+1] + psi_syz_z(k,j,i);
            psi_szz_z(k,j,i) = b_z_half[ind] * psi_szz_z(k,j,i) + a_z_half[ind] * szz_zr;
            szz_zr = szz_zr / K_z_half[ind] + psi_szz_z(k,j,i);

        }

#if FREESURF==0
        else if (gidz-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_sxz_z(k,j,i) = b_z[k] * psi_sxz_z(k,j,i) + a_z[k] * sxz_zr;
            sxz_zr = sxz_zr / K_z[k] + psi_sxz_z(k,j,i);
            psi_syz_z(k,j,i) = b_z[k] * psi_syz_z(k,j,i) + a_z[k] * syz_zr;
            syz_zr = syz_zr / K_z[k] + psi_syz_z(k,j,i);
            psi_szz_z(k,j,i) = b_z_half[k] * psi_szz_z(k,j,i) + a_z_half[k] * szz_zr;
            szz_zr = szz_zr / K_z_half[k] + psi_szz_z(k,j,i);

        }
#endif

        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_sxy_y(k,j,i) = b_y[j] * psi_sxy_y(k,j,i) + a_y[j] * sxy_yr;
            sxy_yr = sxy_yr / K_y[j] + psi_sxy_y(k,j,i);
            psi_syy_y(k,j,i) = b_y_half[j] * psi_syy_y(k,j,i) + a_y_half[j] * syy_yr;
            syy_yr = syy_yr / K_y_half[j] + psi_syy_y(k,j,i);
            psi_syz_y(k,j,i) = b_y[j] * psi_syz_y(k,j,i) + a_y[j] * syz_yr;
            syz_yr = syz_yr / K_y[j] + psi_syz_y(k,j,i);

        }

        else if (gidy>NY-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy - NY+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-j;

            psi_sxy_y(k,j,i) = b_y[ind+1] * psi_sxy_y(k,j,i) + a_y[ind+1] * sxy_yr;
            sxy_yr = sxy_yr / K_y[ind+1] + psi_sxy_y(k,j,i);
            psi_syy_y(k,j,i) = b_y_half[ind] * psi_syy_y(k,j,i) + a_y_half[ind] * syy_yr;
            syy_yr = syy_yr / K_y_half[ind] + psi_syy_y(k,j,i);
            psi_syz_y(k,j,i) = b_y[ind+1] * psi_syz_y(k,j,i) + a_y[ind+1] * syz_yr;
            syz_yr = syz_yr / K_y[ind+1] + psi_syz_y(k,j,i);


        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_sxx_x(k,j,i) = b_x_half[i] * psi_sxx_x(k,j,i) + a_x_half[i] * sxx_xr;
            sxx_xr = sxx_xr / K_x_half[i] + psi_sxx_x(k,j,i);
            psi_sxy_x(k,j,i) = b_x[i] * psi_sxy_x(k,j,i) + a_x[i] * sxy_xr;
            sxy_xr = sxy_xr / K_x[i] + psi_sxy_x(k,j,i);
            psi_sxz_x(k,j,i) = b_x[i] * psi_sxz_x(k,j,i) + a_x[i] * sxz_xr;
            sxz_xr = sxz_xr / K_x[i] + psi_sxz_x(k,j,i);

        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){

            i =gidx - NX+NAB+FDOH+NAB;
            j =gidy-FDOH;
            k =gidz-FDOH;
            ind=2*NAB-1-i;

            psi_sxx_x(k,j,i) = b_x_half[ind] * psi_sxx_x(k,j,i) + a_x_half[ind] * sxx_xr;
            sxx_xr = sxx_xr / K_x_half[ind] + psi_sxx_x(k,j,i);
            psi_sxy_x(k,j,i) = b_x[ind+1] * psi_sxy_x(k,j,i) + a_x[ind+1] * sxy_xr;
            sxy_xr = sxy_xr / K_x[ind+1] + psi_sxy_x(k,j,i);
            psi_sxz_x(k,j,i) = b_x[ind+1] * psi_sxz_x(k,j,i) + a_x[ind+1] * sxz_xr;
            sxz_xr = sxz_xr / K_x[ind+1] + psi_sxz_x(k,j,i);



        }
#endif
    }
#endif

    // Update adjoint velocities
    lvx=((sxx_xr + sxy_yr + sxz_zr)*rip[indp]);
    lvy=((syy_yr + sxy_xr + syz_zr)*rjp[indp]);
    lvz=((szz_zr + sxz_xr + syz_yr)*rkp[indp]);
    vxr[indv]+= lvx;
    vyr[indv]+= lvy;
    vzr[indv]+= lvz;


// Absorbing boundary
#if ABS_TYPE==2
    {
        #if FREESURF==0
        if (gidz-FDOH<NAB){
            vxr[indv]*=taper[gidz-FDOH];
            vyr[indv]*=taper[gidz-FDOH];
            vzr[indv]*=taper[gidz-FDOH];
        }
        #endif

        if (gidz>NZ-NAB-FDOH-1){
            vxr[indv]*=taper[NZ-FDOH-gidz-1];
            vyr[indv]*=taper[NZ-FDOH-gidz-1];
            vzr[indv]*=taper[NZ-FDOH-gidz-1];
        }

        if (gidy-FDOH<NAB){
            vxr[indv]*=taper[gidy-FDOH];
            vyr[indv]*=taper[gidy-FDOH];
            vzr[indv]*=taper[gidy-FDOH];
        }

        if (gidy>NY-NAB-FDOH-1){
            vxr[indv]*=taper[NY-FDOH-gidy-1];
            vyr[indv]*=taper[NY-FDOH-gidy-1];
            vzr[indv]*=taper[NY-FDOH-gidy-1];
        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            vxr[indv]*=taper[gidx-FDOH];
            vyr[indv]*=taper[gidx-FDOH];
            vzr[indv]*=taper[gidx-FDOH];
        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            vxr[indv]*=taper[NX-FDOH-gidx-1];
            vyr[indv]*=taper[NX-FDOH-gidx-1];
            vzr[indv]*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif

// Density gradient calculation on the fly
#if BACK_PROP_TYPE==1
    gradrho[indp]+=-vx[indv]*lvx-vy[indv]*lvy-vz[indv]*lvz;

#if HOUT==1
    Hrho[inp])+= pown(vx[indv],2)+pown(vy[indv],2)+pown(vz[indv],2);
#endif

#endif

#if GRADSRCOUT==1
    //TODO
//    if (nsrc>0){
//
//
//        for (int srci=0; srci<nsrc; srci++){
//
//
//            int i=(int)(srcpos_loc(0,srci)/DH-0.5)+FDOH;
//            int j=(int)(srcpos_loc(1,srci)/DH-0.5)+FDOH;
//            int k=(int)(srcpos_loc(2,srci)/DH-0.5)+FDOH;
//
//            if (i==gidx && j==gidy && k==gidz){
//
//                int SOURCE_TYPE= (int)srcpos_loc(4,srci);
//
//                if (SOURCE_TYPE==2){
//                    /* single force in x */
//                    gradsrc(srci,nt)+= vxr[indv]/rip[inp]/(DH*DH*DH);
//                }
//                else if (SOURCE_TYPE==3){
//                    /* single force in y */
//
//                    gradsrc(srci,nt)+= vyr[indv]/rip[inp]/(DH*DH*DH);
//                }
//                else if (SOURCE_TYPE==4){
//                    /* single force in z */
//
//                    gradsrc(srci,nt)+= vzr[indv]/rip[inp]/(DH*DH*DH);
//                }
//
//            }
//        }
//
//
//    }
#endif
    
}

