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



#define psi_vx_x(z,y,x) psi_vx_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vy_x(z,y,x) psi_vy_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vz_x(z,y,x) psi_vz_x[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]

#define psi_vx_y(z,y,x) psi_vx_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vy_y(z,y,x) psi_vy_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vz_y(z,y,x) psi_vz_y[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]

#define psi_vx_z(z,y,x) psi_vx_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_vy_z(z,y,x) psi_vy_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_vz_z(z,y,x) psi_vz_z[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]



FUNDEF void update_adjs(int offcomm,
                          GLOBARG float *vx,         GLOBARG float *vy,       GLOBARG float *vz,
                          GLOBARG float *sxx,        GLOBARG float *syy,      GLOBARG float *szz,
                          GLOBARG float *sxy,        GLOBARG float *syz,      GLOBARG float *sxz,
                          GLOBARG float *vxbnd,      GLOBARG float *vybnd,    GLOBARG float *vzbnd,
                          GLOBARG float *sxxbnd,     GLOBARG float *syybnd,   GLOBARG float *szzbnd,
                          GLOBARG float *sxybnd,     GLOBARG float *syzbnd,   GLOBARG float *sxzbnd,
                          GLOBARG float *vxr,       GLOBARG float *vyr,     GLOBARG float *vzr,
                          GLOBARG float *sxxr,      GLOBARG float *syyr,    GLOBARG float *szzr,
                          GLOBARG float *sxyr,      GLOBARG float *syzr,    GLOBARG float *sxzr,
                          GLOBARG float *rxx,        GLOBARG float *ryy,      GLOBARG float *rzz,
                          GLOBARG float *rxy,        GLOBARG float *ryz,      GLOBARG float *rxz,
                          GLOBARG float *rxxr,      GLOBARG float *ryyr,    GLOBARG float *rzzr,
                          GLOBARG float *rxyr,      GLOBARG float *ryzr,    GLOBARG float *rxzr,
                          GLOBARG float *M,         GLOBARG float *mu,        GLOBARG float *muipjp,
                          GLOBARG float *mujpkp,      GLOBARG float *muipkp,
                          GLOBARG float *taus,       GLOBARG float *tausipjp, GLOBARG float *tausjpkp,
                          GLOBARG float *tausipkp,   GLOBARG float *taup,     GLOBARG float *eta,
                          GLOBARG float *taper,
                          GLOBARG float *K_x,        GLOBARG float *a_x,      GLOBARG float *b_x,
                          GLOBARG float *K_x_half,   GLOBARG float *a_x_half, GLOBARG float *b_x_half,
                          GLOBARG float *K_y,        GLOBARG float *a_y,      GLOBARG float *b_y,
                          GLOBARG float *K_y_half,   GLOBARG float *a_y_half, GLOBARG float *b_y_half,
                          GLOBARG float *K_z,        GLOBARG float *a_z,      GLOBARG float *b_z,
                          GLOBARG float *K_z_half,   GLOBARG float *a_z_half, GLOBARG float *b_z_half,
                          GLOBARG float *psi_vx_x,    GLOBARG float *psi_vx_y,  GLOBARG float *psi_vx_z,
                          GLOBARG float *psi_vy_x,    GLOBARG float *psi_vy_y,  GLOBARG float *psi_vy_z,
                          GLOBARG float *psi_vz_x,    GLOBARG float *psi_vz_y,  GLOBARG float *psi_vz_z,
                          GLOBARG float *gradrho,    GLOBARG float *gradM,    GLOBARG float *gradmu,
                          GLOBARG float *gradtaup,   GLOBARG float *gradtaus, GLOBARG float *gradsrc,
                          GLOBARG float *Hrho,    GLOBARG float *HM,     GLOBARG float *Hmu,
                          GLOBARG float *Htaup,   GLOBARG float *Htaus,  GLOBARG float *Hsrc,
                          LOCARG)
{
    
    LOCDEF
    
    int m;
    float vxx,vxy,vxz,vyx,vyy,vyz,vzx,vzy,vzz;
    float vxyyx,vyzzy,vxzzx,vxxyyzz,vyyzz,vxxzz,vxxyy;
    float vxxr,vxyr,vxzr,vyxr,vyyr,vyzr,vzxr,vzyr,vzzr;
    float vxyyxr,vyzzyr,vxzzxr,vxxyyzzr,vyyzzr,vxxzzr,vxxyyr;
    float fipjp, fjpkp, fipkp, f, g;
    float lM, lmu;
    #if LVE>0
    float sumrxy,sumryz,sumrxz,sumrxx,sumryy,sumrzz;
    float b,c,e,d,dipjp,djpkp,dipkp;
    int l, indr;;
    float ltaup, ltaus, ltausipjp, ltausipkp, ltausjpkp, lmuipjp, lmuipkp, lmujpkp;
    float leta[LVE];
    #endif
    float lsxx, lsyy, lszz, lsxy, lsxz, lsyz;
    
    
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
    
    #define lvx lvar
    #define lvy lvar
    #define lvz lvar
    #define lvxr lvar
    #define lvyr lvar
    #define lvzr lvar

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
    
    #define lvxr vxr
    #define lvyr vyr
    #define lvzr vzr
    #define lvx vx
    #define lvy vy
    #define lvz vz
    #define lidx gidx
    #define lidy gidy
    #define lidz gidz
    
    #define lsizez NZ
    #define lsizey NY
    #define lsizex NX
    
    #endif
    
    int indp = ((gidx)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((gidy)-FDOH)*(NZ-2*FDOH)+((gidz)-FDOH);
    int indv = (gidx)*NZ*NY+(gidy)*NZ+(gidz);

// Calculation of the velocity spatial derivatives of the forward wavefield if backpropagation is used
#if BACK_PROP_TYPE==1
    {
         #if LOCAL_OFF==0
        load_local_in(vz);
        load_local_haloz(vz);
        load_local_haloy(vz);
        load_local_halox(vz);
        BARRIER
        #endif
        
        vzx = Dxp(lvz);
        vzy = Dyp(lvz);
        vzz = Dzm(lvz);
        
        #if LOCAL_OFF==0
        BARRIER
        load_local_in(vy);
        load_local_haloz(vy);
        load_local_haloy(vy);
        load_local_halox(vy);
        BARRIER
        #endif
        
        vyx = Dxp(lvy);
        vyy = Dym(lvy);
        vyz = Dzp(lvy);

        #if LOCAL_OFF==0
        BARRIER
        load_local_in(vx);
        load_local_haloz(vx);
        load_local_haloy(vx);
        load_local_halox(vx);
        BARRIER
        #endif
        
        vxx = Dxm(lvx);
        vxy = Dyp(lvx);
        vxz = Dzp(lvx);

        BARRIER
    }
#endif

// Calculation of the velocity spatial derivatives of the adjoint wavefield
    {
         #if LOCAL_OFF==0
        load_local_in(vzr);
        load_local_haloz(vzr);
        load_local_haloy(vzr);
        load_local_halox(vzr);
        BARRIER
        #endif

        vzxr = Dxp(lvzr);
        vzyr = Dyp(lvzr);
        vzzr = Dzm(lvzr);

        #if LOCAL_OFF==0
        BARRIER
        load_local_in(vyr);
        load_local_haloz(vyr);
        load_local_haloy(vyr);
        load_local_halox(vyr);
        BARRIER
        #endif

        vyxr = Dxp(lvyr);
        vyyr = Dym(lvyr);
        vyzr = Dzp(lvyr);

        #if LOCAL_OFF==0
        BARRIER
        load_local_in(vxr);
        load_local_haloz(vxr);
        load_local_haloy(vxr);
        load_local_halox(vxr);
        BARRIER
        #endif

        vxxr = Dxm(lvxr);
        vxyr = Dyp(lvxr);
        vxzr = Dzp(lvxr);
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
    
 
    
// Read model parameters into local memory
#if LVE==0
    lM=M[indp];
    lmu=mu[indp];
    fipjp=muipjp[indp];
    fjpkp=mujpkp[indp];
    fipkp=muipkp[indp];
    g=lM;
    f=2.0*lmu;

    
#else
    
    lM=M[indp];
    lmu=u[indp];
    lmuipkp=muipkp[indp];
    lmuipjp=muipjp[indp];
    lmujpkp=mujpkp[indp];
    ltaup=taup[indp];
    ltaus=taus[indp];
    ltausipkp=tausipkp[indp];
    ltausipjp=tausipjp[indp];
    ltausjpkp=tausjpkp[indp];
    
    for (l=0;l<LVE;l++){
        leta[l]=eta[l];
    }
    
    fipjp=lmuipjp*(1.0+ (float)LVE*ltausipjp);
    fjpkp=lmujpkp*(1.0+ (float)LVE*ltausjpkp);
    fipkp=lmuipkp*(1.0+ (float)LVE*ltausipkp);
    g=lM*(1.0+(float)LVE*ltaup);
    f=2.0*lmu*(1.0+(float)LVE*ltaus);
    dipjp=lmuipjp*ltausipjp/DT;
    djpkp=lmujpkp*ltausjpkp/DT;
    dipkp=lmuipkp*ltausipkp/DT;
    d=2.0*lmu*ltaus/DT;
    e=lM*ltaup/DT;
    
    
#endif
    
    
// Backpropagate the forward stresses
#if BACK_PROP_TYPE==1
    {
#if LVE==0

        vxyyx=vxy+vyx;
        vyzzy=vyz+vzy;
        vxzzx=vxz+vzx;
        vxxyyzz=vxx+vyy+vzz;
        vyyzz=vyy+vzz;
        vxxzz=vxx+vzz;
        vxxyy=vxx+vyy;

        sxy[indv]-=(fipjp*vxyyx);
        syz[indv]-=(fjpkp*vyzzy);
        sxz[indv]-=(fipkp*vxzzx);
        sxx[indv]-=((g*vxxyyzz)-(f*vyyzz)) ;
        syy[indv]-=((g*vxxyyzz)-(f*vxxzz)) ;
        szz[indv]-=((g*vxxyyzz)-(f*vxxyy)) ;

// Backpropagation is not stable for viscoelastic wave equation
#else


        /* computing sums of the old memory variables */
        sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
        for (l=0;l<LVE;l++){
            indr = l*NX*NY*NZ + gidx*NY*NZ + gidy*NZ +gidz;
            sumrxy+=rxy[indr];
            sumryz+=ryz[indr];
            sumrxz+=rxz[indr];
            sumrxx+=rxx[indr];
            sumryy+=ryy[indr];
            sumrzz+=rzz[indr];
        }

        /* updating components of the stress tensor, partially */
        vxyyx=vxy+vyx;
        vyzzy=vyz+vzy;
        vxzzx=vxz+vzx;
        vxxyyzz=vxx+vyy+vzz;
        vyyzz=vyy+vzz;
        vxxzz=vxx+vzz;
        vxxyy=vxx+vyy;

        lsxy=(fipjp*vxyyx)+(DT2*sumrxy);
        lsyz=(fjpkp*vyzzy)+(DT2*sumryz);
        lsxz=(fipkp*vxzzx)+(DT2*sumrxz);
        lsxx=((g*vxxyyzz)-(f*vyyzz))+(DT2*sumrxx);
        lsyy=((g*vxxyyzz)-(f*vxxzz))+(DT2*sumryy);
        lszz=((g*vxxyyzz)-(f*vxxyy))+(DT2*sumrzz);


        sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
        for (l=0;l<LVE;l++){
            //Those variables change sign for reverse time
            b=1.0/(1.0+(leta[l]*0.5));
            c=1.0-(leta[l]*0.5);
            indr = l*NX*NY*NZ + gidx*NY*NZ + gidy*NZ +gidz;

            rxy[indr]=b*(rxy[indr]*c-leta[l]*(dipjp*vxyyx));
            ryz[indr]=b*(ryz[indr]*c-leta[l]*(djpkp*vyzzy));
            rxz[indr]=b*(rxz[indr]*c-leta[l]*(dipkp*vxzzx));
            rxx[indr]=b*(rxx[indr]*c-leta[l]*((e*vxxyyzz)-(d*vyyzz)));
            ryy[indr]=b*(ryy[indr]*c-leta[l]*((e*vxxyyzz)-(d*vxxzz)));
            rzz[indr]=b*(rzz[indr]*c-leta[l]*((e*vxxyyzz)-(d*vxxyy)));

            sumrxy+=rxy[indr];
            sumryz+=ryz[indr];
            sumrxz+=rxz[indr];
            sumrxx+=rxx[indr];
            sumryy+=ryy[indr];
            sumrzz+=rzz[indr];
        }

        /* and now the components of the stress tensor are
         completely updated */
        sxy[indv]-=lsxy+(DT2*sumrxy);
        syz[indv]-=lsyz+(DT2*sumryz);
        sxz[indv]-=lsxz+(DT2*sumrxz);
        sxx[indv]-=lsxx+(DT2*sumrxx);
        syy[indv]-=lsyy+(DT2*sumryy);
        szz[indv]-=lszz+(DT2*sumrzz);

#endif

        m=inject_ind(gidz, gidy, gidx);
        if (m!=-1){
            sxx[indv]= sxxbnd[m];
            syy[indv]= syybnd[m];
            szz[indv]= szzbnd[m];
            sxy[indv]= sxybnd[m];
            syz[indv]= syzbnd[m];
            sxz[indv]= sxzbnd[m];
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

            psi_vx_z(k,j,i) = b_z_half[ind] * psi_vx_z(k,j,i) + a_z_half[ind] * vxzr;
            vxzr = vxzr / K_z_half[ind] + psi_vx_z(k,j,i);
            psi_vy_z(k,j,i) = b_z_half[ind] * psi_vy_z(k,j,i) + a_z_half[ind] * vyzr;
            vyzr = vyzr / K_z_half[ind] + psi_vy_z(k,j,i);
            psi_vz_z(k,j,i) = b_z[ind+1] * psi_vz_z(k,j,i) + a_z[ind+1] * vzzr;
            vzzr = vzzr / K_z[ind+1] + psi_vz_z(k,j,i);

        }

    #if FREESURF==0
        else if (gidz-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;


            psi_vx_z(k,j,i) = b_z_half[k] * psi_vx_z(k,j,i) + a_z_half[k] * vxzr;
            vxzr = vxzr / K_z_half[k] + psi_vx_z(k,j,i);
            psi_vy_z(k,j,i) = b_z_half[k] * psi_vy_z(k,j,i) + a_z_half[k] * vyzr;
            vyzr = vyzr / K_z_half[k] + psi_vy_z(k,j,i);
            psi_vz_z(k,j,i) = b_z[k] * psi_vz_z(k,j,i) + a_z[k] * vzzr;
            vzzr = vzzr / K_z[k] + psi_vz_z(k,j,i);


        }
    #endif

        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_vx_y(k,j,i) = b_y_half[j] * psi_vx_y(k,j,i) + a_y_half[j] * vxyr;
            vxyr = vxyr / K_y_half[j] + psi_vx_y(k,j,i);
            psi_vy_y(k,j,i) = b_y[j] * psi_vy_y(k,j,i) + a_y[j] * vyyr;
            vyyr = vyyr / K_y[j] + psi_vy_y(k,j,i);
            psi_vz_y(k,j,i) = b_y_half[j] * psi_vz_y(k,j,i) + a_y_half[j] * vzyr;
            vzyr = vzyr / K_y_half[j] + psi_vz_y(k,j,i);

        }

        else if (gidy>NY-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy - NY+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-j;


            psi_vx_y(k,j,i) = b_y_half[ind] * psi_vx_y(k,j,i) + a_y_half[ind] * vxyr;
            vxyr = vxyr / K_y_half[ind] + psi_vx_y(k,j,i);
            psi_vy_y(k,j,i) = b_y[ind+1] * psi_vy_y(k,j,i) + a_y[ind+1] * vyyr;
            vyyr = vyyr / K_y[ind+1] + psi_vy_y(k,j,i);
            psi_vz_y(k,j,i) = b_y_half[ind] * psi_vz_y(k,j,i) + a_y_half[ind] * vzyr;
            vzyr = vzyr / K_y_half[ind] + psi_vz_y(k,j,i);


        }
    #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_vx_x(k,j,i) = b_x[i] * psi_vx_x(k,j,i) + a_x[i] * vxxr;
            vxxr = vxxr / K_x[i] + psi_vx_x(k,j,i);
            psi_vy_x(k,j,i) = b_x_half[i] * psi_vy_x(k,j,i) + a_x_half[i] * vyxr;
            vyxr = vyxr / K_x_half[i] + psi_vy_x(k,j,i);
            psi_vz_x(k,j,i) = b_x_half[i] * psi_vz_x(k,j,i) + a_x_half[i] * vzxr;
            vzxr = vzxr / K_x_half[i] + psi_vz_x(k,j,i);


        }
    #endif

    #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){

            i =gidx - NX+NAB+FDOH+NAB;
            j =gidy-FDOH;
            k =gidz-FDOH;
            ind=2*NAB-1-i;


            psi_vx_x(k,j,i) = b_x[ind+1] * psi_vx_x(k,j,i) + a_x[ind+1] * vxxr;
            vxxr = vxxr /K_x[ind+1] + psi_vx_x(k,j,i);
            psi_vy_x(k,j,i) = b_x_half[ind] * psi_vy_x(k,j,i) + a_x_half[ind] * vyxr;
            vyxr = vyxr  /K_x_half[ind] + psi_vy_x(k,j,i);
            psi_vz_x(k,j,i) = b_x_half[ind] * psi_vz_x(k,j,i) + a_x_half[ind] * vzxr;
            vzxr = vzxr / K_x_half[ind]  +psi_vz_x(k,j,i);


        }
    #endif

    }
#endif

// Update adjoint stresses
    {
#if LVE==0

    vxyyxr=vxyr+vyxr;
    vyzzyr=vyzr+vzyr;
    vxzzxr=vxzr+vzxr;
    vxxyyzzr=vxxr+vyyr+vzzr;
    vyyzzr=vyyr+vzzr;
    vxxzzr=vxxr+vzzr;
    vxxyyr=vxxr+vyyr;

    lsxy=(fipjp*vxyyxr);
    lsyz=(fjpkp*vyzzyr);
    lsxz=(fipkp*vxzzxr);
    lsxx=((g*vxxyyzzr)-(f*vyyzzr));
    lsyy=((g*vxxyyzzr)-(f*vxxzzr));
    lszz=((g*vxxyyzzr)-(f*vxxyyr));

    sxyr[indv]+=lsxy;
    syzr[indv]+=lsyz;
    sxzr[indv]+=lsxz;
    sxxr[indv]+=lsxx;
    syyr[indv]+=lsyy;
    szzr[indv]+=lszz;


#else

    /* computing sums of the old memory variables */
    sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
    for (l=0;l<LVE;l++){
        indr = l*NX*NY*NZ + gidx*NY*NZ + gidy*NZ +gidz;
        sumrxy+=rxyr[indr];
        sumryz+=ryzr[indr];
        sumrxz+=rxzr[indr];
        sumrxx+=rxxr[indr];
        sumryy+=ryyr[indr];
        sumrzz+=rzzr[indr];
    }

    vxyyxr=vxyr+vyxr;
    vyzzyr=vyzr+vzyr;
    vxzzxr=vxzr+vzxr;
    vxxyyzzr=vxxr+vyyr+vzzr;
    vyyzzr=vyyr+vzzr;
    vxxzzr=vxxr+vzzr;
    vxxyyr=vxxr+vyyr;

    lsxy=(fipjp*vxyyxr)+(DT2*sumrxy);
    lsyz=(fjpkp*vyzzyr)+(DT2*sumryz);
    lsxz=(fipkp*vxzzxr)+(DT2*sumrxz);
    lsxx=((g*vxxyyzzr)-(f*vyyzzr))+(DT2*sumrxx);
    lsyy=((g*vxxyyzzr)-(f*vxxzzr))+(DT2*sumryy);
    lszz=((g*vxxyyzzr)-(f*vxxyyr))+(DT2*sumrzz);

    sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
    for (l=0;l<LVE;l++){
        b=1.0/(1.0+(leta[l]*0.5));
        c=1.0-(leta[l]*0.5);
        indr = l*NX*NY*NZ + gidx*NY*NZ + gidy*NZ +gidz;

        rxy[indr]=b*(rxyr[indr]*c-leta[l]*(dipjp*vxyyxr));
        ryz[indr]=b*(ryzr[indr]*c-leta[l]*(djpkp*vyzzyr));
        rxz[indr]=b*(rxzr[indr]*c-leta[l]*(dipkp*vxzzxr));
        rxx[indr]=b*(rxxr[indr]*c-leta[l]*((e*vxxyyzzr)-(d*vyyzzr)));
        ryy[indr]=b*(ryyr[indr]*c-leta[l]*((e*vxxyyzzr)-(d*vxxzzr)));
        rzz[indr]=b*(rzzr[indr]*c-leta[l]*((e*vxxyyzzr)-(d*vxxyyr)));

        sumrxy=rxyr[indr];
        sumryz=ryzr[indr];
        sumrxz=rxzr[indr];
        sumrxx=rxxr[indr];
        sumryy=ryyr[indr];
        sumrzz=rzzr[indr];
    }

    /* and now the components of the stress tensor are
     completely updated */
    sxyr[indv]+=lsxy+(DT2*sumrxy);
    syzr[indv]+=lsyz+(DT2*sumryz);
    sxzr[indv]+=lsxz+(DT2*sumrxz);
    sxxr[indv]+=lsxx+(DT2*sumrxx);
    syyr[indv]+=lsyy+(DT2*sumryy);
    szzr[indv]+=lszz+(DT2*sumrzz);

#endif
}


// Absorbing boundary
#if ABS_TYPE==2
    {
        if (gidz-FDOH<NAB){
            sxyr[indv]*=taper[gidz-FDOH];
            syzr[indv]*=taper[gidz-FDOH];
            sxzr[indv]*=taper[gidz-FDOH];
            sxxr[indv]*=taper[gidz-FDOH];
            syyr[indv]*=taper[gidz-FDOH];
            szzr[indv]*=taper[gidz-FDOH];
        }

        if (gidz>NZ-NAB-FDOH-1){
            sxyr[indv]*=taper[NZ-FDOH-gidz-1];
            syzr[indv]*=taper[NZ-FDOH-gidz-1];
            sxzr[indv]*=taper[NZ-FDOH-gidz-1];
            sxxr[indv]*=taper[NZ-FDOH-gidz-1];
            syyr[indv]*=taper[NZ-FDOH-gidz-1];
            szzr[indv]*=taper[NZ-FDOH-gidz-1];
        }
        if (gidy-FDOH<NAB){
            sxyr[indv]*=taper[gidy-FDOH];
            syzr[indv]*=taper[gidy-FDOH];
            sxzr[indv]*=taper[gidy-FDOH];
            sxxr[indv]*=taper[gidy-FDOH];
            syyr[indv]*=taper[gidy-FDOH];
            szzr[indv]*=taper[gidy-FDOH];
        }

        if (gidy>NY-NAB-FDOH-1){
            sxyr[indv]*=taper[NY-FDOH-gidy-1];
            syzr[indv]*=taper[NY-FDOH-gidy-1];
            sxzr[indv]*=taper[NY-FDOH-gidy-1];
            sxxr[indv]*=taper[NY-FDOH-gidy-1];
            syyr[indv]*=taper[NY-FDOH-gidy-1];
            szzr[indv]*=taper[NY-FDOH-gidy-1];
        }

#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            sxyr[indv]*=taper[gidx-FDOH];
            syzr[indv]*=taper[gidx-FDOH];
            sxzr[indv]*=taper[gidx-FDOH];
            sxxr[indv]*=taper[gidx-FDOH];
            syyr[indv]*=taper[gidx-FDOH];
            szzr[indv]*=taper[gidx-FDOH];
        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            sxyr[indv]*=taper[NX-FDOH-gidx-1];
            syzr[indv]*=taper[NX-FDOH-gidx-1];
            sxzr[indv]*=taper[NX-FDOH-gidx-1];
            sxxr[indv]*=taper[NX-FDOH-gidx-1];
            syyr[indv]*=taper[NX-FDOH-gidx-1];
            szzr[indv]*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif

//Shear wave modulus and P-wave modulus gradient calculation on the fly
#if BACK_PROP_TYPE==1
    #if RESTYPE==0
    float c1=1.0/(3.0*lM-4.0*lmu,2)/(3.0*lM-4.0*lmu,2);
    float c3=1.0/lmu/lmu;
    float c5=1.0/6.0*c3;

    float dM=c1*( sxx[indv]+syy[indv]+szz[indv] )*( lsxx+lsyy+lszz );

    gradM[indp]+=-dM;
    gradmu[indp]+=-c3*(sxz[indv]*lsxz +sxy[indv]*lsxy +syz[indv]*lsyz )
        + 4.0/3*dM-c5*(lsxx*(2.0*sxx[indv]- syy[indv]-szz[indv] )
                      +lsyy*(2.0*syy[indv]- sxx[indv]-szz[indv] )
                      +lszz*(2.0*szz[indv]- sxx[indv]-syy[indv] ));
    #if HOUT==1
    float dMH=c1*(sxx[indv]+syy[indv]+szz[indv])*(sxx[indv]+syy[indv]+szz[indv]);
    HM[indp]+= dMH;
    Hmu[indp]+=c3*(sxz[indv]*sxz[indv]+sxy[indv]*sxy[indv]+syz[indv]*syz[indv])
                    - 4.0/3*dM
                    +c5*( (2.0*sxx[indv]- syy[indv]-szz[indv])*(2.0*sxx[indv]- syy[indv]-szz[indv]))
                         +(2.0*syy[indv]- sxx[indv]-szz[indv])*(2.0*syy[indv]- sxx[indv]-szz[indv])
                         +(2.0*szz[indv]- sxx[indv]-syy[indv])*(2.0*szz[indv]- sxx[indv]-syy[indv]));
    #endif
    #endif

    #if RESTYPE==1
    float dM=(sxx[indv]+syy[indv]+szz[indv] )*( lsxx+lsyy+lszz );

    gradM[indp]+=-dM;

    #if HOUT==1
    float dMH= (sxx[indv]+syy[indv]+szz[indv])*(sxx[indv]+syy[indv]+szz[indv]);
    HM[indp]+= dMH;

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

