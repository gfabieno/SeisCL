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

FUNDEF void update_s(int offcomm,  int nt,
                       GLOBARG float *vx,         GLOBARG float *vy,            GLOBARG float *vz,
                       GLOBARG float *sxx,        GLOBARG float *syy,           GLOBARG float *szz,
                       GLOBARG float *sxy,        GLOBARG float *syz,           GLOBARG float *sxz,
                       GLOBARG float *M,         GLOBARG float *mu,             GLOBARG float *muipjp,
                       GLOBARG float *mujpkp,      GLOBARG float *muipkp,
                       GLOBARG float *rxx,        GLOBARG float *ryy,           GLOBARG float *rzz,
                       GLOBARG float *rxy,        GLOBARG float *ryz,           GLOBARG float *rxz,
                       GLOBARG float *taus,       GLOBARG float *tausipjp,      GLOBARG float *tausjpkp,
                       GLOBARG float *tausipkp,   GLOBARG float *taup,          GLOBARG float *eta,
                       GLOBARG float *srcpos_loc, GLOBARG float *signals,       GLOBARG float *taper,
                       GLOBARG float *K_x,        GLOBARG float *a_x,          GLOBARG float *b_x,
                       GLOBARG float *K_x_half,   GLOBARG float *a_x_half,     GLOBARG float *b_x_half,
                       GLOBARG float *K_y,        GLOBARG float *a_y,          GLOBARG float *b_y,
                       GLOBARG float *K_y_half,   GLOBARG float *a_y_half,     GLOBARG float *b_y_half,
                       GLOBARG float *K_z,        GLOBARG float *a_z,          GLOBARG float *b_z,
                       GLOBARG float *K_z_half,   GLOBARG float *a_z_half,     GLOBARG float *b_z_half,
                       GLOBARG float *psi_vx_x,    GLOBARG float *psi_vx_y,       GLOBARG float *psi_vx_z,
                       GLOBARG float *psi_vy_x,    GLOBARG float *psi_vy_y,       GLOBARG float *psi_vy_z,
                       GLOBARG float *psi_vz_x,    GLOBARG float *psi_vz_y,       GLOBARG float *psi_vz_z,
                       LOCARG)
{
    LOCDEF
    
    float fipjp, fjpkp, fipkp, f, g;
#if LVE>0
    float leta[LVE];
    int l,indr;
    float sumrxy,sumryz,sumrxz,sumrxx,sumryy,sumrzz;
    float b,c,e,d,dipjp,djpkp,dipkp;
    float lmuipjp, lmuipkp, lmujpkp, ltaup, ltaus, ltausipjp, ltausipkp, ltausjpkp;
    float lsxx, lsyy, lszz, lsxy, lsxz, lsyz;
    float lM, lmu;
#endif
    float vxx,vxy,vxz,vyx,vyy,vyz,vzx,vzy,vzz;
    float vxyyx,vyzzy,vxzzx,vxxyyzz,vyyzz,vxxzz,vxxyy;
    
    
    
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
    
#define lvx vx
#define lvy vy
#define lvz vz
#define lidx gidx
#define lidy gidy
#define lidz gidz
    
#endif
    
    int indp = ((gidx)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((gidy)-FDOH)*(NZ-2*FDOH)+((gidz)-FDOH);
    int indv = (gidx)*NZ*NY+(gidy)*NZ+(gidz);
    
// Calculation of the velocity spatial derivatives
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
    }
    
// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
#if COMM12==0
    if (gidy>(NY-FDOH-1) || gidz>(NZ-FDOH-1) || (gidx-offcomm)>(NX-FDOH-1-LCOMM) ){
        return;
    }
#else
    if (gidy>(NY-FDOH-1) || gidz>(NZ-FDOH-1) ){
        return;
    }
#endif

 
// Correct spatial derivatives to implement CPML
#if ABS_TYPE==1
    {
        int i,j,k,indm;
        if (gidz>NZ-NAB-FDOH-1){
            
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz - NZ+NAB+FDOH+NAB;
            indm=2*NAB-1-k;

            psi_vx_z(k,j,i) = b_z_half[indm] * psi_vx_z(k,j,i) + a_z_half[indm] * vxz;
            vxz = vxz / K_z_half[indm] + psi_vx_z(k,j,i);
            psi_vy_z(k,j,i) = b_z_half[indm] * psi_vy_z(k,j,i) + a_z_half[indm] * vyz;
            vyz = vyz / K_z_half[indm] + psi_vy_z(k,j,i);
            psi_vz_z(k,j,i) = b_z[indm+1] * psi_vz_z(k,j,i) + a_z[indm+1] * vzz;
            vzz = vzz / K_z[indm+1] + psi_vz_z(k,j,i);

        }

#if FREESURF==0
        if (gidz-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;


            psi_vx_z(k,j,i) = b_z_half[k] * psi_vx_z(k,j,i) + a_z_half[k] * vxz;
            vxz = vxz / K_z_half[k] + psi_vx_z(k,j,i);
            psi_vy_z(k,j,i) = b_z_half[k] * psi_vy_z(k,j,i) + a_z_half[k] * vyz;
            vyz = vyz / K_z_half[k] + psi_vy_z(k,j,i);
            psi_vz_z(k,j,i) = b_z[k] * psi_vz_z(k,j,i) + a_z[k] * vzz;
            vzz = vzz / K_z[k] + psi_vz_z(k,j,i);


        }
#endif

        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_vx_y(k,j,i) = b_y_half[j] * psi_vx_y(k,j,i) + a_y_half[j] * vxy;
            vxy = vxy / K_y_half[j] + psi_vx_y(k,j,i);
            psi_vy_y(k,j,i) = b_y[j] * psi_vy_y(k,j,i) + a_y[j] * vyy;
            vyy = vyy / K_y[j] + psi_vy_y(k,j,i);
            psi_vz_y(k,j,i) = b_y_half[j] * psi_vz_y(k,j,i) + a_y_half[j] * vzy;
            vzy = vzy / K_y_half[j] + psi_vz_y(k,j,i);

        }

        if (gidy>NY-NAB-FDOH-1){

            i =gidx-FDOH;
            j =gidy - NY+NAB+FDOH+NAB;
            k =gidz-FDOH;
            indm=2*NAB-1-j;


            psi_vx_y(k,j,i) = b_y_half[indm] * psi_vx_y(k,j,i) + a_y_half[indm] * vxy;
            vxy = vxy / K_y_half[indm] + psi_vx_y(k,j,i);
            psi_vy_y(k,j,i) = b_y[indm+1] * psi_vy_y(k,j,i) + a_y[indm+1] * vyy;
            vyy = vyy / K_y[indm+1] + psi_vy_y(k,j,i);
            psi_vz_y(k,j,i) = b_y_half[indm] * psi_vz_y(k,j,i) + a_y_half[indm] * vzy;
            vzy = vzy / K_y_half[indm] + psi_vz_y(k,j,i);


        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){

            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;

            psi_vx_x(k,j,i) = b_x[i] * psi_vx_x(k,j,i) + a_x[i] * vxx;
            vxx = vxx / K_x[i] + psi_vx_x(k,j,i);
            psi_vy_x(k,j,i) = b_x_half[i] * psi_vy_x(k,j,i) + a_x_half[i] * vyx;
            vyx = vyx / K_x_half[i] + psi_vy_x(k,j,i);
            psi_vz_x(k,j,i) = b_x_half[i] * psi_vz_x(k,j,i) + a_x_half[i] * vzx;
            vzx = vzx / K_x_half[i] + psi_vz_x(k,j,i);


        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){

            i =gidx - NX+NAB+FDOH+NAB;
            j =gidy-FDOH;
            k =gidz-FDOH;
            indm=2*NAB-1-i;


            psi_vx_x(k,j,i) = b_x[indm+1] * psi_vx_x(k,j,i) + a_x[indm+1] * vxx;
            vxx = vxx /K_x[indm+1] + psi_vx_x(k,j,i);
            psi_vy_x(k,j,i) = b_x_half[indm] * psi_vy_x(k,j,i) + a_x_half[indm] * vyx;
            vyx = vyx  /K_x_half[indm] + psi_vy_x(k,j,i);
            psi_vz_x(k,j,i) = b_x_half[indm] * psi_vz_x(k,j,i) + a_x_half[indm] * vzx;
            vzx = vzx / K_x_half[indm]  +psi_vz_x(k,j,i);


        }
#endif
    }
#endif

// Read model parameters into local memory
    {
#if LVE==0
        
        fipjp=muipjp[indp];
        fjpkp=mujpkp[indp];
        fipkp=muipkp[indp];
        g=M[indp];
        f=2.0*mu[indp];
        
#else
        
        lM=M[indp];
        lmu=mu[indp];
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
    }

// Update the stresses
    {
#if LVE==0
        
        vxyyx=vxy+vyx;
        vyzzy=vyz+vzy;
        vxzzx=vxz+vzx;
        vxxyyzz=vxx+vyy+vzz;
        vyyzz=vyy+vzz;
        vxxzz=vxx+vzz;
        vxxyy=vxx+vyy;

        sxy[indv]+=(fipjp*vxyyx);
        syz[indv]+=(fjpkp*vyzzy);
        sxz[indv]+=(fipkp*vxzzx);
        sxx[indv]+=((g*vxxyyzz)-(f*vyyzz));
        syy[indv]+=((g*vxxyyzz)-(f*vxxzz));
        szz[indv]+=((g*vxxyyzz)-(f*vxxyy));
        
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
        sxy[indv]+=lsxy+(DT2*sumrxy);
        syz[indv]+=lsyz+(DT2*sumryz);
        sxz[indv]+=lsxz+(DT2*sumrxz);
        sxx[indv]+=lsxx+(DT2*sumrxx);
        syy[indv]+=lsyy+(DT2*sumryy);
        szz[indv]+=lszz+(DT2*sumrzz);

#endif
    }

// Absorbing boundary    
#if ABS_TYPE==2
    {
        
#if FREESURF==0
        if (gidz-FDOH<NAB){
            sxy[indv]*=taper[gidz-FDOH];
            syz[indv]*=taper[gidz-FDOH];
            sxz[indv]*=taper[gidz-FDOH];
            sxx[indv]*=taper[gidz-FDOH];
            syy[indv]*=taper[gidz-FDOH];
            szz[indv]*=taper[gidz-FDOH];
        }
#endif
        
        if (gidz>NZ-NAB-FDOH-1){
            sxy[indv]*=taper[NZ-FDOH-gidz-1];
            syz[indv]*=taper[NZ-FDOH-gidz-1];
            sxz[indv]*=taper[NZ-FDOH-gidz-1];
            sxx[indv]*=taper[NZ-FDOH-gidz-1];
            syy[indv]*=taper[NZ-FDOH-gidz-1];
            szz[indv]*=taper[NZ-FDOH-gidz-1];
        }
        if (gidy-FDOH<NAB){
            sxy[indv]*=taper[gidy-FDOH];
            syz[indv]*=taper[gidy-FDOH];
            sxz[indv]*=taper[gidy-FDOH];
            sxx[indv]*=taper[gidy-FDOH];
            syy[indv]*=taper[gidy-FDOH];
            szz[indv]*=taper[gidy-FDOH];
        }
        
        if (gidy>NY-NAB-FDOH-1){
            sxy[indv]*=taper[NY-FDOH-gidy-1];
            syz[indv]*=taper[NY-FDOH-gidy-1];
            sxz[indv]*=taper[NY-FDOH-gidy-1];
            sxx[indv]*=taper[NY-FDOH-gidy-1];
            syy[indv]*=taper[NY-FDOH-gidy-1];
            szz[indv]*=taper[NY-FDOH-gidy-1];
        }
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            sxy[indv]*=taper[gidx-FDOH];
            syz[indv]*=taper[gidx-FDOH];
            sxz[indv]*=taper[gidx-FDOH];
            sxx[indv]*=taper[gidx-FDOH];
            syy[indv]*=taper[gidx-FDOH];
            szz[indv]*=taper[gidx-FDOH];
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            sxy[indv]*=taper[NX-FDOH-gidx-1];
            syz[indv]*=taper[NX-FDOH-gidx-1];
            sxz[indv]*=taper[NX-FDOH-gidx-1];
            sxx[indv]*=taper[NX-FDOH-gidx-1];
            syy[indv]*=taper[NX-FDOH-gidx-1];
            szz[indv]*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif
}

