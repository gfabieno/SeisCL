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

/*Update of the stresses in 2D SV*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */

#define psi_vx_x(z,x) psi_vx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_vz_x(z,x) psi_vz_x[(x)*(NZ-2*FDOH)+(z)]

#define psi_vx_z(z,x) psi_vx_z[(x)*(2*NAB)+(z)]
#define psi_vz_z(z,x) psi_vz_z[(x)*(2*NAB)+(z)]

FUNDEF void update_s(int offcomm,
                       GLOBARG float *vx,         GLOBARG float *vz,
                       GLOBARG float *sxx,        GLOBARG float *szz,        GLOBARG float *sxz,
                       GLOBARG float *M,         GLOBARG float *mu,          GLOBARG float *muipkp,
                       GLOBARG float *rxx,        GLOBARG float *rzz,        GLOBARG float *rxz,
                       GLOBARG float *taus,       GLOBARG float *tausipkp,   GLOBARG float *taup,
                       GLOBARG float *eta,        GLOBARG float *taper,
                       GLOBARG float *K_x,        GLOBARG float *a_x,          GLOBARG float *b_x,
                       GLOBARG float *K_x_half,   GLOBARG float *a_x_half,     GLOBARG float *b_x_half,
                       GLOBARG float *K_z,        GLOBARG float *a_z,          GLOBARG float *b_z,
                       GLOBARG float *K_z_half,   GLOBARG float *a_z_half,     GLOBARG float *b_z_half,
                       GLOBARG float *psi_vx_x,    GLOBARG float *psi_vx_z,
                       GLOBARG float *psi_vz_x,    GLOBARG float *psi_vz_z,
                       LOCARG )
{
    
    LOCDEF
    
    float vxx, vzz, vzx, vxz;
    int i,k,l,ind;
    float sumrxz, sumrxx, sumrzz;
    float b,c,e,g,d,f,fipkp,dipkp;
#if LVE>0
    float leta[LVE];
#endif
    float lM, lmu, lmuipkp, ltaup, ltaus, ltausipkp;
    float lsxx, lszz, lsxz;
    
// If we use local memory
#if LOCAL_OFF==0
#ifdef __OPENCL_VERSION__
    int lsizez = get_local_size(0)+2*FDOH;
    int lsizex = get_local_size(1)+2*FDOH;
    int lidz = get_local_id(0)+FDOH;
    int lidx = get_local_id(1)+FDOH;
    int gidz = get_global_id(0)+FDOH;
    int gidx = get_global_id(1)+FDOH+offcomm;
#else
    int lsizez = blockDim.x+2*FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
#endif

    #define lvx lvar
    #define lvz lvar
    
// If local memory is turned off
#elif LOCAL_OFF==1
    
#ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
    int glsizez = (NZ-2*FDOH);
    int gidz = gid%glsizez+FDOH;
    int gidx = (gid/glsizez)+FDOH+offcomm;
#else
    int lsizez = blockDim.x+2*FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
#endif
    
    
    #define lvx vx
    #define lvz vz
    
#endif

    int indp = (gidx-FDOH)*(NZ-2*FDOH)+(gidz-FDOH);
    int indv = gidx*NZ+gidz;
    
// Calculation of the velocity spatial derivatives
    {
#if LOCAL_OFF==0
        load_local_in(vx);
        load_local_haloz(vx);
        load_local_halox(vx);
        BARRIER
#endif
        vxx = Dxm(lvx);
        vxz = Dzp(lvx);
       
        
#if LOCAL_OFF==0
        BARRIER
        load_local_in(vz);
        load_local_haloz(vz);
        load_local_halox(vz);
        BARRIER
#endif
        vzz = Dzm(lvz);
        vzx = Dxp(lvz);
    }
    
    
// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
#if LOCAL_OFF==0
#if COMM12==0
    if (gidz>(NZ-FDOH-1) || (gidx-offcomm)>(NX-FDOH-1-LCOMM) ){
        return;
    }
#else
    if (gidz>(NZ-FDOH-1) ){
        return;
    }
#endif
#endif

    
// Correct spatial derivatives to implement CPML
    
#if ABS_TYPE==1
        {
        
        if (gidz>NZ-NAB-FDOH-1){
            
            i =gidx-FDOH;
            k =gidz - NZ+NAB+FDOH+NAB;
            ind=2*NAB-1-k;
            
            psi_vx_z(k,i) = b_z_half[ind] * psi_vx_z(k,i) + a_z_half[ind] * vxz;
            vxz = vxz / K_z_half[ind] + psi_vx_z(k,i);
            psi_vz_z(k,i) = b_z[ind+1] * psi_vz_z(k,i) + a_z[ind+1] * vzz;
            vzz = vzz / K_z[ind+1] + psi_vz_z(k,i);
            
        }
        
#if FREESURF==0
        else if (gidz-FDOH<NAB){
            
            i =gidx-FDOH;
            k =gidz-FDOH;
            
            
            psi_vx_z(k,i) = b_z_half[k] * psi_vx_z(k,i) + a_z_half[k] * vxz;
            vxz = vxz / K_z_half[k] + psi_vx_z(k,i);
            psi_vz_z(k,i) = b_z[k] * psi_vz_z(k,i) + a_z[k] * vzz;
            vzz = vzz / K_z[k] + psi_vz_z(k,i);
            
            
        }
#endif
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            
            i =gidx-FDOH;
            k =gidz-FDOH;
            
            psi_vx_x(k,i) = b_x[i] * psi_vx_x(k,i) + a_x[i] * vxx;
            vxx = vxx / K_x[i] + psi_vx_x(k,i);
            psi_vz_x(k,i) = b_x_half[i] * psi_vz_x(k,i) + a_x_half[i] * vzx;
            vzx = vzx / K_x_half[i] + psi_vz_x(k,i);
            
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            
            i =gidx - NX+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-i;
            
            
            psi_vx_x(k,i) = b_x[ind+1] * psi_vx_x(k,i) + a_x[ind+1] * vxx;
            vxx = vxx /K_x[ind+1] + psi_vx_x(k,i);
            psi_vz_x(k,i) = b_x_half[ind] * psi_vz_x(k,i) + a_x_half[ind] * vzx;
            vzx = vzx / K_x_half[ind]  +psi_vz_x(k,i);
            
            
        }
#endif
       }
#endif
    

// Read model parameters into local memory
    {
#if LVE==0
        fipkp=muipkp[indp];
        f=2.0*mu[indp];
        g=M[indp];
        
#else
        
        lM=M[indp];
        lmu=mu[indp];
        lmuipkp=muipkp[indp];
        ltaup=taup[indp];
        ltaus=taus[indp];
        ltausipkp=tausipkp[indp];
        
        for (l=0;l<LVE;l++){
            leta[l]=eta[l];
        }
        
        fipkp=lmuipkp*(1.0+ (float)LVE*ltausipkp);
        g=lM*(1.0+(float)LVE*ltaup);
        f=2.0*lmu*(1.0+(float)LVE*ltaus);
        dipkp=lmuipkp*ltausipkp/DT;
        d=2.0*lmu*ltaus/DT;
        e=lM*ltaup/DT;
        
#endif
    }
    
// Update the stresses
    {
#if LVE==0

        sxz[indv]+=(fipkp*(vxz+vzx));
        sxx[indv]+=(g*(vxx+vzz))-(f*vzz);
        szz[indv]+=(g*(vxx+vzz))-(f*vxx);

#else
        
        
        /* computing sums of the old memory variables */
        sumrxz=sumrxx=sumrzz=0;
        int indr;
        for (l=0;l<LVE;l++){
            indr = l*NX*NZ + gidx*NZ+gidz;
            sumrxz+=rxz[indr];
            sumrxx+=rxx[indr];
            sumrzz+=rzz[indr];
        }
        
        
        /* updating components of the stress tensor, partially */
        lsxz=(fipkp*(vxz+vzx))+(DT2*sumrxz);
        lsxx=((g*(vxx+vzz))-(f*vzz))+(DT2*sumrxx);
        lszz=((g*(vxx+vzz))-(f*vxx))+(DT2*sumrzz);
        
        
        /* now updating the memory-variables and sum them up*/
        sumrxz=sumrxx=sumrzz=0;
        for (l=0;l<LVE;l++){
            b=1.0/(1.0+(leta[l]*0.5));
            c=1.0-(leta[l]*0.5);
            indr = l*NX*NZ + gidx*NZ+gidz;
            rxz[indr]=b*(rxz[indr]*c-leta[l]*(dipkp*(vxz+vzx)));
            rxx[indr]=b*(rxx[indr]*c-leta[l]*((e*(vxx+vzz))-(d*vzz)));
            rzz[indr]=b*(rzz[indr]*c-leta[l]*((e*(vxx+vzz))-(d*vxx)));
            
            sumrxz+=rxz[indr];
            sumrxx+=rxx[indr];
            sumrzz+=rzz[indr];
        }

        
        /* and now the components of the stress tensor are
         completely updated */
        sxz[indv]+= lsxz + (DT2*sumrxz);
        sxx[indv]+= lsxx + (DT2*sumrxx) ;
        szz[indv]+= lszz + (DT2*sumrzz) ;
        
#endif
    }

// Absorbing boundary
#if ABS_TYPE==2
    {
#if FREESURF==0
        if (gidz-FDOH<NAB){
            sxx[indv]*=taper[gidz-FDOH];
            szz[indv]*=taper[gidz-FDOH];
            sxz[indv]*=taper[gidz-FDOH];
        }
#endif
        
        if (gidz>NZ-NAB-FDOH-1){
            sxx[indv]*=taper[NZ-FDOH-gidz-1];
            szz[indv]*=taper[NZ-FDOH-gidz-1];
            sxz[indv]*=taper[NZ-FDOH-gidz-1];
        }
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            sxx[indv]*=taper[gidx-FDOH];
            szz[indv]*=taper[gidx-FDOH];
            sxz[indv]*=taper[gidx-FDOH];
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            sxx[indv]*=taper[NX-FDOH-gidx-1];
            szz[indv]*=taper[NX-FDOH-gidx-1];
            sxz[indv]*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif
    
}

