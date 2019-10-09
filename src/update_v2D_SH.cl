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

/*Update of the velocity in 2D SH*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#define lbnd (FDOH+NAB)

#define rho(z,x)    rho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,x)    rip[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rjp(z,x)    rjp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,x)    rkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define uipkp(z,x) uipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define ujpkp(z,x) ujpkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define uipjp(z,x) uipjp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define u(z,x)        u[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define pi(z,x)      pi[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradrho(z,x)  gradrho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradM(z,x)  gradM[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradmu(z,x)  gradmu[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaup(z,x)  gradtaup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaus(z,x)  gradtaus[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define taus(z,x)        taus[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausipkp(z,x) tausipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausipjp(z,x) tausipjp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausjpkp(z,x) tausjpkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define taup(z,x)        taup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define vx(z,x)  vx[(x)*(NZ)+(z)]
#define vy(z,x)  vy[(x)*(NZ)+(z)]
#define vz(z,x)  vz[(x)*(NZ)+(z)]
#define sxx(z,x) sxx[(x)*(NZ)+(z)]
#define szz(z,x) szz[(x)*(NZ)+(z)]
#define sxz(z,x) sxz[(x)*(NZ)+(z)]
#define sxy(z,x) sxy[(x)*(NZ)+(z)]
#define syz(z,x) syz[(x)*(NZ)+(z)]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxy(z,x,l) rxy[(l)*NX*NZ+(x)*NZ+(z)]
#define ryz(z,x,l) ryz[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_sxx_x(z,x) psi_sxx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_x(z,x) psi_sxz_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_z(z,x) psi_sxz_z[(x)*(2*NAB)+(z)]
#define psi_szz_z(z,x) psi_szz_z[(x)*(2*NAB)+(z)]
#define psi_sxy_x(z,x) psi_sxy_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_syz_z(z,x) psi_syz_z[(x)*(2*NAB)+(z)]

#define psi_vyx(z,x) psi_vyx[(x)*(NZ-2*FDOH)+(z)]
#define psi_vyz(z,x) psi_vyz[(x)*(2*NAB)+(z)]

#if LOCAL_OFF==0

#define lvar(z,x)  lvar[(x)*lsizez+(z)]

#endif


#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vy0(y,x) vy0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define ry(y,x) ry[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#define PI (3.141592653589793238462643383279502884197169)

#ifdef __OPENCL_VERSION__
#define FUNDEF __kernel
#define LFUNDEF
#define GLOBARG __global
#define LOCARG __local float *lvar
#define LOCDEF
#define BARRIER barrier(CLK_LOCAL_MEM_FENCE);
#else
#define FUNDEF extern "C" __global__
#define LFUNDEF extern "C" __device__
#define GLOBARG
#define LOCARG float *nullarg
#define LOCDEF extern __shared__ float lvar[];
#define BARRIER __syncthreads();
#endif

FUNDEF void update_v(int offcomm, int nt,
                       GLOBARG float *vy,         GLOBARG float *sxy,        GLOBARG float *syz,
                       GLOBARG float *rho,
                       GLOBARG float *taper,
                       GLOBARG float *K_x,        GLOBARG float *a_x,          GLOBARG float *b_x,
                       GLOBARG float *K_x_half,   GLOBARG float *a_x_half,     GLOBARG float *b_x_half,
                       GLOBARG float *K_z,        GLOBARG float *a_z,          GLOBARG float *b_z,
                       GLOBARG float *K_z_half,   GLOBARG float *a_z_half,     GLOBARG float *b_z_half,
                       GLOBARG float *psi_sxy_x,  GLOBARG float *psi_syz_z,
                       LOCARG)
{

    LOCDEF
    float sxy_x;
    float syz_z;
    
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
    
#define lsxy lvar
#define lsyz lvar

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
    
#define lsxy sxy
#define lsyz syz
#define lidx gidx
#define lidz gidz
    
#endif
 

// Calculation of the stresses spatial derivatives
    {
#if LOCAL_OFF==0
        BARRIER
        lsxy(lidz,lidx)=sxy(gidz,gidx);
        if (lidx<2*FDOH)
            lsxy(lidz,lidx-FDOH)=sxy(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxy(lidz,lidx+lsizex-3*FDOH)=sxy(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxy(lidz,lidx+FDOH)=sxy(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxy(lidz,lidx-lsizex+3*FDOH)=sxy(gidz,gidx-lsizex+3*FDOH);
        BARRIER
#endif
        
#if   FDOH ==1
        sxy_x = HC1*(lsxy(lidz,lidx)   - lsxy(lidz,lidx-1));
#elif FDOH ==2
        sxy_x = (HC1*(lsxy(lidz,lidx)   - lsxy(lidz,lidx-1))
                      +HC2*(lsxy(lidz,lidx+1) - lsxy(lidz,lidx-2)));
#elif FDOH ==3
        sxy_x = (HC1*(lsxy(lidz,lidx)  -lsxy(lidz,lidx-1))+
                      HC2*(lsxy(lidz,lidx+1)-lsxy(lidz,lidx-2))+
                      HC3*(lsxy(lidz,lidx+2)-lsxy(lidz,lidx-3)));
#elif FDOH ==4
        sxy_x = (HC1*(lsxy(lidz,lidx)  -lsxy(lidz,lidx-1))+
                      HC2*(lsxy(lidz,lidx+1)-lsxy(lidz,lidx-2))+
                      HC3*(lsxy(lidz,lidx+2)-lsxy(lidz,lidx-3))+
                      HC4*(lsxy(lidz,lidx+3)-lsxy(lidz,lidx-4)));
#elif FDOH ==5
        sxy_x = (HC1*(lsxy(lidz,lidx)  -lsxy(lidz,lidx-1))+
                      HC2*(lsxy(lidz,lidx+1)-lsxy(lidz,lidx-2))+
                      HC3*(lsxy(lidz,lidx+2)-lsxy(lidz,lidx-3))+
                      HC4*(lsxy(lidz,lidx+3)-lsxy(lidz,lidx-4))+
                      HC5*(lsxy(lidz,lidx+4)-lsxy(lidz,lidx-5)));
        
#elif FDOH ==6
        sxy_x = (HC1*(lsxy(lidz,lidx)  -lsxy(lidz,lidx-1))+
                      HC2*(lsxy(lidz,lidx+1)-lsxy(lidz,lidx-2))+
                      HC3*(lsxy(lidz,lidx+2)-lsxy(lidz,lidx-3))+
                      HC4*(lsxy(lidz,lidx+3)-lsxy(lidz,lidx-4))+
                      HC5*(lsxy(lidz,lidx+4)-lsxy(lidz,lidx-5))+
                      HC6*(lsxy(lidz,lidx+5)-lsxy(lidz,lidx-6)));
#endif
        
#if LOCAL_OFF==0
        BARRIER
        lsyz(lidz,lidx)=syz(gidz,gidx);
        if (lidz<2*FDOH)
            lsyz(lidz-FDOH,lidx)=syz(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lsyz(lidz+FDOH,lidx)=syz(gidz+FDOH,gidx);
        BARRIER
#endif
        
#if   FDOH ==1
        syz_z = HC1*(lsyz(lidz,lidx)   - lsyz(lidz-1,lidx));
#elif FDOH ==2
        syz_z = (HC1*(lsyz(lidz,lidx)   - lsyz(lidz-1,lidx))
                      +HC2*(lsyz(lidz+1,lidx) - lsyz(lidz-2,lidx)));
#elif FDOH ==3
        syz_z = (HC1*(lsyz(lidz,lidx)  -lsyz(lidz-1,lidx))+
                      HC2*(lsyz(lidz+1,lidx)-lsyz(lidz-2,lidx))+
                      HC3*(lsyz(lidz+2,lidx)-lsyz(lidz-3,lidx)));
#elif FDOH ==4
        syz_z = (HC1*(lsyz(lidz,lidx)  -lsyz(lidz-1,lidx))+
                      HC2*(lsyz(lidz+1,lidx)-lsyz(lidz-2,lidx))+
                      HC3*(lsyz(lidz+2,lidx)-lsyz(lidz-3,lidx))+
                      HC4*(lsyz(lidz+3,lidx)-lsyz(lidz-4,lidx)));
#elif FDOH ==5
        syz_z = (HC1*(lsyz(lidz,lidx)  -lsyz(lidz-1,lidx))+
                      HC2*(lsyz(lidz+1,lidx)-lsyz(lidz-2,lidx))+
                      HC3*(lsyz(lidz+2,lidx)-lsyz(lidz-3,lidx))+
                      HC4*(lsyz(lidz+3,lidx)-lsyz(lidz-4,lidx))+
                      HC5*(lsyz(lidz+4,lidx)-lsyz(lidz-5,lidx)));
#elif FDOH ==6
        syz_z = (HC1*(lsyz(lidz,lidx)  -lsyz(lidz-1,lidx))+
                      HC2*(lsyz(lidz+1,lidx)-lsyz(lidz-2,lidx))+
                      HC3*(lsyz(lidz+2,lidx)-lsyz(lidz-3,lidx))+
                      HC4*(lsyz(lidz+3,lidx)-lsyz(lidz-4,lidx))+
                      HC5*(lsyz(lidz+4,lidx)-lsyz(lidz-5,lidx))+
                      HC6*(lsyz(lidz+5,lidx)-lsyz(lidz-6,lidx)));
#endif
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
        int i,k, ind;
        
        if (gidz>NZ-NAB-FDOH-1){
            
            i =gidx-FDOH;
            k =gidz - NZ+NAB+FDOH+NAB;
            ind=2*NAB-1-k;
            
            psi_syz_z(k,i) = b_z[ind+1] * psi_syz_z(k,i) + a_z[ind+1] * syz_z;
            syz_z = syz_z / K_z[ind+1] + psi_syz_z(k,i);
        }
        
#if FREESURF==0
        else if (gidz-FDOH<NAB){
            
            i =gidx-FDOH;
            k =gidz-FDOH;
            
            psi_syz_z(k,i) = b_z[k] * psi_syz_z(k,i) + a_z[k] * syz_z;
            syz_z = syz_z / K_z[k] + psi_syz_z(k,i);
        }
#endif
        
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            
            i =gidx-FDOH;
            k =gidz-FDOH;
            
            psi_sxy_x(k,i) = b_x[i] * psi_sxy_x(k,i) + a_x[i] * sxy_x;
            sxy_x = sxy_x / K_x[i] + psi_sxy_x(k,i);
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            
            i =gidx - NX+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-i;
            
            psi_sxy_x(k,i) = b_x[ind] * psi_sxy_x(k,i) + a_x[ind] * sxy_x;
            sxy_x = sxy_x / K_x[ind] + psi_sxy_x(k,i);
            
        }
#endif
    }
#endif

// Update the velocities
    {
        vy(gidz,gidx)+= ((sxy_x + syz_z)*rho(gidz,gidx));
    }

// Absorbing boundary    
#if ABS_TYPE==2
    {
#if FREESURF==0
        if (gidz-FDOH<NAB){
            vy(gidz,gidx)*=taper[gidz-FDOH];
        }
#endif
        
        if (gidz>NZ-NAB-FDOH-1){
            vy(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
        }
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            vy(gidz,gidx)*=taper[gidx-FDOH];
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            vy(gidz,gidx)*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif
    
    
}


