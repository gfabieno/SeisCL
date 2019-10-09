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

/*Update of the velocity in 2D SV*/

//Define useful macros to be able to write a matrix formulation in 2D with OpenCl 
#define lbnd (FDOH+NAB)

#define rho(z,x)    rho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,x)    rip[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rjp(z,x)    rjp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,x)    rkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define muipkp(z,x) muipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define mu(z,x)        mu[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define M(z,x)      M[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradrho(z,x)  gradrho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradM(z,x)  gradM[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradmu(z,x)  gradmu[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaup(z,x)  gradtaup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaus(z,x)  gradtaus[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define taus(z,x)        taus[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausipkp(z,x) tausipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define taup(z,x)        taup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define vx(z,x)  vx[(x)*(NZ)+(z)]
#define vy(z,x)  vy[(x)*(NZ)+(z)]
#define vz(z,x)  vz[(x)*(NZ)+(z)]
#define p(z,x) p[(x)*(NZ)+(z)]
#define p(z,x) p[(x)*(NZ)+(z)]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxy(z,x,l) rxy[(l)*NX*NZ+(x)*NZ+(z)]
#define ryz(z,x,l) ryz[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_p_x(z,x) psi_p_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_p_z(z,x) psi_p_z[(x)*(2*NAB)+(z)]

#if LOCAL_OFF==0

#define lvar(z,x)  lvar[(x)*lsizez+(z)]

#endif


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

FUNDEF void update_v(int offcomm,
                       GLOBARG float *vx,      GLOBARG float *vz,
                       GLOBARG float *p,
                       GLOBARG float *rip,     GLOBARG float *rkp,
                       GLOBARG float *taper,
                       GLOBARG float *K_z,        GLOBARG float *a_z,          GLOBARG float *b_z,
                       GLOBARG float *K_z_half,   GLOBARG float *a_z_half,     GLOBARG float *b_z_half,
                       GLOBARG float *K_x,        GLOBARG float *a_x,          GLOBARG float *b_x,
                       GLOBARG float *K_x_half,   GLOBARG float *a_x_half,     GLOBARG float *b_x_half,
                       GLOBARG float *psi_p_x,    GLOBARG float *psi_p_z,
                       LOCARG)
{

    LOCDEF
    float p_x;
    float p_z;

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
    
    #define lp lvar
    #define lp lvar
    
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
    
#define lp p
#define lp p
#define lidx gidx
#define lidz gidz
    
#endif

// Calculation of the stresses spatial derivatives
    {
#if LOCAL_OFF==0
        lp(lidz,lidx)=p(gidz, gidx);
        if (lidx<2*FDOH)
            lp(lidz,lidx-FDOH)=p(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lp(lidz,lidx+lsizex-3*FDOH)=p(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lp(lidz,lidx+FDOH)=p(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lp(lidz,lidx-lsizex+3*FDOH)=p(gidz,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lp(lidz-FDOH,lidx)=p(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lp(lidz+FDOH,lidx)=p(gidz+FDOH,gidx);
        BARRIER
#endif
        
#if   FDOH ==1
        p_x = HC1*(lp(lidz,lidx+1) - lp(lidz,lidx));
#elif FDOH ==2
        p_x = (HC1*(lp(lidz,lidx+1) - lp(lidz,lidx))
                      +HC2*(lp(lidz,lidx+2) - lp(lidz,lidx-1)));
#elif FDOH ==3
        p_x = (HC1*(lp(lidz,lidx+1)-lp(lidz,lidx))+
                      HC2*(lp(lidz,lidx+2)-lp(lidz,lidx-1))+
                      HC3*(lp(lidz,lidx+3)-lp(lidz,lidx-2)));
#elif FDOH ==4
        p_x = (HC1*(lp(lidz,lidx+1)-lp(lidz,lidx))+
                      HC2*(lp(lidz,lidx+2)-lp(lidz,lidx-1))+
                      HC3*(lp(lidz,lidx+3)-lp(lidz,lidx-2))+
                      HC4*(lp(lidz,lidx+4)-lp(lidz,lidx-3)));
#elif FDOH ==5
        p_x = (HC1*(lp(lidz,lidx+1)-lp(lidz,lidx))+
                      HC2*(lp(lidz,lidx+2)-lp(lidz,lidx-1))+
                      HC3*(lp(lidz,lidx+3)-lp(lidz,lidx-2))+
                      HC4*(lp(lidz,lidx+4)-lp(lidz,lidx-3))+
                      HC5*(lp(lidz,lidx+5)-lp(lidz,lidx-4)));
#elif FDOH ==6
        p_x = (HC1*(lp(lidz,lidx+1)-lp(lidz,lidx))+
                      HC2*(lp(lidz,lidx+2)-lp(lidz,lidx-1))+
                      HC3*(lp(lidz,lidx+3)-lp(lidz,lidx-2))+
                      HC4*(lp(lidz,lidx+4)-lp(lidz,lidx-3))+
                      HC5*(lp(lidz,lidx+5)-lp(lidz,lidx-4))+
                      HC6*(lp(lidz,lidx+6)-lp(lidz,lidx-5)));
#endif

#if   FDOH ==1
        p_z = HC1*(lp(lidz+1,lidx) - lp(lidz,lidx));
#elif FDOH ==2
        p_z = (HC1*(lp(lidz+1,lidx) - lp(lidz,lidx))
                      +HC2*(lp(lidz+2,lidx) - lp(lidz-1,lidx)));
#elif FDOH ==3
        p_z = (HC1*(lp(lidz+1,lidx)-lp(lidz,lidx))+
                      HC2*(lp(lidz+2,lidx)-lp(lidz-1,lidx))+
                      HC3*(lp(lidz+3,lidx)-lp(lidz-2,lidx)));
#elif FDOH ==4
        p_z = (HC1*(lp(lidz+1,lidx)-lp(lidz,lidx))+
                      HC2*(lp(lidz+2,lidx)-lp(lidz-1,lidx))+
                      HC3*(lp(lidz+3,lidx)-lp(lidz-2,lidx))+
                      HC4*(lp(lidz+4,lidx)-lp(lidz-3,lidx)));
#elif FDOH ==5
        p_z = (HC1*(lp(lidz+1,lidx)-lp(lidz,lidx))+
                      HC2*(lp(lidz+2,lidx)-lp(lidz-1,lidx))+
                      HC3*(lp(lidz+3,lidx)-lp(lidz-2,lidx))+
                      HC4*(lp(lidz+4,lidx)-lp(lidz-3,lidx))+
                      HC5*(lp(lidz+5,lidx)-lp(lidz-4,lidx)));
#elif FDOH ==6
        p_z = (HC1*(lp(lidz+1,lidx)-lp(lidz,lidx))+
                      HC2*(lp(lidz+2,lidx)-lp(lidz-1,lidx))+
                      HC3*(lp(lidz+3,lidx)-lp(lidz-2,lidx))+
                      HC4*(lp(lidz+4,lidx)-lp(lidz-3,lidx))+
                      HC5*(lp(lidz+5,lidx)-lp(lidz-4,lidx))+
                      HC6*(lp(lidz+6,lidx)-lp(lidz-5,lidx)));
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
        int i,k,ind;
        
        if (gidz>NZ-NAB-FDOH-1){
            
            i =gidx-FDOH;
            k =gidz - NZ+NAB+FDOH+NAB;
            ind=2*NAB-1-k;
            
            psi_p_z(k,i) = b_z_half[ind] * psi_p_z(k,i) + a_z_half[ind] * p_z;
            p_z = p_z / K_z_half[ind] + psi_p_z(k,i);
            
        }
        
#if FREESURF==0
        else if (gidz-FDOH<NAB){
            
            i =gidx-FDOH;
            k =gidz-FDOH;

            psi_p_z(k,i) = b_z_half[k] * psi_p_z(k,i) + a_z_half[k] * p_z;
            p_z = p_z / K_z_half[k] + psi_p_z(k,i);
            
        }
#endif
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            
            i =gidx-FDOH;
            k =gidz-FDOH;
            
            psi_p_x(k,i) = b_x_half[i] * psi_p_x(k,i) + a_x_half[i] * p_x;
            p_x = p_x / K_x_half[i] + psi_p_x(k,i);
            
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            
            i =gidx - NX+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-i;
            
            psi_p_x(k,i) = b_x_half[ind] * psi_p_x(k,i) + a_x_half[ind] * p_x;
            p_x = p_x / K_x_half[ind] + psi_p_x(k,i);
        }
#endif
    }
#endif

// Update the velocities
    {
        vx(gidz,gidx)+= (p_x*rip(gidz,gidx));
        vz(gidz,gidx)+= (p_z*rkp(gidz,gidx));
    }

// Absorbing boundary
#if ABS_TYPE==2
    {
        if (gidz-FDOH<NAB){
            vx(gidz,gidx)*=taper[gidz-FDOH];
            vz(gidz,gidx)*=taper[gidz-FDOH];
        }
        
        if (gidz>NZ-NAB-FDOH-1){
            vx(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
            vz(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
        }
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            vx(gidz,gidx)*=taper[gidx-FDOH];
            vz(gidz,gidx)*=taper[gidx-FDOH];
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            vx(gidz,gidx)*=taper[NX-FDOH-gidx-1];
            vz(gidz,gidx)*=taper[NX-FDOH-gidx-1];
        }
#endif 
    }
#endif
    
}

