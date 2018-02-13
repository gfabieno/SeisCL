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
#define lbnd (FDOH+NAB)

#define rho(z,x)    rho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,x)    rip[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rjp(z,x)    rjp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,x)    rkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define M(z,x)      M[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradrho(z,x)  gradrho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradM(z,x)  gradM[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaup(z,x)  gradtaup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define taup(z,x)        taup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define vx(z,x)  vx[(x)*(NZ)+(z)]
#define vz(z,x)  vz[(x)*(NZ)+(z)]
#define p(z,x) p[(x)*(NZ)+(z)]

#define rp(z,x,l) rp[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_vx_x(z,x) psi_vx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_vz_z(z,x) psi_vz_z[(x)*(2*NAB)+(z)]


#if LOCAL_OFF==0

#define lvar(z,x)  lvar[(x)*lsizez+(z)]

#endif


#define vxout(y,x) vxout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#define PI (3.141592653589793238462643383279502884197169)
#define signals(y,x) signals[(y)*NT+(x)]

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

FUNDEF void update_s(int offcomm,
                       GLOBARG float *vx,         GLOBARG float *vz,
                       GLOBARG float *p,
                       GLOBARG float *M,
                       GLOBARG float *rp,         GLOBARG float *taup,
                       GLOBARG float *eta,        GLOBARG float *taper,
                       GLOBARG float *K_x,        GLOBARG float *a_x,          GLOBARG float *b_x,
                       GLOBARG float *K_x_half,   GLOBARG float *a_x_half,     GLOBARG float *b_x_half,
                       GLOBARG float *K_z,        GLOBARG float *a_z,          GLOBARG float *b_z,
                       GLOBARG float *K_z_half,   GLOBARG float *a_z_half,     GLOBARG float *b_z_half,
                       GLOBARG float *psi_vx_x,   GLOBARG float *psi_vz_z,
                       LOCARG )
{
    
    LOCDEF
    
    float vxx, vzz;
    int i,k,l,ind;
    float sumrp;
    float b,c,e,g;
#if LVE>0
    float leta[LVE];
#endif
    float lM, ltaup;
    float lp;
    
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
#define lidx gidx
#define lidz gidz
    
#endif

// Calculation of the velocity spatial derivatives
    {
#if LOCAL_OFF==0
        lvx(lidz,lidx)=vx(gidz, gidx);
        if (lidx<2*FDOH)
            lvx(lidz,lidx-FDOH)=vx(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvx(lidz,lidx+lsizex-3*FDOH)=vx(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvx(lidz,lidx+FDOH)=vx(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvx(lidz,lidx-lsizex+3*FDOH)=vx(gidz,gidx-lsizex+3*FDOH);
        BARRIER
#endif
        
#if   FDOH==1
        vxx = HC1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1));
#elif FDOH==2
        vxx = (  HC1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + HC2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               );
#elif FDOH==3
        vxx = (  HC1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + HC2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + HC3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               );
#elif FDOH==4
        vxx = (   HC1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + HC2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + HC3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               + HC4*(lvx(lidz, lidx+3)-lvx(lidz, lidx-4))
               );

#elif FDOH==5
        vxx = (  HC1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + HC2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + HC3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               + HC4*(lvx(lidz, lidx+3)-lvx(lidz, lidx-4))
               + HC5*(lvx(lidz, lidx+4)-lvx(lidz, lidx-5))
               );
#elif FDOH==6
        vxx = (  HC1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + HC2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + HC3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               + HC4*(lvx(lidz, lidx+3)-lvx(lidz, lidx-4))
               + HC5*(lvx(lidz, lidx+4)-lvx(lidz, lidx-5))
               + HC6*(lvx(lidz, lidx+5)-lvx(lidz, lidx-6))
               );
#endif
        
        
#if LOCAL_OFF==0
        BARRIER
        lvz(lidz,lidx)=vz(gidz, gidx);
        if (lidz<2*FDOH)
            lvz(lidz-FDOH,lidx)=vz(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvz(lidz+FDOH,lidx)=vz(gidz+FDOH,gidx);
        BARRIER
#endif
        
#if   FDOH==1
        vzz = HC1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx));
#elif FDOH==2
        vzz = (  HC1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + HC2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               );
#elif FDOH==3
        vzz = (  HC1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + HC2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + HC3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               );
#elif FDOH==4
        vzz = (  HC1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + HC2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + HC3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               + HC4*(lvz(lidz+3, lidx)-lvz(lidz-4, lidx))
               );
#elif FDOH==5
        vzz = (  HC1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + HC2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + HC3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               + HC4*(lvz(lidz+3, lidx)-lvz(lidz-4, lidx))
               + HC5*(lvz(lidz+4, lidx)-lvz(lidz-5, lidx))
               );
#elif FDOH==6
        vzz = (  HC1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + HC2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + HC3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               + HC4*(lvz(lidz+3, lidx)-lvz(lidz-4, lidx))
               + HC5*(lvz(lidz+4, lidx)-lvz(lidz-5, lidx))
               + HC6*(lvz(lidz+5, lidx)-lvz(lidz-6, lidx))
               );
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
        
        if (gidz>NZ-NAB-FDOH-1){
            
            i =gidx-FDOH;
            k =gidz - NZ+NAB+FDOH+NAB;
            ind=2*NAB-1-k;
            
            psi_vz_z(k,i) = b_z[ind+1] * psi_vz_z(k,i) + a_z[ind+1] * vzz;
            vzz = vzz / K_z[ind+1] + psi_vz_z(k,i);
            
        }
        
#if FREESURF==0
        else if (gidz-FDOH<NAB){
            
            i =gidx-FDOH;
            k =gidz-FDOH;

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
            
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            
            i =gidx - NX+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-i;
            
            
            psi_vx_x(k,i) = b_x[ind+1] * psi_vx_x(k,i) + a_x[ind+1] * vxx;
            vxx = vxx /K_x[ind+1] + psi_vx_x(k,i);

        }
#endif
       }
#endif
    

// Read model parameters into local memory
    {
#if LVE==0
        g=M(gidz, gidx);
#else
        
        lM=M(gidz,gidx);
        ltaup=taup(gidz,gidx);
        
        for (l=0;l<LVE;l++){
            leta[l]=eta[l];
        }

        g=lM*(1.0+(float)LVE*ltaup);
        e=lM*ltaup;
        
#endif
    }
    
// Update the stresses
    {
#if LVE==0
        p(gidz, gidx)+=(g*(vxx+vzz));
#else
        
        
        /* computing sums of the old memory variables */
        sumrp=0;
        for (l=0;l<LVE;l++){
            sumrp+=rp(gidz,gidx,l);
        }
        
        
        /* updating components of the stress tensor, partially */
        lp=(g*(vxx+vzz))+(DT2*sumrp);
      
        
        /* now updating the memory-variables and sum them up*/
        sumrp=0;
        for (l=0;l<LVE;l++){
            b=1.0/(1.0+(leta[l]*0.5));
            c=1.0-(leta[l]*0.5);
            
            rp(gidz,gidx,l)=b*(rp(gidz,gidx,l)*c-leta[l]*e*(vxx+vzz));
            
            sumrp+=rp(gidz,gidx,l);
        }

        
        /* and now the components of the stress tensor are
         completely updated */
        p(gidz, gidx)+= lp + (DT2*sumrp) ;
        
#endif
    }

// Absorbing boundary
#if ABS_TYPE==2
    {
        if (gidz-FDOH<NAB){
            p(gidz,gidx)*=taper[gidz-FDOH];
        }
        
        if (gidz>NZ-NAB-FDOH-1){
            p(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
        }
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            p(gidz,gidx)*=taper[gidx-FDOH];
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            p(gidz,gidx)*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif
    
}

