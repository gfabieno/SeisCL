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

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#define lbnd (FDOH+NAB)

#define rho(z,x)    rho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,x)    rip[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,x)    rkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define uipkp(z,x) uipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define u(z,x)        u[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define pi(z,x)      pi[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradrho(z,x)  gradrho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradM(z,x)  gradM[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradmu(z,x)  gradmu[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaup(z,x)  gradtaup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaus(z,x)  gradtaus[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define Hrho(z,x)  Hrho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define HM(z,x)  HM[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define Hmu(z,x)  Hmu[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define Htaup(z,x)  Htaup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define Htaus(z,x)  Htaus[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define taus(z,x)        taus[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausipkp(z,x) tausipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define taup(z,x)        taup[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define vx(z,x)  vx[(x)*(NZ)+(z)]
#define vz(z,x)  vz[(x)*(NZ)+(z)]
#define sxx(z,x) sxx[(x)*(NZ)+(z)]
#define szz(z,x) szz[(x)*(NZ)+(z)]
#define sxz(z,x) sxz[(x)*(NZ)+(z)]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]

#define vxr(z,x)  vxr[(x)*(NZ)+(z)]
#define vzr(z,x)  vzr[(x)*(NZ)+(z)]
#define sxxr(z,x) sxxr[(x)*(NZ)+(z)]
#define szzr(z,x) szzr[(x)*(NZ)+(z)]
#define sxzr(z,x) sxzr[(x)*(NZ)+(z)]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_sxx_x(z,x) psi_sxx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_x(z,x) psi_sxz_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_z(z,x) psi_sxz_z[(x)*(2*NAB)+(z)]
#define psi_szz_z(z,x) psi_szz_z[(x)*(2*NAB)+(z)]


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
#define rec_pos(y,x) rec_pos[(y)*8+(x)]
#define gradsrc(y,x) gradsrc[(y)*NT+(x)]

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


// Find boundary indice for boundary injection in backpropagation
LFUNDEF int evarm( int k, int i){
    
    
#if NUM_DEVICES==1 & NLOCALP==1

    int NXbnd = (NX-2*FDOH-2*NAB);
    int NZbnd = (NZ-2*FDOH-2*NAB);

    int m=-1;
    i-=lbnd;
    k-=lbnd;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH)  && (i>FDOH-1 && i<NXbnd-FDOH) )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (i<FDOH){//front
        m=i*NZbnd+k;
    }
    else if (i>NXbnd-1-FDOH){//back
        i=i-NXbnd+FDOH;
        m=NZbnd*FDOH+i*NZbnd+k;
    }
    else if (k<FDOH){//up
        i=i-FDOH;
        m=NZbnd*FDOH*2+i+k*(NXbnd-2.0*FDOH);
    }
    else {//down
        i=i-FDOH;
        k=k-NZbnd+FDOH;
        m=NZbnd*FDOH*2+(NXbnd-2*FDOH)*FDOH+i+k*(NXbnd-2.0*FDOH);
    }
    
    

#elif DEVID==0 & MYGROUPID==0
    
    int NXbnd = (NX-2*FDOH-NAB);
    int NZbnd = (NZ-2*FDOH-2*NAB);
    
    int m=-1;
    i-=lbnd;
    k-=lbnd;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH)  && i>FDOH-1  )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (i<FDOH){//front
        m=i*NZbnd+k;
    }
    else if (k<FDOH){//up
        i=i-FDOH;
        m=NZbnd*FDOH+i+k*(NXbnd-FDOH);
    }
    else {//down
        i=i-FDOH;
        k=k-NZbnd+FDOH;
        m=NZbnd*FDOH+(NXbnd-FDOH)*FDOH+i+k*(NXbnd-FDOH);
    }

#elif DEVID==NUM_DEVICES-1 & MYGROUPID==NLOCALP-1
    int NXbnd = (NX-2*FDOH-NAB);
    int NZbnd = (NZ-2*FDOH-2*NAB);
    
    int m=-1;
    i-=FDOH;
    k-=lbnd;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH) && i<NXbnd-FDOH )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i>NXbnd-1 )
        m=-1;
    else if (i>NXbnd-1-FDOH){
        i=i-NXbnd+FDOH;
        m=i*NZbnd+k;
    }
    else if (k<FDOH){//up
        m=NZbnd*FDOH+i+k*(NXbnd-FDOH);
    }
    else {//down
        k=k-NZbnd+FDOH;
        m=NZbnd*FDOH+(NXbnd-FDOH)*FDOH+i+k*(NXbnd-FDOH);
    }
    
#else
    
    int NXbnd = (NX-2*FDOH);
    int NZbnd = (NZ-2*FDOH-2*NAB);
    
    int m=-1;
    i-=FDOH;
    k-=lbnd;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH) )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (k<FDOH){//up
        m=i+k*(NXbnd);
    }
    else {//down
        k=k-NZbnd+FDOH;
        m=(NXbnd)*FDOH+i+k*(NXbnd);
    }
    
    
#endif
    
    
    return m;
 
}


FUNDEF void update_adjv(int offcomm,
                          GLOBARG float*vx,         GLOBARG float*vz,
                          GLOBARG float*sxx,        GLOBARG float*szz,
                          GLOBARG float*sxz,
                          GLOBARG float*vxbnd,      GLOBARG float*vzbnd,
                          GLOBARG float*sxxbnd,     GLOBARG float*szzbnd,
                          GLOBARG float*sxzbnd,
                          GLOBARG float*vxr,       GLOBARG float*vzr,
                          GLOBARG float*sxxr,      GLOBARG float*szzr,
                          GLOBARG float*sxzr,
                          GLOBARG float*rip,        GLOBARG float*rkp,
                          GLOBARG float*taper,
                          GLOBARG float*K_x,        GLOBARG float*a_x,          GLOBARG float*b_x,
                          GLOBARG float*K_x_half,   GLOBARG float*a_x_half,     GLOBARG float*b_x_half,
                          GLOBARG float*K_z,        GLOBARG float*a_z,          GLOBARG float*b_z,
                          GLOBARG float*K_z_half,   GLOBARG float*a_z_half,     GLOBARG float*b_z_half,
                          GLOBARG float*psi_sxx_x,  GLOBARG float*psi_sxz_x,
                          GLOBARG float*psi_sxz_z,  GLOBARG float*psi_szz_z,
                          GLOBARG float*gradrho, GLOBARG float*gradsrc,
                          GLOBARG float*Hrho,    GLOBARG float*Hsrc,
                        LOCARG)
{

    LOCDEF
    
    int g,i,j,k,m;
    float sxx_xr;
    float szz_zr;
    float sxz_xr;
    float sxz_zr;
    float sxx_x;
    float szz_z;
    float sxz_x;
    float sxz_z;
    float lvx;
    float lvz;
    
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

#define lsxx lvar
#define lszz lvar
#define lsxz lvar
    
#define lsxxr lvar
#define lszzr lvar
#define lsxzr lvar
 
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
    
#define lsxx sxx
#define lszz szz
#define lsxz sxz
    
#define lsxxr sxxr
#define lszzr szzr
#define lsxzr sxzr
    
#define lidx gidx
#define lidz gidz
    
#define lsizez NZ
#define lsizex NX
    
#endif
    
// Calculation of the stress spatial derivatives of the forward wavefield if backpropagation is used
#if BACK_PROP_TYPE==1
    {
#if LOCAL_OFF==0
        lsxx(lidz,lidx)=sxx(gidz, gidx);
        if (lidx<2*FDOH)
            lsxx(lidz,lidx-FDOH)=sxx(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxx(lidz,lidx+lsizex-3*FDOH)=sxx(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxx(lidz,lidx+FDOH)=sxx(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxx(lidz,lidx-lsizex+3*FDOH)=sxx(gidz,gidx-lsizex+3*FDOH);
        BARRIER
#endif
        
#if   FDOH ==1
        sxx_x = HC1*(lsxx(lidz,lidx+1) - lsxx(lidz,lidx));
#elif FDOH ==2
        sxx_x = (HC1*(lsxx(lidz,lidx+1) - lsxx(lidz,lidx))
                      +HC2*(lsxx(lidz,lidx+2) - lsxx(lidz,lidx-1)));
#elif FDOH ==3
        sxx_x = (HC1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
                      HC2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
                      HC3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2)));
#elif FDOH ==4
        sxx_x = (HC1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
                      HC2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
                      HC3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2))+
                      HC4*(lsxx(lidz,lidx+4)-lsxx(lidz,lidx-3)));
#elif FDOH ==5
        sxx_x = (HC1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
                      HC2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
                      HC3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2))+
                      HC4*(lsxx(lidz,lidx+4)-lsxx(lidz,lidx-3))+
                      HC5*(lsxx(lidz,lidx+5)-lsxx(lidz,lidx-4)));
#elif FDOH ==6
        sxx_x = (HC1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
                      HC2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
                      HC3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2))+
                      HC4*(lsxx(lidz,lidx+4)-lsxx(lidz,lidx-3))+
                      HC5*(lsxx(lidz,lidx+5)-lsxx(lidz,lidx-4))+
                      HC6*(lsxx(lidz,lidx+6)-lsxx(lidz,lidx-5)));
#endif
        
        
#if LOCAL_OFF==0
        BARRIER
        lszz(lidz,lidx)=szz(gidz, gidx);
        if (lidz<2*FDOH)
            lszz(lidz-FDOH,lidx)=szz(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lszz(lidz+FDOH,lidx)=szz(gidz+FDOH,gidx);
        BARRIER
#endif
        
#if   FDOH ==1
        szz_z = HC1*(lszz(lidz+1,lidx) - lszz(lidz,lidx));
#elif FDOH ==2
        szz_z = (HC1*(lszz(lidz+1,lidx) - lszz(lidz,lidx))
                      +HC2*(lszz(lidz+2,lidx) - lszz(lidz-1,lidx)));
#elif FDOH ==3
        szz_z = (HC1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
                      HC2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
                      HC3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx)));
#elif FDOH ==4
        szz_z = (HC1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
                      HC2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
                      HC3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx))+
                      HC4*(lszz(lidz+4,lidx)-lszz(lidz-3,lidx)));
#elif FDOH ==5
        szz_z = (HC1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
                      HC2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
                      HC3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx))+
                      HC4*(lszz(lidz+4,lidx)-lszz(lidz-3,lidx))+
                      HC5*(lszz(lidz+5,lidx)-lszz(lidz-4,lidx)));
#elif FDOH ==6
        szz_z = (HC1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
                      HC2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
                      HC3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx))+
                      HC4*(lszz(lidz+4,lidx)-lszz(lidz-3,lidx))+
                      HC5*(lszz(lidz+5,lidx)-lszz(lidz-4,lidx))+
                      HC6*(lszz(lidz+6,lidx)-lszz(lidz-5,lidx)));
#endif
        
#if LOCAL_OFF==0
        BARRIER
        lsxz(lidz,lidx)=sxz(gidz, gidx);
        
        if (lidx<2*FDOH)
            lsxz(lidz,lidx-FDOH)=sxz(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lsxz(lidz,lidx+lsizex-3*FDOH)=sxz(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lsxz(lidz,lidx+FDOH)=sxz(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lsxz(lidz,lidx-lsizex+3*FDOH)=sxz(gidz,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lsxz(lidz-FDOH,lidx)=sxz(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lsxz(lidz+FDOH,lidx)=sxz(gidz+FDOH,gidx);
        BARRIER
#endif
        
#if   FDOH ==1
        sxz_z = HC1*(lsxz(lidz,lidx)   - lsxz(lidz-1,lidx));
        sxz_x = HC1*(lsxz(lidz,lidx)   - lsxz(lidz,lidx-1));
#elif FDOH ==2
        sxz_z = (HC1*(lsxz(lidz,lidx)   - lsxz(lidz-1,lidx))
                      +HC2*(lsxz(lidz+1,lidx) - lsxz(lidz-2,lidx)));
        sxz_x = (HC1*(lsxz(lidz,lidx)   - lsxz(lidz,lidx-1))
                      +HC2*(lsxz(lidz,lidx+1) - lsxz(lidz,lidx-2)));
#elif FDOH ==3
        sxz_z = (HC1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
                      HC2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
                      HC3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx)));
        
        sxz_x = (HC1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
                      HC2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
                      HC3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3)));
#elif FDOH ==4
        sxz_z = (HC1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
                      HC2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
                      HC3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx))+
                      HC4*(lsxz(lidz+3,lidx)-lsxz(lidz-4,lidx)));
        
        sxz_x = (HC1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
                      HC2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
                      HC3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3))+
                      HC4*(lsxz(lidz,lidx+3)-lsxz(lidz,lidx-4)));
#elif FDOH ==5
        sxz_z = (HC1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
                      HC2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
                      HC3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx))+
                      HC4*(lsxz(lidz+3,lidx)-lsxz(lidz-4,lidx))+
                      HC5*(lsxz(lidz+4,lidx)-lsxz(lidz-5,lidx)));
        
        sxz_x = (HC1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
                      HC2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
                      HC3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3))+
                      HC4*(lsxz(lidz,lidx+3)-lsxz(lidz,lidx-4))+
                      HC5*(lsxz(lidz,lidx+4)-lsxz(lidz,lidx-5)));
#elif FDOH ==6
        
        sxz_z = (HC1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
                      HC2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
                      HC3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx))+
                      HC4*(lsxz(lidz+3,lidx)-lsxz(lidz-4,lidx))+
                      HC5*(lsxz(lidz+4,lidx)-lsxz(lidz-5,lidx))+
                      HC6*(lsxz(lidz+5,lidx)-lsxz(lidz-6,lidx)));
        
        sxz_x = (HC1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
                      HC2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
                      HC3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3))+
                      HC4*(lsxz(lidz,lidx+3)-lsxz(lidz,lidx-4))+
                      HC5*(lsxz(lidz,lidx+4)-lsxz(lidz,lidx-5))+
                      HC6*(lsxz(lidz,lidx+5)-lsxz(lidz,lidx-6)));
#endif
        BARRIER
}
#endif

// Calculation of the stress spatial derivatives of the adjoint wavefield
#if LOCAL_OFF==0
    lsxxr(lidz,lidx)=sxxr(gidz, gidx);
    if (lidx<2*FDOH)
        lsxxr(lidz,lidx-FDOH)=sxxr(gidz,gidx-FDOH);
    if (lidx+lsizex-3*FDOH<FDOH)
        lsxxr(lidz,lidx+lsizex-3*FDOH)=sxxr(gidz,gidx+lsizex-3*FDOH);
    if (lidx>(lsizex-2*FDOH-1))
        lsxxr(lidz,lidx+FDOH)=sxxr(gidz,gidx+FDOH);
    if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
        lsxxr(lidz,lidx-lsizex+3*FDOH)=sxxr(gidz,gidx-lsizex+3*FDOH);
    BARRIER
#endif
    
#if   FDOH ==1
    sxx_xr = HC1*(lsxxr(lidz,lidx+1) - lsxxr(lidz,lidx));
#elif FDOH ==2
    sxx_xr = (HC1*(lsxxr(lidz,lidx+1) - lsxxr(lidz,lidx))
                  +HC2*(lsxxr(lidz,lidx+2) - lsxxr(lidz,lidx-1)));
#elif FDOH ==3
    sxx_xr = (HC1*(lsxxr(lidz,lidx+1)-lsxxr(lidz,lidx))+
                  HC2*(lsxxr(lidz,lidx+2)-lsxxr(lidz,lidx-1))+
                  HC3*(lsxxr(lidz,lidx+3)-lsxxr(lidz,lidx-2)));
#elif FDOH ==4
    sxx_xr = (HC1*(lsxxr(lidz,lidx+1)-lsxxr(lidz,lidx))+
                  HC2*(lsxxr(lidz,lidx+2)-lsxxr(lidz,lidx-1))+
                  HC3*(lsxxr(lidz,lidx+3)-lsxxr(lidz,lidx-2))+
                  HC4*(lsxxr(lidz,lidx+4)-lsxxr(lidz,lidx-3)));
#elif FDOH ==5
    sxx_xr = (HC1*(lsxxr(lidz,lidx+1)-lsxxr(lidz,lidx))+
                  HC2*(lsxxr(lidz,lidx+2)-lsxxr(lidz,lidx-1))+
                  HC3*(lsxxr(lidz,lidx+3)-lsxxr(lidz,lidx-2))+
                  HC4*(lsxxr(lidz,lidx+4)-lsxxr(lidz,lidx-3))+
                  HC5*(lsxxr(lidz,lidx+5)-lsxxr(lidz,lidx-4)));
#elif FDOH ==6
    sxx_xr = (HC1*(lsxxr(lidz,lidx+1)-lsxxr(lidz,lidx))+
                  HC2*(lsxxr(lidz,lidx+2)-lsxxr(lidz,lidx-1))+
                  HC3*(lsxxr(lidz,lidx+3)-lsxxr(lidz,lidx-2))+
                  HC4*(lsxxr(lidz,lidx+4)-lsxxr(lidz,lidx-3))+
                  HC5*(lsxxr(lidz,lidx+5)-lsxxr(lidz,lidx-4))+
                  HC6*(lsxxr(lidz,lidx+6)-lsxxr(lidz,lidx-5)));
#endif
    
    
#if LOCAL_OFF==0
    BARRIER
    lszzr(lidz,lidx)=szzr(gidz, gidx);
    if (lidz<2*FDOH)
        lszzr(lidz-FDOH,lidx)=szzr(gidz-FDOH,gidx);
    if (lidz>(lsizez-2*FDOH-1))
        lszzr(lidz+FDOH,lidx)=szzr(gidz+FDOH,gidx);
    BARRIER
#endif
    
#if   FDOH ==1
    szz_zr = HC1*(lszzr(lidz+1,lidx) - lszzr(lidz,lidx));
#elif FDOH ==2
    szz_zr = (HC1*(lszzr(lidz+1,lidx) - lszzr(lidz,lidx))
                  +HC2*(lszzr(lidz+2,lidx) - lszzr(lidz-1,lidx)));
#elif FDOH ==3
    szz_zr = (HC1*(lszzr(lidz+1,lidx)-lszzr(lidz,lidx))+
                  HC2*(lszzr(lidz+2,lidx)-lszzr(lidz-1,lidx))+
                  HC3*(lszzr(lidz+3,lidx)-lszzr(lidz-2,lidx)));
#elif FDOH ==4
    szz_zr = (HC1*(lszzr(lidz+1,lidx)-lszzr(lidz,lidx))+
                  HC2*(lszzr(lidz+2,lidx)-lszzr(lidz-1,lidx))+
                  HC3*(lszzr(lidz+3,lidx)-lszzr(lidz-2,lidx))+
                  HC4*(lszzr(lidz+4,lidx)-lszzr(lidz-3,lidx)));
#elif FDOH ==5
    szz_zr = (HC1*(lszzr(lidz+1,lidx)-lszzr(lidz,lidx))+
                  HC2*(lszzr(lidz+2,lidx)-lszzr(lidz-1,lidx))+
                  HC3*(lszzr(lidz+3,lidx)-lszzr(lidz-2,lidx))+
                  HC4*(lszzr(lidz+4,lidx)-lszzr(lidz-3,lidx))+
                  HC5*(lszzr(lidz+5,lidx)-lszzr(lidz-4,lidx)));
#elif FDOH ==6
    szz_zr = (HC1*(lszzr(lidz+1,lidx)-lszzr(lidz,lidx))+
                  HC2*(lszzr(lidz+2,lidx)-lszzr(lidz-1,lidx))+
                  HC3*(lszzr(lidz+3,lidx)-lszzr(lidz-2,lidx))+
                  HC4*(lszzr(lidz+4,lidx)-lszzr(lidz-3,lidx))+
                  HC5*(lszzr(lidz+5,lidx)-lszzr(lidz-4,lidx))+
                  HC6*(lszzr(lidz+6,lidx)-lszzr(lidz-5,lidx)));
#endif
    
#if LOCAL_OFF==0
    BARRIER
    lsxzr(lidz,lidx)=sxzr(gidz, gidx);
    
    if (lidx<2*FDOH)
        lsxzr(lidz,lidx-FDOH)=sxzr(gidz,gidx-FDOH);
    if (lidx+lsizex-3*FDOH<FDOH)
        lsxzr(lidz,lidx+lsizex-3*FDOH)=sxzr(gidz,gidx+lsizex-3*FDOH);
    if (lidx>(lsizex-2*FDOH-1))
        lsxzr(lidz,lidx+FDOH)=sxzr(gidz,gidx+FDOH);
    if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
        lsxzr(lidz,lidx-lsizex+3*FDOH)=sxzr(gidz,gidx-lsizex+3*FDOH);
    if (lidz<2*FDOH)
        lsxzr(lidz-FDOH,lidx)=sxzr(gidz-FDOH,gidx);
    if (lidz>(lsizez-2*FDOH-1))
        lsxzr(lidz+FDOH,lidx)=sxzr(gidz+FDOH,gidx);
    BARRIER
#endif
    
#if   FDOH ==1
    sxz_zr = HC1*(lsxzr(lidz,lidx)   - lsxzr(lidz-1,lidx));
    sxz_xr = HC1*(lsxzr(lidz,lidx)   - lsxzr(lidz,lidx-1));
#elif FDOH ==2
    sxz_zr = (HC1*(lsxzr(lidz,lidx)   - lsxzr(lidz-1,lidx))
                  +HC2*(lsxzr(lidz+1,lidx) - lsxzr(lidz-2,lidx)));
    sxz_xr = (HC1*(lsxzr(lidz,lidx)   - lsxzr(lidz,lidx-1))
                  +HC2*(lsxzr(lidz,lidx+1) - lsxzr(lidz,lidx-2)));
#elif FDOH ==3
    sxz_zr = (HC1*(lsxzr(lidz,lidx)  -lsxzr(lidz-1,lidx))+
                  HC2*(lsxzr(lidz+1,lidx)-lsxzr(lidz-2,lidx))+
                  HC3*(lsxzr(lidz+2,lidx)-lsxzr(lidz-3,lidx)));
    
    sxz_xr = (HC1*(lsxzr(lidz,lidx)  -lsxzr(lidz,lidx-1))+
                  HC2*(lsxzr(lidz,lidx+1)-lsxzr(lidz,lidx-2))+
                  HC3*(lsxzr(lidz,lidx+2)-lsxzr(lidz,lidx-3)));
#elif FDOH ==4
    sxz_zr = (HC1*(lsxzr(lidz,lidx)  -lsxzr(lidz-1,lidx))+
                  HC2*(lsxzr(lidz+1,lidx)-lsxzr(lidz-2,lidx))+
                  HC3*(lsxzr(lidz+2,lidx)-lsxzr(lidz-3,lidx))+
                  HC4*(lsxzr(lidz+3,lidx)-lsxzr(lidz-4,lidx)));
    
    sxz_xr = (HC1*(lsxzr(lidz,lidx)  -lsxzr(lidz,lidx-1))+
                  HC2*(lsxzr(lidz,lidx+1)-lsxzr(lidz,lidx-2))+
                  HC3*(lsxzr(lidz,lidx+2)-lsxzr(lidz,lidx-3))+
                  HC4*(lsxzr(lidz,lidx+3)-lsxzr(lidz,lidx-4)));
#elif FDOH ==5
    sxz_zr = (HC1*(lsxzr(lidz,lidx)  -lsxzr(lidz-1,lidx))+
                  HC2*(lsxzr(lidz+1,lidx)-lsxzr(lidz-2,lidx))+
                  HC3*(lsxzr(lidz+2,lidx)-lsxzr(lidz-3,lidx))+
                  HC4*(lsxzr(lidz+3,lidx)-lsxzr(lidz-4,lidx))+
                  HC5*(lsxzr(lidz+4,lidx)-lsxzr(lidz-5,lidx)));
    
    sxz_xr = (HC1*(lsxzr(lidz,lidx)  -lsxzr(lidz,lidx-1))+
                  HC2*(lsxzr(lidz,lidx+1)-lsxzr(lidz,lidx-2))+
                  HC3*(lsxzr(lidz,lidx+2)-lsxzr(lidz,lidx-3))+
                  HC4*(lsxzr(lidz,lidx+3)-lsxzr(lidz,lidx-4))+
                  HC5*(lsxzr(lidz,lidx+4)-lsxzr(lidz,lidx-5)));
#elif FDOH ==6
    
    sxz_zr = (HC1*(lsxzr(lidz,lidx)  -lsxzr(lidz-1,lidx))+
                  HC2*(lsxzr(lidz+1,lidx)-lsxzr(lidz-2,lidx))+
                  HC3*(lsxzr(lidz+2,lidx)-lsxzr(lidz-3,lidx))+
                  HC4*(lsxzr(lidz+3,lidx)-lsxzr(lidz-4,lidx))+
                  HC5*(lsxzr(lidz+4,lidx)-lsxzr(lidz-5,lidx))+
                  HC6*(lsxzr(lidz+5,lidx)-lsxzr(lidz-6,lidx)));
    
    sxz_xr = (HC1*(lsxzr(lidz,lidx)  -lsxzr(lidz,lidx-1))+
                  HC2*(lsxzr(lidz,lidx+1)-lsxzr(lidz,lidx-2))+
                  HC3*(lsxzr(lidz,lidx+2)-lsxzr(lidz,lidx-3))+
                  HC4*(lsxzr(lidz,lidx+3)-lsxzr(lidz,lidx-4))+
                  HC5*(lsxzr(lidz,lidx+4)-lsxzr(lidz,lidx-5))+
                  HC6*(lsxzr(lidz,lidx+5)-lsxzr(lidz,lidx-6)));
#endif

    
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


// Backpropagate the forward velocity
#if BACK_PROP_TYPE==1
    {

        lvx=((sxx_x + sxz_z)*rip(gidz,gidx));
        lvz=((szz_z + sxz_x)*rkp(gidz,gidx));
        vx(gidz,gidx)-= lvx;
        vz(gidz,gidx)-= lvz;
        
        // Inject the boundary values
        m=evarm(gidz,  gidx);
        if (m!=-1){
            vx(gidz, gidx)= vxbnd[m];
            vz(gidz, gidx)= vzbnd[m];
        }
    }
#endif

// Correct adjoint spatial derivatives to implement CPML
#if ABS_TYPE==1
    {
    int ind;
    
    if (gidz>NZ-NAB-FDOH-1){
        
        i =gidx-FDOH;
        k =gidz - NZ+NAB+FDOH+NAB;
        ind=2*NAB-1-k;
        
        psi_sxz_z(k,i) = b_z[ind+1] * psi_sxz_z(k,i) + a_z[ind+1] * sxz_zr;
        sxz_zr = sxz_zr / K_z[ind+1] + psi_sxz_z(k,i);
        psi_szz_z(k,i) = b_z_half[ind] * psi_szz_z(k,i) + a_z_half[ind] * szz_zr;
        szz_zr = szz_zr / K_z_half[ind] + psi_szz_z(k,i);
        
    }
    
#if FREESURF==0
    else if (gidz-FDOH<NAB){
        
        i =gidx-FDOH;
        k =gidz-FDOH;
        
        psi_sxz_z(k,i) = b_z[k] * psi_sxz_z(k,i) + a_z[k] * sxz_zr;
        sxz_zr = sxz_zr / K_z[k] + psi_sxz_z(k,i);
        psi_szz_z(k,i) = b_z_half[k] * psi_szz_z(k,i) + a_z_half[k] * szz_zr;
        szz_zr = szz_zr / K_z_half[k] + psi_szz_z(k,i);
        
    }
#endif
    
#if DEVID==0 & MYLOCALID==0
    if (gidx-FDOH<NAB){
        
        i =gidx-FDOH;
        k =gidz-FDOH;
        
        psi_sxx_x(k,i) = b_x_half[i] * psi_sxx_x(k,i) + a_x_half[i] * sxx_xr;
        sxx_xr = sxx_xr / K_x_half[i] + psi_sxx_x(k,i);
        psi_sxz_x(k,i) = b_x[i] * psi_sxz_x(k,i) + a_x[i] * sxz_xr;
        sxz_xr = sxz_xr / K_x[i] + psi_sxz_x(k,i);
        
    }
#endif
    
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-NAB-FDOH-1){
        
        i =gidx - NX+NAB+FDOH+NAB;
        k =gidz-FDOH;
        ind=2*NAB-1-i;
        
        psi_sxx_x(k,i) = b_x_half[ind] * psi_sxx_x(k,i) + a_x_half[ind] * sxx_xr;
        sxx_xr = sxx_xr / K_x_half[ind] + psi_sxx_x(k,i);
        psi_sxz_x(k,i) = b_x[ind+1] * psi_sxz_x(k,i) + a_x[ind+1] * sxz_xr;
        sxz_xr = sxz_xr / K_x[ind+1] + psi_sxz_x(k,i);
        
    }
#endif
    }
#endif
    
    // Update adjoint velocities
    lvx=((sxx_xr + sxz_zr)*rip(gidz,gidx));
    lvz=((szz_zr + sxz_xr)*rkp(gidz,gidx));
    vxr(gidz,gidx)+= lvx;
    vzr(gidz,gidx)+= lvz;
 
    

// Absorbing boundary
#if ABS_TYPE==2
    {
#if FREESURF==0
    if (gidz-FDOH<NAB){
        vxr(gidz,gidx)*=taper[gidz-FDOH];
        vzr(gidz,gidx)*=taper[gidz-FDOH];
    }
#endif
    
    if (gidz>NZ-NAB-FDOH-1){
        vxr(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
        vzr(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
    }
    
#if DEVID==0 & MYLOCALID==0
    if (gidx-FDOH<NAB){
        vxr(gidz,gidx)*=taper[gidx-FDOH];
        vzr(gidz,gidx)*=taper[gidx-FDOH];
    }
#endif
    
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-NAB-FDOH-1){
        vxr(gidz,gidx)*=taper[NX-FDOH-gidx-1];
        vzr(gidz,gidx)*=taper[NX-FDOH-gidx-1];
    }
#endif
    }
#endif
    
    
// Density gradient calculation on the fly
#if BACK_PROP_TYPE==1
    gradrho(gidz,gidx)+=-vx(gidz,gidx)*lvx-vz(gidz,gidx)*lvz;
    
//#if HOUT==1
//    Hrho(gidz,gidx)+= pown(vx(gidz,gidx),2)+pown(vz(gidz,gidx),2);
//#endif

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
//            int k=(int)(srcpos_loc(2,srci)/DH-0.5)+FDOH;
//            
//            if (i==gidx && k==gidz){
//                
//                int SOURCE_TYPE= (int)srcpos_loc(4,srci);
//                
//                if (SOURCE_TYPE==2){
//                    /* single force in x */
//                    gradsrc(srci,nt)+= vxr(gidz,gidx)/rip(gidx,gidz)/(DH*DH);
//                }
//                else if (SOURCE_TYPE==4){
//                    /* single force in z */
//                    
//                    gradsrc(srci,nt)+= vzr(gidz,gidx)/rkp(gidx,gidz)/(DH*DH);
//                }
//                
//            }
//        }
//        
//        
//    }
#endif
    
}

