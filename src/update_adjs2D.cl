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

/*Adjoint update of the stresses in 2D SV*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#define lbnd (FDOH+NAB)

#define rho(z,x)    rho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,x)    rip[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,x)    rkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define muipkp(z,x) muipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define mu(z,x)        mu[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define M(z,x)      M[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
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

#define vx_r(z,x)  vx_r[(x)*(NZ)+(z)]
#define vz_r(z,x)  vz_r[(x)*(NZ)+(z)]
#define sxx_r(z,x) sxx_r[(x)*(NZ)+(z)]
#define szz_r(z,x) szz_r[(x)*(NZ)+(z)]
#define sxz_r(z,x) sxz_r[(x)*(NZ)+(z)]


#define rxx_r(z,x,l) rxx_r[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz_r(z,x,l) rzz_r[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz_r(z,x,l) rxz_r[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_vx_x(z,x) psi_vx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_vz_x(z,x) psi_vz_x[(x)*(NZ-2*FDOH)+(z)]

#define psi_vx_z(z,x) psi_vx_z[(x)*(2*NAB)+(z)]
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
#define gradsrc(y,x) gradsrc[(y)*NT+(x)]




// Find boundary indice for boundary injection in backpropagation
extern "C" __device__ int evarm( int k, int i){
    
    
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
    i-=FDOH;;
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

extern "C" __global__ void update_adjs(int offcomm,
                          float *vx,         float *vz,       float *sxx,
                          float *szz,        float *sxz,      float *vxbnd,
                          float *vzbnd,      float *sxxbnd,   float *szzbnd,
                          float *sxzbnd,     float *vx_r,     float *vz_r,
                          float *sxx_r,      float *szz_r,    float *sxz_r,
                          float *rxx,        float *rzz,      float *rxz,
                          float *rxx_r,      float *rzz_r,    float *rxz_r,
                          float *M,         float *mu,        float *muipkp,
                          float *taus,       float *tausipkp, float *taup,
                          float *eta,        float *taper,
                          float *K_x,        float *a_x,      float *b_x,
                          float *K_x_half,   float *a_x_half, float *b_x_half,
                          float *K_z,        float *a_z,      float *b_z,
                          float *K_z_half,   float *a_z_half, float *b_z_half,
                          float *psi_vx_x,    float *psi_vx_z,
                          float *psi_vz_x,    float *psi_vz_z,
                          float *gradrho,    float *gradM,     float *gradmu,
                          float *gradtaup,   float *gradtaus,  float *gradsrc,
                          float *Hrho,    float *HM,     float *Hmu,
                          float *Htaup,   float *Htaus,  float *Hsrc)
{

    extern __shared__ float lvar[];
    
    int i,j,k,m;
    float vxx,vxz,vzx,vzz;
    float vxzzx,vxxzz;
    float vxx_r,vxz_r,vzx_r,vzz_r;
    float vxzzx_r,vxxzz_r;
    float lsxz, lsxx, lszz;
    float fipkp, f, g;
    float sumrxz, sumrxx, sumrzz;
    float b,c,e,d,dipkp;
    int l;
    float leta[LVE];
    float lM, lmu, lmuipkp, ltaup, ltaus, ltausipkp;
    
// If we use local memory
#if LOCAL_OFF==0
    int lsizez = blockDim.x+2*FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
    
#define lvx lvar
#define lvz lvar
#define lvx_r lvar
#define lvz_r lvar

// If local memory is turned off
#elif LOCAL_OFF==1
    
    int lsizez = blockDim.x+2*FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
    
#define lvx_r vx_r
#define lvz_r vz_r
#define lvx vx
#define lvz vz
#define lidx gidx
#define lidz gidz
    
#define lsizez NZ
#define lsizex NX
    
#endif
    
// Calculation of the velocity spatial derivatives of the forward wavefield if backpropagation is used
#if BACK_PROP_TYPE==1
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
        if (lidz<2*FDOH)
            lvx(lidz-FDOH,lidx)=vx(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvx(lidz+FDOH,lidx)=vx(gidz+FDOH,gidx);
        __syncthreads();
#endif
        
#if   FDOH==1
        vxx = HC1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))/DH;
        vxz = HC1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))/DH;
#elif FDOH==2
        vxx = (  HC1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + HC2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               )/DH;
        vxz = (  HC1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + HC2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               )/DH;
#elif FDOH==3
        vxx = (  HC1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + HC2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + HC3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               )/DH;
        vxz = (  HC1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + HC2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               + HC3*(lvx(lidz+3, lidx)-lvx(lidz-2, lidx))
               )/DH;
#elif FDOH==4
        vxx = (   HC1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + HC2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + HC3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               + HC4*(lvx(lidz, lidx+3)-lvx(lidz, lidx-4))
               )/DH;
        vxz = (  HC1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + HC2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               + HC3*(lvx(lidz+3, lidx)-lvx(lidz-2, lidx))
               + HC4*(lvx(lidz+4, lidx)-lvx(lidz-3, lidx))
               )/DH;
#elif FDOH==5
        vxx = (  HC1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + HC2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + HC3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               + HC4*(lvx(lidz, lidx+3)-lvx(lidz, lidx-4))
               + HC5*(lvx(lidz, lidx+4)-lvx(lidz, lidx-5))
               )/DH;
        vxz = (  HC1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + HC2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               + HC3*(lvx(lidz+3, lidx)-lvx(lidz-2, lidx))
               + HC4*(lvx(lidz+4, lidx)-lvx(lidz-3, lidx))
               + HC5*(lvx(lidz+5, lidx)-lvx(lidz-4, lidx))
               )/DH;
#elif FDOH==6
        vxx = (  HC1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + HC2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + HC3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               + HC4*(lvx(lidz, lidx+3)-lvx(lidz, lidx-4))
               + HC5*(lvx(lidz, lidx+4)-lvx(lidz, lidx-5))
               + HC6*(lvx(lidz, lidx+5)-lvx(lidz, lidx-6))
               )/DH;
        vxz = (  HC1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + HC2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               + HC3*(lvx(lidz+3, lidx)-lvx(lidz-2, lidx))
               + HC4*(lvx(lidz+4, lidx)-lvx(lidz-3, lidx))
               + HC5*(lvx(lidz+5, lidx)-lvx(lidz-4, lidx))
               + HC6*(lvx(lidz+6, lidx)-lvx(lidz-5, lidx))
               )/DH;
#endif
        
        
#if LOCAL_OFF==0
        __syncthreads();
        lvz(lidz,lidx)=vz(gidz, gidx);
        if (lidx<2*FDOH)
            lvz(lidz,lidx-FDOH)=vz(gidz,gidx-FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvz(lidz,lidx+FDOH)=vz(gidz,gidx+FDOH);
        if (lidz<2*FDOH)
            lvz(lidz-FDOH,lidx)=vz(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvz(lidz+FDOH,lidx)=vz(gidz+FDOH,gidx);
        __syncthreads();
#endif
        
#if   FDOH==1
        vzz = HC1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))/DH;
        vzx = HC1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))/DH;
#elif FDOH==2
        vzz = (  HC1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + HC2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               )/DH;
        vzx = (  HC1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + HC2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               )/DH;
#elif FDOH==3
        vzz = (  HC1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + HC2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + HC3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               )/DH;
        vzx = (  HC1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + HC2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               + HC3*(lvz(lidz, lidx+3)-lvz(lidz, lidx-2))
               )/DH;
#elif FDOH==4
        vzz = (  HC1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + HC2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + HC3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               + HC4*(lvz(lidz+3, lidx)-lvz(lidz-4, lidx))
               )/DH;
        vzx = (  HC1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + HC2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               + HC3*(lvz(lidz, lidx+3)-lvz(lidz, lidx-2))
               + HC4*(lvz(lidz, lidx+4)-lvz(lidz, lidx-3))
               )/DH;
#elif FDOH==5
        vzz = (  HC1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + HC2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + HC3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               + HC4*(lvz(lidz+3, lidx)-lvz(lidz-4, lidx))
               + HC5*(lvz(lidz+4, lidx)-lvz(lidz-5, lidx))
               )/DH;
        vzx = (  HC1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + HC2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               + HC3*(lvz(lidz, lidx+3)-lvz(lidz, lidx-2))
               + HC4*(lvz(lidz, lidx+4)-lvz(lidz, lidx-3))
               + HC5*(lvz(lidz, lidx+5)-lvz(lidz, lidx-4))
               )/DH;
#elif FDOH==6
        vzz = (  HC1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + HC2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + HC3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               + HC4*(lvz(lidz+3, lidx)-lvz(lidz-4, lidx))
               + HC5*(lvz(lidz+4, lidx)-lvz(lidz-5, lidx))
               + HC6*(lvz(lidz+5, lidx)-lvz(lidz-6, lidx))
               )/DH;
        vzx = (  HC1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + HC2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               + HC3*(lvz(lidz, lidx+3)-lvz(lidz, lidx-2))
               + HC4*(lvz(lidz, lidx+4)-lvz(lidz, lidx-3))
               + HC5*(lvz(lidz, lidx+5)-lvz(lidz, lidx-4))
               + HC6*(lvz(lidz, lidx+6)-lvz(lidz, lidx-5))
               )/DH;
#endif
        
        __syncthreads();
    }
#endif
    
// Calculation of the velocity spatial derivatives of the adjoint wavefield
    {
#if LOCAL_OFF==0
        lvx_r(lidz,lidx)=vx_r(gidz, gidx);
        if (lidx<2*FDOH)
            lvx_r(lidz,lidx-FDOH)=vx_r(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvx_r(lidz,lidx+lsizex-3*FDOH)=vx_r(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvx_r(lidz,lidx+FDOH)=vx_r(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvx_r(lidz,lidx-lsizex+3*FDOH)=vx_r(gidz,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvx_r(lidz-FDOH,lidx)=vx_r(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvx_r(lidz+FDOH,lidx)=vx_r(gidz+FDOH,gidx);
        __syncthreads();
#endif
    
#if   FDOH==1
    vxx_r = HC1*(lvx_r(lidz, lidx)  -lvx_r(lidz, lidx-1))/DH;
    vxz_r = HC1*(lvx_r(lidz+1, lidx)-lvx_r(lidz, lidx))/DH;
#elif FDOH==2
    vxx_r = (  HC1*(lvx_r(lidz, lidx)  -lvx_r(lidz, lidx-1))
           + HC2*(lvx_r(lidz, lidx+1)-lvx_r(lidz, lidx-2))
           )/DH;
    vxz_r = (  HC1*(lvx_r(lidz+1, lidx)-lvx_r(lidz, lidx))
           + HC2*(lvx_r(lidz+2, lidx)-lvx_r(lidz-1, lidx))
           )/DH;
#elif FDOH==3
    vxx_r = (  HC1*(lvx_r(lidz, lidx)  -lvx_r(lidz, lidx-1))
           + HC2*(lvx_r(lidz, lidx+1)-lvx_r(lidz, lidx-2))
           + HC3*(lvx_r(lidz, lidx+2)-lvx_r(lidz, lidx-3))
           )/DH;
    vxz_r = (  HC1*(lvx_r(lidz+1, lidx)-lvx_r(lidz, lidx))
           + HC2*(lvx_r(lidz+2, lidx)-lvx_r(lidz-1, lidx))
           + HC3*(lvx_r(lidz+3, lidx)-lvx_r(lidz-2, lidx))
           )/DH;
#elif FDOH==4
    vxx_r = (   HC1*(lvx_r(lidz, lidx)  -lvx_r(lidz, lidx-1))
           + HC2*(lvx_r(lidz, lidx+1)-lvx_r(lidz, lidx-2))
           + HC3*(lvx_r(lidz, lidx+2)-lvx_r(lidz, lidx-3))
           + HC4*(lvx_r(lidz, lidx+3)-lvx_r(lidz, lidx-4))
           )/DH;
    vxz_r = (  HC1*(lvx_r(lidz+1, lidx)-lvx_r(lidz, lidx))
           + HC2*(lvx_r(lidz+2, lidx)-lvx_r(lidz-1, lidx))
           + HC3*(lvx_r(lidz+3, lidx)-lvx_r(lidz-2, lidx))
           + HC4*(lvx_r(lidz+4, lidx)-lvx_r(lidz-3, lidx))
           )/DH;
#elif FDOH==5
    vxx_r = (  HC1*(lvx_r(lidz, lidx)  -lvx_r(lidz, lidx-1))
           + HC2*(lvx_r(lidz, lidx+1)-lvx_r(lidz, lidx-2))
           + HC3*(lvx_r(lidz, lidx+2)-lvx_r(lidz, lidx-3))
           + HC4*(lvx_r(lidz, lidx+3)-lvx_r(lidz, lidx-4))
           + HC5*(lvx_r(lidz, lidx+4)-lvx_r(lidz, lidx-5))
           )/DH;
    vxz_r = (  HC1*(lvx_r(lidz+1, lidx)-lvx_r(lidz, lidx))
           + HC2*(lvx_r(lidz+2, lidx)-lvx_r(lidz-1, lidx))
           + HC3*(lvx_r(lidz+3, lidx)-lvx_r(lidz-2, lidx))
           + HC4*(lvx_r(lidz+4, lidx)-lvx_r(lidz-3, lidx))
           + HC5*(lvx_r(lidz+5, lidx)-lvx_r(lidz-4, lidx))
           )/DH;
#elif FDOH==6
    vxx_r = (  HC1*(lvx_r(lidz, lidx)  -lvx_r(lidz, lidx-1))
           + HC2*(lvx_r(lidz, lidx+1)-lvx_r(lidz, lidx-2))
           + HC3*(lvx_r(lidz, lidx+2)-lvx_r(lidz, lidx-3))
           + HC4*(lvx_r(lidz, lidx+3)-lvx_r(lidz, lidx-4))
           + HC5*(lvx_r(lidz, lidx+4)-lvx_r(lidz, lidx-5))
           + HC6*(lvx_r(lidz, lidx+5)-lvx_r(lidz, lidx-6))
           )/DH;
    vxz_r = (  HC1*(lvx_r(lidz+1, lidx)-lvx_r(lidz, lidx))
           + HC2*(lvx_r(lidz+2, lidx)-lvx_r(lidz-1, lidx))
           + HC3*(lvx_r(lidz+3, lidx)-lvx_r(lidz-2, lidx))
           + HC4*(lvx_r(lidz+4, lidx)-lvx_r(lidz-3, lidx))
           + HC5*(lvx_r(lidz+5, lidx)-lvx_r(lidz-4, lidx))
           + HC6*(lvx_r(lidz+6, lidx)-lvx_r(lidz-5, lidx))
           )/DH;
#endif
    
    
#if LOCAL_OFF==0
    __syncthreads();
    lvz_r(lidz,lidx)=vz_r(gidz, gidx);
    if (lidx<2*FDOH)
        lvz_r(lidz,lidx-FDOH)=vz_r(gidz,gidx-FDOH);
    if (lidx>(lsizex-2*FDOH-1))
        lvz_r(lidz,lidx+FDOH)=vz_r(gidz,gidx+FDOH);
    if (lidz<2*FDOH)
        lvz_r(lidz-FDOH,lidx)=vz_r(gidz-FDOH,gidx);
    if (lidz>(lsizez-2*FDOH-1))
        lvz_r(lidz+FDOH,lidx)=vz_r(gidz+FDOH,gidx);
    __syncthreads();
#endif
    
#if   FDOH==1
    vzz_r = HC1*(lvz_r(lidz, lidx)  -lvz_r(lidz-1, lidx))/DH;
    vzx_r = HC1*(lvz_r(lidz, lidx+1)-lvz_r(lidz, lidx))/DH;
#elif FDOH==2
    vzz_r = (  HC1*(lvz_r(lidz, lidx)  -lvz_r(lidz-1, lidx))
           + HC2*(lvz_r(lidz+1, lidx)-lvz_r(lidz-2, lidx))
           )/DH;
    vzx_r = (  HC1*(lvz_r(lidz, lidx+1)-lvz_r(lidz, lidx))
           + HC2*(lvz_r(lidz, lidx+2)-lvz_r(lidz, lidx-1))
           )/DH;
#elif FDOH==3
    vzz_r = (  HC1*(lvz_r(lidz, lidx)  -lvz_r(lidz-1, lidx))
           + HC2*(lvz_r(lidz+1, lidx)-lvz_r(lidz-2, lidx))
           + HC3*(lvz_r(lidz+2, lidx)-lvz_r(lidz-3, lidx))
           )/DH;
    vzx_r = (  HC1*(lvz_r(lidz, lidx+1)-lvz_r(lidz, lidx))
           + HC2*(lvz_r(lidz, lidx+2)-lvz_r(lidz, lidx-1))
           + HC3*(lvz_r(lidz, lidx+3)-lvz_r(lidz, lidx-2))
           )/DH;
#elif FDOH==4
    vzz_r = (  HC1*(lvz_r(lidz, lidx)  -lvz_r(lidz-1, lidx))
           + HC2*(lvz_r(lidz+1, lidx)-lvz_r(lidz-2, lidx))
           + HC3*(lvz_r(lidz+2, lidx)-lvz_r(lidz-3, lidx))
           + HC4*(lvz_r(lidz+3, lidx)-lvz_r(lidz-4, lidx))
           )/DH;
    vzx_r = (  HC1*(lvz_r(lidz, lidx+1)-lvz_r(lidz, lidx))
           + HC2*(lvz_r(lidz, lidx+2)-lvz_r(lidz, lidx-1))
           + HC3*(lvz_r(lidz, lidx+3)-lvz_r(lidz, lidx-2))
           + HC4*(lvz_r(lidz, lidx+4)-lvz_r(lidz, lidx-3))
           )/DH;
#elif FDOH==5
    vzz_r = (  HC1*(lvz_r(lidz, lidx)  -lvz_r(lidz-1, lidx))
           + HC2*(lvz_r(lidz+1, lidx)-lvz_r(lidz-2, lidx))
           + HC3*(lvz_r(lidz+2, lidx)-lvz_r(lidz-3, lidx))
           + HC4*(lvz_r(lidz+3, lidx)-lvz_r(lidz-4, lidx))
           + HC5*(lvz_r(lidz+4, lidx)-lvz_r(lidz-5, lidx))
           )/DH;
    vzx_r = (  HC1*(lvz_r(lidz, lidx+1)-lvz_r(lidz, lidx))
           + HC2*(lvz_r(lidz, lidx+2)-lvz_r(lidz, lidx-1))
           + HC3*(lvz_r(lidz, lidx+3)-lvz_r(lidz, lidx-2))
           + HC4*(lvz_r(lidz, lidx+4)-lvz_r(lidz, lidx-3))
           + HC5*(lvz_r(lidz, lidx+5)-lvz_r(lidz, lidx-4))
           )/DH;
#elif FDOH==6
    vzz_r = (  HC1*(lvz_r(lidz, lidx)  -lvz_r(lidz-1, lidx))
           + HC2*(lvz_r(lidz+1, lidx)-lvz_r(lidz-2, lidx))
           + HC3*(lvz_r(lidz+2, lidx)-lvz_r(lidz-3, lidx))
           + HC4*(lvz_r(lidz+3, lidx)-lvz_r(lidz-4, lidx))
           + HC5*(lvz_r(lidz+4, lidx)-lvz_r(lidz-5, lidx))
           + HC6*(lvz_r(lidz+5, lidx)-lvz_r(lidz-6, lidx))
           )/DH;
    vzx_r = (  HC1*(lvz_r(lidz, lidx+1)-lvz_r(lidz, lidx))
           + HC2*(lvz_r(lidz, lidx+2)-lvz_r(lidz, lidx-1))
           + HC3*(lvz_r(lidz, lidx+3)-lvz_r(lidz, lidx-2))
           + HC4*(lvz_r(lidz, lidx+4)-lvz_r(lidz, lidx-3))
           + HC5*(lvz_r(lidz, lidx+5)-lvz_r(lidz, lidx-4))
           + HC6*(lvz_r(lidz, lidx+6)-lvz_r(lidz, lidx-5))
           )/DH;
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

    
// Read model parameters into local memory
#if LVE==0
    fipkp=muipkp(gidz, gidx)*DT;
    lmu=mu(gidz, gidx);
    lM=M(gidz, gidx);
    f=2.0*lmu*DT;
    g=lM*DT;
    
#else
    
    lM=M(gidz,gidx);
    lmu=mu(gidz,gidx);
    lmuipkp=muipkp(gidz,gidx);
    ltaup=taup(gidz,gidx);
    ltaus=taus(gidz,gidx);
    ltausipkp=tausipkp(gidz,gidx);
    
    for (l=0;l<LVE;l++){
        leta[l]=eta[l];
    }
    
    fipkp=lmuipkp*DT*(1.0+ (float)LVE*ltausipkp);
    g=lM*(1.0+(float)LVE*ltaup)*DT;
    f=2.0*lmu*(1.0+(float)LVE*ltaus)*DT;
    dipkp=lmuipkp*ltausipkp;
    d=2.0*lmu*ltaus;
    e=lM*ltaup;
    
#endif

    
// Backpropagate the forward stresses
#if BACK_PROP_TYPE==1
    {
#if LVE==0
    
    sxz(gidz, gidx)-=(fipkp*(vxz+vzx));
    sxx(gidz, gidx)-=(g*(vxx+vzz))-(f*vzz) ;
    szz(gidz, gidx)-=(g*(vxx+vzz))-(f*vxx) ;

// Backpropagation is not stable for viscoelastic wave equation
#else
    /* computing sums of the old memory variables */
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<LVE;l++){
        sumrxz+=rxz(gidz,gidx,l);
        sumrxx+=rxx(gidz,gidx,l);
        sumrzz+=rzz(gidz,gidx,l);
    }
    
    /* updating components of the stress tensor, partially */
    lsxz=(fipkp*(vxz+vzx))+(DT2*sumrxz);
    lsxx=((g*(vxx+vzz))-(f*vzz))+(DT2*sumrxx);
    lszz=((g*(vxx+vzz))-(f*vxx))+(DT2*sumrzz);
    
    
    /* now updating the memory-variables and sum them up*/
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<LVE;l++){

        b=1.0/(1.0-(leta[l]*0.5));
        c=1.0+(leta[l]*0.5);
        
        rxz(gidz,gidx,l)=b*(rxz(gidz,gidx,l)*c-leta[l]*(dipkp*(vxz+vzx)));
        rxx(gidz,gidx,l)=b*(rxx(gidz,gidx,l)*c-leta[l]*((e*(vxx+vzz))-(d*vzz)));
        rzz(gidz,gidx,l)=b*(rzz(gidz,gidx,l)*c-leta[l]*((e*(vxx+vzz))-(d*vxx)));
        
        sumrxz+=rxz(gidz,gidx,l);
        sumrxx+=rxx(gidz,gidx,l);
        sumrzz+=rzz(gidz,gidx,l);
    }
    
    /* and now the components of the stress tensor are
     completely updated */
    sxz(gidz, gidx)-= lsxz + (DT2*sumrxz);
    sxx(gidz, gidx)-= lsxx + (DT2*sumrxx) ;
    szz(gidz, gidx)-= lszz + (DT2*sumrzz) ;

    
#endif

    m=evarm(gidz,  gidx);
    if (m!=-1){
        sxx(gidz, gidx)= sxxbnd[m];
        szz(gidz, gidx)= szzbnd[m];
        sxz(gidz, gidx)= sxzbnd[m];
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
        
        psi_vx_z(k,i) = b_z_half[ind] * psi_vx_z(k,i) + a_z_half[ind] * vxz_r;
        vxz_r = vxz_r / K_z_half[ind] + psi_vx_z(k,i);
        psi_vz_z(k,i) = b_z[ind+1] * psi_vz_z(k,i) + a_z[ind+1] * vzz_r;
        vzz_r = vzz_r / K_z[ind+1] + psi_vz_z(k,i);
        
    }
    
#if FREESURF==0
    else if (gidz-FDOH<NAB){
        
        i =gidx-FDOH;
        k =gidz-FDOH;
        
        
        psi_vx_z(k,i) = b_z_half[k] * psi_vx_z(k,i) + a_z_half[k] * vxz_r;
        vxz_r = vxz_r / K_z_half[k] + psi_vx_z(k,i);
        psi_vz_z(k,i) = b_z[k] * psi_vz_z(k,i) + a_z[k] * vzz_r;
        vzz_r = vzz_r / K_z[k] + psi_vz_z(k,i);
        
        
    }
#endif
    
#if DEVID==0 & MYLOCALID==0
    if (gidx-FDOH<NAB){
        
        i =gidx-FDOH;
        k =gidz-FDOH;
        
        psi_vx_x(k,i) = b_x[i] * psi_vx_x(k,i) + a_x[i] * vxx_r;
        vxx_r = vxx_r / K_x[i] + psi_vx_x(k,i);
        psi_vz_x(k,i) = b_x_half[i] * psi_vz_x(k,i) + a_x_half[i] * vzx_r;
        vzx_r = vzx_r / K_x_half[i] + psi_vz_x(k,i);
        
    }
#endif
    
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-NAB-FDOH-1){
        
        i =gidx - NX+NAB+FDOH+NAB;
        k =gidz-FDOH;
        ind=2*NAB-1-i;
        
        
        psi_vx_x(k,i) = b_x[ind+1] * psi_vx_x(k,i) + a_x[ind+1] * vxx_r;
        vxx_r = vxx_r /K_x[ind+1] + psi_vx_x(k,i);
        psi_vz_x(k,i) = b_x_half[ind] * psi_vz_x(k,i) + a_x_half[ind] * vzx_r;
        vzx_r = vzx_r / K_x_half[ind]  +psi_vz_x(k,i);
        
        
    }
#endif
    }
#endif
    
// Update adjoint stresses
    {
#if LVE==0
    
        lsxz=(fipkp*(vxz_r+vzx_r));
        lsxx=((g*(vxx_r+vzz_r))-(f*vzz_r));
        lszz=((g*(vxx_r+vzz_r))-(f*vxx_r));
        
        sxz_r(gidz, gidx)+=lsxz;
        sxx_r(gidz, gidx)+=lsxx;
        szz_r(gidz, gidx)+=lszz;

#else

    /* computing sums of the old memory variables */
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<LVE;l++){
        sumrxz+=rxz_r(gidz,gidx,l);
        sumrxx+=rxx_r(gidz,gidx,l);
        sumrzz+=rzz_r(gidz,gidx,l);
    }
   
    /* updating components of the stress tensor, partially */
    lsxz=(fipkp*(vxz_r+vzx_r))+(DT2*sumrxz);
    lsxx=((g*(vxx_r+vzz_r))-(f*vzz_r))+(DT2*sumrxx);
    lszz=((g*(vxx_r+vzz_r))-(f*vxx_r))+(DT2*sumrzz);
    
    
    /* now updating the memory-variables and sum them up*/
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<LVE;l++){
        //those variables change sign in reverse time
        b=1.0/(1.0+(leta[l]*0.5));
        c=1.0-(leta[l]*0.5);
   
        
        rxz_r(gidz,gidx,l)=b*(rxz_r(gidz,gidx,l)*c-leta[l]*(dipkp*(vxz_r+vzx_r)));
        rxx_r(gidz,gidx,l)=b*(rxx_r(gidz,gidx,l)*c-leta[l]*((e*(vxx_r+vzz_r))-(d*vzz_r)));
        rzz_r(gidz,gidx,l)=b*(rzz_r(gidz,gidx,l)*c-leta[l]*((e*(vxx_r+vzz_r))-(d*vxx_r)));
        
        sumrxz+=rxz_r(gidz,gidx,l);
        sumrxx+=rxx_r(gidz,gidx,l);
        sumrzz+=rzz_r(gidz,gidx,l);
    }
    
    /* and now the components of the stress tensor are
     completely updated */
    sxz_r(gidz, gidx)+=lsxz + (DT2*sumrxz);
    sxx_r(gidz, gidx)+= lsxx + (DT2*sumrxx) ;
    szz_r(gidz, gidx)+= lszz + (DT2*sumrzz) ;
    
    
#endif
    }

// Absorbing boundary
#if ABS_TYPE==2
    {
    if (gidz-FDOH<NAB){
        sxz_r(gidz,gidx)*=taper[gidz-FDOH];
        sxx_r(gidz,gidx)*=taper[gidz-FDOH];
        szz_r(gidz,gidx)*=taper[gidz-FDOH];
    }
    
    if (gidz>NZ-NAB-FDOH-1){
        sxz_r(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
        sxx_r(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
        szz_r(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
    }

    
#if DEVID==0 & MYLOCALID==0
    if (gidx-FDOH<NAB){
        sxz_r(gidz,gidx)*=taper[gidx-FDOH];
        sxx_r(gidz,gidx)*=taper[gidx-FDOH];
        szz_r(gidz,gidx)*=taper[gidx-FDOH];
    }
#endif
    
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-NAB-FDOH-1){
        sxz_r(gidz,gidx)*=taper[NX-FDOH-gidx-1];
        sxx_r(gidz,gidx)*=taper[NX-FDOH-gidx-1];
        szz_r(gidz,gidx)*=taper[NX-FDOH-gidx-1];
    }
#endif
    }
#endif
    
// Shear wave modulus and P-wave modulus gradient calculation on the fly
#if BACK_PROP_TYPE==1
    float c1=1.0/pown(2.0*lM-2.0*lmu,2);
    float c3=1.0/pown(lmu,2);
    float c5=0.25*c3;
    
    float dM=c1*( sxx(gidz,gidx)+szz(gidz,gidx) )*( lsxx+lszz );
    
    gradM(gidz,gidx)+=-dM;
    gradmu(gidz,gidx)+=-c3*(sxz(gidz,gidx)*lsxz)+dM-c5*(  (sxx(gidz,gidx)-szz(gidz,gidx))*(lsxx-lszz)  );
    
#if HOUT==1
    float dMH=c1*pown( sxx(gidz,gidx)+szz(gidz,gidx),2);
    HM(gidz,gidx)+= dMH;
    Hmu(gidz,gidx)+=c3*pown(sxz(gidz,gidx),2)-dM+c5*pown(sxx(gidz,gidx)-szz(gidz,gidx),2) ;
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
//                int i=(int)(srcpos_loc(0,srci)/DH-0.5)+FDOH;
//                int k=(int)(srcpos_loc(2,srci)/DH-0.5)+FDOH;
//                
//                
//                if (i==gidx && k==gidz){
//                    
//                    pressure=( sxx_r(gidz,gidx)+szz_r(gidz,gidx) )/(2.0*DH*DH);
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

