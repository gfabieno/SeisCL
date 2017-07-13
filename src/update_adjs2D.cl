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
#define lbnd (fdoh+nab)

#define rho(z,x)    rho[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rip(z,x)    rip[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rkp(z,x)    rkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define uipkp(z,x) uipkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define u(z,x)        u[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define pi(z,x)      pi[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradrho(z,x)  gradrho[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradM(z,x)  gradM[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradmu(z,x)  gradmu[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradtaup(z,x)  gradtaup[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradtaus(z,x)  gradtaus[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define taus(z,x)        taus[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausipkp(z,x) tausipkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define taup(z,x)        taup[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define vx(z,x)  vx[(x)*NZ+(z)]
#define vz(z,x)  vz[(x)*NZ+(z)]
#define sxx(z,x) sxx[(x)*NZ+(z)]
#define szz(z,x) szz[(x)*NZ+(z)]
#define sxz(z,x) sxz[(x)*NZ+(z)]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]

#define vx_r(z,x)  vx_r[(x)*NZ+(z)]
#define vz_r(z,x)  vz_r[(x)*NZ+(z)]
#define sxx_r(z,x) sxx_r[(x)*NZ+(z)]
#define szz_r(z,x) szz_r[(x)*NZ+(z)]
#define sxz_r(z,x) sxz_r[(x)*NZ+(z)]


#define rxx_r(z,x,l) rxx_r[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz_r(z,x,l) rzz_r[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz_r(z,x,l) rxz_r[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_vxx(z,x) psi_vxx[(x)*(NZ-2*fdoh)+(z)]
#define psi_vzx(z,x) psi_vzx[(x)*(NZ-2*fdoh)+(z)]

#define psi_vxz(z,x) psi_vxz[(x)*(2*nab)+(z)]
#define psi_vzz(z,x) psi_vzz[(x)*(2*nab)+(z)]


#if local_off==0

#define lvar(z,x)  lvar[(x)*lsizez+(z)]

#endif


#define vxout(y,x) vxout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#define PI (3.141592653589793238462643383279502884197169)
#define srcpos_loc(y,x) srcpos_loc[(y)*nsrc+(x)]
#define signals(y,x) signals[(y)*NT+(x)]
#define gradsrc(y,x) gradsrc[(y)*NT+(x)]



// Find boundary indice for boundary injection in backpropagation
int evarm( int k, int i){
    
    
#if num_devices==1 & NLOCALP==1
    
    int NXbnd = (NX-2*fdoh-2*nab);
    int NZbnd = (NZ-2*fdoh-2*nab);
    
    int m=-1;
    i-=lbnd;
    k-=lbnd;
    
    if ( (k>fdoh-1 && k<NZbnd-fdoh)  && (i>fdoh-1 && i<NXbnd-fdoh) )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (i<fdoh){//front
        m=i*NZbnd+k;
    }
    else if (i>NXbnd-1-fdoh){//back
        i=i-NXbnd+fdoh;
        m=NZbnd*fdoh+i*NZbnd+k;
    }
    else if (k<fdoh){//up
        i=i-fdoh;
        m=NZbnd*fdoh*2+i+k*(NXbnd-2.0*fdoh);
    }
    else {//down
        i=i-fdoh;
        k=k-NZbnd+fdoh;
        m=NZbnd*fdoh*2+(NXbnd-2*fdoh)*fdoh+i+k*(NXbnd-2.0*fdoh);
    }
    
    
    
#elif dev==0 & MYGROUPID==0
    
    int NXbnd = (NX-2*fdoh-nab);
    int NZbnd = (NZ-2*fdoh-2*nab);
    
    int m=-1;
    i-=lbnd;
    k-=lbnd;
    
    if ( (k>fdoh-1 && k<NZbnd-fdoh)  && i>fdoh-1  )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (i<fdoh){//front
        m=i*NZbnd+k;
    }
    else if (k<fdoh){//up
        i=i-fdoh;
        m=NZbnd*fdoh+i+k*(NXbnd-fdoh);
    }
    else {//down
        i=i-fdoh;
        k=k-NZbnd+fdoh;
        m=NZbnd*fdoh+(NXbnd-fdoh)*fdoh+i+k*(NXbnd-fdoh);
    }
    
#elif dev==num_devices-1 & MYGROUPID==NLOCALP-1
    int NXbnd = (NX-2*fdoh-nab);
    int NZbnd = (NZ-2*fdoh-2*nab);
    
    int m=-1;
    i-=fdoh;
    k-=lbnd;
    
    if ( (k>fdoh-1 && k<NZbnd-fdoh) && i<NXbnd-fdoh )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i>NXbnd-1 )
        m=-1;
    else if (i>NXbnd-1-fdoh){
        i=i-NXbnd+fdoh;
        m=i*NZbnd+k;
    }
    else if (k<fdoh){//up
        m=NZbnd*fdoh+i+k*(NXbnd-fdoh);
    }
    else {//down
        k=k-NZbnd+fdoh;
        m=NZbnd*fdoh+(NXbnd-fdoh)*fdoh+i+k*(NXbnd-fdoh);
    }
    
#else
    
    int NXbnd = (NX-2*fdoh);
    int NZbnd = (NZ-2*fdoh-2*nab);
    
    int m=-1;
    i-=fdoh;;
    k-=lbnd;
    
    if ( (k>fdoh-1 && k<NZbnd-fdoh) )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (k<fdoh){//up
        m=i+k*(NXbnd);
    }
    else {//down
        k=k-NZbnd+fdoh;
        m=(NXbnd)*fdoh+i+k*(NXbnd);
    }
    
    
#endif
    
    
    return m;
    
}

__kernel void update_adjs(int offcomm, int nsrc,  int nt,
                          __global float *vx,         __global float *vz,       __global float *sxx,
                          __global float *szz,        __global float *sxz,      __global float *vxbnd,
                          __global float *vzbnd,      __global float *sxxbnd,   __global float *szzbnd,
                          __global float *sxzbnd,     __global float *vx_r,     __global float *vz_r,
                          __global float *sxx_r,      __global float *szz_r,    __global float *sxz_r,
                          __global float *rxx,        __global float *rzz,      __global float *rxz,
                          __global float *rxx_r,      __global float *rzz_r,    __global float *rxz_r,
                          __global float *pi,         __global float *u,        __global float *uipkp,
                          __global float *taus,       __global float *tausipkp, __global float *taup,
                          __global float *eta,
                          __global float *srcpos_loc, __global float *signals,  __global float *taper,
                          __global float *K_x,        __global float *a_x,      __global float *b_x,
                          __global float *K_x_half,   __global float *a_x_half, __global float *b_x_half,
                          __global float *K_z,        __global float *a_z,      __global float *b_z,
                          __global float *K_z_half,   __global float *a_z_half, __global float *b_z_half,
                          __global float *psi_vxx,    __global float *psi_vxz,
                          __global float *psi_vzx,    __global float *psi_vzz,
                          __global float *gradrho,    __global float *gradM,     __global float *gradmu,
                          __global float *gradtaup,   __global float *gradtaus,  __global float *gradsrc,
                          __local  float *lvar)
{

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
    float leta[Lve];
    float lpi, lu, luipkp, ltaup, ltaus, ltausipkp;
    
// If we use local memory
#if local_off==0
    int lsizez = get_local_size(0)+2*fdoh;
    int lsizex = get_local_size(1)+2*fdoh;
    int lidz = get_local_id(0)+fdoh;
    int lidx = get_local_id(1)+fdoh;
    int gidz = get_global_id(0)+fdoh;
    int gidx = get_global_id(1)+fdoh+offcomm;
    
#define lvx lvar
#define lvz lvar
#define lvx_r lvar
#define lvz_r lvar

// If local memory is turned off
#elif local_off==1
    
    int gid = get_global_id(0);
    int glsizez = (NZ-2*fdoh);
    int gidz = gid%glsizez+fdoh;
    int gidx = (gid/glsizez)+fdoh+offcomm;
    
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
#if back_prop_type==1
    {
#if local_off==0
        lvx(lidz,lidx)=vx(gidz, gidx);
        if (lidx<2*fdoh)
            lvx(lidz,lidx-fdoh)=vx(gidz,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvx(lidz,lidx+lsizex-3*fdoh)=vx(gidz,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvx(lidz,lidx+fdoh)=vx(gidz,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvx(lidz,lidx-lsizex+3*fdoh)=vx(gidz,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvx(lidz-fdoh,lidx)=vx(gidz-fdoh,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvx(lidz+fdoh,lidx)=vx(gidz+fdoh,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh==1
        vxx = hc1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))/DH;
        vxz = hc1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))/DH;
#elif fdoh==2
        vxx = (  hc1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + hc2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               )/DH;
        vxz = (  hc1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + hc2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               )/DH;
#elif fdoh==3
        vxx = (  hc1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + hc2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + hc3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               )/DH;
        vxz = (  hc1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + hc2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               + hc3*(lvx(lidz+3, lidx)-lvx(lidz-2, lidx))
               )/DH;
#elif fdoh==4
        vxx = (   hc1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + hc2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + hc3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               + hc4*(lvx(lidz, lidx+3)-lvx(lidz, lidx-4))
               )/DH;
        vxz = (  hc1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + hc2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               + hc3*(lvx(lidz+3, lidx)-lvx(lidz-2, lidx))
               + hc4*(lvx(lidz+4, lidx)-lvx(lidz-3, lidx))
               )/DH;
#elif fdoh==5
        vxx = (  hc1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + hc2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + hc3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               + hc4*(lvx(lidz, lidx+3)-lvx(lidz, lidx-4))
               + hc5*(lvx(lidz, lidx+4)-lvx(lidz, lidx-5))
               )/DH;
        vxz = (  hc1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + hc2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               + hc3*(lvx(lidz+3, lidx)-lvx(lidz-2, lidx))
               + hc4*(lvx(lidz+4, lidx)-lvx(lidz-3, lidx))
               + hc5*(lvx(lidz+5, lidx)-lvx(lidz-4, lidx))
               )/DH;
#elif fdoh==6
        vxx = (  hc1*(lvx(lidz, lidx)  -lvx(lidz, lidx-1))
               + hc2*(lvx(lidz, lidx+1)-lvx(lidz, lidx-2))
               + hc3*(lvx(lidz, lidx+2)-lvx(lidz, lidx-3))
               + hc4*(lvx(lidz, lidx+3)-lvx(lidz, lidx-4))
               + hc5*(lvx(lidz, lidx+4)-lvx(lidz, lidx-5))
               + hc6*(lvx(lidz, lidx+5)-lvx(lidz, lidx-6))
               )/DH;
        vxz = (  hc1*(lvx(lidz+1, lidx)-lvx(lidz, lidx))
               + hc2*(lvx(lidz+2, lidx)-lvx(lidz-1, lidx))
               + hc3*(lvx(lidz+3, lidx)-lvx(lidz-2, lidx))
               + hc4*(lvx(lidz+4, lidx)-lvx(lidz-3, lidx))
               + hc5*(lvx(lidz+5, lidx)-lvx(lidz-4, lidx))
               + hc6*(lvx(lidz+6, lidx)-lvx(lidz-5, lidx))
               )/DH;
#endif
        
        
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lvz(lidz,lidx)=vz(gidz, gidx);
        if (lidx<2*fdoh)
            lvz(lidz,lidx-fdoh)=vz(gidz,gidx-fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvz(lidz,lidx+fdoh)=vz(gidz,gidx+fdoh);
        if (lidz<2*fdoh)
            lvz(lidz-fdoh,lidx)=vz(gidz-fdoh,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvz(lidz+fdoh,lidx)=vz(gidz+fdoh,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh==1
        vzz = hc1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))/DH;
        vzx = hc1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))/DH;
#elif fdoh==2
        vzz = (  hc1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + hc2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               )/DH;
        vzx = (  hc1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + hc2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               )/DH;
#elif fdoh==3
        vzz = (  hc1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + hc2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + hc3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               )/DH;
        vzx = (  hc1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + hc2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               + hc3*(lvz(lidz, lidx+3)-lvz(lidz, lidx-2))
               )/DH;
#elif fdoh==4
        vzz = (  hc1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + hc2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + hc3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               + hc4*(lvz(lidz+3, lidx)-lvz(lidz-4, lidx))
               )/DH;
        vzx = (  hc1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + hc2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               + hc3*(lvz(lidz, lidx+3)-lvz(lidz, lidx-2))
               + hc4*(lvz(lidz, lidx+4)-lvz(lidz, lidx-3))
               )/DH;
#elif fdoh==5
        vzz = (  hc1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + hc2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + hc3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               + hc4*(lvz(lidz+3, lidx)-lvz(lidz-4, lidx))
               + hc5*(lvz(lidz+4, lidx)-lvz(lidz-5, lidx))
               )/DH;
        vzx = (  hc1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + hc2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               + hc3*(lvz(lidz, lidx+3)-lvz(lidz, lidx-2))
               + hc4*(lvz(lidz, lidx+4)-lvz(lidz, lidx-3))
               + hc5*(lvz(lidz, lidx+5)-lvz(lidz, lidx-4))
               )/DH;
#elif fdoh==6
        vzz = (  hc1*(lvz(lidz, lidx)  -lvz(lidz-1, lidx))
               + hc2*(lvz(lidz+1, lidx)-lvz(lidz-2, lidx))
               + hc3*(lvz(lidz+2, lidx)-lvz(lidz-3, lidx))
               + hc4*(lvz(lidz+3, lidx)-lvz(lidz-4, lidx))
               + hc5*(lvz(lidz+4, lidx)-lvz(lidz-5, lidx))
               + hc6*(lvz(lidz+5, lidx)-lvz(lidz-6, lidx))
               )/DH;
        vzx = (  hc1*(lvz(lidz, lidx+1)-lvz(lidz, lidx))
               + hc2*(lvz(lidz, lidx+2)-lvz(lidz, lidx-1))
               + hc3*(lvz(lidz, lidx+3)-lvz(lidz, lidx-2))
               + hc4*(lvz(lidz, lidx+4)-lvz(lidz, lidx-3))
               + hc5*(lvz(lidz, lidx+5)-lvz(lidz, lidx-4))
               + hc6*(lvz(lidz, lidx+6)-lvz(lidz, lidx-5))
               )/DH;
#endif
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif
    
// Calculation of the velocity spatial derivatives of the adjoint wavefield
    {
#if local_off==0
        lvx_r(lidz,lidx)=vx_r(gidz, gidx);
        if (lidx<2*fdoh)
            lvx_r(lidz,lidx-fdoh)=vx_r(gidz,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvx_r(lidz,lidx+lsizex-3*fdoh)=vx_r(gidz,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvx_r(lidz,lidx+fdoh)=vx_r(gidz,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvx_r(lidz,lidx-lsizex+3*fdoh)=vx_r(gidz,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvx_r(lidz-fdoh,lidx)=vx_r(gidz-fdoh,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvx_r(lidz+fdoh,lidx)=vx_r(gidz+fdoh,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   fdoh==1
    vxx_r = hc1*(lvx_r(lidz, lidx)  -lvx_r(lidz, lidx-1))/DH;
    vxz_r = hc1*(lvx_r(lidz+1, lidx)-lvx_r(lidz, lidx))/DH;
#elif fdoh==2
    vxx_r = (  hc1*(lvx_r(lidz, lidx)  -lvx_r(lidz, lidx-1))
           + hc2*(lvx_r(lidz, lidx+1)-lvx_r(lidz, lidx-2))
           )/DH;
    vxz_r = (  hc1*(lvx_r(lidz+1, lidx)-lvx_r(lidz, lidx))
           + hc2*(lvx_r(lidz+2, lidx)-lvx_r(lidz-1, lidx))
           )/DH;
#elif fdoh==3
    vxx_r = (  hc1*(lvx_r(lidz, lidx)  -lvx_r(lidz, lidx-1))
           + hc2*(lvx_r(lidz, lidx+1)-lvx_r(lidz, lidx-2))
           + hc3*(lvx_r(lidz, lidx+2)-lvx_r(lidz, lidx-3))
           )/DH;
    vxz_r = (  hc1*(lvx_r(lidz+1, lidx)-lvx_r(lidz, lidx))
           + hc2*(lvx_r(lidz+2, lidx)-lvx_r(lidz-1, lidx))
           + hc3*(lvx_r(lidz+3, lidx)-lvx_r(lidz-2, lidx))
           )/DH;
#elif fdoh==4
    vxx_r = (   hc1*(lvx_r(lidz, lidx)  -lvx_r(lidz, lidx-1))
           + hc2*(lvx_r(lidz, lidx+1)-lvx_r(lidz, lidx-2))
           + hc3*(lvx_r(lidz, lidx+2)-lvx_r(lidz, lidx-3))
           + hc4*(lvx_r(lidz, lidx+3)-lvx_r(lidz, lidx-4))
           )/DH;
    vxz_r = (  hc1*(lvx_r(lidz+1, lidx)-lvx_r(lidz, lidx))
           + hc2*(lvx_r(lidz+2, lidx)-lvx_r(lidz-1, lidx))
           + hc3*(lvx_r(lidz+3, lidx)-lvx_r(lidz-2, lidx))
           + hc4*(lvx_r(lidz+4, lidx)-lvx_r(lidz-3, lidx))
           )/DH;
#elif fdoh==5
    vxx_r = (  hc1*(lvx_r(lidz, lidx)  -lvx_r(lidz, lidx-1))
           + hc2*(lvx_r(lidz, lidx+1)-lvx_r(lidz, lidx-2))
           + hc3*(lvx_r(lidz, lidx+2)-lvx_r(lidz, lidx-3))
           + hc4*(lvx_r(lidz, lidx+3)-lvx_r(lidz, lidx-4))
           + hc5*(lvx_r(lidz, lidx+4)-lvx_r(lidz, lidx-5))
           )/DH;
    vxz_r = (  hc1*(lvx_r(lidz+1, lidx)-lvx_r(lidz, lidx))
           + hc2*(lvx_r(lidz+2, lidx)-lvx_r(lidz-1, lidx))
           + hc3*(lvx_r(lidz+3, lidx)-lvx_r(lidz-2, lidx))
           + hc4*(lvx_r(lidz+4, lidx)-lvx_r(lidz-3, lidx))
           + hc5*(lvx_r(lidz+5, lidx)-lvx_r(lidz-4, lidx))
           )/DH;
#elif fdoh==6
    vxx_r = (  hc1*(lvx_r(lidz, lidx)  -lvx_r(lidz, lidx-1))
           + hc2*(lvx_r(lidz, lidx+1)-lvx_r(lidz, lidx-2))
           + hc3*(lvx_r(lidz, lidx+2)-lvx_r(lidz, lidx-3))
           + hc4*(lvx_r(lidz, lidx+3)-lvx_r(lidz, lidx-4))
           + hc5*(lvx_r(lidz, lidx+4)-lvx_r(lidz, lidx-5))
           + hc6*(lvx_r(lidz, lidx+5)-lvx_r(lidz, lidx-6))
           )/DH;
    vxz_r = (  hc1*(lvx_r(lidz+1, lidx)-lvx_r(lidz, lidx))
           + hc2*(lvx_r(lidz+2, lidx)-lvx_r(lidz-1, lidx))
           + hc3*(lvx_r(lidz+3, lidx)-lvx_r(lidz-2, lidx))
           + hc4*(lvx_r(lidz+4, lidx)-lvx_r(lidz-3, lidx))
           + hc5*(lvx_r(lidz+5, lidx)-lvx_r(lidz-4, lidx))
           + hc6*(lvx_r(lidz+6, lidx)-lvx_r(lidz-5, lidx))
           )/DH;
#endif
    
    
#if local_off==0
    barrier(CLK_LOCAL_MEM_FENCE);
    lvz_r(lidz,lidx)=vz_r(gidz, gidx);
    if (lidx<2*fdoh)
        lvz_r(lidz,lidx-fdoh)=vz_r(gidz,gidx-fdoh);
    if (lidx>(lsizex-2*fdoh-1))
        lvz_r(lidz,lidx+fdoh)=vz_r(gidz,gidx+fdoh);
    if (lidz<2*fdoh)
        lvz_r(lidz-fdoh,lidx)=vz_r(gidz-fdoh,gidx);
    if (lidz>(lsizez-2*fdoh-1))
        lvz_r(lidz+fdoh,lidx)=vz_r(gidz+fdoh,gidx);
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
#if   fdoh==1
    vzz_r = hc1*(lvz_r(lidz, lidx)  -lvz_r(lidz-1, lidx))/DH;
    vzx_r = hc1*(lvz_r(lidz, lidx+1)-lvz_r(lidz, lidx))/DH;
#elif fdoh==2
    vzz_r = (  hc1*(lvz_r(lidz, lidx)  -lvz_r(lidz-1, lidx))
           + hc2*(lvz_r(lidz+1, lidx)-lvz_r(lidz-2, lidx))
           )/DH;
    vzx_r = (  hc1*(lvz_r(lidz, lidx+1)-lvz_r(lidz, lidx))
           + hc2*(lvz_r(lidz, lidx+2)-lvz_r(lidz, lidx-1))
           )/DH;
#elif fdoh==3
    vzz_r = (  hc1*(lvz_r(lidz, lidx)  -lvz_r(lidz-1, lidx))
           + hc2*(lvz_r(lidz+1, lidx)-lvz_r(lidz-2, lidx))
           + hc3*(lvz_r(lidz+2, lidx)-lvz_r(lidz-3, lidx))
           )/DH;
    vzx_r = (  hc1*(lvz_r(lidz, lidx+1)-lvz_r(lidz, lidx))
           + hc2*(lvz_r(lidz, lidx+2)-lvz_r(lidz, lidx-1))
           + hc3*(lvz_r(lidz, lidx+3)-lvz_r(lidz, lidx-2))
           )/DH;
#elif fdoh==4
    vzz_r = (  hc1*(lvz_r(lidz, lidx)  -lvz_r(lidz-1, lidx))
           + hc2*(lvz_r(lidz+1, lidx)-lvz_r(lidz-2, lidx))
           + hc3*(lvz_r(lidz+2, lidx)-lvz_r(lidz-3, lidx))
           + hc4*(lvz_r(lidz+3, lidx)-lvz_r(lidz-4, lidx))
           )/DH;
    vzx_r = (  hc1*(lvz_r(lidz, lidx+1)-lvz_r(lidz, lidx))
           + hc2*(lvz_r(lidz, lidx+2)-lvz_r(lidz, lidx-1))
           + hc3*(lvz_r(lidz, lidx+3)-lvz_r(lidz, lidx-2))
           + hc4*(lvz_r(lidz, lidx+4)-lvz_r(lidz, lidx-3))
           )/DH;
#elif fdoh==5
    vzz_r = (  hc1*(lvz_r(lidz, lidx)  -lvz_r(lidz-1, lidx))
           + hc2*(lvz_r(lidz+1, lidx)-lvz_r(lidz-2, lidx))
           + hc3*(lvz_r(lidz+2, lidx)-lvz_r(lidz-3, lidx))
           + hc4*(lvz_r(lidz+3, lidx)-lvz_r(lidz-4, lidx))
           + hc5*(lvz_r(lidz+4, lidx)-lvz_r(lidz-5, lidx))
           )/DH;
    vzx_r = (  hc1*(lvz_r(lidz, lidx+1)-lvz_r(lidz, lidx))
           + hc2*(lvz_r(lidz, lidx+2)-lvz_r(lidz, lidx-1))
           + hc3*(lvz_r(lidz, lidx+3)-lvz_r(lidz, lidx-2))
           + hc4*(lvz_r(lidz, lidx+4)-lvz_r(lidz, lidx-3))
           + hc5*(lvz_r(lidz, lidx+5)-lvz_r(lidz, lidx-4))
           )/DH;
#elif fdoh==6
    vzz_r = (  hc1*(lvz_r(lidz, lidx)  -lvz_r(lidz-1, lidx))
           + hc2*(lvz_r(lidz+1, lidx)-lvz_r(lidz-2, lidx))
           + hc3*(lvz_r(lidz+2, lidx)-lvz_r(lidz-3, lidx))
           + hc4*(lvz_r(lidz+3, lidx)-lvz_r(lidz-4, lidx))
           + hc5*(lvz_r(lidz+4, lidx)-lvz_r(lidz-5, lidx))
           + hc6*(lvz_r(lidz+5, lidx)-lvz_r(lidz-6, lidx))
           )/DH;
    vzx_r = (  hc1*(lvz_r(lidz, lidx+1)-lvz_r(lidz, lidx))
           + hc2*(lvz_r(lidz, lidx+2)-lvz_r(lidz, lidx-1))
           + hc3*(lvz_r(lidz, lidx+3)-lvz_r(lidz, lidx-2))
           + hc4*(lvz_r(lidz, lidx+4)-lvz_r(lidz, lidx-3))
           + hc5*(lvz_r(lidz, lidx+5)-lvz_r(lidz, lidx-4))
           + hc6*(lvz_r(lidz, lidx+6)-lvz_r(lidz, lidx-5))
           )/DH;
#endif
    
    }
    
// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
#if local_off==0
#if comm12==0
    if (gidz>(NZ-fdoh-1) || (gidx-offcomm)>(NX-fdoh-1-lcomm) ){
        return;
    }
    
#else
    if (gidz>(NZ-fdoh-1) ){
        return;
    }
#endif
#endif

    
// Read model parameters into local memory
#if Lve==0
    fipkp=uipkp(gidz, gidx)*DT;
    lu=u(gidz, gidx);
    lpi=pi(gidz, gidx);
    f=2.0*lu*DT;
    g=lpi*DT;
    
#else
    
    lpi=pi(gidz,gidx);
    lu=u(gidz,gidx);
    luipkp=uipkp(gidz,gidx);
    ltaup=taup(gidz,gidx);
    ltaus=taus(gidz,gidx);
    ltausipkp=tausipkp(gidz,gidx);
    
    for (l=0;l<Lve;l++){
        leta[l]=eta[l];
    }
    
    fipkp=luipkp*DT*(1.0+ (float)Lve*ltausipkp);
    g=lpi*(1.0+(float)Lve*ltaup)*DT;
    f=2.0*lu*(1.0+(float)Lve*ltaus)*DT;
    dipkp=luipkp*ltausipkp;
    d=2.0*lu*ltaus;
    e=lpi*ltaup;
    
#endif
    
    
// Backpropagate the forward stresses
#if back_prop_type==1
    {
#if Lve==0
    
    sxz(gidz, gidx)-=(fipkp*(vxz+vzx));
    sxx(gidz, gidx)-=(g*(vxx+vzz))-(f*vzz) ;
    szz(gidz, gidx)-=(g*(vxx+vzz))-(f*vxx) ;

// Backpropagation is not stable for viscoelastic wave equation
#else
    /* computing sums of the old memory variables */
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<Lve;l++){
        sumrxz+=rxz(gidz,gidx,l);
        sumrxx+=rxx(gidz,gidx,l);
        sumrzz+=rzz(gidz,gidx,l);
    }
    
    /* updating components of the stress tensor, partially */
    lsxz=(fipkp*(vxz+vzx))+(dt2*sumrxz);
    lsxx=((g*(vxx+vzz))-(f*vzz))+(dt2*sumrxx);
    lszz=((g*(vxx+vzz))-(f*vxx))+(dt2*sumrzz);
    
    
    /* now updating the memory-variables and sum them up*/
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<Lve;l++){

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
    sxz(gidz, gidx)-= lsxz + (dt2*sumrxz);
    sxx(gidz, gidx)-= lsxx + (dt2*sumrxx) ;
    szz(gidz, gidx)-= lszz + (dt2*sumrzz) ;

    
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
#if abs_type==1
    {
    int ind;
    
    if (gidz>NZ-nab-fdoh-1){
        
        i =gidx-fdoh;
        k =gidz - NZ+nab+fdoh+nab;
        ind=2*nab-1-k;
        
        psi_vxz(k,i) = b_z_half[ind] * psi_vxz(k,i) + a_z_half[ind] * vxz_r;
        vxz_r = vxz_r / K_z_half[ind] + psi_vxz(k,i);
        psi_vzz(k,i) = b_z[ind+1] * psi_vzz(k,i) + a_z[ind+1] * vzz_r;
        vzz_r = vzz_r / K_z[ind+1] + psi_vzz(k,i);
        
    }
    
#if freesurf==0
    else if (gidz-fdoh<nab){
        
        i =gidx-fdoh;
        k =gidz-fdoh;
        
        
        psi_vxz(k,i) = b_z_half[k] * psi_vxz(k,i) + a_z_half[k] * vxz_r;
        vxz_r = vxz_r / K_z_half[k] + psi_vxz(k,i);
        psi_vzz(k,i) = b_z[k] * psi_vzz(k,i) + a_z[k] * vzz_r;
        vzz_r = vzz_r / K_z[k] + psi_vzz(k,i);
        
        
    }
#endif
    
#if dev==0 & MYLOCALID==0
    if (gidx-fdoh<nab){
        
        i =gidx-fdoh;
        k =gidz-fdoh;
        
        psi_vxx(k,i) = b_x[i] * psi_vxx(k,i) + a_x[i] * vxx_r;
        vxx_r = vxx_r / K_x[i] + psi_vxx(k,i);
        psi_vzx(k,i) = b_x_half[i] * psi_vzx(k,i) + a_x_half[i] * vzx_r;
        vzx_r = vzx_r / K_x_half[i] + psi_vzx(k,i);
        
    }
#endif
    
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-nab-fdoh-1){
        
        i =gidx - NX+nab+fdoh+nab;
        k =gidz-fdoh;
        ind=2*nab-1-i;
        
        
        psi_vxx(k,i) = b_x[ind+1] * psi_vxx(k,i) + a_x[ind+1] * vxx_r;
        vxx_r = vxx_r /K_x[ind+1] + psi_vxx(k,i);
        psi_vzx(k,i) = b_x_half[ind] * psi_vzx(k,i) + a_x_half[ind] * vzx_r;
        vzx_r = vzx_r / K_x_half[ind]  +psi_vzx(k,i);
        
        
    }
#endif
    }
#endif
    
// Update adjoint stresses
    {
#if Lve==0
    
        lsxz=(fipkp*(vxz_r+vzx_r));
        lsxx=((g*(vxx_r+vzz_r))-(f*vzz_r));
        lszz=((g*(vxx_r+vzz_r))-(f*vxx_r));
        
        sxz_r(gidz, gidx)+=lsxz;
        sxx_r(gidz, gidx)+=lsxx;
        szz_r(gidz, gidx)+=lszz;

#else

    /* computing sums of the old memory variables */
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<Lve;l++){
        sumrxz+=rxz_r(gidz,gidx,l);
        sumrxx+=rxx_r(gidz,gidx,l);
        sumrzz+=rzz_r(gidz,gidx,l);
    }
   
    /* updating components of the stress tensor, partially */
    lsxz=(fipkp*(vxz_r+vzx_r))+(dt2*sumrxz);
    lsxx=((g*(vxx_r+vzz_r))-(f*vzz_r))+(dt2*sumrxx);
    lszz=((g*(vxx_r+vzz_r))-(f*vxx_r))+(dt2*sumrzz);
    
    
    /* now updating the memory-variables and sum them up*/
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<Lve;l++){
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
    sxz_r(gidz, gidx)+=lsxz + (dt2*sumrxz);
    sxx_r(gidz, gidx)+= lsxx + (dt2*sumrxx) ;
    szz_r(gidz, gidx)+= lszz + (dt2*sumrzz) ;
    
    
#endif
    }

// Absorbing boundary
#if abs_type==2
    {
    if (gidz-fdoh<nab){
        sxz_r(gidz,gidx)*=taper[gidz-fdoh];
        sxx_r(gidz,gidx)*=taper[gidz-fdoh];
        szz_r(gidz,gidx)*=taper[gidz-fdoh];
    }
    
    if (gidz>NZ-nab-fdoh-1){
        sxz_r(gidz,gidx)*=taper[NZ-fdoh-gidz-1];
        sxx_r(gidz,gidx)*=taper[NZ-fdoh-gidz-1];
        szz_r(gidz,gidx)*=taper[NZ-fdoh-gidz-1];
    }

    
#if dev==0 & MYLOCALID==0
    if (gidx-fdoh<nab){
        sxz_r(gidz,gidx)*=taper[gidx-fdoh];
        sxx_r(gidz,gidx)*=taper[gidx-fdoh];
        szz_r(gidz,gidx)*=taper[gidx-fdoh];
    }
#endif
    
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-nab-fdoh-1){
        sxz_r(gidz,gidx)*=taper[NX-fdoh-gidx-1];
        sxx_r(gidz,gidx)*=taper[NX-fdoh-gidx-1];
        szz_r(gidz,gidx)*=taper[NX-fdoh-gidx-1];
    }
#endif
    }
#endif
    
// Shear wave modulus and P-wave modulus gradient calculation on the fly
#if back_prop_type==1
    float c1=1.0/pown(2.0*lpi-2.0*lu,2);
    float c3=1.0/pown(lu,2);
    float c5=0.25*c3;
    
    float dM=c1*( sxx(gidz,gidx)+szz(gidz,gidx) )*( lsxx+lszz );
    
    gradM(gidz,gidx)+=dM;
    gradmu(gidz,gidx)+=c3*(sxz(gidz,gidx)*lsxz)-dM+c5*(  (sxx(gidz,gidx)-szz(gidz,gidx))*(lsxx-lszz)  );
#endif

#if gradsrcout==1
    float pressure;
    if (nsrc>0){
        
        for (int srci=0; srci<nsrc; srci++){
            
            int SOURCE_TYPE= (int)srcpos_loc(4,srci);
            
            if (SOURCE_TYPE==1){
                int i=(int)(srcpos_loc(0,srci)/DH-0.5)+fdoh;
                int k=(int)(srcpos_loc(2,srci)/DH-0.5)+fdoh;
                
                
                if (i==gidx && k==gidz){
                    
                    pressure=( sxx_r(gidz,gidx)+szz_r(gidz,gidx) )/(2.0*DH*DH);
                    if ( (nt>0) && (nt< NT ) ){
                        gradsrc(srci,nt+1)+=pressure;
                        gradsrc(srci,nt-1)-=pressure;
                    }
                    else if (nt==0)
                        gradsrc(srci,nt+1)+=pressure;
                    else if (nt==NT)
                        gradsrc(srci,nt-1)-=pressure;
                    
                }
            }
        }
    }
    
#endif

}

