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
#define rho(z,y,x)     rho[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rip(z,y,x)     rip[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rjp(z,y,x)     rjp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rkp(z,y,x)     rkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define uipjp(z,y,x) uipjp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define ujpkp(z,y,x) ujpkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define uipkp(z,y,x) uipkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define u(z,y,x)         u[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define pi(z,y,x)       pi[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradrho(z,y,x)   gradrho[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradM(z,y,x)   gradM[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradmu(z,y,x)   gradmu[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradtaup(z,y,x)   gradtaup[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradtaus(z,y,x)   gradtaus[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define taus(z,y,x)         taus[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausipjp(z,y,x) tausipjp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausjpkp(z,y,x) tausjpkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausipkp(z,y,x) tausipkp[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define taup(z,y,x)         taup[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define vx(z,y,x)   vx[(x)*NY*NZ+(y)*NZ+(z)]
#define vy(z,y,x)   vy[(x)*NY*NZ+(y)*NZ+(z)]
#define vz(z,y,x)   vz[(x)*NY*NZ+(y)*NZ+(z)]
#define sxx(z,y,x) sxx[(x)*NY*NZ+(y)*NZ+(z)]
#define syy(z,y,x) syy[(x)*NY*NZ+(y)*NZ+(z)]
#define szz(z,y,x) szz[(x)*NY*NZ+(y)*NZ+(z)]
#define sxy(z,y,x) sxy[(x)*NY*NZ+(y)*NZ+(z)]
#define syz(z,y,x) syz[(x)*NY*NZ+(y)*NZ+(z)]
#define sxz(z,y,x) sxz[(x)*NY*NZ+(y)*NZ+(z)]

#define rxx(z,y,x,l) rxx[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryy(z,y,x,l) ryy[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rzz(z,y,x,l) rzz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxy(z,y,x,l) rxy[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryz(z,y,x,l) ryz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxz(z,y,x,l) rxz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]

#define psi_vxx(z,y,x) psi_vxx[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_vyx(z,y,x) psi_vyx[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_vzx(z,y,x) psi_vzx[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]

#define psi_vxy(z,y,x) psi_vxy[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_vyy(z,y,x) psi_vyy[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_vzy(z,y,x) psi_vzy[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]

#define psi_vxz(z,y,x) psi_vxz[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]
#define psi_vyz(z,y,x) psi_vyz[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]
#define psi_vzz(z,y,x) psi_vzz[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]



#if local_off==0

#define lvar(z,y,x)   lvar[(x)*lsizey*lsizez+(y)*lsizez+(z)]

#endif



#define PI (3.141592653589793238462643383279502884197169)
#define srcpos_loc(y,x) srcpos_loc[(y)*nsrc+(x)]
#define signals(y,x) signals[(y)*NT+(x)]




__kernel void update_s(int offcomm, int nsrc,  int nt,
                       __global float *vx,         __global float *vy,            __global float *vz,
                       __global float *sxx,        __global float *syy,           __global float *szz,
                       __global float *sxy,        __global float *syz,           __global float *sxz,
                       __global float *pi,         __global float *u,             __global float *uipjp,
                       __global float *ujpkp,      __global float *uipkp,
                       __global float *rxx,        __global float *ryy,           __global float *rzz,
                       __global float *rxy,        __global float *ryz,           __global float *rxz,
                       __global float *taus,       __global float *tausipjp,      __global float *tausjpkp,
                       __global float *tausipkp,   __global float *taup,          __global float *eta,
                       __global float *srcpos_loc, __global float *signals,       __global float *taper,
                       __global float *K_x,        __global float *a_x,          __global float *b_x,
                       __global float *K_x_half,   __global float *a_x_half,     __global float *b_x_half,
                       __global float *K_y,        __global float *a_y,          __global float *b_y,
                       __global float *K_y_half,   __global float *a_y_half,     __global float *b_y_half,
                       __global float *K_z,        __global float *a_z,          __global float *b_z,
                       __global float *K_z_half,   __global float *a_z_half,     __global float *b_z_half,
                       __global float *psi_vxx,    __global float *psi_vxy,       __global float *psi_vxz,
                       __global float *psi_vyx,    __global float *psi_vyy,       __global float *psi_vyz,
                       __global float *psi_vzx,    __global float *psi_vzy,       __global float *psi_vzz,
                       __local  float *lvar)

{
    
    /* Standard staggered grid kernel, finite difference order of 4.  */
    
    int i,j,k,l,ind;
    float fipjp, fjpkp, fipkp, f, g;
    float sumrxy,sumryz,sumrxz,sumrxx,sumryy,sumrzz;
    float b,c,e,d,dipjp,djpkp,dipkp;
    float lpi, lu, luipjp, luipkp, lujpkp, ltaup, ltaus, ltausipjp, ltausipkp, ltausjpkp;
#if Lve>0
    float leta[Lve];
#endif
    float vxx,vxy,vxz,vyx,vyy,vyz,vzx,vzy,vzz;
    float vxyyx,vyzzy,vxzzx,vxxyyzz,vyyzz,vxxzz,vxxyy;
    
    float lsxx, lsyy, lszz, lsxy, lsxz, lsyz;
    
// If we use local memory
#if local_off==0
    
    int lsizez = get_local_size(0)+2*fdoh;
    int lsizey = get_local_size(1)+2*fdoh;
    int lsizex = get_local_size(2)+2*fdoh;
    int lidz = get_local_id(0)+fdoh;
    int lidy = get_local_id(1)+fdoh;
    int lidx = get_local_id(2)+fdoh;
    int gidz = get_global_id(0)+fdoh;
    int gidy = get_global_id(1)+fdoh;
    int gidx = get_global_id(2)+fdoh+offcomm;
    
#define lvx lvar
#define lvy lvar
#define lvz lvar

// If local memory is turned off
#elif local_off==1
    
    int gid = get_global_id(0);
    int glsizez = (NZ-2*fdoh);
    int glsizey = (NY-2*fdoh);
    int gidz = gid%glsizez+fdoh;
    int gidy = (gid/glsizez)%glsizey+fdoh;
    int gidx = gid/(glsizez*glsizey)+fdoh+offcomm;
    
#define lvx vx
#define lvy vy
#define lvz vz
#define lidx gidx
#define lidy gidy
#define lidz gidz
    
#endif
    
// Calculation of the velocity spatial derivatives
    {
#if local_off==0
        lvx(lidz,lidy,lidx)=vx(gidz, gidy, gidx);
        if (lidy<2*fdoh)
            lvx(lidz,lidy-fdoh,lidx)=vx(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lvx(lidz,lidy+lsizey-3*fdoh,lidx)=vx(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lvx(lidz,lidy+fdoh,lidx)=vx(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lvx(lidz,lidy-lsizey+3*fdoh,lidx)=vx(gidz,gidy-lsizey+3*fdoh,gidx);
        if (lidx<2*fdoh)
            lvx(lidz,lidy,lidx-fdoh)=vx(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvx(lidz,lidy,lidx+lsizex-3*fdoh)=vx(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvx(lidz,lidy,lidx+fdoh)=vx(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvx(lidz,lidy,lidx-lsizex+3*fdoh)=vx(gidz,gidy,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvx(lidz-fdoh,lidy,lidx)=vx(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvx(lidz+fdoh,lidy,lidx)=vx(gidz+fdoh,gidy,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh==1
        vxx = (lvx(lidz,lidy,lidx)-lvx(lidz,lidy,lidx-1))/DH;
        vxy = (lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))/DH;
        vxz = (lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))/DH;
#elif fdoh==2
        vxx = (hc1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
               hc2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2)))/DH;
        
        vxy = (hc1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
               hc2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx)))/DH;
        
        vxz = (hc1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
               hc2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx)))/DH;
#elif fdoh==3
        vxx = (hc1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
               hc2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
               hc3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3)))/DH;
        
        vxy = (hc1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
               hc2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
               hc3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx)))/DH;
        
        vxz = (hc1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
               hc2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
               hc3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx)))/DH;
#elif fdoh==4
        vxx = (hc1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
               hc2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
               hc3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3))+
               hc4*(lvx(lidz,lidy,lidx+3)-lvx(lidz,lidy,lidx-4)))/DH;
        
        vxy = (hc1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
               hc2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
               hc3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx))+
               hc4*(lvx(lidz,lidy+4,lidx)-lvx(lidz,lidy-3,lidx)))/DH;
        
        vxz = (hc1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
               hc2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
               hc3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx))+
               hc4*(lvx(lidz+4,lidy,lidx)-lvx(lidz-3,lidy,lidx)))/DH;
#elif fdoh==5
        vxx = (hc1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
               hc2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
               hc3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3))+
               hc4*(lvx(lidz,lidy,lidx+3)-lvx(lidz,lidy,lidx-4))+
               hc5*(lvx(lidz,lidy,lidx+4)-lvx(lidz,lidy,lidx-5)))/DH;
        
        vxy = (hc1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
               hc2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
               hc3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx))+
               hc4*(lvx(lidz,lidy+4,lidx)-lvx(lidz,lidy-3,lidx))+
               hc5*(lvx(lidz,lidy+5,lidx)-lvx(lidz,lidy-4,lidx)))/DH;
        
        vxz = (hc1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
               hc2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
               hc3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx))+
               hc4*(lvx(lidz+4,lidy,lidx)-lvx(lidz-3,lidy,lidx))+
               hc5*(lvx(lidz+5,lidy,lidx)-lvx(lidz-4,lidy,lidx)))/DH;
#elif fdoh==6
        vxx = (hc1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
               hc2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
               hc3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3))+
               hc4*(lvx(lidz,lidy,lidx+3)-lvx(lidz,lidy,lidx-4))+
               hc5*(lvx(lidz,lidy,lidx+4)-lvx(lidz,lidy,lidx-5))+
               hc6*(lvx(lidz,lidy,lidx+5)-lvx(lidz,lidy,lidx-6)))/DH;
        
        vxy = (hc1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
               hc2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
               hc3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx))+
               hc4*(lvx(lidz,lidy+4,lidx)-lvx(lidz,lidy-3,lidx))+
               hc5*(lvx(lidz,lidy+5,lidx)-lvx(lidz,lidy-4,lidx))+
               hc6*(lvx(lidz,lidy+6,lidx)-lvx(lidz,lidy-5,lidx)))/DH;
        
        vxz = (hc1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
               hc2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
               hc3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx))+
               hc4*(lvx(lidz+4,lidy,lidx)-lvx(lidz-3,lidy,lidx))+
               hc5*(lvx(lidz+5,lidy,lidx)-lvx(lidz-4,lidy,lidx))+
               hc6*(lvx(lidz+6,lidy,lidx)-lvx(lidz-5,lidy,lidx)))/DH;
#endif
        
        
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lvy(lidz,lidy,lidx)=vy(gidz, gidy, gidx);
        if (lidy<2*fdoh)
            lvy(lidz,lidy-fdoh,lidx)=vy(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lvy(lidz,lidy+lsizey-3*fdoh,lidx)=vy(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lvy(lidz,lidy+fdoh,lidx)=vy(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lvy(lidz,lidy-lsizey+3*fdoh,lidx)=vy(gidz,gidy-lsizey+3*fdoh,gidx);
        if (lidx<2*fdoh)
            lvy(lidz,lidy,lidx-fdoh)=vy(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvy(lidz,lidy,lidx+lsizex-3*fdoh)=vy(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvy(lidz,lidy,lidx+fdoh)=vy(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvy(lidz,lidy,lidx-lsizex+3*fdoh)=vy(gidz,gidy,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvy(lidz-fdoh,lidy,lidx)=vy(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvy(lidz+fdoh,lidy,lidx)=vy(gidz+fdoh,gidy,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh==1
        vyx = (lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))/DH;
        vyy = (lvy(lidz,lidy,lidx)-lvy(lidz,lidy-1,lidx))/DH;
        vyz = (lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))/DH;
#elif fdoh==2
        vyx = (hc1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
               hc2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1)))/DH;
        
        vyy = (hc1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
               hc2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx)))/DH;
        
        vyz = (hc1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
               hc2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx)))/DH;
#elif fdoh==3
        vyx = (hc1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
               hc2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
               hc3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2)))/DH;
        
        vyy = (hc1*(lvy(lidz,lidy,lidx)-lvy(lidz,lidy-1,lidx))+
               hc2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
               hc3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx)))/DH;
        
        vyz = (hc1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
               hc2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
               hc3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx)))/DH;
#elif fdoh==4
        vyx = (hc1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
               hc2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
               hc3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2))+
               hc4*(lvy(lidz,lidy,lidx+4)-lvy(lidz,lidy,lidx-3)))/DH;
        
        vyy = (hc1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
               hc2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
               hc3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx))+
               hc4*(lvy(lidz,lidy+3,lidx)-lvy(lidz,lidy-4,lidx)))/DH;
        
        vyz = (hc1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
               hc2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
               hc3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx))+
               hc4*(lvy(lidz+4,lidy,lidx)-lvy(lidz-3,lidy,lidx)))/DH;
#elif fdoh==5
        vyx = (hc1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
               hc2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
               hc3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2))+
               hc4*(lvy(lidz,lidy,lidx+4)-lvy(lidz,lidy,lidx-3))+
               hc5*(lvy(lidz,lidy,lidx+5)-lvy(lidz,lidy,lidx-4)))/DH;
        
        vyy = (hc1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
               hc2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
               hc3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx))+
               hc4*(lvy(lidz,lidy+3,lidx)-lvy(lidz,lidy-4,lidx))+
               hc5*(lvy(lidz,lidy+4,lidx)-lvy(lidz,lidy-5,lidx)))/DH;
        
        vyz = (hc1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
               hc2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
               hc3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx))+
               hc4*(lvy(lidz+4,lidy,lidx)-lvy(lidz-3,lidy,lidx))+
               hc5*(lvy(lidz+5,lidy,lidx)-lvy(lidz-4,lidy,lidx)))/DH;
#elif fdoh==6
        vyx = (hc1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
               hc2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
               hc3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2))+
               hc4*(lvy(lidz,lidy,lidx+4)-lvy(lidz,lidy,lidx-3))+
               hc5*(lvy(lidz,lidy,lidx+5)-lvy(lidz,lidy,lidx-4))+
               hc6*(lvy(lidz,lidy,lidx+6)-lvy(lidz,lidy,lidx-5)))/DH;
        
        vyy = (hc1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
               hc2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
               hc3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx))+
               hc4*(lvy(lidz,lidy+3,lidx)-lvy(lidz,lidy-4,lidx))+
               hc5*(lvy(lidz,lidy+4,lidx)-lvy(lidz,lidy-5,lidx))+
               hc6*(lvy(lidz,lidy+5,lidx)-lvy(lidz,lidy-6,lidx)))/DH;
        
        vyz = (hc1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
               hc2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
               hc3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx))+
               hc4*(lvy(lidz+4,lidy,lidx)-lvy(lidz-3,lidy,lidx))+
               hc5*(lvy(lidz+5,lidy,lidx)-lvy(lidz-4,lidy,lidx))+
               hc6*(lvy(lidz+6,lidy,lidx)-lvy(lidz-5,lidy,lidx)))/DH;
#endif
        
        
        
        
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lvz(lidz,lidy,lidx)=vz(gidz, gidy, gidx);
        if (lidy<2*fdoh)
            lvz(lidz,lidy-fdoh,lidx)=vz(gidz,gidy-fdoh,gidx);
        if (lidy+lsizey-3*fdoh<fdoh)
            lvz(lidz,lidy+lsizey-3*fdoh,lidx)=vz(gidz,gidy+lsizey-3*fdoh,gidx);
        if (lidy>(lsizey-2*fdoh-1))
            lvz(lidz,lidy+fdoh,lidx)=vz(gidz,gidy+fdoh,gidx);
        if (lidy-lsizey+3*fdoh>(lsizey-fdoh-1))
            lvz(lidz,lidy-lsizey+3*fdoh,lidx)=vz(gidz,gidy-lsizey+3*fdoh,gidx);
        if (lidx<2*fdoh)
            lvz(lidz,lidy,lidx-fdoh)=vz(gidz,gidy,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvz(lidz,lidy,lidx+lsizex-3*fdoh)=vz(gidz,gidy,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvz(lidz,lidy,lidx+fdoh)=vz(gidz,gidy,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvz(lidz,lidy,lidx-lsizex+3*fdoh)=vz(gidz,gidy,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvz(lidz-fdoh,lidy,lidx)=vz(gidz-fdoh,gidy,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvz(lidz+fdoh,lidy,lidx)=vz(gidz+fdoh,gidy,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
        
#if   fdoh==1
        vzx = (lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))/DH;
        vzy = (lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))/DH;
        vzz = (lvz(lidz,lidy,lidx)-lvz(lidz-1,lidy,lidx))/DH;
#elif fdoh==2
        
        vzx = (hc1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
               hc2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1)))/DH;
        
        vzy = (hc1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
               hc2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx)))/DH;
        
        vzz = (hc1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
               hc2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx)))/DH;
#elif fdoh==3
        vzx = (hc1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
               hc2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
               hc3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2)))/DH;
        
        vzy = (hc1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
               hc2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
               hc3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx)))/DH;
        
        vzz = (hc1*(lvz(lidz,lidy,lidx)-lvz(lidz-1,lidy,lidx))+
               hc2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
               hc3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx)))/DH;
#elif fdoh==4
        vzx = (hc1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
               hc2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
               hc3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2))+
               hc4*(lvz(lidz,lidy,lidx+4)-lvz(lidz,lidy,lidx-3)))/DH;
        
        vzy = (hc1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
               hc2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
               hc3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx))+
               hc4*(lvz(lidz,lidy+4,lidx)-lvz(lidz,lidy-3,lidx)))/DH;
        
        vzz = (hc1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
               hc2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
               hc3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx))+
               hc4*(lvz(lidz+3,lidy,lidx)-lvz(lidz-4,lidy,lidx)))/DH;
#elif fdoh==5
        vzx = (hc1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
               hc2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
               hc3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2))+
               hc4*(lvz(lidz,lidy,lidx+4)-lvz(lidz,lidy,lidx-3))+
               hc5*(lvz(lidz,lidy,lidx+5)-lvz(lidz,lidy,lidx-4)))/DH;
        
        vzy = (hc1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
               hc2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
               hc3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx))+
               hc4*(lvz(lidz,lidy+4,lidx)-lvz(lidz,lidy-3,lidx))+
               hc5*(lvz(lidz,lidy+5,lidx)-lvz(lidz,lidy-4,lidx)))/DH;
        
        vzz = (hc1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
               hc2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
               hc3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx))+
               hc4*(lvz(lidz+3,lidy,lidx)-lvz(lidz-4,lidy,lidx))+
               hc5*(lvz(lidz+4,lidy,lidx)-lvz(lidz-5,lidy,lidx)))/DH;
#elif fdoh==6
        vzx = (hc1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
               hc2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
               hc3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2))+
               hc4*(lvz(lidz,lidy,lidx+4)-lvz(lidz,lidy,lidx-3))+
               hc5*(lvz(lidz,lidy,lidx+5)-lvz(lidz,lidy,lidx-4))+
               hc6*(lvz(lidz,lidy,lidx+6)-lvz(lidz,lidy,lidx-5)))/DH;
        
        vzy = (hc1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
               hc2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
               hc3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx))+
               hc4*(lvz(lidz,lidy+4,lidx)-lvz(lidz,lidy-3,lidx))+
               hc5*(lvz(lidz,lidy+5,lidx)-lvz(lidz,lidy-4,lidx))+
               hc6*(lvz(lidz,lidy+6,lidx)-lvz(lidz,lidy-5,lidx)))/DH;
        
        vzz = (hc1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
               hc2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
               hc3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx))+
               hc4*(lvz(lidz+3,lidy,lidx)-lvz(lidz-4,lidy,lidx))+
               hc5*(lvz(lidz+4,lidy,lidx)-lvz(lidz-5,lidy,lidx))+
               hc6*(lvz(lidz+5,lidy,lidx)-lvz(lidz-6,lidy,lidx)))/DH;
#endif
    }
    
// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
#if comm12==0
    if (gidy>(NY-fdoh-1) || gidz>(NZ-fdoh-1) || (gidx-offcomm)>(NX-fdoh-1-lcomm) ){
        return;
    }
#else
    if (gidy>(NY-fdoh-1) || gidz>(NZ-fdoh-1) ){
        return;
    }
#endif

 
// Correct spatial derivatives to implement CPML
#if abs_type==1
    {
        
        if (gidz>NZ-nab-fdoh-1){
            
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz - NZ+nab+fdoh+nab;
            ind=2*nab-1-k;
            
            psi_vxz(k,j,i) = b_z_half[ind] * psi_vxz(k,j,i) + a_z_half[ind] * vxz;
            vxz = vxz / K_z_half[ind] + psi_vxz(k,j,i);
            psi_vyz(k,j,i) = b_z_half[ind] * psi_vyz(k,j,i) + a_z_half[ind] * vyz;
            vyz = vyz / K_z_half[ind] + psi_vyz(k,j,i);
            psi_vzz(k,j,i) = b_z[ind+1] * psi_vzz(k,j,i) + a_z[ind+1] * vzz;
            vzz = vzz / K_z[ind+1] + psi_vzz(k,j,i);
            
        }
        
#if freesurf==0
        else if (gidz-fdoh<nab){
            
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;
            
            
            psi_vxz(k,j,i) = b_z_half[k] * psi_vxz(k,j,i) + a_z_half[k] * vxz;
            vxz = vxz / K_z_half[k] + psi_vxz(k,j,i);
            psi_vyz(k,j,i) = b_z_half[k] * psi_vyz(k,j,i) + a_z_half[k] * vyz;
            vyz = vyz / K_z_half[k] + psi_vyz(k,j,i);
            psi_vzz(k,j,i) = b_z[k] * psi_vzz(k,j,i) + a_z[k] * vzz;
            vzz = vzz / K_z[k] + psi_vzz(k,j,i);
            
            
        }
#endif
        
        if (gidy-fdoh<nab){
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;
            
            psi_vxy(k,j,i) = b_y_half[j] * psi_vxy(k,j,i) + a_y_half[j] * vxy;
            vxy = vxy / K_y_half[j] + psi_vxy(k,j,i);
            psi_vyy(k,j,i) = b_y[j] * psi_vyy(k,j,i) + a_y[j] * vyy;
            vyy = vyy / K_y[j] + psi_vyy(k,j,i);
            psi_vzy(k,j,i) = b_y_half[j] * psi_vzy(k,j,i) + a_y_half[j] * vzy;
            vzy = vzy / K_y_half[j] + psi_vzy(k,j,i);
            
        }
        
        else if (gidy>NY-nab-fdoh-1){
            
            i =gidx-fdoh;
            j =gidy - NY+nab+fdoh+nab;
            k =gidz-fdoh;
            ind=2*nab-1-j;
            
            
            psi_vxy(k,j,i) = b_y_half[ind] * psi_vxy(k,j,i) + a_y_half[ind] * vxy;
            vxy = vxy / K_y_half[ind] + psi_vxy(k,j,i);
            psi_vyy(k,j,i) = b_y[ind+1] * psi_vyy(k,j,i) + a_y[ind+1] * vyy;
            vyy = vyy / K_y[ind+1] + psi_vyy(k,j,i);
            psi_vzy(k,j,i) = b_y_half[ind] * psi_vzy(k,j,i) + a_y_half[ind] * vzy;
            vzy = vzy / K_y_half[ind] + psi_vzy(k,j,i);
            
            
        }
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;
            
            psi_vxx(k,j,i) = b_x[i] * psi_vxx(k,j,i) + a_x[i] * vxx;
            vxx = vxx / K_x[i] + psi_vxx(k,j,i);
            psi_vyx(k,j,i) = b_x_half[i] * psi_vyx(k,j,i) + a_x_half[i] * vyx;
            vyx = vyx / K_x_half[i] + psi_vyx(k,j,i);
            psi_vzx(k,j,i) = b_x_half[i] * psi_vzx(k,j,i) + a_x_half[i] * vzx;
            vzx = vzx / K_x_half[i] + psi_vzx(k,j,i);
            
            
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            
            i =gidx - NX+nab+fdoh+nab;
            j =gidy-fdoh;
            k =gidz-fdoh;
            ind=2*nab-1-i;
            
            
            psi_vxx(k,j,i) = b_x[ind+1] * psi_vxx(k,j,i) + a_x[ind+1] * vxx;
            vxx = vxx /K_x[ind+1] + psi_vxx(k,j,i);
            psi_vyx(k,j,i) = b_x_half[ind] * psi_vyx(k,j,i) + a_x_half[ind] * vyx;
            vyx = vyx  /K_x_half[ind] + psi_vyx(k,j,i);
            psi_vzx(k,j,i) = b_x_half[ind] * psi_vzx(k,j,i) + a_x_half[ind] * vzx;
            vzx = vzx / K_x_half[ind]  +psi_vzx(k,j,i);
            
            
        }
#endif
    }
#endif

// Read model parameters into local memory
    {
#if Lve==0
        
        fipjp=uipjp(gidz,gidy,gidx)*DT;
        fjpkp=ujpkp(gidz,gidy,gidx)*DT;
        fipkp=uipkp(gidz,gidy,gidx)*DT;
        g=pi(gidz,gidy,gidx)*DT;
        f=2.0*u(gidz,gidy,gidx)*DT;
        
#else
        
        lpi=pi(gidz,gidy,gidx);
        lu=u(gidz,gidy,gidx);
        luipkp=uipkp(gidz,gidy,gidx);
        luipjp=uipjp(gidz,gidy,gidx);
        lujpkp=ujpkp(gidz,gidy,gidx);
        ltaup=taup(gidz,gidy,gidx);
        ltaus=taus(gidz,gidy,gidx);
        ltausipkp=tausipkp(gidz,gidy,gidx);
        ltausipjp=tausipjp(gidz,gidy,gidx);
        ltausjpkp=tausjpkp(gidz,gidy,gidx);
        
        for (l=0;l<Lve;l++){
            leta[l]=eta[l];
        }
        
        fipjp=luipjp*DT*(1.0+ (float)Lve*ltausipjp);
        fjpkp=lujpkp*DT*(1.0+ (float)Lve*ltausjpkp);
        fipkp=luipkp*DT*(1.0+ (float)Lve*ltausipkp);
        g=lpi*(1.0+(float)Lve*ltaup)*DT;
        f=2.0*lu*(1.0+(float)Lve*ltaus)*DT;
        dipjp=luipjp*ltausipjp;
        djpkp=lujpkp*ltausjpkp;
        dipkp=luipkp*ltausipkp;
        d=2.0*lu*ltaus;
        e=lpi*ltaup;
        
        
#endif
    }

// Update the stresses
    {
#if Lve==0
        
        vxyyx=vxy+vyx;
        vyzzy=vyz+vzy;
        vxzzx=vxz+vzx;
        vxxyyzz=vxx+vyy+vzz;
        vyyzz=vyy+vzz;
        vxxzz=vxx+vzz;
        vxxyy=vxx+vyy;

        
        sxy(gidz,gidy,gidx)+=(fipjp*vxyyx);
        syz(gidz,gidy,gidx)+=(fjpkp*vyzzy);
        sxz(gidz,gidy,gidx)+=(fipkp*vxzzx);
        sxx(gidz,gidy,gidx)+=((g*vxxyyzz)-(f*vyyzz)) ;
        syy(gidz,gidy,gidx)+=((g*vxxyyzz)-(f*vxxzz)) ;
        szz(gidz,gidy,gidx)+=((g*vxxyyzz)-(f*vxxyy)) ;
        
        
#else
        
        /* computing sums of the old memory variables */
        sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
        for (l=0;l<Lve;l++){
            sumrxy+=rxy(gidz,gidy,gidx,l);
            sumryz+=ryz(gidz,gidy,gidx,l);
            sumrxz+=rxz(gidz,gidy,gidx,l);
            sumrxx+=rxx(gidz,gidy,gidx,l);
            sumryy+=ryy(gidz,gidy,gidx,l);
            sumrzz+=rzz(gidz,gidy,gidx,l);
        }
        
        /* updating components of the stress tensor, partially */
        
        vxyyx=vxy+vyx;
        vyzzy=vyz+vzy;
        vxzzx=vxz+vzx;
        vxxyyzz=vxx+vyy+vzz;
        vyyzz=vyy+vzz;
        vxxzz=vxx+vzz;
        vxxyy=vxx+vyy;
        
        lsxy=(fipjp*vxyyx)+(dt2*sumrxy);
        lsyz=(fjpkp*vyzzy)+(dt2*sumryz);
        lsxz=(fipkp*vxzzx)+(dt2*sumrxz);
        lsxx=((g*vxxyyzz)-(f*vyyzz))+(dt2*sumrxx);
        lsyy=((g*vxxyyzz)-(f*vxxzz))+(dt2*sumryy);
        lszz=((g*vxxyyzz)-(f*vxxyy))+(dt2*sumrzz);
        
        
        sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
        for (l=0;l<Lve;l++){
            b=1.0/(1.0+(leta[l]*0.5));
            c=1.0-(leta[l]*0.5);
            
            rxy(gidz,gidy,gidx,l)=b*(rxy(gidz,gidy,gidx,l)*c-leta[l]*(dipjp*vxyyx));
            ryz(gidz,gidy,gidx,l)=b*(ryz(gidz,gidy,gidx,l)*c-leta[l]*(djpkp*vyzzy));
            rxz(gidz,gidy,gidx,l)=b*(rxz(gidz,gidy,gidx,l)*c-leta[l]*(dipkp*vxzzx));
            rxx(gidz,gidy,gidx,l)=b*(rxx(gidz,gidy,gidx,l)*c-leta[l]*((e*vxxyyzz)-(d*vyyzz)));
            ryy(gidz,gidy,gidx,l)=b*(ryy(gidz,gidy,gidx,l)*c-leta[l]*((e*vxxyyzz)-(d*vxxzz)));
            rzz(gidz,gidy,gidx,l)=b*(rzz(gidz,gidy,gidx,l)*c-leta[l]*((e*vxxyyzz)-(d*vxxyy)));
            
            
            
            sumrxy+=rxy(gidz,gidy,gidx,l);
            sumryz+=ryz(gidz,gidy,gidx,l);
            sumrxz+=rxz(gidz,gidy,gidx,l);
            sumrxx+=rxx(gidz,gidy,gidx,l);
            sumryy+=ryy(gidz,gidy,gidx,l);
            sumrzz+=rzz(gidz,gidy,gidx,l);
        }

        /* and now the components of the stress tensor are
         completely updated */
        sxy(gidz,gidy,gidx)+=lsxy+(dt2*sumrxy);
        syz(gidz,gidy,gidx)+=lsyz+(dt2*sumryz);
        sxz(gidz,gidy,gidx)+=lsxz+(dt2*sumrxz);
        sxx(gidz,gidy,gidx)+=lsxx+(dt2*sumrxx);
        syy(gidz,gidy,gidx)+=lsyy+(dt2*sumryy);
        szz(gidz,gidy,gidx)+=lszz+(dt2*sumrzz);
        
        
#endif
    }

// Absorbing boundary    
#if abstype==2
    {
        
#if freesurf==0
        if (gidz-fdoh<nab){
            sxy(gidz,gidy,gidx)*=taper[gidz-fdoh];
            syz(gidz,gidy,gidx)*=taper[gidz-fdoh];
            sxz(gidz,gidy,gidx)*=taper[gidz-fdoh];
            sxx(gidz,gidy,gidx)*=taper[gidz-fdoh];
            syy(gidz,gidy,gidx)*=taper[gidz-fdoh];
            szz(gidz,gidy,gidx)*=taper[gidz-fdoh];
        }
#endif
        
        if (gidz>NZ-nab-fdoh-1){
            sxy(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            syz(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            sxz(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            sxx(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            syy(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
            szz(gidz,gidy,gidx)*=taper[NZ-fdoh-gidz-1];
        }
        if (gidy-fdoh<nab){
            sxy(gidz,gidy,gidx)*=taper[gidy-fdoh];
            syz(gidz,gidy,gidx)*=taper[gidy-fdoh];
            sxz(gidz,gidy,gidx)*=taper[gidy-fdoh];
            sxx(gidz,gidy,gidx)*=taper[gidy-fdoh];
            syy(gidz,gidy,gidx)*=taper[gidy-fdoh];
            szz(gidz,gidy,gidx)*=taper[gidy-fdoh];
        }
        
        if (gidy>NY-nab-fdoh-1){
            sxy(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            syz(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            sxz(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            sxx(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            syy(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            szz(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
        }
        
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            sxy(gidz,gidy,gidx)*=taper[gidx-fdoh];
            syz(gidz,gidy,gidx)*=taper[gidx-fdoh];
            sxz(gidz,gidy,gidx)*=taper[gidx-fdoh];
            sxx(gidz,gidy,gidx)*=taper[gidx-fdoh];
            syy(gidz,gidy,gidx)*=taper[gidx-fdoh];
            szz(gidz,gidy,gidx)*=taper[gidx-fdoh];
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            sxy(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            syz(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            sxz(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            sxx(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            syy(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            szz(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
        }
#endif
    }
#endif
    
}

