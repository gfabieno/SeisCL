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
#define rho(z,y,x)     rho[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,y,x)     rip[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rjp(z,y,x)     rjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,y,x)     rkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define uipjp(z,y,x) uipjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define ujpkp(z,y,x) ujpkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define uipkp(z,y,x) uipkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define u(z,y,x)         u[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define pi(z,y,x)       pi[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradrho(z,y,x)   gradrho[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradM(z,y,x)   gradM[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradmu(z,y,x)   gradmu[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaup(z,y,x)   gradtaup[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define gradtaus(z,y,x)   gradtaus[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define taus(z,y,x)         taus[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausipjp(z,y,x) tausipjp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausjpkp(z,y,x) tausjpkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define tausipkp(z,y,x) tausipkp[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define taup(z,y,x)         taup[((x)-FDOH)*(NY-2*FDOH)*(NZ-2*FDOH)+((y)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

#define vx(z,y,x)   vx[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define vy(z,y,x)   vy[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define vz(z,y,x)   vz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxx(z,y,x) sxx[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define syy(z,y,x) syy[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define szz(z,y,x) szz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxy(z,y,x) sxy[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define syz(z,y,x) syz[(x)*NY*(NZ)+(y)*(NZ)+(z)]
#define sxz(z,y,x) sxz[(x)*NY*(NZ)+(y)*(NZ)+(z)]

#define rxx(z,y,x,l) rxx[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryy(z,y,x,l) ryy[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rzz(z,y,x,l) rzz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxy(z,y,x,l) rxy[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define ryz(z,y,x,l) ryz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]
#define rxz(z,y,x,l) rxz[(l)*NX*NY*NZ+(x)*NY*NZ+(y)*NZ+(z)]

#define psi_vxx(z,y,x) psi_vxx[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vyx(z,y,x) psi_vyx[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vzx(z,y,x) psi_vzx[(x)*(NY-2*FDOH)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]

#define psi_vxy(z,y,x) psi_vxy[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vyy(z,y,x) psi_vyy[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]
#define psi_vzy(z,y,x) psi_vzy[(x)*(2*NAB)*(NZ-2*FDOH)+(y)*(NZ-2*FDOH)+(z)]

#define psi_vxz(z,y,x) psi_vxz[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_vyz(z,y,x) psi_vyz[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]
#define psi_vzz(z,y,x) psi_vzz[(x)*(NY-2*FDOH)*(2*NAB)+(y)*(2*NAB)+(z)]



#if LOCAL_OFF==0

#define lvar(z,y,x)   lvar[(x)*lsizey*lsizez+(y)*lsizez+(z)]

#endif



#define PI (3.141592653589793238462643383279502884197169)
#define signals(y,x) signals[(y)*NT+(x)]



__kernel void update_s(int offcomm,  int nt,
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
#if LVE>0
    float leta[LVE];
#endif
    float vxx,vxy,vxz,vyx,vyy,vyz,vzx,vzy,vzz;
    float vxyyx,vyzzy,vxzzx,vxxyyzz,vyyzz,vxxzz,vxxyy;
    
    float lsxx, lsyy, lszz, lsxy, lsxz, lsyz;
    
// If we use local memory
#if LOCAL_OFF==0
    
    int lsizez = get_local_size(0)+2*FDOH;
    int lsizey = get_local_size(1)+2*FDOH;
    int lsizex = get_local_size(2)+2*FDOH;
    int lidz = get_local_id(0)+FDOH;
    int lidy = get_local_id(1)+FDOH;
    int lidx = get_local_id(2)+FDOH;
    int gidz = get_global_id(0)+FDOH;
    int gidy = get_global_id(1)+FDOH;
    int gidx = get_global_id(2)+FDOH+offcomm;
    
#define lvx lvar
#define lvy lvar
#define lvz lvar

// If local memory is turned off
#elif LOCAL_OFF==1
    
    int gid = get_global_id(0);
    int glsizez = (NZ-2*FDOH);
    int glsizey = (NY-2*FDOH);
    int gidz = gid%glsizez+FDOH;
    int gidy = (gid/glsizez)%glsizey+FDOH;
    int gidx = gid/(glsizez*glsizey)+FDOH+offcomm;
    
#define lvx vx
#define lvy vy
#define lvz vz
#define lidx gidx
#define lidy gidy
#define lidz gidz
    
#endif
    
// Calculation of the velocity spatial derivatives
    {
#if LOCAL_OFF==0
        lvx(lidz,lidy,lidx)=vx(gidz, gidy, gidx);
        if (lidy<2*FDOH)
            lvx(lidz,lidy-FDOH,lidx)=vx(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lvx(lidz,lidy+lsizey-3*FDOH,lidx)=vx(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lvx(lidz,lidy+FDOH,lidx)=vx(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lvx(lidz,lidy-lsizey+3*FDOH,lidx)=vx(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lvx(lidz,lidy,lidx-FDOH)=vx(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvx(lidz,lidy,lidx+lsizex-3*FDOH)=vx(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvx(lidz,lidy,lidx+FDOH)=vx(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvx(lidz,lidy,lidx-lsizex+3*FDOH)=vx(gidz,gidy,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvx(lidz-FDOH,lidy,lidx)=vx(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvx(lidz+FDOH,lidy,lidx)=vx(gidz+FDOH,gidy,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   FDOH==1
        vxx = (lvx(lidz,lidy,lidx)-lvx(lidz,lidy,lidx-1));
        vxy = (lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx));
        vxz = (lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx));
#elif FDOH==2
        vxx = (HC1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
               HC2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2)));
        
        vxy = (HC1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
               HC2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx)));
        
        vxz = (HC1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
               HC2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx)));
#elif FDOH==3
        vxx = (HC1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
               HC2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
               HC3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3)));
        
        vxy = (HC1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
               HC2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
               HC3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx)));
        
        vxz = (HC1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
               HC2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
               HC3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx)));
#elif FDOH==4
        vxx = (HC1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
               HC2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
               HC3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3))+
               HC4*(lvx(lidz,lidy,lidx+3)-lvx(lidz,lidy,lidx-4)));
        
        vxy = (HC1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
               HC2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
               HC3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx))+
               HC4*(lvx(lidz,lidy+4,lidx)-lvx(lidz,lidy-3,lidx)));
        
        vxz = (HC1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
               HC2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
               HC3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx))+
               HC4*(lvx(lidz+4,lidy,lidx)-lvx(lidz-3,lidy,lidx)));
#elif FDOH==5
        vxx = (HC1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
               HC2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
               HC3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3))+
               HC4*(lvx(lidz,lidy,lidx+3)-lvx(lidz,lidy,lidx-4))+
               HC5*(lvx(lidz,lidy,lidx+4)-lvx(lidz,lidy,lidx-5)));
        
        vxy = (HC1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
               HC2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
               HC3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx))+
               HC4*(lvx(lidz,lidy+4,lidx)-lvx(lidz,lidy-3,lidx))+
               HC5*(lvx(lidz,lidy+5,lidx)-lvx(lidz,lidy-4,lidx)));
        
        vxz = (HC1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
               HC2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
               HC3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx))+
               HC4*(lvx(lidz+4,lidy,lidx)-lvx(lidz-3,lidy,lidx))+
               HC5*(lvx(lidz+5,lidy,lidx)-lvx(lidz-4,lidy,lidx)));
#elif FDOH==6
        vxx = (HC1*(lvx(lidz,lidy,lidx)  -lvx(lidz,lidy,lidx-1))+
               HC2*(lvx(lidz,lidy,lidx+1)-lvx(lidz,lidy,lidx-2))+
               HC3*(lvx(lidz,lidy,lidx+2)-lvx(lidz,lidy,lidx-3))+
               HC4*(lvx(lidz,lidy,lidx+3)-lvx(lidz,lidy,lidx-4))+
               HC5*(lvx(lidz,lidy,lidx+4)-lvx(lidz,lidy,lidx-5))+
               HC6*(lvx(lidz,lidy,lidx+5)-lvx(lidz,lidy,lidx-6)));
        
        vxy = (HC1*(lvx(lidz,lidy+1,lidx)-lvx(lidz,lidy,lidx))+
               HC2*(lvx(lidz,lidy+2,lidx)-lvx(lidz,lidy-1,lidx))+
               HC3*(lvx(lidz,lidy+3,lidx)-lvx(lidz,lidy-2,lidx))+
               HC4*(lvx(lidz,lidy+4,lidx)-lvx(lidz,lidy-3,lidx))+
               HC5*(lvx(lidz,lidy+5,lidx)-lvx(lidz,lidy-4,lidx))+
               HC6*(lvx(lidz,lidy+6,lidx)-lvx(lidz,lidy-5,lidx)));
        
        vxz = (HC1*(lvx(lidz+1,lidy,lidx)-lvx(lidz,lidy,lidx))+
               HC2*(lvx(lidz+2,lidy,lidx)-lvx(lidz-1,lidy,lidx))+
               HC3*(lvx(lidz+3,lidy,lidx)-lvx(lidz-2,lidy,lidx))+
               HC4*(lvx(lidz+4,lidy,lidx)-lvx(lidz-3,lidy,lidx))+
               HC5*(lvx(lidz+5,lidy,lidx)-lvx(lidz-4,lidy,lidx))+
               HC6*(lvx(lidz+6,lidy,lidx)-lvx(lidz-5,lidy,lidx)));
#endif
        
        
#if LOCAL_OFF==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lvy(lidz,lidy,lidx)=vy(gidz, gidy, gidx);
        if (lidy<2*FDOH)
            lvy(lidz,lidy-FDOH,lidx)=vy(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lvy(lidz,lidy+lsizey-3*FDOH,lidx)=vy(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lvy(lidz,lidy+FDOH,lidx)=vy(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lvy(lidz,lidy-lsizey+3*FDOH,lidx)=vy(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lvy(lidz,lidy,lidx-FDOH)=vy(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvy(lidz,lidy,lidx+lsizex-3*FDOH)=vy(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvy(lidz,lidy,lidx+FDOH)=vy(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvy(lidz,lidy,lidx-lsizex+3*FDOH)=vy(gidz,gidy,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvy(lidz-FDOH,lidy,lidx)=vy(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvy(lidz+FDOH,lidy,lidx)=vy(gidz+FDOH,gidy,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   FDOH==1
        vyx = (lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx));
        vyy = (lvy(lidz,lidy,lidx)-lvy(lidz,lidy-1,lidx));
        vyz = (lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx));
#elif FDOH==2
        vyx = (HC1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
               HC2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1)));
        
        vyy = (HC1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
               HC2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx)));
        
        vyz = (HC1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
               HC2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx)));
#elif FDOH==3
        vyx = (HC1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
               HC2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
               HC3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2)));
        
        vyy = (HC1*(lvy(lidz,lidy,lidx)-lvy(lidz,lidy-1,lidx))+
               HC2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
               HC3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx)));
        
        vyz = (HC1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
               HC2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
               HC3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx)));
#elif FDOH==4
        vyx = (HC1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
               HC2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
               HC3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2))+
               HC4*(lvy(lidz,lidy,lidx+4)-lvy(lidz,lidy,lidx-3)));
        
        vyy = (HC1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
               HC2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
               HC3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx))+
               HC4*(lvy(lidz,lidy+3,lidx)-lvy(lidz,lidy-4,lidx)));
        
        vyz = (HC1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
               HC2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
               HC3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx))+
               HC4*(lvy(lidz+4,lidy,lidx)-lvy(lidz-3,lidy,lidx)));
#elif FDOH==5
        vyx = (HC1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
               HC2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
               HC3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2))+
               HC4*(lvy(lidz,lidy,lidx+4)-lvy(lidz,lidy,lidx-3))+
               HC5*(lvy(lidz,lidy,lidx+5)-lvy(lidz,lidy,lidx-4)));
        
        vyy = (HC1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
               HC2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
               HC3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx))+
               HC4*(lvy(lidz,lidy+3,lidx)-lvy(lidz,lidy-4,lidx))+
               HC5*(lvy(lidz,lidy+4,lidx)-lvy(lidz,lidy-5,lidx)));
        
        vyz = (HC1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
               HC2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
               HC3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx))+
               HC4*(lvy(lidz+4,lidy,lidx)-lvy(lidz-3,lidy,lidx))+
               HC5*(lvy(lidz+5,lidy,lidx)-lvy(lidz-4,lidy,lidx)));
#elif FDOH==6
        vyx = (HC1*(lvy(lidz,lidy,lidx+1)-lvy(lidz,lidy,lidx))+
               HC2*(lvy(lidz,lidy,lidx+2)-lvy(lidz,lidy,lidx-1))+
               HC3*(lvy(lidz,lidy,lidx+3)-lvy(lidz,lidy,lidx-2))+
               HC4*(lvy(lidz,lidy,lidx+4)-lvy(lidz,lidy,lidx-3))+
               HC5*(lvy(lidz,lidy,lidx+5)-lvy(lidz,lidy,lidx-4))+
               HC6*(lvy(lidz,lidy,lidx+6)-lvy(lidz,lidy,lidx-5)));
        
        vyy = (HC1*(lvy(lidz,lidy,lidx)  -lvy(lidz,lidy-1,lidx))+
               HC2*(lvy(lidz,lidy+1,lidx)-lvy(lidz,lidy-2,lidx))+
               HC3*(lvy(lidz,lidy+2,lidx)-lvy(lidz,lidy-3,lidx))+
               HC4*(lvy(lidz,lidy+3,lidx)-lvy(lidz,lidy-4,lidx))+
               HC5*(lvy(lidz,lidy+4,lidx)-lvy(lidz,lidy-5,lidx))+
               HC6*(lvy(lidz,lidy+5,lidx)-lvy(lidz,lidy-6,lidx)));
        
        vyz = (HC1*(lvy(lidz+1,lidy,lidx)-lvy(lidz,lidy,lidx))+
               HC2*(lvy(lidz+2,lidy,lidx)-lvy(lidz-1,lidy,lidx))+
               HC3*(lvy(lidz+3,lidy,lidx)-lvy(lidz-2,lidy,lidx))+
               HC4*(lvy(lidz+4,lidy,lidx)-lvy(lidz-3,lidy,lidx))+
               HC5*(lvy(lidz+5,lidy,lidx)-lvy(lidz-4,lidy,lidx))+
               HC6*(lvy(lidz+6,lidy,lidx)-lvy(lidz-5,lidy,lidx)));
#endif
        
        
        
        
#if LOCAL_OFF==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lvz(lidz,lidy,lidx)=vz(gidz, gidy, gidx);
        if (lidy<2*FDOH)
            lvz(lidz,lidy-FDOH,lidx)=vz(gidz,gidy-FDOH,gidx);
        if (lidy+lsizey-3*FDOH<FDOH)
            lvz(lidz,lidy+lsizey-3*FDOH,lidx)=vz(gidz,gidy+lsizey-3*FDOH,gidx);
        if (lidy>(lsizey-2*FDOH-1))
            lvz(lidz,lidy+FDOH,lidx)=vz(gidz,gidy+FDOH,gidx);
        if (lidy-lsizey+3*FDOH>(lsizey-FDOH-1))
            lvz(lidz,lidy-lsizey+3*FDOH,lidx)=vz(gidz,gidy-lsizey+3*FDOH,gidx);
        if (lidx<2*FDOH)
            lvz(lidz,lidy,lidx-FDOH)=vz(gidz,gidy,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvz(lidz,lidy,lidx+lsizex-3*FDOH)=vz(gidz,gidy,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvz(lidz,lidy,lidx+FDOH)=vz(gidz,gidy,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvz(lidz,lidy,lidx-lsizex+3*FDOH)=vz(gidz,gidy,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvz(lidz-FDOH,lidy,lidx)=vz(gidz-FDOH,gidy,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvz(lidz+FDOH,lidy,lidx)=vz(gidz+FDOH,gidy,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
        
#if   FDOH==1
        vzx = (lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx));
        vzy = (lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx));
        vzz = (lvz(lidz,lidy,lidx)-lvz(lidz-1,lidy,lidx));
#elif FDOH==2
        
        vzx = (HC1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
               HC2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1)));
        
        vzy = (HC1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
               HC2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx)));
        
        vzz = (HC1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
               HC2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx)));
#elif FDOH==3
        vzx = (HC1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
               HC2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
               HC3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2)));
        
        vzy = (HC1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
               HC2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
               HC3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx)));
        
        vzz = (HC1*(lvz(lidz,lidy,lidx)-lvz(lidz-1,lidy,lidx))+
               HC2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
               HC3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx)));
#elif FDOH==4
        vzx = (HC1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
               HC2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
               HC3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2))+
               HC4*(lvz(lidz,lidy,lidx+4)-lvz(lidz,lidy,lidx-3)));
        
        vzy = (HC1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
               HC2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
               HC3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx))+
               HC4*(lvz(lidz,lidy+4,lidx)-lvz(lidz,lidy-3,lidx)));
        
        vzz = (HC1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
               HC2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
               HC3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx))+
               HC4*(lvz(lidz+3,lidy,lidx)-lvz(lidz-4,lidy,lidx)));
#elif FDOH==5
        vzx = (HC1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
               HC2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
               HC3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2))+
               HC4*(lvz(lidz,lidy,lidx+4)-lvz(lidz,lidy,lidx-3))+
               HC5*(lvz(lidz,lidy,lidx+5)-lvz(lidz,lidy,lidx-4)));
        
        vzy = (HC1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
               HC2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
               HC3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx))+
               HC4*(lvz(lidz,lidy+4,lidx)-lvz(lidz,lidy-3,lidx))+
               HC5*(lvz(lidz,lidy+5,lidx)-lvz(lidz,lidy-4,lidx)));
        
        vzz = (HC1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
               HC2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
               HC3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx))+
               HC4*(lvz(lidz+3,lidy,lidx)-lvz(lidz-4,lidy,lidx))+
               HC5*(lvz(lidz+4,lidy,lidx)-lvz(lidz-5,lidy,lidx)));
#elif FDOH==6
        vzx = (HC1*(lvz(lidz,lidy,lidx+1)-lvz(lidz,lidy,lidx))+
               HC2*(lvz(lidz,lidy,lidx+2)-lvz(lidz,lidy,lidx-1))+
               HC3*(lvz(lidz,lidy,lidx+3)-lvz(lidz,lidy,lidx-2))+
               HC4*(lvz(lidz,lidy,lidx+4)-lvz(lidz,lidy,lidx-3))+
               HC5*(lvz(lidz,lidy,lidx+5)-lvz(lidz,lidy,lidx-4))+
               HC6*(lvz(lidz,lidy,lidx+6)-lvz(lidz,lidy,lidx-5)));
        
        vzy = (HC1*(lvz(lidz,lidy+1,lidx)-lvz(lidz,lidy,lidx))+
               HC2*(lvz(lidz,lidy+2,lidx)-lvz(lidz,lidy-1,lidx))+
               HC3*(lvz(lidz,lidy+3,lidx)-lvz(lidz,lidy-2,lidx))+
               HC4*(lvz(lidz,lidy+4,lidx)-lvz(lidz,lidy-3,lidx))+
               HC5*(lvz(lidz,lidy+5,lidx)-lvz(lidz,lidy-4,lidx))+
               HC6*(lvz(lidz,lidy+6,lidx)-lvz(lidz,lidy-5,lidx)));
        
        vzz = (HC1*(lvz(lidz,lidy,lidx)  -lvz(lidz-1,lidy,lidx))+
               HC2*(lvz(lidz+1,lidy,lidx)-lvz(lidz-2,lidy,lidx))+
               HC3*(lvz(lidz+2,lidy,lidx)-lvz(lidz-3,lidy,lidx))+
               HC4*(lvz(lidz+3,lidy,lidx)-lvz(lidz-4,lidy,lidx))+
               HC5*(lvz(lidz+4,lidy,lidx)-lvz(lidz-5,lidy,lidx))+
               HC6*(lvz(lidz+5,lidy,lidx)-lvz(lidz-6,lidy,lidx)));
#endif
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
        
        if (gidz>NZ-NAB-FDOH-1){
            
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz - NZ+NAB+FDOH+NAB;
            ind=2*NAB-1-k;
            
            psi_vxz(k,j,i) = b_z_half[ind] * psi_vxz(k,j,i) + a_z_half[ind] * vxz;
            vxz = vxz / K_z_half[ind] + psi_vxz(k,j,i);
            psi_vyz(k,j,i) = b_z_half[ind] * psi_vyz(k,j,i) + a_z_half[ind] * vyz;
            vyz = vyz / K_z_half[ind] + psi_vyz(k,j,i);
            psi_vzz(k,j,i) = b_z[ind+1] * psi_vzz(k,j,i) + a_z[ind+1] * vzz;
            vzz = vzz / K_z[ind+1] + psi_vzz(k,j,i);
            
        }
        
#if FREESURF==0
        else if (gidz-FDOH<NAB){
            
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;
            
            
            psi_vxz(k,j,i) = b_z_half[k] * psi_vxz(k,j,i) + a_z_half[k] * vxz;
            vxz = vxz / K_z_half[k] + psi_vxz(k,j,i);
            psi_vyz(k,j,i) = b_z_half[k] * psi_vyz(k,j,i) + a_z_half[k] * vyz;
            vyz = vyz / K_z_half[k] + psi_vyz(k,j,i);
            psi_vzz(k,j,i) = b_z[k] * psi_vzz(k,j,i) + a_z[k] * vzz;
            vzz = vzz / K_z[k] + psi_vzz(k,j,i);
            
            
        }
#endif
        
        if (gidy-FDOH<NAB){
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;
            
            psi_vxy(k,j,i) = b_y_half[j] * psi_vxy(k,j,i) + a_y_half[j] * vxy;
            vxy = vxy / K_y_half[j] + psi_vxy(k,j,i);
            psi_vyy(k,j,i) = b_y[j] * psi_vyy(k,j,i) + a_y[j] * vyy;
            vyy = vyy / K_y[j] + psi_vyy(k,j,i);
            psi_vzy(k,j,i) = b_y_half[j] * psi_vzy(k,j,i) + a_y_half[j] * vzy;
            vzy = vzy / K_y_half[j] + psi_vzy(k,j,i);
            
        }
        
        else if (gidy>NY-NAB-FDOH-1){
            
            i =gidx-FDOH;
            j =gidy - NY+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-j;
            
            
            psi_vxy(k,j,i) = b_y_half[ind] * psi_vxy(k,j,i) + a_y_half[ind] * vxy;
            vxy = vxy / K_y_half[ind] + psi_vxy(k,j,i);
            psi_vyy(k,j,i) = b_y[ind+1] * psi_vyy(k,j,i) + a_y[ind+1] * vyy;
            vyy = vyy / K_y[ind+1] + psi_vyy(k,j,i);
            psi_vzy(k,j,i) = b_y_half[ind] * psi_vzy(k,j,i) + a_y_half[ind] * vzy;
            vzy = vzy / K_y_half[ind] + psi_vzy(k,j,i);
            
            
        }
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            
            i =gidx-FDOH;
            j =gidy-FDOH;
            k =gidz-FDOH;
            
            psi_vxx(k,j,i) = b_x[i] * psi_vxx(k,j,i) + a_x[i] * vxx;
            vxx = vxx / K_x[i] + psi_vxx(k,j,i);
            psi_vyx(k,j,i) = b_x_half[i] * psi_vyx(k,j,i) + a_x_half[i] * vyx;
            vyx = vyx / K_x_half[i] + psi_vyx(k,j,i);
            psi_vzx(k,j,i) = b_x_half[i] * psi_vzx(k,j,i) + a_x_half[i] * vzx;
            vzx = vzx / K_x_half[i] + psi_vzx(k,j,i);
            
            
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            
            i =gidx - NX+NAB+FDOH+NAB;
            j =gidy-FDOH;
            k =gidz-FDOH;
            ind=2*NAB-1-i;
            
            
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
#if LVE==0
        
        fipjp=uipjp(gidz,gidy,gidx);
        fjpkp=ujpkp(gidz,gidy,gidx);
        fipkp=uipkp(gidz,gidy,gidx);
        g=pi(gidz,gidy,gidx);
        f=2.0*u(gidz,gidy,gidx);
        
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
        
        for (l=0;l<LVE;l++){
            leta[l]=eta[l];
        }
        
        fipjp=luipjp*(1.0+ (float)LVE*ltausipjp);
        fjpkp=lujpkp*(1.0+ (float)LVE*ltausjpkp);
        fipkp=luipkp*(1.0+ (float)LVE*ltausipkp);
        g=lpi*(1.0+(float)LVE*ltaup);
        f=2.0*lu*(1.0+(float)LVE*ltaus);
        dipjp=luipjp*ltausipjp;
        djpkp=lujpkp*ltausjpkp;
        dipkp=luipkp*ltausipkp;
        d=2.0*lu*ltaus;
        e=lpi*ltaup;
        
        
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
        
        sxy(gidz,gidy,gidx)+=(fipjp*vxyyx);
        syz(gidz,gidy,gidx)+=(fjpkp*vyzzy);
        sxz(gidz,gidy,gidx)+=(fipkp*vxzzx);
        sxx(gidz,gidy,gidx)+=((g*vxxyyzz)-(f*vyyzz)) + amp;
        syy(gidz,gidy,gidx)+=((g*vxxyyzz)-(f*vxxzz)) + amp;
        szz(gidz,gidy,gidx)+=((g*vxxyyzz)-(f*vxxyy)) + amp;
        
        
#else
        
        /* computing sums of the old memory variables */
        sumrxy=sumryz=sumrxz=sumrxx=sumryy=sumrzz=0;
        for (l=0;l<LVE;l++){
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
        sxy(gidz,gidy,gidx)+=lsxy+(DT2*sumrxy);
        syz(gidz,gidy,gidx)+=lsyz+(DT2*sumryz);
        sxz(gidz,gidy,gidx)+=lsxz+(DT2*sumrxz);
        sxx(gidz,gidy,gidx)+=lsxx+(DT2*sumrxx)+amp;
        syy(gidz,gidy,gidx)+=lsyy+(DT2*sumryy)+amp;
        szz(gidz,gidy,gidx)+=lszz+(DT2*sumrzz)+amp;
        
        
#endif
    }

// Absorbing boundary    
#if abstype==2
    {
        
#if FREESURF==0
        if (gidz-FDOH<NAB){
            sxy(gidz,gidy,gidx)*=taper[gidz-FDOH];
            syz(gidz,gidy,gidx)*=taper[gidz-FDOH];
            sxz(gidz,gidy,gidx)*=taper[gidz-FDOH];
            sxx(gidz,gidy,gidx)*=taper[gidz-FDOH];
            syy(gidz,gidy,gidx)*=taper[gidz-FDOH];
            szz(gidz,gidy,gidx)*=taper[gidz-FDOH];
        }
#endif
        
        if (gidz>NZ-NAB-FDOH-1){
            sxy(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
            syz(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
            sxz(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
            sxx(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
            syy(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
            szz(gidz,gidy,gidx)*=taper[NZ-FDOH-gidz-1];
        }
        if (gidy-FDOH<NAB){
            sxy(gidz,gidy,gidx)*=taper[gidy-FDOH];
            syz(gidz,gidy,gidx)*=taper[gidy-FDOH];
            sxz(gidz,gidy,gidx)*=taper[gidy-FDOH];
            sxx(gidz,gidy,gidx)*=taper[gidy-FDOH];
            syy(gidz,gidy,gidx)*=taper[gidy-FDOH];
            szz(gidz,gidy,gidx)*=taper[gidy-FDOH];
        }
        
        if (gidy>NY-NAB-FDOH-1){
            sxy(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
            syz(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
            sxz(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
            sxx(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
            syy(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
            szz(gidz,gidy,gidx)*=taper[NY-FDOH-gidy-1];
        }
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            sxy(gidz,gidy,gidx)*=taper[gidx-FDOH];
            syz(gidz,gidy,gidx)*=taper[gidx-FDOH];
            sxz(gidz,gidy,gidx)*=taper[gidx-FDOH];
            sxx(gidz,gidy,gidx)*=taper[gidx-FDOH];
            syy(gidz,gidy,gidx)*=taper[gidx-FDOH];
            szz(gidz,gidy,gidx)*=taper[gidx-FDOH];
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            sxy(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
            syz(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
            sxz(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
            sxx(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
            syy(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
            szz(gidz,gidy,gidx)*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif
    
}

