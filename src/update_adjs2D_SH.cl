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

/*Adjoint update of the stresses in 2D SH*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#define rho(z,x)    rho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,x)    rip[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rjp(z,x)    rjp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,x)    rkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define muipkp(z,x) muipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define mujpkp(z,x) mujpkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define muipjp(z,x) muipjp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
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

#define vx_r(z,x)  vx_r[(x)*(NZ)+(z)]
#define vy_r(z,x)  vy_r[(x)*(NZ)+(z)]
#define vz_r(z,x)  vz_r[(x)*(NZ)+(z)]
#define sxx_r(z,x) sxx_r[(x)*(NZ)+(z)]
#define szz_r(z,x) szz_r[(x)*(NZ)+(z)]
#define sxz_r(z,x) sxz_r[(x)*(NZ)+(z)]
#define sxy_r(z,x) sxy_r[(x)*(NZ)+(z)]
#define syz_r(z,x) syz_r[(x)*(NZ)+(z)]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxy(z,x,l) rxy[(l)*NX*NZ+(x)*NZ+(z)]
#define ryz(z,x,l) ryz[(l)*NX*NZ+(x)*NZ+(z)]

#define rxx_r(z,x,l) rxx_r[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz_r(z,x,l) rzz_r[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz_r(z,x,l) rxz_r[(l)*NX*NZ+(x)*NZ+(z)]
#define rxy_r(z,x,l) rxy_r[(l)*NX*NZ+(x)*NZ+(z)]
#define ryz_r(z,x,l) ryz_r[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_sxx_x(z,x) psi_sxx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_x(z,x) psi_sxz_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_sxz_z(z,x) psi_sxz_z[(x)*(2*NAB)+(z)]
#define psi_szz_z(z,x) psi_szz_z[(x)*(2*NAB)+(z)]
#define psi_sxy_x(z,x) psi_sxy_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_syz_z(z,x) psi_syz_z[(x)*(2*NAB)+(z)]

#define psi_vy_x(z,x) psi_vy_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_vy_z(z,x) psi_vy_z[(x)*(2*NAB)+(z)]

#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vy0(y,x) vy0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define ry(y,x) ry[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#if LOCAL_OFF==0

#define lvar(z,x)  lvar[(x)*lsizez+(z)]

#endif


#define PI (3.141592653589793238462643383279502884197169)
#define signals(y,x) signals[(y)*NT+(x)]
#define rec_pos(y,x) rec_pos[(y)*8+(x)]
#define gradsrc(y,x) gradsrc[(y)*NT+(x)]

// Find boundary indice for boundary injection in backpropagation
LFUNDEF int evarm( int k, int i){
    
    
#if NUM_DEVICES==1 & NLOCALP==1
    
    int NXbnd = (NX-2*FDOH-2*NAB);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH- 2*NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH- NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif
    
    int m=-1;
    i-=lbnd;
    k-=lbnds;
    
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
        m=NZbnd*FDOH*2+(NXbnd-2*FDOH)*FDOH+i+k*(NXbnd-2.0*FDOH);
    }
    else {//down
        i=i-FDOH;
        k=k-NZbnd+FDOH;
        m=NZbnd*FDOH*2+i+k*(NXbnd-2.0*FDOH);
    }
    
    
    
#elif DEVID==0 & MYLOCALID==0
    
    int NXbnd = (NX-2*FDOH-NAB);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH- 2*NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH- NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif
    
    int m=-1;
    i-=lbnd;
    k-=lbnds;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH)  && i>FDOH-1  )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (i<FDOH){//front
        m=i*NZbnd+k;
    }
    else if (k<FDOH){//up
        i=i-FDOH;
        m=NZbnd*FDOH+(NXbnd-FDOH)*FDOH+i+k*(NXbnd-FDOH);
        
    }
    else {//down
        i=i-FDOH;
        k=k-NZbnd+FDOH;
        m=NZbnd*FDOH+i+k*(NXbnd-FDOH);
    }
    
#elif DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    int NXbnd = (NX-2*FDOH-NAB);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH- 2*NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH- NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif
    
    int m=-1;
    i-=FDOH;
    k-=lbnds;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH) && i<NXbnd-FDOH )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i>NXbnd-1 )
        m=-1;
    else if (i>NXbnd-1-FDOH){
        i=i-NXbnd+FDOH;
        m=i*NZbnd+k;
    }
    else if (k<FDOH){//up
        m=NZbnd*FDOH+(NXbnd-FDOH)*FDOH+i+k*(NXbnd-FDOH);
    }
    else {//down
        k=k-NZbnd+FDOH;
        m=NZbnd*FDOH+i+k*(NXbnd-FDOH);
    }
    
#else
    
    int NXbnd = (NX-2*FDOH);
#if FREESURF==0
    int NZbnd = (NZ- 2*FDOH- 2*NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH+NAB;
#else
    int NZbnd = (NZ- 2*FDOH- NAB);
    int lbnd = FDOH+NAB;
    int lbnds = FDOH;
#endif
    
    int m=-1;
    i-=FDOH;
    k-=lbnds;
    
    if ( (k>FDOH-1 && k<NZbnd-FDOH) )
        m=-1;
    else if (k<0 || k>NZbnd-1 || i<0 || i>NXbnd-1 )
        m=-1;
    else if (k<FDOH){//up
        m=(NXbnd)*FDOH+i+k*(NXbnd);
    }
    else {//down
        k=k-NZbnd+FDOH;
        m=i+k*(NXbnd);
    }
    
    
#endif
    
    
    return m;
    
}

__kernel void update_adjs(int offcomm, 
                          __global float *vy,
                          __global float *sxy,        __global float *syz,
                          __global float *vybnd,
                          __global float *sxybnd,     __global float *syzbnd,
                          __global float *vy_r,
                          __global float *sxy_r,      __global float *syz_r,
                          __global float *rxy,        __global float *ryz,
                          __global float *rxy_r,      __global float *ryz_r,
                          __global float *muipjp,      __global float *mujpkp,
                          __global float *tausipjp,   __global float *tausjpkp,
                          __global float *eta,        __global float *taper,
                          __global float *K_x,        __global float *a_x,      __global float *b_x,
                          __global float *K_x_half,   __global float *a_x_half, __global float *b_x_half,
                          __global float *K_z,        __global float *a_z,      __global float *b_z,
                          __global float *K_z_half,   __global float *a_z_half, __global float *b_z_half,
                          __global float *psi_vy_x,    __global float *psi_vy_z,
                          __global float *gradrho,    __global float *gradmu,   __global float *gradsrc,
                          __global float *Hrho,    __global float *Hmu,
                          __local  float *lvar)
{

    int i,k,l,ind;
    float fipjp, fjpkp;
    float sumrxy,sumryz;
    float b,c,dipjp,djpkp;
    float lmuipjp, lmujpkp, ltausipjp, ltausjpkp;
#if LVE>0
    float leta[LVE];
#endif
    float vyx,vyz;
    float vxyyx,vyzzy,vxzzx,vxxyyzz,vyyzz,vxxzz,vxxyy;
    float vyx_r,vyz_r;
    float vxyyx_r,vyzzy_r,vxzzx_r,vxxyyzz_r,vyyzz_r,vxxzz_r,vxxyy_r;
    
    float lsxy, lsyz;
    
    
// If we use local memory
#if LOCAL_OFF==0
    int lsizez = get_local_size(0)+2*FDOH;
    int lsizex = get_local_size(1)+2*FDOH;
    int lidz = get_local_id(0)+FDOH;
    int lidx = get_local_id(1)+FDOH;
    int gidz = get_global_id(0)+FDOH;
    int gidx = get_global_id(1)+FDOH+offcomm;
    
#define lvy lvar
#define lvy_r lvar
    
// If local memory is turned off
#elif LOCAL_OFF==1
    
    int gid = get_global_id(0);
    int glsizez = (NZ-2*FDOH);
    int gidz = gid%glsizez+FDOH;
    int gidx = (gid/glsizez)+FDOH+offcomm;
    
    
#define lvy vy
#define lvy_r vy_r
#define lidx gidx
#define lidz gidz
    
#define lsizez NZ
#define lsizex NX
    
#endif
    

// Calculation of the velocity spatial derivatives of the forward wavefield if backpropagation is used
#if BACK_PROP_TYPE==1
    {
#if LOCAL_OFF==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lvy(lidz,lidx)=vy(gidz,  gidx);
        if (lidx<2*FDOH)
            lvy(lidz,lidx-FDOH)=vy(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvy(lidz,lidx+lsizex-3*FDOH)=vy(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvy(lidz,lidx+FDOH)=vy(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvy(lidz,lidx-lsizex+3*FDOH)=vy(gidz,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvy(lidz-FDOH,lidx)=vy(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvy(lidz+FDOH,lidx)=vy(gidz+FDOH,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   FDOH==1
        vyx = (lvy(lidz,lidx+1)-lvy(lidz,lidx));
        vyz = (lvy(lidz+1,lidx)-lvy(lidz,lidx));
#elif FDOH==2
        vyx = (HC1*(lvy(lidz,lidx+1)-lvy(lidz,lidx))+
               HC2*(lvy(lidz,lidx+2)-lvy(lidz,lidx-1)));
        
        vyz = (HC1*(lvy(lidz+1,lidx)-lvy(lidz,lidx))+
               HC2*(lvy(lidz+2,lidx)-lvy(lidz-1,lidx)));
#elif FDOH==3
        vyx = (HC1*(lvy(lidz,lidx+1)-lvy(lidz,lidx))+
               HC2*(lvy(lidz,lidx+2)-lvy(lidz,lidx-1))+
               HC3*(lvy(lidz,lidx+3)-lvy(lidz,lidx-2)));
        
        vyz = (HC1*(lvy(lidz+1,lidx)-lvy(lidz,lidx))+
               HC2*(lvy(lidz+2,lidx)-lvy(lidz-1,lidx))+
               HC3*(lvy(lidz+3,lidx)-lvy(lidz-2,lidx)));
#elif FDOH==4
        vyx = (HC1*(lvy(lidz,lidx+1)-lvy(lidz,lidx))+
               HC2*(lvy(lidz,lidx+2)-lvy(lidz,lidx-1))+
               HC3*(lvy(lidz,lidx+3)-lvy(lidz,lidx-2))+
               HC4*(lvy(lidz,lidx+4)-lvy(lidz,lidx-3)));
        
        vyz = (HC1*(lvy(lidz+1,lidx)-lvy(lidz,lidx))+
               HC2*(lvy(lidz+2,lidx)-lvy(lidz-1,lidx))+
               HC3*(lvy(lidz+3,lidx)-lvy(lidz-2,lidx))+
               HC4*(lvy(lidz+4,lidx)-lvy(lidz-3,lidx)));
#elif FDOH==5
        vyx = (HC1*(lvy(lidz,lidx+1)-lvy(lidz,lidx))+
               HC2*(lvy(lidz,lidx+2)-lvy(lidz,lidx-1))+
               HC3*(lvy(lidz,lidx+3)-lvy(lidz,lidx-2))+
               HC4*(lvy(lidz,lidx+4)-lvy(lidz,lidx-3))+
               HC5*(lvy(lidz,lidx+5)-lvy(lidz,lidx-4)));
        
        vyz = (HC1*(lvy(lidz+1,lidx)-lvy(lidz,lidx))+
               HC2*(lvy(lidz+2,lidx)-lvy(lidz-1,lidx))+
               HC3*(lvy(lidz+3,lidx)-lvy(lidz-2,lidx))+
               HC4*(lvy(lidz+4,lidx)-lvy(lidz-3,lidx))+
               HC5*(lvy(lidz+5,lidx)-lvy(lidz-4,lidx)));
#elif FDOH==6
        vyx = (HC1*(lvy(lidz,lidx+1)-lvy(lidz,lidx))+
               HC2*(lvy(lidz,lidx+2)-lvy(lidz,lidx-1))+
               HC3*(lvy(lidz,lidx+3)-lvy(lidz,lidx-2))+
               HC4*(lvy(lidz,lidx+4)-lvy(lidz,lidx-3))+
               HC5*(lvy(lidz,lidx+5)-lvy(lidz,lidx-4))+
               HC6*(lvy(lidz,lidx+6)-lvy(lidz,lidx-5)));
        
        vyz = (HC1*(lvy(lidz+1,lidx)-lvy(lidz,lidx))+
               HC2*(lvy(lidz+2,lidx)-lvy(lidz-1,lidx))+
               HC3*(lvy(lidz+3,lidx)-lvy(lidz-2,lidx))+
               HC4*(lvy(lidz+4,lidx)-lvy(lidz-3,lidx))+
               HC5*(lvy(lidz+5,lidx)-lvy(lidz-4,lidx))+
               HC6*(lvy(lidz+6,lidx)-lvy(lidz-5,lidx)));
#endif

    }
#endif
    
// Calculation of the velocity spatial derivatives of the adjoint wavefield
    {
#if LOCAL_OFF==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lvy_r(lidz,lidx)=vy_r(gidz,  gidx);
        if (lidx<2*FDOH)
            lvy_r(lidz,lidx-FDOH)=vy_r(gidz,gidx-FDOH);
        if (lidx+lsizex-3*FDOH<FDOH)
            lvy_r(lidz,lidx+lsizex-3*FDOH)=vy_r(gidz,gidx+lsizex-3*FDOH);
        if (lidx>(lsizex-2*FDOH-1))
            lvy_r(lidz,lidx+FDOH)=vy_r(gidz,gidx+FDOH);
        if (lidx-lsizex+3*FDOH>(lsizex-FDOH-1))
            lvy_r(lidz,lidx-lsizex+3*FDOH)=vy_r(gidz,gidx-lsizex+3*FDOH);
        if (lidz<2*FDOH)
            lvy_r(lidz-FDOH,lidx)=vy_r(gidz-FDOH,gidx);
        if (lidz>(lsizez-2*FDOH-1))
            lvy_r(lidz+FDOH,lidx)=vy_r(gidz+FDOH,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   FDOH==1
        vyx_r = (lvy_r(lidz,lidx+1)-lvy_r(lidz,lidx));
        vyz_r = (lvy_r(lidz+1,lidx)-lvy_r(lidz,lidx));
#elif FDOH==2
        vyx_r = (HC1*(lvy_r(lidz,lidx+1)-lvy_r(lidz,lidx))+
               HC2*(lvy_r(lidz,lidx+2)-lvy_r(lidz,lidx-1)));
        
        vyz_r = (HC1*(lvy_r(lidz+1,lidx)-lvy_r(lidz,lidx))+
               HC2*(lvy_r(lidz+2,lidx)-lvy_r(lidz-1,lidx)));
#elif FDOH==3
        vyx_r = (HC1*(lvy_r(lidz,lidx+1)-lvy_r(lidz,lidx))+
               HC2*(lvy_r(lidz,lidx+2)-lvy_r(lidz,lidx-1))+
               HC3*(lvy_r(lidz,lidx+3)-lvy_r(lidz,lidx-2)));
        
        vyz_r = (HC1*(lvy_r(lidz+1,lidx)-lvy_r(lidz,lidx))+
               HC2*(lvy_r(lidz+2,lidx)-lvy_r(lidz-1,lidx))+
               HC3*(lvy_r(lidz+3,lidx)-lvy_r(lidz-2,lidx)));
#elif FDOH==4
        vyx_r = (HC1*(lvy_r(lidz,lidx+1)-lvy_r(lidz,lidx))+
               HC2*(lvy_r(lidz,lidx+2)-lvy_r(lidz,lidx-1))+
               HC3*(lvy_r(lidz,lidx+3)-lvy_r(lidz,lidx-2))+
               HC4*(lvy_r(lidz,lidx+4)-lvy_r(lidz,lidx-3)));
        
        vyz_r = (HC1*(lvy_r(lidz+1,lidx)-lvy_r(lidz,lidx))+
               HC2*(lvy_r(lidz+2,lidx)-lvy_r(lidz-1,lidx))+
               HC3*(lvy_r(lidz+3,lidx)-lvy_r(lidz-2,lidx))+
               HC4*(lvy_r(lidz+4,lidx)-lvy_r(lidz-3,lidx)));
#elif FDOH==5
        vyx_r = (HC1*(lvy_r(lidz,lidx+1)-lvy_r(lidz,lidx))+
               HC2*(lvy_r(lidz,lidx+2)-lvy_r(lidz,lidx-1))+
               HC3*(lvy_r(lidz,lidx+3)-lvy_r(lidz,lidx-2))+
               HC4*(lvy_r(lidz,lidx+4)-lvy_r(lidz,lidx-3))+
               HC5*(lvy_r(lidz,lidx+5)-lvy_r(lidz,lidx-4)));
        
        vyz_r = (HC1*(lvy_r(lidz+1,lidx)-lvy_r(lidz,lidx))+
               HC2*(lvy_r(lidz+2,lidx)-lvy_r(lidz-1,lidx))+
               HC3*(lvy_r(lidz+3,lidx)-lvy_r(lidz-2,lidx))+
               HC4*(lvy_r(lidz+4,lidx)-lvy_r(lidz-3,lidx))+
               HC5*(lvy_r(lidz+5,lidx)-lvy_r(lidz-4,lidx)));
#elif FDOH==6
        vyx_r = (HC1*(lvy_r(lidz,lidx+1)-lvy_r(lidz,lidx))+
               HC2*(lvy_r(lidz,lidx+2)-lvy_r(lidz,lidx-1))+
               HC3*(lvy_r(lidz,lidx+3)-lvy_r(lidz,lidx-2))+
               HC4*(lvy_r(lidz,lidx+4)-lvy_r(lidz,lidx-3))+
               HC5*(lvy_r(lidz,lidx+5)-lvy_r(lidz,lidx-4))+
               HC6*(lvy_r(lidz,lidx+6)-lvy_r(lidz,lidx-5)));
        
        vyz_r = (HC1*(lvy_r(lidz+1,lidx)-lvy_r(lidz,lidx))+
               HC2*(lvy_r(lidz+2,lidx)-lvy_r(lidz-1,lidx))+
               HC3*(lvy_r(lidz+3,lidx)-lvy_r(lidz-2,lidx))+
               HC4*(lvy_r(lidz+4,lidx)-lvy_r(lidz-3,lidx))+
               HC5*(lvy_r(lidz+5,lidx)-lvy_r(lidz-4,lidx))+
               HC6*(lvy_r(lidz+6,lidx)-lvy_r(lidz-5,lidx)));
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
    
    fipjp=muipjp(gidz,gidx);
    fjpkp=mujpkp(gidz,gidx);
    
#else
    
    lmuipjp=muipjp(gidz,gidx);
    lmujpkp=mujpkp(gidz,gidx);
    ltausipjp=tausipjp(gidz,gidx);
    ltausjpkp=tausjpkp(gidz,gidx);
    
    for (l=0;l<LVE;l++){
        leta[l]=eta[l];
    }
    
    fipjp=lmuipjp*(1.0+ (float)LVE*ltausipjp);
    fjpkp=lmujpkp*(1.0+ (float)LVE*ltausjpkp);
    dipjp=lmuipjp*ltausipjp;
    djpkp=lmujpkp*ltausjpkp;
    
#endif
    
// Backpropagate the forward stresses
#if BACK_PROP_TYPE==1
    {
#if LVE==0
        
        sxy(gidz,gidx)-=(fipjp*vyx);
        syz(gidz,gidx)-=(fjpkp*vyz);
        
// Backpropagation is not stable for viscoelastic wave equation
#else
        
        /* computing sums of the old memory variables */
        sumrxy=sumryz=0;
        for (l=0;l<LVE;l++){
            sumrxy+=rxy(gidz,gidx,l);
            sumryz+=ryz(gidz,gidx,l);
        }
        
        /* updating components of the stress tensor, partially */
        
        lsxy=(fipjp*vxyyx)+(DT2*sumrxy);
        lsyz=(fjpkp*vyzzy)+(DT2*sumryz);
        
        sumrxy=sumryz=0;
        for (l=0;l<LVE;l++){
            b=1.0/(1.0+(leta[l]*0.5));
            c=1.0-(leta[l]*0.5);
            
            rxy(gidz,gidx,l)=b*(rxy(gidz,gidx,l)*c-leta[l]*(dipjp*vxyyx));
            ryz(gidz,gidx,l)=b*(ryz(gidz,gidx,l)*c-leta[l]*(djpkp*vyzzy));
            
            sumrxy+=rxy(gidz,gidx,l);
            sumryz+=ryz(gidz,gidx,l);
        }
        
        /* and now the components of the stress tensor are
         completely updated */
        sxy(gidz,gidx)-=lsxy+(DT2*sumrxy);
        syz(gidz,gidx)-=lsyz+(DT2*sumryz);
        
        
#endif
        
        m=evarm(gidz,  gidx);
        if (m!=-1){
            sxy(gidz, gidx)= sxybnd[m];
            syz(gidz, gidx)= szzbnd[m];
        }
    }
#endif
    
// Correct adjoint spatial derivatives to implement CPML
#if ABS_TYPE==1
    {
        
        
        if (gidz>NZ-NAB-FDOH-1){
            
            i =gidx-FDOH;
            k =gidz - NZ+NAB+FDOH+NAB;
            ind=2*NAB-1-k;
            
            psi_vy_z(k,i) = b_z_half[ind] * psi_vy_z(k,i) + a_z_half[ind] * vyz_r;
            vyz_r = vyz_r / K_z_half[ind] + psi_vy_z(k,i);
        }
        
#if FREESURF==0
        else if (gidz-FDOH<NAB){
            
            i =gidx-FDOH;
            k =gidz-FDOH;
            
            psi_vy_z(k,i) = b_z_half[k] * psi_vy_z(k,i) + a_z_half[k] * vyz_r;
            vyz_r = vyz_r / K_z_half[k] + psi_vy_z(k,i);
            
        }
#endif
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            
            i =gidx-FDOH;
            k =gidz-FDOH;
            
            psi_vy_x(k,i) = b_x_half[i] * psi_vy_x(k,i) + a_x_half[i] * vyx_r;
            vyx_r = vyx_r / K_x_half[i] + psi_vy_x(k,i);
            
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            
            i =gidx - NX+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-i;
            
            psi_vy_x(k,i) = b_x_half[ind] * psi_vy_x(k,i) + a_x_half[ind] * vyx_r;
            vyx_r = vyx_r  /K_x_half[ind] + psi_vy_x(k,i);
        }
#endif

    }
#endif

// Update adjoint stresses
    {
#if LVE==0
    sxy_r(gidz,gidx)+=(fipjp*vyx_r);
    syz_r(gidz,gidx)+=(fjpkp*vyz_r);
    
    
#else
    
        /* computing sums of the old memory variables */
        sumrxy=sumryz=0;
        for (l=0;l<LVE;l++){
            sumrxy+=rxy_r(gidz,gidx,l);
            sumryz+=ryz_r(gidz,gidx,l);
        }
        
        /* updating components of the stress tensor, partially */
        
        lsxy=(fipjp*vyx_r)+(DT2*sumrxy);
        lsyz=(fjpkp*vyz_r)+(DT2*sumryz);
        
        sumrxy=sumryz=0;
        for (l=0;l<LVE;l++){
            b=1.0/(1.0+(leta[l]*0.5));
            c=1.0-(leta[l]*0.5);
            
            rxy_r(gidz,gidx,l)=b*(rxy_r(gidz,gidx,l)*c-leta[l]*(dipjp*vyx_r));
            ryz_r(gidz,gidx,l)=b*(ryz_r(gidz,gidx,l)*c-leta[l]*(djpkp*vyz_r));
            
            sumrxy+=rxy_r(gidz,gidx,l);
            sumryz+=ryz_r(gidz,gidx,l);
        }
        
        /* and now the components of the stress tensor are
         completely updated */
        sxy_r(gidz,gidx)+=lsxy+(DT2*sumrxy);
        syz_r(gidz,gidx)+=lsyz+(DT2*sumryz);
#endif
}
    
// Absorbing boundary
#if ABS_TYPE==2
    {
#if FREESURF==0
        if (gidz-FDOH<NAB){
            sxy_r(gidz,gidx)*=taper[gidz-FDOH];
            syz_r(gidz,gidx)*=taper[gidz-FDOH];
        }
#endif
        
        if (gidz>NZ-NAB-FDOH-1){
            sxy_r(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
            syz_r(gidz,gidx)*=taper[NZ-FDOH-gidz-1];
        }
        
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            sxy_r(gidz,gidx)*=taper[gidx-FDOH];
            syz_r(gidz,gidx)*=taper[gidx-FDOH];
        }
#endif
        
#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            sxy_r(gidz,gidx)*=taper[NX-FDOH-gidx-1];
            syz_r(gidz,gidx)*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif
    
// Shear wave modulus gradient calculation on the fly    
#if BACK_PROP_TYPE==1

    gradmu(gidz,gidy,gidx)+=-(sxy(gidz,gidx)*fipjp*vyx_r+syz(gidz,gidx)*fipjp*vyz_r)/(pown( (fipjp/DT+fjpkp/DT)/2.0,2));
#endif
    


}

