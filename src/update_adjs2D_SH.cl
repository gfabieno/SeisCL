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
#define lbnd (fdoh+nab)

#define rho(z,x)    rho[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rip(z,x)    rip[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rjp(z,x)    rjp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rkp(z,x)    rkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define uipkp(z,x) uipkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define ujpkp(z,x) ujpkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define uipjp(z,x) uipjp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define u(z,x)        u[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define pi(z,x)      pi[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradrho(z,x)  gradrho[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradM(z,x)  gradM[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradmu(z,x)  gradmu[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradtaup(z,x)  gradtaup[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define gradtaus(z,x)  gradtaus[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define taus(z,x)        taus[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausipkp(z,x) tausipkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausipjp(z,x) tausipjp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausjpkp(z,x) tausjpkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define taup(z,x)        taup[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define vx(z,x)  vx[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vy(z,x)  vy[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vz(z,x)  vz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxx(z,x) sxx[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define szz(z,x) szz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxz(z,x) sxz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxy(z,x) sxy[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define syz(z,x) syz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]

#define vx_r(z,x)  vx_r[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vy_r(z,x)  vy_r[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vz_r(z,x)  vz_r[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxx_r(z,x) sxx_r[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define szz_r(z,x) szz_r[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxz_r(z,x) sxz_r[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxy_r(z,x) sxy_r[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define syz_r(z,x) syz_r[(x)*(NZ+NZ_al16)+(z)+NZ_al0]

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

#define psi_sxx_x(z,x) psi_sxx_x[(x)*(NZ-2*fdoh)+(z)]
#define psi_sxz_x(z,x) psi_sxz_x[(x)*(NZ-2*fdoh)+(z)]
#define psi_sxz_z(z,x) psi_sxz_z[(x)*(2*nab)+(z)]
#define psi_szz_z(z,x) psi_szz_z[(x)*(2*nab)+(z)]
#define psi_sxy_x(z,x) psi_sxy_x[(x)*(NZ-2*fdoh)+(z)]
#define psi_syz_z(z,x) psi_syz_z[(x)*(2*nab)+(z)]

#define psi_vyx(z,x) psi_vyx[(x)*(NZ-2*fdoh)+(z)]
#define psi_vyz(z,x) psi_vyz[(x)*(2*nab)+(z)]

#define vxout(y,x) vxout[(y)*NT+(x)]
#define vyout(y,x) vyout[(y)*NT+(x)]
#define vzout(y,x) vzout[(y)*NT+(x)]
#define vx0(y,x) vx0[(y)*NT+(x)]
#define vy0(y,x) vy0[(y)*NT+(x)]
#define vz0(y,x) vz0[(y)*NT+(x)]
#define rx(y,x) rx[(y)*NT+(x)]
#define ry(y,x) ry[(y)*NT+(x)]
#define rz(y,x) rz[(y)*NT+(x)]

#if local_off==0

#define lvar(z,x)  lvar[(x)*lsizez+(z)]

#endif


#define PI (3.141592653589793238462643383279502884197169)
#define srcpos_loc(y,x) srcpos_loc[(y)*nsrc+(x)]
#define signals(y,x) signals[(y)*NT+(x)]
#define rec_pos(y,x) rec_pos[(y)*8+(x)]
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
                          __global float *vy,
                          __global float *sxy,        __global float *syz,
                          __global float *vybnd,
                          __global float *sxybnd,     __global float *syzbnd,
                          __global float *vy_r,
                          __global float *sxy_r,      __global float *syz_r,
                          __global float *rxy,        __global float *ryz,
                          __global float *rxy_r,      __global float *ryz_r,
                          __global float *uipjp,      __global float *ujpkp,
                          __global float *tausipjp,   __global float *tausjpkp,
                          __global float *eta,
                          __global float *srcpos_loc, __global float *signals,  __global float *taper,
                          __global float *K_x,        __global float *a_x,      __global float *b_x,
                          __global float *K_x_half,   __global float *a_x_half, __global float *b_x_half,
                          __global float *K_z,        __global float *a_z,      __global float *b_z,
                          __global float *K_z_half,   __global float *a_z_half, __global float *b_z_half,
                          __global float *psi_vyx,    __global float *psi_vyz,
                          __global float *gradrho,    __global float *gradmu,   __global float *gradsrc,
                          __local  float *lvar)
{

    int i,k,l,ind;
    float fipjp, fjpkp;
    float sumrxy,sumryz;
    float b,c,dipjp,djpkp;
    float luipjp, lujpkp, ltausipjp, ltausjpkp;
#if Lve>0
    float leta[Lve];
#endif
    float vyx,vyz;
    float vxyyx,vyzzy,vxzzx,vxxyyzz,vyyzz,vxxzz,vxxyy;
    float vyx_r,vyz_r;
    float vxyyx_r,vyzzy_r,vxzzx_r,vxxyyzz_r,vyyzz_r,vxxzz_r,vxxyy_r;
    
    float lsxy, lsyz;
    
    
// If we use local memory
#if local_off==0
    int lsizez = get_local_size(0)+2*fdoh;
    int lsizex = get_local_size(1)+2*fdoh;
    int lidz = get_local_id(0)+fdoh;
    int lidx = get_local_id(1)+fdoh;
    int gidz = get_global_id(0)+fdoh;
    int gidx = get_global_id(1)+fdoh+offcomm;
    
#define lvy lvar
#define lvy_r lvar
    
// If local memory is turned off
#elif local_off==1
    
    int gid = get_global_id(0);
    int glsizez = (NZ-2*fdoh);
    int gidz = gid%glsizez+fdoh;
    int gidx = (gid/glsizez)+fdoh+offcomm;
    
    
#define lvy vy
#define lvy_r vy_r
#define lidx gidx
#define lidz gidz
    
#define lsizez NZ
#define lsizex NX
    
#endif
    

// Calculation of the velocity spatial derivatives of the forward wavefield if backpropagation is used
#if back_prop_type==1
    {
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lvy(lidz,lidx)=vy(gidz,  gidx);
        if (lidx<2*fdoh)
            lvy(lidz,lidx-fdoh)=vy(gidz,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvy(lidz,lidx+lsizex-3*fdoh)=vy(gidz,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvy(lidz,lidx+fdoh)=vy(gidz,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvy(lidz,lidx-lsizex+3*fdoh)=vy(gidz,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvy(lidz-fdoh,lidx)=vy(gidz-fdoh,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvy(lidz+fdoh,lidx)=vy(gidz+fdoh,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh==1
        vyx = (lvy(lidz,lidx+1)-lvy(lidz,lidx))/DH;
        vyz = (lvy(lidz+1,lidx)-lvy(lidz,lidx))/DH;
#elif fdoh==2
        vyx = (hc1*(lvy(lidz,lidx+1)-lvy(lidz,lidx))+
               hc2*(lvy(lidz,lidx+2)-lvy(lidz,lidx-1)))/DH;
        
        vyz = (hc1*(lvy(lidz+1,lidx)-lvy(lidz,lidx))+
               hc2*(lvy(lidz+2,lidx)-lvy(lidz-1,lidx)))/DH;
#elif fdoh==3
        vyx = (hc1*(lvy(lidz,lidx+1)-lvy(lidz,lidx))+
               hc2*(lvy(lidz,lidx+2)-lvy(lidz,lidx-1))+
               hc3*(lvy(lidz,lidx+3)-lvy(lidz,lidx-2)))/DH;
        
        vyz = (hc1*(lvy(lidz+1,lidx)-lvy(lidz,lidx))+
               hc2*(lvy(lidz+2,lidx)-lvy(lidz-1,lidx))+
               hc3*(lvy(lidz+3,lidx)-lvy(lidz-2,lidx)))/DH;
#elif fdoh==4
        vyx = (hc1*(lvy(lidz,lidx+1)-lvy(lidz,lidx))+
               hc2*(lvy(lidz,lidx+2)-lvy(lidz,lidx-1))+
               hc3*(lvy(lidz,lidx+3)-lvy(lidz,lidx-2))+
               hc4*(lvy(lidz,lidx+4)-lvy(lidz,lidx-3)))/DH;
        
        vyz = (hc1*(lvy(lidz+1,lidx)-lvy(lidz,lidx))+
               hc2*(lvy(lidz+2,lidx)-lvy(lidz-1,lidx))+
               hc3*(lvy(lidz+3,lidx)-lvy(lidz-2,lidx))+
               hc4*(lvy(lidz+4,lidx)-lvy(lidz-3,lidx)))/DH;
#elif fdoh==5
        vyx = (hc1*(lvy(lidz,lidx+1)-lvy(lidz,lidx))+
               hc2*(lvy(lidz,lidx+2)-lvy(lidz,lidx-1))+
               hc3*(lvy(lidz,lidx+3)-lvy(lidz,lidx-2))+
               hc4*(lvy(lidz,lidx+4)-lvy(lidz,lidx-3))+
               hc5*(lvy(lidz,lidx+5)-lvy(lidz,lidx-4)))/DH;
        
        vyz = (hc1*(lvy(lidz+1,lidx)-lvy(lidz,lidx))+
               hc2*(lvy(lidz+2,lidx)-lvy(lidz-1,lidx))+
               hc3*(lvy(lidz+3,lidx)-lvy(lidz-2,lidx))+
               hc4*(lvy(lidz+4,lidx)-lvy(lidz-3,lidx))+
               hc5*(lvy(lidz+5,lidx)-lvy(lidz-4,lidx)))/DH;
#elif fdoh==6
        vyx = (hc1*(lvy(lidz,lidx+1)-lvy(lidz,lidx))+
               hc2*(lvy(lidz,lidx+2)-lvy(lidz,lidx-1))+
               hc3*(lvy(lidz,lidx+3)-lvy(lidz,lidx-2))+
               hc4*(lvy(lidz,lidx+4)-lvy(lidz,lidx-3))+
               hc5*(lvy(lidz,lidx+5)-lvy(lidz,lidx-4))+
               hc6*(lvy(lidz,lidx+6)-lvy(lidz,lidx-5)))/DH;
        
        vyz = (hc1*(lvy(lidz+1,lidx)-lvy(lidz,lidx))+
               hc2*(lvy(lidz+2,lidx)-lvy(lidz-1,lidx))+
               hc3*(lvy(lidz+3,lidx)-lvy(lidz-2,lidx))+
               hc4*(lvy(lidz+4,lidx)-lvy(lidz-3,lidx))+
               hc5*(lvy(lidz+5,lidx)-lvy(lidz-4,lidx))+
               hc6*(lvy(lidz+6,lidx)-lvy(lidz-5,lidx)))/DH;
#endif

    }
#endif
    
// Calculation of the velocity spatial derivatives of the adjoint wavefield
    {
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lvy_r(lidz,lidx)=vy_r(gidz,  gidx);
        if (lidx<2*fdoh)
            lvy_r(lidz,lidx-fdoh)=vy_r(gidz,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lvy_r(lidz,lidx+lsizex-3*fdoh)=vy_r(gidz,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lvy_r(lidz,lidx+fdoh)=vy_r(gidz,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lvy_r(lidz,lidx-lsizex+3*fdoh)=vy_r(gidz,gidx-lsizex+3*fdoh);
        if (lidz<2*fdoh)
            lvy_r(lidz-fdoh,lidx)=vy_r(gidz-fdoh,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lvy_r(lidz+fdoh,lidx)=vy_r(gidz+fdoh,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh==1
        vyx_r = (lvy_r(lidz,lidx+1)-lvy_r(lidz,lidx))/DH;
        vyz_r = (lvy_r(lidz+1,lidx)-lvy_r(lidz,lidx))/DH;
#elif fdoh==2
        vyx_r = (hc1*(lvy_r(lidz,lidx+1)-lvy_r(lidz,lidx))+
               hc2*(lvy_r(lidz,lidx+2)-lvy_r(lidz,lidx-1)))/DH;
        
        vyz_r = (hc1*(lvy_r(lidz+1,lidx)-lvy_r(lidz,lidx))+
               hc2*(lvy_r(lidz+2,lidx)-lvy_r(lidz-1,lidx)))/DH;
#elif fdoh==3
        vyx_r = (hc1*(lvy_r(lidz,lidx+1)-lvy_r(lidz,lidx))+
               hc2*(lvy_r(lidz,lidx+2)-lvy_r(lidz,lidx-1))+
               hc3*(lvy_r(lidz,lidx+3)-lvy_r(lidz,lidx-2)))/DH;
        
        vyz_r = (hc1*(lvy_r(lidz+1,lidx)-lvy_r(lidz,lidx))+
               hc2*(lvy_r(lidz+2,lidx)-lvy_r(lidz-1,lidx))+
               hc3*(lvy_r(lidz+3,lidx)-lvy_r(lidz-2,lidx)))/DH;
#elif fdoh==4
        vyx_r = (hc1*(lvy_r(lidz,lidx+1)-lvy_r(lidz,lidx))+
               hc2*(lvy_r(lidz,lidx+2)-lvy_r(lidz,lidx-1))+
               hc3*(lvy_r(lidz,lidx+3)-lvy_r(lidz,lidx-2))+
               hc4*(lvy_r(lidz,lidx+4)-lvy_r(lidz,lidx-3)))/DH;
        
        vyz_r = (hc1*(lvy_r(lidz+1,lidx)-lvy_r(lidz,lidx))+
               hc2*(lvy_r(lidz+2,lidx)-lvy_r(lidz-1,lidx))+
               hc3*(lvy_r(lidz+3,lidx)-lvy_r(lidz-2,lidx))+
               hc4*(lvy_r(lidz+4,lidx)-lvy_r(lidz-3,lidx)))/DH;
#elif fdoh==5
        vyx_r = (hc1*(lvy_r(lidz,lidx+1)-lvy_r(lidz,lidx))+
               hc2*(lvy_r(lidz,lidx+2)-lvy_r(lidz,lidx-1))+
               hc3*(lvy_r(lidz,lidx+3)-lvy_r(lidz,lidx-2))+
               hc4*(lvy_r(lidz,lidx+4)-lvy_r(lidz,lidx-3))+
               hc5*(lvy_r(lidz,lidx+5)-lvy_r(lidz,lidx-4)))/DH;
        
        vyz_r = (hc1*(lvy_r(lidz+1,lidx)-lvy_r(lidz,lidx))+
               hc2*(lvy_r(lidz+2,lidx)-lvy_r(lidz-1,lidx))+
               hc3*(lvy_r(lidz+3,lidx)-lvy_r(lidz-2,lidx))+
               hc4*(lvy_r(lidz+4,lidx)-lvy_r(lidz-3,lidx))+
               hc5*(lvy_r(lidz+5,lidx)-lvy_r(lidz-4,lidx)))/DH;
#elif fdoh==6
        vyx_r = (hc1*(lvy_r(lidz,lidx+1)-lvy_r(lidz,lidx))+
               hc2*(lvy_r(lidz,lidx+2)-lvy_r(lidz,lidx-1))+
               hc3*(lvy_r(lidz,lidx+3)-lvy_r(lidz,lidx-2))+
               hc4*(lvy_r(lidz,lidx+4)-lvy_r(lidz,lidx-3))+
               hc5*(lvy_r(lidz,lidx+5)-lvy_r(lidz,lidx-4))+
               hc6*(lvy_r(lidz,lidx+6)-lvy_r(lidz,lidx-5)))/DH;
        
        vyz_r = (hc1*(lvy_r(lidz+1,lidx)-lvy_r(lidz,lidx))+
               hc2*(lvy_r(lidz+2,lidx)-lvy_r(lidz-1,lidx))+
               hc3*(lvy_r(lidz+3,lidx)-lvy_r(lidz-2,lidx))+
               hc4*(lvy_r(lidz+4,lidx)-lvy_r(lidz-3,lidx))+
               hc5*(lvy_r(lidz+5,lidx)-lvy_r(lidz-4,lidx))+
               hc6*(lvy_r(lidz+6,lidx)-lvy_r(lidz-5,lidx)))/DH;
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
    
    fipjp=uipjp(gidz,gidx)*DT;
    fjpkp=ujpkp(gidz,gidx)*DT;
    
#else
    
    luipjp=uipjp(gidz,gidx);
    lujpkp=ujpkp(gidz,gidx);
    ltausipjp=tausipjp(gidz,gidx);
    ltausjpkp=tausjpkp(gidz,gidx);
    
    for (l=0;l<Lve;l++){
        leta[l]=eta[l];
    }
    
    fipjp=luipjp*DT*(1.0+ (float)Lve*ltausipjp);
    fjpkp=lujpkp*DT*(1.0+ (float)Lve*ltausjpkp);
    dipjp=luipjp*ltausipjp;
    djpkp=lujpkp*ltausjpkp;
    
#endif
    
// Backpropagate the forward stresses
#if back_prop_type==1
    {
#if Lve==0
        
        sxy(gidz,gidx)-=(fipjp*vyx);
        syz(gidz,gidx)-=(fjpkp*vyz);
        
// Backpropagation is not stable for viscoelastic wave equation
#else
        
        /* computing sums of the old memory variables */
        sumrxy=sumryz=0;
        for (l=0;l<Lve;l++){
            sumrxy+=rxy(gidz,gidx,l);
            sumryz+=ryz(gidz,gidx,l);
        }
        
        /* updating components of the stress tensor, partially */
        
        lsxy=(fipjp*vxyyx)+(dt2*sumrxy);
        lsyz=(fjpkp*vyzzy)+(dt2*sumryz);
        
        sumrxy=sumryz=0;
        for (l=0;l<Lve;l++){
            b=1.0/(1.0+(leta[l]*0.5));
            c=1.0-(leta[l]*0.5);
            
            rxy(gidz,gidx,l)=b*(rxy(gidz,gidx,l)*c-leta[l]*(dipjp*vxyyx));
            ryz(gidz,gidx,l)=b*(ryz(gidz,gidx,l)*c-leta[l]*(djpkp*vyzzy));
            
            sumrxy+=rxy(gidz,gidx,l);
            sumryz+=ryz(gidz,gidx,l);
        }
        
        /* and now the components of the stress tensor are
         completely updated */
        sxy(gidz,gidx)-=lsxy+(dt2*sumrxy);
        syz(gidz,gidx)-=lsyz+(dt2*sumryz);
        
        
#endif
        
        m=evarm(gidz,  gidx);
        if (m!=-1){
            sxy(gidz, gidx)= sxybnd[m];
            syz(gidz, gidx)= szzbnd[m];
        }
    }
#endif
    
// Correct adjoint spatial derivatives to implement CPML
#if abs_type==1
    {
        
        
        if (gidz>NZ-nab-fdoh-1){
            
            i =gidx-fdoh;
            k =gidz - NZ+nab+fdoh+nab;
            ind=2*nab-1-k;
            
            psi_vyz(k,i) = b_z_half[ind] * psi_vyz(k,i) + a_z_half[ind] * vyz_r;
            vyz_r = vyz_r / K_z_half[ind] + psi_vyz(k,i);
        }
        
#if freesurf==0
        else if (gidz-fdoh<nab){
            
            i =gidx-fdoh;
            k =gidz-fdoh;
            
            psi_vyz(k,i) = b_z_half[k] * psi_vyz(k,i) + a_z_half[k] * vyz_r;
            vyz_r = vyz_r / K_z_half[k] + psi_vyz(k,i);
            
        }
#endif
        
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            
            i =gidx-fdoh;
            k =gidz-fdoh;
            
            psi_vyx(k,i) = b_x_half[i] * psi_vyx(k,i) + a_x_half[i] * vyx_r;
            vyx_r = vyx_r / K_x_half[i] + psi_vyx(k,i);
            
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            
            i =gidx - NX+nab+fdoh+nab;
            k =gidz-fdoh;
            ind=2*nab-1-i;
            
            psi_vyx(k,i) = b_x_half[ind] * psi_vyx(k,i) + a_x_half[ind] * vyx_r;
            vyx_r = vyx_r  /K_x_half[ind] + psi_vyx(k,i);
        }
#endif

    }
#endif

// Update adjoint stresses
    {
#if Lve==0
    sxy_r(gidz,gidx)+=(fipjp*vyx_r);
    syz_r(gidz,gidx)+=(fjpkp*vyz_r);
    
    
#else
    
        /* computing sums of the old memory variables */
        sumrxy=sumryz=0;
        for (l=0;l<Lve;l++){
            sumrxy+=rxy_r(gidz,gidx,l);
            sumryz+=ryz_r(gidz,gidx,l);
        }
        
        /* updating components of the stress tensor, partially */
        
        lsxy=(fipjp*vyx_r)+(dt2*sumrxy);
        lsyz=(fjpkp*vyz_r)+(dt2*sumryz);
        
        sumrxy=sumryz=0;
        for (l=0;l<Lve;l++){
            b=1.0/(1.0+(leta[l]*0.5));
            c=1.0-(leta[l]*0.5);
            
            rxy_r(gidz,gidx,l)=b*(rxy_r(gidz,gidx,l)*c-leta[l]*(dipjp*vyx_r));
            ryz_r(gidz,gidx,l)=b*(ryz_r(gidz,gidx,l)*c-leta[l]*(djpkp*vyz_r));
            
            sumrxy+=rxy_r(gidz,gidx,l);
            sumryz+=ryz_r(gidz,gidx,l);
        }
        
        /* and now the components of the stress tensor are
         completely updated */
        sxy_r(gidz,gidx)+=lsxy+(dt2*sumrxy);
        syz_r(gidz,gidx)+=lsyz+(dt2*sumryz);
#endif
}
    
// Absorbing boundary
#if abs_type==2
    {
#if freesurf==0
        if (gidz-fdoh<nab){
            sxy_r(gidz,gidx)*=taper[gidz-fdoh];
            syz_r(gidz,gidx)*=taper[gidz-fdoh];
        }
#endif
        
        if (gidz>NZ-nab-fdoh-1){
            sxy_r(gidz,gidx)*=taper[NZ-fdoh-gidz-1];
            syz_r(gidz,gidx)*=taper[NZ-fdoh-gidz-1];
        }
        
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            sxy_r(gidz,gidx)*=taper[gidx-fdoh];
            syz_r(gidz,gidx)*=taper[gidx-fdoh];
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            sxy_r(gidz,gidx)*=taper[NX-fdoh-gidx-1];
            syz_r(gidz,gidx)*=taper[NX-fdoh-gidx-1];
        }
#endif
    }
#endif
    
// Shear wave modulus gradient calculation on the fly    
#if back_prop_type==1

                 gradmu(gidz,gidy,gidx)+=(sxy(gidz,gidx)*fipjp*vyx_r+syz(gidz,gidx)*fipjp*vyz_r)/(pown( (fipjp/DT+fjpkp/DT)/2.0,2));
#endif
    


}

