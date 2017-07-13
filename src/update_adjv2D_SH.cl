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

/*Adjoint update of the velocities in 2D SH*/

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

#define vx(z,x)  vx[(x)*NZ+(z)]
#define vy(z,x)  vy[(x)*NZ+(z)]
#define vz(z,x)  vz[(x)*NZ+(z)]
#define sxx(z,x) sxx[(x)*NZ+(z)]
#define szz(z,x) szz[(x)*NZ+(z)]
#define sxz(z,x) sxz[(x)*NZ+(z)]
#define sxy(z,x) sxy[(x)*NZ+(z)]
#define syz(z,x) syz[(x)*NZ+(z)]

#define vx_r(z,x)  vx_r[(x)*NZ+(z)]
#define vy_r(z,x)  vy_r[(x)*NZ+(z)]
#define vz_r(z,x)  vz_r[(x)*NZ+(z)]
#define sxx_r(z,x) sxx_r[(x)*NZ+(z)]
#define szz_r(z,x) szz_r[(x)*NZ+(z)]
#define sxz_r(z,x) sxz_r[(x)*NZ+(z)]
#define sxy_r(z,x) sxy_r[(x)*NZ+(z)]
#define syz_r(z,x) syz_r[(x)*NZ+(z)]

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
    i-=fdoh;
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


__kernel void update_adjv(int offcomm, int nsrc,  int ng, int nt,
                          __global float *vy,
                          __global float *sxy,        __global float *syz,
                          __global float *vybnd,
                          __global float *sxybnd,     __global float *syzbnd,
                          __global float *vy_r,
                          __global float *sxy_r,      __global float *syz_r,
                          __global float *ry,
                          __global float *rjp,
                          __global float *srcpos_loc, __global float *signals, __global float *rec_pos,
                          __global float *taper,
                          __global float *K_x,        __global float *a_x,          __global float *b_x,
                          __global float *K_x_half,   __global float *a_x_half,     __global float *b_x_half,
                          __global float *K_z,        __global float *a_z,          __global float *b_z,
                          __global float *K_z_half,   __global float *a_z_half,     __global float *b_z_half,
                          __global float *psi_sxy_x,  __global float *psi_syz_z,
                          __local  float *lvar, __global float *gradrho, __global float *gradsrc)
{

    int g,i,j,k,m, ind;
    float sxy_x, syz_z;
    float sxy_x_r, syz_z_r;
    float lvy;

// If we use local memory
#if local_off==0
    int lsizez = get_local_size(0)+2*fdoh;
    int lsizex = get_local_size(1)+2*fdoh;
    int lidz = get_local_id(0)+fdoh;
    int lidx = get_local_id(1)+fdoh;
    int gidz = get_global_id(0)+fdoh;
    int gidx = get_global_id(1)+fdoh+offcomm;
    
#define lsxy lvar
#define lsyz lvar
    
#define lsxy_r lvar
#define lsyz_r lvar
    
// If local memory is turned off
#elif local_off==1
    
    int gid = get_global_id(0);
    int glsizez = (NZ-2*fdoh);
    int gidz = gid%glsizez+fdoh;
    int gidx = (gid/glsizez)+fdoh+offcomm;
    
#define lsxy sxy
#define lsyz syz
    
#define lsxy_r sxy_r
#define lsyz_r syz_r
    
#define lidx gidx
#define lidz gidz
    
#define lsizez NZ
#define lsizex NX
    
#endif

// Calculation of the stress spatial derivatives of the forward wavefield if backpropagation is used
#if back_prop_type==1
    {
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lsxy(lidz,lidx)=sxy(gidz,gidx);
        if (lidx<2*fdoh)
            lsxy(lidz,lidx-fdoh)=sxy(gidz,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lsxy(lidz,lidx+lsizex-3*fdoh)=sxy(gidz,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lsxy(lidz,lidx+fdoh)=sxy(gidz,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lsxy(lidz,lidx-lsizex+3*fdoh)=sxy(gidz,gidx-lsizex+3*fdoh);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh ==1
        sxy_x = dtdh*hc1*(lsxy(lidz,lidx)   - lsxy(lidz,lidx-1));
#elif fdoh ==2
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidx)   - lsxy(lidz,lidx-1))
                      +hc2*(lsxy(lidz,lidx+1) - lsxy(lidz,lidx-2)));
#elif fdoh ==3
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidx)  -lsxy(lidz,lidx-1))+
                      hc2*(lsxy(lidz,lidx+1)-lsxy(lidz,lidx-2))+
                      hc3*(lsxy(lidz,lidx+2)-lsxy(lidz,lidx-3)));
#elif fdoh ==4
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidx)  -lsxy(lidz,lidx-1))+
                      hc2*(lsxy(lidz,lidx+1)-lsxy(lidz,lidx-2))+
                      hc3*(lsxy(lidz,lidx+2)-lsxy(lidz,lidx-3))+
                      hc4*(lsxy(lidz,lidx+3)-lsxy(lidz,lidx-4)));
#elif fdoh ==5
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidx)  -lsxy(lidz,lidx-1))+
                      hc2*(lsxy(lidz,lidx+1)-lsxy(lidz,lidx-2))+
                      hc3*(lsxy(lidz,lidx+2)-lsxy(lidz,lidx-3))+
                      hc4*(lsxy(lidz,lidx+3)-lsxy(lidz,lidx-4))+
                      hc5*(lsxy(lidz,lidx+4)-lsxy(lidz,lidx-5)));
        
#elif fdoh ==6
        sxy_x = dtdh*(hc1*(lsxy(lidz,lidx)  -lsxy(lidz,lidx-1))+
                      hc2*(lsxy(lidz,lidx+1)-lsxy(lidz,lidx-2))+
                      hc3*(lsxy(lidz,lidx+2)-lsxy(lidz,lidx-3))+
                      hc4*(lsxy(lidz,lidx+3)-lsxy(lidz,lidx-4))+
                      hc5*(lsxy(lidz,lidx+4)-lsxy(lidz,lidx-5))+
                      hc6*(lsxy(lidz,lidx+5)-lsxy(lidz,lidx-6)));
#endif
        
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lsyz(lidz,lidx)=syz(gidz,gidx);
        if (lidz<2*fdoh)
            lsyz(lidz-fdoh,lidx)=syz(gidz-fdoh,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lsyz(lidz+fdoh,lidx)=syz(gidz+fdoh,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh ==1
        syz_z = dtdh*hc1*(lsyz(lidz,lidx)   - lsyz(lidz-1,lidx));
#elif fdoh ==2
        syz_z = dtdh*(hc1*(lsyz(lidz,lidx)   - lsyz(lidz-1,lidx))
                      +hc2*(lsyz(lidz+1,lidx) - lsyz(lidz-2,lidx)));
#elif fdoh ==3
        syz_z = dtdh*(hc1*(lsyz(lidz,lidx)  -lsyz(lidz-1,lidx))+
                      hc2*(lsyz(lidz+1,lidx)-lsyz(lidz-2,lidx))+
                      hc3*(lsyz(lidz+2,lidx)-lsyz(lidz-3,lidx)));
#elif fdoh ==4
        syz_z = dtdh*(hc1*(lsyz(lidz,lidx)  -lsyz(lidz-1,lidx))+
                      hc2*(lsyz(lidz+1,lidx)-lsyz(lidz-2,lidx))+
                      hc3*(lsyz(lidz+2,lidx)-lsyz(lidz-3,lidx))+
                      hc4*(lsyz(lidz+3,lidx)-lsyz(lidz-4,lidx)));
#elif fdoh ==5
        syz_z = dtdh*(hc1*(lsyz(lidz,lidx)  -lsyz(lidz-1,lidx))+
                      hc2*(lsyz(lidz+1,lidx)-lsyz(lidz-2,lidx))+
                      hc3*(lsyz(lidz+2,lidx)-lsyz(lidz-3,lidx))+
                      hc4*(lsyz(lidz+3,lidx)-lsyz(lidz-4,lidx))+
                      hc5*(lsyz(lidz+4,lidx)-lsyz(lidz-5,lidx)));
#elif fdoh ==6
        syz_z = dtdh*(hc1*(lsyz(lidz,lidx)  -lsyz(lidz-1,lidx))+
                      hc2*(lsyz(lidz+1,lidx)-lsyz(lidz-2,lidx))+
                      hc3*(lsyz(lidz+2,lidx)-lsyz(lidz-3,lidx))+
                      hc4*(lsyz(lidz+3,lidx)-lsyz(lidz-4,lidx))+
                      hc5*(lsyz(lidz+4,lidx)-lsyz(lidz-5,lidx))+
                      hc6*(lsyz(lidz+5,lidx)-lsyz(lidz-6,lidx)));
#endif

    }
#endif
// Calculation of the stress spatial derivatives of the adjoint wavefield
    {
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lsxy_r(lidz,lidx)=sxy_r(gidz,gidx);
        if (lidx<2*fdoh)
            lsxy_r(lidz,lidx-fdoh)=sxy_r(gidz,gidx-fdoh);
        if (lidx+lsizex-3*fdoh<fdoh)
            lsxy_r(lidz,lidx+lsizex-3*fdoh)=sxy_r(gidz,gidx+lsizex-3*fdoh);
        if (lidx>(lsizex-2*fdoh-1))
            lsxy_r(lidz,lidx+fdoh)=sxy_r(gidz,gidx+fdoh);
        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
            lsxy_r(lidz,lidx-lsizex+3*fdoh)=sxy_r(gidz,gidx-lsizex+3*fdoh);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh ==1
        sxy_x_r = dtdh*hc1*(lsxy_r(lidz,lidx)   - lsxy_r(lidz,lidx-1));
#elif fdoh ==2
        sxy_x_r = dtdh*(hc1*(lsxy_r(lidz,lidx)   - lsxy_r(lidz,lidx-1))
                      +hc2*(lsxy_r(lidz,lidx+1) - lsxy_r(lidz,lidx-2)));
#elif fdoh ==3
        sxy_x_r = dtdh*(hc1*(lsxy_r(lidz,lidx)  -lsxy_r(lidz,lidx-1))+
                      hc2*(lsxy_r(lidz,lidx+1)-lsxy_r(lidz,lidx-2))+
                      hc3*(lsxy_r(lidz,lidx+2)-lsxy_r(lidz,lidx-3)));
#elif fdoh ==4
        sxy_x_r = dtdh*(hc1*(lsxy_r(lidz,lidx)  -lsxy_r(lidz,lidx-1))+
                      hc2*(lsxy_r(lidz,lidx+1)-lsxy_r(lidz,lidx-2))+
                      hc3*(lsxy_r(lidz,lidx+2)-lsxy_r(lidz,lidx-3))+
                      hc4*(lsxy_r(lidz,lidx+3)-lsxy_r(lidz,lidx-4)));
#elif fdoh ==5
        sxy_x_r = dtdh*(hc1*(lsxy_r(lidz,lidx)  -lsxy_r(lidz,lidx-1))+
                      hc2*(lsxy_r(lidz,lidx+1)-lsxy_r(lidz,lidx-2))+
                      hc3*(lsxy_r(lidz,lidx+2)-lsxy_r(lidz,lidx-3))+
                      hc4*(lsxy_r(lidz,lidx+3)-lsxy_r(lidz,lidx-4))+
                      hc5*(lsxy_r(lidz,lidx+4)-lsxy_r(lidz,lidx-5)));
        
#elif fdoh ==6
        sxy_x_r = dtdh*(hc1*(lsxy_r(lidz,lidx)  -lsxy_r(lidz,lidx-1))+
                      hc2*(lsxy_r(lidz,lidx+1)-lsxy_r(lidz,lidx-2))+
                      hc3*(lsxy_r(lidz,lidx+2)-lsxy_r(lidz,lidx-3))+
                      hc4*(lsxy_r(lidz,lidx+3)-lsxy_r(lidz,lidx-4))+
                      hc5*(lsxy_r(lidz,lidx+4)-lsxy_r(lidz,lidx-5))+
                      hc6*(lsxy_r(lidz,lidx+5)-lsxy_r(lidz,lidx-6)));
#endif
        
#if local_off==0
        barrier(CLK_LOCAL_MEM_FENCE);
        lsyz_r(lidz,lidx)=syz_r(gidz,gidx);
        if (lidz<2*fdoh)
            lsyz_r(lidz-fdoh,lidx)=syz_r(gidz-fdoh,gidx);
        if (lidz>(lsizez-2*fdoh-1))
            lsyz_r(lidz+fdoh,lidx)=syz_r(gidz+fdoh,gidx);
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        
#if   fdoh ==1
        syz_z_r = dtdh*hc1*(lsyz_r(lidz,lidx)   - lsyz_r(lidz-1,lidx));
#elif fdoh ==2
        syz_z_r = dtdh*(hc1*(lsyz_r(lidz,lidx)   - lsyz_r(lidz-1,lidx))
                      +hc2*(lsyz_r(lidz+1,lidx) - lsyz_r(lidz-2,lidx)));
#elif fdoh ==3
        syz_z_r = dtdh*(hc1*(lsyz_r(lidz,lidx)  -lsyz_r(lidz-1,lidx))+
                      hc2*(lsyz_r(lidz+1,lidx)-lsyz_r(lidz-2,lidx))+
                      hc3*(lsyz_r(lidz+2,lidx)-lsyz_r(lidz-3,lidx)));
#elif fdoh ==4
        syz_z_r = dtdh*(hc1*(lsyz_r(lidz,lidx)  -lsyz_r(lidz-1,lidx))+
                      hc2*(lsyz_r(lidz+1,lidx)-lsyz_r(lidz-2,lidx))+
                      hc3*(lsyz_r(lidz+2,lidx)-lsyz_r(lidz-3,lidx))+
                      hc4*(lsyz_r(lidz+3,lidx)-lsyz_r(lidz-4,lidx)));
#elif fdoh ==5
        syz_z_r = dtdh*(hc1*(lsyz_r(lidz,lidx)  -lsyz_r(lidz-1,lidx))+
                      hc2*(lsyz_r(lidz+1,lidx)-lsyz_r(lidz-2,lidx))+
                      hc3*(lsyz_r(lidz+2,lidx)-lsyz_r(lidz-3,lidx))+
                      hc4*(lsyz_r(lidz+3,lidx)-lsyz_r(lidz-4,lidx))+
                      hc5*(lsyz_r(lidz+4,lidx)-lsyz_r(lidz-5,lidx)));
#elif fdoh ==6
        syz_z_r = dtdh*(hc1*(lsyz_r(lidz,lidx)  -lsyz_r(lidz-1,lidx))+
                      hc2*(lsyz_r(lidz+1,lidx)-lsyz_r(lidz-2,lidx))+
                      hc3*(lsyz_r(lidz+2,lidx)-lsyz_r(lidz-3,lidx))+
                      hc4*(lsyz_r(lidz+3,lidx)-lsyz_r(lidz-4,lidx))+
                      hc5*(lsyz_r(lidz+4,lidx)-lsyz_r(lidz-5,lidx))+
                      hc6*(lsyz_r(lidz+5,lidx)-lsyz_r(lidz-6,lidx)));
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

// Backpropagate the forward velocity
#if back_prop_type==1
    {
        lvy=((sxy_x + syz_z)/rjp(gidz,gidx));

        vy(gidz,gidx)-= lvy;
        
        m=evarm(gidz,  gidx);
        if (m!=-1){
            vy(gidz, gidx)= vybnd[m];
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
        
        psi_syz_z(k,i) = b_z[ind+1] * psi_syz_z(k,i) + a_z[ind+1] * syz_z_r;
        syz_z_r = syz_z_r / K_z[ind+1] + psi_syz_z(k,i);
    }
    
#if freesurf==0
    else if (gidz-fdoh<nab){
        
        i =gidx-fdoh;
        k =gidz-fdoh;
        
        psi_syz_z(k,i) = b_z[k] * psi_syz_z(k,i) + a_z[k] * syz_z_r;
        syz_z_r = syz_z_r / K_z[k] + psi_syz_z(k,i);
    }
#endif
    
    
#if dev==0 & MYLOCALID==0
    if (gidx-fdoh<nab){
        
        i =gidx-fdoh;
        k =gidz-fdoh;
        
        psi_sxy_x(k,i) = b_x[i] * psi_sxy_x(k,i) + a_x[i] * sxy_x_r;
        sxy_x_r = sxy_x_r / K_x[i] + psi_sxy_x(k,i);
    }
#endif
    
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-nab-fdoh-1){
        
        i =gidx - NX+nab+fdoh+nab;
        k =gidz-fdoh;
        ind=2*nab-1-i;
        
        psi_sxy_x(k,i) = b_x[ind+1] * psi_sxy_x(k,i) + a_x[ind+1] * sxy_x_r;
        sxy_x_r = sxy_x_r / K_x[ind+1] + psi_sxy_x(k,i);
        
    }
#endif
    }
#endif

    // Update adjoint velocities
    lvy=((sxy_x_r + syz_z_r)/rjp(gidz,gidx));
    vy_r(gidz,gidx)+= lvy;

 
    
// Absorbing boundary
#if abs_type==2
    {
#if freesurf==0
        if (gidz-fdoh<nab){
            vy_r(gidz,gidx)*=taper[gidz-fdoh];
        }
#endif
        
        if (gidz>NZ-nab-fdoh-1){
            vy_r(gidz,gidx)*=taper[NZ-fdoh-gidz-1];
        }
        
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            vy_r(gidz,gidx)*=taper[gidx-fdoh];
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            vy_r(gidz,gidx)*=taper[NX-fdoh-gidx-1];
        }
#endif
    }
#endif

// Density gradient calculation on the fly
#if back_prop_type==1
    gradrho(gidz,gidx)+=vy(gidz,gidx)*lvy;
#endif
    
#if gradsrcout==1
    if (nsrc>0){
        
        
        for (int srci=0; srci<nsrc; srci++){
            
            
            int i=(int)(srcpos_loc(0,srci)/DH-0.5)+fdoh;
            int k=(int)(srcpos_loc(2,srci)/DH-0.5)+fdoh;
            
            if (i==gidx && k==gidz){
                
                int SOURCE_TYPE= (int)srcpos_loc(4,srci);
                
                if (SOURCE_TYPE==3){
                    /* single force in y */
                    
                    gradsrc(srci,nt)+= vy_r(gidz,gidx)/rjp(gidx,gidz)/(DH*DH);
                }

                
            }
        }
        
        
    }
#endif
    
}

