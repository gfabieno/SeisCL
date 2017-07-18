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

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]

#define psi_sxx_x(z,x) psi_sxx_x[(x)*(NZ-2*fdoh)+(z)]
#define psi_sxz_x(z,x) psi_sxz_x[(x)*(NZ-2*fdoh)+(z)]
#define psi_sxz_z(z,x) psi_sxz_z[(x)*(2*nab)+(z)]
#define psi_szz_z(z,x) psi_szz_z[(x)*(2*nab)+(z)]


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
                          __global float *vx,         __global float *vz,
                          __global float *sxx,        __global float *szz,
                          __global float *sxz,
                          __global float *vxbnd,      __global float *vzbnd,
                          __global float *sxxbnd,     __global float *szzbnd,
                          __global float *sxzbnd,
                          __global float *vx_r,       __global float *vz_r,
                          __global float *sxx_r,      __global float *szz_r,
                          __global float *sxz_r,
                          __global float *rx,         __global float *rz,
                          __global float *rip,        __global float *rkp,
                          __global float *srcpos_loc, __global float *signals, __global float *rec_pos,
                          __global float *taper,
                          __global float *K_x,        __global float *a_x,          __global float *b_x,
                          __global float *K_x_half,   __global float *a_x_half,     __global float *b_x_half,
                          __global float *K_z,        __global float *a_z,          __global float *b_z,
                          __global float *K_z_half,   __global float *a_z_half,     __global float *b_z_half,
                          __global float *psi_sxx_x,  __global float *psi_sxz_x,
                          __global float *psi_sxz_z,  __global float *psi_szz_z,
                          __local  float *lvar,       __global float *gradrho, __global float *gradsrc)
{

    int g,i,j,k,m;
    float sxx_x_r;
    float szz_z_r;
    float sxz_x_r;
    float sxz_z_r;
    float sxx_x;
    float szz_z;
    float sxz_x;
    float sxz_z;
    float lvx;
    float lvz;
    
// If we use local memory
#if local_off==0
    
    int lsizez = get_local_size(0)+2*fdoh;
    int lsizex = get_local_size(1)+2*fdoh;
    int lidz = get_local_id(0)+fdoh;
    int lidx = get_local_id(1)+fdoh;
    int gidz = get_global_id(0)+fdoh;
    int gidx = get_global_id(1)+fdoh+offcomm;

#define lsxx lvar
#define lszz lvar
#define lsxz lvar
    
#define lsxx_r lvar
#define lszz_r lvar
#define lsxz_r lvar
 
// If local memory is turned off
#elif local_off==1
    
    int gid = get_global_id(0);
    int glsizez = (NZ-2*fdoh);
    int gidz = gid%glsizez+fdoh;
    int gidx = (gid/glsizez)+fdoh+offcomm;
    
#define lsxx sxx
#define lszz szz
#define lsxz sxz
    
#define lsxx_r sxx_r
#define lszz_r szz_r
#define lsxz_r sxz_r
    
#define lidx gidx
#define lidz gidz
    
#define lsizez NZ
#define lsizex NX
    
#endif
    
// Calculation of the stress spatial derivatives of the forward wavefield if backpropagation is used
#if back_prop_type==1
    {
//#if local_off==0
//        lsxx(lidz,lidx)=sxx(gidz, gidx);
//        if (lidx<2*fdoh)
//            lsxx(lidz,lidx-fdoh)=sxx(gidz,gidx-fdoh);
//        if (lidx+lsizex-3*fdoh<fdoh)
//            lsxx(lidz,lidx+lsizex-3*fdoh)=sxx(gidz,gidx+lsizex-3*fdoh);
//        if (lidx>(lsizex-2*fdoh-1))
//            lsxx(lidz,lidx+fdoh)=sxx(gidz,gidx+fdoh);
//        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
//            lsxx(lidz,lidx-lsizex+3*fdoh)=sxx(gidz,gidx-lsizex+3*fdoh);
//        barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//        
//#if   fdoh ==1
//        sxx_x = dtdh*hc1*(lsxx(lidz,lidx+1) - lsxx(lidz,lidx));
//#elif fdoh ==2
//        sxx_x = dtdh*(hc1*(lsxx(lidz,lidx+1) - lsxx(lidz,lidx))
//                      +hc2*(lsxx(lidz,lidx+2) - lsxx(lidz,lidx-1)));
//#elif fdoh ==3
//        sxx_x = dtdh*(hc1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
//                      hc2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
//                      hc3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2)));
//#elif fdoh ==4
//        sxx_x = dtdh*(hc1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
//                      hc2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
//                      hc3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2))+
//                      hc4*(lsxx(lidz,lidx+4)-lsxx(lidz,lidx-3)));
//#elif fdoh ==5
//        sxx_x = dtdh*(hc1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
//                      hc2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
//                      hc3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2))+
//                      hc4*(lsxx(lidz,lidx+4)-lsxx(lidz,lidx-3))+
//                      hc5*(lsxx(lidz,lidx+5)-lsxx(lidz,lidx-4)));
//#elif fdoh ==6
//        sxx_x = dtdh*(hc1*(lsxx(lidz,lidx+1)-lsxx(lidz,lidx))+
//                      hc2*(lsxx(lidz,lidx+2)-lsxx(lidz,lidx-1))+
//                      hc3*(lsxx(lidz,lidx+3)-lsxx(lidz,lidx-2))+
//                      hc4*(lsxx(lidz,lidx+4)-lsxx(lidz,lidx-3))+
//                      hc5*(lsxx(lidz,lidx+5)-lsxx(lidz,lidx-4))+
//                      hc6*(lsxx(lidz,lidx+6)-lsxx(lidz,lidx-5)));
//#endif
//        
//        
//#if local_off==0
//        barrier(CLK_LOCAL_MEM_FENCE);
//        lszz(lidz,lidx)=szz(gidz, gidx);
//        if (lidz<2*fdoh)
//            lszz(lidz-fdoh,lidx)=szz(gidz-fdoh,gidx);
//        if (lidz>(lsizez-2*fdoh-1))
//            lszz(lidz+fdoh,lidx)=szz(gidz+fdoh,gidx);
//        barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//        
//#if   fdoh ==1
//        szz_z = dtdh*hc1*(lszz(lidz+1,lidx) - lszz(lidz,lidx));
//#elif fdoh ==2
//        szz_z = dtdh*(hc1*(lszz(lidz+1,lidx) - lszz(lidz,lidx))
//                      +hc2*(lszz(lidz+2,lidx) - lszz(lidz-1,lidx)));
//#elif fdoh ==3
//        szz_z = dtdh*(hc1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
//                      hc2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
//                      hc3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx)));
//#elif fdoh ==4
//        szz_z = dtdh*(hc1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
//                      hc2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
//                      hc3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx))+
//                      hc4*(lszz(lidz+4,lidx)-lszz(lidz-3,lidx)));
//#elif fdoh ==5
//        szz_z = dtdh*(hc1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
//                      hc2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
//                      hc3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx))+
//                      hc4*(lszz(lidz+4,lidx)-lszz(lidz-3,lidx))+
//                      hc5*(lszz(lidz+5,lidx)-lszz(lidz-4,lidx)));
//#elif fdoh ==6
//        szz_z = dtdh*(hc1*(lszz(lidz+1,lidx)-lszz(lidz,lidx))+
//                      hc2*(lszz(lidz+2,lidx)-lszz(lidz-1,lidx))+
//                      hc3*(lszz(lidz+3,lidx)-lszz(lidz-2,lidx))+
//                      hc4*(lszz(lidz+4,lidx)-lszz(lidz-3,lidx))+
//                      hc5*(lszz(lidz+5,lidx)-lszz(lidz-4,lidx))+
//                      hc6*(lszz(lidz+6,lidx)-lszz(lidz-5,lidx)));
//#endif
//        
//#if local_off==0
//        barrier(CLK_LOCAL_MEM_FENCE);
//        lsxz(lidz,lidx)=sxz(gidz, gidx);
//        
//        if (lidx<2*fdoh)
//            lsxz(lidz,lidx-fdoh)=sxz(gidz,gidx-fdoh);
//        if (lidx+lsizex-3*fdoh<fdoh)
//            lsxz(lidz,lidx+lsizex-3*fdoh)=sxz(gidz,gidx+lsizex-3*fdoh);
//        if (lidx>(lsizex-2*fdoh-1))
//            lsxz(lidz,lidx+fdoh)=sxz(gidz,gidx+fdoh);
//        if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
//            lsxz(lidz,lidx-lsizex+3*fdoh)=sxz(gidz,gidx-lsizex+3*fdoh);
//        if (lidz<2*fdoh)
//            lsxz(lidz-fdoh,lidx)=sxz(gidz-fdoh,gidx);
//        if (lidz>(lsizez-2*fdoh-1))
//            lsxz(lidz+fdoh,lidx)=sxz(gidz+fdoh,gidx);
//        barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//        
//#if   fdoh ==1
//        sxz_z = dtdh*hc1*(lsxz(lidz,lidx)   - lsxz(lidz-1,lidx));
//        sxz_x = dtdh*hc1*(lsxz(lidz,lidx)   - lsxz(lidz,lidx-1));
//#elif fdoh ==2
//        sxz_z = dtdh*(hc1*(lsxz(lidz,lidx)   - lsxz(lidz-1,lidx))
//                      +hc2*(lsxz(lidz+1,lidx) - lsxz(lidz-2,lidx)));
//        sxz_x = dtdh*(hc1*(lsxz(lidz,lidx)   - lsxz(lidz,lidx-1))
//                      +hc2*(lsxz(lidz,lidx+1) - lsxz(lidz,lidx-2)));
//#elif fdoh ==3
//        sxz_z = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
//                      hc2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
//                      hc3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx)));
//        
//        sxz_x = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
//                      hc2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
//                      hc3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3)));
//#elif fdoh ==4
//        sxz_z = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
//                      hc2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
//                      hc3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx))+
//                      hc4*(lsxz(lidz+3,lidx)-lsxz(lidz-4,lidx)));
//        
//        sxz_x = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
//                      hc2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
//                      hc3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3))+
//                      hc4*(lsxz(lidz,lidx+3)-lsxz(lidz,lidx-4)));
//#elif fdoh ==5
//        sxz_z = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
//                      hc2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
//                      hc3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx))+
//                      hc4*(lsxz(lidz+3,lidx)-lsxz(lidz-4,lidx))+
//                      hc5*(lsxz(lidz+4,lidx)-lsxz(lidz-5,lidx)));
//        
//        sxz_x = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
//                      hc2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
//                      hc3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3))+
//                      hc4*(lsxz(lidz,lidx+3)-lsxz(lidz,lidx-4))+
//                      hc5*(lsxz(lidz,lidx+4)-lsxz(lidz,lidx-5)));
//#elif fdoh ==6
//        
//        sxz_z = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz-1,lidx))+
//                      hc2*(lsxz(lidz+1,lidx)-lsxz(lidz-2,lidx))+
//                      hc3*(lsxz(lidz+2,lidx)-lsxz(lidz-3,lidx))+
//                      hc4*(lsxz(lidz+3,lidx)-lsxz(lidz-4,lidx))+
//                      hc5*(lsxz(lidz+4,lidx)-lsxz(lidz-5,lidx))+
//                      hc6*(lsxz(lidz+5,lidx)-lsxz(lidz-6,lidx)));
//        
//        sxz_x = dtdh*(hc1*(lsxz(lidz,lidx)  -lsxz(lidz,lidx-1))+
//                      hc2*(lsxz(lidz,lidx+1)-lsxz(lidz,lidx-2))+
//                      hc3*(lsxz(lidz,lidx+2)-lsxz(lidz,lidx-3))+
//                      hc4*(lsxz(lidz,lidx+3)-lsxz(lidz,lidx-4))+
//                      hc5*(lsxz(lidz,lidx+4)-lsxz(lidz,lidx-5))+
//                      hc6*(lsxz(lidz,lidx+5)-lsxz(lidz,lidx-6)));
//#endif
//        barrier(CLK_LOCAL_MEM_FENCE);
}
#endif
//
//// Calculation of the stress spatial derivatives of the adjoint wavefield
//#if local_off==0
//    lsxx_r(lidz,lidx)=sxx_r(gidz, gidx);
//    if (lidx<2*fdoh)
//        lsxx_r(lidz,lidx-fdoh)=sxx_r(gidz,gidx-fdoh);
//    if (lidx+lsizex-3*fdoh<fdoh)
//        lsxx_r(lidz,lidx+lsizex-3*fdoh)=sxx_r(gidz,gidx+lsizex-3*fdoh);
//    if (lidx>(lsizex-2*fdoh-1))
//        lsxx_r(lidz,lidx+fdoh)=sxx_r(gidz,gidx+fdoh);
//    if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
//        lsxx_r(lidz,lidx-lsizex+3*fdoh)=sxx_r(gidz,gidx-lsizex+3*fdoh);
//    barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//    
//#if   fdoh ==1
//    sxx_x_r = dtdh*hc1*(lsxx_r(lidz,lidx+1) - lsxx_r(lidz,lidx));
//#elif fdoh ==2
//    sxx_x_r = dtdh*(hc1*(lsxx_r(lidz,lidx+1) - lsxx_r(lidz,lidx))
//                  +hc2*(lsxx_r(lidz,lidx+2) - lsxx_r(lidz,lidx-1)));
//#elif fdoh ==3
//    sxx_x_r = dtdh*(hc1*(lsxx_r(lidz,lidx+1)-lsxx_r(lidz,lidx))+
//                  hc2*(lsxx_r(lidz,lidx+2)-lsxx_r(lidz,lidx-1))+
//                  hc3*(lsxx_r(lidz,lidx+3)-lsxx_r(lidz,lidx-2)));
//#elif fdoh ==4
//    sxx_x_r = dtdh*(hc1*(lsxx_r(lidz,lidx+1)-lsxx_r(lidz,lidx))+
//                  hc2*(lsxx_r(lidz,lidx+2)-lsxx_r(lidz,lidx-1))+
//                  hc3*(lsxx_r(lidz,lidx+3)-lsxx_r(lidz,lidx-2))+
//                  hc4*(lsxx_r(lidz,lidx+4)-lsxx_r(lidz,lidx-3)));
//#elif fdoh ==5
//    sxx_x_r = dtdh*(hc1*(lsxx_r(lidz,lidx+1)-lsxx_r(lidz,lidx))+
//                  hc2*(lsxx_r(lidz,lidx+2)-lsxx_r(lidz,lidx-1))+
//                  hc3*(lsxx_r(lidz,lidx+3)-lsxx_r(lidz,lidx-2))+
//                  hc4*(lsxx_r(lidz,lidx+4)-lsxx_r(lidz,lidx-3))+
//                  hc5*(lsxx_r(lidz,lidx+5)-lsxx_r(lidz,lidx-4)));
//#elif fdoh ==6
//    sxx_x_r = dtdh*(hc1*(lsxx_r(lidz,lidx+1)-lsxx_r(lidz,lidx))+
//                  hc2*(lsxx_r(lidz,lidx+2)-lsxx_r(lidz,lidx-1))+
//                  hc3*(lsxx_r(lidz,lidx+3)-lsxx_r(lidz,lidx-2))+
//                  hc4*(lsxx_r(lidz,lidx+4)-lsxx_r(lidz,lidx-3))+
//                  hc5*(lsxx_r(lidz,lidx+5)-lsxx_r(lidz,lidx-4))+
//                  hc6*(lsxx_r(lidz,lidx+6)-lsxx_r(lidz,lidx-5)));
//#endif
//    
//    
//#if local_off==0
//    barrier(CLK_LOCAL_MEM_FENCE);
//    lszz_r(lidz,lidx)=szz_r(gidz, gidx);
//    if (lidz<2*fdoh)
//        lszz_r(lidz-fdoh,lidx)=szz_r(gidz-fdoh,gidx);
//    if (lidz>(lsizez-2*fdoh-1))
//        lszz_r(lidz+fdoh,lidx)=szz_r(gidz+fdoh,gidx);
//    barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//    
//#if   fdoh ==1
//    szz_z_r = dtdh*hc1*(lszz_r(lidz+1,lidx) - lszz_r(lidz,lidx));
//#elif fdoh ==2
//    szz_z_r = dtdh*(hc1*(lszz_r(lidz+1,lidx) - lszz_r(lidz,lidx))
//                  +hc2*(lszz_r(lidz+2,lidx) - lszz_r(lidz-1,lidx)));
//#elif fdoh ==3
//    szz_z_r = dtdh*(hc1*(lszz_r(lidz+1,lidx)-lszz_r(lidz,lidx))+
//                  hc2*(lszz_r(lidz+2,lidx)-lszz_r(lidz-1,lidx))+
//                  hc3*(lszz_r(lidz+3,lidx)-lszz_r(lidz-2,lidx)));
//#elif fdoh ==4
//    szz_z_r = dtdh*(hc1*(lszz_r(lidz+1,lidx)-lszz_r(lidz,lidx))+
//                  hc2*(lszz_r(lidz+2,lidx)-lszz_r(lidz-1,lidx))+
//                  hc3*(lszz_r(lidz+3,lidx)-lszz_r(lidz-2,lidx))+
//                  hc4*(lszz_r(lidz+4,lidx)-lszz_r(lidz-3,lidx)));
//#elif fdoh ==5
//    szz_z_r = dtdh*(hc1*(lszz_r(lidz+1,lidx)-lszz_r(lidz,lidx))+
//                  hc2*(lszz_r(lidz+2,lidx)-lszz_r(lidz-1,lidx))+
//                  hc3*(lszz_r(lidz+3,lidx)-lszz_r(lidz-2,lidx))+
//                  hc4*(lszz_r(lidz+4,lidx)-lszz_r(lidz-3,lidx))+
//                  hc5*(lszz_r(lidz+5,lidx)-lszz_r(lidz-4,lidx)));
//#elif fdoh ==6
//    szz_z_r = dtdh*(hc1*(lszz_r(lidz+1,lidx)-lszz_r(lidz,lidx))+
//                  hc2*(lszz_r(lidz+2,lidx)-lszz_r(lidz-1,lidx))+
//                  hc3*(lszz_r(lidz+3,lidx)-lszz_r(lidz-2,lidx))+
//                  hc4*(lszz_r(lidz+4,lidx)-lszz_r(lidz-3,lidx))+
//                  hc5*(lszz_r(lidz+5,lidx)-lszz_r(lidz-4,lidx))+
//                  hc6*(lszz_r(lidz+6,lidx)-lszz_r(lidz-5,lidx)));
//#endif
//    
//#if local_off==0
//    barrier(CLK_LOCAL_MEM_FENCE);
//    lsxz_r(lidz,lidx)=sxz_r(gidz, gidx);
//    
//    if (lidx<2*fdoh)
//        lsxz_r(lidz,lidx-fdoh)=sxz_r(gidz,gidx-fdoh);
//    if (lidx+lsizex-3*fdoh<fdoh)
//        lsxz_r(lidz,lidx+lsizex-3*fdoh)=sxz_r(gidz,gidx+lsizex-3*fdoh);
//    if (lidx>(lsizex-2*fdoh-1))
//        lsxz_r(lidz,lidx+fdoh)=sxz_r(gidz,gidx+fdoh);
//    if (lidx-lsizex+3*fdoh>(lsizex-fdoh-1))
//        lsxz_r(lidz,lidx-lsizex+3*fdoh)=sxz_r(gidz,gidx-lsizex+3*fdoh);
//    if (lidz<2*fdoh)
//        lsxz_r(lidz-fdoh,lidx)=sxz_r(gidz-fdoh,gidx);
//    if (lidz>(lsizez-2*fdoh-1))
//        lsxz_r(lidz+fdoh,lidx)=sxz_r(gidz+fdoh,gidx);
//    barrier(CLK_LOCAL_MEM_FENCE);
//#endif
//    
//#if   fdoh ==1
//    sxz_z_r = dtdh*hc1*(lsxz_r(lidz,lidx)   - lsxz_r(lidz-1,lidx));
//    sxz_x_r = dtdh*hc1*(lsxz_r(lidz,lidx)   - lsxz_r(lidz,lidx-1));
//#elif fdoh ==2
//    sxz_z_r = dtdh*(hc1*(lsxz_r(lidz,lidx)   - lsxz_r(lidz-1,lidx))
//                  +hc2*(lsxz_r(lidz+1,lidx) - lsxz_r(lidz-2,lidx)));
//    sxz_x_r = dtdh*(hc1*(lsxz_r(lidz,lidx)   - lsxz_r(lidz,lidx-1))
//                  +hc2*(lsxz_r(lidz,lidx+1) - lsxz_r(lidz,lidx-2)));
//#elif fdoh ==3
//    sxz_z_r = dtdh*(hc1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz-1,lidx))+
//                  hc2*(lsxz_r(lidz+1,lidx)-lsxz_r(lidz-2,lidx))+
//                  hc3*(lsxz_r(lidz+2,lidx)-lsxz_r(lidz-3,lidx)));
//    
//    sxz_x_r = dtdh*(hc1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz,lidx-1))+
//                  hc2*(lsxz_r(lidz,lidx+1)-lsxz_r(lidz,lidx-2))+
//                  hc3*(lsxz_r(lidz,lidx+2)-lsxz_r(lidz,lidx-3)));
//#elif fdoh ==4
//    sxz_z_r = dtdh*(hc1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz-1,lidx))+
//                  hc2*(lsxz_r(lidz+1,lidx)-lsxz_r(lidz-2,lidx))+
//                  hc3*(lsxz_r(lidz+2,lidx)-lsxz_r(lidz-3,lidx))+
//                  hc4*(lsxz_r(lidz+3,lidx)-lsxz_r(lidz-4,lidx)));
//    
//    sxz_x_r = dtdh*(hc1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz,lidx-1))+
//                  hc2*(lsxz_r(lidz,lidx+1)-lsxz_r(lidz,lidx-2))+
//                  hc3*(lsxz_r(lidz,lidx+2)-lsxz_r(lidz,lidx-3))+
//                  hc4*(lsxz_r(lidz,lidx+3)-lsxz_r(lidz,lidx-4)));
//#elif fdoh ==5
//    sxz_z_r = dtdh*(hc1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz-1,lidx))+
//                  hc2*(lsxz_r(lidz+1,lidx)-lsxz_r(lidz-2,lidx))+
//                  hc3*(lsxz_r(lidz+2,lidx)-lsxz_r(lidz-3,lidx))+
//                  hc4*(lsxz_r(lidz+3,lidx)-lsxz_r(lidz-4,lidx))+
//                  hc5*(lsxz_r(lidz+4,lidx)-lsxz_r(lidz-5,lidx)));
//    
//    sxz_x_r = dtdh*(hc1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz,lidx-1))+
//                  hc2*(lsxz_r(lidz,lidx+1)-lsxz_r(lidz,lidx-2))+
//                  hc3*(lsxz_r(lidz,lidx+2)-lsxz_r(lidz,lidx-3))+
//                  hc4*(lsxz_r(lidz,lidx+3)-lsxz_r(lidz,lidx-4))+
//                  hc5*(lsxz_r(lidz,lidx+4)-lsxz_r(lidz,lidx-5)));
//#elif fdoh ==6
//    
//    sxz_z_r = dtdh*(hc1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz-1,lidx))+
//                  hc2*(lsxz_r(lidz+1,lidx)-lsxz_r(lidz-2,lidx))+
//                  hc3*(lsxz_r(lidz+2,lidx)-lsxz_r(lidz-3,lidx))+
//                  hc4*(lsxz_r(lidz+3,lidx)-lsxz_r(lidz-4,lidx))+
//                  hc5*(lsxz_r(lidz+4,lidx)-lsxz_r(lidz-5,lidx))+
//                  hc6*(lsxz_r(lidz+5,lidx)-lsxz_r(lidz-6,lidx)));
//    
//    sxz_x_r = dtdh*(hc1*(lsxz_r(lidz,lidx)  -lsxz_r(lidz,lidx-1))+
//                  hc2*(lsxz_r(lidz,lidx+1)-lsxz_r(lidz,lidx-2))+
//                  hc3*(lsxz_r(lidz,lidx+2)-lsxz_r(lidz,lidx-3))+
//                  hc4*(lsxz_r(lidz,lidx+3)-lsxz_r(lidz,lidx-4))+
//                  hc5*(lsxz_r(lidz,lidx+4)-lsxz_r(lidz,lidx-5))+
//                  hc6*(lsxz_r(lidz,lidx+5)-lsxz_r(lidz,lidx-6)));
//#endif
//
//    
//// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
//#if local_off==0
//#if comm12==0
//    if (gidz>(NZ-fdoh-1) || (gidx-offcomm)>(NX-fdoh-1-lcomm) ){
//        return;
//    }
//    
//#else
//    if (gidz>(NZ-fdoh-1) ){
//        return;
//    }
//#endif
//#endif
//
//
//// Backpropagate the forward velocity
//#if back_prop_type==1
//    {
//        
//        lvx=((sxx_x + sxz_z)/rip(gidz,gidx));
//        lvz=((szz_z + sxz_x)/rkp(gidz,gidx));
//        vx(gidz,gidx)-= lvx;
//        vz(gidz,gidx)-= lvz;
//        
//        // Inject the boundary values
//        m=evarm(gidz,  gidx);
//        if (m!=-1){
//            vx(gidz, gidx)= vxbnd[m];
//            vz(gidz, gidx)= vzbnd[m];
//        }
//    }
//#endif
//
//// Correct adjoint spatial derivatives to implement CPML
//#if abs_type==1
//    {
//    int ind;
//    
//    if (gidz>NZ-nab-fdoh-1){
//        
//        i =gidx-fdoh;
//        k =gidz - NZ+nab+fdoh+nab;
//        ind=2*nab-1-k;
//        
//        psi_sxz_z(k,i) = b_z[ind+1] * psi_sxz_z(k,i) + a_z[ind+1] * sxz_z_r;
//        sxz_z_r = sxz_z_r / K_z[ind+1] + psi_sxz_z(k,i);
//        psi_szz_z(k,i) = b_z_half[ind] * psi_szz_z(k,i) + a_z_half[ind] * szz_z_r;
//        szz_z_r = szz_z_r / K_z_half[ind] + psi_szz_z(k,i);
//        
//    }
//    
//#if freesurf==0
//    else if (gidz-fdoh<nab){
//        
//        i =gidx-fdoh;
//        k =gidz-fdoh;
//        
//        psi_sxz_z(k,i) = b_z[k] * psi_sxz_z(k,i) + a_z[k] * sxz_z_r;
//        sxz_z_r = sxz_z_r / K_z[k] + psi_sxz_z(k,i);
//        psi_szz_z(k,i) = b_z_half[k] * psi_szz_z(k,i) + a_z_half[k] * szz_z_r;
//        szz_z_r = szz_z_r / K_z_half[k] + psi_szz_z(k,i);
//        
//    }
//#endif
//    
//#if dev==0 & MYLOCALID==0
//    if (gidx-fdoh<nab){
//        
//        i =gidx-fdoh;
//        k =gidz-fdoh;
//        
//        psi_sxx_x(k,i) = b_x_half[i] * psi_sxx_x(k,i) + a_x_half[i] * sxx_x_r;
//        sxx_x_r = sxx_x_r / K_x_half[i] + psi_sxx_x(k,i);
//        psi_sxz_x(k,i) = b_x[i] * psi_sxz_x(k,i) + a_x[i] * sxz_x_r;
//        sxz_x_r = sxz_x_r / K_x[i] + psi_sxz_x(k,i);
//        
//    }
//#endif
//    
//#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
//    if (gidx>NX-nab-fdoh-1){
//        
//        i =gidx - NX+nab+fdoh+nab;
//        k =gidz-fdoh;
//        ind=2*nab-1-i;
//        
//        psi_sxx_x(k,i) = b_x_half[ind] * psi_sxx_x(k,i) + a_x_half[ind] * sxx_x_r;
//        sxx_x_r = sxx_x_r / K_x_half[ind] + psi_sxx_x(k,i);
//        psi_sxz_x(k,i) = b_x[ind+1] * psi_sxz_x(k,i) + a_x[ind+1] * sxz_x_r;
//        sxz_x_r = sxz_x_r / K_x[ind+1] + psi_sxz_x(k,i);
//        
//    }
//#endif
//    }
//#endif
//    
//    // Update adjoint velocities
//    lvx=((sxx_x_r + sxz_z_r)/rip(gidz,gidx));
//    lvz=((szz_z_r + sxz_x_r)/rkp(gidz,gidx));
//    vx_r(gidz,gidx)+= lvx;
//    vz_r(gidz,gidx)+= lvz;
// 
//    
//
//// Absorbing boundary
//#if abs_type==2
//    {
//    if (gidz-fdoh<nab){
//        vx_r(gidz,gidx)*=taper[gidz-fdoh];
//        vz_r(gidz,gidx)*=taper[gidz-fdoh];
//    }
//    
//    if (gidz>NZ-nab-fdoh-1){
//        vx_r(gidz,gidx)*=taper[NZ-fdoh-gidz-1];
//        vz_r(gidz,gidx)*=taper[NZ-fdoh-gidz-1];
//    }
//    
//#if dev==0 & MYLOCALID==0
//    if (gidx-fdoh<nab){
//        vx_r(gidz,gidx)*=taper[gidx-fdoh];
//        vz_r(gidz,gidx)*=taper[gidx-fdoh];
//    }
//#endif
//    
//#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
//    if (gidx>NX-nab-fdoh-1){
//        vx_r(gidz,gidx)*=taper[NX-fdoh-gidx-1];
//        vz_r(gidz,gidx)*=taper[NX-fdoh-gidx-1];
//    }
//#endif
//    }
//#endif
//    
//    
//// Density gradient calculation on the fly
//#if back_prop_type==1
//    gradrho(gidz,gidx)+=vx(gidz,gidx)*lvx+vz(gidz,gidx)*lvz;
//#endif
//    
//#if gradsrcout==1
//    if (nsrc>0){
//        
//        
//        for (int srci=0; srci<nsrc; srci++){
//            
//            
//            int i=(int)(srcpos_loc(0,srci)/DH-0.5)+fdoh;
//            int k=(int)(srcpos_loc(2,srci)/DH-0.5)+fdoh;
//            
//            if (i==gidx && k==gidz){
//                
//                int SOURCE_TYPE= (int)srcpos_loc(4,srci);
//                
//                if (SOURCE_TYPE==2){
//                    /* single force in x */
//                    gradsrc(srci,nt)+= vx_r(gidz,gidx)/rip(gidx,gidz)/(DH*DH);
//                }
//                else if (SOURCE_TYPE==4){
//                    /* single force in z */
//                    
//                    gradsrc(srci,nt)+= vz_r(gidz,gidx)/rkp(gidx,gidz)/(DH*DH);
//                }
//                
//            }
//        }
//        
//        
//    }
//#endif
    
}

