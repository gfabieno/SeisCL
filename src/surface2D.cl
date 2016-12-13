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

/*This is the kernel that implement the free surface condition in 2D*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#define rho(z,x)    rho[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rip(z,x)    rip[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define rkp(z,x)    rkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define uipkp(z,x) uipkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define u(z,x)        u[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define pi(z,x)      pi[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define grad(z,x)  grad[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define grads(z,x) grads[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define amp1(z,x)  amp1[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define amp2(z,x)  amp2[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define taus(z,x)        taus[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define tausipkp(z,x) tausipkp[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define taup(z,x)        taup[((x)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

#define vx(z,x)  vx[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define vz(z,x)  vz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxx(z,x) sxx[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define szz(z,x) szz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]
#define sxz(z,x) sxz[(x)*(NZ+NZ_al16)+(z)+NZ_al0]

#define rxx(z,x,l) rxx[(l)*NX*NZ+(x)*NZ+(z)]
#define rzz(z,x,l) rzz[(l)*NX*NZ+(x)*NZ+(z)]
#define rxz(z,x,l) rxz[(l)*NX*NZ+(x)*NZ+(z)]

#if local_off==0

#define lvx(z,x)  lvx[(x)*lsizez+(z)]
#define lvz(z,x)  lvz[(x)*lsizez+(z)]
#define lsxx(z,x) lsxx[(x)*lsizez+(z)]
#define lszz(z,x) lszz[(x)*lsizez+(z)]
#define lsxz(z,x) lsxz[(x)*lsizez+(z)]

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



__kernel void surface(        __global float *vx,         __global float *vz,
                              __global float *sxx,        __global float *szz,      __global float *sxz,
                              __global float *pi,         __global float *u,        __global float *rxx,
                              __global float *rzz,        __global float *taus,     __global float *taup,
                              __global float *eta)
{
    /*Indice definition */
    int gidx = get_global_id(0) + fdoh;
    int gidz=fdoh;
    
    /* Global work size is padded to be a multiple of local work size. The padding elements must not be updated */
    if ( gidx>(NX-fdoh-1) ){
        return;
    }
    
    float f, g, h;
    float  vxx, vzz;
    int m;
    
    /*Mirroring the components of the stress tensor to make
     a stress free surface (method of imaging, Levander, 1988)*/
    szz(gidz, gidx)=0.0;
#if Lve>0
    rzz(gidz, gidx)=0.0;
#endif
    
    for (m=1; m<=fdoh; m++) {
        szz(gidz-m,  gidx)=-szz(gidz+m,  gidx);
        sxz(gidz-m,  gidx)=-sxz(gidz+m-1, gidx);
    }
				
    
#if   fdoh==1
    {
        vxx = (vx(gidz,gidx)-vx(gidz,gidx-1))/DH;
        vzz = (vz(gidz,gidx)-vz(gidz-1,gidx))/DH;
    }
#elif fdoh==2
    {
        vxx = (hc1*(vx(gidz,gidx)  -vx(gidz,gidx-1))+
               hc2*(vx(gidz,gidx+1)-vx(gidz,gidx-2)))/DH;
        
        
        vzz = (hc1*(vz(gidz,gidx)  -vz(gidz-1,gidx))+
               hc2*(vz(gidz+1,gidx)-vz(gidz-2,gidx)))/DH;
    }
#elif fdoh==3
    {
        vxx = (hc1*(vx(gidz,gidx)  -vx(gidz,gidx-1))+
               hc2*(vx(gidz,gidx+1)-vx(gidz,gidx-2))+
               hc3*(vx(gidz,gidx+2)-vx(gidz,gidx-3)))/DH;
        
        vzz = (hc1*(vz(gidz,gidx)-vz(gidz-1,gidx))+
               hc2*(vz(gidz+1,gidx)-vz(gidz-2,gidx))+
               hc3*(vz(gidz+2,gidx)-vz(gidz-3,gidx)))/DH;
        
    }
#elif fdoh==4
    {
        vxx = (hc1*(vx(gidz,gidx)  -vx(gidz,gidx-1))+
               hc2*(vx(gidz,gidx+1)-vx(gidz,gidx-2))+
               hc3*(vx(gidz,gidx+2)-vx(gidz,gidx-3))+
               hc4*(vx(gidz,gidx+3)-vx(gidz,gidx-4)))/DH;
        
        vzz = (hc1*(vz(gidz,gidx)  -vz(gidz-1,gidx))+
               hc2*(vz(gidz+1,gidx)-vz(gidz-2,gidx))+
               hc3*(vz(gidz+2,gidx)-vz(gidz-3,gidx))+
               hc4*(vz(gidz+3,gidx)-vz(gidz-4,gidx)))/DH;
    }
#elif fdoh==5
    {
        vxx = (hc1*(vx(gidz,gidx)  -vx(gidz,gidx-1))+
               hc2*(vx(gidz,gidx+1)-vx(gidz,gidx-2))+
               hc3*(vx(gidz,gidx+2)-vx(gidz,gidx-3))+
               hc4*(vx(gidz,gidx+3)-vx(gidz,gidx-4))+
               hc5*(vx(gidz,gidx+4)-vx(gidz,gidx-5)))/DH;
        
        
        vzz = (hc1*(vz(gidz,gidx)  -vz(gidz-1,gidx))+
               hc2*(vz(gidz+1,gidx)-vz(gidz-2,gidx))+
               hc3*(vz(gidz+2,gidx)-vz(gidz-3,gidx))+
               hc4*(vz(gidz+3,gidx)-vz(gidz-4,gidx))+
               hc5*(vz(gidz+4,gidx)-vz(gidz-5,gidx)))/DH;
        
        
    }
#elif fdoh==6
    {
        vxx = (hc1*(vx(gidz,gidx)  -vx(gidz,gidx-1))+
               hc2*(vx(gidz,gidx+1)-vx(gidz,gidx-2))+
               hc3*(vx(gidz,gidx+2)-vx(gidz,gidx-3))+
               hc4*(vx(gidz,gidx+3)-vx(gidz,gidx-4))+
               hc5*(vx(gidz,gidx+4)-vx(gidz,gidx-5))+
               hc6*(vx(gidz,gidx+5)-vx(gidz,gidx-6)))/DH;
        
        
        vzz = (hc1*(vz(gidz,gidx)  -vz(gidz-1,gidx))+
               hc2*(vz(gidz+1,gidx)-vz(gidz-2,gidx))+
               hc3*(vz(gidz+2,gidx)-vz(gidz-3,gidx))+
               hc4*(vz(gidz+3,gidx)-vz(gidz-4,gidx))+
               hc5*(vz(gidz+4,gidx)-vz(gidz-5,gidx))+
               hc6*(vz(gidz+5,gidx)-vz(gidz-6,gidx)))/DH;
    }
#endif
    

#if Lve==0
				f=u(gidz,  gidx)*2.0;
				g=pi(gidz,  gidx);
				h=-(DT*(g-f)*(g-f)*(vxx)/g)-(DT*(g-f)*vzz);
				sxx(gidz,  gidx)+=h;
#else
    float b,d,e;
    /* partially updating sxx  in the same way*/
    f=u(gidz,  gidx)*2.0*(1.0+L*taus(gidz,  gidx));
    g=pi(gidz,  gidx)*(1.0+L*taup(gidz,  gidx));
    h=-(DT*(g-f)*(g-f)*(vxx+vyy)/g)-(DT*(g-f)*vzz);
    sxx(gidz,  gidx)+=h-(DT/2.0*rxx(gidz,  gidx));
    
    /* updating the memory-variable rxx at the free surface */
    
    d=2.0*u(gidz,  gidx)*taus(gidz,  gidx);
    e=pi(gidz,  gidx)*taup(gidz,  gidx);
    for (m=0;m<Lve;m++){
        b=eta[m]/(1.0+(eta[m]*0.5));
        h=b*(((d-e)*((f/g)-1.0)*(vxx+vyy))-((d-e)*vzz));
        rxx(gidz,  gidx)+=h;
    }
    
    /*completely updating the stresses sxx  */
    sxx(gidz,  gidx)+=(DT/2.0*rxx(gidz,  gidx));
    
#endif

}



