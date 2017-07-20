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

/*This is the kernel that implement the free surface condition in 3D*/

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
#define grad(z,y,x)   grad[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define grads(z,y,x) grads[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define amp1(z,y,x)   amp1[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]
#define amp2(z,y,x)   amp2[((x)-fdoh)*(NY-2*fdoh)*(NZ-2*fdoh)+((y)-fdoh)*(NZ-2*fdoh)+((z)-fdoh)]

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


#if local_off==0

#define lvx(z,y,x)   lvx[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lvy(z,y,x)   lvy[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lvz(z,y,x)   lvz[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsxx(z,y,x) lsxx[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsyy(z,y,x) lsyy[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lszz(z,y,x) lszz[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsxy(z,y,x) lsxy[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsyz(z,y,x) lsyz[(x)*lsizey*lsizez+(y)*lsizez+(z)]
#define lsxz(z,y,x) lsxz[(x)*lsizey*lsizez+(y)*lsizez+(z)]

#endif

#define psi_vxx(z,y,x) psi_vxx[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_vyx(z,y,x) psi_vyx[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_vzx(z,y,x) psi_vzx[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]

#define psi_vxy(z,y,x) psi_vxy[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_vyy(z,y,x) psi_vyy[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define psi_vzy(z,y,x) psi_vzy[(x)*(2*nab)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]

#define psi_vxz(z,y,x) psi_vxz[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]
#define psi_vyz(z,y,x) psi_vyz[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]
#define psi_vzz(z,y,x) psi_vzz[(x)*(NY-2*fdoh)*(2*nab)+(y)*(2*nab)+(z)]



#define PI (3.141592653589793238462643383279502884197169)
#define srcpos_loc(y,x) srcpos_loc[(y)*nsrc+(x)]
#define signals(y,x) signals[(y)*NT+(x)]



__kernel void surface(        __global float *vx,         __global float *vy,       __global float *vz,
                              __global float *sxx,        __global float *syy,      __global float *szz,
                              __global float *sxy,        __global float *syz,      __global float *sxz,
                              __global float *pi,         __global float *u,        __global float *rxx,
                              __global float *ryy,        __global float *rzz,
                              __global float *taus,       __global float *taup,     __global float *eta, __global float *K_x, __global float *psi_vxx,
                              __global float *K_y, __global float *psi_vyy)
{
    /*Indice definition */
    int gidy = get_global_id(0) + fdoh;
    int gidx = get_global_id(1) + fdoh;
    int gidz=fdoh;
    
    /* Global work size is padded to be a multiple of local work size. The padding elements must not be updated */
    if (gidy>(NY-fdoh-1) || gidx>(NX-fdoh-1) ){
        return;
    }
    
    float f, g, h;
    float  vxx, vyy, vzz;
    int m;
    
    /*Mirroring the components of the stress tensor to make
     a stress free surface (method of imaging, Levander, 1988)*/
    szz(gidz, gidy, gidx)=0.0;
#if Lve>0
    rzz(gidz, gidy, gidx)=0.0;
#endif
    
    for (m=1; m<=fdoh; m++) {
        szz(gidz-m, gidy, gidx)=-szz(gidz+m, gidy, gidx);
        sxz(gidz-m, gidy, gidx)=-sxz(gidz+m-1, gidy, gidx);
        syz(gidz-m, gidy, gidx)=-syz(gidz+m-1, gidy, gidx);
				}
				
    
#if   fdoh==1
    {
        vxx = (vx(gidz,gidy,gidx)-vx(gidz,gidy,gidx-1))/DH;
        vyy = (vy(gidz,gidy,gidx)-vy(gidz,gidy-1,gidx))/DH;
        vzz = (vz(gidz,gidy,gidx)-vz(gidz-1,gidy,gidx))/DH;
    }
#elif fdoh==2
    {
        vxx = (hc1*(vx(gidz,gidy,gidx)  -vx(gidz,gidy,gidx-1))+
               hc2*(vx(gidz,gidy,gidx+1)-vx(gidz,gidy,gidx-2)))/DH;
        
        vyy = (hc1*(vy(gidz,gidy,gidx)  -vy(gidz,gidy-1,gidx))+
               hc2*(vy(gidz,gidy+1,gidx)-vy(gidz,gidy-2,gidx)))/DH;
        
        vzz = (hc1*(vz(gidz,gidy,gidx)  -vz(gidz-1,gidy,gidx))+
               hc2*(vz(gidz+1,gidy,gidx)-vz(gidz-2,gidy,gidx)))/DH;
    }
#elif fdoh==3
    {
        vxx = (hc1*(vx(gidz,gidy,gidx)  -vx(gidz,gidy,gidx-1))+
               hc2*(vx(gidz,gidy,gidx+1)-vx(gidz,gidy,gidx-2))+
               hc3*(vx(gidz,gidy,gidx+2)-vx(gidz,gidy,gidx-3)))/DH;
        
        vyy = (hc1*(vy(gidz,gidy,gidx)-vy(gidz,gidy-1,gidx))+
               hc2*(vy(gidz,gidy+1,gidx)-vy(gidz,gidy-2,gidx))+
               hc3*(vy(gidz,gidy+2,gidx)-vy(gidz,gidy-3,gidx)))/DH;
        
        vzz = (hc1*(vz(gidz,gidy,gidx)-vz(gidz-1,gidy,gidx))+
               hc2*(vz(gidz+1,gidy,gidx)-vz(gidz-2,gidy,gidx))+
               hc3*(vz(gidz+2,gidy,gidx)-vz(gidz-3,gidy,gidx)))/DH;
        
    }
#elif fdoh==4
    {
        vxx = (hc1*(vx(gidz,gidy,gidx)  -vx(gidz,gidy,gidx-1))+
               hc2*(vx(gidz,gidy,gidx+1)-vx(gidz,gidy,gidx-2))+
               hc3*(vx(gidz,gidy,gidx+2)-vx(gidz,gidy,gidx-3))+
               hc4*(vx(gidz,gidy,gidx+3)-vx(gidz,gidy,gidx-4)))/DH;

        vyy = (hc1*(vy(gidz,gidy,gidx)  -vy(gidz,gidy-1,gidx))+
               hc2*(vy(gidz,gidy+1,gidx)-vy(gidz,gidy-2,gidx))+
               hc3*(vy(gidz,gidy+2,gidx)-vy(gidz,gidy-3,gidx))+
               hc4*(vy(gidz,gidy+3,gidx)-vy(gidz,gidy-4,gidx)))/DH;
        
        vzz = (hc1*(vz(gidz,gidy,gidx)  -vz(gidz-1,gidy,gidx))+
               hc2*(vz(gidz+1,gidy,gidx)-vz(gidz-2,gidy,gidx))+
               hc3*(vz(gidz+2,gidy,gidx)-vz(gidz-3,gidy,gidx))+
               hc4*(vz(gidz+3,gidy,gidx)-vz(gidz-4,gidy,gidx)))/DH;
    }
#elif fdoh==5
    {
        vxx = (hc1*(vx(gidz,gidy,gidx)  -vx(gidz,gidy,gidx-1))+
               hc2*(vx(gidz,gidy,gidx+1)-vx(gidz,gidy,gidx-2))+
               hc3*(vx(gidz,gidy,gidx+2)-vx(gidz,gidy,gidx-3))+
               hc4*(vx(gidz,gidy,gidx+3)-vx(gidz,gidy,gidx-4))+
               hc5*(vx(gidz,gidy,gidx+4)-vx(gidz,gidy,gidx-5)))/DH;
        
        vyy = (hc1*(vy(gidz,gidy,gidx)  -vy(gidz,gidy-1,gidx))+
               hc2*(vy(gidz,gidy+1,gidx)-vy(gidz,gidy-2,gidx))+
               hc3*(vy(gidz,gidy+2,gidx)-vy(gidz,gidy-3,gidx))+
               hc4*(vy(gidz,gidy+3,gidx)-vy(gidz,gidy-4,gidx))+
               hc5*(vy(gidz,gidy+4,gidx)-vy(gidz,gidy-5,gidx)))/DH;
        
        vzz = (hc1*(vz(gidz,gidy,gidx)  -vz(gidz-1,gidy,gidx))+
               hc2*(vz(gidz+1,gidy,gidx)-vz(gidz-2,gidy,gidx))+
               hc3*(vz(gidz+2,gidy,gidx)-vz(gidz-3,gidy,gidx))+
               hc4*(vz(gidz+3,gidy,gidx)-vz(gidz-4,gidy,gidx))+
               hc5*(vz(gidz+4,gidy,gidx)-vz(gidz-5,gidy,gidx)))/DH;
        
        
    }
#elif fdoh==6
    {
        vxx = (hc1*(vx(gidz,gidy,gidx)  -vx(gidz,gidy,gidx-1))+
               hc2*(vx(gidz,gidy,gidx+1)-vx(gidz,gidy,gidx-2))+
               hc3*(vx(gidz,gidy,gidx+2)-vx(gidz,gidy,gidx-3))+
               hc4*(vx(gidz,gidy,gidx+3)-vx(gidz,gidy,gidx-4))+
               hc5*(vx(gidz,gidy,gidx+4)-vx(gidz,gidy,gidx-5))+
               hc6*(vx(gidz,gidy,gidx+5)-vx(gidz,gidy,gidx-6)))/DH;
        
        vyy = (hc1*(vy(gidz,gidy,gidx)  -vy(gidz,gidy-1,gidx))+
               hc2*(vy(gidz,gidy+1,gidx)-vy(gidz,gidy-2,gidx))+
               hc3*(vy(gidz,gidy+2,gidx)-vy(gidz,gidy-3,gidx))+
               hc4*(vy(gidz,gidy+3,gidx)-vy(gidz,gidy-4,gidx))+
               hc5*(vy(gidz,gidy+4,gidx)-vy(gidz,gidy-5,gidx))+
               hc6*(vy(gidz,gidy+5,gidx)-vy(gidz,gidy-6,gidx)))/DH;
        
        vzz = (hc1*(vz(gidz,gidy,gidx)  -vz(gidz-1,gidy,gidx))+
               hc2*(vz(gidz+1,gidy,gidx)-vz(gidz-2,gidy,gidx))+
               hc3*(vz(gidz+2,gidy,gidx)-vz(gidz-3,gidy,gidx))+
               hc4*(vz(gidz+3,gidy,gidx)-vz(gidz-4,gidy,gidx))+
               hc5*(vz(gidz+4,gidy,gidx)-vz(gidz-5,gidy,gidx))+
               hc6*(vz(gidz+5,gidy,gidx)-vz(gidz-6,gidy,gidx)))/DH;
    }
#endif


// Absorbing boundary
#if abstype==2
{
    if (gidy-fdoh<nab){
        sxx(gidz,gidy,gidx)*=1.0/taper[gidy-fdoh];
        syy(gidz,gidy,gidx)*=1.0/taper[gidy-fdoh];
    }
    
    if (gidy>NY-nab-fdoh-1){
        sxx(gidz,gidy,gidx)*=1.0/taper[NY-fdoh-gidy-1];
        syy(gidz,gidy,gidx)*=1.0/taper[NY-fdoh-gidy-1];
    }
    
#if dev==0 & MYLOCALID==0
    if (gidx-fdoh<nab){
        sxx(gidz,gidy,gidx)*=1.0/taper[gidx-fdoh];
        syy(gidz,gidy,gidx)*=1.0/taper[gidx-fdoh];
    }
#endif
    
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-nab-fdoh-1){
        sxx(gidz,gidy,gidx)*=1.0/taper[NX-fdoh-gidx-1];
        syy(gidz,gidy,gidx)*=1.0/taper[NX-fdoh-gidx-1];
    }
#endif
}
#endif
    
// Correct spatial derivatives to implement CPML
#if abs_type==1
    {
        int i,j,k,ind;
        
        if (gidy-fdoh<nab){
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;
            
            vyy = vyy / K_y[j] + psi_vyy(k,j,i);
        }
        
        else if (gidy>NY-nab-fdoh-1){
            
            i =gidx-fdoh;
            j =gidy - NY+nab+fdoh+nab;
            k =gidz-fdoh;
            ind=2*nab-1-j;
            vyy = vyy / K_y[ind+1] + psi_vyy(k,j,i);
        }
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            
            i =gidx-fdoh;
            j =gidy-fdoh;
            k =gidz-fdoh;

            vxx = vxx / K_x[i] + psi_vxx(k,j,i);
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            
            i =gidx - NX+nab+fdoh+nab;
            j =gidy-fdoh;
            k =gidz-fdoh;
            ind=2*nab-1-i;
            vxx = vxx /K_x[ind+1] + psi_vxx(k,j,i);
        }
#endif
    }
#endif

    
#if Lve==0
				f=u(gidz, gidy, gidx)*2.0;
				g=pi(gidz, gidy, gidx);
				h=-(DT*(g-f)*(g-f)*(vxx+vyy)/g)-(DT*(g-f)*vzz);
				sxx(gidz, gidy, gidx)+=h;
				syy(gidz, gidy, gidx)+=h;
#else
    float b,d,e;
    /* partially updating sxx and syy in the same way*/
    f=u(gidz, gidy, gidx)*2.0*(1.0+L*taus(gidz, gidy, gidx));
    g=pi(gidz, gidy, gidx)*(1.0+L*taup(gidz, gidy, gidx));
    h=-(DT*(g-f)*(g-f)*(vxx+vyy)/g)-(DT*(g-f)*vzz);
    sxx(gidz, gidy, gidx)+=h-(DT/2.0*rxx(gidz, gidy, gidx));
    syy(gidz, gidy, gidx)+=h-(DT/2.0*ryy(gidz, gidy, gidx));
    
    /* updating the memory-variable rxx, ryy at the free surface */
    
    d=2.0*u(gidz, gidy, gidx)*taus(gidz, gidy, gidx);
    e=pi(gidz, gidy, gidx)*taup(gidz, gidy, gidx);
    for (m=0;m<Lve;m++){
        b=eta[m]/(1.0+(eta[m]*0.5));
        h=b*(((d-e)*((f/g)-1.0)*(vxx+vyy))-((d-e)*vzz));
        rxx(gidz, gidy, gidx)+=h;
        ryy(gidz, gidy, gidx)+=h;
    }
    
    /*completely updating the stresses sxx and syy */
    sxx(gidz, gidy, gidx)+=(DT/2.0*rxx(gidz, gidy, gidx));
    syy(gidz, gidy, gidx)+=(DT/2.0*ryy(gidz, gidy, gidx));
    
#endif

// Absorbing boundary
#if abstype==2
    {
        if (gidy-fdoh<nab){
            sxx(gidz,gidy,gidx)*=taper[gidy-fdoh];
            syy(gidz,gidy,gidx)*=taper[gidy-fdoh];
        }
        
        if (gidy>NY-nab-fdoh-1){
            sxx(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
            syy(gidz,gidy,gidx)*=taper[NY-fdoh-gidy-1];
        }
        
#if dev==0 & MYLOCALID==0
        if (gidx-fdoh<nab){
            sxx(gidz,gidy,gidx)*=taper[gidx-fdoh];
            syy(gidz,gidy,gidx)*=taper[gidx-fdoh];
        }
#endif
        
#if dev==num_devices-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-nab-fdoh-1){
            sxx(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
            syy(gidz,gidy,gidx)*=taper[NX-fdoh-gidx-1];
        }
#endif
    }
#endif
    
}



