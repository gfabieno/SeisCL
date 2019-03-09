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
#define rho(z,x)    rho[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rip(z,x)    rip[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define rkp(z,x)    rkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define muipkp(z,x) muipkp[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define mu(z,x)        mu[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define M(z,x)      M[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define grad(z,x)  grad[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define grads(z,x) grads[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define amp1(z,x)  amp1[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]
#define amp2(z,x)  amp2[((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)]

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

#if LOCAL_OFF==0

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

#define psi_vx_x(z,x) psi_vx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_vzx(z,x) psi_vzx[(x)*(NZ-2*FDOH)+(z)]

#define psi_vxz(z,x) psi_vxz[(x)*(2*NAB)+(z)]
#define psi_vzz(z,x) psi_vzz[(x)*(2*NAB)+(z)]


#define PI (3.141592653589793238462643383279502884197169)
#define signals(y,x) signals[(y)*NT+(x)]



__kernel void surface(        __global float *vx,         __global float *vz,
                              __global float *sxx,        __global float *szz,
                              __global float *sxz,        __global float *M,
                              __global float *mu,         __global float *rxx,
                              __global float *rzz,        __global float *taus,
                              __global float *taup,       __global float *eta,
                              __global float *K_x,        __global float *psi_vx_x,
                              __global float *taper, pdir)
{
    /*Indice definition */
    int gidx = get_global_id(0) + FDOH;
    int gidz=FDOH;
    
    /* Global work size is padded to be a multiple of local work size. The padding elements must not be updated */
    if ( gidx>(NX-FDOH-1) ){
        return;
    }
    
    float f, g, h;
    float sump;
    float  vxx, vzz;
    int m;
    
    /*Mirroring the components of the stress tensor to make
     a stress free surface (method of imaging, Levander, 1988)*/
    szz(gidz, gidx)=0.0;
#if LVE>0
    for (int l=0; l<LVE; l++){
        rzz(gidz, gidx, l)=0.0;
    }
#endif
    
    for (m=1; m<=FDOH; m++) {
        szz(gidz-m,  gidx)=-szz(gidz+m,  gidx);
        sxz(gidz-m,  gidx)=-sxz(gidz+m-1, gidx);
    }

#if   FDOH==1
    {
        vxx = (vx(gidz,gidx)-vx(gidz,gidx-1));
        vzz = (vz(gidz,gidx)-vz(gidz-1,gidx));
    }
#elif FDOH==2
    {
        vxx = (HC1*(vx(gidz,gidx)  -vx(gidz,gidx-1))+
               HC2*(vx(gidz,gidx+1)-vx(gidz,gidx-2)));


        vzz = (HC1*(vz(gidz,gidx)  -vz(gidz-1,gidx))+
               HC2*(vz(gidz+1,gidx)-vz(gidz-2,gidx)));
    }
#elif FDOH==3
    {
        vxx = (HC1*(vx(gidz,gidx)  -vx(gidz,gidx-1))+
               HC2*(vx(gidz,gidx+1)-vx(gidz,gidx-2))+
               HC3*(vx(gidz,gidx+2)-vx(gidz,gidx-3)));

        vzz = (HC1*(vz(gidz,gidx)-vz(gidz-1,gidx))+
               HC2*(vz(gidz+1,gidx)-vz(gidz-2,gidx))+
               HC3*(vz(gidz+2,gidx)-vz(gidz-3,gidx)));

    }
#elif FDOH==4
    {
        vxx = (HC1*(vx(gidz,gidx)  -vx(gidz,gidx-1))+
               HC2*(vx(gidz,gidx+1)-vx(gidz,gidx-2))+
               HC3*(vx(gidz,gidx+2)-vx(gidz,gidx-3))+
               HC4*(vx(gidz,gidx+3)-vx(gidz,gidx-4)));

        vzz = (HC1*(vz(gidz,gidx)  -vz(gidz-1,gidx))+
               HC2*(vz(gidz+1,gidx)-vz(gidz-2,gidx))+
               HC3*(vz(gidz+2,gidx)-vz(gidz-3,gidx))+
               HC4*(vz(gidz+3,gidx)-vz(gidz-4,gidx)));
    }
#elif FDOH==5
    {
        vxx = (HC1*(vx(gidz,gidx)  -vx(gidz,gidx-1))+
               HC2*(vx(gidz,gidx+1)-vx(gidz,gidx-2))+
               HC3*(vx(gidz,gidx+2)-vx(gidz,gidx-3))+
               HC4*(vx(gidz,gidx+3)-vx(gidz,gidx-4))+
               HC5*(vx(gidz,gidx+4)-vx(gidz,gidx-5)));


        vzz = (HC1*(vz(gidz,gidx)  -vz(gidz-1,gidx))+
               HC2*(vz(gidz+1,gidx)-vz(gidz-2,gidx))+
               HC3*(vz(gidz+2,gidx)-vz(gidz-3,gidx))+
               HC4*(vz(gidz+3,gidx)-vz(gidz-4,gidx))+
               HC5*(vz(gidz+4,gidx)-vz(gidz-5,gidx)));


    }
#elif FDOH==6
    {
        vxx = (HC1*(vx(gidz,gidx)  -vx(gidz,gidx-1))+
               HC2*(vx(gidz,gidx+1)-vx(gidz,gidx-2))+
               HC3*(vx(gidz,gidx+2)-vx(gidz,gidx-3))+
               HC4*(vx(gidz,gidx+3)-vx(gidz,gidx-4))+
               HC5*(vx(gidz,gidx+4)-vx(gidz,gidx-5))+
               HC6*(vx(gidz,gidx+5)-vx(gidz,gidx-6)));


        vzz = (HC1*(vz(gidz,gidx)  -vz(gidz-1,gidx))+
               HC2*(vz(gidz+1,gidx)-vz(gidz-2,gidx))+
               HC3*(vz(gidz+2,gidx)-vz(gidz-3,gidx))+
               HC4*(vz(gidz+3,gidx)-vz(gidz-4,gidx))+
               HC5*(vz(gidz+4,gidx)-vz(gidz-5,gidx))+
               HC6*(vz(gidz+5,gidx)-vz(gidz-6,gidx)));
    }
#endif



    // Correct spatial derivatives to implement CPML
#if ABS_TYPE==1
    {
        int i,k,ind;
#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){

            i =gidx-FDOH;
            k =gidz-FDOH;

            vxx = vxx / K_x[i] + psi_vx_x(k,i);
        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){

            i =gidx - NX+NAB+FDOH+NAB;
            k =gidz-FDOH;
            ind=2*NAB-1-i;
            vxx = vxx /K_x[ind+1] + psi_vx_x(k,i);
        }
#endif
    }
#endif

#if LVE==0
    f=mu(gidz,  gidx)*2.0;
    g=M(gidz,  gidx);
    h=-((g-f)*(g-f)*(vxx)/g)-((g-f)*vzz);
    // Absorbing boundary
#if ABS_TYPE==2
    {

#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            h*=taper[gidx-FDOH];
        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            h*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif
    sxx(gidz,  gidx)+=pdir*h;
#else
    float b,d,e;
    /* partially updating sxx  in the same way*/
    f=mu(gidz,  gidx)*2.0*(1.0+LVE*taus(gidz,  gidx));
    g=M(gidz,  gidx)*(1.0+LVE*taup(gidz,  gidx));
    h=-((g-f)*(g-f)*(vxx)/g)-((g-f)*vzz);
#if ABS_TYPE==2
    {

#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            h*=taper[gidx-FDOH];
        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            h*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif
    
    sump=0;
    for (m=0;m<LVE;m++){
        sump+=rxx(gidz,  gidx, m);
    }
    sxx(gidz,  gidx)+=pdir* (h - DT/2.0*sump);
    
    
    /* updating the memory-variable rxx at the free surface */
    d=2.0*mu(gidz,  gidx)*taus(gidz,  gidx);
    e=M(gidz,  gidx)*taup(gidz,  gidx);
    
    sump=0;
    for (m=0;m<LVE;m++){
        b=eta[m]/(1.0+(eta[m]*0.5));
        h=b*(((d-e)*((f/g)-1.0)*vxx)-((d-e)*vzz));
        rxx(gidz,  gidx, m)+=pdir*h;
        /*completely updating the stresses sxx  */
    }
    sxx(gidz,  gidx)+=pdir*(DT/2.0*sump);
#endif
    
}



