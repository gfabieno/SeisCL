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
#define vxr(z,x)  vxr[(x)*(NZ)+(z)]
#define vzr(z,x)  vzr[(x)*(NZ)+(z)]
#define sxxr(z,x) sxxr[(x)*(NZ)+(z)]
#define szzr(z,x) szzr[(x)*(NZ)+(z)]
#define sxzr(z,x) sxzr[(x)*(NZ)+(z)]

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



__kernel void surface_adj(__global float *vxr,         __global float *vzr,
                          __global float *sxxr,        __global float *szzr,
                          __global float *sxzr,        __global float *M,
                          __global float *rip,        __global float *rkp,
                          __global float *mu,         __global float *rxx,
                          __global float *rzz,        __global float *taus,
                          __global float *taup,       __global float *eta,
                          __global float *K_x,        __global float *psi_vx_x,
                          __global float *taper)
{
    /*Indice definition */
    int gidx = get_global_id(0) + FDOH;
    int gidz=FDOH;

    /* Global work size is padded to be a multiple of local work size. The padding elements must not be updated */
    if ( gidx>(NX-FDOH-1) ){
        return;
    }

    float f, g, hx, hz;
    float  vxx, vzz;
    int m;
    float lszz[2*FDOH+1], lsxx[2*FDOH+1];
    float lmu[2*FDOH+1], lM[2*FDOH+1];
    float temp1, temp2, a;

    //Preadjoint transformations
    lszz[FDOH] = szzr(gidz,  gidx);
    lsxx[FDOH] = sxxr(gidz,  gidx);
    lmu[FDOH] = mu(gidz,  gidx);
    lM[FDOH] = M(gidz,  gidx);
    for (m=1; m<=FDOH; m++) {
        lszz[FDOH-m] = szzr(gidz-m,  gidx);
        lszz[FDOH+m] = szzr(gidz+m,  gidx);
        lsxx[FDOH-m] = sxxr(gidz-m,  gidx);
        lsxx[FDOH+m] = sxxr(gidz+m,  gidx);
        lmu[FDOH-m] = lmu[FDOH+m] = mu(gidz+m,  gidx);
        lM[FDOH-m] = lM[FDOH+m] = M(gidz+m,  gidx);
    }
    temp1 = lsxx[FDOH];
    temp2 = lszz[FDOH];
    a = 1.0/( lM[FDOH]*lM[FDOH] - (lM[FDOH]-2*lmu[FDOH])*(lM[FDOH]-2*lmu[FDOH]));
    lsxx[FDOH] = a * ( lM[FDOH] * temp1 + (-lM[FDOH] + 2 * lmu[FDOH]) * temp2 );
    lszz[FDOH] = a * ( lM[FDOH] * temp2 + (-lM[FDOH] + 2 * lmu[FDOH]) * temp1 );
    for (m=1; m<=FDOH; m++) {
        a = 1.0/( lM[FDOH+m]*lM[FDOH+m] - (lM[FDOH+m]-2*lmu[FDOH+m])*(lM[FDOH+m]-2*lmu[FDOH+m]));
        temp1 = lsxx[FDOH+m];
        temp2 = lszz[FDOH+m];
        lsxx[FDOH+m] = a * (  lM[FDOH+m] * temp1 + (-lM[FDOH+m] + 2 * lmu[FDOH+m]) * temp2 );
        lszz[FDOH+m] = a * (  lM[FDOH+m] * temp2 + (-lM[FDOH+m] + 2 * lmu[FDOH+m]) * temp1 );
        temp1 = lsxx[FDOH-m];
        temp2 = lszz[FDOH-m];
        lsxx[FDOH-m] = a * (  lM[FDOH-m] * temp1 + (-lM[FDOH-m] + 2 * lmu[FDOH-m]) * temp2 );
        lszz[FDOH-m] = a * (  lM[FDOH-m] * temp2 + (-lM[FDOH-m] + 2 * lmu[FDOH-m]) * temp1 );
    }


    /*Mirroring the components of the stress tensor to make
     a stress free surface (method of imaging, Levander, 1988)*/
    lszz[FDOH]=0.0;
//#if LVE>0
//    rzz(gidz, gidx)=0.0;
//#endif

    for (m=1; m<=FDOH; m++) {
        lszz[FDOH+m]+=-lszz[FDOH-m];
        sxzr(gidz+m-1,  gidx)+=-sxzr(gidz-m, gidx);
        lszz[gidz-m]=0;
        sxzr(gidz-m,  gidx)=0;
    }


    f=lmu[FDOH]*2.0;
    g=lM[FDOH];
    hx=-((g-f)*(g-f)/g);
    hz=-(g-f);

// Absorbing boundary
#if ABS_TYPE==2
    {

#if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            hx*=taper[gidx-FDOH];
            hz*=taper[gidx-FDOH];
        }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            hx*=taper[NX-FDOH-gidx-1];
            hz*=taper[NX-FDOH-gidx-1];
        }
#endif
    }
#endif

#if   FDOH==1
    {
        vxr(gidz,gidx) += hx*lsxx[FDOH] * rip(gidz,gidx);
        vxr(gidz,gidx-1) += -hx*lsxx[FDOH] * rip(gidz,gidx-1);
        vzr(gidz,gidx) += hz*lsxx[FDOH] * rkp(gidz,gidx);
        vzr(gidz-1,gidx) += -hz*lsxx[FDOH] * rkp(gidz-1,gidx);

    }
#elif FDOH==2
    {
        vxr(gidz,gidx) += HC1*hx*lsxx[FDOH] * rip(gidz,gidx);
        vxr(gidz,gidx-1) += -HC1*hx*lsxx[FDOH] * rip(gidz,gidx-1);
        vxr(gidz,gidx+1) += HC2*hx*lsxx[FDOH] * rip(gidz,gidx+1);
        vxr(gidz,gidx-2) += -HC2*hx*lsxx[FDOH] * rip(gidz,gidx-2);

        vzr(gidz,gidx) += HC1*hz*lsxx[FDOH] * rkp(gidz,gidx);
        vzr(gidz-1,gidx) += -HC1*hz*lsxx[FDOH] * rkp(gidz-1,gidx);
        vzr(gidz+1,gidx) += HC2*hz*lsxx[FDOH] * rkp(gidz+1,gidx);
        vzr(gidz-2,gidx) += -HC2*hz*lsxx[FDOH] * rkp(gidz-2,gidx);

    }
#elif FDOH==3
    {
        vxr(gidz,gidx) += HC1*hx*lsxx[FDOH] * rip(gidz,gidx);
        vxr(gidz,gidx-1) += -HC1*hx*lsxx[FDOH] * rip(gidz,gidx-1);
        vxr(gidz,gidx+1) += HC2*hx*lsxx[FDOH] * rip(gidz,gidx+1);
        vxr(gidz,gidx-2) += -HC2*hx*lsxx[FDOH] * rip(gidz,gidx-2);
        vxr(gidz,gidx+2) += HC3*hx*lsxx[FDOH] * rip(gidz,gidx+2);
        vxr(gidz,gidx-3) += -HC3*hx*lsxx[FDOH] * rip(gidz,gidx-3);

        vzr(gidz,gidx) += HC1*hz*lsxx[FDOH] * rkp(gidz,gidx);
        vzr(gidz-1,gidx) += -HC1*hz*lsxx[FDOH] * rkp(gidz-1,gidx);
        vzr(gidz+1,gidx) += HC2*hz*lsxx[FDOH] * rkp(gidz+1,gidx);
        vzr(gidz-2,gidx) += -HC2*hz*lsxx[FDOH] * rkp(gidz-2,gidx);
        vzr(gidz+2,gidx) += HC3*hz*lsxx[FDOH] * rkp(gidz+2,gidx);
        vzr(gidz-3,gidx) += -HC3*hz*lsxx[FDOH] * rkp(gidz-3,gidx);

    }
#elif FDOH==4
    {
        vxr(gidz,gidx) += HC1*hx*lsxx[FDOH] * rip(gidz,gidx);
        vxr(gidz,gidx-1) += -HC1*hx*lsxx[FDOH] * rip(gidz,gidx-1);
        vxr(gidz,gidx+1) += HC2*hx*lsxx[FDOH] * rip(gidz,gidx+1);
        vxr(gidz,gidx-2) += -HC2*hx*lsxx[FDOH] * rip(gidz,gidx-2);
        vxr(gidz,gidx+2) += HC3*hx*lsxx[FDOH] * rip(gidz,gidx+2);
        vxr(gidz,gidx-3) += -HC3*hx*lsxx[FDOH] * rip(gidz,gidx-3);
        vxr(gidz,gidx+3) += HC4*hx*lsxx[FDOH] * rip(gidz,gidx+3);
        vxr(gidz,gidx-4) += -HC4*hx*lsxx[FDOH] * rip(gidz,gidx-4);

        vzr(gidz,gidx) += HC1*hz*lsxx[FDOH] * rkp(gidz,gidx);
        vzr(gidz-1,gidx) += -HC1*hz*lsxx[FDOH] * rkp(gidz-1,gidx);
        vzr(gidz+1,gidx) += HC2*hz*lsxx[FDOH] * rkp(gidz+1,gidx);
        vzr(gidz-2,gidx) += -HC2*hz*lsxx[FDOH] * rkp(gidz-2,gidx);
        vzr(gidz+2,gidx) += HC3*hz*lsxx[FDOH] * rkp(gidz+2,gidx);
        vzr(gidz-3,gidx) += -HC3*hz*lsxx[FDOH] * rkp(gidz-3,gidx);
        vzr(gidz+3,gidx) += HC4*hz*lsxx[FDOH] * rkp(gidz+3,gidx);
        vzr(gidz-4,gidx) += -HC4*hz*lsxx[FDOH] * rkp(gidz-4,gidx);

    }
#elif FDOH==5
    {
        vxr(gidz,gidx) += HC1*hx*lsxx[FDOH] * rip(gidz,gidx);
        vxr(gidz,gidx-1) += -HC1*hx*lsxx[FDOH] * rip(gidz,gidx-1);
        vxr(gidz,gidx+1) += HC2*hx*lsxx[FDOH] * rip(gidz,gidx+1);
        vxr(gidz,gidx-2) += -HC2*hx*lsxx[FDOH] * rip(gidz,gidx-2);
        vxr(gidz,gidx+2) += HC3*hx*lsxx[FDOH] * rip(gidz,gidx+2);
        vxr(gidz,gidx-3) += -HC3*hx*lsxx[FDOH] * rip(gidz,gidx-3);
        vxr(gidz,gidx+3) += HC4*hx*lsxx[FDOH] * rip(gidz,gidx+3);
        vxr(gidz,gidx-4) += -HC4*hx*lsxx[FDOH] * rip(gidz,gidx-4);
        vxr(gidz,gidx+4) += HC5*hx*lsxx[FDOH] * rip(gidz,gidx+4);
        vxr(gidz,gidx-5) += -HC5*hx*lsxx[FDOH] * rip(gidz,gidx-5);

        vzr(gidz,gidx) += HC1*hz*lsxx[FDOH] * rkp(gidz,gidx);
        vzr(gidz-1,gidx) += -HC1*hz*lsxx[FDOH] * rkp(gidz-1,gidx);
        vzr(gidz+1,gidx) += HC2*hz*lsxx[FDOH] * rkp(gidz+1,gidx);
        vzr(gidz-2,gidx) += -HC2*hz*lsxx[FDOH] * rkp(gidz-2,gidx);
        vzr(gidz+2,gidx) += HC3*hz*lsxx[FDOH] * rkp(gidz+2,gidx);
        vzr(gidz-3,gidx) += -HC3*hz*lsxx[FDOH] * rkp(gidz-3,gidx);
        vzr(gidz+3,gidx) += HC4*hz*lsxx[FDOH] * rkp(gidz+3,gidx);
        vzr(gidz-4,gidx) += -HC4*hz*lsxx[FDOH] * rkp(gidz-4,gidx);
        vzr(gidz+4,gidx) += HC5*hz*lsxx[FDOH] * rkp(gidz+4,gidx);
        vzr(gidz-5,gidx) += -HC5*hz*lsxx[FDOH] * rkp(gidz-5,gidx);

    }
#elif FDOH==6
    {
        vxr(gidz,gidx) += HC1*hx*lsxx[FDOH] * rip(gidz,gidx);
        vxr(gidz,gidx-1) += -HC1*hx*lsxx[FDOH] * rip(gidz,gidx-1);
        vxr(gidz,gidx+1) += HC2*hx*lsxx[FDOH] * rip(gidz,gidx+1);
        vxr(gidz,gidx-2) += -HC2*hx*lsxx[FDOH] * rip(gidz,gidx-2);
        vxr(gidz,gidx+2) += HC3*hx*lsxx[FDOH] * rip(gidz,gidx+2);
        vxr(gidz,gidx-3) += -HC3*hx*lsxx[FDOH] * rip(gidz,gidx-3);
        vxr(gidz,gidx+3) += HC4*hx*lsxx[FDOH] * rip(gidz,gidx+3);
        vxr(gidz,gidx-4) += -HC4*hx*lsxx[FDOH] * rip(gidz,gidx-4);
        vxr(gidz,gidx+4) += HC5*hx*lsxx[FDOH] * rip(gidz,gidx+4);
        vxr(gidz,gidx-5) += -HC5*hx*lsxx[FDOH] * rip(gidz,gidx-5);
        vxr(gidz,gidx+5) += HC6*hx*lsxx[FDOH] * rip(gidz,gidx+5);
        vxr(gidz,gidx-6) += -HC6*hx*lsxx[FDOH] * rip(gidz,gidx-6);

        vzr(gidz,gidx) += HC1*hz*lsxx[FDOH] * rkp(gidz,gidx);
        vzr(gidz-1,gidx) += -HC1*hz*lsxx[FDOH] * rkp(gidz-1,gidx);
        vzr(gidz+1,gidx) += HC2*hz*lsxx[FDOH] * rkp(gidz+1,gidx);
        vzr(gidz-2,gidx) += -HC2*hz*lsxx[FDOH] * rkp(gidz-2,gidx);
        vzr(gidz+2,gidx) += HC3*hz*lsxx[FDOH] * rkp(gidz+2,gidx);
        vzr(gidz-3,gidx) += -HC3*hz*lsxx[FDOH] * rkp(gidz-3,gidx);
        vzr(gidz+3,gidx) += HC4*hz*lsxx[FDOH] * rkp(gidz+3,gidx);
        vzr(gidz-4,gidx) += -HC4*hz*lsxx[FDOH] * rkp(gidz-4,gidx);
        vzr(gidz+4,gidx) += HC5*hz*lsxx[FDOH] * rkp(gidz+4,gidx);
        vzr(gidz-5,gidx) += -HC5*hz*lsxx[FDOH] * rkp(gidz-5,gidx);
        vzr(gidz+5,gidx) += HC6*hz*lsxx[FDOH] * rkp(gidz+5,gidx);
        vzr(gidz-6,gidx) += -HC6*hz*lsxx[FDOH] * rkp(gidz-6,gidx);

    }
#endif


    //Perform the post-adjoint transformation
    temp1 = lsxx[FDOH];
    temp2 = lszz[FDOH];
    lsxx[FDOH] =  ( lM[FDOH] * temp1 + ( lM[FDOH] - 2 * lmu[FDOH]) * temp2 );
    lszz[FDOH] =  ( lM[FDOH] * temp2 + ( lM[FDOH] - 2 * lmu[FDOH]) * temp1 );
    for (m=1; m<=FDOH; m++) {

        temp1 = lsxx[FDOH+m];
        temp2 = lszz[FDOH+m];
        lsxx[FDOH+m] =  ( lM[FDOH+m] * temp1 + ( lM[FDOH+m] - 2 * lmu[FDOH+m]) * temp2 );
        lszz[FDOH+m] =  ( lM[FDOH+m] * temp2 + ( lM[FDOH+m] - 2 * lmu[FDOH+m]) * temp1 );
        temp1 = lsxx[FDOH-m];
        temp2 = lszz[FDOH-m];
        lsxx[FDOH-m] =  ( lM[FDOH-m] * temp1 + ( lM[FDOH-m] - 2 * lmu[FDOH-m]) * temp2 );
        lszz[FDOH-m] =  ( lM[FDOH-m] * temp2 + ( lM[FDOH-m] - 2 * lmu[FDOH-m]) * temp1 );
    }

    szzr(gidz,  gidx) = lszz[FDOH];
    sxxr(gidz,  gidx) = lsxx[FDOH];
    for (m=1; m<=FDOH; m++) {
        szzr(gidz-m,  gidx) = lszz[FDOH-m];
        szzr(gidz+m,  gidx) = lszz[FDOH+m];
        sxxr(gidz-m,  gidx) = lsxx[FDOH-m];
        sxxr(gidz+m,  gidx) = lsxx[FDOH+m];
    }


//    // Correct spatial derivatives to implement CPML
//#if ABS_TYPE==1
//    {
//        int i,k,ind;
//#if DEVID==0 & MYLOCALID==0
//        if (gidx-FDOH<NAB){
//
//            i =gidx-FDOH;
//            k =gidz-FDOH;
//
//            vxx = vxx / K_x[i] + psi_vx_x(k,i);
//        }
//#endif
//
//#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//        if (gidx>NX-NAB-FDOH-1){
//
//            i =gidx - NX+NAB+FDOH+NAB;
//            k =gidz-FDOH;
//            ind=2*NAB-1-i;
//            vxx = vxx /K_x[ind+1] + psi_vx_x(k,i);
//        }
//#endif
//    }
//#endif


//    float b,d,e;
//    /* partially updating sxx  in the same way*/
//    f=mu(gidz,  gidx)*2.0*(1.0+L*taus(gidz,  gidx));
//    g=M(gidz,  gidx)*(1.0+L*taup(gidz,  gidx));
//    h=-((g-f)*(g-f)*(vxx)/g)-((g-f)*vzz);
//    sxx(gidz,  gidx)+=h-(DT/2.0*rxx(gidz,  gidx));
//
//    /* updating the memory-variable rxx at the free surface */
//
//    d=2.0*mu(gidz,  gidx)*taus(gidz,  gidx);
//    e=M(gidz,  gidx)*taup(gidz,  gidx);
//    for (m=0;m<LVE;m++){
//        b=eta[m]/(1.0+(eta[m]*0.5));
//        h=b*(((d-e)*((f/g)-1.0)*(vxx+vyy))-((d-e)*vzz));
//        rxx(gidz,  gidx)+=h;
//    }
//
//    /*completely updating the stresses sxx  */
//    sxx(gidz,  gidx)+=(DT/2.0*rxx(gidz,  gidx));


}



