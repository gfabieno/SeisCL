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

//TODO: Adjoint surface for viscoelastic propagation

#define psi_vx_x(z,x) psi_vx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_vzx(z,x) psi_vzx[(x)*(NZ-2*FDOH)+(z)]

#define psi_vxz(z,x) psi_vxz[(x)*(2*NAB)+(z)]
#define psi_vzz(z,x) psi_vzz[(x)*(2*NAB)+(z)]

#define indv(z,x)  (x)*(NZ)+(z)
#define indp(z,x)  ((x)-FDOH)*(NZ-2*FDOH)+((z)-FDOH)

FUNDEF void surface_adj(GLOBARG float *vxr,        GLOBARG float *vzr,
                        GLOBARG float *sxxr,       GLOBARG float *szzr,
                        GLOBARG float *sxzr,       GLOBARG float *M,
                        GLOBARG float *rip,        GLOBARG float *rkp,
                        GLOBARG float *mu,         GLOBARG float *rxx,
                        GLOBARG float *rzz,        GLOBARG float *taus,
                        GLOBARG float *taup,       GLOBARG float *eta,
                        GLOBARG float *K_x,        GLOBARG float *psi_vx_x,
                        GLOBARG float *taper)
{
    /*Indice definition */
    #ifdef __OPENCL_VERSION__
    int gidx = get_global_id(0) + FDOH;
    #else
    int gidx = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    #endif
    int gidz=FDOH;

    /* Global work size is padded to be a multiple of local work size.
     The padding elements must not be updated */
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
    lszz[FDOH] = szzr[indv(gidz,  gidx)];
    lsxx[FDOH] = sxxr[indv(gidz,  gidx)];
    lmu[FDOH] = mu[indp(gidz,  gidx)];
    lM[FDOH] = M[indp(gidz,  gidx)];
    for (m=1; m<=FDOH; m++) {
        lszz[FDOH-m] = szzr[indv(gidz-m,  gidx)];
        lszz[FDOH+m] = szzr[indv(gidz+m,  gidx)];
        lsxx[FDOH-m] = sxxr[indv(gidz-m,  gidx)];
        lsxx[FDOH+m] = sxxr[indv(gidz+m,  gidx)];
        lmu[FDOH-m] = lmu[FDOH+m] = mu[indp(gidz+m,  gidx)];
        lM[FDOH-m] = lM[FDOH+m] = M[indp(gidz+m,  gidx)];
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
        sxzr[indv(gidz+m-1,gidx)]+=-sxzr[indv(gidz-m, gidx)];
        lszz[gidz-m]=0;
        sxzr[indv(gidz-m,gidx)]=0;
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
        vxr[indv(gidz,gidx)] += hx*lsxx[FDOH] * rip[indp(gidz, gidx)];
        vxr[indv(gidz,gidx-1)] += -hx*lsxx[FDOH] * rip[indp(gidz, gidx-1)];
        vzr[indv(gidz,gidx)] += hz*lsxx[FDOH] * rkp[indp(gidz, gidx)];
        vzr[indv(gidz-1,gidx)] += -hz*lsxx[FDOH] * rkp[indp(gidz-1, gidx)];

    }
#elif FDOH==2
    {
        vxr[indv(gidz,gidx)] += HC1*hx*lsxx[FDOH] * rip[indp(gidz, gidx)];
        vxr[indv(gidz,gidx-1)] += -HC1*hx*lsxx[FDOH] * rip[indp(gidz, gidx-1)];
        vxr[indv(gidz,gidx+1)] += HC2*hx*lsxx[FDOH] * rip[indp(gidz, gidx+1)];
        vxr[indv(gidz,gidx-2)] += -HC2*hx*lsxx[FDOH] * rip[indp(gidz, gidx-2)];

        vzr[indv(gidz,gidx)] += HC1*hz*lsxx[FDOH] * rkp[indp(gidz, gidx)];
        vzr[indv(gidz-1,gidx)] += -HC1*hz*lsxx[FDOH] * rkp[indp(gidz-1, gidx)];
        vzr[indv(gidz+1,gidx)] += HC2*hz*lsxx[FDOH] * rkp[indp(gidz+1, gidx)];
        vzr[indv(gidz-2,gidx)] += -HC2*hz*lsxx[FDOH] * rkp[indp(gidz-2, gidx)];

    }
#elif FDOH==3
    {
        vxr[indv(gidz,gidx)] += HC1*hx*lsxx[FDOH] * rip[indp(gidz, gidx)];
        vxr[indv(gidz,gidx-1)] += -HC1*hx*lsxx[FDOH] * rip[indp(gidz, gidx-1)];
        vxr[indv(gidz,gidx+1)] += HC2*hx*lsxx[FDOH] * rip[indp(gidz, gidx+1)];
        vxr[indv(gidz,gidx-2)] += -HC2*hx*lsxx[FDOH] * rip[indp(gidz, gidx-2)];
        vxr[indv(gidz,gidx+2)] += HC3*hx*lsxx[FDOH] * rip[indp(gidz, gidx+2)];
        vxr[indv(gidz,gidx-3)] += -HC3*hx*lsxx[FDOH] * rip[indp(gidz, gidx-3)];

        vzr[indv(gidz,gidx)] += HC1*hz*lsxx[FDOH] * rkp[indp(gidz, gidx)];
        vzr[indv(gidz-1,gidx)] += -HC1*hz*lsxx[FDOH] * rkp[indp(gidz-1, gidx)];
        vzr[indv(gidz+1,gidx)] += HC2*hz*lsxx[FDOH] * rkp[indp(gidz+1, gidx)];
        vzr[indv(gidz-2,gidx)] += -HC2*hz*lsxx[FDOH] * rkp[indp(gidz-2, gidx)];
        vzr[indv(gidz+2,gidx)] += HC3*hz*lsxx[FDOH] * rkp[indp(gidz+2, gidx)];
        vzr[indv(gidz-3,gidx)] += -HC3*hz*lsxx[FDOH] * rkp[indp(gidz-3, gidx)];

    }
#elif FDOH==4
    {
        vxr[indv(gidz,gidx)] += HC1*hx*lsxx[FDOH] * rip[indp(gidz, gidx)];
        vxr[indv(gidz,gidx-1)] += -HC1*hx*lsxx[FDOH] * rip[indp(gidz, gidx-1)];
        vxr[indv(gidz,gidx+1)] += HC2*hx*lsxx[FDOH] * rip[indp(gidz, gidx+1)];
        vxr[indv(gidz,gidx-2)] += -HC2*hx*lsxx[FDOH] * rip[indp(gidz, gidx-2)];
        vxr[indv(gidz,gidx+2)] += HC3*hx*lsxx[FDOH] * rip[indp(gidz, gidx+2)];
        vxr[indv(gidz,gidx-3)] += -HC3*hx*lsxx[FDOH] * rip[indp(gidz, gidx-3)];
        vxr[indv(gidz,gidx+3)] += HC4*hx*lsxx[FDOH] * rip[indp(gidz, gidx+3)];
        vxr[indv(gidz,gidx-4)] += -HC4*hx*lsxx[FDOH] * rip[indp(gidz, gidx-4)];

        vzr[indv(gidz,gidx)] += HC1*hz*lsxx[FDOH] * rkp[indp(gidz, gidx)];
        vzr[indv(gidz-1,gidx)] += -HC1*hz*lsxx[FDOH] * rkp[indp(gidz-1, gidx)];
        vzr[indv(gidz+1,gidx)] += HC2*hz*lsxx[FDOH] * rkp[indp(gidz+1, gidx)];
        vzr[indv(gidz-2,gidx)] += -HC2*hz*lsxx[FDOH] * rkp[indp(gidz-2, gidx)];
        vzr[indv(gidz+2,gidx)] += HC3*hz*lsxx[FDOH] * rkp[indp(gidz+2, gidx)];
        vzr[indv(gidz-3,gidx)] += -HC3*hz*lsxx[FDOH] * rkp[indp(gidz-3, gidx)];
        vzr[indv(gidz+3,gidx)] += HC4*hz*lsxx[FDOH] * rkp[indp(gidz+3, gidx)];
        vzr[indv(gidz-4,gidx)] += -HC4*hz*lsxx[FDOH] * rkp[indp(gidz-4, gidx)];

    }
#elif FDOH==5
    {
        vxr[indv(gidz,gidx)] += HC1*hx*lsxx[FDOH] * rip[indp(gidz, gidx)];
        vxr[indv(gidz,gidx-1)] += -HC1*hx*lsxx[FDOH] * rip[indp(gidz, gidx-1)];
        vxr[indv(gidz,gidx+1)] += HC2*hx*lsxx[FDOH] * rip[indp(gidz, gidx+1)];
        vxr[indv(gidz,gidx-2)] += -HC2*hx*lsxx[FDOH] * rip[indp(gidz, gidx-2)];
        vxr[indv(gidz,gidx+2)] += HC3*hx*lsxx[FDOH] * rip[indp(gidz, gidx+2)];
        vxr[indv(gidz,gidx-3)] += -HC3*hx*lsxx[FDOH] * rip[indp(gidz, gidx-3)];
        vxr[indv(gidz,gidx+3)] += HC4*hx*lsxx[FDOH] * rip[indp(gidz, gidx+3)];
        vxr[indv(gidz,gidx-4)] += -HC4*hx*lsxx[FDOH] * rip[indp(gidz, gidx-4)];
        vxr[indv(gidz,gidx+4)] += HC5*hx*lsxx[FDOH] * rip[indp(gidz, gidx+4)];
        vxr[indv(gidz,gidx-5)] += -HC5*hx*lsxx[FDOH] * rip[indp(gidz, gidx-5)];

        vzr[indv(gidz,gidx)] += HC1*hz*lsxx[FDOH] * rkp[indp(gidz, gidx)];
        vzr[indv(gidz-1,gidx)] += -HC1*hz*lsxx[FDOH] * rkp[indp(gidz-1, gidx)];
        vzr[indv(gidz+1,gidx)] += HC2*hz*lsxx[FDOH] * rkp[indp(gidz+1, gidx)];
        vzr[indv(gidz-2,gidx)] += -HC2*hz*lsxx[FDOH] * rkp[indp(gidz-2, gidx)];
        vzr[indv(gidz+2,gidx)] += HC3*hz*lsxx[FDOH] * rkp[indp(gidz+2, gidx)];
        vzr[indv(gidz-3,gidx)] += -HC3*hz*lsxx[FDOH] * rkp[indp(gidz-3, gidx)];
        vzr[indv(gidz+3,gidx)] += HC4*hz*lsxx[FDOH] * rkp[indp(gidz+3, gidx)];
        vzr[indv(gidz-4,gidx)] += -HC4*hz*lsxx[FDOH] * rkp[indp(gidz-4, gidx)];
        vzr[indv(gidz+4,gidx)] += HC5*hz*lsxx[FDOH] * rkp[indp(gidz+4, gidx)];
        vzr[indv(gidz-5,gidx)] += -HC5*hz*lsxx[FDOH] * rkp[indp(gidz-5, gidx)];

    }
#elif FDOH==6
    {
        vxr[indv(gidz,gidx)] += HC1*hx*lsxx[FDOH] * rip[indp(gidz, gidx)];
        vxr[indv(gidz,gidx-1)] += -HC1*hx*lsxx[FDOH] * rip[indp(gidz, gidx-1)];
        vxr[indv(gidz,gidx+1)] += HC2*hx*lsxx[FDOH] * rip[indp(gidz, gidx+1)];
        vxr[indv(gidz,gidx-2)] += -HC2*hx*lsxx[FDOH] * rip[indp(gidz, gidx-2)];
        vxr[indv(gidz,gidx+2)] += HC3*hx*lsxx[FDOH] * rip[indp(gidz, gidx+2)];
        vxr[indv(gidz,gidx-3)] += -HC3*hx*lsxx[FDOH] * rip[indp(gidz, gidx-3)];
        vxr[indv(gidz,gidx+3)] += HC4*hx*lsxx[FDOH] * rip[indp(gidz, gidx+3)];
        vxr[indv(gidz,gidx-4)] += -HC4*hx*lsxx[FDOH] * rip[indp(gidz, gidx-4)];
        vxr[indv(gidz,gidx+4)] += HC5*hx*lsxx[FDOH] * rip[indp(gidz, gidx+4)];
        vxr[indv(gidz,gidx-5)] += -HC5*hx*lsxx[FDOH] * rip[indp(gidz, gidx-5)];
        vxr[indv(gidz,gidx+5)] += HC6*hx*lsxx[FDOH] * rip[indp(gidz, gidx+5)];
        vxr[indv(gidz,gidx-6)] += -HC6*hx*lsxx[FDOH] * rip[indp(gidz, gidx-6)];

        vzr[indv(gidz,gidx)] += HC1*hz*lsxx[FDOH] * rkp[indp(gidz, gidx)];
        vzr[indv(gidz-1,gidx)] += -HC1*hz*lsxx[FDOH] * rkp[indp(gidz-1, gidx)];
        vzr[indv(gidz+1,gidx)] += HC2*hz*lsxx[FDOH] * rkp[indp(gidz+1, gidx)];
        vzr[indv(gidz-2,gidx)] += -HC2*hz*lsxx[FDOH] * rkp[indp(gidz-2, gidx)];
        vzr[indv(gidz+2,gidx)] += HC3*hz*lsxx[FDOH] * rkp[indp(gidz+2, gidx)];
        vzr[indv(gidz-3,gidx)] += -HC3*hz*lsxx[FDOH] * rkp[indp(gidz-3, gidx)];
        vzr[indv(gidz+3,gidx)] += HC4*hz*lsxx[FDOH] * rkp[indp(gidz+3, gidx)];
        vzr[indv(gidz-4,gidx)] += -HC4*hz*lsxx[FDOH] * rkp[indp(gidz-4, gidx)];
        vzr[indv(gidz+4,gidx)] += HC5*hz*lsxx[FDOH] * rkp[indp(gidz+4, gidx)];
        vzr[indv(gidz-5,gidx)] += -HC5*hz*lsxx[FDOH] * rkp[indp(gidz-5, gidx)];
        vzr[indv(gidz+5,gidx)] += HC6*hz*lsxx[FDOH] * rkp[indp(gidz+5, gidx)];
        vzr[indv(gidz-6,gidx)] += -HC6*hz*lsxx[FDOH] * rkp[indp(gidz-6, gidx)];

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

    szzr[indv(gidz,  gidx)] = lszz[FDOH];
    sxxr[indv(gidz,  gidx)] = lsxx[FDOH];
    for (m=1; m<=FDOH; m++) {
        szzr[indv(gidz-m,  gidx)] = lszz[FDOH-m];
        szzr[indv(gidz+m,  gidx)] = lszz[FDOH+m];
        sxxr[indv(gidz-m,  gidx)] = lsxx[FDOH-m];
        sxxr[indv(gidz+m,  gidx)] = lsxx[FDOH+m];
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



