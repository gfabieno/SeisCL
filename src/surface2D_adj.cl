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

#define indv(z,x)  (x)*(NZ*DIV)+(z)
#define indp(z,x)  ((x)-FDOH)*(NZ*DIV-2*FDOH)+((z)-FDOH)

FUNDEF void surface_adj(GLOBARG __prec *vxr,        GLOBARG __prec *vzr,
                        GLOBARG __prec *sxxr,       GLOBARG __prec *szzr,
                        GLOBARG __prec *sxzr,       GLOBARG __prec *M,
                        GLOBARG __prec *rip,        GLOBARG __prec *rkp,
                        GLOBARG __prec *mu,         GLOBARG __prec *rxx,
                        GLOBARG __prec *rzz,        GLOBARG __prec *taus,
                        GLOBARG __prec *taup,       GLOBARG float *eta,
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

    __prec f, g, hx, hz;
    __prec  vxx, vzz;
    int m;
    __prec lszz, lsxx;
    __prec lmu, lM;
    __prec temp1, temp2, a;

    //Preadjoint transformations
    lszz = szzr[indv(gidz,  gidx)];
    lsxx = sxxr[indv(gidz,  gidx)];
    lmu = mu[indp(gidz,  gidx)];
    lM = M[indp(gidz,  gidx)];

    temp1 = lsxx;
    temp2 = lszz;
    a = (__prec)1.0/( lM*lM - (lM-(__prec)2.0*lmu)*(lM-(__prec)2.0*lmu));
    lsxx = a * ( lM * temp1 + (-lM + (__prec)2.0 * lmu) * temp2 );
    lszz = a * ( lM * temp2 + (-lM + (__prec)2.0 * lmu) * temp1 );


    /*Mirroring the components of the stress tensor to make
     a stress free surface (method of imaging, Levander, 1988)*/
    lszz=0.0;
//#if LVE>0
//    rzz(gidz, gidx)=0.0;
//#endif




    f=lmu*(__prec)2.0;
    g=lM;
    hx=-((g-f)*(g-f)/g);
    hz=-(g-f);

//// Absorbing boundary
//#if ABS_TYPE==2
//    {
//
//#if DEVID==0 & MYLOCALID==0
//        if (gidx-FDOH<NAB){
//            hx*=taper[gidx-FDOH];
//            hz*=taper[gidx-FDOH];
//        }
//#endif
//
//#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//        if (gidx>NX-NAB-FDOH-1){
//            hx*=taper[NX-FDOH-gidx-1];
//            hz*=taper[NX-FDOH-gidx-1];
//        }
//#endif
//    }
//#endif

#if   FDOH==1
    {
        vxr[indv(gidz,gidx)] += hx*lsxx * rip[indp(gidz, gidx)];
        if ( (gidx - FDOH - 1) >= 0)
            vxr[indv(gidz,gidx-1)] += -hx*lsxx * rip[indp(gidz, gidx-1)];
        vzr[indv(gidz,gidx)] += hz*lsxx * rkp[indp(gidz, gidx)];


    }
#elif FDOH==2
    {
        vxr[indv(gidz,gidx)] += (__prec)HC1*hx*lsxx * rip[indp(gidz, gidx)];
        if ( (gidx - FDOH - 1) >= 0)
            vxr[indv(gidz,gidx-1)] += -(__prec)HC1*hx*lsxx * rip[indp(gidz, gidx-1)];
        if ( (gidx-FDOH+1) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+1)] += (__prec)HC2*hx*lsxx * rip[indp(gidz, gidx+1)];
        if ( (gidx - FDOH - 2) >= 0)
            vxr[indv(gidz,gidx-2)] += -(__prec)HC2*hx*lsxx * rip[indp(gidz, gidx-2)];

        vzr[indv(gidz,gidx)] += (__prec)HC1*hz*lsxx * rkp[indp(gidz, gidx)];
        vzr[indv(gidz+1,gidx)] += (__prec)HC2*hz*lsxx * rkp[indp(gidz+1, gidx)];

    }
#elif FDOH==3
    {
        vxr[indv(gidz,gidx)] += (__prec)HC1*hx*lsxx * rip[indp(gidz, gidx)];
        if ( (gidx - FDOH - 1) >= 0)
            vxr[indv(gidz,gidx-1)] += -(__prec)HC1*hx*lsxx * rip[indp(gidz, gidx-1)];
        if ( (gidx-FDOH+1) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+1)] += (__prec)HC2*hx*lsxx * rip[indp(gidz, gidx+1)];
        if ( (gidx - FDOH - 2) >= 0)
            vxr[indv(gidz,gidx-2)] += -(__prec)HC2*hx*lsxx * rip[indp(gidz, gidx-2)];
        if ( (gidx-FDOH+2) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+2)] += (__prec)HC3*hx*lsxx * rip[indp(gidz, gidx+2)];
        if ( (gidx - FDOH - 3) >= 0)
            vxr[indv(gidz,gidx-3)] += -(__prec)HC3*hx*lsxx * rip[indp(gidz, gidx-3)];

        vzr[indv(gidz,gidx)] += (__prec)HC1*hz*lsxx * rkp[indp(gidz, gidx)];
        vzr[indv(gidz+1,gidx)] += (__prec)HC2*hz*lsxx * rkp[indp(gidz+1, gidx)];
        vzr[indv(gidz+2,gidx)] += (__prec)HC3*hz*lsxx * rkp[indp(gidz+2, gidx)];

    }
#elif FDOH==4
    {

        vxr[indv(gidz,gidx)] += (__prec)HC1*hx*lsxx * rip[indp(gidz, gidx)];
        if ( (gidx - FDOH - 1) >= 0)
            vxr[indv(gidz,gidx-1)] += -(__prec)HC1*hx*lsxx * rip[indp(gidz, gidx-1)];
        if ( (gidx-FDOH+1) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+1)] += (__prec)HC2*hx*lsxx * rip[indp(gidz, gidx+1)];
        if ( (gidx - FDOH - 2) >=0)
            vxr[indv(gidz,gidx-2)] += -(__prec)HC2*hx*lsxx * rip[indp(gidz, gidx-2)];
        if ( (gidx-FDOH+2) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+2)] += (__prec)HC3*hx*lsxx * rip[indp(gidz, gidx+2)];
        if ( (gidx - FDOH - 3) >=0)
            vxr[indv(gidz,gidx-3)] += -(__prec)HC3*hx*lsxx * rip[indp(gidz, gidx-3)];
        if ( (gidx-FDOH+3) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+3)] += (__prec)HC4*hx*lsxx * rip[indp(gidz, gidx+3)];
        if ( (gidx - FDOH - 4) >=0)
            vxr[indv(gidz,gidx-4)] += -(__prec)HC4*hx*lsxx * rip[indp(gidz, gidx-4)];

        vzr[indv(gidz,gidx)] += (__prec)HC1*hz*lsxx * rkp[indp(gidz, gidx)];
        vzr[indv(gidz+1,gidx)] += (__prec)HC2*hz*lsxx * rkp[indp(gidz+1, gidx)];
        vzr[indv(gidz+2,gidx)] += (__prec)HC3*hz*lsxx * rkp[indp(gidz+2, gidx)];
        vzr[indv(gidz+3,gidx)] += (__prec)HC4*hz*lsxx * rkp[indp(gidz+3, gidx)];

    }
#elif FDOH==5
    {
        vxr[indv(gidz,gidx)] += (__prec)HC1*hx*lsxx * rip[indp(gidz, gidx)];
        if ( (gidx - FDOH - 1) >= 0)
            vxr[indv(gidz,gidx-1)] += -(__prec)HC1*hx*lsxx * rip[indp(gidz, gidx-1)];
        if ( (gidx-FDOH+1) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+1)] += (__prec)HC2*hx*lsxx * rip[indp(gidz, gidx+1)];
        if ( (gidx - FDOH - 2) >= 0)
            vxr[indv(gidz,gidx-2)] += -(__prec)HC2*hx*lsxx * rip[indp(gidz, gidx-2)];
        if ( (gidx-FDOH+2) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+2)] += (__prec)HC3*hx*lsxx * rip[indp(gidz, gidx+2)];
        if ( (gidx - FDOH - 3) >= 0)
            vxr[indv(gidz,gidx-3)] += -(__prec)HC3*hx*lsxx * rip[indp(gidz, gidx-3)];
        if ( (gidx-FDOH+3) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+3)] += (__prec)HC4*hx*lsxx * rip[indp(gidz, gidx+3)];
        if ( (gidx - FDOH - 4) >= 0)
            vxr[indv(gidz,gidx-4)] += -(__prec)HC4*hx*lsxx * rip[indp(gidz, gidx-4)];
        if ( (gidx-FDOH+4) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+4)] += (__prec)HC5*hx*lsxx * rip[indp(gidz, gidx+4)];
        if ( (gidx - FDOH - 5) >= 0)
            vxr[indv(gidz,gidx-5)] += -(__prec)HC5*hx*lsxx * rip[indp(gidz, gidx-5)];

        vzr[indv(gidz,gidx)] += (__prec)HC1*hz*lsxx * rkp[indp(gidz, gidx)];
        vzr[indv(gidz+1,gidx)] += (__prec)HC2*hz*lsxx * rkp[indp(gidz+1, gidx)];
        vzr[indv(gidz+2,gidx)] += (__prec)HC3*hz*lsxx * rkp[indp(gidz+2, gidx)];
        vzr[indv(gidz+3,gidx)] += (__prec)HC4*hz*lsxx * rkp[indp(gidz+3, gidx)];
        vzr[indv(gidz+4,gidx)] += (__prec)HC5*hz*lsxx * rkp[indp(gidz+4, gidx)];

    }
#elif FDOH==6
    {
        vxr[indv(gidz,gidx)] += (__prec)HC1*hx*lsxx * rip[indp(gidz, gidx)];
        if ( (gidx - FDOH - 1) >= 0)
            vxr[indv(gidz,gidx-1)] += -(__prec)HC1*hx*lsxx * rip[indp(gidz, gidx-1)];
        if ( (gidx-FDOH+1) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+1)] += (__prec)HC2*hx*lsxx * rip[indp(gidz, gidx+1)];
        if ( (gidx - FDOH - 2) >= 0)
            vxr[indv(gidz,gidx-2)] += -(__prec)HC2*hx*lsxx * rip[indp(gidz, gidx-2)];
        if ( (gidx-FDOH+2) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+2)] += (__prec)HC3*hx*lsxx * rip[indp(gidz, gidx+2)];
        if ( (gidx - FDOH - 3) >= 0)
            vxr[indv(gidz,gidx-3)] += -(__prec)HC3*hx*lsxx * rip[indp(gidz, gidx-3)];
        if ( (gidx-FDOH+3) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+3)] += (__prec)HC4*hx*lsxx * rip[indp(gidz, gidx+3)];
        if ( (gidx - FDOH - 4) >= 0)
            vxr[indv(gidz,gidx-4)] += -(__prec)HC4*hx*lsxx * rip[indp(gidz, gidx-4)];
        if ( (gidx-FDOH+4) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+4)] += (__prec)HC5*hx*lsxx * rip[indp(gidz, gidx+4)];
        if ( (gidx - FDOH - 5) >= 0)
            vxr[indv(gidz,gidx-5)] += -(__prec)HC5*hx*lsxx * rip[indp(gidz, gidx-5)];
        if ( (gidx-FDOH+5) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx+5)] += (__prec)HC6*hx*lsxx * rip[indp(gidz, gidx+5)];
        if ( (gidx - FDOH - 6) >= 0)
            vxr[indv(gidz,gidx-6)] += -(__prec)HC6*hx*lsxx * rip[indp(gidz, gidx-6)];

        vzr[indv(gidz,gidx)] += (__prec)HC1*hz*lsxx * rkp[indp(gidz, gidx)];
        vzr[indv(gidz+1,gidx)] += (__prec)HC2*hz*lsxx * rkp[indp(gidz+1, gidx)];
        vzr[indv(gidz+2,gidx)] += (__prec)HC3*hz*lsxx * rkp[indp(gidz+2, gidx)];
        vzr[indv(gidz+3,gidx)] += (__prec)HC4*hz*lsxx * rkp[indp(gidz+3, gidx)];
        vzr[indv(gidz+4,gidx)] += (__prec)HC5*hz*lsxx * rkp[indp(gidz+4, gidx)];
        vzr[indv(gidz+5,gidx)] += (__prec)HC6*hz*lsxx * rkp[indp(gidz+5, gidx)];

    }
#endif


    //Perform the post-adjoint transformation
    temp1 = lsxx;
    temp2 = lszz;
    lsxx =  ( lM * temp1 + ( lM - (__prec)2.0 * lmu) * temp2 );
    lszz =  ( lM * temp2 + ( lM - (__prec)2.0 * lmu) * temp1 );

    szzr[indv(gidz,  gidx)] = lszz;
    sxxr[indv(gidz,  gidx)] = lsxx;



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



