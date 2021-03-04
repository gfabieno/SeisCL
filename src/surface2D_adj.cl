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
    __cprec1 hc[6] = {HC1, HC2, HC3, HC4, HC5, HC6};

    /* Global work size is padded to be a multiple of local work size.
     The padding elements must not be updated */
    if ( gidx>(NX-FDOH-1) ){
        return;
    }

    __cprec1 hx, hz;
    __cprec1 lszz, lsxx;
    __cprec1 lmu, lM;


    //Preadjoint transformations T
    lszz = -szzr[indv(gidz,  gidx)];
    lsxx = -sxxr[indv(gidz,  gidx)];
    lmu = mu[indp(gidz,  gidx)];
    lM = M[indp(gidz,  gidx)];
    
    //Preadjoint transformations L
    {
        __cprec1 temp1, temp2, a;
        temp1 = lsxx;
        temp2 = lszz;
        a = (__cprec1)1.0/( (__cprec1)4.0*lM*lmu - (__cprec1)4.0*lmu*lmu);
        lsxx = a * ( lM * temp1 + (-lM + (__cprec1)2.0 * lmu) * temp2 );
        lszz = a * ( lM * temp2 + (-lM + (__cprec1)2.0 * lmu) * temp1 );
    }

    /*Adjoint of the mirror method*/
    lszz=0.0;

    hx=-((lM-lmu*(__cprec1)2.0)*(lM-lmu*(__cprec1)2.0)/lM);
    hz=-(lM-lmu*(__cprec1)2.0);

    for (int m=0; m<FDOH; m++){
        if ( (gidx-FDOH + m) < (NX - 2*FDOH))
            vxr[indv(gidz,gidx + m)]+=(hc[m] * hx * lsxx *
                                       (__cprec1)rip[indp(gidz,gidx+m)]);
        if ( (gidx - (m + 1)) > (FDOH - 1))
            vxr[indv(gidz,gidx-(m+1))]+=(-hc[m] * hx* lsxx *
                                         (__cprec1)rip[indp(gidz,gidx-(m+1))]);
        vzr[indv(gidz+m,gidx)] += (hc[m] *hz * lsxx
                                   * (__cprec1)rkp[indp(gidz+m,gidx)]);
    }

    //Perform the post-adjoint transformation T L^-1
    szzr[indv(gidz,  gidx)] = -( lM * lszz + ( lM - (__cprec1)2.0 * lmu) * lsxx );
    sxxr[indv(gidz,  gidx)] = -( lM * lsxx + ( lM - (__cprec1)2.0 * lmu) * lszz );

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



}



