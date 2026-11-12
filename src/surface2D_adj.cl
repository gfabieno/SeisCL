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


    lszz = szzr[indv(gidz,  gidx)];
    lsxx = sxxr[indv(gidz,  gidx)];
    lmu = mu[indp(gidz,  gidx)];
    lM = M[indp(gidz,  gidx)];
    
    //Preadjoint transformations L
    {
        __cprec1 temp1, temp2, a;
        temp1 = lsxx;
        temp2 = lszz;
        a = (__cprec1)1.0/( (__cprec1)4.0*lM*lmu - (__cprec1)4.0*lmu*lmu);
        lsxx = -a * ( lM * temp1 + (-lM + (__cprec1)2.0 * lmu) * temp2 );
        lszz = -a * ( lM * temp2 + (-lM + (__cprec1)2.0 * lmu) * temp1 );
    }

    /*Adjoint of the mirror method*/
    lszz=0.0;

    hx=-((lM-lmu*(__cprec1)2.0)*(lM-lmu*(__cprec1)2.0)/lM);
    hz=-(lM-lmu*(__cprec1)2.0);

//    // Absorbing boundary
//    #if ABS_TYPE==2
//    {
//            #if DEVID==0 & MYLOCALID==0
//                if (gidx-FDOH<NAB){
//                    hx*=taper[gidx-FDOH];
//                    hz*=taper[gidx-FDOH];
//                }
//            #endif
//
//            #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
//                if (gidx>NX-NAB-FDOH-1){
//                    hx*=taper[NX-FDOH-gidx-1];
//                    hz*=taper[NX-FDOH-gidx-1];
//                }
//            #endif
//            }
//    #endif

#if FP16!=0
    /* FP16 path (no surface_adj_fold): scatter into vxr of neighbouring
       columns. Not race-free: other threads add to the same cells. */
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
#endif

    //Perform the post-adjoint transformation L^-1
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





/* Adjoint of the free surface, first part (FREESURF==1, 2D elastic, FP16==0).
   Launched just before surface_adj, between the adjoint stress update and the
   adjoint velocity update (the place of the surface kernel in the forward
   step). One thread per column; each thread writes only its own column, so
   the kernel is race-free. The adjoint fields are stored as T*lambda with
   T = diag(B, -C) (symmetrised adjoint).

   1. Adjoint of the stress images. In the forward step, update_v reads the
      images that the surface kernel writes above the surface (szz through
      Dzp, sxz through Dzm). Their adjoint is a contribution of the adjoint
      velocities to the image rows, which the transpose of the mirroring folds
      back, with a minus sign, into the rows below the surface:
        img_szz(k) = -sum_{m>=k}   hc[m] * vzr(FDOH-k+m)
        img_sxz(k) = -sum_{m>=k-1} hc[m] * vxr(FDOH-k+m+1)
        szzr(FDOH+k) += M img_szz, sxxr(FDOH+k) += (M-2mu) img_szz,
        sxzr(FDOH+k-1) += muipkp img_sxz;
      the image rows of the adjoint stresses stay zero. Uses the adjoint
      velocities before step 2 changes them.
   2. Adjoint of the sxx correction, h = hx*Dxm(vx) + hz*Dzm(vz): as a gather
      of the contributions of the surface sxx of the neighbouring columns
      (read before surface_adj transforms them) into this column's vxr, and of
      this column's into its vzr. surface_adj used to scatter into vxr of
      neighbouring columns with +=, a write race between threads.
   Derived and checked against the exact transpose of one time step:
   ~/data/claude/torch-robust/fsadj/ (ref.py, derive.py). */
FUNDEF void surface_adj_fold(GLOBARG __prec *vxr,  GLOBARG __prec *vzr,
                             GLOBARG __prec *sxxr, GLOBARG __prec *szzr,
                             GLOBARG __prec *sxzr, GLOBARG __prec *M,
                             GLOBARG __prec *mu,   GLOBARG __prec *muipkp,
                             GLOBARG __prec *rip,  GLOBARG __prec *rkp)
{
    #ifdef __OPENCL_VERSION__
    int gidx = get_global_id(0) + FDOH;
    #else
    int gidx = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    #endif
    if ( gidx>(NX-FDOH-1) ){
        return;
    }
    int gidz = FDOH;
    __cprec1 hc[6] = {HC1, HC2, HC3, HC4, HC5, HC6};

    // 1. Adjoint of the stress images (reads vxr, vzr of this column first)
    for (int k=1; k<=FDOH; k++){
        __cprec1 img_szz = 0;
        __cprec1 img_sxz = 0;
        for (int m=k; m<FDOH; m++){
            img_szz -= hc[m] * (__cprec1)vzr[indv(FDOH-k+m, gidx)];
        }
        for (int m=k-1; m<FDOH; m++){
            img_sxz -= hc[m] * (__cprec1)vxr[indv(FDOH-k+m+1, gidx)];
        }
        int z = FDOH + k;
        __cprec1 lM = M[indp(z, gidx)];
        __cprec1 lmu = mu[indp(z, gidx)];
        szzr[indv(z, gidx)] += lM * img_szz;
        sxxr[indv(z, gidx)] += (lM - (__cprec1)2.0 * lmu) * img_szz;
        sxzr[indv(z-1, gidx)] += (__cprec1)muipkp[indp(z-1, gidx)] * img_sxz;
    }
    for (int m=1; m<=FDOH; m++){
        szzr[indv(FDOH-m, gidx)] = 0;
        sxzr[indv(FDOH-m, gidx)] = 0;
    }

    // 2. Adjoint of the sxx correction. hx*lsxx of column c, lsxx being the
    //    surface sxx of column c after the transform L (as in surface_adj).
    __cprec1 dvx = 0;
    for (int m=0; m<FDOH; m++){
        // column c = gidx - m wrote +hc[m]*hx*lsxx*rip into gidx (forward Dxm)
        int c = gidx - m;
        if (c >= FDOH){
            __cprec1 cM = M[indp(gidz, c)], cmu = mu[indp(gidz, c)];
            __cprec1 a = (__cprec1)1.0/( (__cprec1)4.0*cM*cmu - (__cprec1)4.0*cmu*cmu);
            __cprec1 lsxx = -a * ( cM * (__cprec1)sxxr[indv(gidz, c)]
                                  + (-cM + (__cprec1)2.0 * cmu) * (__cprec1)szzr[indv(gidz, c)] );
            dvx += hc[m] * (-((cM-cmu*(__cprec1)2.0)*(cM-cmu*(__cprec1)2.0)/cM)) * lsxx;
        }
        // column c = gidx + m + 1 wrote -hc[m]*hx*lsxx*rip into gidx
        c = gidx + m + 1;
        if (c <= NX-FDOH-1){
            __cprec1 cM = M[indp(gidz, c)], cmu = mu[indp(gidz, c)];
            __cprec1 a = (__cprec1)1.0/( (__cprec1)4.0*cM*cmu - (__cprec1)4.0*cmu*cmu);
            __cprec1 lsxx = -a * ( cM * (__cprec1)sxxr[indv(gidz, c)]
                                  + (-cM + (__cprec1)2.0 * cmu) * (__cprec1)szzr[indv(gidz, c)] );
            dvx -= hc[m] * (-((cM-cmu*(__cprec1)2.0)*(cM-cmu*(__cprec1)2.0)/cM)) * lsxx;
        }
    }
    vxr[indv(gidz, gidx)] += dvx * (__cprec1)rip[indp(gidz, gidx)];
    {
        __cprec1 lM = M[indp(gidz, gidx)], lmu = mu[indp(gidz, gidx)];
        __cprec1 a = (__cprec1)1.0/( (__cprec1)4.0*lM*lmu - (__cprec1)4.0*lmu*lmu);
        __cprec1 lsxx = -a * ( lM * (__cprec1)sxxr[indv(gidz, gidx)]
                              + (-lM + (__cprec1)2.0 * lmu) * (__cprec1)szzr[indv(gidz, gidx)] );
        __cprec1 hz = -(lM-lmu*(__cprec1)2.0);
        for (int m=0; m<FDOH; m++){
            vzr[indv(gidz+m, gidx)] += hc[m] * hz * lsxx * (__cprec1)rkp[indp(gidz+m, gidx)];
        }
    }
}
