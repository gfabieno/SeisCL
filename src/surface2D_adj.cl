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



FUNDEF void surface_adj(GLOBARG STO *vxr,        GLOBARG STO *vzr,
                        GLOBARG STO *sxxr,       GLOBARG STO *szzr,
                        GLOBARG STO *sxzr,       GLOBARG STO *M,
                        GLOBARG STO *rip,        GLOBARG STO *rkp,
                        GLOBARG STO *mu,         GLOBARG STO *rxx,
                        GLOBARG STO *rzz,        GLOBARG STO *taus,
                        GLOBARG STO *taup,       GLOBARG float *eta,
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


    lszz = ldf(szzr, indv(gidz,  gidx));
    lsxx = ldf(sxxr, indv(gidz,  gidx));
    lmu = ldf(mu, indp(gidz,  gidx));
    lM = ldf(M, indp(gidz,  gidx));
    
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


    //Perform the post-adjoint transformation L^-1
    stf(szzr, indv(gidz,  gidx), -( lM * lszz + ( lM - (__cprec1)2.0 * lmu) * lsxx ));
    stf(sxxr, indv(gidz,  gidx), -( lM * lsxx + ( lM - (__cprec1)2.0 * lmu) * lszz ));

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
FUNDEF void surface_adj_fold(GLOBARG STO *vxr,  GLOBARG STO *vzr,
                             GLOBARG STO *sxxr, GLOBARG STO *szzr,
                             GLOBARG STO *sxzr, GLOBARG STO *M,
                             GLOBARG STO *mu,   GLOBARG STO *muipkp,
                             GLOBARG STO *rip,  GLOBARG STO *rkp)
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
            img_szz -= hc[m] * ldf(vzr, indv(FDOH-k+m, gidx));
        }
        for (int m=k-1; m<FDOH; m++){
            img_sxz -= hc[m] * ldf(vxr, indv(FDOH-k+m+1, gidx));
        }
        int z = FDOH + k;
        __cprec1 lM = ldf(M, indp(z, gidx));
        __cprec1 lmu = ldf(mu, indp(z, gidx));
        stf(szzr, indv(z, gidx), ldf(szzr, indv(z, gidx)) + (lM * img_szz));
        stf(sxxr, indv(z, gidx), ldf(sxxr, indv(z, gidx)) + ((lM - (__cprec1)2.0 * lmu) * img_szz));
        stf(sxzr, indv(z-1, gidx), ldf(sxzr, indv(z-1, gidx)) + (ldf(muipkp, indp(z-1, gidx)) * img_sxz));
    }
    for (int m=1; m<=FDOH; m++){
        stf(szzr, indv(FDOH-m, gidx), 0);
        stf(sxzr, indv(FDOH-m, gidx), 0);
    }

    // 2. Adjoint of the sxx correction. hx*lsxx of column c, lsxx being the
    //    surface sxx of column c after the transform L (as in surface_adj).
    __cprec1 dvx = 0;
    for (int m=0; m<FDOH; m++){
        // column c = gidx - m wrote +hc[m]*hx*lsxx*rip into gidx (forward Dxm)
        int c = gidx - m;
        if (c >= FDOH){
            __cprec1 cM = ldf(M, indp(gidz, c)), cmu = ldf(mu, indp(gidz, c));
            __cprec1 a = (__cprec1)1.0/( (__cprec1)4.0*cM*cmu - (__cprec1)4.0*cmu*cmu);
            __cprec1 lsxx = -a * ( cM * ldf(sxxr, indv(gidz, c))
                                  + (-cM + (__cprec1)2.0 * cmu) * ldf(szzr, indv(gidz, c)) );
            dvx += hc[m] * (-((cM-cmu*(__cprec1)2.0)*(cM-cmu*(__cprec1)2.0)/cM)) * lsxx;
        }
        // column c = gidx + m + 1 wrote -hc[m]*hx*lsxx*rip into gidx
        c = gidx + m + 1;
        if (c <= NX-FDOH-1){
            __cprec1 cM = ldf(M, indp(gidz, c)), cmu = ldf(mu, indp(gidz, c));
            __cprec1 a = (__cprec1)1.0/( (__cprec1)4.0*cM*cmu - (__cprec1)4.0*cmu*cmu);
            __cprec1 lsxx = -a * ( cM * ldf(sxxr, indv(gidz, c))
                                  + (-cM + (__cprec1)2.0 * cmu) * ldf(szzr, indv(gidz, c)) );
            dvx -= hc[m] * (-((cM-cmu*(__cprec1)2.0)*(cM-cmu*(__cprec1)2.0)/cM)) * lsxx;
        }
    }
    stf(vxr, indv(gidz, gidx), ldf(vxr, indv(gidz, gidx)) + (dvx * ldf(rip, indp(gidz, gidx))));
    {
        __cprec1 lM = ldf(M, indp(gidz, gidx)), lmu = ldf(mu, indp(gidz, gidx));
        __cprec1 a = (__cprec1)1.0/( (__cprec1)4.0*lM*lmu - (__cprec1)4.0*lmu*lmu);
        __cprec1 lsxx = -a * ( lM * ldf(sxxr, indv(gidz, gidx))
                              + (-lM + (__cprec1)2.0 * lmu) * ldf(szzr, indv(gidz, gidx)) );
        __cprec1 hz = -(lM-lmu*(__cprec1)2.0);
        for (int m=0; m<FDOH; m++){
            stf(vzr, indv(gidz+m, gidx), ldf(vzr, indv(gidz+m, gidx)) + (hc[m] * hz * lsxx * ldf(rkp, indp(gidz+m, gidx))));
        }
    }
}



#ifndef SURFROWS
#define SURFROWS (FDOH+1)
#endif
/* Gradient in the top rows under a stress-image free surface (FREESURF==1,
   2D elastic, BACK_PROP_TYPE 1, FP16==0), in the direct form
       dJ/dtheta = sum_t lambda(t) . dP_t/dtheta x(t)
   instead of the summation-by-parts form of update_adjs2D.cl and
   update_adjv2D.cl, which needs every change of the adjoint field at a cell
   to be the update increment: false in the rows that the adjoint of the
   surface (surface_adj_fold, surface_adj) changes.

   Launched at the start of each adjoint time step t, after the residuals,
   before update_adjs. There the adjoint fields are exactly T*eta_t, with
   eta_t the adjoint of the state right after the forward velocity update of
   step t, T = diag(B, -C): checked against the exact transpose of the time
   step (~/data/claude/torch-robust/surfgrad/pairing2.py). Its stress part is
   T*nu_t, nu_t the adjoint of the stresses after the forward stress update
   and before the surface kernel. The forward velocities are still v(t+1).

   Moduli, rows 0..SURFROWS-1:  dJ/dtheta += nu . (dC/dtheta) eps(v(t+1)).
     On the surface row the forward update (stress update + surface kernel)
     is szz = 0, sxx += (M - (M-2mu)^2/M) vxx, so d/dM = 4 mu^2/M^2 vxx and
     d/dmu = 4 (M-2mu)/M vxx. The velocity rows above the surface are never
     written, zero in both passes, as the forward stencils saw them.
   With pressure receivers, the residual injected at this step is taken
   out first (it belongs to the next step).
   Density, rows 0..FDOH-1 (stored in the boundary buffer): the accumulator of
     update_adjv2D.cl is -B^2 dJ/dB = -sum_t a_v(t) * (v(t+1) - taper*v(t)),
     with v(t) the stored state plus the force source of the step (no stencil
     on the unstored images).
   (notes: ~/data/claude/notes/surface-gradient.md) */
#ifdef HAVE_INJECT_IND   /* compiled only with header_injectbnd.cl (the surf_grad program) */
FUNDEF void surface_grad(GLOBARG STO *vx,         GLOBARG STO *vz,
                         GLOBARG STO *vxr,        GLOBARG STO *vzr,
                         GLOBARG STO *sxxr,       GLOBARG STO *szzr,
                         GLOBARG STO *sxzr,       GLOBARG STO *M,
                         GLOBARG STO *mu,         GLOBARG STO *muipkp,
                         GLOBARG STO *vxbnd,      GLOBARG STO *vzbnd,
                         GLOBARG float *taper,
                         GLOBARG float *gradM,       GLOBARG float *gradmu,
                         GLOBARG float *gradmuipkp,  GLOBARG float *gradrip,
                         GLOBARG float *gradrkp,     GLOBARG float *src,
                         GLOBARG float *src_pos,     int nsrc,
                         int nt,                     GLOBARG float *pout,
                         GLOBARG float *rec_pos,     int nrec,
                         int src_scale,              int res_scale)
{
    #ifdef __OPENCL_VERSION__
    int gidx = get_global_id(0) + FDOH;
    #else
    int gidx = blockIdx.x*blockDim.x + threadIdx.x + FDOH;
    #endif
    if ( gidx>(NX-FDOH-1) ){
        return;
    }
    __cprec1 hc[6] = {HC1, HC2, HC3, HC4, HC5, HC6};
    for (int k=0; k<SURFROWS && FDOH+k<NZ-FDOH; k++){
        int z = FDOH+k;
        __cprec1 vxx = 0, vzz = 0, vxz = 0, vzx = 0;
        for (int m=0; m<FDOH; m++){
            vxx += hc[m]*(ldf(vx, indv(z, gidx+m))   - ldf(vx, indv(z, gidx-m-1)));
            vzz += hc[m]*(ldf(vz, indv(z+m, gidx))   - ldf(vz, indv(z-m-1, gidx)));
            vxz += hc[m]*(ldf(vx, indv(z+m+1, gidx)) - ldf(vx, indv(z-m, gidx)));
            vzx += hc[m]*(ldf(vz, indv(z, gidx+m+1)) - ldf(vz, indv(z, gidx-m)));
        }
        __cprec1 lM = ldf(M, indp(z, gidx)), lmu = ldf(mu, indp(z, gidx));
        __cprec1 lmuipkp = ldf(muipkp, indp(z, gidx));
        __cprec1 axx = ldf(sxxr, indv(z, gidx)), azz = ldf(szzr, indv(z, gidx));
        #if PRESOUT==1
        /* The pressure residual injected at this step (residuals.c delays it
           by one sample) is not yet part of T*nu_t: it enters the next step,
           propagated. kernel_residuals adds pout/2 to sxxr and szzr. */
        for (int g=0; g<nrec; g++){
            int ri=(int)(rec_pos[0+8*g]/DH)+FDOH;
            int rk=(int)(rec_pos[2+8*g]/DH)+FDOH;
            if (ri==gidx && rk==z){
                axx -= 0.5f*pow2s(pout[NT*g+nt], res_scale);
                azz -= 0.5f*pow2s(pout[NT*g+nt], res_scale);
            }
        }
        #endif
        __cprec1 det = (__cprec1)4.0*lmu*(lM-lmu);
        __cprec1 nxx = 0, nzz = 0, nxz = 0;
        if (det != 0){
            nxx = -(lM*axx - (lM-(__cprec1)2.0*lmu)*azz)/det;
            nzz = -(lM*azz - (lM-(__cprec1)2.0*lmu)*axx)/det;
        }
        if (lmuipkp != 0){
            nxz = -ldf(sxzr, indv(z, gidx))/lmuipkp;
        }
        __cprec1 gM, gmu;
        if (k==0){
            gM  = nxx*(__cprec1)4.0*lmu*lmu/(lM*lM)*vxx;
            gmu = nxx*(__cprec1)4.0*(lM-(__cprec1)2.0*lmu)/lM*vxx;
        }
        else{
            gM  = (nxx+nzz)*(vxx+vzz);
            gmu = -(__cprec1)2.0*(nxx*vzz + nzz*vxx);
        }
        gradM[indp(z, gidx)] += GSCALE(gM);
        gradmu[indp(z, gidx)] += GSCALE(gmu);
        gradmuipkp[indp(z, gidx)] += GSCALE(nxz*(vxz+vzx));

        if (k<FDOH){
            int mb = inject_ind(z, gidx);
            if (mb!=-1){
                __cprec1 gt = 1;
                #if ABS_TYPE==2
                if (z>NZ-NAB-FDOH-1) gt*=taper[NZ-FDOH-z-1];
                #if DEVID==0 & MYLOCALID==0
                if (gidx-FDOH<NAB) gt*=taper[gidx-FDOH];
                #endif
                #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
                if (gidx>NX-NAB-FDOH-1) gt*=taper[NX-FDOH-gidx-1];
                #endif
                #endif
                /* A force source of this step was added to v(t) before the
                   velocity update (kernel_sources, amp = DT*src): it does
                   not depend on the density, so take it out. */
                __cprec1 sx = 0, sz = 0;
                for (int srci=0; srci<nsrc; srci++){
                    int st = (int)src_pos[4+5*srci];
                    int si = (int)(src_pos[0+5*srci]/DH)+FDOH;
                    int sk = (int)(src_pos[2+5*srci]/DH)+FDOH;
                    if (si==gidx && sk==z){
                        /* the velocities are stored scaled by 2^(src_scale - PARSCALE) */
                        if (st==0) sx += pow2s(DT*src[srci*NT+nt], src_scale - PARSCALE);
                        if (st==2) sz += pow2s(DT*src[srci*NT+nt], src_scale - PARSCALE);
                    }
                }
                gradrip[indp(z, gidx)] += GSCALE(-ldf(vxr, indv(z, gidx))
                    * (ldf(vx, indv(z, gidx)) - gt*(ldf(vxbnd, mb) + sx)));
                gradrkp[indp(z, gidx)] += GSCALE(-ldf(vzr, indv(z, gidx))
                    * (ldf(vz, indv(z, gidx)) - gt*(ldf(vzbnd, mb) + sz)));
            }
        }
    }
}
#endif
