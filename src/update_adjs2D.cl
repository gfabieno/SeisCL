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

/*Adjoint update of the stresses in 2D SV*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */

#define psi_vx_x(z,x) psi_vx_x[(x)*(NZ-2*FDOH)+(z)]
#define psi_vz_x(z,x) psi_vz_x[(x)*(NZ-2*FDOH)+(z)]

#define psi_vx_z(z,x) psi_vx_z[(x)*(2*NAB)+(z)]
#define psi_vz_z(z,x) psi_vz_z[(x)*(2*NAB)+(z)]


FUNDEF void update_adjs(int offcomm,
                          GLOBARG const float * RESTRICT vx,            GLOBARG const float * RESTRICT vz,        GLOBARG float * RESTRICT sxx,
                          GLOBARG float * RESTRICT szz,                 GLOBARG float * RESTRICT sxz,             GLOBARG const float * RESTRICT vxbnd,
                          GLOBARG const float * RESTRICT vzbnd,         GLOBARG const float * RESTRICT sxxbnd,    GLOBARG const float * RESTRICT szzbnd,
                          GLOBARG const float * RESTRICT sxzbnd,        GLOBARG const float * RESTRICT vxr,       GLOBARG const float * RESTRICT vzr,
                          GLOBARG float * RESTRICT sxxr,                GLOBARG float * RESTRICT szzr,            GLOBARG float * RESTRICT sxzr,
                          GLOBARG float * RESTRICT rxx,                 GLOBARG float * RESTRICT rzz,             GLOBARG float * RESTRICT rxz,
                          GLOBARG float * RESTRICT rxxr,                GLOBARG float * RESTRICT rzzr,            GLOBARG float * RESTRICT rxzr,
                          GLOBARG const float * RESTRICT M,             GLOBARG const float * RESTRICT mu,        GLOBARG const float * RESTRICT muipkp,
                          GLOBARG const float * RESTRICT taus,          GLOBARG const float * RESTRICT tausipkp,  GLOBARG const float * RESTRICT taup,
                          GLOBARG const float * RESTRICT eta,           GLOBARG const float * RESTRICT taper,
                          GLOBARG const float * RESTRICT K_x,           GLOBARG const float * RESTRICT a_x,       GLOBARG const float * RESTRICT b_x,
                          GLOBARG const float * RESTRICT K_x_half,      GLOBARG const float * RESTRICT a_x_half,  GLOBARG const float * RESTRICT b_x_half,
                          GLOBARG const float * RESTRICT K_z,           GLOBARG const float * RESTRICT a_z,       GLOBARG const float * RESTRICT b_z,
                          GLOBARG const float * RESTRICT K_z_half,      GLOBARG const float * RESTRICT a_z_half,  GLOBARG const float * RESTRICT b_z_half,
                          GLOBARG float * RESTRICT psi_vx_x,            GLOBARG float * RESTRICT psi_vx_z,
                          GLOBARG float * RESTRICT psi_vz_x,            GLOBARG float * RESTRICT psi_vz_z,
                          GLOBARG const float * RESTRICT gradrho,       GLOBARG float * RESTRICT gradM,           GLOBARG float * RESTRICT gradmu,
                          GLOBARG float * RESTRICT gradmuipkp,          GLOBARG const float * RESTRICT gradtaup,  GLOBARG const float * RESTRICT gradtaus,
                          GLOBARG const float * RESTRICT gradtausipkp,  GLOBARG const float * RESTRICT gradsrc,
                          GLOBARG const float * RESTRICT Hrho,          GLOBARG float * RESTRICT HM,              GLOBARG float * RESTRICT Hmu,
                          GLOBARG const float * RESTRICT Htaup,         GLOBARG const float * RESTRICT Htaus,     GLOBARG const float * RESTRICT Hsrc,
                          GLOBARG const float * RESTRICT src,           GLOBARG const float * RESTRICT src_pos,
                          int nsrc,                                     int nt,
                          int src_scale,
                          GLOBARG const float * RESTRICT pout,          GLOBARG const float * RESTRICT rec_pos,
                          int nrec,                                     int res_scale,
                          LOCARG)
{

    LOCDEF
    
    int i,j,k,m;
    float vxx,vxz,vzx,vzz;
    float vxzzx,vxxzz;
    float vxxr,vxzr,vzxr,vzzr;
    float vxzzxr,vxxzzr;
    float lsxz, lsxx, lszz;
    float fipkp, f, g;
    float sumrxz, sumrxx, sumrzz;
    float b,c,e,d,dipkp;
    int l;
#if LVE>0
    float leta[LVE];
#endif
    float lM, lmu, lmuipkp, ltaup, ltaus, ltausipkp;

// If we use local memory
#if LOCAL_OFF==0
#ifdef __OPENCL_VERSION__
    int lsizez = get_local_size(0)+2*FDOH;
    int lsizex = get_local_size(1)+2*FDOH;
    int lidz = get_local_id(0)+FDOH;
    int lidx = get_local_id(1)+FDOH;
    int gidz = get_global_id(0)+FDOH;
    int gidx = get_global_id(1)+FDOH+offcomm;
#else
    int lsizez = blockDim.x+2*FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
#endif

#define lvx lvar
#define lvz lvar
#define lvxr lvar
#define lvzr lvar

// If local memory is turned off
#elif LOCAL_OFF==1
#ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
    int glsizez = (NZ-2*FDOH);
    int gidz = gid%glsizez+FDOH;
    int gidx = (gid/glsizez)+FDOH+offcomm;
#else
    int lsizez = blockDim.x+2*FDOH;
    int lsizex = blockDim.y+2*FDOH;
    int lidz = threadIdx.x+FDOH;
    int lidx = threadIdx.y+FDOH;
    int gidz = blockIdx.x*blockDim.x + threadIdx.x+FDOH;
    int gidx = blockIdx.y*blockDim.y + threadIdx.y+FDOH+offcomm;
#endif

#define lvxr vxr
#define lvzr vzr
#define lvx vx
#define lvz vz
#define lidx gidx
#define lidz gidz

#define lsizez NZ
#define lsizex NX

#endif
    int indr;
    int indp = (gidx-FDOH)*(NZ-2*FDOH)+(gidz-FDOH);
    int indv = gidx*NZ+gidz;
    
// Calculation of the velocity spatial derivatives of the forward wavefield if backpropagation is used
#if BACK_PROP_TYPE==1
    {
#if LOCAL_OFF==0
        load_local_in(vx);
        load_local_haloz(vx);
        load_local_halox(vx);
        BARRIER
#endif
        vxx = Dxm(lvx);
        vxz = Dzp(lvx);
        
        
#if LOCAL_OFF==0
        BARRIER
        load_local_in(vz);
        load_local_haloz(vz);
        load_local_halox(vz);
        BARRIER
#endif
        vzz = Dzm(lvz);
        vzx = Dxp(lvz);
        BARRIER
    }
#endif

// Calculation of the velocity spatial derivatives of the adjoint wavefield
    {
#if LOCAL_OFF==0
        load_local_in(vxr);
        load_local_haloz(vxr);
        load_local_halox(vxr);
        BARRIER
#endif
        vxxr = Dxm(lvxr);
        vxzr = Dzp(lvxr);
        
        
#if LOCAL_OFF==0
        BARRIER
        load_local_in(vzr);
        load_local_haloz(vzr);
        load_local_halox(vzr);
        BARRIER
#endif
        vzzr = Dzm(lvzr);
        vzxr = Dxp(lvzr);

    }

// To stop updating if we are outside the model (global id must be a multiple of local id in OpenCL, hence we stop if we have a global id outside the grid)
#if LOCAL_OFF==0
#if COMM12==0
    if (gidz>(NZ-FDOH-1) || (gidx-offcomm)>(NX-FDOH-1-LCOMM) ){
        return;
    }

#else
    if (gidz>(NZ-FDOH-1) ){
        return;
    }
#endif
#endif


// Read model parameters into local memory
#if LVE==0
    fipkp=muipkp[indp];
    lmu=mu[indp];
    lM=M[indp];
    f=2.0*lmu;
    g=lM;

#else

    lM=M[indp];
    lmu=mu[indp];
    lmuipkp=muipkp[indp];
    ltaup=taup[indp];
    ltaus=taus[indp];
    ltausipkp=tausipkp[indp];

    for (l=0;l<LVE;l++){
        leta[l]=eta[l];
    }

    fipkp=lmuipkp*(1.0+ (float)LVE*ltausipkp);
    g=lM*(1.0+(float)LVE*ltaup);
    f=2.0*lmu*(1.0+(float)LVE*ltaus);
    dipkp=lmuipkp*ltausipkp/DT;
    d=2.0*lmu*ltaus/DT;
    e=lM*ltaup/DT;

#endif


// Backpropagate the forward stresses
#if BACK_PROP_TYPE==1
    {
#if LVE==0

    sxz[indv]-=(fipkp*(vxz+vzx));
    sxx[indv]-=(g*(vxx+vzz))-(f*vzz) ;
    szz[indv]-=(g*(vxx+vzz))-(f*vxx) ;

// Backpropagation is not stable for viscoelastic wave equation
#else
    /* computing sums of the old memory variables */
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<LVE;l++){
        indr = l*NX*NZ + gidx*NZ+gidz;
        sumrxz+=rxz[indr];
        sumrxx+=rxx[indr];
        sumrzz+=rzz[indr];
    }

    /* updating components of the stress tensor, partially */
    lsxz=(fipkp*(vxz+vzx))+(DT2*sumrxz);
    lsxx=((g*(vxx+vzz))-(f*vzz))+(DT2*sumrxx);
    lszz=((g*(vxx+vzz))-(f*vxx))+(DT2*sumrzz);


    /* now updating the memory-variables and sum them up*/
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<LVE;l++){

        b=1.0/(1.0-(leta[l]*0.5));
        c=1.0+(leta[l]*0.5);
        indr = l*NX*NZ + gidx*NZ+gidz;
        rxz[indr]=b*(rxz[indr]*c-leta[l]*(dipkp*(vxz+vzx)));
        rxx[indr]=b*(rxx[indr]*c-leta[l]*((e*(vxx+vzz))-(d*vzz)));
        rzz[indr]=b*(rzz[indr]*c-leta[l]*((e*(vxx+vzz))-(d*vxx)));

        sumrxz+=rxz[indr];
        sumrxx+=rxx[indr];
        sumrzz+=rzz[indr];
    }
    /* and now the components of the stress tensor are
     completely updated */
    sxz[indv]-= lsxz + (DT2*sumrxz);
    sxx[indv]-= lsxx + (DT2*sumrxx) ;
    szz[indv]-= lszz + (DT2*sumrzz) ;

#endif

    m=inject_ind(gidz,  gidx);
    if (m!=-1){
        sxx[indv]= sxxbnd[m];
        szz[indv]= szzbnd[m];
        sxz[indv]= sxzbnd[m];
    }

    }
#endif

// Correct adjoint spatial derivatives to implement CPML
#if ABS_TYPE==1
    {
    int ind;

    if (gidz>NZ-NAB-FDOH-1){

        i =gidx-FDOH;
        k =gidz - NZ+NAB+FDOH+NAB;
        ind=2*NAB-1-k;

        psi_vx_z(k,i) = b_z_half[ind] * psi_vx_z(k,i) + a_z_half[ind] * vxzr;
        vxzr = vxzr / K_z_half[ind] + psi_vx_z(k,i);
        psi_vz_z(k,i) = b_z[ind+1] * psi_vz_z(k,i) + a_z[ind+1] * vzzr;
        vzzr = vzzr / K_z[ind+1] + psi_vz_z(k,i);

    }

#if FREESURF==0
    else if (gidz-FDOH<NAB){

        i =gidx-FDOH;
        k =gidz-FDOH;


        psi_vx_z(k,i) = b_z_half[k] * psi_vx_z(k,i) + a_z_half[k] * vxzr;
        vxzr = vxzr / K_z_half[k] + psi_vx_z(k,i);
        psi_vz_z(k,i) = b_z[k] * psi_vz_z(k,i) + a_z[k] * vzzr;
        vzzr = vzzr / K_z[k] + psi_vz_z(k,i);


    }
#endif

#if DEVID==0 & MYLOCALID==0
    if (gidx-FDOH<NAB){

        i =gidx-FDOH;
        k =gidz-FDOH;

        psi_vx_x(k,i) = b_x[i] * psi_vx_x(k,i) + a_x[i] * vxxr;
        vxxr = vxxr / K_x[i] + psi_vx_x(k,i);
        psi_vz_x(k,i) = b_x_half[i] * psi_vz_x(k,i) + a_x_half[i] * vzxr;
        vzxr = vzxr / K_x_half[i] + psi_vz_x(k,i);

    }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-NAB-FDOH-1){

        i =gidx - NX+NAB+FDOH+NAB;
        k =gidz-FDOH;
        ind=2*NAB-1-i;


        psi_vx_x(k,i) = b_x[ind+1] * psi_vx_x(k,i) + a_x[ind+1] * vxxr;
        vxxr = vxxr /K_x[ind+1] + psi_vx_x(k,i);
        psi_vz_x(k,i) = b_x_half[ind] * psi_vz_x(k,i) + a_x_half[ind] * vzxr;
        vzxr = vzxr / K_x_half[ind]  +psi_vz_x(k,i);


    }
#endif
    }
#endif

// Update adjoint stresses
    {
#if LVE==0

        lsxz=(fipkp*(vxzr+vzxr));
        lsxx=((g*(vxxr+vzzr))-(f*vzzr));
        lszz=((g*(vxxr+vzzr))-(f*vxxr));

        sxzr[indv]+=lsxz;
        sxxr[indv]+=lsxx;
        szzr[indv]+=lszz;

#else

    /* computing sums of the old memory variables */
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<LVE;l++){
        indr = l*NX*NZ + gidx*NZ+gidz;
        sumrxz+=rxzr[indr];
        sumrxx+=rxxr[indr];
        sumrzz+=rzzr[indr];
    }

    /* updating components of the stress tensor, partially */
    lsxz=(fipkp*(vxzr+vzxr))+(DT2*sumrxz);
    lsxx=((g*(vxxr+vzzr))-(f*vzzr))+(DT2*sumrxx);
    lszz=((g*(vxxr+vzzr))-(f*vxxr))+(DT2*sumrzz);


    /* now updating the memory-variables and sum them up*/
    sumrxz=sumrxx=sumrzz=0;
    for (l=0;l<LVE;l++){
        //those variables change sign in reverse time
        b=1.0/(1.0+(leta[l]*0.5));
        c=1.0-(leta[l]*0.5);

        rxzr[indr]=b*(rxzr[indr]*c-leta[l]*(dipkp*(vxzr+vzxr)));
        rxxr[indr]=b*(rxxr[indr]*c-leta[l]*((e*(vxxr+vzzr))-(d*vzzr)));
        rzzr[indr]=b*(rzzr[indr]*c-leta[l]*((e*(vxxr+vzzr))-(d*vxxr)));

        sumrxz+=rxzr[indr];
        sumrxx+=rxxr[indr];
        sumrzz+=rzzr[indr];
    }

    /* and now the components of the stress tensor are
     completely updated */
    sxzr[indv]+=lsxz + (DT2*sumrxz);
    sxxr[indv]+= lsxx + (DT2*sumrxx) ;
    szzr[indv]+= lszz + (DT2*sumrzz) ;


#endif
    }

// Absorbing boundary
#if ABS_TYPE==2
    {
#if FREESURF==0
    if (gidz-FDOH<NAB){
        sxzr[indv]*=taper[gidz-FDOH];
        sxxr[indv]*=taper[gidz-FDOH];
        szzr[indv]*=taper[gidz-FDOH];
    }
#endif

    if (gidz>NZ-NAB-FDOH-1){
        sxzr[indv]*=taper[NZ-FDOH-gidz-1];
        sxxr[indv]*=taper[NZ-FDOH-gidz-1];
        szzr[indv]*=taper[NZ-FDOH-gidz-1];
    }


#if DEVID==0 & MYLOCALID==0
    if (gidx-FDOH<NAB){
        sxzr[indv]*=taper[gidx-FDOH];
        sxxr[indv]*=taper[gidx-FDOH];
        szzr[indv]*=taper[gidx-FDOH];
    }
#endif

#if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
    if (gidx>NX-NAB-FDOH-1){
        sxzr[indv]*=taper[NX-FDOH-gidx-1];
        sxxr[indv]*=taper[NX-FDOH-gidx-1];
        szzr[indv]*=taper[NX-FDOH-gidx-1];
    }
#endif
    }
#endif

// Shear wave modulus and P-wave modulus gradient calculation on the fly
#if BACK_PROP_TYPE==1
    #if RESTYPE==0
        float c1=1.0/( (2.0*lM-2.0*lmu)*(2.0*lM-2.0*lmu) );


        // The sxz/shear term is evaluated at the muipkp (staggered) position,
        // not the cell-centred mu used by c1/c5's sxx/szz terms -- matching
        // grad_dft2D.cl's imuipkp2 = 1/(muipkp*muipkp). Kept in its own
        // gradmuipkp accumulator (not folded into gradmu) so
        // average_grad_transpose() (calc_grad.c) can apply the harmonic-mean
        // averaging Jacobian to it separately, mirroring grad_dft2D.cl's
        // Gmu/Gmuipkp split.
        float c3=1.0/(fipkp*fipkp);
        float c5=0.25/(lmu*lmu);

        /* The adjoint increment this step is NOT lsxx alone. The reverse loop
         * runs `inject residuals -> update_grid_adj`, so
         *     sigma~(t) = sigma~(t+1) + res(t) + lsxx(t)
         *  => d(sigma~)(t) = res(t) + lsxx(t),
         * and lsxx is only the PROPAGATION part computed here from spatial
         * derivatives -- the residual was added to sxxr before this kernel
         * ran. Pairing the forward field against that partial increment drops
         * res(t), which is nonzero only in receiver cells. Measured there:
         * FD/adjoint = -0.4537 (WRONG SIGN) against 0.995-1.023 at every
         * neighbouring cell, and back_prop_type=2 -- whose frequency-domain
         * form carries the full d(sigma~) -- gets the same cell right (0.9746).
         *
         * This is the exact mirror of the eq. (26a) source term, which this
         * kernel drops at SOURCE cells for the same reason: a forward field
         * paired against an incomplete increment.
         *
         * kernel_residuals() injects pout[NT*g+nt]/n2ave into each of
         * sxxr,szzr for the "p" trans_var, so the TRACE receives exactly
         * pout[NT*g+nt]; being split equally it cancels in the deviatoric (c5)
         * and shear (c3) terms, so only the trace needs it.
         *
         * Invisible to the FD suite: _patch keeps 16 cells clear of receivers
         * as well as sources. */
        float restr = 0.0f;
        /* Only when the "p" trans_var is an output: with velocity receivers
         * (seisout=1) the residual goes into vx/vz instead, `pout` is not a
         * live buffer, and reading it faults. The velocity case is handled in
         * update_adjv2D.cl, where that residual belongs. */
        #if GRADOUT==1 && PRESOUT==1
        if (nrec>0){
            for (int g=0; g<nrec; g++){
                int ri=(int)(rec_pos[0+8*g]/DH)+FDOH;
                int rk=(int)(rec_pos[2+8*g]/DH)+FDOH;
                if (ri==gidx && rk==gidz){
                    #if FP16==0
                    restr += pout[NT*g+nt];
                    #elif defined(__SEISCL__)
                    restr += ldexp(pout[NT*g+nt], res_scale);
                    #else
                    restr += scalbnf(pout[NT*g+nt], res_scale);
                    #endif
                }
            }
        }
        #endif

        float dM=c1*( sxx[indv]+szz[indv] )*( lsxx+lszz+restr );

        gradM[indp]+=-dM;
        gradmuipkp[indp]+=-c3*(sxz[indv]*lsxz);
        gradmu[indp]+=dM-c5*(  (sxx[indv]-szz[indv])*(lsxx-lszz)  );

        /* Source term of the misfit gradient -- GJI 2017 eq. (26a), thesis
         * eq. (3.51):
         *
         *     dJ/dm = -<psi, T dLambda^-1/dm T (A phi' + B phi - s)>
         *
         * The accumulation just above is the (A phi' + B phi) part only: it
         * is the discrete form of -c1M*P1 with P1 = <sigma~_kk, dt sigma_kk>
         * (eq. A2a), obtained by parts -- in reverse time the adjoint
         * increment is d(sigma~) = -dt*dt(sigma~), so
         * sum_t (-dM) = +c1<sigma,dt sigma~> = -c1<sigma~,dt sigma> = -c1M*P1.
         * The "- s" in the bracket never gets integrated by parts, so it
         * survives as a separate, purely local term
         *
         *     +c1M * <sigma~_kk , s_kk>
         *
         * which the correlation cannot produce and which is nonzero ONLY in
         * cells containing a source. That is exactly the observed symptom:
         * a wrong gradient confined to source cells.
         *
         * Only P1 is affected. An isotropic (pressure) source injects the
         * same amp/n2ave into each normal stress, so in P4's deviatoric
         * combination (N-1)s_ii - sum_{j!=i} s_jj = (N-1)s - (N-1)s = 0, and
         * it touches neither the shear stresses (P3) nor the velocities
         * (dJ/drho), so no other dot product picks it up.
         *
         * s_kk is the TRACE of the injected source. kernel_sources() (see
         * automatic_kernels.c) injects amp/n2ave into each of sxx,szz for the
         * "p" trans_var, so the trace receives exactly amp -- with pdir=+1,
         * the FORWARD sign, even though this kernel runs in the adjoint pass
         * where the re-injection that undoes the forward source uses pdir=-1.
         * DT is already inside amp, matching dM's adjoint increment which
         * carries its own dt through the dt/dh-scaled moduli, so the two
         * terms are in the same units. */
        #if GRADOUT==1
        if (nsrc>0){
            for (int srci=0; srci<nsrc; srci++){
                if ((int)src_pos[4+5*srci]==100){
                    int si=(int)(src_pos[0+5*srci]/DH)+FDOH;
                    int sk=(int)(src_pos[2+5*srci]/DH)+FDOH;
                    if (si==gidx && sk==gidz){
                        #if FP16==0
                        float samp = DT*src[srci*NT+nt];
                        #elif defined(__SEISCL__)
                        float samp = ldexp(DT*src[srci*NT+nt], src_scale);
                        #else
                        float samp = scalbnf(DT*src[srci*NT+nt], src_scale);
                        #endif
                        /* The adjoint FIELD (after this step's update), not
                         * the increment dM pairs against: <sigma~,s> is the one
                         * term of the bracket that is never integrated by
                         * parts.  Checked against the two neighbouring
                         * conventions (before the update, and the midpoint);
                         * this one is the exact match. */
                        float Csrc = c1*( sxxr[indv]+szzr[indv] )*samp;
                        gradM[indp]  += Csrc;
                        gradmu[indp] += -Csrc;
                    }
                }
            }
        }
        #endif

        #if HOUT==1
            float dMH=c1*(sxx[indv]+szz[indv])*(sxx[indv]+szz[indv]);
            HM[indp]+= dMH;
            Hmu[indp]+=c3*sxz[indv]*sxz[indv]-dM+c5*(sxx[indv]-szz[indv])*(sxx[indv]-szz[indv]) ;
        #endif
    #endif
    
    #if RESTYPE==1
        /* Missing the c1=1/(2M-2mu)^2 normalization RESTYPE==0 has just
         * above -- gradM here was off from a correctly-calibrated gradient
         * by that entire factor (confirmed: FD ratios of order 1e12-1e14,
         * not ~1, stable across eps -- a real, consistent direction, just
         * unnormalized). Not simply re-added unguarded: c1 divides by
         * (M-mu), which is exactly 0 in FREESURF==2's vacuum band, and
         * avoiding exactly that division is why RESTYPE==1 exists at all
         * (see the v1-constraint comment above, in assign_modeling_case.c).
         * Guarded the same way as the other zero-material guards in this
         * codebase (e.g. the 3D shear-gradient NaN fix on notes/todo.md
         * item 1): zero the coefficient instead of dividing when the
         * denominator is degenerate, rather than computing Inf/NaN and
         * hoping it gets cropped away. */
        float fMmu = 2.0*lM-2.0*lmu;
        float c1 = (fMmu!=0.0) ? 1.0/(fMmu*fMmu) : 0.0;
        /* Same receiver-cell residual term as the RESTYPE==0 branch above. */
        float restr1 = 0.0f;
        #if GRADOUT==1 && PRESOUT==1
        if (nrec>0){
            for (int g=0; g<nrec; g++){
                int ri=(int)(rec_pos[0+8*g]/DH)+FDOH;
                int rk=(int)(rec_pos[2+8*g]/DH)+FDOH;
                if (ri==gidx && rk==gidz){
                    #if FP16==0
                    restr1 += pout[NT*g+nt];
                    #elif defined(__SEISCL__)
                    restr1 += ldexp(pout[NT*g+nt], res_scale);
                    #else
                    restr1 += scalbnf(pout[NT*g+nt], res_scale);
                    #endif
                }
            }
        }
        #endif
        float dM=c1*( sxx[indv]+szz[indv] )*( lsxx+lszz+restr1 );

        gradM[indp]+=-dM;
        #if HOUT==1
        float dMH= c1*(sxx[indv]+szz[indv])*(sxx[indv]+szz[indv]);
        HM[indp]+= dMH;

        #endif

        /* gradmu was never computed at all here (only gradM, above) --
         * the same missing-normalization pattern as c1, for the same
         * documented reason (RESTYPE==0's c3=1/mu^2 divides by zero in
         * the vacuum band). Added with the same zero-guard as c1;
         * gradmuipkp (the sxz-only staggered contribution RESTYPE==0
         * doesn't split out either way) is intentionally still not
         * computed here -- out of scope for this pass, matches item 1's
         * material-averaging work being separate from this fix. */
        float c3 = (lmu!=0.0) ? 1.0/(lmu*lmu) : 0.0;
        float c5 = 0.25*c3;
        gradmu[indp]+=-c3*(sxz[indv]*lsxz)+dM-c5*( (sxx[indv]-szz[indv])*(lsxx-lszz) );
        #if HOUT==1
        Hmu[indp]+=c3*sxz[indv]*sxz[indv]-dM+c5*(sxx[indv]-szz[indv])*(sxx[indv]-szz[indv]);
        #endif
    #endif

#endif

#if GRADSRCOUT==1
//TODO
//    float pressure;
//    if (nsrc>0){
//
//        for (int srci=0; srci<nsrc; srci++){
//
//            int SOURCE_TYPE= (int)srcpos_loc(4,srci);
//
//            if (SOURCE_TYPE==1){
//                int i=(int)(srcpos_loc(0,srci)-0.5)+FDOH;
//                int k=(int)(srcpos_loc(2,srci)-0.5)+FDOH;
//
//
//                if (i==gidx && k==gidz){
//
//                    pressure=( sxxr[indv]+szzr[indv] )/(2.0*DH*DH);
//                    if ( (nt>0) && (nt< NT ) ){
//                        gradsrc(srci,nt+1)+=pressure;
//                        gradsrc(srci,nt-1)-=pressure;
//                    }
//                    else if (nt==0)
//                        gradsrc(srci,nt+1)+=pressure;
//                    else if (nt==NT)
//                        gradsrc(srci,nt-1)-=pressure;
//
//                }
//            }
//        }
//    }

#endif

}

