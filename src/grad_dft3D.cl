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

/* Frequency-domain (BACK_PROP_TYPE==2) gradient correlation, 3D, elastic.
 *
 * Direct 3D extension of grad_dft2D.cl -- same structure, same coefficient
 * formulas (grad_coefelast_0's c[0..4]/c[16..20] are already generic in ND,
 * so they are unchanged), the only difference is the dot products sum over
 * the extra field components (vy, syy, sxy, syz) that 3D has and 2D does not.
 * Cross-checked term by term against the host calc_grad() ND==3, L==0 branch:
 * dot[0]/dot[4] there are exactly d0/d4 here, dot[2] is d2xy+d2xz+d2yz, and
 * dot[8] is d8x+d8y+d8z, and calc_grad.c's cl_dev(a,b,c,N-1) = (N-1)*a-b-c
 * matches the sxx_myyzz/syy_mxxzz/szz_mxxyy combinations used for d4.
 *
 * One work item per model cell. See grad_dft2D.cl for the full rationale
 * (replaces the host calc_grad() for this case, everything in double
 * precision since this runs once per shot rather than once per time step).
 *
 * Each shear plane (sxy/sxz/syz) is driven by its own staggered mu --
 * muipjp/muipkp/mujpkp -- not by the cell-centred mu, so its correlation's
 * coefficient is evaluated at that staggered position and its gradient
 * stored there, mirroring grad_dft2D.cl's muipkp handling. Likewise each
 * velocity component (vx/vy/vz) sits at rip/rjp/rkp, not at a cell-centred
 * rho: gradrho gets no correlation term at all, exactly as in 2D.
 * average_grad_transpose() folds all six staggered gradients back onto the
 * cell-centred mu/rho. This is what makes the density (and shear-plane mu)
 * gradient in the 3D DFT path agree with the FD test -- previously they were
 * evaluated with the cell-centred coefficients and never routed through the
 * averaging transpose at all (notes/3d-gradient-findings.md, "Item 6").
 *
 * L>0 (viscoelastic) is not handled here -- see grad_dft3D_visc.cl.
 * assign_modeling_case.c only selects this kernel when L==0.
 */

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/* NX, NY, NZ are the *padded* extents (clprogram.c defines them as
 * N[i]+FDORDER), matching the X-slowest, Z-fastest layout used everywhere
 * else in the 3D kernels (e.g. update_v3D.cl's indp). */
/* NZS/NYS/NXS, not NZ/NY/NX: the latter are halved on the fastest axis when
 * FP16>0, since the update kernels address the wavefield as float2/half2.
 * This correlation is scalar -- savefreqs writes one float2 spectrum entry
 * per scalar element. See grad_dft2D.cl. */
#define NZM (NZS - 2*FDOH)
#define NYM (NYS - 2*FDOH)
#define NXM (NXS - 2*FDOH)
#define NPAD (NXS*NYS*NZS)
#define indf(f,i,j,k) ((f)*NPAD + ((i)+FDOH)*NYS*NZS + ((j)+FDOH)*NZS + ((k)+FDOH))

/* Scalar half load for the parameter buffers when FP16>1 -- see
 * grad_dft2D.cl for why this is not __pprec/__pconv. */
#if FP16>1
    #define PARARG half
    #ifdef __OPENCL_VERSION__
        #define PARCONV(x) (x)
    #else
        #define PARCONV(x) __half2float(x)
    #endif
#else
    #define PARARG float
    #define PARCONV(x) (x)
#endif

#ifdef __OPENCL_VERSION__
    #define COSPIF(x) cospi(x)
    #define SINPIF(x) sinpi(x)
#else
    #define COSPIF(x) cospif(x)
    #define SINPIF(x) sinpif(x)
#endif

/* itreal(a,b) = Re(conj(a)*i*b), the form <sigma~, dt sigma> needs.  The
 * eq. (26a) source term <sigma~, s> has no time derivative and so needs the
 * plain Re(conj(a)*b). */
LFUNDEF double rreal(float2 a, float2 b)
{
    return (double)a.x*(double)b.x + (double)a.y*(double)b.y;
}

LFUNDEF double itreal(float2 a, float2 b)
{
    return (double)a.y*(double)b.x - (double)a.x*(double)b.y;
}

/* grad_coefelast_1's c[2]=1/mu^2, evaluated at whichever staggered mu drives
 * that shear plane rather than the cell-centred one. See grad_dft2D.cl's
 * imuipkp2 for the 2D precedent. */
LFUNDEF double imu2_stag(double mu_s)
{
    return (mu_s>=1.0) ? 1.0/(mu_s*mu_s) : 0.0;
}

FUNDEF void calc_grad_dft(GLOBARG float * gradfreqsn,
                          GLOBARG PARARG * M,
                          GLOBARG PARARG * mu,
                          GLOBARG PARARG * rho,
                          GLOBARG PARARG * muipjp,
                          GLOBARG PARARG * muipkp,
                          GLOBARG PARARG * mujpkp,
                          GLOBARG float * gradM,
                          GLOBARG float * gradmu,
                          GLOBARG float * gradrho,
                          GLOBARG float * gradmuipjp,
                          GLOBARG float * gradmuipkp,
                          GLOBARG float * gradmujpkp,
                          GLOBARG float * gradrip,
                          GLOBARG float * gradrjp,
                          GLOBARG float * gradrkp,
                          GLOBARG float * HM,
                          GLOBARG float * Hmu,
                          GLOBARG float * Hrho,
                          GLOBARG float2 * fvx_f,
                          GLOBARG float2 * fvy_f,
                          GLOBARG float2 * fvz_f,
                          GLOBARG float2 * fsxx_f,
                          GLOBARG float2 * fsyy_f,
                          GLOBARG float2 * fszz_f,
                          GLOBARG float2 * fsxy_f,
                          GLOBARG float2 * fsxz_f,
                          GLOBARG float2 * fsyz_f,
                          GLOBARG float2 * fvx,
                          GLOBARG float2 * fvy,
                          GLOBARG float2 * fvz,
                          GLOBARG float2 * fsxx,
                          GLOBARG float2 * fsyy,
                          GLOBARG float2 * fszz,
                          GLOBARG float2 * fsxy,
                          GLOBARG float2 * fsxz,
                          GLOBARG float2 * fsyz,
                          GLOBARG float * src,
                          GLOBARG float * src_pos,
                          int nsrc,
                          int src_scale,
                          int res_scale,
                          int par_scale)
{

    #ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
    #else
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    #endif

    if (gid >= NXM*NYM*NZM)
        return;

    int i = gid/(NYM*NZM);
    int rem = gid - i*NYM*NZM;
    int j = rem/NZM;
    int k = rem - j*NZM;
    int f;

    /* Undo the internal non-dimensionalization: the coefficient expressions
     * below are in physical units. Mirrors transf_grad() in reverse. */
    double s2    = pow(2.0, -(double)par_scale);
    double dhdt  = (double)DH/(double)DT;
    double lrho  = (double)PARCONV(rho[gid]);
    double rho_p = (lrho!=0.0) ? (1.0/lrho)*((double)DT/(double)DH)*s2 : 0.0;
    double M_p   = (double)PARCONV(M[gid])*dhdt*s2;
    double mu_p  = (double)PARCONV(mu[gid])*dhdt*s2;
    /* Each shear plane's own staggered mu -- see the file header. */
    double muipjp_p = (double)PARCONV(muipjp[gid])*dhdt*s2;
    double muipkp_p = (double)PARCONV(muipkp[gid])*dhdt*s2;
    double mujpkp_p = (double)PARCONV(mujpkp[gid])*dhdt*s2;
    double imuipjp2 = imu2_stag(muipjp_p);
    double imuipkp2 = imu2_stag(muipkp_p);
    double imujpkp2 = imu2_stag(mujpkp_p);

    /* df = 1/(NTNYQ*dt*DTNYQ), from defines that already exist. */
    double dftdf = 1.0/((double)NTNYQ*(double)DT*(double)DTNYQ);
    /* Parseval factor: 1/(NTNYQ*DTNYQ), not 1/NTNYQ -- see calc_grad.c. */
    double dftnorm = (double)NTNYQ*(double)DTNYQ;

    /* ND is a build-option macro (-D ND=%d), so it must not be shadowed. */
    const double NDd = (double)ND;
    double den = (NDd*M_p - 2.0*(NDd-1.0)*mu_p);
    den = den*den;

    /* Internal (M, mu, rho) coefficients -- grad_coefelast_1, i.e. _0 with
     * the parameterization chain rule factored out. chain_rule_par_type()
     * applies it once on the host, so the c16..c20 group is gone; see
     * grad_dft2D.cl for the derivation and why that removes a whole class of
     * sign bug. Formulas already generic in ND. */
    double iden=0.0, imu2=0.0, i3den=0.0, i2ndmu2=0.0;
    if (den>0.0){
        iden = 1.0/den;
        if (mu_p>=1.0){
            imu2    = 1.0/(mu_p*mu_p);
            i3den   = (NDd+1.0)/3.0*iden;
            i2ndmu2 = imu2/(2.0*NDd);
        }
    }

    /* Which source, if any, sits in this cell (eq. 26a source term). */
    /* A FORCE source is the mirror case: velocity block only, so it
     * corrects d8x/d8y/d8z (hence the STAGGERED buoyancy slots) and
     * leaves every stress correlation alone. The two are mutually
     * exclusive. Types index kernel_sources()'s src_names[] =
     * {vx,vy,vz,p,...}; 100 is the "p" trans_var. */
    int ssrc = -1, sstype = -1;
    for (int q=0; q<nsrc; q++){
        int st = (int)src_pos[4+5*q];
        if ((st==100 || st==0 || st==1 || st==2)
            && (int)(src_pos[0+5*q]/DH)==i
            && (int)(src_pos[1+5*q]/DH)==j
            && (int)(src_pos[2+5*q]/DH)==k){
            ssrc = q; sstype = st; break;
        }
    }

    double gM=0.0, gmu=0.0;
    double gmuipjp=0.0, gmuipkp=0.0, gmujpkp=0.0;
    double grip=0.0, grjp=0.0, grkp=0.0;
#if HOUT==1
    double hM=0.0, hmu=0.0, hrho=0.0;
#endif

    for (f=0; f<NFREQS; f++){

        double w = 2.0*3.14159265358979323846*dftdf*(double)gradfreqsn[f];
        int id = indf(f,i,j,k);

        /* S(w): transform of the injected source in savefreqs' convention;
         * see grad_dft2D.cl for the derivation and the two distinct weights. */
        float2 S; S.x = 0.0f; S.y = 0.0f;
        if (ssrc>=0){
            for (int t=0; t<NT; t++){
                float ang = 2.0f*gradfreqsn[f]*(float)(t-TMIN)
                            /((float)NTNYQ*(float)DTNYQ);
                #if FP16==0
                float amp = DT*src[ssrc*NT+t];
                #elif defined(__OPENCL_VERSION__)
                float amp = ldexp(DT*src[ssrc*NT+t], src_scale);
                #else
                float amp = scalbnf(DT*src[ssrc*NT+t], src_scale);
                #endif
                S.x +=  amp*COSPIF(ang);
                S.y += -amp*SINPIF(ang);
            }
        }

        float2 Fxx = fsxx_f[id], Fyy = fsyy_f[id], Fzz = fszz_f[id];
        float2 Fxy = fsxy_f[id], Fxz = fsxz_f[id], Fyz = fsyz_f[id];
        float2 Axx = fsxx[id],   Ayy = fsyy[id],   Azz = fszz[id];
        float2 Axy = fsxy[id],   Axz = fsxz[id],   Ayz = fsyz[id];

        float2 Fpp, App;
        Fpp.x = Fxx.x + Fyy.x + Fzz.x;  Fpp.y = Fxx.y + Fyy.y + Fzz.y;
        /* savefreqs runs BEFORE the source injection (time_stepping.c), so the
         * stored forward spectrum lacks this step's source while
         * BACK_PROP_TYPE==1 correlates the field after it. Restore it with
         * savefreqs' own per-sample field weight DT*DTNYQ. Only the trace is
         * affected: an isotropic source adds equally to sxx,syy,szz, so the
         * deviatoric combinations and the shear planes are untouched, which is
         * why Fxx/Fyy/Fzz themselves are left alone. */
        if (sstype==100){
            Fpp.x += (float)((double)DT*(double)DTNYQ)*S.x;
            Fpp.y += (float)((double)DT*(double)DTNYQ)*S.y;
        }
        App.x = Axx.x + Ayy.x + Azz.x;  App.y = Axx.y + Ayy.y + Azz.y;

        /* The deviatoric combination of the published P4 (eq. A2d):
         * (N-1)*own - other - other, i.e. the diagonal term carries the
         * weight (N-1) = 2 in 3D, not 1. This read `Fxx.x - Fyy.x - Fzz.x`
         * until 2026-09-03 -- the 2D form (where N-1 == 1) transcribed
         * unchanged into 3D, matching calc_grad.c's then-equally-wrong
         * cl_diff. See cl_dev()'s comment in calc_grad.c and notes/todo.md
         * item 0i; back_prop_type=1's update_adjs3D.cl always had the 2.0. */
        double ndm1 = NDd - 1.0;
        float2 Fxx_myyzz, Fyy_mxxzz, Fzz_mxxyy;
        Fxx_myyzz.x = (float)(ndm1*Fxx.x) - Fyy.x - Fzz.x;
        Fxx_myyzz.y = (float)(ndm1*Fxx.y) - Fyy.y - Fzz.y;
        Fyy_mxxzz.x = (float)(ndm1*Fyy.x) - Fxx.x - Fzz.x;
        Fyy_mxxzz.y = (float)(ndm1*Fyy.y) - Fxx.y - Fzz.y;
        Fzz_mxxyy.x = (float)(ndm1*Fzz.x) - Fxx.x - Fyy.x;
        Fzz_mxxyy.y = (float)(ndm1*Fzz.y) - Fxx.y - Fyy.y;

        /* P1 with the eq. (26a) bracket intact: <sigma~, dt sigma - s>. */
        double d0 = (w*itreal(App, Fpp) - (sstype==100 ? rreal(App, S) : 0.0))/dftnorm;
        /* Kept apart per shear plane -- sxy/sxz/syz are each driven by a
         * different staggered mu, so they cannot be summed before the
         * coefficient is applied. */
        double d2xy = w*itreal(Axy, Fxy)/dftnorm;
        double d2xz = w*itreal(Axz, Fxz)/dftnorm;
        double d2yz = w*itreal(Ayz, Fyz)/dftnorm;
        double d3 = d0;
        double d4 = w*(itreal(Axx, Fxx_myyzz)
                     + itreal(Ayy, Fyy_mxxzz)
                     + itreal(Azz, Fzz_mxxyy))/dftnorm;
        /* vx/vy/vz sit at rip/rjp/rkp respectively (update_v3D.cl): different
         * parameters, so kept apart rather than summed. */
        /* Velocity block: a FORCE source's eq. (26a) term, mirroring the
         * trace above. savefreqs omits the injection here too, so the forward
         * velocity spectrum gets the source restored with the FIELD weight
         * DT*DTNYQ (kernel_sources() adds `amp` straight into vx/vy/vz, with
         * no material factor), and the eq. (26a) term itself is the
         * rate-weighted rreal. dJ/drho has no c-factor (A1a). Each component
         * feeds its own STAGGERED buoyancy slot, which is what lets
         * average_grad_transpose() apply the averaging Jacobian. */
        float2 Fvx = fvx_f[id], Avx = fvx[id];
        float2 Fvy = fvy_f[id], Avy = fvy[id];
        float2 Fvz = fvz_f[id], Avz = fvz[id];
        if (sstype==0){ Fvx.x += (float)((double)DT*(double)DTNYQ)*S.x; Fvx.y += (float)((double)DT*(double)DTNYQ)*S.y; }
        if (sstype==1){ Fvy.x += (float)((double)DT*(double)DTNYQ)*S.x; Fvy.y += (float)((double)DT*(double)DTNYQ)*S.y; }
        if (sstype==2){ Fvz.x += (float)((double)DT*(double)DTNYQ)*S.x; Fvz.y += (float)((double)DT*(double)DTNYQ)*S.y; }
        double d8x = (w*itreal(Avx, Fvx)
                     - (sstype==0 ? rreal(Avx, S) : 0.0))/dftnorm;
        double d8y = (w*itreal(Avy, Fvy)
                     - (sstype==1 ? rreal(Avy, S) : 0.0))/dftnorm;
        double d8z = (w*itreal(Avz, Fvz)
                     - (sstype==2 ? rreal(Avz, S) : 0.0))/dftnorm;

#if HOUT==1
        /* Approximate (Gauss-Newton style) Hessian diagonal, same pattern as
         * grad_dft2D.cl: every term is w^2*|.|^2, non-negative, extended to
         * the extra 3D components. */
        {
            double w2 = w*w;
            double h0 = w2*((double)Fpp.x*(double)Fpp.x
                          + (double)Fpp.y*(double)Fpp.y)/dftnorm;
            double h2 = w2*(((double)Fxy.x*(double)Fxy.x + (double)Fxy.y*(double)Fxy.y)
                          + ((double)Fxz.x*(double)Fxz.x + (double)Fxz.y*(double)Fxz.y)
                          + ((double)Fyz.x*(double)Fyz.x + (double)Fyz.y*(double)Fyz.y))
                          /dftnorm;
            double h3 = h0;
            double h4 = w2*(((double)Fxx_myyzz.x*(double)Fxx_myyzz.x
                           + (double)Fxx_myyzz.y*(double)Fxx_myyzz.y)
                          + ((double)Fyy_mxxzz.x*(double)Fyy_mxxzz.x
                           + (double)Fyy_mxxzz.y*(double)Fyy_mxxzz.y)
                          + ((double)Fzz_mxxyy.x*(double)Fzz_mxxyy.x
                           + (double)Fzz_mxxyy.y*(double)Fzz_mxxyy.y))/dftnorm;
            float2 Vx = fvx_f[id], Vy = fvy_f[id], Vz = fvz_f[id];
            double h8 = w2*(((double)Vx.x*(double)Vx.x + (double)Vx.y*(double)Vx.y)
                          + ((double)Vy.x*(double)Vy.x + (double)Vy.y*(double)Vy.y)
                          + ((double)Vz.x*(double)Vz.x + (double)Vz.y*(double)Vz.y))
                          /dftnorm;
            hM   += h0*iden;
            hmu   += h2*imu2 - h3*i3den + h4*i2ndmu2;
            hrho += h8;
        }
#endif
        gM   += -d0*iden;
        gmu  += d3*i3den - d4*i2ndmu2;         /* sxx/syy/szz: cell-centred mu */
        gmuipjp += -d2xy*imuipjp2;             /* sxy: the averaged mu */
        gmuipkp += -d2xz*imuipkp2;             /* sxz: the averaged mu */
        gmujpkp += -d2yz*imujpkp2;             /* syz: the averaged mu */
        grip += -d8x;
        grjp += -d8y;
        grkp += -d8z;
    }

    /* The internal (M, mu, rho) gradient is the kernel's whole output; the
     * parameterization chain rule runs once on the host
     * (chain_rule_par_type). gradrho gets no correlation term at all:
     * density enters the physics only through rip/rjp/rkp, filled by
     * average_grad_transpose() along with gradmuipjp/muipkp/mujpkp -- see
     * grad_dft2D.cl. */
    gradM[gid]      += (float)gM;
    gradmu[gid]     += (float)gmu;
    gradmuipjp[gid] += (float)gmuipjp;
    gradmuipkp[gid] += (float)gmuipkp;
    gradmujpkp[gid] += (float)gmujpkp;
    gradrip[gid]    += (float)grip;
    gradrjp[gid]    += (float)grjp;
    gradrkp[gid]    += (float)grkp;
#if HOUT==1
    HM[gid]   += (float)hM;
    Hmu[gid]  += (float)hmu;
    Hrho[gid] += (float)hrho;
#endif
}
