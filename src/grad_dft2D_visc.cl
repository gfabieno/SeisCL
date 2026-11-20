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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See file COPYING
 * and/or <http://www.gnu.org/licenses/gpl-3.0.html>.
 --------------------------------------------------------------------------*/

/* Frequency-domain (BACK_PROP_TYPE==2) gradient correlation, 2D P-SV,
 * VISCOELASTIC (LVE>0).
 *
 * The viscoelastic counterpart of grad_dft2D.cl. Until this existed, an L>0
 * DFT gradient fell back to the host calc_grad(), which is #ifdef __SEISCL__
 * and a no-op stub in the CUDA build -- so a viscoelastic gradient came back
 * identically zero under CUDA, silently, and SeisCL.torch (CUDA-only) could
 * not do viscoelastic FWI at all. back_prop_type=1 is not an alternative: it
 * rejects L>0, because reverse-time reconstruction of a dissipative medium is
 * unconditionally unstable.
 *
 * A separate file rather than an #if branch inside grad_dft2D.cl, because
 * prog_args_list() (clprogram.c) parses the *raw* kernel signature text for
 * argument names before any preprocessing. Arguments guarded by #if would
 * still be listed and bound, desynchronising every argument index after them.
 * Same reason grad_dft2D_SH.cl is its own file.
 *
 * Transcribed from calc_grad()'s ND==2, L>0 block, which stays the reference
 * oracle (SEISCL_DFT_HOST=1 / SEISCL_DFT_CHECK=1) -- on an OpenCL build,
 * since that host path does not exist under CUDA.
 *
 * Conventions match grad_dft2D.cl: everything in double, one work item per
 * model cell, scalar padded extents NZS/NXS, the FP16 wavefield scaling undone
 * with src_scale/res_scale/par_scale, and the *internal* (M, mu, rho, taup,
 * taus) gradient emitted -- chain_rule_par_type() applies the parameterization
 * on the host.
 *
 * The shear stress sxz is driven by the staggered muipkp/tausipkp, not the
 * cell-centred mu/taus, so its correlation's coefficients are evaluated at
 * that staggered position and its gradient stored there (gradmuipkp,
 * gradtausipkp), exactly as grad_dft2D.cl already does for its elastic c[2].
 * Likewise the two velocity correlations go to rip/rkp rather than a
 * cell-centred gradrho. average_grad_transpose() folds all four back onto
 * the cell-centred mu/rho/taus. This closes the same material-averaging gap
 * that grad_dft3D.cl's elastic case had (notes/3d-gradient-findings.md,
 * "Item 6") for the viscoelastic 2D path.
 */

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/* Scalar padded extents -- not NZ/NX, which are halved on the fastest axis
 * when FP16>0. See grad_dft2D.cl. */
#define NZM (NZS - 2*FDOH)
#define NXM (NXS - 2*FDOH)
#define NPAD (NXS*NZS)
#define indf(f,i,k) ((f)*NPAD + ((i)+FDOH)*NZS + ((k)+FDOH))
/* Memory-variable spectra carry an extra mechanism axis, laid out
 * f -> l -> x -> z (calc_grad.c's indL). */
#define indr(f,l,i,k) ((f)*NPAD*LVE + (l)*NPAD + ((i)+FDOH)*NZS + ((k)+FDOH))

/* Scalar half load for the parameter buffers when FP16>1 -- see grad_dft2D.cl
 * for why this is not __pprec/__pconv. */
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

/* itreal(a,b) = Re(conj(a)*i*b), the form <sigma~, dt sigma> needs (dt <-> i*w).
 * The eq. (26a) source term <sigma~, s> has no time derivative and so needs the
 * plain Re(conj(a)*b). */
LFUNDEF double rreal(float2 a, float2 b)
{
    return (double)a.x*(double)b.x + (double)a.y*(double)b.y;
}

LFUNDEF double itreal(float2 a, float2 b)
{
    return (double)a.y*(double)b.x - (double)a.x*(double)b.y;
}

/* calc_grad.c's cl_rm: the memory-variable correlation. */
/* calc_grad.c's cl_rm: Re(conj(a)*(tausig + 1/(i*w))*b), i.e. <r~, tausig*r + R>
 * with R the alternative memory variable dt(R) = r (thesis eq. 3.16). That IS
 * A2b's <R~, (1+tausig*dt)r> integrated by parts, using r~ = -dt(R~)
 * (thesis eq. 3.33) -- so the operator itself was never wrong.
 *
 * What was wrong is the DISCRETIZATION. The memory variables are advanced by
 * CRANK-NICOLSON (thesis eq. 2.47), not by the leapfrog centred differences
 * the stresses and velocities use, and CN is the bilinear (Tustin) transform:
 *     i*w  <->  i*(2/dt)*tan(w*dt/2),
 * so the discrete integral is
 *     1/(i*w)  <->  -i*(dt/2)*cot(w*dt/2).
 * That is PURELY IMAGINARY -- CN introduces no real part -- so only 1/w is
 * replaced and tausig is untouched. (An earlier attempt used the inverse of
 * the leapfrog forward-difference operator instead, which does carry a real
 * part, and made taup/taus worse: interior 1.4436 -> 1.9555.)
 *
 * The leapfrog variables keep their own symbol (see sdt/s2dt in the d-terms):
 * two discretizations, two symbols. */
LFUNDEF double rmreal(float2 a, float2 b, double tausig, double w)
{
    double th  = 0.5*w*(double)DT;
    double iwd = 0.5*(double)DT*cos(th)/sin(th);   /* discrete 1/w */
    return tausig*((double)a.x*(double)b.x + (double)a.y*(double)b.y)
         + ((double)a.x*(double)b.y - (double)a.y*(double)b.x)*iwd;
}

/* calc_grad.c's cl_integral: divide the spectrum by i*w. */
/* calc_grad.c's cl_integral: a/(i*w). Crank-Nicolson's discrete integral is
 * -i*(dt/2)*cot(w*dt/2) -- same form, warped frequency, no real part. See
 * rmreal above. */
/* The memory variable enters the stress as the AVERAGE (r^n + r^{n+1})/2 (see
 * update_s2D.cl: DT2*sum(r) is added both before and after the update), whose
 * symbol is
 *     (1+z)/2 = cos(th)*e^{-i*th},   th = w*dt/2,  z = e^{-i*w*dt}
 * while savefreqs DFTs r at INTEGER steps and therefore stores plain R. The
 * stored spectrum has to be corrected by that factor before it is correlated.
 * SEISCL_HALFSTEP_SGN flips the sign of the phase for the time-reversed field. */
#ifndef HSGN
#define HSGN (-1.0)
#endif
LFUNDEF float2 halfstep(float2 a, double w, double sgn)
{
    double th = 0.5*w*(double)DT;
    double cr = cos(th)*cos(th);
    double ci = sgn*sin(th)*cos(th);
    float2 o;
    o.x = (float)((double)a.x*cr - (double)a.y*ci);
    o.y = (float)((double)a.x*ci + (double)a.y*cr);
    return o;
}

LFUNDEF float2 integ(float2 a, double w)
{
    double th  = 0.5*w*(double)DT;
    double iwd = 0.5*(double)DT*cos(th)/sin(th);   /* discrete 1/w */
    float2 o;
    o.x = (float)((double)a.y*iwd);
    o.y = (float)(-(double)a.x*iwd);
    return o;
}

/* grad_coefvisc_1's c[2]/c[5]/c[10]/c[13] group -- the shear-stress
 * correlation's own coefficients -- depends only on mu and taus, evaluated
 * at the staggered muipkp/tausipkp rather than the cell-centred ones. See
 * calc_grad.c's shear_stag_coef, which this mirrors exactly. */
LFUNDEF void shear_stag_coef(double mu_s, double taus_s, double al, double L,
                              double *c2, double *c5, double *c10, double *c13)
{
    *c2=0.0; *c5=0.0; *c10=0.0; *c13=0.0;
    if (mu_s>=1.0){
        if (L>0.0 && taus_s!=0.0){
            double Lst  = 1.0+L*taus_s;
            double ist  = 1.0+al*taus_s;
            double imu2 = 1.0/(mu_s*mu_s);
            *c2  = ist*imu2/Lst;
            *c5  = ist*imu2/taus_s;
            *c10 = (L-al)/(mu_s*Lst*Lst);
            *c13 = 1.0/(mu_s*taus_s*taus_s);
        } else {
            *c2 = 1.0/(mu_s*mu_s);
        }
    }
}

FUNDEF void calc_grad_dft(GLOBARG float * gradfreqsn,
                          GLOBARG float * FL,
                          GLOBARG PARARG * M,
                          GLOBARG PARARG * mu,
                          GLOBARG PARARG * rho,
                          GLOBARG PARARG * taup,
                          GLOBARG PARARG * taus,
                          GLOBARG PARARG * muipkp,
                          GLOBARG PARARG * tausipkp,
                          GLOBARG float * gradM,
                          GLOBARG float * gradmu,
                          GLOBARG float * gradrho,
                          GLOBARG float * gradtaup,
                          GLOBARG float * gradtaus,
                          GLOBARG float * gradmuipkp,
                          GLOBARG float * gradrip,
                          GLOBARG float * gradrkp,
                          GLOBARG float * gradtausipkp,
                          GLOBARG float2 * fvx_f,
                          GLOBARG float2 * fvz_f,
                          GLOBARG float2 * fsxx_f,
                          GLOBARG float2 * fszz_f,
                          GLOBARG float2 * fsxz_f,
                          GLOBARG float2 * frxx_f,
                          GLOBARG float2 * frzz_f,
                          GLOBARG float2 * frxz_f,
                          GLOBARG float2 * fvx,
                          GLOBARG float2 * fvz,
                          GLOBARG float2 * fsxx,
                          GLOBARG float2 * fszz,
                          GLOBARG float2 * fsxz,
                          GLOBARG float2 * frxx,
                          GLOBARG float2 * frzz,
                          GLOBARG float2 * frxz,
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

    if (gid >= NXM*NZM)
        return;

    int i = gid/NZM;
    int k = gid - i*NZM;
    int f, l;

    /* Undo the internal non-dimensionalization; the coefficients below are in
     * physical units. taup/taus are dimensionless and are not scaled. */
    double s2    = pow(2.0, -(double)par_scale);
    double dhdt  = (double)DH/(double)DT;
    double lrho  = (double)PARCONV(rho[gid]);
    double rho_p = (lrho!=0.0) ? (1.0/lrho)*((double)DT/(double)DH)*s2 : 0.0;
    double M_p   = (double)PARCONV(M[gid])*dhdt*s2;
    double mu_p  = (double)PARCONV(mu[gid])*dhdt*s2;
    double taup_p = (double)PARCONV(taup[gid]);
    double taus_p = (double)PARCONV(taus[gid]);
    /* sxz's own staggered mu/taus -- see the file header. */
    double muipkp_p = (double)PARCONV(muipkp[gid])*dhdt*s2;
    double tausipkp_p = (double)PARCONV(tausipkp[gid]);

    double dftdf  = 1.0/((double)NTNYQ*(double)DT*(double)DTNYQ);
    double dftnorm = (double)NTNYQ*(double)DTNYQ;

    /* FP16>0 keeps the wavefield in scaled units -- see grad_dft2D.cl. The
     * memory variables are stresses and carry no extra scaler, so they take
     * the same stress-stress factor. */
    double sc_ss = pow(2.0, -(double)src_scale - (double)res_scale);
    double sc_vv = sc_ss*pow(2.0, 2.0*(double)par_scale);

    const double NDd = (double)ND;
    const double Ld  = (double)LVE;
    /* calc_grad.c sets al=0 unconditionally (its only assignment), so every
     * (1+al*tau) factor in grad_coefvisc_1 is 1. Kept named rather than
     * folded away so the expressions stay comparable with the host. */
    /* The GSLS phase-velocity normalization. M()/mu() (assign_modeling_case.c)
     * divide the stored moduli by (1 + alpha*tau) so that the phase velocity
     * at f0 equals the vp/vs the user supplied -- the elastic convention. The
     * gradient with respect to tau at fixed vp therefore has to carry that
     * dependence, and it enters the coefficients as calc_grad.c's `al`:
     *
     *     al = sum_l r^2/(1+r^2),   r = f0/FL[l]
     *
     * It is NOT zero. calc_grad.c initialises `al=0` and then accumulates
     * into it a few lines later, which is easy to miss; hardcoding 0 here
     * scaled the c[8]/c[10]/c[11]/c[12] group -- i.e. gradtaup and gradtaus --
     * by exactly (1-al), verified against the host oracle at f0/FL = 0.5, 1
     * and 2. */
    double al = 0.0;
    for (l=0; l<LVE; l++){
        double r = (double)FREQ0/(double)FL[l];
        al += r*r/(1.0 + r*r);
    }
    /* GJI 2017 Appendix A is written for the paper's moduli M = rho vp^2,
     * mu = rho vs^2; SeisCL stores them divided by (1 + alpha tau) (phase
     * velocity at f0). The coefficients below need the paper's values, and
     * chain_rule_par_type() converts dJ/dM_paper (notes
     * viscoelastic-gradient-theory.md, 2026-11-17). */
    M_p      *= (1.0 + al*taup_p);
    mu_p     *= (1.0 + al*taus_p);
    muipkp_p *= (1.0 + al*tausipkp_p);

    /* Exact discrete gradient (notes viscoelastic-gradient-theory.md, 2026-11-19 night).
     *
     * Per cell, the stress update of update_s2D.cl,
     *     sigma^n - sigma^{n-1} = G q + DT2 (sum_l r_l^{n-1} + sum_l r_l^n),
     *     r_l^n = b_l (c_l r_l^{n-1} - eta_l E q),   q = D v^{n-1/2},
     * is, in the DFT convention X = sum_n x^n exp(-i w n DT) with u = exp(-i w DT),
     *     (1-u) Sigma = u K(w) Q,   K(w) = G - phi(w) Ehat,
     *     phi(w) = 1/2 sum_l b_l eta_l (1+u)/(1 - b_l c_l u),   Ehat = E*DT.
     * G holds the unrelaxed moduli g = M(1+L taup), f = 2 mu(1+L taus) and Ehat the
     * memory moduli M taup, 2 mu taus; phi -> 1 at w -> 0 (K = relaxed modulus).
     * The back_prop_type 2 adjoint run is this same scheme in reversed time, and the
     * Lagrange multiplier of the stress equation is K_A(w)^{-1} sigma~ with
     * K_A = conj(K). The exact gradient is therefore the elastic one with G^{-1}
     * replaced by K(w)^{-1}, and dK = dG - phi dEhat: no memory spectra are needed.
     * The normal block G = [[g, g-f],[g-f, g]] has eigenvectors (1,1) (value
     * Kp = 2g - f - phi(2e - d)) and (1,-1) (Ks = f - phi d). */
    double ipt = 1.0+al*taup_p, ist = 1.0+al*taus_p;
    double istx = 1.0+al*tausipkp_p;
    double gg = M_p*(1.0+Ld*taup_p)/ipt;          /* unrelaxed P modulus */
    double ff = 2.0*mu_p*(1.0+Ld*taus_p)/ist;     /* 2 x unrelaxed shear modulus */
    double ee = M_p*taup_p/ipt;                   /* memory moduli, times DT */
    double dd = 2.0*mu_p*taus_p/ist;
    double fx = muipkp_p*(1.0+Ld*tausipkp_p)/istx;
    double dx = muipkp_p*tausipkp_p/istx;
    int live = (rho_p>0.0) && (mu_p>=1.0) && (2.0*gg-ff>0.0);
    int livex = (muipkp_p>=1.0);

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
            && (int)(src_pos[2+5*q]/DH)==k){
            ssrc = q; sstype = st; break;
        }
    }

    double Gg=0.0, Gf=0.0, Ge=0.0, Gd=0.0, Gfx=0.0, Gdx=0.0;
    double Grip=0.0, Grkp=0.0;

    for (f=0; f<NFREQS; f++){

        double w = 2.0*3.14159265358979323846*dftdf*(double)gradfreqsn[f];
        double wdt  = w*(double)DT;
        double sdt  = sin(wdt)/(double)DT;
        double s2dt = 2.0*sin(0.5*wdt)*sin(0.5*wdt)/(double)DT;
        int id = indf(f,i,k);

        /* phi(w) */
        double ur = cos(wdt), ui = -sin(wdt);
        double phr = 0.0, phi_ = 0.0;
        for (l=0; l<LVE; l++){
            double et = 2.0*3.14159265358979323846*(double)FL[l]*(double)DT;
            double bb = 1.0/(1.0+0.5*et), cc = 1.0-0.5*et;
            double nr = 1.0+ur, ni = ui;                 /* 1+u */
            double dr = 1.0-bb*cc*ur, di = -bb*cc*ui;    /* 1-bcu */
            double den = dr*dr+di*di;
            double qr = (nr*dr+ni*di)/den, qi = (ni*dr-nr*di)/den;
            phr += 0.5*bb*et*qr; phi_ += 0.5*bb*et*qi;
        }

        /* S(w): transform of the injected source, savefreqs' convention.
         * See grad_dft2D.cl for the derivation and why the two uses below take
         * DIFFERENT weights. */
        /* A force source goes into vx/vy/vz, stored in units 2^(src_scale -
         * par_scale) (kernel_sources): its amplitude carries that shift too. */
        int vshift = (sstype>=0 && sstype<=2) ? par_scale : 0;
        float2 S; S.x = 0.0f; S.y = 0.0f;
        /* Sd: the same transform on the samples savefreqs keeps (t >= TMIN,
         * every DTNYQ steps). It restores the source in a FIELD spectrum,
         * which savefreqs sampled; S (every step) is the eq. (26a) source
         * term. With DTNYQ>1, restoring with S added DTNYQ times the missing
         * increment: test_dft_osamp_convergence errors of 1.45 (DTNYQ-1). */
        float2 Sd; Sd.x = 0.0f; Sd.y = 0.0f;
        if (ssrc>=0){
            for (int t=0; t<NT; t++){
                float ang = 2.0f*gradfreqsn[f]*(float)(t-TMIN)
                            /((float)NTNYQ*(float)DTNYQ);
                #if FP16==0
                float amp = DT*src[ssrc*NT+t];
                #elif defined(__OPENCL_VERSION__)
                float amp = ldexp(DT*src[ssrc*NT+t], src_scale - vshift);
                #else
                float amp = scalbnf(DT*src[ssrc*NT+t], src_scale - vshift);
                #endif
                S.x +=  amp*COSPIF(ang);
                S.y += -amp*SINPIF(ang);
                if (t>=TMIN && (t-TMIN)%DTNYQ==0){
                    Sd.x +=  amp*COSPIF(ang);
                    Sd.y += -amp*SINPIF(ang);
                }
            }
        }

        float2 Fxx = fsxx_f[id], Fzz = fszz_f[id], Fxz = fsxz_f[id];
        float2 Axx = fsxx[id],   Azz = fszz[id],   Axz = fsxz[id];
        double Fpr = (double)Fxx.x + (double)Fzz.x, Fpi = (double)Fxx.y + (double)Fzz.y;
        /* savefreqs runs BEFORE the source injection: restore it (FIELD weight). */
        if (sstype==100){
            Fpr += (double)DT*(double)DTNYQ*(double)Sd.x;
            Fpi += (double)DT*(double)DTNYQ*(double)Sd.y;
        }
        double Apr = (double)Axx.x + (double)Azz.x, Api = (double)Axx.y + (double)Azz.y;
        double Fmr = (double)Fxx.x - (double)Fzz.x, Fmi = (double)Fxx.y - (double)Fzz.y;
        double Amr = (double)Axx.x - (double)Azz.x, Ami = (double)Axx.y - (double)Azz.y;

        /* X = (1-u)/DT F - s : the forward increment (minus the source), as in
         * the elastic sdt/s2dt form: (1-u)/DT = s2dt + i sdt. */
        double Xpr = s2dt*Fpr - sdt*Fpi, Xpi = s2dt*Fpi + sdt*Fpr;
        if (sstype==100){ Xpr -= (double)S.x; Xpi -= (double)S.y; }
        double Xmr = s2dt*Fmr - sdt*Fmi, Xmi = s2dt*Fmi + sdt*Fmr;
        double Xxr = s2dt*(double)Fxz.x - sdt*(double)Fxz.y;
        double Xxi = s2dt*(double)Fxz.y + sdt*(double)Fxz.x;

        /* conj(A) X / K^2 for the three modes (complex). */
        double Tpr=0.0, Tpi=0.0, Tmr=0.0, Tmi=0.0, Txr=0.0, Txi=0.0;
        if (live){
            double kr = (2.0*gg-ff) - phr*(2.0*ee-dd), ki = -phi_*(2.0*ee-dd);
            double k2r = kr*kr-ki*ki, k2i = 2.0*kr*ki, k2n = k2r*k2r+k2i*k2i;
            double cr = Apr*Xpr + Api*Xpi, ci = Apr*Xpi - Api*Xpr;   /* conj(A) X */
            Tpr = (cr*k2r + ci*k2i)/k2n; Tpi = (ci*k2r - cr*k2i)/k2n;
            kr = ff - phr*dd; ki = -phi_*dd;
            k2r = kr*kr-ki*ki; k2i = 2.0*kr*ki; k2n = k2r*k2r+k2i*k2i;
            cr = Amr*Xmr + Ami*Xmi; ci = Amr*Xmi - Ami*Xmr;
            Tmr = (cr*k2r + ci*k2i)/k2n; Tmi = (ci*k2r - cr*k2i)/k2n;
        }
        if (livex){
            double kr = fx - phr*dx, ki = -phi_*dx;
            double k2r = kr*kr-ki*ki, k2i = 2.0*kr*ki, k2n = k2r*k2r+k2i*k2i;
            double cr = (double)Axz.x*Xxr + (double)Axz.y*Xxi;
            double ci = (double)Axz.x*Xxi - (double)Axz.y*Xxr;
            Txr = (cr*k2r + ci*k2i)/k2n; Txi = (ci*k2r - cr*k2i)/k2n;
        }
        double nrm = sc_ss/dftnorm;
        /* dK/dg = 2 v1 v1^T, dK/df = -v1 v1^T + v2 v2^T, dK/de = -phi dK/dg, ... */
        double Pg = Tpr;                         /* Re(conj(App) Xpp / Kp^2) */
        double Pf = -0.5*Tpr + 0.5*Tmr;
        double Pe = -(phr*Tpr - phi_*Tpi);       /* Re(-phi T) */
        double Pd = -(phr*(-0.5*Tpr + 0.5*Tmr) - phi_*(-0.5*Tpi + 0.5*Tmi));
        Gg  += -nrm*Pg;
        Gf  += -nrm*Pf;
        Ge  += -nrm*Pe;
        Gd  += -nrm*Pd;
        Gfx += -nrm*Txr;
        Gdx += -nrm*(-(phr*Txr - phi_*Txi));

        /* Velocity block: a FORCE source's eq. (26a) term, mirroring the
         * trace above. savefreqs omits the injection here too, so the forward
         * velocity spectrum gets the source restored with the FIELD weight
         * DT*DTNYQ (kernel_sources() adds `amp` straight into vx/vy/vz, with
         * no material factor), and the eq. (26a) term itself is the
         * rate-weighted rreal. dJ/drho has no c-factor (A1a). Each component
         * feeds its own STAGGERED buoyancy slot, which is what lets
         * average_grad_transpose() apply the averaging Jacobian. */
        float2 Fvx = fvx_f[id], Avx = fvx[id];
        float2 Fvz = fvz_f[id], Avz = fvz[id];
        if (sstype==0){ Fvx.x += (float)((double)DT*(double)DTNYQ)*Sd.x; Fvx.y += (float)((double)DT*(double)DTNYQ)*Sd.y; }
        if (sstype==2){ Fvz.x += (float)((double)DT*(double)DTNYQ)*Sd.x; Fvz.y += (float)((double)DT*(double)DTNYQ)*Sd.y; }
        double d8x = sc_vv*((sdt*itreal(Avx, Fvx) + s2dt*rreal(Avx, Fvx))
                     - (sstype==0 ? rreal(Avx, S) : 0.0))/dftnorm;
        double d8z = sc_vv*((sdt*itreal(Avz, Fvz) + s2dt*rreal(Avz, Fvz))
                     - (sstype==2 ? rreal(Avz, S) : 0.0))/dftnorm;
        Grip += -d8x;
        Grkp += -d8z;
    }

    /* From (g, f, e, d) to the paper's (M, mu, taup, taus): g = M(1+L taup)/ipt,
     * e = M taup/ipt, f = 2 mu (1+L taus)/ist, d = 2 mu taus/ist (M, mu the paper's
     * moduli, ipt = 1 + al taup; chain_rule_par_type() continues from there). */
    double GM    = Gg*(1.0+Ld*taup_p)/ipt + Ge*taup_p/ipt;
    double Gtaup = (Gg*(Ld-al) + Ge)*M_p/(ipt*ipt);
    double Gmu   = 2.0*(Gf*(1.0+Ld*taus_p) + Gd*taus_p)/ist;
    double Gtaus = 2.0*(Gf*(Ld-al) + Gd)*mu_p/(ist*ist);
    double Gmuipkp   = (Gfx*(1.0+Ld*tausipkp_p) + Gdx*tausipkp_p)/istx;
    double Gtausipkp = (Gfx*(Ld-al) + Gdx)*muipkp_p/(istx*istx);

    /* gradrho/gradmu get no shear or velocity correlation at all: those
     * physics act at the staggered rip/rkp and muipkp positions, filled
     * above and folded back onto the cell-centred parameters by
     * average_grad_transpose() -- the same split grad_dft2D.cl already does
     * for its elastic case. */
    gradM[gid]    += (float)GM;
    gradmu[gid]   += (float)Gmu;
    gradtaup[gid] += (float)Gtaup;
    gradtaus[gid] += (float)Gtaus;
    gradmuipkp[gid]   += (float)Gmuipkp;
    gradtausipkp[gid] += (float)Gtausipkp;
    gradrip[gid] += (float)Grip;
    gradrkp[gid] += (float)Grkp;
}
