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

/* Frequency-domain (BACK_PROP_TYPE==2) gradient correlation, 3D,
 * VISCOELASTIC (LVE>0). The 3D sibling of grad_dft2D_visc.cl and the
 * viscoelastic counterpart of grad_dft3D.cl.
 *
 * Without it an L>0 3D run falls back to the host calc_grad(), which is
 * #ifdef __SEISCL__ and a no-op stub in the CUDA build -- a silently zero
 * gradient.
 *
 * Own file, not an #if branch, because prog_args_list() (clprogram.c) parses
 * the raw signature text before preprocessing: arguments behind an #if would
 * still be listed and bound, desynchronising every index after them.
 *
 * Transcribed from calc_grad()'s ND==3, L>0 block. Six stress components and
 * six memory variables where 2D has three and three; the dot[0..8] structure,
 * the grad_coefvisc_1 coefficients (already generic in ND) and the emitted
 * *internal* (M, mu, rho, taup, taus) gradient are identical to the 2D case.
 *
 * Unlike the SH block, and like the 2D P-SV one, every stress component is
 * corrected by its mechanism's memory-variable integral before the
 * elastic-form dot products.
 *
 * Each shear plane (sxy/sxz/syz) is driven by its own staggered mu/taus --
 * muipjp/tausipjp, muipkp/tausipkp, mujpkp/tausjpkp -- not the cell-centred
 * ones, and each velocity component sits at rip/rjp/rkp. Split the same way
 * grad_dft3D.cl splits its elastic shear/velocity terms (and
 * grad_dft2D_visc.cl still does not, for its one shear plane); this is the
 * viscoelastic half of the same material-averaging gap
 * (notes/3d-gradient-findings.md, "Item 6"). average_grad_transpose() folds
 * every staggered gradient here back onto the cell-centred mu/rho/taus.
 */

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/* Scalar padded extents -- NZ/NY/NX are halved on the fastest axis when
 * FP16>0. See grad_dft2D.cl. */
#define NZM (NZS - 2*FDOH)
#define NYM (NYS - 2*FDOH)
#define NXM (NXS - 2*FDOH)
#define NPAD (NXS*NYS*NZS)
#define indf(f,i,j,k) ((f)*NPAD + ((i)+FDOH)*NYS*NZS + ((j)+FDOH)*NZS + ((k)+FDOH))
#define indr(f,l,i,j,k) ((f)*NPAD*LVE + (l)*NPAD + ((i)+FDOH)*NYS*NZS \
                         + ((j)+FDOH)*NZS + ((k)+FDOH))

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

/* calc_grad.c's cl_rm. */
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

/* calc_grad.c's cl_integral: divide by i*w. */
/* calc_grad.c's cl_integral: a/(i*w). Crank-Nicolson's discrete integral is
 * -i*(dt/2)*cot(w*dt/2) -- same form, warped frequency, no real part. See
 * rmreal above. */
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

LFUNDEF float2 add3(float2 a, float2 b, float2 c)
{
    float2 o; o.x = a.x+b.x+c.x; o.y = a.y+b.y+c.y; return o;
}
/* calc_grad.c's cl_dev: the deviatoric combination of the published P4
 * (eq. A2d) and P6 (A2f), (N-1)*a - b - c. The weight on `a` is (N-1) = 2
 * in 3D, not 1; this was `a-b-c` until 2026-09-03, the 2D form transcribed
 * unchanged. See cl_dev()'s comment in calc_grad.c and notes/todo.md item
 * 0i. */
LFUNDEF float2 dev3(float2 a, float2 b, float2 c, double ndm1)
{
    float2 o;
    o.x = (float)(ndm1*a.x) - b.x - c.x;
    o.y = (float)(ndm1*a.y) - b.y - c.y;
    return o;
}
LFUNDEF float2 sub2(float2 a, float2 b)
{
    float2 o; o.x = a.x-b.x; o.y = a.y-b.y; return o;
}

LFUNDEF float2 add2(float2 a, float2 b)
{
    float2 o; o.x = a.x + b.x; o.y = a.y + b.y;
    return o;
}

/* grad_coefvisc_1's c[2]/c[5]/c[10]/c[13] group -- the shear-stress
 * correlation's own coefficients -- depends only on mu and taus, evaluated
 * at whichever staggered position drives that shear plane. See
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

/* Helpers of the exact discrete gradient (see grad_dft2D_visc.cl).
 * phi(w) = 1/2 sum_l b_l eta_l (1+u)/(1 - b_l c_l u), u = exp(-i w DT). */
LFUNDEF void visc_phi(GLOBARG float *FL, double wdt, double *pr, double *pi)
{
    double ur = cos(wdt), ui = -sin(wdt);
    *pr = 0.0; *pi = 0.0;
    for (int l=0; l<LVE; l++){
        double et = 2.0*3.14159265358979323846*(double)FL[l]*(double)DT;
        double bb = 1.0/(1.0+0.5*et), cc = 1.0-0.5*et;
        double nr = 1.0+ur, ni = ui;
        double dr = 1.0-bb*cc*ur, di = -bb*cc*ui;
        double den = dr*dr+di*di;
        *pr += 0.5*bb*et*(nr*dr+ni*di)/den;
        *pi += 0.5*bb*et*(ni*dr-nr*di)/den;
    }
}
/* (cr + i ci)/K^2, K = kr + i ki */
LFUNDEF void divk2(double cr, double ci, double kr, double ki, double *tr, double *ti)
{
    double k2r = kr*kr-ki*ki, k2i = 2.0*kr*ki, k2n = k2r*k2r+k2i*k2i;
    *tr = (cr*k2r + ci*k2i)/k2n;
    *ti = (ci*k2r - cr*k2i)/k2n;
}
/* (1-u)/DT a = (s2dt + i sdt) a */
LFUNDEF void incr(float2 a, double sdt, double s2dt, double *xr, double *xi)
{
    *xr = s2dt*(double)a.x - sdt*(double)a.y;
    *xi = s2dt*(double)a.y + sdt*(double)a.x;
}
/* conj(A) (1-u)/DT F / (fs - phi ds)^2 for one shear stress */
LFUNDEF void shear_term(float2 A, float2 F, double sdt, double s2dt, double fs,
                        double ds, double phr, double phi_, double *tr, double *ti)
{
    *tr = 0.0; *ti = 0.0;
    if (fs < 1.0) return;
    double xr, xi;
    incr(F, sdt, s2dt, &xr, &xi);
    double cr = (double)A.x*xr + (double)A.y*xi, ci = (double)A.x*xi - (double)A.y*xr;
    divk2(cr, ci, fs - phr*ds, -phi_*ds, tr, ti);
}

FUNDEF void calc_grad_dft(GLOBARG float * gradfreqsn,
                          GLOBARG float * FL,
                          GLOBARG PARARG * M,
                          GLOBARG PARARG * mu,
                          GLOBARG PARARG * rho,
                          GLOBARG PARARG * taup,
                          GLOBARG PARARG * taus,
                          GLOBARG PARARG * muipjp,
                          GLOBARG PARARG * muipkp,
                          GLOBARG PARARG * mujpkp,
                          GLOBARG PARARG * tausipjp,
                          GLOBARG PARARG * tausipkp,
                          GLOBARG PARARG * tausjpkp,
                          GLOBARG float * gradM,
                          GLOBARG float * gradmu,
                          GLOBARG float * gradrho,
                          GLOBARG float * gradtaup,
                          GLOBARG float * gradtaus,
                          GLOBARG float * gradmuipjp,
                          GLOBARG float * gradmuipkp,
                          GLOBARG float * gradmujpkp,
                          GLOBARG float * gradrip,
                          GLOBARG float * gradrjp,
                          GLOBARG float * gradrkp,
                          GLOBARG float * gradtausipjp,
                          GLOBARG float * gradtausipkp,
                          GLOBARG float * gradtausjpkp,
                          GLOBARG float2 * fvx_f,
                          GLOBARG float2 * fvy_f,
                          GLOBARG float2 * fvz_f,
                          GLOBARG float2 * fsxx_f,
                          GLOBARG float2 * fsyy_f,
                          GLOBARG float2 * fszz_f,
                          GLOBARG float2 * fsxy_f,
                          GLOBARG float2 * fsxz_f,
                          GLOBARG float2 * fsyz_f,
                          GLOBARG float2 * frxx_f,
                          GLOBARG float2 * fryy_f,
                          GLOBARG float2 * frzz_f,
                          GLOBARG float2 * frxy_f,
                          GLOBARG float2 * frxz_f,
                          GLOBARG float2 * fryz_f,
                          GLOBARG float2 * fvx,
                          GLOBARG float2 * fvy,
                          GLOBARG float2 * fvz,
                          GLOBARG float2 * fsxx,
                          GLOBARG float2 * fsyy,
                          GLOBARG float2 * fszz,
                          GLOBARG float2 * fsxy,
                          GLOBARG float2 * fsxz,
                          GLOBARG float2 * fsyz,
                          GLOBARG float2 * frxx,
                          GLOBARG float2 * fryy,
                          GLOBARG float2 * frzz,
                          GLOBARG float2 * frxy,
                          GLOBARG float2 * frxz,
                          GLOBARG float2 * fryz,
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

    /* X-slowest, Z-fastest, matching grad_dft3D.cl. */
    int i = gid/(NYM*NZM);
    int j = (gid - i*NYM*NZM)/NZM;
    int k = gid - i*NYM*NZM - j*NZM;
    int f, l;

    double s2    = pow(2.0, -(double)par_scale);
    double dhdt  = (double)DH/(double)DT;
    double lrho  = (double)PARCONV(rho[gid]);
    double rho_p = (lrho!=0.0) ? (1.0/lrho)*((double)DT/(double)DH)*s2 : 0.0;
    double M_p   = (double)PARCONV(M[gid])*dhdt*s2;
    double mu_p  = (double)PARCONV(mu[gid])*dhdt*s2;
    double taup_p = (double)PARCONV(taup[gid]);
    double taus_p = (double)PARCONV(taus[gid]);
    /* Each shear plane's own staggered mu/taus -- see the file header. */
    double muipjp_p = (double)PARCONV(muipjp[gid])*dhdt*s2;
    double muipkp_p = (double)PARCONV(muipkp[gid])*dhdt*s2;
    double mujpkp_p = (double)PARCONV(mujpkp[gid])*dhdt*s2;
    double tausipjp_p = (double)PARCONV(tausipjp[gid]);
    double tausipkp_p = (double)PARCONV(tausipkp[gid]);
    double tausjpkp_p = (double)PARCONV(tausjpkp[gid]);

    double dftdf  = 1.0/((double)NTNYQ*(double)DT*(double)DTNYQ);
    double dftnorm = (double)NTNYQ*(double)DTNYQ;

    double sc_ss = pow(2.0, -(double)src_scale - (double)res_scale);
    double sc_vv = sc_ss*pow(2.0, 2.0*(double)par_scale);

    const double NDd = (double)ND;
    const double Ld  = (double)LVE;
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
    muipjp_p *= (1.0 + al*tausipjp_p);
    muipkp_p *= (1.0 + al*tausipkp_p);
    mujpkp_p *= (1.0 + al*tausjpkp_p);

    /* Exact discrete gradient: see grad_dft2D_visc.cl (K(w) = G - phi(w) Ehat).
     * Normal block G = g J - f (J - I), J the all-ones 3x3: trace mode (1,1,1)
     * with Kp = 3g - 2f - phi(3e - 2d), deviatoric modes with Ks = f - phi d. */
    double ipt = 1.0+al*taup_p, ist = 1.0+al*taus_p;
    double gg = M_p*(1.0+Ld*taup_p)/ipt;
    double ff = 2.0*mu_p*(1.0+Ld*taus_p)/ist;
    double ee = M_p*taup_p/ipt;
    double dd = 2.0*mu_p*taus_p/ist;
    double istxy = 1.0+al*tausipjp_p, istxz = 1.0+al*tausipkp_p, istyz = 1.0+al*tausjpkp_p;
    double fxy = muipjp_p*(1.0+Ld*tausipjp_p)/istxy, dxy = muipjp_p*tausipjp_p/istxy;
    double fxz = muipkp_p*(1.0+Ld*tausipkp_p)/istxz, dxz = muipkp_p*tausipkp_p/istxz;
    double fyz = mujpkp_p*(1.0+Ld*tausjpkp_p)/istyz, dyz = mujpkp_p*tausjpkp_p/istyz;
    int live = (rho_p>0.0) && (mu_p>=1.0) && (NDd*gg-(NDd-1.0)*ff>0.0);

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

    double Gg=0.0, Gf=0.0, Ge=0.0, Gd=0.0;
    double Gfxy=0.0, Gdxy=0.0, Gfxz=0.0, Gdxz=0.0, Gfyz=0.0, Gdyz=0.0;
    double Grip=0.0, Grjp=0.0, Grkp=0.0;

    for (f=0; f<NFREQS; f++){

        double w = 2.0*3.14159265358979323846*dftdf*(double)gradfreqsn[f];
        /* (1-u)/DT = s2dt + i sdt, u = exp(-i w DT): the exact discrete
         * equivalent of back_prop_type 1's quadrature (see grad_dft2D.cl). */
        double wdt  = w*(double)DT;
        double sdt  = sin(wdt)/(double)DT;
        double s2dt = 2.0*sin(0.5*wdt)*sin(0.5*wdt)/(double)DT;
        int id = indf(f,i,j,k);
        double phr, phi_;
        visc_phi(FL, wdt, &phr, &phi_);

        /* S(w): transform of the injected source, savefreqs' convention.
         * Sd: the same on the samples savefreqs keeps, to restore the source
         * in a FIELD spectrum (see grad_dft2D_visc.cl). */
        int vshift = (sstype>=0 && sstype<=2) ? par_scale : 0;
        float2 S; S.x = 0.0f; S.y = 0.0f;
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

        /* Forward increments X = (1-u)/DT F (the trace minus the source). */
        double Xxr, Xxi, Xyr, Xyi, Xzr, Xzi;
        incr(fsxx_f[id], sdt, s2dt, &Xxr, &Xxi);
        incr(fsyy_f[id], sdt, s2dt, &Xyr, &Xyi);
        incr(fszz_f[id], sdt, s2dt, &Xzr, &Xzi);
        double Xpr = Xxr+Xyr+Xzr, Xpi = Xxi+Xyi+Xzi;
        if (sstype==100){
            /* savefreqs runs BEFORE the source injection: restore it (FIELD
             * weight DT*DTNYQ), then remove the eq. (26a) source term S. */
            double rr = (double)DT*(double)DTNYQ*(double)Sd.x;
            double ri = (double)DT*(double)DTNYQ*(double)Sd.y;
            Xpr += s2dt*rr - sdt*ri - (double)S.x;
            Xpi += s2dt*ri + sdt*rr - (double)S.y;
        }
        float2 Axx = fsxx[id], Ayy = fsyy[id], Azz = fszz[id];
        double Apr = (double)Axx.x+(double)Ayy.x+(double)Azz.x;
        double Api = (double)Axx.y+(double)Ayy.y+(double)Azz.y;

        double Tpr=0.0, Tpi=0.0, Tmr=0.0, Tmi=0.0;
        if (live){
            /* trace: conj(App) Xpp / Kp^2 */
            double cr = Apr*Xpr + Api*Xpi, ci = Apr*Xpi - Api*Xpr;
            divk2(cr, ci, NDd*gg-(NDd-1.0)*ff - phr*(NDd*ee-(NDd-1.0)*dd),
                  -phi_*(NDd*ee-(NDd-1.0)*dd), &Tpr, &Tpi);
            /* deviatoric: (sum_i conj(Aii) Xii - conj(App) Xpp / ND) / Ks^2.
             * The source term only touches the trace, which cancels here. */
            double sr = (double)Axx.x*Xxr + (double)Axx.y*Xxi
                      + (double)Ayy.x*Xyr + (double)Ayy.y*Xyi
                      + (double)Azz.x*Xzr + (double)Azz.y*Xzi;
            double si = (double)Axx.x*Xxi - (double)Axx.y*Xxr
                      + (double)Ayy.x*Xyi - (double)Ayy.y*Xyr
                      + (double)Azz.x*Xzi - (double)Azz.y*Xzr;
            double Xp0r = Xxr+Xyr+Xzr, Xp0i = Xxi+Xyi+Xzi;   /* without source */
            sr -= (Apr*Xp0r + Api*Xp0i)/NDd;
            si -= (Apr*Xp0i - Api*Xp0r)/NDd;
            divk2(sr, si, ff - phr*dd, -phi_*dd, &Tmr, &Tmi);
        }
        double nrm = sc_ss/dftnorm;
        double Pf_r = -(NDd-1.0)/NDd*Tpr + Tmr, Pf_i = -(NDd-1.0)/NDd*Tpi + Tmi;
        Gg += -nrm*Tpr;
        Gf += -nrm*Pf_r;
        Ge += nrm*(phr*Tpr - phi_*Tpi);
        Gd += nrm*(phr*Pf_r - phi_*Pf_i);

        /* Shear planes, each with its own staggered moduli. */
        double Tr, Ti;
        shear_term(fsxy[id], fsxy_f[id], sdt, s2dt, fxy, dxy, phr, phi_, &Tr, &Ti);
        Gfxy += -nrm*Tr; Gdxy += nrm*(phr*Tr - phi_*Ti);
        shear_term(fsxz[id], fsxz_f[id], sdt, s2dt, fxz, dxz, phr, phi_, &Tr, &Ti);
        Gfxz += -nrm*Tr; Gdxz += nrm*(phr*Tr - phi_*Ti);
        shear_term(fsyz[id], fsyz_f[id], sdt, s2dt, fyz, dyz, phr, phi_, &Tr, &Ti);
        Gfyz += -nrm*Tr; Gdyz += nrm*(phr*Tr - phi_*Ti);

        /* Velocity block: a FORCE source's eq. (26a) term (see grad_dft2D.cl). */
        float2 Fvx = fvx_f[id], Avx = fvx[id];
        float2 Fvy = fvy_f[id], Avy = fvy[id];
        float2 Fvz = fvz_f[id], Avz = fvz[id];
        if (sstype==0){ Fvx.x += (float)((double)DT*(double)DTNYQ)*Sd.x; Fvx.y += (float)((double)DT*(double)DTNYQ)*Sd.y; }
        if (sstype==1){ Fvy.x += (float)((double)DT*(double)DTNYQ)*Sd.x; Fvy.y += (float)((double)DT*(double)DTNYQ)*Sd.y; }
        if (sstype==2){ Fvz.x += (float)((double)DT*(double)DTNYQ)*Sd.x; Fvz.y += (float)((double)DT*(double)DTNYQ)*Sd.y; }
        double d8x = sc_vv*((sdt*itreal(Avx, Fvx) + s2dt*rreal(Avx, Fvx))
                     - (sstype==0 ? rreal(Avx, S) : 0.0))/dftnorm;
        double d8y = sc_vv*((sdt*itreal(Avy, Fvy) + s2dt*rreal(Avy, Fvy))
                     - (sstype==1 ? rreal(Avy, S) : 0.0))/dftnorm;
        double d8z = sc_vv*((sdt*itreal(Avz, Fvz) + s2dt*rreal(Avz, Fvz))
                     - (sstype==2 ? rreal(Avz, S) : 0.0))/dftnorm;
        Grip += -d8x;
        Grjp += -d8y;
        Grkp += -d8z;
    }

    /* From (g, f, e, d) to the paper's (M, mu, taup, taus): see grad_dft2D_visc.cl. */
    double GM    = Gg*(1.0+Ld*taup_p)/ipt + Ge*taup_p/ipt;
    double Gtaup = (Gg*(Ld-al) + Ge)*M_p/(ipt*ipt);
    double Gmu   = 2.0*(Gf*(1.0+Ld*taus_p) + Gd*taus_p)/ist;
    double Gtaus = 2.0*(Gf*(Ld-al) + Gd)*mu_p/(ist*ist);
    double Gmuipjp = (Gfxy*(1.0+Ld*tausipjp_p) + Gdxy*tausipjp_p)/istxy;
    double Gmuipkp = (Gfxz*(1.0+Ld*tausipkp_p) + Gdxz*tausipkp_p)/istxz;
    double Gmujpkp = (Gfyz*(1.0+Ld*tausjpkp_p) + Gdyz*tausjpkp_p)/istyz;
    double Gtausipjp = (Gfxy*(Ld-al) + Gdxy)*muipjp_p/(istxy*istxy);
    double Gtausipkp = (Gfxz*(Ld-al) + Gdxz)*muipkp_p/(istxz*istxz);
    double Gtausjpkp = (Gfyz*(Ld-al) + Gdyz)*mujpkp_p/(istyz*istyz);

    /* gradrho/gradmu get no shear or velocity correlation at all: those
     * physics act at the staggered rip/rjp/rkp and muipjp/muipkp/mujpkp
     * positions, filled above and folded back onto the cell-centred
     * parameters by average_grad_transpose(). Same split grad_dft3D.cl and
     * grad_dft2D.cl already do. */
    gradM[gid]    += (float)GM;
    gradmu[gid]   += (float)Gmu;
    gradtaup[gid] += (float)Gtaup;
    gradtaus[gid] += (float)Gtaus;
    gradmuipjp[gid] += (float)Gmuipjp;
    gradmuipkp[gid] += (float)Gmuipkp;
    gradmujpkp[gid] += (float)Gmujpkp;
    gradtausipjp[gid] += (float)Gtausipjp;
    gradtausipkp[gid] += (float)Gtausipkp;
    gradtausjpkp[gid] += (float)Gtausjpkp;
    gradrip[gid]  += (float)Grip;
    gradrjp[gid]  += (float)Grjp;
    gradrkp[gid]  += (float)Grkp;
}
