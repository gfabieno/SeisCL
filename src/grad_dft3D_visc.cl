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
/* One shear stress, from the end-corrected spectra: T = conj(A) X / K^2 and
 * the end-of-record memory term U = DT Re(Y X / K), K = fs - phi ds. */
LFUNDEF void shear_term_e(double ar, double ai, double xr, double xi,
                          double yr, double yi, double fs, double ds,
                          double phr, double phi_, double *tr, double *ti, double *ur)
{
    *tr = 0.0; *ti = 0.0; *ur = 0.0;
    if (fs < 1.0) return;
    double kr = fs - phr*ds, ki = -phi_*ds;
    double cr = ar*xr + ai*xi, ci = ar*xi - ai*xr;
    divk2(cr, ci, kr, ki, tr, ti);
    double pr = yr*xr - yi*xi, pi = yr*xi + yi*xr;
    *ur = (double)DT*(pr*kr + pi*ki)/(kr*kr + ki*ki);
}

/* End-of-record terms of the exact discrete correlation (notes
 * viscoelastic-gradient-theory.md, 2026-11-20; per-cell check
 * fd-invest/exact/cell_check.py, exact to round-off).
 *
 * K(w)^-1 is a recursive (relaxation) filter. The DFT of a record that ends
 * with waves still in the grid is not K^-1 times the DFT of the increments:
 * the forward memory variables carry the end of the record round onto its
 * start, and the adjoint ones carry its start onto its end. Per mode m (scalar
 * moduli: unrelaxed Gm, memory Eh = E*DT, relaxed K0 = Gm - L Eh), with
 * a = b c, u = exp(-i w DT), z = conj(u), N = NT - TMIN:
 *   forward:  u K Q = X + u^N [ s_e + sum_l DT2 u (1+a)/(1 - a u) r_e ]
 *             (s_e, r_e: forward state after the last update, saved in
 *             <var>_e before the adjoint run)
 *   adjoint:  conj(K) Lam = A/DT - sum_l beta_l/(1 - a z),
 *             beta_l = a DT2 (1+a)/(b eta) r~_l + b Eh lam1,
 *             lam1 = [ s~ + sum_l DT2 (1+a)/(b eta) r~_l ] / K0
 *             (s~, r~: adjoint state at t = TMIN+1, still in the buffers)
 *   and the memory-modulus gradient gains
 *             Re[ sum_l (beta_l/Eh)/(1 - a u) dEh u Q ].
 * end_adj() builds beta_l and beta_l/Eh once per cell and mode; end_freq()
 * gives, per frequency, the forward term to add to X, DT x the adjoint term to
 * subtract from A, and Y = sum_l (beta_l/Eh)/(1 - a u). */
LFUNDEF void end_adj(GLOBARG float *FL, double sa, const double *ra,
                     double K0, double Eh, double *bet, double *betE)
{
    double v = sa;
    for (int l=0; l<LVE; l++){
        double et = 2.0*3.14159265358979323846*(double)FL[l]*(double)DT;
        double bb = 1.0/(1.0+0.5*et), aa = bb*(1.0-0.5*et);
        v += 0.5*(double)DT*(1.0+aa)/(bb*et)*ra[l];
    }
    double lam1 = (K0!=0.0) ? v/K0 : 0.0;
    for (int l=0; l<LVE; l++){
        double et = 2.0*3.14159265358979323846*(double)FL[l]*(double)DT;
        double bb = 1.0/(1.0+0.5*et), aa = bb*(1.0-0.5*et);
        double c1 = aa*0.5*(double)DT*(1.0+aa)/(bb*et);
        bet[l]  = c1*ra[l] + bb*Eh*lam1;
        betE[l] = ((Eh!=0.0) ? c1*ra[l]/Eh : 0.0) + bb*lam1;
    }
}

LFUNDEF void end_freq(GLOBARG float *FL, double wdt, double nwdt,
                      double se, const double *re,
                      const double *bet, const double *betE,
                      double *bfr, double *bfi, double *bar, double *bai,
                      double *yr, double *yi)
{
    double ur = cos(wdt), ui = -sin(wdt);
    double sr = se, si = 0.0;
    double ar = 0.0, ai = 0.0, wr = 0.0, wi = 0.0;
    for (int l=0; l<LVE; l++){
        double et = 2.0*3.14159265358979323846*(double)FL[l]*(double)DT;
        double bb = 1.0/(1.0+0.5*et), aa = bb*(1.0-0.5*et);
        double dr = 1.0-aa*ur, di = -aa*ui, dn = dr*dr+di*di;
        double ir = dr/dn, ii = -di/dn;                    /* 1/(1 - a u) */
        double c = 0.5*(double)DT*(1.0+aa)*re[l];
        sr += c*(ur*ir - ui*ii); si += c*(ur*ii + ui*ir);  /* DT2 u(1+a)/(1-au) r_e */
        ar += bet[l]*ir;  ai -= bet[l]*ii;                 /* beta/(1 - a z) */
        wr += betE[l]*ir; wi += betE[l]*ii;
    }
    double nr = cos(nwdt), ni = -sin(nwdt);                /* u^N */
    *bfr = nr*sr - ni*si; *bfi = nr*si + ni*sr;
    *bar = (double)DT*ar; *bai = (double)DT*ai;
    *yr = wr; *yi = wi;
}

/* Re(y x / k), complex y, x, k. */
LFUNDEF double re_yxk(double yr, double yi, double xr, double xi, double kr, double ki)
{
    double pr = yr*xr - yi*xi, pi = yr*xi + yi*xr;
    return (pr*kr + pi*ki)/(kr*kr + ki*ki);
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
                          GLOBARG PARARG * sxx_e,
                          GLOBARG PARARG * syy_e,
                          GLOBARG PARARG * szz_e,
                          GLOBARG PARARG * sxy_e,
                          GLOBARG PARARG * sxz_e,
                          GLOBARG PARARG * syz_e,
                          GLOBARG PARARG * rxx_e,
                          GLOBARG PARARG * ryy_e,
                          GLOBARG PARARG * rzz_e,
                          GLOBARG PARARG * rxy_e,
                          GLOBARG PARARG * rxz_e,
                          GLOBARG PARARG * ryz_e,
                          GLOBARG PARARG * sxx,
                          GLOBARG PARARG * syy,
                          GLOBARG PARARG * szz,
                          GLOBARG PARARG * sxy,
                          GLOBARG PARARG * sxz,
                          GLOBARG PARARG * syz,
                          GLOBARG PARARG * rxx,
                          GLOBARG PARARG * ryy,
                          GLOBARG PARARG * rzz,
                          GLOBARG PARARG * rxy,
                          GLOBARG PARARG * rxz,
                          GLOBARG PARARG * ryz,
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

    /* End-of-record terms (grad_dft2D_visc.cl, end_adj/end_freq), per stress
     * component c = xx, yy, zz, xy, xz, yz. The forward term is component-wise
     * (no moduli). The adjoint one needs K0^-1 and Eh^-1, which act per mode:
     * the normal components are split into their mean (trace mode: K0p, Ehp)
     * and their deviation (deviatoric: K0s, Ehs); end_adj is linear in its
     * state inputs, so the two parts add. */
    double se[6], re[6][LVE], be[6][LVE], bE[6][LVE];
    {
        int i0 = indf(0,i,j,k);
        double sa[6], ra[6][LVE];
        se[0] = (double)PARCONV(sxx_e[i0]); sa[0] = (double)PARCONV(sxx[i0]);
        se[1] = (double)PARCONV(syy_e[i0]); sa[1] = (double)PARCONV(syy[i0]);
        se[2] = (double)PARCONV(szz_e[i0]); sa[2] = (double)PARCONV(szz[i0]);
        se[3] = (double)PARCONV(sxy_e[i0]); sa[3] = (double)PARCONV(sxy[i0]);
        se[4] = (double)PARCONV(sxz_e[i0]); sa[4] = (double)PARCONV(sxz[i0]);
        se[5] = (double)PARCONV(syz_e[i0]); sa[5] = (double)PARCONV(syz[i0]);
        for (l=0; l<LVE; l++){
            int il = indr(0,l,i,j,k);
            re[0][l] = (double)PARCONV(rxx_e[il]); ra[0][l] = (double)PARCONV(rxx[il]);
            re[1][l] = (double)PARCONV(ryy_e[il]); ra[1][l] = (double)PARCONV(ryy[il]);
            re[2][l] = (double)PARCONV(rzz_e[il]); ra[2][l] = (double)PARCONV(rzz[il]);
            re[3][l] = (double)PARCONV(rxy_e[il]); ra[3][l] = (double)PARCONV(rxy[il]);
            re[4][l] = (double)PARCONV(rxz_e[il]); ra[4][l] = (double)PARCONV(rxz[il]);
            re[5][l] = (double)PARCONV(ryz_e[il]); ra[5][l] = (double)PARCONV(ryz[il]);
        }
        double Ehp = NDd*ee-(NDd-1.0)*dd, K0p = NDd*gg-(NDd-1.0)*ff - Ld*Ehp;
        double Ehs = dd, K0s = ff - Ld*dd;
        double sm = (sa[0]+sa[1]+sa[2])/NDd, rm[LVE], bp[LVE], bEp[LVE], rd[LVE];
        for (l=0; l<LVE; l++) rm[l] = (ra[0][l]+ra[1][l]+ra[2][l])/NDd;
        end_adj(FL, sm, rm, K0p, Ehp, bp, bEp);
        for (int c=0; c<3; c++){
            for (l=0; l<LVE; l++) rd[l] = ra[c][l] - rm[l];
            end_adj(FL, sa[c]-sm, rd, K0s, Ehs, be[c], bE[c]);
            for (l=0; l<LVE; l++){ be[c][l] += bp[l]; bE[c][l] += bEp[l]; }
        }
        end_adj(FL, sa[3], ra[3], fxy - Ld*dxy, dxy, be[3], bE[3]);
        end_adj(FL, sa[4], ra[4], fxz - Ld*dxz, dxz, be[4], bE[4]);
        end_adj(FL, sa[5], ra[5], fyz - Ld*dyz, dyz, be[5], bE[5]);
    }
    double nwdt0 = (double)DT*(double)(NT - TMIN);

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
        /* Each bin stands for the pair (+f, -f); bin 0 and the Nyquist bin are
         * their own pair and count half. Bin 0 is allowed: no 1/w remains. */
        double dnf = dftnorm/(((int)gradfreqsn[f]==0 || 2*(int)gradfreqsn[f]==NTNYQ) ? 0.5 : 1.0);
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
        /* End-of-record terms: X_c += forward term, A_c -= adjoint term. */
        double Yr[6], Yi[6], Acr[6], Aci[6], Xcr[6], Xci[6];
        {
            float2 Ac[6];
            Ac[0] = fsxx[id]; Ac[1] = fsyy[id]; Ac[2] = fszz[id];
            Ac[3] = fsxy[id]; Ac[4] = fsxz[id]; Ac[5] = fsyz[id];
            incr(fsxy_f[id], sdt, s2dt, &Xcr[3], &Xci[3]);
            incr(fsxz_f[id], sdt, s2dt, &Xcr[4], &Xci[4]);
            incr(fsyz_f[id], sdt, s2dt, &Xcr[5], &Xci[5]);
            Xcr[0] = Xxr; Xci[0] = Xxi; Xcr[1] = Xyr; Xci[1] = Xyi; Xcr[2] = Xzr; Xci[2] = Xzi;
            for (int c=0; c<6; c++){
                double bfr, bfi, bar, bai;
                end_freq(FL, wdt, w*nwdt0, se[c], re[c], be[c], bE[c],
                         &bfr, &bfi, &bar, &bai, &Yr[c], &Yi[c]);
                Xcr[c] += bfr; Xci[c] += bfi;
                Acr[c] = (double)Ac[c].x - bar; Aci[c] = (double)Ac[c].y - bai;
            }
            Xxr = Xcr[0]; Xxi = Xci[0]; Xyr = Xcr[1]; Xyi = Xci[1]; Xzr = Xcr[2]; Xzi = Xci[2];
        }
        double Xpr = Xxr+Xyr+Xzr, Xpi = Xxi+Xyi+Xzi;
        if (sstype==100){
            /* savefreqs runs BEFORE the source injection: restore it (FIELD
             * weight DT*DTNYQ), then remove the eq. (26a) source term S. */
            double rr = (double)DT*(double)DTNYQ*(double)Sd.x;
            double ri = (double)DT*(double)DTNYQ*(double)Sd.y;
            Xpr += s2dt*rr - sdt*ri - (double)S.x;
            Xpi += s2dt*ri + sdt*rr - (double)S.y;
        }
        double Apr = Acr[0]+Acr[1]+Acr[2];
        double Api = Aci[0]+Aci[1]+Aci[2];

        double Tpr=0.0, Tpi=0.0, Tmr=0.0, Tmi=0.0, Upr=0.0, Umr=0.0;
        if (live){
            /* trace: conj(App) Xpp / Kp^2 */
            double cr = Apr*Xpr + Api*Xpi, ci = Apr*Xpi - Api*Xpr;
            divk2(cr, ci, NDd*gg-(NDd-1.0)*ff - phr*(NDd*ee-(NDd-1.0)*dd),
                  -phi_*(NDd*ee-(NDd-1.0)*dd), &Tpr, &Tpi);
            /* deviatoric: (sum_i conj(Aii) Xii - conj(App) Xpp / ND) / Ks^2.
             * The source term only touches the trace, which cancels here. */
            double sr = Acr[0]*Xxr + Aci[0]*Xxi
                      + Acr[1]*Xyr + Aci[1]*Xyi
                      + Acr[2]*Xzr + Aci[2]*Xzi;
            double si = Acr[0]*Xxi - Aci[0]*Xxr
                      + Acr[1]*Xyi - Aci[1]*Xyr
                      + Acr[2]*Xzi - Aci[2]*Xzr;
            double Xp0r = Xxr+Xyr+Xzr, Xp0i = Xxi+Xyi+Xzi;   /* without source */
            sr -= (Apr*Xp0r + Api*Xp0i)/NDd;
            si -= (Apr*Xp0i - Api*Xp0r)/NDd;
            divk2(sr, si, ff - phr*dd, -phi_*dd, &Tmr, &Tmi);
            /* End-of-record memory-modulus term, same mode split:
             * Up = DT Re(Ys Xpp / Kp), Um = DT Re[(sum_c Yc Xc - Ys Xp0/ND)/Ks]. */
            double Ysr = Yr[0]+Yr[1]+Yr[2], Ysi = Yi[0]+Yi[1]+Yi[2];
            Upr = (double)DT*re_yxk(Ysr, Ysi, Xpr, Xpi,
                      NDd*gg-(NDd-1.0)*ff - phr*(NDd*ee-(NDd-1.0)*dd),
                      -phi_*(NDd*ee-(NDd-1.0)*dd));
            double yr = 0.0, yi = 0.0;
            for (int c=0; c<3; c++){
                yr += Yr[c]*Xcr[c] - Yi[c]*Xci[c];
                yi += Yr[c]*Xci[c] + Yi[c]*Xcr[c];
            }
            yr -= (Ysr*Xp0r - Ysi*Xp0i)/NDd;
            yi -= (Ysr*Xp0i + Ysi*Xp0r)/NDd;
            Umr = (double)DT*re_yxk(yr, yi, 1.0, 0.0, ff - phr*dd, -phi_*dd);
        }
        double nrm = sc_ss/dnf;
        double Pf_r = -(NDd-1.0)/NDd*Tpr + Tmr, Pf_i = -(NDd-1.0)/NDd*Tpi + Tmi;
        Gg += -nrm*Tpr;
        Gf += -nrm*Pf_r;
        Ge += nrm*(phr*Tpr - phi_*Tpi) - nrm*Upr;
        Gd += nrm*(phr*Pf_r - phi_*Pf_i) - nrm*(-(NDd-1.0)/NDd*Upr + Umr);

        /* Shear planes, each with its own staggered moduli. */
        double Tr, Ti;
        double Ur;
        shear_term_e(Acr[3], Aci[3], Xcr[3], Xci[3], Yr[3], Yi[3], fxy, dxy, phr, phi_, &Tr, &Ti, &Ur);
        Gfxy += -nrm*Tr; Gdxy += nrm*(phr*Tr - phi_*Ti) - nrm*Ur;
        shear_term_e(Acr[4], Aci[4], Xcr[4], Xci[4], Yr[4], Yi[4], fxz, dxz, phr, phi_, &Tr, &Ti, &Ur);
        Gfxz += -nrm*Tr; Gdxz += nrm*(phr*Tr - phi_*Ti) - nrm*Ur;
        shear_term_e(Acr[5], Aci[5], Xcr[5], Xci[5], Yr[5], Yi[5], fyz, dyz, phr, phi_, &Tr, &Ti, &Ur);
        Gfyz += -nrm*Tr; Gdyz += nrm*(phr*Tr - phi_*Ti) - nrm*Ur;

        /* Velocity block: a FORCE source's eq. (26a) term (see grad_dft2D.cl). */
        float2 Fvx = fvx_f[id], Avx = fvx[id];
        float2 Fvy = fvy_f[id], Avy = fvy[id];
        float2 Fvz = fvz_f[id], Avz = fvz[id];
        if (sstype==0){ Fvx.x += (float)((double)DT*(double)DTNYQ)*Sd.x; Fvx.y += (float)((double)DT*(double)DTNYQ)*Sd.y; }
        if (sstype==1){ Fvy.x += (float)((double)DT*(double)DTNYQ)*Sd.x; Fvy.y += (float)((double)DT*(double)DTNYQ)*Sd.y; }
        if (sstype==2){ Fvz.x += (float)((double)DT*(double)DTNYQ)*Sd.x; Fvz.y += (float)((double)DT*(double)DTNYQ)*Sd.y; }
        double d8x = sc_vv*((sdt*itreal(Avx, Fvx) + s2dt*rreal(Avx, Fvx))
                     - (sstype==0 ? rreal(Avx, S) : 0.0))/dnf;
        double d8y = sc_vv*((sdt*itreal(Avy, Fvy) + s2dt*rreal(Avy, Fvy))
                     - (sstype==1 ? rreal(Avy, S) : 0.0))/dnf;
        double d8z = sc_vv*((sdt*itreal(Avz, Fvz) + s2dt*rreal(Avz, Fvz))
                     - (sstype==2 ? rreal(Avz, S) : 0.0))/dnf;
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
