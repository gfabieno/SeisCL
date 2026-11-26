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

/* Frequency-domain (back_prop_type 2) gradient correlation, 2D SH,
 * VISCOELASTIC (LVE > 0).
 *
 * Built from grad_dft3D_visc.cl by dropping every term that is zero by
 * symmetry (y-invariance, SH polarisation: vx = vz = 0; sxx, syy, szz, sxz and
 * their memory variables are 0; notes sh-gradient.md, step 2). What remains is
 * grad_dft3D_visc.cl's
 *   Gmuipjp   += -c2xy d2xy + c5xy d5xy,    Gtausipjp += -c10xy d2xy + c13xy d5xy,
 *   Gmujpkp   += -c2yz d2yz + c5yz d5yz,    Gtausjpkp += -c10yz d2yz + c13yz d5yz,
 *   Grjp      += -d8y  (-> gradrho: rjp = rho when nothing varies along y),
 * with the same exact discrete correlation (K(w) = f - phi(w) d per shear
 * stress, end-of-record terms; see grad_dft2D_visc.cl) and the same DFT
 * conventions. GM, Gmu, Gtaup, Gtaus (normal stresses) vanish.
 * Replaces the earlier grad_dft2D_SH_visc.cl (from the removed host
 * calc_grad(), raw stresses, last mechanism only, never run).
 */

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/* Scalar padded extents (see grad_dft3D.cl). */
#define NZM (NZS - 2*FDOH)
#define NXM (NXS - 2*FDOH)
#define NPAD (NXS*NZS)
#define indf(f,i,k) ((f)*NPAD + ((i)+FDOH)*NZS + ((k)+FDOH))
#define indr(f,l,i,k) ((f)*NPAD*LVE + (l)*NPAD + ((i)+FDOH)*NZS + ((k)+FDOH))

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

/* itreal(a,b) = Re(conj(a)*i*b) (<s~, dt s>); rreal(a,b) = Re(conj(a)*b). */
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

FUNDEF void calc_grad_dft(GLOBARG float * gradfreqsn,
                          GLOBARG float * FL,
                          GLOBARG PARARG * rho,
                          GLOBARG PARARG * muipjp,
                          GLOBARG PARARG * mujpkp,
                          GLOBARG PARARG * tausipjp,
                          GLOBARG PARARG * tausjpkp,
                          GLOBARG float * gradrho,
                          GLOBARG float * gradmuipjp,
                          GLOBARG float * gradmujpkp,
                          GLOBARG float * gradtausipjp,
                          GLOBARG float * gradtausjpkp,
                          GLOBARG float2 * fvy_f,
                          GLOBARG float2 * fsxy_f,
                          GLOBARG float2 * fsyz_f,
                          GLOBARG float2 * frxy_f,
                          GLOBARG float2 * fryz_f,
                          GLOBARG float2 * fvy,
                          GLOBARG float2 * fsxy,
                          GLOBARG float2 * fsyz,
                          GLOBARG float2 * frxy,
                          GLOBARG float2 * fryz,
                          GLOBARG PARARG * sxy_e,
                          GLOBARG PARARG * syz_e,
                          GLOBARG PARARG * rxy_e,
                          GLOBARG PARARG * ryz_e,
                          GLOBARG PARARG * sxy,
                          GLOBARG PARARG * syz,
                          GLOBARG PARARG * rxy,
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

    if (gid >= NXM*NZM)
        return;

    /* X-slowest, Z-fastest, as the parameter arrays. */
    int i = gid/NZM;
    int k = gid - i*NZM;
    int f, l;

    double s2    = pow(2.0, -(double)par_scale);
    double dhdt  = (double)DH/(double)DT;
    double muipjp_p = (double)PARCONV(muipjp[gid])*dhdt*s2;
    double mujpkp_p = (double)PARCONV(mujpkp[gid])*dhdt*s2;
    double tausipjp_p = (double)PARCONV(tausipjp[gid]);
    double tausjpkp_p = (double)PARCONV(tausjpkp[gid]);
    double dftdf  = 1.0/((double)NTNYQ*(double)DT*(double)DTNYQ);
    double dftnorm = (double)NTNYQ*(double)DTNYQ;
    double sc_ss = pow(2.0, -(double)src_scale - (double)res_scale);
    double sc_vv = sc_ss*pow(2.0, 2.0*(double)par_scale);
    const double Ld  = (double)LVE;
    /* GSLS phase-velocity normalisation al (see grad_dft3D_visc.cl). */
    double al = 0.0;
    for (l=0; l<LVE; l++){
        double r = (double)FREQ0/(double)FL[l];
        al += r*r/(1.0 + r*r);
    }
    /* The paper's moduli at each node (grad_dft3D_visc.cl). */
    muipjp_p *= (1.0 + al*tausipjp_p);
    mujpkp_p *= (1.0 + al*tausjpkp_p);
    /* Exact discrete gradient (grad_dft2D_visc.cl): K(w) = f - phi(w) d per shear
     * stress, f = mu(1+L taus)/ist, d = mu taus/ist (paper's mu, ist = 1+al taus). */
    double istxy = 1.0+al*tausipjp_p, istyz = 1.0+al*tausjpkp_p;
    double fxy = muipjp_p*(1.0+Ld*tausipjp_p)/istxy, dxy = muipjp_p*tausipjp_p/istxy;
    double fyz = mujpkp_p*(1.0+Ld*tausjpkp_p)/istyz, dyz = mujpkp_p*tausjpkp_p/istyz;
    double Gfxy=0.0, Gdxy=0.0, Gfyz=0.0, Gdyz=0.0;

    /* End-of-record terms (grad_dft2D_visc.cl, end_adj/end_freq), per shear
     * stress c = xy, yz. */
    double se[2], re[2][LVE], be[2][LVE], bE[2][LVE];
    {
        int i0 = indf(0,i,k);
        double ra[2][LVE];
        se[0] = (double)PARCONV(sxy_e[i0]); se[1] = (double)PARCONV(syz_e[i0]);
        for (l=0; l<LVE; l++){
            int il = indr(0,l,i,k);
            re[0][l] = (double)PARCONV(rxy_e[il]); ra[0][l] = (double)PARCONV(rxy[il]);
            re[1][l] = (double)PARCONV(ryz_e[il]); ra[1][l] = (double)PARCONV(ryz[il]);
        }
        end_adj(FL, (double)PARCONV(sxy[i0]), ra[0], fxy - Ld*dxy, dxy, be[0], bE[0]);
        end_adj(FL, (double)PARCONV(syz[i0]), ra[1], fyz - Ld*dyz, dyz, be[1], bE[1]);
    }
    double nwdt0 = (double)DT*(double)(NT - TMIN);

    /* Which source, if any, sits in this cell (eq. 26a source term). In SH the
     * only source acting on these fields is the vy force (code 1). */
    int ssrc = -1, sstype = -1;
    for (int q=0; q<nsrc; q++){
        int st = (int)src_pos[4+5*q];
        if (st==1
            && (int)(src_pos[0+5*q]/DH)==i+OFFSET   /* global x (domain decomposition) */
            && (int)(src_pos[2+5*q]/DH)==k){
            ssrc = q; sstype = st; break;
        }
    }

    double Gmuipjp=0.0, Gmujpkp=0.0, Gtausipjp=0.0, Gtausjpkp=0.0, Grho=0.0;

    for (f=0; f<NFREQS; f++){

        double w = 2.0*3.14159265358979323846*dftdf*(double)gradfreqsn[f];
        /* Exact discrete equivalent of back_prop_type 1's quadrature: see
         * grad_dft3D.cl. */
        double wdt  = w*(double)DT;
        double sdt  = sin(wdt)/(double)DT;
        double s2dt = 2.0*sin(0.5*wdt)*sin(0.5*wdt)/(double)DT;
        int id = indf(f,i,k);
        /* Each bin stands for the pair (+f, -f); bin 0 and the Nyquist bin are
         * their own pair and count half. Bin 0 is allowed: no 1/w remains. */
        double dnf = dftnorm/(((int)gradfreqsn[f]==0 || 2*(int)gradfreqsn[f]==NTNYQ) ? 0.5 : 1.0);

        /* S(w), Sd(w): see grad_dft3D.cl. A force source is stored in units
         * 2^(src_scale - par_scale). */
        int vshift = (sstype==1) ? par_scale : 0;
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

        double phr, phi_, Tr, Ti, nrm = sc_ss/dnf;
        visc_phi(FL, wdt, &phr, &phi_);
        double Ur;
        for (int c=0; c<2; c++){
            float2 A = (c==0) ? fsxy[id] : fsyz[id];
            double xr, xi, yr, yi, bfr, bfi, bar, bai;
            incr((c==0) ? fsxy_f[id] : fsyz_f[id], sdt, s2dt, &xr, &xi);
            end_freq(FL, wdt, w*nwdt0, se[c], re[c], be[c], bE[c],
                     &bfr, &bfi, &bar, &bai, &yr, &yi);
            shear_term_e((double)A.x - bar, (double)A.y - bai, xr + bfr, xi + bfi,
                         yr, yi, (c==0) ? fxy : fyz, (c==0) ? dxy : dyz,
                         phr, phi_, &Tr, &Ti, &Ur);
            if (c==0){ Gfxy += -nrm*Tr; Gdxy += nrm*(phr*Tr - phi_*Ti) - nrm*Ur; }
            else     { Gfyz += -nrm*Tr; Gdyz += nrm*(phr*Tr - phi_*Ti) - nrm*Ur; }
        }

        /* Velocity block (3-D d8y): vy, its force source restored with the
         * FIELD weight DT*DTNYQ, and the eq. (26a) term. */
        float2 Fvy = fvy_f[id], Avy = fvy[id];
        if (sstype==1){ Fvy.x += (float)((double)DT*(double)DTNYQ)*Sd.x; Fvy.y += (float)((double)DT*(double)DTNYQ)*Sd.y; }
        double d8y = sc_vv*((sdt*itreal(Avy, Fvy) + s2dt*rreal(Avy, Fvy))
                     - (sstype==1 ? rreal(Avy, S) : 0.0))/dnf;

        Grho      += -d8y;
    }

    Gmuipjp = (Gfxy*(1.0+Ld*tausipjp_p) + Gdxy*tausipjp_p)/istxy;
    Gmujpkp = (Gfyz*(1.0+Ld*tausjpkp_p) + Gdyz*tausjpkp_p)/istyz;
    Gtausipjp = (Gfxy*(Ld-al) + Gdxy)*muipjp_p/(istxy*istxy);
    Gtausjpkp = (Gfyz*(Ld-al) + Gdyz)*mujpkp_p/(istyz*istyz);
    gradmuipjp[gid]   += (float)Gmuipjp;
    gradmujpkp[gid]   += (float)Gmujpkp;
    gradtausipjp[gid] += (float)Gtausipjp;
    gradtausjpkp[gid] += (float)Gtausjpkp;
    gradrho[gid]      += (float)Grho;
}
