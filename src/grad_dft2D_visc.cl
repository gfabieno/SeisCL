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

LFUNDEF double itreal(float2 a, float2 b)
{
    return (double)a.y*(double)b.x - (double)a.x*(double)b.y;
}

/* calc_grad.c's cl_rm: the memory-variable correlation. */
LFUNDEF double rmreal(float2 a, float2 b, double tausig, double w)
{
    return tausig*((double)a.x*(double)b.x + (double)a.y*(double)b.y)
         + ((double)a.x*(double)b.y - (double)a.y*(double)b.x)/w;
}

/* calc_grad.c's cl_integral: divide the spectrum by i*w. */
LFUNDEF float2 integ(float2 a, double w)
{
    float2 o;
    o.x = (float)((double)a.y/w);
    o.y = (float)(-(double)a.x/w);
    return o;
}

FUNDEF void calc_grad_dft(GLOBARG float * gradfreqsn,
                          GLOBARG float * FL,
                          GLOBARG PARARG * M,
                          GLOBARG PARARG * mu,
                          GLOBARG PARARG * rho,
                          GLOBARG PARARG * taup,
                          GLOBARG PARARG * taus,
                          GLOBARG float * gradM,
                          GLOBARG float * gradmu,
                          GLOBARG float * gradrho,
                          GLOBARG float * gradtaup,
                          GLOBARG float * gradtaus,
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

    /* grad_coefvisc_1: the *internal* (M, mu, taup, taus) coefficients, i.e.
     * grad_coefvisc_0 with the parameterization chain rule factored out.
     * c[16..23] do not exist here -- chain_rule_par_type() derives the density
     * contribution from gradM/gradmu instead. */
    double f1 = NDd*M_p*(1.0+Ld*taup_p)*(1.0+al*taus_p)
              - 2.0*(NDd-1.0)*mu_p*(1.0+Ld*taus_p)*(1.0+al*taup_p);
    double f2 = NDd*M_p*taup_p*(1.0+al*taus_p)
              - 2.0*(NDd-1.0)*mu_p*taus_p*(1.0+al*taup_p);
    double fact1 = f1*f1;
    double fact2 = f2*f2;

    double c0=0.0,c1=0.0,c2=0.0,c3=0.0,c4=0.0,c5=0.0,c6=0.0,c7=0.0;
    double c8=0.0,c9=0.0,c10=0.0,c11=0.0,c12=0.0,c13=0.0,c14=0.0,c15=0.0;

    int live = (rho_p>0.0) && (fact1>0.0) && (fact2>0.0) && (mu_p>=1.0)
               && (taus_p!=0.0) && (taup_p!=0.0);
    if (live){
        double ipt = 1.0+al*taup_p;
        double ist = 1.0+al*taus_p;
        double Lst = 1.0+Ld*taus_p;
        double imu2 = 1.0/(mu_p*mu_p);

        c0  = (1.0+Ld*taup_p)*ipt*ist*ist/fact1;
        c1  = taup_p*ipt*ist*ist/fact2;
        c2  = ist*imu2/Lst;
        c3  = (NDd+1.0)/3.0*Lst*ist*ipt*ipt/fact1;
        c4  = ist*imu2/(2.0*NDd*Lst);
        c5  = ist*imu2/taus_p;
        c6  = (NDd+1.0)/3.0*taus_p*ist*ipt*ipt/fact2;
        c7  = ist*imu2/(2.0*NDd*taus_p);

        c8  = M_p*(Ld-al)*ist*ist/fact1;
        c9  = M_p*ist*ist/fact2;
        c10 = (Ld-al)/(mu_p*Lst*Lst);
        c11 = (NDd+1.0)/3.0*mu_p*(Ld-al)*ipt*ipt/fact1;
        c12 = (Ld-al)/(2.0*NDd*mu_p*Lst*Lst);
        c13 = 1.0/(mu_p*taus_p*taus_p);
        c14 = (NDd+1.0)/3.0*mu_p*ipt*ipt/fact2;
        c15 = 1.0/(2.0*NDd*mu_p*taus_p*taus_p);

    }

    double GM=0.0, Gmu=0.0, Gtaup=0.0, Gtaus=0.0, Grho=0.0;

    for (f=0; f<NFREQS; f++){

        double w = 2.0*3.14159265358979323846*dftdf*(double)gradfreqsn[f];
        int id = indf(f,i,k);

        float2 Fxx = fsxx_f[id], Fzz = fszz_f[id], Fxz = fsxz_f[id];
        float2 Axx = fsxx[id],   Azz = fszz[id],   Axz = fsxz[id];

        double d1=0.0, d5=0.0, d7=0.0;

        for (l=0; l<LVE; l++){
            int il = indr(f,l,i,k);
            /* tausig = 1/(2*pi*FL[l]), calc_grad.c's tausigl. */
            double tausig = 1.0/(2.0*3.14159265358979323846*(double)FL[l]);

            float2 Rxx_f = frxx_f[il], Rzz_f = frzz_f[il], Rxz_f = frxz_f[il];
            float2 Rxx_a = frxx[il],   Rzz_a = frzz[il],   Rxz_a = frxz[il];

            /* The stored stress spectrum still contains the memory-variable
             * contribution; each mechanism's integral is subtracted off,
             * cumulatively over l, before the elastic-form dots below. The
             * host does this in place in its own buffers; locals here. */
            float2 ixx_f = integ(Rxx_f, w), izz_f = integ(Rzz_f, w);
            float2 ixz_f = integ(Rxz_f, w);
            float2 ixx_a = integ(Rxx_a, w), izz_a = integ(Rzz_a, w);
            float2 ixz_a = integ(Rxz_a, w);
            Fxx.x -= ixx_f.x; Fxx.y -= ixx_f.y;
            Fzz.x -= izz_f.x; Fzz.y -= izz_f.y;
            Fxz.x -= ixz_f.x; Fxz.y -= ixz_f.y;
            Axx.x -= ixx_a.x; Axx.y -= ixx_a.y;
            Azz.x -= izz_a.x; Azz.y -= izz_a.y;
            Axz.x -= ixz_a.x; Axz.y -= ixz_a.y;

            float2 Rpp_f, Rpp_a, Rmm_f, Rmz_f;
            Rpp_f.x = Rxx_f.x + Rzz_f.x;  Rpp_f.y = Rxx_f.y + Rzz_f.y;
            Rpp_a.x = Rxx_a.x + Rzz_a.x;  Rpp_a.y = Rxx_a.y + Rzz_a.y;
            Rmm_f.x = Rxx_f.x - Rzz_f.x;  Rmm_f.y = Rxx_f.y - Rzz_f.y;
            Rmz_f.x = Rzz_f.x - Rxx_f.x;  Rmz_f.y = Rzz_f.y - Rxx_f.y;

            d1 += sc_ss*rmreal(Rpp_a, Rpp_f, tausig, w)/dftnorm;
            d5 += sc_ss*rmreal(Rxz_a, Rxz_f, tausig, w)/dftnorm;
            d7 += sc_ss*(rmreal(Rxx_a, Rmm_f, tausig, w)
                       + rmreal(Rzz_a, Rmz_f, tausig, w))/dftnorm;
        }
        double d6 = d1;

        float2 Fpp, App, Fmm, Fmz;
        Fpp.x = Fxx.x + Fzz.x;  Fpp.y = Fxx.y + Fzz.y;
        App.x = Axx.x + Azz.x;  App.y = Axx.y + Azz.y;
        Fmm.x = Fxx.x - Fzz.x;  Fmm.y = Fxx.y - Fzz.y;
        Fmz.x = Fzz.x - Fxx.x;  Fmz.y = Fzz.y - Fxx.y;

        double d0 = sc_ss*w*itreal(App, Fpp)/dftnorm;
        double d2 = sc_ss*w*itreal(Axz, Fxz)/dftnorm;
        double d3 = d0;
        double d4 = sc_ss*w*(itreal(Axx, Fmm) + itreal(Azz, Fmz))/dftnorm;
        double d8 = sc_vv*w*(itreal(fvx[id], fvx_f[id])
                           + itreal(fvz[id], fvz_f[id]))/dftnorm;

        GM   += -c0*d0 + c1*d1;
        Gmu  += -c2*d2 + c3*d3 - c4*d4 + c5*d5 - c6*d6 + c7*d7;
        Gtaup+= -c8*d0 + c9*d1;
        Gtaus+= -c10*d2 + c11*d3 - c12*d4 + c13*d5 - c14*d6 + c15*d7;
        Grho += -d8;
    }

    gradM[gid]    += (float)GM;
    gradmu[gid]   += (float)Gmu;
    gradtaup[gid] += (float)Gtaup;
    gradtaus[gid] += (float)Gtaus;
    gradrho[gid]  += (float)Grho;

    /* NOT YET SPLIT ONTO THE STAGGERED PARAMETERS. The elastic kernel sends
     * the sxz correlation to gradmuipkp and the two velocity correlations to
     * gradrip/gradrkp, and average_grad_transpose folds them back. The
     * viscoelastic host block this is validated against still accumulates
     * cell-centred, so this kernel does too -- device and host have to agree
     * for SEISCL_DFT_CHECK to mean anything. Doing it properly here needs the
     * shear terms (d2 and d5, both shear-stress correlations) evaluated at
     * muipkp *and* their taus counterparts at tausipkp, in the host block
     * first. Follow-up; it is the same averaging gap the elastic path had. */
}
