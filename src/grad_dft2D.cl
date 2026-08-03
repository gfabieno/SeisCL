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

/* Frequency-domain (BACK_PROP_TYPE==2) gradient correlation, 2D P-SV, elastic.
 *
 * One work item per model cell. Replaces the host-side calc_grad() for this
 * case: it removes the two per-shot PCIe transfers of both spectra and makes
 * the DFT gradient work in the CUDA build at all, where calc_grad() is a no-op
 * stub. calc_grad() is retained as a reference implementation, selectable with
 * SEISCL_DFT_HOST=1 and cross-checkable with SEISCL_DFT_CHECK=1.
 *
 * Everything here is double precision. This kernel runs once per shot, not once
 * per time step, so the cost is irrelevant (well under a millisecond even at
 * NFREQS=128), and double keeps it bit-comparable with the host reference, which
 * is what makes SEISCL_DFT_CHECK a tight test rather than a smoke test.
 *
 * Conventions, all verified against the host implementation and against a
 * float64 numpy reference (SeisCL/tests/dft_reference.py):
 *
 *   itreal(a,b) = a.y*b.x - a.x*b.y = Im(a * conj(b))
 *   w           = 2*pi*DFTDF*bin,  DFTDF = 1/(NTNYQ*dt*DTNYQ)
 *   every dot product carries a 1/NTNYQ factor
 *   spectra are indexed f*num_ele + (i+FDOH)*NZ + (k+FDOH), the padded layout
 *
 * The coefficients are the grad_coefelast_0 / grad_coefvisc_0 family. They are
 * expressions in the *physical* stiffnesses and density, while cl_par holds the
 * internally non-dimensionalized values, so they are converted here exactly as
 * the host does.
 */

#ifdef __OPENCL_VERSION__
/* OpenCL 1.2 requires double precision to be enabled explicitly. __SEISCL__ is
 * a host-side macro and is *not* visible to the device compiler; the portable
 * guard in device sources is __OPENCL_VERSION__ (see header_CUDACL.cl). */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/* NZS and NXS are the *scalar* padded extents (clprogram.c defines them as
 * N[i]+FDORDER). Deliberately not NZ/NX: those are halved on the fastest axis
 * whenever FP16>0, since the update kernels address the wavefield as
 * float2/half2. This kernel is scalar -- savefreqs accumulates one float2
 * spectrum entry per scalar element (kernel_savefreqs indexes gid<num_ele, and
 * Init_OpenCL.c sizes cl_fvar as 2*sizeof(float)*num_ele*NFREQS regardless of
 * FP16) -- so using NZ here made the correlation cover and stride only half the
 * grid at FP16=1, giving a gradient uncorrelated with the FP16=0 one. */
#define NZM (NZS - 2*FDOH)
#define NXM (NXS - 2*FDOH)
#define NPAD (NXS*NZS)
#define indf(f,i,k) ((f)*NPAD + ((i)+FDOH)*NZS + ((k)+FDOH))

LFUNDEF double itreal(float2 a, float2 b)
{
    return (double)a.y*(double)b.x - (double)a.x*(double)b.y;
}

FUNDEF void calc_grad_dft(GLOBARG float * gradfreqsn,
                          GLOBARG float * M,
                          GLOBARG float * mu,
                          GLOBARG float * rho,
                          GLOBARG float * gradM,
                          GLOBARG float * gradmu,
                          GLOBARG float * gradrho,
                          GLOBARG float * HM,
                          GLOBARG float * Hmu,
                          GLOBARG float * Hrho,
                          GLOBARG float2 * fvx_f,
                          GLOBARG float2 * fvz_f,
                          GLOBARG float2 * fsxx_f,
                          GLOBARG float2 * fszz_f,
                          GLOBARG float2 * fsxz_f,
                          GLOBARG float2 * fvx,
                          GLOBARG float2 * fvz,
                          GLOBARG float2 * fsxx,
                          GLOBARG float2 * fszz,
                          GLOBARG float2 * fsxz,
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
    int f;

    /* Undo the internal non-dimensionalization: the coefficient expressions
     * below are in physical units. Mirrors transf_grad() in reverse. */
    double s2    = pow(2.0, -(double)par_scale);
    double dhdt  = (double)DH/(double)DT;
    double lrho  = (double)rho[gid];
    double rho_p = (lrho!=0.0) ? (1.0/lrho)*((double)DT/(double)DH)*s2 : 0.0;
    double M_p   = (double)M[gid]*dhdt*s2;
    double mu_p  = (double)mu[gid]*dhdt*s2;

    /* df = 1/(NTNYQ*dt*DTNYQ), from defines that already exist. */
    double dftdf = 1.0/((double)NTNYQ*(double)DT*(double)DTNYQ);
    /* Parseval factor: 1/(NTNYQ*DTNYQ), not 1/NTNYQ -- see calc_grad.c. Without
     * the DTNYQ the gradient scales linearly with the decimation factor. */
    double dftnorm = (double)NTNYQ*(double)DTNYQ;

    /* FP16>0 keeps the wavefield in scaled units, so the spectra savefreqs
     * accumulated are not in physical units and the coefficient expressions
     * below (which are) would be off by a power of two. seisout recovers a
     * stored value as ldexp(var, -src_scale + var->scaler)
     * (automatic_kernels.c's kernel_varout), and set_par_scale gives only
     * vx/vy/vz a nonzero scaler, equal to par_scale -- stresses keep 0. The
     * forward spectra therefore carry 2^(scaler-src_scale) and the adjoint
     * ones 2^(scaler-res_scale), so each product below needs one factor:
     *
     *   stress x stress (d0,d2,d3,d4):  2^(-src_scale-res_scale)
     *   velocity x velocity (d8):       2^(2*par_scale-src_scale-res_scale)
     *
     * There are no mixed velocity/stress products. All three scales are 0
     * when FP16==0, so this is exactly a no-op there. */
    double sc_ss = pow(2.0, -(double)src_scale - (double)res_scale);
    double sc_vv = sc_ss*pow(2.0, 2.0*(double)par_scale);

    /* ND is a build-option macro (-D ND=%d), so it must not be shadowed. */
    const double NDd = (double)ND;
    double den = (NDd*M_p - 2.0*(NDd-1.0)*mu_p);
    den = den*den;

    /* grad_coefelast_0, L==0. c[1], c[5..15], c[17], c[21..23] are zero. */
    double c0=0.0, c2=0.0, c3=0.0, c4=0.0;
    double c16=0.0, c18=0.0, c19=0.0, c20=0.0;
    if (den>0.0){
        c0  = 2.0*sqrt(rho_p*M_p)/den;
        c16 = M_p/rho_p/den;
        /* Fluid cells drop every shear-related coefficient. Guarded with a
         * branch, not a select: 1/(mu*mu) is inf at mu==0 and would poison the
         * result under fast math even though it is multiplied by zero. */
        if (mu_p>=1.0){
            c2  = 2.0*sqrt(rho_p*mu_p)/(mu_p*mu_p);
            c3  = 2.0*sqrt(rho_p*mu_p)*(NDd+1.0)/3.0/den;
            c4  = 2.0*sqrt(rho_p*mu_p)/(2.0*NDd*mu_p*mu_p);
            c18 = mu_p/rho_p/(mu_p*mu_p);
            c19 = mu_p/rho_p*(NDd+1.0)/3.0/den;
            c20 = mu_p/rho_p/(2.0*NDd*mu_p*mu_p);
        }
    }

    double gM=0.0, gmu=0.0, grho=0.0;
#if HOUT==1
    double hM=0.0, hmu=0.0, hrho=0.0;
#endif

    for (f=0; f<NFREQS; f++){

        double w = 2.0*3.14159265358979323846*dftdf*(double)gradfreqsn[f];
        int id = indf(f,i,k);

        float2 Fxx = fsxx_f[id], Fzz = fszz_f[id], Fxz = fsxz_f[id];
        float2 Axx = fsxx[id],   Azz = fszz[id],   Axz = fsxz[id];

        float2 Fpp, App, Fmm, Fmz;
        Fpp.x = Fxx.x + Fzz.x;  Fpp.y = Fxx.y + Fzz.y;   /* fwd  sxx+szz */
        App.x = Axx.x + Azz.x;  App.y = Axx.y + Azz.y;   /* adj  sxx+szz */
        Fmm.x = Fxx.x - Fzz.x;  Fmm.y = Fxx.y - Fzz.y;   /* fwd  sxx-szz */
        Fmz.x = Fzz.x - Fxx.x;  Fmz.y = Fzz.y - Fxx.y;   /* fwd  szz-sxx */

        double d0 = sc_ss*w*itreal(App, Fpp)/dftnorm;
        double d2 = sc_ss*w*itreal(Axz, Fxz)/dftnorm;
        double d3 = d0;
        double d4 = sc_ss*w*(itreal(Axx, Fmm) + itreal(Azz, Fmz))/dftnorm;
        double d8 = sc_vv*w*(itreal(fvx[id], fvx_f[id])
                     + itreal(fvz[id], fvz_f[id]))/dftnorm;

#if HOUT==1
        /* Approximate (Gauss-Newton style) Hessian diagonal, transcribed from
         * the host calc_grad 2D HOUT block. cl_norm(cl_derivative(a, w)) is
         * w^2*|a|^2, so every term here is non-negative; the sign pattern is
         * the one the host uses, which differs from the gradient's only in the
         * velocity term, chosen so the result stays positive. */
        {
            double w2 = w*w;
            /* These are forward-only squares, so both factors are the
             * forward one -- 2^(-2*src_scale), not 2^(-src_scale-res_scale).
             * Same no-op at FP16==0 as the gradient terms above. */
            double sh_ss = pow(2.0, -2.0*(double)src_scale);
            double sh_vv = sh_ss*pow(2.0, 2.0*(double)par_scale);
            double h0 = sh_ss*w2*((double)Fpp.x*(double)Fpp.x
                          + (double)Fpp.y*(double)Fpp.y)/dftnorm;
            double h2 = sh_ss*w2*((double)Fxz.x*(double)Fxz.x
                          + (double)Fxz.y*(double)Fxz.y)/dftnorm;
            double h3 = h0;
            double h4 = sh_ss*w2*(((double)Fmm.x*(double)Fmm.x
                           + (double)Fmm.y*(double)Fmm.y)
                          + ((double)Fmz.x*(double)Fmz.x
                           + (double)Fmz.y*(double)Fmz.y))/dftnorm;
            float2 Vx = fvx_f[id], Vz = fvz_f[id];
            double h8 = sh_vv*w2*(((double)Vx.x*(double)Vx.x
                           + (double)Vx.y*(double)Vx.y)
                          + ((double)Vz.x*(double)Vz.x
                           + (double)Vz.y*(double)Vz.y))/dftnorm;
            hM   += c0*h0;
            hmu  += c2*h2 - c3*h3 + c4*h4;
            hrho += h8 - c16*h0 - c18*h2 + c19*h3 - c20*h4;
        }
#endif
        gM   += -c0*d0;
        gmu  += -c2*d2 + c3*d3 - c4*d4;
        /* The c16..c20 group is the parameterization chain rule and carries the
         * same signs as gM and gmu above, matching transf_grad's
         * gradrho += M/rho*gradM + mu/rho*gradmu for BACK_PROP_TYPE==1. */
        grho += -d8 - c16*d0 - c18*d2 + c19*d3 - c20*d4;
    }

    gradM[gid]   += (float)gM;
    gradmu[gid]  += (float)gmu;
    gradrho[gid] += (float)grho;
#if HOUT==1
    HM[gid]   += (float)hM;
    Hmu[gid]  += (float)hmu;
    Hrho[gid] += (float)hrho;
#endif
}
