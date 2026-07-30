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

/* NZ and NX are the *padded* extents (clprogram.c:304-315 defines them as
 * N[i]+FDORDER), so the model extents are NZ-2*FDOH and NX-2*FDOH. */
#define NZM (NZ - 2*FDOH)
#define NXM (NX - 2*FDOH)
#define NPAD (NX*NZ)
#define indf(f,i,k) ((f)*NPAD + ((i)+FDOH)*NZ + ((k)+FDOH))

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
                          GLOBARG float2 * fvx_f,
                          GLOBARG float2 * fvz_f,
                          GLOBARG float2 * fsxx_f,
                          GLOBARG float2 * fszz_f,
                          GLOBARG float2 * fsxz_f,
                          GLOBARG float2 * fvx,
                          GLOBARG float2 * fvz,
                          GLOBARG float2 * fsxx,
                          GLOBARG float2 * fszz,
                          GLOBARG float2 * fsxz)
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
    double s2    = pow(2.0, -(double)PARSCALE);
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

        double d0 = w*itreal(App, Fpp)/dftnorm;
        double d2 = w*itreal(Axz, Fxz)/dftnorm;
        double d3 = d0;
        double d4 = w*(itreal(Axx, Fmm) + itreal(Azz, Fmz))/dftnorm;
        double d8 = w*(itreal(fvx[id], fvx_f[id])
                     + itreal(fvz[id], fvz_f[id]))/dftnorm;

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
}
