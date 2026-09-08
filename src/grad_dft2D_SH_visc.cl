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

/* Frequency-domain (BACK_PROP_TYPE==2) gradient correlation, 2D SH,
 * VISCOELASTIC (LVE>0). Counterpart of grad_dft2D_SH.cl, and the SH sibling
 * of grad_dft2D_visc.cl.
 *
 * Without it an L>0 SH run falls back to the host calc_grad(), which is
 * #ifdef __SEISCL__ and a no-op stub in the CUDA build -- a silently zero
 * gradient.
 *
 * Own file, not an #if branch, because prog_args_list() (clprogram.c) parses
 * the raw signature text before preprocessing: arguments behind an #if would
 * still be listed and bound, desynchronising every index after them.
 *
 * Transcribed from calc_grad()'s ND==21, L>0 block. Two things there are
 * deliberately mirrored rather than "fixed", so that SEISCL_DFT_CHECK
 * compares like with like:
 *
 *   - dot[0] is formed from the *raw* stress spectra. The 2D P-SV block first
 *     subtracts each mechanism's memory-variable integral from the stresses;
 *     the SH block does not. Whether that is intentional or an omission in
 *     the host is an open question -- see
 *     notes/viscoelastic-inversion-plan.md.
 *   - dot[1] is *assigned* inside the mechanism loop, not accumulated, so for
 *     LVE>1 only the last mechanism contributes. The P-SV block uses +=.
 *     Almost certainly a host bug, but reproducing it keeps the oracle
 *     comparison meaningful; fix both together.
 */

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/* Scalar padded extents -- not NZ/NX, halved on the fastest axis when
 * FP16>0. See grad_dft2D.cl. */
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

LFUNDEF double itreal(float2 a, float2 b)
{
    return (double)a.y*(double)b.x - (double)a.x*(double)b.y;
}

/* calc_grad.c's cl_rm. */
LFUNDEF double rmreal(float2 a, float2 b, double tausig, double w)
{
    return tausig*((double)a.x*(double)b.x + (double)a.y*(double)b.y)
         + ((double)a.x*(double)b.y - (double)a.y*(double)b.x)/w;
}

FUNDEF void calc_grad_dft(GLOBARG float * gradfreqsn,
                          GLOBARG float * FL,
                          GLOBARG PARARG * mu,
                          GLOBARG PARARG * rho,
                          GLOBARG PARARG * taus,
                          GLOBARG float * gradmu,
                          GLOBARG float * gradrho,
                          GLOBARG float * gradtaus,
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

    double s2    = pow(2.0, -(double)par_scale);
    double dhdt  = (double)DH/(double)DT;
    double lrho  = (double)PARCONV(rho[gid]);
    double rho_p = (lrho!=0.0) ? (1.0/lrho)*((double)DT/(double)DH)*s2 : 0.0;
    double mu_p  = (double)PARCONV(mu[gid])*dhdt*s2;
    double taus_p = (double)PARCONV(taus[gid]);

    double dftdf  = 1.0/((double)NTNYQ*(double)DT*(double)DTNYQ);
    double dftnorm = (double)NTNYQ*(double)DTNYQ;

    /* Undo the FP16>0 wavefield scaling -- see grad_dft2D.cl. vy carries
     * par_scale (set_par_scale scales the velocities only); the stresses and
     * their memory variables carry none. */
    double sc_ss = pow(2.0, -(double)src_scale - (double)res_scale);
    double sc_vv = sc_ss*pow(2.0, 2.0*(double)par_scale);

    const double Ld = (double)LVE;
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

    /* grad_coefvisc_1_SH: the internal (mu, taus) coefficients. */
    double c0=0.0, c1=0.0, c2=0.0, c3=0.0;
    if (rho_p>0.0 && mu_p>=1.0 && taus_p!=0.0){
        double ist = 1.0+al*taus_p;
        double Lst = 1.0+Ld*taus_p;
        double imu2 = 1.0/(mu_p*mu_p);
        c0 = ist*imu2/Lst;
        c1 = ist*imu2/taus_p;
        c2 = (Ld-al)/(Lst*Lst*mu_p);
        c3 = 1.0/(taus_p*taus_p*mu_p);
    }

    double Gmu=0.0, Gtaus=0.0, Grho=0.0;

    for (f=0; f<NFREQS; f++){

        double w = 2.0*3.14159265358979323846*dftdf*(double)gradfreqsn[f];
        int id = indf(f,i,k);

        /* Raw stresses: the host's SH block applies no memory-variable
         * correction here. */
        double d0 = sc_ss*w*(itreal(fsxy[id], fsxy_f[id])
                           + itreal(fsyz[id], fsyz_f[id]))/dftnorm;

        double d1 = 0.0;
        for (l=0; l<LVE; l++){
            int il = indr(f,l,i,k);
            double tausig = 1.0/(2.0*3.14159265358979323846*(double)FL[l]);
            /* Assignment, not +=, mirroring the host. */
            d1 = sc_ss*(rmreal(frxy[il], frxy_f[il], tausig, w)
                      + rmreal(fryz[il], fryz_f[il], tausig, w))/dftnorm;
        }

        double d2 = sc_vv*w*itreal(fvy[id], fvy_f[id])/dftnorm;

        Gmu   += -c0*d0 + c1*d1;
        Gtaus += -c2*d0 + c3*d1;
        Grho  += -d2;
    }

    gradmu[gid]   += (float)Gmu;
    gradtaus[gid] += (float)Gtaus;
    gradrho[gid]  += (float)Grho;
}
