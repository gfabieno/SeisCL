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

/* Frequency-domain (BACK_PROP_TYPE==2) gradient correlation, 2D SH, elastic.
 *
 * Companion to grad_dft2D.cl; see that file for the conventions. SH has only
 * the shear modulus and density, so M/gradM are not arguments at all -- naming
 * them here would leave them unmatched by prog_create's by-name binding, which
 * hands unmatched arguments a placeholder that is not a valid OpenCL memory
 * object (it segfaults). The host calc_grad has no HOUT block for SH either,
 * so no Hessian is produced here.
 *
 * Transcribed from the ND==21 branch of calc_grad(), with grad_coefelast_0_SH
 * for the coefficients.
 */

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define NZM (NZ - 2*FDOH)
#define NXM (NX - 2*FDOH)
#define NPAD (NX*NZ)
#define indf(f,i,k) ((f)*NPAD + ((i)+FDOH)*NZ + ((k)+FDOH))

LFUNDEF double itreal(float2 a, float2 b)
{
    return (double)a.y*(double)b.x - (double)a.x*(double)b.y;
}

FUNDEF void calc_grad_dft(GLOBARG float * gradfreqsn,
                          GLOBARG float * mu,
                          GLOBARG float * rho,
                          GLOBARG float * gradmu,
                          GLOBARG float * gradrho,
                          GLOBARG float2 * fvy_f,
                          GLOBARG float2 * fsxy_f,
                          GLOBARG float2 * fsyz_f,
                          GLOBARG float2 * fvy,
                          GLOBARG float2 * fsxy,
                          GLOBARG float2 * fsyz)
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

    double s2    = pow(2.0, -(double)PARSCALE);
    double dhdt  = (double)DH/(double)DT;
    double lrho  = (double)rho[gid];
    double rho_p = (lrho!=0.0) ? (1.0/lrho)*((double)DT/(double)DH)*s2 : 0.0;
    double mu_p  = (double)mu[gid]*dhdt*s2;

    double dftdf  = 1.0/((double)NTNYQ*(double)DT*(double)DTNYQ);
    double dftnorm = (double)NTNYQ*(double)DTNYQ;

    /* grad_coefelast_0_SH. Vacuum and fluid cells contribute nothing. */
    double c0=0.0, c4=0.0;
    if (rho_p>0.0 && mu_p>=1.0){
        c0 = 2.0*sqrt(rho_p*mu_p)/(mu_p*mu_p);
        c4 = mu_p/rho_p/(mu_p*mu_p);
    }

    double gmu=0.0, grho=0.0;

    for (f=0; f<NFREQS; f++){

        double w = 2.0*3.14159265358979323846*dftdf*(double)gradfreqsn[f];
        int id = indf(f,i,k);

        double d0 = w*(itreal(fsxy[id], fsxy_f[id])
                     + itreal(fsyz[id], fsyz_f[id]))/dftnorm;
        double d2 = w*itreal(fvy[id], fvy_f[id])/dftnorm;

        gmu  += -c0*d0;
        /* c4 is vs^2 times the internal coefficient of d0, so it carries the
         * same sign as gmu -- see the equivalent group in grad_dft2D.cl. */
        grho += -d2 - c4*d0;
    }

    gradmu[gid]  += (float)gmu;
    gradrho[gid] += (float)grho;
}
