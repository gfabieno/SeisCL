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

/* Frequency-domain (back_prop_type 2) gradient correlation, 2D SH, ELASTIC.
 *
 * Built from grad_dft3D.cl by dropping every term that is zero by symmetry
 * (y-invariance, SH polarisation: vx = vz = 0, sxx = syy = szz = sxz = 0;
 * notes sh-gradient.md, step 2). What remains is exactly grad_dft3D.cl's
 *   gmuipjp += -d2xy/muipjp^2   (sxy at x+1/2, harmonic muipjp),
 *   gmujpkp += -d2yz/mujpkp^2   (syz at z+1/2, harmonic mujpkp),
 *   grjp    += -d8y             (vy; rjp = rho when nothing varies along y),
 * with the same DFT conventions (discrete d/dt, source restoration, FP16
 * scaling). gM and gmu (normal stresses) vanish. The staggered gradients are
 * folded back onto mu by average_grad_transpose() (calc_grad.c); the vy term
 * goes straight to gradrho (no average in SH).
 * Replaces the earlier grad_dft2D_SH.cl, written from the removed host
 * calc_grad() and never run (the SH adjoint did not work).
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

/* grad_coefelast_1's 1/mu^2 at a staggered mu (grad_dft3D.cl). */
LFUNDEF double imu2_stag(double mu_s)
{
    return (mu_s>=1.0) ? 1.0/(mu_s*mu_s) : 0.0;
}

FUNDEF void calc_grad_dft(GLOBARG float * gradfreqsn,
                          GLOBARG PARARG * rho,
                          GLOBARG PARARG * muipjp,
                          GLOBARG PARARG * mujpkp,
                          GLOBARG float * gradrho,
                          GLOBARG float * gradmuipjp,
                          GLOBARG float * gradmujpkp,
                          GLOBARG float2 * fvy_f,
                          GLOBARG float2 * fsxy_f,
                          GLOBARG float2 * fsyz_f,
                          GLOBARG float2 * fvy,
                          GLOBARG float2 * fsxy,
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

    if (gid >= NXM*NZM)
        return;

    /* X-slowest, Z-fastest, as the parameter arrays. */
    int i = gid/NZM;
    int k = gid - i*NZM;
    int f;

    double s2    = pow(2.0, -(double)par_scale);
    double dhdt  = (double)DH/(double)DT;
    double muipjp_p = (double)PARCONV(muipjp[gid])*dhdt*s2;
    double mujpkp_p = (double)PARCONV(mujpkp[gid])*dhdt*s2;
    double imuipjp2 = imu2_stag(muipjp_p);
    double imujpkp2 = imu2_stag(mujpkp_p);
    double dftdf = 1.0/((double)NTNYQ*(double)DT*(double)DTNYQ);
    double dftnorm = (double)NTNYQ*(double)DTNYQ;
    double sc_ss = pow(2.0, -(double)src_scale - (double)res_scale);
    double sc_vv = sc_ss*pow(2.0, 2.0*(double)par_scale);

    /* Which source, if any, sits in this cell (eq. 26a source term). In SH the
     * only source acting on these fields is the vy force (code 1). */
    int ssrc = -1, sstype = -1;
    for (int q=0; q<nsrc; q++){
        int st = (int)src_pos[4+5*q];
        if (st==1
            && (int)(src_pos[0+5*q]/DH)==i
            && (int)(src_pos[2+5*q]/DH)==k){
            ssrc = q; sstype = st; break;
        }
    }

    double gmuipjp=0.0, gmujpkp=0.0, grho=0.0;

    for (f=0; f<NFREQS; f++){

        double w = 2.0*3.14159265358979323846*dftdf*(double)gradfreqsn[f];
        /* Exact discrete equivalent of back_prop_type 1's quadrature: see
         * grad_dft3D.cl. */
        double wdt  = w*(double)DT;
        double sdt  = sin(wdt)/(double)DT;
        double s2dt = 2.0*sin(0.5*wdt)*sin(0.5*wdt)/(double)DT;
        int id = indf(f,i,k);

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

        float2 Fxy = fsxy_f[id], Fyz = fsyz_f[id];
        float2 Axy = fsxy[id],   Ayz = fsyz[id];
        double d2xy = sc_ss*(sdt*itreal(Axy, Fxy) + s2dt*rreal(Axy, Fxy))/dftnorm;
        double d2yz = sc_ss*(sdt*itreal(Ayz, Fyz) + s2dt*rreal(Ayz, Fyz))/dftnorm;

        /* Velocity block (3-D d8y): vy, its force source restored with the
         * FIELD weight DT*DTNYQ, and the eq. (26a) term. */
        float2 Fvy = fvy_f[id], Avy = fvy[id];
        if (sstype==1){ Fvy.x += (float)((double)DT*(double)DTNYQ)*Sd.x; Fvy.y += (float)((double)DT*(double)DTNYQ)*Sd.y; }
        double d8y = sc_vv*((sdt*itreal(Avy, Fvy) + s2dt*rreal(Avy, Fvy))
                     - (sstype==1 ? rreal(Avy, S) : 0.0))/dftnorm;

        gmuipjp += -d2xy*imuipjp2;
        gmujpkp += -d2yz*imujpkp2;
        grho    += -d8y;
    }

    gradmuipjp[gid] += (float)gmuipjp;
    gradmujpkp[gid] += (float)gmujpkp;
    gradrho[gid]    += (float)grho;
}
