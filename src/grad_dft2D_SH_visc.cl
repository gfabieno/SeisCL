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
 * with the same memory-variable handling (forward s' = s - integ(r), adjoint
 * s~' = s~ + integ(r~), halfstep on the forward r), the same shear_stag_coef
 * (thesis eq. 3.54 c1^mu, c4^mu, c1^taus, c4^taus; checked with sympy) and the
 * same DFT conventions. GM, Gmu, Gtaup, Gtaus (normal stresses) vanish.
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

/* calc_grad.c's cl_rm: Re(conj(a)*(tausig + 1/(i*w))*b) (grad_dft3D_visc.cl). */
LFUNDEF double rmreal(float2 a, float2 b, double tausig, double w)
{
    double th  = 0.5*w*(double)DT;
    double iwd = 0.5*(double)DT*cos(th)/sin(th);   /* discrete 1/w */
    return tausig*((double)a.x*(double)b.x + (double)a.y*(double)b.y)
         + ((double)a.x*(double)b.y - (double)a.y*(double)b.x)*iwd;
}

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

LFUNDEF float2 sub2(float2 a, float2 b)
{
    float2 o; o.x = a.x-b.x; o.y = a.y-b.y; return o;
}

LFUNDEF float2 add2(float2 a, float2 b)
{
    float2 o; o.x = a.x + b.x; o.y = a.y + b.y;
    return o;
}

/* grad_dft3D_visc.cl's shear_stag_coef, unchanged. */
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
    double c2xy,c5xy,c10xy,c13xy;
    double c2yz,c5yz,c10yz,c13yz;
    shear_stag_coef(muipjp_p, tausipjp_p, al, Ld, &c2xy, &c5xy, &c10xy, &c13xy);
    shear_stag_coef(mujpkp_p, tausjpkp_p, al, Ld, &c2yz, &c5yz, &c10yz, &c13yz);

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
        double d5xy=0.0, d5yz=0.0;
        for (l=0; l<LVE; l++){
            int il = indr(f,l,i,k);
            double tausig = 1.0/(2.0*3.14159265358979323846*(double)FL[l]);
            float2 Rxy_f = halfstep(frxy_f[il], w, HSGN);
            float2 Ryz_f = halfstep(fryz_f[il], w, HSGN);
            float2 Rxy_a = frxy[il], Ryz_a = fryz[il];
            /* forward s' = s - sum R; adjoint s~' = s~ + integ(r~) (thesis 3.33) */
            Fxy = sub2(Fxy, integ(Rxy_f, w));
            Fyz = sub2(Fyz, integ(Ryz_f, w));
            Axy = add2(Axy, integ(Rxy_a, w));
            Ayz = add2(Ayz, integ(Ryz_a, w));
            d5xy += sc_ss*rmreal(Rxy_a, Rxy_f, tausig, w)/dftnorm;
            d5yz += sc_ss*rmreal(Ryz_a, Ryz_f, tausig, w)/dftnorm;
        }
        double d2xy = sc_ss*(sdt*itreal(Axy, Fxy) + s2dt*rreal(Axy, Fxy))/dftnorm;
        double d2yz = sc_ss*(sdt*itreal(Ayz, Fyz) + s2dt*rreal(Ayz, Fyz))/dftnorm;

        /* Velocity block (3-D d8y): vy, its force source restored with the
         * FIELD weight DT*DTNYQ, and the eq. (26a) term. */
        float2 Fvy = fvy_f[id], Avy = fvy[id];
        if (sstype==1){ Fvy.x += (float)((double)DT*(double)DTNYQ)*Sd.x; Fvy.y += (float)((double)DT*(double)DTNYQ)*Sd.y; }
        double d8y = sc_vv*((sdt*itreal(Avy, Fvy) + s2dt*rreal(Avy, Fvy))
                     - (sstype==1 ? rreal(Avy, S) : 0.0))/dftnorm;

        Gmuipjp   += -c2xy*d2xy + c5xy*d5xy;
        Gmujpkp   += -c2yz*d2yz + c5yz*d5yz;
        Gtausipjp += -c10xy*d2xy + c13xy*d5xy;
        Gtausjpkp += -c10yz*d2yz + c13yz*d5yz;
        Grho      += -d8y;
    }

    gradmuipjp[gid]   += (float)Gmuipjp;
    gradmujpkp[gid]   += (float)Gmujpkp;
    gradtausipjp[gid] += (float)Gtausipjp;
    gradtausjpkp[gid] += (float)Gtausjpkp;
    gradrho[gid]      += (float)Grho;
}
