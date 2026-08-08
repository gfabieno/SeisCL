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

/* calc_grad.c's cl_integral: divide by i*w. */
LFUNDEF float2 integ(float2 a, double w)
{
    float2 o;
    o.x = (float)((double)a.y/w);
    o.y = (float)(-(double)a.x/w);
    return o;
}

LFUNDEF float2 add3(float2 a, float2 b, float2 c)
{
    float2 o; o.x = a.x+b.x+c.x; o.y = a.y+b.y+c.y; return o;
}
/* calc_grad.c's cl_diff: a - b - c. */
LFUNDEF float2 diff3(float2 a, float2 b, float2 c)
{
    float2 o; o.x = a.x-b.x-c.x; o.y = a.y-b.y-c.y; return o;
}
LFUNDEF float2 sub2(float2 a, float2 b)
{
    float2 o; o.x = a.x-b.x; o.y = a.y-b.y; return o;
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

    double dftdf  = 1.0/((double)NTNYQ*(double)DT*(double)DTNYQ);
    double dftnorm = (double)NTNYQ*(double)DTNYQ;

    double sc_ss = pow(2.0, -(double)src_scale - (double)res_scale);
    double sc_vv = sc_ss*pow(2.0, 2.0*(double)par_scale);

    const double NDd = (double)ND;
    const double Ld  = (double)LVE;
    const double al = 0.0;   /* calc_grad.c's only assignment */

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
        int id = indf(f,i,j,k);

        float2 Fxx = fsxx_f[id], Fyy = fsyy_f[id], Fzz = fszz_f[id];
        float2 Fxy = fsxy_f[id], Fxz = fsxz_f[id], Fyz = fsyz_f[id];
        float2 Axx = fsxx[id],   Ayy = fsyy[id],   Azz = fszz[id];
        float2 Axy = fsxy[id],   Axz = fsxz[id],   Ayz = fsyz[id];

        double d1=0.0, d5=0.0, d7=0.0;

        for (l=0; l<LVE; l++){
            int il = indr(f,l,i,j,k);
            double tausig = 1.0/(2.0*3.14159265358979323846*(double)FL[l]);

            float2 Rxx_f = frxx_f[il], Ryy_f = fryy_f[il], Rzz_f = frzz_f[il];
            float2 Rxy_f = frxy_f[il], Rxz_f = frxz_f[il], Ryz_f = fryz_f[il];
            float2 Rxx_a = frxx[il],   Ryy_a = fryy[il],   Rzz_a = frzz[il];
            float2 Rxy_a = frxy[il],   Rxz_a = frxz[il],   Ryz_a = fryz[il];

            /* Every stress carries its mechanisms' contribution; strip each
             * one off, cumulatively over l, before the elastic-form dots. */
            Fxx = sub2(Fxx, integ(Rxx_f, w));
            Fyy = sub2(Fyy, integ(Ryy_f, w));
            Fzz = sub2(Fzz, integ(Rzz_f, w));
            Fxy = sub2(Fxy, integ(Rxy_f, w));
            Fxz = sub2(Fxz, integ(Rxz_f, w));
            Fyz = sub2(Fyz, integ(Ryz_f, w));
            Axx = sub2(Axx, integ(Rxx_a, w));
            Ayy = sub2(Ayy, integ(Ryy_a, w));
            Azz = sub2(Azz, integ(Rzz_a, w));
            Axy = sub2(Axy, integ(Rxy_a, w));
            Axz = sub2(Axz, integ(Rxz_a, w));
            Ayz = sub2(Ayz, integ(Ryz_a, w));

            float2 Rpp_f = add3(Rxx_f, Ryy_f, Rzz_f);
            float2 Rpp_a = add3(Rxx_a, Ryy_a, Rzz_a);
            /* Each memory variable's own stress minus the other two. */
            float2 Rxx_m = diff3(Rxx_f, Ryy_f, Rzz_f);
            float2 Ryy_m = diff3(Ryy_f, Rxx_f, Rzz_f);
            float2 Rzz_m = diff3(Rzz_f, Rxx_f, Ryy_f);

            d1 += sc_ss*rmreal(Rpp_a, Rpp_f, tausig, w)/dftnorm;
            d5 += sc_ss*(rmreal(Rxy_a, Rxy_f, tausig, w)
                       + rmreal(Rxz_a, Rxz_f, tausig, w)
                       + rmreal(Ryz_a, Ryz_f, tausig, w))/dftnorm;
            d7 += sc_ss*(rmreal(Rxx_a, Rxx_m, tausig, w)
                       + rmreal(Ryy_a, Ryy_m, tausig, w)
                       + rmreal(Rzz_a, Rzz_m, tausig, w))/dftnorm;
        }
        double d6 = d1;

        float2 Spp_f = add3(Fxx, Fyy, Fzz);
        float2 Spp_a = add3(Axx, Ayy, Azz);
        float2 Sxx_m = diff3(Fxx, Fyy, Fzz);
        float2 Syy_m = diff3(Fyy, Fxx, Fzz);
        float2 Szz_m = diff3(Fzz, Fxx, Fyy);

        double d0 = sc_ss*w*itreal(Spp_a, Spp_f)/dftnorm;
        double d2 = sc_ss*w*(itreal(Axy, Fxy) + itreal(Axz, Fxz)
                           + itreal(Ayz, Fyz))/dftnorm;
        double d3 = d0;
        double d4 = sc_ss*w*(itreal(Axx, Sxx_m) + itreal(Ayy, Syy_m)
                           + itreal(Azz, Szz_m))/dftnorm;
        double d8 = sc_vv*w*(itreal(fvx[id], fvx_f[id])
                           + itreal(fvy[id], fvy_f[id])
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

    /* Not split onto the staggered parameters -- see grad_dft2D_visc.cl. The
     * host ND==3 viscoelastic block accumulates cell-centred too, so device
     * and host stay comparable under SEISCL_DFT_CHECK. */
}
