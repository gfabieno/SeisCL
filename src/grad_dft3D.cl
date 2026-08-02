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

/* Frequency-domain (BACK_PROP_TYPE==2) gradient correlation, 3D, elastic.
 *
 * Direct 3D extension of grad_dft2D.cl -- same structure, same coefficient
 * formulas (grad_coefelast_0's c[0..4]/c[16..20] are already generic in ND,
 * so they are unchanged), the only difference is the dot products sum over
 * the extra field components (vy, syy, sxy, syz) that 3D has and 2D does not.
 * Cross-checked term by term against the host calc_grad() ND==3, L==0 branch
 * (calc_grad.c:623-747): dot[0]/dot[2]/dot[4]/dot[8] there are exactly d0/d2/
 * d4/d8 here, and cl_diff(a,b,c)=a-b-c (calc_grad.c:46) matches the sign of
 * the sxx_myyzz/syy_mxxzz/szz_mxxyy combinations used for d4.
 *
 * One work item per model cell. See grad_dft2D.cl for the full rationale
 * (replaces the host calc_grad() for this case, everything in double
 * precision since this runs once per shot rather than once per time step).
 *
 * L>0 (viscoelastic) is not handled here -- it needs the memory-variable
 * correlation terms (dot[1],dot[5..7]) and calc_grad.c's own ND==3 L>0
 * branch has a known copy-paste bug (calc_grad.c:672-674: rxx_myyzz,
 * ryy_mxxzz and rzz_mxxyy are all assigned the same expression) that would
 * need fixing first. assign_modeling_case.c only selects this kernel when
 * L==0.
 */

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

/* NX, NY, NZ are the *padded* extents (clprogram.c defines them as
 * N[i]+FDORDER), matching the X-slowest, Z-fastest layout used everywhere
 * else in the 3D kernels (e.g. update_v3D.cl's indp). */
#define NZM (NZ - 2*FDOH)
#define NYM (NY - 2*FDOH)
#define NXM (NX - 2*FDOH)
#define NPAD (NX*NY*NZ)
#define indf(f,i,j,k) ((f)*NPAD + ((i)+FDOH)*NY*NZ + ((j)+FDOH)*NZ + ((k)+FDOH))

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
                          GLOBARG float2 * fvy_f,
                          GLOBARG float2 * fvz_f,
                          GLOBARG float2 * fsxx_f,
                          GLOBARG float2 * fsyy_f,
                          GLOBARG float2 * fszz_f,
                          GLOBARG float2 * fsxy_f,
                          GLOBARG float2 * fsxz_f,
                          GLOBARG float2 * fsyz_f,
                          GLOBARG float2 * fvx,
                          GLOBARG float2 * fvy,
                          GLOBARG float2 * fvz,
                          GLOBARG float2 * fsxx,
                          GLOBARG float2 * fsyy,
                          GLOBARG float2 * fszz,
                          GLOBARG float2 * fsxy,
                          GLOBARG float2 * fsxz,
                          GLOBARG float2 * fsyz)
{

    #ifdef __OPENCL_VERSION__
    int gid = get_global_id(0);
    #else
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    #endif

    if (gid >= NXM*NYM*NZM)
        return;

    int i = gid/(NYM*NZM);
    int rem = gid - i*NYM*NZM;
    int j = rem/NZM;
    int k = rem - j*NZM;
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
    /* Parseval factor: 1/(NTNYQ*DTNYQ), not 1/NTNYQ -- see calc_grad.c. */
    double dftnorm = (double)NTNYQ*(double)DTNYQ;

    /* ND is a build-option macro (-D ND=%d), so it must not be shadowed. */
    const double NDd = (double)ND;
    double den = (NDd*M_p - 2.0*(NDd-1.0)*mu_p);
    den = den*den;

    /* grad_coefelast_0, L==0. c[1], c[5..15], c[17], c[21..23] are zero.
     * Identical to grad_dft2D.cl: these formulas are already generic in ND. */
    double c0=0.0, c2=0.0, c3=0.0, c4=0.0;
    double c16=0.0, c18=0.0, c19=0.0, c20=0.0;
    if (den>0.0){
        c0  = 2.0*sqrt(rho_p*M_p)/den;
        c16 = M_p/rho_p/den;
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
        int id = indf(f,i,j,k);

        float2 Fxx = fsxx_f[id], Fyy = fsyy_f[id], Fzz = fszz_f[id];
        float2 Fxy = fsxy_f[id], Fxz = fsxz_f[id], Fyz = fsyz_f[id];
        float2 Axx = fsxx[id],   Ayy = fsyy[id],   Azz = fszz[id];
        float2 Axy = fsxy[id],   Axz = fsxz[id],   Ayz = fsyz[id];

        /* cl_diff(a,b,c) = a-b-c (calc_grad.c:46); cl_add sums all three. */
        float2 Fpp, App;
        Fpp.x = Fxx.x + Fyy.x + Fzz.x;  Fpp.y = Fxx.y + Fyy.y + Fzz.y;
        App.x = Axx.x + Ayy.x + Azz.x;  App.y = Axx.y + Ayy.y + Azz.y;

        float2 Fxx_myyzz, Fyy_mxxzz, Fzz_mxxyy;
        Fxx_myyzz.x = Fxx.x - Fyy.x - Fzz.x;  Fxx_myyzz.y = Fxx.y - Fyy.y - Fzz.y;
        Fyy_mxxzz.x = Fyy.x - Fxx.x - Fzz.x;  Fyy_mxxzz.y = Fyy.y - Fxx.y - Fzz.y;
        Fzz_mxxyy.x = Fzz.x - Fxx.x - Fyy.x;  Fzz_mxxyy.y = Fzz.y - Fxx.y - Fyy.y;

        double d0 = w*itreal(App, Fpp)/dftnorm;
        double d2 = w*(itreal(Axy, Fxy) + itreal(Axz, Fxz) + itreal(Ayz, Fyz))
                    /dftnorm;
        double d3 = d0;
        double d4 = w*(itreal(Axx, Fxx_myyzz)
                     + itreal(Ayy, Fyy_mxxzz)
                     + itreal(Azz, Fzz_mxxyy))/dftnorm;
        double d8 = w*(itreal(fvx[id], fvx_f[id])
                     + itreal(fvy[id], fvy_f[id])
                     + itreal(fvz[id], fvz_f[id]))/dftnorm;

#if HOUT==1
        /* Approximate (Gauss-Newton style) Hessian diagonal, same pattern as
         * grad_dft2D.cl: every term is w^2*|.|^2, non-negative, extended to
         * the extra 3D components. */
        {
            double w2 = w*w;
            double h0 = w2*((double)Fpp.x*(double)Fpp.x
                          + (double)Fpp.y*(double)Fpp.y)/dftnorm;
            double h2 = w2*(((double)Fxy.x*(double)Fxy.x + (double)Fxy.y*(double)Fxy.y)
                          + ((double)Fxz.x*(double)Fxz.x + (double)Fxz.y*(double)Fxz.y)
                          + ((double)Fyz.x*(double)Fyz.x + (double)Fyz.y*(double)Fyz.y))
                          /dftnorm;
            double h3 = h0;
            double h4 = w2*(((double)Fxx_myyzz.x*(double)Fxx_myyzz.x
                           + (double)Fxx_myyzz.y*(double)Fxx_myyzz.y)
                          + ((double)Fyy_mxxzz.x*(double)Fyy_mxxzz.x
                           + (double)Fyy_mxxzz.y*(double)Fyy_mxxzz.y)
                          + ((double)Fzz_mxxyy.x*(double)Fzz_mxxyy.x
                           + (double)Fzz_mxxyy.y*(double)Fzz_mxxyy.y))/dftnorm;
            float2 Vx = fvx_f[id], Vy = fvy_f[id], Vz = fvz_f[id];
            double h8 = w2*(((double)Vx.x*(double)Vx.x + (double)Vx.y*(double)Vx.y)
                          + ((double)Vy.x*(double)Vy.x + (double)Vy.y*(double)Vy.y)
                          + ((double)Vz.x*(double)Vz.x + (double)Vz.y*(double)Vz.y))
                          /dftnorm;
            hM   += c0*h0;
            hmu  += c2*h2 - c3*h3 + c4*h4;
            hrho += h8 - c16*h0 - c18*h2 + c19*h3 - c20*h4;
        }
#endif
        gM   += -c0*d0;
        gmu  += -c2*d2 + c3*d3 - c4*d4;
        /* c16..c20 group: parameterization chain rule, same signs as gM/gmu,
         * matching transf_grad's gradrho += M/rho*gradM + mu/rho*gradmu for
         * BACK_PROP_TYPE==1 -- see grad_dft2D.cl. */
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
