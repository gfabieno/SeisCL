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

/* Preparation of the model parameters on the device (Init_OpenCL.c,
   param_prep_run). The input is the raw model as the user gives it (the
   arrays <name>_raw: whole model, float, the values of the HDF5 file or of the
   torch tensors); the output is what the update kernels read (cl_par of this
   device: its x slice, scaled, float or half). It replaces the host functions
   M(), mu(), rho(), set_par_scale(), check_stability(), the tau averages
   (ave_arithmetic_tau) and the float to half conversion of Init_model.c, and
   computes the same bits: same expressions, same float and double promotions,
   the float divisions done in double then rounded (= the correctly rounded
   float division of the host, also under OpenCL, whose float division may be
   2.5 ulp), no a*b+c that a compiler could fuse.

   prep_reduce: max of the transformed M (par_scale) and the vp, vs extrema of
       check_stability, over the whole model; prep_red holds the float bits
       (every value >= 0, so the unsigned order is the float order).
   prep_store: transforms, scales by 2^par_scale (M, mu) and 2^-par_scale
       (buoyancy), stores M, mu, rho, taup, taus of this device's cells.
   prep_tausipkp, prep_tausipjp, prep_tausjpkp: the arithmetic averages of
       taus, with the edges of ave_arithmetic_tau (cells it never writes stay 0).

   prep_cst (floats; ints stored as their bits), filled by param_prep_run:
       0 dt, 1 dh, 2 ws = 2 pi f0, 3 sumu (shear phase-velocity correction),
       4 2^par_scale, 5 2^-par_scale, 6 par_type, 7 halfpar, 8 first global
       cell of this device, 9 cells of this device, 10 cells of the model,
       11 NZ, 12 NY, 13 NX (whole model, NY = 1 in 2-D), 16.. pts[l] = dt/eta[l]. */

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define PREP_GID ((int)get_global_id(0))
#define PREP_GSIZE ((int)get_global_size(0))
#define PREP_I(c, k) as_int((c)[k])
#define PREP_BITS(x) as_uint(x)
#define PREP_MAX(p, v) atomic_max((p), (v))
#define PREP_MIN(p, v) atomic_min((p), (v))
#define PREP_UINT uint
#else
#define PREP_GID ((int)(blockIdx.x * blockDim.x + threadIdx.x))
#define PREP_GSIZE ((int)(gridDim.x * blockDim.x))
#define PREP_I(c, k) __float_as_int((c)[k])
#define PREP_BITS(x) __float_as_uint(x)
#define PREP_MAX(p, v) atomicMax((p), (v))
#define PREP_MIN(p, v) atomicMin((p), (v))
#define PREP_UINT unsigned int
#endif

/* Storage of the parameters (as average_params.cl) */
#if FP16>1
    #ifdef __OPENCL_VERSION__
        #define PPAR half
        #define PPST(p, i, v) vstore_half_rte((v), (i), (p))
    #else
        #define PPAR __half
        #define PPST(p, i, v) ((p)[(i)] = __float2half_rn((v)))
    #endif
#else
    #define PPAR float
    #define PPST(p, i, v) ((p)[(i)] = (v))
#endif
/* halfpar (FP16 1): values rounded through half, as Init_model.c */
LFUNDEF float prep_halfround(float v)
{
#ifdef __OPENCL_VERSION__
    half h[1];
    vstore_half_rte(v, 0, h);
    return vload_half(0, h);
#else
    return __half2float(__float2half_rn(v));
#endif
}

/* Which parameters exist (assign_modeling_case.c append_par) */
#define PREP_HAS_M (ND!=21)
#define PREP_HAS_MU (ND!=22)
#define PREP_HAS_TAUP (LVE>0)
#define PREP_HAS_TAUS (LVE>0 && ND!=22)

/* float a/b, correctly rounded */
LFUNDEF float prep_div(float a, float b)
{
    return (float)((double)a / (double)b);
}

/* taus after the par_type transform (mu() in assign_modeling_case.c) */
LFUNDEF float prep_taus(GLOBARG const float *mu_raw, GLOBARG const float *taus_raw,
                        GLOBARG const float *c, int g)
{
#if PREP_HAS_TAUS
    float ts = taus_raw[g];
    if (PREP_I(c, 6) == 3){
        float m = mu_raw[g];
        if (m > 0){
            ts = prep_div(ts, m - ts);
        }
        else {
            ts = 0;
        }
    }
    return ts;
#else
    return 0.0f;
#endif
}

/* M, mu (transformed, phase-velocity corrected, not scaled), taup, taus of cell g */
LFUNDEF void prep_cell(GLOBARG const float *M_raw, GLOBARG const float *mu_raw,
                       GLOBARG const float *rho_raw, GLOBARG const float *taup_raw,
                       GLOBARG const float *taus_raw, GLOBARG const float *c, int g,
                       float *Mt, float *mut, float *taupt, float *taust)
{
    float dt = c[0], dh = c[1], ws = c[2], sumu = c[3];
    int pt = PREP_I(c, 6);
    float r = rho_raw[g];
    *Mt = 0; *mut = 0; *taupt = 0; *taust = 0;
#if PREP_HAS_TAUP
    *taupt = taup_raw[g];
#endif
#if PREP_HAS_TAUS
    *taust = taus_raw[g];
#endif
#if PREP_HAS_M
    {
        float x = M_raw[g];
        float v = 0;
        if (pt == 0){
            v = prep_div(x * x * r * dt, dh);
        }
        else if (pt == 1){
            v = prep_div(x * dt, dh);
        }
        else if (pt == 2){
            float q = prep_div(x, r);
            v = prep_div(q * q * r * dt, dh);
        }
        else if (pt == 3){
    #if PREP_HAS_TAUP
            float tp = *taupt;
            float vp = x - tp;
            *taupt = prep_div(tp, x - tp);
            v = prep_div(vp * vp * r * dt, dh);
    #endif
        }
    #if LVE>0
        {
            float sumpi = 0.0f;
            for (int l = 0; l < LVE; l++){
                float p = c[16 + l];
                sumpi += ((ws * ws * p * p * (*taupt)) / (1.0 + ws * ws * p * p));
            }
            v = v / (1.0 + sumpi);
        }
    #endif
        *Mt = v;
    }
#endif
#if PREP_HAS_MU
    {
        float x = mu_raw[g];
        float v = 0;
        if (pt == 0){
            v = prep_div(x * x * r * dt, dh);
        }
        else if (pt == 1){
            v = prep_div(x * dt, dh);
        }
        else if (pt == 2){
            float q = prep_div(x, r);
            v = prep_div(q * q * r * dt, dh);
        }
        else if (pt == 3){
            float vs = 0;
    #if PREP_HAS_TAUS
            if (x > 0){
                vs = x - *taust;
            }
            *taust = prep_taus(mu_raw, taus_raw, c, g);
    #endif
            v = prep_div(vs * vs * r * dt, dh);
        }
    #if LVE>0
        v = v / (1.0 + sumu * (*taust));
    #endif
        *mut = v;
    }
#endif
}

/* Buoyancy dt/(dh rho) * s (rho()), 0 in a vacuum cell */
LFUNDEF float prep_buoy(float r, float dt, float dh, float s)
{
    if (r == 0) return 0.0f;
    return (float)(1.0 / r * dt / dh * s);
}

FUNDEF void prep_reduce(GLOBARG const float *M_raw, GLOBARG const float *mu_raw,
                        GLOBARG const float *rho_raw, GLOBARG const float *taup_raw,
                        GLOBARG const float *taus_raw, GLOBARG const float *prep_cst,
                        GLOBARG PREP_UINT *prep_red)
{
    GLOBARG const float *c = prep_cst;
    float dt = c[0], dh = c[1];
    int n = PREP_I(c, 10);
    float Mmax = 0, vpmax = 0, vsmax = 0, vpmin = 99999, vsmin = 99999;
    for (int g = PREP_GID; g < n; g += PREP_GSIZE){
        float Mt, mut, taupt, taust;
        prep_cell(M_raw, mu_raw, rho_raw, taup_raw, taus_raw, c, g, &Mt, &mut, &taupt, &taust);
        float b = prep_buoy(rho_raw[g], dt, dh, 1.0f);
#if PREP_HAS_M
        if (Mmax < Mt) Mmax = Mt;
        float vp = (float)(sqrt((double)(Mt * b)) / dt * dh);
        if (vpmax < vp) vpmax = vp;
        if (vpmin > vp && vp > 0.1) vpmin = vp;
#endif
#if PREP_HAS_MU
        float vs = (float)(sqrt((double)(mut * b)) / dt * dh);
        if (vsmax < vs) vsmax = vs;
        if (vsmin > vs && vs > 0.1) vsmin = vs;
#endif
    }
    PREP_MAX(&prep_red[0], PREP_BITS(Mmax));
    PREP_MAX(&prep_red[1], PREP_BITS(vpmax));
    PREP_MIN(&prep_red[2], PREP_BITS(vpmin));
    PREP_MAX(&prep_red[3], PREP_BITS(vsmax));
    PREP_MIN(&prep_red[4], PREP_BITS(vsmin));
}

LFUNDEF float prep_final(float v, GLOBARG const float *c)
{
#if FP16==1
    if (PREP_I(c, 7) > 0) v = prep_halfround(v);
#endif
    return v;
}

FUNDEF void prep_store(GLOBARG const float *M_raw, GLOBARG const float *mu_raw,
                       GLOBARG const float *rho_raw, GLOBARG const float *taup_raw,
                       GLOBARG const float *taus_raw, GLOBARG const float *prep_cst,
                       GLOBARG PPAR *M, GLOBARG PPAR *mu, GLOBARG PPAR *rho,
                       GLOBARG PPAR *taup, GLOBARG PPAR *taus)
{
    GLOBARG const float *c = prep_cst;
    int i = PREP_GID;
    if (i >= PREP_I(c, 9)) return;
    int g = PREP_I(c, 8) + i;
    float Mt, mut, taupt, taust;
    prep_cell(M_raw, mu_raw, rho_raw, taup_raw, taus_raw, c, g, &Mt, &mut, &taupt, &taust);
#if PREP_HAS_M
    #if FP16>0
    Mt *= c[4];
    #endif
    PPST(M, i, prep_final(Mt, c));
#endif
#if PREP_HAS_MU
    #if FP16>0
    mut *= c[4];
    #endif
    PPST(mu, i, prep_final(mut, c));
#endif
    PPST(rho, i, prep_final(prep_buoy(rho_raw[g], c[0], c[1], c[5]), c));
#if PREP_HAS_TAUP
    PPST(taup, i, prep_final(taupt, c));
#endif
#if PREP_HAS_TAUS
    PPST(taus, i, prep_final(taust, c));
#endif
}

/* ave_arithmetic_tau of assign_modeling_case.c for one cell, directions
   d0 = (z, y, x) and d1: the average of four taus where the host loop computes
   it, a copy of the cell on the edges it copies, and 0 where it writes nothing
   (the parameter buffers start at zero). */
LFUNDEF float prep_tau_ave(GLOBARG const float *mu_raw, GLOBARG const float *taus_raw,
                           GLOBARG const float *c, int g,
                           int d00, int d01, int d02, int d10, int d11, int d12)
{
    int gnz = PREP_I(c, 11), gny = PREP_I(c, 12), gnx = PREP_I(c, 13);
    int k = g % gnz, j = (g / gnz) % gny, i = g / (gnz * gny);
    float v = 0.0f;
    if (k < gnz - d00 - d10 && j < gny - d01 - d11 && i < gnx - d02 - d12){
        int g2 = (i + d02) * gny * gnz + (j + d01) * gnz + (k + d00);
        int g3 = (i + d12) * gny * gnz + (j + d11) * gnz + (k + d10);
        int g4 = (i + d02 + d12) * gny * gnz + (j + d01 + d11) * gnz + (k + d00 + d10);
        v = 0.25 * (prep_taus(mu_raw, taus_raw, c, g) + prep_taus(mu_raw, taus_raw, c, g2)
                    + prep_taus(mu_raw, taus_raw, c, g3) + prep_taus(mu_raw, taus_raw, c, g4));
    }
    int gx0 = 0, gy0 = 0, gz0 = 0;
    if (d02 == 1) gx0 = gnx - 1;
    else if (d01 == 1) gy0 = gny - 1;
    if (d00 == 1) gz0 = gnz - 1;
    if (k >= gz0 && j >= gy0 && i >= gx0) v = prep_taus(mu_raw, taus_raw, c, g);
    if (d12 == 1) gx0 = gnx - 1;
    else if (d11 == 1) gy0 = gny - 1;
    if (d10 == 1) gz0 = gnz - 1;
    if (k >= gz0 && j >= gy0 && i >= gx0) v = prep_taus(mu_raw, taus_raw, c, g);
    return v;
}

/* The directions of tausipkp, tausjpkp, tausipjp (assign_modeling_case.c);
   SH (ND 21): the 3-D staggering with nothing varying along y. */
FUNDEF void prep_tausipkp(GLOBARG const float *mu_raw, GLOBARG const float *taus_raw,
                          GLOBARG const float *prep_cst, GLOBARG PPAR *tausipkp)
{
    GLOBARG const float *c = prep_cst;
    int i = PREP_GID;
    if (i >= PREP_I(c, 9)) return;
    int g = PREP_I(c, 8) + i;
    PPST(tausipkp, i, prep_final(prep_tau_ave(mu_raw, taus_raw, c, g, 0, 0, 1, 1, 0, 0), c));
}
FUNDEF void prep_tausjpkp(GLOBARG const float *mu_raw, GLOBARG const float *taus_raw,
                          GLOBARG const float *prep_cst, GLOBARG PPAR *tausjpkp)
{
    GLOBARG const float *c = prep_cst;
    int i = PREP_GID;
    if (i >= PREP_I(c, 9)) return;
    int g = PREP_I(c, 8) + i;
#if ND==21
    float v = prep_tau_ave(mu_raw, taus_raw, c, g, 1, 0, 0, 0, 0, 0);
#else
    float v = prep_tau_ave(mu_raw, taus_raw, c, g, 0, 1, 0, 1, 0, 0);
#endif
    PPST(tausjpkp, i, prep_final(v, c));
}
FUNDEF void prep_tausipjp(GLOBARG const float *mu_raw, GLOBARG const float *taus_raw,
                          GLOBARG const float *prep_cst, GLOBARG PPAR *tausipjp)
{
    GLOBARG const float *c = prep_cst;
    int i = PREP_GID;
    if (i >= PREP_I(c, 9)) return;
    int g = PREP_I(c, 8) + i;
#if ND==21
    float v = prep_tau_ave(mu_raw, taus_raw, c, g, 0, 0, 1, 0, 0, 0);
#else
    float v = prep_tau_ave(mu_raw, taus_raw, c, g, 0, 0, 1, 0, 1, 0);
#endif
    PPST(tausipjp, i, prep_final(v, c));
}
