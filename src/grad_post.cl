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

/* Gradient post-processing on the device (calc_grad.c, gpost_run): the host
   functions unpack_par_fp16, add_rec_grad_dft, unscale_grad(_dft),
   average_grad_transpose, unscale_par and chain_rule_par_type, on whole-model
   float arrays of the first device, with the same expressions, the same
   float/double promotions and the same order of every sum (the scatter passes
   of the averaging transposes), each product and sum rounded explicitly.

   One work-item per cell. Pointer arguments a, b, c, d, e, f, g, h are set
   by calc_grad.c before each launch; gp_cst (doubles) holds the scalars:
       0 cells, 1 NZ, 2 NY, 3 NX (whole model, NY = 1 in 2-D),
       4.. direction(s) and offsets, see each kernel; 20 factor, 21 al,
       22 muphys, 23 dt, 24 dh, 25 2^-par_scale (float), 26 flags. */

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF
#define GP_GID ((int)get_global_id(0))
#define FMUL(a, b) ((a) * (b))
#define FADD(a, b) ((a) + (b))
#define DMUL(a, b) ((a) * (b))
#define DADD(a, b) ((a) + (b))
#define DSUB(a, b) ((a) - (b))
#define GP_H2F(p, i) vload_half((i), (p))
#define GP_HALF half
#else
#define GP_GID ((int)(blockIdx.x * blockDim.x + threadIdx.x))
#define FMUL(a, b) __fmul_rn((a), (b))
#define FADD(a, b) __fadd_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))
#define DADD(a, b) __dadd_rn((a), (b))
#define DSUB(a, b) __dsub_rn((a), (b))
#define GP_H2F(p, i) __half2float((p)[(i)])
#define GP_HALF __half
#endif
#if FP16>1
#define GPAR GP_HALF
#define GPLD(p, i) GP_H2F(p, i)
#else
#define GPAR float
#define GPLD(p, i) ((p)[(i)])
#endif

#define GPN ((int)c[0])
#define GPNZ ((int)c[1])
#define GPNY ((int)c[2])
#define GPNX ((int)c[3])

/* a (float, whole) [c[4] + i] = stored parameter b (this device's slice) */
FUNDEF void gp_cvt(GLOBARG float *a, GLOBARG const GPAR *b, GLOBARG const double *gp_cst)
{
    GLOBARG const double *c = gp_cst;
    int i = GP_GID;
    if (i >= (int)c[5]) return;
    a[(int)c[4] + i] = GPLD(b, i);
}

/* add_rec_grad_dft: a += b, b = 0 */
FUNDEF void gp_addrec(GLOBARG float *a, GLOBARG float *b, GLOBARG const double *gp_cst)
{
    GLOBARG const double *c = gp_cst;
    int i = GP_GID;
    if (i >= GPN) return;
    a[i] = FADD(a[i], b[i]);
    b[i] = 0.0f;
}

/* a *= factor (float), unscale_grad, unscale_grad_dft, unscale_par for M, mu */
FUNDEF void gp_scale(GLOBARG float *a, GLOBARG const double *gp_cst)
{
    GLOBARG const double *c = gp_cst;
    int i = GP_GID;
    if (i >= GPN) return;
    a[i] = FMUL(a[i], (float)c[20]);
}

/* unscale_par for rho (a) */
FUNDEF void gp_unbuoy(GLOBARG float *a, GLOBARG const double *gp_cst)
{
    GLOBARG const double *c = gp_cst;
    int i = GP_GID;
    if (i >= GPN) return;
    float r = a[i];
    if (r == 0){
        a[i] = 0;
    }
    else {
        a[i] = (float)(1.0 / r * (float)c[23] / (float)c[24] * (float)c[25]);
    }
}

/* grad_T_arithmetic_rho: pin a, gin b, gout d; direction (z, y, x) c[4..6].
   A cell receives the contribution of the node behind it, then its own, then
   the edge copy (the order of the host loops). */
FUNDEF void gp_trho(GLOBARG const float *a, GLOBARG const float *b, GLOBARG float *d,
                    GLOBARG const double *gp_cst)
{
    GLOBARG const double *c = gp_cst;
    int n = GP_GID;
    if (n >= GPN) return;
    int gnz = GPNZ, gny = GPNY, gnx = GPNX;
    int dz = (int)c[4], dy = (int)c[5], dx = (int)c[6];
    int k = n % gnz, j = (n / gnz) % gny, i = n / (gnz * gny);
    int off = dx * gny * gnz + dy * gnz + dz;
    float g = d[n];
    /* as ind2 of the node n - dir */
    if (i - dx >= 0 && j - dy >= 0 && k - dz >= 0 && i - dx < gnx - dx && j - dy < gny - dy && k - dz < gnz - dz){
        int m = n - off;
        if (!(a[m] == 0.0 && a[n] == 0.0) && a[n] != 0.0) g = (float)DADD((double)g, DMUL(0.5, (double)b[m]));
    }
    /* as ind1 of node n */
    if (i < gnx - dx && j < gny - dy && k < gnz - dz){
        if (!(a[n] == 0.0 && a[n + off] == 0.0) && a[n] != 0.0) g = (float)DADD((double)g, DMUL(0.5, (double)b[n]));
    }
    int gx0 = 0, gy0 = 0, gz0 = 0;
    if (dx == 1) gx0 = gnx - 1;
    else if (dy == 1) gy0 = gny - 1;
    if (dz == 1) gz0 = gnz - 1;
    if (i >= gx0 && j >= gy0 && k >= gz0) g = FADD(g, b[n]);
    d[n] = g;
}

/* the node range of a 4-point average, directions c[4..9] (d0 z, y, x, d1 z, y, x) */
LFUNDEF int gp_node(GLOBARG const double *c, int n)
{
    int gnz = GPNZ, gny = GPNY, gnx = GPNX;
    int k = n % gnz, j = (n / gnz) % gny, i = n / (gnz * gny);
    return i < gnx - (int)c[6] - (int)c[9] && j < gny - (int)c[5] - (int)c[8]
        && k < gnz - (int)c[4] - (int)c[7];
}
LFUNDEF int gp_off(GLOBARG const double *c, int dd)
{
    int gnz = GPNZ, gny = GPNY;
    int o1 = (int)c[6] * gny * gnz + (int)c[5] * gnz + (int)c[4];
    int o2 = (int)c[9] * gny * gnz + (int)c[8] * gnz + (int)c[7];
    return dd == 0 ? 0 : (dd == 1 ? o1 : (dd == 2 ? o2 : o1 + o2));
}

/* grad_T_harmonic_mu, node factors: pin a, gin b, w (double) e, valid f */
FUNDEF void gp_tmu_w(GLOBARG const float *a, GLOBARG const float *b, GLOBARG double *e,
                     GLOBARG int *f, GLOBARG const double *gp_cst)
{
    GLOBARG const double *c = gp_cst;
    int n = GP_GID;
    if (n >= GPN) return;
    f[n] = 0;
    if (!gp_node(c, n)) return;
    float p0 = a[n], p1 = a[n + gp_off(c, 1)], p2 = a[n + gp_off(c, 2)], p3 = a[n + gp_off(c, 3)];
    if (p0 == 0 || p1 == 0 || p2 == 0 || p3 == 0) return;
    double s = DADD(DADD(DADD(1.0 / p0, 1.0 / p1), 1.0 / p2), 1.0 / p3);
    if (s == 0.0) return;
    double avg = 4.0 / s;
    e[n] = DMUL(DMUL(DMUL((double)b[n], 0.25), avg), avg);
    f[n] = 1;
}

/* grad_T_harmonic_mu, one pass (contribution c[10] of every node): gout d */
FUNDEF void gp_tmu_pass(GLOBARG const float *a, GLOBARG const double *e, GLOBARG const int *f,
                        GLOBARG float *d, GLOBARG const double *gp_cst)
{
    GLOBARG const double *c = gp_cst;
    int n = GP_GID;
    if (n >= GPN) return;
    if (!f[n]) return;
    int t = n + gp_off(c, (int)c[10]);
    float pc = a[t];
    d[t] = (float)DADD((double)d[t], e[n] / (double)FMUL(pc, pc));
}

/* grad_T_arithmetic_tau, one pass: gin b, gout d */
FUNDEF void gp_ttau_pass(GLOBARG const float *b, GLOBARG float *d, GLOBARG const double *gp_cst)
{
    GLOBARG const double *c = gp_cst;
    int n = GP_GID;
    if (n >= GPN) return;
    if (!gp_node(c, n)) return;
    int t = n + gp_off(c, (int)c[10]);
    d[t] = (float)DADD((double)d[t], DMUL(0.25, (double)b[n]));
}

/* grad_T_edge_copy: gin b, gout d */
FUNDEF void gp_edge(GLOBARG const float *b, GLOBARG float *d, GLOBARG const double *gp_cst)
{
    GLOBARG const double *c = gp_cst;
    int n = GP_GID;
    if (n >= GPN) return;
    int gnz = GPNZ, gny = GPNY, gnx = GPNX;
    int k = n % gnz, j = (n / gnz) % gny, i = n / (gnz * gny);
    for (int dd = 0; dd < 2; dd++){
        int dz = (int)c[4 + 3 * dd], dy = (int)c[5 + 3 * dd], dx = (int)c[6 + 3 * dd];
        if (dz == 0 && dy == 0 && dx == 0) continue;
        int gx0 = 0, gy0 = 0, gz0 = 0;
        if (dx == 1) gx0 = gnx - 1;
        else if (dy == 1) gy0 = gny - 1;
        if (dz == 1) gz0 = gnz - 1;
        if (i >= gx0 && j >= gy0 && k >= gz0) d[n] = FADD(d[n], b[n]);
    }
}

/* shear_node_transpose with L > 0, before the transposes:
   gmun a, pmun b, ptaun e (all float), gtaun f (or 0: c[26]), gm d, gt g, cm h, ct (last) */
FUNDEF void gp_sn_pre(GLOBARG const float *a, GLOBARG const float *b, GLOBARG const float *e,
                      GLOBARG const float *f, GLOBARG float *d, GLOBARG float *g,
                      GLOBARG float *h, GLOBARG float *ct, GLOBARG const double *gp_cst)
{
    GLOBARG const double *c = gp_cst;
    int i = GP_GID;
    if (i >= GPN) return;
    double al = c[21], muphys = c[22];
    d[i] = (float)DMUL((double)a[i], DADD(1.0, DMUL(al, (double)e[i])));
    float gtn = ((int)c[26] & 1) ? f[i] : 0.0f;
    g[i] = (float)DADD((double)gtn, DMUL(DMUL((double)FMUL(a[i], b[i]), muphys), al));
    h[i] = 0; ct[i] = 0;
}

/* shear_node_transpose with L > 0, after: gmu a, gtau b (c[26] & 2), cm d, ct e, ptau f, pmu g */
FUNDEF void gp_sn_post(GLOBARG float *a, GLOBARG float *b, GLOBARG const float *d,
                       GLOBARG const float *e, GLOBARG const float *f, GLOBARG const float *g,
                       GLOBARG const double *gp_cst)
{
    GLOBARG const double *c = gp_cst;
    int i = GP_GID;
    if (i >= GPN) return;
    double al = c[21], muphys = c[22];
    double k = DADD(1.0, DMUL(al, (double)f[i]));
    a[i] = (float)DADD((double)a[i], (double)d[i] / k);
    if ((int)c[26] & 2){
        double x = DMUL(DMUL((double)FMUL(d[i], g[i]), muphys), al) / k;
        b[i] = (float)DADD((double)b[i], DSUB((double)e[i], x));
    }
}

/* chain_rule_par_type, par_type 0: rho a, gradrho b, M d, gradM e, mu f, gradmu g,
   taup h, taus (last); c[26] bits: 1 M, 2 mu, 4 taup, 8 taus, 16 gradM, 32 gradmu */
FUNDEF void gp_chain(GLOBARG const float *a, GLOBARG float *b, GLOBARG const float *d,
                     GLOBARG float *e, GLOBARG const float *f, GLOBARG float *g,
                     GLOBARG const float *h, GLOBARG const float *ts, GLOBARG const double *gp_cst)
{
    GLOBARG const double *c = gp_cst;
    int i = GP_GID;
    if (i >= GPN) return;
    int fl = (int)c[26];
    double al = c[21];
    float rho = a[i];
    if (rho != 0){
        double kp = (fl & 4) ? DADD(1.0, DMUL(al, (double)h[i])) : 1.0;
        double ks = (fl & 8) ? DADD(1.0, DMUL(al, (double)ts[i])) : 1.0;
        float gr = b[i];
        if (fl & 1) gr = (float)DADD((double)gr, DMUL(DMUL((double)d[i], kp) / (double)rho, (double)e[i]));
        if ((fl & 2) && f[i] > 0) gr = (float)DADD((double)gr, DMUL(DMUL((double)f[i], ks) / (double)rho, (double)g[i]));
        b[i] = gr;
    }
    if (fl & 16){
        double nrmp = (fl & 4) ? sqrt(DADD(1.0, DMUL(al, (double)h[i]))) : 1.0;
        e[i] = (float)DMUL(DMUL(DMUL(2.0, sqrt(DMUL((double)rho, (double)d[i]))), nrmp), (double)e[i]);
    }
    if (fl & 32){
        double nrms = (fl & 8) ? sqrt(DADD(1.0, DMUL(al, (double)ts[i]))) : 1.0;
        g[i] = (float)DMUL(DMUL(DMUL(2.0, sqrt(DMUL((double)rho, (double)f[i]))), nrms), (double)g[i]);
    }
}
