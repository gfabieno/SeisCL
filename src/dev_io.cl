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

/* Residual scaling on the device (dev_io.c): the kernels of res_scale() and
   rec_grad_dft() of residuals.c, with the same expressions and roundings
   (explicitly rounded products and sums, so that no a*b+c is fused).

   dio_rec_gather: on each device, the parameters at the receivers it owns
       (the x slice OFFSET .. OFFSET+NX-1), as float, into dio_recpar
       (DIO_NP values per receiver: rip rjp rkp rho M mu taup taus muipkp
       muipjp mujpkp tausipkp tausipjp tausjpkp; the last is 1 for an owned
       receiver).
   dio_res_scale: one work-item per receiver: scales the residual of one
       output variable (first argument, set by dev_io.c) as res_scale(), and
       the maximum |residual| (float bits) into dio_red.
   dio_recmod: one work-item per receiver: the parameters of the receiver
       cell, from the device that owns it, into recmod (rec_mod_fill of
       residuals.c: stress receivers).
   dio_recgrad: one work-item per receiver: the receiver-cell sum of
       rec_grad_dft(), in double, into dio_acc.
   dio_recgrad_add: one work-item: adds the sums to the gradient term, in the
       receiver order of the host (several receivers may share a cell).

   dio_cst (floats; ints as their bits), filled by dev_io.c before each launch:
       0 dh, 1 dt, 2 nrec, 3 NT, 4 tmax, 5 scaler, 6 kind (0 vx, 1 vy, 2 vz,
       3 p, 4 + stress_kind: sxx syy szz sxz sxy syz), 7 number of slots of dio_recpar_all, 8 ngmax, 9 L, 10 first row of
       the shot, 11 OFFSET, 12 cells along x of this device, 13 N[0], 14 N[1],
       15 slot of this device, 16 tmin, 17 sc of rec_grad_dft. */

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF
#define DIO_GID ((int)get_global_id(0))
#define DIO_I(c, k) as_int((c)[k])
#define DIO_BITS(x) as_uint(x)
#define DIO_MAX(p, v) atomic_max((p), (v))
#define DIO_UINT uint
#define FMUL(a, b) ((a) * (b))
#define FADD(a, b) ((a) + (b))
#define FSUB(a, b) ((a) - (b))
#define DMUL(a, b) ((a) * (b))
#define DADD(a, b) ((a) + (b))
#define DIO_H2F(p, i) vload_half((i), (p))
#define DIO_HALF half
#else
#define DIO_GID ((int)(blockIdx.x * blockDim.x + threadIdx.x))
#define DIO_I(c, k) __float_as_int((c)[k])
#define DIO_BITS(x) __float_as_uint(x)
#define DIO_MAX(p, v) atomicMax((p), (v))
#define DIO_UINT unsigned int
#define FMUL(a, b) __fmul_rn((a), (b))
#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))
#define DADD(a, b) __dadd_rn((a), (b))
#define DIO_H2F(p, i) __half2float((p)[(i)])
#define DIO_HALF __half
#endif

#if FP16>1
#define DPAR DIO_HALF
#define DLD(p, i) DIO_H2F(p, i)
#else
#define DPAR float
#define DLD(p, i) ((p)[(i)])
#endif

#define DIO_NP 15

/* Receiver g of the shot: its cell (as residuals.c: x = pos/dh, truncated) */
LFUNDEF void dio_cell(GLOBARG const float *rec_pos, GLOBARG const float *c, int g,
                      int *x, int *y, int *z)
{
    float dh = c[0];
    *x = (int)(rec_pos[0 + 8 * g] / dh);
    *y = (int)(rec_pos[1 + 8 * g] / dh);
    *z = (int)(rec_pos[2 + 8 * g] / dh);
}

FUNDEF void dio_rec_gather(GLOBARG const DPAR *rip, GLOBARG const DPAR *rjp,
                           GLOBARG const DPAR *rkp, GLOBARG const DPAR *rho,
                           GLOBARG const DPAR *M, GLOBARG const DPAR *mu,
                           GLOBARG const DPAR *taup, GLOBARG const DPAR *taus,
                           GLOBARG const DPAR *muipkp, GLOBARG const DPAR *muipjp,
                           GLOBARG const DPAR *mujpkp, GLOBARG const DPAR *tausipkp,
                           GLOBARG const DPAR *tausipjp, GLOBARG const DPAR *tausjpkp,
                           GLOBARG const float *rec_pos, GLOBARG float *dio_recpar,
                           GLOBARG const float *dio_cst)
{
    GLOBARG const float *c = dio_cst;
    int g = DIO_GID;
    if (g >= DIO_I(c, 2)) return;
    int x, y, z;
    dio_cell(rec_pos, c, g, &x, &y, &z);
    int off = DIO_I(c, 11), nx = DIO_I(c, 12), n0 = DIO_I(c, 13), n1 = DIO_I(c, 14);
    GLOBARG float *o = &dio_recpar[(DIO_I(c, 15) * DIO_I(c, 8) + g) * DIO_NP];
    for (int k = 0; k < DIO_NP; k++) o[k] = 0.0f;
    if (x < off || x >= off + nx) return;
#if NDIM==2
    int i = (x - off) * n0 + z;
#else
    int i = ((x - off) * n1 + y) * n0 + z;
#endif
#if ND!=21
    o[0] = DLD(rip, i);
    o[2] = DLD(rkp, i);
#endif
#if ND==3
    o[1] = DLD(rjp, i);
#endif
    o[3] = DLD(rho, i);
#if ND!=21
    o[4] = DLD(M, i);
#endif
#if ND!=22
    o[5] = DLD(mu, i);
#endif
#if LVE>0
    o[6] = DLD(taup, i);
    #if ND!=22
    o[7] = DLD(taus, i);
    #endif
#endif
#if ND==2 || ND==3
    o[8] = DLD(muipkp, i);
    #if ND==3
    o[9] = DLD(muipjp, i);
    o[10] = DLD(mujpkp, i);
    #endif
    #if LVE>0
    o[11] = DLD(tausipkp, i);
        #if ND==3
    o[12] = DLD(tausipjp, i);
    o[13] = DLD(tausjpkp, i);
        #endif
    #endif
#endif
    o[DIO_NP - 1] = 1.0f;
}

/* the parameters of receiver g: the slot of the device that owns it */
LFUNDEF GLOBARG const float * dio_owner(GLOBARG const float *dio_recpar_all,
                                        GLOBARG const float *c, int g)
{
    int ns = DIO_I(c, 7), ng = DIO_I(c, 8);
    for (int d = 0; d < ns; d++){
        GLOBARG const float *p = &dio_recpar_all[(d * ng + g) * DIO_NP];
        if (p[DIO_NP - 1] == 1.0f) return p;
    }
    return &dio_recpar_all[g * DIO_NP];
}

FUNDEF void dio_res_scale(GLOBARG float *res, GLOBARG const float *dio_recpar_all,
                          GLOBARG const float *dio_cst, GLOBARG DIO_UINT *dio_red)
{
    GLOBARG const float *c = dio_cst;
    int g = DIO_GID;
    if (g >= DIO_I(c, 2)) return;
    float dh = c[0], dt = c[1];
    int NTt = DIO_I(c, 3), tmax = DIO_I(c, 4), scaler = DIO_I(c, 5), kind = DIO_I(c, 6);
    GLOBARG const float *p = dio_owner(dio_recpar_all, c, g);
    GLOBARG float *r = &res[(DIO_I(c, 10) + g) * NTt];
    float resmax = 0;
    if (kind < 3){
        /* vx, vy, vz: the staggered buoyancy (SH: vy uses rho) */
        float par = (kind == 0) ? p[0] : (kind == 1 ? (ND == 21 ? p[3] : p[1]) : p[2]);
        float sc2 = 1.0f;
        for (int k = 0; k < (scaler > 0 ? scaler : -scaler); k++) sc2 = (scaler > 0) ? sc2 * 2.0f : sc2 * 0.5f;
        float parscal = (float)(1.0 / par * dh / dt * sc2);
        double f = 1.0 / parscal * dt;
        for (int t = 0; t < tmax; t++){
            float v = (float)DMUL((double)r[t], f);
            r[t] = v;
            if (resmax < fabs(v)) resmax = fabs(v);
        }
    }
    else if (kind >= 4){
        /* stress receivers: the diagonal modulus (rec_mod_diag of residuals.c) */
        int q = kind - 4;
        float Lf = (float)DIO_I(c, 9);
        float md = (q <= 2) ? FMUL(p[4], FADD(1.0f, FMUL(Lf, p[6])))
                 : (q == 3) ? FMUL(p[8], FADD(1.0f, FMUL(Lf, p[11])))
                 : (q == 4) ? FMUL(p[9], FADD(1.0f, FMUL(Lf, p[12])))
                 :            FMUL(p[10], FADD(1.0f, FMUL(Lf, p[13])));
        int s2 = -2 * scaler;
        float sc2 = 1.0f;
        for (int k = 0; k < (s2 > 0 ? s2 : -s2); k++) sc2 = (s2 > 0) ? sc2 * 2.0f : sc2 * 0.5f;
        float parscal = FMUL(FMUL(-md, dt) / dh, sc2);
        float f = FMUL(parscal, dt);
        for (int t = 0; t < tmax; t++){
            float v = FMUL(r[t], f);
            r[t] = v;
#if FP16>0
            if (resmax < fabs(v)) resmax = fabs(v);
#endif
        }
    }
    else {
        /* p: the unrelaxed trace modulus N M - 2 (N-1) mu (residuals.c) */
        float NDf = (float)NDIM;
        float ltp = 1.0f, lts = 1.0f;
        int L = DIO_I(c, 9);
        if (L > 0){
            ltp = FADD(1.0f, FMUL((float)L, p[6]));
            lts = FADD(1.0f, FMUL((float)L, p[7]));
        }
        float trmod = FSUB(FMUL(FMUL(NDf, p[4]), ltp),
                           FMUL(FMUL(FMUL(2.0f, NDf - 1.0f), p[5]), lts));
        int s2 = -2 * scaler;
        float sc2 = 1.0f;
        for (int k = 0; k < (s2 > 0 ? s2 : -s2); k++) sc2 = (s2 > 0) ? sc2 * 2.0f : sc2 * 0.5f;
        float parscal = FMUL(FMUL(-trmod, dt) / dh, sc2);
        float f = FMUL(parscal, dt);
        for (int t = 0; t < tmax; t++){
            float v = FMUL(r[t], f);
            r[t] = v;
#if FP16>0
            if (resmax < fabs(v)) resmax = fabs(v);
#endif
        }
    }
    DIO_MAX(&dio_red[0], DIO_BITS(resmax));
}

FUNDEF void dio_recgrad(GLOBARG const float *out, GLOBARG const float *res,
                        GLOBARG const float *dio_cst, GLOBARG double *dio_acc)
{
    GLOBARG const float *c = dio_cst;
    int g = DIO_GID;
    if (g >= DIO_I(c, 2)) return;
    int NTt = DIO_I(c, 3), tmax = DIO_I(c, 4), tmin = DIO_I(c, 16);
    GLOBARG const float *o = &out[(DIO_I(c, 10) + g) * NTt];
    GLOBARG const float *r = &res[(DIO_I(c, 10) + g) * NTt];
    double acc = 0.0;
    for (int t = tmin + 1; t < tmax; t++){
        acc = DADD(acc, DMUL((double)r[t], (double)o[t] - (double)o[t - 1]));
    }
    dio_acc[g] = acc;
}

FUNDEF void dio_recgrad_add(GLOBARG float *gradrec, GLOBARG const float *rec_pos,
                            GLOBARG const float *dio_cst, GLOBARG const double *dio_acc)
{
    GLOBARG const float *c = dio_cst;
    if (DIO_GID != 0) return;
    int nrec = DIO_I(c, 2), n0 = DIO_I(c, 13), n1 = DIO_I(c, 14);
    float sc = c[17];
    for (int g = 0; g < nrec; g++){
        int x, y, z;
        dio_cell(rec_pos, c, g, &x, &y, &z);
#if NDIM==2
        int pos = x * n0 + z;
#else
        int pos = x * n0 * n1 + y * n0 + z;
#endif
        gradrec[pos] = FADD(gradrec[pos], (float)(-sc * dio_acc[g]));
    }
}

/* reduce_seis() of time_stepping.c: adds the traces of the receivers this
   device owns (posx = floor(x/dh) in its slice) to the seismograms of the
   shot, which the call zeroes first (dio_reset). varout: this device's
   recordings, or their copy on the first device (dio.stage). */
FUNDEF void dio_add_out(GLOBARG float *out, GLOBARG const float *varout,
                        GLOBARG const float *rec_pos, GLOBARG const float *dio_cst)
{
    GLOBARG const float *c = dio_cst;
    int g = DIO_GID;
    if (g >= DIO_I(c, 2)) return;
    int NTt = DIO_I(c, 3), off = DIO_I(c, 11), nx = DIO_I(c, 12);
    int posx = (int)floor(rec_pos[8 * g] / c[0]);
    if (posx < off || posx >= off + nx) return;
    GLOBARG float *o = &out[(DIO_I(c, 10) + g) * NTt];
    GLOBARG const float *v = &varout[g * NTt];
    for (int t = 0; t < NTt; t++) o[t] = FADD(o[t], v[t]);
}

/* rec_mod_fill of residuals.c on the device: the parameters of the cell of
   receiver g (owner's slot) in the order of recmod (F.h) */
FUNDEF void dio_recmod(GLOBARG const float *dio_recpar_all, GLOBARG const float *dio_cst,
                       GLOBARG float *recmod)
{
    GLOBARG const float *c = dio_cst;
    int g = DIO_GID;
    if (g >= DIO_I(c, 2)) return;
    GLOBARG const float *p = dio_owner(dio_recpar_all, c, g);
    GLOBARG float *o = &recmod[g * 10];
    o[0] = p[4];  o[1] = p[5];  o[2] = p[6];  o[3] = p[7];
    o[4] = p[8];  o[5] = p[11]; o[6] = p[9];  o[7] = p[12];
    o[8] = p[10]; o[9] = p[13];
}
