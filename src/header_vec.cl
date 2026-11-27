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

/* Storage, cells per thread and stencils for the update kernels.

   One kernel for every precision mode. The arithmetic is always float and
   written for one cell at a time; only the storage changes:
       FP16 0: float storage, 1 cell per thread,
       FP16 1: float storage, 2 cells per thread (along z),
       FP16 2: half storage,  2 cells per thread.
   The only gain of FP16 is in memory traffic: half storage halves it, and two
   cells per thread allow one 2-wide load per array (decisive for half storage,
   +10-20 % for float). Vector arithmetic brings nothing on a GPU (it compiles to
   scalar instructions), and half arithmetic (the former FP16 3) is slower.

   Grid sizes in cells, with the FDOH halo: NZS, NYS (3D), NX. A thread owns the
   NV cells z = Z0 + NV*t .. Z0 + NV*t + NV-1 of its column, aligned on NV in the
   padded array; with an odd FDOH the first and last thread own one halo cell,
   which it skips (INTERIOR_Z). The shared tiles are float, with a z halo ZH
   (FDOH rounded up to a multiple of NV) so that every load is aligned.

   The adjoint kernels put #pragma unroll on their loops over the NV cells: without
   it the compiler keeps the large loop (gradient, source term) rolled and the
   per-cell arrays (lsxx[c], lM[c], ...) go to local memory; with FP16 1/2 that made
   update_adjs 2-3x slower (measured). The forward kernels are left as they are:
   unrolled, the 3-D FP16 1 forward kernels were slower. */

#ifndef NV
#define NV DIV
#endif
#ifndef NOTILE
#define NOTILE 0      /* experiment, see the end of this file */
#endif
#ifndef CPMLINL
#define CPMLINL 0     /* experiment: CPML adjoint inside the 3D adjoint updates */
#endif
#if FP16==2
#define STO half
#else
#define STO float
#endif
/* Operand of the derivative functions: a float tile in local memory or, without
   tiles (NOTILE), the stored array itself, read through ldf (half -> float for FP16 2). */
#if NOTILE
#define TPTR GLOBARG const STO *
#define TV(t, i) ldf((t), (i))
#else
#define TPTR LOCID const float *
#define TV(t, i) ((t)[i])
#endif

#ifndef NZS
#define NZS (NZ*DIV)
#endif
#if ND==3 && !defined(NYS)
#define NYS NY
#endif

#define Z0 (FDOH - FDOH%NV)
#define ZH (FDOH + (NV - FDOH%NV)%NV)
#define INTERIOR_Z(z) ((z)>=FDOH && (z)<NZS-FDOH)

/* ---- one value / NV aligned values, as float ---------------------------- */
LFUNDEF float ldf(GLOBARG const STO *a, int i)
{
#if FP16==2
    #ifdef __OPENCL_VERSION__
    return vload_half(i, a);
    #else
    return __half2float(a[i]);
    #endif
#else
    return a[i];
#endif
}

LFUNDEF void stf(GLOBARG STO *a, int i, float v)
{
#if FP16==2
    #ifdef __OPENCL_VERSION__
    vstore_half(v, i, a);
    #else
    a[i] = __float2half(v);
    #endif
#else
    a[i] = v;
#endif
}

/* NV values a[i .. i+NV-1], i a multiple of NV: one load. */
LFUNDEF void ldn(GLOBARG const STO *a, int i, float *v)
{
#if NV==1
    v[0] = ldf(a, i);
#else
    #ifdef __OPENCL_VERSION__
        #if FP16==2
        float2 t = vloada_half2(0, a + i);
        #else
        float2 t = vload2(0, a + i);
        #endif
    #else
        #if FP16==2
        float2 t = __half22float2(*(const __half2 *)(a + i));
        #else
        float2 t = *(const float2 *)(a + i);
        #endif
    #endif
    v[0] = t.x; v[1] = t.y;
#endif
}

LFUNDEF void stn(GLOBARG STO *a, int i, const float *v)
{
#if NV==1
    stf(a, i, v[0]);
#else
    #ifdef __OPENCL_VERSION__
        #if FP16==2
        vstorea_half2((float2)(v[0], v[1]), 0, a + i);
        #else
        vstore2((float2)(v[0], v[1]), 0, a + i);
        #endif
    #else
        #if FP16==2
        *(__half2 *)(a + i) = __floats2half2_rn(v[0], v[1]);
        #else
        *(float2 *)(a + i) = make_float2(v[0], v[1]);
        #endif
    #endif
#endif
}

/* ---- thread and tile geometry (2D: z, x) --------------------------------- */
#ifdef __OPENCL_VERSION__
#define LID0 get_local_id(0)
#define LID1 get_local_id(1)
#define LSZ0 get_local_size(0)
#define LSZ1 get_local_size(1)
#define GRP0 get_group_id(0)
#define GRP1 get_group_id(1)
#else
#define LID0 threadIdx.x
#define LID1 threadIdx.y
#define LSZ0 blockDim.x
#define LSZ1 blockDim.y
#define GRP0 blockIdx.x
#define GRP1 blockIdx.y
#endif

/* Fill a float tile [tx][tz] (z fastest) with a[z, x] for z = z0 .. z0+tz-1,
   x = x0 .. x0+tx-1 (zero outside the array), cooperatively, NV values per load. */
LFUNDEF void load_tile2(LOCID float *tile, GLOBARG const STO *a, int z0, int x0, int tz, int tx)
{
    int nthr = LSZ0 * LSZ1;
    int nzv = tz / NV;
    for (int i = LID0 + LID1 * LSZ0; i < nzv * tx; i += nthr){
        int lz = (i % nzv) * NV, lx = i / nzv;
        int gz = z0 + lz, gx = x0 + lx;
        if (gz >= 0 && gz + NV <= NZS && gx >= 0 && gx < NX){
            ldn(a, gx * NZS + gz, &tile[lx * tz + lz]);
        }
        else {
            for (int c = 0; c < NV; c++) tile[lx * tz + lz + c] = 0.0f;
        }
    }
}

/* ---- staggered derivatives on a tile (t[lx*tz + lz]) --------------------- */
#define HCV {HC1, HC2, HC3, HC4, HC5, HC6}
LFUNDEF float tdxp(TPTR t, int tz, int lz, int lx)
{   const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++) d += hc[m] * (TV(t, (lx+m+1)*tz + lz) - TV(t, (lx-m)*tz + lz));
    return d; }
LFUNDEF float tdxm(TPTR t, int tz, int lz, int lx)
{   const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++) d += hc[m] * (TV(t, (lx+m)*tz + lz) - TV(t, (lx-m-1)*tz + lz));
    return d; }
LFUNDEF float tdzp(TPTR t, int tz, int lz, int lx)
{   const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++) d += hc[m] * (TV(t, lx*tz + lz+m+1) - TV(t, lx*tz + lz-m));
    return d; }
LFUNDEF float tdzm(TPTR t, int tz, int lz, int lx)
{   const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++) d += hc[m] * (TV(t, lx*tz + lz+m) - TV(t, lx*tz + lz-m-1));
    return d; }

/* ---- the NV values of a model parameter for this thread (2D) ------------- */
/* Parameters have no halo (index (x-FDOH)*(NZS-2FDOH) + z-FDOH): the cells of a
   thread are aligned for one load only when FDOH%NV == 0. Halo cells get 0. */
LFUNDEF void ldp2(GLOBARG const STO *a, int gidx, int zt, float *v)
{
#if FDOH%NV==0
    ldn(a, (gidx-FDOH)*(NZS-2*FDOH) + (zt-FDOH), v);
#else
    for (int c=0; c<NV; c++)
        v[c] = INTERIOR_Z(zt+c) ? ldf(a, (gidx-FDOH)*(NZS-2*FDOH) + (zt+c-FDOH)) : 0.0f;
#endif
}

/* ---- scaled storage (FP16 > 0) -------------------------------------------- */
/* With FP16 > 0 the fields and parameters are stored scaled by powers of two
   (set_par_scale, res_scale, src_scale) so that half precision does not
   overflow. The gradient accumulations carry the factor
   2^(2 PARSCALE - src_scale - res_scale), which the host removes (unscale_grad,
   2^par_scale); FP16 0: no scaling. */
LFUNDEF float pow2s(float a, int e)
{
#if FP16==0
    return a;
#elif defined(__OPENCL_VERSION__)
    return ldexp(a, e);
#else
    return scalbnf(a, e);
#endif
}
#define GSCALE(a) pow2s((a), 2*PARSCALE - src_scale - res_scale)

/* ---- 3D: thread geometry, tiles, stencils, parameters --------------------- */
#ifdef __OPENCL_VERSION__
#define LID2 get_local_id(2)
#define LSZ2 get_local_size(2)
#define GRP2 get_group_id(2)
#else
#define LID2 threadIdx.z
#define LSZ2 blockDim.z
#define GRP2 blockIdx.z
#endif

#if ND==3
/* Fill a float tile [tx][ty][tz] (z fastest) with a[z, y, x], z = z0 .. z0+tz-1,
   y = y0 .. , x = x0 .. (zero outside the array), NV values per load. */
LFUNDEF void load_tile3(LOCID float *tile, GLOBARG const STO *a, int z0, int y0, int x0,
                        int tz, int ty, int tx)
{
    int nthr = LSZ0 * LSZ1 * LSZ2;
    int nzv = tz / NV;
    for (int i = LID0 + LSZ0*(LID1 + LSZ1*LID2); i < nzv * ty * tx; i += nthr){
        int lz = (i % nzv) * NV, ly = (i / nzv) % ty, lx = i / (nzv * ty);
        int gz = z0 + lz, gy = y0 + ly, gx = x0 + lx;
        int it = (lx * ty + ly) * tz + lz;
        if (gz >= 0 && gz + NV <= NZS && gy >= 0 && gy < NYS && gx >= 0 && gx < NX){
            ldn(a, (gx * NYS + gy) * NZS + gz, &tile[it]);
        }
        else {
            for (int c = 0; c < NV; c++) tile[it + c] = 0.0f;
        }
    }
}

/* the NV values of a model parameter for this thread (3D, no halo) */
LFUNDEF void ldp3(GLOBARG const STO *a, int gidx, int gidy, int zt, float *v)
{
    int base = ((gidx-FDOH)*(NYS-2*FDOH) + (gidy-FDOH))*(NZS-2*FDOH);
#if FDOH%NV==0
    ldn(a, base + (zt-FDOH), v);
#else
    for (int c=0; c<NV; c++)
        v[c] = INTERIOR_Z(zt+c) ? ldf(a, base + (zt+c-FDOH)) : 0.0f;
#endif
}
#endif

/* Staggered derivatives along any axis of a tile: element i, stride s
   (1 along z, tz along y, tz*ty along x in 3D; tz along x in 2D). */
LFUNDEF float tdp(TPTR t, int i, int s)
{   const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++) d += hc[m] * (TV(t, i + (m+1)*s) - TV(t, i - m*s));
    return d; }
LFUNDEF float tdm(TPTR t, int i, int s)
{   const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++) d += hc[m] * (TV(t, i + m*s) - TV(t, i - (m+1)*s));
    return d; }

/* ---- CPMLINL (experiment): exact CPML adjoint inside update_adjs/update_adjv ----
   3D, BACK_PROP_TYPE 1, one device, NOTILE, FP16 0 (clprogram.c, cpml_inline()).
   Replaces cpml_adj3D.cl and its cpf buffers. The filter of cpml_adj3D.cl along one
   axis depends only on the coordinate along that axis, so the filtered value of a
   stencil neighbour is computed where it is used:
       f = e/K[c] + (b[c]*psit_old + a[c]*e)   in a layer of that axis,
         computed as e*Ka[c] + b[c]*psit_old, Ka = 1/K + a (not bit-identical),
       f = e                                    elsewhere,
       f = 0                                    in the halo (the cpf buffers hold zeros there).
   Each work-item writes psit_new of its own cell. The adjoint memories psi_*r are
   double buffered (2 halves of PSI_H* floats, Init_OpenCL.c): read half nt&1, write
   the other, so no work-item overwrites a value that a neighbour still reads.
   Axis parameters: g coordinate (padded), n length (padded), low: the low layer
   exists (z: FREESURF==0), full: coefficients on the full grid (index indm+1 in the
   high layer) or the half grid (indm), pb/ps: psi index = pb + l*ps for layer index l. */
#if CPMLINL
#if !NOTILE || NV!=1
#error "CPMLINL needs NOTILE and one cell per work-item (FP16 0)"
#endif
#define PSI_HX (2*NAB*(NYS-2*FDOH)*(NZS-2*FDOH))
#define PSI_HY (2*NAB*(NX-2*FDOH)*(NZS-2*FDOH))
#define PSI_HZ (2*NAB*(NX-2*FDOH)*(NYS-2*FDOH))
#if FREESURF==0
#define CPI_LOWZ 1
#else
#define CPI_LOWZ 0
#endif
/* layer index of coordinate g (interior), or -1; *c: index of the coefficients */
LFUNDEF int cpi_layer(int g, int n, int low, int full, int *c)
{
    if (g > n-NAB-FDOH-1){ int l = g-n+NAB+FDOH+NAB; *c = 2*NAB-1-l+full; return l; }
    if (low && g-FDOH < NAB){ *c = g-FDOH; return g-FDOH; }
    return -1;
}
LFUNDEF float cpi_val(GLOBARG const STO *f, int ind, GLOBARG const float *psi, int pb, int ps,
                      int g, int n, int low, int full,
                      GLOBARG const float *Ka, GLOBARG const float *b)
{
    if (g < FDOH || g > n-FDOH-1) return 0.0f;
    float e = ldf(f, ind);
    int c;
    int l = cpi_layer(g, n, low, full, &c);
    if (l < 0) return e;
    return e * Ka[c] + b[c] * psi[pb + l*ps];   /* Ka = 1/K + a (CPML_ka) */
}
/* no layer and no halo within FDOH of g: the plain derivative */
#define CPI_CLEAR(g, n, low) ((g)-FDOH >= FDOH + (low)*NAB && (g)+FDOH <= (n)-NAB-FDOH-1)
LFUNDEF float cpi_tdp(GLOBARG const STO *f, int ind, int s, GLOBARG const float *psi, int pb, int ps,
                      int g, int n, int low, int full,
                      GLOBARG const float *Ka, GLOBARG const float *b)
{
    if (CPI_CLEAR(g, n, low)) return tdp(f, ind, s);
    const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++)
        d += hc[m] * (cpi_val(f, ind + (m+1)*s, psi, pb, ps, g+m+1, n, low, full, Ka, b)
                      - cpi_val(f, ind - m*s, psi, pb, ps, g-m, n, low, full, Ka, b));
    return d;
}
LFUNDEF float cpi_tdm(GLOBARG const STO *f, int ind, int s, GLOBARG const float *psi, int pb, int ps,
                      int g, int n, int low, int full,
                      GLOBARG const float *Ka, GLOBARG const float *b)
{
    if (CPI_CLEAR(g, n, low)) return tdm(f, ind, s);
    const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++)
        d += hc[m] * (cpi_val(f, ind + m*s, psi, pb, ps, g+m, n, low, full, Ka, b)
                      - cpi_val(f, ind - (m+1)*s, psi, pb, ps, g-m-1, n, low, full, Ka, b));
    return d;
}
/* the three memories of one axis, in the cell's own values (psi_new = b*psi_old + a*e; one layer test,
   fields read only in a layer). fullN: field N on the full grid (coefficients a, b) or half (ah, bh). */
LFUNDEF void cpi_upd_one(GLOBARG const STO *f, int ind, GLOBARG float *psi, int H, int o, int i,
                         GLOBARG const float *a, GLOBARG const float *b, int c)
{
    float e = ldf(f, ind);
    psi[(o^1)*H + i] = b[c] * psi[o*H + i] + a[c] * e;
}
LFUNDEF void cpi_upd3(int ind, int g, int n, int low, int H, int o, int pb, int ps,
                      GLOBARG const STO *f1, GLOBARG float *p1, int full1,
                      GLOBARG const STO *f2, GLOBARG float *p2, int full2,
                      GLOBARG const STO *f3, GLOBARG float *p3, int full3,
                      GLOBARG const float *a, GLOBARG const float *b,
                      GLOBARG const float *ah, GLOBARG const float *bh)
{
    int c;
    int l = cpi_layer(g, n, low, 0, &c);
    if (l < 0) return;
    int hi = l != c;                     /* high layer: the full-grid coefficient is c+1 */
    int i = pb + l*ps;
    if (full1) cpi_upd_one(f1, ind, p1, H, o, i, a, b, c+hi); else cpi_upd_one(f1, ind, p1, H, o, i, ah, bh, c);
    if (full2) cpi_upd_one(f2, ind, p2, H, o, i, a, b, c+hi); else cpi_upd_one(f2, ind, p2, H, o, i, ah, bh, c);
    if (full3) cpi_upd_one(f3, ind, p3, H, o, i, a, b, c+hi); else cpi_upd_one(f3, ind, p3, H, o, i, ah, bh, c);
}
/* per-cell axis parameters; cpi_o: the half of a psi_*r array read this step (the other is written) */
#define CPI_AXES \
    int cpi_pbx = (gidy-FDOH)*(NZS-2*FDOH) + (zt-FDOH), cpi_psx = (NYS-2*FDOH)*(NZS-2*FDOH); \
    int cpi_pby = (gidx-FDOH)*2*NAB*(NZS-2*FDOH) + (zt-FDOH), cpi_psy = NZS-2*FDOH; \
    int cpi_pbz = ((gidx-FDOH)*(NYS-2*FDOH) + (gidy-FDOH))*2*NAB, cpi_psz = 1; \
    int cpi_o = nt & 1, cpi_n = cpi_o ^ 1;
#define CPI_X(psi) psi + cpi_o*PSI_HX, cpi_pbx, cpi_psx, gidx, NX, 1
#define CPI_Y(psi) psi + cpi_o*PSI_HY, cpi_pby, cpi_psy, gidy, NYS, 1
#define CPI_Z(psi) psi + cpi_o*PSI_HZ, cpi_pbz, cpi_psz, zt, NZS, CPI_LOWZ
#endif

/* ---- domain decomposition along x (time_stepping.c, comm.c) ---------------- */
/* Only the first subdomain has the low x layer and only the last the high one. The x halo
   of an inner cut holds the neighbour's cells (exchanged every update): interior cells, not
   a border. One device and one process: X_FIRST = X_LAST = 1, everything as before. */
#define X_FIRST (DEVID==0 && MYLOCALID==0)
#define X_LAST (DEVID==NUM_DEVICES-1 && MYLOCALID==NLOCALP-1)
/* cpml_adj*: their x range also covers the halo of an inner cut (Init_OpenCL.c), where they
   write the identity copy (no x layer there; the z and y filters are not read in the halo) */
#define CPML_XH0 (X_FIRST ? 0 : FDOH)
#define CPML_XH1 (X_LAST ? 0 : FDOH)
#define CPML_HALOX(x) ((x)<FDOH || (x)>NX-FDOH-1)

/* ---- exact CPML adjoint (ABS_TYPE 1, BACK_PROP_TYPE 1) -------------------- */
/* cpml_adj2D/3D.cl filter the adjoint fields into the cpf buffers only in the
   cells of an absorbing layer (CPML_CELL; the top layer only without a free
   surface; padded indices). Elsewhere the filter is the identity, so the
   adjoint updates read the field itself there (load_tile*_cpf), and the cpf
   buffer in the layers and in the halo (where it holds zeros, never written:
   reading the field there would change the result under a free surface). */
#if ABS_TYPE==1 && BACK_PROP_TYPE==1
#if FREESURF==0
#define CPML_TOPZ(z) ((z)-FDOH<NAB)
#else
#define CPML_TOPZ(z) 0
#endif
#define CPML_Z(z) ((z)>NZS-NAB-FDOH-1 || CPML_TOPZ(z))
#define CPML_X(x) ((X_FIRST && (x)-FDOH<NAB) || (X_LAST && (x)>NX-NAB-FDOH-1))
#define HALO_Z(z) ((z)<FDOH || (z)>NZS-FDOH-1)
#define HALO_X(x) ((X_FIRST && (x)<FDOH) || (X_LAST && (x)>NX-FDOH-1))
#if ND==3
#define CPML_Y(y) ((y)-FDOH<NAB || (y)>NYS-NAB-FDOH-1)
#define HALO_Y(y) ((y)<FDOH || (y)>NYS-FDOH-1)
#define CPML_CELL3(z,y,x) (CPML_Z(z) || CPML_Y(y) || CPML_X(x))
#define READ_CPF3(z,y,x) (CPML_CELL3(z,y,x) || HALO_Z(z) || HALO_Y(y) || HALO_X(x))
#else
#define CPML_CELL2(z,x) (CPML_Z(z) || CPML_X(x))
#define READ_CPF2(z,x) (CPML_CELL2(z,x) || HALO_Z(z) || HALO_X(x))
#endif

/* Whole tiles: decided once per work group (the tile box is the same for all
   its work-items), so that only the few tiles that straddle the edge of a layer
   test each element. A box [z0,z1) x [y0,y1) x [x0,x1) is RAW when it lies in
   the interior outside every layer, CPF when it lies in one layer slab (which
   includes the halo next to it). */
#if FREESURF==0
#define CPML_ZLO (FDOH+NAB)
#else
#define CPML_ZLO FDOH
#endif
#define CPML_ZHI (NZS-NAB-FDOH)
#define CPML_XLO (X_FIRST ? FDOH+NAB : 0)
#define CPML_XHI (X_LAST ? NX-NAB-FDOH : NX)
#if FREESURF==0
#define CPF_TOPBOX(z1) ((z1) <= CPML_ZLO)
#else
#define CPF_TOPBOX(z1) 0
#endif
#define RAW_BOX2(z0,z1,x0,x1) ((z0)>=CPML_ZLO && (z1)<=CPML_ZHI && (x0)>=CPML_XLO && (x1)<=CPML_XHI)
#define CPF_BOX2(z0,z1,x0,x1) ((z0)>=CPML_ZHI || CPF_TOPBOX(z1) || (x1)<=CPML_XLO || (x0)>=CPML_XHI)
#if ND==3
#define CPML_YLO (FDOH+NAB)
#define CPML_YHI (NYS-NAB-FDOH)
#define RAW_BOX3(z0,z1,y0,y1,x0,x1) (RAW_BOX2(z0,z1,x0,x1) && (y0)>=CPML_YLO && (y1)<=CPML_YHI)
#define CPF_BOX3(z0,z1,y0,y1,x0,x1) (CPF_BOX2(z0,z1,x0,x1) || (y1)<=CPML_YLO || (y0)>=CPML_YHI)
#endif
/* Without tiles (NOTILE): decided per work-item. A work-item (cells z..z+NV-1 at x, y) reads the
   field itself when its stencil box is RAW (NT_RAW*), the cpf buffer otherwise. The cpf cells such a
   reader can reach lie within 2*FDOH+NV-1 (z) and 2*FDOH (x, y) of a layer or of the halo, so
   cpml_adj_* writes only there (NT_SKIP*: no reader needs this cell). */
#define NT_MZ (2*FDOH+NV-1)
#define NT_RAW2(z,x) RAW_BOX2((z)-FDOH, (z)+NV+FDOH, (x)-FDOH, (x)+FDOH+1)
#define NT_SKIP2(z,x) RAW_BOX2((z)-NT_MZ, (z)+NT_MZ+1, (x)-2*FDOH, (x)+2*FDOH+1)
#if ND==3
#define NT_RAW3(z,y,x) RAW_BOX3((z)-FDOH, (z)+NV+FDOH, (y)-FDOH, (y)+FDOH+1, (x)-FDOH, (x)+FDOH+1)
#define NT_SKIP3(z,y,x) RAW_BOX3((z)-NT_MZ, (z)+NT_MZ+1, (y)-2*FDOH, (y)+2*FDOH+1, (x)-2*FDOH, (x)+2*FDOH+1)
#endif

/* NV values at element i of cpf (where sel(c) is true) or of the field a. */
#define LDN_CPF(cpf, a, i, v, sel) do{ \
        int n_cpf = 0; \
        for (int c = 0; c < NV; c++) n_cpf += (sel(c)) ? 1 : 0; \
        if (n_cpf == NV) ldn(cpf, i, v); \
        else if (n_cpf == 0) ldn(a, i, v); \
        else for (int c = 0; c < NV; c++) (v)[c] = (sel(c)) ? ldf(cpf, (i)+c) : ldf(a, (i)+c); \
    } while(0)

#if ND==3
/* 3D: not used for now. Reading the fields in place costs the 3D adjoint
   kernels more (registers: 48 -> 60 for update_adjv, -20 % occupancy, measured
   on an A6000) than the filter copies it saves; cpml_adj3D.cl still writes the
   copies everywhere. Kept for the redesign of the 3D tiles. */
#define CPML_INPLACE3 0
LFUNDEF void load_tile3_cpf(LOCID float *tile, GLOBARG const STO *cpf, GLOBARG const STO *a,
                            int z0, int y0, int x0, int tz, int ty, int tx)
{
#if !CPML_INPLACE3
    load_tile3(tile, cpf, z0, y0, x0, tz, ty, tx); return;
#endif
    if (RAW_BOX3(z0, z0+tz, y0, y0+ty, x0, x0+tx)){ load_tile3(tile, a, z0, y0, x0, tz, ty, tx); return; }
    if (CPF_BOX3(z0, z0+tz, y0, y0+ty, x0, x0+tx)){ load_tile3(tile, cpf, z0, y0, x0, tz, ty, tx); return; }
    int nthr = LSZ0 * LSZ1 * LSZ2;
    int nzv = tz / NV;
    for (int i = LID0 + LSZ0*(LID1 + LSZ1*LID2); i < nzv * ty * tx; i += nthr){
        int lz = (i % nzv) * NV, ly = (i / nzv) % ty, lx = i / (nzv * ty);
        int gz = z0 + lz, gy = y0 + ly, gx = x0 + lx;
        int it = (lx * ty + ly) * tz + lz;
        if (gz >= 0 && gz + NV <= NZS && gy >= 0 && gy < NYS && gx >= 0 && gx < NX){
            #define SEL3(c) READ_CPF3(gz+(c), gy, gx)
            LDN_CPF(cpf, a, (gx * NYS + gy) * NZS + gz, &tile[it], SEL3);
            #undef SEL3
        }
        else {
            for (int c = 0; c < NV; c++) tile[it + c] = 0.0f;
        }
    }
}
#else
LFUNDEF void load_tile2_cpf(LOCID float *tile, GLOBARG const STO *cpf, GLOBARG const STO *a,
                            int z0, int x0, int tz, int tx)
{
    if (RAW_BOX2(z0, z0+tz, x0, x0+tx)){ load_tile2(tile, a, z0, x0, tz, tx); return; }
    if (CPF_BOX2(z0, z0+tz, x0, x0+tx)){ load_tile2(tile, cpf, z0, x0, tz, tx); return; }
    int nthr = LSZ0 * LSZ1;
    int nzv = tz / NV;
    for (int i = LID0 + LID1 * LSZ0; i < nzv * tx; i += nthr){
        int lz = (i % nzv) * NV, lx = i / nzv;
        int gz = z0 + lz, gx = x0 + lx;
        if (gz >= 0 && gz + NV <= NZS && gx >= 0 && gx < NX){
            #define SEL2(c) READ_CPF2(gz+(c), gx)
            LDN_CPF(cpf, a, gx * NZS + gz, &tile[lx * tz + lz], SEL2);
            #undef SEL2
        }
        else {
            for (int c = 0; c < NV; c++) tile[lx * tz + lz + c] = 0.0f;
        }
    }
}
#endif
#endif

/* ---- no shared tiles (NOTILE, clprogram.c: notile_on) ---------------------- */
/* The kernels keep their code: a "tile" is the stored array itself, with the global
   strides, read through TV (ldf: float, or half converted to float for FP16 2). */
#if NOTILE
#ifdef __OPENCL_VERSION__
/* the __local kernel argument stays (the host passes 4 bytes) under another name */
#undef LOCARG
#define LOCARG __local float *nt_unused
#endif
#undef LOCDEF
#define LOCDEF TPTR lvar = 0;
#undef BARRIER
#define BARRIER
#define load_tile2(t, a, z0, x0, tz, tx) ((t) = (a))
#define load_tile3(t, a, z0, y0, x0, tz, ty, tx) ((t) = (a))
/* exact CPML adjoint: the field itself or the filtered copy, per work-item (NT_RAW*; uses the
   kernel's zt, gidy, gidx) */
#define load_tile2_cpf(t, cpf, a, z0, x0, tz, tx) ((t) = NT_RAW2(zt, gidx) ? (a) : (cpf))
#define load_tile3_cpf(t, cpf, a, z0, y0, x0, tz, ty, tx) ((t) = NT_RAW3(zt, gidy, gidx) ? (a) : (cpf))
#endif

/* The FP16 scale of the model is the kernel argument par_scale (set at each
   shot by time_stepping.c), not a build option: one compiled program serves
   every model. (OpenCL build options are upper-cased, so it cannot be passed
   as -D PARSCALE=par_scale.) */
#ifndef PARSCALE
#define PARSCALE par_scale
#endif
