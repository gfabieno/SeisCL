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
   (FDOH rounded up to a multiple of NV) so that every load is aligned. */

#ifndef NV
#define NV DIV
#endif
#if FP16==2
#define STO half
#else
#define STO float
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
LFUNDEF float tdxp(LOCID const float *t, int tz, int lz, int lx)
{   const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++) d += hc[m] * (t[(lx+m+1)*tz + lz] - t[(lx-m)*tz + lz]);
    return d; }
LFUNDEF float tdxm(LOCID const float *t, int tz, int lz, int lx)
{   const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++) d += hc[m] * (t[(lx+m)*tz + lz] - t[(lx-m-1)*tz + lz]);
    return d; }
LFUNDEF float tdzp(LOCID const float *t, int tz, int lz, int lx)
{   const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++) d += hc[m] * (t[lx*tz + lz+m+1] - t[lx*tz + lz-m]);
    return d; }
LFUNDEF float tdzm(LOCID const float *t, int tz, int lz, int lx)
{   const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++) d += hc[m] * (t[lx*tz + lz+m] - t[lx*tz + lz-m-1]);
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
LFUNDEF float tdp(LOCID const float *t, int i, int s)
{   const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++) d += hc[m] * (t[i + (m+1)*s] - t[i - m*s]);
    return d; }
LFUNDEF float tdm(LOCID const float *t, int i, int s)
{   const float hc[6] = HCV; float d = 0.0f;
    for (int m = 0; m < FDOH; m++) d += hc[m] * (t[i + m*s] - t[i - (m+1)*s]);
    return d; }

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
#define CPML_X(x) ((x)-FDOH<NAB || (x)>NX-NAB-FDOH-1)
#define HALO_Z(z) ((z)<FDOH || (z)>NZS-FDOH-1)
#define HALO_X(x) ((x)<FDOH || (x)>NX-FDOH-1)
#if ND==3
#define CPML_Y(y) ((y)-FDOH<NAB || (y)>NYS-NAB-FDOH-1)
#define HALO_Y(y) ((y)<FDOH || (y)>NYS-FDOH-1)
#define CPML_CELL3(z,y,x) (CPML_Z(z) || CPML_Y(y) || CPML_X(x))
#define READ_CPF3(z,y,x) (CPML_CELL3(z,y,x) || HALO_Z(z) || HALO_Y(y) || HALO_X(x))
#else
#define CPML_CELL2(z,x) (CPML_Z(z) || CPML_X(x))
#define READ_CPF2(z,x) (CPML_CELL2(z,x) || HALO_Z(z) || HALO_X(x))
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
LFUNDEF void load_tile3_cpf(LOCID float *tile, GLOBARG const STO *cpf, GLOBARG const STO *a,
                            int z0, int y0, int x0, int tz, int ty, int tx)
{
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
