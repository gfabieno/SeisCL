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
