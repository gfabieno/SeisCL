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

/*GPU port of the staggered-grid material-parameter averaging done on the
  host in src/assign_modeling_case.c's ave_arithmetic_rho()/ave_harmonic_mu()
  (rip/rjp/rkp buoyancy, muipkp/muipjp/mujpkp shear modulus). One kernel per
  named parameter, matching SeisCL's existing convention of separate named
  kernels rather than a single kernel parametrized by a runtime direction
  vector (c.f. update_v/update_s).

  Dimension-generic like their CPU counterparts: all six kernels use a full
  3D thread index (gidz, gidy, gidx) and the flat index
  ind = gidx*PARNY*PARNZ + gidy*PARNZ + gidz, matching
  ave_arithmetic_rho/ave_harmonic_mu's ind = i*NY*NZ+j*NZ+k (i=x, j=y, k=z).
  For 2D (ND==2), PARNY is fixed at 1 by clprogram.c's get_build_options, so
  gidy is always 0 and this collapses to the plain ind = gidx*PARNZ+gidz used
  before this file was made 3D-generic -- ave_rip/ave_rkp/ave_muipkp are
  unchanged in behaviour for 2D, just launched with wdim=3 (gsize[1]=1)
  instead of wdim=2 (see Init_OpenCL.c). ave_rjp/ave_muipjp/ave_mujpkp are
  registered and launched only for ND==3 (there is no y-direction average in
  2D), see assign_modeling_case.c's append_par calls for "rjp"/"muipjp"/
  "mujpkp" (transform=NULL for ND==2||ND==3, as with rip/rkp/muipkp before).

  Unlike the interior update_v/update_s kernels, this operates on the raw
  N-sized parameter arrays (in/out), NOT the FDORDER-padded wavefield-sized
  arrays -- so it uses PARNZ/PARNY/PARNX (added to get_build_options in
  src/clprogram.c specifically for this kernel), not the NZ/NY/NX macros
  other kernels rely on (those are sized for the padded wavefield arrays and
  would be wrong here).

  Runs once at model-upload time (Init_CUDA/Init_OpenCL in
  src/Init_OpenCL.c, right after the raw M/mu/rho parameter buffers are
  uploaded), replacing the host-side ave_arithmetic_rho()/ave_harmonic_mu()
  computation for ND==2 and ND==3 -- see assign_modeling_case.c's append_par
  calls for "rip"/"rjp"/"rkp"/"muipkp"/"muipjp"/"mujpkp" (transform=NULL, so
  Init_model()'s transform loop skips the CPU computation and this kernel is
  the only thing that fills them) and the read-back to host in
  Init_OpenCL.c immediately after the kernel launch (res_scale() in
  residuals.c reads rip/rkp's *host* gl_par directly, so it must stay valid
  even though the computation itself now happens on-device).

  Viscoelastic (tausipkp/tausipjp/tausjpkp, same arithmetic-average pattern
  as rip/rkp/rjp) is the same mechanical extension, not yet done.

  Storage type. With FP16>1 the device parameter buffers hold one half per
  cell (Init_model.c converts gl_par in place, Init_OpenCL.c halves
  cl_par.size), in the same order as the float arrays. The kernels used to
  read and write float there, so they wrote one buffer length past the end
  of every averaged parameter (Xid 31 on cgm-geo1, 2026-11-10). APAR is the
  storage type, AVLD reads one cell as float, AVST rounds a float to the
  storage type (round to nearest even, like float_to_half_full_rtne on the
  host). The arithmetic is always float. */

#if FP16>1
    #ifdef __OPENCL_VERSION__
        #define APAR half
        #define AVLD(p, i) vload_half((i), (p))
        #define AVST(p, i, v) vstore_half_rte((v), (i), (p))
    #else
        #define APAR __half
        #define AVLD(p, i) __half2float((p)[(i)])
        #define AVST(p, i, v) ((p)[(i)] = __float2half_rn((v)))
    #endif
#else
    #define APAR float
    #define AVLD(p, i) ((p)[(i)])
    #define AVST(p, i, v) ((p)[(i)] = (v))
#endif

/* Thread index and bounds check, the same in every kernel. */
#ifdef __OPENCL_VERSION__
    #define AVE_GID int gidz = get_global_id(0); \
                    int gidy = get_global_id(1); \
                    int gidx = get_global_id(2);
#else
    #define AVE_GID int gidz = blockIdx.x * blockDim.x + threadIdx.x; \
                    int gidy = blockIdx.y * blockDim.y + threadIdx.y; \
                    int gidx = blockIdx.z * blockDim.z + threadIdx.z;
#endif
#define AVE_IND(x, y, z) ((x) * PARNY * PARNZ + (y) * PARNZ + (z))

/* Buoyancy average between two cells (eq. 6, same zero-guard as
   ave_arithmetic_rho in assign_modeling_case.c): both vacuum -> 0; one
   vacuum -> twice the solid neighbour; both solid -> harmonic combination. */
LFUNDEF float ave_two(float a, float b)
{
    if (a == 0.0f && b == 0.0f) return 0.0f;
    if (a == 0.0f) return 2.0f * b;
    if (b == 0.0f) return 2.0f * a;
    return 2.0f / (1.0f / a + 1.0f / b);
}

/* Shear modulus average over four cells (eq. 8, same zero-guard as
   ave_harmonic_mu): zero if any of the four cells is vacuum. */
LFUNDEF float ave_four(float a, float b, float c, float d)
{
    if (a == 0.0f || b == 0.0f || c == 0.0f || d == 0.0f) return 0.0f;
    return 4.0f / (1.0f / a + 1.0f / b + 1.0f / c + 1.0f / d);
}



/* rip: average along x. The last plane keeps the cell value. */
FUNDEF void ave_rip(GLOBARG APAR *rho, GLOBARG APAR *rip)
{
    AVE_GID
    if (gidz >= PARNZ || gidy >= PARNY || gidx >= PARNX) return;
    int i1 = AVE_IND(gidx, gidy, gidz);
    if (gidx == PARNX - 1) {
        AVST(rip, i1, AVLD(rho, i1));
        return;
    }
    AVST(rip, i1, ave_two(AVLD(rho, i1), AVLD(rho, AVE_IND(gidx + 1, gidy, gidz))));
}

/* rjp: average along y (3-D only). */
FUNDEF void ave_rjp(GLOBARG APAR *rho, GLOBARG APAR *rjp)
{
    AVE_GID
    if (gidz >= PARNZ || gidy >= PARNY || gidx >= PARNX) return;
    int i1 = AVE_IND(gidx, gidy, gidz);
    if (gidy == PARNY - 1) {
        AVST(rjp, i1, AVLD(rho, i1));
        return;
    }
    AVST(rjp, i1, ave_two(AVLD(rho, i1), AVLD(rho, AVE_IND(gidx, gidy + 1, gidz))));
}

/* rkp: average along z. */
FUNDEF void ave_rkp(GLOBARG APAR *rho, GLOBARG APAR *rkp)
{
    AVE_GID
    if (gidz >= PARNZ || gidy >= PARNY || gidx >= PARNX) return;
    int i1 = AVE_IND(gidx, gidy, gidz);
    if (gidz == PARNZ - 1) {
        AVST(rkp, i1, AVLD(rho, i1));
        return;
    }
    AVST(rkp, i1, ave_two(AVLD(rho, i1), AVLD(rho, AVE_IND(gidx, gidy, gidz + 1))));
}

/* muipkp: average over the x-z cell. */
FUNDEF void ave_muipkp(GLOBARG APAR *mu, GLOBARG APAR *muipkp)
{
    AVE_GID
    if (gidz >= PARNZ || gidy >= PARNY || gidx >= PARNX) return;
    int i1 = AVE_IND(gidx, gidy, gidz);
    if (gidx == PARNX - 1 || gidz == PARNZ - 1) {
        AVST(muipkp, i1, AVLD(mu, i1));
        return;
    }
    AVST(muipkp, i1, ave_four(AVLD(mu, i1),
                              AVLD(mu, AVE_IND(gidx + 1, gidy, gidz)),
                              AVLD(mu, AVE_IND(gidx, gidy, gidz + 1)),
                              AVLD(mu, AVE_IND(gidx + 1, gidy, gidz + 1))));
}

/* muipjp: average over the x-y cell (3-D only). */
FUNDEF void ave_muipjp(GLOBARG APAR *mu, GLOBARG APAR *muipjp)
{
    AVE_GID
    if (gidz >= PARNZ || gidy >= PARNY || gidx >= PARNX) return;
    int i1 = AVE_IND(gidx, gidy, gidz);
    if (gidx == PARNX - 1 || gidy == PARNY - 1) {
        AVST(muipjp, i1, AVLD(mu, i1));
        return;
    }
    AVST(muipjp, i1, ave_four(AVLD(mu, i1),
                              AVLD(mu, AVE_IND(gidx + 1, gidy, gidz)),
                              AVLD(mu, AVE_IND(gidx, gidy + 1, gidz)),
                              AVLD(mu, AVE_IND(gidx + 1, gidy + 1, gidz))));
}

/* mujpkp: average over the y-z cell (3-D only). */
FUNDEF void ave_mujpkp(GLOBARG APAR *mu, GLOBARG APAR *mujpkp)
{
    AVE_GID
    if (gidz >= PARNZ || gidy >= PARNY || gidx >= PARNX) return;
    int i1 = AVE_IND(gidx, gidy, gidz);
    if (gidy == PARNY - 1 || gidz == PARNZ - 1) {
        AVST(mujpkp, i1, AVLD(mu, i1));
        return;
    }
    AVST(mujpkp, i1, ave_four(AVLD(mu, i1),
                              AVLD(mu, AVE_IND(gidx, gidy + 1, gidz)),
                              AVLD(mu, AVE_IND(gidx, gidy, gidz + 1)),
                              AVLD(mu, AVE_IND(gidx, gidy + 1, gidz + 1))));
}
