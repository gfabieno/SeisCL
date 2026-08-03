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
  as rip/rkp/rjp) is the same mechanical extension, not yet done. */

FUNDEF void ave_rip(GLOBARG float *rho, GLOBARG float *rip)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0);
    int gidy = get_global_id(1);
    int gidx = get_global_id(2);
#else
    int gidz = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;
    int gidx = blockIdx.z * blockDim.z + threadIdx.z;
#endif
    if (gidz >= PARNZ || gidy >= PARNY || gidx >= PARNX) return;

    int ind1 = gidx * PARNY * PARNZ + gidy * PARNZ + gidz;
    if (gidx == PARNX - 1) {
        rip[ind1] = rho[ind1];
        return;
    }
    int ind2 = (gidx + 1) * PARNY * PARNZ + gidy * PARNZ + gidz;
    // Same eq. 6 zero-guard as ave_arithmetic_rho (assign_modeling_case.c):
    // both vacuum -> 0; one vacuum -> twice the solid neighbor's buoyancy;
    // both solid -> harmonic-of-density combination.
    if (rho[ind1] == 0.0f && rho[ind2] == 0.0f) {
        rip[ind1] = 0.0f;
    } else if (rho[ind1] == 0.0f) {
        rip[ind1] = 2.0f * rho[ind2];
    } else if (rho[ind2] == 0.0f) {
        rip[ind1] = 2.0f * rho[ind1];
    } else {
        rip[ind1] = 2.0f / (1.0f / rho[ind1] + 1.0f / rho[ind2]);
    }
}

FUNDEF void ave_rjp(GLOBARG float *rho, GLOBARG float *rjp)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0);
    int gidy = get_global_id(1);
    int gidx = get_global_id(2);
#else
    int gidz = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;
    int gidx = blockIdx.z * blockDim.z + threadIdx.z;
#endif
    if (gidz >= PARNZ || gidy >= PARNY || gidx >= PARNX) return;

    int ind1 = gidx * PARNY * PARNZ + gidy * PARNZ + gidz;
    if (gidy == PARNY - 1) {
        rjp[ind1] = rho[ind1];
        return;
    }
    int ind2 = gidx * PARNY * PARNZ + (gidy + 1) * PARNZ + gidz;
    if (rho[ind1] == 0.0f && rho[ind2] == 0.0f) {
        rjp[ind1] = 0.0f;
    } else if (rho[ind1] == 0.0f) {
        rjp[ind1] = 2.0f * rho[ind2];
    } else if (rho[ind2] == 0.0f) {
        rjp[ind1] = 2.0f * rho[ind1];
    } else {
        rjp[ind1] = 2.0f / (1.0f / rho[ind1] + 1.0f / rho[ind2]);
    }
}

FUNDEF void ave_rkp(GLOBARG float *rho, GLOBARG float *rkp)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0);
    int gidy = get_global_id(1);
    int gidx = get_global_id(2);
#else
    int gidz = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;
    int gidx = blockIdx.z * blockDim.z + threadIdx.z;
#endif
    if (gidz >= PARNZ || gidy >= PARNY || gidx >= PARNX) return;

    int ind1 = gidx * PARNY * PARNZ + gidy * PARNZ + gidz;
    if (gidz == PARNZ - 1) {
        rkp[ind1] = rho[ind1];
        return;
    }
    int ind2 = gidx * PARNY * PARNZ + gidy * PARNZ + (gidz + 1);
    if (rho[ind1] == 0.0f && rho[ind2] == 0.0f) {
        rkp[ind1] = 0.0f;
    } else if (rho[ind1] == 0.0f) {
        rkp[ind1] = 2.0f * rho[ind2];
    } else if (rho[ind2] == 0.0f) {
        rkp[ind1] = 2.0f * rho[ind1];
    } else {
        rkp[ind1] = 2.0f / (1.0f / rho[ind1] + 1.0f / rho[ind2]);
    }
}

FUNDEF void ave_muipkp(GLOBARG float *mu, GLOBARG float *muipkp)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0);
    int gidy = get_global_id(1);
    int gidx = get_global_id(2);
#else
    int gidz = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;
    int gidx = blockIdx.z * blockDim.z + threadIdx.z;
#endif
    if (gidz >= PARNZ || gidy >= PARNY || gidx >= PARNX) return;

    int ind1 = gidx * PARNY * PARNZ + gidy * PARNZ + gidz;
    if (gidx == PARNX - 1 || gidz == PARNZ - 1) {
        muipkp[ind1] = mu[ind1];
        return;
    }
    int ind2 = (gidx + 1) * PARNY * PARNZ + gidy * PARNZ + gidz;
    int ind3 = gidx * PARNY * PARNZ + gidy * PARNZ + (gidz + 1);
    int ind4 = (gidx + 1) * PARNY * PARNZ + gidy * PARNZ + (gidz + 1);
    // Same eq. 8 zero-guard as ave_harmonic_mu (assign_modeling_case.c):
    // zero if any of the 4 contributing nodes is vacuum.
    if (mu[ind1] == 0.0f || mu[ind2] == 0.0f ||
        mu[ind3] == 0.0f || mu[ind4] == 0.0f) {
        muipkp[ind1] = 0.0f;
    } else {
        muipkp[ind1] = 4.0f / (1.0f / mu[ind1] + 1.0f / mu[ind2] +
                               1.0f / mu[ind3] + 1.0f / mu[ind4]);
    }
}

FUNDEF void ave_muipjp(GLOBARG float *mu, GLOBARG float *muipjp)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0);
    int gidy = get_global_id(1);
    int gidx = get_global_id(2);
#else
    int gidz = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;
    int gidx = blockIdx.z * blockDim.z + threadIdx.z;
#endif
    if (gidz >= PARNZ || gidy >= PARNY || gidx >= PARNX) return;

    int ind1 = gidx * PARNY * PARNZ + gidy * PARNZ + gidz;
    if (gidx == PARNX - 1 || gidy == PARNY - 1) {
        muipjp[ind1] = mu[ind1];
        return;
    }
    int ind2 = (gidx + 1) * PARNY * PARNZ + gidy * PARNZ + gidz;
    int ind3 = gidx * PARNY * PARNZ + (gidy + 1) * PARNZ + gidz;
    int ind4 = (gidx + 1) * PARNY * PARNZ + (gidy + 1) * PARNZ + gidz;
    if (mu[ind1] == 0.0f || mu[ind2] == 0.0f ||
        mu[ind3] == 0.0f || mu[ind4] == 0.0f) {
        muipjp[ind1] = 0.0f;
    } else {
        muipjp[ind1] = 4.0f / (1.0f / mu[ind1] + 1.0f / mu[ind2] +
                               1.0f / mu[ind3] + 1.0f / mu[ind4]);
    }
}

FUNDEF void ave_mujpkp(GLOBARG float *mu, GLOBARG float *mujpkp)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0);
    int gidy = get_global_id(1);
    int gidx = get_global_id(2);
#else
    int gidz = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;
    int gidx = blockIdx.z * blockDim.z + threadIdx.z;
#endif
    if (gidz >= PARNZ || gidy >= PARNY || gidx >= PARNX) return;

    int ind1 = gidx * PARNY * PARNZ + gidy * PARNZ + gidz;
    if (gidy == PARNY - 1 || gidz == PARNZ - 1) {
        mujpkp[ind1] = mu[ind1];
        return;
    }
    int ind2 = gidx * PARNY * PARNZ + (gidy + 1) * PARNZ + gidz;
    int ind3 = gidx * PARNY * PARNZ + gidy * PARNZ + (gidz + 1);
    int ind4 = gidx * PARNY * PARNZ + (gidy + 1) * PARNZ + (gidz + 1);
    if (mu[ind1] == 0.0f || mu[ind2] == 0.0f ||
        mu[ind3] == 0.0f || mu[ind4] == 0.0f) {
        mujpkp[ind1] = 0.0f;
    } else {
        mujpkp[ind1] = 4.0f / (1.0f / mu[ind1] + 1.0f / mu[ind2] +
                               1.0f / mu[ind3] + 1.0f / mu[ind4]);
    }
}
