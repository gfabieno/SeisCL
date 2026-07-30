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

/*GPU port of the 2D staggered-grid material-parameter averaging done on the
  host in src/assign_modeling_case.c's ave_arithmetic_rho()/ave_harmonic_mu()
  (rip/rkp buoyancy, muipkp shear modulus). One kernel per named parameter
  (rip, rkp, muipkp), matching SeisCL's existing convention of separate named
  kernels rather than a single kernel parametrized by a runtime direction
  vector (c.f. update_v/update_s). 2D elastic only -- matches this project's
  scope (see notes/vacuum-freesurface-plan.md); 3D (rjp/muipjp/mujpkp) and
  viscoelastic (tausipkp/tausipjp/tausjpkp, same arithmetic-average pattern
  as rip/rkp) are the same mechanical extension, not yet done.

  Unlike the interior update_v/update_s kernels, this operates on the raw
  N-sized parameter arrays (pin/pout), NOT the FDORDER-padded wavefield-sized
  arrays -- so NZ/NX are taken as explicit runtime arguments here rather than
  the NZ/NX build-time -D macros other kernels rely on (those are sized for
  the padded wavefield arrays, would be wrong for these).

  This kernel is meant to run once at model-upload time, replacing the
  host-side ave_arithmetic_rho()/ave_harmonic_mu() computation and its
  associated host-to-device transfer of the derived rip/rkp/muipkp arrays --
  only the raw M/mu/rho would need uploading. NOT YET WIRED into
  assign_modeling_case.c/Init_OpenCL.c's kernel registration/launch
  scheduling (a materially larger change touching the live simulation
  pipeline); validated standalone against the CPU reference instead (see
  notes/vacuum-freesurface-plan.md, Phase 8). */

FUNDEF void ave_rip(GLOBARG float *rho, GLOBARG float *rip, int NZ, int NX)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0);
    int gidx = get_global_id(1);
#else
    int gidz = blockIdx.x * blockDim.x + threadIdx.x;
    int gidx = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (gidz >= NZ || gidx >= NX) return;

    int ind1 = gidx * NZ + gidz;
    if (gidx == NX - 1) {
        rip[ind1] = rho[ind1];
        return;
    }
    int ind2 = (gidx + 1) * NZ + gidz;
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

FUNDEF void ave_rkp(GLOBARG float *rho, GLOBARG float *rkp, int NZ, int NX)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0);
    int gidx = get_global_id(1);
#else
    int gidz = blockIdx.x * blockDim.x + threadIdx.x;
    int gidx = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (gidz >= NZ || gidx >= NX) return;

    int ind1 = gidx * NZ + gidz;
    if (gidz == NZ - 1) {
        rkp[ind1] = rho[ind1];
        return;
    }
    int ind2 = gidx * NZ + (gidz + 1);
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

FUNDEF void ave_muipkp(GLOBARG float *mu, GLOBARG float *muipkp, int NZ, int NX)
{
#ifdef __OPENCL_VERSION__
    int gidz = get_global_id(0);
    int gidx = get_global_id(1);
#else
    int gidz = blockIdx.x * blockDim.x + threadIdx.x;
    int gidx = blockIdx.y * blockDim.y + threadIdx.y;
#endif
    if (gidz >= NZ || gidx >= NX) return;

    int ind1 = gidx * NZ + gidz;
    if (gidx == NX - 1 || gidz == NZ - 1) {
        muipkp[ind1] = mu[ind1];
        return;
    }
    int ind2 = (gidx + 1) * NZ + gidz;
    int ind3 = gidx * NZ + (gidz + 1);
    int ind4 = (gidx + 1) * NZ + (gidz + 1);
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
