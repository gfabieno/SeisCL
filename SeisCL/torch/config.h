// User-facing scalar configuration for the PyTorch binding.
//
// Mirrors the subset of SeisCL.py's constructor kwargs needed to drive one
// forward/gradient call; geometry, parameters and residuals are passed
// separately (see run_forward/run_backward in bindings.cpp). Defaults match
// SeisCL.py's constructor where applicable.

#ifndef SEISCL_TORCH_CONFIG_H
#define SEISCL_TORCH_CONFIG_H

#include <cstdint>
#include <vector>

namespace seiscl_torch {

struct Config {
    std::vector<int64_t> N;
    int ND = 2;
    float dh = 1.0f;
    float dt = 1.0f;
    int NT = 0;
    int FDORDER = 8;
    int MAXRELERROR = 0;
    int FREESURF = 0;
    int NAB = 16;
    int ABS_TYPE = 1;
    float VPPML = 3500.0f;
    float FPML = 15.0f;
    float NPOWER = 2.0f;
    float K_MAX_CPML = 2.0f;
    float abpc = 4.0f;
    int L = 0;
    float f0 = 15.0f;
    int par_type = 0;
    int FP16 = 0;
    int restype = 0;
    int GRADSRCOUT = 0;
    int HOUT = 0;
    int BACK_PROP_TYPE = 1;
    // BACK_PROP_TYPE=2 (DFT gradient) only. The frequencies the gradient is
    // correlated at, in Hz; NFREQS is its length and the DFT path cannot
    // produce a gradient without it. dft_osamp sets how often savefreqs
    // fires (DTNYQ = ceil((1/dft_osamp)/fmax/dt)); 64 is the historical
    // hardcoded value and fires on every time step.
    std::vector<float> gradfreqs;
    float dft_osamp = 64.0f;
    // First time step included in the DFT accumulation.
    int tmin = 0;
    int nmax_dev = 1;
    // Only meaningful for OpenCL builds (src/Init_OpenCL.c's device-type
    // fallback logic is entirely #ifdef __SEISCL__); inert here since
    // seiscl_core is always built for CUDA. Kept for API familiarity with
    // SeisCL.py.
    int pref_device_type = 4;
};

}  // namespace seiscl_torch

#endif  // SEISCL_TORCH_CONFIG_H
