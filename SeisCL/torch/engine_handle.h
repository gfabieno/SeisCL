// A built, reusable engine instance: one CUDA context, one set of compiled
// kernels, one set of allocated device+host buffers.
//
// Building all of that (assign_modeling_case -> Init_cst -> Init_data ->
// Init_model -> Init_CUDA) is what the binding used to redo on every call.
// A handle keeps it alive so that a repeat call with the same CacheKey only
// has to refresh values and re-run time_stepping().

#ifndef SEISCL_TORCH_ENGINE_HANDLE_H
#define SEISCL_TORCH_ENGINE_HANDLE_H

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "cache_key.h"
#include "config.h"

extern "C" {
#include "F.h"
}

namespace seiscl_torch {

struct EngineHandle {
    model m;
    device *dev = nullptr;
    CacheKey key;
    bool built = false;

    // Set when a forward pass ran with SKIP_CHECKPOINT_FILE: the boundary
    // wavefield for pending_ckpt's run is sitting in this handle's buffers
    // and was never written to disk, so the matching backward pass must
    // either reuse this handle or have the file flushed to it first. While
    // this is set the handle is pinned against LRU eviction, since evicting
    // it would destroy the only copy.
    bool pending_valid = false;
    std::string pending_ckpt;

    EngineHandle();
    ~EngineHandle();

    // model/device own raw pointers with no copy semantics.
    EngineHandle(const EngineHandle &) = delete;
    EngineHandle &operator=(const EngineHandle &) = delete;
};

parameter *find_par(model &m, const std::string &name);
variable *find_var(model &m, const std::string &name);

void require_cpu(const torch::Tensor &t, const std::string &what);

void apply_config(model &m, const Config &cfg, int gradout, int inputres);

void set_params(model &m, const py::dict &params);

void set_srcrec(model &m, torch::Tensor &src, torch::Tensor &src_pos,
                torch::Tensor &rec_pos,
                const std::vector<std::string> &output_fields);

// Full build path, for a cache miss. Returns the engine's state code (0 = ok).
int engine_build(EngineHandle &h, const Config &cfg, int gradout, int inputres,
                 const py::dict &params, torch::Tensor &src,
                 torch::Tensor &src_pos, torch::Tensor &rec_pos,
                 const std::vector<std::string> &output_fields);

// Cache-hit paths: new values into existing allocations.
int engine_refresh_params(EngineHandle &h, const py::dict &params);

int engine_refresh_srcrec(EngineHandle &h, torch::Tensor &src,
                          torch::Tensor &src_pos, torch::Tensor &rec_pos);

// Clear the host-side accumulators that time_stepping() adds into.
void engine_reset_outputs(EngineHandle &h);

// Bytes one shot's boundary checkpoint occupies. Multiplied by the shot
// count, this is what a RAM-backed checkpoint would cost.
std::size_t checkpoint_bytes_per_shot(const EngineHandle &h);

}  // namespace seiscl_torch

#endif  // SEISCL_TORCH_ENGINE_HANDLE_H
