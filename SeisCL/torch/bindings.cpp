// pybind11/torch C++ extension exposing SeisCL's forward-modeling and
// adjoint-gradient engine (seiscl_core, see src/seiscl_api.c and
// CMakeLists.txt's BUILD_TORCH_CORE target) as in-memory calls on CPU
// torch tensors. No subprocess, no MPI, and no full-model/geometry/output
// HDF5 files -- only run_backward() touches HDF5 at all, and only for a
// small internal checkpoint file (see below). SeisCL/torch/op.py wraps
// run_forward()/run_backward() in a torch.autograd.Function.
//
// Built engines (CUDA context + compiled kernels + device buffers) are
// cached and reused across calls that share a CacheKey; see
// engine_cache.h and engine_handle.h. A repeat call with the same problem
// shape only refreshes values and re-runs time_stepping().
//
// The gradient path reuses the engine's existing INPUTRES=1 two-call
// protocol (src/time_stepping.c:850-859): time_stepping() skips the forward
// loop entirely when INPUTRES=1 and GRADOUT=1, instead restoring the saved
// forward boundary wavefield from an HDF5 checkpoint file written by a
// prior INPUTRES=1, GRADOUT=0 call. This is the same protocol SeisCL.py's
// set_forward()/set_backward() already use across two subprocesses --
// reused here in-process instead of inventing new engine behavior.
//
// Parameter and gradient tensors are flat, row-major float32 arrays of
// length prod(N), matching the internal layout of model.pars[i].gl_par
// (see the indexing macros at the top of src/Init_model.c). Reshaping to
// a convenient N-dimensional shape is left to the Python caller.
//
// `params` dict keys are always the engine's *internal* parameter names --
// "M", "mu", "rho" (elastic; add "taup"/"taus" when cfg.L>0) -- regardless
// of cfg.par_type (verified empirically against assign_modeling_case.c's
// append_par() calls, which always register "M"/"mu"/"rho" as the
// parameter's .name; only its to_read field, unused by this binding, is
// the param_type-dependent HDF5 dataset name like "/vp"). par_type instead
// changes how the raw values stored under "M"/"mu" get *interpreted*: for
// par_type=0 they're actually vp/vs in m/s, squared into Lame parameters
// in place by Init_model(). This differs from SeisCL.py's Python-level
// `params` dict, which uses "vp"/"vs"/"rho" (matching the HDF5 dataset
// names) for par_type=0 -- don't assume the two are interchangeable.

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// engine_handle.h pulls in F.h (which has no include guard, so it must be
// included exactly once per translation unit).
#include "cache_key.h"
#include "config.h"
#include "engine_cache.h"
#include "engine_handle.h"

namespace seiscl_torch {
namespace {

// Where the boundary checkpoint of a multi-shot gradient run lives.
// "auto" keeps it in RAM while it fits the budget and spills to a real file
// above that -- a realistic 2D survey (400x400, NT=3000, 50 shots) needs
// ~19 GB, so this cannot simply always be memory.
enum class CkptPolicy { kAuto, kMemory, kFile };
CkptPolicy g_ckpt_policy = CkptPolicy::kAuto;
// 0 means "derive it from free memory"; anything else is an explicit cap.
std::size_t g_ckpt_budget = 0;
// Of the memory the kernel says is available, how much a checkpoint may
// claim. Not all of it: the checkpoint is held for the whole forward/backward
// pair while the rest of the process keeps allocating, and overcommitting here
// trades a slower run for an OOM kill.
double g_ckpt_ram_fraction = 0.5;

// What can actually be allocated right now without swapping. MemAvailable is
// the kernel's own estimate and accounts for reclaimable page cache, which
// MemFree does not.
std::size_t available_ram_bytes() {
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    unsigned long long value = 0;
    std::string unit;
    while (meminfo >> key >> value >> unit) {
        if (key == "MemAvailable:") {
            return static_cast<std::size_t>(value) * 1024;
        }
        meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
#ifdef _SC_AVPHYS_PAGES
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<std::size_t>(pages) *
               static_cast<std::size_t>(page_size);
    }
#endif
    return 0;
}

bool checkpoint_fits_in_memory(std::size_t bytes) {
    switch (g_ckpt_policy) {
        case CkptPolicy::kMemory: return true;
        case CkptPolicy::kFile: return false;
        case CkptPolicy::kAuto:
        default: {
            if (g_ckpt_budget) return bytes <= g_ckpt_budget;
            std::size_t avail = available_ram_bytes();
            // Unable to tell: fall back to the file rather than risk an OOM.
            if (!avail) return false;
            return static_cast<double>(bytes) <=
                   g_ckpt_ram_fraction * static_cast<double>(avail);
        }
    }
}

// Per-shot counts of rows sharing an id, from column `idcol` of a `stride`-
// wide row-major table. Ids must be sorted ascending (the engine's own
// convention, see seiscl_set_srcrec()).
std::vector<int> counts_per_shot(const float *table, int rows, int stride,
                                 const char *what) {
    std::vector<int> counts;
    if (rows <= 0) return counts;
    float thisid = table[3];
    int n = 1;
    for (int i = 1; i < rows; i++) {
        float id = table[3 + i * stride];
        if (id == thisid) {
            n += 1;
        } else if (id > thisid) {
            counts.push_back(n);
            n = 1;
            thisid = id;
        } else {
            throw std::invalid_argument(
                std::string(what) +
                " shot ids (column 3) must be sorted in ascending order");
        }
    }
    counts.push_back(n);
    return counts;
}

CacheKey make_cache_key(const Config &cfg, int allns, int allng,
                        const std::vector<int> &nsrc,
                        const std::vector<int> &nrec,
                        const std::vector<std::string> &output_fields,
                        int gradout, int inputres) {
    CacheKey k;
    k.N = cfg.N;
    k.ND = cfg.ND;
    k.dh = cfg.dh;
    k.dt = cfg.dt;
    k.NT = cfg.NT;
    k.FDORDER = cfg.FDORDER;
    k.MAXRELERROR = cfg.MAXRELERROR;
    k.FREESURF = cfg.FREESURF;
    k.NAB = cfg.NAB;
    k.ABS_TYPE = cfg.ABS_TYPE;
    k.VPPML = cfg.VPPML;
    k.FPML = cfg.FPML;
    k.NPOWER = cfg.NPOWER;
    k.K_MAX_CPML = cfg.K_MAX_CPML;
    k.abpc = cfg.abpc;
    k.L = cfg.L;
    k.f0 = cfg.f0;
    k.par_type = cfg.par_type;
    k.FP16 = cfg.FP16;
    k.restype = cfg.restype;
    k.GRADSRCOUT = cfg.GRADSRCOUT;
    k.HOUT = cfg.HOUT;
    k.BACK_PROP_TYPE = cfg.BACK_PROP_TYPE;
    k.gradfreqs = cfg.gradfreqs;
    k.dft_osamp = cfg.dft_osamp;
    k.tmin = cfg.tmin;
    k.nmax_dev = cfg.nmax_dev;
    k.pref_device_type = cfg.pref_device_type;
    k.allns = allns;
    k.allng = allng;
    k.nsrc = nsrc;
    k.nrec = nrec;
    k.output_fields = output_fields;
    std::sort(k.output_fields.begin(), k.output_fields.end());
    k.gradout = gradout;
    k.inputres = inputres;
    return k;
}

// The geometry-derived part of the cache key, computed before we can decide
// hit or miss. Shape validation happens here too so a bad call fails before
// touching the cache.
CacheKey geometry_cache_key(const Config &cfg, torch::Tensor &src_pos,
                            torch::Tensor &rec_pos,
                            const std::vector<std::string> &output_fields,
                            int gradout, int inputres) {
    if (src_pos.dim() != 2 || src_pos.size(1) != 5) {
        throw std::invalid_argument("src_pos must have shape [allns, 5]");
    }
    if (rec_pos.dim() != 2 || rec_pos.size(1) != 8) {
        throw std::invalid_argument("rec_pos must have shape [allng, 8]");
    }
    torch::Tensor sp = src_pos.to(torch::kFloat32).contiguous();
    torch::Tensor rp = rec_pos.to(torch::kFloat32).contiguous();
    require_cpu(sp, "src_pos");
    require_cpu(rp, "rec_pos");

    int allns = static_cast<int>(sp.size(0));
    int allng = static_cast<int>(rp.size(0));
    std::vector<int> nsrc =
        counts_per_shot(sp.data_ptr<float>(), allns, 5, "src_pos");
    std::vector<int> nrec =
        counts_per_shot(rp.data_ptr<float>(), allng, 8, "rec_pos");
    if (nsrc.size() != nrec.size()) {
        throw std::invalid_argument(
            "src_pos and rec_pos must contain the same number of shot ids");
    }
    return make_cache_key(cfg, allns, allng, nsrc, nrec, output_fields,
                          gradout, inputres);
}

py::dict collect_data(model &m) {
    py::dict result;
    for (int i = 0; i < m.nvars; i++) {
        if (m.vars[i].to_output && m.vars[i].gl_varout) {
            torch::Tensor out =
                torch::from_blob(m.vars[i].gl_varout[0],
                                 {m.src_recs.allng, m.NT}, torch::kFloat32)
                    .clone();
            result[py::str(m.vars[i].name)] = out;
        }
    }
    return result;
}

// Zero the NAB-wide absorbing-boundary strip, matching SeisCL.py's
// _crop_boundary() (SeisCL/SeisCL.py) -- see that function's docstring for
// the full derivation; summarized here.
//
// Whether cropping is actually needed -- not just whether NAB rows exist --
// depends on *why* the boundary's gradient would be wrong, which differs by
// BACK_PROP_TYPE:
//  - BACK_PROP_TYPE==1 reconstructs the forward wavefield from a saved
//    boundary checkpoint by undoing each timestep's raw update, but never
//    divides back out the absorbing boundary's own damping (Cerjan's taper
//    multiply, or CPML's memory-variable recursion) that the forward pass
//    applied at that cell -- so the "reconstructed" field inside the
//    absorbing band silently diverges from the true one as backpropagation
//    proceeds, regardless of ABS_TYPE. Always cropped.
//  - BACK_PROP_TYPE==2 (DFT) never reconstructs anything: both the forward
//    and adjoint fields are computed by real time-stepping over the entire
//    grid. For ABS_TYPE==2 (Cerjan), a taper is just a real per-cell
//    multiply -- its own adjoint -- so the gradient inside the tapered band
//    is numerically valid there (only physically less interesting). Not
//    cropped by default. ABS_TYPE==1 (CPML) is a more delicate
//    discrete-adjoint question, not verified the same way, so it stays
//    cropped by default regardless of BACK_PROP_TYPE.
//
// Internal layout has no FDOH/NAB padding, matching update_adj{v,s}{2D,3D}.cl's
// `indp` addressing: X-slowest/Z-fastest flat in 2D (gl_par[x*NZ+z]), and
// X-slowest/Y-middle/Z-fastest in 3D (gl_par[x*(NY*NZ)+y*NZ+z]) -- matching
// what SeisCL.py's write_model() produces via np.transpose().
void crop_boundary(float *grad, const model &m) {
    if (m.BACK_PROP_TYPE != 1 && m.ABS_TYPE != 1) return;
    int nab = m.NAB;
    // FREESURF==0's top band is cropped for the general reason above.
    // FREESURF==1 has its own accurate free-surface gradient handling right
    // up to z=0, no crop needed. FREESURF==2's vacuum band
    // (supplied by the caller in the model, not created by the engine) is only FDOH deep --
    // NAB does not apply to that edge at all once a free surface is active
    // (CPML is already disabled there regardless of NAB) -- but those FDOH
    // rows hold physically meaningless nonzero gradient values (no real
    // material to invert for), so they still need masking, just a much
    // thinner band than FREESURF==0's.
    int ztop = (m.FREESURF == 2) ? m.FDOH : nab;
    if (m.NDIM == 2) {
        int nz = m.N[0], nx = m.N[1];
        for (int x = 0; x < nx; x++) {
            for (int z = 0; z < nz; z++) {
                bool in_boundary = (m.FREESURF != 1 && z < ztop) || z >= nz - nab ||
                                   x < nab || x >= nx - nab;
                if (in_boundary) grad[x * nz + z] = 0.0f;
            }
        }
    } else if (m.NDIM == 3) {
        int nz = m.N[0], ny = m.N[1], nx = m.N[2];
        for (int x = 0; x < nx; x++) {
            for (int y = 0; y < ny; y++) {
                for (int z = 0; z < nz; z++) {
                    bool in_boundary = (m.FREESURF != 1 && z < ztop) ||
                                       z >= nz - nab || y < nab || y >= ny - nab ||
                                       x < nab || x >= nx - nab;
                    if (in_boundary) grad[x * (ny * nz) + y * nz + z] = 0.0f;
                }
            }
        }
    }
}

py::dict collect_grads(model &m) {
    py::dict result;
    for (int i = 0; i < m.npars; i++) {
        if (m.pars[i].to_grad && m.pars[i].gl_grad) {
            crop_boundary(m.pars[i].gl_grad, m);
            torch::Tensor g = torch::from_blob(m.pars[i].gl_grad,
                                               {m.pars[i].num_ele},
                                               torch::kFloat32)
                                  .clone();
            result[py::str(m.pars[i].name)] = g;
        }
    }
    return result;
}

// A handle's pending checkpoint (see EngineHandle::pending_valid) is the
// only copy of a forward pass whose backward has not run yet -- for a
// single-shot run it lives solely in this handle's own buffers. Call this
// before anything overwrites those buffers (another forward reusing the
// handle, a rekey displacing it, or a backward call restoring a *different*
// checkpoint into it), or that pending forward's data is lost outright and
// its own eventual backward call can no longer recover it. A no-op if
// nothing is pending.
void flush_pending_checkpoint(EngineHandle &h) {
    if (!h.pending_valid) return;
    int flushed;
    if (h.m.CKPT_IN_MEMORY && h.m.CKPT_FILE_ID > 0) {
        flushed = checkpoint_image_to_disk(h.m.CKPT_FILE_ID,
                                           h.pending_ckpt.c_str());
    } else {
        flushed = checkpoint_flush(&h.m, &h.dev, h.pending_ckpt.c_str());
    }
    h.pending_valid = false;
    if (flushed) {
        throw std::runtime_error(
            "failed to flush a pending SeisCL checkpoint to " +
            h.pending_ckpt);
    }
}

// Build on a miss, refresh on a hit. Any failure evicts the handle rather
// than leaving a half-built or stale one for the next call to reuse.
// build_gradout is what the engine is *built* for -- it decides which
// kernels get compiled and which host/device buffers exist. run_gradout is
// which leg time_stepping() executes on this particular call. They differ
// for the forward leg of a gradient run: it is built grad-capable so that
// the matching backward call can reuse the very same handle (and hence the
// very same boundary buffers), but it still runs the forward path.
EngineHandle *prepare_engine(const Config &cfg, const CacheKey &key,
                             int build_gradout, int run_gradout, int inputres,
                             const py::dict &params, torch::Tensor &src,
                             torch::Tensor &src_pos, torch::Tensor &rec_pos,
                             const std::vector<std::string> &output_fields) {
    EngineCache &cache = global_engine_cache();
    bool was_hit = false;
    EngineHandle *h = cache.get_or_create(key, &was_hit);
    int state = 0;

    try {
        if (was_hit) {
            state = engine_refresh_srcrec(*h, src, src_pos, rec_pos);
            if (!state) state = engine_refresh_params(*h, params);
        } else {
            state = engine_build(*h, cfg, build_gradout, inputres, params, src,
                                 src_pos, rec_pos, output_fields);
        }
        h->m.GRADOUT = run_gradout;
        h->m.INPUTRES = inputres;
        if (!state) engine_reset_outputs(*h);
    } catch (...) {
        cache.evict(key);
        throw;
    }

    if (state) {
        cache.evict(key);
        throw std::runtime_error("SeisCL engine setup failed (state=" +
                                 std::to_string(state) + ")");
    }

    // "Every output field" was requested as an empty list, but which fields
    // that actually means is only known now. Re-register under the resolved
    // names so a later run_backward -- which necessarily names its residual
    // fields -- looks up this same handle.
    if (!was_hit && key.output_fields.empty()) {
        std::vector<std::string> resolved;
        for (int i = 0; i < h->m.nvars; i++) {
            if (h->m.vars[i].to_output) resolved.push_back(h->m.vars[i].name);
        }
        std::sort(resolved.begin(), resolved.end());
        CacheKey resolved_key = key;
        resolved_key.output_fields = resolved;
        std::unique_ptr<EngineHandle> displaced =
            cache.rekey(key, resolved_key);

        // The displaced handle's own forward already ran and (since it is
        // pending_valid) its matching backward has not -- its boundary
        // wavefield exists only in the buffers we are about to free via
        // this unique_ptr's destructor. Flush it to disk first, exactly as
        // run_forward() does when about to overwrite its own handle's
        // buffers with a new forward pass.
        if (displaced) flush_pending_checkpoint(*displaced);
    }
    return h;
}

py::dict run_forward(const Config &cfg, const py::dict &params,
                     torch::Tensor src, torch::Tensor src_pos,
                     torch::Tensor rec_pos,
                     const std::string &checkpoint_path,
                     const std::vector<std::string> &output_fields) {
    bool has_checkpoint = !checkpoint_path.empty();

    int inputres = has_checkpoint ? 1 : 0;
    // A forward that will be differentiated is built grad-capable, so its
    // handle is the same one run_backward will look up.
    int build_gradout = has_checkpoint ? 1 : 0;
    CacheKey key = geometry_cache_key(cfg, src_pos, rec_pos, output_fields,
                                      build_gradout, inputres);

    EngineHandle *h = prepare_engine(cfg, key, build_gradout, /*run_gradout=*/0,
                                     inputres, params, src, src_pos, rec_pos,
                                     output_fields);

    // The boundary wavefield only has to reach the matching backward pass,
    // which will reuse this very handle -- so for a single shot it can stay
    // in the buffers it already occupies instead of being round-tripped
    // through HDF5. Multi-shot runs still need the file: checkpoint_d2h()
    // runs per shot inside the shot loop, so only the last shot is ever
    // resident.
    // BACK_PROP_TYPE=2 keeps no checkpoint at all: its adjoint call
    // re-runs the forward pass to refill the frequency buffers, so
    // there is nothing to hand over or spill.
    bool uses_checkpoint = has_checkpoint && cfg.BACK_PROP_TYPE == 1;
    bool skip_file = uses_checkpoint && h->m.src_recs.ns == 1;

    // More than one shot: every shot's wavefield has to survive until the
    // adjoint pass, which the shared buffers cannot do on their own (they
    // are reused per shot). Keep the per-shot datasets, but back them with
    // RAM rather than disk when they fit.
    bool in_memory = false;
    if (uses_checkpoint && !skip_file) {
        std::size_t bytes =
            checkpoint_bytes_per_shot(*h) * h->m.src_recs.ns;
        in_memory = checkpoint_fits_in_memory(bytes);
    }

    // A previous forward on this handle whose backward never ran is about to
    // have its buffers overwritten. Persist its checkpoint now, so that
    // backward can still fall back to the file.
    if (h->pending_valid && h->pending_ckpt != checkpoint_path) {
        try {
            flush_pending_checkpoint(*h);
        } catch (...) {
            global_engine_cache().evict(key);
            throw;
        }
    }

    h->m.SKIP_CHECKPOINT_FILE = skip_file ? 1 : 0;
    h->m.CKPT_IN_MEMORY = in_memory ? 1 : 0;

    struct filenames files;
    std::memset(&files, 0, sizeof(files));
    if (has_checkpoint) {
        std::strncpy(files.checkpoint, checkpoint_path.c_str(),
                    sizeof(files.checkpoint) - 1);
    }

    int state = time_stepping(&h->m, &h->dev, files);
    if (state) {
        global_engine_cache().evict(key);
        throw std::runtime_error("SeisCL forward pass failed (state=" +
                                 std::to_string(state) + ")");
    }

    if (skip_file || in_memory) {
        h->pending_valid = true;
        h->pending_ckpt = checkpoint_path;
    }

    return collect_data(h->m);
}

py::dict run_backward(const Config &cfg, const py::dict &params,
                      torch::Tensor src, torch::Tensor src_pos,
                      torch::Tensor rec_pos, const py::dict &residuals,
                      const std::string &checkpoint_path) {
    if (checkpoint_path.empty()) {
        throw std::invalid_argument(
            "checkpoint_path is required for the backward pass "
            "(must be the path used by the matching run_forward call)");
    }

    // Restrict to_output the same way run_forward does, to the fields
    // residuals were actually supplied for -- to_output only drives
    // seismogram-output allocation/collection here (unused by the
    // gradient path itself), but keeping the requested-field set narrow
    // avoids invoking the varsout kernel-generation path with an
    // unnecessarily large field combination (see set_srcrec's comment).
    std::vector<std::string> output_fields;
    output_fields.reserve(residuals.size());
    for (auto item : residuals) {
        output_fields.push_back(py::cast<std::string>(item.first));
    }

    CacheKey key = geometry_cache_key(cfg, src_pos, rec_pos, output_fields,
                                      /*gradout=*/1, /*inputres=*/1);

    EngineHandle *h =
        prepare_engine(cfg, key, /*build_gradout=*/1, /*run_gradout=*/1,
                       /*inputres=*/1, params, src, src_pos, rec_pos,
                       output_fields);

    // Residuals (grad_output from the downstream torch loss) take the
    // place of the reference data readhdf5() would normally have supplied
    // for res_calc() to consume; INPUTRES=1 skips res_calc() and uses
    // gl_var_res directly (src/time_stepping.c:843-848).
    try {
        for (auto item : residuals) {
            std::string name = py::cast<std::string>(item.first);
            torch::Tensor t = py::cast<torch::Tensor>(item.second)
                                  .to(torch::kFloat32)
                                  .contiguous();
            require_cpu(t, "Residual " + name);
            variable *var = find_var(h->m, name);
            if (!var) {
                throw std::invalid_argument("Unknown field: " + name);
            }
            int64_t expected =
                static_cast<int64_t>(h->m.src_recs.allng) * h->m.NT;
            if (t.numel() != expected) {
                throw std::invalid_argument(
                    "Residual " + name + " has " +
                    std::to_string(t.numel()) + " elements, expected " +
                    std::to_string(expected));
            }
            // engine_build() allocated these before Init_CUDA, which is the
            // only point at which they can be created (see the comment
            // there); here we only overwrite the values.
            if (!var->gl_var_res) {
                throw std::runtime_error(
                    "no residual buffer allocated for field " + name);
            }
            std::memcpy(var->gl_var_res[0], t.data_ptr<float>(),
                        sizeof(float) * expected);
        }
    } catch (...) {
        global_engine_cache().evict(key);
        throw;
    }

    // This handle may be resident with a *different* call's still-pending
    // checkpoint -- e.g. two differentiable forwards sharing this exact
    // CacheKey ran before either backward did, and the cache only has room
    // for one handle per key. Restoring this call's own checkpoint below
    // (or finding it already resident, for a single-shot from_memory hit)
    // is about to overwrite these buffers either way, so flush the other
    // one first or its own eventual backward call loses it outright.
    if (h->pending_valid && h->pending_ckpt != checkpoint_path) {
        try {
            flush_pending_checkpoint(*h);
        } catch (...) {
            global_engine_cache().evict(key);
            throw;
        }
    }

    // If this is the handle the matching forward ran on and nothing has
    // overwritten its buffers since, the boundary wavefield is already where
    // the adjoint pass needs it. Otherwise fall back to the file, which the
    // forward either wrote itself or had flushed to it.
    // BACK_PROP_TYPE=2 never writes a checkpoint -- time_stepping() re-runs
    // the forward pass on this call instead -- so there is nothing to locate.
    bool uses_checkpoint = cfg.BACK_PROP_TYPE == 1;
    bool from_memory =
        !uses_checkpoint ||
        (h->pending_valid && h->pending_ckpt == checkpoint_path);
    // Single shot keeps the wavefield in the engine's own buffers; multiple
    // shots keep the per-shot datasets in a RAM-backed HDF5 file.
    bool via_ram_file = uses_checkpoint && from_memory &&
                        h->m.CKPT_IN_MEMORY && h->m.CKPT_FILE_ID > 0;
    if (uses_checkpoint && !from_memory) {
        std::ifstream f(checkpoint_path.c_str());
        if (!f.good()) {
            throw std::runtime_error(
                "SeisCL checkpoint " + checkpoint_path +
                " is neither resident in the engine cache nor on disk; the "
                "matching forward pass may have been discarded");
        }
        // Whatever the forward did, this run must read the real file.
        h->m.CKPT_IN_MEMORY = 0;
    }
    h->m.SKIP_CHECKPOINT_FILE =
        (uses_checkpoint && from_memory && !via_ram_file) ? 1 : 0;

    struct filenames files;
    std::memset(&files, 0, sizeof(files));
    std::strncpy(files.checkpoint, checkpoint_path.c_str(),
                sizeof(files.checkpoint) - 1);

    int state = time_stepping(&h->m, &h->dev, files);
    // Consumed either way: the adjoint pass has overwritten the buffers.
    h->pending_valid = false;
    if (state) {
        global_engine_cache().evict(key);
        throw std::runtime_error("SeisCL backward pass failed (state=" +
                                 std::to_string(state) + ")");
    }

    return collect_grads(h->m);
}

}  // namespace
}  // namespace seiscl_torch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using seiscl_torch::Config;

    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("N", &Config::N)
        .def_readwrite("ND", &Config::ND)
        .def_readwrite("dh", &Config::dh)
        .def_readwrite("dt", &Config::dt)
        .def_readwrite("NT", &Config::NT)
        .def_readwrite("FDORDER", &Config::FDORDER)
        .def_readwrite("MAXRELERROR", &Config::MAXRELERROR)
        .def_readwrite("FREESURF", &Config::FREESURF)
        .def_readwrite("NAB", &Config::NAB)
        .def_readwrite("ABS_TYPE", &Config::ABS_TYPE)
        .def_readwrite("VPPML", &Config::VPPML)
        .def_readwrite("FPML", &Config::FPML)
        .def_readwrite("NPOWER", &Config::NPOWER)
        .def_readwrite("K_MAX_CPML", &Config::K_MAX_CPML)
        .def_readwrite("abpc", &Config::abpc)
        .def_readwrite("L", &Config::L)
        .def_readwrite("f0", &Config::f0)
        .def_readwrite("par_type", &Config::par_type)
        .def_readwrite("FP16", &Config::FP16)
        .def_readwrite("restype", &Config::restype)
        .def_readwrite("GRADSRCOUT", &Config::GRADSRCOUT)
        .def_readwrite("HOUT", &Config::HOUT)
        .def_readwrite("BACK_PROP_TYPE", &Config::BACK_PROP_TYPE)
        .def_readwrite("gradfreqs", &Config::gradfreqs)
        .def_readwrite("dft_osamp", &Config::dft_osamp)
        .def_readwrite("tmin", &Config::tmin)
        .def_readwrite("nmax_dev", &Config::nmax_dev)
        .def_readwrite("pref_device_type", &Config::pref_device_type);

    m.def("run_forward", &seiscl_torch::run_forward,
         "Run SeisCL forward modeling",
         py::arg("cfg"), py::arg("params"), py::arg("src"),
         py::arg("src_pos"), py::arg("rec_pos"),
         py::arg("checkpoint_path") = std::string(),
         py::arg("output_fields") = std::vector<std::string>());

    m.def("run_backward", &seiscl_torch::run_backward,
         "Run SeisCL's adjoint pass given residuals, reading the "
         "checkpoint written by a matching run_forward call",
         py::arg("cfg"), py::arg("params"), py::arg("src"),
         py::arg("src_pos"), py::arg("rec_pos"), py::arg("residuals"),
         py::arg("checkpoint_path"));

    m.def("set_engine_cache_size",
         [](std::size_t n) {
             seiscl_torch::global_engine_cache().set_max_size(n);
         },
         "Set how many built engines (CUDA context + compiled kernels + "
         "device buffers) may be kept alive for reuse. Each one pins GPU "
         "memory. Default 2. The most recent entry is never trimmed, so 0 "
         "still leaves one behind -- use clear_engine_cache() to free it.",
         py::arg("n"));

    m.def("engine_cache_size",
         []() { return seiscl_torch::global_engine_cache().size(); },
         "Number of engines currently held in the reuse cache.");

    m.def("set_checkpoint_policy",
         [](const std::string &policy, py::object budget, py::object fraction) {
             if (policy == "auto") {
                 seiscl_torch::g_ckpt_policy = seiscl_torch::CkptPolicy::kAuto;
             } else if (policy == "memory") {
                 seiscl_torch::g_ckpt_policy = seiscl_torch::CkptPolicy::kMemory;
             } else if (policy == "file") {
                 seiscl_torch::g_ckpt_policy = seiscl_torch::CkptPolicy::kFile;
             } else {
                 throw std::invalid_argument(
                     "checkpoint policy must be 'auto', 'memory' or 'file'");
             }
             // None leaves the budget alone; 0 restores "derive from free
             // memory", which is the default.
             if (!budget.is_none()) {
                 seiscl_torch::g_ckpt_budget = budget.cast<std::size_t>();
             }
             if (!fraction.is_none()) {
                 double f = fraction.cast<double>();
                 if (!(f > 0.0) || f > 1.0) {
                     throw std::invalid_argument(
                         "ram_fraction must be in (0, 1]");
                 }
                 seiscl_torch::g_ckpt_ram_fraction = f;
             }
         },
         "Where the boundary checkpoint of a multi-shot gradient run is "
         "kept. 'auto' (the default) keeps it in RAM while it fits, and "
         "spills to a file otherwise; 'memory' and 'file' force one or the "
         "other. What 'fits' means: by default ram_fraction (0.5) of the "
         "memory the kernel reports as available right now, or budget_bytes "
         "if you set an explicit cap (0 restores the free-memory rule). "
         "Single-shot runs keep the wavefield in the engine's own buffers "
         "and ignore all of this.",
         py::arg("policy") = std::string("auto"),
         py::arg("budget_bytes") = py::none(),
         py::arg("ram_fraction") = py::none());

    m.def("available_ram_bytes", &seiscl_torch::available_ram_bytes,
         "Memory the kernel currently reports as available, which is what "
         "the 'auto' checkpoint policy sizes itself against. 0 if unknown.");

    m.def("clear_engine_cache",
         []() { seiscl_torch::global_engine_cache().clear(); },
         "Free every cached engine and its GPU resources. The next call "
         "rebuilds from scratch.");

    m.def("_shutdown_engine_cache",
         []() { seiscl_torch::global_engine_cache().clear(); },
         "atexit hook: frees cached engines while the CUDA driver is still "
         "alive, rather than relying on static destruction order.");
}
