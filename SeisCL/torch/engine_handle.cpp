#include "engine_handle.h"

#include <cstring>
#include <stdexcept>

namespace seiscl_torch {

EngineHandle::EngineHandle() { std::memset(&m, 0, sizeof(model)); }

EngineHandle::~EngineHandle() {
    // time_stepping() deliberately leaves a RAM-backed checkpoint open for
    // the adjoint pass; this is where it finally goes away.
    if (m.CKPT_FILE_ID > 0) H5Fclose(m.CKPT_FILE_ID);
    if (built || dev) Free_OpenCL(&m, dev);
}

std::size_t checkpoint_bytes_per_shot(const EngineHandle &h) {
    std::size_t total = 0;
    for (int d = 0; d < static_cast<int>(h.m.NUM_DEVICES); d++) {
        for (int i = 0; i < h.dev[d].nvars; i++) {
            const variable &v = h.dev[d].vars[i];
            if (v.cl_var.host) total += sizeof(float) * v.num_ele;
            if (v.cl_varbnd.host) total += v.cl_varbnd.sizepin;
            // buf1/buf2 are each stored twice, before and after the device
            // read (the "h" and "d" datasets in checkpoint_d2h).
            if (v.cl_buf1.host) total += 2 * v.cl_buf1.size;
            if (v.cl_buf2.host) total += 2 * v.cl_buf2.size;
        }
    }
    return total;
}

// parameter/variable lookup that doesn't rely on get_par()/get_var()
// (src/clmodel.c), which return the address of a local stack variable
// when a name isn't found -- fine within the C engine's own call
// conventions (the caller never dereferences a miss there) but not safe to
// rely on for error checking from here.
parameter *find_par(model &m, const std::string &name) {
    for (int i = 0; i < m.npars; i++) {
        if (name == m.pars[i].name) return &m.pars[i];
    }
    return nullptr;
}

variable *find_var(model &m, const std::string &name) {
    for (int i = 0; i < m.nvars; i++) {
        if (name == m.vars[i].name) return &m.vars[i];
    }
    return nullptr;
}

// .to(kFloat32) converts dtype but never moves device, so a CUDA tensor
// would reach the std::memcpy calls below as a device pointer -- undefined
// behavior rather than an error. All inputs must be host-resident.
void require_cpu(const torch::Tensor &t, const std::string &what) {
    if (t.is_cuda()) {
        throw std::invalid_argument(
            what + " must be a CPU tensor (got a CUDA tensor); call .cpu() "
                   "on it first");
    }
}

void apply_config(model &m, const Config &cfg, int gradout, int inputres) {
    if (cfg.N.empty() || cfg.N.size() > MAX_DIMS) {
        throw std::invalid_argument(
            "cfg.N must have between 1 and " + std::to_string(MAX_DIMS) +
            " dimensions");
    }
    for (size_t i = 0; i < cfg.N.size(); i++) {
        m.N[i] = static_cast<int>(cfg.N[i]);
    }
    m.ND = cfg.ND;
    m.NDIM = (cfg.ND == 3) ? 3 : 2;
    m.dh = cfg.dh;
    m.dt = cfg.dt;
    m.NT = cfg.NT;
    m.tmax = cfg.NT;
    m.FDORDER = cfg.FDORDER;
    m.FDOH = cfg.FDORDER / 2;
    m.MAXRELERROR = cfg.MAXRELERROR;
    m.FREESURF = cfg.FREESURF;
    m.NAB = cfg.NAB;
    m.ABS_TYPE = cfg.ABS_TYPE;
    m.VPPML = cfg.VPPML;
    m.FPML = cfg.FPML;
    m.NPOWER = cfg.NPOWER;
    m.K_MAX_CPML = cfg.K_MAX_CPML;
    m.abpc = cfg.abpc;
    m.L = cfg.L;
    m.f0 = cfg.f0;
    m.par_type = cfg.par_type;
    m.FP16 = cfg.FP16;
    m.restype = cfg.restype;
    m.GRADSRCOUT = cfg.GRADSRCOUT;
    m.HOUT = cfg.HOUT;
    m.BACK_PROP_TYPE = cfg.BACK_PROP_TYPE;
    m.nmax_dev = cfg.nmax_dev;
    m.pref_device_type = static_cast<DEVICE_TYPE>(cfg.pref_device_type);

    if (inputres && cfg.BACK_PROP_TYPE != 1) {
        // The checkpoint file this binding relies on for the INPUTRES=1
        // two-call protocol is only written by save_bnd() when
        // BACK_PROP_TYPE==1 (src/time_stepping.c:798-800,
        // src/assign_modeling_case.c:952). BACK_PROP_TYPE==2 (DFT/
        // frequency-domain gradient) also needs a "gradfreqs" constants
        // array this binding doesn't populate -- unsupported for now.
        throw std::invalid_argument(
            "SeisCL/torch currently only supports cfg.BACK_PROP_TYPE=1 "
            "(boundary-storage gradient)");
    }

    if (cfg.L > 0 && static_cast<int>(cfg.FL.size()) != cfg.L) {
        // Caught here rather than at the memcpy below so the message names
        // the real problem. Without it, a zero-filled FL propagates through
        // eta() into dt/eta[l] and the run returns Inf/NaN seismograms with
        // no diagnostic at all.
        throw std::invalid_argument(
            "cfg.L=" + std::to_string(cfg.L) + " requires cfg.FL to hold "
            "exactly " + std::to_string(cfg.L) + " attenuation-mechanism "
            "center frequencies (got " + std::to_string(cfg.FL.size()) + ")");
    }

    m.GRADOUT = gradout;
    m.INPUTRES = inputres;
    m.VARSOUT = 1;

    // Single process, single group: no MPI (mirrors SeisCL_MPI.c:main()'s
    // __NOMPI__ branch).
    m.GID = 0;
    m.GNP = 1;
    m.LID = 0;
    m.LNP = 1;
    m.NGROUP = 1;
    m.MYGROUPID = 0;
    m.MYLOCALID = 0;
    m.MPI_NPROC_SHOT = 1;
    m.NLOCALP = 1;
    m.MPI_INIT = 0;
}

void set_params(model &m, const py::dict &params) {
    for (auto item : params) {
        std::string name = py::cast<std::string>(item.first);
        // CUDA parameter tensors are accepted and copied down here. This is
        // a convenience, not a fast path: Init_model_values() runs
        // set_par_scale/transform/check_stability on the host gl_par array
        // before it is uploaded, so the values have to pass through host
        // memory either way. Doing it here just saves every caller holding
        // GPU model parameters from writing .cpu() at the call site.
        torch::Tensor t = py::cast<torch::Tensor>(item.second)
                              .to(torch::kCPU, torch::kFloat32)
                              .contiguous();
        parameter *par = find_par(m, name);
        if (!par) {
            throw std::invalid_argument("Unknown parameter: " + name);
        }
        if (t.numel() != par->num_ele) {
            throw std::invalid_argument(
                "Parameter " + name + " has " + std::to_string(t.numel()) +
                " elements, expected " + std::to_string(par->num_ele));
        }
        std::memcpy(par->gl_par, t.data_ptr<float>(),
                    sizeof(float) * par->num_ele);
    }
}

void set_srcrec(model &m, torch::Tensor &src, torch::Tensor &src_pos,
                torch::Tensor &rec_pos,
                const std::vector<std::string> &output_fields) {
    src = src.to(torch::kFloat32).contiguous();
    src_pos = src_pos.to(torch::kFloat32).contiguous();
    rec_pos = rec_pos.to(torch::kFloat32).contiguous();
    require_cpu(src, "src");
    require_cpu(src_pos, "src_pos");
    require_cpu(rec_pos, "rec_pos");

    if (src_pos.dim() != 2 || src_pos.size(1) != 5) {
        throw std::invalid_argument("src_pos must have shape [allns, 5]");
    }
    if (rec_pos.dim() != 2 || rec_pos.size(1) != 8) {
        throw std::invalid_argument("rec_pos must have shape [allng, 8]");
    }
    int allns = static_cast<int>(src_pos.size(0));
    int allng = static_cast<int>(rec_pos.size(0));
    if (src.numel() != static_cast<int64_t>(allns) * m.NT) {
        throw std::invalid_argument("src must have shape [allns, NT]");
    }

    int state = seiscl_set_srcrec(&m, src_pos.data_ptr<float>(), allns,
                                  src.data_ptr<float>(),
                                  rec_pos.data_ptr<float>(), allng);
    if (state) {
        throw std::runtime_error("seiscl_set_srcrec failed");
    }

    // The set of to_output fields feeds automatic_kernels.c's varsout
    // kernel generation (a single combined kernel built from the specific
    // set of requested fields, not one kernel per field) -- request only
    // what's asked for rather than always every declared field, matching
    // how a real caller (e.g. SeisCL.py's seisout presets) would use it.
    if (output_fields.empty()) {
        for (int i = 0; i < m.nvars; i++) {
            m.vars[i].to_output = 1;
        }
    } else {
        for (int i = 0; i < m.nvars; i++) {
            m.vars[i].to_output = 0;
        }
        for (const auto &name : output_fields) {
            variable *var = find_var(m, name);
            if (!var) {
                throw std::invalid_argument("Unknown output field: " + name);
            }
            var->to_output = 1;
        }
    }
}

int engine_build(EngineHandle &h, const Config &cfg, int gradout, int inputres,
                 const py::dict &params, torch::Tensor &src,
                 torch::Tensor &src_pos, torch::Tensor &rec_pos,
                 const std::vector<std::string> &output_fields) {

    apply_config(h.m, cfg, gradout, inputres);

    int state = assign_modeling_case(&h.m);
    if (state) return state;

    // May throw; the caller evicts the handle, whose destructor frees
    // whatever assign_modeling_case already allocated.
    set_params(h.m, params);
    set_srcrec(h.m, src, src_pos, rec_pos, output_fields);

    // FL is normally read from the csts file (read_hdf5.c, dataset "/FL");
    // with no file, fill it here. assign_modeling_case() has allocated the
    // array by now (append_cst(m,"FL","/FL",m->L,NULL)) and Init_cst() is
    // what runs eta()'s transform over it, so this must sit between the
    // two. Same shape as the geometry/parameter uploads just above.
    if (!state && h.m.L > 0) {
        constants *fl = get_cst(h.m.csts, h.m.ncsts, "FL");
        if (!fl || fl->num_ele != h.m.L) {
            throw std::runtime_error(
                "FL constants array was not registered as expected");
        }
        std::memcpy(fl->gl_cst, cfg.FL.data(), sizeof(float) * h.m.L);
    }

    if (!state) state = Init_cst(&h.m);
    if (!state) state = Init_data(&h.m);

    // Residual buffers must exist before Init_CUDA: it snapshots each
    // variable struct into the per-device copy (di->vars[i]=m->vars[i]),
    // and initialize_adj() then dereferences that copy's gl_var_res for
    // every to_output variable (time_stepping.c:637). Allocating after the
    // build would leave the device copy holding a NULL pointer.
    if (!state && gradout) {
        for (int i = 0; i < h.m.nvars && !state; i++) {
            if (h.m.vars[i].to_output && !h.m.vars[i].gl_var_res) {
                state = var_alloc_out(&h.m.vars[i].gl_var_res, &h.m);
            }
        }
        for (int i = 0; i < h.m.ntvars && !state; i++) {
            if (h.m.trans_vars[i].to_output &&
                !h.m.trans_vars[i].gl_var_res) {
                state = var_alloc_out(&h.m.trans_vars[i].gl_var_res, &h.m);
            }
        }
    }

    if (!state) state = Init_model(&h.m);
    if (!state) state = Init_CUDA(&h.m, &h.dev);

    if (!state) h.built = true;
    return state;
}

int engine_refresh_params(EngineHandle &h, const py::dict &params) {

    // Same host memcpy as the build path, straight into the existing
    // gl_par arrays.
    set_params(h.m, params);

    // Re-derive scaling/transforms/FP16 from those raw values. Includes
    // check_stability, which depends on the values (CFL) and so must run
    // again even though the grid is unchanged.
    int state = Init_model_values(&h.m);
    if (state) return state;

    // Upload. Init_CUDA does this once at build time (Init_OpenCL.c:783);
    // here it has to happen on every refresh. dev[d].pars[i].cl_par.host
    // still aliases into h.m.pars[i].gl_par (Init_OpenCL.c:761), so the
    // values just written are what gets sent.
    for (int d = 0; d < static_cast<int>(h.m.NUM_DEVICES); d++) {
#ifndef __SEISCL__
        // Mirrors the per-device context switch Init_CUDA does before
        // touching a device's buffers (Init_OpenCL.c:371).
        if (cuCtxSetCurrent(h.dev[d].context)) return 1;
#endif
        for (int i = 0; i < h.m.npars; i++) {
            state = clbuf_send(&h.dev[d].queue, &h.dev[d].pars[i].cl_par);
            if (state) return state;
        }
    }
    return 0;
}

int engine_refresh_srcrec(EngineHandle &h, torch::Tensor &src,
                          torch::Tensor &src_pos, torch::Tensor &rec_pos) {

    src = src.to(torch::kFloat32).contiguous();
    src_pos = src_pos.to(torch::kFloat32).contiguous();
    rec_pos = rec_pos.to(torch::kFloat32).contiguous();
    require_cpu(src, "src");
    require_cpu(src_pos, "src_pos");
    require_cpu(rec_pos, "rec_pos");

    int allns = static_cast<int>(src_pos.size(0));
    int allng = static_cast<int>(rec_pos.size(0));
    sources_records &sr = h.m.src_recs;

    // The per-shot counts are part of the cache key, so a reused handle has
    // exactly this geometry's shape and all the pointer chaining set up at
    // build time still holds. Only the values change, and seiscl_set_srcrec()
    // is allocate-only with no free-first guard, so write in place rather
    // than calling it again.
    std::memcpy(sr.src_pos[0], src_pos.data_ptr<float>(),
                sizeof(float) * allns * 5);
    std::memcpy(sr.src[0], src.data_ptr<float>(),
                sizeof(float) * allns * h.m.NT);
    std::memcpy(sr.rec_pos[0], rec_pos.data_ptr<float>(),
                sizeof(float) * allng * 8);

    // dev[d].src_recs is a struct copy made at build time (Init_OpenCL.c:800),
    // but its src/src_pos/rec_pos members are the same pointers as h.m's, so
    // the writes above are already visible per device. time_stepping()
    // re-uploads cl_src/cl_src_pos/cl_rec_pos per shot unconditionally
    // (time_stepping.c:563-575), so no device transfer is needed here.
    return 0;
}

void engine_reset_outputs(EngineHandle &h) {
    // reduce_seis() accumulates into gl_varout with += (time_stepping.c:64),
    // relying on it being freshly zeroed. That used to come free from the
    // per-call GMALLOC; on a reused handle it has to be done explicitly, or
    // each call adds to the previous one's seismogram.
    //
    // Device-side wavefield and gradient buffers do NOT need this: the
    // init_f/init_adj/grads.init kernels re-zero them every call already
    // (time_stepping.c:605, :650, :727). gl_grad is likewise overwritten
    // wholesale by clbuf_read (:986) rather than accumulated.
    std::size_t nbytes =
        sizeof(float) * h.m.src_recs.allng * h.m.NT;
    for (int i = 0; i < h.m.nvars; i++) {
        if (h.m.vars[i].gl_varout && h.m.vars[i].gl_varout[0]) {
            std::memset(h.m.vars[i].gl_varout[0], 0, nbytes);
        }
    }
    for (int i = 0; i < h.m.ntvars; i++) {
        if (h.m.trans_vars[i].gl_varout && h.m.trans_vars[i].gl_varout[0]) {
            std::memset(h.m.trans_vars[i].gl_varout[0], 0, nbytes);
        }
    }
    // Also accumulated across a run (residuals.c:293-302).
    h.m.rms = 0.0f;
    h.m.rmsnorm = 0.0f;
}

}  // namespace seiscl_torch
