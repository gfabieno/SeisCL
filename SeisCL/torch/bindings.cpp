// pybind11/torch C++ extension exposing SeisCL's forward-modeling and
// adjoint-gradient engine (seiscl_core, see src/seiscl_api.c and
// CMakeLists.txt's BUILD_TORCH_CORE target) as in-memory calls on CPU
// torch tensors. No subprocess, no MPI, and no full-model/geometry/output
// HDF5 files -- only run_backward() touches HDF5 at all, and only for a
// small internal checkpoint file (see below). SeisCL/torch/op.py wraps
// run_forward()/run_backward() in a torch.autograd.Function.
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

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "F.h"
}

namespace {

// User-facing scalar configuration. Mirrors the subset of SeisCL.py's
// constructor kwargs needed to drive one forward/gradient call; geometry,
// parameters and residuals are passed separately (see run_forward/
// run_backward below). Defaults match SeisCL.py's constructor where
// applicable.
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
    int nmax_dev = 1;
    // Only meaningful for OpenCL builds (src/Init_OpenCL.c's device-type
    // fallback logic is entirely #ifdef __SEISCL__); inert here since
    // seiscl_core is always built for CUDA. Kept for API familiarity with
    // SeisCL.py.
    int pref_device_type = 4;
};

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
        torch::Tensor t = py::cast<torch::Tensor>(item.second)
                              .to(torch::kFloat32)
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

py::dict collect_grads(model &m) {
    py::dict result;
    for (int i = 0; i < m.npars; i++) {
        if (m.pars[i].to_grad && m.pars[i].gl_grad) {
            torch::Tensor g = torch::from_blob(m.pars[i].gl_grad,
                                               {m.pars[i].num_ele},
                                               torch::kFloat32)
                                  .clone();
            result[py::str(m.pars[i].name)] = g;
        }
    }
    return result;
}

py::dict run_forward(const Config &cfg, const py::dict &params,
                     torch::Tensor src, torch::Tensor src_pos,
                     torch::Tensor rec_pos,
                     const std::string &checkpoint_path,
                     const std::vector<std::string> &output_fields) {
    bool has_checkpoint = !checkpoint_path.empty();

    model m;
    std::memset(&m, 0, sizeof(model));
    device *dev = nullptr;

    apply_config(m, cfg, /*gradout=*/0, /*inputres=*/has_checkpoint ? 1 : 0);

    int state = assign_modeling_case(&m);
    if (!state) {
        try {
            set_params(m, params);
            set_srcrec(m, src, src_pos, rec_pos, output_fields);
        } catch (...) {
            Free_OpenCL(&m, dev);
            throw;
        }
    }

    if (!state) state = Init_cst(&m);
    if (!state) state = Init_data(&m);
    if (!state) state = Init_model(&m);
    if (!state) state = Init_CUDA(&m, &dev);

    struct filenames files;
    std::memset(&files, 0, sizeof(files));
    if (has_checkpoint) {
        std::strncpy(files.checkpoint, checkpoint_path.c_str(),
                    sizeof(files.checkpoint) - 1);
    }

    if (!state) state = time_stepping(&m, &dev, files);

    py::dict result;
    if (!state) {
        result = collect_data(m);
    }
    Free_OpenCL(&m, dev);

    if (state) {
        throw std::runtime_error("SeisCL forward pass failed (state=" +
                                 std::to_string(state) + ")");
    }
    return result;
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

    model m;
    std::memset(&m, 0, sizeof(model));
    device *dev = nullptr;

    apply_config(m, cfg, /*gradout=*/1, /*inputres=*/1);

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

    int state = assign_modeling_case(&m);
    if (!state) {
        try {
            set_params(m, params);
            set_srcrec(m, src, src_pos, rec_pos, output_fields);
        } catch (...) {
            Free_OpenCL(&m, dev);
            throw;
        }
    }

    if (!state) state = Init_cst(&m);
    if (!state) state = Init_data(&m);

    // Residuals (grad_output from the downstream torch loss) take the
    // place of the reference data readhdf5() would normally have supplied
    // for res_calc() to consume; INPUTRES=1 skips res_calc() and uses
    // gl_var_res directly (src/time_stepping.c:843-848).
    if (!state) {
        try {
            for (auto item : residuals) {
                std::string name = py::cast<std::string>(item.first);
                torch::Tensor t = py::cast<torch::Tensor>(item.second)
                                      .to(torch::kFloat32)
                                      .contiguous();
                variable *var = find_var(m, name);
                if (!var) {
                    throw std::invalid_argument("Unknown field: " + name);
                }
                int64_t expected =
                    static_cast<int64_t>(m.src_recs.allng) * m.NT;
                if (t.numel() != expected) {
                    throw std::invalid_argument(
                        "Residual " + name + " has " +
                        std::to_string(t.numel()) + " elements, expected " +
                        std::to_string(expected));
                }
                if (var_alloc_out(&var->gl_var_res, &m)) {
                    throw std::runtime_error(
                        "var_alloc_out failed for residual " + name);
                }
                std::memcpy(var->gl_var_res[0], t.data_ptr<float>(),
                            sizeof(float) * expected);
            }
        } catch (...) {
            Free_OpenCL(&m, dev);
            throw;
        }
    }

    if (!state) state = Init_model(&m);
    if (!state) state = Init_CUDA(&m, &dev);

    struct filenames files;
    std::memset(&files, 0, sizeof(files));
    std::strncpy(files.checkpoint, checkpoint_path.c_str(),
                sizeof(files.checkpoint) - 1);

    if (!state) state = time_stepping(&m, &dev, files);

    py::dict result;
    if (!state) {
        result = collect_grads(m);
    }
    Free_OpenCL(&m, dev);

    if (state) {
        throw std::runtime_error("SeisCL backward pass failed (state=" +
                                 std::to_string(state) + ")");
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
        .def_readwrite("nmax_dev", &Config::nmax_dev)
        .def_readwrite("pref_device_type", &Config::pref_device_type);

    m.def("run_forward", &run_forward, "Run SeisCL forward modeling",
         py::arg("cfg"), py::arg("params"), py::arg("src"),
         py::arg("src_pos"), py::arg("rec_pos"),
         py::arg("checkpoint_path") = std::string(),
         py::arg("output_fields") = std::vector<std::string>());

    m.def("run_backward", &run_backward,
         "Run SeisCL's adjoint pass given residuals, reading the "
         "checkpoint written by a matching run_forward call",
         py::arg("cfg"), py::arg("params"), py::arg("src"),
         py::arg("src_pos"), py::arg("rec_pos"), py::arg("residuals"),
         py::arg("checkpoint_path"));
}
