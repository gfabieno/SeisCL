// Signature identifying a reusable engine build (see engine_cache.h).
//
// Two calls sharing a CacheKey allocate identical buffers and compile
// identical kernels, so the second can reuse the first's EngineHandle and
// only refresh values. Buffer sizing is a pure function of the Config
// fields below (via the set_size callbacks in src/assign_modeling_case.c)
// plus the geometry counts; kernel selection additionally depends on the
// requested output fields and on whether adjoint kernels are needed.

#ifndef SEISCL_TORCH_CACHE_KEY_H
#define SEISCL_TORCH_CACHE_KEY_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace seiscl_torch {

struct CacheKey {
    std::vector<int64_t> N;
    int ND;
    float dh;
    float dt;
    int NT;
    int FDORDER;
    int MAXRELERROR;
    int FREESURF;
    int NAB;
    int ABS_TYPE;
    float VPPML;
    float FPML;
    float NPOWER;
    float K_MAX_CPML;
    float abpc;
    int L;
    // Not just L: FL's values feed eta() during Init_cst(), which a cache
    // hit skips entirely (engine_refresh_params re-runs Init_model_values
    // only). Two runs differing solely in FL would otherwise silently share
    // the first one's relaxation times.
    std::vector<float> FL;
    float f0;
    int par_type;
    int FP16;
    int restype;
    int GRADSRCOUT;
    int HOUT;
    int BACK_PROP_TYPE;
    int nmax_dev;
    int pref_device_type;

    int allns;
    int allng;
    // Per-shot source/receiver counts, not just their totals. These size and
    // chain every per-shot pointer array in the engine (seiscl_set_srcrec()'s
    // src/src_pos/rec_pos, var_alloc_out()'s gl_varout/gl_varin/gl_var_res,
    // and the device geometry buffers sized from nsmax/ngmax). Keying on the
    // full distribution means a reused build always has identical chaining,
    // so refreshing geometry is a pure value copy; any redistribution -- even
    // one holding allns/allng fixed -- rebuilds instead.
    std::vector<int> nsrc;
    std::vector<int> nrec;

    std::vector<std::string> output_fields;  // sorted by the builder

    // Both flags are baked into the build, not just the run: GRADOUT gates
    // adjoint kernel compilation (Init_OpenCL.c:1072, :1126) and the gl_grad
    // host allocation (clmodel.c:64), and GRADOUT||INPUTRES gates the
    // boundary-storage savebnd kernel (assign_modeling_case.c:952). A handle
    // built with one combination cannot serve another, so the forward leg of
    // a gradient run (GRADOUT=0, INPUTRES=1) and its backward leg (GRADOUT=1,
    // INPUTRES=1) are deliberately separate cache entries -- which is why the
    // cache defaults to holding two.
    int gradout;
    int inputres;

    bool operator==(const CacheKey &o) const {
        // Floats compare exactly on purpose: repeat calls pass the same
        // Config values bit-for-bit, and tolerancing here would merge runs
        // that genuinely differ.
        return N == o.N && ND == o.ND && dh == o.dh && dt == o.dt &&
               NT == o.NT && FDORDER == o.FDORDER &&
               MAXRELERROR == o.MAXRELERROR && FREESURF == o.FREESURF &&
               NAB == o.NAB && ABS_TYPE == o.ABS_TYPE && VPPML == o.VPPML &&
               FPML == o.FPML && NPOWER == o.NPOWER &&
               K_MAX_CPML == o.K_MAX_CPML && abpc == o.abpc && L == o.L &&
               FL == o.FL &&
               f0 == o.f0 && par_type == o.par_type && FP16 == o.FP16 &&
               restype == o.restype && GRADSRCOUT == o.GRADSRCOUT &&
               HOUT == o.HOUT && BACK_PROP_TYPE == o.BACK_PROP_TYPE &&
               nmax_dev == o.nmax_dev &&
               pref_device_type == o.pref_device_type && allns == o.allns &&
               allng == o.allng && nsrc == o.nsrc && nrec == o.nrec &&
               output_fields == o.output_fields && gradout == o.gradout &&
               inputres == o.inputres;
    }
};

struct CacheKeyHash {
    template <typename T>
    static void combine(std::size_t &seed, const T &v) {
        seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    std::size_t operator()(const CacheKey &k) const {
        std::size_t h = 0;
        for (int64_t n : k.N) combine(h, n);
        combine(h, k.ND);
        combine(h, k.dh);
        combine(h, k.dt);
        combine(h, k.NT);
        combine(h, k.FDORDER);
        combine(h, k.MAXRELERROR);
        combine(h, k.FREESURF);
        combine(h, k.NAB);
        combine(h, k.ABS_TYPE);
        combine(h, k.VPPML);
        combine(h, k.FPML);
        combine(h, k.NPOWER);
        combine(h, k.K_MAX_CPML);
        combine(h, k.abpc);
        combine(h, k.L);
        for (float f : k.FL) combine(h, f);
        combine(h, k.f0);
        combine(h, k.par_type);
        combine(h, k.FP16);
        combine(h, k.restype);
        combine(h, k.GRADSRCOUT);
        combine(h, k.HOUT);
        combine(h, k.BACK_PROP_TYPE);
        combine(h, k.nmax_dev);
        combine(h, k.pref_device_type);
        combine(h, k.allns);
        combine(h, k.allng);
        for (int n : k.nsrc) combine(h, n);
        for (int n : k.nrec) combine(h, n);
        for (const std::string &f : k.output_fields) combine(h, f);
        combine(h, k.gradout);
        combine(h, k.inputres);
        return h;
    }
};

}  // namespace seiscl_torch

#endif  // SEISCL_TORCH_CACHE_KEY_H
