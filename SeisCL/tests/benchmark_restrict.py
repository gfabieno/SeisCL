#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B benchmark: baseline SeisCL_MPI vs. a build with `const`/`restrict`
pointer qualifiers added to the 8 core forward/adjoint elastic update
kernels (src/update_{v,s}{2D,3D}.cl, src/update_adj{v,s}{2D,3D}.cl).

Two modes are timed for each dimensionality, since both forward and adjoint
kernels were modified:
  - "forward": a plain forward-modeling run (withgrad=False), exercising
    only update_v/update_s.
  - "grad": a two-execute forward-then-adjoint run using the boundary-
    storage checkpoint (back_prop_type=1, the default). The first execute()
    forward-models and streams the boundary checkpoint to disk (INPUTRES=1,
    GRADOUT=0); the second execute() loads that checkpoint and runs *only*
    the adjoint pass (INPUTRES=1, GRADOUT=1 skips re-running the forward
    loop -- see src/time_stepping.c:731-859). Only the second execute() is
    timed for this mode, since that's the one that isolates
    update_adjv/update_adjs.

For each mode, output is compared between one baseline run and one restrict
run (max abs diff / relative RMSE) to confirm the qualifiers didn't change
results -- they shouldn't (restrict/const are compiler hints, not algorithm
changes), but GPU instruction scheduling could in principle shift floating-
point reduction order at the ULP level, so this checks a tight tolerance
rather than asserting bit-exact equality.

This machine's integrated GPU (or whatever OpenCL device SeisCL_MPI picks)
is not representative of production HPC hardware -- this gives a relative
speedup signal and a correctness check, not an absolute performance number.

Usage:
  python benchmark_restrict.py \
      --baseline-bin /path/to/build_baseline/SeisCL_MPI \
      --restrict-bin /path/to/build_restrict/SeisCL_MPI \
      --dims all --n 100 --nt 200 --repeats 5
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from SeisCL.SeisCL import SeisCL

FDORDER = 4
# Holberg stability constant for FDORDER=4, and the sqrt(ND) CFL factor,
# mirroring src/assign_modeling_case.c:801,815,817.
_HOLBERG_GAMMA = 1.184614


def stable_dt(dh, vp, ND, safety=0.9):
    dtstable = dh / (_HOLBERG_GAMMA * np.sqrt(3.0 if ND == 3 else 2.0) * vp)
    return safety * dtstable


def build_model(ND, N, nt, workdir):
    """A homogeneous elastic whole-space, single shot, crossline receiver
    array -- mirrors the geometry pattern used by
    SeisCL/tests/test_analytics.py and SeisCL/tests/benchmark_deepwave.py.
    """
    vp, vs, rho = 3500.0, 2000.0, 2000.0
    dh = 8.0
    nab = max(16, N // 6)

    seis = SeisCL()
    seis.with_mpi = False
    seis.ND = ND
    seis.N = np.array([N for _ in range(ND)])
    seis.dh = dh
    seis.dt = stable_dt(dh, vp, ND)
    seis.NT = nt
    seis.f0 = 15
    seis.freesurf = 0
    seis.FDORDER = FDORDER
    seis.abs_type = 2
    seis.nab = nab
    seis.abpc = 3
    seis.seisout = 1
    seis.workdir = workdir

    nbuf = FDORDER * 2
    sx = (nab + nbuf) * dh
    sy = N // 2 * dh if ND == 3 else 0.0
    sz = N // 2 * dh
    offmin = 5 * dh
    offmax = (N - nab - nbuf) * dh - sx
    gx = np.arange(sx + offmin, sx + offmax, dh)
    gy = gx * 0 + sy
    gz = gx * 0 + sz
    gsid = gx * 0
    gid = np.arange(len(gx))

    seis.src_pos_all = np.stack([[sx], [sy], [sz], [0], [2]], axis=0)
    seis.rec_pos = np.stack(
        [gx, gy, gz, gsid, gid, gx * 0 + 2, gx * 0, gx * 0], axis=0)
    seis.rec_pos_all = seis.rec_pos

    params = {
        "vp": np.full(seis.N, vp, dtype=np.float32),
        "vs": np.full(seis.N, vs, dtype=np.float32),
        "rho": np.full(seis.N, rho, dtype=np.float32),
    }
    return seis, params


def with_binary_on_path(binary_path):
    binary_dir = os.path.dirname(os.path.abspath(binary_path))
    old_path = os.environ["PATH"]
    os.environ["PATH"] = binary_dir + os.pathsep + old_path
    return old_path


def run_forward(binary_path, ND, N, nt, workdir):
    old_path = with_binary_on_path(binary_path)
    try:
        seis, params = build_model(ND, N, nt, workdir)
        seis.set_forward(seis.src_pos_all[3, :], params, withgrad=False)
        seis.write_data({})
        os.chdir(workdir)
        t0 = time.perf_counter()
        seis.execute()
        elapsed = time.perf_counter() - t0
        data = seis.read_data()
        return elapsed, data
    finally:
        os.environ["PATH"] = old_path


def run_grad(binary_path, ND, N, nt, workdir):
    """Two-execute checkpoint flow; returns (forward_time, adjoint_time, grad)."""
    old_path = with_binary_on_path(binary_path)
    try:
        seis, params = build_model(ND, N, nt, workdir)
        seis.set_forward(seis.src_pos_all[3, :], params, withgrad=False)
        seis.inputres = 1  # forward pass streams the boundary checkpoint to disk
        seis.write_csts(workdir)
        seis.write_data({})
        os.chdir(workdir)
        t0 = time.perf_counter()
        seis.execute()
        fwd_elapsed = time.perf_counter() - t0
        data = seis.read_data()

        seis.set_backward(residuals=data)
        t0 = time.perf_counter()
        seis.execute()
        adj_elapsed = time.perf_counter() - t0
        grad = seis.read_grad()
        return fwd_elapsed, adj_elapsed, grad
    finally:
        os.environ["PATH"] = old_path


def compare_arrays(name, arrays_a, arrays_b, rtol):
    ok = True
    for a, b in zip(arrays_a, arrays_b):
        a, b = np.asarray(a), np.asarray(b)
        denom = np.maximum(np.abs(a).max(), 1e-30)
        max_abs_diff = np.abs(a - b).max()
        rel_diff = max_abs_diff / denom
        status = "OK" if rel_diff <= rtol else "MISMATCH"
        if rel_diff > rtol:
            ok = False
        print(f"    {name}: max_abs_diff={max_abs_diff:.3e} "
              f"rel_diff={rel_diff:.3e} [{status}]")
    return ok


def timeit(fn, repeats):
    times = [fn() for _ in range(repeats)]
    return np.mean(times), np.std(times)


def bench_dim(ND, args):
    print(f"\n=== {ND}D (N={args.n}, NT={args.nt}, repeats={args.repeats}) ===")
    workdir_base = os.path.join(args.workdir, f"{ND}d")
    rtol = args.rtol

    # ---- forward-only mode ----
    print("-- forward-only (update_v / update_s) --")
    wd_a = os.path.join(workdir_base, "fwd_baseline")
    wd_b = os.path.join(workdir_base, "fwd_restrict")
    os.makedirs(wd_a, exist_ok=True)
    os.makedirs(wd_b, exist_ok=True)

    t_a, data_a = run_forward(args.baseline_bin, ND, args.n, args.nt, wd_a)
    t_b, data_b = run_forward(args.restrict_bin, ND, args.n, args.nt, wd_b)
    ok = compare_arrays("data", data_a, data_b, rtol)

    mean_a, std_a = timeit(
        lambda: run_forward(args.baseline_bin, ND, args.n, args.nt, wd_a)[0],
        args.repeats)
    mean_b, std_b = timeit(
        lambda: run_forward(args.restrict_bin, ND, args.n, args.nt, wd_b)[0],
        args.repeats)
    speedup = mean_a / mean_b if mean_b > 0 else float("nan")
    print(f"    baseline: {mean_a*1000:.1f} +/- {std_a*1000:.1f} ms")
    print(f"    restrict: {mean_b*1000:.1f} +/- {std_b*1000:.1f} ms")
    print(f"    speedup:  {speedup:.3f}x  correctness={'PASS' if ok else 'FAIL'}")

    # ---- forward+gradient mode (adjoint kernels timed separately) ----
    print("-- gradient (update_adjv / update_adjs, adjoint pass only timed) --")
    wd_a = os.path.join(workdir_base, "grad_baseline")
    wd_b = os.path.join(workdir_base, "grad_restrict")
    os.makedirs(wd_a, exist_ok=True)
    os.makedirs(wd_b, exist_ok=True)

    fwd_a, adj_a, grad_a = run_grad(args.baseline_bin, ND, args.n, args.nt, wd_a)
    fwd_b, adj_b, grad_b = run_grad(args.restrict_bin, ND, args.n, args.nt, wd_b)
    ok = compare_arrays("grad", grad_a, grad_b, rtol)

    def adj_only(binary, wd):
        return run_grad(binary, ND, args.n, args.nt, wd)[1]

    mean_a, std_a = timeit(lambda: adj_only(args.baseline_bin, wd_a), args.repeats)
    mean_b, std_b = timeit(lambda: adj_only(args.restrict_bin, wd_b), args.repeats)
    speedup = mean_a / mean_b if mean_b > 0 else float("nan")
    print(f"    baseline (adjoint pass): {mean_a*1000:.1f} +/- {std_a*1000:.1f} ms")
    print(f"    restrict (adjoint pass): {mean_b*1000:.1f} +/- {std_b*1000:.1f} ms")
    print(f"    speedup:  {speedup:.3f}x  correctness={'PASS' if ok else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bin", required=True,
                        help="Path to the unmodified SeisCL_MPI binary")
    parser.add_argument("--restrict-bin", required=True,
                        help="Path to the const/restrict-qualified SeisCL_MPI binary")
    parser.add_argument("--dims", choices=["2d", "3d", "all"], default="all")
    parser.add_argument("--n", type=int, default=100,
                        help="Grid size per dimension")
    parser.add_argument("--nt", type=int, default=200, help="Number of timesteps")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=1e-5,
                        help="Relative-difference tolerance for the correctness check")
    parser.add_argument("--workdir", default="./benchmark_restrict_wd")
    args = parser.parse_args()

    args.baseline_bin = os.path.abspath(args.baseline_bin)
    args.restrict_bin = os.path.abspath(args.restrict_bin)
    args.workdir = os.path.abspath(args.workdir)
    os.makedirs(args.workdir, exist_ok=True)

    print("NOTE: this machine's OpenCL device is not representative of "
          "production HPC hardware -- treat these numbers as a relative "
          "speedup signal and a correctness check, not an absolute "
          "performance benchmark.")

    dims = [2, 3] if args.dims == "all" else [int(args.dims[0])]
    for ND in dims:
        bench_dim(ND, args)


if __name__ == "__main__":
    main()
