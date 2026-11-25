#!/usr/bin/env python3
"""Domain decomposition: a shot split along x over several GPUs and/or MPI processes must give the
seismograms and the gradient (back_prop_type 1) of the same shot on one GPU.

The source is put one cell after the first cut between subdomains and the receivers one cell before
the last cut, where a subdomain and its neighbour share cells (the halo, FDORDER/2 cells wide).
For every FP16 level, the split run is compared with one GPU at the same FP16:

    relative difference = ||split - one GPU|| / ||one GPU||
    pass: <= 1e-5 at FP16 0, <= 1e-2 at FP16 1 and 2 (half precision, decision of the plan
    perf-ddecomp-2026-11-20)

Also checked: a 2-D split into subdomains narrower than 48 cells must be refused at start-up
(the update kernels need 3 work groups of 16 cells along x); and, with several MPI processes, when
only one process is refused (uneven split) the others must stop too, not wait for it forever.

Usage (one GPU per device or process, through the gpu tool):
    python test_decomp.py --bin <SeisCL_MPI> --workdir <folder> --nd 3 --ngpu 3 --nmpi 1
    python test_decomp.py ... --nd 2 --ngpu 1 --nmpi 4 --abs_type 1 --fp16 0,1,2
    python test_decomp.py ... --nd 3 --L 3 --bpt 2     (viscoelastic: DFT gradient only)
Prints one PASS / FAIL line per check; exits with 1 if any check fails.
"""
import argparse
import os
import signal
import subprocess
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from SeisCL.SeisCL import SeisCL  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--bin", required=True, help="SeisCL_MPI to run (absolute path)")
parser.add_argument("--workdir", required=True, help="folder for the SeisCL files")
parser.add_argument("--nd", type=int, default=3, help="2 or 3 dimensions")
parser.add_argument("--ngpu", type=int, default=3, help="devices per process")
parser.add_argument("--nmpi", type=int, default=1, help="MPI processes per shot")
parser.add_argument("--L", type=int, default=0, help="attenuation mechanisms (0: elastic)")
parser.add_argument("--abs_type", type=int, default=2, help="1: CPML, 2: Cerjan")
parser.add_argument("--bpt", type=int, default=1,
                    help="back_prop_type: 1 (backpropagation, elastic only) or 2 (DFT, at gradfreqs)")
parser.add_argument("--fp16", default="0,1,2", help="FP16 levels to test")
parser.add_argument("--src-gap", type=int, default=1, help="source this many cells after the first cut")
parser.add_argument("--rec-gap", type=int, default=1, help="receivers this many cells before the last cut")
parser.add_argument("--save", default="", help="write both runs' seismograms and gradients to this .npz (diagnosis)")
parser.add_argument("--no-refusal", action="store_true", help="skip the refusal check")
args = parser.parse_args()

TOL = {0: 1e-5, 1: 1e-2, 2: 1e-2}
nfail = 0


def report(ok, text):
    global nfail
    nfail += not ok
    print("%s  %s" % ("PASS" if ok else "FAIL", text), flush=True)


def subdomain_cuts(nx, nmpi, ngpu):
    """Global x of every cut, as Init_OpenCL.c splits the grid: first over the processes, then over
    the devices of each process; the first subdomains get one more cell when it does not divide."""
    widths = []
    for p in range(nmpi):
        wp = nx // nmpi + (1 if p < nx % nmpi else 0)
        widths += [wp // ngpu + (1 if d < wp % ngpu else 0) for d in range(ngpu)]
    return list(np.cumsum(widths)[:-1]), widths


# ---- model and geometry -----------------------------------------------------------------------
os.makedirs(args.workdir, exist_ok=True)
seis = SeisCL(workdir=args.workdir)
seis.progname = os.path.abspath(args.bin)
seis.with_mpi = True
seis.dh, seis.dt, seis.NT, seis.FDORDER = 10, 0.0008, 875, 8
seis.abs_type = args.abs_type
seis.ND = args.nd
seis.freesurf = 0
seis.N = np.array([64, 256]) if args.nd == 2 else np.array([64, 64, 96])
seis.L = args.L
seis.back_prop_type = args.bpt
if args.bpt == 2:
    seis.gradfreqs = np.array([4.0, 8.0, 12.0, 16.0])   # Hz, around the source peak
if args.L > 1:
    seis.FL = np.array([5.0, 15.0, 30.0, 45.0][:args.L])
elif args.L == 1:
    seis.FL = np.array(15)

nx = seis.N[-1]
cuts, widths = subdomain_cuts(nx, args.nmpi, args.ngpu)
print("N %s, subdomain widths %s, cuts at x = %s" % (list(seis.N), widths, cuts))
xsrc = (cuts[0] if cuts else nx // 3) + args.src_gap     # default: one cell after the first cut
xrec = (cuts[-1] if cuts else 2 * nx // 3) - args.rec_gap    # default: one cell before the last cut
# (no cut: --ngpu 1 --nmpi 1 runs the one-GPU case twice, to check it alone)
z0 = seis.nab + 5
ymid = seis.N[1] // 2 if args.nd == 3 else 0
seis.src_pos_all = np.array([[xsrc * seis.dh], [ymid * seis.dh], [z0 * seis.dh], [0], [100]])
nrec = seis.N[0] - seis.nab - z0 - 5
seis.rec_pos_all = np.zeros((8, nrec))
seis.rec_pos_all[0, :] = xrec * seis.dh
seis.rec_pos_all[1, :] = ymid * seis.dh
seis.rec_pos_all[2, :] = (z0 + np.arange(nrec)) * seis.dh
seis.rec_pos_all[4, :] = np.arange(nrec) + 1
seis.src_all = None
print("source at x = %d, receivers at x = %d (padded halo: %d cells)" % (xsrc, xrec, seis.FDORDER // 2))

pars = {"vp": np.zeros(seis.N) + 3500, "vs": np.zeros(seis.N) + 2000, "rho": np.zeros(seis.N) + 2000}
if args.L > 0:
    pars["taup"] = np.zeros(seis.N) + 0.1
    pars["taus"] = np.zeros(seis.N) + 0.1
names = list(pars)

# observed data: the same shot with a faster box in the middle, so the residuals are not zero
true = {k: v.copy() for k, v in pars.items()}
true["vp"][tuple(slice(n // 2 - 5, n // 2 + 5) for n in seis.N)] = 4000
seis.file_din = os.path.join(args.workdir, "SeisCL_din.mat")
seis.FP16 = 0
seis.nmax_dev, seis.NP, seis.MPI_NPROC_SHOT = 1, 1, 1
seis.set_forward(seis.src_pos_all[3, :], true, withgrad=False)
seis.execute()
seis.write_data({"p": seis.read_data()[0]}, filename="SeisCL_din.mat")


def run(ngpu, nmpi, fp16, withgrad):
    seis.nmax_dev, seis.NP, seis.MPI_NPROC_SHOT, seis.FP16 = ngpu, nmpi, nmpi, fp16
    seis.set_forward(seis.src_pos_all[3, :], pars, withgrad=withgrad)
    seis.execute()
    data = [np.array(d) for d in seis.read_data()]
    grad = [np.array(g) for g in seis.read_grad()] if withgrad else None
    return data, grad


def reldiff(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# ---- split against one GPU --------------------------------------------------------------------
split = "%d GPU x %d process" % (args.ngpu, args.nmpi)
for fp16 in [int(f) for f in args.fp16.split(",")]:
    tag = ("ND %d L %d abs_type %d bpt %d FP16 %d, %s vs 1 GPU"
           % (args.nd, args.L, args.abs_type, args.bpt, fp16, split))
    try:
        d1, g1 = run(1, 1, fp16, True)
        ds, gs = run(args.ngpu, args.nmpi, fp16, True)
    except Exception as e:
        lines = [ln for ln in str(e).splitlines() if ln.strip() and "warning" not in ln.lower()]
        print("      " + "\n      ".join(lines[-12:]))
        report(False, "%s: run failed: %s" % (tag, lines[-1][:200] if lines else ""))
        continue
    if args.save:
        np.savez(args.save.replace(".npz", "_fp%d.npz" % fp16), names=names, cuts=cuts, xsrc=xsrc, xrec=xrec,
                 **{"d1_%d" % n: a for n, a in enumerate(d1)}, **{"ds_%d" % n: a for n, a in enumerate(ds)},
                 **{"g1_" + n: a for n, a in zip(names, g1)}, **{"gs_" + n: a for n, a in zip(names, gs)})
    err = max(reldiff(a, b) for a, b in zip(ds, d1))
    report(np.isfinite(err) and err <= TOL[fp16],
           "%s, seismograms: rel. diff %.2e <= %.0e" % (tag, err, TOL[fp16]))
    for name, a, b in zip(names, gs, g1):
        err = reldiff(a, b)
        if err > TOL[fp16]:
            dx = np.abs(a - b).reshape(-1, a.shape[-1]).max(axis=0)   # max over z (and y), per x
            print("      %s: |difference| largest at x = %s (cuts %s)"
                  % (name, list(np.argsort(dx)[::-1][:4]), cuts))
        report(np.isfinite(err) and err <= TOL[fp16],
               "%s, gradient %-4s: rel. diff %.2e <= %.0e" % (tag, name, err, TOL[fp16]))

# ---- refusals: an error exit (status 1-127), with the message, never a crash or a hang ----------
def run_raw(ngpu, nmpi, limit=120, withgrad=False):
    """Run the current model and return (exit status, stderr), or (None, "") after a hang
    (the whole process group is killed). A status >= 128 is a signal (139: segmentation fault)."""
    seis.nmax_dev, seis.NP, seis.MPI_NPROC_SHOT, seis.FP16 = ngpu, nmpi, nmpi, 0
    seis.set_forward(seis.src_pos_all[3, :], pars, withgrad=withgrad)
    p = subprocess.Popen(seis.callcmd(seis.workdir), shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = p.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        return None, ""
    return p.returncode, err.decode()


def refused(nx, ngpu, nmpi, what):
    global pars
    seis.N = np.array([64, nx])
    seis.src_pos_all[0, :] = (nx // 2 - 4) * seis.dh     # outside the absorbing layers
    seis.rec_pos_all[0, :] = (nx // 2 + 4) * seis.dh
    pars = {k: np.zeros(seis.N) + v.flat[0] for k, v in pars.items()}
    rc, err = run_raw(ngpu, nmpi)
    msg = [ln for ln in err.splitlines() if ln.startswith("Error")][:1]
    if rc is None:
        report(False, "%s: still running after 120 s (hang)" % what)
    else:
        report(0 < rc < 128 and "cannot run the update kernels" in err,
               "%s (widths %s): error exit, status %d (1-127), message: %s"
               % (what, subdomain_cuts(nx, nmpi, ngpu)[1], rc, msg[0][:80] if msg else "none"))


if not args.no_refusal and args.nd == 2 and args.ngpu * args.nmpi > 1:
    refused(16 * args.ngpu * args.nmpi, args.ngpu, args.nmpi, "2-D subdomains of 16 cells refused")
if not args.no_refusal and args.nd == 2 and args.nmpi > 1:
    # nx = 48 nmpi - 1: the first processes get 48 cells, the last 47 (< 48: refused alone); the
    # others must stop too (MPI_Abort), not wait for it at the first exchange
    refused(48 * args.nmpi - 1, 1, args.nmpi, "one of %d processes refused, all stop" % args.nmpi)
if not args.no_refusal and args.nmpi > 1:
    # process 0 alone reads the input files: a gradient without its observed data fails there,
    # before start-up, while the others wait in the broadcasts of Init_MPI; all must stop
    din = seis.file_din
    seis.file_din = os.path.join(args.workdir, "no_such_data.mat")
    rc, err = run_raw(args.ngpu, args.nmpi, withgrad=True)
    seis.file_din = din
    msg = [ln for ln in err.splitlines() if ln.startswith("Error")][:1]
    report(rc is not None and 0 < rc < 128,
           "process 0 fails before start-up (no data file), all %d stop: %s"
           % (args.nmpi, "still running after 120 s (hang)" if rc is None else
              "status %d (1-127), message: %s" % (rc, msg[0][:70] if msg else "none")))

print("test_decomp: %d check(s) failed" % nfail)
sys.exit(1 if nfail else 0)
