"""The 4-point material averages of the host (src/assign_modeling_case.c) and their transposes in the gradient
(src/calc_grad.c), compiled from the engine sources and checked on small grids.

ave_arithmetic_tau (tausipkp, tausipjp, tausjpkp) and ave_harmonic_mu (muipkp, ..., host path of SH and of
SEISCL_HOST_AVERAGE=1) average four cells where the four exist and copy the cell on the last plane of each of
their two directions. Until 2026-11-26 they did not reset the plane bounds between the two directions: the
last plane of the second direction was copied only where it crossed the first (one row in 2-D), and the rest
of it stayed 0. The transposes (grad_T_edge_copy) added the cell twice where the two planes cross.

Checks, for 2-D and 3-D grids and the direction pairs of SeisCL (x-z, x-y, y-z, and the 2-point SH pairs):
  1. the output on each last plane equals the input there (threshold: exact);
  2. dot test <A x, y> = <x, A^T y>, with A the average linearised by a central difference and A^T
     grad_T_arithmetic_tau / grad_T_harmonic_mu (relative difference below 1e-4 for tau, 1e-3 for mu,
     float32 averages).
The device versions (prep_params.cl, average_params.cl, grad_post.cl) are checked against these host
functions bit for bit by test_torch_binding.py (test_param_prep_matches_host) and SEISCL_HOST_GRAD=1.

Usage: python SeisCL/tests/test_average_edges.py   (needs gcc; no GPU)
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

SRC = os.environ.get("SEISCL_SRC",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))


def c_function(path, name):
    """The text of the C function `name` (signature to its closing brace) in file path."""
    text = open(path).read()
    m = re.search(r"^(static\s+)?(void|int)\s+%s\s*\(" % name, text, re.M)
    if not m:
        raise RuntimeError("no function %s in %s" % (name, path))
    i = text.index("{", m.end())
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[m.start():j + 1]
    raise RuntimeError("unbalanced braces in %s" % name)


HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define GMALLOC(x, sz) do{ x = calloc(1, sz); if(!x) state = 1; }while(0)
#define GFree(x) do{ free(x); x = NULL; }while(0)
%(functions)s
/* argv: op (0 tau, 1 mu, 2 tau_T, 3 mu_T) ndim nz ny nx d00 d01 d02 d10 d11 d12; stdin: inputs; stdout: result */
int main(int argc, char ** argv){
    int op = atoi(argv[1]), ndim = atoi(argv[2]);
    int N[3] = {atoi(argv[3]), atoi(argv[4]), atoi(argv[5])};
    int dir[2][3] = {{atoi(argv[6]), atoi(argv[7]), atoi(argv[8])},
                     {atoi(argv[9]), atoi(argv[10]), atoi(argv[11])}};
    int Nd[3];
    if (ndim==3){ Nd[0]=N[0]; Nd[1]=N[1]; Nd[2]=N[2]; } else { Nd[0]=N[0]; Nd[1]=N[2]; Nd[2]=1; }
    int n = N[0]*N[1]*N[2], i;
    float * a = calloc(n, 4), * b = calloc(n, 4), * c = calloc(n, 4);
    for (i=0;i<n;i++) if (scanf("%%f", &a[i])!=1) return 2;
    if (op>=2) for (i=0;i<n;i++) if (scanf("%%f", &b[i])!=1) return 2;
    if (op==0) ave_arithmetic_tau(a, c, Nd, ndim, dir);
    if (op==1) ave_harmonic_mu(a, c, Nd, ndim, dir);
    if (op==2) grad_T_arithmetic_tau(b, c, Nd, ndim, dir);          /* a unused */
    if (op==3) grad_T_harmonic_mu(a, b, c, Nd, ndim, dir);          /* pin a, gin b */
    for (i=0;i<n;i++) printf("%%.9e\n", c[i]);
    return 0;
}
"""


def build(d):
    amc = os.path.join(SRC, "assign_modeling_case.c")
    cg = os.path.join(SRC, "calc_grad.c")
    funcs = [c_function(amc, "ave_arithmetic_tau"), c_function(amc, "ave_harmonic_mu")]
    for name in ("scatter_order", "edge_of", "grad_T_edge_copy", "grad_T_harmonic_mu",
                 "grad_T_arithmetic_tau"):
        try:
            funcs.append(c_function(cg, name))
        except RuntimeError:
            if name != "edge_of":       # added with the fix
                raise
    src = os.path.join(d, "ave.c")
    with open(src, "w") as f:
        f.write(HARNESS % {"functions": "\n\n".join(funcs)})
    exe = os.path.join(d, "ave")
    subprocess.run(["gcc", "-O0", "-o", exe, src, "-lm"], check=True)
    return exe


def run(exe, op, ndim, shape, dirs, *arrays):
    nz, ny, nx = shape
    args = [exe, str(op), str(ndim), str(nz), str(ny), str(nx)] + [str(v) for d in dirs for v in d]
    inp = "\n".join("%.9e" % v for a in arrays for v in a.ravel())
    out = subprocess.run(args, input=inp, capture_output=True, text=True, check=True).stdout
    return np.array(out.split(), dtype=np.float64)


def last_planes(shape, dirs):
    """Mask (flat, x-slowest) of the cells on the last plane of either direction."""
    nz, ny, nx = shape
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    m = np.zeros(shape, bool)
    for d in dirs:
        if d[2]:
            m |= i == nx - 1
        if d[1]:
            m |= j == ny - 1
        if d[0]:
            m |= k == nz - 1
    return m.transpose(2, 1, 0).ravel()       # C index i*NY*NZ + j*NZ + k


CASES = [  # (name, ndim, (nz, ny, nx), dirs in (z, y, x))
    ("2-D x-z (tausipkp)", 2, (7, 1, 6), ((0, 0, 1), (1, 0, 0))),
    ("3-D x-z (tausipkp)", 3, (5, 4, 6), ((0, 0, 1), (1, 0, 0))),
    ("3-D x-y (tausipjp)", 3, (5, 4, 6), ((0, 0, 1), (0, 1, 0))),
    ("3-D y-z (tausjpkp)", 3, (5, 4, 6), ((0, 1, 0), (1, 0, 0))),
    ("SH z only (tausjpkp)", 2, (7, 1, 6), ((1, 0, 0), (0, 0, 0))),
    ("SH x only (tausipjp)", 2, (7, 1, 6), ((0, 0, 1), (0, 0, 0))),
]


def check_case(exe, name, ndim, shape, dirs, rng):
    ok = True
    n = int(np.prod(shape))
    x = np.float32(1.0 + 0.5 * rng.random(n)).astype(np.float64)   # exact through the text I/O
    edge = last_planes(shape, dirs)
    for op, op_t, label in ((0, 2, "tau"), (1, 3, "mu")):
        out = run(exe, op, ndim, shape, dirs, x)
        bad = int(np.sum(np.float32(out[edge]) != np.float32(x[edge])))
        good = bad == 0
        ok &= good
        print("%s  %-22s %s: last planes hold the cell (%d of %d cells differ, threshold 0)"
              % ("PASS" if good else "FAIL", name, label, bad, int(edge.sum())))
        # dot test, A linearised at x by a central difference
        dx = rng.standard_normal(n)
        y = rng.standard_normal(n)
        h = 0.1 if op == 0 else 1e-2      # tau is linear; mu: O(h^2) truncation
        ax = (run(exe, op, ndim, shape, dirs, x + h * dx) - run(exe, op, ndim, shape, dirs, x - h * dx)) / (2 * h)
        aty = run(exe, op_t, ndim, shape, dirs, x, y)
        lhs, rhs = float(ax @ y), float(dx @ aty)
        rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
        good = rel < (1e-3 if op == 1 else 1e-4)
        ok &= good
        print("%s  %-22s %s: dot test <Ax,y> %.6e  <x,A^T y> %.6e  rel %.1e (threshold %s)"
              % ("PASS" if good else "FAIL", name, label, lhs, rhs, rel, "1e-3" if op == 1 else "1e-4"))
    return ok


def test_average_edges():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(dir=os.environ.get("SEISCL_TEST_TMP", os.getcwd())) as d:
        exe = build(d)
        ok = all([check_case(exe, *c, rng) for c in CASES])
    assert ok, "averages or their transposes are wrong on the last planes (see FAIL lines)"


if __name__ == "__main__":
    try:
        test_average_edges()
    except AssertionError as e:
        print(e)
        sys.exit(1)
