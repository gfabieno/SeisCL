"""Crosswell viscoelastic experiment: a circular inclusion in Qp/Qs.

Building block for the viscoelastic FWI notebook. Kept as a plain module so
the notebook can import it rather than duplicating the setup, and so it can be
run headless as a smoke test:

    SEISCL_BIN=<build-ocl> python visco_crosswell.py

WHY OpenCL AND NOT CUDA
-----------------------
A viscoelastic (L>0) gradient only works on an **OpenCL** build today. The
on-device DFT correlation is registered for `m->L==0` only
(assign_modeling_case.c), so an L>0 run falls back to the host `calc_grad()`,
which is `#ifdef __SEISCL__` and a no-op stub in the CUDA build. The result is
a silently **zero** gradient -- no error, no warning. Measured on a 60x60
crosswell model:

    build      L=0  vp=6.147e-19  vs=1.021e-18  rho=4.046e-19   OK
    build      L=1  vp=0.000e+00  vs=0.000e+00  rho=0.000e+00   ZERO
    build-ocl  L=0  vp=6.147e-19  vs=7.129e-19  rho=3.590e-19   OK
    build-ocl  L=1  vp=6.364e-19  vs=7.208e-19  rho=3.518e-19
                    taup=6.031e-15 taus=1.891e-15               OK

The same follows for `SeisCL.torch`, which is CUDA-only: it cannot produce a
viscoelastic gradient at all until the viscoelastic DFT device kernel is
written (todo.md item 6's remaining half). back_prop_type=1 is not an
alternative either -- it rejects L>0 outright, because reverse-time
reconstruction of a dissipative medium is unconditionally unstable
(todo.md item 17).
"""
import os

import numpy as np

from SeisCL import SeisCL

# Crosswell: two vertical wells, sources in the left one, receivers in the
# right one. Small enough to iterate on interactively.
NZ, NX = 120, 80
DH = 5.0
DT = 4e-4
NT = 1500
NAB = 16
F0 = 60.0

VP0, VS0, RHO0 = 2500.0, 1400.0, 2200.0
# tau is SeisCL's relaxation-time parameter, not Q -- SeisCL/Q_tau.py converts.
# Background is near-elastic; the inclusion is strongly attenuating.
TAUP0, TAUS0 = 0.002, 0.002
TAUP_INC, TAUS_INC = 0.05, 0.05

# One relaxation mechanism centred on the source frequency.
L = 1
FL = np.array([F0])


def circular_inclusion(nz=NZ, nx=NX, radius=12.0, dvp=0.0, dvs=0.0,
                       dtaup=TAUP_INC - TAUP0, dtaus=TAUS_INC - TAUS0):
    """Background model plus a circular anomaly at the centre.

    Defaults perturb only the attenuation (taup/taus), leaving vp/vs/rho
    homogeneous -- the cleanest first target for a viscoelastic inversion,
    since any recovered structure has to come from the attenuation kernels
    rather than from velocity.
    """
    zz, xx = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    r = np.sqrt((zz - nz / 2.0) ** 2 + (xx - nx / 2.0) ** 2)
    inside = (r <= radius).astype(np.float64)

    return {
        "vp": np.full((nz, nx), VP0) + dvp * inside,
        "vs": np.full((nz, nx), VS0) + dvs * inside,
        "rho": np.full((nz, nx), RHO0),
        "taup": np.full((nz, nx), TAUP0) + dtaup * inside,
        "taus": np.full((nz, nx), TAUS0) + dtaus * inside,
    }


def background():
    """The starting model: same as the truth minus the inclusion."""
    return circular_inclusion(dtaup=0.0, dtaus=0.0)


def make_seiscl(workdir, **overrides):
    cfg = dict(N=np.array([NZ, NX]), ND=2, dh=DH, dt=DT, NT=NT, FDORDER=8,
               freesurf=0, abs_type=2, nab=NAB, f0=F0, seisout=2,
               param_type=0, L=L, FL=FL)
    cfg.update(overrides)
    s = SeisCL(workdir=workdir, **cfg)

    # Crosswell geometry: sources down the left well, receivers down the
    # right one, both clear of the absorbing strip.
    zsrc = np.arange(NAB + 8, NZ - NAB - 8, 8) * DH
    xs = (NAB + 6) * DH
    ns = len(zsrc)
    s.src_pos_all = np.stack([np.full(ns, xs), np.zeros(ns), zsrc,
                              np.arange(ns, dtype=float),
                              np.full(ns, 100.0)])

    zrec = np.arange(NAB + 6, NZ - NAB - 6, 3) * DH
    xr = (NX - NAB - 6) * DH
    nr = len(zrec)
    gx = np.tile(np.full(nr, xr), ns)
    gz = np.tile(zrec, ns)
    sid = np.repeat(np.arange(ns, dtype=float), nr)
    rid = np.tile(np.arange(nr, dtype=float), ns)
    z0 = np.zeros(ns * nr)
    s.rec_pos_all = np.stack([gx, z0, gz, sid, rid, z0, z0, z0])

    s.src_all = None
    return s


def forward(workdir, params, **overrides):
    s = make_seiscl(workdir, **overrides)
    s.set_forward(s.src_pos_all[3, :], params, withgrad=False)
    s.execute()
    return s, s.read_data()


def observed(workdir, params):
    """Model 'observed' data and leave it where a gradient run can read it."""
    s, d = forward(workdir, params)
    s.write_data({"p": d[0]}, filename="obs_din.mat")
    return os.path.join(s.workdir, "obs_din.mat")


def gradient(workdir, params, din, gradfreqs):
    """Viscoelastic gradient via back_prop_type=2. OpenCL build only."""
    s = make_seiscl(workdir, gradout=1, back_prop_type=2,
                    gradfreqs=np.asarray(gradfreqs, dtype=float))
    s.file_din = din
    s.set_forward(s.src_pos_all[3, :], params, withgrad=True)
    s.execute()
    # read_grad returns [vp, vs, rho, taup, taus] when L>0
    return s, s.read_grad()


if __name__ == "__main__":
    wd = os.environ.get("VISCO_WORKDIR",
                        "/userdata/u/gfabien/claude/visco-work/_wd")
    os.makedirs(wd, exist_ok=True)
    os.chdir(wd)   # callcmd resolves file_din against the cwd

    true_p = circular_inclusion()
    start_p = background()
    print("model %dx%d, %d shots x %d receivers, NT=%d"
          % (NZ, NX, true_p["vp"].shape[0] and len(
              np.arange(NAB + 8, NZ - NAB - 8, 8)),
             len(np.arange(NAB + 6, NZ - NAB - 6, 3)), NT))

    din = observed(wd, true_p)
    print("observed data written to", os.path.basename(din))

    _, d0 = forward(wd, start_p)
    print("background data: max|p| = %.4e" % np.abs(d0[0]).max())

    s, g = gradient(wd, start_p, din, [40.0, 60.0, 80.0])
    names = ["vp", "vs", "rho", "taup", "taus"]
    for n, a in zip(names, g):
        a = np.asarray(a)
        print("  grad%-5s max|.| = %.4e   %s"
              % (n, np.abs(a).max(), "OK" if np.abs(a).max() > 0 else "ZERO"))
    assert np.abs(np.asarray(g[3])).max() > 0, (
        "gradtaup is zero -- viscoelastic gradients need the OpenCL build; "
        "the CUDA build has no host calc_grad and no L>0 device kernel")
    print("visco_crosswell: PASS")
