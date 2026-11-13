"""Builder for docs/notebooks/Inversion/AbsorbingLayerGradient.ipynb.

Repo convention: notebooks are generated from a builder script, then executed
with nbclient. This one runs on numpy alone (no SeisCL build, no GPU).

    python docs/builders/build_absorbing_gradient_theory.py
    python -m nbclient docs/notebooks/Inversion/AbsorbingLayerGradient.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md(r"""# The gradient inside the absorbing layers

SeisCL computes the misfit gradient with the adjoint-state method. With
`back_prop_type=1` the forward wavefield is not stored: it is rebuilt during the
adjoint pass by running the forward time step backwards, from values saved on a
thin band of cells. This notebook explains

1. why the gradient was wrong inside the absorbing layers (and in a thin band
   just inside them), and what `gradabs=1` changes;
2. why, with the Cerjan taper (`abs_type=2`), the adjoint needs nothing more;
3. the exact discrete adjoint of the CPML (`abs_type=1`), obtained by a change
   of variables, and where it lives in the code;
4. a small numerical check of that adjoint against the exact matrix transpose;
5. the finite-difference checks of the new code, and how to use it.

Notation: one time step updates the velocities, then the stresses,
$$v^{n+1} = v^n + B\,\tilde D s^n, \qquad s^{n+1} = s^n + C\,\tilde D' v^{n+1},$$
where $B$ holds the buoyancies (`rip`, `rkp`), $C$ the moduli, $D$ and $D'$ the
staggered derivative stencils ($D' = -D^T$), and $\tilde D$ a derivative after
the absorbing-layer treatment. The adjoint fields that SeisCL stores are
$a = T\lambda$ with $T = \mathrm{diag}(B, -C)$; in these variables the adjoint
time step uses the same stencils as the forward one.""")

md(r"""## 1. Why the gradient was wrong in the layers

Running a time step backwards undoes the update of the field, but not the
absorption: the taper of Cerjan and the memory variables of the CPML remove
energy, and that cannot be undone. So, inside the layers, the rebuilt forward
field is wrong. SeisCL saves, at every time step, the forward field on a band of
`FDOH` cells (half the stencil length) just inside the layers, and overwrites the
rebuilt field there; the interior is then exact, because its stencils never reach
the layers.

Two consequences, both measured with finite differences (section 5):

* in the layers the gradient is meaningless; `read_grad` sets it to zero
  (`cropgrad`);
* in the band of `FDOH` cells, the shear-modulus and density gradients were also
  wrong (ratios between 0.40 and 1.49), because the forward strain there uses
  rebuilt values from the layers. This band is not cropped.

**`gradabs=1`** saves the forward field in the whole layers and the band, at every
time step (`savebnd2D.cl`, `savebnd3D.cl`, `header_injectbnd.cl`), and injects it
during the adjoint pass. The rebuilt field is then exact everywhere. The cost is
memory on the host: (number of layer and band cells) $\times$ (number of fields:
5 in 2-D, 9 in 3-D) $\times$ 4 bytes $\times$ `NT`. Example: a 2-D grid of
$64\times80$ cells with `nab=10` and `FDORDER=8` has 3248 such cells, so
$3248 \times 5 \times 4 \times 1200 = 78$ MB. A 3-D grid of $200^3$ with
`nab=20` has about $3.9\times10^6$ cells: 140 MB per time step, too much for
long runs. Hence `gradabs` is off by default.""")

md(r"""## 2. Cerjan: the adjoint is already exact

With the Cerjan taper $\Gamma$ (a diagonal matrix, 1 outside the layers), the
forward step is
$$v^{n+1} = \Gamma\,(v^n + B D s^n), \qquad s^{n+1} = \Gamma\,(s^n + C D' v^{n+1}).$$
Write it $P = S\,V$ with $V = \begin{pmatrix}\Gamma & \Gamma B D\\ 0 & I\end{pmatrix}$
and $S = \begin{pmatrix} I & 0\\ \Gamma C D' & \Gamma\end{pmatrix}$. In the
variables $a = T\lambda$,
$$T S^T T^{-1} = \begin{pmatrix} I & B D\,\Gamma \\ 0 & \Gamma\end{pmatrix},\qquad
  T V^T T^{-1} = \begin{pmatrix} \Gamma & 0 \\ C D' \Gamma & I\end{pmatrix}.$$
Read as a sequence of operations, the adjoint is: taper the stresses, update the
velocities, taper the velocities, update the stresses, ... SeisCL's adjoint
kernels do: update the stresses, taper them, update the velocities, taper them,
... It is the same infinite sequence. So the adjoint propagation needs no change.

**The gradient formula** also stays the same. For the buoyancy, the exact
gradient is $\sum_n \lambda_v^{n+1}\,\Gamma\, D s^n$. With $f^n = \Gamma^{-1}v^{n+1} - v^n$
the forward increment and $\lambda_v^{n} = \Gamma\lambda_v^{n+1} + \Delta^n$ the
adjoint recursion, a summation by parts gives
$$\sum_n \lambda_v^{n+1}\Gamma f^n
 = \sum_n \lambda_v^{n+1} v^{n+1} - \sum_n (\lambda_v^n - \Delta^n) v^n
 = \sum_n \Delta^n v^n,$$
(the boundary terms vanish: zero initial field, zero final adjoint). This is what
`update_adjv2D.cl` accumulates (`gradrip += -vx*lvx`, the sign comes from $T$).
The same holds for the moduli. So, once the forward field is exact in the
layers (`gradabs=1`), the gradient there is exact.""")

md(r"""## 3. CPML: the exact adjoint by a change of variables

In the CPML (Komatitsch and Martin, 2007), every derivative $d$ of a field, at a
cell of the layer of its direction, is filtered where it is used:
$$\psi^{n} = b\,\psi^{n-1} + a\,d^n, \qquad \tilde d^n = d^n/K + \psi^n,$$
with $a$, $b$, $K$ that depend on the cell and on the term (`a_x`, `b_x_half`,
`K_z`, ...). For example the update of $v_x$ uses
$\tilde d = \mathrm{CPML}_{\sigma_{xx,x}}(D_x\sigma_{xx})$, filtered with the
coefficients of the $v_x$ position.

**Transpose of one term.** Take $v' = v + B\tilde d$, $\psi' = b\psi + a\,d$,
$d = D\sigma$. With $e = B\lambda_{v'}$ (the stored adjoint velocity) and
$\mu = \lambda_{\psi'}$, the transpose gives
$$\lambda_\psi = b\,(e + \mu), \qquad
  \lambda_\sigma \mathrel{+}= D^T\!\left[e/K + a\,(e + \mu)\right].$$
**Change of variables.** Let $\tilde\psi = a\,(e + \mu)$, the memory as the
adjoint sees it ($\tilde\psi = a\lambda_\psi/b$ one step later). Then
$$\tilde\psi^{\,n} = b\,\tilde\psi^{\,n+1} + a\,e^n, \qquad
  \lambda_\sigma \mathrel{+}= D^T\!\left[e^n/K + \tilde\psi^{\,n}\right].$$
This is **the forward recursion, applied to the adjoint field before the
derivative**, instead of to the derivative. Because $a$, $b$, $K$ vary in space,
"filter then differentiate" is not the same as "differentiate then filter": the
old adjoint kernels did the latter, with the coefficients of the wrong grid
position, and their gradient was wrong everywhere (the finite-difference ratio
was 0.0000, section 5).

**Which coefficients.** The adjoint field at a cell is filtered with the
coefficients of the forward term computed at that same cell:

| adjoint field | filtered with (2-D) | then | used for |
|---|---|---|---|
| $v_x$ | $\sigma_{xx,x}$ (x, half), $\sigma_{xz,z}$ (z, full) | $D^-_x$, $D^+_z$ | $\sigma_{xx}, \sigma_{zz}$; $\sigma_{xz}$ |
| $v_z$ | $\sigma_{zz,z}$ (z, half), $\sigma_{xz,x}$ (x, full) | $D^-_z$, $D^+_x$ | $\sigma_{xx}, \sigma_{zz}$; $\sigma_{xz}$ |
| $\sigma_{xx}$ | $v_{x,x}$ (x, full) | $D^+_x$ | $v_x$ |
| $\sigma_{zz}$ | $v_{z,z}$ (z, full) | $D^+_z$ | $v_z$ |
| $\sigma_{xz}$ | $v_{x,z}$ (z, half), $v_{z,x}$ (x, half) | $D^-_z$, $D^-_x$ | $v_x$; $v_z$ |

In 3-D every velocity is filtered in three directions and the shear stresses in
two (nine filtered fields; see the header of `cpml_adj3D.cl`).

**In the code.** Before each adjoint update, a small kernel (`cpml_adj_v`,
`cpml_adj_s` in `src/cpml_adj2D.cl` and `src/cpml_adj3D.cl`) updates the
memories $\tilde\psi$ (the adjoint copies `psi_*r` of the forward memories, same
layout) and writes the filtered fields into scratch buffers (`cpfa`...`cpfd` in
2-D, `cpf1`...`cpf9` in 3-D; copies of the fields outside the layers). The
adjoint kernels `update_adjs*.cl` and `update_adjv*.cl` differentiate these
buffers instead of the fields (`time_stepping.c`, `update_grid_adj`). The
gradient formula is unchanged (the summation by parts of section 2 holds with
$\Gamma = 1$, since the CPML does not damp the fields themselves). The scratch
buffers are not exchanged between devices, so CPML gradients with
`back_prop_type=1` need a single device.""")

md(r"""## 4. Numerical check against the exact transpose

The cell below builds, in 1-D, the forward time step with a CPML layer on the
left as an explicit matrix (fields and memories), computes the exact gradient of
$J = w\cdot x^N$ with respect to the initial fields by $(P^N)^T w$, and compares
it with the adjoint described above (filter, then differentiate). Every
coefficient is random, so the check does not depend on the CPML profile. The same
check in 2-D, with the actual stencils of the kernels, is
`torch-robust/cpmladj/ref.py` of the developer notes (agreement $4\times10^{-16}$).""")

code(r"""import numpy as np
rng = np.random.default_rng(0)
n, NL, h = 24, 6, np.array([1.17, -0.065])      # cells, layer cells, FD coefficients (4th order)
B, C = 0.3 + 0.1 * rng.random(n), 2.0 + rng.random(n)
# one (a, b, K) per term and per layer cell: term "s" = derivative of s used by v, "v" = derivative of v used by s
coef = {t: [(0.1 + 0.3 * rng.random(), 0.5 + 0.4 * rng.random(), 1 + rng.random()) for _ in range(NL)] for t in "sv"}

def g(f, i):    # zero outside the grid (the halo of the kernels)
    return f[i] if 0 <= i < n else 0.0
def Dp(f, i):   # forward difference (v uses it on s)
    return sum(h[m] * (g(f, i + m + 1) - g(f, i - m)) for m in range(2))
def Dm(f, i):   # backward difference (s uses it on v)
    return sum(h[m] * (g(f, i + m) - g(f, i - m - 1)) for m in range(2))

N = 2 * n + 2 * NL                               # state: v, s, psi_s, psi_v
def step(u):
    v, s, ps, pv = u[:n].copy(), u[n:2*n].copy(), u[2*n:2*n+NL].copy(), u[2*n+NL:].copy()
    d = np.array([Dp(s, i) for i in range(n)])
    for i in range(NL):
        a, b, K = coef["s"][i]; ps[i] = b * ps[i] + a * d[i]; d[i] = d[i] / K + ps[i]
    v = v + B * d
    d = np.array([Dm(v, i) for i in range(n)])
    for i in range(NL):
        a, b, K = coef["v"][i]; pv[i] = b * pv[i] + a * d[i]; d[i] = d[i] / K + pv[i]
    s = s + C * d
    return np.concatenate([v, s, ps, pv])

P = np.column_stack([step(e) for e in np.eye(N)])
T = np.diag(np.concatenate([B, -C]))

def adjoint(a_v, a_s, nsteps):
    mt_s, mt_v = np.zeros(NL), np.zeros(NL)      # adjoint memories psi~
    for _ in range(nsteps):
        f = a_s.copy()                           # filter the adjoint stress (coefficients of the "v" term)
        for i in range(NL):
            a, b, K = coef["v"][i]; mt_v[i] = b * mt_v[i] + a * a_s[i]; f[i] = a_s[i] / K + mt_v[i]
        a_v = a_v + B * np.array([Dp(f, i) for i in range(n)])
        f = a_v.copy()                           # filter the adjoint velocity (coefficients of the "s" term)
        for i in range(NL):
            a, b, K = coef["s"][i]; mt_s[i] = b * mt_s[i] + a * a_v[i]; f[i] = a_v[i] / K + mt_s[i]
        a_s = a_s + C * np.array([Dm(f, i) for i in range(n)])
    return np.linalg.solve(T, np.concatenate([a_v, a_s]))

w = rng.standard_normal(N); w[2 * n:] = 0
for nsteps in (1, 3, 10):
    exact = (np.linalg.matrix_power(P, nsteps).T @ w)[:2 * n]
    a = T @ w[:2 * n]
    err = np.abs(adjoint(a[:n], a[n:], nsteps) - exact).max() / np.abs(exact).max()
    print("%2d steps: max relative difference %.1e" % (nsteps, err))""")

md(r"""The adjoint agrees with the exact transpose to rounding error. Differentiating
first and filtering after, as the old kernels did, gives errors of 6 to 43 % of
the gradient in this small example (1, 3 and 10 steps).

## 5. Finite-difference checks of the code

Ratio of a two-sided finite difference of the misfit to the adjoint gradient, for
a random perturbation restricted to one region: the layers, the band of `FDOH`
cells inside them, or the rest (the interior). Cross-well geometry of
`SeisCL/tests/test_gradient_fd.py`, 2-D ($64\times80$) and 3-D ($44\times44\times76$),
`nab=10`, `FDORDER=8`, pressure source unless stated (developer notes,
`torch-robust/gradvar/inv/diag_layer.py`, runs M to U).

| case | layers | band | interior |
|---|---|---|---|
| Cerjan 2-D, before (`gradabs` not available) | 0 (cropped) | vs 0.79, rho -1.16 | 1.0008-1.0037 |
| Cerjan 2-D, `gradabs=1` | 1.0002-1.0006 | 0.9997-1.0006 | 1.0008-1.0037 |
| Cerjan 2-D, vertical force, velocity receivers, `gradabs=1` | 0.9995-0.9999 | 0.9959-1.0001 | 1.0006-1.0019 |
| Cerjan 3-D, `gradabs=1` | 0.9973-1.0007 | 0.9993-1.0028 | 1.0005-1.0242 |
| CPML 2-D, before | 0 (cropped) | 0.0000 | 0.0000 |
| CPML 2-D, after, `gradabs=0` | 0 (cropped) | vs 0.76, rho -0.43 | 1.0007-1.0036 |
| CPML 2-D, after, `gradabs=1` | 0.9995-1.0010 | 1.0000-1.0003 | 1.0007-1.0036 |
| CPML 3-D, after, `gradabs=1` | 0.9989-0.9999 | 0.9964-1.0004 | 1.0006-1.0069 |

The same results with the OpenCL build and through `SeisCL.torch`
(`Config.GRADABS`). With a free surface the layers and band are exact too; the top
rows of the model are a separate matter (the gradient in the first rows under the
surface is not yet exact).

## 6. How to use it

```python
seis = SeisCL(..., abs_type=2, back_prop_type=1, gradabs=1)   # or abs_type=1 (CPML)
```

or `cfg.GRADABS = 1` with `SeisCL.torch`. `read_grad` then returns the gradient
in the layers too (only the vacuum rows of `freesurf=2` are still zeroed).
Limits: `back_prop_type=1`, `FP16=0`, a single device, 2-D P-SV or 3-D elastic.
Estimate the memory first (section 1). Without `gradabs`, do not interpret the
gradient within `nab + FDORDER/2` cells of an absorbing edge.""")

nb["cells"] = c
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "docs/notebooks/Inversion/AbsorbingLayerGradient.ipynb")
print("wrote docs/notebooks/Inversion/AbsorbingLayerGradient.ipynb")
