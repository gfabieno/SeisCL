"""Builder for docs/notebooks/Inversion/FreeSurfaceGradient.ipynb.

Repo convention: notebooks are generated from a builder script, then executed
with nbclient. This one runs on numpy alone (no SeisCL build, no GPU).

    python docs/builders/build_freesurface_gradient_theory.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md(r"""# The gradient in the top rows under a free surface

With a surface acquisition, the rows just under the free surface are where the
data are most sensitive, and where an inversion first changes the model. This
notebook explains three defects of the gradient there, and of the density
gradient at contrasts, and how they were corrected:

1. `freesurf=1` (stress image): the gradient of the rows $0..$`FDOH` was wrong
   with `back_prop_type=1` (ratios between $-2.6$ and $0.9$ in rows 0-3);
2. `freesurf=2` (vacuum): the density gradient of the first row under the vacuum
   was wrong (ratio 1.29, 3.34 with pressure data);
3. any model with a density contrast: the density gradient at the contrast was
   off (ratio 1.021 at a 1500/2500 kg/m$^3$ interface).

These rows are model cells, not ghost cells: the ghost cells are the `FDOH`
padding rows above row 0, where `freesurf=1` writes the images of the stresses;
they carry no parameter.

Notation as in [AbsorbingLayerGradient](AbsorbingLayerGradient.ipynb): adjoint
fields stored as $a = T\lambda$, $T = \mathrm{diag}(B, -C)$.""")

md(r"""## 1. Two forms of the same gradient

For one time step $x^{t+1} = P_t(\theta)\,x^t$, the gradient of the misfit is,
by definition, the **direct form**
$$\frac{\partial J}{\partial\theta} = \sum_t \lambda^{t+1}\cdot\frac{\partial P_t}{\partial\theta}\,x^t .$$
For the density, $\partial P/\partial B$ is the spatial derivative of the
stresses, $f^t/B$ with $f^t$ the forward increment of the velocity. SeisCL's
adjoint kernels accumulate instead a **summation-by-parts form**: forward field
times the increment $\Delta a$ that the kernel adds to the adjoint field,
$$-\sum_t a^{t+1}\, f^t = \sum_t v^t\,\Delta a^t \quad(\text{up to the sign convention}),$$
which avoids storing the forward increments. The two are equal **only if every
change of the adjoint field at that cell is the kernel's increment**.

Near a stress-image free surface it is not: the adjoint of the surface
(`surface_adj_fold`, `surface_adj`) adds the adjoint of the stress images back
into the rows $1..$`FDOH`, transforms the surface row, and adds terms to the
adjoint velocities of the rows $0..$`FDOH`$-1$. The summation-by-parts form
misses all of that.

**Correction.** In those rows the gradient is computed in the direct form, by a
new kernel `surface_grad` (`surface2D_adj.cl`, `surface3D_adj.cl`). The
difficulty is the pairing in time: which adjoint field multiplies which forward
increment. A matrix model of the GPU time step (developer notes,
`torch-robust/surfgrad/pairing2.py`) shows that **at the start of each adjoint
step, before the stress update, the adjoint fields are exactly $T\eta_t$**, with
$\eta_t$ the adjoint of the forward state right after the velocity update of
step $t$; their stress part is $T\nu_t$, $\nu_t$ the adjoint of the stresses
before the surface kernel. So `surface_grad` runs first in the adjoint step and
accumulates
* moduli: $\nu\cdot\frac{\partial C}{\partial\theta}\,\varepsilon(v^{t+1})$, with
  $\nu = -C^{-1}a$;
* density: $-a_v\,(v^{t+1} - v^t)$, with $v^t$ the stored forward state (a
  stencil would read the stress images, which are not rebuilt), minus a force
  source of that step.

On the surface row the forward update is the stress update followed by the
surface kernel: $\sigma_{zz} = 0$ and, in 2-D,
$\sigma_{xx} \mathrel{+}= \left(M - \frac{(M-2\mu)^2}{M}\right)\partial_x v_x$, so
$$\frac{\partial}{\partial M} = \frac{4\mu^2}{M^2}\,\partial_x v_x,\qquad
  \frac{\partial}{\partial \mu} = \frac{4(M-2\mu)}{M}\,\partial_x v_x .$$
(3-D: the kernel header.) On the matrix model these formulas give the exact
derivative to 6 digits, rows 0-6.""")

md(r"""## 2. The density averaging transpose

The velocity updates use buoyancies at staggered points,
$b_{i+1/2} = 2/(\rho_i + \rho_{i+1})$, i.e. the **arithmetic average of the
density**; with one vacuum neighbour (`freesurf=2`) this gives $2/\rho$, the
improved vacuum formulation. The adjoint kernels accumulate the gradient against
that averaged density ($-B^2\,\partial J/\partial B = \partial J/\partial\rho_{avg}$),
so the transpose of the averaging is simply
$$\frac{\partial J}{\partial \rho_i} \mathrel{+}= \tfrac12\,\frac{\partial J}{\partial \rho_{avg}}$$
for each of the two cells, vacuum or not. The code applied instead the Jacobian
of the buoyancy average, $\tfrac12 (b_{avg}/b_i)^2 = \tfrac12(\rho_i/\rho_{avg})^2$,
which equals $\tfrac12$ only for a uniform density, and returned 0 when one
neighbour was vacuum. The cell below shows the size of the error at a contrast.""")

code(r"""import numpy as np
for r1, r2 in ((2000, 2000), (2000, 2100), (1500, 2500), (1000, 2700)):
    ravg = 0.5 * (r1 + r2)
    old = [0.5 * (r / ravg) ** 2 for r in (r1, r2)]
    print("rho %4d / %4d : old weights %.3f %.3f   exact 0.500 0.500" % (r1, r2, *old))
print("vacuum / 2000 : old weights 0 (term dropped)   exact 0 and 0.500")""")

md(r"""In a full gradient each cell also receives exact terms from its other
neighbours, so the error at a 1500/2500 interface was about 2 %; it grows with
the contrast (water / sediment, sediment / salt).

## 3. Finite-difference checks

Ratio of a two-sided finite difference of the misfit to the adjoint gradient,
for a random perturbation of **one row** (away from the absorbing layers), FD
step 0.125 % (developer notes, `torch-robust/gradvar/inv/diag_layer.py`; test
`SeisCL/tests/test_gradient_surface.py`).

| case | rows | before | after |
|---|---|---|---|
| 2-D `freesurf=1`, cross-well, velocity data | 1, 2, 3 (vs) | -2.57, 0.90, 1.02 | 1.0001, 1.0000, 1.0052 |
| 2-D `freesurf=1`, force source 10 cells deep, receivers at row 1 | 0-5 | | 0.990-1.005 |
| 2-D `freesurf=1`, pressure receivers at row 1 | receiver row | 0.62 (vp) | 0.988 |
| 3-D `freesurf=1`, cross-well | 0-2 (rho) | 0.41, 0.38, 0.88 | 1.008, 0.93*, 1.008 |
| 3-D `freesurf=1`, force source 8 cells deep, receivers at row 1 | 0-5 | | 0.996-1.004 |
| 2-D `freesurf=2`, first medium row, rho, velocity / pressure data | 4 | 1.29 / 3.34 | 1.0000 / 0.9997 |
| 2-D density contrast 1500/2500 | interface rows | 1.021, 1.006 | 1.0001, 1.0002 |

\* FD signal ten times smaller than on the neighbouring rows. The surface row
itself has a very small FD signal in the cross-well geometry (its sensitivity
to $v_p$ is weak), so it is checked with a surface acquisition.

## 4. Limits

* `restype=1`, the `FP16` kernels and the viscoelastic case are not covered by
  `surface_grad`.
* In CPML layer columns at the surface, the strain used by `surface_grad` is not
  CPML-filtered; this matters only with `gradabs=1`, in the layer corners.""")

nb["cells"] = c
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "docs/notebooks/Inversion/FreeSurfaceGradient.ipynb")
print("wrote docs/notebooks/Inversion/FreeSurfaceGradient.ipynb")
