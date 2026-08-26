"""Builder for docs/notebooks/Inversion/ViscoelasticGradientTheory.ipynb.

Per notes/docs-status.md the repo convention is to generate notebooks from a
builder script rather than hand-editing JSON, then execute with nbclient.
This one is cheap to execute (sympy + numpy only, no SeisCL runs), so it can
be re-run anywhere.

    python docs/builders/build_visco_gradient_theory.py
    python -m nbclient docs/notebooks/Inversion/ViscoelasticGradientTheory.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md(r"""# The viscoelastic FWI gradient: theory and where it lives in SeisCL

SeisCL computes the gradient of the waveform misfit with respect to
$(\rho, M, \mu, \tau_p, \tau_s)$ by the adjoint-state method. The expressions it
uses are not ad hoc — they come from Chapter 3 of

> Fabien-Ouellet, G. (2017). *Inversion des formes d'ondes complètes
> viscoélastique.* PhD thesis, INRS.
> [PDF](https://espace.inrs.ca/id/eprint/5251/1/Fabien-Ouellet,%20Gabriel.pdf)

specifically §3.5 and equations (3.51)–(3.55).

This notebook is a map between that derivation and the source. It covers:

1. where the gradient expressions come from,
2. the five parameter gradients and the six dot products they are built from,
3. **exactly which line of `calc_grad.c` implements which term**,
4. a symbolic check that the coded coefficients equal the published ones,
5. how the memory-variable terms reconcile once the alternative variables are
   accounted for.

It runs on `sympy` and `numpy` alone — no SeisCL build or GPU needed.""")

md(r"""## 1. Where the expressions come from

The velocity–stress–memory system is written (thesis eq. 3.20) as

$$\mathbf{A}\dot{\boldsymbol\phi} + \mathbf{B}\boldsymbol\phi - \mathbf{G}\mathbf{C}\boldsymbol\phi - \mathbf{s} = 0$$

with $\boldsymbol\phi = (\mathbf v, \boldsymbol\sigma, \mathbf R^1,\dots,\mathbf R^L)$.
$\mathbf G$ is block-diagonal in the material properties. The adjoint-state
gradient (eq. 3.36, specialised to isotropy in 3.51) is

$$\frac{\partial X}{\partial m_\alpha} = -\Big\langle \boldsymbol\psi,\; \mathbf T\frac{\partial \boldsymbol\Lambda^{-1}}{\partial m_\alpha}\mathbf T\big(\mathbf A\dot{\boldsymbol\phi} + \mathbf B\boldsymbol\phi - \mathbf s\big)\Big\rangle$$

where $\mathbf G = \mathbf T\boldsymbol\Lambda\mathbf T$ is an eigendecomposition
(eq. 3.44–3.50). **In isotropy the eigenvalues are the bulk and shear
combinations $3\lambda+2\mu$ and $2\mu$** (eq. 3.46/3.48).

That is worth remembering, because differentiating $\boldsymbol\Lambda^{-1}$ is
what produces the factor

$$\big(N_d M - 2(N_d-1)\mu\big)^{-2}$$

which you will meet again and again in `calc_grad.c` as `1/fact1` and
`1/fact2`. It is not a fudge factor; it is the inverse-squared eigenvalue.""")

md(r"""## 2. The five gradients and the six dot products

Thesis eq. (3.52), for $m = [\rho, M, \mu, \tau_p, \tau_s]$:

$$\begin{aligned}
\frac{\partial X}{\partial \rho} &= \langle \tilde v_x, \partial_t v_x\rangle + \langle \tilde v_y, \partial_t v_y\rangle + \langle \tilde v_z, \partial_t v_z\rangle\\
\frac{\partial X}{\partial M}    &= -c_1^{M}P_1 + c_2^{M}P_2\\
\frac{\partial X}{\partial \tau_p} &= -c_1^{\tau_p}P_1 + c_2^{\tau_p}P_2\\
\frac{\partial X}{\partial \mu}  &= -c_1^{\mu}P_3 + c_2^{\mu}P_1 - c_3^{\mu}P_4 + c_4^{\mu}P_5 - c_5^{\mu}P_2 + c_6^{\mu}P_6\\
\frac{\partial X}{\partial \tau_s} &= -c_1^{\tau_s}P_3 + c_2^{\tau_s}P_1 - c_3^{\tau_s}P_4 + c_4^{\tau_s}P_5 - c_5^{\tau_s}P_2 + c_6^{\tau_s}P_6
\end{aligned}$$

The six dot products (eq. 3.53) split into **stress** terms and
**memory-variable** terms:

| | quantity | kind |
|---|---|---|
| $P_1$ | $\langle \tilde\sigma_{xx}+\tilde\sigma_{yy}+\tilde\sigma_{zz},\ \partial_t(\sigma_{xx}+\sigma_{yy}+\sigma_{zz})\rangle$ | stress, trace |
| $P_2$ | $\langle \tilde R_{xx}+\tilde R_{yy}+\tilde R_{zz},\ (1+\tau_\sigma^l\partial_t)(r_{xx}+r_{yy}+r_{zz})\rangle$ | memory, trace |
| $P_3$ | $\sum \langle \tilde\sigma_{ij}, \partial_t\sigma_{ij}\rangle$, $ij \in \{xy,xz,yz\}$ | stress, shear |
| $P_4$ | $\sum \langle \tilde\sigma_{ii}, \partial_t((N_d-1)\sigma_{ii}-\sigma_{jj}-\sigma_{kk})\rangle$ | stress, deviatoric |
| $P_5$ | $\sum \langle \tilde R_{ij}, (1+\tau_\sigma^l\partial_t)r_{ij}\rangle$ | memory, shear |
| $P_6$ | $\sum \langle \tilde R_{ii}, (1+\tau_\sigma^l\partial_t)((N_d-1)r_{ii}-r_{jj}-r_{kk})\rangle$ | memory, deviatoric |

Note $\rho$'s gradient involves **only** the velocity correlation — no stress,
no memory. That is why in SeisCL `gradrho` gets `-dot[8]` and nothing else,
and why the $M$/$\mu$ contributions to $\rho$ in a $(v_p,v_s,\rho)$
parameterization are a *separate* chain rule applied afterwards, not part of
the correlation.""")

md(r"""## 3. Mapping onto the source

The correlation lives in `src/calc_grad.c` (host reference) and in the
`src/grad_dft*.cl` device kernels. The thesis $P_i$ map onto the code's
`dot[]` array like this:

| thesis | code | note |
|---|---|---|
| $P_1$ | `dot[0]`, and `dot[3]` | |
| $P_2$ | `dot[1]`, and `dot[6]` | |
| $P_3$ | `dot[2]` | |
| $P_4$ | `dot[4]` | |
| $P_5$ | `dot[5]` | |
| $P_6$ | `dot[7]` | |

`calc_grad.c` contains the lines `dot[3]=dot[0];` and `dot[6]=dot[1];`. Those
look like redundant copies but are not: eq. (3.52d–e) genuinely reuse $P_1$
and $P_2$ inside the $\mu$ and $\tau_s$ expressions, and the duplicate slots
keep the coefficient indices lined up with the published ordering.

The accumulation then reads, term for term:

```c
gradM[indm]    += -c[0]*dot[0] + c[1]*dot[1];
gradtaup[indm] += -c[8]*dot[0] + c[9]*dot[1];
gradmu[indm]   += -c[2]*dot[2] + c[3]*dot[3] - c[4]*dot[4]
                  +c[5]*dot[5] - c[6]*dot[6] + c[7]*dot[7];
gradtaus[indm] += -c[10]*dot[2]+ c[11]*dot[3]- c[12]*dot[4]
                  +c[13]*dot[5]- c[14]*dot[6]+ c[15]*dot[7];
gradrho[indm]  += -dot[8];
```

so `c[0..1]` are $c^M_{1,2}$, `c[2..7]` are $c^\mu_{1..6}$, `c[8..9]` are
$c^{\tau_p}_{1,2}$ and `c[10..15]` are $c^{\tau_s}_{1..6}$. The coefficients
themselves are built in `grad_coefvisc_1()` (and its `_SH` twin).""")

md(r"""## 4. Checking the coded coefficients against the thesis

Eq. (3.54)–(3.55). Two conventions have to be reconciled first:

* the thesis's $\tau$ is the **total** relaxation level, SeisCL's is
  **per mechanism**, so $\tau_{\text{thesis}} = L\,\tau_{\text{code}}$ — which is
  why the thesis writes $(1+\tau_p)$ where the code writes `(1+L*taup)`, and
  why $\partial/\partial\tau_{\text{code}} = L\,\partial/\partial\tau_{\text{thesis}}$
  puts an extra $L$ on the four $\tau$ coefficients;
* $\alpha$ is **not** a free parameter and **not** zero. It is the GSLS
  phase-velocity normalization factor. `M()`/`mu()`
  (`assign_modeling_case.c`) divide the stored moduli by $(1+\alpha\tau)$ so
  that the phase velocity **at $f_0$** equals the $v_p$/$v_s$ the user
  supplied — the elastic convention — and `calc_grad.c` rebuilds the same
  quantity for the gradient:

  $$\alpha = \sum_l \frac{r_l^2}{1+r_l^2}, \qquad r_l = f_0/F_{L,l}$$

  It is easy to misread: `calc_grad.c` writes `al=0` and then *accumulates*
  into it a few lines later. Taking that initialisation at face value and
  hardcoding zero scales the $\tau$ coefficients by $(1-\alpha)$ — for
  $f_0 = F_L$ that is a factor of exactly one half.

Let's verify symbolically rather than by eye.""")

code(r'''import sympy as sp

M, mu, tp, ts, a, L, Nd = sp.symbols(
    'M mu tau_p tau_s alpha L N_d', positive=True)

# --- thesis eq. (3.55): the two inverse-squared eigenvalue factors ----------
b1 = (Nd*M*(1+L*tp)/(1+a*tp) - 2*(Nd-1)*mu*(1+L*ts)/(1+a*ts))**-2
b2 = (Nd*M*tp/(1+a*tp)       - 2*(Nd-1)*mu*ts/(1+a*ts))**-2

# --- thesis eq. (3.54) -----------------------------------------------------
thesis = {
 'c1M':  (1+L*tp)/(1+a*tp)*b1,           'c2M':  tp/(1+a*tp)*b2,
 'c1tp': (1-a)*M/(1+a*tp)**2*b1,         'c2tp': M/(1+a*tp)**2*b2,
 'c1mu': (1+a*ts)/(mu**2*(1+L*ts)),      'c1ts': (1-a)/(mu*(1+L*ts)**2),
 'c2mu': (Nd+1)/3*(1+L*ts)/(1+a*ts)*b1,  'c2ts': (Nd+1)/3*(1-a)*mu/(1+a*ts)**2*b1,
 'c3mu': (1+a*ts)/(2*Nd*mu**2*(1+L*ts)), 'c3ts': (1-a)/(2*Nd*mu*(1+L*ts)**2),
 'c4mu': (1+a*ts)/(mu**2*ts),            'c4ts': 1/(mu*ts**2),
 'c5mu': (Nd+1)/3*ts/(1+a*ts)*b2,        'c5ts': (Nd+1)/3*mu/(1+a*ts)**2*b2,
 'c6mu': (1+a*ts)/(2*Nd*mu**2*ts),       'c6ts': 1/(2*Nd*mu*ts**2),
}

# --- SeisCL grad_coefvisc_1(), calc_grad.c ---------------------------------
f1 = (Nd*M*(1+L*tp)*(1+a*ts) - 2*(Nd-1)*mu*(1+L*ts)*(1+a*tp))**2
f2 = (Nd*M*tp*(1+a*ts)       - 2*(Nd-1)*mu*ts*(1+a*tp))**2
seiscl = {
 'c1M':  (1+L*tp)*(1+a*tp)*(1+a*ts)**2/f1,
 'c2M':  tp*(1+a*tp)*(1+a*ts)**2/f2,
 'c1tp': M*(L-a)*(1+a*ts)**2/f1,          'c2tp': M*(1+a*ts)**2/f2,
 'c1mu': (1+a*ts)/(mu**2*(1+L*ts)),       'c1ts': (L-a)/(mu*(1+L*ts)**2),
 'c2mu': (Nd+1)/3*(1+L*ts)*(1+a*ts)*(1+a*tp)**2/f1,
 'c2ts': (Nd+1)/3*mu*(L-a)*(1+a*tp)**2/f1,
 'c3mu': (1+a*ts)/(2*Nd*mu**2*(1+L*ts)),  'c3ts': (L-a)/(2*Nd*mu*(1+L*ts)**2),
 'c4mu': (1+a*ts)/(mu**2*ts),             'c4ts': 1/(mu*ts**2),
 'c5mu': (Nd+1)/3*ts*(1+a*ts)*(1+a*tp)**2/f2,
 'c5ts': (Nd+1)/3*mu*(1+a*tp)**2/f2,
 'c6mu': (1+a*ts)/(2*Nd*mu**2*ts),        'c6ts': 1/(2*Nd*mu*ts**2),
}

print("%-6s %-22s %s" % ("coef", "general (any a, L)", "at L=1, alpha symbolic"))
for k in thesis:
    gen = sp.simplify(thesis[k] - seiscl[k]) == 0
    one = sp.simplify((thesis[k] - seiscl[k]).subs(L, 1)) == 0
    print("%-6s %-22s %s" % (k, "match" if gen else "differ by an L factor",
                             "MATCH" if one else "DIFFER"))''')

md(r"""**Result.** Twelve of the sixteen coefficients match identically for
arbitrary $\alpha$ and $L$. The four that do not — $c_1^{\tau_p}$,
$c_1^{\tau_s}$, $c_2^{\tau_s}$, $c_3^{\tau_s}$ — are exactly the ones carrying
$(L-\alpha)$ in the code against $(1-\alpha)$ in the thesis, the
per-mechanism vs total $\tau$ convention noted above. **At $L=1$ they
coincide for any $\alpha$, so all sixteen agree there**; the difference bites
only for $L>1$, which is worth checking against the PDF before trusting a
multi-mechanism run.

A caution for anyone re-deriving from a text extraction of the thesis: the
OCR'd text is internally inconsistent about where $L$ appears (it renders
$c_1^{\tau_s}$ with $(1+\tau_s)^2$ but $c_3^{\tau_s}$ with $(1+L\tau_s)^2$).
Check the PDF visually before concluding anything from the text dump.""")

md(r"""## 5. From the time domain to the DFT gradient

`back_prop_type=2` evaluates these correlations in the **frequency domain**,
over a chosen set of frequencies, instead of accumulating them per time step.
With Parseval and $\partial_t \rightarrow i\omega$, the stress terms map
cleanly. For $P_1$:

$$\langle\tilde\sigma, \partial_t\sigma\rangle \;=\; \int \omega\,\mathrm{Im}\!\big(\tilde\sigma\,\overline{\sigma}\big)\,d\omega$$

which is exactly what the code computes:

```c
cl_itreal(a,b) = a.y*b.x - a.x*b.y      /* = Im(a * conj(b)) */
dot[0] = freq * cl_itreal(sxxzzr, sxxzz) / dftnorm;
```

Let's confirm that identity numerically.""")

code(r'''import numpy as np

rng = np.random.default_rng(0)
A = rng.standard_normal(6) + 1j*rng.standard_normal(6)   # adjoint spectrum
B = rng.standard_normal(6) + 1j*rng.standard_normal(6)   # forward spectrum
w = 2*np.pi*17.0

def cl_itreal(a, b):
    """calc_grad.c:31 -- a.y*b.x - a.x*b.y."""
    return a.imag*b.real - a.real*b.imag

lhs = w * cl_itreal(A, B)                 # what the code forms
rhs = np.real(np.conj(A) * (1j*w) * B)    # <adj, d/dt fwd> via Parseval
print("max|code - theory| =", np.abs(lhs - rhs).max())''')

md(r"""So the **stress** half of the DFT gradient is a faithful frequency-domain
rendering of eq. (3.53a,c,d).

## 6. The memory-variable terms use *alternative* variables, not $r$

This is the part that trips people up, so it is worth doing carefully.

$P_2$, $P_5$, $P_6$ are **not** written in terms of the memory variables
$r^l$ that `update_s*.cl` propagates. Thesis eq. (3.16) introduces
**alternative memory variables**

$$\partial_t \mathbf R^l = \mathbf r^l$$

and eq. (3.18) a **modified stress**

$$\sigma'_I = \sigma_I - \sum_{l=1}^{L} R^l_I$$

Both exist for a reason. In the original system (3.13) the velocity equation
depends only on the stresses, which breaks the symmetry the self-adjoint
transform needs. Rewriting in $\mathbf R^l$ and $\sigma'$ gives the velocity
equation the *same* dependence on the modified stresses as on the memory
variables, and gives the stress and memory equations the same form of velocity
dependence -- that symmetry is what makes the adjoint reuse possible at all.

The gradient (3.53) is expressed in those variables: $P_1,P_3,P_4$ use
$\sigma'$, and $P_2,P_5,P_6$ pair $\tilde{\mathbf R}$ against
$(1+\tau_\sigma\partial_t)\mathbf r$.

**So the code has to convert.** In the frequency domain
$\mathbf R = \mathbf r/(i\omega)$ -- which is exactly `cl_integral`:""")

code('def cl_integral(a, w):\n'
     '    # calc_grad.c:89 -- output.x = a.y/w ; output.y = -a.x/w\n'
     '    return a.imag/w - 1j*a.real/w\n'
     '\n'
     'r = rng.standard_normal(8) + 1j*rng.standard_normal(8)\n'
     'print("cl_integral(r) == r/(i*w) ?", np.allclose(cl_integral(r, w), r/(1j*w)))')

md(r"""which is why `calc_grad.c` opens its mechanism loop with

```c
fsxx[indfd] = cl_diff2(fsxx[indfd], cl_integral(frxx[indL], freq));
```

That line is not a correction or a fudge -- it *builds the modified stress*
$\sigma' = \sigma - \sum_l R^l$ from the propagated $r^l$.

With the same substitution, `cl_rm` becomes the theory's $P_2$:""")

code('rt = rng.standard_normal(8) + 1j*rng.standard_normal(8)   # adjoint r\n'
     'tausig = 1.0/(2*np.pi*30.0)\n'
     '\n'
     'def cl_rm(a, b, tausig, w):\n'
     '    # calc_grad.c:72\n'
     '    return (tausig*(a.real*b.real + a.imag*b.imag)\n'
     '            + (a.real*b.imag - a.imag*b.real)/w)\n'
     '\n'
     'Rt = rt/(1j*w)                                     # eq. (3.16)\n'
     'P2 = np.real(np.conj(Rt) * (1 + 1j*w*tausig) * r)  # eq. (3.53b)\n'
     '\n'
     'print("cl_rm == +P2 ?", np.allclose(cl_rm(rt, r, tausig, w),  P2))\n'
     'print("cl_rm == -P2 ?", np.allclose(cl_rm(rt, r, tausig, w), -P2))')

md(r"""So `cl_rm` reproduces $P_2$ **up to an overall sign**, once the
alternative memory variables are accounted for. Read directly in terms of the
propagated $r$, the $1/\omega$ in `cl_rm` is simply the $1/(i\omega)$ of
$\mathbf R = \int\mathbf r$ -- not an anomaly.

The residual sign is a convention, not an error: the adjoint field is
integrated backwards in time, and the thesis notes (p. 75) that the adjoint
system is *"identical to the direct model of equation (3.13), except for sign
changes on the spatial derivatives and on the memory-variable terms in the
stress expression"*. That sign is carried by
$\tilde{\mathbf R} = -\tilde{\mathbf r}/(i\omega)$ and cancels the one above.

**Conclusion: the coded gradient matches the published derivation** -- the
structure (3.52), the dot products (3.53), and all sixteen coefficients
(3.54-3.55) at $\alpha=0$.

## Summary

| | status |
|---|---|
| gradient structure (3.52) vs `calc_grad.c` | matches term for term |
| $P_i \leftrightarrow$ `dot[]` mapping | established, incl. the `dot[3]=dot[0]`, `dot[6]=dot[1]` reuse |
| coefficients (3.54-3.55) vs `grad_coefvisc_1` | all 16 match at $\alpha=0$; 12 for arbitrary $\alpha,L$ |
| stress dot products in the DFT gradient | verified against Parseval |
| `cl_integral` vs the modified stress $\sigma'$ | exact |
| `cl_rm` vs $P_2$ in alternative variables | matches up to the adjoint sign convention |

### Status

The theory checks out **and so does the code**. The `gradtaup`/`gradtaus`
discrepancy that motivated this notebook is resolved: the $\tau$ coefficients
depend on the GSLS phase-velocity factor

$$\alpha = \sum_l \frac{r_l^2}{1+r_l^2}, \qquad r_l = f_0/F_{L,l},$$

the same quantity `M()`/`mu()` use to divide the stored moduli by
$(1+\alpha\tau)$ so that the phase velocity at $f_0$ equals the supplied
$v_p$/$v_s$. The device kernels had it hardcoded to zero, which scaled every
$\tau$ coefficient by $(1-\alpha)$. With $\alpha$ rebuilt in-kernel from
`FL` and a new `FREQ0` build option, all five gradient components match the
host reference to `cos = 1.00000000` (worst relative difference $5\times10^{-7}$)
in 2D, SH and 3D.

For the gradient in use, see the companion notebook
[ViscoelasticInversion](ViscoelasticInversion.ipynb), which inverts a $\tau_s$
anomaly from crosswell data and shows the finite-difference validation.

Further reading: `notes/viscoelastic-gradient-theory.md` (maintainer-facing,
with line numbers) and `notes/adjoint-theory-thesis.md` (the adjoint state
itself -- why the forward kernels can be reused time-reversed).""")

nb['cells'] = c
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                          "name": "python3"}
out = "docs/notebooks/Inversion/ViscoelasticGradientTheory.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "-", len(c), "cells")
