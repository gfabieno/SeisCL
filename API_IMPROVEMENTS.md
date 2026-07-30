# SeisCL API improvement notes

Running list of API changes that would make SeisCL easier and more robust to
use, collected while reviewing and updating the documentation. Like
`CLAUDE.md`, this is meta-documentation for AI-assisted workflows — not part of
upstream, don't push it.

Each item states the evidence, why it hurts, and a concrete fix. Items marked
**verified** were reproduced on real hardware (Quadro P6000, CUDA 12.1) against
a local build of `torch-autograd-binding` @ `6aa3f10`.

---

## 1. `file_din` means two different things in two places — **verified**

`write_data()` (`SeisCL/SeisCL.py:702`) writes to
`os.path.join(workdir, self.file_din)`, i.e. it treats `file_din` as **relative
to `workdir`**. `callcmd()` (`SeisCL/SeisCL.py:534`) passes `self.file_din` to
the subprocess **raw**, and `execute()` never passes `cwd=`, so the C binary
resolves it against **the Python process's cwd**.

The two disagree for every relative `file_din` whenever `workdir != os.getcwd()`
— which is the default situation, since `workdir` defaults to `"./seiscl"`.
Reproduced with `workdir="./wd_asdoc"`:

```
write_data() wrote  : .../run3/wd_asdoc/SeisCL_din.mat   True
callcmd() passes    : SeisCL_din.mat
binary will resolve : .../run3/SeisCL_din.mat            False
RESULT: raised SeisCLError HDF5-DIAG: ... unable to open file
```

Today every caller must remember the incantation
`seis.file_din = os.path.abspath(os.path.join(seis.workdir, "SeisCL_din.mat"))`
— it appears in `SeisCL/tests/test_consistency.py` and in
`docs/notebooks/DeepLearning/SeisCLinPyTorch.ipynb` precisely because of this.

**Fix:** make the two agree. Either resolve `file_din` against `workdir` inside
`callcmd()` when it is relative (one line, matches `write_data()`'s existing
semantics and needs no caller changes), or make `file_din` a property whose
setter normalizes to an absolute path under `workdir`. A prior attempt at the
first option was reverted (`fb22f43` "Revert callcmd() to trust file_din as
given; fix callers instead") — but "fix callers instead" leaves the two methods
permanently inconsistent, so the property option is probably the better
reconciliation: it keeps `callcmd()` trusting `file_din` *and* makes the value
it trusts always correct.

Note also that `file_din` is derived once in `__init__`
(`self.file_din = file+"_din.mat"`, `SeisCL.py:414`) and does **not** track
later assignments to `self.workdir`, which is what makes the trap easy to hit.

### 1b. Related: the failure mode is unhelpful, not silent

Worth correcting a belief I had recorded earlier: a *forward-only* run does not
silently produce garbage from a missing din file. `set_forward()` writes only
`_csts.mat` and `_model.mat`, and the source/receiver geometry travels in
`_csts.mat` (`SeisCL.csts` includes `src_pos`/`rec_pos`/`src`) — so `_din.mat`
is genuinely not needed unless reference data is required (`gradout`, `rmsout`,
`resout`). A forward-only run with no din file at all is correct: verified
byte-identical output between the default `workdir` and the absolute-`file_din`
form.

When din *is* required and missing, the user gets a raw 20-line `HDF5-DIAG`
stack dump.

**Fix:** validate in Python before launching. If any of
`gradout`/`rmsout`/`resout` is set, check that the path `callcmd()` will
actually pass is readable, and raise `SeisCLError` naming that path.

## 2. Misspelled parameter names are silently ignored — **verified**

The documented workflow sets ~50 parameters by attribute assignment
(`seis.NT = 1000`). There is no `__setattr__` guard, so a typo silently creates
a new attribute and the run proceeds with the default:

```python
s.freesurface = 1   # real name is 'freesurf'
s.NTT = 5000        # real name is 'NT'
# -> accepted silently; s.freesurf is still 0, s.NT is still 875
```

For a class whose entire interface is attribute assignment, and whose names are
inconsistently abbreviated (`freesurf`, `nab`, `abpc`, `NT`, `FDORDER`,
`gradout`), this is the single easiest way to get a wrong answer that looks
right.

**Fix:** add `__slots__`, or a `__setattr__` that rejects names outside the
known set (the `csts` whitelist plus the non-`csts` attributes) with a
suggestion from `difflib.get_close_matches`.

## 3. `execute()` decides success from stderr, ignoring the exit code

```python
stdout, stderr = pipes.communicate()
if stderr:
    raise SeisCLError(stderr.decode())
```

Two failure modes: anything a library writes to stderr (an MPI or CUDA warning,
a deprecation notice) is reported as a fatal SeisCL error; and a nonzero exit
with empty stderr is reported as success.

**Fix:** branch on `pipes.returncode`, and include stderr in the message as
context rather than as the trigger.

## 4. The two Python APIs disagree on `params` dict keys

`SeisCL` uses `"vp"`/`"vs"`/`"rho"` for `param_type=0` (matching HDF5 dataset
names); `SeisCL.torch` requires the engine's internal `"M"`/`"mu"`/`"rho"` for
the *same* `par_type=0`, with the same values. Documented now (`docs/api.rst`,
and the PyTorch notebook), but it still silently breaks code ported between the
two — a wrong key raises nothing useful in the class path.

**Fix:** accept either spelling in `SeisCL.torch.seiscl_forward`, or ship a
`translate_params()` helper. Also worth rejecting unknown keys explicitly.

## 5. The two Python APIs disagree on array layout

`SeisCL` accepts `(NZ, NX)` numpy arrays and transposes internally in
`write_model()`. `SeisCL.torch` requires the caller to pre-flatten to the
engine's internal `(NX, NZ)` C order (`arr_zx.T.ravel()`). The notebook has to
define `flatten()`/`unflatten()` helpers just to bridge this.

**Fix:** have `seiscl_forward` accept `(NZ, NX)`-shaped tensors and do the
transpose itself, matching the class. Keep flat input as an accepted form.

## 6. `csts` (an internal dict) leaks into documented usage

`docs/notebooks/1_SimpleExample.ipynb` builds the model with
`np.zeros(seis.csts['N'])`, teaching users the internal representation even
though `seis.N` is a property that returns the same thing.

**Fix:** use `seis.N` in the docs (worth doing regardless), and consider making
`csts` a read-only mapping so it can't be mutated out from under the properties.

## 7. `SeisCL()` construction hard-fails when no backend is found

The constructor raises `SeisCLError` if neither `SeisCL_MPI` nor the
`seiscl:v0` Docker image is present. That makes the class unimportable-in-
practice for anyone reading the docs without a build, and means `docs/api.rst`
can only autodoc it because autodoc never instantiates.

**Fix:** defer backend discovery to `execute()`, or keep the check but make it a
warning at construction, so that geometry helpers, `ricker_wavelet()`,
`save_segy()` and the plotting utilities work without a compiled engine.

## 8. Smaller items

- `SeisCL.torch.Config.pref_device_type` is exposed but inert (`seiscl_core` is
  always CUDA; the device-type fallback in `src/Init_OpenCL.c` is entirely
  `#ifdef __SEISCL__`). Either honour it or drop it from the binding.
- `SeisCL.py` imports `matplotlib`, `obspy`, `scipy` unconditionally, so
  headless/minimal installs must carry all of them for a pure forward run.
  Make the plotting and SEGY imports lazy.
- `get_par`/`get_var`/`get_cst` (`src/clmodel.c`) return the address of a local
  stack variable on a name miss rather than NULL — `SeisCL/torch/bindings.cpp`
  already works around this with its own `find_par()`. Worth fixing at the
  source.
- `docs/conf.py` had `sys.path.insert(0, '../SeisCL')`, which put the *package
  directory* on the path so that `import SeisCL` resolved to `SeisCL/SeisCL.py`
  as a top-level module — working only by accident, and ambiguous with the
  editable install. Fixed to `'..'`; the package layout (a module `SeisCL.py`
  inside a package `SeisCL`, re-exported by `from .SeisCL import *`) is itself a
  recurring source of confusion.
