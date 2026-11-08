# AGENTS.md

Operating contract for AI coding agents (Claude Code, etc.) working in this
repository. Read this fully before writing any code. `CLAUDE.md` mirrors this
file for Claude-specific tooling; if the two ever disagree, **this file wins**.

## What pyRadBio is

A radiobiological-modelling library for ion-beam therapy, centred on the
linear-quadratic (LQ) model `-ln S(D) = αD + βD²`. It computes ion- and
energy-resolved α/β tables, z\* tables, survival curves, RBE, and the dose
required for a target survival level, for both single ions and mixed
radiation fields.

It implements these model families (inventory, references and open work:
`ROADMAP.md`):

- **LEM family (LEM I–IV)** — implemented here, since no maintained,
  permissively-licensed implementation exists. This is the bulk of the
  original work.
- **MKM family (MKM, modified MKM, MCF MKM, SMK, OSMK, IMK)** — implemented
  here on our own track-structure and microdosimetry engines. `MKMModel`'s
  default `z*` backend is the native `NativeMKMBackend`. The external
  MIT-licensed package **pyMKM** (`BeppeMagro/pymkm`, Magro et al.,
  *Computation* 2025, doi:10.3390/computation13110264) is an **optional
  validation dependency** (`mkm` extra), never a default: it is the
  **cross-check reference** for the family, and every native MKM piece must be
  tested against pyMKM where the two overlap (`requires_pymkm` tests, run by a
  dedicated CI job).
- **Stopping power** — every model's default LET source goes through the one
  seam `track.stopping_power.default_stopping_power()`, which returns
  `DefaultStoppingPower`: ICRU 90's procedure (`track.stopping.ICRU90Procedure`,
  within ~0.3% of ICRU 90/49 for p, alpha, C) for Z 1-18, Hubert et al. 1990's alpha
  scaling (`track.stopping.HubertAlphaScaling`) for Z 19-92 from 2.5 MeV/u, and the
  analytic `SimpleBetheStoppingPower` elsewhere, anchored to the adjacent branch at its
  range edge so the default is continuous (unvalidated there).
  The `track.stopping` classes (XSTAR, ICRU by edition, MSTAR, Bethe with
  switchable corrections, nuclear stopping), NS1970, SRIM tables, libdEdx (GPL,
  runtime opt-in) and pyMKM's tables via the pyMKM adapter are explicit opt-ins.
  See `docs/references/StoppingPower.md`. Raw stopping-power data stays
  git-ignored in `data/stopping/<source>/` with a committed `PROVENANCE.md`; only
  our own regridded tables ship. Do not bundle third-party stopping-power data
  whose licence restricts redistribution or use (e.g. Geant4's G4EMLOW), and never
  use SRIM data.
- **Phenomenological LET-based RBE models, RMF, Katz, NanOx** — implemented
  here from their source papers.

Above them sits a shared LQ / mixed-field / survival / RBE / benchmark layer
that is model-agnostic.

## Hard rules

1. **Test-Driven Development is mandatory.** For every behavioural change:
   write a failing test first (red), implement the minimum to pass (green),
   then refactor. Commits that add behaviour without a corresponding test
   will be rejected in review. Never delete, skip or `xfail` a test to make CI
   pass.
2. **Golden tests are sacred.** Tests marked `@pytest.mark.golden` pin model
   output to published reference values and must cite the DOI in the test
   docstring. Never weaken a golden tolerance to make a test pass — if it
   fails, the implementation is wrong (or the reference was misread; document
   which).
3. **Licence hygiene.** MIT only. Do not copy code from non-MIT/BSD/Apache
   sources, with one closed list of exceptions: code whose authors' notice grants
   use, copying, modification and distribution without copyleft. Each admitted
   source is named here, its notice is reproduced verbatim in
   `THIRD_PARTY_NOTICES.md` (which states that the notice, not MIT, applies to those
   files), and the derived files say so in their header. Admitted so far: **MSTAR
   3.12** (Paul & Schinner, freeware; condition: keep the authors' names in the
   source comments) → `pyradbio/track/stopping/mstar/`. Adding a source to this
   list is a maintainer decision; GPL/LGPL and other copyleft code never
   qualifies. `batuff/Survival` may inform *architecture discussions only* — do
   not copy its code; its numerical output is an independent cross-check, not
   ground truth (`docs/references/survival_repo_notes.md`). pyMKM is MIT and may
   be called as an optional validation dependency, not vendored and not made a
   default.
4. **Never commit PIDE data.** The PIDE database is distributed only after
   registration and may not be redistributed. The benchmark reads from the
   git-ignored `data/pide/` folder by default (override via `PYRADBIO_PIDE_DIR`
   or an explicit `--pide` path); only its `README.md` is tracked. Tested
   against PIDE 3.4.
5. **No silent physics.** Every model parameter, default, and unit must be
   explicit and documented at the point of definition with a paper/equation
   reference. No magic numbers.
6. **Numerical integration discipline** — see "Numerics" below. This is a
   correctness requirement, not a style preference.

## Toolchain

- Python ≥ 3.10, flat package layout, import root is `pyradbio`.
- Install dev env: `pip install -e ".[all]"`.
- Run tests: `pytest` (golden + unit). `pytest -m "not slow"` for the fast loop.
- Lint/format: `ruff check . && ruff format .`
- Types: `mypy pyradbio` (strict).
- All four (pytest, ruff check, ruff format --check, mypy) must pass before a
  task is considered done. CI enforces them.
- Docs: `pip install -e ".[docs]"` (Python ≥ 3.11), then `zensical build --strict`
  (CI runs it) or `zensical serve`.
- Docstrings are numpy-style **Markdown**, rendered by mkdocstrings. Link with
  ``[`Name`][]`` for a name in scope or ``[`Name`][pyradbio.full.path]``;
  document an attribute with a string literal directly below it; write notes as
  `!!! note` / `!!! warning` admonitions and code as fenced blocks. No Sphinx
  roles, `#:` comments or reST directives: they render as literal text, so
  `tests/docs/test_docstring_style.py` rejects them, and the strict build
  rejects a link that does not resolve.

## Optional dependency boundaries

Core library code must not casually import dependencies from optional extras.
Before using `pandas`, `pyarrow`, `openpyxl`, `matplotlib`, `numba`, `pymkm`, or
any other extra-only package, check `pyproject.toml` and keep the import inside
the feature layer that owns that extra. In particular:

- `pandas`/`pyarrow`/`openpyxl` belong to the `tables` extra and the table I/O
  layer (`pyradbio/tables/`). Model and backend seams should return small typed
  Python objects, not DataFrames.
- `matplotlib` belongs to `examples/` and `validation/` (through the
  matplotlib-free helpers of `pyradbio.plotting`); the only library module that
  may draw is `benchmark/report.py` (`tests/test_import_layers.py`).
- `numba` belongs behind explicit acceleration paths.
- `pymkm` belongs only in `models/mkm/pymkm_backend.py` (the pyMKM adapter:
  `PyMKMBackend`, `PyMKMStoppingPower`). `MKMModel` and shared core code must
  remain importable and fully functional without the `mkm` extra.
- Layering: `core/`, `track/` and `micro/` never import `models/`,
  `benchmark/`, `tables/` or `cli` (enforced by `tests/test_import_layers.py`).

## Numerics (read carefully)

These are the project's known correctness traps. Getting them wrong produces
plausible-but-wrong tables.

- **Mixed-field LQ mixing.** Dose-average α: `ᾱ = Σ dᵢ αᵢ / Σ dᵢ`.
  Combine β via the dose-averaged square root: `√β̄ = Σ dᵢ √βᵢ / Σ dᵢ`
  (Zaider–Rossi / Kanai convention). For the MKM, the mixing happens on the
  saturation-corrected dose-mean specific energy `z*_1D` (dose-averaged), not
  on α directly — see `mMKM_Inaniwa2010.md`, eq. (8) and appendix.
- **Spectral / intra-bin integration.** When integrating a model quantity `Q`
  over a fluence spectrum φ(E), the dose weight is `d ∝ φ(E)·S(E)` with `S`
  the (mass) stopping power. Across a wide energy bin, `S`, `z*`, and `α` all
  vary, and the *product* `S·Q` does **not** equal `S(E_center)·Q(E_center)`.
  The numerically significant error is the **intra-bin covariance** between
  stopping power and the looked-up quantity, not the bin-mean approximation
  alone. Integrate products (`S·z*`, `S·α`, `S·√β`) on a **fine logarithmic
  sub-grid (~16 nodes/bin)**, not at bin centres. `core/mixing.py` owns this
  (`dose_quadrature`), and models reach it only through the
  `BiophysicalModel.alpha_beta` template -- never write a per-model mixing loop.
  The mixing tests must include a bin wide enough to expose the covariance error and
  assert the sub-grid result against a dense-reference integral.
- **Survival → dose inversion.** Solving `S(D)=S_target` for `D` under the LQ
  model is a quadratic; prefer the closed form
  `D = (-α + √(α² - 4β·lnS)) / (2β)` and guard `β→0`.
- **LEM local-dose integrals.** The radial dose `D(r) ∝ LET/r²` is singular;
  respect `r_min`/`r_max` cutoffs per LEM version and normalise so the radial
  integral reproduces the LET. Validate `r_max(E)` and (LEM III) the
  velocity-dependent `r_min(β)=β·r_c` against the cited parameterisations.

## Layout & ownership

```
pyradbio/
  core/      LQ model, radiation fields, photon refs, cell/target, MIXING (sub-grid)
  track/     radial-dose models + stopping power (stopping/: water tables, ICRU 90
             procedure = default for Z 1-18, Hubert alpha scaling Z 19-92)
  micro/     microdosimetry: lineal-energy spectra and specific-energy engines
  models/    base ABC; lem/ = LEM I–IV; mkm/ = engine-agnostic MKMModel +
             MKMBackend protocol (NativeMKMBackend default, PyMKMBackend
             optional cross-check) and the spectrum-based MKM variants; phenomenological/,
             rmf/, katz/, nanox/
  tables/    alpha-beta & z* table builders + serialization (csv/parquet/pyMKM .txt)
  benchmark/ PIDE loader (user-supplied), paper reference data + inputs, metrics,
             sweeps, MC checkpoint I/O, report
  data/      bundled reference parameters (cell lines, etc.) — small, cited
  plotting.py  figure folder + save helpers for the scripts (no matplotlib import)
  cli.py     `pyradbio` entry point
examples/    runnable, notebook-pairable jupytext demos (see examples/README.md)
validation/  paper reconstructions + cross-code/data checks; data/ = tracked MC
             checkpoints (see validation/README.md)
scripts/     offline MC generators that write validation/data/ (see scripts/README.md)
```

Every model family implements one interface on `BiophysicalModel`
(`models/base.py`): `alpha_beta(field)`, `survival(field, D)`,
`dose_for_survival(field, S)`, `rbe_at_survival(field, S)`, `rbe_at_dose(field, D)`,
`reference()` and `alpha_beta_table(...)`, plus a `capabilities: ModelCapabilities`
ClassVar. A new model implements `_alpha_beta_ion` and declares `capabilities`;
model-specific axes (pO2, dose schedule, ...) are keyword-only arguments, never new
positional ones. Details: `docs/architecture.md`.

## Examples, validation and scripts

Three folders, three jobs; each README holds the authoring rules — follow them.

- `examples/` **demonstrates** a capability: a small gallery of **jupytext
  percent** scripts (plain `.py` that also opens as a notebook) that run with
  only the `plot` extra, no external data and no network, within the 120 s
  budget. The gallery is a fixed list (`tests/examples`), so a new example is a
  deliberate decision.
- `validation/` **scores** a model against a published model line, another
  code or a dataset (jupytext percent, minutes allowed). External data goes
  through `pyradbio.datasets`, optional extras through `_common.require_extra`.
  Every validation script supports `PYRADBIO_QUICK=1` (same figures, cheap
  settings).
- `scripts/` holds offline generators (argparse, `main()`, `--dry-run`, no I/O
  at import). Their outputs are the tracked checkpoints in `validation/data/`,
  each gated by a fast numeric-fingerprint test.

**Illustrating capabilities is a continuous task.** Each major capability
should be illustrated by an example **or a validation script**; add or extend
one whenever a change introduces a user-facing capability.

- Name examples after the **concept**, never after a milestone or task, so the
  gallery stays meaningful over time; name validation scripts after the paper
  or code they score (`smk_inaniwa2018.py`).
- Every example is smoke-tested automatically (runs, writes a PNG, stays within
  budget, imports only stdlib/numpy/scipy/matplotlib/pyradbio); every
  validation script is smoke-tested in quick mode by the `@slow` suite.

## Definition of done

A change is done when: its golden + unit tests are green, coverage on touched
modules ≥ 90%, ruff + mypy clean, its public API has docstrings with
equation/DOI references, and `CHANGELOG.md` is updated.

## When unsure

Prefer to (a) read the relevant summary in `docs/references/`, (b) read the
cited paper section, (c) inspect the actual pyMKM API (`python -c "import
pymkm; help(pymkm.mktable)"`) rather than assuming it. Record any assumption
you could not verify as a `# ASSUMPTION:` comment and a note in the PR.
