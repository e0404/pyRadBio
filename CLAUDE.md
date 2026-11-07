# CLAUDE.md

Guidance for Claude Code in this repository.

**`AGENTS.md` is the authoritative contract — read it first.** This file adds
only Claude-specific workflow notes and does not override anything there.

## Start-of-task checklist

1. Read `AGENTS.md` (rules, numerics, layout, definition-of-done).
2. For the model you are touching, read its summary in `docs/references/`
   before writing code.

## Workflow (TDD loop)

- Write the failing test first; `pytest -m "not slow"` runs the fast suite.
- Make it pass with the minimum change.
- `ruff format . && ruff check --fix . && mypy pyradbio` before committing.
- One logical change per commit; conventional-commit messages
  (`feat(lem): …`, `test(core): …`, `fix(mixing): …`).

## Subagent delegation

These are good candidates to farm out to subagents and fold the results back
into `docs/references/`:

- Digest a primary paper into / refine its summary file (template in
  `docs/references/README.md`).
- Check a pyMKM interop question against the installed package source and
  `docs/references/pyMKM_API.md`; do **not** assume the API from memory.

## Things Claude tends to get wrong here — don't

- Do **not** break the MKM backend seam. `models/mkm/model.py` (`MKMModel`)
  owns only the biophysics and delegates `z*_1D` to an `MKMBackend`
  (`models/mkm/base.py`). `NativeMKMBackend` is the **default** backend;
  pyMKM (`models/mkm/pymkm_backend.py`, the only file that imports pyMKM) is an
  optional cross-check reference, never a default. Any new engine is another
  backend implementing the protocol, tested against pyMKM where they overlap;
  never make `MKMModel` import pyMKM directly.
- Do **not** reach for pyMKM (or a new Bethe instance) as a stopping-power
  source. The library default is `default_stopping_power()` (ICRU 90's procedure
  for Z 1-18, Hubert alpha scaling for Z 19-92 from 2.5 MeV/u, analytic Bethe
  anchored to them elsewhere); go through that factory rather than
  hard-coding a source. Tests that pin Bethe numbers construct
  `SimpleBetheStoppingPower` explicitly.
- Do **not** import optional-extra dependencies just because they are available
  in your local venv. `pandas`, `pyarrow`, and `openpyxl` belong to the
  `tables` extra / table I/O layer; `matplotlib` belongs to the `examples/`
  and `validation/` scripts; `numba` belongs behind acceleration gates;
  `pymkm` belongs only in the
  guarded MKM backend or a lazy adapter. For model and backend seams, prefer
  small typed Python objects over DataFrames.
- Do **not** evaluate spectral product integrals at bin centres (see
  AGENTS.md → Numerics). This is the single most common silent error.
- Do **not** loosen a `@pytest.mark.golden` tolerance to go green.
- Do **not** download or commit PIDE. The loader defaults to the git-ignored
  `data/pide/` folder (only its `README.md` is tracked) and also accepts an
  explicit `--pide` path / `PYRADBIO_PIDE_DIR`. Never add a PIDE file to git.
- Do **not** invent paper values — cite the DOI and the table/equation, and
  if a value is not in a paper you have, mark it `# UNVERIFIED` and note it in
  the model's summary in `docs/references/`.

## Verifying the environment

```bash
pip install -e ".[all]"
pytest -q
pip install -e ".[mkm]" && pytest -m requires_pymkm   # optional pyMKM cross-checks
```
