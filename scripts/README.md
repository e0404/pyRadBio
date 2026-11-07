# pyRadBio scripts

Batch and offline generators: long Monte-Carlo runs and data-preparation jobs
whose outputs other parts of the repository read. They are not demonstrations
(see `examples/`) and draw no figures (see `validation/`).

## Conventions

- A plain module (not jupytext) with an `argparse` parser, a
  `main(argv: list[str] | None = None) -> int` function and an
  `if __name__ == "__main__": raise SystemExit(main())` guard.
- **No I/O at import.** Importing a generator reads no dataset, file or
  environment-dependent path; everything is resolved inside `main()`.
- **`--dry-run`** prints the plan (points, sample sizes, output path) and a cost
  estimate, then returns 0 without computing anything or touching a dataset.
- **`--out DIR`** names the output folder. Generators write the tracked
  checkpoints in `validation/data/` (or an external dataset folder such as
  `$PYRADBIO_SURVIVAL_DIR`) and record their provenance (stopping power, engine,
  seeds) in the file they write.
- The `fast` extra (numba) is preferred for long runs; pin `engine="numpy"`
  wherever a result must be reproducible bit for bit.

`tests/scripts/test_scripts_cli.py` checks the first three rules for every
top-level generator in this folder.

## Generators

| script | writes | cost | regenerate |
|---|---|---|---|
| `lem1_mc_reference.py alpha-beta` | `validation/data/mc_results.csv` (Scholz 1997 stochastic-LEM alpha/beta points) | ~1 h 50 min with numba; `--high-energy` adds hours | resumes point by point; `--only particle:E` recomputes points, `--fresh` the whole catalogue |
| `lem1_mc_reference.py z-slope` | `validation/data/sk1997_z_slope.csv` (Scholz 1997 Fig. 4 specific-energy slope) | ~7 min with numba | as above |
| `lem1_mc_reference.py sk1996-final-slope` | `validation/data/sk1996_mc_observables.csv` (Scholz & Kraft 1996 final slope) | hours | as above |
| `lem4_friedrich2012_mc.py` | `validation/data/lem4_friedrich2012_mc.csv` (Friedrich 2012 Fig. 4 LEM IV MC, one row per LET x seed) | ~1.6-2.2 h with numba; run detached | resumes; `--fresh` |
| `lem_mc_vs_survival.py` | `validation/data/survival_our_mc.json` (our MC points for the Survival comparison) | overnight; needs Survival's SRIM tables (`PYRADBIO_SURVIVAL_SRC_DIR`) | resumes; `--fresh` |
| `survival_paired_mc.py` | Survival's own reference-MC points, into the local `PYRADBIO_SURVIVAL_DIR` (never committed; Survival is GPL-2) | days; needs a locally built `survival.exe` | resumes from its `checkpoint.json`; `--refine` re-runs points short of target |

`lem1_mc_reference.py` refuses to extend a file whose fingerprint is stale at
the same stopping power; `--stopping-power bethe` is allowed and drops (or
recomputes) the rows recorded at another LET. Every generator accepts
`--dry-run`, which also prints its cost estimate.

## `scripts/stopping/`

The water stopping-power pipeline (`fetch_sources.py`, `build_tables.py`,
`mstar_reference.py`) lives in its own subfolder with its own tests
(`tests/scripts/test_*stopping*.py`); it fetches the raw sources into the
git-ignored `data/stopping/` and writes the shipped tables under
`pyradbio/track/stopping/data/`.
