# pyRadBio examples

Small, self-contained scripts that demonstrate the library on common
radiobiology tasks. Each example computes something, prints a short summary,
and writes a plot into `examples/figures/`.

The examples are written in [jupytext](https://jupytext.readthedocs.io)
**percent** format: they are plain `.py` scripts you can run directly, *and*
they open as Jupyter notebooks (cells are delimited by `# %%`).

Paper reconstructions, cross-code checks and comparisons against external data
live in [`validation/`](../validation/README.md); offline Monte-Carlo generators
live in [`scripts/`](../scripts/README.md).

| Script | What it shows | Runtime |
| --- | --- | --- |
| `survival_and_rbe.py` | LQ survival curves, D10, and RBE at iso-survival for a photon reference and an ion. | ~2 s |
| `dose_protraction_and_bed.py` | The time axis: the Lea-Catcheside `G` for fractionated and continuous schedules, what protraction does to a survival curve, and the fraction-size isoeffect in BED/EQD2. | ~2 s |
| `water_stopping_power.py` | LET vs energy in water for several ions, from `default_stopping_power()`. | ~4 s |
| `radial_dose_profiles.py` | Amorphous-track radial dose D(r) for every track model (LEM I/II/III, Kiefer-Chatterjee, ...). | ~10 s |
| `lineal_energy_spectra.py` | Analytic single-event lineal-energy spectra f1(y)/d1(y) and y_F, y_D, y* for several ions, site sizes and shapes (sphere vs TEPC-like cylinder). | ~21 s |
| `mixed_field_parameters.py` | `alpha_beta(field)` on binned spectrum components: dose-weighted mixing (Pfuhl 2020, doi:10.1667/RR15456.1, eqs. 9-11), one wide energy bin, and the bin-centre error the bin quadrature removes. | ~15 s |
| `lem1_variants.py` | The LEM I approximation ladder on two axes, alpha (`cross_section` / `lethal_central_saturating` / `lethal_full`) and beta (`scholz1997_two_track` / `_stochastic` / `rapid_lem` / `slope_matching`), and the historically coherent bundles. | ~24 s |
| `mkm_zstar_and_alpha.py` | MKM (Inaniwa 2010, doi:10.1088/0031-9155/55/22/008): carbon alpha and z*_1D vs LET, then z*_1D vs energy for H-Ne (the Fig. 5 family), on the native backend. | ~30 s |
| `mcf_mkm_rbe_vs_let.py` | MCF MKM and mMKM RBE_10 and alpha/beta vs LET for H-Ne, driven by the bundled Hartzell 2025 AMF spectra; the overkill peak and the MCF-vs-mMKM beta divergence. | ~3 s |
| `smk_nucleus_stochastics.py` | What SMK (Inaniwa & Kanematsu 2018, doi:10.1088/1361-6560/aabede) adds over mMKM: a quality-dependent beta, non-LQ survival, the dose ceiling of the truncated expansion, and the cost of the site geometry (disk vs sphere: -9..+29% on z*_1D over 1-100 MeV/u, +39% at 400 MeV/u). | ~13 s |
| `osmk_oxygen_effect.py` | OSMK 2021 and 2023 (doi:10.1088/1361-6560/ac29cc, doi:10.1093/jrr/rrad049): survival at graded pO2, OER vs LET with its particle-type dependence, and what the 2023 event-by-event revision changes. | ~25-50 s |
| `phenomenological_rbe_models.py` | The RBE_min/RBE_max family (Wilkens & Oelfke, Carabe, Frese, Wedenberg, McNamara, Mairani helium): RBE vs LET, tissue and dose, and the LQ pair each model implies. | ~4 s |
| `rmf_dsb_and_binary_misrepair.py` | RMF (Carlson 2008, doi:10.1667/RR1046.1; Frese 2012, doi:10.1016/j.ijrobp.2011.06.1983): MCDS DSB yields, the one-track vs intratrack-pair split, the absent RBE turnover, and RBE's dependence on (alpha/beta)_x alone. | ~3 s |
| `katz_ion_kill_and_gamma_kill.py` | Katz 1971 (doi:10.2307/3573247): the ion-kill / gamma-kill split, the loss of the shoulder, the overkill turnover, and k-vs-LET curves that do not overlap across ions. | ~4 s |
| `nanox_local_and_global_events.py` | NanOx (Cunha 2017; Alcocer-Avila 2022, 2023; Monini 2020): the effective local lethal function, local events alone giving a pure exponential, and why the missing nanodosimetric spectrum cannot be substituted. | ~4 s |
| `model_comparison_alpha_beta.py` | LEM I-IV, MCF MKM, mMKM and RMF side by side on one V79 photon reference: alpha(E) and beta(E) for a proton and a carbon ion. | ~25-50 s |

Runtimes are full runs on a laptop; `tests/examples` fails an example that takes
longer than 120 s (locally, without coverage).

## Requirements

The examples plot with matplotlib (the `plot` extra); pairing/opening them as
notebooks needs jupytext:

```bash
pip install -e ".[plot]"
pip install jupytext      # only needed for the notebook workflow
```

## Running as scripts

```bash
python examples/survival_and_rbe.py
```

Figures are written to `examples/figures/` (created on first run, git-ignored;
`PYRADBIO_FIGURES_DIR` overrides the folder).

## Opening as notebooks

Pair a script with an `.ipynb` (kept in sync as you edit either side), or do a
one-off conversion:

```bash
jupytext --set-formats ipynb,py:percent examples/survival_and_rbe.py  # pair
jupytext --sync examples/survival_and_rbe.py                          # sync pair
jupytext --to ipynb examples/survival_and_rbe.py                      # one-off
```

With the Jupytext extension installed, JupyterLab/Notebook can also open the
`.py` files directly as notebooks. The committed `.py` is the source of truth;
generated `.ipynb` files are not tracked.

## Authoring new examples (contributors and coding agents)

An example **demonstrates** a capability. If a script scores a model against a
paper, another code or a dataset, it belongs in `validation/`; if it computes
reference data offline, it belongs in `scripts/`.

Keep examples **concept-oriented, not tied to the implementation plan or
milestones**: name them after the radiobiology idea (`survival_and_rbe`,
`radial_dose_profiles`), never after a phase or version.

Conventions:

- **Format:** jupytext percent. Start with the jupytext + kernelspec YAML header
  (copy one from an existing example), then alternate `# %% [markdown]` and
  `# %%` cells. Do **not** wrap the body in a `main()` function or an
  `if __name__ == "__main__"` guard: top-level cells must execute in order both
  as a script and as a notebook.
- **Public API first:** import from `pyradbio` (or a documented submodule). If an
  example needs something awkward to reach, that is a hint the public API needs
  improving; prefer fixing the API over reaching into internals.
- **Imports:** only the standard library, numpy, scipy, matplotlib and
  `pyradbio`, and never the pyMKM adapter (`pyradbio.models.mkm.pymkm_backend`).
  `tests/examples/test_examples_imports.py` checks this statically, because CI
  installs every extra and could not tell otherwise.
- **Figures:** `FIGURES = figures_dir(globals().get("__file__"))`, then
  `save_figure(fig, "<name>.png", directory=FIGURES)` (both from
  `pyradbio.plotting`); this works as a script and in a notebook and prints the
  path. Do not call `matplotlib.use(...)`; let the backend default so notebooks
  render inline.
- **Self-contained & cheap:** only the `plot` extra, no external data, no
  network, deterministic (seed any RNG). Aim for well under a minute and about
  200 lines; the smoke test enforces the 120 s ceiling.
- **Keep it honest:** illustrative parameter values are fine, but label them as
  such; cite a DOI when a value comes from a paper.

Every example is smoke-tested automatically (`tests/examples/test_examples_smoke.py`:
it runs, writes a PNG and stays within the budget). The gallery itself is a
fixed list (`test_examples_set_is_exactly`), so adding or removing an example
means updating that list and this table.

See the repository `AGENTS.md` ("Examples, validation and scripts") for *when*
to add one.
