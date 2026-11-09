# Examples

The repository ships 16 short, self-contained demonstrations in
[`examples/`](https://github.com/e0404/pyRadBio/tree/develop/examples). Each one computes
something, prints a short summary and writes a plot. They use only the `plot` extra, need no
external data and no network, and are deterministic. The examples are not part of the
installed package, so run them from a clone of the repository.

## Running them

```bash
git clone https://github.com/e0404/pyRadBio.git
cd pyRadBio
pip install -e ".[plot]"          # pyradbio + matplotlib

python examples/survival_and_rbe.py
```

Figures go to `examples/figures/`, which is created on the first run and is git-ignored. Set
the `PYRADBIO_FIGURES_DIR` environment variable to write them somewhere else. The helpers
behind this, [`figures_dir()`][pyradbio.plotting.figures_dir] and
[`save_figure()`][pyradbio.plotting.save_figure] in `pyradbio.plotting`, import no plotting
library themselves, and `save_figure` prints the path it wrote.

### As notebooks

Every example is a [jupytext](https://jupytext.readthedocs.io) **percent** script: a plain
`.py` file with cells delimited by `# %%`, whose header declares the pairing. It runs
directly with `python`, and it also opens as a Jupyter notebook:

```bash
pip install jupytext
jupytext --to ipynb examples/survival_and_rbe.py            # one-off conversion
jupytext --set-formats ipynb,py:percent examples/survival_and_rbe.py   # keep a pair in sync
```

With the Jupytext extension installed, JupyterLab can also open the `.py` files directly. The
`.py` file is the source of truth, and generated notebooks are not tracked.

## The scripts

Runtimes are for a full run on a laptop, as listed in the repository's
[`examples/README.md`](https://github.com/e0404/pyRadBio/blob/develop/examples/README.md).

### Basics and physics

| Script | What it shows | Guide page | Runtime |
| --- | --- | --- | --- |
| [`survival_and_rbe.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/survival_and_rbe.py) | LQ survival curves, D10, and RBE at iso-survival for a photon reference and an ion | [Core concepts](concepts.md) | ~2 s |
| [`water_stopping_power.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/water_stopping_power.py) | LET against energy in water for several ions, from `default_stopping_power()` | [Physics: stopping power](physics.md#stopping-power) | ~4 s |
| [`radial_dose_profiles.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/radial_dose_profiles.py) | Amorphous-track radial dose \( D(r) \) for LEM I, II, III and Kiefer-Chatterjee | [Physics: radial dose](physics.md#radial-dose-and-track-structure) | ~10 s |
| [`lineal_energy_spectra.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/lineal_energy_spectra.py) | Analytic single-event lineal-energy spectra and their moments for several ions, site sizes and shapes | [Physics: microdosimetry](physics.md#microdosimetry) | ~21 s |
| [`dose_protraction_and_bed.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/dose_protraction_and_bed.py) | The time axis: Lea-Catcheside \( G \) for fractionated and continuous schedules, and the BED/EQD2 fraction-size isoeffect | [Physics: time axis](physics.md#the-time-axis) | ~2 s |
| [`mixed_field_parameters.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/mixed_field_parameters.py) | `alpha_beta(field)` on binned spectrum components: dose-weighted mixing, one wide energy bin, and the bin-centre error the quadrature removes | [Radiation fields and mixing](fields.md) | ~15 s |

### Models

| Script | What it shows | Guide page | Runtime |
| --- | --- | --- | --- |
| [`lem1_variants.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/lem1_variants.py) | The LEM I approximation ladder for alpha and beta, and the historically coherent bundles | [Local effect model](models/lem.md) | ~24 s |
| [`mkm_zstar_and_alpha.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/mkm_zstar_and_alpha.py) | MKM: carbon alpha and \( z^*_{1D} \) against LET, then \( z^*_{1D} \) against energy for H-Ne, on the native backend | [MKM family](models/mkm.md) | ~30 s |
| [`mcf_mkm_rbe_vs_let.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/mcf_mkm_rbe_vs_let.py) | MCF MKM and mMKM RBE and alpha/beta against LET for H-Ne, driven by the bundled Hartzell 2025 spectra | [MKM family](models/mkm.md) | ~3 s |
| [`smk_nucleus_stochastics.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/smk_nucleus_stochastics.py) | What SMK adds over mMKM: a quality-dependent beta, non-LQ survival, and the cost of the site geometry | [MKM family](models/mkm.md) | ~13 s |
| [`osmk_oxygen_effect.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/osmk_oxygen_effect.py) | OSMK 2021 and 2023: survival at graded pO2, OER against LET, and what the 2023 revision changes | [MKM family](models/mkm.md), [Physics: oxygen axis](physics.md#the-oxygen-axis) | ~25-50 s |
| [`phenomenological_rbe_models.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/phenomenological_rbe_models.py) | The RBE-min/RBE-max family: RBE against LET, tissue and dose, and the LQ pair each model implies | [Phenomenological proton RBE](models/phenomenological.md) | ~4 s |
| [`rmf_dsb_and_binary_misrepair.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/rmf_dsb_and_binary_misrepair.py) | RMF: MCDS DSB yields, the one-track against intratrack-pair split, and RBE's dependence on \( (\alpha/\beta)_x \) alone | [RMF, Katz and NanOx](models/other.md) | ~3 s |
| [`katz_ion_kill_and_gamma_kill.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/katz_ion_kill_and_gamma_kill.py) | Katz 1971: the ion-kill and gamma-kill split, the lost shoulder, and the overkill turnover | [RMF, Katz and NanOx](models/other.md) | ~4 s |
| [`nanox_local_and_global_events.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/nanox_local_and_global_events.py) | NanOx: the effective local lethal function, and why the missing nanodosimetric spectrum cannot be substituted | [RMF, Katz and NanOx](models/other.md) | ~4 s |
| [`model_comparison_alpha_beta.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/model_comparison_alpha_beta.py) | LEM I-IV, MCF MKM, mMKM and RMF side by side on one V79 photon reference: alpha(E) and beta(E) for a proton and a carbon ion | [Models overview](models/index.md) | ~25-50 s |

!!! note "Illustrative parameters"
    The examples use illustrative parameter values and label them as such; where a value
    comes from a paper, the script cites its DOI. They demonstrate a capability, so they do
    not substitute for fitting a model to your own cell line.

## Validation and scripts

Two further folders live in the repository and are also run from a clone.

[`validation/`](https://github.com/e0404/pyRadBio/tree/develop/validation) holds the paper
reconstructions, cross-code comparisons and the comparisons against external data. Where an
example demonstrates a capability in seconds, a validation script scores a model against a
published model line, another code or a dataset, and may take minutes. Each script is a
jupytext percent file like the examples. Run one from the repository root, with
`PYRADBIO_QUICK=1` for a quick mode that draws the same figures with cheap settings (only
the full mode's numbers are quoted):

```bash
python validation/elsasser2008_lem3.py
PYRADBIO_QUICK=1 python validation/elsasser2007_lem2.py
```

A script that lacks its external data or an optional extra stops with a one-line message.
Figures go to `validation/figures/`. The folder's
[`README.md`](https://github.com/e0404/pyRadBio/blob/develop/validation/README.md) lists every
script with its paper, data and extras, runtime and the checkpoints it reads.

[`scripts/`](https://github.com/e0404/pyRadBio/tree/develop/scripts) holds the offline
generators: long Monte-Carlo runs whose outputs the validation scripts read as tracked
checkpoints in `validation/data/`. They are not demonstrations and draw no figures. Every
generator accepts `--dry-run`, which prints the plan and a cost estimate without computing
anything, and resumes point by point if interrupted. Some take hours, so the
[`scripts/README.md`](https://github.com/e0404/pyRadBio/blob/develop/scripts/README.md)
documents the cost of each. You only need them to regenerate a checkpoint, as a fresh clone
already contains the tracked results.
