# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.18] - 2026-11-07

### Added
- **Documentation:** the README, `docs/architecture.md`, paper digests and reconstruction notes in `docs/references/`, the model inventory and open work in `ROADMAP.md`, and the contributor and coding-agent rules in `AGENTS.md` / `CLAUDE.md`.
- **Data provenance:** every shipped data file names its source and DOI (or its provenance record) in its header; `THIRD_PARTY_NOTICES.md` credits the publications and RadPhysBio (CC BY 4.0) the data come from; PIDE and other registration-gated data are never bundled.
- `CITATION.cff`.

## [0.0.17] - 2026-11-07

### Added
- **Validation:** `validation/` holds the paper reconstructions and the comparisons against TRiP98, the Survival code (including its reference Monte Carlo), pyMKM, DeCunha 2025, Hartzell 2025 and PIDE (external data, user-supplied), each with a `PYRADBIO_QUICK=1` quick mode.
- `scripts/` holds the offline Monte-Carlo generators (`--dry-run`), whose checkpoints are tracked in `validation/data/` and gated by fast numeric-fingerprint tests.

## [0.0.16] - 2026-11-07

### Added
- **Examples:** `examples/` holds 16 short demonstrations (plot extra only, no external data), each run by a smoke test.
- `pyradbio.plotting.figures_dir()` / `save_figure()`, matplotlib-free figure-path helpers for scripts (`PYRADBIO_FIGURES_DIR` overrides the folder).

## [0.0.15] - 2026-11-07

### Added
- **Benchmarking against measured data** (`pyradbio.benchmark`): `pyradbio benchmark` scores any registered model against PIDE (user-supplied, never bundled), RadPhysBio or a custom CSV and states each dataset's caveats in its report; `benchmark.crosscheck` compares the databases against each other.
- **CLI:** the `pyradbio survival`, `pyradbio table` and `pyradbio benchmark` commands; `pyradbio table` selects the stopping power and model options from the same literals the models validate.

## [0.0.14] - 2026-11-07

### Added
- **NanOx** (`models/nanox/`, Cunha et al. 2017; Monini et al. 2020): the assembly and calibration layer with published cell lines. The ion route raises, because NanOx's nanodosimetric Monte Carlo is not public.
- Interface tests across every model family: capabilities, binned-spectrum mixing and alpha-beta tables.

## [0.0.13] - 2026-11-07

### Added
- **Katz cellular track-structure model** (`models/katz/`, Katz et al. 1971): ion-kill and gamma-kill modes in the grain-count regime, with four published parameter sets and the LET as keyword `let_keV_um=`. It is not an LQ model and has no `beta`.

## [0.0.12] - 2026-11-07

### Added
- **RMF** (`models/rmf/`, Carlson et al. 2008; Frese et al. 2012) on the MCDS DSB yields.

## [0.0.11] - 2026-11-07

### Added
- **Phenomenological LET-based proton RBE models** (`models/phenomenological/`): thirteen published fits on a shared `RBEMinMaxModel` base (Wilkens & Oelfke, Carabe, Frese, Wedenberg, McNamara, Mairani, Tilly 2 and 10, Chen, Jones, Rorvik, Peeler, Unkelbach), with RBE at a given survival level, dose or reference dose.

## [0.0.10] - 2026-11-07

### Added
- OSMK 2021 and 2023 (Inaniwa et al. 2021; Inaniwa & Kanematsu 2023) with the oxygen partial pressure as keyword `pO2_mmHg=`, and IMK (Matsuya et al. 2025) with its dose-rate dependence via `schedule=`.
- Published parameter sets as constructors, e.g. `IMKModel.hsg_matsuya2025()`.
- Paper reconstructions: digitized figures with loaders and golden tests for Inaniwa et al. 2021 and Matsuya et al. 2025.

## [0.0.9] - 2026-11-07

### Added
- **MKM** (`models/mkm/`): an engine-agnostic `MKMModel` behind the `MKMBackend` protocol, with `NativeMKMBackend` (analytic impact-parameter `z*_1D`) as the default and `PyMKMBackend` as an optional cross-check (`mkm` extra, the only module that imports pyMKM).
- An import-layer test: `core/`, `track/` and `micro/` never import `models/`, `benchmark/`, `tables/` or `cli`, and only the MKM adapter imports pyMKM.
- z* tables, CSV and Parquet table output (`tables` extra) and pyMKM `.txt` interop (`tables/zstar.py`, `tables/io.py`).
- Spectrum-based MKMs with a default microdosimetry engine: MCF MKM (Parisi et al. 2022), modified MKM (Kase et al. 2006; Inaniwa et al. 2010) and SMK (Inaniwa & Kanematsu 2018, with Sato & Furusawa 2012's domain and nucleus treatments); the impact-parameter microdosimetry is cached per geometry (`clear_microdosimetry_cache()`).
- Published parameter sets as constructors, e.g. `StochasticMKM.hsg_inaniwa2018()`.
- Paper reconstructions: digitized figures with loaders and golden tests for Inaniwa & Kanematsu 2018 and Sato & Furusawa 2012.

## [0.0.8] - 2026-11-07

### Added
- **Microdosimetry** (`pyradbio.micro`): single-event lineal-energy spectra from an analytic engine, an amorphous-track engine with optional stochastic delta rays, Sato's analytical microdosimetric function and tabulated spectra (incl. the Hartzell 2025 AMF curves); multi-event specific-energy densities; and an analytic impact-parameter integrator.
- MCDS DSB yields (`micro/mcds.py`, Semenenko & Stewart 2006).

## [0.0.7] - 2026-11-07

### Added
- **LEM IV** (DSB-pattern / cluster-index formalism, Elsasser et al. 2010; Friedrich et al. 2012) with the `ClusterResponse` protocol.
- LEM IV: the GLOBLE effect model, explicit-track and intertrack Monte-Carlo `beta`, the local-dose full simulation, a discrete-DSB engine, and selectable cluster-index estimators.
- Paper reconstructions: digitized figures with loaders and golden tests for Elsasser et al. 2010 and Friedrich et al. 2012.

## [0.0.6] - 2026-11-07

### Added
- **LEM II** (cluster extension, Elsasser & Scholz 2007) and **LEM III** (velocity-dependent track core, Elsasser et al. 2008), sharing `LEMBase` and the local-dose Monte Carlo with LEM I.
- Paper reconstructions: digitized figures with loaders and golden tests for Elsasser & Scholz 2007 and Elsasser et al. 2008, the printed track-structure table of Elsasser et al. 2008b, and the Manganaro et al. 2018 parameters.

## [0.0.5] - 2026-11-07

### Added
- **LEM I** (`models/lem/`, Scholz & Kraft 1996, with the Scholz et al. 1997 approximation) on a shared `LEMBase`.
- LEM: selectable `initial_slope_model` and `beta_model` routes (slope matching, Scholz 1997 two-track, rapidLEM, local dose, Monte Carlo), including a TRiP98/rapidLEM compatibility mode, and one execution-engine literal, `Engine` (`"auto"`/`"numpy"`/`"numba"`).
- A local-dose Monte Carlo for the LEM with `numpy` (reproducible), `numba` and multiprocess engines.
- Paper reconstructions: digitized figures with loaders and golden tests for Scholz & Kraft 1996 and Scholz et al. 1997.
- Benchmark helpers: `benchmark.sweeps` (`sweep_energies`, `sweep_lets`, `LetSweep`), `benchmark.metrics.shape_residuals` / `ratio_summary`, and `benchmark.checkpoint` (CSV/JSON Monte-Carlo checkpoints with a provenance header and a numeric-fingerprint check).

## [0.0.4] - 2026-11-07

### Added
- **One model interface** on `BiophysicalModel`: `alpha_beta(field)`, `survival(field, dose_Gy)`, `dose_for_survival(field, S)`, `rbe_at_survival(field, S)`, `rbe_at_dose(field, dose_Gy)`, `reference()` (a `PhotonParameters`, or `PhotonReferenceError`) and `alpha_beta_table()`; RBE is formed on the log-kill, so it stays finite where survival underflows. Model-specific axes are keyword-only.
- `ModelCapabilities` (`ion_route`, `mixed_field`, `lq_exact`, `let_route`, `spectrum_route`) and `LETModel` with the LET-route mirror (`alpha_beta_from_let`, `survival_from_let`, `dose_for_survival_from_let`, `rbe_at_survival_from_let`, `rbe_at_dose_from_let`).
- Models are frozen dataclasses (copy with `dataclasses.replace`), published parameter sets are model instances applied with `dataclasses.replace(preset, **overrides)`, and every option is a `Literal` validated at construction.
- Typed alpha-beta tables (`tables/`) whose headers record the resolved model and stopping-power settings.

## [0.0.3] - 2026-11-07

### Added
- **Track structure:** `ScholzKraftRadialDose` (`r_max = gamma E^delta`, LEM I/II/III constructors, velocity-dependent core, sigma = 4 nm radical diffusion that conserves the track's energy out to `r_support_um`) and the Kiefer-Chatterjee radial dose.

## [0.0.2] - 2026-11-07

### Added
- **Stopping power:** `default_stopping_power()`, the one factory every model default goes through, returns `DefaultStoppingPower`: `ICRU90Procedure` for Z 1-18, `HubertAlphaScaling` (Hubert, Bimbot & Gauvin 1990 on ICRU 90 alphas) for Z 19-92 from 2.5 MeV/u, and `SimpleBetheStoppingPower` elsewhere, anchored to the adjacent branch at its range edge.
- **Water stopping powers** (`track/stopping/`), all with one interface (energy per nucleon, out-of-range errors, per-ion metadata, `quantity="electronic"|"total"`): `XSTAR` (NIST PSTAR/ASTAR, MSTAR or z_eff-scaled PSTAR for Z 3-18), `ICRU(edition="2005"|"2009"|"2016")` (ICRU 49/73/90), a Python port of MSTAR 3.12 (under its authors' freeware notice, `THIRD_PARTY_NOTICES.md`), `Bethe` (PDG 2024) with selectable effective charge and density-effect, shell, Barkas, Bloch, Mott, Ahlen and Lindhard-Sorensen corrections, `ICRU90Procedure` and `WaterNuclearStopping` (ICRU 49 Table 4.1, Moliere protons).
- The stopping-power tables ship as our own regridded copies; raw sources stay git-ignored in `data/stopping/<source>/` with a committed `PROVENANCE.md`, and `scripts/stopping/fetch_sources.py` / `build_tables.py` fetch and rebuild them deterministically. Measured accuracy of every source: `docs/references/StoppingPower.md`.
- Further stopping-power sources and helpers: the Northcliffe & Schilling 1970 water table, a SRIM reader, optional libdedx, `energy_for_let_MeV_u()` (incl. `branch="bracket"`), `describe_stopping_power()`, `let_from_mass_stopping_power()`, isotope masses to Z 92, and the unit constants `MEV_PER_CM_TO_KEV_PER_UM` and `AVOGADRO_PER_MOL`.
- External data folders (PIDE, RadPhysBio, SRIM, libdedx) are configured in one place, `pyradbio/datasets.py`, from environment variables or a `.env` file.

## [0.0.1] - 2026-11-07

### Added
- **LQ core** (`core/lq.py`): `LQParameters`, `survival`, `log_kill`, `dose_for_survival` (stable quadratic inverse with a continuous `beta -> 0` limit), `rbe`, `bed` / `eqd2`, and the LQ fits `fit_lq_linearised()` / `fit_lq_weighted()`.
- **Radiation fields and mixing** (`core/radiation.py`, `core/mixing.py`): `Ion`, `SpectrumComponent` (mono-energetic or binned), `RadiationField`, fluence/dose conversion; mixed fields dose-average alpha and sqrt(beta) (Zaider-Rossi), and binned components are integrated on a log sub-grid with phi(E)S(E) weights (`dose_quadrature`), never at the bin centre.
- **Photon reference and cell geometry:** `PhotonParameters`, the shouldered-exponential reference with transition dose `D_t` (Scholz 1997; Elsasser & Scholz 2007 eq. 7), with `dose_for_survival` / `dose_for_log_kill`; `CellTarget`; the oxygen axis (`core/oxygen.py`) and chromatin-based subnuclear-domain sizing (`core/chromatin.py`).
- **Time axis** (`core/protraction.py`): dose schedules and the Lea-Catcheside `G` factor; `survival`, `log_kill`, `dose_for_survival` and `rbe` take a keyword-only `g` (default `1.0`, an acute dose).
- **Packaging:** extras `tables`, `benchmark`, `fast` (numba), `plot`, `dev` and `all`, plus `mkm` (pyMKM, not in `all`); numpy below 2.3 on Python 3.10-3.13 and >= 2.3 on 3.14; `pyradbio.__version__` read from the installed distribution.

[Unreleased]: https://github.com/e0404/pyRadBio/compare/v0.0.18...HEAD
[0.0.18]: https://github.com/e0404/pyRadBio/compare/v0.0.17...v0.0.18
[0.0.17]: https://github.com/e0404/pyRadBio/compare/v0.0.16...v0.0.17
[0.0.16]: https://github.com/e0404/pyRadBio/compare/v0.0.15...v0.0.16
[0.0.15]: https://github.com/e0404/pyRadBio/compare/v0.0.14...v0.0.15
[0.0.14]: https://github.com/e0404/pyRadBio/compare/v0.0.13...v0.0.14
[0.0.13]: https://github.com/e0404/pyRadBio/compare/v0.0.12...v0.0.13
[0.0.12]: https://github.com/e0404/pyRadBio/compare/v0.0.11...v0.0.12
[0.0.11]: https://github.com/e0404/pyRadBio/compare/v0.0.10...v0.0.11
[0.0.10]: https://github.com/e0404/pyRadBio/compare/v0.0.9...v0.0.10
[0.0.9]: https://github.com/e0404/pyRadBio/compare/v0.0.8...v0.0.9
[0.0.8]: https://github.com/e0404/pyRadBio/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/e0404/pyRadBio/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/e0404/pyRadBio/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/e0404/pyRadBio/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/e0404/pyRadBio/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/e0404/pyRadBio/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/e0404/pyRadBio/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/e0404/pyRadBio/releases/tag/v0.0.1
