# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/e0404/pyRadBio/compare/v0.0.8...HEAD
[0.0.8]: https://github.com/e0404/pyRadBio/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/e0404/pyRadBio/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/e0404/pyRadBio/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/e0404/pyRadBio/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/e0404/pyRadBio/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/e0404/pyRadBio/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/e0404/pyRadBio/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/e0404/pyRadBio/releases/tag/v0.0.1
