# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2026-11-07

### Added
- **LQ core** (`core/lq.py`): `LQParameters`, `survival`, `log_kill`, `dose_for_survival` (stable quadratic inverse with a continuous `beta -> 0` limit), `rbe`, `bed` / `eqd2`, and the LQ fits `fit_lq_linearised()` / `fit_lq_weighted()`.
- **Radiation fields and mixing** (`core/radiation.py`, `core/mixing.py`): `Ion`, `SpectrumComponent` (mono-energetic or binned), `RadiationField`, fluence/dose conversion; mixed fields dose-average alpha and sqrt(beta) (Zaider-Rossi), and binned components are integrated on a log sub-grid with phi(E)S(E) weights (`dose_quadrature`), never at the bin centre.
- **Photon reference and cell geometry:** `PhotonParameters`, the shouldered-exponential reference with transition dose `D_t` (Scholz 1997; Elsasser & Scholz 2007 eq. 7), with `dose_for_survival` / `dose_for_log_kill`; `CellTarget`; the oxygen axis (`core/oxygen.py`) and chromatin-based subnuclear-domain sizing (`core/chromatin.py`).
- **Time axis** (`core/protraction.py`): dose schedules and the Lea-Catcheside `G` factor; `survival`, `log_kill`, `dose_for_survival` and `rbe` take a keyword-only `g` (default `1.0`, an acute dose).
- **Packaging:** extras `tables`, `benchmark`, `fast` (numba), `plot`, `dev` and `all`, plus `mkm` (pyMKM, not in `all`); numpy below 2.3 on Python 3.10-3.13 and >= 2.3 on 3.14; `pyradbio.__version__` read from the installed distribution.

[Unreleased]: https://github.com/e0404/pyRadBio/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/e0404/pyRadBio/releases/tag/v0.0.1
