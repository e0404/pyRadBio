# Architecture

```
                +-----------------------------+
                |        core (LQ layer)       |
                |  lq, radiation, photon, cell |
                |  mixing (sub-grid spectral)  |
                |  oxygen, protraction (axes)  |
                +--------------+--------------+
                               |
        +----------------------+----------------------+
        |                                             |
   track (radial dose,                        models/base (ABC)
   stopping power: analytic                           |
   Bethe = default)                        +----------+-----------+
        |                                  |                      |
        +-----------> models/lem            models/mkm/MKMModel
                      lem1..lem4                  |  (engine-agnostic)
                               \                  | MKMBackend protocol
                                \                 +----+--------------+
                                 \                     |              |
                                  \        NativeMKMBackend   PyMKMBackend
                                   \          (default)        -> pyMKM (optional
                                                                 cross-check)
                                    \               /
                                     +------+------+
                                +------+-------+
                                       |
                              tables (alpha-beta, z*, io)
                                       |
                              benchmark (datasets: PIDE / RadPhysBio / csv, metrics, report)
                                       |
                                     cli
```

Principles
- The LQ/mixing core is model-agnostic; models depend on it, not vice versa.
- The MKM model is engine-agnostic: `MKMModel` owns the biophysics
  (`alpha = alpha0 + beta*z*_1D`) and delegates the `z*_1D` computation to an
  `MKMBackend`. `NativeMKMBackend` is the default; `PyMKMBackend` wraps pyMKM as
  an optional cross-check (the `mkm` extra). Any other engine is a drop-in
  sibling backend, no model or caller changes required.
- Every model's default LET is `track.stopping_power.default_stopping_power()`:
  ICRU 90's procedure for Z 1-18, Hubert's alpha scaling for Z 19-92 from
  2.5 MeV/u, and analytic Bethe anchored to them elsewhere. The other
  `track.stopping` classes and the tabulated sources (NS1970, SRIM, libdEdx,
  pyMKM) are opt-ins.
- Layering: core, track and micro never import models, benchmark, tables or cli
  (`tests/test_import_layers.py`); pyMKM is imported only by the MKM adapter.
- pyRadBio owns LEM I-IV and the unifying layer.
- Optional dependencies (pymkm, pandas/pyarrow, numba, matplotlib) are extras;
  the core imports cleanly without them.
- Numerics hazards are isolated in core/mixing.py and track/ and pinned by tests.
- One mixed-field implementation: `BiophysicalModel.alpha_beta(ion | field)` is a
  template. A model implements `_alpha_beta_ion(ion)`; a field is expanded by
  `core.mixing.dose_quadrature` into `(ion, dose)` nodes (one per mono-energetic
  component, `sub_nodes` per binned component on a log sub-grid with weights
  phi(E)S(E) from the model's own stopping power), and the per-family mixing
  quantities are dose-averaged over them: `(alpha, sqrt(beta))` by default,
  `z*_1D` for `MKMModel` (`_mixing_quantities` / `_lq_from_mixed`), the LET for
  the phenomenological models. Katz, NanOx and the spectrum MKMs refuse fields
  by overriding `_alpha_beta_field`.

Model interface (`models/base.py`)
- Every family implements the same positional interface on `BiophysicalModel`:
  `alpha_beta(field)`, `survival(field, dose_Gy)` (scalar or array dose),
  `dose_for_survival(field, S)`, `rbe_at_survival(field, S)`,
  `rbe_at_dose(field, dose_Gy)`, `reference()` and `alpha_beta_table(ions, energies)`
  (an `AlphaBetaTable` for every model). `field` is an `Ion` or a `RadiationField`.
- The base class supplies LQ defaults. Non-LQ survival (SMK/OSMK eq. 24, Katz
  eq. 10) overrides `survival`, `dose_for_survival` and `_log_kill`; RBE then
  inverts the model's own curve. RBE is formed on the log-kill, so it stays finite
  where `S` underflows.
- `reference()` returns the photon reference as `PhotonParameters` (`photon` on LEM,
  RMF, the phenomenological models and optionally `MKMModel`; `alpha_ref/beta_ref`
  on the spectrum MKMs; eq. (1) at `reference_z_star_1D_Gy` for IMK) and raises
  `PhotonReferenceError` where none is defined. Katz has a multi-target, non-LQ
  photon curve: `reference()` raises and its RBE overrides
  `_reference_dose_for_log_kill` instead.
- Model-specific axes are keyword-only extensions, never new positional
  arguments: OSMK `pO2_mmHg=`, IMK `schedule=`, Katz `let_keV_um=`.
- `ModelCapabilities` (`ion_route`, `mixed_field`, `lq_exact`, `let_route`,
  `spectrum_route`) is a `ClassVar` on every family. `LETModel` is the base of the
  phenomenological family (`alpha_beta_from_let` and the `*_from_let` survival/RBE
  mirror); the benchmark harness routes on it. Spectrum entry points keep their own
  names (`alpha_beta_from_spectrum`, `rbe_at_survival_from_spectrum`, ...).
- `NanOxModel` is a standalone assembly class (`capabilities.ion_route=False`), not a
  `BiophysicalModel`, until an ion route exists.
- Every model is a frozen dataclass; `__post_init__` resolves defaults once via
  `object.__setattr__`. LEM keeps `radial_dose` and `cluster` as optional constructor
  inputs; `resolved_radial_dose` and `resolved_cluster` expose the active components
  in fields with `init=False`. Thus `dataclasses.replace` rebuilds defaults after a
  stopping-power or photon-parameter change, preserves explicit injections, and
  accepts `None` to reset an injection to the version default.
  Published parameter sets are stored as model instances and
  applied with `dataclasses.replace(preset, **overrides)`. String options are
  `Literal`s, validated at construction from `typing.get_args` (`check_choice`); the
  LEM execution engine is the single `models.lem.engine.Engine`.
- All phenomenological RBE routes use the same `PhotonParameters` curve. Supplying
  `D_t` explicitly extends the reference with an exponential tail; the published
  pure-LQ closed forms and high-dose `RBE_min` limit apply when `D_t is None`.
- Non-LQ RBE uses direct log effects: Katz sums ion-kill and multi-target gamma-kill
  effects; SMK/OSMK subtract `log1p(gamma D)` from the LQ core. Numerical SMK uses
  log-sum-exp over the existing multi-event masses. This prevents survival
  underflow without changing the Taylor validity range or the FFT tail accuracy.
