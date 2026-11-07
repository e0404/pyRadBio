# Roadmap

What pyRadBio implements, what it does not yet implement, and the open problems
in what it does. Entries are not commitments and are not in priority order
within a section. Details of each model live in its summary in
`docs/references/`; an entry here names the gap and what would close it.

## Model inventory

| Model | Module | Reference |
| --- | --- | --- |
| LEM I | `models/lem/lem1.py` | Scholz & Kraft 1996; Scholz et al. 1997 |
| LEM II | `models/lem/lem2.py` | Elsässer & Scholz 2007, doi:10.1667/RR0467.1 |
| LEM III | `models/lem/lem3.py` | Elsässer et al. 2008, doi:10.1016/j.ijrobp.2008.02.037 |
| LEM IV (+ GLOBLE) | `models/lem/lem4.py` | Elsässer et al. 2010, doi:10.1016/j.ijrobp.2010.05.014; Friedrich et al. 2012 |
| rapidLEM | `models/lem/lem1.py` variants | Krämer & Scholz 2006, doi:10.1088/0031-9155/51/8/001 |
| MKM (Hawkins/Kase) | `models/mkm/` (native backend; pyMKM optional) | Hawkins 1996; Kase et al. 2008, doi:10.1088/0031-9155/53/1/003 |
| modified MKM | `models/mkm/spectrum_models.py::ModifiedMKM` | Kase et al. 2006, doi:10.1667/RR0536.1; Inaniwa et al. 2010 |
| MCF MKM | `models/mkm/spectrum_models.py::MCFMKMModel` | Parisi et al. 2022, doi:10.1088/1361-6560/ac7375 |
| SMK | `models/mkm/spectrum_models.py::StochasticMKM` | Inaniwa & Kanematsu 2018, doi:10.1088/1361-6560/aabede; Sato & Furusawa 2012, doi:10.1667/RR2842.1 |
| OSMK 2021 / 2023 | `models/mkm/osmk.py` | Inaniwa et al. 2021, doi:10.1088/1361-6560/ac29cc; Inaniwa & Kanematsu 2023, doi:10.1093/jrr/rrad049 |
| IMK | `models/mkm/imk.py::IMKModel` | Matsuya et al. 2025, doi:10.1002/mp.70040 |
| Phenomenological LET-based RBE (13 models) | `models/phenomenological/` | see `Phenomenological_RBE_models.md` |
| RMF (+ MCDS DSB yields) | `models/rmf/`, `micro/mcds.py` | Carlson et al. 2008, doi:10.1667/RR1046.1; Frese et al. 2012, doi:10.1016/j.ijrobp.2011.06.1983; Semenenko & Stewart 2006, doi:10.1088/0031-9155/51/7/004 |
| Katz cellular track structure (grain-count regime) | `models/katz/` | Katz et al. 1971, doi:10.2307/3573247 |
| NanOx (assembly + calibration layer) | `models/nanox/` | Cunha et al. 2017, doi:10.1088/1361-6560/aa54c9; Monini et al. 2020, doi:10.1667/RR15463.1; Alcocer-Ávila et al. 2022/2023 |
| Lineal-energy engines | `micro/` | Sato et al. 2006/2012; Xapsos 1994 |

Cross-cutting layers: LQ core, mixed-field mixing on a sub-grid, the oxygen
(`core/oxygen.py`) and time (`core/protraction.py`) axes, radial dose and
stopping power (`track/`), the benchmark harness, table I/O and the CLI.

## Models not yet implemented

- **LPL** (Curtis 1986, doi:10.2307/3576798) and **RMR** (Tobias 1985,
  doi:10.2307/3583515). RMF reduces to each in limits stated by Carlson et al.
  2008 (RMR: `λ_F = 0`, `f_R = 1`, no intratrack binary misrepair; LPL: no
  intratrack binary misrepair, `φ = 0`, `λ_F = 0`, `γ = 1`). About 50 lines each,
  and they would turn those limits into tests of `RMFModel`. Curtis 1986 has an
  erratum (Radiat. Res. 119:584, 1989, doi:10.2307/3577534) that must be read
  before any LPL coefficient ships.
- **UNIVERSE / DSB-repair-network models** (Liew, Friedrich, Mairani): the
  continuation of the GLOBLE work.
- **BIANCA** (Ballarini & Carante): chromosome aberrations as well as survival.
- **Multiscale approach (MSA)** (Surdutovich & Solov'yov 2014,
  doi:10.1140/epjd/e2014-50264-6): analytic, a distinct lineage.
- **FLASH / ultra-high dose rate** oxygen-depletion and ROS models (Labarbe et
  al. 2020; Petersson et al. 2020); builds on the oxygen and time axes.
- **Spectrum-weighted phenomenological models**, Belli et al. 1997 and Rørvik's
  weighted model: they integrate over an energy or LET spectrum, so they need a
  spectrum input rather than a scalar `LET_d`.
- Out of scope: full Monte-Carlo track structure (PARTRAC class).

## Gaps in implemented models

- **NanOx forward ion route.** `NanOxModel.alpha_beta()` raises
  `NanOxNanodosimetryRequired`: it needs the single-event specific-energy
  spectrum `dP/dz` in 20 x 10 nm targets and the `RCE`, both from the Lyon
  group's non-public chain and figure-only in the papers. Routes: a
  track-structure code (e.g. Geant4-DNA) scoring the targets, or the authors.
  Once open, the reconstruction target is Monini et al. 2019 Fig. 1
  (doi:10.1016/j.phro.2019.10.004). Do not substitute an amorphous-track
  spectrum. See `NanOx.md`.
- **NanOx low-energy extension** (Alcocer-Ávila et al. 2024, Med. Phys.,
  doi:10.1002/mp.17407; BNCT / targeted radionuclide therapy): not implemented.
- **MCDS**: only the closed-form DSB-yield fit ships, not the MCDS Monte-Carlo
  algorithm.
- **NanOx LCBF**, Monini et al. 2020's second effective-local-lethal-function
  representation (eqs. 21-22, Table 1): ~40 lines; low priority, since later
  papers use the error function.
- **Katz track-width regime.** Needs the site-averaged dose of Butts & Katz
  1967 (Radiat. Res. 30:855), or a digitization of Katz 1971 Fig. 3. Until then
  points above roughly `z*² = 90` are under-predicted. See `Katz1971.md`.
- **IMK**: the PLL-to-LL kinetic equations and the non-targeted (bystander)
  channel; Matsuya 2025 Fig. 5 (needs Si/Ar/Fe beyond the bundled Sato AMF
  coefficients) and Fig. 8 (a separate parameter set).
- **MCF MKM site geometry.** mMKM and SMK score on the MKM disk/cylinder;
  `MCFMKMModel` stays on the spherical `d(y)` route because its `c(y)` couples
  domain and nucleus per event. Moving it needs the Parisi 2022 text and a rule
  for pairing the two sites on a common impact-parameter grid.
- **Mixed fields for the spectrum MKM models**: `_alpha_beta_field` raises
  `NotImplementedError`; SOBP support (Hartzell 2025 Fig. 9 / eq. 5) also needs
  the SOBP `d(y)`.
- **Sato AMF coverage**: coefficients exist only for Z <= 10 and the Fig.-4 site
  size; add other ions and explicit site-size dependence (Hartzell 2025 Fig. 7),
  and guard the AMF interpolation artifact for ions heavier than Ne below
  10 MeV/u. An optional reader for a user-supplied PHITS `tsed.dat` (git-ignored,
  pointed to by an environment variable) would bypass the fit.
- **LEM V**: a spectrum-dependent DSB yield (Pfuhl 2021, ch. 5) as the physics
  below 0.5 MeV/u, where no LEM IV estimator reproduces the TRiP98 tables.
- **Full dose response as output**: expose the LEM IV intertrack Monte Carlo's
  `S(D)`, not only the fitted `alpha`/`beta`.
- **pyMKM cross-check of the SMK/OSMK stochastic entry points**, once pyMKM
  exposes them.

## Physics and numerics

- **Stopping power, shell and Barkas corrections** are reconstructed, not ICRU
  49's exact Model 1: the two oxygen shell thetas and the L-shell scaling are
  calibrated; Barkas uses `γ = 1` rather than ICRU 49's 1.29 (Ashley et al.
  1974's table may absorb the factor); the Barkas term below `x = 1` needs
  Ashley-Ritchie-Brandt 1972's inner-shell term, so it raises below
  ~0.2 MeV/u. See `StoppingPower.md`.
- **Lindhard & Sørensen 1996 low-Z limit**: their "v -> c" values for Z = 10,
  18, 36 are not reproduced (possibly finite nuclear size, not implemented).
  Below 0.3% of the stopping power for Z <= 18 at therapy energies.
- **Default stopping power for Z > 18 below 2.5 MeV/u** is unvalidated (no
  water reference held); above a few hundred MeV/u the Hubert alpha scaling
  carries the alpha's Mott and Lindhard-Sørensen terms, not the heavy ion's.
- **Molière's full quantum nuclear-stopping cross section** for protons at
  0.8-3 MeV would remove a +5% Born residual (at most 4e-5 of the total).
- **Amorphous-track engine at MKM domain sizes**: `ȳ_D ≈ 0` for
  `r_d ≈ 0.26 µm` at most energies, which makes it unusable for the MKM integral.
- **Amorphous high-energy `y_D` undershoot**: 10-25% low for low-LET ions above
  ~50 MeV/u; a distribution-shape problem that needs a track-structure
  delta-ray layer.
- **LEM IV**, each detailed in `LEM_IV_OPEN_CHOICES.md`:
  - fix the giant-loop lattice registration in the mean field (marginalise the
    lattice phase) and retire the legacy 0.95 cluster-index cap with it; this
    moves every LEM IV golden (entries 3.1, 1.1);
  - re-derive the mean-field cluster index together with the biases that
    compensate it: it over-clusters (0.99 / 0.31 against Pfuhl's 0.80 / 0.19)
    (entry 2.3), and moving DSB placement to plain dose (Variant B) belongs to the
    same job (entry 2.4);
  - replace `box_averaged_profile` with a per-loop 2-D integral or an
    orientation-aware average; it smears tracks about 2.5x too wide (entry 3.2);
  - the Friedrich 2012 Fig. 4 `beta_I` roll-off above LET ~70 is not reachable
    from photon data; it needs an ion-data target or an assumed clustered-loop
    lethality (entry 2.7, section 4);
  - below ~2 MeV/u the TRiP98 tables are not reproduced (one 540 nm column
    carries the damage; section 4);
  - beyond ~100 keV/µm deep overkill makes the LQ fit of `beta` ill-conditioned
    and needs a saturation-aware estimator;
  - Elsässer 2010 Fig. 4a protons above 25 keV/µm diverge (deep overkill); the
    golden clips there;
  - the MC impact cut and the numerical resolutions are chosen, not derived
    (entries 1.3, 3.3).
- **LEM I**, each detailed in `LEM_I_ScholzKraft.md`:
  - the Scholz 1997 two-track `beta` goes negative in deep saturation (U at
    12-20 MeV/u, Ar at 8-12) and is clamped to 0 with a warning; the same cause
    leaves carbon `beta` 6-11% low at 18-40 MeV/u. Needs a resummation or a
    saturated/unsaturated split (above `D_t` the series sums in closed form);
  - Scholz & Kraft 1996 Fig. 3 uranium `sigma_1%` is +19% / +37% at 16 / 20
    MeV/u; untried lead: the `beta_x D_t^2` sensitivity;
  - Scholz 1997 Fig. 1 carbon `alpha` (exact branch) is 10-12% low above
    50 MeV/u; next suspects: the profile shape or the cross-section construction;
  - Scholz & Kraft 1996 Fig. 4 proton `RBE_80` sits at 0.919 of the paper, not a
    sampling artifact.
- **LEM II**, detailed in `LEM_II_Elsasser2007.md`: the CHO LEM I residual
  (-14.4%) closes at no nuclear radius (check CHO `D_t = 40` Gy and
  `alpha_x = 0.228`); `validation/elsasser2007_lem2.py` still defaults to
  the eq. (8) initial slope although the paper's method is the cross-section
  integral.
- **LEM I-III residual at low LET** against the papers' model lines: express the
  Elsässer & Scholz 2007 Fig. 8/11 residuals resolved by LET rather than as one
  bias, and check whether the low-LET end is separate from the per-cell offset.
  Score against the papers' model lines, not the data.
- **OSMK V79 mid-LET OER excess** (up to 18% high at 60-150 keV/µm against
  Inaniwa 2021 Fig. 6). Localised to the ratio `z_d,D / z_R`; the evidence points
  at the published V79 `z_R`, unproven. See `OSMK_Inaniwa2021_2023.md`.
- **LEM III KS-1 outlier**: the highest-`alpha_x` line overkills at 77 keV/µm
  independent of radius. See `LEM_III_Elsasser2008.md`.

## Validation and reconstruction

- **One stopping power per model run.** Several MKM-family validation scripts
  and goldens invert LET on one stopping power and run the model on another;
  audit the `@slow` tests for the same defect (e.g.
  `tests/models/lem/test_lem4_localdose.py`).
- **Regenerate the LEM I Monte-Carlo checkpoints** in `validation/data/`: their
  data rows predate the checkpoint fingerprint, and `sk1997_z_slope.csv` does
  not record its stopping power (rerun `scripts/lem1_mc_reference.py z-slope` on
  `ns1970` with a `stopping_power` column).
- **Scholz & Kraft 1996 Fig. 4 at 16 and 20 MeV/u** lies beyond the NS1970
  grid (12 MeV/u); those points need another stopping-power source.
- **Survival-code LEM II/III cross-check**: the paired Monte-Carlo points come
  from the unpatched Survival binary; regenerate them on a patched build. The
  MC `sigma_beta` is not monotone in LET (peak near 292 keV/µm), mechanism
  untested. See `survival_repo_notes.md`.
- **Cross-family comparison against measured data**, to be written up with the
  standing caveats: the phenomenological fits are partly circular against the
  measured corpus, the families do not share an x-axis definition, and a
  residual against data must not be used to choose between implementations.
- **MCF MKM / mMKM golden against Hartzell 2025 Table 1** (six SOBP depths plus
  the monoenergetic values): the table still needs transcribing; only trend and
  magnitude tests exist. Also score model spectra at `r_d = 0.28 µm`, not only at
  the 0.3 µm digitization radius.
- **Cross-check the Sato AMF engine against the amorphous route**, two
  independent native `d(y)` engines.
- **LEM IV against the TRiP98 tables**: below 0.5 MeV/u every estimator misses
  for Z >= 3 and protons run high; the spinal-cord and chordoma files differ by
  an unexplained constant offset. The `discrete_dsb` photon limit
  is 5.5% low for an unknown reason.
- **Tilly 2005 slopes** (`q = 0.309`, `0.550964`) are known only through Rørvik
  et al. 2018's appendix; digitise Tilly's Fig. 2 to check them.
- **Condense `validation/elsasser2007_lem2.py`** (1314 lines) and
  `elsasser2008_lem3.py`, checking the figures pixel by pixel.
- **PIDE loader**: consume more workbook columns (author/year, oxygenation, dose
  rate, RBE at fixed survival).

## Literature to obtain or read

- Kase et al. 2008 (doi:10.1088/0031-9155/53/1/003): write its summary.
- Krämer & Scholz 2000 (doi:10.1088/0031-9155/45/11/313): biological dose
  optimisation, context for table use.
- Butts & Katz 1967 (Radiat. Res. 30:855): Katz track-width regime.
- Cox et al. 1977 (doi:10.1080/09553007714550661): the V79 `alpha_r` paired with
  NanOx's `beta_r = 0.0259`.
- Curtis 1986 erratum (doi:10.2307/3577534): before LPL.
- Ashley et al. 1974 and Bichsel 1967 (UCRL-17538): the remaining ICRU 49
  Model 1 sources.
- Parisi et al. 2022 full text: settles the MCF MKM site geometry.
- Sato et al. 2023 (doi:10.1088/1361-6560/ace14c), the newer AMF: settles which
  AMF version Parisi and Hartzell use.
- TRiP98's own LEM IV protocol (dose window, realisation count, whether it fits
  one LQ at all) and Friedrich 2012's Fig. 4 fit window, both unstated; and
  whether `l_DSB = 540 nm` and `r_c = 6.5 nm` hold at extreme LET.
- A second published LEM IV reference `beta_I` for another cell line: would
  settle the fit-window question and whether Friedrich 2012's `S>0.01` legend is
  a fit protocol (`LEM_IV_OPEN_CHOICES.md` entry 1.2). The papers are silent on
  the survival convention (`<e^-E>` vs `e^-<E>`, entry 2.1).
- Pfuhl et al. 2022 Table 2: record its page number in
  `validation/pide_pfuhl2022.py`.
