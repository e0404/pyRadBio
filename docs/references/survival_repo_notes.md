# batuff/Survival: LEM I-III implementation and cross-checks

INFN Torino / UniTO "Survival" toolkit. Reference paper: Manganaro et al., *"Survival": a
simulation toolkit introducing a modular approach for radiobiological evaluations in ion beam
therapy*, Phys. Med. Biol. **63**(8) 08NT01 (2018), doi:10.1088/1361-6560/aab697. Code: GPL-2,
https://github.com/batuff/Survival. The only other open-source LEM I-III implementation known
to us: a second opinion on whether a disagreement with a paper is ours.

> **Standing caveat (AGENTS.md rule 3).** Architecture inspiration and a check on the
> **approximation formulas** only: do not copy its code or adopt its numbers as references
> (section 6: three defects). Its data (SRIM tables, eta table, JSON exports) is GPL-2 output of
> a local build and is never redistributed.

All numbers were measured by compiling it (MinGW g++ 13.1, GSL 2.7.1) and running grids
(section 8).

## 1. What Survival implements

| Model | Track class | Photon response | `r_min` | Radical diffusion |
|---|---|---|---|---|
| `LEMI` | `Track_Scholz2000` | `LQ` | 10 nm | none |
| `LEMII` | `Track_Elsasser2007` | `LQ2` (eta) | 0.3 nm | sigma = 4 nm |
| `LEMIII` | `Track_Elsasser2008` | `LQ3` (eta) | 0.04 um x beta | sigma = 4 nm |

Constants equal pyRadBio's. Profile `D(r) = lambda LET/r^2`,
`lambda = CONV/(2 pi rho (0.5 + ln(r_max/r_min)))`, `r_max = 0.062 E^1.7` um. LEM IV is not
implemented.

## 2. The two approximate approaches (`src/Calculus.cpp`)

Both put one track at the nucleus centre, get `S_1`, convert to alpha_z/alpha_D, slope-match
beta_z, rescale to beta_D, and mix with LET weights (`alpha` linear, `sqrt(beta)` linear:
Zaider-Rossi). `d_LET = CONV LET/(rho pi R_N^2)`; `d_in` = mean dose the centred track
deposits inside the nucleus.

* **`rapidLEM_Scholz2006`** (GSI; Kramer & Scholz 2006; `Calculus.cpp:988`):
  `alpha_z = -ln S_1/d_LET`, `alpha_D = (1 - S_1)/d_LET`,
  `beta_z = max((s_max - alpha_z)/(2 D_t), 0)`, `beta_D = (alpha_D/alpha_z)^2 beta_z`.
* **`rapidLEM_Russo2011`** ("INFN approach"; Russo, PhD thesis, Torino 2011; `:921`): the same,
  but escaping energy is credited with the photon alpha:
  `alpha_z = alpha_X (1 - d_in/d_LET) + (-ln S_1)/d_LET`,
  `alpha_D = alpha_X (1 - d_in/d_LET) + (1 - S_1)/d_LET`. This is the `alpha_x (1-f)` halo term
  of pyRadBio's LEM I photon-floor fix, reached independently.

**Naming trap**: `rapidINFN_alphaIon_betaIon` (the paper's UML) is a stub that calls `exit(1)`
(`Calculus.cpp:479`); the runnable INFN approach is `rapidLEM_Russo2011`.

The MC path (`calculusType MonteCarlo`) samples tracks, builds survival at each imposed dose
(default 1-6 Gy) and LQ-fits (`usefulFunctions.cpp:fit_LQ`: weighted `(D, D^2)` fit, no
intercept, weights `1/(sigma_S/S)^2`). Its beta is a fit-window quantity; quote the window.

## 3. Stopping power

Survival converts E to LET with SRIM tables `data/<Ion>_Srim.dat` (H..Ne, 161 rows,
8.3e-4 to 8.3e2 MeV/u): **electronic stopping only** (`usefulFunctions.cpp:111`) and
**linear-in-energy interpolation**. `pyradbio.track.srim` reads a user's local copy (SRIM
data is never bundled) with both as defaults: max |rel err| 2.7e-6 over 4 ions x 16 energies;
turning either off shifts LET by 0.1% (nuclear) and 0.24% (log-log).

```python
from pyradbio.track.srim import srim_stopping_power

sp = srim_stopping_power(("H", "He", "C", "O"))  # PYRADBIO_SURVIVAL_SRC_DIR
```

## 4. Measured agreement

HSG (alpha_X = 0.313, beta_X = 0.0615, R_N = 5 um), D_t = 30 / 6 / 19 Gy (LEM I / II / III;
the paper's table 1); H, He, C, O; 16 energies 1-400 MeV/u; on Survival's LET. Metric: median
|relative error| per ion. **F** = pyRadBio's `N_1`, `D_1` through Survival's exact rapid
formula; **S** = pyRadBio as shipped.

### 4.1 alpha

| model | vs | H (F/S) | He (F/S) | C (F/S) | O (F/S) |
|---|---|---|---|---|---|
| LEM I | Scholz2006 | 4.2 / 9.4 % | 3.8 / 7.3 % | 2.8 / 6.1 % | 2.6 / 6.2 % |
| LEM I | Russo2011 | 2.8 / 2.8 % | 2.7 / 2.7 % | 2.1 / 2.5 % | 1.7 / 2.5 % |
| LEM II | Scholz2006 | 6.3 / 11.5 % | 4.1 / 8.1 % | 1.4 / 1.8 % | 1.6 / 1.5 % |
| LEM II | Russo2011 | 5.4 / 5.3 % | 3.9 / 3.8 % | 1.4 / 1.2 % | 1.6 / 1.4 % |
| LEM III | Scholz2006 | 6.7 / 26.7 % | 7.0 / 19.3 % | 7.2 / 11.5 % | 5.3 / 9.8 % |
| LEM III | Russo2011 | 6.6 / 6.5 % | 6.9 / 6.9 % | 6.9 / 7.3 % | 5.1 / 5.8 % |

Our exact impact-parameter alpha reproduces the INFN correction, not the GSI one. Pinned by
`test_golden_lem1_matches_survival_rapidlem_hsg` (LEM I alpha median |rel err| 10-400 MeV/u:
1H 6.0%, 4He 3.3%, 12C 2.6%; bound < 12%).

### 4.2 The residual LEM I 2.6% is `r_max`

`r_max` ours/Survival = 0.8065 = 0.05/0.062 at every energy (3 / 30 / 300 MeV/u: 0.324 /
16.22 / 813.0 um vs 0.401 / 20.11 / 1008.1 um). We use gamma = 0.05 for LEM I (Scholz & Kraft
1996 eq. 10); Survival's `Track_Scholz2000` hard-codes `GAMMA = 0.062` (Elsasser & Scholz 2007)
for all versions. LEM I alpha median |ratio-1| with `r_max_gamma_um = 0.062` and nothing else
changed: vs Russo2011 2.57% to **0.42%**; vs Scholz2006 7.04% to 4.44%.

**Do not change our gamma**; each paper's value is applied where it belongs
(`LEM1_R_MAX_GAMMA_UM` vs `R_MAX_GAMMA_UM`, `track/scholz_kraft.py`). The comparison fixes it
explicitly (`validation/survival_code_lem.py` `MATCH_SURVIVAL_R_MAX`, default on;
`benchmark/manganaro2018.py` `match_r_max`). With gamma matched, LEM I alpha against the
reference MC is -0.5% to +1.4% (8000-cell carbon, LET 53-747) and within 0.8% for protons
(LET 0.7-12), vs +1.1 to +7.6% unmatched.

### 4.3 beta: a default, not a defect

LEM I beta median |rel err| vs Survival rapid, H / He / C / O: `slope_matching` (default)
2.7 / 3.8 / **16.7** / **41.2** %; `rapid_lem` 1.0 / 1.5 / 2.2 / 2.9 %. Survival slope-matches
the unsaturated alpha_z and rescales by `(alpha_D/alpha_z)^2`; our `slope_matching` applies
`(s_max - alpha_I)/(2 D_t)` to the saturated alpha_I. `rapid_lem` (`beta_D = f^2 beta_z`,
`f = (1-e^{-N_1})/N_1`) is Survival's formula; `scholz1997_two_track` (exact pair-overlap
integral, Scholz et al. 1997) is the LEM-native beta. 12C at 1 / 3 / 10 MeV/u (LET 747 / 414 /
174): slope_matching 0.0557 / 0.0474 / 0.0402, rapid_lem 0.0000 / 0.0004 / 0.0131, two_track
0.0000 / 0.0000 / 0.0034, Survival 0.0000 / 0.0007 / 0.0142.

`slope_matching` has a spurious deep-overkill beta revival (to ~0.9 beta_x at LET 747): exact
alpha_I is not monotone in LET, and slope matching reads its thinning as room under `s_max`
(C and O, all three versions). All three beta models exist for LEM I-III (on `LEMBase`; they
need a pointwise local-dose response). `rapid_lem` brings the LEM II beta error against
Survival's rapid output from 17-23% to 4.4-6.6%: agreement with their approximation, not
accuracy (5.2). LEM II/III once overrode only the scalar `_photon_log_kill`, not its array
twin, so an array-based beta gave an un-enhanced LEM I response; both are now overridden
together and pinned equal.

**LEM IV is excluded by construction** (a nonlinear function of a global DSB-pattern statistic):
`LEM4.scholz1997_two_track_beta` raises and the constructor rejects that `beta_model`;
`rapid_lem` is routed through `LEM4.single_traversal_log_kill`; for a general LEM IV beta use
`beta_model="intertrack_mc"`.

### 4.4 The LEM III residual is Survival's `logS_t3` bug, not eta

Driving our LEM II/III with Survival's eta table changes LEM III agreement by 0.02 points
(6.51% to 6.49%, H): eta is not the lever. LET-binned ratio pyRadBio/Survival, LEM III vs
Russo2011, was a bump: 1.000 (0.2-1 keV/um), 1.069, **1.133** (3-10), **1.124** (10-30), 1.086,
1.028, 1.015 (300-1200). **Ruled out**: the radial-dose convolution grid (tightening
`doseCutoff` 1e-8 to 1e-10, `lengthDoseCurve` 300 to 8000, step x10 finer, +-20 to +-40 sigma
moves carbon alpha at LET 26 by +0.011% against a +12.4% discrepancy).

**Cause**: patching `CellLine.cpp:131,157` (drop the stray `exp`, section 6a) moves their LEM
III alpha onto ours: at LET 12.4 / 25.9 / 67.3 / 174.5, shipped 0.4322 / 0.6548 / 1.0402 /
1.5506, patched 0.4994 / 0.7304 / 1.1176 / 1.5990, ours 0.5000 / 0.7324 / 1.1226 / 1.5863.
Their MC carries the same defect (`getLogSurvival_X` dispatches to `parametrization_LQ3`):
LEM III MC at LET 67.3 is 1.0369 +- 0.0130 shipped vs 1.1381 +- 0.0135 patched. Full sweep, median |alpha ratio-1| vs Russo2011 (H/He/C/O), LEM II 5.28/3.78/1.23/1.40% shipped to 0.41/1.18/0.84/0.79% patched; LEM III 6.51/6.88/7.26/5.81% to 0.10/0.30/0.42/0.57%; the patched binned LEM III ratio is flat (1.0005 to 1.0132). `validation/survival_code_lem.py` draws both builds (shipped dashed, patched solid with `x`).
Lesson: their MC and rapid paths share the defective photon response, so their agreeing with
each other is no evidence about us.

## 5. The reference Monte Carlo

Grid: HSG, carbon 3-300 MeV/u for all three versions plus protons 3-100 MeV/u for LEM I, doses
1-6 Gy, `-precision 500`; plus a LEM I carbon pass at 1-40 MeV/u, `-precision 8000`. Cost is driven by track extent, not LET (carbon 300 MeV/u: 33 s per cell vs 0.07-0.11 s at 1-40 MeV/u); at 8000 cells the carbon LET 53-747 range costs a flat 565-655 s per point, so large high-LET error bars are a choice of N.

### 5.1 alpha: the INFN correction against the reference

Survival's approximation vs its own MC at LET 12.4 / 25.9 / 67.3 / 174.5 / 414:

| | 12.4 | 25.9 | 67.3 | 174.5 | 414 |
|---|---|---|---|---|---|
| LEM I Scholz2006 | -18.5% | -9.2% | -3.6% | +2.0% | +4.8% |
| LEM I Russo2011 | -1.3% | +1.2% | +0.4% | +2.0% | +4.8% |
| LEM III Scholz2006 | -34.8% | -17.7% | -4.9% | -1.2% | +14.0% |
| LEM III Russo2011 | -0.1% | -0.7% | +0.3% | -1.0% | +14.1% |

This reproduces the paper's central claim (its section 4). For LEM II/III both paths share the
defective `logS_t2/3`, so internal consistency is not validation: **only LEM I's MC is a clean
external reference.** Ours vs that MC: LEM I +0.5 to +7.6% (carbon), +1.1 to +4.6% (protons),
the gamma offset of 4.2.

### 5.2 beta: the MC settles the model choice, and it is `two_track`

MC beta is a measurement: its error grows with LET as the 1-6 Gy curve straightens, and a
near-zero true beta scatters symmetrically through zero (paper fig. 5(b) shows the same).
Negative fits are expected; discarding them biases the mean up. LEM I carbon at 8000 cells,
deviation in sigma:

| LET | MC beta | slope_matching | rapid_lem | two_track |
|---|---|---|---|---|
| 746.6 | +0.0020 +- 0.0027 | +20.0 | -0.7 | -0.7 |
| 413.9 | +0.0010 +- 0.0062 | +7.5 | -0.1 | -0.2 |
| 174.5 | +0.0061 +- 0.0033 | +10.2 | +2.1 | -0.8 |
| 93.8 | +0.0173 +- 0.0017 | +14.2 | +5.7 | -1.9 |
| 53.5 | +0.0294 +- 0.0009 | +17.7 | +10.2 | -3.4 |

Worst over all 16 points (LET 0.7-747): `scholz1997_two_track` 3.4 sigma, `rapid_lem` 10.2,
`slope_matching` 20.0. Against our own MC (13 points H/He/C/O, LET 12-1105): two_track max
2.5 sigma (median 0.8), rapid_lem 44.5 (1.3), slope_matching 94.9 (14.5); our analytic alpha
vs our MC median 0.45%, max 1.27%. At LET 12.4 (LEM II) the reference beta is 0.0186 vs
Survival rapid 0.0437 and ours 0.0420. Photon limit: LEM I protons at LET 0.7 give MC beta
0.06126 +- 0.00013 vs beta_X 0.0615. Beta is not a function of LET (protons at 12.1: 0.0148;
carbon at 12.4: 0.0474; two_track 0.0178 / 0.0489). A 500-cell beta is not converged (LET 414:
0.0295 +- 0.0199 vs 0.0010 +- 0.0062 at 8000 cells).

### 5.3 MC vs MC (`scripts/lem_mc_vs_survival.py`, cache `validation/data/survival_our_mc.json`)

Controlled for gamma = 0.062, HSG parameters, the 1-6 Gy window and the same fit. LEM I carbon,
8 points (LET 53-747): **every point agrees within 1.3 sigma on alpha (typically ~0.4%, max
1.2%) and 0.5 sigma on beta**, error bars within ~15% at equal N. Ours is 50-600x faster; the
numba engine is not bit-reproducible, so pin `engine="numpy"` for exact numbers. The MC serves
LEM I-III through `LEMBase._kernel_eta_table` (empty = eta 1 = LEM I); `_check_mc_response`
raises if the table disagrees with `_photon_log_kill_array`, so a missing table cannot return a
LEM I answer under another name. LEM IV's MC is separate code and not covered (all three MC
entry points raise).

High energies (`OUR_MC_HIGH_E`, carbon 300 / proton 100 MeV/u) need `b_cut_um` = 10 um with
threshold `n_bar >= 2000` (as in `scripts/lem1_mc_reference.py alpha-beta`); ceiling
~511 MeV/u (`_KELLERER_B_MAX_NODES = 20000`). 18 paired points, alpha median / significance:
LEM I 1.01% / 1.3 sigma (max 2.8); LEM II 2.19% / 1.2 (max 5.2); LEM III 11.61% / 6.6 (max
33.5). LEM III carbon vs their MC at 30 / 100 / 300 MeV/u: +8.73 / +11.61 / +16.35%, vs their
rapid unpatched +8.38 / +12.43 / +16.51%, vs rapid patched +0.88 / +0.80 / +0.81%: their MC
carries the 6a defect independently. Carbon 3 MeV/u points are noise (their alpha bar is 6-7%).

Budgeting is per point from a pilot (`N` proportional to `(sigma_measured/sigma_target)^2`,
relative targets on alpha, absolute on beta, 8 repeats). The LEM II/III MC penalty (~20x LEM I
at carbon 300) came from `_analytic_profile` rejecting their radical-diffusion profile, not
from the eta lookup; a closed form beyond ~175 nm (tolerance 1e-3; energy independent) removed
it (5-8x faster). A far-field cut scan (10-40 um) shows no monotone dependence (alpha spread
0.19-0.63%, inside statistics), so 10 um stands for all versions. The Jensen de-bias of
`-ln(mean S)` is <= 0.78% at N = 500, <= 0.06% at 8000.

### 5.4 The paired MC run, 57 of 60 points (`scripts/survival_paired_mc.py`)

Drives a local `survival.exe` over (version x H/He/C/O x 3/10/30/100/300 MeV/u); resumable, locked, JSON re-exported per point. Survival's cost (setup + per-cell) is not predictable across versions (LEM I proton 300: ~0 + 7598 s/cell; LEM II helium 300: 49 s + 1841 s/cell), so the driver probes with 2 and 20 cells. At high energy precision does not follow 1/sqrt(N) (LEM I proton 300 reaches 0.22% with 2 cells); 3-10 MeV/u needs 15 000-35 000 cells for ~1%. Median sigma_alpha 0.55% (best 0.22%); LEM III helium and LEM II/III protons at 300 MeV/u are unobtainable (one cell exceeds four hours).

| MC vs MC, 57 pairs | n | median \|alpha ratio-1\| | median \|z\| |
|---|---:|---:|---:|
| LEM I | 20 | 0.66% | 1.01 |
| LEM II | 19 | 2.73% | 3.53 |
| LEM III | 18 | 8.63% | 11.50 |

The LEM II/III offsets are the 6a defect through a second path (unpatched binary; the patched
analytic builds close 2.6% and 6.6% to 0.7% and 0.4%); not a pyRadBio disagreement.

## 6. Defects found in Survival (do not adopt its numbers)

**(a) `logS_t2`/`logS_t3` are exponentiated (LEM II and III only).** `CellLine.cpp:131,157` set
`logS_t2 = exp(-alpha D_t - beta D_t^2)` where the LEM I path (`:105`) correctly sets the log;
both are used as a log in `return logS_t2 - s2 (eta dose - D_t2)` (`:555`). For HSG at
D_t = 6 Gy the transition log-survival should be -4.09 but is `exp(-4.09) = 0.0167`; at
D_t3 = 19 Gy it is -28.15 vs ~0, a 100% error in the outer track that dominates alpha. Patched,
their LEM III alpha moves +15.6% at LET 12.4, +11.5% at 25.9, +7.4% at 67.3 (onto ours within
0.1-0.8%), LEM II +0.8 to +3.5%.

**(b) `Track_Elsasser2008::getDistance`** takes `log10(localDose)/(lambda let)` where
`log10(localDose/(lambda let))` is meant (`Track_Elsasser2008.cpp:167`); harmless now (only
caller is in the commented-out `rapidINFN` body).

**(c) `betheBloch_inv_Srim`** skips 5 header lines where the file has 4
(`usefulFunctions.cpp:197`), dropping the first data row (LET-to-energy inversion, extreme low
end only).

Also `main.cpp:276` passes `genomeLength = 0` to `addParametrization_LQ2/3` (would divide by
zero in `analyticDamageEnhancement`; harmless because the shipped path reads the eta table).

## 7. The eta table is a digitization, not a computation

`data/eta_Elsasser2007.dat` (200 log-spaced nodes, 1e2 to 5e6 Gy) peaks at eta = 14.807 at
4.571e5 Gy, essentially our digitization of Elsasser & Scholz 2007 fig. 2 (14.79), but cannot
corroborate it. Survival's own closed form (`analyticDamageEnhancement`,
`eta = 1 + e^{-a2 D}(1 - e^{-a1 D/2})^2/(a2 D)`, `a1/a2 = alpha_SSB/alpha_DSB = 41.67`) peaks
at 9.01 and fits the table to 34% RMS; both Survival generators are >= 1 by construction, while
the table has 6 nodes below 1 (min 0.9871 at 124 Gy) and 15 sign changes on its rising limb. So
it is fig. 2 traced and resampled onto the generator's grid (pinned by
`test_survival_eta_table_is_a_traced_figure_not_either_of_its_own_generators`). Independent
closed-form recomputations sit at 13.40 (ours) and 13.51 (Friedrich 2015); the xfail
`test_cluster_model_reproduces_digitized_fig2` is unchanged.

## 8. Reproducing this

Survival needs GSL (the repo bundles only macOS `.dylib`s; `make install` fails on an unquoted
`C:/Program Files` inside libtool, so link the build tree directly).

```bash
./configure --prefix=... --disable-shared --enable-static && make -j8   # GSL 2.7.1
g++ -O3 -fopenmp -std=c++11 -D WIN32 -D AMD64 -I./include -I$GSL \
    src/*.cpp -L$GSL/.libs -L$GSL/cblas/.libs -lgsl -lgslcblas -lm -o survival.exe
DATA=<Survival>/data ./survival.exe -output LQ_pars -model LEMI \
    -calculusType rapidLEM_Scholz2006 -ion C -energies 120 600 1200   # NOTE: total MeV
```

Gotchas: `-energies` is **total MeV** (multiply by A); the CSV `meanLET` column is MeV/um while
the console prints keV/um; `DATA` must be set or the binary `exit(1)`s. JSON exports follow
`<MODEL>_<calculus>_<cell>_<ref>.json` (read by `benchmark.reference_tables.load_survival_table`
from `PYRADBIO_SURVIVAL_DIR`). `PYRADBIO_SURVIVAL_SRC_DIR` points to the checkout
(`pyradbio.datasets.SURVIVAL_SRC`): `pyradbio.track.srim` reads the SRIM tables and
`benchmark.reference_tables.load_survival_eta_table` the eta table.
