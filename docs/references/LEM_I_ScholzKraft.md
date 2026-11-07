# LEM I — Scholz & Kraft 1996 / Scholz et al. 1997
doi:10.1016/0273-1177(95)00784-C ; approx. doi:10.1007/s004110050055

Scholz & Kraft 1996 (SK1996) is the model paper; Scholz, Kellerer,
Kraft-Weyrather & Kraft 1997 (SK1997) adds the explicit stochastic calculation
and its fast approximation. Scholz & Kraft 1994 is cited below only for its
Fig. 4 caption and eq. (4).

## One-paragraph idea
The biological effect is determined *locally* by the spatial local-dose
distribution inside the cell nucleus, with the local dose-effect relation taken
from the photon survival curve (radiation-quality independence of local effect).
Integrate the photon lethal-event density over the nucleus to get ion survival.

## Inputs
- Photon reference per cell line: `alpha_x`, `beta_x`, `D_t`; effective nuclear
  area/radius.
- Ion `(Z, A, E)`; LET from the library stopping power; radial dose below.
- SK1996 Figs. 2-4 (V79, `A_nucl = 50 um2`, `D_t = 12` Gy): `FIGURE_INPUTS` in
  `benchmark/scholz_kraft1996.py`. SK1997 Figs. 1-5 (CHO, `alpha_x = 0.18`,
  `beta_x = 0.028`, `D_t = 30` Gy, 10 um nuclear diameter, used as-is):
  `CHO_INPUTS` in `benchmark/scholz1997.py`.

## Core equations
- S_x(D) = exp(-N_x(D)),  N_x(D) = -ln S_x(D)              (SK1996 eqs. 1-2)
- LQ photon input: N(D) = alpha_x D + beta_x D^2 below D_t, exponential tail above
  with s_max = alpha_x + 2 beta_x D_t (SK1997 eq. 1).
- N_ion = ∫_V nu(d(x,y,z)) dV,  nu = N_x(d)/V              (eqs. 4-6)
- S_ion = exp(-N_ion)                                      (eq. 7)
- Radial dose: D(r) = lambda LET/r^2 for r_min <= r <= r_max, constant core for
  r < r_min; r_min = 0.01 um (10 nm); r_max = gamma E^delta, gamma = 0.05,
  delta = 1.7 (SK1996 eqs. 9-10; LEM II/III use gamma = 0.062, see
  `LEM_TrackStructure_Elsasser2008b.md`).
- **Initial slope (pyRadBio default, `initial_slope_model="cross_section"`).**
  `sigma_0 = ∫ (1 - exp(-nu(b))) 2 pi b db`, with `nu(b)` the nuclear-averaged
  log-kill of one track at impact parameter b (eqs. 4-7), and
  `alpha = sigma_0 / (A_eff D_1)`, `D_1 = 0.16 LET / (rho A)`. It includes the
  indirect traversals (track misses, 1/r^2 tail still irradiates; SK1996 Fig. 3
  discussion), so `alpha -> alpha_x` at low LET and `sigma -> A_eff` (overkill) at
  high LET. The central-traversal-only shortcut drops toward zero at low LET.
- **SK1996's `sigma_inact` is a final-slope observable, not `sigma_0`.** SK1996
  (before Fig. 2): `sigma_inact` "were determined from the final slope of
  survival curves"; the SK1994 Fig. 4 caption fixes it at the 1% survival level.
  For a shouldered curve `sigma_final > sigma_0` strictly.
  `LEM1.inactivation_cross_section_um2` is `sigma_0`.
- **SK1997 approximation** (eqs. 9-11): `-ln S = alpha_z z + beta_z z^2` in nucleus
  specific energy z; `alpha_z = v_r / Delta z` from a central traversal;
  `beta_z = (s_max - alpha_z)/(2 D_t)`. Exposed as
  `beta_model="scholz1997_stochastic"` (converted back to a dose-domain beta by a
  second-order Poisson traversal expansion).
- **SK1997 explicit LQ** (Figs. 2, 3, 5): full impact-parameter cross-section as
  `alpha`, spatial two-track overlap as `beta`: `LEM1.scholz1997_explicit_lq`. Do
  not substitute `scholz1997_approximate_log_kills` (the central-traversal /
  specific-energy Poisson approximation) for it.
- Single-hit consistency check between panels: `RBE_80 ~ 5.4 sigma / (A D_1)`.

## Parameters & default values
- Effective nuclear cross-section < geometric (SK1996, 'geometrical structure';
  CHO 75 um2 geometric vs 50 um2 effective). Same construction as Todd 1985's
  fluence-weighted area, see `LEM_II_Elsasser2007.md` ("Todd 1985").
- Per cell line: alpha_x, beta_x, D_t, effective nuclear size.
- Test handle: V79 `alpha_x = 0.093`, `beta_x = 0.026`, `D_t = 30` Gy,
  `R_n = 3.8` um (Elsasser & Scholz 2007 Fig. 6 'LEM old');
  `test_golden_lem1_rbe_let_has_single_maximum_v79`.

## Initial-slope and beta variants
Shared by LEM I/II/III on `LEMBase`; KS2006 forms in
`rapidLEM_KraemerScholz2006.md`.

| `initial_slope_model` | formula | behaviour |
|---|---|---|
| `cross_section` (default) | impact-parameter `sigma_0/(A_eff D_1)` | exact; overkill thin-down |
| `lethal_central` | `min(N_1/D_1, s_max)` | TRiP98-table compatible; plateau at `s_max/alpha_x` |
| `lethal_full` | `∫ nu(b) 2 pi b db / (A_eff D_1)` | overkill-free, keeps indirect traversals |
| `lethal_central_saturating` | `(1 - exp(-N_1))/D_1` | KS2006 eq. 9 |

- `lethal_central` saturates because `N_1 = s_max D_1` whenever the track fits
  inside the nucleus with local dose above `D_t` everywhere; the TRiP plateau
  `s_max/alpha_x` is reproduced exactly for every table tested. Raises for LEM IV. Golden:
  `test_golden_lem1_lethal_central_matches_full_trip_curve` (whole TRiP LEM I
  RBE_alpha curve to a few %). The default and TRiP agree in the therapeutic
  window and diverge below ~10-30 keV/um.
- **SK1997 approximation branch (`scholz1997_central_initial_slope`) carries the
  halo term:** `alpha = (1 - e^-N_1)/D_1 + alpha_x (1 - f)`, with `f` the
  in-nucleus energy fraction (`LEM1.traversal_energy_fraction`; 0.93 at 20, 0.68
  at 100, 0.55 at 400 MeV/u). The bare form converged to `alpha_x f`, i.e. below
  `alpha_x` above ~100 MeV/u for protons. With the halo it agrees with the exact
  cross-section to <=0.3% at every high-energy point; `f = 1` exactly where the
  track fits inside the nucleus. **Do not replace it by a clamp at `alpha_x`**:
  the exact branch legitimately goes below `alpha_x` at extreme LET (Xe 1 MeV/u
  0.32 `alpha_x`, U 0.20; `test_overkill_thindown_takes_alpha_below_the_photon_slope`).
  `lethal_central_saturating` is deliberately left without the halo (KS2006 eq. 9).
- Beta models: `slope_matching` (default, `beta_I = (s_max - alpha_I)/(2 D_t)`),
  `rapid_lem` (KS2006 eq. 10), `scholz1997_two_track` (exact O(phi^2) pair
  excess), plus LEM I's `scholz1997_stochastic` and `scholz1997_montecarlo`.
- Photon reference doses must respect `D_t`: `PhotonParameters.dose_for_survival`
  inverts the shouldered->exponential curve (the plain LQ root is too small above
  `D_t`). For the SK1996 V79 inputs the 1% photon dose is 9.2 Gy < `D_t`, so the
  branches coincide there.

## Implementation notes / gotchas

**Traversal-conditioned survival (one curve gives SK1996 Figs. 3 and 4).**
`scholz1997_traversal_conditioned_survival` estimates
`g_k = E[exp(-N) | exactly k tracks in the b_max disc]`, which is
dose-independent; `S(D) = sum_k Poisson(k; lambda(D)) g_k` is monotone and
analytically differentiable, the final slope is a closed-form functional of `g`,
and iso-survival doses are root-finds. `g_0 = 1` makes the rare zero-traversal
histories that carry 1% survival exact. Do not go back to per-dose resampling.
- Uncertainties: delete-one jackknife over batches, never a two-stream
  difference (~1 dof; an allocator stops on a lucky pair). Measured on Ar
  3 MeV/u: the per-dose path at 1000 histories had a true 15% spread while
  reporting 2.1%; the conditioned path reaches ~1% of the converged 53.8 um2.
- `benchmark.scholz_kraft1996.estimate_final_slope_cross_section` /
  `estimate_iso_survival_rbe` remain for sampled `(dose, S)` pairs: isotonic
  projection + one local quadratic. Keep **one** estimator; switching on
  monotonicity makes the choice a function of MC noise (~10% on U 20 MeV/u).
- The `g_k` are the cluster expansion: `-ln S = lambda(1 - kappa_1) -
  kappa_2 lambda^2/2 - kappa_3 lambda^3/6 - ...`, `kappa_1 = g_1`,
  `kappa_2 = g_2 - g_1^2`, `kappa_3 = g_3 - 3 g_1 g_2 + 2 g_1^3`. Any n-body
  term's size is a few lines against an existing curve.
- Pick the path per observable: conditioning wins for SK1996 (final slope at 1%)
  but costs 23x more per realization for SK1997's 0.5-3 Gy fit (carbon 6 MeV/u),
  which keeps the per-dose path with an 8-stream jackknife pooled in survival
  before the fit (never average per-stream fits; the fit is nonlinear in S).

**Kellerer weighted sampling (the field integral).** SK1997 select points "in
such a way that their spatial density is proportional to the local dose, d",
citing Kellerer 1985 (*Fundamentals of Microdosimetry*, sec. VI.A.2): pick track
`i` with probability `∝ d_bar(b_i)`, then a radius with density `∝ D(r)`
restricted to the nucleus and an azimuth inside the arc
(`LEMBase.nuclear_dose_table`, `LEMBase.kellerer_field_integral`). Then
`p(x) = d(x)/(A D_tot)`, `D_tot = sum_i d_bar(b_i)` exact, and
`N_leth = D_tot <h(d)/d>` with `h(d)/d` bounded in `[alpha_x, s_max]`, so the
variance does not grow with the track count (measured flat to `n_bar = 336 293`;
beyond that is extrapolation).
- Near/far split `b_cut_um`: tracks beyond it are a uniform background whose
  moments are closed-form compound-Poisson. Choose `b_cut` from the measured
  far-field fluctuation (0.34% proton 70, 0.25% carbon 100, 4.5% carbon 40), not
  a fixed radius.
- Interpolate the inverted radial CDF in `log r` (exact for a 1/r^2 profile).
- The compiled ratio-sum kernel (`_mc_kernel`) is seed-reproducible (sampling
  stays in NumPy; pinned at rel 1e-9).
- Recurring defect class: effort allocated by track count rather than dose
  contribution. Anything keyed to `n_bar` deserves suspicion where `r_max`
  varies strongly with energy.
- The exact two-track kernel is `G(d_i,d_j) = h(d_i+d_j) - h(d_i) - h(d_j)`; the
  unsaturated product `2 beta_x d_i d_j` is +22% to +337% wrong at `D_t = 30` Gy.

**Field points per track.** The per-dose estimator collapses below ~1 sample per
track (carbon 50 at 0.55 returned `beta = 2.2 beta_x`, unphysical); a clean
`alpha` is not a sufficient gate for the `beta` of the same fit. The
traversal-conditioned estimator is far less sensitive (it evaluates every prefix
from one sweep; deep survival is carried by the best-sampled small-k terms).
Proton 20 MeV/u (`lambda = 9150` at its 1% dose):

| `n_field_points` | samples/track | `sigma_1%` | RBE 80/20/1% |
|---|---|---|---|
| 2000 (default) | 0.15 | 0.3240 +/- 0.0022 | 1.945 / 1.337 / 1.123 |
| 6000 | 0.46 | 0.3171 +/- 0.0034 | 1.938 / 1.334 / 1.119 |
| 18000 | 1.38 | 0.3159 +/- 0.0037 | 1.928 / 1.337 / 1.121 |

i.e. `sigma_1%` 2.5% high and one-signed at the default, RBEs within 1%. Size a
convergence study by whether one rung's error bar is smaller than the step
between rungs, not by the number of rungs; never quote an approximation's
accuracy against an unconverged MC (at 200 realizations Ar 20 MeV/u read
64.9 +/- 5.9; converged 48.5 +/- 1.4).

**The "LQ approximation" is the truncated cluster expansion, not a fit.**
`LEMBase.approximate_inactivation_cross_section_um2` returns
`sigma_approx(S) = 0.16 LET (alpha_I + 2 beta_I D_S) / rho`. With
`beta_model="scholz1997_two_track"`, `alpha_I = sigma_0` (first order) and
`beta_I` the pair-overlap excess (second order, `1 - e^-Delta` and `e^-nu_1`
resumming some higher structure); its error is third order in track overlap.
Against the converged 40-point SK1996 checkpoint: `sigma_1%` within 5% at 37/40
points, every RBE within 3% at 111/120 (particle, energy, level); exceptions are
U 16 and 20 MeV/u (-14%, -25% on sigma; -9%, -15% on RBE_1%). Fig. 4's p/He/C are
within 4.4% at every level (1% series within 1.7%).
- **Never draw it with `beta_model="slope_matching"`.** `sigma <= pi (R + r_max)^2`
  (geometric ceiling). Slope matching breached it at 84 of Fig. 2's 126 grid
  points, worst 40x (U 1 MeV/u: 1993 um2 against a 50 um2 nucleus). Mechanism:
  `beta_I = (s_max - alpha_I)/(2 D_t)` is a high-dose asymptotic identity; at high
  LET `alpha_I -> 0`, `beta_I -> s_max/(2 D_t) = 0.046 Gy^-2 ~ beta_x` (SK1996
  V79), and at `D_1% = 9.9` Gy `sigma_approx -> 0.16 LET s_max / rho`, unbounded
  in LET. `tests/models/lem/test_lem1.py` asserts the ceiling on both cross-section
  methods and pins the slope-matching breach as a documented defect.

**The cluster series does not converge order by order above `D_t`.** For affine
`h` the n-th inclusion-exclusion excess is `psi_n = (-1)^n beta_x D_t^2` for every
`n >= 2` (5.7600 with the SK1996 V79 inputs, verified n = 2..6;
`test_saturated_branch_generates_the_non_decaying_alternating_term`). For `k`
fully overlapping saturated tracks the exact excess is `beta_x D_t^2 (k-1)`;
truncation gives `C(k,2)` at second and `C(k,2) - C(k,3)` at third order
(k = 20: 19 exact vs 190 and -950). `-ln S` from the `g_k`, truncated:

| case | `lambda` | exact | O(1) | O(2) | O(3) | O(4) |
|---|---|---|---|---|---|---|
| U 5 MeV/u @ 1% | 5.1 | 4.605 | 4.603 | 4.609 | 4.601 | 4.607 |
| U 20 MeV/u @ 1% | 16.4 | 4.605 | 3.880 | 6.255 | -2.789 | 46.2 |
| C 20 MeV/u @ 1% | 233 | 4.605 | 3.610 | 11.4 | 6742 | -1.09e6 |

Gains must come from resummation, not more orders. At 1-3 Gy heavy ions have
`lambda < 1` (U 20 MeV/u: 0.26, 0.78) and second order reproduces the exact
log-kill to 0.4%, so the two-track beta is sound over SK1997's clinical range.
Below `D_t` `h` is quadratic and the pair sum is exact; all higher-order interplay
comes from `-beta_x (d - D_t)^2 1[d > D_t]`. `LEMBase.saturation_radius_um` /
`saturated_area_fraction` locate a case on that map, but the saturated **area**
does not bound the error: at n = 400 the saturated 0.26% of the area carries
51.8% of the pair integrand. The untried lever is a saturated/unsaturated split:
above `D_t` the whole series sums to `beta_x D_t^2 (k-1)` with no integral (99.9%
of the nucleus for U 20 MeV/u).

**Shell integration matches SK1994 eq. (4).** `LEMBase._impact_log_kill` uses
shells concentric with the track, each weighted by the arc inside the nuclear
disk (`_angular_fraction_inside`, law of cosines = the paper's
`phi2(r) - phi1(r)`), over `s in [max(s0, b-R), min(r_max, b+R)]`. Differences
that change no number: a fixed merged grid (512-node log profile + 257-node
linear overlap band; refining 8x moves `sigma_0` by 0.016%) instead of 5%-dose
shells, and trapezoid instead of a rectangle sum (not Simpson: merged grids can
place nodes ~1e-14 apart). For overlapping tracks the paper says only "the
effect has to be determined based on the total local dose"; we use the weighted
field points above.

**Fig. 4 energy range is 1-20 MeV/u** (SK1996 Fig. 4 caption). The SK1994
sibling says 1-10 MeV/u for p/He; do not "fix" the reconstruction to it.
**Residuals stay inside the paper curve's support**
(`compare_points_to_curve`; Fig. 3 linear in E, Fig. 4 log in LET; no
extrapolation). The jackknife error measures MC precision, not a confidence
interval on the model-paper residual. Keep sweeps at `E >= ~0.5` MeV/u: below
~0.4 MeV/u `r_max = 0.05 E^1.7` drops under `r_min = 10 nm`.

## Reconstruction status
Scripts: `validation/scholz_kraft1996_lem1.py`, `validation/scholz1997_lem1.py`,
`examples/lem1_variants.py`; MC checkpoints `validation/data/sk1996_mc_observables.csv`,
`mc_results.csv`, `sk1997_z_slope.csv`, generated by `scripts/lem1_mc_reference.py`.
Both papers' figures are hand-digitized (`benchmark/scholz_kraft1996.py`,
`benchmark/scholz1997.py`).

- **SK1996 Fig. 2.** Uranium `sigma` legitimately exceeds the 50 um2 nucleus (to
  ~110 um2; impact parameters that miss still cause inactivation); reproduced to <1%.
- **SK1996 Fig. 3** (paper's own LET below 12 MeV/u). `sigma_1%` vs `sigma_0` RMS:
  carbon 4.2% vs 9.1%, argon 1.7% vs 3.0%. Uranium within 4% up to 12 MeV/u, then
  +19% at 16 and +37% at 20 (open, below). With the paper's inputs `sigma_0`
  alone matches the digitized line to 2.4% (Ar) / 2.1% (U) RMS, ruling out
  stopping power, `r_max`, geometry and normalisation for the first-order term;
  carbon `sigma_0` is ~10% low because its shouldered curve needs the final
  slope.
- **SK1996 Fig. 4.** Proton `RBE_80` sits at ratio 0.919 to the paper (12.7%
  max); RBEs move <1% across a 9x field-point change, so it is a genuine
  model-paper disagreement. Deriving RBE from the paper's own Fig. 3 `sigma`
  shows Figs. 3 and 4 mutually inconsistent for carbon (~15% at the mid-LET
  peak). That derived RBE (`paper_sigma_derived_rbe_1pct`,
  `iso_survival_rbe_from_final_slope`) is an upper bound that tightens with LET
  (a final slope lacks the intercept; table in the function docstring), so do not
  read a low-LET gap against Fig. 4 as a paper inconsistency.
- **SK1997 Fig. 1** (`alpha`). MC tracks the deterministic cross-section to
  <=0.5% at every point. Approximation branch vs the paper's approximation curve
  above 15 MeV/u: RMS 2.7% (protons), 1.5% (carbon). Exact branch: carbon runs
  ~10-12% under Fig. 1 from 50 MeV/u (open, below).
- **SK1997 Figs. 2/4** stored as LQ line fits (R^2 >= 0.998); **Fig. 5 derived**
  (`benchmark/scholz1997.py` docstring). Fig. 4's carbon "400 MeV/u" exact panel
  is a paper typo (its `alpha_z` matches 200; stored as 200). Fig. 4's exact
  (left) panels condition the spatial model on z:
  `LEM1.scholz1997_specific_energy_survival` (z-conditioned MC; not recoverable
  from the traversal-conditioned `g_k`, since z varies at fixed track count);
  the right panels are eqs. (10)-(11), `scholz1997_specific_energy_slope`.
- **SK1997 Fig. 3** (`beta`), MC vs `scholz1997_two_track_beta` where beta is
  measurable (1.2-3.7% precision):

  | | range | mean MC/two-track | spread |
  |---|---|---|---|
  | protons | 4-12 MeV/u | **0.991** | all within 3% |
  | carbon | 18-40 MeV/u | **1.086** | +6% to +11%, every point >= 2.4 sigma |

  Below p 4 and C 18 MeV/u beta is near zero and the ratio is noise. The carbon
  shortfall has the same cause as the uranium `sigma_1%` one: the pair term
  under-captures dense overlap. As the field homogenises, the two-track beta
  saturates at `1.003 beta_x` for protons and approaches it from below for carbon
  (0.92 at 20 GeV/u).
- **SK1997 Fig. 5** proton 200 MeV/u: RMS 0.408 -> 0.038 via
  `scholz1997_explicit_lq`.
- **Elsasser 2008b Table 1 (CR10 = LEM I):** -0.9% / +1.3% at 13 / 77 keV/um, see
  `LEM_TrackStructure_Elsasser2008b.md`.

## Known limits / open findings
- Reproduces Z-dependent RBE-LET branches; RBE-max shifts 30 keV/um (p) ->
  ~300 keV/um (C) (SK1996 abstract, Fig. 4). Systematically underestimates the
  carbon Bragg-peak/entrance RBE ratio (motivates LEM II/III).
- **OPEN: SK1996 Fig. 3 uranium `sigma_1%` +19% at 16, +37% at 20 MeV/u.** Physically
  a shoulder is expected (`r_max = 9.8` um vs a 4 um nucleus, `lambda ~ 21` at the
  1% dose), but the paper's calc line *flattens* above 12 MeV/u (log-log slope
  0.40 -> 0.28) while ours accelerates (0.81). The paper's uranium data stop at
  13.75 MeV/u while its curve runs to 19.9: calculation vs calculation, unanchored.
  Matching by switching to `sigma_0` would fit the wrong quantity. These points
  need >=1000 histories; sub-1000 jackknife bars are optimistic. Ruled out:
  - NS70 -> Bethe stopping-power seam at 12 MeV/u: `d ln sigma / d ln LET` at fixed
    E is 0.40-0.45 and the gap 4.9% / 3.6% -> ~2%.
  - MC estimator: traversal-conditioned vs Kellerer agree to 0.1% at 16 MeV/u
    (119.0 vs 118.9 um2; paper 99.8).
  - Digitization: dense (188-point) smooth trace, no kink.
  - Generic large-track behaviour: C and Ar at the same `r_max/R_eff = 1.40` sit at
    1.037 and 1.015.
  - Integration scheme (below) and field-point resolution (`n_field_points`
    2000/6000/12000 agree for U 20 MeV/u; only 1000 is measurably low).
  - Shell-averaging bias: real for light ions (C 20 MeV/u, 239 tracks, 5%-shell
    averaging -22% vs pointwise, by Jensen on convex `h`), but -0.0% for U 20 MeV/u
    because 99.9% of its nucleus is above `D_t` (C 7.1%, Ar 16.7%), where `h` is
    linear.
  Untried lead: above `D_t` the pair excess is the constant `beta_x D_t^2`, so the
  uranium shoulder scales as `D_t^2` (`D_t = 8` Gy: 0.44x, 20 Gy: 2.78x).
- **`scholz1997_two_track_beta` clamps a negative result to 0** (with a
  `UserWarning`). It fires in deep saturation: U exactly 0 at 12/16/20 MeV/u; Ar 0
  at 8 and 12 but 1.9e-5 and 2.0e-4 at 16 and 20 (a sign change, not a limit). So
  the deterministic uranium `sigma_1%` equals `sigma_0` only because of the clamp.
  Needs resummation or the saturated/unsaturated split, not a different clamp.
- **OPEN: SK1997 Fig. 1 exact-branch carbon alpha ~10-12% low above 50 MeV/u**
  (ratio 0.924 at 50, 0.877 at 400; hump to 1.04 near 4 MeV/u). Our exact and
  approximation branches coincide (<=0.3%) at high energy where the paper's differ
  by ~13% at carbon 400, so look at the profile shape or the cross-section
  construction. The paper contradicts itself exactly there: carbon 400 Fig. 1
  `alpha = 0.4388` vs Fig. 2 0.3847 (ours 0.3852), Fig. 3 vs Fig. 2 beta 0.0223 vs
  0.01937. Ruled out:
  - Numerics: 4x every node count moves it 0.13%.
  - Constant `r_min`: 1 nm overshoots to +38% at 400, 40 nm undershoots to -50%.
  - Velocity-dependent core: shape spread 1.810, worse than shipped.
  - Track radius: a 36-point `(gamma, delta)` scan scored by `max/min` of
    ours/paper gives 1.292 shipped, best 1.271 (`0.05/1.6`); the level is buyable
    (`0.062/1.35`) only by worsening the shape. Do not read a power law off one
    knob forced to absorb a residual.
- SK1996 and SK1997 high-energy residuals (uranium sigma high, carbon alpha low)
  have opposite signs: not one effect.

## Ruled out (approaches, one line each)
- Integration technique as a source of Fig. 3 residuals: the paper's shell scheme
  anchored at the area-weighted mean radius reproduces our trapezoid to <=0.05%
  in `nu(b)` and `sigma_0` (C, Ar, U, 1-20 MeV/u). Where in the shell `D_i` is taken
  is unstated and worth +2.6%/-2.4% in `nu(b)` at 5% dose steps, but only
  +2.5%/-2.1% (C 20), +1.2%/-1.1% (U), +0.2%/-0.2% (Ar) in `sigma_0`.
- A third-order track term: diverges (above), and `psi_3` needs the full triangle
  geometry (roughly the square of the pair cost).
- Tabulated pair sum / low-rank coefficient table / control-variate beta
  correction: obsolete once Kellerer sampling made field points cheap and
  unbiased.
- Uniform track selection (removed sampler): at carbon 100 only 0.12% of
  track-targeted points landed inside the nucleus. Uniform and weighted agreed
  where they overlapped (carbon 20: alpha 0.7868 vs 0.7858, beta 0.00940 vs
  0.00974).
- GPU port: Amdahl ceiling 1.0-1.3x (the compiled kernel is 3.4-21.5% of a call).
- The old fixed-`n_field_points` "`n_bar` ceiling" (carbon 50 at 2000 points:
  alpha -5.9%, beta 2.17 `beta_x`; at 16000: +1.5%, 0.54) was field-point
  under-resolution, not a physical limit.
