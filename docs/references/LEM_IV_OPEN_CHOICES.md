# LEM IV: open and ambiguous model choices

A register of every place where pyRadBio's LEM IV (`pyradbio/models/lem/lem4.py`)
had to *choose* something the source papers do not pin down, plus the scientific
residuals that are still open. It says plainly which defaults are principled,
which are conventions and which are fudges.

Companion documents: `LEM_IV_Elsasser2010.md` (primary paper, Fig. 4
reconstruction), `LEM_IV_Friedrich2012.md` (Figs. 3-4, the `beta_I` record, the
registration defect).

## How to read this

Each entry says what the choice is, what the papers say, what we do, and what
would settle it.

- **[FREE]** a number with no paper basis. We picked it; it changes results.
- **[FORK]** two or more defensible readings of one published prescription that
  give measurably different answers.
- **[DEFECT]** a known-wrong implementation detail deliberately left unfixed, with
  the reason.

**Several entries interact, and some of the agreement in the shipped
reconstructions comes from compensating errors.** Do not tune one in isolation
and read the RBE as confirmation (see "Why RMS cannot pick the engine").

---

## 1. Free parameters (no paper basis)

### 1.1 [FREE] The 0.95 cluster-index cap -- `legacy_trip98_cluster_index_cap`

**Papers:** Elsasser 2010, Friedrich 2012 and Pfuhl all state `choose D_eq such
that C_gamma(D_eq) = C_I`, unbounded. No cap appears anywhere.

**Us:** off by default. The original 0.95 was, by its own code comment, "the
highest cap that removes the artefact for the TRiP98 carbon curves" -- tuned on
the data it was then scored against. Its stated rationale (the inversion is
ill-conditioned near `C_I -> 1`) does not survive measurement:
`d(log D_eq)/d(log(1 - C_I))` is 0.27 at the cap and *falls* to 0.03 deep in
overkill.

**Still required** by the three published-figure goldens (Friedrich Fig. 3,
Elsasser Fig. 4a and 4b), `validation/lem4_elsasser2010_friedrich2012.py` and
`examples/model_comparison_alpha_beta.py`. Uncapped, those panels grow a spurious
local RBE *maximum* at ~430-440 keV/um.

**Why it cannot simply be deleted:** it masked two defects in the same rim
quantity. One (a penumbra quadrature bug) was fixed when the cap was removed; the
other, the giant-loop lattice registration (entry 3.1), produces the bump. Capped
reproduces the figures via a tuned constant; uncapped exposes a real defect.

**Settled by:** fixing 3.1.

### 1.2 [FREE] The LQ fit window -- `_LOCALDOSE_DOSE_FRACTIONS`

**Papers:** neither Friedrich 2012 nor TRiP98 states the dose range of the fit
that turns a simulated survival curve into `alpha_I`/`beta_I`. Marked
`# UNVERIFIED` in the source.

**Us:** 0.05-0.85 x `D_t` (12 fractions). The previous absolute window
(0.4-3.0 Gy) had no basis either: it misread a Friedrich 2012 sentence about the
single-particle approximation as a fit protocol, and its "insensitive to the
window" check was made at LET 70, where `beta` has already collapsed.

**Choice of ceiling.** Refitting the *same* simulated points per window, RMS of
`ln(ours/paper)` over LET 34-67: 0.312 (0.05-0.32 `D_t`), 0.161 (0.05-0.63),
**0.096** (0.05-0.85), 0.099 (0.10-0.85), 0.109 (0.20-0.63). Over the wider LET
13-70 under `mean effect` (entry 2.1, the convention that reproduces the panel):

| ceiling / `D_t` | 0.70 | 0.75 | 0.80 | **0.85** | 0.90 | 0.95 | 1.00 | 1.10 |
|---|---|---|---|---|---|---|---|---|
| RMS `ln(beta)` LET 13-50 | 0.080 | 0.062 | 0.044 | **0.025** | **0.007** | 0.015 | 0.035 | 0.074 |
| RMS `ln(beta)` LET 13-70 | 0.123 | 0.106 | 0.090 | **0.076** | 0.065 | **0.060** | 0.062 | 0.082 |

The wide-range optimum is ~0.90-0.95; **the default stays 0.85** (re-tuning a
free parameter on its own target, for a residual bounded at +0.6% of `RBE_10`,
buys nothing). The neighbourhood is flat, and the ceiling that reproduces the
paper's `beta` exactly, 8.15-8.77 Gy (0.86-0.92 `D_t`) over LET 17.5-55, brackets
0.85.

**What the window is worth** (Fig. 4 cell):

| | old (0.4-3.0 Gy) | this (0.05-0.85 `D_t`) |
|---|---|---|
| `beta` ratio to Fig. 4, LET 40 / 50 / 55 | 1.23 / 1.52 / 1.44 | 1.04 / 1.03 / 1.02 |
| our `beta_I` peak | LET 45 @ 0.0585 | LET 27.5 @ 0.0524 (paper: LET 24 @ 0.0522) |
| worst `RBE_10` cost of the `beta` residual | +5.4% | +0.61% |
| discrete engine, RMS `beta` (both conventions) | ~0.006-0.007 | ~0.0021 |

The old "roll-off starts late from a 12% higher maximum" defect was largely this
window. **The agreement is partly by construction** (the window was chosen on the
`beta_I` it is judged against), so any claim that our `beta_I` reproduces the
paper must state the window. **It also inverted a ranking:** under the old window
the discrete engine's `stochastic` survival improved `beta` and worsened `alpha`;
under this one, the reverse. At deep overkill the fitted `beta` is meaningless
under *any* window (12C 1 MeV/u: the response is concave then convex); `alpha` is
the robust half.

#### Survival-bounded window (Fig. 4 legend `S>0.01`): implemented, not adopted

Friedrich 2012 Fig. 4's legend reads "Full Simulation, S>0.01" in both panels
(caption silent). Read as a fit protocol, the ceiling is the dose where simulated
survival reaches 1%, i.e. LET-dependent. Opt-in as `LEM4.fit_survival_bound`
(`survival_bounded_ceiling_Gy` thresholds the simulated `S = e^{-E}`, not the
fit; node shape and lower bound unchanged). Same simulated response, Fig. 4 cell:

| LET | 30 | 40 | 50 | 55 |
|---|---|---|---|---|
| `S>0.01` ceiling [Gy] | 6.01 | 5.30 | 4.68 | 4.41 |
| `beta` / paper, `S>0.01` | 1.046 | **1.194** | **1.312** | **1.288** |
| `beta` / paper, 0.05-0.85 `D_t` | 1.014 | 1.038 | 1.033 | 1.021 |

A lower ceiling reads a higher `beta`, so `S>0.01` had to read high. If the paper
did fit over `S>0.01`, our simulated survival curve differs from theirs and the
fixed window's agreement is compensation. Why the fixed window is kept anyway:

1. **The paper's curve fits no constant survival bound.** The ceiling that
   reproduces its `beta` is a near-constant *dose* (8.15-8.77 Gy over LET 17.5-55,
   7% spread) while the survival there spans 1.7e-3 to 7.8e-5, a factor 21.
   (LET 13-15 excluded: `beta` is window-independent to 0.7% there. Above LET 60
   the required ceiling runs away -- the known high-LET tail.)
2. **Family against family, both tuned, fixed wins.** The best survival threshold
   (`S>1e-4`) scores RMS `ln(beta)` 0.089 over LET 13-70, worse than every fixed
   ceiling from 0.85 to 1.10 `D_t` (0.060-0.082).
3. **Robust to the escapes.** Under the exact `<e^{-E}>` convention (entry 2.1) on
   a paired explicit-track MC, LET 45/50/55/65 give 1.12/1.11/1.28/1.07 bounded
   against 0.95/0.93/1.00/0.97 fixed. Four lower-bound/node-spacing protocols at
   LET 50 give 1.22-1.35. `alpha` moves <2% either way.
4. **Unbounded on radioresistant cells.** On a chordoma cell
   (`D_t` = 30 Gy, Krämer & Scholz 2006) the `S>0.01` ceiling runs 5.0-29.5 Gy over 0.5-3 MeV/u and
   reaches 36.7 Gy by 20 MeV/u, past `D_t`: a quadratic fitted across the photon
   curve's linear tail.

`alpha` does not decide it: on `full_simulation` chordoma points (H-C, 0.5-3 MeV/u,
ceiling changing up to 5x) the bounded and fixed windows agree equally well with the
TRiP98 tables, and `alpha` moves only a few per cent per point.

**Not settled:** whether the legend is a fit protocol at all (it may mark the
simulation's statistical validity range, since an MC cannot resolve survival much
below 1/N), and if so where our simulated response departs from theirs. Needs a
second reference `beta_I` from a different cell, which the corpus lacks.

### 1.3 [FREE] The MC impact cut

`max(8 r_eff, 0.15 r_sim)` (`_MC_IMPACT_CUT_FACTOR`,
`_MC_IMPACT_CUT_MIN_DOMAIN_FRACTION`). Beyond the cut, tracks are replaced by
their analytic mean. Both numbers are convergence choices. The domain fraction
exists because `8 r_eff` alone was *biased low* at LET 15 by 0.0004 in `beta`, the
size of the intertrack enhancement it was built to measure, while passing two
validations that never varied the controlling quantity.

### 1.4 [FREE] `GLOBLE_N_LOOPS` = 2000

Friedrich, Durante & Scholz 2012 (doi:10.1667/RR2964.1) Fig. 5B: `N_L` ≈ 2000 is
the best fit to the photon ensemble. It is a fit, not independent of
`SUBVOLUME_SIDE_NM` and `NUCLEUS_VOLUME_UM3`; 3000 (the paper's illustrative
geometric value) gave a +31% V79 peak overshoot.

GLOBLE lethalities (`_globle_lethalities`, eqs. 8-9): `eps_i = alpha_x/a_DSB`,
`eps_c = 2(N_L beta_x + a_DSB alpha_x)/a_DSB^2`, `a_DSB = 30 /Gy`. This reproduces
the photon LQ only when DSBs are spread over exactly `N_L` loops. Inverting the
paper's XRS/CHO/V79 sets (`eps_i` = 0.06 / 0.0072 / 0.0036, `N_L` = 3000) recovers
its stated `eps_c = 0.18` for all three. **Two loop counts, on purpose:** the
single-particle path uses the empirical 2000; the intertrack MC calibrates `eps_c`
against the geometric grid `MC_GRID_N_LOOPS = V/l^3 = 3175` it actually bins into
(pinned by `test_lem4_globle_n_loops_is_configurable` and
`test_intertrack_mc_eps_c_calibrated_to_grid_recovers_photon_beta`).

### 1.5 [FREE] Giant-loop geometry -- `SUBVOLUME_SIDE_NM` = 540, `NUCLEUS_VOLUME_UM3` = 500

Elsasser 2010's fitted generic value and reference nuclear volume. They enter
`DSB_PER_GY_PER_LOOP` together, so only the combination is constrained; the split
is ours. The genome-size lever (`LEM_II_Elsasser2007.md`) moves RBE ~2.4% per 10%
and must not be fitted -- it launders a magnitude offset through a clustering
parameter.

### 1.6 [FREE] `effective_size_factor`

Default 1.0; every benchmark sets it explicitly. The old 0.8 double-shrank nuclei
whose papers already quote an effective radius. It traces to Todd 1985 via
Elsasser's ref. (17), whose shrinkage is fluence-dependent and **vanishes as
fluence -> 0**, i.e. at every `RBE_alpha` panel. Treat any reappearance of 0.8 as
a bug.

---

## 2. Forks: two defensible readings

### 2.1 [FORK] Survival convention -- `survival_convention`, and its beta twin

`mean_survival` (default) is `S = <e^-E>`, giving `alpha_I = (1 - e^-N_1)/D_1`
(Pfuhl 2021 eq. 2.32, the standard LEM formula, bounded by `1/D_1`).
`mean_effect` is `S = e^-<E>`, giving the unsaturated `alpha_I = N_1/D_1`. The two
are identical above ~10 MeV/u.

**On alpha it is a trade.** TRiP98's LEM IV tables behave like `mean_effect`:
over H-Ne, the `mean_survival` cell-to-cell ratio misses TRiP's by exactly the
saturation factor predicted from `N_1` alone, while `e^-<E>` matches it everywhere
(`validation/trip98_lem4_rbe.py`). But `mean_effect` has no overkill ceiling, is not
the formula the papers state, and makes light ions worse (protons sub-MeV, chordoma).
It is a uniform win for Z >= 3.

**On beta it decides Friedrich 2012 Fig. 4.** Over LET 15-40 the deterministic
`exp(-<E>)` convolution reproduces the intertrack enhancement: RMS **1.9%**, rise
+3.1% against the paper's +3.5%, peak at LET 28 against 24. The exact
`<exp(-E)>` suppresses it to +1.2%; the B2 analytic overkill (entry 2.6) removes
it (monotone fall). A no-go drawn under one convention does not transfer to the
other.

**Settled by:** nothing yet; the papers do not say. **Always report which one.**

### 2.2 [FORK] Alpha extraction -- `initial_slope_model`

`cross_section` (default) takes the `D -> 0` limit. `full_simulation` fits LQ to
a simulated survival curve, TRiP98's `-iter 50000` protocol (Pfuhl 2021
ch. 2.3.2).

Elsasser 2008b's **printed** Table 1 (the corpus's only digitization-free
cross-check) settles `cross_section` for that paper, to ~2% on all six numbers.
TRiP98's alpha is *fitted*, and because overkill beta is negative the fit
*lifts* alpha; switching roughly halves the error in the TRiP98 low-energy band.
The faithful route depends on the target; the two differ below ~10 MeV/u.

**Caveat:** `full_simulation` scores the wrong object in one respect -- see
`LEM_IV_Friedrich2012.md`, "Why `full_simulation` does not track TRiP98 at
high Z".

### 2.3 [FORK] Cluster-index estimator -- `cluster_index_model`

| model | what it is | where it is right |
|---|---|---|
| `mean_field` (default) | Elsasser 2010 eq. 2 on the continuous pattern, track at a subvolume centre | above ~3.5 MeV/u; also closest below ~0.25 |
| `discrete` | deterministic average-of-ratios, marginalised over impact parameter and lattice phase | ~2.5 MeV/u and up, reconciled with the MC to 1-7% |
| `discrete_mc` | sampled 3-D DSB pattern | below ~2 MeV/u; the only one reproducing the Pfuhl Table 3 anchors |
| `trip98_composite` | `discrete` while the profile fits one loop (2.206 MeV/u), else `mean_field` | the TRiP98 reconstruction |

**The default over-clusters and we keep it.** `mean_field` is a ratio of
mean-field areas, so Jensen pushes `C_I` above the discrete average-of-ratios:
~0.99 for 12C 2.5 MeV/u and ~0.31 for 20Ne 31 MeV/u, against Pfuhl 2020 Table 3's
0.80 / 0.19 (doi:10.1667/RR15456.1). A secondary contributor is the 2-D
transverse projection x depth scaling, which collapses the z-column that carries
the clustering of a compact high-LET track. Routing the RBE path through the
deterministic discrete estimator is **2.6-12x worse**, because the over-clustering
is a load-bearing cancellation against other biases (entries 1.1, 2.4). The final
RBE is validated while this intermediate quantity is known wrong: a good RBE does
not license a claim about any internal quantity.

**Settled by:** re-deriving the estimator and what compensates for it *together*.

### 2.4 [FORK] `eta` convention -- Variant A vs Variant B

Does the SSB->DSB enhancement `eta(D)` (Friedrich 2015) scale the DSB
**placement** (A: DSBs distributed by `eta(D) D`) or only the DSB **count** (B:
yield scaled by `<eta>`, positions by the plain `D(r)`)? The papers do not say.

**The code uses both.** `mean_field` (default) and `discrete` place by
`eta(D) D` (Variant A); `discrete_mc` places by plain dose (Variant B). Measured
with the discrete 3-D binning, only the placement density swapped (count held
fixed; seed 1, 6000 realisations):

| beam | B (plain `D`) | A (`eta D`) | `mean_field` | Pfuhl Table 3 |
|---|---|---|---|---|
| 12C 2.5 MeV/u (LET 437, saturated) | 0.747 | 0.863 | 0.991 | 0.80 |
| 20Ne 31 MeV/u (LET 183, unsaturated) | 0.195 | 0.296 | 0.314 | 0.19 |

Variant B matches the anchors; Variant A biases `C_I` up by ~0.10-0.12. For the
unsaturated beam the placement bias is almost the whole mean-field
over-clustering (0.296 vs 0.314); for the saturated beam both this and the
Jensen/z-collapse of entry 2.3 contribute. They are distinct mechanisms.

**Settled by:** if the intermediate `C_I` is ever made physical, switch placement
to B *and* re-run the full RBE golden suite -- the cap and slope matching were
tuned around the biased `C_I`. Do not change it blind.

### 2.5 [FORK] `beta_model` -- which is "the paper's own method" depends on the panel

`slope_matching` (default) is the Kraemer-Scholz 2006 approximation -- a
Friedrich 2012 device, correct for its Fig. 3. Elsasser 2010 derives its
coefficients "by the method of the original LEM (13)" (Scholz 1997, the full
simulation), so `local_dose` is the faithful engine for Fig. 4b -- though on the
corrected Fig. 4b reference it *adds* RBE_10 bias (numbers in
`LEM_IV_Elsasser2010.md`), so fidelity and fit disagree. One engine for both
panels is wrong for one of them.

`intertrack_mc` and `discrete_dsb` are documented cross-checks, not production:
`discrete_dsb` was built to recover the high-LET roll-off and provably cannot.

### 2.6 [FORK] `local_dose_overkill` (the B2 correction)

Approximates `<exp(-E)>` by evaluating the mean-effect curve at `n D_1`. Against
the exact MC it **over-corrects by 1.9-5x**, yet gives the best RMS against
Fig. 4's `beta_I` of any engine -- compensation, not accuracy. Off by default.

### 2.7 [FORK] `effect_model` -- `photon_equivalent` vs `globle`

Fig. 4b was computed with `D_eq`, so GLOBLE must not be scored against it.
GLOBLE's beta is pinned at `beta_x` by dose conservation wherever per-loop
occupancy is small (measured mu <= 0.2 through LET 55), so it carries no LET
information there.

**Neither can be calibrated to the high-LET `beta_I` from photon data.** The
per-loop lethality `eps(k)` (loop with `k` DSBs) is not identifiable from the
photon response: a pure-LQ photon curve gives exactly
`eps(k) = eps_i k + b k(k-1)`, `b = beta_x N_L/a_DSB^2`, so GLOBLE eq. 2's
truncation at `k = 2` is forced. Inverting LQ + `D_t` tail for `eps(k)` is
ill-posed (weighted condition number 3.9e18; `sum_k P(k;m) k = m`, so the tail
fixes only a trend). `D_t` and clustering live in disjoint dose regimes: at
`D_t` = 9.5 Gy a loop holds 0.09 DSBs; loops cluster (`m = 1`) only near 106 Gy.
So the families bracket the paper: `photon_equivalent` extrapolates the linear
tail to ~100-250 Gy (collapses `beta`, under-produces `alpha` at high
clustering), `globle` extrapolates the LQ (`beta` ~ `beta_x`, over-produces
`alpha`). Closing the gap needs an ion-data target or an explicit shape
assumption for `eps(k > 2)`.

---

## 3. Known defects, deliberately unfixed

### 3.1 [DEFECT] The giant-loop lattice registration in the mean field

`_damage_pattern` places the track exactly on a subvolume centre. Every isolated
DSB -- all of `1 - C_I` -- is then the sliver of profile spilling over the
subvolume boundary, and once the support shrinks past half a side (0.270 um,
2.206 MeV/u) that sliver vanishes. `1 - C_I` collapses six orders of magnitude
(1.5e-1 at 12C 4 MeV/u to 5.4e-8 at 1.5) and `D_eq` runs 309 -> 1730 Gy.
Marginalising the lattice phase over 64 R2 offsets removes it: `1 - C_I` stays
0.08-0.21 and `D_eq` rises smoothly 264 -> 396 Gy.

The discrete estimator already marginalises the phase; the mean field does not.
**Not fixed because** it moves `C_I` everywhere, hence every golden, and collides
with 2.3 (the over-clustering is load-bearing). This and 1.1 are one piece of
work.

### 3.2 [DEFECT] `box_averaged_profile` smears the track ~2.5x too wide

It reads a square's average as a function of radial distance alone. A 33.8 nm
12C 0.126 MeV/u track gets an 84 nm 90%-to-10% transition, over-populating
neighbouring loops at exactly the occupancy `1 - C_I` reads. This is why the
deterministic `discrete` estimator runs 3-6x above the sampled MC below
~1 MeV/u. Refining the quadrature cannot help -- the error is in the integrand.
Needs a real per-loop 2-D integral or an orientation-aware box average.

### 3.3 [DEFECT] Numerical resolutions chosen, not derived

`_CORE_HALF_SUBVOX` = 4 and `_FINE_STEP_NM` = 3.0 (reproduce a finer reference to
<1% across the therapy LET range), `_RADIAL_NODES` = 1500, `_DISCRETE_N_IMPACT` =
384, `_DISCRETE_MC_REALISATIONS` = 16000, `_INTERTRACK_REALIZATIONS` = 600, MC
field points per track = 8. Below ~1 sample per track the MC gives an unphysical
`beta` while `alpha` stays clean, so **alpha is not a convergence gate**.
Convergence in `1 - C_I` is *oscillatory*, so two node counts are never a test.

---

## 4. Open scientific residuals (not choices)

- **Fig. 4 `beta_I` late roll-off above LET ~70.** Ratios to the paper 1.15 / 2.1 /
  3.9 at LET 70 / 85 / 100. The enhancement band 15-40 is closed under
  `mean effect` and LET 55 agrees (0.999). Not closable by a photon-calibrated
  effect model (entry 2.7).
- **TRiP98 divergence below ~2 MeV/u.** `N_dmg` pins at 0.2916 um^2 (one 540 nm
  column), both magnitude carriers die, RBE goes pure `1/LET`. GLOBLE,
  full-sim alpha and multi-traversal were each measured and rejected as the fix.
- **Seed-scatter error model.** The Fig. 4 beta quadratic gives chi2 0.65 on 4 dof
  (p ~ 0.96) -- too good, so the seed-scatter error is probably overestimated or
  the seeds are not independent. The sign and size of the enhancement are safe;
  **the 4.3 sigma is not**, and should not be quoted until this is checked.
- **LEM I carbon alpha 10-12% under Scholz-Kraft 1997 above 50 MeV/u**; numerics,
  `r_min`, velocity core and `r_max` all eliminated (inherited, not LEM IV).
- **KS-1 high-`alpha_x` overkill outlier** in the LEM III per-cell dRBE goldens.

## 5. Choices that are *not* open (recorded so they stay settled)

- **`r_c` = 6.5 nm** velocity core -- Pfuhl eq. 1, cited, not LEM III's 40 nm.
- **Initial-slope method = `cross_section` for Elsasser 2008b** -- fixed by that
  paper's printed Table 1; core radius and `D_t` are degenerate there, so do not
  read either off it.
- **`GLOBLE_N_LOOPS` = 2000, not 3000.**
- **`effective_size_factor` = 1.0** by default.
- **Elsasser 2010 Fig. 4b's y axis runs 0-11, not 0-10.** The old ~-10% "HSG alpha
  deficit" was the resulting 10/9 digitization error, not the model
  (`LEM_IV_Elsasser2010.md`).
- **Low-LET `RBE_alpha -> 1` from above** on the `globle` and
  `photon_equivalent` effect models (`test_lem4_low_let_ion_is_marginally_more_effective_than_photons`).
  The single-particle default path keeps a documented low-LET dip below 1 and is
  excluded.

---

## Why RMS cannot pick the engine

- `local_dose` + `local_dose_overkill` has the best RMS against Fig. 4's `beta_I`
  while being *measured* to over-correct 1.9-5x.
- The discrete engine's `stochastic` survival improves `beta` RMS and degrades
  `alpha` RMS at the same time -- fit quality and rigour rank oppositely.
- `mean_field` over-clusters badly and still gives the best RBE, because the bias
  cancels downstream.

The working rule: **score against the papers' model lines, never against the
experimental points**, quote the protocol axes (survival convention, fit window,
alpha route, cluster-index estimator) with every number, and treat an RMS
improvement obtained by changing one axis as a hypothesis, not a result.
