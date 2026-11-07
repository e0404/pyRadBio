# LEM II — Elsässer & Scholz 2007 (doi:10.1667/RR0467.1)

Radiat. Res. 167:319-329. Code: `models/lem/lem2.py` (`LEM2`, `ClusterModel`,
`Friedrich2015ClusterModel`); data and per-panel provenance:
`benchmark/elsasser2007.py` (module docstring); reconstruction:
`validation/elsasser2007_lem2.py`.

## One-paragraph idea
Two additions to LEM I: (1) a *cluster extension* — clustered SSBs forming
additional DSBs at high local dose modify the high-dose tail of the photon
survival curve; (2) a refined radial dose profile with a much smaller core
radius plus explicit radical-diffusion smearing.

## Core equations
- eta(D) = [N(DSB) + N(SSB+SSB->DSB)] / N(DSB)             (eq. 9)
  with SSB = 1250/cell/Gy, DSB = 30/cell/Gy, stagger h = 25 bp; onset ~1e3 Gy,
  maximum at ~5e5 Gy (Fig. 2). Rearranged:
  `eta - 1 = (a_SSB/a_DSB) q(mu)/mu`, `mu = a_SSB D h / L_gen`.
- Photon survival with cluster enhancement on the exponential tail (eq. 10):
  shouldered LQ below `D_t`; above it `-ln S_x(D_t) + s_max (eta(D) D - D_t)`.
  **eta multiplies the dose; it is not integrated over it** (the integrated
  reading misses Fig. 3 by 22.7% RMS; `benchmark/elsasser2007.py`).
- Radial dose: r_min = 0.3 nm (was 10 nm); r_max = gamma E^delta,
  gamma = 0.062 um (MeV/u)^-1.7, delta = 1.7.
- Radical diffusion: convolve D(r) with a 2-D Gaussian sigma = 4 nm; smeared
  profile via modified Bessel I0 (eqs. 11-12).
- Initial-slope approximation: alpha_I = (1 - S_1)/D_1,
  D_1 = 0.16 * LET / (rho * A)  (eq. 8; LET keV/um, A um^2, rho g/cm^3). **The
  paper's curves were not computed with it** — see "Initial-slope method".
  **Printed with `rho` in the numerator** (p. 321: "D1 = 0.16 LET·ρ/A"), which is
  dimensionally wrong: 1 keV/um = 1.602e-10 J/m over `rho A` = 1e-9 kg/m (1 g/cm^3,
  1 um^2) is 0.1602 Gy, so `rho` divides. The paper's results are unaffected (`rho`
  = 1). Kase et al. 2008 (doi:10.1088/0031-9155/53/1/003) eq. (15) quotes the inverse,
  the traversals per gray `phi = 1/D_1`, as `0.16 rho A / LET`; correct is
  `rho A / (0.16 LET)`. Its eq. (13) text gives the Hawkins form `z_1Dn = 0.16 LET/A`
  (`rho = 1`), which agrees. Scholz 1997 prints no prefactor (traversals per unit dose
  `N_T,E`, eq. 5). The code uses `0.16 LET / (rho A)`.

## Parameters (photon inputs; D_t old -> new after refit)
- V79: alpha_x = 0.093, beta_x = 0.026, D_t 30 -> 7.5 Gy, R_n = 3.8 um.
- CHO: alpha_x = 0.228, beta_x = 0.020, D_t 40 -> 9.5 Gy, R_n = 4.7 um.
- HSG (Fig. 11): alpha_x = 0.313, beta_x = 0.0615, D_t 30 -> 6.0 Gy, R_n = 5.0 um.
- Stagger-size sensitivity h in {10, 20, 25, 35} bp (Fig. 7).
- Genome size `L_gen`: **not stated**; we default to 6.0e9 bp (Friedrich, Durante
  & Scholz 2012 use `S_G = 6e9` bp; Friedrich 2015 uses 5.4e9).
- gamma: use the stated 0.062 for both thin (LEM I) and thick lines — the r_max
  parameterisation (after eq. 6, p. 320) is introduced for the *original* model as
  well; `ScholzKraftRadialDose.lem1()`'s 0.05 is the SK1996/1997 value.
- Fig. 11 data x is Furusawa 2000's *dose-averaged* LET (p. 326), not track-segment.

## Key results / golden values
- Raises the RBE-max/RBE-min ratio vs LEM I (Fig. 6 'LEM new' > 'LEM old').
- Both the cluster extension AND the refined radial dose are needed; combining
  them without reducing D_t gives unrealistically high RBE (Fig. 6 thin dashed).
- D_t cross-check via single-hit-multitarget slope (Discussion of parameters):
  HSG 1/D0 = 1.23 vs S_t = 1.11; predicted D_t = 7.4 vs used 6.0 Gy.
- Goldens in `tests/models/lem/test_lem2.py` (`test_golden_cluster_eta_onset_and_maximum`,
  `..._raises_rbe_max_to_min_ratio_v79`, `..._without_dt_reduction_is_unrealistically_high`,
  `..._stagger_size_orders_rbe_ratio`, the two `friedrich2015` goldens) and
  `tests/benchmark/test_elsasser2007.py::test_golden_paper_curves_follow_the_full_average_not_its_stated_equation_8`.

## Verdict: reconstruction closed
The residuals below are not attributable to a defect on our side; no further
parameter search is warranted. The paper contradicts itself in three places:

1. Fig. 3's caption contradicts Fig. 8's caption on CHO's `D_t` (30 vs 40).
2. Fig. 2's eta is unreachable from the paper's stated rule, and Friedrich 2015's
   closed-form recomputation from the same inputs lands on ours (13.5), not 14.79.
3. The figures do not behave like the paper's own eq. (8) (monotone residual in
   `r_max/R_eff`, −10.2% -> −0.1%, where the exact average is flat to ~1%).

Five quantities are never given: genome size, `beta_I`, the effective-size
procedure (deferred to Todd 1985, which does not apply at alpha), the
normalisation convention, and restricted vs unrestricted LET.

All LEM versions share `LEMBase` (geometry, stopping power, radial-dose protocol,
impact-parameter integration), which reproduces Friedrich 2012 (LEM IV) and the
MKM-family papers; the residual localises to version-specific layers or the
papers. **The reference is the paper's model lines**; experimental points
(Weyrather 1999, Furusawa 2000) are context only, and the paper flags its own
low-LET model-vs-data gap (pp. 327-328).

## Initial-slope method: the cross-section integral, not eq. (8)
The "Comparison of Modifications" section (p. 323) states one procedure for all
RBE_alpha curves: "simulate only a single ion traversal at the center of the cell
nucleus [...] Using Eq. (8)". That is `initial_slope_model="lethal_central_saturating"`.
It drops every track that misses the nucleus, so its error tracks `r_max/R_eff`.
Bias on Fig. 6 `lem_old` (no clusters, all parameters stated), `METHOD_BANDS`:

| LET [keV/um] | r_max/R_eff | eq. (8) | exact impact integral |
|---|---|---|---|
| 13–30 | 84 | **−10.2%** | +0.6% |
| 30–70 | 11.3 | −5.4% | −0.1% |
| 70–150 | 2.2 | −1.8% | +0.1% |
| 150–300 | 0.5 | −0.7% | +0.2% |
| 300–600 | 0.1 | −0.1% | +1.7% |

The monotone eq. (8) residual is the signature of a reference *not* made with
eq. (8) (introduced under "Approximation ... For treatment planning"). Settled
independently on printed numbers by Elsässer et al. 2008b Table 1, whose sec. 4
cites the SK1997 cross-section integral: `LEM_TrackStructure_Elsasser2008b.md`.
`validation/elsasser2007_lem2.py` still runs eq. (8) by default
(`INITIAL_SLOPE_MODEL`); the residual tables below are on that basis.

Gamma vs method, bias on Fig. 6 `lem_old` — a residual-minimising gamma just
absorbs the method difference, so take the stated 0.062:

| | gamma = 0.05 | gamma = 0.062 |
|---|---|---|
| eq. (8) | −1.8% | −4.2% |
| exact impact integral | +2.4% | +0.4% |

## Discrepancy budget (against the paper's model lines, eq. (8))

| # | source | size | status |
|---|---|---|---|
| 1 | initial-slope method (eq. 8 vs full average) | up to −10%, monotone in `r_max/R_eff` | measured; curves match the average |
| 2 | stopping power: our LET(E) 1.9–10.5% low vs the four (E, LET) pairs in the Fig. 10 caption | −3 to −7% in alpha | measured; masks part of the deficit (below) |
| 3 | effective nuclear area (Todd 1985) | ±10–15% per ±7.5–15% in radius | best single lever, not sufficient |
| 4 | gamma 0.05 vs 0.062 | 2.4% on Fig. 6 `lem_old` | not resolvable from residuals; 0.062 confirmed by 2008b |
| 5 | eta rule (Fig. 2 unreachable) | ≤5% over track-reachable doses | LEM II only |
| 6 | genome size | ~2.4% per ±10% in `L_gen`; ±4% | LEM II only; do not fit |
| 7 | normalisation (radial integral over `[0, r_max]` incl. core, or outside it) | few % in the core | unverifiable |
| 8 | `beta_I` method | — | affects Figs. 9/10/12 only |
| 9 | restricted vs unrestricted LET | — | assumed unrestricted |

**#2 works against us.** Our LET(E) runs low, so a given plotted LET sits at a
lower energy (smaller `r_max`, more dose retained), inflating alpha; `D_1` is
unaffected. Adopting the paper's implied E<->LET would move alpha by −3.4 / −7.2 /
−3.6 / −4.5 / −3.1 / −2.5 / −4.9 / −6.4% at LET = 14 / 20 / 32.5 / 50 / 100 / 153.5
/ 250 / 339 keV/um, taking V79 from −4% to about −8 to −11%.

### Per-panel residuals: inherited from LEM I, per cell line
LEM I = `original_lem` line, LEM II = `modified_lem`; `ratio` = LEM II / LEM I
increment (cancels #1-#4 and #7 to first order):

| panel | alpha_x | R_n | LEM I | LEM II | ratio |
|---|---|---|---|---|---|
| Fig. 6 V79 | 0.093 | 3.8 | −4.0% | −6.5% | −2.7% |
| Fig. 8 V79 | 0.093 | 3.8 | −4.3% | −7.0% | −3.6% |
| Fig. 8 CHO | 0.228 | 4.7 | **−14.4%** | −7.7% | **+7.7%** |
| Fig. 11 HSG He | 0.313 | 5.0 | −1.8% | −2.9% | −1.2% |
| Fig. 11 HSG C | 0.313 | 5.0 | −5.6% | −4.3% | +0.8% |

LEM II adds `+0.2% ± 4.5%` on top of LEM I (`−1.7% ± 1.9%` without CHO): the
extension is very nearly right. Score LEM II defects on the ratio column, not the
raw bias. Quote the ratio spread, not `corr(LEM I, LEM II)` (0.65; 0.46 without
CHO), which five points and CHO's lever arm dominate.

The offset spans −14.4% to −1.8% across cell lines while nearly flat in LET within
a panel (unlike #1), and is uniformly negative. `R_n` scan on the LEM I lines:

| panel | 0.85x | 0.925x | **1.0x** | 1.075x | 1.15x |
|---|---|---|---|---|---|
| Fig. 6 V79 | −11.4% | −7.5% | **−4.0%** | −0.9% | +2.0% |
| Fig. 8 V79 | −11.7% | −7.8% | **−4.3%** | −1.0% | +1.8% |
| Fig. 8 CHO | −19.6% | −16.8% | **−14.4%** | −12.3% | −10.5% |
| Fig. 11 HSG He | −5.4% | −3.4% | **−1.8%** | −0.4% | +0.7% |
| Fig. 11 HSG C | −15.8% | −10.5% | **−5.6%** | −1.1% | +3.1% |

V79 closes at ~1.10–1.15x and HSG at ~1.10x (He) / ~1.13x (C): one ~1.1x
effective-size correction closes four of five panels. **CHO closes at no radius**
(its `s_max = 0.228 + 2·0.020·40 = 1.83` saturates the nucleus); it is the cell
line with the confirmed Fig. 3 caption error, so check its `D_t = 40` /
`alpha_x = 0.228` next, not the geometry.

### Todd 1985 (ref. 17): the effective size shrinks with fluence
Ref. (17) = Todd, Wood, Walker & Weiss, *Radiat. Res. Suppl.* 8, S5–S12 (1985),
Methods p. S-7:

```
S = exp(-sigma_bar f) = sum_i x_i exp(-sigma_i f)          (2)
sigma_bar = -ln[ sum_i x_i exp(-sigma_i f) ] / f           (3)
```

Eq. (3) was "also used to calculate the average nuclear cross section from
histograms" of nuclear area; for T-1E cells it reduced 200 to 120 um^2 "at a
typical alpha-particle fluence". That is 0.60 in area = 0.78 in radius; SK1996's
75 -> 50 um^2 is 0.67 = 0.82 in radius: the origin of the ~0.8
`CellTarget.effective_size_factor` (default 1.0; 0.8 belongs only to LEM III's
geometric Suzuki areas). Limits: `f -> inf` gives `min_i sigma_i`; `f -> 0` gives
`sum_i x_i sigma_i`, the arithmetic mean, so **the shrinkage vanishes at the
initial slope**, which is what every RBE_alpha panel plots. At alpha the right
operation is averaging `sigma_i = A_i (1 - S_1(A_i))` over the size distribution.
If the quoted radii are already Todd-reduced, the alpha-relevant size is larger
than quoted — consistent with V79 and HSG closing at ~1.1x.

### Genome size: strong RBE lever, no Fig. 2 lever
`L_gen` and `h` cancel from eta's peak height and only rescale the dose axis
(`eta(4D)` at `4x L_gen` equals `eta(D)` to 1e-9). Friedrich 2015's closed form gives
`eta_max = 13.51` for every `L_gen` from 2.7e9 to 1.2e10; our Monte-Carlo rule stays at
13.40-13.46 (sampling noise). Fig. 2 needs `max[q/mu] = 0.3310`; our rule caps at
0.2976 (default `L_gen` 6e9; 0.2983 at the fitted 8.6e9), Friedrich 2015's at 0.3003.
The best-fit `L_gen` against the whole Fig. 2 curve (7.9e9) only improves relRMS 21.6% -> 18.9%: a shape mismatch. But it moves
the RBE panels; bias of the LEM II / LEM I ratio:

| L_gen [bp] | Fig. 6 V79 | Fig. 8 V79 | Fig. 8 CHO | Fig. 11 He | Fig. 11 C |
|---|---|---|---|---|---|
| 3.0e9 | +19.1% | +18.0% | +32.0% | +29.7% | +21.6% |
| **6.0e9** | **−2.7%** | **−3.6%** | **+7.7%** | **−1.2%** | **+0.8%** |
| 1.2e10 | −17.9% | −18.8% | −9.5% | −19.0% | −14.5% |
| *zero at* | *5.4e9* | *5.2e9* | *7.9e9* | *5.8e9* | *6.2e9* |

Four panels want 5.2–6.2e9, CHO alone 7.9e9: keep 6.0e9 and carry ±4% as budget.
Fitting the raw bias instead launders the uniform LEM I offset through a cluster
parameter (every panel would want a smaller genome; HSG ~5.1e9).

### Fig. 2's eta peak is the outlier, not ours
Friedrich, Durante & Scholz 2015 (doi:10.1093/rpd/ncv147) recompute eta from the
same alpha_SSB = 1250, alpha_DSB = 30, t = 25 bp with a closed form and a stricter
coincidence rule (`Friedrich2015ClusterModel`): peak 13.51, on top of our 13.40,
8.6% below Fig. 2's 14.79, agreeing with ours to 3.0% RMS over the <= 1e5 Gy a
track reaches. The rules diverge only above ~1e6 Gy, where the 2015 paper says
its implementation is undetermined. Fig. 2 was digitized twice (hand vs raster
auto-trace, 0.9% RMS apart). Friedrich 2015 eq. (10),
`beta_DSB = alpha_SSB^2 t / (2 L_gen)`, gives alpha/beta = 8294 Gy against its
quoted ~8300.

## Paper defects
- **Fig. 3 caption `D_t = 30` Gy is an error**: it is CHO's panel, and the Fig. 8
  caption gives CHO "D_t = 40 Gy (old)". Panel b's 10 Gy is an illustration, not
  CHO's refit (9.5 Gy). `FIG3_D_T_OLD_GY`.
- **Fig. 2** is unreachable from the stated rule (above).
- **Eq. (8)'s `D_1`** prints `rho` in the numerator; it divides (Equations).
- **Fig. 11 y axis**: ticks 0.25 apart, labelled every 0.5 from 0.5 to 3.0; the
  lowest tick is an unlabelled 0.25 with frame below it. Anchoring it as 0 gives
  `dig = 1.0909 true − 0.2727`; all six Fig. 11 CSVs carry
  `alpha_true = alpha_dig * 2.75/3 + 0.25` (the `note` field records it), guarded
  by `test_fig11_y_axis_is_the_recalibrated_one`. Cross-checked against Furusawa
  2000's HSG carbon alphas in RadPhysBio. Check both ends of both axes: a
  scale-plus-origin slip leaves the top value nearly right (2.9 -> 2.891).

Recalibrated Fig. 11 scores (`lethal_central_saturating`, gamma = 0.062):

| Fig. 11 line | bias | relRMS |
|---|---|---|
| carbon original-LEM | −5.7% | 6.2% |
| carbon modified-LEM | −4.4% | 5.5% |
| helium original-LEM | −1.9% | 2.1% |
| helium modified-LEM | −3.0% | 3.1% |

(`validation/elsasser2007_lem2.py` on the default stopping power.) The paper's conceded sub-30 keV/um model-vs-data
mismatch (p. 327) shrinks from ~3x to ~2x above the lowest data points.

## Ruled out
- Digitization: bounded at 0.3–1.5% by two independent cross-traces (Fig. 6 vs 8,
  Fig. 6 vs 7).
- eta below D_t: the paper keeps plain LQ there ("for doses below D_t the values
  for eta(D) are negligible"), as we do.
- Absolute radial dose: Fig. 4 validates it against measurement, RMS(log10)
  0.010–0.052 over 7 decades. Fig. 4's "modified" line is the **un-convolved**
  profile (compare with `radical_diffusion=False`).
- For HSG: `alpha_x`, `D_t`, nuclear geometry and the impact-parameter
  cross-section (the former +12% offset was the Fig. 11 axis).

## Known limits
- Residual per-cell-line offset (CHO −14.4% on LEM I) unexplained; see budget.
- eta peak below the paper's Fig. 2 (13.40 vs 14.79); nearly RBE-neutral
  (Fig. 12 XRS check in the `benchmark/elsasser2007.py` docstring).
