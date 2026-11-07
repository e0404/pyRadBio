# Phenomenological LET-based RBE models: the RBEmin/RBEmax family

Covers `pyradbio/models/phenomenological/`: thirteen models on the shared
`RBEMinMaxModel` base (`WilkensOelfke`, `LinearScaling`, `Carabe`, `Wedenberg`,
`McNamara`, `MairaniHelium`, `Tilly2`, `Tilly10`, `Chen`, `Jones`,
`RorvikUnweighted`, `Peeler`, `Unkelbach`). Tests:
`tests/models/test_phenomenological_rbe.py`,
`tests/models/test_phenomenological_rorvik_group.py`.

## One-paragraph idea

Not mechanistic. Each model is a regression fit to in-vitro survival data that
predicts an ion's LQ parameters from the dose-averaged LET `LET_d` and the
tissue's photon ratio `(alpha/beta)_x`. They share one skeleton, introduced by
Carabe-Fernandez, Dale & Jones 2007 (doi:10.1080/09553000601087176): the ion
parameters are the photon ones scaled by the two asymptotic limits of the
RBE-vs-dose curve. A published model contributes exactly two closed forms
(`RBE_max`, `RBE_min`); survival, RBE and mixed-field handling come from the
shared LQ layer. Purpose here: a phenomenological baseline against which the
mechanistic families (LEM, MKM, ...) can be judged.

## Core equations

Skeleton (Carabe-Fernandez 2007 eqs. 1-3; McNamara 2015 eqs. 2-3):

```
RBE_max = alpha_ion / alpha_x          (D -> 0 limit)
RBE_min = sqrt(beta_ion / beta_x)      (D -> inf limit)
alpha_ion = RBE_max * alpha_x ,  beta_ion = RBE_min^2 * beta_x
```

RBE at a given **ion** dose `D` (McNamara 2015 eq. 2, `r = (alpha/beta)_x`):

```
RBE(D) = [ sqrt( r^2 + 4 r RBE_max D + 4 RBE_min^2 D^2 ) - r ] / (2 D)
```

`RBEMinMaxModel.rbe_at_dose` evaluates this closed form instead of inverting
`core.lq`: the survival route `S = exp(-aD - bD^2)` underflows above ~100 Gy and
cannot reach the `D -> inf` asymptote that defines `RBE_min`. The iso-survival
definition is `rbe_at_survival`; the two agree at `S = S(D)` (pinned).

**A third, non-interchangeable parameterisation.** Carabe-Fernandez 2007 eq. (5)
gives RBE as a function of the **reference (photon)** dose per fraction `d_L`
(`rbe_at_reference_dose_from_let`):

```
RBE(d_L) = [ r RBE_max + sqrt( r^2 RBE_max^2 + 4 d_L RBE_min^2 (r + d_L) ) ]
           / (2 (r + d_L))
```

It is a different function of a different argument from `rbe_at_dose_from_let`;
they agree only on matching pairs `d_L = RBE * D_ion` (pinned). All three share
the same asymptotes.

## Parameters

`L = LET_d` [keV/um], `r = (alpha/beta)_x` [Gy], `alpha_x` [Gy^-1],
`lambda` [um keV^-1 Gy^-1].

| Class | Paper (DOI) | Ion | `RBE_max` | `RBE_min` |
|---|---|---|---|---|
| `WilkensOelfke` | Wilkens & Oelfke 2004 (10.1088/0031-9155/49/13/004) | p | `(a0 + lambda L)/alpha_x`, `a0=0.1`, `lambda=0.02` | 1 |
| `LinearScaling` | Frese et al. 2011 (10.1016/j.ijrobp.2009.10.022) | p | `(a0 + lambda L)/alpha_x`, `a0=alpha_x-0.5 lambda`, `lambda=0.008` | 1 |
| `Carabe` | Carabe et al. 2012 (10.1088/0031-9155/57/5/1159) | p | `0.843 + 0.154*2.686 L/r` | `1.09 + 0.006*2.686 L/r` |
| `Wedenberg` | Wedenberg et al. 2013 (10.3109/0284186X.2012.705892) | p | `1 + q L/r`, `q=0.434` | 1 |
| `McNamara` | McNamara et al. 2015 (10.1088/0031-9155/60/21/8399) | p | `p0 + p1 L/r` | `p2 + p3 sqrt(r) L` |
| `MairaniHelium` | Mairani et al. 2016 (10.1088/0031-9155/61/2/888) | He | `1 + (p0 + 1/r) p1 L^2 e^(-p2 L)` | 1 |
| `Tilly2` | Tilly et al. 2005, PMB 50:2765 (10.1088/0031-9155/50/12/003) | p | `1 + 0.309 L/r` | 1 |
| `Tilly10` | same paper | p | `1 + 0.550964 L/r` | 1 |
| `Chen` | Chen & Ahmad 2012, Radiat. Prot. Dosim. 149:116 (10.1093/rpd/ncr218) | p | see below | 1 |
| `Jones` | Jones 2015b, Cancers 7:460 (10.3390/cancers7010460) | p | `1 + s(alpha_U/alpha_x - 1)` | `sqrt(1 + s(beta_U/beta_x - 1))` |
| `RorvikUnweighted` | Rorvik et al. 2017, Med. Phys. 44:2586 (10.1002/mp.12216) | p | `1 + 0.645 L/r` | 1 |
| `Peeler` | Peeler 2016, PhD dissertation (**no DOI**) | p | `0.75 + 0.00143 L^3/r` | `1.24 + 0.00074 r L^3` |
| `Unkelbach` | Unkelbach et al. 2016, IJROBP 96:1097 (10.1016/j.ijrobp.2016.08.038) | p | `1 + 0.04 L` | `= RBE_max` |

Coefficients:

- McNamara (sec. 3.1, with SE): `p0 = 0.99064` (0.014125), `p1 = 0.35605`
  (0.015038), `p2 = 1.1012` (0.0059972), `p3 = -0.0038703` (0.00091303), R^2 = 0.255.
- Mairani (table 4, **fQE** row): `k0 = 1.36938e-1`, `k1 = 9.73154e-3`,
  `k2 = 1.51998e-2`. fQE is one of ten weighting functions fitted there and the
  one selected (lowest RMSE 1.665, jackknife 24.9). Rorvik instead uses the
  revised fLE2: never mix coefficients across weighting functions.
- Wedenberg (eq. 2, Results): `q = 0.434`, 95% CI (0.366, 0.513); 10 cell lines,
  24 RBE values, LET 6-30 keV/um. `RBE_min = 1` is supported (beta-LET slope CI
  -0.01 to 0.05 includes zero).
- Chen (eqs. 8-9): `alpha_p(L) = alpha_0 + [1 - exp(-lambda_1 L^2)]/(lambda_2 L)`,
  `alpha_0 = 0.1`, `beta_p = beta_x`; so `RBE_max = alpha_p/alpha_x`. `alpha_0` is a
  V79 ion-response intercept, not the tissue `alpha_x`.
- Jones (eqs. 1-2): `alpha_U = (Q/j)(1 - exp(-j alpha_x))`, `Q = 10.62`,
  `j = 3.92 Gy`; `beta_U = (R/u)(1 - exp(-u beta_x))`, `R = 2.5`, `u = 25 Gy^2`;
  `s = (L - 0.22)/30.28` (paper's turnover LET 30.5 keV/um).
- Wilkens & Oelfke: `alpha_0 = 0.1` (+-0.02), `lambda = 0.02` (+-0.002), eq. (4)
  and fig. 1(a); eq. (5) defines `beta_p := beta_x`; V79/Co-60 reference
  `alpha_x = 0.112`, `beta_x = 0.0298` (sec. 2.2.3, Tilly et al. 1999).
- Frese 2011: eq. (6) `alpha_0 = alpha_x - 0.5 keV/um * lambda`, eq. (8) for
  `lambda`; table 3 prints `alpha_x`/`beta_x`/`alpha_0` for 17 tissues, table 4 the
  four per-patient `lambda_1.1` (0.0074/0.0070/0.0086/0.0091, mean 0.008, ~+-13%).
- Unkelbach: `c = 0.04 um/keV`, calibrated to RBE 1.1 at the SOBP mean
  `LET_d = 2.5 keV/um`. The paper's alternatives are `c ~ 0.02` for
  `(alpha/beta)_x = 10 Gy` and `c ~ 0.05` for 2 Gy (linearising McNamara at
  1.67 Gy/fraction); `slope_um_keV` takes any of them.

`WilkensOelfke` and `LinearScaling` are one functional form (Frese eq. 3 = Wilkens
eq. 4) with different `(a0, lambda)` calibration; the linear term is in `alpha`,
hence the explicit `1/alpha_x` in `RBE_max`. `Wedenberg`, `Tilly2`, `Tilly10` and
`RorvikUnweighted` are one form `1 + qL/r`, `RBE_min = 1`, with `q` = 0.434 / 0.309
/ 0.550964 / 0.645: a 2.1x spread of fitted slope for the same equation, the
package's cleanest measure of database choice versus model choice. `Unkelbach` is
the only member with `RBE_max = RBE_min` (dose- and tissue-independent RBE); it is
the RBE implied by an LET-optimisation method, not a survival fit, and its eq. (1)
is a pure exponential with no quadratic term, which is why the equality is exact.

### Provenance tiers

**Tier 1 (verified against the primary paper):** Wilkens & Oelfke, Frese 2011,
Wedenberg, McNamara, Mairani. The Wilkens paper prints both RBE closed forms,
eq. (6) at fixed dose and eq. (7) at fixed survival, so a mix-up of the two RBE
definitions cannot pass. McNamara's equations are images in the PMC manuscript:
coefficients and prose were read, the typeset equations were not.

**`Carabe` (not read from the primary, treated as sufficient).** Coefficients
reproduced term-for-term (incl. the factored `0.154*2.686`) in Rorvik et al. 2018
appendix (doi:10.1088/1361-6560/aad9db); the form is described in McNamara 2015
sec. 2.3; the formalism is Carabe-Fernandez 2007. McNamara's reference list
resolves "Carabe et al 2012" to this (range-uncertainty) paper. Wedenberg's
Discussion states Carabe fitted **V79 only**, so the `1/(alpha/beta)_x` scaling is
an extrapolation away from one cell line, the likely reason it separates from the
others far from `(alpha/beta)_x ~ 5 Gy`. The paper would add only its own
uncertainty estimates.

**The seven added from Rorvik's appendix (`Tilly2`, `Tilly10`, `Chen`, `Jones`,
`RorvikUnweighted`, `Peeler`, `Unkelbach`): built from the appendix, then checked
against all six primaries; now tier 1 except the Tilly slopes.** Rorvik et al. 2018 appendix B restates eleven models as an
`RBE_max/RBE_min` pair and prints no DOIs; all DOIs above were resolved against
Crossref. Appendix B reproduces our five independently-read models (WIL, FRE,
WED, CAR, MCN) exactly (`test_rorvik_appendix_reproduces_our_independently_read_models`),
but five-for-five did not generalise: reading the primaries changed four of the
seven (below). A validation set of five bounds the risk for a sixth case; it does
not remove it.

#### What the primaries changed

- **Chen: the appendix is wrong by a factor of 10.** It writes
  `RBE_max = 1 + [1 - exp(-0.0013 L^2)]/(0.045 L)`, adding the LET term to 1
  without dividing by `alpha_x` (under its own `alpha_x = alpha_0 = 0.1` the
  constant would be 0.0045). On the paper's fig.-6 case (V79, `alpha_x = 0.1`,
  `beta_x = 0.03`) at `LET_d = 30`: eq. (8) gives `alpha_p = 0.611`, `RBE_10 = 2.23`;
  the appendix gives `alpha_p = 0.151`, `RBE_10 = 1.10`, at a LET where fig. 6 plots
  measured V79 RBE well above 2. Pinned by
  `test_chen_primary_disagrees_with_the_rorvik_reformulation`. Chen therefore keys
  on `alpha_x` (like `WilkensOelfke`, `LinearScaling`, `Jones`).
- **Jones: both saturation constants differ.** Paper `Q/j = 2.7092` (appendix
  2.696, 0.5% apart); `beta_U`: paper `R/u = 0.1`, `u = 25` (the paper states a
  "maximum ceiling value of 0.1 Gy^-2"), appendix `0.06`, `u = 50` (a model
  disagreement, not rounding). The appendix prints `3.92 Gy^-1` and `50 Gy^-2`;
  those units are impossible (the exponent must be dimensionless). Eq. (1)
  integrates `d(alpha_U)/d(alpha_L) = Q - j alpha_U`, so `Q` is dimensionless and
  `j`, `u` carry Gy, Gy^2. The appendix's denominator 30.28 = 30.5 - 0.22 is
  confirmed.
- **Peeler: our own transposition.** We had read Rorvik's table A.1 `(alpha/beta)_x`
  and `LET_d` range columns the wrong way round. Correct (dissertation tables 3-4):
  `LET_d` 0.9-19.0 keV/um (12 measured values 0.94-19.4), `(alpha/beta)_x`
  2.6-5.1 Gy (H1437 0.08/0.031 = 2.6; H460 0.36/0.071 = 5.1). Eqs. (30)-(31)
  confirm all four coefficients; the cell lines are H460 and H1437 (human lung).
  The cube is deliberate: linear models "agree well ... up to approximately
  10 keV/um" and the data keep rising.
- **Unkelbach: the "McMahon" variant does not exist.** Rorvik's "McMahon
  configuration ... 0.055" has no counterpart in the paper (neither value nor
  name); the `mcmahon()` preset was removed.
- **Tilly2/Tilly10 stay second-hand.** Tilly et al. 2005 says only that
  `alpha/alpha_ref` "were empirically fitted as linear functions of the LET where
  the slope is determined by the `alpha_ref/beta_ref` ratio, as shown in figure 2";
  it prints no `q`. `0.309` and `0.550964` are Rorvik-only (presumably read off
  fig. 2). The paper does supply the basis for `RBE_min = 1` ("beta values were
  assumed to be equal to the low-LET reference value up to a LET of 20 eV/nm") and
  the cell lines per band (V79-379A 2.7 Gy, V79-753B 2.8 Gy; SQ20B 7.6 Gy,
  C3H10T1/2 15 Gy; Rorvik's 7.7 is 7.6 in the paper).
- **RorvikUnweighted: `q = 0.645` confirmed**, with an internal inconsistency in the
  source: the 2017 discussion says 0.635, table A.2 and appendix A.4 print 0.645.
  0.645 is right: for a linear weighting function the spectrum reduction is exact
  (`int L d(L) dL` is `LET_d` by definition).

#### Source defects (also for the upstream authors)

- Rorvik 2018 appendix B: Chen factor 10; Jones `beta_U` constants (0.06/50 vs
  0.1/25) and impossible exponent units; Unkelbach "McMahon 0.055" unsourceable
  (details above).
- Rorvik 2017: 0.635 vs 0.645 (above).
- matRad (code, not paper): `matRad_MCNamara.m` carries `p0 = 0.999064` (a constant
  +0.008424 offset in `RBE_max`; the paper and Rorvik's appendix both give 0.99064)
  and `matRad_LinearScalingModel.m` has a dead clamp.

### Citation traps

- `Jones` is Jones **2015b** (*Cancers*), not 2015a (*Front. Oncol.*): Rorvik's
  table 1 credits JON to 2015b.
- `Peeler` is the **2016 PhD dissertation**, not Peeler, Mirkovic, Titt et al. 2016
  (Radiother. Oncol. 121:395), a clinical-outcome study that fits no
  `RBE_max/RBE_min`. The dissertation has no DOI, hence `RBEMinMaxModel.reference_citation`
  (a title must not go in a field named `reference_doi`).
- Rorvik 2018 prints no DOIs; four of the seven first-written citations were wrong
  (two about the paper itself, Jones and Peeler above; the others a wrong DOI or,
  for Chen, wrong year/volume/pages). A DOI not resolved against Crossref is a
  guess.

## Validity domains

Encoded as `valid_let_keV_um`, `valid_alpha_beta_ratio_Gy`. Outside the box the
model **warns and returns the extrapolated value**; it never clamps.

- **McNamara**: `LET_d < 20 keV/um`, `(alpha/beta)_x < 30 Gy`, proton doses 1-10 Gy
  (sec. 2.1).
- **Wilkens & Oelfke** (and **LinearScaling**, same form): `LET_d` 0.3-30 keV/um
  (sec. 2.1). Above ~30 keV/um fig. 1(a) shows `alpha_p` decreasing, so the linear
  form has the wrong sign of slope.
- **Wedenberg**: `LET_d` up to 30 keV/um ("Equation 1 is not valid for LETs higher
  than 30 keV/um").
- **Peeler**: `LET_d` 0.9-19.0 keV/um, `(alpha/beta)_x` 2.6-5.1 Gy.
- **Tilly2 / Tilly10**: `(alpha/beta)_x` 2.7-2.8 Gy and 7.7-15 Gy (two classes
  because a single class carries only one domain).
- Rorvik-sourced models take domains from table A.1; `Chen` declares its V79
  range (2.2-2.8 Gy); `Unkelbach` declares none (no database behind it).
- `Carabe` and `MairaniHelium` state none we have read; their `valid_*` are `None`
  rather than an invented box.

## Known limits / failure modes

- `Carabe` does not reduce to the photon reference at zero LET (`RBE_max(0) = 0.843`);
  `WilkensOelfke` likewise (`0.1/0.112 = 0.89` with V79 defaults). So "RBE -> 1 as
  LET -> 0" is not a family-wide criterion (tested for `Wedenberg`,
  `MairaniHelium` only). `LinearScaling` hits RBE = 1 at 0.5 keV/um (its anchor).
- `Carabe` can predict RBE **rising** with dose (`RBE_min` exceeds `RBE_max` at
  high `(alpha/beta)_x`, low LET; McNamara 2015 sec. 3.2). Nothing in the base class
  may assume `RBE_min <= RBE_max`.
- `McNamara`'s `RBE_min` goes negative far out of domain, at
  `LET_d = p2/(|p3| sqrt(r))`: 164 keV/um at `r = 3`, 64 keV/um at `r = 20`. Since
  `beta = RBE_min^2 beta_x` squares the sign away, an unguarded code would predict a
  rising beta; `RBEMinMax` rejects a negative limit.
- `MairaniHelium` is the only one with a high-LET turnover (`L^2 e^(-p2 L)` peaks at
  `2/p2 ~ 132 keV/um`).
- `radiation_modes` is enforced: `MairaniHelium` on a proton (or `Wedenberg` on
  helium) raises.
- `LinearScaling`'s `lambda = 0.008` is a planning constant (head-and-neck IMPT,
  2 Gy/fraction; "has to be adjusted for other fractionation doses to keep the
  mean RBE at 1.1"). Use `lambda_for_target_rbe()` to recalibrate. LET sensitivity
  scales as `lambda/alpha_x`, so radioresistant tissues respond more.
- **Not implemented, deliberately:** Belli et al. 1997 (BEL, weighting functions
  integrated over an energy spectrum) and Rorvik 2017 weighted (RORW, quartic
  weighting over a LET spectrum). Neither is a function of a scalar `LET_d`;
  evaluating either at `LET_d` would be a different model under the published name.
  (`MairaniHelium` is the pre-existing exception: its weighting function is
  evaluated at `LET_d`.)

## Mixed fields: ordering rule

A field is reduced to a single `LET_d` first and the model is evaluated once on
it, never per component then averaged in `alpha` (they differ whenever the model
is non-linear in LET: `McNamara` via `RBE_min`, `MairaniHelium` via `f_QE`; a test
pins a case where they disagree). A binned component contributes its
bin-integrated LET, `int phi S^2 dE / int phi S dE` on a 16-node log sub-grid
(`let_d_keV_um(field, sub_nodes=...)`), not the LET at its representative energy.

## Golden values for tests

- **Wilkens & Oelfke eqs. (6) and (7)**: RBE at fixed dose and at fixed survival,
  over (LET, dose) and (LET, SF) grids against the base class's generic route.
- **Carabe-Fernandez 2007 eq. (5) and worked eq. (14)**:
  `RBE = [49.58 + sqrt(2458.69 + 336.48 d_L + 20.71 d_L^2)]/(32.49 + 2 d_L)`
  (mouse oesophagus) at 1%; the printed constants are rounded (20.71 implies
  `RBE_min = 2.2760`, not the tabulated 2.27).
- **Frese 2011 tables 3 and 4**: `alpha_0` for nine tissues (eq. 6) and the four
  `lambda_1.1` values (eq. 8) to printed precision; `lambda_for_target_rbe`
  round-trips (calibrate, evaluate, 1.1 comes back).
- Smoke test: at `LET_d = 2.5 keV/um` (McNamara 2015 sec. 3.2 SOBP plateau mean),
  2 Gy, `(alpha/beta)_x = 3 Gy`, every proton model lands in `1.0 < RBE < 1.25`.
- Chen primary-vs-appendix disagreement (above).

## Implementation notes / gotchas

- A constant offset of 0.008424 in `RBE_max` against another code is the McNamara
  `p0` digit (0.999064 vs 0.99064); a wrong slope or form would instead vary with
  LET and `(alpha/beta)_x`.
- **A PIDE comparison validates our implementation, not the model.** McNamara 2015
  is fitted to substantially all published in-vitro proton data, which is what PIDE
  aggregates; Wedenberg and Carabe are fitted to subsets (McNamara sec. 2.3). State
  this wherever such a comparison is reported.
- Provenance is reported on every benchmark run for the Tier-2 models.
