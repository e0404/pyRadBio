# IMK — integrated microdosimetric-kinetic model — Matsuya et al. 2025 (doi:10.1002/mp.70040)

Matsuya Y, Saga R, Wang Y, Sato T, "Equivalent relative biological effectiveness
for cell survival and micronuclei formation: insights from a biophysical
approach", *Med. Phys.* 52(10):e70040 (2025), open access. The only IMK paper
printing the ion form of the survival equation with a complete parameter table.
The acronym (*integrated* MK) is coined in Matsuya et al. 2018 *Sci. Rep.*
**8**:4849; this paper cites 2018a/b (refs 20, 21) for the model and 2015 *JRR*
**56**:90 / 2017 *Med. Phys.* **44**:5522 (refs 26, 27, 24, 25) for the repair
kinetics.

Paper set (Matsuya et al. 2020 *PMB* groups its refs 30-33 as "the IMK model"):

| role | reference |
| --- | --- |
| ion form + printed parameter table | Matsuya et al. 2025, *Med. Phys.* 52:e70040, doi:10.1002/mp.70040 |
| coins "IMK"; targeted + non-targeted effects | Matsuya et al. 2018, *Sci. Rep.* 8:4849, doi:10.1038/s41598-018-23202-y |
| dose rate + cell-cycle distribution | Matsuya et al. 2018, *Sci. Rep.* 8:8287, doi:10.1038/s41598-018-26556-5 |
| repair kinetics / `F` derived | Matsuya et al. 2015, *J. Radiat. Res.* 56:90, doi:10.1093/jrr/rru090 |
| LET/quality dependence via `y*`; mixed field | Matsuya et al. 2020, *Cells* 9:1117, doi:10.3390/cells9051117 |
| oxygen | Matsuya et al. 2020, *Phys. Med. Biol.* 65:095006, doi:10.1088/1361-6560/ab7d14 |
| how `a+c` is fitted (MCMC) | Matsuya, Kimura & Date 2017, *Med. Phys.* 44:5522, doi:10.1002/mp.12508 |
| MK precursor, DNA-damage yields | Matsuya et al. 2014, *J. Radiat. Res.* 55:484, doi:10.1093/jrr/rrt222 |

## One-paragraph idea

The Hawkins MK model plus two pieces. (i) Quality enters as in the **modified
MKM**: `alpha = alpha0 + z*_1D beta0`, `z*_1D = y*/(rho pi r_d^2)` with the Kase
saturation `y*` (sec. 2.2, eqs. 1-2). (ii) The quadratic term is multiplied by
the **Lea-Catcheside time factor** `F`, written in the DSB repair-rate constant
`(a+c)` (eq. 1). Underneath, potentially lethal lesions (PLLs, read as DSBs)
become lethal lesions (LLs) at rates `a` (single) and `b_d` (pairwise) or are
repaired at rate `c`. New in this paper, assumption (iv): an LL forms a
**micronucleus** with probability `h`, so `F_MN = h <w>` (sec. 2.3, eq. 3). In
code, `alpha` is `ModifiedMKM.alpha_beta_from_z_star_1D` and `F` is
`core.protraction.lea_catcheside_g` of one continuous block.

## Inputs

| symbol | meaning | unit | source |
| --- | --- | --- | --- |
| `D`, `Ddot`, `T` | dose, dose rate, delivery time | Gy, Gy/h, h | eq. (1) |
| `y`, `f(y)` | lineal energy; its **frequency** density | keV/µm, 1/(keV/µm) | eq. (2) |
| `y0` | saturation parameter | keV/µm | eq. (2); **150** for HSG/SCC, sec. 4.1 |
| `r_d` | domain radius | µm | table 1 |
| `rho` | domain density | g/cm³ | eq. (1); water, 1.0 |
| `alpha0`, `beta0` | cell-specific LL coefficients | Gy⁻¹, Gy⁻² | table 1 |
| `(a+c)` | PLL→LL + PLL-repair rates, summed | h⁻¹ | table 1 |
| `h` | probability an LL forms an MN | – | tables 1-2 |

The paper scores `y*` with the PHITS `[t-sed]` tally (sec. 3.1.1).

## Core equations

Survival, eq. (1):

    -ln S = <w> = (alpha0 + y*/(rho pi r_d^2) beta0) Ddot T
                  + 2 beta0 / ((a+c)^2 T^2) [(a+c) T + e^-((a+c)T) - 1] (Ddot T)^2
           = (alpha0 + z*_1D beta0) D + F beta0 D^2

`<w>` is the mean number of LLs per nucleus. With `x = (a+c) T`,
`F = 2 (x - 1 + e^-x)/x^2`: exactly `lea_catcheside_g` of one block of duration
`T` with `lam = a+c`. The paper names it the Lea-Catcheside factor (end of sec.
2.2) and only writes the single-block case; its own dose-rate experiment is
fractionated (sec. 3.1.4), so use the general `DoseSchedule` route.

Saturation, eq. (2):

    y* = y0^2 integral [1 - exp(-y^2/y0^2)] f(y) dy / integral y f(y) dy

identical to `_SpectrumMKModel.y_star_D` written on `d(y) = y f(y)/y_F`.

Micronuclei, eq. (3):

    F_MN = h <w> = (alpha_m0 + z*_1D beta_m0) D + F beta_m0 D^2

with `alpha_m0 = h alpha0`, `beta_m0 = h beta0`. In rates,
`alpha_m0 = h a k_d <G> / (a+c)`, `beta_m0 = h b_d k_d^2 <G>^2 / [2 p (a+c)]`
(`k_d` PLL yield per DNA per Gy, `<G>` DNA per nucleus, `p` domains per
nucleus); these are not separately identifiable, hence `alpha0`/`beta0`.

RBE, eqs. (4)-(6):

    D10    = (1/(2 beta)) [ sqrt(alpha^2 + 4 beta S*) - alpha ],  S* = -ln(0.1)
    RBE_SF = D10(photon) / D10*
    RBE_MN = alpha_m* / alpha_m(photon) = alpha* / alpha(photon)

- `h` cancels in `RBE_MN`: `RBE_MN == RBE_alpha`, the point of the study.
- Eq. (6) is the `D -> 0` limit ("the maximum RBE value"), not an RBE at a dose.
- Eq. (4) assumes acute irradiation, `beta ~= beta0` (sec. 2.4). A protracted
  `beta = F beta0` in `RBE_SF` is a generalisation the paper does not score.

## Parameters

**Table 1** (sec. 4.1), printed with MCMC 1-sigma uncertainties:

| parameter | HSG | SCC (VII) | WI-38 | DU145 | unit |
| --- | --- | --- | --- | --- | --- |
| `alpha0` | 1.50e-1 ± 6.87e-2 | 8.25e-2 ± 4.34e-2 | 5.90e-1 ± 1.07e-1 | 7.74e-2 ± 3.64e-2 | Gy⁻¹ |
| `beta0` | 4.67e-2 ± 7.21e-3 | 2.98e-2 ± 3.39e-3 | 1.63e-2 ± 1.32e-2 | 3.80e-2 ± 4.98e-3 | Gy⁻² |
| `a+c` | 2.19 ± 4.00e-1 | 2.19 ± 3.92e-1 | 3.71e-1 ± 3.84e-2 | 2.10 ± 9.11e-1 | h⁻¹ |
| `r_d` | 4.20e-1 | 4.20e-1 | 5.00e-1 | 5.00e-1 | µm |
| `h` | 2.75e-1 ± 6.18e-3 | (same) | (same) | (same) | – |

`y0 = 150 keV/µm` and `r_d = 0.42 µm` for HSG and SCC are stated in sec. 4.1
prose. `h = 0.275` is the cell-line-independent fit of fig. 3, used for all four
lines in the predictions.

**Table 2** (sec. 4.1): per-cell-line `h`, fitted separately — CHO
3.65e-1 ± 3.17e-2, HSG 2.45e-1 ± 8.68e-3, U2OS 2.61e-1 ± 3.87e-2, MCF-12A
2.29e-1 ± 3.72e-2, M5 1.96e-1 ± 4.94e-2, V79 1.97e-1 ± 5.70e-2, SCC VII
2.81e-1 ± 7.47e-3 (radiations per fit: `models/mkm/imk.py`,
`_H_MATSUYA2025_TABLE2`). HSG differs between tables (0.275 vs 0.245) by
construction; never mix them in one calculation.

## Known limits / failure modes

- **`y*` is PHITS-scored**, and the experimental beams (depths in a 6-cm SOBP,
  various energies) are approximated by mono-energetic simulations even by the
  authors (sec. 4.2). Our reconstructions run on **our** `d(y)`; state the
  offset, do not tune it away.
- **The photon `y*` is never printed**; it is recovered from fig. 2 (below).
- `r_d` is set by matching the carbon LET dependence (sec. 3.1.1 step iii), so
  `r_d` and `y0` carry the LET calibration and cannot be reused from another
  MKM-family fit of the same cell line (same trap as SMK/OSMK, see
  `OSMK_Inaniwa2021_2023.md` note 1).
- DU145 `alpha0` and WI-38 `beta0` have ~50-80% relative uncertainty.
- Eq. (1) is deterministic: no nucleus-level stochastics (that is SMK), no
  overkill beyond the `y*` saturation; classic MKM high-LET behaviour.

## What fig. 2 gives back: the paper's own `y*`

Fig. 2's *calculated* curves are plain LQ; fitting one and inverting eq. (1) with
the printed `alpha0`/`beta0` gives `z*_1D = (alpha_fitted - alpha0)/beta0`. Both
panels hand-digitized (raster) in `benchmark/matsuya2025.py`, fitted over the
full traced range:

| panel | curve | alpha | beta | z*_1D [Gy] | y* [keV/µm] |
| --- | --- | --- | --- | --- | --- |
| 2a HSG | X-rays 200 kVp | 0.1988 | 0.0466 | **1.045** | **3.62** |
| 2a HSG | C 13 keV/µm | 0.2865 | 0.0465 | 2.923 | 10.11 |
| 2a HSG | C 100 keV/µm | 0.9058 | 0.0467 | 16.184 | 55.98 |
| 2b SCC | X-rays | 0.1119 | 0.0299 | **0.987** | **3.41** |
| 2b SCC | C 46 keV/µm | 0.2913 | 0.0298 | 7.007 | 24.24 |
| 2b SCC | C 80 keV/µm | 0.4051 | 0.0298 | 10.825 | 37.45 |

Checks: every fitted `beta` is the printed `beta0` within 0.5% (y-axis
calibration and `F = 1`); the two independent panels agree on the photon `y*` to
5.6%; ~3.5 keV/µm is physically sensible for kV X-rays at a 0.84 µm site.

**Trap: this paper's photon fit is not Furusawa's.** Its X-ray points give
`alpha = 0.197`, `beta = 0.0454`; Furusawa et al. 2000 report `alpha_x = 0.313`,
`beta_x = 0.061` for HSG at 200 kVp (factor 1.6). Neither Furusawa nor our mMKM
HSG defaults may stand in as IMK's photon arm.

**The four carbon anchors are not one curve in LET.** `y*` runs 10.1 -> 24.4 ->
37.4 -> 56.0 keV/µm over LET 13 -> 46 -> 80 -> 100; the last step is a log-log
slope of 1.76, impossible for a saturating `y*(LET)`. The panels are different
beams (HSG's carbon is a 290 MeV/n SOBP whose fragments lower `y*` at matched
LET), so `y*` is beam-specific; no microdosimetry route closes this.

Our routes against the anchors at LET 13/46/80/100: impact/cylinder
−30/+15/+31/+8%, impact/sphere −6/+39/+47/+15%, Sato-AMF +8/+40/+53/+22%,
analytic +21/+56/+52/+15% (`validation/imk_matsuya2025.py`).

**The paper's route is `sato-AMF`:** the domain is a **sphere** (sec. 2.1) and
`y*` comes from PHITS `[t-sed]` (sec. 3.1.1), which evaluates Sato's AMF
(`micro.sato.SatoLinealEnergyEngine`, spherical). The MKM-family default
(`site_geometry="cylinder"`) is Inaniwa's disk convention, right for mMKM/SMK and
wrong here. The faithful route is not the closest-scoring one (disk +6% vs +31%
on average, with the wrong LET slope); do not pick a route by its residual.

## What fig. 4 settles: `h = 0.275`, and a second reading of `y*`

Fig. 4: HSG micronuclei for the fig. 2a qualities, linear axes, symmetric error
bars, table-1 parameters in the legend; digitized in `benchmark/matsuya2025.py`.
By eq. (3) it is a zero-free-parameter prediction (`h ×` fig. 2a), and
`h = beta_m/beta0` reads off the quadratic term without `z*_1D`:

| curve | alpha_m | beta_m | h = beta_m/beta0 | z*_1D from fig. 4 | z*_1D from fig. 2a |
| --- | --- | --- | --- | --- | --- |
| X-rays 200 kVp | 0.0545 | 0.01283 | 0.2748 | 1.039 | 1.045 |
| C 13 keV/µm | 0.0784 | 0.01282 | 0.2746 | 2.905 | 2.923 |
| C 100 keV/µm | 0.2499 | 0.01273 | 0.2726 | 16.420 | 16.184 |

- **Table 1's `h`, not table 2's:** all three land on 0.275 (RMS residual
  0.6-0.7% pointwise); table 2's HSG 0.245 is 11% off. `IMKModel` defaults
  `mn_probability = 0.275`.
- **Second recovery of `z*_1D`:** 0.6 / 0.6 / 1.5% from fig. 2a, from a different
  panel, quantity and axis scaling. The photon `z*_1D = 1.045 Gy` is therefore a
  measured quantity.
- Fig. 4's LET-100 points sit on their own line (RMS relative residual 0.11 vs
  1.6 and 2.4 for the others), so this panel is not swapped; its X-ray and LET-13
  lines differ by only ~30%, so it cannot decide a swap of that pair.
- **Digitization floor:** ~0.0011 MN/BN-cell per pixel; the three lowest points
  sit at 1, 2 and 4 pixels. `FIG4_MIN_FIT_DOSE_GY = 0.05` excludes them from fits
  (they stay in the CSV).

## What fig. 7 validates: the time factor, on one panel out of two

Fig. 7: MN dose-rate dependence for DU145 (a) and WI-38 (b), 4 Gy, 1-190 Gy/h,
model mean, 95% CI and four measured points each; digitized in
`benchmark/matsuya2025.py` (log x traced as the exponent, rate = `10**x`).

- **Its y is relative**, normalised at 60 Gy/h (sec. 3.1.3); the digitized point
  there reads 1.0002 (DU145) / 0.9979 (WI-38). Reading it as an absolute MN count
  is the trap.
- **Delivery is continuous** (the line is smooth in rate); `F` differs from the
  fractionated protocol by < 0.1% at the four measured rates.
- **DU145 is a sharp test of `lea_catcheside_g`.** The relative frequency is
  affine in `F`, so `dose_rate_shape_fit` fits `A + B F(rate)`: max residual
  **0.0019** (0.22% of the 0.45 span, 58-point curve) at the printed
  `a+c = 2.10 /h`; 0.048 at half and 0.053 at double that rate (25-fold worse).
  The forward prediction from printed parameters (no fit; `h` cancels) lies inside
  the paper's 95% CI at every node, worst deviation 0.015 from the mean line.
- **WI-38 is not a second validation.** Its effect is 0.028 in y; the same fit
  gives 0.0041 / 0.0039 / 0.0042 at the printed rate, half and double. It confirms
  "no dose-rate dependence" only; pinned so it is not counted twice.
- **Neither panel yields a photon reference** (`alpha` nearly cancels in a ratio):
  a DU145 fit of the photon `z*_1D` gives 0.736 Gy but the RMS only doubles outside
  [0.255, 1.299] Gy; WI-38 bottoms out at 0.0 and is nearly as good at 11 Gy. So
  DU145 and WI-38 have no `reference_z_star_1D_Gy`.

## Golden values for tests

| claim | value | paper location |
| --- | --- | --- |
| `F -> 1` as `T -> 0` | 1.0 | eq. (1), acute limit |
| `F = 2(x-1+e^-x)/x^2`, `x = (a+c)T` | `x = 1` → 0.735758882342885 | eq. (1) |
| `RBE_MN` independent of `h` | exact | eq. (6) |
| `F_MN / (-ln S) = h` at any dose | 0.275 (table 1) | eq. (3) |
| `h` from fig. 4's three curves | 0.2748 / 0.2746 / 0.2726 | fig. 4, inverted |
| fig. 4 re-recovers fig. 2a's `z*_1D` | within 0.6 / 0.6 / 1.5% | figs. 2a + 4 |
| fig. 7a shape residual at printed `a+c` | 0.0019 (0.22% of span) | fig. 7a, `A + B F` |
| ... at half / double that rate | 0.048 / 0.053 | fig. 7a |
| fig. 7 relative MN at 60 Gy/h | 1 by construction | sec. 3.1.3 |
| `D10` = LQ inversion at `S = 0.1` | — | eq. (4) |
| HSG `alpha(z*_1D -> 0) = alpha0` | 0.150 Gy⁻¹ | table 1 |
| HSG photon `z*_1D` from fig. 2a | 1.045 Gy | fig. 2a, inverted |
| SCC photon `z*_1D` from fig. 2b | 0.987 Gy | fig. 2b, inverted |
| every fig.-2 calculated `beta` | printed `beta0`, ±0.5% | fig. 2 |

## Implementation notes / gotchas

- `models/mkm/imk.py::IMKModel` subclasses `ModifiedMKM`: `alpha` inherited,
  only `beta` overridden. If `ModifiedMKM`'s `alpha` moves, IMK moves with it.
- **Rate vs half-time.** `T_half = ln2/(a+c)`; HSG `a+c = 2.19 h⁻¹` is
  `T_half = 0.317 h = 19 min` (within the ~0.2-5 h literature span of repair
  half-times). `protraction.repair_half_time_h_from_rate` converts at the
  boundary; `protraction.py` stays half-time-native.
- `y0 = 150 keV/µm` goes through `z0_Gy` (`y0_keV_um = z0_Gy * rho pi r_d^2`), so
  the constructors pass `z0_Gy = 150 / (KEVUM * rho pi r_d^2)`. Do **not** let the
  geometric `saturation_y0_keV_um` default (from `R_n` and `beta`) apply.
- The paper gives no nucleus radius; `R_n` is inert here (the explicit `y0`
  overrides the only place it enters). Constructors set the mMKM default; it is
  not a fitted quantity.
- Implement `RBE_MN` as `alpha*/alpha_photon`, manifestly `h`-free.
