# OSMK — Oxygen-effect-incorporated SMK (Inaniwa & Kanematsu 2021 / 2023)

Two papers, two *different* formulations, both shipped
(`models/mkm/osmk.py::OSMK2021` / `::OSMK2023`):

- **OSMK 2021** — Inaniwa, Kanematsu, Shinoto, Koto & Yamada, *Phys. Med. Biol.*
  **66** (2021) 205007, doi:10.1088/1361-6560/ac29cc (the "original OSMK model").
- **OSMK 2023** — Inaniwa & Kanematsu, *J. Radiat. Res.* **64** (2023) 685-692,
  doi:10.1093/jrr/rrad049, the "event-by-event" revision (open access).

Equations and parameters are from the primary texts. Prerequisite:
[`SMK_Inaniwa2018.md`](SMK_Inaniwa2018.md).

## One-paragraph idea

Under hypoxia initial DNA lesions can be *chemically repaired* instead of fixed
by oxygen. OSMK defines a **relative radioresistance** `R(pO2) >= 1` (the ratio
of `z*_d` needed for equal effect at `pO2` to that at the aerobic reference) and
reduces only the *sub-lethal* lesion rate, `k_d -> k_d / R`; the lethal rate
`lambda_d` is untouched. Since `alpha_S ∝ k_d` and `beta_0 ∝ k_d^2`, this gives
`alpha_S -> alpha_S/R`, `beta_0 -> beta_0/R^2`, and nothing else in the SMK
algebra changes. The two papers differ only in what makes `R` shrink for dense
tracks (the OER's fall with LET).

## Inputs

| symbol | unit | meaning | source |
|---|---|---|---|
| `pO2` | mmHg | oxygen partial pressure (160 = 21% at 1 atm) | 2021 sec 3.1 |
| `K` | mmHg | pO2 at which `R = (1+Rmax)/2` (see the misprint below) | 2021 eq. (13) |
| `z_d,D`, `z*_d,D`, `z_n,D` | Gy | as in SMK eqs. (11), (12), (7) | SMK |
| `alpha_L` | Gy^-1 | `p·lambda_d`, lethal, oxygen-independent | 2021 eq. (6) |
| `alpha_S` | Gy^-1 | `p·k_d·G_alpha`, sub-lethal, oxygen-dependent | 2021 eq. (7) |
| `beta_0` | Gy^-2 | `k_d^2·G_beta` | 2021 eq. (8) |

`alpha_L + alpha_S` is SMK's `alpha_0`. `K = 3 mmHg` is assumed for all cell
lines and qualities in both papers (2021 sec 3.2: a photon measurement, Wouters &
Brown 1997 / Hall & Giaccia 2006; "a first approximation" for high LET).

## Core equations (paper numbering)

### Shared

    R(pO2) = (pO2 + K) / (pO2 + K / Rmax)                       2021 (13) = 2023 (1)

    S(D, pO2) = exp(-a' D - b' D^2) * [1 + D {-b' + 0.5 (a' + 2 b' D)^2} z_n,D]
                                                                2021 (16) = 2023 (10)
    a' = alpha_L + alpha_S R^-1 + z*_d,D beta_0 R^-2            (17)/(11)
    b' = beta_0 R^-2 (z*_d,D / z_d,D)                           (18)/(12)

i.e. SMK eq. (24) with the primed pair. `z_n,D` is not modified by oxygen.

### OSMK 2021 — `Rmax` carries the LET dependence

    Rmax(z_d,D) = [(z_d,D/z_R)^g + R_m] / [(z_d,D/z_R)^g + 1]              (14)
    R(pO2, z_d,D) = (pO2 + K) [ pO2 + K ((z_d,D/z_R)^g + 1)/((z_d,D/z_R)^g + R_m) ]^-1   (15)

`R_m` is `Rmax` for very low `z_d,D`; `Rmax -> 1` as `z_d,D -> inf`. The
microdosimetry is not touched by `pO2`.

### OSMK 2023 — `Rmax` is a cell constant, the *site* carries the LET dependence

    r_d' = r_d / f_rd(pO2)                                                  (6)
    z_0' = z_0 * f_z0(pO2)                                                  (7)
    f_rd(pO2) =  1 + ((f_rd_max - 1)/(Rmax - 1)) (R(pO2) - 1)               (8)
    f_z0(pO2) = [1 + ((f_z0_max - 1)/(Rmax - 1)) (R(pO2) - 1)]^2            (9)

**`f_z0` is squared, `f_rd` is not** (eq. 9). `z_d,D'`, `z*_d,D'` are rescored at
the modified geometry ("event-by-event"). Rationale (2023 Materials & Methods):
radiosensitisation raises domain size (Hawkins 2001), so radioresistance lowers
it (Bopp et al. 2016); fewer primary lesions saturate later, so `z_0` rises.

## Parameters

### OSMK 2021, table 1 (`r_n = 8.1 um`, `K = 3 mmHg`)

| | HSG | V79 | CHO |
|---|---|---|---|
| `z0` (Gy) | 88.0 | 98.0 | 46.0 |
| `alpha_L` (Gy^-1) | 0.00 | 0.00 | 0.01 |
| `alpha_S` (Gy^-1) | 0.21 | 0.10 | 0.17 |
| `beta_0` (Gy^-2) | 0.043 | 0.026 | 0.045 |
| `r_d` (um) | 0.23 | 0.29 | 0.35 |
| `z_R` (Gy) | 28.0 | 29.0 | 9.7 |
| `g` (gamma) | 1.30 | 1.60 | 1.31 |
| `R_m` | 2.9 | 2.5 | 3.1 |
| `r^2` | 0.947 | 0.934 | 0.988 |

### OSMK 2023, table 1 (`r_n = 8.1 um`, `K = 3 mmHg`)

| | CHO | HSG | V79 |
|---|---|---|---|
| `alpha_L` (Gy^-1) | 0.0 | 0.0 | 0.0 |
| `alpha_S` (Gy^-1) | 0.132 | 0.0797 | **0.0** |
| `beta_0` (Gy^-2) | 0.0591 | 0.0677 | 0.0374 |
| `r_d` (um) | 0.41 | 0.29 | 0.30 |
| `f_rd_max` | 2.69 | 3.00 | 2.50 |
| `z0` (Gy) | 37.0 | 60.0 | 73.0 |
| `f_z0_max` | 3.23 | 3.53 | 2.96 |
| `Rmax` | 3.51 | 4.46 | 3.41 |
| `r^2` | 0.979 | 0.949 | 0.934 |

Column order differs (2021 HSG/V79/CHO, 2023 CHO/HSG/V79). Cross-check: 2023's
text gives the original model's `r^2` as 0.988 / 0.947 / 0.934 for CHO / HSG /
V79, matching the 2021 table.

Fitting data (both papers): Furusawa et al. 2000 + 2012 erratum (HSG, V79;
He/C/Ne; 18.5-654.0 keV/um; 160 and 0 mmHg) and Tinganelli et al. 2015 (CHO;
C/N/O/Si; pO2 = 0, 1.14, 3.8, 15.2, 160 mmHg). n = 485 / 639 / 136 for HSG /
V79 / CHO (2021 sec 3.2).

## Known limits / failure modes

- **Never tested below Z = 2** (2021 discussion: test at lower LET before proton
  use).
- **`R` is evaluated outside the rate equations**: `k_d' = k_d/R` is taken
  independent of `z_d` — "not strictly correct" (2021 discussion), but it keeps
  eqs. (2)-(3) analytic; the fit absorbs the error.
- **`K = 3 mmHg` is a photon number** reused for all LETs.
- **The 2023 revision buys little**: ≤ 1.9 %pt difference in the SOBP target and
  fragment tail, up to 4.2/4.4 %pt at the entrance (attributed to the refit).
  "The original OSMK model can still provide an accurate estimation."
- Inherits every SMK limit, including the eq.-(24) truncation ceiling.
- Out of scope: the treatment-planning halves (2021 eqs. 20-24 + the pancreatic
  case; 2023 fig. 4 SOBP), which need beam transport and a TPS.

## Golden values for tests

- **Aerobic limit.** `R -> 1` as `pO2 -> inf`, so `alpha'`, `beta'` reduce
  exactly to SMK eqs. (15)/(16) with `alpha_0 = alpha_L + alpha_S` (both
  formulations).
- **Misprint: `R(K)`.** Both papers describe `K` as the pO2 where
  `R = (1+Rmax)/2` (2021 after eq. 13; 2023 after eq. 1), the arithmetic
  midpoint. Eq. (13) gives `R(K) = 2 Rmax / (1 + Rmax)`, the **harmonic** midpoint
  (HSG-2023, `Rmax = 4.46`: 1.634, not 2.73). The prose holds for the usual form
  `(m·pO2 + K)/(pO2 + K)`, not this reciprocal one. `K` is assumed, not fitted,
  so nothing downstream changes; pin the algebra.
- **Anoxic limit.** `R(0) = Rmax` exactly; for 2021, `Rmax(z_d,D)` lies between
  `R_m` (low `z_d,D`) and 1 (high).
- **2023 scaling at `pO2 = 0`:** `f_rd = f_rd_max`, `f_z0 = f_z0_max^2`. HSG:
  `r_d' = 0.29/3.00 = 0.09667 um`, `z_0' = 60.0 * 3.53^2 = 747.65 Gy`.
- **OER falls to unity above ~200 keV/um** from ~3 at low LET (2021 discussion);
  `OER = D_10(pO2) / D_10(160 mmHg)`.
- 2021 OER fit quality: `r^2` = 0.907 (HSG, fig. 5), 0.870 (V79, fig. 6).

## Figs. 5-7 reconstruction

Thirty series in `pyradbio.benchmark.inaniwa2021`: HSG and V79 × He/C/Ne (figs.
5-6, anoxic only) and CHO × C/N/O/Si × {0, 1.14, 3.8, 15.2} mmHg (fig. 7), each
{measured points, the paper's OSMK curve}. Scored by
`validation/osmk_inaniwa2021.py`, goldens in `tests/benchmark/test_inaniwa2021.py`.
Fig. 7's measured series carry the mean and the top of the 1-sigma bar
(`OERCurve.oer` / `.oer_upper` / `.sigma`); figs. 5-6 have no error bars.

**Digitization.** All three panels are raster and hand-digitized. An earlier
automated colour-mask trace of figs. 5-6 agrees to 0.5% in LET and 0.005 in OER
(median ratio 1.000-1.004 over the fig.-6 series); the hand pass carries the V79
helium model curve to 220 keV/um (the trace stopped at 89), extending the V79
helium scoring range to 18.5-174 keV/um. The four graphical traps of the
automated pass are listed in the `benchmark.inaniwa2021` module docstring.

**LET axis: figs. 5-6 start at 15.0 keV/um, fig. 7 at 10.** Only `100` and
`1000` are labelled. Fitting `x = x0 + d*log10(v)` to the 17 interior ticks gives
`d = 315.9` / `316.2` / `287.9` px (figs. 5 / 6 / 7, every tick within 0.5 px)
against 576/577 px frames: figs. 5-6 span 1.823 decades (left edge
`1000 * 10^-1.823 = 15.0`), fig. 7 spans 2.001 (left edge 9.98). Reading figs. 5-6
as 10-1000 compresses LETs by up to 1.5× at the left edge and moves HSG from +3%
to +15%. Confirmations not used in the calibration: figs. 5-6 then land on the
stated fitted range (helium's lowest point 18.53, neon's highest 654.8); fig. 7's
measured LETs match the LETs printed on fig. 4's panels (51.1; 29.8 / 68.0 /
124.5; 124.5; 307.6 keV/um) within 0.3%.

**Physical trap.** The curves are drawn beyond attainable LETs: helium tops out
near its Bragg peak (~178 keV/um) and neon cannot go below ~22 keV/um.
`benchmark.inaniwa2021.reconstruction_lets` intersects the fitted range, the
digitized span and the attainable window.

**Result** (`OSMK2021`, driven by ion species and quoted LET via
`energy_for_let_MeV_u`, against the paper's own curves):

| cell | `r_d` (um) | `z_R` (Gy) | median ratio | RMS log | max \|dev\| |
|---|---|---|---|---|---|
| **CHO** (fig. 7) | 0.35 | 9.7 | 1.001 | **0.4%** | 1.7% |
| **HSG** (fig. 5) | 0.23 | 28.0 | 1.033 | 3.1% | 5.2% |
| **V79** (fig. 6) | 0.29 | 29.0 | 1.073 | 9.3% | 17.6% |

- **CHO** reproduces at 0.4% RMS with published parameters over four species and
  four pressures; its 1.14 / 3.8 / 15.2 mmHg arms (0.36 / 0.28 / 0.16%) are the
  only test of `R(pO2)` at intermediate pressure, i.e. of `K = 3 mmHg`.
- **HSG** has a near-uniform +2 to +5% offset (no LET dependence), consistent
  with the ~7% low `z_d,D` measured independently in the SMK fig.-4
  reconstruction; applying that factor gives 1.4% RMS.
- Both formulations reproduce the **particle-type ordering at equal LET**
  (He < C < Ne; sec 4.2): at matched LET the lighter, slower ion has the shorter
  delta-ray range, denser track, larger `z_d,D`, smaller oxygen effect.

### V79 mid-LET excess (open)

V79 runs ~7% high overall, up to 18% at 60-150 keV/um, near zero at both ends: a
*horizontal shift* of the eq.-(14) sigmoid. Inverting the chain per LET for the
`Rmax` that reproduces the paper's OER, then refitting eq. (14) against our own
`z_d,D`:

| cell | published (`z_R`, `g`, `R_m`) | free refit | RMS |
|---|---|---|---|
| HSG | 28.0, 1.30, 2.9 | **28.8**, 1.311, 2.775 | 0.28% |
| V79 | 29.0, 1.60, 2.5 | **21.6**, 1.657, 2.622 | 0.37% |

Eq. (14) reaches the paper's curves at < 0.4% from our microdosimetry; for V79
only `z_R` moves (25%), `g` and `R_m` stay within 3.6% / 4.9%. Eq. (14) reads
`z_d,D` only via `(z_d,D/z_R)^g`, so "our `z_d,D` is x% low" and "the published
`z_R` is x% high" cannot be separated here (`z ×1.25 = z_R ×0.80`).

Ruled out:

- Transcription error: table 1 re-read from the rendered PDF page; values match.
- Eq.-(24) ceiling: anoxic `D_10` reaches at most 45% of `survival_valid_to_Gy`.
- The measured `z_d,D` bias: ~7% at both domain sizes; rescaling nulls HSG at
  ×1.07, V79 needs ×1.25.
- Digitization: hand re-trace agrees with the automated trace (above); the paper's
  V79 curves track their own points at median 0.99 / 1.04 / 1.03 (He / C / Ne).
- An `r_d`-dependent bias in our `z_d,D`: the required factors are ×1.07 (HSG,
  0.23 um), ×1.25 (V79, 0.29 um), none (CHO, 0.35 um) — not monotone. Pinned by
  `test_the_correction_v79_needs_is_not_a_domain_radius_effect`.

Status: the discrepancy is one number, `z_d,D/z_R` for V79. The evidence points
at the published V79 `z_R = 29.0` Gy (~23 Gy fits its own figure with `g`, `R_m`
untouched), but this is not proven: the printed value is 29.0, and a V79-specific
difference between Inaniwa's `z_d,D` and ours is not excluded. Partial
explanation: V79's sigmoid is steeper (`g` 1.60 vs 1.30) and shorter (`R_m` 2.5
vs 2.9), so a given `z_d,D` error costs it more mid-sigmoid (shape, not size).
Pinned in `tests/benchmark/test_inaniwa2021.py`.

## Implementation notes / gotchas

1. **The three parameter tables are independent refits; never mix them with SMK
   2018.** HSG: `z0` = 66.0 (SMK 2018) / 88.0 (OSMK 2021) / 60.0 (OSMK 2023);
   `r_d` = 0.28 / 0.23 / 0.29; `beta_0` = 0.0568 / 0.043 / 0.0677.
2. **Aerobic (160 mmHg) is not `R = 1`.** `R(160)` = 1.014 for HSG-2023,
   1.013-1.018 across the tables. With note 1, OSMK-2021 at 160 mmHg is not
   numerically SMK-2018 and must not be tested as if it were.
3. **`alpha_L = 0` for HSG and V79 in both papers** (CHO 0.01 / 0.0), read as
   confirming Hawkins 2009's `lambda_d << k_d`; the whole `alpha` is then
   oxygen-sensitive. **V79-2023 also has `alpha_S = 0`**, so
   `alpha' = z*_d,D' beta_0 R^-2`; code must not assume `alpha_0 > 0`.
4. **OSMK 2023 rescored per `pO2`** at `r_d'` (HSG anoxia: 0.29 -> 0.097 um), so
   any spectrum or engine bound to `r_d` is invalid for `pO2 < inf`. 2021 has no
   such coupling.
5. **pyMKM rounds.** `mktable/compute.py::_get_osmk2023_corrected_parameters`
   does `round(domain_radius / f_rd, 3)` and `round(z0 * f_z0, 2)`; the
   `sftable/compute.py` path rounds `r_d'` to 2 decimals (HSG anoxia:
   0.0967 -> 0.10 um, 3.4% in radius, ~7% in `z_d`). Compare the 2021 path exactly
   and the 2023 path with a tolerance (or a pre-rounded `r_d`). pyMKM's algebra
   (`biology/oxygen_effect.py`) matches the papers exactly, including the square
   on `f_z0`.
6. **`R` for OSMK 2021 uses the uncorrected `z_d,D`**; in 2023 `R` does not depend
   on `z` and is computed before the rescoring.
