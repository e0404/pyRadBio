# SMK — Stochastic Microdosimetric Kinetic model (Inaniwa & Kanematsu 2018)

Inaniwa T. & Kanematsu N., *Adaptation of stochastic microdosimetric kinetic
model for charged-particle therapy treatment planning*, Phys. Med. Biol. **63**
(2018) 095011, doi:10.1088/1361-6560/aabede. Equation numbers are the paper's.
It is the treatment-planning adaptation of Sato & Furusawa 2012, *Radiat. Res.*
178:341 (the non-adapted SMK).

## One-paragraph idea

The MK model evaluates survival at the *mean* nucleus specific energy, ignoring
cell-to-cell fluctuations; at high LET and high dose per fraction it
**underestimates survival**. Sato & Furusawa's SMK carries the stochastics of
both the domain (`z_d`) and nucleus (`z_n`) specific energy but needs costly
convolutions. This paper replaces them by a **second-order Taylor expansion of
`S_n(z_n)` about `z_n = D`**, which collapses the nucleus stochastics onto one
scalar `z̄_n,D`. Consequences: `β` becomes quality dependent, and survival gains
a non-LQ multiplicative factor.

## Inputs

| Symbol | Meaning | Unit | Source |
| --- | --- | --- | --- |
| `α0`, `β0` | LQ coefficients in the `LET → 0` limit | Gy⁻¹, Gy⁻² | table 1 |
| `z0` | saturation parameter, eq. (1) | Gy | **fitted**, table 1 |
| `r_d` | domain radius | µm | table 1 |
| `R_n` | nucleus radius | µm | measured, 8.1 (Suzuki 2000) |
| `z̄*_d,D` | dose-mean **saturation-corrected** domain specific energy per event | Gy | eq. (12) |
| `z̄_d,D` | dose-mean domain specific energy per event | Gy | eq. (11) |
| `z̄_n,D` | dose-mean **nucleus** specific energy per event | Gy | eq. (7) |

SMK needs the single-event spectrum at **two** site sizes (`r_d` and `R_n`), the
one structural difference from mMKM / MCF MKM. Saturation, eq. (1):
`z'_d = z0 √(1 − exp[−(z_d/z0)²])`.

## Core equations

    α_SMK = α0 + z̄*_d,D β0                                                  (15)
    β_SMK = β0 (z̄*_d,D / z̄_d,D)                                             (16)
    S(D) = exp(−α_SMK D − β_SMK D²) · {1 + D[−β_SMK + ½(α_SMK + 2 β_SMK D)²] z̄_n,D}   (24)

`α_SMK` is identical to the MK-model `α`. Origin: the full effect, eq. (8),

    ln S_n(z_n) = −α0 z_n (z̄*_d,F / z̄_d,F) − β0 (z̄_d,D z_n + z_n²)(z̄*_d,D / z̄_d,D)

becomes eqs. (14)-(16) under the **paper's own** approximation eq. (13),
`z̄*_d,F / z̄_d,F ≈ 1`. Eq. (24) follows from expanding `S_n` to second order
about `z_n = D` (eqs. 18-21) using only the moments eqs. (5)-(6):
`∫z_n f_n dz_n = D`, `∫z_n² f_n dz_n = z̄_n,D D + D²`.

Independent check: the cumulant expansion
`E[e^{−f(z)}] ≈ e^{−f(D)}[1 + ½Var(z_n)(f'(D)² − f''(D))]` with
`Var(z_n) = z̄_n,D D`, `f' = α + 2βD`, `f'' = 2β` reproduces eq. (24) exactly.

## Parameters — table 1

Least-squares fits (eq. 25) to Furusawa et al. 2000 HSG/V79 survival under
³He/¹²C/²⁰Ne, ~300 points per cell line. Shipped as
`StochasticMKM.hsg_inaniwa2018()` / `.v79_inaniwa2018()`.

| | HSG SMK | HSG MK | V79 SMK | V79 MK |
| --- | --- | --- | --- | --- |
| `α0` (Gy⁻¹) | 0.174 | 0.150 | 0.103 | 0.136 |
| `β0` (Gy⁻²) | 0.0568 | 0.0593 | 0.0210 | 0.0200 |
| `r_d` (µm) | 0.28 | 0.29 | 0.23 | 0.24 |
| `R_n` (µm) | **8.1** | 3.9 | **8.1** | 3.6 |
| `z0` (Gy) | 66.0 | 55.0 | 122.0 | 105.8 |
| `Δ²/n` | 0.0229 | 0.0452 | 0.0430 | 0.0575 |

Photon reference for the eq.-(26) biological dose (HSG): `α_X = 0.331 Gy⁻¹`,
`β_X = 0.0593 Gy⁻²` (Kanai et al. 2006) — **not** `β0`.

**Trap 1 — `z0` is a free fit, not the Kase relation.** mMKM ties `z0` to
`(R_n/r_d)²/√(β0(1+(R_n/r_d)²))`; for the HSG SMK row that gives ~121 Gy, the
paper fits **66.0 Gy** (factor 1.84). In SMK `R_n` is used *only* for `z̄_n,D`
(sec. 3.2). Hence the explicit `z0_Gy` field.

**Trap 2 — `R_n` means something else than in MKM.** The MK columns carry an
effective radius (3.6-3.9 µm) entangled with the saturation; the SMK columns the
*measured* mean nucleus radius, 8.1 µm (Suzuki et al. 2000, sixteen human cell
lines, range 6.7-9.5 µm). Using 4.3 µm overestimates `z̄_n,D` by
`(8.1/4.3)² ≈ 3.5×`.

## Known limits / failure modes

- **Eq. (24) is a truncated expansion**, valid when the number of events per
  nucleus is large (sec. 3.1). `1 + γD` grows without bound. `StochasticMKM`
  clips `S` to `[0, 1]` and exposes `survival_valid_to_Gy`, solving
  `γ(D)·D = expansion_validity_limit` (default 1.0). With the paper's HSG
  parameters and the bundled carbon AMF spectra (nucleus approximated by the
  domain-scored spectrum, as the goldens do):

  | E [MeV/u] | `z̄_d,D` [Gy] | `z̄*_d,D` [Gy] | `z̄_n,D` [Gy] | `α_SMK` | `β_SMK` | `γD = 1` at |
  | --- | --- | --- | --- | --- | --- | --- |
  | 1 | 508.3 | 9.44 | 0.607 | 0.710 | 0.0011 | 6.31 Gy |
  | 10 | 94.4 | 32.39 | 0.113 | 2.014 | 0.0195 | 3.82 Gy |
  | 100 | 12.2 | 11.78 | 0.015 | 0.843 | 0.0547 | 17.81 Gy |

  The self-computed cylinder route (`StochasticMKM.survival_valid_to_Gy(ion)`)
  gives 6.3 / 3.1 / 18.0 Gy (see `examples/smk_nucleus_stochastics.py`). The paper
  states accuracy for `D ≲ 10 Gy` for a therapeutic carbon beam (sec. 2) and plans
  to 15 Gy (RBE) / ~12 Gy absorbed (sec. 4.2.3, table 2), so the ceiling is the
  same order as the paper's doses. At 10 MeV/u it (3.82 Gy) sits slightly below
  `D(S=0.1) = 4.04 Gy`, so an iso-survival `RBE₁₀` there is marginally outside the
  expansion.
- The exact double-stochastic SMK (Sato & Furusawa 2012) has no truncation; the
  paper finds its adapted model "comparable or even more accurate" against the
  same data (sec. 5).
- Out of scope: fig. 5 (pencil-beam depth distributions, needs beam transport)
  and table 2 (biological dose over a GTV, needs a TPS).

## Golden values for tests

`tests/models/test_smk.py`, `@pytest.mark.golden`:

1. Table 1 parameters for HSG and V79, exact.
2. `z0` free of the Kase relation: fitted 66.0 Gy vs geometric 121.3 Gy (rel 1e-2),
   ratio 1.84 (rel 0.02).
3. `z̄_d,D / z̄_n,D` "about three orders of magnitude" (sec. 4.2.1): for a contained
   track the geometric `(R_n/r_d)² = (8.1/0.28)² = 837`, ion- and
   energy-independent. Catches trap 2.
4. Expansion ceiling 6.31 / 3.82 / 17.81 Gy (rel 0.02) for carbon at 1 / 10 / 100
   MeV/u, table above.

Non-golden: agreement with pyMKM `model="stochastic"` at `rel=1e-12`; `α_SMK`
equal to the mMKM `α` to machine precision; the `(1+γD)` factor against a direct
Poisson-weighted average of `exp(−f(k z_1))`, to `<2e-3`, the residual shrinking
`>2.5×` per halving of `z_1` (a genuine second-order truncation).

## Fig. 4 reconstruction

All 30 series of fig. 4 (`z̄_d,D`, `z̄*_d,D`, `z̄_n,D` for `Z_p = 1-10`) are in
`pyradbio/benchmark/inaniwa2018.py` (read from the PDF's **vector paths**; axis
calibration and the two extraction traps are in that module), reconstructed by
`validation/smk_inaniwa2018.py`, goldens in `tests/benchmark/test_inaniwa2018.py`.
Pooled over all ten species, within each panel's resolvable range, against
`StochasticMKM.hsg_inaniwa2018()` (Kiefer-Chatterjee radial dose,
`SimpleBetheStoppingPower`):

| quantity | n | median ratio | RMS log-ratio | worst |
| --- | --- | --- | --- | --- |
| `z̄_d,D` (eq. 11) | 134 | 0.933 | 0.065 | 13.0% |
| **`z̄*_d,D` (eq. 12)** | 223 | **1.005** | **0.049** | 10.3% |
| `z̄_n,D` (eq. 7) | 142 | 0.940 | 0.059 | 12.9% |

`z̄*_d,D`, which builds `α_SMK`, is unbiased. The two uncorrected dose means run
~6% low because they are dominated by the track **core**, where our track model
and stopping power differ most from the paper's and which the saturation
correction flattens away. `β_SMK` depends on the ratio `z̄*_d,D / z̄_d,D`, so the
bias propagates there (~+7%). (`benchmark.inaniwa2018.Z_D_BIAS = 1.07` is this
offset, a diagnostic only.)

Normalisation-free checks:

- **Peak positions:** `z̄*_d,D` peaks at 0.8 / 11.6 / 38.5 MeV/u for He / C / Ne
  vs the paper's "around 1 / 12 / 40" (sec. 4.2.1).
- **Two-site ratio:** `z̄_d,D / z̄_n,D` starts at 837 while the track is contained
  in both sites, then falls to ~600 by 20 MeV/u as `r_max` outgrows the 0.28 µm
  domain but not the 8.1 µm nucleus; we track the fall-off within 3-6%. So the
  paper's "three orders of magnitude" is a contained-track statement.

## Figs. 2-3 reconstruction

All 54 survival series (HSG fig. 2, V79 fig. 3; 9 panels each of
`{measured, SMK, MK}`) are in `benchmark/inaniwa2018.py`, scored in part 2 of
`validation/smk_inaniwa2018.py`. Fig. 3 is stroked vectors; fig. 2 is flattened
to filled outlines (centreline recovered by binning; its legend sits inside
panel (a) in data colours — see the module). Nothing is fitted:
`energy_for_let_MeV_u` inverts the quoted LET (high-energy root), then track +
disk geometry + eqs. (15), (16), (24).

`D_10` ratio against the paper's own curves:

| | SMK | MK |
| --- | --- | --- |
| HSG (fig. 2) | 0.987 – 1.005 | 0.997 – 1.008 |
| V79 (fig. 3) | 0.990 – 1.024 | 0.997 – 1.062 |

SMK reproduces to 1-2%, MK to ~6%; the MK arm is looser because its effective
`R_n` (3.6-3.9 µm) places it where the track models differ most.

Eq. (25) `Δ²/n = mean[log10(S_exp) − log10(S_cal)]²` on the digitized points:

| | SMK | MK | paper table 1 |
| --- | --- | --- | --- |
| HSG | 0.0140 | 0.0697 | 0.0229 / 0.0452 |
| V79 | 0.0125 | 0.0295 | 0.0430 / 0.0575 |

Same ordering as table 1; absolute values differ because the figures show three
LETs per ion (~50 points) against ~9 (~300 points) in the fit. SMK's advantage
sits in the top-LET panels: 654 keV/µm Ne (HSG) gives 0.0109 vs 0.5674 (×52);
528 keV/µm Ne and 432 keV/µm C (V79) give ×13 and ×7. At low LET the two models
coincide and MK is marginally better (digitization noise).

## Implementation notes / gotchas

- `α_SMK` uses `z̄*_d,D`; `β_SMK` uses the **ratio** `z̄*_d,D / z̄_d,D`. Guard
  `z̄_d,D → 0`.
- **Geometry.** The paper's sensitive volumes are **cylinders** of radius `r`,
  length `2r`, tracks parallel to the axis (sec. 3.2, after Inaniwa 2010).
  `site_geometry="cylinder"` (default for the ion route) integrates the radial
  dose over domain and nucleus **disks**, identical to `NativeMKMBackend` and
  pyMKM's `SpecificEnergy`. `"sphere"` is selectable and shifts `z*_1D` by
  −3 / −9 / +29 / +39% at 1 / 10 / 100 / 400 MeV/u (same with SimpleBethe, so not
  a stopping-power effect). A supplied `d(y)` bypasses this (the caller owns the
  geometry; `z = y/(K ρ π r²)` holds for both a mean-chord sphere and an
  axis-parallel cylinder). See also `Microdosimetry.md`.
- The nucleus impact-parameter profile needs the same high node counts as the
  domain; at default counts it jittered `z̄_n,D` by ~2% around 0.5-1 MeV/u.
  `specific_energies` scores both at the high counts.
- SMK is **not** an LQ model: `alpha_beta()` returns the LQ core
  `(α_SMK, β_SMK)`; iso-survival RBE inverts eq. (24) numerically.
- OSMK attaches to exactly eqs. (15)-(16) as oxygen corrections plus scaling of
  `r_d` and `z0`: see [`OSMK_Inaniwa2021_2023.md`](OSMK_Inaniwa2021_2023.md).
