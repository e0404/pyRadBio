# rapidLEM — Krämer & Scholz 2006 (the "low-dose approximation")

**Krämer & Scholz, *Rapid calculation of biological effects in ion radiotherapy*,
Phys. Med. Biol. 51 (2006) 1959–1970, doi:10.1088/0031-9155/51/8/001.**

## Idea
The fast (analytic, sampling-free) derivative of the LEM used by TRiP98 for
treatment planning. It maps the intrinsic LEM coefficients `(α_z, β_z)` — the
per-track dose-response of eq (1), `-ln S(z_cn) = α_z z_cn + β_z z_cn²` — to the
macroscopic LQ coefficients `(α_D, β_D)` of eq (2), `-ln S(D) = α_D D + β_D D²`,
without the classical Monte-Carlo traversal sampling (repeated 500–1000× per
voxel). Accuracy vs the classic method is < 5 % at therapeutic doses
(≤ ~10 GyE/fraction), with 1–2 orders of magnitude speed-up.

## Core equations (sec. 2.2)

    d_1 = C · LET / A                      (eq 7)   [A = nucleus area]
    S_1 = exp(-α_z d_1)                     (eq 7)   single-traversal survival
    α_D = (1 - S_1) / d_1                   (eq 9)   = (1 - exp(-α_z d_1)) / d_1
    f   = α_D / α_z = (1 - exp(-α_z d_1))/(α_z d_1)   (eq 10)
    β_D = f² · β_z                                    (eq 10)

- Eq 9 is the 1-hit / low-dose limit of eq (3), `α_D = lim_{D→0} -ln S(D)/D`, with
  the Poisson hit probability `η = 1 - exp(-A F_η)` reduced to one hit. **No
  `s_max` cap**: `α_D → α_z` at low LET and `α_D → 1/d_1 = A/(C·LET)` at high LET
  (thins down as 1/LET).
- Eq 10 is a scaling *ansatz*, squared because β pairs with D²; the paper flags
  β_D as "less justified", with deviations growing at high dose (hence
  *low-dose* approximation). `f(E)` (Fig 1): `f → 1` as LET → 0, dipping to a
  minimum at the LET maximum (`E ≈ 0.3 MeV/u`), classified by
  `α_z d_1(E=0.3) ∝ α_z/A`.
- High-dose survival (eq 16): `-ln S = α_D D + β_D D²` for `D ≤ D_cut`, else
  `(β_D D_cut + α_D) D_cut + (D - D_cut)·s_max`.
- Mixed fields (eqs 11): dose-averaged `α` and `√β`, the convention in
  `core/mixing.py`.

## Table 1 base data (12C)

| tissue | α_x (Gy⁻¹) | β_x (Gy⁻²) | D_cut (Gy) | s_max (Gy⁻¹) | A (µm²) | α_z d_1(0.3) |
|---|---|---|---|---|---|---|
| Skin | 0.0172 | 0.0029 | 15 | 0.10 | 28 | 0.48 |
| CHO-K1 | 0.20 | 0.019 | 30 | 1.3 | 95 | 1.8 |
| Chordoma | 0.10 | 0.05 | 30 | 3.1 | 78.5 | 5.3 |
| Lung | 0.0684 | 0.0167 | 15 | 0.57 | 12.6 | 6.0 |

`s_max = α_x + 2 β_x D_cut`. High `α_z d_1` (lung, chordoma) → steep `f` drop →
the low-dose approximation deviates most there and at the Bragg peak.

## Mapping to pyRadBio (`models/lem/base.py`, shared by LEM I/II/III)

| KS2006 | pyRadBio | note |
|---|---|---|
| eq 9 `α_D` | `initial_slope_model="lethal_central_saturating"` (`LEMBase.lethal_central_saturating_initial_slope`) | central traversal only: thins down at high LET, **under-shoots `α_x` at low LET** (misses indirect large-`b` traversals) |
| eq 10 `β_D = f² β_z` | `beta_model="rapid_lem"` (`LEMBase.rapid_lem_beta`), `f = (1-exp(-N_1))/N_1`, `β_z = (s_max-α_z)/(2 D_t)` (Scholz 1997 eq 11) | same central-traversal caveat |
| — | `cross_section` (default α) | the impact-parameter integral over all `b`; same `1 - exp` saturation and thin-down, and recovers `α_x` |
| — | `lethal_central`, `min(N_1/D_1, s_max)` | **not** eq 9: no `1-exp` saturation (`N_1/D_1 ≈ α_z`), hard `s_max` plateau; reproduces the TRiP98 tables |
| — | `slope_matching` (default β), `(s_max - α_I)/(2 D_t)` | not eq 10; over-steepens carbon β ~2× vs the exact `scholz1997_two_track` |

Pair `lethal_central_saturating` + `rapid_lem` for the literal KS2006 rapidLEM.
Variant overview: `LEM_I_ScholzKraft.md` ("Initial-slope and beta variants").

## Cluster-enhancement cap (LEM II/III)
KS2006 predates the cluster `eta(D)` (Elsässer 2007/2008), so it cannot speak to
it. For internal consistency, `lethal_central` caps `N_1/D_1` at
`s_max·eta_max` (`LEMBase._max_local_slope`, overridden in LEM II/III with
`ClusterModel.eta_max`), because the local tail slope is `s_max·eta(D)` (Elsässer
2007 eq. 10). Capping at the un-enhanced `s_max` would silently drop the
enhancement that `cross_section` and `lethal_full` keep
(`test_lem2.py::test_lem2_lethal_central_keeps_cluster_enhancement`). LEM I
(`eta = 1`) is unchanged; the cap is never active there.
