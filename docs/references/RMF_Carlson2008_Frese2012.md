# RMF: Repair-Misrepair-Fixation (Carlson et al. 2008; Frese et al. 2012)

- **Carlson, Stewart, Semenenko & Sandison (2008)**, *Combined Use of Monte Carlo
  DNA Damage Simulations and Deterministic Repair Models to Examine Putative
  Mechanisms of Cell Killing*, Radiat. Res. **169**(4):447-459,
  **doi:10.1667/RR1046.1** (the model).
- **Frese, Yu, Stewart & Carlson (2012)**, *A Mechanism-Based Approach to Predict
  the Relative Biological Effectiveness of Protons and Carbon Ions in Radiation
  Therapy*, Int. J. Radiat. Oncol. Biol. Phys. **83**(1):442-450,
  **doi:10.1016/j.ijrobp.2011.06.1983** (the application and the formulation
  implemented).
- DSB yields: Semenenko & Stewart (2006), doi:10.1088/0031-9155/51/7/004, see
  `MCDS_SemenenkoStewart2006.md` (`pyradbio/micro/mcds.py`).

Implemented in `pyradbio/models/rmf/model.py`; tests `tests/models/test_rmf.py`.

**Name collision:** `models/phenomenological/frese.py::LinearScaling` is Frese et
al. **2011** (*Application of Constant vs. Variable RBE in IMPT*), a different
model. The RMF paper is Frese **2012**.

## One-paragraph idea

Cell killing is linked to DSB induction and processing, not to a local dose or
specific energy, so the biology is independent of the LEM/MKM lineage. A cell dies
either because a single DSB is misrepaired or fixed (one-track, linear in dose) or
because two DSBs interact pairwise (*binary misrepair*). The pairwise process
happens both within a track (**intratrack**: linear in dose, the source of the LET
effect) and between tracks (**intertrack**: quadratic). Time-integrating the
kinetic scheme yields the LQ parameters directly.

## Inputs

| symbol | meaning | unit | source |
|---|---|---|---|
| `Sigma(q)` | initial DSB yield | Gy^-1 Gbp^-1 | MCDS closed form |
| `zbar_F(q)` | frequency-mean specific energy | Gy | Frese eq. (7) |
| `theta` | tissue parameter (one-track lethality) | Gbp | derived, Table 1 |
| `kappa` | tissue parameter (pairwise interaction) | Gbp^2 | derived, Table 1 |
| `d` | site **diameter** | um | 5 um, both papers |
| `alpha_x, beta_x` | photon reference | Gy^-1, Gy^-2 | per tissue |

## Core equations (numbered as in Frese 2012)

```
(5)  alpha(q) = theta Sigma(q) + kappa zbar_F(q) Sigma^2(q)
(6)  beta(q)  = (kappa/2) Sigma^2(q)
(7)  zbar_F   = 0.204 LET / d^2                     [Gy]
(8)  alpha(q) = (Sigma/Sigma_x)[alpha_x + (2 beta_x/Sigma_x)(Sigma zbar_F - Sigma_x zbar_F,x)]
(9)  beta(q)  = (Sigma/Sigma_x)^2 beta_x
(10) dose-average alpha and dose-average sqrt(beta)   (Zaider-Rossi / Wilkens-Oelfke)
(13) alpha_D = theta <Sigma>_D + kappa <zbar_F Sigma^2>_D
(14) beta_D  = (kappa/2) <Sigma>_D^2
(16) RBE = [sqrt(alpha_x^2 + 4 beta_x D (alpha_D + beta_D D)) - alpha_x] / (2 beta_x D)
(17) RBE written out: depends on (alpha/beta)_x but NOT on alpha_x
```

Carlson's general forms, which Frese specialises:

```
(9C)  alpha = [1 - f_R(1-theta)] Sigma + kappa zbar_F (f_R Sigma)^2
(10C) beta  = (kappa/2)(f_R Sigma)^2
      kappa = (eta/lambda)[gamma - theta],  theta = [(1-a) phi lambda_R + lambda_F]/lambda
(2C)  zbar_F = 0.204 LET/d^2            (identical to Frese eq. 7)
(11C) G(lambda,T) = 2/(lambda T)^2 (e^{-lambda T} + lambda T - 1)    Lea-Catcheside
```

Frese eq. (5) is Carlson eq. (9C) with `f_R = 1` (fraction of potentially
rejoinable DSBs); Carlson's Results conclude "all classes of DSB are potentially
rejoinable". `RMFModel.carlson_alpha()` keeps the general form so that conclusion
stays checkable.

## Parameters

`theta` and `kappa` are **not free inputs**: Frese Table 1 derives them from each
tissue's printed `(alpha_x, beta_x)` by inverting eqs. (5)-(6):

```
kappa = 2 beta_x / Sigma_x^2
theta = (alpha_x - kappa zbar_F,x Sigma_x^2) / Sigma_x
```

**Reference radiations** (Frese, "Low-LET reference radiosensitivity parameters"):

| reference | `Sigma_x` [Gy^-1 Gbp^-1] | LET [keV/um] | used for |
|---|---|---|---|
| Co-60 | 8.27 | 0.36 | in-vivo / clinical sets |
| 200 kVp X-rays | 8.68 | 1.53 | Furusawa in-vitro cell lines |

Neither is reproducible from the MCDS closed form (both simulated with a
secondary-electron spectrum; MCNP for 200 kVp); they sit just above eq. (3)'s
`x -> 0` limit of 8.15, the only available cross-check.

**Table 1** (all six reproduced):

| tissue | `alpha_x` | `beta_x` | `(alpha/beta)_x` | `theta` (Gbp) | `kappa` (Gbp^2) | `theta/kappa` | ref. |
|---|---|---|---|---|---|---|---|
| T1 | 0.0305 | 0.0585 | 0.52 | 0.0033 | 1.6e-3 | 2.15 | 200 kVp |
| HSG | 0.313 | 0.0615 | 5.1 | 0.0359 | 1.6e-3 | 21.98 | 200 kVp |
| V79 | 0.184 | 0.020 | 9.2 | 0.0211 | 5.3e-4 | 39.82 | 200 kVp |
| Chordoma | 0.1 | 0.050 | 2 | 0.0121 | 1.5e-3 | 8.25 | Co-60 |
| Prostate | 0.15 | 0.048 | 3.1 | 0.0181 | 1.4e-3 | 12.79 (see below) | Co-60 |
| Head & neck | 0.25 | 0.025 | 10 | 0.0302 | 7.3e-4 | 41.33 | Co-60 |

**Table 1 defect:** Prostate's printed `theta/kappa` is 12.79 but its own printed
`alpha_x`/`beta_x` give 12.90 (0.8% out). Every other row is self-consistent to
<~0.1%. Almost certainly rounding, not a model issue.

## Known limits / failure modes

- **No RBE turnover.** `Sigma` and `zbar_F` rise with LET, so eqs. (5)-(6) rise
  monotonically. Frese: measured parameters peak between 100 and 250 keV/um while
  RMF "predicts increasing parameters beyond this point"; "for LET values that
  exceed ~150 keV/um, the model over-predicts measured alpha and beta". The code
  **warns above 150 keV/um** and never clamps.
- **Low-LET bias** (Frese): the model "tends to underestimate alpha and overestimate
  beta for lower LET values".
- **The site size is a DIAMETER.** `d = 5 um` in eq. (7) is a nucleus *diameter*
  (both papers). The `R_n = 5 um` used by the LEM/MKM entries here is a *radius*: a
  factor 2 in `d`, 4 in `zbar_F`.
- **Eq. (7) is crude.** Carlson: straight tracks, continuous slowing down, no
  multiple scattering; for some site sizes and particles "the accuracy of eq. (2) is
  poor".
- **Intra-bin covariance in eqs. (13)-(14):** the *product* `zbar_F Sigma^2` is
  dose-averaged, not the product of averages (AGENTS.md, Numerics).
- MCDS's own domain (`1 <= Z_eff^2/beta^2 <= 3200`) bounds `Sigma` from below.

## Golden values for tests

Table 1's derived `theta`/`kappa` columns and the limits both papers state in
words (`tests/models/test_rmf.py`). Because `theta`, `kappa` are printed to 2-3
significant figures, the goldens assert that our value **rounds to** the printed
one (on T1's `theta = 0.0033` half an ulp is already 1.5%, so a relative tolerance
would say nothing).

## Implementation notes / gotchas

- **RMF's `RBE_min` is the DSB-yield ratio, exactly:** eq. (9) gives
  `sqrt(beta/beta_x) = Sigma/Sigma_x`. RMF lands on the same `RBE_min/RBE_max`
  skeleton as `models/phenomenological` with mechanistically computed limits. It is
  deliberately **not** a subclass of `RBEMinMaxModel` (that base takes a scalar
  `LET_d`; RMF's limits are functions of `(Z, beta)`).
- **RBE is independent of `alpha_x`** at fixed `(alpha/beta)_x` (eq. 17; tested).
- **"Independent lineage" needs qualifying:** the repair kinetics are independent of
  LEM/MKM, but the track-quality parameterisation is Katz's. MCDS eq. (1) is the
  Barkas effective charge, identical to Katz 1971 eq. (12) and to
  `track/stopping_power.py::effective_charge`; Semenenko & Stewart cite Katz (1970)
  for `Z^2/beta^2` as the interpolation variable. See `Katz1971.md`.
- **Two unit systems:** Carlson per cell (Gy^-1 cell^-1), Frese per Gbp (Gy^-1
  Gbp^-1); 6.0 Gbp/cell, implied consistently by all four MCDS yield pairs. Ratios
  like `Sigma/Sigma_x` are unit-free, so eqs. (8)-(9) are the safest forms.

## Ancestors (not implemented)

Carlson sec. "RMF model" states the limits in which RMF reduces to each:
**RMR** when `lambda_F = 0`, `f_R = 1` and intratrack binary misrepair is
negligible; **LPL** when intratrack binary misrepair is negligible, `phi = 0`,
`lambda_F = 0` and `gamma = 1`. Both are ~50-line models that would make those
reductions testable.

- **LPL**: Curtis 1986, Radiat. Res. **106**(2):252-270, doi:10.2307/3576798. It has
  a published erratum, Radiat. Res. **119**(3):584 (1989), doi:10.2307/3577534; read
  it before shipping any LPL coefficient.
- **RMR**: Tobias 1985, Radiat. Res. Suppl. **8**:S77-S95, doi:10.2307/3583515
  (Crossref carries a duplicate record, 10.2307/3576635; prefer the supplement
  one).
