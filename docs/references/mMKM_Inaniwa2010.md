# modified MKM — Inaniwa et al. 2010 (doi:10.1088/0031-9155/55/22/008)

> The `z*_1D` route of `MKMModel` runs on the native `NativeMKMBackend` by
> default; pyMKM (`PyMKMBackend`) is the optional cross-check (see `AGENTS.md`).
> This file holds the equations and golden values both are checked against.

## One-paragraph idea
Predicts RBE in mixed therapeutic ion fields via the LQ model with an
ion-dependent linear term and a constant beta. The radiation-quality dependence
enters only through `z*_1D`, the saturation-corrected dose-mean specific energy
of a subcellular *domain*, computed from the Kiefer-Chatterjee amorphous track
integrated over domain/nucleus geometry.

## Core equations
- `S = exp(-L_n)`                                              (eq. 1)
- `L_n = (alpha0 + beta*z*_1D) D + beta D^2`                    (eq. 2)
- `z*_1D = ∫ z_sat z f1(z) dz / ∫ z f1(z) dz`                   (eq. 3)
- `z_sat = (z0^2 / z)(1 - exp(-z^2/z0^2))`                      (eq. 4)
- `z0 = (R_n/r_d)^2 / sqrt(beta (1 + (R_n/r_d)^2))`             (eq. 5)
- `alpha_MKM := alpha0 + beta*z*_1D`; RBE via eq. (6)
- mixed field: `z*_1D,mix = Σ_j w_j d_ij (z*_1D)_ij / Σ_j w_j d_ij` (eq. 8,
  appendix), i.e. `z*_1D` is **dose-averaged** across the field.

## Parameters & golden values (aerobic HSG)
- `beta = 0.0615 Gy^-2` (sec. 2.2, from Kase 2008).
- `r_d = 0.32 um`, `R_n = 3.9 um`, `alpha0 = 0.172 Gy^-1` (sec. 2.2; weighted
  least-squares fit to Furusawa 2000 `D10` for He/C/Ne).
- Current HSG correction (sec. 2.5): measured `alpha = 0.282 Gy^-1` at
  7.3 mmWEL where `z*_1D = 3.43 Gy` gives `alpha0 = alpha - beta*z*_1D =
  0.0708 Gy^-1`.
- SOBP-centre verification (sec. 3.2): fitted `alpha = 0.652` vs
  `alpha_MKM = 0.664 Gy^-1`.

## Known limits
- `beta` is radiation-independent; this breaks down where the variance of `z_n`
  in the nucleus matters (very high LET, overkill), where the quadratic term
  actually decreases with LET. SMK addresses this
  ([`SMK_Inaniwa2018.md`](SMK_Inaniwa2018.md)).
- `alpha0` carries the cell radiosensitivity; `R_n`, `r_d` are assumed
  sensitivity-independent (sec. 2.5), a source of residual SOBP under-prediction.
- Eq. (5) and `beta` come from Kase et al. 2008
  (doi:10.1088/0031-9155/53/1/003), which has not been read for this summary.

## Implementation notes
- Geometry: domain and nucleus are cylinders/disks perpendicular to the track,
  not spheres (see `SMK_Inaniwa2018.md`, implementation notes, and
  `Microdosimetry.md`).
- pyMKM mapping: `r_d`/`R_n`/`beta0` -> `MKTableParameters`; read
  `z_bar_star_domain` (= `z*_1D`); form `alpha_MKM = alpha0 + beta*z*_1D`. API
  details: [`pyMKM_API.md`](pyMKM_API.md).
