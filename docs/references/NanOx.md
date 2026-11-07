# NanOx: Cunha et al. 2017 / Alcocer-Avila et al. 2022, 2023 (and four more)

Implemented in `pyradbio/models/nanox/` (assembly and calibration layer only; the
forward ion route raises). Tests `tests/models/test_nanox.py`; example
`examples/nanox_local_and_global_events.py`; validation
`validation/nanox_alcocer_avila2023.py`.

## One-paragraph idea

NanOx (NANodosimetry and OXidative stress) is the Lyon group's answer to two
inconsistencies they identify in the LEM and MKM: the treatment of local versus
non-local events, and the incomplete accounting of the stochastic nature of energy
deposition. Survival factorises into two independent classes of event. **Local
lethal events** occur in nanometric targets (20 x 10 nm water cylinders), each able
to kill the cell on its own; the LEM's local dose is replaced by a stochastic
quantity, the **restricted specific energy** `z`, so no Poisson law is assumed at
the local scale. Local events alone give a **pure exponential** survival curve.
**Non-local (in the current version "global") lethal events** describe toxic
accumulation of oxidative stress from reactive chemical species at the scale of the
cell nucleus and supply the entire shoulder. The two-factor structure is borrowed
from Katz's ion-kill/gamma-kill, but NanOx's two channels are two different
mechanisms at two scales.

## The corpus (seven papers; DOIs Crossref-verified)

| Paper | DOI | Role |
|---|---|---|
| Cunha, Monini, Testa & Beuve 2017, *Phys. Med. Biol.* **62**(4):1248-1268 | `10.1088/1361-6560/aa54c9` | founding paper |
| Monini, Testa & Beuve 2017, *Acta Phys. Pol. B* **48**:1653 | `10.5506/aphyspolb.48.1653` | short companion; prints `beta_G = beta_r/eta^2` as eq. (3.2) |
| Monini, Cunha, Testa & Beuve 2018, *Cancers* **10**(4):87 | `10.3390/cancers10040087` | parameter sensitivity; **Table 5 prints `beta_r`** |
| Monini, Alphonse, Rodriguez-Lafrasse, Testa & Beuve 2019, *Phys. Imaging Radiat. Oncol.* **12**:17-21 | `10.1016/j.phro.2019.10.004` | LEM I-IV / MKM / NanOx head-to-head (Table 2: chi^2) |
| Monini, Cunha, Chollier, Testa & Beuve 2020, *Radiat. Res.* **193**(4):331 | `10.1667/RR15463.1` | **determines the ELLF**; eq. (28) |
| Alcocer-Avila, Monini, Cunha, Testa & Beuve 2022, *Front. Phys.* **10**:1011063 | `10.3389/fphy.2022.1011063` | hadrontherapy implementation; **the equation numbering followed here** |
| Alcocer-Avila, Monini, Cunha, Testa & Beuve 2023, *Front. Phys.* **11**:1011062 | `10.3389/fphy.2023.1011062` | general formalism |

Also: Alcocer-Avila, Levrague, Delorme, Testa & Beuve 2024, *Med. Phys.*,
`10.1002/mp.17407` (arXiv:2409.13257), the low-energy / BNCT / TRT extension where
the penumbra may be neglected (not implemented).

## Inputs

| Symbol | Meaning | Units | Source |
|---|---|---|---|
| `d_s`, `L_s` | sensitive-volume **diameter** and length (cylinder = nucleus) | um | 2022 Simplification 2 |
| `d_t`, `L_t` | local-target **diameter** and length | nm | 2023 Simplification 4 |
| `h`, `z0`, `sigma` | ELLF height, threshold, width | -, Gy, Gy | 2020 eq. (28) |
| `beta_G`, `alpha_G` | global-event LQ coefficients | Gy^-2, Gy^-1 | 2022 eq. (11) |
| `T_RCE` | time at which radical yields are scored | s | 2022 sec. 2.1.2.2 |
| `e_c` | edge of the square track-core cross-section | um | 2022 sec. 2.1.2 |
| `eta` | restricted fraction of the energy loss, `<z> = eta D` | - | 2023 Definition 2 |
| `a` | unit conversion `0.1602 Gy keV^-1 um^3` | - | 2022 eq. (35) |
| **`dP/dz|hit,c`**, `<z>_hit,c` | single-event restricted specific-energy spectrum in hit local targets in the core | Gy^-1, Gy | 2022 eq. (26), **LQD Monte Carlo** |
| **`RCE(ion,E)`** | relative chemical effectiveness `G/G_r` at `T_RCE` | - | 2022 eq. (10), **LQD/PHYCHEML/CHEM** |

The last two are the boundary (Known limits).

## Core equations (Alcocer-Avila et al. 2022)

```
(3)   S = S_L x S_NL                                   two independent classes
(4)   n*_i = -ln(1 - f(z_i))                           effective no. of local lethal events
(6)   S_L  = prod_k exp(-F(z_k))                       exact; no Poisson law
(7)   F(z) = -N ln(1 - f(z))                           the ELLF, over N targets
(8)   F(z) = (h/2)[1 + erf((z - z0)/sigma)]            = Monini 2020 eq. (28)
(9)   Ztilde = RCE * Z                                 chemical specific energy
(10)  RCE = G / G_r
(11)  S_G = exp(-alpha_G Ztilde - beta_G Ztilde^2)     the whole shoulder
(14)  V_c = L_s e_c^2                                  core, a square parallelepiped
(16)  r_c = V_c/V_s ;  r_p = 1 - r_c
(26)  alpha_c = (1/<z>_hit,c) int (dP/dz)_hit,c F(z) dz
(27)  n* = alpha_c r_c Z_c + alpha_r,L r_p Z_p         core/penumbra assembly
(35)  Kin = D sigma_s/(a LET) ;  Kout = D(Sigma - sigma_s)/(a LET)
(45)  alpha = Sigma (1 - S_1) / (a LET)
```

From Alcocer-Avila et al. 2023:

```
(34)  C_L = alpha_r,L/(alpha_r,L + alpha_G) = 1   =>   alpha_G = 0
(33)  alpha'_r = alpha_r/eta ;  beta'_r = beta_r/eta^2   (sigma_Z << <Z>)
(35)  alpha_r,L = alpha'_r ;  beta_G = beta'_r
(40)  S_L = exp(-n*)
(42)  n* = int (dP/dz) F(z) dz
(49)  S(D) = exp(-lambda (1 - S_1))                    Poisson over impacts
(51)  S(D) = exp(-Sigma (1 - S_1) D / (a LET))
(53)  alpha = zeta/(a LET),  zeta = Sigma (1 - S_1)    the Kanai relation, derived
(57)  zeta_2 = zeta_1                                  independent of Sigma
```

Eq. (52) is the structural result: with only local lethal events the survival curve
is a pure exponential, which is why non-local events are needed and the sharpest
formal difference from the LEM (whose local events alone are meant to give the
shoulder).

## Parameters

Monini 2020 Table 1 (ELF columns), identical in Monini 2018 Table 1, Monini 2019
Table 1 and Alcocer-Avila 2022 Table 1; `beta_r` from Monini 2018 Table 5.

| | V79 | CHO-K1 | HSG |
|---|---|---|---|
| `d_s` [um] (diameter) | 9.8 | 11.8 | 14.0 |
| `R_SV` [um] (radius, same object) | 4.9 | 5.9 | 7.0 |
| `L_s` [um] | 1.0 | 1.0 | 1.0 |
| `z0` [Gy] | 22 789 | 14 507 | 15 654 |
| `sigma` [Gy] | 8 117 | 2 781 | 549 |
| `h` | 225 841 | 104 810 | 179 439 |
| `beta_G` [Gy^-2] | 0.0405 | 0.0625 | 0.0961 |
| `beta_r` [Gy^-2] | 0.0259 | 0.0200 x 2 | 0.0615 |

Fixed for all lines: `d_t = 20 nm`, `L_t = 10 nm`, `alpha_G = 0`, `T_RCE = 1e-11 s`,
`e_c = 200 nm`, `eta = 0.8`. Monini 2018 concludes only five parameters are
cell-line specific: `(d_s, h, z0, sigma, beta_G)`.

**Where the `beta_r` come from, and why there is no `alpha_r`.** Monini 2018
Table 5 cites each `beta_r` to one measurement, chosen because "both the linear and
the quadratic components of a given photon irradiation were the closest to the
average values over all the available publications":

| | `beta_r` source |
|---|---|
| HSG | Furusawa et al. 2000, *Radiat. Res.* **154**:485, `10.1667/0033-7587(2000)154[0485:IOAAHC]2.0.CO;2` (Monini 2018's reference list misprints the journal as *J. Radiat. Res.*) |
| V79 | Cox, Thacker, Goodhead, Masson & Wilkinson 1977, *Int. J. Radiat. Biol.* **31**:561, `10.1080/09553007714550661` |
| CHO-K1 | Weyrather, Ritter, Scholz & Kraft 1999, *Int. J. Radiat. Biol.* **75**(11):1357, `10.1080/095530099139232`, **x 2**: no single measurement matched the mean `(alpha_r, beta_r)` pair, so "we applied a correction factor to the `beta_r` coefficient" |

The paired **`alpha_r` is printed in no NanOx paper**, so the presets carry `beta_r`
for provenance but no `PhotonParameters`. Nearest in the corpus: Monini 2019
Table 1 prints `alpha_x = 0.313` (HSG) and `0.228` (CHO-K1, undoubled
`beta_x = 0.020`) as the photon reference given to the **LEM and MKM**, and for V79
gives them `0.184 / 0.020`, a different photon fit from NanOx's `beta_r = 0.0259`.
Cunha 2017 prints `alpha = 0.112 Gy^-1` for V79 (Tilly et al. 1999), which belongs
with that paper's `beta_r = 0.0298`. The example and tests use `(0.112, 0.0259)`: a
**mixed pair**, harmless because `alpha_x` cancels out of `beta_G`.

## Golden values for tests

| Quantity | Value | Tolerance | Location |
|---|---|---|---|
| `beta_G` from `beta_r = 0.0259` | 0.0405 | rounds-to | Monini 2018 Tables 1 + 5 |
| `beta_G` from `beta_r = 0.0400` | 0.0625 | rounds-to | as above |
| `beta_G` from `beta_r = 0.0615` | 0.0961 | rounds-to | as above |
| `beta_G` from `beta_r = 0.0298` | 0.0466 | rounds-to | Cunha 2017 sec. 3.2.4 |
| `sigma_s` for `d_s = 9.78 um` | 75 um^2 | +-0.2 | Cunha 2017 sec. 3.2.1 |
| ELLF parameters, 3 cell lines | table above | exact | Monini 2020 Table 1 |

`beta_G = beta_r/eta^2` with **eta = 0.8 exactly** is the corpus's one closed-form
coefficient chain and reproduces every printed `beta_G` (two independent V79 fits,
three cell lines). This pins eta harder than the authors' simulation does
(Alcocer-Avila 2023 Table 4: 0.798-0.802). `rounds_to()` (shared with
`tests/models/test_rmf.py`) is used because the papers print `beta_G` to three
significant figures.

## Known limits / failure modes

**The forward ion route is not implemented (the whole scope decision).** Eq. (26)
needs `dP/dz|hit,c`, the single-event restricted specific-energy spectrum in
20 x 10 nm targets inside the 200 nm track core, and eq. (10) needs `RCE(ion, E)`.
Both come from the group's non-public **LQD / PHYCHEML / CHEM** Monte-Carlo chain;
there is no NanOx code release and **neither quantity is tabulated in any of the
seven papers**: `dP/dz` appears only as 2022 Figs. 4-5 (two ions), `RCE` only as
2023 Fig. 5 (three ions). Monini 2020 has exactly one table; Cunha 2017's three
contain no alpha and no RCE.

`NanOxModel.alpha_beta()` raises `NanOxNanodosimetryRequired` naming both missing
quantities; `NanOxNanodosimetry` is the protocol for a caller who has them. NanOx is
the one family not registered in the PIDE harness.

**Why no substitute is shipped.** `pyradbio.micro` has no nanometre path (smallest
site exercised: a 100 nm sphere; working regime 0.5-2 um; `CylindricalSite` has no
volume integrator; `AmorphousTrackEngine` documents a small-site bias an order of
magnitude above a NanOx target). Feeding amorphous-track input to NanOx would
reintroduce the approximation the model removes, and any residual against a NanOx
curve would be uninterpretable (our spectrum or our assembly?). Panel 6 of the
example shows the sensitivity: holding the median `z` fixed and varying only a
log-normal spectrum's width moves `alpha_c` by 17x. The same reasoning keeps the
Katz track-width regime out of `models/katz`.

**Limits stated by the papers:** track-segment regime only (no projectile/target
fragmentation, 2023 sec. 2.3.1); dose rate not modelled (2023 discussion; contrasts
with the MKM); no bystander effects; a single sensitive volume (the nucleus) in the
hadrontherapy version, while the general formalism allows several. Alcocer-Avila
2023 Table 5 (p. 12): the penumbra/core energy ratio depends on energy, not ion
type, peaking at 50-100 MeV/u (about 0.63) with a 200 nm square core
(`validation/nanox_alcocer_avila2023.py`).

## Implementation notes / gotchas

**1. `d_s` is a DIAMETER; `R_SV` is a RADIUS.** Cunha 2017 and Alcocer-Avila 2022
print diameters, Monini 2018/2019/2020 radii of the same object. Settled
dimensionally: Cunha 2017 sec. 3.2.1 gives `d_s = 9.78 um` and `sigma_s = 75 um^2`,
and `pi (9.78/2)^2 = 75.1` (the paper prints the unit as "um^-2", a typo). Likewise
`d_t = 20 nm` is a diameter, while the 2022 figure captions call the same target
"10 nm radius and length".

**2. `F(z)` is not a probability and does not vanish at zero.** It is
`-N ln(1-f)`, an effective count over the N targets, saturating at `h` (order 10^5),
not 1. An erf has no compact support, so `F(0) = (h/2)(1 - erf(z0/sigma))` is small
but non-zero: 8.1 for V79, i.e. 3.6e-5 of `h`. It never matters: eq. (26)
integrates over targets that were *hit*, and the core term is scaled by
`r_c ~ 5e-4`.

**3. There is a FOURTH, EARLIER V79 parameter set; do not mix it in.** Cunha 2017
sec. 3.2.3 fits V79 at `z0 = 20 200 Gy, sigma = 6 000 Gy, h = 180 000` with its own
photon reference `beta_r = 0.0298` (Tilly et al. 1999) giving `beta_G = 0.0466`.
Every later paper uses `22 789 / 8 117 / 225 841` with `beta_r = 0.0259`. Not
shipped as a preset.

**4. The core/penumbra split is absent from the founding paper** ("core" and
"penumbra" occur zero times in Cunha 2017); it is a later simplification for the
hadrontherapy implementation. A third reason the 2017 parameters cannot be fed to
this module's machinery.

**5. `beta_r` for CHO-K1 is printed as "0.0200 x 2"** (Monini 2018 Table 5 doubles
its literature source). Carried as printed (0.0400), which reproduces the published
`beta_G`.

**6. The Sigma-independence of `zeta` is a real invariant, not a tautology.** Eqs.
(54)-(57): `S_1` is defined relative to the surface of influence, so enlarging the
surface dilutes `S_1` by exactly the amount that keeps `Sigma(1-S_1)` fixed. Worth
testing directly.

**7. The LCBF is a second ELLF representation (not shipped).** Monini 2020 eqs.
(21)-(22) build `F` from ramp basis functions `F_j = 0 | s(z-I_j) | 1` with interval
edges `(0, 5 000, 10 000, 15 000, 20 000, 25 000, 30 000, 50 000) Gy` and weights
`x0..x7` in Table 1. Eight edges give seven intervals but eight weights are printed;
the eighth weight `x7` is zero for all three cell lines, so it never contributes,
and reading `x_j` as the weight of the ramp on `[I_j, I_{j+1}]` is the only
convention that gives every printed weight an interval. Printed non-zero weights:
V79 `x3 = 153247.8, x4 = 19314.8`; CHO-K1 `x2 = 44253.1, x3 = 71714.6`; HSG
`x0 = 4.2, x2 = 12291.3, x3 = 196002`. Under this reading the LCBF is a different
fit from the erf, not a re-expression: its saturation `sum(x_j)` sits -24% / +11% /
+16% from the erf's `h` for V79 / CHO-K1 / HSG and its half-height within ~20% of
`z0`, matching the two distinct curves in the paper's Fig. 2. Only the erf ships
(every later paper uses it).

**8. Positioning against this package's other models** (2023 sec. 4): NanOx takes
the *local lethal event* from LEM I but drops the Poisson law; its local/global pair
maps onto the MKM's type I / type II lesions, except that global events model
accumulation rather than pairwise interaction; LEM IV keys on the initial DSB
distribution and still uses an amorphous radial dose. Monini 2019 Table 2 scores
NanOx against LEM I-IV and MKM on chi^2.
