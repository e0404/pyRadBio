# Track structure — Elsässer, Cunrath, Krämer & Scholz 2008 (doi:10.1088/1367-2630/10/7/075005)

New J. Phys. **10** 075005, cited by Friedrich 2012 as **"Elsässer et al. 2008b"**.
The *other* 2008 paper (IJROBP 71 866–872 = LEM III, `LEM_III_Elsasser2008.md`) is
"2008a"; they share the 13 / 77 keV/µm beam qualities and are easy to confuse.
Data: `benchmark/elsasser2008b.py` (`TRACK_MODELS`); goldens:
`tests/benchmark/test_elsasser2008b.py`.

## Idea
Not a LEM version paper: it surveys amorphous track-structure models and asks
what each does to RBE within the LEM. Useful to us for two reasons:

1. **Table 1 is printed**: `RBE_alpha` with uncertainties, five models × two
   LETs, so no digitization anywhere in the chain.
2. **It states the core / `D_t` degeneracy** (sec. 4): "the LEM gives good results
   for all track structure models by choosing the threshold parameter `Dt`
   appropriately." Never move the core without re-fitting `D_t` (this nearly
   produced a wrong LEM IV conclusion, see `LEM_IV_Friedrich2012.md`).

## Track-structure menu (sec. 3.1.5)

| label | core | used by |
|---|---|---|
| `CR10` | constant `r_min = 10 nm` | **LEM I** (diffusion folded into the core) |
| `CR03` | constant `r_min = 0.3 nm` | **LEM II** |
| `EDR11` | `r_min = beta*r_c`, `r_c = 11.6 nm` | comparison with KC / MKM |
| `EDR40` | `r_min = beta*r_c`, `r_c = 40 nm` | **LEM III** |
| `KC` | Kiefer–Chatterjee, `r_c = 11.6 nm` | comparison only |
| `MC` | delta-electron transport | comparison only |

Shared: `r_max = gamma E^delta`, **`gamma = 0.062`, `delta = 1.7`** (sec. 3.1.1,
Kiefer & Straaten) for every model in the family, confirming LEM II's stated
value (LEM I's own 0.05 remains the SK1996/1997 value). All but `CR10` are
convolved with the `sigma = 4 nm` radical-diffusion Gaussian (sec. 3.3, eq. 13,
the `I_0` Bessel form); `CR10` carries **no cluster `eta`** (p. 13).

`r_c` provenance (sec. 3.1.2): Mozumder's adiabatic-invariance argument gives
**9.3 nm** for liquid water (`n = 1.33`, `omega_0 = 17 eV/hbar`); Chatterjee &
Schaefer **11.6 nm**; **LEM III's 40 nm is a chi^2 fit to rat spinal-cord
tolerance**. LEM IV's `r_c = 6.5 nm` (Pfuhl 2020) is smaller than all of them.

## Table 1 golden
HSG, `alpha_x = 0.313`, `beta_x = 0.0615`, `r_n = 5 um`, carbon. Each model has
its **own fitted `D_t`** (Fig. 6 caption): 6 (`CR03`), 23 (`EDR40`), 30 (`CR10`),
1.6 (`MC`), 0.01 (`KC`) Gy; the paper calls the last two physically unrealistic.
Printed values and their ±10%-`D_t` uncertainties: `TRACK_MODELS`.

pyRadBio vs the six implementable numbers (`test_golden_table1_rbe_alpha`):

| | CR03 (LEM II) | CR10 (LEM I) | EDR40 (LEM III) |
|---|---|---|---|
| 13 keV/µm | −2.2% | −0.9% | −1.9% |
| 77 keV/µm | −3.9% | +1.3% | −2.1% |

Three cores spanning ~100× in central dose, one parameter set, no tuning.
Insensitive to the stopping power (Simple Bethe vs pyMKM/Geant4 differ by <0.3%).

## The initial-slope method is the cross-section integral
Sec. 4: "we apply the approximation described in [10]" — [10] is Scholz,
Kellerer, Kraft-Weyrather & Kraft 1997 (Radiat. Environ. Biophys. 36 59–66), the
inactivation cross-section `sigma = int (1 - exp(-nu(b))) 2 pi b db`, i.e.
`initial_slope_model="cross_section"`. At 13 keV/µm (carbon ~282 MeV/u,
`r_max/R_eff ~ 180`):

| method | CR03 | CR10 | EDR40 |
|---|---|---|---|
| `cross_section` | **−2.2%** | **−0.9%** | **−1.9%** |
| `lethal_central_saturating` (E&S 2007 eq. 8) | −18.0% | −17.6% | −29.5% |
| `lethal_central` (TRiP98) | −17.4% | −16.8% | −29.1% |
| `lethal_full` (no overkill) | −1.6% | −0.2% | −1.6% |

`lethal_full` overshoots 77 keV/µm by +6 to +13%, so **`cross_section` is the
only method satisfying both columns**
(`test_golden_table1_selects_the_cross_section_initial_slope`). This confirms on
printed numbers what the LEM II reconstruction found from digitized figures
(`LEM_II_Elsasser2007.md`).

## Model-vs-data statement
Conclusion, p. 16: the LEM predicts cell inactivation "for particles with an LET
above 50 keV/µm with an excellent accuracy", but "the predicted values for
low-LET particles are significantly overestimated", the same trend as earlier
lighter-particle comparisons. A property of the LEM, not of our implementation.
