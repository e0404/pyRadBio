# LEM III — Elsässer, Krämer & Scholz 2008 (doi:10.1016/j.ijrobp.2008.02.037)

IJROBP 71:866-872 ("Elsässer 2008a"; not the New J. Phys. track-structure paper,
`LEM_TrackStructure_Elsasser2008b.md`). Code: `models/lem/lem3.py` (`LEM3`);
data: `benchmark/elsasser2008.py`; reconstruction: `validation/elsasser2008_lem3.py`.

## One-paragraph idea
LEM II plus a *velocity-dependent* extension of the inner track core: at higher
specific energy the primary ionisation/excitation events spread further from the
trajectory, so the core radius grows with beta. Empirically corrects the
residual entrance-channel RBE over-prediction of LEM II.

## Core equations / parameters
- Radial dose form as LEM II (eq. 1); r_max = gamma E^delta,
  gamma = 0.062 um (MeV/u)^-1.7, delta = 1.7 (eq. 2).
- Velocity-dependent core: r_min(beta) = beta * r_c, beta = v/c,
  r_c = 40 nm (eq. 6; fit to rat spinal-cord data; cf. Mozumder 9.3 nm).
- Photon survival: shouldered LQ -> exponential at D_t (eq. 5), with LEM II's
  cluster tail.
- Effective nuclear radius = **80% of geometric, on the RADIUS**: Methods ("the
  effective nuclear radius is approximately 80% of the average nuclear radius")
  and Table 1 caption ("taken as 80% of the average value"); `sqrt(0.8)` on the
  area is ruled out. This is the one paper whose per-line size is *geometric*,
  so it is the sole place `CellTarget.effective_size_factor = 0.8` belongs.
- Rat spinal cord in vivo (Fig. 1 caption): alpha = 0.1, beta = 0.05,
  R_n = 5 um, D_t = 30 (LEM I) / 9.5 (LEM II) / 60 (LEM III) Gy
  (`RAT_SPINAL_CORD`).

## Implementation
`LEM3` subclasses `LEM2` and overrides only the radial dose
(`ScholzKraftRadialDose.lem3`, `r_c = 40 nm`). The three figures (rat spinal cord,
two clinical depth-dose profiles) are treatment-planning outputs a survival model
cannot reproduce end-to-end, so nothing is digitized: **Tables 1-2 are the
reconstruction.** `SUZUKI_PANEL` holds the Suzuki 2000 15-cell panel (photon LQ,
experimental RBE_alpha at 13/77 keV/um, per-version `D_t`, the paper's signed
ΔRBE_alpha for LEM I/II/III) plus each line's nuclear area from Suzuki 2000
Table 3 (doi:10.1016/S0360-3016(00)00568-X); radius `0.8 * sqrt(area/pi)`.

## Golden values (Tables 1-2, signed ΔRBE_alpha = RBE_LEM/RBE_exp − 1)
Paper panel means (`PAPER_AVERAGE_DELTA`), 13 / 77 keV/um:
LEM I +0.45 / −0.17; LEM II +0.29 / −0.10; LEM III +0.11 / −0.02.

Ours (Bethe stopping power, per-line Suzuki areas, Suzuki Table 2 photon LQ):

| channel | LEM I | LEM II | LEM III | paper (III) |
| --- | --- | --- | --- | --- |
| 13 keV/µm (entrance) | +0.47 | +0.27 | **+0.12** | +0.11 |
| 77 keV/µm (Bragg)    | -0.14 | -0.14 | **-0.05** | -0.02 |

Per-cell goldens (`tests/models/lem/test_lem3.py`):
- `test_golden_lem3_reproduces_paper_per_cell_entrance_deviation` — 13 keV/µm,
  all 15 lines: per-cell RMS 0.026 (LEM I) / 0.017 (LEM II) / 0.014 (LEM III).
- `test_golden_lem3_reproduces_paper_per_cell_bragg_deviation` — 77 keV/µm,
  14 lines (KS-1 excluded): per-cell RMS 0.037 (LEM II) / 0.004 (LEM III).
- `test_golden_lem3_suzuki_entrance_channel_convergence` — the means above.

The entrance over-prediction shrinks I -> II -> III as in the paper. The 77 keV/µm
means read a touch low because KS-1 alone drags them.

## Paper defects
Table 1 prints alpha_x/beta_x **rounded** (2/3 decimals); its RBE_alpha column
reproduces only from Suzuki 2000 Table 2's measured values, so the benchmark uses
Suzuki's. With the rounded values the 13 keV/µm RMS was 0.058 / 0.025 / 0.034 and
LEM III's 77 keV/µm RMS 0.029. **SK-MG-1 beta_x is misprinted** (0.054 vs Suzuki's
0.05292): `ELSASSER2008_BETA_X_MISPRINTS`.

## Known limits
- **KS-1 outlier (open).** Highest `alpha_x` of the panel (0.74 Gy⁻¹). Our LEM
  overkills its high-LET track centre: ΔRBE_alpha(77) ≈ −0.50 (predicted
  RBE_alpha ~1.5 vs experimental 3.0) against the paper's 0.00, **independent of
  nuclear radius**. Not a stopping-power or nuclear-size effect; excluded from the
  77 keV/µm golden (`OUTLIER_CELLS_77`).
- Track modification is empirical (no validated microphysics); restrict to
  carbon. For protons/He at low energy, agreement is worse than LEM II.
- Small residual entrance-channel RBE over-prediction remains.
