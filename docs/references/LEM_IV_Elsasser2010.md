# LEM IV — Elsässer et al. 2010 (doi:10.1016/j.ijrobp.2010.05.014)

Int. J. Radiat. Oncol. Biol. Phys. 78(4):1177-1183. The **primary** LEM IV paper
(Friedrich 2012, `LEM_IV_Friedrich2012.md`, is the proceedings follow-up).
Introduces the DSB-cluster generalisation of the LEM and fixes its one new
parameter (`l_DSB`) on proton/He/C data. `pyradbio/models/lem/lem4.py` implements
the single-particle (central-traversal) approximation; every free parameter, fork
and known defect of that implementation is in `LEM_IV_OPEN_CHOICES.md`.

## One-paragraph idea
The effect is driven by the *initial DNA damage distribution*, not the local dose
directly. The nucleus is tiled into cubic subvolumes of side `l_DSB`; the local
dose sets the mean DSB per subvolume (~30 DSB/Gy/cell). A subvolume with one DSB
is *isolated* (iDSB), with ≥2 *clustered* (cDSB). Matching the ion cluster index
`C_I` to the photon `C_γ(D)` gives a photon-equivalent dose `D_eq`; the
single-track effect is the photon effect at `D_eq` scaled by the ratio of damaged
subvolumes. Macroscopic `α_ion`/`β_ion` then follow "by the method of the original
LEM (13)" (Scholz 1997, the full simulation).

## Core equations
- **Damage density (eq. 1):**
  `ΔE(d_loc) = -ln S_γ(d_loc)·(ΔV/V_Nucleus) = (α_γ d_loc + β_γ d_loc²)·(ΔV/V)`,
  linear above `D_t`. Mean DSB per subvolume `30·(ΔV/V)·d_loc` → per-subvolume
  yield `κ_dsb = 30·l_DSB³/V ≈ 0.00945 /Gy` (l_DSB = 540 nm, V = 500 µm³); Poisson
  mean `m = κ_dsb·d_loc`.
- **Cluster index (eq. 2):** `C = N^Ion_cDSB / (N^Ion_cDSB + N^Ion_iDSB)`.
- **Photon-equivalent dose:** `D_eq` such that `C_γ(D_eq) = C_I`.
- **Effect scaling (eq. 3):**
  `E_I = (N^Ion_iDSB / N^γ_iDSB)·E_eq = (N^Ion_cDSB / N^γ_cDSB)·E_eq`,
  `E_eq = -ln S_γ(D_eq)`. The forms coincide because `C_I = C_γ(D_eq)` fixes the
  cDSB:iDSB ratio; equivalently `E_I = (N^Ion_total/N^γ_total)·E_eq` (pyRadBio's
  `κ`, an exact algebraic identity). `α_I = E_I/D_I`.

## Parameters (fixed for all cells and ions)
- `l_DSB = 540 nm` — **fitted** to minimise the deviation over the Fig. 4 data
  sets ("might be slightly readjusted" with more data). Consistent with the ~2 Mbp
  giant loop (~3000-3200 subvolumes for V = 500 µm³).
- `V_Nucleus = 500 µm³`; DSB yield ≈ 30 /Gy/cell.
- `r_min = 6.5 nm` "as defined in ref. 15" (the LEM III velocity-dependent core).
  The paper quotes it as a constant; we implement the velocity-scaled
  `r_min = β·6.5 nm` (Pfuhl 2020 eq. 1, doi:10.1667/RR15456.1). Measured against
  Fig. 4b's model lines, the literal fixed 6.5 nm core is about three times worse:
  RBE_α bias −27.1% (C) / −29.1% (He) against −9.3% / −9.9% for the velocity core.
  Both were scored on the old, 10/9-high digitization (see below); a uniform
  rescale of the reference does not change which core is worse.

| Fig | cell / ions | α_γ [Gy⁻¹] | β_γ [Gy⁻²] | D_t [Gy] | r_N [µm] |
|-----|-------------|-----------|-----------|----------|----------|
| 2   | CHO, p + C (3-D phantom) | 0.105 | 0.025 | 20   | 5 |
| 3   | CHO, He (SOBP)           | 0.228 | 0.02  | 17.5 | 5 |
| 4a  | V79, p + He (Belli 1998) | 0.129 | 0.049 | 3    | 5 |
| 4b  | HSG, C + He (Furusawa 2000) | 0.313 | 0.0615 | 7.5 | 5 |

The Fig. 4 inputs live in `pyradbio.benchmark.elsasser2010.FIGURE_INPUTS`. Fig. 4b
(HSG) is the key monoenergetic RBE_α/RBE_10 vs LET benchmark: model RBE_α peaks
~6 near ~100-200 keV/µm, RBE_10 below it.

## Fig. 4 data (p. 1181)
`pyradbio/benchmark/elsasser2010.py` holds all 12 series (Fig. 4a: model + data,
p + He, RBE_α only; Fig. 4b: model + data × RBE_α/RBE_10, C + He). They are
extracted from the publisher's PDF **vector paths** (PyMuPDF `get_drawings()`:
tick segments, marker paths classified by fill/stroke, every polyline vertex), so
the axis calibration is measured, not anchored by eye. Prefer this to
rasterize-and-pick for any vector figure. The module docstring carries the
provenance; the CSV row counts are authoritative.

Figure facts worth knowing:
- **Fig. 4b's y axis runs 0-11, not 0-10.** Both panels draw 12 unit-spaced
  left-axis ticks over 147 pt: majors at 0 2 4 6 8 10, minors at odd values, and
  an **unlabelled tick at 11 on the frame edge**. Panel (b): 348.66 pt = 0,
  201.66 pt = 11. The x axis is log, 70 pt per decade (decade majors at
  60.73 / 130.73 / 200.73 / 270.73 pt from LET 1). An earlier hand digitization
  anchored the top on the 9 tick as 10, putting all eight Fig. 4b series a factor
  10/9 high (old/vector ratio 1.1085-1.1113 on data and model lines alike). That
  the *model lines* moved too rules out the alternative reading (data
  renormalised to `α_x = 0.2817`, which is also 0.313 × 9/10). Fig. 4a's
  calibration was correct (He data and both model lines within 0.1%).
- **Fig. 4b has no error bars.** Close-LET pairs (18.5/18.6, 23.6/24.0,
  32.4/33.0, two at 46.0, 53.8/54.0, 70.3/71.0 keV/µm) are distinct Furusawa
  measurements, not bar endpoints; the panel draws 13 filled and 12 open helium
  squares, `y_err` is `None`.
- **Fig. 4a proton data do have bars.** Values sit on the drawn marker centres,
  `y_err` = half-height of the drawn bar. (Averaging markers with bar endpoints
  had put them ~7% high.)
- Fig. 4a helium data: 5 points, LET 20-79 keV/µm, no bars. Proton model line:
  24 points, LET 5.4-39.8 keV/µm.

## Reconstruction status and golden values
`tests/models/lem/test_lem4.py`, default `effect_model="photon_equivalent"` (the
`D_eq` mapping the paper used), with `legacy_trip98_cluster_index_cap=0.95`
(required, see `LEM_IV_OPEN_CHOICES.md` entries 1.1 and 3.1). Bounds are in the
tests; current RMS of RBE:

- `test_golden_lem4_reproduces_elsasser2010_hsg_figure4b` (C, He) vs the paper's
  model lines: carbon RBE_α **0.124**, RBE_10 **0.050**; helium RBE_α **0.175**,
  RBE_10 **0.149**. The test also pins RMS against the Furusawa data.
- `test_golden_lem4_reproduces_elsasser2010_v79_figure4a`: proton RBE_α vs the
  model line **0.24** (clipped to LET ≤ 25), vs Belli 1998 data **0.62**; helium
  vs the model line **0.57**, vs data **0.50**.
- `test_dsb_yield_per_subvolume_matches_elsasser2010_density`:
  `κ_dsb = 30·0.540³/500 ≈ 0.00945 /Gy` (abs 1e-4).

**β method for Fig. 4b.** The paper's coefficients come from the original LEM's
full simulation, so `beta_model="local_dose"` is the textually faithful engine;
`slope_matching` (Kraemer-Scholz) is a Friedrich 2012 device. On the corrected
reference the full simulation *adds* RBE_10 bias against the model lines: carbon
+0.34% → +3.24%, helium +2.16% → +5.16% (`slope_matching` → `local_dose`,
recomputed by `validation/lem4_elsasser2010_friedrich2012.py` on the default
stopping power). RBE_α bias is under 1% on both ions (−0.70% / −0.04%). Faithful
and best-fitting are different questions; see `LEM_IV_OPEN_CHOICES.md` entry 2.5
and "Why RMS cannot pick the engine".

**Magnitude.** The single-central-traversal `(1−e^{−N_1})/D_1` convention was not
the cause of the early magnitude gap. The fix was scoring DSBs over **discrete
540 nm subvolumes** (Pfuhl 2020 eqs. 1-7) instead of pointwise `P(m(r))` (which
over-saturates the `1/r²` core), plus the Friedrich 2015 DSB-yield enhancement
`η(D)` and the `r_c = 6.5 nm` velocity core. V79 carbon peak: ~7 with
`effect_model="globle"`, ~6.6 with `D_eq`, vs the paper's ~6.5. Details in
`LEM_IV_Friedrich2012.md`.

## Known limits
- **Fig. 4a protons above LET ~25 keV/µm:** the paper's proton line levels off
  near ~6.5 while the single-particle model keeps rising past 9 by LET 40, at
  energies nominally inside `SimpleBetheStoppingPower`'s range. A deep-overkill
  limitation of the single-particle model, not a digitization gap; the golden
  clips there.
- **GLOBLE is not scored against Fig. 4b** — the paper computed it with `D_eq`
  (`LEM_IV_OPEN_CHOICES.md` entry 2.7).

## Ruled out
- Impact-parameter cross-section `σ = ∫(1−e^{−E(b)}) 2πb db`, `α = σ/(A·D_1)` as
  the magnitude fix: peak-neutral (offset-track value within <2% of central
  traversal at the peak). Its only real benefit, the low-LET `RBE_α < 1` dip, is
  handled by the linear escaped-energy indirect term in `initial_slope`.
- "HSG RBE_α deficit of ~−10% on both ions": it was the 10/9 axis error above.
- Fixed (not velocity-scaled) 6.5 nm core: see Parameters.

## Related
- `LEM_IV_Friedrich2012.md` — proceedings follow-up (intertrack β, full simulation).
- `LEM_IV_OPEN_CHOICES.md` — free parameters, forks and defects of our LEM IV.
- `LEM_III_Elsasser2008.md` — the `r_min(β)` velocity core and radial dose reused here.
- `LEM_I_ScholzKraft.md` — the "original LEM" macroscopic α method eq. (3) refers to.
- Pfuhl et al. 2022 (doi:10.1002/mp.15343) — LEM IV as a known under-predictor.
