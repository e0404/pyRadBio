# LEM IV — Friedrich et al. 2012 (doi:10.3109/09553002.2011.611213)

Friedrich T, Scholz U, Elsasser T, Durante M, Scholz M, *Int. J. Radiat. Biol.*
88(1-2):103-107 (2012). Proceedings follow-up to the primary LEM IV paper: read
`LEM_IV_Elsasser2010.md` first (eqs. 1-3, the `l_DSB = 540 nm` fit, per-figure
calibration). Free parameters, forks and deliberately unfixed defects are registered in
`LEM_IV_OPEN_CHOICES.md` (cited as "OC <entry>" below).

Companion papers: Pfuhl, Friedrich & Scholz 2020, *Radiat. Res.* 193:130-142,
doi:10.1667/RR15456.1 (eqs. 1-7, Table 3); Friedrich, Durante & Scholz 2012 "GLOBLE",
*Radiat. Res.* 178(5):385-394, doi:10.1667/RR2964.1 (eqs. 2/8/9, Fig. 5B); Friedrich,
Durante & Scholz 2015, *Radiat. Prot. Dosimetry* 166:61 (`eta`); Elsasser, Cunrath,
Kraemer & Scholz 2008b, *New J. Phys.* 10:075005, doi:10.1088/1367-2630/10/7/075005
(radial-dose survey); Pfuhl 2021 PhD thesis ch. 2.3.2 (full simulation); Pfuhl et al.
2022, *Med. Phys.* 49:714-726, doi:10.1002/mp.15343.

## One-paragraph idea
Generalises the LEM from local *dose* to the spatial *DSB damage pattern*. DNA forms
~2 Mbp giant loops; a loop with one DSB holds an isolated DSB (iDSB), with >= 2 a
clustered DSB (cDSB). The cluster index of a pattern, matched to the photon pattern,
defines a photon-equivalent dose `D_eq` from which the ion effect is scaled. The paper
compares the **single-particle approximation** (`alpha_I` from one traversal, `beta_I`
by Kraemer-Scholz slope matching) with a **full simulation** placing many tracks and
fitting LQ to the simulated survival (Fig. 4): `alpha_I` is stated identical; the full
simulation gives larger `beta_I` at intermediate LET, its drop "shifted to higher LET",
even rising "around 30 keV/um for carbon". Clustering is *regional* (Mbp/um), not the
few-bp multiple-damage-site notion of Nikjoo/Ottolenghi/Ward.

## Inputs
Ion `(Z, A, E [MeV/u])` -> LET and `r_max`; photon LQ `(alpha_x, beta_x, D_t)`; nucleus
radius `R_n` (`R_eff` after `effective_size_factor`, OC 1.6); constants below.

## Core equations
Friedrich 2012 / Elsasser 2010:

    C = N_cDSB / (N_iDSB + N_cDSB)            C_g(D_eq) = C_I
    kappa = ln S_ion(D_ion) / ln S_g(D_eq) = [(N_iDSB+N_cDSB)_Ion D_ion] / [(N_iDSB+N_cDSB)_g D_eq]

Pfuhl 2020 eqs. (1)-(7), implemented by `LEM4` in mean-field single-particle form:
(1) `1/r^2` track, `r_min = beta r_c`, `r_c = 6.5 nm`, `r_max = 0.062 E^1.7` um;
(2) `C = n_c/(n_i+n_c)`; (3) `C_g(D_eq) = C_I`; (4) `eta`-enhanced shouldered-to-linear
photon effect `E_g,eq`; (5) `D_t = 4 Gy + 1.1 alpha_g/beta_g` (TRiP98 headers override
it; 0.189/0.013 gives 20, matching the table); (6) `E = E_g,eq (n_i+n_c)_ion /
(n_i+n_c)_g`; (7) `S = e^-E`.

Per loop, `m = kappa_dsb x` (subvolume average of `eta(d) d`): `P_iDSB = m e^-m`,
`P_cDSB = 1 - e^-m (1 + m)`, `P_dmg = 1 - e^-m`; `kappa_dsb = 30 x 0.540^3/500 ~ 0.00945
/Gy` (`DSB_PER_GY_PER_LOOP`). Single-particle slope (`initial_slope`):

    N_1 = kappa (-ln S_g(D_eq)),   kappa = N_total,Ion / (pi R_eff^2 P_dmg(m(D_eq)))
    alpha_I = (1 - e^-N_1)/D_1 + alpha_x (1 - D_1^in/D_1)

`D_1`: LET-based mean single-traversal dose (Elsasser 2010 eq. 8); the second term is the
indirect impact-parameter term (below). Slope matching (Fig. 4 "approximation"):
`beta_I = (s_max - alpha_I)/(2 D_t)`, `s_max = alpha_x + 2 beta_x D_t` (1.200 Fig. 4
cell, 0.864 V79). GLOBLE (eqs. 2, 8, 9): `E = n_i eps_i + n_c eps_c`,
`eps_i = alpha_g/a_DSB`, `eps_c = 2 (N_L beta_g + a_DSB alpha_g)/a_DSB^2`; no `D_eq`/`D_t`.

## Parameters & default values

| quantity | value | source | code |
|---|---|---|---|
| DSB yield `a_DSB` | 30 /Gy/cell | Elsasser 2010 (Stenerlow 2003 / Prise 2001) | `DSB_PER_GY_CELL` |
| loop side / nuclear volume | 540 nm / 500 um^3 | Elsasser 2010 | `SUBVOLUME_SIDE_NM`, `NUCLEUS_VOLUME_UM3` (OC 1.5) |
| geometric loop count | `V/l^3 ~ 3175` | geometry | `MC_GRID_N_LOOPS` |
| GLOBLE `N_L` | 2000 | GLOBLE Fig. 5B | `GLOBLE_N_LOOPS` (OC 1.4) |
| core `r_c`; diffusion `sigma` | 6.5 nm; 4 nm | Pfuhl 2020 eq. 1; Elsasser & Scholz 2007 | `LEM4_RC_NM` |
| SSB->DSB stagger | <= 25 bp | LEM II; Friedrich 2015 | `SSB_STAGGER_BP` |

Cells: **Fig. 3** V79 `0.184 / 0.02 / 17 Gy / R_n 4.7 um`; **Fig. 4** "hypothetical"
`0.25 / 0.05 / 9.5 Gy / 5.0 um` (`effective_size_factor = 1.0`); **Pfuhl Table 3**
photon `0.189 / 0.013 / 20 Gy`.

## Golden values for tests

| test | pins (measured) | paper |
|---|---|---|
| `test_golden_lem4_reproduces_friedrich2012_v79_carbon_figure3` | RMS vs model line `RBE_alpha` < 0.2 (0.09), `RBE_10` < 0.1 (0.03); vs Furusawa 2000 data < 1.2 / < 0.85 (0.99 / 0.69); legacy cap 0.95 | Fig. 3, p. 106 |
| `test_golden_lem4_v79_carbon_rbe_alpha_shape`, `..._rbe10_below_rbe_alpha` | shape only | Fig. 3 |
| `test_golden_lem4_reproduces_friedrich2012_fig4_approximation` | RMS `alpha_I` < 0.05 (0.010), `beta_I` < 0.01 (0.0023) | Fig. 4, p. 107 |
| `test_golden_lem4_local_dose_reproduces_friedrich2012_fig4_full_simulation` | RMS `beta` < 0.010, `alpha` < 0.15; `beta(LET 10) = beta_x` +-5%; `max beta > beta_x`; last < 0.02 `beta_x` | Fig. 4 |
| `test_golden_lem4_discrete_dsb_against_friedrich2012_fig4` | RMS `beta` < 0.008, `alpha` < 0.13; photon limit +-6% | Fig. 4 |
| `test_golden_fig3_rbe10_matches_the_papers_own_single_particle_approximation` | relRMS `slope_matching` < 0.025 (1.70%); `local_dose` > 0.045 and > 2.5x it (6.10%) | Fig. 3 |
| `test_golden_lem4_discrete_cluster_index_matches_pfuhl` | sampled `C_I` 12C 0.70-0.82, 20Ne 0.16-0.23 | Pfuhl Table 3 |
| `test_golden_lem4_analytic_cluster_index_matches_pfuhl_without_sampling` | `xfail(strict=True)`, bounds 0.70-0.88 / 0.15-0.23 unchanged | Pfuhl Table 3 |
| `test_golden_lem4_globle_v79_carbon_peak` | `5.5 < RBE_alpha,max < 7.5` (~7.0) | Elsasser 2010 Fig. 4a |

Digitizations: `pyradbio/benchmark/friedrich2012.py` (Fig. 3 and Fig. 4, all series).
Panels: `validation/lem4_elsasser2010_friedrich2012.py`, MC overlay from
`scripts/lem4_friedrich2012_mc.py`. TRiP98: `validation/trip98_lem4_rbe.py` (external
tables, `PYRADBIO_TRIP98_DIR`).

## Reconstruction status

**Fig. 3 (V79 carbon, p. 106).** The model line peaks at `RBE_alpha` 6.67 near LET 186
keV/um; ours 6.60 (-1%); RMS over LET 8-775 is 0.09 / 0.03 (`RBE_alpha` / `RBE_10`).
This needs the legacy 0.95 cap (uncapped 0.224 / 0.139, from the mean-field
registration bump below; the cap binds from LET 397). **The line is the single-particle
approximation** (p. 106: "This comparison is based on the single particle approximation
as described above"; the Discussion blames the intermediate-LET `RBE_10`
underestimation on the approximate `b`-term). So `slope_matching` agrees by construction
in its `beta` half, `local_dose` sits above it in the direction the paper predicts
(+12.1% at LET 56), and Fig. 3 cannot constrain the full-simulation `beta`.

**Fig. 4 approximation (p. 107).** RMS 0.010 `alpha_I`, 0.0023 `beta_I`. (The `alpha_I`
y-axis is `[0, 1.5]`; a first calibration to `[0, 0.06]` gave a constant ~25x error.)

**Paper defect: the approximation `beta` contradicts its own `alpha` above LET ~50.**
They are tied by slope matching, yet:

| LET | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|
| paper `alpha_ap` | 0.477 | 0.634 | 0.797 | 0.946 | 1.070 |
| paper `beta_ap` | 0.0381 | 0.0292 | 0.0179 | 0.0081 | 0.0005 |
| `(s_max - alpha_ap)/(2 D_t)` | 0.0380 | 0.0298 | 0.0212 | 0.0134 | 0.0068 |

Our `alpha` tracks `alpha_ap` to <1% above LET 30; digitization checked. Above LET ~50
take the baseline from `alpha`, never the plotted `beta_ap`; never decompose
`beta = baseline + enhancement` above LET ~79 (baseline goes negative). The plotted
approximation is `max(slope_matching, 0)` above LET 85.

**Fig. 4 full-simulation `beta_I`.** Window `0.05-0.85 D_t`, three survival conventions:

| LET | 15 | 21 | 24 | 28 | 32 | 40 |
|---|---|---|---|---|---|---|
| paper | 0.0504 | 0.0517 | 0.0522 | 0.0520 | 0.0510 | 0.0462 |
| mean effect `exp(-<E>)` | 0.0508 | 0.0518 | 0.0522 | 0.0524 | 0.0520 | 0.0480 |
| exact `<exp(-E)>` (MC) | 0.0496 | 0.0502 | 0.0499 | 0.0497 | 0.0492 | 0.0451 |
| B2 (`local_dose_overkill`) | 0.0486 | 0.0485 | 0.0483 | 0.0479 | 0.0469 | 0.0422 |

- **Enhancement, LET 15-40: reproduced** under the mean effect (RMS 1.9%, peak LET 28 vs
  24, rise +3.1% vs +3.5%); exact gives +1.2%, B2 none.
- **LET 55 agrees** (exact MC 0.0269 +- 0.0008 vs 0.0269).
- **Late roll-off above LET ~70: OPEN.** Exact MC / paper = 1.154 / 2.105 / 3.861 at LET
  70 / 85 / 100 (0.0125 / 0.0083 / 0.0037 vs 0.0108 / 0.0039 / 0.0010); small absolute
  differences, but monotone and window-independent.
- Validity band of `beta_model="local_dose"`: within ~4% over LET 15-50, 1.18x at 65,
  1.31x at 70; worth at most +0.61% of `RBE_10`. The default stays `slope_matching`.
- The reported `alpha_I` is the analytic `initial_slope` (RMS 0.0297 vs the paper's
  full-simulation `alpha_I`); `alpha` read off the full-simulation fit folds overkill in
  above LET ~60 (1.88 vs 1.56 at LET 130).

**Window caveat.** Fig. 4's legend reads "Full Simulation, S>0.01"; implemented as
`fit_survival_bound` (opt-in), that window reproduces the paper *worse* (`beta`/paper
1.312 at LET 50 vs 1.033 on `0.05-0.85 D_t`), and the paper's curve fits no single survival bound (OC 1.2). Every
"reproduces Fig. 4" here refers to a free window tuned on this curve, not independent
validation. Three successive Fig. 4 "defects" dissolved into protocol choices (window
twice, convention once): vary the protocol axes before adding an engine.

**Pfuhl 2020 Table 3 anchors** (single traversal, photon 0.189/0.013/20): 12C 2.5 MeV/u
LET 437: `C_I = 0.80`, `alpha_I = 0.800`; 20Ne 31 MeV/u LET 183: `C_I = 0.19`,
`alpha_I = 0.822`. The paper averages the *ratio* over realisations with a finite DSB
count (~30 `D_1`); our mean field (ratio of averages) gives 0.991 / ~0.31; sampled
`discrete_cluster_index` 0.745 / 0.194; marginalised `analytic_discrete_cluster_index`
0.851 / 0.268. A random track-to-lattice registration is required (corner binning gives
Ne 0.11, centre binning over-counts). A single-track discrete MC gives `alpha_I` 0.72.

## Implementation notes (`models/lem/lem4.py`, `_localdose.py`, `_discrete.py`, `_intertrack_kernel.py`)

### How the magnitude was resolved
Not by the impact-parameter cross-section (it fixes only the low-LET dip; an earlier
"2.81/20.4" came from the plain photon-log-kill cross-section, i.e. LEM III). The levers:
(a) DSBs scored into **discrete 540 nm subvolumes** with the `eta`-enhanced yield
integrated over each (Pfuhl 2020), instead of `P(m(r))` at the pointwise dose, which
over-saturates the core and pins `C_I ~ 0.13`; (b) `r_c = 6.5 nm`; (c) the Poisson slope
`(1 - e^-N_1)/D_1`. V79 carbon went from ~2.1 to a peak ~5 at ~200 keV/um. The ~1.2-1.4x V79 peak deficit estimated then predates the
digitization; the digitized residual is -1%.

### Impact-parameter cross-section (indirect term implemented; direct deferred)
"Method of the original LEM" (Scholz 1997): `sigma = int (1 - e^{-E(b)}) 2 pi b db`,
`alpha = sigma/(A_eff D_1)`; we score the central traversal. **Indirect term,
implemented:** for `b > R_eff` a track acts as a weak photon field; summed,
`alpha_x (1 - D_1^in/D_1)`; exact at low LET, zero when `r_max <= R_eff`; it lifts the
entrance-channel `RBE_alpha` from below 1 to about 1. **Direct term ("Route A"), tested, not
shipped:** `b = 0` reproduces the central pattern exactly and the integral is
peak-neutral (LET 26-530 within ~1-2%), but once `r_max ~ 62 nm << 540 nm` a track
saturates one loop at *every* `b < R_eff`; uncapped it overshoots the TRiP98 tables
severely at 1 MeV/u. It is not the deep-overkill fix.

### The track core: `r_c = 6.5 nm` is a load-bearing anachronism
Friedrich 2012 p. 104 cites its radial dose to Elsasser 2008b, a survey (sec. 3.1.5:
`CR10` LEM I, `CR03` LEM II, `EDR40` = `r_c` 40 nm LEM III, `EDR11`, KC/MC) that pins no
LEM IV core. We ship Pfuhl 2020 eq. 1. Core and `D_t` are degenerate (2008b sec. 4), so
each was re-fitted in `D_t` against Fig. 4's full-simulation `alpha`: shipped core best
`D_t` 9.5 (Pfuhl eq. 5's own value), `alpha` RMS 0.0276; `EDR40` best `D_t` >= 40, RMS
0.2057 (34-38% low); `CR03` best 6.0 (2008b's HSG value), RMS 0.1286 (47-53% high).
Whatever the paper ran behaves like a sharper core than it cites. `beta` barely
discriminates (RMS 0.0085-0.0095): the core shifts `C_I`, the enhancement is its slope.

### Full-simulation `beta`: the local-dose convolution (`beta_model="local_dose"`)
Elsasser & Scholz 2007's local-effect picture. Tracks are Poisson with areal intensity
`phi = D / I_track`; the local dose is compound Poisson with jumps = the
subvolume-averaged single-track dose `g(rho)`; `f_D(d)` from the exact characteristic
function summed over radial nodes, one FFT; `N_ion(D) = N_sv <rho(d)>`; LQ fit over
`fit_doses_Gy()`. The infinite track plane folds in the far penumbra analytically.
Deterministic, ~9 s/ion.
- **The DSB channel sees `eta(d) d`, not `d`** (`eta` acts on the pointwise core dose;
  averaging plain dose first loses ~2x in `C_I` at 100 keV/um). `phi` uses plain dose.
- **Theorem:** any response that is a function of the *physical* local dose has
  `beta_I <= beta_x` (`Var` is linear in `D`, so it lands in `alpha`; measured
  `Var/D = 4.6006`, `beta = 0.050000` at every LET with `D_t` out of reach).
- `local_dose_kernel`: `"cluster_index"` (default); `"photon_effect"`
  (`rho = -ln S_x(d)`, identical to a per-subvolume `D_eq` mapping); `"globle"` cannot
  fall (single-track subvolume dose <= ~35 Gy at 100 keV/um, 55 Gy before delta escape
  being the energy bound, so `m <= 0.33` and the `D^2` term is `beta_x`; pinned by
  `test_local_dose_globle_kernel_stays_at_beta_x_at_every_let`); `"discrete_cluster_index"`.

### The structural no-go: the intertrack slope is pinned by dose conservation
`<h> = D`, `<h^2> = D^2 + Var` with `Var` linear in `D`, so `C_I ~ (kappa/2)[D + Var/D]`:
LET cancels from the slope. Measured `dC_I/dD / (kappa/2)` = 1.001, 1.012, 1.026, 1.046,
1.058, 1.041 at LET 20/30/40/55/70/100 while traversals/Gy fall 24.5 -> 4.9. In any
formulation where subvolume doses have mean `D` and `C_I = <P_cDSB>/<P_dmg>`, the
enhancement cannot roll off. This bounds the open late roll-off.

### The discrete-DSB engine (`beta_model="discrete_dsb"`)
`Poisson(D/D_1)` traversing tracks on the `R_eff` disk with discrete DSBs from their
profile, plus the escaped dose `D (1 - f_ret)` as a uniform Poisson background (`f_ret`
from the two-disk overlap; dose-conserving, no free parameter). **Discreteness is
invisible to `C_I`:** a Poisson total split over loops gives independent Poisson loop
counts, so occupancy is compound Poisson, `G(z) = E[e^{m(z-1)}]`, `P(N=0) = E[e^-m]`,
`P(N=1) = E[m e^-m]` — the continuum expectations. `C_I` never sees track provenance;
track pairs `(D/D_1)^2` and per-pair coincidence `~ D_1^2` cancel.
- `discrete_convention="mean_counts"` reproduces the convolution (`beta` 0.0388 /
  0.0183 / 0.0094 / 0.0049 vs 0.0387 / 0.0183 / 0.0094 / 0.0049 at LET 55/70/85/100).
  `"per_realisation"` (default) differs by -2 to -4% at LET 10-40 and +9/+23/+55% at
  70/85/100 (+3.4% bias even in the photon limit); with the different track population,
  the two engines are not identical as shipped.
- Photon limit 5.5% low (`beta(LET 10)` 0.0472), grid-converged, so physical; the deficit
  fits no simple form. Ships as a cross-check.
- Deterministic discrete `C_I` (`_discrete.expected_damaged_ratio`): given the impact
  parameter, loop counts are independent `Poisson(mu_j)`; with `A = #{N_j >= 2}`,
  `B = #{N_j >= 1}`, `q_j = e^-mu_j`, `p2_j = 1 - e^-mu_j (1 + mu_j)`:

      E[A/B | B>=1] = int_0^1 G(t) sum_j p2_j/(q_j + (1-q_j) t) dt / (1 - prod_j q_j),
      G(t) = prod_k (q_k + (1-q_k) t)          (from 1/B = int_0^1 t^(B-1) dt)

  Do **not** divide the per-loop mean by `n_z` (the track is uniform along the beam).

### Survival convention and the B2 correction
`exp(-<E>)` vs `<exp(-E)>` brackets the paper; the enhancement's amplitude is
convention-sensitive, its location is not (OC 2.1; B2: OC 2.6).
- **B2 (`local_dose_overkill=True`) is compensation.** `local_dose_mc_beta_pair` gives
  both conventions from the same realisations; 12 seeds x 400: B2 drop / exact drop =
  1.87 (LET 55), 4.95 (70), -0.90 (100). Convolution + exact drop gives `beta` 0.0346 /
  0.0172 / 0.0075, within 2-5% of `discrete_survival="stochastic"` (0.0364 / 0.0176 /
  0.0079). Pinned by `test_b2_overcorrects_against_the_exact_stochastic_survival`.
- On the production kernel the overkill reduction of `beta` is 6/12/20/29/47% at LET
  25/40/55/70/100. Stochastic `beta` turns back up past ~100 keV/um (deep-overkill
  convexity, physical; the term B2 drops). Neither is a default; never pick on RMS.

### Intertrack-cDSB Monte-Carlo for beta_I (`beta_model="intertrack_mc"`, diagnostic)
Tracks over `R_sim = R_eff + r_max` (`Poisson((D/D_1)(R_sim/R_eff)^2)`) place discrete
DSBs (radius from the inverse CDF of `eta(D(r)) D(r) r`) into one shared 540 nm grid
masked to the disk; GLOBLE per-loop effect; `S = exp(-<E>)`; common random numbers via a
per-track arrival dose; numpy reference + numba `prange` (not bit-reproducible).
- `eps_c` uses `N_L = MC_GRID_N_LOOPS`, not `globle_n_loops`: the photon limit recovers
  `beta_x N_L(eps_c)/N_grid`; 2000/3175 undershot the plateau ~40%.
- Sample the pdf to `r_max` and mask once (clipping *and* masking breaks the photon limit).
  The uniform-field limit matches the analytic GLOBLE `N_lethal` to ~1% at 10-50 Gy
  (GLOBLE Table 1 discreteness check).
- **Its `beta` is flat at `beta_x`:** over one common window, 11 LET points 10-100,
  weighted mean 0.0492 +- 0.0011, `chi2` 10.9 / 10 dof — the near-miss background
  clusters like a photon field. An apparent LET rise was the LET-dependent adaptive
  window. Pinned by `test_intertrack_beta_is_flat_at_beta_x_over_a_common_fit_window`.
- Errors: `intertrack_mc_beta_jackknife` (batch jackknife, `IntertrackSurvival`,
  `base._delete_one_jackknife`; validated against seed scatter) and
  `intertrack_mc_beta_to_precision` (capped, warns). `fit_alpha_beta` no longer floors
  `beta` at 0 (the clamp inflated the LET-130 mean 0.069 -> 0.148).
- `mc_effect_model="photon_equivalent"` humps to 3-17x the paper (0.1790 at LET 70):
  a small-count bias of per-realisation `C_I` from integer counts (2.80x at ~12 DSBs,
  1.29x at ~60, 1.07x at ~180). Expected loop counts from the sampled field
  (`local_dose_mc_beta`, `local_dose_mc_alpha_beta`) remove it.
- Cost ~ `(R_sim/R_eff)^2` (~7e7 tracks per realisation at LET 10). The explicit-track
  local-dose MC replaces far tracks by their mean beyond `local_dose_mc_impact_cut`
  (OC 1.3); the cut must also be a domain fraction (floor 0.15 `r_sim`): a fixed 40 um
  cut at LET 13-18 biased the baseline ~0.0004 low, the size of the enhancement.

### Low energy / deep overkill against the TRiP98 tables

#### What exactly saturates, and the cap reconciliation
Below ~2.5 MeV/u carbon `N_dmg` pins at one subvolume column, **0.2916 um^2 = 0.540^2**
(LET 480-768), with `N_cDSB` equal; it unpins at 3.16 MeV/u. `N_1` freezes and
`RBE_alpha x LET` is four-figure flat over 0.79-2.0 MeV/u: a spurious minimum near
0.3 MeV/u where the TRiP98 tables keep climbing. `effect_model="globle"` shows the same
`1/LET` signature and is worse. Every `C_I` -> effect path must clip via
`_cluster_index_bounds` (`test_every_cluster_index_path_honours_the_overkill_cap`).
Below ~1 MeV/u carbon's CSDA range is comparable to the 10 um nucleus, so constant-LET
traversal is unphysical at the lowest tabulated energies; 1-2.5 MeV/u is the fair target.

#### TRiP's alpha is FITTED, not the D -> 0 slope
TRiP's tables come from the full simulation (`-iter 50000`; Pfuhl 2021 ch. 2.3.2: hit
classes weighted "according to the requested irradiation dose level"), so their `alpha`
is an LQ fit over a finite window. Under overkill `-ln S(D)` is concave: a negative
`beta` lifts `alpha`. Shipped as `initial_slope_model="full_simulation"` (four fixed
seeds). Against the TRiP98 carbon curves it roughly halves the sub-MeV error of
`cross_section` and is worse above ~1.2 MeV/u (spinal cord) / ~6 MeV/u (chordoma);
there the analytic slope is better and 30x cheaper (1.8 s at 1 MeV/u, 62 s at 50;
`_FULLSIM_DOMAIN_WARN`). The lift is cell-dependent (V79 1.11x vs spinal cord 1.43x at
1 MeV/u). A separate offset between the two cell files is open. Trap found on the way: consumers integrated only
to `r_max` while diffusion moves dose past it (4.6% of LET recovered at 0.100 MeV/u);
models now expose `r_support_um = r_max + 8 sigma`.

#### The 0.95 cluster-index cap
In no paper (all state the mapping unbounded); tuned on TRiP98 carbon (OC 1.1). Its
rationale is false: `d log D_eq / d log(1 - C_I)` is 0.27 at the cap and 0.03 deep in
overkill; `D_eq` is 453 Gy at 0.95 and 2991 Gy at the float64 limit (`C_g` = 1.0 at 3173 Gy). `_CLUSTER_INDEX_MAX = 1 - 1e-15` is the derived
guard; `legacy_trip98_cluster_index_cap` restores 0.95. The cap had masked a quadrature
artefact: the penumbra area as (radial integral) - (fine-grid sum) *was* the whole
sub-MeV `1 - C_I`; `_outside_square_fraction` now integrates the complement
(`(4/pi) acos(s/r)` for `s < r < s sqrt 2`). Fitting in one loop does not empty the
pattern of iDSB: at 1 MeV/u protons have `C_I` 0.1208, helium 0.6579, carbon
`1 - 9.0e-11`; track density, not geometry, sets saturation.

#### What actually fixes it: match `m`, not `C_I`
A photon dose gives every loop `m_g = kappa_dsb eta(D) D`, so `C_g` is monotone in `m_g`
alone; a single-loop ion pattern is one `m`. Hence `C_g(D_eq) = C_I <=> m_g(D_eq) = m`
(Pfuhl eq. 3 either way), but only the second avoids cancellation:
`1 - C_I = m e^-m/(1 - e^-m)` underflows near `m ~ 37`. Routes agree to <= 5.3e-13
relative up to `m = 14.25` (2.8e-8 at 26.4); the radial `m` matches the 2-D pattern's to
0.01%. `LEM4.mean_field_deq`
matches `m` below 2.206 MeV/u (`pattern_is_single_loop`) and inverts `C_g` above; the seam
is continuous to 3e-5.

#### What this costs against TRiP98
Exact (no cap), below 6 MeV/u every ion from H to C overshoots the tables, increasingly
with Z (for H the cap never binds); with the legacy cap carbon lands below them. The
damage is confined below 1 MeV/u, where `m` reaches 26-39 and `D_eq` 2200-4200 Gy; over
1-10 MeV/u every ion agrees well. The carbon agreement under the cap is the cap's doing;
published-figure goldens set it, so those matches are contingent on a tuned constant.

#### Why `full_simulation` does not track TRiP98 at high Z
Fine for H/He/Li sub-MeV, divergent for Be/B/C. Refuted hypothesis: the
`cluster_index` kernel's ratio-of-averages leaks mean-field over-clustering.
`local_dose_kernel="discrete_cluster_index"` (average of ratios) makes Li-C sub-MeV 2-6x
*worse* (two saturated columns give `A/B ~ 1`). The per-realisation whole-nucleus `C_I`
is a different object from the per-traversal one LEM IV is calibrated on. Open
(`# UNVERIFIED`): TRiP's dose window, the realisation count against `-iter 50000`, and
whether TRiP fits one LQ across a window at all.

#### The survival convention is NOT the explanation either
`mean_survival` cuts the 12C sub-MeV error by compensation: its `1/D_1` ceiling lies
far above the tables. Both cell files share one nucleus radius, so
`alpha_spinal/alpha_chordoma` at a fixed seed isolates the convention:
over 25 points (H-C, 0.126-5 MeV/u) `mean_effect` median 1.0019 (0.985-1.029),
`mean_survival` 1.1136 (0.995-1.725). TRiP behaves like `e^-<E>` (OC 2.1).

#### Three cluster-index estimators, three energy bands
`cluster_index_model`: `"mean_field"` (default), `"discrete"` (analytic), `"discrete_mc"`
(sampled, 16000 realisations), `"trip98_composite"` (OC 2.3; rationale in the
`LEM4ClusterIndexModel` comment). Judge noise on `1 - C_I`, never `C_I`. Against TRiP98
`discrete_mc` is best below 0.5 MeV/u, `discrete` at 0.5-2 MeV/u
and `mean_field` at 2-6 MeV/u. The estimator that reproduces Pfuhl's anchors is worst at
2-6 MeV/u, where the over-clustering mean field fits best: its +2.0 log-odds bias is
absorbed downstream. Routing the RBE
path through the (pre-fix) discrete `C_I` took Fig. 3 V79 RMS 0.092 / 0.053 -> 1.123 /
0.598 and the peak 6.60 -> 5.07: **change the effect mapping and the cluster index
together or not at all**. Protons overshoot below 0.5 MeV/u under all three. TRiP's
Li-6 curve has a step at its lowest energies in both files; treat those points as suspect.

#### The giant-loop lattice registration defect
`analytic_discrete_cluster_index` placed tracks at `(b, 0)`, leaving the y-registration to
the parity of `subvolume_centres`: `1 - C_I` (12C 0.126 MeV/u) pinned 0.0631 / **0.0148**
/ 0.0618 vs phase-averaged 0.0524 / 0.0562 / 0.0557 at `R_n` 4.7 / 5.0 / 5.3 um; 5.0 um,
the standard LEM nucleus, is the singular one. **Fixed** there (one R2 low-discrepancy lattice offset
per impact parameter; positions weighted by `P(N_dmg >= 1)`, worth ~24% of `1 - C_I`):
analytic and sampled now agree to 1-7% (12C 2.5 and 10, 20Ne 31, 1H 1 MeV/u). They place
`eta` differently (placement vs count, 1.0-1.9x on `1 - C_I`; OC 2.4). Below ~1 MeV/u the
analytic stays 1.4-6.1x above the MC because `box_averaged_profile` smears the track
~2.5x (OC 3.2). Post-fix `discrete / mean_field` is nowhere near 1 (12C 10 MeV/u -6.8%),
so a 10 MeV/u continuity seam is gone.

**The same defect survives in the mean field** and is the local maximum of `RBE(LET)` at
~430-440 keV/um (OC 3.1): with the track dead-centre, `1 - C_I` is the boundary sliver,
which vanishes once `r_max + 8 sigma` < 0.270 um (2.206 MeV/u). 12C V79, pinned vs
phase-averaged: `1 - C_I` 1.54e-1 / 2.13e-1 at 4.0 MeV/u, 9.78e-3 / 1.39e-1 at 2.5,
5.36e-8 / 7.97e-2 at 1.5; `D_eq` 309 / 264, 639 / 323, 1730 / 396 Gy. Not applied:
phase-averaging moves every `C_I` (0.846 -> 0.787 at 12C 4 MeV/u) and every golden, and
the over-clustering is load-bearing. The legacy cap removes the bump on Fig. 3; no Fig. 4
`beta_I` result depends on it.

#### Why the distal tip can't be cleanly reproduced
TRiP uses the `D_eq` mapping (Elsasser 2010 eq. 3 = Pfuhl eq. 6); the `C_I` its tables
imply is smooth and never reaches 1. The sampled discrete `C_I` fed through the chain
improves the tip modestly but degrades 2.5-10 MeV/u: the tables follow the continuous
approximation. GLOBLE (`effect_model="globle"`) is the developers' bounded, cap-free
alternative but diverges from the tables (OC 2.7).

### GLOBLE `N_L` calibration and the V79 carbon peak
The GLOBLE paper is photon-only (ion use stated as an assumption for future work). It uses
`N_L = S_G/S_L = 6e9 bp / 2 Mbp = 3000` only for illustrative curves (Figs. 2-4), and
`N_L ~ 2000` as the best fit to >150 photon curves (Fig. 5B; Fertil & Malaise 1985, Steel
et al. 1989, Friedrich et al. 2012). Default 2000 moves the V79 carbon peak from 8.52
(+31%) to ~7.0 (+8%); `N_L` is the dominant lever (peak 6.1-9.0 over 1500-3175).

### The LQ fit window
`_LOCALDOSE_DOSE_FRACTIONS` = 0.05-0.85 `D_t` (OC 1.2), replacing 0.4-3.0 Gy, which had no
literature basis (the quoted "at lower doses around 2-3 Gy the approximation is
sufficiently accurate" concerns the approximation, not a fit range) and whose
"insensitivity" was measured at LET 70, where `beta` has already collapsed. Neither
Friedrich 2012 nor TRiP98 states a window (`# UNVERIFIED`). Chosen on RMS
`ln(ours/paper)`: 0.096 (excluding the registration-hypersensitive LET 77.5) vs 0.312 for
a window like the old one. Relative to `D_t` (where the photon curve goes linear); fixed
fractions, never LET-dependent. Against TRiP98 it is better at 1-10 MeV/u for every ion
and below 1 MeV/u for Be, B and C.

**Two things it exposed.** (1) At deep overkill the fitted `beta` is meaningless under
any window: 12C 1 MeV/u V79 `E/D` falls 0.704 -> 0.545 (1.3-5.4 Gy) then climbs to 0.783
(14.4 Gy); `alpha` is robust (0.6312 vs 0.6172). (2) Against TRiP98 the two fitted
kernels differ widely over 1-10 MeV/u (`discrete_cluster_index` errs several times more
than `cluster_index`): a bound on the near-unity ratio `u` says nothing about `C_I = 1 - u` (uniform field, `D_t` 22 Gy:
-2.98% in `u` is +31.8% in `C_I` at 18.7 Gy).

### Numerics worth not re-deriving (`_localdose.py`)
- No Gaussian/Gamma for the many small penumbra jumps: the negative tail wraps around the
  periodic FFT grid (mean 148 Gy for a true 0.4 Gy). Sum the exact CF directly.
- `g(rho)` must conserve dose-area: `box_averaged_profile` integrates in polar coordinates
  with an exact circle-in-square measure (crude box quadrature +35% on the core; skipping
  the direction average -8%).
- The point mass at zero must not wrap: the grid spans twice the support, upper half folded
  to `d = 0`; `CompoundPoissonSolver.wrapped_mass` (~1e-7) exposes it.
- Never sample the pointwise `1/r^2` dose at subvolume centres: spurious collapse (0.036 ->
  -0.037 over LET 55-100) where subvolume averaging gives 0.044 -> 0.028.
- Defaults bound at definition time make patching a module global a vacuous convergence
  test; pass the step explicitly. A ragged loop lattice must normalise by its own count.

## Known limits
- **Late roll-off above LET ~70 (open).** By the no-go, nothing that conserves dose and
  scores loop occupancy rolls the enhancement off. Per-loop lethality `eps(k)` is not
  identifiable from photon data: a pure LQ photon curve forces `eps(k) = eps_i k +
  b k(k-1)`, `b = beta_x N_L/a_DSB^2` (GLOBLE eq. 2 is this truncated at `k = 2`); inverting
  LQ + `D_t` for `eps(k)` has condition number 3.9e18; `D_t` (0.09 DSB/loop at 9.5 Gy) and
  clustering (onset `m = 1`, ~106 Gy) are disjoint dose regimes. The `D_eq` family
  collapses `beta` but under-produces `alpha`; GLOBLE keeps `beta ~ beta_x` but
  over-produces `alpha`. Closing it needs ion-data calibration or an `eps(k > 2)`
  assumption; one published full-simulation panel exists (one cell, one ion).
- Mean-field over-clustering and loop-centre registration; published-figure matches rest
  on compensation and the legacy cap (OC 1.1, 3.1).
- Sub-MeV TRiP98 agreement is poor for heavy ions under every estimator.
- Loop size, `V`, `r_c` assumed cell- and LET-independent (paper limitation; OC 1.5).
- Single-particle `RBE_alpha` dips slightly below 1 under ~30 keV/um (escaped delta
  energy still counted in `D_1`).

## Ruled out
- *LET shift of the Fig. 4 residual:* best LET scale 1.198 (`beta`), 0.967 (`alpha`, no
  gain) — `alpha` unshifted, so no LET-scale error.
- *LET smearing:* integrated enhancements differ 1.92x; a Gaussian LET kernel leaves the
  peak 22% too tall.
- *Core width:* `sigma` 2 nm breaks the photon limit; 7-14 nm right-shifts the hump.
- *Registration-free connected-component clustering:* LET-independent ~1.4x `C_I` lift.
- *Constant `C_I` scaling:* `f = 1.3` sends `beta(LET 10)` 0.0503 -> 0.0662.
- *Continuum restricted to traversing tracks + uniform background:* `dC_I/dD` identical to
  3-4 significant figures.
- *Damping `C_I`'s dose-dependence by `lambda`:* Fig. 4 needs `lambda` 0.294 at LET 70, a
  sub-photon clustering rate the no-go forbids.
- *`alpha/s_max` as mechanism:* pinning `alpha` to the paper made `beta` worse and failed
  in sign (old window).
- *Clipping clustering at `r_eff`:* `beta` RMS 0.0046 vs 0.0063 but `alpha` 0.095 vs 0.077
  and a free radius (old window).
- *Shrinking `l_DSB`:* `beta(LET 10)` 0.039 at 135 nm.
- *"Full-simulation `beta` = slope matching on another `alpha`":* implied 0.209 vs 0.417
  at LET 25.
- *Intertrack MC without domain extension:* 0.020 vs 0.052 at LET 20.
- *Bounded per-loop single-particle effect for deep overkill:* undershoots the TRiP98
  tables at 1 MeV/u and adds a spurious mid-LET hump.
- *Inverting Fig. 3's `RBE_10` for a second `beta_I`:* a 2% `alpha` error becomes ~85% at
  LET 70, and the line is the approximation anyway.
