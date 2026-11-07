# Microdosimetry — lineal-energy engines and the MKM `d(y)` routes

How `pyradbio.micro` turns an ion into single-event lineal-energy spectra and
specific energies, and how the MKM family consumes them (mMKM, MCF MKM, SMK,
OSMK, IMK). Not a single-paper summary; each claim names its source.

## Sources

- Kellerer 1985, "Fundamentals of microdosimetry": chord distributions,
  track-segment deposit `eps = LET * l`, impact-parameter method.
- Xapsos et al. 1994, *Radiat. Res.* 137, 152: log-normal energy-loss straggling.
- Kase et al. 2006, *Radiat. Res.* 166:629, doi:10.1667/RR0536.1: mMKM, `y*`.
- Sato, Watanabe & Niita 2006, *Radiat. Prot. Dosim.* 122:41-45,
  doi:10.1093/rpd/ncl407: the analytical microdosimetric function (AMF), Eq 7-8.
- Sato, Watanabe, Sihver & Niita 2012, *Int. J. Radiat. Biol.* 88:143-150,
  doi:10.3109/09553002.2011.611216: the same form in `eps` (Eq 1), PHITS `[T-SED]`.
- Sato et al. 2023, *Phys. Med. Biol.*, doi:10.1088/1361-6560/ace14c: the newer
  AMF (not read).
- Sato & Furusawa 2012, *Radiat. Res.* 178:341-356, doi:10.1667/RR2842.1,
  eqs. (24)-(28): multi-event densities.
- Parisi, Beltran & Furutani 2022a, *Int. J. Mol. Sci.* 23:12491,
  doi:10.3390/ijms232012491; Parisi, Furutani, Sato & Beltran 2022c, *Phys. Med.
  Biol.* 67:185013, doi:10.1088/1361-6560/ac7375; Parisi et al. 2024b, *Quantum
  Beam Sci.* 8:18, doi:10.3390/qubs8030018; Parisi, Furutani & Beltran 2022b,
  *Phys. Med. Biol.* 67:095014: MCF MKM.
- Hartzell et al. 2025, *Phys. Med. Biol.* 70:105010,
  doi:10.1088/1361-6560/adcfec: TOPAS AMF; eqs. 6-12, HSG parameters (sec. 2.3);
  digitized figs. 4 and 7 (`pyradbio.micro.hartzell2025`).
- DeCunha et al. 2025, *Med. Phys.* 52(5):3471-3480, doi:10.1002/mp.17561:
  Geant4-DNA proton `y` library, 1 um sphere (`micro.library`).
- Electron range: Cole 1969, *Radiat. Res.* 38:7-33; Waligorski, Hamm & Katz 1986,
  doi:10.1016/1359-0189(86)90074-6; NIST ESTAR.

## Conventions

- `f1(y)` normalised on `y`; `d1(y) = y f1(y)/y_F`; moments `y_F`, `y_D`.
- `y* = y0^2 ∫ (1 - exp(-(y/y0)^2)) f dy / ∫ y f dy` (Kase 2006) =
  `Spectrum.y_star(y0)`.
- `z(y) = y / (rho pi r^2)` (keV/um -> Gy via `KEVUM_PER_GY_UM2_GCM3`), valid for
  a sphere (`l_bar = 4r/3`) and for an axis-parallel cylinder of radius `r`,
  length `2r`. The *distributions* differ between the two (below).
- Sphere: `f(l) = 2l/d^2`, `l_bar = 2d/3`. Cylinder (`CylindricalSite`): chords
  sampled under mu-randomness; exact checks `l_bar = 4V/S = 2dh/(d+2h)`, longest
  chord `sqrt(d^2+h^2)`.
- Large-site geometric limit: `y_F -> LET`, `y_D -> (9/8) LET` (also Hartzell
  sec. 4's low-energy anchor).

## Routes

| route | module | site | output | used by |
| --- | --- | --- | --- | --- |
| impact parameter | `micro/impact_parameter.py` | disk (default) or sphere | moments `z_F`, `z_D`, `z*_1D`; no spectrum | `NativeMKMBackend`; mMKM/SMK/OSMK/IMK `alpha_beta(ion)` with no engine injected |
| amorphous track | `AmorphousTrackEngine` (+ `DeltaRayMonteCarlo`) | sphere | `f1(y)` | default engine of the `d(y)` route (`_default_micro_engine`, domain sphere `2 r_d`); MCF MKM always |
| analytic | `AnalyticSpectrumEngine` | any site (delta tail: sphere) | `f1(y)` | injected |
| Sato AMF | `SatoLinealEnergyEngine` | sphere | `f1(y)` | injected; IMK paper route (`sato-AMF`) |
| tabulated | `TabulatedLinealEnergySpectrum`, `DeCunhaLinealEnergyLibrary` | given | `f1(y)` | comparison |
| multi-event | `micro/multi_event.py` | any | `f(z; D)` | stochastic MK survival `S = ∫ S_n f_n dz_n` |

- **Disk vs sphere.** The MKM domain/nucleus are cylinders with tracks along the
  axis (Inaniwa 2010; Inaniwa & Kanematsu 2018 sec. 3.2), `z_1` = disk area-mean
  of `D(r)`; this is pyMKM's `SpecificEnergy` geometry. The sphere overshoots
  `z*_1D` at high energy (numbers: `SMK_Inaniwa2018.md`, implementation notes).
- An injected engine fixes its own geometry and overrides `site_geometry`;
  supplying `d(y)` directly bypasses both.
- MCF MKM stays on the spherical `d(y)` route: its `c(y)` couples domain and
  nucleus **per event** through the shared `y` (`z_n/z_d = (r_d/R_n)^2`), which
  independent per-site `z(b)` profiles do not reproduce. `site_geometry` has no
  effect on it.
- Multi-event: the Poisson sum of convolutions (Sato & Furusawa eqs. 26-27) is a
  compound Poisson distribution, `Σ_k P(lambda;k) F_1^k = exp(lambda(F_1 - 1))`,
  evaluated with one FFT pair at any `lambda` (no iteration cap, no separate
  high-dose branch; the `k = 0` term is automatic).

## Analytic engine (Xapsos-Kellerer)

    f1(y) = l_bar ∫_0^d f_chord(l) p_eps(y l_bar | l) dl

`p_eps` log-normal matched to the Bohr mean and variance (default; `"gaussian"`
leaks to negative energies in small sites and biases `y_F` high). Straggling is
mean-preserving, so `y_F = LET` exactly; `y_D` grows with the variance. Bohr
variance is non-relativistic (sub-percent at therapy energies).

- `restricted=True` (sites smaller than the delta-ray range): mean = restricted
  LET `L_Delta = LET - Delta_L` with the Berger-Seltzer offset
  `Delta_L = (K/2)(Z/A) rho (z_eff^2/beta^2)[ln(T_max/T_cut) - beta^2(1 - T_cut/T_max)]`;
  Bohr variance scaled by `T_cut/T_max`; `R(T_cut) = d` with
  `R(T) = r_max (T/T_max)^delta`, `delta = 1.7` (Scholz-Kraft). Inactive for
  `r_max <= d`.
- `include_delta_tail=True` (spheres): an analytic delta-ray "toucher" term,
  Scholz-Kraft `1/r^2` penumbra normalised to LET on `[r_min, r_max]`
  (`r_min = 10 nm`), each impact annulus weighted by area **and** by
  `1 - exp(-lambda(b))`, `lambda = eps(b)/<w>`; deposits below one quantum
  `T_min = I` dropped (floor `T_min/l_bar`). Moves C/O/Ne `y_D` at 10 MeV/u from
  ~1.28× to ~0.94× the Hartzell AMF and cuts the MCF MKM V79-carbon `RBE_alpha`
  over-prediction by ~10-25% (most at mid energies). Off by default; tuned for
  sub-um domains and high-LET heavy ions, coarse for low-LET ions in larger sites.

## Amorphous-track engine

Impact points uniform in the plane: `b` pdf `2b/b_max^2`, `b_max = R_s + r_max`;
`eps(b) = C ∫_V D(r) dV`, `y = eps/l_bar`. Resolves the toucher tail and
restricted-LET behaviour automatically; deterministic (no straggling). The
smeared mean dose gives every far track a tiny deposit, so the low-`y` tail is
over-populated and `y_F` biased low for `R_s << r_max`; `y_D` is the robust
moment.

Small-site construction (fixes for the former `y_D ≈ 0` collapse at
`r_d ≈ 0.26 um`, e.g. 12C at 10 MeV/u, which drove the MCF MKM V79-carbon
geo-ratio to 0.20):

1. a **core-resolving** impact grid (a linear grid to `b_max` skips the crosser
   core once `r_max >> R_s`);
2. `f1(y)` as a **frequency histogram** of annulus weights, not
   `pdf/|dy/db|` (whose integrable singularity at `y -> 0` collapsed the moments;
   `eps(b)` is also mildly non-monotone), with `y_F`/`y_D` overridden by the exact
   impact-parameter moments;
3. `_sample_impact` uses a cumulative trapezoid on the actual spacing (a
   `cumsum` assumed a uniform grid).

`DeltaRayMonteCarlo` (`stochastic=`) resamples `eps(b)` as a compound Poisson
sum of `1/T^2` collisions on `[T_min = I = 78 eV, T_max]` (Kellerer-Rossi, ICRU
36): adds crosser straggling, drops non-events (cures the toucher tail).
Importance-sampled from `2 pi b (1 - exp(-lambda(b)))`. Defaults:
`partial_deposition=True` (a delta deposits only what it loses in an `l_bar`
path), `soft_core=True` (Gaussian soft core below `T_0`, from the core radius or
1 keV; `soft_variance="collision"`, broader than `"bohr"`). Use a dense-core
radial dose (Kiefer-Chatterjee, ~60% of near-track energy in the core); a diffuse
Scholz-Kraft `1/r^2` (~5-7% within `r_0`) leaves a dip at `~T_0/l_bar`.

Electron range `R = A T^EXP`, `A = 0.0457 um`, `EXP = 1.748` (ESTAR CSDA fit,
<2% at 1/10/50/100 keV), deposition radius `R_dep = R_csda / DELTA_DETOUR_FACTOR`
(2.0, an X90-type modelling parameter):

| T (keV) | code fit | Cole 1969 | ESTAR CSDA |
|---|---|---|---|
| 10 | 2.56 | 2.70 | 2.52 |
| 100 | 143.2 | 150.4 | 143.1 |

Status: carbon `y_D` within ~12% of the Hartzell AMF (H-Ne checked). As the MCF
MKM engine it moves the V79-carbon `RBE_alpha` geo-ratio from 1.28 (analytic,
crosser-only) to **1.09** (`Δ̄` +0.36 -> +0.14, spread `R̄` 0.41 -> 0.27).

**Known limit: low-LET / high-energy `y_D` undershoot.** Against DeCunha protons
(1 um sphere), amorphous/library `y_D` = 0.93 / 0.91 / 0.82 / 0.78 / 0.73 at
10.5 / 30.5 / 50.5 / 100.5 / 150.5 MeV. Against Hartzell (0.3 um radius): H 100
MeV/u 0.70, C 100 1.11, Ne 100 1.19; the MKM regime (≤ a few MeV/u, high LET) is
0.91-1.03. Cause: a shape mismatch. At 100.5 MeV the library `y d(y)` peaks at
`y ≈ 0.55` with a long mid-`y` shoulder (1.5-10) carrying `y_D` to 1.63; the model
has a tall narrow peak at 0.90 falling off after 1.5. The shoulder comes from
rare, spatially correlated single deltas depositing 1-7 keV locally, which a mean
radial dose plus compound-Poisson resampling cannot produce. Ruled out:

- the electron range (three references agree to ~7%);
- detour factor -> 1 (longer `R_dep`, fewer contained deltas, worse);
- a Cole-1969 Bragg-aware deposition (protons 0.93 -> 0.74, worse);
- `soft_core` off (over-broadens, `y_D` overshoots).

For faithful low-LET / large-site proton spectra use the DeCunha library.

## Sato AMF engine

Sato 2006 Eq 7 in energy imparted `eps` [keV] (`y = eps/l_bar`):

    eps f1(eps) = A eps^2 / (exp{B[eps - C (dE/dx) phi_s]} + 1)                 primary peak
                + Σ_{k=1,2} (mu_k eps/w_i)/(j_k-1) ((j_k-1)/j_k)^(eps/w_i)        delta rays
                + Σ_{i=1..6} P_i delta(eps - eps_pi)                            excitation peaks
                + P_7/(sqrt(2pi) Gamma) exp{-(eps - eps_p7)^2/(2 Gamma^2)}        Auger peak

Fixed: `w_i = 10.9 eV`; `eps_pi = 10.9, 13.5, 17.0, 21.4, 26.3, 533 eV`;
`eps_p7 ≈ 700 eV`, `Gamma = 100 eV`; `A` from `B`, `C` (Eq 8). The sub-keV peaks
are omitted (Sato 2006: "trivial influence"). `B`, `C`, `mu_k`, `j_k` are
**least-squares fit outputs** (Sato 2006 p.4); PHITS stores them in `tsed.dat`
(not in any paper; Hartzell sec. 2.2 confirms the AMF reads per-(ion, energy,
domain) data from it).

Our stand-in (`micro/sato_coefficients.py`): fit Eq 7 to the digitized Hartzell
fig. 4 curves. Traps: the frequency density is `f(y) ∝ [eps f1]/y^2`; the fit
needs a `y_D`-matching term because the digitized curves stop at `y ~ 0.1`, leaving
the delta-ray extrapolation (amplified by `1/y^2`) unconstrained. Coverage: Z = 1,
2, 3, 6, 8, 10 (others raise); `log E` interpolation between 1/10/100 MeV/u,
clamped outside; fitted at 0.3 um radius, the peak scaling with the actual site via
`C L phi_s` (no explicit site-size dependence). Build the engine with the same
stopping power as the coefficient fit (`default_stopping_power()`).

## MCF MKM and mMKM from `d(y)`

MCF MKM (Parisi 2022a eqs. 3-5, 14, transcribed from the Europe PMC full text;
the Hartzell 2025 OCR of its eqs. 8-9 is garbled):

    X(y)   = (alpha0 + beta0 z_d(y)) z_n(y) + beta0 z_n(y)^2
    c(y)   = (1 - exp(-X)) / X                      -> 1 as y -> 0 (Taylor 1 - X/2 below X = 1e-8)
    alpha  = ∫ (alpha0 + beta0 z_d(y)) c(y) d(y) dy
    beta   = beta0 (∫ c(y) d(y) dy)^2
    alpha0 = alpha_ref - beta0 z_d(y_D,ref),   beta0 = beta_ref

mMKM (Kase 2006 / Hartzell eqs. 10-12):

    y0    = KEVUM rho pi r_d R_n^2 / sqrt(beta_ref (r_d^2 + R_n^2))
    y*_D  = y0^2 ∫ (1 - exp(-(y/y0)^2)) d(y)/y dy
    alpha = alpha_ref + beta_ref (y*_D - y*_D,ref) / (KEVUM rho pi r_d^2),   beta = beta_ref

RBE (Hartzell eqs. 6-7):

    RBE_D = [-alpha_ref + sqrt(alpha_ref^2 + 4 beta_ref (alpha_p D + beta_p D^2))] / (2 beta_ref D)
    RBE_S = [alpha_p + sqrt(alpha_p^2 - 4 beta_p ln S)] / [alpha_ref + sqrt(alpha_ref^2 - 4 beta_ref ln S)]

Parameters. HSG, 6 MV reference (Hartzell sec. 2.3; `MCFMKMModel` defaults):
`R_n = 4.5 um`, `r_d = 0.28 um`, `y_D,ref = 2.5 keV/um`, `alpha_ref = 0.217 Gy^-1`,
`beta_ref = 0.0615 Gy^-2`, `rho = 1.0`; the mMKM uses `R_n = 4.3 um` (chosen to
align it with MCF MKM). V79 (Parisi 2024b): `y_D,ref = 4`, `r_d = 0.27`,
`R_n = 4.0`. `y_D,ref` must match the reference radiation (~2.3 keV/um for
60Co/6 MV, ~4.3 for 200 kVp; Parisi sec. 2.5.3). Parisi sizes `r_d` per cell line
from DNA content (eqs. 15-17): `core.chromatin.domain_radius_from_genome`.

V79-carbon diagnosis: the ~28% `RBE_alpha` over-prediction of the analytic engine
is the **spectrum**, not the parameters or the stopping power: swapping
SimpleBethe for pyMKM's Geant4 tables moved it 0.3% (domain `y_D` identical), and
Parisi 2024b's exact V79 parameters moved `alpha0` only 0.128 -> 0.129.

Numerics: renormalise `d(y)` to unit area before integrating (digitized or
truncated spectra are not normalised); Simpson on the spectrum's own grid is for
finely sampled `d(y)` only; integrate a coarse binned histogram per bin (AGENTS.md
-> Numerics). `Spectrum.y_F_override`/`y_D_override` carry exact count-weighted
moments for such histograms (non-uniform Simpson weights can turn `∫ f dy`
negative). The digitized Hartzell spectra are scored at 0.3 um radius, the HSG
domain is 0.28 um.

## Regression / acceptance pins

| test | pins |
| --- | --- |
| `tests/micro/test_lineal_energy.py` | sphere chord moments, cylinder Cauchy mean chord; geometric limit `y_F = LET` (rel 2e-3), `y_D = 1.125 LET` (rel 3e-3); restricted `y_F` = `L_Delta` (rel 2e-2); delta-tail section incl. `test_delta_tail_moves_yD_toward_hartzell_amf`; `test_amorphous_no_degeneracy_at_mkm_domain_size`; `test_amorphous_yD_tracks_hartzell_amf_for_carbon` (rel 0.15); `test_delta_ray_range_matches_cole_and_estar` (rel 0.07, 10 and 100 keV) |
| `tests/micro/test_sato_engine.py` | normalisation (rel 1e-2); carbon peak vs Hartzell (rel 0.25); delta-ray tail present; `y_D` vs Hartzell (rel 0.2); MCF MKM V79-carbon `alpha` below the restricted analytic engine at 10/30/100 MeV/u; non-spherical sites and uncovered Z refused |
| `tests/models/test_spectrum_models.py` | MCF photon limit, `beta` below `beta_ref` at high LET, `RBE_10` overkill peak; mMKM `y*` = `Spectrum.y_star`, constant `beta`; `test_delta_tail_reduces_v79_carbon_alpha_overprediction`; cylinder default and pyMKM agreement |

## Known limits

- Amorphous engine: `y_D` ~10-25% low for low-LET ions above ~50 MeV/u (above).
- Sato AMF: six species only, energy anchors 1/10/100 MeV/u, one site size.
- The PHITS AMF has a known interpolation artifact for ions heavier than 20Ne
  below 10 MeV/u (Parisi 2022 note).
- MCF MKM / mMKM / SMK spectrum models do not mix `RadiationField`s
  (`NotImplementedError`); pass a combined `d(y)` or evaluate per ion.
