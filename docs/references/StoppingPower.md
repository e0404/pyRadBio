# Water stopping powers: sources, conventions, measured accuracy

What `pyradbio.track.stopping` implements, where every number comes from, and how well
each source reproduces the evaluated tables. Accuracy figures were **measured** against
the raw source files (not committed; see `data/stopping/README.md`) and are pinned by a
test unless marked otherwise.

## Conventions (`track/stopping/base.py`)

* **Energy axis** is kinetic energy per nucleon E/A [MeV/u] (a proton's MeV/u is its MeV).
  NIST tables (PSTAR/ASTAR, ICRU 90 C) print total kinetic energy; the fetch/build scripts
  divide by A. ICRU 73 prints E/A1 per element without distinguishing per nucleon from per u
  (its footnote 6, <= 0.25% from Li up).
* **Velocity** from true nuclear masses: NIST isotopes v4.1 (AME2012), nuclear mass =
  atomic mass - Z m_e. A tabulated ion serves its other isotopes **at equal velocity**
  (equal gamma). Nuclear stopping never follows velocity; it is evaluated per isotope.
* **Quantity**: `"electronic"` (default) or `"total"` (electronic + nuclear).
* **Interpolation**: not-a-knot cubic in log-log (PCHIP available). Shipped tables hold
  50 points per decade. Sources computed per energy (MSTAR, ICRU90Procedure, the shell
  table) are interpolated on their grid nodes by default and evaluated directly on request
  (`evaluation="direct"`).
* **Out of range** raises `OutOfRangeError`; nothing is extrapolated. ICRU 73 tables start
  at 0.025 MeV/u, so the ICRU class raises below.
* **Material**: liquid water only.
* **Metadata**: `metadata(Z, A)` gives an `IonSourceInfo` (source, reference, I-value, range,
  edition, options); `settings()` gives the resolved options of the whole source, which
  alpha/beta table headers record under `model_config`.

## Classes

| class | what | Z | I [eV] |
|---|---|---|---|
| `XSTAR` | NIST PSTAR/ASTAR (ICRU 49) for p, He; MSTAR 3.12 (default) or z_eff-scaled PSTAR for Z 3-18 | 1-18 | 75 |
| `ICRU(edition)` | `"2016"`: ICRU 90 p, He, C + ICRU 73 errata Z 3-18; `"2009"`: ICRU 49 + 73 errata; `"2005"`: ICRU 49 + 73 | 1-18 | per ion: 75 / 78 / 67.2 / 78 |
| `MSTAR` | our port of MSTAR 3.12, evaluated from the user's `msdbs1.d` | 3-18 | 75 (its alpha data) |
| `Bethe` | PDG 2024 eq. 34.5 with switchable density effect, shell, Barkas, Bloch, Mott, Ahlen, Lindhard-Sorensen | 1-18 | any (default 78) |
| `ICRU90Procedure` | ICRU 90's carbon procedure (Z 3-18) and ICRU 49's p/alpha procedure, at any I | 1-18 | any (default 78) |
| `HubertAlphaScaling` | Hubert, Bimbot & Gauvin 1990: (gamma Z/2)^2 S_alpha, alpha from ICRU90Procedure, 2.5 MeV/u up | 3-92 | the alpha's (78) |
| `WaterNuclearStopping` | ICRU 49 eqs. 4.14-4.15 + Table 4.1 (+ ICRU 90 fn. 35) | any in the isotope table | none |
| `SimpleBetheStoppingPower` (`track/stopping_power.py`) | analytic Bethe with Barkas z_eff, no corrections | any | 78 |

NS1970 (Northcliffe & Schilling) is documented in `NorthcliffeSchilling1970.md`; the SRIM
reader in `survival_repo_notes.md` section 3.

## Library default

`default_stopping_power()` returns `DefaultStoppingPower`:

* `ICRU90Procedure()` for Z 1-18 inside its range;
* `HubertAlphaScaling` (Hubert et al. 1990 (gamma Z/2)^2 S_alpha, ICRU 90's alpha) for
  Z 19-92 from 2.5 MeV/u. Fitted by them to O-U data, ~5% in solids; for Z 3-18 in water
  it is within 2.5% of MSTAR above 10 MeV/u, +7% for Si/Ar at 2.5-10 MeV/u, 1.2% on
  ICRU 90 carbon. No water reference above Ar is held;
* `SimpleBetheStoppingPower` elsewhere (Z > 18 below 2.5 MeV/u, Z > 92, outside ranges),
  **anchored** to the adjacent branch at its nearest edge,
  `S = S_Bethe(E) S_branch(E_edge)/S_Bethe(E_edge)` (`source()` = `"simple_bethe_anchored"`;
  plain `"simple_bethe"` only where no branch exists: Z > 92, isotopes without a shipped
  mass). Unvalidated there.

`DefaultStoppingPower.source(ion)` names the branch. Every model default goes through this
factory.

**The 2.5 MeV/u seam.** An unanchored switch from SimpleBethe to Hubert jumps by +5.8% (K),
+8.9% (Fe), +14.4% (Kr), +21.9% (Xe), +21.7% (Pb), +18.8% (U). Anchoring makes the value
continuous; the slope kinks by |d ln S/d ln E| = 0.012 (K), 0.006 (Fe), 0.009 (Kr),
0.005 (Xe), 0.052 (Pb), 0.085 (U). Shapes below 2.5 MeV/u measured for Z 6/10/14/18
against MSTAR, each anchored to MSTAR at 2.5 MeV/u, worst per band 0.1-0.5 / 0.5-1 /
1-2.5 MeV/u:

| shape below 2.5 MeV/u | C | Ne | Si | Ar |
|---|---|---|---|---|
| SimpleBethe (chosen) | -19 / -5.2 / +0.7% | -13 / -9.4 / -3.1% | -14 / -10 / -3.9% | +12 / -2.3 / -0.5% |
| Brandt-Kitagawa 1982 x ICRU 90 protons | +13 / +5.7 / +4.6% | +34 / +7.0 / +4.8% | +37 / +8.9 / +6.1% | +64 / +21 / +12% |
| Hubert's gamma extrapolated below its fit | -68 / +5.3 / +5.2% | -89 / -15 / -1.6% | -97 / -24 / -5.7% | -99.6 / -23 / -4.7% |

For sensitivity studies switch `Bethe`'s corrections one at a time, or swap any class
into a model's `stopping_power` field.

## Sources and provenance

Raw data lives git-ignored in `data/stopping/<source>/` with a committed `PROVENANCE.md`
(origin, citation, retrieval date, notice, checksums). `scripts/stopping/fetch_sources.py`
fetches or checks it; `scripts/stopping/build_tables.py` (deterministic, `--check`)
writes our own regridded tables into `pyradbio/track/stopping/data/`.

| folder | source | shipped as |
|---|---|---|
| `pstar/`, `astar/` | NIST SRD 124 (ICRU 49) | `pstar/astar_water_liquid.csv` |
| `icru90/` | NIST's ICRU 90 addendum (Tables A.3, A.9, A.12, A.15) | `icru90_water_liquid.csv`, delta at 78 eV |
| `icru73/` | libdedx raw ICRU 73 files, verified equal to the printed tables (848/848) | `icru73_2005/2009_water_liquid.csv` |
| `mstar/` | MSTAR 3.12 source + database (IAEA, freeware; notice in `THIRD_PARTY_NOTICES.md`) | `mstar_c/d_water_liquid.csv` (port values at nodes) |
| `estar/`, `pdg/` | ESTAR delta at 75 eV; PDG Sternheimer coefficients (CC BY) | `density_effect_...`, `sternheimer_...` |
| `nist_isotopes/` | NIST isotope masses | `nist_isotope_masses.csv` |
| `arb1973/` | Ashley, Ritchie & Brandt 1973 Table I (hand transcription) | `barkas_arb1973.csv` (F, x >= 1) |
| `ahlen1980/` | Ahlen 1980 Table XII, cos chi (hand transcription) | `ahlen1980_cos_chi.csv` |
| `brandt1982/` | Brandt & Kitagawa 1982 Table II, q(y_r) (hand transcription) | `brandt1982_ionization.csv` |
| `icru49/` | ICRU 49 Tables 3.1/3.5/3.7 (water) and Table 4.1 (hand transcriptions) | `icru49_empirical_water.csv`, `icru49_reduced_nuclear.csv` |
| none | computed: Walske K/L shells + H atom | `shell_corrections_water.csv` |

No SRIM, no Geant4, no libdedx *data* in the library (libdedx's ICRU 73 transcription is
used as raw input only, verified against the printed report). ICRU 73 I-value: 67.2 eV
implied for the original edition, 78.0 eV for the errata (mapping in
`data/stopping/icru73/PROVENANCE.md`).

## Measured accuracy

### Tables against their own sources (round trip at the source points)

PSTAR 2.8e-5, ASTAR 4.3e-5, ICRU 73 4.6e-6 / 8.2e-6 (2005 / 2009), ICRU 90 <= 8.3e-7,
ARB F 1.9e-6, ICRU 49 Table 4.1 1.9e-7. Leave-one-out near the stopping maximum (the
data-only interpolation bound): PSTAR cubic 0.19% (PCHIP 0.45%); ICRU 90 cubic ~0.3%
(PCHIP 1.5-2.3%).

### MSTAR port

Python port vs the gfortran-compiled original: <= 2.0e-15 (24154/24196 bit-identical;
without MSTAR's REAL*4 constants it would be 2.4e-7 off). Vs pyMKM's MSTAR (mode b):
5.2e-6. Shipped tables vs direct: 6.8e-6.

### Bethe and its corrections

* Plain Bethe vs PSTAR / ICRU 90 p and alpha: +0.6-0.7% at 10 MeV/u, <= 0.08% above
  100 MeV/u. Vs ICRU 90 carbon above 30 MeV/u: 0.62% plain, 0.059% with
  Lindhard-Sorensen, 0.061% Bloch + Mott, 0.058% Bloch + Ahlen 1980's close-collision Mott
  term G/2 (eq. 4.23, w0 = 0; ICRU 90's own choice for C). Mott worsens p/alpha.
  `ICRU90Procedure` defaults to Bloch + Ahlen; `higher_order="lindhard_sorensen"` is the
  exact alternative.
* **Shell** (`shell.py`): Walske's K (1952) and L (1956) shells computed exactly; all of
  Walske's printed S, T and Tables I-III reproduced (C_L to <= 0.2%), and Khandelwal 1968's
  Tables 1-3 (B_K: median 1.8e-5 over 552 cells). Khandelwal's tables carry defects (the
  eta = 2 row is 0.3% low against his neighbouring rows, eta = 7 at theta <= 0.68 is 0.7%
  high, three single-cell misprints), and Walske's eq. 17 coefficient 22/3 should be 25/3
  (Khandelwal eq. 18; confirmed by our exact hydrogen atom). Water = 2 H atoms (exact) +
  O K (theta_K = 0.7) + 6/8 of Walske's L (theta_L = 0.5). **The two theta and the 6/8 are
  calibrated/assumed**, fitted to PSTAR protons; Bichsel 1967 (UCRL-17538) would give his
  own fitted choices for O.
* **Barkas** (`barkas.py`): ICRU 49 eq. 2.8 with F recovered from ARB 1973 Table I
  (x >= 1, i.e. >= ~0.2 MeV/u); it raises below. **gamma = 1, not ICRU 49's printed 1.29**:
  with 1.29 our zB1 is ~1.27x BEST's (ICRU 90 Fig. 4.1, digitised) and alphas miss ASTAR by
  +2.4%; with 1 they are at -0.14%. Ashley et al. 1974 (ICRU 49's F tabulation) was not
  available and may absorb the factor.
* Bethe + shell + Barkas + Bloch, 0.5-1000 MeV/u: PSTAR <= 0.13%, ICRU 90 p <= 0.15%,
  ASTAR <= 0.14%, ICRU 90 alpha <= 0.14% (only PSTAR was used for the fit).
* **Ruled out: physical thetas.** Khandelwal 1968 defines theta_s as observed ionization
  potential over Z_s^2 Ry/n^2; with Bearden & Burr 1967 (Rev. Mod. Phys. 39, 125, Table I,
  8 O: K 532.0, L_I 23.7, L_II,III 7.1 eV) and Z_K = 7.7, Z_L = 3.85 this gives theta_K =
  0.66, theta_2s = 0.47, theta_2p = 0.14. Bethe + Barkas + Bloch vs PSTAR at 0.5-1 / 1-2 /
  2-5 MeV/u (calibration: +0.13 / +0.09 / -0.03%):
  - K, 2s, 2p separately: +14.5 / +7.2 / +2.1%, and the C/Z of ICRU 90 Fig. 4.1 turns
    negative (118% rms);
  - K and one L theta = 0.25 (electron-weighted binding) x 6/8: +3.5 / +1.6 / +0.6%;
  - theta_K = 0.66 with the calibrated L (0.5-0.6): +0.25-0.33 / +0.19-0.26 / +0.06-0.10%.

  The 2p electrons (theta = 0.14) fail: hydrogenic wave functions do not describe loosely
  bound valence electrons (why Bichsel fitted Model 1's scale factors, ICRU 49 sec. 2.4).
  The K shell is near physical anyway (0.66 vs fitted 0.70, a 0.15% effect at 0.5-2 MeV/u).
* **Khandelwal & Merzbacher 1966 (Phys. Rev. 151, 12) 2s/2p split is unusable as printed**:
  L_I + 3 L_II equals Walske's L form factor exactly, but at large Q the subshells
  integrate to 2/3 (2s) and 10/9 (2p) instead of 1 each. Khandelwal 1968's Table 5 (L_II)
  matches to 0.2% for eta <= 3; his Table 4 (L_I) has their slope but sits 0.15-0.19 lower
  in B (unexplained). Correct hydrogenic subshell factors (Choi, Merzbacher & Khandelwal,
  At. Data 5, 291 (1973)) were not pursued.

### Effective charge of Bethe

`Bethe` was bare-charge, which overestimates partially stripped ions (Si / Ar +17% / +23%
vs MSTAR at 4.7-10 MeV/u). `effective_charge` defaults to Hubert, Bimbot & Gauvin (1990),
applied for Z >= 3. Worst |dev| over Z 3-18 at 4.7-300 MeV/u, with shell + Barkas + Bloch +
Ahlen:

| `effective_charge` | MSTAR | ICRU 73 err. | ICRU 90 C |
|---|---|---|---|
| `None` (bare) | 22.8% | 16.2% | 0.04% |
| `"hubert"` (default; alpha-referenced, Z2 = 8 for water) | 5.1% | 4.6% | 1.08% |
| `"fitted"` (Barkas form, coefficient 162 fitted to MSTAR) | 3.3% | 5.5% | 1.28% |
| `"brandt_kitagawa"` (1982 eq. 23 + Table II) | 12.1% | 16.9% | 7.6% |
| `"barkas"` (1963, coefficient 125) | 13.7% | 18.3% | 4.0% |

With `Bethe`'s default settings (no corrections) bare is off by up to 43.5% and Hubert by
6.6% (Barkas 2.8%, fitted 17.8%; its coefficient was fitted with corrections on). Barkas
describes the mean charge *state*; the stopping charge is larger (close collisions see the
nucleus). Brandt-Kitagawa as printed keeps too many bound electrons at these velocities.
Every effective charge costs ~1% on ICRU 90 carbon, which is bare-charge above 5 MeV/u,
so `ICRU90Procedure` stays bare above its stripping-scaled thresholds.

### ICRU90Procedure

* Thresholds per ion: ICRU 90's carbon values 2.5 / 5 MeV/u scaled by max(1, (Z/6)^(4/3))
  (Bohr's stripping criterion; Ar: 21.6 MeV/u). Worst band vs MSTAR over 0.5-300 MeV/u:
  Li 1.5%, C/O/Ne 2.0%, Si 1.5%, Ar 1.6%. Carbon's values per nucleon
  (`threshold_scaling="none"`) put Si / Ar +16% / +21% above MSTAR at 2.5-10 MeV/u. Below
  ~10 MeV/u the procedure is 5-8% under the ICRU 73 errata (MSTAR vs ICRU 73, the choice
  ICRU 90 made for carbon).
* Carbon vs ICRU 90 Table A.15 (electronic): 0.54% (< 0.01 MeV/u, the alpha table's edge),
  <= 0.067% below 1 MeV/u, <= 0.070% above (default Bloch + Ahlen; 0.055% with
  Lindhard-Sorensen). Without shell + Barkas: -0.34% in the blend.
* p and alpha with ICRU 49's procedure, below T1 / in the T1-T2 gap / above T2: PSTAR
  (I = 75) 0.020 / 0.13 / 0.13%; ICRU 90 p (78) 0.020 / 0.13 / 0.15%; ASTAR 0.049 / 0.26 /
  0.14%; ICRU 90 alpha 0.049 / 0.33 / 0.14%. The gap residual is the unpublished spline
  knots of ICRU 49 sec. 3.5. ICRU 90's p/alpha tables equal ICRU 49's below T1.
* Total (electronic + nuclear) vs S_tot: PSTAR 0.14%, ICRU 90 p 0.14%, ASTAR 0.25%,
  ICRU 90 alpha 0.33%, ICRU 90 C 0.10%, 1 keV/u to 1000 MeV/u.

### Nuclear stopping

ICRU 49 eqs. 4.14-4.15 with **Table 4.1** (not the eq. 4.16 fit, which misses the tables by
3-5% and by +20% at high T^ where the table saturates, S^ T^ -> 5.83). ZBL with
a = z^0.23 + Z^0.23 (ICRU 90 fn. 35): ASTAR, ICRU 90 alpha and C nuclear columns to
<= 0.07%. Protons follow ICRU 49 sec. 4.3: Moliere's potential classically (Table 4.1)
below Z_t/10 MeV of each target atom (0.1 MeV for H, 0.8 MeV for O), Moliere's quantum
cross section above, here in Born approximation (analytic for his three-exponential
screening, nonrelativistic velocity, relativistic momentum cutoff; ICRU's T^ proportional
to T). Vs PSTAR's nuclear column: 0.10% below 0.1 MeV, -2.7% at 0.1-0.8 MeV, +5.0% at
0.8-3 MeV (oxygen's Sommerfeld parameter 1.4-0.7, where Born is not valid), -1.4% at
3-10 MeV, a flat -2.4% from 10 MeV to 10 GeV (classical only: +5% at 1-100 MeV, -19% at
10 GeV). Nuclear stopping is < 0.1% of the total there, so the worst band is <= 4e-5 of
S_tot; the switch leaves a +7.9% step in oxygen's nuclear term at 0.8 MeV (7e-5 of S_tot).
Defaults: ICRU 49's method (Moliere p) for the ICRU-based classes and ICRU90Procedure; ZBL
for every ion in Bethe and MSTAR.

## Known limits

* Shell/Barkas parameters are a calibration, not ICRU 49's exact Model 1.
* F below x = 1 (< ~0.2 MeV/u) needs Ashley, Ritchie & Brandt 1972's L(x) inner-shell term
  (figure only), so Barkas raises there.
* Lindhard-Sorensen "v -> c" values for low Z: not reproduced. Our point-nucleus
  Delta L - Delta L_NR reproduces all six printed high-Z values (Z = 84, 92 at gamma = 100,
  1.5, 1.2; golden test), but for Z = 10, 18, 36 LS quote 0.119, 0.220, 0.473 where we get
  0.125, 0.241, 0.551 (our excess over the linear Mott term scales as Z^2 and is ~2.6x
  LS's; possibly their finite nuclear size, LS Sec. VI, not implemented). For Z <= 18 at
  gamma <= 2 the difference is <~ 0.03 in L (~0.3% of S).
* Proton nuclear stopping at 0.8-3 MeV: Moliere's full quantum cross section (any
  Sommerfeld parameter) would remove the +5% Born residual (<= 4e-5 of S_tot).
* Above a few hundred MeV/u the Hubert alpha scaling uses the alpha's Mott and
  Lindhard-Sorensen terms, not the heavy ion's.
* The default is unvalidated for Z > 18 below 2.5 MeV/u (no water reference held).

## References

ICRU Report 49 (1993); ICRU Report 73 (2005) and its errata (Sigmund, Schinner & Paul
2009); ICRU Report 90 (2014); Navas et al. (PDG), Phys. Rev. D 110, 030001 (2024);
Sternheimer & Peierls, Phys. Rev. B 3, 3681 (1971); Ahlen, Rev. Mod. Phys. 52, 121 (1980);
Lindhard & Sorensen, Phys. Rev. A 53, 2443 (1996); Walske, Phys. Rev. 88, 1283 (1952) and
101, 940 (1956); Khandelwal, Nucl. Phys. A116, 97 (1968); Khandelwal & Merzbacher, Phys.
Rev. 151, 12 (1966); Bearden & Burr, Rev. Mod. Phys. 39, 125 (1967); Moliere, Z.
Naturforsch. 2a, 133 (1947); Hubert, Bimbot & Gauvin, At. Data Nucl. Data Tables 46, 1
(1990); Brandt & Kitagawa, Phys. Rev. B 25, 5631 (1982); Ashley, Ritchie & Brandt, Phys.
Rev. B 5, 2393 (1972) and Phys. Rev. A 8, 2402 (1973); Paul & Schinner, NIM B 179, 299
(2001), NIM B 195, 166 (2002), ADNDT 85, 377 (2003).
