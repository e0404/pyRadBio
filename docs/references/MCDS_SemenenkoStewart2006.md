# MCDS: Monte Carlo Damage Simulation (Semenenko & Stewart 2004 / 2006)

- **Semenenko & Stewart (2004)**, *A Fast Monte Carlo Algorithm to Simulate the
  Spectrum of DNA Damages Formed by Ionizing Radiation*, Radiat. Res.
  **161**(4):451-457, **doi:10.1667/RR3140** (the algorithm).
- **Semenenko & Stewart (2006)**, *Fast Monte Carlo simulation of DNA damage formed
  by electrons and light ions*, Phys. Med. Biol. **51**(7):1693-1706,
  **doi:10.1088/0031-9155/51/7/004** (the revision and the source of everything
  implemented).

Implemented in `pyradbio/micro/mcds.py`; tests `tests/micro/test_mcds.py`.

## One-paragraph idea

MCDS is a fast surrogate for track-structure DNA-damage simulation: it scatters
strand breaks and base damages at random over a DNA segment whose length encodes
radiation quality, then clusters and classifies them as SSB, DSB or base damage.
**We do not run that algorithm.** Sec. 2.2 fits published track-structure results
(Nikjoo et al., Friedland et al.) to a closed form, a two-parameter saturation
curve in `Z_eff^2/beta^2`, which is what this package implements and RMF consumes.

## Core equations

```
(1)  Z_eff = Z [1 - exp(-125 beta Z^{-2/3})]           Barkas 1963
(2)  beta  = sqrt(1 - 1/(1 + T/m0c^2)^2)
(3)  f(x)  = u + (v - u) x/(x + w),        x = Z_eff^2/beta^2
```

`f(1) ~ f(0) = u` (low-quality limit), `f(inf) = v` (saturation), `w` the
half-saturation point.

## Parameters (sec. 2.2, eq. 5)

| quantity | `u` | `v` | `w` | unit |
|---|---|---|---|---|
| DSB yield | 48.9 | 164.5 | 759 | Gy^-1 cell^-1 |
| SSB yield | 1134 | 291 | 1615 | Gy^-1 cell^-1 |
| `n_seg` | 149 200 | 25 600 | 267 | bp Gy^-1 cell^-1 |

Algorithm inputs (eq. 5): `sigma_Sb = 1300 Gy^-1 cell^-1`, `f = 3` base damages per
strand break, `N_min = 9 bp`. `f = 3` is carried over from the 2004 paper
(base-damage endpoints were not available for the fit).

**DSB rises, SSB falls:** denser tracks convert single- into double-strand breaks.
Any implementation where both move the same way is wrong.

**Two unit systems:** the paper prints every yield twice (48.9/8.15, 164.5/27.4,
1134/189, 291/48.5); all four give **6.0 Gbp per cell**. Carlson 2008 works per
cell, Frese 2012 per Gbp.

## Known limits / failure modes

- **Fit domain `1 <= Z_eff^2/beta^2 <= 3200`** (sec. 2.3). The code warns outside
  it and evaluates anyway; it never clamps.
- Stated minimum kinetic energies: `T_min ~ 0.00008 MeV` (electrons), `0.105 MeV`
  (protons), `2 MeV` (alpha), the energies at which `x` hits 3200. Goldened: ours
  give 3194 / 3197 / 3185 against 3200, checking beta, effective charge and the
  quality parameter in one shot.
- **Electrons use bare `Z^2/beta^2`**, not the effective charge (sec. 2.3).
- **The reference photon yields RMF needs are not from this formula:** Frese's
  `Sigma_x = 8.27` (Co-60) and `8.68` (200 kVp) were simulated with a
  secondary-electron spectrum; they sit just above eq. (3)'s `x -> 0` limit of
  8.15 Gy^-1 Gbp^-1, the only cross-check.

## Golden values for tests

The printed fit constants, the stated yield ranges, the dual-unit pairs and the
three `T_min` values.

## Implementation notes / gotchas

- Eq. (1) is already in the package as `track/stopping_power.py::effective_charge`,
  the same formula as Katz et al. 1971 eq. (12); `micro/mcds.py` delegates rather
  than duplicating. Sec. 2.2 cites Katz (1970) for using `Z^2/beta^2` as the
  interpolation variable, so RMF's *biology* is independent but its
  *track-quality axis* is Katz's (see `RMF_Carlson2008_Frese2012.md`,
  `Katz1971.md`).
- Lives in `micro/`, not `models/`: nothing in it is RMF-specific (NanOx, BIANCA
  and the UNIVERSE line would want DSB yields too).
