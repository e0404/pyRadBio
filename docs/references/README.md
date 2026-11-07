# Reference summaries

Condensed, cited notes on the primary literature, written so an agent can
implement a model without re-reading the full PDF. **Every quantitative claim
here cites a paper section/equation.** If you refine a summary, keep the
citations and reconcile any code discrepancy in the same PR.

## Template for a model summary

```
# <Model> — <Authors Year> (doi:...)
## One-paragraph idea
## Inputs (with units + source eq.)
## Core equations (numbered as in the paper)
## Parameters & default values (with the table/section they come from)
## Known limits / failure modes
## Golden values for tests (value, tolerance, paper location)
## Implementation notes / gotchas
```

The model inventory and all open work live in `ROADMAP.md` at the repository
root, not here.

## Index
- `StoppingPower.md` — **water stopping powers**: conventions, every source and its
  provenance, the library default, and the measured accuracy of each class
  against PSTAR/ASTAR, ICRU 73 and ICRU 90
- `NorthcliffeSchilling1970.md` — the NS1970 water stopping-power table
- `Microdosimetry.md` — the lineal-energy engines (analytic, amorphous track,
  Sato AMF, impact parameter) and the MKM `d(y)` routes, incl. MCF MKM
- `mMKM_Inaniwa2010.md` — modified MKM (equations and golden values)
- `SMK_Inaniwa2018.md` — stochastic MKM (nucleus-level stochastics; quality-dependent β;
  Fig. 4 reconstruction results)
- `OSMK_Inaniwa2021_2023.md` — oxygen-effect SMK, **both** published formulations
  (2021 `Rmax(z_d,D)` vs 2023 site-scaling); the oxygen axis; figs. 5-6 OER
  reconstruction and the open V79 mid-LET excess
- `IMK_Matsuya2025.md` — integrated MK model (Matsuya lineage): mMKM's `alpha`
  plus a repair-rate-driven Lea-Catcheside `beta`, the micronucleus channel, and
  the fig.-2 inversion that recovers the paper's unprinted `y*` (photon included)
- `Phenomenological_RBE_models.md` — the RBEmin/RBEmax family (Wedenberg /
  McNamara / Carabe / Mairani-helium): shared skeleton, per-coefficient
  provenance (uneven! read before trusting one), validity domains, and the
  matRad cross-check incl. the McNamara `p0` discrepancy
- `RMF_Carlson2008_Frese2012.md` — **Repair-Misrepair-Fixation**: the DSB
  ion-kill/binary-misrepair lineage, Frese Table 1's six derived parameter sets,
  why its `RBE_min` *is* the DSB-yield ratio, the 5 um DIAMETER-vs-radius trap,
  and the one internally inconsistent row in its table
- `MCDS_SemenenkoStewart2006.md` — the closed-form DNA-damage yields RMF eats
- `Katz1971.md` — **Katz cellular track-structure theory** (1971), the family
  ancestor: four parameters, the ion-kill/gamma-kill split, all ten published
  parameter sets, and why only the grain-count regime is implementable from
  the paper. Also records two defects in the printed tables
- `NanOx.md` — **NanOx** (Lyon; seven papers): local lethal events in 20 x 10 nm
  targets plus oxidative-stress global events. Ships the assembly and
  calibration layer only — the nanodosimetric `dP/dz` and `RCE` are non-public
  and figure-only, so the ion route raises. The `β_G = β_r/η²` golden chain, the
  diameter-vs-radius trap, the superseded 2017 V79 set, and where each `β_r`
  comes from
- `LEM_I_ScholzKraft.md` — LEM I
- `LEM_II_Elsasser2007.md` — LEM II (cluster + radial dose)
- `LEM_III_Elsasser2008.md` — LEM III (velocity-dependent core)
- `LEM_IV_Elsasser2010.md` — LEM IV (DSB cluster index; primary paper + calibration)
- `LEM_IV_Friedrich2012.md` — LEM IV (DSB cluster index; proceedings follow-up)
- `LEM_IV_OPEN_CHOICES.md` — LEM IV: every free parameter, fork and known
  defect, and what would settle each. Read before tuning anything in LEM IV.
- `pide_schema.md` — PIDE 3.4 loader contract (headers verified against the workbook)
- `RadPhysBio.md` — the open-access RadPhysBio database (Zanni 2024): export
  semantics, the ion/photon pairing this package builds, what a benchmark on it
  can say, and its cross-checks against PIDE and against our own digitizations
- `pyMKM_API.md` — the pyMKM API pyRadBio calls, verified against the installed
  package (units, the `.txt` table format)
- `survival_repo_notes.md` — the batuff/Survival code and our cross-checks against it
