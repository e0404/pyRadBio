# MKM family

The microdosimetric-kinetic models (MKM) express cell killing through the specific energy
\( z \) deposited in small sub-nuclear domains. pyRadBio implements the family at two levels:

- [`MKMModel`][pyradbio.models.mkm.model.MKMModel] is the classic modified MKM of Inaniwa et
  al. 2010 in the amorphous-track formulation, \( \alpha = \alpha_0 + \beta_0 z^*_{1D} \).
  It owns the biophysics and delegates the \( z^*_{1D} \) computation to a pluggable
  backend.
- The **spectrum-based models** take their radiation quality from a single-event
  lineal-energy distribution \( d(y) \): modified MKM, MCF MKM, SMK, OSMK 2021 and 2023, and
  IMK.

| Class | Model | Reference |
| --- | --- | --- |
| [`MKMModel`][pyradbio.models.mkm.model.MKMModel] | MKM on a pluggable \( z^*_{1D} \) backend | Kase et al. 2008 (doi:10.1088/0031-9155/53/1/003); Inaniwa et al. 2010 (doi:10.1088/0031-9155/55/22/008) |
| [`ModifiedMKM`][pyradbio.models.mkm.spectrum_models.ModifiedMKM] | modified MKM | Kase et al. 2006 (doi:10.1667/RR0536.1); Inaniwa et al. 2010 |
| [`MCFMKMModel`][pyradbio.models.mkm.spectrum_models.MCFMKMModel] | Mayo Clinic Florida MKM | Parisi et al. 2022 (doi:10.1088/1361-6560/ac7375) |
| [`StochasticMKM`][pyradbio.models.mkm.spectrum_models.StochasticMKM] | SMK | Inaniwa & Kanematsu 2018 (doi:10.1088/1361-6560/aabede) |
| [`OSMK2021`][pyradbio.models.mkm.osmk.OSMK2021] | SMK with the oxygen effect | Inaniwa et al. 2021 (doi:10.1088/1361-6560/ac29cc) |
| [`OSMK2023`][pyradbio.models.mkm.osmk.OSMK2023] | event-by-event revision of OSMK | Inaniwa & Kanematsu 2023 (doi:10.1093/jrr/rrad049) |
| [`IMKModel`][pyradbio.models.mkm.imk.IMKModel] | integrated MK with a repair channel | Matsuya et al. 2025 (doi:10.1002/mp.70040) |

Import everything from `pyradbio.models.mkm`. No optional extra is needed.

## `MKMModel` and its backend

`MKMModel` defaults to the Inaniwa 2010 HSG parameters and the native backend, so it runs
with the core dependencies:

```python
from pyradbio import Ion
from pyradbio.models.mkm import MKMModel, NativeMKMBackend

carbon = Ion(Z=6, A=12, energy_MeV_u=100.0)

mkm = MKMModel()  # Inaniwa 2010 HSG defaults, native backend
print(mkm.alpha_beta(carbon))  # LQParameters(alpha, beta)

# geometry has one owner: build it on the backend explicitly ...
explicit = MKMModel(backend=NativeMKMBackend(domain_radius_um=0.32, nucleus_radius_um=3.9))
print(explicit.backend.z_star_1D_Gy(carbon))  # z*_1D in Gy
```

`MKMModel` has no knowledge of any engine. It only calls the three methods of the
[`MKMBackend`][pyradbio.models.mkm.base.MKMBackend] protocol: `z_star_1D_Gy(ion)`,
`z_star_1D_table_Gy(ions, energies)` and `let_keV_um(ion)`. Any object that implements them
can be passed as `backend=`:

- [`NativeMKMBackend`][pyradbio.models.mkm.native_backend.NativeMKMBackend] is the default.
  It computes \( z^*_{1D} \) from the Kiefer-Chatterjee radial dose by analytic
  impact-parameter integration, with LET from the library default stopping power.
- [`PyMKMBackend`][pyradbio.models.mkm.pymkm_backend.PyMKMBackend] delegates to the external
  [pyMKM](https://github.com/BeppeMagro/pymkm) package. It is an independent cross-check, not
  a default: it needs the `mkm` extra and is the only module that imports pyMKM.

With an injected backend the model's geometry fields take the backend's values; setting one
to a different value raises `ValueError`, so the geometry cannot silently disagree.

```py
# pip install "pyradbio[mkm]"
from pyradbio.models.mkm.pymkm_backend import PyMKMBackend

check = MKMModel(backend=PyMKMBackend(domain_radius_um=0.32, nucleus_radius_um=3.9))
check.alpha_beta(carbon)
```

`MKMModel` mixes radiation fields like the LEM does (see
[Radiation fields and mixing](../fields.md)); it mixes on \( z^*_{1D} \), as in the MKM
paper. For RBE it needs a photon reference: pass `photon=PhotonParameters(...)`, since the
\( \alpha_0, \beta_0 \) describe the cell and not the photon beam.

## Spectrum-based models

The other classes share a base that handles cell and reference calibration, the
microdosimetry plumbing and RBE. Their default cell is HSG under a 6 MV photon reference
(`alpha_ref`, `beta_ref`, `yD_ref_keV_um`); the published constructors replace the full
set. They give an LQ pair for an ion, and they differ in how radiation quality enters:

- `ModifiedMKM`: \( \alpha \) follows the saturation-corrected dose-mean lineal energy
  \( y^*_D \); \( \beta \) is radiation-independent.
- `MCFMKMModel`: both \( \alpha \) and \( \beta \) depend on radiation quality through a
  per-event saturation correction \( c(y) \). Parisi sizes the domain from the cell's DNA
  content; [`domain_radius_from_genome`][pyradbio.core.chromatin.domain_radius_from_genome]
  computes that radius.
- `StochasticMKM` (SMK): adds nucleus-level stochastics, so \( \beta \) becomes
  quality-dependent and **survival is no longer LQ**. `alpha_beta` returns the LQ core;
  use `survival` for the curve.
- `OSMK2021`, `OSMK2023`: SMK with the oxygen axis, see [below](#the-oxygen-axis).
- `IMKModel`: modified MKM plus a repair channel and the micronucleus channel, see
  [below](#dose-rate-and-the-time-axis-imk).

### The microdosimetry engine

How an *ion* is turned into specific energies is decided by one field, `engine`:

- **No engine (default)**: the analytic impact-parameter route. It builds no \( d(y) \) at
  all; it integrates the Kiefer-Chatterjee radial dose over the site in `site_geometry`.
  `"cylinder"` (default) is the MKM's own disk geometry; `"sphere"` is selectable for
  comparison and overshoots \( z^*_{1D} \) at high energy. The result is memoised per site
  geometry across instances, so scans over cell or oxygen parameters score each ion once.
- **An injected `SpectrumEngine`**: the single-event \( d(y) \) route, scored by that
  engine in its own site. The model then takes the moments of that spectrum.
- **A spectrum you hold**: `alpha_beta_from_spectrum(spectrum)` and
  `alpha_beta_from_dose_spectrum(y_keV_um, d_of_y)` bypass the engine; you own the
  geometry.

The routes use different geometry, so their numbers need not agree. `MCFMKMModel` is the
exception: it always stays on the \( d(y) \) route, with the spherical sites of its paper.

```python
from dataclasses import replace

from pyradbio.micro.lineal_energy import AmorphousTrackEngine, SphericalSite
from pyradbio.models.mkm import ModifiedMKM
from pyradbio.track.kiefer_chatterjee import KieferChatterjeeRadialDose
from pyradbio.track.stopping_power import default_stopping_power

mmkm = ModifiedMKM.hsg_mk_inaniwa2018()
print(mmkm.alpha_beta(carbon))  # impact-parameter route, cylinder sites

# the same model, driven by the single-event d(y) of an injected engine
sp = default_stopping_power()
engine = AmorphousTrackEngine(
    radial_dose=KieferChatterjeeRadialDose(stopping_power=sp),
    site=SphericalSite(diameter_um=2.0 * mmkm.domain_radius_um),
    stopping_power=sp,
)
print(replace(mmkm, engine=engine).alpha_beta(carbon))
```

!!! note "A spectrum engine is bound to one site size"
    `SMK` needs the spectrum at the domain and at the nucleus, so it takes `engine` for the
    domain and `nucleus_engine` for the nucleus; they cannot be the same object.

The impact-parameter cache holds references to the models' stopping powers and radial doses,
and keys stopping powers by identity. After mutating one of them in place, call
[`clear_microdosimetry_cache`][pyradbio.models.mkm.spectrum_models.clear_microdosimetry_cache]:

```python
from pyradbio.models.mkm.spectrum_models import clear_microdosimetry_cache

clear_microdosimetry_cache()
```

## Published parameter sets

| Constructor | Source |
| --- | --- |
| `ModifiedMKM.hsg_mk_inaniwa2018()`, `.v79_mk_inaniwa2018()` | MK-model column of Inaniwa & Kanematsu 2018, Table 1 |
| `StochasticMKM.hsg_inaniwa2018()`, `.v79_inaniwa2018()` | SMK column of the same table |
| `OSMK2021.hsg_inaniwa2021()`, `.v79_inaniwa2021()`, `.cho_inaniwa2021()` | Inaniwa et al. 2021, Table 1 |
| `OSMK2023.hsg_inaniwa2023()`, `.v79_inaniwa2023()`, `.cho_inaniwa2023()` | Inaniwa & Kanematsu 2023, Table 1 |
| `IMKModel.hsg_matsuya2025()`, `.scc_matsuya2025()`, `.wi38_matsuya2025()`, `.du145_matsuya2025()` | Matsuya et al. 2025, Table 1 |

Each accepts keyword overrides, and `dataclasses.replace` works on the result, see the
[model overview](index.md#published-presets). A preset only applies to its own class.

!!! warning "The three HSG fits of SMK, OSMK 2021 and OSMK 2023 are independent"
    They are three separate fits to the same cell line, and the domain radius and the
    saturation parameter differ between them (`z0` is 66.0 Gy for the SMK, 88.0 Gy for
    OSMK 2021 and 60.0 Gy for OSMK 2023; `r_d` is 0.28, 0.23 and 0.29 um).
    **Never mix them.** Building an OSMK from SMK parameters plus oxygen parameters is wrong
    in every direction. Likewise, the SMK's `R_n = 8.1` um is the measured mean nucleus
    radius and enters only \( z_n \); it is not the ~3.9 um effective radius of the MK-model
    column. Use the constructor that matches your model and change a field only on purpose.

```python
from pyradbio.models.mkm import StochasticMKM

smk = StochasticMKM.hsg_inaniwa2018()
print(smk.alpha_beta(carbon))  # the LQ core (alpha_SMK, beta_SMK)
print(smk.survival(carbon, 2.0))  # eq. (24): not exp(-alpha D - beta D^2)
print(smk.survival_valid_to_Gy(carbon))  # where the truncated expansion stops being controlled
```

The SMK survival is a truncated second-order expansion. It is clipped to \( [0, 1] \), and
`survival_valid_to_Gy` reports the dose beyond which the curve is extrapolation, so check it
per case. For an expansion-free evaluation, the SMK can set `nucleus_treatment="numerical"`
(Sato & Furusawa 2012's multi-event integral); `domain_treatment="sato_moments"` selects
that paper's domain treatment. The defaults are Inaniwa & Kanematsu's.

## The oxygen axis

OSMK divides the sub-lethal damage by a relative radioresistance \( R(p\mathrm{O}_2) \) and
leaves the lethal channel alone. The oxygen tension is **an argument, not a field**: it
describes the irradiation environment, not the cell. Pass it as the keyword-only
`pO2_mmHg` to `alpha_beta`, `survival`, `dose_for_survival`, `rbe_at_survival` and
`rbe_at_dose`:

```python
import math

from pyradbio.models.mkm import OSMK2021, OSMK2023

osmk21 = OSMK2021.hsg_inaniwa2021()
osmk23 = OSMK2023.hsg_inaniwa2023()

print(osmk21.alpha_beta(carbon, pO2_mmHg=1.0))
print(osmk21.oer_at_survival(carbon, pO2_mmHg=1.0))  # OER at 10% survival
print(osmk23.alpha_beta(carbon, pO2_mmHg=1.0))
print(osmk23.alpha_beta(carbon, pO2_mmHg=math.inf))  # the strict R = 1 limit
```

`pO2_mmHg` defaults to 160 mmHg, the papers' experimental aerobic reference. That is *not*
\( R = 1 \): pass `math.inf` for the strict limit, which reproduces the SMK. The two
formulations differ only in how the fall of the OER with LET is produced: OSMK 2021 makes
\( R_\text{max} \) depend on radiation quality, OSMK 2023 shrinks the domain and rescores
the specific energies at the hypoxic geometry. The latter refuses an injected spectrum
engine for hypoxic evaluation, because an engine is bound to one site size; the default
impact-parameter route has no such limit. The RBE returned is against the photon
reference, not the combined RBE/OER of the paper's eq. (20); divide by `oer_at_survival`
for that.

## Dose rate and the time axis (IMK)

`IMKModel` is the modified MKM with an explicit repair channel. Its quadratic term is
scaled by a Lea-Catcheside factor, so it depends on how the dose is delivered. Timing is
again an argument: pass a
[`DoseSchedule`][pyradbio.core.protraction.DoseSchedule] as the keyword-only `schedule` to
`alpha_beta`, `survival`, `dose_for_survival` and `rbe_at_survival`. `schedule=None`
(default) is an acute dose. Only \( \beta \) depends on the schedule; the repair rate is a
field of the model (`repair_rate_per_h`).

```python
from pyradbio import DoseSchedule
from pyradbio.models.mkm import IMKModel

imk = IMKModel.hsg_matsuya2025()
acute = imk.alpha_beta(carbon)
slow = imk.alpha_beta(carbon, schedule=DoseSchedule.at_dose_rate(1.0, duration_h=2.0))
print(acute, slow)
print(imk.rbe_at_survival(carbon, 0.10))
print(imk.survival(carbon, 2.0, schedule=DoseSchedule.continuous(2.0, duration_h=2.0)))
```

`DoseSchedule` also builds `acute`, `fractionated` and arbitrary block schedules, and
`imk.time_factor(schedule)` returns the factor itself. A model built by hand needs
`reference_z_star_1D_Gy` before RBE routes work; the HSG and SCC VII presets carry a
recovered photon value, the WI-38 and DU145 presets do not, so pass one in. The
per-cell-line micronucleus probabilities of the paper's Table 2 are in
[`matsuya2025_h_values`][pyradbio.models.mkm.imk.matsuya2025_h_values]; do not mix a
Table 2 value into a Table 1 parameter set.

## Further reading

Paper digests (contributor notes, linked on GitHub):

- [Modified MKM: Inaniwa et al. 2010](https://github.com/e0404/pyRadBio/blob/develop/docs/references/mMKM_Inaniwa2010.md)
- [Microdosimetry engines and the MKM \( d(y) \) routes, including MCF MKM](https://github.com/e0404/pyRadBio/blob/develop/docs/references/Microdosimetry.md)
- [SMK: Inaniwa & Kanematsu 2018](https://github.com/e0404/pyRadBio/blob/develop/docs/references/SMK_Inaniwa2018.md)
- [OSMK 2021 and 2023](https://github.com/e0404/pyRadBio/blob/develop/docs/references/OSMK_Inaniwa2021_2023.md)
- [IMK: Matsuya et al. 2025](https://github.com/e0404/pyRadBio/blob/develop/docs/references/IMK_Matsuya2025.md)
- [pyMKM API notes](https://github.com/e0404/pyRadBio/blob/develop/docs/references/pyMKM_API.md)

Examples (they plot with the `plot` extra):

- [`mkm_zstar_and_alpha.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/mkm_zstar_and_alpha.py):
  carbon alpha and \( z^*_{1D} \) against LET and energy on the native backend.
- [`mcf_mkm_rbe_vs_let.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/mcf_mkm_rbe_vs_let.py):
  MCF MKM and modified MKM RBE against LET.
- [`smk_nucleus_stochastics.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/smk_nucleus_stochastics.py):
  what the SMK adds over the modified MKM.
- [`osmk_oxygen_effect.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/osmk_oxygen_effect.py):
  survival and OER at graded oxygen tension.
- [`dose_protraction_and_bed.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/dose_protraction_and_bed.py):
  the time axis behind the IMK schedule.

Known deviations from the published figures are documented in
[`ROADMAP.md`](https://github.com/e0404/pyRadBio/blob/develop/ROADMAP.md) and in the digests
above. See also the [model overview](index.md) and [Examples](../examples.md).
