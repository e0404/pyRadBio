# RMF, Katz and NanOx

Three further model families sit outside the [LEM](lem.md), [MKM](mkm.md) and
[phenomenological](phenomenological.md) groups. Each differs from the rest in what it models, and two
of them differ in what an "LQ model" means:

| Family | Basis | LQ? | Mixed fields | Photon reference |
| --- | --- | --- | --- | --- |
| [`RMFModel`][pyradbio.models.rmf.model.RMFModel] | DNA double-strand-break (DSB) induction and binary misrepair | yes | yes | `PhotonParameters` you supply |
| [`KatzCellularModel`][pyradbio.models.katz.model.KatzCellularModel] | ion-kill and gamma-kill from delta-ray dose | **no**, no \( \beta \) | no | its own multi-target curve |
| [`NanOxModel`][pyradbio.models.nanox.model.NanOxModel] | local lethal events in nanometric targets plus global events | no | no | calibration layer only |

The `capabilities` class attribute of each model
([`ModelCapabilities`][pyradbio.models.base.ModelCapabilities]) states these facts so a caller can
branch on them without provoking an exception.

## RMF: repair-misrepair-fixation

[`RMFModel`][pyradbio.models.rmf.model.RMFModel] implements the Repair-Misrepair-Fixation model of
Carlson, Stewart, Semenenko & Sandison 2008 (Radiat. Res. 169:447,
doi:10.1667/RR1046.1) in the formulation Frese, Yu, Stewart & Carlson 2012 (Int. J. Radiat. Oncol.
Biol. Phys. 83:442, doi:10.1016/j.ijrobp.2011.06.1983) applied to protons and carbon ions. Cell
killing is linked to DSB induction: a single DSB may be misrepaired or fixed (a one-track, linear
effect), or two DSBs interact to form a lethal exchange, within one track (intratrack, still
linear in dose) or between tracks (quadratic). The LQ parameters are

\[
\alpha = \theta\,\Sigma + \kappa\,z_F\,\Sigma^2,
\qquad
\beta = \frac{\kappa}{2}\,\Sigma^2,
\qquad
z_F = \frac{0.204\,\mathrm{LET}}{d^2},
\]

with \( \Sigma \) the DSB yield per Gy per Gbp, \( z_F \) the frequency-mean specific energy in a
site of diameter \( d \), and \( \theta \), \( \kappa \) two tissue parameters.

The two tissue parameters are not free. They are derived from the photon reference and the
radiation it was measured with (`reference_radiation`: `CO60_REFERENCE` by default, or
`X_RAY_200KVP_REFERENCE`), so the model has no free biological parameters beyond the site
diameter. The six parameter sets Frese et al. 2012 print in their Table 1 are available as
constructors, and as a dictionary:

```python
from pyradbio import Ion
from pyradbio.models.rmf import RMFModel

print(list(RMFModel.published_presets()))  # t1, hsg, v79, chordoma, prostate, head_and_neck

model = RMFModel.v79_furusawa2000()
carbon = Ion(Z=6, A=12, energy_MeV_u=100.0)

print(model.alpha_beta(carbon))  # LQParameters
print(model.rbe_at_survival(carbon, 0.1))  # iso-survival RBE at S = 0.1
print(model.alpha_terms(carbon))  # one-track and intratrack parts of alpha
print(model.rbe_dsb(carbon))  # DSB-yield ratio, which is the model's RBE_min
```

!!! warning "The site size is a diameter"
    `site_diameter_um` (default 5 um) is a nucleus **diameter**. The `R_n = 5 um` used by the LEM and
    MKM families is a **radius**. Confusing them changes \( d \) by a factor of 2 and \( z_F \) by a
    factor of 4.

### DSB yields from MCDS

The DSB yield \( \Sigma \) comes from [`pyradbio.micro.mcds`][pyradbio.micro.mcds], the closed-form
fit of the MCDS damage-yield simulations in Semenenko & Stewart 2006 (Phys. Med. Biol.
51:1693, doi:10.1088/0031-9155/51/7/004). No Monte Carlo is run. The yield is a function of the
track-quality variable \( x = Z_\text{eff}^2/\beta^2 \):

```python
from pyradbio.micro import MCDSYields
from pyradbio.micro.mcds import quality_parameter

yields = MCDSYields()
x = quality_parameter(carbon.Z, carbon.beta())
print(x, yields.dsb_per_Gy_per_Gbp(x))
print(yields.dsb_per_Gy_per_Gbp_for_ion(carbon))  # the same, from the ion
```

The fit is stated for \( 1 \le x \le 3200 \). `MCDSYields` methods take `check_domain=True`
to warn outside that range; the default is `False`.

### Validity range

Frese et al. state the RMF form holds for particles with a stopping power up to at least 100 to
150 keV/\( \mu \)m. Above `RMF_LET_CEILING_KEV_UM` (150 keV/\( \mu \)m; settable per model with
`let_ceiling_keV_um`) the model **warns and still evaluates**. The paper reports that beyond that
LET the model over-predicts \( \alpha \) and \( \beta \), because \( \Sigma \) and \( z_F \) keep
rising where the data turn over. There is no clamping and no turnover by construction:

```python
import warnings

slow_carbon = Ion(Z=6, A=12, energy_MeV_u=2.0)  # a stopping power far above the ceiling
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model.alpha_beta(slow_carbon)
print(caught[0].message)
```

Mixed fields use the inherited template (dose-averaged \( \alpha \) and \( \sqrt\beta \)); see
[Radiation fields and mixing](../fields.md). RMF is not a subclass of the phenomenological family
because its limits depend on \( (Z, \beta) \), not on a scalar \( \mathrm{LET}_d \).

Further reading: the example
[`rmf_dsb_and_binary_misrepair.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/rmf_dsb_and_binary_misrepair.py)
and the digests
[RMF_Carlson2008_Frese2012.md](https://github.com/e0404/pyRadBio/blob/develop/docs/references/RMF_Carlson2008_Frese2012.md)
and
[MCDS_SemenenkoStewart2006.md](https://github.com/e0404/pyRadBio/blob/develop/docs/references/MCDS_SemenenkoStewart2006.md).

## Katz cellular track-structure model

[`KatzCellularModel`][pyradbio.models.katz.model.KatzCellularModel] is the model of Katz, Ackerson,
Homayoonfar & Sharma 1971 (Radiat. Res. 47:402, doi:10.2307/3573247), the ancestor of the
amorphous-track models in this package. A cell is described by four fitted parameters:
`sigma0_cm2`, `E0_erg_cm3`, `kappa` and `m`. They are stored in the paper's own cgs units and
exposed in package units as `sigma0_um2` and `E0_Gy`.

A cell hit by an ion can be killed in two ways:

- **Ion-kill**: a single traversing ion kills the cell outright. `ion_kill_survival` is
  \( e^{-\sigma F} \) with the cross section \( \sigma = \sigma_0 P \).
- **Gamma-kill**: the fraction \( 1-P \) of the dose that does not act that way pools across ions
  and kills through the same multi-target statistic as gamma rays (`gamma_kill_survival`).

\( P \) is both the single-ion kill probability and the fraction of the dose routed to ion-kill.
The survivors of ion-kill are the initial population for gamma-kill, so the surviving fraction
(`survival`, eq. 10 of the paper) is the product of the two.

Ten parameter sets are printed in the paper's Figs. 6 to 15 and available as constructors and via
`KatzCellularModel.published_presets()` (spores, haploid yeast, T-1 human kidney cells under several
conditions, Chinese hamster cells, HeLa cells).

```python
from pyradbio.models.katz import KatzCellularModel

katz = KatzCellularModel.chinese_hamster_skarsgard1967()
print(list(KatzCellularModel.published_presets()))

print(katz.survival(carbon, 2.0))
print(katz.ion_kill_survival(carbon, 2.0))
print(katz.gamma_kill_survival(carbon, 2.0))
print(katz.rbe_at_survival(carbon, 0.1))
```

### Not an LQ model

Katz is **not an LQ model and has no \( \beta \)**. Its shoulder comes from the exponent `m` of the
multi-target statistic, so the low-dose gamma-kill term behaves as \( D^m \), not as \( D^2 \).
`alpha_beta` therefore returns the paper's extrapolated radiosensitivity \( k \) (eq. 14) as
`alpha` and exactly `beta = 0`. This is a descriptor and not a survival curve; use `survival` for
anything dose-resolved.

```python
print(katz.alpha_beta(carbon))  # beta is exactly 0
print(katz.capabilities)  # lq_exact=False, mixed_field=False

try:
    katz.reference()
except Exception as error:
    print(type(error).__name__)  # no LQ photon reference
```

The consequences are in `capabilities`: `lq_exact=False` and `mixed_field=False`. Mixed fields
raise `NotImplementedError`, because eq. (10) is a product of two non-LQ modes and not a dose
average. `reference()` raises `PhotonReferenceError`: the model carries its own gamma-ray curve
(`photon_survival`, from `E_0` and `m`), which `rbe_at_survival` and `rbe_at_dose` use directly.

### The `let_keV_um` keyword

The routes that take an ion use `stopping_power` (the library default) for its LET. Katz's own fits
were made against the 1960s stopping-power tables, so every equation also takes an explicit
`let_keV_um` keyword to impose a measured or tabulated LET instead:

```python
print(katz.survival(carbon, 2.0, let_keV_um=80.0))
print(katz.rbe_at_survival(carbon, 0.1, let_keV_um=80.0))
```

The same keyword is accepted by `ion_kill_survival`, `gamma_kill_survival`, `dose_for_survival` and
`rbe_at_dose`.

!!! note "Scope"
    Only the grain-count regime (eq. 8) is implemented, the regime in which the four-parameter fits
    were made. The track-width regime needs a numerically tabulated site dose that the paper gives
    only as figures, so it is not implemented (`track_width_regime_implemented` is `False`).

Further reading: the example
[`katz_ion_kill_and_gamma_kill.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/katz_ion_kill_and_gamma_kill.py)
and the digest
[Katz1971.md](https://github.com/e0404/pyRadBio/blob/develop/docs/references/Katz1971.md).

## NanOx

NanOx (NANodosimetry and OXidative stress; Cunha et al. 2017, Phys. Med. Biol. 62:1248,
doi:10.1088/1361-6560/aa54c9; Alcocer-Avila et al. 2022, Front. Phys. 10:1011063,
doi:10.3389/fphy.2022.1011063; Alcocer-Avila et al. 2023, Front. Phys. 11:1011062,
doi:10.3389/fphy.2023.1011062) factorises survival into **local lethal events** in nanometric
targets and **global events** that stand for oxidative-stress accumulation and supply the whole
shoulder.

!!! warning "Only the assembly and calibration layer is implemented"
    The ion route is not available. NanOx needs a nanodosimetric spectrum \( dP/dz \) and a relative
    chemical effectiveness \( \mathrm{RCE} \), which come from the authors' non-public Monte-Carlo
    chain and appear only as figures in the published papers. pyRadBio ships no substitute, because
    an amorphous-track stand-in would reintroduce the approximation NanOx was built to remove.
    `NanOxModel` is therefore **not** a `BiophysicalModel`: it cannot be handed to a table builder
    or the benchmark harness.

`NanOxModel.alpha_beta` always raises
[`NanOxNanodosimetryRequired`][pyradbio.models.nanox.base.NanOxNanodosimetryRequired], a subclass
of `NotImplementedError`, so that a sweep over many models treats it as "this model does not cover
this case". The message states what is missing and which methods work instead. It also raises when a
[`NanOxNanodosimetry`][pyradbio.models.nanox.base.NanOxNanodosimetry] backend is supplied, because
the configuration averaging of the 2022 paper (eqs. 36 to 43) is not implemented either.

```python
from pyradbio.models.nanox import NanOxModel, NanOxNanodosimetryRequired

nanox = NanOxModel.v79()  # also NanOxModel.cho_k1() and NanOxModel.hsg()
print(NanOxModel.capabilities)  # ion_route=False

try:
    nanox.alpha_beta(carbon)
except NanOxNanodosimetryRequired as error:
    print(isinstance(error, NotImplementedError))
    print(str(error)[:120], "...")
```

### What does work

The three published cell lines (V79, CHO-K1, HSG; Monini et al. 2020 Table 1) carry the effective
local lethal function (ELLF), the sensitive-volume geometry and \( \beta_G \). Everything
downstream of the nanodosimetric spectrum is available as explicit steps:

- Geometry and the core/penumbra split: `geometric_cross_section_um2`, `core_fraction`,
  `penumbra_fraction`.
- The photon calibration chain \( \beta_G = \beta_r/\eta^2 \): the static method `beta_G_from_photon`
  and `with_beta_G_from_photon`, which reproduces the reference radiation's LQ curve exactly in
  `photon_survival`.
- The ELLF itself, `nanox.cell_line.ellf`, and the effective number of local lethal events for a
  spectrum **you** supply as a [`LocalTargetSpectrum`][pyradbio.models.nanox.ellf.LocalTargetSpectrum]
  (`effective_local_lethal_events`, `alpha_core`).
- Track-segment closed forms: `alpha_from_cross_section`, `inactivation_cross_section_um2`,
  `survival_local`, and `survival_track_segment` (a mean-field limit that is exact for the reference
  radiation and approximate for ions).

```python
from pyradbio import PhotonParameters
from pyradbio.models.nanox import LocalTargetSpectrum

# beta_x: the V79 beta_r of Monini et al. 2018, Table 5; alpha_x is illustrative
photon = PhotonParameters(alpha_x=0.2, beta_x=0.0259)
nanox = NanOxModel.v79(photon=photon).with_beta_G_from_photon()

print(NanOxModel.beta_G_from_photon(photon.beta_x))
print(nanox.core_fraction(), nanox.penumbra_fraction())
print(nanox.photon_survival(2.0))  # the photon reference's own LQ curve

# A local lethal-event count from a spectrum you supply. This log-normal is an
# illustration of the call, not NanOx physics.
spectrum = LocalTargetSpectrum.log_normal(median_z_Gy=2.0e4, sigma_ln=1.0)
print(nanox.effective_local_lethal_events(spectrum), nanox.alpha_core(spectrum))

# Track-segment closed forms for an assumed cross section and LET.
print(nanox.alpha_from_cross_section(zeta_um2=10.0, let_keV_um=100.0))
print(nanox.survival_local(dose_Gy=2.0, zeta_um2=10.0, let_keV_um=100.0))
```

Two traps are documented in the module: the sensitive-volume size is a **diameter** in the 2017 and
2022 papers but a radius in Monini et al. 2018 to 2020, and only the later V79 parameter set ships,
because the 2017 one was produced without the core/penumbra split and is not interchangeable.

Further reading: the example
[`nanox_local_and_global_events.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/nanox_local_and_global_events.py)
and the digest
[NanOx.md](https://github.com/e0404/pyRadBio/blob/develop/docs/references/NanOx.md).
