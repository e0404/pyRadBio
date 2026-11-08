# Phenomenological proton RBE

The `pyradbio.models.phenomenological` package holds thirteen closed-form RBE
parameterisations that predict an ion's LQ parameters from the **dose-averaged LET**
\( \mathrm{LET}_d \) and the tissue's photon reference. They carry no track structure and no
microdosimetry: each is a published regression fit to in-vitro survival data. They are what a
treatment-planning system can evaluate from a LET map alone, and they are the baseline the
mechanistic models ([LEM](lem.md), [MKM](mkm.md)) are worth judging against.

All thirteen derive from [`RBEMinMaxModel`][pyradbio.models.phenomenological.base.RBEMinMaxModel],
which writes the ion's LQ parameters as the photon ones scaled by the two asymptotic limits of the
RBE-versus-dose curve (Carabe-Fernandez, Dale & Jones 2007, doi:10.1080/09553000601087176):

\[
\mathrm{RBE}_{\max} = \frac{\alpha_\text{ion}}{\alpha_x}\quad (D\to 0),
\qquad
\mathrm{RBE}_{\min} = \sqrt{\frac{\beta_\text{ion}}{\beta_x}}\quad (D\to\infty),
\]

so that \( \alpha_\text{ion} = \mathrm{RBE}_{\max}\,\alpha_x \) and
\( \beta_\text{ion} = \mathrm{RBE}_{\min}^2\,\beta_x \). A published model supplies only the two
closed forms \( \mathrm{RBE}_{\max}(\mathrm{LET}_d, \cdot) \) and
\( \mathrm{RBE}_{\min}(\mathrm{LET}_d, \cdot) \); survival and RBE then follow from the shared LQ
algebra.

## The thirteen models

Every model is constructed from a
[`PhotonParameters`][pyradbio.core.photon.PhotonParameters] reference, which is model *state*:
the models are parameterised by the tissue. `alpha_beta_ratio_Gy` is derived from it. DOIs are the
`reference_doi` attribute of each class; the dissertation behind `Peeler` has none and carries a
`reference_citation` instead.

| Class | Publication | DOI | Tissue input |
| --- | --- | --- | --- |
| [`WilkensOelfke`][pyradbio.models.phenomenological.wilkens.WilkensOelfke] | Wilkens & Oelfke 2004, Phys. Med. Biol. 49:2811 | 10.1088/0031-9155/49/13/004 | \( \alpha_x \) |
| [`LinearScaling`][pyradbio.models.phenomenological.frese.LinearScaling] | Frese et al. 2011, Int. J. Radiat. Oncol. Biol. Phys. 79:80 | 10.1016/j.ijrobp.2009.10.022 | \( \alpha_x \) |
| [`Carabe`][pyradbio.models.phenomenological.carabe.Carabe] | Carabe et al. 2012, Phys. Med. Biol. 57:1159 | 10.1088/0031-9155/57/5/1159 | \( (\alpha/\beta)_x \) |
| [`Wedenberg`][pyradbio.models.phenomenological.wedenberg.Wedenberg] | Wedenberg et al. 2013, Acta Oncol. 52:580 | 10.3109/0284186X.2012.705892 | \( (\alpha/\beta)_x \) |
| [`McNamara`][pyradbio.models.phenomenological.mcnamara.McNamara] | McNamara et al. 2015, Phys. Med. Biol. 60:8399 | 10.1088/0031-9155/60/21/8399 | \( (\alpha/\beta)_x \) |
| [`MairaniHelium`][pyradbio.models.phenomenological.mairani.MairaniHelium] | Mairani et al. 2016, Phys. Med. Biol. 61:888 (helium) | 10.1088/0031-9155/61/2/888 | \( (\alpha/\beta)_x \) |
| [`Tilly2`][pyradbio.models.phenomenological.tilly.Tilly2] | Tilly et al. 2005, Phys. Med. Biol. 50:2765 (low-ratio fit) | 10.1088/0031-9155/50/12/003 | \( (\alpha/\beta)_x \) |
| [`Tilly10`][pyradbio.models.phenomenological.tilly.Tilly10] | Tilly et al. 2005, Phys. Med. Biol. 50:2765 (high-ratio fit) | 10.1088/0031-9155/50/12/003 | \( (\alpha/\beta)_x \) |
| [`Chen`][pyradbio.models.phenomenological.chen.Chen] | Chen & Ahmad 2012, Radiat. Prot. Dosimetry 149:116 | 10.1093/rpd/ncr218 | \( \alpha_x \) |
| [`Jones`][pyradbio.models.phenomenological.jones.Jones] | Jones 2015b, Cancers 7:460 | 10.3390/cancers7010460 | \( \alpha_x \), \( \beta_x \) |
| [`RorvikUnweighted`][pyradbio.models.phenomenological.rorvik.RorvikUnweighted] | Rorvik et al. 2017, Med. Phys. 44:2586 (unweighted arm) | 10.1002/mp.12216 | \( (\alpha/\beta)_x \) |
| [`Peeler`][pyradbio.models.phenomenological.peeler.Peeler] | Peeler 2016, PhD dissertation, as reformulated in Rorvik et al. 2018 | none (see above; Rorvik 2018 is 10.1088/1361-6560/aad9db) | \( (\alpha/\beta)_x \) |
| [`Unkelbach`][pyradbio.models.phenomenological.unkelbach.Unkelbach] | Unkelbach et al. 2016, Int. J. Radiat. Oncol. Biol. Phys. 96:1097 | 10.1016/j.ijrobp.2016.08.038 | none (\( \mathrm{LET}_d \) only) |

All are proton fits except `MairaniHelium`, which is fitted for helium. `radiation_modes` is
enforced: a model asked for another ion raises a `ValueError` rather than extrapolating.
`WilkensOelfke`, `LinearScaling` and `Unkelbach` expose their fitted constants as constructor
arguments (`alpha0_Gy`, `lambda_um_keV_Gy`, `entrance_let_keV_um`, `slope_um_keV`);
`LinearScaling.lambda_for_target_rbe` recalibrates \( \lambda \) for another plan.

!!! warning "Uneven provenance"
    The coefficients were not all read from the primary paper with the same care. The Tilly slopes
    are second-hand (the paper prints none, they come from Rorvik et al.), and `Unkelbach` is
    the RBE implied by an LET-optimisation method, not a survival fit. Read the model digest
    [Phenomenological_RBE_models.md](https://github.com/e0404/pyRadBio/blob/develop/docs/references/Phenomenological_RBE_models.md)
    before trusting one. The class docstrings of the
    [API reference][pyradbio.models.phenomenological] carry the equations and caveats per model.

Each class declares the range it was fitted on (`valid_let_keV_um`,
`valid_alpha_beta_ratio_Gy`). Outside that box the model **warns and returns the extrapolation**; it
never clamps. A range of `None` means the paper states none.

## The LET route

A treatment-planning system holds a LET map, not an ion spectrum, so the natural entry point takes
the LET directly and needs no stopping power. Both examples below use a photon reference with
\( (\alpha/\beta)_x = 3 \) Gy.

```python
from pyradbio import PhotonParameters
from pyradbio.models.phenomenological import McNamara

photon = PhotonParameters(alpha_x=0.15, beta_x=0.05)
model = McNamara(photon)

limits = model.rbe_min_max(3.0)  # an RBEMinMax(rbe_min=..., rbe_max=...)
lq = model.alpha_beta_from_let(3.0)  # LQParameters(alpha=..., beta=...)
s2 = model.survival_from_let(3.0, 2.0)  # surviving fraction after 2 Gy
d10 = model.dose_for_survival_from_let(3.0, 0.1)  # ion dose [Gy] at S = 0.1
rbe_10 = model.rbe_at_survival_from_let(3.0, 0.1)  # iso-survival RBE at S = 0.1
```

`LET` is the dose-averaged LET in keV/\( \mu \)m. The survival and dose methods are those of every
[`LETModel`][pyradbio.models.base.LETModel]; `survival_from_let` also accepts an array of doses.

## Three definitions of "RBE at a dose"

The family is used with three RBE definitions that are **not** interchangeable. The code keeps them
as separate, explicitly named methods:

| Method | Argument | Definition |
| --- | --- | --- |
| `rbe_at_survival_from_let(let, S)` | a surviving fraction \( S \) | iso-survival RBE \( D_\text{ref}(S)/D_\text{ion}(S) \) |
| `rbe_at_dose_from_let(let, D)` | the **ion** dose \( D \) | \( D_\text{ref}/D \) for the same effect \( -\ln S \) |
| `rbe_at_reference_dose_from_let(let, d_L)` | the **reference (photon)** dose per fraction \( d_L \) | Carabe-Fernandez et al. 2007 eq. (5) |

The last two are both called "RBE at 2 Gy" in the literature, and they are different functions: at
the same numeric argument they are not equal. They agree only on matching pairs,
\( d_L = \mathrm{RBE} \cdot D_\text{ion} \) (the base class documents that a test pins this
correspondence). For a pure LQ reference, `rbe_at_dose_from_let` is the closed form of McNamara et
al. 2015 (doi:10.1088/0031-9155/60/21/8399, eq. 2) with \( r = (\alpha/\beta)_x \):

\[
\mathrm{RBE}(D) = \frac{\sqrt{r^2 + 4 r\,\mathrm{RBE}_{\max} D + 4\,\mathrm{RBE}_{\min}^2 D^2} - r}{2D}.
\]

```python
rbe_at_ion_dose = model.rbe_at_dose_from_let(3.0, 2.0)  # 2 Gy of protons
rbe_at_photon_dose = model.rbe_at_reference_dose_from_let(3.0, 2.0)  # 2 Gy photon-equivalent

# The two agree only on matching pairs: d_L = RBE * D_ion.
d_ref = rbe_at_ion_dose * 2.0
assert abs(model.rbe_at_reference_dose_from_let(3.0, d_ref) - rbe_at_ion_dose) < 1e-9
```

If the photon reference carries a `D_t` (an exponential tail above a threshold dose), the RBE
methods invert that tail consistently. That is an explicit reference extension and not part of the
published LQ fits; the high-dose \( \mathrm{RBE}_{\min} \) limit applies only when `D_t is None`.

!!! note "Do not assume ordering or unity"
    \( \mathrm{RBE}_{\min} \) is not constrained to be below \( \mathrm{RBE}_{\max} \) (for
    `Carabe` it can exceed it, so RBE may rise with dose), and several models do not return an RBE
    of 1 at zero LET: `Carabe` and `WilkensOelfke` do not reduce to the photon reference there.
    That is a property of the fits.

## The ion and field route

The same models also implement the common
[`BiophysicalModel`][pyradbio.models.base.BiophysicalModel] interface (`alpha_beta`,
`survival`, `rbe_at_dose`, `rbe_at_survival`) for an
[`Ion`][pyradbio.core.radiation.Ion] or a
[`RadiationField`][pyradbio.core.radiation.RadiationField]. This needs a stopping power to turn the
ion or field into a dose-averaged LET; pass one at construction, or the call raises. Use the
library default:

```python
from pyradbio import Ion, RadiationField, SpectrumComponent
from pyradbio.track.stopping_power import default_stopping_power

model = McNamara(photon, stopping_power=default_stopping_power())

proton = Ion(Z=1, A=1, energy_MeV_u=100.0)
print(model.alpha_beta(proton))
print(model.rbe_at_dose(proton, 2.0))

field = RadiationField(
    [
        SpectrumComponent(Ion(1, 1, 60.0), dose_Gy=1.0),
        SpectrumComponent(Ion(1, 1, 10.0), dose_Gy=1.0),
    ]
)
print(model.let_d_keV_um(field))  # dose-averaged LET of the field
print(model.alpha_beta(field))
```

A field is reduced to a single \( \mathrm{LET}_d \) **first**, and the model is evaluated once on
it. It is never evaluated per component and then averaged in \( \alpha \), unlike the
[default mixing rule](../fields.md) of the other families. The two orderings differ whenever the
model is non-linear in LET, and the first is the definition of these models: they are fits to RBE as
a function of one \( \mathrm{LET}_d \). Binned components are integrated on a log sub-grid, as
described in [Radiation fields and mixing](../fields.md).

The class attribute `capabilities` records what each model supports: every member sets
`let_route=True`, and none has a `spectrum_route`.

## Choosing a model

The thirteen are fitted to different data sets, so choosing one is a modelling choice and not a
measurement. `Wedenberg`, `Tilly2`, `Tilly10` and
`RorvikUnweighted`, for instance, share one equation, \( 1 + q\,\mathrm{LET}_d/(\alpha/\beta)_x \)
with \( \mathrm{RBE}_{\min}=1 \), and differ only in the slope \( q \) of their data base. A
sensible use is to evaluate several and compare them, which is what the example does:
[`examples/phenomenological_rbe_models.py`](https://github.com/e0404/pyRadBio/blob/develop/examples/phenomenological_rbe_models.py)
(needs matplotlib).
