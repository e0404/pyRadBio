"""Phenomenological LET-based RBE models -- the RBEmin/RBEmax family.

These models share one algebraic skeleton (Carabe-Fernandez et al. 2007,
doi:10.1080/09553000601087176): the ion LQ parameters are the photon ones
scaled by the two asymptotic RBE limits,

    RBE_max = alpha_ion / alpha_x        (the D -> 0 limit)
    RBE_min = sqrt(beta_ion / beta_x)    (the D -> inf limit)
    =>  alpha_ion = RBE_max * alpha_x,  beta_ion = RBE_min**2 * beta_x

so a published model contributes only two closed forms in ``LET_d`` and
``(alpha/beta)_x``. The acceptance below is therefore split in two: the shared
algebra (asymptotics, the LQ mapping, field handling, validity domains) is
tested once against the base class, and each published model is tested only for
its own two functions against the coefficients printed in its paper.

Equations and coefficient provenance are restated in
``docs/references/Phenomenological_RBE_models.md``.
"""

from __future__ import annotations

import math
import warnings

import pytest

from pyradbio.core.lq import survival
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion, RadiationField, SpectrumComponent
from pyradbio.models.phenomenological import (
    Carabe,
    LinearScaling,
    MairaniHelium,
    McNamara,
    RBEMinMax,
    RBEMinMaxModel,
    Wedenberg,
    WilkensOelfke,
)
from tests.conftest import LETFunction

# A generic soft-tissue photon reference: (alpha/beta)_x = 3 Gy.
PHOTON_AB3 = PhotonParameters(alpha_x=0.15, beta_x=0.05)
# An early-responding / tumour-like reference: (alpha/beta)_x = 10 Gy.
PHOTON_AB10 = PhotonParameters(alpha_x=0.30, beta_x=0.03)


class _ConstantModel(RBEMinMaxModel):
    """Minimal concrete model: fixed limits, so the base algebra is isolated."""

    model_id = "TEST"
    reference_doi = "n/a"
    radiation_modes = ("protons",)

    def _limits(self, let_keV_um: float) -> RBEMinMax:
        return RBEMinMax(rbe_min=1.2, rbe_max=2.0)


# --------------------------------------------------------------------------
# The shared skeleton
# --------------------------------------------------------------------------


def test_the_limits_map_onto_lq_parameters_as_alpha_scale_and_beta_square() -> None:
    """alpha = RBE_max*alpha_x and beta = RBE_min**2*beta_x (Carabe 2007 eq. 3).

    The square on RBE_min is the whole reason the pair is (min, max) rather than
    (alpha-scale, beta-scale): RBE_min is defined as sqrt(beta/beta_x).
    """
    model = _ConstantModel(photon=PHOTON_AB3)
    lq = model.alpha_beta_from_let(5.0)

    assert lq.alpha == pytest.approx(2.0 * PHOTON_AB3.alpha_x)
    assert lq.beta == pytest.approx(1.2**2 * PHOTON_AB3.beta_x)


def test_rbe_tends_to_rbe_max_as_dose_tends_to_zero() -> None:
    """RBE(D -> 0) = alpha_ion/alpha_x = RBE_max."""
    model = _ConstantModel(photon=PHOTON_AB3)
    approach = [model.rbe_at_dose_from_let(5.0, d) for d in (1e-2, 1e-3, 1e-4)]

    assert approach[-1] == pytest.approx(2.0, rel=1e-3)
    # and it must approach from the correct side, monotonically
    assert approach[0] < approach[1] < approach[2] < 2.0


def test_rbe_tends_to_rbe_min_as_dose_grows() -> None:
    """RBE(D -> inf) = sqrt(beta_ion/beta_x) = RBE_min."""
    model = _ConstantModel(photon=PHOTON_AB3)

    assert model.rbe_at_dose_from_let(5.0, 1e5) == pytest.approx(1.2, rel=1e-3)


def test_rbe_at_dose_agrees_with_the_core_lq_inversion() -> None:
    """The closed form and the LQ iso-survival route must give the same number.

    ``rbe_at_dose`` evaluates McNamara 2015 eq. (2) directly,
    ``RBE = (1/2D)*[sqrt(r**2 + 4*r*RBE_max*D + 4*RBE_min**2*D**2) - r]`` with
    ``r = (alpha/beta)_x`` (doi:10.1088/0031-9155/60/21/8399), because that form
    survives to arbitrarily large dose where ``S = exp(-aD-bD^2)`` underflows.
    Here it is checked against the independent route -- build alpha/beta, then
    invert through :mod:`pyradbio.core.lq` -- at a clinical dose where both work.

    They agree by the identity ``D_ion(S(D)) = D``, so this also pins the two
    RBE definitions against each other at their one common point.
    """
    model = _ConstantModel(photon=PHOTON_AB10)
    ratio = PHOTON_AB10.alpha_x / PHOTON_AB10.beta_x
    dose = 2.0

    hand_written = (
        math.sqrt(ratio**2 + 4.0 * ratio * 2.0 * dose + 4.0 * 1.2**2 * dose**2) - ratio
    ) / (2.0 * dose)
    via_core_lq = model.rbe_at_survival_from_let(5.0, model.survival_from_let(5.0, dose))

    assert model.rbe_at_dose_from_let(5.0, dose) == pytest.approx(hand_written)
    assert model.rbe_at_dose_from_let(5.0, dose) == pytest.approx(via_core_lq)


def test_survival_is_consistent_with_the_rbe_it_reports() -> None:
    """The photon dose RBE*D must reproduce the ion survival at D."""
    model = _ConstantModel(photon=PHOTON_AB3)
    dose = 2.0
    lq = model.alpha_beta_from_let(5.0)

    s_ion = survival(dose, lq.alpha, lq.beta)
    d_photon = dose * model.rbe_at_dose_from_let(5.0, dose)
    s_photon = survival(d_photon, PHOTON_AB3.alpha_x, PHOTON_AB3.beta_x)

    assert s_ion == pytest.approx(s_photon)


def test_a_zero_beta_photon_reference_is_rejected() -> None:
    """(alpha/beta)_x is the models' tissue parameter, so beta_x = 0 is undefined."""
    with pytest.raises(ValueError, match="beta_x"):
        _ConstantModel(photon=PhotonParameters(alpha_x=0.3, beta_x=0.0))


def test_negative_let_is_rejected() -> None:
    model = _ConstantModel(photon=PHOTON_AB3)
    with pytest.raises(ValueError, match="LET"):
        model.alpha_beta_from_let(-1.0)


# --------------------------------------------------------------------------
# Radiation-field entry point: the models take LET_d, so a field must be
# reduced to *one* dose-averaged LET before the model is evaluated -- never
# evaluated per component and then averaged in alpha.
# --------------------------------------------------------------------------


def test_mixed_field_dose_averages_let_before_evaluating_the_model() -> None:
    """LET_d = sum(d_i L_i)/sum(d_i), then one model evaluation.

    Averaging alpha over components instead would be a different (and wrong)
    model: RBE_max is non-linear in LET for McNamara/Mairani, so the two routes
    disagree. Here the model is linear in LET, which makes the *expected* value
    exactly computable, and a separate test pins the ordering for a non-linear
    model.
    """
    lets = {100.0: 10.0, 20.0: 30.0}  # keV/um, keyed by energy

    def stopping_power(ion: Ion) -> float:
        return lets[ion.energy_MeV_u]

    field = RadiationField(
        components=(
            SpectrumComponent(ion=Ion(Z=1, A=1, energy_MeV_u=100.0), dose_Gy=1.0),
            SpectrumComponent(ion=Ion(Z=1, A=1, energy_MeV_u=20.0), dose_Gy=3.0),
        )
    )
    model = Wedenberg(photon=PHOTON_AB3, stopping_power=LETFunction(stopping_power))

    expected_let_d = (1.0 * 10.0 + 3.0 * 30.0) / 4.0
    assert model.let_d_keV_um(field) == pytest.approx(expected_let_d)
    assert model.alpha_beta(field).alpha == pytest.approx(
        model.alpha_beta_from_let(expected_let_d).alpha
    )


def test_non_linear_model_is_evaluated_after_averaging_not_before() -> None:
    """Ordering matters: mean(f(L)) != f(mean(L)) for a non-linear RBE_max."""
    lets = {150.0: 4.0, 10.0: 40.0}

    def stopping_power(ion: Ion) -> float:
        return lets[ion.energy_MeV_u]

    field = RadiationField(
        components=(
            SpectrumComponent(ion=Ion(Z=2, A=4, energy_MeV_u=150.0), dose_Gy=1.0),
            SpectrumComponent(ion=Ion(Z=2, A=4, energy_MeV_u=10.0), dose_Gy=1.0),
        )
    )
    model = MairaniHelium(photon=PHOTON_AB3, stopping_power=LETFunction(stopping_power))

    after = model.alpha_beta(field).alpha
    before = 0.5 * (model.alpha_beta_from_let(4.0).alpha + model.alpha_beta_from_let(40.0).alpha)

    assert after == pytest.approx(model.alpha_beta_from_let(22.0).alpha)
    assert after != pytest.approx(before, rel=1e-3)


def test_a_field_needs_a_stopping_power() -> None:
    model = Wedenberg(photon=PHOTON_AB3)
    with pytest.raises(ValueError, match="stopping_power"):
        model.alpha_beta(Ion(Z=1, A=1, energy_MeV_u=100.0))


def test_a_single_ion_is_evaluated_at_its_own_let() -> None:
    """The Ion entry point is the one-component case of the field one."""
    model = Wedenberg(photon=PHOTON_AB3, stopping_power=LETFunction(lambda ion: 7.5))
    ion = Ion(Z=1, A=1, energy_MeV_u=100.0)

    assert model.let_d_keV_um(ion) == pytest.approx(7.5)
    assert model.alpha_beta(ion).alpha == pytest.approx(model.alpha_beta_from_let(7.5).alpha)


def test_a_non_positive_dose_is_rejected() -> None:
    """RBE at zero dose is the RBE_max *limit*, not a value the formula can take."""
    model = _ConstantModel(photon=PHOTON_AB3)
    with pytest.raises(ValueError, match="dose"):
        model.rbe_at_dose_from_let(5.0, 0.0)


def test_a_negative_rbe_limit_is_rejected_rather_than_squared_away() -> None:
    """beta = RBE_min**2*beta_x would turn a negative limit back into a positive beta.

    Guarding the limit itself is the only place the sign survives -- exactly what
    happens to McNamara's RBE_min far outside its fitted range.
    """
    with pytest.raises(ValueError, match="rbe_min"):
        RBEMinMax(rbe_min=-0.1, rbe_max=1.0)
    with pytest.raises(ValueError, match="rbe_max"):
        RBEMinMax(rbe_min=1.0, rbe_max=-0.1)


# --------------------------------------------------------------------------
# Validity domains. Every one of these models is a fit over a bounded range;
# extrapolating is the standard way they get misused.
# --------------------------------------------------------------------------


def test_mcnamara_warns_outside_its_fitted_let_range() -> None:
    """McNamara 2015 fitted only LET_d < 20 keV/um (doi:10.1088/0031-9155/60/21/8399)."""
    model = McNamara(photon=PHOTON_AB3)
    with pytest.warns(UserWarning, match="20"):
        model.rbe_min_max(25.0)


def test_mcnamara_warns_outside_its_fitted_alpha_beta_range() -> None:
    """McNamara 2015 fitted only (alpha/beta)_x < 30 Gy."""
    model = McNamara(photon=PhotonParameters(alpha_x=0.35, beta_x=0.01))  # 35 Gy
    with pytest.warns(UserWarning, match="30"):
        model.rbe_min_max(5.0)


def test_inside_the_fitted_range_there_is_no_warning() -> None:
    model = McNamara(photon=PHOTON_AB3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        assert model.rbe_min_max(5.0).rbe_max > 1.0


def test_helium_model_rejects_a_proton() -> None:
    """Mairani 2016's fit is helium-specific; applying it to protons is an error."""
    model = MairaniHelium(photon=PHOTON_AB3, stopping_power=LETFunction(lambda ion: 5.0))
    with pytest.raises(ValueError, match="helium"):
        model.alpha_beta(Ion(Z=1, A=1, energy_MeV_u=100.0))


# --------------------------------------------------------------------------
# The published parameterisations, one test each.
# --------------------------------------------------------------------------


@pytest.mark.golden
def test_wedenberg_alpha_is_linear_in_let_with_a_unity_rbe_min() -> None:
    """Wedenberg et al. 2013, doi:10.3109/0284186X.2012.705892, eq. (5)-(6).

    ``alpha/alpha_x = 1 + q*LET_d/(alpha/beta)_x`` with ``q = 0.434 Gy um/keV``,
    and ``beta`` independent of LET, i.e. ``RBE_min = 1`` exactly.
    """
    model = Wedenberg(photon=PHOTON_AB3)
    ratio = PHOTON_AB3.alpha_x / PHOTON_AB3.beta_x
    let = 6.0

    limits = model.rbe_min_max(let)
    assert limits.rbe_max == pytest.approx(1.0 + 0.434 * let / ratio)
    assert limits.rbe_min == 1.0  # exact, not approximate

    lq = model.alpha_beta_from_let(let)
    assert lq.beta == PHOTON_AB3.beta_x  # beta untouched


@pytest.mark.golden
def test_wedenberg_reduces_to_the_photon_reference_at_zero_let() -> None:
    """A zero-LET limit of RBE = 1 is the sanity anchor for the whole family."""
    model = Wedenberg(photon=PHOTON_AB3)
    lq = model.alpha_beta_from_let(0.0)

    assert lq.alpha == pytest.approx(PHOTON_AB3.alpha_x)
    assert lq.beta == pytest.approx(PHOTON_AB3.beta_x)
    assert model.rbe_at_dose_from_let(0.0, 2.0) == pytest.approx(1.0)


@pytest.mark.golden
def test_mcnamara_uses_the_published_fit_coefficients() -> None:
    """McNamara et al. 2015, doi:10.1088/0031-9155/60/21/8399, sec. 3.1.

    ``RBE_max = p0 + p1*LET_d/(alpha/beta)_x``,
    ``RBE_min = p2 + p3*sqrt((alpha/beta)_x)*LET_d``, with
    p0 = 0.99064, p1 = 0.35605, p2 = 1.1012, p3 = -0.0038703.

    NOTE p0: the paper prints **0.99064** (SE 0.014125), confirmed against the
    paper. Some other implementations carry 0.999064 instead, which shows up as a
    constant 0.008424 offset in RBE_max; see the reference doc.
    """
    model = McNamara(photon=PHOTON_AB3)
    ratio = PHOTON_AB3.alpha_x / PHOTON_AB3.beta_x
    let = 5.0

    limits = model.rbe_min_max(let)
    assert limits.rbe_max == pytest.approx(0.99064 + 0.35605 * let / ratio)
    assert limits.rbe_min == pytest.approx(1.1012 - 0.0038703 * math.sqrt(ratio) * let)


@pytest.mark.golden
def test_carabe_uses_the_published_fit_coefficients() -> None:
    """Carabe et al. 2012, doi:10.1088/0031-9155/57/5/1159.

    Both limits are linear in LET_d with a ``1/(alpha/beta)_x`` slope:
    ``RBE_max = 0.843 + 0.413*LET_d/(alpha/beta)_x``,
    ``RBE_min = 1.09  + 0.0161*LET_d/(alpha/beta)_x``.

    The slopes are checked against those **rounded** literature values (as quoted
    by e.g. Rorvik et al. 2018), which is why the tolerance is 2e-3 rather than
    exact: the implementation keeps the factored ``0.154*2.686`` / ``0.006*2.686``
    form, giving 0.41364 and 0.016116. Both roundings appear in the literature and
    they differ by ~0.16%; the factored form is the one implementations carry.
    """
    model = Carabe(photon=PHOTON_AB3)
    ratio = PHOTON_AB3.alpha_x / PHOTON_AB3.beta_x
    let = 5.0

    limits = model.rbe_min_max(let)
    assert limits.rbe_max == pytest.approx(0.843 + 0.413 * let / ratio, rel=2e-3)
    assert limits.rbe_min == pytest.approx(1.09 + 0.0161 * let / ratio, rel=2e-3)
    # ...and exactly against the factored form actually implemented
    assert limits.rbe_max == pytest.approx(0.843 + 0.154 * 2.686 * let / ratio)
    assert limits.rbe_min == pytest.approx(1.09 + 0.006 * 2.686 * let / ratio)


@pytest.mark.golden
def test_carabe_rbe_min_exceeds_unity_so_rbe_can_rise_with_dose() -> None:
    """Carabe is the one model here whose RBE can *increase* with dose.

    Its RBE_min = 1.09 + ... > RBE_max at high (alpha/beta)_x and low LET, so
    the D -> inf asymptote sits above the D -> 0 one (McNamara 2015 sec. 3.2
    discusses exactly this). A base class that assumed RBE_min <= RBE_max would
    silently break this model, so the ordering is pinned here.
    """
    model = Carabe(photon=PHOTON_AB10)  # (alpha/beta)_x = 10 Gy
    limits = model.rbe_min_max(1.0)

    assert limits.rbe_min > limits.rbe_max
    assert model.rbe_at_dose_from_let(1.0, 8.0) > model.rbe_at_dose_from_let(1.0, 1.0)


@pytest.mark.golden
def test_mairani_helium_uses_the_published_quadratic_exponential_form() -> None:
    """Mairani et al. 2016, doi:10.1088/0031-9155/61/2/888.

    ``RBE_max = 1 + (p0 + 1/(alpha/beta)_x) * p1*LET**2*exp(-p2*LET)``, with
    p0 = 0.136938, p1 = 0.00973154, p2 = 0.0151998, and ``RBE_min = 1``.
    """
    model = MairaniHelium(photon=PHOTON_AB3)
    ratio = PHOTON_AB3.alpha_x / PHOTON_AB3.beta_x
    let = 20.0

    f_qe = 9.73154e-3 * let**2 * math.exp(-1.51998e-2 * let)
    limits = model.rbe_min_max(let)

    assert limits.rbe_max == pytest.approx(1.0 + (1.36938e-1 + 1.0 / ratio) * f_qe)
    assert limits.rbe_min == 1.0


@pytest.mark.golden
def test_mairani_helium_rbe_max_turns_over_at_high_let() -> None:
    """``LET**2*exp(-p2*LET)`` peaks at ``LET = 2/p2`` ~ 132 keV/um.

    That maximum is the model's built-in overkill roll-over and is the one
    qualitative feature distinguishing it from the linear proton models.
    """
    model = MairaniHelium(photon=PHOTON_AB3)
    peak_let = 2.0 / 1.51998e-2

    at_peak = model.rbe_min_max(peak_let).rbe_max
    assert at_peak > model.rbe_min_max(peak_let * 0.8).rbe_max
    assert at_peak > model.rbe_min_max(peak_let * 1.2).rbe_max


# --------------------------------------------------------------------------
# Cross-model behaviour the literature agrees on.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [Wedenberg, McNamara, Carabe])
def test_rbe_rises_with_let_and_falls_with_alpha_beta(factory: type[RBEMinMaxModel]) -> None:
    """Every proton model here predicts dRBE/dLET > 0 and dRBE/d(a/b)_x < 0.

    McNamara 2015 sec. 3.2 states this as the common qualitative result of the
    three models, so it is a shared acceptance criterion rather than a per-model
    golden.
    """
    soft = factory(photon=PHOTON_AB3)
    early = factory(photon=PHOTON_AB10)

    assert soft.rbe_at_dose_from_let(8.0, 2.0) > soft.rbe_at_dose_from_let(2.0, 2.0)
    assert early.rbe_at_dose_from_let(8.0, 2.0) < soft.rbe_at_dose_from_let(8.0, 2.0)


@pytest.mark.parametrize("factory", [Wedenberg, McNamara, Carabe])
def test_proton_rbe_near_the_clinical_plateau_is_close_to_the_clinical_1_1(
    factory: type[RBEMinMaxModel],
) -> None:
    """At LET_d ~ 2.5 keV/um and 2 Gy every model should land near RBE 1.1.

    2.5 keV/um is the average plateau LET_d of McNamara 2015's simulated SOBP
    (sec. 3.2), and the clinical constant is 1.1 -- so this is the one place the
    whole family is anchored to something externally known. The band is wide on
    purpose: it is a smoke test, not a golden.
    """
    model = factory(photon=PHOTON_AB3)
    assert 1.0 < model.rbe_at_dose_from_let(2.5, 2.0) < 1.25


# --------------------------------------------------------------------------
# Wilkens & Oelfke 2004, and Frese et al. 2011's recalibration of it.
#
# These two share one functional form -- alpha linear in LET, beta fixed at
# beta_x -- and differ only in how (alpha_0, lambda) are set. Both papers print
# their own closed forms and parameter tables, so unusually for this family the
# acceptance below is against published *numbers*, not just published algebra.
# --------------------------------------------------------------------------

# Wilkens & Oelfke 2004 sec. 2.2.3: V79 vs 60Co (Tilly et al. 1999).
WILKENS_V79_ALPHA_X = 0.112  # Gy^-1
WILKENS_V79_BETA_X = 0.0298  # Gy^-2
PHOTON_V79_WILKENS = PhotonParameters(alpha_x=WILKENS_V79_ALPHA_X, beta_x=WILKENS_V79_BETA_X)


@pytest.mark.golden
def test_wilkens_alpha_is_linear_in_let_with_beta_pinned_to_the_photon_beta() -> None:
    """Wilkens & Oelfke 2004, doi:10.1088/0031-9155/49/13/004, eqs. (4)-(5).

    ``alpha_p(L) = alpha_0 + lambda*L`` with the fig. 1(a) V79 fit
    ``alpha_0 = 0.1 Gy^-1``, ``lambda = 0.02 um keV^-1 Gy^-1``; and eq. (5)
    defines ``beta_p := beta_x`` outright -- a constant beta_p which "for
    simplicity shall be the beta_x of the reference radiation" -- so
    ``RBE_min = 1`` exactly.
    """
    model = WilkensOelfke(photon=PHOTON_V79_WILKENS)
    let = 5.0

    limits = model.rbe_min_max(let)
    assert limits.rbe_min == 1.0  # exact: eq. (5) is a definition
    assert limits.rbe_max == pytest.approx((0.1 + 0.02 * let) / WILKENS_V79_ALPHA_X)

    lq = model.alpha_beta_from_let(let)
    assert lq.alpha == pytest.approx(0.1 + 0.02 * let)  # eq. (4) directly
    assert lq.beta == WILKENS_V79_BETA_X


@pytest.mark.golden
def test_wilkens_rbe_at_dose_reproduces_the_papers_own_equation_6() -> None:
    """Wilkens & Oelfke 2004 eq. (6) -- an independent check of the base class.

    ``RBE = [sqrt(alpha_x^2 + 4 beta_x D (alpha_0 + lambda L + beta_x D)) - alpha_x]
    / (2 beta_x D)``

    The base class does not code this: it builds RBE_min/RBE_max and evaluates
    McNamara's eq. (2). The two are algebraically the same only if the whole
    mapping is right, so this is a genuine cross-derivation from a second paper.
    """
    model = WilkensOelfke(photon=PHOTON_V79_WILKENS)
    a_x, b_x = WILKENS_V79_ALPHA_X, WILKENS_V79_BETA_X

    for let in (0.5, 2.0, 10.0, 25.0):
        for dose in (0.5, 2.0, 8.0):
            alpha_p = 0.1 + 0.02 * let
            expected = (math.sqrt(a_x**2 + 4.0 * b_x * dose * (alpha_p + b_x * dose)) - a_x) / (
                2.0 * b_x * dose
            )
            assert model.rbe_at_dose_from_let(let, dose) == pytest.approx(expected), (let, dose)


@pytest.mark.golden
def test_wilkens_iso_survival_rbe_reproduces_the_papers_own_equation_7() -> None:
    """Wilkens & Oelfke 2004 eq. (7) -- the same check for the *other* RBE definition.

    ``RBE(SF) = [sqrt(alpha_x^2 - 4 beta_x ln SF) - alpha_x]
    / [sqrt(alpha_p^2 - 4 beta_x ln SF) - alpha_p]``

    Having both eq. (6) and eq. (7) from one paper is what makes this pair
    valuable: they pin ``rbe_at_dose`` and ``rbe_at_survival`` independently, so
    a mix-up between the two RBE definitions cannot pass.
    """
    model = WilkensOelfke(photon=PHOTON_V79_WILKENS)
    a_x, b_x = WILKENS_V79_ALPHA_X, WILKENS_V79_BETA_X

    for let in (1.0, 10.0, 25.0):
        for survival_fraction in (0.5, 0.1, 0.01):
            alpha_p = 0.1 + 0.02 * let
            ln_sf = math.log(survival_fraction)
            expected = (math.sqrt(a_x**2 - 4.0 * b_x * ln_sf) - a_x) / (
                math.sqrt(alpha_p**2 - 4.0 * b_x * ln_sf) - alpha_p
            )
            got = model.rbe_at_survival_from_let(let, survival_fraction)
            assert got == pytest.approx(expected), (let, survival_fraction)


@pytest.mark.golden
def test_wilkens_v79_constructor_carries_the_papers_own_photon_reference() -> None:
    """Sec. 2.2.3: V79 relative to 60Co is alpha_x = 0.112, beta_x = 0.0298 (Tilly 1999)."""
    model = WilkensOelfke.v79_wilkens2004()

    assert model.photon.alpha_x == pytest.approx(WILKENS_V79_ALPHA_X)
    assert model.photon.beta_x == pytest.approx(WILKENS_V79_BETA_X)
    assert model.alpha0_Gy == pytest.approx(0.1)
    assert model.lambda_um_keV_Gy == pytest.approx(0.02)


def test_wilkens_warns_outside_its_stated_let_range() -> None:
    """Sec. 2.1 restricts the model to dose-averaged LET 0.3-30 keV/um.

    Above ~30 keV/um the paper's own fig. 1(a) shows alpha_p *decreasing*, so the
    linear form is not merely unfitted there -- it has the wrong sign of slope.
    """
    model = WilkensOelfke.v79_wilkens2004()
    with pytest.warns(UserWarning, match="30"):
        model.rbe_min_max(45.0)
    with pytest.warns(UserWarning, match="0.3"):
        model.rbe_min_max(0.1)


# -- Frese et al. 2011 -------------------------------------------------------

#: Frese et al. 2011 table 3, a subset: (tissue, alpha_x, beta_x, alpha_0).
#: alpha_x/beta_x were derived from D50, gamma and (alpha/beta) via Lof et al.
FRESE_TABLE3 = [
    ("nasopharyngeal_t1_t2", 0.1124, 0.0112, 0.1084),
    ("nasopharyngeal_t3_t4", 0.0854, 0.0085, 0.0814),
    ("subclinical_disease", 0.0554, 0.0055, 0.0514),
    ("brainstem", 0.0532, 0.0266, 0.0492),
    ("spinal_cord", 0.0407, 0.0203, 0.0367),
    ("spinal_cord_schultheiss", 0.0577, 0.0663, 0.0537),
    ("parotids_eisbruch", 0.1300, 0.0433, 0.1260),
    ("inner_ears", 0.0265, 0.0088, 0.0225),
    ("larynx", 0.0281, 0.0094, 0.0241),
]


@pytest.mark.golden
@pytest.mark.parametrize(("tissue", "alpha_x", "beta_x", "alpha_0"), FRESE_TABLE3)
def test_frese_alpha_zero_matches_the_printed_table_3(
    tissue: str, alpha_x: float, beta_x: float, alpha_0: float
) -> None:
    """Frese et al. 2011, doi:10.1016/j.ijrobp.2009.10.022, eq. (6) vs table 3.

    ``alpha_0 = alpha_x - 0.5 keV/um * lambda``, from the argument that in the
    entrance region of a proton beam the RBE is 1.0 at an entrance LET of about
    0.5 keV/um. With lambda = 0.008 that is a flat -0.004 Gy^-1 on every row, and
    table 3 prints both columns, so this is a printed golden rather than a
    derived one.
    """
    model = LinearScaling(photon=PhotonParameters(alpha_x=alpha_x, beta_x=beta_x))
    assert model.alpha0_Gy == pytest.approx(alpha_0, abs=5e-5), tissue


@pytest.mark.golden
def test_frese_rbe_is_exactly_one_at_the_entrance_let() -> None:
    """The whole calibration exists to force RBE(0.5 keV/um) = 1.

    That anchor is the model's defining property, and it holds for **any**
    tissue, since alpha_0 is defined relative to that tissue's own alpha_x.
    """
    for _tissue, alpha_x, beta_x, _alpha_0 in FRESE_TABLE3:
        model = LinearScaling(photon=PhotonParameters(alpha_x=alpha_x, beta_x=beta_x))
        assert model.rbe_min_max(0.5).rbe_max == pytest.approx(1.0)
        assert model.rbe_at_dose_from_let(0.5, 2.0) == pytest.approx(1.0)


#: Frese et al. 2011 table 4: (patient, alpha_x, beta_x, mean LET, mean D, lambda_1.1).
FRESE_TABLE4 = [
    ("A", 0.0854, 0.0085, 2.13, 2.00, 0.0074),
    ("B", 0.0854, 0.0085, 2.22, 2.00, 0.0070),
    ("C", 0.1124, 0.0112, 2.35, 2.00, 0.0086),
    ("D", 0.1124, 0.0112, 2.25, 2.00, 0.0091),
]


@pytest.mark.golden
@pytest.mark.parametrize(
    ("patient", "alpha_x", "beta_x", "mean_let", "mean_dose", "lam"), FRESE_TABLE4
)
def test_frese_lambda_calibration_reproduces_the_printed_table_4(
    patient: str, alpha_x: float, beta_x: float, mean_let: float, mean_dose: float, lam: float
) -> None:
    """Frese et al. 2011 eq. (8), against the four per-patient values in table 4.

    ``lambda_1.1 = [alpha_x (RBE-1) + beta_x D (RBE^2-1)] / (L - 0.5 keV/um)``
    is the lambda that makes the mean RBE in the target equal the clinical 1.1
    for a plan of mean LET ``L`` and mean fraction dose ``D``. Patients A/B are
    stage T3-T4 and C/D are T1-T2, which is why they use different photon
    parameters.
    """
    got = LinearScaling.lambda_for_target_rbe(
        target_rbe=1.1,
        photon=PhotonParameters(alpha_x=alpha_x, beta_x=beta_x),
        mean_let_keV_um=mean_let,
        mean_dose_Gy=mean_dose,
    )
    assert got == pytest.approx(lam, abs=5e-5), patient


@pytest.mark.golden
def test_frese_default_lambda_is_the_four_patient_average() -> None:
    """Results sec.: the average value for all 4 patients was 0.008 um/(keV Gy)."""
    lambdas = [row[-1] for row in FRESE_TABLE4]
    assert sum(lambdas) / len(lambdas) == pytest.approx(0.008, abs=5e-5)
    assert LinearScaling(photon=PHOTON_AB3).lambda_um_keV_Gy == pytest.approx(0.008)


@pytest.mark.golden
def test_frese_calibration_round_trips_through_the_model() -> None:
    """A lambda from eq. (8) must actually produce that RBE at that (L, D).

    eq. (8) is derived by inverting the model, so this closes the loop:
    calibrate, then evaluate, and the target RBE must come back.
    """
    photon = PhotonParameters(alpha_x=0.1124, beta_x=0.0112)
    lam = LinearScaling.lambda_for_target_rbe(
        target_rbe=1.1, photon=photon, mean_let_keV_um=2.35, mean_dose_Gy=2.0
    )
    model = LinearScaling(photon=photon, lambda_um_keV_Gy=lam)

    assert model.rbe_at_dose_from_let(2.35, 2.0) == pytest.approx(1.1)


def test_frese_with_zero_lambda_is_a_constant_rbe_of_one() -> None:
    """Results sec.: nonspecific normal tissues got lambda = 0.

    With lambda = 0 the model collapses to alpha_0 = alpha_x and RBE = 1 at every
    LET -- the paper assigns those tissues a flat RBE of 1.1 separately, outside
    the model.
    """
    model = LinearScaling(photon=PHOTON_AB3, lambda_um_keV_Gy=0.0)

    for let in (0.5, 5.0, 25.0):
        assert model.rbe_min_max(let).rbe_max == pytest.approx(1.0)
        assert model.rbe_at_dose_from_let(let, 2.0) == pytest.approx(1.0)


def test_wilkens_and_frese_are_the_same_form_with_different_calibration() -> None:
    """Frese's eq. (3) *is* Wilkens' eq. (4); only (alpha_0, lambda) differ.

    Pinning this keeps the two classes honest: if they ever stop agreeing when
    given the same coefficients, one of them has grown a term the other lacks.
    """
    photon = PhotonParameters(alpha_x=0.1124, beta_x=0.0112)
    frese = LinearScaling(photon=photon)
    same = WilkensOelfke(
        photon=photon,
        alpha0_Gy=frese.alpha0_Gy,
        lambda_um_keV_Gy=frese.lambda_um_keV_Gy,
    )

    for let in (1.0, 5.0, 20.0):
        assert same.rbe_min_max(let).rbe_max == pytest.approx(frese.rbe_min_max(let).rbe_max)
        assert same.rbe_min_max(let).rbe_min == pytest.approx(frese.rbe_min_max(let).rbe_min)


def test_frese_calibration_rejects_a_mean_let_at_or_below_the_entrance_let() -> None:
    """eq. (8) divides by ``L - 0.5 keV/um``, which is a real physical constraint.

    A plan whose mean target LET does not exceed the entrance LET has no LET
    contrast for the model to calibrate against; the formula would divide by zero
    or return a negative lambda.
    """
    with pytest.raises(ValueError, match="entrance LET"):
        LinearScaling.lambda_for_target_rbe(
            target_rbe=1.1,
            photon=PHOTON_AB3,
            mean_let_keV_um=0.5,
            mean_dose_Gy=2.0,
        )


# --------------------------------------------------------------------------
# The third RBE definition: RBE as a function of the REFERENCE (photon) dose
# per fraction, which is the form the family's originating paper uses.
# --------------------------------------------------------------------------


@pytest.mark.golden
def test_rbe_at_reference_dose_matches_carabe_fernandez_2007_eq_5() -> None:
    """Carabe-Fernandez, Dale & Jones 2007, doi:10.1080/09553000601087176, eq. (5).

    ``RBE = [r*RBE_max + sqrt(r^2 RBE_max^2 + 4 d_L RBE_min^2 (r + d_L))]
            / (2 (r + d_L))``  with ``r = (alpha/beta)_x``

    Note the argument is ``d_L``, the **reference** (photon) dose per fraction --
    not the ion dose that ``rbe_at_dose`` takes. Both are in use, they are
    different functions of different arguments, and conflating them is the
    obvious way to misread any of these papers.
    """
    model = _ConstantModel(photon=PHOTON_AB10)  # RBE_max 2.0, RBE_min 1.2
    ratio = PHOTON_AB10.alpha_x / PHOTON_AB10.beta_x

    for reference_dose in (0.5, 2.0, 8.0):
        expected = (
            ratio * 2.0
            + math.sqrt(
                ratio**2 * 2.0**2 + 4.0 * reference_dose * 1.2**2 * (ratio + reference_dose)
            )
        ) / (2.0 * (ratio + reference_dose))
        got = model.rbe_at_reference_dose_from_let(5.0, reference_dose)
        assert got == pytest.approx(expected), reference_dose


@pytest.mark.golden
def test_rbe_at_reference_dose_has_the_same_two_asymptotes() -> None:
    """Carabe 2007: as d_L -> 0, RBE -> RBE_max; as d_L -> inf, RBE -> RBE_min.

    The paper states both limits explicitly, and they are the definitions of the
    two parameters, so any parameterisation of the dose axis must share them.
    """
    model = _ConstantModel(photon=PHOTON_AB10)

    assert model.rbe_at_reference_dose_from_let(5.0, 1e-4) == pytest.approx(2.0, rel=1e-3)
    assert model.rbe_at_reference_dose_from_let(5.0, 1e6) == pytest.approx(1.2, rel=1e-3)


def test_the_two_dose_parameterisations_agree_where_they_describe_the_same_point() -> None:
    """``RBE`` at ion dose ``D`` equals ``RBE`` at reference dose ``RBE*D``.

    The two functions take different arguments but describe the same isoeffect,
    so they must agree when handed the matching pair. This is what makes it safe
    to have both.
    """
    model = _ConstantModel(photon=PHOTON_AB3)
    ion_dose = 2.0

    rbe = model.rbe_at_dose_from_let(5.0, ion_dose)
    assert model.rbe_at_reference_dose_from_let(5.0, rbe * ion_dose) == pytest.approx(rbe)


@pytest.mark.golden
def test_rbe_at_reference_dose_reproduces_carabe_2007_worked_example() -> None:
    """Carabe-Fernandez et al. 2007 eq. (14), the mouse-oesophagus LD50 case.

    The paper substitutes its fitted ``(alpha/beta)_L = 16.25 Gy``,
    ``RBE_max = 3.05`` and ``RBE_min = 2.27`` (table I) into eq. (5) and prints
    the resulting closed form::

        RBE = [49.58 + sqrt(2458.69 + 336.48 d_L + 20.71 d_L^2)] / (32.49 + 2 d_L)

    Checking our eq. (5) against those *printed* coefficients tests the algebra
    end-to-end. The tolerance is 1%: the printed constants are rounded from
    slightly more precise fitted values (e.g. 20.71 implies RBE_min = 2.2760
    rather than the tabulated 2.27), so exact agreement is not expected and
    demanding it would be testing their rounding, not our algebra.
    """

    class _Oesophagus(RBEMinMaxModel):
        model_id = "CDJ2007"
        reference_doi = "10.1080/09553000601087176"
        radiation_modes = ("protons",)

        def _limits(self, let_keV_um: float) -> RBEMinMax:
            return RBEMinMax(rbe_min=2.27, rbe_max=3.05)

    # (alpha/beta)_L = 16.25 Gy; only the ratio matters here.
    model = _Oesophagus(photon=PhotonParameters(alpha_x=0.1625, beta_x=0.01))

    for reference_dose in (1.0, 2.0, 5.0, 10.0):
        printed = (
            49.58 + math.sqrt(2458.69 + 336.48 * reference_dose + 20.71 * reference_dose**2)
        ) / (32.49 + 2.0 * reference_dose)
        got = model.rbe_at_reference_dose_from_let(1.0, reference_dose)
        assert got == pytest.approx(printed, rel=1e-2), reference_dose


# --------------------------------------------------------------------------
# Validity domains recovered from the primary papers (round 3).
# --------------------------------------------------------------------------


def test_wedenberg_warns_above_the_30_kev_limit_its_paper_states() -> None:
    """Wedenberg et al. 2013 after eq. (1): the model is not valid above 30 keV/um.

    "Equation 1 is not valid for LETs higher than 30 keV/mum since it does not
    account for the decrease at higher LETs." The fit itself used LET 6-30
    keV/um over 10 cell lines and 24 RBE values.
    """
    model = Wedenberg(photon=PHOTON_AB3)
    with pytest.warns(UserWarning, match="30"):
        model.rbe_min_max(40.0)


def test_wedenberg_is_silent_inside_its_fitted_range() -> None:
    model = Wedenberg(photon=PHOTON_AB3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert model.rbe_min_max(20.0).rbe_max > 1.0


def test_a_non_positive_reference_dose_is_rejected() -> None:
    """Like the ion-dose form, d_L = 0 is the RBE_max *limit*, not a value."""
    model = _ConstantModel(photon=PHOTON_AB3)
    with pytest.raises(ValueError, match="reference dose"):
        model.rbe_at_reference_dose_from_let(5.0, 0.0)
