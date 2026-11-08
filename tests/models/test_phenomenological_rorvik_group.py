"""The six models added from Rorvik et al. 2018's appendix-B reformulations.

Rorvik et al. 2018 (Phys. Med. Biol. 63:185013, doi:10.1088/1361-6560/aad9db)
restates eleven published proton-RBE models as a pair of closed forms for
``RBE_max`` and ``RBE_min`` -- exactly this package's
`pyradbio.models.phenomenological.base.RBEMinMaxModel` skeleton. Six of
those eleven were already implemented from their own primary papers; the models
here are the remainder that a scalar ``LET_d`` can express.

**Provenance is one tier weaker than the rest of the family.** These coefficients
come from a *secondary* source: none of the primary papers is available here. The
source is validated rather than assumed -- ``test_rorvik_appendix_reproduces_our
_independently_read_models`` checks its reformulations of the five models we did
read from their primaries, and they agree exactly. See
``docs/references/Phenomenological_RBE_models.md``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import ClassVar

import pytest

from pyradbio.core.photon import PhotonParameters
from pyradbio.models.phenomenological import (
    Carabe,
    Chen,
    Jones,
    LinearScaling,
    McNamara,
    Peeler,
    RBEMinMax,
    RorvikUnweighted,
    Tilly2,
    Tilly10,
    Unkelbach,
    Wedenberg,
    WilkensOelfke,
)

# The photon reference Rorvik's appendix uses for its Wilkens/Chen worked cases.
RORVIK_ALPHA_X = 0.112
RORVIK_BETA_X = 0.0298
RORVIK_RATIO = RORVIK_ALPHA_X / RORVIK_BETA_X  # 3.758... Gy, the paper's "3.76"


def _photon(ratio_Gy: float = RORVIK_RATIO, beta_x: float = RORVIK_BETA_X) -> PhotonParameters:
    return PhotonParameters(alpha_x=ratio_Gy * beta_x, beta_x=beta_x)


def _quiet(model: object, let: float) -> object:
    """Evaluate outside a fitted domain without the warning failing the test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.rbe_min_max(let)  # type: ignore[attr-defined]


# --- the source itself -----------------------------------------------------


def test_rorvik_appendix_reproduces_our_independently_read_models() -> None:
    """Why these coefficients are trustworthy despite being second-hand.

    Rorvik's appendix B also reformulates five models this package implemented
    from their *own* papers. If its transcription of those is exact, its
    transcription of the rest is credible. (Mairani is excluded: we implement the
    fQE weighting the paper itself selects, Rorvik the revised fLE2 -- a known,
    documented divergence, not a disagreement about the same number.)
    """
    let = 10.0
    r = RORVIK_RATIO
    photon = _photon()

    # WIL: "RBE_max = 0.892 + 0.179 (keV/um)^-1 LET_d" at alpha_x = 0.112 Gy^-1.
    wil = _quiet(WilkensOelfke(photon=PhotonParameters(alpha_x=0.112, beta_x=0.0298)), let)
    assert wil.rbe_max == pytest.approx(0.892 + 0.179 * let, rel=2e-3)  # paper rounds to 3 dp
    # FRE: "RBE_max = 1 + 0.008/alpha_x (LET_d - 0.5 keV/um)".
    fre = _quiet(LinearScaling(photon=PhotonParameters(alpha_x=0.112, beta_x=0.0298)), let)
    assert fre.rbe_max == pytest.approx(1.0 + 0.008 / 0.112 * (let - 0.5))
    # WED / CAR / MCN, term for term.
    assert _quiet(Wedenberg(photon=photon), let).rbe_max == pytest.approx(1 + 0.434 * let / r)
    car = _quiet(Carabe(photon=photon), let)
    assert car.rbe_max == pytest.approx(0.843 + 0.154 * 2.686 * let / r)
    assert car.rbe_min == pytest.approx(1.09 + 0.006 * 2.686 * let / r)
    mcn = _quiet(McNamara(photon=photon), let)
    assert mcn.rbe_max == pytest.approx(0.99064 + 0.35605 * let / r)
    assert mcn.rbe_min == pytest.approx(1.1012 - 0.0038703 * math.sqrt(r) * let)


# --- Tilly (TIL2 / TIL10) --------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "q"),
    [(Tilly2, 0.309), (Tilly10, 0.550964)],
)
def test_tilly_is_a_linear_slope_in_let_over_the_tissue_ratio(cls: type, q: float) -> None:
    let = 12.0
    limits = _quiet(cls(photon=_photon()), let)
    assert limits.rbe_max == pytest.approx(1.0 + q * let / RORVIK_RATIO)
    assert limits.rbe_min == 1.0


def test_tilly_splits_into_two_classes_because_the_databases_differ() -> None:
    """TIL2 and TIL10 are two fits to two (alpha/beta)_x bands, not one model.

    Table A.1: TIL2 covers 2.7-2.8 Gy, TIL10 covers 7.7-15 Gy. A single class with
    a swappable slope could not carry two different fitted domains.
    """
    assert Tilly2.valid_alpha_beta_ratio_Gy == (2.7, 2.8)
    assert Tilly10.valid_alpha_beta_ratio_Gy == (7.7, 15.0)
    assert Tilly2.valid_let_keV_um == (7.7, 20.0)
    assert Tilly10.valid_let_keV_um == (7.7, 19.7)


# --- Chen ------------------------------------------------------------------


def test_chen_matches_the_primarys_alpha_equation() -> None:
    """Chen & Ahmad eq. (8) is a statement about alpha_p, not about RBE_max."""
    let, alpha_x = 15.0, 0.112
    expected_alpha = 0.1 + (1.0 - math.exp(-0.0013 * let * let)) / (0.045 * let)
    model = Chen(photon=PhotonParameters(alpha_x=alpha_x, beta_x=0.0298))
    assert model.alpha_ion_Gy(let) == pytest.approx(expected_alpha)
    assert _quiet(model, let).rbe_max == pytest.approx(expected_alpha / alpha_x)
    assert _quiet(model, let).rbe_min == 1.0


def test_chen_is_finite_at_zero_let() -> None:
    """Eq. (8) is 0/0 at LET = 0; the limit is alpha_p = alpha_0.

    (1 - exp(-l1 L^2)) / (l2 L) -> (l1/l2) L -> 0, so a naive implementation
    raises ZeroDivisionError exactly at the family's photon anchor.
    """
    assert Chen(photon=_photon()).alpha_ion_Gy(0.0) == pytest.approx(0.1)


def test_chen_keys_on_alpha_x_like_wilkens_not_on_the_ratio() -> None:
    """alpha_0 = 0.1 is a V79 ion-response intercept, so alpha_x is its axis."""
    let = 15.0
    a = _quiet(Chen(photon=PhotonParameters(alpha_x=0.2, beta_x=0.05)), let)  # r = 4
    b = _quiet(Chen(photon=PhotonParameters(alpha_x=0.4, beta_x=0.1)), let)  # r = 4
    assert a.rbe_max != pytest.approx(b.rbe_max)
    assert Chen.valid_alpha_beta_ratio_Gy == (2.2, 2.8)  # its V79 database


def test_chen_primary_disagrees_with_the_rorvik_reformulation() -> None:
    """Rorvik's appendix drops the 1/alpha_x on the LET term -- a factor of 10.

    Checked on the paper's own V79 worked case (fig. 6: alpha_x = 0.1 Gy^-1,
    beta_x = 0.03 Gy^-2) at the top of its LET range, where eq. (8) gives an
    RBE_10 above 2 -- consistent with the measured V79 points the figure plots --
    and the reformulation collapses to the entrance-plateau value of 1.1.
    """
    let, alpha_x, beta_x = 30.0, 0.1, 0.03
    model = Chen(photon=PhotonParameters(alpha_x=alpha_x, beta_x=beta_x))
    term = (1.0 - math.exp(-0.0013 * let * let)) / (0.045 * let)

    rorvik_rbe_max = 1.0 + term
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = model.rbe_at_survival_from_let(let, 0.1)
    # Ours follows eq. (8); Rorvik's alpha would be (1 + term) * alpha_x.
    assert _quiet(model, let).rbe_max == pytest.approx((0.1 + term) / alpha_x)
    assert _quiet(model, let).rbe_max == pytest.approx(10.0 * rorvik_rbe_max - 9.0)
    assert ours > 2.0


# --- Jones -----------------------------------------------------------------


def _jones_expected(let: float, alpha_x: float, beta_x: float) -> tuple[float, float]:
    # Jones eqs. (1) and (2): alpha_U = (Q/j)(1 - e^-j.alpha_L), Q = 10.62, j = 3.92;
    # beta_U = (R/u)(1 - e^-u.beta_L), R = 2.5, u = 25.
    alpha_u = (10.62 / 3.92) * (1.0 - math.exp(-3.92 * alpha_x))
    beta_u = (2.5 / 25.0) * (1.0 - math.exp(-25.0 * beta_x))
    scale = (let - 0.22) / (30.5 - 0.22)
    return (
        math.sqrt(1.0 + scale * (beta_u / beta_x - 1.0)),
        1.0 + scale * (alpha_u / alpha_x - 1.0),
    )


def test_jones_beta_ceiling_is_the_papers_stated_value() -> None:
    """R/u = 0.1 Gy^-2 is exactly the "maximum ceiling value" the paper states.

    Rorvik's appendix prints 0.06 with u = 50 instead; the paper's own sentence
    is what settles it, and it is why this model was rebuilt from the primary.
    """
    huge_beta = Jones(photon=PhotonParameters(alpha_x=1.0, beta_x=100.0))
    assert huge_beta.beta_saturated_Gy2 == pytest.approx(0.1)


def test_jones_turnover_let_comes_from_the_primary() -> None:
    """30.5 keV/um is the paper's stated turnover; 30.5 - 0.22 = Rorvik's 30.28."""
    model = Jones(photon=_photon())
    assert model._let_scale_keV_um == pytest.approx(30.28)


def test_jones_matches_the_printed_closed_forms() -> None:
    let, alpha_x, beta_x = 12.0, 0.15, 0.05
    rbe_min, rbe_max = _jones_expected(let, alpha_x, beta_x)
    limits = _quiet(Jones(photon=PhotonParameters(alpha_x=alpha_x, beta_x=beta_x)), let)
    assert limits.rbe_max == pytest.approx(rbe_max)
    assert limits.rbe_min == pytest.approx(rbe_min)


def test_jones_depends_on_alpha_x_and_beta_x_separately_not_on_their_ratio() -> None:
    """Like Wilkens and Frese: an "RBE vs (alpha/beta)_x" sweep is not its axis."""
    let = 12.0
    a = _quiet(Jones(photon=PhotonParameters(alpha_x=0.2, beta_x=0.05)), let)  # r = 4
    b = _quiet(Jones(photon=PhotonParameters(alpha_x=0.4, beta_x=0.1)), let)  # r = 4
    assert a.rbe_max != pytest.approx(b.rbe_max)


def test_jones_reduces_to_the_photon_reference_at_its_own_anchor_let() -> None:
    """At LET = 0.22 keV/um the scale factor vanishes, so both limits are 1."""
    limits = _quiet(Jones(photon=PhotonParameters(alpha_x=0.15, beta_x=0.05)), 0.22)
    assert limits.rbe_max == pytest.approx(1.0)
    assert limits.rbe_min == pytest.approx(1.0)


def test_jones_rejects_a_let_where_its_rbe_min_would_go_imaginary() -> None:
    """beta_u < beta_x makes the radicand fall with LET and eventually go negative.

    A bare math.sqrt would raise ValueError("math domain error") with no hint of
    which model or which input caused it.
    """
    model = Jones(photon=PhotonParameters(alpha_x=0.5, beta_x=0.5))
    with pytest.raises(ValueError, match="JON"):
        _quiet(model, 300.0)


# --- Rorvik (unweighted) ---------------------------------------------------


def test_rorvik_unweighted_matches_the_printed_slope() -> None:
    let = 12.0
    limits = _quiet(RorvikUnweighted(photon=_photon()), let)
    assert limits.rbe_max == pytest.approx(1.0 + 0.645 * let / RORVIK_RATIO)
    assert limits.rbe_min == 1.0


# --- Peeler ----------------------------------------------------------------


def test_peeler_matches_the_printed_cubic() -> None:
    let, r = 4.0, RORVIK_RATIO
    limits = _quiet(Peeler(photon=_photon()), let)
    assert limits.rbe_max == pytest.approx(0.75 + 0.00143 * let**3 / r)
    assert limits.rbe_min == pytest.approx(1.24 + 0.00074 * r * let**3)


def test_peeler_domains_are_not_transposed() -> None:
    """The dissertation's 12 LET_d values span 0.94-19.4 keV/um, not 2.6-5.1.

    Rorvik's table A.1 carries both ranges and this package first read its two
    columns the wrong way round, which put a cubic's fit window an order of
    magnitude too low. 2.6-5.1 Gy is the *tissue* range: its two cell lines are
    H1437 at (alpha/beta)_x = 0.08/0.031 = 2.6 Gy and H460 at 0.36/0.071 = 5.1 Gy
    (its table 4).
    """
    assert Peeler.valid_let_keV_um == (0.9, 19.0)
    assert Peeler.valid_alpha_beta_ratio_Gy == (2.6, 5.1)
    # inside the LET window, outside the tissue window -> exactly one warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Peeler(photon=_photon(ratio_Gy=12.0)).rbe_min_max(10.0)
    assert len(caught) == 1
    assert "alpha/beta" in str(caught[0].message)


def test_peeler_does_not_reduce_to_unity_at_zero_let() -> None:
    """RBE_max(0) = 0.75 and RBE_min(0) = 1.24 -- like Carabe, it has no anchor."""
    limits = _quiet(Peeler(photon=_photon()), 0.0)
    assert limits.rbe_max == pytest.approx(0.75)
    assert limits.rbe_min == pytest.approx(1.24)


# --- Unkelbach -------------------------------------------------------------


def test_unkelbach_has_equal_limits_so_its_rbe_is_dose_independent() -> None:
    """RBE_max = RBE_min is what makes this an LET-optimisation ansatz, not a fit."""
    let = 5.0
    model = Unkelbach(photon=_photon())
    limits = _quiet(model, let)
    assert limits.rbe_max == pytest.approx(1.0 + 0.04 * let)
    assert limits.rbe_min == pytest.approx(limits.rbe_max)
    lo = model.rbe_at_dose_from_let(let, 0.01)
    hi = model.rbe_at_dose_from_let(let, 50.0)
    assert lo == pytest.approx(hi, rel=1e-9)


def test_unkelbach_slope_is_the_papers_own_sobp_calibration() -> None:
    """c = 0.04 um/keV gives RBE = 1.1 at the SOBP mean LET_d of 2.5 keV/um."""
    model = Unkelbach(photon=_photon())
    assert _quiet(model, 2.5).rbe_max == pytest.approx(1.1)


def test_unkelbach_has_no_mcmahon_variant() -> None:
    """Rorvik "McMahon configuration ... 0.055" is in neither the primary nor here.

    The paper own alternatives are c ~ 0.02 and c ~ 0.05, from linearising
    McNamara, and are offered as a sensitivity range rather than as fitted
    constants -- so they are constants to pass, not a preset to call.
    """
    assert not hasattr(Unkelbach, "mcmahon")
    let = 5.0
    hi = _quiet(Unkelbach(photon=_photon(), slope_um_keV=0.05), let)
    assert hi.rbe_max == pytest.approx(1.0 + 0.05 * let)


def test_unkelbach_is_tissue_independent() -> None:
    let = 5.0
    a = _quiet(Unkelbach(photon=_photon(ratio_Gy=2.0)), let)
    b = _quiet(Unkelbach(photon=_photon(ratio_Gy=15.0)), let)
    assert a.rbe_max == pytest.approx(b.rbe_max)


# --- family-wide ------------------------------------------------------------

NEW_MODELS = (Tilly2, Tilly10, Chen, Jones, RorvikUnweighted, Peeler, Unkelbach)


@pytest.mark.parametrize("cls", NEW_MODELS)
def test_every_new_model_is_a_proton_fit_that_cites_itself(cls: type) -> None:
    assert cls.radiation_modes == ("protons",)  # type: ignore[attr-defined]
    assert cls.applicable_z() == frozenset({1})  # type: ignore[attr-defined]
    # A DOI, or -- for Peeler, a dissertation -- a full citation. Never neither.
    assert cls.reference_doi or cls.reference_citation  # type: ignore[attr-defined]


def test_peeler_cites_the_dissertation_not_the_clinical_paper() -> None:
    """Rorvik's table 1 credits "(Peeler 2016)", single-author: the dissertation.

    The Radiother. Oncol. paper of the same year is a clinical-outcomes study and
    fits none of the 48 in-vitro points this model is credited with. Putting its
    DOI here would misattribute the coefficients.
    """
    assert Peeler.reference_doi == ""
    assert "dissertation" in Peeler.reference_citation
    assert "radonc" not in Peeler.reference_citation
    # and the warning still says where the model comes from
    assert Peeler.reference_label() == Peeler.reference_citation


def test_a_model_declaring_neither_doi_nor_citation_is_rejected() -> None:
    from pyradbio.models.phenomenological.base import RBEMinMaxModel

    @dataclass(frozen=True, slots=True)
    class _Anonymous(RBEMinMaxModel):
        model_id: ClassVar[str] = "ANON"
        reference_doi: ClassVar[str] = ""

        def _limits(self, let_keV_um: float) -> RBEMinMax:
            return RBEMinMax(rbe_min=1.0, rbe_max=1.0)

    with pytest.raises(ValueError, match="reference_doi"):
        _Anonymous(photon=_photon())


@pytest.mark.parametrize("cls", NEW_MODELS)
def test_every_new_model_maps_its_limits_onto_lq(cls: type) -> None:
    photon = _photon()
    let = 10.0
    model = cls(photon=photon)  # type: ignore[call-arg]
    limits = _quiet(model, let)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lq = model.alpha_beta_from_let(let)
    assert lq.alpha == pytest.approx(limits.rbe_max * photon.alpha_x)
    assert lq.beta == pytest.approx(limits.rbe_min**2 * photon.beta_x)


@pytest.mark.parametrize("cls", NEW_MODELS)
def test_every_new_model_has_a_distinct_model_id(cls: type) -> None:
    assert cls.model_id  # type: ignore[attr-defined]


def test_new_model_ids_are_unique() -> None:
    ids = [c.model_id for c in NEW_MODELS]  # type: ignore[attr-defined]
    assert len(set(ids)) == len(ids)
