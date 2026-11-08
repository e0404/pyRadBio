"""Goldens against Sato & Furusawa 2012 fig. A1 (doi:10.1667/RR2842.1).

The one figure in that paper with no PHITS transport in it, so the whole
``f_n(z_n; D)`` chain -- the Fermi single-event density (eqs. 24-25), the
compound-Poisson multi-event density (eqs. 26-28) and the survival integral
(eq. 7) -- can be scored against the paper with nothing else in the way.
Provenance of the digitized A-2 curve: `pyradbio.benchmark.sato2012`.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradbio.benchmark.sato2012 import (
    ALPHA_GY_INV,
    BETA_GY_INV2,
    NUCLEUS_RADIUS_UM,
    figure_a1_panel,
    survival_A3,
    survival_A4,
)
from pyradbio.core.lq import LQParameters
from pyradbio.micro.multi_event import single_event_density
from pyradbio.models.mkm.spectrum_models import StochasticMKM

_LQ = LQParameters(ALPHA_GY_INV, BETA_GY_INV2)


def _ours(panel_label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Our eq.-(7) survival on the paper's own dose grid for one panel."""
    panel = figure_a1_panel(panel_label)
    smk = StochasticMKM(nucleus_radius_um=NUCLEUS_RADIUS_UM, nucleus_treatment="numerical")
    z, mass = single_event_density(let_keV_um=panel.let_keV_um, radius_um=NUCLEUS_RADIUS_UM)
    ours = np.asarray(
        smk.survival_from_nucleus_density(panel.dose_Gy, lq=_LQ, z_Gy=z, single_event_mass=mass)
    )
    return panel.dose_Gy, panel.survival_A2, ours


# --------------------------------------------------------------------------
# The digitization itself, checked before anything is scored against it
# --------------------------------------------------------------------------
def test_figure_a1_doses_land_on_the_papers_plotting_grid() -> None:
    """The extractor was never told the grid; recovering it validates the x axis.

    0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10 Gy.

    Tolerance is **absolute** (0.02 Gy), not relative: extraction error is
    positional -- a fixed fraction of a point on the page -- so a relative bound
    would be dominated by the smallest doses for no physical reason. 0.02 Gy is
    a fifth of a point on this axis; the worst point misses by 0.0016 Gy.
    """
    expected = np.array([0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0])
    for label in ("A", "B"):
        got = figure_a1_panel(label).dose_Gy
        assert got == pytest.approx(expected, abs=0.02)


def test_panel_a_reproduces_plain_lq_at_low_dose() -> None:
    """Panel A is the paper's "small variance" case: all three curves coincide.

    Validates the y calibration against a curve known in closed form, in the
    region where the paper itself says the distinction vanishes.
    """
    panel = figure_a1_panel("A")
    low = panel.dose_Gy < 2.0
    ratio = panel.survival_A2[low] / survival_A4(panel.dose_Gy[low])
    assert np.all(np.abs(ratio - 1.0) < 0.01)


def test_unknown_panel_is_rejected() -> None:
    with pytest.raises(ValueError):
        figure_a1_panel("C")


# --------------------------------------------------------------------------
# The closed-form conventions
# --------------------------------------------------------------------------
def test_a3_reduces_to_a4_when_the_events_are_tiny() -> None:
    """``z_D -> 0`` removes the variance term and A-3 collapses onto plain LQ."""
    d = np.array([1.0, 5.0, 10.0])
    assert survival_A3(d, z_dose_mean_Gy=0.0) == pytest.approx(survival_A4(d), rel=1e-12)


def test_a3_lies_below_a4_and_the_gap_grows_with_let() -> None:
    """Moving the log inside the integral *lowers* survival, more so at high LET."""
    d = 10.0
    small = survival_A3(d, z_dose_mean_Gy=0.0163)  # ~LET 10 in a 6.2 um sphere
    large = survival_A3(d, z_dose_mean_Gy=0.326)  # ~LET 200
    assert large < small < survival_A4(d)


# --------------------------------------------------------------------------
# The golden: our eq. (7) against the paper's own A-2 curve
# --------------------------------------------------------------------------
@pytest.mark.golden
def test_golden_figure_a1_panel_a_small_variance() -> None:
    """Fig. A1(A), mono-energetic 10 keV/um.

    Reference: Sato & Furusawa 2012, doi:10.1667/RR2842.1, fig. A1 panel A.
    """
    _d, paper, ours = _ours("A")
    ratio = ours / paper
    assert np.sqrt(np.mean(np.log(ratio) ** 2)) < 0.01
    assert np.median(ratio) == pytest.approx(1.0, abs=0.01)


@pytest.mark.golden
def test_golden_figure_a1_panel_b_large_variance() -> None:
    """Fig. A1(B), mono-energetic 200 keV/um -- the panel with the content.

    Reference: Sato & Furusawa 2012, doi:10.1667/RR2842.1, fig. A1 panel B. This
    is the case the whole double-stochastic argument rests on: the nucleus
    specific energy has a large variance, and ignoring it (A-3/A-4) understates
    survival badly at high dose.

    Tolerance is 3% RMS on the log ratio. The single worst point is the last
    (10 Gy, ~7% low), where the excess is largest and therefore most sensitive to
    the Fermi straggling width -- see
    `pyradbio.micro.multi_event.DEFAULT_FERMI_WIDTH_FRACTION`, which is
    measured to move the dose-mean single-event specific energy by 9%.
    """
    _d, paper, ours = _ours("B")
    ratio = ours / paper
    assert np.sqrt(np.mean(np.log(ratio) ** 2)) < 0.03
    assert np.median(ratio) == pytest.approx(1.0, abs=0.01)


@pytest.mark.golden
def test_golden_figure_a1_reproduces_the_20x_departure_from_lq() -> None:
    """The paper's own claim, quantified: A-3/A-4 are "much smaller" than A-2.

    At 200 keV/um and 10 Gy the paper's A-2 sits a factor ~20 above plain LQ.
    A model that merely tracked LQ would pass a loose per-point tolerance while
    missing the entire effect, so the effect size is asserted directly.
    """
    panel = figure_a1_panel("B")
    _d, paper, ours = _ours("B")
    at_10 = int(np.argmax(panel.dose_Gy))
    lq_at_10 = float(survival_A4(panel.dose_Gy[at_10]))
    assert paper[at_10] / lq_at_10 == pytest.approx(20.3, rel=0.05)
    assert ours[at_10] / lq_at_10 > 15.0


def test_panel_b_effect_is_absent_in_panel_a() -> None:
    """The same machinery must *not* manufacture an excess at low LET."""
    panel_a = figure_a1_panel("A")
    at_10 = int(np.argmax(panel_a.dose_Gy))
    assert panel_a.survival_A2[at_10] / float(survival_A4(panel_a.dose_Gy[at_10])) < 1.5
