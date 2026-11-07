"""Inaniwa et al. 2021 figs. 5-7 -- the OSMK oxygen-enhancement-ratio benchmark.

doi:10.1088/1361-6560/ac29cc. The digitization and its traps are documented in
:mod:`pyradbio.benchmark.inaniwa2021`; the reconstruction result and the open
V79 discrepancy in ``docs/references/OSMK_Inaniwa2021_2023.md``.

Two layers:

* **digitization integrity** -- the thirty series load, cover the LET range the
  paper fitted, and satisfy properties the figures themselves assert (the OER
  falls from ~3 towards unity; the model curve passes through its own points);
* **@golden reconstruction** -- our :class:`OSMK2021`, driven from nothing but
  the ion species, the quoted LET and the oxygen pressure, against the paper's
  own curves. CHO (fig. 7) lands at RMS 0.4% over four species and four
  pressures with the published parameters untouched, and HSG at 3.1%; V79 at
  9.3% is a known open gap, pinned here so it cannot silently drift.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from pyradbio.benchmark.inaniwa2021 import (
    CONTENTS,
    FITTED_LET_RANGE_KEV_UM,
    INANIWA2021_DOI,
    SPECIES,
    OERCurve,
    get_oer_curve,
    load_oer_curves,
    measured_points,
    model_curves,
    model_oer,
    oer_ratios,
    pressures,
    reconstruction_lets,
    with_scaled_z_domain,
)
from pyradbio.core.radiation import Ion
from pyradbio.models.mkm.osmk import OSMK2021, OSMK2023
from pyradbio.models.mkm.spectrum_models import microdosimetry_cache_info
from pyradbio.track.stopping_power import (
    SimpleBetheStoppingPower,
    default_stopping_power,
    energy_for_let_MeV_u,
)

CELL_MODELS = {
    "HSG": OSMK2021.hsg_inaniwa2021,
    "V79": OSMK2021.v79_inaniwa2021,
    "CHO": OSMK2021.cho_inaniwa2021,
}
ANOXIC_CELLS = ["HSG", "V79", "CHO"]


def _our_oer(model: OSMK2021, curve: OERCurve, lets: np.ndarray) -> np.ndarray:
    """Our OER at ``lets``, inverting LET on the default stopping power; an
    unreachable LET is an error here, never a silent NaN."""
    return model_oer(model, curve, lets, default_stopping_power(), unreachable="raise")


def _our_ratios(model: OSMK2021, cell: str, n: int = 18) -> np.ndarray:
    """Our OER / the paper's, pooled over every species and pO2 of one figure."""
    return oer_ratios(model, cell, default_stopping_power(), n, unreachable="raise")


def _sample_lets(curve: OERCurve, n: int = 18) -> np.ndarray:
    return reconstruction_lets(curve, default_stopping_power(), n)


class TestDigitizationIntegrity:
    def test_all_thirty_series_load(self) -> None:
        curves = load_oer_curves()
        expected = sum(2 * len(sp) for fig in CONTENTS.values() for sp in fig.values())
        assert len(curves) == expected == 30  # 6 + 6 + 18
        assert {c.cell_line for c in curves} == {"HSG", "V79", "CHO"}
        assert {c.species for c in curves} == set(SPECIES)
        assert {c.series for c in curves} == {"exp", "osmk"}
        assert all(c.doi == INANIWA2021_DOI for c in curves)

    def test_only_fig7_varies_the_oxygen_pressure(self) -> None:
        """Figs. 5-6 are anoxic only; fig. 7 is the one with a pO2 axis."""
        assert pressures("HSG") == pressures("V79") == (0.0,)
        assert pressures("CHO") == (0.0, 1.14, 3.8, 15.2)

    def test_only_fig7_carries_error_bars(self) -> None:
        """Fig. 7 draws 1-sigma bars; figs. 5-6 draw bare markers."""
        for curve in load_oer_curves():
            has_bars = bool(curve.oer_upper)
            assert has_bars == (curve.figure == "fig7_cho" and curve.series == "exp")
            if has_bars:
                assert np.all(curve.sigma > 0.0)
                assert np.all(curve.sigma < 0.4)  # 0.05-0.23 measured
            else:
                assert np.all(curve.sigma == 0.0)

    def test_series_are_sorted_and_positive(self) -> None:
        for curve in load_oer_curves():
            # ties are allowed: two measurements can share a quoted LET
            assert np.all(np.diff(curve.let) >= 0.0)
            assert np.all(curve.values >= 1.0 - 0.05)  # an OER below 1 is unphysical

    @pytest.mark.parametrize("cell", ["HSG", "V79"])
    def test_measured_points_span_the_fitted_LET_range(self, cell: str) -> None:
        """Furusawa 2000 covers 18.5-654 keV/um; the union of the three ions should.

        This doubles as the **axis-calibration regression**. Neither panel labels
        its left frame edge -- Origin draws only 100 and 1000 -- so reading that
        edge as 10 rather than its true 15.0 is the natural mistake, and it
        compresses LET by up to a factor 1.5 at the left while leaving the right
        edge exact, which no eyeball check on the plot will catch. Landing on the
        quoted endpoints is an independent check that it was not made, since the
        fitted range is stated in the text rather than read off the figure.
        """
        pts = measured_points(cell)
        fitted = FITTED_LET_RANGE_KEV_UM[cell]
        lo = min(c.let[0] for c in pts.values())
        assert lo == pytest.approx(fitted[0], abs=0.6)  # 18.53 / 18.59
        hi = max(c.let[-1] for c in pts.values())
        if cell == "HSG":
            assert hi == pytest.approx(fitted[1], abs=6.0)  # 654.8
        else:  # V79's neon data stops short of the fitted top, at 529 keV/um
            assert hi > 250.0

    def test_fig7_measured_LETs_match_the_values_the_paper_states_in_words(self) -> None:
        """The fig.-7 axis check, and the strongest one available anywhere here.

        Fig. 4's six survival panels are *labelled* with their LET and pO2, so
        those numbers are prose, not pixels. Fig. 7's measured points must land
        on them. Unlike figs. 5-6, fig. 7's frame really does start at 10 keV/um
        (tick fit: 287.9 px/decade over 576 px), so the panels are not alike and
        this is what catches assuming they are.
        """
        quoted = {  # (species, pO2 mmHg) -> LETs quoted in fig. 4's panel labels
            ("C", 0.0): [51.1],
            ("C", 1.14): [29.8, 68.0, 124.5],
            ("C", 3.8): [124.5],
            ("Si", 15.2): [307.6],
        }
        for (species, po2), lets in quoted.items():
            got = get_oer_curve("CHO", species, "exp", po2).let
            for want in lets:
                nearest = float(got[np.argmin(np.abs(np.log(got / want)))])
                assert nearest == pytest.approx(want, rel=0.005)

    @pytest.mark.parametrize("cell", ["HSG", "V79"])
    def test_the_OER_falls_from_about_three_towards_unity(self, cell: str) -> None:
        """The figure's headline claim (paper sec 4.2), read straight off the trace."""
        for curve in model_curves(cell).values():
            assert 2.7 < curve.values.max() < 3.4
            if curve.let[-1] > 500.0:  # only the curves that reach the high-LET end
                assert curve.values[-1] < 1.1

    @pytest.mark.parametrize("cell", ANOXIC_CELLS)
    def test_model_curves_are_monotone_decreasing_past_their_peak(self, cell: str) -> None:
        for po2 in pressures(cell):
            for curve in model_curves(cell, po2).values():
                peak = int(np.argmax(curve.values))
                tail = curve.values[peak:]
                # a hand trace carries +-1 px of quantization (~0.008 OER units)
                assert np.all(np.diff(tail) < 0.02)

    def test_fig7_OER_falls_as_the_oxygen_pressure_rises(self) -> None:
        """Fig. 7's headline: more oxygen, less protection.

        Carbon is the only species drawn at all four pressures, so it is the one
        series that shows the whole pO2 axis at a fixed LET.
        """
        let = np.array([60.0])
        values = [
            float(get_oer_curve("CHO", "C", "osmk", p).interp(let)[0]) for p in pressures("CHO")
        ]
        assert values == sorted(values, reverse=True)
        assert values[0] > 2.2  # anoxic
        assert values[-1] < 1.2  # 15.2 mmHg is nearly aerobic already

    @pytest.mark.parametrize("cell", ["HSG", "V79"])
    def test_the_paper_curve_tracks_its_own_measured_points(self, cell: str) -> None:
        """A self-consistency check on the digitization, independent of our model.

        The paper fitted these curves to these points, so the residual scatter is
        experimental, not systematic -- the *median* ratio must sit near 1.

        HSG helium is the one that comes close to the bound, at 0.927. That is
        the figure, not the digitization: its six black squares sit visibly below
        the black He curve at low LET (2.88 and 2.97 against a curve plateauing
        at 3.25), because the paper fitted all three ions jointly.
        """
        for species, pts in measured_points(cell).items():
            curve = model_curves(cell)[species]
            inside = curve.covers(pts.let)
            ratio = pts.values[inside] / curve.interp(pts.let[inside])
            assert np.median(ratio) == pytest.approx(1.0, abs=0.08)

    @pytest.mark.parametrize("cell", ["HSG", "V79"])
    def test_helium_model_curve_is_truncated_where_it_is_overdrawn(self, cell: str) -> None:
        """Documented limitation: He is drawn first, so C and Ne paint over it."""
        helium = get_oer_curve(cell, "He", "osmk")
        assert helium.let[-1] < 300.0  # ~220, against 770 (C) and 986 (Ne)
        assert not helium.covers(np.array([400.0]))[0]
        assert get_oer_curve(cell, "Ne", "osmk").covers(np.array([400.0]))[0]

    def test_reconstruction_range_stops_at_the_ion_Bragg_peak_LET(self) -> None:
        """Helium cannot reach the LETs its drawn curve spans -- the trap the
        ``reconstruction_lets`` helper exists to close."""
        sp = SimpleBetheStoppingPower()
        curve = get_oer_curve("HSG", "He", "osmk")
        helium = reconstruction_lets(curve, sp)
        carbon = reconstruction_lets(get_oer_curve("HSG", "C", "osmk"), sp)
        assert helium[-1] < 180.0  # He tops out near its Bragg peak, ~178 keV/um
        assert curve.let[-1] > helium[-1]  # yet its curve is drawn past that (220.6)
        assert carbon[-1] > 600.0  # carbon reaches the full fitted range

    def test_reconstruction_range_also_stops_at_the_minimum_attainable_LET(self) -> None:
        """The symmetric half: no neon beam is *below* ~22 keV/um at any energy.

        Both digitized neon curves happen to start just above that, so the floor
        is exercised here on a curve drawn down to the bottom of the fitted range
        -- which is what the figure would give if neon's trace were extended left.
        """
        sp = SimpleBetheStoppingPower()
        real = get_oer_curve("V79", "Ne", "osmk")
        drawn_lower = OERCurve(
            figure=real.figure,
            cell_line=real.cell_line,
            species=real.species,
            series=real.series,
            let_keV_um=(15.0, *real.let_keV_um),
            oer=(real.oer[0], *real.oer),
            source_page=real.source_page,
        )
        neon = reconstruction_lets(drawn_lower, sp)
        assert neon[0] > FITTED_LET_RANGE_KEV_UM["V79"][0]  # 18.5 unreachable for neon
        assert neon[0] > drawn_lower.let[0]  # the attainable floor binds, not the trace
        assert neon[0] == pytest.approx(22.4, abs=0.5)

    def test_unknown_series_raises(self) -> None:
        with pytest.raises(KeyError, match="no OER curve"):
            get_oer_curve("HSG", "Si", "osmk")  # fig. 5 carries no silicon
        with pytest.raises(KeyError, match="no OER curve"):
            get_oer_curve("CHO", "N", "osmk", 3.8)  # nitrogen is anoxic-only in fig. 7


class TestReconstructionHelpers:
    """``model_oer`` / ``oer_ratios`` / ``with_scaled_z_domain`` -- the shared machinery
    of ``validation/osmk_inaniwa2021.py`` and the goldens below."""

    CARBON = Ion(6, 12, 30.0)

    def test_scaling_moves_only_the_uncorrected_domain_z(self) -> None:
        base = OSMK2021.hsg_inaniwa2021()
        z = base.specific_energies_at(self.CARBON, 0.0)
        zs = with_scaled_z_domain(base, 1.07).specific_energies_at(self.CARBON, 0.0)
        assert zs.z_1D_Gy == z.z_1D_Gy * 1.07
        assert zs.z_star_1D_Gy == z.z_star_1D_Gy
        assert zs.z_nucleus_Gy == z.z_nucleus_Gy

    def test_unit_factor_reproduces_the_model_exactly(self) -> None:
        base = OSMK2021.v79_inaniwa2021()
        same = with_scaled_z_domain(base, 1.0)
        for po2 in (0.0, 3.8):
            assert same.oer_at_survival(self.CARBON, po2) == base.oer_at_survival(self.CARBON, po2)

    def test_copies_every_field_and_scaled_models_are_independent(self) -> None:
        """A class-level ``factor`` would move every instance at once; this does not."""
        base = OSMK2021.cho_inaniwa2021()
        a = with_scaled_z_domain(base, 1.25)
        b = with_scaled_z_domain(base, 0.95)
        assert isinstance(a, OSMK2021)
        for f in fields(OSMK2021):
            assert getattr(a, f.name) == getattr(base, f.name)
        z = base.specific_energies_at(self.CARBON, 0.0).z_1D_Gy
        assert a.specific_energies_at(self.CARBON, 0.0).z_1D_Gy == z * 1.25
        assert b.specific_energies_at(self.CARBON, 0.0).z_1D_Gy == z * 0.95

    def test_scaling_composes(self) -> None:
        base = OSMK2021.hsg_inaniwa2021()
        twice = with_scaled_z_domain(with_scaled_z_domain(base, 1.1), 1.2)
        z = base.specific_energies_at(self.CARBON, 0.0).z_1D_Gy
        assert twice.specific_energies_at(self.CARBON, 0.0).z_1D_Gy == z * (1.2 * 1.1)

    def test_the_2023_formulation_is_refused(self) -> None:
        with pytest.raises(TypeError, match="OSMK2021"):
            with_scaled_z_domain(OSMK2023.hsg_inaniwa2023(), 1.07)  # type: ignore[arg-type]

    def test_model_oer_is_the_hand_inversion(self) -> None:
        sp = SimpleBetheStoppingPower()
        model = OSMK2021.hsg_inaniwa2021()
        curve = get_oer_curve("CHO", "C", "osmk", 3.8)  # a non-zero hypoxic arm
        energy = energy_for_let_MeV_u(sp, 6, 12, 100.0)
        want = model.oer_at_survival(Ion(6, 12, energy), pO2_mmHg=3.8)
        assert model_oer(model, curve, [100.0], sp)[0] == want

    def test_an_unreachable_LET_is_nan_or_raises(self) -> None:
        """Helium tops out near 178 keV/um (SimpleBethe), so 300 keV/um has no beam."""
        sp = SimpleBetheStoppingPower()
        model = OSMK2021.hsg_inaniwa2021()
        helium = get_oer_curve("HSG", "He", "osmk")
        got = model_oer(model, helium, [300.0], sp)
        assert np.isnan(got[0])
        with pytest.raises(ValueError):
            model_oer(model, helium, [300.0], sp, unreachable="raise")

    def test_oer_ratios_pools_every_series_in_order(self) -> None:
        """CHO, because only fig. 7 has more than one pressure: pressure-major order."""
        sp = SimpleBetheStoppingPower()
        model = OSMK2021.cho_inaniwa2021()
        want: list[float] = []
        for po2 in pressures("CHO"):
            for curve in model_curves("CHO", po2).values():
                lets = reconstruction_lets(curve, sp, 2)
                want += list(model_oer(model, curve, lets, sp) / curve.interp(lets))
        got = oer_ratios(model, "CHO", sp, 2)
        assert len(got) == 18  # (4 + 1 + 2 + 2) species-pressure combinations x 2
        assert got.tolist() == want

    def test_oer_ratios_drops_nan_only_in_nan_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under ``"raise"`` a NaN from the model side must reach the golden's
        median/RMS (and fail it) rather than be pooled away."""
        import pyradbio.benchmark.inaniwa2021 as mod

        def nan_first(*args: object, **kwargs: object) -> np.ndarray:
            ours = np.ones(len(args[2]))  # type: ignore[arg-type]
            ours[0] = np.nan
            return ours

        monkeypatch.setattr(mod, "model_oer", nan_first)
        sp = SimpleBetheStoppingPower()
        model = OSMK2021.hsg_inaniwa2021()
        kept = oer_ratios(model, "HSG", sp, 3, unreachable="raise")
        assert len(kept) == 3 * 3 and np.isnan(kept).sum() == 3  # He/C/Ne, one NaN each
        dropped = oer_ratios(model, "HSG", sp, 3)
        assert len(dropped) == 3 * 2 and not np.isnan(dropped).any()

    def test_scaled_copy_is_served_from_the_base_models_cache_entry(self) -> None:
        """What keeps the panel-5 scan of the validation script cheap: a scaled copy
        shares the microdosimetry-cache entry of its base (one hit, no new miss)."""
        base = OSMK2021.v79_inaniwa2021()
        base.specific_energies(self.CARBON)  # the entry now exists, whatever came before
        before = microdosimetry_cache_info()
        with_scaled_z_domain(base, 1.25).specific_energies_at(self.CARBON, 0.0)
        after = microdosimetry_cache_info()
        assert after.hits == before.hits + 1
        assert after.misses == before.misses


@pytest.mark.golden
class TestReconstruction:
    """OSMK 2021 vs the paper's own curves, from the ion, its LET and the pO2.

    Ground truth is figs. 5-7, which predate the 2023 revision, so only
    :class:`OSMK2021` is scored here.
    """

    def test_cho_reconstructs_the_paper_curves_at_every_oxygen_pressure(self) -> None:
        """Fig. 7, and the best reconstruction in this benchmark by an order of
        magnitude: **0.4% RMS with the published parameters untouched**.

        Four ion species (C, N, O, Si) and four oxygen pressures, so it exercises
        the whole chain far more broadly than figs. 5-6 do -- and it does so with
        no adjustment of any kind, which is what makes the V79 gap below a
        statement about V79 rather than about our implementation.
        """
        model = OSMK2021.cho_inaniwa2021()
        r = _our_ratios(model, "CHO", 8)
        assert len(r) == 72  # (4 + 1 + 2 + 2) species-pressure combinations x 8
        assert np.median(r) == pytest.approx(1.001, abs=0.006)
        assert np.sqrt(np.mean(np.log(r) ** 2)) < 0.010  # 0.42% measured
        assert np.max(np.abs(r - 1.0)) < 0.025  # 1.68% measured

    def test_cho_validates_the_assumed_K_at_intermediate_oxygen_pressures(self) -> None:
        """``K = 3 mmHg`` is *assumed* for every cell line and every LET, never
        fitted -- a photon measurement reused at high LET (2021 sec 3.2).

        Figs. 5-6 use only 0 and 160 mmHg, the two ends of the Alper &
        Howard-Flanders curve, so they constrain ``Rmax`` but say nothing about
        its shape in between. Fig. 7's 1.14 / 3.8 / 15.2 mmHg arms are the only
        published data that test ``K`` at all, and they reproduce to <0.4%.
        """
        model = OSMK2021.cho_inaniwa2021()
        for po2 in (1.14, 3.8, 15.2):
            ratios = []
            for curve in model_curves("CHO", po2).values():
                lets = _sample_lets(curve, 8)
                ratios += list(_our_oer(model, curve, lets) / curve.interp(lets))
            r = np.array(ratios)
            assert np.sqrt(np.mean(np.log(r) ** 2)) < 0.008  # 0.16-0.36% measured

    def test_hsg_reconstructs_the_paper_curves(self) -> None:
        model = OSMK2021.hsg_inaniwa2021()
        r = _our_ratios(model, "HSG")
        assert np.median(r) == pytest.approx(1.033, abs=0.015)
        assert np.sqrt(np.mean(np.log(r) ** 2)) < 0.040  # 3.1% measured
        assert np.max(np.abs(r - 1.0)) < 0.065  # 5.2% measured

    def test_hsg_residual_is_an_offset_not_a_shape_error(self) -> None:
        """All three species biased the same way, by about the same amount.

        This is the diagnostic that points at a scale error in ``z_d,D`` rather
        than at eq. (14) itself: the residual is a narrow band about +3%, not a
        LET-dependent excursion. It does drift to roughly zero at the very top of
        the range, where the OER has saturated at unity and there is nothing left
        to get wrong -- so the band is checked, not strict positivity.
        """
        model = OSMK2021.hsg_inaniwa2021()
        spreads = []
        for curve in model_curves("HSG").values():
            lets = _sample_lets(curve)
            r = _our_oer(model, curve, lets) / curve.interp(lets)
            assert np.all(r > 0.97)
            assert np.all(r < 1.06)
            spreads.append(r.max() - r.min())
        assert max(spreads) < 0.08  # 0.048 measured
        # ... and the three species agree with each other on the offset
        medians = [
            float(np.median(_our_oer(model, c, _sample_lets(c)) / c.interp(_sample_lets(c))))
            for c in model_curves("HSG").values()
        ]
        assert max(medians) - min(medians) < 0.02

    def test_v79_carries_a_known_mid_LET_excess(self) -> None:
        """**Open finding.** V79 runs ~7% high overall and up to ~18% mid-LET.

        Ruled out: a table-1 transcription error (re-read from the rendered PDF
        page); the eq.-(24) expansion ceiling (the anoxic ``D_10`` reaches at most
        45% of it); the digitization (re-traced by hand, agreeing with an
        independent automated trace); and -- see
        :meth:`test_the_correction_v79_needs_is_not_a_domain_radius_effect` --
        anything geometric in our own microdosimetry. The residual is a single
        constant: eq. (14) reads ``z_d,D`` only through ``(z_d,D/z_R)^g``, and V79
        needs that ratio moved 25% where HSG needs 6% and CHO none. Pinned so it
        cannot drift silently; see ``docs/references/OSMK_Inaniwa2021_2023.md``.
        """
        model = OSMK2021.v79_inaniwa2021()
        r = _our_ratios(model, "V79")
        assert np.median(r) == pytest.approx(1.073, abs=0.020)
        assert 0.07 < np.sqrt(np.mean(np.log(r) ** 2)) < 0.115  # 9.3% measured

    def test_v79_excess_is_localised_mid_LET(self) -> None:
        """The shape that says it is the eq.-(14) sigmoid *position*, not height."""
        model = OSMK2021.v79_inaniwa2021()
        curve = model_curves("V79")["C"]
        low = _our_oer(model, curve, np.array([20.0])) / curve.interp(np.array([20.0]))
        mid = _our_oer(model, curve, np.array([90.0])) / curve.interp(np.array([90.0]))
        high = _our_oer(model, curve, np.array([400.0])) / curve.interp(np.array([400.0]))
        assert abs(low[0] - 1.0) < 0.05
        assert mid[0] - 1.0 > 0.10
        assert abs(high[0] - 1.0) < 0.08

    def test_the_correction_v79_needs_is_not_a_domain_radius_effect(self) -> None:
        """Closes the leading explanation for the V79 excess: it is not our ``z_d,D``.

        Eq. (14) reads ``z_d,D`` only as ``(z_d,D/z_R)^g``, so "our ``z_d,D`` is
        x% low" and "the published ``z_R`` is x% high" are the same statement from
        inside these figures. The only thing that could make the required
        correction differ *between cell lines* while our code stays fixed is the
        domain radius, since ``z_0`` does not enter the unsaturated ``z_d,D``.

        That hypothesis predicts a monotone trend in ``r_d``. What the three fits
        actually need is +7% at 0.23 um (HSG), +25% at 0.29 (V79) and nothing at
        0.35 (CHO) -- not monotone, and worst in the middle. This test pins the
        decisive half: applying V79's correction to CHO, the *largest* domain,
        makes CHO markedly worse rather than better.
        """
        cho = OSMK2021.cho_inaniwa2021()
        assert cho.domain_radius_um > OSMK2021.v79_inaniwa2021().domain_radius_um
        assert cho.domain_radius_um > OSMK2021.hsg_inaniwa2021().domain_radius_um

        def rms(factor: float) -> float:
            r = _our_ratios(with_scaled_z_domain(cho, factor), "CHO", 6)
            return float(np.sqrt(np.mean(np.log(r) ** 2)))

        assert rms(1.00) < 0.010  # published parameters: 0.4%
        assert rms(1.07) > 3 * rms(1.00)  # HSG's correction already hurts CHO
        assert rms(1.25) > 8 * rms(1.00)  # V79's is far worse

    @pytest.mark.parametrize("cell", ["HSG", "V79"])
    def test_we_reproduce_the_particle_type_ordering_at_equal_LET(self, cell: str) -> None:
        """Both the figure and our model put He below C below Ne at equal LET.

        This is the feature the paper singles out as beyond any dose-averaged-LET
        OER model (sec 4.2), so reproducing the *ordering* matters more than the
        absolute level.
        """
        model = CELL_MODELS[cell]()
        let = np.array([100.0])
        paper = {s: float(c.interp(let)[0]) for s, c in model_curves(cell).items()}
        ours = {s: float(_our_oer(model, c, let)[0]) for s, c in model_curves(cell).items()}
        assert paper["He"] < paper["C"] < paper["Ne"]
        assert ours["He"] < ours["C"] < ours["Ne"]
