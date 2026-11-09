"""Manganaro et al. 2018: the inputs of the pyRadBio-vs-Survival LEM I-III comparison.

[batuff/Survival](https://github.com/batuff/Survival) (Manganaro, Russo, Bourhaleb,
Fausti, Giordanengo, Monaco, Sacchi, Vignati, Cirio & Attili, Phys. Med. Biol. 63
(2018) 08NT01, doi:10.1088/1361-6560/aab697) is the only other open-source LEM I-III
implementation. pyRadBio is compared with it on the **HSG LEM row of the paper's
table 1** (``alpha_x = 0.313 Gy^-1``, ``beta_x = 0.0615 Gy^-2``, ``R_N = 5 um``;
``D_t`` = 30 / 6 / 19 Gy for LEM I / II / III) by three scripts, which all read
their inputs from here so that they cannot drift apart:

* ``scripts/survival_paired_mc.py`` drives a locally built ``survival.exe`` over the
  paired grid [`PAIRED_IONS`][] x [`PAIRED_ENERGIES_MEV_U`][] and writes the
  ``*_MonteCarloPaired_*`` exports named by [`survival_table_filename`][];
* ``scripts/lem_mc_vs_survival.py`` runs pyRadBio's own Monte Carlo
  ([`adaptive_montecarlo_point`][]) on the same cell and writes the hash-keyed
  cache ``validation/data/survival_our_mc.json`` ([`mc_cache_key`][]);
* ``validation/survival_code_lem.py`` draws the comparison and reads that cache.

Survival's data (its SRIM tables, eta table and JSON exports) is GPL-2 output of a
local build and is never redistributed; see ``docs/references/survival_repo_notes.md``
for the build and run recipe. Nothing here reads a dataset: callers pass the
stopping power (``pyradbio.track.srim.srim_stopping_power`` for the comparison).

The Survival-style LQ fit is [`pyradbio.core.lq.fit_lq_weighted`][].
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from pyradbio.benchmark.checkpoint import config_key
from pyradbio.core.cell import CellTarget
from pyradbio.core.lq import fit_lq_weighted
from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion
from pyradbio.models.lem.engine import Engine
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.models.lem.lem2 import LEM2
from pyradbio.models.lem.lem3 import LEM3
from pyradbio.track.scholz_kraft import R_MAX_GAMMA_UM, ScholzKraftRadialDose
from pyradbio.track.stopping_power import StoppingPower

__all__ = [
    "FINGERPRINT_POINTS",
    "HSG_LEM",
    "LEM_MODELS",
    "OUR_MC",
    "PAIRED_ENERGIES_MEV_U",
    "PAIRED_IONS",
    "HsgLemCell",
    "LemVersion",
    "OurMcSettings",
    "adaptive_montecarlo_point",
    "mc_cache_key",
    "mc_cache_payload",
    "mc_fingerprint",
    "montecarlo_lq_repeats",
    "survival_matched_lem",
    "survival_table_filename",
]

LemVersion = Literal["LEMI", "LEMII", "LEMIII"]
"""Survival's names for the three LEM versions (its ``-model`` argument)."""

LEM_MODELS: Mapping[str, type[LEM1] | type[LEM2] | type[LEM3]] = MappingProxyType(
    {"LEMI": LEM1, "LEMII": LEM2, "LEMIII": LEM3}
)
"""pyRadBio's model class for each Survival LEM version."""

PAIRED_IONS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {"H": (1, 1), "He": (2, 4), "C": (6, 12), "O": (8, 16)}
)
"""The paired grid's ions, Survival's ``-ion`` symbol -> ``(Z, A)``."""

PAIRED_ENERGIES_MEV_U: tuple[float, ...] = (3.0, 10.0, 30.0, 100.0, 300.0)
"""The paired grid's energies [MeV/u]: Survival's own MC grid (carbon 3-300, protons
3-100) extended to every ion of [`PAIRED_IONS`][].
"""


@dataclass(frozen=True, slots=True)
class HsgLemCell:
    """The photon reference and nucleus of one Manganaro 2018 table-1 row."""

    alpha_x: float
    beta_x: float
    nucleus_radius_um: float
    D_t: Mapping[str, float]

    def photon(self, model: str) -> PhotonParameters:
        """``(alpha_x, beta_x, D_t)`` for LEM version ``model`` (``"LEMI"`` ...)."""
        return PhotonParameters(alpha_x=self.alpha_x, beta_x=self.beta_x, D_t=self.D_t[model])

    def cell_target(self) -> CellTarget:
        """The nucleus, ``effective_size_factor`` passed explicitly (``1.0``, as-is)."""
        return CellTarget(nucleus_radius_um=self.nucleus_radius_um, effective_size_factor=1.0)


HSG_LEM = HsgLemCell(
    alpha_x=0.313,
    beta_x=0.0615,
    nucleus_radius_um=5.0,
    D_t=MappingProxyType({"LEMI": 30.0, "LEMII": 6.0, "LEMIII": 19.0}),
)
"""Manganaro et al. 2018 table 1, HSG LEM row (doi:10.1088/1361-6560/aab697)."""


def survival_table_filename(model: str, calculus: str) -> str:
    """The export name ``<model>_<calculus>_HSG_Manganaro2018.json``.

    ``calculus`` is Survival's ``rapidLEM_*`` approximation (``"Scholz2006"``,
    ``"Russo2011"``), a Monte-Carlo pass (``"MonteCarlo"``, ``"MonteCarloHiP"``,
    ``"MonteCarloPaired"``) or a patched-build variant (``"Russo2011Patched"``);
    the files live in ``$PYRADBIO_SURVIVAL_DIR`` and are read with
    [`pyradbio.benchmark.reference_tables.load_survival_table`][].
    """
    return f"{model}_{calculus}_HSG_Manganaro2018.json"


def survival_matched_lem(
    model: str,
    stopping_power: StoppingPower,
    *,
    match_r_max: bool = True,
    **kwargs: Any,
) -> LEM1 | LEM2 | LEM3:
    """A pyRadBio LEM on [`HSG_LEM`][], driven by ``stopping_power`` (Survival's LET).

    ``r_max = gamma E^1.7``: Survival hard-codes ``gamma = 0.062`` (Elsasser &
    Scholz 2007) for every LEM version, while pyRadBio's LEM I uses the
    ``gamma = 0.05`` of Scholz & Kraft 1996 eq. 10. Each is right for its own paper,
    so with ``match_r_max`` (default) LEM I is set to Survival's value *here* rather
    than changing the library default (``docs/references/survival_repo_notes.md``
    section 4.2); LEM II/III already default to 0.062. An explicit ``radial_dose``
    in ``kwargs`` wins. Other ``kwargs`` (``beta_model=...``) go to the model.
    """
    if match_r_max and model == "LEMI" and "radial_dose" not in kwargs:
        kwargs["radial_dose"] = replace(
            ScholzKraftRadialDose.lem1(stopping_power=stopping_power),
            r_max_gamma_um=R_MAX_GAMMA_UM,
        )
    return LEM_MODELS[model](
        photon=HSG_LEM.photon(model),
        cell=HSG_LEM.cell_target(),
        stopping_power=stopping_power,
        **kwargs,
    )


@dataclass(frozen=True, slots=True)
class OurMcSettings:
    """pyRadBio's Monte-Carlo settings for the Survival comparison (cache contract).

    Every field that can change a cached number is part of the cache key
    ([`mc_cache_payload`][]); the grids decide *which* points exist.

    Attributes:
        cache_version: Bumped by hand when the estimator changes in a way the key
            cannot see.
        doses_Gy: Survival's own MC dose set.
        n_repeat: Independent seeds per point. 8 rather than 4: sigma is a std over
            repeats, and 3 degrees of freedom once read an unlucky draw as a
            systematic.
        target_rel_alpha, target_abs_beta: Precision asked of each point; beta's is
            absolute because beta legitimately crosses zero at high LET.
        n_pilot, n_min, n_max: Pilot realizations and the clip on the sized count.
        b_cut_um, b_cut_min_nbar: Near/far split radius, applied only from this many
            mean traversals at the top dose (``LEMBase.kellerer_field_integral``).
        low_e_ions, low_e_grid, high_e, high_e_ions, high_e_grid: The energy grid
            per (model, Z): 3 and 10 MeV/u are on Survival's MC grid, 1 and 20 are
            ours alone; the high-energy pass reaches Survival's 30/100/300.
    """

    cache_version: int = 1
    doses_Gy: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    n_repeat: int = 8
    target_rel_alpha: float = 0.003
    target_abs_beta: float = 0.02 * HSG_LEM.beta_x
    n_pilot: int = 400
    n_min: int = 250
    n_max: int = 200_000
    b_cut_um: float = 10.0
    b_cut_min_nbar: float = 2000.0
    low_e_ions: Mapping[str, tuple[int, ...]] = field(
        default_factory=lambda: {m: (1, 2, 6, 8) for m in LEM_MODELS}
    )
    low_e_grid: Mapping[int, tuple[float, ...]] = field(
        default_factory=lambda: {
            1: (1.0, 3.0, 10.0),
            2: (1.0, 3.0, 10.0),
            6: (1.0, 3.0, 10.0, 20.0),
            8: (1.0, 3.0, 10.0, 20.0),
        }
    )
    high_e: bool = True
    high_e_ions: Mapping[str, tuple[int, ...]] = field(
        default_factory=lambda: {m: (1, 2, 6, 8) for m in LEM_MODELS}
    )
    high_e_grid: Mapping[int, tuple[float, ...]] = field(
        default_factory=lambda: {
            1: (30.0, 100.0, 300.0),
            2: (30.0, 100.0, 300.0),
            6: (30.0, 100.0, 300.0),
            8: (30.0, 100.0, 300.0),
        }
    )

    def models(self) -> list[str]:
        """The LEM versions with any point, low- and high-energy passes united."""
        return list(dict.fromkeys([*self.low_e_ions, *(self.high_e_ions if self.high_e else ())]))

    def energies(self, model: str, Z: int) -> tuple[float, ...]:
        """The energies to simulate for one (model, ion), sorted and deduplicated."""
        grid: tuple[float, ...] = ()
        if Z in self.low_e_ions.get(model, ()):
            grid += self.low_e_grid.get(Z, ())
        if self.high_e and Z in self.high_e_ions.get(model, ()):
            grid += self.high_e_grid.get(Z, ())
        return tuple(sorted(set(grid)))


OUR_MC = OurMcSettings()
"""The settings the committed cache was generated with."""


def mc_cache_payload(
    model: str,
    Z: int,
    A: int,
    energy_MeV_u: float,
    lem: LEM1 | LEM2 | LEM3,
    stopping_power: StoppingPower,
    settings: OurMcSettings = OUR_MC,
) -> dict[str, Any]:
    """Everything that can change one cached Monte-Carlo point, and nothing else.

    Any of these moving must invalidate the point, so they are hashed rather than
    trusted to a version bump; the LEM version string alone is not enough (the same
    ``LEMII`` at another ``D_t`` or ``r_max`` is another calculation). ``lem`` is
    [`survival_matched_lem`][] ``(model, stopping_power)``. This is the
    payload of the cache key.
    """
    radial = lem.resolved_radial_dose
    ion = Ion(Z, A, energy_MeV_u)
    return {
        "v": settings.cache_version,
        "model": model,
        "Z": Z,
        "A": A,
        "E": energy_MeV_u,
        "alpha_x": HSG_LEM.alpha_x,
        "beta_x": HSG_LEM.beta_x,
        "R_N": HSG_LEM.nucleus_radius_um,
        "D_t": HSG_LEM.D_t[model],
        "r_max_gamma": radial.r_max_gamma_um,  # type: ignore[attr-defined]
        "r_min_um": radial.r_min_um(ion),
        "sigma_nm": getattr(radial, "sigma_nm", 0.0),
        "doses": list(settings.doses_Gy),
        "repeats": settings.n_repeat,
        "b_cut": settings.b_cut_um,
        "b_cut_min_nbar": settings.b_cut_min_nbar,
        "target_alpha": settings.target_rel_alpha,
        "target_beta": settings.target_abs_beta,
        "n_pilot": settings.n_pilot,
        "n_min": settings.n_min,
        "n_max": settings.n_max,
        "let": stopping_power.let_keV_um(ion),
    }


def mc_cache_key(
    model: str,
    Z: int,
    A: int,
    energy_MeV_u: float,
    lem: LEM1 | LEM2 | LEM3,
    stopping_power: StoppingPower,
    settings: OurMcSettings = OUR_MC,
) -> str:
    """[`config_key`][] of [`mc_cache_payload`][]."""
    return config_key(mc_cache_payload(model, Z, A, energy_MeV_u, lem, stopping_power, settings))


def montecarlo_lq_repeats(
    lem: LEM1 | LEM2 | LEM3,
    ion: Ion,
    doses_Gy: npt.ArrayLike,
    *,
    n_realizations: int,
    n_repeat: int,
    **mc_kw: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``(alphas, betas)`` from ``n_repeat`` independent Monte-Carlo runs (seeds 0..n-1).

    Each run is ``lem.scholz1997_montecarlo_survival(ion, doses, n_realizations=...,
    seed=s, **mc_kw)``; every run's ``-ln S`` is fitted with Survival's own LQ fit
    ([`fit_lq_weighted`][]: no intercept, weighted by the
    inverse spread of ``-ln S`` over the repeats). The spread of the returned
    coefficients is therefore the sigma of a *single* ``n_realizations`` run -- the
    quantity Survival reports from its covariance matrix, so directly comparable.
    ``mc_kw`` takes ``engine=`` and ``b_cut_um=``.
    """
    doses = np.asarray(doses_Gy, dtype=float)
    log_kill = np.array(
        [
            -np.log(
                lem.scholz1997_montecarlo_survival(
                    ion, doses, n_realizations=n_realizations, seed=s, **mc_kw
                )
            )
            for s in range(n_repeat)
        ]
    )
    sigma = log_kill.std(axis=0, ddof=1)
    alphas, betas = np.array([fit_lq_weighted(doses, lk, sigma) for lk in log_kill]).T
    return alphas, betas


def adaptive_montecarlo_point(
    lem: LEM1 | LEM2 | LEM3,
    ion: Ion,
    stopping_power: StoppingPower,
    settings: OurMcSettings = OUR_MC,
    *,
    engine: Engine = "auto",
) -> dict[str, float]:
    """One cache row: pilot, size ``n_realizations`` to the precision targets, run.

    A flat budget was badly misallocated (at 4000 x 4 the relative sigma on alpha
    ran from 0.02% for protons at 100 MeV/u to 1.98% for oxygen at 10): a low-LET
    point puts many traversals through every cell, so the cell-to-cell spread of
    ``-ln S`` is small, while a high-LET point has few. So a pilot of
    ``settings.n_pilot`` measures the spread, the count is scaled by the square of
    the larger miss of the two targets (sigma ~ 1/sqrt(N)) and clipped to
    ``[n_min, n_max]``; if that is no more than the pilot, the pilot stands. Tracks
    beyond ``settings.b_cut_um`` become one background draw where the top dose
    puts at least ``settings.b_cut_min_nbar`` tracks in reach of the nucleus.

    Returns ``{"let", "alpha", "alpha_sigma", "beta", "beta_sigma", "n_real"}``
    (means and single-run sigmas over the repeats). ``n_real`` is the *sized* count
    even when the pilot stands (it is then ``<= n_pilot``, and the numbers are the
    pilot's) -- the bookkeeping the committed cache carries. ``engine`` defaults to
    ``"auto"`` (numba when installed).
    """
    doses = np.asarray(settings.doses_Gy, dtype=float)
    r_eff = lem.cell.effective_nucleus_radius_um()
    a_eff = np.pi * r_eff * r_eff
    b_max = r_eff + lem.resolved_radial_dose.r_max_um(ion)
    n_bar = doses[-1] / (a_eff * lem.mean_traversal_dose_Gy(ion)) * np.pi * b_max * b_max
    kw: dict[str, Any] = {"engine": engine}
    if n_bar >= settings.b_cut_min_nbar:
        kw["b_cut_um"] = settings.b_cut_um

    def run(n_real: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return montecarlo_lq_repeats(
            lem, ion, doses, n_realizations=n_real, n_repeat=settings.n_repeat, **kw
        )

    a_p, b_p = run(settings.n_pilot)
    rel_a = a_p.std(ddof=1) / max(abs(a_p.mean()), 1e-12)
    abs_b = b_p.std(ddof=1)
    grow = max(
        (rel_a / settings.target_rel_alpha) ** 2, (abs_b / settings.target_abs_beta) ** 2, 0.0
    )
    n_real = int(np.clip(settings.n_pilot * grow, settings.n_min, settings.n_max))
    a, b = (a_p, b_p) if n_real <= settings.n_pilot else run(n_real)
    return {
        "let": float(stopping_power.let_keV_um(ion)),
        "alpha": float(a.mean()),
        "alpha_sigma": float(a.std(ddof=1)),
        "beta": float(b.mean()),
        "beta_sigma": float(b.std(ddof=1)),
        "n_real": float(n_real),
    }


FINGERPRINT_POINTS: tuple[Mapping[str, Any], ...] = (
    MappingProxyType(
        {
            "model": "LEMI",
            "Z": 6,
            "A": 12,
            "energy_MeV_u": 10.0,
            "adaptive": MappingProxyType(
                {
                    "n_repeat": 3,
                    "n_pilot": 4,
                    "n_min": 2,
                    "n_max": 8,
                    "b_cut_um": 6.0,
                    "b_cut_min_nbar": 0.0,
                }
            ),
        }
    ),
    MappingProxyType(
        {"model": "LEMII", "Z": 8, "A": 16, "energy_MeV_u": 1.0, "n_realizations": 20}
    ),
    MappingProxyType(
        {"model": "LEMIII", "Z": 8, "A": 16, "energy_MeV_u": 1.0, "n_realizations": 20}
    ),
)
"""The numeric fingerprint of the committed cache: small,
data-free, deterministic points through the generator's own code path (numpy
engine, seeds ``0..n_repeat-1``) on the analytic Bethe stopping power, since
Survival's SRIM tables are not redistributed and the fingerprint must run in CI
(an SRIM change still shows: the LET is part of every cache key). The first point
runs the whole [`adaptive_montecarlo_point`][] (pilot, sizing, re-run) on
shrunken [`OurMcSettings`][] -- ``b_cut_min_nbar = 0`` and ``b_cut_um = 6``
put tracks beyond the cut at carbon 10 MeV/u (``b_max ~ 8 um``), so the near/far
split's background draw is pinned too; the other two run
[`montecarlo_lq_repeats`][] on LEM II and LEM III at the same ion, energy and
seeds -- the shared cluster/eta machinery, once on LEM II's own radial dose and
once on LEM III's velocity-dependent core (``LEM3`` overrides only the radial dose).
"""
_FINGERPRINT_REPEATS = 3


def mc_fingerprint(
    points: Sequence[Mapping[str, Any]] = FINGERPRINT_POINTS,
) -> list[dict[str, Any]]:
    """Recompute the cache fingerprint: one record per point of ``points``.

    Each record holds the point's inputs (``model``, ``Z``, ``A``, ``energy_MeV_u``
    and either ``adaptive`` -- the [`OurMcSettings`][] fields replaced on
    [`OUR_MC`][] -- or ``n_realizations`` + ``n_repeat``), ``engine="numpy"``,
    ``stopping_power="simple_bethe"`` and the results ``alpha``, ``alpha_sigma``,
    ``beta``, ``beta_sigma`` (means and spreads over seeds ``0..n_repeat-1``; an
    adaptive point adds ``let`` and the sized ``n_real``). A plain record without
    ``n_repeat`` gets ``3``. The test compares the stored records with
    [`assert_fingerprint`][pyradbio.benchmark.checkpoint.assert_fingerprint].
    """
    from pyradbio.track.stopping_power import SimpleBetheStoppingPower

    stopping_power = SimpleBetheStoppingPower()
    out: list[dict[str, Any]] = []
    for point in points:
        model, Z, A = str(point["model"]), int(point["Z"]), int(point["A"])
        energy = float(point["energy_MeV_u"])
        lem = survival_matched_lem(model, stopping_power)
        ion = Ion(Z, A, energy)
        record: dict[str, Any] = {"model": model, "Z": Z, "A": A, "energy_MeV_u": energy}
        if "adaptive" in point:
            overrides = dict(point["adaptive"])
            row = adaptive_montecarlo_point(
                lem, ion, stopping_power, replace(OUR_MC, **overrides), engine="numpy"
            )
            record.update(adaptive=overrides, engine="numpy", stopping_power="simple_bethe")
            record.update(row)
        else:
            n_real = int(point["n_realizations"])
            n_repeat = int(point.get("n_repeat", _FINGERPRINT_REPEATS))
            alphas, betas = montecarlo_lq_repeats(
                lem,
                ion,
                OUR_MC.doses_Gy,
                n_realizations=n_real,
                n_repeat=n_repeat,
                engine="numpy",
            )
            record.update(
                n_realizations=n_real,
                n_repeat=n_repeat,
                engine="numpy",
                stopping_power="simple_bethe",
                alpha=float(alphas.mean()),
                alpha_sigma=float(alphas.std(ddof=1)),
                beta=float(betas.mean()),
                beta_sigma=float(betas.std(ddof=1)),
            )
        out.append(record)
    return out
