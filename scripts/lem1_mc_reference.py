"""Offline stochastic-LEM Monte-Carlo references for the LEM I reconstructions.

Writes the three tracked checkpoints the LEM I validation scripts overlay, one
subcommand each:

``alpha-beta``: ``validation/data/mc_results.csv``, read by ``validation/scholz1997_lem1.py``
    (Figs. 1 and 3).
    The full spatially explicit LEM (``LEM1.scholz1997_montecarlo_survival``, the
    Scholz et al. 1997 "exact" path, Kellerer weighted sampling) at proton and
    carbon points, fitted to ``-ln S/D = alpha + beta D`` over 0.5-3 Gy (eq. 8),
    with a delete-one jackknife over 8 independent streams
    (``pyradbio.benchmark.scholz1997.montecarlo_lq_jackknife``). Every row records
    ``alpha_rel_dev`` against the exact deterministic ``LEM1.initial_slope``: the two
    come from the same fit to the same survivals, so a row whose alpha has drifted
    has not earned belief in its beta. The MC is too slow to run inline (the
    34-point catalogue took about 1 h 50 m with numba);
    ``--high-energy`` adds the n_bar > 3900 set, hours for its protons. Use the
    numba engine (``fast`` extra): it compiles the whole realization but is not
    bit-reproducible, which is why the budget is split across independent streams.
``z-slope``: ``validation/data/sk1997_z_slope.csv``, read by ``validation/scholz1997_lem1.py``
    (Fig. 4).
    The survival conditioned on nuclear specific energy ``z``
    (``LEM1.scholz1997_specific_energy_survival``) at the p/C energies Fig. 4 draws.
    Cost scales with the traversals needed to reach ``z_max`` over the whole
    ``b_max`` disc (23 for carbon at 1 MeV/u, 2.3e8 for protons at 200); the near/far
    split keeps it tractable, and the eight points take about 7 min with numba.
    Keeps the ``LEM1`` default stopping power.
``sk1996-final-slope``: ``validation/data/sk1996_mc_observables.csv``, read by
    ``validation/scholz_kraft1996_lem1.py`` (Figs. 2-4).
    Scholz & Kraft 1996's ``sigma_inact`` is the final slope at 1% survival,
    ``sigma_1% = 0.16 LET d[-ln S]/dD``, not the low-dose single-track
    cross-section; it and Fig. 4's iso-survival RBE at 80/20/1% are read exactly off
    one traversal-conditioned curve per point
    (``pyradbio.benchmark.scholz_kraft1996.estimate_final_slope_point``). Batches
    are pooled until every plotted observable reaches its target precision.

Every subcommand resumes: existing rows are kept unless their point is recomputed,
and the file is rewritten sorted after every point. The file's provenance header
(``# key: value`` lines) carries a numeric fingerprint -- 1-2 points recomputed at
a tiny sample size with ``engine="numpy"`` and fixed seeds -- which
``tests/benchmark/test_lem1_mc_fingerprints.py`` recomputes in CI. A run refuses
to merge new rows into a file whose fingerprint (at the same stopping power) no
longer matches the code; ``--fresh`` then discards EVERY existing point of the file,
not only the ``--only`` selection, so it means recomputing the whole catalogue. A
run at another stopping power than the recorded fingerprint is not refused: its
rows are a different physical point, handled per row as before (``alpha-beta``
drops the other-LET rows, ``sk1996-final-slope`` recomputes them when selected).
The header's ``data_rows`` and ``stopping_power`` lines are rewritten on every save,
so a partial rerun is recorded as a mixed file rather than inheriting the old text.

Usage::

    python scripts/lem1_mc_reference.py --dry-run                 # plan of all three
    python scripts/lem1_mc_reference.py alpha-beta --only carbon:20,carbon:25
    python scripts/lem1_mc_reference.py alpha-beta --engine numpy --high-energy
    python scripts/lem1_mc_reference.py z-slope --only carbon:8,proton:1
    python scripts/lem1_mc_reference.py sk1996-final-slope --only carbon:3,argon:8
    python scripts/lem1_mc_reference.py alpha-beta --n-total-scale 0.001 --out /tmp/x

References: Scholz et al. 1997, doi:10.1007/s004110050055; Scholz & Kraft 1996,
doi:10.1016/0273-1177(95)00784-C.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pyradbio.benchmark import scholz1997 as s97
from pyradbio.benchmark import scholz_kraft1996 as sk
from pyradbio.benchmark.checkpoint import (
    FINGERPRINT_KEY,
    CsvCheckpoint,
    assert_fingerprint,
    parse_point_spec,
)
from pyradbio.benchmark.stopping import select_stopping_power
from pyradbio.core.radiation import Ion
from pyradbio.models.lem import MC_KELLERER_FIELD_POINTS, resolve_mc_engine
from pyradbio.models.lem.lem1 import LEM1
from pyradbio.track.northcliffe_schilling import ENERGY_RANGE_MEV_U

GENERATOR = "scripts/lem1_mc_reference.py"
OUTPUT_FILES = {
    "alpha-beta": "mc_results.csv",
    "z-slope": "sk1997_z_slope.csv",
    "sk1996-final-slope": "sk1996_mc_observables.csv",
}

# --- alpha-beta: the point catalogue and its cost model -----------------------------
#
# (particle, energy_MeV_u, total_realizations), kept where the MC still matches the
# exact cross-section alpha to <1%. The totals are budgeted for a TARGET PRECISION ON
# BETA (about 1.5-2% for the carbon tail at 20-40 MeV/u; 0.4-1.0% for the protons),
# not for uniform runtime. The per-dose path keeps its numba kernel rather than the
# traversal-conditioned survival: conditioning removes the 6x redundancy of
# resimulating each fit dose, but its prefix kernel is numpy-only and costs 23x more
# per realization than two numba streams over all six doses (measured at carbon
# 6 MeV/u). Here the fit window 0.5-3 Gy is well sampled at every dose, so throughput
# wins; the 1996 final slope is the opposite case.
DEFAULT_POINTS: tuple[tuple[str, float, int], ...] = (
    ("proton", 1.0, 176_000),
    ("proton", 1.5, 160_000),
    ("proton", 2.0, 152_000),
    ("proton", 3.0, 96_000),
    ("proton", 4.0, 64_000),
    ("proton", 5.0, 120_000),
    ("proton", 6.0, 96_000),
    ("proton", 7.0, 64_000),
    ("proton", 8.0, 40_000),
    ("proton", 10.0, 20_000),
    ("proton", 12.0, 9_000),
    ("carbon", 1.0, 400_000),
    ("carbon", 2.0, 400_000),
    ("carbon", 3.0, 400_000),
    ("carbon", 4.0, 400_000),
    ("carbon", 5.0, 400_000),
    ("carbon", 6.0, 304_000),
    ("carbon", 7.0, 304_000),
    ("carbon", 8.0, 304_000),
    ("carbon", 9.0, 304_000),
    ("carbon", 10.0, 240_000),
    ("carbon", 12.0, 240_000),
    ("carbon", 15.0, 168_000),
    ("carbon", 18.0, 120_000),
    ("carbon", 20.0, 720_000),
    ("carbon", 25.0, 544_000),
    ("carbon", 30.0, 256_000),
    ("carbon", 35.0, 128_000),
    ("carbon", 40.0, 96_000),
)
# Run only with --high-energy. They answer one question: does the stochastic beta
# approach the photon beta_x from below (the track field becomes homogeneous), as the
# two-track beta does (1.003 beta_x for protons, 0.92 beta_x for carbon at 20 GeV/u),
# or overshoot it? Reachable only with weighted sampling, which is flat in n_bar per
# sample. Ordered cheapest-first: carbon needs ~36x fewer tracks than protons for the
# same dose and b_cut removes almost all of them (carbon 400: 1 224 near tracks out
# of 9.6e6; proton 200: 29 778 out of 2.3e7), so the two proton points at the end are
# hours and everything before them is minutes.
HIGH_ENERGY_POINTS: tuple[tuple[str, float, int], ...] = (
    ("proton", 20.0, 96_000),
    ("carbon", 70.0, 96_000),
    ("carbon", 100.0, 96_000),
    ("carbon", 200.0, 96_000),
    ("carbon", 400.0, 96_000),
    ("proton", 40.0, 96_000),
    ("proton", 70.0, 96_000),
    ("proton", 100.0, 96_000),
    ("proton", 200.0, 96_000),
)

# Weighted-sampler wall clock on the compiled path. The sample count does not scale
# with n_bar, so the driver is n_field_points * n_near plus a fixed per-call cost.
# Measured (N=96000 x 6 doses unless noted): carbon 20 (n_near 110, N=4000 x 3) 3.0 s;
# carbon 70 (389) 119 s; carbon 100 (510) 141 s; proton 20 (3 940) 1881 s; proton 40
# (9 002) 1495 s; proton 70 (14 010) 1282 s. The throughput is NOT constant: the inner
# loop's d2 <= r_max2 cut-off only discriminates when r_max < gen_radius + R_eff, and
# where it bites (proton 20, carbon 20) the mispredicted branch costs more than the
# arithmetic it skips. The rate below is the worst measured one, so the estimate
# over-predicts the branchless points by up to 5x -- the right direction for a warning.
_KELLERER_SECONDS_PER_CALL = 1.1e-4
_KELLERER_EVALUATIONS_PER_SECOND = 2.8e9


def _near_traversals(model: LEM1, ion: Ion, n_bar: float) -> float:
    """Tracks the weighted sampler generates after the near/far split (cost proxy)."""
    cut = s97.KELLERER_B_CUT_UM if n_bar >= s97.KELLERER_B_CUT_MIN_NBAR else None
    return model.near_traversals(ion, n_bar, cut)


def estimated_seconds(n_bar: float, n_total: int, n_near: float | None = None) -> float:
    """Rough wall clock of one alpha-beta point; used only to warn before a long run."""
    near = n_bar if n_near is None else n_near
    # n_near is quoted at the largest fit dose and scales with dose, so the six-dose
    # sum is near * sum(D) / max(D).
    dose_factor = float(sum(s97.MC_FIT_DOSES_GY) / max(s97.MC_FIT_DOSES_GY))
    evaluations = MC_KELLERER_FIELD_POINTS * near * dose_factor
    return n_total * (
        _KELLERER_SECONDS_PER_CALL * len(s97.MC_FIT_DOSES_GY)
        + evaluations / _KELLERER_EVALUATIONS_PER_SECOND
    )


def _scaled(n_total: int, scale: float) -> int:
    """``n_total`` times ``--n-total-scale``, at least one history per stream."""
    return max(s97.MC_N_STREAMS, round(n_total * scale))


# --- shared plumbing -------------------------------------------------------------------


def _name(stopping_power: object) -> str:
    """The ``.name`` every ``select_stopping_power`` result carries (what a row records)."""
    return str(stopping_power.name)  # type: ignore[attr-defined]


def _resolve_engine(choice: str) -> Literal["numba", "numpy"]:
    try:
        return resolve_mc_engine(choice)  # type: ignore[arg-type]
    except ValueError as exc:
        raise SystemExit("engine='numba' needs the 'fast' extra; try --engine numpy") from exc


def _select(points: Sequence[Any], only: set[tuple[str, float]] | None) -> list[Any]:
    selected = [p for p in points if only is None or (p[0], p[1]) in only]
    if not selected:
        raise SystemExit("no points selected; check --only against the point catalogue")
    return selected


class _Provenance:
    """What this run wrote, so the header can say which rows are whose.

    ``data_rows`` and ``stopping_power`` are rewritten before every save: a run that
    has (re)written every stored point owns the file outright, while a partial rerun
    (``--only``) leaves a *mixed* file whose older rows keep their earlier
    description. ``stopping_power`` is read off the rows where the file has that
    column (``alpha-beta``, ``sk1996-final-slope``); ``z-slope`` has none
    and records this run's text instead.
    """

    def __init__(self, checkpoint: CsvCheckpoint, run: str, stopping: str) -> None:
        self.run = run
        self.stopping = stopping
        self.written: set[tuple[str, float]] = set()
        stored = bool(len(checkpoint))
        header = checkpoint.header
        self._earlier_rows = (
            self._others(str(header.get("data_rows", "provenance not recorded")))
            if stored
            else None
        )
        self._earlier_stopping = (
            self._others(str(header.get("stopping_power", "not recorded"))) if stored else None
        )

    @staticmethod
    def _others(text: str) -> str:
        """The description of the rows that predate the last partial run."""
        return text.split("; the others: ", 1)[1] if text.startswith("mixed: ") else text

    def stamp(self, checkpoint: CsvCheckpoint) -> None:
        own = self._earlier_rows is None or set(checkpoint.keys()) <= self.written
        if own:
            checkpoint.header["data_rows"] = f"written by {self.run}"
        else:
            points = ", ".join(f"{p}:{e:g}" for p, e in sorted(self.written))
            checkpoint.header["data_rows"] = (
                f"mixed: some points regenerated by {self.run} (the last run: {points}); "
                f"the others: {self._earlier_rows}"
            )
        if "stopping_power" in (checkpoint.fieldnames or ()):
            names = sorted(
                {
                    str(row.get("stopping_power", ""))
                    for key in checkpoint.keys()
                    for row in checkpoint.rows(key)
                }
            )
            checkpoint.header["stopping_power"] = (
                names[0]
                if len(names) == 1
                else f"mixed per row (see the stopping_power column): {', '.join(names)}"
            )
        elif own:
            checkpoint.header["stopping_power"] = self.stopping
        else:
            checkpoint.header["stopping_power"] = (
                f"mixed: {self.stopping} for the regenerated points; "
                f"the others: {self._earlier_stopping}"
            )


def _same_let(recorded: Any, active: str | None) -> bool:
    """Whether a recorded fingerprint was computed at this run's stopping power.

    A fingerprint without a ``stopping_power`` entry (``z-slope``) always applies.
    """
    names = {entry.get("stopping_power") for entry in recorded if isinstance(entry, dict)}
    names.discard(None)
    return active is None or not names or names == {active}


def _open_checkpoint(
    path: Path,
    fieldnames: Sequence[str],
    *,
    text_columns: frozenset[str],
    one_row_per_point: bool,
    fresh: bool,
    fingerprint: Callable[[], list[dict[str, Any]]],
    active_stopping: str | None,
    header: dict[str, str],
) -> tuple[CsvCheckpoint, _Provenance]:
    """Open ``path``, refuse a stale fingerprint, and stamp the provenance header.

    The fingerprint is compared only when it was recorded at ``active_stopping``: at
    another LET the stored rows are other physical points, which each subcommand
    already handles per row, and refusing would block runs that are valid.
    ``header["stopping_power"]`` describes this run's LET; `_Provenance`
    writes the file's own line from it and the rows.
    """
    checkpoint = CsvCheckpoint(
        path, fieldnames, text_columns=text_columns, one_row_per_point=one_row_per_point
    )
    if fresh:
        for key in checkpoint.keys():
            checkpoint.discard(key)
        checkpoint.header = {}
    recomputed = fingerprint()
    recorded = checkpoint.header.get(FINGERPRINT_KEY)
    if len(checkpoint) and recorded is not None and _same_let(recorded, active_stopping):
        try:
            assert_fingerprint(recorded, recomputed, generator="lem1_mc_reference")
        except AssertionError as exc:
            raise SystemExit(
                f"{path}: {exc}\nThe code no longer computes what its rows were computed "
                "with, so new rows cannot be merged into them. Rerun with --fresh, which "
                f"discards EVERY existing point of {path.name} (not only the --only "
                "selection): plan to recompute the whole catalogue."
            ) from exc
    header = dict(header)
    provenance = _Provenance(checkpoint, header["generator"], header.pop("stopping_power"))
    checkpoint.header = {**checkpoint.header, **header, FINGERPRINT_KEY: recomputed}
    return checkpoint, provenance


def _save(checkpoint: CsvCheckpoint, provenance: _Provenance, key: tuple[str, float]) -> None:
    """Record ``key`` as written by this run, restamp the header and save."""
    provenance.written.add(key)
    provenance.stamp(checkpoint)
    checkpoint.save()


# --- alpha-beta --------------------------------------------------------------------------


def _alpha_beta(args: argparse.Namespace, out_dir: Path) -> int:
    catalogue = DEFAULT_POINTS + (HIGH_ENERGY_POINTS if args.high_energy else ())
    points = [
        (particle, energy, _scaled(n_total, args.n_total_scale))
        for particle, energy, n_total in _select(catalogue, args.only)
    ]
    stopping = select_stopping_power(args.stopping_power)
    model = s97.cho_lem1(stopping, beta_model="scholz1997_montecarlo")
    budget = 0.0
    for particle, energy, n_total in points:
        z, a = s97.SPECIES[particle]
        ion = Ion(z, a, energy)
        n_bar = model.mean_traversals(ion, float(s97.MC_FIT_DOSES_GY[-1]))
        budget += estimated_seconds(n_bar, n_total, _near_traversals(model, ion, n_bar))
    out_csv = out_dir / OUTPUT_FILES["alpha-beta"]
    if args.dry_run:
        print(
            f"alpha-beta -> {out_csv}: {len(points)} points, LET={_name(stopping)}, "
            f"engine={args.engine}, field points={MC_KELLERER_FIELD_POINTS}, "
            f"estimated {budget / 3600:.1f} h with numba (numpy is slower)"
        )
        for particle, energy, n_total in points:
            print(f"  {particle:7s} {energy:6.1f} MeV/u  N={n_total}")
        return 0

    engine = _resolve_engine(args.engine)
    if engine == "numba":
        # Pay the one-off JIT compile before timing the real points.
        model.scholz1997_montecarlo_survival(
            Ion(6, 12, 6.0), [1.0], n_realizations=500, engine="numba"
        )
    checkpoint, provenance = _open_checkpoint(
        out_csv,
        s97.MC_REFERENCE_FIELDS,
        text_columns=s97.MC_REFERENCE_TEXT_COLUMNS,
        one_row_per_point=True,
        fresh=args.fresh,
        fingerprint=lambda: s97.mc_reference_fingerprint(
            select_stopping_power(args.stopping_power)
        ),
        active_stopping=_name(stopping),
        header={
            "generator": f"{GENERATOR} alpha-beta",
            "stopping_power": _name(stopping),
            "last_run": f"engine={engine} seed={args.seed} n_total_scale={args.n_total_scale:g}",
        },
    )
    _filter_alpha_beta_rows(checkpoint, model)
    print(
        f"engine={engine}  LET={_name(stopping)}  "
        f"existing points={len(checkpoint)}  to compute={len(points)}  "
        f"field points={MC_KELLERER_FIELD_POINTS}  estimated {budget / 3600:.1f} h"
    )
    for i, (particle, energy, n_total) in enumerate(points):
        t0 = time.perf_counter()
        result = s97.montecarlo_lq_jackknife(
            model, particle, energy, n_total, engine, seed=args.seed + 17 * i
        )
        checkpoint.put(result)
        # Checkpoint after every point.
        _save(checkpoint, provenance, (particle, energy))
        dt = time.perf_counter() - t0
        print(
            f"  {particle:7s} {energy:5.1f} MeV/u  "
            f"alpha={result['alpha']:.4f}+/-{result['alpha_err']:.4f}  "
            f"(x-sec {result['alpha_cross_section']:.4f})  "
            f"beta={result['beta']:+.5f}+/-{result['beta_err']:.5f}  "
            f"(two-track {result['two_track_beta']:.5f})  "
            f"N={int(result['n_realizations'])} nfp={int(result['n_field_points'])} "
            f"nbar={result['n_bar']:.0f} dalpha={100 * result['alpha_rel_dev']:+.1f}%  {dt:.0f}s"
        )

    print(f"wrote {out_csv} ({len(checkpoint)} points)")
    return 0


def _filter_alpha_beta_rows(checkpoint: CsvCheckpoint, model: LEM1) -> None:
    """Drop rows of an older error estimator or another LET; backfill derivable columns.

    ``error_estimator_version`` below the current one is a different meaning of
    ``*_err``, and a different stopping power is a different physical point (13% in
    alpha at carbon 1 MeV/u), so both are recomputed, never carried forward. Columns
    added after a row was written are derivable (``n_bar`` and ``n_field_points`` from
    the ion and the sampler, ``alpha_rel_dev`` from the two stored alphas) and are
    backfilled instead of forcing a rerun.
    """
    active = _name(model.stopping_power)
    for key in checkpoint.keys():
        row = checkpoint.rows(key)[0]
        if float(row.get("error_estimator_version") or 0.0) < s97.MC_ERROR_ESTIMATOR_VERSION:
            checkpoint.discard(key)
            continue
        if str(row.get("stopping_power", "")) != active:
            checkpoint.discard(key)
            continue
        backfilled = ("n_bar", "n_field_points", "alpha_rel_dev", "sampling")
        if all(row.get(column) for column in backfilled):
            continue
        parsed: dict[str, Any] = checkpoint.record(key)
        z, a = s97.SPECIES[str(parsed["particle"])]
        ion = Ion(z, a, float(parsed["energy_MeV_u"]))
        parsed.setdefault("n_bar", model.mean_traversals(ion, float(s97.MC_FIT_DOSES_GY[-1])))
        parsed.setdefault("n_field_points", float(MC_KELLERER_FIELD_POINTS))
        parsed.setdefault(
            "alpha_rel_dev", float(parsed["alpha"]) / float(parsed["alpha_cross_section"]) - 1.0
        )
        parsed.setdefault("sampling", "kellerer")
        checkpoint.put(parsed)


# --- z-slope -----------------------------------------------------------------------------


def _z_slope(args: argparse.Namespace, out_dir: Path) -> int:
    points = _select(s97.Z_SLOPE_POINTS, args.only)
    model = s97.cho_lem1(beta_model="scholz1997_montecarlo")
    out_csv = out_dir / OUTPUT_FILES["z-slope"]
    if args.dry_run:
        print(
            f"z-slope -> {out_csv}: {len(points)} points, LET=LEM1 default, "
            f"engine={args.engine}, {s97.Z_SLOPE_N_POISSON_MEANS} means x "
            f"{s97.Z_SLOPE_REALIZATIONS_PER_MEAN} realizations, "
            f"{s97.Z_SLOPE_N_FIELD_POINTS} field points; all 8 points ~7 min with numba"
        )
        for particle, energy in points:
            z, a = s97.SPECIES[particle]
            traversals = s97.z_traversals(model, Ion(z, a, energy), s97.Z_SLOPE_Z_MAX_GY)
            print(f"  {particle:7s} {energy:6.1f} MeV/u  traversals@z_max={traversals:11.0f}")
        return 0

    engine = _resolve_engine(args.engine)
    if engine == "numba":
        # Pay the one-off JIT compile before timing. It must be large enough to fill
        # the bins, or the warm-up raises instead of compiling.
        model.scholz1997_specific_energy_survival(
            Ion(6, 12, 1.0),
            z_max_Gy=s97.Z_SLOPE_Z_MAX_GY,
            realizations_per_mean=300,
            b_cut_um=s97.Z_SLOPE_B_CUT_UM,
            seed=0,
            engine="numba",
        )
    checkpoint, provenance = _open_checkpoint(
        out_csv,
        s97.Z_SLOPE_FIELDS,
        text_columns=s97.Z_SLOPE_TEXT_COLUMNS,
        one_row_per_point=False,
        fresh=args.fresh,
        fingerprint=s97.z_slope_fingerprint,
        active_stopping=None,
        header={
            "generator": f"{GENERATOR} z-slope",
            "stopping_power": "LEM1 default (default_stopping_power())",
            "last_run": f"engine={engine} seed={args.seed}",
        },
    )
    print(f"engine={engine}  existing points={len(checkpoint)}  to compute={len(points)}")
    for i, (particle, energy) in enumerate(points):
        z, a = s97.SPECIES[particle]
        ion = Ion(z, a, energy)
        t0 = time.perf_counter()
        point = s97.z_slope_point(model, particle, energy, seed=args.seed + 13 * i, engine=engine)
        checkpoint.put_group(point.rows)
        # Checkpoint after every point.
        _save(checkpoint, provenance, (particle, energy))
        curve, z_lq = point.curve, point.z_lq
        slope = curve.slope_Gy_inv()
        approx = z_lq.alpha + 2.0 * z_lq.beta * curve.z_Gy
        ratio = slope / approx
        line = np.polyval(np.polyfit(curve.z_Gy, slope, 1), curve.z_Gy)
        straight_pct = 100.0 * float(np.sqrt(np.mean((slope - line) ** 2))) / float(slope.mean())
        _, chi2 = curve.straightness()
        dt = time.perf_counter() - t0
        print(
            f"  {particle:7s} {energy:6.1f} MeV/u  bins={curve.z_Gy.size:2d}  "
            f"traversals@z_max={s97.z_traversals(model, ion, s97.Z_SLOPE_Z_MAX_GY):11.0f}  "
            f"exact/approx {ratio[0]:.3f} -> {ratio[-1]:.3f}  "
            f"straight to {straight_pct:.2f}% (chi2/dof {chi2:.1f})  {dt:.0f}s"
        )

    print(f"wrote {out_csv} ({len(checkpoint)} points)")
    return 0


# --- sk1996-final-slope ------------------------------------------------------------------


def _sk1996(args: argparse.Namespace, out_dir: Path) -> int:
    selected = sorted(args.only if args.only is not None else set(sk.MC_POINTS))
    out_csv = out_dir / OUTPUT_FILES["sk1996-final-slope"]
    if args.dry_run:
        print(
            f"sk1996-final-slope -> {out_csv}: {len(selected)} points, "
            f"LET={args.stopping_power}, estimator=v{sk.OBSERVABLE_ESTIMATOR_VERSION}, "
            f"pilot {args.batches} x {args.realizations_per_batch} histories, ceiling "
            f"{args.realizations}, target {100 * args.target_relative_error:g}%; "
            "cost not modelled (XL: the p/He entrance points have thousands of penumbra "
            "traversals per cell)"
        )
        for particle, energy in selected:
            print(f"  {particle:7s} {energy:5.1f} MeV/u")
        return 0

    stopping = select_stopping_power(args.stopping_power)
    full, bracket = sk.v79_lem1_models(stopping)
    checkpoint, provenance = _open_checkpoint(
        out_csv,
        sk.MC_OBSERVABLE_FIELDS,
        text_columns=sk.MC_OBSERVABLE_TEXT_COLUMNS,
        one_row_per_point=True,
        fresh=args.fresh,
        fingerprint=lambda: sk.mc_observables_fingerprint(
            select_stopping_power(args.stopping_power)
        ),
        active_stopping=_name(stopping),
        header={
            "generator": f"{GENERATOR} sk1996-final-slope",
            "stopping_power": _name(stopping),
            "last_run": f"seed={args.seed} target_relative_error={args.target_relative_error:g}",
        },
    )
    print(
        f"points={len(selected)}  existing={len(checkpoint)}  "
        f"estimator=v{sk.OBSERVABLE_ESTIMATOR_VERSION}  LET={_name(stopping)} "
        f"(1970 table valid to {ENERGY_RANGE_MEV_U[1]:g} MeV/u)"
    )
    for index, (particle, energy) in enumerate(selected):
        key = (particle, energy)
        reuse_reason = (
            sk.checkpoint_reuse_reason(
                checkpoint.record(key),
                target_relative_error=args.target_relative_error,
                max_realizations=args.realizations,
                stopping_power=_name(stopping),
            )
            if not args.force and key in checkpoint
            else None
        )
        if reuse_reason is not None:
            print(f"  {particle:7s} {energy:5.1f} MeV/u  checkpoint {reuse_reason}; skipping")
            continue
        started = time.perf_counter()
        result = sk.estimate_final_slope_point(
            full,
            bracket,
            particle,
            energy,
            n_batches=args.batches,
            realizations_per_batch=args.realizations_per_batch,
            max_realizations=args.realizations,
            target_relative_error=args.target_relative_error,
            seed=args.seed + 101 * index,
            max_mean_traversals=args.max_mean_traversals,
        )
        checkpoint.put(result)
        _save(checkpoint, provenance, key)
        print(
            f"  {particle:7s} {energy:5.1f} MeV/u  "
            f"sigma_1%={result['sigma_1pct_um2']:.3f} +/- "
            f"{result['sigma_1pct_err_um2']:.3f} um2  "
            f"sigma_0={result['sigma_low_dose_um2']:.3f} um2  "
            f"RBE80={result['rbe_80pct']:.3f}  "
            f"rel.err={100.0 * result['max_relative_error']:.1f}%  "
            f"N={int(result['n_realizations'])}"
            f"{' (cap)' if not result['precision_converged'] else ''}  "
            f"{time.perf_counter() - started:.0f}s"
        )
    print(f"wrote {out_csv}")
    return 0


# --- CLI ---------------------------------------------------------------------------------


def _points(spec: str) -> set[tuple[str, float]]:
    return parse_point_spec(spec)


def _sk_points(spec: str) -> set[tuple[str, float]]:
    return parse_point_spec(spec, sk.SPECIES)


def _common(parser: argparse.ArgumentParser, *, top: bool) -> None:
    """``--dry-run`` / ``--out`` before or after the subcommand.

    The subcommand copies use ``SUPPRESS`` defaults so they never overwrite a value
    given before the subcommand.
    """
    default: Any = False if top else argparse.SUPPRESS
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=default,
        help="print the plan and a cost estimate; compute nothing, write nothing",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None if top else argparse.SUPPRESS,
        help="output folder (default: validation/data/ of this checkout)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _common(parser, top=True)
    sub = parser.add_subparsers(dest="command")

    ab = sub.add_parser("alpha-beta", help="Scholz 1997 Figs. 1/3 MC alpha/beta -> mc_results.csv")
    _common(ab, top=False)
    ab.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "numba", "numpy"],
        help="MC execution backend (default: auto -> numba if the fast extra is present)",
    )
    ab.add_argument(
        "--only",
        type=_points,
        default=None,
        help="comma list of particle:energy to (re)compute, e.g. carbon:20,proton:6",
    )
    ab.add_argument(
        "--high-energy",
        action="store_true",
        help="also run HIGH_ENERGY_POINTS; read alpha_rel_dev before believing their beta",
    )
    ab.add_argument(
        "--stopping-power",
        default="ns1970",
        choices=["ns1970", "bethe"],
        help="LET source (default: ns1970, matching the reconstruction); changing it "
        "invalidates every stored row",
    )
    ab.add_argument("--seed", type=int, default=1000, help="base RNG seed (numpy engine)")
    ab.add_argument(
        "--n-total-scale",
        type=float,
        default=1.0,
        help="multiply every point's history budget (testing; at least 8 histories)",
    )
    ab.add_argument("--fresh", action="store_true", help="discard existing rows")

    zs = sub.add_parser("z-slope", help="Scholz 1997 Fig. 4 exact S(z) -> sk1997_z_slope.csv")
    _common(zs, top=False)
    zs.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "numba", "numpy"],
        help="MC backend (default: auto -> numba if the fast extra is present)",
    )
    zs.add_argument(
        "--only", type=_points, default=None, help="comma list of particle:energy, e.g. carbon:8"
    )
    zs.add_argument("--seed", type=int, default=7, help="base RNG seed")
    zs.add_argument("--fresh", action="store_true", help="discard existing rows")

    fs = sub.add_parser(
        "sk1996-final-slope",
        help="Scholz & Kraft 1996 Figs. 2-4 sigma_1%% and RBE -> sk1996_mc_observables.csv",
    )
    _common(fs, top=False)
    fs.add_argument("--only", type=_sk_points, help="particle:energy pairs to compute")
    fs.add_argument(
        "--realizations",
        type=int,
        default=sk.DEFAULT_MAX_REALIZATIONS,
        help="maximum total realizations per point (adaptive ceiling)",
    )
    fs.add_argument("--batches", type=int, default=sk.DEFAULT_BATCHES, help="pilot batch count")
    fs.add_argument(
        "--realizations-per-batch",
        type=int,
        default=sk.DEFAULT_REALIZATIONS_PER_BATCH,
        help="realizations in each jackknife batch",
    )
    fs.add_argument(
        "--target-relative-error",
        type=float,
        default=sk.DEFAULT_TARGET_RELATIVE_ERROR,
        help="target relative standard error for sigma and every RBE",
    )
    fs.add_argument(
        "--force", action="store_true", help="recompute rows that already meet the target"
    )
    fs.add_argument("--seed", type=int, default=1996)
    fs.add_argument(
        "--stopping-power",
        choices=("ns1970", "bethe"),
        default="ns1970",
        help="LET source; ns1970 is the paper's own table with a Bethe tail above 12 MeV/u",
    )
    fs.add_argument(
        "--max-mean-traversals",
        type=float,
        default=sk.DEFAULT_MAX_MEAN_TRAVERSALS,
        help="safety guard (default covers the expensive proton 20 MeV/u endpoint)",
    )
    fs.add_argument("--fresh", action="store_true", help="discard existing rows")
    return parser


_COMMANDS: dict[str, Callable[[argparse.Namespace, Path], int]] = {
    "alpha-beta": _alpha_beta,
    "z-slope": _z_slope,
    "sk1996-final-slope": _sk1996,
}


def _default_out_dir() -> Path:
    """``validation/data/`` of the checkout this script lives in."""
    return Path(__file__).resolve().parents[1] / "validation" / "data"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = args.out if args.out is not None else _default_out_dir()
    if args.command is None:
        if not args.dry_run:
            parser.error("choose a subcommand: alpha-beta, z-slope or sk1996-final-slope")
        # The plan of every subcommand, each with its defaults.
        status = 0
        for command in _COMMANDS:
            sub_args = parser.parse_args([command, "--dry-run"])
            status |= _COMMANDS[command](sub_args, out_dir)
        return status
    return _COMMANDS[args.command](args, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
