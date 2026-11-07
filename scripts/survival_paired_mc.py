"""Generate paired Survival reference-MC points, resumably.

Drives a locally built `survival.exe` over the (model x ion x energy) grid that
``validation/survival_code_lem.py`` plots, sizes each point's cell count from a
cheap pilot, and merges the result into the JSON exports
``reference_tables.load_survival_table`` reads. Nothing from Survival is
redistributed -- this writes to the *local* ``PYRADBIO_SURVIVAL_DIR`` (or ``--out``).

Why size at all: Survival's cost and its precision run in *opposite* directions,
exactly as ours do. A flat cell count spends hours over-resolving the cheap end
and still misses target at the dear one.

**The cost model is measured, not assumed.** Charging a constant 24 s of setup
and blaming the rest on the cells overestimates the per-cell cost tenfold at high
energy: oxygen 300 MeV/u looks like 4.2 s/cell against a true ~0.4 s, so the
projection blows the ceiling, under-buys cells and caps the point at 0.90% where
the same ceiling reaches 0.500% in a quarter of the time. `pilot()` therefore runs
TWO points (2 and 20 cells) and solves for setup and slope separately. The split
varies enormously and in ways no scaling rule predicted -- 129 s + 3.4 s/cell for
LEM I carbon 300, but 49 s + 1841 s/cell for LEM II helium 300, and ~0 s + 7598
s/cell for LEM I protons 300.

**Cells buy far less precision at high energy than 1/sqrt(N) suggests.** At 300
MeV/u a single cell is already crossed by a huge number of tracks: LEM I protons
returned 0.22% from TWO cells, and LEM II helium 0.63% from two, while 3 MeV/u
points need 15000-35000 cells for ~1%. So at the top of the grid the cell count
is a cost knob, not a precision knob, and the ladder in `pilot()` descends to 1
cell rather than abandoning a point.

Three of the 60 points are unobtainable this way: LEM III helium and LEM II/III
protons at 300 MeV/u, where a *single* cell exceeds four hours.

`-precision` also accepts a relative-error target directly, and that was tried
first: a 2% target at carbon 300 ran past 1494 s where 20 cells took 189 s. Its
criterion is on the survival rather than on alpha and does not stop early, so
this sizes the cell count itself.

Paths are resolved in ``main()``, never at import: the source checkout
(``PYRADBIO_SURVIVAL_SRC_DIR``, GPL-2, never redistributed) supplies the binary's
``data/`` tables; ``--exe`` / ``$PYRADBIO_SURVIVAL_EXE`` (default
``<src>/../survival.exe``) the binary; ``--work`` / ``$PYRADBIO_SURVIVAL_WORK``
(default ``<src>/../svgen``) the scratch folder holding ``checkpoint.json``.

Usage::

    python scripts/survival_paired_mc.py --dry-run          # the plan, no compute
    python scripts/survival_paired_mc.py                    # fill the grid
    python scripts/survival_paired_mc.py --refine --maxE 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from pyradbio.benchmark.manganaro2018 import (
    HSG_LEM,
    PAIRED_ENERGIES_MEV_U,
    PAIRED_IONS,
    survival_table_filename,
)

# Manganaro et al. 2018 table 1, HSG LEM row (pyradbio.benchmark.manganaro2018).
ALPHA_X, BETA_X, R_N = HSG_LEM.alpha_x, HSG_LEM.beta_x, HSG_LEM.nucleus_radius_um
D_T = dict(HSG_LEM.D_t)
IONS = {ion: A for ion, (_, A) in PAIRED_IONS.items()}  # Survival's -ion symbol -> A
Z_OF = {ion: Z for ion, (Z, _) in PAIRED_IONS.items()}
ENERGIES = PAIRED_ENERGIES_MEV_U
# Cheapest first, so the checkpoint gains rows early and a kill costs little.
_ION_ORDER = {"O": 0, "C": 1, "He": 2, "H": 3}

PILOT_CELLS = 20
PILOT_MIN_CELLS = 2
# Sized for a machine that keeps interrupting long runs. Three console-interrupt
# kills in two hours, each costing the whole in-flight point (survival.exe is a
# black box -- there is no partial result to resume). A 45 min point that banks
# beats a 4 h point that never finishes; `--refine` can buy the precision back
# later, and a banked noisy point is refinable while a missing one is not.
PILOT_SECONDS = 600
LONG_SECONDS = 2700
LAST_RESORT_SECONDS = 4 * 3600  # only for points no short window can ever hold
PROBE_FRACTION = 0.35  # probe budget as a fraction of the ceiling in force
TARGET_REL_ALPHA = 0.005  # matches our own 0.3% closely enough not to dominate
REFINE_FACTOR = 1.2  # a point is "short" only above 1.2 * target, i.e. 0.6%
MIN_CELLS, MAX_CELLS = 40, 120_000
MAX_SECONDS = 5400  # per point; a point that cannot make target in 90 min is logged
PROJECT_SAFETY = 0.75  # size the final run to this fraction of the ceiling
DESCRIPTION = (
    "Generated with batuff/Survival (GPL-2, doi:10.1088/1361-6560/aab697) built from "
    "source; parameters are Manganaro et al. 2018 table 1, HSG LEM row. Cell count "
    "sized per point from a pilot to a target relative sigma on alpha. See "
    "docs/references/survival_repo_notes.md."
)


@dataclass(frozen=True)
class Paths:
    """Where the driver reads the binary and writes its results (resolved in ``main``)."""

    exe: Path
    data: Path  # the source checkout's data/ tables, passed to survival.exe as $DATA
    work: Path  # survival.exe writes its CSV to $PWD, so it needs a scratch dir of its own
    out_dir: Path  # where the merged JSON exports land

    @property
    def checkpoint(self) -> Path:
        return self.work / "checkpoint.json"


def point_key(model: str, ion: str, e_u: float) -> str:
    """The checkpoint key ``model|ion|E`` (``E`` formatted ``:g``)."""
    return f"{model}|{ion}|{e_u:g}"


def plan_points(cp: Mapping[str, Any]) -> list[tuple[str, str, float]]:
    """The ``(model, ion, E)`` points still to run, cheapest first.

    Every grid point (``D_T`` x ``IONS`` x ``ENERGIES``) whose key is not in the
    checkpoint ``cp``, sorted by energy, then oxygen -> carbon -> helium -> protons,
    then model.
    """
    return sorted(
        (
            (m, ion, e)
            for m in D_T
            for ion in IONS
            for e in ENERGIES
            if point_key(m, ion, e) not in cp
        ),
        key=lambda t: (t[2], _ION_ORDER[t[1]], t[0]),
    )


def load_checkpoint(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def save_checkpoint(path: Path, cp: dict) -> None:
    path.write_text(json.dumps(cp, indent=1, sort_keys=True))


def survival_command(paths: Paths, model: str, ion: str, e_u: float, cells: int) -> list[str]:
    """The ``survival.exe`` command line for one Monte-Carlo point."""
    total_mev = IONS[ion] * e_u  # -energies is TOTAL MeV, not MeV/u
    return [
        str(paths.exe),
        "-output",
        "LQ_pars",
        "-model",
        model,
        "-calculusType",
        "MonteCarlo",
        "-precision",
        str(cells),
        "-parallelismType",
        "0",
        "-ion",
        ion,
        "-energies",
        f"{total_mev:g}",
        # NOT optional. Survival's defaults are alpha0=0.312, beta0=0.073,
        # rNucleus=4.611 -- close enough to the Manganaro table-1 HSG row to
        # look right in a log and wrong in a comparison. beta0 is off by 19%.
        "-cellType",
        "HSG",
        "-LEM_alpha0",
        f"{ALPHA_X:g}",
        "-LEM_beta0",
        f"{BETA_X:g}",
        "-LEM_rNucleus",
        f"{R_N:g}",
        "-LEM_Dt",
        f"{D_T[model]:g}",
    ]


def run_point(
    paths: Paths,
    model: str,
    ion: str,
    e_u: float,
    cells: int,
    timeout: float | None = None,
    *,
    max_seconds: float = MAX_SECONDS,
) -> tuple[dict, float]:
    """One Survival MC point. Returns its parsed CSV row and the wall time.

    ``timeout=None`` means ``max_seconds``, the ceiling in force: a refinement
    pass raises it by passing a larger one.
    """
    timeout = max_seconds if timeout is None else timeout
    csv_path = paths.work / f"NewProject_LQparameters_{model}.csv"
    csv_path.unlink(missing_ok=True)
    env = {**os.environ, "DATA": str(paths.data)}
    t0 = time.perf_counter()
    proc = subprocess.run(
        survival_command(paths, model, ion, e_u, cells),
        cwd=paths.work,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    dt = time.perf_counter() - t0
    if proc.returncode != 0 or not csv_path.exists():
        # Keep the whole transcript: rc=-1 (0xFFFFFFFF) is not one of their
        # exit(1) error paths, it is abnormal termination, and it has only ever
        # struck O at 300 MeV/u -- right after the 4-level `new Pixel[]`
        # structure, with a single run already resident at 5.3 GB. stdout's tail
        # alone cannot distinguish an allocation failure from anything else.
        dump = paths.work / f"fail_{model}_{ion}_{e_u:g}_{cells}.txt"
        dump.write_text(
            f"rc={proc.returncode} cells={cells}\n=== stdout ===\n{proc.stdout}"
            f"\n=== stderr ===\n{proc.stderr}"
        )
        raise RuntimeError(
            f"survival.exe failed (rc={proc.returncode}) stderr={proc.stderr[-200:]!r} "
            f"dump={dump.name}"
        )
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError("survival.exe wrote no rows")
    # A non-positive sigma is not a perfect measurement, it is a missing one --
    # and it would sail through every precision check as 0.00%, outranking real
    # results in the refinement comparison. Relevant now that the ladder can drop
    # to a single cell, where an across-cell spread may not be defined.
    if float(rows[-1]["alpha_sigma"]) <= 0.0:
        raise RuntimeError(
            f"survival.exe returned alpha_sigma={rows[-1]['alpha_sigma']} at {cells} cells"
        )
    return rows[-1], dt


def pilot(
    paths: Paths, model: str, ion: str, e_u: float, *, max_seconds: float = MAX_SECONDS
) -> tuple[dict, float, int, float, float, float]:
    """Two-point cost probe: measure the setup and the per-cell slope separately.

    A single pilot cannot tell the two apart, and assuming a constant 24 s setup
    is wrong by an order of magnitude at the top of the grid. Survival builds a
    "Pixel structure" per run -- 3.1M pixels for a 100 MeV/u proton -- before any
    cell is sampled, and it grows with track extent. Oxygen at 300 MeV/u then
    looks like 4.2 s/cell when the true slope is ~0.4 s/cell and ~100 s is setup;
    the projection blows the ceiling early, under-buys cells, and caps the point
    at 0.90% when the same ceiling could have reached target.

    Returns the best row, total probe seconds, its cell count, the fitted
    (fixed, per_cell), and the ceiling to size the final run against.
    """
    budget = max(PILOT_SECONDS, PROBE_FRACTION * max_seconds)
    # 2 cells is essentially a setup-only measurement.
    try:
        row2, t2 = run_point(
            paths, model, ion, e_u, PILOT_MIN_CELLS, timeout=budget, max_seconds=max_seconds
        )
    except subprocess.TimeoutExpired:
        # The 2-cell probe timing out does NOT prove cells are free -- it is also
        # what a huge per-cell cost looks like, and the two differ by an order of
        # magnitude in what the point costs to finish. Measure 20 cells under the
        # long ceiling, then bound the slope by what the timeout tells us:
        # t(2) > budget, so per_cell <= (t20 - budget) / 18. Taking the bound is
        # pessimistic on cell count and safe on cost -- the previous version
        # asserted per_cell = 0, which claimed cells were free and then capped at
        # 20 anyway, buying none of them (LEMII|C|300: 2.6 h for 2.03%).
        print("    2-cell probe timed out: long run, slope bounded", flush=True)
        # Descend the cell count until something fits the long window. Without
        # this ladder a point whose 20-cell run exceeds the window raises and is
        # lost outright -- and with the window now at 45 min that is a live risk
        # for LEM II/III helium and protons at 300 MeV/u.
        # Short window first so the point banks before the next interruption.
        # The last rung uses the 4 h window: when the SETUP alone exceeds 45 min
        # no cell count can help, and dropping the point is worse than risking a
        # long run -- an interrupted attempt is retried by the auto-restart and
        # may land between disruptions, whereas a dropped point never returns.
        # The last rung takes the FEWEST cells, not the most. Spending the 4 h rung
        # on 20 cells after 2 cells had already failed at 45 min would mean the
        # cheapest run available is never tried: LEMII|He|300 burned 6.25 h and
        # returned nothing. When the short rungs have all failed the
        # goal is any answer at all, and 2 cells is the likeliest to land.
        # 20 cells stays FIRST: when it fits, it is the good answer (LEMIII|H|100
        # reached 0.41% that way, which a cheapest-first ladder would have thrown
        # away for a 2-cell 1.3%). The 5-cell rung is dropped -- once 20 has
        # failed it rarely helps and costs another full window -- so a hopeless
        # point burns 1.5 h of short rungs, not 2.25 h, before its one long try.
        # The 1-cell rung is the floor. LEM II helium at 300 MeV/u measured
        # ~1841 s per cell and still returned 0.63% from TWO cells -- at this
        # energy a single cell is crossed by so many tracks that it is already a
        # good average, so the count is a cost knob far more than a precision
        # one. LEMIII|He|300 exceeded 4 h even at 2 cells; 1 cell is the only
        # thing left to try before declaring a point unobtainable here.
        ladder = (
            (PILOT_CELLS, LONG_SECONDS),
            (PILOT_MIN_CELLS, LONG_SECONDS),
            (PILOT_MIN_CELLS, LAST_RESORT_SECONDS),
            (1, LAST_RESORT_SECONDS),
        )
        for c, window in ladder:
            try:
                row, dt = run_point(
                    paths, model, ion, e_u, c, timeout=window, max_seconds=max_seconds
                )
            except subprocess.TimeoutExpired:
                print(f"    {c} cells exceed {window / 60:.0f} min, next rung", flush=True)
                continue
            per_cell = max(dt - budget, 0.0) / max(c - PILOT_MIN_CELLS, 1)
            fixed = max(dt - c * per_cell, 0.0)
            print(
                f"    bounded cost: {fixed:.0f}s setup + <={per_cell:.1f}s/cell at {c} cells",
                flush=True,
            )
            # A point granted the long probe deserves the long ceiling; otherwise
            # the long pilot is sized against the 90 min one.
            return row, dt, c, fixed, per_cell, max(LONG_SECONDS, dt)
        raise subprocess.TimeoutExpired("survival.exe", LAST_RESORT_SECONDS) from None
    # Give the 20-cell probe the LONG window, not the probe budget. Its row is
    # the pilot we keep regardless, so a timeout here throws away real work --
    # and all a timeout teaches is a LOWER bound on the slope, which is the
    # optimistic direction: sizing from it overbuys, overruns, and falls back to
    # the 2-cell row. That shipped 6.4% where the 20-cell row gave 2.03%.
    # Measure the second point instead of inferring it.
    try:
        row20, t20 = run_point(
            paths, model, ion, e_u, PILOT_CELLS, timeout=LONG_SECONDS, max_seconds=max_seconds
        )
    except subprocess.TimeoutExpired:
        # 20 cells cannot be had in 4 h. Ship the 2-cell row and price cells so
        # high that sizing cannot buy any -- the honest end of the road.
        print("    20 cells exceed the long window; shipping the 2-cell row", flush=True)
        return row2, t2 + LONG_SECONDS, PILOT_MIN_CELLS, t2, LONG_SECONDS, max_seconds
    per_cell = max(t20 - t2, 0.0) / (PILOT_CELLS - PILOT_MIN_CELLS)
    fixed = max(t2 - PILOT_MIN_CELLS * per_cell, 0.0)
    print(f"    cost: {fixed:.0f}s setup + {per_cell:.2f}s/cell", flush=True)
    # A point that needed more than the probe budget is an expensive one; size it
    # against the long ceiling rather than the 90 min default.
    ceiling = max_seconds if t20 <= budget else LONG_SECONDS
    return row20, t2 + t20, PILOT_CELLS, fixed, per_cell, ceiling


def sized_point(
    paths: Paths, model: str, ion: str, e_u: float, *, max_seconds: float = MAX_SECONDS
) -> dict:
    """Pilot, then buy the cell count the target needs."""
    row, t_pilot, pilot_cells, fixed, per_cell, ceiling = pilot(
        paths, model, ion, e_u, max_seconds=max_seconds
    )
    a, sa = float(row["alpha"]), float(row["alpha_sigma"])
    rel = sa / max(abs(a), 1e-12)
    grow = (rel / TARGET_REL_ALPHA) ** 2
    cells = int(min(max(pilot_cells * grow, MIN_CELLS), MAX_CELLS))
    projected = fixed + cells * per_cell
    capped = False
    # Size against a fraction of the ceiling, not the ceiling itself. A count
    # estimated at 0.99 * MAX_SECONDS overruns as often as not -- and an overrun
    # costs the whole 90 min and still falls back to the pilot, so the estimate
    # has to fit comfortably or not be bought at all.
    affordable = PROJECT_SAFETY * ceiling
    if projected > affordable:
        # Floor at the pilot, NOT at MIN_CELLS: when even 40 cells cannot fit the
        # ceiling, buying them anyway just burns the ceiling and falls back here.
        cells = max(int((affordable - fixed) / max(per_cell, 1e-9)), pilot_cells)
        capped = True
    if cells <= pilot_cells:
        final, t_final = row, 0.0
        cells = pilot_cells
    else:
        try:
            final, t_final = run_point(
                paths, model, ion, e_u, cells, timeout=ceiling, max_seconds=max_seconds
            )
        except subprocess.TimeoutExpired:
            # The per-cell estimate came from a 20-cell pilot and can be low. Keep
            # the pilot rather than losing the point: a noisy paired point still
            # constrains the comparison, a missing one constrains nothing.
            final, t_final, cells, capped = row, 0.0, pilot_cells, True
    return {
        "alpha": float(final["alpha"]),
        "beta": float(final["beta"]),
        "alpha_sigma": float(final["alpha_sigma"]),
        "beta_sigma": float(final["beta_sigma"]),
        "let_keV_um": float(final["meanLET"]) * 1e3,  # meanLET is MeV/um in the CSV
        "cells": cells,
        "seconds": round(t_pilot + t_final, 1),
        "capped": capped,
    }


def export_payload(model: str, cp: Mapping[str, Any]) -> dict:
    """The JSON export of every finished point for one model (`load_survival_table` form)."""
    data = []
    for ion, A in IONS.items():
        rows = [
            (e, cp[point_key(model, ion, e)]) for e in ENERGIES if point_key(model, ion, e) in cp
        ]
        if not rows:
            continue
        data.append(
            {
                "Z": Z_OF[ion],
                "A": A,
                "energies": [e for e, _ in rows],
                "alpha": [r["alpha"] for _, r in rows],
                "beta": [r["beta"] for _, r in rows],
                "alpha_sigma": [r["alpha_sigma"] for _, r in rows],
                "beta_sigma": [r["beta_sigma"] for _, r in rows],
                "cells": [r["cells"] for _, r in rows],
            }
        )
    return {
        "meta": {
            "model": model,
            "description": DESCRIPTION,
            "modelParameters": {
                "alphaX": ALPHA_X,
                "betaX": BETA_X,
                "rNucleus": R_N,
                "Dt": D_T[model],
                "cellLine": "HSG",
            },
            "calculusType": f"{model}_MonteCarlo",
            "targetRelSigmaAlpha": TARGET_REL_ALPHA,
        },
        "data": data,
    }


def export(paths: Paths, model: str, cp: Mapping[str, Any]) -> None:
    """Merge every finished point for one model into its JSON export."""
    out = paths.out_dir / survival_table_filename(model, "MonteCarloPaired")
    out.write_text(json.dumps(export_payload(model, cp), indent=1))


def acquire_lock(work: Path) -> IO[str] | None:
    """Refuse to start a second driver.

    Killing a driver does NOT kill the survival.exe it spawned -- the grandchild
    outlives it and keeps a 32-thread job on the CPU. Relaunching on top of that
    put three concurrent jobs on the machine and crashed one outright
    (rc=-1 while building a 3.1M-pixel structure). Two drivers would also race on
    `NewProject_LQparameters_<model>.csv`, which is per model but not per run.
    """
    lock = open(work / "driver.lock", "w")  # held open for the whole run
    if sys.platform != "win32":  # survival.exe is a Windows build; no lock needed
        return lock
    import msvcrt  # the lock is the Windows flavour

    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("another driver holds the lock; refusing to start", file=sys.stderr)
        return None
    return lock


def rel_of(row: Mapping[str, Any]) -> float:
    return row["alpha_sigma"] / max(abs(row["alpha"]), 1e-12)


def refine(paths: Paths, cp: dict, ceiling: float, max_energy: float = 1e9) -> int:
    """Re-run the points that ran out of ceiling, with a bigger one.

    Two different things get flagged `capped` and only one is a modelling error.
    LEMI|O|300 capped because a constant-setup cost model overestimates per-cell
    cost tenfold; LEMII/III|O|300 capped honestly, having spent their full allowance
    to reach ~0.95%. Both want more ceiling, so both are re-run here.

    A refined point REPLACES the old one only if it is actually better. A crash
    or a worse draw must not cost us a result we already hold -- the old value
    stays and the attempt is logged.
    """
    # Select on precision, NOT on the `capped` flag. Capping is how the sizer
    # got there, not a defect in the result: seven 100 MeV/u points are flagged
    # capped at 0.57% against a 0.5% target, and buying hours of compute to move
    # 0.57 -> 0.50 is churn. REFINE_FACTOR sets what actually counts as short.
    targets = sorted(
        (
            k
            for k in cp
            if not k.endswith("failed")
            and isinstance(cp[k], dict)
            and rel_of(cp[k]) > TARGET_REL_ALPHA * REFINE_FACTOR
            and float(k.split("|")[2]) <= max_energy
        ),
        key=lambda k: (float(k.split("|")[2]), _ION_ORDER[k.split("|")[1]]),
    )
    backup = paths.checkpoint.with_suffix(".prerefine.json")
    backup.write_text(json.dumps(cp, indent=1, sort_keys=True))
    print(
        f"refining {len(targets)} points at ceiling={ceiling:g}s (backup: {backup.name})",
        flush=True,
    )
    for key in targets:
        model, ion, e_u = key.split("|")[0], key.split("|")[1], float(key.split("|")[2])
        old = rel_of(cp[key])
        try:
            row = sized_point(paths, model, ion, e_u, max_seconds=ceiling)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(
                f"  KEEP {key}: refinement failed ({type(exc).__name__}), old {old:.2%} stands"[
                    :200
                ],
                flush=True,
            )
            continue
        new = rel_of(row)
        if new < old:
            cp[key] = row
            save_checkpoint(paths.checkpoint, cp)
            export(paths, model, cp)
            print(
                f"  {key:<22} {old:.2%} -> {new:.2%}  cells={row['cells']:>7} "
                f"{row['seconds']:>8.1f}s{' CAPPED' if row['capped'] else ''}",
                flush=True,
            )
        else:
            print(f"  KEEP {key}: new draw {new:.2%} not better than {old:.2%}", flush=True)
    for model in D_T:
        export(paths, model, cp)
    print("DONE", flush=True)
    return 0


def fill(paths: Paths, cp: dict) -> int:
    """Run every grid point not yet in the checkpoint, cheapest first."""
    todo = plan_points(cp)
    print(f"{len(cp)} cached, {len(todo)} to run", flush=True)
    for model, ion, e_u in todo:
        key = point_key(model, ion, e_u)
        try:
            try:
                row = sized_point(paths, model, ion, e_u)
            except RuntimeError as exc:
                # Crashes here have been transient: LEMI|O|300 died with rc=-1 on
                # one attempt and completed on the next with identical arguments.
                # If it is memory, a retry after the previous run has released
                # its several GB is exactly what is needed; if it is not, the
                # second failure costs one probe and is recorded either way.
                print(f"    {key} crashed, one retry: {exc}"[:200], flush=True)
                time.sleep(30)
                row = sized_point(paths, model, ion, e_u)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"  SKIP {key}: {type(exc).__name__} {exc}"[:200], flush=True)
            cp[key + "|failed"] = str(exc)[:200]
            save_checkpoint(paths.checkpoint, cp)
            continue
        cp[key] = row
        save_checkpoint(paths.checkpoint, cp)
        export(paths, model, cp)
        flag = " CAPPED" if row["capped"] else ""
        print(
            f"  {key:<22} cells={row['cells']:>7} {row['seconds']:>8.1f}s  "
            f"alpha={row['alpha']:.4f}+-{row['alpha_sigma']:.4f} "
            f"({row['alpha_sigma'] / abs(row['alpha']):.2%}){flag}",
            flush=True,
        )
    for model in D_T:
        export(paths, model, cp)
    print("DONE", flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the paired Survival reference-MC exports (resumable).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and cost; run and read nothing"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="folder for the *_MonteCarloPaired_HSG_Manganaro2018.json exports "
        "(default: $PYRADBIO_SURVIVAL_DIR)",
    )
    parser.add_argument(
        "--exe",
        type=Path,
        help="survival.exe (default: $PYRADBIO_SURVIVAL_EXE or <src>/../survival.exe)",
    )
    parser.add_argument(
        "--work",
        type=Path,
        help="scratch folder with checkpoint.json "
        "(default: $PYRADBIO_SURVIVAL_WORK or <src>/../svgen)",
    )
    parser.add_argument("--refine", action="store_true", help="re-run points short of target")
    parser.add_argument(
        "--ceiling", type=float, default=LONG_SECONDS, help="per-point ceiling for --refine [s]"
    )
    parser.add_argument(
        "--maxE", type=float, default=1e9, help="--refine only points up to this energy [MeV/u]"
    )
    return parser


def _print_plan(args: argparse.Namespace) -> None:
    todo = plan_points({})
    print(
        f"survival_paired_mc: {len(todo)} points = {len(D_T)} models x {len(IONS)} ions x "
        f"{len(ENERGIES)} energies (cheapest first; points already in checkpoint.json are skipped)"
    )
    for model, ion, e_u in todo:
        print(f"  {point_key(model, ion, e_u)}")
    print(
        f"HSG LEM row: alpha_x={ALPHA_X:g} beta_x={BETA_X:g} R_N={R_N:g} um "
        f"D_t={', '.join(f'{m}={d:g}' for m, d in D_T.items())} Gy"
    )
    print(
        f"sizing: pilot {PILOT_MIN_CELLS} + {PILOT_CELLS} cells, target sigma_alpha/alpha="
        f"{TARGET_REL_ALPHA:g}, cells in [{MIN_CELLS}, {MAX_CELLS}], ceiling {MAX_SECONDS} s "
        f"(long {LONG_SECONDS} s, last resort {LAST_RESORT_SECONDS} s)"
    )
    print(
        f"cost: up to ~{MAX_SECONDS / 3600:.1f} h per point plus pilots; the 2026 run of the "
        "full grid took several days (3 of 60 points need > 4 h for one cell)"
    )
    out = args.out if args.out is not None else "$PYRADBIO_SURVIVAL_DIR"
    print(f"writes: {out}/{survival_table_filename('<model>', 'MonteCarloPaired')}")
    if args.refine:
        print(f"--refine: points above {TARGET_REL_ALPHA * REFINE_FACTOR:.2%} up to {args.maxE:g}")


def resolve_paths(args: argparse.Namespace) -> Paths:
    """The binary, data, scratch and output folders (reads the dataset configuration)."""
    from pyradbio.datasets import SURVIVAL, SURVIVAL_SRC, dataset_dir

    src = dataset_dir(SURVIVAL_SRC)
    exe = args.exe or Path(os.environ.get("PYRADBIO_SURVIVAL_EXE", src.parent / "survival.exe"))
    work = args.work or Path(os.environ.get("PYRADBIO_SURVIVAL_WORK", src.parent / "svgen"))
    out_dir = args.out if args.out is not None else dataset_dir(SURVIVAL)
    return Paths(exe=Path(exe), data=src / "data", work=Path(work), out_dir=Path(out_dir))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run:
        _print_plan(args)
        return 0
    paths = resolve_paths(args)
    paths.work.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(paths.work)
    if lock is None:
        return 2
    if not paths.exe.exists():
        print(
            f"missing {paths.exe}; build it per survival_repo_notes.md section 8", file=sys.stderr
        )
        return 1
    cp = load_checkpoint(paths.checkpoint)
    if args.refine:
        return refine(paths, cp, args.ceiling, args.maxE)
    return fill(paths, cp)


if __name__ == "__main__":
    raise SystemExit(main())
