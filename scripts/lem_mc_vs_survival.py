"""Regenerate pyRadBio's Monte-Carlo points for the Survival LEM I-III comparison.

``validation/survival_code_lem.py`` overlays pyRadBio's own reference Monte Carlo
(``LEMBase.scholz1997_montecarlo_survival``) on Survival's, for LEM I, II and III
on the Manganaro et al. 2018 HSG cell (``pyradbio.benchmark.manganaro2018``),
driven by Survival's own SRIM LET. It reads the points from the committed cache
``validation/data/survival_our_mc.json``; this script writes it.

**The cache.** One JSON object ``{key: point}``: ``key`` is the 16-hex-digit hash
(``manganaro2018.mc_cache_key``) of everything that can change the point -- the
cell, ``D_t``, ``r_max``/``r_min`` and the core width of the radial dose, the LET,
the dose set and every sizing setting -- so a rerun recomputes only what moved and
the reader never picks up a stale entry. The reserved key ``"_provenance"``
records the stopping power, the engine, the code version and a **numeric
fingerprint**: three small, data-free, deterministic points
(``manganaro2018.mc_fingerprint``: one whole adaptive LEM I point incl. the near/far
split, one LEM II and one LEM III point; numpy engine, fixed seeds, analytic Bethe), which
``tests/benchmark/test_survival_cache_fingerprint.py`` recomputes in CI. When that
test fails, the Monte-Carlo code has changed: delete the cache and regenerate all
of it (``--fresh``), which also refreshes the header. ``--refresh-provenance``
rewrites the header *without* recomputing any point; it is only for a header-only
change (a new fingerprint point, a version string) once the entries are proven
current -- using it to silence a failing fingerprint leaves stale numbers behind a
green test. ``--fresh`` is refused with ``--only``/``--models`` unless ``--out``
names another folder, because it would replace the whole committed cache by a subset.

**Every point is sized, not fixed** (``manganaro2018.adaptive_montecarlo_point``):
a pilot of 400 realisations x 8 seeds measures the spread, and the count is set
to reach 0.3% relative on alpha and 2% of beta_x absolute on beta (beta legitimately
crosses zero at high LET). A flat 4000 x 4 budget runs from 0.02% (protons, 100
MeV/u) to 1.98% (oxygen, 10): a low-LET point puts many traversals through every
cell, so its cell-to-cell spread is small. 8 seeds rather than 4, because a 4-seed
sigma has 3 degrees of freedom -- noisy enough that an unlucky draw can read as
"LEM II's near/far cut is not converged".

**Reaching Survival's energies (carbon and oxygen to 300 MeV/u, all versions).**
Beyond ``b_cut_um = 10`` um tracks stop being resolved individually and enter as
one Gaussian draw with closed-form compound-Poisson moments
(``LEMBase.kellerer_field_integral``), applied where the top dose puts at least
2000 tracks in reach of the nucleus: carbon 300 has ``n_bar = 9.7e6`` traversals
at 6 Gy, of which 2131 are near, ~30 h enumerated against ~25 s. Radius and
threshold are the Scholz 1997 batch's (below ``n_bar ~ 2000`` the far field
fluctuates by ~4.5% and is worth resolving; above it 0.25-0.34%). The cut is
validated by a scan, not two points: 10/15/20/25/40 um at N = 2000 x 4 move alpha
by at most 0.44% (carbon 100) and 0.63% (carbon 300) for LEM II and <= 0.40% for
LEM III, with no monotone dependence on the radius -- scatter, inside a point's
statistical error, so 10 um stands. A LEM II/III point costs ~3-5x a LEM I one at
matched energy (2.7x at oxygen 300, 4.8x at protons 100): the radical-diffusion
profile has a closed far-field form (interpolating it once per field point x track
would cost ~20x). See ``docs/references/survival_repo_notes.md`` section 5.

LEM IV is excluded by construction: its effect is not a pointwise function of the
local dose, so its Monte Carlo raises rather than returning a LEM I answer.

Usage::

    python scripts/lem_mc_vs_survival.py --dry-run           # the plan, no compute
    python scripts/lem_mc_vs_survival.py                     # fill missing points
    python scripts/lem_mc_vs_survival.py --only LEMI:C:20 --out /tmp/check
    python scripts/lem_mc_vs_survival.py --fresh             # after a fingerprint change
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pyradbio.benchmark import manganaro2018 as m18

CACHE_NAME = "survival_our_mc.json"
#: Survival's ``-ion`` symbols -> Z, for ``--only``.
_ION_Z = {ion: Z for ion, (Z, _) in m18.PAIRED_IONS.items()}


def default_out_dir() -> Path:
    """``validation/data/`` of this checkout."""
    return Path(__file__).resolve().parents[1] / "validation" / "data"


def plan_points(
    models: Sequence[str] | None = None,
    only: set[tuple[str, int, float]] | None = None,
    settings: m18.OurMcSettings = m18.OUR_MC,
) -> list[tuple[str, int, int, float]]:
    """``(model, Z, A, E)`` for every point of the grid, in the reader's order."""
    out = []
    for model in settings.models():
        if models is not None and model not in models:
            continue
        for Z, A in m18.PAIRED_IONS.values():
            for e_u in settings.energies(model, Z):
                if only is None or (model, Z, e_u) in only:
                    out.append((model, Z, A, e_u))
    return out


def parse_only(spec: str) -> set[tuple[str, int, float]]:
    """``"LEMI:C:20,LEMII:He:300"`` -> ``{("LEMI", 6, 20.0), ("LEMII", 2, 300.0)}``."""
    out = set()
    for token in spec.split(","):
        model, ion, energy = (part.strip() for part in token.split(":"))
        if model not in m18.LEM_MODELS:
            raise argparse.ArgumentTypeError(f"unknown model {model!r}")
        if ion not in _ION_Z:
            raise argparse.ArgumentTypeError(f"unknown ion {ion!r} (one of {sorted(_ION_Z)})")
        out.add((model, _ION_Z[ion], float(energy)))
    return out


def provenance(engine: str) -> dict[str, Any]:
    """The cache header: sources, engine, code version and the numeric fingerprint."""
    import pyradbio

    return {
        "generator": "scripts/lem_mc_vs_survival.py",
        "stopping_power": (
            "Survival's SRIM tables (pyradbio.track.srim.srim_stopping_power for H, He, C, "
            "O: electronic stopping, linear-in-energy interpolation)"
        ),
        "engine": engine,
        "code_version": f"pyradbio {pyradbio.__version__}",
        "fingerprint": m18.mc_fingerprint(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate validation/data/survival_our_mc.json (pyRadBio's MC points).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and cost; compute and read nothing"
    )
    parser.add_argument(
        "--out", type=Path, help=f"folder holding {CACHE_NAME} (default: validation/data/)"
    )
    parser.add_argument("--models", help="comma-separated subset of LEMI,LEMII,LEMIII")
    parser.add_argument("--only", type=parse_only, help="points MODEL:ION:E[,...], e.g. LEMI:C:20")
    parser.add_argument(
        "--engine",
        choices=("auto", "numpy", "numba"),
        default="auto",
        help="Monte-Carlo engine (default auto: numba when installed, as the cache was made)",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="start a new cache (delete the old file first)"
    )
    parser.add_argument(
        "--refresh-provenance",
        action="store_true",
        help=(
            "rewrite only the _provenance header (fingerprint included) and run no point; "
            "ONLY for header-only changes once the entries are proven current -- never to "
            "silence a failing fingerprint test (regenerate with --fresh instead)"
        ),
    )
    return parser


def _print_plan(points: list[tuple[str, int, int, float]], out: Path, engine: str) -> None:
    s = m18.OUR_MC
    print(f"lem_mc_vs_survival: {len(points)} points -> {out / CACHE_NAME} (engine={engine})")
    for model, Z, _A, e_u in points:
        print(f"  {model:7s} Z={Z} E={e_u:g} MeV/u")
    print(
        f"per point: pilot {s.n_pilot} x {s.n_repeat} seeds, sized to sigma_alpha/alpha="
        f"{s.target_rel_alpha:g} and sigma_beta={s.target_abs_beta:.3g} Gy^-2, realisations "
        f"in [{s.n_min}, {s.n_max}]; near/far cut {s.b_cut_um:g} um from "
        f"n_bar >= {s.b_cut_min_nbar:g}; doses {list(s.doses_Gy)} Gy"
    )
    print(
        "cost (numba, this laptop): seconds per point at low LET, up to ~1 h for the "
        "high-LET points that hit the 200000-realisation clip; the full 78-point grid is "
        "an overnight run, the numpy engine ~25x slower. Points already cached are skipped."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.fresh and (args.only or args.models) and args.out is None:
        parser.error(
            "--fresh with --only/--models would replace the whole committed cache by a "
            "subset; pass --out to write the subset elsewhere"
        )
    models = args.models.split(",") if args.models else None
    points = plan_points(models, args.only)
    out_dir = args.out if args.out is not None else default_out_dir()
    if args.dry_run:
        _print_plan(points, out_dir, args.engine)
        return 0

    from pyradbio.benchmark.checkpoint import JsonPointCache
    from pyradbio.core.radiation import Ion
    from pyradbio.track.srim import srim_stopping_power

    path = out_dir / CACHE_NAME
    if args.fresh:
        path.unlink(missing_ok=True)
    cache = JsonPointCache(path, newline="\n")
    if args.refresh_provenance:
        cache.provenance = provenance(args.engine)
        cache.save()
        print(f"rewrote the _provenance header of {path}")
        return 0
    if cache.provenance is None:
        cache.provenance = provenance(args.engine)
        cache.save()
    elif cache.provenance.get("engine") != args.engine:
        print(
            f"warning: {path.name} was generated with engine={cache.provenance.get('engine')!r}, "
            f"this run uses {args.engine!r}"
        )

    stopping = srim_stopping_power(("H", "He", "C", "O"))
    lems = {m: m18.survival_matched_lem(m, stopping) for m in {p[0] for p in points}}
    done = 0
    for model, Z, A, e_u in points:
        lem = lems[model]
        key = m18.mc_cache_key(model, Z, A, e_u, lem, stopping)
        if key in cache:
            continue
        t0 = time.perf_counter()
        row = m18.adaptive_montecarlo_point(lem, Ion(Z, A, e_u), stopping, engine=args.engine)
        cache.put(key, {**row, "model": model, "Z": Z, "A": A, "E": float(e_u)})
        done += 1
        print(
            f"  {model:7s} Z={Z} E={e_u:g}: alpha={row['alpha']:.4f}+-{row['alpha_sigma']:.4f} "
            f"beta={row['beta']:.4f}+-{row['beta_sigma']:.4f} n_real={row['n_real']:.0f} "
            f"({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )
    print(f"{done} points computed, {len(points) - done} already cached -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
