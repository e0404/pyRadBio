"""Offline LEM IV explicit-track Monte Carlo for Friedrich et al. 2012, Fig. 4.

Friedrich, Scholz, Elsasser, Durante & Scholz 2012, Int. J. Radiat. Biol.
88(1-2):103-107, doi:10.3109/09553002.2011.611213, Fig. 4 (p. 107): the
"hypothetical" cell's ``alpha_I(LET)`` and ``beta_I(LET)``.

For each carbon LET this runs ``LEM4.local_dose_mc_alpha_beta``: real track
configurations are sampled, each 540 nm subvolume's local dose is accumulated,
LEM IV's whole-nucleus cluster index is scored on the pattern, and ``alpha_I``
*and* ``beta_I`` are read off the one simulated survival curve under the exact
``<exp(-E)>`` convention -- the Scholz 1997 prescription Elsasser 2010 invokes for
the macroscopic coefficients. Every seed shares one fit window
(``LEM4.fit_doses_Gy()``, the window of the deterministic engines on the same
panels), and the error bar is the plain scatter of the seed mean.

Output: ``<out>/lem4_friedrich2012_mc.csv`` (default ``validation/data/``), one row
per ``(let_keV_um, seed)``, written through
`pyradbio.benchmark.checkpoint.CsvCheckpoint` after every seed, so an
interrupted run resumes where it stopped. The header records the provenance
(stopping power, engine, window, sample sizes, code version) and the numeric
fingerprint: one small NumPy run at a fixed seed
(`pyradbio.benchmark.friedrich2012.FIG4_MC_FINGERPRINT_POINTS`), which
``tests/models/lem/test_lem4_mc_fingerprint.py`` recomputes. A resumed run
recomputes it first and refuses to add rows if the code has changed since the
banked ones. ``validation/lem4_elsasser2010_friedrich2012.py`` overlays the result.

Cost: roughly 2 h for the 12 production LETs with the numba box-average kernel
(the ``fast`` extra); low LET is the expensive end (``--dry-run`` prints the
per-LET estimate). Run it detached (scheduled task) on a machine that will not
sleep or reboot; see ``scripts/README.md``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from pyradbio.benchmark import friedrich2012 as f12

OUTPUT_NAME = f12.FIG4_MC_CSV_NAME

# Measured wall time [min] per LET at 8 seeds x 400 realisations with the numba
# kernel: LET 100 2.8, 70 1.5, and with the impact cut 5 at LET 40 and 5 at
# LET 30. LET 15: 48 min, recorded for the panel's own 8 x 400 run (LET 13 was left
# out at 133 min); it was placed next to a 10 x 500 measurement,
# so the sample size is UNVERIFIED -- the production CSV's wall_s column settles it.
# Interpolated log-log in LET, flat outside; a rough guide only.
_COST_ANCHORS_MIN: tuple[tuple[float, float], ...] = (
    (15.0, 48.0),
    (30.0, 5.0),
    (40.0, 5.0),
    (70.0, 1.5),
    (100.0, 2.8),
)
_ANCHOR_SAMPLES = 8 * 400
# The NumPy box-averaged profile adds ~4.5 s per (LET, seed) call over numba.
_NUMPY_EXTRA_S_PER_CALL = 4.5

# Header keys that must match before a run may add rows to an existing file.
_CONFIG_KEYS = ("model", "stopping_power", "engine", "window_Gy", "n_realizations", "n_seeds")


def _parse_lets(text: str) -> tuple[float, ...]:
    try:
        lets = tuple(float(tok) for tok in text.split(",") if tok.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--lets must be comma-separated numbers: {exc}") from exc
    if not lets or any(not (math.isfinite(x) and x > 0.0) for x in lets):
        raise argparse.ArgumentTypeError(f"--lets needs positive LETs, got {text!r}")
    return tuple(sorted(set(lets)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lem4_friedrich2012_mc.py",
        description=(
            "LEM IV explicit-track MC (alpha_I, beta_I) per carbon LET for Friedrich et al. "
            "2012 Fig. 4; writes lem4_friedrich2012_mc.csv (resumable)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder (default: validation/data/ of this checkout)",
    )
    parser.add_argument(
        "--lets",
        type=_parse_lets,
        default=f12.FIG4_MC_LETS,
        help="comma-separated carbon LETs [keV/um] (default: the 12 production LETs)",
    )
    parser.add_argument(
        "--realizations",
        type=int,
        default=f12.FIG4_MC_REALIZATIONS,
        help=f"realisations per seed (default {f12.FIG4_MC_REALIZATIONS})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=f12.FIG4_MC_SEEDS,
        help=f"independent seeds 0..N-1 per LET, >= 2 (default {f12.FIG4_MC_SEEDS})",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "numpy", "numba"),
        default="auto",
        help="box-averaged profile kernel (default auto = numba when installed)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard an existing output file instead of resuming it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and a cost estimate, compute nothing",
    )
    return parser


def estimate_minutes(let_keV_um: float, n_seeds: int, n_realizations: int, engine: str) -> float:
    """Rough wall time [min] of one LET point (see ``_COST_ANCHORS_MIN``)."""
    xs = [math.log(x) for x, _ in _COST_ANCHORS_MIN]
    ys = [math.log(y) for _, y in _COST_ANCHORS_MIN]
    x = math.log(let_keV_um)
    if x <= xs[0]:
        y = ys[0]
    elif x >= xs[-1]:
        y = ys[-1]
    else:
        i = next(k for k in range(1, len(xs)) if x <= xs[k])
        t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
        y = ys[i - 1] + t * (ys[i] - ys[i - 1])
    minutes = math.exp(y) * n_seeds * n_realizations / _ANCHOR_SAMPLES
    if engine == "numpy":
        minutes += n_seeds * _NUMPY_EXTRA_S_PER_CALL / 60.0
    return minutes


def _default_out() -> Path:
    return Path(__file__).resolve().parents[1] / "validation" / "data"


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an OS lock on ``path`` (released by the OS if the process dies).

    A second generator on the same output refuses to start instead of writing the
    same file concurrently. The holder removes the lock file on every exit, including
    an exception or Ctrl-C, so a run into ``validation/data/`` leaves nothing but the
    CSV to commit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    acquired = False
    try:
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SystemExit(f"{path} is locked: another generator is writing this output") from exc
        acquired = True
        yield
    finally:
        handle.close()
        if acquired:  # never remove a lock another generator holds
            with suppress(OSError):
                path.unlink()


def _print_plan(args: argparse.Namespace, out_file: Path) -> None:
    print("LEM IV explicit-track MC, Friedrich et al. 2012 Fig. 4 (hypothetical cell)")
    print(f"  output       {out_file}")
    print(f"  LETs         {', '.join(f'{x:g}' for x in args.lets)} keV/um (carbon-12)")
    print(f"  per LET      {args.seeds} seeds x {args.realizations} realisations")
    print(f"  engine       {args.engine}")
    fp = ", ".join(f"LET {let:g} n={n} seed={s}" for let, n, s in f12.FIG4_MC_FINGERPRINT_POINTS)
    print(f"  fingerprint  {fp} (engine numpy, ~6 s)")
    total = 0.0
    print("  estimated cost (rough; anchored on measured numba runs):")
    for let in args.lets:
        minutes = estimate_minutes(let, args.seeds, args.realizations, args.engine)
        total += minutes
        print(f"    LET {let:6.1f}  ~{minutes:6.1f} min")
    print(f"  total        ~{total:.0f} min ({total / 60.0:.1f} h)")


def _header(args: argparse.Namespace, model: Any, doses: Any) -> dict[str, Any]:
    import pyradbio
    from pyradbio.models.lem import resolve_mc_engine
    from pyradbio.track.stopping_power import describe_stopping_power

    resolved = args.engine if args.engine != "auto" else resolve_mc_engine("auto")
    return {
        "generator": "scripts/lem4_friedrich2012_mc.py",
        "source": f"Friedrich et al. 2012, doi:{f12.FRIEDRICH2012_DOI}, Fig. 4 (hypothetical cell)",
        "model": (
            "LEM4(fig4_hypothetical, local_dose_mc_impact_cut='auto'); "
            "local_dose_mc_alpha_beta(overkill=True), exact <exp(-E)>"
        ),
        "stopping_power": describe_stopping_power(model.stopping_power),
        "engine": f"{args.engine} (box-average kernel: {resolved})",
        "window_Gy": (
            f"{float(doses[0])!r}-{float(doses[-1])!r} ({len(doses)} doses, LEM4.fit_doses_Gy())"
        ),
        "n_realizations": str(args.realizations),
        "n_seeds": str(args.seeds),
        "pyradbio_version": pyradbio.__version__,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seeds < 2:
        parser.error("--seeds must be >= 2 (the error bar is the scatter of the seed mean)")
    if args.realizations < 1:
        parser.error("--realizations must be >= 1")
    out_dir = args.out if args.out is not None else _default_out()
    out_file = out_dir / OUTPUT_NAME

    if args.dry_run:
        _print_plan(args, out_file)
        return 0

    from pyradbio.benchmark.checkpoint import FINGERPRINT_KEY, CsvCheckpoint, assert_fingerprint
    from pyradbio.track.stopping_power import default_stopping_power

    sp = default_stopping_power()
    model = f12.fig4_mc_model(engine=args.engine, stopping_power=sp)
    doses = model.fit_doses_Gy()
    config = _header(args, model, doses)

    with _exclusive_lock(out_file.with_name(out_file.name + ".lock")):
        if args.fresh and out_file.exists():
            out_file.unlink()
        ckpt = CsvCheckpoint(
            out_file, f12.FIG4_MC_FIELDS, key_columns=f12.FIG4_MC_KEY_COLUMNS, text_columns=()
        )
        print("computing the fingerprint (engine numpy) ...", flush=True)
        fingerprint = f12.fig4_mc_fingerprint(stopping_power=sp)
        if ckpt.header:
            stale = [k for k in _CONFIG_KEYS if ckpt.header.get(k) != config[k]]
            if stale:
                print(
                    f"{out_file} was written with a different configuration ({', '.join(stale)}); "
                    "rerun with --fresh to replace it",
                    file=sys.stderr,
                )
                return 2
            try:
                assert_fingerprint(
                    ckpt.header.get(FINGERPRINT_KEY), fingerprint, generator=f12.FIG4_MC_GENERATOR
                )
            except AssertionError as exc:
                print(
                    f"{out_file}: the code changed since its rows were banked ({exc})",
                    file=sys.stderr,
                )
                return 2
            print(f"resuming {out_file} ({len(ckpt)} rows banked)")
        else:
            ckpt.header = {**config, FINGERPRINT_KEY: fingerprint}
            ckpt.save()

        print(
            f"\nFriedrich 2012 Fig. 4: explicit-track MC, alpha_I AND beta_I off one simulated\n"
            f"survival curve (exact <exp(-E)>, the Scholz 1997 prescription), window "
            f"{doses[0]:.2f}-{doses[-1]:.2f} Gy,\n{args.seeds} seeds x {args.realizations} "
            f"realisations -> {out_file}:",
            flush=True,
        )
        for let in args.lets:
            try:
                ion = f12.fig4_mc_ion(let, sp)
            except ValueError:
                print(f"    LET {let:5.0f}: unreachable for carbon on the energy bracket, skipped")
                continue
            for seed in range(args.seeds):
                if (float(let), float(seed)) in ckpt:
                    continue
                t0 = time.perf_counter()
                alpha, beta = model.local_dose_mc_alpha_beta(
                    ion, doses_Gy=doses, n_realizations=args.realizations, seed=seed
                )
                ckpt.put(
                    {
                        "let_keV_um": float(let),
                        "seed": seed,
                        "energy_MeV_u": ion.energy_MeV_u,
                        "n_realizations": args.realizations,
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "wall_s": round(time.perf_counter() - t0, 1),
                    }
                )
                ckpt.save()  # bank every seed
            records = [ckpt.record((float(let), float(s))) for s in range(args.seeds)]
            a, a_err, b, b_err = f12.fig4_mc_seed_statistics(
                [[r["alpha"], r["beta"]] for r in records]
            )
            print(
                f"    LET {let:5.0f}: alpha={a:.4f}+-{a_err:.4f}  beta={b:+.4f}+-{b_err:.4f}",
                flush=True,
            )
    print(f"\nwrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
