"""pyRadBio command-line interface.

Subcommands:
    pyradbio survival --alpha A --beta B --dose D
    pyradbio table    --model lem1 --alpha-x .. --beta-x .. --dt .. --rn ..
                      --ions 1,6 --energies 10,100 --out table.csv
                      [--mass-numbers ..] [--stopping-power ..]
                      [--beta-model ..] [--initial-slope-model ..]
                      [--survival-convention ..] [--cluster-index-model ..]
                      [--cluster-index-cap ..] [--local-dose-overkill]
    pyradbio benchmark [--dataset pide|radphysbio|csv] [--path PATH]
                       --model M --out DIR

``table`` always writes the model's RESOLVED configuration into the CSV metadata
header -- not the flags that were typed -- so a table states the protocol it was
computed under even when every axis was left at its default. LEM IV has a dozen
such axes and they change the numbers without changing the call; see
``docs/references/LEM_IV_OPEN_CHOICES.md``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import get_args

from pyradbio.core.elements import default_mass_number
from pyradbio.core.lq import survival
from pyradbio.core.radiation import Ion
from pyradbio.models.base import BiophysicalModel
from pyradbio.track.stopping_power import StoppingPower


def _parse_int_list(text: str) -> list[int]:
    return [int(tok) for tok in text.split(",") if tok.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(tok) for tok in text.split(",") if tok.strip()]


def _require_flags(args: argparse.Namespace, flags: Sequence[str], model: str) -> None:
    """Exit cleanly (no traceback) when a model's required flags are missing."""
    missing = [f"--{f.replace('_', '-')}" for f in flags if getattr(args, f) is None]
    if missing:
        raise SystemExit(f"pyradbio table --model {model}: missing {', '.join(missing)}")


#: ``--flag`` -> model dataclass field, for the optional model-behaviour flags.
#: Applied only when the chosen model actually has the field; passing one to a
#: model that does not is an error rather than a silent no-op.
_BEHAVIOUR_FLAGS: dict[str, str] = {
    "beta_model": "beta_model",
    "initial_slope_model": "initial_slope_model",
    "survival_convention": "survival_convention",
    "cluster_index_model": "cluster_index_model",
    "cluster_index_cap": "legacy_trip98_cluster_index_cap",
    "local_dose_overkill": "local_dose_overkill",
    "engine": "engine",
}


def _behaviour_overrides(args: argparse.Namespace, cls: type, model: str) -> dict[str, object]:
    """Requested model-behaviour overrides, validated against ``cls``'s fields."""
    from dataclasses import fields as dataclass_fields

    available = {f.name for f in dataclass_fields(cls)}
    overrides: dict[str, object] = {}
    unsupported: list[str] = []
    for flag, field_name in _BEHAVIOUR_FLAGS.items():
        value = getattr(args, flag, None)
        if value is None or value is False:
            continue
        if field_name in available:
            overrides[field_name] = value
        else:
            unsupported.append(f"--{flag.replace('_', '-')}")
    if unsupported:
        raise SystemExit(
            f"pyradbio table --model {model}: {', '.join(sorted(unsupported))} "
            f"not supported by {cls.__name__}"
        )
    return overrides


#: ``--stopping-power`` choices. ``default`` is the library default
#: (:func:`~pyradbio.track.stopping_power.default_stopping_power`: ICRU 90's procedure
#: for Z 1-18, Hubert alpha scaling for Z 19-92 from 2.5 MeV/u, anchored analytic Bethe
#: elsewhere); ``bethe`` is the analytic
#: :class:`~pyradbio.track.stopping_power.SimpleBetheStoppingPower` alone; the
#: ``icru*`` / ``xstar`` choices are the :mod:`pyradbio.track.stopping` tables (Z 1-18,
#: no extrapolation); ``libdedx`` is GPL and only works when the user installed it.
STOPPING_POWER_CHOICES = (
    "default",
    "bethe",
    "icru90-procedure",
    "icru2016",
    "icru2009",
    "icru2005",
    "xstar",
    "ns1970",
    "libdedx",
)


def _resolve_stopping_power(name: str | None) -> StoppingPower | None:
    """``--stopping-power`` -> a stopping-power instance, or None for the model default.

    Omitting the flag leaves each model on its own default, which is
    :func:`~pyradbio.track.stopping_power.default_stopping_power`.
    """
    if name is None:
        return None
    from pyradbio.track.stopping_power import (
        LibdedxStoppingPower,
        SimpleBetheStoppingPower,
        default_stopping_power,
    )

    if name == "default":
        return default_stopping_power()
    if name == "bethe":
        return SimpleBetheStoppingPower()
    if name == "icru90-procedure":
        from pyradbio.track.stopping import ICRU90Procedure

        return ICRU90Procedure()
    if name in ("icru2016", "icru2009", "icru2005"):
        from pyradbio.track.stopping import ICRU

        return ICRU(edition=name[4:])  # type: ignore[arg-type]
    if name == "xstar":
        from pyradbio.track.stopping import XSTAR

        return XSTAR()
    if name == "ns1970":
        from pyradbio.track.northcliffe_schilling import NorthcliffeSchillingWithBetheTail

        return NorthcliffeSchillingWithBetheTail()
    if name == "libdedx":
        try:
            return LibdedxStoppingPower()
        except ImportError as exc:  # pragma: no cover - depends on a user-installed package
            raise SystemExit(f"pyradbio table --stopping-power libdedx: {exc}") from exc
    raise SystemExit(f"unknown --stopping-power {name!r}; use {'|'.join(STOPPING_POWER_CHOICES)}")


def _build_lem(model: str) -> Callable[[argparse.Namespace], BiophysicalModel]:
    def builder(args: argparse.Namespace) -> BiophysicalModel:
        from pyradbio.core.cell import CellTarget
        from pyradbio.core.photon import PhotonParameters
        from pyradbio.models.lem.base import LEMBase
        from pyradbio.models.lem.lem1 import LEM1
        from pyradbio.models.lem.lem2 import LEM2
        from pyradbio.models.lem.lem3 import LEM3
        from pyradbio.models.lem.lem4 import LEM4

        _require_flags(args, ["alpha_x", "beta_x", "dt", "rn"], model)
        photon = PhotonParameters(alpha_x=args.alpha_x, beta_x=args.beta_x, D_t=args.dt)
        cell = CellTarget(
            nucleus_radius_um=args.rn,
            **(
                {}
                if args.effective_size_factor is None
                else {"effective_size_factor": args.effective_size_factor}
            ),
        )
        classes: dict[str, type[LEMBase]] = {
            "lem1": LEM1,
            "lem2": LEM2,
            "lem3": LEM3,
            "lem4": LEM4,
        }
        cls = classes[model]
        kwargs: dict[str, object] = _behaviour_overrides(args, cls, model)
        stopping_power = _resolve_stopping_power(args.stopping_power)
        if stopping_power is not None:
            kwargs["stopping_power"] = stopping_power
        return cls(photon=photon, cell=cell, **kwargs)  # type: ignore[arg-type]

    return builder


def _build_mkm(args: argparse.Namespace) -> BiophysicalModel:
    """MKM on the pyMKM-free :class:`NativeMKMBackend` (the library default)."""
    from pyradbio.models.mkm.model import MKMModel
    from pyradbio.models.mkm.native_backend import NativeMKMBackend

    _require_flags(args, ["alpha0", "beta0", "rd", "rn"], "mkm")
    backend = NativeMKMBackend(
        domain_radius_um=args.rd, nucleus_radius_um=args.rn, beta0=args.beta0
    )
    stopping_power = _resolve_stopping_power(args.stopping_power)
    if stopping_power is not None:
        backend = replace(backend, stopping_power=stopping_power)
    return MKMModel(alpha0=args.alpha0, backend=backend)


#: model name -> builder. LEM I-IV share the photon+cell flags; MKM uses its own.
_MODEL_BUILDERS: dict[str, Callable[[argparse.Namespace], BiophysicalModel]] = {
    "lem1": _build_lem("lem1"),
    "lem2": _build_lem("lem2"),
    "lem3": _build_lem("lem3"),
    "lem4": _build_lem("lem4"),
    "mkm": _build_mkm,
}


def _cmd_survival(args: argparse.Namespace) -> int:
    s = survival(args.dose, args.alpha, args.beta)
    print(f"S({args.dose:g} Gy) = {s:.6f}")
    return 0


def _cmd_table(args: argparse.Namespace) -> int:
    from pyradbio.tables.alpha_beta import build_alpha_beta_table
    from pyradbio.tables.io import write_csv

    model = _MODEL_BUILDERS[args.model](args)
    zs = _parse_int_list(args.ions)
    if args.mass_numbers is None:
        masses = [default_mass_number(z) for z in zs]
    else:
        masses = _parse_int_list(args.mass_numbers)
        if len(masses) != len(zs):
            raise SystemExit(
                f"pyradbio table: --mass-numbers has {len(masses)} entries but --ions has {len(zs)}"
            )
    ions = [Ion(Z=z, A=a, energy_MeV_u=1.0) for z, a in zip(zs, masses, strict=True)]
    energies = _parse_float_list(args.energies)
    # ``build_alpha_beta_table`` records the model's RESOLVED configuration, so the
    # table names the protocol it was computed under rather than the flags that
    # happened to be typed. Only the CLI's own framing goes in here.
    table = build_alpha_beta_table(model, ions, energies, metadata={"model": args.model})
    write_csv(table, args.out)
    print(f"Wrote {len(table.rows)} rows to {args.out}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from pyradbio.benchmark.dataset import load_dataset
    from pyradbio.benchmark.harness import build_model_spec, run_benchmark
    from pyradbio.benchmark.pide import load_pide_dataset
    from pyradbio.benchmark.report import write_report

    # ``--pide PATH`` is the original spelling of ``--dataset pide --path PATH``.
    path = args.path if args.path is not None else args.pide
    try:
        if args.dataset == "pide":
            dataset = load_pide_dataset(path, variant=args.variant)
        else:
            dataset = load_dataset(args.dataset, path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    # Map CLI flags to the harness kwargs, keeping only those the user supplied
    # (the rest fall back to build_model_factory's documented defaults).
    flag_to_kwarg = {"dt": "D_t", "rn": "nucleus_radius_um", "rd": "domain_radius_um"}
    factory_kwargs: dict[str, float] = {
        flag_to_kwarg.get(name, name): value
        for name in ("dt", "rn", "rd", "alpha0", "beta0")
        if (value := getattr(args, name)) is not None
    }
    try:
        spec = build_model_spec(args.model, **factory_kwargs)
        result = run_benchmark(dataset, model_name=args.model, spec=spec, limit=args.limit)
    except ValueError as exc:
        raise SystemExit(f"pyradbio benchmark --model {args.model}: {exc}") from exc
    if result.n_points == 0:
        raise SystemExit(
            f"No {dataset.name} points were evaluable for this model "
            f"(all {result.n_skipped} skipped: uncovered ion/energy)."
        )
    report = write_report(result, args.out, plots=not args.no_plots)
    print(
        f"{args.model} on {dataset.name}: {result.n_points} points ({result.n_skipped} skipped, "
        f"{result.n_out_of_scope} out of the model's fitted scope), "
        f"mean |log err| alpha = {result.mean_abs_log_error_alpha:.4f}, "
        f"SMAPE = {result.smape_alpha:.4f}"
    )
    print(f"Wrote report to {report}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    # The model-behaviour choices come from the models' own Literal annotations, so
    # the CLI cannot drift from what the constructors accept.
    from pyradbio.benchmark.dataset import dataset_names
    from pyradbio.benchmark.harness import benchmark_models
    from pyradbio.models.lem.base import InitialSlopeModel
    from pyradbio.models.lem.engine import ENGINES
    from pyradbio.models.lem.lem1 import LEM1BetaModel
    from pyradbio.models.lem.lem4 import (
        LEM4BetaModel,
        LEM4ClusterIndexModel,
        LEM4SurvivalConvention,
    )

    parser = argparse.ArgumentParser(prog="pyradbio")
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")

    surv = sub.add_parser("survival", help="LQ surviving fraction for one dose")
    surv.add_argument("--alpha", type=float, required=True, help="alpha [Gy^-1]")
    surv.add_argument("--beta", type=float, required=True, help="beta [Gy^-2]")
    surv.add_argument("--dose", type=float, required=True, help="dose [Gy]")
    surv.set_defaults(func=_cmd_survival)

    table = sub.add_parser("table", help="build an alpha/beta table and write CSV")
    table.add_argument("--model", choices=sorted(_MODEL_BUILDERS), required=True)
    table.add_argument("--rn", type=float, required=True, help="nucleus radius [um]")
    # LEM I-IV photon reference (required for the lem* models).
    table.add_argument("--alpha-x", type=float, help="photon alpha_x [Gy^-1] (lem*)")
    table.add_argument("--beta-x", type=float, help="photon beta_x [Gy^-2] (lem*)")
    table.add_argument("--dt", type=float, help="transition dose D_t [Gy] (lem*)")
    # MKM parameters (required for --model mkm).
    table.add_argument("--alpha0", type=float, help="MKM alpha0 [Gy^-1] (mkm)")
    table.add_argument("--beta0", type=float, help="MKM beta0 [Gy^-2] (mkm)")
    table.add_argument("--rd", type=float, help="MKM domain radius [um] (mkm)")
    table.add_argument("--ions", required=True, help="comma-separated atomic numbers, e.g. 1,6")
    table.add_argument(
        "--mass-numbers",
        help="comma-separated mass numbers matching --ions, e.g. 1,13; default is the "
        "most abundant isotope of each element",
    )
    table.add_argument("--energies", required=True, help="comma-separated MeV/u, e.g. 10,100")
    table.add_argument(
        "--effective-size-factor",
        type=float,
        help="scale applied to the nucleus radius (default 1.0 -- a paper quoting an "
        "EFFECTIVE radius wants it left alone)",
    )
    table.add_argument(
        "--stopping-power",
        choices=list(STOPPING_POWER_CHOICES),
        help="LET source (default: the library default -- ICRU 90's procedure for Z 1-18, "
        "Hubert alpha scaling for Z 19-92 from 2.5 MeV/u, anchored analytic Bethe "
        "elsewhere; bethe = analytic Bethe alone; icru2016/2009/2005 = the "
        "ICRU 90/73/49 tables by edition; icru90-procedure; xstar = PSTAR/ASTAR/MSTAR; "
        "ns1970 = Northcliffe & Schilling 1970 + Bethe tail; libdedx needs the separately "
        "installed, GPL libdedx package)",
    )
    # Model-behaviour axes. These change the numbers without changing the call, so
    # the resolved values are written into the table's metadata header either way --
    # see docs/references/LEM_IV_OPEN_CHOICES.md for what each one costs.
    table.add_argument(
        "--beta-model",
        choices=list(dict.fromkeys((*get_args(LEM1BetaModel), *get_args(LEM4BetaModel)))),
        help="how beta_I is obtained (lem*; default slope_matching, the "
        "Kraemer-Scholz single-particle approximation)",
    )
    table.add_argument(
        "--initial-slope-model",
        choices=list(get_args(InitialSlopeModel)),
        help="alpha_I route (lem*; default cross_section, the D->0 limit). "
        "full_simulation is TRiP98's fitted protocol (lem4 only); the lethal_* "
        "routes are lem1-lem3 only",
    )
    table.add_argument(
        "--survival-convention",
        choices=list(get_args(LEM4SurvivalConvention)),
        help="LEM IV: S = <e^-E> (default, bounded) or S = e^-<E> (unsaturated, what "
        "TRiP98's own tables behave like)",
    )
    table.add_argument(
        "--cluster-index-model",
        choices=list(get_args(LEM4ClusterIndexModel)),
        help="LEM IV cluster-index estimator (default mean_field)",
    )
    table.add_argument(
        "--cluster-index-cap",
        type=float,
        help="LEM IV: clip C_I at this value (0.95 reproduces TRiP98 and every "
        "published-figure golden). Uncapped is the papers' own prescription but "
        "carries a spurious RBE maximum near 430 keV/um",
    )
    table.add_argument(
        "--local-dose-overkill",
        action="store_true",
        help="LEM IV: apply the B2 single-track overkill correction (measured to "
        "over-correct 1.9-5x; off by default)",
    )
    table.add_argument(
        "--engine",
        choices=list(ENGINES),
        help="numerical backend (default auto: numba when installed)",
    )
    table.add_argument("--out", required=True, help="output CSV path")
    table.set_defaults(func=_cmd_table)

    bench = sub.add_parser("benchmark", help="score a model against a measured-survival dataset")
    bench.add_argument(
        "--dataset",
        default="pide",
        choices=[*dataset_names()],
        help="reference dataset: pide (default), radphysbio, or csv (a user file in the "
        "shape documented in pyradbio.benchmark.dataset.load_reference_csv)",
    )
    bench.add_argument(
        "--path",
        default=None,
        help="dataset file or folder; defaults to the dataset's configured location "
        "(PYRADBIO_PIDE_DIR / PYRADBIO_RADPHYSBIO_DIR); required for --dataset csv",
    )
    bench.add_argument(
        "--pide",
        required=False,
        help="alias for '--dataset pide --path PATH' (kept for older scripts)",
    )
    bench.add_argument(
        "--model",
        default="lem1",
        choices=[*benchmark_models()],
        help="model to benchmark (LEM, MKM family, phenomenological family)",
    )
    bench.add_argument("--out", default="benchmark_results", help="output report directory")
    bench.add_argument(
        "--variant",
        default="fit_then_paper",
        choices=["fit", "paper", "fit_then_paper", "paper_then_fit"],
        help="which PIDE LQ columns to use (*_fit vs *_paper); PIDE only",
    )
    bench.add_argument("--limit", type=int, default=None, help="cap evaluated points")
    bench.add_argument("--no-plots", action="store_true", help="skip PNG plots (no matplotlib)")
    # Model knobs (LEM: dt/rn; generic mkm: alpha0/beta0/rd/rn). Defaults live in
    # the harness; the preset-based models refuse these rather than ignore them.
    bench.add_argument("--dt", type=float, default=None, help="LEM transition dose D_t [Gy]")
    bench.add_argument("--rn", type=float, default=None, help="nucleus radius [um]")
    bench.add_argument("--rd", type=float, default=None, help="MKM domain radius [um]")
    bench.add_argument("--alpha0", type=float, default=None, help="MKM alpha0 [Gy^-1]")
    bench.add_argument("--beta0", type=float, default=None, help="MKM beta0 [Gy^-2]")
    bench.set_defaults(func=_cmd_benchmark)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from pyradbio import __version__

        print(__version__)
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
