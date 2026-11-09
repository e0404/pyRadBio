"""Named stopping-power choices for the LEM I paper reconstructions.

Scholz & Kraft 1996 (doi:10.1016/0273-1177(95)00784-C) and Scholz et al. 1997
(doi:10.1007/s004110050055) normalise the LEM radial dose to "the tables of
Hubert et al. and Northcliffe and Schilling". The validation scripts that
reconstruct them (``validation/scholz_kraft1996_lem1.py``,
``validation/scholz1997_lem1.py``) and the offline Monte-Carlo generator that
writes their checkpoints (``scripts/lem1_mc_reference.py``) select the LET
source by a short CLI/environment string. This module is the one map from that
string to a stopping-power object, so the plotted curves and the checkpoint rows
cannot be computed at different LETs without the mismatch being visible.

Every object returned by [`select_stopping_power`][] carries a ``name``
attribute. That is the value the generators stamp into the ``stopping_power``
column of a checkpoint and the readers compare against, which is why plain
analytic Bethe is returned as the *named*
[`BetheOnly`][pyradbio.track.northcliffe_schilling.BetheOnly] rather than an unnamed
[`SimpleBetheStoppingPower`][pyradbio.track.stopping_power.SimpleBetheStoppingPower].

Choices:

``"ns1970"``
    [`NorthcliffeSchillingWithBetheTail`][pyradbio.track.northcliffe_schilling.NorthcliffeSchillingWithBetheTail]
    (``name = "ns1970+bethe"``): the paper's own water table inside its
    0.0125-12 MeV/u grid, analytic Bethe above it, with the seam counted.
``"bethe"``
    [`BetheOnly`][pyradbio.track.northcliffe_schilling.BetheOnly] (``name = "bethe"``).
``"pymkm[:SOURCE]"``
    pyMKM's tabulated LET (optional ``mkm`` extra; ``SOURCE`` defaults to
    ``geant4_11_3_0``), named ``"pymkm:<SOURCE>"``. Imported lazily, so this
    module never imports pyMKM.
``"libdedx[:PROGRAM]"``
    libdEdx (optional GPL package; ``PROGRAM`` defaults to ``DEFAULT``), named
    ``"libdedx:<PROGRAM>"``.

The labels are stamped into the figure titles.
"""

from __future__ import annotations

from pyradbio.track.northcliffe_schilling import ENERGY_RANGE_MEV_U, build_stopping_power
from pyradbio.track.stopping_power import LibdedxStoppingPower, StoppingPower

__all__ = [
    "STOPPING_POWER_CHOICES",
    "select_stopping_power",
    "stopping_power_label",
]

STOPPING_POWER_CHOICES = "ns1970|bethe|pymkm[:SOURCE]|libdedx[:PROGRAM]"
"""The accepted forms of a stopping-power choice, for help and error texts."""

_DEFAULT_PYMKM_SOURCE = "geant4_11_3_0"
_DEFAULT_LIBDEDX_PROGRAM = "DEFAULT"


def _option(choice: str, default: str) -> str:
    return choice.split(":", 1)[1] if ":" in choice else default


def _check(choice: str) -> None:
    if choice in ("ns1970", "bethe"):
        return
    for prefix in ("pymkm", "libdedx"):
        if choice == prefix or choice.startswith(f"{prefix}:"):
            return
    raise ValueError(f"unknown LEM1_STOPPING={choice!r}; use {STOPPING_POWER_CHOICES}")


def select_stopping_power(choice: str) -> StoppingPower:
    """The stopping power for ``choice``; the returned object always has ``.name``.

    Parameters
    ----------
    choice:
        One of [`STOPPING_POWER_CHOICES`][].

    Returns
    -------
    StoppingPower
        A fresh instance (the ``ns1970`` hybrid keeps per-instance fallback tallies,
        so callers that report the tally must not share one).

    Raises
    ------
    ValueError
        For an unknown choice.
    ImportError
        When the optional package behind ``pymkm`` or ``libdedx`` is not installed.
    """
    _check(choice)
    if choice in ("ns1970", "bethe"):
        return build_stopping_power(choice)
    if choice.startswith("pymkm"):
        from pyradbio.models.mkm.pymkm_backend import PyMKMStoppingPower

        source = _option(choice, _DEFAULT_PYMKM_SOURCE)
        table = PyMKMStoppingPower(source=source)  # type: ignore[arg-type]
        setattr(table, "name", f"pymkm:{source}")  # noqa: B010 - not a declared attribute
        return table
    program = _option(choice, _DEFAULT_LIBDEDX_PROGRAM)
    libdedx = LibdedxStoppingPower(program=program)
    setattr(libdedx, "name", f"libdedx:{program}")  # noqa: B010 - not a declared attribute
    return libdedx


def stopping_power_label(choice: str) -> str:
    """The short human-readable label the LEM I figures stamp into their titles.

    Kept short on purpose: it is printed in every figure title, and the long
    form overran the axes.

    Raises
    ------
    ValueError
        For an unknown choice.
    """
    _check(choice)
    if choice == "ns1970":
        return f"Northcliffe & Schilling 1970 + Bethe above {ENERGY_RANGE_MEV_U[1]:g} MeV/u"
    if choice == "bethe":
        return "analytic Bethe (paper-era proxy)"
    if choice.startswith("pymkm"):
        return f"pyMKM {_option(choice, _DEFAULT_PYMKM_SOURCE)} (tabulated)"
    return f"libdEdx {_option(choice, _DEFAULT_LIBDEDX_PROGRAM)} (updated)"
