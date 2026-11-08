"""Reader for MSTAR's alpha-stopping database ``msdbs1.d`` (routine MSRDB1).

Python port of the MSTAR subroutine package by Andreas Schinner and Helmut Paul
(program system MORESTAR, V 3.12, 24-05-2004), freeware; the authors' names are
kept as they ask.

This file is under the MSTAR notice in ``THIRD_PARTY_NOTICES.md`` (AGENTS.md rule 3),
not pyRadBio's MIT licence.

Layout (from MSRDB1 and the file header): ``#`` comment lines; section 1 is the
list of alpha energies [MeV, total He energy]; section 2 has, per target, a line
``<ID> '<TAG>'`` followed by one electronic stopping power [MeV cm^2/mg] per
energy. The data are M.J. Berger's ASTAR values (NISTIR 4999, program ASTAR v2 of
1993, i.e. ICRU 49) extended above 1 GeV by the authors' program BEST.

The database is raw source data and is **not** shipped with pyRadBio: pass the
path of a local copy (``data/stopping/mstar/MSTAR312/Sources/msdbs1.d``, see that
folder's PROVENANCE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path


class MSTARDatabaseError(ValueError):
    """``msdbs1.d`` does not have the structure MSRDB1 accepts (its RC=15 checks)."""


@dataclass(frozen=True)
class AlphaTable:
    """One target's alpha electronic stopping power versus total alpha energy."""

    target_id: int
    name: str
    energy_MeV: tuple[float, ...]
    stopping_MeV_cm2_mg: tuple[float, ...]


@dataclass(frozen=True)
class MSTARDatabase:
    """All targets of ``msdbs1.d``; [`target`][] returns one [`AlphaTable`][]."""

    energy_MeV: tuple[float, ...]
    targets: dict[int, tuple[str, tuple[float, ...]]]

    @classmethod
    def from_file(cls, path: str | Path) -> MSTARDatabase:
        """Parse ``msdbs1.d`` with the checks MSRDB1 applies."""
        energies: list[float] = []
        targets: dict[int, tuple[str, list[float]]] = {}
        current: list[float] | None = None
        for raw in Path(path).read_text(encoding="latin-1").splitlines():
            line = raw.replace("\t", " ").strip()
            if not line or line.startswith("#"):
                continue
            if line.count("'") == 2:
                head, _, rest = line.partition("'")
                target_id = int(head)
                if target_id <= 0 or target_id in targets:
                    raise MSTARDatabaseError(f"bad target id {target_id}")
                if current is not None and len(current) != len(energies):
                    raise MSTARDatabaseError("targets have unequal numbers of values")
                current = []
                targets[target_id] = (rest.partition("'")[0].strip(), current)
            elif line.count("'") == 0 and line.count(".") == 1:
                value = float(line.replace("D", "E").replace("d", "e"))
                if value <= 0.0:
                    raise MSTARDatabaseError("non-positive entry")
                (energies if current is None else current).append(value)
            else:
                raise MSTARDatabaseError(f"unreadable line {raw!r}")
        if not targets or len(energies) < 4 or len(current or []) != len(energies):
            raise MSTARDatabaseError("incomplete database")
        if any(b <= a for a, b in pairwise(energies)):
            raise MSTARDatabaseError("energies are not strictly increasing")
        return cls(
            tuple(energies),
            {k: (name, tuple(values)) for k, (name, values) in targets.items()},
        )

    def target(self, target_id: int) -> AlphaTable:
        try:
            name, values = self.targets[target_id]
        except KeyError:
            raise KeyError(f"target ID {target_id} not in the MSTAR database") from None
        return AlphaTable(target_id, name, self.energy_MeV, values)


__all__ = ["AlphaTable", "MSTARDatabase", "MSTARDatabaseError"]
