"""Generate reference values from the ORIGINAL MSTAR 3.12 FORTRAN for the Python port.

Compiles the authors' sources (``data/stopping/mstar/MSTAR312/Sources/*.f``, without
their sample main program ``mstar.f``) together with the small driver below, and
runs MSTAR1 for liquid water (target 276), every ion Z 3-18, every mode a/b/c/d/g/h,
at every energy of MSTAR's own alpha grid (divided by 4, i.e. per nucleon) and at
the geometric midpoints between them (to exercise the spline between nodes).

Output (git-ignored, like all raw data): ``data/stopping/mstar/reference/
mstar_water_reference.txt``, one line per call ``Z1 MODE_IN MODE_USED E STE RC``
(E in MeV/u, STE in MeV cm^2/mg), read by ``tests/track/stopping/test_mstar.py``.

Needs a FORTRAN compiler (``--fc``, default ``gfortran``); the bundled DOS
executable is never used. Usage::

    python scripts/stopping/mstar_reference.py [--fc gfortran] [--raw data/stopping]
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "stopping"

#: Driver program (ours, not part of MSTAR): reads "Z1 MODE E" lines, calls MSTAR1.
DRIVER = """\
* Reference-value driver for the pyRadBio MSTAR port (not part of MSTAR).
      PROGRAM REFDRV
        IMPLICIT NONE
        INTEGER Z1,RC,ID
        CHARACTER CG,CGIN
        CHARACTER*80 NAME
        DOUBLE PRECISION E,STE
        ID=276
10      CONTINUE
          READ(5,*,END=20) Z1,CGIN,E
          CG=CGIN
          STE=0.0D0
          CALL MSTAR1(ID,CG,Z1,E,NAME,STE,RC)
          WRITE(6,'(I3,1X,A1,1X,A1,1X,ES24.16,1X,ES24.16,1X,I3)')
     *      Z1,CGIN,CG,E,STE,RC
        GOTO 10
20      STOP
      END
"""

MODES = "abcdgh"
IONS = range(3, 19)


def alpha_energies(database: Path) -> list[float]:
    """Section 1 of ``msdbs1.d``: the alpha energies [MeV]."""
    energies = []
    for line in database.read_text(encoding="latin-1").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "'" in s:
            break
        energies.append(float(s))
    return energies


def reference_energies(database: Path) -> list[float]:
    """Per-nucleon energies: the database grid / 4 and the geometric midpoints."""
    nodes = [e / 4.0 for e in alpha_energies(database)]
    mids = [math.sqrt(a * b) for a, b in pairwise(nodes)]
    return sorted(nodes + mids)


def generate(raw: Path, fc: str) -> Path:
    sources = raw / "mstar" / "MSTAR312" / "Sources"
    database = sources / "msdbs1.d"
    out_dir = raw / "mstar" / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        build = Path(tmp)
        for f in sources.glob("*.f"):
            if f.name.lower() != "mstar.f":
                shutil.copy(f, build / f.name)
        (build / "refdrv.f").write_text(DRIVER, encoding="ascii")
        exe = build / "refdrv.exe"
        subprocess.run(
            [fc, "-O0", "-std=legacy", "-o", str(exe), "refdrv.f", "mssubs.f"],
            cwd=build,
            check=True,
        )
        shutil.copy(database, build / "msdbs1.d")  # MSRDB1 opens it from the cwd
        lines = [
            f"{z} {m} {e!r}" for z in IONS for m in MODES for e in reference_energies(database)
        ]
        result = subprocess.run(
            [str(exe)],
            cwd=build,
            input="\n".join(lines) + "\n",
            capture_output=True,
            text=True,
            check=True,
        )
        version = subprocess.run([fc, "--version"], capture_output=True, text=True, check=True)
    target = out_dir / "mstar_water_reference.txt"
    header = [
        "# Reference values from the ORIGINAL MSTAR 3.12 (Schinner & Paul) FORTRAN,",
        "# compiled with scripts/stopping/mstar_reference.py; liquid water (ID 276).",
        f"# compiler: {version.stdout.splitlines()[0]}",
        "# columns: Z1 MODE_IN MODE_USED E[MeV/u] STE[MeV cm2/mg] RC",
    ]
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != len(lines):
        raise RuntimeError(f"expected {len(lines)} results, got {len(rows)}")
    target.write_text("\n".join(header + rows) + "\n", encoding="ascii")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fc", default="gfortran", help="FORTRAN compiler")
    parser.add_argument("--raw", type=Path, default=RAW)
    args = parser.parse_args(argv)
    target = generate(args.raw, args.fc)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
