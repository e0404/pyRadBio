"""Keep local restricted datasets outside version control (synthetic paths only)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_pide_folder_is_ignored_except_for_its_readme(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is needed to verify repository ignore rules")
    root = Path(__file__).resolve().parents[1]
    (tmp_path / ".gitignore").write_bytes((root / ".gitignore").read_bytes())
    subprocess.run([git, "init", "--quiet", str(tmp_path)], check=True, capture_output=True)
    paths = [
        "data/pide/PIDE3.4.xlsx",
        "data/pide/pide_publications_pmid.csv",
        "data/pide/IonRawData.dat",
        "data/pide/reports/crosscheck.md",
        "data/pide/README.md",
        "pyradbio/benchmark/data/pide_publications_pmid.csv",
    ]
    result = subprocess.run(
        [git, "check-ignore", "--no-index", "--stdin", "-z"],
        cwd=tmp_path,
        input="\0".join(paths) + "\0",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    ignored = set(paths) - {"data/pide/README.md"}
    assert set(result.stdout.rstrip("\0").split("\0")) == ignored


def test_raw_stopping_sources_are_ignored_except_provenance(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is needed to verify repository ignore rules")
    root = Path(__file__).resolve().parents[1]
    (tmp_path / ".gitignore").write_bytes((root / ".gitignore").read_bytes())
    subprocess.run([git, "init", "--quiet", str(tmp_path)], check=True, capture_output=True)
    ignored = [
        "data/stopping/pstar/pstar_water_liquid.html",
        "data/stopping/pstar/pstar_water_liquid.csv",
        "data/stopping/pstar/FETCHED.json",
        "data/stopping/icru90/nist_icru90_update.pdf",
        "data/stopping/mstar/MSTAR312.zip",
        "data/stopping/mstar/MSTAR312/Sources/msdbs1.d",
        "data/stopping/icru73/icru73_2009_water_liquid.csv",
        "data/stopping/icru73/libdedx/ICRU73.dat",
    ]
    tracked = [
        "data/stopping/README.md",
        "data/stopping/pstar/PROVENANCE.md",
        "data/stopping/mstar/PROVENANCE.md",
        "data/stopping/icru73/README.md",
    ]
    result = subprocess.run(
        [git, "check-ignore", "--no-index", "--stdin", "-z"],
        cwd=tmp_path,
        input="\0".join(ignored + tracked) + "\0",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    assert set(result.stdout.rstrip("\0").split("\0")) == set(ignored)


def _checkout_root() -> tuple[str, Path]:
    """``(git, repository root)``, skipping when this is not a git checkout."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is needed to verify tracked files")
    root = Path(__file__).resolve().parents[1]
    probe = subprocess.run(
        [git, "rev-parse", "--show-toplevel"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or Path(probe.stdout.strip()).resolve() != root:
        pytest.skip("not running from a git checkout of pyRadBio")
    return git, root


def test_validation_checkpoints_are_tracked() -> None:
    """Every ``validation/data/*.csv`` (our own offline-MC checkpoints) is committed.

    The validation scripts read these files on a fresh clone; an untracked one
    would make a script work only on the machine that generated it.
    """
    git, root = _checkout_root()
    on_disk = sorted(p.relative_to(root).as_posix() for p in root.glob("validation/data/*.csv"))
    listed = subprocess.run(
        [git, "ls-files", "-z", "--", "validation/data"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tracked = set(filter(None, listed.split("\0")))
    assert [path for path in on_disk if path not in tracked] == []


def test_validation_figures_are_ignored_but_checkpoints_are_not() -> None:
    git, root = _checkout_root()
    paths = [
        "validation/figures/sk1997_fig1_alpha_vs_energy.png",
        "validation/figures/nested/cache.npz",
        "validation/data/mc_results.csv",
        "validation/_common.py",
    ]
    result = subprocess.run(
        [git, "check-ignore", "--no-index", "--stdin", "-z"],
        cwd=root,
        input="\0".join(paths) + "\0",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    assert set(filter(None, result.stdout.split("\0"))) == set(paths[:2])
