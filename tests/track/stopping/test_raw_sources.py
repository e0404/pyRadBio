"""The raw source files on disk are the ones each PROVENANCE.md describes.

PROVENANCE.md files are committed; the data they describe is not. Every file listed
in a PROVENANCE checksum table must match its recorded SHA-256, so the per-source
tests provably read the data whose origin, citation and notice are on record.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from tests.track.stopping.conftest import REPO_ROOT, stopping_data_root

_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")
_PROVENANCE = sorted((REPO_ROOT / "data" / "stopping").glob("*/PROVENANCE.md"))


def _checksums(provenance: Path) -> dict[str, str]:
    rows = (_ROW.match(line) for line in provenance.read_text(encoding="utf-8").splitlines())
    return {m.group(1): m.group(2) for m in rows if m}


def test_every_source_has_a_provenance_file_with_checksums() -> None:
    sources = {p.parent.name for p in _PROVENANCE}
    assert {
        "pstar",
        "astar",
        "icru90",
        "icru73",
        "mstar",
        "nist_isotopes",
        "estar",
        "pdg",
        "arb1973",
        "icru49",
        "ahlen1980",
        "brandt1982",
    } <= sources
    for provenance in _PROVENANCE:
        text = provenance.read_text(encoding="utf-8")
        assert _checksums(provenance), provenance
        for heading in ("**Origin.**", "**Retrieved.**", "**Citation.**", "**Notice"):
            assert heading in text, f"{provenance}: missing {heading}"


@pytest.mark.parametrize("provenance", _PROVENANCE, ids=lambda p: p.parent.name)
def test_raw_files_match_their_provenance_checksums(provenance: Path) -> None:
    folder = stopping_data_root() / provenance.parent.name
    expected = _checksums(provenance)
    present = {name: folder / name for name in expected if (folder / name).is_file()}
    if not present:
        pytest.skip(
            f"{folder.name}: raw data not fetched (python scripts/stopping/fetch_sources.py)"
        )
    assert present.keys() == expected.keys(), f"{folder.name}: partially fetched"
    for name, path in present.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[name], name
