"""Every shipped data file credits its source (THIRD_PARTY_NOTICES.md, "Data")."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "pyradbio"
DATA_FILES = sorted(PACKAGE.glob("**/data/*.csv"))

#: A DOI, a pointer to the source's PROVENANCE.md, or a table pyRadBio computed itself.
_ATTRIBUTION = re.compile(r"doi[:./ ]|10\.\d{4,}/|PROVENANCE\.md|computed by pyRadBio", re.I)


def test_data_files_are_found() -> None:
    assert len(DATA_FILES) > 100


@pytest.mark.parametrize("path", DATA_FILES, ids=lambda p: p.relative_to(PACKAGE).as_posix())
def test_data_file_header_credits_its_source(path: Path) -> None:
    with path.open(encoding="utf-8") as fh:
        head = "".join(line for _, line in zip(range(40), fh, strict=False))
    assert _ATTRIBUTION.search(head), f"{path.name}: header names no DOI or provenance"
