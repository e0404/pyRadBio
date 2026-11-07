"""Locate the git-ignored raw stopping-power sources (data/stopping/<source>/).

Tests that read raw source values skip cleanly when the files are absent; fetch
them with ``python scripts/stopping/fetch_sources.py``. ``PYRADBIO_STOPPING_DATA``
overrides the default folder. The central accuracy checks have CI-runnable twins
against the shipped, regridded tables in ``test_shipped_anchors.py``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def stopping_data_root() -> Path:
    override = os.environ.get("PYRADBIO_STOPPING_DATA")
    return Path(override) if override else REPO_ROOT / "data" / "stopping"


@pytest.fixture
def raw_source_file() -> Callable[[str, str], Path]:
    """``raw_source_file(source, name)`` -> path, or skip if it has not been fetched."""

    def find(source: str, name: str) -> Path:
        path = stopping_data_root() / source / name
        if not path.is_file():
            pytest.skip(
                f"raw stopping-power source {source}/{name} not fetched; run "
                "python scripts/stopping/fetch_sources.py (data is not committed)"
            )
        return path

    return find
