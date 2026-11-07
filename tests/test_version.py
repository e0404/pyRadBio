"""The package version has one source: ``pyproject.toml``."""

from __future__ import annotations

import re
from pathlib import Path

import pyradbio


def _pyproject_version() -> str:
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    project = text.split("[project]", 1)[1]
    match = re.search(r'^version\s*=\s*"([^"]+)"', project, flags=re.M)
    assert match is not None, "no version in [project]"
    return match.group(1)


def test_version_matches_pyproject() -> None:
    assert pyradbio.__version__ == _pyproject_version()
