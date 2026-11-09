"""Docstrings are Markdown for mkdocstrings, with no Sphinx/reStructuredText left in them.

The API reference renders docstrings as Markdown, where a Sphinx role, a ``#:``
attribute comment or a reST directive shows up as literal text instead of failing
the strict build. So they are kept out here instead (``AGENTS.md`` -> docstrings):

* cross-references are ``[`Name`][]`` (resolved in the docstring's scope) or
  ``[`Name`][pyradbio.full.path]``;
* an attribute is documented by a string literal right below it;
* notes and warnings are ``!!! note`` / ``!!! warning`` admonitions and code is
  fenced.

Roles and ``#:`` comments are checked in every tracked Python file; the
directive and literal-block rules only in the package, whose docstrings are the
ones rendered.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SOURCES = ("pyradbio", "tests", "examples", "validation", "scripts", "benchmark")

_SPHINX_ROLE = re.compile(r":(?:py:)?(?:class|meth|func|mod|attr|data|exc|obj|const):`")
_ATTRIBUTE_COMMENT = re.compile(r"^\s*#:")
_REST_DIRECTIVE = re.compile(r"^\s*\.\. [a-z-]+::")
_LITERAL_BLOCK = re.compile(r"(?<!:)::\s*$")


def _files(*folders: str) -> list[Path]:
    return sorted(p for folder in folders for p in (ROOT / folder).rglob("*.py"))


def _offending(pattern: re.Pattern[str], files: list[Path]) -> list[str]:
    hits = []
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line) and "http" not in line:
                hits.append(f"{path.relative_to(ROOT).as_posix()}:{number}: {line.strip()}")
    return hits


@pytest.mark.parametrize(
    ("pattern", "folders"),
    [
        pytest.param(_SPHINX_ROLE, _SOURCES, id="sphinx-roles"),
        pytest.param(_ATTRIBUTE_COMMENT, _SOURCES, id="attribute-comments"),
        pytest.param(_REST_DIRECTIVE, ("pyradbio",), id="rest-directives"),
        pytest.param(_LITERAL_BLOCK, ("pyradbio",), id="rest-literal-blocks"),
    ],
)
def test_no_sphinx_syntax(pattern: re.Pattern[str], folders: tuple[str, ...]) -> None:
    assert _files("pyradbio"), "no package sources found -- did the layout change?"
    assert _offending(pattern, _files(*folders)) == []
