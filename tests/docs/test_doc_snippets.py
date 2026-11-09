"""Every ``python`` code block in the README and the documentation runs.

The README and the user guide show short examples; this keeps them from rotting.
The blocks of one page run in order in one namespace, so a page can build on its
earlier blocks. A block fenced as ``py`` instead of ``python`` is shown but not run
(it needs an optional extra or external data). ``docs/references/`` is not part
of the site and is skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```python[^\n]*\n(?P<body>.*?)^(?P=indent)```",
    re.MULTILINE | re.DOTALL,
)


def _code_blocks(page: Path) -> list[str]:
    blocks = []
    for match in _FENCE.finditer(page.read_text(encoding="utf-8")):
        indent = match["indent"]
        lines = match["body"].splitlines(keepends=True)
        blocks.append("".join(line.removeprefix(indent) for line in lines))
    return blocks


_PAGES = [
    ROOT / "README.md",
    *sorted(
        page
        for page in DOCS.rglob("*.md")
        if "references" not in page.relative_to(DOCS).parts and _code_blocks(page)
    ),
]


def test_the_readme_has_code_blocks() -> None:
    assert _code_blocks(ROOT / "README.md"), "no python code blocks found -- did the pattern break?"


@pytest.mark.parametrize("page", _PAGES, ids=lambda page: page.relative_to(ROOT).as_posix())
def test_page_code_blocks_run(page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # blocks that write files write them here
    namespace: dict[str, object] = {"__name__": "__docs__"}
    for number, block in enumerate(_code_blocks(page), start=1):
        exec(compile(block, f"{page.name}, block {number}", "exec"), namespace)
