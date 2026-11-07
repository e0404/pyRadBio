"""Shared pytest fixtures and PIDE-path option."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from pyradbio.core.radiation import Ion


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--pide",
        action="store",
        default=None,
        help="PIDE file or folder; defaults to data/pide/. Enables @benchmark tests.",
    )


@pytest.fixture
def pide_path(request: pytest.FixtureRequest) -> str:
    """Resolve a PIDE file: explicit --pide wins, else the default data/pide/ folder.

    Skips (rather than fails) when no PIDE file is available, so CI without the
    non-redistributable data stays green.
    """
    from pyradbio.benchmark.pide import find_pide_file

    explicit = request.config.getoption("--pide")
    try:
        return str(find_pide_file(explicit))
    except FileNotFoundError:
        pytest.skip(
            "PIDE benchmark tests require a PIDE file: pass --pide <path> or place "
            "one in data/pide/ (override with PYRADBIO_PIDE_DIR). Data not bundled."
        )


def _have_pymkm() -> bool:
    try:
        import pymkm  # noqa: F401
    except ImportError:
        return False
    return True


_PYMKM_SKIP_REASON = "pyMKM not installed ([mkm] extra)"
requires_pymkm = pytest.mark.skipif(not _have_pymkm(), reason=_PYMKM_SKIP_REASON)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tag every pyMKM-gated test with the ``requires_pymkm`` marker.

    pyMKM is an optional validation reference, so its tests are skip-gated (some
    modules define their own ``skipif`` with the same reason). The marker lets the
    dedicated CI job select exactly them: ``pytest -m requires_pymkm``.
    """
    del config
    for item in items:
        for mark in item.iter_markers("skipif"):
            if mark.kwargs.get("reason") == _PYMKM_SKIP_REASON:
                item.add_marker(pytest.mark.requires_pymkm)
                break


class LETFunction:
    """A :class:`~pyradbio.track.stopping_power.StoppingPower` from a plain ``ion -> LET``.

    Tests that pin a LET per ion (rather than exercise a real stopping power)
    wrap the lookup in this, since every model's ``stopping_power`` is typed by
    the protocol rather than as a bare callable.
    """

    def __init__(self, let_of: Callable[[Ion], float]) -> None:
        self._let_of = let_of

    def let_keV_um(self, ion: Ion) -> float:
        return float(self._let_of(ion))
