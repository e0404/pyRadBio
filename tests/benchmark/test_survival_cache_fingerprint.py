"""The committed Survival-comparison MC cache still matches the code.

``validation/data/survival_our_mc.json`` holds pyRadBio's own Monte-Carlo points for
``validation/survival_code_lem.py``. Its ``_provenance`` header records a numeric
fingerprint: three small points through the generator's own code path
(``manganaro2018.FINGERPRINT_POINTS``: one whole ``adaptive_montecarlo_point`` on
LEM I -- pilot, sizing and the near/far split's background draw -- and one
``montecarlo_lq_repeats`` each on LEM II and LEM III at the same ion, energy and
seeds, so the pair shares the cluster/eta machinery and differs only in the radial
dose; numpy engine, fixed seeds, analytic Bethe -- data-free, so this runs in CI in
a few seconds). A change to the LEM Monte Carlo, the radial dose, the sizing or the LQ fit
moves it, and the committed points must then be regenerated.

Known limits: the fingerprint runs on SimpleBethe, not the SRIM tables the entries
were made with (not redistributable; an SRIM change still invalidates every entry,
because the LET is part of each cache key and the validation script then reports
the points as missing).
"""

from __future__ import annotations

import json
from pathlib import Path

from pyradbio.benchmark import manganaro2018 as m18
from pyradbio.benchmark.checkpoint import (
    FINGERPRINT_KEY,
    PROVENANCE_KEY,
    assert_fingerprint,
    read_header,
)

CACHE = Path(__file__).resolve().parents[2] / "validation" / "data" / "survival_our_mc.json"
GENERATOR = "lem_mc_vs_survival"  # scripts/lem_mc_vs_survival.py --fresh


def test_cache_fingerprint_is_reproduced_by_the_code() -> None:
    recorded = read_header(CACHE).get(FINGERPRINT_KEY)
    assert_fingerprint(recorded, m18.mc_fingerprint(recorded or ()), generator=GENERATOR)


def test_cache_header_records_its_sources() -> None:
    header = read_header(CACHE)
    assert header["generator"] == f"scripts/{GENERATOR}.py"
    assert header["engine"] == "auto"
    assert "SRIM" in header["stopping_power"]
    assert header["code_version"].startswith("pyradbio ")
    assert [p["model"] for p in header[FINGERPRINT_KEY]] == ["LEMI", "LEMII", "LEMIII"]


def test_cache_entries_have_the_reader_fields() -> None:
    """Every entry carries what validation/survival_code_lem.py plots."""
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    data.pop(PROVENANCE_KEY)
    assert len(data) == 78
    fields = {"let", "alpha", "alpha_sigma", "beta", "beta_sigma", "n_real", "model", "Z", "A", "E"}
    for key, entry in data.items():
        assert len(key) == 16 and set(entry) == fields, key
