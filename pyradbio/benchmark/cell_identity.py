"""Shared cell-line identity for benchmark pairing and database cross-checks."""

from __future__ import annotations

import re
from collections.abc import Mapping

__all__ = ["CELL_LINE_ALIASES", "cell_key"]


# --- cell-line identity -------------------------------------------------------

#: Greek capitals (and mu) that RadPhysBio's export uses in place of the Latin
#: look-alikes (a Greek capital kappa in "TK-1", a Greek capital alpha in
#: "A-549", the "mu2" uveal melanoma line spelled with a small mu). Built from
#: code points so the source stays ASCII: U+0391 alpha, U+0392 beta, U+0395
#: epsilon, U+0396 zeta, U+0397 eta, U+0399 iota, U+039A kappa, U+039C mu,
#: U+039D nu, U+039F omicron, U+03A1 rho, U+03A4 tau, U+03A5 upsilon, U+03A7
#: chi, U+03BC small mu and U+00B5 micro sign.
_GREEK_LOOKALIKES: Mapping[int, str] = {
    0x0391: "A",
    0x0392: "B",
    0x0395: "E",
    0x0396: "Z",
    0x0397: "H",
    0x0399: "I",
    0x039A: "K",
    0x039C: "M",
    0x039D: "N",
    0x039F: "O",
    0x03A1: "P",
    0x03A4: "T",
    0x03A5: "Y",
    0x03A7: "X",
    0x03BC: "mu",
    0x00B5: "mu",
}

#: Normalised alias -> canonical normalised key. Keys are what :func:`cell_key`
#: produces *before* alias resolution (lower-case alphanumerics only), so
#: ``"HSGc-C5"`` -> ``"hsgcc5"`` -> ``"hsg"``. Sources of each alias:
#:
#: * ``hsgcc5``: HSGc-C5 is the JCRB designation of the HSG line.
#: * ``ag01522`` / ``ag1522b`` / ``ag1522d``: sub-strain (passage) suffixes of
#:   the AG1522 fibroblast line, used by PIDE where a paper gives them; RadPhysBio
#:   writes ``AG1522`` throughout.
#: * ``u87`` -> ``u87mg``: the same glioblastoma line (U-87 MG).
#: * ``at1ostn`` / ``susatn``: RadPhysBio writes ``AT1OS/T-n`` and ``SuSa/T-n``
#:   for Kitajima 2010's AT1OS and SuSa.
#: * ``hek293`` -> ``hek``: RadPhysBio writes ``HEK/293`` for HEK.
#: * ``apfre19`` -> ``arpe19``: a misspelling of ARPE-19.
#: * ``tg98g`` -> ``t98g``: a misspelling of T98G.
#:
#: ``T1``/``T-1``, ``A549``/``A-549``, ``HeLa S3``/``HeLaS3``, ``LC-1 sq``/
#: ``LC-1sq`` and ``U-251MG(KO)``/``U-251MGKO`` collapse by normalisation alone
#: and need no alias.
CELL_LINE_ALIASES: Mapping[str, str] = {
    "hsgcc5": "hsg",
    "ag01522": "ag1522",
    "ag1522b": "ag1522",
    "ag1522d": "ag1522",
    "ag01522b": "ag1522",
    "u87": "u87mg",
    "at1ostn": "at1os",
    "susatn": "susa",
    "hek293": "hek",
    "apfre19": "arpe19",
    "tg98g": "t98g",
}


def cell_key(name: str) -> str:
    """Case- and punctuation-insensitive cell-line identity with alias folding.

    Greek look-alike capitals are mapped to Latin, everything but ``[a-z0-9]``
    is dropped, and :data:`CELL_LINE_ALIASES` is applied once.
    """
    key = re.sub(r"[^a-z0-9]", "", name.translate(_GREEK_LOOKALIKES).lower())
    return CELL_LINE_ALIASES.get(key, key)
