"""The execution-engine choice shared by every LEM compiled / Monte-Carlo path.

Defined once here (a leaf module, so the low-level kernels can use it without an
import cycle) and used by the local-dose Monte Carlo of LEM I-III, LEM IV's
DSB-pattern scoring, its intertrack Monte Carlo and the local-dose box average:

* ``"numpy"`` -- the vectorised, seed-reproducible reference and the accuracy anchor;
* ``"numba"`` -- the compiled kernel (the ``fast`` extra); errors if numba is missing;
* ``"auto"`` -- numba when importable (and applicable), else numpy.
"""

from __future__ import annotations

from typing import Literal, get_args

Engine = Literal["auto", "numpy", "numba"]

#: The allowed :data:`Engine` values, derived from the ``Literal``.
ENGINES: tuple[Engine, ...] = get_args(Engine)

__all__ = ["ENGINES", "Engine"]
