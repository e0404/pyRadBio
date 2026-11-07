"""Phenomenological LET-based RBE models (the RBEmin/RBEmax family).

Closed-form parameterisations of ion RBE in the dose-averaged LET ``LET_d`` and
the tissue parameter ``(alpha/beta)_x``. They carry no track structure and no
microdosimetry: each is a fit to published in-vitro survival data, reduced to
two functions on the shared skeleton in :mod:`.base`.

    ==============  ========================  =========  ===================
    Class           Paper                     Ion        RBE_min
    ==============  ========================  =========  ===================
    WilkensOelfke   Wilkens & Oelfke 2004     protons    1 (fixed)
    Carabe          Carabe et al. 2012        protons    1.09 + s L / r
    LinearScaling   Frese et al. 2011         protons    1 (fixed)
    Wedenberg       Wedenberg et al. 2013     protons    1 (fixed)
    McNamara        McNamara et al. 2015      protons    p2 + p3 sqrt(r) L
    MairaniHelium   Mairani et al. 2016       helium     1 (fixed)
    ==============  ========================  =========  ===================

``WilkensOelfke`` and ``LinearScaling`` share one functional form -- ``alpha``
linear in LET, ``beta`` fixed at ``beta_x`` -- and differ only in how
``(alpha_0, lambda)`` are set: Wilkens fits both to in-vitro V79 data, Frese ties
``alpha_0`` to the tissue's own ``alpha_x`` through an entrance-RBE-of-1 anchor
and calibrates ``lambda`` so a plan averages the clinical RBE of 1.1.

**What these are for.** They are not competitors to the mechanistic models in
``models/lem`` and ``models/mkm`` -- they are the *baseline* those models are
worth judging against: they give LEM I-IV and the MKM family something to be
compared with besides each other.

**What a PIDE comparison does and does not show.** McNamara et al. 2015 is fitted
to all published in-vitro proton survival data, which is substantially what PIDE
aggregates. Agreement with PIDE therefore validates **our implementation of the
published parameterisation**, not the model's predictive correctness. Say so
wherever such a comparison is reported.

See ``docs/references/Phenomenological_RBE_models.md``.
"""

from pyradbio.models.phenomenological.base import RBEMinMax, RBEMinMaxModel
from pyradbio.models.phenomenological.carabe import Carabe
from pyradbio.models.phenomenological.chen import Chen
from pyradbio.models.phenomenological.frese import LinearScaling
from pyradbio.models.phenomenological.jones import Jones
from pyradbio.models.phenomenological.mairani import MairaniHelium
from pyradbio.models.phenomenological.mcnamara import McNamara
from pyradbio.models.phenomenological.peeler import Peeler
from pyradbio.models.phenomenological.rorvik import RorvikUnweighted
from pyradbio.models.phenomenological.tilly import Tilly2, Tilly10
from pyradbio.models.phenomenological.unkelbach import Unkelbach
from pyradbio.models.phenomenological.wedenberg import Wedenberg
from pyradbio.models.phenomenological.wilkens import WilkensOelfke

__all__ = [
    "Carabe",
    "Chen",
    "Jones",
    "LinearScaling",
    "MairaniHelium",
    "McNamara",
    "Peeler",
    "RBEMinMax",
    "RBEMinMaxModel",
    "RorvikUnweighted",
    "Tilly2",
    "Tilly10",
    "Unkelbach",
    "Wedenberg",
    "WilkensOelfke",
]
