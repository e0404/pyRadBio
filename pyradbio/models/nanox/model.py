"""NanOx: NANodosimetry and OXidative stress.

Cunha, Monini, Testa & Beuve 2017, *Phys. Med. Biol.* **62**(4):1248-1268,
doi:10.1088/1361-6560/aa54c9 (the founding paper); Alcocer-Avila, Monini, Cunha,
Testa & Beuve 2023, *Front. Phys.* **11**:1011062, doi:10.3389/fphy.2023.1011062
(the general formalism); Alcocer-Avila et al. 2022, *Front. Phys.*
**10**:1011063, doi:10.3389/fphy.2022.1011063 (the hadrontherapy
implementation, whose equation numbering this module follows); Monini, Cunha,
Chollier, Testa & Beuve 2020, *Radiat. Res.* **193**(4):331,
doi:10.1667/RR15463.1 (determination of the ELLF); Monini, Cunha, Testa & Beuve
2018, *Cancers* **10**(4):87, doi:10.3390/cancers10040087 (parameter
sensitivity).

NanOx factorises cell survival into two independent classes of event
(2022 eq. 3):

```text
S = S_L x S_NL
```

**Local lethal events** happen in nanometric targets (20 x 10 nm cylinders),
each able to kill the cell on its own, and are treated by single-hit
single-target theory. Crucially the local dose is replaced by a *stochastic*
quantity -- the restricted specific energy ``z`` -- so no Poisson assumption is
made at the local scale, which is the model's principal departure from the LEM.
Local events alone give a **pure exponential** survival curve (2023 eq. 52).

**Non-local lethal events**, in the current version restricted to "global"
events, describe toxic accumulation -- oxidative stress from reactive chemical
species -- at the micrometric scale of the sensitive volume, and supply the
entire shoulder (2022 eq. 11).

**What this module implements, and what it does not.** Everything downstream of
the nanodosimetric spectrum: the ELLF, the effective number of local lethal
events, the core/penumbra assembly, the Poisson impact statistics, the global-
event branch, the photon calibration, and the track-segment closed forms. What
it does **not** implement is the spectrum itself. ``dP/dz|hit,c`` and
``RCE(ion, E)`` come from the group's non-public LQD/PHYCHEML/CHEM Monte Carlo
and are figure-only in all seven NanOx papers, so the forward ion route raises
[`NanOxNanodosimetryRequired`][] unless a
[`NanOxNanodosimetry`][] backend is supplied.
See ``docs/references/NanOx.md`` for the full boundary.

**Geometry trap.** ``d_s`` is a sensitive-volume **diameter** in Cunha 2017 and
Alcocer-Avila 2022; Monini 2018/2019/2020 print the same object as a **radius**
``R_SV``. They agree at a factor of two, and Cunha 2017's own
``sigma_s = 75 um^2`` for ``d_s = 9.78 um`` settles which is which.

**Parameter-set trap.** Cunha 2017 sec. 3.2.3 fits V79 at ``z0 = 20200 Gy,
sigma = 6000 Gy, h = 180000`` with ``beta_r = 0.0298``. Every later paper uses
``22789 / 8117 / 225841`` with ``beta_r = 0.0259``, and the 2017 numbers were
produced *without* the core/penumbra split (which does not appear in that
paper at all). The two sets are not interchangeable; only the later one ships.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, NoReturn

from pyradbio.core.photon import PhotonParameters
from pyradbio.core.radiation import Ion, RadiationField
from pyradbio.models.base import ModelCapabilities
from pyradbio.models.nanox.base import NanOxNanodosimetry, NanOxNanodosimetryRequired
from pyradbio.models.nanox.ellf import (
    DEFAULT_LOCAL_TARGET,
    EffectiveLocalLethalFunction,
    LocalTarget,
    LocalTargetSpectrum,
)

GY_PER_KEV_UM3: float = 0.1602
"""Unit conversion in Alcocer-Avila et al. 2022 eq. (35) and 2023 eq. (50):
``a = 0.1602 Gy keV^-1 um^3``, i.e. the dose in Gy deposited in 1 um^3 of
unit-density water by 1 keV. Turns a fluence-LET product into a dose.
"""

ETA_RESTRICTED_FRACTION: float = 0.8
"""The fraction of energy loss that NanOx counts as *restricted* specific
energy -- ionisations, excitations and electron attachments, but not the
energy that merely heats the medium (2023 Definition 2, ``<z> = eta D``).

Alcocer-Avila et al. 2023 Table 4 reports 0.798-0.802 from LQD across ions
and energies and says "eta is approximately equal to 0.8 for all practical
purposes". **0.8 exactly is the value that reproduces every published**
``beta_G``: 0.0298/0.64 -> 0.0466, 0.0259/0.64 -> 0.0405, 0.0400/0.64 ->
0.0625, 0.0615/0.64 -> 0.0961, all matching the printed figures.
"""

CORE_EDGE_UM: float = 0.2
"""Edge of the square cross-section of the track core (2022 sec. 2.1.2),
200 nm. Chosen so the density of events in the penumbra is comparable to
reference low-LET radiation, while staying far below the sensitive-volume
radius. A computational device, not a physical boundary -- and one absent
from the 2017 founding paper.
"""

T_RCE_S: float = 1.0e-11
"""The time after each impact at which primary radical yields are scored
(2022 sec. 2.1.2.2). Monini et al. 2018 showed the choice matters little:
moving it to 1e-8 s leaves ``alpha`` unchanged and shifts ``D_1%`` by
<~15%.
"""


@dataclass(frozen=True, slots=True)
class NanOxCellLine:
    """The five cell-line parameters a NanOx calculation needs.

    Monini et al. 2018 established that predictions over a wide LET range rest
    on just ``(d_s, h, z0, sigma, beta_G)`` when a single sensitive volume is
    used; everything else can be fixed on general grounds.
    """

    label: str
    sensitive_volume_diameter_um: float
    """Sensitive-volume **diameter** [um]. The cell nucleus, modelled as a
    cylinder oriented along the beam (2022 Simplification 2).
    """
    sensitive_volume_length_um: float
    """Sensitive-volume length [um]. 1.0 in every published set -- Monini et al.
    2018 calls it "the lowest reasonable value that can mimic the thickness
    of the nuclei of flattened cells".
    """
    ellf: EffectiveLocalLethalFunction
    beta_G_per_Gy2: float
    """Quadratic coefficient of the global-event response [Gy^-2] (2022 eq. 11)."""
    alpha_G_per_Gy: float = 0.0
    """Linear coefficient of the global-event response [Gy^-1]. Zero in every
    published NanOx calculation: ``C_L = 1`` (2023 eq. 34) routes the whole
    reference-radiation linear term to local events so that the local and
    global branches can be adjusted independently.
    """
    reference_beta_x_per_Gy2: float | None = None
    """The photon ``beta_r`` [Gy^-2] this line's ``beta_G`` was derived from,
    recorded for provenance. ``beta_G = beta_r / eta^2``.
    """

    def __post_init__(self) -> None:
        if self.sensitive_volume_diameter_um <= 0.0:
            raise ValueError(
                f"sensitive_volume_diameter_um must be > 0, "
                f"got {self.sensitive_volume_diameter_um!r}"
            )
        if self.sensitive_volume_length_um <= 0.0:
            raise ValueError(
                f"sensitive_volume_length_um must be > 0, got {self.sensitive_volume_length_um!r}"
            )
        if self.beta_G_per_Gy2 < 0.0:
            raise ValueError(f"beta_G_per_Gy2 must be >= 0, got {self.beta_G_per_Gy2!r}")
        if self.alpha_G_per_Gy < 0.0:
            raise ValueError(f"alpha_G_per_Gy must be >= 0, got {self.alpha_G_per_Gy!r}")


def _cell_line(
    label: str,
    *,
    diameter_um: float,
    z0_Gy: float,
    sigma_Gy: float,
    h: float,
    beta_G: float,
    beta_x: float,
) -> NanOxCellLine:
    return NanOxCellLine(
        label=label,
        sensitive_volume_diameter_um=diameter_um,
        sensitive_volume_length_um=1.0,
        ellf=EffectiveLocalLethalFunction(h=h, z0_Gy=z0_Gy, sigma_Gy=sigma_Gy),
        beta_G_per_Gy2=beta_G,
        reference_beta_x_per_Gy2=beta_x,
    )


V79 = _cell_line(
    "V79", diameter_um=9.8, z0_Gy=22789.0, sigma_Gy=8117.0, h=225841.0, beta_G=0.0405, beta_x=0.0259
)
"""Chinese hamster lung fibroblasts. Monini et al. 2020 Table 1 (ELF columns),
identical in Monini 2018 Table 1, Monini 2019 Table 1 and Alcocer-Avila 2022
Table 1. ``beta_r = 0.0259 Gy^-2`` from Monini 2018 Table 5, which cites it
to Cox, Thacker, Goodhead, Masson & Wilkinson 1977, *Int. J. Radiat. Biol.*
**31**:561, doi:10.1080/09553007714550661. The paired ``alpha_r`` is printed
in no NanOx paper, which is why the presets carry no ``PhotonParameters``.
"""

CHO_K1 = _cell_line(
    "CHO-K1",
    diameter_um=11.8,
    z0_Gy=14507.0,
    sigma_Gy=2781.0,
    h=104810.0,
    beta_G=0.0625,
    beta_x=0.0400,
)
"""Chinese hamster ovary cells. ``beta_r`` is printed by Monini 2018 Table 5 as
"0.0200 x 2" -- the paper deliberately doubles its literature source; carried
as printed, and 0.0400/0.64 does reproduce the printed ``beta_G``. The
undoubled 0.0200 is cited to Weyrather, Ritter, Scholz & Kraft 1999,
*Int. J. Radiat. Biol.* **75**(11):1357, doi:10.1080/095530099139232; the
doubling is Monini 2018's own correction "to reproduce the average value"
over publications, no single measurement matching the mean pair.
"""

HSG = _cell_line(
    "HSG",
    diameter_um=14.0,
    z0_Gy=15654.0,
    sigma_Gy=549.0,
    h=179439.0,
    beta_G=0.0961,
    beta_x=0.0615,
)
"""Human salivary gland tumour cells. ``beta_r = 0.0615 Gy^-2`` (Monini 2018
Table 5, citing Furusawa et al. 2000, *Radiat. Res.* **154**:485,
doi:10.1667/0033-7587(2000)154[0485:IOAAHC]2.0.CO;2), also quoted by
Alcocer-Avila 2023 sec. 3.4.
"""


@dataclass(frozen=True, slots=True)
class NanOxModel:
    """NanOx assembly and calibration layer.

    See the module docstring for the scope boundary. In short: every closed form
    the papers print is here; the nanodosimetric spectrum they integrate is not,
    and must be injected via ``nanodosimetry``.

    **Not a** [`BiophysicalModel`][pyradbio.models.base.BiophysicalModel]. That interface
    promises ``alpha_beta``, survival and RBE for an ion, and NanOx can deliver
    none of them until the configuration average of 2022 eqs. (36)-(43) has a
    nanodosimetric source, so this is a standalone assembly class: it cannot be
    handed to a table builder or the benchmark harness and fail there. It
    declares [`capabilities`][] like every family, with ``ion_route=False``,
    and keeps an [`alpha_beta`][] that raises with the reason. Re-parent it
    once an ion route exists.
    """

    cell_line: NanOxCellLine
    photon: PhotonParameters | None = None
    """Photon reference for this cell line. Required by anything that needs
    ``alpha_r,L`` (2023 eq. 35).
    """
    local_target: LocalTarget = DEFAULT_LOCAL_TARGET
    core_edge_um: float = CORE_EDGE_UM
    eta: float = ETA_RESTRICTED_FRACTION
    t_rce_s: float = T_RCE_S
    nanodosimetry: NanOxNanodosimetry | None = field(default=None)
    """Supplies ``dP/dz`` and ``RCE``. ``None`` -- the default -- means the
    forward ion route is unavailable and will raise.
    """

    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities(
        ion_route=False, mixed_field=False, lq_exact=False
    )
    """Advertised so that callers can branch without provoking an exception.
    No ion route while no public nanodosimetric source exists for the local
    targets; survival is the non-LQ product ``S_L x S_NL``.
    """

    def __post_init__(self) -> None:
        if self.core_edge_um <= 0.0:
            raise ValueError(f"core_edge_um must be > 0, got {self.core_edge_um!r}")
        if not 0.0 < self.eta <= 1.0:
            raise ValueError(f"eta must be in (0, 1], got {self.eta!r}")
        if self.core_edge_um >= self.cell_line.sensitive_volume_diameter_um:
            raise ValueError(
                f"core_edge_um {self.core_edge_um!r} must be smaller than the "
                f"sensitive-volume diameter {self.cell_line.sensitive_volume_diameter_um!r}"
            )

    # -- geometry (2022 eqs. 14-17) ----------------------------------------

    @property
    def sensitive_volume_diameter_um(self) -> float:
        return self.cell_line.sensitive_volume_diameter_um

    def geometric_cross_section_um2(self) -> float:
        """``sigma_s``, the nucleus cross-section seen by a parallel beam [um^2]."""
        return math.pi * (0.5 * self.sensitive_volume_diameter_um) ** 2

    def sensitive_volume_um3(self) -> float:
        """``V_s = sigma_s L_s`` [um^3]."""
        return self.geometric_cross_section_um2() * self.cell_line.sensitive_volume_length_um

    def core_volume_um3(self) -> float:
        """``V_c = L_s e_c^2`` [um^3] -- eq. (14), a square parallelepiped."""
        return self.cell_line.sensitive_volume_length_um * self.core_edge_um**2

    def core_fraction(self) -> float:
        """``r_c = V_c / V_s`` -- eq. (16)."""
        return self.core_volume_um3() / self.sensitive_volume_um3()

    def penumbra_fraction(self) -> float:
        """``r_p = V_p / V_s = 1 - r_c`` -- eq. (16)."""
        return 1.0 - self.core_fraction()

    # -- calibration against the reference radiation (2023 eqs. 32-35) ------

    @staticmethod
    def beta_G_from_photon(beta_x_per_Gy2: float, *, eta: float = ETA_RESTRICTED_FRACTION) -> float:
        """``beta_G = beta_r / eta^2`` -- Monini et al. 2017 (AppPolB) eq. (3.2).

        Also Alcocer-Avila 2023 eqs. (33) + (35): with ``C_L = 1`` the whole
        quadratic term of the reference-radiation response is carried by global
        events, and the microscopic coefficient differs from the macroscopic one
        only by the restricted-energy fraction.

        This is the corpus's one closed-form coefficient chain, and it
        reproduces every published ``beta_G`` (see the module constant
        [`ETA_RESTRICTED_FRACTION`][]).
        """
        if beta_x_per_Gy2 < 0.0:
            raise ValueError(f"beta_x_per_Gy2 must be >= 0, got {beta_x_per_Gy2!r}")
        return beta_x_per_Gy2 / eta**2

    def alpha_reference_local(self) -> float:
        """``alpha_r,L = alpha_r / eta`` [Gy^-1] -- 2023 eqs. (33) + (35).

        The linear coefficient for local lethal events under the reference
        radiation. It also stands in for the *penumbra* coefficient of any ion
        (2022 Approximation 2): the penumbra is made of fast delta electrons, so
        its nanometric spectra resemble those of photons.
        """
        if self.photon is None:
            raise ValueError(
                "alpha_reference_local() needs a photon reference; "
                "construct the model with photon=PhotonParameters(...)"
            )
        return self.photon.alpha_x / self.eta

    # -- local lethal events (2023 eqs. 42-43; 2022 eq. 26) ----------------

    def effective_local_lethal_events(self, spectrum: LocalTargetSpectrum) -> float:
        """``n* = int (dP/dz) F(z) dz`` -- 2023 eq. (42).

        The *effective* number of local lethal events, i.e. ``-ln(1-f)`` summed
        over targets, so that ``S_L = exp(-n*)`` exactly (eq. 40) with no Poisson
        assumption anywhere.
        """
        ellf = self.cell_line.ellf
        return float(
            sum(p * ellf(z) for z, p in zip(spectrum.z_Gy, spectrum.probability, strict=True))
        )

    def alpha_core(self, spectrum: LocalTargetSpectrum) -> float:
        """``alpha_c = <F(z)> / <z>_hit`` [Gy^-1] -- 2022 eq. (26).

        The proportionality coefficient linking the effective number of local
        lethal events in the track core to the restricted specific energy
        deposited there. Both the spectrum and its mean must describe *hit*
        targets in the core volume.
        """
        return self.effective_local_lethal_events(spectrum) / spectrum.mean_hit_z_Gy

    def local_lethal_events(
        self,
        *,
        alpha_core_per_Gy: float,
        core_specific_energy_Gy: float,
        penumbra_specific_energy_Gy: float,
    ) -> float:
        """``n* = alpha_c r_c Z_c + alpha_r,L r_p Z_p`` -- 2022 eq. (27).

        The core/penumbra assembly: the core carries its own, radiation-quality
        specific coefficient, while the penumbra is treated as reference
        radiation (Approximation 2).
        """
        return (
            alpha_core_per_Gy * self.core_fraction() * core_specific_energy_Gy
            + self.alpha_reference_local() * self.penumbra_fraction() * penumbra_specific_energy_Gy
        )

    def survival_local_from_events(self, n_star: float) -> float:
        """``S_L = exp(-n*)`` -- 2023 eq. (40)."""
        return math.exp(-n_star)

    # -- global events (2022 eqs. 9-11) ------------------------------------

    def chemical_specific_energy_Gy(self, z_Gy: float, rce: float) -> float:
        """``Ztilde = RCE * Z`` [Gy] -- eq. (9).

        ``RCE`` is the ratio of this radiation's radical yield to the reference
        radiation's (eq. 10), so it is 1 for the reference by definition.
        """
        if rce < 0.0:
            raise ValueError(f"rce must be >= 0, got {rce!r}")
        return rce * z_Gy

    def survival_global(self, z_tilde_Gy: float) -> float:
        """``S_G = exp(-alpha_G Ztilde - beta_G Ztilde^2)`` -- eq. (11).

        With ``alpha_G = 0`` this is the entire source of the shoulder in a
        NanOx survival curve.
        """
        return math.exp(
            -self.cell_line.alpha_G_per_Gy * z_tilde_Gy
            - self.cell_line.beta_G_per_Gy2 * z_tilde_Gy**2
        )

    # -- impact statistics (2022 eqs. 34-35) -------------------------------

    def mean_impacts(
        self, *, dose_Gy: float, let_keV_um: float, surface_um2: float
    ) -> tuple[float, float]:
        """``(Kin, Kout)``, the mean impacts inside and outside the nucleus.

        Eq. (35):

        ```text
        Kin  = D sigma_s / (a LET)
        Kout = D (Sigma - sigma_s) / (a LET)
        ```

        Each is Poisson-distributed and the two are independent (eq. 34), which
        is what lets the core and penumbra contributions be averaged separately.
        ``Sigma`` is the surface of influence: large enough that a track outside
        it cannot deposit energy in the nucleus.
        """
        if dose_Gy < 0.0:
            raise ValueError(f"dose_Gy must be >= 0, got {dose_Gy!r}")
        if let_keV_um <= 0.0:
            raise ValueError(f"let_keV_um must be > 0, got {let_keV_um!r}")
        sigma_s = self.geometric_cross_section_um2()
        if surface_um2 < sigma_s:
            raise ValueError(
                f"surface_um2 {surface_um2!r} must be at least the nucleus cross "
                f"section {sigma_s:.4g} um^2"
            )
        per_area = dose_Gy / (GY_PER_KEV_UM3 * let_keV_um)
        return per_area * sigma_s, per_area * (surface_um2 - sigma_s)

    # -- track-segment closed forms (2023 sec. 3.5) ------------------------

    def alpha_from_cross_section(self, zeta_um2: float, let_keV_um: float) -> float:
        """``alpha = zeta / (a LET)`` [Gy^-1] -- 2023 eq. (53).

        With only local lethal events the survival curve is a pure exponential
        whose slope is set by the inactivation cross section. This is the
        relation of Kanai et al., recovered by NanOx rather than assumed.
        """
        if let_keV_um <= 0.0:
            raise ValueError(f"let_keV_um must be > 0, got {let_keV_um!r}")
        if zeta_um2 < 0.0:
            raise ValueError(f"zeta_um2 must be >= 0, got {zeta_um2!r}")
        return zeta_um2 / (GY_PER_KEV_UM3 * let_keV_um)

    def inactivation_cross_section_um2(self, alpha_per_Gy: float, let_keV_um: float) -> float:
        """``zeta = alpha a LET`` [um^2] -- eq. (53) inverted.

        The direction that turns a published or measured ``alpha`` into NanOx's
        own inactivation cross section, which is how Alcocer-Avila et al. 2022
        eq. (45) builds its representative dataset. Note it needs no surface of
        influence.
        """
        if alpha_per_Gy < 0.0:
            raise ValueError(f"alpha_per_Gy must be >= 0, got {alpha_per_Gy!r}")
        if let_keV_um <= 0.0:
            raise ValueError(f"let_keV_um must be > 0, got {let_keV_um!r}")
        return alpha_per_Gy * GY_PER_KEV_UM3 * let_keV_um

    def single_impact_survival(self, *, zeta_um2: float, surface_um2: float) -> float:
        """``S_1 = 1 - zeta/Sigma`` -- 2023 eq. (53) rearranged.

        The mean survival to a single impact somewhere in the surface of
        influence. Unlike ``zeta`` this *does* depend on ``Sigma``: a bigger
        surface means more of the impacts miss.
        """
        if surface_um2 <= 0.0:
            raise ValueError(f"surface_um2 must be > 0, got {surface_um2!r}")
        if not 0.0 <= zeta_um2 <= surface_um2:
            raise ValueError(f"zeta_um2 {zeta_um2!r} must lie in [0, {surface_um2!r}]")
        return 1.0 - zeta_um2 / surface_um2

    def inactivation_cross_section_from_single_impact(
        self, single_impact_survival: float, *, surface_um2: float
    ) -> float:
        """``zeta = Sigma (1 - S_1)`` -- 2023 eq. (54).

        Independent of ``Sigma`` (eqs. 55-57): enlarging the surface adds only
        impacts that cannot reach the nucleus, so ``S_1`` rises by exactly the
        amount that keeps the product fixed.
        """
        if surface_um2 <= 0.0:
            raise ValueError(f"surface_um2 must be > 0, got {surface_um2!r}")
        if not 0.0 <= single_impact_survival <= 1.0:
            raise ValueError(
                f"single_impact_survival must be in [0, 1], got {single_impact_survival!r}"
            )
        return surface_um2 * (1.0 - single_impact_survival)

    def survival_local(self, *, dose_Gy: float, zeta_um2: float, let_keV_um: float) -> float:
        """``S(D) = exp(-zeta D / (a LET))`` -- 2023 eq. (51).

        Local lethal events only. The Poisson average over the number of impacts
        (eq. 49) collapses to a pure exponential, which is precisely why NanOx
        needs non-local events to produce a shoulder at all.
        """
        if dose_Gy < 0.0:
            raise ValueError(f"dose_Gy must be >= 0, got {dose_Gy!r}")
        return math.exp(-self.alpha_from_cross_section(zeta_um2, let_keV_um) * dose_Gy)

    # -- mean-field track-segment survival (2023 eqs. 24-33) ---------------

    def survival_track_segment(
        self, *, dose_Gy: float, alpha_local_per_Gy: float, rce: float = 1.0
    ) -> float:
        """``S(D) = exp(-alpha_L D) x S_G(RCE eta D)`` for a track-segment field.

        The full model averages the two survival factors over the distribution
        of specific energy in the sensitive volume (2023 eq. 24). This method
        evaluates the **mean-field limit** the paper itself takes in eq. (33),
        valid when ``sigma_Z << <Z> = eta D`` -- i.e. for doses above a few cGy
        with many tracks. It is exact for the reference radiation by
        construction (see [`photon_survival`][]) and approximate for ions,
        where the configuration average genuinely matters; Alcocer-Avila et al.
        2023 sec. 3.4 shows ``beta`` does not follow ``beta_G RCE^2`` once that
        averaging is done properly.

        ``alpha_local_per_Gy`` is the local-event slope per unit *macroscopic*
        dose -- eq. (53)'s ``zeta/(a LET)``, or ``alpha_x`` for the reference
        radiation.
        """
        if dose_Gy < 0.0:
            raise ValueError(f"dose_Gy must be >= 0, got {dose_Gy!r}")
        if alpha_local_per_Gy < 0.0:
            raise ValueError(f"alpha_local_per_Gy must be >= 0, got {alpha_local_per_Gy!r}")
        if rce < 0.0:
            raise ValueError(f"rce must be >= 0, got {rce!r}")
        z_tilde = rce * self.eta * dose_Gy
        return math.exp(-alpha_local_per_Gy * dose_Gy) * self.survival_global(z_tilde)

    def photon_survival(self, dose_Gy: float) -> float:
        """Survival under the reference radiation. Reproduces the LQ input exactly.

        Both factors of ``eta`` cancel: ``alpha_r,L = alpha_x/eta`` acts on
        ``<Z> = eta D``, and ``beta_G = beta_x/eta^2`` acts on
        ``Ztilde = eta D`` since ``RCE = 1`` by definition for the reference.
        So this returns ``exp(-alpha_x D - beta_x D^2)``, which is the
        consistency condition the calibration was derived from (2023 eqs.
        27-35) rather than an independent prediction.
        """
        if self.photon is None:
            raise ValueError(
                "photon_survival() needs a photon reference; "
                "construct the model with photon=PhotonParameters(...)"
            )
        return self.survival_track_segment(
            dose_Gy=dose_Gy, alpha_local_per_Gy=self.photon.alpha_x, rce=1.0
        )

    def with_beta_G_from_photon(self) -> NanOxModel:
        """A copy whose ``beta_G`` is *derived* from the photon reference.

        The shipped presets carry the ``beta_G`` the papers print, rounded to
        three significant figures. That rounding costs ~0.08% against
        ``beta_r/eta^2``, which is invisible physically but stops
        [`photon_survival`][] from closing to machine precision. Use this
        when the exact self-consistency matters, or when supplying a photon
        reference other than the one a preset was fitted with.
        """
        if self.photon is None:
            raise ValueError(
                "with_beta_G_from_photon() needs a photon reference; "
                "construct the model with photon=PhotonParameters(...)"
            )
        derived = replace(
            self.cell_line,
            beta_G_per_Gy2=self.beta_G_from_photon(self.photon.beta_x, eta=self.eta),
            reference_beta_x_per_Gy2=self.photon.beta_x,
        )
        return replace(self, cell_line=derived)

    def effective_beta_per_Gy2(self, rce: float) -> float:
        """``beta = beta_G (RCE eta)^2`` [Gy^-2], the mean-field quadratic term.

        The whole of NanOx's LET dependence of ``beta`` runs through ``RCE``:
        dense tracks let primary radicals recombine, so ``G`` and hence ``RCE``
        fall as LET rises, and the shoulder flattens. Note the caveat on
        [`survival_track_segment`][] -- 2023 Fig. 6 shows the properly
        averaged ``beta/beta_r`` is *not* simply ``RCE^2``.
        """
        if rce < 0.0:
            raise ValueError(f"rce must be >= 0, got {rce!r}")
        return self.cell_line.beta_G_per_Gy2 * (rce * self.eta) ** 2

    # -- the (missing) ion route --------------------------------------------

    def alpha_beta(self, field: Ion | RadiationField) -> NoReturn:
        """Always raises [`NanOxNanodosimetryRequired`][].

        Kept as a signpost: the message says what is missing and which assembly
        methods work instead. See `_raise_ion_route_missing`.
        """
        self._raise_ion_route_missing()

    def _raise_ion_route_missing(self) -> NoReturn:
        """``alpha_beta`` always raises -- see the module docstring.

        Raises [`NanOxNanodosimetryRequired`][].
        Without a backend the message names the two missing quantities; *with*
        one it names the remaining obstacle, the configuration average of 2022
        eqs. (36)-(43), and points at the assembly methods that do work. A
        backend is therefore necessary for the ion route but not sufficient.
        """
        if self.nanodosimetry is None:
            raise NanOxNanodosimetryRequired(
                "NanOx cannot predict alpha/beta for an ion without nanodosimetric "
                "input. Alcocer-Avila et al. 2022 eq. (26) integrates dP/dz, the "
                "single-event restricted specific-energy spectrum in the 20 x 10 nm "
                "local targets of the track core, and eq. (10) needs RCE, the "
                "relative chemical effectiveness. Both come from the LQD / PHYCHEML "
                "/ CHEM Monte-Carlo chain, which is not public and is shown only as "
                "figures in the NanOx papers, so this package ships no substitute -- "
                "an amorphous-track stand-in would reintroduce exactly the "
                "approximation NanOx was built to remove. A NanOxNanodosimetry "
                "backend supplies those two quantities, but alpha_beta() raises "
                "with one too: the averaging over irradiation configurations "
                "(eqs. 36-43) needs per-impact specific-energy distributions that "
                "are likewise figure-only, and is not implemented. With a backend, "
                "assemble a calculation explicitly from alpha_core(), "
                "local_lethal_events() and survival_global(). See "
                "docs/references/NanOx.md."
            )
        raise NanOxNanodosimetryRequired(
            "A NanOxNanodosimetry backend was supplied, but the averaging over "
            "irradiation configurations (Alcocer-Avila et al. 2022 eqs. 36-43) is "
            "not implemented in this release; use alpha_core(), "
            "local_lethal_events() and survival_global() to assemble a "
            "calculation explicitly."
        )

    # -- published parameter sets ------------------------------------------

    @classmethod
    def v79(cls, **kwargs: Any) -> NanOxModel:
        """V79 Chinese hamster lung fibroblasts (Monini et al. 2020 Table 1)."""
        return cls._preset(V79, **kwargs)

    @classmethod
    def cho_k1(cls, **kwargs: Any) -> NanOxModel:
        """CHO-K1 Chinese hamster ovary cells (Monini et al. 2020 Table 1)."""
        return cls._preset(CHO_K1, **kwargs)

    @classmethod
    def hsg(cls, **kwargs: Any) -> NanOxModel:
        """HSG human salivary gland tumour cells (Monini et al. 2020 Table 1)."""
        return cls._preset(HSG, **kwargs)

    @classmethod
    def _preset(cls, cell_line: NanOxCellLine, **kwargs: Any) -> NanOxModel:
        diameter = kwargs.pop("sensitive_volume_diameter_um", None)
        if diameter is not None:
            cell_line = replace(cell_line, sensitive_volume_diameter_um=float(diameter))
        return replace(cls(cell_line=cell_line), **kwargs)

    @classmethod
    def published_presets(cls) -> dict[str, Callable[[], NanOxModel]]:
        """The three cell lines with published NanOx parameter sets.

        Cunha et al. 2017's earlier V79 fit is deliberately absent -- see the
        module docstring's parameter-set trap.
        """
        return {"v79": cls.v79, "cho_k1": cls.cho_k1, "hsg": cls.hsg}
