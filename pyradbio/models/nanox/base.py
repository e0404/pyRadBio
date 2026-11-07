"""The NanOx nanodosimetry seam, and what happens when it is not supplied.

NanOx's forward prediction of ``alpha(LET)`` needs two quantities that the
published corpus shows only as figures and that come from the group's own,
non-public Monte-Carlo chain (**LQD** for transport, **PHYCHEML**/**CHEM** for
radiolysis):

* ``dP/dz|hit,c`` and ``<z>_hit,c`` -- the single-event restricted
  specific-energy spectrum in the 20 nm x 10 nm local targets inside the track
  core, integrated by Alcocer-Avila et al. 2022 eq. (26);
* ``RCE(ion, E) = G/G_r`` -- the relative chemical effectiveness at
  ``T_RCE = 1e-11 s``, eq. (10).

This module defines the protocol a caller may implement to supply them. This
package deliberately ships **no** implementation: the only track-structure
machinery here is amorphous (``pyradbio.track``) and micrometric
(``pyradbio.micro``), and NanOx exists specifically to reject amorphous,
mean-field local dose. Substituting one would make any comparison against a
published NanOx curve uninterpretable -- a disagreement could be the spectrum
or the assembly -- so the ion route refuses instead. The same treatment as the
track-width regime of :mod:`pyradbio.models.katz`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pyradbio.core.radiation import Ion
from pyradbio.models.nanox.ellf import LocalTargetSpectrum


class NanOxNanodosimetryRequired(NotImplementedError):
    """Raised when a NanOx route needs nanodosimetric input that is not present.

    Subclasses :class:`NotImplementedError` so that callers sweeping many models
    (for instance a benchmark harness) treat it as "this model does not cover
    this case" rather than as a bug.
    """


@runtime_checkable
class NanOxNanodosimetry(Protocol):
    """Supplies the two Monte-Carlo quantities NanOx cannot compute in closed form.

    Implement this against LQD, Geant4-DNA, or any track-structure code that can
    score energy deposits in nanometric targets. Both methods are per
    (ion type, energy) under track-segment conditions -- Alcocer-Avila et al.
    2022 Hypothesis 1 -- so the returned values may be cached on ``(Z, A, E)``.
    """

    def core_target_spectrum(self, ion: Ion) -> LocalTargetSpectrum:
        """``dP/dz|hit,c`` and ``<z>_hit,c`` for local targets in the track core."""
        ...

    def relative_chemical_effectiveness(self, ion: Ion) -> float:
        """``RCE = G/G_r`` for the representative radical at ``T_RCE`` (eq. 10).

        Dimensionless, and 1 by definition for the reference radiation. Alcocer-
        Avila et al. 2023 Fig. 5 shows it rising as LET falls, non-linearly.
        """
        ...
