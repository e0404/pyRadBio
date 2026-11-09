"""Water stopping-power sources with a common interface and cited provenance.

Every class returns the mass stopping power of liquid water [MeV cm^2/g] versus
kinetic energy per nucleon [MeV/u] (see [`pyradbio.track.stopping.base`][] for the
conventions), raises [`OutOfRangeError`][] instead of extrapolating, and records
its source per ion. The library default,
[`pyradbio.track.stopping_power.default_stopping_power`][], combines
[`ICRU90Procedure`][] (Z 1-18), Hubert's alpha scaling (Z 19-92 from 2.5 MeV/u)
and analytic Bethe elsewhere.
"""

from pyradbio.track.stopping.barkas import WaterBarkasCorrection
from pyradbio.track.stopping.base import (
    LIQUID_WATER,
    IonSourceInfo,
    MassStoppingPowerModel,
    OutOfRangeError,
    UnsupportedIonError,
    beta_from_energy_per_nucleon,
    energy_per_nucleon_MeV_u,
    total_kinetic_energy_MeV,
)
from pyradbio.track.stopping.bethe import Bethe
from pyradbio.track.stopping.effective_charge import (
    BarkasEffectiveCharge,
    BrandtKitagawaEffectiveCharge,
    EffectiveChargeModel,
    HubertEffectiveCharge,
    barkas_effective_charge,
)
from pyradbio.track.stopping.grid import (
    IonTable,
    LogEnergyGrid,
    RegriddedTable,
    TableFile,
    leave_one_out_relative_error,
    read_table_csv,
    write_table_csv,
)
from pyradbio.track.stopping.hubert import HubertAlphaScaling
from pyradbio.track.stopping.icru import ICRU
from pyradbio.track.stopping.icru90_procedure import ICRU90Procedure
from pyradbio.track.stopping.masses import atomic_mass_u, nuclear_mass_MeV
from pyradbio.track.stopping.mstar import MSTAR
from pyradbio.track.stopping.nuclear import WaterNuclearStopping
from pyradbio.track.stopping.shell import WaterShellCorrection
from pyradbio.track.stopping.xstar import XSTAR, XSTAR_I_VALUE_EV

__all__ = [
    "ICRU",
    "LIQUID_WATER",
    "MSTAR",
    "XSTAR",
    "XSTAR_I_VALUE_EV",
    "BarkasEffectiveCharge",
    "Bethe",
    "BrandtKitagawaEffectiveCharge",
    "EffectiveChargeModel",
    "HubertAlphaScaling",
    "HubertEffectiveCharge",
    "ICRU90Procedure",
    "IonSourceInfo",
    "IonTable",
    "LogEnergyGrid",
    "MassStoppingPowerModel",
    "OutOfRangeError",
    "RegriddedTable",
    "TableFile",
    "UnsupportedIonError",
    "WaterBarkasCorrection",
    "WaterNuclearStopping",
    "WaterShellCorrection",
    "atomic_mass_u",
    "barkas_effective_charge",
    "beta_from_energy_per_nucleon",
    "energy_per_nucleon_MeV_u",
    "leave_one_out_relative_error",
    "nuclear_mass_MeV",
    "read_table_csv",
    "total_kinetic_energy_MeV",
    "write_table_csv",
]
