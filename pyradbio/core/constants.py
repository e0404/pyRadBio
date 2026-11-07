"""Physical constants and unit conventions for pyRadBio.

Unit conventions used throughout the package (be strict and explicit):
  - kinetic energy per nucleon : MeV/u
  - LET (unrestricted, LET_inf): keV/um
  - geometric radii            : um
  - absorbed dose / spec. energy: Gy
  - alpha                      : Gy^-1
  - beta                       : Gy^-2

Every constant below MUST carry a source comment. No magic numbers elsewhere.
"""

from __future__ import annotations

# Density of liquid water, used for fluence<->dose and z* geometry (g/cm^3).
WATER_DENSITY_G_CM3: float = 1.0

# keV/um -> Gy*um^2 conversion for single-traversal dose D1 = 0.16*LET/(rho*A)
# (Elsasser & Scholz 2007, eq. 8, which misprints rho in the numerator; A in um^2,
# LET in keV/um, rho in g/cm^3).
LEM_D1_PREFACTOR: float = 0.16

# Fluence x LET -> absorbed dose:
#   D[Gy] = FLUENCE_LET_TO_DOSE_GY * fluence[cm^-2] * LET[keV/um] / rho[g/cm^3].
# Derivation: 1 keV/um = 1.602176634e-10 J/cm; per particle per cm^2 and per
# g/cm^3 of density this gives J/kg = Gy. The factor is the elementary charge in
# coulombs scaled to these units (CODATA 2018 e = 1.602176634e-19 C).
FLUENCE_LET_TO_DOSE_GY: float = 1.602176634e-9

# Local-dose <-> LET energy conservation (radial dose):
#   LET[keV/um] = KEVUM_PER_GY_UM2_GCM3 * rho[g/cm^3] * integral(D(r)[Gy]*2*pi*r dr)[um^2]
# i.e. the energy deposited per unit track length equals the cross-sectional
# integral of the local energy density (D*rho). Unit reduction:
#   1 Gy*um^2*(g/cm^3) = (J/kg)(1e-12 m^2)(1e3 kg/m^3) = 1e-9 J/m,
#   1 keV/um = 1.602176634e-10 J/m  =>  factor = 1e-9 / 1.602176634e-10.
KEVUM_PER_GY_UM2_GCM3: float = 1.0e-9 / 1.602176634e-10  # ~6.2415 keV/um per Gy*um^2*(g/cm^3)

# Linear stopping power unit change: 1 MeV/cm = 1e3 keV / 1e4 um = 0.1 keV/um.
# A mass stopping power [MeV cm^2/g] times the density [g/cm^3] is MeV/cm, so
#   LET[keV/um] = S/rho[MeV cm^2/g] * rho[g/cm^3] * MEV_PER_CM_TO_KEV_PER_UM.
# Every stopping-power table pyRadBio reads (NS1970, libdEdx, pyMKM) is
# in MeV cm^2/g or MeV/cm; pyMKM's "let" column is MeV/cm (not keV/um).
MEV_PER_CM_TO_KEV_PER_UM: float = 0.1

# Atomic mass unit energy equivalent, used for ion kinematics (beta = v/c from
# kinetic energy per nucleon). CODATA 2018: u = 931.49410242 MeV/c^2.
ATOMIC_MASS_UNIT_MEV: float = 931.49410242

# Electron rest energy, for the Bethe stopping-power formula (CODATA 2018).
ELECTRON_MASS_MEV: float = 0.51099895000

# Avogadro constant, exact by the 2019 SI definition.
AVOGADRO_PER_MOL: float = 6.02214076e23

# Bohr radius and Hartree energy (CODATA 2018), for atomic-unit scattering formulae.
BOHR_RADIUS_CM: float = 0.529177210903e-8
HARTREE_MEV: float = 27.211386245988e-6

# Mean excitation energy of liquid water (ICRU 90 / NIST): I = 78 eV.
WATER_MEAN_EXCITATION_EV: float = 78.0

# Bethe mass-stopping-power coefficient K = 4*pi*N_A*r_e^2*m_e*c^2
# = 0.307075 MeV*cm^2/mol (PDG). Multiply by (Z/A)_medium to get MeV*cm^2/g.
BETHE_K_MEV_CM2_MOL: float = 0.307075

# Electron density proxy for liquid water: <Z/A> = 0.55509 (PDG material table).
WATER_Z_OVER_A: float = 0.55509

# Bohr energy-loss straggling coefficient. The variance of the energy lost by an
# ion over a path length s (non-relativistic Bohr limit) is
#   Omega^2[MeV^2] = BOHR_STRAGGLING_MEV2_CM2_MOL * z_eff^2 * (Z/A)[mol/g]
#                    * rho[g/cm^3] * s[cm].
# The coefficient is 4*pi*N_A*r_e^2*(m_e c^2)^2 = K * m_e c^2, i.e. the Bethe K
# times the electron rest energy (Leo, "Techniques for Nuclear and Particle
# Physics", 2nd ed., eq. 2.98; == 0.1569 MeV^2 cm^2/mol). Derived from the two
# constants above so it stays consistent and carries no independent magic number.
BOHR_STRAGGLING_MEV2_CM2_MOL: float = BETHE_K_MEV_CM2_MOL * ELECTRON_MASS_MEV
