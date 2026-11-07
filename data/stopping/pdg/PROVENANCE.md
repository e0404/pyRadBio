# PDG: Sternheimer density-effect coefficients and plasma energy of liquid water

**Origin.** Particle Data Group, Atomic and Nuclear Properties of Materials (2024):
* `https://pdg.lbl.gov/2024/AtomicNuclearProperties/MUE/muE_water_liquid.txt` (muon
  dE/dx table for liquid water; its header line "Sternheimer coef:" gives a, k, x_0,
  x_1, I, Cbar, delta_0);
* `https://pdg.lbl.gov/2024/AtomicNuclearProperties/HTML/water_liquid.html` (material
  page; plasma energy hbar*omega_p = 21.47 eV, <Z/A>, density).
Fetched by `scripts/stopping/fetch_sources.py --only pdg`.

**Retrieved.** 2026-10-14 (UTC timestamp in `FETCHED.json`).

**Citation.** S. Navas et al. (Particle Data Group), Review of Particle Physics, Phys. Rev.
D 110, 030001 (2024), chapter 34 "Passage of Particles Through Matter", eq. (34.7)
(Sternheimer's parameterization of delta) and the Atomic and Nuclear Properties tables.
The coefficients go back to R.M. Sternheimer, M.J. Berger and S.M. Seltzer, Atom. Data
Nucl. Data Tables 30, 261 (1984) (PDG ref. [18]); PDG's explanation page says the
density-effect parameters of its tables follow Sternheimer, Seltzer and Berger and the
algorithm of Sternheimer and Peierls (Phys. Rev. B 3, 3681 (1971)).

**Notice.** The Review of Particle Physics 2024 is published under the Creative Commons
Attribution 4.0 licence (Crossref licence metadata for doi:10.1103/PhysRevD.110.030001:
https://creativecommons.org/licenses/by/4.0/). Attribution as above.

**Which I-value the coefficients belong to.** The file lists them
next to I = 79.7 eV (PDG's own value for water), but Cbar is fixed by I and the plasma
energy, Cbar = 2 ln(I / hbar*omega_p) + 1. With PDG's hbar*omega_p = 21.47 eV this gives
3.5017 for I = 75 eV -- exactly the printed Cbar -- and 3.5801 / 3.6232 for 78 / 79.7 eV.
The coefficients are therefore the I = 75 eV set (ICRU 37/49 water); pyRadBio uses them
only with I = 75 eV.

**Files.** The raw downloads are static files; the extracted CSV holds the seven
coefficients and the plasma energy as printed.

| file | sha256 | bytes |
|---|---|---|
| `muE_water_liquid.txt` | `2cebc6a422ad9f8fa90b0e06a2e3c6e9ae68c2ecbe49d0717f85514b698599c2` | 18078 |
| `water_liquid.html` | `940a3ff98bb5d9cde2662b864e2586942269e261d64bfbabaa4df773cf47cc5c` | 4578 |
| `pdg_water_liquid_sternheimer.csv` | `8a27541d09c8bff365c3fd31584f34d824e3e9fa1684af8c98bcad064a9a83d1` | 332 |
