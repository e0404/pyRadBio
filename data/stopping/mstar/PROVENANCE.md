# MSTAR 3.12 (Paul & Schinner): FORTRAN source and database

**Origin.** IAEA Nuclear Data Section, "Electronic Stopping Power of Matter for Ions",
program page https://www-nds.iaea.org/stopping/mstar, archive
https://www-nds.iaea.org/stopping-media/MSTAR312.zip.

**Retrieved.** The IAEA site rejects scripted (non-browser) clients with
HTTP 403, so the archive is **downloaded by hand** into this folder;
`scripts/stopping/fetch_sources.py` then verifies its checksum and extracts the sources
and documentation. The bundled DOS executable (`MSTAR312/DOS exe/`) is never extracted.
(This copy was downloaded with curl using a browser user-agent, once, for the record.)

**Citation.**

* [1] H. Paul and A. Schinner, "An empirical approach to the stopping power of solids
  and gases for ions from 3Li to 18Ar", Nucl. Instr. Meth. Phys. Res. B 179 (2001) 299.
* [2] H. Paul and A. Schinner, same title, Part II, Nucl. Instr. Meth. Phys. Res. B 195
  (2002) 166.
* [3] H. Paul and A. Schinner, "Empirical stopping power tables for ions from 3Li to 18Ar
  and from 0.001 to 1000 MeV/nucleon in solids and gases", Atomic Data Nucl. Data Tables
  85 (2003) 377.
* Source header (`mssubs.f`): "WRITTEN BY ANDREAS SCHINNER AND HELMUT PAUL", program
  system MORESTAR, V 3.12 (24-05-2004).

**Notice (licence, verbatim from the program page).** "This program and/or the
corresponding subroutine package is hereby released as freeware. This means you can
freely use, copy, modify, and/or distribute the source code. However, please do not
remove the authors' names from the source code comments. The authors cannot be made
responsible for any damage caused by the use of their program code. YOU ARE USING THE
ENTIRE PROGRAM SYSTEM OR ANY PART OF IT ENTIRELY AT YOUR OWN RISK!" Our Python port
(`pyradbio/track/stopping/mstar/`) keeps Paul's and Schinner's names in its header.

**Built-in data.** The database `msdbs1.d` states no terms of its own. Per its
header and `mssubs.f`, it holds the alpha electronic stopping powers of M.J. Berger's
ASTAR ("NISTIR 4999 (1992); Program ASTAR, Version 2 of 12 August 1993", i.e. the
ICRU 49 alpha data, NIST SRD 124: see `../astar/PROVENANCE.md` for NIST's notice),
section 1 alpha energies in MeV (total He energy), section 2 stopping powers in
MeV cm^2/mg, with "extensions to ASTAR data ... calculated using program BEST
(25/09/01)" and later Ni, Ta, B and Zr targets. MSTAR transforms these alpha values
into stopping powers for 2 < Z1 < 19 with Paul's fitting formulae.

**Modes, from the program page and `mssubs.f`.** 'c' condensed targets, all ions
together (Z2-independent coefficients); 'd' condensed, fitted to the particular ion
only; 'g' gaseous, all ions together (H2 and He targets treated separately); 'h'
gaseous, particular ion (not Mg, Al, Si, P); 'a' automatic state of aggregation, always
c|g; **'b' automatic state, d|h if available, otherwise c|g** -- the authors' own
recommendation ("We suggest to use mode 'b' which should be good for most purposes").
Where mode 'd' has too little data or deviates negligibly from 'c', MSTAR silently uses
'c'. On exit the routine reports the mode actually used.

**Units.** Input energy "in MeV / nucleon" (`MSTAR1` argument `E`); output electronic
stopping power in MeV cm^2/mg. Read from `mstar1.f`: MSTAR evaluates the
alpha table at exactly `4.0D0*E` (total alpha energy for 4 nucleons) and returns
`STE*FOUT*Z1**2/4`; it never uses the ion mass. That is our E/A convention.

**Reference values from the original program.** `scripts/stopping/mstar_reference.py`
compiles the sources above (without the sample main `mstar.f`, and never the DOS
executable) with a small driver of ours and writes
`reference/mstar_water_reference.txt` (git-ignored; Z 3-18 x modes a/b/c/d/g/h x the
database energies and their midpoints, 25,248 calls). It is compiler-dependent output
and therefore not in the checksum table; the file was made with GNU
Fortran 13.1.0 (MinGW-W64). The Python port reproduces it to <= 2.0e-15.

**Files.**

| file | sha256 | bytes |
|---|---|---|
| `MSTAR312.zip` | `63f646a1332079724711a73cd6de9319312d589159d009bd79e977576393e9e0` | 373635 |
| `MSTAR312/Sources/msdbs1.d` | `f6309e06d4f0e68f43f7da81aa41828d0dcfbc5fbfb7303df1aa6e9047fe713c` | 205889 |
| `MSTAR312/Sources/msemsg.f` | `c364f33b4fa236087dd875cabf6bb0e835c69eabfbfa4543aba6d645272491b6` | 4630 |
| `MSTAR312/Sources/mserf1.f` | `23e0310c3c35a8733edc973a260ad0f16bb4b6ea004133da06f64a816dc9b671` | 2870 |
| `MSTAR312/Sources/msid2z.f` | `6cef8331e8ca3ed60bfb18cbbce4c0fb9826d8c78c0e9685454da168c16497a0` | 4896 |
| `MSTAR312/Sources/mspaul.f` | `d8b59a55a0dd848a085917285cb70bf99a3de0eb09215ab54870492364dbd36e` | 14931 |
| `MSTAR312/Sources/msrdb1.f` | `4c34099e79343a9e7540b1b650e6b90f4050c5b91dd817ec4d232177037425c3` | 6779 |
| `MSTAR312/Sources/msseal.f` | `484e481477e2804c210fc87db82c9ab9e525663743b70fa78bdd4cca69d22b75` | 3879 |
| `MSTAR312/Sources/msspl1.f` | `c6c2b0e5f3bfd0be0b6e6f8e140e62fd18f79468a4c15844c282308b74ae35d4` | 3375 |
| `MSTAR312/Sources/msspl2.f` | `9ea435ff0fe8d044b9f4faaaaab601d6f0cfcb6da7f60b631dae10eae699be69` | 3300 |
| `MSTAR312/Sources/mssubs.f` | `97903edc7c2e78466b5e13baa2393e1ecc8f559915df7dbf859125e859e354ec` | 8992 |
| `MSTAR312/Sources/mstar.f` | `821157bf2ad48996df6b85240fb42a198eed1880585e264f6aaf4df42f58fbd9` | 11732 |
| `MSTAR312/Sources/mstar1.f` | `7f24ee3b7aec4c3f2d703db825de1f307d62a2cd51238ee37b9a2e362f25929e` | 6570 |
| `MSTAR312/Sources/History.txt` | `ad97494e04a0095cb00500cca30b207b93ae89ca9afb6f0397bcb294bccc5a88` | 54 |
| `MSTAR312/Doc/MSTARInstr.htm` | `fd3ea4a6243e5f3220b4bc3d765a1c166739dd10f81eaa75088528c7e0a47940` | 68579 |
