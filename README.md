# hetero1d

**A complete 1D semiconductor heterostructure solver in Python.**
Design quantum wells, multiple wells, superlattices or any layer stack from
III-V materials and compute electronic, optical, transport and magneto-optical
properties: under electric and magnetic fields, at any temperature, with or
without doping.

Method: effective-mass envelope-function approximation, BenDaniel-Duke
finite differences with position-dependent mass.

---

## What it can compute

| Category | Quantities |
|---|---|
| **Structure** | Band profile Ec(z), Ev(z); band gaps, offsets, masses, g-factors of every layer; lattice-mismatch warning |
| **Quantum states** | Electron, heavy-hole and light-hole subband energies and wavefunctions; bound/continuum classification |
| **State analysis** | Probability in each layer, mean position ⟨z⟩, spatial spread, effective mass of each state |
| **Intersubband optics** | Transition energies and wavelengths, dipole matrix elements z_ij, oscillator strengths, **absolute absorption spectrum** (per pass) |
| **Interband optics** | e-hh and e-lh transition energies and wavelengths, electron-hole overlaps, absorption spectrum (TE) |
| **Carrier statistics** | 2D density of states, Fermi level, population of each subband (Fermi-Dirac, any temperature) |
| **Scattering** | LO-phonon intersubband scattering rates and lifetimes (Fröhlich interaction) |
| **Transport / tunnelling** | Transmission coefficient T(E), resonant tunnelling, resonance widths |
| **Superlattices** | Miniband energies and widths, Bloch dispersion E(q) |
| **Doped structures** | Self-consistent Schrödinger-Poisson: band bending, electron density, Fermi level |
| **Electric field** | Stark shifts, quantum-confined Stark effect (QCSE), field-dependent overlaps, dipoles and intersubband energies |
| **Magnetic field ∥ z** | Landau levels, spin Zeeman splitting, Landau fan, Landau-level DOS, filling factor, Fermi level vs B (Shubnikov-de Haas-like oscillations) |
| **Magnetic field ∥ plane** | Diamagnetic shift, in-plane dispersion E_n(k_y) |
| **Materials** | 12 binaries, 17 ternaries, 4 quaternaries, temperature-dependent band gaps, lattice matching, user-defined materials |

**Materials included**
- Binaries: GaAs, AlAs, InAs, InP, GaSb, AlSb, GaP, AlP, InSb, GaN, AlN, InN (nitrides: zinc-blende, approximate)
- Ternaries: AlGaAs, InGaAs, InAlAs, GaAsSb, AlGaSb, InGaP, InAsSb, InGaSb, InAlSb, AlAsSb, InAsP, GaAsP, AlGaP, InAlP, AlGaN, InGaN, InAlN
- Quaternaries: InGaAsP, InGaAsSb, AlGaInAs, AlGaInP

---

## Installation
```bash
pip install hetero1d
```
or the latest version from GitHub:
```bash
pip install git+https://github.com/erw-del/hetero1d.git
```
---

## Quick start
```python
import numpy as np
from hetero1d import Structure, Solver

# Al0.3Ga0.7As 40 nm | GaAs 20 nm | Al0.3Ga0.7As 40 nm
s = Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=20, Lb=40)
s.describe()                         # band edges, masses, g-factors of each layer

sol = Solver(s, T=300).solve(n_e=6, n_h=6)
sol.report(n2d_cm2=5e11)             # all results as tables
sol.plot_summary("result.png")       # 6-panel figure
```
Energies are on an absolute scale (GaAs conduction-band edge = 0.6225 eV at 300 K).
The report's "E-edge" column gives each level measured from the well bottom.

---

## 1. Building structures

Change `Lw` (well width) and `Lb` (barrier width), in nm.

```python
# single well
s = Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=20, Lb=40)

# several wells with different widths (Lb = inner barriers, Lb_out = outer barriers)
s = Structure.multi_well(3, well="GaAs", barrier=("AlGaAs", 0.3),
                         Lw=[15, 10, 20], Lb=[5, 8], Lb_out=30)

# superlattice, 10 periods
s = Structure.superlattice(10, well="GaAs", barrier=("AlGaAs", 0.3), Lw=5, Lb=2)

# any layer stack, layer by layer (thickness nm, optional doping cm^-3)
s = Structure()
s.add("AlGaAs", 40, x=0.3).add("GaAs", 20).add("AlGaAs", 5, x=0.3)
s.add("GaAs", 10).add("AlGaAs", 40, x=0.3, doping=1e17)

# other materials and substrates
s = Structure.single_well(well=("InGaAs", 0.53), barrier=("InAlAs", 0.52),
                          Lw=8, Lb=20, substrate="InP")

# quaternary: Ga_x In_(1-x) As_y P_(1-y)
s = Structure.single_well(well=("InGaAsP", 0.39, 0.85), barrier="InP",
                          Lw=8, Lb=20, substrate="InP")
```

## 2. Materials
```python
from hetero1d import material, list_materials, lattice_matched, add_material

list_materials()                                  # all materials and what x, y mean
m = material("AlGaAs", x=0.3, T=300)              # all parameters as a dict
print(m["Eg"], m["Ec"], m["Ev"], m["me"], m["g"])

x = lattice_matched("InGaAs", "InP")              # -> 0.534
x = lattice_matched("InGaAsP", "InP", fixed=("y", 0.85))

add_material("MyMat", Ec=0.70, Ev=-0.75, me=0.07, mhh=0.4, mlh=0.08, eps_s=12.5)
```

## 3. Energies, wavefunctions and state analysis
```python
sol = Solver(s, T=300).solve(n_e=6, n_h=6)

sol.Ee, sol.psi_e          # electron energies (eV) and wavefunctions on grid sol.z (nm)
sol.Ehh, sol.psi_hh        # heavy holes
sol.Elh, sol.psi_lh        # light holes
sol.e_bound                # True for bound states
E_meV = (sol.Ee - sol.Ec.min()) * 1000            # energies above the well bottom

sol.layer_probabilities(sol.psi_e[:, 0])          # probability in each layer
zmean, spread = sol.expect_z(sol.psi_e[:, 0])     # <z> and spatial spread (nm)
```

**Save to CSV** (opens in Excel / Origin):
```python
data = np.column_stack([sol.z, sol.Ec, sol.Ev, sol.psi_e, sol.psi_hh])
header = "z_nm,Ec_eV,Ev_eV," + \
         ",".join(f"psi_e{i}" for i in range(sol.psi_e.shape[1])) + "," + \
         ",".join(f"psi_hh{i}" for i in range(sol.psi_hh.shape[1]))
np.savetxt("wavefunctions.csv", data, delimiter=",", header=header, comments="")

np.savetxt("energies.csv",
           np.column_stack([np.arange(len(E_meV)), sol.Ee, E_meV, sol.e_bound]),
           delimiter=",", header="n,E_eV,E_above_well_meV,bound", comments="")
```

## 4. Intersubband optics (quantum-well infrared photodetectors, cascade lasers)
```python
for t in sol.intersubband():      # i, j, dE (eV), lam_um, z (nm), f
    print(t["i"], "->", t["j"], t["dE"]*1000, "meV", t["lam_um"], "um", "f =", t["f"])

E = np.linspace(0.01, 0.3, 1000)                              # photon energy (eV)
A = sol.isb_absorption(E, n2d_cm2=5e11, gamma=0.005)          # absorption per pass (fraction)
```

## 5. Interband optics (lasers, LEDs, photoluminescence)
```python
for t in sol.interband()[:5]:     # e, h, band ('hh'/'lh'), E (eV), lam_nm, overlap2
    print(f"e{t['e']}-{t['band']}{t['h']}: {t['E']:.4f} eV, {t['lam_nm']:.1f} nm")

E = np.linspace(1.4, 1.7, 1000)
A = sol.interband_absorption(E, gamma=0.005)                  # relative units
```

## 6. Carrier statistics
```python
EF, populations = sol.fermi_level(5e11)    # Fermi level (eV), cm^-2 in each subband
```

## 7. LO-phonon scattering
```python
for t in sol.lo_phonon_rates():            # i, f, dE, rate (1/s), tau_ps
    print(t["i"], "->", t["f"], f"{t['tau_ps']:.3f} ps")
```

## 8. Tunnelling and transmission
```python
rtd = Structure()                          # double-barrier resonant tunnelling diode
rtd.add("GaAs", 20).add("AlGaAs", 5, x=0.3).add("GaAs", 6).add("AlGaAs", 5, x=0.3).add("GaAs", 20)
r = Solver(rtd, dz=0.02)
E = np.linspace(0.67, 0.70, 20000)
T = r.transmission(E)                      # T(E) between 0 and 1
```

## 9. Superlattice minibands
```python
sl = Structure.superlattice(10, well="GaAs", barrier=("AlGaAs", 0.3), Lw=5, Lb=2)
q, d, bands = Solver(sl).minibands(np.linspace(0.62, 1.22, 6000))
print(bands)                               # list of (E_min, E_max) for each miniband
```

## 10. Doped structures: Schrödinger-Poisson
```python
s = Structure()
s.add("AlGaAs", 30, x=0.3).add("AlGaAs", 10, x=0.3, doping=3e17).add("AlGaAs", 10, x=0.3)
s.add("GaAs", 15).add("AlGaAs", 10, x=0.3).add("AlGaAs", 10, x=0.3, doping=3e17).add("AlGaAs", 30, x=0.3)
sol = Solver(s, T=77).schrodinger_poisson()
sol.solve()
sol.VH, sol.n3d, sol.EF                    # Hartree potential, electron density, Fermi level
```

## 11. Electric field
```python
sol = Solver(s, field=50).solve()                       # 50 kV/cm along z
r = sol.stark_sweep(np.linspace(-80, 80, 33))           # dict: E_e0hh0, ov_e0hh0, E01, z01, ...
sol.plot_stark("stark.png", np.linspace(-80, 80, 33))   # QCSE, overlap, intersubband vs field
```

## 12. Magnetic field
```python
# B along the growth axis: Landau levels and Zeeman splitting
sol = Solver(s, T=4.2, Bz=5).solve()
sol.g_factor(0)                         # electron g-factor of subband 0
sol.zeeman()                            # spin splitting of each subband (eV)
sol.landau_levels(N_max=10)             # list of Landau levels
sol.fermi_level_B(3e11)                 # Fermi level in the field
sol.plot_magnetic("landau.png", np.linspace(0.5, 10, 200), n2d_cm2=3e11)

# B in the layer plane: diamagnetic shift and dispersion
sol = Solver(s, T=4.2, Bpar=10).solve()
E_ky = sol.inplane_dispersion(np.linspace(-0.2, 0.2, 41))
```

## 13. Other options
```python
Solver(s, T=77)             # temperature (K), band gaps follow Varshni law
Solver(s, dz=0.05)          # finer grid (nm), default 0.1
```

---

## Validation
| Test | Reference | hetero1d |
|---|---|---|
| 10 nm GaAs/Al0.3Ga0.7As, E1 | 30.842 meV (exact analytic) | 30.850 meV |
| Double-barrier resonance | 61.95 meV (well level) | T = 1.0000 at 61.91 meV |
| In0.53Ga0.47As/In0.52Al0.48As ΔEc | ≈ 0.52 eV (literature) | 0.522 eV |
| Landau spacing, 5 T | ħeB/m* | exact |
| In-plane diamagnetic shift, 1 T | 2nd-order perturbation theory | 0.0235 meV (identical) |

## Model limits
Single-band effective mass (no k·p band mixing, no non-parabolicity), Γ valley only
(X/L valleys not included), no strain, nitrides without polarisation fields,
Zeeman and Landau levels for electrons only.

## Parameters
I. Vurgaftman, J. R. Meyer, L. R. Ram-Mohan, J. Appl. Phys. 89, 5815 (2001);
nitrides: I. Vurgaftman, J. R. Meyer, J. Appl. Phys. 94, 3675 (2003).

## Author
Ravi Kumar, 2026. MIT License.
