# hetero1d

1D semiconductor heterostructure solver (effective-mass envelope-function method,
BenDaniel-Duke finite differences).

## Features
* single, multiple and asymmetric quantum wells, superlattices, custom layer stacks
* 12 binaries, 17 ternaries, 4 quaternaries (III-V), lattice matching
* electron, heavy-hole, light-hole subbands and wavefunctions
* intersubband / interband transitions and absorption, LO-phonon lifetimes
* transmission T(E), superlattice minibands, Schroedinger-Poisson
* electric field (Stark effect, QCSE), magnetic field (Landau levels, Zeeman, in-plane B)

## Installation
Directly from GitHub:
```bash
pip install git+https://github.com/erw-del/hetero1d.git
```
or download the code, then inside the folder:
```bash
pip install -e .
```

## 1. Build a structure

**Single well** (Al0.3Ga0.7As 40 nm | GaAs 20 nm | Al0.3Ga0.7As 40 nm).
Change `Lw` (well width) and `Lb` (barrier width) in nm:
```python
import numpy as np
from hetero1d import Structure, Solver
s = Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=20, Lb=40)
```

**Several wells** with different widths (`Lb` = inner barriers, `Lb_out` = outer barriers):
```python
s = Structure.multi_well(3, well="GaAs", barrier=("AlGaAs", 0.3),
                         Lw=[15, 10, 20], Lb=[5, 8], Lb_out=30)
```

**Superlattice** (10 periods):
```python
s = Structure.superlattice(10, well="GaAs", barrier=("AlGaAs", 0.3), Lw=5, Lb=2)
```

**Any layer stack**, layer by layer (thickness in nm, optional doping in cm^-3):
```python
s = Structure()
s.add("AlGaAs", 40, x=0.3).add("GaAs", 20).add("AlGaAs", 5, x=0.3)
s.add("GaAs", 10).add("AlGaAs", 40, x=0.3, doping=1e17)
```

**Other materials** (`list_materials()` shows all):
```python
s = Structure.single_well(well=("InGaAs", 0.53), barrier=("InAlAs", 0.52),
                          Lw=8, Lb=20, substrate="InP")
```

## 2. Solve and see results
```python
sol = Solver(s, T=300).solve(n_e=6, n_h=6)   # 6 electron and 6 hole states
sol.report(n2d_cm2=5e11)                     # all tables
sol.plot_summary("result.png")               # 6-panel figure
```
Energies are on an absolute scale (GaAs conduction-band edge = 0.6225 eV at 300 K).
The report's "E-edge" column gives each level measured from the well bottom.

## 3. Save energies and wavefunctions (CSV, opens in Excel/Origin)
```python
data = np.column_stack([sol.z, sol.Ec, sol.Ev, sol.psi_e, sol.psi_hh])
header = "z_nm,Ec_eV,Ev_eV," + \
         ",".join(f"psi_e{i}" for i in range(sol.psi_e.shape[1])) + "," + \
         ",".join(f"psi_hh{i}" for i in range(sol.psi_hh.shape[1]))
np.savetxt("wavefunctions.csv", data, delimiter=",", header=header, comments="")

E_meV = (sol.Ee - sol.Ec.min()) * 1000      # energies above the well bottom
np.savetxt("energies.csv",
           np.column_stack([np.arange(len(E_meV)), sol.Ee, E_meV, sol.e_bound]),
           delimiter=",", header="n,E_eV,E_above_well_meV,bound", comments="")
```

## 4. Electric field
```python
sol = Solver(s, field=50).solve()                       # 50 kV/cm
sol.plot_stark("stark.png", np.linspace(-80, 80, 33))   # Stark effect sweep
```

## 5. Magnetic field
```python
sol = Solver(s, T=4.2, Bz=5).solve()       # 5 T along growth axis: Landau levels, Zeeman
sol.report(n2d_cm2=3e11)
sol.plot_magnetic("landau.png", np.linspace(0.5, 10, 200), n2d_cm2=3e11)

sol = Solver(s, T=4.2, Bpar=10).solve()    # 10 T in-plane: diamagnetic shift
```

## 6. Other useful options
```python
Solver(s, T=77)                  # temperature (K)
Solver(s, dz=0.05)               # finer grid (nm), default 0.1
T = sol.transmission(np.linspace(0.9, 1.3, 2000))   # tunnelling T(E)
EF, pops = sol.fermi_level(5e11) # Fermi level and subband populations
```

Run all built-in examples with `hetero1d-demo`.

## Model limits
Single-band effective mass (no k.p mixing, no non-parabolicity), Gamma valley only,
no strain.

## Parameters
I. Vurgaftman, J. R. Meyer, L. R. Ram-Mohan, J. Appl. Phys. 89, 5815 (2001).

## Author
Ravi Kumar, 2026. MIT License.
