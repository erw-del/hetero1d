# hetero1d

1D semiconductor heterostructure solver (effective-mass envelope function).

* single, multiple and asymmetric quantum wells, superlattices, custom stacks
* 12 binaries, 17 ternaries, 4 quaternaries (III-V), lattice matching
* electron, heavy-hole, light-hole subbands and wavefunctions (BenDaniel-Duke FDM)
* intersubband / interband transitions and absorption, LO-phonon lifetimes
* transmission T(E), superlattice minibands, Schroedinger-Poisson
* electric field (Stark effect, QCSE), magnetic field (Landau levels, Zeeman, in-plane B)

## Install

```bash
pip install hetero1d
```

## Example

```python
from hetero1d import Structure, Solver

# Al0.3Ga0.7As 40 nm | GaAs 20 nm | Al0.3Ga0.7As 40 nm
s = Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=20, Lb=40)
sol = Solver(s, T=300).solve()
sol.report(n2d_cm2=5e11)
sol.plot_summary("qw.png")
```

Run all built-in examples: `hetero1d-demo`

## Model limits
Single-band effective mass (no k.p mixing, no non-parabolicity), Gamma valley only,
no strain. Parameters: Vurgaftman, Meyer, Ram-Mohan, J. Appl. Phys. 89, 5815 (2001).
