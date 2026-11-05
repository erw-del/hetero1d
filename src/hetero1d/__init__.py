"""
hetero1d - 1D semiconductor heterostructure solver
(quantum wells, multiple wells, superlattices; electric and magnetic fields)

Quick start
-----------
>>> from hetero1d import Structure, Solver
>>> s = Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=20, Lb=40)
>>> sol = Solver(s, T=300).solve()
>>> sol.report()
"""
from .core import (
    Structure, Layer, Solver,
    material, add_material, list_materials, lattice_matched,
    BINARIES, TERNARIES, QUATERNARIES,
)

__version__ = "0.1.3"
__all__ = ["Structure", "Layer", "Solver", "material", "add_material",
           "list_materials", "lattice_matched", "BINARIES", "TERNARIES",
           "QUATERNARIES", "__version__"]
