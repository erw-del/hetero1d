"""
hetero1d - 1D semiconductor heterostructure solver
(quantum wells, multiple wells, superlattices, graded and custom geometries;
electric and magnetic fields)

Quick start
-----------
>>> from hetero1d import Structure, Solver
>>> s = Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=20, Lb=40)
>>> sol = Solver(s, T=300).solve()
>>> sol.report()

Other geometries
----------------
>>> Structure.parabolic_well(Lw=40)            # graded parabolic AlGaAs well
>>> Structure.triangular_well(Lw=20)
>>> Structure.sawtooth(n_wells=4, Lw=20)
>>> from hetero1d import potentials            # formula potentials for V_ext
>>> Solver(Structure().add("GaAs", 200), V_ext=potentials.parabolic(10, center=100))
"""
from .core import (
    Structure, Layer, Solver,
    material, add_material, list_materials, lattice_matched,
    BINARIES, TERNARIES, QUATERNARIES, SHAPES,
)
from . import potentials

__version__ = "0.4.2"
__all__ = ["Structure", "Layer", "Solver", "material", "add_material",
           "list_materials", "lattice_matched", "BINARIES", "TERNARIES",
           "QUATERNARIES", "SHAPES", "potentials", "__version__"]
