"""
hetero1d.potentials -- ready-made potential shapes for Solver(..., V_ext=...)

Every function returns V(z): z in nm (measured from the left edge of the
structure), V in eV. V_ext is an electrostatic potential: it shifts the
conduction AND valence bands together (electrons see a well, holes a hill).
For wells that confine electrons and holes, use the graded-composition
builders instead (Structure.parabolic_well, triangular_well, ...).

    from hetero1d import Structure, Solver, potentials
    s = Structure().add("GaAs", 200)
    sol = Solver(s, V_ext=potentials.parabolic(hw_meV=10, center=100)).solve()
"""
import numpy as np

HB2M0 = 0.0380998            # hbar^2/(2 m0), eV nm^2
KV_CM = 1e-4                 # 1 kV/cm = 1e-4 eV/nm


def parabolic(hw_meV=10.0, center=0.0, mass=0.067):
    """Harmonic oscillator with level spacing hw_meV for an electron of the given mass:
    E_n = hw (n + 1/2)."""
    hw = hw_meV * 1e-3
    c = (hw / 2) ** 2 * mass / HB2M0
    return lambda z: c * (np.asarray(z) - center) ** 2


def triangular(field_kVcm=50.0, start=0.0, height=None):
    """Linear rise eF(z - start) from start (wall = edge of the box or a barrier),
    0 before start. Optional height caps the potential (finite triangular well)."""
    F = field_kVcm * KV_CM
    def V(z):
        v = np.where(np.asarray(z) >= start, F * (np.asarray(z) - start), 0.0)
        return np.minimum(v, height) if height is not None else v
    return V


def v_shape(slope_kVcm=50.0, center=0.0, height=None):
    """Symmetric V: e*F*|z - center| (e.g. a delta-doped layer); optional cap."""
    F = slope_kVcm * KV_CM
    def V(z):
        v = F * np.abs(np.asarray(z) - center)
        return np.minimum(v, height) if height is not None else v
    return V


def sawtooth(n=4, period=20.0, height=0.15, start=20.0, gap=0.0):
    """n triangles of width `period` rising from 0 to `height`, separated by flat
    barriers of width `gap`; flat at `height` outside."""
    total = n * (period + gap) - gap
    def V(z):
        z = np.asarray(z, float)
        u = np.mod(z - start, period + gap)
        inside = (z >= start) & (z < start + total) & (u < period)
        return np.where(inside, height * u / period, height)
    return V


def step(position=0.0, height=0.1):
    """Potential step of the given height at `position`."""
    return lambda z: np.where(np.asarray(z) >= position, height, 0.0)


def square_well(left=0.0, right=10.0, depth=0.2):
    """Square well of the given depth between left and right (0 outside, -depth inside)."""
    return lambda z: np.where((np.asarray(z) > left) & (np.asarray(z) < right), -depth, 0.0)


def poschl_teller(depth=0.2, width=5.0, center=0.0):
    """Pöschl-Teller well -depth / cosh^2((z - center)/width), exactly solvable."""
    return lambda z: -depth / np.cosh((np.asarray(z) - center) / width) ** 2


def gaussian(depth=0.1, width=5.0, center=0.0):
    """Gaussian well -depth * exp(-(z - center)^2 / (2 width^2))."""
    return lambda z: -depth * np.exp(-(np.asarray(z) - center) ** 2 / (2 * width ** 2))


def uniform_field(field_kVcm=10.0, z0=0.0):
    """Uniform field e*F*(z - z0) (same as Solver(field=...) apart from the zero point)."""
    F = field_kVcm * KV_CM
    return lambda z: F * (np.asarray(z) - z0)


def combine(*potentials):
    """Sum of several potentials: combine(parabolic(...), uniform_field(...))."""
    return lambda z: sum(np.asarray(p(z), float) for p in potentials)
