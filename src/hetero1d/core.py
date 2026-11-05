"""
hetero1d.py  --  1D heterostructure solver (effective-mass, envelope function)
=============================================================================
Version 2: electric-field sweeps (Stark / QCSE), magnetic fields
(Landau levels, Zeeman, in-plane field), extended material database
(binaries, ternaries, quaternaries, Kane parameters, g-factors).

Build any layer stack (single well, multiple wells, superlattice, custom),
choose materials and alloys, and compute:

  * Band profile Ec(z), Ev(z) with temperature, electric field, magnetic field
  * Electron, heavy-hole and light-hole subbands and wavefunctions
    (BenDaniel-Duke FDM, position-dependent mass)
  * Localisation per layer, <z>, spread
  * Intersubband transitions: energies, wavelengths, dipoles, oscillator
    strengths, absolute absorption (per pass, z-polarised)
  * Interband e-hh / e-lh transitions and absorption (relative, TE)
  * 2D density of states, Fermi level, subband populations
  * LO-phonon intersubband lifetimes (Froehlich, golden rule)
  * Transmission T(E) by transfer matrix (tunnelling, resonances)
  * Superlattice minibands from the Bloch condition
  * Self-consistent Schroedinger-Poisson for doped structures
  * Electric field:  Stark shifts, quantum-confined Stark effect (QCSE),
                     overlap and dipole versus field
  * Magnetic field along z (growth axis):  Landau levels, spin Zeeman
                     splitting, Landau fan, Landau-level DOS, filling factor,
                     Fermi level versus B (Shubnikov-de Haas-like oscillation)
  * Magnetic field in-plane: diamagnetic shift, dispersion E_n(k_y)

Units: nm, eV, doping cm^-3, sheet density cm^-2, field kV/cm, B in tesla, K.

Model limits (state these when publishing):
  - single-band effective mass for each carrier (no k.p band mixing, no
    non-parabolicity); Gamma valley only (X/L valleys not included: AlGaAs
    x > ~0.45, AlAs, AlP, AlSb, GaP are indirect in reality)
  - strain not included (warning printed for mismatched layers)
  - g-factors: two-band Roth formula from Eg, Ep, Delta_so (electrons only)
  - Landau levels: decoupled subbands (tilted fields treated approximately)
  - nitrides: zinc-blende parameters, no spontaneous/piezo polarisation
    (wurtzite III-N quantum wells need polarisation fields: use `field`)
  - main parameter source: I. Vurgaftman, J. R. Meyer, L. R. Ram-Mohan,
    J. Appl. Phys. 89, 5815 (2001); nitrides: Vurgaftman & Meyer,
    J. Appl. Phys. 94, 3675 (2003) (some nitride values approximate)
"""

import copy
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh_tridiagonal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HB2M0 = 0.0380998          # hbar^2/(2 m0)                eV nm^2
KB = 8.617333e-5           # Boltzmann                    eV/K
POISSON = 18.0951          # e/eps0 (n in nm^-3, z in nm) eV nm
DOS0 = 1.0 / (2 * np.pi * HB2M0)   # m0/(pi hbar^2)       eV^-1 nm^-2
MU_B = 5.7883818e-5        # Bohr magneton                eV/T
HBAR_E_M0 = 1.1576766e-4   # hbar e / m0  (hbar*w_c for m*=1)  eV/T
L_B1 = 25.6556             # magnetic length at 1 T       nm
EB_H = 2.417989e10         # e/h  (Landau degeneracy per spin)  cm^-2 T^-1

# SI (absorption, phonon rates)
Q_SI = 1.602176634e-19
HBAR_SI = 1.054571817e-34
M0_SI = 9.1093837e-31
EPS0_SI = 8.8541878128e-12
C_SI = 2.99792458e8


# ---------------------------------------------------------------------------
# Material database
# ---------------------------------------------------------------------------
# Eg0 [eV] (Gamma gap at 0 K), Varshni alpha [eV/K], beta [K], VBO [eV]
# (absolute valence-band position, InSb = 0 scale), me [m0], Luttinger g1, g2,
# eps_s, eps_inf, hwLO [eV], a [A], Ep [eV] (Kane energy), Dso [eV]
_KEYS = ("Eg0", "alpha", "beta", "VBO", "me", "g1", "g2",
         "eps_s", "eps_inf", "hwLO", "a", "Ep", "Dso")

BINARIES = {
    #        Eg0     alpha     beta  VBO    me      g1     g2    eps_s  eps_inf hwLO     a        Ep    Dso
    "GaAs": (1.519,  5.405e-4, 204, -0.80, 0.067,  6.98, 2.06, 12.90, 10.89, 0.03625, 5.65325, 28.8, 0.341),
    "AlAs": (3.099,  8.850e-4, 530, -1.33, 0.150,  3.76, 0.82, 10.06,  8.16, 0.05009, 5.6611,  21.1, 0.28),
    "InAs": (0.417,  2.760e-4,  93, -0.59, 0.026, 20.00, 8.50, 15.15, 12.30, 0.02960, 6.0583,  21.5, 0.39),
    "InP":  (1.4236, 3.630e-4, 162, -0.94, 0.0795, 5.08, 1.60, 12.50,  9.61, 0.04280, 5.8697,  20.7, 0.108),
    "GaSb": (0.812,  4.170e-4, 140, -0.03, 0.039, 13.40, 4.70, 15.70, 14.44, 0.02880, 6.0959,  27.0, 0.76),
    "AlSb": (2.386,  4.200e-4, 140, -0.41, 0.140,  5.18, 1.19, 12.04, 10.24, 0.04210, 6.1355,  18.7, 0.676),
    "GaP":  (2.886,  5.771e-4, 372, -1.27, 0.130,  4.05, 0.49, 11.10,  9.11, 0.05000, 5.4505,  31.4, 0.08),
    "AlP":  (3.63,   5.771e-4, 372, -1.74, 0.220,  3.35, 0.71,  9.80,  7.50, 0.06220, 5.4672,  17.7, 0.07),
    "InSb": (0.235,  3.200e-4, 170,  0.00, 0.0135, 34.8, 15.5, 16.80, 15.70, 0.02370, 6.4794,  23.3, 0.81),
    # zinc-blende nitrides (approximate; no polarisation)
    "GaN":  (3.299,  5.930e-4, 600, -2.64, 0.150,  2.70, 0.76,  9.50,  5.35, 0.09120, 4.50,    25.0, 0.017),
    "AlN":  (4.90,   5.930e-4, 600, -3.44, 0.250,  1.92, 0.47,  8.07,  4.46, 0.11200, 4.38,    27.1, 0.019),
    "InN":  (0.78,   2.450e-4, 624, -2.00, 0.054,  3.72, 1.26, 15.30,  8.40, 0.07300, 4.98,    25.0, 0.005),
}
APPROXIMATE = {"GaN", "AlN", "InN"}

# Ternary A_x B_(1-x):  (binary at x, binary at 1-x, Eg bowing C(x),
#                        VBO bowing, me bowing, what x means)
#   P = x P_A + (1-x) P_B - x(1-x) C
TERNARIES = {
    "AlGaAs": ("AlAs", "GaAs", lambda x: -0.127 + 1.310 * x, 0.0, 0.0, "Al"),
    "InGaAs": ("InAs", "GaAs", lambda x: 0.477,  -0.38, 0.0091, "In"),
    "InAlAs": ("InAs", "AlAs", lambda x: 0.70,   -0.64, 0.049,  "In"),
    "GaAsSb": ("GaSb", "GaAs", lambda x: 1.43,   -1.06, 0.0,    "Sb"),
    "AlGaSb": ("AlSb", "GaSb", lambda x: -0.044 + 1.22 * x, 0.0, 0.0, "Al"),
    "InGaP":  ("InP",  "GaP",  lambda x: 0.65,    0.0,  0.0,    "In"),
    "InAsSb": ("InSb", "InAs", lambda x: 0.67,    0.0,  0.035,  "Sb"),
    "InGaSb": ("InSb", "GaSb", lambda x: 0.415,   0.0,  0.0092, "In"),
    "InAlSb": ("InSb", "AlSb", lambda x: 0.43,    0.0,  0.0,    "In"),
    "AlAsSb": ("AlSb", "AlAs", lambda x: 0.80,   -1.71, 0.0,    "Sb"),
    "InAsP":  ("InAs", "InP",  lambda x: 0.10,    0.0,  0.0,    "As"),
    "GaAsP":  ("GaAs", "GaP",  lambda x: 0.19,    0.0,  0.0,    "As"),
    "AlGaP":  ("AlP",  "GaP",  lambda x: 0.0,     0.0,  0.0,    "Al"),
    "InAlP":  ("InP",  "AlP",  lambda x: -0.48,   0.0,  0.0,    "In"),
    "AlGaN":  ("AlN",  "GaN",  lambda x: 0.70,    0.0,  0.0,    "Al"),
    "InGaN":  ("InN",  "GaN",  lambda x: 1.40,    0.0,  0.0,    "In"),
    "InAlN":  ("InN",  "AlN",  lambda x: 2.50,    0.0,  0.0,    "In"),
}

# Quaternaries (Vurgaftman 2001 interpolation from the ternaries)
#  type "AB-CD": A_x B_(1-x) C_y D_(1-y)    (x on cations, y on anions)
#                ternaries given as (name, function of x or y -> ternary x)
#  type "ABC-D": A_x B_y C_(1-x-y) D        (three cations)
QUATERNARIES = {
    # Ga_x In_(1-x) As_y P_(1-y)   (telecom lasers on InP)
    "InGaAsP": dict(kind="AB-CD", desc="Ga_x In_(1-x) As_y P_(1-y)",
                    ABC=("InGaAs", lambda x: 1 - x),    # Ga_x In_(1-x) As
                    ABD=("InGaP",  lambda x: 1 - x),    # Ga_x In_(1-x) P
                    ACD=("GaAsP",  lambda y: y),        # Ga As_y P_(1-y)
                    BCD=("InAsP",  lambda y: y)),       # In As_y P_(1-y)
    # Ga_x In_(1-x) As_y Sb_(1-y)   (mid-IR on GaSb / InAs)
    "InGaAsSb": dict(kind="AB-CD", desc="Ga_x In_(1-x) As_y Sb_(1-y)",
                     ABC=("InGaAs", lambda x: 1 - x),
                     ABD=("InGaSb", lambda x: 1 - x),
                     ACD=("GaAsSb", lambda y: 1 - y),
                     BCD=("InAsSb", lambda y: 1 - y)),
    # Al_x Ga_y In_(1-x-y) As   (on InP)
    "AlGaInAs": dict(kind="ABC-D", desc="Al_x Ga_y In_(1-x-y) As",
                     # Vurgaftman: T_ij(u) = i_(1-u) j_u
                     AB=("AlGaAs", lambda u: 1 - u),    # Al_(1-u) Ga_u As
                     BC=("InGaAs", lambda v: v),        # Ga_(1-v) In_v As
                     AC=("InAlAs", lambda w: w)),       # Al_(1-w) In_w As
    # Al_x Ga_y In_(1-x-y) P   (red LEDs/lasers on GaAs)
    "AlGaInP": dict(kind="ABC-D", desc="Al_x Ga_y In_(1-x-y) P",
                    AB=("AlGaP", lambda u: 1 - u),
                    BC=("InGaP", lambda v: v),
                    AC=("InAlP", lambda w: w)),
}

CUSTOM = {}


def add_material(name, Ec, Ev, me, mhh, mlh, eps_s, eps_inf=None, hwLO=0.036,
                 a=5.65325, g=2.0, mhh_par=None, mlh_par=None):
    """Add your own material (band edges in eV, masses in m0, g = electron g-factor)."""
    CUSTOM[name] = dict(Ec=Ec, Ev=Ev, Eg=Ec - Ev, me=me, mhh=mhh, mlh=mlh,
                        mhh_par=mhh_par or mhh, mlh_par=mlh_par or mlh,
                        eps_s=eps_s, eps_inf=eps_inf or eps_s, hwLO=hwLO, a=a,
                        g=g, Ep=np.nan, Dso=np.nan, approx=False)


def _raw_binary(name, T):
    p = dict(zip(_KEYS, BINARIES[name]))
    p["Eg"] = p["Eg0"] - p["alpha"] * T**2 / (T + p["beta"])
    return p


def _raw_ternary(name, x, T):
    A, B, bow, bow_v, bow_m, _ = TERNARIES[name]
    pa, pb = _raw_binary(A, T), _raw_binary(B, T)
    p = {k: x * pa[k] + (1 - x) * pb[k] for k in _KEYS + ("Eg",)}
    p["Eg"] -= x * (1 - x) * bow(x)
    p["VBO"] -= x * (1 - x) * bow_v
    p["me"] -= x * (1 - x) * bow_m
    return p


def _raw_quaternary(name, x, y, T):
    q = QUATERNARIES[name]
    keys = _KEYS + ("Eg",)
    if q["kind"] == "AB-CD":
        def T3(entry, arg):
            tname, f = entry
            return _raw_ternary(tname, f(arg), T)
        tABC, tABD = T3(q["ABC"], x), T3(q["ABD"], x)
        tACD, tBCD = T3(q["ACD"], y), T3(q["BCD"], y)
        wx, wy = x * (1 - x), y * (1 - y)
        if wx + wy < 1e-12:      # corner = binary: pick the matching ternary end
            return tABC if y > 0.5 else tABD
        return {k: (wx * (y * tABC[k] + (1 - y) * tABD[k]) +
                    wy * (x * tACD[k] + (1 - x) * tBCD[k])) / (wx + wy) for k in keys}
    else:   # ABC-D
        zc = 1 - x - y
        if zc < -1e-9:
            raise ValueError(f"{name}: need x + y <= 1")
        u, v, w = (1 - x + y) / 2, (2 - x - 2 * y) / 2, (2 - 2 * x - y) / 2
        tAB = _raw_ternary(q["AB"][0], q["AB"][1](u), T)
        tBC = _raw_ternary(q["BC"][0], q["BC"][1](v), T)
        tAC = _raw_ternary(q["AC"][0], q["AC"][1](w), T)
        wAB, wBC, wAC = x * y, y * zc, x * zc
        s = wAB + wBC + wAC
        if s < 1e-12:            # binary or edge ternary
            if x > 0.999: return _raw_ternary(q["AB"][0], q["AB"][1](0.0), T)   # A binary
            if y > 0.999: return _raw_ternary(q["AB"][0], q["AB"][1](1.0), T)   # B binary
            return _raw_ternary(q["BC"][0], q["BC"][1](1.0), T)                 # C binary
        return {k: (wAB * tAB[k] + wBC * tBC[k] + wAC * tAC[k]) / s for k in keys}


def material(name, x=None, y=None, T=300.0):
    """
    Material parameters at temperature T.
      binaries:     material("GaAs")
      ternaries:    material("AlGaAs", x=0.3)        (x = first element, see list_materials())
      quaternaries: material("InGaAsP", x=0.4, y=0.85)
    Returns dict with Ec, Ev, Eg, me, mhh, mlh (along z), mhh_par, mlh_par
    (in-plane), g (electron g-factor), eps_s, eps_inf, hwLO, a, Ep, Dso.
    """
    if name in CUSTOM:
        return dict(CUSTOM[name], name=name)
    if name in BINARIES:
        p, label = _raw_binary(name, T), name
    elif name in TERNARIES:
        if x is None:
            raise ValueError(f"{name} needs x (= {TERNARIES[name][5]} fraction)")
        p, label = _raw_ternary(name, x, T), f"{name}(x={x:g})"
    elif name in QUATERNARIES:
        if x is None or y is None:
            raise ValueError(f"{name} needs x and y: {QUATERNARIES[name]['desc']}")
        p, label = _raw_quaternary(name, x, y, T), f"{name}(x={x:g},y={y:g})"
    else:
        raise ValueError(f"Unknown material '{name}'. Use list_materials().")

    Eg, Ep, D = p["Eg"], p["Ep"], p["Dso"]
    g = 2.0 - 2.0 * Ep * D / (3.0 * Eg * (Eg + D))       # Roth two-band formula
    approx = any(b in label for b in APPROXIMATE)
    return dict(name=label, Eg=Eg, Ev=p["VBO"], Ec=p["VBO"] + Eg, me=p["me"],
                mhh=1.0 / (p["g1"] - 2 * p["g2"]), mlh=1.0 / (p["g1"] + 2 * p["g2"]),
                mhh_par=1.0 / (p["g1"] + p["g2"]), mlh_par=1.0 / (p["g1"] - p["g2"]),
                eps_s=p["eps_s"], eps_inf=p["eps_inf"], hwLO=p["hwLO"], a=p["a"],
                Ep=Ep, Dso=D, g=g, approx=approx)


def list_materials():
    """Print every material and what its composition parameters mean."""
    print("Binaries:   " + ", ".join(BINARIES) + "   (nitrides: zinc-blende, approximate)")
    print("Ternaries (x = fraction of the first element named):")
    for n, t in TERNARIES.items():
        print(f"   {n:8s} x = {t[5]} fraction   ({t[0]} at x=1, {t[1]} at x=0)")
    print("Quaternaries:")
    for n, q in QUATERNARIES.items():
        print(f"   {n:9s} {q['desc']}")
    if CUSTOM:
        print("Custom:     " + ", ".join(CUSTOM))


def lattice_matched(name, substrate="InP", fixed=None, T=300.0):
    """
    Find the composition lattice-matched to `substrate`.
      ternary:      lattice_matched("InGaAs", "InP")               -> x
      quaternary:   lattice_matched("InGaAsP", "InP", fixed=("y", 0.6)) -> x
    """
    from scipy.optimize import brentq
    a_sub = material(substrate, T=T)["a"]
    if name in TERNARIES:
        f = lambda x: material(name, x, T=T)["a"] - a_sub
        return brentq(f, 0.0, 1.0)
    key, val = fixed
    if key == "y":
        f = lambda x: material(name, x, val, T=T)["a"] - a_sub
        hi = 1.0 - val if QUATERNARIES[name]["kind"] == "ABC-D" else 1.0
    else:
        f = lambda y: material(name, val, y, T=T)["a"] - a_sub
        hi = 1.0 - val if QUATERNARIES[name]["kind"] == "ABC-D" else 1.0
    return brentq(f, 0.0, hi)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
class Layer:
    def __init__(self, material, thickness, x=None, doping=0.0, y=None, label=None):
        self.material = material
        self.thickness = thickness    # nm
        self.x = x
        self.y = y
        self.doping = doping          # donors cm^-3 (negative = acceptors)
        self.label = label

    def params(self, T):
        return material(self.material, self.x, self.y, T)

    def __repr__(self):
        comp = ""
        if self.x is not None:
            comp = f"(x={self.x:g}" + (f",y={self.y:g})" if self.y is not None else ")")
        d = f", Nd={self.doping:.1e}" if self.doping else ""
        return f"{self.material}{comp} {self.thickness:g} nm{d}"


def _spec(m):
    """('GaAs',) / ('AlGaAs', 0.3) / ('InGaAsP', 0.4, 0.85) -> name, x, y"""
    if isinstance(m, str):
        return m, None, None
    m = tuple(m) + (None, None)
    return m[0], m[1], m[2]


class Structure:
    """A stack of layers from left (z=0) to right."""

    def __init__(self, layers=None, substrate="GaAs", name="structure"):
        self.layers = list(layers or [])
        self.substrate = substrate
        self.name = name
        self.period = None

    def add(self, material, thickness, x=None, doping=0.0, y=None):
        self.layers.append(Layer(material, thickness, x, doping, y))
        return self

    def _add_spec(self, spec, thickness):
        n, x, y = _spec(spec)
        return self.add(n, thickness, x, y=y)

    @classmethod
    def single_well(cls, well=("GaAs",), barrier=("AlGaAs", 0.3), Lw=10.0, Lb=20.0, **kw):
        """well/barrier: 'GaAs', ('AlGaAs', 0.3) or ('InGaAsP', 0.4, 0.85)"""
        s = cls(name=kw.pop("name", "single QW"), **kw)
        s._add_spec(barrier, Lb)._add_spec(well, Lw)._add_spec(barrier, Lb)
        return s

    @classmethod
    def multi_well(cls, n_wells, well=("GaAs",), barrier=("AlGaAs", 0.3),
                   Lw=10.0, Lb=5.0, Lb_out=20.0, **kw):
        """
        n_wells wells. Lw, Lb, well, barrier: single values or lists, e.g.
          Lw=[15,10,20], Lb=[5,8], well=[("GaAs",), ("InGaAs",0.1), ...]
        barrier list has n_wells+1 entries (outer + inner barriers).
        """
        Lw = list(np.broadcast_to(Lw, n_wells))
        Lb = list(np.broadcast_to(Lb, max(n_wells - 1, 1)))
        wells = well if isinstance(well, list) else [well] * n_wells
        bars = barrier if isinstance(barrier, list) else [barrier] * (n_wells + 1)
        s = cls(name=kw.pop("name", f"{n_wells}-well structure"), **kw)
        s._add_spec(bars[0], Lb_out)
        for i in range(n_wells):
            s._add_spec(wells[i], float(Lw[i]))
            if i < n_wells - 1:
                s._add_spec(bars[i + 1], float(Lb[i]))
        s._add_spec(bars[-1], Lb_out)
        return s

    @classmethod
    def superlattice(cls, n_periods, well=("GaAs",), barrier=("AlAs",),
                     Lw=5.0, Lb=2.0, Lb_out=20.0, **kw):
        s = cls.multi_well(n_periods, well, barrier, Lw, Lb, Lb_out,
                           name=kw.pop("name", f"{n_periods}-period superlattice"), **kw)
        wn, wx, wy = _spec(well)
        bn, bx, by = _spec(barrier)
        s.period = [Layer(wn, Lw, wx, y=wy), Layer(bn, Lb, bx, y=by)]
        return s

    @property
    def total_thickness(self):
        return sum(l.thickness for l in self.layers)

    def describe(self, T=300.0):
        a_sub = material(self.substrate, T=T)["a"]
        print(f"\nStructure: {self.name}   (total {self.total_thickness:g} nm, T = {T:g} K, "
              f"substrate {self.substrate})")
        print("  #   layer                              Ec (eV)  Ev (eV)  Eg (eV)  me      mhh    mlh    g_e     mismatch")
        for i, l in enumerate(self.layers):
            p = l.params(T)
            mis = (p["a"] - a_sub) / a_sub * 100
            warn = "  <- strained (not modelled)" if abs(mis) > 0.5 else ""
            if p.get("approx"):
                warn += "  [approx. parameters]"
            print(f"  {i:<3d} {str(l):34s} {p['Ec']:8.4f} {p['Ev']:8.4f} {p['Eg']:7.4f} "
                  f"{p['me']:7.4f} {p['mhh']:6.3f} {p['mlh']:6.3f} {p['g']:7.3f} {mis:7.2f}%{warn}")
# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class Solver:
    def __init__(self, structure, T=300.0, field=0.0, dz=0.1, Bz=0.0, Bpar=0.0, z0=None):
        """
        field: electric field along z in kV/cm (positive tilts the bands up to
               the right for electrons).
        Bz:    magnetic field along z (growth axis), tesla -> Landau levels, Zeeman
        Bpar:  magnetic field in the layer plane (along x), tesla -> diamagnetic
               term (hbar k_y + e B z)^2 / 2m.  z0 = -hbar k_y / (e B) is the
               orbit centre (nm); default = centre of the structure.
        dz:    grid step (nm).
        """
        self.s = structure
        self.T = T
        self.field = field
        self.dz = dz
        self.Bz = Bz
        self.Bpar = Bpar
        self.VH = None          # Hartree potential (Schroedinger-Poisson)
        self._build_grid()
        self.z0 = self.z[-1] / 2 if z0 is None else z0

    # ---------------- grid ----------------
    def _build_grid(self):
        L = self.s.total_thickness
        n = int(round(L / self.dz)) + 1
        self.z = np.linspace(0, L, n)
        self.dz = self.z[1] - self.z[0]
        edges = np.cumsum([0] + [l.thickness for l in self.s.layers])
        idx = np.clip(np.searchsorted(edges, self.z, side="right") - 1, 0, len(self.s.layers) - 1)
        self.layer_index = idx
        self.layer_edges = edges

        P = [l.params(self.T) for l in self.s.layers]
        self.params = P
        get = lambda key: np.array([P[i][key] for i in idx])
        self.Ec0, self.Ev0 = get("Ec"), get("Ev")
        self.me, self.mhh, self.mlh = get("me"), get("mhh"), get("mlh")
        self.mhh_par, self.mlh_par = get("mhh_par"), get("mlh_par")
        self.g = get("g")
        self.eps = get("eps_s")
        self.Nd = np.array([self.s.layers[i].doping for i in idx]) * 1e-21   # nm^-3

    @property
    def Ufield(self):
        """potential energy of the field (eV), zero at the structure centre"""
        return self.field * 1e-4 * (self.z - self.z[-1] / 2)

    @property
    def Ec(self):
        v = self.Ec0 + self.Ufield
        return v + self.VH if self.VH is not None else v

    @property
    def Ev(self):
        v = self.Ev0 + self.Ufield
        return v + self.VH if self.VH is not None else v

    # ---------------- Schroedinger (BenDaniel-Duke FDM) ----------------
    def _solve(self, V, m, nstates):
        dz = self.dz
        inv_m = 0.5 * (1 / m[:-1] + 1 / m[1:])
        t = HB2M0 / dz**2
        d = t * (inv_m[:-1] + inv_m[1:]) + V[1:-1]
        o = -t * inv_m[1:-1]
        # exact, deterministic LAPACK solver for the tridiagonal Hamiltonian
        # (ARPACK/eigsh uses a random start vector: results could depend on
        #  the call history when states are nearly degenerate)
        k = min(nstates, len(d) - 2)
        E, psi = eigh_tridiagonal(d, o, select="i", select_range=(0, k - 1))
        full = np.zeros((len(self.z), k))
        full[1:-1] = psi
        full /= np.sqrt(np.sum(full**2, axis=0) * dz)
        # sign convention: largest lobe positive
        for i in range(k):
            if full[np.argmax(np.abs(full[:, i])), i] < 0:
                full[:, i] *= -1
        return E, full

    def solve(self, n_e=6, n_h=6):
        """Electrons, heavy holes, light holes. Hole energies are absolute
        band energies (valence-band side), i.e. E_hole < Ev of the well."""
        self.Ee, self.psi_e = self._solve(self.Ec + self.V_dia(self.me), self.me, n_e)
        Eh, self.psi_hh = self._solve(-self.Ev + self.V_dia(self.mhh_par), self.mhh, n_h)
        El, self.psi_lh = self._solve(-self.Ev + self.V_dia(self.mlh_par), self.mlh, n_h)
        self.Ehh, self.Elh = -Eh, -El
        if self.field == 0:
            # bound = below the lower of the two outer barrier edges
            self.e_bound = self.Ee < min(self.Ec[0], self.Ec[-1])
            self.hh_bound = self.Ehh > max(self.Ev[0], self.Ev[-1])
            self.lh_bound = self.Elh > max(self.Ev[0], self.Ev[-1])
        else:
            # with an electric field the band edges at the two ends differ, so
            # "bound" (quasi-bound) = mostly localised inside the inner layers
            def localised(psi):
                inner = (self.layer_index > 0) & (self.layer_index < len(self.s.layers) - 1)
                return np.sum(psi[inner] ** 2, axis=0) * self.dz > 0.8
            self.e_bound = localised(self.psi_e)
            self.hh_bound = localised(self.psi_hh)
            self.lh_bound = localised(self.psi_lh)
        return self

    # ---------------- state analysis ----------------
    def layer_probabilities(self, psi):
        P = np.zeros(len(self.s.layers))
        for i in range(len(self.s.layers)):
            P[i] = np.sum(psi[self.layer_index == i] ** 2) * self.dz
        return P

    def mass_average(self, psi, m):
        return np.sum(psi**2 * m) * self.dz

    def expect_z(self, psi):
        zm = np.sum(psi**2 * self.z) * self.dz
        dz2 = np.sum(psi**2 * (self.z - zm) ** 2) * self.dz
        return zm, np.sqrt(dz2)

    # ---------------- intersubband ----------------
    def intersubband(self, bound_only=True):
        """list of dicts: i, j, dE (eV), lambda (um), z_ij (nm), f_ij"""
        res = []
        n = len(self.Ee)
        for i in range(n):
            for j in range(i + 1, n):
                if bound_only and not (self.e_bound[i] and self.e_bound[j]):
                    continue
                dE = self.Ee[j] - self.Ee[i]
                zij = np.sum(self.psi_e[:, i] * self.z * self.psi_e[:, j]) * self.dz
                m = 0.5 * (self.mass_average(self.psi_e[:, i], self.me) +
                           self.mass_average(self.psi_e[:, j], self.me))
                f = m * dE * zij**2 / HB2M0          # = 2 m* dE z^2 / hbar^2
                res.append(dict(i=i, j=j, dE=dE, lam_um=1.23984 / dE, z=zij, f=f, m=m))
        return res

    def fermi_level(self, n2d_cm2, states=None):
        """Fermi level (eV) and populations (cm^-2) for sheet density n2d."""
        states = np.arange(len(self.Ee)) if states is None else states
        E = self.Ee[states]
        m = np.array([self.mass_average(self.psi_e[:, i], self.me) for i in states])
        kT = KB * self.T
        n_target = n2d_cm2 * 1e-14                       # nm^-2

        def ntot(EF):
            return np.sum(m * DOS0 * kT * np.logaddexp(0, (EF - E) / kT))

        lo, hi = E.min() - 1.0, E.min() + 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ntot(mid) < n_target else (lo, mid)
        EF = 0.5 * (lo + hi)
        pops = m * DOS0 * kT * np.logaddexp(0, (EF - E) / kT) * 1e14   # cm^-2
        if E.max() < EF + 5 * kT:
            import warnings
            warnings.warn("fermi_level: the highest computed state is less than 5 kT above "
                          "the Fermi level; call solve() with more states (n_e) for accurate "
                          "populations.", RuntimeWarning, stacklevel=2)
        return EF, pops

    def isb_absorption(self, energies, n2d_cm2=1e11, gamma=0.005):
        """
        Absolute absorption per pass (fraction) for light polarised along z,
        Lorentzian broadening gamma (eV, HWHM). Returns array like energies.
            A(E) = sum dN * hbar*pi*e^2 f /(2 eps0 m* c n_r) * L(E)
        """
        EF, pops = self.fermi_level(n2d_cm2)
        nr = np.sqrt(np.mean([p["eps_inf"] for p in self.params]))
        A = np.zeros_like(energies, dtype=float)
        for t in self.intersubband():
            dN = (pops[t["i"]] - pops[t["j"]]) * 1e4             # m^-2
            L = gamma / np.pi / ((energies - t["dE"]) ** 2 + gamma**2) / Q_SI   # 1/J
            pref = HBAR_SI * np.pi * Q_SI**2 * t["f"] / (2 * EPS0_SI * t["m"] * M0_SI * C_SI * nr)
            A += dN * pref * L
        return A

    # ---------------- interband ----------------
    def interband(self, bound_only=True):
        res = []
        for band, Eh, psih, bnd, w in (("hh", self.Ehh, self.psi_hh, self.hh_bound, 0.75),
                                       ("lh", self.Elh, self.psi_lh, self.lh_bound, 0.25)):
            for i in range(len(self.Ee)):
                for j in range(len(Eh)):
                    if bound_only and not (self.e_bound[i] and bnd[j]):
                        continue
                    ov = np.sum(self.psi_e[:, i] * psih[:, j]) * self.dz
                    E = self.Ee[i] - Eh[j]
                    res.append(dict(e=i, h=j, band=band, E=E, lam_nm=1239.84 / E,
                                    overlap2=ov**2, weight=w))
        return sorted(res, key=lambda r: r["E"])

    def interband_absorption(self, energies, gamma=0.005):
        """relative units: sum weight*|<e|h>|^2 * broadened 2D step"""
        A = np.zeros_like(energies, dtype=float)
        for t in self.interband():
            A += t["weight"] * t["overlap2"] * (0.5 + np.arctan((energies - t["E"]) / gamma) / np.pi)
        return A

    # ---------------- LO phonon scattering ----------------
    def lo_phonon_rates(self):
        """
        Intersubband LO-phonon EMISSION rates from the bottom of subband i to f
        (Ferreira & Bastard, PRB 40, 1074 (1989)), bulk-like phonons of the
        material where state i is mostly localised. Returns list of dicts.
        """
        res = []
        kT = KB * self.T
        stride = max(1, len(self.z) // 1500)
        zs = self.z[::stride]
        dzs = self.dz * stride
        D = np.abs(zs[:, None] - zs[None, :]) * 1e-9          # m
        for i in range(len(self.Ee)):
            if not self.e_bound[i]:
                continue
            lay = int(np.argmax(self.layer_probabilities(self.psi_e[:, i])))
            p = self.params[lay]
            hw = p["hwLO"]
            inv_ep = 1 / p["eps_inf"] - 1 / p["eps_s"]
            nLO = 1 / np.expm1(hw / kT)
            m = self.mass_average(self.psi_e[:, i], self.me)
            for f in range(i):
                dE = self.Ee[i] - self.Ee[f]
                if dE <= hw:
                    continue
                q0 = np.sqrt(2 * m * M0_SI * (dE - hw) * Q_SI) / HBAR_SI   # 1/m
                rho = self.psi_e[::stride, i] * self.psi_e[::stride, f]      # nm^-1
                Aq = rho @ np.exp(-q0 * D) @ rho * dzs**2                     # dimensionless
                # Fermi golden rule + Froehlich coupling, in-plane k_i = 0:
                # 1/tau = m* e^2 w_LO (1/eps_inf - 1/eps_s) A(q0) (n+1) / (4 eps0 hbar^2 q0)
                rate = (m * M0_SI * Q_SI**2 * (hw * Q_SI / HBAR_SI) * inv_ep /
                        (4 * HBAR_SI**2 * EPS0_SI)) * Aq / q0 * (nLO + 1)
                res.append(dict(i=i, f=f, dE=dE, rate=rate, tau_ps=1e12 / rate))
        return res

    # ---------------- scattering: transfer matrix ----------------
    def transmission(self, energies, V=None, m=None):
        """
        Electron transmission T(E) through the whole structure (transfer matrix
        on the FDM grid, BenDaniel-Duke matching, leads = first/last layers).
        """
        V = self.Ec if V is None else V
        m = self.me if m is None else m
        E = np.asarray(energies, dtype=complex) + 1e-9j
        M11 = np.ones_like(E); M12 = np.zeros_like(E)
        M21 = np.zeros_like(E); M22 = np.ones_like(E)
        for Vj, mj in zip(V, m):
            k = np.sqrt(mj * (E - Vj) / HB2M0)
            c, s = np.cos(k * self.dz), np.sin(k * self.dz)
            a11, a12, a21, a22 = c, mj * s / k, -k * s / mj, c
            M11, M12, M21, M22 = (a11 * M11 + a12 * M21, a11 * M12 + a12 * M22,
                                  a21 * M11 + a22 * M21, a21 * M12 + a22 * M22)
        kL = np.sqrt(m[0] * (E - V[0]) / HB2M0)
        kR = np.sqrt(m[-1] * (E - V[-1]) / HB2M0)
        gL, gR = 1j * kL / m[0], 1j * kR / m[-1]
        r = (gR * (M11 + M12 * gL) - (M21 + M22 * gL)) / ((M21 - M22 * gL) - gR * (M11 - M12 * gL))
        t = (M11 + M12 * gL) + r * (M11 - M12 * gL)
        T = np.real(kR / m[-1]) / np.real(kL / m[0]) * np.abs(t) ** 2
        T[(np.real(energies) <= V[0]) | (np.real(energies) <= V[-1])] = 0.0
        return np.clip(np.real(T), 0, 1)

    # ---------------- superlattice minibands ----------------
    def minibands(self, energies, period=None, dz=0.02):
        """
        Bloch condition cos(q d) = Tr(M_period)/2 for an infinite superlattice
        made of `period` (list of Layers; default = structure.period).
        Returns q (1/nm, NaN in gaps), d (nm), list of (Emin, Emax) bands.
        """
        period = period or self.s.period
        if period is None:
            raise ValueError("No period defined (use Structure.superlattice or pass period=[...])")
        V, m = [], []
        for l in period:
            p = l.params(self.T)
            n = max(1, int(round(l.thickness / dz)))
            V += [p["Ec"]] * n
            m += [p["me"]] * n
        dz_eff = sum(l.thickness for l in period) / len(V)
        d = sum(l.thickness for l in period)
        E = np.asarray(energies, dtype=complex) + 1e-12j
        M11 = np.ones_like(E); M12 = np.zeros_like(E)
        M21 = np.zeros_like(E); M22 = np.ones_like(E)
        for Vj, mj in zip(V, m):
            k = np.sqrt(mj * (E - Vj) / HB2M0)
            c, s = np.cos(k * dz_eff), np.sin(k * dz_eff)
            a11, a12, a21, a22 = c, mj * s / k, -k * s / mj, c
            M11, M12, M21, M22 = (a11 * M11 + a12 * M21, a11 * M12 + a12 * M22,
                                  a21 * M11 + a22 * M21, a21 * M12 + a22 * M22)
        half_tr = np.real(0.5 * (M11 + M22))
        allowed = np.abs(half_tr) <= 1
        q = np.where(allowed, np.arccos(np.clip(half_tr, -1, 1)) / d, np.nan)
        bands, inside = [], False
        for e, a in zip(energies, allowed):
            if a and not inside:
                start, inside = e, True
            if not a and inside:
                bands.append((start, prev)); inside = False
            prev = e
        if inside:
            bands.append((start, prev))
        return q, d, bands

    # ---------------- Schroedinger-Poisson ----------------
    def schrodinger_poisson(self, n_e=8, mix=0.1, tol=1e-5, max_iter=600, verbose=True,
                            right_bc="neumann"):
        """
        Self-consistent electron density for the layer dopings (donors fully
        ionised). Electrons from donors fill the subbands (charge neutral).

        The number of states is increased automatically until the highest
        state lies more than 10 kT above the Fermi level (no truncation error).

        Boundary conditions for the Hartree potential:
          left end:  V = 0 (reference)
          right end: right_bc = "neumann"   zero field (default; correct for a
                                            thick undoped buffer, e.g. HEMTs)
                     right_bc = "dirichlet" V = 0 (symmetric, thin outer layers)
        """
        n2d = np.sum(self.Nd) * self.dz * 1e14            # cm^-2
        if n2d <= 0:
            raise ValueError("No donors in the structure")
        self.VH = np.zeros_like(self.z)
        dz = self.dz
        eps_half = 0.5 * (self.eps[:-1] + self.eps[1:])
        from scipy.sparse.linalg import splu
        if right_bc == "dirichlet":
            main = (eps_half[:-1] + eps_half[1:]) / dz**2
            off = -eps_half[1:-1] / dz**2
            sl = slice(1, -1)                              # unknowns: interior nodes
        elif right_bc == "neumann":
            # unknowns: nodes 1..N-1; last node is a half cell with zero outward field
            main = np.append((eps_half[:-1] + eps_half[1:]) / dz**2, 2 * eps_half[-1] / dz**2)
            off_up = -eps_half[1:] / dz**2
            off_lo = off_up.copy()
            off_lo[-1] *= 2                                 # half-cell row
            sl = slice(1, None)
        else:
            raise ValueError("right_bc must be 'neumann' or 'dirichlet'")
        if right_bc == "dirichlet":
            A = sp.diags([off, main, off], [-1, 0, 1], format="csc")
        else:
            A = sp.diags([off_lo, main, off_up], [-1, 0, 1], format="csc")
        lu = splu(A)
        kT = KB * self.T
        for it in range(max_iter):
            self.Ee, self.psi_e = self._solve(self.Ec, self.me, n_e)
            EF, pops = self.fermi_level(n2d)
            while self.Ee[-1] < EF + 10 * kT and n_e < 200:   # enough empty states?
                n_e += 4
                self.Ee, self.psi_e = self._solve(self.Ec, self.me, n_e)
                EF, pops = self.fermi_level(n2d)
            n3d = np.sum(pops * 1e-14 * self.psi_e**2, axis=1)       # nm^-3
            rhs = POISSON * (n3d - self.Nd)
            Vnew = np.zeros_like(self.z)
            Vnew[sl] = lu.solve(rhs[sl])             # -d/dz(eps dV/dz) = rhs  -> V
            err = np.max(np.abs(Vnew - self.VH))
            self.VH = (1 - mix) * self.VH + mix * Vnew
            if err < tol:
                break
        self.EF, self.n3d = EF, n3d
        if verbose:
            print(f"Schroedinger-Poisson: {it+1} iterations, change {err:.2e} eV, "
                  f"n2D = {n2d:.3e} cm^-2, EF = {EF:.4f} eV")
        return self


    # ---------------- magnetic field ----------------
    def V_dia(self, m_par):
        """In-plane field: (hbar k_y + e B z)^2/(2 m_par) = hbar^2 (z-z0)^2 / (2 m l_B^4)."""
        if self.Bpar == 0:
            return 0.0
        lB = L_B1 / np.sqrt(abs(self.Bpar))
        return HB2M0 / m_par * (self.z - self.z0) ** 2 / lB**4

    def g_factor(self, n):
        """electron g-factor of subband n: average of g(z) over |psi|^2"""
        return np.sum(self.psi_e[:, n] ** 2 * self.g) * self.dz

    def zeeman(self, B=None):
        """spin splitting g mu_B B (eV) for each electron subband (total |B|)"""
        B = np.hypot(self.Bz, self.Bpar) if B is None else B
        return np.array([self.g_factor(n) * MU_B * B for n in range(len(self.Ee))])

    def landau_levels(self, B=None, N_max=10, bound_only=True):
        """
        Landau levels for B along z:
          E = E_n + hbar w_c,n (N + 1/2) + s g_n mu_B B,   s = +-1/2
        Returns list of dicts sorted by energy.
        """
        B = self.Bz if B is None else B
        out = []
        for n in range(len(self.Ee)):
            if bound_only and not self.e_bound[n]:
                continue
            m = self.mass_average(self.psi_e[:, n], self.me)
            hwc = HBAR_E_M0 * B / m
            gz = self.g_factor(n) * MU_B * B
            for N in range(N_max + 1):
                for s in (+0.5, -0.5):
                    out.append(dict(n=n, N=N, s=s, E=self.Ee[n] + hwc * (N + 0.5) + s * gz,
                                    hwc=hwc))
        return sorted(out, key=lambda d: d["E"])

    def landau_dos(self, energies, B=None, gamma=0.001, N_max=60):
        """Landau-level DOS (cm^-2 eV^-1), Gaussian broadening gamma (eV)"""
        B = self.Bz if B is None else B
        dos = np.zeros_like(energies, dtype=float)
        deg = EB_H * B                      # states per cm^2 per spin level
        for L in self.landau_levels(B, N_max):
            dos += deg * np.exp(-(energies - L["E"]) ** 2 / (2 * gamma**2)) / (np.sqrt(2 * np.pi) * gamma)
        return dos

    def fermi_level_B(self, n2d_cm2, B=None, gamma=0.001, N_max=60):
        """Fermi level (eV) for sheet density n2d in field B along z."""
        B = self.Bz if B is None else B
        if B == 0:
            return self.fermi_level(n2d_cm2)[0]
        kT = KB * self.T
        e = np.linspace(self.Ee[0] - 0.05, self.Ee[0] + 0.3, 6000)
        dos = self.landau_dos(e, B, gamma, N_max)
        de = e[1] - e[0]

        def n(EF):
            return np.sum(dos / (1 + np.exp(np.clip((e - EF) / kT, -60, 60)))) * de

        lo, hi = e[0], e[-1]
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if n(mid) < n2d_cm2 else (lo, mid)
        return 0.5 * (lo + hi)

    def landau_fan(self, B_values, n2d_cm2=None, N_max=8, gamma=0.001):
        """Landau levels and (optional) Fermi level / filling factor versus B."""
        levels, EF, nu = [], [], []
        for B in B_values:
            levels.append([L["E"] for L in self.landau_levels(B, N_max)])
            if n2d_cm2:
                EF.append(self.fermi_level_B(n2d_cm2, B, gamma))
                nu.append(n2d_cm2 / (EB_H * B) if B > 0 else np.inf)
        return np.array(levels), np.array(EF), np.array(nu)

    def inplane_dispersion(self, ky_values, n_e=4):
        """
        In-plane field Bpar: subband energies versus k_y (1/nm).
        Orbit centre z0 = -k_y l_B^2 (measured from the structure centre).
        """
        if self.Bpar == 0:
            raise ValueError("Set Bpar (tesla) for the in-plane dispersion")
        lB = L_B1 / np.sqrt(abs(self.Bpar))
        zc, out = self.z[-1] / 2, []
        z0_save = self.z0
        for ky in ky_values:
            self.z0 = zc - ky * lB**2
            E, _ = self._solve(self.Ec + self.V_dia(self.me), self.me, n_e)
            out.append(E)
        self.z0 = z0_save
        return np.array(out)

    # ---------------- electric field sweep ----------------
    def active_states(self, psi, threshold=0.5):
        """
        Indices of states with > threshold probability inside the inner layers
        (all layers except the first and last). With a strong field the tilted
        outer barriers can trap spurious states near the boundaries; this
        picks the states that really belong to the wells.
        """
        inner = (self.layer_index > 0) & (self.layer_index < len(self.s.layers) - 1)
        P = np.sum(psi[inner] ** 2, axis=0) * self.dz
        return np.where(P > threshold)[0]
    def stark_sweep(self, fields, n_e=4, n_h=4):
        """
        Repeat the calculation for each field (kV/cm).
        Note: at strong fields weakly bound states (higher electron levels,
        light holes) become tunnelling resonances; in the finite box they
        anticross with continuum states of the tilted barriers, so values
        beyond that field depend on the box. Check with transmission().
        Returns dict with
        electron / hh / lh levels, e0-hh0 and e0-lh0 transition energies,
        their overlaps |<e0|h0>|^2, ISB 0->1 energy and dipole.
        """
        res = dict(F=np.asarray(fields, float), Ee=[], Ehh=[], Elh=[], E_e0hh0=[],
                   ov_e0hh0=[], E_e0lh0=[], ov_e0lh0=[], E01=[], z01=[], zmean_e0=[], zmean_hh0=[])
        # For each field take the LOWEST states localised inside the wells
        # (active_states). This ignores spurious states that the tilted outer
        # barriers create near the box edges, and always finds the true
        # (field-shifted) ground state, also in wide wells.
        for F in fields:
            sv = copy.copy(self)
            sv.field = F
            sv.solve(n_e + 4, n_h + 4)          # extra states: spurious ones are skipped
            ie = sv.active_states(sv.psi_e)
            ihh = sv.active_states(sv.psi_hh)
            ilh = sv.active_states(sv.psi_lh)
            e0, e1, ih, il = ie[0], ie[1], ihh[0], ilh[0]
            res["Ee"].append(sv.Ee[ie[:n_e]]); res["Ehh"].append(sv.Ehh[ihh[:n_h]])
            res["Elh"].append(sv.Elh[ilh[:n_h]])
            res["E_e0hh0"].append(sv.Ee[e0] - sv.Ehh[ih])
            res["ov_e0hh0"].append((np.sum(sv.psi_e[:, e0] * sv.psi_hh[:, ih]) * sv.dz) ** 2)
            res["E_e0lh0"].append(sv.Ee[e0] - sv.Elh[il])
            res["ov_e0lh0"].append((np.sum(sv.psi_e[:, e0] * sv.psi_lh[:, il]) * sv.dz) ** 2)
            res["E01"].append(sv.Ee[e1] - sv.Ee[e0])
            res["z01"].append(np.sum(sv.psi_e[:, e0] * sv.z * sv.psi_e[:, e1]) * sv.dz)
            res["zmean_e0"].append(sv.expect_z(sv.psi_e[:, e0])[0])
            res["zmean_hh0"].append(sv.expect_z(sv.psi_hh[:, ih])[0])
        for k in res:
            if k not in ("Ee", "Ehh", "Elh"):
                res[k] = np.array(res[k])
        return res

    def plot_stark(self, filename, fields, n_e=4, n_h=4):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r = self.stark_sweep(fields, n_e, n_h)
        F = r["F"]
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].plot(F, (r["E_e0hh0"] - r["E_e0hh0"][np.argmin(np.abs(F))]) * 1000, "-o", ms=3, label="e0-hh0")
        ax[0].plot(F, (r["E_e0lh0"] - r["E_e0lh0"][np.argmin(np.abs(F))]) * 1000, "-s", ms=3, label="e0-lh0")
        ax[0].set_xlabel("Field (kV/cm)"); ax[0].set_ylabel("Stark shift (meV)")
        ax[0].set_title("Quantum-confined Stark effect"); ax[0].legend()
        ax[1].plot(F, r["ov_e0hh0"], "-o", ms=3, label="|<e0|hh0>|$^2$")
        ax[1].plot(F, r["ov_e0lh0"], "-s", ms=3, label="|<e0|lh0>|$^2$")
        ax[1].set_xlabel("Field (kV/cm)"); ax[1].set_ylabel("Overlap"); ax[1].legend()
        ax[1].set_title("Electron-hole overlap")
        ax[2].plot(F, r["E01"] * 1000, "-o", ms=3, color="k")
        ax2 = ax[2].twinx()
        ax2.plot(F, np.abs(r["z01"]), "--", color="C3")
        ax[2].set_xlabel("Field (kV/cm)"); ax[2].set_ylabel("E1 - E0 (meV)")
        ax2.set_ylabel("|z01| (nm)", color="C3"); ax[2].set_title("Intersubband 0->1")
        plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close(fig)
        return r

    def plot_magnetic(self, filename, B_values, n2d_cm2=5e11, gamma=0.001, ky_max=0.2):
        """Landau fan + Fermi level (Bz) and, if Bpar != 0, in-plane dispersion."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lev, EF, nu = self.landau_fan(B_values, n2d_cm2, N_max=8, gamma=gamma)
        ncol = 3 if self.Bpar else 2
        fig, ax = plt.subplots(1, ncol, figsize=(5 * ncol, 4))
        E0 = self.Ee[0]
        for k in range(lev.shape[1]):
            ax[0].plot(B_values, (lev[:, k] - E0) * 1000, color="gray", lw=0.6)
        ax[0].plot(B_values, (EF - E0) * 1000, "r", lw=1.5, label="Fermi level")
        ax[0].set_ylim(-5, max((EF - E0).max() * 1000 * 2.5, 20))
        ax[0].set_xlabel("B$_z$ (T)"); ax[0].set_ylabel("E - E$_0$ (meV)")
        ax[0].set_title(f"Landau fan (n$_{{2D}}$={n2d_cm2:.1e} cm$^{{-2}}$)"); ax[0].legend()
        # DOS at the Fermi level (SdH-like oscillation)
        dosEF = [self.landau_dos(np.array([e]), B, gamma)[0] for e, B in zip(EF, B_values)]
        ax[1].plot(1 / np.asarray(B_values), np.array(dosEF) / 1e13, "k")
        ax[1].set_xlabel("1/B (1/T)"); ax[1].set_ylabel("DOS at E$_F$ (10$^{13}$ cm$^{-2}$eV$^{-1}$)")
        ax[1].set_title("DOS at E$_F$ (periodic in 1/B)")
        if self.Bpar:
            ky = np.linspace(-ky_max, ky_max, 41)
            E = self.inplane_dispersion(ky)
            for n in range(E.shape[1]):
                ax[2].plot(ky, (E[:, n] - E.min()) * 1000)
            ax[2].set_xlabel("k$_y$ (1/nm)"); ax[2].set_ylabel("E (meV)")
            ax[2].set_title(f"In-plane B = {self.Bpar} T: dispersion")
        plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close(fig)

    # ---------------- report ----------------
    def report(self, n2d_cm2=None):
        s = self
        print(f"\n{'='*78}\nRESULTS  ({self.s.name}, T = {self.T:g} K, F = {self.field:g} kV/cm, "
              f"Bz = {self.Bz:g} T, Bpar = {self.Bpar:g} T)\n{'='*78}")
        Ecmin = self.Ec.min()

        def table(title, E, psi, bound, ref, sign):
            print(f"\n{title}")
            print("  n   E (eV)     E-edge (meV)  bound   <z> (nm)  dz (nm)   main layers")
            for n in range(len(E)):
                P = s.layer_probabilities(psi[:, n])
                top = np.argsort(P)[::-1][:2]
                where = ", ".join(f"L{k} {P[k]*100:.0f}%" for k in top if P[k] > 0.05)
                zm, sz = s.expect_z(psi[:, n])
                print(f"  {n:<3d}{E[n]:9.4f}   {sign*(E[n]-ref)*1000:10.2f}    {'yes' if bound[n] else 'no ':5s}"
                      f"  {zm:8.2f}  {sz:7.2f}   {where}")

        table("Electron subbands (E-edge from lowest Ec)", self.Ee, self.psi_e, self.e_bound, Ecmin, 1)
        table("Heavy-hole subbands (E-edge from highest Ev, positive = down)",
              self.Ehh, self.psi_hh, self.hh_bound, self.Ev.max(), -1)
        table("Light-hole subbands", self.Elh, self.psi_lh, self.lh_bound, self.Ev.max(), -1)

        isb = self.intersubband()
        if isb:
            print("\nIntersubband transitions (bound states)")
            print("  i->j   dE (meV)   lambda (um)   z_ij (nm)   f_ij")
            for t in isb:
                if t["f"] < 1e-3:            # skip forbidden / negligible transitions
                    continue
                print(f"  {t['i']}->{t['j']}  {t['dE']*1000:9.2f}   {t['lam_um']:10.2f}   {t['z']:9.3f}  {t['f']:7.3f}")

        ib = self.interband()
        if ib:
            print("\nInterband transitions (lowest 8)")
            print("  e - h      E (eV)   lambda (nm)  |<e|h>|^2")
            for t in ib[:8]:
                print(f"  e{t['e']}-{t['band']}{t['h']}   {t['E']:8.4f}   {t['lam_nm']:9.1f}   {t['overlap2']:8.4f}")

        lo = self.lo_phonon_rates()
        if lo:
            print("\nLO-phonon emission (from subband bottom)")
            print("  i->f   dE (meV)   rate (1/s)    tau (ps)")
            for t in lo:
                if t["rate"] < 1e9:          # skip negligible (spatially separated states)
                    continue
                print(f"  {t['i']}->{t['f']}  {t['dE']*1000:9.2f}   {t['rate']:10.3e}   {t['tau_ps']:8.3f}")

        print("\nElectron g-factors and spin splitting")
        print("  n   g_e      Zeeman (ueV) at |B|")
        Bt = np.hypot(self.Bz, self.Bpar)
        for n in range(len(self.Ee)):
            if self.e_bound[n]:
                print(f"  {n:<3d}{self.g_factor(n):7.3f}   {self.g_factor(n)*MU_B*Bt*1e6:9.2f}")
        if self.Bz:
            print(f"\nLandau levels (Bz = {self.Bz} T), lowest 10:")
            print("  subband N  spin    E (eV)     hbar*wc (meV)")
            for L in self.landau_levels(N_max=10)[:10]:
                print(f"  {L['n']:<7d}{L['N']:<3d}{'+' if L['s']>0 else '-':5s} {L['E']:9.5f}   {L['hwc']*1000:8.3f}")
            if n2d_cm2:
                print(f"  filling factor nu = {n2d_cm2/(EB_H*self.Bz):.3f},  "
                      f"EF = {self.fermi_level_B(n2d_cm2):.5f} eV")

        if n2d_cm2:
            EF, pops = self.fermi_level(n2d_cm2)
            print(f"\nSheet density {n2d_cm2:.2e} cm^-2:  EF = {EF:.4f} eV "
                  f"({(EF-self.Ee[0])*1000:.2f} meV above e0)")
            for i, p in enumerate(pops):
                if p > 1e6:
                    print(f"   subband {i}: {p:.3e} cm^-2")

    # ---------------- plots ----------------
    def plot_bands(self, ax, n_show=4, scale=0.05):
        ax.plot(self.z, self.Ec, "k", lw=1.2)
        ax.plot(self.z, self.Ev, "k", lw=1.2)
        for n in range(min(n_show, len(self.Ee))):
            if self.e_bound[n]:
                ax.plot(self.z, self.Ee[n] + scale * self.psi_e[:, n] ** 2 / np.max(self.psi_e[:, n] ** 2), lw=1)
        for n in range(min(n_show, len(self.Ehh))):
            if self.hh_bound[n]:
                ax.plot(self.z, self.Ehh[n] - scale * self.psi_hh[:, n] ** 2 / np.max(self.psi_hh[:, n] ** 2), lw=1)
        ax.set_xlabel("z (nm)"); ax.set_ylabel("Energy (eV)")
        ax.set_title(self.s.name)

    def plot_summary(self, filename, n2d_cm2=1e11, gamma=0.005):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

        # 1 band diagram
        self.plot_bands(ax[0, 0])

        # 2 electron states zoom
        a = ax[0, 1]
        a.plot(self.z, self.Ec, "k")
        for n in range(len(self.Ee)):
            if self.e_bound[n]:
                a.plot(self.z, self.Ee[n] + 0.4 * (self.Ec.max() - self.Ec.min()) / len(self.Ee) *
                       self.psi_e[:, n] / np.abs(self.psi_e[:, n]).max(), lw=1, label=f"e{n}")
        a.set_ylim(self.Ec.min() - 0.02, self.Ec.max() + 0.05)
        a.set_xlabel("z (nm)"); a.set_ylabel("Energy (eV)"); a.set_title("Electron wavefunctions")
        a.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.8)

        # 3 transmission
        a = ax[0, 2]
        Elead = max(self.Ec[0], self.Ec[-1])          # T = 0 below the lead band edge
        Es = np.linspace(Elead + 1e-4, Elead + 0.4, 4000)
        T = self.transmission(Es)
        a.semilogy(Es, np.maximum(T, 1e-12), "k")
        a.set_xlabel("Energy (eV)"); a.set_ylabel("T(E)"); a.set_title("Electron transmission")

        # 4 ISB absorption
        a = ax[1, 0]
        isb = self.intersubband()
        if isb:
            Emax = max(t["dE"] for t in isb) * 1.4
            e = np.linspace(0.001, Emax, 2000)
            a.plot(e * 1000, self.isb_absorption(e, n2d_cm2, gamma) * 100, "k")
        a.set_xlabel("Photon energy (meV)"); a.set_ylabel("Absorption per pass (%)")
        a.set_title(f"Intersubband (n2D={n2d_cm2:.0e} cm$^{{-2}}$, z-pol.)")

        # 5 interband absorption
        a = ax[1, 1]
        ib = self.interband()
        if ib:
            e = np.linspace(ib[0]["E"] - 0.05, ib[0]["E"] + 0.35, 2000)
            a.plot(e, self.interband_absorption(e, gamma), "k")
        a.set_xlabel("Photon energy (eV)"); a.set_ylabel("Absorption (arb.)")
        a.set_title("Interband (TE, hh:lh = 3:1)")

        # 6 minibands or DOS
        a = ax[1, 2]
        if self.s.period is not None:
            p0 = self.s.period[0].params(self.T)
            e = np.linspace(p0["Ec"] + 1e-4, p0["Ec"] + 0.8, 4000)
            q, d, bands = self.minibands(e)
            a.plot(q * d / np.pi, e, "k.", ms=1.5)
            for (b0, b1) in bands:
                a.axhspan(b0, b1, color="C0", alpha=0.15)
            a.set_xlabel(r"$q d/\pi$"); a.set_ylabel("Energy (eV)"); a.set_title("Superlattice minibands")
        else:
            e = np.linspace(self.Ec.min(), max(self.Ec[0], self.Ec[-1]), 2000)
            dos = np.zeros_like(e)
            for n in range(len(self.Ee)):
                if self.e_bound[n]:
                    m = self.mass_average(self.psi_e[:, n], self.me)
                    dos += (e >= self.Ee[n]) * m * DOS0 * 1e14
            a.plot(e, dos, "k")
            a.set_xlabel("Energy (eV)"); a.set_ylabel(r"2D DOS (cm$^{-2}$ eV$^{-1}$)")
            a.set_title("Electron density of states")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close(fig)


