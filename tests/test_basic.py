import numpy as np
from scipy.optimize import brentq
from hetero1d import Structure, Solver, material
from hetero1d.core import HB2M0


def test_finite_well_matches_analytic():
    w, b = material("GaAs"), material("AlGaAs", 0.3)
    V0, L = b["Ec"] - w["Ec"], 10.0

    def f(E):
        k = np.sqrt(w["me"] * E / HB2M0)
        kap = np.sqrt(b["me"] * (V0 - E) / HB2M0)
        return k / w["me"] * np.tan(k * L / 2) - kap / b["me"]

    E_exact = brentq(f, 1e-4, 0.055)
    sol = Solver(Structure.single_well(Lw=10, Lb=25), dz=0.05).solve()
    assert abs((sol.Ee[0] - w["Ec"]) - E_exact) < 1e-4


def test_quaternary_reduces_to_ternary():
    a = material("AlGaInAs", 0.3, 0.7)["Eg"]
    b = material("AlGaAs", 0.3)["Eg"]
    assert abs(a - b) < 1e-10


def test_landau_spacing():
    sol = Solver(Structure.single_well(Lw=10, Lb=25), T=4.2, Bz=5).solve()
    L = sol.landau_levels()
    m = sol.mass_average(sol.psi_e[:, 0], sol.me)
    assert abs(L[0]["hwc"] - 1.1576766e-4 * 5 / m) < 1e-9


def test_exciton_limits_and_trend():
    """binding between bulk Ry and 2D 4Ry, and stronger in narrower wells"""
    E = []
    for Lw in (4, 10, 20):
        sol = Solver(Structure.single_well(Lw=Lw, Lb=30), T=4.2, dz=0.05).solve(2, 2)
        x = sol.exciton()
        mu, eps = x["mu"], x["eps"]
        Ry = 1.439964**2 * mu / (4 * HB2M0 * eps**2) * 1000      # meV
        assert Ry < x["Eb_meV"] < 4 * Ry
        E.append(x["Eb_meV"])
    assert E[0] > E[1] > E[2]


def test_potentials_match_exact_solutions():
    """V_ext shapes: harmonic oscillator and triangular (Airy) well"""
    from scipy.special import ai_zeros
    from hetero1d import potentials
    box = Structure().add("GaAs", 200)
    sol = Solver(box, V_ext=potentials.parabolic(10, center=100)).solve(4, 1)
    E = (sol.Ee - sol.Ec.min()) * 1000
    assert np.allclose(E[:4], [5, 15, 25, 35], atol=0.01)
    sol = Solver(Structure().add("GaAs", 150), dz=0.05, V_ext=potentials.triangular(50)).solve(3, 1)
    E = (sol.Ee - sol.Ec.min()) * 1000
    exact = (HB2M0 / 0.067) ** (1 / 3) * (50e-4) ** (2 / 3) * (-ai_zeros(3)[0]) * 1000
    assert np.allclose(E[:3], exact, atol=0.01)


def test_graded_geometries():
    """builders: mirror symmetry, equal-width wells degenerate, holes confined"""
    lev = lambda s: (Solver(s).solve(3, 3).Ee[:3])
    a = lev(Structure.triangular_well(Lw=20))
    b = lev(Structure.triangular_well(Lw=20, wall="right"))
    assert np.allclose(a, b, atol=1e-6)
    sol = Solver(Structure.multi_graded_well(3, "parabolic", Lw=20, gap=5)).solve(3, 3)
    assert (sol.Ee[2] - sol.Ee[0]) * 1000 < 0.1            # three nearly degenerate levels
    sol = Solver(Structure.graded_well("cosine", Lw=30)).solve(2, 2)
    assert (sol.Ee[1] - sol.Ee[0]) * 1000 > 10              # a real single well, not a double well
    assert sol.hh_bound[0]                                    # holes confined in graded wells


def test_strain():
    """biaxial strain: zero when lattice-matched, correct signs, InGaAs/GaAs gap, critical thickness"""
    from hetero1d import material, lattice_matched
    from hetero1d.strain import layer_strain
    d = layer_strain(material("InGaAs", lattice_matched("InGaAs", "InP")), material("InP")["a"])
    assert abs(d["exx"]) < 1e-6 and abs(d["dEc"]) < 1e-6
    asub = material("GaAs")["a"]
    m = material("InGaAs", 0.2)
    d = layer_strain(m, asub)
    assert d["exx"] < 0 and d["dEhh"] > d["dElh"]                         # compressive: heavy hole on top
    assert 1.18 < m["Eg"] + d["dEc"] - d["dEhh"] < 1.21                  # strained gap ~1.20 eV
    assert 9 < d["hc"] < 14                                              # buried In0.2GaAs: ~11 nm
    t = layer_strain(material("InGaAs", 0.4), material("InP")["a"])
    assert t["exx"] > 0 and t["dElh"] > t["dEhh"]                         # tensile: light hole on top
    s = Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=10, Lb=25)
    E0 = Solver(s).solve(1, 1).Ee[0]
    assert abs(E0 - Solver(s, strain=False).solve(1, 1).Ee[0]) < 1e-12   # default unchanged


def test_nonparabolic():
    """energy-dependent mass: default unchanged, matches the exact solution, lowers high levels"""
    s = Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=10, Lb=30)
    par = Solver(s, dz=0.02).solve(3, 1)
    npb = Solver(s, dz=0.02, nonparabolic=True).solve(3, 1)
    E_par = (par.Ee - par.Ec.min()) * 1000
    E_np = (npb.Ee - npb.Ec.min()) * 1000
    assert abs(E_np[1] - 117.45) < 0.05          # exact energy-dependent BDD solution
    assert abs(E_np[0] - 31.41) < 0.05
    assert E_np[1] < E_par[1] and E_np[2] < E_par[2]
    assert abs(Solver(s).solve(1, 1).Ee[0] - Solver(s, nonparabolic=False).solve(1, 1).Ee[0]) < 1e-12
    S = npb.psi_e.T @ npb.psi_e * npb.dz                                   # orthonormal states
    assert np.allclose(S, np.eye(len(S)), atol=1e-10)
    z02 = np.sum(npb.psi_e[:, 0] * npb.z * npb.psi_e[:, 2]) * npb.dz      # forbidden by parity
    assert abs(z02) < 1e-6
