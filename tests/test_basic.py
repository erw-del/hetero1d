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
