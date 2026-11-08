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
    """energy-dependent mass: default unchanged, both Kane models match exact solutions"""
    s = Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=10, Lb=30)
    par = Solver(s, dz=0.02).solve(3, 1)
    k2 = Solver(s, dz=0.02, nonparabolic="kane2").solve(3, 1)
    k3 = Solver(s, dz=0.02, nonparabolic=True).solve(3, 1)          # True = three-band Kane
    lev = lambda v: (v.Ee - v.Ec.min()) * 1000
    assert abs(lev(k2)[1] - 117.45) < 0.05 and abs(lev(k2)[0] - 31.41) < 0.05   # exact, two-band
    assert abs(lev(k3)[1] - 117.66) < 0.05 and abs(lev(k3)[2] - 233.25) < 0.05  # exact, three-band
    assert lev(k3)[2] < lev(par)[2]
    assert abs(Solver(s).solve(1, 1).Ee[0] - Solver(s, nonparabolic=False).solve(1, 1).Ee[0]) < 1e-12
    S = k3.psi_e.T @ k3.psi_e * k3.dz                                    # orthonormal states
    assert np.allclose(S, np.eye(len(S)), atol=1e-10)
    z02 = np.sum(k3.psi_e[:, 0] * k3.z * k3.psi_e[:, 2]) * k3.dz         # forbidden by parity
    assert abs(z02) < 1e-6


def test_nonparabolic_transmission():
    """transfer matrix with energy-dependent mass = exact single-barrier formula"""
    from hetero1d import material
    from hetero1d.core import HB2M0
    sol = Solver(Structure().add("GaAs", 20).add("AlGaAs", 5, x=0.3).add("GaAs", 20), dz=0.01, nonparabolic=True)
    w, b = material("GaAs"), material("AlGaAs", 0.3)
    V0, Ek = b["Ec"] - w["Ec"], 0.05
    r = lambda e, p: (2 / p["Eg"] + 1 / (p["Eg"] + p["Dso"])) / (2 / (p["Eg"] + e) + 1 / (p["Eg"] + p["Dso"] + e))
    mw, mb = w["me"] * r(Ek, w), b["me"] * r(Ek - V0, b)
    k, q = np.sqrt(mw * Ek / HB2M0), np.sqrt(mb * (V0 - Ek) / HB2M0)
    a, c = k / mw, q / mb
    T_exact = 1 / (1 + ((a**2 + c**2) / (2 * a * c)) ** 2 * np.sinh(q * 5) ** 2)
    assert abs(sol.transmission(np.array([w["Ec"] + Ek]))[0] - T_exact) < 1e-6


def test_strain_split_off():
    """light hole coupled to the split-off band: In0.2GaAs hh-lh splitting ~90 meV"""
    from hetero1d import material
    from hetero1d.strain import layer_strain
    d = layer_strain(material("InGaAs", 0.2), material("GaAs")["a"])
    assert 80 < (d["dEhh"] - d["dElh"]) * 1000 < 95


def test_thermal_lo_rates():
    """thermal LO rates: cold limit = subband-bottom rate; hot electrons scatter below threshold"""
    s = Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=10, Lb=25)
    sv = Solver(s, T=4.2).solve(3, 1)
    cold = [r for r in sv.lo_phonon_rates() if (r["i"], r["f"]) == (1, 0)][0]["tau_ps"]
    th = [r for r in sv.lo_phonon_rates(thermal=True, Te=4.2) if (r["i"], r["f"]) == (1, 0)][0]["tau_ps"]
    assert abs(th - cold) / cold < 0.01
    wide = Solver(Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=25, Lb=25), T=300).solve(3, 1)
    assert not any((r["i"], r["f"]) == (1, 0) for r in wide.lo_phonon_rates())   # below threshold: no cold rate
    hot = [r for r in wide.lo_phonon_rates(thermal=True) if (r["i"], r["f"]) == (1, 0)][0]["tau_ps"]
    assert abs(hot - 0.481) < 0.01                                              # converged value


def test_gain():
    """gain: absorption without carriers, transparency ~1e12 cm^-2, Bernard-Duraffourg"""
    import warnings
    warnings.simplefilter("ignore")
    sol = Solver(Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=8, Lb=25), T=300).solve(6, 10)
    E = np.linspace(1.40, 1.75, 1200)
    assert sol.gain(E, 0).max() < 0
    g = sol.gain(E, 3e12)
    assert 2500 < g.max() < 4500
    dEF = sol.fermi_level(3e12)[0] - sol.hole_fermi_level(3e12)[0]
    assert E[g > 0].max() < dEF + 0.002


def test_schrodinger_poisson_runs():
    """self-consistent doping works, with and without non-parabolicity"""
    for npar in (False, True):
        d = Structure()
        d.add("AlGaAs", 30, x=0.3).add("AlGaAs", 10, x=0.3, doping=3e17).add("AlGaAs", 10, x=0.3)
        d.add("GaAs", 15).add("AlGaAs", 10, x=0.3).add("AlGaAs", 10, x=0.3, doping=3e17).add("AlGaAs", 30, x=0.3)
        sv = Solver(d, T=77, nonparabolic=npar).schrodinger_poisson(verbose=False)
        sv.solve(4, 2)
        assert sv.EF > sv.Ee[0]



def test_gain_unchanged_by_strain_flag_when_lattice_matched():
    """regression (0.4.1): strain=True must not change the gain of a lattice-matched well"""
    import warnings
    warnings.simplefilter("ignore")
    s = Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=8, Lb=25)
    E = np.linspace(1.40, 1.70, 800)
    g0 = Solver(s, T=300).solve(6, 10).gain(E, 3e12).max()
    g1 = Solver(s, T=300, strain=True).solve(6, 10).gain(E, 3e12).max()
    assert abs(g1 - g0) / g0 < 0.02



def test_stark_sweep_single_bound_level():
    """regression (0.4.2): a well with only one bound electron level must not crash the Stark sweep"""
    import warnings
    warnings.simplefilter("ignore")
    s = Structure.single_well(well=("InGaAs", 0.2), barrier="GaAs", Lw=10, Lb=25)
    r = Solver(s, T=300, strain=True).stark_sweep(np.linspace(-20, 20, 5))
    assert np.all(np.isfinite(r["E_e0hh0"])) and np.all(np.isnan(r["E01"]))
