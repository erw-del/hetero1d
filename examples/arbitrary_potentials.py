"""
arbitrary_potentials.py -- hetero1d example
===========================================

Solve quantum wells of any shape with hetero1d, in two ways:

  1. V_ext:  write the potential as a formula (parabolic, triangular, V-shaped ...)
  2. Graded composition: build the shape from thin alloy layers (real grown wells)

Every result is compared with the exact analytic solution where one exists:
  * parabolic well   E_n = hbar*omega (n + 1/2)
  * triangular well  E_n = (hbar^2 / 2m)^(1/3) (eF)^(2/3) a_n,  a_n = zeros of Ai(-x)

Run:   python arbitrary_potentials.py
Needs: hetero1d >= 0.1.2   (pip install hetero1d)
Output: printed tables and the figure arbitrary_potentials.png

Note on V_ext: it is an electrostatic potential, so it shifts the conduction and
valence bands together. A potential that traps electrons pushes holes out. To
confine electrons AND holes (optical transitions, excitons), use graded composition.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import ai_zeros

from hetero1d import Structure, Solver
from hetero1d.core import HB2M0          # hbar^2 / (2 m0) = 0.0380998 eV nm^2

M_GAAS = 0.067                           # GaAs electron effective mass (m0)
KV_PER_CM = 1e-4                         # 1 kV/cm = 1e-4 eV/nm for an electron


def levels_meV(sol, n):
    """Lowest n electron levels in meV, measured from the bottom of the potential."""
    return (sol.Ee[:n] - sol.Ec.min()) * 1000


def print_comparison(title, numeric, exact):
    print(f"\n{title}")
    print("   n    hetero1d (meV)    exact (meV)    difference (meV)")
    for n, (a, b) in enumerate(zip(numeric, exact)):
        print(f"  {n:2d}    {a:12.4f}    {b:11.4f}    {a - b:+14.4f}")


# ---------------------------------------------------------------------------
# 1. Parabolic (harmonic-oscillator) well, written as a formula with V_ext
# ---------------------------------------------------------------------------
hw = 0.010                                       # wanted level spacing hbar*omega (eV)
c = (hw / 2) ** 2 * M_GAAS / HB2M0               # V = c (z - z0)^2 gives that spacing
z0 = 100.0                                       # centre of the parabola (nm)

parabolic = Structure(name="parabolic well (V_ext)").add("GaAs", 200)
sol_par = Solver(parabolic, T=300, dz=0.1,
                 V_ext=lambda z: c * (z - z0) ** 2).solve(n_e=5, n_h=1)

E_par = levels_meV(sol_par, 5)
exact_par = hw * 1000 * (np.arange(5) + 0.5)
print_comparison("1. Parabolic well, hbar*omega = 10 meV", E_par, exact_par)

# ---------------------------------------------------------------------------
# 2. Triangular well: hard wall at z = 0 (edge of the box) plus a uniform field
# ---------------------------------------------------------------------------
F_kVcm = 50.0
F = F_kVcm * KV_PER_CM                           # slope in eV/nm

triangular = Structure(name="triangular well (V_ext)").add("GaAs", 150)
sol_tri = Solver(triangular, T=300, dz=0.05,
                 V_ext=lambda z: F * z).solve(n_e=4, n_h=1)

E_tri = levels_meV(sol_tri, 4)
a_n = -ai_zeros(4)[0]                            # 2.3381, 4.0880, 5.5206, 6.7867
exact_tri = (HB2M0 / M_GAAS) ** (1 / 3) * F ** (2 / 3) * a_n * 1000
print_comparison(f"2. Triangular well, F = {F_kVcm:g} kV/cm (Airy-function solution)", E_tri, exact_tri)

# ---------------------------------------------------------------------------
# 3. Real graded parabolic AlGaAs well, built from 0.2 nm layers
# ---------------------------------------------------------------------------
width, x_edge, slice_nm = 40.0, 0.3, 0.2
graded = Structure(name="graded parabolic AlGaAs well")
graded.add("AlGaAs", 20, x=x_edge)
for zc in np.arange(slice_nm / 2, width, slice_nm):           # centre of each slice
    x = x_edge * ((zc - width / 2) / (width / 2)) ** 2         # 0 in the middle, x_edge at the edges
    graded.add("AlGaAs", slice_nm, x=round(x, 4))
graded.add("AlGaAs", 20, x=x_edge)

sol_gr = Solver(graded, T=300).solve(n_e=6, n_h=6)
E_gr = levels_meV(sol_gr, 5)
H_gr = (sol_gr.Ev.max() - sol_gr.Ehh[:5]) * 1000

print("\n3. Graded parabolic Al(x)Ga(1-x)As well, 40 nm, x = 0 ... 0.3")
print("   electrons   (meV):", np.round(E_gr, 2))
print("   spacings    (meV):", np.round(np.diff(E_gr), 2),
      " (slightly decreasing: composition-dependent mass, finite barrier)")
print("   heavy holes (meV):", np.round(H_gr, 2), " (holes are confined too)")
first = sol_gr.interband()[0]
print(f"   lowest interband transition: {first['E']:.4f} eV ({first['lam_nm']:.1f} nm)")

# ---------------------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
panels = (
    (ax[0], sol_par, "Parabolic well (V_ext)", 8, (40, 160), (-5, 60)),
    (ax[1], sol_tri, f"Triangular well, {F_kVcm:g} kV/cm (V_ext)", 8, (0, 45), (-5, 190)),
    (ax[2], sol_gr, "Graded parabolic AlGaAs well", 15, None, (-5, 280)),
)
for a, sol, title, scale, xlim, ylim in panels:
    E0 = sol.Ec.min()
    a.plot(sol.z, (sol.Ec - E0) * 1000, "k", lw=1.5)
    for n in range(4):
        En = (sol.Ee[n] - E0) * 1000
        psi = sol.psi_e[:, n] / np.abs(sol.psi_e[:, n]).max()
        a.plot(sol.z, En + scale * psi, label=f"E$_{n}$ = {En:.1f} meV")
    a.set_title(title)
    a.set_xlabel("z (nm)")
    a.set_ylabel("Energy above potential minimum (meV)")
    if xlim:
        a.set_xlim(*xlim)
    a.set_ylim(*ylim)
    a.legend(fontsize=8)
plt.tight_layout()
plt.savefig("arbitrary_potentials.png", dpi=150)
print("\nSaved figure: arbitrary_potentials.png")

if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps"):
    plt.show()
