"""
geometries.py -- hetero1d example: every built-in well shape in a few lines.
Needs hetero1d >= 0.1.4.   Run:  python geometries.py
"""
import numpy as np
import matplotlib.pyplot as plt
from hetero1d import Structure, Solver, potentials

# ---- real grown structures (graded AlGaAs, electrons AND holes confined) ----
structures = {
    "Square well":          Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=10, Lb=20),
    "Parabolic well":       Structure.parabolic_well(Lw=40),
    "Triangular well":      Structure.triangular_well(Lw=20),
    "V-shaped well":        Structure.v_well(Lw=30),
    "Step well":            Structure.step_well(Lw=6, Ls=6),
    "Double well":          Structure.double_well(Lw=10, Lb=2),
    "Sawtooth (4 periods)": Structure.sawtooth(4, Lw=20),
    "3 parabolic wells":    Structure.multi_graded_well(3, "parabolic", Lw=20, gap=5),
    "Custom shape u^3":     Structure.graded_well(lambda u: abs(2 * u - 1) ** 3, Lw=30),
}

fig, axes = plt.subplots(3, 4, figsize=(18, 11))
axes = axes.ravel()
for ax, (name, s) in zip(axes, structures.items()):
    sol = Solver(s, T=300).solve(n_e=4, n_h=2)
    E0 = sol.Ec.min()
    ax.plot(sol.z, (sol.Ec - E0) * 1000, "k")
    for n in range(4):
        if sol.e_bound[n]:
            En = (sol.Ee[n] - E0) * 1000
            ax.plot(sol.z, En + 12 * sol.psi_e[:, n] / np.abs(sol.psi_e[:, n]).max())
    ax.set_title(name); ax.set_xlabel("z (nm)"); ax.set_ylabel("meV")
    print(f"{name:22s} e0 = {(sol.Ee[0] - E0) * 1000:7.2f} meV")

# ---- formula potentials with V_ext (electrons only; shifts both bands) ----
box = Structure().add("GaAs", 150)
formulas = {
    "V_ext: harmonic, 10 meV":   potentials.parabolic(hw_meV=10, center=75),
    "V_ext: triangular, 50 kV/cm": potentials.triangular(field_kVcm=50),
    "V_ext: Pöschl-Teller":      potentials.poschl_teller(depth=0.2, width=5, center=75),
}
for ax, (name, V) in zip(axes[len(structures):], formulas.items()):
    sol = Solver(box, dz=0.05, V_ext=V).solve(n_e=4, n_h=1)
    E0 = sol.Ec.min()
    ax.plot(sol.z, (sol.Ec - E0) * 1000, "k")
    for n in range(4):
        En = (sol.Ee[n] - E0) * 1000
        ax.plot(sol.z, En + 8 * sol.psi_e[:, n] / np.abs(sol.psi_e[:, n]).max())
    ax.set_ylim(-5, 220); ax.set_title(name); ax.set_xlabel("z (nm)")
    print(f"{name:28s} e0 = {(sol.Ee[0] - E0) * 1000:7.2f} meV")

plt.tight_layout()
plt.savefig("geometries.png", dpi=120)
plt.show()
