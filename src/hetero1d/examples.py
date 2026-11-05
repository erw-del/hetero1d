"""Run the built-in examples:  python -m hetero1d.examples   or   hetero1d-demo"""
import numpy as np
from .core import *


def main():

    list_materials()

    # 1) single GaAs / Al0.3Ga0.7As quantum well, 10 nm
    s1 = Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=10, Lb=25)
    s1.describe()
    sol1 = Solver(s1, T=300).solve()
    sol1.report(n2d_cm2=5e11)
    sol1.plot_summary("ex1_single_qw.png", n2d_cm2=5e11)

    # 2) three different wells (15, 10, 20 nm)
    s2 = Structure.multi_well(3, well="GaAs", barrier=("AlGaAs", 0.3),
                              Lw=[15, 10, 20], Lb=[20, 10], Lb_out=25, name="triple QW")
    sol2 = Solver(s2, T=300).solve(n_e=8)
    sol2.report()
    sol2.plot_summary("ex2_triple_qw.png")

    # 3) superlattice and minibands
    s3 = Structure.superlattice(10, well="GaAs", barrier=("AlGaAs", 0.3), Lw=5, Lb=2, Lb_out=20)
    sol3 = Solver(s3, T=300).solve(n_e=12)
    Ec_w = material("GaAs", T=300)["Ec"]
    q, d, bands = sol3.minibands(np.linspace(Ec_w, Ec_w + 0.6, 6000))
    print("\nSuperlattice minibands:")
    for b in bands:
        print(f"   {b[0]:.4f} .. {b[1]:.4f} eV   width {(b[1]-b[0])*1000:.1f} meV")
    sol3.plot_summary("ex3_superlattice.png")

    # 4) InGaAs/InAlAs (lattice-matched to InP) under 20 kV/cm
    xIn_GaAs = lattice_matched("InGaAs", "InP")
    xIn_AlAs = lattice_matched("InAlAs", "InP")
    print(f"\nLattice-matched to InP: In{xIn_GaAs:.3f}GaAs, In{xIn_AlAs:.3f}AlAs")
    s4 = Structure.single_well(well=("InGaAs", xIn_GaAs), barrier=("InAlAs", xIn_AlAs),
                               Lw=8, Lb=20, substrate="InP", name="InGaAs/InAlAs QW, 20 kV/cm")
    s4.describe()
    sol4 = Solver(s4, T=300, field=20).solve()
    sol4.report(n2d_cm2=1e12)
    sol4.plot_summary("ex4_ingaas_qw_field.png", n2d_cm2=1e12)

    # 5) modulation-doped QW, Schroedinger-Poisson
    s5 = Structure(name="modulation-doped QW")
    s5.add("AlGaAs", 30, 0.3).add("AlGaAs", 10, 0.3, doping=3e17).add("AlGaAs", 10, 0.3)
    s5.add("GaAs", 15).add("AlGaAs", 10, 0.3).add("AlGaAs", 10, 0.3, doping=3e17).add("AlGaAs", 30, 0.3)
    sol5 = Solver(s5, T=77).schrodinger_poisson()
    sol5.solve(n_e=6)
    sol5.report(n2d_cm2=sol5.Nd.sum() * sol5.dz * 1e14)
    sol5.plot_summary("ex5_sp_doped.png", n2d_cm2=sol5.Nd.sum() * sol5.dz * 1e14)

    # 6) ELECTRIC FIELD: quantum-confined Stark effect, 10 nm GaAs QW
    sol6 = Solver(s1, T=300)
    r = sol6.plot_stark("ex6_stark.png", np.linspace(-80, 80, 33))
    print("\nQCSE (10 nm GaAs/AlGaAs QW):")
    for F, E, ov in zip(r["F"][::8], r["E_e0hh0"][::8], r["ov_e0hh0"][::8]):
        print(f"   F = {F:6.1f} kV/cm   E(e0-hh0) = {E:.4f} eV   overlap^2 = {ov:.3f}")

    # 7) MAGNETIC FIELD along z: Landau levels, Zeeman, Fermi level vs B
    sol7 = Solver(s1, T=4.2, Bz=5.0).solve()
    sol7.report(n2d_cm2=3e11)
    sol7.plot_magnetic("ex7_landau.png", np.linspace(0.5, 10, 300), n2d_cm2=3e11)

    # 8) MAGNETIC FIELD in-plane: diamagnetic shift and dispersion, 20 nm QW
    s8 = Structure.single_well(well="GaAs", barrier=("AlGaAs", 0.3), Lw=20, Lb=25, name="20 nm QW")
    E0 = Solver(s8, T=4.2).solve().Ee[0]
    for B in (5, 10, 20):
        sB = Solver(s8, T=4.2, Bpar=B).solve()
        print(f"   Bpar = {B:2d} T: diamagnetic shift of e0 = {(sB.Ee[0]-E0)*1000:.3f} meV")
    sol8 = Solver(s8, T=4.2, Bpar=10).solve()
    sol8.plot_magnetic("ex8_inplane_B.png", np.linspace(0.5, 10, 200), n2d_cm2=3e11)

    # 9) QUATERNARY: InGaAsP/InP quantum well for 1.55 um
    y = 0.85
    x = lattice_matched("InGaAsP", "InP", fixed=("y", y))
    s9 = Structure.single_well(well=("InGaAsP", x, y), barrier="InP", Lw=8, Lb=20,
                               substrate="InP", name=f"Ga{x:.2f}In{1-x:.2f}As{y}P{1-y:.2f} / InP QW")
    s9.describe()
    sol9 = Solver(s9, T=300).solve()
    t = sol9.interband()[0]
    print(f"   lowest interband transition: {t['E']:.4f} eV  ({t['lam_nm']:.0f} nm)")


if __name__ == "__main__":
    main()
