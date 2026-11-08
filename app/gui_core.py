"""
gui_core.py -- calculations and figures behind the hetero1d web app (no Streamlit code here,
so everything can be tested and reused, e.g. in notebooks).
"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hetero1d import Structure, Solver, material

BINARY = ["GaAs", "AlAs", "InAs", "InP", "GaSb", "AlSb", "GaP", "AlP", "InSb", "GaN", "AlN", "InN"]
TERNARY = ["AlGaAs", "InGaAs", "InAlAs", "GaAsSb", "AlGaSb", "InGaP", "InAsSb", "InGaSb", "InAlSb", "AlAsSb",
           "InAsP", "GaAsP", "AlGaP", "InAlP", "AlGaN", "InGaN", "InAlN"]
QUATERNARY = ["InGaAsP", "InGaAsSb", "AlGaInAs", "AlGaInP"]
ALL_MATERIALS = BINARY + TERNARY + QUATERNARY
SHAPES = ["Single well", "Multiple wells", "Superlattice", "Parabolic well", "Triangular well", "V-shaped well",
          "Step well", "Custom layers"]


def spec(name, x=None, y=None):
    """Material spec tuple as hetero1d expects it."""
    if name in BINARY:
        return (name,)
    if name in TERNARY:
        return (name, float(x))
    return (name, float(x), float(y))


def n_composition(name):
    return 0 if name in BINARY else (1 if name in TERNARY else 2)


def build_structure(p):
    """Build a hetero1d Structure from a dict of user inputs."""
    shape = p["shape"]
    well = spec(p["well"], p.get("xw"), p.get("yw"))
    bar = spec(p["barrier"], p.get("xb"), p.get("yb"))
    sub = p.get("substrate", "GaAs")
    if shape == "Single well":
        return Structure.single_well(well=well, barrier=bar, Lw=p["Lw"], Lb=p["Lb"], substrate=sub)
    if shape == "Multiple wells":
        return Structure.multi_well(int(p["n_wells"]), well=well, barrier=bar, Lw=p["Lw"], Lb=p["Lb_inner"],
                                    Lb_out=p["Lb"], substrate=sub)
    if shape == "Superlattice":
        return Structure.superlattice(int(p["n_wells"]), well=well, barrier=bar, Lw=p["Lw"], Lb=p["Lb_inner"],
                                      Lb_out=p["Lb"], substrate=sub)
    if shape in ("Parabolic well", "Triangular well", "V-shaped well"):
        alloy = p.get("graded_alloy", "AlGaAs")
        kw = dict(Lw=p["Lw"], Lb=p["Lb"], alloy=alloy, x_center=p["x_center"], x_edge=p["x_edge"], substrate=sub)
        if shape == "Parabolic well":
            return Structure.parabolic_well(**kw)
        if shape == "Triangular well":
            return Structure.triangular_well(**kw)
        return Structure.v_well(**kw)
    if shape == "Step well":
        step = spec(p["step"], p.get("xs"), p.get("ys"))
        return Structure.step_well(well=well, step=step, barrier=bar, Lw=p["Lw"], Ls=p["Ls"], Lb=p["Lb"], substrate=sub)
    if shape == "Custom layers":
        s = Structure(substrate=sub, name="custom")
        for row in p["layers"]:
            name = row["material"]
            if not name or not row.get("thickness"):
                continue
            nc = n_composition(name)
            s.add(name, float(row["thickness"]), x=(float(row.get("x") or 0) if nc else None),
                  y=(float(row.get("y") or 0) if nc == 2 else None), doping=float(row.get("doping") or 0))
        if not s.layers:
            raise ValueError("add at least one layer")
        return s
    raise ValueError(f"unknown shape {shape}")


def solve(p):
    """Build and solve; returns (structure, solver, warnings list)."""
    s = build_structure(p)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np_mode = {"off": False, "three-band Kane": True, "two-band Kane": "kane2"}[p.get("nonparabolic", "off")]
        sol = Solver(s, T=p["T"], field=p.get("field", 0.0), dz=p.get("dz", 0.1), Bz=p.get("Bz", 0.0),
                     strain=p.get("strain", False), nonparabolic=np_mode)
        if p.get("self_consistent"):
            sol.schrodinger_poisson(verbose=False)
        sol.solve(n_e=int(p.get("n_e", 6)), n_h=int(p.get("n_h", 6)))
    msgs = sorted({str(w.message) for w in caught})
    return s, sol, msgs


def level_tables(sol):
    """Energies (meV, relative to the band edges) and bound flags for electrons, heavy and light holes."""
    Ec0, Ev0, Evl = sol.Ec.min(), sol.Ev.max(), sol.Ev_lh.max()
    rows = []
    for n in range(max(len(sol.Ee), len(sol.Ehh))):
        rows.append({
            "state": n,
            "electron (meV)": round(float((sol.Ee[n] - Ec0) * 1000), 2) if n < len(sol.Ee) else None,
            "e bound": bool(sol.e_bound[n]) if n < len(sol.Ee) else None,
            "heavy hole (meV)": round(float((Ev0 - sol.Ehh[n]) * 1000), 2) if n < len(sol.Ehh) else None,
            "hh bound": bool(sol.hh_bound[n]) if n < len(sol.Ehh) else None,
            "light hole (meV)": round(float((Evl - sol.Elh[n]) * 1000), 2) if n < len(sol.Elh) else None,
            "lh bound": bool(sol.lh_bound[n]) if n < len(sol.Elh) else None,
        })
    return rows


def transition_tables(sol):
    inter = [{"transition": f"e{t['e']}-{t['band']}{t['h']}", "energy (eV)": round(float(t["E"]), 4),
              "wavelength (nm)": round(float(t["lam_nm"]), 1), "overlap²": round(float(t["overlap2"]), 3)}
             for t in sol.interband()[:12]]
    intra = [{"transition": f"e{t['i']}→e{t['j']}", "energy (meV)": round(float(t["dE"]) * 1000, 1),
              "wavelength (µm)": round(float(t["lam_um"]), 2), "dipole (nm)": round(float(t["z"]), 3),
              "oscillator strength": round(float(t["f"]), 3)} for t in sol.intersubband()]
    return inter, intra


def band_figure(sol, n_show=4, scale=0.05, holes=True):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(sol.z, sol.Ec, "k", lw=1.2, label="Ec")
    ax.plot(sol.z, sol.Ev, "k", lw=1.2, label="Ev (heavy hole)")
    if np.max(np.abs(sol.Ev_lh - sol.Ev)) > 1e-4:
        ax.plot(sol.z, sol.Ev_lh, "k:", lw=1, label="Ev (light hole)")
    for n in range(min(n_show, len(sol.Ee))):
        if sol.e_bound[n]:
            ax.plot(sol.z, sol.Ee[n] + scale * sol.psi_e[:, n] / np.abs(sol.psi_e[:, n]).max(), label=f"e{n}")
    if holes:
        for n in range(min(max(1, n_show // 2), len(sol.Ehh))):
            if sol.hh_bound[n]:
                ax.plot(sol.z, sol.Ehh[n] - scale * sol.psi_hh[:, n] / np.abs(sol.psi_hh[:, n]).max(), "--", label=f"hh{n}")
        if sol.lh_bound[0]:
            ax.plot(sol.z, sol.Elh[0] - scale * sol.psi_lh[:, 0] / np.abs(sol.psi_lh[:, 0]).max(), ":", label="lh0")
    ax.set_xlabel("z (nm)"); ax.set_ylabel("Energy (eV)")
    ax.legend(fontsize=7, ncol=2, loc="center right")
    fig.tight_layout()
    return fig


def conduction_figure(sol, n_show=6, scale=None):
    """Zoom on the conduction band with energies in meV above the well bottom."""
    E0 = sol.Ec.min()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sol.z, (sol.Ec - E0) * 1000, "k")
    bound = [n for n in range(min(n_show, len(sol.Ee))) if sol.e_bound[n]]
    if bound:
        spacing = np.diff([(sol.Ee[n] - E0) * 1000 for n in bound]).min() if len(bound) > 1 else 30
        sc = scale or 0.4 * spacing
        for n in bound:
            En = (sol.Ee[n] - E0) * 1000
            ax.plot(sol.z, En + sc * sol.psi_e[:, n] / np.abs(sol.psi_e[:, n]).max(), label=f"e{n}: {En:.1f} meV")
            ax.axhline(En, color="gray", lw=0.3, ls=":")
    top = (min(sol.Ec[0], sol.Ec[-1]) - E0) * 1000
    ax.set_ylim(-0.05 * top, 1.15 * top if top > 0 else 100)
    ax.set_xlabel("z (nm)"); ax.set_ylabel("Energy above the well bottom (meV)")
    ax.legend(fontsize=7); fig.tight_layout()
    return fig


def spectra_figure(sol, kind, n2d=5e11, gamma=0.006, polarization="TE"):
    fig, ax = plt.subplots(figsize=(8, 4))
    if kind == "intersubband absorption":
        t = sol.intersubband()
        if not t:
            ax.text(0.5, 0.5, "no bound intersubband transition", ha="center", transform=ax.transAxes)
        else:
            Emax = max(tt["dE"] for tt in t) * 1.5
            E = np.linspace(0.005, Emax, 1200)
            ax.plot(E * 1000, sol.isb_absorption(E, n2d_cm2=n2d, gamma=gamma) * 100)
            ax.set_xlabel("Photon energy (meV)"); ax.set_ylabel("Absorption per pass (%)")
    else:
        E0 = sol.interband()[0]["E"]
        E = np.linspace(E0 - 0.06, E0 + 0.25, 1200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind == "interband absorption":
                g = -sol.gain(E, 0, gamma=gamma, polarization=polarization)
                ax.plot(E, g); ax.set_ylabel("Absorption (cm$^{-1}$)")
            else:
                for n in (n2d * 0.5, n2d, n2d * 2):
                    ax.plot(E, sol.gain(E, n, gamma=gamma, polarization=polarization), label=f"{n:.1e} cm$^{{-2}}$")
                ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=7); ax.set_ylabel("Gain (cm$^{-1}$)")
        ax.set_xlabel("Photon energy (eV)")
    ax.set_title(kind); fig.tight_layout()
    return fig


def export_csv(sol):
    """CSV text: position, band edges and all wavefunctions."""
    cols = [sol.z, sol.Ec, sol.Ev, sol.Ev_lh]
    head = ["z_nm", "Ec_eV", "Ev_eV", "Ev_lh_eV"]
    for n in range(sol.psi_e.shape[1]):
        cols.append(sol.psi_e[:, n]); head.append(f"psi_e{n}")
    for n in range(sol.psi_hh.shape[1]):
        cols.append(sol.psi_hh[:, n]); head.append(f"psi_hh{n}")
    for n in range(sol.psi_lh.shape[1]):
        cols.append(sol.psi_lh[:, n]); head.append(f"psi_lh{n}")
    data = np.column_stack(cols)
    lines = [",".join(head)] + [",".join(f"{v:.6g}" for v in row) for row in data]
    return "\n".join(lines)


def material_table(name, x=None, y=None, T=300.0):
    m = material(name, x, y, T=T)
    keys = ["Eg", "Ec", "Ev", "me", "mhh", "mlh", "mhh_par", "mlh_par", "eps_s", "eps_inf", "hwLO", "a", "Ep", "Dso", "g"]
    units = {"Eg": "eV", "Ec": "eV", "Ev": "eV", "hwLO": "eV", "a": "Å", "Ep": "eV", "Dso": "eV",
             "me": "m0", "mhh": "m0", "mlh": "m0", "mhh_par": "m0", "mlh_par": "m0"}
    return [{"parameter": k, "value": round(float(m[k]), 5), "unit": units.get(k, "")} for k in keys]
