"""
gui_core.py -- every calculation, table and figure behind the hetero1d web app.
No Streamlit code here: all functions can be tested and reused in notebooks.
Tables are lists of dicts; to_csv() turns any of them (or column arrays) into CSV text.
"""
import io
import contextlib
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hetero1d import Structure, Solver, material, lattice_matched, potentials

BINARY = ["GaAs", "AlAs", "InAs", "InP", "GaSb", "AlSb", "GaP", "AlP", "InSb", "GaN", "AlN", "InN"]
TERNARY = ["AlGaAs", "InGaAs", "InAlAs", "GaAsSb", "AlGaSb", "InGaP", "InAsSb", "InGaSb", "InAlSb", "AlAsSb",
           "InAsP", "GaAsP", "AlGaP", "InAlP", "AlGaN", "InGaN", "InAlN"]
QUATERNARY = ["InGaAsP", "InGaAsSb", "AlGaInAs", "AlGaInP"]
ALL_MATERIALS = BINARY + TERNARY + QUATERNARY
SHAPES = ["Single well", "Double well", "Multiple wells", "Superlattice", "Step well", "Graded well",
          "Multiple graded wells", "Sawtooth", "Measured profile", "Custom layers", "Uniform box (for V_ext)"]
GRADED_SHAPES = ["parabolic", "triangular", "triangular_right", "v", "half_parabolic", "cosine", "square"]
POTENTIALS = {
    "parabolic": dict(hw_meV=10.0, center=50.0, mass=0.067),
    "triangular": dict(field_kVcm=50.0, start=0.0, height=0.3),
    "v_shape": dict(slope_kVcm=50.0, center=50.0, height=0.3),
    "sawtooth": dict(n=4, period=20.0, height=0.15, start=20.0, gap=0.0),
    "step": dict(position=50.0, height=0.1),
    "square_well": dict(left=40.0, right=60.0, depth=0.2),
    "poschl_teller": dict(depth=0.2, width=5.0, center=50.0),
    "gaussian": dict(depth=0.1, width=5.0, center=50.0),
    "uniform_field": dict(field_kVcm=10.0, z0=50.0),
}


# ---------------------------------------------------------------- CSV
def to_csv(rows=None, header=None, columns=None):
    """CSV text from a list of dicts, or from column arrays with a header."""
    out = io.StringIO()
    if rows is not None:
        rows = list(rows)
        if not rows:
            return ""
        keys = list(rows[0].keys())
        out.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k)
                if v is None:
                    vals.append("")
                elif isinstance(v, (float, np.floating)):
                    vals.append(f"{v:.8g}")
                else:
                    s = str(v)
                    vals.append(f'"{s}"' if "," in s else s)
            out.write(",".join(vals) + "\n")
    else:
        out.write(",".join(header) + "\n")
        data = np.column_stack([np.asarray(c, dtype=float) for c in columns])
        for row in data:
            out.write(",".join(f"{v:.8g}" for v in row) + "\n")
    return out.getvalue()


# ---------------------------------------------------------------- structure and solver
def spec(name, x=None, y=None):
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
    sub = p.get("substrate", "GaAs")
    well = spec(p.get("well", "GaAs"), p.get("xw"), p.get("yw"))
    bar = spec(p.get("barrier", "AlGaAs"), p.get("xb", 0.3), p.get("yb"))
    graded = dict(alloy=p.get("graded_alloy", "AlGaAs"), x_center=p.get("x_center", 0.0), x_edge=p.get("x_edge", 0.3))
    if shape == "Single well":
        return Structure.single_well(well=well, barrier=bar, Lw=p["Lw"], Lb=p["Lb"], substrate=sub)
    if shape == "Double well":
        return Structure.double_well(well=well, barrier=bar, Lw=(p["Lw"], p.get("Lw2", p["Lw"])), Lb=p["Lb_inner"],
                                     Lb_out=p["Lb"], substrate=sub)
    if shape == "Multiple wells":
        return Structure.multi_well(int(p["n_wells"]), well=well, barrier=bar, Lw=p["Lw"], Lb=p["Lb_inner"],
                                    Lb_out=p["Lb"], substrate=sub)
    if shape == "Superlattice":
        return Structure.superlattice(int(p["n_wells"]), well=well, barrier=bar, Lw=p["Lw"], Lb=p["Lb_inner"],
                                      Lb_out=p["Lb"], substrate=sub)
    if shape == "Step well":
        step = spec(p["step"], p.get("xs"), p.get("ys"))
        return Structure.step_well(well=well, step=step, barrier=bar, Lw=p["Lw"], Ls=p["Ls"], Lb=p["Lb"], substrate=sub)
    if shape == "Graded well":
        return Structure.graded_well(p.get("graded_shape", "parabolic"), Lw=p["Lw"], Lb=p["Lb"], substrate=sub, **graded)
    if shape == "Multiple graded wells":
        return Structure.multi_graded_well(int(p["n_wells"]), p.get("graded_shape", "parabolic"), Lw=p["Lw"],
                                           gap=p.get("gap", 5.0), Lb_out=p["Lb"], substrate=sub, **graded)
    if shape == "Sawtooth":
        return Structure.sawtooth(int(p["n_wells"]), Lw=p["Lw"], gap=p.get("gap", 0.0), Lb_out=p["Lb"], substrate=sub, **graded)
    if shape == "Measured profile":
        z, x = p.get("profile_z") or [], p.get("profile_x") or []
        if len(z) < 2:
            raise ValueError("upload a CSV file with columns z (nm) and x")
        return Structure.from_profile(p.get("graded_alloy", "AlGaAs"), z, x, Lb=p.get("Lb", 0.0),
                                      x_barrier=p.get("x_edge"), substrate=sub)
    if shape == "Uniform box (for V_ext)":
        m = spec(p.get("box_material", "GaAs"), p.get("xbox"), p.get("ybox"))
        return Structure(substrate=sub, name="box").add(m[0], p["box_length"], x=m[1] if len(m) > 1 else None,
                                                        y=m[2] if len(m) > 2 else None)
    if shape == "Custom layers":
        s = Structure(substrate=sub, name="custom")
        for row in p["layers"]:
            name, t = row.get("material"), row.get("thickness")
            if not name or not t:
                continue
            nc = n_composition(name)
            x = float(row.get("x") or 0) if nc else None
            y = float(row.get("y") or 0) if nc == 2 else None
            x_end = row.get("x_end")
            dop = float(row.get("doping") or 0)
            if nc and x_end not in (None, "") and float(x_end) != x:
                x1, tt = float(x_end), float(t)
                s.add_graded(name, tt, x=lambda z, x0=x, x1=x1, tt=tt: x0 + (x1 - x0) * z / tt, y=y, doping=dop)
            else:
                s.add(name, float(t), x=x, y=y, doping=dop)
        if not s.layers:
            raise ValueError("add at least one layer")
        return s
    raise ValueError(f"unknown structure type {shape}")


def build_potential(plist):
    """Combine the selected V_ext shapes (list of dicts with 'type' and parameters)."""
    funcs = []
    for item in plist or []:
        kind = item.get("type", "none")
        if kind == "none":
            continue
        args = {k: item[k] for k in POTENTIALS[kind] if k in item}
        if kind == "sawtooth":
            args["n"] = int(args["n"])
        if kind in ("triangular", "v_shape") and args.get("height") in (0, 0.0):
            args["height"] = None
        funcs.append(getattr(potentials, kind)(**args))
    if not funcs:
        return None
    return funcs[0] if len(funcs) == 1 else potentials.combine(*funcs)


def solve(p):
    """Build and solve; returns (structure, solver, list of warning messages)."""
    s = build_structure(p)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np_mode = {"off": False, "three-band Kane": True, "two-band Kane": "kane2"}[p.get("nonparabolic", "off")]
        sol = Solver(s, T=p["T"], field=p.get("field", 0.0), dz=p.get("dz", 0.1), Bz=p.get("Bz", 0.0),
                     Bpar=p.get("Bpar", 0.0), z0=p.get("z0") or None, V_ext=build_potential(p.get("potentials")),
                     strain=p.get("strain", False), nonparabolic=np_mode)
        if p.get("self_consistent"):
            sol.schrodinger_poisson(mix=p.get("sp_mix", 0.1), tol=p.get("sp_tol", 1e-5),
                                    right_bc=p.get("sp_bc", "neumann"), verbose=False)
        sol.solve(n_e=int(p.get("n_e", 6)), n_h=int(p.get("n_h", 6)))
    return s, sol, sorted({str(w.message) for w in caught})


def text_of(func, *a, **k):
    """Capture what a hetero1d method prints (report, describe)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        func(*a, **k)
    return buf.getvalue()


# ---------------------------------------------------------------- states
def level_tables(sol):
    Ec0, Ev0, Evl = sol.Ec.min(), sol.Ev.max(), sol.Ev_lh.max()
    rows = []
    for n in range(max(len(sol.Ee), len(sol.Ehh))):
        rows.append({
            "state": n,
            "electron (meV)": round(float((sol.Ee[n] - Ec0) * 1000), 3) if n < len(sol.Ee) else None,
            "electron (eV, absolute)": round(float(sol.Ee[n]), 6) if n < len(sol.Ee) else None,
            "e bound": bool(sol.e_bound[n]) if n < len(sol.Ee) else None,
            "heavy hole (meV)": round(float((Ev0 - sol.Ehh[n]) * 1000), 3) if n < len(sol.Ehh) else None,
            "hh bound": bool(sol.hh_bound[n]) if n < len(sol.Ehh) else None,
            "light hole (meV)": round(float((Evl - sol.Elh[n]) * 1000), 3) if n < len(sol.Elh) else None,
            "lh bound": bool(sol.lh_bound[n]) if n < len(sol.Elh) else None,
        })
    return rows


def state_analysis(s, sol):
    """Per state: position, spread, mass, the region where it is mostly found."""
    rows = []
    for kind, E, psi, bound in (("electron", sol.Ee, sol.psi_e, sol.e_bound), ("heavy hole", sol.Ehh, sol.psi_hh, sol.hh_bound),
                                ("light hole", sol.Elh, sol.psi_lh, sol.lh_bound)):
        for n in range(len(E)):
            zm, sp = sol.expect_z(psi[:, n])
            name, pmax = max(sol.region_probabilities(psi[:, n]), key=lambda r: r[1])
            rows.append({"carrier": kind, "state": n, "energy (eV)": round(float(E[n]), 6), "bound": bool(bound[n]),
                         "<z> (nm)": round(float(zm), 3), "spread (nm)": round(float(sp), 3),
                         "mostly in": name, "probability there": round(float(pmax), 4),
                         "mass (m0)": round(float(sol.level_mass(n)), 5) if kind == "electron" else None})
    return rows


def region_table(sol):
    """Probability of every electron state in every region (graded sections grouped)."""
    rows = []
    for n in range(len(sol.Ee)):
        row = {"state": f"e{n}"}
        for i, (name, pr) in enumerate(sol.region_probabilities(sol.psi_e[:, n])):
            row[f"{i}: {name}"] = round(float(pr), 5)
        rows.append(row)
    return rows


def wavefunction_columns(sol):
    cols = [sol.z, sol.Ec, sol.Ev, sol.Ev_lh]
    head = ["z_nm", "Ec_eV", "Ev_eV", "Ev_lh_eV"]
    for tag, psi in (("e", sol.psi_e), ("hh", sol.psi_hh), ("lh", sol.psi_lh)):
        for n in range(psi.shape[1]):
            cols.append(psi[:, n]); head.append(f"psi_{tag}{n}")
    return head, cols


def transition_tables(sol, bound_only=True):
    inter = [{"transition": f"e{t['e']}-{t['band']}{t['h']}", "energy (eV)": round(float(t["E"]), 5),
              "wavelength (nm)": round(float(t["lam_nm"]), 2), "overlap^2": round(float(t["overlap2"]), 4),
              "weight": t["weight"]} for t in sol.interband(bound_only=bound_only)]
    intra = [{"transition": f"e{t['i']}-e{t['j']}", "energy (meV)": round(float(t["dE"]) * 1000, 3),
              "wavelength (um)": round(float(t["lam_um"]), 3), "dipole (nm)": round(float(t["z"]), 4),
              "oscillator strength": round(float(t["f"]), 4), "mass (m0)": round(float(t["m"]), 4)}
             for t in sol.intersubband(bound_only=bound_only)]
    return inter, intra


# ---------------------------------------------------------------- statistics, excitons, scattering
def fermi_tables(sol, n2d, p2d, T=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        EF, pe = sol.fermi_level(n2d, T=T)
        EFv, phh, plh = sol.hole_fermi_level(p2d, T=T)
    rows = [{"carrier": "electron", "state": n, "energy (eV)": round(float(sol.Ee[n]), 6),
             "population (cm^-2)": float(pe[n]), "share (%)": round(float(pe[n] / n2d * 100), 4)} for n in range(len(pe))]
    rows += [{"carrier": "heavy hole", "state": n, "energy (eV)": round(float(sol.Ehh[n]), 6),
              "population (cm^-2)": float(phh[n]), "share (%)": round(float(phh[n] / p2d * 100), 4)} for n in range(len(phh))]
    rows += [{"carrier": "light hole", "state": n, "energy (eV)": round(float(sol.Elh[n]), 6),
              "population (cm^-2)": float(plh[n]), "share (%)": round(float(plh[n] / p2d * 100), 4)} for n in range(len(plh))]
    return float(EF), float(EFv), rows


def exciton_row(sol, ie=0, ih=0, hole="hh"):
    x = sol.exciton(ie, ih, hole=hole)
    return {"electron level": ie, "hole level": ih, "hole": hole, "binding energy (meV)": round(float(x["Eb_meV"]), 4),
            "radius (nm)": round(float(x["lambda_nm"]), 4), "exciton line (eV)": round(float(x["E_transition_eV"]), 6),
            "exciton line (nm)": round(1239.84 / float(x["E_transition_eV"]), 3), "reduced mass (m0)": round(float(x["mu"]), 5),
            "permittivity": round(float(x["eps"]), 4)}


def lo_table(sol, thermal=False, Te=None, n2d=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rates = sol.lo_phonon_rates(thermal=thermal, Te=Te, n2d_cm2=n2d)
    rows = []
    for r in rates:
        row = {"from": f"e{r['i']}", "to": f"e{r['f']}", "spacing (meV)": round(float(r["dE"]) * 1000, 3),
               "rate (1/s)": float(r["rate"]), "lifetime (ps)": float(r["tau_ps"])}
        if thermal:
            row["emission rate (1/s)"] = float(r["rate_em"]); row["absorption rate (1/s)"] = float(r["rate_abs"])
        rows.append(row)
    return rows


# ---------------------------------------------------------------- spectra, tunnelling, minibands
SPECTRA = ["intersubband absorption", "interband absorption (cm-1)", "interband absorption (relative)", "gain"]


def spectrum(sol, kind, n2d=5e11, p2d=None, gamma=0.006, polarization="TE"):
    """Returns (energies eV, values, label)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "intersubband absorption":
            Emax = max([tt["dE"] for tt in sol.intersubband()], default=0.2) * 1.5
            E = np.linspace(0.005, Emax, 1500)
            return E, sol.isb_absorption(E, n2d_cm2=n2d, gamma=gamma), "absorption per pass (fraction)"
        tr = sol.interband() or sol.interband(bound_only=False)
        E0 = tr[0]["E"]
        E = np.linspace(E0 - 0.06, E0 + 0.25, 1500)
        if kind == "interband absorption (relative)":
            return E, sol.interband_absorption(E, gamma=gamma), "absorption (relative)"
        if kind == "interband absorption (cm-1)":
            return E, -sol.gain(E, 0, gamma=gamma, polarization=polarization), "absorption (cm^-1)"
        return E, sol.gain(E, n2d, p2d_cm2=p2d, gamma=gamma, polarization=polarization), "gain (cm^-1)"


def transmission_data(sol, E_from_meV, E_to_meV, n_points=4000, nonparabolic=None):
    """
    T(E), energies in meV above the conduction-band edge of the first layer. Resonances (local maxima
    with T > 0.3) are re-scanned on a fine grid around each peak, so that even very narrow resonances
    get accurate heights, widths (FWHM) and lifetimes (hbar / width).
    """
    Ec_lead = sol.Ec[0]
    Em = np.linspace(E_from_meV, E_to_meV, n_points)

    def T_of(e_meV):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(sol.transmission(Ec_lead + np.asarray(e_meV) / 1000, nonparabolic=nonparabolic))

    T = T_of(Em)
    step = Em[1] - Em[0]
    res = []
    for i in range(1, len(T) - 1):
        if T[i] > 0.3 and T[i] >= T[i - 1] and T[i] > T[i + 1]:
            fine = np.linspace(Em[i] - 3 * step, Em[i] + 3 * step, 3001)       # zoom on the peak
            Tf = T_of(fine)
            k = int(np.argmax(Tf))
            half = Tf[k] / 2
            above = fine[Tf >= half]
            width = above.max() - above.min() if len(above) > 1 else 0.0
            if width >= 5.9 * step:                                           # broad: measure on a wider window
                wide = np.linspace(Em[i] - 40 * step, Em[i] + 40 * step, 4001)
                Tw = T_of(wide)
                above = wide[Tw >= half]
                width = above.max() - above.min()
            res.append({"resonance (meV)": round(float(fine[k]), 5), "T max": round(float(Tf[k]), 5),
                        "width (meV)": round(float(width), 6),
                        "lifetime (ps)": round(float(0.6582 / width), 4) if width > 0 else None})
    return Em, T, res


def minibands_table(sol, E_span_meV=500, dz=0.02):
    base = float(np.min(sol.Ec))
    E = np.linspace(base, base + E_span_meV / 1000, 6000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q, d, bands = sol.minibands(E, dz=dz)
    rows = [{"miniband": i, "bottom (meV)": round(float((a - base) * 1000), 3), "top (meV)": round(float((b - base) * 1000), 3),
             "width (meV)": round(float((b - a) * 1000), 3),
             "note": "cut at the end of the search range" if abs((b - base) * 1000 - E_span_meV) < 1e-3 else ""}
            for i, (a, b) in enumerate(bands)]
    return E, np.asarray(q), float(d), rows, base


# ---------------------------------------------------------------- fields
def stark_data(p, fields, n_e=8, n_h=8):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, sol, _ = solve(dict(p, field=0.0))
        r = sol.stark_sweep(np.asarray(fields, dtype=float), n_e=n_e, n_h=n_h)
    keys = ["F", "E_e0hh0", "ov_e0hh0", "E_e0lh0", "ov_e0lh0", "E01", "z01", "zmean_e0", "zmean_hh0"]
    return {k: np.asarray(r[k], dtype=float) for k in keys}


def landau_data(sol, B, B_values, n2d, N_max=8, gamma=0.001):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        levels = [{"subband": L["n"], "Landau index N": L["N"], "spin": L["s"], "energy (eV)": round(float(L["E"]), 7),
                   "cyclotron energy (meV)": round(float(L["hwc"]) * 1000, 5)} for L in sol.landau_levels(B=B, N_max=N_max)]
        lev, EF, nu = sol.landau_fan(np.asarray(B_values, float), n2d_cm2=n2d, N_max=N_max, gamma=gamma)
        Eg = np.linspace(sol.Ee[0] - 0.005, sol.Ee[0] + 0.06, 1500)
        dos = sol.landau_dos(Eg, B=B, gamma=gamma)
        EF_B = sol.fermi_level_B(n2d, B=B, gamma=gamma)
        zee = sol.zeeman(B=B)
    spin = [{"subband": n, "g-factor": round(float(sol.g_factor(n)), 5), "Zeeman splitting (meV)": round(float(zee[n]) * 1000, 6)}
            for n in range(len(zee))]
    return levels, np.asarray(lev), np.asarray(EF), np.asarray(nu), Eg, np.asarray(dos), float(EF_B), spin


def dispersion_data(sol, ky_max=0.2, n_ky=81, n_e=4):
    ky = np.linspace(-ky_max, ky_max, n_ky)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        E = np.asarray(sol.inplane_dispersion(ky, n_e=n_e))
    return ky, E


# ---------------------------------------------------------------- materials
def material_table(name, x=None, y=None, T=300.0):
    m = material(name, x, y, T=T)
    keys = ["Eg", "Ec", "Ev", "me", "mhh", "mlh", "mhh_par", "mlh_par", "eps_s", "eps_inf", "hwLO", "a", "Ep", "Dso", "g",
            "C11", "C12", "ac", "av", "b"]
    units = {"Eg": "eV", "Ec": "eV", "Ev": "eV", "hwLO": "eV", "a": "Angstrom", "Ep": "eV", "Dso": "eV", "me": "m0", "mhh": "m0",
             "mlh": "m0", "mhh_par": "m0", "mlh_par": "m0", "C11": "GPa", "C12": "GPa", "ac": "eV", "av": "eV", "b": "eV"}
    return [{"parameter": k, "value": round(float(m[k]), 6), "unit": units.get(k, "")} for k in keys if k in m]


def all_materials_table(T=300.0):
    """Every binary (and the end points of each alloy are binaries) with its main parameters."""
    rows = []
    for name in BINARY:
        m = material(name, T=T)
        rows.append({"material": name, "Eg (eV)": round(float(m["Eg"]), 5), "Ec (eV)": round(float(m["Ec"]), 5),
                     "Ev (eV)": round(float(m["Ev"]), 5), "me (m0)": round(float(m["me"]), 5), "a (A)": round(float(m["a"]), 5)})
    return rows


def lattice_matched_value(name, substrate, fixed=None):
    return float(lattice_matched(name, substrate, fixed=fixed))


# ---------------------------------------------------------------- figures
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
        if len(sol.Elh) and sol.lh_bound[0]:
            ax.plot(sol.z, sol.Elh[0] - scale * sol.psi_lh[:, 0] / np.abs(sol.psi_lh[:, 0]).max(), ":", label="lh0")
    ax.set_xlabel("z (nm)"); ax.set_ylabel("Energy (eV)"); ax.legend(fontsize=7, ncol=2, loc="center right")
    fig.tight_layout()
    return fig


def conduction_figure(sol, n_show=6):
    E0 = sol.Ec.min()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sol.z, (sol.Ec - E0) * 1000, "k")
    bound = [n for n in range(min(n_show, len(sol.Ee))) if sol.e_bound[n]]
    if bound:
        Es = [(sol.Ee[n] - E0) * 1000 for n in bound]
        gaps = np.diff(Es)
        sc = 0.4 * (gaps.min() if len(gaps) and gaps.min() > 0.5 else 30)
        for n, En in zip(bound, Es):
            ax.plot(sol.z, En + sc * sol.psi_e[:, n] / np.abs(sol.psi_e[:, n]).max(), label=f"e{n}: {En:.2f} meV")
            ax.axhline(En, color="gray", lw=0.3, ls=":")
    top = (min(sol.Ec[0], sol.Ec[-1]) - E0) * 1000
    ax.set_ylim(-0.05 * max(top, 1), 1.15 * top if top > 0 else 100)
    ax.set_xlabel("z (nm)"); ax.set_ylabel("Energy above the minimum (meV)"); ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def xy_figure(x, ys, labels, xlabel, ylabel, title="", logy=False, hlines=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    for y, lab in zip(ys, labels):
        ax.plot(x, y, label=lab)
    if logy:
        ax.set_yscale("log")
    for h in hlines or []:
        ax.axhline(h, color="k", lw=0.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if any(labels):
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fan_figure(B, lev, EF):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(B, lev, color="gray", lw=0.6)
    ax.plot(B, EF, "r", lw=1.5, label="Fermi level")
    lo, hi = np.nanmin(lev), np.nanmax(EF)
    ax.set_ylim(lo - 0.002, hi + 0.5 * (hi - lo) + 0.002)
    ax.set_xlabel("B (T)"); ax.set_ylabel("Energy (eV)"); ax.set_title("Landau fan"); ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
