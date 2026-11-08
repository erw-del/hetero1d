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
    """
    Stark sweep: for each field, the lowest electron, heavy-hole and light-hole states localised in the
    wells (active states). Quantities that do not exist at a field (for example e1 in a well with a single
    bound level, or a carrier that has tunnelled out) are returned as NaN instead of stopping the sweep.
    """
    keys = ["F", "E_e0hh0", "ov_e0hh0", "E_e0lh0", "ov_e0lh0", "E01", "z01", "zmean_e0", "zmean_hh0"]
    out = {k: [] for k in keys}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for F in np.asarray(fields, dtype=float):
            _, sv, _ = solve(dict(p, field=float(F), n_e=max(int(p.get("n_e", 6)), n_e) + 4,
                                  n_h=max(int(p.get("n_h", 6)), n_h) + 4))
            ie, ih, il = sv.active_states(sv.psi_e), sv.active_states(sv.psi_hh), sv.active_states(sv.psi_lh)
            ov = lambda a, b: float((np.sum(a * b) * sv.dz) ** 2)
            out["F"].append(F)
            e0 = ie[0] if len(ie) else None
            out["E_e0hh0"].append(sv.Ee[e0] - sv.Ehh[ih[0]] if e0 is not None and len(ih) else np.nan)
            out["ov_e0hh0"].append(ov(sv.psi_e[:, e0], sv.psi_hh[:, ih[0]]) if e0 is not None and len(ih) else np.nan)
            out["E_e0lh0"].append(sv.Ee[e0] - sv.Elh[il[0]] if e0 is not None and len(il) else np.nan)
            out["ov_e0lh0"].append(ov(sv.psi_e[:, e0], sv.psi_lh[:, il[0]]) if e0 is not None and len(il) else np.nan)
            if len(ie) > 1:
                out["E01"].append(sv.Ee[ie[1]] - sv.Ee[e0])
                out["z01"].append(float(np.sum(sv.psi_e[:, e0] * sv.z * sv.psi_e[:, ie[1]]) * sv.dz))
            else:
                out["E01"].append(np.nan); out["z01"].append(np.nan)
            out["zmean_e0"].append(sv.expect_z(sv.psi_e[:, e0])[0] if e0 is not None else np.nan)
            out["zmean_hh0"].append(sv.expect_z(sv.psi_hh[:, ih[0]])[0] if len(ih) else np.nan)
    res = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    if np.all(np.isnan(res["E_e0hh0"])) and np.all(np.isnan(res["E_e0lh0"])):
        raise ValueError("no electron-hole pair stays inside the wells at any field (for example holes are not confined "
                         "by a V_ext potential, which shifts both bands together). Use a real well or deeper barriers.")
    return res


def landau_data(sol, B, B_values, n2d, N_max=8, gamma=0.001):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        levels = [{"subband": L["n"], "Landau index N": L["N"], "spin": L["s"], "energy (eV)": round(float(L["E"]), 7),
                   "cyclotron energy (meV)": round(float(L["hwc"]) * 1000, 5)} for L in sol.landau_levels(B=B, N_max=N_max)]
        Bv = np.asarray(B_values, float)
        hwc_min = 1.1577e-4 * Bv.min() / max(sol.level_mass(0), 0.01)          # cyclotron energy at the lowest field
        N_fan = int(min(400, max(N_max, 0.05 / hwc_min + 5)))                   # fill a 50 meV window
        lev, EF, nu = sol.landau_fan(Bv, n2d_cm2=n2d, N_max=N_fan, gamma=gamma)
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


# ---------------------------------------------------------------- figures (professional style)
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.linewidth": 0.9, "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5, "axes.grid": True, "grid.alpha": 0.18, "grid.linestyle": "-",
    "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "#cccccc", "legend.fontsize": 8,
    "figure.dpi": 140, "savefig.dpi": 200, "lines.linewidth": 1.6, "image.cmap": "turbo",
}
PALETTE = ["#1f5fa8", "#d1495b", "#2a9d8f", "#e9a23b", "#6a4c93", "#3d405b", "#8ab17d", "#c06c84", "#118ab2", "#ef476f"]
CMAP = "turbo"


def _fig(w=8.5, h=4.6, ncols=1, nrows=1, **kw):
    plt.rcParams.update(STYLE)
    plt.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=PALETTE)
    return plt.subplots(nrows, ncols, figsize=(w, h), **kw)


def _layer_colours(s):
    """One colour per layer from its material and composition (graded sections shade smoothly)."""
    import matplotlib.colors as mc
    cmap = plt.get_cmap("cividis")
    names = sorted({l.material for l in s.layers})
    out = []
    for l in s.layers:
        base = names.index(l.material) / max(1, len(names) - 1)
        x = l.x if l.x is not None else 0.0
        out.append(cmap(0.15 + 0.7 * (0.5 * base + 0.5 * x)))
    return out


def layer_strip(ax, s, sol, height=0.06):
    """Coloured strip above the axes showing the layer stack (material names on wide layers).
    Any axes title is moved above the strip."""
    if ax.get_title():
        ax.set_title(ax.get_title(), pad=22)
    from matplotlib.patches import Rectangle
    edges = np.concatenate([[0], np.cumsum([l.thickness for l in s.layers])])
    cols = _layer_colours(s)
    trans = ax.get_xaxis_transform()
    span = edges[-1]
    last = None
    for i, l in enumerate(s.layers):
        ax.add_patch(Rectangle((edges[i], 1.01), l.thickness, height, transform=trans, color=cols[i], clip_on=False, lw=0))
        label = l.material + (f" {l.x:.2g}" if l.x is not None and l.label is None else "")
        if l.label is not None:
            label = l.label
        if l.thickness > 0.07 * span and label != last:
            ax.text(edges[i] + l.thickness / 2, 1.01 + height / 2, label, transform=trans, ha="center", va="center",
                    fontsize=7, color="white", clip_on=False)
        last = label


def band_figure(sol, s=None, n_show=4, holes=True, fill=True):
    """Band diagram with filled |psi|^2 lobes at their energies and the layer stack on top."""
    fig, ax = _fig(9, 5.2)
    z = sol.z
    Ec, Ev, Evl = sol.Ec, sol.Ev, sol.Ev_lh
    top, bot = Ec.max(), Ev.min()
    ax.fill_between(z, Ec, top + 0.08 * (top - bot), color="#1f5fa8", alpha=0.07, lw=0)
    ax.fill_between(z, Ev, bot - 0.08 * (top - bot), color="#d1495b", alpha=0.07, lw=0)
    ax.plot(z, Ec, color="#1f5fa8", lw=2, label="E$_c$")
    ax.plot(z, Ev, color="#d1495b", lw=2, label="E$_v$ (heavy hole)")
    if np.max(np.abs(Evl - Ev)) > 1e-4:
        ax.plot(z, Evl, color="#e9a23b", lw=1.4, ls="--", label="E$_v$ (light hole)")
    amp = 0.07 * (top - bot)                                   # lobe height relative to the plotted range
    ecol = plt.get_cmap("Blues")
    hcol = plt.get_cmap("Reds")
    nb = [n for n in range(min(n_show, len(sol.Ee))) if sol.e_bound[n]]
    for k, n in enumerate(nb):
        d = sol.psi_e[:, n] ** 2; d = d / d.max()
        c = ecol(0.85 - 0.5 * k / max(1, len(nb)))
        ax.fill_between(z, sol.Ee[n], sol.Ee[n] + amp * d, color=c, alpha=0.55 if fill else 0, lw=0)
        ax.plot(z, sol.Ee[n] + amp * d, color=c, lw=1)
        ax.hlines(sol.Ee[n], z[0], z[-1], color=c, lw=0.5, ls=":")
    for E_, lab in _group_labels([sol.Ee[n] for n in nb], [f"e{n}" for n in nb], 0.02 * (top - bot)):
        ax.text(z[-1], E_, f" {lab}", va="center", fontsize=8, color="#1f5fa8")
    if holes:
        for k, n in enumerate([n for n in range(min(max(2, n_show // 2), len(sol.Ehh))) if sol.hh_bound[n]]):
            d = sol.psi_hh[:, n] ** 2; d = d / d.max()
            c = hcol(0.85 - 0.3 * k)
            ax.fill_between(z, sol.Ehh[n], sol.Ehh[n] - amp * d, color=c, alpha=0.5, lw=0)
            ax.plot(z, sol.Ehh[n] - amp * d, color=c, lw=1)
        hb = [n for n in range(min(max(2, n_show // 2), len(sol.Ehh))) if sol.hh_bound[n]]
        for E_, lab in _group_labels([sol.Ehh[n] for n in hb], [f"hh{n}" for n in hb], 0.02 * (top - bot)):
            ax.text(z[-1], E_, f" {lab}", va="center", fontsize=8, color="#d1495b")
        if len(sol.Elh) and sol.lh_bound[0]:
            d = sol.psi_lh[:, 0] ** 2; d = d / d.max()
            ax.plot(sol.z, sol.Elh[0] - amp * d, color="#e9a23b", lw=1)
    ax.set_xlim(z[0], z[-1])
    ax.set_xlabel("Position z (nm)"); ax.set_ylabel("Energy (eV)")
    ax.legend(loc="center left", fontsize=8)
    if s is not None:
        layer_strip(ax, s, sol)
    fig.tight_layout()
    return fig


def _group_labels(energies, names, tol):
    """Merge labels of (nearly) degenerate levels: 'e0-e2' instead of three overlapping labels."""
    groups = []
    for E, nm in zip(energies, names):
        if groups and abs(E - groups[-1][0][-1]) < tol:
            groups[-1][0].append(E); groups[-1][1].append(nm)
        else:
            groups.append(([E], [nm]))
    return [(np.mean(es), ns[0] if len(ns) == 1 else f"{ns[0]}–{ns[-1]}") for es, ns in groups]


def conduction_figure(sol, s=None, n_show=6):
    """Conduction-band zoom: |psi|^2 lobes filled with a colour scale, energies in meV."""
    fig, ax = _fig(9, 4.6)
    E0 = sol.Ec.min()
    Ecm = (sol.Ec - E0) * 1000
    top = (min(sol.Ec[0], sol.Ec[-1]) - E0) * 1000
    ax.fill_between(sol.z, Ecm, max(Ecm.max(), top) * 1.15 + 1, color="#1f5fa8", alpha=0.08, lw=0)
    ax.plot(sol.z, Ecm, color="#1f5fa8", lw=2)
    bound = [n for n in range(min(n_show, len(sol.Ee))) if sol.e_bound[n]]
    cmap = plt.get_cmap(CMAP)
    if bound:
        Es = [(sol.Ee[n] - E0) * 1000 for n in bound]
        gaps = np.diff(Es)
        big = gaps[gaps > 0.10 * max(top, 1)] if len(gaps) else np.array([])
        amp = 0.45 * big.min() if len(big) else 0.25 * max(top, 20)          # ignore near-degenerate splittings
        amp = min(amp, 0.25 * max(top, 20))
        for k, (n, En) in enumerate(zip(bound, Es)):
            d = sol.psi_e[:, n] ** 2; d = d / d.max()
            c = cmap(0.1 + 0.8 * k / max(1, len(bound) - 1))
            ax.fill_between(sol.z, En, En + amp * d, color=c, alpha=0.45, lw=0)
            ax.plot(sol.z, En + amp * d, color=c, lw=1)
            ax.hlines(En, sol.z[0], sol.z[-1], color=c, lw=0.6, ls=":")
        for En, lab in _group_labels(Es, [f"e{n}" for n in bound], 0.02 * max(top, 1)):
            ax.text(sol.z[-1], En, f"  {lab}  {En:.2f} meV", va="center", fontsize=8, color="#222222")
    ax.set_xlim(sol.z[0], sol.z[-1])
    ax.set_ylim(-0.04 * max(top, 1), (top if top > 0 else 100) * 1.12)
    ax.set_xlabel("Position z (nm)"); ax.set_ylabel("Energy above the band minimum (meV)")
    if s is not None:
        layer_strip(ax, s, sol)
    fig.tight_layout()
    return fig


def ldos_map(sol, carrier="electrons", gamma_meV=3.0, nE=500):
    """Local density of states map  sum_n |psi_n(z)|^2 L(E - E_n)  (energies in eV)."""
    if carrier == "electrons":
        E_lv, psi, lo, hi = sol.Ee, sol.psi_e, sol.Ec.min() - 0.02, max(sol.Ec.max(), sol.Ee[-1]) + 0.02
    else:
        E_lv, psi = np.concatenate([sol.Ehh, sol.Elh]), np.concatenate([sol.psi_hh, sol.psi_lh], axis=1)
        lo, hi = min(sol.Ev.min(), E_lv.min()) - 0.02, sol.Ev.max() + 0.02
    E = np.linspace(lo, hi, nE)
    g = gamma_meV / 1000
    L = g / np.pi / ((E[:, None] - E_lv[None, :]) ** 2 + g ** 2)            # (nE, n)
    M = L @ (psi ** 2).T                                                        # (nE, nz)
    return E, M


def ldos_figure(sol, s=None, carrier="electrons", gamma_meV=3.0):
    E, M = ldos_map(sol, carrier, gamma_meV)
    fig, ax = _fig(9, 5.0)
    im = ax.pcolormesh(sol.z, E, np.log10(M + M.max() * 1e-4), cmap="magma", shading="auto", rasterized=True)
    edge = sol.Ec if carrier == "electrons" else sol.Ev
    ax.plot(sol.z, edge, color="white", lw=1.4)
    if carrier != "electrons" and np.max(np.abs(sol.Ev_lh - sol.Ev)) > 1e-4:
        ax.plot(sol.z, sol.Ev_lh, color="white", lw=1, ls="--")
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("log$_{10}$ LDOS (arb.)")
    ax.set_xlabel("Position z (nm)"); ax.set_ylabel("Energy (eV)")
    ax.set_title(f"Local density of states: {carrier}")
    ax.grid(False)
    if s is not None:
        layer_strip(ax, s, sol)
    fig.tight_layout()
    return fig


def density_waterfall(sol, n_show=8):
    """|psi|^2 of the electron states stacked (waterfall), colour-coded."""
    fig, ax = _fig(9, 4.8)
    cmap = plt.get_cmap(CMAP)
    n = min(n_show, len(sol.Ee))
    for k in range(n):
        d = sol.psi_e[:, k] ** 2
        off = k * 1.1 * (sol.psi_e ** 2).max()
        c = cmap(0.1 + 0.8 * k / max(1, n - 1))
        ax.fill_between(sol.z, off, off + d, color=c, alpha=0.65, lw=0)
        ax.plot(sol.z, off + d, color=c, lw=1)
        ax.text(sol.z[-1], off, f" e{k}{'' if sol.e_bound[k] else ' (unbound)'}", fontsize=8, va="bottom", color=c)
    ax.set_xlabel("Position z (nm)"); ax.set_ylabel("|ψ|² (offset)"); ax.set_yticks([])
    ax.set_title("Electron probability densities")
    fig.tight_layout()
    return fig


def line_figure(x, ys, labels, xlabel, ylabel, title="", logy=False, fill=False, hlines=None, vlines=None, markers=None,
                logx=False):
    fig, ax = _fig(9, 4.6)
    for i, (y, lab) in enumerate(zip(ys, labels)):
        c = PALETTE[i % len(PALETTE)]
        ax.plot(x, y, color=c, label=lab or None)
        if fill:
            ax.fill_between(x, y, 0 if not logy else np.nanmin(y[np.isfinite(y)]), color=c, alpha=0.15, lw=0)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    for h in hlines or []:
        ax.axhline(h, color="#333333", lw=0.7)
    for v in vlines or []:
        ax.axvline(v, color="#888888", lw=0.7, ls="--")
    for (mx, my, txt) in markers or []:
        ax.plot(mx, my, "o", color="#d1495b", ms=5)
        ax.annotate(txt, (mx, my), textcoords="offset points", xytext=(5, 6), fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if any(labels):
        ax.legend()
    fig.tight_layout()
    return fig


def map_figure(x, y, M, xlabel, ylabel, clabel, title="", lines=None, cmap=CMAP, log=False, symmetric=False):
    fig, ax = _fig(9, 5.0)
    Z = np.log10(np.maximum(M, np.nanmax(M) * 1e-5)) if log else M
    if symmetric:
        vmax = np.nanmax(np.abs(Z)); im = ax.pcolormesh(x, y, Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    else:
        im = ax.pcolormesh(x, y, Z, cmap=cmap, shading="auto", rasterized=True)
    for (lx, ly, lab, col) in lines or []:
        ax.plot(lx, ly, color=col, lw=1.5, label=lab)
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label(clabel)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(False)
    if lines:
        ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def gain_map(sol, densities, polarization="TE", gamma=0.006):
    tr = sol.interband() or sol.interband(bound_only=False)
    E0 = tr[0]["E"]
    E = np.linspace(E0 - 0.05, E0 + 0.22, 500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = np.array([sol.gain(E, n, gamma=gamma, polarization=polarization) for n in densities])
    return E, np.asarray(densities), G


def stark_map(stk, gamma=0.006, nE=500):
    """Absorption map (field, energy) from the e0-hh0 and e0-lh0 transitions of a Stark sweep."""
    F = stk["F"]
    Emin = np.nanmin(np.concatenate([stk["E_e0hh0"], stk["E_e0lh0"]])) - 0.04
    Emax = np.nanmax(np.concatenate([stk["E_e0hh0"], stk["E_e0lh0"]])) + 0.04
    E = np.linspace(Emin, Emax, nE)
    L = lambda E0: gamma / np.pi / ((E[:, None] - np.nan_to_num(E0, nan=1e3)[None, :]) ** 2 + gamma ** 2)
    A = (0.75 * np.nan_to_num(stk["ov_e0hh0"])[None, :] * L(stk["E_e0hh0"])
         + 0.25 * np.nan_to_num(stk["ov_e0lh0"])[None, :] * L(stk["E_e0lh0"]))
    return E, F, A


def landau_map(sol, Bv, n2d=None, gamma=0.001, nE=500, E_span=0.05, N_max=None):
    """
    Landau-level density of states D(E, B) = sum_levels (eB/h) L(E - E_level(B)), in states/(cm^2 eV),
    computed directly from the level energies so the field can be sampled finely (no aliasing of the
    rapidly crossing high levels). With n2d also the Fermi level at each field.
    """
    Bv = np.asarray(Bv, float)
    if N_max is None:                                       # enough levels to fill the energy window at the lowest field
        hwc_min = 1.1577e-4 * Bv.min() / max(sol.level_mass(0), 0.01)        # hbar e B / m*, eV
        N_max = int(min(400, E_span / hwc_min + 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lev, EF, nu = sol.landau_fan(Bv, n2d_cm2=n2d if n2d else None, N_max=N_max, gamma=gamma)
    lev = np.asarray(lev, float)                                           # (nB, n_levels), spin-resolved
    E = np.linspace(sol.Ee[0] - 0.004, sol.Ee[0] + E_span, nE)
    deg = 2.417989e10 * Bv                                                 # eB/h per spin, cm^-2 (per tesla: 2.418e10)
    D = np.zeros((nE, len(Bv)))
    for j in range(len(Bv)):
        e = lev[j][np.isfinite(lev[j])]
        e = e[(e > E[0] - 10 * gamma) & (e < E[-1] + 10 * gamma)]
        if len(e):
            D[:, j] = deg[j] * np.sum(gamma / np.pi / ((E[:, None] - e[None, :]) ** 2 + gamma ** 2), axis=1)
    return E, Bv, D, (np.asarray(EF, float) if n2d else None)


def landau_map_figure(E, B, D, EF=None, title="Landau-level density of states"):
    """DOS map with the energy axis fixed to the map range; the Fermi level as a high-contrast line."""
    import matplotlib.patheffects as pe
    fig, ax = _fig(9, 5.0)
    Z = np.log10(np.maximum(D, D.max() * 1e-4))
    im = ax.pcolormesh(B, E, Z, cmap="inferno", shading="auto", rasterized=True)
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("log$_{10}$ DOS (states cm$^{-2}$ eV$^{-1}$)")
    note = ""
    if EF is not None:
        inside = (EF >= E[0]) & (EF <= E[-1])
        if inside.any():
            ax.plot(B, np.where(inside, EF, np.nan), color="#00e5ff", lw=2.2, ls="--", label="Fermi level",
                    path_effects=[pe.Stroke(linewidth=4, foreground="black"), pe.Normal()])
            ax.legend(loc="upper left", facecolor="#222222", labelcolor="white", edgecolor="#555555")
        if not inside.all():
            below = np.nanmax(EF[~inside]) < E[0]
            note = ("Fermi level below the lowest Landau level (non-degenerate electrons: lower the temperature "
                    "or raise the density)" if below else "Fermi level above the plotted energy range")
    ax.set_ylim(E[0], E[-1]); ax.set_xlim(B[0], B[-1])
    ax.set_xlabel("Magnetic field B (T)"); ax.set_ylabel("Energy (eV)"); ax.set_title(title); ax.grid(False)
    fig.tight_layout()
    return fig, note


def lo_te_sweep(sol, Te_values, n2d=None):
    """Thermal lifetime of every downward transition versus electron temperature."""
    out = {}
    for Te in Te_values:
        for r in lo_table(sol, thermal=True, Te=Te, n2d=n2d):
            if int(r["from"][1:]) > int(r["to"][1:]):
                out.setdefault(f"{r['from']}->{r['to']}", []).append(r["lifetime (ps)"])
    return np.asarray(Te_values), out


def lo_matrix_figure(rows, n):
    """Heat map of lifetimes (ps) from state i (row) to state f (column)."""
    M = np.full((n, n), np.nan)
    for r in rows:
        i, f = int(r["from"][1:]), int(r["to"][1:])
        if i < n and f < n and np.isfinite(r["lifetime (ps)"]):
            M[i, f] = r["lifetime (ps)"]
    fig, ax = _fig(6.2, 5.0)
    im = ax.imshow(np.log10(M), cmap="viridis_r", origin="lower")
    for i in range(n):
        for f in range(n):
            if np.isfinite(M[i, f]):
                ax.text(f, i, f"{M[i, f]:.3g}", ha="center", va="center", fontsize=8,
                        color="white" if np.log10(M[i, f]) > np.nanmean(np.log10(M)) else "black")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"e{k}" for k in range(n)]); ax.set_yticklabels([f"e{k}" for k in range(n)])
    ax.minorticks_off(); ax.grid(False)
    ax.set_xlabel("final state"); ax.set_ylabel("initial state"); ax.set_title("LO-phonon lifetimes (ps)")
    cb = fig.colorbar(im, ax=ax); cb.set_label("log$_{10}$ lifetime (ps)")
    fig.tight_layout()
    return fig


def strain_figure(s, sol):
    fig, ax = _fig(9, 4.4)
    names = [f"{i}: {l.material}" for i, l in enumerate(s.layers)]
    idx = np.arange(len(names))
    d = sol.layer_strain
    w = 0.27
    ax.bar(idx - w, [x["dEc"] * 1000 for x in d], w, label="ΔE$_c$", color="#1f5fa8")
    ax.bar(idx, [x["dEhh"] * 1000 for x in d], w, label="ΔE$_{hh}$", color="#d1495b")
    ax.bar(idx + w, [x["dElh"] * 1000 for x in d], w, label="ΔE$_{lh}$", color="#e9a23b")
    ax.axhline(0, color="#333333", lw=0.7)
    if len(names) <= 25:
        ax.set_xticks(idx); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Band-edge shift (meV)"); ax.set_title("Strain-induced band shifts per layer"); ax.legend()
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.tight_layout()
    return fig


def fan_figure(B, lev, EF=None, sol=None, E_span=0.05, levels_info=None):
    """
    Landau fan: every Landau level versus field (energy axis fixed to E0-4 meV ... E0+E_span), the Fermi
    level where it lies inside that window. Returns (figure, note).
    """
    import matplotlib.patheffects as pe
    lev = np.asarray(lev, float)
    E0 = np.nanmin(lev) if sol is None else float(sol.Ee[0])
    lo, hi = E0 - 0.004, E0 + E_span
    fig, ax = _fig(9, 4.8)
    n = lev.shape[1]
    cols = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, max(n, 2)))
    order = np.argsort(np.nanmin(lev, axis=0))
    for rank, j in enumerate(order):
        y = lev[:, j]
        if np.nanmin(y) > hi:
            continue
        ax.plot(B, y, color=cols[rank], lw=0.9, alpha=0.85)
    note = ""
    if EF is not None:
        EF = np.asarray(EF, float)
        inside = (EF >= lo) & (EF <= hi)
        if inside.any():
            ax.plot(B, np.where(inside, EF, np.nan), color="#d1495b", lw=2.4, label="Fermi level",
                    path_effects=[pe.Stroke(linewidth=4.2, foreground="white"), pe.Normal()])
            ax.legend(loc="upper left")
        if not inside.all():
            below = np.nanmax(EF[~inside]) < lo
            note = ("Fermi level below the lowest Landau level (non-degenerate electrons: lower the temperature "
                    "or raise the density)" if below else "Fermi level above the plotted energy range")
    ax.set_ylim(lo, hi); ax.set_xlim(B[0], B[-1])
    ax.set_xlabel("Magnetic field B (T)"); ax.set_ylabel("Energy (eV)"); ax.set_title("Landau fan")
    fig.tight_layout()
    return fig, note


def minibands_figure(E, q, d, rows, base):
    """Superlattice dispersion q(E) with the minibands shaded."""
    fig, ax = _fig(9, 4.8)
    for k, r in enumerate(rows):
        ax.axhspan(r["bottom (meV)"], r["top (meV)"], color=PALETTE[k % len(PALETTE)], alpha=0.15, lw=0)
        ax.text(1.01, (r["bottom (meV)"] + r["top (meV)"]) / 2, f" {r['width (meV)']:.1f} meV", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8, color=PALETTE[k % len(PALETTE)])
    ax.plot(q * d / np.pi, (E - base) * 1000, color="#1f5fa8", lw=1.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Bloch wavevector q d / π"); ax.set_ylabel("Energy above the band minimum (meV)")
    ax.set_title(f"Superlattice minibands (period {d:.2f} nm); shaded: allowed bands")
    fig.tight_layout()
    return fig


# backwards-compatible name
xy_figure = line_figure
