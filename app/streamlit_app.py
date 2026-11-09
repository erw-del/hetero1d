"""
hetero1d web app: every feature of hetero1d 0.4.1, professional figures, every result as CSV.
Run locally:  streamlit run app/streamlit_app.py
"""
import io
import json
import zipfile
import numpy as np
import streamlit as st
import hetero1d
import gui_core as g

st.set_page_config(page_title="hetero1d", page_icon="⚛️", layout="wide")
st.title("hetero1d · semiconductor heterostructure simulator")
st.caption(f"hetero1d {hetero1d.__version__} · single-band envelope functions with strain and non-parabolicity · "
           "[GitHub](https://github.com/erw-del/hetero1d) · [PyPI](https://pypi.org/project/hetero1d/)")

PAGES = ["Band diagram", "Density maps", "States", "Transitions", "Spectra", "Gain map", "LO phonons", "Statistics",
         "Exciton", "Tunnelling", "Minibands", "Stark effect", "Magnetic field", "Strain", "Materials", "Report",
         "Download all"]


def csv_button(label, csv_text, filename):
    if csv_text:
        st.download_button(f"⬇ {label} (CSV)", csv_text, filename, "text/csv", key="dl_" + filename)


def show(fig):
    st.pyplot(fig, use_container_width=True)


def material_input(label, default, key, x_default=0.3, y_default=0.5):
    name = st.selectbox(label, g.ALL_MATERIALS, index=g.ALL_MATERIALS.index(default), key=key)
    nc = g.n_composition(name)
    x = y = None
    if nc >= 1:
        x = st.number_input(f"{label}: x", 0.0, 1.0, x_default, 0.01, format="%.3f", key=key + "_x")
    if nc == 2:
        y = st.number_input(f"{label}: y", 0.0, 1.0, y_default, 0.01, format="%.3f", key=key + "_y")
    return name, x, y


def potential_input(i):
    kind = st.selectbox(f"Potential {i}", ["none"] + list(g.POTENTIALS), key=f"pot{i}")
    item = dict(type=kind)
    if kind != "none":
        for k, v in g.POTENTIALS[kind].items():
            item[k] = st.number_input(f"{kind}: {k}", value=float(v), key=f"pot{i}_{k}", format="%.4g")
    return item


# ================================================================ sidebar: inputs
with st.sidebar:
    st.header("1 · Structure")
    shape = st.selectbox("Type", g.SHAPES)
    p = dict(shape=shape)
    if shape in ("Single well", "Double well", "Multiple wells", "Superlattice", "Step well"):
        p["well"], p["xw"], p["yw"] = material_input("Well", "GaAs", "well", 0.2)
        p["barrier"], p["xb"], p["yb"] = material_input("Barrier", "AlGaAs", "bar", 0.3)
        p["Lw"] = st.number_input("Well width (nm)", 0.5, 200.0, 10.0, 0.5)
        if shape == "Double well":
            p["Lw2"] = st.number_input("Second well width (nm)", 0.5, 200.0, 10.0, 0.5)
            p["Lb_inner"] = st.number_input("Barrier between the wells (nm)", 0.3, 100.0, 2.0, 0.1)
        if shape in ("Multiple wells", "Superlattice"):
            p["n_wells"] = st.number_input("Number of wells / periods", 2, 40, 3)
            p["Lb_inner"] = st.number_input("Barrier between wells (nm)", 0.3, 100.0, 5.0, 0.1)
        if shape == "Step well":
            p["step"], p["xs"], p["ys"] = material_input("Step layer", "AlGaAs", "step", 0.15)
            p["Ls"] = st.number_input("Step width (nm)", 0.5, 100.0, 6.0, 0.5)
        p["Lb"] = st.number_input("Outer barriers (nm)", 2.0, 300.0, 25.0, 1.0)
    elif shape in ("Graded well", "Multiple graded wells", "Sawtooth", "Measured profile"):
        p["graded_alloy"] = st.selectbox("Graded alloy", g.TERNARY, index=g.TERNARY.index("AlGaAs"))
        if shape != "Measured profile":
            if shape != "Sawtooth":
                p["graded_shape"] = st.selectbox("Shape", g.GRADED_SHAPES)
            p["x_center"] = st.number_input("x at the well bottom", 0.0, 1.0, 0.0, 0.01)
        p["x_edge"] = st.number_input("x of the barrier", 0.0, 1.0, 0.3, 0.01)
        if shape != "Measured profile":
            p["Lw"] = st.number_input("Width of each well (nm)", 1.0, 300.0, 40.0 if shape == "Graded well" else 20.0, 1.0)
        if shape in ("Multiple graded wells", "Sawtooth"):
            p["n_wells"] = st.number_input("Number of wells", 2, 30, 3 if shape != "Sawtooth" else 4)
            p["gap"] = st.number_input("Barrier between wells (nm)", 0.0, 100.0, 5.0 if shape != "Sawtooth" else 0.0, 0.5)
        if shape == "Measured profile":
            up = st.file_uploader("Profile CSV: columns z (nm), x", type=["csv", "txt"])
            if up is not None:
                data = np.genfromtxt(io.StringIO(up.getvalue().decode()), delimiter=",", skip_header=1)
                data = data[~np.isnan(data).any(axis=1)]
                p["profile_z"], p["profile_x"] = data[:, 0].tolist(), data[:, 1].tolist()
            else:
                st.caption("No file yet: a Gaussian example profile is used.")
                z = np.linspace(0, 30, 61)
                p["profile_z"], p["profile_x"] = z.tolist(), (0.3 * (1 - np.exp(-((z - 15) / 6) ** 2))).tolist()
        p["Lb"] = st.number_input("Barriers (nm)", 0.0, 300.0, 20.0, 1.0)
    elif shape == "Uniform box (for V_ext)":
        p["box_material"], p["xbox"], p["ybox"] = material_input("Material", "GaAs", "box", 0.0)
        p["box_length"] = st.number_input("Length (nm)", 5.0, 2000.0, 100.0, 5.0)
        st.caption("Add a formula potential in section 3 (e.g. parabolic) to create the well.")
    else:
        st.caption("Layers from left to right. x_end ≠ x makes a linearly graded layer. "
                   "Doping in cm⁻³, typed like 1e17, 10e17 or 1x10^17 (negative = acceptors); needs Schrödinger–Poisson (section 4).")
        default = [
            dict(material="AlGaAs", thickness=25.0, x=0.3, x_end=np.nan, y=np.nan, doping="0"),
            dict(material="GaAs", thickness=10.0, x=np.nan, x_end=np.nan, y=np.nan, doping="0"),
            dict(material="AlGaAs", thickness=25.0, x=0.3, x_end=np.nan, y=np.nan, doping="0"),
        ]
        column_config = {
            "material": st.column_config.SelectboxColumn("material", options=g.ALL_MATERIALS),
            "thickness": st.column_config.NumberColumn("thickness (nm)", min_value=0.0, format="%.2f"),
            "x": st.column_config.NumberColumn("x", min_value=0.0, max_value=1.0, format="%.3f"),
            "x_end": st.column_config.NumberColumn("x_end (graded)", min_value=0.0, max_value=1.0, format="%.3f"),
            "y": st.column_config.NumberColumn("y", min_value=0.0, max_value=1.0, format="%.3f"),
            "doping": st.column_config.TextColumn(
                "doping (cm⁻³)",
                help="Type e.g. 1e17, 10e17, 3.5e18 or 1x10^17; negative = acceptors (-2e17). "
                     "Switch on Schrödinger–Poisson in section 4."),
        }
        rows = st.data_editor(default, num_rows="dynamic", use_container_width=True, column_config=column_config)
        p["layers"] = list(rows)
    p["substrate"] = st.selectbox("Substrate", ["GaAs", "InP", "GaSb", "InAs", "GaP", "InSb"])

    st.header("2 · Conditions")
    p["T"] = st.slider("Temperature (K)", 1.0, 600.0, 300.0, 1.0)
    p["field"] = st.slider("Electric field (kV/cm)", -300.0, 300.0, 0.0, 1.0)
    p["Bz"] = st.slider("Magnetic field along z, Bz (T)", 0.0, 40.0, 0.0, 0.5)
    p["Bpar"] = st.slider("In-plane magnetic field, Bpar (T)", 0.0, 40.0, 0.0, 0.5)
    if p["Bpar"] > 0:
        p["z0"] = st.number_input("Orbit centre z0 (nm, 0 = centre)", 0.0, 2000.0, 0.0)
    p["strain"] = st.checkbox("Strain (layers strained to the substrate)", False)
    p["nonparabolic"] = st.selectbox("Non-parabolicity", ["off", "three-band Kane", "two-band Kane"])

    st.header("3 · Extra potential (V_ext)")
    st.caption("Added to both bands, like an electric potential. Two shapes can be combined.")
    p["potentials"] = [potential_input(1), potential_input(2)]

    st.header("4 · Doping and numerics")
    p["self_consistent"] = st.checkbox("Self-consistent Schrödinger–Poisson (needs doped layers)", False)
    if p["self_consistent"]:
        p["sp_mix"] = st.number_input("Mixing", 0.01, 1.0, 0.1, 0.01)
        p["sp_tol"] = st.number_input("Tolerance (eV)", 1e-7, 1e-3, 1e-5, format="%.1e")
        p["sp_bc"] = st.selectbox("Right boundary", ["neumann", "dirichlet"])
    p["dz"] = st.select_slider("Grid step dz (nm)", [0.01, 0.02, 0.05, 0.1, 0.2], value=0.1)
    p["n_e"] = st.number_input("Electron states", 1, 60, 8)
    p["n_h"] = st.number_input("Hole states (heavy and light, each)", 1, 60, 8)

KEY = json.dumps(p, sort_keys=True, default=str)


@st.cache_resource(show_spinner=False, max_entries=24)
def run(key):
    return g.solve(json.loads(key))


try:
    with st.spinner("Solving..."):
        s, sol, msgs = run(KEY)
except Exception as exc:
    st.error(f"Could not solve this structure: {exc}")
    st.stop()
for m in msgs + g.bound_check(sol):
    st.warning(m)

levels = g.level_tables(sol)
inter, intra = g.transition_tables(sol)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Electron e0", f"{levels[0]['electron (meV)']:.2f} meV")
c2.metric("Heavy hole hh0", f"{levels[0]['heavy hole (meV)']:.2f} meV")
c3.metric("Lowest transition", f"{inter[0]['wavelength (nm)']:.1f} nm" if inter else "-")
c4.metric("Bound electron states", int(sol.e_bound.sum()))
lo_quick = g.lo_table(sol)
c5.metric("LO lifetime e1→e0", f"{lo_quick[0]['lifetime (ps)']:.3f} ps" if lo_quick else "-")

page = st.radio("Results", PAGES, horizontal=True, label_visibility="collapsed")
st.divider()

# ================================================================ pages
if page == "Band diagram":
    n_show = st.slider("States to draw", 1, 16, 4)
    show(g.band_figure(sol, s, n_show=n_show))
    show(g.conduction_figure(sol, s, n_show=max(n_show, 6)))
    h, cols = g.wavefunction_columns(sol)
    csv_button("Band edges and all wavefunctions", g.to_csv(header=h, columns=cols), "wavefunctions.csv")
    st.text(g.text_of(s.describe, T=p["T"]))
    csv_button("Layer table", g.to_csv([{"layer": i, "material": l.material, "thickness (nm)": l.thickness, "x": l.x,
                                         "y": l.y, "doping (cm^-3)": l.doping} for i, l in enumerate(s.layers)]), "layers.csv")

elif page == "Density maps":
    gam = st.slider("Level broadening for the maps (meV)", 0.5, 20.0, 3.0)
    for carrier in ("electrons", "holes"):
        show(g.ldos_figure(sol, s, carrier, gam))
        E, M = g.ldos_map(sol, carrier, gam, nE=300)
        csv_button(f"LDOS map, {carrier} (energy rows × position columns)",
                   g.to_csv(header=["E_eV"] + [f"z={z:.3f}" for z in sol.z[::5]], columns=[E] + [M[:, i] for i in range(0, M.shape[1], 5)]),
                   f"ldos_{carrier}.csv")
    show(g.density_waterfall(sol, n_show=min(10, len(sol.Ee))))

elif page == "States":
    st.caption("Electrons: meV above the conduction-band minimum. Holes: meV below the valence-band maximum.")
    st.dataframe(levels, use_container_width=True)
    csv_button("Energy levels", g.to_csv(levels), "levels.csv")
    ana = g.state_analysis(s, sol)
    st.subheader("Where each state lives")
    st.dataframe(ana, use_container_width=True)
    csv_button("State analysis (<z>, spread, region, mass)", g.to_csv(ana), "state_analysis.csv")
    reg = g.region_table(sol)
    st.dataframe(reg, use_container_width=True)
    csv_button("Probability of every electron state in every region", g.to_csv(reg), "region_probabilities.csv")

elif page == "Transitions":
    all_tr = st.checkbox("Include transitions involving unbound states", False)
    i2, i3 = g.transition_tables(sol, bound_only=not all_tr)
    st.subheader("Interband (electron to hole)")
    st.dataframe(i2, use_container_width=True)
    csv_button("Interband transitions", g.to_csv(i2), "interband.csv")
    st.subheader("Intersubband (electron to electron)")
    if i3:
        st.dataframe(i3, use_container_width=True)
        csv_button("Intersubband transitions", g.to_csv(i3), "intersubband.csv")
    else:
        st.info("No intersubband transitions between bound states.")

elif page == "Spectra":
    kind = st.radio("Spectrum", g.SPECTRA, horizontal=True)
    cA, cB, cC, cD = st.columns(4)
    n2d = cA.number_input("Electron density (cm⁻²)", 1e9, 1e14, 3e12 if kind == "gain" else 5e11, format="%.2e")
    p2d = cB.number_input("Hole density (cm⁻², gain)", 1e9, 1e14, n2d, format="%.2e")
    gamma = cC.slider("Broadening (meV)", 1.0, 30.0, 6.0) / 1000
    pol = cD.radio("Polarisation", ["TE", "TM"]) if kind not in ("intersubband absorption", "interband absorption (relative)") else "TE"
    E, v, lab = g.spectrum(sol, kind, n2d=n2d, p2d=p2d, gamma=gamma, polarization=pol)
    isb = kind == "intersubband absorption"
    show(g.line_figure(E * 1000 if isb else E, [v * 100 if isb else v], [""],
                       "Photon energy (meV)" if isb else "Photon energy (eV)",
                       "absorption per pass (%)" if isb else lab, kind.capitalize(), fill=True, hlines=[0]))
    csv_button(f"Spectrum: {kind}", g.to_csv(header=["photon_energy_eV", "wavelength_nm", lab.replace(",", "")],
                                             columns=[E, 1239.84 / E, v]), "spectrum.csv")

elif page == "Gain map":
    cA, cB, cC = st.columns(3)
    n1 = cA.number_input("Lowest density (cm⁻²)", 1e10, 1e14, 3e11, format="%.2e")
    n2 = cB.number_input("Highest density (cm⁻²)", 1e10, 1e14, 8e12, format="%.2e")
    pol = cC.radio("Polarisation", ["TE", "TM"], horizontal=True)
    dens = np.geomspace(n1, n2, 30)
    with st.spinner("Computing the gain map..."):
        E, nn, G = g.gain_map(sol, dens, polarization=pol)
    show(g.map_figure(E, nn / 1e12, G, "Photon energy (eV)", "Carrier density per well (10¹² cm⁻²)", "material gain (cm⁻¹)",
                      f"Gain map ({pol}): red = gain, blue = absorption", symmetric=True))
    peak = G.max(axis=1)
    ntr = float(np.interp(0, peak, dens)) if peak.min() < 0 < peak.max() else None
    show(g.line_figure(dens / 1e12, [peak], [""], "Carrier density per well (10¹² cm⁻²)", "peak gain (cm⁻¹)",
                       "Peak gain versus density" + (f" · transparency {ntr:.2e} cm⁻²" if ntr else ""), hlines=[0]))
    csv_button("Gain map (density rows × photon-energy columns)",
               g.to_csv(header=["density_cm-2"] + [f"E={e:.4f}" for e in E[::5]], columns=[dens] + [G[:, i] for i in range(0, G.shape[1], 5)]),
               "gain_map.csv")
    csv_button("Peak gain versus density", g.to_csv(header=["density_cm-2", "peak_gain_cm-1"], columns=[dens, peak]), "peak_gain.csv")

elif page == "LO phonons":
    st.subheader("LO-phonon scattering")
    st.caption("Emission of longitudinal-optical phonons (Fröhlich coupling). From the subband bottom, or thermally averaged "
               "over hot electrons (emission and absorption, all pairs of levels).")
    lo = g.lo_table(sol)
    if lo:
        st.dataframe(lo, use_container_width=True)
        csv_button("LO-phonon rates (subband bottom)", g.to_csv(lo), "lo_rates.csv")
    else:
        st.info("No transition from the subband bottom: every spacing is below the LO-phonon energy. Use the thermal rates below.")
    cA, cB = st.columns(2)
    Te = cA.number_input("Electron temperature (K)", 1.0, 2000.0, float(p["T"]))
    nlo = cB.number_input("Sheet density for Fermi-Dirac (0 = Boltzmann)", 0.0, 1e14, 0.0, format="%.2e")
    with st.spinner("Thermal rates..."):
        lot = g.lo_table(sol, thermal=True, Te=Te, n2d=nlo or None)
    nb = int(sol.e_bound.sum())
    if lot and nb > 1:
        show(g.lo_matrix_figure(lot, nb))
    st.dataframe(lot, use_container_width=True)
    csv_button("Thermal LO-phonon rates", g.to_csv(lot), "lo_rates_thermal.csv")
    if st.checkbox("Lifetimes versus electron temperature"):
        Tes = np.array([10, 30, 50, 77, 120, 200, 300, 450, 700])
        with st.spinner("Temperature sweep..."):
            Tv, curves = g.lo_te_sweep(sol, Tes, n2d=nlo or None)
        if curves:
            show(g.line_figure(Tv, list(curves.values()), list(curves.keys()), "Electron temperature (K)", "Lifetime (ps)",
                               "Thermal LO-phonon lifetimes", logy=True))
            csv_button("Lifetimes versus electron temperature", g.to_csv(header=["Te_K"] + list(curves), columns=[Tv] + list(curves.values())),
                       "lo_lifetimes_vs_Te.csv")

elif page == "Statistics":
    cA, cB, cC = st.columns(3)
    ne = cA.number_input("Electron sheet density (cm⁻²)", 1e8, 1e14, 5e11, format="%.2e")
    nh = cB.number_input("Hole sheet density (cm⁻²)", 1e8, 1e14, 5e11, format="%.2e")
    Ts = cC.number_input("Temperature (K)", 1.0, 600.0, float(p["T"]))
    EF, EFv, fr = g.fermi_tables(sol, ne, nh, T=Ts)
    m1, m2 = st.columns(2)
    m1.metric("Electron Fermi level", f"{EF:.5f} eV", f"{(EF - sol.Ee[0]) * 1000:+.2f} meV from e0")
    m2.metric("Hole quasi-Fermi level", f"{EFv:.5f} eV", f"{(sol.Ehh[0] - EFv) * 1000:+.2f} meV from hh0")
    st.dataframe(fr, use_container_width=True)
    csv_button("Fermi levels and populations",
               g.to_csv([{"electron Fermi level (eV)": EF, "hole Fermi level (eV)": EFv}]) + "\n" + g.to_csv(fr), "fermi_populations.csv")
    if st.checkbox("Fermi level versus density"):
        dens = np.geomspace(1e10, 2e13, 40)
        EFs = [g.fermi_tables(sol, d, d, T=Ts)[0] for d in dens]
        show(g.line_figure(dens, [(np.array(EFs) - sol.Ee[0]) * 1000], [""], "Sheet density (cm⁻²)", "E$_F$ − e$_0$ (meV)",
                           "Fermi level versus sheet density", logx=True, hlines=[0]))
        csv_button("Fermi level versus density", g.to_csv(header=["density_cm-2", "EF_eV"], columns=[dens, EFs]), "fermi_vs_density.csv")

elif page == "Exciton":
    cA, cB, cC = st.columns(3)
    ie = cA.number_input("Electron level", 0, max(0, len(sol.Ee) - 1), 0)
    ih = cB.number_input("Hole level", 0, max(0, len(sol.Ehh) - 1), 0)
    hole = cC.radio("Hole", ["hh", "lh"], horizontal=True)
    with st.spinner("Excitons..."):
        rows = [g.exciton_row(sol, ie, ih, hole)]
        if (ie, ih, hole) != (0, 0, "hh"):
            rows.append(g.exciton_row(sol, 0, 0, "hh"))
        if hole != "lh":
            rows.append(g.exciton_row(sol, 0, 0, "lh"))
    st.dataframe(rows, use_container_width=True)
    csv_button("Excitons", g.to_csv(rows), "excitons.csv")
    st.caption("Variational 1s exciton (binding energy is a lower bound). Best results at low temperature with dz = 0.05 nm.")

elif page == "Tunnelling":
    st.caption("Transmission through the whole structure; energies above the conduction-band edge of the first layer.")
    cA, cB, cC, cD = st.columns(4)
    e1 = cA.number_input("From (meV)", 0.01, 5000.0, 1.0)
    e2 = cB.number_input("To (meV)", 0.1, 5000.0, 400.0)
    npt = cC.number_input("Points", 200, 40000, 4000, 200)
    tnp = cD.selectbox("Mass", ["as solver", "parabolic", "non-parabolic"])
    with st.spinner("Transmission..."):
        Em, Tt, res = g.transmission_data(sol, e1, e2, int(npt),
                                          nonparabolic={"as solver": None, "parabolic": False, "non-parabolic": True}[tnp])
    marks = [(r["resonance (meV)"], r["T max"], f"{r['resonance (meV)']:.2f} meV") for r in res]
    show(g.line_figure(Em, [np.maximum(Tt, 1e-12)], [""], "Electron energy (meV)", "Transmission T(E)", "Transmission",
                       logy=True, fill=True, markers=marks))
    st.dataframe(res, use_container_width=True) if res else st.info("No resonance above T = 0.3 in this range.")
    csv_button("Transmission T(E)", g.to_csv(header=["E_meV", "T"], columns=[Em, Tt]), "transmission.csv")
    csv_button("Resonances", g.to_csv(res), "resonances.csv")

elif page == "Minibands":
    if getattr(s, "period", None):
        span = st.number_input("Search range above the band minimum (meV)", 50.0, 3000.0, 500.0)
        with st.spinner("Minibands..."):
            E, q, d, mb, base = g.minibands_table(sol, span)
        st.dataframe(mb, use_container_width=True)
        show(g.minibands_figure(E, q, d, mb, base))
        csv_button("Minibands", g.to_csv(mb), "minibands.csv")
        csv_button("Dispersion q(E)", g.to_csv(header=["E_meV", "q_per_nm"], columns=[(E - base) * 1000, np.nan_to_num(q, nan=-1)]),
                   "superlattice_dispersion.csv")
    else:
        st.info("Minibands need a periodic structure: choose Superlattice, Multiple graded wells or Sawtooth.")

elif page == "Stark effect":
    cA, cB, cC = st.columns(3)
    f1 = cA.number_input("From (kV/cm)", -500.0, 500.0, -80.0)
    f2 = cB.number_input("To (kV/cm)", -500.0, 500.0, 80.0)
    nf = cC.number_input("Steps", 3, 151, 41)
    try:
        with st.spinner("Stark sweep..."):
            stk = g.stark_data(json.loads(KEY), np.linspace(f1, f2, int(nf)))
    except ValueError as exc:
        st.warning(f"Stark sweep not possible: {exc}")
        st.stop()
    F = stk["F"]
    Es, Fm, A = g.stark_map(stk)
    show(g.map_figure(Fm, Es, A, "Electric field (kV/cm)", "Photon energy (eV)", "absorption (arb.)",
                      "Quantum-confined Stark effect: absorption map"))
    i0 = int(np.argmin(abs(F)))
    show(g.line_figure(F, [(stk["E_e0hh0"] - stk["E_e0hh0"][i0]) * 1000, (stk["E_e0lh0"] - stk["E_e0lh0"][i0]) * 1000],
                       ["e0-hh0", "e0-lh0"], "Electric field (kV/cm)", "Stark shift (meV)", "Transition shifts"))
    show(g.line_figure(F, [stk["ov_e0hh0"], stk["ov_e0lh0"]], ["e0-hh0", "e0-lh0"], "Electric field (kV/cm)",
                       "Overlap |⟨e|h⟩|²", "Electron-hole overlap"))
    if np.all(np.isnan(stk["E01"])):
        st.info("Only one bound electron level: no e0→e1 intersubband values (shown as empty in the CSV).")
    csv_button("Stark sweep (energies, overlaps, dipole, positions)", g.to_csv(header=list(stk.keys()), columns=list(stk.values())),
               "stark_sweep.csv")
    csv_button("Stark absorption map (energy rows × field columns)",
               g.to_csv(header=["E_eV"] + [f"F={f:.1f}" for f in Fm], columns=[Es] + [A[:, i] for i in range(A.shape[1])]), "stark_map.csv")

elif page == "Magnetic field":
    cA, cB, cC, cD = st.columns(4)
    B = cA.number_input("B for levels (T)", 0.1, 60.0, float(p["Bz"]) if p["Bz"] > 0 else 5.0)
    b1 = cB.number_input("Fan from (T)", 0.05, 60.0, 0.5)
    b2 = cC.number_input("Fan to (T)", 0.1, 60.0, 10.0)
    nB = cD.number_input("Sheet density (cm⁻²)", 1e9, 1e13, 3e11, format="%.2e")
    gam = st.slider("Landau-level broadening (meV)", 0.1, 5.0, 1.0) / 1000
    if not np.any(sol.e_bound):
        st.error("Landau levels need bound electron subbands, and this structure has none: the well's conduction band "
                 "lies above the barriers' (well and barrier swapped?). For InGaAs/GaAs use InGaAs as the well and "
                 "GaAs as the barrier.")
        st.stop()
    with st.spinner("Landau levels..."):
        Bv = np.linspace(b1, b2, 160)
        lev, levarr, EFB, nu, Eg, dos, EFb, spin = g.landau_data(sol, B, Bv, nB, gamma=gam)
    if levarr.size == 0 or not np.isfinite(levarr).any():
        st.error("No Landau levels in this structure (no bound subbands).")
        st.stop()
    with st.spinner("Landau-level map..."):
        Bm = np.linspace(b1, b2, 600)                                   # fine field grid: no aliasing
        Em, Bm, D, _ = g.landau_map(sol, Bm, gamma=gam)
        EFm = np.interp(Bm, Bv, EFB)                                    # Fermi level from the fan
    fig, note = g.landau_map_figure(Em, Bm, D, EFm)
    show(fig)
    if note:
        st.info(note + f" (temperature {p['T']:g} K, density {nB:.1e} cm⁻²). Landau quantisation is best seen at "
                "low temperature, e.g. 4.2 K.")
    fig_fan, _ = g.fan_figure(Bv, levarr, EFB, sol=sol)
    show(fig_fan)
    show(g.filling_figure(Bv, nu, nB))
    st.caption("ν depends only on the density and the field (ν = n h / eB). Red lines: even integer fillings, where a "
               "spin-degenerate set of Landau levels is completely full — the Fermi level jumps there (fan above) and "
               "measured resistance shows Shubnikov–de Haas minima and quantum Hall plateaus. Grey dotted: odd fillings "
               "(spin-split levels, visible at high field and low temperature).")
    m1, m2 = st.columns(2)
    m1.metric(f"Fermi level at {B:g} T", f"{EFb:.5f} eV")
    m2.metric("Cyclotron energy", f"{lev[0]['cyclotron energy (meV)']:.3f} meV" if lev else "-")
    st.dataframe(spin, use_container_width=True)
    st.dataframe(lev, use_container_width=True)
    csv_button("Landau levels", g.to_csv(lev), "landau_levels.csv")
    csv_button("g-factors and Zeeman splittings", g.to_csv(spin), "zeeman.csv")
    csv_button("Landau fan (levels, Fermi level, filling factor)",
               g.to_csv(header=["B_T", "EF_eV", "filling_factor"] + [f"level_{i}_eV" for i in range(levarr.shape[1])],
                        columns=[Bv, EFB, nu] + [levarr[:, i] for i in range(levarr.shape[1])]), "landau_fan.csv")
    csv_button("Landau DOS map (energy rows × field columns)",
               g.to_csv(header=["E_eV"] + [f"B={b:.3f}" for b in Bm[::4]], columns=[Em] + [D[:, i] for i in range(0, D.shape[1], 4)]),
               "landau_dos_map.csv")
    st.subheader("In-plane field: dispersion E(ky)")
    if p["Bpar"] > 0:
        kmax = st.number_input("ky range (1/nm)", 0.01, 1.0, 0.2)
        ky, Ed = g.dispersion_data(sol, kmax)
        show(g.line_figure(ky, [Ed[:, i] for i in range(Ed.shape[1])], [f"e{i}" for i in range(Ed.shape[1])],
                           "k_y (1/nm)", "Energy (eV)", f"In-plane dispersion at B∥ = {p['Bpar']:g} T"))
        csv_button("In-plane dispersion", g.to_csv(header=["ky_per_nm"] + [f"E{i}_eV" for i in range(Ed.shape[1])],
                                                   columns=[ky] + [Ed[:, i] for i in range(Ed.shape[1])]), "inplane_dispersion.csv")
    else:
        st.info("Set an in-plane field Bpar in the sidebar.")

elif page == "Strain":
    if p["strain"] and sol.layer_strain:
        show(g.strain_figure(s, sol))
        srows = [{"layer": str(l), "eps_xx (%)": round(d["exx"] * 100, 4), "eps_zz (%)": round(d["ezz"] * 100, 4),
                  "dEc (meV)": round(d["dEc"] * 1000, 2), "dEhh (meV)": round(d["dEhh"] * 1000, 2), "dElh (meV)": round(d["dElh"] * 1000, 2),
                  "critical thickness (nm)": round(d["hc"], 2) if d["hc"] < 1e6 else None,
                  "exceeds h_c": bool(l.thickness > d["hc"])} for l, d in zip(s.layers, sol.layer_strain)]
        st.dataframe(srows, use_container_width=True)
        csv_button("Strain of every layer", g.to_csv(srows), "strain.csv")
    else:
        st.info("Switch on Strain in section 2 of the sidebar.")

elif page == "Materials":
    name, x, y = material_input("Material", "AlGaAs", "lookup", 0.3)
    Tm = st.number_input("Temperature (K)", 1.0, 1000.0, float(p["T"]))
    mt = g.material_table(name, x, y, T=Tm)
    st.dataframe(mt, use_container_width=True)
    csv_button("Material parameters", g.to_csv(mt), "material.csv")
    st.subheader("Lattice matching")
    cA, cB, cC = st.columns(3)
    alloy = cA.selectbox("Alloy", g.TERNARY + g.QUATERNARY, index=g.TERNARY.index("InGaAs"))
    subm = cB.selectbox("Substrate for matching", ["InP", "GaAs", "GaSb", "InAs"])
    fixed = None
    if alloy in g.QUATERNARY:
        fx = cC.selectbox("Fixed composition", ["y", "x"])
        fixed = (fx, st.number_input(f"value of {fx}", 0.0, 1.0, 0.5))
    try:
        st.success(f"Lattice-matched composition: {g.lattice_matched_value(alloy, subm, fixed):.4f}")
    except Exception as exc:
        st.error(str(exc))
    st.subheader("All binaries")
    am = g.all_materials_table(Tm)
    st.dataframe(am, use_container_width=True)
    csv_button("Binary materials", g.to_csv(am), "binaries.csv")

elif page == "Report":
    nrep = st.number_input("Sheet density for the report (0 = none)", 0.0, 1e14, 0.0, format="%.2e")
    rep = g.text_of(sol.report, n2d_cm2=nrep or None)
    st.text(rep)
    st.download_button("⬇ Full text report", rep, "hetero1d_report.txt", "text/plain")
    st.code(f"import gui_core as g\np = {json.dumps(json.loads(KEY), default=str)}\nstructure, sol, warnings = g.solve(p)\nsol.report()",
            language="python")

elif page == "Download all":
    st.write("One ZIP file with every main result of the current calculation as CSV, plus the text report and the inputs.")
    if st.button("Prepare ZIP"):
        with st.spinner("Collecting all results..."):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                h, cols = g.wavefunction_columns(sol)
                z.writestr("wavefunctions.csv", g.to_csv(header=h, columns=cols))
                z.writestr("levels.csv", g.to_csv(levels))
                z.writestr("state_analysis.csv", g.to_csv(g.state_analysis(s, sol)))
                z.writestr("region_probabilities.csv", g.to_csv(g.region_table(sol)))
                z.writestr("interband.csv", g.to_csv(inter))
                z.writestr("intersubband.csv", g.to_csv(intra))
                for kd in g.SPECTRA:
                    E, v, lab = g.spectrum(sol, kd, n2d=3e12 if kd == "gain" else 5e11)
                    z.writestr(f"spectrum_{kd.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
                               g.to_csv(header=["photon_energy_eV", lab.replace(",", "")], columns=[E, v]))
                z.writestr("fermi_populations_5e11.csv", g.to_csv(g.fermi_tables(sol, 5e11, 5e11)[2]))
                z.writestr("lo_rates.csv", g.to_csv(g.lo_table(sol)))
                z.writestr("lo_rates_thermal.csv", g.to_csv(g.lo_table(sol, thermal=True)))
                z.writestr("excitons.csv", g.to_csv([g.exciton_row(sol), g.exciton_row(sol, hole="lh")]))
                Em, Tt, res = g.transmission_data(sol, 1.0, 400.0)
                z.writestr("transmission.csv", g.to_csv(header=["E_meV", "T"], columns=[Em, Tt]))
                z.writestr("resonances.csv", g.to_csv(res))
                z.writestr("layers.csv", g.to_csv([{"layer": i, "material": l.material, "thickness (nm)": l.thickness, "x": l.x,
                                                    "y": l.y, "doping (cm^-3)": l.doping} for i, l in enumerate(s.layers)]))
                if p["strain"] and sol.layer_strain:
                    z.writestr("strain.csv", g.to_csv([{"layer": str(l), "eps_xx": d["exx"], "eps_zz": d["ezz"], "dEc_eV": d["dEc"],
                                                        "dEhh_eV": d["dEhh"], "dElh_eV": d["dElh"], "hc_nm": d["hc"]}
                                                       for l, d in zip(s.layers, sol.layer_strain)]))
                z.writestr("report.txt", g.text_of(sol.report))
                z.writestr("inputs.json", json.dumps(json.loads(KEY), indent=1))
        st.download_button("⬇ Download ZIP", buf.getvalue(), "hetero1d_results.zip", "application/zip")
