"""
hetero1d web app.   Run locally:  streamlit run app/streamlit_app.py
"""
import json
import streamlit as st
import hetero1d
import gui_core as g

st.set_page_config(page_title="hetero1d", page_icon="⚛️", layout="wide")
st.title("hetero1d: quantum wells in your browser")
st.caption(f"hetero1d {hetero1d.__version__} · single-band envelope-function solver · "
           "[GitHub](https://github.com/erw-del/hetero1d) · [PyPI](https://pypi.org/project/hetero1d/)")


def material_input(label, default, key, x_default=0.3, y_default=0.5):
    """Material selector with composition fields as needed. Returns (name, x, y)."""
    name = st.selectbox(label, g.ALL_MATERIALS, index=g.ALL_MATERIALS.index(default), key=key)
    nc = g.n_composition(name)
    x = y = None
    if nc >= 1:
        x = st.number_input(f"{label}: x", 0.0, 1.0, x_default, 0.01, format="%.3f", key=key + "_x")
    if nc == 2:
        y = st.number_input(f"{label}: y", 0.0, 1.0, y_default, 0.01, format="%.3f", key=key + "_y")
    return name, x, y


# ---------------------------------------------------------------- sidebar: structure and conditions
with st.sidebar:
    st.header("Structure")
    shape = st.selectbox("Type", g.SHAPES)
    p = dict(shape=shape)
    if shape in ("Single well", "Multiple wells", "Superlattice", "Step well"):
        p["well"], p["xw"], p["yw"] = material_input("Well", "GaAs", "well", 0.2)
        p["barrier"], p["xb"], p["yb"] = material_input("Barrier", "AlGaAs", "bar", 0.3)
        p["Lw"] = st.number_input("Well width (nm)", 0.5, 100.0, 10.0, 0.5)
        if shape in ("Multiple wells", "Superlattice"):
            p["n_wells"] = st.number_input("Number of wells / periods", 2, 30, 3)
            p["Lb_inner"] = st.number_input("Barrier between wells (nm)", 0.5, 100.0, 5.0, 0.5)
        if shape == "Step well":
            p["step"], p["xs"], p["ys"] = material_input("Step layer", "AlGaAs", "step", 0.15)
            p["Ls"] = st.number_input("Step width (nm)", 0.5, 50.0, 6.0, 0.5)
        p["Lb"] = st.number_input("Outer barriers (nm)", 2.0, 200.0, 25.0, 1.0)
    elif shape in ("Parabolic well", "Triangular well", "V-shaped well"):
        p["graded_alloy"] = st.selectbox("Graded alloy", g.TERNARY, index=g.TERNARY.index("AlGaAs"))
        p["x_center"] = st.number_input("x at the well bottom", 0.0, 1.0, 0.0, 0.01)
        p["x_edge"] = st.number_input("x of the barrier", 0.0, 1.0, 0.3, 0.01)
        p["Lw"] = st.number_input("Well width (nm)", 2.0, 200.0, 40.0 if shape == "Parabolic well" else 20.0, 1.0)
        p["Lb"] = st.number_input("Barriers (nm)", 2.0, 200.0, 20.0, 1.0)
        p["well"] = p["barrier"] = "GaAs"
    else:
        st.caption("Layers from left to right. Composition x (and y for quaternaries) where needed; doping in cm⁻³.")
        default = [dict(material="AlGaAs", thickness=25.0, x=0.3, y=None, doping=0.0),
                   dict(material="GaAs", thickness=10.0, x=None, y=None, doping=0.0),
                   dict(material="AlGaAs", thickness=25.0, x=0.3, y=None, doping=0.0)]
        rows = st.data_editor(default, num_rows="dynamic", use_container_width=True,
                              column_config={"material": st.column_config.SelectboxColumn("material", options=g.ALL_MATERIALS)})
        p["layers"] = list(rows)
        p["well"] = p["barrier"] = "GaAs"
        p["self_consistent"] = st.checkbox("Self-consistent with doping (Schrödinger-Poisson)", False)
    p["substrate"] = st.selectbox("Substrate", ["GaAs", "InP", "GaSb", "InAs"], index=0)

    st.header("Conditions")
    p["T"] = st.slider("Temperature (K)", 1.0, 500.0, 300.0, 1.0)
    p["field"] = st.slider("Electric field (kV/cm)", -200.0, 200.0, 0.0, 1.0)
    p["Bz"] = st.slider("Magnetic field along z (T)", 0.0, 30.0, 0.0, 0.5)
    p["strain"] = st.checkbox("Strain (layers strained to the substrate)", False)
    p["nonparabolic"] = st.selectbox("Non-parabolicity", ["off", "three-band Kane", "two-band Kane"])
    with st.expander("Numerics"):
        p["dz"] = st.select_slider("Grid step dz (nm)", [0.02, 0.05, 0.1, 0.2], value=0.1)
        p["n_e"] = st.number_input("Electron states", 1, 40, 6)
        p["n_h"] = st.number_input("Hole states (each)", 1, 40, 6)


@st.cache_resource(show_spinner=False, max_entries=20)
def run(key):
    return g.solve(json.loads(key))


try:
    with st.spinner("Solving..."):
        s, sol, msgs = run(json.dumps(p, sort_keys=True, default=str))
except Exception as exc:                               # show input errors instead of a traceback
    st.error(f"Could not solve this structure: {exc}")
    st.stop()

for m in msgs:
    st.warning(m)

# ---------------------------------------------------------------- headline numbers
levels = g.level_tables(sol)
inter, intra = g.transition_tables(sol)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Electron e0", f"{levels[0]['electron (meV)']:.2f} meV")
c2.metric("Heavy hole hh0", f"{levels[0]['heavy hole (meV)']:.2f} meV")
c3.metric("Lowest transition", f"{inter[0]['wavelength (nm)']:.1f} nm" if inter else "-")
c4.metric("Bound electron states", int(sol.e_bound.sum()))

tabs = st.tabs(["Band diagram", "Energies", "Transitions", "Spectra", "Exciton", "Strain", "Materials", "Export"])

with tabs[0]:
    st.pyplot(g.band_figure(sol))
    st.pyplot(g.conduction_figure(sol))

with tabs[1]:
    st.caption("Electrons: meV above the conduction-band minimum. Holes: meV below the valence-band maximum.")
    st.dataframe(levels, use_container_width=True)

with tabs[2]:
    st.subheader("Interband (electron to hole)")
    st.dataframe(inter, use_container_width=True)
    st.subheader("Intersubband (electron to electron)")
    st.dataframe(intra, use_container_width=True) if intra else st.info("No bound intersubband transitions.")

with tabs[3]:
    kind = st.radio("Spectrum", ["intersubband absorption", "interband absorption", "gain"], horizontal=True)
    n2d = st.number_input("Sheet density (cm⁻²)", 1e10, 5e13, 3e12 if kind == "gain" else 5e11, format="%.2e")
    gamma = st.slider("Broadening (meV)", 1.0, 20.0, 6.0) / 1000
    pol = st.radio("Polarisation", ["TE", "TM"], horizontal=True) if kind != "intersubband absorption" else "TE"
    st.pyplot(g.spectra_figure(sol, kind, n2d=n2d, gamma=gamma, polarization=pol))

with tabs[4]:
    if st.button("Compute the e0-hh0 exciton"):
        x = sol.exciton()
        e1, e2, e3 = st.columns(3)
        e1.metric("Binding energy", f"{x['Eb_meV']:.2f} meV")
        e2.metric("Radius", f"{x['lambda_nm']:.1f} nm")
        e3.metric("Exciton line", f"{1239.84 / x['E_transition_eV']:.1f} nm")
        st.caption("Variational (lower bound for the binding energy). Use a low temperature and dz = 0.05 nm for best results.")

with tabs[5]:
    if p["strain"] and sol.layer_strain:
        rows = [{"layer": str(l), "eps_xx (%)": round(d["exx"] * 100, 3), "eps_zz (%)": round(d["ezz"] * 100, 3),
                 "dEc (meV)": round(d["dEc"] * 1000, 1), "dEhh (meV)": round(d["dEhh"] * 1000, 1),
                 "dElh (meV)": round(d["dElh"] * 1000, 1), "critical thickness (nm)": (round(d["hc"], 1) if d["hc"] < 1e6 else None)}
                for l, d in zip(s.layers, sol.layer_strain)]
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("Switch on Strain in the sidebar to see the strain of every layer.")

with tabs[6]:
    name, x, y = material_input("Material", "AlGaAs", "lookup", 0.3)
    st.dataframe(g.material_table(name, x, y, T=p["T"]), use_container_width=True)

with tabs[7]:
    st.download_button("Download band edges and wavefunctions (CSV)", g.export_csv(sol), "hetero1d_result.csv", "text/csv")
    st.download_button("Download energy levels (CSV)",
                       "\n".join([",".join(levels[0].keys())] + [",".join(str(v) for v in r.values()) for r in levels]),
                       "hetero1d_levels.csv", "text/csv")
    st.caption("Reproduce this calculation in Python (gui_core.py is in the app folder of the repository):")
    st.code(f"import gui_core as g\np = {json.dumps(p, default=str)}\nstructure, sol, warnings = g.solve(p)\nsol.report()",
            language="python")
