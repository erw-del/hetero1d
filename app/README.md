# hetero1d web app

Every feature of hetero1d 0.4.1 in the browser, with publication-quality figures and every result
downloadable as CSV (plus one ZIP with everything).

## Result pages
Band diagram · Density maps (local density of states) · States · Transitions · Spectra · Gain map ·
**LO phonons** · Statistics · Exciton · Tunnelling · Minibands · Stark effect · Magnetic field · Strain ·
Materials · Report · Download all

Choose the page from the list below the headline numbers; only the selected page is computed.

## Inputs (sidebar)
Structures: single, double, multiple and step wells; superlattices; graded wells of any shape; multiple graded
wells; sawtooth; measured composition profile (CSV upload); custom layer stacks with doping and graded
layers; uniform box for formula potentials. All 33 materials incl. quaternaries. Temperature, electric field,
Bz, in-plane Bpar, strain, non-parabolicity, all 10 V_ext shapes (two combinable), Schrödinger-Poisson.

## Run on your computer
```bash
pip install streamlit git+https://github.com/erw-del/hetero1d.git
streamlit run app/streamlit_app.py
```

## Files
- `streamlit_app.py`: the interface
- `gui_core.py`: calculations, tables, CSV export and figures (plain Python, reusable in notebooks)
- `requirements.txt`: packages for Streamlit Community Cloud
