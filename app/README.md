# hetero1d web app

A browser interface for hetero1d: build a structure (single, multiple and step wells, superlattices,
parabolic, triangular and V-shaped wells, or any custom layer stack), set temperature, electric and
magnetic fields, strain and non-parabolicity, and get band diagrams, energy levels, transitions,
absorption and gain spectra, excitons, strain tables and material data, with CSV downloads.

## Run on your computer
```bash
pip install streamlit git+https://github.com/erw-del/hetero1d.git
streamlit run app/streamlit_app.py
```
Your browser opens at http://localhost:8501.

## Files
- `streamlit_app.py`: the interface (Streamlit)
- `gui_core.py`: all calculations and figures (plain Python, reusable in notebooks)
- `requirements.txt`: packages for Streamlit Community Cloud
