# hetero1d web app

Every feature of hetero1d 0.4.1 in the browser, and every result downloadable as CSV
(plus one ZIP with everything).

## What it covers
| Area | Features |
|---|---|
| Structures | single, double, multiple and step wells; superlattices; graded wells of any shape (parabolic, triangular left/right, V, half-parabolic, cosine, square); multiple graded wells; sawtooth; measured composition profile (CSV upload); custom layer stacks with doping and linearly graded layers; uniform box for formula potentials; all 33 materials incl. quaternaries; substrate |
| Conditions | temperature, electric field, Bz, in-plane field Bpar with orbit centre, strain, non-parabolicity (two- and three-band Kane) |
| Potentials | all 10 V_ext shapes of hetero1d.potentials, two combinable |
| Doping | self-consistent Schrödinger-Poisson (mixing, tolerance, boundary condition) |
| States | energies, bound flags, <z>, spread, level masses, probability per region, all wavefunctions |
| Optics | interband and intersubband transitions; intersubband absorption, interband absorption (cm^-1 and relative), gain with separate electron and hole densities, TE/TM |
| Statistics | electron and hole (quasi-)Fermi levels and populations at any temperature |
| Excitons | any electron/hole level, heavy or light hole |
| Scattering | LO-phonon rates from the subband bottom and thermal (hot electrons, Boltzmann or Fermi-Dirac) |
| Tunnelling | transmission T(E), parabolic or non-parabolic, resonances with widths and lifetimes |
| Superlattices | minibands and dispersion |
| Electric field | Stark sweep: shifts, overlaps, intersubband energy and dipole, positions |
| Magnetic field | Landau levels, Landau fan with Fermi level and filling factor, Landau DOS, g-factors, Zeeman splitting, in-plane dispersion |
| Materials | parameters of any material at any temperature (incl. strain parameters), lattice matching, binaries table |
| Output | CSV for every table, plot and spectrum; text report; one ZIP with everything; Python code to reproduce |

Slower analyses have an on/off switch in their tab, so the page stays fast (under 1 s for normal use).

## Run on your computer
```bash
pip install streamlit git+https://github.com/erw-del/hetero1d.git
streamlit run app/streamlit_app.py
```

## Files
- `streamlit_app.py`: the interface
- `gui_core.py`: every calculation, table, CSV and figure (plain Python, reusable in notebooks)
- `requirements.txt`: packages for Streamlit Community Cloud
