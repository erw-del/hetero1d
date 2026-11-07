"""
hetero1d.strain -- biaxial strain of pseudomorphic layers grown on (001) substrates.

A layer whose lattice constant a differs from the substrate's a_sub is
compressed or stretched in the growth plane so that it fits the substrate
(pseudomorphic growth), and relaxes along the growth direction z:

    eps_xx = eps_yy = (a_sub - a) / a
    eps_zz = -2 (C12 / C11) eps_xx

The strain shifts the band edges (Bir-Pikus deformation potentials, single-band
form without spin-orbit coupling of the light hole):

    hydrostatic part  tr = eps_xx + eps_yy + eps_zz
    dEc  = a_c tr
    dEv  = -a_v tr                      (a_v as tabulated by Vurgaftman: negative)
    shear part        S  = b (eps_xx - eps_zz)
    dE_hh = dEv + S                     compressive strain (eps_xx < 0): heavy hole up
    dE_lh = dEv - S                     tensile strain (eps_xx > 0):     light hole up
    (from 0.4: the light hole is coupled to the split-off band, see band_shifts)

Critical thickness (Matthews-Blakeslee, 60-degree dislocations): above h_c a
strained layer relaxes by forming misfit dislocations, and the pseudomorphic
assumption (and usually the device) fails. Buried (capped) layers use the
double-kink value, twice that of a surface layer. Real layers can exceed h_c
metastably (low growth temperature), so treat it as a design guideline.

Parameters: I. Vurgaftman, J. R. Meyer, L. R. Ram-Mohan, J. Appl. Phys. 89, 5815 (2001).
"""
import numpy as np


def biaxial_strain(a, a_sub, C11, C12):
    """In-plane and growth-direction strain of a pseudomorphic (001) layer."""
    exx = (a_sub - a) / a
    ezz = -2.0 * C12 / C11 * exx
    return exx, ezz


def band_shifts(p, exx, ezz, split_off=True):
    """
    Band-edge shifts (eV) for a material parameter dict p with keys ac, av, b
    (and Dso for the split-off coupling). Returns (dEc, dEhh, dElh).

    The heavy hole is decoupled at k = 0: dEhh = dEv + S.
    The light hole couples to the split-off band through the shear strain; the
    2x2 problem at k = 0 gives (with Q = -S, Delta = spin-orbit splitting)
        dElh = dEv + (Q - Delta + sqrt(Delta^2 + 2 Delta Q + 9 Q^2)) / 2
    which tends to dEv - S for a large Delta (the uncoupled result).
    """
    tr = 2 * exx + ezz
    dEc = p["ac"] * tr
    dEv = -p["av"] * tr
    S = p["b"] * (exx - ezz)
    D = p.get("Dso", np.nan)
    if split_off and D is not None and np.isfinite(D) and D > 0:
        Q = -S
        dElh = dEv + 0.5 * (Q - D + np.sqrt(D * D + 2 * D * Q + 9 * Q * Q))
    else:
        dElh = dEv - S
    return dEc, dEv + S, dElh


def critical_thickness(a, a_sub, C11, C12, capped=True):
    """
    Matthews-Blakeslee critical thickness (nm), 60-degree misfit dislocations,
    in the standard form (as quoted by People & Bean 1985):

        h_c = b (1 - nu/4) / (4 pi f (1 + nu)) * (ln(h_c / b) + 1)

    with b = a/sqrt(2) and f the lattice mismatch. This is for an uncapped
    surface layer; a buried layer (e.g. a quantum well with a cap) is about twice
    as stable (double-kink criterion), so capped=True returns 2 x h_c.
    Returns inf for an unstrained layer.
    """
    f = abs(a_sub - a) / a_sub
    if f < 1e-6:
        return np.inf
    nu = C12 / (C11 + C12)                     # Poisson ratio for (001)
    bd = a / np.sqrt(2) * 0.1                  # Burgers vector a/sqrt(2), Angstrom -> nm
    pref = bd * (1 - nu / 4) / (4 * np.pi * f * (1 + nu))
    h = 10 * bd
    for _ in range(500):                       # solve h = pref (ln(h/b) + 1) by iteration
        h_new = max(pref * (np.log(h / bd) + 1), bd)
        if abs(h_new - h) < 1e-10:
            break
        h = h_new
    return 2 * h if capped else h


def layer_strain(p, a_sub, capped=True):
    """
    Full strain information for one layer: dict with exx, ezz, dEc, dEhh, dElh,
    hc (critical thickness, nm). p must contain a, C11, C12, ac, av, b.
    """
    if any(p.get(k) is None for k in ("a", "C11", "C12", "ac", "av", "b")):
        return dict(exx=0.0, ezz=0.0, dEc=0.0, dEhh=0.0, dElh=0.0, hc=np.inf, missing=True)
    exx, ezz = biaxial_strain(p["a"], a_sub, p["C11"], p["C12"])
    dEc, dEhh, dElh = band_shifts(p, exx, ezz)
    hc = critical_thickness(p["a"], a_sub, p["C11"], p["C12"], capped=capped)
    return dict(exx=exx, ezz=ezz, dEc=dEc, dEhh=dEhh, dElh=dElh, hc=hc, missing=False)
