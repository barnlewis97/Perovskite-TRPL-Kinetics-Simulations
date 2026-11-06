"""
Interactive GUI for simulating TRPL decays with the rate-equation models in
``globalfit_functions.py``.

For a chosen model and any number of initial carrier densities, the GUI plots

    (a) the normalised TRPL decay (log-log or lin-log),
    (b) the differential lifetime  tau_diff = (-1/2 dln(I_PL)/dt)^-1
    (c) the differential recombination coefficient
        k_diff = -1/(2 n_PL) dln(I_PL)/dt

against the PL-derived carrier coordinate n_PL(t) = n_0 sqrt(I_PL(t) / I_PL(0)),
following the definitions in paper_kdiff_BTD.pdf (Eq. S1).

Each parameter can be ticked off, in which case it is set to zero in the rate
equations and its process disappears from the band-diagram schematic. Units follow
globalfit_functions.py (cm^-3 and ns); tau_diff and k_diff are reported in s and
cm^3 s^-1.

Curves are drawn as dots. The derivatives always use the full time grid; "Dots
per curve" only thins what is drawn, so the same time points appear in all three
plots.

Run with:  python TRPL_GUI.py   (in an environment with jax and diffrax)
"""

import base64
import io
import math
import os
import re
import sys
import threading
import tkinter as tk
import warnings
from tkinter import filedialog, font as tkfont
from tkinter import messagebox, ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import IdentityTransform
from PIL import Image, ImageDraw, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import globalfit_functions as gf  # noqa: E402  (sets JAX x64 and EQX_ON_ERROR)

import diffrax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


# =============================================================================
# FullREM has rate equations only in globalfit_functions, so solve it here
# =============================================================================

@jax.jit
def TRPL_FullREM(t, N0, krad, ni, beta_n_t1, beta_p_t1, e_n_t1, e_p_t1, N_t1,
                 beta_n_t2, beta_p_t2, e_n_t2, e_p_t2, N_t2):
    """
    TRPL signal for the full two-trap SRH model, normalised to t = 0.

    Starts from n = p = N0 with both traps empty; I_PL ∝ krad * n * p.

    Returns
    -------
    tuple of jnp.ndarray
        (log10 signal, n, p, n_t1, n_t2).
    """
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(gf.FullREM_Model),
        diffrax.Kvaerno5(),
        t0=t[0],
        t1=t[-1],
        dt0=t[1] - t[0],
        y0=jnp.array([N0, N0, 0.0, 0.0]),
        args=jnp.array([krad, ni,
                        beta_n_t1, beta_p_t1, e_n_t1, e_p_t1, N_t1,
                        beta_n_t2, beta_p_t2, e_n_t2, e_p_t2, N_t2]),
        saveat=diffrax.SaveAt(ts=t),
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-8),
        max_steps=100000,
        throw=False,
    )
    n, p, nt1, nt2 = (sol.ys[:, i] for i in range(4))
    sig = krad * n * p
    sig = sig / sig[0]
    return jnp.log10(sig), n, p, nt1, nt2


# =============================================================================
# Model registry
# =============================================================================
# ``description`` is one sentence shown above the schematic.
# Each parameter is (name, default value, unit, description, mathtext symbol);
# symbols follow the BTD notation (T trapping, DT de-trapping, DP depopulation,
# HE hole emission, B bimolecular, A Auger; 1 shallow and 2 deep), whereas
# ``name`` is the argument name used in globalfit_functions.
# ``equations`` is a list of (left, right) mathtext rows, aligned on the relation
# at the start of ``right``.
# ``schematic`` lists the components of the band diagram (see draw_schematic).
# ``run`` maps (t, n0, params) to the log10 TRPL signal from globalfit_functions,
# called with zero background (the GUI adds its own background after normalisation
# so it means the same thing for every model).
#
# The default emission rates of the full models satisfy detailed balance with n_i
# (e_n e_p = beta_n beta_p n_i^2); larger values would generate carriers thermally
# and give the decays an artificial plateau.

RATE = "ns⁻¹"
BIMOL = "cm³ ns⁻¹"
AUGER = "cm⁶ ns⁻¹"
DENS = "cm⁻³"
# The same rate constants per second, shown in grey beside the inputs
PER_SECOND = {RATE: "s⁻¹", BIMOL: "cm³ s⁻¹", AUGER: "cm⁶ s⁻¹"}

# Trap level positions in the band diagram (see draw_schematic for the band edges)
SHALLOW, DEEP = 2.15, 1.85


def ddt(var):
    """Mathtext for d(var)/dt."""
    return rf"\dfrac{{\mathrm{{d}}{var}}}{{\mathrm{{d}}t}}"


def k(sub):
    """Mathtext for a rate constant with an upright subscript."""
    return rf"k_{{\mathrm{{{sub}}}}}"


def Nt(i):
    return rf"N_{{\mathrm{{T{i}}}}}"


def nt(i):
    return rf"n_{{\mathrm{{T{i}}}}}"


IPL = r"I_{\mathrm{PL}}"
DN, DP = r"\Delta n", r"\Delta p"
NT, nT = r"N_{\mathrm{T}}", r"n_{\mathrm{T}}"
P0 = "p_0"
Ni = r"n_{\mathrm{i}}"
QQ = r"\qquad "
KB, KA = k("B"), k("A")
KT, KDT, KDP, KHE = k("T"), k("DT"), k("DP"), k("HE")
KT1, KDP1, KDT1, KHE1 = k("T1"), k("DP1"), k("DT1"), k("HE1")
KT2, KDP2, KDT2, KHE2 = k("T2"), k("DP2"), k("DT2"), k("HE2")
KTi, KDPi, KDTi, KHEi = k("Ti"), k("DPi"), k("DTi"), k("HEi")

MODELS = {
    "ABC": dict(
        description="First-order trap-assisted, bimolecular radiative and Auger recombination, "
                    "with no carriers stored in traps.",
        params=[
            ("k_A", 1e-4, RATE, "First-order (trap) recombination", "A"),
            ("k_B", 1e-19, BIMOL, "Bimolecular (radiative)", "B"),
            ("k_C", 1e-37, AUGER, "Auger", "C"),
        ],
        equations=[
            (ddt(DN), rf"= -A {DN} - B {DN}^2 - C {DN}^3"),
            (IPL, rf"\propto B {DN}^2"),
        ],
        schematic=[
            dict(kind="srh", k="k_A"),
            dict(kind="radiative", k="k_B"),
            dict(kind="auger", k="k_C"),
        ],
        run=lambda t, n0, p: gf.TRPL_ABC(t, n0, p["k_A"], p["k_B"], p["k_C"], 0.0)[0],
    ),
    "BTD": dict(
        description="Radiative and Auger recombination plus one trap of finite density, whose "
                    "electrons either de-trap to the CB or recombine with VB holes.",
        params=[
            ("ka", 1e-37, AUGER, "Auger", KA),
            ("kt", 1e-17, BIMOL, "Trapping (capture)", KT),
            ("kb", 1e-19, BIMOL, "Bimolecular (radiative)", KB),
            ("kdt", 1e-3, RATE, "De-trapping (trap → CB)", KDT),
            ("kdp", 1e-19, BIMOL, "Depopulation (trap → VB)", KDP),
            ("NT", 1e15, DENS, "Trap density", NT),
            ("p0", 1e13, DENS, "Background hole density", P0),
        ],
        equations=[
            (ddt(DN), rf"= -{KB} {DN} p - {KA}({DN} p^2 + p {DN}^2)"
                      rf" - {KT} {DN}({NT} - {nT}) + {KDT} {nT}"),
            (ddt(nT), rf"= {KT} {DN}({NT} - {nT}) - {KDT} {nT} - {KDP} {nT} p"),
            (ddt(DP), rf"= -{KB} {DN} p - {KA}({DN} p^2 + p {DN}^2) - {KDP} {nT} p"),
            ("p", rf"= {DP} + {P0},{QQ}{IPL} \propto {DN} p"),
        ],
        schematic=[
            dict(kind="trap", level=DEEP, capture="kt", emit="kdt", hole="kdp", density="NT"),
            dict(kind="radiative", k="kb"),
            dict(kind="auger", k="ka"),
            dict(kind="background", k="p0"),
        ],
        run=lambda t, n0, p: gf.TRPL_BTD(t, p["ka"], p["kt"], p["kb"], p["kdt"], p["kdp"],
                                         p["NT"], p["p0"], n0, 0.0)[0],
    ),
    "DT": dict(
        description="Radiative and Auger recombination plus a shallow trap with first-order "
                    "capture and emission and a first-order deep-trap loss.",
        params=[
            ("k_c", 1e-3, RATE, "Shallow-trap capture", KT1),
            ("k_deep", 1e-4, RATE, "Deep-trap capture", KT2),
            ("k_e", 1e-4, RATE, "Shallow-trap emission", KDT1),
            ("k_rad", 1e-19, BIMOL, "Radiative", KB),
            ("k_aug", 1e-37, AUGER, "Auger", KA),
            ("p0", 1e13, DENS, "Background hole density", P0),
        ],
        equations=[
            (ddt(DN), rf"= -{KB} {DN} {DP} - \dfrac{{1}}{{2}}{KA}({DN}^2 {DP} + {DP}^2 {DN})"
                      rf" - {KT1} {DN} + {KDT1} {nt(1)} - {KT2} {DN}"),
            (ddt(nt(1)), rf"= {KT1} {DN} - {KDT1} {nt(1)}"),
            (DP, rf"= {DN} + {nt(1)},{QQ}{IPL} \propto {DN}({DN} + {P0})"),
        ],
        schematic=[
            dict(kind="trap", level=SHALLOW, capture="k_c", emit="k_e"),
            dict(kind="trap", level=DEEP, capture="k_deep"),
            dict(kind="radiative", k="k_rad"),
            dict(kind="auger", k="k_aug"),
            dict(kind="background", k="p0"),
        ],
        run=lambda t, n0, p: gf.TRPL_DT(t, n0, p["k_c"], p["k_deep"], p["k_e"], p["k_rad"],
                                        p["k_aug"], p["p0"], 0.0)[0],
    ),
    "DTShallowVar": dict(
        description="Radiative recombination plus a shallow trap of finite density with "
                    "capture and emission and a first-order deep-trap loss.",
        params=[
            ("krad", 1e-19, BIMOL, "Radiative", KB),
            ("beta_n_t1", 1e-17, BIMOL, "Shallow-trap capture", KT1),
            ("e_n_t1", 1e-3, RATE, "Shallow-trap emission", KDT1),
            ("N_t1", 1e15, DENS, "Shallow-trap density", Nt(1)),
            ("beta_n_t2", 1e-4, RATE, "Deep-trap capture", KT2),
        ],
        equations=[
            (ddt(DN), rf"= -{KB} {DN} {DP} - {KT1} {DN}({Nt(1)} - {nt(1)})"
                      rf" + {KDT1} {nt(1)} - {KT2} {DN}"),
            (ddt(nt(1)), rf"= {KT1} {DN}({Nt(1)} - {nt(1)}) - {KDT1} {nt(1)}"),
            (DP, rf"= {DN} + {nt(1)},{QQ}{IPL} \propto {DN} {DP}"),
        ],
        schematic=[
            dict(kind="trap", level=SHALLOW, capture="beta_n_t1", emit="e_n_t1", density="N_t1"),
            dict(kind="trap", level=DEEP, capture="beta_n_t2"),
            dict(kind="radiative", k="krad"),
        ],
        run=lambda t, n0, p: gf.TRPL_DTShallowVar(t, p["krad"], p["beta_n_t1"], p["e_n_t1"],
                                                  p["N_t1"], p["beta_n_t2"], n0, 0.0)[0],
    ),
    "DTDeepVar": dict(
        description="Radiative recombination plus a first-order shallow trap with emission and "
                    "a deep trap of finite density whose electrons recombine with VB holes.",
        params=[
            ("krad", 1e-19, BIMOL, "Radiative", KB),
            ("beta_n_t1", 1e-3, RATE, "Shallow-trap capture", KT1),
            ("e_n_t1", 1e-3, RATE, "Shallow-trap emission", KDT1),
            ("beta_n_t2", 1e-18, BIMOL, "Deep-trap electron capture", KT2),
            ("beta_p_t2", 1e-19, BIMOL, "Deep-trap hole capture", KDP2),
            ("N_t2", 1e15, DENS, "Deep-trap density", Nt(2)),
        ],
        equations=[
            (ddt(DN), rf"= -{KB} {DN} {DP} - {KT1} {DN} + {KDT1} {nt(1)}"
                      rf" - {KT2} {DN}({Nt(2)} - {nt(2)})"),
            (ddt(nt(1)), rf"= {KT1} {DN} - {KDT1} {nt(1)}"),
            (ddt(nt(2)), rf"= {KT2} {DN}({Nt(2)} - {nt(2)}) - {KDP2} {DP} {nt(2)}"),
            (ddt(DP), rf"= -{KB} {DN} {DP} - {KDP2} {DP} {nt(2)},{QQ}{IPL} \propto {DN} {DP}"),
        ],
        schematic=[
            dict(kind="trap", level=SHALLOW, capture="beta_n_t1", emit="e_n_t1"),
            dict(kind="trap", level=DEEP, capture="beta_n_t2", hole="beta_p_t2", density="N_t2"),
            dict(kind="radiative", k="krad"),
        ],
        run=lambda t, n0, p: gf.TRPL_DTDeepVar(t, p["krad"], p["beta_n_t1"], p["e_n_t1"],
                                               p["beta_n_t2"], p["beta_p_t2"], p["N_t2"],
                                               n0, 0.0)[0],
    ),
    "ShallowTrapVar": dict(
        description="Radiative and Auger recombination plus one trap of finite density, "
                    "whose electrons can only return to the CB.",
        params=[
            ("k_c", 1e-17, BIMOL, "Capture", KT),
            ("k_e", 1e-3, RATE, "Emission", KDT),
            ("k_rad", 1e-19, BIMOL, "Radiative", KB),
            ("k_aug", 1e-37, AUGER, "Auger", KA),
            ("NT", 1e15, DENS, "Trap density", NT),
            ("p0", 1e13, DENS, "Background hole density", P0),
        ],
        equations=[
            (ddt(DN), rf"= -{KB} {DN} {DP} - {KA} {DN}^2 {DP} - {KT} {DN}({NT} - {nT})"
                      rf" + {KDT} {nT}"),
            (ddt(nT), rf"= {KT} {DN}({NT} - {nT}) - {KDT} {nT}"),
            (DP, rf"= {DN} + {nT},{QQ}{IPL} \propto {DN}({DP} + {P0})"),
        ],
        schematic=[
            dict(kind="trap", level=SHALLOW, capture="k_c", emit="k_e", density="NT"),
            dict(kind="radiative", k="k_rad"),
            dict(kind="auger", k="k_aug"),
            dict(kind="background", k="p0"),
        ],
        run=lambda t, n0, p: gf.TRPL_ShallowTrapVar(t, n0, p["k_c"], p["k_e"], p["k_rad"],
                                                    p["k_aug"], p["NT"], p["p0"], 0.0)[0],
    ),
    "Full Model - 1 Trap": dict(
        description="Radiative recombination with thermal generation plus one trap of finite "
                    "density with electron and hole capture and emission.",
        params=[
            ("krad", 1e-19, BIMOL, "Radiative", KB),
            ("ni", 1e5, DENS, "Intrinsic carrier density", Ni),
            ("beta_n_t1", 1e-17, BIMOL, "Electron capture", KT),
            ("e_n_t1", 1e-3, RATE, "Electron emission", KDT),
            ("beta_p_t1", 1e-19, BIMOL, "Hole capture", KDP),
            ("e_p_t1", 1e-23, RATE, "Hole emission", KHE),
            ("N_t1", 1e15, DENS, "Trap density", NT),
        ],
        equations=[
            (ddt(DN), rf"= -{KB}({DN} {DP} - {Ni}^2) - {KT} {DN}({NT} - {nT}) + {KDT} {nT}"),
            (ddt(DP), rf"= -{KB}({DN} {DP} - {Ni}^2) - {KDP} {DP} {nT} + {KHE}({NT} - {nT})"),
            (ddt(nT), rf"= {KT} {DN}({NT} - {nT}) - {KDP} {DP} {nT}"
                      rf" - {KDT} {nT} + {KHE}({NT} - {nT})"),
            (IPL, rf"\propto {DN} {DP}"),
        ],
        schematic=[
            dict(kind="trap", level=DEEP, capture="beta_n_t1", emit="e_n_t1",
                 hole="beta_p_t1", hole_emit="e_p_t1", density="N_t1"),
            dict(kind="radiative", k="krad"),
        ],
        # The second trap of the full solver is switched off by giving it zero density
        run=lambda t, n0, p: TRPL_FullREM(t, n0, p["krad"], p["ni"], p["beta_n_t1"], p["beta_p_t1"],
                                          p["e_n_t1"], p["e_p_t1"], p["N_t1"],
                                          0.0, 0.0, 0.0, 0.0, 0.0)[0],
    ),
    "Full Model - 2 Traps": dict(
        description="Radiative recombination with thermal generation plus two traps of finite "
                    "density, each with electron and hole capture and emission.",
        params=[
            ("krad", 1e-19, BIMOL, "Radiative", KB),
            ("ni", 1e5, DENS, "Intrinsic carrier density", Ni),
            ("beta_n_t1", 1e-17, BIMOL, "Trap 1 electron capture", KT1),
            ("e_n_t1", 1e-3, RATE, "Trap 1 electron emission", KDT1),
            ("beta_p_t1", 1e-19, BIMOL, "Trap 1 hole capture", KDP1),
            ("e_p_t1", 1e-23, RATE, "Trap 1 hole emission", KHE1),
            ("N_t1", 1e15, DENS, "Trap 1 density", Nt(1)),
            ("beta_n_t2", 1e-18, BIMOL, "Trap 2 electron capture", KT2),
            ("e_n_t2", 1e-13, RATE, "Trap 2 electron emission", KDT2),
            ("beta_p_t2", 1e-19, BIMOL, "Trap 2 hole capture", KDP2),
            ("e_p_t2", 1e-14, RATE, "Trap 2 hole emission", KHE2),
            ("N_t2", 1e15, DENS, "Trap 2 density", Nt(2)),
        ],
        equations=[
            (ddt(DN), rf"= -{KB}({DN} {DP} - {Ni}^2) - \sum_{{i=1,2}}"
                      rf"\left[{KTi} {DN}({Nt('i')} - {nt('i')}) - {KDTi} {nt('i')}\right]"),
            (ddt(DP), rf"= -{KB}({DN} {DP} - {Ni}^2) - \sum_{{i=1,2}}"
                      rf"\left[{KDPi} {DP} {nt('i')} - {KHEi}({Nt('i')} - {nt('i')})\right]"),
            (ddt(nt("i")), rf"= {KTi} {DN}({Nt('i')} - {nt('i')}) - {KDPi} {DP} {nt('i')}"
                           rf" - {KDTi} {nt('i')} + {KHEi}({Nt('i')} - {nt('i')})"),
            (IPL, rf"\propto {DN} {DP}"),
        ],
        schematic=[
            dict(kind="trap", level=SHALLOW, capture="beta_n_t1", emit="e_n_t1",
                 hole="beta_p_t1", hole_emit="e_p_t1", density="N_t1"),
            dict(kind="trap", level=DEEP, capture="beta_n_t2", emit="e_n_t2",
                 hole="beta_p_t2", hole_emit="e_p_t2", density="N_t2"),
            dict(kind="radiative", k="krad"),
        ],
        run=lambda t, n0, p: TRPL_FullREM(t, n0, p["krad"], p["ni"], p["beta_n_t1"], p["beta_p_t1"],
                                          p["e_n_t1"], p["e_p_t1"], p["N_t1"], p["beta_n_t2"],
                                          p["beta_p_t2"], p["e_n_t2"], p["e_p_t2"], p["N_t2"])[0],
    ),
}

# Symbol key entries beyond the parameters: (mathtext symbol, meaning, unit)
TRAPPED = "Trapped electron density"
MODELS["ABC"]["variables"] = []
MODELS["BTD"]["variables"] = [(nT, TRAPPED, DENS), ("p", "Total hole density", DENS)]
MODELS["DT"]["variables"] = [(nt(1), f"{TRAPPED}, shallow trap", DENS)]
MODELS["DTShallowVar"]["variables"] = [(nt(1), f"{TRAPPED}, shallow trap", DENS)]
MODELS["DTDeepVar"]["variables"] = [(nt(1), f"{TRAPPED}, shallow trap", DENS),
                                    (nt(2), f"{TRAPPED}, deep trap", DENS)]
MODELS["ShallowTrapVar"]["variables"] = [(nT, TRAPPED, DENS)]
MODELS["Full Model - 1 Trap"]["variables"] = [(nT, TRAPPED, DENS)]
MODELS["Full Model - 2 Traps"]["variables"] = [(nt("i"), f"{TRAPPED}, trap i", DENS)]
COMMON_VARIABLES = [(DN, "Excess electron density", DENS), (DP, "Excess hole density", DENS),
                    (IPL, "Photoluminescence intensity", "a.u.")]
ANALYSIS_SYMBOLS = [(r"\Delta n_0", "Initial carrier density", DENS),
                    (r"n_{\mathrm{PL}}", "PL-derived carrier density", DENS),
                    (r"\tau_{\mathrm{diff}}", "Differential lifetime", "s"),
                    (r"k_{\mathrm{diff}}", "Differential recombination coefficient", "cm³ s⁻¹")]
SUBSCRIPT_KEY = ("T trapping,  DT de-trapping,  DP depopulation,  HE hole emission\n"
                 "B bimolecular,  A Auger,  1 and 2 shallow and deep trap")

# Definitions from paper_kdiff_BTD.pdf (Eq. S1), shown beside the rate equations
DEFINITIONS = [
    (r"\tau_{\mathrm{diff}}",
     rf"= \left(-\dfrac{{1}}{{2}}\dfrac{{\mathrm{{d}}\ln {IPL}}}{{\mathrm{{d}}t}}\right)^{{-1}}"),
    (r"k_{\mathrm{diff}}",
     rf"= -\dfrac{{1}}{{2 n_{{\mathrm{{PL}}}}}}\dfrac{{\mathrm{{d}}\ln {IPL}}}{{\mathrm{{d}}t}}"),
    (r"n_{\mathrm{PL}}",
     rf"= \Delta n_0 \sqrt{{{IPL}(t)\,/\,{IPL}(0)}}"),
]


# =============================================================================
# Simulation and differential transformations
# =============================================================================

def time_grid(t_max, n_points):
    """Log-spaced times from 0 to t_max (ns), t_i = 10^x_i - 1 as in the paper."""
    return np.logspace(0, np.log10(t_max + 1), n_points) - 1


def simulate(model, params, n0s, t, bkg):
    """
    Simulate every initial density and apply the differential transformations.

    Returns a list of dicts with t, PL, n_PL, tau_diff (s), k_diff (cm^3 s^-1).
    """
    run = MODELS[model]["run"]
    t_jax = jnp.asarray(t)
    results = []
    for n0 in n0s:
        with np.errstate(invalid="ignore"):
            pl = 10 ** np.asarray(run(t_jax, float(n0), params))
        # Once the PL has decayed by many decades the solver can push n
        # slightly negative, giving log10 of a negative number. Truncate the
        # curve at the first such point.
        invalid = ~(np.isfinite(pl) & (pl > 0))
        ok = not invalid[0]
        if invalid.any():
            pl[invalid.argmax():] = np.nan
        pl = pl + bkg
        pl = pl / pl[0]

        dlnI_dt = np.gradient(np.log(pl), t * 1e-9)          # s^-1
        n_pl = n0 * np.sqrt(pl)                              # PL-derived n (cm^-3)
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_diff = 1.0 / (-0.5 * dlnI_dt)
            k_diff = -dlnI_dt / (2 * n_pl)
        results.append(dict(n0=n0, t=t, pl=pl, n_pl=n_pl,
                            tau_diff=tau_diff, k_diff=k_diff, ok=ok))
    return results


def sci_label(x):
    """Format 1.5e16 as 1.5×10¹⁶ in mathtext, in the same font as the tick labels."""
    exp = int(np.floor(np.log10(x)))
    mant = x / 10**exp
    if np.isclose(mant, 1):
        return rf"$\mathdefault{{10^{{{exp}}}}}$"
    return rf"$\mathdefault{{{mant:.3g}\times10^{{{exp}}}}}$"


def converted_text(text, factor, unit):
    """'(value × factor unit)' for an input's text, or '' if it is not a number."""
    try:
        value = float(text) * factor
    except ValueError:
        return ""
    # Round away float noise and tidy the exponent: 1e-08 -> 1e-8, 1e+06 -> 1e6
    number = re.sub(r"e\+?(-?)0*", r"e\1", f"{value:.4g}")
    return f"({number} {unit})"


def export_array(results):
    """
    Decays in the HMC repository's data format, shape (1 + n_decays, n_time).

    Row 0 is the time axis (ns), then one normalised decay per row from the highest
    initial carrier density to the lowest.
    """
    decays = sorted(results, key=lambda r: r["n0"], reverse=True)
    return np.vstack([results[0]["t"]] + [r["pl"] for r in decays])


def fader(i, n):
    """Red→blue colour sequence matching the simulation notebooks."""
    return gf.colorFader("red", "blue", factor=i / max(n - 1, 1))


def dot_indices(n_points, n_dots):
    """Indices of ``n_dots`` samples evenly spaced along the (log-spaced) time grid."""
    if n_dots >= n_points:
        return np.arange(n_points)
    return np.unique(np.linspace(0, n_points - 1, n_dots).round().astype(int))


# =============================================================================
# Appearance
# =============================================================================

BG = "#0b2a5b"
CARD = "#e8f2fb"
FIELD = "#ffffff"
BORDER = "#a9c1da"
INK = "#14233a"
MUTED = "#4d6078"
GREY = "#7a7a7a"
GRID = "#e6ebf2"
ACCENT = "#1f6fd1"
ACCENT_ACTIVE = "#185bb0"
DISABLED = "#c9d6e5"
ON_BG = "#d7e6f7"
DEFAULT_DOTS = 70
CARD_RADIUS, CARD_INSET = 10, 3     # the inset keeps a panel's contents clear of its corners
UI_FONT = ("Segoe UI" if "Segoe UI" in {f.name for f in font_manager.fontManager.ttflist}
           else "DejaVu Sans")

matplotlib.rcParams.update({
    "font.family": UI_FONT,
    "font.size": 9.5,
    "mathtext.fontset": "cm",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlepad": 10,
    "axes.edgecolor": "#8b93a1",
    "axes.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "legend.frameon": False,
    "figure.facecolor": CARD,
    "axes.facecolor": FIELD,
})


def rounded_image(width, height, radius, fill, border, corner):
    """An anti-aliased rounded box, with ``corner`` showing outside its rounded corners."""
    width, height = max(width, 2 * radius + 2), max(height, 2 * radius + 2)
    # The corners are drawn oversized and scaled down; only they need anti-aliasing
    scale, side = 4, radius + 1
    tile = Image.new("RGB", (2 * side * scale, 2 * side * scale), corner)
    ImageDraw.Draw(tile).rounded_rectangle((0, 0, 2 * side * scale - 1, 2 * side * scale - 1),
                                           radius * scale, fill=fill, outline=border, width=scale)
    tile = tile.resize((2 * side, 2 * side), Image.LANCZOS)

    image = Image.new("RGB", (width, height), fill)
    ImageDraw.Draw(image).rectangle((0, 0, width - 1, height - 1), fill=fill, outline=border)
    image.paste(tile.crop((0, 0, side, side)), (0, 0))
    image.paste(tile.crop((side, 0, 2 * side, side)), (width - side, 0))
    image.paste(tile.crop((0, side, side, 2 * side)), (0, height - side))
    image.paste(tile.crop((side, side, 2 * side, 2 * side)), (width - side, height - side))
    return image


def render_math(rows, fontsize, dpi, color=INK, bg=CARD):
    """
    Typeset (left, right) mathtext rows, aligned between the columns, as a PNG.

    Returns (base64 PNG, width in pixels).
    """
    fig = Figure(dpi=dpi)
    renderer = FigureCanvasAgg(fig).get_renderer()

    def text(s):
        return fig.text(0, 0, f"${s}$" if s else "", transform=IdentityTransform(),
                        fontsize=fontsize, color=color, va="baseline")

    px = fontsize * dpi / 72
    gap = 0.35 * px if any(left for left, _ in rows) else 0
    row_gap = 0.7 * px
    placed = []
    for left, right in rows:
        pair = text(left), text(right)
        placed.append((pair, [t.get_window_extent(renderer) for t in pair]))
    col = max(boxes[0].width for _, boxes in placed)
    width = col + gap + max(boxes[1].width for _, boxes in placed)

    y = 0.0
    for i, ((left, right), (lbox, rbox)) in enumerate(placed):
        ascent = max(lbox.y1, rbox.y1)
        descent = max(-lbox.y0, -rbox.y0)
        y -= ascent + (row_gap if i else 0)
        left.set_position((col - lbox.width, y))
        right.set_position((col + gap, y))
        y -= descent
    pad = 2
    W, H = math.ceil(width) + 2 * pad, math.ceil(-y) + 2 * pad
    for pair, _ in placed:
        for t in pair:
            x, yy = t.get_position()
            t.set_position((x + pad, yy + H - pad))
    fig.set_size_inches(W / dpi, H / dpi)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=bg)
    return base64.b64encode(buf.getvalue()), W


def tick_box(checked, size, dpi):
    """A rounded tick box as a base64 PNG."""
    fig = Figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor=ACCENT if checked else FIELD,
                                edgecolor=ACCENT if checked else "#8b93a1", lw=1))
    if checked:
        ax.plot([0.27, 0.44, 0.74], [0.5, 0.31, 0.7], color="white", lw=1.8,
                solid_capstyle="round", solid_joinstyle="round")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=CARD)
    return base64.b64encode(buf.getvalue())


def info_icon(size, dpi):
    """A round 'i' button as a base64 PNG."""
    fig = Figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Circle((0.5, 0.5), 0.47, facecolor=ACCENT, edgecolor="none"))
    ax.text(0.5, 0.48, "i", ha="center", va="center", color="white", fontweight="bold",
            fontsize=0.62 * size * 72 / dpi)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=CARD)
    return base64.b64encode(buf.getvalue())


# Harmless while the window is first being laid out at a tiny size
warnings.filterwarnings("ignore", message="constrained_layout not applied")


def enable_hidpi():
    """Stop Windows bitmap-scaling the window, which blurs the plots and equations."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (ImportError, AttributeError, OSError):
        pass


# =============================================================================
# Band-diagram schematic
# =============================================================================
# Each model's ``schematic`` is a list of components, drawn left to right in the
# order given and only if active (ticked with a non-zero value):
#   srh         first-order trap recombination: electron and hole meet at a level
#   trap        level with optional capture, emit, hole, hole_emit arrows; the whole
#               trap goes if its density is off or none of its arrows are active
#   radiative   band-to-band recombination emitting a photon
#   auger       electron-electron-hole recombination
#   background  background holes in the VB

VB_TOP, CB_BOTTOM, BAND = 1.15, 2.75, 1.15     # band edges and band thickness
ELECTRON_Y, HOLE_Y, RADIUS = CB_BOTTOM + 0.33, VB_TOP - 0.33, 0.27
COMPONENT_WIDTH = dict(srh=1.5, radiative=2.1, auger=2.1, background=1.0)
TRAP_WIDTH = (2.3, 1.85)   # width of a trap with and without labels on its right

CB_CMAP = LinearSegmentedColormap.from_list("cb", ["#a9cbea", "#f1f7fd"])
VB_CMAP = LinearSegmentedColormap.from_list("vb", ["#001f5c", "#1a5a9c", "#e1e9f1"])


def component_visible(comp, active):
    if comp["kind"] == "trap":
        arrows = ("capture", "emit", "hole", "hole_emit")
        return (comp.get("density", True) is True or comp["density"] in active) and any(
            comp.get(a) in active for a in arrows)
    return comp["k"] in active


def component_width(comp, active):
    if comp["kind"] != "trap":
        return COMPONENT_WIDTH[comp["kind"]]
    has_right_side = comp.get("energy") or any(
        comp.get(a) in active for a in ("emit", "density", "hole_emit"))
    return TRAP_WIDTH[0] if has_right_side else TRAP_WIDTH[1]


def draw_schematic(ax, components, active, symbols):
    """Draw the band diagram for the active components onto ``ax``."""
    visible = [c for c in components if component_visible(c, active)]
    width = max(sum(component_width(c, active) for c in visible), 3.2)
    top = CB_BOTTOM + BAND
    x_limits, y_limits = (-0.65, width + 0.12), (-0.12, top + 0.12)
    # Text and line sizes follow the drawing, which shrinks to fit more components
    fig_w, fig_h = ax.figure.get_size_inches() * 72
    unit = min(fig_w / (x_limits[1] - x_limits[0]), fig_h / (y_limits[1] - y_limits[0]))
    weight = unit / 36

    def arrow(p, q, rad=0.0, color="black", lw=1.8):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=0.3 * unit,
                                     lw=lw * weight, color=color, shrinkA=0, shrinkB=0,
                                     zorder=4, connectionstyle=f"arc3,rad={rad}"))

    def label(x, y, name, ha="left", color="black"):
        ax.text(x, y, f"${symbols[name]}$", ha=ha, va="center", fontsize=0.32 * unit,
                color=color, zorder=6)

    def carrier(x, y, sign):
        fill = "#1f5f9e" if sign == "−" else "white"
        ax.add_patch(Circle((x, y), RADIUS, facecolor=fill, edgecolor="black",
                            lw=1.3 * weight, zorder=3))
        ax.text(x, y, sign, ha="center", va="center_baseline", fontsize=0.3 * unit,
                color="black", zorder=4)

    def level(cx, y):
        ax.plot([cx - 0.34, cx + 0.34], [y, y], color="black", lw=1.6 * weight,
                zorder=4, solid_capstyle="butt")

    def photon(x, y):
        xs = np.linspace(x, x + 0.55, 80)
        ax.plot(xs, y + 0.055 * np.sin(2 * np.pi * (xs - x) / 0.2), color="#ee4d2e",
                lw=2.6 * weight, zorder=4)
        ax.add_patch(Polygon([(x + 0.55, y - 0.15), (x + 0.8, y), (x + 0.55, y + 0.15)],
                             facecolor="#ee4d2e", edgecolor="none", zorder=4))

    def burst(x, y):
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        radii = np.where(np.arange(24) % 2, 0.1, 0.2)
        ax.add_patch(Polygon(np.c_[x + radii * np.cos(angles), y + radii * np.sin(angles)],
                             facecolor="#fff200", edgecolor="black", lw=weight, zorder=5))

    edge_cb, edge_vb = CB_BOTTOM - 0.05, VB_TOP + 0.05
    x = 0.0
    for comp in visible:
        kind = comp["kind"]
        if kind == "srh":
            cx, y = x + 0.95, (VB_TOP + CB_BOTTOM) / 2
            level(cx, y)
            carrier(cx, ELECTRON_Y, "−")
            carrier(cx, HOLE_Y, "+")
            arrow((cx, edge_cb), (cx, y + 0.04))
            arrow((cx, edge_vb), (cx, y - 0.04))
            label(cx - 0.42, y, comp["k"], ha="right")
        elif kind == "trap":
            cx, y = x + 1.3, comp["level"]
            level(cx, y)
            # A level placed by its "energy" can sit near a band. Its rate labels then stay
            # clear of the band edges and the level, and its density and energy text go in
            # a column of their own to the right, which nothing can run into.
            energy = comp.get("energy")
            has_density = comp.get("density") in active
            clear = 0.4 if energy else 0.0
            top_max, bottom_min = (edge_cb - 0.14, edge_vb + 0.14) if energy else (edge_cb, edge_vb)
            if energy:
                middle = (VB_TOP + CB_BOTTOM) / 2
                if has_density:
                    label(cx + 1.35, middle + 0.15, comp["density"], color=MUTED)
                ax.text(cx + 1.35, middle - 0.15 if has_density else middle, energy, ha="left",
                        va="center", fontsize=0.22 * unit, color=MUTED, zorder=6)
            elif has_density:
                label(cx + 0.42, y, comp["density"], color=MUTED)
            if comp.get("capture") in active:
                ex = cx - 0.6
                carrier(ex, ELECTRON_Y, "−")
                arrow((ex, edge_cb), (cx - 0.28, y + 0.05), rad=0.3)
                label(ex - 0.08, min((edge_cb + y) / 2 - 0.02, top_max), comp["capture"], ha="right")
            if comp.get("emit") in active:
                arrow((cx + 0.28, y + 0.05), (cx, edge_cb), rad=0.3)
                label(max(cx + 0.32, cx + clear), min((edge_cb + y) / 2 + 0.1, top_max), comp["emit"])
            both = comp.get("hole") in active and comp.get("hole_emit") in active
            if comp.get("hole") in active:
                hx = cx - 0.1 if both else cx
                carrier(cx, HOLE_Y, "+")
                arrow((hx, edge_vb), (hx, y - 0.04))
                hole_y = max((edge_vb + y) / 2, bottom_min)
                if both:
                    label(min(hx - 0.1, cx - clear), hole_y, comp["hole"], ha="right")
                else:
                    label(max(hx + 0.1, cx + clear), hole_y, comp["hole"])
            if comp.get("hole_emit") in active:
                carrier(cx, HOLE_Y, "+")
                arrow((cx + 0.1, y - 0.04), (cx + 0.1, edge_vb))
                label(max(cx + 0.2, cx + clear), max((edge_vb + y) / 2, bottom_min),
                      comp["hole_emit"])
        elif kind == "radiative":
            cx, y = x + 0.65, (VB_TOP + CB_BOTTOM) / 2
            carrier(cx, ELECTRON_Y, "−")
            carrier(cx, HOLE_Y, "+")
            arrow((cx, edge_cb), (cx, edge_vb))
            photon(cx + 0.18, y - 0.12)
            label(cx + 0.18, y + 0.3, comp["k"])
        elif kind == "auger":
            cx = x + 0.55
            carrier(cx, ELECTRON_Y, "−")
            carrier(cx + 0.52, ELECTRON_Y, "−")
            burst(cx + 0.26, ELECTRON_Y)
            carrier(cx, HOLE_Y, "+")
            arrow((cx, edge_cb), (cx, edge_vb))
            arrow((cx + 0.52, ELECTRON_Y + RADIUS), (cx + 0.52, CB_BOTTOM + BAND - 0.06))
            label(cx + 0.12, (VB_TOP + CB_BOTTOM) / 2, comp["k"])
        elif kind == "background":
            cx = x + 0.35
            carrier(cx, HOLE_Y, "+")
            label(cx, HOLE_Y - 0.48, comp["k"], ha="center", color="white")
        x += component_width(comp, active)

    if not visible:
        ax.text(width / 2, (VB_TOP + CB_BOTTOM) / 2, "No active processes", ha="center",
                va="center", color=MUTED, fontsize=0.28 * unit)

    gradient = np.linspace(0, 1, 256)[:, None]
    for cmap, y0, y1 in ((CB_CMAP, CB_BOTTOM, top), (VB_CMAP, 0, VB_TOP)):
        ax.imshow(gradient, cmap=cmap, origin="lower", extent=(0, width, y0, y1),
                  aspect="auto", zorder=1)
        ax.add_patch(Rectangle((0, y0), width, y1 - y0, fill=False, edgecolor="black",
                               lw=1.6 * weight, zorder=2))
    ax.text(0.09, top - 0.1, "CB", ha="left", va="top", fontsize=0.38 * unit, color="black",
            zorder=6)
    ax.text(0.09, 0.08, "VB", ha="left", va="bottom", fontsize=0.38 * unit, color="white",
            zorder=6)
    arrow((-0.2, 0), (-0.2, top), color="#7a7a7a", lw=3)
    ax.text(-0.42, top / 2, "Energy", rotation=90, ha="center", va="center",
            fontsize=0.3 * unit, fontweight="bold", color="#7a7a7a")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal")
    ax.set_axis_off()


# =============================================================================
# GUI
# =============================================================================

class TRPLApp(tk.Tk):
    # Overridden by the tick-box builder in TRPL_GUI_Selective.py
    START_MODEL = "BTD"
    PARAM_HINT = "Untick to set to zero and remove from the schematic"

    def __init__(self):
        enable_hidpi()
        super().__init__()
        self.title("TRPL Kinetics Simulator")
        self.configure(background=BG)
        self.dpi = self.winfo_fpixels("1i")
        self._apply_theme()
        self.geometry("1500x820")
        self.pack_propagate(False)      # the window must not resize to fit its contents

        self.model_var = tk.StringVar(value=self.START_MODEL)
        self.n0_var = tk.StringVar(value="1e14, 1e15, 1e16, 1e17, 1e18")
        self.tmax_var = tk.StringVar(value="150000")
        self.npts_var = tk.StringVar(value="1000")
        self.bkg_var = tk.StringVar(value="0")
        self.xscale_var = tk.StringVar(value="log")
        self.dots_var = tk.StringVar(value=str(DEFAULT_DOTS))
        self.status_var = tk.StringVar(value="Ready.")

        # Per-model parameter state, kept when switching between models
        self.param_state = {
            m: {name: (tk.BooleanVar(value=True), tk.StringVar(value=f"{val:g}"))
                for name, val, *_ in spec["params"]}
            for m, spec in MODELS.items()
        }
        # Grey read-only conversions of the inputs, following what is typed
        self.param_si = {
            m: {name: self._converted(self.param_state[m][name][1], 1e9, PER_SECOND[unit])
                for name, _, unit, *_ in spec["params"] if unit in PER_SECOND}
            for m, spec in MODELS.items()
        }
        self.tmax_us = self._converted(self.tmax_var, 1e-3, "µs")
        self.results = None
        self._worker = None
        self._symbols = {}       # rendered parameter symbols
        self._eq_width = None
        self._eq_job = None
        self._schematic_job = None
        self.key_window = None

        self._build_controls()
        self._build_main()
        self._show_params()

        # Ticking a parameter or zeroing its value rebuilds the schematic
        for state in self.param_state.values():
            for on, val in state.values():
                on.trace_add("write", self._queue_schematic)
                val.trace_add("write", self._queue_schematic)

        # Maximise once mapped: done any earlier, the geometry above overrides it
        self.after(10, self._maximise)

    def _converted(self, source, factor, unit):
        """A StringVar holding ``source`` converted to other units, kept up to date."""
        target = tk.StringVar(value=converted_text(source.get(), factor, unit))
        source.trace_add("write", lambda *_: target.set(
            converted_text(source.get(), factor, unit)))
        return target

    def _maximise(self):
        try:
            self.state("zoomed")
        except tk.TclError:
            pass                        # not available on every platform

    # ----------------------------------------------------------------- theme
    def _apply_theme(self):
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(name).configure(family=UI_FONT, size=10)
        self.option_add("*TCombobox*Listbox.font", (UI_FONT, 10))
        self.option_add("*TCombobox*Listbox.background", FIELD)
        self.option_add("*TCombobox*Listbox.foreground", INK)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", "white")

        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=CARD, foreground=INK, bordercolor=BORDER,
                    lightcolor=CARD, darkcolor=CARD, troughcolor=BG,
                    focuscolor=ACCENT, font=(UI_FONT, 10))
        s.configure("Side.TFrame", background=BG)
        s.configure("Status.TLabel", background=BG, foreground=ON_BG, font=(UI_FONT, 9))
        s.configure("Heading.TLabel", font=(UI_FONT, 10, "bold"))
        s.configure("Muted.TLabel", foreground=MUTED, font=(UI_FONT, 9))
        s.configure("Converted.TLabel", foreground=GREY, font=(UI_FONT, 9))
        for widget in ("TEntry", "TCombobox", "TSpinbox"):
            s.configure(widget, fieldbackground=FIELD, padding=2, arrowcolor=MUTED,
                        lightcolor=FIELD, darkcolor=FIELD, background=FIELD)
            s.map(widget,
                  bordercolor=[("focus", ACCENT)],
                  lightcolor=[("focus", ACCENT)],
                  darkcolor=[("focus", ACCENT)],
                  fieldbackground=[("disabled", DISABLED), ("readonly", FIELD)],
                  foreground=[("disabled", MUTED)],
                  selectbackground=[("readonly", FIELD)],
                  selectforeground=[("readonly", INK)])
        size = round(15 * self.dpi / 96)
        self._ticks = [tk.PhotoImage(master=self, data=tick_box(checked, size, self.dpi))
                       for checked in (False, True)]
        s.element_create("Tick.indicator", "image", self._ticks[0],
                         ("selected", self._ticks[1]), padding=(0, 0, 6, 0), sticky="")
        s.layout("TCheckbutton", [("Checkbutton.padding", {"sticky": "nswe", "children": [
            ("Tick.indicator", {"side": "left", "sticky": ""}),
            ("Checkbutton.label", {"side": "left", "sticky": ""})]})])
        s.configure("TCheckbutton", padding=(0, 1))
        s.map("TCheckbutton", background=[("active", CARD)])
        s.configure("TRadiobutton", indicatorcolor=FIELD)
        s.map("TRadiobutton", background=[("active", CARD)],
              indicatorcolor=[("selected", ACCENT), ("pressed", BORDER)])
        self._round_widgets(s)
        s.configure("Vertical.TScrollbar", background=BORDER, arrowcolor=INK, troughcolor=BG,
                    bordercolor=BG, lightcolor=BORDER, darkcolor=BORDER)

    def _round_widgets(self, s):
        """Give the input fields and buttons rounded edges, drawn as stretchable images."""
        radius = round(5 * self.dpi / 96)
        size = 2 * radius + 6
        self._round_images = []

        def image(fill, border, corner):
            photo = ImageTk.PhotoImage(rounded_image(size, size, radius, fill, border, corner),
                                       master=self)
            self._round_images.append(photo)
            return photo

        s.element_create("Rounded.field", "image", image(FIELD, BORDER, CARD),
                         ("disabled", image(DISABLED, BORDER, CARD)),
                         ("focus", image(FIELD, ACCENT, CARD)),
                         border=radius + 2, sticky="nsew", padding=(4, 2))
        field = lambda children: [("Rounded.field", {"sticky": "nsew", "children": children})]
        area = lambda widget: (f"{widget}.padding", {"expand": "1", "sticky": "nsew", "children": [
            (f"{widget}.textarea", {"sticky": "nsew"})]})
        s.layout("TEntry", field([area("Entry")]))
        s.layout("TCombobox", field([("Combobox.downarrow", {"side": "right", "sticky": "ns"}),
                                     area("Combobox")]))
        s.layout("TSpinbox", field([
            ("null", {"side": "right", "sticky": "", "children": [
                ("Spinbox.uparrow", {"side": "top", "sticky": "e"}),
                ("Spinbox.downarrow", {"side": "bottom", "sticky": "e"})]}),
            area("Spinbox")]))
        for widget in ("TEntry", "TCombobox", "TSpinbox"):
            # The rounded image is the border now, so nothing else should draw one
            s.configure(widget, bordercolor=FIELD)
            s.map(widget, bordercolor=[], lightcolor=[], darkcolor=[])

        def button(style, fill, active, corner, **options):
            element = f"Rounded{style.replace('.', '')}.bg"
            s.element_create(element, "image", image(fill, BORDER if fill == CARD else fill, corner),
                             ("disabled", image(DISABLED, DISABLED, corner)),
                             ("active", image(active, BORDER if fill == CARD else active, corner)),
                             border=radius + 2, padding=0, sticky="nsew")
            s.layout(style, [(element, {"sticky": "nsew", "children": [
                ("Button.padding", {"sticky": "nsew", "children": [
                    ("Button.label", {"sticky": ""})]})]})])
            s.configure(style, **options)
            s.map(style, foreground=[("disabled", MUTED)])

        hover = "#d3e4f6"
        button("TButton", CARD, hover, BG, padding=(10, 6), foreground=INK)
        button("Accent.TButton", ACCENT, ACCENT_ACTIVE, BG, padding=(14, 7), foreground="white",
               font=(UI_FONT, 10, "bold"))
        button("Card.TButton", CARD, hover, CARD, padding=(8, 1), foreground=INK)

    def _card(self, parent, title=None, **pack):
        """A rounded bordered panel; returns the frame to put its contents in."""
        outer = tk.Frame(parent, background=BG)
        outer.pack(**{"fill": tk.X, "pady": (0, 6), **pack})
        # The panel is an image behind the contents, redrawn to fit whenever it resizes
        backdrop = tk.Label(outer, background=BG, borderwidth=0)
        backdrop.place(x=0, y=0, relwidth=1, relheight=1)

        def redraw(event):
            photo = ImageTk.PhotoImage(rounded_image(event.width, event.height, CARD_RADIUS, CARD,
                                                     BORDER, BG), master=self)
            backdrop.configure(image=photo)
            backdrop.image = photo

        outer.bind("<Configure>", redraw)
        inner = ttk.Frame(outer, padding=(7, 5))
        inner.pack(fill=tk.BOTH, expand=True, padx=CARD_INSET, pady=CARD_INSET)
        if title:
            ttk.Label(inner, text=title, style="Heading.TLabel").pack(anchor="w", pady=(0, 6))
        return inner

    # ---------------------------------------------------------------- layout
    def _build_controls(self):
        side = ttk.Frame(self, style="Side.TFrame", padding=(10, 10, 4, 10))
        side.pack(side=tk.LEFT, fill=tk.Y)

        # Simulate stays visible at the bottom; the settings scroll if they overflow
        footer = ttk.Frame(side, style="Side.TFrame")
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        btns = ttk.Frame(footer, style="Side.TFrame")
        btns.pack(fill=tk.X)
        self.run_btn = ttk.Button(btns, text="Simulate", style="Accent.TButton",
                                  command=self._run)
        self.run_btn.pack(side=tk.LEFT)
        ttk.Button(btns, text="Reset parameters", command=self._reset_params).pack(
            side=tk.LEFT, padx=6)
        self.export_btn = ttk.Button(btns, text="Export data…", command=self._export)
        self.export_btn.state(["disabled"])
        self.export_btn.pack(side=tk.LEFT)
        self.status_label = ttk.Label(footer, textvariable=self.status_var,
                                      style="Status.TLabel")
        self.status_label.pack(anchor="w", pady=(6, 0))

        holder = ttk.Frame(side, style="Side.TFrame")
        holder.pack(fill=tk.BOTH, expand=True)
        self.side_canvas = tk.Canvas(holder, background=BG, highlightthickness=0,
                                     borderwidth=0)
        self.side_scroll = ttk.Scrollbar(holder, orient=tk.VERTICAL,
                                         command=self.side_canvas.yview)
        self.side_canvas.configure(yscrollcommand=self.side_scroll.set)
        self.side_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        panel = ttk.Frame(self.side_canvas, style="Side.TFrame")
        self.side_canvas.create_window(0, 0, window=panel, anchor="nw")
        panel.bind("<Configure>", self._fit_sidebar)
        self.side_canvas.bind("<Configure>", lambda e: self._update_scrollbar())
        self.bind_all("<MouseWheel>", self._on_wheel)

        self._build_model_card(panel)

        card = self._card(panel, "Initial carrier densities")
        ttk.Entry(card, textvariable=self.n0_var).pack(fill=tk.X)
        ttk.Label(card, text="Δn₀ in cm⁻³, comma separated, any number",
                  style="Muted.TLabel").pack(anchor="w", pady=(4, 0))

        card = self._card(panel, "Processes and parameters")
        ttk.Label(card, text=self.PARAM_HINT, style="Muted.TLabel").pack(anchor="w", pady=(0, 6))
        self.param_frame = ttk.Frame(card)
        self.param_frame.pack(fill=tk.X)

        card = self._card(panel, "Simulation")
        grid = ttk.Frame(card)
        grid.pack(fill=tk.X)
        for row, (label, var) in enumerate([("Max time (ns)", self.tmax_var),
                                            ("Time points", self.npts_var),
                                            ("Background (rel. to I₀)", self.bkg_var)]):
            ttk.Label(grid, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
            ttk.Entry(grid, textvariable=var, width=12).grid(row=row, column=1, sticky="w", pady=2)
        ttk.Label(grid, textvariable=self.tmax_us, style="Converted.TLabel").grid(
            row=0, column=2, sticky="w", padx=(8, 0))

        card = self._card(panel, "Display", pady=0)
        grid = ttk.Frame(card)
        grid.pack(fill=tk.X)
        ttk.Label(grid, text="Dots per curve").grid(row=0, column=0, sticky="w",
                                                    padx=(0, 12), pady=2)
        dots = ttk.Spinbox(grid, textvariable=self.dots_var, from_=10, to=10000, increment=10,
                           width=10, command=self._plot)
        dots.grid(row=0, column=1, columnspan=2, sticky="w", pady=2)
        dots.bind("<FocusOut>", lambda e: self._plot())
        ttk.Label(grid, text="TRPL time axis").grid(row=1, column=0, sticky="w",
                                                    padx=(0, 12), pady=2)
        ttk.Radiobutton(grid, text="log", variable=self.xscale_var, value="log",
                        command=self._plot).grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(grid, text="linear", variable=self.xscale_var, value="linear",
                        command=self._plot).grid(row=1, column=2, sticky="w", padx=(8, 0))

        self.bind("<Return>", lambda e: self._run())

    def _build_model_card(self, panel):
        card = self._card(panel, "Model")
        combo = ttk.Combobox(card, textvariable=self.model_var, state="readonly",
                             values=list(MODELS))
        combo.pack(fill=tk.X)
        combo.bind("<<ComboboxSelected>>", lambda e: self._show_params())

    def _fit_sidebar(self, event):
        """Size the sidebar to its contents."""
        self.side_canvas.configure(width=event.width,
                                   scrollregion=(0, 0, event.width, event.height))
        self.status_label.configure(wraplength=event.width)
        self._update_scrollbar()

    def _update_scrollbar(self):
        """Show the sidebar scrollbar only if its contents overflow."""
        region = self.side_canvas.bbox("all")
        overflow = region and region[3] > self.side_canvas.winfo_height()
        if overflow and not self.side_scroll.winfo_manager():
            self.side_scroll.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 0))
        elif not overflow and self.side_scroll.winfo_manager():
            self.side_scroll.pack_forget()
            self.side_canvas.yview_moveto(0)

    def _on_wheel(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        if (widget is not None and str(widget).startswith(str(self.side_canvas))
                and self.side_scroll.winfo_manager()):
            self.side_canvas.yview_scroll(-int(event.delta / 120), "units")

    def _build_main(self):
        main = ttk.Frame(self, style="Side.TFrame", padding=(6, 10, 10, 10))
        main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.eq_card = self._card(main)
        header = ttk.Frame(self.eq_card)
        header.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        self.model_label = ttk.Label(header, style="Heading.TLabel")
        self.model_label.pack(side=tk.LEFT, anchor="n", padx=(0, 10))
        self.info_image = tk.PhotoImage(
            master=self, data=info_icon(round(22 * self.dpi / 96), self.dpi))
        self.info_btn = ttk.Label(header, image=self.info_image, cursor="hand2")
        self.info_btn.pack(side=tk.RIGHT, anchor="n", padx=(10, 0))
        self.info_btn.bind("<Button-1>", lambda e: self._show_key())
        self.desc_label = ttk.Label(header)
        self.desc_label.pack(side=tk.LEFT, anchor="n")
        self.schematic_size = (round(5 * self.dpi), round(2.4 * self.dpi))
        self.schematic_fig = Figure(figsize=(5, 2.4), dpi=self.dpi)
        self.schematic_ax = self.schematic_fig.add_axes([0, 0, 1, 1])
        self.schematic_canvas = FigureCanvasTkAgg(self.schematic_fig, master=self.eq_card)
        self.schematic_canvas.get_tk_widget().pack(side=tk.LEFT, anchor="n", padx=(0, 20))
        self.eq_image = tk.PhotoImage(master=self)
        self.def_image = tk.PhotoImage(master=self)
        self.eq_label = ttk.Label(self.eq_card, image=self.eq_image)
        self.eq_label.pack(side=tk.LEFT, anchor="center")
        self.def_label = ttk.Label(self.eq_card, image=self.def_image)
        self.eq_card.bind("<Configure>", self._on_eq_resize)

        card = self._card(main, fill=tk.BOTH, expand=True, pady=0)
        self.fig = Figure(figsize=(13, 5), dpi=self.dpi, layout="constrained")
        self.fig.get_layout_engine().set(w_pad=0.08, h_pad=0.06, wspace=0.03)
        self.axes = self.fig.subplots(1, 3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=card)
        toolbar = NavigationToolbar2Tk(self.canvas, card, pack_toolbar=False)
        toolbar.configure(background=CARD)
        for child in toolbar.winfo_children():
            child.configure(background=CARD)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._plot()

    def _symbol(self, tex):
        if tex not in self._symbols:
            png, _ = render_math([("", tex)], 12, self.dpi)
            self._symbols[tex] = tk.PhotoImage(master=self, data=png)
        return self._symbols[tex]

    def _show_params(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        model = self.model_var.get()
        self.model_label.configure(text=model)
        self.desc_label.configure(text=MODELS[model]["description"])
        self._show_equations()
        self._draw_schematic()
        if self.key_window is not None and self.key_window.winfo_exists():
            self._show_key()
        for row, (name, _, unit, desc, tex) in enumerate(MODELS[model]["params"]):
            on, val = self.param_state[model][name]
            entry = ttk.Entry(self.param_frame, textvariable=val, width=9)
            cb = ttk.Checkbutton(self.param_frame, image=self._symbol(tex), compound=tk.LEFT,
                                 variable=on,
                                 command=lambda e=entry, v=on: self._toggle(e, v))
            cb.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=1)
            entry.grid(row=row, column=1, sticky="w", pady=1)
            ttk.Label(self.param_frame, text=unit).grid(row=row, column=2, sticky="w", padx=(8, 6))
            if name in self.param_si[model]:
                # Fixed width so the sidebar does not resize while a value is typed
                ttk.Label(self.param_frame, textvariable=self.param_si[model][name], width=19,
                          style="Converted.TLabel").grid(row=row, column=3, sticky="w",
                                                         padx=(0, 6))
            ttk.Label(self.param_frame, text=desc, style="Muted.TLabel").grid(
                row=row, column=4, sticky="w")
            self._toggle(entry, on)

    def _show_key(self):
        """Open the symbol key for the current model, or refresh it if already open."""
        model = self.model_var.get()
        window = self.key_window
        if window is None or not window.winfo_exists():
            window = self.key_window = tk.Toplevel(self)
            window.configure(background=BG)
            window.transient(self)
            window.resizable(False, False)
            window.bind("<Escape>", lambda e: window.destroy())
            placed = False
        else:
            placed = True
            for child in window.winfo_children():
                child.destroy()
        window.title(f"Symbol key: {model}")

        def section(parent, title, entries):
            card = self._card(parent, title)
            grid = ttk.Frame(card)
            grid.pack(fill=tk.X)
            for row, (tex, meaning, unit) in enumerate(entries):
                ttk.Label(grid, image=self._symbol(tex)).grid(
                    row=row, column=0, sticky="w", padx=(0, 12), pady=2)
                ttk.Label(grid, text=meaning).grid(row=row, column=1, sticky="w", padx=(0, 12))
                ttk.Label(grid, text=unit, style="Muted.TLabel").grid(row=row, column=2, sticky="w")

        body = ttk.Frame(window, style="Side.TFrame", padding=10)
        body.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(body, style="Side.TFrame")
        left.pack(side=tk.LEFT, fill=tk.Y, anchor="n", padx=(0, 8))
        right = ttk.Frame(body, style="Side.TFrame")
        right.pack(side=tk.LEFT, fill=tk.Y, anchor="n")
        section(left, "Parameters",
                [(tex, desc, unit) for _, _, unit, desc, tex in MODELS[model]["params"]])
        section(right, "Variables", MODELS[model]["variables"] + COMMON_VARIABLES)
        section(right, "Differential analysis", ANALYSIS_SYMBOLS)
        if model != "ABC":
            card = self._card(right, "Subscripts")
            ttk.Label(card, text=SUBSCRIPT_KEY, justify=tk.LEFT).pack(anchor="w")

        if not placed:
            window.update_idletasks()
            x = self.info_btn.winfo_rootx() - window.winfo_reqwidth()
            y = self.info_btn.winfo_rooty() + self.info_btn.winfo_height() + 4
            window.geometry(f"+{max(x, 0)}+{y}")

    def _show_equations(self):
        """Typeset the equations, fitting them and the schematic to the panel width."""
        rows = MODELS[self.model_var.get()]["equations"]
        card = self.eq_card.winfo_width()
        base_w, base_h = self.schematic_size
        width = min(base_w, int(card * 0.42))
        widget = self.schematic_canvas.get_tk_widget()
        if width > 0 and width != int(widget.cget("width")):
            height = round(base_h * width / base_w)
            widget.configure(width=width, height=height)
            self.schematic_fig.set_size_inches(width / self.dpi, height / self.dpi)
            self._queue_schematic()
        available = card - width - 60       # panel padding and gaps

        full, smallest = 13, 10
        eq_png, eq_w = render_math(rows, full, self.dpi)
        def_png, def_w = render_math(DEFINITIONS, full, self.dpi, color=MUTED)
        size, show_definitions = full, True
        if available > 0 and eq_w + def_w > available:
            size = 0.98 * full * available / (eq_w + def_w)
            if size < smallest:             # no room for both: the rate equations win
                show_definitions = False
                size = min(full, 0.98 * full * available / eq_w)
            if size != full:
                eq_png, _ = render_math(rows, size, self.dpi)
                def_png, _ = render_math(DEFINITIONS, size, self.dpi, color=MUTED)
        self.eq_image.configure(data=eq_png)
        self.def_image.configure(data=def_png)
        if show_definitions:
            self.def_label.pack(side=tk.RIGHT, anchor="center", padx=(20, 0))
        else:
            self.def_label.pack_forget()

    def _on_eq_resize(self, event):
        if event.width != self._eq_width:
            self._eq_width = event.width
            self.desc_label.configure(wraplength=max(event.width - 220, 200))
            if self._eq_job:
                self.after_cancel(self._eq_job)
            self._eq_job = self.after(120, self._show_equations)

    def _active_params(self, model):
        """Parameters that are ticked and not zero (unparsable text counts as active)."""
        active = set()
        for name, (on, val) in self.param_state[model].items():
            try:
                nonzero = float(val.get()) != 0
            except ValueError:
                nonzero = True
            if on.get() and nonzero:
                active.add(name)
        return active

    def _queue_schematic(self, *_):
        if self._schematic_job is None:
            self._schematic_job = self.after_idle(self._draw_schematic)

    def _draw_schematic(self):
        self._schematic_job = None
        model = self.model_var.get()
        spec = MODELS[model]
        self.schematic_ax.clear()
        draw_schematic(self.schematic_ax, spec["schematic"], self._active_params(model),
                       {name: tex for name, _, _, _, tex in spec["params"]})
        self.schematic_canvas.draw_idle()

    @staticmethod
    def _toggle(entry, on):
        entry.state(["!disabled"] if on.get() else ["disabled"])

    def _reset_params(self):
        model = self.model_var.get()
        for name, val, *_ in MODELS[model]["params"]:
            on, v = self.param_state[model][name]
            on.set(True)
            v.set(f"{val:g}")
        self._show_params()

    # ------------------------------------------------------------ simulation
    def _read_params(self, model):
        params = {}
        for name, *_ in MODELS[model]["params"]:
            on, val = self.param_state[model][name]
            if on.get():
                try:
                    params[name] = float(val.get())
                except ValueError:
                    raise ValueError(f"Parameter {name}: '{val.get()}' is not a number")
            else:
                params[name] = 0.0
        return params

    def _read_inputs(self):
        model = self.model_var.get()
        params = self._read_params(model)

        try:
            n0s = [float(s) for s in re.split(r"[,;\s]+", self.n0_var.get().strip()) if s]
        except ValueError:
            raise ValueError("Initial carrier densities must be numbers, e.g. 1e15, 1e16")
        if not n0s or any(n <= 0 for n in n0s):
            raise ValueError("Enter at least one positive initial carrier density")

        t_max = float(self.tmax_var.get())
        n_pts = int(self.npts_var.get())
        bkg = float(self.bkg_var.get())
        if t_max <= 0 or n_pts < 10:
            raise ValueError("Max time must be > 0 and time points ≥ 10")
        return model, params, sorted(n0s), time_grid(t_max, n_pts), bkg

    def _run(self):
        if self._worker and self._worker.is_alive():
            return
        try:
            model, params, n0s, t, bkg = self._read_inputs()
        except ValueError as err:
            messagebox.showerror("Invalid input", str(err))
            return

        self.run_btn.state(["disabled"])
        self.status_var.set(f"Simulating {model} for {len(n0s)} densities "
                            "(first run of a model compiles with JAX)…")
        self._outcome = None

        def work():
            try:
                self._outcome = ("ok", model, simulate(model, params, n0s, t, bkg))
            except Exception as err:  # report solver errors in the GUI
                self._outcome = ("error", model, err)

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        self.after(100, self._poll)

    def _poll(self):
        if self._worker.is_alive():
            self.after(100, self._poll)
            return
        self.run_btn.state(["!disabled"])
        kind, model, payload = self._outcome
        if kind == "error":
            self.status_var.set("Simulation failed.")
            messagebox.showerror("Simulation failed", repr(payload))
            return
        self.results = (model, payload)
        self.export_btn.state(["!disabled"])
        bad = [r["n0"] for r in payload if not r["ok"]]
        if bad:
            self.status_var.set(
                f"{model}: no valid PL for Δn₀ = {', '.join(f'{b:.1e}' for b in bad)} "
                "(radiative term excluded or solver failed).")
        else:
            self.status_var.set(f"{model}: simulated {len(payload)} densities.")
        self._plot()

    def _export(self):
        """Save the latest simulation on its full time grid."""
        model, results = self.results
        path = filedialog.asksaveasfilename(
            title="Export simulated TRPL", defaultextension=".npy",
            initialfile=f"SimTRPL_{model}.npy",
            filetypes=[("NumPy data", "*.npy"), ("All files", "*.*")])
        if not path:
            return
        data = export_array(results)
        try:
            np.save(path, data)
        except OSError as err:
            messagebox.showerror("Export failed", str(err))
            return
        status = f"Saved {os.path.basename(path)}, shape {data.shape}."
        if np.isnan(data).any():
            status += (" It contains NaNs where a decay fell below solver precision, "
                       "which the HMC scripts reject; reduce the max time to avoid them.")
        self.status_var.set(status)

    # -------------------------------------------------------------- plotting
    def _n_dots(self):
        """Dots per curve from the control; a bad entry goes back to the default."""
        try:
            n = int(float(self.dots_var.get()))
            if n >= 2:
                return n
        except ValueError:
            pass
        self.dots_var.set(str(DEFAULT_DOTS))
        return DEFAULT_DOTS

    def _plot(self):
        ax_pl, ax_tau, ax_k = self.axes
        for ax in self.axes:
            ax.clear()

        dots = dict(linestyle="none", marker="o", markersize=3, markeredgewidth=0)
        # A curve with no valid PL is all NaN, which a log axis cannot autoscale
        results = [r for r in self.results[1] if r["ok"]] if self.results else []
        if results:
            n_dots = self._n_dots()
            for i, r in enumerate(results):
                c = fader(i, len(results))
                idx = dot_indices(len(r["t"]), n_dots)
                t, pl, n_pl = r["t"][idx], r["pl"][idx], r["n_pl"][idx]
                ax_pl.plot(t, pl, color=c, label=sci_label(r["n0"]), **dots)
                for ax, y in ((ax_tau, r["tau_diff"][idx]), (ax_k, r["k_diff"][idx])):
                    mask = np.isfinite(y) & (y > 0) & (n_pl > 0)
                    ax.plot(n_pl[mask], y[mask], color=c, **dots)
            ax_pl.legend(title=r"$\Delta n_0$ (cm⁻³)", loc="lower left", markerscale=1.8,
                         handletextpad=0.1, labelspacing=0.35)
        else:
            message = "No valid PL to plot" if self.results else "Press Simulate to run the model"
            ax_tau.text(0.5, 0.5, message, color=MUTED,
                        ha="center", va="center", transform=ax_tau.transAxes)

        ax_pl.set_xscale(self.xscale_var.get())
        ax_pl.set_yscale("log")
        for ax in (ax_tau, ax_k):
            ax.set_xscale("log")
            ax.set_yscale("log")

        n_label = r"$n_{\mathrm{PL}}$ (cm⁻³)"
        ax_pl.set_xlabel("Time (ns)")
        ax_pl.set_ylabel("Normalised PL (a.u.)")
        ax_pl.set_title("TRPL")
        ax_tau.set_xlabel(n_label)
        ax_tau.set_ylabel(r"$\tau_{\mathrm{diff}}$ (s)")
        ax_tau.set_title("Differential lifetime")
        ax_k.set_xlabel(n_label)
        ax_k.set_ylabel(r"$k_{\mathrm{diff}}$ (cm³ s⁻¹)")
        ax_k.set_title("Differential coefficient")
        for ax in self.axes:
            ax.tick_params(which="both", direction="in")
        self.canvas.draw_idle()


if __name__ == "__main__":
    TRPLApp().mainloop()
