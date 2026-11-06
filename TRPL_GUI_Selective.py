"""
Build a TRPL model from individual processes and any number of traps.

Every process from the models in ``globalfit_functions.py`` can be switched on, and
traps can be added or removed freely. The equations, band-diagram schematic and
symbol key update to show the model built so far. The plots, differential analysis
and export are those of ``TRPL_GUI.py``, which this file needs alongside it.

Subscripts on the traps are EC electron capture, EE electron emission, HC hole
capture and HE hole emission. There are two ways to set the parameters:

Manual tab
    Every rate is set independently. With p = dp + p0 and sums over the traps i,

        dn/dt    = -kB (n p - ni^2) - kA (n p^2 + p n^2) - kNR n
                   + sum_i [ -kECi n Si + kEEi nTi ]
        ddp/dt   = -kB (n p - ni^2) - kA (n p^2 + p n^2) - kNR n
                   + sum_i [ -kHCi p nTi + kHEi Si ]
        dnTi/dt  = kECi n Si - kEEi nTi - kHCi p nTi + kHEi Si
        I_PL     ∝ kB n p

    where Si = NTi - nTi if the trap's density is ticked. Without a density the trap
    cannot saturate: Si = 1, so capture is first order (ns^-1) and hole emission, which
    needs a density of empty traps, is unavailable. The decay starts from n = dp = n0
    with the traps empty.

SRH tab
    The same equations with detailed balance enforced, and n = n0 + dn, p = p0 + dp
    around thermal equilibrium (n0, p0) in place of p = dp + p0. Each trap is set by its depth below the conduction band, capture coefficients and
    density, which fix the emission rates

        kEE = kEC NC exp(-(EC - Et) / kB T)
        kHE = kHC NV exp(-(Et - EV) / kB T)

    and ni^2 = NC NV exp(-Eg / kB T). The decay starts from thermal equilibrium, found
    from charge neutrality with electron-occupied traps negatively charged, plus n0
    of each carrier, and I_PL ∝ kB (n p - ni^2).

    The equations and procedure follow Heumann, Rudolph, Huang, Kirchartz and Dreessen,
    "Rapid Parameter Estimation from Photoluminescence Decays of Halide Perovskite Thin
    Films", arXiv:2609.18438, Eqs. 4-9 and 20-23; see GUI_GUIDE.md for the theory and
    other references. The defaults are from that work: the
    band gap, kB and first trap of its 85:15 film fit, T = 300 K and NC = NV = 2.2e18
    cm^-3 (from Staub et al., Phys. Rev. Applied 6, 044017), which give ni = 4.96e4
    cm^-3. The Auger and p-doping defaults are from Kumar et al., arXiv:2608.14368 (model
    table in its supplement). All are starting points for a triple-cation film, to be
    replaced as needed.

Manual ticks reproduce the fixed models as follows. "Exact" means the same equations
for the same values; the others differ because those models do not share one set of
conventions.

    BTD, DTShallowVar, DTDeepVar, Full Model   exact
    ABC                                        exact with k_A = C / 2
    DT              k_A = k_aug / 2; here p0 acts in the rate equations and
                    I_PL ∝ n p, whereas DT uses I_PL ∝ n (n + p0)
    ShallowTrapVar  here p0 acts in the rate equations and Auger is
                    k_A (n p^2 + p n^2), whereas it uses k_aug n^2 p

Units are cm^-3, ns and eV.

Run with:  python TRPL_GUI_Selective.py   (in an environment with jax and diffrax)
"""

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

import TRPL_GUI as base
from TRPL_GUI import (AUGER, BIMOL, DENS, DN, DP, GREY, IPL, KA, KB, MUTED, PER_SECOND, RATE, Ni,
                      P0, VB_TOP, CB_BOTTOM)

SELECTIVE = "Selective"
KNR = base.k("NR")
K_BOLTZMANN = 8.617333e-5      # eV / K
WARNING = "#b45309"
RATIO_DECADES = 2.5            # decades of CB/VB ratio over which a Manual trap moves most

base.SUBSCRIPT_KEY = ("EC electron capture,  EE electron emission,  HC hole capture,\n"
                      "HE hole emission,  B bimolecular,  A Auger,  NR non-radiative")
base.TRAP_WIDTH = (3.5, 2.1)   # room for the longer trap labels and the depth or ratio text
# How close a trap may be drawn to each band, leaving its arrows visible
CB_MARGIN, VB_MARGIN = 0.15, 0.25


# =============================================================================
# General rate equations
# =============================================================================

def Selective_Model(t, y, args):
    """
    Rate equations for the general model (see the module docstring).

    Parameters
    ----------
    t : float
        Time (unused, required by Diffrax).
    y : jnp.ndarray
        State vector [n, dp, n_t1, n_t2, ...].
    args : tuple of jnp.ndarray
        See ``pack_args``.

    Returns
    -------
    jnp.ndarray
        Time derivative of ``y``.
    """
    (kB, ni, kA, kNR, p0, _), traps = args
    kEC, kEE, kHC, kHE, NT, finite = traps.T

    # Keep densities within a physical range
    n = jnp.clip(y[0], 1e-10, 1e20)
    p = jnp.clip(y[1], 1e-10, 1e20) + p0
    nt = jnp.clip(y[2:], 0.0, jnp.where(finite > 0, NT, 1e20))
    sites = jnp.where(finite > 0, NT - nt, 1.0)

    pair_loss = kB * (n * p - ni**2) + kA * (n * p**2 + p * n**2) + kNR * n
    capture, emission = kEC * n * sites, kEE * nt
    hole_capture, hole_emission = kHC * p * nt, kHE * sites

    dn_dt = -pair_loss - jnp.sum(capture - emission)
    ddp_dt = -pair_loss - jnp.sum(hole_capture - hole_emission)
    dnt_dt = capture - emission - hole_capture + hole_emission
    return jnp.concatenate([jnp.stack([dn_dt, ddp_dt]), dnt_dt])


@jax.jit
def TRPL_Selective(t, y0, args):
    """
    TRPL signal for the general model, normalised to t = 0.

    Each number of traps compiles once; other changes to the model do not recompile.
    The signal is NaN without radiative recombination, as no PL is emitted.

    Returns
    -------
    jnp.ndarray
        log10 signal.
    """
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(Selective_Model),
        diffrax.Kvaerno5(),
        t0=t[0],
        t1=t[-1],
        dt0=t[1] - t[0],
        y0=y0,
        args=args,
        saveat=diffrax.SaveAt(ts=t),
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-8),
        max_steps=100000,
        throw=False,
    )
    kB, p0, equilibrium_product = args[0][0], args[0][4], args[0][5]
    sig = kB * (sol.ys[:, 0] * (sol.ys[:, 1] + p0) - equilibrium_product)
    return jnp.log10(sig / sig[0])


def pack_args(params):
    """The arguments of ``Selective_Model`` from a dict of active parameters."""
    get = lambda name: params.get(name, 0.0)
    shared = [get("k_B"), get("n_i"), get("k_A"), get("k_NR"), get("p0"),
              get("_pl_offset")]
    traps = [[get(f"k_EC{i}") + get(f"k_EC{i}_f"), get(f"k_EE{i}"), get(f"k_HC{i}"),
              get(f"k_HE{i}"), get(f"N_T{i}"), float(f"N_T{i}" in params)]
             for i in range(1, params["_n_traps"] + 1)]
    return jnp.array(shared), jnp.array(traps).reshape(-1, 6)


def run_model(t, n0, params):
    """log10 TRPL from a dict of active parameters and the initial carrier density."""
    n_eq, p_eq, nt_eq = params.get("_equilibrium", (0.0, 0.0, [0.0] * params["_n_traps"]))
    y0 = jnp.array([n0 + n_eq, n0 + p_eq, *nt_eq])
    return TRPL_Selective(t, y0, pack_args(params))


# =============================================================================
# SRH statistics
# =============================================================================

def intrinsic_density(Eg, T, NC, NV):
    """n_i = sqrt(NC NV) exp(-Eg / 2 kB T), in cm^-3."""
    return math.sqrt(NC * NV) * math.exp(-Eg / (2 * K_BOLTZMANN * T))


def emission_rates(depth, k_EC, k_HC, Eg, T, NC, NV):
    """Detailed-balance (electron, hole) emission rates (ns^-1) of a trap ``depth`` below CB."""
    kT = K_BOLTZMANN * T
    return k_EC * NC * math.exp(-depth / kT), k_HC * NV * math.exp(-(Eg - depth) / kT)


def thermal_equilibrium(traps, doping, Eg, T, NC, NV):
    """
    Equilibrium (n, p, [n_t per trap]) from charge neutrality, p = n + sum(n_t) + doping.

    ``traps`` is a list of (depth below CB, density); energies are relative to EV = 0.
    """
    kT = K_BOLTZMANN * T

    def densities(fermi):
        n = NC * math.exp((fermi - Eg) / kT)
        p = NV * math.exp(-fermi / kT)
        trapped = [density / (1 + math.exp((Eg - depth - fermi) / kT))
                   for depth, density in traps]
        return n, p, trapped

    def net_charge(fermi):
        n, p, trapped = densities(fermi)
        return p - n - sum(trapped) - doping

    return densities(brentq(net_charge, -1.0, Eg + 1.0, xtol=1e-10))


# =============================================================================
# Parameters and presets
# =============================================================================
# Each parameter is (name, default value, unit, description, mathtext symbol), as in
# TRPL_GUI. A trap's capture has one Manual row with two values: k_EC (second order)
# is used if the density is ticked and k_EC_f (first order, in ns^-1) if it is not.

RECOMBINATION_PARAMS = [
    ("k_B", 1e-19, BIMOL, "Bimolecular (radiative)", KB),
    ("n_i", 1e5, DENS, "Intrinsic carrier density", Ni),
    ("k_A", 1e-37, AUGER, "Auger", KA),
    ("k_NR", 1e-4, RATE, "First-order non-radiative", KNR),
    ("p0", 1e13, DENS, "Background hole density", P0),
]
SRH_RECOMBINATION_PARAMS = [
    ("k_B", 1e-20, BIMOL, "Bimolecular (radiative)", KB),
    ("k_A", 8e-37, AUGER, "Auger", KA),
    ("k_NR", 1e-4, RATE, "First-order non-radiative", KNR),
    ("p0", 3.27e15, DENS, "Net p-doping", r"N_{\mathrm{A}}"),
]
MATERIAL_PARAMS = [
    ("E_g", 1.625, "eV", "Band gap", r"E_{\mathrm{g}}"),
    ("T", 300, "K", "Temperature", "T"),
    ("N_C", 2.2e18, DENS, "CB density of states", r"N_{\mathrm{C}}"),
    ("N_V", 2.2e18, DENS, "VB density of states", r"N_{\mathrm{V}}"),
]
CALCULATED_NI = "Calculated; tick to set by hand"
MANUAL_NI_WARNING = "Set by hand: breaks detailed balance"


def manual_trap_params(i):
    return [
        (f"k_EC{i}", 1e-17, BIMOL, "Electron capture", base.k(f"EC{i}")),
        (f"k_EC{i}_f", 1e-3, RATE, "Electron capture (first order)", base.k(f"EC{i}")),
        (f"k_EE{i}", 1e-3, RATE, "Electron emission", base.k(f"EE{i}")),
        (f"k_HC{i}", 1e-19, BIMOL, "Hole capture", base.k(f"HC{i}")),
        (f"k_HE{i}", 1e-23, RATE, "Hole emission", base.k(f"HE{i}")),
        (f"N_T{i}", 1e15, DENS, "Trap density", base.Nt(i)),
    ]


def depth_symbol(i):
    return rf"E_{{\mathrm{{C}}}} - E_{{\mathrm{{t{i}}}}}"


def srh_trap_params(i):
    return [
        (f"depth{i}", 0.15, "eV", "Depth below CB", depth_symbol(i)),
        (f"k_EC{i}", 2.7e-19, BIMOL, "Electron capture", base.k(f"EC{i}")),
        (f"k_HC{i}", 2.7e-19, BIMOL, "Hole capture", base.k(f"HC{i}")),
        (f"N_T{i}", 6.6e16, DENS, "Trap density", base.Nt(i)),
    ]


CUSTOM = "Custom"
SRH_LABEL = "SRH model"
# Equivalent to the fixed models (see the module docstring), with the same values as
# their defaults in TRPL_GUI
PRESETS = {
    "ABC": dict(shared={"k_NR": 1e-4, "k_B": 1e-19, "k_A": 5e-38}, traps=[]),
    "BTD": dict(shared={"k_A": 1e-37, "k_B": 1e-19, "p0": 1e13},
                traps=[{"k_EC": 1e-17, "k_EE": 1e-3, "k_HC": 1e-19, "N_T": 1e15}]),
    "DT (approximate)": dict(shared={"k_NR": 1e-4, "k_B": 1e-19, "k_A": 5e-38, "p0": 1e13},
                             traps=[{"k_EC_f": 1e-3, "k_EE": 1e-4}]),
    "DTShallowVar": dict(shared={"k_B": 1e-19, "k_NR": 1e-4},
                         traps=[{"k_EC": 1e-17, "k_EE": 1e-3, "N_T": 1e15}]),
    "DTDeepVar": dict(shared={"k_B": 1e-19},
                      traps=[{"k_EC_f": 1e-3, "k_EE": 1e-3},
                             {"k_EC": 1e-18, "k_HC": 1e-19, "N_T": 1e15}]),
    "ShallowTrapVar (approximate)": dict(shared={"k_B": 1e-19, "k_A": 5e-38, "p0": 1e13},
                                         traps=[{"k_EC": 1e-17, "k_EE": 1e-3, "N_T": 1e15}]),
    "Full Model - 1 Trap": dict(shared={"k_B": 1e-19, "n_i": 1e5},
                                traps=[{"k_EC": 1e-17, "k_EE": 1e-3, "k_HC": 1e-19,
                                        "k_HE": 1e-23, "N_T": 1e15}]),
    "Full Model - 2 Traps": dict(shared={"k_B": 1e-19, "n_i": 1e5},
                                 traps=[{"k_EC": 1e-17, "k_EE": 1e-3, "k_HC": 1e-19,
                                         "k_HE": 1e-23, "N_T": 1e15},
                                        {"k_EC": 1e-18, "k_EE": 1e-13, "k_HC": 1e-19,
                                         "k_HE": 1e-14, "N_T": 1e15}]),
}


def model_signature(shared, traps):
    """Which processes a model has, regardless of the order of its traps."""
    return set(shared), sorted(sorted(trap) for trap in traps if trap)


def matching_preset(active, n_traps):
    """The preset with exactly these processes, else CUSTOM."""
    shared = active & {name for name, *_ in RECOMBINATION_PARAMS}
    kinds = ("k_EC", "k_EC_f", "k_EE", "k_HC", "k_HE", "N_T")
    traps = []
    for i in range(1, n_traps + 1):
        traps.append({kind for kind in kinds
                      if (f"k_EC{i}_f" if kind == "k_EC_f" else f"{kind}{i}") in active})
    signature = model_signature(shared, traps)
    for name, preset in PRESETS.items():
        if model_signature(preset["shared"], preset["traps"]) == signature:
            return name
    return CUSTOM


# =============================================================================
# From ticks to a model
# =============================================================================

def effective_params(shared, traps):
    """
    The parameters that take part in the model, given ticked rows and their values.

    ``shared`` maps ticked recombination rows to values, and ``traps`` holds one such
    dict per trap with keys k_EC, k_EE, k_HC, k_HE, N_T. Zero-valued rows drop out, as
    do those with no effect: n_i without radiative recombination, and everything on a
    trap that nothing can populate. A ticked density of zero removes the whole trap.
    Trap names gain their position, and first-order capture is named k_ECi_f.
    """
    active = {name for name in ("k_B", "k_A", "k_NR", "p0") if shared.get(name, 0.0) != 0}
    if "k_B" in active and shared.get("n_i", 0.0) != 0:
        active.add("n_i")
    for i, trap in enumerate(traps, 1):
        on = lambda kind: trap.get(kind, 0.0) != 0
        finite = "N_T" in trap
        hole_emission = finite and on("k_HE")
        if (finite and not on("N_T")) or not (on("k_EC") or hole_emission):
            continue
        active |= {f"{kind}{i}" for kind in ("k_EE", "k_HC") if on(kind)}
        if on("k_EC"):
            active.add(f"k_EC{i}" if finite else f"k_EC{i}_f")
        if hole_emission:
            active.add(f"k_HE{i}")
        if finite:
            active.add(f"N_T{i}")
    return active


def inert_reason(kind, trap):
    """Why a ticked, non-zero row of a trap has no effect."""
    if "N_T" in trap and trap["N_T"] == 0:
        return "density is zero"
    if kind == "k_HE" and "N_T" not in trap:
        return "needs a density"
    return "needs capture"


def signed_terms(terms):
    """Mathtext for 'a - b + c' from (sign, term) pairs."""
    return " ".join(f"{sign} {term}" if i or sign == "-" else term
                    for i, (sign, term) in enumerate(terms))


def signed_sum(terms):
    return f"= {signed_terms(terms)}" if terms else "= 0"


def has_trap(a, i):
    return any(name in a for name in (f"k_EC{i}", f"k_EC{i}_f", f"k_HE{i}"))


def trap_terms(a, i, label, electron, holes, sites):
    """(electron, hole, occupancy) terms for trap ``i``, written with subscript ``label``."""
    trapped = base.nt(label)
    electrons, hole_terms, filled = [], [], []
    if f"k_EC{i}" in a or f"k_EC{i}_f" in a:
        capture = rf"{base.k(f'EC{label}')} {electron} {sites}"
        electrons.append(("-", capture))
        filled.append(("+", capture))
    if f"k_EE{i}" in a:
        emission = rf"{base.k(f'EE{label}')} {trapped}"
        electrons.append(("+", emission))
        filled.append(("-", emission))
    if f"k_HC{i}" in a:
        depopulation = rf"{base.k(f'HC{label}')} {holes} {trapped}"
        hole_terms.append(("-", depopulation))
        filled.append(("-", depopulation))
    if f"k_HE{i}" in a:
        hole_emission = rf"{base.k(f'HE{label}')} {sites}"
        hole_terms.append(("+", hole_emission))
        filled.append(("+", hole_emission))
    return electrons, hole_terms, filled


def build_equations(a, n_traps, srh=False):
    """Rows of (left, right) mathtext for the active parameters ``a``."""
    traps = [i for i in range(1, n_traps + 1) if has_trap(a, i)]
    # The SRH tab adds the excess carriers to their thermal equilibrium densities
    electron = rf"(n_0 + {DN})" if srh else DN
    holes = rf"(p_0 + {DP})" if srh else "p" if "p0" in a else DP if traps else DN

    def product(p_power=1):
        if holes == electron:
            return rf"{electron}^{{{1 + p_power}}}"
        return rf"{electron} {holes}" + (rf"^{p_power}" if p_power > 1 else "")

    shared = []
    if "k_B" in a:
        shared.append(("-", rf"{KB}({product()} - {Ni}^2)" if "n_i" in a
                       else rf"{KB} {product()}"))
    if "k_A" in a:
        shared.append(("-", rf"2 {KA} {electron}^3" if holes == electron
                       else rf"{KA}({product(2)} + {holes} {electron}^2)"))
    if "k_NR" in a:
        shared.append(("-", rf"{KNR} {electron}"))
    electrons, hole_terms, trap_rows, site_rows = list(shared), list(shared), [], []

    # The SRH form is longer, so it needs the general form sooner to stay readable
    if len(traps) <= (1 if srh else 2):
        for i in traps:
            sites = rf"({base.Nt(i)} - {base.nt(i)})" if f"N_T{i}" in a else ""
            e_terms, h_terms, filled = trap_terms(a, i, i, electron, holes, sites)
            electrons += e_terms
            hole_terms += h_terms
            trap_rows.append((base.ddt(base.nt(i)), signed_sum(filled)))
    else:
        # Too long to write out: one general form, covering what any of the traps has
        names = {f"{kind}i" for kind in ("k_EC", "k_EE", "k_HC", "k_HE")
                 if any(f"{kind}{i}" in a or f"{kind}{i}_f" in a for i in traps)}
        finite = [f"N_T{i}" in a for i in traps]
        sites = rf"({base.Nt('i')} - {base.nt('i')})" if all(finite) else "S_i"
        e_terms, h_terms, filled = trap_terms(names, "i", "i", electron, holes, sites)
        for terms, target in ((e_terms, electrons), (h_terms, hole_terms)):
            if terms:
                target.append(("+", rf"\sum_{{i}}\left[{signed_terms(terms)}\right]"))
        trap_rows.append((base.ddt(base.nt("i")), signed_sum(filled)))
        if not all(finite):
            site_rows.append(("S_i", rf"= {base.Nt('i')} - {base.nt('i')}\ "
                                     r"\mathrm{or}\ 1\ \mathrm{(no\ density)}"))

    rows = [(base.ddt(DN), signed_sum(electrons))]
    if traps or srh:
        rows.append((base.ddt(DP), signed_sum(hole_terms)))
    rows += trap_rows + site_rows
    if not srh:
        excess = DP if traps else DN
        if "p0" in a:
            rows.append(("p", rf"= {excess} + {P0}"))
        elif not traps:
            rows.append((DP, rf"= {DN}"))
    if srh:
        if "p0" in a:
            trapped = rf" + \sum_{{i}} {base.nt('i')}^0" if traps else ""
            rows.append(("p_0", rf"= n_0{trapped} + N_{{\mathrm{{A}}}}"))
        kT = r"k_{\mathrm{B}}T"
        if traps:
            rows.append((base.k("EEi"), rf"= {base.k('ECi')} N_{{\mathrm{{C}}}} \exp\left(-"
                                        rf"\dfrac{{E_{{\mathrm{{C}}}} - E_{{\mathrm{{ti}}}}}}{{{kT}}}\right)"))
            rows.append((base.k("HEi"), rf"= {base.k('HCi')} N_{{\mathrm{{V}}}} \exp\left(-"
                                        rf"\dfrac{{E_{{\mathrm{{ti}}}} - E_{{\mathrm{{V}}}}}}{{{kT}}}\right)"))
        rows.append((IPL, rf"\propto {KB}({product()} - {Ni}^2)" if "k_B" in a else "= 0"))
    else:
        rows.append((IPL, rf"\propto {product()}" if "k_B" in a else "= 0"))
    return rows


def trap_level(depth, Eg):
    """
    Height of a trap in the schematic, to scale across the band gap.

    Only a level almost at a band edge is held back, so that its arrows stay visible;
    its depth label still gives the true value.
    """
    gap = CB_BOTTOM - VB_TOP
    return float(np.clip(CB_BOTTOM - gap * depth / Eg, VB_TOP + VB_MARGIN, CB_BOTTOM - CB_MARGIN))


def coupling_ratio(trap, i, a, n_ref):
    """
    How strongly trap ``i`` exchanges electrons with the CB compared with the VB.

    R = (k_EC n_ref + k_EE) / (k_HC n_ref + k_HE), with first-order capture counted as
    it stands, as it is already in ns^-1. ``trap`` holds its ticked rows and values, and
    ``n_ref`` stands in for both carrier densities. Infinite if it has no VB channel.
    """
    cb, vb = 0.0, 0.0
    if f"k_EC{i}" in a:
        cb += trap["k_EC"] * n_ref
    if f"k_EC{i}_f" in a:
        cb += trap["k_EC"]
    if f"k_EE{i}" in a:
        cb += trap["k_EE"]
    if f"k_HC{i}" in a:
        vb += trap["k_HC"] * n_ref
    if f"k_HE{i}" in a:
        vb += trap["k_HE"]
    return cb / vb if vb else math.inf


def ratio_level(ratio):
    """Height of a Manual trap: towards the CB for R > 1 and the VB for R < 1, staying in the gap."""
    middle = (VB_TOP + CB_BOTTOM) / 2
    if math.isnan(ratio):
        return middle
    if ratio <= 0 or math.isinf(ratio):
        position = -1.0 if ratio <= 0 else 1.0
    else:
        position = math.tanh(math.log10(ratio) / RATIO_DECADES)
    room = (CB_BOTTOM - CB_MARGIN - middle) if position > 0 else (middle - VB_TOP - VB_MARGIN)
    return middle + room * position


def ratio_text(ratio):
    if math.isnan(ratio):
        return ""
    if math.isinf(ratio):
        return "R → ∞"
    if 0.01 <= ratio < 1000:
        return f"R = {float(f'{ratio:.2g}'):g}"
    return "R = " + base.re.sub(r"e\+?(-?)0*", r"e\1", f"{ratio:.1e}")


def build_schematic(a, n_traps, levels=None, energies=None):
    """
    Components of the band diagram for the active parameters ``a``.

    ``levels`` holds each trap's height in the diagram, mid-gap if not given, and
    ``energies`` the small text beside it: its depth, or its ratio on the Manual tab.
    """
    components = []
    if "k_NR" in a:
        components.append(dict(kind="srh", k="k_NR"))
    for i in range(1, n_traps + 1):
        level = levels[i - 1] if levels else (VB_TOP + CB_BOTTOM) / 2
        comp = dict(kind="trap", level=level, emit=f"k_EE{i}", hole=f"k_HC{i}",
                    hole_emit=f"k_HE{i}",
                    capture=f"k_EC{i}_f" if f"k_EC{i}_f" in a else f"k_EC{i}")
        if f"N_T{i}" in a:
            comp["density"] = f"N_T{i}"
        if energies and energies[i - 1]:
            comp["energy"] = energies[i - 1]
        components.append(comp)
    components.append(dict(kind="radiative", k="k_B"))
    components.append(dict(kind="auger", k="k_A"))
    components.append(dict(kind="background", k="p0"))
    return components


def build_variables(a, n_traps, srh=False):
    """Symbol key entries for the variables beyond those of every model."""
    variables = [(base.nt(i), f"{base.TRAPPED}, trap {i}", DENS)
                 for i in range(1, n_traps + 1) if has_trap(a, i)]
    if srh:
        variables += [("n_0", "Equilibrium electron density", DENS),
                      ("p_0", "Equilibrium hole density", DENS)]
    elif "p0" in a:
        variables.append(("p", "Total hole density", DENS))
    return variables


def describe(a, n_traps, srh=False):
    """One sentence listing the processes in the model."""
    parts = [text for name, text in (("k_B", "radiative"), ("n_i", "thermal generation"),
                                     ("k_A", "Auger"), ("k_NR", "first-order loss"),
                                     ("p0", "p-doping" if srh else "background holes"))
             if name in a]
    traps = [i for i in range(1, n_traps + 1) if has_trap(a, i)]
    if len(traps) > 2:
        parts.append(f"{len(traps)} traps")
    else:
        for i in traps:
            steps = [text for names, text in (
                ((f"k_EC{i}", f"k_EC{i}_f"), "capture"), ((f"k_EE{i}",), "emission"),
                ((f"k_HC{i}",), "hole capture"), ((f"k_HE{i}",), "hole emission"))
                if any(name in a for name in names)]
            kind = "finite" if f"N_T{i}" in a else "unsaturable"
            parts.append(f"{kind} trap {i} ({', '.join(steps)})")
    if not parts:
        return "No active processes: tick some on the left."
    sentence = ", ".join(parts)
    return f"{sentence[0].upper()}{sentence[1:]}."


base.MODELS[SELECTIVE] = dict(description="", params=[], equations=[], schematic=[], variables=[],
                              run=run_model)


# =============================================================================
# GUI
# =============================================================================

@dataclass
class Field:
    """One parameter's input: its value, its optional tick and its grey conversion."""
    value: tk.StringVar
    on: tk.BooleanVar = None
    converted: tk.StringVar = None

    def number(self):
        """The value as a float; unparsable text is NaN, which counts as non-zero."""
        try:
            return float(self.value.get())
        except ValueError:
            return math.nan


class SelectiveApp(base.TRPLApp):
    """TRPLApp with one model, whose MODELS entry is rebuilt from the Manual or SRH tab."""

    START_MODEL = SELECTIVE
    PARAM_HINT = ("Manual sets every rate. SRH sets trap energies, and detailed balance\n"
                  "sets the emission rates.")

    # ------------------------------------------------------------ parameter state
    def _field(self, value, unit, ticked=None):
        """A Field whose changes redraw the model."""
        var = tk.StringVar(value=f"{value:g}")
        var.trace_add("write", self._queue_schematic)
        on = None
        if ticked is not None:
            on = tk.BooleanVar(value=ticked)
            on.trace_add("write", self._queue_schematic)
        converted = (self._converted(var, 1e9, PER_SECOND[unit]) if unit in PER_SECOND else None)
        return Field(var, on, converted)

    def _new_manual_trap(self, values=None):
        """Fields of a Manual trap, ticking those in ``values`` and setting their values."""
        values = values or {}
        trap = {}
        for name, default, unit, *_ in manual_trap_params(0):
            kind = name.replace("0", "")
            ticked = kind in values or (kind == "k_EC" and "k_EC_f" in values)
            trap[kind] = self._field(values.get(kind, default), unit,
                                     None if kind == "k_EC_f" else ticked)
        return trap

    def _new_srh_trap(self):
        return {name.rstrip("0"): self._field(default, unit)
                for name, default, unit, *_ in srh_trap_params(0)}

    def _build_model_card(self, panel):
        self.title("TRPL Kinetics Simulator: model builder")
        self.spec = base.MODELS[SELECTIVE]
        self.preset_var = tk.StringVar(value=CUSTOM)
        self.n0_var.trace_add("write", self._queue_schematic)    # sets the Manual trap ratio
        self.manual_shared = {name: self._field(default, unit, name == "k_B")
                              for name, default, unit, *_ in RECOMBINATION_PARAMS}
        self.manual_traps = [self._new_manual_trap()]
        self.srh_shared = {name: self._field(default, unit, name == "k_B")
                           for name, default, unit, *_ in SRH_RECOMBINATION_PARAMS}
        self.material = {name: self._field(default, unit)
                         for name, default, unit, *_ in MATERIAL_PARAMS}
        self.ni_field = self._field(4.96e4, DENS, False)
        self.srh_traps = [self._new_srh_trap()]
        self.notebook = None

        card = self._card(panel, "Model")
        self.preset_combo = ttk.Combobox(card, textvariable=self.preset_var, state="readonly",
                                         values=list(PRESETS))
        self.preset_combo.pack(fill=tk.X)
        self.preset_combo.bind("<<ComboboxSelected>>",
                               lambda e: self._load_preset(self.preset_var.get()))
        ttk.Label(card, text="Load a fixed model into the Manual tab as a starting point",
                  style="Muted.TLabel").pack(anchor="w", pady=(4, 0))

    def _is_srh(self):
        return self.notebook is not None and self.notebook.index("current") == 1

    def _load_preset(self, name):
        preset = PRESETS[name]
        for row, default, *_ in RECOMBINATION_PARAMS:
            field = self.manual_shared[row]
            field.on.set(row in preset["shared"])
            field.value.set(f"{preset['shared'].get(row, default):g}")
        self.manual_traps = [self._new_manual_trap(values) for values in preset["traps"]]
        self.notebook.select(0)
        self._rebuild_traps()

    def _reset_params(self):
        """Back to the starting state of the tab in view."""
        if self._is_srh():
            for row, default, *_ in SRH_RECOMBINATION_PARAMS:
                self.srh_shared[row].on.set(row == "k_B")
                self.srh_shared[row].value.set(f"{default:g}")
            for row, default, *_ in MATERIAL_PARAMS:
                self.material[row].value.set(f"{default:g}")
            self.ni_field.on.set(False)
            self.srh_traps = [self._new_srh_trap()]
        else:
            for row, default, *_ in RECOMBINATION_PARAMS:
                self.manual_shared[row].on.set(row == "k_B")
                self.manual_shared[row].value.set(f"{default:g}")
            self.manual_traps = [self._new_manual_trap()]
        self._rebuild_traps()

    # ---------------------------------------------------------------- the model
    @staticmethod
    def _capture_kind(trap):
        return "k_EC" if trap["N_T"].on.get() else "k_EC_f"

    def _manual_ticks(self):
        """Ticked rows and values of the Manual tab, for ``effective_params``."""
        shared = {name: f.number() for name, f in self.manual_shared.items() if f.on.get()}
        traps = []
        for trap in self.manual_traps:
            ticks = {kind: f.number() for kind, f in trap.items()
                     if f.on is not None and f.on.get()}
            if "k_EC" in ticks:
                ticks["k_EC"] = trap[self._capture_kind(trap)].number()
            traps.append(ticks)
        return shared, traps

    def _material_values(self):
        return tuple(self.material[name].number() for name in ("E_g", "T", "N_C", "N_V"))

    def _srh_ni(self):
        """n_i in use: the one set by hand if ticked, else calculated (NaN if it cannot be)."""
        if self.ni_field.on.get():
            return self.ni_field.number()
        Eg, T, NC, NV = self._material_values()
        try:
            return intrinsic_density(Eg, T, NC, NV)
        except (ValueError, ZeroDivisionError, OverflowError):
            return math.nan

    def _srh_emission(self, trap):
        """Emission rates of an SRH trap, NaN until its inputs can give them."""
        try:
            return emission_rates(trap["depth"].number(), trap["k_EC"].number(),
                                  trap["k_HC"].number(), *self._material_values())
        except (ValueError, ZeroDivisionError, OverflowError):
            return math.nan, math.nan

    def _srh_ticks(self):
        """Rows and values of the SRH tab, for ``effective_params``."""
        shared = {name: f.number() for name, f in self.srh_shared.items() if f.on.get()}
        shared["n_i"] = self._srh_ni()
        traps = []
        for trap in self.srh_traps:
            emission, hole_emission = self._srh_emission(trap)
            traps.append(dict(k_EC=trap["k_EC"].number(), k_EE=emission,
                              k_HC=trap["k_HC"].number(), k_HE=hole_emission,
                              N_T=trap["N_T"].number()))
        return shared, traps

    def _tab_inputs(self):
        return self._srh_ticks() if self._is_srh() else self._manual_ticks()

    def _active_params(self, model):
        return effective_params(*self._tab_inputs())

    def _read_params(self, model):
        if self._is_srh():
            Eg, T, NC, NV = self._material_values()
            if any(math.isnan(x) or x <= 0 for x in (Eg, T, NC, NV)):
                raise ValueError("Band gap, temperature and densities of states must be positive")
        shared, traps = self._tab_inputs()
        active = effective_params(shared, traps)
        flat = dict(shared)
        for i, trap in enumerate(traps, 1):
            for kind, value in trap.items():
                flat[f"{kind}{i}"] = value
                if kind == "k_EC":
                    flat[f"k_EC{i}_f"] = value
        params = {name: flat[name] for name in active}
        for name, value in params.items():
            if math.isnan(value):
                raise ValueError(f"Parameter {name} is not a number")
        params["_n_traps"] = len(traps)
        if self._is_srh():
            self._add_srh_state(params, active, shared.get("p0", 0.0))
        return params

    def _add_srh_state(self, params, active, doping):
        """Check the SRH inputs and add the equilibrium that the decay starts from."""
        Eg, T, NC, NV = self._material_values()
        if math.isnan(doping):
            raise ValueError("Net p-doping is not a number")
        occupied = []
        for i, trap in enumerate(self.srh_traps, 1):
            depth, active_trap = trap["depth"].number(), has_trap(active, i)
            if active_trap and not 0 < depth < Eg:
                raise ValueError(f"Trap {i}: depth must lie within the band gap")
            occupied.append((depth, params[f"N_T{i}"]) if active_trap else None)
        n_eq, p_eq, trapped = thermal_equilibrium([t for t in occupied if t], doping,
                                                  Eg, T, NC, NV)
        # Doping enters only through this equilibrium, as the carriers here are total
        params.pop("p0", None)
        filled = iter(trapped)
        params["_equilibrium"] = (n_eq, p_eq, [next(filled) if t else 0.0 for t in occupied])
        params["_pl_offset"] = params.get("n_i", 0.0) ** 2

    def _spec_params(self, active):
        """Entries for the symbol key and the schematic's labels."""
        n_traps = len(self.srh_traps) if self._is_srh() else len(self.manual_traps)
        if self._is_srh():
            table = list(MATERIAL_PARAMS)
            table.append(("n_i", self._srh_ni(), DENS, "Intrinsic carrier density", Ni))
            table += [p for p in SRH_RECOMBINATION_PARAMS if p[0] in active]
            for i in range(1, n_traps + 1):
                if has_trap(active, i):
                    table.append((f"depth{i}", 0, "eV", "Trap depth", depth_symbol(i)))
                    table += [p for p in manual_trap_params(i) if p[0] in active]
            return [(n, v, u, "Calculated by detailed balance" if n[:4] in ("k_EE", "k_HE") else d, t)
                    for n, v, u, d, t in table]
        table = list(RECOMBINATION_PARAMS)
        for i in range(1, n_traps + 1):
            table += manual_trap_params(i)
        return [p for p in table if p[0] in active]

    # --------------------------------------------------------------------- layout
    def _show_params(self):
        """Build the Manual and SRH tabs; their rows follow the state from then on."""
        style = ttk.Style(self)
        style.configure("TNotebook", background=base.CARD, borderwidth=0)
        style.configure("TNotebook.Tab", background=base.DISABLED, padding=(12, 4))
        style.map("TNotebook.Tab", background=[("selected", base.CARD)])
        style.configure("Warning.TLabel", foreground=WARNING, font=(base.UI_FONT, 9))

        self.notebook = ttk.Notebook(self.param_frame)
        self.notebook.pack(fill=tk.X)
        self.manual_tab = ttk.Frame(self.notebook, padding=(0, 8, 0, 0))
        self.srh_tab = ttk.Frame(self.notebook, padding=(0, 8, 0, 0))
        self.notebook.add(self.manual_tab, text="Manual")
        self.notebook.add(self.srh_tab, text="SRH (detailed balance)")
        self.notebook.bind("<<NotebookTabChanged>>", self._queue_schematic)

        self.rows = dict(manual={}, srh={})
        row = self._section(self.manual_tab, 0, "Recombination")
        for name, _, unit, desc, tex in RECOMBINATION_PARAMS:
            self.rows["manual"][name] = self._row(self.manual_tab, row, tex, self.manual_shared[name],
                                                  unit)
            row += 1
        self.manual_first_trap_row = row

        row = self._section(self.srh_tab, 0, "Material")
        self.material_rows = {}
        for name, _, unit, desc, tex in MATERIAL_PARAMS:
            self.material_rows[name] = self._row(self.srh_tab, row, tex, self.material[name], unit)
            self.material_rows[name]["desc"].configure(text=desc)
            row += 1
        self.ni_row = self._row(self.srh_tab, row, Ni, self.ni_field, DENS)
        row = self._section(self.srh_tab, row + 1, "Recombination")
        for name, _, unit, desc, tex in SRH_RECOMBINATION_PARAMS:
            self.rows["srh"][name] = self._row(self.srh_tab, row, tex, self.srh_shared[name], unit)
            self.rows["srh"][name]["desc"].configure(text=desc)
            row += 1
        self.srh_first_trap_row = row

        self.trap_widgets = dict(manual=[], srh=[])
        self._rebuild_traps()

    def _section(self, frame, row, title):
        ttk.Label(frame, text=title, style="Heading.TLabel").grid(
            row=row, column=0, columnspan=5, sticky="w", pady=(6 if row else 0, 2))
        return row + 1

    def _row(self, frame, row, tex, field, unit, owned=None):
        """One input row; ``owned`` collects the widgets if they will be rebuilt."""
        entry = ttk.Entry(frame, textvariable=field.value, width=9)
        if field.on is not None:
            label = ttk.Checkbutton(frame, image=self._symbol(tex), compound=tk.LEFT,
                                    variable=field.on)
        else:
            label = ttk.Label(frame, image=self._symbol(tex))
        label.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=1)
        entry.grid(row=row, column=1, sticky="w", pady=1)
        unit_label = ttk.Label(frame, text=unit)
        unit_label.grid(row=row, column=2, sticky="w", padx=(8, 6))
        # Fixed width so the sidebar does not resize while a value is typed
        converted = ttk.Label(frame, width=17, style="Converted.TLabel",
                              textvariable=field.converted)
        converted.grid(row=row, column=3, sticky="w", padx=(0, 6))
        desc = ttk.Label(frame, style="Muted.TLabel")
        desc.grid(row=row, column=4, sticky="w")
        if owned is not None:
            owned += [label, entry, unit_label, converted, desc]
        return dict(entry=entry, unit=unit_label, converted=converted, desc=desc)

    def _trap_header(self, frame, row, i, traps, owned):
        title = ttk.Label(frame, text=f"Trap {i}", style="Heading.TLabel")
        title.grid(row=row, column=0, columnspan=3, sticky="w", pady=(6, 2))
        remove = ttk.Button(frame, text="Remove trap", style="Card.TButton",
                            command=lambda: self.after_idle(self._remove_trap, traps, i - 1))
        remove.grid(row=row, column=4, sticky="e", pady=(6, 2))
        owned += [title, remove]
        return row + 1

    def _trap_footer(self, frame, row, add, owned):
        button = ttk.Button(frame, text="Add Trap", style="Card.TButton", command=add)
        button.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 2))
        warning = ttk.Label(frame, style="Warning.TLabel", wraplength=470, justify=tk.LEFT,
                            text="⚠ Each trap adds a rate equation. Too many traps can cause "
                                 "numerical instability and make the simulation crash.")
        warning.grid(row=row + 1, column=0, columnspan=5, sticky="w")
        owned += [button, warning]

    def _rebuild_traps(self):
        """Lay out both tabs' traps again, after one was added or removed."""
        for widgets in self.trap_widgets.values():
            for widget in widgets:
                widget.destroy()
            widgets.clear()
        for key in self.rows:
            self.rows[key] = {k: v for k, v in self.rows[key].items() if not isinstance(k, tuple)}
        self.srh_labels = []

        owned = self.trap_widgets["manual"]
        row = self.manual_first_trap_row
        for i, trap in enumerate(self.manual_traps, 1):
            row = self._trap_header(self.manual_tab, row, i, self.manual_traps, owned)
            for name, _, unit, desc, tex in manual_trap_params(i):
                kind = name.replace(str(i), "", 1) if name.endswith("_f") else name[:-len(str(i))]
                if kind == "k_EC_f":
                    continue
                self.rows["manual"][(i, kind)] = self._row(self.manual_tab, row, tex, trap[kind],
                                                           unit, owned)
                row += 1
        self._trap_footer(self.manual_tab, row, self._add_manual_trap, owned)

        owned = self.trap_widgets["srh"]
        row = self.srh_first_trap_row
        for i, trap in enumerate(self.srh_traps, 1):
            row = self._trap_header(self.srh_tab, row, i, self.srh_traps, owned)
            for name, _, unit, desc, tex in srh_trap_params(i):
                kind = name[:-len(str(i))]
                widgets = self._row(self.srh_tab, row, tex, trap[kind], unit, owned)
                widgets["desc"].configure(text=desc)
                self.rows["srh"][(i, kind)] = widgets
                row += 1
            labels = []
            for tex in (base.k(f"EE{i}"), base.k(f"HE{i}")):
                symbol = ttk.Label(self.srh_tab, image=self._symbol(tex))
                symbol.grid(row=row, column=0, sticky="w", pady=1)
                value = ttk.Label(self.srh_tab, style="Converted.TLabel")
                value.grid(row=row, column=1, columnspan=4, sticky="w")
                owned += [symbol, value]
                labels.append(value)
                row += 1
            self.srh_labels.append(labels)
        self._trap_footer(self.srh_tab, row, self._add_srh_trap, owned)
        self._queue_schematic()

    def _add_manual_trap(self):
        self.manual_traps.append(self._new_manual_trap())
        self._rebuild_traps()

    def _add_srh_trap(self):
        self.srh_traps.append(self._new_srh_trap())
        self._rebuild_traps()

    def _remove_trap(self, traps, index):
        del traps[index]
        self._rebuild_traps()

    # ------------------------------------------------------------------- refresh
    @staticmethod
    def _rate_text(value):
        if math.isnan(value):
            return "calculated from the above"
        return (f"= {value:.3g} ns⁻¹  "
                f"{base.converted_text(str(value), 1e9, 's⁻¹')}, by detailed balance")

    def _reference_density(self):
        """Geometric mean of the entered initial densities, or 1e16 until they are valid."""
        try:
            densities = [float(s) for s in base.re.split(r"[,;\s]+", self.n0_var.get().strip()) if s]
            return math.exp(sum(map(math.log, densities)) / len(densities))
        except (ValueError, ZeroDivisionError):
            return 1e16

    def _refresh_manual_rows(self, shared, traps, active):
        rows = self.rows["manual"]
        for name, _, _, desc, _ in RECOMBINATION_PARAMS:
            field, widgets = self.manual_shared[name], rows[name]
            self._toggle(widgets["entry"], field.on)
            if name == "n_i" and name in shared and shared[name] != 0 and name not in active:
                desc = f"{desc} (needs k_B)"
            widgets["desc"].configure(text=desc)
        for i, trap in enumerate(self.manual_traps, 1):
            ticks = traps[i - 1]
            table = {name.replace(str(i), "", 1) if name.endswith("_f") else name[:-len(str(i))]:
                     (name, unit, desc) for name, _, unit, desc, _ in manual_trap_params(i)}
            for kind in ("k_EC", "k_EE", "k_HC", "k_HE", "N_T"):
                widgets = rows[(i, kind)]
                self._toggle(widgets["entry"], trap[kind].on)
                shown = self._capture_kind(trap) if kind == "k_EC" else kind
                name, unit, desc = table[shown]
                if kind == "k_EC":
                    # Only on a real change, as this would move the cursor while typing
                    if str(widgets["entry"].cget("textvariable")) != str(trap[shown].value):
                        widgets["entry"].configure(textvariable=trap[shown].value)
                        widgets["converted"].configure(textvariable=trap[shown].converted)
                    widgets["unit"].configure(text=unit)
                if kind in ticks and ticks[kind] != 0 and name not in active:
                    desc = f"{desc} ({inert_reason(kind, ticks)})"
                widgets["desc"].configure(text=desc)

    def _refresh_srh_rows(self):
        for name, _, _, _, _ in SRH_RECOMBINATION_PARAMS:
            self._toggle(self.rows["srh"][name]["entry"], self.srh_shared[name].on)
        self._toggle(self.ni_row["entry"], self.ni_field.on)
        if self.ni_field.on.get():
            self.ni_row["desc"].configure(text=MANUAL_NI_WARNING, style="Warning.TLabel")
        else:
            value = self._srh_ni()
            if not math.isnan(value) and self.ni_field.value.get() != f"{value:.3g}":
                self.ni_field.value.set(f"{value:.3g}")
            self.ni_row["desc"].configure(text=CALCULATED_NI, style="Muted.TLabel")
        Eg = self.material["E_g"].number()
        for i, trap in enumerate(self.srh_traps, 1):
            emission, hole_emission = self._srh_emission(trap)
            for label, value in zip(self.srh_labels[i - 1], (emission, hole_emission)):
                label.configure(text=self._rate_text(value))
            depth = trap["depth"].number()
            widgets = self.rows["srh"][(i, "depth")]
            if not math.isnan(depth) and 0 < depth < Eg:
                widgets["converted"].configure(text=f"(E_t − E_V = {Eg - depth:.3g} eV)")
            else:
                widgets["converted"].configure(text="(outside the band gap)")

    def _draw_schematic(self):
        """Rebuild the model from the tab in view, then draw it."""
        if self.notebook is None:
            return
        srh = self._is_srh()
        shared, traps = self._tab_inputs()
        active = effective_params(shared, traps)
        if srh:
            self._refresh_srh_rows()
        else:
            self._refresh_manual_rows(shared, traps, active)

        levels = energies = None
        if srh:
            Eg = self.material["E_g"].number()
            depths = [t["depth"].number() for t in self.srh_traps]
            in_gap = [Eg > 0 and 0 < d < Eg for d in depths]
            levels = [trap_level(d, Eg) if ok else (VB_TOP + CB_BOTTOM) / 2
                      for d, ok in zip(depths, in_gap)]
            energies = [f"{d:.3g} eV" if ok else "" for d, ok in zip(depths, in_gap)]
        else:
            n_ref = self._reference_density()
            ratios = [coupling_ratio(trap, i, active, n_ref) if has_trap(active, i) else math.nan
                      for i, trap in enumerate(traps, 1)]
            levels = [ratio_level(r) for r in ratios]
            energies = [ratio_text(r) for r in ratios]
        n_traps = len(traps)
        equations = build_equations(active, n_traps, srh)
        changed = equations != self.spec["equations"]
        self.spec.update(equations=equations,
                         schematic=build_schematic(active, n_traps, levels, energies),
                         variables=build_variables(active, n_traps, srh),
                         params=self._spec_params(active))
        preset = SRH_LABEL if srh else matching_preset(active, n_traps)
        self.preset_var.set(preset)
        self.model_label.configure(text=preset if preset != CUSTOM else "Custom model")
        self.desc_label.configure(text=describe(active, n_traps, srh))
        if changed:
            self._show_equations()
            if self.key_window is not None and self.key_window.winfo_exists():
                self._show_key()
        super()._draw_schematic()


if __name__ == "__main__":
    SelectiveApp().mainloop()
