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
equations. Units follow globalfit_functions.py (cm^-3 and ns); tau_diff and
k_diff are reported in s and cm^3 s^-1.

Run with:  python TRPL_GUI.py   (in an environment with jax and diffrax)
"""

import os
import re
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

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
# Each parameter is (name, default value, unit, description). ``run`` maps
# (t, n0, params) to the log10 TRPL signal from globalfit_functions, called
# with zero background (the GUI adds its own background after normalisation
# so it means the same thing for every model).

RATE = "ns⁻¹"
BIMOL = "cm³ ns⁻¹"
AUGER = "cm⁶ ns⁻¹"
DENS = "cm⁻³"

MODELS = {
    "ABC": dict(
        params=[
            ("k_A", 1e-4, RATE, "First-order (trap) recombination"),
            ("k_B", 1e-19, BIMOL, "Bimolecular (radiative)"),
            ("k_C", 1e-37, AUGER, "Auger"),
        ],
        equations="dn/dt = −k_A n − k_B n² − k_C n³\nI_PL ∝ k_B n²",
        run=lambda t, n0, p: gf.TRPL_ABC(t, n0, p["k_A"], p["k_B"], p["k_C"], 0.0)[0],
    ),
    "BTD": dict(
        params=[
            ("ka", 1e-37, AUGER, "Auger"),
            ("kt", 1e-17, BIMOL, "Trapping (capture)"),
            ("kb", 1e-19, BIMOL, "Bimolecular (radiative)"),
            ("kdt", 1e-3, RATE, "De-trapping (trap → CB)"),
            ("kdp", 1e-19, BIMOL, "Depopulation (trap → VB)"),
            ("NT", 1e15, DENS, "Trap density"),
            ("p0", 1e13, DENS, "Background hole density"),
        ],
        equations=("dn/dt = −kb n p − ka(n p² + p n²) − kt n(NT − nT) + kdt nT\n"
                   "dnT/dt = kt n(NT − nT) − kdt nT − kdp nT p\n"
                   "dΔp/dt = −kb n p − ka(n p² + p n²) − kdp nT p\n"
                   "p = Δp + p0,  I_PL ∝ n p"),
        run=lambda t, n0, p: gf.TRPL_BTD(t, p["ka"], p["kt"], p["kb"], p["kdt"], p["kdp"],
                                         p["NT"], p["p0"], n0, 0.0)[0],
    ),
    "DT": dict(
        params=[
            ("k_c", 1e-3, RATE, "Shallow-trap capture"),
            ("k_deep", 1e-4, RATE, "Deep-trap capture"),
            ("k_e", 1e-4, RATE, "Shallow-trap emission"),
            ("k_rad", 1e-19, BIMOL, "Radiative"),
            ("k_aug", 1e-37, AUGER, "Auger"),
            ("p0", 1e13, DENS, "Background hole density"),
        ],
        equations=("dn/dt = −k_rad n p − ½k_aug(n²p + p²n) − k_c n + k_e nT − k_deep n\n"
                   "dnT/dt = k_c n − k_e nT\n"
                   "p = n + nT,  I_PL ∝ n(n + p0)"),
        run=lambda t, n0, p: gf.TRPL_DT(t, n0, p["k_c"], p["k_deep"], p["k_e"], p["k_rad"],
                                        p["k_aug"], p["p0"], 0.0)[0],
    ),
    "DTShallowVar": dict(
        params=[
            ("krad", 1e-19, BIMOL, "Radiative"),
            ("beta_n_t1", 1e-17, BIMOL, "Shallow-trap capture"),
            ("e_n_t1", 1e-3, RATE, "Shallow-trap emission"),
            ("N_t1", 1e15, DENS, "Shallow-trap density"),
            ("beta_n_t2", 1e-4, RATE, "Deep-trap capture"),
        ],
        equations=("dn/dt = −krad n p − β_n,t1 n(N_t1 − n_t1) + e_n,t1 n_t1 − β_n,t2 n\n"
                   "dn_t1/dt = β_n,t1 n(N_t1 − n_t1) − e_n,t1 n_t1\n"
                   "p = n + n_t1,  I_PL ∝ n p"),
        run=lambda t, n0, p: gf.TRPL_DTShallowVar(t, p["krad"], p["beta_n_t1"], p["e_n_t1"],
                                                  p["N_t1"], p["beta_n_t2"], n0, 0.0)[0],
    ),
    "DTDeepVar": dict(
        params=[
            ("krad", 1e-19, BIMOL, "Radiative"),
            ("beta_n_t1", 1e-3, RATE, "Shallow-trap capture"),
            ("e_n_t1", 1e-3, RATE, "Shallow-trap emission"),
            ("beta_n_t2", 1e-18, BIMOL, "Deep-trap electron capture"),
            ("beta_p_t2", 1e-19, BIMOL, "Deep-trap hole capture"),
            ("N_t2", 1e15, DENS, "Deep-trap density"),
        ],
        equations=("dn/dt = −krad n p − β_n,t1 n + e_n,t1 n_t1 − β_n,t2 n(N_t2 − n_t2)\n"
                   "dn_t1/dt = β_n,t1 n − e_n,t1 n_t1\n"
                   "dn_t2/dt = β_n,t2 n(N_t2 − n_t2) − β_p,t2 p n_t2\n"
                   "dp/dt = −krad n p − β_p,t2 p n_t2,  I_PL ∝ n p"),
        run=lambda t, n0, p: gf.TRPL_DTDeepVar(t, p["krad"], p["beta_n_t1"], p["e_n_t1"],
                                               p["beta_n_t2"], p["beta_p_t2"], p["N_t2"],
                                               n0, 0.0)[0],
    ),
    "ShallowTrapVar": dict(
        params=[
            ("k_c", 1e-17, BIMOL, "Capture"),
            ("k_e", 1e-3, RATE, "Emission"),
            ("k_rad", 1e-19, BIMOL, "Radiative"),
            ("k_aug", 1e-37, AUGER, "Auger"),
            ("NT", 1e15, DENS, "Trap density"),
            ("p0", 1e13, DENS, "Background hole density"),
        ],
        equations=("dn/dt = −k_rad n p − k_aug n²p − k_c n(NT − nT) + k_e nT\n"
                   "dnT/dt = k_c n(NT − nT) − k_e nT\n"
                   "p = n + nT,  I_PL ∝ n(p + p0)"),
        run=lambda t, n0, p: gf.TRPL_ShallowTrapVar(t, n0, p["k_c"], p["k_e"], p["k_rad"],
                                                    p["k_aug"], p["NT"], p["p0"], 0.0)[0],
    ),
    "FullREM": dict(
        params=[
            ("krad", 1e-19, BIMOL, "Radiative"),
            ("ni", 1e5, DENS, "Intrinsic carrier density"),
            ("beta_n_t1", 1e-17, BIMOL, "Trap 1 electron capture"),
            ("beta_p_t1", 0.0, BIMOL, "Trap 1 hole capture"),
            ("e_n_t1", 1e-3, RATE, "Trap 1 electron emission"),
            ("e_p_t1", 0.0, RATE, "Trap 1 hole emission"),
            ("N_t1", 1e15, DENS, "Trap 1 density"),
            ("beta_n_t2", 1e-18, BIMOL, "Trap 2 electron capture"),
            ("beta_p_t2", 1e-19, BIMOL, "Trap 2 hole capture"),
            ("e_n_t2", 0.0, RATE, "Trap 2 electron emission"),
            ("e_p_t2", 0.0, RATE, "Trap 2 hole emission"),
            ("N_t2", 1e15, DENS, "Trap 2 density"),
        ],
        equations=("dn/dt = −krad(np − ni²)\n"
                   "        − Σᵢ[β_n,ti n(N_ti − n_ti) − e_n,ti n_ti]\n"
                   "dp/dt = −krad(np − ni²)\n"
                   "        − Σᵢ[β_p,ti p n_ti − e_p,ti(N_ti − n_ti)]\n"
                   "dn_ti/dt = β_n,ti n(N_ti − n_ti) − β_p,ti p n_ti\n"
                   "           − e_n,ti n_ti + e_p,ti(N_ti − n_ti)\n"
                   "I_PL ∝ n p"),
        run=lambda t, n0, p: TRPL_FullREM(t, n0, *(p[name] for name, *_ in
                                                   MODELS["FullREM"]["params"]))[0],
    ),
}


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
    """Format 1.5e16 as 1.5×10¹⁶ in mathtext."""
    exp = int(np.floor(np.log10(x)))
    mant = x / 10**exp
    if np.isclose(mant, 1):
        return rf"$10^{{{exp}}}$"
    return rf"${mant:.3g}\times10^{{{exp}}}$"


def fader(i, n):
    """Red→blue colour sequence matching the simulation notebooks."""
    return gf.colorFader("red", "blue", factor=i / max(n - 1, 1))


# =============================================================================
# GUI
# =============================================================================

class TRPLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TRPL Kinetics Simulator")
        try:
            self.state("zoomed")        # maximised on Windows
        except tk.TclError:
            self.geometry("1500x820")

        self.model_var = tk.StringVar(value="BTD")
        self.n0_var = tk.StringVar(value="1e14, 1e15, 1e16, 1e17, 1e18")
        self.tmax_var = tk.StringVar(value="150000")
        self.npts_var = tk.StringVar(value="1000")
        self.bkg_var = tk.StringVar(value="0")
        self.xscale_var = tk.StringVar(value="log")
        self.status_var = tk.StringVar(value="Ready.")

        # Per-model parameter state, kept when switching between models
        self.param_state = {
            m: {name: (tk.BooleanVar(value=True), tk.StringVar(value=f"{val:g}"))
                for name, val, *_ in spec["params"]}
            for m, spec in MODELS.items()
        }
        self.results = None
        self._worker = None

        self._build_controls()
        self._build_figure()
        self._show_params()

    # ---------------------------------------------------------------- layout
    def _build_controls(self):
        panel = ttk.Frame(self, padding=8)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(panel, text="Model", font=("", 10, "bold")).pack(anchor="w")
        combo = ttk.Combobox(panel, textvariable=self.model_var, state="readonly",
                             values=list(MODELS), width=30)
        combo.pack(anchor="w", fill=tk.X)
        combo.bind("<<ComboboxSelected>>", lambda e: self._show_params())

        btns = ttk.Frame(panel)
        btns.pack(anchor="w", fill=tk.X, pady=(6, 0))
        self.run_btn = ttk.Button(btns, text="Simulate", command=self._run)
        self.run_btn.pack(side=tk.LEFT)
        ttk.Button(btns, text="Reset parameters", command=self._reset_params).pack(
            side=tk.LEFT, padx=6)
        ttk.Label(panel, textvariable=self.status_var, wraplength=380,
                  foreground="#1a5490").pack(anchor="w", pady=(4, 0))

        self.eq_label = ttk.Label(panel, font=("Consolas", 8), justify=tk.LEFT,
                                  foreground="#444")
        self.eq_label.pack(anchor="w", pady=(4, 8))

        ttk.Label(panel, text="Initial carrier densities Δn₀ (cm⁻³)",
                  font=("", 10, "bold")).pack(anchor="w")
        ttk.Label(panel, text="comma separated, any number").pack(anchor="w")
        ttk.Entry(panel, textvariable=self.n0_var, width=40).pack(anchor="w", fill=tk.X,
                                                                   pady=(0, 8))

        ttk.Label(panel, text="Parameters (untick → set to 0)",
                  font=("", 10, "bold")).pack(anchor="w")
        self.param_frame = ttk.Frame(panel)
        self.param_frame.pack(anchor="w", fill=tk.X, pady=(2, 8))

        sim = ttk.LabelFrame(panel, text="Simulation", padding=4)
        sim.pack(anchor="w", fill=tk.X, pady=(0, 8))
        for row, (label, var) in enumerate([("Max time (ns)", self.tmax_var),
                                            ("Time points", self.npts_var),
                                            ("Background (rel. to I₀)", self.bkg_var)]):
            ttk.Label(sim, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(sim, textvariable=var, width=12).grid(row=row, column=1, sticky="w")

        scale = ttk.LabelFrame(panel, text="TRPL time axis", padding=4)
        scale.pack(anchor="w", fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(scale, text="log-log", variable=self.xscale_var, value="log",
                        command=self._plot).pack(side=tk.LEFT)
        ttk.Radiobutton(scale, text="lin-log", variable=self.xscale_var, value="linear",
                        command=self._plot).pack(side=tk.LEFT, padx=8)

        self.bind("<Return>", lambda e: self._run())

    def _build_figure(self):
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(13, 5), dpi=100)
        self.axes = self.fig.subplots(1, 3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        NavigationToolbar2Tk(self.canvas, right).update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._plot()

    def _show_params(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        model = self.model_var.get()
        spec = MODELS[model]
        self.eq_label.config(text=spec["equations"])
        for row, (name, _, unit, desc) in enumerate(spec["params"]):
            on, val = self.param_state[model][name]
            entry = ttk.Entry(self.param_frame, textvariable=val, width=11)
            cb = ttk.Checkbutton(self.param_frame, text=name, variable=on,
                                 command=lambda e=entry, v=on: self._toggle(e, v))
            cb.grid(row=row, column=0, sticky="w")
            entry.grid(row=row, column=1, sticky="w", padx=4)
            ttk.Label(self.param_frame, text=unit, width=9).grid(row=row, column=2, sticky="w")
            ttk.Label(self.param_frame, text=desc, foreground="#666").grid(
                row=row, column=3, sticky="w")
            self._toggle(entry, on)

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
    def _read_inputs(self):
        model = self.model_var.get()
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
        bad = [r["n0"] for r in payload if not r["ok"]]
        if bad:
            self.status_var.set(
                f"{model}: no valid PL for Δn₀ = {', '.join(f'{b:.1e}' for b in bad)} "
                "(radiative term excluded or solver failed).")
        else:
            self.status_var.set(f"{model}: simulated {len(payload)} densities.")
        self._plot()

    # -------------------------------------------------------------- plotting
    def _plot(self):
        ax_pl, ax_tau, ax_k = self.axes
        for ax in self.axes:
            ax.clear()

        if self.results:
            model, results = self.results
            for i, r in enumerate(results):
                c = fader(i, len(results))
                label = sci_label(r["n0"])
                ax_pl.plot(r["t"], r["pl"], color=c, label=label)
                for ax, y in ((ax_tau, r["tau_diff"]), (ax_k, r["k_diff"])):
                    mask = np.isfinite(y) & (y > 0) & (r["n_pl"] > 0)
                    ax.plot(r["n_pl"][mask], y[mask], color=c)
            ax_pl.legend(title=r"$\Delta n_0$ (cm$^{-3}$)", fontsize="small",
                         title_fontsize="small", loc="lower left")
            self.fig.suptitle(f"{model} model", fontsize="large")

        ax_pl.set_xscale(self.xscale_var.get())
        ax_pl.set_yscale("log")
        for ax in (ax_tau, ax_k):
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax_pl.set_xlabel("Time (ns)")
        ax_pl.set_ylabel("Normalised PL (a.u.)")
        ax_pl.set_title("TRPL")
        ax_tau.set_xlabel(r"PL-derived carrier density $n_{PL}$ (cm$^{-3}$)")
        ax_tau.set_ylabel(r"$\tau_{diff}$ (s)")
        ax_tau.set_title("Differential lifetime")
        ax_k.set_xlabel(r"PL-derived carrier density $n_{PL}$ (cm$^{-3}$)")
        ax_k.set_ylabel(r"$k_{diff}$ (cm$^{3}$ s$^{-1}$)")
        ax_k.set_title("Differential coefficient")
        for ax in self.axes:
            ax.tick_params(which="both", direction="in")
        self.fig.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    TRPLApp().mainloop()
