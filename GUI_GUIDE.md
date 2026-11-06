# TRPL GUI Guide

Installation, instructions and theory for the two interactive simulators:

| Program | Use it to |
|---------|-----------|
| `TRPL_GUI.py` | Simulate one of the fixed models in `globalfit_functions.py` |
| `TRPL_GUI_Selective.py` | Build your own model from individual processes and any number of traps, either setting every rate by hand or using Shockley-Read-Hall (SRH) statistics |

**Contents:** [Installation](#installation) · [Using the simulators](#using-the-simulators) · [The model builder](#the-model-builder) · [Theory](#theory) · [Limitations](#limitations) · [Troubleshooting](#troubleshooting) · [References](#references)

---

## Installation

You need Python 3.11 or later, with Tkinter. Tkinter comes with the python.org and Anaconda installers on Windows and macOS. On Debian or Ubuntu, install it with `sudo apt install python3-tk`.

Download the code:

```bash
git clone https://github.com/barnlewis97/Perovskite-TRPL-Kinetics-Simulations.git
cd Perovskite-TRPL-Kinetics-Simulations
```

Then install the packages using **one** of these options.

**pip, in a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**conda, using the exact environment from `HMC_env.yml`:**

```bash
conda env create -f HMC_env.yml
conda activate MCMC_env
```

Start either program from the repository folder:

```bash
python TRPL_GUI.py
python TRPL_GUI_Selective.py
```

`TRPL_GUI_Selective.py` needs `TRPL_GUI.py` and `globalfit_functions.py` beside it, so keep the three files together.

---

## Using the simulators

These steps are the same in both programs.

1. **Choose the model.** Pick one from the drop-down in `TRPL_GUI.py`, or [build one](#the-model-builder) in `TRPL_GUI_Selective.py`.
2. **Enter the initial carrier densities.** Use cm⁻³, separated by commas, for example `1e14, 1e15, 1e16`. Enter as many as you like.
3. **Set the parameters.** Inputs use cm⁻³ and ns. The grey brackets show the same rate per second, and the maximum time in µs. They are for reading only.
4. **Set the simulation options.**
   - **Max time (ns):** the end of the simulated decay.
   - **Time points:** the number of log-spaced points, from t = 0 to the max time. More points give smoother derivatives.
   - **Background:** a constant added to the normalised PL, relative to its initial value, before the differential analysis.
5. **Press Simulate** (or Enter). A new model or trap count takes a few seconds the first time, while JAX compiles it.
6. **Read the three plots.**
   - **TRPL:** normalised PL against time. The Display card switches the time axis between log and linear.
   - **Differential lifetime:** τ_diff against n_PL.
   - **Differential coefficient:** k_diff against n_PL.
7. **Adjust the display.** "Dots per curve" only thins what is drawn; the derivatives and the export always use the full time grid. The toolbar under the plots zooms, pans and saves figures.
8. **Press the round i button** for a key to every symbol in the current model.
9. **Press Export data…** to save the latest simulation as a `.npy` file in the format the HMC repository reads. Its shape is `(1 + n_decays, n_time)`, with time in ns in row 0, then one normalised decay per row from the highest initial density to the lowest.

---

## The model builder

`TRPL_GUI_Selective.py` shows the model you have built above the plots: its equations and band diagram update with every change. Trap subscripts are:

| Subscript | Process | Older name in `TRPL_GUI.py` |
|-----------|---------|-----------------------------|
| EC | Electron capture | T (trapping) |
| EE | Electron emission | DT (de-trapping) |
| HC | Hole capture | DP (depopulation) |
| HE | Hole emission | HE |

Traps are simply numbered, with no assumption that any is shallow or deep. It starts with one trap. **Add Trap** adds another with all its parameters, and **Remove trap** beside a trap's heading deletes it; the rest are renumbered.

> ⚠️ Each trap adds a rate equation. Too many traps can cause numerical instability and make the simulation crash.

The parameters are set on one of two tabs. Each tab keeps its own values and traps, and the one in view is what gets simulated.

### Manual tab

Every rate is independent, so nothing enforces detailed balance. Use it to explore mechanisms or to reproduce a published phenomenological model.

- **Tick a process to add it.** Its terms join the equations and its arrow joins the band diagram.
- **Trap density:** with N_T ticked, capture is `k_EC Δn (N_T − n_T)` in cm³ ns⁻¹, and the trap can saturate. With it unticked, the trap cannot saturate, and capture is first order, `k_EC Δn` in ns⁻¹. The row's unit and value switch to match.
- **Processes with no effect:** the row says why, for example "needs capture". Emission and hole capture need a populated trap, hole emission needs a density, and n_i needs k_B. A ticked density of zero removes that trap.
- **Band diagram:** each trap's height follows how strongly it exchanges electrons with each band, using `R = (k_EC n_ref + k_EE) / (k_HC n_ref + k_HE)`, shown beside it. Strong capture and emission with weak depopulation gives a large R and a level near the CB; the reverse moves it towards the VB. `n_ref` is the geometric mean of your Δn₀ values, and first-order capture counts as `k_EC` directly. **R is only a guide to how the trap behaves, not an energy,** and the position is not to scale. Use the SRH tab for real trap energies.
- **Radiative recombination:** without `k_B` no PL is emitted, so nothing is plotted and the status line says so.
- **Model drop-down:** this loads a fixed model as a starting point, then you can change anything. The heading names the model whenever the ticks match one.

How closely each loaded model matches the original (largest difference in log₁₀ PL, with PL above 10⁻⁶ and Δn₀ from 10¹⁴ to 10¹⁸ cm⁻³):

| Model | Difference | Notes |
|-------|-----------|-------|
| `DTDeepVar`, `Full Model - 2 Traps` | below 10⁻¹² | Same equations and solver settings |
| `DTShallowVar`, `Full Model - 1 Trap` | about 2×10⁻⁴ | Same equations, numerical differences only |
| `BTD`, `ABC` | below 10⁻² | Same equations, with `k_A = C/2` for ABC; the originals use a looser solver tolerance |
| `ShallowTrapVar (approximate)` | about 10⁻² | Auger is `k_aug n² p` in the original, and p₀ only affects its PL |
| `DT (approximate)` | **up to 1.4 decades** | The original uses `I_PL ∝ n(n + p₀)`, not `n p`, and p₀ only affects its PL. With p₀ = 0, n(t) agrees to better than 1% |

### SRH (detailed balance) tab

Emission rates are not free parameters here. Each trap is set by its depth, capture coefficients and density, and the emission rates follow, shown in grey under the trap. They update as you type.

- **Material:** E_g, T, N_C and N_V. n_i is calculated from these. Tick it to set n_i by hand, but a warning appears because detailed balance with the traps is then no longer exact.
- **Recombination:** k_B, k_A, k_NR and net p-doping (N_A) can each be ticked. Doping changes the equilibrium the decay starts from.
- **Each trap:** `E_C − E_t` (the depth below the conduction band, which must lie inside the band gap), `k_EC`, `k_HC` and `N_T`. The grey bracket gives the depth from the valence band.
- **Band diagram:** each trap is drawn to scale across the band gap and labelled with its depth below the conduction band, so it moves as you type. A level very close to a band edge is held just inside it to keep its arrows visible, and its label still gives the true depth.

Default values, which are starting points for a triple-cation film and should be replaced for your material:

| Parameter | Default | Source |
|-----------|---------|--------|
| E_g | 1.625 eV | Heumann et al. [1], 85:15 film |
| T | 300 K | [1] |
| N_C = N_V | 2.2×10¹⁸ cm⁻³ | [1], taken from Staub et al. [6] |
| n_i | 4.93×10⁴ cm⁻³ (calculated) | [1] quotes 4.96×10⁴ |
| k_B | 10⁻²⁰ cm³ ns⁻¹ (10⁻¹¹ cm³ s⁻¹) | [1], 85:15 film |
| New trap: depth | 0.15 eV | [1], 85:15 film, defect 1 |
| New trap: N_T | 6.6×10¹⁶ cm⁻³ | [1], 85:15 film, defect 1 |
| New trap: k_EC = k_HC | 2.7×10⁻¹⁹ cm³ ns⁻¹ | From τ_SRH = 56 ns in [1], using k = 1/(τ N_T) |
| k_A | 8×10⁻³⁷ cm⁶ ns⁻¹ | Kumar et al. [5] |
| p-doping | 3.27×10¹⁵ cm⁻³ | [5] |
| k_NR | 10⁻⁴ ns⁻¹ | Placeholder, not from the literature |

Hüpkes et al. [2] Table 1 collects calculated trap energies and capture coefficients for specific defects, which is a good source for realistic inputs.

---

## Theory

### Rate equations

For excess electrons, holes and the electrons in each trap i, the general model is

$$\frac{dn}{dt} = -k_B(np - n_i^2) - k_A(np^2 + pn^2) - k_{NR}n - \sum_i \left[k_{EC,i}\, n\,(N_{T,i} - n_{T,i}) - k_{EE,i}\, n_{T,i}\right]$$

$$\frac{dp}{dt} = -k_B(np - n_i^2) - k_A(np^2 + pn^2) - k_{NR}n - \sum_i \left[k_{HC,i}\, p\, n_{T,i} - k_{HE,i}(N_{T,i} - n_{T,i})\right]$$

$$\frac{dn_{T,i}}{dt} = k_{EC,i}\, n\,(N_{T,i} - n_{T,i}) - k_{EE,i}\, n_{T,i} - k_{HC,i}\, p\, n_{T,i} + k_{HE,i}(N_{T,i} - n_{T,i})$$

Each term has a direct meaning:

- **Radiative recombination,** `k_B(np − n_i²)`: band-to-band emission minus thermal generation. The generation term appears in the equations but isn't drawn in the band diagram.
- **Auger,** `k_A(np² + pn²)`: three-carrier recombination, using one coefficient for both the eeh and ehh processes.
- **Electron capture,** `k_EC n (N_T − n_T)`: proportional to the free electrons and the empty traps.
- **Electron emission,** `k_EE n_T`: a trapped electron returning to the conduction band.
- **Hole capture,** `k_HC p n_T`: a hole recombining with a trapped electron. This is the step that actually removes a carrier pair through the trap.
- **Hole emission,** `k_HE (N_T − n_T)`: an electron promoted from the valence band into an empty trap.
- **First-order loss,** `k_NR n`: removes one electron and one hole, with no trap population tracked.

The equations conserve charge, so `p − n − Σ n_T` stays constant. On the Manual tab, p = Δp + p₀, the decay starts from Δn = Δp = Δn₀ with empty traps, and I_PL ∝ k_B n p.

### Detailed balance and trap depth

In thermal equilibrium every process must be balanced by its reverse. For a trap this fixes each emission rate from its capture coefficient and the trap energy [1–4]:

$$k_{EE} = k_{EC}\, n_1, \qquad n_1 = N_C \exp\left(-\frac{E_C - E_t}{k_B T}\right)$$

$$k_{HE} = k_{HC}\, p_1, \qquad p_1 = N_V \exp\left(-\frac{E_t - E_V}{k_B T}\right)$$

Here n₁ and p₁ are the electron and hole densities you would have if the Fermi level sat at the trap level. Multiplying them gives

$$n_1 p_1 = N_C N_V \exp\left(-\frac{E_g}{k_B T}\right) = n_i^2$$

This is why independently chosen emission rates can be thermodynamically inconsistent, and why the SRH tab sets energies instead. Heumann et al. [1] make this point directly: without the trap depth as an explicit parameter, detailed balance is not enforced. A trap's depth also determines how it behaves:

- **Deep traps** (small n₁ and p₁): emission is negligible, so capture is effectively irreversible and the trap acts as a recombination centre.
- **Shallow traps** (large n₁ or p₁): carriers are captured and re-emitted many times before recombining. They rarely dominate recombination directly, but they store charge and strongly shape the decay [2–4].

For a single trap in steady state, these rate equations reduce to the classic SRH rate, with τ_n = 1/(k_EC N_T) and τ_p = 1/(k_HC N_T) [2,3]:

$$R_{SRH} = \frac{np - n_i^2}{\tau_p (n + n_1) + \tau_n (p + p_1)}$$

### Initial conditions in SRH mode

Traps can be partly filled before the laser pulse, so the SRH tab starts from thermal equilibrium rather than from empty traps [1]. With E_V = 0,

$$n_0 = N_C e^{(E_F - E_g)/k_B T}, \quad p_0 = N_V e^{-E_F/k_B T}, \quad n_{T,0} = \frac{N_T}{1 + e^{(E_t - E_F)/k_B T}}$$

and the Fermi level is solved numerically from charge neutrality:

$$p_0 - n_0 - \sum_i n_{T,0,i} - N_A = 0$$

The pulse then adds Δn₀ to both n and p, so the SRH tab displays the rate equations for the excess carriers, with n = n₀ + Δn and p = p₀ + Δp. Unlike p₀ on the Manual tab, which is a fixed background, these n₀ and p₀ are calculated, and doping is the input N_A. n_T is the total trapped density, starting from n_T,0. The program takes all traps to be acceptor-like (neutral when empty, negative when filled), as in [1,2], with N_A the net p-doping. Since `np = n_i²` at equilibrium, every net rate vanishes there. I_PL ∝ k_B(np − n_i²), so the signal is the excess over the dark level.

### Differential analysis

A decay that isn't a single exponential has no single lifetime, so the decay time is treated as a quantity that changes through the decay [3–4, 7]. With the factor of 2 reflecting I_PL ∝ np in the high-level, n ≈ p picture:

$$\tau_{diff} = \left(-\frac{1}{2}\frac{d\ln I_{PL}}{dt}\right)^{-1}, \quad k_{diff} = -\frac{1}{2 n_{PL}}\frac{d\ln I_{PL}}{dt}, \quad n_{PL} = \Delta n_0 \sqrt{\frac{I_{PL}(t)}{I_{PL}(0)}}$$

n_PL is the carrier density inferred from the PL alone. When trapped charge makes n and p differ, n_PL is not the true electron density, so model comparisons should be made on these measured-style quantities.

### Shallow traps and power-law decays

In an intrinsic material with a high density of shallow, acceptor-like traps near the conduction band, the free carriers follow a bimolecular-like law with an effective coefficient [4]:

$$-\frac{dn}{dt} = k_n n^2, \qquad k_n = \frac{1 + k_{rad}\,\tau_{SRH,p}\, n_1}{\tau_{SRH,p}\, n_1}$$

This gives PL ∝ 1/t² at long times and τ_diff ∝ t, which means the measured decay time keeps growing with the observation window [3,4]. Yuan et al. [3] use this to explain decay times reaching hundreds of microseconds without needing deep traps to control the dynamics. Try it on the SRH tab: one shallow trap gives a PL slope approaching −2 on the log-log plot.

### Numerics

The equations are stiff, so they are solved with the implicit Kvaerno5 method from Diffrax, with adaptive steps (`rtol = 1e-5`, `atol = 1e-8`). Derivatives for the differential analysis are taken numerically on the full time grid. Time is log-spaced as `t_i = 10^{x_i} − 1`.

---

## Limitations

These are simplifications in the simulators, so check that they are acceptable before fitting real data.

- **One pulse only.** Every decay starts from equilibrium. Real measurements use repeated pulses, and if carriers remain when the next pulse arrives, the repetition rate changes the decay; Yuan et al. [4] show this matters for power-law decays. A pulse train isn't simulated.
- **Acceptor-like traps only.** Donor-like traps follow the same rate equations but have a different charge balance, which would change the SRH starting equilibrium. Hüpkes et al. [2] Table 1 lists both types.
- **Single charge transition.** Amphoteric defects with several charge states aren't included.
- **No Auger-assisted trapping.** Kumar et al. [5] include a trap with an Auger capture channel, which isn't available here.
- **Zero-dimensional.** There is no diffusion, surface recombination, photon recycling or reabsorption, so uniform excitation of a film is assumed.
- **`k_NR` on the SRH tab isn't detailed-balanced.** It is a phenomenological loss with no generation partner, so equilibrium is only a true fixed point with it switched off. For a thermodynamically consistent loss, add a deep trap instead.
- **Fixed T and E_g.** Trap energies, N_C, N_V and rates do not vary with temperature on their own; change each by hand.

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---------|---------------------|
| `No module named 'tkinter'` | Install Tkinter: `sudo apt install python3-tk` on Debian or Ubuntu. On other systems, use the python.org or Anaconda installer. |
| `No module named 'jax'` or `'diffrax'` | The environment isn't active. Activate it and rerun [the install step](#installation). |
| Pause at the first Simulate | Normal: JAX is compiling. Repeat runs with the same trap count are fast. |
| "No valid PL" in the status line | k_B is unticked or zero, or the solver failed for those densities. |
| Curves stop before the max time | PL has fallen below solver precision. Those points are NaN and the HMC scripts reject them, so reduce the max time before exporting. |
| Simulation fails or looks unstable | Remove traps, shorten the max time, or use less extreme rates. Very fast emission (very shallow traps) makes the equations stiffer. |
| Rows say "needs capture" | That process has no effect until the trap can be populated. |
| Band diagram too small to read | It scales to fit, so it gets small with many traps. The equations and symbol key still list everything. |

---

## References

1. R. Heumann, T. Rudolph, G. Huang, T. Kirchartz, C. Dreessen, *Rapid Parameter Estimation from Photoluminescence Decays of Halide Perovskite Thin Films*, [arXiv:2609.18438](https://arxiv.org/abs/2609.18438). Rate equations, detailed balance, initial conditions and default values.
2. J. Hüpkes, U. Rau, T. Kirchartz, *Impact of Trap Depth on the Steady-State and Transient Photoluminescence in Halide Perovskite Films*, Adv. Energy Mater. 2025, e03157. Review of trap depth and charge state, with a table of defect levels.
3. Y. Yuan et al., *Shallow defects and variable photoluminescence decay times up to 280 µs in triple-cation perovskites*, Nat. Mater. 23, 391 (2024), [doi:10.1038/s41563-023-01771-2](https://doi.org/10.1038/s41563-023-01771-2).
4. Y. Yuan, G. Yan, C. Dreessen, T. Kirchartz, *Understanding Power-Law Photoluminescence Decays and Bimolecular Recombination in Lead-Halide Perovskites*, Adv. Energy Mater. 15, 2403279 (2025), [doi:10.1002/aenm.202403279](https://doi.org/10.1002/aenm.202403279). Effective recombination coefficient and repetition-rate effects.
5. J. Kumar et al., *Unraveling the Roles of Shallow, Deep and Auger Trapping in Charge Carrier Recombination in Triple-Cation Perovskites*, [arXiv:2608.14368](https://arxiv.org/abs/2608.14368).
6. F. Staub et al., *Beyond Bulk Lifetimes: Insights into Lead Halide Perovskite Films from Time-Resolved Photoluminescence*, Phys. Rev. Applied 6, 044017 (2016).
7. B. A. I. Lewis et al., *Differential Transformations for Understanding Realistic Recombination Models in Perovskite Semiconductors* (manuscript, `paper_kdiff_BTD.pdf`). Definitions of τ_diff, k_diff and n_PL (Eq. S1), and the BTD model.
8. W. Shockley, W. T. Read, *Statistics of the recombinations of holes and electrons*, Phys. Rev. 87, 835 (1952).
