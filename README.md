# TRPL Rate Equation Models for Charge Carrier Dynamics

A JAX-accelerated library for simulating and fitting Time-Resolved Photoluminescence (TRPL) decay curves in semiconductor materials using physically-motivated rate equation models (REMs). Designed for use with Bayesian inference workflows (HMC, ABC).

---

## Overview

This repository provides:

- A library of ODE-based charge carrier models (`globalfit_functions.py`) compiled with JAX JIT for fast, differentiable simulation
- Jupyter notebooks demonstrating simulation and analysis of TRPL decays for different physical regimes
- An interactive GUI (`TRPL_GUI.py`) for simulating any model, with a band-diagram schematic that follows the selected processes and `.npy` export for the HMC fitting pipeline
- A pinned `requirements.txt` for reproducible installation

The models are intended for fitting experimental TRPL data from semiconductor thin films (e.g. perovskites), and are compatible with probabilistic inference frameworks such as NumPyro (HMC/NUTS) or approximate Bayesian computation (ABC). The Bayesian fitting pipeline that uses these models is in the companion repository [HMC-for-Perovskite-TRPL-Kinetics](https://github.com/barnlewis97/HMC-for-Perovskite-TRPL-Kinetics).

---

## Models

Every model in `globalfit_functions.py` follows the same layout:

| Function | Purpose |
|----------|---------|
| `<Name>_Model(t, y, args)` | Rate equations (Diffrax right-hand side) |
| `solve_<Name>(t, ...)` | Solves the ODE system at the times in `t` |
| `TRPL_<Name>(t, ..., bkg)` | Returns `(log10 TRPL signal, carrier densities...)` |

The TRPL signal is normalised to its value at t = 0 with a background `bkg` added. Carrier densities are returned in the order free electrons, holes, trapped electrons. Units are cm⁻³ and ns throughout.

### ABC Model — `TRPL_ABC`
A standard three-process model for charge carrier decay. `TRPL_AB` is the same model with `k_C = 0`.

$$\frac{dn}{dt} = -k_A n - k_B n^2 - k_C n^3$$

| Parameter | Description |
|-----------|-------------|
| `k_A` | SRH (trap-mediated) recombination rate constant (ns⁻¹) |
| `k_B` | Bimolecular recombination rate constant (cm³ ns⁻¹) |
| `k_C` | Auger recombination rate constant (cm⁶ ns⁻¹) |
| `n_0` | Initial carrier density (cm⁻³) |

Returns `(signal, n)`.

---

### BTD Model — `TRPL_BTD`
Bimolecular-Trapping-Detrapping model tracking electrons, trapped carriers, and holes separately, including Auger recombination, trapping, detrapping, and trap depopulation:

| Parameter | Description |
|-----------|-------------|
| `ka` | Auger rate constant (cm⁶ ns⁻¹) |
| `kt` | Trapping rate constant (cm³ ns⁻¹) |
| `kb` | Bimolecular rate constant (cm³ ns⁻¹) |
| `kdt` | Detrapping rate constant (ns⁻¹, conduction band emission) |
| `kdp` | Depopulation rate constant (cm³ ns⁻¹, trap → valence band) |
| `NT` | Total trap density (cm⁻³) |
| `p0` | Background / equilibrium hole density (cm⁻³) |
| `N0` | Initial carrier density (cm⁻³) |

Returns `(signal, n_e, n_h, n_t)`.

---

### Dual Trap Models
Two trap populations: a shallow trap that captures and re-emits electrons (de-trapping, no recombination) and a deep trap that removes carriers non-radiatively (depopulation). The variants differ in which trap has a finite density, so that capture scales with (N_T − n_T):

| Model | Shallow trap | Deep trap | Returns |
|-------|--------------|-----------|---------|
| `TRPL_DT` | constant capture rate | constant capture rate | `(signal, n, p, n_t)` |
| `TRPL_DTShallowVar` | finite density N_t1 | constant capture rate | `(signal, n, p, n_t1)` |
| `TRPL_DTDeepVar` | constant capture rate | finite density N_t2, hole capture by filled traps | `(signal, n, p, n_t1, n_t2)` |

`DT` is the model from [DOI: 10.1103/PRXEnergy.4.013001](https://doi.org/10.1103/PRXEnergy.4.013001) and includes Auger recombination. `DTShallowVar` gets holes from charge neutrality (p = n + n_t1), so deep-trap recombination is treated as instantaneous.

---

### ShallowTrapVar Model — `TRPL_ShallowTrapVar`
A single shallow trap with finite density N_T, plus radiative and Auger recombination. Returns `(signal, n, p, n_t)`.

---

### Full REM — `FullREM_Model`
A general 4-state model tracking electrons, holes, and two independent trap populations with full capture/emission kinetics for each trap (rate equations only, no solver):

$$\frac{dn}{dt},\ \frac{dp}{dt},\ \frac{dn_{t1}}{dt},\ \frac{dn_{t2}}{dt}$$

---

## Analysis Utilities

| Function | Description |
|----------|-------------|
| `diff_lifetime` | Differential carrier lifetime τ(t) from PL trace |
| `diff_constant` | Differential bimolecular rate constant k(t) from PL trace |
| `relative_QFLS` | Quasi-Fermi level splitting relative to a reference density |
| `add_noise` | Add Gaussian noise to simulated signals |
| `colorFader` | Interpolate between two matplotlib colours |
| `standardise` | Zero-mean, unit-variance normalisation |
| `normalise` | Max-normalise an array |

---

## Notebooks

There is a simulation notebook for each of the `ABC`, `BTD`, `DT`, `DTShallowVar` and `DTDeepVar` models (`<Name> Simulation.ipynb`). Each simulates TRPL decays across a range of initial carrier densities and produces:
- TRPL decay curves (log-log) with and without background signal
- Differential lifetime τ vs QFLS
- Differential rate constant k vs QFLS
- Carrier concentration dynamics (free electrons, trapped electrons, holes, available traps) for the trap models
- Stacked contribution plots showing the relative weight of each recombination and trapping process over time
- Parameter sweeps showing the effect of the key trap parameters (the BTD notebook sweeps k_T, N_T and k_DP)

---

## Interactive GUI

**See [GUI_GUIDE.md](GUI_GUIDE.md) for full installation steps, instructions, theory, limitations and troubleshooting.**

`TRPL_GUI.py` simulates any of the models (`ABC`, `BTD`, `DT`, `DTShallowVar`, `DTDeepVar`, `ShallowTrapVar`, `Full Model - 1 Trap`, `Full Model - 2 Traps`) from a Tkinter window. Both full models are solved with `FullREM_Model`, the 1-trap version by setting the second trap's density to zero:

```bash
python TRPL_GUI.py
```

- Pick the model from the drop-down menu. A one-sentence description, a band-diagram schematic and the typeset rate equations are shown above the plots. The round **i** button opens a key to every symbol used by that model.
- Enter any number of initial carrier densities, comma separated (e.g. `1e14, 1e15, 1e16`).
- Edit the parameter values (cm⁻³ and ns units, as in `globalfit_functions.py`). Unticking a parameter sets it to zero in the rate equations and removes its process from the schematic. Symbols follow the BTD notation in every model (k_T trapping, k_DT de-trapping, k_DP depopulation, k_HE hole emission, k_B bimolecular, k_A Auger, N_T trap density, with 1 for the shallow and 2 for the deep trap), except the ABC model, which uses A, B and C. Carrier densities are written Δn and Δp; in the BTD model p = Δp + p₀ is the total hole density. The argument names in `globalfit_functions.py` are unchanged.
- Three plots are produced:
  - the normalised TRPL, with a log-log / lin-log toggle for the time axis
  - τ_diff = (−½ d ln I_PL/dt)⁻¹ against the PL-derived carrier density n_PL = Δn₀ √(I_PL/I_PL(0)) (log-log)
  - k_diff = −(1/2n_PL) d ln I_PL/dt against n_PL (log-log)

  These are the definitions in the k_diff/BTD paper (Eq. S1). An optional background is added to the normalised PL before the transformations.
- Curves are drawn as dots. "Dots per curve" only thins what is plotted; the derivatives and the export always use the full time grid.
- **Export data** saves the latest simulation as `SimTRPL_<model>.npy` (the name can be changed), in the format the HMC repository reads: shape `(1 + n_decays, n_time)`, with the time axis in ns in row 0 and one normalised decay per row from the highest initial carrier density to the lowest.

The first simulation with each model takes a few seconds while JAX compiles it.

### Building a model from its processes

`TRPL_GUI_Selective.py` has the same plots, schematic, symbol key and export, but builds the model from individual processes instead of offering a fixed list. It needs `TRPL_GUI.py` alongside it:

```bash
python TRPL_GUI_Selective.py
```

It starts with one trap. **Add Trap** adds as many more as you like and **Remove trap** deletes one, though too many can cause numerical instability and make the simulation crash. Trap subscripts are EC electron capture, EE electron emission, HC hole capture and HE hole emission. There are two tabs:

- **Manual**: tick processes and set every rate independently. A trap with no density ticked cannot saturate and captures at a first-order rate. The drop-down loads any fixed model as a starting point. `BTD`, `DTShallowVar`, `DTDeepVar` and both full models are reproduced with the same equations, and `ABC` with k_A = C/2. `DT` and `ShallowTrapVar` are only approximate, because those models treat p₀ and the PL or Auger terms differently; the module docstring gives the details.
- **SRH (detailed balance)**: set each trap's depth, capture coefficients and density. The emission rates and n_i follow from detailed balance, and the decay starts from thermal equilibrium found from charge neutrality, following [Heumann et al., arXiv:2609.18438](https://arxiv.org/abs/2609.18438) (Eqs. 4–9 and 20–23). The default material values (E_g = 1.625 eV, T = 300 K, N_C = N_V = 2.2×10¹⁸ cm⁻³) and first trap are from that work. Change them to suit your material. Setting n_i by hand is allowed, but detailed balance with the traps is then no longer exact.

Each new number of traps compiles once, taking a few seconds.

---

## Installation

Requires Python 3.11 or later.

```bash
git clone https://github.com/barnlewis97/Perovskite-TRPL-Kinetics-Simulations.git
cd Perovskite-TRPL-Kinetics-Simulations

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook                 # then open any of the simulation notebooks

python TRPL_GUI.py               # or start a GUI
python TRPL_GUI_Selective.py
```

Alternatively, `conda env create -f HMC_env.yml` followed by `conda activate MCMC_env` creates the exact conda environment. The GUIs also need Tkinter, which comes with the python.org and Anaconda installers; on Debian or Ubuntu, run `sudo apt install python3-tk`.

`requirements.txt` pins the package versions used for the thesis simulations. For GPU acceleration, install the matching CUDA `jaxlib` wheel afterwards (see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)).

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `jax` / `jaxlib` | Accelerated numerical computing with JIT compilation |
| `diffrax` | JAX-native ODE solver (Kvaerno5 stiff solver used throughout) |
| `equinox` | JAX neural network / filter utilities |
| `numpy` | Standard numerical routines |
| `scipy` | Fermi-level root finding in the SRH model builder |
| `matplotlib` | Plotting |
| `jupyter` | Running the notebooks |

Exact versions are pinned in `requirements.txt`. Inference with NumPyro and ArviZ is handled in the [companion HMC repository](https://github.com/barnlewis97/HMC-for-Perovskite-TRPL-Kinetics).

---

## Usage

```python
from globalfit_functions import *
import jax.numpy as jnp

# Define time axis (log-spaced, 0–100 µs)
time = jnp.logspace(0, jnp.log10(100001), 1000) - 1

# Simulate BTD model TRPL decay
log_signal, n_e, n_p, n_t = TRPL_BTD(
    t=time,
    ka=1e-37,   # Auger
    kt=1e-17,   # Trapping
    kb=1e-19,   # Bimolecular
    kdt=1e-3,   # Detrapping
    kdp=1e-19,  # Depopulation
    NT=1e15,    # Trap density
    p0=1e13,    # Background holes
    N0=1e16,    # Initial carriers
    bkg=1e-5    # Background signal
)
```

---

## Physical Background

Time-Resolved Photoluminescence (TRPL) measures the decay of radiative recombination following pulsed photoexcitation. In semiconductor thin films (particularly metal halide perovskites), the decay shape is governed by competing recombination pathways:

- **Radiative (bimolecular):** proportional to n·p, the primary TRPL signal source
- **SRH / trap-mediated:** carriers captured by sub-bandgap defect states
- **Auger:** three-particle process dominant at high injection densities
- **Trap dynamics:** filling, emission, and depopulation of trap states modify the apparent carrier lifetime

The models here solve these coupled rate equations numerically using the stiff Kvaerno5 solver via Diffrax, and are JIT-compiled with JAX for use in gradient-based (HMC) and gradient-free (ABC) Bayesian inference loops.

---

## Citation

If you use this code in your research, please cite this repository and any relevant model references, including:

- [DOI: 10.1103/PRXEnergy.4.013001](https://doi.org/10.1103/PRXEnergy.4.013001) (DT Model)

---

## Licence

MIT Licence — see `LICENSE` for details.
