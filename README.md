# TRPL Rate Equation Models for Charge Carrier Dynamics

A JAX-accelerated library for simulating and fitting Time-Resolved Photoluminescence (TRPL) decay curves in semiconductor materials using physically-motivated rate equation models (REMs). Designed for use with Bayesian inference workflows (HMC, ABC).

---

## Overview

This repository provides:

- A library of ODE-based charge carrier models (`globalfit_functions.py`) compiled with JAX JIT for fast, differentiable simulation
- Jupyter notebooks demonstrating simulation and analysis of TRPL decays for different physical regimes
- A conda environment specification for full reproducibility

The models are intended for fitting experimental TRPL data from semiconductor thin films (e.g. perovskites), and are compatible with probabilistic inference frameworks such as NumPyro (HMC/NUTS) or approximate Bayesian computation (ABC).

---

## Models

### ABC Model — `TRPL_ABC_Model`
A standard three-process model for charge carrier decay:

$$\frac{dn}{dt} = -k_A n - k_B n^2 - k_C n^3$$

| Parameter | Description |
|-----------|-------------|
| `k_A` | SRH (trap-mediated) recombination rate constant (ns⁻¹) |
| `k_B` | Bimolecular recombination rate constant (cm³ ns⁻¹) |
| `k_C` | Auger recombination rate constant (cm⁶ ns⁻¹) |
| `n_0` | Initial carrier density (cm⁻³) |

---

### BTD Model — `TRPL_BTD_Model`
An extended model tracking electrons, trapped carriers, and holes separately, including Auger recombination, trapping, detrapping, and trap depopulation:

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

---

### Dual Trap Models — `DualTrap_Model`, `DTShallowVar`, `DTDeepVar_Model`
Rate equation models with two distinct trap populations (shallow and deep), for materials exhibiting complex multi-exponential decays. Variants explore different assumptions about trap filling, charge neutrality, and which carriers are tracked explicitly.

---

### Full REM — `Full_REM_Model`
A general 4-state model tracking electrons, holes, and two independent trap populations with full capture/emission kinetics for each trap:

$$\frac{dn}{dt},\ \frac{dp}{dt},\ \frac{dn_{t1}}{dt},\ \frac{dn_{t2}}{dt}$$

---

### DT Model — `DT_Model` / `TRPL_DT_Model`
Two-variable model (electrons + shallow trap) including radiative recombination, Auger recombination, capture, deep-trap loss, and emission. Based on [DOI: 10.1103/PRXEnergy.4.013001](https://doi.org/10.1103/PRXEnergy.4.013001).

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

### `ABC_Simulation.ipynb`
Simulates TRPL decay curves and differential transformations across a range of initial carrier densities using the ABC model. Produces:
- TRPL decay curves (log-log) with and without background signal
- Differential lifetime τ vs QFLS
- Differential rate constant k vs QFLS
- Stacked contribution plots showing relative weight of Auger, bimolecular, and trapping recombination over time

### `BTD_Simulation.ipynb`
Simulates the full BTD model across a range of injection densities. Produces:
- TRPL decay curves
- Differential lifetime and rate constant vs QFLS
- Individual carrier concentration dynamics (free electrons, trapped electrons, holes, available traps)
- Stacked process contribution plots

---

## Installation

### Using Conda (recommended)

```bash
conda env create -f HMC_env.yml
conda activate MCMC_env
```

This installs all required packages including JAX, Diffrax, Equinox, NumPyro, ArviZ, and standard scientific Python libraries.

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `jax` / `jaxlib` | Accelerated numerical computing with JIT compilation |
| `diffrax` | JAX-native ODE solver (Kvaerno5 stiff solver used throughout) |
| `equinox` | JAX neural network / filter utilities |
| `numpyro` | Probabilistic programming and HMC/NUTS sampling |
| `arviz` | Bayesian inference diagnostics and visualisation |
| `numpy`, `scipy` | Standard numerical routines |
| `matplotlib`, `seaborn`, `plotly` | Plotting |

> **Note:** This environment was built on Windows (x64). Linux/macOS users may need to adjust or recreate the environment from the package list.

---

## Usage

```python
from globalfit_functions import *
import jax.numpy as jnp

# Define time axis (log-spaced, 0–100 µs)
time = jnp.logspace(0, jnp.log10(100001), 1000) - 1

# Simulate BTD model TRPL decay
log_signal, n_e, n_t, n_p = TRPL_BTD_Model(
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
