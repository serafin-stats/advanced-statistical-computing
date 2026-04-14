# Advanced Statistical Computing — PM-520
### Coursework in Bayesian inference, JAX, and computational statistics

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![JAX](https://img.shields.io/badge/JAX-accelerated-orange)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

---

## Overview

This repository contains my independent lab work and a companion project summary
from **PM-520: Advanced Statistical Computing**, a graduate-level course at USC's
Biostatistics graduate program covering the theory and implementation of modern computational
inference methods.

The course emphasizes that statistical inference at scale requires more than
knowing the right model — it requires understanding numerical stability,
algorithmic efficiency, and the tradeoffs between exact and approximate methods.
All implementations use **Python and JAX**, with a focus on differentiable,
JIT-compiled statistical computation.

This work directly motivated an independent follow-up project applying these
methods to clinical trial simulation:
[bayesian-adaptive-trial-igf1](https://github.com/serafin-stats/bayesian-adaptive-trial-igf1)

---

## Course Topics & Lab Notebooks

Each notebook below represents independent work completed as part of the course
lab sequence. Topics build progressively from numerical foundations through
full Bayesian inference pipelines.

| Notebook | Topic | Key concepts |
|----------|-------|--------------|
| `Lab_0_NumericsCheatSheet.ipynb` | Numerical computing foundations | Float precision, logsumexp trick, numerical stability |
| `Lab_1_Introduction.ipynb` | Python + JAX introduction | JIT compilation, automatic differentiation |
| `Lab_2_LinearSolve.ipynb` | Linear systems | Numerical solvers, condition numbers |
| `Lab_3_Optimization_PtI.ipynb` | Optimization I | Gradient descent, convergence |
| `Lab_4_Optimization_PtII.ipynb` | Optimization II | Natural gradient descent, second-order methods |
| `Lab_5_ExpFam_Divergences.ipynb` | Exponential families & divergences | Sufficient statistics, KL divergence, exponential family GLMs |
| `Lab_6_Divergences.ipynb` | Statistical divergences | f-divergences, variational representations |
| `Lab_7_Intro_Bayesian_Inference.ipynb` | Bayesian inference | Priors, posteriors, conjugate models |
| `Lab_8_Variational_Inference_PtI.ipynb` | Variational inference I | ELBO, mean-field approximation |
| `Lab_9_Variational_Inference_PtII.ipynb` | Variational inference II | Stochastic VI, reparameterization trick |
| `Lab_10_Variational_Inference_PtIII.ipynb` | Variational inference III | Advanced VI, normalizing flows |
| `Lab_11_MCMC_BlackJAX.ipynb` | MCMC with BlackJAX | NUTS sampler, HMC, convergence diagnostics |
| `Lab_12_Gibbs_Sampling.ipynb` | Gibbs sampling | Conditional distributions, mixing |
| `Lab_13_HMC.ipynb` | Hamiltonian Monte Carlo | Leapfrog integrator, energy conservation |

---

## Final Project Summary

### From OLS to MCMC: Predicting IGF-1 in the UK Biobank

**Collaborator:** Jessica George  
**Course:** PM-520, May 2025

#### Background

Insulin-like growth factor 1 (IGF-1) is a hepatokine implicated in colorectal
cancer risk and progression. This project compared three inference methods for
predicting log-transformed IGF-1 levels in 1,000 randomly sampled UK Biobank
participants using age, BMI, diabetic status, sex, and five polygenic risk scores
as predictors.

#### Methods

Three approaches were compared:

**Ordinary Least Squares (OLS)** — baseline frequentist regression providing
point estimates and confidence intervals. Implemented in R using standard
`lm()` regression with `gtsummary` for reporting.

**MCMC (No-U-Turn Sampler)** — Bayesian inference via NUTS with uninformative
priors, 10,000 samples, 500-iteration burn-in. Implemented in Python using JAX.
Step size 1e-4, inverse mass matrix 0.5, 60 integration steps.

**Adaptive MCMC** — NUTS with automatic windowed adaptation of tuning parameters
during warm-up. Same prior specification as standard MCMC.

#### Key Results

| Method | Runtime | MAE vs OLS | Acceptance rate |
|--------|---------|------------|-----------------|
| OLS | < 1 sec | — | — |
| MCMC | ~103 sec | 0.022 | ~98% |
| Adaptive MCMC | ~15 sec | 0.001 | ~94% |

- Adaptive MCMC ran **7× faster** than standard MCMC
- Adaptive MCMC achieved **22× lower MAE** relative to OLS estimates
- Standard MCMC showed poor mixing for 4 of 5 polygenic risk score parameters;
  adaptive MCMC resolved convergence for all parameters during burn-in
- OLS multivariate results: older age (β = −0.01), higher BMI (β = −0.01),
  diabetes (β = −0.09), and female sex (β = −0.04) all associated with
  lower log-IGF-1 (all p < 0.005)

#### Takeaway

Adaptive MCMC automates hyperparameter tuning during warm-up, eliminating the
need for manual step size and mass matrix specification while improving both
runtime efficiency and posterior accuracy. For large biobank datasets where
manual tuning is impractical, adaptive methods offer a compelling default choice.

> **Note:** The Python simulation code for this project was developed
> collaboratively. This repository contains my independent lab work.
> The full methodology is documented in the project write-up linked below.

---

## How This Work Connects to the Portfolio

This course established the computational foundation for subsequent independent
work. The progression looks like:

```
PM-520 Labs                    →    Final Project           →    Independent Extension
─────────────────────────────       ──────────────────────       ──────────────────────────────
Bayesian inference (Lab 7)          OLS vs MCMC vs            Bayesian adaptive trial
Variational inference (8-10)        Adaptive MCMC             simulation calibrated from
MCMC & HMC (11, 13)                 on UK Biobank IGF-1       UK Biobank parameters
JAX implementation (1-4)            prediction                (bayesian-adaptive-trial-igf1)
```

The key intellectual step from coursework to the independent project was
embedding Bayesian updating inside a *decision loop* — using each posterior
not just as a summary of evidence but as an actionable input to a clinical
trial stop/continue rule.

---

## Repository Structure

```
.
├── README.md
├── Lab_0_NumericsCheatSheet.ipynb
├── Lab_1_Introduction.ipynb
├── Lab_2_LinearSolve.ipynb
├── Lab_3_Optimization_PtI.ipynb
├── Lab_4_Optimization_PtII.ipynb
├── Lab_5_ExpFam_Divergences.ipynb
├── Lab_6_Divergences.ipynb
├── Lab_7_Intro_Bayesian_Inference.ipynb
├── Lab_8_Variational_Inference_PtI.ipynb
├── Lab_9_Variational_Inference_PtII.ipynb
├── Lab_10_Variational_Inference_PtIII.ipynb
├── Lab_11_MCMC_BlackJAX.ipynb
├── Lab_12_Gibbs_Sampling.ipynb
└── Lab_13_HMC.ipynb
```

---

## Environment Setup

```bash
conda create -n pm520 python=3.11 -y
conda activate pm520
pip install numpy pandas matplotlib scipy jax jupyter blackjax
```

---

## Context

This course is part of the USC Graduate Biostatistics Program and is designed for
second-year and beyond students interested in designing and implementing
computational inferential tools for research. Topics covered include:

- Numerical stability and the logsumexp trick
- Automatic differentiation and JIT compilation via JAX
- Optimization: gradient descent, natural gradient descent
- Exponential families and statistical divergences
- Variational inference and the evidence lower bound (ELBO)
- Bayesian inference: conjugate models through full MCMC
- Hamiltonian Monte Carlo and Gibbs sampling

---

## Related Projects

| Project | Description | Language |
|---------|-------------|----------|
| [bayesian-adaptive-trial-igf1](https://github.com/serafin-stats/bayesian-adaptive-trial-igf1) | Bayesian adaptive clinical trial simulation, IGF-1/CRC | Python + JAX |
| [breast-implant-ratio-analysis](https://github.com/serafin-stats/breast-implant-ratio-analysis) | Ordinal & binary regression, post-operative outcomes | R |

---

*Casandra Serafin · MS Biostatistics · [LinkedIn](https://www.linkedin.com/in/casandra-serafin/)*
