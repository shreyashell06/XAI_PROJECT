# Replication results vs. paper

Ours = this repo (seed 42, evaluation protocol: seat). Paper = Lai et al., ICLR 2024. Accuracy in %.


## Headline numbers (averaged over the 7 paper perturbations)

| Dataset | LF-CBM acc ours / paper | FVLC acc ours / paper | base TCPC ours / paper | FVLC TCPC ours / paper | base TOPC ours / paper | FVLC TOPC ours / paper | seed-only TCPC ours |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | 86.51 / 86.32 | 85.31 / 86.48 | 1.02e-01 / 1.59e-01 | 8.62e-02 / 8.49e-03 | 1.57e-01 / 6.94e-02 | 1.38e-01 / 8.70e-03 | – |

## Table 1 – Accuracy

| Method | CIFAR-10 ours | CIFAR-10 paper |
|---|---|---|
| Standard (no interpretability) | – | 88.80 |
| Label-free CBM | 86.51 | 86.32 |
| WP1(5%) – base | – | 86.47 |
| WP1(5%) – FVLC | – | 86.34 |
| WP1(10%) – base | 84.44 | 86.25 |
| WP1(10%) – FVLC | 84.72 | 86.39 |
| WP2 – base | – | 86.41 |
| WP2 – FVLC | – | 86.22 |
| IP – base | 86.25 | 86.62 |
| IP – FVLC | 86.41 | 86.88 |
| WP1(5%)+WP2 – base | – | 86.49 |
| WP1(5%)+WP2 – FVLC | – | 86.43 |
| WP1(10%)+WP2 – base | – | 86.30 |
| WP1(10%)+WP2 – FVLC | – | 86.38 |
| WP1(10%)+WP2+IP – base | 84.55 | 85.96 |
| WP1(10%)+WP2+IP – FVLC | 84.81 | 86.70 |
| Seed only (control) – base | – | – |
| Seed only (control) – FVLC | – | – |

## Table 2 – Stability (TCPC / TOPC, lower is better)


### CIFAR-10

| Perturbation | Method | TCPC ours | TCPC paper | TOPC ours | TOPC paper |
|---|---|---|---|---|---|
| WP1(5%) | base | – | 1.55e-01 | – | 6.32e-02 |
| WP1(5%) | FVLC | – | 1.12e-03 | – | 8.55e-03 |
| WP1(10%) | base | 1.01e-01 | 1.99e-01 | 1.56e-01 | 8.36e-02 |
| WP1(10%) | FVLC | 8.61e-02 | 1.19e-03 | 1.38e-01 | 7.40e-03 |
| WP2 | base | – | 1.53e-01 | – | 4.99e-02 |
| WP2 | FVLC | – | 1.10e-02 | – | 8.72e-03 |
| IP | base | 1.05e-01 | 1.68e-01 | 1.59e-01 | 6.28e-02 |
| IP | FVLC | 8.67e-02 | 8.02e-03 | 1.36e-01 | 8.29e-03 |
| WP1(5%)+WP2 | base | – | 1.85e-01 | – | 3.50e-02 |
| WP1(5%)+WP2 | FVLC | – | 1.20e-02 | – | 7.46e-03 |
| WP1(10%)+WP2 | base | – | 1.17e-01 | – | 8.62e-02 |
| WP1(10%)+WP2 | FVLC | – | 1.18e-02 | – | 9.41e-03 |
| WP1(10%)+WP2+IP | base | 1.01e-01 | 1.36e-01 | 1.55e-01 | 1.05e-01 |
| WP1(10%)+WP2+IP | FVLC | 8.58e-02 | 1.43e-02 | 1.38e-01 | 1.11e-02 |
| Seed only (control) | base | – | – | – | – |
| Seed only (control) | FVLC | – | – | – | – |

Average TCPC/TOPC reduction FVLC vs base: **ours 13.9%**, paper 90.3% (paper claims ~90% on average).


## Table 3 – Ablation (TCPC / TOPC). Paper row order inferred from the PDF layout.


### CIFAR-10

| Perturbation | Losses | TCPC ours | TCPC paper | TOPC ours | TOPC paper |
|---|---|---|---|---|---|
| WP1(10%) | L1 | 9.89e-02 | 1.99e-01 | 1.55e-01 | 8.36e-02 |
| WP1(10%) | L1+L2 | 1.01e-01 | 2.09e-02 | 1.56e-01 | 3.14e-02 |
| WP1(10%) | L1+L3 | 1.01e-01 | 1.80e-02 | 1.59e-01 | 1.79e-02 |
| WP1(10%) | L1+L4 | 7.37e-02 | 4.78e-03 | 1.05e-01 | 3.11e-02 |
| WP1(10%) | L1+L2+L3 | 1.02e-01 | 1.67e-02 | 1.57e-01 | 1.11e-02 |
| WP1(10%) | L1+L2+L4 | 8.34e-02 | 1.65e-03 | 1.34e-01 | 1.01e-02 |
| WP1(10%) | L1+L3+L4 | 7.63e-02 | 1.63e-03 | 1.11e-01 | 1.02e-02 |
| WP1(10%) | L1+L2+L3+L4 | 8.61e-02 | 1.19e-03 | 1.38e-01 | 7.40e-03 |
| IP | L1 | 1.03e-01 | 1.68e-01 | 1.58e-01 | 6.28e-02 |
| IP | L1+L2 | 1.05e-01 | 6.39e-02 | 1.59e-01 | 2.55e-02 |
| IP | L1+L3 | 1.07e-01 | 3.82e-02 | 1.64e-01 | 3.91e-02 |
| IP | L1+L4 | 7.43e-02 | 1.63e-02 | 1.03e-01 | 1.73e-02 |
| IP | L1+L2+L3 | 1.06e-01 | 2.04e-02 | 1.60e-01 | 2.10e-02 |
| IP | L1+L2+L4 | 8.39e-02 | 1.23e-02 | 1.33e-01 | 1.27e-02 |
| IP | L1+L3+L4 | 7.58e-02 | 1.26e-02 | 1.08e-01 | 1.09e-02 |
| IP | L1+L2+L3+L4 | 8.67e-02 | 8.02e-03 | 1.36e-01 | 8.29e-03 |
| WP1(10%)+WP2+IP | L1 | 9.56e-02 | 1.36e-01 | 1.50e-01 | 1.05e-01 |
| WP1(10%)+WP2+IP | L1+L2 | 1.01e-01 | 7.37e-02 | 1.55e-01 | 5.75e-02 |
| WP1(10%)+WP2+IP | L1+L3 | 1.01e-01 | 7.85e-02 | 1.59e-01 | 5.85e-02 |
| WP1(10%)+WP2+IP | L1+L4 | 7.35e-02 | 4.59e-02 | 1.05e-01 | 3.67e-02 |
| WP1(10%)+WP2+IP | L1+L2+L3 | 1.02e-01 | 5.81e-02 | 1.57e-01 | 4.52e-02 |
| WP1(10%)+WP2+IP | L1+L2+L4 | 8.35e-02 | 5.84e-02 | 1.35e-01 | 4.53e-02 |
| WP1(10%)+WP2+IP | L1+L3+L4 | 7.55e-02 | 2.91e-02 | 1.10e-01 | 2.31e-02 |
| WP1(10%)+WP2+IP | L1+L2+L3+L4 | 8.58e-02 | 1.43e-02 | 1.38e-01 | 1.11e-02 |

## Extra diagnostics (ours only)

| Dataset | Perturbation | Method | top-k overlap | KL(y‖y') | pred. agreement | TCPC (global) |
|---|---|---|---|---|---|---|
| CIFAR-10 | WP1(10%) | base | 0.892 | 1.66e-01 | 0.8153 | 9.51e-02 |
| CIFAR-10 | WP1(10%) | fvlc | 0.910 | 1.19e-01 | 0.8469 | 8.16e-02 |
| CIFAR-10 | IP | base | 0.885 | 1.73e-01 | 0.8274 | 9.95e-02 |
| CIFAR-10 | IP | fvlc | 0.909 | 1.16e-01 | 0.8613 | 8.26e-02 |
| CIFAR-10 | WP1(10%)+WP2+IP | base | 0.894 | 1.65e-01 | 0.8146 | 9.47e-02 |
| CIFAR-10 | WP1(10%)+WP2+IP | fvlc | 0.911 | 1.18e-01 | 0.8478 | 8.13e-02 |
