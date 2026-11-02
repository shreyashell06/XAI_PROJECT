# EC-FVLC variants on cub

| Run | Settings | Model | Acc ↑ | Evid. corr. held-out ↑ | Nuisance TCPC ↓ | Paired IP TCPC ↓ | Worst-case TCPC r=0.1 ↓ | Top-k overlap attack ↑ | Top-k non-zero weight ↑ | Attr. AUROC ↑ | Within-species AUROC ↑ | Locality ↑ | Own/distant ratio ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | λ4=?, ?, r=?, views=occlude+crop+cutmix+nuisance | lfcbm | 73.71 | 0.278 | 0.171 | – | 0.365 | 0.739 | 0.530 | 0.690 | 0.545 | 0.028 | 1.13 |
| current | λ4=?, ?, r=?, views=occlude+crop+cutmix+nuisance | fvlc | 72.94 | 0.186 | 0.164 | 0.0096 | 0.203 | 0.794 | 0.506 | 0.645 | 0.529 | 0.033 | 1.12 |
| current | λ4=?, ?, r=?, views=occlude+crop+cutmix+nuisance | ecfvlc_l0.003 | 73.66 | 0.222 | 0.174 | 0.0099 | 0.248 | 0.782 | 0.523 | 0.655 | 0.531 | 0.027 | 1.12 |
| current | λ4=?, ?, r=?, views=occlude+crop+cutmix+nuisance | ecfvlc_l0.01 | 73.97 | 0.260 | 0.165 | 0.0085 | 0.276 | 0.768 | 0.544 | 0.667 | 0.535 | 0.023 | 1.13 |
| current | λ4=?, ?, r=?, views=occlude+crop+cutmix+nuisance | ecfvlc_l0.03 | 73.96 | 0.294 | 0.140 | 0.0054 | 0.277 | 0.754 | 0.560 | 0.677 | 0.537 | 0.020 | 1.14 |
| current | λ4=?, ?, r=?, views=occlude+crop+cutmix+nuisance | ecfvlc_l0.1 | 73.01 | 0.319 | 0.118 | 0.0027 | 0.273 | 0.742 | 0.561 | 0.685 | 0.539 | 0.018 | 1.14 |
| current | λ4=?, ?, r=?, views=occlude+crop+cutmix+nuisance | ecfvlc_l0.3 | 71.64 | 0.332 | 0.108 | 0.0011 | 0.283 | 0.721 | 0.520 | 0.689 | 0.539 | 0.016 | 1.14 |
| per-sample | λ4=0.01, per_sample, r=0.05, views=occlude+crop+cutmix+nuisance | lfcbm | 73.71 | 0.278 | 0.171 | – | 0.365 | 0.739 | 0.530 | 0.690 | 0.545 | 0.028 | 1.13 |
| per-sample | λ4=0.01, per_sample, r=0.05, views=occlude+crop+cutmix+nuisance | fvlc | 73.39 | 0.188 | 0.165 | 0.0070 | 0.204 | 0.797 | 0.510 | 0.645 | 0.529 | 0.032 | 1.12 |
| per-sample | λ4=0.01, per_sample, r=0.05, views=occlude+crop+cutmix+nuisance | ecfvlc_l0.03 | 74.39 | 0.294 | 0.141 | 0.0045 | 0.273 | 0.762 | 0.562 | 0.677 | 0.538 | 0.020 | 1.14 |
| per-sample | λ4=0.01, per_sample, r=0.05, views=occlude+crop+cutmix+nuisance | ecfvlc_l0.1 | 73.37 | 0.319 | 0.119 | 0.0021 | 0.272 | 0.747 | 0.563 | 0.685 | 0.539 | 0.018 | 1.14 |
| per-sample | λ4=0.01, per_sample, r=0.05, views=occlude+crop+cutmix+nuisance | ecfvlc_l0.3 | 71.90 | 0.332 | 0.108 | 0.0008 | 0.283 | 0.724 | 0.521 | 0.689 | 0.539 | 0.016 | 1.14 |
| per-sample+L4 | λ4=0.05, per_sample, r=0.1, views=occlude+crop+cutmix+nuisance | lfcbm | 73.71 | 0.278 | 0.171 | – | 0.365 | 0.739 | 0.530 | 0.690 | 0.545 | 0.028 | 1.13 |
| per-sample+L4 | λ4=0.05, per_sample, r=0.1, views=occlude+crop+cutmix+nuisance | fvlc | 74.28 | 0.201 | 0.163 | 0.0092 | 0.193 | 0.828 | 0.527 | 0.650 | 0.530 | 0.026 | 1.13 |
| per-sample+L4 | λ4=0.05, per_sample, r=0.1, views=occlude+crop+cutmix+nuisance | ecfvlc_l0.03 | 74.20 | 0.297 | 0.125 | 0.0047 | 0.208 | 0.814 | 0.563 | 0.678 | 0.538 | 0.019 | 1.14 |
| per-sample+L4 | λ4=0.05, per_sample, r=0.1, views=occlude+crop+cutmix+nuisance | ecfvlc_l0.1 | 73.39 | 0.321 | 0.110 | 0.0023 | 0.223 | 0.794 | 0.561 | 0.686 | 0.539 | 0.017 | 1.15 |
| per-sample+L4 | λ4=0.05, per_sample, r=0.1, views=occlude+crop+cutmix+nuisance | ecfvlc_l0.3 | 71.68 | 0.333 | 0.104 | 0.0009 | 0.252 | 0.758 | 0.518 | 0.690 | 0.539 | 0.015 | 1.14 |
| small-occlusion | λ4=0.05, per_sample, r=0.1, views=occlude+occlude_small+crop+cutmix+nuisance | lfcbm | 73.71 | 0.255 | 0.171 | – | 0.365 | 0.739 | 0.530 | 0.690 | 0.545 | 0.028 | 1.13 |
| small-occlusion | λ4=0.05, per_sample, r=0.1, views=occlude+occlude_small+crop+cutmix+nuisance | fvlc | 74.28 | 0.183 | 0.163 | 0.0092 | 0.193 | 0.828 | 0.527 | 0.650 | 0.530 | 0.026 | 1.13 |
| small-occlusion | λ4=0.05, per_sample, r=0.1, views=occlude+occlude_small+crop+cutmix+nuisance | ecfvlc_l0.03 | 73.99 | 0.268 | 0.126 | 0.0053 | 0.205 | 0.813 | 0.561 | 0.675 | 0.537 | 0.020 | 1.14 |
| small-occlusion | λ4=0.05, per_sample, r=0.1, views=occlude+occlude_small+crop+cutmix+nuisance | ecfvlc_l0.1 | 73.26 | 0.292 | 0.110 | 0.0025 | 0.217 | 0.794 | 0.559 | 0.682 | 0.538 | 0.018 | 1.15 |
| small-occlusion | λ4=0.05, per_sample, r=0.1, views=occlude+occlude_small+crop+cutmix+nuisance | ecfvlc_l0.3 | 71.68 | 0.304 | 0.102 | 0.0010 | 0.244 | 0.761 | 0.519 | 0.687 | 0.538 | 0.017 | 1.15 |

## Selected models (reference FVLC worst-case TCPC = 0.203)

| Run | Rule | Model | Acc ↑ | Evid. corr. held-out ↑ | Nuisance TCPC ↓ | Paired IP TCPC ↓ | Worst-case TCPC r=0.1 ↓ | Top-k overlap attack ↑ | Top-k non-zero weight ↑ | Attr. AUROC ↑ | Within-species AUROC ↑ | Locality ↑ | Own/distant ratio ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | A | ecfvlc_l0.1 | 73.01 | 0.319 | 0.118 | 0.0027 | 0.273 | 0.742 | 0.561 | 0.685 | 0.539 | 0.018 | 1.14 |
| current | B | none qualifies | | | | | | | | | | | |
| per-sample | A | ecfvlc_l0.1 | 73.37 | 0.319 | 0.119 | 0.0021 | 0.272 | 0.747 | 0.563 | 0.685 | 0.539 | 0.018 | 1.14 |
| per-sample | B | none qualifies | | | | | | | | | | | |
| per-sample+L4 | A | ecfvlc_l0.1 | 73.39 | 0.321 | 0.110 | 0.0023 | 0.223 | 0.794 | 0.561 | 0.686 | 0.539 | 0.017 | 1.15 |
| per-sample+L4 | B | none qualifies | | | | | | | | | | | |
| small-occlusion | A | ecfvlc_l0.03 | 73.99 | 0.268 | 0.126 | 0.0053 | 0.205 | 0.813 | 0.561 | 0.675 | 0.537 | 0.020 | 1.14 |
| small-occlusion | B | none qualifies | | | | | | | | | | | |

Rule A: best held-out evidence correlation within 1 accuracy point of FVLC. Rule B: same, and worst-case TCPC no worse than the reference FVLC.
