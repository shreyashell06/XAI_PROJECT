# Phase 2: EC-FVLC vs FVLC (cub)

Seed 42, paired protocol, views occlude, crop, cutmix, nuisance, lambda5 sweep [0.003, 0.01, 0.03, 0.1, 0.3]. Arrows show which direction is better.

## Before vs after

Selected EC-FVLC: **ecfvlc_l0.1** (highest held-out-CLIP evidence correlation among lambda5 values within 1 accuracy point of FVLC).

| | FVLC (before) | EC-FVLC (after) |
|---|---|---|
| accuracy | 72.94 | 73.01 |
| worst-case TCPC r=0.1 | 0.203 | 0.273 |
| paired IP TCPC | 0.0096 | 0.0027 |
| evidence corr train clip | 0.242 | 0.444 |
| evidence corr oracle clip | 0.186 | 0.319 |
| nuisance tcpc | 0.164 | 0.118 |
| locality | 0.033 | 0.018 |
| within-species attribute AUROC | 0.529 | 0.539 |

## Accuracy, stability and decision relevance

| Model | Acc ↑ | worst-case TCPC r=0.01 ↓ | worst-case TCPC r=0.05 ↓ | worst-case TCPC r=0.1 ↓ | top-k overlap r=0.1 ↑ | paired IP TCPC ↓ | top-k with non-zero weight ↑ | activations vs contributions ↑ |
|---|---|---|---|---|---|---|---|---|
| constant | 0.52 | 0.000 | 0.000 | 0.000 | 1.000 | – | 0.200 | 0.200 |
| lfcbm | 73.71 | 0.036 | 0.182 | 0.365 | 0.739 | – | 0.530 | 0.373 |
| fvlc | 72.94 | 0.020 | 0.102 | 0.203 | 0.794 | 0.010 | 0.506 | 0.350 |
| fvlc_views | 73.96 | 0.024 | 0.119 | 0.237 | 0.791 | 0.018 | 0.533 | 0.379 |
| ecfvlc_l0.003 | 73.66 | 0.025 | 0.124 | 0.248 | 0.782 | 0.010 | 0.523 | 0.371 |
| ecfvlc_l0.01 | 73.97 | 0.028 | 0.138 | 0.276 | 0.768 | 0.008 | 0.544 | 0.393 |
| ecfvlc_l0.03 | 73.96 | 0.028 | 0.138 | 0.277 | 0.754 | 0.005 | 0.560 | 0.414 |
| ecfvlc_l0.1 | 73.01 | 0.027 | 0.136 | 0.273 | 0.742 | 0.003 | 0.561 | 0.428 |
| ecfvlc_l0.3 | 71.64 | 0.028 | 0.142 | 0.283 | 0.721 | 0.001 | 0.520 | 0.410 |

## Sensitivity to label-free edits (held-out test edits)

| Model | evidence corr. (train_clip) ↑ | evidence corr. (oracle_clip) ↑ | nuisance-edit TCPC ↓ |
|---|---|---|---|
| constant | 0.000 | 0.000 | 0.000 |
| lfcbm | 0.368 | 0.278 | 0.171 |
| fvlc | 0.242 | 0.186 | 0.164 |
| fvlc_views | 0.338 | 0.256 | 0.173 |
| ecfvlc_l0.003 | 0.294 | 0.222 | 0.174 |
| ecfvlc_l0.01 | 0.351 | 0.260 | 0.165 |
| ecfvlc_l0.03 | 0.404 | 0.294 | 0.140 |
| ecfvlc_l0.1 | 0.444 | 0.319 | 0.118 |
| ecfvlc_l0.3 | 0.465 | 0.332 | 0.108 |

Evidence corr. = per-concept correlation between the change of the concept score and the change of CLIP's image-text similarity under occlusion / crop / CutMix of test images, averaged over concepts. `oracle_clip` uses a CLIP model never seen in training.

## Sensitivity and grounding on CUB human annotations

| Model | own-part change ↑ | distant-part change ↓ | locality ↑ | drop when concept is on ↑ | attribute AUROC ↑ | within-species AUROC ↑ |
|---|---|---|---|---|---|---|
| constant | 0.000 | 0.000 | 0.000 | – | 0.500 | 0.500 |
| lfcbm | 0.233 | 0.205 | 0.028 | 0.098 | 0.690 | 0.545 |
| fvlc | 0.301 | 0.268 | 0.033 | 0.122 | 0.645 | 0.529 |
| fvlc_views | 0.231 | 0.204 | 0.027 | 0.086 | 0.685 | 0.543 |
| ecfvlc_l0.003 | 0.249 | 0.222 | 0.027 | 0.092 | 0.655 | 0.531 |
| ecfvlc_l0.01 | 0.202 | 0.179 | 0.023 | 0.067 | 0.667 | 0.535 |
| ecfvlc_l0.03 | 0.166 | 0.146 | 0.020 | 0.052 | 0.677 | 0.537 |
| ecfvlc_l0.1 | 0.140 | 0.123 | 0.018 | 0.051 | 0.685 | 0.539 |
| ecfvlc_l0.3 | 0.127 | 0.111 | 0.016 | 0.050 | 0.689 | 0.539 |

Locality over 133 part-named concepts; attribute AUROC over 110 concepts and 8999 (concept, species) pairs. Scores in standardised units. Locality = own-part change minus distant-part change (Raman et al., TMLR 2025).
