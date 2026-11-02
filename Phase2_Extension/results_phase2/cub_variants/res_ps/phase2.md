# Phase 2: EC-FVLC vs FVLC (cub)

Seed 42, paired protocol, views occlude, crop, cutmix, nuisance, lambda5 sweep [0.03, 0.1, 0.3]. Arrows show which direction is better.

## Before vs after

Selected EC-FVLC: **ecfvlc_l0.1** (highest held-out-CLIP evidence correlation among lambda5 values within 1 accuracy point of FVLC).

| | FVLC (before) | EC-FVLC (after) |
|---|---|---|
| accuracy | 73.39 | 73.37 |
| worst-case TCPC r=0.1 | 0.204 | 0.272 |
| paired IP TCPC | 0.0070 | 0.0021 |
| evidence corr train clip | 0.245 | 0.444 |
| evidence corr oracle clip | 0.188 | 0.319 |
| nuisance tcpc | 0.165 | 0.119 |
| locality | 0.032 | 0.018 |
| within-species attribute AUROC | 0.529 | 0.539 |

## Accuracy, stability and decision relevance

| Model | Acc ↑ | worst-case TCPC r=0.01 ↓ | worst-case TCPC r=0.05 ↓ | worst-case TCPC r=0.1 ↓ | top-k overlap r=0.1 ↑ | paired IP TCPC ↓ | top-k with non-zero weight ↑ | activations vs contributions ↑ |
|---|---|---|---|---|---|---|---|---|
| constant | 0.52 | 0.000 | 0.000 | 0.000 | 1.000 | – | 0.200 | 0.200 |
| lfcbm | 73.71 | 0.036 | 0.182 | 0.365 | 0.739 | – | 0.530 | 0.373 |
| fvlc | 73.39 | 0.020 | 0.102 | 0.204 | 0.797 | 0.007 | 0.510 | 0.353 |
| ecfvlc_l0.03 | 74.39 | 0.027 | 0.137 | 0.273 | 0.762 | 0.004 | 0.562 | 0.414 |
| ecfvlc_l0.1 | 73.37 | 0.027 | 0.136 | 0.272 | 0.747 | 0.002 | 0.563 | 0.430 |
| ecfvlc_l0.3 | 71.90 | 0.028 | 0.142 | 0.283 | 0.724 | 0.001 | 0.521 | 0.411 |

## Sensitivity to label-free edits (held-out test edits)

| Model | evidence corr. (train_clip) ↑ | evidence corr. (oracle_clip) ↑ | nuisance-edit TCPC ↓ |
|---|---|---|---|
| constant | 0.000 | 0.000 | 0.000 |
| lfcbm | 0.368 | 0.278 | 0.171 |
| fvlc | 0.245 | 0.188 | 0.165 |
| ecfvlc_l0.03 | 0.404 | 0.294 | 0.141 |
| ecfvlc_l0.1 | 0.444 | 0.319 | 0.119 |
| ecfvlc_l0.3 | 0.465 | 0.332 | 0.108 |

Evidence corr. = per-concept correlation between the change of the concept score and the change of CLIP's image-text similarity under occlusion / crop / CutMix of test images, averaged over concepts. `oracle_clip` uses a CLIP model never seen in training.

## Sensitivity and grounding on CUB human annotations

| Model | own-part change ↑ | distant-part change ↓ | locality ↑ | drop when concept is on ↑ | attribute AUROC ↑ | within-species AUROC ↑ |
|---|---|---|---|---|---|---|
| constant | 0.000 | 0.000 | 0.000 | – | 0.500 | 0.500 |
| lfcbm | 0.233 | 0.205 | 0.028 | 0.098 | 0.690 | 0.545 |
| fvlc | 0.294 | 0.262 | 0.032 | 0.119 | 0.645 | 0.529 |
| ecfvlc_l0.03 | 0.166 | 0.146 | 0.020 | 0.054 | 0.677 | 0.538 |
| ecfvlc_l0.1 | 0.141 | 0.123 | 0.018 | 0.053 | 0.685 | 0.539 |
| ecfvlc_l0.3 | 0.127 | 0.111 | 0.016 | 0.051 | 0.689 | 0.539 |

Locality over 133 part-named concepts; attribute AUROC over 110 concepts and 8999 (concept, species) pairs. Scores in standardised units. Locality = own-part change minus distant-part change (Raman et al., TMLR 2025).
