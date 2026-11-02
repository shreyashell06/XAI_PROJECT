# Phase 2: EC-FVLC vs FVLC (cub)

Seed 42, paired protocol, views occlude, crop, cutmix, nuisance, lambda5 sweep [0.03, 0.1, 0.3]. Arrows show which direction is better.

## Before vs after

Selected EC-FVLC: **ecfvlc_l0.1** (highest held-out-CLIP evidence correlation among lambda5 values within 1 accuracy point of FVLC).

| | FVLC (before) | EC-FVLC (after) |
|---|---|---|
| accuracy | 74.28 | 73.39 |
| worst-case TCPC r=0.1 | 0.193 | 0.223 |
| paired IP TCPC | 0.0092 | 0.0023 |
| evidence corr train clip | 0.262 | 0.446 |
| evidence corr oracle clip | 0.201 | 0.321 |
| nuisance tcpc | 0.163 | 0.110 |
| locality | 0.026 | 0.017 |
| within-species attribute AUROC | 0.530 | 0.539 |

## Accuracy, stability and decision relevance

| Model | Acc ↑ | worst-case TCPC r=0.01 ↓ | worst-case TCPC r=0.05 ↓ | worst-case TCPC r=0.1 ↓ | top-k overlap r=0.1 ↑ | paired IP TCPC ↓ | top-k with non-zero weight ↑ | activations vs contributions ↑ |
|---|---|---|---|---|---|---|---|---|
| constant | 0.52 | 0.000 | 0.000 | 0.000 | 1.000 | – | 0.200 | 0.200 |
| lfcbm | 73.71 | 0.036 | 0.182 | 0.365 | 0.739 | – | 0.530 | 0.373 |
| fvlc | 74.28 | 0.019 | 0.096 | 0.193 | 0.828 | 0.009 | 0.527 | 0.373 |
| ecfvlc_l0.03 | 74.20 | 0.021 | 0.104 | 0.208 | 0.814 | 0.005 | 0.563 | 0.415 |
| ecfvlc_l0.1 | 73.39 | 0.022 | 0.111 | 0.223 | 0.794 | 0.002 | 0.561 | 0.429 |
| ecfvlc_l0.3 | 71.68 | 0.025 | 0.126 | 0.252 | 0.758 | 0.001 | 0.518 | 0.408 |

## Sensitivity to label-free edits (held-out test edits)

| Model | evidence corr. (train_clip) ↑ | evidence corr. (oracle_clip) ↑ | nuisance-edit TCPC ↓ |
|---|---|---|---|
| constant | 0.000 | 0.000 | 0.000 |
| lfcbm | 0.368 | 0.278 | 0.171 |
| fvlc | 0.262 | 0.201 | 0.163 |
| ecfvlc_l0.03 | 0.407 | 0.297 | 0.125 |
| ecfvlc_l0.1 | 0.446 | 0.321 | 0.110 |
| ecfvlc_l0.3 | 0.467 | 0.333 | 0.104 |

Evidence corr. = per-concept correlation between the change of the concept score and the change of CLIP's image-text similarity under occlusion / crop / CutMix of test images, averaged over concepts. `oracle_clip` uses a CLIP model never seen in training.

## Sensitivity and grounding on CUB human annotations

| Model | own-part change ↑ | distant-part change ↓ | locality ↑ | drop when concept is on ↑ | attribute AUROC ↑ | within-species AUROC ↑ |
|---|---|---|---|---|---|---|
| constant | 0.000 | 0.000 | 0.000 | – | 0.500 | 0.500 |
| lfcbm | 0.233 | 0.205 | 0.028 | 0.098 | 0.690 | 0.545 |
| fvlc | 0.232 | 0.206 | 0.026 | 0.081 | 0.650 | 0.530 |
| ecfvlc_l0.03 | 0.153 | 0.134 | 0.019 | 0.046 | 0.678 | 0.538 |
| ecfvlc_l0.1 | 0.135 | 0.117 | 0.017 | 0.049 | 0.686 | 0.539 |
| ecfvlc_l0.3 | 0.124 | 0.108 | 0.015 | 0.049 | 0.690 | 0.539 |

Locality over 133 part-named concepts; attribute AUROC over 110 concepts and 8999 (concept, species) pairs. Scores in standardised units. Locality = own-part change minus distant-part change (Raman et al., TMLR 2025).
