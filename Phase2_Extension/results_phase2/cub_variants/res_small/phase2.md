# Phase 2: EC-FVLC vs FVLC (cub)

Seed 42, paired protocol, views occlude, occlude_small, crop, cutmix, nuisance, lambda5 sweep [0.03, 0.1, 0.3]. Arrows show which direction is better.

## Before vs after

Selected EC-FVLC: **ecfvlc_l0.03** (highest held-out-CLIP evidence correlation among lambda5 values within 1 accuracy point of FVLC).

| | FVLC (before) | EC-FVLC (after) |
|---|---|---|
| accuracy | 74.28 | 73.99 |
| worst-case TCPC r=0.1 | 0.193 | 0.205 |
| paired IP TCPC | 0.0092 | 0.0053 |
| evidence corr train clip | 0.243 | 0.370 |
| evidence corr oracle clip | 0.183 | 0.268 |
| nuisance tcpc | 0.163 | 0.126 |
| locality | 0.026 | 0.020 |
| within-species attribute AUROC | 0.530 | 0.537 |

## Accuracy, stability and decision relevance

| Model | Acc ↑ | worst-case TCPC r=0.01 ↓ | worst-case TCPC r=0.05 ↓ | worst-case TCPC r=0.1 ↓ | top-k overlap r=0.1 ↑ | paired IP TCPC ↓ | top-k with non-zero weight ↑ | activations vs contributions ↑ |
|---|---|---|---|---|---|---|---|---|
| constant | 0.52 | 0.000 | 0.000 | 0.000 | 1.000 | – | 0.200 | 0.200 |
| lfcbm | 73.71 | 0.036 | 0.182 | 0.365 | 0.739 | – | 0.530 | 0.373 |
| fvlc | 74.28 | 0.019 | 0.096 | 0.193 | 0.828 | 0.009 | 0.527 | 0.373 |
| ecfvlc_l0.03 | 73.99 | 0.021 | 0.103 | 0.205 | 0.813 | 0.005 | 0.561 | 0.411 |
| ecfvlc_l0.1 | 73.26 | 0.022 | 0.109 | 0.217 | 0.794 | 0.002 | 0.559 | 0.425 |
| ecfvlc_l0.3 | 71.68 | 0.024 | 0.122 | 0.244 | 0.761 | 0.001 | 0.519 | 0.408 |

## Sensitivity to label-free edits (held-out test edits)

| Model | evidence corr. (train_clip) ↑ | evidence corr. (oracle_clip) ↑ | nuisance-edit TCPC ↓ |
|---|---|---|---|
| constant | 0.000 | 0.000 | 0.000 |
| lfcbm | 0.344 | 0.255 | 0.171 |
| fvlc | 0.243 | 0.183 | 0.163 |
| ecfvlc_l0.03 | 0.370 | 0.268 | 0.126 |
| ecfvlc_l0.1 | 0.408 | 0.292 | 0.110 |
| ecfvlc_l0.3 | 0.429 | 0.304 | 0.102 |

Evidence corr. = per-concept correlation between the change of the concept score and the change of CLIP's image-text similarity under occlusion / crop / CutMix of test images, averaged over concepts. `oracle_clip` uses a CLIP model never seen in training.

## Sensitivity and grounding on CUB human annotations

| Model | own-part change ↑ | distant-part change ↓ | locality ↑ | drop when concept is on ↑ | attribute AUROC ↑ | within-species AUROC ↑ |
|---|---|---|---|---|---|---|
| constant | 0.000 | 0.000 | 0.000 | – | 0.500 | 0.500 |
| lfcbm | 0.233 | 0.205 | 0.028 | 0.098 | 0.690 | 0.545 |
| fvlc | 0.232 | 0.206 | 0.026 | 0.081 | 0.650 | 0.530 |
| ecfvlc_l0.03 | 0.158 | 0.138 | 0.020 | 0.038 | 0.675 | 0.537 |
| ecfvlc_l0.1 | 0.139 | 0.121 | 0.018 | 0.038 | 0.682 | 0.538 |
| ecfvlc_l0.3 | 0.128 | 0.112 | 0.017 | 0.037 | 0.687 | 0.538 |

Locality over 133 part-named concepts; attribute AUROC over 110 concepts and 8999 (concept, species) pairs. Scores in standardised units. Locality = own-part change minus distant-part change (Raman et al., TMLR 2025).
