# Phase 2: EC-FVLC vs FVLC (cifar10)

Seed 42, paired protocol, views occlude, crop, cutmix, nuisance, lambda5 sweep [0.01, 0.03, 0.1, 0.3]. Arrows show which direction is better.

## Before vs after

Selected EC-FVLC: **ecfvlc_l0.3** (highest held-out-CLIP evidence correlation among lambda5 values within 1 accuracy point of FVLC).

| | FVLC (before) | EC-FVLC (after) |
|---|---|---|
| accuracy | 86.50 | 86.39 |
| worst-case TCPC r=0.1 | 0.188 | 0.151 |
| paired IP TCPC | 0.0293 | 0.0119 |
| evidence corr train clip | 0.475 | 0.601 |
| evidence corr oracle clip | 0.237 | 0.331 |
| nuisance tcpc | 0.747 | 0.366 |

## Accuracy, stability and decision relevance

| Model | Acc ↑ | worst-case TCPC r=0.01 ↓ | worst-case TCPC r=0.05 ↓ | worst-case TCPC r=0.1 ↓ | top-k overlap r=0.1 ↑ | paired IP TCPC ↓ | top-k with non-zero weight ↑ | activations vs contributions ↑ |
|---|---|---|---|---|---|---|---|---|
| constant | 10.00 | 0.000 | 0.000 | 0.000 | 1.000 | – | 0.100 | 0.100 |
| lfcbm | 86.51 | 0.018 | 0.091 | 0.182 | 0.813 | – | 0.561 | 0.442 |
| fvlc | 86.50 | 0.019 | 0.094 | 0.188 | 0.818 | 0.029 | 0.560 | 0.437 |
| fvlc_views | 86.53 | 0.018 | 0.092 | 0.183 | 0.820 | 0.025 | 0.560 | 0.439 |
| ecfvlc_l0.01 | 86.59 | 0.018 | 0.089 | 0.179 | 0.797 | 0.019 | 0.563 | 0.450 |
| ecfvlc_l0.03 | 86.82 | 0.016 | 0.080 | 0.161 | 0.778 | 0.015 | 0.569 | 0.457 |
| ecfvlc_l0.1 | 86.71 | 0.015 | 0.075 | 0.150 | 0.749 | 0.012 | 0.573 | 0.463 |
| ecfvlc_l0.3 | 86.39 | 0.015 | 0.076 | 0.151 | 0.723 | 0.012 | 0.564 | 0.452 |

## Sensitivity to label-free edits (held-out test edits)

| Model | evidence corr. (train_clip) ↑ | evidence corr. (oracle_clip) ↑ | nuisance-edit TCPC ↓ |
|---|---|---|---|
| constant | 0.000 | 0.000 | 0.000 |
| lfcbm | 0.495 | 0.251 | 0.744 |
| fvlc | 0.475 | 0.237 | 0.747 |
| fvlc_views | 0.484 | 0.244 | 0.733 |
| ecfvlc_l0.01 | 0.545 | 0.294 | 0.540 |
| ecfvlc_l0.03 | 0.572 | 0.311 | 0.447 |
| ecfvlc_l0.1 | 0.592 | 0.324 | 0.389 |
| ecfvlc_l0.3 | 0.601 | 0.331 | 0.366 |

Evidence corr. = per-concept correlation between the change of the concept score and the change of CLIP's image-text similarity under occlusion / crop / CutMix of test images, averaged over concepts. `oracle_clip` uses a CLIP model never seen in training.
