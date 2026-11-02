# Phase 2: EC-FVLC vs FVLC (cifar100)

Seed 42, paired protocol, views occlude, crop, cutmix, nuisance, lambda5 sweep [0.01, 0.03, 0.1, 0.3]. Arrows show which direction is better.

## Before vs after

Selected EC-FVLC: **ecfvlc_l0.3** (highest held-out-CLIP evidence correlation among lambda5 values within 1 accuracy point of FVLC).

| | FVLC (before) | EC-FVLC (after) |
|---|---|---|
| accuracy | 64.91 | 64.88 |
| worst-case TCPC r=0.1 | 0.127 | 0.083 |
| paired IP TCPC | 0.0762 | 0.0086 |
| evidence corr train clip | 0.397 | 0.579 |
| evidence corr oracle clip | 0.185 | 0.310 |
| nuisance tcpc | 0.770 | 0.249 |

## Accuracy, stability and decision relevance

| Model | Acc ↑ | worst-case TCPC r=0.01 ↓ | worst-case TCPC r=0.05 ↓ | worst-case TCPC r=0.1 ↓ | top-k overlap r=0.1 ↑ | paired IP TCPC ↓ | top-k with non-zero weight ↑ | activations vs contributions ↑ |
|---|---|---|---|---|---|---|---|---|
| constant | 1.00 | 0.000 | 0.000 | 0.000 | 1.000 | – | 0.000 | 0.000 |
| lfcbm | 65.09 | 0.014 | 0.069 | 0.137 | 0.783 | – | 0.351 | 0.312 |
| fvlc | 64.91 | 0.013 | 0.064 | 0.127 | 0.791 | 0.076 | 0.350 | 0.310 |
| fvlc_views | 65.15 | 0.013 | 0.063 | 0.126 | 0.787 | 0.072 | 0.351 | 0.311 |
| ecfvlc_l0.01 | 65.28 | 0.011 | 0.053 | 0.106 | 0.772 | 0.028 | 0.359 | 0.317 |
| ecfvlc_l0.03 | 65.54 | 0.009 | 0.045 | 0.090 | 0.759 | 0.016 | 0.371 | 0.326 |
| ecfvlc_l0.1 | 65.50 | 0.008 | 0.040 | 0.080 | 0.740 | 0.010 | 0.382 | 0.336 |
| ecfvlc_l0.3 | 64.88 | 0.008 | 0.042 | 0.083 | 0.719 | 0.009 | 0.379 | 0.333 |

## Sensitivity to label-free edits (held-out test edits)

| Model | evidence corr. (train_clip) ↑ | evidence corr. (oracle_clip) ↑ | nuisance-edit TCPC ↓ |
|---|---|---|---|
| constant | 0.000 | 0.000 | 0.000 |
| lfcbm | 0.448 | 0.218 | 0.770 |
| fvlc | 0.397 | 0.185 | 0.770 |
| fvlc_views | 0.423 | 0.197 | 0.744 |
| ecfvlc_l0.01 | 0.500 | 0.257 | 0.480 |
| ecfvlc_l0.03 | 0.532 | 0.278 | 0.368 |
| ecfvlc_l0.1 | 0.562 | 0.298 | 0.286 |
| ecfvlc_l0.3 | 0.579 | 0.310 | 0.249 |

Evidence corr. = per-concept correlation between the change of the concept score and the change of CLIP's image-text similarity under occlusion / crop / CutMix of test images, averaged over concepts. `oracle_clip` uses a CLIP model never seen in training.
