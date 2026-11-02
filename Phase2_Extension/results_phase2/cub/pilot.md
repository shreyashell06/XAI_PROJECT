# Phase 2 pilot: two-sided faithfulness

Dataset **cub**, seed 42, paired protocol (FVLC = Algorithm 1, Gaussian init). Arrows show which direction is better.

## Accuracy, stability and decision relevance

| Model | Acc ↑ | worst-case TCPC r=0.01 ↓ | worst-case TCPC r=0.05 ↓ | worst-case TCPC r=0.1 ↓ | top-k overlap r=0.1 ↑ | top-k with non-zero weight ↑ | activations vs contributions ↑ |
|---|---|---|---|---|---|---|---|
| constant | 0.52 | 0.000 | 0.000 | 0.000 | 1.000 | 0.200 | 0.200 |
| lfcbm | 73.71 | 0.036 | 0.182 | 0.365 | 0.739 | 0.530 | 0.373 |
| fvlc | 72.94 | 0.020 | 0.102 | 0.203 | 0.794 | 0.506 | 0.350 |

## Sensitivity and grounding on CUB human annotations

| Model | own-part change ↑ | distant-part change ↓ | locality ↑ | drop when concept is on ↑ | attribute AUROC ↑ | within-species AUROC ↑ |
|---|---|---|---|---|---|---|
| constant | 0.000 | 0.000 | 0.000 | – | 0.500 | 0.500 |
| lfcbm | 0.233 | 0.205 | 0.028 | 0.098 | 0.690 | 0.545 |
| fvlc | 0.301 | 0.268 | 0.033 | 0.122 | 0.645 | 0.529 |

Locality over 133 part-named concepts; attribute AUROC over 110 concepts and 8999 (concept, species) pairs. Scores in standardised units. Locality = own-part change minus distant-part change (Raman et al., TMLR 2025).

## Go / no-go

FVLC minus LF-CBM: locality +0.005, within-species AUROC -0.016. Share of FVLC's top-k concepts with a non-zero weight for the predicted class: 0.51. Constant explainer worst-case TCPC: 0.000, vs 0.203 for FVLC (it 'wins' on stability). **GO**: build EC-FVLC.
