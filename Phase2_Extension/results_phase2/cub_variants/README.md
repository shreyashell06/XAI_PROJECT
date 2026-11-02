# CUB variants (Kaggle T4, seed 42, 2026-10-22)

Full table: `cub_variants.md` (made by `compare_runs.py`). Runs: `res_ps` = per-sample L4 PGD; `res_ps_l4` = per-sample,
lambda4 = 0.05, rho radius 0.1 (the evaluation radius); `res_small` = res_ps_l4 + an 8-15% occlusion edit.

**Final configuration for the report: per-sample L4, lambda4 = 0.05, radius 0.1 (`res_ps_l4`).**

| | FVLC, paper settings | FVLC, same settings | EC-FVLC λ5=0.03 | EC-FVLC λ5=0.1 |
|---|---|---|---|---|
| Accuracy ↑ | 72.94 | 74.28 | 74.20 | 73.39 |
| Evidence corr., held-out CLIP ↑ | 0.186 | 0.201 | 0.297 | 0.321 |
| Nuisance-edit TCPC ↓ | 0.164 | 0.163 | 0.125 | 0.110 |
| Paired IP TCPC ↓ | 0.0096 | 0.0092 | 0.0047 | 0.0023 |
| Worst-case TCPC r=0.1 ↓ | 0.203 | 0.193 | 0.208 | 0.223 |
| Top-k overlap under attack ↑ | 0.794 | 0.828 | 0.814 | 0.794 |
| Top-k with non-zero weight ↑ | 0.506 | 0.527 | 0.563 | 0.561 |
| Attribute AUROC ↑ | 0.645 | 0.650 | 0.678 | 0.686 |
| Within-species AUROC ↑ | 0.529 | 0.530 | 0.538 | 0.539 |
| Own/distant part ratio | 1.12 | 1.13 | 1.14 | 1.15 |

- Matching L4 to the evaluation attack (per-sample, radius 0.1, weight 0.05) cut EC-FVLC's worst-case stability cost from
  +36% (0.277 vs 0.203) to +8% (0.208 vs 0.193, same-settings FVLC); evidence gains are unchanged.
- Against FVLC with the paper's settings, EC-FVLC λ5=0.03 is better on every metric except worst-case TCPC, where it is
  0.208 vs 0.203 (within 0.005). Rule B (no worst-case cost at all) is narrowly missed.
- Small part-sized occlusions did not help: CLIP barely reacts to them (held-out evidence corr. on that edit 0.05-0.10),
  and locality, AUROC and the three common edits are unchanged or slightly worse. Dropped.
- Locality stays ≈ 0 (own/distant ratio 1.12-1.15) for every model: a limitation of the pooled ResNet-18 features.
