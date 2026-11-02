# Phase 2 summary: EC-FVLC vs FVLC on three datasets

Seed 42, paired protocol, Kaggle T4. EC-FVLC row = lambda5 with the highest evidence correlation under the held-out CLIP
(ViT-L/14, never used in training) among values within 1 accuracy point of FVLC. Arrows show which direction is better.

| Dataset | Model | Acc ↑ | Evidence corr. (held-out CLIP) ↑ | Nuisance-edit TCPC ↓ | Paired IP TCPC ↓ | Worst-case TCPC r=0.1 ↓ | Top-k overlap under attack ↑ | Top-k with non-zero weight ↑ |
|---|---|---|---|---|---|---|---|---|
| cub | Label-free CBM | 73.71 | 0.278 | 0.171 | – | 0.365 | 0.739 | 0.530 |
| cub | FVLC (before) | 72.94 | 0.186 | 0.164 | 0.0096 | 0.203 | 0.794 | 0.506 |
| cub | FVLC + edits, no L5 | 73.96 | 0.256 | 0.173 | 0.0181 | 0.237 | 0.791 | 0.533 |
| cub | **EC-FVLC λ5=0.1 (after)** | 73.01 | 0.319 | 0.118 | 0.0027 | 0.273 | 0.742 | 0.561 |
| cifar10 | Label-free CBM | 86.51 | 0.251 | 0.744 | – | 0.182 | 0.813 | 0.561 |
| cifar10 | FVLC (before) | 86.50 | 0.237 | 0.747 | 0.0293 | 0.188 | 0.818 | 0.560 |
| cifar10 | FVLC + edits, no L5 | 86.53 | 0.244 | 0.733 | 0.0246 | 0.183 | 0.820 | 0.560 |
| cifar10 | **EC-FVLC λ5=0.3 (after)** | 86.39 | 0.331 | 0.366 | 0.0119 | 0.151 | 0.723 | 0.564 |
| cifar100 | Label-free CBM | 65.09 | 0.218 | 0.770 | – | 0.137 | 0.783 | 0.351 |
| cifar100 | FVLC (before) | 64.91 | 0.185 | 0.770 | 0.0762 | 0.127 | 0.791 | 0.350 |
| cifar100 | FVLC + edits, no L5 | 65.15 | 0.197 | 0.744 | 0.0717 | 0.126 | 0.787 | 0.351 |
| cifar100 | **EC-FVLC λ5=0.3 (after)** | 64.88 | 0.310 | 0.249 | 0.0086 | 0.083 | 0.719 | 0.379 |

CUB only (human annotations): attribute AUROC LF-CBM 0.690, FVLC 0.645, EC-FVLC 0.685; within-species 0.545 / 0.529 / 0.539;
part-masking locality ≈ 0 for every model (own/distant change ratio ≈ 1.13). Full tables: `cub/phase2.md`, `cifar10/phase2.md`, `cifar100/phase2.md`.
