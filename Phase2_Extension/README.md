# Replicating FVLC: *Faithful Vision-Language Interpretation via Concept Bottleneck Models* (ICLR 2024)

DA351 Explainable AI, course project, **Phase 1 (replication)**.

Original paper: Songning Lai, Lijie Hu, Junxiao Wang, Laure Berti-Equille, Di Wang.
*Faithful Vision-Language Interpretation via Concept Bottleneck Models*, ICLR 2024.
[OpenReview](https://openreview.net/forum?id=rp0EdI8X4e).

**Relation to the official code.** The authors' repository is https://github.com/xll0328/FVLC (MIT; a single
commit from Feb 2026). It is a fork of Label-free CBM plus draft FVLC scripts. As of 2026-09-24 those scripts
do not run:
- `train_fcbm_all.py` and `train_fcbm_projonly.py` fail with a `SyntaxError` and use undefined arguments
  (`args.x_pgd_radius`, `batch`, ...).
- `eval.py` raises `NotImplementedError`.
- There is no code for the WP1/WP2/IP perturbations or for TCPC/TOPC.

This repository is therefore an **independent re-implementation of FVLC from the paper**. Label-free CBM
([Oikarinen et al., ICLR 2023](https://github.com/Trustworthy-ML-Lab/Label-free-CBM)) is re-implemented from its
public code. Where the official FVLC drafts reveal an intent, we note it in "Ambiguities" below.

## What is reproduced

| Paper result | What it shows | Script |
|---|---|---|
| **Table 1** – accuracy of Standard / Label-free CBM / base vs FVLC under 7 perturbations | FVLC keeps accuracy | `run_experiments.py` + `make_tables.py` |
| **Table 2** – TCPC / TOPC stability under 7 perturbations | FVLC is ~90% more stable | same |
| **Table 3** – ablation of L2 / L3 / L4 | every regulariser helps | `run_experiments.py --ablation` |
| **Figure 3** – concept weights before/after perturbation | qualitative | `visualize.py` |

Datasets: **CIFAR-10, CIFAR-100, CUB-200-2011**. Places365 (~100 GB) and HAM10000 (appendix) are out of scope
for the compute we have. The P-CBM row of Table 1 is quoted from the paper and not re-run.

## Method in one paragraph

Label-free CBM: (1) an LLM produces a concept set per class. (2–3) A concept bottleneck layer
`g(x) = W_c b_f(x)` (standardised) is learned so that each neuron matches the CLIP similarity
`M_ij = E_I(x_i)·E_T(p_j)`, using the cos³ loss of Eq. 1. (4) A sparse linear layer `y = W_F g(x)` is fitted
with GLM-SAGA. **FVLC** keeps everything frozen except the concept layer and learns `W̃_c` through the min–max
objective of Eq. 2 / Algorithm 1:

```
L1  KL(y(x,c̃) || y(x,c))                    closeness of prediction
L2  L_k1(g̃(x), g(x))                        similarity of explanation (top-k surrogate, Eq. 7)
L3  max_{|δ|∞≤R2} KL(y(x,c̃) || y(x,c̃+δ))    stability of prediction      (PGD)
L4  max_{|ρ|∞≤R1} L_k2(g̃(x), g̃(x)+ρ)        stability of explanation     (PGD)
```

Perturbations (Sec. 6.1): **WP1** swaps 5% or 10% of the concept words for synonyms. **WP2** adds
N(0, 0.001) noise to the CLIP text embeddings. **IP** adds N(0, 0.001) noise to the normalised input pixels.
The combinations of these give 7 settings. Metrics: `TCPC = ||c1−c2||/||c1||` and `TOPC = ||y1−y2||/||y1||`.

## Repository layout

```
fvlc/
  config.py        all hyper-parameters, each tagged [LF] / [FVLC] (paper Table 5) / [OURS]
  data.py          CIFAR-10/100, CUB loaders; IP noise wrapper
  models.py        backbones (CLIP-RN50, ResNet-18-CUB) and CLIP ViT-B/16
  features.py      caches backbone / CLIP image / CLIP text embeddings
  concepts.py      LLM concept generation + filters (App. C); WP1 synonym perturbation
  cbm.py           Label-free CBM: cos³ projection learning, concept filtering, sparse final layer
  fvlc.py          FVLC: top-k surrogate (Eq. 7), PGD for δ/ρ, Algorithm 1 training loop
  metrics.py       TCPC, TOPC, top-k overlap, accuracy
  pipeline.py      vanilla → perturbed base → FVLC → evaluation
  paper_results.py numbers from Tables 1–3 of the paper, used for the comparison
  third_party/glm_saga.py   vendored GLM-SAGA solver (MadryLab)
run_experiments.py  runs one dataset end-to-end, resumable (results/<ds>/results.json)
make_tables.py      writes results/tables.md (ours vs paper) + results/fig_stability.png
visualize.py        Figure-3-style concept plots
generate_concepts.py (optional) regenerate concept sets with any OpenAI-compatible LLM
notebooks/FVLC_colab.ipynb  one-click Colab/Kaggle run
tests/test_core.py  CPU tests on synthetic data (no downloads)
data/               class names + GPT-3 concept sets released by Label-free CBM
```

## Setup

```bash
git clone <this-repo> && cd fvlc-replication
pip install -r requirements.txt
python tests/test_core.py            # ~30 s on CPU, checks the whole pipeline on synthetic data
```

Python ≥ 3.10 with a CUDA GPU (a Colab/Kaggle T4 is enough). **Random seed: 42** everywhere
(`--seed` to change it). The perturbed baseline uses seed 43 (`base_seed_offset=1`).

### Datasets

| Dataset | Link | How |
|---|---|---|
| CIFAR-10 / CIFAR-100 | https://www.cs.toronto.edu/~kriz/cifar.html | downloaded automatically by torchvision |
| CUB-200-2011 | https://data.caltech.edu/records/65de6-vp158 | `bash scripts/download_cub.sh` → `data/CUB/{train,test}` |

Pretrained models download automatically: CLIP RN50 and ViT-B/16 (OpenAI), and ResNet-18-CUB
(pytorchcv/imgclsmob). Concept sets are the GPT-3 sets published by Label-free CBM. `generate_concepts.py`
re-creates them with a current LLM, since GPT-3 has been retired.

## Running

```bash
python run_experiments.py --dataset cifar10               # Tables 1 + 2      (~1 h on a T4)
python run_experiments.py --dataset cifar10 --ablation    # + Table 3         (+~45 min)
python run_experiments.py --dataset cifar100
bash scripts/download_cub.sh && python run_experiments.py --dataset cub
python make_tables.py                                     # results/tables.md, results/fig_stability.png
python visualize.py --dataset cifar10 --perturbation WP1_10+WP2+IP --index 0
```

Useful flags: `--perturbations WP2 IP` (subset), `--limit_train 2000 --fvlc_epochs 1` (smoke test),
`--force` (recompute), and any hyper-parameter such as `--fvlc_lr`, `--k1`, `--pgd_mode per_sample` or
`--rho_space concept`. Cached features go to `activations/` (set `FVLC_ACT_DIR` to put them on Google Drive).

## Ambiguities and our choices

The paper leaves several details unspecified. Every choice is a config flag, so each one can be ablated.

1. **What "base under perturbation" means.** In Table 2 the base TCPC is about 0.15, even for tiny noise
   (σ = 0.001). That cannot come from pushing the noise through a fixed model, so we read it as: *re-run the
   Label-free CBM concept-layer learning under the perturbation*. We keep the vanilla concept indices and W_F,
   so the vectors stay comparable.
2. **Training randomness.** The perturbed re-run uses seed+1. We add a **SEED_ONLY** control (no perturbation,
   new seed) to measure how much of the "instability" comes from training stochasticity alone.
3. **FVLC initialisation and targets.** Algorithm 1 initialises W̃_c randomly. We start from the perturbed
   base W_c (`--fvlc_init gaussian` gives the literal version). L1/L2 are anchored to the *vanilla* g(x), y(x,c).
   g̃ uses the vanilla standardisation, so W̃_c = W_c is reachable.
4. **Where ρ lives.** Taken literally, `L_k2(g̃, g̃+ρ)` with ρ in concept space depends on W̃_c only through the
   non-differentiable top-k index sets, so L4 would have zero gradient. Table 5 calls ρ's settings
   "x pgd", and the official draft (`train_fcbm_all.py`) applies its "X_PGDer" to the backbone features. So we
   put ρ on the backbone features (`--rho_space feature`, default), in units of each
   feature dimension's std. Raw CLIP-RN50 features have std ≈ 0.025, so an unscaled radius of 0.05 would
   shift every dimension by about 2σ. In our smoke test that made L4 dominate the loss and FVLC *less* stable.
5. **Universal vs per-sample perturbations.** Algorithm 1 averages gradients over batches, which gives one
   shared δ and ρ (`--pgd_mode universal`, default). `per_sample` is also available.
6. **Unreported hyper-parameters:** k1 = k2 = 10, 20 epochs, Adam with lr 1e-3 and cosine decay, batch 512.
   PGD radii, steps and λ follow Table 5.
7. **WP1** uses seeded WordNet synonym replacement instead of GPT-3 (`method="llm"` in `fvlc/concepts.py`
   gives the paper's version). We keep only the two most frequent senses, single lower-case words, and
   adjective senses for modifiers. WordNet swaps are still noisier than an LLM's (e.g. "yellow feet" →
   "chicken pes"), so our WP1 is, if anything, a *harsher* perturbation than the paper's.
8. **TCPC/TOPC reduction.** We report the mean of per-sample ratios, and the global Frobenius ratio in the
   diagnostics table.
9. Like Label-free CBM, the projection layer early-stops on the test split. We keep this for fidelity to the
   original pipeline.
10. **Top-k surrogate.** We implement Eq. 7 of the paper, the symmetric ℓ1 distance on the top-k entries. The
    official draft `metric.py` instead uses `-mean(pred[topk(gt)])` with K = 5, which differs from the paper.

## Three evaluation protocols (Tables 2 and 3)

The paper does not say how TCPC/TOPC are computed for FVLC, so the repo implements three readings
(`--protocol`):

| | FVLC reference in L1/L2 | TCPC/TOPC compare | Where the reading comes from |
|---|---|---|---|
| `anchor` (our first run) | clean vanilla LF-CBM | vanilla on clean inputs vs base/FVLC on the perturbed pipeline | literal reading of Sec. 6.1 |
| `seat` | LF-CBM of the same concept space (Def. 2) | the **same** model on clean inputs vs a test-time ℓ∞ PGD attack on its features | SEAT (Hu et al., AAAI 2023, same lab); the official FVLC repo re-uses its `PGDAttacker` |
| `paired` (+ `--fvlc_init gaussian`) | clean vanilla LF-CBM | each method vs **itself** trained without the perturbation (base: vanilla vs LF-CBM under P; FVLC: FVLC-clean vs FVLC under P) | symmetric reading of "c1 before, c2 after the perturbation"; random init as in Algorithm 1 |

1,000-image CIFAR-10 check (CPU):

| Protocol | base TCPC → FVLC TCPC (image noise, IP) | What drives the gain |
|---|---|---|
| anchor | 0.35 → 0.26 (≈25%) | L2 |
| seat | 0.10 → 0.089 (≈13%); L1+L4 alone 0.066 | L4, as the paper claims |
| paired | 0.35 → 0.03–0.04 (**≈88–91%**), close to the paper's ~90% | mainly L1: distilling from the fixed vanilla model |

Caveat for `paired`: for the word perturbations (WP1/WP2), FVLC's TCPC is exactly 0. FVLC's training never
uses the concept text; it only sees backbone features and the vanilla model's outputs. So the ~90% gain for
WP1/WP2 holds by construction, and the informative number is IP.

## Results

Final results (Kaggle T4, seed 42) and the one-page comparison with the paper are in `REPLICATION_SUMMARY.md`.

| Folder | Contents |
|---|---|
| `results_paired/` | main protocol, Tables 1-3 for CIFAR-10, CIFAR-100 and CUB |
| `results_seat/` | CIFAR-10 SEAT-style attack protocol, including the full Table 3 ablation |
| `results_dense/` | Table 1 "Standard" row with a weakly regularised probe (`python scripts/dense_standard.py --dataset cifar100`) |
| `results/` | earlier anchor protocol |

## Phase 2 (in progress): two-sided faithfulness

**Gap.** FVLC treats "faithful" as "stable": Def. 2, Eq. 2 and TCPC/TOPC reward a concept layer for *not*
changing, but never check that it changes when the visual evidence for a concept changes, which the paper's own
Sec. 2 and 5 say faithfulness needs. An explainer that ignores its input is perfectly "stable".
**Planned fix.** Evidence-Consistent FVLC (EC-FVLC), one extra loss term (concept infidelity). Full proposal:
project doc `phase2-gap-and-proposal.md`.

**Step 1, pilot (go/no-go).** `run_pilot.py` scores a constant explainer, Label-free CBM and FVLC on

| Side | Metric | Code |
|---|---|---|
| stability | worst-case TCPC / top-k overlap under a test-time feature PGD attack | `faithfulness.worst_case_stability` |
| decision | share of top-k concepts with a non-zero final weight for the predicted class; activations vs contributions | `faithfulness.decision_relevance` |
| sensitivity | part-masking locality on CUB keypoints (Raman et al., TMLR 2025) | `faithfulness.locality` |
| grounding | AUROC of each concept score against CUB's per-image attribute labels, overall and within species | `faithfulness.attribute_auroc` |

```bash
bash scripts/download_cub.sh               # now also keeps data/CUB_meta/ (parts, boxes, attributes)
python tests/test_faithfulness.py          # CPU, fake CUB, ~10 s
python run_pilot.py --dataset cub          # results_phase2/cub/pilot.md  (~30-40 min on a T4)
```
Kaggle: `notebooks/kaggle/kaggle_phase2_pilot_cub.ipynb`.

**Step 2, EC-FVLC.** One new term in Eq. (2): **L5 = E_T (1/k) Σ_j (Δg̃_j − a_j Δs_j)²**, where T is a label-free
image edit (`fvlc/views.py`: occlusion, crop, CutMix, and a nuisance edit), Δg̃_j the change of concept neuron j,
Δs_j the change of CLIP's standardised image-text similarity for concept j, and a_j = corr(g_j, s_j) on clean
training images. When an edit leaves the evidence unchanged, L5 is a stability term; when it changes it, L5 asks
the neuron to move with it. `lambda5 = 0` is FVLC exactly. Only W̃c is trained; no concept labels are used.

```bash
python tests/test_ecfvlc.py                              # CPU, synthetic: L5 definition, views, trade-off
python run_phase2.py --dataset cub                       # results_phase2/cub/phase2.md   (~1.5-2 h on a T4)
python run_phase2.py --dataset cifar10 --lambda5 0.01 0.03 0.1
```
Rows: constant explainer, LF-CBM, FVLC (before), FVLC + views without L5 (ablation), EC-FVLC for each lambda5
(after). Extra metrics: evidence correlation on held-out test edits, scored by the training CLIP and by a CLIP
never used in training (`oracle_clip`, default ViT-L/14); paired IP stability (each model trained on the clean and
the IP pipeline). Kaggle: `kaggle_phase2_cub.ipynb`, `kaggle_phase2_cifar10.ipynb`. Code: `fvlc/views.py`,
`fvlc/phase2.py`, L5 in `fvlc/fvlc.py`, view caching in `fvlc/features.py`. Concepts are mapped to CUB body parts and
(part, colour) attributes by the word lists in `fvlc/cub_meta.py` (133 / 111 of the 208 CUB concepts).

## Credits

* Label-free CBM (Oikarinen et al., 2023), https://github.com/Trustworthy-ML-Lab/Label-free-CBM: pipeline design, concept sets, class files, CUB split,
  hyper-parameters. `fvlc/cbm.py` re-implements `train_cbm.py` / `similarity.py`.
* GLM-SAGA (Wong et al., 2021), https://github.com/MadryLab/glm_saga: vendored unmodified in `fvlc/third_party/` (see upstream for its license).
* CLIP (Radford et al., 2021); pytorchcv / imgclsmob ResNet-18-CUB.
* AI assistance is documented in `AI_USAGE.md`.
