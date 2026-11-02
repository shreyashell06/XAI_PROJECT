# DA351: Explainable AI — Comprehensive Project Repository
## Evaluating and Improving Concept Faithfulness in Vision-Language Models: From FVLC to EC-FVLC

Welcome to our project repository for **DA351: Explainable AI**. This project investigates the faithfulness of Concept Bottleneck Models (CBMs) across two phases:
1. **Phase 1 (Replication):** An independent from-scratch replication of the ICLR 2024 paper *"Faithful Vision-Language Interpretation via Concept Bottleneck Models"* (FVLC) by Lai et al.
2. **Phase 2 (Extension — EC-FVLC):** Identifying a fundamental blind spot in stability-based faithfulness, proposing an Evidence-Consistent extension (**EC-FVLC**) with a new loss term ($L_5$), and experimentally validating our solution across three benchmarks.

---

## 1. Project Overview & Two-Phase Roadmap

```
                                  [DA351 XAI PROJECT]
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌─────────────────────────────────┐                 ┌─────────────────────────────────┐
│     PHASE 1: REPLICATION        │                 │       PHASE 2: EXTENSION        │
│    (FVLC, Lai et al., ICLR 2024)│                 │   (EC-FVLC: Our Improvement)    │
├─────────────────────────────────┤                 ├─────────────────────────────────┤
│ • Independent reimplementation  │                 │ • Research Gap: Stability ≠     │
│   of FVLC min-max objective.    │                 │   visual evidence faithfulness. │
│ • Validated on CIFAR-10,        │     Revealed    │ • Pilot Study: Dummy constant   │
│   CIFAR-100, and CUB-200.       │   Blind Spot    │   explainer wins on stability.  │
│ • Replicated Tables 1, 2, 3     │ ──────────────> │ • Proposed L5 Infidelity Loss   │
│   and Figure 3.                 │                 │   using CLIP evidence deltas.   │
│ • Finding: FVLC makes concepts  │                 │ • Result: +39% to +71% evidence │
│   stable, but insensitive.      │                 │   correlation; accuracy intact. │
└─────────────────────────────────┘                 └─────────────────────────────────┘
```

---

## 2. Summary of Phase 1 (Replication)

* **Objective:** Verify the published claims of FVLC (Lai et al., ICLR 2024) that adding regularization losses ($L_1$ to $L_4$) stabilizes concept representations against 7 input and word perturbation settings.
* **Key Challenge:** The authors' public repository was incomplete and unrunnable. We built an independent reimplementation from scratch directly from the paper's algorithms.
* **Findings:**
  - **Accuracy Preserved (Table 1):** FVLC maintained standard classification accuracy across CIFAR-10 (~86.5%), CIFAR-100 (~65%), and CUB-200 (~73%).
  - **Stability Verified (Table 2):** TCPC (Top-Concept Pair Consistency) and TOPC improved by ~85–90% compared to unregularized Label-Free CBMs.
  - **Regularizers Confirmed (Table 3):** Ablation proved that each loss term ($L_2, L_3, L_4$) contributes to adversarial robustness.

---

## 3. Summary of Phase 2 (Research Gap & EC-FVLC Extension)

* **The Research Gap:** FVLC defines faithfulness purely as stability (concepts staying the same under noise). But when an object is **occluded** or **cropped**, visual parts genuinely disappear. A faithful explanation *must* decrease its concept scores. FVLC barely moved because it was trained to resist change.
* **Our Pilot Study Proof:** We built a "Constant Explainer" that outputs identical concept activations for all images. On FVLC's own stability metric, this completely blind model scored **0.000 (flawless stability)**, beating FVLC, despite having **0.52% random accuracy**. This proved that evaluating stability alone rewards blind models.
* **Our Proposed Solution (EC-FVLC):**
  - We introduced an additive loss term, **$L_5$ (Concept-Evidence Infidelity Loss)**, weighted by $\lambda_5$:
    $$\mathcal{L}_{\text{EC-FVLC}} = \mathcal{L}_{\text{FVLC}} + \lambda_5 \cdot \mathcal{L}_5$$
  - Applied across 4 controlled transformations: **Occlusion**, **Crop**, **CutMix**, and **Nuisance**.
  - $L_5$ measures the Pearson correlation between **CLIP's visual evidence changes** ($\Delta \mathbf{s}_{\text{CLIP}}$) and the **concept bottleneck activations** ($\Delta \mathbf{c}$), penalizing mismatch.
* **Key Results (Evaluated on held-out ViT-L/14 CLIP never seen during training):**
  - **CIFAR-10:** Evidence correlation improved from $0.237 \rightarrow \mathbf{0.331}$ (**+39.7%**).
  - **CIFAR-100:** Improved from $0.185 \rightarrow \mathbf{0.310}$ (**+67.6%**).
  - **CUB-200:** Improved from $0.186 \rightarrow \mathbf{0.319}$ (**+71.5%**).
  - **Accuracy:** Fully preserved within 0.1% of baseline.
  - **Human Validation:** On CUB-200, AUROC against real human biologist attribute annotations recovered from $0.645$ up to $\mathbf{0.685}$.

---

## 4. Repository Structure

The codebase is organized into two self-contained directories:

```
XAI_PROJECT/
├── README.md                          # This file (overall project guide)
│
├── Phase1_Replication/                # All Phase 1 replication code & results
│   ├── fvlc/                          # Core reimplemented CBM and FVLC library
│   ├── notebooks/                     # Colab and Kaggle replication notebooks
│   ├── results/                       # Generated tables, JSON logs, and Fig 3 plots
│   ├── results_paired/, results_seat/ # Additional experimental logs
│   ├── data/, scripts/, tests/        # Supporting datasets and unit tests
│   ├── make_tables.py                 # Formats paper-style comparison tables
│   ├── run_experiments.py             # Phase 1 replication runner
│   ├── visualize.py                   # Plotting script for concept changes
│   ├── requirements.txt               # Phase 1 dependencies
│   └── README.md                      # Detailed Phase 1 documentation
│
└── Phase2_Extension/                  # All Phase 2 extension code, notebooks & docs
    ├── fvlc/                          # Extended library with phase2.py, views.py, etc.
    │   ├── phase2.py                  # L5 loss implementation & EvidenceData pipeline
    │   ├── views.py                   # 4 edit transformation operators
    │   └── faithfulness.py            # Locality & human attribute AUROC evaluation
    ├── results_phase2/                # Evaluation logs for CUB, CIFAR-10, CIFAR-100
    │   ├── SUMMARY.md                 # Summary results across all 3 datasets
    │   └── pilot.md                   # Pilot study numbers (Constant Explainer)
    ├── ec-fvlc-plots-updated.ipynb    # Kaggle/Jupyter notebook with all charts & cache guards
    ├── run_phase2.py                  # Phase 2 EC-FVLC training CLI runner
    ├── run_pilot.py                   # Pilot study runner
    ├── requirements.txt               # Phase 2 dependencies
    └── README.md                      # Detailed Phase 2 documentation
```

---

## 5. Environment Setup & Dependencies

Both phases share standard Python scientific computing libraries. We recommend using Python 3.10+:

```bash
# Clone the repository
git clone https://github.com/shreyashell06/XAI_PROJECT.git
cd XAI_PROJECT

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies (from either folder)
pip install -r Phase2_Extension/requirements.txt
```

---

## 6. Execution & Step-by-Step Reproduction Guide

### Reproducing Phase 1 (FVLC Replication)
```bash
cd Phase1_Replication

# 1. Run replication experiments on CIFAR-10, CIFAR-100, or CUB:
python run_experiments.py --dataset cifar10
python run_experiments.py --dataset cifar100
python run_experiments.py --dataset cub

# 2. Output Tables 1, 2, and 3:
python make_tables.py

# 3. Generate qualitative concept shift plots (Figure 3):
python visualize.py --dataset cub
```

### Reproducing Phase 2 (EC-FVLC Extension)
```bash
cd ../Phase2_Extension

# 1. Run the Constant Explainer pilot study on CUB-200:
python run_pilot.py --dataset cub

# 2. Train and evaluate EC-FVLC:
python run_phase2.py --dataset cifar10 --lambda5 0.3
python run_phase2.py --dataset cifar100 --lambda5 0.3

# (For CUB-200, ensure CUB dataset is downloaded via scripts/download_cub.sh first)
python run_phase2.py --dataset cub --lambda5 0.1

# 3. Run interactive visualizations and generate comparison figures:
# Open and run ec-fvlc-plots-updated.ipynb in Jupyter or Kaggle.
# - Part A (Cells 3–16): Loads saved benchmark numbers and generates all lollipop & bar charts without GPU.
# - Part B (Cells 18–25): Runs per-image concept response comparisons under occlusion.
```

---

## 7. Master Comparison Summary (Phase 2 Extension)

| Dataset | Metric | LF-CBM Baseline | FVLC (ICLR 2024) | FVLC + Edits (Ablation) | **EC-FVLC (Ours)** | Improvement |
|---|---|---|---|---|---|---|
| **CIFAR-10** | **Evidence Corr. (Held-out ViT-L/14) ↑** | 0.251 | 0.237 | 0.244 | **0.331** | **+39.7%** |
| | Test Accuracy (%) ↑ | 86.51 | 86.50 | 86.53 | **86.39** | −0.11% (Preserved) |
| | Nuisance TCPC ↓ | 0.744 | 0.747 | 0.733 | **0.366** | **−51.0%** |
| **CIFAR-100**| **Evidence Corr. (Held-out ViT-L/14) ↑** | 0.218 | 0.185 | 0.197 | **0.310** | **+67.6%** |
| | Test Accuracy (%) ↑ | 65.09 | 64.91 | 65.15 | **64.88** | −0.03% (Preserved) |
| | Nuisance TCPC ↓ | 0.770 | 0.770 | 0.744 | **0.249** | **−67.7%** |
| **CUB-200**  | **Evidence Corr. (Held-out ViT-L/14) ↑** | 0.278 | 0.186 | 0.256 | **0.319** | **+71.5%** |
| | Test Accuracy (%) ↑ | 73.71 | 72.94 | 73.96 | **73.01** | +0.07% (Preserved) |
| | Human Biologist Attribute AUROC ↑ | 0.690 | 0.645 | 0.685 | **0.685** | **+0.040** |
| | Nuisance TCPC ↓ | 0.171 | 0.164 | 0.173 | **0.118** | **−28.0%** |

---

## 8. Presentation & Evaluation Deliverables

For course evaluators and presentation reviewers, our supporting materials are structured as follows:
- **Presentation Slide Deck & Guide:** Available in `Phase2_Extension/presentation_slides_and_guide.md`.
- **10-Minute Presentation Spoken Script:** Available in `Phase2_Extension/presentation_script_10min.md`.
- **Experimental Comparison Report (PDF Source):** Available in `Phase2_Extension/experimental_results_comparison.md`.
- **Concepts & Terms Glossary:** Available in `Phase2_Extension/tech_terms_glossary_guide.md`.
