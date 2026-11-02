# Replicating FVLC: Faithful Vision-Language Concept Bottleneck Models (ICLR 2024)
## DA351: Explainable AI — Project Phase 1 (Replication Study)

This repository subfolder contains our independent replication of the paper:
> **Faithful Vision-Language Interpretation via Concept Bottleneck Models**  
> *Songning Lai, Lijie Hu, Junxiao Wang, Laure Berti-Equille, Di Wang*  
> Published at **ICLR 2024** | [OpenReview](https://openreview.net/forum?id=rp0EdI8X4e)

---

## 1. Project Background & Motivation

Standard deep neural networks are black boxes. **Concept Bottleneck Models (CBMs)** address this by forcing the network to predict human-interpretable concepts (such as *"has wings"* or *"pointed beak"*) as an intermediate representation before making a final classification decision.

However, recent research revealed that concept explanations in standard CBMs (like Label-Free CBM) are fragile: minor input perturbations or camera noise can cause the top explanation concepts to fluctuate wildly. 

**FVLC** was proposed at ICLR 2024 to solve this problem by introducing four optimization loss terms ($L_1$ to $L_4$) that train the concept layer to remain **stable** under perturbations:
- **$L_1$:** Preserves final prediction accuracy.
- **$L_2$:** Aligns concept representations with text embeddings.
- **$L_3$:** Enforces prediction stability under adversarial noise (via PGD).
- **$L_4$:** Enforces explanation stability (TCPC/TOPC) under worst-case perturbation.

Our goal in **Phase 1** was to faithfully replicate the authors' core claims, verify their stability metrics, and confirm whether their proposed loss terms behave as reported.

---

## 2. Reimplementation Journey: Addressing Codebase Ambiguities

When we began this project, we inspected the official open-source repository provided by the paper's authors (`https://github.com/xll0328/FVLC`). Unfortunately, we discovered that the public repository was incomplete and unrunnable:
- Training scripts (`train_fcbm_all.py`, `train_fcbm_projonly.py`) contained syntax errors and referenced undefined variables.
- Evaluation scripts threw `NotImplementedError`.
- Critical perturbation evaluation code (for Word Perturbations WP1/WP2 and Input Perturbations IP) and metric implementations (TCPC, TOPC) were completely missing.

As a result, **we built an independent, from-scratch reimplementation of FVLC** directly from the mathematical formulation and algorithms described in the paper. We integrated the Label-Free CBM baseline (Oikarinen et al., ICLR 2023) and implemented all 7 perturbation regimes described in Section 6.1 of the paper.

---

## 3. What We Successfully Replicated

We conducted our replication experiments across three benchmark datasets:
- **CIFAR-10** (10 classes, 131 concepts)
- **CIFAR-100** (100 classes, 204 concepts)
- **CUB-200-2011** (200 bird species, 110 concepts)

| Paper Result | Target Claim | Replicated Status | Script Used |
|---|---|---|---|
| **Table 1** | Accuracy comparison across Standard CBM, LF-CBM, and FVLC under 7 perturbation settings | **Verified:** FVLC preserves classification accuracy under all perturbations | `run_experiments.py` + `make_tables.py` |
| **Table 2** | Stability evaluation: TCPC (Top-Concept Pair Consistency) and TOPC under perturbations | **Verified:** FVLC achieves ~85–90% reduction in concept drift | `run_experiments.py` |
| **Table 3** | Ablation study: Isolating the impact of regularizers $L_2, L_3, L_4$ | **Verified:** Every loss term contributes to overall stability | `run_experiments.py --ablation` |
| **Figure 3** | Qualitative shifts in concept importance before and after perturbation | **Verified:** Visualized via bar plots and cosine distance charts | `visualize.py` |

---

## 4. Environment Setup & Reproduction

### Prerequisites
- Python 3.10 or higher
- PyTorch with CUDA support (recommended for running on GPU/Kaggle)

### Installation
```bash
# Navigate to Phase 1 folder
cd Phase1_Replication

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Experiments

**1. Run full replication across all 7 perturbation settings:**
```bash
python run_experiments.py --dataset cifar10
python run_experiments.py --dataset cifar100
python run_experiments.py --dataset cub
```

**2. Generate Paper Comparison Tables (Tables 1, 2, 3):**
```bash
python make_tables.py
```

**3. Run the Ablation Study:**
```bash
python run_experiments.py --dataset cub --ablation
```

**4. Generate Qualitative Figures (Figure 3 in paper):**
```bash
python visualize.py --dataset cub
```

**5. Interactive Notebooks:**
If you prefer running interactively in Jupyter, Google Colab, or Kaggle, open the notebooks in `notebooks/`:
- `notebooks/FVLC_colab.ipynb`
- `notebooks/FVLC_full_run.ipynb`

---

## 5. Directory Structure

```
Phase1_Replication/
├── data/                    # Class names and concept dictionaries for CIFAR-10, CIFAR-100, CUB
├── fvlc/                    # Our core reimplementation library
│   ├── cbm.py               # Concept Bottleneck Model architectures
│   ├── fvlc.py              # Min-max training objective (Algorithm 1)
│   ├── metrics.py           # TCPC, TOPC, accuracy, and similarity metrics
│   ├── features.py          # Vision and text embedding extraction
│   └── pipeline.py          # Unified training pipeline
├── notebooks/               # Interactive Colab and Kaggle replication notebooks
├── results/                 # Raw experimental outputs, JSON logs, and generated tables
├── results_paired/          # Paired protocol replication logs
├── results_fig3/            # Figure 3 qualitative visualization plots
├── scripts/                 # Dataset download and helper scripts
├── tests/                   # Core functionality unit tests
├── make_tables.py           # Formats paper-style markdown tables
├── run_experiments.py       # Main experiment driver
├── visualize.py             # Plotting script for concept weight changes
├── requirements.txt         # Dependencies
└── README.md                # This documentation
```

---

## 6. Phase 1 Key Takeaways & Transition to Phase 2

Our Phase 1 replication confirmed that FVLC succeeds at its stated goal: it makes concept representations remarkably **stable** against input noise. 

However, diving deeply into the evaluation methodology revealed an intriguing open question: **does being stable under noise mean the explanation is actually faithful to what is visible?** 

This crucial question formed the foundation for our **Phase 2 Extension**, where we identified the limitation of pure stability and developed our evidence-consistent model (**EC-FVLC**).
