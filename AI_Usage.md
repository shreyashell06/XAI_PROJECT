# AI Usage & Transparency Statement
## (Phase 1 & Phase 2)

This document provides a transparent, comprehensive record of how AI assistance was utilized throughout our project. 

We used **Claude (Anthropic — Claude 3.5 Sonnet / Claude Sonnet)** primarily as a **technical pair programmer, debugging assistant, and brainstorming partner**. All actual model training, GPU experiments on Kaggle, mathematical verifications, parameter tuning, and analytical conclusions were conducted and validated by our team.

---

## 1. Summary: AI Assistance vs. Team Contribution

| Task Category | Role of Claude Sonnet | Team Verification & Modification (Our Work) |
|---|---|---|
| **Phase 1: Code Scaffolding** | Drafted initial PyTorch class skeletons for the CBM pipeline based on paper equations. | The official paper code was broken; we audited all equations against Sections 3 & 4 of the ICLR 2024 paper, fixed WordNet synset errors in WP1, and resolved GPU memory bottlenecks. |
| **Phase 1: Replication Runs** | Suggested scripts to format LaTeX/Markdown tables (`make_tables.py`). | We executed all runs on Colab and Kaggle T4 GPUs across CIFAR-10, CIFAR-100, and CUB-200, cross-verifying every single number against Tables 1, 2, and 3. |
| **Phase 2: Research Gap Brainstorming** | Discussed potential limitations of pure stability in concept models. | We formulated the specific hypothesis that stability $\neq$ faithfulness to visual evidence, and designed the **Constant Explainer** pilot study to prove that a blind model scores a perfect 0.000 TCPC. |
| **Phase 2: $L_5$ Loss & Implementation** | Assisted in drafting the vectorized Pearson correlation penalty between CLIP evidence deltas and concept activations. | We structured the 4 image edit transformations (occlude, crop, cutmix, nuisance), conducted the $\lambda_5$ hyperparameter sweeps on Kaggle, and chose optimal values using our accuracy-preservation rule. |
| **Phase 2: Evaluation & Plotting** | Assisted with matplotlib layout adjustments (converting scatter plots into clean lollipop/line charts). | We designed the held-out evaluation protocol using ViT-L/14 CLIP to avoid circular evaluation, added disk-caching mechanisms so plots run without full GPU re-training, and validated against human biologist annotations. |

---

## 2. Chronological Log of Prompts and Iterations

### Phase 1: Replication of FVLC (ICLR 2024)

#### Entry 1.1: Project Setup & Independent Reimplementation
* **Tool Used:** Claude Sonnet
* **Prompt/Purpose:** *"We are replicating the ICLR 2024 paper 'Faithful Vision-Language Concept Bottleneck Models' on CIFAR-10, CIFAR-100, and CUB-200. Help us structure an independent modular PyTorch codebase implementing Algorithm 1 and the min-max loss."*
* **Output Provided:** Initial directory layout, template for `fvlc/cbm.py` and `run_experiments.py`.
* **How We Modified & Verified It:**  
  1. We found that the generated perturbation code for WP1 (synonym replacement) selected nonsensical WordNet senses (e.g., confusing the color "black" with proper nouns). We manually constrained WordNet synsets to adjectives and visual nouns.
  2. Batch size 512 caused CUDA Out-Of-Memory errors on our Colab instances; we added command-line `--batch_size` arguments and optimized data loaders.
  3. We verified that the adversarial PGD radius matched the paper's per-dimension feature standard deviation rather than raw Euclidean distance.

#### Entry 1.2: Verification of Tables 1, 2, and 3
* **Tool Used:** Claude Sonnet
* **Prompt/Purpose:** *"Help us write a script to parse our output JSON logs for tables matching Table 1, Table 2, and Table 3 from the paper."*
* **Output Provided:** `make_tables.py` script.
* **How We Modified & Verified It:**  
  We audited the generated values against `results.json` directly. We ran all seed-42 training runs ourselves and confirmed that FVLC improved TCPC stability by ~85–90% while maintaining classification accuracy.

---

### Phase 2: Research Gap & EC-FVLC Extension

#### Entry 2.1: Designing the $L_5$ Loss & Edit Transformations
* **Tool Used:** Claude Sonnet
* **Prompt/Purpose:** *"We want to propose a loss term $L_5$ that penalizes concept scores if they don't drop when visual evidence is physically removed. Help us write a vectorized Pearson correlation loss comparing CLIP similarity changes to concept score changes across 4 edit types."*
* **Output Provided:** Draft implementation of the Pearson correlation formula in PyTorch.
* **How We Modified & Verified It:**  
  1. We tested the loss behavior on synthetic dummy tensors before integrating it into `fvlc/phase2.py`.
  2. We noticed numerical instability when variance approached zero (division by zero in Pearson $r$); we added an epsilon clamp (`1e-12`).
  3. We separated the edits into evidence-destroying (occlude, crop, cutmix) where concept scores should drop, and nuisance edits where concept scores should remain still.

#### Entry 2.3: Hyperparameter Tuning & Visualization Refinement
* **Tool Used:** Claude Sonnet
* **Prompt/Purpose:** *"The scatter plot of concept changes under occlusion is cluttered. Help us redesign Figure 2 as a lollipop/horizontal line chart showing scores at the end of each line, and fix a variable scope issue when running in Kaggle without GPU."*
* **Output Provided:** Matplotlib code converting scatter dots to horizontal bars with text annotations.
* **How We Modified & Verified It:**  
  1. We added an automatic cache loader (`partB_cache.npz`) inside the notebook so that Part B visualization cells could be re-rendered instantly without retraining models on GPU.
  2. We conducted the $\lambda_5$ grid search (0.01 to 0.5) on Kaggle T4 GPUs, selecting $\lambda_5 = 0.3$ for CIFAR and $\lambda_5 = 0.1$ for CUB based on our criteria of keeping accuracy within 1% of baseline.

---

## 3. Reflections on AI Assistance

Using Claude Sonnet as a pair programming assistant accelerated our development cycle, particularly for:
- Writing repetitive matplotlib boilerplate and formatting markdown tables.
- Debugging obscure PyTorch broadcasting errors during vector correlation calculations.

However, **critical scientific thinking, hypothesis formulation, and empirical verification were strictly driven by our team**:
- The insight that a Constant Explainer achieves 0.000 TCPC was our formulation to expose the paper's metric limitation.
- All numbers reported in our submission were generated by our own compute runs on Kaggle, not simulated or fabricated.
