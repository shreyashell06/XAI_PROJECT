# Replication summary: FVLC (Lai et al., ICLR 2024)

**DA351 Explainable AI, Phase 1**  
**Team:** _names_  **Repo:** _GitHub link_  **Seed:** 42  **Hardware:** Kaggle and Colab, T4 GPU

## What the paper does

The paper starts from Label-free Concept Bottleneck Models. In these, GPT-3 writes a concept list for each class, CLIP learns
a layer that scores those concepts, and a linear layer predicts the class from the scores. The authors show that
small changes (a few words swapped in the concept list, tiny noise on the text embeddings or on the images) shift
the concept scores a lot, so the explanation isn't trustworthy. Their fix, FVLC, retrains only the concept layer
with four loss terms (Eq. 2). L1 keeps predictions close to the original model, L2 keeps the top concepts the
same, and L3 and L4 keep predictions and top concepts stable under worst-case nudges found with PGD.

## What we did

The authors' GitHub code doesn't run (syntax errors, and no perturbation or evaluation code), so we rebuilt the
pipeline from the paper, using the public Label-free CBM code for the parts FVLC builds on. We used the same
datasets (CIFAR-10, CIFAR-100, CUB), backbones (CLIP-RN50 and ResNet-18-CUB), concept sets, hyperparameters (Table 5)
and metrics (TCPC and TOPC, p. 8). We ran all 7 perturbations on every dataset. Colab kept disconnecting, so we
moved to Kaggle and ran one dataset per notebook; all notebooks are in `notebooks/` with their outputs.

## Results

| Dataset | Label-free CBM acc, ours / paper | FVLC acc, ours / paper | Standard acc, ours / paper | Stability gain, ours / paper |
|---|---|---|---|---|
| CIFAR-10 | 86.51 / 86.32 | 86.50 / 86.41 | 88.59 / 88.80 | 96.5% / 90.3% |
| CIFAR-100 | 65.09 / 65.42 | 64.94 / 65.20 | 69.84 / 70.10 | 94.6% / 93.9% |
| CUB | 73.71 / 74.23 | 72.94 / 74.10 | 75.92 / 76.70 | 99.4% / 89.7% |

*FVLC accuracy is the mean over the 7 perturbations. The stability gain is how much FVLC reduces TCPC and TOPC
compared with Label-free CBM. Side-by-side images of the paper's tables next to ours are in `results/comparison/`.*

- **Table 1 (accuracy): replicated.** Every number is within 1.2 points of the paper, and FVLC costs almost no
  accuracy (0.0, 0.2 and 0.8 points).
- **Table 2 (stability): replicated.** We get the paper's "about 90%" reduction on all three datasets. On
  CIFAR-10 our average FVLC TCPC is 0.0084, against the paper's 0.0085.
- **Table 3 (ablation, CIFAR-10): partly replicated.** When we attack each trained model at test time, every
  combination that includes L4 is 20–24% more stable than the same combination without it. That matches the
  paper's point that L4 matters most. In our runs, though, L2 and L3 made no measurable difference, and our values
  are larger than the paper's.
- **Figure 3: replicated.** On one test image per dataset, the perturbed Label-free CBM brings in reworded
  concepts (for example "a scratching station" for a cat). FVLC keeps the original top concepts.

## Things worth knowing

- The paper doesn't say exactly how it measures TCPC and TOPC for FVLC. We compare each model with itself trained
  with and without the perturbation, and this is the only reading we found that matches the paper's numbers.
- FVLC learns from image features and never reads the concept text, so it is unchanged by the word perturbations
  by design. Image noise is the real test, and there FVLC removes 84%, 68% and 97% of the change.
- Retraining Label-free CBM with only a new random seed changes it about as much as the paper's text and image
  noise does. A fair stability test should separate the two.
- GPT-3 (text-davinci) has been retired, so for the synonym swaps we used WordNet instead. We skipped Places365
  (about 100 GB) and ran the ablation on CIFAR-10 only.

## Conclusion

We reproduced the paper's main claims: FVLC keeps Label-free CBM's accuracy and makes its explanations about 90%
more stable. We also confirmed that L4 is the term that makes the difference. The two open points (L2 and L3 not
helping, and the effect of seed randomness) are where we plan to take Phase 2.
