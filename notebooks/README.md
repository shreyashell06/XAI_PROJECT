# Notebooks

Which notebook produced which results folder:

| Notebook | Platform | What it ran | Results folder |
|---|---|---|---|
| `FVLC_colab.ipynb` | Google Colab (T4) | First full run, anchor protocol: Tables 1–3 and the Figure 3 plots (Steps 1–8); template, outputs not saved | `results/` (incl. `results/<dataset>/fig3_*.png`) |
| `kaggle/kaggle_paired_cifar10.ipynb` | Kaggle (T4), actual run with outputs | Paired protocol, CIFAR-10: Tables 1, 2 and 3 | `results_paired/cifar10/` |
| `kaggle/kaggle_paired_cifar100.ipynb` | Kaggle (T4), actual run with outputs | Paired protocol, CIFAR-100: Tables 1 and 2 | `results_paired/cifar100/` |
| `kaggle/kaggle_paired_cub.ipynb` | Kaggle (T4), actual run with outputs | Paired protocol, CUB: Tables 1 and 2 | `results_paired/cub/` |
| `kaggle/kaggle_extra_cifar10.ipynb` | Kaggle (T4), actual run with outputs | Table 3 under the seat protocol + dense Standard baseline | `results_seat/`, `results_dense/cifar10/` |
| `kaggle/kaggle_extra_cifar100.ipynb` | Kaggle (T4), actual run with outputs | Dense Standard baseline | `results_dense/cifar100/` |
| `kaggle/kaggle_figure3.ipynb` | Kaggle (T4), actual run with outputs | Figure 3: concept-weight plots for CIFAR-10, CIFAR-100, CUB | `results_fig3/` |
| `FVLC_full_run.ipynb` | Colab | Optional: everything (all 3 protocols, all datasets) in one resumable run | `results_anchor/`, `results_seat/`, `results_paired/` |

We moved from Colab to Kaggle because Colab kept disconnecting, and ran one dataset per notebook.
All runs use seed 42. `make_tables.py` turns any results folder into `tables.md` and `fig_stability.png`,
and `results/comparison/` holds the side-by-side paper-vs-ours images.
