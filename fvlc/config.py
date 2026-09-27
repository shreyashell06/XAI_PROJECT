"""Hyper-parameters for every dataset.

Values come from two sources and each one is tagged:
  [LF]   Label-free CBM repo `training_commands.txt` / `train_cbm.py` (Oikarinen et al., 2023)
  [FVLC] Table 5 of the FVLC paper (Lai et al., ICLR 2024)
  [OURS] Not specified in the paper -> our choice (documented in README "Ambiguities").
"""
from copy import deepcopy

COMMON = dict(
    seed=42,                      # [OURS] global random seed
    clip_name="ViT-B/16",         # [LF] CLIP model used to build the activation matrix M
    batch_size=512,               # [FVLC/LF] batch size for caching activations
    saga_batch_size=256,          # [FVLC/LF]
    proj_batch_size=5000,         # [FVLC] (LF default is 50000)
    proj_steps=1000,              # [FVLC/LF]
    proj_lr=1e-3,                 # [LF]
    interpretability_cutoff=0.45, # [FVLC/LF]
    num_workers=2,
    # ---------------- FVLC (Algorithm 1) ----------------
    pgd_radius=0.1,               # [FVLC] R2, radius of delta (concept space), l_inf
    pgd_steps=10,                 # [FVLC] P
    pgd_step_size=0.02,           # [FVLC] gamma
    x_pgd_radius=0.05,            # [FVLC] R1, radius of rho, l_inf
    x_pgd_steps=10,               # [FVLC] Q
    x_pgd_step_size=0.01,         # [FVLC] tau
    lambda1=1e-2, lambda2=1e-2, lambda3=1e-2, lambda4=1e-2,  # [FVLC]
    k1=10, k2=10,                 # [OURS] top-k sizes (not reported in the paper)
    fvlc_epochs=20,               # [OURS] outer SGD epochs (T = epochs * N / batch)
    fvlc_batch_size=512,          # [OURS]
    fvlc_lr=1e-3,                 # [OURS] Adam + cosine decay to 0
    fvlc_init="base",             # [OURS] "base" (start from perturbed LF-CBM W_c) or "gaussian" (Alg. 1 line 2)
    rho_space="feature",          # [OURS] where rho lives: "feature" (backbone feats, 'x pgd') or "concept"
    pgd_mode="universal",         # [FVLC] Alg. 1 averages grads over batches -> one delta/rho shared by all x
    # ---------------- Evaluation protocol (see README "Two evaluation protocols") ----------------
    protocol="anchor",            # "anchor": FVLC anchored to the clean vanilla model; TCPC/TOPC = change vs vanilla
                                  # "seat":   FVLC anchored to the LF-CBM of its own concept space (Def. 2) and
                                  #           TCPC/TOPC = change of the SAME model under a test-time PGD attack on
                                  #           its input features (the evaluation used in SEAT, Hu et al. AAAI'23,
                                  #           whose PGD code the official FVLC repo re-uses)
                                  # "paired": each method is compared with ITSELF trained without the perturbation
                                  #           (base: vanilla vs LF-CBM under P; FVLC: FVLC-clean vs FVLC under P).
                                  #           Use with fvlc_init="gaussian" (Algorithm 1, line 2).
    eval_radius=None,             # [OURS] test-time attack radius (std units); None -> x_pgd_radius
    eval_steps=10,                # [OURS] test-time PGD steps
    eval_step_size=None,          # [OURS] None -> x_pgd_step_size
    # ---------------- Perturbations (Sec. 6.1) ----------------
    wp2_sigma=1e-3,               # [FVLC] Gaussian noise on CLIP text embeddings
    ip_sigma=1e-3,                # [FVLC] Gaussian noise on normalised input pixels
    base_seed_offset=1,           # [OURS] perturbed LF-CBM is re-trained with seed+offset
)

DATASETS = {
    "cifar10": dict(
        backbone="clip_RN50", feature_layer=None,
        concept_set="data/concept_sets/cifar10_filtered.txt",
        clip_cutoff=0.25, lam=0.0007, n_iters=1000,           # [LF]
        std_lam=0.0007, std_n_iters=1000,                      # [LF] sparse standard model
    ),
    "cifar100": dict(
        backbone="clip_RN50", feature_layer=None,
        concept_set="data/concept_sets/cifar100_filtered.txt",
        clip_cutoff=0.25, lam=0.0007, n_iters=1000,
        std_lam=0.003, std_n_iters=1000,
    ),
    "cub": dict(
        backbone="resnet18_cub", feature_layer="features.final_pool",
        concept_set="data/concept_sets/cub_filtered.txt",
        clip_cutoff=0.26, lam=0.0002, n_iters=5000,
        std_lam=0.00002, std_n_iters=5000,
    ),
}

# Perturbation settings of Tables 1-3. Keys are used in result files.
PERTURBATIONS = {
    "SEED_ONLY":    dict(wp1=0.0,  wp2=False, ip=False),   # [OURS] control: only the training seed changes
    "WP1_5":        dict(wp1=0.05, wp2=False, ip=False),
    "WP1_10":       dict(wp1=0.10, wp2=False, ip=False),
    "WP2":          dict(wp1=0.0,  wp2=True,  ip=False),
    "IP":           dict(wp1=0.0,  wp2=False, ip=True),
    "WP1_5+WP2":    dict(wp1=0.05, wp2=True,  ip=False),
    "WP1_10+WP2":   dict(wp1=0.10, wp2=True,  ip=False),
    "WP1_10+WP2+IP": dict(wp1=0.10, wp2=True, ip=True),
}
PAPER_PERTURBATIONS = [k for k in PERTURBATIONS if k != "SEED_ONLY"]

# Ablation (Table 3): L1 is always on; toggle L2/L3/L4.
ABLATIONS = {
    "L1":          (False, False, False),
    "L1+L2":       (True,  False, False),
    "L1+L3":       (False, True,  False),
    "L1+L4":       (False, False, True),
    "L1+L2+L3":    (True,  True,  False),
    "L1+L2+L4":    (True,  False, True),
    "L1+L3+L4":    (False, True,  True),
    "L1+L2+L3+L4": (True,  True,  True),
}

LABEL_FILES = {
    "cifar10": "data/cifar10_classes.txt",
    "cifar100": "data/cifar100_classes.txt",
    "cub": "data/cub_classes.txt",
}


def get_config(dataset: str, **overrides) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset}; choose from {list(DATASETS)}")
    cfg = deepcopy(COMMON)
    cfg.update(deepcopy(DATASETS[dataset]))
    cfg["dataset"] = dataset
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg
