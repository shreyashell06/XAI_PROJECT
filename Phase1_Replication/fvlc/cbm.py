"""Label-free CBM steps 2-4: concept bottleneck layer (CBL) W_c and sparse final layer W_F.

Re-implemented from Oikarinen et al. (2023), https://github.com/Trustworthy-ML-Lab/Label-free-CBM
(train_cbm.py / similarity.py). Changes: works on in-memory tensors, seeded, returns objects.
"""
import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from .third_party.glm_saga import IndexedTensorDataset, glm_saga


def cos_similarity_cubed_single(clip_feats, target_feats):
    """sim(P_i, q_i) of Eq. (1): cosine similarity of mean-centred, cubed activation patterns.
    Column i of `clip_feats` (M_{:,i}) is compared only with column i of `target_feats` (q_i)."""
    clip_feats = clip_feats.float() - clip_feats.float().mean(dim=0, keepdim=True)
    target_feats = target_feats - target_feats.mean(dim=0, keepdim=True)
    clip_feats, target_feats = clip_feats ** 3, target_feats ** 3
    clip_feats = clip_feats / clip_feats.norm(p=2, dim=0, keepdim=True)
    target_feats = target_feats / target_feats.norm(p=2, dim=0, keepdim=True)
    return (target_feats * clip_feats).sum(dim=0)


@dataclass
class ConceptLayer:
    """g(x) = (W_c b_f(x) - mu) / sigma   (standardisation as in Label-free CBM)."""
    W: torch.Tensor      # (k, d)
    mean: torch.Tensor   # (1, k)
    std: torch.Tensor    # (1, k)

    def __call__(self, feats):
        return (feats @ self.W.T - self.mean) / self.std

    def to(self, device):
        return ConceptLayer(self.W.to(device), self.mean.to(device), self.std.to(device))

    def state(self):
        return {"W": self.W.cpu(), "mean": self.mean.cpu(), "std": self.std.cpu()}


@dataclass
class FinalLayer:
    W: torch.Tensor  # (C, k)
    b: torch.Tensor  # (C,)

    def __call__(self, c):
        return c @ self.W.T + self.b

    def to(self, device):
        return FinalLayer(self.W.to(device), self.b.to(device))

    def state(self):
        return {"W": self.W.cpu(), "b": self.b.cpu()}


def clip_cutoff_mask(M_train, cutoff):
    """Delete concepts whose mean top-5 CLIP activation over the train set is <= cutoff."""
    highest = torch.topk(M_train, dim=0, k=5)[0].mean(dim=0)
    return highest > cutoff


def learn_projection(feats, M, val_feats, val_M, steps, batch_size, lr, device, seed, init=None, verbose=True):
    """Minimise Eq. (1) -sum_i sim(M_{:,i}, q_i) over W_c with Adam; early stopping on val (every 50 steps).

    Returns the best W_c (k, d) on CPU and the best validation similarity.
    """
    g = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    k, d = M.shape[1], feats.shape[1]
    proj = torch.nn.Linear(d, k, bias=False)
    if init is None:
        bound = 1.0 / d ** 0.5  # same distribution as nn.Linear default, but from our seeded generator
        with torch.no_grad():
            proj.weight.uniform_(-bound, bound, generator=g)
    else:
        with torch.no_grad():
            proj.weight.copy_(init)
    proj = proj.to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)
    feats_d, M_d = feats.to(device), M.to(device)
    val_feats_d, val_M_d = val_feats.to(device), val_M.to(device)
    bs = min(batch_size, len(feats))
    idx_all = list(range(len(feats)))

    best_val, best_w, best_step = float("inf"), None, 0
    for i in range(steps):
        batch = torch.LongTensor(rng.sample(idx_all, k=bs)).to(device)
        loss = -cos_similarity_cubed_single(M_d[batch], proj(feats_d[batch])).mean()
        loss.backward()
        opt.step()
        if i % 50 == 0 or i == steps - 1:
            with torch.no_grad():
                val_loss = -cos_similarity_cubed_single(val_M_d, proj(val_feats_d)).mean().item()
            if i == 0 or val_loss < best_val:
                best_val, best_w, best_step = val_loss, proj.weight.detach().clone(), i
            else:  # stop as soon as val similarity decreases (Label-free CBM behaviour)
                break
        opt.zero_grad()
    if verbose:
        print(f"  projection: best step {best_step}, val similarity {-best_val:.4f}")
    return best_w.cpu(), -best_val


@torch.no_grad()
def interpretability_mask(W, val_feats, val_M, cutoff):
    sim = cos_similarity_cubed_single(val_M, val_feats @ W.T)
    return sim > cutoff, sim


@torch.no_grad()
def standardise(W, train_feats):
    c = train_feats @ W.T
    return ConceptLayer(W, c.mean(dim=0, keepdim=True), c.std(dim=0, keepdim=True))


def train_sparse_final(train_x, train_y, val_x, val_y, n_classes, lam, n_iters, batch_size, device, seed):
    """Sparse elastic-net final layer (GLM-SAGA), identical settings to Label-free CBM.
    Returns FinalLayer and the solver's metrics dict."""
    torch.manual_seed(seed)
    train_loader = DataLoader(IndexedTensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    linear = torch.nn.Linear(train_x.shape[1], n_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    metadata = {"max_reg": {"nongrouped": lam}}
    out = glm_saga(linear, train_loader, 0.1, n_iters, 0.99, epsilon=1, k=1, val_loader=val_loader,
                   do_zero=False, metadata=metadata, n_ex=len(train_x), n_classes=n_classes)
    p = out["path"][0]
    W, b = p["weight"], p["bias"]
    nnz = (W.abs() > 1e-5).sum().item()
    return FinalLayer(W, b), {"metrics": p["metrics"], "nnz": nnz, "total": W.numel()}
