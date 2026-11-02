"""Fast CPU tests on synthetic data (no downloads). Run:  python -m pytest -q tests  (or python tests/test_core.py)"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fvlc import metrics as M  # noqa: E402
from fvlc.cbm import (clip_cutoff_mask, cos_similarity_cubed_single, interpretability_mask,  # noqa: E402
                      learn_projection, standardise, train_sparse_final)
from fvlc.config import get_config  # noqa: E402
from fvlc.fvlc import FVLCTrainer, kl, topk_surrogate  # noqa: E402


def synthetic(n=1200, d=32, k=12, C=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    W_true = torch.randn(k, d, generator=g)
    y = torch.randint(0, C, (n,), generator=g)
    means = torch.randn(C, d, generator=g) * 2
    feats = means[y] + torch.randn(n, d, generator=g)
    Mclip = torch.tanh(feats @ W_true.T / d ** 0.5) * 0.3 + 0.2  # fake CLIP activation matrix
    return feats, Mclip, y


def test_surrogate_and_kl():
    c = torch.randn(8, 20)
    assert topk_surrogate(c, c, 5).item() == 0
    assert topk_surrogate(c, c + 1, 5).item() > 0
    assert abs(kl(c, c).item()) < 1e-6
    assert M.topk_overlap(c, c, 5) == 1.0
    assert M.tcpc(c, c) == 0 and abs(M.tcpc(c, 2 * c) - 1) < 1e-6


def test_lfcbm_pipeline():
    feats, Mclip, y = synthetic()
    tr, te = slice(0, 1000), slice(1000, 1200)
    keep = clip_cutoff_mask(Mclip[tr], 0.0)
    assert keep.all()
    W, sim = learn_projection(feats[tr], Mclip[tr], feats[te], Mclip[te], steps=300, batch_size=500,
                              lr=1e-2, device="cpu", seed=0, verbose=False)
    assert sim > 0.5, sim
    mask, _ = interpretability_mask(W, feats[te], Mclip[te], 0.0)
    layer = standardise(W[mask], feats[tr])
    final, info = train_sparse_final(layer(feats[tr]), y[tr], layer(feats[te]), y[te], 4, lam=0.0007,
                                     n_iters=50, batch_size=256, device="cpu", seed=0)
    acc = M.accuracy(final(layer(feats[te])), y[te])
    assert acc > 0.6, acc
    return feats, Mclip, y, layer, final


def test_fvlc_reduces_instability():
    feats, Mclip, y, vanilla, final = test_lfcbm_pipeline()
    tr, te = slice(0, 1000), slice(1000, 1200)
    # "perturbed" base: different seed + noisy CLIP matrix
    Mp = Mclip + 0.05 * torch.randn(Mclip.shape, generator=torch.Generator().manual_seed(1))
    W, _ = learn_projection(feats[tr], Mp[tr][:, :vanilla.W.shape[0]], feats[te], Mp[te][:, :vanilla.W.shape[0]],
                            steps=300, batch_size=500, lr=1e-2, device="cpu", seed=1, verbose=False)
    base = standardise(W, feats[tr])
    cfg = get_config("cifar10", fvlc_epochs=30, fvlc_batch_size=200, fvlc_lr=1e-2, k1=3, k2=3)
    for pgd_mode in ["universal", "per_sample"]:
        cfg["pgd_mode"] = pgd_mode
        trainer = FVLCTrainer(cfg, final, base, vanilla, "cpu", seed=0, verbose=False)
        c_ref = vanilla(feats[tr])
        layer, hist = trainer.fit(feats[tr], c_ref, final(c_ref))
        c0 = vanilla(feats[te])
        t_base, t_fvlc = M.tcpc(c0, base(feats[te])), M.tcpc(c0, layer(feats[te]))
        o_base = M.topc(final(c0), final(base(feats[te])))
        o_fvlc = M.topc(final(c0), final(layer(feats[te])))
        print(f"{pgd_mode}: TCPC base {t_base:.3e} -> FVLC {t_fvlc:.3e} | TOPC {o_base:.3e} -> {o_fvlc:.3e}")
        assert hist[-1]["loss"] < hist[0]["loss"]
        assert o_fvlc < o_base


if __name__ == "__main__":
    test_surrogate_and_kl()
    test_fvlc_reduces_instability()
    print("all tests passed")
