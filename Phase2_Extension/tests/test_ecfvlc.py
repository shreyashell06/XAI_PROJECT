"""CPU tests for Phase 2 EC-FVLC: image edits, the L5 loss and the trainer (synthetic data, no downloads).
Run:  python tests/test_ecfvlc.py   (or python -m pytest -q tests)"""
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fvlc.cbm import FinalLayer, standardise  # noqa: E402
from fvlc.config import get_config  # noqa: E402
from fvlc.fvlc import FVLCTrainer, concept_infidelity, evidence_targets  # noqa: E402
from fvlc.views import DEFAULT_VIEWS, ViewDataset, apply_view  # noqa: E402


def test_views():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (40, 60, 3)).astype(np.uint8))
    base = [(img, 0), (Image.fromarray(np.zeros((30, 30, 3), np.uint8)), 1)]
    for v in DEFAULT_VIEWS + ("clean",):
        a = ViewDataset(base, v, None, seed=1)[0][0]
        b = ViewDataset(base, v, None, seed=1)[0][0]
        assert a.size == img.size and np.array_equal(np.asarray(a), np.asarray(b)), v   # same size, deterministic
    x = np.asarray(img, np.float32)
    occ = np.asarray(apply_view(img, "occlude", np.random.default_rng(0)), np.float32)
    frac = (np.abs(occ - x).sum(2) > 0).mean()
    assert 0.2 < frac < 0.55, frac
    yy, xx = np.mgrid[0:40, 0:60]
    smooth = Image.fromarray(np.stack([xx * 4, yy * 6, (xx + yy) * 2], 2).astype(np.uint8))   # natural-ish image
    nui = np.asarray(apply_view(smooth, "nuisance", np.random.default_rng(0)), np.float32)
    assert np.abs(nui - np.asarray(smooth, np.float32)).mean() < 20


def test_l5_definition():
    g = torch.Generator().manual_seed(0)
    c = torch.randn(5, 7, generator=g)
    cv = torch.randn(5, 3, 7, generator=g)
    assert concept_infidelity(c, cv, cv - c.unsqueeze(1)).item() < 1e-12            # matches the evidence
    assert abs(concept_infidelity(c, cv, torch.zeros_like(cv)).item()
               - (cv - c.unsqueeze(1)).pow(2).mean().item()) < 1e-6                  # ds = 0 -> stability term
    assert concept_infidelity(c, c.unsqueeze(1).expand(-1, 3, -1), torch.ones(5, 3, 7)).item() == 1.0  # constant
    # evidence targets: identical views -> zero target
    clip = torch.randn(5, 16, generator=g)
    t = evidence_targets(clip, clip.unsqueeze(1).repeat(1, 2, 1), torch.randn(7, 16, generator=g),
                         torch.ones(7), torch.zeros(7), torch.ones(7))
    assert t.abs().max() < 1e-5


def _synthetic(n=1500, d=24, k=8, C=3, seed=0):
    """Backbone feats x; 'CLIP evidence' s = x @ T. Edited views zero a random half of the dimensions, which
    changes the evidence. The vanilla layer reads T plus a class-leakage direction."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, C, (n,), generator=g)
    means = torch.randn(C, d, generator=g) * 2
    x = means[y] + torch.randn(n, d, generator=g)
    T = torch.randn(k, d, generator=g) / d ** 0.5
    masks = (torch.rand(n, 3, d, generator=g) > 0.5).float()
    xv = x.unsqueeze(1) * masks
    s, sv = x @ T.T, xv @ T.T
    leak = means[torch.randint(0, C, (k,), generator=g)] / d ** 0.5
    vanilla = standardise(T + 0.7 * leak, x)
    return x, xv, s, sv, y, vanilla


def test_ecfvlc_tracks_evidence_better():
    x, xv, s, sv, y, vanilla = _synthetic()
    tr, te = slice(0, 1200), slice(1200, None)
    final = FinalLayer(torch.randn(3, 8, generator=torch.Generator().manual_seed(1)), torch.zeros(3))
    s_std = s[tr].std(0)
    target = (sv - s.unsqueeze(1)) / s_std                       # a_j = 1 for simplicity
    c_ref = vanilla(x[tr])
    out = {}
    for lam in [0.0, 0.01, 0.1, 1.0]:
        cfg = get_config("cifar10", fvlc_epochs=40, fvlc_batch_size=200, fvlc_lr=1e-2, k1=3, k2=3,
                         fvlc_init="gaussian", lambda5=lam, pgd_steps=2, x_pgd_steps=2)
        trainer = FVLCTrainer(cfg, final, vanilla, vanilla, "cpu", seed=0, verbose=False)
        layer, hist = trainer.fit(x[tr], c_ref, final(c_ref), views=dict(feats=xv[tr], target=target[tr]))
        assert hist[-1]["loss"] < hist[0]["loss"]
        with torch.no_grad():
            dg = layer(xv[te].reshape(-1, x.shape[1])).reshape(-1, 3, 8) - layer(x[te]).unsqueeze(1)
            ds = sv[te] - s[te].unsqueeze(1)
            a, b = dg.reshape(-1, 8), ds.reshape(-1, 8)
            a, b = a - a.mean(0), b - b.mean(0)
            corr = ((a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0))).mean().item()
            agree = (final(layer(x[te])).argmax(1) == final(vanilla(x[te])).argmax(1)).float().mean().item()
        out[lam] = (corr, agree, hist[-1]["L5"])
    print("lambda5 -> (evidence corr on held-out, agreement with vanilla, L5):", out)
    # a small lambda5 tracks the evidence much better at no cost in agreement with the vanilla model ...
    assert out[0.01][0] > out[0.0][0] + 0.1 and out[0.01][1] > 0.85
    # ... and a large one trades agreement (the vanilla model's class leakage) for evidence tracking
    assert out[1.0][0] > out[0.01][0] and out[1.0][1] < out[0.01][1]


if __name__ == "__main__":
    test_views()
    test_l5_definition()
    test_ecfvlc_tracks_evidence_better()
    print("all EC-FVLC tests passed")
