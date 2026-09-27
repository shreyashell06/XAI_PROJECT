"""Cache backbone features b_f(x), CLIP image embeddings E_I(x) and CLIP text embeddings E_T(p).

Everything downstream (CBL learning, final layer, FVLC, evaluation) works on these cached tensors,
which is exactly what Label-free CBM does and what makes the experiments cheap.
"""
import hashlib
import os

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import GaussianNoiseDataset, get_dataset
from .models import get_backbone, load_clip

ACT_DIR = os.environ.get("FVLC_ACT_DIR", "activations")


@torch.no_grad()
def _run(feature_fn, dataset, batch_size, device, num_workers):
    out = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.startswith("cuda"))
    for x, _ in tqdm(loader, desc="features", leave=False):
        out.append(feature_fn(x.to(device)).float().cpu())
    return torch.cat(out)


def _cached(path, compute):
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = compute()
    torch.save(t, path)
    return t


def image_features(cfg, split: str, device: str, ip_sigma: float = 0.0, limit: int | None = None):
    """Returns dict(backbone=(N,d) tensor, clip=(N,512) tensor) for the given split.

    ip_sigma > 0 applies the IP perturbation (same noise for both models).
    limit keeps only the first `limit` images (debug / smoke tests).
    """
    ds, bb = cfg["dataset"], cfg["backbone"]
    clip_tag = cfg["clip_name"].replace("/", "")
    tag = (f"_ip{ip_sigma:g}" if ip_sigma > 0 else "") + (f"_n{limit}" if limit else "")
    bb_path = os.path.join(ACT_DIR, ds, f"{split}_{bb}{tag}.pt")
    clip_path = os.path.join(ACT_DIR, ds, f"{split}_clip{clip_tag}{tag}.pt")

    def make(model_fn_and_pre):
        fn, pre = model_fn_and_pre
        data = get_dataset(ds, split, transform=pre)
        if ip_sigma > 0:
            data = GaussianNoiseDataset(data, ip_sigma, seed=cfg["seed"] + (0 if split == "train" else 1))
        if limit:
            data = Subset(data, range(min(limit, len(data))))
        return _run(fn, data, cfg["batch_size"], device, cfg["num_workers"])

    feats_bb = _cached(bb_path, lambda: make(get_backbone(bb, device)))

    def clip_fn():
        model, pre = load_clip(cfg["clip_name"], device)
        return make((lambda x: model.encode_image(x).float(), pre))
    feats_clip = _cached(clip_path, clip_fn)
    return {"backbone": feats_bb, "clip": feats_clip}


def text_features(cfg, concepts: list[str], device: str) -> torch.Tensor:
    """Raw (un-normalised) CLIP text embeddings E_T(p_j), cached by the content of the concept list."""
    import clip
    h = hashlib.md5("\n".join(concepts).encode()).hexdigest()[:10]
    path = os.path.join(ACT_DIR, "text", f"{cfg['clip_name'].replace('/', '')}_{h}.pt")

    @torch.no_grad()
    def compute():
        model, _ = load_clip(cfg["clip_name"], device)
        out = []
        for i in range(0, len(concepts), 256):
            tok = clip.tokenize(concepts[i:i + 256], truncate=True).to(device)
            out.append(model.encode_text(tok).float().cpu())
        return torch.cat(out)
    return _cached(path, compute)


def clip_activation_matrix(img_clip: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
    """M_{i,j} = E_I(x_i) . E_T(p_j) with both embeddings L2-normalised (Label-free CBM)."""
    img = img_clip / img_clip.norm(dim=1, keepdim=True)
    t = txt / txt.norm(dim=1, keepdim=True)
    return img @ t.T
