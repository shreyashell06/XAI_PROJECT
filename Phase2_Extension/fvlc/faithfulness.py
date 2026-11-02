"""Phase 2: two-sided faithfulness metrics for concept layers.

FVLC (Lai et al., ICLR 2024) measures only stability (TCPC/TOPC). A faithful concept score must also be
*sensitive*: it should change when the visual evidence for its concept changes (FVLC paper, Sec. 2 and 5).
This module measures both sides, plus whether the explanation is the one that drives the decision.

  constant_layer        an explainer that ignores its input: perfect TCPC/TOPC, no information
  decision_relevance    are the top-k activated concepts the ones the final layer uses for the predicted class?
  worst_case_stability  TCPC / top-k overlap under a test-time l_inf PGD attack on backbone features
  PartMaskDataset       CUB test images with body parts masked (human keypoints), for locality
  locality              Raman et al. (TMLR 2025) masking test: score should drop when the concept's own part is
                        masked (relevant) and stay put when a distant part is masked (irrelevant)
  attribute_auroc       does the concept score detect the human-annotated attribute? (all images, and within
                        each species, which gives no credit for merely recognising the class)

All functions take a ConceptLayer (fvlc.cbm) and cached backbone features, so any model (LF-CBM, FVLC,
EC-FVLC, ...) is evaluated the same way.
"""
import math
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .cbm import ConceptLayer
from .cub_meta import CANONICAL_GROUPS
from .fvlc import attack_features
from . import metrics as M


# --------------------------------------------------------------------------------------------- baselines
@torch.no_grad()
def constant_layer(layer: ConceptLayer, train_feats) -> ConceptLayer:
    """g(x) = mean_train g(x) for every x: W = 0, (0 - mean) / std = mean_c with mean = -mean_c, std = 1."""
    c_mean = layer(train_feats).mean(dim=0, keepdim=True)
    return ConceptLayer(torch.zeros_like(layer.W), -c_mean, torch.ones_like(c_mean))


# --------------------------------------------------------------------------------------------- decision
@torch.no_grad()
def decision_relevance(layer, final, feats, k=10, eps=1e-5):
    """Def. 2 stabilises the top-k concept *activations*; the class is decided by contributions W_F[y_hat, j] c_j.
    Returns
      nonzero_frac       share of the top-k activated concepts with a non-zero final weight for y_hat
      positive_frac      share with a positive contribution W_F[y_hat, j] * c_j > 0 (they argue *for* y_hat)
      act_vs_contrib     top-k overlap between activations c and contributions W_F[y_hat] * c
    """
    c = layer(feats)
    k = min(k, c.shape[1])
    y_hat = final(c).argmax(dim=1)
    w = final.W[y_hat]                                   # (N, k_concepts)
    top = torch.topk(c, k, dim=1).indices
    w_top = w.gather(1, top)
    contrib = w * c
    return dict(
        nonzero_frac=(w_top.abs() > eps).float().mean().item(),
        positive_frac=((w_top * c.gather(1, top)) > eps).float().mean().item(),
        act_vs_contrib=M.topk_overlap(c, contrib, k),
    )


# --------------------------------------------------------------------------------------------- stability
def worst_case_stability(layer, final, test_feats, train_feats, radii, steps=10, seed=0, device="cpu", k=10):
    """Same attack as the SEAT protocol of Phase 1 (pipeline.evaluate_attack), at several radii (in units of each
    feature's training std). An explainer that ignores its input scores 0 here by construction."""
    scale = train_feats.std(dim=0, keepdim=True)
    k = min(k, layer.W.shape[0])
    out = {}
    for R in radii:
        c0, y0, c1, y1 = attack_features(layer, final, test_feats, scale, R, steps, R / 5, seed=seed, device=device)
        out[f"{R:g}"] = dict(tcpc=M.tcpc(c0, c1), topc=M.topc(y0, y1), topk_overlap=M.topk_overlap(c0, c1, k),
                             pred_agreement=(y0.argmax(1) == y1.argmax(1)).float().mean().item())
    return out


# --------------------------------------------------------------------------------------------- locality
def _resize_crop_geometry(w, h, resize=256, crop=224):
    """Mirror torchvision Resize(resize) + CenterCrop(crop): returns (sx, sy, left, top)."""
    if w <= h:
        nw, nh = resize, int(resize * h / w)
    else:
        nw, nh = int(resize * w / h), resize
    left, top = int(round((nw - crop) / 2.0)), int(round((nh - crop) / 2.0))
    return nw / w, nh / h, left, top


class PartMaskDataset(Dataset):
    """For each image returns the clean image followed by one masked copy per variant.

    variants : list of tuples of keypoint part names; every visible keypoint of the variant gets a square box of
               side `mask_frac * sqrt(bbox area)` filled with 0 (= the ImageNet mean after normalisation).
    Also returns the keypoints in crop coordinates, visibility inside the crop, the box side, and which
    variants had at least one visible keypoint.
    """

    def __init__(self, samples, image_ids, meta, variants, preprocess, mask_frac=0.3, crop=224, resize=256):
        self.samples, self.ids, self.meta = samples, image_ids, meta
        self.variants, self.pre, self.frac = variants, preprocess, mask_frac
        self.crop, self.resize = crop, resize
        self.var_rows = [[meta.part_id[p] - 1 for p in v] for v in variants]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, n):
        path, _ = self.samples[n]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        x = self.pre(img)
        cid = self.ids[n]
        sx, sy, left, top = _resize_crop_geometry(w, h, self.resize, self.crop)
        kp = self.meta.kp[cid].clone()
        kp[:, 0] = kp[:, 0] * sx - left
        kp[:, 1] = kp[:, 1] * sy - top
        inside = (kp[:, 0] >= 0) & (kp[:, 0] < self.crop) & (kp[:, 1] >= 0) & (kp[:, 1] < self.crop)
        kp[:, 2] = ((kp[:, 2] > 0) & inside).float()
        bw, bh = self.meta.bbox[cid][2:]
        side = self.frac * math.sqrt(bw * bh) * math.sqrt(sx * sy)
        out, valid = [x], []
        for rows in self.var_rows:
            xm = x.clone()
            vis = [r for r in rows if kp[r, 2] > 0]
            for r in vis:
                cx, cy = kp[r, 0].item(), kp[r, 1].item()
                x0, x1 = int(max(0, cx - side / 2)), int(min(self.crop, cx + side / 2 + 1))
                y0, y1 = int(max(0, cy - side / 2)), int(min(self.crop, cy + side / 2 + 1))
                xm[:, y0:y1, x0:x1] = 0.0
            out.append(xm)
            valid.append(len(vis) > 0)
        return torch.stack(out), kp, torch.tensor(side), torch.tensor(valid)


def locality_variants(concept_maps):
    """Canonical part groups (used as 'irrelevant' masks) + every distinct keypoint set named by a concept
    (used as that concept's 'relevant' mask). Returns (variants, canonical_idx, concept_variant)."""
    variants = [tuple(v) for v in CANONICAL_GROUPS.values()]
    canonical_idx = list(range(len(variants)))
    concept_variant = []
    for m in concept_maps:
        if not m.kp_parts:
            concept_variant.append(-1)
            continue
        key = tuple(sorted(m.kp_parts))
        if key not in [tuple(sorted(v)) for v in variants]:
            variants.append(key)
        concept_variant.append([tuple(sorted(v)) for v in variants].index(key))
    return variants, canonical_idx, concept_variant


@torch.no_grad()
def cache_partmask_features(feature_fn, preprocess, samples, image_ids, meta, variants, path=None,
                            mask_frac=0.3, batch_size=8, num_workers=2, device="cpu"):
    """Backbone features of every (image, variant). Cached to `path` (fp16) so any concept layer can be scored
    later without touching the images again. Returns dict(feats=(N, 1+V, d), kp=(N, 15, 3), side=(N,),
    valid=(N, V), variants=list)."""
    if path and os.path.exists(path):
        cache = torch.load(path, map_location="cpu")
        if cache["variants"] == [tuple(v) for v in variants] and cache["mask_frac"] == mask_frac:
            return cache
    ds = PartMaskDataset(samples, image_ids, meta, variants, preprocess, mask_frac)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats, kps, sides, valids = [], [], [], []
    for x, kp, side, valid in tqdm(loader, desc="part-mask features", leave=False):
        B, V1 = x.shape[:2]
        f = feature_fn(x.flatten(0, 1).to(device)).float().cpu()
        feats.append(f.view(B, V1, -1).half())
        kps.append(kp)
        sides.append(side)
        valids.append(valid)
    cache = dict(feats=torch.cat(feats), kp=torch.cat(kps), side=torch.cat(sides), valid=torch.cat(valids),
                 variants=[tuple(v) for v in variants], mask_frac=mask_frac)
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(cache, path)
    return cache


def _far_groups(cache, meta, canonical_idx, rel_variant):
    """(N, len(canonical_idx)) bool: canonical group g is 'irrelevant' for the relevant keypoint set in image i if
    it has a visible keypoint and every visible keypoint of g is more than one box side away from every visible
    keypoint of the relevant set (so the two masks cannot overlap)."""
    kp, side = cache["kp"], cache["side"]
    rel_rows = [meta.part_id[p] - 1 for p in cache["variants"][rel_variant]]
    out = torch.zeros(len(kp), len(canonical_idx), dtype=torch.bool)
    for gi, g in enumerate(canonical_idx):
        g_rows = [meta.part_id[p] - 1 for p in cache["variants"][g]]
        if set(g_rows) & set(rel_rows):
            continue
        a = kp[:, rel_rows]                                  # (N, r, 3)
        b = kp[:, g_rows]                                    # (N, q, 3)
        d = torch.cdist(a[..., :2], b[..., :2])              # (N, r, q)
        both = (a[..., 2:3] > 0) & (b[..., 2].unsqueeze(1) > 0)
        d = torch.where(both, d, torch.full_like(d, float("inf")))
        any_b = (b[..., 2] > 0).any(dim=1)
        out[:, gi] = any_b & (d.amin(dim=(1, 2)) > side)
    return out


@torch.no_grad()
def locality(layer, cache, meta, concept_maps, canonical_idx, concept_variant, min_images=20, active_thr=1.0):
    """Part-masking locality (Raman et al., TMLR 2025), in standardised concept units.

    For every concept that names a body part and every image where that part is visible:
      relevant change   |c_j(x) - c_j(x with the concept's parts masked)|      (want: large)
      irrelevant change mean |c_j(x) - c_j(x with a distant part masked)|      (want: small)
      locality          relevant change - irrelevant change                    (want: > 0)
      relevant drop     signed c_j(x) - c_j(x masked); only meaningful on `active` images, where the concept is
                        strongly on (c_j(x) > active_thr, one std above its training mean) and hiding its part
                        should lower the score
    Averaged over images, then over concepts. Magnitudes are used for `all` because on an image where the
    concept is absent, hiding the part may legitimately raise or lower the score.
    """
    F = cache["feats"].float()
    N, V1, d = F.shape
    C = layer(F.view(-1, d)).view(N, V1, -1)
    c0, cv = C[:, 0], C[:, 1:]
    valid = cache["valid"]
    per = {"all": [], "active": []}
    far_cache = {}
    for j, v in enumerate(concept_variant):
        if v < 0:
            continue
        if v not in far_cache:
            far_cache[v] = _far_groups(cache, meta, canonical_idx, v)
        far = far_cache[v] & valid[:, canonical_idx]
        ok = valid[:, v] & far.any(dim=1)
        rel = c0[:, j] - cv[:, v, j]
        irr_all = (c0[:, j:j + 1] - cv[:, canonical_idx, j]).abs()
        irr = (irr_all * far).sum(dim=1) / far.sum(dim=1).clamp_min(1)
        for name, mask in (("all", ok), ("active", ok & (c0[:, j] > active_thr))):
            if mask.sum() < min_images:
                continue
            per[name].append(torch.stack([rel[mask].mean(), rel[mask].abs().mean(), irr[mask].mean()]))
    out = {}
    for name, rows in per.items():
        if not rows:
            out[name] = dict(n_concepts=0)
            continue
        r = torch.stack(rows)
        rel, rel_abs, irr = r.mean(dim=0).tolist()
        out[name] = dict(n_concepts=len(rows), relevant_drop=rel, relevant_abs=rel_abs, irrelevant_abs=irr,
                         locality=rel_abs - irr, ratio=rel_abs / max(irr, 1e-12))
    return out


# --------------------------------------------------------------------------------------------- AUROC
def auroc(scores, labels):
    """Mann-Whitney AUROC with average ranks for ties (a constant score gives exactly 0.5)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels) > 0.5
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def concept_label_matrix(meta, concept_maps, image_ids, min_certainty=3):
    """(N, k) labels in {1, 0, -1 (unknown)}; computed once per dataset and reused for every model."""
    return torch.stack([meta.concept_labels(m, image_ids, min_certainty) for m in concept_maps], dim=1)


@torch.no_grad()
def attribute_auroc(layer, test_feats, labels, targets, min_pos=10, min_within=3):
    """Does c_j(x) detect the human-annotated attribute named by concept j?
      overall      AUROC over all test images (per concept, averaged over concepts with >= min_pos of each label)
      within       AUROC among images of the same species (per (concept, species) pair with >= min_within of each
                   label, averaged). Recognising the species earns nothing here, only the attribute itself does.
    """
    c = layer(test_feats).cpu().numpy()
    L = labels.numpy()
    t = np.asarray(targets)
    overall, within = [], []
    for j in range(L.shape[1]):
        known = L[:, j] >= 0
        if (L[known, j] == 1).sum() >= min_pos and (L[known, j] == 0).sum() >= min_pos:
            overall.append(auroc(c[known, j], L[known, j]))
        for cls in np.unique(t[known]):
            m = known & (t == cls)
            if (L[m, j] == 1).sum() >= min_within and (L[m, j] == 0).sum() >= min_within:
                within.append(auroc(c[m, j], L[m, j]))
    return dict(overall=float(np.mean(overall)) if overall else float("nan"), n_concepts=len(overall),
                within_species=float(np.mean(within)) if within else float("nan"), n_pairs=len(within))
