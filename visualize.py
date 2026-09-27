"""Figure 3 of the paper: concept weights for one test image, before/after perturbation and after FVLC.

  python visualize.py --dataset cifar10 --perturbation WP1_10+WP2+IP --index 0 --topn 10
Writes results/<dataset>/fig3_<perturbation>_<index>.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fvlc.cbm import ConceptLayer, FinalLayer
from fvlc.config import PERTURBATIONS, get_config
from fvlc.data import get_classes, get_dataset
from fvlc.features import image_features
from fvlc.utils import get_device, read_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--perturbation", default="WP1_10+WP2+IP", choices=list(PERTURBATIONS))
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = get_config(a.dataset)
    d = os.path.join(a.out, a.dataset)
    s = torch.load(os.path.join(d, "vanilla.pt"))
    vanilla, final = ConceptLayer(**s["layer"]), FinalLayer(**s["final"])
    base = ConceptLayer(**torch.load(os.path.join(d, "layers", f"{a.perturbation}_base.pt")))
    fvlc = ConceptLayer(**torch.load(os.path.join(d, "layers", f"{a.perturbation}_fvlc.pt")))
    concepts, classes = s["concepts"], get_classes(a.dataset)
    ip = PERTURBATIONS[a.perturbation]["ip"]
    dev = get_device()
    clean = image_features(cfg, "test", dev)["backbone"][a.index:a.index + 1]
    pert = image_features(cfg, "test", dev, cfg["ip_sigma"] if ip else 0.0)["backbone"][a.index:a.index + 1]
    pert_names = read_lines(os.path.join(d, f"concepts_{a.perturbation}.txt")) \
        if os.path.exists(os.path.join(d, f"concepts_{a.perturbation}.txt")) else concepts

    rows = [("c  (vanilla, clean)", vanilla(clean)[0], concepts),
            ("c + perturbation (LF-CBM)", base(pert)[0], pert_names),
            ("c~ + perturbation (FVLC)", fvlc(pert)[0], concepts)]
    img, label = get_dataset(a.dataset, "test")[a.index]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), gridspec_kw={"width_ratios": [1, 1.6, 1.6, 1.6]})
    axes[0].imshow(img)
    axes[0].set_title(f"true: {classes[label]}")
    axes[0].axis("off")
    for ax, (title, c, names) in zip(axes[1:], rows):
        pred = final(c[None]).argmax().item()
        contrib = c * final.W[pred]  # contribution of each concept to the predicted class logit
        top = torch.topk(contrib, a.topn).indices.flip(0)
        ax.barh(range(a.topn), contrib[top].tolist(), color=["#4f81bd" if v > 0 else "#c0504d" for v in contrib[top]])
        ax.set_yticks(range(a.topn), [names[i][:32] for i in top.tolist()], fontsize=8)
        ax.set_title(f"{title}\npred: {classes[pred]}", fontsize=9)
    fig.tight_layout()
    path = os.path.join(d, f"fig3_{a.perturbation}_{a.index}.png")
    fig.savefig(path, dpi=150)
    print("Wrote", path)


if __name__ == "__main__":
    main()
