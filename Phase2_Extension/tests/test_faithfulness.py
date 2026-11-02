"""CPU tests for the Phase 2 faithfulness metrics, on synthetic data and a tiny fake CUB (no downloads).
Run:  python tests/test_faithfulness.py   (or python -m pytest -q tests)"""
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fvlc import faithfulness as Fa  # noqa: E402
from fvlc import metrics as M  # noqa: E402
from fvlc.cbm import ConceptLayer, FinalLayer, standardise  # noqa: E402
from fvlc.cub_meta import CUBMeta, ConceptMap, parse_concept  # noqa: E402

PARTS = ["back", "beak", "belly", "breast", "crown", "forehead", "left eye", "left leg", "left wing", "nape",
         "right eye", "right leg", "right wing", "tail", "throat"]


def test_auroc():
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=200), rng.integers(0, 2, 200)
    s[::7] = s[0]  # ties
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]])
    assert abs(Fa.auroc(s, y) - brute) < 1e-9
    assert Fa.auroc(np.zeros(10), [0, 1] * 5) == 0.5


def test_parse_concept():
    m = parse_concept("a brown back and white belly")
    assert m.groups == [(("back",), ("brown",)), (("belly",), ("white",))] and m.kp_parts == ("back", "belly")
    assert parse_concept("beak and legs are black").groups == [(("bill",), ("black",)), (("leg",), ("black",))]
    assert parse_concept("a dark, glossy bill").groups == [] and parse_concept("a dark, glossy bill").kp_parts
    assert parse_concept("a duck-like bird").kp_parts == ()
    assert parse_concept("a yellow body with black wings").groups == [(("wing",), ("black",))]


def test_constant_and_decision():
    g = torch.Generator().manual_seed(0)
    feats = torch.randn(300, 16, generator=g)
    layer = standardise(torch.randn(8, 16, generator=g), feats)
    final = FinalLayer(torch.randn(3, 8, generator=g) * (torch.rand(3, 8, generator=g) > 0.5), torch.zeros(3))
    const = Fa.constant_layer(layer, feats)
    assert torch.allclose(const(feats[:5]), layer(feats).mean(0, keepdim=True).expand(5, -1), atol=1e-5)
    wc = Fa.worst_case_stability(const, final, feats, feats, [0.1])
    assert wc["0.1"]["tcpc"] == 0.0 and wc["0.1"]["topk_overlap"] == 1.0   # perfectly "stable"
    wc_l = Fa.worst_case_stability(layer, final, feats, feats, [0.1])
    assert wc_l["0.1"]["tcpc"] > 0
    d = Fa.decision_relevance(layer, final, feats, k=3)
    assert all(0 <= v <= 1 for v in d.values())


def _fake_cub(root, n=24, size=200):
    """Images with a coloured square at the 'back' keypoint and another at the 'tail' keypoint. The back square's
    brightness is the 'black back' evidence; the tail square is independent."""
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(root, "meta"), exist_ok=True)
    lines = {k: [] for k in ["images", "bbox", "locs", "attr_labels"]}
    samples, back_val = [], []
    for i in range(1, n + 1):
        rel = f"001.Fake/img_{i:03d}.png"
        path = os.path.join(root, "test", rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img = np.full((size, size, 3), 120, np.uint8)
        bx, by = rng.integers(40, 80), rng.integers(40, 160)      # back keypoint, left half
        tx, ty = rng.integers(130, 160), rng.integers(40, 160)    # tail keypoint, right half
        v = int(rng.integers(0, 2))
        img[by - 12:by + 12, bx - 12:bx + 12] = 255 if v else 0
        img[ty - 12:ty + 12, tx - 12:tx + 12] = rng.integers(0, 255)
        Image.fromarray(img).save(path)
        samples.append((path, 0))
        back_val.append(v)
        lines["images"].append(f"{i} {rel}")
        lines["bbox"].append(f"{i} 20 20 160 160")
        for p in range(1, 16):
            x, y, vis = (bx, by, 1) if PARTS[p - 1] == "back" else (tx, ty, 1) if PARTS[p - 1] == "tail" else (0, 0, 0)
            lines["locs"].append(f"{i} {p} {x} {y} {vis}")
        lines["attr_labels"].append(f"{i} 1 {v} 4 0.0")
        lines["attr_labels"].append(f"{i} 2 {1 - v} 4 0.0")
    w = lambda name, rows: open(os.path.join(root, "meta", name), "w").write("\n".join(rows) + "\n")  # noqa: E731
    w("images.txt", lines["images"])
    w("bounding_boxes.txt", lines["bbox"])
    w("parts.txt", [f"{i} {p}" for i, p in enumerate(PARTS, 1)])
    w("part_locs.txt", lines["locs"])
    w("attributes.txt", ["1 has_back_color::white", "2 has_back_color::black"])
    w("image_attribute_labels.txt", lines["attr_labels"])
    return samples, back_val


def test_locality_and_attribute_auroc_on_fake_cub():
    from torchvision import transforms
    with tempfile.TemporaryDirectory() as root:
        samples, back_val = _fake_cub(root)
        meta = CUBMeta(os.path.join(root, "meta"))
        ids = meta.image_ids(samples)
        pre = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                  transforms.Normalize([0.5] * 3, [0.5] * 3)])
        # "backbone": mean intensity of an 8x8 grid of patches -> 64-d features
        feature_fn = lambda x: torch.nn.functional.adaptive_avg_pool2d(x.mean(1, keepdim=True), 8).flatten(1)  # noqa
        cmaps = [parse_concept("a white back"), parse_concept("a black tail"), ConceptMap("a bird")]
        variants, canon, cvar = Fa.locality_variants(cmaps)
        cache = Fa.cache_partmask_features(feature_fn, pre, samples, ids, meta, variants, mask_frac=0.2,
                                           num_workers=0)
        F0 = cache["feats"][:, 0].float()
        # a grounded layer: concept 0 = brightness at the back keypoint's patch (read from the clean image),
        # concept 1 and 2 = noise directions. Built by regressing the true back value on the features.
        y = torch.tensor(back_val, dtype=torch.float32)
        # ridge regression in closed form (torch.linalg.lstsq on this under-determined system differs across
        # LAPACK builds, e.g. on Kaggle)
        X = F0 - F0.mean(0)
        Wb = torch.linalg.solve(X.T @ X + 1e-2 * torch.eye(X.shape[1]), X.T @ (y - y.mean()))[None]
        W = torch.cat([Wb, torch.randn(2, F0.shape[1]) * 0.01])
        layer = standardise(W, F0)
        const = Fa.constant_layer(layer, F0)
        loc = Fa.locality(layer, cache, meta, cmaps, canon, cvar, min_images=5)
        loc_c = Fa.locality(const, cache, meta, cmaps, canon, cvar, min_images=5)
        assert loc["all"]["n_concepts"] == 2
        assert loc_c["all"]["locality"] == 0.0
        labels = Fa.concept_label_matrix(meta, cmaps, ids)
        assert labels[:, 0].tolist() == back_val and (labels[:, 2] == -1).all()
        au = Fa.attribute_auroc(layer, F0, labels, [0] * len(ids), min_pos=3)
        au_c = Fa.attribute_auroc(const, F0, labels, [0] * len(ids), min_pos=3)
        print("fake CUB  grounded:", loc["all"], au, "\n          constant:", loc_c["all"], au_c)
        assert au["overall"] > 0.8 and au_c["overall"] == 0.5
        assert loc["all"]["locality"] > 0 and loc["all"]["relevant_abs"] > loc["all"]["irrelevant_abs"]


if __name__ == "__main__":
    test_auroc()
    test_parse_concept()
    test_constant_and_decision()
    test_locality_and_attribute_auroc_on_fake_cub()
    print("all faithfulness tests passed")
