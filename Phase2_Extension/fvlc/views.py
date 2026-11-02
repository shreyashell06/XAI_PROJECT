"""Phase 2: label-free image edits ("views") for EC-FVLC.

Each view edits the raw PIL image *before* model preprocessing, so the backbone and every CLIP model see the
same edited image. Edits are seeded per (seed, image index, view), so they are identical across runs, models
and sessions.

  evidence-changing (the concept evidence in the image changes):
    occlude   a box covering 25-50% of the image is filled with the ImageNet mean colour (Cutout, DeVries &
              Taylor 2017)
    occlude_small  the same with an 8-15% box, roughly the size of one bird part (aimed at locality)
    crop      a random region covering 20-50% of the area is cropped and resized back (random-resized crop)
    cutmix    a 30-50% box is replaced by the same box of another training image (CutMix, Yun et al. 2019)
  evidence-preserving (nuisance):
    nuisance  brightness/contrast +-10%, Gaussian pixel noise (sigma 0.01-0.05 on [0, 1]), JPEG quality 40-80
  clean      no edit (used for oracle features of the original image)
"""
import io

import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

VIEW_IDS = {"clean": 0, "occlude": 1, "crop": 2, "cutmix": 3, "nuisance": 4, "occlude_small": 5}
EVIDENCE_VIEWS = ("occlude", "crop", "cutmix", "occlude_small")
NUISANCE_VIEWS = ("nuisance",)
DEFAULT_VIEWS = ("occlude", "crop", "cutmix", "nuisance")
MEAN_RGB = (124, 116, 104)


def _box(rng, w, h, lo, hi, aspect=(0.5, 2.0)):
    area = rng.uniform(lo, hi) * w * h
    r = np.exp(rng.uniform(np.log(aspect[0]), np.log(aspect[1])))
    bw = int(round(min(w, max(1, np.sqrt(area * r)))))
    bh = int(round(min(h, max(1, np.sqrt(area / r)))))
    x0 = int(rng.integers(0, w - bw + 1))
    y0 = int(rng.integers(0, h - bh + 1))
    return x0, y0, x0 + bw, y0 + bh


def apply_view(img: Image.Image, view: str, rng: np.random.Generator, other=None) -> Image.Image:
    """img: RGB PIL image. other: callable(rng) -> another RGB PIL image (only for cutmix)."""
    if view == "clean":
        return img
    w, h = img.size
    if view == "occlude":
        out = img.copy()
        out.paste(MEAN_RGB, _box(rng, w, h, 0.25, 0.5))
        return out
    if view == "occlude_small":
        out = img.copy()
        out.paste(MEAN_RGB, _box(rng, w, h, 0.08, 0.15))
        return out
    if view == "crop":
        return img.crop(_box(rng, w, h, 0.2, 0.5, aspect=(3 / 4, 4 / 3))).resize((w, h), Image.BILINEAR)
    if view == "cutmix":
        donor = other(rng).convert("RGB").resize((w, h), Image.BILINEAR)
        b = _box(rng, w, h, 0.3, 0.5)
        out = img.copy()
        out.paste(donor.crop(b), b[:2])
        return out
    if view == "nuisance":
        out = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.1))
        out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.9, 1.1))
        a = np.asarray(out, dtype=np.float32) / 255.0
        a = a + rng.normal(0, rng.uniform(0.01, 0.05), a.shape).astype(np.float32)
        out = Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=int(rng.integers(40, 81)))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(view)


class ViewDataset(Dataset):
    """Wraps a dataset that returns (PIL image, label) and returns (transform(edited image), label)."""

    def __init__(self, base: Dataset, view: str, transform, seed: int = 0):
        self.base, self.view, self.transform, self.seed = base, view, transform, seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]
        img = img.convert("RGB")
        rng = np.random.default_rng([self.seed, int(i), VIEW_IDS[self.view]])

        def other(r):
            j = int(r.integers(0, len(self.base) - 1))
            return self.base[j + (j >= i)][0]
        img = apply_view(img, self.view, rng, other)
        return (self.transform(img) if self.transform else img), y
