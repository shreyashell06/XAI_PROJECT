"""Datasets and the input-perturbation (IP) wrapper.

Links
  CIFAR-10/100 : downloaded automatically by torchvision (https://www.cs.toronto.edu/~kriz/cifar.html)
  CUB-200-2011 : https://data.caltech.edu/records/65de6-vp158  -> run `bash scripts/download_cub.sh`
"""
import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets

from .config import LABEL_FILES
from .utils import read_lines

DATA_ROOT = os.environ.get("FVLC_DATA_ROOT", os.path.expanduser("~/.cache/fvlc_data"))
CUB_ROOT = os.environ.get("FVLC_CUB_ROOT", "data/CUB")


def get_classes(dataset: str) -> list[str]:
    return read_lines(LABEL_FILES[dataset])


def get_dataset(dataset: str, split: str, transform=None):
    """split in {"train", "test"}. The test split is the 'val' split in Label-free CBM."""
    train = split == "train"
    if dataset == "cifar10":
        return datasets.CIFAR10(DATA_ROOT, train=train, download=True, transform=transform)
    if dataset == "cifar100":
        return datasets.CIFAR100(DATA_ROOT, train=train, download=True, transform=transform)
    if dataset == "cub":
        path = os.path.join(CUB_ROOT, "train" if train else "test")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} not found. Run `bash scripts/download_cub.sh` first.")
        return datasets.ImageFolder(path, transform=transform)
    raise ValueError(dataset)


def get_targets(dataset: str, split: str) -> torch.Tensor:
    return torch.LongTensor(get_dataset(dataset, split).targets)


class GaussianNoiseDataset(Dataset):
    """IP perturbation (Sec. 6.1): x' = x + N(0, sigma^2), added *after* pixel normalisation.

    Noise is generated from a per-index generator, so every model (backbone and CLIP) sees
    exactly the same noise for the same image, and runs are reproducible.
    """

    def __init__(self, base: Dataset, sigma: float, seed: int = 0):
        self.base, self.sigma, self.seed = base, sigma, seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.sigma > 0:
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(idx))
            x = x + self.sigma * torch.randn(x.shape, generator=g, dtype=x.dtype)
        return x, y
