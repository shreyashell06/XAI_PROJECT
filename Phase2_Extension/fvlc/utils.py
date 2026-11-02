import json
import os
import random



def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix every source of randomness we control (Python, NumPy, PyTorch CPU/GPU)."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str | None = None) -> str:
    import torch
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f.read().split("\n") if l.strip()]


def write_lines(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
