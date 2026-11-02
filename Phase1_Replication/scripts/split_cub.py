"""Split CUB_200_2011 into CUB/train and CUB/test (adapted from Label-free CBM's split_cub_dataset.py;
grayscale images are skipped, as in the original)."""
import os
import shutil

from PIL import Image

base = "CUB_200_2011"
with open(os.path.join(base, "images.txt")) as f:
    images = [l.split() for l in f if l.strip()]
with open(os.path.join(base, "train_test_split.txt")) as f:
    split = dict(l.split() for l in f if l.strip())
n = {"train": 0, "test": 0}
for img_id, rel in images:
    src = os.path.join(base, "images", rel)
    if Image.open(src).mode != "RGB":
        continue
    part = "train" if split[img_id] == "1" else "test"
    dst = os.path.join("CUB", part, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    n[part] += 1
print(n)
assert n == {"train": 5990, "test": 5790}, n
