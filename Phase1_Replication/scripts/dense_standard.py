"""Table 1 "Standard" row as a DENSE linear probe on the backbone features (no sparsity).

Label-free CBM's train_standard.py fits a *sparse* probe (lam=0.0007 / 0.003). The paper's Standard numbers
(88.80 / 70.10 / 76.70) match a dense probe better. This script fits both, reusing the cached features:

  python scripts/dense_standard.py --dataset cifar100 --out results_dense
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from fvlc import metrics as M
from fvlc.cbm import train_sparse_final
from fvlc.config import get_config
from fvlc.pipeline import Experiment
from fvlc.utils import get_device, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--out", default="results_dense")
ap.add_argument("--lams", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4])
ap.add_argument("--n_iters", type=int, default=200)
ap.add_argument("--limit_train", type=int, default=None)
a = ap.parse_args()

cfg = get_config(a.dataset)
dev = get_device()
exp = Experiment(cfg, dev, a.out, limit_train=a.limit_train, verbose=False)
tr, te = exp.feats("train")["backbone"], exp.feats("test")["backbone"]
mu, sd = tr.mean(0, keepdim=True), tr.std(0, keepdim=True)
res = {}
for lam in a.lams:
    set_seed(cfg["seed"])
    final, info = train_sparse_final((tr - mu) / sd, exp.y_train, (te - mu) / sd, exp.y_test, len(exp.classes),
                                     lam, a.n_iters, cfg["saga_batch_size"], dev, cfg["seed"])
    acc = M.accuracy(final((te - mu) / sd), exp.y_test)
    res[str(lam)] = dict(acc=acc, nonzero_frac=info["nnz"] / info["total"])
    print(f"[dense standard] {a.dataset} lam={lam:g}: test acc {100*acc:.2f}%  (non-zero weights {100*info['nnz']/info['total']:.0f}%)", flush=True)
os.makedirs(os.path.join(a.out, a.dataset), exist_ok=True)
json.dump(res, open(os.path.join(a.out, a.dataset, "dense_standard.json"), "w"), indent=2)
