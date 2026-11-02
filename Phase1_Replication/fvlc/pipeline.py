"""End-to-end experiment logic: vanilla LF-CBM, perturbed LF-CBM ("base"), FVLC, evaluation.

Protocol (our reading of Sec. 6 / Fig. 2, see README "Ambiguities"):
  vanilla : LF-CBM trained on clean data with the original concept set  -> g(x), y(x, c) = W_F g(x)
  base_P  : the CBL is re-learned under perturbation P (synonym-swapped concept words, noisy text
            embeddings and/or noisy input images) with a fresh seed; W_F and the concept indices are
            kept from the vanilla model                                     -> g_P(x)
  FVLC_P  : starting from base_P, W~c is trained with Eq. (2) / Algorithm 1 on the perturbed pipeline,
            anchored to the vanilla g(x), y(x, c)                             -> g~(x)
  TCPC/TOPC compare the vanilla concepts/outputs on the clean test set with base_P / FVLC_P on the
  (perturbed) test set; accuracy is measured on the (perturbed) test set.
"""
import os
import time

import torch

from . import metrics as M
from .cbm import (ConceptLayer, FinalLayer, clip_cutoff_mask, interpretability_mask, learn_projection,
                  standardise, train_sparse_final)
from .concepts import perturb_words
from .config import PERTURBATIONS
from .data import get_classes, get_targets
from .features import clip_activation_matrix, image_features, text_features
from .fvlc import FVLCTrainer, attack_features
from .utils import read_lines, set_seed, write_lines


class Experiment:
    def __init__(self, cfg, device, out_dir="results", limit_train=None, verbose=True):
        self.cfg, self.device, self.verbose = cfg, device, verbose
        self.out = os.path.join(out_dir, cfg["dataset"])
        os.makedirs(os.path.join(self.out, "layers"), exist_ok=True)
        self.limit = limit_train
        self.classes = get_classes(cfg["dataset"])
        self._feat_cache = {}
        self.y_train = self._targets("train")
        self.y_test = self._targets("test")
        if limit_train:
            self.y_train = self.y_train[:limit_train]
            self.y_test = self.y_test[:limit_train]

    def _targets(self, split):
        """Labels, cached next to the features so later sessions don't need the raw images (e.g. CUB)."""
        from .features import ACT_DIR
        path = os.path.join(ACT_DIR, self.cfg["dataset"], f"{split}_targets.pt")
        if os.path.exists(path):
            return torch.load(path)
        y = get_targets(self.cfg["dataset"], split)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(y, path)
        return y

    # ------------------------------------------------------------------ features
    def feats(self, split, ip=False):
        key = (split, ip)
        if key not in self._feat_cache:
            self._feat_cache[key] = image_features(self.cfg, split, self.device,
                                                   ip_sigma=self.cfg["ip_sigma"] if ip else 0.0, limit=self.limit)
        return self._feat_cache[key]

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ------------------------------------------------------------------ vanilla LF-CBM
    def build_vanilla(self, force=False):
        path = os.path.join(self.out, "vanilla.pt")
        if os.path.exists(path) and not force:
            s = torch.load(path)
            self.vanilla = ConceptLayer(**s["layer"])
            self.final = FinalLayer(**s["final"])
            self.concepts = s["concepts"]
            self.vanilla_info = s["info"]
            self.log(f"[vanilla] loaded ({len(self.concepts)} concepts, test acc {s['info']['acc']:.4f})")
            return self.vanilla_info

        cfg, dev = self.cfg, self.device
        set_seed(cfg["seed"])
        t0 = time.time()
        concepts = read_lines(cfg["concept_set"])
        tr, te = self.feats("train"), self.feats("test")
        txt = text_features(cfg, concepts, dev)
        M_tr, M_te = clip_activation_matrix(tr["clip"], txt), clip_activation_matrix(te["clip"], txt)

        keep1 = clip_cutoff_mask(M_tr, cfg["clip_cutoff"])
        concepts = [c for c, k in zip(concepts, keep1.tolist()) if k]
        M_tr, M_te = M_tr[:, keep1], M_te[:, keep1]
        self.log(f"[vanilla] {len(keep1)} concepts -> {len(concepts)} after CLIP cutoff")

        W, _ = learn_projection(tr["backbone"], M_tr, te["backbone"], M_te, cfg["proj_steps"],
                                cfg["proj_batch_size"], cfg["proj_lr"], dev, cfg["seed"], verbose=self.verbose)
        keep2, _ = interpretability_mask(W, te["backbone"], M_te, cfg["interpretability_cutoff"])
        W = W[keep2]
        concepts = [c for c, k in zip(concepts, keep2.tolist()) if k]
        self.log(f"[vanilla] {len(concepts)} concepts after interpretability cutoff")

        layer = standardise(W, tr["backbone"])
        final, finfo = train_sparse_final(layer(tr["backbone"]), self.y_train, layer(te["backbone"]), self.y_test,
                                          len(self.classes), cfg["lam"], cfg["n_iters"], cfg["saga_batch_size"],
                                          dev, cfg["seed"])
        acc = M.accuracy(final(layer(te["backbone"])), self.y_test)
        info = dict(acc=acc, n_concepts=len(concepts), nnz=finfo["nnz"], total_weights=finfo["total"],
                    time_s=time.time() - t0)
        torch.save(dict(layer=layer.state(), final=final.state(), concepts=concepts, info=info), path)
        write_lines(os.path.join(self.out, "concepts.txt"), concepts)
        self.vanilla, self.final, self.concepts, self.vanilla_info = layer, final, concepts, info
        self.log(f"[vanilla] test acc {acc:.4f}  ({finfo['nnz']}/{finfo['total']} non-zero final weights)")
        return info

    # ------------------------------------------------------------------ standard (no interpretability)
    def build_standard(self):
        cfg = self.cfg
        set_seed(cfg["seed"])
        tr, te = self.feats("train")["backbone"], self.feats("test")["backbone"]
        mu, sd = tr.mean(0, keepdim=True), tr.std(0, keepdim=True)
        final, finfo = train_sparse_final((tr - mu) / sd, self.y_train, (te - mu) / sd, self.y_test,
                                          len(self.classes), cfg["std_lam"], cfg["std_n_iters"],
                                          cfg["saga_batch_size"], self.device, cfg["seed"])
        acc = M.accuracy(final((te - mu) / sd), self.y_test)
        self.log(f"[standard] test acc {acc:.4f}")
        return dict(acc=acc, nnz=finfo["nnz"])

    # ------------------------------------------------------------------ perturbed LF-CBM (base)
    def build_base(self, pert_name):
        cfg, dev = self.cfg, self.device
        p = PERTURBATIONS[pert_name]
        seed = cfg["seed"] + cfg["base_seed_offset"]
        set_seed(seed)
        concepts, changes = perturb_words(self.concepts, p["wp1"], seed=seed)
        txt = text_features(cfg, concepts, dev)
        if p["wp2"]:
            g = torch.Generator().manual_seed(seed)
            txt = txt + cfg["wp2_sigma"] * torch.randn(txt.shape, generator=g)
        tr, te = self.feats("train", p["ip"]), self.feats("test", p["ip"])
        M_tr, M_te = clip_activation_matrix(tr["clip"], txt), clip_activation_matrix(te["clip"], txt)
        W, _ = learn_projection(tr["backbone"], M_tr, te["backbone"], M_te, cfg["proj_steps"],
                                cfg["proj_batch_size"], cfg["proj_lr"], dev, seed, verbose=self.verbose)
        layer = standardise(W, tr["backbone"])
        if changes:
            write_lines(os.path.join(self.out, f"concepts_{pert_name}.txt"), concepts)
        return layer, dict(n_word_changes=len(changes), example_changes=[list(c) for c in changes[:10]])

    # ------------------------------------------------------------------ FVLC
    def train_fvlc(self, pert_name, base_layer, use_l2=True, use_l3=True, use_l4=True):
        cfg = self.cfg
        set_seed(cfg["seed"])
        p = PERTURBATIONS[pert_name]
        f_in = self.feats("train", p["ip"])["backbone"]
        if cfg["protocol"] == "seat":
            # Def. 2: FVLC is defined w.r.t. the vanilla CBM *of the same concept space*, i.e. the LF-CBM trained
            # on this (perturbed) pipeline.
            ref = base_layer
            with torch.no_grad():
                c_ref = ref(f_in)
                y_ref = self.final(c_ref)
        else:
            ref = self.vanilla
            with torch.no_grad():
                c_ref = self.vanilla(self.feats("train")["backbone"])
                y_ref = self.final(c_ref)
        trainer = FVLCTrainer(cfg, self.final, base_layer, ref, self.device, use_l2, use_l3, use_l4,
                              seed=cfg["seed"], verbose=self.verbose)
        layer, hist = trainer.fit(f_in, c_ref, y_ref)
        return layer, hist

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, layer, pert_name, fvlc_flags=None):
        """fvlc_flags=None -> `layer` is a base (LF-CBM) layer; otherwise the (use_l2, use_l3, use_l4) it was trained with."""
        if self.cfg["protocol"] == "seat":
            return self.evaluate_attack(layer, pert_name)
        if self.cfg["protocol"] == "paired" and fvlc_flags is not None:
            return self.evaluate_paired(layer, pert_name, fvlc_flags)
        return self.evaluate_anchor(layer, pert_name)

    def fvlc_clean(self, flags):
        """FVLC trained on the UNPERTURBED pipeline with the same seed / init / flags (cached)."""
        cache = self.__dict__.setdefault("_fvlc_clean", {})
        if flags not in cache:
            cache[flags], _ = self.train_fvlc("SEED_ONLY", self.vanilla, *flags)
        return cache[flags]

    def evaluate_paired(self, layer, pert_name, flags):
        """Symmetric reading of Sec. 6.1: TCPC/TOPC compare each method with ITSELF before vs after the
        perturbation. Base: vanilla LF-CBM vs LF-CBM re-trained under P. FVLC: FVLC trained on the clean
        pipeline vs FVLC trained under P (same seed, same init, same losses)."""
        p = PERTURBATIONS[pert_name]
        clean = self.fvlc_clean(flags)  # trains (with gradients) on first use
        with torch.no_grad():
            c0 = clean(self.feats("test")["backbone"])
            y0 = self.final(c0)
            c1 = layer(self.feats("test", p["ip"])["backbone"])
            y1 = self.final(c1)
        return self._metrics(c0, y0, c1, y1)

    def _metrics(self, c0, y0, c1, y1):
        k1 = self.cfg["k1"]
        return dict(
            acc=M.accuracy(y1, self.y_test),
            tcpc=M.tcpc(c0, c1), topc=M.topc(y0, y1),
            tcpc_global=M.tcpc(c0, c1, "global"), topc_global=M.topc(y0, y1, "global"),
            topk_overlap=M.topk_overlap(c0, c1, k1), kl=M.kl_div(y0, y1),
            pred_agreement=(y0.argmax(1) == y1.argmax(1)).float().mean().item(),
        )

    def evaluate_attack(self, layer, pert_name):
        """SEAT-style: change of the SAME model's concepts/outputs under a test-time l_inf PGD attack on its
        backbone features (radius in per-dim std units), maximising KL between clean and attacked predictions."""
        cfg = self.cfg
        p = PERTURBATIONS[pert_name]
        te = self.feats("test", p["ip"])["backbone"]
        scale = self.feats("train", p["ip"])["backbone"].std(dim=0, keepdim=True)
        R = cfg["eval_radius"] if cfg["eval_radius"] is not None else cfg["x_pgd_radius"]
        step = cfg["eval_step_size"] if cfg["eval_step_size"] is not None else cfg["x_pgd_step_size"]
        c0, y0, c1, y1 = attack_features(layer, self.final, te, scale, R, cfg["eval_steps"], step,
                                         seed=cfg["seed"] + 7, device=self.device)
        k1 = cfg["k1"]
        return dict(
            acc=M.accuracy(y0, self.y_test), acc_attacked=M.accuracy(y1, self.y_test),
            tcpc=M.tcpc(c0, c1), topc=M.topc(y0, y1),
            tcpc_global=M.tcpc(c0, c1, "global"), topc_global=M.topc(y0, y1, "global"),
            topk_overlap=M.topk_overlap(c0, c1, k1), kl=M.kl_div(y0, y1),
            pred_agreement=(y0.argmax(1) == y1.argmax(1)).float().mean().item(),
        )

    @torch.no_grad()
    def evaluate_anchor(self, layer, pert_name):
        p = PERTURBATIONS[pert_name]
        te_clean = self.feats("test")["backbone"]
        te_p = self.feats("test", p["ip"])["backbone"]
        c0 = self.vanilla(te_clean)
        y0 = self.final(c0)
        c1 = layer(te_p)
        y1 = self.final(c1)
        k1 = self.cfg["k1"]
        return dict(
            acc=M.accuracy(y1, self.y_test),
            tcpc=M.tcpc(c0, c1), topc=M.topc(y0, y1),
            tcpc_global=M.tcpc(c0, c1, "global"), topc_global=M.topc(y0, y1, "global"),
            topk_overlap=M.topk_overlap(c0, c1, k1), kl=M.kl_div(y0, y1),
            pred_agreement=(y0.argmax(1) == y1.argmax(1)).float().mean().item(),
        )

    def save_layer(self, layer, name):
        torch.save(layer.state(), os.path.join(self.out, "layers", f"{name}.pt"))
