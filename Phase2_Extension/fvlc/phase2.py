"""Phase 2 glue: train FVLC variants (incl. EC-FVLC) and evaluate any concept layer on both sides of faithfulness.

Used by run_pilot.py (go/no-go) and run_phase2.py (before vs after).
"""
import os
from copy import deepcopy

import torch

from . import faithfulness as Fa
from . import metrics as M
from .cbm import ConceptLayer
from .features import ACT_DIR, text_features, view_features
from .fvlc import FVLCTrainer, evidence_targets
from .utils import set_seed
from .views import EVIDENCE_VIEWS, NUISANCE_VIEWS


# ============================================================================ training
class EvidenceData:
    """CLIP evidence for L5 on the training set: backbone features of the edited images and targets a_j * ds_j."""

    def __init__(self, exp, views, limit=None):
        cfg, dev = exp.cfg, exp.device
        self.views = list(views)
        tr = exp.feats("train")
        vf = view_features(cfg, "train", self.views, dev, limit=limit)
        txt = text_features(cfg, exp.concepts, dev)
        t = txt.float() / txt.float().norm(dim=1, keepdim=True)
        s0 = (tr["clip"] / tr["clip"].norm(dim=1, keepdim=True)) @ t.T                 # (N, k) clean evidence
        g0 = exp.vanilla(tr["backbone"])
        gz = (g0 - g0.mean(0)) / g0.std(0)
        sz = (s0 - s0.mean(0)) / s0.std(0)
        self.a = (gz * sz).mean(0)                                                     # corr(g_j, s_j)
        self.s_std = s0.std(0)
        self.feats = vf["backbone"]
        self.target = evidence_targets(tr["clip"], vf["clip"], txt, self.a, s0.mean(0), self.s_std).half()

    def as_views(self):
        return dict(feats=self.feats, target=self.target)


def train_variant(exp, kind, lambda5=0.0, ip=False, evidence=None, verbose=True):
    """Paired-protocol training (FVLC anchored to the clean vanilla LF-CBM, Gaussian init = Algorithm 1).

    kind  "fvlc"        Eq. (2) as in the paper (Phase 1 implementation)
          "fvlc_views"  ablation: the edited images are added as extra training inputs for L1-L4, no L5
          "ecfvlc"      EC-FVLC: Eq. (2) + lambda5 * L5
    ip    train on the IP-perturbed pipeline (for the paired stability test)
    """
    cfg = deepcopy(exp.cfg)
    cfg["lambda5"] = lambda5 if kind == "ecfvlc" else 0.0
    set_seed(cfg["seed"])
    f_in = exp.feats("train", ip)["backbone"]
    with torch.no_grad():
        c_ref = exp.vanilla(exp.feats("train")["backbone"])
        y_ref = exp.final(c_ref)
    views = None
    if kind == "fvlc_views":
        vf = evidence.feats.float()
        N, V, d = vf.shape
        with torch.no_grad():
            c_v = exp.vanilla(vf.reshape(N * V, d))
        f_in = torch.cat([f_in, vf.reshape(N * V, d)])
        c_ref = torch.cat([c_ref, c_v])
        y_ref = torch.cat([y_ref, exp.final(c_v)])
    elif kind == "ecfvlc":
        views = evidence.as_views()
    trainer = FVLCTrainer(cfg, exp.final, exp.vanilla, exp.vanilla, exp.device, seed=cfg["seed"], verbose=verbose)
    layer, hist = trainer.fit(f_in, c_ref, y_ref, views=views)
    return layer, hist


def get_or_train(exp, name, kind, lambda5=0.0, ip=False, evidence=None, retrain=False):
    path = os.path.join(exp.out, "layers", f"{name}{'_IP' if ip else ''}.pt")
    if os.path.exists(path) and not retrain:
        return ConceptLayer(**torch.load(path))
    print(f"[train] {name}{' (IP pipeline)' if ip else ''}")
    layer, _ = train_variant(exp, kind, lambda5, ip, evidence)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(layer.state(), path)
    return layer


# ============================================================================ evaluation
class EvalContext:
    """Everything that does not depend on the model being evaluated, computed once."""

    def __init__(self, exp, radii=(0.01, 0.05, 0.1), mask_frac=0.3, limit_test=None, evidence_views=None,
                 oracle=True, locality_every=1):
        self.exp, self.radii = exp, list(radii)
        cfg, dev = exp.cfg, exp.device
        self.tr, self.te = exp.feats("train")["backbone"], exp.feats("test")["backbone"]
        self.cub = None
        if cfg["dataset"] == "cub":
            self._init_cub(mask_frac, limit_test, locality_every)
        self.ev = None
        if evidence_views:
            self._init_evidence(list(evidence_views), oracle)

    def _init_cub(self, mask_frac, limit_test, every=1):
        from .cub_meta import CUBMeta, map_concepts
        from .data import get_dataset
        from .models import get_backbone
        exp = self.exp
        meta = CUBMeta()
        test_ds = get_dataset("cub", "test")
        samples = test_ds.samples[:limit_test] if limit_test else test_ds.samples
        samples = samples[::every]                     # every n-th image: still covers all species
        cmaps = map_concepts(exp.concepts)
        variants, canon, cvar = Fa.locality_variants(cmaps)
        fn, pre = get_backbone(exp.cfg["backbone"], exp.device)
        tag = (f"_n{limit_test}" if limit_test else "") + (f"_every{every}" if every > 1 else "")
        cache = Fa.cache_partmask_features(
            fn, pre, samples, meta.image_ids(samples), meta, variants,
            path=os.path.join(ACT_DIR, "cub", f"partmask_f{mask_frac:g}{tag}.pt"), mask_frac=mask_frac,
            num_workers=exp.cfg["num_workers"], device=exp.device)
        labels = Fa.concept_label_matrix(meta, cmaps, meta.image_ids(test_ds.samples))
        self.cub = dict(meta=meta, cmaps=cmaps, canon=canon, cvar=cvar, cache=cache, labels=labels,
                        n_part=sum(bool(m.kp_parts) for m in cmaps), n_attr=sum(bool(m.groups) for m in cmaps))

    def _init_evidence(self, views, oracle):
        """Test-set edits, scored by the training CLIP and (if oracle) by a held-out CLIP."""
        exp, cfg, dev = self.exp, self.exp.cfg, self.exp.device
        vf = view_features(cfg, "test", views, dev, models=("backbone",))["backbone"]
        clips = {"train_clip": cfg["clip_name"]}
        if oracle:
            clips["oracle_clip"] = cfg["oracle_clip"]
        ds = {}
        for key, name in clips.items():
            f = view_features(cfg, "test", views, dev, models=("clip",), clip_name=name)["clip"].float()
            if name == cfg["clip_name"]:
                f0 = exp.feats("test")["clip"].float().unsqueeze(1)           # already cached
            else:
                f0 = view_features(cfg, "test", ["clean"], dev, models=("clip",), clip_name=name)["clip"].float()
            f = torch.cat([f0, f], dim=1)
            txt = text_features(cfg, exp.concepts, dev, clip_name=name).float()
            t = txt / txt.norm(dim=1, keepdim=True)
            s = (f / f.norm(dim=2, keepdim=True)) @ t.T                                  # (N, 1+V, k)
            ds[key] = s[:, 1:] - s[:, :1]
        self.ev = dict(views=views, feats=vf, ds=ds)

    # ------------------------------------------------------------------------
    @torch.no_grad()
    def evidence_scores(self, layer):
        """evidence correlation: per concept, corr(dg_j, ds_j) over (image, evidence edit) pairs, averaged over
        concepts (0 for a concept whose score never moves). nuisance TCPC: ||dg|| / ||g|| under nuisance edits."""
        ev = self.ev
        N, V, d = ev["feats"].shape
        c0 = layer(self.te)
        cv = layer(ev["feats"].float().reshape(N * V, d)).reshape(N, V, -1)
        dg = cv - c0.unsqueeze(1)
        e_idx = [i for i, v in enumerate(ev["views"]) if v in EVIDENCE_VIEWS]
        n_idx = [i for i, v in enumerate(ev["views"]) if v in NUISANCE_VIEWS]
        out = {}
        for key, ds in ev["ds"].items():
            if not e_idx:
                break
            a = dg[:, e_idx].reshape(-1, dg.shape[2])
            b = ds[:, e_idx].reshape(-1, ds.shape[2])
            a, b = a - a.mean(0), b - b.mean(0)
            den = a.norm(dim=0) * b.norm(dim=0)
            corr = torch.where(den > 1e-12, (a * b).sum(0) / den.clamp_min(1e-12), torch.zeros_like(den))
            out[f"evidence_corr_{key}"] = corr.mean().item()
            for i in e_idx:
                ai, bi = dg[:, i] - dg[:, i].mean(0), ds[:, i] - ds[:, i].mean(0)
                den = ai.norm(dim=0) * bi.norm(dim=0)
                ci = torch.where(den > 1e-12, (ai * bi).sum(0) / den.clamp_min(1e-12), torch.zeros_like(den))
                out[f"evidence_corr_{key}_{ev['views'][i]}"] = ci.mean().item()
        if n_idx:
            out["nuisance_tcpc"] = M.tcpc(c0, cv[:, n_idx[0]])
        return out

    def evaluate(self, layer, paired_ip_layer=None):
        exp, cfg = self.exp, self.exp.cfg
        e = dict(acc=M.accuracy(exp.final(layer(self.te)), exp.y_test),
                 decision=Fa.decision_relevance(layer, exp.final, self.te, k=cfg["k1"]),
                 worst_case=Fa.worst_case_stability(layer, exp.final, self.te, self.tr, self.radii,
                                                    seed=cfg["seed"] + 7, device=exp.device, k=cfg["k1"]))
        if paired_ip_layer is not None:
            with torch.no_grad():
                c0 = layer(self.te)
                c1 = paired_ip_layer(exp.feats("test", True)["backbone"])
            e["paired_ip"] = dict(tcpc=M.tcpc(c0, c1), topc=M.topc(exp.final(c0), exp.final(c1)),
                                  acc=M.accuracy(exp.final(c1), exp.y_test))
        if self.cub:
            c = self.cub
            e["locality"] = Fa.locality(layer, c["cache"], c["meta"], c["cmaps"], c["canon"], c["cvar"])
            e["attribute_auroc"] = Fa.attribute_auroc(layer, self.te, c["labels"], exp.y_test)
        if self.ev:
            e["evidence"] = self.evidence_scores(layer)
        return e


# ============================================================================ report
def _f(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def markdown_tables(R):
    models, radii = list(R["models"]), R["radii"]
    r_last = f"{radii[-1]:g}"
    has_ip = any("paired_ip" in R["models"][m] for m in models)
    L = ["## Accuracy, stability and decision relevance", "",
         "| Model | Acc ↑ | " + " | ".join(f"worst-case TCPC r={r} ↓" for r in radii) +
         f" | top-k overlap r={r_last} ↑ |" + (" paired IP TCPC ↓ |" if has_ip else "") +
         " top-k with non-zero weight ↑ | activations vs contributions ↑ |",
         "|---" * (5 + len(radii) + has_ip) + "|"]
    for m in models:
        e = R["models"][m]
        wc = e["worst_case"]
        ip = e.get("paired_ip", {}).get("tcpc") if has_ip else None
        L.append(f"| {m} | {_f(100 * e['acc'], 2)} | " + " | ".join(_f(wc[f'{r:g}']['tcpc']) for r in radii) +
                 f" | {_f(wc[r_last]['topk_overlap'])} |" + (f" {_f(ip)} |" if has_ip else "") +
                 f" {_f(e['decision']['nonzero_frac'])} | {_f(e['decision']['act_vs_contrib'])} |")
    first = R["models"][models[0]]
    if "evidence" in first:
        keys = [k for k in first["evidence"] if k.startswith("evidence_corr_") and k.count("_") == 3]
        L += ["", "## Sensitivity to label-free edits (held-out test edits)", "",
              "| Model | " + " | ".join(k.replace("evidence_corr_", "evidence corr. (") + ") ↑" for k in keys) +
              " | nuisance-edit TCPC ↓ |", "|---" * (2 + len(keys)) + "|"]
        for m in models:
            ev = R["models"][m]["evidence"]
            L.append(f"| {m} | " + " | ".join(_f(ev.get(k)) for k in keys) + f" | {_f(ev.get('nuisance_tcpc'))} |")
        L += ["", "Evidence corr. = per-concept correlation between the change of the concept score and the change "
              "of CLIP's image-text similarity under occlusion / crop / CutMix of test images, averaged over "
              "concepts. `oracle_clip` uses a CLIP model never seen in training."]
    if "locality" in first:
        L += ["", "## Sensitivity and grounding on CUB human annotations", "",
              "| Model | own-part change ↑ | distant-part change ↓ | locality ↑ | drop when concept is on ↑ | "
              "attribute AUROC ↑ | within-species AUROC ↑ |", "|---" * 7 + "|"]
        for m in models:
            e = R["models"][m]
            la, lc, au = e["locality"]["all"], e["locality"]["active"], e["attribute_auroc"]
            L.append(f"| {m} | {_f(la.get('relevant_abs'))} | {_f(la.get('irrelevant_abs'))} | "
                     f"{_f(la.get('locality'))} | {_f(lc.get('relevant_drop'))} | {_f(au['overall'])} | "
                     f"{_f(au['within_species'])} |")
        ref = R["models"].get("lfcbm", first)
        L += ["", f"Locality over {ref['locality']['all'].get('n_concepts', 0)} part-named concepts; attribute "
              f"AUROC over {ref['attribute_auroc']['n_concepts']} concepts and {ref['attribute_auroc']['n_pairs']} "
              "(concept, species) pairs. Scores in standardised units. Locality = own-part change minus "
              "distant-part change (Raman et al., TMLR 2025)."]
    return L
