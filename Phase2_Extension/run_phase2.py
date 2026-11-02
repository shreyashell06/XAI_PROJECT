"""Phase 2: EC-FVLC (Evidence-Consistent FVLC) vs FVLC, before vs after.

Models (all paired protocol: anchored to the clean vanilla LF-CBM, Gaussian init as in Algorithm 1)
  constant          explainer that ignores its input (what TCPC/TOPC reward)
  lfcbm             vanilla Label-free CBM
  fvlc              FVLC, Eq. (2)                                                 <- BEFORE
  fvlc_views        ablation: same edited images as extra inputs for L1-L4, no L5 (more data, no consistency)
  ecfvlc_l<lambda>  EC-FVLC, Eq. (2) + lambda5 * L5 (concept infidelity)          <- AFTER (lambda5 sweep)

Metrics: accuracy; worst-case stability (feature PGD at several radii); paired IP stability (each model trained on
the clean and on the IP pipeline, as in Phase 1); decision relevance; evidence correlation on held-out test edits
(training CLIP and a held-out CLIP oracle); nuisance-edit TCPC; on CUB also part-masking locality and attribute
AUROC from the human annotations.

Examples
  python run_phase2.py --dataset cub                                  # ~1-1.5 h on a Kaggle T4
  python run_phase2.py --dataset cifar10 --lambda5 0.1 1              # ~2 h
  python run_phase2.py --dataset cub --no_oracle --no_paired_ip       # faster
  python run_phase2.py --dataset cub --limit_test 300 --fvlc_epochs 2 # smoke run

Output: results_phase2/<dataset>/phase2.json and phase2.md (trained layers in results_phase2/<dataset>/layers/).
"""
import argparse
import os
import time

from fvlc import faithfulness as Fa
from fvlc.config import get_config
from fvlc.phase2 import EvalContext, EvidenceData, get_or_train, markdown_tables
from fvlc.pipeline import Experiment
from fvlc.utils import get_device, load_json, save_json, set_seed


def summary(R, lambdas):
    """Before (FVLC) vs after (best EC-FVLC by the main sensitivity metric) in a few lines."""
    m = R["models"]

    def score(e):
        # selection metric fixed in advance: evidence correlation scored by the held-out CLIP (never used in
        # training); falls back to the training CLIP if the oracle was skipped
        ev = e.get("evidence", {})
        return ev.get("evidence_corr_oracle_clip", ev.get("evidence_corr_train_clip", float("-inf")))
    ecs = [n for n in m if n.startswith("ecfvlc")]
    ok = [n for n in ecs if m[n]["acc"] >= m["fvlc"]["acc"] - 0.01] or ecs
    best = max(ok, key=lambda n: score(m[n]))
    r = f"{R['radii'][-1]:g}"
    lines = [f"Selected EC-FVLC: **{best}** (highest held-out-CLIP evidence correlation among lambda5 values within "
             "1 accuracy point of FVLC).", "",
             "| | FVLC (before) | EC-FVLC (after) |", "|---|---|---|",
             f"| accuracy | {100 * m['fvlc']['acc']:.2f} | {100 * m[best]['acc']:.2f} |",
             f"| worst-case TCPC r={r} | {m['fvlc']['worst_case'][r]['tcpc']:.3f} | {m[best]['worst_case'][r]['tcpc']:.3f} |"]
    if "paired_ip" in m["fvlc"] and "paired_ip" in m[best]:
        lines.append(f"| paired IP TCPC | {m['fvlc']['paired_ip']['tcpc']:.4f} | {m[best]['paired_ip']['tcpc']:.4f} |")
    if "evidence" in m["fvlc"]:
        for k in ("evidence_corr_train_clip", "evidence_corr_oracle_clip", "nuisance_tcpc"):
            if k in m["fvlc"]["evidence"]:
                lines.append(f"| {k.replace('_', ' ')} | {m['fvlc']['evidence'][k]:.3f} | {m[best]['evidence'][k]:.3f} |")
    if "locality" in m["fvlc"]:
        lines += [f"| locality | {m['fvlc']['locality']['all']['locality']:.3f} | "
                  f"{m[best]['locality']['all']['locality']:.3f} |",
                  f"| within-species attribute AUROC | {m['fvlc']['attribute_auroc']['within_species']:.3f} | "
                  f"{m[best]['attribute_auroc']['within_species']:.3f} |"]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "cub"])
    ap.add_argument("--out", default="results_phase2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--lambda5", nargs="+", type=float, default=[0.003, 0.01, 0.03, 0.1, 0.3],
                    help="EC-FVLC sweep (L1-L4 use 0.01 each)")
    ap.add_argument("--views", nargs="+", default=None, help="image edits for L5 (default: config)")
    ap.add_argument("--radii", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    ap.add_argument("--mask_frac", type=float, default=0.3)
    ap.add_argument("--limit_test", type=int, default=None, help="debug: part-masking on the first N test images")
    ap.add_argument("--locality_every", type=int, default=1,
                    help="part-masking on every n-th test image (e.g. 3 on CPU; all species stay covered)")
    ap.add_argument("--no_oracle", action="store_true", help="skip the held-out CLIP (saves ~10-20 min)")
    ap.add_argument("--no_paired_ip", action="store_true", help="skip training every model on the IP pipeline")
    ap.add_argument("--no_views_ablation", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    for k, t in [("seed", int), ("fvlc_epochs", int), ("k1", int), ("num_workers", int), ("batch_size", int),
                 ("oracle_clip", str), ("fvlc_lr", float), ("lambda4", float), ("x_pgd_radius", float),
                 ("pgd_mode", str)]:
        ap.add_argument(f"--{k}", type=t, default=None)
    args = ap.parse_args()

    cfg = get_config(args.dataset, protocol="paired", fvlc_init="gaussian", seed=args.seed,
                     fvlc_epochs=args.fvlc_epochs, k1=args.k1, num_workers=args.num_workers,
                     batch_size=args.batch_size, oracle_clip=args.oracle_clip, fvlc_lr=args.fvlc_lr,
                     views=tuple(args.views) if args.views else None, lambda4=args.lambda4,
                     x_pgd_radius=args.x_pgd_radius, pgd_mode=args.pgd_mode)
    device = get_device(args.device)
    set_seed(cfg["seed"])
    t0 = time.time()
    exp = Experiment(cfg, device, args.out)
    exp.build_vanilla()
    print(f"Dataset {args.dataset} | {len(exp.concepts)} concepts | views {cfg['views']} | lambda5 {args.lambda5}")

    evidence = EvidenceData(exp, cfg["views"])
    print(f"[evidence] a_j = corr(g_j, CLIP_j): mean {evidence.a.mean():.3f}, min {evidence.a.min():.3f}")

    specs = [("fvlc", "fvlc", 0.0)]
    if not args.no_views_ablation:
        specs.append(("fvlc_views", "fvlc_views", 0.0))
    specs += [(f"ecfvlc_l{l:g}", "ecfvlc", l) for l in args.lambda5]
    trained, trained_ip = {}, {}
    for name, kind, lam in specs:
        trained[name] = get_or_train(exp, name, kind, lam, False, evidence, args.retrain)
        if not args.no_paired_ip:
            trained_ip[name] = get_or_train(exp, name, kind, lam, True, evidence, args.retrain)

    ctx = EvalContext(exp, args.radii, args.mask_frac, args.limit_test, evidence_views=cfg["views"],
                      oracle=not args.no_oracle, locality_every=args.locality_every)
    tr = exp.feats("train")["backbone"]
    models = {"constant": Fa.constant_layer(exp.vanilla, tr), "lfcbm": exp.vanilla, **trained}
    res_path = os.path.join(exp.out, "phase2.json")
    R = load_json(res_path) if not args.retrain else {}
    R.update(settings={k: cfg[k] for k in ("lambda1", "lambda2", "lambda3", "lambda4", "pgd_mode", "x_pgd_radius",
                                           "fvlc_epochs")},
             dataset=args.dataset, seed=cfg["seed"], radii=args.radii, views=list(cfg["views"]),
             lambda5=args.lambda5, n_concepts=len(exp.concepts),
             a_mean=evidence.a.mean().item(), models={})
    for name, layer in models.items():
        R["models"][name] = e = ctx.evaluate(layer, trained_ip.get(name))
        print(f"[{name}] acc {e['acc']:.4f}  evidence {e.get('evidence')}  locality {e.get('locality', {}).get('all')}")
        save_json(R, res_path)

    R["time_s"] = time.time() - t0
    save_json(R, res_path)
    write_report(R, os.path.join(exp.out, "phase2.md"))


def write_report(R, path):
    L = [f"# Phase 2: EC-FVLC vs FVLC ({R['dataset']})", "",
         f"Seed {R['seed']}, paired protocol, views {', '.join(R['views'])}, lambda5 sweep {R['lambda5']}. "
         "Arrows show which direction is better.", "", "## Before vs after", ""] + summary(R, R["lambda5"]) + \
        [""] + markdown_tables(R)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
