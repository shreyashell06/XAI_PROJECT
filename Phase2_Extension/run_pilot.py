"""Phase 2 pilot: is FVLC's stability bought without faithfulness? (go/no-go for EC-FVLC)

Scores three explainers on the same dataset with the two-sided metrics of fvlc/faithfulness.py:
  constant  an explainer that ignores its input (what TCPC/TOPC reward)
  lfcbm     the vanilla Label-free CBM
  fvlc      FVLC trained on the clean pipeline (paired protocol, Gaussian init = Algorithm 1)

Metrics: accuracy, decision relevance, worst-case stability (test-time feature PGD, several radii) and, on
CUB, part-masking locality and attribute AUROC (overall and within species) from the human annotations.

Examples
  python run_pilot.py --dataset cub                    # full pilot (~30-40 min on a Kaggle T4)
  python run_pilot.py --dataset cifar10                # no part/attribute metrics on CIFAR
  python run_pilot.py --dataset cub --limit_test 300   # quick check of the part-masking code

Output: results_phase2/<dataset>/pilot.json and pilot.md
Go/no-go (proposal sec. 7): GO if FVLC's locality / attribute AUROC are no better than LF-CBM's, or if decision
relevance is low.
"""
import argparse
import os
import time

from fvlc import faithfulness as Fa
from fvlc.config import get_config
from fvlc.phase2 import EvalContext, get_or_train, markdown_tables
from fvlc.pipeline import Experiment
from fvlc.utils import get_device, save_json, set_seed


def write_markdown(R, path):
    L = ["# Phase 2 pilot: two-sided faithfulness", "",
         f"Dataset **{R['dataset']}**, seed {R['seed']}, paired protocol (FVLC = Algorithm 1, Gaussian init). "
         "Arrows show which direction is better.", ""] + markdown_tables(R) + ["", "## Go / no-go", "", R["verdict"]]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def verdict(R):
    m = R["models"]
    lines = []
    if "locality" in m["fvlc"]:
        dl = m["fvlc"]["locality"]["all"].get("locality", float("nan")) - \
            m["lfcbm"]["locality"]["all"].get("locality", float("nan"))
        da = m["fvlc"]["attribute_auroc"]["within_species"] - m["lfcbm"]["attribute_auroc"]["within_species"]
        lines.append(f"FVLC minus LF-CBM: locality {dl:+.3f}, within-species AUROC {da:+.3f}.")
        better = dl > 0.05 and da > 0.01
    else:
        better = False
    dr = m["fvlc"]["decision"]["nonzero_frac"]
    lines.append(f"Share of FVLC's top-k concepts with a non-zero weight for the predicted class: {dr:.2f}.")
    r_key = f"{R['radii'][-1]:g}"
    lines.append(f"Constant explainer worst-case TCPC: {m['constant']['worst_case'][r_key]['tcpc']:.3f}, vs "
                 f"{m['fvlc']['worst_case'][r_key]['tcpc']:.3f} for FVLC (it 'wins' on stability).")
    go = (not better) or dr < 0.5
    lines.append("**GO**: build EC-FVLC." if go else
                 "**NO-GO**: FVLC is clearly better grounded than LF-CBM and its explanations drive the decision; "
                 "switch to the worst-case-stability fallback.")
    return " ".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "cub"])
    ap.add_argument("--out", default="results_phase2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--radii", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    ap.add_argument("--mask_frac", type=float, default=0.3, help="mask side as a fraction of sqrt(bird bbox area)")
    ap.add_argument("--limit_test", type=int, default=None, help="debug: part-masking on the first N test images")
    ap.add_argument("--locality_every", type=int, default=1,
                    help="part-masking on every n-th test image (e.g. 3 on CPU; all species stay covered)")
    ap.add_argument("--retrain_fvlc", action="store_true")
    for k, t in [("seed", int), ("fvlc_epochs", int), ("k1", int), ("num_workers", int), ("batch_size", int)]:
        ap.add_argument(f"--{k}", type=t, default=None)
    args = ap.parse_args()

    cfg = get_config(args.dataset, protocol="paired", fvlc_init="gaussian", seed=args.seed,
                     fvlc_epochs=args.fvlc_epochs, k1=args.k1, num_workers=args.num_workers,
                     batch_size=args.batch_size)
    device = get_device(args.device)
    set_seed(cfg["seed"])
    t0 = time.time()
    exp = Experiment(cfg, device, args.out)
    exp.build_vanilla()
    tr = exp.feats("train")["backbone"]

    fvlc = get_or_train(exp, "fvlc", "fvlc", retrain=args.retrain_fvlc)   # paired protocol, clean pipeline
    models = {"constant": Fa.constant_layer(exp.vanilla, tr), "lfcbm": exp.vanilla, "fvlc": fvlc}

    ctx = EvalContext(exp, args.radii, args.mask_frac, args.limit_test, locality_every=args.locality_every)
    R = dict(dataset=args.dataset, seed=cfg["seed"], radii=args.radii, k=cfg["k1"], models={},
             n_concepts=len(exp.concepts))
    for name, layer in models.items():
        R["models"][name] = e = ctx.evaluate(layer)
        print(f"[{name}] acc {e['acc']:.4f}  decision {e['decision']}  locality {e.get('locality', {}).get('all')}")

    R["verdict"] = verdict(R)
    R["time_s"] = time.time() - t0
    save_json(R, os.path.join(exp.out, "pilot.json"))
    write_markdown(R, os.path.join(exp.out, "pilot.md"))
    print("\n" + open(os.path.join(exp.out, "pilot.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
