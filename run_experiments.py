"""Run the replication for one dataset: Tables 1, 2 (and optionally 3) of the FVLC paper.

Examples
  python run_experiments.py --dataset cifar10                      # Tables 1 + 2
  python run_experiments.py --dataset cifar10 --ablation           # + Table 3
  python run_experiments.py --dataset cifar10 --perturbations WP2 IP --fvlc_epochs 5
  python run_experiments.py --dataset cifar10 --limit_train 2000 --fvlc_epochs 1   # quick smoke test

Results are written incrementally to results/<dataset>/results.json (re-running skips finished entries;
use --force to recompute).
"""
import argparse
import time

from fvlc.config import ABLATIONS, PERTURBATIONS, get_config
from fvlc.pipeline import Experiment
from fvlc.utils import get_device, load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "cub"])
    ap.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS), choices=list(PERTURBATIONS))
    ap.add_argument("--ablation", action="store_true", help="also run Table 3 (L2/L3/L4 on/off)")
    ap.add_argument("--ablation_perturbations", nargs="+",
                    default=["WP1_10", "WP2", "IP", "WP1_10+WP2", "WP1_10+WP2+IP"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip_standard", action="store_true", help="skip the (slow) sparse standard baseline")
    ap.add_argument("--limit_train", type=int, default=None, help="debug: use only the first N train/test samples")
    # common overrides (anything in fvlc/config.py can be added the same way)
    for k, t in [("seed", int), ("fvlc_epochs", int), ("fvlc_lr", float), ("fvlc_batch_size", int),
                 ("k1", int), ("k2", int), ("fvlc_init", str), ("rho_space", str), ("pgd_mode", str),
                 ("base_seed_offset", int), ("proj_batch_size", int), ("concept_set", str), ("n_iters", int), ("std_n_iters", int),
                 ("lambda1", float), ("lambda2", float), ("lambda3", float), ("lambda4", float),
                 ("batch_size", int), ("num_workers", int), ("protocol", str), ("eval_radius", float),
                 ("eval_steps", int), ("eval_step_size", float), ("x_pgd_radius", float), ("pgd_radius", float)]:
        ap.add_argument(f"--{k}", type=t, default=None)
    args = ap.parse_args()

    overrides = {k: getattr(args, k) for k in vars(args)
                 if k not in {"dataset", "perturbations", "ablation", "ablation_perturbations", "device",
                              "out", "force", "limit_train", "skip_standard"}}
    cfg = get_config(args.dataset, **overrides)
    device = get_device(args.device)
    print(f"Dataset {args.dataset} | device {device} | seed {cfg['seed']}")

    exp = Experiment(cfg, device, args.out, limit_train=args.limit_train)
    res_path = f"{exp.out}/results.json"
    R = {} if args.force else load_json(res_path)
    R["config"] = cfg

    R["vanilla"] = exp.build_vanilla(force=args.force)
    if "standard" not in R and not args.skip_standard:
        R["standard"] = exp.build_standard()
    save_json(R, res_path)

    R.setdefault("perturbations", {})
    todo = list(dict.fromkeys(args.perturbations + (args.ablation_perturbations if args.ablation else [])))
    for pn in todo:
        entry = R["perturbations"].setdefault(pn, {})
        need_main = pn in args.perturbations and "fvlc" not in entry
        need_abl = args.ablation and pn in args.ablation_perturbations and \
            len(entry.get("ablation", {})) < len(ABLATIONS)
        if not (need_main or need_abl):
            print(f"[{pn}] done, skipping")
            continue
        print(f"\n===== {pn} =====")
        t0 = time.time()
        base, binfo = exp.build_base(pn)
        entry["base"] = exp.evaluate(base, pn) | binfo
        exp.save_layer(base, f"{pn}_base")
        print(f"[{pn}] base : acc {entry['base']['acc']:.4f}  TCPC {entry['base']['tcpc']:.3e}  "
              f"TOPC {entry['base']['topc']:.3e}")
        save_json(R, res_path)

        if need_main:
            layer, hist = exp.train_fvlc(pn, base)
            entry["fvlc"] = exp.evaluate(layer, pn, (True, True, True)) | {"history": hist}
            exp.save_layer(layer, f"{pn}_fvlc")
            print(f"[{pn}] FVLC : acc {entry['fvlc']['acc']:.4f}  TCPC {entry['fvlc']['tcpc']:.3e}  "
                  f"TOPC {entry['fvlc']['topc']:.3e}")
            save_json(R, res_path)

        if need_abl:
            abl = entry.setdefault("ablation", {})
            for name, (l2, l3, l4) in ABLATIONS.items():
                if name in abl:
                    continue
                if name == "L1+L2+L3+L4" and "fvlc" in entry:
                    abl[name] = {k: v for k, v in entry["fvlc"].items() if k != "history"}
                    continue
                layer, _ = exp.train_fvlc(pn, base, l2, l3, l4)
                abl[name] = exp.evaluate(layer, pn, (l2, l3, l4))
                print(f"[{pn}] ablation {name:12s}: TCPC {abl[name]['tcpc']:.3e}  TOPC {abl[name]['topc']:.3e}")
                save_json(R, res_path)
        entry["time_s"] = time.time() - t0
        save_json(R, res_path)
    print(f"\nSaved {res_path}. Build tables with: python make_tables.py")


if __name__ == "__main__":
    main()
