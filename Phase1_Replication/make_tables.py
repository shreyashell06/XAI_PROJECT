"""Aggregate results/<dataset>/results.json into side-by-side comparison tables and figures.

  python make_tables.py            -> results/tables.md, results/fig_stability.png
"""
import argparse
import os

from fvlc import paper_results as P
from fvlc.config import ABLATIONS, PAPER_PERTURBATIONS
from fvlc.utils import load_json

NAMES = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "cub": "CUB"}
PRETTY = {"WP1_5": "WP1(5%)", "WP1_10": "WP1(10%)", "WP2": "WP2", "IP": "IP", "WP1_5+WP2": "WP1(5%)+WP2",
          "WP1_10+WP2": "WP1(10%)+WP2", "WP1_10+WP2+IP": "WP1(10%)+WP2+IP", "SEED_ONLY": "Seed only (control)"}


def f_acc(x):
    return "–" if x is None else f"{x:.2f}"


def f_sci(x):
    return "–" if x is None else f"{x:.2e}"


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    R = {ds: load_json(os.path.join(args.results, ds, "results.json")) for ds in P.DATASETS}
    R = {k: v for k, v in R.items() if v}
    if not R:
        raise SystemExit("No results found. Run run_experiments.py first.")
    dss = list(R)
    protos = sorted({R[d]["config"].get("protocol", "anchor") for d in dss})
    out = ["# Replication results vs. paper\n",
           "Ours = this repo (seed %s, evaluation protocol: %s). Paper = Lai et al., ICLR 2024. Accuracy in %%.\n" %
           (", ".join(str(R[d]["config"]["seed"]) for d in dss), ", ".join(protos))]

    # ---------------- Headline numbers (for the 1-page summary)
    out.append("\n## Headline numbers (averaged over the 7 paper perturbations)\n")
    out += ["| Dataset | LF-CBM acc ours / paper | FVLC acc ours / paper | base TCPC ours / paper | "
            "FVLC TCPC ours / paper | base TOPC ours / paper | FVLC TOPC ours / paper | seed-only TCPC ours |",
            "|---|---|---|---|---|---|---|---|"]

    def mean(v):
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    for d in dss:
        o = lambda m, k: mean([get(R[d], "perturbations", pn, m, k) for pn in PAPER_PERTURBATIONS])
        pa = mean([P.acc(f"{pn}/fvlc", d) for pn in PAPER_PERTURBATIONS])
        ps = {(m, i): mean([P.stability(f"{pn}/{m}", d)[i] for pn in PAPER_PERTURBATIONS])
              for m in ["base", "fvlc"] for i in (0, 1)}
        fa = o("fvlc", "acc")
        out.append(
            f"| {NAMES[d]} | {f_acc(100 * R[d]['vanilla']['acc'])} / {f_acc(P.acc('Label-free CBM', d))} | "
            f"{f_acc(None if fa is None else 100 * fa)} / {f_acc(pa)} | "
            f"{f_sci(o('base', 'tcpc'))} / {f_sci(ps['base', 0])} | {f_sci(o('fvlc', 'tcpc'))} / {f_sci(ps['fvlc', 0])} | "
            f"{f_sci(o('base', 'topc'))} / {f_sci(ps['base', 1])} | {f_sci(o('fvlc', 'topc'))} / {f_sci(ps['fvlc', 1])} | "
            f"{f_sci(get(R[d], 'perturbations', 'SEED_ONLY', 'base', 'tcpc'))} |")

    # ---------------- Table 1
    out.append("\n## Table 1 – Accuracy\n")
    hdr = "| Method | " + " | ".join(f"{NAMES[d]} ours | {NAMES[d]} paper" for d in dss) + " |"
    out += [hdr, "|" + "---|" * (1 + 2 * len(dss))]

    def row(label, ours, paper):
        cells = []
        for d in dss:
            o = ours(d)
            cells += [f_acc(None if o is None else 100 * o), f_acc(paper(d))]
        out.append(f"| {label} | " + " | ".join(cells) + " |")

    row("Standard (no interpretability)", lambda d: get(R[d], "standard", "acc"), lambda d: P.acc("Standard", d))
    row("Label-free CBM", lambda d: get(R[d], "vanilla", "acc"), lambda d: P.acc("Label-free CBM", d))
    for pn in PAPER_PERTURBATIONS + ["SEED_ONLY"]:
        for m in ["base", "fvlc"]:
            paper = (lambda d, pn=pn, m=m: P.acc(f"{pn}/{m}", d)) if pn != "SEED_ONLY" else (lambda d: None)
            row(f"{PRETTY[pn]} – {m if m == 'base' else 'FVLC'}",
                lambda d, pn=pn, m=m: get(R[d], "perturbations", pn, m, "acc"), paper)

    # ---------------- Table 2
    out.append("\n## Table 2 – Stability (TCPC / TOPC, lower is better)\n")
    for d in dss:
        out.append(f"\n### {NAMES[d]}\n")
        out += ["| Perturbation | Method | TCPC ours | TCPC paper | TOPC ours | TOPC paper |",
                "|---|---|---|---|---|---|"]
        for pn in PAPER_PERTURBATIONS + ["SEED_ONLY"]:
            for m in ["base", "fvlc"]:
                pt = P.stability(f"{pn}/{m}", d) if pn != "SEED_ONLY" else (None, None)
                out.append(f"| {PRETTY[pn]} | {m if m == 'base' else 'FVLC'} | "
                           f"{f_sci(get(R[d], 'perturbations', pn, m, 'tcpc'))} | {f_sci(pt[0])} | "
                           f"{f_sci(get(R[d], 'perturbations', pn, m, 'topc'))} | {f_sci(pt[1])} |")
        # average reduction
        red_o, red_p = [], []
        for pn in PAPER_PERTURBATIONS:
            b, f = get(R[d], "perturbations", pn, "base"), get(R[d], "perturbations", pn, "fvlc")
            if b and f:
                red_o += [1 - f["tcpc"] / b["tcpc"], 1 - f["topc"] / b["topc"]]
            pb, pf = P.stability(f"{pn}/base", d), P.stability(f"{pn}/fvlc", d)
            red_p += [1 - pf[0] / pb[0], 1 - pf[1] / pb[1]]
        if red_o:
            out.append(f"\nAverage TCPC/TOPC reduction FVLC vs base: **ours {100 * sum(red_o) / len(red_o):.1f}%**, "
                       f"paper {100 * sum(red_p) / len(red_p):.1f}% (paper claims ~90% on average).\n")

    # ---------------- Table 3
    if any(get(R[d], "perturbations", pn, "ablation") for d in dss for pn in PAPER_PERTURBATIONS):
        out.append("\n## Table 3 – Ablation (TCPC / TOPC). Paper row order inferred from the PDF layout.\n")
        for d in dss:
            pns = [pn for pn in PAPER_PERTURBATIONS if get(R[d], "perturbations", pn, "ablation")]
            if not pns:
                continue
            out.append(f"\n### {NAMES[d]}\n")
            out += ["| Perturbation | Losses | TCPC ours | TCPC paper | TOPC ours | TOPC paper |",
                    "|---|---|---|---|---|---|"]
            for pn in pns:
                for a in ABLATIONS:
                    o = get(R[d], "perturbations", pn, "ablation", a) or {}
                    pt = P.ablation(pn, a, d)
                    out.append(f"| {PRETTY[pn]} | {a} | {f_sci(o.get('tcpc'))} | {f_sci(pt[0])} | "
                               f"{f_sci(o.get('topc'))} | {f_sci(pt[1])} |")

    # ---------------- extra diagnostics
    out.append("\n## Extra diagnostics (ours only)\n")
    out += ["| Dataset | Perturbation | Method | top-k overlap | KL(y‖y') | pred. agreement | TCPC (global) |",
            "|---|---|---|---|---|---|---|"]
    for d in dss:
        for pn in PAPER_PERTURBATIONS + ["SEED_ONLY"]:
            for m in ["base", "fvlc"]:
                e = get(R[d], "perturbations", pn, m)
                if e:
                    out.append(f"| {NAMES[d]} | {PRETTY[pn]} | {m} | {e['topk_overlap']:.3f} | {e['kl']:.2e} | "
                               f"{e['pred_agreement']:.4f} | {e['tcpc_global']:.2e} |")

    path = os.path.join(args.results, "tables.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"Wrote {path}")
    plot(R, dss, args.results)


def plot(R, dss, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    colors = {"base ours": "#c0504d", "FVLC ours": "#4f81bd", "base paper": "#f2b8b6", "FVLC paper": "#b8cce4"}
    fig, axes = plt.subplots(len(dss), 2, figsize=(13, 3.2 * len(dss)), squeeze=False)
    x = np.arange(len(PAPER_PERTURBATIONS))
    for r, d in enumerate(dss):
        for c, metric in enumerate(["tcpc", "topc"]):
            ax = axes[r, c]
            series = {
                "base ours": [get(R[d], "perturbations", pn, "base", metric) for pn in PAPER_PERTURBATIONS],
                "base paper": [P.stability(f"{pn}/base", d)[c] for pn in PAPER_PERTURBATIONS],
                "FVLC ours": [get(R[d], "perturbations", pn, "fvlc", metric) for pn in PAPER_PERTURBATIONS],
                "FVLC paper": [P.stability(f"{pn}/fvlc", d)[c] for pn in PAPER_PERTURBATIONS],
            }
            for i, (name, vals) in enumerate(series.items()):
                vals = [np.nan if v is None else v for v in vals]
                ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, label=name, color=colors[name])
            ax.set_yscale("log")
            ax.set_xticks(x, [PRETTY[p] for p in PAPER_PERTURBATIONS], rotation=25, ha="right", fontsize=8)
            ax.set_title(f"{NAMES[d]} – {metric.upper()} (log scale, lower = more stable)", fontsize=10)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_stability.png"), dpi=150)
    print(f"Wrote {out_dir}/fig_stability.png")


if __name__ == "__main__":
    main()
