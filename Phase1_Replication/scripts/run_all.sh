#!/usr/bin/env bash
# Full replication in one go: every dataset x every evaluation protocol (anchor, seat, paired).
# Resumable: re-running skips anything already finished (results are written incrementally).
#
#   bash scripts/run_all.sh <output_root> [datasets...]        (default datasets: cifar10 cifar100 cub)
#
# Writes <output_root>/results_{anchor,seat,paired}/<dataset>/results.json and a tables.md per protocol.
# The Table 3 ablation runs on the datasets listed in ABLATION_DS (default: cifar10; it is the slow part).
set -u
OUT=${1:?usage: bash scripts/run_all.sh <output_root> [datasets...]}
shift
DATASETS=${*:-cifar10 cifar100 cub}
ABLATION_DS=${ABLATION_DS:-cifar10}
PY=${PY:-python run_experiments.py}   # overridable for testing
EXTRA=${EXTRA:-}                      # extra flags for every run (e.g. --batch_size 128)

for ds in $DATASETS; do
  echo "================ $ds ================"
  if [ "$ds" = "cub" ] && [ ! -d "${FVLC_CUB_ROOT:-data/CUB}" ] && \
     [ ! -f "${FVLC_ACT_DIR:-activations}/cub/test_resnet18_cub_ip0.001.pt" ]; then
    bash scripts/download_cub.sh || exit 1
  fi
  ABL=""
  case " $ABLATION_DS " in *" $ds "*) ABL="--ablation" ;; esac

  echo "---- [$ds] protocol A: anchor (Table 1 + 2 + 3) ----"
  $PY --dataset "$ds" $ABL $EXTRA --out "$OUT/results_anchor" || exit 1

  for P in seat paired; do                       # reuse the vanilla LF-CBM trained above
    mkdir -p "$OUT/results_$P/$ds"
    cp -n "$OUT/results_anchor/$ds/vanilla.pt" "$OUT/results_$P/$ds/" 2>/dev/null
  done

  echo "---- [$ds] protocol B: seat (test-time attack) ----"
  $PY --dataset "$ds" --protocol seat --pgd_mode per_sample --skip_standard $ABL $EXTRA \
      --out "$OUT/results_seat" || exit 1

  echo "---- [$ds] protocol C: paired (each method vs itself) ----"
  $PY --dataset "$ds" --protocol paired --fvlc_init gaussian --skip_standard $ABL $EXTRA \
      --out "$OUT/results_paired" || exit 1
  echo "================ $ds done ================"
done

for P in anchor seat paired; do
  python make_tables.py --results "$OUT/results_$P" || true
done
echo "ALL DONE"
