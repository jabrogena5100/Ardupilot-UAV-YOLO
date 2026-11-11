#!/usr/bin/env bash
# Run the six frozen checkpoints on ONE external dataset of eval/datasets_external.yaml on the local CPU, one cell per process (resumable, logs per cell).
#   tmux new -s ext_eval 'bash eval/external_scripts/run_external_cpu.sh fireman_primary'
# Outputs only under local_data/external_eval/runs/ext_v1_<dataset>/ (predictions, cells, gts, logs). Never touches docs/baseline, runs/ or any frozen artifact:
# the baseline verifier is run before and after and both results are logged.
set -u
DS=${1:?dataset name from eval/datasets_external.yaml}
cd "$(dirname "$0")/../.."
OUT=${OUT:-local_data/external_eval/runs/ext_v1_${DS}}
VENV=${VENV:-/home/joshua/venv-external-eval}
. "$VENV/bin/activate"
mkdir -p "$OUT/logs"
{
  date -u; echo "commit $(git rev-parse HEAD)"; git status --short; python -V
  pip freeze 2>/dev/null | grep -iE "^(torch|torchvision|ultralytics|numpy|pillow|pyyaml)=="
  echo "--- baseline verifier (before)"; python eval/freeze_baseline.py --verify
} > "$OUT/logs/run_info_before.txt" 2>&1
MODELS="model_a model_a_seed1 model_a_seed2 model_bv2 model_bv2_seed1 model_bv2_seed2"
COMMON=(--config eval/datasets_external.yaml --var BACKUPS=local_data/backups --var EXTERNAL_ROOT=local_data/external_eval/views)
for m in $MODELS; do
  if [ -f "$OUT/cells/${m}__${DS}.json" ]; then echo "skip $m (cell exists)"; continue; fi
  echo "=== $m on $DS $(date -u +%T)"
  python -m eval.run_matrix "${COMMON[@]}" --out "$OUT" --cells "${m}:${DS}" 2>&1 | tee -a "$OUT/logs/${m}.log"
done
CELLS=$(for m in $MODELS; do printf "%s:%s," "$m" "$DS"; done); CELLS=${CELLS%,}
python -m eval.run_matrix "${COMMON[@]}" --out "$OUT" --preds-from "$OUT/predictions" --cells "$CELLS" > "$OUT/logs/summary_offline.log" 2>&1
{ date -u; echo "--- baseline verifier (after)"; python eval/freeze_baseline.py --verify; git status --short; } > "$OUT/logs/run_info_after.txt" 2>&1
echo DONE > "$OUT/logs/DONE"
