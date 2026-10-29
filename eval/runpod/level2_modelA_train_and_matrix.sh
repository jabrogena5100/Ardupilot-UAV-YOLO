#!/bin/bash
# Level 2 on one RunPod GPU pod (1x RTX 4090, volume `wildfire-datasets` mounted at /workspace).
# Prereqs uploaded by the operator: /workspace/incoming/{synth_smoke_v1.tar, synth_smoke_v1.tar.sha256sum,
# synth_smoke_v1.SHA256SUMS, repo_eval.tar}. Never overwrites existing runs/checkpoints (aborts instead).
# Stages: 0 env+dataset verify, 1 harness dry-run, 2 Model B reproduction through the harness,
#         3 Model A training (recipe: docs/synthetic/synth_smoke_v1_spec.md section 5),
#         4 full 2x2 matrix with Model A pinned in a config copy stored with the results.
set +e
W=/workspace; IN=$W/incoming; RUNS=$W/runs
SYN=$W/synth_smoke_v1; REPO=$W/repo; OUT=$RUNS/level2_v1
BOREAL_SPLIT=$W/splits/boreal_v1
RUN=$RUNS/synth_smoke_a_v1
LOG=$RUNS/synth_smoke_a_v1_train.log
echo "=== START $(date -u +%FT%TZ)"
for p in "$RUN" "$OUT" "$SYN" "$REPO"; do [ -e "$p" ] && { echo "ABORT: $p exists (refusing to overwrite)"; exit 1; }; done
[ -f $RUNS/real_boreal/real_boreal_best_v1.pt ] || { echo "ABORT: Model B checkpoint missing"; exit 1; }

echo "--- env"
pip install -q --break-system-packages "ultralytics==8.4.173" pyyaml 2>&1 | tail -2
python3 - <<'PY' || { echo "ABORT: ultralytics/torch import failed"; exit 1; }
import ultralytics, torch, platform
print("ultralytics", ultralytics.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(), "py", platform.python_version(), torch.cuda.get_device_name(0))
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
mkdir -p $RUNS $OUT

echo "--- stage 0: dataset verify"
( cd $IN && sha256sum -c synth_smoke_v1.tar.sha256sum ) || { echo "ABORT: tar hash"; exit 1; }
tar xf $IN/synth_smoke_v1.tar -C $W || { echo "ABORT: untar"; exit 1; }
( cd $SYN && sha256sum -c --quiet $IN/synth_smoke_v1.SHA256SUMS ) && echo "ALL 10616 FILE HASHES OK" || { echo "ABORT: file hash mismatch"; exit 1; }
cat > $SYN/data_rw.yaml <<YML
# path-rewritten COPY of data.yaml (originals are hashed and untouched)
path: $SYN
train: images/train
val: images/val
test: images/test
names:
  0: smoke
YML
mkdir -p $REPO && tar xf $IN/repo_eval.tar -C $REPO
cd $REPO
VARS="--var BOREAL_SPLIT=$BOREAL_SPLIT --var SYNTH_ROOT=$SYN --var RUNS=$RUNS"
python3 -m unittest tests.test_eval_metrics 2>&1 | tail -3

echo "--- stage 1: harness dry-run (dataset pins, Model B hash)"
python3 -m eval.run_matrix --out $OUT $VARS --cells model_b:boreal_test,model_b:synth_test_pos,model_b:synth_test_all --dry-run || { echo "ABORT: dry-run"; exit 1; }

echo "--- stage 2: Model B through the harness (reproduction check)"
python3 -m eval.run_matrix --out $OUT/model_b_only $VARS --cells model_b:boreal_test,model_b:synth_test_pos,model_b:synth_test_all 2>&1 | tail -40
echo "STAGE2_DONE $(date -u +%FT%TZ)"

echo "--- stage 3: Model A training"
T0=$(date +%s)
timeout 6000 yolo detect train model=yolov8s.pt data=$SYN/data_rw.yaml epochs=100 patience=100 imgsz=640 \
  batch=32 seed=0 deterministic=True workers=8 plots=True optimizer=AdamW lr0=0.002 momentum=0.9 \
  warmup_bias_lr=0.0 project=$RUNS name=synth_smoke_a_v1 exist_ok=False > $LOG 2>&1
RC=$?; T1=$(date +%s)
echo "TRAIN_EXIT_CODE=$RC TRAIN_WALL_SECONDS=$((T1-T0))  (124 = timeout)"
tail -c 1200 $LOG | tr '\r' '\n' | grep -v '^\s*$' | tail -6
[ -f $RUN/weights/best.pt ] || { echo "NO best.pt - abort"; exit 1; }
cp $RUN/weights/best.pt $RUN/synth_smoke_best_v1.pt
SHA=$(sha256sum $RUN/synth_smoke_best_v1.pt | cut -d' ' -f1); echo "MODEL_A_SHA256=$SHA"
TRAIN_RC=$RC TRAIN_SECS=$((T1-T0)) SHA=$SHA python3 - <<'PY'
import os, json, csv, yaml, platform, subprocess, ultralytics, torch
RUN="/workspace/runs/synth_smoke_a_v1"
rows=[{k.strip():v for k,v in r.items()} for r in csv.DictReader(open(f"{RUN}/results.csv"))]
best=max(rows,key=lambda x:float(x["metrics/mAP50-95(B)"])); last=rows[-1]
print("EPOCHS", len(rows), "BEST_EPOCH", best["epoch"], "val_mAP50", best["metrics/mAP50(B)"], "val_mAP50-95", best["metrics/mAP50-95(B)"])
print("LAST", last["epoch"], "val_mAP50-95", last["metrics/mAP50-95(B)"], "train_box", last["train/box_loss"], "val_box", last["val/box_loss"])
json.dump({"experiment":"synth_smoke_a_v1 (Model A baseline)","train_rc":int(os.environ["TRAIN_RC"]),
  "train_wall_seconds":int(os.environ["TRAIN_SECS"]),"complete":int(os.environ["TRAIN_RC"])==0 and len(rows)==100,
  "train_args":yaml.safe_load(open(f"{RUN}/args.yaml")),"best_epoch":int(best["epoch"]),
  "checkpoint_sha256":os.environ["SHA"],
  "environment":{"ultralytics":ultralytics.__version__,"torch":torch.__version__,"cuda":torch.version.cuda,
    "cudnn":torch.backends.cudnn.version(),"python":platform.python_version(),"gpu":torch.cuda.get_device_name(0),
    "driver":subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),
    "pip_freeze":subprocess.run(["pip","freeze"],capture_output=True,text=True).stdout.splitlines()}},
  open(f"{RUN}/train_config_v1.json","w"), indent=2)
PY

echo "--- stage 4: full matrix, Model A pinned in a config copy"
sed "s/sha256: null /sha256: $SHA /" eval/datasets.yaml > $OUT/datasets_pinned.yaml
grep -n "sha256: $SHA" $OUT/datasets_pinned.yaml || { echo "ABORT: pin substitution failed"; exit 1; }
python3 -m eval.run_matrix --config $OUT/datasets_pinned.yaml --out $OUT $VARS 2>&1 | tail -60
( cd $SYN && sha256sum -c --quiet $IN/synth_smoke_v1.SHA256SUMS ) && echo "DATASET STILL INTACT (hashes)" || echo "WARNING: dataset hash mismatch after run"
ls -la $OUT $OUT/cells
echo "=== DONE $(date -u +%FT%TZ)"
