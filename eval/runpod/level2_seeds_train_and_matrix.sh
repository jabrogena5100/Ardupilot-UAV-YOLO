#!/bin/bash
# Level 2 v2: two extra independent seeds (1, 2) for Model A and Model B, then every checkpoint through the harness.
# One RunPod pod (1x RTX 4090, volume `wildfire-datasets` at /workspace). Never overwrites existing runs or checkpoints.
# Prereq (uploaded by the operator): /workspace/incoming/repo_eval_v2.tar. /workspace/synth_smoke_v1 already on the volume.
# Protocol (identical for all four new runs): YOLOv8s COCO-pretrained, 100 epochs, patience 100, imgsz 640, batch 32,
# deterministic, workers 8, AdamW lr0 0.002 momentum 0.9 warmup_bias_lr 0.0 (what Model B's `optimizer=auto` resolved to).
set +e
W=/workspace; IN=$W/incoming; RUNS=$W/runs; SYN=$W/synth_smoke_v1; REPO=$W/repo_v2; OUT=$RUNS/level2_v2
SPL=$W/splits/boreal_v1; GUARD_SECS=${GUARD_SECS:-11000}
T_START=$(date +%s)
echo "=== START $(date -u +%FT%TZ)"
for p in "$REPO" "$OUT" $RUNS/synth_smoke_a_v1_seed1 $RUNS/synth_smoke_a_v1_seed2 $RUNS/real_boreal_seed1 $RUNS/real_boreal_seed2; do
  [ -e "$p" ] && { echo "ABORT: $p exists (refusing to overwrite)"; exit 1; }; done

echo "--- env"
pip install -q --break-system-packages "ultralytics==8.4.173" pyyaml 2>&1 | tail -1
python3 - <<'PY' || { echo "ABORT: ultralytics/torch import failed"; exit 1; }
import ultralytics, torch, platform
print("ultralytics", ultralytics.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(), "py", platform.python_version(), torch.cuda.get_device_name(0))
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "HOST nproc=$(nproc) $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) mem=$(free -g | awk '/Mem:/{print $2}')GB"
mkdir -p $OUT $REPO && tar xf $IN/repo_eval_v2.tar -C $REPO || { echo "ABORT: repo tar"; exit 1; }
python3 - <<'PY'
import os, glob, statistics
from PIL import Image
def probe(tag, d, n=300):
    fs = sorted(os.listdir(d))[:n]
    sz = [os.path.getsize(os.path.join(d, f)) for f in fs]
    dims = [Image.open(os.path.join(d, f)).size for f in fs[:20]]
    print(f"IMGPROBE {tag}: mean file size {statistics.mean(sz)/1e3:.0f} KB over {len(fs)} files; first-20 dims {sorted(set(dims))[:4]}")
probe("boreal_train", "/workspace/splits/boreal_v1/images/train")
probe("synth_train", "/workspace/synth_smoke_v1/images/train")
PY

echo "--- integrity of inputs"
( cd $SYN && sha256sum -c --quiet $IN/synth_smoke_v1.SHA256SUMS ) && echo "SYNTH 10616 HASHES OK" || { echo "ABORT: synth hash mismatch"; exit 1; }
echo "1dc35ef293b6b35bb2dadaf5b6d5887ece89ca9d55a445bfdde258286c8355b6  $RUNS/real_boreal/real_boreal_best_v1.pt" | sha256sum -c || { echo "ABORT: Model B seed0 hash"; exit 1; }
echo "1da10c4814a905d92deec2d9c36185f872eabea447d47c389b9b065468009564  $RUNS/synth_smoke_a_v1/synth_smoke_best_v1.pt" | sha256sum -c || { echo "ABORT: Model A seed0 hash"; exit 1; }
cd $REPO && python3 -m unittest tests.test_eval_metrics tests.test_aggregate_seeds 2>&1 | tail -3

train_one () {   # RUNNAME DATA SEED CKPTNAME
  local NAME=$1 DATA=$2 SEED=$3 CK=$4 RUN=$RUNS/$1
  if [ $(( $(date +%s) - T_START )) -gt $GUARD_SECS ]; then echo "SKIP $NAME: time guard"; return; fi
  echo "--- train $NAME seed=$SEED $(date -u +%FT%TZ) load=$(cut -d' ' -f1 /proc/loadavg)"
  local T0=$(date +%s)
  timeout 6000 yolo detect train model=yolov8s.pt data=$DATA epochs=100 patience=100 imgsz=640 batch=32 seed=$SEED \
    deterministic=True workers=8 plots=True optimizer=AdamW lr0=0.002 momentum=0.9 warmup_bias_lr=0.0 \
    project=$RUNS name=$NAME exist_ok=False > $RUNS/${NAME}_train.log 2>&1
  local RC=$?; local T1=$(date +%s)
  echo "TRAIN_EXIT $NAME rc=$RC wall=$((T1-T0))s"
  [ -f $RUN/weights/best.pt ] || { echo "NO best.pt for $NAME"; return; }
  cp $RUN/weights/best.pt $RUN/$CK
  local SHA=$(sha256sum $RUN/$CK | cut -d' ' -f1); echo "CKPT $NAME $CK sha256=$SHA"
  RUN=$RUN CK=$CK SEED=$SEED RC=$RC SECS=$((T1-T0)) SHA=$SHA NPROC=$(nproc) python3 - <<'PY'
import os, json, csv, yaml, platform, subprocess, ultralytics, torch
RUN=os.environ["RUN"]
rows=[{k.strip():v for k,v in r.items()} for r in csv.DictReader(open(f"{RUN}/results.csv"))]
best=max(rows,key=lambda x:float(x["metrics/mAP50-95(B)"])); t=[float(r["time"]) for r in rows]
ep=sorted(t[i]-t[i-1] for i in range(1,len(t)))
print("EPOCHS",len(rows),"best_epoch",best["epoch"],"val_mAP50",best["metrics/mAP50(B)"],"val_mAP50-95",best["metrics/mAP50-95(B)"],"median_epoch_s",round(ep[len(ep)//2],1))
json.dump({"seed":int(os.environ["SEED"]),"train_rc":int(os.environ["RC"]),"train_wall_seconds":int(os.environ["SECS"]),
  "epochs_completed":len(rows),"complete":int(os.environ["RC"])==0 and len(rows)==100,"best_epoch":int(best["epoch"]),
  "median_epoch_seconds":ep[len(ep)//2],"checkpoint":os.environ["CK"],"checkpoint_sha256":os.environ["SHA"],
  "train_args":yaml.safe_load(open(f"{RUN}/args.yaml")),
  "environment":{"ultralytics":ultralytics.__version__,"torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
    "python":platform.python_version(),"gpu":torch.cuda.get_device_name(0),"nproc":int(os.environ["NPROC"]),
    "driver":subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),
    "pip_freeze":subprocess.run(["pip","freeze"],capture_output=True,text=True).stdout.splitlines()}},
  open(f"{RUN}/train_config.json","w"),indent=2)
PY
}
train_one synth_smoke_a_v1_seed1 $SYN/data_rw.yaml 1 synth_smoke_best_v1_seed1.pt
train_one synth_smoke_a_v1_seed2 $SYN/data_rw.yaml 2 synth_smoke_best_v1_seed2.pt
train_one real_boreal_seed1 $SPL/data.yaml 1 real_boreal_best_v1_seed1.pt
train_one real_boreal_seed2 $SPL/data.yaml 2 real_boreal_best_v1_seed2.pt
echo "TRAINING_DONE $(date -u +%FT%TZ) elapsed=$(( $(date +%s) - T_START ))s"

echo "--- pin checkpoints + evaluate"
cd $REPO
python3 - <<PY
import yaml, hashlib, os
cfg = yaml.safe_load(open("eval/datasets.yaml"))
runs = "$RUNS"; avail = []
for name, m in cfg["models"].items():
    path = m["checkpoint"].replace("\${RUNS}", runs)
    if os.path.exists(path):
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if m.get("sha256") and m["sha256"] != h: raise SystemExit(f"pinned hash mismatch for {name}")
        m["sha256"] = h; avail.append(name)
    else:
        print("MISSING checkpoint, excluded:", name)
yaml.safe_dump(cfg, open("$OUT/datasets_pinned.yaml", "w"), sort_keys=False)
open("$OUT/available_models.txt", "w").write(",".join(avail))
print("models evaluated:", avail)
PY
CELLS=$(python3 - <<PY
m = open("$OUT/available_models.txt").read().split(",")
d = ["boreal_test", "synth_test_pos", "synth_test_all", "boreal_empty_train_diag"]
print(",".join(f"{a}:{b}" for a in m for b in d))
PY
)
python3 -m eval.run_matrix --config $OUT/datasets_pinned.yaml --out $OUT \
  --var BOREAL_SPLIT=$SPL --var SYNTH_ROOT=$SYN --var RUNS=$RUNS --cells "$CELLS" 2>&1 | tail -120
( cd $SYN && sha256sum -c --quiet $IN/synth_smoke_v1.SHA256SUMS ) && echo "DATASET STILL INTACT (hashes)" || echo "WARNING: dataset hash mismatch after run"
echo "$(sha256sum $RUNS/real_boreal/real_boreal_best_v1.pt | cut -c1-16) / $(sha256sum $RUNS/synth_smoke_a_v1/synth_smoke_best_v1.pt | cut -c1-16) seed0 checkpoints re-hashed"
ls $OUT/cells | wc -l
echo "=== DONE $(date -u +%FT%TZ) total_elapsed=$(( $(date +%s) - T_START ))s"
