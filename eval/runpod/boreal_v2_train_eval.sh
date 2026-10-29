#!/bin/bash
# Boreal v2: materialise the frozen clip-disjoint split, train 3 new Model B seeds (the established protocol, unchanged), then evaluate the
# new Model B checkpoints and the EXISTING Model A checkpoints through the unified harness (eval/run_matrix.py) on the Boreal v2 test set.
# Runs on ONE RunPod pod (1x RTX 4090, volume `wildfire-datasets` at /workspace) under eval/runpod/pod_supervisor.py, which stops the pod.
# Never overwrites or modifies anything that exists: aborts if any target path exists; reads (never writes) the v1 split, datasets and checkpoints.
# Protocol per seed (identical to level2_seeds_train_and_matrix.sh / handoff section B.4): YOLOv8s from COCO yolov8s.pt, 100 epochs, patience 100,
# imgsz 640, batch 32, deterministic, workers 8, AdamW lr0 0.002 momentum 0.9 warmup_bias_lr 0.0 (what optimizer=auto resolved to for the v1 seed 0).
set +e
REPO=$(cd "$(dirname "$0")/../.." && pwd)            # <bundle>/repo
W=/workspace; IN=$W/incoming; RUNS=$W/runs; SYN=$W/synth_smoke_v1; SRC=$W/boreal/boreal_yolo
TAG=${RUN_TAG:-boreal_v2}; OUT=$RUNS/$TAG; SPL2=$W/splits/boreal_v2; SCAN=$W/integrity/boreal_scan.json.gz
SCAN_SHA=8a4db032e44c4040872d6b4863af9efefe7b2e5ac3b1f6e48b6f993a3e400e8e
T_START=$(date +%s)
echo "=== START $(date -u +%FT%TZ) tag=$TAG git=${GIT_COMMIT:-unknown} bundle=${BUNDLE_SHA256:-unknown}"
for p in "$OUT" "$SPL2.partial"; do
  [ -e "$p" ] && { echo "ABORT: $p exists (refusing to overwrite)"; exit 1; }; done
# $SPL2 (the frozen split) may exist from an earlier attempt: it is reused read-only after its file-list hashes are re-verified below

echo "--- env"
pip install -q --break-system-packages "ultralytics==8.4.173" pyyaml pillow numpy 2>&1 | tail -1
python3 - <<'PY' || { echo "ABORT: ultralytics/torch import failed"; exit 1; }
import ultralytics, torch, platform
print("ultralytics", ultralytics.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(), "py", platform.python_version(), torch.cuda.get_device_name(0))
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "HOST nproc=$(nproc) $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) mem=$(free -g | awk '/Mem:/{print $2}')GB"
mkdir -p $OUT

# Live progress + results endpoint (port 8000 via the RunPod HTTPS proxy): serves ONLY $EXP (status.json + manifest-listed small files), token required.
EXP=$OUT/export; mkdir -p $EXP; SRV_PID=""
echo '{"files": {}}' > $EXP/export_manifest.json
if [ -z "$NO_SERVE" ] && [ -n "$RESULTS_TOKEN" ]; then
  python3 $REPO/eval/runpod/results_server.py --root $EXP --port 8000 --serve-seconds 21000 > $OUT/results_server.log 2>&1 &
  SRV_PID=$!; echo "results endpoint started pid=$SRV_PID (token required; serves only $EXP)"
fi
(
  while true; do
    PHASE=$(cat $OUT/.phase 2>/dev/null || echo starting)
    python3 - "$OUT" "$PHASE" "$T_START" <<'PY' > $EXP/status.tmp 2>/dev/null && mv -f $EXP/status.tmp $EXP/status.json
import csv, glob, json, os, subprocess, sys, time
out, phase, t0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
seeds = {}
for d in sorted(glob.glob(out + "/real_boreal_v2_seed*/")):
    n = os.path.basename(d.rstrip("/"))
    try:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(d + "results.csv"))]
        last = rows[-1] if rows else {}
        seeds[n] = {"epochs_done": len(rows), "last_val_mAP50": last.get("metrics/mAP50(B)"), "last_val_mAP50-95": last.get("metrics/mAP50-95(B)"),
                    "elapsed_s_in_run": last.get("time")}
    except Exception:
        seeds[n] = {"epochs_done": 0}
logs = sorted(glob.glob(out + "/*_train.log")) + [out + "/run_matrix.log"]
tail = ""
for lg in reversed(logs):
    if os.path.exists(lg):
        tail = subprocess.run(["tail", "-c", "1500", lg], capture_output=True, text=True).stdout.replace("\r", "\n")[-1200:]
        break
print(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "phase": phase, "elapsed_s": int(time.time()) - t0, "seeds": seeds,
                  "log_tail": tail, "checkpoints": sorted(os.path.basename(p) for p in glob.glob(out + "/real_boreal_v2_seed*/real_boreal_v2_best_seed*.pt"))}))
PY
    sleep 30
  done
) &
STATUS_PID=$!

echo "--- integrity of existing artifacts (read-only)"
cd $REPO
python3 - <<'PY' || { echo "ABORT: existing checkpoint pin check failed"; exit 1; }
import yaml, hashlib, os, sys
cfg = yaml.safe_load(open("eval/datasets.yaml")); bad = 0
for n, m in cfg["models"].items():
    p = m["checkpoint"].replace("${RUNS}", "/workspace/runs")
    if not os.path.exists(p): print("MISSING", n, p); bad += 1; continue
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(("OK " if h == m["sha256"] else "MISMATCH "), n, h[:16]); bad += (h != m["sha256"])
sys.exit(1 if bad else 0)
PY
( cd $SYN && sha256sum -c --quiet $IN/synth_smoke_v1.SHA256SUMS ) && echo "SYNTH 10616 HASHES OK" || { echo "ABORT: synth hash mismatch"; exit 1; }
python3 -m unittest tests.test_eval_metrics tests.test_aggregate_seeds tests.test_run_matrix 2>&1 | tail -3

echo materialise > $OUT/.phase
if [ -e "$SPL2" ]; then
  echo "--- Boreal v2 split already on the volume: verifying against the frozen hashes (read-only reuse)"
  python3 - <<'PY' || { echo "ABORT: existing boreal_v2 split does not match the frozen hashes"; exit 3; }
import json, hashlib, os, sys
frozen = json.load(open("docs/experiments/boreal_v2/split_audit.json"))["filelist_sha256"]
for p in ("train", "val", "test"):
    names = sorted(os.listdir(f"/workspace/splits/boreal_v2/images/{p}"))
    h = hashlib.sha256("\n".join(names).encode()).hexdigest()
    print(p, len(names), "OK" if h == frozen[p] else "MISMATCH")
    if h != frozen[p]: sys.exit(1)
PY
else
echo "--- materialise Boreal v2 split"
python3 eval/runpod/boreal_v2_materialize.py --src $SRC --dest $SPL2 --filelists docs/experiments/boreal_v2/filelists.json \
  --scan $SCAN --scan-sha256 $SCAN_SHA \
  $(python3 -c "
import json; a=json.load(open('docs/experiments/boreal_v2/split_audit.json'))['filelist_sha256']
print(' '.join(f'--expect {p}={h}' for p,h in a.items()))") 2>&1 | tail -20
fi
[ -f $SPL2/data.yaml ] || { echo "ABORT: split materialisation failed"; exit 3; }
cp $SPL2/manifest.json $OUT/split_manifest_materialised.json
python3 - <<'PY'
import os, statistics
from PIL import Image
d = "/workspace/splits/boreal_v2/images/train"
for n in sorted(os.listdir(d))[::800][:4]:
    print("IMGPROBE", n, Image.open(os.path.join(d, n)).size)
PY

if [ "$CACHE_MODE" = "ram" ]; then
  echo "--- protocol deviation check: cache=ram must be pixel-identical and fit in RAM"
  python3 eval/runpod/cache_equivalence_check.py --images $SPL2/images/train --n 16 --out $OUT/PROTOCOL_DEVIATION.json --min-mem-gb 16 \
    || { echo "ABORT: cache=ram equivalence/memory check failed"; exit 3; }
fi
# Host-speed guard (same as level2_v2): after epoch 4 the mean of epochs 2-4 is compared with the per-run limit; a slow host aborts (exit 3).
GUARD_SECS=${GUARD_SECS:-12500}; LIM=${EPOCH_LIMIT_S:-60}
# CACHE_MODE: empty = the v1 protocol (no cache). 'ram' = Ultralytics cache=ram (speed-only deviation, recorded in PROTOCOL_DEVIATION.json).
CACHE_ARG=""; [ -n "$CACHE_MODE" ] && CACHE_ARG="cache=$CACHE_MODE"
train_one () {   # SEED
  local SEED=$1 NAME=real_boreal_v2_seed$1 CK=real_boreal_v2_best_seed$1.pt RUN=$OUT/real_boreal_v2_seed$1
  if [ $(( $(date +%s) - T_START )) -gt $GUARD_SECS ]; then echo "SKIP $NAME: time guard"; return; fi
  echo "train_seed_$SEED" > $OUT/.phase
  echo "--- train $NAME seed=$SEED $(date -u +%FT%TZ) load=$(cut -d' ' -f1 /proc/loadavg)"
  local T0=$(date +%s)
  timeout 9000 yolo detect train model=yolov8s.pt data=$SPL2/data.yaml epochs=100 patience=100 imgsz=640 batch=32 seed=$SEED \
    deterministic=True workers=8 plots=True optimizer=AdamW lr0=0.002 momentum=0.9 warmup_bias_lr=0.0 $CACHE_ARG \
    project=$OUT name=$NAME exist_ok=False > $OUT/${NAME}_train.log 2>&1 &
  local PID=$!; local CHECKED=0
  while kill -0 $PID 2>/dev/null; do
    sleep 20
    if [ $CHECKED -eq 0 ] && [ -f $RUN/results.csv ] && [ $(wc -l < $RUN/results.csv) -ge 5 ]; then
      CHECKED=1
      local MEAN=$(python3 -c "
import csv
r=[{k.strip():v for k,v in x.items()} for x in csv.DictReader(open('$RUN/results.csv'))]
t=[float(x['time']) for x in r]; d=[t[i]-t[i-1] for i in (1,2,3)]; print(round(sum(d)/3,1))")
      echo "SPEEDCHECK $NAME mean epoch 2-4 = ${MEAN}s (limit ${LIM}s) host load=$(cut -d' ' -f1 /proc/loadavg)"
      if python3 -c "import sys; sys.exit(0 if float('$MEAN') > float('$LIM') else 1)"; then
        echo "SLOW_HOST: aborting; re-create the pod"; kill $PID; sleep 5; pkill -f "[y]olo detect train"; exit 3
      fi
    fi
  done
  wait $PID; local RC=$?; local T1=$(date +%s)
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
print("EPOCHS",len(rows),"best_epoch_by_val_mAP50-95",best["epoch"],"val_mAP50",best["metrics/mAP50(B)"],"val_mAP50-95",best["metrics/mAP50-95(B)"],"median_epoch_s",round(ep[len(ep)//2],1))
json.dump({"split":"boreal_v2","seed":int(os.environ["SEED"]),"train_rc":int(os.environ["RC"]),"train_wall_seconds":int(os.environ["SECS"]),
  "epochs_completed":len(rows),"complete":int(os.environ["RC"])==0 and len(rows)==100,"best_epoch_by_val_mAP50-95":int(best["epoch"]),
  "median_epoch_seconds":ep[len(ep)//2],"checkpoint":os.environ["CK"],"checkpoint_sha256":os.environ["SHA"],
  "git_commit":os.environ.get("GIT_COMMIT","unknown"),"bundle_sha256":os.environ.get("BUNDLE_SHA256","unknown"),
  "train_args":yaml.safe_load(open(f"{RUN}/args.yaml")),
  "environment":{"ultralytics":ultralytics.__version__,"torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
    "python":platform.python_version(),"gpu":torch.cuda.get_device_name(0),"nproc":int(os.environ["NPROC"]),
    "driver":subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),
    "pip_freeze":subprocess.run(["pip","freeze"],capture_output=True,text=True).stdout.splitlines()}},
  open(f"{RUN}/train_config.json","w"),indent=2)
PY
}
for S in 0 1 2; do train_one $S; done
echo "TRAINING_DONE $(date -u +%FT%TZ) elapsed=$(( $(date +%s) - T_START ))s"

echo evaluate > $OUT/.phase
echo "--- pin checkpoints + evaluate (unified harness, same code path for every cell)"
cd $REPO
python3 - <<PY
import yaml, hashlib, os
cfg = yaml.safe_load(open("eval/datasets_boreal_v2.yaml")); avail = []
for name, m in cfg["models"].items():
    m["checkpoint"] = m["checkpoint"].replace("/boreal_v2/", "/$TAG/")
    path = m["checkpoint"].replace("\${RUNS}", "$RUNS")
    if os.path.exists(path):
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if m.get("sha256") and m["sha256"] != h: raise SystemExit(f"pinned hash mismatch for {name}")
        m["sha256"] = h; avail.append(name)
    else:
        print("MISSING checkpoint, excluded:", name)
yaml.safe_dump(cfg, open("$OUT/datasets_boreal_v2_pinned.yaml", "w"), sort_keys=False)
d = ["boreal_v2_test", "boreal_v2_test_pos", "boreal_v2_test_neg", "synth_test_pos", "synth_test_all"]
open("$OUT/cells.txt", "w").write(",".join(f"{a}:{b}" for a in avail for b in d))
print("models evaluated:", avail)
PY
python3 -m eval.run_matrix --config $OUT/datasets_boreal_v2_pinned.yaml --out $OUT/level2_boreal_v2 --bootstrap 1000 \
  --var BOREAL_V2_SPLIT=$SPL2 --var SYNTH_ROOT=$SYN --var RUNS=$RUNS --cells "$(cat $OUT/cells.txt)" > $OUT/run_matrix.log 2>&1
MRC=$?; tail -40 $OUT/run_matrix.log
python3 -m eval.aggregate_seeds --cells $OUT/level2_boreal_v2/cells --boreal-dataset boreal_v2_test \
  --out $OUT/level2_boreal_v2/seed_aggregation.md --json $OUT/level2_boreal_v2/seed_aggregation.json 2>&1 | tail -3

echo "--- post-run integrity"
python3 - <<'PY'
import yaml, hashlib, os, json
cfg = yaml.safe_load(open("eval/datasets.yaml")); rep = {"existing_checkpoints": {}}
for n, m in cfg["models"].items():
    p = m["checkpoint"].replace("${RUNS}", "/workspace/runs"); h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    rep["existing_checkpoints"][n] = {"sha256": h, "unchanged": h == m["sha256"]}
man = json.load(open("/workspace/splits/boreal_v2/manifest.json"))
rep["boreal_v2_filelist_sha256"] = {p: v["filelist_sha256"] for p, v in man["partitions"].items()}
rep["ok"] = all(v["unchanged"] for v in rep["existing_checkpoints"].values())
json.dump(rep, open("/workspace/runs/" + os.environ.get("RUN_TAG", "boreal_v2") + "/INTEGRITY.json", "w"), indent=2)
print("INTEGRITY ok =", rep["ok"])
PY
( cd $SYN && sha256sum -c --quiet $IN/synth_smoke_v1.SHA256SUMS ) && echo "DATASET STILL INTACT (hashes)" || echo "WARNING: dataset hash mismatch after run"

echo "--- results bundle (small artifacts only; checkpoints stay on the volume)"
cd $OUT && tar czf $OUT/results_bundle.tgz --exclude='*.pt' --exclude='weights' level2_boreal_v2 real_boreal_v2_seed*/results.csv real_boreal_v2_seed*/args.yaml \
  real_boreal_v2_seed*/train_config.json real_boreal_v2_seed*_train.log datasets_boreal_v2_pinned.yaml split_manifest_materialised.json INTEGRITY.json run_matrix.log $([ -f PROTOCOL_DEVIATION.json ] && echo PROTOCOL_DEVIATION.json) 2>/dev/null
sha256sum $OUT/results_bundle.tgz | tee $OUT/results_bundle.tgz.sha256; ls -la $OUT/results_bundle.tgz
n_ok=$(grep -l "\"complete\": true" $OUT/real_boreal_v2_seed*/train_config.json 2>/dev/null | wc -l)
JOB_RC=1; [ "$n_ok" = 3 ] && [ "$MRC" = 0 ] && JOB_RC=0
echo "=== JOB_RESULT $(date -u +%FT%TZ) total_elapsed=$(( $(date +%s) - T_START ))s trainings_complete=$n_ok matrix_rc=$MRC job_rc=$JOB_RC"

echo "--- export for retrieval (small artifacts only; never checkpoints; copies into the NEW export dir, originals untouched)"
echo export > $OUT/.phase
cp $OUT/results_bundle.tgz $OUT/results_bundle.tgz.sha256 $OUT/INTEGRITY.json $OUT/PROTOCOL_DEVIATION.json $EXP/ 2>/dev/null
for S in 0 1 2; do cp $OUT/real_boreal_v2_seed$S/train_config.json $EXP/train_config_seed$S.json 2>/dev/null; done
python3 - <<PY
import os, sys
sys.path.insert(0, "$REPO/eval/runpod")
import results_server as RS
names = [n for n in sorted(os.listdir("$EXP")) if n not in ("export_manifest.json", "status.json", "status.tmp")]
print("EXPORT_MANIFEST", {k: v["bytes"] for k, v in RS.write_manifest("$EXP", names).items()})
PY
echo "done_job_rc_$JOB_RC" > $OUT/.phase
if [ -n "$SRV_PID" ]; then
  W=${SERVE_SECONDS:-900}; echo "RESULTS READY: waiting up to ${W}s for the fetcher (POST /ack) $(date -u +%FT%TZ)"
  for i in $(seq 1 $W); do kill -0 $SRV_PID 2>/dev/null || { echo "results fetched/acknowledged"; break; }; sleep 1; done
  kill $SRV_PID 2>/dev/null
else
  echo "NOT serving (NO_SERVE set or RESULTS_TOKEN missing): fetch $OUT/results_bundle.tgz from the volume manually"
fi
kill $STATUS_PID 2>/dev/null
echo "=== DONE $(date -u +%FT%TZ) total_elapsed=$(( $(date +%s) - T_START ))s"
exit $JOB_RC
