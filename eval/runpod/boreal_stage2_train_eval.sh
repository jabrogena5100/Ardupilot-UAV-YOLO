set +e
RUNS=/workspace/runs
RUN=$RUNS/real_boreal
SPL=/workspace/splits/boreal_v1
LOG=$RUNS/real_boreal_v1_train.log
mkdir -p $RUNS
echo "=== STAGE2 START $(date -u +%FT%TZ)"
if [ -e "$RUN" ]; then echo "ABORT: $RUN already exists (refusing to overwrite/auto-rename)"; sleep 600; exit 1; fi
pip install -q "ultralytics==8.4.173" 2>&1 | tail -2

echo "--- preflight: frozen split hashes"
python3 - <<'EOF'
import os, json, hashlib, sys
SPL="/workspace/splits/boreal_v1"; SRC="/workspace/boreal/boreal_yolo"
m=json.load(open(f"{SPL}/manifest.json"))
ok=True
for s,v in m["splits"].items():
    names=sorted(os.listdir(f"{SRC}/images/{s}"))
    h=hashlib.sha256("\n".join(names).encode()).hexdigest()
    good = (h==v["filelist_sha256"] and len(names)==v["images"])
    ok &= good
    print(s, len(names), "MATCH" if good else "MISMATCH")
# caveat: raw images with no label file (excluding Empty-Images = intentional negatives)
R="/workspace/boreal"; miss=[]
for site in ("Evo","Heinola","Karkkila","Ruokolahti"):
    im={os.path.splitext(f)[0] for f in os.listdir(f"{R}/{site}-Images")}
    lb={os.path.splitext(f)[0] for f in os.listdir(f"{R}/{site}-Labels")}
    miss += [f"{site}-Images/{x}" for x in sorted(im-lb)]
print("raw images without label file (excl. Empty):", len(miss)); print(miss)
json.dump({"raw_images_without_label_file": miss, "note": "recorded as caveat, NOT fixed (source is read-only)"}, open(f"{SPL}/caveats.json","w"), indent=2)
sys.exit(0 if ok else 3)
EOF
if [ $? -ne 0 ]; then echo "ABORT: split hash mismatch"; sleep 600; exit 1; fi

echo "--- env"
python3 - <<'EOF'
import ultralytics, torch, platform, subprocess
print("ultralytics", ultralytics.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(), "py", platform.python_version(), torch.cuda.get_device_name(0))
EOF
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "--- TRAIN START $(date -u +%FT%TZ)"
( while sleep 180; do
    if [ -f $RUN/results.csv ]; then echo "[progress $(date -u +%T)] $(tail -n1 $RUN/results.csv | cut -d, -f1,2,3,4,5,7,8,9,10,11 )"; fi
  done ) &
PROG=$!
T0=$(date +%s)
timeout 6000 yolo detect train model=yolov8s.pt data=$SPL/data.yaml \
  epochs=100 patience=100 imgsz=640 batch=32 seed=0 deterministic=True workers=8 plots=True \
  project=$RUNS name=real_boreal exist_ok=False > $LOG 2>&1
RC=$?
T1=$(date +%s)
kill $PROG 2>/dev/null
echo "TRAIN_EXIT_CODE=$RC TRAIN_WALL_SECONDS=$((T1-T0))  (124 = hit timeout)"
tail -c 1500 $LOG | tr '\r' '\n' | grep -v '^\s*$' | tail -8

if [ ! -f $RUN/weights/best.pt ]; then echo "NO best.pt - abort eval"; sleep 900; exit 1; fi
cp $RUN/weights/best.pt $RUN/real_boreal_best_v1.pt
sha256sum $RUN/real_boreal_best_v1.pt

echo "--- FINAL EVAL on frozen TEST set $(date -u +%FT%TZ)"
TRAIN_RC=$RC TRAIN_SECS=$((T1-T0)) python3 - <<'EOF'
import os, json, glob, random, shutil, hashlib, platform, subprocess, csv, yaml
import ultralytics, torch
from ultralytics import YOLO
RUN="/workspace/runs/real_boreal"; SPL="/workspace/splits/boreal_v1"
ck=f"{RUN}/real_boreal_best_v1.pt"
m=YOLO(ck)
r=m.val(data=f"{SPL}/data.yaml", split="test", imgsz=640, batch=32, seed=0, deterministic=True,
        project="/workspace/runs/real_boreal_eval", name="test_v1", exist_ok=False, plots=True, save_json=False, workers=8)
b=r.box
res={"split":"test","checkpoint":ck,"images":701,
     "precision":float(b.mp),"recall":float(b.mr),"mAP50":float(b.map50),"mAP50-95":float(b.map),
     "note":"P/R are Ultralytics mean P/R at the max-F1 confidence operating point; val conf=0.001, iou=0.7",
     "speed_ms":r.speed}
print("TEST_METRICS", json.dumps(res))
json.dump(res, open(f"{RUN}/test_metrics_v1.json","w"), indent=2)

# representative predictions: 12 evenly spaced test images, fixed seed, conf 0.25
imgs=sorted(open(f"{SPL}/test.txt").read().split())
random.Random(0).shuffle(imgs)
pick=sorted(imgs[:12])
m.predict(source=pick, imgsz=640, conf=0.25, save=True, project="/workspace/runs/real_boreal_eval", name="test_predictions_v1", exist_ok=False)
print("PRED_IMAGES", [os.path.basename(p) for p in pick])

# best epoch / curve summary from results.csv
rows=list(csv.DictReader(open(f"{RUN}/results.csv")))
rows=[{k.strip():v for k,v in x.items()} for x in rows]
best=max(rows,key=lambda x:float(x["metrics/mAP50-95(B)"]))
last=rows[-1]
print("EPOCHS_COMPLETED", len(rows), "BEST_EPOCH", best["epoch"], "best_val_mAP50-95", best["metrics/mAP50-95(B)"], "best_val_mAP50", best["metrics/mAP50(B)"])
print("LAST_EPOCH", last["epoch"], "val_mAP50-95", last["metrics/mAP50-95(B)"], "train_box", last["train/box_loss"], "train_cls", last["train/cls_loss"], "val_box", last["val/box_loss"], "val_cls", last["val/cls_loss"])

cfg={"experiment":"real_boreal_v1 (Model B, Level 1/2)",
     "train_rc":int(os.environ["TRAIN_RC"]),"train_wall_seconds":int(os.environ["TRAIN_SECS"]),
     "complete": int(os.environ["TRAIN_RC"])==0 and len(rows)==100,
     "model":"yolov8s.pt","epochs":100,"imgsz":640,"batch":32,"seed":0,"deterministic":True,"patience":100,
     "train_args":yaml.safe_load(open(f"{RUN}/args.yaml")),
     "split_manifest":json.load(open(f"{SPL}/manifest.json")),
     "dataset_caveats":json.load(open(f"{SPL}/caveats.json")),
     "dataset_notes":["single class: smoke (no fire class)","test set has 1 unlabeled image and no explicit negative images (all 256 Empty negatives are in train)","site distribution differs across splits"],
     "environment":{"ultralytics":ultralytics.__version__,"torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
                    "python":platform.python_version(),"gpu":torch.cuda.get_device_name(0),
                    "driver":subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),
                    "pip_freeze":subprocess.run(["pip","freeze"],capture_output=True,text=True).stdout.splitlines()},
     "checkpoint":ck,"checkpoint_sha256":hashlib.sha256(open(ck,"rb").read()).hexdigest(),
     "test_metrics":res}
json.dump(cfg, open(f"{RUN}/train_config_v1.json","w"), indent=2)
print("CONFIG_WRITTEN complete=", cfg["complete"])
EOF
echo "--- artifacts"
ls $RUN $RUN/weights /workspace/runs/real_boreal_eval/* 2>&1 | head -80
echo "--- source untouched check"
find /workspace/boreal -newer $SPL/data.yaml -type f 2>/dev/null | head -5
find /workspace/boreal -name '*.cache' 2>/dev/null | head -3
echo "=== STAGE2 DONE $(date -u +%FT%TZ)"
sleep 1800
