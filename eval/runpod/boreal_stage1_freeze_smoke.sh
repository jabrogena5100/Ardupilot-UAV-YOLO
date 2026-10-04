set +e
echo "=== STAGE1 START $(date -u +%FT%TZ)"
pip install -q ultralytics 2>&1 | tail -2
python3 - <<'EOF'
import os, re, json, hashlib, collections, time, sys
SRC = "/workspace/boreal/boreal_yolo"
OUT = "/workspace/splits/boreal_v1"
vid = lambda n: re.sub(r"_?frame\d+$", "", os.path.splitext(n)[0])

# ---- freeze split (existing video-disjoint split, source untouched; symlink farm) ----
os.makedirs(OUT, exist_ok=True)
manifest = {"source": SRC, "split_origin": "pre-existing images/{train,val,test} in boreal_yolo, verified video-disjoint, 0 cross-split md5 duplicates",
            "splits": {}}
for s in ("train", "val", "test"):
    names = sorted(os.listdir(f"{SRC}/images/{s}"))
    for sub in ("images", "labels"):
        os.makedirs(f"{OUT}/{sub}/{s}", exist_ok=True)
    nlab = 0
    for n in names:
        d = f"{OUT}/images/{s}/{n}"
        if not os.path.lexists(d):
            os.symlink(f"{SRC}/images/{s}/{n}", d)
        l = os.path.splitext(n)[0] + ".txt"
        if os.path.exists(f"{SRC}/labels/{s}/{l}"):
            nlab += 1
            dl = f"{OUT}/labels/{s}/{l}"
            if not os.path.lexists(dl):
                os.symlink(f"{SRC}/labels/{s}/{l}", dl)
    with open(f"{OUT}/{s}.txt", "w") as f:
        f.write("\n".join(f"{OUT}/images/{s}/{n}" for n in names) + "\n")
    manifest["splits"][s] = {
        "images": len(names), "images_with_label_file": nlab, "background_images": len(names) - nlab,
        "videos": len({vid(n) for n in names}),
        "per_site": dict(collections.Counter(n.split("__")[0] for n in names)),
        "filelist_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest()}
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
open(f"{OUT}/data.yaml", "w").write(
    f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: smoke\n")
print("FROZEN SPLIT MANIFEST"); print(json.dumps(manifest, indent=2))
EOF

echo "--- image size sample"
python3 - <<'EOF'
import os, itertools
from PIL import Image
d = "/workspace/splits/boreal_v1/images/train"
for n in itertools.islice(sorted(os.listdir(d)), 0, 4000, 800):
    print(n, Image.open(f"{d}/{n}").size)
EOF
du -shL /workspace/splits/boreal_v1/images/train 2>/dev/null | tail -1

echo "--- ultralytics / env"
python3 -c "import ultralytics,torch;print('ultralytics',ultralytics.__version__,'torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.get_device_name(0))"

echo "--- SMOKE TRAIN (1 epoch) $(date -u +%FT%TZ)"
cd /workspace && mkdir -p runs
T0=$(date +%s)
timeout 1080 yolo detect train model=yolov8s.pt data=/workspace/splits/boreal_v1/data.yaml \
  epochs=1 imgsz=640 batch=32 seed=0 deterministic=True workers=8 \
  project=/workspace/runs/smoke_stage1 name=boreal_smoke exist_ok=True plots=False 2>&1 | tr '\r' '\n' | grep -vE '^\s*$' | tail -60
T1=$(date +%s)
echo "SMOKE_WALL_SECONDS=$((T1-T0)) (includes pip-free setup: model download, AMP check, dataset scan, 1 train epoch, 1 val, final val)"

echo "--- source dataset untouched check (files newer than split manifest)"
find /workspace/boreal -newer /workspace/splits/boreal_v1/data.yaml -type f 2>/dev/null | head
find /workspace/boreal -name '*.cache' 2>/dev/null | head
echo "--- smoke run artifacts"
ls /workspace/runs/smoke_stage1/boreal_smoke /workspace/runs/smoke_stage1/boreal_smoke/weights 2>&1
echo "=== STAGE1 DONE $(date -u +%FT%TZ)"
sleep 1800
