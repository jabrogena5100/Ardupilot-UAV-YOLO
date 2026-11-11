#!/usr/bin/env python3
"""Extract the 2,149 smoke-positive official-test images of FASDD_UAV.zip (plus original and smoke-only labels) into a NEW directory and verify them.

    python3 eval/external_scripts/extract_fasdd_positive.py --zip /path/FASDD_UAV.zip --out local_data/external_eval/fasdd_uav_test_v1

Checks (all must pass, otherwise the directory is left as <out>_FAILED and the script exits 1):
  * the YOLO test list gives the pinned positive list (eval/fasdd_uav_test_smoke_positive_v1.txt, 2,149 names)
  * the MD5 of every extracted image equals the MD5 recorded when the test split was first extracted on the RunPod volume
    (local_data/external_eval/screening_v2/hashes_v2.json.gz), so a different copy of the dataset is detected
  * the smoke-only labels hold 2,877 smoke boxes (the COCO smoke count of the test split); class 1 -> 0, fire (class 0) dropped, coordinates untouched
The zip is only read. Class ids: 0 = fire, 1 = smoke (FASDD COCO categories).
"""
import argparse, gzip, hashlib, json, os, posixpath, shutil, sys, zipfile

LIST = "eval/fasdd_uav_test_smoke_positive_v1.txt"
HASHES = "local_data/external_eval/screening_v2/hashes_v2.json.gz"
POS_PREFIX = ("smoke_", "bothFireAndSmoke_")
EXPECT_BOXES = 2877


def extract(zip_path, out, pinned_list=LIST, hashes=HASHES, expect_boxes=EXPECT_BOXES):
    z = zipfile.ZipFile(zip_path)
    stem = lambda p: posixpath.splitext(posixpath.basename(p))[0]
    test = [stem(l.strip()) for l in z.read("annotations/YOLO_UAV/test.txt").decode().splitlines() if l.strip()]
    pos = sorted(s for s in test if s.startswith(POS_PREFIX))
    want = sorted(x.strip()[:-4] for x in open(pinned_list) if x.strip())
    if pos != want:
        raise SystemExit(f"positive list differs from {pinned_list}: {len(pos)} vs {len(want)}")
    md5 = json.load(gzip.open(hashes))["fasdd"] if hashes else None
    if os.path.exists(out) or os.path.exists(out + "_FAILED"):
        raise SystemExit(f"{out} (or {out}_FAILED) exists; refusing to overwrite")
    tmp = out + "_partial"
    for d in ("images", "labels_original", "labels_smoke_only"):
        os.makedirs(os.path.join(tmp, d))
    boxes = 0
    for s in pos:
        raw = z.read(f"images/{s}.jpg")
        if md5 is not None and hashlib.md5(raw).hexdigest() != md5[s]["md5"]:
            shutil.move(tmp, out + "_FAILED")
            raise SystemExit(f"MD5 mismatch for {s}: this is not the copy of FASDD_UAV that was screened; directory kept as {out}_FAILED")
        open(os.path.join(tmp, "images", s + ".jpg"), "wb").write(raw)
        lab = z.read(f"annotations/YOLO_UAV/labels/{s}.txt")
        open(os.path.join(tmp, "labels_original", s + ".txt"), "wb").write(lab)
        keep = []
        for ln in lab.decode().splitlines():
            p = ln.split()
            if len(p) >= 5 and int(p[0]) == 1:
                keep.append("0 " + " ".join(p[1:]))
        boxes += len(keep)
        open(os.path.join(tmp, "labels_smoke_only", s + ".txt"), "w").write("\n".join(keep) + ("\n" if keep else ""))
    if boxes != expect_boxes:
        shutil.move(tmp, out + "_FAILED")
        raise SystemExit(f"{boxes} smoke boxes, expected {expect_boxes}")
    os.rename(tmp, out)
    return {"images": len(pos), "smoke_boxes": boxes}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(extract(a.zip, a.out)))
