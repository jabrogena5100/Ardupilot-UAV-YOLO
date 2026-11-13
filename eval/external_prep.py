#!/usr/bin/env python3
"""Preparation tools for the external (FASDD_UAV) evaluation. No model is loaded; no detector output is used by any step.

  screen   deterministic 100-image annotation-screening sample + 30-image manual spot-check subset
  overlap  exact (MD5) and perceptual (dHash64) comparison of candidate images against the Boreal scan
  remap    class remap of YOLO labels: smoke -> 0, fire dropped (class ids are read from the dataset's own class list, never assumed)

All three take explicit directories/files because the on-disk layout of FASDD_UAV.zip is not yet verified (see docs/external_eval/readiness.md).
Selection uses only file names, a seed and SHA-256; it is reproducible and independent of model behaviour.
"""
import argparse, csv, gzip, hashlib, json, os, sys

import numpy as np

PREFIXES = ("FireAndSmoke", "NeitherFireNorSmoke", "Smoke", "Fire")   # canonical stratum names (the screening quotas use these)
# Real FASDD_UAV file names (verified in FASDD_UAV.zip, 2026-11-10) are lower-camel: bothFireAndSmoke_UAV000003.jpg, neitherFireNorSmoke_UAV000004.jpg,
# smoke_UAV000001.jpg, fire_UAV000005.jpg. Matching is case-insensitive on the file-name prefix and longest-match first; the capitalised
# record-description spellings stay accepted. Both map to the canonical names above.
_PREFIX_TO_STRATUM = (("bothfireandsmoke", "FireAndSmoke"), ("fireandsmoke", "FireAndSmoke"), ("neitherfirenorsmoke", "NeitherFireNorSmoke"),
                      ("smoke", "Smoke"), ("fire", "Fire"))
SMOKE_POS = {"Smoke", "FireAndSmoke"}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def stratum(name):
    low = os.path.basename(name).lower()
    for p, canon in _PREFIX_TO_STRATUM:
        if low.startswith(p):
            return canon
    return "UNKNOWN"


def rank_key(seed, salt, name):
    return hashlib.sha256(f"{seed}:{salt}:{name}".encode()).hexdigest()


def draw(names, n, seed, salt):
    return sorted(sorted(names, key=lambda x: rank_key(seed, salt, x))[:n])


def screening_sample(names, seed, quota=None, spot_quota=None):
    """names: candidate image file names (official test split). Default quota = 100 images: Smoke 20, FireAndSmoke 20, Fire 30, NeitherFireNorSmoke 30
    (the two negative-labelled strata get the most weight because label completeness there is the open question). Spot-check subset = 30 of the 100
    (6/6/9/9), drawn with an independent hash salt. Returns rows (name, stratum, spot_check)."""
    quota = quota or {"Smoke": 20, "FireAndSmoke": 20, "Fire": 30, "NeitherFireNorSmoke": 30}
    spot_quota = spot_quota or {"Smoke": 6, "FireAndSmoke": 6, "Fire": 9, "NeitherFireNorSmoke": 9}
    by = {}
    for nm in names:
        by.setdefault(stratum(nm), []).append(nm)
    rows = []
    for s, q in quota.items():
        have = by.get(s, [])
        if len(have) < q:
            raise SystemExit(f"stratum {s}: only {len(have)} candidates, need {q}; report and revise the protocol, do not pad from another stratum")
        pick = draw(have, q, seed, "screen")
        spot = set(draw(pick, spot_quota[s], seed, "spot"))
        rows += [(nm, s, int(nm in spot)) for nm in pick]
    return rows


def dhash64(path, resample="lanczos"):
    from PIL import Image
    flt = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR, "box": Image.BOX}[resample]
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((9, 8), flt), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def popcount64(x):
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def overlap(images, ref_md5, ref_dhash, tol=4, filters=("lanczos", "bilinear", "box")):
    """images: list of paths. ref_md5: set of hex md5; ref_dhash: dict name -> int. A candidate matches if its MD5 is in ref_md5, or if for any
    resampling filter the Hamming distance to any reference hash, or to that hash's complement (bit-convention guard), is <= tol.
    Returns per-image dicts. Crops, rotations and strong aspect changes are NOT detected (documented limitation)."""
    names = list(ref_dhash)
    ref = np.array([ref_dhash[n] for n in names], dtype=np.uint64)
    out = []
    for p in images:
        row = {"image": os.path.basename(p), "md5_match": md5(p) in ref_md5, "min_dist": 64, "nearest": ""}
        for f in filters:
            d = popcount64(ref ^ np.uint64(dhash64(p, f)))
            d = np.minimum(d, 64 - d)
            k = int(d.argmin())
            if d[k] < row["min_dist"]:
                row["min_dist"], row["nearest"] = int(d[k]), names[k]
        row["flagged"] = bool(row["md5_match"] or row["min_dist"] <= tol)
        out.append(row)
    return out


def remap_yolo(label_dir, out_dir, class_names):
    """class_names: ordered list from the dataset's class file. Requires exactly {'fire','smoke'} (case-insensitive). smoke -> 0, fire lines dropped.
    Writes one file per input (empty when no smoke box). Returns counts."""
    low = [c.strip().lower() for c in class_names]
    if sorted(low) != ["fire", "smoke"]:
        raise SystemExit(f"unexpected class names {class_names}; refusing to guess a mapping")
    smoke_id = low.index("smoke")
    os.makedirs(out_dir, exist_ok=True)
    c = {"files": 0, "smoke_boxes": 0, "fire_boxes_dropped": 0, "files_with_smoke": 0}
    for fn in sorted(os.listdir(label_dir)):
        if not fn.endswith(".txt") or fn == "classes.txt":
            continue
        keep = []
        with open(os.path.join(label_dir, fn)) as fi:
            lines = fi.readlines()
        for ln in lines:
            p = ln.split()
            if len(p) < 5:
                continue
            if int(p[0]) == smoke_id:
                keep.append("0 " + " ".join(p[1:]))
            else:
                c["fire_boxes_dropped"] += 1
        with open(os.path.join(out_dir, fn), "w") as fo:
            fo.write("\n".join(keep) + ("\n" if keep else ""))
        c["files"] += 1
        c["smoke_boxes"] += len(keep)
        c["files_with_smoke"] += bool(keep)
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--test-list", required=True, help="text file: one test-split image file name per line (derived from the dataset's own split)")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", required=True)
    o = sub.add_parser("overlap")
    o.add_argument("--images", required=True)
    o.add_argument("--boreal-scan", default="local_data/integrity/boreal_scan.json.gz")
    o.add_argument("--tol", type=int, default=4)
    o.add_argument("--out", required=True)
    r = sub.add_parser("remap")
    r.add_argument("--labels", required=True)
    r.add_argument("--classes", required=True, help="the dataset's class list file, one name per line in id order")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "screen":
        names = [x.strip() for x in open(a.test_list) if x.strip()]
        rows = screening_sample(names, a.seed)
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "stratum", "spot_check", "screener_smoke_visible", "screener_smoke_labeled", "screener_verdict", "owner_verdict", "notes"])
            for nm, st, sp in rows:
                w.writerow([nm, st, sp, "", "", "", "", ""])
        print(f"wrote {a.out}: {len(rows)} images, {sum(r[2] for r in rows)} spot-check")
    elif a.cmd == "overlap":
        scan = json.load(gzip.open(a.boreal_scan))
        rm, rd = set(), {}
        for sp, d in scan.items():
            for nm, v in d.items():
                rm.add(v[0]); rd[f"{sp}/{nm}"] = int(v[3], 16)
        imgs = sorted(os.path.join(a.images, f) for f in os.listdir(a.images) if f.lower().endswith(IMG_EXT))
        res = overlap(imgs, rm, rd, a.tol)
        json.dump({"tol_bits": a.tol, "n_images": len(res), "n_flagged": sum(x["flagged"] for x in res), "rows": res}, open(a.out, "w"), indent=1)
        print(f"{len(res)} images, {sum(x['flagged'] for x in res)} flagged -> {a.out}")
    else:
        names = [x.strip() for x in open(a.classes) if x.strip()]
        print(json.dumps(remap_yolo(a.labels, a.out, names)))


if __name__ == "__main__":
    main()
