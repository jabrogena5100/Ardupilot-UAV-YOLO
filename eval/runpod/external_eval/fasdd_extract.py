import collections, hashlib, io, json, os, posixpath, shutil, sys, time, zipfile
from concurrent.futures import ProcessPoolExecutor

ZIP = os.environ.get("FZ", "/workspace/datasets/FASDD_UAV.zip")
ROOT = os.environ.get("FROOT", "/workspace/external_eval")
RUNS = os.environ.get("FRUNS", "/workspace/runs/fasdd_extract")
SHA_EXPECT = os.environ.get("FSHA", "7f4202a893e98f6ee3960f268aaeaf2fdeb277fa7305440125069cb158cbd0b2")
NEXP = int(os.environ.get("FN", "4181"))
FINAL = os.path.join(ROOT, "fasdd_uav_test_v1")
PART = FINAL + "_partial"
T0 = time.time()
R = {}


def emit(k, **kw):
    R[k] = kw
    print(json.dumps({"stage": k, "t": round(time.time() - T0, 1), **kw}, default=str), flush=True)


stem = lambda p: posixpath.splitext(posixpath.basename(p))[0]


def hashes(path):
    """md5 + dHash64 (lanczos/bilinear/box), bit order identical to eval/external_prep.py::dhash64; also actual size."""
    from PIL import Image
    data = open(path, "rb").read()
    out = {"md5": hashlib.md5(data).hexdigest()}
    with Image.open(path) as im:
        out["size"] = list(im.size)
        g = im.convert("L")
        for nm, f in (("lanczos", Image.LANCZOS), ("bilinear", Image.BILINEAR), ("box", Image.BOX)):
            px = list(g.resize((9, 8), f).tobytes())
            v = 0
            for r in range(8):
                for c in range(8):
                    v = (v << 1) | int(px[r * 9 + c + 1] > px[r * 9 + c])
            out[nm] = "%016x" % v
    return out


def main():
    os.makedirs(RUNS, exist_ok=True)
    try:
        fd = os.open(os.path.join(RUNS, "STARTED"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(time.time()).encode()); os.close(fd)
    except FileExistsError:
        print("GUARD: a previous run already started (marker exists); exiting without doing anything", flush=True)
        return
    if os.path.exists(FINAL) or os.path.exists(PART):
        emit("abort", reason="output directory already exists; refusing to overwrite", final=os.path.exists(FINAL), partial=os.path.exists(PART))
        return
    os.makedirs(ROOT, exist_ok=True)
    st0 = os.stat(ZIP)
    emit("pre", zip_bytes=st0.st_size, zip_mtime=st0.st_mtime, ls_workspace=sorted(os.listdir("/workspace")) if os.path.isdir("/workspace") else None,
         ls_external_eval=sorted(os.listdir(ROOT)))
    z = zipfile.ZipFile(ZIP)
    test = [stem(l.strip()) for l in z.read("annotations/YOLO_UAV/test.txt").decode().splitlines() if l.strip()]
    assert len(test) == len(set(test)) == NEXP, (len(test), len(set(test)))
    tset = set(test)
    coco = json.loads(z.read("annotations/COCO_UAV/Annotations/test.json"))
    cdim = {stem(i["file_name"]): (i["width"], i["height"]) for i in coco["images"]}
    assert set(cdim) == tset

    # --- find the out-of-range YOLO box (all labels, read-only) and the VOC size mismatches (test split, vs COCO)
    import xml.etree.ElementTree as ET
    oor = []
    allnames = [n for n in z.namelist() if n.startswith("annotations/YOLO_UAV/labels/") and n.endswith(".txt")]
    for n in allnames:
        for k, ln in enumerate(z.read(n).decode().splitlines()):
            p = ln.split()
            if len(p) == 5:
                c = int(p[0]); x, y, w, h = map(float, p[1:])
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    oor.append({"label_file": n, "line_no": k + 1, "line": ln, "class": c, "in_test_split": stem(n) in tset,
                                "x_minus_w2": round(x - w / 2, 5), "x_plus_w2": round(x + w / 2, 5), "y_minus_h2": round(y - h / 2, 5), "y_plus_h2": round(y + h / 2, 5)})
    emit("out_of_range_yolo_boxes", n=len(oor), boxes=oor)
    vm = []
    for s in sorted(tset):
        r = ET.fromstring(z.read(f"annotations/VOC_UAV/Annotations/{s}.xml"))
        sz = r.find("size")
        vs = (int(sz.findtext("width")), int(sz.findtext("height")))
        if vs != tuple(cdim[s]):
            vm.append({"image": s, "voc_size": vs, "coco_size": cdim[s], "voc_depth": sz.findtext("depth"), "voc_filename": r.findtext("filename")})
    emit("voc_size_mismatch_test", n=len(vm), items=vm)

    # --- extract the 4181 test images + original annotations into a NEW directory
    for d in ("images", "labels_original", "labels_smoke_only", "annotations_coco", "annotations_voc"):
        os.makedirs(os.path.join(PART, d))
    stats = {"images": 0, "bytes_images": 0, "fire_dropped": 0, "smoke_kept": 0, "smoke_files": 0}
    for s in test:
        data = z.read(f"images/{s}.jpg")                      # CRC checked by zipfile on read
        open(os.path.join(PART, "images", s + ".jpg"), "wb").write(data)
        stats["images"] += 1; stats["bytes_images"] += len(data)
        raw = z.read(f"annotations/YOLO_UAV/labels/{s}.txt")
        open(os.path.join(PART, "labels_original", s + ".txt"), "wb").write(raw)
        keep = []
        for ln in raw.decode().splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            if int(p[0]) == 1:
                keep.append("0 " + " ".join(p[1:])); stats["smoke_kept"] += 1
            elif int(p[0]) == 0:
                stats["fire_dropped"] += 1
            else:
                raise SystemExit("unexpected class id " + p[0])
        stats["smoke_files"] += bool(keep)
        open(os.path.join(PART, "labels_smoke_only", s + ".txt"), "w").write("\n".join(keep) + ("\n" if keep else ""))
        open(os.path.join(PART, "annotations_voc", s + ".xml"), "wb").write(z.read(f"annotations/VOC_UAV/Annotations/{s}.xml"))
    open(os.path.join(PART, "annotations_coco", "test.json"), "wb").write(z.read("annotations/COCO_UAV/Annotations/test.json"))
    open(os.path.join(PART, "test.txt"), "wb").write(z.read("annotations/YOLO_UAV/test.txt"))
    emit("extract", **stats)

    # --- hashes + actual image sizes (parallel), written outside the dataset dir
    paths = [os.path.join(PART, "images", s + ".jpg") for s in test]
    with ProcessPoolExecutor(4) as ex:
        hs = list(ex.map(hashes, paths, chunksize=32))
    H = {s: h for s, h in zip(test, hs)}
    json.dump(H, open(os.path.join(RUNS, "fasdd_test_hashes.json"), "w"))
    szm = [s for s in test if tuple(H[s]["size"]) != tuple(cdim[s])]
    emit("hash", n=len(H), md5_unique=len({h["md5"] for h in hs}), image_size_vs_coco_mismatch=len(szm), mismatch_examples=[{"image": s, "actual": H[s]["size"], "coco": cdim[s]} for s in szm[:10]])

    # --- verification
    li = sorted(os.listdir(os.path.join(PART, "images"))); lo = sorted(os.listdir(os.path.join(PART, "labels_original"))); ls_ = sorted(os.listdir(os.path.join(PART, "labels_smoke_only")))
    vox = sorted(os.listdir(os.path.join(PART, "annotations_voc")))
    sti = [x[:-4] for x in li]
    ok_names = sti == sorted(test) == [x[:-4] for x in lo] == [x[:-4] for x in ls_] == [x[:-4] for x in vox]
    # pairing: re-derive smoke labels from original and compare; count empty
    bad_pair = 0
    for s in test:
        o = [l.split() for l in open(os.path.join(PART, "labels_original", s + ".txt")).read().splitlines() if l.strip()]
        sm = [l.split() for l in open(os.path.join(PART, "labels_smoke_only", s + ".txt")).read().splitlines() if l.strip()]
        want = [["0"] + p[1:] for p in o if p[0] == "1"]
        if sm != want:
            bad_pair += 1
    # COCO smoke annotation count for the test split vs our smoke boxes
    coco_smoke = sum(1 for a in coco["annotations"] if a["category_id"] == 1)
    cats = collections.Counter()
    for s in test:
        cl = {l.split()[0] for l in open(os.path.join(PART, "labels_original", s + ".txt")).read().splitlines() if l.strip()}
        cats["both" if cl == {"0", "1"} else "smoke_only" if cl == {"1"} else "fire_only" if cl == {"0"} else "empty"] += 1
    tot = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(PART) for f in fs)
    du = shutil.disk_usage("/workspace" if os.path.isdir("/workspace") else ROOT)
    emit("verify", n_images=len(li), n_labels_original=len(lo), n_labels_smoke=len(ls_), n_voc=len(vox), names_identical_across_dirs=ok_names,
         duplicate_names=len(test) - len(set(test)), smoke_labels_mismatch_vs_rederived=bad_pair, smoke_boxes_extracted=stats["smoke_kept"], coco_smoke_annotations=coco_smoke,
         smoke_boxes_match_coco=stats["smoke_kept"] == coco_smoke, categories=dict(cats), extracted_bytes=tot, extracted_gb=round(tot / 1e9, 3), vol_free_gb=round(du.free / 1e9, 1))

    # --- dHash clustering inside the test split (no Boreal needed): connected components at several Hamming radii
    names = test
    for key in ("lanczos",):
        v = [int(H[s][key], 16) for s in names]
        n = len(v)
        pos = {s for s in names if any(l.startswith("1 ") for l in open(os.path.join(PART, "labels_original", s + ".txt")).read().splitlines())}
        edges = [(i, j, (v[i] ^ v[j]).bit_count()) for i in range(n) for j in range(i + 1, n) if (v[i] ^ v[j]).bit_count() <= 8]
        for tol in (0, 2, 4, 6, 8):
            par = list(range(n))
            def f(a):
                while par[a] != a:
                    par[a] = par[par[a]]; a = par[a]
                return a
            for i, j, d in edges:
                if d <= tol:
                    par[f(i)] = f(j)
            comp = collections.defaultdict(list)
            for i in range(n):
                comp[f(i)].append(i)
            sizes = sorted((len(c) for c in comp.values()), reverse=True)
            pc = [c for c in comp.values() if any(names[i] in pos for i in c)]
            emit(f"cluster_{key}_tol{tol}", clusters_all=len(comp), largest=sizes[:8], singletons=sum(1 for x in sizes if x == 1),
                 clusters_with_smoke_positive=len(pc), size_median=sizes[len(sizes) // 2])
    # exact duplicates by md5 inside test
    md = collections.Counter(H[s]["md5"] for s in names)
    emit("exact_dupes_within_test", dup_groups=sum(1 for x in md.values() if x > 1))

    # --- seal: zip untouched?
    st1 = os.stat(ZIP)
    h = hashlib.sha256()
    with open(ZIP, "rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    emit("zip_after", bytes=st1.st_size, mtime_unchanged=st1.st_mtime == st0.st_mtime, sha256=h.hexdigest(), sha256_matches_recorded=h.hexdigest() == SHA_EXPECT)
    ok = ok_names and bad_pair == 0 and len(li) == NEXP and h.hexdigest() == SHA_EXPECT
    if ok:
        open(os.path.join(PART, "EXTRACTION_COMPLETE.json"), "w").write(json.dumps({"images": len(li), "zip_sha256": h.hexdigest(), "tool": "fasdd_extract.py"}))
        os.rename(PART, FINAL)
        emit("sealed", final=FINAL)
    else:
        emit("NOT_SEALED", reason="verification failed; partial dir left in place for inspection", partial=PART)
    json.dump(R, open(os.path.join(RUNS, "report.json"), "w"), indent=1, default=str)
    emit("done")


if __name__ == "__main__":
    main()
