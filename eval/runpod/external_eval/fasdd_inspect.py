#!/usr/bin/env python3
"""Read-only FASDD_UAV.zip inspection. Prints JSON lines to stdout (pod logs). No extraction to disk; only a small report is written."""
import collections, hashlib, io, json, os, posixpath, random, re, shutil, sys, threading, time, traceback, zipfile

ZIP = "/workspace/datasets/FASDD_UAV.zip"
EXPECT = 14888184890
OUT = "/workspace/runs/fasdd_inspect"
T0 = time.time()
REPORT = {}


def emit(stage, **kw):
    kw = {"stage": stage, "t": round(time.time() - T0, 1), **kw}
    print(json.dumps(kw, default=str), flush=True)
    REPORT.setdefault(stage, {}).update({k: v for k, v in kw.items() if k not in ("stage",)})


def guarded(fn):
    def w(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            emit("error_" + fn.__name__, error=repr(e), tb=traceback.format_exc()[-800:])
    return w


@guarded
def stage_size():
    st = os.stat(ZIP)
    du = shutil.disk_usage("/workspace")
    emit("size", bytes=st.st_size, expected=EXPECT, match=st.st_size == EXPECT, vol_total_gb=round(du.total / 1e9, 1),
         vol_used_gb=round(du.used / 1e9, 1), vol_free_gb=round(du.free / 1e9, 1))
    emit("ls_datasets", files={f: os.path.getsize(os.path.join("/workspace/datasets", f)) for f in os.listdir("/workspace/datasets")})


SHA = {}


@guarded
def sha_job():
    h = hashlib.sha256()
    with open(ZIP, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b)
    SHA["v"] = h.hexdigest()
    emit("sha256", sha256=SHA["v"])


def classify(n):
    return n.lower().rsplit(".", 1)[-1] if "." in n else ""


@guarded
def stage_structure(z):
    infos = z.infolist()
    ext = collections.Counter()
    ext_bytes = collections.Counter()
    top = collections.Counter()
    dirs = collections.Counter()
    for i in infos:
        if i.is_dir():
            continue
        e = classify(i.filename)
        ext[e] += 1
        ext_bytes[e] += i.file_size
        parts = i.filename.split("/")
        top["/".join(parts[:1])] += 1
        dirs["/".join(parts[:-1])] += 1
    emit("structure", entries=len(infos), compressed_bytes=sum(i.compress_size for i in infos),
         uncompressed_bytes=sum(i.file_size for i in infos), ext_count=dict(ext), ext_uncompressed_bytes=dict(ext_bytes),
         top_level=dict(top), zip64=any(i.header_offset > 0xFFFFFFFF for i in infos),
         compress_types=dict(collections.Counter(i.compress_type for i in infos)))
    emit("dirs", dirs=dict(dirs.most_common(60)), n_dirs=len(dirs))
    emit("first_names", names=[i.filename for i in infos[:15]])
    return infos


@guarded
def stage_text(z, infos):
    texts = [i for i in infos if not i.is_dir() and classify(i.filename) in ("yaml", "yml", "txt", "json", "csv", "md", "names", "xml") and i.file_size < 3_000_000]
    # classify text-like files by directory: split lists, class files, README, but skip per-image labels/annotations
    bydir = collections.defaultdict(list)
    for i in texts:
        bydir[posixpath.dirname(i.filename)].append(i)
    small_dirs = {d: v for d, v in bydir.items() if len(v) <= 12}
    big_dirs = {d: len(v) for d, v in bydir.items() if len(v) > 12}
    emit("text_dirs", n_text=len(texts), many_file_dirs=big_dirs, few_file_dirs={d: [posixpath.basename(i.filename) for i in v] for d, v in small_dirs.items()})
    shown = 0
    for d, v in sorted(small_dirs.items()):
        for i in v:
            if shown >= 25:
                return
            raw = z.read(i)
            try:
                s = raw.decode("utf-8", "replace")
            except Exception:
                continue
            lines = s.splitlines()
            ext = classify(i.filename)
            if ext == "json" and i.file_size > 200_000:
                try:
                    j = json.loads(s)
                    emit("json_file", name=i.filename, size=i.file_size, keys=list(j)[:10] if isinstance(j, dict) else "list",
                         categories=j.get("categories") if isinstance(j, dict) else None,
                         n_images=len(j.get("images", [])) if isinstance(j, dict) else None,
                         n_annotations=len(j.get("annotations", [])) if isinstance(j, dict) else None,
                         image_keys=list(j["images"][0]) if isinstance(j, dict) and j.get("images") else None,
                         ann_keys=list(j["annotations"][0]) if isinstance(j, dict) and j.get("annotations") else None,
                         first_image=j["images"][0] if isinstance(j, dict) and j.get("images") else None,
                         first_ann=j["annotations"][0] if isinstance(j, dict) and j.get("annotations") else None,
                         info=j.get("info") if isinstance(j, dict) else None, licenses=j.get("licenses") if isinstance(j, dict) else None)
                except Exception as e:
                    emit("json_parse_fail", name=i.filename, error=repr(e))
            else:
                emit("text_file", name=i.filename, size=i.file_size, n_lines=len(lines), head=lines[:12] if ext != "txt" or len(lines) < 12 else lines[:4])
            shown += 1


@guarded
def stage_yolo(z, infos):
    """All per-image label .txt files (YOLO): class histogram, box geometry, edge-touch, empty files, per-top-dir (split) stats."""
    img_ext = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
    imgs = [i for i in infos if not i.is_dir() and classify(i.filename) in img_ext]
    stems_img = collections.defaultdict(list)
    for i in imgs:
        stems_img[posixpath.splitext(posixpath.basename(i.filename))[0]].append(i.filename)
    labels = [i for i in infos if not i.is_dir() and classify(i.filename) == "txt" and "label" in i.filename.lower()]
    emit("yolo_candidates", n_images=len(imgs), n_label_txt=len(labels), dup_image_stems=sum(1 for v in stems_img.values() if len(v) > 1),
         example_images=[i.filename for i in imgs[:3]], example_labels=[i.filename for i in labels[:3]])
    if not labels:
        return
    # group key = directory path with the last component (images/labels) normalised away
    stats = collections.defaultdict(lambda: {"files": 0, "empty": 0, "boxes": 0, "cls": collections.Counter(), "bad_lines": 0,
                                             "w": [], "h": [], "area": [], "edge": 0, "nobox_ge2cols": 0, "cls_by_prefix": collections.defaultdict(collections.Counter)})
    label_stems = {}
    for i in labels:
        key = posixpath.dirname(i.filename)
        st = stats[key]
        st["files"] += 1
        stem = posixpath.splitext(posixpath.basename(i.filename))[0]
        label_stems.setdefault(key, set()).add(stem)
        s = z.read(i).decode("utf-8", "replace").strip()
        prefix = re.match(r"[A-Za-z]+", stem)
        prefix = prefix.group(0) if prefix else "?"
        if not s:
            st["empty"] += 1
            st["cls_by_prefix"][prefix]["<empty>"] += 1
            continue
        for ln in s.splitlines():
            p = ln.split()
            if len(p) != 5:
                st["bad_lines"] += 1
                continue
            try:
                c = int(float(p[0])); x, y, w, h = map(float, p[1:])
            except ValueError:
                st["bad_lines"] += 1
                continue
            st["boxes"] += 1
            st["cls"][c] += 1
            st["cls_by_prefix"][prefix][c] += 1
            if len(st["w"]) < 40000:
                st["w"].append(w); st["h"].append(h); st["area"].append(w * h)
            if x - w / 2 <= 0.005 or x + w / 2 >= 0.995 or y - h / 2 <= 0.005 or y + h / 2 >= 0.995:
                st["edge"] += 1

    def q(v):
        if not v:
            return None
        v = sorted(v); n = len(v)
        return {k: round(v[min(n - 1, int(f * (n - 1)))], 4) for k, f in (("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95))}
    for key, st in sorted(stats.items()):
        # pair labels with images located in the sibling image dir
        emit("yolo_group", dir=key, label_files=st["files"], empty_labels=st["empty"], boxes=st["boxes"], bad_lines=st["bad_lines"],
             class_hist=dict(st["cls"]), edge_touch_frac=round(st["edge"] / st["boxes"], 3) if st["boxes"] else None,
             w=q(st["w"]), h=q(st["h"]), area=q(st["area"]),
             cls_by_name_prefix={k: dict(v) for k, v in st["cls_by_prefix"].items()})
    all_label_stems = set().union(*label_stems.values())
    emit("yolo_pairing", images_without_label=sum(1 for s in stems_img if s not in all_label_stems),
         labels_without_image=sum(1 for s in all_label_stems if s not in stems_img))
    pref = collections.Counter(re.match(r"[A-Za-z]+", s).group(0) if re.match(r"[A-Za-z]+", s) else "?" for s in stems_img)
    emit("name_prefixes", prefixes=dict(pref.most_common(12)))
    # which split dir holds which image dir
    imgdirs = collections.Counter(posixpath.dirname(i.filename) for i in imgs)
    emit("image_dirs", dirs=dict(imgdirs))
    # cross-split image-stem duplication
    owner = collections.defaultdict(set)
    for i in imgs:
        owner[posixpath.splitext(posixpath.basename(i.filename))[0]].add(posixpath.dirname(i.filename))
    emit("stem_in_multiple_dirs", n=sum(1 for v in owner.values() if len(v) > 1))


@guarded
def stage_images(z, infos):
    from PIL import Image
    img_ext = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
    imgs = [i for i in infos if not i.is_dir() and classify(i.filename) in img_ext]
    rnd = random.Random(20261110)
    sample = rnd.sample(imgs, min(300, len(imgs)))
    dims = collections.Counter(); modes = collections.Counter(); fmts = collections.Counter(); sizes = []
    bad = 0
    gray_like = 0
    for i in sample:
        try:
            with z.open(i) as f:
                im = Image.open(io.BytesIO(f.read()))
                dims[im.size] += 1; modes[im.mode] += 1; fmts[im.format] += 1; sizes.append(i.file_size)
                if im.mode == "RGB":
                    t = im.resize((64, 64)); px = list(t.getdata())
                    # near-grayscale heuristic (thermal/IR often has R=G=B)
                    if sum(1 for r, g, b in px if abs(r - g) < 4 and abs(g - b) < 4) / len(px) > 0.98:
                        gray_like += 1
        except Exception:
            bad += 1
    emit("images_sample", n=len(sample), undecodable=bad, dims_top=[[list(k), v] for k, v in dims.most_common(12)], n_distinct_dims=len(dims),
         modes=dict(modes), formats=dict(fmts), near_grayscale_in_sample=gray_like,
         mean_file_kb=round(sum(sizes) / len(sizes) / 1024, 1) if sizes else None, total_images=len(imgs))
    # save 6 sample names for later targeted viewing
    emit("sample_names", names=[i.filename for i in sample[:6]])


@guarded
def stage_crc(z):
    t = time.time()
    bad = z.testzip()
    emit("crc", testzip_first_bad=bad, ok=bad is None, seconds=round(time.time() - t, 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    emit("start", python=sys.version.split()[0], cpus=os.cpu_count())
    stage_size()
    th = threading.Thread(target=sha_job, daemon=True)
    th.start()
    z = zipfile.ZipFile(ZIP)
    infos = stage_structure(z)
    if infos:
        stage_text(z, infos)
        stage_yolo(z, infos)
        stage_images(z, infos)
    stage_crc(z)
    th.join(timeout=1200)
    emit("done", sha256=SHA.get("v"))
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(REPORT, f, indent=1, default=str)


if __name__ == "__main__":
    main()
