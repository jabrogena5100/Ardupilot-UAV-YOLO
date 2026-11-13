import collections, json, os, posixpath, re, time, zipfile
import xml.etree.ElementTree as ET

Z = os.environ.get("FZ", "/workspace/datasets/FASDD_UAV.zip")
OUT = os.environ.get("FOUT", "/workspace/runs/fasdd_validate")
T0 = time.time()
R = {}


def emit(k, **kw):
    R[k] = kw
    print(json.dumps({"stage": k, "t": round(time.time() - T0, 1), **kw}, default=str), flush=True)


z = zipfile.ZipFile(Z)
names = {i.filename for i in z.infolist()}
stem = lambda p: posixpath.splitext(posixpath.basename(p))[0]
lists = {s: [stem(l.strip()) for l in z.read(f"annotations/YOLO_UAV/{s}.txt").decode().splitlines() if l.strip()] for s in ("train", "val", "test")}
sets = {s: set(v) for s, v in lists.items()}
allimg = {stem(n) for n in names if n.startswith("images/") and n.endswith(".jpg")}
emit("yolo_lists", n={s: len(v) for s, v in lists.items()}, dup_within={s: len(lists[s]) - len(sets[s]) for s in lists},
     overlaps={"train_val": len(sets["train"] & sets["val"]), "train_test": len(sets["train"] & sets["test"]), "val_test": len(sets["val"] & sets["test"])},
     union_equals_all_images=(sets["train"] | sets["val"] | sets["test"]) == allimg, union=len(sets["train"] | sets["val"] | sets["test"]), all_images=len(allimg))

# COCO
coco = {}
for s in ("train", "val", "test"):
    j = json.loads(z.read(f"annotations/COCO_UAV/Annotations/{s}.json"))
    coco[s] = j
cs = {}
for s, j in coco.items():
    fs = [stem(i["file_name"]) for i in j["images"]]
    cs[s] = set(fs)
    emit("coco_" + s, n_images=len(fs), unique=len(set(fs)), equals_yolo_list=set(fs) == sets[s], only_in_coco=sorted(set(fs) - sets[s])[:5],
         only_in_yolo=sorted(sets[s] - set(fs))[:5], n_only_coco=len(set(fs) - sets[s]), n_only_yolo=len(sets[s] - set(fs)),
         categories=j["categories"], n_ann=len(j["annotations"]), ann_cat_hist=dict(collections.Counter(a["category_id"] for a in j["annotations"])),
         images_with_no_ann=len({i["id"] for i in j["images"]} - {a["image_id"] for a in j["annotations"]}))

# VOC: no split lists exist; parse all xml
voc = {}
for n in names:
    if n.startswith("annotations/VOC_UAV/Annotations/") and n.endswith(".xml"):
        r = ET.fromstring(z.read(n))
        sz = r.find("size")
        objs = []
        for o in r.findall("object"):
            b = o.find("bndbox")
            objs.append((o.findtext("name"), [float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]))
        voc[stem(n)] = (int(sz.findtext("width")), int(sz.findtext("height")), objs)
vnames = collections.Counter(o[0] for v in voc.values() for o in v[2])
emit("voc", n_xml=len(voc), equals_all_images=set(voc) == allimg, object_names=dict(vnames), has_split_list=any("ImageSets" in n for n in names))

# YOLO labels, all images
lab = {}
bad = 0
for s in allimg:
    rows = []
    for ln in z.read(f"annotations/YOLO_UAV/labels/{s}.txt").decode().splitlines():
        p = ln.split()
        if len(p) == 5:
            c = int(p[0]); x, y, w, h = map(float, p[1:])
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                bad += 1
            rows.append((c, x, y, w, h))
    lab[s] = rows
emit("yolo_range", out_of_range_boxes=bad)

# cross-format agreement per image (COCO has pixel size; tolerance 2 px)
def agree(split):
    j = coco[split]
    imgs = {i["id"]: i for i in j["images"]}
    anns = collections.defaultdict(list)
    for a in j["annotations"]:
        anns[a["image_id"]].append(a)
    cnt_mis = box_mis = size_mis = voc_cnt_mis = voc_box_mis = voc_size_mis = cls_name_bad = 0
    for iid, im in imgs.items():
        s = stem(im["file_name"]); W, H = im["width"], im["height"]
        y = sorted((c, round((x - w / 2) * W), round((yy - h / 2) * H), round(w * W), round(h * H)) for c, x, yy, w, h in lab[s])
        c = sorted((a["category_id"], round(a["bbox"][0]), round(a["bbox"][1]), round(a["bbox"][2]), round(a["bbox"][3])) for a in anns[iid])
        if len(y) != len(c):
            cnt_mis += 1
        elif any(yy[0] != cc[0] or any(abs(p - q) > 2 for p, q in zip(yy[1:], cc[1:])) for yy, cc in zip(y, c)):
            box_mis += 1
        v = voc.get(s)
        if v:
            if (v[0], v[1]) != (W, H):
                voc_size_mis += 1
            vc = sorted((0 if n == "fire" else 1, round(b[0]), round(b[1]), round(b[2] - b[0]), round(b[3] - b[1])) for n, b in v[2])
            if len(vc) != len(c):
                voc_cnt_mis += 1
            elif any(a[0] != b[0] or any(abs(p - q) > 2 for p, q in zip(a[1:], b[1:])) for a, b in zip(vc, c)):
                voc_box_mis += 1
    emit("agree_" + split, images=len(imgs), yolo_vs_coco_count_mismatch=cnt_mis, yolo_vs_coco_box_mismatch_gt2px=box_mis,
         voc_vs_coco_size_mismatch=voc_size_mis, voc_vs_coco_count_mismatch=voc_cnt_mis, voc_vs_coco_box_mismatch_gt2px=voc_box_mis)

for s in ("test", "train", "val"):
    agree(s)

# test split composition + smoke-only stats
def q(v):
    if not v:
        return None
    v = sorted(v); n = len(v)
    return {k: round(v[min(n - 1, int(f * (n - 1)))], 4) for k, f in (("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95))}


def edge(b, m=0.005):
    _, x, y, w, h = b
    return x - w / 2 <= m or x + w / 2 >= 1 - m or y - h / 2 <= m or y + h / 2 >= 1 - m


def cat(rows):
    cl = {r[0] for r in rows}
    return "both" if cl == {0, 1} else "smoke_only" if cl == {1} else "fire_only" if cl == {0} else "empty"


def prefix(s):
    return re.match(r"[A-Za-z]+", s).group(0)


for sp in ("test", "train", "val"):
    cats = collections.Counter(); catpre = collections.defaultdict(collections.Counter)
    sm = []; fi = []
    per_img_smoke = []
    for s in sets[sp]:
        rows = lab[s]; c = cat(rows); cats[c] += 1; catpre[prefix(s)][c] += 1
        sb = [r for r in rows if r[0] == 1]; fb = [r for r in rows if r[0] == 0]
        sm += sb; fi += fb
        if sb:
            per_img_smoke.append(len(sb))
    emit("composition_" + sp, images=len(sets[sp]), categories=dict(cats), smoke_positive_images=cats["both"] + cats["smoke_only"],
         by_prefix={k: dict(v) for k, v in catpre.items()},
         smoke_boxes=len(sm), fire_boxes=len(fi),
         smoke_only_stats={"edge_touch_frac_0.005": round(sum(edge(b) for b in sm) / max(len(sm), 1), 3), "edge_touch_frac_0.02": round(sum(edge(b, .02) for b in sm) / max(len(sm), 1), 3),
                           "w": q([b[3] for b in sm]), "h": q([b[4] for b in sm]), "area": q([b[3] * b[4] for b in sm]),
                           "boxes_per_smoke_img": q(per_img_smoke), "area_lt_0.001_frac": round(sum(b[3] * b[4] < 0.001 for b in sm) / max(len(sm), 1), 3)},
         fire_only_stats={"edge_touch_frac_0.005": round(sum(edge(b) for b in fi) / max(len(fi), 1), 3), "area": q([b[3] * b[4] for b in fi])} if fi else None,
         pooled_edge_touch_frac_m005=round(sum(edge(b) for b in sm + fi) / max(len(sm + fi), 1), 3))

pos = R["composition_test"]["smoke_positive_images"]
emit("sufficiency", test_smoke_positive_images=pos, gate_G4_min_500=pos >= 500, note="dHash cluster count (>=20) not computed here; overlap/clustering deferred")
os.makedirs(OUT, exist_ok=True)
json.dump(R, open(os.path.join(OUT, "report.json"), "w"), indent=1, default=str)
emit("done", report=OUT + "/report.json")
