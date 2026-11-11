import sys, os, io, json, csv, glob, zlib, zipfile, hashlib, time, collections, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rangefile import RangeFile
from PIL import Image, ImageDraw, ImageFont
B = "/home/joshua/src/Ardupilot-UAV-YOLO/local_data/external_eval/fireman/"
OUT = B + "screening_v1"
SEED = 20261109
EXCL = {x for x in os.environ.get("EXCL", "").split(",") if x}
URL = "https://zenodo.org/records/13732947/files/Multiclass.zip?download=1"
def rk(salt, name): return hashlib.sha256(f"{SEED}:{salt}:{name}".encode()).hexdigest()
def draw(names, n, salt): return sorted(sorted(names, key=lambda x: rk(salt, x))[:n])
frames = {}   # (video) -> list of dict
for p in sorted(glob.glob(B + "annotations/*__annotations.xml")):
    split, vid, _ = os.path.basename(p).split("__"); v = f"{split}/{vid}"
    r = ET.parse(p).getroot(); ims = sorted(r.findall("image"), key=lambda e: int(e.get("id")))
    rows = []; coords = []
    for im in ims:
        bs = [b for b in im.findall("box") if b.get("label") == "smoke_region"]
        fb = [b for b in im.findall("box") if b.get("label") == "fire_region"]
        W, H = int(im.get("width")), int(im.get("height"))
        rows.append({"video": v, "id": int(im.get("id")), "name": im.get("name"), "W": W, "H": H,
                     "smoke": [[float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")] for b in bs], "fire": [[float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")] for b in fb]})
        coords.append(rows[-1]["smoke"][0] if len(bs) == 1 else [np.nan] * 4)
    c = np.array(coords); d2 = np.abs(c[2:] - 2 * c[1:-1] + c[:-2]); ok = ~np.isnan(d2).any(axis=1)
    kf = list(np.where((d2.max(axis=1) >= 0.02) & ok)[0] + 1)
    for i, rw in enumerate(rows): rw["dist_kf"] = int(min(abs(i - k) for k in kf)) if kf else None
    frames[v] = rows
elig = [v for v in frames if v not in EXCL]
pos_videos = [v for v in elig if any(r["smoke"] for r in frames[v])]
sel = []
for v in pos_videos:
    have = [str(r["id"]) for r in frames[v] if r["smoke"]]
    for i in draw(have, 2, f"screenP:{v}"): sel.append((v, int(i), "P"))
neg_pool = {v: [str(r["id"]) for r in frames[v] if not r["smoke"]] for v in elig}
neg_pool = {v: x for v, x in neg_pool.items() if x}
n_target = 100 - len(sel); got = []; capped = {v: draw(x, min(6, len(x)), f"screenN:{v}") for v, x in neg_pool.items()}
order = sorted([(rk("negorder", f"{v}:{i}"), v, i) for v, L in capped.items() for i in L])
for _, v, i in order[:n_target]: got.append((v, int(i), "N"))
sel += got
spot = set(draw([f"{v}|{i}" for v, i, k in sel if k == "P"], 19, "spot") + draw([f"{v}|{i}" for v, i, k in sel if k == "N"], 11, "spot"))
print("eligible videos", len(elig), "smoke videos", len(pos_videos), "sample", len(sel), "P", sum(1 for s in sel if s[2] == "P"), "N", sum(1 for s in sel if s[2] == "N"), "spot", len(spot), flush=True)
rf = RangeFile(URL); z = zipfile.ZipFile(rf); final = rf.final
infos = {i.filename: i for i in z.infolist() if i.filename.startswith("Multiclass/Unimodal/rgb/") and i.filename.endswith(".jpg")}
def fetch(info):
    a = info.header_offset; n = 30 + 600 + info.compress_size
    for att in range(12):
        try:
            req = urllib.request.Request(final, headers={"Range": f"bytes={a}-{a + n - 1}", "User-Agent": "range-sample/1"})
            with urllib.request.urlopen(req, timeout=120) as r: h = r.read()
            nl, ml = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
            d = h[30 + nl + ml: 30 + nl + ml + info.compress_size]
            return zlib.decompress(d, -15) if info.compress_type == 8 else d
        except urllib.error.HTTPError as e:
            err = e; time.sleep(20 + att * 20 if e.code == 429 else 1 + att * 2)
        except Exception as e:
            err = e; time.sleep(1 + att * 2)
    raise err
def path(v, rw): return f"Multiclass/Unimodal/rgb/{v}/images/frame_{rw['id']:06d}.jpg"
lookup = {(v, rw["id"]): rw for v, L in frames.items() for rw in L}
with ThreadPoolExecutor(3) as ex: raws = list(ex.map(lambda s: fetch(infos[path(s[0], lookup[(s[0], s[1])])]), sel))
os.makedirs(OUT)
try: font = ImageFont.load_default(size=15)
except TypeError: font = ImageFont.load_default()
order = sorted(range(len(sel)), key=lambda i: (sel[i][2] != "P", sel[i][0], sel[i][1]))
W2, H2, LB = 560, 373, 36; man = []
for sh in range((len(order) + 11) // 12):
    ch = order[sh * 12:(sh + 1) * 12]
    sheet = Image.new("RGB", (3 * W2, 4 * (H2 + LB)), (30, 30, 30)); d = ImageDraw.Draw(sheet)
    for k, i in enumerate(ch):
        v, fid, kind = sel[i]; rw = lookup[(v, fid)]; no = order.index(i) + 1
        im = Image.open(io.BytesIO(raws[i])).convert("RGB"); w0, h0 = im.size
        sc = min((W2 - 4) / w0, H2 / h0); tw, th = int(w0 * sc), int(h0 * sc); ti = im.resize((tw, th), Image.LANCZOS); dr = ImageDraw.Draw(ti)
        for b in rw["smoke"]: dr.rectangle([b[0] / rw["W"] * tw, b[1] / rw["H"] * th, b[2] / rw["W"] * tw, b[3] / rw["H"] * th], outline=(255, 255, 0), width=2)
        for b in rw["fire"]: dr.rectangle([b[0] / rw["W"] * tw, b[1] / rw["H"] * th, b[2] / rw["W"] * tw, b[3] / rw["H"] * th], outline=(255, 60, 60), width=2)
        x0, y0 = (k % 3) * W2, (k // 3) * (H2 + LB); sheet.paste(ti, (x0 + 2, y0 + LB))
        sp = f"{v}|{fid}" in spot
        d.text((x0 + 4, y0 + 2), f"#{no} {v} f{fid}", fill=(255, 255, 255), font=font)
        d.text((x0 + 4, y0 + 18), f"{'SMOKE-BOX' if kind == 'P' else 'NO-SMOKE-BOX'}{' SPOT' if sp else ''} {w0}x{h0} kf-dist={rw['dist_kf']}", fill=(160, 220, 255), font=font)
        man.append({"no": no, "sheet": f"sheet_{sh + 1:02d}.jpg", "video": v, "frame_id": fid, "stratum": kind, "spot_check": int(sp), "width": w0, "height": h0,
                    "n_smoke_boxes": len(rw["smoke"]), "n_fire_boxes": len(rw["fire"]), "dist_to_keyframe": rw["dist_kf"]})
    sheet.save(f"{OUT}/sheet_{sh + 1:02d}.jpg", quality=88)
man.sort(key=lambda m: m["no"])
with open(f"{OUT}/screening_manifest.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(list(man[0]) + ["screener_smoke_visible", "screener_box_verdict", "owner_verdict", "notes"])
    for m in man: w.writerow([m[k] for k in man[0]] + ["", "", "", ""])
json.dump({"seed": SEED, "excluded_videos": sorted(EXCL), "n": len(man), "spot": len(spot)}, open(f"{OUT}/sample_info.json", "w"))
print("done", len(man))
