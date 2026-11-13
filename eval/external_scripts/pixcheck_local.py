import sys, os, io, json, tarfile, zipfile, zlib, time, collections, urllib.request, urllib.error
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rangefile import RangeFile
B = "/home/joshua/src/Ardupilot-UAV-YOLO/local_data/external_eval/"
REF = B + "refthumbs_v1/"; OUT = B + "fireman_pixcheck_v1/"
fr = {}
for l in open(B + "fireman/work/frames_every20.jsonl"):
    try: o = json.loads(l)
    except Exception: continue
    fr[o["name"]] = o
names = sorted(fr)
short = lambda n: n.split("Unimodal/rgb/")[1].replace("/images/", "/").replace(".jpg", "")
Q = np.array([np.frombuffer(bytes.fromhex(fr[n]["thumb"]), np.uint8) for n in names], np.float32)
R = np.load(REF + "refs32.npy").astype(np.float32); idx = json.load(open(REF + "refs_index.json"))
def zn(a):
    a = a - a.mean(1, keepdims=True); n = np.linalg.norm(a, axis=1, keepdims=True); return a / np.maximum(n, 1e-6), n[:, 0]
Qn, qnorm = zn(Q); Rn, rnorm = zn(R)
S = Qn @ Rn.T
isb = np.array([x["ds"] == "boreal" for x in idx])
qstd = Q.std(1)
rows = []
for i, n in enumerate(names):
    r = {"frame": short(n), "video": short(n).rsplit("/", 1)[0], "query_std": float(qstd[i])}
    for dsn, mask in (("boreal", isb), ("fasdd", ~isb)):
        j = np.where(mask)[0][int(np.argmax(S[i, mask]))]
        r[dsn] = {"ref": idx[j]["key"], "ncc32": round(float(S[i, j]), 4), "ref_i": int(j), "ref_std": float(R[j].std())}
    rows.append(r)
def verdict(x, q): 
    if q < 8 or x["ref_std"] < 8: return "low_texture"
    return "duplicate" if x["ncc32"] >= 0.97 else "near_similar" if x["ncc32"] >= 0.90 else "no_match"
summ = {}
flagged = set(json.load(open(B + "fireman/work/pixcheck_flagged.json"))) if os.path.exists(B + "fireman/work/pixcheck_flagged.json") else set()
for dsn in ("boreal", "fasdd"):
    v = collections.Counter(verdict(r[dsn], r["query_std"]) for r in rows)
    vf = collections.Counter(verdict(r[dsn], r["query_std"]) for r in rows if r["frame"] in flagged)
    nc = np.array([r[dsn]["ncc32"] for r in rows])
    summ[dsn] = {"all_frames": dict(v), "hash_flagged_frames": dict(vf), "max_ncc32": float(nc.max()), "p99": float(np.percentile(nc, 99)), "median": float(np.median(nc)),
                 "top10": [[r["frame"], r[dsn]["ref"], r[dsn]["ncc32"]] for r in sorted(rows, key=lambda r: -r[dsn]["ncc32"])[:10]]}
os.makedirs(OUT, exist_ok=False)
json.dump({"summary": summ, "rows": rows}, open(OUT + "pixcheck_local.json", "w"))
print(json.dumps(summ, indent=1)[:3500])
# visual pair sheets: top 12 per dataset, FireMan frame fetched from Zenodo (rate-limit tolerant), reference = 160x120 thumb from the volume export
tf = tarfile.open(REF + "refs_jpg.tar"); members = {m.name: m for m in tf.getmembers()}
def refjpg(j): return Image.open(io.BytesIO(tf.extractfile(members[f"{j:05d}.jpg"]).read())).convert("RGB")
rf = RangeFile("https://zenodo.org/records/13732947/files/Multiclass.zip?download=1"); z = zipfile.ZipFile(rf); final = rf.final
infos = {i.filename: i for i in z.infolist() if i.filename.startswith("Multiclass/Unimodal/rgb/")}
def fetch(n):
    info = infos[n]; a = info.header_offset
    for att in range(10):
        try:
            h = urllib.request.urlopen(urllib.request.Request(final, headers={"Range": f"bytes={a}-{a + 630 + info.compress_size - 1}"}), timeout=120).read()
            nl, ml = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little"); d = h[30 + nl + ml:30 + nl + ml + info.compress_size]
            return zlib.decompress(d, -15) if info.compress_type == 8 else d
        except urllib.error.HTTPError as e: time.sleep(20 * (att + 1) if e.code == 429 else 2)
        except Exception: time.sleep(2)
    raise RuntimeError(n)
try: font = ImageFont.load_default(size=13)
except TypeError: font = ImageFont.load_default()
pairs = []
for dsn in ("boreal", "fasdd"):
    for r in sorted(rows, key=lambda r: -r[dsn]["ncc32"])[:12]: pairs.append((dsn, r))
W, H, LAB = 320, 180, 34; sheets = []
for sh in range((len(pairs) + 7) // 8):
    ch = pairs[sh * 8:(sh + 1) * 8]; sheet = Image.new("RGB", (2 * (2 * W + 16), 4 * (H + LAB)), (30, 30, 30)); d = ImageDraw.Draw(sheet)
    for k, (dsn, r) in enumerate(ch):
        x0, y0 = (k % 2) * (2 * W + 16), (k // 2) * (H + LAB)
        fm = Image.open(io.BytesIO(fetch(f"Multiclass/Unimodal/rgb/{r['frame'].replace('/', '/images/', 2)[:0]}" if False else "Multiclass/Unimodal/rgb/" + r["video"] + "/images/" + r["frame"].rsplit("/", 1)[1] + ".jpg"))).convert("RGB")
        rj = refjpg(r[dsn]["ref_i"])
        for m, im in enumerate((fm, rj)):
            sc = min(W / im.size[0], H / im.size[1]); im = im.resize((int(im.size[0] * sc), int(im.size[1] * sc)), Image.LANCZOS); sheet.paste(im, (x0 + m * W, y0 + LAB))
        d.text((x0 + 3, y0 + 1), f"FireMan {r['frame']} | {dsn} {r[dsn]['ref'].split('__')[-1][:34]}", fill=(255, 255, 255), font=font)
        d.text((x0 + 3, y0 + 17), f"NCC 32px {r[dsn]['ncc32']}  query std {r['query_std']:.0f} ref std {r[dsn]['ref_std']:.0f}", fill=(160, 220, 255), font=font)
    fn = f"pairs_{sh + 1:02d}.jpg"; sheet.save(OUT + fn, quality=88); sheets.append(fn)
print("sheets", sheets)
