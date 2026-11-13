import sys, os, json, zipfile, zlib, io, hashlib, time, threading, urllib.request, urllib.error, random
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rangefile import RangeFile
import numpy as np
from PIL import Image
URL = "https://zenodo.org/records/13732947/files/Multiclass.zip?download=1"
OUT = "/home/joshua/src/Ardupilot-UAV-YOLO/local_data/external_eval/fireman/work/frames_every20.jsonl"
STEP = int(os.environ.get("STEP", "20"))
rf = RangeFile(URL); z = zipfile.ZipFile(rf); final = rf.final
infos = {i.filename: i for i in z.infolist() if i.filename.startswith("Multiclass/Unimodal/rgb/") and i.filename.endswith(".jpg")}
print("rgb jpgs", len(infos), "central dir MB", round(rf.transferred / 1e6, 1), flush=True)
def fnum(n): return int(n.rsplit("_", 1)[1].split(".")[0])
PER = int(os.environ.get("PER", "30"))
byv = {}
for n in infos: byv.setdefault(n.split("Unimodal/rgb/")[1].rsplit("/images/", 1)[0], []).append(n)
sel = {}
for v, L in byv.items():
    L.sort(key=fnum); k = min(PER, len(L)); idx = sorted({int(round(i * (len(L) - 1) / max(k - 1, 1))) for i in range(k)})
    sel[v] = [L[i] for i in idx]
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["name"])
        except Exception: pass
targets = []
for r in range(PER + 1):                      # round-robin across videos so partial runs cover every video
    for v in sorted(sel):
        if r < len(sel[v]) and sel[v][r] not in done: targets.append(sel[v][r])
print("targets", len(targets), "already", len(done), flush=True)
def bits2int(b): return int.from_bytes(np.packbits(np.asarray(b, dtype=np.uint8)).tobytes(), "big")
def fetch(info):
    a = info.header_offset; n = 30 + 600 + info.compress_size
    for attempt in range(12):
        try:
            req = urllib.request.Request(final, headers={"Range": f"bytes={a}-{a + n - 1}", "User-Agent": "range-sample/1"})
            with urllib.request.urlopen(req, timeout=120) as r: h = r.read()
            nl, ml = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
            data = h[30 + nl + ml: 30 + nl + ml + info.compress_size]
            return zlib.decompress(data, -15) if info.compress_type == 8 else data
        except urllib.error.HTTPError as e:
            err = e
            ra = e.headers.get("Retry-After") if e.code == 429 else None
            time.sleep(min(120, int(ra)) if ra and ra.isdigit() else (20 + attempt * 20 if e.code == 429 else 1 + attempt * 2))
        except Exception as e:
            time.sleep(1 + attempt * 2); err = e
    raise err
def work(name):
    raw = fetch(infos[name])
    im = Image.open(io.BytesIO(raw)); W, H = im.size; g = im.convert("L")
    o = {"name": name, "size": [W, H], "md5": hashlib.md5(raw).hexdigest(), "bytes": len(raw)}
    for nm, f in (("lanczos", Image.LANCZOS), ("bilinear", Image.BILINEAR), ("box", Image.BOX)):
        a = np.asarray(g.resize((9, 8), f), dtype=np.int16)
        o[nm] = "%016x" % bits2int((a[:, 1:] > a[:, :-1]).flatten())
    o["thumb"] = g.resize((32, 32), Image.BILINEAR).tobytes().hex()
    rgb = np.asarray(im.convert("RGB").resize((32, 32), Image.BILINEAR), dtype=np.float32)
    o["mean_rgb"] = [round(float(x), 1) for x in rgb.reshape(-1, 3).mean(0)]
    return o
lock = threading.Lock(); cnt = [0]; t0 = time.time()
def run(name):
    try: o = work(name)
    except Exception as e:
        print("FAILED", name, repr(e)[:80], flush=True); time.sleep(30); return
    with lock:
        open(OUT, "a").write(json.dumps(o) + "\n"); cnt[0] += 1
        if cnt[0] % 100 == 0: print(cnt[0], "of", len(targets), round(time.time() - t0), "s", flush=True)
with ThreadPoolExecutor(int(os.environ.get('THREADS','3'))) as ex: list(ex.map(run, targets))
print("DONE", cnt[0], round(time.time() - t0), "s", flush=True)
