import collections, csv, gzip, hashlib, hmac, http.server, io, json, os, random, shutil, threading, time, urllib.error, urllib.request, zipfile, zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WS = os.environ.get("WS", "/workspace")
RUNS = os.path.join(WS, "runs", "fireman_pixcheck"); OUT = os.path.join(WS, "external_eval", "fireman_pixcheck_v1")
DS = os.path.join(WS, "external_eval", "fasdd_uav_test_v1", "images")
URL = "https://zenodo.org/records/13732947/files/Multiclass.zip?download=1"
INP = json.loads(os.environ["PIX_INPUT"]) if "PIX_INPUT" in os.environ else json.load(open(os.environ["PIX_FILE"]))
T0 = time.time(); R = {}
def emit(k, **kw):
    R[k] = kw; print(json.dumps({"stage": k, "t": round(time.time() - T0, 1), **kw}, default=str), flush=True)

class RangeFile:
    def __init__(s, url, block=4 << 20):
        s.url, s.block, s.pos, s.cache = url, block, 0, collections.OrderedDict()
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "pix/1"}), timeout=60)
        s.size = int(r.headers["Content-Range"].split("/")[-1]); s.final = r.geturl(); r.read()
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, o, w=0):
        s.pos = o if w == 0 else s.pos + o if w == 1 else s.size + o; return s.pos
    def _b(s, i):
        if i in s.cache: return s.cache[i]
        a, b = i * s.block, min((i + 1) * s.block, s.size) - 1
        for att in range(8):
            try:
                d = urllib.request.urlopen(urllib.request.Request(s.final, headers={"Range": f"bytes={a}-{b}", "User-Agent": "pix/1"}), timeout=120).read(); break
            except urllib.error.HTTPError as e:
                if e.code != 429 or att == 7: raise
                time.sleep(20 * (att + 1))
        s.cache[i] = d
        if len(s.cache) > 8: s.cache.popitem(last=False)
        return d
    def read(s, n=-1):
        n = s.size - s.pos if n < 0 else n; out = []
        while n > 0 and s.pos < s.size:
            i, o = divmod(s.pos, s.block); d = s._b(i)[o:o + n]
            if not d: break
            out.append(d); s.pos += len(d); n -= len(d)
        return b"".join(out)
    def readinto(s, b):
        d = s.read(len(b)); b[:len(d)] = d; return len(d)

def norm(a):
    a = np.asarray(a, np.float64).ravel(); a = a - a.mean(); n = np.linalg.norm(a)
    return a / n if n > 1e-6 else None

def thumbs(path):
    im = Image.open(path); g = im.convert("L")
    return np.asarray(g.resize((32, 32), Image.BILINEAR), np.uint8).tobytes(), np.asarray(g.resize((64, 64), Image.BILINEAR), np.uint8).tobytes(), list(im.size)

def main():
    os.makedirs(RUNS, exist_ok=True)
    try:
        fd = os.open(os.path.join(RUNS, "STARTED"), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, b"1"); os.close(fd)
    except FileExistsError:
        print("GUARD: already started; exiting", flush=True); return
    if os.path.exists(OUT): emit("abort", reason="output exists"); return
    names = INP["flagged"] + INP["controls"]
    emit("input", flagged=len(INP["flagged"]), controls=len(INP["controls"]))
    # 1) FireMan frames from Zenodo by range requests
    rf = RangeFile(URL); z = zipfile.ZipFile(rf)
    infos = {i.filename: i for i in z.infolist() if i.filename.startswith("Multiclass/Unimodal/rgb/") and i.filename.endswith(".jpg")}
    def full(n): v, f = n.rsplit("/", 1); return f"Multiclass/Unimodal/rgb/{v}/images/{f}.jpg"
    def fetch(n):
        info = infos[full(n)]; a = info.header_offset
        for att in range(10):
            try:
                h = urllib.request.urlopen(urllib.request.Request(rf.final, headers={"Range": f"bytes={a}-{a + 630 + info.compress_size - 1}", "User-Agent": "pix/1"}), timeout=120).read()
                nl, ml = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little"); d = h[30 + nl + ml:30 + nl + ml + info.compress_size]
                return zlib.decompress(d, -15) if info.compress_type == 8 else d
            except urllib.error.HTTPError as e:
                time.sleep(20 * (att + 1) if e.code == 429 else 2)
            except Exception:
                time.sleep(2)
        raise RuntimeError("fetch failed " + n)
    os.makedirs(os.path.join(RUNS, "fm"), exist_ok=True)
    def get(n):
        p = os.path.join(RUNS, "fm", n.replace("/", "__") + ".jpg"); open(p, "wb").write(fetch(n)); return p
    with ThreadPoolExecutor(3) as ex: fpaths = dict(zip(names, ex.map(get, names)))
    emit("fireman_fetched", n=len(fpaths))
    # 2) reference thumbs: all Boreal + all FASDD test images, identical function
    idx = {}
    for dp, _, fs in os.walk(os.path.join(WS, "boreal")):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and (f not in idx or "boreal_yolo" in dp): idx[f] = os.path.join(dp, f)
    scan = json.load(gzip.open(os.path.join(WS, "integrity", "boreal_scan.json.gz")))
    bkeys = [f"{sp}/{nm}" for sp, d in scan.items() for nm in d if nm in idx]
    fkeys = sorted(f[:-4] for f in os.listdir(DS))
    refs = [("boreal", k, idx[k.split("/", 1)[1]]) for k in bkeys] + [("fasdd", k, os.path.join(DS, k + ".jpg")) for k in fkeys]
    emit("refs", boreal=len(bkeys), fasdd=len(fkeys))
    with ProcessPoolExecutor(4) as ex:
        rt = list(ex.map(thumbs, [r[2] for r in refs], chunksize=16)); ft = list(ex.map(thumbs, [fpaths[n] for n in names]))
    emit("thumbs", refs=len(rt), fireman=len(ft))
    R32 = np.array([norm(np.frombuffer(t[0], np.uint8)) if norm(np.frombuffer(t[0], np.uint8)) is not None else np.zeros(1024) for t in rt])
    rows = []
    for n, t in zip(names, ft):
        q = norm(np.frombuffer(t[0], np.uint8)); q = q if q is not None else np.zeros(1024)
        s = R32 @ q; order = np.argsort(-s)[:5]; res = []
        for j in order:
            a, b = norm(np.frombuffer(t[1], np.uint8)), norm(np.frombuffer(rt[j][1], np.uint8))
            n64 = float(a @ b) if a is not None and b is not None else None
            res.append({"ds": refs[j][0], "ref": refs[j][1], "ncc32": round(float(s[j]), 4), "ncc64": None if n64 is None else round(n64, 4), "ref_size": rt[j][2]})
        best = max(res, key=lambda r: (r["ncc64"] if r["ncc64"] is not None else -1))
        sb = max((r for r in res if r["ds"] == "boreal"), key=lambda r: r["ncc32"], default=None); sf = max((r for r in res if r["ds"] == "fasdd"), key=lambda r: r["ncc32"], default=None)
        rows.append({"frame": n, "is_control": n in INP["controls"], "size": t[2], "top5": res, "best": best,
                     "verdict": "pixel_duplicate" if (best["ncc64"] or 0) >= 0.97 else "near_similar" if (best["ncc64"] or 0) >= 0.90 else "no_match"})
    # best over ALL refs per dataset (top-5 across mixed datasets may hide one dataset)
    for r, t in zip(rows, ft):
        q = norm(np.frombuffer(t[0], np.uint8)); q = q if q is not None else np.zeros(1024); s = R32 @ q
        for dsn in ("boreal", "fasdd"):
            m = [i for i, x in enumerate(refs) if x[0] == dsn]; j = m[int(np.argmax(s[m]))]
            a, b = norm(np.frombuffer(t[1], np.uint8)), norm(np.frombuffer(rt[j][1], np.uint8))
            r["best_" + dsn] = {"ref": refs[j][1], "ncc32": round(float(s[j]), 4), "ncc64": None if a is None or b is None else round(float(a @ b), 4)}
    vc = collections.Counter((r["is_control"], r["verdict"]) for r in rows)
    emit("verdicts", counts={f"{'control' if k[0] else 'flagged'}:{k[1]}": v for k, v in vc.items()},
         max_ncc64_flagged={d: max((r["best_" + d]["ncc64"] or 0) for r in rows if not r["is_control"]) for d in ("boreal", "fasdd")},
         max_ncc64_controls={d: max((r["best_" + d]["ncc64"] or 0) for r in rows if r["is_control"]) for d in ("boreal", "fasdd")},
         median_best_ncc64_flagged={d: float(np.median([(r["best_" + d]["ncc64"] or 0) for r in rows if not r["is_control"]])) for d in ("boreal", "fasdd")},
         dup_list=[{"frame": r["frame"], **r["best"]} for r in rows if r["verdict"] == "pixel_duplicate"], near_list=[{"frame": r["frame"], **r["best"]} for r in rows if r["verdict"] == "near_similar"][:40])
    # 3) pair sheets: highest-similarity pairs per dataset + the hash-nearest ones are the flagged ones
    os.makedirs(OUT)
    try: font = ImageFont.load_default(size=14)
    except TypeError: font = ImageFont.load_default()
    pairs = []
    for dsn in ("boreal", "fasdd"):
        top = sorted(rows, key=lambda r: -(r["best_" + dsn]["ncc64"] or 0))[:16]
        pairs += [(dsn, r["frame"], r["best_" + dsn]) for r in top]
    ref_path = {(x[0], x[1]): x[2] for x in refs}
    W, HH, LAB = 400, 270, 36; files = []
    for sh in range((len(pairs) + 9) // 10):
        ch = pairs[sh * 10:(sh + 1) * 10]
        sheet = Image.new("RGB", (2 * (2 * W + 20), 5 * (HH + LAB)), (30, 30, 30)); d = ImageDraw.Draw(sheet)
        for k, (dsn, fn, b) in enumerate(ch):
            x0, y0 = (k % 2) * (2 * W + 20), (k // 2) * (HH + LAB)
            for m, pth in enumerate((fpaths[fn], ref_path[(dsn, b["ref"])])):
                im = Image.open(pth).convert("RGB"); sc = min(W / im.size[0], HH / im.size[1]); im = im.resize((int(im.size[0] * sc), int(im.size[1] * sc)), Image.LANCZOS); sheet.paste(im, (x0 + m * W, y0 + LAB))
            d.text((x0 + 4, y0 + 2), f"FireMan {fn}  |  {dsn} {b['ref'].split('__')[-1][:40]}", fill=(255, 255, 255), font=font)
            d.text((x0 + 4, y0 + 18), f"NCC 32px {b['ncc32']}  NCC 64px {b['ncc64']}", fill=(160, 220, 255), font=font)
        fn_ = f"pairs_{sh + 1:02d}.jpg"; sheet.save(os.path.join(OUT, fn_), quality=88); files.append(fn_)
    emit("pair_sheets", n_pairs=len(pairs), n_sheets=len(files), layout="left FireMan frame, right most-similar reference image (64px grey NCC)")
    json.dump({"rows": rows}, open(os.path.join(OUT, "pixcheck_rows.json"), "w")); json.dump(R, open(os.path.join(OUT, "report.json"), "w"), indent=1, default=str)
    files += ["pixcheck_rows.json", "report.json"]
    man = {n: {"bytes": os.path.getsize(os.path.join(OUT, n)), "sha256": hashlib.sha256(open(os.path.join(OUT, n), "rb").read()).hexdigest()} for n in files}
    json.dump({"files": man}, open(os.path.join(OUT, "export_manifest.json"), "w")); emit("exported", files={k: v["bytes"] for k, v in man.items()})
    tok = os.environ.get("RESULTS_TOKEN", "").encode()
    if len(tok) < 24: emit("serve_skipped"); return
    st = {"ack": False, "fails": 0}
    class Hd(http.server.BaseHTTPRequestHandler):
        def log_message(s, *a): pass
        def _ok(s):
            if st["fails"] >= 20: return False
            g = s.headers.get("Authorization", ""); g = g[7:] if g.startswith("Bearer ") else ""
            if hmac.compare_digest(g.encode(), tok): return True
            st["fails"] += 1; time.sleep(0.5); return False
        def _s(s, c, b=b"", ct="application/octet-stream"):
            s.send_response(c); s.send_header("Content-Type", ct); s.send_header("Content-Length", str(len(b))); s.end_headers(); s.wfile.write(b)
        def do_GET(s):
            if not s._ok(): return s._s(401, b"unauthorized")
            if s.path == "/manifest": return s._s(200, json.dumps({"files": man}).encode(), "application/json")
            if s.path.startswith("/f/") and s.path[3:] in man: return s._s(200, open(os.path.join(OUT, s.path[3:]), "rb").read())
            s._s(404, b"not found")
        def do_POST(s):
            if not s._ok(): return s._s(401, b"unauthorized")
            if s.path == "/ack": st["ack"] = True; return s._s(200, b"ack")
            s._s(404, b"not found")
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", 8000), Hd); srv.daemon_threads = True; threading.Thread(target=srv.serve_forever, daemon=True).start()
    emit("serving", port=8000); t_end = time.time() + int(os.environ.get("SERVE_S", "600"))
    while time.time() < t_end and not st["ack"]: time.sleep(1)
    srv.shutdown(); emit("serve_end", acked=st["ack"])

if __name__ == "__main__": main()
