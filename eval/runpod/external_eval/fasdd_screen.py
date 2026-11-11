import collections, csv, gzip, hashlib, hmac, http.server, json, os, random, shutil, sys, threading, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

WS = os.environ.get("WS", "/workspace")
RUNS = os.path.join(WS, "runs", "fasdd_screen")
DS = os.path.join(WS, "external_eval", "fasdd_uav_test_v1")
OUT = os.path.join(WS, "external_eval", "screening_v1")
SCAN = os.path.join(WS, "integrity", "boreal_scan.json.gz")
SCAN_SHA = os.environ.get("FSCANSHA", "8a4db032e44c4040872d6b4863af9efefe7b2e5ac3b1f6e48b6f993a3e400e8e")
SEED = 20261109
QUOTA = {"Smoke": 20, "FireAndSmoke": 20, "Fire": 30, "NeitherFireNorSmoke": 30}
SPOT = {"Smoke": 6, "FireAndSmoke": 6, "Fire": 9, "NeitherFireNorSmoke": 9}
if os.environ.get("FQ"):
    QUOTA = {"Smoke": 2, "FireAndSmoke": 2, "Fire": 3, "NeitherFireNorSmoke": 3}; SPOT = {"Smoke": 1, "FireAndSmoke": 1, "Fire": 1, "NeitherFireNorSmoke": 1}
REAL2STRATUM = (("bothFireAndSmoke", "FireAndSmoke"), ("neitherFireNorSmoke", "NeitherFireNorSmoke"), ("smoke", "Smoke"), ("fire", "Fire"))
T0 = time.time()
R = {}


def emit(k, **kw):
    R[k] = kw
    print(json.dumps({"stage": k, "t": round(time.time() - T0, 1), **kw}, default=str), flush=True)


def stratum(n):
    for p, s in REAL2STRATUM:
        if n.startswith(p):
            return s
    return "UNKNOWN"


def rank_key(salt, name):   # identical definition to eval/external_prep.py (sha256 of "seed:salt:name")
    return hashlib.sha256(f"{SEED}:{salt}:{name}".encode()).hexdigest()


def draw(names, n, salt):
    return sorted(sorted(names, key=lambda x: rank_key(salt, x))[:n])


def pc(x):
    x = x.astype(np.uint64)
    return np.bitwise_count(x).astype(np.int64) if hasattr(np, "bitwise_count") else np.array([bin(int(v)).count("1") for v in x])


def dh(path):
    """md5, size, dHash64 (lanczos/bilinear/box; bit order identical to eval/external_prep.py), near-gray flag"""
    from PIL import Image
    data = open(path, "rb").read()
    o = {"md5": hashlib.md5(data).hexdigest()}
    with Image.open(path) as im:
        o["size"] = list(im.size)
        g = im.convert("L")
        for nm, f in (("lanczos", Image.LANCZOS), ("bilinear", Image.BILINEAR), ("box", Image.BOX)):
            px = list(g.resize((9, 8), f).tobytes()); v = 0
            for r in range(8):
                for c in range(8):
                    v = (v << 1) | int(px[r * 9 + c + 1] > px[r * 9 + c])
            o[nm] = "%016x" % v
        t = im.convert("RGB").resize((64, 64)); px = list(t.tobytes())
        trip = [(px[i], px[i + 1], px[i + 2]) for i in range(0, len(px), 3)]
        o["gray"] = sum(1 for r, g_, b in trip if abs(r - g_) < 4 and abs(g_ - b) < 4) / len(trip)
    return o


def gray_only(p):
    return dh(p)["gray"]


def main():
    os.makedirs(RUNS, exist_ok=True)
    try:
        fd = os.open(os.path.join(RUNS, "STARTED"), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(time.time()).encode()); os.close(fd)
    except FileExistsError:
        print("GUARD: already started; exiting", flush=True); return
    if os.path.exists(OUT):
        emit("abort", reason="screening_v1 already exists"); return
    H = json.load(open(os.path.join(RUNS, "..", "fasdd_extract", "fasdd_test_hashes.json")))
    test = sorted(H)
    emit("discover", integrity=sorted(os.listdir(os.path.join(WS, "integrity"))) if os.path.isdir(os.path.join(WS, "integrity")) else None,
         boreal=sorted(os.listdir(os.path.join(WS, "boreal"))) if os.path.isdir(os.path.join(WS, "boreal")) else None, n_fasdd_test_hashes=len(test))
    # ---- Boreal reference
    scan = None
    if os.path.isfile(SCAN):
        sh = hashlib.sha256(open(SCAN, "rb").read()).hexdigest()
        emit("scan_file", path=SCAN, sha256=sh, matches_recorded=sh == SCAN_SHA)
        if sh == SCAN_SHA:
            scan = json.load(gzip.open(SCAN))
    else:
        emit("scan_file", path=SCAN, exists=False)
    flagged = {}
    ov = None
    if scan is None:
        emit("overlap_unavailable", reason="Boreal scan missing or hash differs from the recorded one; no substitute reference used")
    else:
        names, rd, rm = [], [], {}
        for sp, d in scan.items():
            for nm, v in d.items():
                if v[3] and not str(v[3]).startswith("ERR"):
                    names.append(f"{sp}/{nm}"); rd.append(int(v[3], 16))
                rm.setdefault(v[0], []).append(f"{sp}/{nm}")
        emit("scan_loaded", per_split={k: len(v) for k, v in scan.items()}, n_hashes=len(rd), n_unique_md5=len(rm))
        ref = np.array(rd, dtype=np.uint64)
        # P1: recompute dHash from the Boreal source images for a seeded sample and compare with the stored scan values
        idx = {}
        for dp, _, fs in os.walk(os.path.join(WS, "boreal")):
            for f in fs:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    idx.setdefault(f, os.path.join(dp, f))
        cand = [(sp, nm) for sp, d in scan.items() for nm in d if nm in idx]
        rnd = random.Random(SEED); smp = rnd.sample(cand, min(300, len(cand)))
        res = {f: [] for f in ("lanczos", "bilinear", "box")}; mdok = 0
        for sp, nm in smp:
            o = dh(idx[nm]); st = int(scan[sp][nm][3], 16); mdok += o["md5"] == scan[sp][nm][0]
            for f in res:
                res[f].append(bin(int(o[f], 16) ^ st).count("1"))
        emit("p1_convention_check", boreal_images_indexed=len(idx), scan_names_found=len(cand), sampled=len(smp), md5_equal_to_scan=mdok,
             per_filter={f: {"median_hamming": float(np.median(v)) if v else None, "frac_le2": round(sum(x <= 2 for x in v) / max(len(v), 1), 3),
                             "frac_eq0": round(sum(x == 0 for x in v) / max(len(v), 1), 3), "frac_ge60_complement": round(sum(x >= 62 for x in v) / max(len(v), 1), 3)} for f, v in res.items()})
        # overlap
        rows = []
        hist = collections.Counter(); hist_plain = collections.Counter()
        for s in test:
            h = H[s]; md = rm.get(h["md5"], [])
            best, bn, bf, bplain = 64, "", "", 64
            for f in ("lanczos", "bilinear", "box"):
                d = pc(ref ^ np.uint64(int(h[f], 16))); dm = np.minimum(d, 64 - d)
                k = int(dm.argmin())
                if dm[k] < best:
                    best, bn, bf = int(dm[k]), names[k], f
                bplain = min(bplain, int(d.min()))
            rows.append({"image": s, "md5_match": md, "min_dist": best, "plain_min_dist": bplain, "nearest": bn, "filter": bf})
            hist[min(best, 20)] += 1; hist_plain[min(bplain, 20)] += 1
        for r in rows:
            if r["md5_match"] or r["min_dist"] <= 4:
                flagged[r["image"]] = r
        emit("overlap", n_test=len(rows), md5_exact_matches=sum(1 for r in rows if r["md5_match"]), flagged_le4_or_md5=len(flagged),
             n_le0=sum(r["min_dist"] == 0 for r in rows), n_le2=sum(r["min_dist"] <= 2 for r in rows), n_le4=sum(r["min_dist"] <= 4 for r in rows),
             n_le6=sum(r["min_dist"] <= 6 for r in rows), n_le8=sum(r["min_dist"] <= 8 for r in rows),
             flagged_only_via_complement=sum(1 for r in flagged.values() if r["plain_min_dist"] > 4 and not r["md5_match"]),
             min_dist_hist_with_complement=dict(sorted(hist.items())), min_dist_hist_plain=dict(sorted(hist_plain.items())),
             flagged_examples=[{k: r[k] for k in ("image", "min_dist", "plain_min_dist", "nearest", "filter", "md5_match")} for r in sorted(flagged.values(), key=lambda r: r["min_dist"])[:40]],
             ambiguous_5_to_8_examples=[{k: r[k] for k in ("image", "min_dist", "nearest", "filter")} for r in sorted(rows, key=lambda r: r["min_dist"]) if 5 <= r["min_dist"] <= 8][:30])
        ov = {r["image"]: r for r in rows}
    # ---- labels and groups
    lab = {}
    for s in test:
        ls = [l.split() for l in open(os.path.join(DS, "labels_original", s + ".txt")).read().splitlines() if l.strip()]
        lab[s] = ls
    pos = {s for s in test if any(l[0] == "1" for l in lab[s])}
    emit("test_labels", n=len(test), smoke_positive=len(pos), by_stratum=dict(collections.Counter(stratum(s) for s in test)))
    # ---- gray flag
    with ProcessPoolExecutor(4) as ex:
        gr = list(ex.map(gray_only, [os.path.join(DS, "images", s + ".jpg") for s in test], chunksize=32))
    gray = {s for s, g in zip(test, gr) if g > 0.98}
    emit("near_gray", n=len(gray), frac=round(len(gray) / len(test), 4), by_stratum=dict(collections.Counter(stratum(s) for s in gray)), heuristic=">98% of 64x64 pixels with |R-G|<4 and |G-B|<4")
    # ---- sample (candidates exclude overlap-flagged images)
    cands = [s for s in test if s not in flagged]
    by = collections.defaultdict(list)
    for s in cands:
        by[stratum(s)].append(s)
    rows_s = []; short = {}
    for st, q in QUOTA.items():
        have = by.get(st, [])
        if len(have) < q:
            short[st] = {"have": len(have), "need": q}
        pick = draw(have, min(q, len(have)), "screen")
        spot = set(draw(pick, min(SPOT[st], len(pick)), "spot"))
        rows_s += [(s, st, int(s in spot)) for s in pick]
    emit("sample", n=len(rows_s), seed=SEED, candidates=len(cands), excluded_by_overlap=len(flagged), short_strata=short,
         per_stratum=dict(collections.Counter(r[1] for r in rows_s)), spot_checks=sum(r[2] for r in rows_s), gray_in_sample=sum(1 for r in rows_s if r[0] in gray),
         note="stratum from the real file-name prefixes (bothFireAndSmoke/smoke/fire/neitherFireNorSmoke); eval/external_prep.py::stratum expects capitalised prefixes and returns UNKNOWN for these names, so its screening_sample() was not called, only its draw/rank rule was reproduced")
    insample = {r[0] for r in rows_s}
    extras = draw([s for s in cands if s in gray and s not in insample], 12, "gray")
    emit("g5_extras", n=len(extras), names=extras)
    # ---- clustering sensitivity (same method as the extraction job: connected components over dHash64 lanczos)
    def clusters(sub, tols=(2, 4, 6)):
        v = [int(H[s]["lanczos"], 16) for s in sub]; n = len(v)
        edges = [(i, j, (v[i] ^ v[j]).bit_count()) for i in range(n) for j in range(i + 1, n) if (v[i] ^ v[j]).bit_count() <= 6]
        out = {}
        for t in tols:
            par = list(range(n))
            def f(a):
                while par[a] != a:
                    par[a] = par[par[a]]; a = par[a]
                return a
            for i, j, d in edges:
                if d <= t:
                    par[f(i)] = f(j)
            comp = collections.defaultdict(list)
            for i in range(n):
                comp[f(i)].append(i)
            sz = sorted((len(c) for c in comp.values()), reverse=True)
            out[f"r{t}"] = {"clusters": len(comp), "singletons": sum(1 for x in sz if x == 1), "largest": sz[:6], "median_size": sz[len(sz) // 2],
                            "images_in_largest_pct": round(100 * sz[0] / n, 1)}
        return out
    emit("cluster_sensitivity_all_test", n=len(test), **clusters(test))
    ps = sorted(pos); emit("cluster_sensitivity_smoke_positive", n=len(ps), **clusters(ps))
    ps2 = sorted(pos - set(flagged)); emit("cluster_sensitivity_smoke_positive_after_overlap_exclusion", n=len(ps2), **clusters(ps2))
    # ---- contact sheets + manifest
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(OUT)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    items = [(i + 1, s, st, sp, "S") for i, (s, st, sp) in enumerate(sorted(rows_s, key=lambda r: (list(QUOTA).index(r[1]), r[0])))]
    items += [(f"G{i + 1}", s, stratum(s), 0, "G5") for i, s in enumerate(extras)]
    W, Hh, LAB = 560, 373, 36
    per = 12
    manifest = []; files = []
    for sh in range((len(items) + per - 1) // per):
        chunk = items[sh * per:(sh + 1) * per]
        sheet = Image.new("RGB", (3 * W, 4 * (Hh + LAB)), (30, 30, 30)); d = ImageDraw.Draw(sheet)
        for k, (no, s, st, sp, kind) in enumerate(chunk):
            x0, y0 = (k % 3) * W, (k // 3) * (Hh + LAB)
            im = Image.open(os.path.join(DS, "images", s + ".jpg")).convert("RGB"); w0, h0 = im.size
            sc = min((W - 4) / w0, Hh / h0); tw, th = int(w0 * sc), int(h0 * sc)
            th_im = im.resize((tw, th), Image.LANCZOS); dr = ImageDraw.Draw(th_im)
            for l in lab[s]:
                c = int(l[0]); x, y, w, h = map(float, l[1:])
                dr.rectangle([(x - w / 2) * tw, (y - h / 2) * th, (x + w / 2) * tw, (y + h / 2) * th], outline=(255, 255, 0) if c == 1 else (255, 60, 60), width=2)
            sheet.paste(th_im, (x0 + 2, y0 + LAB))
            nsm = sum(1 for l in lab[s] if l[0] == "1"); nfi = sum(1 for l in lab[s] if l[0] == "0")
            d.text((x0 + 4, y0 + 2), f"#{no} {s}", fill=(255, 255, 255), font=font)
            d.text((x0 + 4, y0 + 18), f"{st}{' SPOT' if sp else ''}{' GRAY' if s in gray else ''} {w0}x{h0} smoke={nsm} fire={nfi}", fill=(160, 220, 255), font=font)
            manifest.append({"no": no, "sheet": f"sheet_{sh + 1:02d}.jpg", "image": s + ".jpg", "stratum": st, "kind": kind, "spot_check": sp, "width": w0, "height": h0,
                             "n_smoke_boxes": nsm, "n_fire_boxes": nfi, "near_gray": int(s in gray), "boreal_min_dist": (ov[s]["min_dist"] if ov else None)})
        fn = f"sheet_{sh + 1:02d}.jpg"; sheet.save(os.path.join(OUT, fn), quality=88); files.append(fn)
    with open(os.path.join(OUT, "screening_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); cols = list(manifest[0]) + ["screener_smoke_visible", "screener_smoke_labeled", "screener_verdict", "owner_verdict", "notes"]
        w.writerow(cols)
        for m in manifest:
            w.writerow([m[k] for k in manifest[0]] + ["", "", "", "", ""])
    files.append("screening_manifest.csv")
    emit("sheets", n_sheets=len(files) - 1, n_items=len(items), legend="yellow box = smoke (FASDD class 1), red box = fire (class 0); #1-100 protocol sample, G1-G12 near-gray extras for the RGB check")
    json.dump(R, open(os.path.join(RUNS, "report.json"), "w"), indent=1, default=str)
    shutil.copy(os.path.join(RUNS, "report.json"), os.path.join(OUT, "report.json")); files.append("report.json")
    man = {n: {"bytes": os.path.getsize(os.path.join(OUT, n)), "sha256": hashlib.sha256(open(os.path.join(OUT, n), "rb").read()).hexdigest()} for n in files}
    json.dump({"files": man}, open(os.path.join(OUT, "export_manifest.json"), "w"))
    emit("exported", files={n: m["bytes"] for n, m in man.items()}, dir=OUT)
    # ---- serve (authenticated, read-only, allowlisted; same routes as eval/runpod/results_server.py so fetch_results.py works unchanged)
    tok = os.environ.get("RESULTS_TOKEN", "").encode()
    if len(tok) < 24:
        emit("serve_skipped", reason="RESULTS_TOKEN missing"); return
    st = {"ack": False, "fails": 0}

    class Hd(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _ok(self):
            if st["fails"] >= 20: return False
            g = self.headers.get("Authorization", ""); g = g[7:] if g.startswith("Bearer ") else ""
            if hmac.compare_digest(g.encode(), tok): return True
            st["fails"] += 1; time.sleep(0.5); return False
        def _s(self, code, body=b"", ct="application/octet-stream"):
            self.send_response(code); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if not self._ok(): return self._s(401, b"unauthorized")
            if self.path == "/manifest": return self._s(200, json.dumps({"files": man}).encode(), "application/json")
            if self.path.startswith("/f/") and self.path[3:] in man: return self._s(200, open(os.path.join(OUT, self.path[3:]), "rb").read())
            self._s(404, b"not found")
        def do_POST(self):
            if not self._ok(): return self._s(401, b"unauthorized")
            if self.path == "/ack": st["ack"] = True; return self._s(200, b"ack")
            self._s(404, b"not found")
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", 8000), Hd); srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    emit("serving", port=8000, window_s=int(os.environ.get("SERVE_S", "600")))
    t_end = time.time() + int(os.environ.get("SERVE_S", "600"))
    while time.time() < t_end and not st["ack"]:
        time.sleep(1)
    srv.shutdown()
    emit("serve_end", acked=st["ack"])


if __name__ == "__main__":
    main()
