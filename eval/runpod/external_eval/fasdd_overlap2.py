import collections, csv, gzip, hashlib, hmac, http.server, json, os, random, shutil, threading, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

WS = os.environ.get("WS", "/workspace")
RUNS = os.path.join(WS, "runs", "fasdd_overlap_v2")
DS = os.path.join(WS, "external_eval", "fasdd_uav_test_v1")
V1 = os.path.join(WS, "external_eval", "screening_v1")
OUT = os.path.join(WS, "external_eval", "screening_v2")
SCAN = os.path.join(WS, "integrity", "boreal_scan.json.gz")
SCAN_SHA = os.environ.get("FSCANSHA", "8a4db032e44c4040872d6b4863af9efefe7b2e5ac3b1f6e48b6f993a3e400e8e")
SEED = 20261109
QUOTA = {"Smoke": 20, "FireAndSmoke": 20, "Fire": 30, "NeitherFireNorSmoke": 30}
SPOT = {"Smoke": 6, "FireAndSmoke": 6, "Fire": 9, "NeitherFireNorSmoke": 9}
if os.environ.get("FQ"):
    QUOTA = {"Smoke": 2, "FireAndSmoke": 2, "Fire": 3, "NeitherFireNorSmoke": 3}; SPOT = {"Smoke": 1, "FireAndSmoke": 1, "Fire": 1, "NeitherFireNorSmoke": 1}
PFX = (("bothfireandsmoke", "FireAndSmoke"), ("fireandsmoke", "FireAndSmoke"), ("neitherfirenorsmoke", "NeitherFireNorSmoke"), ("smoke", "Smoke"), ("fire", "Fire"))
T0 = time.time(); R = {}


def emit(k, **kw):
    R[k] = kw
    print(json.dumps({"stage": k, "t": round(time.time() - T0, 1), **kw}, default=str), flush=True)


def stratum(n):
    low = os.path.basename(n).lower()
    for p, c in PFX:
        if low.startswith(p):
            return c
    return "UNKNOWN"


def draw(names, n, salt):
    return sorted(sorted(names, key=lambda x: hashlib.sha256(f"{SEED}:{salt}:{x}".encode()).hexdigest())[:n])


def pc(x):
    x = np.asarray(x).astype(np.uint64)
    return np.bitwise_count(x).astype(np.int64) if hasattr(np, "bitwise_count") else np.array([bin(int(v)).count("1") for v in x.ravel()]).reshape(x.shape)


def bits2int(b):
    return int.from_bytes(np.packbits(np.asarray(b, dtype=np.uint8)).tobytes(), "big")


def dh2(path):
    """ONE implementation used for both datasets: md5, size, dHash64 (next>prev, row-major MSB-first) for lanczos/bilinear/box, 32x32 gray thumb, near-gray ratio."""
    from PIL import Image
    data = open(path, "rb").read()
    o = {"md5": hashlib.md5(data).hexdigest()}
    with Image.open(path) as im:
        o["size"] = list(im.size)
        g = im.convert("L")
        for nm, f in (("lanczos", Image.LANCZOS), ("bilinear", Image.BILINEAR), ("box", Image.BOX)):
            a = np.asarray(g.resize((9, 8), f), dtype=np.int16)
            o[nm] = "%016x" % bits2int((a[:, 1:] > a[:, :-1]).flatten())
        o["thumb"] = g.resize((32, 32), Image.BILINEAR).tobytes()
        px = im.convert("RGB").resize((64, 64)).tobytes()
        o["gray"] = sum(1 for i in range(0, len(px), 3) if abs(px[i] - px[i + 1]) < 4 and abs(px[i + 1] - px[i + 2]) < 4) / 4096
    return o


def variants(path):
    from PIL import Image
    L = Image.open(path).convert("L"); out = {}
    for fn, f in (("lanczos", Image.LANCZOS), ("bilinear", Image.BILINEAR), ("bicubic", Image.BICUBIC), ("box", Image.BOX), ("nearest", Image.NEAREST), ("hamming", Image.HAMMING)):
        for shp in ("h", "v"):
            a = np.asarray(L.resize((9, 8) if shp == "h" else (8, 9), f), dtype=np.int16)
            prev, nxt = (a[:, :-1], a[:, 1:]) if shp == "h" else (a[:-1, :], a[1:, :])
            for cn, bits in (("next>prev", nxt > prev), ("prev>next", prev > nxt), ("next>=prev", nxt >= prev), ("prev>=next", prev >= nxt)):
                for on, b in (("rowmajor", bits.flatten()), ("colmajor", bits.T.flatten()), ("rowmajor_lsbfirst", bits.flatten()[::-1])):
                    out[f"{fn}|{shp}|{cn}|{on}"] = bits2int(b)
    return out


def ncc(a, b):
    a = np.frombuffer(a, np.uint8).astype(np.float64); b = np.frombuffer(b, np.uint8).astype(np.float64)
    a -= a.mean(); b -= b.mean(); na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a @ b) / (na * nb)) if na > 1e-6 and nb > 1e-6 else None


def main():
    os.makedirs(RUNS, exist_ok=True)
    try:
        fd = os.open(os.path.join(RUNS, "STARTED"), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(time.time()).encode()); os.close(fd)
    except FileExistsError:
        print("GUARD: already started; exiting", flush=True); return
    if os.path.exists(OUT):
        emit("abort", reason="screening_v2 exists"); return
    sha = hashlib.sha256(open(SCAN, "rb").read()).hexdigest()
    emit("scan_file", sha256=sha, matches_recorded=sha == SCAN_SHA)
    scan = json.load(gzip.open(SCAN))
    old = json.load(open(os.path.join(WS, "runs", "fasdd_extract", "fasdd_test_hashes.json")))
    test = sorted(old)
    idx = {}
    for dp, _, fs in os.walk(os.path.join(WS, "boreal")):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                if f not in idx or "boreal_yolo" in dp:
                    idx[f] = os.path.join(dp, f)
    bnames = [(sp, nm) for sp, d in scan.items() for nm in d if nm in idx]
    emit("boreal_index", scan_images=sum(len(d) for d in scan.values()), found_on_volume=len(bnames), files_indexed=len(idx))
    # ---- 1) why the stored hashes were incompatible: variant grid on 300 seeded Boreal images
    rnd = random.Random(SEED); smp = rnd.sample(bnames, min(300, len(bnames)))
    with ProcessPoolExecutor(4) as ex:
        vs = list(ex.map(variants, [idx[nm] for _, nm in smp], chunksize=8))
    agg = collections.defaultdict(list)
    for (sp, nm), v in zip(smp, vs):
        st = int(scan[sp][nm][4 - 1], 16)
        for k, h in v.items():
            agg[k].append(bin(h ^ st).count("1"))
    tab = sorted(({"variant": k, "frac_exact": round(sum(x == 0 for x in d) / len(d), 3), "frac_le2": round(sum(x <= 2 for x in d) / len(d), 3), "median": float(np.median(d))} for k, d in agg.items()),
                 key=lambda r: (-r["frac_exact"], -r["frac_le2"], r["median"]))
    ours = [r for r in tab if r["variant"] == "lanczos|h|next>prev|rowmajor"][0]
    emit("stored_hash_variant_study", sampled=len(smp), n_variants=len(tab), top10=tab[:10], ours_used_in_v1=ours,
         note="next>prev = right pixel brighter than left (the definition used for FASDD and in eval/external_prep.py); prev>next is the inverse comparison")
    # ---- 2) hash BOTH datasets in this run with the same function
    bpaths = [idx[nm] for _, nm in bnames]; fpaths = [os.path.join(DS, "images", s + ".jpg") for s in test]
    with ProcessPoolExecutor(4) as ex:
        BH = list(ex.map(dh2, bpaths, chunksize=16)); FH = list(ex.map(dh2, fpaths, chunksize=16))
    md5_ok = sum(1 for (sp, nm), h in zip(bnames, BH) if h["md5"] == scan[sp][nm][0])
    stored_consistent = sum(1 for s, h in zip(test, FH) if all(h[f] == old[s][f] for f in ("lanczos", "bilinear", "box")))
    emit("hashed", boreal=len(BH), fasdd=len(FH), boreal_md5_equal_to_scan=md5_ok, fasdd_hash_equal_to_extraction_run=stored_consistent)
    json.dump({"boreal": {f"{sp}/{nm}": {k: h[k] for k in ("md5", "size", "lanczos", "bilinear", "box")} for (sp, nm), h in zip(bnames, BH)},
               "fasdd": {s: {k: h[k] for k in ("md5", "size", "lanczos", "bilinear", "box")} for s, h in zip(test, FH)}}, open(os.path.join(RUNS, "hashes_v2.json"), "w"))
    # ---- 3) overlap
    F = np.array([[int(h[f], 16) for f in ("lanczos", "bilinear", "box")] for h in FH], dtype=np.uint64)
    B = np.array([[int(h[f], 16) for f in ("lanczos", "bilinear", "box")] for h in BH], dtype=np.uint64)
    bmd5 = collections.defaultdict(list)
    for (sp, nm), h in zip(bnames, BH):
        bmd5[h["md5"]].append(f"{sp}/{nm}")
    res = []; hubc = collections.Counter(); bgray = None
    for i, s in enumerate(test):
        d = np.min(np.stack([pc(B[:, f] ^ F[i, f]) for f in range(3)]), axis=0)
        order = np.argsort(d, kind="stable")[:12]
        cand = [(int(j), int(d[j])) for j in order if d[j] <= 8]
        dl = pc(B[:, 0] ^ F[i, 0])
        r = {"image": s, "min_dist": int(d.min()), "lanczos_only_min": int(dl.min()), "md5": bmd5.get(FH[i]["md5"], []), "cands": []}
        for j, dd in cand:
            n_ = ncc(FH[i]["thumb"], BH[j]["thumb"])
            r["cands"].append({"b": f"{bnames[j][0]}/{bnames[j][1]}", "j": j, "d": dd, "ncc": None if n_ is None else round(n_, 4)})
            if dd <= 4:
                hubc[j] += 1
        nn = [c["ncc"] for c in r["cands"] if c["ncc"] is not None]
        r["best_ncc"] = max(nn) if nn else None
        r["lowtex"] = any(c["ncc"] is None for c in r["cands"]) or float(np.std(np.frombuffer(FH[i]["thumb"], np.uint8))) < 12
        res.append(r)
    byi = {r["image"]: r for r in res}
    cnt = {t: sum(r["min_dist"] <= t for r in res) for t in (0, 2, 4, 6, 8)}
    cntl = {t: sum(r["lanczos_only_min"] <= t for r in res) for t in (0, 2, 4, 6, 8)}
    emit("overlap_counts", n_test=len(test), md5_exact=sum(1 for r in res if r["md5"]), min_over_3_filters=cnt, lanczos_only=cntl,
         v1_report_for_comparison={"le0": 8, "le2": 26, "le4": 167, "le6": 528, "le8": 992})
    def cls(r, t):
        if r["min_dist"] > t:
            return None
        if r["md5"]:
            return "pixel_confirmed"
        b = r["best_ncc"]
        return "pixel_confirmed" if (b is not None and b >= 0.95) else "collision_like" if (b is None or b < 0.80) else "ambiguous"
    table = {}
    for t in (0, 2, 4, 6, 8):
        c = collections.Counter(cls(r, t) for r in res if r["min_dist"] <= t)
        table[f"le{t}"] = dict(c)
    emit("overlap_classified_by_pixel_check", rule="pixel_confirmed: best 32x32 NCC >= 0.95 (or MD5); collision_like: best NCC < 0.80 or no texture; else ambiguous", table=table)
    hubs = sorted(hubc.items(), key=lambda kv: -kv[1])[:15]
    emit("hubs", top=[{"boreal": f"{bnames[j][0]}/{bnames[j][1]}", "n_fasdd_within_4": n, "lanczos_hash": "%016x" % B[j, 0], "thumb_std": round(float(np.std(np.frombuffer(BH[j]["thumb"], np.uint8))), 1)} for j, n in hubs],
         n_boreal_hubs_ge5=sum(1 for n in hubc.values() if n >= 5), n_lowtexture_flagged_le4=sum(1 for r in res if r["min_dist"] <= 4 and r["lowtex"]))
    S_pix = {r["image"] for r in res if cls(r, 4) == "pixel_confirmed"}
    S_hash = {r["image"] for r in res if r["min_dist"] <= 4 or r["md5"]}
    S_amb = {r["image"] for r in res if cls(r, 4) == "ambiguous"}
    S_wide_pix = {r["image"] for r in res if cls(r, 8) == "pixel_confirmed"}
    emit("exclusion_sets", pixel_confirmed_le4=len(S_pix), hash_le4_literal=len(S_hash), ambiguous_le4=len(S_amb), pixel_confirmed_le8=len(S_wide_pix))
    json.dump({"rows": [r for r in res if r["min_dist"] <= 8]}, open(os.path.join(RUNS, "overlap_v2_rows_le8.json"), "w"))
    # ---- 4) sample validity
    v1 = {}
    with open(os.path.join(V1, "screening_manifest.csv")) as f:
        for row in csv.DictReader(f):
            v1.setdefault(row["kind"], []).append(row["image"][:-4])
    gray = {s for s, h in zip(test, FH) if h["gray"] > 0.98}
    def redraw(excl):
        by = collections.defaultdict(list)
        for s in test:
            if s not in excl:
                by[stratum(s)].append(s)
        rows, short = [], {}
        for st, q in QUOTA.items():
            have = by.get(st, [])
            if len(have) < q: short[st] = len(have)
            pick = draw(have, min(q, len(have)), "screen"); spot = set(draw(pick, min(SPOT[st], len(pick)), "spot"))
            rows += [(s, st, int(s in spot)) for s in pick]
        ex_ = draw([s for s in by_all(excl) if s in gray and s not in {r[0] for r in rows}], 12, "gray")
        return rows, ex_, short
    def by_all(excl):
        return [s for s in test if s not in excl]
    v1s = set(v1.get("S", [])); v1g = v1.get("G5", [])
    out = {}
    for nm, ex_set in (("pixel_confirmed", S_pix), ("hash_le4_literal", S_hash)):
        rows, ex2, short = redraw(ex_set)
        new = {r[0] for r in rows}
        out[nm] = {"n_excluded": len(ex_set), "v1_images_inside_exclusion": len(v1s & ex_set), "same_as_v1": new == v1s, "dropped_from_v1": sorted(v1s - new), "added_vs_v1": sorted(new - v1s),
                   "g5_extras_same_as_v1": sorted(ex2) == sorted(v1g), "short": short}
    emit("sample_validity", v1_sample=len(v1s), v1_g5_extras=len(v1g), **{k: v for k, v in out.items()}, primary_rule="exclude MD5 matches and 4-bit hash hits confirmed by the pixel check (pixel_confirmed_le4)")
    rows, extras, short = redraw(S_pix)
    changed = (set(r[0] for r in rows) != v1s)
    # ---- 5) pair sheets + (if changed) updated screening sheets
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(OUT)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    allpairs = [(r["image"], c) for r in res for c in r["cands"] if c["ncc"] is not None]
    genuine = sorted(allpairs, key=lambda p: -p[1]["ncc"])[:20]
    seen, coll = set(), []
    for j, n in hubs:
        pairs = [(r["image"], c) for r in res for c in r["cands"] if c["j"] == j and c["d"] <= 4 and c["ncc"] is not None]
        for p in sorted(pairs, key=lambda p: p[1]["d"])[:1]:
            coll.append(p)
    rnd2 = random.Random(SEED)
    lowncc = [(r["image"], c) for r in res if r["min_dist"] <= 4 for c in r["cands"][:1] if c["ncc"] is not None and c["ncc"] < 0.8]
    coll += rnd2.sample(lowncc, min(8, len(lowncc)))
    pairs = [("closest_by_pixels", p) for p in genuine] + [("hub_collision", p) for p in coll[:20]]
    W, HH, LAB = 400, 270, 40
    files = []
    for sh in range((len(pairs) + 9) // 10):
        ch = pairs[sh * 10:(sh + 1) * 10]
        sheet = Image.new("RGB", (2 * 2 * W + 20, 5 * (HH + LAB)), (30, 30, 30)); d = ImageDraw.Draw(sheet)
        for k, (kind, (s, c)) in enumerate(ch):
            x0, y0 = (k % 2) * (2 * W + 20), (k // 2) * (HH + LAB)
            for m, pth in enumerate((os.path.join(DS, "images", s + ".jpg"), idx[c["b"].split("/", 1)[1]])):
                im = Image.open(pth).convert("RGB"); sc = min(W / im.size[0], HH / im.size[1]); im = im.resize((int(im.size[0] * sc), int(im.size[1] * sc)), Image.LANCZOS)
                sheet.paste(im, (x0 + m * W, y0 + LAB))
            d.text((x0 + 4, y0 + 2), f"{kind} | FASDD {s}  |  Boreal {c['b'].split('__', 1)[-1][:46]}", fill=(255, 255, 255), font=font)
            d.text((x0 + 4, y0 + 20), f"hash dist {c['d']}  ncc {c['ncc']}  boreal hub n={hubc.get(c['j'], 0)}", fill=(160, 220, 255), font=font)
        fn = f"pairs_{sh + 1:02d}.jpg"; sheet.save(os.path.join(OUT, fn), quality=88); files.append(fn)
    emit("pair_sheets", n_pairs=len(pairs), n_sheets=len(files), layout="left = FASDD test image, right = nearest Boreal image; first 20 pairs = highest pixel similarity among hash candidates, then hub collisions")
    lab = {s: [l.split() for l in open(os.path.join(DS, "labels_original", s + ".txt")).read().splitlines() if l.strip()] for s in test}
    if changed:
        items = [(i + 1, s, st, sp, "S") for i, (s, st, sp) in enumerate(sorted(rows, key=lambda r: (list(QUOTA).index(r[1]), r[0])))]
        items += [(f"G{i + 1}", s, stratum(s), 0, "G5") for i, s in enumerate(extras)]
        W2, H2, LB2 = 560, 373, 36
        man = []; nsh = (len(items) + 11) // 12
        for sh in range(nsh):
            chunk = items[sh * 12:(sh + 1) * 12]
            sheet = Image.new("RGB", (3 * W2, 4 * (H2 + LB2)), (30, 30, 30)); d = ImageDraw.Draw(sheet)
            for k, (no, s, st, sp, kind) in enumerate(chunk):
                x0, y0 = (k % 3) * W2, (k // 3) * (H2 + LB2)
                im = Image.open(os.path.join(DS, "images", s + ".jpg")).convert("RGB"); w0, h0 = im.size
                sc = min((W2 - 4) / w0, H2 / h0); tw, th = int(w0 * sc), int(h0 * sc); ti = im.resize((tw, th), Image.LANCZOS); dr = ImageDraw.Draw(ti)
                for l in lab[s]:
                    x, y, w, h = map(float, l[1:])
                    dr.rectangle([(x - w / 2) * tw, (y - h / 2) * th, (x + w / 2) * tw, (y + h / 2) * th], outline=(255, 255, 0) if l[0] == "1" else (255, 60, 60), width=2)
                sheet.paste(ti, (x0 + 2, y0 + LB2))
                new = (s not in (v1s if kind == "S" else set(v1g)))
                nsm = sum(1 for l in lab[s] if l[0] == "1"); nfi = sum(1 for l in lab[s] if l[0] == "0")
                d.text((x0 + 4, y0 + 2), f"#{no} {s}{'  ** NEW vs v1 **' if new else ''}", fill=(255, 220, 120) if new else (255, 255, 255), font=font)
                d.text((x0 + 4, y0 + 18), f"{st}{' SPOT' if sp else ''}{' GRAY' if s in gray else ''} {w0}x{h0} smoke={nsm} fire={nfi}", fill=(160, 220, 255), font=font)
                man.append({"no": no, "sheet": f"sheet_{sh + 1:02d}.jpg", "image": s + ".jpg", "stratum": st, "kind": kind, "spot_check": sp, "new_vs_v1": int(new), "width": w0, "height": h0,
                            "n_smoke_boxes": nsm, "n_fire_boxes": nfi, "near_gray": int(s in gray), "boreal_min_dist": byi[s]["min_dist"], "best_ncc": byi[s]["best_ncc"]})
            fn = f"sheet_{sh + 1:02d}.jpg"; sheet.save(os.path.join(OUT, fn), quality=88); files.append(fn)
        with open(os.path.join(OUT, "screening_manifest.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(list(man[0]) + ["screener_smoke_visible", "screener_smoke_labeled", "screener_verdict", "owner_verdict", "notes"])
            for m in man:
                w.writerow([m[k] for k in man[0]] + ["", "", "", "", ""])
        files.append("screening_manifest.csv")
    emit("screening_v2", regenerated=changed, n_new_vs_v1=len(set(r[0] for r in rows) - v1s), n_dropped=len(v1s - set(r[0] for r in rows)), per_stratum=dict(collections.Counter(r[1] for r in rows)),
         spot=sum(r[2] for r in rows), short=short, g5_extras_changed=sorted(extras) != sorted(v1g))
    json.dump(R, open(os.path.join(RUNS, "report.json"), "w"), indent=1, default=str)
    for n in ("report.json",):
        shutil.copy(os.path.join(RUNS, n), os.path.join(OUT, n)); files.append(n)
    for n in ("hashes_v2.json", "overlap_v2_rows_le8.json"):
        with open(os.path.join(RUNS, n), "rb") as fi, gzip.open(os.path.join(OUT, n + ".gz"), "wb") as fo:
            shutil.copyfileobj(fi, fo)
        files.append(n + ".gz")
    mf = {n: {"bytes": os.path.getsize(os.path.join(OUT, n)), "sha256": hashlib.sha256(open(os.path.join(OUT, n), "rb").read()).hexdigest()} for n in files}
    json.dump({"files": mf}, open(os.path.join(OUT, "export_manifest.json"), "w"))
    emit("exported", files={n: m["bytes"] for n, m in mf.items()})
    tok = os.environ.get("RESULTS_TOKEN", "").encode()
    if len(tok) < 24:
        emit("serve_skipped"); return
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
            if self.path == "/manifest": return self._s(200, json.dumps({"files": mf}).encode(), "application/json")
            if self.path.startswith("/f/") and self.path[3:] in mf: return self._s(200, open(os.path.join(OUT, self.path[3:]), "rb").read())
            self._s(404, b"not found")
        def do_POST(self):
            if not self._ok(): return self._s(401, b"unauthorized")
            if self.path == "/ack": st["ack"] = True; return self._s(200, b"ack")
            self._s(404, b"not found")
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", 8000), Hd); srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    emit("serving", port=8000)
    t_end = time.time() + int(os.environ.get("SERVE_S", "600"))
    while time.time() < t_end and not st["ack"]:
        time.sleep(1)
    srv.shutdown(); emit("serve_end", acked=st["ack"])


if __name__ == "__main__":
    main()
