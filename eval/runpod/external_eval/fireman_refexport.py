import collections, gzip, hashlib, hmac, http.server, io, json, os, tarfile, threading, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

WS = os.environ.get("WS", "/workspace")
RUNS = os.path.join(WS, "runs", "fireman_refexport"); OUT = os.path.join(WS, "external_eval", "refthumbs_v1")
DS = os.path.join(WS, "external_eval", "fasdd_uav_test_v1", "images")
T0 = time.time()
def emit(k, **kw): print(json.dumps({"stage": k, "t": round(time.time() - T0, 1), **kw}, default=str), flush=True)

def thumbs(path):
    im = Image.open(path); g = im.convert("L")
    t32 = np.asarray(g.resize((32, 32), Image.BILINEAR), np.uint8).tobytes()      # identical to the FireMan thumbnail function
    rgb = im.convert("RGB"); rgb.thumbnail((160, 120), Image.BILINEAR)
    b = io.BytesIO(); rgb.save(b, "JPEG", quality=60)
    return t32, b.getvalue(), list(im.size)

def main():
    os.makedirs(RUNS, exist_ok=True)
    try:
        fd = os.open(os.path.join(RUNS, "STARTED"), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, b"1"); os.close(fd)
    except FileExistsError:
        print("GUARD: already started; exiting", flush=True); return
    if os.path.exists(OUT): emit("abort", reason="output exists"); return
    idx = {}
    for dp, _, fs in os.walk(os.path.join(WS, "boreal")):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and (f not in idx or "boreal_yolo" in dp): idx[f] = os.path.join(dp, f)
    scan = json.load(gzip.open(os.path.join(WS, "integrity", "boreal_scan.json.gz")))
    refs = [("boreal", f"{sp}/{nm}", idx[nm]) for sp, d in scan.items() for nm in d if nm in idx]
    refs += [("fasdd", f[:-4], os.path.join(DS, f)) for f in sorted(os.listdir(DS))]
    emit("refs", n=len(refs), boreal=sum(1 for r in refs if r[0] == "boreal"))
    with ProcessPoolExecutor(4) as ex: res = list(ex.map(thumbs, [r[2] for r in refs], chunksize=16))
    emit("thumbs", n=len(res))
    os.makedirs(OUT)
    np.save(os.path.join(OUT, "refs32.npy"), np.array([np.frombuffer(r[0], np.uint8) for r in res], np.uint8))
    json.dump([{"ds": r[0], "key": r[1], "size": x[2]} for r, x in zip(refs, res)], open(os.path.join(OUT, "refs_index.json"), "w"))
    with tarfile.open(os.path.join(OUT, "refs_jpg.tar"), "w") as tf:
        for i, x in enumerate(res):
            ti = tarfile.TarInfo(f"{i:05d}.jpg"); ti.size = len(x[1]); tf.addfile(ti, io.BytesIO(x[1]))
    files = ["refs32.npy", "refs_index.json", "refs_jpg.tar"]
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
    emit("serving", port=8000); t_end = time.time() + int(os.environ.get("SERVE_S", "900"))
    while time.time() < t_end and not st["ack"]: time.sleep(1)
    srv.shutdown(); emit("serve_end", acked=st["ack"])

if __name__ == "__main__": main()
