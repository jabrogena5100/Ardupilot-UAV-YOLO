import gzip, glob, hashlib, hmac, http.server, json, os, posixpath, sys, tarfile, time, zipfile
ZIP = os.environ.get("FZIP", "/workspace/datasets/FASDD_UAV.zip"); ZB = 14888184890; WANT = "41864dcd5ff9af4f73e78a3103debafbd5c46bcf8fedfc3e783117062e1df734"
EXT = "/workspace/external_eval/fasdd_uav_test_positive_v1"; EXP = "/workspace/runs/fasdd_pod_export_v1"; TOK = os.environ["RESULTS_TOKEN"]
def emit(k, **kw): print(json.dumps({"stage": k, **kw}, default=str), flush=True)
if not os.path.isfile(ZIP):
    emit("zip_missing", datasets=os.listdir("/workspace/datasets"), found=glob.glob("/workspace/**/FASDD*", recursive=True)[:10]); sys.exit(2)
emit("zip", bytes=os.path.getsize(ZIP), expected=ZB)
if os.path.getsize(ZIP) != ZB: sys.exit(3)
z = zipfile.ZipFile(ZIP); st = lambda p: posixpath.splitext(posixpath.basename(p))[0]
test = [st(l.strip()) for l in z.read("annotations/YOLO_UAV/test.txt").decode().splitlines() if l.strip()]
pos = sorted(s for s in test if s.startswith(("smoke_", "bothFireAndSmoke_")))
emit("list", n=len(pos), sha=hashlib.sha256("\n".join(pos).encode()).hexdigest(), want=WANT)
if len(pos) != 2149 or hashlib.sha256("\n".join(pos).encode()).hexdigest() != WANT: sys.exit(4)
hp = (glob.glob("/workspace/runs/fasdd_overlap_v2/hashes_v2.json*") + glob.glob("/workspace/external_eval/**/hashes_v2.json*", recursive=True))
if not hp: emit("hashes_missing"); sys.exit(5)
md5 = json.load((gzip.open if hp[0].endswith(".gz") else open)(hp[0]))["fasdd"]
if os.path.exists(EXT) or os.path.exists(EXP): emit("refuse_overwrite"); sys.exit(6)
for d in ("images", "labels_original", "labels_smoke_only"): os.makedirs(os.path.join(EXT + "_partial", d))
boxes = 0
for s in pos:
    raw = z.read("images/%s.jpg" % s)
    if hashlib.md5(raw).hexdigest() != md5[s]["md5"]: emit("md5_mismatch", name=s); sys.exit(7)
    open("%s_partial/images/%s.jpg" % (EXT, s), "wb").write(raw)
    lab = z.read("annotations/YOLO_UAV/labels/%s.txt" % s); open("%s_partial/labels_original/%s.txt" % (EXT, s), "wb").write(lab)
    keep = ["0 " + " ".join(l.split()[1:]) for l in lab.decode().splitlines() if len(l.split()) >= 5 and int(l.split()[0]) == 1]
    boxes += len(keep); open("%s_partial/labels_smoke_only/%s.txt" % (EXT, s), "w").write("\n".join(keep) + ("\n" if keep else ""))
emit("boxes", n=boxes)
if boxes != 2877: sys.exit(8)
os.rename(EXT + "_partial", EXT); os.makedirs(EXP)
tp = EXP + "/fasdd_positive_v1.tar"
with tarfile.open(tp, "w") as tf: tf.add(EXT, arcname="fasdd_uav_test_positive_v1")
h = hashlib.sha256(); [h.update(b) for b in iter(lambda f=open(tp, "rb"): f.read(1 << 20), b"")]
man = {"bytes": os.path.getsize(tp), "sha256": h.hexdigest()}; emit("tar", **man)
state = {"ack": False}
class H(http.server.BaseHTTPRequestHandler):
    def ok(self): return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + TOK)
    def do_GET(self):
        if not self.ok(): self.send_response(403); self.end_headers(); return
        if self.path == "/manifest": b = json.dumps(man).encode(); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        elif self.path == "/f/tar":
            self.send_response(200); self.send_header("Content-Length", str(man["bytes"])); self.end_headers()
            with open(tp, "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""): self.wfile.write(b)
        else: self.send_response(404); self.end_headers()
    def do_POST(self):
        if self.ok() and self.path == "/ack": state["ack"] = True
        self.send_response(200); self.end_headers()
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(("0.0.0.0", 8000), H); srv.timeout = 1; t0 = time.time()
while not state["ack"] and time.time() - t0 < 1800: srv.handle_request()
emit("done", acked=state["ack"])
