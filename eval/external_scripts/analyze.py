import json, gzip, collections, os, sys, statistics
import numpy as np
B = "/home/joshua/src/Ardupilot-UAV-YOLO/local_data/external_eval/"
fr = {}
for l in open(B + "fireman/work/frames_every20.jsonl"):
    try: o = json.loads(l)
    except Exception: continue
    fr[o["name"]] = o
names = sorted(fr)
H = json.load(gzip.open(B + "screening_v2/hashes_v2.json.gz"))
bor, fas = H["boreal"], H["fasdd"]
F = ("lanczos", "bilinear", "box")
def arr(d, keys): return np.array([[int(d[k][f], 16) for f in F] for k in keys], dtype=np.uint64)
fm = np.array([[int(fr[n][f], 16) for f in F] for n in names], dtype=np.uint64)
bn = sorted(bor); fn_ = sorted(fas)
bm, fsm = arr(bor, bn), arr(fas, fn_)
def pc(x): return np.bitwise_count(x).astype(np.int64)
def mind(Q, R):
    out = np.full(len(Q), 64); arg = np.zeros(len(Q), int)
    for i in range(len(Q)):
        d = np.min(np.stack([pc(R[:, f] ^ Q[i, f]) for f in range(3)]), axis=0)
        j = int(d.argmin()); out[i] = d[j]; arg[i] = j
    return out, arg
vid = lambda n: n.split("Unimodal/rgb/")[1].rsplit("/images/", 1)[0]
vids = np.array([vid(n) for n in names])
dB, aB = mind(fm, bm); dF, aF = mind(fm, fsm)
def thr(d): return {t: int((d <= t).sum()) for t in (0, 2, 4, 6, 8)}
print("FireMan frames hashed:", len(names), "videos:", len(set(vids)))
print("vs Boreal (4,954):", thr(dB), " vs FASDD test (4,181):", thr(dF))
md5b = {v["md5"] for v in bor.values()}; md5f = {v["md5"] for v in fas.values()}
print("md5 matches Boreal:", sum(fr[n]["md5"] in md5b for n in names), "FASDD:", sum(fr[n]["md5"] in md5f for n in names))
# null reference: FASDD test -> Boreal (known no overlap) distribution of min distance, vs FireMan -> Boreal
dFB, _ = mind(fsm[:1500], bm)
q = lambda d: {p: float(np.percentile(d, p)) for p in (1, 5, 25, 50)}
print("min-dist percentiles FireMan->Boreal", q(dB), "| FASDD->Boreal (reference)", q(dFB))
# per video: min and 5th pct distance to Boreal and nearest Boreal event
ev = lambda b: b.split("/", 1)[1].split("-")[0] if "/" in b else b
per = collections.defaultdict(list)
for n, d, j in zip(names, dB, aB): per[vid(n)].append((int(d), ev(bn[j])))
print("video | n_frames | min dist to Boreal | median | nearest event of the closest frame")
rows = []
for v, L in sorted(per.items()):
    ds = [x[0] for x in L]; best = min(L)
    rows.append((v, len(L), best[0], int(np.median(ds)), best[1])); print(*rows[-1])
json.dump({"n_frames": len(names), "vs_boreal": thr(dB), "vs_fasdd": thr(dF), "per_video": rows, "pct_fireman_boreal": q(dB), "pct_fasdd_boreal": q(dFB)}, open(B + "fireman/work/overlap_boreal.json", "w"), indent=1)
# cross-video near duplicates inside FireMan (different videos)
cv = []
for i in range(len(names)):
    d = np.min(np.stack([pc(fm[:, f] ^ fm[i, f]) for f in range(3)]), axis=0)
    for j in np.where((d <= 4) & (vids != vids[i]) & (np.arange(len(names)) > i))[0]:
        cv.append((vids[i], vids[j], int(d[j])))
pairs = collections.Counter((a, b) for a, b, d in cv)
print("cross-video near-duplicate frame pairs (<=4 bits):", len(cv), " video pairs:", len(pairs))
for (a, b), c in pairs.most_common(15): print("  ", a, b, c)
# frame correlation within video at lag 20 (thumb NCC)
T = {n: np.frombuffer(bytes.fromhex(fr[n]["thumb"]), np.uint8).astype(np.float64) for n in names}
def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-6 and nb > 1e-6 else None
byv = collections.defaultdict(list)
for n in names: byv[vid(n)].append(n)
lagn = {}
for v, L in byv.items():
    L.sort(key=lambda n: int(n.rsplit("_", 1)[1].split(".")[0]))
    x = [ncc(T[a], T[b]) for a, b in zip(L, L[1:])]; x = [y for y in x if y is not None]
    lagn[v] = round(float(np.median(x)), 3) if x else None
print("median NCC between frames 20 apart, per video:", lagn)
json.dump({"cross_video_pairs": [[a, b, c] for (a, b), c in pairs.items()], "lag20_ncc": lagn}, open(B + "fireman/work/internal_similarity.json", "w"))
