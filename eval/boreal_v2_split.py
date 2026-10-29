"""Build and verify the clip-disjoint Boreal v2 split from the committed scan (no images needed).

Input : docs/experiments/level2_v2_audit/boreal_scan.json.gz  {split: {image: [md5, bytes, [w,h], dhash64_hex, n_label_lines|-1]}}
Output: docs/experiments/boreal_v2/{split_assignment.json, split_audit.json, filelists.json}  (see main())

Unit of assignment = normalised source clip (site prefix and `_frameN` removed, case-folded; Empty-Images frames join their clip).
The assignment is chosen by a seeded, deterministic constrained search whose objective uses ONLY dataset statistics (frame counts,
sites, background frames, box counts, cross-partition near-duplicate frame pairs). No model output of any kind is used, so the split
cannot be tuned to a score. Everything needed to rerun is recorded (seed, constraints, weights, objective of the top solutions).
"""
import argparse
import collections
import gzip
import hashlib
import itertools
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.boreal_split_audit import SPLITS, parse  # noqa: E402

PARTS = ("train", "val", "test")
SEARCH_SEED = 20261029
N_RESTARTS = 400
NEAR_BITS = 4
TARGET_FRAC = {"val": 0.15, "test": 0.15}
CONSTRAINTS = {
    "val_frames": [600, 900], "test_frames": [600, 900],
    "min_sites_in_val": 3, "min_sites_in_test": 3, "min_clips_in_val": 3, "min_clips_in_test": 3,
    "min_true_negative_frames": {"train": 100, "val": 30, "test": 50},   # Empty-Images frames
    "max_single_clip_share_of_test": 0.40, "max_single_clip_share_of_val": 0.50,
    # added before freezing, after inspecting the first candidate's composition (statistics only, no model output):
    "every_site_in_val_or_test": True,            # candidate 1 left Evo out of val and test entirely
    "min_true_negative_clips_in_test": 2,         # candidate 1 drew 59 of 61 test negatives from one all-background clip
    "min_mixed_negative_clips_in_test": 1,        # a "mixed" clip has boxes AND >= 10 Empty-Images frames (pre-smoke background of a smoke clip)
    "mixed_clip_min_negative_frames": 10,
}
WEIGHTS = {"size": 10.0, "pos_prevalence": 2.0, "boxes_per_pos": 1.0, "near_pairs": 5.0}


def clip_site(clip):
    for k in ("evo", "heinola", "karkkila", "ruokolahti"):
        if clip.startswith(k):
            return k
    raise ValueError("unknown site for clip " + clip)


def load(path):
    return json.load(gzip.open(path))


def clip_table(scan):
    clips = collections.OrderedDict()
    for s in SPLITS:
        for n, (md5, nbytes, wh, dh, nl) in sorted(scan[s].items()):
            v, f = parse(n)
            c = clips.setdefault(v, {"clip": v, "site": clip_site(v), "frames": 0, "pos": 0, "boxes": 0, "true_neg": 0,
                                     "unlabeled_nonempty": 0, "v1_split": collections.Counter(), "files": []})
            c["frames"] += 1
            c["v1_split"][s] += 1
            if nl > 0:
                c["pos"] += 1
                c["boxes"] += nl
            elif n.startswith("Empty-Images__"):
                c["true_neg"] += 1
            else:
                c["unlabeled_nonempty"] += 1  # raw frame with no label file but not from Empty-Images: NOT a verified negative
            c["files"].append({"name": n, "source_split": s, "md5": md5, "bytes": nbytes, "dhash": dh, "n_boxes": max(nl, 0),
                               "has_label_file": nl >= 0, "frame": f})
    return clips


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def near_pairs_by_clip(clips, bits=NEAR_BITS):
    """{(clipA, clipB): number of frame pairs (one frame from each clip) with dHash Hamming <= bits}, A < B."""
    names, h, owner = [], [], []
    for cname, c in clips.items():
        for f in c["files"]:
            if f["dhash"] and not f["dhash"].startswith("ERR"):
                h.append(int(f["dhash"], 16)); owner.append(cname)
    h = np.array(h, dtype=np.uint64); owner = np.array(owner)
    out = collections.Counter()
    for i in range(0, len(h), 256):
        d = popcount64(h[i:i + 256, None] ^ h[None, :])
        for a in range(d.shape[0]):
            row = d[a].copy(); row[: i + a + 1] = 99
            for j in np.where(row <= bits)[0]:
                if owner[i + a] != owner[j]:
                    out[tuple(sorted((str(owner[i + a]), str(owner[j]))))] += 1
    return dict(out)


def stats(assign, clips):
    st = {p: {"clips": [], "frames": 0, "pos": 0, "boxes": 0, "true_neg": 0, "unlabeled_nonempty": 0,
              "sites": collections.Counter(), "frames_by_clip": {}, "neg_clips": [], "mixed_neg_clips": []} for p in PARTS}
    for cname, p in assign.items():
        c = clips[cname]; s = st[p]
        s["clips"].append(cname)
        for k in ("frames", "pos", "boxes", "true_neg", "unlabeled_nonempty"):
            s[k] += c[k]
        s["sites"][c["site"]] += c["frames"]
        s["frames_by_clip"][cname] = c["frames"]
        if c["true_neg"] > 0:
            s["neg_clips"].append(cname)
            if c["pos"] > 0 and c["true_neg"] >= CONSTRAINTS["mixed_clip_min_negative_frames"]:
                s["mixed_neg_clips"].append(cname)
    return st


def cross_near(assign, near):
    return sum(v for (a, b), v in near.items() if assign[a] != assign[b])


def feasible(st, total):
    cs = CONSTRAINTS
    why = []
    if not (cs["val_frames"][0] <= st["val"]["frames"] <= cs["val_frames"][1]): why.append("val_frames")
    if not (cs["test_frames"][0] <= st["test"]["frames"] <= cs["test_frames"][1]): why.append("test_frames")
    if len(st["val"]["sites"]) < cs["min_sites_in_val"]: why.append("val_sites")
    if len(st["test"]["sites"]) < cs["min_sites_in_test"]: why.append("test_sites")
    if len(st["train"]["sites"]) < 4: why.append("train_sites")
    if len(st["val"]["clips"]) < cs["min_clips_in_val"]: why.append("val_clips")
    if len(st["test"]["clips"]) < cs["min_clips_in_test"]: why.append("test_clips")
    for p, m in cs["min_true_negative_frames"].items():
        if st[p]["true_neg"] < m: why.append(p + "_neg")
    if CONSTRAINTS["every_site_in_val_or_test"] and (set(st["val"]["sites"]) | set(st["test"]["sites"])) != {"evo", "heinola", "karkkila", "ruokolahti"}:
        why.append("site_in_eval")
    if len(st["test"]["neg_clips"]) < cs["min_true_negative_clips_in_test"]: why.append("test_neg_clips")
    if len(st["test"]["mixed_neg_clips"]) < cs["min_mixed_negative_clips_in_test"]: why.append("test_mixed_neg_clips")
    if max(st["test"]["frames_by_clip"].values(), default=0) > cs["max_single_clip_share_of_test"] * st["test"]["frames"]: why.append("test_clip_share")
    if max(st["val"]["frames_by_clip"].values(), default=0) > cs["max_single_clip_share_of_val"] * st["val"]["frames"]: why.append("val_clip_share")
    return why


def objective(assign, clips, near, total_frames, total_near):
    st = stats(assign, clips)
    why = feasible(st, total_frames)
    if why:
        return 1e9 + 1000 * len(why), why, st
    o = 0.0
    for p in ("val", "test"):
        o += WEIGHTS["size"] * ((st[p]["frames"] - TARGET_FRAC[p] * total_frames) / (TARGET_FRAC[p] * total_frames)) ** 2
    prev = {p: st[p]["pos"] / st[p]["frames"] for p in PARTS}
    o += WEIGHTS["pos_prevalence"] * sum((prev[p] - prev["train"]) ** 2 for p in ("val", "test")) * 10
    bpp = {p: st[p]["boxes"] / max(1, st[p]["pos"]) for p in PARTS}
    o += WEIGHTS["boxes_per_pos"] * sum((bpp[p] - bpp["train"]) ** 2 for p in ("val", "test"))
    o += WEIGHTS["near_pairs"] * cross_near(assign, near) / max(1, total_near)
    return o, [], st


def search(clips, near, seed=SEARCH_SEED, restarts=N_RESTARTS):
    rng = random.Random(seed)
    names = list(clips)
    total_frames = sum(c["frames"] for c in clips.values())
    total_near = sum(near.values())
    best = {}
    for r in range(restarts):
        a = {n: rng.choices(PARTS, weights=[0.7, 0.15, 0.15])[0] for n in names}
        cur, _, _ = objective(a, clips, near, total_frames, total_near)
        improved = True
        while improved:  # steepest-descent over single moves and pair swaps
            improved = False
            cand = []
            for n in names:
                for p in PARTS:
                    if a[n] != p:
                        b = dict(a); b[n] = p; cand.append(b)
            for n1, n2 in itertools.combinations(names, 2):
                if a[n1] != a[n2]:
                    b = dict(a); b[n1], b[n2] = a[n2], a[n1]; cand.append(b)
            scored = [(objective(b, clips, near, total_frames, total_near)[0], b) for b in cand]
            o, b = min(scored, key=lambda x: (x[0], sorted(x[1].items())))
            if o < cur - 1e-12:
                cur, a, improved = o, b, True
        key = tuple(sorted(a.items()))
        best[key] = cur
    ranked = sorted(best.items(), key=lambda kv: (kv[1], kv[0]))
    return [(dict(k), o) for k, o in ranked]


def filelists(assign, clips):
    fl = {p: [] for p in PARTS}
    for cname, p in assign.items():
        fl[p] += [f["name"] for f in clips[cname]["files"]]
    return {p: sorted(v) for p, v in fl.items()}


def list_hash(names):
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def verify(assign, clips, near):
    """Independent checks on the final assignment, recomputed from per-file records (not from the search's own statistics)."""
    fl = filelists(assign, clips)
    rec = {f["name"]: f for c in clips.values() for f in c["files"]}
    part_of = {n: p for p, v in fl.items() for n in v}
    out = {"counts": {p: len(v) for p, v in fl.items()}, "total": sum(len(v) for v in fl.values())}
    out["all_files_assigned_once"] = (out["total"] == len(rec) and len(part_of) == len(rec))
    vids = {p: {parse(n)[0] for n in v} for p, v in fl.items()}
    out["videos"] = {p: len(v) for p, v in vids.items()}
    out["pairwise_shared_videos"] = {f"{a}&{b}": sorted(vids[a] & vids[b]) for a, b in itertools.combinations(PARTS, 2)}
    out["video_disjoint"] = all(not v for v in out["pairwise_shared_videos"].values())
    md5 = collections.defaultdict(set)
    for n, r in rec.items():
        md5[r["md5"]].add(part_of[n])
    out["md5_groups_spanning_partitions"] = sum(1 for s in md5.values() if len(s) > 1)
    out["md5_duplicate_groups_total"] = sum(1 for g in collections.Counter(r["md5"] for r in rec.values()).values() if g > 1)
    # dHash near-duplicates across partitions, all thresholds from exact to 8 bits
    byp = {p: np.array([int(rec[n]["dhash"], 16) for n in fl[p] if rec[n]["dhash"] and not rec[n]["dhash"].startswith("ERR")], dtype=np.uint64)
           for p in PARTS}
    cross = {}
    for a, b in itertools.combinations(PARTS, 2):
        cnt = collections.Counter()
        for i in range(0, len(byp[a]), 256):
            d = popcount64(byp[a][i:i + 256, None] ^ byp[b][None, :])
            for t in (0, 2, 4, 8):
                cnt[f"<={t}"] += int((d <= t).sum())
        cross[f"{a}&{b}"] = dict(cnt)
    out["dhash_cross_partition_pairs"] = cross
    out["dhash_cross_partition_pairs_<=4_by_clip_pair"] = {f"{a}|{b}": v for (a, b), v in sorted(near.items(), key=lambda kv: -kv[1])
                                                           if assign[a] != assign[b]}
    out["dhash_within_clip_note"] = "adjacent frames of one clip are similar by design; only cross-partition pairs matter"
    st = stats(assign, clips)
    out["partitions"] = {p: {"clips": sorted(st[p]["clips"]), "frames": st[p]["frames"], "frames_with_boxes": st[p]["pos"],
                             "boxes": st[p]["boxes"], "true_negative_frames_(Empty-Images)": st[p]["true_neg"],
                             "unlabeled_nonempty_frames": st[p]["unlabeled_nonempty"], "clips_with_true_negatives": sorted(st[p]["neg_clips"]),
                             "mixed_negative_clips": sorted(st[p]["mixed_neg_clips"]), "frames_by_site": dict(st[p]["sites"]),
                             "frames_by_clip": st[p]["frames_by_clip"], "pos_prevalence": round(st[p]["pos"] / st[p]["frames"], 4),
                             "boxes_per_positive_frame": round(st[p]["boxes"] / max(1, st[p]["pos"]), 4)} for p in PARTS}
    out["filelist_sha256"] = {p: list_hash(v) for p, v in fl.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="docs/experiments/level2_v2_audit/boreal_scan.json.gz")
    ap.add_argument("--outdir", default="docs/experiments/boreal_v2")
    ap.add_argument("--restarts", type=int, default=N_RESTARTS)
    a = ap.parse_args(argv)
    scan = load(a.scan)
    clips = clip_table(scan)
    near = near_pairs_by_clip(clips)
    ranked = search(clips, near, restarts=a.restarts)
    assign, obj = ranked[0]
    total = sum(c["frames"] for c in clips.values())
    rep = verify(assign, clips, near)
    v1 = {p: {c: clips[c]["v1_split"].get(p, 0) for c in clips if clips[c]["v1_split"].get(p, 0)} for p in SPLITS}
    os.makedirs(a.outdir, exist_ok=True)
    fl = filelists(assign, clips)
    rec = {f["name"]: f for c in clips.values() for f in c["files"]}
    with open(os.path.join(a.outdir, "filelists.json"), "w") as f:
        json.dump({p: [{"name": n, "source_split": rec[n]["source_split"], "has_label_file": rec[n]["has_label_file"],
                        "n_boxes": rec[n]["n_boxes"], "md5": rec[n]["md5"]} for n in fl[p]] for p in PARTS}, f, indent=0, sort_keys=True)
    with open(os.path.join(a.outdir, "split_assignment.json"), "w") as f:
        json.dump({"name": "boreal_v2", "construction": {
            "unit": "normalised source clip (Empty-Images frames join their clip)", "scan_sha256_expected": "8a4db032e44c4040872d6b4863af9efefe7b2e5ac3b1f6e48b6f993a3e400e8e",
            "search": "seeded steepest-descent with random restarts over single moves and pair swaps", "seed": SEARCH_SEED, "restarts": a.restarts,
            "constraints": CONSTRAINTS, "weights": WEIGHTS, "near_dup_bits": NEAR_BITS, "target_fraction": TARGET_FRAC,
            "objective_uses_model_outputs": False,
            "design_history": ["candidate 1 (constraints without the three 'added before freezing' rules) was generated and its composition inspected; "
                               "it was rejected because Evo was absent from val and test and 59 of 61 test negatives came from one all-background clip. "
                               "The three rules were added and the search rerun. No model was trained or evaluated on candidate 1."], "objective_value": obj,
            "n_distinct_local_optima": len(ranked), "top_objectives": [round(o, 4) for _, o in ranked[:10]],
            "top_alternatives_test_clips": [sorted(c for c, p in s.items() if p == "test") for s, _ in ranked[:5]]},
            "assignment": dict(sorted(assign.items())), "v1_membership_by_clip": v1,
            "clips": {c: {k: v for k, v in d.items() if k not in ("files", "v1_split")} | {"v1_split": dict(d["v1_split"])}
                      for c, d in clips.items()}}, f, indent=1, sort_keys=True)
    with open(os.path.join(a.outdir, "split_audit.json"), "w") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    print(json.dumps({"objective": obj, "counts": rep["counts"], "video_disjoint": rep["video_disjoint"],
                      "md5_spanning": rep["md5_groups_spanning_partitions"], "filelist_sha256": rep["filelist_sha256"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
