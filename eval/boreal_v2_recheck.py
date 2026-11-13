"""Independent recheck of the frozen Boreal v2 split from the committed scan (methodological-limitation inventory).

Reports (a) the exact cross-partition near-duplicate frame pairs, (b) same-site clips with consecutive DJI numbers that sit in
different partitions (possible consecutive segments of one recording: UNPROVEN either way), with the number of cross-clip frame pairs
at <=4 and <=8 dHash bits. These are documented limitations, not proof of independence. Usage:
  python3 -m eval.boreal_v2_recheck [--out docs/experiments/boreal_v2/recheck.json]
"""
import argparse
import collections
import gzip
import json
import re

import numpy as np

from eval import boreal_v2_split as B
from eval.boreal_split_audit import parse

D = "docs/experiments/boreal_v2"


def recheck(scan_path="docs/experiments/level2_v2_audit/boreal_scan.json.gz", asg_path=D + "/split_assignment.json"):
    scan = json.load(gzip.open(scan_path))
    asg = json.load(open(asg_path))["assignment"]
    clips = B.clip_table(scan)
    rec = {f["name"]: (c, f) for c, d in clips.items() for f in d["files"]}
    names = [f["name"] for c in clips.values() for f in c["files"] if f["dhash"] and not f["dhash"].startswith("ERR")]
    h = np.array([int(rec[n][1]["dhash"], 16) for n in names], dtype=np.uint64)
    part = np.array([asg[rec[n][0]] for n in names])
    clip = np.array([rec[n][0] for n in names])
    pairs, by8 = [], collections.Counter()
    for i in range(0, len(h), 256):
        d = B.popcount64(h[i:i + 256, None] ^ h[None, :])
        for a in range(d.shape[0]):
            ia = i + a
            for j in np.where((d[a] <= 8) & (part != part[ia]))[0]:
                if j > ia:
                    k = tuple(sorted((str(clip[ia]), str(clip[j]))))
                    by8[k] += 1
                    if d[a][j] <= 4:
                        pairs.append({"hamming": int(d[a][j]), "a": names[ia], "a_partition": str(part[ia]), "b": names[j], "b_partition": str(part[j])})
    # consecutive-number neighbours (same site, DJI number differs by 1) assigned to different partitions
    num = {c: (clips[c]["site"], int(re.search(r"(\d+)$", c).group(1))) for c in clips}
    adj = []
    for a in sorted(clips):
        for b in sorted(clips):
            if a < b and num[a][0] == num[b][0] and abs(num[a][1] - num[b][1]) == 1 and asg[a] != asg[b]:
                adj.append({"clips": [a, b], "partitions": [asg[a], asg[b]], "cross_partition_pairs_le8": by8.get(tuple(sorted((a, b))), 0)})
    return {"near_pairs_le4_cross_partition": sorted(pairs, key=lambda x: (x["hamming"], x["a"])),
            "n_pairs_le4": len(pairs), "test_frames_with_train_or_val_neighbour_le4": sorted({p["a"] if p["a_partition"] == "test" else p["b"]
                                                                                   for p in pairs if "test" in (p["a_partition"], p["b_partition"])}),
            "cross_partition_clip_pairs_le8": {"|".join(k): v for k, v in sorted(by8.items(), key=lambda kv: -kv[1])},
            "consecutive_number_same_site_clips_in_different_partitions": adj,
            "interpretation": "dHash (9x8 grayscale) flags candidates only; consecutive DJI numbers may be segments of one flight. "
                              "Independence of these clips is NOT established; treat as documented limitations."}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=D + "/recheck.json")
    a = ap.parse_args()
    r = recheck()
    json.dump(r, open(a.out, "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, (list, dict)) or len(v) < 12 else f"[{len(v)} items]") for k, v in r.items() if k != "interpretation"}, indent=1))
