"""Boreal split integrity audit from a saved scan (name, md5, size, dhash, label count per image).

Input: `boreal_scan.json.gz` = {split: {image_name: [md5, bytes, [w, h], dhash64_hex, n_label_lines(-1 = no label file)]}}.
It is produced on the pod by a read-only scan of /workspace/boreal/boreal_yolo (see docs/experiments/boreal_split_audit.md).
Video identity is normalised: the site prefix (`Evo-Images__`, `Empty-Images__`, ...) is dropped and the `_frameN`
suffix removed, then lower-cased, so `Empty-Images__karkkila_DJI_0003_frame24` and `Karkkila-Images__karkkila_DJI_0003_frame0`
map to the same source clip `karkkila_dji_0003`.
Usage: python3 -m eval.boreal_split_audit SCAN.json.gz [--out report.json]
"""
import argparse
import collections
import gzip
import itertools
import json
import re

FRAME = re.compile(r"^(?:.*?__)?(?P<vid>.*?)_?frame(?P<f>\d+)\.[A-Za-z]+$")
SPLITS = ("train", "val", "test")


def parse(name):
    m = FRAME.match(name)
    if not m:
        raise ValueError("unparseable image name: " + name)
    return m.group("vid").lower(), int(m.group("f"))


def site_of(name):
    return name.split("__")[0]


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def audit(scan, near_bits=4):
    rep = {"counts": {s: len(scan[s]) for s in SPLITS}}
    vids = {s: collections.defaultdict(list) for s in SPLITS}
    raw_vids = {s: set() for s in SPLITS}
    for s in SPLITS:
        for n in scan[s]:
            v, f = parse(n)
            vids[s][v].append((f, n))
            raw_vids[s].add(re.sub(r"_?frame\d+$", "", n.rsplit(".", 1)[0]))
    rep["videos_normalised"] = {s: len(vids[s]) for s in SPLITS}
    rep["videos_prefixed_original_definition"] = {s: len(raw_vids[s]) for s in SPLITS}
    rep["prefixed_pairwise_overlap"] = {f"{a}&{b}": len(raw_vids[a] & raw_vids[b]) for a, b in itertools.combinations(SPLITS, 2)}
    rep["normalised_pairwise_overlap"] = {}
    rep["overlap_detail"] = {}
    for a, b in itertools.combinations(SPLITS, 2):
        both = sorted(set(vids[a]) & set(vids[b]))
        rep["normalised_pairwise_overlap"][f"{a}&{b}"] = len(both)
        det = []
        for v in both:
            fa = sorted(f for f, _ in vids[a][v]); fb = sorted(f for f, _ in vids[b][v])
            sa = collections.Counter(site_of(n) for _, n in vids[a][v]); sb = collections.Counter(site_of(n) for _, n in vids[b][v])
            gap = min(abs(x - y) for x in fa for y in fb)
            det.append({"video": v, a + "_frames": len(fa), b + "_frames": len(fb), a + "_sites": dict(sa), b + "_sites": dict(sb),
                        a + "_range": [fa[0], fa[-1]], b + "_range": [fb[0], fb[-1]], "min_frame_gap": gap,
                        "n_" + a + "_with_label_file": sum(1 for _, n in vids[a][v] if scan[a][n][4] >= 0),
                        "n_" + b + "_with_label_file": sum(1 for _, n in vids[b][v] if scan[b][n][4] >= 0)})
        rep["overlap_detail"][f"{a}&{b}"] = det
    # exact content duplicates (md5) within and across splits
    by_md5 = collections.defaultdict(list)
    for s in SPLITS:
        for n, r in scan[s].items():
            by_md5[r[0]].append((s, n))
    dup = [g for g in by_md5.values() if len(g) > 1]
    rep["md5_duplicate_groups"] = len(dup)
    rep["md5_cross_split_groups"] = sum(1 for g in dup if len({s for s, _ in g}) > 1)
    rep["md5_cross_split_examples"] = [g for g in dup if len({s for s, _ in g}) > 1][:10]
    # near duplicates by 64-bit dHash across splits (threshold near_bits). Frames of one clip a few frames apart can be similar by design.
    ent = [(s, n, r[3]) for s in SPLITS for n, r in scan[s].items() if r[3] and not r[3].startswith("ERR")]
    rep["dhash_errors"] = sum(1 for s in SPLITS for r in scan[s].values() if not r[3] or r[3].startswith("ERR"))
    exact_hash = collections.defaultdict(list)
    for s, n, h in ent:
        exact_hash[h].append((s, n))
    rep["dhash_identical_cross_split_groups"] = sum(1 for g in exact_hash.values() if len({s for s, _ in g}) > 1)
    near = collections.Counter(); ex = []
    for (sa, na, ha), (sb, nb, hb) in itertools.combinations(ent, 2):
        if sa != sb and hamming(ha, hb) <= near_bits:
            near[f"{sa}&{sb}"] += 1
            same = parse(na)[0] == parse(nb)[0]
            if len(ex) < 25:
                ex.append([sa, na, sb, nb, hamming(ha, hb), same])
    rep["dhash_near_pairs_cross_split"] = dict(near); rep["dhash_near_threshold_bits"] = near_bits
    rep["dhash_near_examples"] = ex
    # near pairs cross-split broken down by whether both frames are from the same normalised video
    cross_same = collections.Counter(); cross_diff = collections.Counter()
    for (sa, na, ha), (sb, nb, hb) in itertools.combinations(ent, 2):
        if sa != sb and hamming(ha, hb) <= near_bits:
            (cross_same if parse(na)[0] == parse(nb)[0] else cross_diff)[f"{sa}&{sb}"] += 1
    rep["dhash_near_pairs_same_video"] = dict(cross_same); rep["dhash_near_pairs_different_video"] = dict(cross_diff)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scan"); ap.add_argument("--out"); ap.add_argument("--near-bits", type=int, default=4)
    a = ap.parse_args()
    rep = audit(json.load(gzip.open(a.scan)), a.near_bits)
    s = json.dumps(rep, indent=2, sort_keys=True)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
