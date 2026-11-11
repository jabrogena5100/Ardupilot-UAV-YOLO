#!/usr/bin/env python3
"""FireMan-UAV-RGBT (Zenodo 10.5281/zenodo.13732947, Multiclass RGB) preparation for the external evaluation.

No model is loaded and no image is read here. Everything is derived from the per-video CVAT annotation XML files.

  parse_cvat      CVAT image-export XML -> frames with their rectangles
  smoke_yolo      smoke_region rectangles -> YOLO lines (class 0), tolerant of at most 2 px overshoot, never silently correcting more
  select_frames   deterministic evenly spaced frame choice per video (documented rule below)
  build_manifest  pinned frame list (+ SHA-256 of the sorted frame names) with per-video provenance and scene cluster
  verify_manifest recompute from the XML and compare (exit 0 = identical)
  convert         write the YOLO label files for a manifest

Frame choice rule: for a video with N annotated frames sorted by frame id and a budget k, take the frame at sorted position
round(i * (N - 1) / (k - 1)) for i = 0..k-1 (duplicates removed); all frames if N <= k.

Provenance tiers (see docs/external_eval/fireman_video_metadata.json, evidence in docs/external_eval/dataset_audit.md):
  verified_2023  the paired thermal frames of this video carry an on-screen date of 2023-08-26 (cannot overlap Boreal, which was recorded in 2022)
  unverified     recording date not established; may coincide with Boreal burns (2022-07-06 Karkkila, 2022-08-15 Ruokolahti)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Tuple

SMOKE_LABEL = "smoke_region"
FIRE_LABEL = "fire_region"
CLIP_TOL_PX = 2.0
SCHEMA = "fireman_frames_v1"


def parse_cvat(source) -> List[dict]:
    """source: path or XML bytes/str. Returns frames sorted by id: {id, name, width, height, boxes: [(label, x1, y1, x2, y2)]}."""
    is_xml = isinstance(source, bytes) or (isinstance(source, str) and source.lstrip().startswith("<"))
    root = ET.fromstring(source) if is_xml else ET.parse(source).getroot()
    frames = []
    for im in root.findall("image"):
        boxes = []
        for b in im.findall("box"):
            boxes.append((b.get("label"), float(b.get("xtl")), float(b.get("ytl")), float(b.get("xbr")), float(b.get("ybr"))))
        frames.append({"id": int(im.get("id")), "name": im.get("name"), "width": int(im.get("width")), "height": int(im.get("height")), "boxes": boxes})
    frames.sort(key=lambda f: f["id"])
    return frames


def smoke_yolo(frame: dict, tol: float = CLIP_TOL_PX) -> Tuple[List[str], dict]:
    """YOLO lines `0 xc yc w h` (normalised) for the frame's smoke_region boxes, and an issue counter.
    A box that overshoots the image by <= tol px is clipped and counted (`clipped`); a larger overshoot raises ValueError, as does a
    non-positive area after clipping (`degenerate`). Nothing is dropped or fixed silently."""
    W, H = frame["width"], frame["height"]
    lines, issues = [], {"clipped": 0, "multi_box": 0}
    smoke = [b for b in frame["boxes"] if b[0] == SMOKE_LABEL]
    if len(smoke) > 1:
        issues["multi_box"] = 1
    for _, x1, y1, x2, y2 in smoke:
        if x1 > x2 or y1 > y2:
            raise ValueError(f"frame {frame['id']}: inverted box")
        over = max(-x1, -y1, x2 - W, y2 - H, 0.0)
        if over > tol:
            raise ValueError(f"frame {frame['id']}: box exceeds the image by {over:.1f} px (> {tol})")
        if over > 0:
            issues["clipped"] += 1
        x1, y1, x2, y2 = max(x1, 0.0), max(y1, 0.0), min(x2, float(W)), min(y2, float(H))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            raise ValueError(f"frame {frame['id']}: degenerate box")
        lines.append("0 %.6f %.6f %.6f %.6f" % ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
    return lines, issues


def keyframe_distances(frames: List[dict], tol: float = 0.02) -> Dict[int, Optional[int]]:
    """Distance (in frames) from each frame to the nearest slope change of the single-smoke-box coordinates, i.e. to the nearest probable CVAT keyframe.
    Between keyframes CVAT interpolates linearly, so coordinates move exactly linearly (|second difference| < tol px); a slope change marks a keyframe.
    Frames without exactly one smoke box break the series. None for every frame when the video has no slope change (one static or one linear box)."""
    fr = sorted(frames, key=lambda f: f["id"])
    pts = []
    for f in fr:
        b = [x for x in f["boxes"] if x[0] == SMOKE_LABEL]
        pts.append(b[0][1:] if len(b) == 1 else None)
    kf = []
    for i in range(1, len(fr) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        if a is None or b is None or c is None:
            continue
        if max(abs(a[k] - 2 * b[k] + c[k]) for k in range(4)) >= tol:
            kf.append(i)
    return {f["id"]: (min(abs(i - k) for k in kf) if kf else None) for i, f in enumerate(fr)}


def select_frames(frame_ids: Iterable[int], k: int) -> List[int]:
    ids = sorted(frame_ids)
    n = len(ids)
    if n <= k:
        return ids
    if k <= 1:
        return ids[:1]
    pos = sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})
    return [ids[p] for p in pos]


def video_key(path: str) -> str:
    split, vid, _ = os.path.basename(path).split("__")
    return f"{split}/{vid}"


def frame_name(video: str, frame_id: int) -> str:
    return f"{video.replace('/', '__')}__frame_{frame_id:06d}"


def names_sha256(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def build_manifest(xml_dir: str, metadata: Dict[str, dict], per_video: int = 150, tiers: Optional[Iterable[str]] = None) -> dict:
    """metadata[video] = {tier, cluster, ...} (docs/external_eval/fireman_video_metadata.json['videos']). tiers = which provenance tiers to include."""
    keep = set(tiers) if tiers else None
    out_frames, videos, issues = [], {}, {"clipped": 0, "multi_box": 0}
    for p in sorted(glob.glob(os.path.join(xml_dir, "*__annotations.xml"))):
        v = video_key(p)
        meta = metadata[v]
        if keep is not None and meta["tier"] not in keep:
            continue
        frames = {f["id"]: f for f in parse_cvat(p)}
        chosen = select_frames(frames, per_video)
        n_pos = 0
        for fid in chosen:
            f = frames[fid]
            lines, iss = smoke_yolo(f)
            for k in issues:
                issues[k] += iss[k]
            n_pos += bool(lines)
            out_frames.append([v, fid, f["width"], f["height"], len(lines)])
        videos[v] = {"tier": meta["tier"], "cluster": meta["cluster"], "n_total_frames": len(frames), "n_selected": len(chosen), "n_selected_with_smoke": n_pos}
    names = [frame_name(v, fid) for v, fid, *_ in out_frames]
    return {"schema": SCHEMA, "source": "FireMan-UAV-RGBT v1.0.0-beta.2 Multiclass.zip (md5 ddc790b901bfc51e196976145f49ea5d), RGB annotations only",
            "rule": "evenly spaced by sorted frame id: position round(i*(N-1)/(k-1)); all frames when N <= k", "per_video": per_video,
            "tiers": sorted(keep) if keep else "all", "n_videos": len(videos), "n_frames": len(out_frames), "n_frames_with_smoke": sum(1 for f in out_frames if f[4]),
            "names_sha256": names_sha256(names), "issues": issues, "videos": videos, "frames": out_frames}


def verify_manifest(manifest: dict, xml_dir: str, metadata: Dict[str, dict]) -> List[str]:
    """Rebuild with the manifest's own parameters; return a list of differences (empty = identical)."""
    tiers = None if manifest["tiers"] == "all" else manifest["tiers"]
    again = build_manifest(xml_dir, metadata, manifest["per_video"], tiers)
    problems = []
    for key in ("n_videos", "n_frames", "n_frames_with_smoke", "names_sha256", "issues"):
        if again[key] != manifest[key]:
            problems.append(f"{key}: manifest {manifest[key]!r} != recomputed {again[key]!r}")
    if again["frames"] != manifest["frames"]:
        problems.append("frame list differs")
    actual = names_sha256(frame_name(v, f) for v, f, *_ in manifest["frames"])
    if actual != manifest["names_sha256"]:
        problems.append("names_sha256 does not match the frame list stored in the manifest")
    return problems


def convert(manifest: dict, xml_dir: str, out_dir: str) -> dict:
    """Write one YOLO label file per manifest frame (empty file when the frame has no smoke box). Refuses to overwrite."""
    os.makedirs(out_dir, exist_ok=False)
    by_video: Dict[str, List[int]] = {}
    for v, fid, *_ in manifest["frames"]:
        by_video.setdefault(v, []).append(fid)
    n = pos = boxes = 0
    for v, fids in by_video.items():
        split, vid = v.split("/")
        frames = {f["id"]: f for f in parse_cvat(os.path.join(xml_dir, f"{split}__{vid}__annotations.xml"))}
        for fid in fids:
            lines, _ = smoke_yolo(frames[fid])
            with open(os.path.join(out_dir, frame_name(v, fid) + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            n += 1; pos += bool(lines); boxes += len(lines)
    return {"files": n, "files_with_smoke": pos, "smoke_boxes": boxes}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("manifest", "verify", "convert"):
        s = sub.add_parser(name)
        s.add_argument("--xml-dir", required=True, help="directory of <split>__<video>__annotations.xml files (local_data/external_eval/fireman/annotations)")
        s.add_argument("--metadata", default="docs/external_eval/fireman_video_metadata.json")
        if name == "manifest":
            s.add_argument("--per-video", type=int, default=150)
            s.add_argument("--tiers", default="", help="comma separated provenance tiers to include (default all)")
            s.add_argument("--out", required=True)
        else:
            s.add_argument("--manifest", required=True)
        if name == "convert":
            s.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    meta = json.load(open(a.metadata))["videos"]
    if a.cmd == "manifest":
        m = build_manifest(a.xml_dir, meta, a.per_video, [t for t in a.tiers.split(",") if t] or None)
        if os.path.exists(a.out):
            raise SystemExit(f"{a.out} exists; refusing to overwrite a pinned manifest")
        json.dump(m, open(a.out, "w"), indent=1)
        print(f"{m['n_videos']} videos, {m['n_frames']} frames ({m['n_frames_with_smoke']} with smoke), names sha256 {m['names_sha256']}")
    elif a.cmd == "verify":
        probs = verify_manifest(json.load(open(a.manifest)), a.xml_dir, meta)
        print("OK: manifest reproduces exactly" if not probs else "\n".join(probs))
        return 0 if not probs else 1
    else:
        print(json.dumps(convert(json.load(open(a.manifest)), a.xml_dir, a.out_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
