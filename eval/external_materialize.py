#!/usr/bin/env python3
"""Build the harness views (images/test + labels/test) that eval/datasets_external.yaml points at. Files are symlinked, never copied or edited.

  fasdd     <fasdd_uav_test_v1>/images + labels_smoke_only restricted to eval/fasdd_uav_test_smoke_positive_v1.txt
  fireman   frames fetched for a pinned manifest (eval/fireman_frames_*_v1.json) + labels converted from the CVAT XML by eval/fireman_prep.py

Both refuse to overwrite an existing view, stop on a missing file, and print the file-list pin (sha256 of the sorted image names joined by newlines,
the definition used by eval/run_matrix.py) so it can be compared with the yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import List

try:
    from . import fireman_prep as P          # python3 -m eval.external_materialize / imported by tests
except ImportError:                         # python3 eval/external_materialize.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fireman_prep as P


def filelist_sha256(names: List[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def _link(src: str, dst: str) -> None:
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    os.symlink(os.path.abspath(src), dst)


def fasdd_view(src_root: str, list_file: str, out_root: str) -> dict:
    names = [x.strip() for x in open(list_file) if x.strip()]
    if len(names) != len(set(names)):
        raise ValueError("duplicate names in the image list")
    img_dir, lab_dir = os.path.join(out_root, "images", "test"), os.path.join(out_root, "labels", "test")
    os.makedirs(img_dir, exist_ok=False)
    os.makedirs(lab_dir, exist_ok=False)
    for n in names:
        stem = os.path.splitext(n)[0]
        _link(os.path.join(src_root, "images", n), os.path.join(img_dir, n))
        _link(os.path.join(src_root, "labels_smoke_only", stem + ".txt"), os.path.join(lab_dir, stem + ".txt"))
    empty = [n for n in names if os.path.getsize(os.path.join(lab_dir, os.path.splitext(n)[0] + ".txt")) == 0]
    if empty:
        raise ValueError(f"{len(empty)} listed images have an empty smoke label (first: {empty[0]}); the list must be smoke-positive only")
    return {"n_images": len(names), "filelist_sha256": filelist_sha256(names)}


def fireman_view(manifest_path: str, frames_dir: str, xml_dir: str, out_root: str) -> dict:
    manifest = json.load(open(manifest_path))
    names = [P.frame_name(v, f) + ".jpg" for v, f, *_ in manifest["frames"]]
    missing = [n for n in names if not os.path.exists(os.path.join(frames_dir, n))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frames of the manifest are not in {frames_dir} (first: {missing[0]})")
    img_dir, lab_dir = os.path.join(out_root, "images", "test"), os.path.join(out_root, "labels", "test")
    os.makedirs(img_dir, exist_ok=False)
    for n in names:
        _link(os.path.join(frames_dir, n), os.path.join(img_dir, n))
    conv = P.convert(manifest, xml_dir, lab_dir)          # creates labels/test, refuses to overwrite
    return {"n_images": len(names), "filelist_sha256": filelist_sha256(names), "labels": conv}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fasdd")
    f.add_argument("--src", required=True, help="fasdd_uav_test_v1 directory (images/, labels_smoke_only/)")
    f.add_argument("--list", default="eval/fasdd_uav_test_smoke_positive_v1.txt")
    f.add_argument("--out", required=True)
    m = sub.add_parser("fireman")
    m.add_argument("--manifest", required=True)
    m.add_argument("--frames-dir", required=True, help="directory with <split>__<video>__frame_NNNNNN.jpg files")
    m.add_argument("--xml-dir", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = fasdd_view(a.src, a.list, a.out) if a.cmd == "fasdd" else fireman_view(a.manifest, a.frames_dir, a.xml_dir, a.out)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
