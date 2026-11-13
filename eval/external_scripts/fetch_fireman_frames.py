#!/usr/bin/env python3
"""Download exactly the frames of a pinned FireMan manifest from the Zenodo zip by HTTP range requests (a few MB of directory plus ~0.3 MB per frame).

    python3 eval/external_scripts/fetch_fireman_frames.py --manifest eval/fireman_frames_primary_v1.json --out local_data/external_eval/fireman/frames_primary_v1

Writes <split>__<video>__frame_NNNNNN.jpg (the names the manifest and eval/external_materialize.py use). Resumable (existing files are kept), 3 threads,
backs off on HTTP 429 (Zenodo throttles after a few thousand requests; a pod address was blocked outright, so run this from the laptop).
Never downloads the 33 GB archive. Local one-off tool: not unit-tested.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request, zlib, zipfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from rangefile import RangeFile
from eval import fireman_prep as P

URL = "https://zenodo.org/records/13732947/files/Multiclass.zip?download=1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    os.makedirs(a.out, exist_ok=True)
    todo = [(v, f) for v, f, *_ in man["frames"] if not os.path.exists(os.path.join(a.out, P.frame_name(v, f) + ".jpg"))]
    print(f"{len(man['frames'])} frames in the manifest, {len(todo)} to fetch", flush=True)
    if not todo:
        return 0
    rf = RangeFile(URL)
    z = zipfile.ZipFile(rf)
    final = rf.final
    infos = {i.filename: i for i in z.infolist() if i.filename.startswith("Multiclass/Unimodal/rgb/")}

    def fetch(item):
        v, fid = item
        info = infos[f"Multiclass/Unimodal/rgb/{v}/images/frame_{fid:06d}.jpg"]
        off = info.header_offset
        for att in range(12):
            try:
                h = urllib.request.urlopen(urllib.request.Request(final, headers={"Range": f"bytes={off}-{off + 630 + info.compress_size - 1}"}), timeout=120).read()
                nl, ml = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
                d = h[30 + nl + ml:30 + nl + ml + info.compress_size]
                raw = zlib.decompress(d, -15) if info.compress_type == 8 else d
                tmp = os.path.join(a.out, P.frame_name(v, fid) + ".jpg.part")
                open(tmp, "wb").write(raw)
                os.replace(tmp, os.path.join(a.out, P.frame_name(v, fid) + ".jpg"))
                return True
            except urllib.error.HTTPError as e:
                time.sleep(20 * (att + 1) if e.code == 429 else 2 * (att + 1))
            except Exception:
                time.sleep(2 * (att + 1))
        print("FAILED", v, fid, flush=True)
        return False

    with ThreadPoolExecutor(a.threads) as ex:
        ok = sum(ex.map(fetch, todo))
    print(f"fetched {ok} of {len(todo)}")
    return 0 if ok == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
