#!/usr/bin/env python3
"""Cloud/laptop side: download the files advertised by results_server.py, verify every SHA-256, then acknowledge.

    RESULTS_TOKEN_FILE=/path/to/token python3 fetch_results.py --base https://<pod_id>-8000.proxy.runpod.net --dest DIR [--no-ack]

DIR must not exist (nothing is overwritten). The token is read from the file/env and never printed. Exit 0 = all files verified.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

UA = "wildfire-fetch/1"


def get(base, path, token, timeout=60, method="GET"):
    req = urllib.request.Request(base.rstrip("/") + path, method=method, data=b"" if method == "POST" else None,
                                 headers={"Authorization": "Bearer " + token, "User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch(base, dest, token, ack=True, retries=3, sleep=time.sleep):
    if os.path.exists(dest):
        raise FileExistsError(dest)
    manifest = None
    for i in range(retries):
        try:
            manifest = json.load(get(base, "/manifest", token))["files"]
            break
        except (urllib.error.URLError, OSError) as e:
            err = type(e).__name__ + (f" {e.code}" if hasattr(e, "code") else "")
            sleep(2 ** i)
    if manifest is None:
        raise ConnectionError("manifest unreachable: " + err)
    tmp = dest + ".partial"
    os.makedirs(tmp)
    for name, ent in sorted(manifest.items()):
        h = hashlib.sha256()
        with get(base, "/f/" + name, token, timeout=300) as r, open(os.path.join(tmp, name), "wb") as f:
            for b in iter(lambda: r.read(1 << 20), b""):
                h.update(b)
                f.write(b)
        if h.hexdigest() != ent["sha256"] or os.path.getsize(os.path.join(tmp, name)) != ent["bytes"]:
            raise ValueError("hash/size mismatch for " + name)
    os.rename(tmp, dest)
    if ack:
        get(base, "/ack", token, method="POST").read()
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--no-ack", action="store_true")
    a = ap.parse_args(argv)
    tok = os.environ.get("RESULTS_TOKEN") or open(os.environ["RESULTS_TOKEN_FILE"]).read().strip()
    try:
        m = fetch(a.base, a.dest, tok, ack=not a.no_ack)
    except Exception as e:   # no URL parts or tokens in the message
        print("FETCH FAILED:", type(e).__name__, str(e)[:120].replace(tok, "***"), file=sys.stderr)
        return 2
    print("fetched and verified", len(m), "files into", a.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
