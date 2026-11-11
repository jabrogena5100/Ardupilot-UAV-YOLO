import json, os, time
WS = os.environ.get("WS", "/workspace")
ZIP = os.path.join(WS, "datasets", "FASDD_UAV.zip")
ZIP_BYTES = int(os.environ.get("FZB", "14888184890"))
EXT = os.path.join(WS, "external_eval", "fasdd_uav_test_v1")
NEXP = int(os.environ.get("FN", "4181"))
RUNS = os.path.join(WS, "runs", "volume_cleanup")
T0 = time.time()
def emit(k, **kw): print(json.dumps({"stage": k, "t": round(time.time() - T0, 1), **kw}, default=str), flush=True)
def du(path):
    tot = 0
    for dp, dn, fs in os.walk(path, followlinks=False):
        for f in fs:
            p = os.path.join(dp, f)
            if not os.path.islink(p):
                try: tot += os.lstat(p).st_size
                except OSError: pass
    return tot
def report(tag):
    top = {}
    for n in sorted(os.listdir(WS)):
        p = os.path.join(WS, n)
        if os.path.islink(p): top[n] = "symlink"
        elif os.path.isfile(p): top[n] = os.path.getsize(p)
        else: top[n] = du(p)
    num = {k: v for k, v in top.items() if isinstance(v, int)}
    emit("usage_" + tag, total_gb=round(sum(num.values()) / 1e9, 2), top_level_gb={k: round(v / 1e9, 3) for k, v in sorted(num.items(), key=lambda kv: -kv[1])}, symlinks=[k for k, v in top.items() if v == "symlink"])
    sub = {}
    for n in ("datasets", "runs", "external_eval", "boreal"):
        p = os.path.join(WS, n)
        if os.path.isdir(p) and not os.path.islink(p):
            sub[n] = {c: round(du(os.path.join(p, c)) / 1e9, 3) if os.path.isdir(os.path.join(p, c)) and not os.path.islink(os.path.join(p, c)) else round(os.path.getsize(os.path.join(p, c)) / 1e9, 3) for c in sorted(os.listdir(p))}
    emit("usage_detail_" + tag, **{k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:25]) for k, v in sub.items()})
def main():
    os.makedirs(RUNS, exist_ok=True)
    try:
        fd = os.open(os.path.join(RUNS, "STARTED"), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(time.time()).encode()); os.close(fd)
    except FileExistsError:
        print("GUARD: already started; exiting", flush=True); return
    report("before")
    ok = {}
    ok["sealed_marker"] = os.path.isfile(os.path.join(EXT, "EXTRACTION_COMPLETE.json"))
    for d in ("images", "labels_original", "labels_smoke_only", "annotations_voc"):
        ok["n_" + d] = len(os.listdir(os.path.join(EXT, d))) if os.path.isdir(os.path.join(EXT, d)) else -1
    h = os.path.join(WS, "runs", "fasdd_extract", "fasdd_test_hashes.json")
    ok["hashes_entries"] = len(json.load(open(h))) if os.path.isfile(h) else -1
    ok["screening_v1"] = os.path.isdir(os.path.join(WS, "external_eval", "screening_v1")); ok["screening_v2"] = os.path.isdir(os.path.join(WS, "external_eval", "screening_v2"))
    ok["zip_is_regular_file"] = os.path.isfile(ZIP) and not os.path.islink(ZIP) and os.path.realpath(ZIP) == os.path.join(os.path.realpath(WS), "datasets", "FASDD_UAV.zip")
    ok["zip_size_matches"] = ok["zip_is_regular_file"] and os.path.getsize(ZIP) == ZIP_BYTES
    safe = ok["sealed_marker"] and all(ok["n_" + d] == NEXP for d in ("images", "labels_original", "labels_smoke_only", "annotations_voc")) and ok["hashes_entries"] == NEXP and ok["screening_v1"] and ok["screening_v2"] and ok["zip_size_matches"]
    emit("safety_checks", **ok, safe_to_delete=safe)
    if safe:
        os.remove(ZIP)
        emit("deleted", path=ZIP, bytes=ZIP_BYTES, exists_after=os.path.exists(ZIP), datasets_dir=sorted(os.listdir(os.path.join(WS, "datasets"))))
        report("after")
    else:
        emit("NOT_DELETED", reason="a safety check failed; nothing was changed")
if __name__ == "__main__": main()
