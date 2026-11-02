#!/usr/bin/env python3
"""
eval/bench_perception_cpu.py — CPU latency and memory of the perception path on the machine it runs on.

One model per process (so peak RSS is attributable). Measures: model load time, render time per frame (the virtual
camera), YOLO predict time per frame (imgsz 640, conf 0.25 floor, CPU, threads=N), peak RSS, and detections at the
provisional 0.25 threshold on frames rendered over a burning cell and over empty ground. It is a RESOURCE benchmark:
detection counts here are a smoke test of the wiring, not a perception result and not evidence about missions.

    python3 eval/bench_perception_cpu.py --weights local_data/backups/boreal_v2/real_boreal_v2_best_seed0.pt --tag B2s0
"""
import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frames", type=int, default=8, help="frames over fire (an equal number are empty)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--alt", type=float, default=120.0)
    ap.add_argument("--pitch", type=float, default=90.0)
    ap.add_argument("--scenario-seed", type=int, default=1)
    a = ap.parse_args()

    import numpy as np
    from mission.geo import ne_to_latlon
    from perception import detector as D
    from sim.camera import GridSpec
    from sim.frame_synth import CameraSpec, FrameSynth, PlumeModel, sources_from_fire_cells

    lat0, lon0 = 21.2970, -157.8170
    t = time.perf_counter()
    sha = D.verify_weights(a.weights, None)
    pred = D.make_ultralytics_predictor(a.weights, 0.25, threads=a.threads)
    pred(np.zeros((720, 1280, 3), np.uint8))                      # warm-up, not timed
    load_s = time.perf_counter() - t
    rss_model = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    g = GridSpec.from_miles(1.0, 10)
    synth = FrameSynth(a.scenario_seed, lat0, lon0, camera=CameraSpec(pitch_deg=a.pitch), plume=PlumeModel())
    src = sources_from_fire_cells([(4, 5, 1.0), (5, 5, 0.8)], g, PlumeModel(), a.scenario_seed)
    rng = np.random.default_rng(0)
    rows = []
    synth.render(lat0, lon0, a.alt, 0.0, [], "warm")
    for k in range(2 * a.frames):
        over_fire = k < a.frames
        base = src[k % len(src)]
        n, e = (base.n + rng.uniform(-25, 25), base.e + rng.uniform(-25, 25)) if over_fire else (-300.0 + 10 * k, -300.0)
        lat, lon = ne_to_latlon(n, e, lat0, lon0)
        t0 = time.perf_counter()
        frame = synth.render(lat, lon, a.alt, float(rng.uniform(0, 360)), src, k)
        t_r = time.perf_counter() - t0
        t0 = time.perf_counter()
        boxes = pred(frame)
        t_i = time.perf_counter() - t0
        rows.append({"over_fire": over_fire, "render_ms": t_r * 1e3, "infer_ms": t_i * 1e3, "n_boxes": len(boxes),
                     "max_conf": max([float(c) for _, c in boxes], default=0.0)})
    inf = [r["infer_ms"] for r in rows]
    fire = [r for r in rows if r["over_fire"]]
    emp = [r for r in rows if not r["over_fire"]]
    out = {
        "tag": a.tag, "weights": a.weights, "sha256": sha, "threads": a.threads, "alt_m": a.alt, "pitch_deg": a.pitch,
        "load_plus_warmup_s": round(load_s, 2), "infer_ms_median": round(statistics.median(inf), 1),
        "infer_ms_max": round(max(inf), 1), "render_ms_median": round(statistics.median(r["render_ms"] for r in rows), 1),
        "peak_rss_mb_after_model": round(rss_model, 0),
        "peak_rss_mb_total": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 0),
        "frames_over_fire": len(fire), "frames_with_box_over_fire": sum(1 for r in fire if r["n_boxes"] > 0),
        "frames_empty": len(emp), "frames_with_box_empty": sum(1 for r in emp if r["n_boxes"] > 0),
        "env": {"python": platform.python_version(), "cpus": os.cpu_count(), "machine": platform.machine()},
    }
    try:
        import torch, ultralytics
        out["env"].update(torch=torch.__version__, ultralytics=ultralytics.__version__)
    except Exception:
        pass
    print(json.dumps(out))


if __name__ == "__main__":
    main()
