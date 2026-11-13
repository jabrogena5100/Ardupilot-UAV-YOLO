#!/usr/bin/env python3
"""
dashboard/replay_export.py — write ONE self-contained HTML file of the replay dashboard.

    python3 dashboard/replay_export.py --pilot final --out replay_final_standalone.html

Inlines style.css, replay.css, replay.js and the saved run data (same JSON the server
would return). No network, no localhost, no API. Read-only on results; the trajectory
cache must already exist (dashboard/replay_build.py). Nothing is added or interpolated.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from dashboard import replay_data as rd  # noqa: E402

ST = os.path.join(HERE, "static")


def _r(n):
    with open(os.path.join(ST, n)) as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="append", default=None, help="repeatable; default final")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pilots = a.pilot or ["final"]

    L = rd.discover()
    L["runs"] = [r for r in L["runs"] if r["pilot"] in pilots]
    L["planned"] = [q for q in L["planned"] if q["pilot"] in pilots]
    L["pilots"] = {k: v for k, v in L["pilots"].items() if k in pilots}
    L["discarded"] = []  # refer to pilot 1 attempts, not part of this export
    runs = {r["id"]: rd.load_run(r["id"]) for r in L["runs"]}
    payload = {"runs": L, "run": runs}
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")

    html = _r("replay.html")
    html = html.replace('<link rel="stylesheet" href="style.css" />', "<style>\n" + _r("style.css") + "\n</style>")
    html = html.replace('<link rel="stylesheet" href="replay.css" />', "<style>\n" + _r("replay.css") + "\n</style>")
    html = html.replace('<script src="replay.js"></script>',
                        "<script>window.__REPLAY_EMBEDDED__ = " + blob + ";</script>\n<script>\n" + _r("replay.js") + "\n</script>")
    html = html.replace("<small>recorded ArduPilot SITL runs</small>",
                        "<small>recorded ArduPilot SITL runs · standalone snapshot of " + ", ".join(pilots) + "</small>")
    for needle in ("window.__REPLAY_EMBEDDED__", "standalone snapshot"):  # fail loudly if the template changed under us
        assert needle in html, f"export template no longer contains {needle!r}"
    with open(a.out, "w") as f:
        f.write(html)
    print(f"wrote {a.out}: {os.path.getsize(a.out)} bytes, {len(runs)} runs, {len(L['planned'])} planned")


if __name__ == "__main__":
    main()
