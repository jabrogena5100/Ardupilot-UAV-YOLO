"""demo_multi.py: the four-UAV demonstration config is the existing planner's config plus display settings; the dashboard serves
only what it received. No SITL is started here."""
import argparse
import json
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

import demo_multi as DM
from dashboard import server as srv


def _args(**kw):
    d = dict(drones=4, duration=900, alt=60.0, speed=15.0, ignite="5,5", p_ignite=0.0025, burn_steps=300, fire_seed=7)
    d.update(kw)
    return argparse.Namespace(**d)


class ConfigTest(unittest.TestCase):
    def test_four_agents_quadrants_oracle_off_no_detector(self):
        cfg = DM.build_config(_args())
        self.assertEqual(len(cfg["agents"]), 4)
        self.assertEqual([a["partition"]["id"] for a in cfg["agents"]], [0, 1, 2, 3])
        self.assertTrue(all(a["partition"]["scheme"] == "quadrants" for a in cfg["agents"]))
        self.assertFalse(cfg["fire_model"]["oracle_hazards"])
        self.assertFalse(cfg["perception"]["enabled"])
        self.assertEqual(cfg["fire_model"]["start_gate"]["agents"], 4)
        self.assertEqual([a["conn"] for a in cfg["agents"]], [f"udp:127.0.0.1:{14555 + k}" for k in range(4)])

    def test_meta_regions_partition_the_grid(self):
        cfg = DM.build_config(_args())
        meta = DM.build_meta(cfg, _args())
        cells = [tuple(c) for u in meta["uavs"] for c in u["cells"]]
        self.assertEqual(len(cells), 100)
        self.assertEqual(len(set(cells)), 100)                       # disjoint, together the whole 10 x 10 grid
        self.assertEqual([u["sys"] for u in meta["uavs"]], [1, 2, 3, 4])
        self.assertEqual(len({u["color"] for u in meta["uavs"]}), 4)
        self.assertFalse(meta["perception"])
        self.assertIn("MULTI-UAV ARCHITECTURE DEMONSTRATION", meta["banner"][0])
        self.assertIn("separate from the validated single-UAV research experiments", meta["banner"][1])

    def test_frozen_fire_file_is_only_read(self):
        before = open(DM.LP.FIRE_PATH, "rb").read()
        DM.build_config(_args(p_ignite=0.01))
        self.assertEqual(before, open(DM.LP.FIRE_PATH, "rb").read())


class ServerTest(unittest.TestCase):
    def test_trails_and_demo_endpoints(self):
        m = srv.Mirror()
        t0 = time.time()
        for k in range(5):      # 5 POSE packets 0.3 s apart: only samples >= 0.5 s apart are kept
            m.ingest({"type": "POSE", "sys": 1, "lat": 21.3 + k * 1e-5, "lon": -157.8, "t": t0 + 0.3 * k})
        self.assertEqual(len(m.trails["1"]), 3)
        meta = {"uavs": [], "banner": ["x", "y"]}
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(m, meta))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            self.assertEqual(json.load(urllib.request.urlopen(base + "/api/demo")), meta)
            self.assertEqual(len(json.load(urllib.request.urlopen(base + "/api/trails"))["1"]), 3)
            self.assertEqual(urllib.request.urlopen(base + "/multi.html").status, 200)
            self.assertEqual(urllib.request.urlopen(base + "/index.html").status, 200)   # the single-UAV dashboard is untouched
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_demo_endpoint_absent_without_meta(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(srv.Mirror()))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            with self.assertRaises(urllib.error.HTTPError) as c:
                urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/api/demo")
            self.assertEqual(c.exception.code, 404)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
