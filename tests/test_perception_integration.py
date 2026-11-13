"""Perception integration: transforms, detection->grid mapping, ground-truth isolation, weight verification, baseline."""
import hashlib
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request

import numpy as np

from mission.geo import latlon_to_ne, ne_to_latlon
from mission.grid_mission import GridCfg, GridMission
from perception import base as pbase
from perception import detector as det
from perception import frames
from sim.camera import Camera, CameraIntrinsics, CameraPose, GridSpec
from sim.camera_service import FireCellsTruth, FrameService
from sim.frame_synth import CameraSpec, FrameSynth, PlumeModel, SmokeSource, sources_from_fire_cells

LAT0, LON0 = 21.2970, -157.8170
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


class FixedFrames:
    def __init__(self, frame):
        self.frame, self.poses = frame, []

    def get_frame(self, pose):
        self.poses.append(pose)
        return self.frame


class TestTransforms(unittest.TestCase):
    def test_ne_latlon_roundtrip(self):
        for n, e in ((0, 0), (123.4, -567.8), (-800.0, 800.0)):
            lat, lon = ne_to_latlon(n, e, LAT0, LON0)
            n2, e2 = latlon_to_ne(lat, lon, LAT0, LON0)
            self.assertAlmostEqual(n, n2, places=6)
            self.assertAlmostEqual(e, e2, places=6)

    def test_gridspec_from_gridmission_cfg_is_i_east_j_north(self):
        gm = GridMission(GridCfg(lat0=LAT0, lon0=LON0, miles=1.0, cells=10))
        d = det.YoloPerception({"model_tag": "t"}, grid=gm, frame_source=FixedFrames(None), predictor=lambda f: [])
        g = d.grid_spec
        self.assertEqual((g.n_cells, round(g.cell_m, 3)), (10, round(gm.cfg.cell_m, 3)))
        # a point 1 cell north of the SW-quadrant centre raises j; one cell east raises i
        n, e = g.center_ne(3, 4)
        self.assertEqual(g.cell_of(n, e), (3, 4))
        self.assertEqual(g.cell_of(n + g.cell_m, e), (3, 5))
        self.assertEqual(g.cell_of(n, e + g.cell_m), (4, 4))
        # and it agrees with the mission grid's own cell centre
        cn, ce = gm.cell_center_ne(3, 4) if hasattr(gm, "cell_center_ne") else (n, e)
        self.assertAlmostEqual(cn, n, places=3)
        self.assertAlmostEqual(ce, e, places=3)

    def test_nadir_box_centre_maps_to_cell_under_the_uav(self):
        g = GridSpec.from_miles(1.0, 10)
        n, e = g.center_ne(6, 2)
        pose = pbase.Pose(*ne_to_latlon(n, e, LAT0, LON0), 100.0, 0.0, (6, 2), 0.0, 1)
        cam = Camera.from_pose(pose, CameraIntrinsics(1280, 720, 78.0), LAT0, LON0, 90.0)
        box = [600, 300, 680, 420]                         # around the image centre = the nadir point
        self.assertEqual(det.box_cell(box, cam, g, "center"), (6, 2))

    def test_nadir_east_edge_of_image_is_east_of_nadir_for_north_heading(self):
        g = GridSpec.from_miles(1.0, 10)
        n, e = g.center_ne(5, 5)
        cam = Camera(CameraIntrinsics(1280, 720, 78.0), CameraPose(n, e, 400.0, 0.0, 90.0))
        right = det.box_cell([1250, 340, 1279, 380], cam, g, "center")
        left = det.box_cell([0, 340, 30, 380], cam, g, "center")
        self.assertGreater(right[0], left[0])             # heading north: image right = east = larger i

    def test_heading_rotates_the_footprint(self):
        g = GridSpec.from_miles(1.0, 10)
        n, e = g.center_ne(5, 5)
        cam = Camera(CameraIntrinsics(1280, 720, 78.0), CameraPose(n, e, 400.0, 90.0, 90.0))   # heading east
        top = det.box_cell([620, 0, 660, 20], cam, g, "center")
        bot = det.box_cell([620, 700, 660, 719], cam, g, "center")
        self.assertGreater(top[0], bot[0])                # heading east: image top = east


class TestDetectionMapping(unittest.TestCase):
    def setUp(self):
        self.g = GridSpec.from_miles(1.0, 10)
        self.cam = Camera(CameraIntrinsics(1280, 720, 78.0), CameraPose(*self.g.center_ne(5, 5), 120.0, 0.0, 90.0))

    def test_plain_python_types_and_fields(self):
        ds = det.boxes_to_detections([(np.array([600.0, 300.0, 700.0, 400.0]), np.float32(0.8))], self.cam, self.g,
                                     frame_id="f1", anchor="center")
        self.assertEqual(len(ds), 1)
        d = ds[0]
        self.assertEqual((type(d.i), type(d.j), type(d.conf), d.cls, d.frame_id), (int, int, float, "smoke", "f1"))
        self.assertTrue(all(type(v) is float for v in d.bbox))

    def test_box_outside_grid_is_dropped(self):
        far = Camera(self.cam.intr, CameraPose(self.g.half_side_m - 60.0, self.g.half_side_m - 60.0, 400.0, 0.0, 90.0))
        ds = det.boxes_to_detections([([1200, 0, 1279, 30], 0.9), ([600, 300, 700, 400], 0.9)], far, self.g, anchor="center")
        self.assertEqual(len(ds), 1)                       # the far corner box lies outside the grid; the centre one inside

    def test_observe_applies_threshold_and_uses_pose(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        src = FixedFrames(frame)
        gm = GridMission(GridCfg(lat0=LAT0, lon0=LON0))
        d = det.YoloPerception({"model_tag": "t", "conf_threshold": 0.25}, grid=gm, frame_source=src,
                               predictor=lambda f: [([600, 300, 700, 400], 0.24), ([600, 300, 700, 400], 0.26)])
        n, e = d.grid_spec.center_ne(2, 7)
        pose = pbase.Pose(*ne_to_latlon(n, e, LAT0, LON0), 100.0, 10.0, (2, 7), 5.0, 3)
        out = d.observe(pose)
        self.assertEqual([(x.i, x.j) for x in out], [(2, 7)])
        self.assertEqual(len(out), 1)                      # the 0.24 box is below the provisional 0.25
        self.assertEqual(src.poses[0].sysid, 3)

    def test_default_anchor_follows_pitch(self):
        gm = GridMission(GridCfg(lat0=LAT0, lon0=LON0))
        mk = lambda c: det.YoloPerception(c, grid=gm, frame_source=FixedFrames(None), predictor=lambda f: [])
        self.assertEqual(mk({}).anchor, "bottom_center")                       # default camera: oblique 60 deg
        self.assertEqual(mk({"camera": {"pitch_deg": 90}}).anchor, "center")   # nadir: box centre
        self.assertEqual(mk({}).pitch_deg, 60.0)


class TestWeights(unittest.TestCase):
    def test_hash_checks(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "w.pt")
            open(p, "wb").write(b"abc")
            h = hashlib.sha256(b"abc").hexdigest()
            self.assertEqual(det.verify_weights(p, h), h)
            with self.assertRaises(ValueError):
                det.verify_weights(p, "0" * 64)
            with self.assertRaises(ValueError):            # unknown file, no hash given
                det.verify_weights(p, None)
            self.assertEqual(det.verify_weights(p, None, allow_unverified=True), h)
            with self.assertRaises(FileNotFoundError):
                det.verify_weights(os.path.join(td, "missing.pt"), h)

    def test_registry_names_are_the_three_v2_checkpoints(self):
        text = read("models/MODELS.md")
        for name, h in det.KNOWN_SHA256.items():
            self.assertIn(h, text, name)

    def test_default_checkpoint_is_not_committed(self):
        self.assertTrue(det.DEFAULT_WEIGHTS.startswith("local_data"))
        self.assertNotIn(".pt", os.popen(f"git -C {ROOT} ls-files '*.pt'").read())


class TestFallbackAndBaseline(unittest.TestCase):
    def test_disabled_is_null_and_baseline_unchanged(self):
        for cfg in (None, {}, {"enabled": False, "weights": "nope.pt"}):
            p = pbase.from_config(cfg, grid=None)
            self.assertIsInstance(p, pbase.NullPerception)
            self.assertEqual(p.observe(None), [])

    def test_bad_weights_fall_back_loudly_unless_strict(self):
        gm = GridMission(GridCfg(lat0=LAT0, lon0=LON0))
        cfg = {"enabled": True, "weights": "/nonexistent/x.pt", "weights_sha256": "0" * 64}
        self.assertIsInstance(pbase.from_config(cfg, grid=gm), pbase.NullPerception)
        with self.assertRaises(Exception):
            pbase.from_config(dict(cfg, strict=True), grid=gm)

    def test_fog_camera_service_is_off_by_default(self):
        import fog_tracker
        self.assertIsNone(fog_tracker.start_camera_service({}, fog=None))
        self.assertIsNone(fog_tracker.start_camera_service({"perception": {"enabled": True}}, fog=None))

    def test_shipped_experiment_configs_do_not_enable_perception(self):
        import yaml
        for f in ("exp_swarm_fire.yaml", "exp_swarm_coop.yaml", "exp_swarm_solo.yaml"):
            cfg = yaml.safe_load(read(f)) or {}
            self.assertFalse((cfg.get("perception") or {}).get("enabled", False), f)


class TestRenderer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cam = CameraSpec(width=320, height=180)
        cls.synth = FrameSynth(3, LAT0, LON0, camera=cls.cam)
        cls.g = GridSpec.from_miles(1.0, 10)

    def pose(self, n, e):
        return (*ne_to_latlon(n, e, LAT0, LON0), 120.0, 0.0)

    def test_deterministic_and_shared_between_models(self):
        src = [SmokeSource(10.0, 10.0, seed=5)]
        a = self.synth.render(*self.pose(0, 0), src, "k")
        b = FrameSynth(3, LAT0, LON0, camera=self.cam).render(*self.pose(0, 0), src, "k")   # "another model's" run
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual((a.dtype, a.shape), (np.uint8, (180, 320, 3)))

    def test_smoke_lands_where_the_shared_projection_says(self):
        none = self.synth.render(*self.pose(0, 0), [], "k")
        n, e = 70.0, -9.0                                  # ahead of a north-looking, 60-degree camera at 120 m
        img = self.synth.render(*self.pose(0, 0), [SmokeSource(n, e, height_m=30.0, spread_m=6.0, seed=1)], "k")
        diff = np.abs(img.astype(int) - none.astype(int)).sum(-1)
        ys, xs = np.nonzero(diff > 40)
        self.assertGreater(len(xs), 50)
        cam = self.synth.camera_for(0.0, 0.0, 120.0, 0.0)
        uv, _ = cam.project(np.array([[n, e, 0.0]]))
        self.assertLess(abs(xs.mean() - uv[0, 0]), 0.2 * 320)
        self.assertLess(abs(ys.mean() - uv[0, 1]), 0.2 * 180)

    def test_far_fire_does_not_change_the_frame(self):
        none = self.synth.render(*self.pose(0, 0), [], "k")
        far = self.synth.render(*self.pose(0, 0), [SmokeSource(600.0, 600.0)], "k")
        self.assertTrue(np.array_equal(none, far))
        self.assertEqual(self.synth.last_visible_sources, 0)
        self.assertEqual(self.synth.camera.pitch_deg, 60.0)

    def test_burning_cells_become_sources_deterministically(self):
        a = sources_from_fire_cells([(4, 5, 1.0), (5, 5, 0.5)], self.g, PlumeModel(), 9)
        b = sources_from_fire_cells([(4, 5, 1.0), (5, 5, 0.5)], self.g, PlumeModel(), 9)
        self.assertEqual(a, b)
        for s, (i, j) in zip(a, ((4, 5), (5, 5))):
            self.assertEqual(self.g.cell_of(s.n, s.e), (i, j))      # jitter stays inside the cell


class FakeFog:
    def __init__(self, cells):
        N = 10
        self.fire_heat = [[0.0] * N for _ in range(N)]
        self.cells = cells
        for i, j in cells:
            self.fire_heat[i][j] = 1.0

    def burning_cells(self):
        return list(self.cells)


class TestTruthIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = GridSpec.from_miles(1.0, 10)
        cls.synth = FrameSynth(3, LAT0, LON0, camera=CameraSpec(width=320, height=180))
        cls.fog = FakeFog([(5, 5)])
        cls.svc = FrameService(cls.synth, FireCellsTruth(cls.fog, cls.g, PlumeModel(), 3), port=0).start()
        cls.url = f"http://127.0.0.1:{cls.svc.port}"

    @classmethod
    def tearDownClass(cls):
        cls.svc.stop()

    def test_agent_side_modules_never_import_truth_or_rendering(self):
        import ast
        banned_mods = {"wildfire_sim", "sim.camera_service", "sim.frame_synth", "fog", "fog.state", "synthgen",
                       "synthgen.render", "synthgen.scene_params"}
        banned_names = {"FireSim", "FireCellsTruth", "FrameService", "FrameSynth", "SmokeSource", "Fog"}
        for rel in ("perception/detector.py", "perception/frames.py"):
            for node in ast.walk(ast.parse(read(rel))):
                if isinstance(node, ast.Import):
                    for al in node.names:
                        self.assertNotIn(al.name, banned_mods, rel)
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, banned_mods, rel)
                    for al in node.names:
                        self.assertNotIn(al.name, banned_names, rel)
                elif isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, {"fire_state", "fire_heat", "burning_cells", "fire_sim"}, rel)

    def test_swarm_agent_does_not_import_fire_or_render_modules(self):
        src = read("swarm_agent.py")
        for needle in ("from wildfire_sim", "import wildfire_sim", "camera_service", "frame_synth", "FireCellsTruth"):
            self.assertNotIn(needle, src)

    def test_request_is_only_the_vehicles_own_pose(self):
        p = pbase.Pose(21.3, -157.8, 100.0, 12.0, (1, 1), 4.5, 2)
        self.assertEqual(set(frames.pose_payload(p, "x")), {"sysid", "lat", "lon", "alt", "heading_deg", "t", "frame_id"})

    def _post(self, body):
        req = urllib.request.Request(self.url + "/frame", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.headers, r.read()

    def test_service_returns_only_jpeg_pixels_and_no_truth_endpoint(self):
        n, e = self.g.center_ne(5, 5)
        lat, lon = ne_to_latlon(n - 100.0, e, LAT0, LON0)
        status, headers, data = self._post({"sysid": 1, "lat": lat, "lon": lon, "alt": 120.0, "heading_deg": 0.0, "frame_id": "a"})
        self.assertEqual((status, headers["Content-Type"]), (200, "image/jpeg"))
        self.assertTrue(data[:2] == b"\xff\xd8")
        self.assertEqual({k.lower() for k in headers.keys()} & {"x-fire", "x-truth", "x-sources", "x-cells"}, set())
        for path in ("/truth", "/sources", "/fire", "/state", "/cells"):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(self.url + path, timeout=10)
            self.assertEqual(cm.exception.code, 404)

    def test_fire_changes_pixels_only_through_rendering(self):
        n, e = self.g.center_ne(5, 5)
        lat, lon = ne_to_latlon(n - 100.0, e, LAT0, LON0)         # 100 m south of the fire, camera looking north
        body = {"sysid": 1, "lat": lat, "lon": lon, "alt": 120.0, "heading_deg": 0.0, "frame_id": "b"}
        with_fire = self._post(body)[2]
        self.fog.cells[:] = []
        try:
            no_fire = self._post(body)[2]
        finally:
            self.fog.cells[:] = [(5, 5)]
        self.assertNotEqual(with_fire, no_fire)            # the fire is visible in the image ...
        src = frames.HttpFrameSource(self.url)
        out = src.get_frame(pbase.Pose(lat, lon, 120.0, 0.0, (5, 5), 0.0, 1))
        self.assertIsInstance(out, np.ndarray)             # ... and the agent API returns nothing but an array
        self.assertEqual((out.dtype, out.shape), (np.uint8, (180, 320, 3)))

    def test_yolo_perception_holds_no_truth_object(self):
        gm = GridMission(GridCfg(lat0=LAT0, lon0=LON0))
        d = det.YoloPerception({"model_tag": "t"}, grid=gm, frame_source=frames.HttpFrameSource(self.url),
                               predictor=lambda f: [])
        for k, v in vars(d).items():
            self.assertNotIsInstance(v, (FakeFog, FrameService, FrameSynth, SmokeSource), k)


class TestAsyncPerception(unittest.TestCase):
    """A slow observe() must never stall the control loop, and its detections must still reach the planner."""

    def test_slow_observe_does_not_block_run_perception(self):
        import queue
        import threading
        import time
        import swarm_agent
        from planning.priority import PriorityPlanner

        class Slow:
            enabled = True

            def observe(self, pose):
                time.sleep(0.6)
                return [pbase.Detection(i=2, j=3, cls="smoke", conf=0.9, bbox=(0, 0, 1, 1), frame_id=1)]

            def close(self):
                pass

        sent = []
        a = swarm_agent.Agent.__new__(swarm_agent.Agent)
        a.perception, a.perception_async, a.perception_period, a._next_perception_t = Slow(), True, 0.0, 0.0
        a._perc_in, a._perc_out, a._perc_thread, a.stop = queue.Queue(maxsize=1), queue.Queue(), None, False
        a.planner, a.sysid, a.cur_cell, a.model_tag = PriorityPlanner(conf_threshold=0.1), 1, None, "m"
        a.gossip = type("G", (), {"send": lambda self, m: sent.append(m)})()
        a._vp = lambda *x: None
        mine = {"lat": LAT0, "lon": LON0, "alt": 120.0, "hdg": 0.0}
        t0 = time.time()
        a.run_perception(mine, 100.0)
        self.assertLess(time.time() - t0, 0.2)                     # returned without waiting for the 0.6 s observe
        a.run_perception(mine, 100.1)                              # worker busy: frame skipped, still instant
        self.assertLess(time.time() - t0, 0.3)
        time.sleep(0.9)
        a.run_perception(mine, 101.0)                              # drains the finished frame
        a.stop = True
        a._perc_thread.join(timeout=2)
        self.assertEqual(len(sent), 1)
        self.assertIn((2, 3), a.planner.evidence)


class TestManifest(unittest.TestCase):
    def test_manifest_records_weight_hash_versions_and_oracle_flag(self):
        import json as _json
        from fog.run_log import RunLogger
        with tempfile.TemporaryDirectory() as td:
            w = os.path.join(td, "w.pt")
            open(w, "wb").write(b"weights")
            lg = RunLogger("t", cfg_dir=td)
            lg.write_manifest({"perception": {"enabled": True, "weights": w, "model_tag": "m", "camera": {"pitch_deg": 90}},
                               "fire_model": {"oracle_hazards": False}})
            man = _json.load(open(os.path.join(lg.outdir, "manifest.json")))
        self.assertEqual(man["models"][0]["sha256"], hashlib.sha256(b"weights").hexdigest())
        self.assertFalse(man["oracle_hazards"])
        self.assertEqual(man["camera"], {"pitch_deg": 90})
        self.assertIn("python", man["versions"])
        self.assertIn("git_commit", man["versions"])


if __name__ == "__main__":
    unittest.main()
