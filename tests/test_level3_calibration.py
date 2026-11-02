"""Level 3 methodology: scenario split, fire calibration rule, threshold selection, gate, yaw, import isolation."""
import json
import os
import subprocess
import sys
import unittest

from eval import level3_calibrate_detector as CD
from eval import level3_fire_calibration as FC
from eval import level3_scenarios as SC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestScenarioSplit(unittest.TestCase):
    def test_pinned_split_has_no_leakage(self):
        sc = SC.load()
        SC.verify(sc)                                              # seeds, texture slots and ignition cells disjoint; slots recomputed
        self.assertEqual(sc["heldout_sha256"], SC.digest(sc))
        self.assertEqual((len(sc["calibration"]), len(sc["heldout"])), (SC.N_CAL, SC.N_HELD))
        self.assertTrue({s["texture_slot"] for s in sc["calibration"]} <= set(SC.CAL_SLOTS))
        self.assertTrue({s["texture_slot"] for s in sc["heldout"]} <= set(SC.HELD_SLOTS))
        for s in sc["calibration"] + sc["heldout"]:
            i, j = s["ignition"]
            self.assertGreaterEqual(max(abs(i - SC.TAKEOFF_CELL[0]), abs(j - SC.TAKEOFF_CELL[1])), 3)

    def test_regeneration_matches_the_pinned_file(self):
        self.assertEqual(SC.build(), {k: SC.load()[k] for k in ("calibration", "heldout")})

    def test_calibration_tools_refuse_heldout_seeds(self):
        held = SC.load()["heldout"][0]["seed"]
        with self.assertRaises(PermissionError):
            SC.require_calibration([SC.load()["calibration"][0]["seed"], held])
        SC.require_calibration([s["seed"] for s in SC.load()["calibration"]])


class TestFireCalibration(unittest.TestCase):
    def test_frozen_settings_are_what_the_rule_selects_on_calibration_scenarios(self):
        rows, best = FC.sweep(FC.T_DEFAULT)
        frozen = json.load(open(FC.OUT))
        self.assertEqual((best["p_ignite"], best["burn_steps"]), (frozen["p_ignite"], frozen["burn_steps"]))
        self.assertEqual(frozen["rule_version"], FC.RULE_VERSION)
        for r in best["per_scenario"]:
            self.assertTrue(FC.LO <= r["burned_frac"] <= FC.HI and r["burning_at_T"] >= 1 and r["burning_at_60s"] >= FC.EARLY)

    def test_sweep_uses_only_calibration_scenarios(self):
        cal = {s["seed"] for s in SC.load()["calibration"]}
        rows, _ = FC.sweep(60)
        self.assertEqual({x["seed"] for r in rows for x in r["per_scenario"]}, cal)


def frame(kind, vis, burning, dets, scen=1, pas="fire60", m=0):
    return {"scenario": scen, "pass": pas, "kind": kind, "frame": m, "t_sim": 0, "visibility": vis, "burning": burning, "dets": {"X": dets}}


class TestThresholdSelection(unittest.TestCase):
    def frames(self):
        b = [(5, 5)]
        return [
            frame("fire", "empty", b, [], m=0),
            frame("fire", "visible", b, [{"conf": 0.9, "cell": (5, 5)}, {"conf": 0.2, "cell": (0, 0)}], m=1),   # true 0.9, false 0.2
            frame("fire", "visible", b, [{"conf": 0.5, "cell": (5, 6)}], m=2),                                  # true (within 1 cell)
            frame("free", "empty", b, [{"conf": 0.3, "cell": (1, 1)}], pas="free90", m=0),                      # false alarm on empty
            frame("free", "empty", b, [], pas="free90", m=1),
        ]

    def test_metrics_at_thresholds(self):
        lo = CD.metrics(self.frames(), "X", 0.10)
        self.assertEqual((lo["tp"], lo["fp"]), (2, 2))
        self.assertEqual(lo["precision"], 0.5)
        self.assertEqual(lo["recall"], 1.0)
        self.assertEqual(lo["empty_frame_alarm_rate"], 0.3333)       # 1 alarm over 3 empty frames (fire 0, free 0, free 1)
        hi = CD.metrics(self.frames(), "X", 0.60)
        self.assertEqual((hi["tp"], hi["fp"], hi["precision"], hi["recall"]), (1, 0, 1.0, 0.5))
        self.assertEqual(hi["latency_median_s"], 0.0)               # first visible frame (1) already has the true detection

    def test_latency_and_censoring(self):
        b = [(5, 5)]
        fr = [frame("fire", "visible", b, [], m=0), frame("fire", "visible", b, [{"conf": 0.8, "cell": (5, 5)}], m=1),
              frame("fire", "visible", b, [], pas="fire120", m=0)]
        m = CD.metrics(fr, "X", 0.5)
        self.assertEqual((m["latency_median_s"], m["passes_detected"], m["passes_censored"]), (CD.DT, 1, 1))

    def test_rule_is_argmax_f05_with_ties_to_higher_tau(self):
        d = {"scenarios": [s["seed"] for s in SC.load()["calibration"]], "pitch_deg": 60, "hfov_deg": 78, "alt_m": 120,
             "fire": {}, "arms": ["X"], "frames": self.frames()}
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p, o = os.path.join(td, "f.json"), os.path.join(td, "t.json")
            json.dump(d, open(p, "w"))
            CD.select(p, o)
            res = json.load(open(o))["arms"]["X"]
        best = max(res["sweep"], key=lambda r: (r["f0.5"], r["tau"]))
        self.assertEqual(res["tau"], best["tau"])
        self.assertTrue(res["usable"])
        self.assertGreater(res["tau"], 0.10)                         # 0.9-conf true detection survives; the 0.2/0.3 false ones do not

    def test_selection_refuses_heldout_scenarios(self):
        import tempfile
        d = {"scenarios": [SC.load()["heldout"][0]["seed"]], "pitch_deg": 60, "hfov_deg": 78, "alt_m": 120, "fire": {}, "arms": [], "frames": []}
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "f.json")
            json.dump(d, open(p, "w"))
            with self.assertRaises(PermissionError):
                CD.select(p, os.path.join(td, "o.json"))


class TestGateYawImports(unittest.TestCase):
    def test_gate_ready_counts_fresh_airborne_agents(self):
        import fog_tracker
        info = {1: {"alt": 118.0, "last_t": 100.0}, 2: {"alt": 30.0, "last_t": 100.0}, 3: {"alt": 120.0, "last_t": 50.0}, 4: {"alt": None, "last_t": 100.0}}
        self.assertEqual(fog_tracker.gate_ready(info, 100.0, now=101.0), 1)
        self.assertEqual(fog_tracker.gate_ready(info, 20.0, now=101.0), 2)
        self.assertEqual(fog_tracker.gate_ready({}, 100.0, now=0.0), 0)

    def test_track_yaw_is_a_compass_bearing_and_holds_when_slow(self):
        import math
        from mission.geo import track_yaw
        self.assertAlmostEqual(track_yaw(5.0, 0.0, None), 0.0)                       # north
        self.assertAlmostEqual(track_yaw(0.0, 5.0, None), math.pi / 2)               # east
        self.assertAlmostEqual(track_yaw(-5.0, 0.0, None), math.pi)                  # south
        self.assertEqual(track_yaw(0.5, 0.5, 1.23), 1.23)                            # below 2 m/s: keep the last yaw

    def test_cell_index_moved_without_change(self):
        import fog.state
        import mission.geo as g
        self.assertIs(fog.state.cell_index, g.cell_index)
        half, cell, N = 804.672, 160.9344, 10
        self.assertEqual(g.cell_index(1.0, 1.0, half, cell, N), (5, 5))       # just inside; the exact origin sits on a float cell boundary
        self.assertEqual(g.cell_index(-half, -half, half, cell, N), (0, 0))
        self.assertEqual(g.cell_index(160.0, -161.0, half, cell, N), (3, 5))        # i from east, j from north
        self.assertIsNone(g.cell_index(half, 0.0, half, cell, N))
        self.assertIsNone(g.cell_index(0.0, -half - 1e-6, half, cell, N))

    def test_agent_side_perception_imports_no_fog_fire_or_renderer(self):
        code = ("import sys; sys.path.insert(0, %r); import perception.base, perception.detector, perception.frames, sim.camera; "
                "print(sorted(m for m in sys.modules if m.split('.')[0] in ('fog', 'wildfire_sim', 'synthgen', 'fog_tracker')))" % ROOT)
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True).stdout.strip()
        self.assertEqual(out, "[]")


if __name__ == "__main__":
    unittest.main()
