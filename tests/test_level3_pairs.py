"""Paired Level 3 plan: arms differ only in the perception arm; scenarios come from the pinned split; held-out is guarded."""
import copy
import json
import os
import tempfile
import unittest

import yaml

from eval import level3_pairs as L
from eval import level3_scenarios as SC
from perception import detector as D


class TestPairs(unittest.TestCase):
    def setUp(self):
        self._thr = L.THR_PATH
        self.tmp = tempfile.TemporaryDirectory()
        L.THR_PATH = os.path.join(self.tmp.name, "thr.json")          # an isolated thresholds file per test

    def tearDown(self):
        L.THR_PATH = self._thr
        self.tmp.cleanup()

    def write_thr(self, taus):
        json.dump({"arms": {a: {"tau": t} for a, t in taus.items()}}, open(L.THR_PATH, "w"))

    def test_plan_calibration_pairs_every_arm_and_never_uses_heldout(self):
        self.write_thr({"A0": 0.4, "B2s0": 0.3})
        rows = L.plan("calibration", list(L.ARMS), os.path.join(self.tmp.name, "p"))
        cal = {s["seed"] for s in SC.load()["calibration"]}
        held = {s["seed"] for s in SC.load()["heldout"]}
        self.assertEqual({r["scenario"] for r in rows}, cal)
        self.assertEqual(len(rows), len(cal) * len(L.ARMS))
        for seed in cal:
            cfgs = {r["arm"]: yaml.safe_load(open(r["config"])) for r in rows if r["scenario"] == seed}
            L.assert_paired(cfgs)
            self.assertEqual({c["fire_model"]["random_seed"] for c in cfgs.values()}, {seed})
            self.assertEqual({tuple(map(tuple, c["fire_model"]["initial_ignitions"])) for c in cfgs.values()},
                             {tuple(map(tuple, [next(s for s in SC.load()["calibration"] if s["seed"] == seed)["ignition"]]))})
            self.assertEqual(cfgs["A0"]["perception"]["conf_threshold"], 0.4)
            self.assertEqual(cfgs["B2s0"]["perception"]["conf_threshold"], 0.3)
            self.assertFalse(cfgs["baseline"]["perception"]["enabled"])
            self.assertFalse(any(c["fire_model"]["oracle_hazards"] for c in cfgs.values()))
            self.assertTrue(all(c["perception"]["camera"]["yaw_to_track"] for c in cfgs.values()))   # identical flight in every arm
            self.assertTrue(all(c["fire_model"]["start_gate"]["agents"] == 1 for c in cfgs.values()))
        self.assertFalse(held & {r["scenario"] for r in rows})

    def test_heldout_is_refused_without_the_flag_and_before_freezing(self):
        with self.assertRaises(PermissionError):
            L.plan("heldout", ["baseline"], os.path.join(self.tmp.name, "h"))
        with self.assertRaises(PermissionError):                 # flag given but thresholds not frozen yet
            L.plan("heldout", ["baseline"], os.path.join(self.tmp.name, "h"), final=True)
        self.write_thr({"A0": 0.4})
        rows = L.plan("heldout", ["baseline", "A0"], os.path.join(self.tmp.name, "h"), final=True)
        self.assertEqual({r["scenario"] for r in rows}, {s["seed"] for s in SC.load()["heldout"]})

    def test_unpaired_difference_or_uncalibrated_tau_is_refused(self):
        self.write_thr({"A0": 0.4, "B2s0": 0.3})
        base = L.build_base()
        sc = SC.load()["calibration"][0]
        a, b = L.make_config(base, "A0", sc), L.make_config(base, "B2s0", sc)
        L.assert_paired({"A0": a, "B2s0": b})
        c = copy.deepcopy(b)
        c["perception"]["rate_hz"] = 0.5
        with self.assertRaises(ValueError):
            L.assert_paired({"A0": a, "B2s0": c})
        d = copy.deepcopy(b)
        d["perception"]["conf_threshold"] = 0.55                   # not the calibrated value
        with self.assertRaises(ValueError):
            L.assert_paired({"A0": a, "B2s0": d})
        e = copy.deepcopy(b)
        e["vehicle_defaults"]["alt"] = 50
        with self.assertRaises(ValueError):
            L.assert_paired({"A0": a, "B2s0": e})

    def test_scenarios_differ_and_agent_seeds_follow_the_scenario(self):
        self.write_thr({})
        base = L.build_base()
        s1, s2 = SC.load()["calibration"][:2]
        a, b = L.make_config(base, "A0", s1), L.make_config(base, "A0", s2)
        self.assertNotEqual(L.strip_arm(a), L.strip_arm(b))
        self.assertEqual(L.make_config(base, "A0", s1)["agents"][0]["seed"], L.make_config(base, "B2s0", s1)["agents"][0]["seed"])

    def test_arm_checkpoints_are_registry_files(self):
        for arm, v in L.ARMS.items():
            if v is not None:
                self.assertIn(os.path.basename(v[0]), D.KNOWN_SHA256, arm)

    def test_base_uses_the_frozen_fire_and_the_documented_camera(self):
        base = L.build_base()
        fire = L.load_fire()
        self.assertEqual((base["fire_model"]["p_ignite"], base["fire_model"]["burn_steps"]), (fire["p_ignite"], fire["burn_steps"]))
        self.assertEqual((base["perception"]["camera"]["pitch_deg"], base["perception"]["camera"]["hfov_deg"]), (60, 78))
        self.assertFalse(base["fire_model"]["oracle_hazards"])
        self.assertEqual(base["fog_tracker"]["max_seconds"], fire["T_s"])

    def test_process_matcher_ignores_shells_that_only_mention_names(self):
        self.assertEqual(L._pids({"definitely_not_a_running_program.py"}), set())


if __name__ == "__main__":
    unittest.main()
