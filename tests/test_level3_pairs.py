"""Paired Level 3 plan: arms differ only in the perception arm; scenarios are fixed by the seed."""
import os
import tempfile
import unittest

import yaml

from eval import level3_pairs as L
from perception import detector as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "exp_swarm_perception_single.yaml")


class TestPairs(unittest.TestCase):
    def test_plan_writes_every_arm_and_seed_and_is_paired(self):
        with tempfile.TemporaryDirectory() as td:
            rows = L.plan(BASE, [1, 2], list(L.ARMS), td)
            self.assertEqual(len(rows), 2 * len(L.ARMS))
            for seed in (1, 2):
                cfgs = {r["arm"]: yaml.safe_load(open(r["config"])) for r in rows if r["seed"] == seed}
                L.assert_paired(cfgs)
                self.assertEqual({c["fire_model"]["random_seed"] for c in cfgs.values()}, {seed})
                self.assertEqual({c["perception"]["camera_service"]["scenario_seed"] for c in cfgs.values()}, {seed})
                self.assertEqual(len({str([a["seed"] for a in c["agents"]]) for c in cfgs.values()}), 1)
                self.assertFalse(cfgs["baseline"]["perception"]["enabled"])
                self.assertTrue(all(c["perception"]["enabled"] for a, c in cfgs.items() if a != "baseline"))
                self.assertFalse(any(c["fire_model"]["oracle_hazards"] for c in cfgs.values()))

    def test_unpaired_difference_is_refused(self):
        base = yaml.safe_load(open(BASE))
        a, b = L.make_config(base, "A0", 1), L.make_config(base, "B2s0", 1)
        L.assert_paired({"A0": a, "B2s0": b})
        b["perception"]["conf_threshold"] = 0.5
        with self.assertRaises(ValueError):
            L.assert_paired({"A0": a, "B2s0": b})
        c = L.make_config(base, "B2s0", 1)
        c["vehicle_defaults"] = {"alt": 50}
        with self.assertRaises(ValueError):
            L.assert_paired({"A0": a, "B2s0": c})

    def test_scenarios_differ_between_seeds(self):
        base = yaml.safe_load(open(BASE))
        a, b = L.make_config(base, "A0", 1), L.make_config(base, "A0", 2)
        self.assertNotEqual(L.strip_arm(a), L.strip_arm(b))

    def test_arm_checkpoints_are_registry_files(self):
        for arm, v in L.ARMS.items():
            if v is not None:
                self.assertIn(os.path.basename(v[0]), D.KNOWN_SHA256, arm)

    def test_base_config_is_pair_ready(self):
        base = yaml.safe_load(open(BASE))
        self.assertFalse(base["fire_model"]["oracle_hazards"])
        self.assertEqual(base["perception"]["conf_threshold"], 0.25)
        self.assertEqual(base["perception"]["camera"]["pitch_deg"], 90)


if __name__ == "__main__":
    unittest.main()
