import json
import os
import tempfile
import unittest

import numpy as np
import yaml

from eval import metrics as M
from eval import run_matrix as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH = os.path.join(ROOT, "local_data", "datasets", "synth_smoke_v1")


def oracle(ckpt, images, s):
    """Predict exactly the ground truth with conf 0.9 (stands in for a perfect model)."""
    out = {}
    for n, img in images.items():
        g = M.load_gt(R.label_path(img))
        out[n] = np.concatenate([g[:, 1:5], np.full((len(g), 1), 0.9), g[:, :1]], axis=1) if len(g) else np.zeros((0, 6))
    return out


class TestRunMatrix(unittest.TestCase):
    def test_label_path(self):
        self.assertEqual(R.label_path("/d/images/test/a.jpg"), "/d/labels/test/a.txt")
        with self.assertRaises(ValueError):
            R.label_path("/d/pics/a.jpg")

    def test_pin_mismatch_aborts(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(f"{d}/images/test"); os.makedirs(f"{d}/labels/test")
            open(f"{d}/images/test/a.jpg", "wb").write(b"x")
            with self.assertRaises(SystemExit):
                R.load_dataset("x", {"images_dir": f"{d}/images/test", "n_images": 2})
            ds = R.load_dataset("x", {"images_dir": f"{d}/images/test", "n_images": 1})
            self.assertEqual(ds["n_images"], 1)

    def test_unresolved_variable(self):
        with self.assertRaises(KeyError):
            R.expand({"a": "${NOPE}/x"}, {})

    @unittest.skipUnless(os.path.isdir(SYNTH), "local synth_smoke_v1 not present")
    def test_frozen_synth_pins_and_oracle_cell(self):
        cfg = R.expand(yaml.safe_load(open(os.path.join(ROOT, "eval", "datasets.yaml"))), {"SYNTH_ROOT": SYNTH, "BOREAL_SPLIT": "/nonexistent", "RUNS": "/nonexistent"})
        for name in ("synth_test_pos", "synth_test_all"):
            ds = R.load_dataset(name, cfg["datasets"][name])   # raises if the frozen pin does not match
        pos = R.load_dataset("synth_test_pos", cfg["datasets"]["synth_test_pos"])
        with tempfile.TemporaryDirectory() as out:
            cell = R.run_cell("oracle", {"checkpoint": __file__, "train_domain": "x"}, pos, cfg["settings"],
                              oracle, out, None, ["test"])
            m = cell["metrics"]
            self.assertEqual(m["n_gt_boxes"], 701)
            self.assertAlmostEqual(m["mAP50"], 0.995, places=3)
            self.assertEqual(m["at_conf"]["0.25"]["fn"], 0)
            # recompute from the saved predictions (no model): identical metrics
            again = R.run_cell("oracle", {"checkpoint": __file__}, pos, cfg["settings"], None,
                               out, os.path.join(out, "predictions"), ["test"])
            self.assertEqual(json.dumps(again["metrics"], sort_keys=True), json.dumps(m, sort_keys=True))
            self.assertTrue(os.path.exists(os.path.join(out, "gts", "synth_test_pos.json.gz")))
            R.write_summary([cell], cfg, out)
            self.assertTrue(os.path.exists(os.path.join(out, "matrix.md")))


if __name__ == "__main__":
    unittest.main()
