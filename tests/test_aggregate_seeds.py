import json
import os
import tempfile
import unittest

import numpy as np

from eval import aggregate_seeds as A


def cell(model, train, ds, m50, m5095, rec):
    return {"model": model, "train_domain": train, "dataset": ds, "checkpoint_sha256": "x",
            "metrics": {"mAP50": m50, "mAP50-95": m5095,
                        "at_conf": {"0.25": {"recall": rec, "precision": 0.9, "fp_per_image": 0.1}}}}


class TestAggregate(unittest.TestCase):
    def test_stats_and_deltas(self):
        with tempfile.TemporaryDirectory() as d:
            vals = {"A": {"synth_test_pos": [0.99, 0.98, 0.97], "boreal_test": [0.40, 0.45, 0.50]},
                    "B": {"boreal_test": [0.95, 0.94, 0.96], "synth_test_pos": [0.70, 0.75, 0.80]}}
            for fam, tr in (("A", "synthetic"), ("B", "real")):
                for ds, v in vals[fam].items():
                    for s in range(3):
                        name = f"model_{fam.lower()}" + (f"_seed{s}" if s else "")
                        c = cell(name, tr, ds, v[s], v[s] / 2, v[s])
                        json.dump(c, open(os.path.join(d, f"{name}__{ds}.json"), "w"))
            r = A.aggregate(d)
            st = r["cells"]["A|boreal_test"]["metrics"]["mAP50"]
            self.assertAlmostEqual(st["mean"], 0.45); self.assertAlmostEqual(st["sd"], np.std([.40, .45, .50], ddof=1))
            da = r["deltas"]["A"]["metrics"]["mAP50"]
            np.testing.assert_allclose(da["values"], [0.40 - 0.99, 0.45 - 0.98, 0.50 - 0.97])
            self.assertEqual(r["cells"]["B|boreal_test"]["seeds"], [0, 1, 2])
            self.assertIn("not seed uncertainty", A.render(r))


if __name__ == "__main__":
    unittest.main()
