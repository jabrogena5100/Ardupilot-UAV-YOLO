import os
import unittest

from eval import boreal_v2_report as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELLS = os.path.join(ROOT, "docs", "experiments", "level2_v2", "cells")   # v1 cells are used only as a format fixture


class ReportTests(unittest.TestCase):
    def test_reproduces_known_means_from_cells(self):
        md, res = R.build(CELLS, boreal_ds="boreal_test", neg_ds="boreal_empty_train_diag", b_prefix="model_b")
        t = res["tables"]
        mean = lambda k, m: sum(v[m] for v in t[k].values()) / len(t[k])
        self.assertAlmostEqual(mean("A|boreal_test", "mAP50"), 0.375, places=3)      # level2_v2 committed value
        self.assertAlmostEqual(mean("B|boreal_test", "mAP50"), 0.954, places=3)
        self.assertIn("not directly comparable", md)
        self.assertIn("Negative-scene diagnostics", md)

    def test_stats_sd_is_sample_sd(self):
        st = R.stats([0.5, 0.7, 0.9])
        self.assertAlmostEqual(st["sd"], 0.2)
        self.assertIsNone(R.stats([0.5])["sd"])
        self.assertIsNone(R.stats([None])["mean"])


if __name__ == "__main__":
    unittest.main()
