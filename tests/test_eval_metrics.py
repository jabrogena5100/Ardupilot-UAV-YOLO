import unittest

import numpy as np

from eval import metrics as M


def gt(*boxes):
    return np.array([[0, *b] for b in boxes], float).reshape(-1, 5)


def pr(*boxes):
    return np.array([[*b[:4], b[4], 0] for b in boxes], float).reshape(-1, 6)


class TestMetrics(unittest.TestCase):
    def test_perfect(self):
        g = {"a": gt((0.2, 0.2, 0.4, 0.4)), "b": gt((0.5, 0.5, 0.9, 0.9))}
        p = {"a": pr((0.2, 0.2, 0.4, 0.4, 0.9)), "b": pr((0.5, 0.5, 0.9, 0.9, 0.8))}
        r = M.evaluate(p, g)
        self.assertAlmostEqual(r["mAP50"], 0.995, places=6)   # Ultralytics-style ceiling
        self.assertAlmostEqual(r["mAP50-95"], 0.995, places=6)
        self.assertEqual(r["at_conf"]["0.25"]["fn"], 0)
        self.assertAlmostEqual(r["max_f1"]["f1"], 1.0)
        self.assertAlmostEqual(r["loc_center_err_diag_mean"], 0.0)

    def test_no_predictions(self):
        r = M.evaluate({}, {"a": gt((0.2, 0.2, 0.4, 0.4))})
        self.assertEqual(r["mAP50"], 0.0)
        self.assertEqual(r["at_conf"]["0.25"]["fn"], 1)
        self.assertIsNone(r["max_f1"])

    def test_false_positive_on_empty_images(self):
        g = {"a": gt((0.2, 0.2, 0.4, 0.4)), "e1": gt(), "e2": gt()}
        p = {"a": pr((0.2, 0.2, 0.4, 0.4, 0.9)), "e1": pr((0.1, 0.1, 0.3, 0.3, 0.6)), "e2": pr((0.1, 0.1, 0.3, 0.3, 0.1))}
        c = M.evaluate(p, g)["at_conf"]["0.25"]
        self.assertEqual((c["tp"], c["fp"], c["fn"]), (1, 1, 0))
        self.assertAlmostEqual(c["image_fp_rate_on_empty"], 0.5)  # e2's detection is below 0.25

    def test_no_empty_images_gives_none(self):
        c = M.evaluate({}, {"a": gt((0.2, 0.2, 0.4, 0.4))})["at_conf"]["0.25"]
        self.assertIsNone(c["image_fp_rate_on_empty"])

    def test_known_ap(self):
        # 2 GT; ranked detections TP, FP, TP -> precision 1, .5, .667 at recall .5, .5, 1
        g = {"a": gt((0.0, 0.0, 0.2, 0.2), (0.5, 0.5, 0.7, 0.7))}
        p = {"a": pr((0.0, 0.0, 0.2, 0.2, 0.9), (0.8, 0.0, 0.9, 0.1, 0.8), (0.5, 0.5, 0.7, 0.7, 0.7))}
        r = M.evaluate(p, g)
        # envelope: p=1 for recall<=.5, p=.667 for .5<recall<=1 -> AP = .5*1 + .5*.667 (101-pt approx)
        self.assertAlmostEqual(r["mAP50"], 0.5 + 0.5 * 2 / 3, delta=0.01)

    def test_one_to_one_matching(self):
        g = {"a": gt((0.2, 0.2, 0.4, 0.4))}
        p = {"a": pr((0.2, 0.2, 0.4, 0.4, 0.9), (0.2, 0.2, 0.4, 0.4, 0.8))}
        c = M.evaluate(p, g)["at_conf"]["0.25"]
        self.assertEqual((c["tp"], c["fp"]), (1, 1))

    def test_localisation_and_iou_thresholds(self):
        g = {"a": gt((0.2, 0.2, 0.4, 0.4))}
        p = {"a": pr((0.21, 0.2, 0.41, 0.4, 0.9))}   # IoU 0.818
        r = M.evaluate(p, g)
        self.assertAlmostEqual(r["mAP50"], 0.995, places=6)
        self.assertLess(r["mAP50-95"], 0.99)
        self.assertAlmostEqual(r["at_conf"]["0.25"]["loc_mean_iou_tp"], 0.19 / 0.21, places=3)

    def test_edge_touch_split(self):
        g = {"a": gt((0.0, 0.2, 0.3, 0.5)), "b": gt((0.4, 0.4, 0.6, 0.6))}
        p = {"b": pr((0.4, 0.4, 0.6, 0.6, 0.9))}
        e = M.evaluate(p, g)["recall_by_edge_touch"]
        self.assertEqual(e["edge_touching"]["recall"], 0.0)
        self.assertEqual(e["not_edge_touching"]["recall"], 1.0)

    def test_bootstrap_ci(self):
        rng = np.random.default_rng(1)
        g, p = {}, {}
        for i in range(60):
            g[f"i{i}"] = gt((0.2, 0.2, 0.4, 0.4))
            if rng.random() < 0.7:
                p[f"i{i}"] = pr((0.2, 0.2, 0.4, 0.4, 0.9))
        a = M.bootstrap_ci(p, g, n=60, seed=3)
        b = M.bootstrap_ci(p, g, n=60, seed=3)
        self.assertEqual(a, b)                               # deterministic
        lo, hi = a["ci"]["recall"]
        point = M.evaluate(p, g)["at_conf"]["0.25"]["recall"]
        self.assertLessEqual(lo, point); self.assertGreaterEqual(hi, point)
        self.assertLess(hi - lo, 0.4)

    def test_load_gt(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "x.txt")
            open(f, "w").write("0 0.5 0.5 0.2 0.4\n")
            a = M.load_gt(f)
            np.testing.assert_allclose(a, [[0, 0.4, 0.3, 0.6, 0.7]])
            self.assertEqual(M.load_gt(os.path.join(d, "none.txt")).shape, (0, 5))


if __name__ == "__main__":
    unittest.main()
