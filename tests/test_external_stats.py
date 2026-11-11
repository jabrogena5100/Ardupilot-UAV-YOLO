import unittest

import numpy as np

from eval import external_stats as S


class TestBootstrap(unittest.TestCase):
    def test_point_is_macro_average_over_clusters_and_seed_mean(self):
        a = np.array([[0.2, 0.4, 0.6], [0.4, 0.6, 0.8]])          # 2 seeds, 3 clusters
        r = S.cluster_bootstrap({"A": a}, n_boot=200, seed=1)
        self.assertAlmostEqual(r["families"]["A"]["point"], 0.5)
        self.assertAlmostEqual(r["families"]["A"]["seed_sd"], np.std([0.4, 0.6], ddof=1))

    def test_reproducible_and_seed_sensitive(self):
        rng = np.random.default_rng(0)
        a, b = rng.random((3, 12)), rng.random((3, 12))
        r1 = S.cluster_bootstrap({"A": a, "B": b}, diff=("A", "B"), n_boot=500, seed=7)
        r2 = S.cluster_bootstrap({"A": a, "B": b}, diff=("A", "B"), n_boot=500, seed=7)
        self.assertEqual(r1, r2)
        self.assertNotEqual(r1["difference"]["ci"], S.cluster_bootstrap({"A": a, "B": b}, diff=("A", "B"), n_boot=500, seed=8)["difference"]["ci"])

    def test_difference_is_paired(self):
        # B = A + 0.1 in every cluster: a paired interval has zero width around 0.1 even though A varies a lot between clusters
        a = np.array([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7]])
        r = S.cluster_bootstrap({"A": a, "B": a + 0.1}, diff=("A", "B"), n_boot=2000, seed=3)
        lo, hi = r["difference"]["ci"]
        self.assertAlmostEqual(lo, 0.1, places=9); self.assertAlmostEqual(hi, 0.1, places=9)
        self.assertEqual(r["difference"]["p_greater_than_zero"], 1.0)
        self.assertGreater(r["families"]["A"]["ci"][1] - r["families"]["A"]["ci"][0], 0.2)    # each family alone is uncertain

    def test_interval_widens_with_fewer_clusters(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0.5, 0.2, 200)
        w = lambda n: np.diff(S.cluster_bootstrap({"A": x[:n]}, n_boot=2000, seed=0)["families"]["A"]["ci"])[0]
        self.assertGreater(w(8), w(100))

    def test_coverage_of_the_true_mean_is_reasonable(self):
        rng = np.random.default_rng(2)
        hit = 0
        for t in range(100):
            x = rng.normal(0.4, 0.15, 30)
            lo, hi = S.cluster_bootstrap({"A": x}, n_boot=400, seed=t)["families"]["A"]["ci"]
            hit += lo <= 0.4 <= hi
        self.assertGreaterEqual(hit, 88)          # nominal 95% with 30 clusters; loose bound, bootstrap under-covers a little

    def test_nan_clusters_are_skipped(self):
        a = np.array([[0.5, np.nan, 0.7, np.nan]])
        r = S.cluster_bootstrap({"A": a}, n_boot=300, seed=0)
        self.assertAlmostEqual(r["families"]["A"]["point"], 0.6); self.assertEqual(r["families"]["A"]["n_clusters_with_value"], 2)

    def test_mismatched_clusters_rejected(self):
        with self.assertRaises(ValueError):
            S.cluster_bootstrap({"A": np.zeros((1, 3)), "B": np.zeros((1, 4))}, n_boot=10)


class TestTests(unittest.TestCase):
    def test_sign_flip_exact_value(self):
        # four equal positive differences: only the all-plus and all-minus patterns reach the observed mean -> p = 2/16
        r = S.sign_flip_test([0.1, 0.1, 0.1, 0.1])
        self.assertEqual(r["method"], "exact"); self.assertAlmostEqual(r["p"], 2 / 16)

    def test_sign_flip_symmetric_data_not_significant(self):
        self.assertGreater(S.sign_flip_test([0.1, -0.1, 0.2, -0.2, 0.05, -0.05])["p"], 0.9)

    def test_sign_flip_monte_carlo_for_large_n(self):
        r = S.sign_flip_test(np.full(30, 0.05), seed=0, n_mc=20000)
        self.assertEqual(r["method"], "monte_carlo"); self.assertLess(r["p"], 0.001)

    def test_leave_one_out(self):
        r = S.leave_one_cluster_out([0.2, 0.2, 0.2, -0.5])
        self.assertTrue(r["min"] < 0 < r["max"]); self.assertFalse(r["all_same_sign"])
        self.assertTrue(S.leave_one_cluster_out([0.1, 0.2, 0.3])["all_same_sign"])

    def test_paired_differences_drop_missing(self):
        d = S.paired_cluster_differences([[0.1, np.nan, 0.3]], [[0.2, 0.5, 0.2]])
        self.assertTrue(np.allclose(d, [0.1, -0.1]))


class TestClusters(unittest.TestCase):
    def test_dhash_clusters(self):
        a = 0xF0F0F0F0F0F0F0F0
        h = [a, a ^ 0b11, a ^ (1 << 63), 0x0123456789ABCDEF, 0x0123456789ABCDEF ^ 1, ~a & 0xFFFFFFFFFFFFFFFF]
        self.assertEqual(S.dhash_clusters(h, 4), [0, 0, 0, 1, 1, 2])
        self.assertEqual(S.dhash_clusters(h, 0), [0, 1, 2, 3, 4, 5])
        self.assertEqual(S.dhash_clusters(h, 1), [0, 1, 0, 2, 2, 3])                   # radius 1: only the 1-bit neighbours link

    def test_chaining_merges_through_intermediate(self):
        a = 0
        h = [a, 0b1111, 0b11111111]                      # 4 bits apart each, 8 bits end to end
        self.assertEqual(S.dhash_clusters(h, 4), [0, 0, 0])
        self.assertEqual(S.dhash_clusters(h, 3), [0, 1, 2])


class TestEffectiveN(unittest.TestCase):
    def test_iid_has_n_eff_close_to_n(self):
        x = np.random.default_rng(0).normal(size=2000)
        self.assertGreater(S.effective_n(x)["n_eff"], 1000)

    def test_smooth_series_has_small_n_eff(self):
        x = np.cumsum(np.random.default_rng(1).normal(size=2000))
        self.assertLess(S.effective_n(x)["n_eff"], 100)

    def test_constant_series(self):
        self.assertIsNone(S.effective_n([1.0] * 50)["n_eff"])


class TestClusterScores(unittest.TestCase):
    def test_uses_metrics_evaluate_per_cluster(self):
        # two clusters; cluster c1 perfect detections, c2 misses everything; empty cluster c3 -> NaN
        gt = {"a": np.array([[0, .1, .1, .5, .5]]), "b": np.array([[0, .2, .2, .6, .6]]), "c": np.array([[0, .1, .1, .5, .5]]), "d": np.zeros((0, 5))}
        pr = {"a": np.array([[.1, .1, .5, .5, .9, 0]]), "b": np.array([[.2, .2, .6, .6, .8, 0]]), "c": np.zeros((0, 6)), "d": np.zeros((0, 6))}
        cl = {"a": "c1", "b": "c1", "c": "c2", "d": "c3"}
        s = S.cluster_scores(pr, gt, cl)
        self.assertGreater(s["c1"], 0.9); self.assertEqual(s["c2"], 0.0); self.assertTrue(np.isnan(s["c3"]))


if __name__ == "__main__":
    unittest.main()
