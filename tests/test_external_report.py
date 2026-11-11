import json, os, tempfile, unittest

import numpy as np
import yaml

from eval import external_report as R
from eval import run_matrix as RM


def make_run(d, n_clusters=30, per=3, a_quality=0.5, b_quality=0.9, seed=0):
    """Fake run directory: images i=0..N-1, one GT box each; model quality = probability that a prediction matches the GT box."""
    rng = np.random.default_rng(seed)
    names = [f"img{i:04d}.jpg" for i in range(n_clusters * per)]
    gts = {n: np.array([[0, .2, .2, .6, .6]]) for n in names}
    RM.save_gts(os.path.join(d, "gts", "ds.json.gz"), gts) if os.path.isdir(os.path.join(d, "gts")) else None
    os.makedirs(os.path.join(d, "gts"), exist_ok=True); os.makedirs(os.path.join(d, "predictions"), exist_ok=True)
    RM.save_gts(os.path.join(d, "gts", "ds.json.gz"), gts)
    for m, fam in R.FAMILY.items():
        q = a_quality if fam == "A" else b_quality
        preds = {}
        for n in names:
            if rng.random() < q:
                preds[n] = np.array([[.2, .2, .6, .6, .9, 0]])
            else:
                preds[n] = np.array([[.7, .7, .9, .9, .8, 0]])
        RM.save_preds(os.path.join(d, "predictions", f"{m}__ds.json.gz"), preds)
    clusters = {n: {"platform_by_size": "tower_fixed_camera" if i % 2 == 0 else "uav_nadir_1080p", **{f"cluster_r{r}": f"R{r}_{i // per:04d}" for r in (2, 4, 6)}} for i, n in enumerate(names)}
    json.dump({"images": clusters}, open(os.path.join(d, "clusters.json"), "w"))
    return names, gts


class TestReportPieces(unittest.TestCase):
    def test_gate(self):
        self.assertEqual(R.gate(5), "descriptive_only"); self.assertEqual(R.gate(8), "exact_test_only"); self.assertEqual(R.gate(19), "exact_test_only"); self.assertEqual(R.gate(20), "interval_and_test")

    def test_family_table_and_difference(self):
        pm = {m: {k: (0.5 if R.FAMILY[m] == "A" else 0.8) + 0.01 * R.SEED[m] for k in R.KEYS} for m in R.FAMILY}
        ft = R.family_table(pm)
        self.assertAlmostEqual(ft["A"]["mAP50"]["mean"], 0.51); self.assertAlmostEqual(ft["B"]["mAP50"]["mean"], 0.81)
        self.assertAlmostEqual(ft["B_minus_A"]["mAP50"], 0.30); self.assertAlmostEqual(ft["A"]["mAP50"]["sd"], 0.01)

    def test_cluster_analysis_respects_the_gate(self):
        with tempfile.TemporaryDirectory() as d:
            names, gts = make_run(d, n_clusters=5)
            preds, g = R.load_run(d, "ds")
            co = {n: f"c{i // 3}" for i, n in enumerate(names)}
            r = R.cluster_analysis(preds, g, co, names, n_boot=200)
            self.assertEqual(r["gate"], "descriptive_only"); self.assertNotIn("bootstrap", r); self.assertNotIn("sign_flip", r)

    def test_cluster_analysis_detects_a_clear_difference(self):
        with tempfile.TemporaryDirectory() as d:
            names, gts = make_run(d, n_clusters=40)
            preds, g = R.load_run(d, "ds")
            co = {n: f"c{i // 3}" for i, n in enumerate(names)}
            r = R.cluster_analysis(preds, g, co, names, n_boot=300)
            self.assertEqual(r["gate"], "interval_and_test")
            diff = r["bootstrap"]["difference"]
            self.assertGreater(diff["ci"][0], 0)                       # B (0.9) clearly above A (0.5)
            self.assertLess(r["sign_flip"]["p"], 0.01)


class TestRestrictionsAndEstimands(unittest.TestCase):
    def test_restricted_subset_gets_point_values_only(self):
        with tempfile.TemporaryDirectory() as d:
            names, gts = make_run(d, n_clusters=40)
            preds, g = R.load_run(d, "ds")
            co = {n: f"c{i // 3}" for i, n in enumerate(names)}
            r = R.cluster_analysis(preds, g, co, names, n_boot=100, allow_inference=False)
            self.assertEqual(r["gate"], "descriptive_only_by_config")
            for k in ("bootstrap", "sign_flip", "leave_one_cluster_out"):
                self.assertNotIn(k, r)
            self.assertIn("B_minus_A", r["cluster_macro_mAP50"]); self.assertEqual(r["estimand"], "cluster_macro_mAP50")

    def test_config_restriction_blocks_all_and_adds_exclusion_subset(self):
        with tempfile.TemporaryDirectory() as d:
            make_run(d, n_clusters=40)
            cfg = os.path.join(d, "cfg.yaml")
            yaml.safe_dump({"datasets": {"ds": {"clusters": os.path.join(d, "clusters.json")}},
                            "analysis": {"gates": {"platform_groups": {"uav_nadir_1080p": {"inference": "descriptive_only"}}}}}, open(cfg, "w"))
            out = os.path.join(d, "report")
            R.main(["--run-dir", d, "--dataset", "ds", "--config", cfg, "--out", out, "--n-boot", "100"])
            ca = json.load(open(os.path.join(out, "report.json")))["cluster_analysis"]
            self.assertEqual(ca["cluster_r6/all"]["gate"], "descriptive_only_by_config"); self.assertNotIn("bootstrap", ca["cluster_r6/all"])
            self.assertEqual(ca["cluster_r6/uav_nadir_1080p"]["gate"], "descriptive_only_by_config")
            self.assertIn("bootstrap", ca["cluster_r6/tower_fixed_camera"])
            self.assertIn("bootstrap", ca["cluster_r6/all_excluding_uav_nadir_1080p"])
            md = open(os.path.join(out, "report.md")).read()
            self.assertIn("Pooled AP", md); self.assertIn("not population-level inference", md); self.assertNotIn("p 0.0000", md)

    def test_monte_carlo_p_is_never_reported_as_zero(self):
        from eval import external_stats as S
        r = S.sign_flip_test(np.full(30, 0.05), seed=0, n_mc=2000)
        self.assertGreater(r["p"], 0); self.assertAlmostEqual(r["p_floor"], 1 / 2001)
        txt = S.format_p(r)
        self.assertIn("floor", txt); self.assertNotIn("0.0000", txt)
        self.assertIn("exact", S.format_p(S.sign_flip_test([0.1, 0.1, 0.1, 0.1])))


class TestEndToEnd(unittest.TestCase):
    def test_report_for_a_fasdd_like_run(self):
        with tempfile.TemporaryDirectory() as d:
            make_run(d, n_clusters=40)
            cfg = os.path.join(d, "cfg.yaml")
            yaml.safe_dump({"datasets": {"ds": {"clusters": os.path.join(d, "clusters.json")}}}, open(cfg, "w"))
            out = os.path.join(d, "report")
            R.main(["--run-dir", d, "--dataset", "ds", "--config", cfg, "--out", out, "--n-boot", "100"])
            rep = json.load(open(os.path.join(out, "report.json")))
            self.assertEqual(rep["n_images"], 120)
            self.assertGreater(rep["overall"]["B_minus_A"]["mAP50"], 0.2)
            self.assertEqual(sorted(rep["groups"]["platform_by_size"]), ["tower_fixed_camera", "uav_nadir_1080p"])
            self.assertEqual(rep["cluster_analysis"]["cluster_r6/all"]["gate"], "interval_and_test")
            self.assertTrue(os.path.exists(os.path.join(out, "report.md")))
            with self.assertRaises(FileExistsError):
                R.main(["--run-dir", d, "--dataset", "ds", "--config", cfg, "--out", out])          # never overwrites a report


if __name__ == "__main__":
    unittest.main()
