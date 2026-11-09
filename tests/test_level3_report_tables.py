"""Validation of eval/level3_report_tables.py output against independent artifacts, plus documentation consistency checks.

The artifact checks use the committed docs/experiments/level3_final/ files; the provenance check reads the gitignored raw runs and is skipped when they are absent."""
import csv
import glob
import json
import os
import unittest

import numpy as np

from eval import level3_report_tables as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "experiments", "level3_final")
RAW = os.path.join(ROOT, "local_data", "level3", "final")


def rd(name):
    return list(csv.DictReader(open(os.path.join(OUT, name))))


def text(rel):
    return open(os.path.join(ROOT, rel)).read()


class TestHelpers(unittest.TestCase):
    def test_flip_p_floor(self):
        self.assertAlmostEqual(T.flip_p([1, 2, 3, 4]), 0.125)       # 4 clusters: 2/2^4
        self.assertAlmostEqual(T.flip_p([1] * 8), 2 / 256)

    def test_holm_monotone_and_capped(self):
        adj = T.holm([0.125, 0.125, 0.5, 1.0])
        self.assertEqual(adj, [0.5, 0.5, 1.0, 1.0])
        self.assertTrue(all(0 <= a <= 1 for a in adj))

    def test_censoring_substitution(self):
        self.assertEqual(T.cval({"ign_first_visit_s": ""}, "ign_first_visit_s"), 600.0)
        self.assertEqual(T.cval({"visited_frac": "0.2"}, "visited_frac"), 0.2)


class TestArtifacts(unittest.TestCase):
    def setUp(self):
        self.rows = rd("per_run.csv")

    def test_window_correction_in_committed_per_run(self):
        r = {x["run"]: x for x in self.rows}["l3-B2s0-2014"]
        self.assertEqual((r["det_true"], r["det_after_window"]), ("67", "9"))           # 9 records stamped 600.01-600.16 s are excluded
        self.assertEqual(sum(int(x["det_after_window"]) for x in self.rows), 9)
        self.assertEqual(sum(x["mode"] == "seq" for x in self.rows), 26)
        self.assertEqual(sum(x["mode"] == "par" for x in self.rows), 30)

    def test_era_subset_cell_counts_in_committed_stats(self):
        sens = json.load(open(os.path.join(OUT, "cluster_stats.json")))["sensitivity_by_era"]
        self.assertEqual((sens["sequential_scenarios"], sens["parallel_scenarios"]), ([2001, 2002, 2004], [2006, 2012, 2014, 2015]))
        self.assertTrue(all(v["sequential"][-1] == 3 and v["parallel"][-1] == 3 for v in sens["counts"].values()))   # cells per subset: three, not four

    def test_run_count_and_censoring(self):
        self.assertEqual(len(self.rows), 56)
        self.assertEqual(sum(1 for r in self.rows if r["ign_first_visit_s"] == ""), 15)
        self.assertEqual([r["run"] for r in self.rows if r["arm"] != "baseline" and r["first_true_det_s"] == ""], ["l3-B2s1-2004"])
        self.assertEqual(sum(1 for r in self.rows if r["arm"] == "baseline" and r["first_true_det_s"] != ""), 0)

    def test_four_cells_two_scenarios_each(self):
        scen = sorted({int(r["scenario"]) for r in self.rows})
        self.assertEqual(scen, sorted(T.CELL_OF))
        self.assertEqual(sorted(len([s for s in scen if T.CELL_OF[s] == c]) for c in set(T.CELL_OF.values())), [2, 2, 2, 2])
        pinned = {x["seed"]: tuple(x["ignition"]) for x in json.load(open(os.path.join(ROOT, "eval", "level3_scenarios.json")))["heldout"]}
        self.assertEqual(pinned, T.CELL_OF)

    def test_seed_summary_matches_stats_json(self):
        stats = json.load(open(os.path.join(OUT, "stats.json")))["contrasts"]
        arms = {r["arm"]: r for r in rd("seed_summary.csv")}
        for fam, arm_list in (("A", ["A0", "A1", "A2"]), ("B", ["B2s0", "B2s1", "B2s2"])):
            for key in ("first_true_det_s", "ign_first_visit_s", "visited_frac", "det_false", "inv_false_s", "inv_true_s"):
                mean = np.mean([float(arms[a][key]) for a in arm_list])
                self.assertAlmostEqual(mean, stats[key]["family_means"][fam], places=6, msg=(fam, key))

    def test_seed_sd_recomputed_from_per_run(self):
        # independent recomputation of the family SD over the 3 seed means (ddof=1) for coverage
        seeds = []
        for arm in ("A0", "A1", "A2"):
            seeds.append(np.mean([float(r["visited_frac"]) for r in self.rows if r["arm"] == arm]))
        from_table = [r for r in rd("seed_summary.csv") if r["arm"] in ("A0", "A1", "A2")]
        self.assertEqual(len(from_table), 3)
        text_md = open(os.path.join(OUT, "seed_summary.md")).read()
        self.assertIn("%.3f +- %.3f" % (np.mean(seeds), np.std(seeds, ddof=1)), text_md)

    def test_false_detection_extras_consistent(self):
        extra = {r["run"]: r for r in rd("per_run_extra.csv")}
        for r in self.rows:
            e = extra[r["run"]]
            self.assertEqual(int(e["det_false_obs"]), int(r["det_false"]))
            self.assertLessEqual(int(e["det_false_cells"]), int(e["det_false_obs"]))

    def test_cluster_stats_floor_and_holm(self):
        res = json.load(open(os.path.join(OUT, "cluster_stats.json")))["results"]
        self.assertTrue(res)
        for r in res:
            self.assertGreaterEqual(r["p_signflip_cells_unadj"], 0.125 - 1e-9)
            self.assertGreaterEqual(r["p_cells_holm"], 0.125 - 1e-9)
            self.assertEqual(r["cells_pos"] + r["cells_neg"] + r["cells_zero"], 4)
            self.assertEqual(len(r["scenario_diffs"]), 8)

    def test_censoring_table_totals(self):
        c = rd("censoring.csv")
        self.assertEqual(sum(int(r["ign_visit_censored"]) for r in c), 15)
        self.assertEqual(sum(int(r["runs"]) for r in c), 56)

    @unittest.skipUnless(os.path.isdir(os.path.join(RAW, "results")), "raw runs not present")
    def test_provenance_split(self):
        rows = rd("provenance.csv")
        main = [r for r in rows if r["set"] == "results"]
        self.assertEqual(sum(r["git_dirty"] == "False" for r in main), 26)
        self.assertEqual(sum(r["git_dirty"] == "True" for r in main), 30)
        self.assertEqual(len([r for r in rows if r["set"] == "results_repeat"]), 2)
        self.assertTrue(all(r["parallel_slot"] == "" for r in main if r["git_dirty"] == "False"))


class TestDocsConsistency(unittest.TestCase):
    DOCS = ["docs/experiments/level3_final.md", "docs/project-status.md", "docs/handoffs/level3_final_audit.md"]

    def test_no_stale_provenance_claim(self):
        for d in self.DOCS:
            t = text(d).lower()
            self.assertNotIn("every `run_meta.json` has `git_dirty: true`", t, d)
            self.assertNotIn("all runs have `git_dirty: true`", t, d)

    def test_final_doc_states_key_facts(self):
        t = text("docs/experiments/level3_final.md")
        for needle in ("26 clean", "15 of 56", "0.125", "reactive", "probability map", "repeated detection observations", "cannot be recovered", "not declared scientifically validated"):
            self.assertIn(needle.lower(), t.lower(), needle)

    def test_fire_clock_confound_documented_everywhere(self):
        for d, needles in (("docs/experiments/level3_final.md", ("fire-clock confound", "realized fire", "radius", "observed-only", "log lines")),
                           ("docs/experiments/level3_methodology.md", ("## 16. fire-clock confound", "next_fire_step", "realized fire trajectory")),
                           ("docs/handoffs/level3_final_audit.md", ("fire-clock", "resolved", "unresolved", "not declared")),
                           ("docs/project-status.md", ("fire-clock",))):
            t = text(d).lower()
            for n in needles:
                self.assertIn(n.lower(), t, (d, n))
        for d in ("docs/experiments/level3_final.md", "docs/experiments/level3_methodology.md"):
            self.assertNotIn("the scene, fire and detector are deterministic", text(d).lower(), d)

    def test_no_four_cell_claim_for_the_parallel_only_subset(self):
        import re
        for d in ("docs/experiments/level3_final.md", "docs/experiments/level3_methodology.md", "docs/handoffs/level3_final_audit.md", "docs/experiments/level3_final/cluster_stats.md"):
            t = text(d)
            self.assertIsNone(re.search(r"2006, 2012, 2014, 2015[^)\n]{0,40}4 cells", t), d)
            self.assertIsNone(re.search(r"parallel-only[^.\n]{0,120}all 4 cells", t), d)

    def test_docs_state_capped_interpretation_and_fire_clock_limits(self):
        t = text("docs/experiments/level3_final.md").lower()
        for needle in ("capped", "in general a bound on the true difference", "0 <= t <= 600.0", "association, not proof", "not established"):
            self.assertTrue(needle in t, "missing: " + needle)
        for bad in ("a lower bound on the gap", "lower bounds on the size of the difference", "whose fire ran fastest", "adds noise to a-b rather than a systematic"):
            self.assertFalse(bad in t, "stale claim: " + bad)

    def test_baseline_first_detection_is_not_applicable_in_csv(self):
        base = [r for r in rd("seed_summary.csv") if r["arm"] == "baseline"][0]
        self.assertEqual((base["first_true_det_s"], base["first_true_det_s_n_observed"]), ("NA", "NA"))

    def test_no_significance_claims_for_scenarios(self):
        t = text("docs/experiments/level3_final.md")
        self.assertNotIn("p=0.0078", t)
        self.assertNotIn("statistically significant", t.replace("no contrast can be significant", ""))

    def test_numbers_in_doc_match_tables(self):
        t = text("docs/experiments/level3_final.md")
        cs = {(r["metric"], r["contrast"]): r for r in json.load(open(os.path.join(OUT, "cluster_stats.json")))["results"]}
        self.assertIn("%.1f" % cs[("first_true_det_s", "A-B")]["mean_diff"], t)
        self.assertIn("%.1f" % cs[("ign_first_visit_s", "B-baseline")]["mean_diff"], t)
        self.assertIn("%.3f" % cs[("visited_frac", "A-B")]["mean_diff"], t)


if __name__ == "__main__":
    unittest.main()
