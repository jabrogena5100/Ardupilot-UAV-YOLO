"""Calculation tests for eval/level3_report_tables.py on SYNTHETIC inputs with hand-computed expected values.

Unlike tests/test_level3_report_tables.py (which validates the committed artifacts), these run the functions on constructed data, so a wrong formula
(ddof, cell averaging, sign-flip counting, Holm, era subsets, fire-clock ratio, censoring bookkeeping) fails here. Passing does not establish scientific validity.
"""
import csv
import itertools
import json
import os
import tempfile
import unittest

import numpy as np

from eval import level3_report_tables as T

SCEN = sorted(T.CELL_OF)                    # the 8 held-out scenarios; cells: (3,2)=2001,2005 (2,1)=2002,2012 (1,0)=2004,2014 (0,1)=2006,2015


def mk(arm, scen, **kw):
    r = {"run": f"l3-{arm}-{scen}", "arm": arm, "family": T.FAM[arm], "scenario": str(scen), "first_true_det_s": "100", "ign_first_visit_s": "100", "visited_frac": "0.2",
         "det_true": "10", "det_false": "0", "inv_true_s": "0", "inv_false_s": "0"}
    r.update({k: ("" if v is None else str(v)) for k, v in kw.items()})
    return r


def extras(rows):
    return [{"run": r["run"], "det_false_cells": int(float(r["det_false"]) // 2), "det_false_obs": int(float(r["det_false"]))} for r in rows]


def csv_rows(path):
    return list(csv.DictReader(open(path)))


class TestHelpers(unittest.TestCase):
    def test_flip_p_against_independent_count(self):
        for d in ([1, 1, 1, -1], [2, 3, 5, 7], [1, -1, 2, -2], [0, 0, 1, 1]):
            tot = abs(sum(d))
            expect = sum(1 for sg in itertools.product((1, -1), repeat=4) if abs(sum(s * v for s, v in zip(sg, d))) >= tot - 1e-12) / 16
            self.assertAlmostEqual(T.flip_p(d), expect, msg=d)
        self.assertAlmostEqual(T.flip_p([1, 2, 3, 4]), 0.125)

    def test_holm_known_values(self):
        self.assertEqual(T.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])           # sorted 0.01*3, 0.03*2, 0.04*1 with running max
        self.assertEqual(T.holm([0.5, 0.5]), [1.0, 1.0])

    def test_cval_censors_only_time_to_event_metrics(self):
        self.assertEqual(T.cval({"ign_first_visit_s": "", "visited_frac": ""}, "ign_first_visit_s"), 600.0)
        self.assertIsNone(T.cval({"visited_frac": ""}, "visited_frac"))

    def test_extra_rows_counts_observations_vs_distinct_cells_and_ignores_pre_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "l3-B2s0-2001")
            os.makedirs(d)
            with open(os.path.join(d, "detections_l3-B2s0-2001.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "sys", "i", "j", "cls", "conf", "model", "truth_state", "correct", "tag", "strategy"])
                for t, i, j, ok in [(-3, 9, 9, 0), (10, 5, 5, 0), (12, 5, 5, 0), (14, 5, 5, 0), (16, 6, 6, 0), (20, 3, 2, 1), (22, 3, 2, 1)]:
                    w.writerow([t, 1, i, j, "smoke", 0.5, "m", 0, ok, "t", "s"])
            e = T.extra_rows([{"run": "l3-B2s0-2001"}], tmp)[0]
            self.assertEqual((e["det_false_obs"], e["det_false_cells"], e["det_true_obs"], e["det_true_cells"]), (4, 2, 2, 1))


class TestSeedSummary(unittest.TestCase):
    def build(self):
        rows = []
        cov = {"A0": 0.1, "A1": 0.2, "A2": 0.3, "B2s0": 0.15, "B2s1": 0.15, "B2s2": 0.15, "baseline": 0.25}
        for arm, c in cov.items():
            for s in SCEN[:2]:
                rows.append(mk(arm, s, visited_frac=c, first_true_det_s=None if arm == "baseline" else 100,
                               ign_first_visit_s=(None if (arm == "A0" and s == SCEN[1]) else 100)))
        return rows

    def test_family_mean_and_sd_over_seeds_and_baseline_na(self):
        rows = self.build()
        with tempfile.TemporaryDirectory() as out:
            tab, fam = T.seed_summary(rows, extras(rows), out)
            a = [f for f in fam if f["family"] == "A"][0]
            self.assertAlmostEqual(a["visited_frac_mean"], 0.2)
            self.assertAlmostEqual(a["visited_frac_sd"], 0.1)                      # ddof = 1: sqrt(((.1)^2+0+(.1)^2)/2); ddof = 0 would give 0.0816
            b = [f for f in fam if f["family"] == "B"][0]
            self.assertAlmostEqual(b["visited_frac_sd"], 0.0)
            c = {r["arm"]: r for r in csv_rows(os.path.join(out, "seed_summary.csv"))}
            self.assertEqual((c["baseline"]["first_true_det_s"], c["baseline"]["first_true_det_s_n_observed"]), ("NA", "NA"))
            self.assertEqual(c["A0"]["first_true_det_s"], "100.0")
            # censoring bookkeeping: A0 has one of two ignition-cell visits unobserved -> lower-bound mean (100 + 600)/2 and 1 observed
            self.assertAlmostEqual(float(c["A0"]["ign_first_visit_s"]), 350.0)
            self.assertEqual(c["A0"]["ign_first_visit_s_n_observed"], "1")
            self.assertEqual(tab["baseline"]["det_false"], 0.0)


class TestClusterStats(unittest.TestCase):
    def run_stats(self, fa, fb, fbase=None, era=None, metric_kw="first_true_det_s"):
        """fa/fb/fbase: scenario -> value of `metric_kw` for family A / B / baseline."""
        rows = []
        for s in SCEN:
            for arm in ("A0", "A1", "A2"):
                rows.append(mk(arm, s, **{metric_kw: fa[s]}))
            for arm in ("B2s0", "B2s1", "B2s2"):
                rows.append(mk(arm, s, **{metric_kw: fb[s]}))
            rows.append(mk("baseline", s, **{metric_kw: (fbase or fa)[s]}))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        T.cluster_stats(rows, extras(rows), tmp.name, era=era)
        res = json.load(open(os.path.join(tmp.name, "cluster_stats.json")))
        return {(r["metric"], r["contrast"]): r for r in res["results"]}, res, tmp.name

    def test_cell_level_averaging_and_direction_counts(self):
        # A-B scenario differences = fa (fb = 0). Cell means: (3,2): (100+300)/2=200, (2,1): 50, (1,0): 10, (0,1): -20
        fa = {2001: 100.0, 2005: 300.0, 2002: 40.0, 2012: 60.0, 2004: 10.0, 2014: 10.0, 2006: -30.0, 2015: -10.0}
        fb = {s: 0.0 for s in SCEN}
        r, _, _ = self.run_stats(fa, fb)
        ab = r[("first_true_det_s", "A-B")]
        self.assertEqual(ab["cell_diffs"], {"0,1": -20.0, "1,0": 10.0, "2,1": 50.0, "3,2": 200.0})
        self.assertAlmostEqual(ab["mean_diff"], (200 + 50 + 10 - 20) / 4)
        self.assertEqual((ab["cells_pos"], ab["cells_neg"], ab["cells_zero"]), (3, 1, 0))
        self.assertEqual((ab["scenarios_pos"], ab["scenarios_neg"]), (6, 2))
        lo, hi = ab["cell_cluster_boot_ci95"]
        self.assertGreaterEqual(lo, -20.0)
        self.assertLessEqual(hi, 200.0)
        # exact sign-flip p over the four cell differences, counted independently
        cd = [-20.0, 10.0, 50.0, 200.0]
        n_ge = sum(1 for sg in itertools.product((1, -1), repeat=4) if abs(sum(s * v for s, v in zip(sg, cd))) >= abs(sum(cd)) - 1e-12)
        self.assertAlmostEqual(ab["p_signflip_cells_unadj"], n_ge / 16)
        self.assertEqual(len(ab["scenario_diffs"]), 8)

    def test_uniform_effect_hits_the_four_cluster_p_floor_and_holm_cannot_help(self):
        fa = {s: 100.0 for s in SCEN}
        fb = {s: 200.0 for s in SCEN}
        r, res, _ = self.run_stats(fa, fb)
        ab = r[("first_true_det_s", "A-B")]
        self.assertAlmostEqual(ab["mean_diff"], -100.0)
        self.assertEqual(ab["cell_cluster_boot_ci95"], [-100.0, -100.0])
        self.assertEqual((ab["cells_neg"], ab["scenarios_neg"]), (4, 8))
        self.assertAlmostEqual(ab["p_signflip_cells_unadj"], 0.125)               # 2 / 2^4: cannot be below 0.125 with four clusters
        self.assertTrue(all(x["p_signflip_cells_unadj"] >= 0.125 - 1e-12 and x["p_cells_holm"] >= 0.125 - 1e-12 for x in res["results"]))
        self.assertNotIn(("first_true_det_s", "A-baseline"), r)                    # detection contrast against the no-detector baseline is not computed

    def test_baseline_contrasts_use_family_mean_over_seeds(self):
        fa = {s: 0.20 for s in SCEN}
        fb = {s: 0.10 for s in SCEN}
        fbase = {s: 0.30 for s in SCEN}
        r, _, _ = self.run_stats(fa, fb, fbase, metric_kw="visited_frac")
        self.assertAlmostEqual(r[("visited_frac", "A-baseline")]["mean_diff"], -0.10)
        self.assertAlmostEqual(r[("visited_frac", "B-baseline")]["mean_diff"], -0.20)
        self.assertAlmostEqual(r[("visited_frac", "A-B")]["mean_diff"], 0.10)

    def test_censored_runs_enter_as_600_lower_bounds_in_contrasts(self):
        # every B run has no ignition-cell visit (censored) while A visits at 100 s: A-B = 100 - 600 in every scenario
        rows = []
        for s in SCEN:
            rows += [mk(a, s, ign_first_visit_s=100) for a in ("A0", "A1", "A2", "baseline")] + [mk(a, s, ign_first_visit_s=None) for a in ("B2s0", "B2s1", "B2s2")]
        with tempfile.TemporaryDirectory() as out:
            T.cluster_stats(rows, extras(rows), out)
            r = {(x["metric"], x["contrast"]): x for x in json.load(open(os.path.join(out, "cluster_stats.json")))["results"]}
            self.assertAlmostEqual(r[("ign_first_visit_s", "A-B")]["mean_diff"], -500.0)

    def test_direction_counts_and_era_subsets(self):
        fa = {2001: 5.0, 2005: 5.0, 2002: 5.0, 2012: -5.0, 2004: 5.0, 2014: -5.0, 2006: -5.0, 2015: -5.0}
        fb = {s: 0.0 for s in SCEN}
        era = {2001: "seq", 2002: "seq", 2004: "seq", 2005: "mixed", 2006: "par", 2012: "par", 2014: "par", 2015: "par"}
        r, res, out = self.run_stats(fa, fb, era=era)
        dc = {(m, c): rest for m, c, *rest in T.direction_counts(list(r.values()), [2001, 2002, 2004])}
        self.assertEqual(dc[("first_true_det_s", "A-B")][:4], [3, 0, 3, 0])       # sequential subset: 3 positive scenarios, 3 positive cells
        dc_par = {(m, c): rest for m, c, *rest in T.direction_counts(list(r.values()), [2006, 2012, 2014, 2015])}
        self.assertEqual(dc_par[("first_true_det_s", "A-B")][:4], [0, 4, 0, 3])   # parallel subset: all negative
        sens = res["sensitivity_by_era"]
        self.assertEqual((sens["sequential_scenarios"], sens["parallel_scenarios"], sens["mixed_scenarios"]), ([2001, 2002, 2004], [2006, 2012, 2014, 2015], [2005]))
        self.assertEqual(sens["counts"]["first_true_det_s|A-B"]["sequential"][:4], [3, 0, 3, 0])
        self.assertIn("fire-clock", open(os.path.join(out, "cluster_stats.md")).read())

    def test_cell_value_is_the_mean_not_the_median_when_a_cell_has_three_scenarios(self):
        # the real design has 2 scenarios per cell (mean == median), so pin the averaging rule with a patched 3-scenario cell
        from unittest import mock
        cell_of = {1: (0, 0), 2: (0, 0), 3: (0, 0), 4: (1, 1)}
        fa = {1: 0.0, 2: 0.0, 3: 90.0, 4: 10.0}
        rows = []
        for sc in cell_of:
            rows += [mk(a, sc, first_true_det_s=fa[sc]) for a in ("A0", "A1", "A2")] + [mk(a, sc, first_true_det_s=0) for a in ("B2s0", "B2s1", "B2s2")] + [mk("baseline", sc)]
        with mock.patch.dict(T.CELL_OF, cell_of, clear=True), tempfile.TemporaryDirectory() as out:
            T.cluster_stats(rows, extras(rows), out)
            r = {(x["metric"], x["contrast"]): x for x in json.load(open(os.path.join(out, "cluster_stats.json")))["results"]}
        self.assertEqual(r[("first_true_det_s", "A-B")]["cell_diffs"], {"0,0": 30.0, "1,1": 10.0})
        self.assertAlmostEqual(r[("first_true_det_s", "A-B")]["mean_diff"], 20.0)

    def test_era_of_scenarios_classification(self):
        prov = [{"set": "results", "run": "l3-A0-2001", "parallel_slot": None}, {"set": "results", "run": "l3-B2s0-2001", "parallel_slot": ""},
                {"set": "results", "run": "l3-A0-2005", "parallel_slot": None}, {"set": "results", "run": "l3-B2s1-2005", "parallel_slot": 1},
                {"set": "results", "run": "l3-A0-2006", "parallel_slot": 0}, {"set": "results_repeat", "run": "l3-A0-2001", "parallel_slot": 1}]
        self.assertEqual(T.era_of_scenarios(prov), {2001: "seq", 2005: "mixed", 2006: "par"})        # repeats do not change the main-set classification


class TestCensoringTable(unittest.TestCase):
    def test_counts_and_medians(self):
        rows = [mk("A0", 2001, ign_first_visit_s=100), mk("A0", 2002, ign_first_visit_s=300), mk("A0", 2004, ign_first_visit_s=None),
                mk("B2s0", 2001, ign_first_visit_s=None, first_true_det_s=None), mk("baseline", 2001, first_true_det_s=None, ign_first_visit_s=50)]
        with tempfile.TemporaryDirectory() as out:
            T.censoring(rows, out)
            c = {r["arm"]: r for r in csv_rows(os.path.join(out, "censoring.csv"))}
        self.assertEqual((c["A0"]["ign_visit_observed"], c["A0"]["ign_visit_censored"], c["A0"]["ign_visit_observed_median_s"]), ("2", "1", "200.0"))
        self.assertEqual((c["B2s0"]["true_det_observed"], c["B2s0"]["true_det_censored"]), ("0", "1"))
        self.assertEqual(c["baseline"]["true_det_censored"], "")                         # not applicable without a detector
        self.assertEqual(c["A0"]["true_det_observed"], "3")


class TestFireClock(unittest.TestCase):
    def write(self, root, run, fire, tick_s, slot):
        d = os.path.join(root, "results", run)
        os.makedirs(d)
        with open(os.path.join(d, f"fire_cells_{run}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["i", "j", "fire_state_end", "ignite_t", "first_visit_t", "visited"])
            for (i, j), t in fire.items():
                w.writerow([i, j, 1, t, "", 0])
        with open(os.path.join(d, f"coverage_t_{run}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "visited_frac"])
            for k in range(int(100 / tick_s) + 1):
                w.writerow([round(k * tick_s, 3), 0.1])
        json.dump({"slot": 0} if slot else {"git_dirty": False}, open(os.path.join(d, "run_meta.json"), "w"))

    def test_ratio_burned_counts_tick_rate_and_baseline_reference(self):
        base = {(1, 0): 0.0, (2, 0): 10.0, (3, 0): 20.0, (4, 0): 40.0}
        slow = {(1, 0): 0.0, (2, 0): 10.5, (3, 0): 21.0, (4, 0): 42.0, (5, 0): 60.0}      # x1.05 on cells with baseline ignition > 5 s; one extra cell
        with tempfile.TemporaryDirectory() as root:
            self.write(root, "l3-baseline-2006", base, 0.5, True)
            self.write(root, "l3-A0-2006", slow, 0.55, True)
            rows = {r["run"]: r for r in T.fire_clock_rows(root)}
        self.assertAlmostEqual(rows["l3-A0-2006"]["fire_ratio_vs_baseline"], 1.05)
        self.assertEqual((rows["l3-A0-2006"]["burned_cells"], rows["l3-A0-2006"]["baseline_burned_cells"]), (5, 4))
        self.assertAlmostEqual(rows["l3-baseline-2006"]["fire_ratio_vs_baseline"], 1.0)
        self.assertAlmostEqual(rows["l3-A0-2006"]["fog_ticks_per_s"], 1 / 0.55, places=6)
        self.assertEqual(rows["l3-A0-2006"]["mode"], "par")


class TestLinkageMapping(unittest.TestCase):
    def test_level2_values_come_from_the_matching_seed_and_family(self):
        rows = [mk(a, s) for a in T.ARMS for s in SCEN[:1]]
        with tempfile.TemporaryDirectory() as out:
            tab, _ = T.seed_summary(rows, extras(rows), out)
            T.linkage(rows, tab, out, None)
            link = {r["arm"]: r for r in csv_rows(os.path.join(out, "linkage.csv"))}
        rep = json.load(open(os.path.join(T.ROOT, "docs/experiments/boreal_v2/results/report.json")))["tables"]
        for arm, fam, seed in (("A1", "A", "1"), ("B2s2", "B", "2"), ("B2s0", "B", "0")):
            self.assertAlmostEqual(float(link[arm]["L2_mAP50"]), rep[f"{fam}|synth_test_pos"][seed]["mAP50"], places=9)
        thr = json.load(open(os.path.join(T.ROOT, "eval/level3_thresholds.json")))["arms"]
        self.assertEqual(float(link["A2"]["level3_tau"]), thr["A2"]["tau"])


@unittest.skipUnless(os.path.isdir(os.path.join(T.ROOT, "local_data", "level3", "final", "results")), "raw runs not present")
class TestRegeneration(unittest.TestCase):
    def test_tables_regenerate_identically_from_recorded_artifacts(self):
        committed = os.path.join(T.ROOT, "docs", "experiments", "level3_final")
        with tempfile.TemporaryDirectory() as out:
            with open(os.path.join(committed, "per_run.csv")) as src, open(os.path.join(out, "per_run.csv"), "w") as dst:
                dst.write(src.read())
            import sys
            old = sys.argv
            sys.argv = ["x", "--out", out]
            try:
                T.main()
            finally:
                sys.argv = old
            for name in ("seed_summary.csv", "seed_summary.md", "cluster_stats.json", "cluster_stats.md", "censoring.csv", "censoring.md", "linkage.csv", "linkage.md",
                         "provenance.csv", "per_run_extra.csv", "fire_clock.csv", "fire_clock.md"):
                self.assertEqual(open(os.path.join(out, name)).read(), open(os.path.join(committed, name)).read(), name)


if __name__ == "__main__":
    unittest.main()
