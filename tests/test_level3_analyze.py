"""Unit tests for eval/level3_analyze.py on small SYNTHETIC run artifacts with hand-computed expected values.

These tests exercise the calculations (not committed outputs): detection scoring columns, first true detection, ignition-cell first visit, right-censoring,
coverage, INVESTIGATE-line classification, family mapping and the paired statistics. Passing them shows the code does what its docstring says on known inputs;
it does not establish that the experiment is scientifically valid.
"""
import csv
import itertools
import os
import tempfile
import unittest

import numpy as np

from eval import level3_analyze as A

DET_HEADER = ["t", "sys", "i", "j", "cls", "conf", "model", "truth_state", "correct", "tag", "strategy"]
FIRE_HEADER = ["i", "j", "fire_state_end", "ignite_t", "first_visit_t", "visited"]


def write_run(root, tag, fire, dets, log_lines=None, visited_frac=0.37):
    """fire: [(i, j, ignite_t, first_visit_t or '', visited 0/1)]; dets: [(t, i, j, correct)]; log_lines: INVESTIGATE target cells (one HUD line each)."""
    d = os.path.join(root, tag)
    os.makedirs(d)
    with open(os.path.join(d, f"experiment_end_{tag}.txt"), "w") as f:
        f.write(f"reason=timeout\ntime_s=600.4\nvisited_frac={visited_frac}\ntag={tag}\n")
    with open(os.path.join(d, f"fire_cells_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIRE_HEADER)
        for i, j, ign, fv, vis in fire:
            w.writerow([i, j, 1, ign, fv, vis])
    with open(os.path.join(d, f"detections_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(DET_HEADER)
        for t, i, j, ok in dets:
            w.writerow([t, 1, i, j, "smoke", 0.9, "m", 1, ok, tag, "l3"])
    if log_lines is not None:
        with open(os.path.join(d, "agent_A.log"), "w") as f:
            f.write("[udp|sys1] v=(0,0,0) mode=TRAVERSE→0,0 exp=x strat=l3 cell=0,0 traverse=0% tgt=0,0\n")      # not an INVESTIGATE line
            for c in log_lines:
                f.write(f"[udp|sys1] v=(1,0,0) mode=INVESTIGATE→{c[0]},{c[1]} exp=x strat=l3 cell=2,2 traverse=5% tgt={c[0]},{c[1]}\n")
    return d


class TestRunMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_full_known_example(self):
        fire = [(3, 2, 0.0, 120.5, 1), (3, 3, 50.0, "", 0), (4, 2, 80.0, 300.0, 1)]
        dets = [(-5.0, 3, 2, 1), (-7.0, 8, 8, 0),      # before the start gate (one would-be true, one would-be false): ignored by every count
                (60.0, 0, 0, 0), (150.0, 7, 7, 0),       # two false
                (100.0, 3, 2, 1), (90.0, 3, 3, 1)]       # two true, deliberately unsorted: first true detection is the MIN time
        targets = [(3, 2)] * 3 + [(4, 3)] + [(0, 0)] * 2 + [(7, 7)] + [(5, 4)]      # (5,4) is at Chebyshev distance exactly 2 from every burned cell
        d = write_run(self.tmp.name, "l3-A0-2001", fire, dets, targets, visited_frac=0.37)
        m = A.run_metrics(d)
        self.assertEqual((m["arm"], m["family"], m["scenario"]), ("A0", "A", 2001))
        self.assertEqual(m["first_true_det_s"], 90.0)
        self.assertEqual((m["det_true"], m["det_false"]), (2, 2))
        self.assertEqual(m["ign_first_visit_s"], 120.5)            # the cell with ignite_t <= 0.5, not the earliest-visited cell
        self.assertEqual((m["fire_cells"], m["fire_cells_visited"]), (3, 2))
        self.assertAlmostEqual(m["fire_cells_visited_frac"], 2 / 3)
        self.assertAlmostEqual(m["visited_frac"], 0.37)
        # INVESTIGATE: (3,2) x3 and (4,3) (Chebyshev 1 of burned (3,3)/(4,2)/(3,2)) are near burned cells; (0,0) x2, (7,7) and (5,4) (distance 2) are not
        self.assertEqual((m["inv_true_s"], m["inv_false_s"], m["n_investigate_targets"]), (4, 4, 5))

    def test_censoring_when_ignition_cell_never_visited_and_no_true_detection(self):
        fire = [(1, 0, 0.0, "", 0), (1, 1, 30.0, 400.0, 1)]
        d = write_run(self.tmp.name, "l3-B2s1-2004", fire, [(50.0, 5, 5, 0)], [], visited_frac=0.1)
        m = A.run_metrics(d)
        self.assertIsNone(m["ign_first_visit_s"])
        self.assertIsNone(m["first_true_det_s"])
        self.assertEqual(A.censored(m, "ign_first_visit_s"), A.T_END)
        self.assertEqual(A.censored(m, "first_true_det_s"), A.T_END)
        self.assertEqual(A.censored(m, "visited_frac"), 0.1)       # non-time metrics are never censored
        self.assertEqual((m["det_true"], m["det_false"], m["family"]), (0, 1, "B"))

    def test_baseline_has_no_detections_and_no_log(self):
        d = write_run(self.tmp.name, "l3-baseline-2002", [(2, 1, 0.0, 87.0, 1)], [], None)
        m = A.run_metrics(d)
        self.assertEqual((m["family"], m["det_true"], m["det_false"], m["inv_true_s"], m["inv_false_s"]), ("baseline", 0, 0, 0, 0))
        self.assertIsNone(m["first_true_det_s"])
        self.assertEqual(m["ign_first_visit_s"], 87.0)

    def test_only_cells_with_ignite_t_at_most_half_second_count_as_ignition_cell(self):
        # two cells burn; the one igniting at 0.4 s (clock jitter) is the ignition cell, the one at 0.6 s is not
        d = write_run(self.tmp.name, "l3-A1-2005", [(3, 3, 0.6, 5.0, 1), (3, 2, 0.4, 111.0, 1)], [], None)      # the non-ignition cell is listed first on purpose
        self.assertEqual(A.run_metrics(d)["ign_first_visit_s"], 111.0)

    def test_family_mapping_covers_all_seven_arms(self):
        self.assertEqual(A.FAMILY, {"baseline": "baseline", "A0": "A", "A1": "A", "A2": "A", "B2s0": "B", "B2s1": "B", "B2s2": "B"})
        for arm, fam in A.FAMILY.items():
            d = write_run(self.tmp.name, f"l3-{arm}-2012", [(2, 1, 0.0, 10.0, 1)], [], None)
            self.assertEqual(A.run_metrics(d)["family"], fam)

    def test_false_detection_flag_is_read_from_the_correct_column(self):
        # a detection with correct=0 must never be a true detection, even at an early time
        d = write_run(self.tmp.name, "l3-A2-2006", [(0, 1, 0.0, 9.0, 1)], [(1.0, 9, 9, 0), (200.0, 0, 1, 1)], [])
        m = A.run_metrics(d)
        self.assertEqual((m["first_true_det_s"], m["det_false"], m["det_true"]), (200.0, 1, 1))


def row(arm, scenario, **kw):
    r = {"run": f"l3-{arm}-{scenario}", "arm": arm, "family": A.FAMILY[arm], "scenario": scenario, "first_true_det_s": 100.0, "ign_first_visit_s": 100.0,
         "visited_frac": 0.2, "det_false": 0, "inv_false_s": 0, "inv_true_s": 0}
    r.update(kw)
    return r


class TestStatistics(unittest.TestCase):
    def test_family_by_scenario_is_mean_over_seeds_and_censors(self):
        rows = [row("A0", 1, ign_first_visit_s=100.0), row("A1", 1, ign_first_visit_s=200.0), row("A2", 1, ign_first_visit_s=None), row("B2s0", 1, ign_first_visit_s=50.0)]
        got = A.family_by_scenario(rows, "A", "ign_first_visit_s")
        self.assertAlmostEqual(got[1], (100 + 200 + 600) / 3)       # missing visit enters as the 600 s censor value (a lower bound)
        self.assertEqual(A.family_by_scenario(rows, "B", "ign_first_visit_s"), {1: 50.0})

    def test_paired_all_positive_has_minimum_sign_flip_p(self):
        d = list(range(1, 9))
        x = {s: float(v) for s, v in enumerate(d)}
        y = {s: 0.0 for s in x}
        r = A.paired(x, y)
        self.assertEqual((r["n_scenarios"], r["n_pos"], r["n_neg"], r["n_zero"]), (8, 8, 0, 0))
        self.assertAlmostEqual(r["mean_diff"], 4.5)
        self.assertAlmostEqual(r["p_signflip"], 2 / 2 ** 8)
        self.assertLess(r["ci95"][0], 4.5)
        self.assertGreater(r["ci95"][1], 4.5)
        self.assertGreaterEqual(r["ci95"][0], 1.0)
        self.assertLessEqual(r["ci95"][1], 8.0)

    def test_paired_p_matches_brute_force_enumeration(self):
        d = [3.0, -1.0, 2.0, 0.0, 5.0]
        x = {s: v for s, v in enumerate(d)}
        y = {s: 0.0 for s in x}
        total = abs(sum(d))
        n_ge = sum(1 for signs in itertools.product((1, -1), repeat=len(d)) if abs(sum(s * v for s, v in zip(signs, d))) >= total - 1e-12)
        self.assertAlmostEqual(A.paired(x, y)["p_signflip"], n_ge / 2 ** len(d))

    def test_paired_symmetric_differences_give_p_one_and_zero_counts(self):
        x = {1: 1.0, 2: -1.0, 3: 0.0, 4: 0.0}
        y = {s: 0.0 for s in x}
        r = A.paired(x, y)
        self.assertEqual((r["n_pos"], r["n_neg"], r["n_zero"]), (1, 1, 2))
        self.assertAlmostEqual(r["mean_diff"], 0.0)
        self.assertAlmostEqual(r["p_signflip"], 1.0)

    def test_paired_uses_only_common_scenarios_and_is_deterministic(self):
        x = {1: 2.0, 2: 4.0, 99: 1000.0}
        y = {1: 1.0, 2: 1.0}
        r1, r2 = A.paired(x, y), A.paired(x, y)
        self.assertEqual(r1["n_scenarios"], 2)
        self.assertAlmostEqual(r1["mean_diff"], 2.0)
        self.assertEqual(r1, r2)
        self.assertIsNone(A.paired({1: 1.0}, {2: 1.0}))

    def test_contrasts_signs_and_baseline_detection_contrast_undefined(self):
        rows = []
        for s in (1, 2):
            rows += [row("baseline", s, visited_frac=0.30, first_true_det_s=None, ign_first_visit_s=50.0),
                     row("A0", s, visited_frac=0.20, first_true_det_s=100.0), row("A1", s, visited_frac=0.20, first_true_det_s=100.0), row("A2", s, visited_frac=0.20, first_true_det_s=100.0),
                     row("B2s0", s, visited_frac=0.10, first_true_det_s=250.0), row("B2s1", s, visited_frac=0.10, first_true_det_s=250.0), row("B2s2", s, visited_frac=0.10, first_true_det_s=250.0)]
        c = A.contrasts(rows)
        self.assertIsNone(c["first_true_det_s"]["A-baseline"])
        self.assertIsNone(c["first_true_det_s"]["B-baseline"])
        self.assertAlmostEqual(c["first_true_det_s"]["A-B"]["mean_diff"], -150.0)
        self.assertAlmostEqual(c["visited_frac"]["A-baseline"]["mean_diff"], -0.10)
        self.assertAlmostEqual(c["visited_frac"]["B-baseline"]["mean_diff"], -0.20)
        self.assertAlmostEqual(c["visited_frac"]["A-B"]["mean_diff"], 0.10)
        self.assertEqual(c["visited_frac"]["A-B"]["n_pos"], 2)
        self.assertEqual(c["first_true_det_s"]["family_means"], {"baseline": 600.0, "A": 100.0, "B": 250.0})   # baseline censored value is shown but never contrasted

    def test_seed_table_mean_sd_and_censoring(self):
        rows = [row("A0", s, visited_frac=v, ign_first_visit_s=t) for s, v, t in ((1, 0.1, 100.0), (2, 0.2, None), (3, 0.3, 300.0))]
        t = A.seed_table(rows)
        self.assertAlmostEqual(t["visited_frac"]["A0"]["mean"], 0.2)
        self.assertAlmostEqual(t["visited_frac"]["A0"]["sd"], 0.1)                 # sample SD, ddof = 1
        self.assertAlmostEqual(t["ign_first_visit_s"]["A0"]["mean"], (100 + 600 + 300) / 3)
        self.assertEqual(t["ign_first_visit_s"]["A0"]["n"], 3)


if __name__ == "__main__":
    unittest.main()
