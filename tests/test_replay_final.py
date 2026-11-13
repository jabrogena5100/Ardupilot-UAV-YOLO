"""Replay dashboard vs the recorded FINAL Level 3 runs.

Synthetic-tree tests always run. Tests against the real local_data/level3/final tree are skipped
when that (gitignored) data is absent, so CI on a clean checkout stays green; on the project
machine they verify run selection, timeline, coordinate mapping and displayed metrics against
the recorded artifacts. Nothing here writes to the results tree.
"""
import csv
import glob
import json
import math
import os
import statistics
import unittest

from dashboard import replay_data as rd

FINAL = os.path.join(rd.L3, "final")
PER_RUN = os.path.join(rd.ROOT, "docs", "experiments", "level3_final", "per_run.csv")
HAVE = os.path.isdir(os.path.join(FINAL, "results")) and os.path.exists(PER_RUN)
ARMS = ["baseline", "A0", "A1", "A2", "B2s0", "B2s1", "B2s2"]


def _per_run():
    return {r["tag"]: r for r in csv.DictReader(open(PER_RUN))}


@unittest.skipUnless(HAVE, "final Level 3 data not present")
class FinalRunsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.disc = rd.discover()
        cls.final = [r for r in cls.disc["runs"] if r["pilot"] == "final"]
        cls.pr = _per_run()

    def test_selection_is_the_56_matrix_runs(self):
        mx = json.load(open(os.path.join(FINAL, "matrix.json")))
        want = {(str(c["scenario"]), c["arm"]) for c in mx["runs"]}
        got = {(r["scenario"], r["arm"]) for r in self.final}
        self.assertEqual(len(self.final), 56)
        self.assertEqual(got, want)                      # no run dropped, none invented, no duplicates
        self.assertTrue(all(r["status"] == "completed" for r in self.final))
        self.assertFalse([q for q in self.disc["planned"] if q["pilot"] == "final"])
        self.assertEqual(len({r["id"] for r in self.final}), 56)

    def test_repeats_are_not_listed_as_experiments(self):
        self.assertFalse([r for r in self.final if "repeat" in r["dir"]])

    def test_every_scenario_has_all_seven_arms(self):
        by = {}
        for r in self.final:
            by.setdefault(r["scenario"], set()).add(r["arm"])
        self.assertEqual(len(by), 8)
        for scn, arms in by.items():
            self.assertEqual(arms, set(ARMS), scn)

    def test_recorded_metrics_match_per_run_table_and_raw_files(self):
        for r in self.final:
            d = rd.load_run(r["id"])
            rec, row = d["recorded"], self.pr[r["tag"]]
            self.assertIsNotNone(rec, r["id"])
            # recorded values equal the analysis table
            for k_rec, k_csv in (("det_true", "det_true"), ("det_false", "det_false"), ("visited_frac", "visited_frac"),
                                 ("inv_true_s", "inv_true_s"), ("inv_false_s", "inv_false_s")):
                self.assertAlmostEqual(rec[k_rec], float(row[k_csv]), places=6, msg=f"{r['id']} {k_rec}")
            # ...and agree with the raw per-run files the UI also draws from
            tp = [q for q in d["detections"] if q["correct"] == 1 and 0 <= q["t"] <= 600.0]
            fp = [q for q in d["detections"] if q["correct"] == 0 and 0 <= q["t"] <= 600.0]
            self.assertEqual(len(tp), int(rec["det_true"]), r["id"])
            self.assertEqual(len(fp), int(rec["det_false"]), r["id"])
            if tp:
                self.assertAlmostEqual(min(q["t"] for q in tp), rec["first_true_det_s"], places=2, msg=r["id"])
            ign = [c for c in d["fire"] if c["ignite_t"] <= 0.5]
            self.assertEqual(len(ign), 1, r["id"])
            self.assertEqual([list(x) for x in d["meta"]["ignition"]], [[ign[0]["i"], ign[0]["j"]]], r["id"])
            v = ign[0]["first_visit_t"]
            if v is not None and v <= 600.0:
                self.assertAlmostEqual(v, rec["ign_first_visit_s"], places=2, msg=r["id"])
            else:
                self.assertTrue(rec["ign_first_visit_censored"], r["id"])
            self.assertEqual(int(rec["fire_cells"]), len(d["fire"]), r["id"])

    def test_censoring_is_flagged_not_imputed(self):
        for r in self.final:
            rec = rd.load_run(r["id"])["recorded"]
            if r["arm"] == "baseline":                   # no detector: first detection does not exist (n/a), it is NOT 600 s
                self.assertTrue(rec["first_true_det_na"])
                self.assertIsNone(rec["first_true_det_capped_s"])
                self.assertFalse(rec["first_true_det_censored"])
            elif rec["first_true_det_s"] is None:
                self.assertTrue(rec["first_true_det_censored"])
                self.assertEqual(rec["first_true_det_capped_s"], 600.0)
            else:
                self.assertFalse(rec["first_true_det_censored"])
                self.assertEqual(rec["first_true_det_capped_s"], rec["first_true_det_s"])
        # documented count: 15 of 56 runs have no observed ignition-cell visit (level3_final.md sec. 6)
        n = sum(1 for r in self.final if rd.load_run(r["id"])["recorded"]["ign_first_visit_censored"])
        self.assertEqual(n, 15)

    def test_investigation_targets_agree_with_recorded_counts(self):
        for r in self.final:
            d = rd.load_run(r["id"])
            inv, rec = d["investigation"], d["recorded"]
            self.assertIsNotNone(inv, r["id"])
            self.assertFalse(inv["timed"])               # targets are never placed on the time axis
            self.assertEqual(len(inv["targets"]), int(rec["n_investigate_targets"]), r["id"])
            near = sum(t["hud_lines"] for t in inv["targets"] if t["near_fire"])
            far = sum(t["hud_lines"] for t in inv["targets"] if not t["near_fire"])
            self.assertEqual((near, far), (int(rec["inv_true_s"]), int(rec["inv_false_s"])), r["id"])
            if r["arm"] == "baseline":
                self.assertEqual(inv["targets"], [])

    def test_trajectory_timeline_and_coordinates(self):
        checked = 0
        for r in self.final:
            d = rd.load_run(r["id"])
            tj = d["trajectory"]
            self.assertIsNotNone(tj, f"{r['id']}: run scripts/replay_build first")
            ts = [s[0] for s in tj["samples"]]
            self.assertEqual(ts, sorted(ts))             # monotone timeline, no reordering
            self.assertGreaterEqual(ts[0], 0.0)
            self.assertLessEqual(ts[-1], 620.0)          # the tlog runs a few seconds past the 600 s window; the UI clock stops at 600
            self.assertEqual(tj["gaps"], [])
            N, side = d["meta"]["N"], d["meta"]["side_m"]
            self.assertEqual(N, 10)
            self.assertAlmostEqual(side, 1609.344, places=3)
            first_in = {}
            for t, n, e, *_ in tj["samples"]:
                c = rd_cell(n, e, side, N)
                if 0 <= c[0] < N and 0 <= c[1] < N and c not in first_in:
                    first_in[c] = t
            # the cell index from telemetry must agree with fog's own first-visit times (median within 10 s)
            res = [first_in[(c["i"], c["j"])] - c["first_t"] for c in d["cells"]
                   if c["first_t"] is not None and (c["i"], c["j"]) in first_in]
            self.assertGreater(len(res), 5, r["id"])
            self.assertLess(abs(statistics.median(res)), 10.0, r["id"])
            checked += 1
        self.assertEqual(checked, 56)

    def test_unvisited_cells_have_no_visit_time(self):
        for r in self.final[::9]:
            for c in rd.load_run(r["id"])["cells"]:
                self.assertEqual(c["first_t"] is None, not c["visited"], r["id"])

    def test_coverage_curve_matches_cell_first_visits(self):
        d = rd.load_run("final/l3-A0-2001")
        cov = d["coverage_t"]
        for t, vf in list(zip(cov["t"], cov["visited_frac"]))[::40]:
            n = sum(1 for c in d["cells"] if c["first_t"] is not None and c["first_t"] <= t)
            self.assertAlmostEqual(n / 100.0, vf, delta=0.021, msg=f"t={t}")

    def test_fire_timeline_reproduces_logged_counts(self):
        for r in self.final[::7]:
            d = rd.load_run(r["id"])
            fc = d["fire_check"]
            self.assertIsNotNone(fc, r["id"])
            # per-cell ignition times vs the logged counts: step-boundary rounding only
            self.assertLessEqual(fc["max_abs_diff_burning_cells"], 2, r["id"])
            self.assertLessEqual(fc["max_abs_diff_burnt_cells"], 2, r["id"])

    def test_oracle_off_and_single_uav_in_every_run(self):
        for r in self.final:
            d = rd.load_run(r["id"])
            self.assertFalse(d["meta"]["oracle_hazards"], r["id"])
            self.assertEqual(d["end"]["agents_n"], "1", r["id"])

    def test_baseline_has_no_detector_and_says_so(self):
        d = rd.load_run("final/l3-baseline-2001")
        self.assertFalse(d["meta"]["detector_enabled"])
        self.assertIn("detections", d["missing"])
        self.assertEqual(d["detections"], [])

    def test_matched_scenario_shares_initial_ignition(self):
        by = {}
        for r in self.final:
            d = rd.load_run(r["id"])
            by.setdefault(r["scenario"], set()).add(json.dumps(d["meta"]["ignition"]))
        for scn, s in by.items():
            self.assertEqual(len(s), 1, f"scenario {scn} has arms with different ignition cells")


def rd_cell(n, e, side, N):
    """Same rule as dashboard/replay_build.py and neCell() in static/replay.js."""
    c = side / N
    return (int(math.floor((e + side / 2) / c)), int(math.floor((n + side / 2) / c)))


class CoordinateMappingTest(unittest.TestCase):
    def test_cardinal_directions(self):
        side, N = 1609.344, 10
        self.assertEqual(rd_cell(0.0, 0.0, side, N), (5, 5))             # grid centre
        self.assertEqual(rd_cell(-side / 2 + 1, -side / 2 + 1, side, N), (0, 0))   # south-west corner cell
        self.assertEqual(rd_cell(side / 2 - 1, side / 2 - 1, side, N), (9, 9))     # north-east corner cell
        self.assertEqual(rd_cell(side / 2 - 1, -side / 2 + 1, side, N), (0, 9))    # north = +j, west = low i
        self.assertEqual(rd_cell(-side / 2 + 1, side / 2 - 1, side, N), (9, 0))    # east = +i

    def test_js_mapping_matches_python_rule(self):
        js = open(os.path.join(os.path.dirname(rd.__file__), "static", "replay.js")).read()
        # the JS helpers are the single place pixels are computed; pin their formulas
        self.assertIn("((e + side / 2) / side) * W, (1 - (n + side / 2) / side) * W", js)
        self.assertIn("[i * C, (N - 1 - j) * C]", js)
        self.assertIn("Math.floor((e + side / 2) / c), Math.floor((n + side / 2) / c)", js)


if __name__ == "__main__":
    unittest.main()
