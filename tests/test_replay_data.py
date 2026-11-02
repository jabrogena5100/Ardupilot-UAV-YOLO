"""dashboard/replay_data.py: run classification and missing-data reporting on a synthetic results tree."""
import json
import os
import tempfile
import unittest

from dashboard import replay_data as rd


def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


class ReplayDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = (rd.L3, rd.CACHE)
        rd.L3 = self.tmp.name
        rd.CACHE = os.path.join(self.tmp.name, "_cache")
        p = os.path.join(rd.L3, "pilot2")
        _w(os.path.join(p, "matrix.json"), json.dumps({"runs": [
            {"arm": "A0", "scenario": 1001}, {"arm": "A0", "scenario": 1003}]}))
        ok = os.path.join(p, "results", "l3-A0-1001")
        _w(os.path.join(ok, "experiment_end_l3-A0-1001.txt"), "reason=timeout\ndet_true=3\ndet_false=1\n")
        _w(os.path.join(ok, "manifest.json"), json.dumps({"fog_exit_code": 0}))
        _w(os.path.join(ok, "detections_l3-A0-1001.csv"),
           "t,sys,i,j,cls,conf,model,truth_state,correct,tag,strategy\n5.0,1,2,2,smoke,0.9,m,1,1,x,l3\n")
        bad = os.path.join(p, "runs", "20260101-000000-l3-A0-1002")
        _w(os.path.join(bad, "experiment_end_l3-A0-1002.txt"), "")  # crash: empty file

    def tearDown(self):
        rd.L3, rd.CACHE = self._old
        self.tmp.cleanup()

    def test_classification(self):
        d = rd.discover()
        st = {r["tag"]: r["status"] for r in d["runs"]}
        self.assertEqual(st["l3-A0-1001"], "completed")
        self.assertEqual(st["l3-A0-1002"], "interrupted")
        self.assertEqual([(q["arm"], q["scenario"]) for q in d["planned"]], [("A0", "1003")])

    def test_missing_is_reported_not_filled(self):
        r = rd.load_run("pilot2/l3-A0-1001")
        self.assertIsNone(r["fire"])
        self.assertIsNone(r["trajectory"])
        self.assertIn("fire", r["missing"])
        self.assertIn("trajectory", r["missing"])
        self.assertEqual(len(r["detections"]), 1)

    def test_rejects_path_tricks(self):
        self.assertIsNone(rd.load_run("pilot2/../x"))
        self.assertIsNone(rd.load_run("nope/l3-A0-1001"))


if __name__ == "__main__":
    unittest.main()
