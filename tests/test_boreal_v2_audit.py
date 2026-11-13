import json
import os
import tempfile
import unittest

from eval import boreal_v2_audit as A


class AuditTests(unittest.TestCase):
    def test_audit_passes_on_committed_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "audit.json")
            A.out.clear()
            rc = A.main(["--out", out])
            d = json.load(open(out))
        self.assertEqual(rc, 0)
        self.assertEqual(d["summary"]["FAILED"], 0)
        self.assertGreaterEqual(d["summary"]["VERIFIED"], 45)
        ids = {c["id"] for c in d["checks"]}
        for need in ("ckpt.seed0.hash_recorded", "cells.count_30_full_cross_product", "split.filelist_hashes_frozen", "metrics.detection_metrics_recomputed_exact",
                     "deviation.cache_ram_documented"):
            self.assertIn(need, ids)

    def test_committed_audit_record_is_clean(self):
        d = json.load(open(os.path.join(A.EXP, "audit.json")))
        self.assertEqual(d["summary"]["FAILED"], 0)


if __name__ == "__main__":
    unittest.main()
