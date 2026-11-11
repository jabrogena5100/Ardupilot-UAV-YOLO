import hashlib, os, sys, tempfile, unittest, zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "external_scripts"))
import extract_fasdd_positive as X


class TestExtract(unittest.TestCase):
    def make(self, d):
        zp = os.path.join(d, "f.zip")
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("annotations/YOLO_UAV/test.txt", "./images/smoke_UAV1.jpg\n./images/bothFireAndSmoke_UAV2.jpg\n./images/fire_UAV3.jpg\n./images/neitherFireNorSmoke_UAV4.jpg\n")
            for n, lab in (("smoke_UAV1", "1 0.5 0.5 0.2 0.2\n"), ("bothFireAndSmoke_UAV2", "0 0.1 0.1 0.1 0.1\n1 0.6 0.6 0.4 0.4\n"), ("fire_UAV3", "0 0.2 0.2 0.1 0.1\n"), ("neitherFireNorSmoke_UAV4", "")):
                z.writestr(f"images/{n}.jpg", n.encode()); z.writestr(f"annotations/YOLO_UAV/labels/{n}.txt", lab)
        lst = os.path.join(d, "list.txt"); open(lst, "w").write("smoke_UAV1.jpg\nbothFireAndSmoke_UAV2.jpg\n")
        return zp, lst

    def test_extract_and_class_remap(self):
        with tempfile.TemporaryDirectory() as d:
            zp, lst = self.make(d)
            r = X.extract(zp, os.path.join(d, "out"), pinned_list=lst, hashes=None, expect_boxes=2)
            self.assertEqual(r, {"images": 2, "smoke_boxes": 2})
            self.assertEqual(open(os.path.join(d, "out/labels_smoke_only/bothFireAndSmoke_UAV2.txt")).read(), "0 0.6 0.6 0.4 0.4\n")      # fire dropped, class 1 -> 0, coordinates untouched
            self.assertEqual(open(os.path.join(d, "out/labels_original/bothFireAndSmoke_UAV2.txt")).read().count("\n"), 2)             # original kept
            self.assertFalse(os.path.exists(os.path.join(d, "out/images/fire_UAV3.jpg")))
            with self.assertRaises(SystemExit):
                X.extract(zp, os.path.join(d, "out"), pinned_list=lst, hashes=None, expect_boxes=2)                                       # no overwrite

    def test_wrong_box_count_fails_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            zp, lst = self.make(d)
            with self.assertRaises(SystemExit):
                X.extract(zp, os.path.join(d, "out"), pinned_list=lst, hashes=None, expect_boxes=3)
            self.assertTrue(os.path.isdir(os.path.join(d, "out_FAILED"))); self.assertFalse(os.path.exists(os.path.join(d, "out")))

    def test_list_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as d:
            zp, lst = self.make(d); open(lst, "w").write("smoke_UAV1.jpg\n")
            with self.assertRaises(SystemExit):
                X.extract(zp, os.path.join(d, "out"), pinned_list=lst, hashes=None, expect_boxes=2)


if __name__ == "__main__":
    unittest.main()
