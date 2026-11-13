import json, os, tempfile, unittest

from eval import external_materialize as M
from eval import fireman_prep as P
from tests.test_fireman_prep import xml


class TestFasddView(unittest.TestCase):
    def test_view_links_and_pin(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src"); os.makedirs(src + "/images"); os.makedirs(src + "/labels_smoke_only")
            for n, lab in (("smoke_UAV1", "0 .5 .5 .2 .2\n"), ("bothFireAndSmoke_UAV2", "0 .1 .1 .2 .2\n"), ("neitherFireNorSmoke_UAV3", "")):
                open(f"{src}/images/{n}.jpg", "wb").write(b"x"); open(f"{src}/labels_smoke_only/{n}.txt", "w").write(lab)
            lst = os.path.join(d, "list.txt"); open(lst, "w").write("smoke_UAV1.jpg\nbothFireAndSmoke_UAV2.jpg\n")
            r = M.fasdd_view(src, lst, os.path.join(d, "out"))
            self.assertEqual(r["n_images"], 2)
            self.assertEqual(r["filelist_sha256"], M.filelist_sha256(["bothFireAndSmoke_UAV2.jpg", "smoke_UAV1.jpg"]))
            self.assertTrue(os.path.islink(os.path.join(d, "out/images/test/smoke_UAV1.jpg")))
            with self.assertRaises(FileExistsError):
                M.fasdd_view(src, lst, os.path.join(d, "out"))                      # refuses to overwrite

    def test_empty_label_in_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src"); os.makedirs(src + "/images"); os.makedirs(src + "/labels_smoke_only")
            open(f"{src}/images/a.jpg", "wb").write(b"x"); open(f"{src}/labels_smoke_only/a.txt", "w").write("")
            lst = os.path.join(d, "list.txt"); open(lst, "w").write("a.jpg\n")
            with self.assertRaises(ValueError):
                M.fasdd_view(src, lst, os.path.join(d, "out"))

    def test_missing_file_stops(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src"); os.makedirs(src + "/images"); os.makedirs(src + "/labels_smoke_only")
            lst = os.path.join(d, "list.txt"); open(lst, "w").write("nope.jpg\n")
            with self.assertRaises(FileNotFoundError):
                M.fasdd_view(src, lst, os.path.join(d, "out"))


class TestFiremanView(unittest.TestCase):
    def test_view_and_missing_frames(self):
        with tempfile.TemporaryDirectory() as d:
            xd = os.path.join(d, "xml"); os.makedirs(xd)
            open(os.path.join(xd, "test__video_001__annotations.xml"), "w").write(xml([(i, 100, 100, [("smoke_region", 10, 10, 50, 50)]) for i in range(10)]))
            meta = {"test/video_001": {"tier": "verified_2023", "cluster": "C1"}}
            man = P.build_manifest(xd, meta, per_video=4)
            mp = os.path.join(d, "m.json"); json.dump(man, open(mp, "w"))
            fd = os.path.join(d, "frames"); os.makedirs(fd)
            with self.assertRaises(FileNotFoundError):
                M.fireman_view(mp, fd, xd, os.path.join(d, "out0"))                 # frames not fetched yet
            for v, f, *_ in man["frames"]:
                open(os.path.join(fd, P.frame_name(v, f) + ".jpg"), "wb").write(b"x")
            r = M.fireman_view(mp, fd, xd, os.path.join(d, "out"))
            self.assertEqual(r["n_images"], 4); self.assertEqual(r["labels"]["files_with_smoke"], 4)
            self.assertEqual(sorted(os.listdir(os.path.join(d, "out/images/test"))), sorted(P.frame_name(v, f) + ".jpg" for v, f, *_ in man["frames"]))
            self.assertEqual(len(os.listdir(os.path.join(d, "out/labels/test"))), 4)


if __name__ == "__main__":
    unittest.main()
