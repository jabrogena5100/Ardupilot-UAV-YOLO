import os, tempfile, unittest

import numpy as np
from PIL import Image

from eval import external_prep as E


def names():
    out = []
    for p, n in (("Smoke", 60), ("FireAndSmoke", 60), ("Fire", 80), ("NeitherFireNorSmoke", 80)):
        out += [f"{p}_{i}.jpg" for i in range(n)]
    return out


class TestScreen(unittest.TestCase):
    def test_prefix_longest_match(self):
        self.assertEqual(E.stratum("FireAndSmoke_3.jpg"), "FireAndSmoke")
        self.assertEqual(E.stratum("NeitherFireNorSmoke_3.jpg"), "NeitherFireNorSmoke")
        self.assertEqual(E.stratum("Fire_3.jpg"), "Fire")
        self.assertEqual(E.stratum("x.jpg"), "UNKNOWN")

    def test_sample_counts_and_determinism(self):
        r1 = E.screening_sample(names(), 7)
        r2 = E.screening_sample(list(reversed(names())), 7)
        self.assertEqual(r1, r2)                                   # input order does not matter
        self.assertEqual(len(r1), 100)
        self.assertEqual(sum(x[2] for x in r1), 30)
        self.assertEqual(len({x[0] for x in r1}), 100)
        self.assertNotEqual(r1, E.screening_sample(names(), 8))    # seed matters

    def test_too_few_candidates_refuses(self):
        with self.assertRaises(SystemExit):
            E.screening_sample([n for n in names() if not n.startswith("Fire_")], 1)


def real_names():
    """File-name conventions of the real FASDD_UAV.zip (verified 2026-11-10): lower-camel prefixes, e.g. bothFireAndSmoke_UAV000003.jpg."""
    out = []
    for p, n in (("smoke", 60), ("bothFireAndSmoke", 60), ("fire", 80), ("neitherFireNorSmoke", 80)):
        out += [f"{p}_UAV{i:06d}.jpg" for i in range(n)]
    return out


class TestRealFasddNames(unittest.TestCase):
    def test_real_prefixes_map_to_canonical_strata(self):
        self.assertEqual(E.stratum("bothFireAndSmoke_UAV000003.jpg"), "FireAndSmoke")
        self.assertEqual(E.stratum("neitherFireNorSmoke_UAV000004.jpg"), "NeitherFireNorSmoke")
        self.assertEqual(E.stratum("smoke_UAV000001.jpg"), "Smoke")
        self.assertEqual(E.stratum("fire_UAV000005.jpg"), "Fire")
        self.assertEqual(E.stratum("./images/bothFireAndSmoke_UAV000003.jpg"), "FireAndSmoke")   # list-file form
        self.assertEqual(E.stratum("images/fire_UAV000005.jpg"), "Fire")
        self.assertEqual(E.stratum("unknown_UAV000005.jpg"), "UNKNOWN")

    def test_fire_does_not_swallow_fireandsmoke(self):
        self.assertNotEqual(E.stratum("bothFireAndSmoke_UAV000003.jpg"), "Fire")
        self.assertNotEqual(E.stratum("FireAndSmoke_3.jpg"), "Fire")

    def test_screening_on_real_names_is_complete_and_keeps_real_names(self):
        rows = E.screening_sample(real_names(), 20261109)
        self.assertEqual(len(rows), 100)
        counts = {}
        for nm, st, _ in rows:
            counts[st] = counts.get(st, 0) + 1
            self.assertIn(nm, real_names())                          # names come back unchanged, not canonicalised
            self.assertEqual(E.stratum(nm), st)
        self.assertEqual(counts, {"Smoke": 20, "FireAndSmoke": 20, "Fire": 30, "NeitherFireNorSmoke": 30})
        self.assertEqual(sum(x[2] for x in rows), 30)
        self.assertEqual(rows, E.screening_sample(list(reversed(real_names())), 20261109))

    def test_real_names_not_unknown(self):
        self.assertFalse([n for n in real_names() if E.stratum(n) == "UNKNOWN"])


class TestOverlap(unittest.TestCase):
    def test_dhash_and_match(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            base = (rng.random((90, 160)) * 255).astype("uint8")
            a, b, c = (os.path.join(d, x) for x in ("a.png", "b.png", "c.png"))
            Image.fromarray(base).save(a)
            Image.fromarray(base).resize((320, 180)).save(b)                    # same content, rescaled
            Image.fromarray((rng.random((90, 160)) * 255).astype("uint8")).save(c)  # unrelated
            ref = {"train/ref": E.dhash64(a)}
            res = {r["image"]: r for r in E.overlap([a, b, c], set(), ref, tol=6)}
            self.assertTrue(res["a.png"]["flagged"] and res["a.png"]["min_dist"] == 0)
            self.assertTrue(res["b.png"]["flagged"])
            self.assertFalse(res["c.png"]["flagged"])
            self.assertEqual(res["a.png"]["nearest"], "train/ref")

    def test_exact_md5(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.png")
            Image.fromarray(np.zeros((10, 10), "uint8")).save(p)
            r = E.overlap([p], {E.md5(p)}, {"x": 0xFFFFFFFFFFFFFFFF ^ 0xAAAAAAAA})[0]
            self.assertTrue(r["md5_match"] and r["flagged"])

    def test_popcount(self):
        self.assertEqual(int(E.popcount64(np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64))[0]), 64)
        self.assertEqual(int(E.popcount64(np.array([0b1011], dtype=np.uint64))[0]), 3)


class TestRemap(unittest.TestCase):
    def test_smoke_to_zero_fire_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            lab, out = os.path.join(d, "l"), os.path.join(d, "o")
            os.makedirs(lab)
            open(os.path.join(lab, "a.txt"), "w").write("0 .5 .5 .1 .1\n1 .2 .2 .1 .1\n")
            open(os.path.join(lab, "b.txt"), "w").write("0 .5 .5 .1 .1\n")
            c = E.remap_yolo(lab, out, ["fire", "smoke"])               # fire=0, smoke=1 here
            self.assertEqual(open(os.path.join(out, "a.txt")).read(), "0 .2 .2 .1 .1\n")
            self.assertEqual(open(os.path.join(out, "b.txt")).read(), "")
            self.assertEqual((c["smoke_boxes"], c["fire_boxes_dropped"], c["files_with_smoke"]), (1, 2, 1))
            c2 = E.remap_yolo(lab, out + "2", ["smoke", "fire"])        # opposite id order is read, not assumed
            self.assertEqual((c2["smoke_boxes"], c2["fire_boxes_dropped"]), (2, 1))

    def test_unexpected_classes_refused(self):
        with self.assertRaises(SystemExit):
            E.remap_yolo("/nonexistent", "/tmp/x", ["fire", "smoke", "other"])


if __name__ == "__main__":
    unittest.main()
