"""Tests for the synthetic generator: determinism, label correctness, shared projection.
Run: python3 -m unittest discover -s tests -t . -v"""
import copy
import json
import os
import tempfile
import unittest

import numpy as np

from eval.validate_dataset import validate
from synthgen import generate_dataset as gd
from synthgen.annotate import box_area_frac, box_from_alpha, to_yolo_line, yolo_line_to_box
from synthgen.render import Renderer
from synthgen.scene_params import load_config, sample_scene

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "synthgen", "configs", "domain_rand.yaml")


def rd(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def small_cfg():
    """Real config, shrunk for speed (smaller image and texture, one bank entry per split)."""
    cfg = load_config(CFG_PATH)
    cfg = copy.deepcopy(cfg)
    cfg["image"].update(width=320, height=180)
    cfg["background"].update(bank_size={"train": 3, "val": 3, "test": 3}, texture_px=256)
    return cfg


class SceneTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_split_disjoint(self):
        cfg = small_cfg()
        a = sample_scene(cfg, "train", 3, 7, 0, True).to_dict()
        b = sample_scene(cfg, "train", 3, 7, 0, True).to_dict()
        self.assertEqual(a, b)
        self.assertNotEqual(a, sample_scene(cfg, "val", 3, 7, 0, True).to_dict())      # splits use disjoint seeds
        self.assertNotEqual(a, sample_scene(cfg, "train", 3, 7, 1, True).to_dict())    # attempts differ
        self.assertNotEqual(a, sample_scene(cfg, "train", 3, 8, 0, True).to_dict())    # base seed matters

    def test_negative_scene_has_no_smoke(self):
        cfg = small_cfg()
        sp = sample_scene(cfg, "train", 0, 0, 0, False)
        self.assertIsNone(sp.smoke)
        self.assertFalse(sp.is_positive)

    def test_negative_indices_exact_count(self):
        for n, f in ((100, 0.075), (40, 0.075), (3, 0.075), (10, 0.0)):
            self.assertEqual(len(gd.negative_indices(0, "train", n, f)), round(f * n))

    def test_smoke_anchor_uses_the_shared_camera(self):
        # the anchor was placed by unprojecting a pixel through sim/camera.py; projecting it back must land on that pixel
        cfg = small_cfg()
        for idx in range(12):
            sp = sample_scene(cfg, "train", idx, 3, 0, True)
            cam = sp.camera
            uv, depth = cam.project(np.array([sp.smoke.anchor_n, sp.smoke.anchor_e, 0.0]))
            self.assertGreater(float(depth), 0)
            self.assertTrue(0 <= uv[0] <= sp.width and 0 <= uv[1] <= sp.height, (idx, uv))
            n, e, ok = cam.unproject_ground(uv[0], uv[1])
            self.assertTrue(bool(ok))
            self.assertAlmostEqual(float(n), sp.smoke.anchor_n, places=4)
            self.assertAlmostEqual(float(e), sp.smoke.anchor_e, places=4)
            # and the anchor is below the horizon line
            self.assertGreater(float(uv[1]), sp.horizon_row_px)


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = small_cfg()
        cls.r = Renderer(cls.cfg, 0)

    def test_render_is_deterministic(self):
        sp = sample_scene(self.cfg, "train", 1, 0, 0, True)
        a = self.r.render(sp)
        b = self.r.render(sp)
        np.testing.assert_array_equal(a.image, b.image)
        np.testing.assert_array_equal(a.smoke_alpha, b.smoke_alpha)

    def test_positive_has_smoke_and_negative_does_not(self):
        pos = self.r.render(sample_scene(self.cfg, "train", 1, 0, 0, True))
        neg = self.r.render(sample_scene(self.cfg, "train", 2, 0, 0, False))
        self.assertEqual(pos.image.shape, (180, 320, 3))
        self.assertEqual(pos.image.dtype, np.uint8)
        self.assertGreater(float(pos.smoke_alpha.max()), 0.2)
        self.assertEqual(float(neg.smoke_alpha.max()), 0.0)
        self.assertIsNone(neg.info)

    def test_label_box_matches_alpha_mask(self):
        res = self.r.render(sample_scene(self.cfg, "train", 1, 0, 0, True))
        bx = self.cfg["box"]
        box = box_from_alpha(res.smoke_alpha, bx["alpha_threshold"], 0.0)
        self.assertIsNotNone(box)
        x1, y1, x2, y2 = (int(v) for v in box)
        m = res.smoke_alpha >= bx["alpha_threshold"]
        # every mask pixel is inside the box and the box edges touch the mask
        ys, xs = np.nonzero(m)
        self.assertEqual((xs.min(), xs.max() + 1, ys.min(), ys.max() + 1), (x1, x2, y1, y2))

    def test_yolo_line_roundtrip(self):
        box = (32.0, 18.0, 200.0, 120.0)
        line = to_yolo_line(box, 320, 180)
        cls, back = yolo_line_to_box(line, 320, 180)
        self.assertEqual(cls, 0)
        for a, b in zip(box, back):
            self.assertAlmostEqual(a, b, places=2)
        self.assertAlmostEqual(box_area_frac(box, 320, 180), (168 * 102) / (320 * 180))

    def test_no_pixels_from_real_imagery(self):
        # the generator reads only its config; guard against accidental image loading in the render module
        import synthgen.render as render
        with open(render.__file__) as f:
            src = f.read()
        for banned in ("imread", "Image.open", "cv2.VideoCapture"):
            self.assertNotIn(banned, src)


class EndToEndTests(unittest.TestCase):
    def test_generate_validate_and_reproduce(self):
        cfg = small_cfg()
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            for out in (d1, d2):
                r = Renderer(cfg, 5)
                with open(os.path.join(out, "metadata.jsonl"), "w") as mf:
                    for split, n in (("train", 8), ("val", 4), ("test", 4)):
                        gd.generate_split(cfg, r, split, n, 5, out, 8, mf, log_every=0)
                import yaml
                with open(os.path.join(out, "data.yaml"), "w") as yf:
                    yaml.safe_dump({"path": out, "train": "images/train", "val": "images/val", "test": "images/test",
                                    "names": {0: "smoke"}}, yf)
            rep = validate(os.path.join(d1, "data.yaml"))
            self.assertTrue(rep["ok"], rep["errors"])
            self.assertEqual(rep["splits"]["train"]["images"], 8)
            self.assertEqual(rep["splits"]["train"]["background_images"], round(0.075 * 8))
            self.assertEqual(rep["image_sizes"], {"320x180": 16})
            # same seed -> byte-identical images and labels
            for split in ("train", "val", "test"):
                for f in sorted(os.listdir(os.path.join(d1, "images", split))):
                    self.assertEqual(rd(os.path.join(d1, "images", split, f), "rb"), rd(os.path.join(d2, "images", split, f), "rb"))
                    lf = f.replace(".jpg", ".txt")
                    self.assertEqual(rd(os.path.join(d1, "labels", split, lf)), rd(os.path.join(d2, "labels", split, lf)))

    def test_validator_catches_bad_labels(self):
        cfg = small_cfg()
        with tempfile.TemporaryDirectory() as d:
            r = Renderer(cfg, 5)
            with open(os.path.join(d, "metadata.jsonl"), "w") as mf:
                gd.generate_split(cfg, r, "train", 4, 5, d, 8, mf, log_every=0)
            import yaml
            with open(os.path.join(d, "data.yaml"), "w") as yf:
                yaml.safe_dump({"path": d, "train": "images/train", "names": {0: "smoke"}}, yf)
            lbl = sorted(os.listdir(os.path.join(d, "labels", "train")))[0]
            with open(os.path.join(d, "labels", "train", lbl), "w") as f:
                f.write("3 0.5 0.5 0.2 0.2\n")                  # class 3 does not exist
            rep = validate(os.path.join(d, "data.yaml"))
            self.assertFalse(rep["ok"])
            self.assertTrue(any("class 3" in e for e in rep["errors"]))
            os.remove(os.path.join(d, "labels", "train", lbl))   # missing label file
            rep = validate(os.path.join(d, "data.yaml"))
            self.assertTrue(any("missing label" in e for e in rep["errors"]))
            rep = validate(os.path.join(d, "data.yaml"), allow_missing_labels=True)
            self.assertFalse(any("missing label" in e for e in rep["errors"]))


if __name__ == "__main__":
    unittest.main()
