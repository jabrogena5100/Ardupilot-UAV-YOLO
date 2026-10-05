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
    cfg["background"].update(bank_size={"train": 4, "val": 3, "test": 3}, texture_px=256)
    cfg["smoke"]["puff_bank_size"] = {"train": 6, "val": 4, "test": 4}
    return cfg


def gated_cfg():
    """small_cfg with the (default-off) label-level edge gate switched on, for the tests that exercise it."""
    cfg = small_cfg()
    cfg["box"]["edge_policy"]["enforce_on_labels"] = True
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
                    for split, npos, nneg in (("train", 7, 1), ("val", 4, 0), ("test", 3, 1)):
                        gd.generate_split(cfg, r, split, npos, nneg, 5, out, 40, mf, log_every=0)
                import yaml
                with open(os.path.join(out, "data.yaml"), "w") as yf:
                    yaml.safe_dump({"path": out, "train": "images/train", "val": "images/val", "test": "images/test",
                                    "names": {0: "smoke"}}, yf)
            rep = validate(os.path.join(d1, "data.yaml"))
            self.assertTrue(rep["ok"], rep["errors"])
            self.assertEqual(rep["splits"]["train"]["images"], 8)
            self.assertEqual(rep["splits"]["train"]["background_images"], 1)
            self.assertEqual(rep["splits"]["val"]["background_images"], 0)
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
                gd.generate_split(cfg, r, "train", 4, 0, 5, d, 40, mf, log_every=0)
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


class ApprovedChangeTests(unittest.TestCase):
    """The pre-scaling changes approved for synth_smoke_v1."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = small_cfg()

    # --- 1. split-specific, larger sprite banks --------------------------------------
    def test_puff_sprite_banks_are_independent_across_splits(self):
        r = Renderer(self.cfg, 1001)
        banks = {s: r._puffs(s) for s in ("train", "val", "test")}
        for s, n in self.cfg["smoke"]["puff_bank_size"].items():
            self.assertEqual(banks[s][0].shape[0], n)
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            k = min(banks[a][0].shape[0], banks[b][0].shape[0])
            self.assertFalse(np.allclose(banks[a][0][:k], banks[b][0][:k]), (a, b))     # different seeds -> different sprites
            # no sprite of one split appears (even approximately) in another
            for i in range(banks[a][0].shape[0]):
                for j in range(banks[b][0].shape[0]):
                    self.assertGreater(float(np.abs(banks[a][0][i] - banks[b][0][j]).mean()), 1e-3)
        # same split + seed -> identical sprites (reproducible); different base seed -> different
        np.testing.assert_array_equal(Renderer(self.cfg, 1001)._puffs("val")[0], banks["val"][0])
        self.assertFalse(np.allclose(Renderer(self.cfg, 7)._puffs("val")[0], banks["val"][0]))

    def test_default_banks_are_larger(self):
        cfg = load_config(CFG_PATH)
        self.assertEqual(cfg["smoke"]["puff_bank_size"], {"train": 48, "val": 16, "test": 16})
        self.assertEqual(cfg["background"]["bank_size"], {"train": 16, "val": 6, "test": 6})

    # --- 2. drift cap, independent of the width target --------------------------------
    def test_drift_is_capped_and_not_tied_to_width(self):
        cfg = self.cfg
        cap = cfg["smoke"]["max_drift_over_height"]
        per = {}
        for i in range(400):
            sp = sample_scene(cfg, "train", i, 1001, 0, True)
            if sp.smoke.style == "curtain":
                self.assertEqual(sp.smoke.lean, 0.0)
                continue
            self.assertLessEqual(sp.smoke.lean, cap + 1e-9)
            w = sp.smoke.target_box_xyxy[2] - sp.smoke.target_box_xyxy[0]
            per.setdefault(sp.archetype, []).append((sp.smoke.lean, w))
        for arch, v in per.items():
            lean = np.array([a for a, _ in v])
            wid = np.array([b for _, b in v])
            lo, hi = cfg["archetypes"][arch]["lean"]
            # drawn from the archetype's own range (the width bound essentially never binds) ...
            self.assertGreaterEqual(float(lean.min()), lo - 1e-9, arch)
            self.assertLessEqual(float(lean.max()), hi + 1e-9, arch)
            # ... and independent of the width target within an archetype
            self.assertLess(abs(float(np.corrcoef(lean, wid)[0, 1])), 0.35, arch)

    # --- 3. box variants recorded ---------------------------------------------------------
    def test_box_variants_recorded_and_ordered(self):
        r = Renderer(self.cfg, 1001)
        sp = sample_scene(self.cfg, "train", 2, 1001, 0, True)
        res = r.render(sp)
        from synthgen.annotate import box_variants
        v = box_variants(res.smoke_alpha, self.cfg["box"]["variants"], 1)
        self.assertEqual(set(v), {"tight", "default", "haze", "padded"})
        area = {k: box_area_frac(b, 320, 180) for k, b in v.items() if b}
        self.assertLessEqual(area["tight"], area["default"] + 1e-9)       # lower alpha threshold / more padding only grow the box
        self.assertLessEqual(area["default"], area["haze"] + 1e-9)
        self.assertLessEqual(area["default"], area["padded"] + 1e-9)

    # --- 4. per-texture palette jitter ---------------------------------------------------
    def test_palette_jitter_varies_across_the_bank(self):
        r = Renderer(self.cfg, 1001)
        means = []
        for slot, (kind, pyr) in enumerate(r._bank("train")):
            if kind == "canopy":
                means.append(pyr[0][..., :3].reshape(-1, 3).mean(0))
        self.assertGreaterEqual(len(means), 2)
        spread = np.std(np.array(means), axis=0)
        self.assertGreater(float(spread.max()), 0.4)                       # texture-to-texture palette shift (0-255 scale)
        # jitter off -> the same kind has (much) less palette variation
        cfg2 = copy.deepcopy(self.cfg)
        cfg2["background"]["palette_jitter"] = {"brightness": [1, 1], "red_vs_green": [1, 1], "blue_vs_green": [1, 1]}
        means2 = [pyr[0][..., :3].reshape(-1, 3).mean(0) for k, pyr in Renderer(cfg2, 1001)._bank("train") if k == "canopy"]
        self.assertLess(float(np.std(np.array(means2), axis=0).max()), float(spread.max()))

    # --- 5. edge policy ---------------------------------------------------------------------
    def test_image_level_draws_do_not_change_with_the_attempt(self):
        """Archetype, box size and the touch draw are fixed per image index, so retries cannot bias the mix."""
        for i in range(30):
            a0 = sample_scene(self.cfg, "train", i, 1001, 0, True)
            a3 = sample_scene(self.cfg, "train", i, 1001, 3, True)
            self.assertEqual(a0.archetype, a3.archetype)
            self.assertEqual(a0.edge_touch_wanted, a3.edge_touch_wanted)
            self.assertEqual(a0.edge_touch_forced, a3.edge_touch_forced)
            w0 = a0.smoke.target_box_xyxy[2] - a0.smoke.target_box_xyxy[0]
            w3 = a3.smoke.target_box_xyxy[2] - a3.smoke.target_box_xyxy[0]
            h0 = a0.smoke.target_box_xyxy[3] - a0.smoke.target_box_xyxy[1]
            h3 = a3.smoke.target_box_xyxy[3] - a3.smoke.target_box_xyxy[1]
            self.assertAlmostEqual(w0, w3, places=6)
            self.assertAlmostEqual(h0, h3, places=6)
            self.assertNotEqual(a0.cam_alt_m, a3.cam_alt_m)            # nuisance variables do change

    def test_edge_policy_rates_and_archetype_mix(self):
        from synthgen.scene_params import _touches
        cfg = load_config(CFG_PATH)                  # sampling only (no rendering), so the real 1280x720 config is cheap
        ep = cfg["box"]["edge_policy"]
        W, H = cfg["image"]["width"], cfg["image"]["height"]
        N = 1500
        touch, wanted, arch = [], [], []
        for i in range(N):
            sp = sample_scene(cfg, "train", i, 1001, 0, True)
            touch.append(_touches(sp.smoke.target_box_xyxy, W, H, ep["margin"]))
            wanted.append(sp.edge_touch_wanted)
            arch.append(sp.archetype)
        self.assertGreater(float(np.mean(np.array(touch) == np.array(wanted))), 0.90)    # targets honour the draw
        self.assertGreater(float(np.mean(wanted)), 0.25)                                # pooled touch rate: far below the
        self.assertLess(float(np.mean(wanted)), 0.55)                                   # old 0.81, near Boreal's 0.30-0.43
        for k, v in cfg["archetypes"].items():                                         # mix not distorted by the policy
            self.assertAlmostEqual(arch.count(k) / N, v["weight"], delta=0.04, msg=k)

    def test_explicit_pos_neg_counts_and_test_views(self):
        cfg = self.cfg
        self.assertEqual(gd.parse_counts("train=3219:260,val=774:0,test=701:350"),
                         {"train": (3219, 260), "val": (774, 0), "test": (701, 350)})
        self.assertEqual(gd.resolve_counts(cfg, {"train": (5, 1), "val": 10})["train"], (5, 1))
        with tempfile.TemporaryDirectory() as d:
            r = Renderer(cfg, 11)
            with open(os.path.join(d, "metadata.jsonl"), "w") as mf:
                gd.generate_split(cfg, r, "test", 4, 2, 11, d, 40, mf, log_every=0)
            with open(os.path.join(d, "metadata.jsonl")) as mf:
                recs = [json.loads(l) for l in mf]
            self.assertEqual(sum(1 for x in recs if not x["is_positive"]), 2)
            self.assertTrue(all(x["box_variants"] for x in recs if x["is_positive"]))
            self.assertTrue(all(x["label_convention"] == "default" for x in recs))

    # --- 6. label-level edge enforcement -------------------------------------------------
    def test_rendered_labels_follow_the_edge_touch_draw(self):
        cfg = gated_cfg()
        with tempfile.TemporaryDirectory() as d:
            r = Renderer(cfg, 21)
            with open(os.path.join(d, "metadata.jsonl"), "w") as mf:
                gd.generate_split(cfg, r, "train", 24, 0, 21, d, 60, mf, log_every=0)
            with open(os.path.join(d, "metadata.jsonl")) as mf:
                recs = [json.loads(l) for l in mf]
        self.assertEqual(len(recs), 24)
        for rec in recs:
            if not rec["edge_gate_fallback"]:                      # every gated image follows its draw exactly
                self.assertEqual(rec["box_touches_edge"], rec["edge_touch_wanted"], rec["image_id"])
        self.assertGreaterEqual(sum(not r["edge_gate_fallback"] for r in recs), 8)   # the gate succeeds for a good share

    def test_gate_rejects_before_rendering_the_background(self):
        cfg = gated_cfg()
        r = Renderer(cfg, 21)
        # find a scene and force the draw to disagree with its predicted label: render() must return None
        seen_none = seen_ok = False
        for i in range(40):
            sp = sample_scene(cfg, "train", i, 21, 0, True)
            res = r.render(sp, enforce_edge=True)
            if res is None:
                seen_none = True
            else:
                seen_ok = True
            if seen_none and seen_ok:
                break
        self.assertTrue(seen_none and seen_ok)
        # without enforcement a positive scene always renders
        self.assertIsNotNone(r.render(sample_scene(cfg, "train", 0, 21, 0, True)))


if __name__ == "__main__":
    unittest.main()
