import json, os, tempfile, unittest

from eval import fireman_prep as P


def xml(frames, name="FireMan_RGB_Test_Video_001"):
    """frames: list of (id, W, H, [(label, x1, y1, x2, y2), ...]) -> CVAT image-export XML string (same element/attribute names as the real files)."""
    out = ["<annotations><version>1.1</version><meta><task><name>%s</name></task></meta>" % name]
    for fid, W, H, boxes in frames:
        out.append('<image id="%d" name="frame_%06d.jpg" width="%d" height="%d">' % (fid, fid, W, H))
        out.append('<tag label="smoke" source="manual"><attribute name="smoke_presence">%s</attribute></tag>' % ("yes" if boxes else "no"))
        for lab, x1, y1, x2, y2 in boxes:
            out.append('<box label="%s" source="manual" occluded="0" xtl="%s" ytl="%s" xbr="%s" ybr="%s" z_order="0"></box>' % (lab, x1, y1, x2, y2))
        out.append("</image>")
    out.append("</annotations>")
    return "".join(out)


class TestConvert(unittest.TestCase):
    def test_yolo_line_normalised(self):
        f = P.parse_cvat(xml([(0, 200, 100, [("smoke_region", 20, 10, 120, 60), ("fire_region", 1, 1, 5, 5), ("lake", 0, 0, 9, 9)])]))[0]
        lines, iss = P.smoke_yolo(f)
        self.assertEqual(lines, ["0 0.350000 0.350000 0.500000 0.500000"])     # fire and lake rectangles are ignored
        self.assertEqual(iss, {"clipped": 0, "multi_box": 0})

    def test_empty_frame_gives_empty_label(self):
        f = P.parse_cvat(xml([(3, 200, 100, [("fire_region", 1, 1, 5, 5)])]))[0]
        self.assertEqual(P.smoke_yolo(f)[0], [])

    def test_multi_box_is_counted_not_dropped(self):
        f = P.parse_cvat(xml([(0, 100, 100, [("smoke_region", 0, 0, 10, 10), ("smoke_region", 50, 50, 90, 90)])]))[0]
        lines, iss = P.smoke_yolo(f)
        self.assertEqual(len(lines), 2); self.assertEqual(iss["multi_box"], 1)

    def test_small_overshoot_clipped_and_counted_large_refused(self):
        f = P.parse_cvat(xml([(0, 100, 100, [("smoke_region", -1.5, 10, 50, 101.5)])]))[0]
        lines, iss = P.smoke_yolo(f)
        self.assertEqual(iss["clipped"], 1)
        xc, yc, w, h = map(float, lines[0].split()[1:])
        self.assertAlmostEqual(w, 0.5); self.assertAlmostEqual(yc + h / 2, 1.0)
        bad = P.parse_cvat(xml([(0, 100, 100, [("smoke_region", -9, 10, 50, 60)])]))[0]
        with self.assertRaises(ValueError):
            P.smoke_yolo(bad)

    def test_degenerate_and_inverted_refused(self):
        for box in (("smoke_region", 10, 10, 10, 50), ("smoke_region", 50, 10, 10, 50)):
            with self.assertRaises(ValueError):
                P.smoke_yolo(P.parse_cvat(xml([(0, 100, 100, [box])]))[0])


class TestKeyframes(unittest.TestCase):
    def test_slope_changes_are_keyframes(self):
        # box moves linearly for frames 0-5, then changes direction at frame 5, then linearly again until 10
        xs = [0, 2, 4, 6, 8, 10, 9, 8, 7, 6, 5]
        fr = P.parse_cvat(xml([(i, 100, 100, [("smoke_region", x, 10, x + 20, 50)]) for i, x in enumerate(xs)]))
        d = P.keyframe_distances(fr)
        self.assertEqual(d[5], 0)
        self.assertEqual((d[4], d[6], d[3], d[8]), (1, 1, 2, 3))

    def test_no_slope_change_gives_none(self):
        fr = P.parse_cvat(xml([(i, 100, 100, [("smoke_region", 10, 10, 50, 50)]) for i in range(6)]))      # static box
        self.assertTrue(all(v is None for v in P.keyframe_distances(fr).values()))

    def test_frames_without_a_single_box_break_the_series(self):
        fr = P.parse_cvat(xml([(0, 100, 100, [("smoke_region", 0, 0, 10, 10)]), (1, 100, 100, []), (2, 100, 100, [("smoke_region", 30, 0, 40, 10)]), (3, 100, 100, [("smoke_region", 90, 0, 99, 10)])]))
        self.assertTrue(all(v is None for v in P.keyframe_distances(fr).values()))


class TestSelection(unittest.TestCase):
    def test_all_frames_when_few(self):
        self.assertEqual(P.select_frames(range(5), 10), [0, 1, 2, 3, 4])

    def test_evenly_spaced_includes_ends_and_is_deterministic(self):
        ids = list(range(1000, 2000))
        a = P.select_frames(ids, 150)
        self.assertEqual(a, P.select_frames(list(reversed(ids)), 150))        # input order does not matter
        self.assertEqual(len(a), 150); self.assertEqual((a[0], a[-1]), (1000, 1999))
        gaps = [y - x for x, y in zip(a, a[1:])]
        self.assertLessEqual(max(gaps) - min(gaps), 1)

    def test_matches_documented_rule(self):
        ids = list(range(0, 100))
        self.assertEqual(P.select_frames(ids, 5), [0, 25, 50, 74, 99])        # round(i*99/4): 0, 24.75, 49.5, 74.25, 99 (banker's rounding on .5)
        self.assertEqual(P.select_frames(ids, 1), [0])


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        frames = [(i, 640, 512, [("smoke_region", 10, 10, 200, 200)] if i % 3 else []) for i in range(30)]
        for split, vid in (("train", "video_001"), ("test", "video_002"), ("val", "video_003")):
            open(os.path.join(self.d.name, f"{split}__{vid}__annotations.xml"), "w").write(xml(frames))
        self.meta = {"train/video_001": {"tier": "unverified", "cluster": "C1"}, "test/video_002": {"tier": "verified_2023", "cluster": "C2"},
                     "val/video_003": {"tier": "verified_2023", "cluster": "C3"}}

    def tearDown(self):
        self.d.cleanup()

    def test_build_verify_roundtrip(self):
        m = P.build_manifest(self.d.name, self.meta, per_video=10)
        self.assertEqual(m["n_videos"], 3); self.assertEqual(m["n_frames"], 30)
        self.assertEqual(P.verify_manifest(m, self.d.name, self.meta), [])

    def test_tier_filter(self):
        m = P.build_manifest(self.d.name, self.meta, per_video=10, tiers=["verified_2023"])
        self.assertEqual(sorted(m["videos"]), ["test/video_002", "val/video_003"])
        self.assertEqual(P.verify_manifest(m, self.d.name, self.meta), [])

    def test_tampering_detected(self):
        m = P.build_manifest(self.d.name, self.meta, per_video=10)
        bad = json.loads(json.dumps(m)); bad["frames"][0][1] += 1
        self.assertTrue(P.verify_manifest(bad, self.d.name, self.meta))
        # changing an annotation changes the recomputed manifest
        p = os.path.join(self.d.name, "train__video_001__annotations.xml")
        open(p, "w").write(xml([(i, 640, 512, []) for i in range(30)]))
        self.assertTrue(P.verify_manifest(m, self.d.name, self.meta))

    def test_convert_writes_labels_and_refuses_overwrite(self):
        m = P.build_manifest(self.d.name, self.meta, per_video=10)
        out = os.path.join(self.d.name, "labels")
        res = P.convert(m, self.d.name, out)
        self.assertEqual(res["files"], 30); self.assertEqual(res["files_with_smoke"], m["n_frames_with_smoke"])
        self.assertEqual(len(os.listdir(out)), 30)
        with self.assertRaises(FileExistsError):
            P.convert(m, self.d.name, out)

    def test_frame_name_is_unique_across_videos(self):
        self.assertNotEqual(P.frame_name("train/video_001", 5), P.frame_name("val/video_001", 5))


if __name__ == "__main__":
    unittest.main()
