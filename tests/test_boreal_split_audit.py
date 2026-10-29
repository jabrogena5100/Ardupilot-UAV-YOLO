import unittest

from eval.boreal_split_audit import audit, parse


def rec(md5, dh="0000000000000000", nl=1):
    return [md5, 10, [4, 4], dh, nl]


class TestBorealSplitAudit(unittest.TestCase):
    def test_parse_drops_site_prefix_and_frame(self):
        self.assertEqual(parse("Empty-Images__karkkila_DJI_0003_frame24.jpg"), ("karkkila_dji_0003", 24))
        self.assertEqual(parse("Karkkila-Images__karkkila_DJI_0003_frame0.jpg"), ("karkkila_dji_0003", 0))
        self.assertEqual(parse("Evo-Images__evoDJI_0007_frame10.jpg"), ("evodji_0007", 10))

    def test_prefix_only_difference_is_detected_as_shared_clip(self):
        scan = {
            "train": {"Empty-Images__karkkila_DJI_0003_frame24.jpg": rec("a", nl=-1), "Evo-Images__evoDJI_0001_frame1.jpg": rec("b")},
            "val": {"Heinola-Images__heinola_DJI_0001_frame1.jpg": rec("c")},
            "test": {"Karkkila-Images__karkkila_DJI_0003_frame0.jpg": rec("d"), "Karkkila-Images__karkkila_DJI_0003_frame2.jpg": rec("e")},
        }
        r = audit(scan)
        self.assertEqual(r["prefixed_pairwise_overlap"]["train&test"], 0)   # the original, flawed definition sees nothing
        self.assertEqual(r["normalised_pairwise_overlap"]["train&test"], 1)
        self.assertEqual(r["normalised_pairwise_overlap"]["train&val"], 0)
        d = r["overlap_detail"]["train&test"][0]
        self.assertEqual((d["video"], d["train_frames"], d["test_frames"], d["min_frame_gap"]), ("karkkila_dji_0003", 1, 2, 22))

    def test_md5_and_dhash_cross_split(self):
        scan = {
            "train": {"A__x_frame1.jpg": rec("same", "00000000000000ff")},
            "val": {"B__y_frame1.jpg": rec("zzz")},
            "test": {"C__z_frame1.jpg": rec("same", "00000000000000fe")},
        }
        r = audit(scan, near_bits=2)
        self.assertEqual(r["md5_cross_split_groups"], 1)
        self.assertEqual(r["dhash_near_pairs_cross_split"], {"train&test": 1})
        self.assertEqual(r["dhash_near_pairs_different_video"], {"train&test": 1})


if __name__ == "__main__":
    unittest.main()
