"""PriorityPlanner: deterministic, score-then-distance, sticky, expiring; optional per-cell investigation cooldown."""
import unittest

from planning.priority import PriorityPlanner


class TestPlanner(unittest.TestCase):
    def test_score_beats_distance_then_distance_breaks_ties(self):
        p = PriorityPlanner(conf_threshold=0.25)
        p.on_detection(9, 9, "smoke", 0.9, 0.0)
        p.on_detection(1, 1, "smoke", 0.5, 0.0)
        self.assertEqual(p.choose((0, 0)), (9, 9))                 # a confident far cell outranks a marginal near one
        q = PriorityPlanner(conf_threshold=0.25)
        q.on_detection(3, 3, "smoke", 0.6, 0.0)
        q.on_detection(1, 1, "smoke", 0.6, 0.0)
        self.assertEqual(q.choose((0, 0)), (1, 1))                 # equal score: nearer wins

    def test_target_is_sticky_until_the_evidence_expires(self):
        p = PriorityPlanner(detection_ttl_s=20.0, conf_threshold=0.25)
        p.on_detection(5, 5, "smoke", 0.5, 0.0)
        self.assertEqual(p.choose((0, 0)), (5, 5))
        p.on_detection(8, 8, "smoke", 0.99, 1.0)
        self.assertEqual(p.choose((0, 0)), (5, 5))                 # committed: no oscillation toward the better cell
        p.prune(25.0)                                              # (5,5) expired at 20 s, (8,8) at 21 s
        self.assertIsNone(p.choose((0, 0)))

    def test_below_threshold_is_dropped_and_repeat_sightings_raise_the_score(self):
        p = PriorityPlanner(conf_threshold=0.45)
        self.assertFalse(p.on_detection(2, 2, "smoke", 0.4, 0.0))
        self.assertTrue(p.on_detection(2, 2, "smoke", 0.5, 0.0))
        s1 = p.score((2, 2))
        p.on_detection(2, 2, "smoke", 0.5, 1.0)
        self.assertGreater(p.score((2, 2)), s1)

    def test_deterministic_for_identical_input(self):
        def run():
            p = PriorityPlanner(conf_threshold=0.25)
            out = []
            for k, (i, j, c) in enumerate([(3, 4, 0.5), (7, 1, 0.7), (3, 4, 0.6), (2, 2, 0.7)]):
                p.on_detection(i, j, "smoke", c, float(k))
                out.append(p.choose((0, 0)))
            return out
        self.assertEqual(run(), run())

    def test_default_has_no_cooldown_and_reflags_immediately(self):
        p = PriorityPlanner(conf_threshold=0.25)
        p.on_detection(4, 4, "smoke", 0.8, 0.0)
        p.clear((4, 4), 1.0)
        self.assertTrue(p.on_detection(4, 4, "smoke", 0.8, 2.0))   # original behaviour: same smoke re-flags the cell
        self.assertEqual(p.choose((4, 4)), (4, 4))

    def test_cooldown_suppresses_reflagging_then_expires(self):
        p = PriorityPlanner(conf_threshold=0.25, investigate_cooldown_s=60.0)
        p.on_detection(4, 4, "smoke", 0.8, 0.0)
        p.clear((4, 4), 1.0)
        self.assertFalse(p.on_detection(4, 4, "smoke", 0.9, 30.0))
        self.assertIsNone(p.choose((4, 4)))
        self.assertTrue(p.on_detection(4, 4, "smoke", 0.9, 62.0))  # 1 + 60 s have passed
        self.assertTrue(p.on_detection(6, 6, "smoke", 0.9, 30.0))  # other cells are unaffected during the cooldown


if __name__ == "__main__":
    unittest.main()
