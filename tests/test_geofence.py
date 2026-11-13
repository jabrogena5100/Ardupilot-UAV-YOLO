"""Geofence pushback must point back INSIDE the fence (regression: the sign was inverted and a vehicle that left the box
through a corner was driven further out for the rest of the run, see docs/experiments/level3_methodology.md section 10)."""
import unittest

from mission.geo import ne_to_latlon
from swarm_agent import Agent


class _Stub:
    half_side_m = 804.672
    origin_lat, origin_lon = 21.2970, -157.8170
    geofence_push = Agent.geofence_push

    class params:
        verbose = False


class TestGeofence(unittest.TestCase):
    def push(self, n, e):
        lat, lon = ne_to_latlon(n, e, _Stub.origin_lat, _Stub.origin_lon)
        return _Stub().geofence_push(5.0, 5.0, lat, lon)

    def test_outside_each_side_is_pushed_inward(self):
        h = _Stub.half_side_m
        vn, ve = self.push(0.0, -(h + 30))   # west of the fence
        self.assertGreater(ve, 0)
        vn, ve = self.push(0.0, h + 30)      # east
        self.assertLess(ve, 0)
        vn, ve = self.push(h + 30, 0.0)      # north
        self.assertLess(vn, 0)
        vn, ve = self.push(-(h + 30), 0.0)   # south
        self.assertGreater(vn, 0)

    def test_corner_pushed_toward_centre(self):
        h = _Stub.half_side_m
        vn, ve = self.push(-(h + 30), -(h + 30))
        self.assertGreater(vn, 0)
        self.assertGreater(ve, 0)


if __name__ == "__main__":
    unittest.main()
