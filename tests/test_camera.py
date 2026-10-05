"""Round-trip and geometry tests for sim/camera.py. Run: python3 -m unittest discover -s tests -t . -v"""
import math
import random
import unittest

import numpy as np

from fog.state import cell_index
from mission.geo import latlon_to_ne
from perception.base import Pose
from sim.camera import Camera, CameraIntrinsics, CameraPose, GridSpec

INTR = CameraIntrinsics(1280, 720, 78.0)


def cam(n=0.0, e=0.0, alt=100.0, hdg=0.0, pitch=45.0, intr=INTR):
    return Camera(intr, CameraPose(n, e, alt, hdg, pitch))


class IntrinsicsTests(unittest.TestCase):
    def test_vfov_from_aspect(self):
        v = INTR.vfov_deg
        want = math.degrees(2 * math.atan(math.tan(math.radians(39.0)) * 720 / 1280))
        self.assertAlmostEqual(v, want, places=9)

    def test_diagonal_fov_roundtrip(self):
        i = CameraIntrinsics.from_fov(1280, 720, 78.0, "diagonal")
        diag = math.degrees(2 * math.atan(math.hypot(640, 360) / i.fx))
        self.assertAlmostEqual(diag, 78.0, places=9)
        self.assertLess(i.hfov_deg, 78.0)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            CameraIntrinsics.from_fov(1280, 720, 78.0, "vertical")
        with self.assertRaises(ValueError):
            cam(alt=0.0)
        with self.assertRaises(ValueError):
            cam(pitch=95.0)
        with self.assertRaises(ValueError):
            cam(pitch=-50.0)


class ProjectionTests(unittest.TestCase):
    def test_nadir_principal_point_is_directly_below(self):
        c = cam(n=30.0, e=-20.0, alt=50.0, pitch=90.0)
        n, e, ok = c.unproject_ground(np.array(640.0), np.array(360.0))
        self.assertTrue(bool(ok))
        self.assertAlmostEqual(float(n), 30.0, places=6)
        self.assertAlmostEqual(float(e), -20.0, places=6)

    def test_oblique_principal_point_distance_and_heading(self):
        # 45 deg below horizon at 80 m: principal ray lands 80 m ahead
        for hdg, expect in ((0, (80.0, 0.0)), (90, (0.0, 80.0)), (180, (-80.0, 0.0)), (270, (0.0, -80.0))):
            c = cam(alt=80.0, hdg=hdg, pitch=45.0)
            n, e, ok = c.unproject_ground(np.array(640.0), np.array(360.0))
            self.assertTrue(bool(ok))
            self.assertAlmostEqual(float(n), expect[0], places=6)
            self.assertAlmostEqual(float(e), expect[1], places=6)

    def test_image_axes_orientation(self):
        # facing north, a point to the east must appear to the RIGHT of centre, a point farther north higher up
        c = cam(alt=100.0, hdg=0.0, pitch=60.0)
        uv0, _ = c.project_ground(np.array(80.0), np.array(0.0))
        uv_e, _ = c.project_ground(np.array(80.0), np.array(30.0))
        uv_far, _ = c.project_ground(np.array(200.0), np.array(0.0))
        self.assertGreater(uv_e[0], uv0[0])
        self.assertLess(uv_far[1], uv0[1])          # farther ground is closer to the horizon = smaller row

    def test_ground_roundtrip_random_poses(self):
        rng = random.Random(7)
        for _ in range(300):
            c = cam(n=rng.uniform(-300, 300), e=rng.uniform(-300, 300), alt=rng.uniform(10, 400),
                    hdg=rng.uniform(0, 360), pitch=rng.uniform(15, 90))
            u = np.array([rng.uniform(0, 1280) for _ in range(20)])
            v = np.array([rng.uniform(0, 720) for _ in range(20)])
            n, e, ok = c.unproject_ground(u, v)
            m = ok & np.isfinite(n)
            uv, depth = c.project_ground(n[m], e[m])
            self.assertTrue(np.all(depth > 0))
            np.testing.assert_allclose(uv[:, 0], u[m], atol=1e-5)
            np.testing.assert_allclose(uv[:, 1], v[m], atol=1e-5)

    def test_world_point_roundtrip_with_height(self):
        # a point above the ground (a plume top at 30 m) projects, and unprojecting onto the u=30 plane returns it
        c = cam(n=-50.0, e=10.0, alt=120.0, hdg=35.0, pitch=40.0)
        pt = np.array([60.0, 55.0, 30.0])
        uv, depth = c.project(pt)
        n, e, ok = c.unproject_ground(uv[0], uv[1], ground_u=30.0)
        self.assertTrue(bool(ok))
        self.assertAlmostEqual(float(n), 60.0, places=5)
        self.assertAlmostEqual(float(e), 55.0, places=5)

    def test_behind_camera_is_nan(self):
        c = cam(alt=50.0, hdg=0.0, pitch=45.0)
        uv, depth = c.project(np.array([-500.0, 0.0, 49.0]))
        self.assertLess(depth, 0)
        self.assertTrue(np.isnan(uv).all())

    def test_horizon_row(self):
        c = cam(alt=100.0, hdg=0.0, pitch=20.0)
        h = c.horizon_row()
        self.assertAlmostEqual(h, 360.0 - INTR.fy * math.tan(math.radians(20.0)), places=9)
        # rays just below that row hit the ground, rays just above do not
        _, _, below = c.unproject_ground(np.array(640.0), np.array(h + 1.0))
        _, _, above = c.unproject_ground(np.array(640.0), np.array(h - 1.0))
        self.assertTrue(bool(below))
        self.assertFalse(bool(above))
        # a far ground point projects very close to the horizon row
        uv, _ = c.project_ground(np.array(1e6), np.array(0.0))
        self.assertAlmostEqual(float(uv[1]), h, delta=0.2)
        self.assertEqual(cam(pitch=90.0).horizon_row(), -math.inf)

    def test_negative_pitch_horizon_below_centre(self):
        c = cam(alt=100.0, hdg=0.0, pitch=-5.0)
        h = c.horizon_row()
        self.assertGreater(h, 360.0)                                   # horizon below the image centre
        self.assertAlmostEqual(h, 360.0 + INTR.fy * math.tan(math.radians(5.0)), places=9)
        _, _, ok_centre = c.unproject_ground(np.array(640.0), np.array(360.0))
        self.assertFalse(bool(ok_centre))                              # centre ray points above the horizon
        _, _, ok_low = c.unproject_ground(np.array(640.0), np.array(h + 2.0))
        self.assertTrue(bool(ok_low))

    def test_ground_grid_matches_pointwise(self):
        c = cam(alt=90.0, hdg=120.0, pitch=50.0)
        n, e, ok = c.ground_grid()
        self.assertEqual(n.shape, (720, 1280))
        n0, e0, ok0 = c.unproject_ground(np.array(100.5), np.array(200.5))
        self.assertAlmostEqual(float(n[200, 100]), float(n0), places=9)
        self.assertAlmostEqual(float(e[200, 100]), float(e0), places=9)
        self.assertTrue(ok.all())               # pitch 50, vfov ~46 deg: whole image is below the horizon


class FootprintTests(unittest.TestCase):
    def test_nadir_footprint_area(self):
        alt = 60.0
        c = cam(alt=alt, pitch=90.0)
        poly, clipped = c.footprint()
        self.assertFalse(clipped)
        w = 2 * alt * math.tan(math.radians(INTR.hfov_deg / 2))
        h = 2 * alt * math.tan(math.radians(INTR.vfov_deg / 2))
        x, y = poly[:, 1], poly[:, 0]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        self.assertAlmostEqual(area, w * h, delta=1e-6 * w * h)

    def test_horizon_view_is_range_clipped(self):
        c = cam(alt=100.0, pitch=10.0)          # vfov/2 ~ 22.6 deg > 10 deg, so the sky is in view
        poly, clipped = c.footprint(max_range_m=1500.0)
        self.assertTrue(clipped)
        d = np.hypot(poly[:, 0], poly[:, 1])
        self.assertLessEqual(d.max(), 1500.0 + 1e-6)

    def test_cells_in_nadir_footprint(self):
        grid = GridSpec(n_cells=10, cell_m=100.0)       # +-500 m
        # camera over the SW-ish part of cell (4, 5); footprint at 200 m is ~324 x 183 m
        n_c, e_c = grid.center_ne(4, 5)
        c = cam(n=n_c, e=e_c, alt=200.0, pitch=90.0)
        cells = c.cells_in_footprint(grid)
        self.assertIn((4, 5), cells)
        # all returned cells really are inside the footprint rectangle
        w = 2 * 200 * math.tan(math.radians(39.0))
        h = 2 * 200 * math.tan(math.radians(INTR.vfov_deg / 2))
        for i, j in cells:
            cn, ce = grid.center_ne(i, j)
            self.assertLessEqual(abs(ce - e_c), w / 2 + 1e-6)
            self.assertLessEqual(abs(cn - n_c), h / 2 + 1e-6)
        # and an in-footprint cell centre is never missed
        want = [(i, j) for i in range(10) for j in range(10)
                if abs(grid.center_ne(i, j)[1] - e_c) < w / 2 - 1e-6 and abs(grid.center_ne(i, j)[0] - n_c) < h / 2 - 1e-6]
        self.assertEqual(sorted(cells), sorted(want))


class GridTests(unittest.TestCase):
    def test_center_is_inverse_of_fog_cell_index(self):
        g = GridSpec.from_miles(1.0, 10)
        for i in range(10):
            for j in range(10):
                n, e = g.center_ne(i, j)
                self.assertEqual(g.cell_of(n, e), (i, j))
                self.assertEqual(cell_index(n, e, g.half_side_m, g.cell_m, g.n_cells), (i, j))

    def test_axes_i_east_j_north(self):
        g = GridSpec(10, 100.0)
        self.assertEqual(g.cell_of(-499.0, -499.0), (0, 0))       # south-west corner
        self.assertEqual(g.cell_of(-499.0, 499.0), (9, 0))        # east -> i grows
        self.assertEqual(g.cell_of(499.0, -499.0), (0, 9))        # north -> j grows
        self.assertIsNone(g.cell_of(0.0, 600.0))

    def test_bbox_to_cell(self):
        g = GridSpec(10, 100.0)
        n_c, e_c = g.center_ne(6, 3)
        c = cam(n=n_c - 100.0, e=e_c, alt=120.0, hdg=0.0, pitch=45.0)   # looking north at (6,3)
        # the smoke base is the bottom-centre of the box; take the pixel of that ground point
        uv, _ = c.project_ground(np.array(n_c), np.array(e_c))
        bbox = [uv[0] - 60.0, uv[1] - 200.0, uv[0] + 60.0, uv[1]]
        self.assertEqual(c.bbox_to_cell(bbox, g), (6, 3))
        # a box whose base is above the horizon has no ground anchor
        low = cam(alt=120.0, hdg=0.0, pitch=10.0)
        self.assertGreater(low.horizon_row(), 100.0)                  # horizon is well inside the image
        self.assertIsNone(low.bbox_to_cell([600.0, 20.0, 700.0, 100.0], g))
        self.assertIsNotNone(low.bbox_ground_anchor([600.0, 300.0, 700.0, 500.0]))   # base below the horizon


class PoseTests(unittest.TestCase):
    def test_from_pose_matches_geo(self):
        lat0, lon0 = 21.2970, -157.8170
        p = Pose(lat=lat0 + 0.0009, lon=lon0 - 0.0012, alt=25.0, heading_deg=77.0, cell=None, t=0.0)
        c = Camera.from_pose(p, INTR, lat0, lon0, pitch_deg=30.0)
        n, e = latlon_to_ne(p.lat, p.lon, lat0, lon0)
        self.assertAlmostEqual(c.pose.n, n, places=9)
        self.assertAlmostEqual(c.pose.e, e, places=9)
        self.assertEqual(c.pose.heading_deg, 77.0)
        self.assertEqual(c.pose.alt, 25.0)

    def test_from_pose_missing_fields(self):
        with self.assertRaises(ValueError):
            Camera.from_pose(Pose(None, None, None, None, None, 0.0), INTR, 21.0, -157.0, 30.0)


if __name__ == "__main__":
    unittest.main()
