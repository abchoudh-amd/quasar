import unittest

import numpy as np

from quasar.pic.initial_conditions import (
    maxwellian,
    quiet_positions_2d,
    quiet_positions_2d_block,
)


class QuietPositionsBlockTests(unittest.TestCase):

    def test_count_matches_request(self):
        pts = quiet_positions_2d_block(50, 0.0, 1.0, 0.0, 2.0)
        self.assertEqual(pts.shape, (50, 2))

    def test_points_inside_block(self):
        x_min, x_max, y_min, y_max = -1.0, 1.0, 2.0, 5.0
        pts = quiet_positions_2d_block(100, x_min, x_max, y_min, y_max)
        self.assertTrue(np.all(pts[:, 0] >= x_min))
        self.assertTrue(np.all(pts[:, 0] <= x_max))
        self.assertTrue(np.all(pts[:, 1] >= y_min))
        self.assertTrue(np.all(pts[:, 1] <= y_max))

    def test_perfect_square_count(self):
        pts = quiet_positions_2d_block(25, 0.0, 1.0, 0.0, 1.0)
        self.assertEqual(pts.shape[0], 25)


class QuietPositionsUniformTests(unittest.TestCase):

    def test_count_and_bounds(self):
        lx, ly = 3.0, 4.0
        pts = quiet_positions_2d(40, lx, ly)
        self.assertEqual(pts.shape, (40, 2))
        self.assertTrue(np.all(pts[:, 0] >= 0.0))
        self.assertTrue(np.all(pts[:, 0] <= lx))
        self.assertTrue(np.all(pts[:, 1] >= 0.0))
        self.assertTrue(np.all(pts[:, 1] <= ly))


class MaxwellianTests(unittest.TestCase):

    def test_shape_and_drift(self):
        v = maxwellian(10000, thermal_speed=2.0, drift=(1.0, 0.0, -3.0), seed=7)
        self.assertEqual(v.shape, (10000, 3))
        # Sample mean should sit near the drift for a large sample.
        self.assertAlmostEqual(float(np.mean(v[:, 0])), 1.0, delta=0.1)
        self.assertAlmostEqual(float(np.mean(v[:, 2])), -3.0, delta=0.1)

    def test_deterministic_with_seed(self):
        a = maxwellian(100, 1.0, seed=123)
        b = maxwellian(100, 1.0, seed=123)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
