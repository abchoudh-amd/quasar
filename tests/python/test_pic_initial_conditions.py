import unittest
import math

import numpy as np

from quasar.pic.initial_conditions import (
    maxwellian,
    quiet_block_cell_area,
    quiet_block_ring_volume,
    quiet_positions_2d,
    quiet_positions_2d_block,
    quiet_positions_rz_block,
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

    def test_each_axis_is_exactly_stratified_and_centered(self):
        n = 50
        pts = quiet_positions_2d_block(n, -2.0, 4.0, 3.0, 9.0)
        expected_center = np.asarray([1.0, 6.0])
        # Summing binary64 stratum midpoints incurs an absolute error that
        # scales with the coordinate magnitude.  Use a component-wise bound;
        # a fixed multiple of epsilon is only meaningful near unity.
        center_error = ((np.mean(pts, axis=0) - expected_center)
                        / np.maximum(1.0, np.abs(expected_center)))
        np.testing.assert_allclose(center_error, 0.0, rtol=0.0,
                                   atol=8 * np.finfo(float).eps)
        expected = (np.arange(n) + 0.5) / n
        np.testing.assert_allclose(
            np.sort((pts[:, 0] + 2.0) / 6.0), expected, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(
            np.sort((pts[:, 1] - 3.0) / 6.0), expected, rtol=0.0, atol=1e-15)

    def test_nonintegral_and_collapsed_layouts_are_rejected(self):
        for count in (3.0, True, 0, -1):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    quiet_positions_2d_block(count, 0.0, 1.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            quiet_positions_2d_block(
                2, 1.0e308, math.nextafter(1.0e308, math.inf), 0.0, 1.0)
        with self.assertRaises(ValueError):
            quiet_positions_2d_block(
                2, -float.fromhex("0x1.fffffffffffffp+1023"),
                float.fromhex("0x1.fffffffffffffp+1023"), 0.0, 1.0)

    def test_large_translated_origin_retains_local_strata(self):
        n = 8
        lower = math.ldexp(1.0, 900)
        width = math.ldexp(1.0, 858)
        upper = lower + width
        pts = quiet_positions_2d_block(n, lower, upper, lower, upper)
        self.assertGreater(np.unique(pts[:, 0]).size, 1)
        self.assertGreater(np.unique(pts[:, 1]).size, 1)
        expected = (np.arange(n) + 0.5) / n
        np.testing.assert_allclose(
            np.sort((pts[:, 0] - lower) / width), expected,
            rtol=0.0, atol=1.0e-2)
        np.testing.assert_allclose(
            np.sort((pts[:, 1] - lower) / width), expected,
            rtol=0.0, atol=1.0e-2)


class QuietBlockCellAreaTests(unittest.TestCase):

    def test_perfect_square_tiles_exactly(self):
        # 25 = 5*5: cell area is block area / 25 and density*cell_area*N == density*area.
        area = (1.0 - 0.0) * (2.0 - 0.0)
        cell = quiet_block_cell_area(25, 0.0, 1.0, 0.0, 2.0)
        self.assertAlmostEqual(cell * 25, area)

    def test_non_square_represents_exact_block_area(self):
        # The rank-1 lattice contains exactly N equal-weight particles.  There is
        # no partially populated square layout, so every particle represents
        # exactly block_area/N.
        x_min, x_max, y_min, y_max = 0.0, 4.0, 0.0, 2.0
        block_area = (x_max - x_min) * (y_max - y_min)
        cell = quiet_block_cell_area(50, x_min, x_max, y_min, y_max)
        self.assertAlmostEqual(cell, block_area / 50)
        self.assertAlmostEqual(cell * 50, block_area)

    def test_scaled_area_avoids_false_underflow(self):
        self.assertAlmostEqual(
            quiet_block_cell_area(2, 0.0, 1.0e-300, 0.0, 1.0e300), 0.5)


class QuietRingVolumeTests(unittest.TestCase):

    def test_equal_volume_radial_strata_and_total_volume(self):
        n = 37
        r_min, r_max, z_min, z_max = 2.0, 5.0, -1.0, 3.0
        pts = quiet_positions_rz_block(n, r_min, r_max, z_min, z_max)
        u = (pts[:, 0] ** 2 - r_min ** 2) / (r_max ** 2 - r_min ** 2)
        np.testing.assert_allclose(np.sort(u), (np.arange(n) + 0.5) / n,
                                   rtol=0.0, atol=2e-15)
        represented = quiet_block_ring_volume(
            n, r_min, r_max, z_min, z_max) * n
        self.assertAlmostEqual(
            represented, math.pi * (r_max**2 - r_min**2) * (z_max - z_min))

    def test_negative_radius_and_unrepresentable_volume_are_rejected(self):
        with self.assertRaises(ValueError):
            quiet_positions_rz_block(4, -1.0, 1.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            quiet_block_ring_volume(1, 0.0, 1.0e200, 0.0, 1.0e200)

    def test_large_radius_thin_annulus_keeps_equal_volume_strata(self):
        n = 8
        r_min = math.ldexp(1.0, 900)
        width = math.ldexp(1.0, 858)
        r_max = r_min + width
        pts = quiet_positions_rz_block(n, r_min, r_max, 0.0, 1.0)
        # Evaluate the normalized r^2 coordinate in factored form so the oracle
        # itself neither squares the large radii nor subtracts adjacent squares.
        radial_offset = pts[:, 0] - r_min
        u = ((radial_offset / width)
             * ((pts[:, 0] + r_min) / (r_max + r_min)))
        np.testing.assert_allclose(
            np.sort(u), (np.arange(n) + 0.5) / n,
            rtol=0.0, atol=1.5e-2)


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

    def test_pairs_are_local_antithetic_pairs(self):
        drift = np.asarray([1.0, -2.0, 0.5])
        v = maxwellian(9, 2.0, drift=drift, seed=4)
        for i in range(0, 8, 2):
            np.testing.assert_allclose(v[i] + v[i + 1], 2.0 * drift,
                                       rtol=0.0, atol=4 * np.finfo(float).eps)
        np.testing.assert_array_equal(v[-1], drift)

    def test_count_must_be_an_exact_nonnegative_integer(self):
        for count in (2.0, True, -1):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    maxwellian(count, 1.0)

    def test_finite_inputs_that_overflow_sampled_velocities_are_rejected(self):
        largest = np.finfo(float).max
        # Antithetic pairing guarantees that, for each nonzero draw, one member
        # adds a positive O(largest) fluctuation to the positive maximum drift.
        with self.assertRaises(ValueError):
            maxwellian(2, largest, drift=(largest, largest, largest), seed=0)


if __name__ == "__main__":
    unittest.main()
