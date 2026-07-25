"""Unit tests for the PIC CFL helpers (quasar.pic.numerics)."""

import math
import unittest

from quasar.pic.numerics import C_LIGHT, cfl_dt, cfl_limit, j0_zero


class BesselZeroTests(unittest.TestCase):

    def test_low_tabulated_and_high_asymptotic_zeros(self):
        self.assertAlmostEqual(j0_zero(1), 2.404825557695773, places=15)
        self.assertAlmostEqual(j0_zero(6), 18.071063967910922, places=8)
        self.assertAlmostEqual(j0_zero(10), 30.634606468431976, places=9)

    def test_rejects_nonpositive_and_noninteger_indices(self):
        for index in (0, -1, 1.5, True):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    j0_zero(index)


class CflLimitTests(unittest.TestCase):

    def test_order2_matches_closed_form(self):
        dx, dy = 0.01, 0.02
        expected = 1.0 / (C_LIGHT * math.sqrt(1.0 / dx**2 + 1.0 / dy**2))
        self.assertAlmostEqual(cfl_limit(dx, dy, fdtd_order=2), expected, places=18)

    def test_order4_is_stricter_by_seven_sixths(self):
        dx, dy = 0.01, 0.01
        lim2 = cfl_limit(dx, dy, fdtd_order=2)
        lim4 = cfl_limit(dx, dy, fdtd_order=4)
        # The 4th-order curl tightens the limit by a factor 7/6.
        self.assertAlmostEqual(lim4, lim2 * 6.0 / 7.0, places=18)

    def test_custom_wave_speed_scales_inversely(self):
        dx, dy = 0.01, 0.01
        base = cfl_limit(dx, dy, c=1.0)
        self.assertAlmostEqual(cfl_limit(dx, dy, c=2.0), base / 2.0, places=18)

    def test_unsupported_order_raises(self):
        with self.assertRaises(ValueError):
            cfl_limit(0.01, 0.01, fdtd_order=3)

    def test_rejects_invalid_spacing_and_wave_speed(self):
        for dx, dy, c in ((0.0, 1.0, 1.0), (1.0, -1.0, 1.0),
                          (float("inf"), 1.0, 1.0), (1.0, 1.0, 0.0),
                          (1.0, 1.0, float("nan"))):
            with self.subTest(dx=dx, dy=dy, c=c):
                with self.assertRaises(ValueError):
                    cfl_limit(dx, dy, c=c)

    def test_large_aspect_ratio_does_not_overflow_inverse_square(self):
        limit = cfl_limit(1.0e-200, 1.0e100, c=1.0)
        self.assertTrue(math.isfinite(limit))
        self.assertGreater(limit, 0.0)
        self.assertAlmostEqual(limit / 1.0e-200, 1.0, places=14)

    def test_maximum_equal_spacing_with_subunit_wave_speed_is_finite(self):
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        limit = cfl_limit(maximum, maximum, c=0.75)
        expected = (maximum / math.sqrt(2.0)) / 0.75
        self.assertTrue(math.isfinite(limit))
        self.assertAlmostEqual(limit / expected, 1.0, places=14)


class CflDtTests(unittest.TestCase):

    def test_default_safety_is_half(self):
        dx, dy = 0.01, 0.02
        self.assertAlmostEqual(cfl_dt(dx, dy), 0.5 * cfl_limit(dx, dy), places=18)

    def test_safety_factor_scales(self):
        dx, dy = 0.01, 0.02
        self.assertAlmostEqual(cfl_dt(dx, dy, safety=0.25),
                               0.25 * cfl_limit(dx, dy), places=18)

    def test_unsupported_order_raises(self):
        with self.assertRaises(ValueError):
            cfl_dt(0.01, 0.01, fdtd_order=5)

    def test_rejects_invalid_safety_factor(self):
        for safety in (0.0, -0.1, 1.01, float("inf"), float("nan")):
            with self.subTest(safety=safety):
                with self.assertRaises(ValueError):
                    cfl_dt(0.01, 0.01, safety=safety)


if __name__ == "__main__":
    unittest.main()
