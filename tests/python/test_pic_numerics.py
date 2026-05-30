"""Unit tests for the PIC CFL helpers (quasar.pic.numerics)."""

import math
import unittest

from quasar.pic.numerics import C_LIGHT, cfl_dt, cfl_limit


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


if __name__ == "__main__":
    unittest.main()
