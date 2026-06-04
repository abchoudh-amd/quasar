"""Unit tests for the axisymmetric (r-z) PIC CFL helpers.

Covers ``cyl_cfl_limit`` / ``cyl_cfl_dt`` in :mod:`quasar.pic.numerics`.

The closed form asserted against (read from the source) is::

    cyl_cfl_limit(dr, dz, r_min) = 1 / (c * sqrt(1/dr^2 + 1/dz^2))
    cyl_cfl_dt(dr, dz, r_min)    = safety * cyl_cfl_limit(...)   # safety default 0.5

``r_min`` is accepted for forward-compatibility (near-axis tightening) but does
not currently change the returned value for any ``r_min >= 0``; it only guards
against a degenerate ``r_min == 0`` (there is no division by it).
"""

import math
import unittest

from quasar.pic.numerics import C_LIGHT, cyl_cfl_dt, cyl_cfl_limit


class CylCflLimitTests(unittest.TestCase):

    def test_matches_closed_form(self):
        dr, dz, r_min = 0.01, 0.02, 0.005
        expected = 1.0 / (C_LIGHT * math.sqrt(1.0 / dr**2 + 1.0 / dz**2))
        self.assertAlmostEqual(cyl_cfl_limit(dr, dz, r_min), expected, places=18)

    def test_custom_wave_speed_scales_inversely(self):
        dr, dz, r_min = 0.01, 0.01, 0.005
        base = cyl_cfl_limit(dr, dz, r_min, c=1.0)
        # Doubling c halves the limit.
        self.assertAlmostEqual(cyl_cfl_limit(dr, dz, r_min, c=2.0), base / 2.0,
                               places=18)

    def test_smaller_dr_gives_smaller_limit(self):
        dz, r_min = 0.02, 0.005
        coarse = cyl_cfl_limit(0.02, dz, r_min)
        fine = cyl_cfl_limit(0.01, dz, r_min)
        self.assertLess(fine, coarse)

    def test_smaller_dz_gives_smaller_limit(self):
        dr, r_min = 0.02, 0.005
        coarse = cyl_cfl_limit(dr, 0.02, r_min)
        fine = cyl_cfl_limit(dr, 0.01, r_min)
        self.assertLess(fine, coarse)

    def test_r_min_zero_is_safe_and_matches_positive(self):
        dr, dz = 0.01, 0.02
        # r_min == 0 must not raise (guarded; no division by r_min).
        value_zero = cyl_cfl_limit(dr, dz, 0.0)
        self.assertTrue(math.isfinite(value_zero))
        # r_min does not (yet) affect the value: 0.0 matches a small positive r_min.
        value_pos = cyl_cfl_limit(dr, dz, 0.005)
        self.assertAlmostEqual(value_zero, value_pos, places=18)

    def test_r_min_does_not_change_value(self):
        dr, dz = 0.01, 0.02
        # Two different positive r_min values give the identical limit.
        self.assertAlmostEqual(cyl_cfl_limit(dr, dz, 0.001),
                               cyl_cfl_limit(dr, dz, 0.5), places=18)


class CylCflDtTests(unittest.TestCase):

    def test_default_safety_is_half(self):
        dr, dz, r_min = 0.01, 0.02, 0.005
        self.assertAlmostEqual(cyl_cfl_dt(dr, dz, r_min),
                               0.5 * cyl_cfl_limit(dr, dz, r_min), places=18)

    def test_safety_factor_scales_linearly(self):
        dr, dz, r_min = 0.01, 0.02, 0.005
        limit = cyl_cfl_limit(dr, dz, r_min)
        self.assertAlmostEqual(cyl_cfl_dt(dr, dz, r_min, safety=0.25),
                               0.25 * limit, places=18)
        self.assertAlmostEqual(cyl_cfl_dt(dr, dz, r_min, safety=1.0),
                               limit, places=18)

    def test_custom_wave_speed_scales_inversely(self):
        dr, dz, r_min = 0.01, 0.01, 0.005
        base = cyl_cfl_dt(dr, dz, r_min, c=1.0)
        self.assertAlmostEqual(cyl_cfl_dt(dr, dz, r_min, c=2.0), base / 2.0,
                               places=18)

    def test_smaller_dr_gives_smaller_dt(self):
        dz, r_min = 0.02, 0.005
        self.assertLess(cyl_cfl_dt(0.01, dz, r_min), cyl_cfl_dt(0.02, dz, r_min))

    def test_r_min_zero_is_safe(self):
        dr, dz = 0.01, 0.02
        value_zero = cyl_cfl_dt(dr, dz, 0.0)
        self.assertTrue(math.isfinite(value_zero))
        self.assertAlmostEqual(value_zero, cyl_cfl_dt(dr, dz, 0.005), places=18)


if __name__ == "__main__":
    unittest.main()
