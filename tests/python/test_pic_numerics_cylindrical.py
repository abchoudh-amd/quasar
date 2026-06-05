"""Unit tests for the axisymmetric (r-z) PIC CFL helpers.

Covers ``cyl_cfl_limit`` / ``cyl_cfl_dt`` in :mod:`quasar.pic.numerics`.

The closed form asserted against (read from the source) is::

    cyl_cfl_limit(dr, dz) = 1 / (c * sqrt(1/dr^2 + 1/dz^2))
    cyl_cfl_dt(dr, dz)    = safety * cyl_cfl_limit(...)   # safety default 0.5

For the conservative m=0 scheme the volume-weighted curl is mimetic, so the
stability bound is exactly the planar 2nd-order Yee limit (no near-axis
tightening); ``cyl_cfl_limit`` therefore delegates to ``cfl_limit(..., order=2)``.
"""

import math
import unittest

from quasar.pic.numerics import C_LIGHT, cfl_limit, cyl_cfl_dt, cyl_cfl_limit


class CylCflLimitTests(unittest.TestCase):

    def test_matches_closed_form(self):
        dr, dz = 0.01, 0.02
        expected = 1.0 / (C_LIGHT * math.sqrt(1.0 / dr**2 + 1.0 / dz**2))
        self.assertAlmostEqual(cyl_cfl_limit(dr, dz), expected, places=18)

    def test_matches_cartesian_order2_limit(self):
        # The m=0 cylindrical bound is the planar 2nd-order Yee limit.
        dr, dz = 0.01, 0.02
        self.assertAlmostEqual(cyl_cfl_limit(dr, dz),
                               cfl_limit(dr, dz, fdtd_order=2), places=18)

    def test_custom_wave_speed_scales_inversely(self):
        dr, dz = 0.01, 0.01
        base = cyl_cfl_limit(dr, dz, c=1.0)
        # Doubling c halves the limit.
        self.assertAlmostEqual(cyl_cfl_limit(dr, dz, c=2.0), base / 2.0,
                               places=18)

    def test_smaller_dr_gives_smaller_limit(self):
        dz = 0.02
        coarse = cyl_cfl_limit(0.02, dz)
        fine = cyl_cfl_limit(0.01, dz)
        self.assertLess(fine, coarse)

    def test_smaller_dz_gives_smaller_limit(self):
        dr = 0.02
        coarse = cyl_cfl_limit(dr, 0.02)
        fine = cyl_cfl_limit(dr, 0.01)
        self.assertLess(fine, coarse)


class CylCflDtTests(unittest.TestCase):

    def test_default_safety_is_half(self):
        dr, dz = 0.01, 0.02
        self.assertAlmostEqual(cyl_cfl_dt(dr, dz),
                               0.5 * cyl_cfl_limit(dr, dz), places=18)

    def test_safety_factor_scales_linearly(self):
        dr, dz = 0.01, 0.02
        limit = cyl_cfl_limit(dr, dz)
        self.assertAlmostEqual(cyl_cfl_dt(dr, dz, safety=0.25),
                               0.25 * limit, places=18)
        self.assertAlmostEqual(cyl_cfl_dt(dr, dz, safety=1.0),
                               limit, places=18)

    def test_custom_wave_speed_scales_inversely(self):
        dr, dz = 0.01, 0.01
        base = cyl_cfl_dt(dr, dz, c=1.0)
        self.assertAlmostEqual(cyl_cfl_dt(dr, dz, c=2.0), base / 2.0,
                               places=18)

    def test_smaller_dr_gives_smaller_dt(self):
        dz = 0.02
        self.assertLess(cyl_cfl_dt(0.01, dz), cyl_cfl_dt(0.02, dz))


if __name__ == "__main__":
    unittest.main()
