"""Unit tests for the axisymmetric (r-z) PIC CFL helpers.

Covers ``cyl_cfl_limit`` / ``cyl_cfl_dt`` in :mod:`quasar.pic.numerics`.

The order-two closed form asserted against (read from the source) is::

    cyl_cfl_limit(dr, dz) = 1 / (c * sqrt(1/dr^2 + 1/dz^2))
    cyl_cfl_dt(dr, dz)    = safety * cyl_cfl_limit(...)   # safety default 0.5

For the conservative m=0 scheme, order two has the planar Yee limit. The
fourth-order regular-axis closure needs the proved radial bound ``35/6`` rather
than the slightly smaller Cartesian symbol ``49/9``.
"""

import math
import unittest
from fractions import Fraction

import numpy as np

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

    def test_order4_uses_proven_axis_bound_and_is_tighter(self):
        dr, dz = 0.01, 0.02
        order2 = cyl_cfl_limit(dr, dz, fdtd_order=2)
        order4 = cyl_cfl_limit(dr, dz, fdtd_order=4)
        expected = 1.0 / (C_LIGHT * math.sqrt(
            35.0 / (24.0 * dr * dr) + 49.0 / (36.0 * dz * dz)))
        self.assertAlmostEqual(order4, expected, places=18)
        self.assertLess(order4, cfl_limit(dr, dz, fdtd_order=4))
        self.assertLess(order4, order2)

    def test_order4_axis_matrix_small_grid_spectra_obey_bound(self):
        # Independently assemble -A4*B4 on an axis-touching, unit-spaced radial
        # grid with outer PEC parity. nx=2 is the smallest solver-valid grid.
        def radial_wave_matrix(nx):
            def face_value(k, values):
                if k < 0:
                    return -values[-k]
                if k > nx:
                    return values[2 * nx - k]
                return values[k]

            def cell_value(k, values):
                if k < 0:
                    return values[-1 - k]
                if k >= nx:
                    return -values[2 * nx - 1 - k]
                return values[k]

            a = [[Fraction(0) for _ in range(nx + 1)]
                 for _ in range(nx)]
            for i in range(nx):
                radius = Fraction(2 * i + 1, 2)
                for basis in range(nx + 1):
                    values = [Fraction(0) for _ in range(nx + 1)]
                    values[basis] = Fraction(1)

                    def flux(k):
                        return Fraction(k) * face_value(k, values)

                    a[i][basis] = (
                        Fraction(9, 8) * (flux(i + 1) - flux(i))
                        - Fraction(1, 24) * (flux(i + 2) - flux(i - 1))
                    ) / radius

            b = [[Fraction(0) for _ in range(nx)]
                 for _ in range(nx + 1)]
            for face in range(nx + 1):
                for basis in range(nx):
                    values = [Fraction(0) for _ in range(nx)]
                    values[basis] = Fraction(1)
                    b[face][basis] = (
                        Fraction(9, 8)
                        * (cell_value(face, values)
                           - cell_value(face - 1, values))
                        - Fraction(1, 24)
                        * (cell_value(face + 1, values)
                           - cell_value(face - 2, values)))

            return [[-sum(a[i][k] * b[k][j] for k in range(nx + 1))
                     for j in range(nx)] for i in range(nx)]

        expected = {
            2: 5.444010369561416,
            3: 5.4444536388582545,
            4: 5.444444234189444,
        }
        matrices = {}
        for nx, reference in expected.items():
            matrix = radial_wave_matrix(nx)
            matrices[nx] = matrix
            radius = float(np.max(np.linalg.eigvals(
                np.asarray(matrix, dtype=np.float64)).real))
            self.assertAlmostEqual(radius, reference, places=13)
            self.assertLessEqual(radius, 35.0 / 6.0)

        # Exact three-cell evidence against the old Cartesian claim:
        # p(49/9)=det(lambda I-M)=-7/69120 < 0.
        matrix = matrices[3]
        lam = Fraction(49, 9)
        characteristic = [
            [lam * (i == j) - matrix[i][j] for j in range(3)]
            for i in range(3)
        ]
        determinant = (
            characteristic[0][0]
            * (characteristic[1][1] * characteristic[2][2]
               - characteristic[1][2] * characteristic[2][1])
            - characteristic[0][1]
            * (characteristic[1][0] * characteristic[2][2]
               - characteristic[1][2] * characteristic[2][0])
            + characteristic[0][2]
            * (characteristic[1][0] * characteristic[2][1]
               - characteristic[1][1] * characteristic[2][0]))
        self.assertEqual(determinant, Fraction(-7, 69120))
        self.assertGreater(expected[3], float(lam))

    def test_custom_wave_speed_scales_inversely(self):
        dr, dz = 0.01, 0.01
        base = cyl_cfl_limit(dr, dz, c=1.0)
        # Doubling c halves the limit.
        self.assertAlmostEqual(cyl_cfl_limit(dr, dz, c=2.0), base / 2.0,
                               places=18)

    def test_order4_extreme_aspect_ratios_remain_representable(self):
        radial_dominated = cyl_cfl_limit(
            1.0e-300, 1.0e300, c=1.0, fdtd_order=4)
        axial_dominated = cyl_cfl_limit(
            1.0e300, 1.0e-300, c=1.0, fdtd_order=4)
        self.assertGreater(radial_dominated, 0.0)
        self.assertGreater(axial_dominated, 0.0)
        self.assertAlmostEqual(
            radial_dominated / (1.0e-300 / math.sqrt(35.0 / 24.0)),
            1.0, places=15)
        self.assertAlmostEqual(
            axial_dominated / (1.0e-300 / math.sqrt(49.0 / 36.0)),
            1.0, places=15)

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

    def test_order4_dt_uses_order4_limit(self):
        dr, dz = 0.01, 0.02
        self.assertAlmostEqual(
            cyl_cfl_dt(dr, dz, fdtd_order=4),
            0.5 * cyl_cfl_limit(dr, dz, fdtd_order=4), places=18)

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
