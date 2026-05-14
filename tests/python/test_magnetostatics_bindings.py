"""Smoke test for the pybind11 bindings.

Uses the stdlib ``unittest`` (pytest is not installed on every node) and
re-uses the analytical references from Phase 1.F to confirm the Python
surface matches the C++ tests numerically.
"""

import math
import os
import sys
import unittest

import numpy as np

from quasar import Vec3, mu0_over_4pi  # noqa: E402
from quasar.coil import (  # noqa: E402
    BiotSavartEvaluator,
    ConductorSystem,
    Filament,
    LineProbe,
    ObservationGrid,
    PlaneSlice,
    PointCloud,
    circular_loop,
)


def has_hip_runtime() -> bool:
    """Mirror of quasar::backend::has_hip_runtime() based on env vars.

    The CTest wrapper sets QUASAR_HAS_HIP_RUNTIME=1 when the C++ probe
    detected a device at configure time; the test will SKIP otherwise.
    """
    return os.environ.get("QUASAR_HAS_HIP_RUNTIME", "0") == "1"


class CoreBindingsTest(unittest.TestCase):

    def test_vec3_construct_and_compare(self):
        v = Vec3(1.0, 2.0, 3.0)
        self.assertEqual(v.x, 1.0)
        self.assertEqual(v.y, 2.0)
        self.assertEqual(v.z, 3.0)
        self.assertEqual(v, Vec3(1.0, 2.0, 3.0))

    def test_constants_present(self):
        self.assertAlmostEqual(mu0_over_4pi, 1e-7, places=12)


class GeometryBindingsTest(unittest.TestCase):

    def test_circular_loop_round_trip_to_conductor_system(self):
        loop = circular_loop(
            center=Vec3(0, 0, 0), axis=Vec3(0, 0, 1),
            radius_m=0.05, n_segments=16, current_A=2.0,
        )
        self.assertEqual(len(loop.points), 17)  # n + 1 (closed)
        cs = ConductorSystem()
        cs.add(loop)
        self.assertEqual(len(cs), 1)


class ObservationSetBindingsTest(unittest.TestCase):

    def test_observation_grid_to_point_cloud(self):
        g = ObservationGrid()
        g.origin = Vec3(0, 0, 0)
        g.spacing = Vec3(0.1, 0.1, 0.1)
        g.dims = [2, 2, 2]
        pc = g.to_point_cloud()
        self.assertEqual(len(pc), 8)

    def test_plane_slice_to_point_cloud(self):
        s = PlaneSlice()
        s.origin = Vec3(0, 0, 0)
        s.u_step = Vec3(0.5, 0, 0)
        s.v_step = Vec3(0, 0.5, 0)
        s.nu = 3
        s.nv = 4
        pc = s.to_point_cloud()
        self.assertEqual(len(pc), 12)

    def test_line_probe_to_point_cloud(self):
        l = LineProbe()
        l.start = Vec3(0, 0, 0)
        l.end = Vec3(1, 0, 0)
        l.n_points = 5
        pc = l.to_point_cloud()
        self.assertEqual(len(pc), 5)


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class BiotSavartBindingsTest(unittest.TestCase):

    def test_finite_segment_matches_closed_form(self):
        """Mirror of C++ test_finite_segment, exercised through Python."""
        L = 2.0
        I = 1.0
        H = L / 2

        cs = ConductorSystem()
        cs.add(Filament(
            name="seg", current_A=I,
            points=[Vec3(0, 0, -H), Vec3(0, 0, +H)],
        ))

        eval_ = BiotSavartEvaluator()

        for d in (0.1, 1.0, 10.0):
            with self.subTest(d=d):
                pc = PointCloud()
                pc.add(Vec3(d, 0, 0))
                B = eval_.evaluate_B(cs, pc)

                self.assertEqual(B.shape, (1, 3))
                ref_By = mu0_over_4pi * I * 2 * H / (d * math.sqrt(d * d + H * H))
                self.assertLess(abs(B[0, 0]), 1e-14)
                self.assertLess(abs(B[0, 2]), 1e-14)
                self.assertAlmostEqual(B[0, 1], ref_By,
                                       delta=1e-10 * abs(ref_By))

    def test_evaluate_grad_B_returns_n_3_3_array_satisfying_div_b_zero(self):
        """grad_B[i, :, :] should be a 3x3 Jacobian whose trace ~ 0 (div B = 0)."""
        cs = ConductorSystem()
        cs.add(circular_loop(
            center=Vec3(0, 0, 0), axis=Vec3(0, 0, 1),
            radius_m=0.07, n_segments=128, current_A=1.0,
        ))
        pc = PointCloud()
        pc.add(Vec3(0.02, -0.01, 0.03))
        pc.add(Vec3(0.05, 0.02, -0.04))
        pc.add(Vec3(0.0, 0.0, 0.05))

        eval_ = BiotSavartEvaluator()
        gradB = eval_.evaluate_grad_B(cs, pc)
        self.assertEqual(gradB.shape, (3, 3, 3))

        for k in range(gradB.shape[0]):
            jac = gradB[k]
            trace = jac[0, 0] + jac[1, 1] + jac[2, 2]
            scale = float((jac * jac).sum()) ** 0.5
            self.assertLess(abs(trace), 1e-6 * scale,
                            msg=f"div B at point {k} was {trace}")

    def test_circular_loop_on_axis_matches_closed_form(self):
        """Mirror of C++ test_circular_loop_on_axis at N=256, z=0.05."""
        R = 0.1
        I = 1.0
        z = 0.05
        cs = ConductorSystem()
        cs.add(circular_loop(
            center=Vec3(0, 0, 0), axis=Vec3(0, 0, 1),
            radius_m=R, n_segments=256, current_A=I,
        ))
        pc = PointCloud()
        pc.add(Vec3(0, 0, z))

        B = BiotSavartEvaluator().evaluate_B(cs, pc)
        from quasar import mu0
        ref_Bz = mu0 * I * R * R / (2 * (R * R + z * z) ** 1.5)
        self.assertEqual(B.shape, (1, 3))
        self.assertAlmostEqual(B[0, 2], ref_Bz, delta=1e-4 * abs(ref_Bz))


if __name__ == "__main__":
    print("python:", sys.executable, file=sys.stderr)
    unittest.main()
