"""Smoke test for the pybind11 bindings.

Uses the stdlib ``unittest`` (pytest is not installed on every node) and
reuses the analytical finite-segment and circular-loop references to confirm
the Python surface matches the C++ tests numerically.
"""

import math
import os
import sys
import unittest


from quasar import Vec3, mu0_over_4pi  # noqa: E402
from quasar import _core  # noqa: E402
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
        probe = LineProbe()
        probe.start = Vec3(0, 0, 0)
        probe.end = Vec3(1, 0, 0)
        probe.n_points = 5
        pc = probe.to_point_cloud()
        self.assertEqual(len(pc), 5)


class AnalyticEvaluatorBindingsTest(unittest.TestCase):

    def test_registry_names_are_exposed_and_sorted(self):
        names = _core.magnetostatics.field_evaluator_names()
        self.assertEqual(names, sorted(names))
        self.assertTrue({"biot_savart", "uniform", "dipole", "gradient",
                         "file_grid"}.issubset(names))
        self.assertTrue(
            _core.magnetostatics.field_evaluator_provides_vector_potential(
                "biot_savart"))
        self.assertFalse(
            _core.magnetostatics.field_evaluator_provides_vector_potential(
                "uniform"))
        for name in ("biot_savart", "uniform", "dipole", "gradient",
                     "file_grid"):
            with self.subTest(name=name):
                self.assertTrue(
                    _core.magnetostatics.field_evaluator_provides_grad_B(name))

        uniform = _core.magnetostatics.create_field_evaluator("uniform")
        self.assertTrue(uniform.provides_grad_B)

    def test_uniform_electric_field_is_exposed(self):
        eval_ = _core.magnetostatics.UniformEvaluator(
            b0=Vec3(0.0, 0.0, 0.0), e0=Vec3(1.0, -2.0, 3.0))
        source = ConductorSystem()
        points = PointCloud()
        points.add(Vec3(0.5, -1.0, 4.0))
        field = eval_.evaluate_E(source, points)
        self.assertEqual(field.shape, (1, 3))
        self.assertEqual(field[0].tolist(), [1.0, -2.0, 3.0])

    def test_gradient_evaluator_uses_bound_matrix_constructor(self):
        eval_ = _core.magnetostatics.GradientEvaluator(
            b0=Vec3(0.0, 0.0, 1.0),
            grad=[[1.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0],
                  [0.0, 0.0, -3.0]],
        )
        cs = ConductorSystem()
        pc = PointCloud()
        pc.add(Vec3(0.5, 0.25, 0.0))

        B = eval_.evaluate_B(cs, pc)
        self.assertEqual(B.shape, (1, 3))
        self.assertAlmostEqual(B[0, 0], 0.5)
        self.assertAlmostEqual(B[0, 1], 0.5)
        self.assertAlmostEqual(B[0, 2], 1.0)


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class BiotSavartBindingsTest(unittest.TestCase):

    def test_finite_segment_matches_closed_form(self):
        """Mirror of C++ test_finite_segment, exercised through Python."""
        L = 2.0
        current = 1.0
        H = L / 2

        cs = ConductorSystem()
        cs.add(Filament(
            name="seg", current_A=current,
            points=[Vec3(0, 0, -H), Vec3(0, 0, +H)],
        ))

        eval_ = BiotSavartEvaluator()

        for d in (0.1, 1.0, 10.0):
            with self.subTest(d=d):
                pc = PointCloud()
                pc.add(Vec3(d, 0, 0))
                B = eval_.evaluate_B(cs, pc)

                self.assertEqual(B.shape, (1, 3))
                ref_By = mu0_over_4pi * current * 2 * H / (d * math.sqrt(d * d + H * H))
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
        current = 1.0
        z = 0.05
        cs = ConductorSystem()
        cs.add(circular_loop(
            center=Vec3(0, 0, 0), axis=Vec3(0, 0, 1),
            radius_m=R, n_segments=256, current_A=current,
        ))
        pc = PointCloud()
        pc.add(Vec3(0, 0, z))

        B = BiotSavartEvaluator().evaluate_B(cs, pc)
        from quasar import mu0
        ref_Bz = mu0 * current * R * R / (2 * (R * R + z * z) ** 1.5)
        self.assertEqual(B.shape, (1, 3))
        self.assertAlmostEqual(B[0, 2], ref_Bz, delta=1e-4 * abs(ref_Bz))


if __name__ == "__main__":
    print("python:", sys.executable, file=sys.stderr)
    unittest.main()
