"""Real-valued NumPy coercion at the C++ binding boundary."""

import os
import unittest

import numpy as np

from quasar import _core


def has_hip_runtime() -> bool:
    return os.environ.get("QUASAR_HAS_HIP_RUNTIME", "0") == "1"


class BackgroundSamplerArrayTests(unittest.TestCase):

    def test_rejects_complex_and_boolean_coordinates(self):
        for values in (np.array([1.0 + 2.0j]), np.array([True])):
            with self.subTest(dtype=values.dtype):
                with self.assertRaises(ValueError):
                    _core.mhd.sample_mhd_background_profile(
                        "uniform", 0, values, np.zeros(1), {})

    def test_accepts_integer_noncontiguous_views(self):
        x = np.arange(24, dtype=np.int32).reshape(3, 8)[:, ::2]
        y = np.arange(48, dtype=np.uint16).reshape(3, 16)[:, 1::4]
        self.assertFalse(x.flags.c_contiguous)
        self.assertFalse(y.flags.c_contiguous)

        sampled = _core.mhd.sample_mhd_background_profile(
            "linear_vacuum", 0, x, y, {"gradient": 2.0, "shear": -0.5})
        np.testing.assert_array_equal(sampled, 2.0 * x - 0.5 * y)


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class SolverArrayTests(unittest.TestCase):

    @staticmethod
    def _mhd_solver(*, background=False):
        mhd = _core.mhd
        cfg = mhd.MhdConfig()
        cfg.grid = mhd.Grid2D(4, 4, 1.0, 1.0, nghost=0)
        cfg.background.enabled = background
        return mhd.MhdSolver2D(cfg)

    @staticmethod
    def _pic_solver():
        pic = _core.pic
        cfg = pic.EmPicConfig()
        cfg.grid = pic.Grid2D(4, 4, 1.0, 1.0, nghost=1)
        return pic.EmPic2D3V(cfg)

    @staticmethod
    def _storage_size(solver):
        grid = solver.grid()
        return (grid.nx + 2 * grid.nghost) * (grid.ny + 2 * grid.nghost)

    def test_mhd_state_and_background_reject_nonreal_dtypes(self):
        solver = self._mhd_solver(background=True)
        n = self._storage_size(solver)
        with self.assertRaises(ValueError):
            solver.seed_state("rho", np.ones(n, dtype=np.complex128))
        with self.assertRaises(ValueError):
            solver.seed_state("rho", np.ones(n, dtype=np.bool_))
        with self.assertRaises(ValueError):
            solver.seed_background("b0z", np.ones(n, dtype=np.complex128))
        with self.assertRaises(ValueError):
            solver.seed_background("b0z", np.ones(n, dtype=np.bool_))

    def test_mhd_state_accepts_integer_noncontiguous_view(self):
        solver = self._mhd_solver()
        n = self._storage_size(solver)
        values = np.arange(2 * n, dtype=np.int32)[::2]
        self.assertFalse(values.flags.c_contiguous)
        solver.seed_state("mx", values)
        np.testing.assert_array_equal(
            solver.state_component_to_host("mx"), values.astype(np.float64))

    def test_pic_field_and_particles_reject_nonreal_dtypes(self):
        solver = self._pic_solver()
        n = solver.storage_size()
        with self.assertRaises(ValueError):
            solver.seed_field("ex", np.zeros(n, dtype=np.complex128))
        with self.assertRaises(ValueError):
            solver.seed_field("ex", np.zeros(n, dtype=np.bool_))

        index = solver.add_species(
            _core.pic.SpeciesConfig("probe", 0.0, 1.0, 2))
        valid = np.array([0.25, 0.75])
        zero = np.zeros(2)
        with self.assertRaises(ValueError):
            solver.set_species_particles(
                index, valid.astype(np.complex128), valid,
                zero, zero, zero, np.ones(2))
        with self.assertRaises(ValueError):
            solver.set_species_particles(
                index, valid, valid, zero, zero, zero,
                np.ones(2, dtype=np.bool_))

    def test_pic_field_rejects_nonfinite_values_before_device_copy(self):
        solver = self._pic_solver()
        n = solver.storage_size()
        for nonfinite in (np.nan, np.inf, -np.inf):
            values = np.zeros(n)
            values[n // 2] = nonfinite
            with self.subTest(nonfinite=nonfinite):
                with self.assertRaisesRegex(ValueError, "finite"):
                    solver.seed_field("ey", values)

        # A failed upload must leave the live device field untouched.
        np.testing.assert_array_equal(
            solver.field_component_to_host("ey"), np.zeros(n))

    def test_pic_upload_accepts_real_noncontiguous_views_and_integers(self):
        solver = self._pic_solver()
        index = solver.add_species(
            _core.pic.SpeciesConfig("probe", 0.0, 1.0, 2))
        positions = np.array([0.25, -9.0, 0.75, -9.0])[::2]
        zeros = np.zeros(4)[::2]
        weights = np.ones(4, dtype=np.uint16)[::2]
        self.assertFalse(positions.flags.c_contiguous)
        self.assertFalse(weights.flags.c_contiguous)

        solver.set_species_particles(
            index, positions, positions, zeros, zeros, zeros, weights)
        snapshot = solver.species_at(index).to_host()
        self.assertEqual(
            set(snapshot),
            {"x", "y", "vx", "vy", "vz", "weight", "alive"})
        np.testing.assert_array_equal(snapshot["x"], positions)
        np.testing.assert_array_equal(snapshot["weight"], weights)


if __name__ == "__main__":
    unittest.main()
