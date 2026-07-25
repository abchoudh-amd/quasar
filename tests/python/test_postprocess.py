"""Smoke tests for ``quasar.coil.postprocess``.

The data-shaping helpers (``magnitude``, ``reshape_to_grid``, slice
helpers) are tested unconditionally; the matplotlib-based plotting
helpers are only exercised when matplotlib happens to be installed.
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from quasar.coil import postprocess


def _has_matplotlib() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


class MagnitudeTest(unittest.TestCase):

    def test_two_dim_input(self):
        B = np.array([[3.0, 4.0, 0.0],
                      [0.0, 0.0, 5.0]])
        m = postprocess.magnitude(B)
        self.assertEqual(m.shape, (2,))
        np.testing.assert_allclose(m, [5.0, 5.0])

    def test_higher_dim_input(self):
        B = np.zeros((2, 3, 3))
        B[..., 0] = 1.0
        B[..., 1] = 2.0
        B[..., 2] = 2.0
        m = postprocess.magnitude(B)
        self.assertEqual(m.shape, (2, 3))
        np.testing.assert_allclose(m, 3.0)

    def test_rejects_wrong_last_axis(self):
        with self.assertRaises(ValueError):
            postprocess.magnitude(np.zeros((4, 2)))
        with self.assertRaises(ValueError):
            postprocess.magnitude(np.asarray(1.0))

    def test_large_finite_vectors_do_not_false_overflow(self):
        actual = postprocess.magnitude(np.asarray([[1.0e308, 1.0e308, 0.0]]))[0]
        self.assertTrue(np.isfinite(actual))
        self.assertAlmostEqual(actual / 1.0e308, np.sqrt(2.0), places=14)

    def test_rejects_nonreal_nonfinite_and_unrepresentable_vectors(self):
        invalid = (
            np.asarray([[np.nan, 0.0, 0.0]]),
            np.asarray([[1.0 + 1.0j, 0.0, 0.0]]),
            np.asarray([[np.finfo(np.float64).max,
                         np.finfo(np.float64).max, 0.0]]),
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    postprocess.magnitude(values)


class ReshapeToGridTest(unittest.TestCase):

    def test_round_trips_with_x_fastest_layout(self):
        nx, ny, nz = 3, 2, 2
        # Build a flat (M, 3) array where each entry encodes (ix, iy, iz)
        # explicitly, matching the x-fastest linear layout the C++ side uses.
        B = np.zeros((nx * ny * nz, 3))
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    flat = ix + nx * (iy + ny * iz)
                    B[flat] = (float(ix), float(iy), float(iz))

        grid = postprocess.reshape_to_grid(B, [nx, ny, nz])
        self.assertEqual(grid.shape, (nz, ny, nx, 3))

        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    np.testing.assert_allclose(
                        grid[iz, iy, ix],
                        [float(ix), float(iy), float(iz)],
                    )

    def test_rejects_wrong_size(self):
        B = np.zeros((10, 3))
        with self.assertRaises(ValueError):
            postprocess.reshape_to_grid(B, [3, 2, 2])

    def test_rejects_wrong_input_shape(self):
        with self.assertRaises(ValueError):
            postprocess.reshape_to_grid(np.zeros((12,)), [3, 2, 2])

    def test_rejects_nonintegral_or_nonpositive_dims(self):
        for dims in ([2.5, 2, 1], [0, 2, 1]):
            with self.subTest(dims=dims):
                with self.assertRaises(ValueError):
                    postprocess.reshape_to_grid(np.zeros((4, 3)), dims)


class SliceHelpersTest(unittest.TestCase):

    def setUp(self):
        nx, ny, nz = 3, 2, 2
        flat = np.zeros((nx * ny * nz, 3))
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    flat_idx = ix + nx * (iy + ny * iz)
                    flat[flat_idx] = (ix, iy, iz)
        self.grid = postprocess.reshape_to_grid(flat, [nx, ny, nz])

    def test_slice_xy_returns_constant_z(self):
        slab = postprocess.slice_xy(self.grid, iz=1)
        self.assertEqual(slab.shape, (2, 3, 3))
        np.testing.assert_allclose(slab[..., 2], 1.0)

    def test_slice_xz_returns_constant_y(self):
        slab = postprocess.slice_xz(self.grid, iy=0)
        self.assertEqual(slab.shape, (2, 3, 3))
        np.testing.assert_allclose(slab[..., 1], 0.0)

    def test_slice_yz_returns_constant_x(self):
        slab = postprocess.slice_yz(self.grid, ix=2)
        self.assertEqual(slab.shape, (2, 2, 3))
        np.testing.assert_allclose(slab[..., 0], 2.0)

    def test_slice_helpers_reject_lossy_or_boolean_indices(self):
        for index in (0.5, True):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    postprocess.slice_xy(self.grid, iz=index)


@unittest.skipUnless(_has_matplotlib(), "matplotlib not installed")
class PlotHelpersTest(unittest.TestCase):

    def test_plot_magnitude_slice_creates_figure(self):
        flat = np.tile([[1.0, 0.0, 0.0]], (4 * 3 * 2, 1))
        grid = postprocess.reshape_to_grid(flat, [4, 3, 2])
        fig, ax = postprocess.plot_magnitude_slice(grid, axis="z")
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_magnitude_slice_rejects_fractional_index(self):
        flat = np.tile([[1.0, 0.0, 0.0]], (4 * 3 * 2, 1))
        grid = postprocess.reshape_to_grid(flat, [4, 3, 2])
        with self.assertRaises(ValueError):
            postprocess.plot_magnitude_slice(grid, axis="z", index=0.5)

    def test_plot_line_profile_creates_figure(self):
        pos = np.linspace([0, 0, 0], [1, 0, 0], 5)
        B = np.tile([[1.0, 2.0, 3.0]], (5, 1))
        fig, ax = postprocess.plot_line_profile(pos, B, component="z")
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_line_component_does_not_eagerly_form_magnitude(self):
        largest = np.finfo(np.float64).max
        fig, ax = postprocess.plot_line_profile(
            np.asarray([0.0]), np.asarray([[largest, largest, 0.0]]),
            component="x")
        self.assertIsNotNone(fig)
        self.assertEqual(ax.lines[0].get_ydata()[0], largest)

    def test_plot_line_integer_positions_do_not_wrap_differences(self):
        positions = np.asarray([
            [np.iinfo(np.int64).min, 0, 0],
            [np.iinfo(np.int64).max, 0, 0],
        ], dtype=np.int64)
        fig, ax = postprocess.plot_line_profile(
            positions, np.zeros((2, 3)), component="x")
        self.assertIsNotNone(fig)
        distance = ax.lines[0].get_xdata()[1]
        self.assertGreater(distance, 1.0e19)


class RequireMatplotlibTest(unittest.TestCase):
    """Plot helpers should raise a clear ImportError if matplotlib missing."""

    def test_import_error_message_is_actionable(self):
        if _has_matplotlib():
            self.skipTest("matplotlib is installed; cannot verify error path")
        with self.assertRaises(ImportError) as ctx:
            postprocess.plot_magnitude_slice(
                np.zeros((1, 1, 1, 3)), axis="z")
        self.assertIn("matplotlib", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
