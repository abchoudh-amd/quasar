"""Unit tests for the pure-numpy PIC postprocess helpers.

The helpers are pure NumPy and need neither matplotlib nor a GPU.  In
particular, Yee fields use component- and geometry-specific extents: physical
high faces must survive on wall axes, while only periodic endpoint duplicates
may be removed.
"""

import math
from pathlib import Path
import types
import unittest

import numpy as np

from quasar.pic.postprocess import (
    _archive_plane,
    _geometry_labels,
    _plot_archive,
    field_names,
    field_periodicity,
    reshape_with_ghost,
    rms,
    species_names,
    yee_component_coordinates,
    yee_component_view,
)


class RmsTests(unittest.TestCase):

    def test_known_value(self):
        self.assertAlmostEqual(rms([3.0, 4.0]), math.sqrt(12.5), places=12)

    def test_zeros(self):
        self.assertEqual(rms([0.0, 0.0, 0.0]), 0.0)

    def test_large_finite_values_do_not_false_overflow(self):
        self.assertTrue(math.isfinite(rms([1.0e308, -1.0e308])))
        self.assertAlmostEqual(rms([1.0e308, -1.0e308]) / 1.0e308,
                               1.0, places=14)

    def test_complex_values_use_magnitude_without_discarding_imaginary_part(self):
        self.assertAlmostEqual(rms([3.0 + 4.0j, 0.0]), math.sqrt(12.5),
                               places=12)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            rms([])


class ReshapeWithGhostTests(unittest.TestCase):

    def _ghost_padded(self, nx, ny, g):
        """Flat (nx+2g)*(ny+2g) buffer whose value encodes interior vs ghost: an
        interior cell (i,j) holds 1000*j + i (>= 0); ghost cells hold -1."""
        pitch, height = nx + 2 * g, ny + 2 * g
        buf = np.full(pitch * height, -1.0)
        for j in range(ny):
            for i in range(nx):
                buf[(i + g) + pitch * (j + g)] = 1000 * j + i
        return buf

    def test_strips_ghosts_for_each_width(self):
        nx, ny = 5, 3
        for g in (1, 2):
            flat = self._ghost_padded(nx, ny, g)
            interior = reshape_with_ghost(flat, nx, ny, g)
            self.assertEqual(interior.shape, (ny, nx))
            for j in range(ny):
                for i in range(nx):
                    self.assertEqual(interior[j, i], 1000 * j + i)
            self.assertTrue((interior >= 0).all())  # no ghost (-1) leaked in

    def test_no_ghost_buffer_reshapes_directly(self):
        nx, ny = 4, 2
        flat = np.arange(nx * ny, dtype=float)
        interior = reshape_with_ghost(flat, nx, ny, 0)
        self.assertEqual(interior.shape, (ny, nx))
        np.testing.assert_array_equal(interior, flat.reshape(ny, nx))

    def test_rejects_inconsistent_buffer_or_grid_metadata(self):
        with self.assertRaises(ValueError):
            reshape_with_ghost(np.zeros(8), 4, 2, 1)
        with self.assertRaises(ValueError):
            reshape_with_ghost(np.zeros(8), 4.5, 2, 0)


class YeeComponentViewTests(unittest.TestCase):

    @staticmethod
    def _indexed_storage(nx, ny, g):
        """Encode each padded slot by its logical ``(j, i)`` index."""
        pitch = nx + 2 * g
        buf = np.empty((ny + 2 * g, pitch), dtype=float)
        for j in range(-g, ny + g):
            for i in range(-g, nx + g):
                buf[j + g, i + g] = 1000 * j + i
        return buf.reshape(-1)

    def test_cartesian_extents_match_core_yee_layout(self):
        nx, ny, g = 4, 3, 2
        flat = self._indexed_storage(nx, ny, g)
        expected = {
            "ex": (ny, nx + 1),
            "ey": (ny + 1, nx),
            "ez": (ny, nx),
            "bx": (ny + 1, nx),
            "by": (ny, nx + 1),
            "bz": (ny + 1, nx + 1),
        }
        for component, shape in expected.items():
            with self.subTest(component=component):
                view = yee_component_view(
                    flat, nx, ny, g, component, "cartesian")
                self.assertEqual(view.shape, shape)
                self.assertEqual(view[0, 0], 0.0)
                self.assertEqual(view[-1, -1],
                                 1000 * (shape[0] - 1) + shape[1] - 1)

    def test_cylindrical_extents_follow_radial_parity_layout(self):
        nx, ny, g = 4, 3, 2
        flat = self._indexed_storage(nx, ny, g)
        expected = {
            "ex": (ny, nx + 1),       # Er
            "ey": (ny + 1, nx),       # Ez
            "ez": (ny, nx + 1),       # Ephi
            "bx": (ny + 1, nx + 1),   # Br
            "by": (ny, nx),           # Bz
            "bz": (ny + 1, nx + 1),   # Bphi
        }
        for component, shape in expected.items():
            with self.subTest(component=component):
                view = yee_component_view(
                    flat, nx, ny, g, component, "cylindrical")
                self.assertEqual(view.shape, shape)

    def test_periodic_axes_remove_only_endpoint_duplicates(self):
        nx, ny, g = 4, 3, 1
        flat = self._indexed_storage(nx, ny, g)

        wall_bz = yee_component_view(
            flat, nx, ny, g, "bz", periodic_x=False, periodic_y=False)
        periodic_bz = yee_component_view(
            flat, nx, ny, g, "bz", periodic_x=True, periodic_y=True)
        self.assertEqual(wall_bz.shape, (ny + 1, nx + 1))
        self.assertEqual(wall_bz[-1, -1], 1000 * ny + nx)
        self.assertEqual(periodic_bz.shape, (ny, nx))
        self.assertEqual(periodic_bz[-1, -1], 1000 * (ny - 1) + nx - 1)

        mixed_bz = yee_component_view(
            flat, nx, ny, g, "bz", periodic_x=True, periodic_y=False)
        self.assertEqual(mixed_bz.shape, (ny + 1, nx))
        self.assertEqual(mixed_bz[-1, -1], 1000 * ny + nx - 1)

        # Ex is face-staggered only in x.  Periodicity in y must not remove a
        # row because there is no y endpoint duplicate in this component.
        ex = yee_component_view(
            flat, nx, ny, g, "ex", periodic_x=False, periodic_y=True)
        self.assertEqual(ex.shape, (ny, nx + 1))

    def test_legacy_cell_sized_component_is_accepted_without_inventing_faces(self):
        nx, ny = 4, 3
        flat = np.arange(nx * ny, dtype=float)
        view = yee_component_view(flat, nx, ny, 0, "bz")
        self.assertEqual(view.shape, (ny, nx))
        np.testing.assert_array_equal(view, flat.reshape(ny, nx))

    def test_unknown_geometry_and_component_are_rejected(self):
        flat = self._indexed_storage(4, 3, 1)
        with self.assertRaises(ValueError):
            yee_component_view(flat, 4, 3, 1, "rho")
        with self.assertRaises(ValueError):
            yee_component_view(flat, 4, 3, 1, "bz", "spherical")


class YeeComponentCoordinateTests(unittest.TestCase):

    def test_cartesian_face_and_cell_coordinates_use_physical_extent(self):
        ex_x, ex_y = yee_component_coordinates(
            "ex", "cartesian", 4, 2, 10.0, -2.0, 4.0, 2.0)
        np.testing.assert_allclose(ex_x, [10.0, 11.0, 12.0, 13.0, 14.0])
        np.testing.assert_allclose(ex_y, [-1.5, -0.5])

        ez_x, ez_y = yee_component_coordinates(
            "ez", "cartesian", 4, 2, 10.0, -2.0, 4.0, 2.0)
        np.testing.assert_allclose(ez_x, [10.5, 11.5, 12.5, 13.5])
        np.testing.assert_allclose(ez_y, [-1.5, -0.5])

    def test_cylindrical_slot_coordinates_use_r_z_staggering(self):
        ephi_r, ephi_z = yee_component_coordinates(
            "ez", "cylindrical", 2, 2, 0.0, -1.0, 2.0, 4.0)
        np.testing.assert_allclose(ephi_r, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(ephi_z, [0.0, 2.0])

        bz_r, bz_z = yee_component_coordinates(
            "by", "cylindrical", 2, 2, 0.0, -1.0, 2.0, 4.0)
        np.testing.assert_allclose(bz_r, [0.5, 1.5])
        np.testing.assert_allclose(bz_z, [0.0, 2.0])

    def test_periodic_coordinates_omit_high_but_wall_coordinates_retain_it(self):
        wall_x, wall_y = yee_component_coordinates(
            "bz", "cartesian", 4, 2, 1.0, 3.0, 4.0, 2.0)
        periodic_x, periodic_y = yee_component_coordinates(
            "bz", "cartesian", 4, 2, 1.0, 3.0, 4.0, 2.0,
            periodic_x=True, periodic_y=True)
        np.testing.assert_allclose(wall_x, [1, 2, 3, 4, 5])
        np.testing.assert_allclose(wall_y, [3, 4, 5])
        np.testing.assert_allclose(periodic_x, [1, 2, 3, 4])
        np.testing.assert_allclose(periodic_y, [3, 4])

    def test_view_shape_must_match_component_or_legacy_cell_extent(self):
        with self.assertRaises(ValueError):
            yee_component_coordinates(
                "bz", "cartesian", 4, 3, 0.0, 0.0, 1.0, 1.0,
                view_shape=(2, 2))


class FieldPeriodicityTests(unittest.TestCase):

    class _Archive(dict):
        @property
        def files(self):
            return list(self)

    def test_requires_both_sides_of_an_axis(self):
        archive = self._Archive(boundary_field=np.array(
            ["periodic", "periodic", "pec", "pec"]))
        self.assertEqual(field_periodicity(archive), (True, False))

        one_sided = self._Archive(boundary_field=np.array(
            ["axis", "pec", "periodic", "pec"]))
        self.assertEqual(field_periodicity(one_sided), (False, False))

    def test_legacy_archive_conservatively_retains_high_faces(self):
        self.assertEqual(field_periodicity(self._Archive()), (False, False))

    def test_malformed_boundary_metadata_is_rejected(self):
        archive = self._Archive(boundary_field=np.array(["pec", "pec"]))
        with self.assertRaises(ValueError):
            field_periodicity(archive)


class CoordinateFrameLabelTests(unittest.TestCase):

    class _Archive(dict):
        @property
        def files(self):
            return list(self)

    def test_legacy_archive_defaults_to_xy_labels(self):
        plane = _archive_plane(self._Archive())
        axes = _geometry_labels("cartesian", plane)
        self.assertEqual(plane, "xy")
        self.assertEqual(axes[0:2], ("x", "y"))
        self.assertEqual(axes[2]["ey"], "Ey")
        self.assertEqual(axes[2]["bz"], "Bz")

    def test_cartesian_xz_uses_x_z_minus_y_frame(self):
        archive = self._Archive(plane=np.array(["xz"]))
        plane = _archive_plane(archive)
        axis_x, axis_y, components = _geometry_labels("cartesian", plane)
        self.assertEqual((axis_x, axis_y), ("x", "z"))
        self.assertEqual(
            components,
            {"ex": "Ex", "ey": "Ez", "ez": "-Ey",
             "bx": "Bx", "by": "Bz", "bz": "-By"})

    def test_cylindrical_labels_remain_physical_r_z_phi_in_xz_plane(self):
        axis_x, axis_y, components = _geometry_labels("cylindrical", "xz")
        self.assertEqual((axis_x, axis_y), ("r", "z"))
        self.assertEqual(components["ey"], "Ez")
        self.assertEqual(components["bz"], "Bphi")

    def test_invalid_or_malformed_plane_metadata_is_rejected(self):
        for value in ("yz", "", "cartesian"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "plane must be"):
                    _archive_plane(self._Archive(plane=np.array([value])))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _archive_plane(self._Archive(plane=np.array(["xy", "xz"])))


class PlotCoordinateFrameTests(unittest.TestCase):

    class _Archive(dict):
        @property
        def files(self):
            return list(self)

    class _Axes:
        def __init__(self):
            self.title = None
            self.xlabel = None
            self.ylabel = None

        def pcolormesh(self, *_args, **_kwargs):
            return object()

        def set_aspect(self, _aspect):
            pass

        def set_title(self, title):
            self.title = title

        def set_xlabel(self, label):
            self.xlabel = label

        def set_ylabel(self, label):
            self.ylabel = label

    class _Figure:
        def colorbar(self, _image, ax):
            pass

        def tight_layout(self):
            pass

        def savefig(self, _path, dpi):
            pass

    class _Pyplot:
        def __init__(self):
            self.axes = []

        def subplots(self, **_kwargs):
            axes = PlotCoordinateFrameTests._Axes()
            self.axes.append(axes)
            return PlotCoordinateFrameTests._Figure(), axes

        def close(self, _figure):
            pass

    def test_xz_archive_labels_observable_plot_in_physical_frame(self):
        nx, ny, nghost = 2, 2, 1
        archive = self._Archive(
            nx=np.array([nx]), ny=np.array([ny]), nghost=np.array([nghost]),
            geometry=np.array(["cartesian"]), plane=np.array(["xz"]),
            unit_system=np.array(["SI"]),
            origin_x=np.array([1.0]), origin_y=np.array([-2.0]),
            lx=np.array([4.0]), ly=np.array([6.0]),
            external_bz=np.zeros((nx + 2 * nghost) * (ny + 2 * nghost)),
        )
        pyplot = self._Pyplot()

        _plot_archive(archive, Path("out.npz"), Path("."), pyplot)

        self.assertEqual(len(pyplot.axes), 1)
        self.assertEqual(pyplot.axes[0].xlabel, "x (m)")
        self.assertEqual(pyplot.axes[0].ylabel, "z (m)")
        self.assertEqual(pyplot.axes[0].title, "external -By (T)")


class SpeciesNamesTests(unittest.TestCase):

    def test_recovers_sorted_names_ignoring_decoys(self):
        data = types.SimpleNamespace(files=[
            "species_e_x", "species_e_y", "species_e_vx",
            "species_H+_x", "species_H+_y",
            "field_ez", "nx", "ny", "snapshot_field_ez",
        ])
        self.assertEqual(species_names(data), ["H+", "e"])

    def test_empty_when_no_species_keys(self):
        data = types.SimpleNamespace(files=["field_ez", "nx", "ny"])
        self.assertEqual(species_names(data), [])


class FieldNamesTests(unittest.TestCase):

    def test_finds_every_final_component_and_ignores_snapshots(self):
        data = types.SimpleNamespace(files=[
            "field_ez", "field_bx", "field_by", "field_ex", "field_ey",
            "field_bz", "snapshot_field_ez", "external_bz", "nx",
        ])
        self.assertEqual(
            field_names(data),
            ["field_bx", "field_by", "field_bz",
             "field_ex", "field_ey", "field_ez"])


if __name__ == "__main__":
    unittest.main()
