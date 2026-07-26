"""File-backed rectilinear magnetic-field evaluator tests."""

from __future__ import annotations

import io
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import zipfile

import numpy as np
import yaml

from quasar import Vec3, _core
from quasar import _field_grid
from quasar._field_grid import load_file_grid_npz
from quasar.coil import ConductorSystem, PointCloud
from quasar.coil import io as coil_io
from quasar.coil import cli as coil_cli
from quasar.pic import io as pic_io


def _linear_field(x, y, z):
    return np.stack((1.0 + 2.0*x + 3.0*y - z,
                     -2.0 - 4.0*x + 0.5*y + 5.0*z,
                     0.25 + x - 2.0*y - 2.5*z), axis=-1)


def _write_map(path: Path) -> None:
    origin = np.asarray([10.0, -4.0, 2.0])
    spacing = np.asarray([0.5, 2.0, 4.0])
    x = origin[0] + spacing[0] * np.arange(2)
    y = origin[1] + spacing[1] * np.arange(2)
    z = origin[2] + spacing[2] * np.arange(2)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    np.savez(path, B_xyz_grid=_linear_field(xx, yy, zz),
             grid_origin=origin, grid_spacing=spacing,
             dims=np.asarray([2, 2, 2], dtype=np.int64))


class FileGridLoaderTests(unittest.TestCase):

    def test_npz_loads_x_fastest_registry_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.npz"
            _write_map(path)
            params = load_file_grid_npz(tmp, "field.npz", label="field.path")
            self.assertEqual(params["dims"], [2.0, 2.0, 2.0])
            self.assertEqual(params["origin"], [10.0, -4.0, 2.0])
            self.assertEqual(params["spacing"], [0.5, 2.0, 4.0])
            self.assertEqual(len(params["values"]), 24)

    def test_rejects_escape_missing_metadata_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with self.assertRaises(ValueError):
                load_file_grid_npz(base, "../escape.npz", label="field.path")
            np.savez(base / "bad.npz", B_xyz_grid=np.zeros((1, 1, 1, 3)))
            with self.assertRaises(ValueError):
                load_file_grid_npz(base, "bad.npz", label="field.path")
            np.savez(base / "nan.npz",
                     B_xyz_grid=np.full((1, 1, 1, 3), np.nan),
                     grid_origin=np.zeros(3), grid_spacing=np.ones(3))
            with self.assertRaises(ValueError):
                load_file_grid_npz(base, "nan.npz", label="field.path")

    def test_accepts_zero_spacing_only_on_singleton_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            np.savez(base / "singleton.npz",
                     B_xyz_grid=np.zeros((2, 1, 2, 3)),
                     grid_origin=np.zeros(3),
                     grid_spacing=np.asarray([0.5, 0.0, 0.25]))
            params = load_file_grid_npz(
                base, "singleton.npz", label="field.path")
            self.assertEqual(params["dims"], [2.0, 1.0, 2.0])
            self.assertEqual(params["spacing"], [0.5, 1.0, 0.25])

            np.savez(base / "invalid.npz",
                     B_xyz_grid=np.zeros((2, 2, 2, 3)),
                     grid_origin=np.zeros(3),
                     grid_spacing=np.asarray([0.5, 0.0, 0.25]))
            with self.assertRaises(ValueError):
                load_file_grid_npz(base, "invalid.npz", label="field.path")

    def test_rejects_collapsed_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            np.savez(base / "collapsed.npz",
                     B_xyz_grid=np.zeros((1, 1, 2, 3)),
                     grid_origin=np.asarray([1.0e308, 0.0, 0.0]),
                     grid_spacing=np.ones(3))
            with self.assertRaises(ValueError):
                load_file_grid_npz(base, "collapsed.npz", label="field.path")

    def test_rejects_ambiguous_aliases_representations_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            common = {
                "B_xyz_grid": np.zeros((1, 1, 1, 3)),
                "grid_origin": np.zeros(3),
                "grid_spacing": np.ones(3),
            }
            variants = {
                "origin_alias.npz": {**common, "origin": np.zeros(3)},
                "spacing_alias.npz": {**common, "spacing": np.ones(3)},
                "dual_field.npz": {
                    **common, "B_xyz": np.zeros((1, 3)),
                    "dims": np.ones(3, dtype=np.int64)},
                "unknown.npz": {**common, "B_magnitude": np.zeros(1)},
            }
            for name, payload in variants.items():
                with self.subTest(name=name):
                    np.savez(base / name, **payload)
                    with self.assertRaises(ValueError):
                        load_file_grid_npz(base, name, label="field.path")

    def test_rejects_duplicate_members_complex_values_and_bad_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            common = {
                "B_xyz_grid": np.zeros((1, 1, 1, 3)),
                "grid_origin": np.zeros(3),
                "grid_spacing": np.ones(3),
            }

            duplicate = base / "duplicate.npz"
            np.savez(duplicate, **common)
            duplicate_payload = io.BytesIO()
            np.save(duplicate_payload, np.ones(3), allow_pickle=False)
            with zipfile.ZipFile(duplicate, mode="a") as archive:
                archive.writestr("grid_origin.npy", duplicate_payload.getvalue())
            with self.assertRaisesRegex(ValueError, "duplicate archive key"):
                load_file_grid_npz(base, duplicate.name, label="field.path")

            np.savez(base / "complex.npz", **{
                **common,
                "B_xyz_grid": np.full((1, 1, 1, 3), 1.0 + 2.0j),
            })
            with self.assertRaisesRegex(ValueError, "real numeric"):
                load_file_grid_npz(base, "complex.npz", label="field.path")

            np.savez(base / "kind.npz", **{
                **common, "observation_kind": np.asarray("line")})
            with self.assertRaisesRegex(ValueError, "observation_kind"):
                load_file_grid_npz(base, "kind.npz", label="field.path")

    def test_archive_and_point_limits_apply_before_parameter_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = base / "field.npz"
            np.savez(path,
                     B_xyz_grid=np.zeros((1, 1, 2, 3)),
                     grid_origin=np.zeros(3), grid_spacing=np.ones(3))
            with mock.patch.object(_field_grid, "MAX_FIELD_GRID_POINTS", 1):
                with self.assertRaisesRegex(ValueError, "point.*limit|points; limit"):
                    load_file_grid_npz(base, path.name, label="field.path")
            with mock.patch.object(
                    _field_grid, "MAX_FIELD_GRID_ARCHIVE_BYTES",
                    path.stat().st_size - 1):
                with self.assertRaisesRegex(ValueError, "archive is.*limit"):
                    load_file_grid_npz(base, path.name, label="field.path")

    def test_inconsistent_npy_shape_is_rejected_before_numpy_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = base / "forged.npz"

            def encoded(array) -> bytes:
                payload = io.BytesIO()
                np.save(payload, array, allow_pickle=False)
                return payload.getvalue()

            forged = io.BytesIO()
            np.lib.format.write_array_header_1_0(forged, {
                "descr": np.dtype(np.float64).str,
                "fortran_order": False,
                "shape": (1, 1, _field_grid.MAX_FIELD_GRID_POINTS + 1, 3),
            })
            with zipfile.ZipFile(path, mode="w") as archive:
                archive.writestr("B_xyz_grid.npy", forged.getvalue())
                archive.writestr("grid_origin.npy", encoded(np.zeros(3)))
                archive.writestr("grid_spacing.npy", encoded(np.ones(3)))

            with mock.patch.object(
                    _field_grid.np, "load",
                    side_effect=AssertionError("np.load must not be reached")):
                with self.assertRaisesRegex(ValueError, "inconsistent array size"):
                    load_file_grid_npz(base, path.name, label="field.path")

    def test_extreme_affine_endpoint_cancellation_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            largest = np.finfo(np.float64).max
            np.savez(base / "extreme.npz",
                     B_xyz_grid=np.zeros((1, 1, 3, 3)),
                     grid_origin=np.asarray([-largest, 0.0, 0.0]),
                     grid_spacing=np.asarray([largest, 1.0, 1.0]))
            params = load_file_grid_npz(
                base, "extreme.npz", label="field.path")
            self.assertEqual(params["origin"][0], -largest)
            self.assertEqual(params["spacing"][0], largest)

    def test_nonfinite_affine_endpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            largest = np.finfo(np.float64).max
            np.savez(base / "overflow.npz",
                     B_xyz_grid=np.zeros((1, 1, 2, 3)),
                     grid_origin=np.asarray([largest, 0.0, 0.0]),
                     grid_spacing=np.asarray([largest, 1.0, 1.0]))
            with self.assertRaisesRegex(ValueError, "not finite"):
                load_file_grid_npz(
                    base, "overflow.npz", label="field.path")


class FileGridBindingTests(unittest.TestCase):

    def test_registry_configure_interpolates_and_differentiates_linear_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.npz"
            _write_map(path)
            params = load_file_grid_npz(tmp, "field.npz", label="field.path")
            evaluator = _core.magnetostatics.create_field_evaluator("file_grid")
            self.assertFalse(evaluator.provides_grad_B)
            evaluator.configure(params)
            self.assertEqual(type(evaluator).__name__, "FileGridEvaluator")
            self.assertTrue(evaluator.provides_grad_B)

            points = PointCloud()
            points.add(Vec3(10.125, -2.5, 5.0))
            source = ConductorSystem()
            field = evaluator.evaluate_B(source, points)
            expected = _linear_field(10.125, -2.5, 5.0)
            np.testing.assert_allclose(field[0], expected, rtol=0, atol=1e-12)

            gradient = evaluator.evaluate_grad_B(source, points)
            np.testing.assert_allclose(
                gradient[0],
                [[2.0, 3.0, -1.0], [-4.0, 0.5, 5.0], [1.0, -2.0, -2.5]],
                rtol=0, atol=1e-12)


class FileGridDeckTests(unittest.TestCase):

    def test_mapping_parse_requires_an_explicit_file_base(self):
        coil_data = {
            "units": "SI",
            "evaluator": {"type": "file_grid", "path": "field.npz"},
            "observation": {"type": "points", "points_xyz_m": [[0, 0, 0]]},
            "output": {"format": "npz", "path": "out.npz"},
        }
        with self.assertRaises(ValueError):
            coil_io.parse(coil_data)

        pic_data = {
            "units": "normalized",
            "domain": {"nx": 4, "ny": 4, "lx_m": 1.0, "ly_m": 1.0},
            "external_field": {
                "evaluator": {"type": "file_grid", "path": "field.npz"}},
            "time": {"dt_s": "auto", "steps": 1},
        }
        with self.assertRaises(ValueError):
            pic_io.parse(pic_data)

    def test_coil_deck_loads_map_without_dummy_conductors(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_map(base / "field.npz")
            deck_path = base / "coil.yaml"
            deck_path.write_text(yaml.safe_dump({
                "units": "SI",
                "evaluator": {"type": "file_grid", "path": "field.npz"},
                "observation": {"type": "points", "points_xyz_m": [[10, -4, 2]]},
                "output": {"format": "npz", "path": "out.npz",
                           "fields": ["B_xyz"]},
            }))
            deck = coil_io.load(deck_path)
            self.assertTrue(deck.conductors.empty())
            self.assertEqual(deck.evaluator_type, "file_grid")
            self.assertEqual(deck.evaluator_params["dims"], [2.0, 2.0, 2.0])

    def test_pic_deck_loads_map_into_registry_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_map(base / "field.npz")
            deck_path = base / "pic.yaml"
            deck_path.write_text(yaml.safe_dump({
                "units": "normalized",
                "domain": {"nx": 4, "ny": 4, "lx_m": 1.0, "ly_m": 1.0},
                "external_field": {
                    "evaluator": {"type": "file_grid", "path": "field.npz"}},
                "time": {"dt_s": "auto", "steps": 1},
            }))
            deck = pic_io.load(deck_path)
            self.assertIsNotNone(deck.external_field)
            self.assertEqual(deck.external_field.evaluator_type, "file_grid")
            self.assertEqual(
                deck.external_field.evaluator_params()["dims"], [2.0, 2.0, 2.0])

    def test_singleton_coil_archive_round_trips_as_a_field_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            deck = coil_io.parse({
                "units": "SI",
                "evaluator": {"type": "uniform", "B_T": [0, 0, 1]},
                "observation": {
                    "type": "grid",
                    "bounds_m": [[0.0, 1.0], [2.0, 2.0], [0.0, 1.0]],
                    "resolution": [2, 1, 2],
                },
                "output": {"format": "npz", "path": "field.npz",
                           "fields": ["B_xyz_grid"]},
            })
            field = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (4, 1))
            payload, _ = coil_cli._build_payload(deck, field)
            np.savez(base / "field.npz", **payload)

            params = load_file_grid_npz(
                base, "field.npz", label="evaluator.path")
            self.assertEqual(params["dims"], [2.0, 1.0, 2.0])
            self.assertEqual(params["spacing"][1], 1.0)
            evaluator = _core.magnetostatics.create_field_evaluator("file_grid")
            evaluator.configure(params)
            self.assertFalse(evaluator.provides_grad_B)
            points = PointCloud()
            points.add(Vec3(0.25, 2.0, 0.75))
            result = evaluator.evaluate_B(ConductorSystem(), points)
            np.testing.assert_array_equal(result[0], [0.0, 0.0, 1.0])
            with self.assertRaisesRegex(RuntimeError, "full magnetic-field gradient"):
                evaluator.evaluate_grad_B(ConductorSystem(), points)


if __name__ == "__main__":
    unittest.main()
