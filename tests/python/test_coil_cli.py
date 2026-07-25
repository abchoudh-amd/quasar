"""Smoke test for the ``quasar coil`` YAML-driven CLI.

The test materializes a one-loop YAML deck in a temporary directory,
invokes ``python -m quasar.coil.cli run <input>`` as a subprocess, and
verifies the produced ``.npz`` archive contains the expected fields with
the right shape. The CLI test is skipped when no HIP runtime is visible
since the evaluation step launches a HIP kernel.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from quasar.coil import io as coil_io


def has_hip_runtime() -> bool:
    return os.environ.get("QUASAR_HAS_HIP_RUNTIME", "0") == "1"


class CoilIoDeckParseTest(unittest.TestCase):

    def _write(self, body: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False)
        tmp.write(textwrap.dedent(body))
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def _expect_value_error(self, body: str) -> None:
        path = self._write(body)
        try:
            with self.assertRaises(ValueError):
                coil_io.load(path)
        finally:
            path.unlink()

    def test_parses_grid_observation_and_circular_loop(self):
        path = self._write("""
            units: SI
            conductors:
              - name: loop_A
                current_A: 1.0
                geometry:
                  type: circular_loop
                  radius_m: 0.05
                  center_xyz: [0, 0, 0]
                  axis_xyz:   [0, 0, 1]
                  n_segments: 64
            observation:
              type: grid
              bounds_m: [[-0.1, 0.1], [-0.1, 0.1], [-0.05, 0.05]]
              resolution: [4, 4, 2]
            output:
              format: npz
              path: out.npz
              fields: [B_xyz, B_magnitude]
            """)
        try:
            deck = coil_io.load(path)
        finally:
            path.unlink()

        self.assertEqual(deck.units, "SI")
        self.assertEqual(len(deck.conductors), 1)
        self.assertEqual(deck.observation.kind, "grid")
        self.assertEqual(deck.observation.dims, [4, 4, 2])
        self.assertEqual(len(deck.observation.points), 32)
        self.assertEqual(deck.output.format, "npz")
        self.assertEqual(set(deck.output.fields), {"B_xyz", "B_magnitude"})

    def test_rejects_unknown_units(self):
        path = self._write("""
            units: CGS
            conductors: []
            observation: {type: points, points_xyz_m: [[0,0,0]]}
            output: {format: npz, path: x.npz}
            """)
        try:
            with self.assertRaises(ValueError):
                coil_io.load(path)
        finally:
            path.unlink()

    def test_rejects_boolean_physical_values_and_string_vectors(self):
        base = """
            units: SI
            conductors:
              - name: loop
                current_A: 1.0
                geometry:
                  type: circular_loop
                  radius_m: 0.05
                  center_xyz: [0, 0, 0]
                  axis_xyz: [0, 0, 1]
                  n_segments: 16
            observation: {type: points, points_xyz_m: [[0, 0, 0.1]]}
            output: {format: npz, path: out.npz}
            """
        variants = (
            base.replace("current_A: 1.0", "current_A: true"),
            base.replace("center_xyz: [0, 0, 0]",
                         "center_xyz: [false, 0, 0]"),
            base.replace("axis_xyz: [0, 0, 1]", 'axis_xyz: "001"'),
        )
        for body in variants:
            with self.subTest(body=body):
                self._expect_value_error(body)

    def test_rejects_unknown_geometry_type(self):
        path = self._write("""
            units: SI
            conductors:
              - name: bad
                current_A: 1.0
                geometry: {type: triangle_loop, radius_m: 1.0}
            observation: {type: points, points_xyz_m: [[0,0,0]]}
            output: {format: npz, path: x.npz}
            """)
        try:
            with self.assertRaises(ValueError):
                coil_io.load(path)
        finally:
            path.unlink()

    def test_parses_plane_observation(self):
        path = self._write("""
            units: SI
            conductors:
              - name: loop
                current_A: 1.0
                geometry:
                  type: circular_loop
                  radius_m: 0.05
                  center_xyz: [0, 0, 0]
                  axis_xyz:   [0, 0, 1]
                  n_segments: 16
            observation:
              type: plane
              origin_xyz:  [0, 0, 0]
              u_axis_xyz:  [1, 0, 0]
              v_axis_xyz:  [0, 1, 0]
              u_extent_m:  0.2
              v_extent_m:  0.1
              nu: 5
              nv: 3
            output: {format: npz, path: out.npz, fields: [B_xyz]}
            """)
        try:
            deck = coil_io.load(path)
        finally:
            path.unlink()
        self.assertEqual(deck.observation.kind, "plane")
        self.assertEqual(deck.observation.dims, [5, 3])
        self.assertEqual(len(deck.observation.points), 15)

    def test_plane_normalizes_extreme_finite_axes(self):
        largest = float.fromhex("0x1.fffffffffffffp+1023")
        path = self._write(f"""
            units: SI
            conductors:
              - name: loop
                current_A: 1.0
                geometry:
                  type: circular_loop
                  radius_m: 0.05
                  center_xyz: [0, 0, 0]
                  axis_xyz:   [0, 0, 1]
                  n_segments: 16
            observation:
              type: plane
              origin_xyz:  [0, 0, 0]
              u_axis_xyz:  [{largest}, {largest}, {largest}]
              v_axis_xyz:  [1, -1, 0]
              u_extent_m:  0.2
              v_extent_m:  0.1
              nu: 2
              nv: 2
            output: {{format: npz, path: out.npz, fields: [B_xyz]}}
            """)
        try:
            deck = coil_io.load(path)
        finally:
            path.unlink()
        self.assertEqual(len(deck.observation.points), 4)

    def _deck_with_geometry(self, geom_inline: str) -> str:
        # geom_inline is a YAML flow-mapping ({...}) so indentation is irrelevant.
        return (
            "units: SI\n"
            "conductors:\n"
            "  - name: c\n"
            "    current_A: 1.0\n"
            f"    geometry: {geom_inline}\n"
            "observation: {type: points, points_xyz_m: [[0,0,0.2]]}\n"
            "output: {format: npz, path: out.npz, fields: [B_xyz]}\n"
        )

    def test_parses_each_geometry_type(self):
        geometries = {
            "helix": ("{type: helix, center_xyz: [0,0,0], axis_xyz: [0,0,1], "
                      "radius_m: 0.05, pitch_m: 0.01, n_turns: 2, "
                      "n_segments_per_turn: 16}"),
            "solenoid": ("{type: solenoid, center_xyz: [0,0,0], axis_xyz: [0,0,1], "
                         "radius_m: 0.05, length_m: 0.1, n_turns: 4, "
                         "n_segments_per_turn: 16}"),
            "racetrack": ("{type: racetrack, center_xyz: [0,0,0], axis_xyz: [0,0,1], "
                          "straight_length_m: 0.1, arc_radius_m: 0.02, "
                          "n_arc_segments: 16}"),
            "polygon": ("{type: polygon, center_xyz: [0,0,0], axis_xyz: [0,0,1], "
                        "circumradius_m: 0.05, n_sides: 6}"),
            "polyline": "{type: polyline, points_xyz_m: [[0,0,0],[0.1,0,0],[0.1,0.1,0]]}",
        }
        for name, geom in geometries.items():
            with self.subTest(geometry=name):
                path = self._write(self._deck_with_geometry(geom))
                try:
                    deck = coil_io.load(path)
                finally:
                    path.unlink()
                self.assertEqual(len(deck.conductors), 1)

    def test_rejects_missing_required_key_per_geometry(self):
        # Each geometry should raise when a required field is absent (spot-check
        # one omitted key per type).
        missing = {
            "helix": "{type: helix, axis_xyz: [0,0,1]}",
            "solenoid": "{type: solenoid, axis_xyz: [0,0,1]}",
            "racetrack": "{type: racetrack, axis_xyz: [0,0,1]}",
            "polygon": "{type: polygon, axis_xyz: [0,0,1]}",
            "polyline": "{type: polyline}",
        }
        for name, geom in missing.items():
            with self.subTest(geometry=name):
                path = self._write(self._deck_with_geometry(geom))
                try:
                    with self.assertRaises((ValueError, KeyError, TypeError)):
                        coil_io.load(path)
                finally:
                    path.unlink()


class CoilIoErrorPathTest(unittest.TestCase):

    def _write(self, body: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False)
        tmp.write(textwrap.dedent(body))
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def _expect_value_error(self, body: str) -> None:
        path = self._write(body)
        try:
            with self.assertRaises(ValueError):
                coil_io.load(path)
        finally:
            path.unlink()

    def test_rejects_non_mapping_top_level(self):
        self._expect_value_error("- just\n- a\n- list\n")

    def test_rejects_duplicate_yaml_keys(self):
        path = self._write("units: SI\nunits: SI\n")
        try:
            with self.assertRaisesRegex(
                    ValueError, r"duplicate YAML key 'units'.*line 2"):
                coil_io.load(path)
        finally:
            path.unlink()

    def test_numeric_overflow_is_normalized_to_value_error(self):
        from quasar._deck import as_finite, triple

        enormous = 10 ** 10000
        with self.assertRaises(ValueError):
            as_finite(enormous, "value")
        with self.assertRaises(ValueError):
            triple([enormous, 0, 0])

        # Geometry scalars pass through the same schema-level normalization;
        # Python's raw float(enormous) raises OverflowError instead of the
        # loader's documented ValueError.
        # Build the YAML integer text directly. Formatting ``enormous`` itself is
        # capped at 4300 digits by Python 3.11+, which would make the fixture fail
        # before the loader receives it.
        enormous_yaml = "1" + "0" * 10000
        self._expect_value_error(f"""
            units: SI
            conductors:
              - name: loop
                current_A: 1.0
                geometry: {{type: circular_loop, radius_m: {enormous_yaml},
                           center_xyz: [0,0,0], axis_xyz: [0,0,1],
                           n_segments: 16}}
            observation: {{type: points, points_xyz_m: [[0,0,1]]}}
            output: {{format: npz, path: out.npz}}
            """)

    def test_rejects_empty_conductors(self):
        self._expect_value_error("""
            units: SI
            conductors: []
            observation: {type: points, points_xyz_m: [[0,0,0]]}
            output: {format: npz, path: out.npz}
            """)

    def test_rejects_missing_required_key(self):
        # conductor missing 'geometry'
        self._expect_value_error("""
            units: SI
            conductors:
              - {name: loop, current_A: 1.0}
            observation: {type: points, points_xyz_m: [[0,0,0]]}
            output: {format: npz, path: out.npz}
            """)

    def test_rejects_non_npz_output_format(self):
        self._expect_value_error("""
            units: SI
            conductors:
              - name: loop
                current_A: 1.0
                geometry: {type: circular_loop, radius_m: 0.1,
                           center_xyz: [0,0,0], axis_xyz: [0,0,1], n_segments: 16}
            observation: {type: points, points_xyz_m: [[0,0,0]]}
            output: {format: vtk, path: out.vtk}
            """)

    def test_rejects_oversized_observation_grid(self):
        # resolution product exceeds MAX_OBSERVATION_POINTS
        self._expect_value_error("""
            units: SI
            conductors:
              - name: loop
                current_A: 1.0
                geometry: {type: circular_loop, radius_m: 0.1,
                           center_xyz: [0,0,0], axis_xyz: [0,0,1], n_segments: 16}
            observation:
              type: grid
              bounds_m: [[-1, 1], [-1, 1], [-1, 1]]
              resolution: [100000, 100000, 100000]
            output: {format: npz, path: out.npz}
            """)

    def test_extreme_symmetric_grid_interval_has_finite_spacing(self):
        largest = float.fromhex("0x1.fffffffffffffp+1023")
        path = self._write(f"""
            units: SI
            evaluator: {{type: uniform, B_T: [0, 0, 0]}}
            conductors: []
            observation:
              type: grid
              bounds_m: [[{-largest}, {largest}], [0, 0], [0, 0]]
              resolution: [3, 1, 1]
            output: {{format: npz, path: out.npz}}
            """)
        try:
            deck = coil_io.load(path)
        finally:
            path.unlink()
        self.assertEqual(deck.observation.detail.spacing.x, largest)
        self.assertEqual(len(deck.observation.points), 3)

    def test_rejects_oversized_observation_points(self):
        old_limit = coil_io.MAX_OBSERVATION_POINTS
        coil_io.MAX_OBSERVATION_POINTS = 1
        try:
            self._expect_value_error("""
                units: SI
                conductors:
                  - name: loop
                    current_A: 1.0
                    geometry: {type: circular_loop, radius_m: 0.1,
                               center_xyz: [0,0,0], axis_xyz: [0,0,1], n_segments: 16}
                observation: {type: points, points_xyz_m: [[0,0,0], [0,0,0.1]]}
                output: {format: npz, path: out.npz}
                """)
        finally:
            coil_io.MAX_OBSERVATION_POINTS = old_limit

    def _evaluator_deck(self, ev_type: str) -> str:
        return f"""
            units: SI
            conductors:
              - name: loop
                current_A: 1.0
                geometry: {{type: circular_loop, radius_m: 0.1,
                           center_xyz: [0,0,0], axis_xyz: [0,0,1], n_segments: 16}}
            observation: {{type: points, points_xyz_m: [[0,0,0]]}}
            output: {{format: npz, path: out.npz}}
            evaluator: {{type: {ev_type}}}
            """

    def test_file_grid_requires_path(self):
        self._expect_value_error(self._evaluator_deck("file_grid"))

    def test_rejects_unknown_evaluator(self):
        self._expect_value_error(self._evaluator_deck("does_not_exist"))

    def test_registered_plugin_uses_generic_flat_parameter_map(self):
        from quasar import _core

        names = _core.magnetostatics.field_evaluator_names()
        with mock.patch.object(
                _core.magnetostatics, "field_evaluator_names",
                return_value=[*names, "test_plugin"]):
            body = self._evaluator_deck("test_plugin").replace(
                "evaluator: {type: test_plugin}",
                "evaluator: {type: test_plugin, params: {gain: 2, axis: [1, 0, -1]}}")
            path = self._write(body)
            try:
                deck = coil_io.load(path)
            finally:
                path.unlink()
        self.assertEqual(deck.evaluator_type, "test_plugin")
        self.assertEqual(deck.evaluator_params,
                         {"gain": [2.0], "axis": [1.0, 0.0, -1.0]})

    def test_plugin_parameter_map_rejects_nested_boolean_and_nonfinite_values(self):
        from quasar import _core

        names = _core.magnetostatics.field_evaluator_names()
        with mock.patch.object(
                _core.magnetostatics, "field_evaluator_names",
                return_value=[*names, "test_plugin"]):
            for params in ("{bad: [[1, 2]]}", "{bad: true}", "{bad: [.inf]}"):
                with self.subTest(params=params):
                    body = self._evaluator_deck("test_plugin").replace(
                        "evaluator: {type: test_plugin}",
                        f"evaluator: {{type: test_plugin, params: {params}}}")
                    self._expect_value_error(body)

    def test_rejects_missing_dipole_parameters(self):
        self._expect_value_error(self._evaluator_deck("dipole"))

    def test_rejects_ambiguous_evaluator_aliases(self):
        cases = (
            ("uniform", "B_T: [1,0,0], b_tesla: [1,0,0]"),
            ("uniform", "E_V_per_m: [0,1,0], E: [0,1,0]"),
            ("dipole", "moment_Am2: [0,0,1], moment: [0,0,1]"),
            ("dipole", "moment_Am2: [0,0,1], origin_xyz_m: [0,0,0], origin: [0,0,0]"),
            ("gradient", "grad_T_per_m: [[1,0,0],[0,-1,0],[0,0,0]], gradient: [[1,0,0],[0,-1,0],[0,0,0]]"),
            ("gradient", "grad_T_per_m: [[1,0,0],[0,-1,0],[0,0,0]], B0_T: [0,0,0], b0: [0,0,0]"),
            ("gradient", "grad_T_per_m: [[1,0,0],[0,-1,0],[0,0,0]], origin_xyz_m: [0,0,0], origin: [0,0,0]"),
            ("file_grid", "path: field.npz, file: field.npz"),
        )
        for ev_type, params in cases:
            with self.subTest(evaluator=ev_type, params=params):
                body = self._evaluator_deck(ev_type).replace(
                    f"evaluator: {{type: {ev_type}}}",
                    f"evaluator: {{type: {ev_type}, {params}}}")
                self._expect_value_error(body)

    def test_rejects_non_trace_free_gradient(self):
        body = self._evaluator_deck("gradient").replace(
            "evaluator: {type: gradient}",
            ("evaluator: {type: gradient, "
             "grad_T_per_m: [[1,0,0],[0,2,0],[0,0,3]]}")
        )
        self._expect_value_error(body)

    def test_rejects_unknown_output_field(self):
        body = self._evaluator_deck("biot_savart").replace(
            "output: {format: npz, path: out.npz}",
            "output: {format: npz, path: out.npz, fields: [B_xyz, typo]}"
        )
        self._expect_value_error(body)

    def test_rejects_nonfinite_point_observation(self):
        body = self._evaluator_deck("biot_savart").replace(
            "points_xyz_m: [[0,0,0]]", "points_xyz_m: [[.nan,0,0]]"
        )
        self._expect_value_error(body)

    def test_rejects_fractional_counts_and_degenerate_observations(self):
        body = self._evaluator_deck("biot_savart").replace(
            "n_segments: 16", "n_segments: 16.5")
        self._expect_value_error(body)

        body = self._evaluator_deck("biot_savart").replace(
            "observation: {type: points, points_xyz_m: [[0,0,0]]}",
            "observation: {type: points, points_xyz_m: []}")
        self._expect_value_error(body)

    def test_rejects_unknown_schema_keys_and_empty_output_path(self):
        base = self._evaluator_deck("biot_savart")
        variants = (
            base.replace("units: SI", "units: SI\n            typo: 1"),
            base.replace(
                "current_A: 1.0", "current_A: 1.0\n                typo: 1"),
            base.replace("n_segments: 16", "n_segments: 16, typo: 1"),
            base.replace(
                "points_xyz_m: [[0,0,0]]",
                "points_xyz_m: [[0,0,0]], typo: 1"),
            base.replace(
                "output: {format: npz, path: out.npz}",
                "output: {format: npz, path: out.npz, typo: 1}"),
            base.replace("path: out.npz", 'path: ""'),
        )
        for body in variants:
            with self.subTest(body=body):
                self._expect_value_error(body)

    def test_parses_uniform_configuration(self):
        path = self._write(self._evaluator_deck("uniform").replace(
            "evaluator: {type: uniform}",
            "evaluator: {type: uniform, B_T: [1.0, -2.0, 3.0]}"
        ))
        try:
            deck = coil_io.load(path)
        finally:
            path.unlink()
        self.assertEqual(deck.evaluator_params["b0"], [1.0, -2.0, 3.0])


class BuildPayloadTest(unittest.TestCase):
    """CPU-only tests for the output-field assembly (no kernel / GPU needed)."""

    def _write(self, body: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False)
        tmp.write(textwrap.dedent(body))
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def _grid_deck(self, fields: str) -> "coil_io.CoilDeck":
        path = self._write(f"""
            units: SI
            conductors:
              - name: loop
                current_A: 1.0
                geometry: {{type: circular_loop, radius_m: 0.05,
                           center_xyz: [0,0,0], axis_xyz: [0,0,1], n_segments: 16}}
            observation:
              type: grid
              bounds_m: [[-0.1, 0.1], [-0.1, 0.1], [-0.05, 0.05]]
              resolution: [4, 3, 2]
            output: {{format: npz, path: out.npz, fields: {fields}}}
            """)
        try:
            return coil_io.load(path)
        finally:
            path.unlink()

    def _line_deck(self, fields: str) -> "coil_io.CoilDeck":
        path = self._write(f"""
            units: SI
            conductors:
              - name: loop
                current_A: 1.0
                geometry: {{type: circular_loop, radius_m: 0.05,
                           center_xyz: [0,0,0], axis_xyz: [0,0,1], n_segments: 16}}
            observation:
              type: line
              start_xyz: [0, 0, 0]
              end_xyz:   [0, 0, 0.2]
              n_points: 6
            output: {{format: npz, path: out.npz, fields: {fields}}}
            """)
        try:
            return coil_io.load(path)
        finally:
            path.unlink()

    def test_b_xyz_grid_reshapes_to_nz_ny_nx_3(self):
        from quasar.coil.cli import _build_payload
        deck = self._grid_deck("[B_xyz_grid]")
        # resolution [4,3,2] => 24 points, x-fastest ordering as the kernel emits.
        B = np.arange(24 * 3, dtype=np.float64).reshape(24, 3)
        payload, _ = _build_payload(deck, B)
        self.assertEqual(payload["B_xyz_grid"].shape, (2, 3, 4, 3))

    def test_b_magnitude_is_scale_safe_and_payload_shapes_are_checked(self):
        from quasar.coil.cli import _build_payload
        deck = self._line_deck("[B_magnitude]")
        B = np.zeros((6, 3), dtype=np.float64)
        B[0] = [1.0e308, 1.0e308, 0.0]
        payload, _ = _build_payload(deck, B)
        self.assertTrue(np.isfinite(payload["B_magnitude"][0]))
        self.assertAlmostEqual(
            payload["B_magnitude"][0] / 1.0e308, math.sqrt(2.0))

        with self.assertRaises(ValueError):
            _build_payload(deck, np.zeros((5, 3)))
        with self.assertRaises(ValueError):
            _build_payload(deck, np.zeros((6, 2)))

        B.fill(np.finfo(np.float64).max)
        with self.assertRaisesRegex(ValueError, "magnitude.*not representable"):
            _build_payload(deck, B)

    def test_payload_rejects_nonfinite_field_arrays(self):
        from quasar.coil.cli import _build_payload

        b_deck = self._line_deck("[B_xyz]")
        for value in (np.nan, np.inf, -np.inf):
            B = np.zeros((6, 3))
            B[0, 0] = value
            with self.subTest(field="B", value=value):
                with self.assertRaises(ValueError):
                    _build_payload(b_deck, B)

        a_deck = self._line_deck("[A_xyz]")
        A = np.zeros((6, 3))
        A[0, 0] = np.nan
        with self.assertRaises(ValueError):
            _build_payload(a_deck, np.zeros((6, 3)), A)

    def test_b_xyz_grid_rejected_for_non_grid_observation(self):
        with self.assertRaises(ValueError):
            self._line_deck("[B_xyz_grid]")

    def test_vector_potential_rejected_for_analytic_evaluator(self):
        path = self._write("""
            units: SI
            evaluator: {type: uniform, B_T: [0, 0, 1]}
            conductors:
              - name: placeholder
                current_A: 0
                geometry: {type: polyline, points_xyz_m: [[0,0,0],[1,0,0]]}
            observation: {type: points, points_xyz_m: [[0,0,1]]}
            output: {format: npz, path: out.npz, fields: [A_xyz]}
            """)
        try:
            with self.assertRaises(ValueError):
                coil_io.load(path)
        finally:
            path.unlink()


class ConfineOutputPathTest(unittest.TestCase):

    def test_rejects_parent_escape(self):
        from quasar._paths import confine_output_path
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                confine_output_path(tmp, "../escape.npz")

    def test_rejects_absolute_escape(self):
        from quasar._paths import confine_output_path
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                confine_output_path(tmp, "/etc/passwd")

    def test_allows_in_tree_path(self):
        from quasar._paths import confine_output_path
        with tempfile.TemporaryDirectory() as tmp:
            out = confine_output_path(tmp, "sub/out.npz")
            self.assertTrue(str(out).startswith(str(Path(tmp).resolve())))


class CoilAnalyticEvaluatorEndToEndTest(unittest.TestCase):

    def test_uniform_evaluator_is_configured_from_deck(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            yaml_path = tmp / "deck.yaml"
            yaml_path.write_text(textwrap.dedent("""
                units: SI
                evaluator: {type: uniform, B_T: [1.0, -2.0, 3.0]}
                conductors:
                  - name: placeholder
                    current_A: 0.0
                    geometry: {type: polyline, points_xyz_m: [[0,0,0],[1,0,0]]}
                observation: {type: points, points_xyz_m: [[0,0,1]]}
                output: {format: npz, path: out.npz, fields: [B_xyz]}
                """).strip())
            res = subprocess.run(
                [sys.executable, "-m", "quasar.coil.cli", "run", str(yaml_path)],
                capture_output=True, text=True, env={**os.environ})
            self.assertEqual(res.returncode, 0,
                             msg=f"stdout={res.stdout!r} stderr={res.stderr!r}")
            archive = np.load(tmp / "out.npz", allow_pickle=False)
            np.testing.assert_array_equal(
                archive["B_xyz"], np.asarray([[1.0, -2.0, 3.0]]))

    def test_output_path_without_npz_suffix_is_used_exactly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            yaml_path = tmp / "deck.yaml"
            yaml_path.write_text(textwrap.dedent("""
                units: SI
                evaluator: {type: uniform, B_T: [1.0, -2.0, 3.0]}
                observation: {type: points, points_xyz_m: [[0,0,1]]}
                output: {format: npz, path: field-output, fields: [B_xyz]}
                """).strip())

            res = subprocess.run(
                [sys.executable, "-m", "quasar.coil.cli", "run", str(yaml_path)],
                capture_output=True, text=True, env={**os.environ})

            self.assertEqual(res.returncode, 0,
                             msg=f"stdout={res.stdout!r} stderr={res.stderr!r}")
            self.assertTrue((tmp / "field-output").is_file())
            self.assertFalse((tmp / "field-output.npz").exists())
            with np.load(tmp / "field-output", allow_pickle=False) as archive:
                np.testing.assert_array_equal(
                    archive["B_xyz"], np.asarray([[1.0, -2.0, 3.0]]))


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class CoilCliEndToEndTest(unittest.TestCase):

    def test_cli_run_writes_npz_with_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            yaml_path = tmp / "deck.yaml"
            yaml_path.write_text(textwrap.dedent("""
                units: SI
                conductors:
                  - name: loop
                    current_A: 1.0
                    geometry:
                      type: circular_loop
                      radius_m: 0.1
                      center_xyz: [0, 0, 0]
                      axis_xyz:   [0, 0, 1]
                      n_segments: 256
                observation:
                  type: line
                  start_xyz: [0, 0, 0]
                  end_xyz:   [0, 0, 0.2]
                  n_points: 5
                output:
                  format: npz
                  path: out.npz
                  fields: [B_xyz, B_magnitude]
                """).strip())

            # Run as a subprocess to validate the CLI surface end-to-end.
            res = subprocess.run(
                [sys.executable, "-m", "quasar.coil.cli", "run",
                 str(yaml_path)],
                capture_output=True, text=True,
                env={**os.environ},
            )
            self.assertEqual(res.returncode, 0,
                             msg=f"stdout={res.stdout!r} stderr={res.stderr!r}")

            archive = np.load(tmp / "out.npz", allow_pickle=False)
            self.assertIn("B_xyz", archive.files)
            self.assertIn("B_magnitude", archive.files)
            self.assertEqual(archive["B_xyz"].shape, (5, 3))
            self.assertEqual(archive["B_magnitude"].shape, (5,))

            # At the loop center (z=0), B_z should be mu0*I / (2 R) =
            # 4pi*1e-7 * 1 / (2 * 0.1) = 2*pi*1e-6 ~ 6.283185e-6 T.
            ref_center = 4 * math.pi * 1e-7 * 1.0 / (2 * 0.1)
            self.assertAlmostEqual(archive["B_xyz"][0, 2], ref_center,
                                   delta=1e-4 * ref_center)


if __name__ == "__main__":
    unittest.main()
