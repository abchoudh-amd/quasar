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

    def test_rejects_file_grid_evaluator(self):
        # file_grid is registered in C++ but not yet implemented, so the deck layer
        # must reject it (SUPPORTED_EVALUATORS excludes it) before it can reach the
        # raw C++ std::logic_error.
        self._expect_value_error(self._evaluator_deck("file_grid"))

    def test_rejects_unknown_evaluator(self):
        self._expect_value_error(self._evaluator_deck("does_not_exist"))


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
        payload = _build_payload(deck, B)
        self.assertEqual(payload["B_xyz_grid"].shape, (2, 3, 4, 3))

    def test_b_xyz_grid_rejected_for_non_grid_observation(self):
        from quasar.coil.cli import _build_payload
        deck = self._line_deck("[B_xyz_grid]")
        B = np.zeros((6, 3), dtype=np.float64)
        with self.assertRaises(ValueError):
            _build_payload(deck, B)


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
