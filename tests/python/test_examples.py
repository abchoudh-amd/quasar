"""Integration tests for the worked examples in ``examples/``.

Each test invokes ``python -m quasar.coil.cli run <input.yaml>`` for one of
the example decks, loads the produced ``out.npz``, and compares the
computed B-field to a closed-form analytical reference.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def has_hip_runtime() -> bool:
    return os.environ.get("QUASAR_HAS_HIP_RUNTIME", "0") == "1"


def _copy_example(name: str, into: Path) -> Path:
    """Copy an example directory into a sandbox so the test does not write
    artifacts back into the source tree."""
    import shutil

    src = REPO_ROOT / "examples" / name
    dst = into / name
    shutil.copytree(src, dst)
    return dst


def _run_cli(yaml_path: Path) -> None:
    res = subprocess.run(
        [sys.executable, "-m", "quasar.coil.cli", "run",
         str(yaml_path), "--quiet"],
        capture_output=True, text=True,
        env={**os.environ},
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"quasar.coil.cli failed (exit {res.returncode}):\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}")


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class SingleLoopExampleTest(unittest.TestCase):

    def test_axial_field_matches_closed_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = _copy_example("single_loop", Path(tmp))
            _run_cli(workdir / "input.yaml")

            archive = np.load(workdir / "out.npz", allow_pickle=False)
            self.assertEqual(archive["observation_kind"].item(), "line")
            B = archive["B_xyz"]
            self.assertEqual(B.shape, (5, 3))

            # Closed form for a circular loop of radius R, current I, on-axis:
            # B_z = mu0 I R^2 / (2 (R^2 + z^2)^(3/2)).
            R = 0.1
            I = 1.0
            mu0 = 4 * math.pi * 1e-7
            zs = np.linspace(0.0, 0.2, 5)
            ref = mu0 * I * R * R / (2 * (R * R + zs * zs) ** 1.5)

            np.testing.assert_allclose(
                B[:, 2], ref, rtol=1e-4,
                err_msg=f"on-axis B_z={B[:, 2]!r} vs ref={ref!r}")
            self.assertTrue(np.all(np.abs(B[:, 0]) < 1e-12 * np.abs(ref).max()))
            self.assertTrue(np.all(np.abs(B[:, 1]) < 1e-12 * np.abs(ref).max()))


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class SolenoidExampleTest(unittest.TestCase):

    def test_axial_field_profile_matches_surface_current_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = _copy_example("solenoid", Path(tmp))
            _run_cli(workdir / "input.yaml")

            archive = np.load(workdir / "out.npz", allow_pickle=False)
            B = archive["B_xyz"]
            self.assertEqual(B.shape, (29, 3))

            R = 0.02
            L = 0.50
            N = 200
            I = 1.0
            mu0 = 4 * math.pi * 1e-7
            n = N / L

            zs = np.linspace(-0.35, 0.35, 29)

            # Surface-current closed form.
            def Bz_ideal(z):
                return (mu0 * n * I / 2) * (
                    (L / 2 + z) / math.sqrt(R * R + (L / 2 + z) ** 2)
                    + (L / 2 - z) / math.sqrt(R * R + (L / 2 - z) ** 2)
                )

            # Midpoint: 2% tolerance (helix discretization + finite aspect).
            mid_idx = 14  # 29 points -> index 14 is z=0
            self.assertAlmostEqual(zs[mid_idx], 0.0, places=10)
            self.assertAlmostEqual(B[mid_idx, 2], Bz_ideal(0.0),
                                   delta=2e-2 * Bz_ideal(0.0))

            # Half-field-at-the-end signature: at z = +L/2 the field should
            # be near 1/2 of the deep-inside value, within 5%.
            # Find index closest to z = 0.25 (which is L/2).
            end_idx = int(np.argmin(np.abs(zs - 0.25)))
            self.assertLess(abs(B[end_idx, 2]
                                - 0.5 * Bz_ideal(0.0)),
                            0.05 * Bz_ideal(0.0))

            # The helix breaks perfect axial symmetry (each turn has finite
            # pitch and is discretized as a polyline), so on-axis radial
            # components do not vanish to roundoff - they scale with the
            # pitch and the segments-per-turn. We just sanity-check that
            # they are small compared to the dominant B_z.
            self.assertTrue(np.all(np.abs(B[:, 0])
                                   < 5e-2 * np.abs(B[mid_idx, 2])),
                            msg=f"|B_x|/|B_z(mid)| max="
                                f"{np.max(np.abs(B[:, 0])) / abs(B[mid_idx, 2])}")
            self.assertTrue(np.all(np.abs(B[:, 1])
                                   < 5e-2 * np.abs(B[mid_idx, 2])),
                            msg=f"|B_y|/|B_z(mid)| max="
                                f"{np.max(np.abs(B[:, 1])) / abs(B[mid_idx, 2])}")


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class SaddleCoilExampleTest(unittest.TestCase):

    def test_c2_symmetry_kills_in_plane_components_at_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = _copy_example("saddle_coil", Path(tmp))
            _run_cli(workdir / "input.yaml")

            archive = np.load(workdir / "out.npz", allow_pickle=False)
            B = archive["B_xyz"]
            self.assertEqual(B.shape, (5, 3))

            # Index 0 is the origin. The saddle ring is invariant under
            # rotation by pi about the z-axis (phi -> phi + pi maps
            # (x, y, z) -> (-x, -y, z), so the closed loop traverses the
            # same curve). The B-field at a point on this axis must
            # therefore be invariant under that C2 rotation, which forces
            # B_x = B_y = 0 there. Discretization preserves the symmetry
            # because the polyline starts at phi=0 with step 2pi/N (N=32);
            # both the original and the rotated polyline traverse the same
            # set of vertices, so the on-axis numerical B is C2-invariant
            # too.
            B_center = B[0]
            scale = abs(B_center[2])
            self.assertGreater(scale, 0.0,
                               msg="B_z at origin unexpectedly zero")
            self.assertLess(abs(B_center[0]), 1e-12 * scale,
                            msg=f"B_x at origin = {B_center[0]}")
            self.assertLess(abs(B_center[1]), 1e-12 * scale,
                            msg=f"B_y at origin = {B_center[1]}")

            # The off-axis +x and +y points should also obey the C2
            # rotation property: B(R p) = R B(p) where R is the rotation
            # by pi. Point 1 is at (0.04, 0, 0); point 2 is at (0, 0.04,
            # 0). They are not on the rotation axis so they do not need
            # symmetry-by-rotation, but the magnitudes should match by
            # the broader 4-fold "almost-symmetry" of the loop. We just
            # smoke-check that |B| is non-zero there.
            self.assertGreater(np.linalg.norm(B[1]), 0.0)
            self.assertGreater(np.linalg.norm(B[2]), 0.0)


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class HelmholtzPairExampleTest(unittest.TestCase):

    def test_midpoint_field_matches_closed_form_and_is_quasi_uniform(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = _copy_example("helmholtz_pair", Path(tmp))
            _run_cli(workdir / "input.yaml")

            archive = np.load(workdir / "out.npz", allow_pickle=False)
            B = archive["B_xyz"]
            self.assertEqual(B.shape, (9, 3))

            R = 0.1
            I = 1.0
            mu0 = 4 * math.pi * 1e-7
            # (4/5)^(3/2) * mu0 * I / R
            B_z_center = mu0 * I / R * (4.0 / 5.0) ** 1.5

            midpoint = B[4, 2]  # 9 points -> index 4 is z=0
            self.assertAlmostEqual(midpoint, B_z_center,
                                   delta=1e-4 * B_z_center,
                                   msg=f"midpoint B_z={midpoint} vs ref={B_z_center}")

            # 9 points span |z| <= 0.02 m == 0.2 R; the leading O((z/R)^4)
            # Helmholtz deviation is (144/125)*(z/R)^4 ~ 1.8e-3 at the
            # endpoints, so 2.5e-3 is a tight bound.
            B_z_vals = B[:, 2]
            rel = np.abs(B_z_vals - B_z_center) / B_z_center
            self.assertTrue(np.all(rel < 2.5e-3),
                            msg=f"non-uniform across line: rel={rel!r} "
                                f"B_z={B_z_vals!r}")


if __name__ == "__main__":
    unittest.main()
