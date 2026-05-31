"""Integration tests for the worked examples in ``examples/``.

Covers both CLIs. The coil tests invoke ``python -m quasar.coil.cli run
<input.yaml>``, load the produced ``out.npz``, and compare the computed B-field
to a closed-form analytical reference. The PIC tests invoke
``python -m quasar.pic.cli run <input.yaml>`` and check the end-to-end run
(output signature, finiteness, and physical diagnostics) rather than a
closed-form field.
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
         str(yaml_path)],
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
            current = 1.0
            mu0 = 4 * math.pi * 1e-7
            zs = np.linspace(0.0, 0.2, 5)
            ref = mu0 * current * R * R / (2 * (R * R + zs * zs) ** 1.5)

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
            current = 1.0
            mu0 = 4 * math.pi * 1e-7
            n = N / L

            zs = np.linspace(-0.35, 0.35, 29)

            # Surface-current closed form.
            def Bz_ideal(z):
                return (mu0 * n * current / 2) * (
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
            current = 1.0
            mu0 = 4 * math.pi * 1e-7
            # (4/5)^(3/2) * mu0 * current / R
            B_z_center = mu0 * current / R * (4.0 / 5.0) ** 1.5

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


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class SquareToroidExampleTest(unittest.TestCase):

    def test_field_is_finite_and_nontrivial(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = _copy_example("square_toroid", Path(tmp))
            _run_cli(workdir / "input.yaml")

            archive = np.load(workdir / "out.npz", allow_pickle=False)
            B = archive["B_xyz"]
            mag = archive["B_magnitude"]
            # Field is finite everywhere on the meridional plane.
            self.assertTrue(np.all(np.isfinite(B)))
            self.assertTrue(np.all(np.isfinite(mag)))
            # |B| from components matches the reported magnitude.
            np.testing.assert_allclose(
                np.linalg.norm(B, axis=1), mag, rtol=1e-6, atol=0.0)
            # A 1.5 kA toroidal sheet magnet produces a clearly non-zero field
            # inside the bore.
            self.assertGreater(float(np.max(mag)), 1.0e-4)


def _run_pic_cli(yaml_path: Path, steps: int) -> None:
    res = subprocess.run(
        [sys.executable, "-m", "quasar.pic.cli", "run",
         str(yaml_path), "--steps-override", str(steps)],
        capture_output=True, text=True, env={**os.environ},
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"quasar.pic.cli failed (exit {res.returncode}):\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}")


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class SquareToroidPicExampleTest(unittest.TestCase):

    def test_end_to_end_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = _copy_example("square_toroid_pic", Path(tmp))
            _run_pic_cli(workdir / "input.yaml", steps=20)

            data = np.load(workdir / "out.npz", allow_pickle=False)

            for key in data.files:
                arr = data[key]
                if np.issubdtype(arr.dtype, np.floating):
                    self.assertFalse(np.isnan(arr).any(),
                                     msg=f"NaNs in {key!r}")

            ext_bz = data["external_bz"]
            self.assertGreater(float(np.max(np.abs(ext_bz))), 1.0e-3,
                               msg=f"external Bz too small: "
                                   f"|max|={np.max(np.abs(ext_bz))}")

            for sp in ("H+", "mu-"):
                vx = data[f"species_{sp}_vx"]
                alive = data[f"species_{sp}_alive"].astype(bool)
                self.assertGreater(int(alive.sum()), 0,
                                   msg=f"no alive particles in {sp!r}")
                self.assertGreater(float(np.max(np.abs(vx[alive]))), 0.0,
                                   msg=f"{sp!r}: vx remained zero")


# A minimal, self-contained SI PIC deck. The PIC example decks under examples/
# are runnable through the CLI and covered by PicAspirationalExampleTests below;
# this inline deck is kept as a fast, deterministic SI smoke case that exercises
# the same end-to-end run path without depending on any example file.
_MINIMAL_PIC_DECK = """\
units: SI
domain: {nx: 16, ny: 16, lx_m: 1.0, ly_m: 1.0}
numerics: {fdtd_order: 2, shape: cic}
species:
  - name: e
    charge_C: -1.0
    mass_kg: 1.0
    n_particles: 256
    initial:
      distribution: maxwellian_uniform
      density_per_m3: 1.0e+6
      temperature_eV: 10.0
time: {dt_s: auto, steps: 8}
diagnostics: {output_path: out.npz, per_species: true}
boundary: {particle: [periodic, periodic, periodic, periodic]}
"""


def _write_deck(into: Path, text: str) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    deck = into / "input.yaml"
    deck.write_text(text)
    return deck


def _run_example_pic(name: str, steps: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = _copy_example(name, Path(tmp))
        _run_pic_cli(workdir / "input.yaml", steps)
        return np.load(workdir / "out.npz", allow_pickle=False)


def _all_finite(data) -> bool:
    return all(
        not np.isnan(data[k]).any()
        for k in data.files
        if np.issubdtype(data[k].dtype, np.floating))


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class PicAspirationalExampleTests(unittest.TestCase):
    """The nine canonical PIC validation decks must load and run end-to-end and
    show their characteristic signature."""

    def test_two_stream_field_energy_grows(self):
        data = _run_example_pic("two_stream", 200)
        self.assertTrue(_all_finite(data))
        ex = data["snapshot_field_ex"]
        energy = (ex ** 2).sum(axis=1)
        # The two-stream instability grows the longitudinal field energy by orders
        # of magnitude before saturation.
        self.assertGreater(energy[-1], 50.0 * energy[0],
                           msg=f"Ex energy did not grow: {energy[0]} -> {energy[-1]}")

    def test_filtered_two_stream_runs_and_grows(self):
        data = _run_example_pic("filtered_two_stream", 200)
        self.assertTrue(_all_finite(data))
        ex = data["snapshot_field_ex"]
        energy = (ex ** 2).sum(axis=1)
        self.assertGreater(energy[-1], 10.0 * energy[0])

    def test_landau_damping_runs(self):
        data = _run_example_pic("landau_damping", 60)
        self.assertTrue(_all_finite(data))
        self.assertIn("snapshot_field_ex", data.files)

    def test_weibel_grows_transverse_b(self):
        data = _run_example_pic("weibel", 200)
        self.assertTrue(_all_finite(data))
        bz = data["snapshot_field_bz"]
        energy = (bz ** 2).sum(axis=1)
        # Weibel grows a transverse magnetic field from noise.
        self.assertGreater(energy[-1], energy[0],
                           msg=f"Bz energy did not grow: {energy[0]} -> {energy[-1]}")

    def test_em_wave_energy_stays_bounded(self):
        data = _run_example_pic("em_wave_propagation", 120)
        self.assertTrue(_all_finite(data))
        ez = data["snapshot_field_ez"]
        by = data["snapshot_field_by"]
        energy = (ez ** 2 + by ** 2).sum(axis=1)
        # Vacuum propagation on a periodic grid: energy bounded (no blow-up). The
        # instantaneous E^2+B^2 oscillates with the leapfrog half-step staggering.
        self.assertLess(energy.max(), 2.0 * energy.min())

    def test_beam_in_channel_confines_particles(self):
        data = _run_example_pic("beam_in_channel", 60)
        self.assertTrue(_all_finite(data))
        alive = data["species_electron_beam_alive"].astype(bool)
        # Reflecting y walls keep every particle alive.
        self.assertEqual(int(alive.sum()), alive.size,
                         msg="specular walls lost particles")

    def test_pec_cavity_energy_bounded(self):
        data = _run_example_pic("pec_cavity", 120)
        self.assertTrue(_all_finite(data))
        ez = data["snapshot_field_ez"]
        by = data["snapshot_field_by"]
        energy = (ez ** 2 + by ** 2).sum(axis=1)
        # Closed lossless PEC cavity: energy must not blow up.
        self.assertLess(energy.max(), 2.0 * energy.min())

    def test_magnetized_plasma_runs(self):
        data = _run_example_pic("magnetized_plasma", 40)
        self.assertTrue(_all_finite(data))
        ext_bz = data["external_bz"]
        # Uniform 1 T field present in the external buffer.
        self.assertGreater(float(np.max(np.abs(ext_bz))), 0.5)

    def test_coil_confinement_runs(self):
        data = _run_example_pic("coil_confinement", 40)
        self.assertTrue(_all_finite(data))
        ext_bz = data["external_bz"]
        self.assertGreater(float(np.max(np.abs(ext_bz))), 1.0e-4)


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class PicRunSmokeTest(unittest.TestCase):
    """End-to-end CLI run on a minimal SI deck: clean exit, finite fields,
    live moving particles."""

    def test_run_stays_finite_with_moving_particles(self):
        with tempfile.TemporaryDirectory() as tmp:
            deck = _write_deck(Path(tmp) / "case", _MINIMAL_PIC_DECK)
            _run_pic_cli(deck, steps=8)

            out = deck.parent / "out.npz"
            self.assertTrue(out.exists(), msg="no out.npz produced")
            data = np.load(out, allow_pickle=False)

            for key in data.files:
                arr = data[key]
                if np.issubdtype(arr.dtype, np.floating):
                    self.assertFalse(np.isnan(arr).any(), msg=f"NaNs in {key!r}")
                    self.assertFalse(np.isinf(arr).any(), msg=f"Infs in {key!r}")

            alive = data["species_e_alive"].astype(bool)
            self.assertGreater(int(alive.sum()), 0, msg="no live particles")
            vx = data["species_e_vx"][alive]
            self.assertGreater(float(np.max(np.abs(vx))), 0.0, msg="vx all zero")


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class PicCliDiagnosticsTest(unittest.TestCase):
    """Exercise the --log-every / --write-every runtime diagnostic paths."""

    def test_scalar_series_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            deck = _write_deck(Path(tmp) / "case", _MINIMAL_PIC_DECK)
            res = subprocess.run(
                [sys.executable, "-m", "quasar.pic.cli", "run",
                 str(deck),
                 "--steps-override", "8",
                 "--log-every", "2",
                 "--write-every", "4"],
                capture_output=True, text=True, env={**os.environ},
            )
            self.assertEqual(res.returncode, 0,
                             msg=f"cli failed:\n{res.stdout}\n{res.stderr}")

            data = np.load(deck.parent / "out.npz", allow_pickle=False)
            series_keys = [k for k in data.files if k.startswith("series_")]
            self.assertIn("series_step", series_keys)
            # log-every=2 over 8 steps records at 2,4,6,8 -> 4 samples.
            self.assertEqual(int(data["series_step"].shape[0]), 4,
                             msg=f"unexpected series length: {data['series_step']}")
            alive_series = [k for k in series_keys if k.startswith("series_alive_")]
            self.assertTrue(alive_series, msg="no per-species alive series")
            for k in alive_series:
                self.assertTrue((data[k] >= 0).all(), msg=f"negative counts in {k}")


if __name__ == "__main__":
    unittest.main()
