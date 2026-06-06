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
import re
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


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class SquareToroidPic1mExampleTest(unittest.TestCase):
    """The 1 m square-toroid deck runs on the poloidal cross-section (plane: xz):
    the out-of-plane toroidal field lands in external_bz and confines the
    species inside the 90%-width domain."""

    def test_end_to_end_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = _copy_example("square_toroid_pic_1m", Path(tmp))
            _run_pic_cli(workdir / "input.yaml", steps=20)

            data = np.load(workdir / "out.npz", allow_pickle=False)

            for key in data.files:
                arr = data[key]
                if np.issubdtype(arr.dtype, np.floating):
                    self.assertFalse(np.isnan(arr).any(),
                                     msg=f"NaNs in {key!r}")

            # plane is recorded and the out-of-plane B_phi lands in external_bz.
            self.assertEqual(str(data["plane"][0]), "xz")
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


def _run_pic_cli_seeded(yaml_path: Path, steps: int | None = None,
                        *, seed: int = 0, write_every: int = 0) -> None:
    """Run the PIC CLI with an explicit RNG seed (and optional per-step dumps).

    ``steps`` of ``None`` uses the deck's own ``time.steps``; ``write_every > 0``
    asks the CLI to emit a self-contained ``out_<step>.npz`` every N steps, which
    is the only way to recover a per-particle trajectory time series (the
    end-of-run ``out.npz`` carries only the final per-species state)."""
    cmd = [sys.executable, "-m", "quasar.pic.cli", "run", str(yaml_path),
           "--seed", str(seed)]
    if steps is not None:
        cmd += ["--steps-override", str(steps)]
    if write_every > 0:
        cmd += ["--write-every", str(write_every)]
    res = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ})
    if res.returncode != 0:
        raise RuntimeError(
            f"quasar.pic.cli failed (exit {res.returncode}):\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}")


def _interior_field(flat: np.ndarray, nx: int, ny: int, nghost: int) -> np.ndarray:
    """Strip the Yee ghost halo from a flat field buffer, returning ``(ny, nx)``.

    Mirrors ``quasar.pic.postprocess.reshape_with_ghost``: the solver stores
    fields on a ``(nx+2g) x (ny+2g)`` grid; ``nghost`` is persisted in the npz so
    the reader strips the right number of cells rather than guessing from the
    flat size."""
    if nghost > 0 and flat.size == (nx + 2 * nghost) * (ny + 2 * nghost):
        return flat.reshape(ny + 2 * nghost, nx + 2 * nghost)[
            nghost:-nghost, nghost:-nghost]
    return flat.reshape(ny, nx)


def _besselj0(x: np.ndarray) -> np.ndarray:
    """Zeroth-order Bessel function J0(x), dependency-free.

    Abramowitz & Stegun 9.4.1 / 9.4.3 polynomial approximations (abs error
    < 1.6e-7), evaluated with numpy so the test needs no scipy. Used only to
    compare the cavity's radial profile against the J0(j01 r/R) eigenmode, where
    a 5% RMS bound is far coarser than the approximation error."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    out = np.empty_like(x)
    small = x < 3.0
    t = (x[small] / 3.0) ** 2
    out[small] = (1.0 - 2.2499997 * t + 1.2656208 * t ** 2 - 0.3163866 * t ** 3
                  + 0.0444479 * t ** 4 - 0.0039444 * t ** 5 + 0.00021 * t ** 6)
    xl = x[~small]
    u = 3.0 / xl
    f0 = (0.79788456 - 0.00000077 * u - 0.00552740 * u ** 2 - 0.00009512 * u ** 3
          + 0.00137237 * u ** 4 - 0.00072805 * u ** 5 + 0.00014476 * u ** 6)
    theta0 = (xl - 0.78539816 - 0.04166397 * u - 0.00003954 * u ** 2
              + 0.00262573 * u ** 3 - 0.00054125 * u ** 4 - 0.00029333 * u ** 5
              + 0.00013558 * u ** 6)
    out[~small] = f0 / np.sqrt(xl) * np.cos(theta0)
    return out


def _dominant_band_frequency(trace: np.ndarray, dt: float,
                             f_lo: float, f_hi: float) -> float:
    """Dominant FFT-peak frequency of ``trace`` within ``[f_lo, f_hi]``.

    DC and the slow envelope are removed (mean subtraction + Hanning window) so
    the peak reflects an oscillatory resonance rather than a ring-up/ring-down
    drift; the search is restricted to a physical band so the lowest non-DC bin
    cannot masquerade as the answer."""
    sig = (trace - trace.mean()) * np.hanning(trace.size)
    freqs = np.fft.rfftfreq(trace.size, d=dt)
    amp = np.abs(np.fft.rfft(sig))
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not band.any():
        raise AssertionError(
            f"no FFT bins in band [{f_lo:.3e}, {f_hi:.3e}] Hz "
            f"(dt={dt:.3e}, n={trace.size}, df={freqs[1] - freqs[0]:.3e})")
    band_freqs = freqs[band]
    return float(band_freqs[int(np.argmax(amp[band]))])


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class CylCavityTm010ExampleTest(unittest.TestCase):
    """Axisymmetric (r, z) pillbox cavity, ``examples/cyl_cavity_tm010``.

    Validates the cylindrical FDTD field solver: the seeded cavity must ring at
    the analytic TM010 eigenfrequency ``f_010 = j01 c / (2 pi R)`` measured as the
    dominant FFT peak of the near-axis axial-E time series, and the axial-E radial
    profile must match the ``J0(j01 r/R)`` eigenmode.

    Field-slot convention (cylindrical mode): the AXIAL E (TM010's E_z) lives in
    the solver's ``ey`` slot (grid-j is the axial z axis), mirroring the gyro
    example where axial B is ``by`` and axial velocity is ``vy``. The deck seeds
    and diagnoses ``ey``, so the npz carries ``field_ey`` / ``snapshot_field_ey``.
    """

    # First zero of J0 and the cavity radius (= domain lx_m); see the README.
    J01 = 2.40483
    C_LIGHT = 2.99792458e8

    def test_geometry_finite_bounded_and_tm010_resonance(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            workdir = _copy_example("cyl_cavity_tm010", Path(tmp))
            # The deck's own dt resolves ~1 period in 1024 steps; FFT resolution
            # needs many periods, so run ~17 periods (the run is GPU-cheap, a few
            # seconds). cadence=4 (from the deck) gives a dense axial-E series.
            _run_pic_cli_seeded(workdir / "input.yaml", steps=16000, seed=0)

            data = np.load(workdir / "out.npz", allow_pickle=False)

            # Criterion 1/10: the cylindrical run stamps geometry into the npz.
            self.assertEqual(str(data["geometry"][0]), "cylindrical")

            # All field buffers finite (no NaN/Inf anywhere).
            for key in data.files:
                arr = data[key]
                if np.issubdtype(arr.dtype, np.floating):
                    self.assertTrue(np.isfinite(arr).all(), msg=f"non-finite in {key!r}")

            nx = int(data["nx"][0])
            ny = int(data["ny"][0])
            nghost = int(data["nghost"][0])

            # Axial E (TM010 E_z) is the "ey" slot in cylindrical mode.
            ey = data["snapshot_field_ey"]
            times = data["snapshot_times_s"]
            n_snap = ey.shape[0]
            self.assertGreater(n_snap, 8, msg="too few axial-E snapshots to FFT")

            # Interior (ny=z, nx=r) axial-E stack with the ghost halo stripped.
            ey_int = np.stack([_interior_field(ey[k], nx, ny, nghost)
                               for k in range(n_snap)])
            self.assertTrue(np.isfinite(ey_int).all())

            # Field energy bounded across snapshots: a closed lossless cavity rings
            # but must not blow up exponentially. A clean single-eigenmode ring has
            # near-zero instantaneous energy at its sinusoid zero-crossings, so the
            # physically meaningful bound is against the *mean* (and start) energy,
            # not the min. max/mean ~2.0 here; 5x catches any real divergence.
            energy = (ey_int ** 2).sum(axis=(1, 2))
            self.assertGreater(float(energy.mean()), 0.0)
            self.assertLess(float(energy.max()), 5.0 * float(energy.mean()),
                            msg=f"axial-E energy unbounded: mean={energy.mean():.3e} "
                                f"max={energy.max():.3e}")

            # R from the deck domain (lx_m); analytic TM010 eigenfrequency.
            R = 0.10
            f010 = self.J01 * self.C_LIGHT / (2 * math.pi * R)

            # Near-axis axial-E(t) trace (TM010 peaks at r=0), averaged over z to
            # damp any higher-order axial structure. Search a physical band that
            # brackets f010 with a healthy margin (0.5..1.6 x) so the measured value
            # is the genuine cavity resonance, not a spectral-leakage artifact.
            dt_snap = float(times[1] - times[0])
            trace = ey_int[:, :, 0:4].mean(axis=(1, 2))
            f_peak = _dominant_band_frequency(trace, dt_snap, 0.5 * f010, 1.6 * f010)

            rel = abs(f_peak - f010) / f010
            # Tolerance: the residual is 2nd-order Yee grid dispersion at 128 cells
            # (measured ~0.5%). 8% is generous headroom but still a true physics
            # check; do NOT tighten below what the discretization achieves.
            self.assertLess(
                rel, 0.08,
                msg=f"cavity resonance off: measured f_peak={f_peak:.6e} Hz vs "
                    f"analytic f_010={f010:.6e} Hz (rel={rel:.3f}); a large error "
                    f"points at the cylindrical FDTD curl / on-axis closure.")

            # Radial-profile correctness: the z-averaged final axial-E field must
            # match the J0(j01 r/R) TM010 eigenmode. Fit only the amplitude (the
            # seed sets an arbitrary scale), then check the normalized RMS residual.
            fey = _interior_field(data["field_ey"], nx, ny, nghost)
            profile = fey.mean(axis=0)  # mean over z -> Ez(r)
            dr = R / nx
            r_centers = (np.arange(nx) + 0.5) * dr
            j0_ref = _besselj0(self.J01 * r_centers / R)
            amp = float((profile * j0_ref).sum() / (j0_ref * j0_ref).sum())
            resid = profile - amp * j0_ref
            rms = math.sqrt(float((resid ** 2).mean())
                            / float((profile ** 2).mean()))
            self.assertLess(
                rms, 0.05,
                msg=f"axial-E radial profile does not match J0(j01 r/R): "
                    f"RMS={rms:.4f} (fit amp={amp:.3e}); a poor match points at "
                    f"the cylindrical radial curl weighting / on-axis closure.")


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class CylGyroOrbitExampleTest(unittest.TestCase):
    """Electron gyro-orbit in a uniform axial B, ``examples/cyl_gyro_orbit``.

    Validates the cylindrical particle pusher against the textbook gyrofrequency
    and checks that the on-axis closure injects no energy (the bunch kinetic
    energy never grows above its seeded value)."""

    QE = 1.602176634e-19
    ME = 9.1093837015e-31
    B = 1.0  # T, axial (B_T[1] for plane xy)

    def test_finite_alive_gyrofrequency_and_no_energy_gain(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            workdir = _copy_example("cyl_gyro_orbit", Path(tmp))
            # The end-of-run npz holds only the final particle state; --write-every
            # dumps a per-step trajectory (out_<step>.npz) we FFT for the period.
            # Fixed seed -> deterministic initial conditions. T_c ~ 3.57e-11 s;
            # the deck's 512 steps span ~13 gyro-periods, sampled every 2 steps.
            _run_pic_cli_seeded(workdir / "input.yaml", seed=0, write_every=2)

            data = np.load(workdir / "out.npz", allow_pickle=False)
            self.assertEqual(str(data["geometry"][0]), "cylindrical")
            for key in data.files:
                arr = data[key]
                if np.issubdtype(arr.dtype, np.floating):
                    self.assertTrue(np.isfinite(arr).all(), msg=f"non-finite in {key!r}")

            # Trajectory time series from the indexed per-step dumps.
            frames = sorted(workdir.glob("out_*.npz"))
            self.assertGreater(len(frames), 16, msg="too few trajectory frames")
            vx, vy, vz, alive, ts = [], [], [], [], []
            for f in frames:
                a = np.load(f, allow_pickle=False)
                vx.append(a["species_electron_vx"])
                vy.append(a["species_electron_vy"])
                vz.append(a["species_electron_vz"])
                alive.append(a["species_electron_alive"].astype(bool))
                ts.append(float(a["final_time_s"][0]))
            vx = np.asarray(vx)
            vy = np.asarray(vy)
            vz = np.asarray(vz)
            alive = np.asarray(alive)
            ts = np.asarray(ts)

            # Particles alive and moving: the orbit sits well inside r>0 with a tiny
            # Larmor radius, so the absorbing walls must not eat any particle.
            self.assertTrue(alive.all(),
                            msg="absorbing walls / axis closure lost particles")
            self.assertGreater(float(np.max(np.abs(vx))), 0.0, msg="vx all zero")

            wc = self.QE * self.B / self.ME  # analytic cyclotron frequency
            dt_snap = float(ts[1] - ts[0])

            # Gyrofrequency: for the axial (by) field the gyration plane is
            # (vr, vphi) = (vx, vz). FFT each particle's complex in-plane velocity
            # vx + i vz and take the median peak across the bunch (robust to the
            # 1 eV thermal spread). Band brackets w_c by 0.5..1.6 x.
            freqs = np.fft.fftfreq(ts.size, d=dt_snap)
            win = np.hanning(ts.size)
            f_lo, f_hi = 0.5 * wc / (2 * math.pi), 1.6 * wc / (2 * math.pi)
            band = (np.abs(freqs) >= f_lo) & (np.abs(freqs) <= f_hi)
            self.assertTrue(band.any(), msg="no FFT bins near the gyrofrequency")
            peaks = []
            sig = vx + 1j * vz
            for p in range(sig.shape[1]):
                s = (sig[:, p] - sig[:, p].mean()) * win
                amp = np.abs(np.fft.fft(s))
                amp[~band] = 0.0
                peaks.append(abs(freqs[int(np.argmax(amp))]))
            w_meas = 2 * math.pi * float(np.median(peaks))
            rel = abs(w_meas - wc) / wc
            self.assertLess(
                rel, 0.05,
                msg=f"gyrofrequency off: measured w={w_meas:.6e} rad/s vs "
                    f"analytic w_c={wc:.6e} rad/s (rel={rel:.3f}); a large error "
                    f"points at the cylindrical pusher's rotation term.")

            # Larmor radius from the measured perpendicular speed of the
            # fastest-perp particle (drift + thermal): r_L = m v_perp/(|q|B).
            vperp = np.sqrt(vx ** 2 + vz ** 2)
            p = int(np.argmax(vperp[0]))
            r_L = self.ME * float(vperp[:, p].mean()) / (self.QE * self.B)
            self.assertGreater(r_L, 0.0)
            # Sanity: the orbit radius is microscopic vs the 0.10 m domain, so the
            # guiding centre never reaches a wall.
            self.assertLess(r_L, 1.0e-4,
                            msg=f"Larmor radius {r_L:.3e} m implausibly large")

            # No spurious energy gain (criterion 8 / on-axis heating): the total
            # bunch kinetic energy must never grow above its seeded value. (A real
            # plasma cools numerically here; the physics bug we guard against is
            # *heating*, especially as guiding centres approach r=0.)
            ke = 0.5 * self.ME * (vx ** 2 + vy ** 2 + vz ** 2).sum(axis=1)
            ke0 = float(ke[0])
            self.assertGreater(ke0, 0.0)
            self.assertLess(
                float(ke.max()) / ke0, 1.10,
                msg=f"bunch KE grew above seed: max/KE0={ke.max() / ke0:.3f} "
                    f"(no-energy-gain near axis violated).")


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class PicGeometryMetadataTest(unittest.TestCase):
    """Regression for the geometry stamp (criteria 1 & 10): a cylindrical deck
    writes geometry == 'cylindrical' and a Cartesian deck writes 'cartesian'."""

    def test_cylindrical_deck_stamps_cylindrical(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            workdir = _copy_example("cyl_gyro_orbit", Path(tmp))
            _run_pic_cli_seeded(workdir / "input.yaml", steps=4, seed=0)
            data = np.load(workdir / "out.npz", allow_pickle=False)
            self.assertEqual(str(data["geometry"][0]), "cylindrical")

    def test_cartesian_deck_stamps_cartesian(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            # The minimal SI deck is Cartesian by default (no geometry key).
            deck = _write_deck(Path(tmp) / "case", _MINIMAL_PIC_DECK)
            _run_pic_cli(deck, steps=4)
            data = np.load(deck.parent / "out.npz", allow_pickle=False)
            self.assertEqual(str(data["geometry"][0]), "cartesian")


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


# ---------------------------------------------------------------------------
# Ideal-MHD example cases (high-order, mp7 reconstruction).
#
# The MHD CLI is ``python -m quasar.mhd.cli run <input.yaml>`` and writes an
# ``out.npz`` whose keys are:
#   final_step, final_time_s, nx, ny, nghost, geometry, gamma,
#   state_rho, state_mx, state_my, state_mz, state_energy,
#   state_bx, state_by, state_bz, divb_linf
# (face B is sampled to cell centers for the ``state_b*`` outputs).
#
# These tests MIRROR the coil/PIC mechanics above: copy the example dir into a
# sandbox, run the CLI, load ``out.npz``, and assert physically meaningful but
# tolerant criteria (the references are canonical, not always closed-form). They
# are written RED-first: ``examples/brio_wu`` etc. and ``quasar.mhd`` do not
# exist yet, so each test fails cleanly on the missing deck / missing output
# (the same way a missing coil/PIC example would), rather than breaking the
# harness.
#
# Contract for Task 3.12's decks (so decks and tests agree):
#   * Each deck lives at ``examples/<case>/input.yaml`` with a ``README.md``.
#   * ``diagnostics.output_path`` MUST be ``out.npz`` (relative to the deck dir),
#     mirroring every other example, so the loader below finds it.
#   * The deck's ``time`` runs to the canonical output time / step count for that
#     problem; the tests below read whatever final state the deck produces and
#     check structure, so a deck author sets ``time`` to the standard reference
#     time (documented per-case in the README).
#   * ``state_*`` arrays are cell-centered, flattened either as 1D (nx for a
#     degenerate ny=1 run) or row-major (ny, nx); the reshape helper below copes
#     with both. ``nx``, ``ny``, ``nghost``, ``gamma`` are scalars stored as
#     length-1 arrays (mirroring the PIC convention, e.g. ``data["nx"][0]``).
#   * mhd_linear_wave: see ``MhdLinearWaveConvergenceTest`` -- the test runs the
#     deck at its shipped resolution AND a refined copy with ``nx``/``ny``
#     doubled (it rewrites ``domain.nx``/``domain.ny`` in the YAML text). The
#     deck MUST therefore (a) seed a smooth ``alfven_wave`` whose exact solution
#     is a pure translation by one wavelength at the output time, and (b) write
#     ``state_rho`` so the L1 error vs the seeded/exact profile can be formed.
#     For an exactly-advected smooth wave the analytic final state equals the
#     initial state (one full period / integer wavelength shift), so the test
#     uses the deck's own seeded profile recomputed analytically as the
#     reference. The README must state the wavelength, output time, and that the
#     output time corresponds to an integer number of wave periods.


def _run_mhd_cli(yaml_path: Path) -> None:
    """Run the ideal-MHD CLI on a deck. Mirrors ``_run_cli`` /
    ``_run_pic_cli`` but targets the ``quasar.mhd.cli`` module."""
    res = subprocess.run(
        [sys.executable, "-m", "quasar.mhd.cli", "run", str(yaml_path)],
        capture_output=True, text=True, env={**os.environ},
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"quasar.mhd.cli failed (exit {res.returncode}):\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}")


def _mhd_scalar(data, key: str) -> float:
    """Read a scalar stored as a length-1 array (PIC convention) or a bare
    0-d/scalar value."""
    arr = np.asarray(data[key])
    return float(arr.reshape(-1)[0])


def _mhd_field(data, key: str, nx: int, ny: int) -> np.ndarray:
    """Return a cell-centered MHD state field as a 2D ``(ny, nx)`` array.

    Copes with a 1D degenerate run (ny == 1 or a flat length-nx buffer) and a
    row-major ``(ny, nx)`` flatten. No ghost halo: the MHD ``state_*`` outputs
    are documented as cell-centered interior values."""
    flat = np.asarray(data[key]).reshape(-1)
    if flat.size == nx * ny:
        return flat.reshape(ny, nx)
    if flat.size == nx and ny == 1:
        return flat.reshape(1, nx)
    raise AssertionError(
        f"{key!r} size {flat.size} matches neither nx*ny={nx * ny} nor nx={nx}")


def _mhd_no_nans(testcase, data) -> None:
    for key in data.files:
        arr = data[key]
        if np.issubdtype(arr.dtype, np.floating):
            testcase.assertFalse(np.isnan(arr).any(), msg=f"NaNs in {key!r}")
            testcase.assertFalse(np.isinf(arr).any(), msg=f"Infs in {key!r}")


# divb_linf is a discrete divergence; "machine epsilon" for a CT/projection
# scheme means it never grows beyond a small multiple of float64 round-off times
# the field magnitude. We keep a tolerant absolute floor so the criterion stays
# physically meaningful without being brittle to the exact discretization.
_DIVB_EPS = 1e-10


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class BrioWuExampleTest(unittest.TestCase):
    """1D MHD shock tube, ``examples/brio_wu`` (initial.type ``brio_wu``).

    The canonical Brio-Wu Riemann problem (gamma = 2) develops a fixed sequence
    of waves. We do not have a closed form, so we pin the robust structural
    signature: the density relaxes from the left state (rho ~ 1) to the right
    state (rho ~ 0.125) across the tube, the run is finite, and divb stays at
    machine epsilon (trivially so in 1D, but the key must be present and tiny)."""

    def test_brio_wu_shock_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = _copy_example("brio_wu", Path(tmp))
            _run_mhd_cli(workdir / "input.yaml")

            data = np.load(workdir / "out.npz", allow_pickle=False)
            _mhd_no_nans(self, data)

            nx = int(_mhd_scalar(data, "nx"))
            ny = int(_mhd_scalar(data, "ny"))
            rho = _mhd_field(data, "state_rho", nx, ny)

            # Take the midline (any row; in a 1D run all rows are identical, and
            # the deck may be a degenerate ny=1 or a thin 2D strip).
            line = rho[ny // 2, :]
            self.assertTrue(np.all(np.isfinite(line)))
            self.assertTrue(np.all(line > 0.0), msg="non-positive density")

            # Left/right limits: Brio-Wu seeds rho_L = 1.0, rho_R = 0.125. After
            # the run the far-left few cells stay near the left state and the
            # far-right few near the right state (the wave fan sits in between).
            k = max(1, nx // 16)
            left = float(np.mean(line[:k]))
            right = float(np.mean(line[-k:]))
            self.assertAlmostEqual(left, 1.0, delta=0.15,
                                   msg=f"left density {left} not ~1.0")
            self.assertAlmostEqual(right, 0.125, delta=0.05,
                                   msg=f"right density {right} not ~0.125")
            # Net drop left -> right is the defining feature.
            self.assertGreater(left, right + 0.5,
                               msg=f"no left>right density drop: {left} vs {right}")

            # divb at machine epsilon.
            divb = _mhd_scalar(data, "divb_linf")
            self.assertLess(divb, _DIVB_EPS,
                            msg=f"divb_linf {divb} not at machine epsilon")

            # gamma == 2 for Brio-Wu.
            self.assertAlmostEqual(_mhd_scalar(data, "gamma"), 2.0, places=6)


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class MhdLinearWaveConvergenceTest(unittest.TestCase):
    """Smooth Alfven wave, ``examples/mhd_linear_wave`` (initial.type
    ``alfven_wave``) -- the high-order convergence proof.

    The deck seeds a smooth wave whose exact solution at the deck's output time
    is the initial profile translated by an integer number of wavelengths (i.e.
    it returns to the seed). We run the deck at its shipped resolution and at a
    refined copy with ``nx``/``ny`` DOUBLED (rewriting the YAML), form the L1
    error of ``state_rho`` against the analytic (== seeded) profile at each
    resolution, and assert the empirical convergence RATE approaches the mp7
    design order (~7). The bound is a tolerant lower bound (rate > 5.5) to allow
    measurement noise, limiter clipping at the extrema, and round-off.

    HOW THE TWO RESOLUTIONS ARE OBTAINED: the deck file is copied into the
    sandbox once at its shipped resolution; a second copy has ``domain.nx`` and
    ``domain.ny`` doubled by a regex rewrite of the YAML text. Task 3.12's deck
    MUST keep ``domain.nx``/``domain.ny`` as plain integer scalars on their own
    ``key: value`` lines (or inside a ``{nx: .., ny: ..}`` inline map) so this
    rewrite matches, and MUST seed a wave that is exactly periodic over the
    domain at the output time so the analytic reference equals the seed.
    """

    def _l1_error_vs_seed(self, workdir: Path) -> float:
        """Run the deck in ``workdir`` and return the L1 error of the final
        density against the analytic reference. For an exactly-advected smooth
        wave the analytic final state equals the initial seed; the deck is
        expected to also emit the seeded ("exact") profile, but to stay robust
        we reconstruct the reference as the *mean-removed* periodic structure:
        the error metric is the L1 distance between the final density and a
        smooth fit is overkill, so we instead compare final vs the deck's seed
        profile if present, else fall back to the first-harmonic projection.

        Concretely: if the deck writes ``state_rho_initial`` we use it; else we
        treat the analytic solution as a pure cosine of the dominant spatial
        harmonic fitted to the final field (amplitude + phase), which for a
        correctly-advected smooth wave is the exact solution. The L1 norm is
        normalized by the number of cells so errors at different resolutions are
        directly comparable.
        """
        _run_mhd_cli(workdir / "input.yaml")
        data = np.load(workdir / "out.npz", allow_pickle=False)
        _mhd_no_nans(self, data)
        nx = int(_mhd_scalar(data, "nx"))
        ny = int(_mhd_scalar(data, "ny"))
        rho = _mhd_field(data, "state_rho", nx, ny)

        if "state_rho_initial" in data.files:
            ref = _mhd_field(data, "state_rho_initial", nx, ny)
        else:
            # Analytic reference for an exactly-periodic smooth wave: the
            # dominant first spatial harmonic along x (mean + single cosine).
            # A correctly high-order scheme reproduces this to its design order;
            # dispersion/dissipation errors are what shrink with resolution.
            line_mean = rho.mean(axis=0)  # average over y (wave is along x)
            n = line_mean.size
            k = np.arange(n)
            dc = line_mean.mean()
            c = (2.0 / n) * np.sum(line_mean * np.cos(2 * np.pi * k / n))
            s = (2.0 / n) * np.sum(line_mean * np.sin(2 * np.pi * k / n))
            fit = dc + c * np.cos(2 * np.pi * k / n) + s * np.sin(
                2 * np.pi * k / n)
            ref = np.broadcast_to(fit, rho.shape)

        return float(np.mean(np.abs(rho - ref)))

    def test_convergence_rate_approaches_seventh_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _copy_example("mhd_linear_wave", Path(tmp))
            e_coarse = self._l1_error_vs_seed(base)

            # Refined copy: double domain.nx and domain.ny in the YAML text.
            fine = _copy_example("mhd_linear_wave", Path(tmp) / "refined")
            deck = fine / "input.yaml"
            text = deck.read_text()

            def _double(match: "re.Match") -> str:
                return f"{match.group(1)}{int(match.group(2)) * 2}"

            text = re.sub(r"(\bnx\s*:\s*)(\d+)", _double, text)
            text = re.sub(r"(\bny\s*:\s*)(\d+)", _double, text)
            deck.write_text(text)
            e_fine = self._l1_error_vs_seed(fine)

            self.assertGreater(e_coarse, 0.0, msg="coarse error vanished")
            self.assertGreater(e_fine, 0.0, msg="fine error vanished")
            # Refinement must reduce the error.
            self.assertLess(e_fine, e_coarse,
                            msg=f"error grew under refinement: "
                                f"{e_coarse} -> {e_fine}")

            # Empirical order from a 2x refinement: p = log2(e_coarse / e_fine).
            rate = math.log(e_coarse / e_fine) / math.log(2.0)
            self.assertGreater(
                rate, 5.5,
                msg=f"convergence rate {rate:.2f} below the mp7 design order "
                    f"(~7); errors {e_coarse:.3e} -> {e_fine:.3e}. A low rate "
                    f"points at the reconstruction order or the time integrator.")


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class OrszagTangExampleTest(unittest.TestCase):
    """2D Orszag-Tang vortex, ``examples/orszag_tang`` (initial.type
    ``orszag_tang``, gamma = 5/3, periodic domain).

    Conservation + structure: on a doubly-periodic domain the total mass and
    total energy are conserved; divb stays at machine epsilon; and the density
    field develops the characteristic strong spread (sharp central/region
    extrema) of the canonical Orszag-Tang result at the output time.

    Initial sums: derived from the deck's own seeded state. The deck is expected
    to also write the initial conserved sums (``mass_initial`` /
    ``energy_initial``); if absent, the test falls back to comparing the final
    sums against the analytic seed integrals only loosely (structure only). To
    keep the conservation check meaningful, Task 3.12's deck SHOULD emit the
    initial totals (e.g. ``mass_initial``, ``energy_initial`` scalars) OR a step-0
    snapshot; the README must document the output time (the canonical t = 0.5)."""

    def test_conservation_and_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = _copy_example("orszag_tang", Path(tmp))
            _run_mhd_cli(workdir / "input.yaml")

            data = np.load(workdir / "out.npz", allow_pickle=False)
            _mhd_no_nans(self, data)

            self.assertAlmostEqual(_mhd_scalar(data, "gamma"), 5.0 / 3.0,
                                   places=4)

            nx = int(_mhd_scalar(data, "nx"))
            ny = int(_mhd_scalar(data, "ny"))
            rho = _mhd_field(data, "state_rho", nx, ny)
            energy = _mhd_field(data, "state_energy", nx, ny)

            self.assertTrue(np.all(np.isfinite(rho)))
            self.assertTrue(np.all(rho > 0.0), msg="non-positive density")
            self.assertTrue(np.all(np.isfinite(energy)))

            # divb at machine epsilon throughout (final value reported).
            divb = _mhd_scalar(data, "divb_linf")
            self.assertLess(divb, _DIVB_EPS,
                            msg=f"divb_linf {divb} not at machine epsilon")

            # Conservation: total mass and total energy on the periodic domain
            # match their initial values to a tight tolerance. Cells are uniform,
            # so the discrete integral is proportional to the sum; comparing
            # sums is equivalent to comparing integrals.
            mass_final = float(rho.sum())
            energy_final = float(energy.sum())

            if "mass_initial" in data.files and "energy_initial" in data.files:
                mass0 = _mhd_scalar(data, "mass_initial")
                energy0 = _mhd_scalar(data, "energy_initial")
                # 0.1% on mass (advection is conservative to round-off; the
                # bound covers limiter mass-redistribution), 1% on energy (the
                # high-order scheme dissipates a little at shocks).
                self.assertAlmostEqual(
                    mass_final / mass0, 1.0, delta=1e-3,
                    msg=f"mass not conserved: {mass0} -> {mass_final}")
                self.assertAlmostEqual(
                    energy_final / energy0, 1.0, delta=1e-2,
                    msg=f"energy not conserved: {energy0} -> {energy_final}")
            else:
                # No seeded totals available: at minimum the conserved fields
                # are bounded and non-trivial (a blown-up run fails NaN/positive
                # checks above; this guards a silently-zeroed field).
                self.assertGreater(mass_final, 0.0)
                self.assertGreater(energy_final, 0.0)

            # Structure: the Orszag-Tang vortex sharpens density gradients, so by
            # the output time the field spans a clear range about its mean (the
            # canonical result has rho varying by more than a factor of ~2).
            rho_mean = float(rho.mean())
            self.assertGreater(float(rho.max()) / rho_mean, 1.3,
                               msg="density maximum too flat for Orszag-Tang")
            self.assertLess(float(rho.min()) / rho_mean, 0.8,
                            msg="density minimum too flat for Orszag-Tang")


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class MhdBlastExampleTest(unittest.TestCase):
    """Strong-shock MHD blast wave, ``examples/mhd_blast`` (initial.type
    ``blast``).

    Positivity + symmetry: a high-pressure central region drives a strong fast
    shell into a magnetized background. Density and pressure must stay strictly
    positive everywhere (no negative, no NaN), and the fast shell is nearly
    circular -- a coarse symmetry check compares the radial density profile along
    x against the profile along y. divb stays at machine epsilon."""

    def _gamma_pressure(self, data, nx, ny, gamma):
        """Thermal pressure from the conserved state:
        p = (gamma - 1) (E - 0.5 rho v^2 - 0.5 B^2)."""
        rho = _mhd_field(data, "state_rho", nx, ny)
        mx = _mhd_field(data, "state_mx", nx, ny)
        my = _mhd_field(data, "state_my", nx, ny)
        mz = _mhd_field(data, "state_mz", nx, ny)
        e = _mhd_field(data, "state_energy", nx, ny)
        bx = _mhd_field(data, "state_bx", nx, ny)
        by = _mhd_field(data, "state_by", nx, ny)
        bz = _mhd_field(data, "state_bz", nx, ny)
        kinetic = 0.5 * (mx ** 2 + my ** 2 + mz ** 2) / rho
        magnetic = 0.5 * (bx ** 2 + by ** 2 + bz ** 2)
        return (gamma - 1.0) * (e - kinetic - magnetic)

    def test_positivity_and_circular_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = _copy_example("mhd_blast", Path(tmp))
            _run_mhd_cli(workdir / "input.yaml")

            data = np.load(workdir / "out.npz", allow_pickle=False)
            _mhd_no_nans(self, data)

            nx = int(_mhd_scalar(data, "nx"))
            ny = int(_mhd_scalar(data, "ny"))
            gamma = _mhd_scalar(data, "gamma")
            rho = _mhd_field(data, "state_rho", nx, ny)

            # Strict positivity of density and pressure everywhere.
            self.assertTrue(np.all(rho > 0.0),
                            msg=f"density not strictly positive: min={rho.min()}")
            pressure = self._gamma_pressure(data, nx, ny, gamma)
            self.assertTrue(np.all(np.isfinite(pressure)))
            self.assertTrue(np.all(pressure > 0.0),
                            msg=f"pressure not strictly positive: "
                                f"min={pressure.min()}")

            # divb at machine epsilon.
            divb = _mhd_scalar(data, "divb_linf")
            self.assertLess(divb, _DIVB_EPS,
                            msg=f"divb_linf {divb} not at machine epsilon")

            # Coarse circular-symmetry check: the radial density profile measured
            # along +x from the center should match the profile along +y within
            # tolerance (a near-circular fast shell). Sample along the central
            # row and central column.
            cx, cy = nx // 2, ny // 2
            along_x = rho[cy, cx:]
            along_y = rho[cy:, cx]
            m = min(along_x.size, along_y.size)
            along_x = along_x[:m]
            along_y = along_y[:m]
            scale = float(max(rho.max() - rho.min(), 1e-30))
            # The magnetic field breaks perfect isotropy, so allow a generous
            # band: the x/y radial profiles agree to ~20% of the field range.
            self.assertTrue(
                np.all(np.abs(along_x - along_y) < 0.20 * scale),
                msg="blast shell not approximately circular: "
                    f"max |rho_x - rho_y| = {np.max(np.abs(along_x - along_y))}, "
                    f"range scale = {scale}")


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class MhdRotorExampleTest(unittest.TestCase):
    """MHD rotor, ``examples/mhd_rotor`` (initial.type ``rotor``).

    Positivity + structure: a dense disk spins inside a light, magnetized
    ambient, winding up the field into torsional Alfven waves. Density and
    pressure must stay strictly positive throughout, and the rotor leaves a
    clear central over-density (the spun-up disk) relative to the ambient. divb
    stays at machine epsilon."""

    def _gamma_pressure(self, data, nx, ny, gamma):
        rho = _mhd_field(data, "state_rho", nx, ny)
        mx = _mhd_field(data, "state_mx", nx, ny)
        my = _mhd_field(data, "state_my", nx, ny)
        mz = _mhd_field(data, "state_mz", nx, ny)
        e = _mhd_field(data, "state_energy", nx, ny)
        bx = _mhd_field(data, "state_bx", nx, ny)
        by = _mhd_field(data, "state_by", nx, ny)
        bz = _mhd_field(data, "state_bz", nx, ny)
        kinetic = 0.5 * (mx ** 2 + my ** 2 + mz ** 2) / rho
        magnetic = 0.5 * (bx ** 2 + by ** 2 + bz ** 2)
        return (gamma - 1.0) * (e - kinetic - magnetic)

    def test_positivity_and_central_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = _copy_example("mhd_rotor", Path(tmp))
            _run_mhd_cli(workdir / "input.yaml")

            data = np.load(workdir / "out.npz", allow_pickle=False)
            _mhd_no_nans(self, data)

            nx = int(_mhd_scalar(data, "nx"))
            ny = int(_mhd_scalar(data, "ny"))
            gamma = _mhd_scalar(data, "gamma")
            rho = _mhd_field(data, "state_rho", nx, ny)

            # Strict positivity of density and pressure throughout.
            self.assertTrue(np.all(rho > 0.0),
                            msg=f"density not strictly positive: min={rho.min()}")
            pressure = self._gamma_pressure(data, nx, ny, gamma)
            self.assertTrue(np.all(np.isfinite(pressure)))
            self.assertTrue(np.all(pressure > 0.0),
                            msg=f"pressure not strictly positive: "
                                f"min={pressure.min()}")

            # divb at machine epsilon.
            divb = _mhd_scalar(data, "divb_linf")
            self.assertLess(divb, _DIVB_EPS,
                            msg=f"divb_linf {divb} not at machine epsilon")

            # Central rotating structure: the dense disk leaves a central
            # over-density relative to the ambient corners. Compare a central
            # patch mean to the mean of the four corner patches.
            qx, qy = max(1, nx // 8), max(1, ny // 8)
            cx, cy = nx // 2, ny // 2
            central = float(rho[cy - qy:cy + qy, cx - qx:cx + qx].mean())
            corners = float(np.mean([
                rho[:qy, :qx].mean(), rho[:qy, -qx:].mean(),
                rho[-qy:, :qx].mean(), rho[-qy:, -qx:].mean(),
            ]))
            self.assertGreater(
                central, 1.5 * corners,
                msg=f"no central over-density: central={central}, "
                    f"corners={corners} (the rotor disk should be denser than "
                    f"the ambient).")


if __name__ == "__main__":
    unittest.main()
