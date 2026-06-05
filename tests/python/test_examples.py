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


if __name__ == "__main__":
    unittest.main()
