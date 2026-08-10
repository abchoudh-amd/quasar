"""CLI tests for the high-order ideal-MHD module.

Mirrors how the PIC CLI is invoked in ``tests/python/test_examples.py``:
``python -m quasar.mhd.cli run <input.yaml>`` via :mod:`subprocess`, then loads
the produced ``out.npz`` with ``np.load(..., allow_pickle=False)`` and asserts on
its keys. The end-to-end stepping run is guarded by the same
``QUASAR_HAS_HIP_RUNTIME`` skip the PIC example tests use; the invalid-deck path
exits non-zero *before* stepping, so it needs no device.

"""

import argparse
import os
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from quasar.mhd import cli as mhd_cli
from quasar.mhd.io import parse as parse_mhd_deck


def has_hip_runtime() -> bool:
    return os.environ.get("QUASAR_HAS_HIP_RUNTIME", "0") == "1"


# All documented out.npz keys the MHD writer must emit.
EXPECTED_KEYS = (
    "final_step",
    "final_time_s",
    "nx",
    "ny",
    "nghost",
    "units",
    "geometry",
    "gamma",
    "state_rho",
    "state_mx",
    "state_my",
    "state_mz",
    "state_energy",
    "state_bx",
    "state_by",
    "state_bz",
    "divb_linf",
)


def _small_deck(output_path: str = "out.npz") -> dict:
    return {
        "units": "normalized",
        "domain": {"nx": 16, "ny": 16, "lx_m": 1.0, "ly_m": 1.0},
        "geometry": "cartesian",
        "numerics": {
            "gamma": 1.6666667,
            "reconstruction": "mp7",
            "riemann": "hlld",
            "integrator": "ssprk3",
            "ct": "fd_ct_christlieb",
            "positivity": "troubled_cell",
            "rho_floor": 1.0e-8,
            "p_floor": 1.0e-9,
            "cfl": 0.4,
        },
        "initial": {"type": "orszag_tang", "params": {}},
        "time": {"dt_s": "auto", "steps": 3},
        "diagnostics": {"output_path": output_path, "cadence": 1,
                        "fields": ["rho", "energy"], "divb": True},
        "boundary": {
            "fluid": ["periodic", "periodic", "periodic", "periodic"],
            "field": ["periodic", "periodic", "periodic", "periodic"],
        },
    }


def _write_deck(directory: Path, data: dict, name: str = "input.yaml") -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(data))
    return path


def _run_cli(yaml_path: Path):
    return subprocess.run(
        [sys.executable, "-m", "quasar.mhd.cli", "run", str(yaml_path)],
        capture_output=True, text=True, env={**os.environ},
    )


class MhdNativeBackgroundConfigTests(unittest.TestCase):

    def test_disabled_default_does_not_carry_a_curl_free_assertion(self):
        data = _small_deck()
        data["background_field"] = {
            "enabled": False,
            "profile": "not-registered",
            "bx0": "not-a-number",
            "params": ["not", "a", "mapping"],
        }
        cfg = mhd_cli._make_config(parse_mhd_deck(data))
        self.assertIs(cfg.background.enabled, False)
        self.assertIs(cfg.background.curl_free, False)
        self.assertEqual(cfg.background.profile, "uniform")
        self.assertEqual(cfg.background.bx0, 0.0)

    def test_make_config_forwards_analytic_profile_params(self):
        data = _small_deck()
        data["background_field"] = {
            "enabled": True,
            "profile": "linear_vacuum",
            "params": {"gradient": 1.25, "shear": -0.4},
        }
        cfg = mhd_cli._make_config(parse_mhd_deck(data))
        self.assertEqual(cfg.background.params,
                         {"gradient": 1.25, "shear": -0.4})
        self.assertEqual(cfg.background.profile_scale, 1.0)
        # The native registry capability, rather than a frontend assertion,
        # supplies the analytic profile's trusted curl-free proof.
        self.assertIs(cfg.background.curl_free, False)

    def test_make_config_scales_native_analytic_profile_for_si_units(self):
        data = _small_deck()
        data["units"] = "SI"
        data["background_field"] = {
            "enabled": True,
            "profile": "linear_vacuum",
            "params": {"gradient": 1.25, "shear": -0.4},
        }
        cfg = mhd_cli._make_config(parse_mhd_deck(data))
        self.assertAlmostEqual(
            cfg.background.profile_scale,
            1.0 / mhd_cli.mhd_units.SQRT_MU0)

    def test_make_config_does_not_treat_file_params_as_profile_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "background.npz"
            path.touch()
            data = _small_deck()
            data["background_field"] = {
                "enabled": True,
                "profile": "stale-profile-that-is-ignored",
                "file": str(path),
            }
            cfg = mhd_cli._make_config(parse_mhd_deck(data))
        self.assertEqual(cfg.background.params, {})
        self.assertEqual(cfg.background.profile, "uniform")
        self.assertEqual(cfg.background.bx0, 0.0)
        self.assertEqual(cfg.background.profile_scale, 1.0)
        # The default profile is uniform, but arbitrary file samples do not
        # inherit that analytic profile's proof.
        self.assertIs(cfg.background.curl_free, False)

    def test_projected_conductors_are_curl_free_only_without_toroidal_bz0(self):
        data = _small_deck()
        data["units"] = "SI"
        data["geometry"] = "cylindrical"
        data["domain"]["origin_x_m"] = 0.5
        data["numerics"]["reconstruction"] = "muscl_minmod"
        data["boundary"] = {
            "fluid": ["outflow"] * 4,
            "field": ["outflow"] * 4,
        }
        data["background_field"] = {
            "enabled": True,
            "profile": "stale-profile-that-is-ignored",
            "conductors": [{
                "name": "loop",
                "current_A": 1.0,
                "geometry": {
                    "type": "circular_loop",
                    "center_xyz": [0.0, 0.0, 0.0],
                    "axis_xyz": [0.0, 0.0, 1.0],
                    "radius_m": 0.1,
                    "n_segments": 32,
                },
            }],
            "bz0": 0.2,
            "params": {"b_scale": 1.5, "vacuum_project": True},
        }

        cfg = mhd_cli._make_config(parse_mhd_deck(data))

        self.assertEqual(cfg.background.profile, "uniform")
        self.assertEqual(cfg.background.bz0, 0.0)
        self.assertEqual(cfg.background.params, {})
        self.assertEqual(cfg.background.profile_scale, 1.0)
        # vacuum_project proves only the in-plane poloidal field curl-free.
        # A constant cylindrical B_phi=bz0 remains current-carrying.
        self.assertIs(cfg.background.curl_free, False)

        data["background_field"]["bz0"] = 0.0
        cfg = mhd_cli._make_config(parse_mhd_deck(data))
        self.assertIs(cfg.background.curl_free, True)


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class MhdCliRunTests(unittest.TestCase):

    def test_run_writes_npz_with_all_documented_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            deck = _write_deck(workdir, _small_deck())
            res = _run_cli(deck)
            self.assertEqual(
                res.returncode, 0,
                msg=f"mhd cli failed:\nstdout: {res.stdout}\nstderr: {res.stderr}")

            out = workdir / "out.npz"
            self.assertTrue(out.exists(), "out.npz was not written")
            data = np.load(out, allow_pickle=False)
            for key in EXPECTED_KEYS:
                with self.subTest(key=key):
                    self.assertIn(key, data.files)

    def test_run_metadata_matches_deck(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            deck = _write_deck(workdir, _small_deck())
            res = _run_cli(deck)
            self.assertEqual(res.returncode, 0,
                             msg=f"stderr: {res.stderr}")

            data = np.load(workdir / "out.npz", allow_pickle=False)
            self.assertEqual(int(np.asarray(data["nx"]).ravel()[0]), 16)
            self.assertEqual(int(np.asarray(data["ny"]).ravel()[0]), 16)
            self.assertAlmostEqual(
                float(np.asarray(data["gamma"]).ravel()[0]), 1.6666667, places=5)
            self.assertEqual(
                str(np.asarray(data["geometry"]).ravel()[0]), "cartesian")
            self.assertEqual(
                str(np.asarray(data["units"]).ravel()[0]), "normalized")

    def test_run_state_arrays_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            deck = _write_deck(workdir, _small_deck())
            res = _run_cli(deck)
            self.assertEqual(res.returncode, 0, msg=f"stderr: {res.stderr}")

            data = np.load(workdir / "out.npz", allow_pickle=False)
            for key in data.files:
                arr = data[key]
                if np.issubdtype(arr.dtype, np.floating):
                    self.assertFalse(np.isnan(arr).any(), msg=f"NaNs in {key!r}")

    def test_run_divb_linf_finite_and_small(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            deck = _write_deck(workdir, _small_deck())
            res = _run_cli(deck)
            self.assertEqual(res.returncode, 0, msg=f"stderr: {res.stderr}")

            data = np.load(workdir / "out.npz", allow_pickle=False)
            divb = np.asarray(data["divb_linf"], dtype=float)
            self.assertTrue(np.all(np.isfinite(divb)),
                            msg=f"divb_linf not finite: {divb!r}")
            # Constrained-transport keeps div B at the discretization floor.
            self.assertLess(float(np.max(np.abs(divb))), 1.0e-6,
                            msg=f"divb_linf too large: {float(np.max(np.abs(divb)))}")

    def test_run_respects_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            deck = _write_deck(workdir, _small_deck(output_path="result.npz"))
            res = _run_cli(deck)
            self.assertEqual(res.returncode, 0, msg=f"stderr: {res.stderr}")
            self.assertTrue((workdir / "result.npz").exists())

    def test_run_uses_suffixless_output_path_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            deck = _write_deck(workdir, _small_deck(output_path="result"))
            res = _run_cli(deck)
            self.assertEqual(res.returncode, 0, msg=f"stderr: {res.stderr}")
            self.assertTrue((workdir / "result").is_file())
            self.assertFalse((workdir / "result.npz").exists())
            with np.load(workdir / "result", allow_pickle=False) as archive:
                self.assertIn("final_time_s", archive.files)

    def test_t_end_clips_the_last_step_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            data = _small_deck()
            data["time"] = {"dt_s": "auto", "steps": 100, "t_end": 1.0e-8}
            deck = _write_deck(workdir, data)
            res = _run_cli(deck)
            self.assertEqual(res.returncode, 0, msg=f"stderr: {res.stderr}")
            out = np.load(workdir / "out.npz", allow_pickle=False)
            self.assertEqual(float(np.asarray(out["final_time_s"]).ravel()[0]),
                             data["time"]["t_end"])
            self.assertEqual(int(np.asarray(out["final_step"]).ravel()[0]), 1)


class MhdCliInvalidDeckTests(unittest.TestCase):
    """An invalid deck must fail (non-zero exit) before any stepping, so these run
    without a GPU."""

    def _run_expect_failure(self, data: dict):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            deck = _write_deck(workdir, data)
            res = _run_cli(deck)
            self.assertNotEqual(
                res.returncode, 0,
                msg=f"expected non-zero exit for invalid deck; "
                    f"stdout: {res.stdout}\nstderr: {res.stderr}")
            # The failure must come from deck validation, NOT from the module
            # being absent or failing to import. (Without this guard these tests
            # would pass spuriously while quasar.mhd does not yet exist.) The CLI
            # must be invokable and reach its validator before refusing the deck.
            combined = (res.stdout + res.stderr).lower()
            self.assertNotIn("no module named", combined,
                             msg=f"CLI module not importable:\n{res.stderr}")
            self.assertNotIn("error while finding module", combined,
                             msg=f"CLI module not importable:\n{res.stderr}")
            self.assertFalse((workdir / "out.npz").exists(),
                             "out.npz must not be written for an invalid deck")

    def test_bad_gamma_exits_nonzero(self):
        data = _small_deck()
        data["numerics"]["gamma"] = 0.5
        self._run_expect_failure(data)

    def test_unknown_reconstruction_exits_nonzero(self):
        data = _small_deck()
        data["numerics"]["reconstruction"] = "nope"
        self._run_expect_failure(data)

    def test_nonpositive_nx_exits_nonzero(self):
        data = _small_deck()
        data["domain"]["nx"] = 0
        self._run_expect_failure(data)

    def test_unknown_initial_type_exits_nonzero(self):
        data = _small_deck()
        data["initial"]["type"] = "not_a_real_ic"
        self._run_expect_failure(data)


class MhdCliTimeAccountingTests(unittest.TestCase):
    class _Grid:
        nghost = 0

    class _Solver:
        def __init__(self, nx, ny, limits):
            self._state = np.zeros(nx * ny, dtype=np.float64)
            self._limits = iter(limits)
            self.dts = []
            self.divergence_calls = 0

        def grid(self):
            return MhdCliTimeAccountingTests._Grid()

        def state_component_to_host(self, _component):
            return self._state.copy()

        def divergence_b_max(self):
            self.divergence_calls += 1
            return 0.0

        def cfl_limit(self):
            return next(self._limits)

        def step_unchecked(self, dt):
            self.dts.append(float(dt))

        def step(self, dt):
            self.dts.append(float(dt))

    @staticmethod
    def _deck(t_end, steps=8):
        data = _small_deck()
        data["domain"] = {"nx": 2, "ny": 2, "lx_m": 1.0, "ly_m": 1.0}
        data["initial"] = {"type": "blast", "params": {}}
        data["time"] = {"dt_s": "auto", "steps": steps, "t_end": t_end}
        data["diagnostics"] = {
            "output_path": "out.npz", "cadence": 0, "fields": [], "divb": True}
        return parse_mhd_deck(data)

    @staticmethod
    def _args():
        return argparse.Namespace(log_every=0)

    def test_exact_endpoint_is_the_sum_of_actual_solver_steps(self):
        # 0.3 is deliberately not an integer multiple of binary64 0.1. The
        # final call must receive the exact floating residual, not report an
        # epsilon-snapped endpoint after three nominal 0.1 steps.
        deck = self._deck(0.3)
        solver = self._Solver(2, 2, [0.1, 0.1, 0.1, 0.1])
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.npz"
            mhd_cli._run_loop(
                solver, deck, 0.1, True, out_path, self._args())
            out = np.load(out_path, allow_pickle=False)
            final_time = float(np.asarray(out["final_time_s"]).ravel()[0])
        self.assertEqual(len(solver.dts), 3)
        self.assertEqual(solver.dts[-1], 0.3 - (solver.dts[0] + solver.dts[1]))
        self.assertEqual(math.fsum(solver.dts), 0.3)
        self.assertEqual(final_time, 0.3)

    def test_unrepresentable_nonfinal_time_progress_raises(self):
        deck = self._deck(2.0e16)
        solver = self._Solver(2, 2, [1.0e16, 0.5])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "too small to advance"):
                mhd_cli._run_loop(
                    solver, deck, 1.0e16, True, Path(tmp) / "out.npz",
                    self._args())
        self.assertEqual(solver.dts, [1.0e16])

    def test_disabled_divb_performs_no_reductions_and_emits_no_keys(self):
        deck = self._deck(0.2, steps=2)
        deck.diagnostics.divb = False
        deck.diagnostics.cadence = 1
        solver = self._Solver(2, 2, [0.1, 0.1])
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.npz"
            mhd_cli._run_loop(
                solver, deck, 0.1, True, out_path,
                argparse.Namespace(log_every=1))
            with np.load(out_path, allow_pickle=False) as out:
                for key in ("divb_linf", "divb_linf_final",
                            "snapshot_divb_linf"):
                    self.assertNotIn(key, out.files)
        self.assertEqual(solver.divergence_calls, 0)


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class MhdCliOverCflDtTests(unittest.TestCase):
    """An explicit dt above the CFL limit is rejected at the CLI/solver layer (it
    needs the constructed solver + seeded state via MhdSolver2D::cfl_limit()),
    NOT by deck .validate(). The structurally-valid deck passes validation, then
    the run must refuse the over-CFL dt before producing output. Mirrors the
    Cartesian over-CFL guard in test_pic_cli.py. Needs a device, so GPU-guarded."""

    def test_explicit_dt_above_cfl_limit_run_fails_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            data = _small_deck()
            # A huge explicit dt for the seeded IC is far above the CFL stability
            # limit (orszag_tang on a 16x16 unit grid has a sub-unity stable dt).
            data["time"]["dt_s"] = 1.0e6
            deck = _write_deck(workdir, data)
            res = _run_cli(deck)
            self.assertNotEqual(
                res.returncode, 0,
                msg=f"expected non-zero exit for over-CFL dt; "
                    f"stdout: {res.stdout}\nstderr: {res.stderr}")
            combined = (res.stdout + res.stderr).lower()
            self.assertNotIn("no module named", combined,
                             msg=f"CLI module not importable:\n{res.stderr}")
            self.assertFalse((workdir / "out.npz").exists(),
                             "out.npz must not be written when dt exceeds the CFL limit")


if __name__ == "__main__":
    unittest.main()
