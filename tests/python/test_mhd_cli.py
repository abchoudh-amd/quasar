"""CLI tests for the high-order ideal-MHD module.

Mirrors how the PIC CLI is invoked in ``tests/python/test_examples.py``:
``python -m quasar.mhd.cli run <input.yaml>`` via :mod:`subprocess`, then loads
the produced ``out.npz`` with ``np.load(..., allow_pickle=False)`` and asserts on
its keys. The end-to-end stepping run is guarded by the same
``QUASAR_HAS_HIP_RUNTIME`` skip the PIC example tests use; the invalid-deck path
exits non-zero *before* stepping, so it needs no device.

Until the ``quasar.mhd`` package exists, ``python -m quasar.mhd.cli`` exits
non-zero (ModuleNotFoundError), so the run tests fail rather than error -- the
intended RED state.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


def has_hip_runtime() -> bool:
    return os.environ.get("QUASAR_HAS_HIP_RUNTIME", "0") == "1"


# All documented out.npz keys the MHD writer must emit.
EXPECTED_KEYS = (
    "final_step",
    "final_time_s",
    "nx",
    "ny",
    "nghost",
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
