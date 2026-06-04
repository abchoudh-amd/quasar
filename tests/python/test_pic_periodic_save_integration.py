"""Integration tests for periodic per-step snapshot saving in ``quasar pic run``.

``--write-every N`` writes distinct, step-indexed, self-contained snapshot
files ``out_<step>.npz`` (10-digit zero-padded completed-step count, e.g.
``out_0000000002.npz``) into the deck's output directory, in addition to the
unchanged end-of-run ``out.npz``.

These tests drive the real PIC CLI as a subprocess against the smallest PIC
example deck (``square_toroid_pic``). Real PIC stepping needs a HIP GPU, so the
class is gated on a visible HIP runtime and otherwise skips on CPU.
"""

from __future__ import annotations

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


def _run_pic_cli(yaml_path: Path, steps: int, *extra: str) -> subprocess.CompletedProcess:
    """Run ``quasar.pic.cli run <yaml> --steps-override <steps> [extra...]``."""
    return subprocess.run(
        [sys.executable, "-m", "quasar.pic.cli", "run",
         str(yaml_path), "--steps-override", str(steps), *extra],
        capture_output=True, text=True, env={**os.environ},
    )


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class PicPeriodicSaveIntegrationTest(unittest.TestCase):

    def test_write_every_emits_indexed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = _copy_example("square_toroid_pic", Path(tmp))
            res = _run_pic_cli(workdir / "input.yaml", 6, "--write-every", "2")
            self.assertEqual(res.returncode, 0,
                             msg=f"cli failed:\n{res.stdout}\n{res.stderr}")

            # The deck writes out.npz into its own directory (the sandbox copy).
            deck_dir = workdir

            for step in (2, 4, 6):
                per_path = deck_dir / f"out_{step:010d}.npz"
                self.assertTrue(per_path.exists(),
                                msg=f"missing per-step file {per_path.name}")
                data = np.load(per_path, allow_pickle=False)
                self.assertEqual(int(data["final_step"][0]), step,
                                 msg=f"{per_path.name}: final_step "
                                     f"{data['final_step']!r} != {step}")
                field_keys = [k for k in data.files if k.startswith("field_")]
                self.assertTrue(field_keys,
                                msg=f"{per_path.name}: no field_* key present")

            # The end-of-run aggregate out.npz is still written and carries the
            # scalar series (series_* records even without --log-every).
            out = deck_dir / "out.npz"
            self.assertTrue(out.exists(), msg="no end-of-run out.npz produced")
            agg = np.load(out, allow_pickle=False)
            self.assertIn("series_step", agg.files,
                          msg="end-of-run out.npz missing series_step")

    def test_no_write_every_emits_no_indexed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = _copy_example("square_toroid_pic", Path(tmp))
            res = _run_pic_cli(workdir / "input.yaml", 4)
            self.assertEqual(res.returncode, 0,
                             msg=f"cli failed:\n{res.stdout}\n{res.stderr}")

            deck_dir = workdir
            indexed = sorted(deck_dir.glob("out_0*.npz"))
            self.assertEqual(indexed, [],
                             msg=f"unexpected indexed files: "
                                 f"{[p.name for p in indexed]}")
            self.assertTrue((deck_dir / "out.npz").exists(),
                            msg="no end-of-run out.npz produced")


if __name__ == "__main__":
    unittest.main()
