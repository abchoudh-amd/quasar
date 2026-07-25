"""Tests for ``--write-every`` per-step snapshot files in ``quasar.pic.cli``.

``--write-every N`` now writes a DISTINCT step-indexed, self-contained snapshot
file per qualifying step (e.g. ``out_0000000010.npz``) instead of overwriting a
single rolling ``out.npz``. The end-of-run aggregate ``out.npz`` (with the scalar
series and any cadence snapshots) is still written unchanged.

These tests are CPU-only: ``_run_loop`` is driven with a fake solver and a fake
identity ``Units`` so no GPU or real C++ solver is constructed. Fakes are defined
locally so the file is independent of ``test_pic_cli.py``.
"""

import argparse
import glob
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from quasar.pic.cli import _indexed_output_path, _run_loop
from quasar.pic.io import (
    Diagnostics,
    Domain,
    Numerics,
    PicDeck,
    Species,
    SpeciesInitial,
    Time,
)


# ----------------------------------------------------------------------------
# CPU-only fakes (no GPU / no real C++ solver)
# ----------------------------------------------------------------------------
class _FakeUnits:
    """Identity unit converter: every diagnostic conversion returns its input.

    ``_run_loop`` calls ``time_to_si``, ``_snapshot`` calls
    ``field_component_to_si``, and ``_species_to_si`` calls ``length_to_si`` /
    ``velocity_to_si``. All are linear-through-origin, so the identity faithfully
    exercises the flatten/save path without a real deck's normalization."""

    def field_component_to_si(self, name, v):
        return v

    def length_to_si(self, v):
        return v

    def velocity_to_si(self, v):
        return v

    def time_to_si(self, v):
        return v


class _FakeSpecies:
    """A single species' host-side particle arrays for ``species_at(idx).to_host()``."""

    def __init__(self, n=3):
        self._host = {
            "x": np.arange(n, dtype=float),
            "y": np.arange(n, dtype=float) + 0.5,
            "vx": np.zeros(n, dtype=float),
            "vy": np.zeros(n, dtype=float),
            "vz": np.zeros(n, dtype=float),
            "weight": np.ones(n, dtype=float),
        }

    def to_host(self):
        return dict(self._host)


class _FakeSolver:
    """Minimal CPU stand-in for the PIC solver used by ``_run_loop``.

    Returns a small flat array for each field component (``_snapshot`` only
    stores whatever the accessor returns), counts ``step``/``finalize`` calls,
    and exposes per-species host data when the deck requests it."""

    def __init__(self, storage=9, n_particles=3):
        self._storage = storage
        self._species = _FakeSpecies(n_particles)
        self.step_calls = 0
        self.finalize_calls = 0

    def step(self, dt):
        self.step_calls += 1

    def finalize(self):
        self.finalize_calls += 1

    def species_alive_count(self, idx):
        return 3

    def field_component_to_host(self, name):
        return np.full(self._storage, 1.0, dtype=float)

    def external_field_component_to_host(self, name):
        return np.full(self._storage, 2.0, dtype=float)

    def species_at(self, idx):
        return self._species


def _make_deck(out_dir, *, steps=10, per_species=False, output_path="out.npz"):
    """A tiny validated CPU deck.

    ``cadence=0`` disables the periodic in-memory snapshot list (so per-step
    files stay self-contained); ``fields=['bz']`` keeps a single field key. A
    species is always defined (a deck must drive something), but whether its
    host data lands in the npz is governed by ``diagnostics.per_species``."""
    deck = PicDeck(
        domain=Domain(nx=4, ny=4, lx_m=1.0, ly_m=1.0),
        numerics=Numerics(fdtd_order=2, shape="cic"),
        species=[Species(name="e", charge_C=-1.0, mass_kg=1.0, n_particles=3,
                         initial=SpeciesInitial())],
        time=Time(dt_s="auto", steps=steps),
        diagnostics=Diagnostics(output_path=output_path, cadence=0,
                                fields=["bz"], per_species=per_species),
        units="normalized",
    )
    deck.validate()
    return deck


def _drive(out_dir, *, steps=10, write_every=0, log_every=0,
           per_species=False, output_path="out.npz"):
    """Run ``_run_loop`` against the fakes and return ``(out_path, deck, solver)``."""
    deck = _make_deck(out_dir, steps=steps, per_species=per_species,
                      output_path=output_path)
    species_indices = list(range(len(deck.species)))
    solver = _FakeSolver()
    units = _FakeUnits()
    out_path = Path(out_dir) / output_path
    args = argparse.Namespace(write_every=write_every, log_every=log_every)
    _run_loop(solver, deck, species_indices, units, dt=1.0, dt_si=1.0e-9,
              out_path=out_path, args=args)
    return out_path, deck, solver


# ----------------------------------------------------------------------------
# Pure-helper tests for _indexed_output_path
# ----------------------------------------------------------------------------
class IndexedOutputPathTests(unittest.TestCase):

    def test_indexed_path_pads_step_to_ten_digits(self):
        self.assertEqual(
            _indexed_output_path(Path("/d/out.npz"), 10),
            Path("/d/out_0000000010.npz"))

    def test_indexed_path_preserves_dir_and_suffix(self):
        # Parent directory and .npz suffix preserved for a simple stem.
        p = _indexed_output_path(Path("/some/dir/out.npz"), 7)
        self.assertEqual(p.parent, Path("/some/dir"))
        self.assertEqual(p.suffix, ".npz")
        self.assertEqual(p.name, "out_0000000007.npz")
        # Multi-dot stems: only the final suffix is split off.
        multi = _indexed_output_path(Path("/d/out.tar.npz"), 5)
        self.assertEqual(multi, Path("/d/out.tar_0000000005.npz"))

    def test_indexed_names_sort_lexicographically_in_step_order(self):
        steps = [10, 20, 100]
        names = [_indexed_output_path(Path("/d/out.npz"), s).name for s in steps]
        # Sorting the names as strings yields the numeric (ascending) order.
        self.assertEqual(sorted(names), names)
        self.assertEqual(
            names,
            ["out_0000000010.npz", "out_0000000020.npz", "out_0000000100.npz"])


# ----------------------------------------------------------------------------
# _run_loop behavior tests (per-step snapshot files)
# ----------------------------------------------------------------------------
class WriteEveryTests(unittest.TestCase):

    def test_suffixless_aggregate_and_periodic_paths_are_used_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            out_path, _, _ = _drive(
                d, steps=5, write_every=5, output_path="result")
            periodic = Path(d) / "result_0000000005"
            self.assertTrue(out_path.is_file())
            self.assertTrue(periodic.is_file())
            self.assertFalse((Path(d) / "result.npz").exists())
            self.assertFalse((Path(d) / "result_0000000005.npz").exists())
            with np.load(out_path, allow_pickle=False) as archive:
                self.assertEqual(int(archive["final_step"][0]), 5)
            with np.load(periodic, allow_pickle=False) as archive:
                self.assertEqual(int(archive["final_step"][0]), 5)

    def test_write_every_creates_one_file_per_multiple(self):
        with tempfile.TemporaryDirectory() as d:
            out_path, _, _ = _drive(d, steps=10, write_every=5)
            per_files = sorted(os.path.basename(p)
                               for p in glob.glob(os.path.join(d, "out_*.npz")))
            self.assertEqual(
                per_files, ["out_0000000005.npz", "out_0000000010.npz"])
            # End-of-run aggregate still present.
            self.assertTrue(out_path.exists())

    def test_per_step_file_final_step_matches_filename(self):
        with tempfile.TemporaryDirectory() as d:
            _drive(d, steps=10, write_every=5)
            for step in (5, 10):
                per = Path(d) / f"out_{step:010d}.npz"
                self.assertTrue(per.exists(), f"missing {per}")
                with np.load(per) as npz:
                    self.assertEqual(int(npz["final_step"][0]), step)

    def test_per_step_file_is_self_contained_no_series_or_snapshots(self):
        with tempfile.TemporaryDirectory() as d:
            _drive(d, steps=10, write_every=5, per_species=True)
            per = Path(d) / "out_0000000005.npz"
            with np.load(per) as npz:
                keys = list(npz.keys())
            # Field component(s) present.
            self.assertTrue(any(k.startswith("field_") for k in keys), keys)
            # Per-species data present when per_species=True.
            self.assertTrue(any(k.startswith("species_") for k in keys), keys)
            # Self-contained: no time-series and no cadence-snapshot keys.
            self.assertFalse(any(k.startswith("series_") for k in keys), keys)
            self.assertFalse(any(k.startswith("snapshot_") for k in keys), keys)

    def test_write_every_non_multiple_final_step_emits_no_file(self):
        # When write_every does not divide steps, only the multiples get a
        # per-step file; the trailing non-multiple step (7) is captured solely
        # by the end-of-run out.npz, not an out_0000000007.npz.
        with tempfile.TemporaryDirectory() as d:
            out_path, _, _ = _drive(d, steps=7, write_every=5)
            per_files = sorted(os.path.basename(p)
                               for p in glob.glob(os.path.join(d, "out_*.npz")))
            self.assertEqual(per_files, ["out_0000000005.npz"])
            self.assertFalse((Path(d) / "out_0000000007.npz").exists())
            self.assertTrue(out_path.exists())
            with np.load(out_path) as npz:
                self.assertEqual(int(npz["final_step"][0]), 7)

    def test_write_every_zero_creates_no_per_step_files(self):
        with tempfile.TemporaryDirectory() as d:
            out_path, _, _ = _drive(d, steps=10, write_every=0)
            self.assertEqual(glob.glob(os.path.join(d, "out_*.npz")), [])
            self.assertTrue(out_path.exists())

    def test_end_of_run_out_npz_still_written_with_series(self):
        with tempfile.TemporaryDirectory() as d:
            # log_every>0 records the scalar series during the loop; the post-loop
            # flush always writes the aggregate out.npz with the series included.
            out_path, _, _ = _drive(d, steps=10, write_every=5, log_every=5)
            self.assertTrue(out_path.exists())
            with np.load(out_path) as npz:
                keys = list(npz.keys())
            self.assertIn("series_step", keys)


if __name__ == "__main__":
    unittest.main()
