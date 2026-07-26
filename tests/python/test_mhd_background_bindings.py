"""pybind11 bindings smoke test for the MHD static background-field split.

Pins the ``bindings/python/bind_mhd.cpp`` surface added for the
``B = B0 + b`` background-field feature (see the "Python bindings" block of
``plans/mhd-onesided-bc-and-background-field-build-plan.md``):

* ``_core.mhd.MhdBackgroundSpec`` -- default-constructible POD with read/write
  ``enabled`` (bool), ``profile`` (str), ``bx0``/``by0``/``bz0`` and
  ``profile_scale`` (float),
* ``_core.mhd.MhdConfig.background`` -- a read/write ``MhdBackgroundSpec`` that
  round-trips,
* ``_core.mhd.MhdSolver2D.seed_background(component, values)`` for
  ``component in {"b0x","b0y","b0z"}`` and a storage-sized float sequence,
* ``_core.mhd.MhdSolver2D.has_background()`` mirroring ``background.enabled``,
* native background-profile registry and array sampler, including the built-in
  ``"uniform"`` and ``"linear_vacuum"`` profiles.

Mirrors ``tests/python/test_mhd_bindings.py``:

* uses stdlib ``unittest`` (pytest is not installed on every node),
* device-touching construction is guarded by the same ``QUASAR_HAS_HIP_RUNTIME``
  skip and an ``(ImportError, RuntimeError)`` build fallback,
* the tiny-grid construction reuses the identical ``MhdConfig``/``MhdSolver2D``
  idiom (``_core.pic.Grid2D(... nghost=0)``, auto-sized ghost halo, storage
  size read back from ``solver.grid()``).

"""

import os
import unittest

import numpy as np

from quasar import _core


def has_hip_runtime() -> bool:
    """The CTest wrapper sets QUASAR_HAS_HIP_RUNTIME=1 when a device was detected
    at configure time; device-touching tests SKIP otherwise (mirrors PIC/MHD)."""
    return os.environ.get("QUASAR_HAS_HIP_RUNTIME", "0") == "1"


class MhdBackgroundSpecTests(unittest.TestCase):
    """The ``MhdBackgroundSpec`` POD: default-constructible with documented
    defaults, every attribute read/write."""

    def test_spec_type_exposed(self):
        self.assertTrue(hasattr(_core.mhd, "MhdBackgroundSpec"),
                        "_core.mhd.MhdBackgroundSpec missing")

    def test_default_construct_and_defaults(self):
        spec = _core.mhd.MhdBackgroundSpec()
        self.assertIs(spec.enabled, False)
        self.assertEqual(spec.profile, "uniform")
        self.assertEqual(spec.bx0, 0.0)
        self.assertEqual(spec.by0, 0.0)
        self.assertEqual(spec.bz0, 0.0)
        self.assertEqual(spec.profile_scale, 1.0)
        self.assertEqual(spec.params, {})

    def test_attributes_are_read_write(self):
        spec = _core.mhd.MhdBackgroundSpec()
        spec.enabled = True
        spec.profile = "uniform"
        spec.bx0 = 0.25
        spec.by0 = -0.5
        spec.bz0 = 1.0
        spec.profile_scale = 2.5
        spec.params = {"gradient": 2.0, "shear": -0.25}
        self.assertIs(spec.enabled, True)
        self.assertEqual(spec.profile, "uniform")
        self.assertEqual(spec.bx0, 0.25)
        self.assertEqual(spec.by0, -0.5)
        self.assertEqual(spec.bz0, 1.0)
        self.assertEqual(spec.profile_scale, 2.5)
        self.assertEqual(spec.params, {"gradient": 2.0, "shear": -0.25})


class MhdConfigBackgroundRoundTripTests(unittest.TestCase):
    """``MhdConfig.background`` is a read/write ``MhdBackgroundSpec`` that
    round-trips the assigned values."""

    def test_config_has_background_attribute(self):
        cfg = _core.mhd.MhdConfig()
        self.assertTrue(hasattr(cfg, "background"),
                        "MhdConfig.background missing")

    def test_background_round_trips(self):
        cfg = _core.mhd.MhdConfig()
        spec = _core.mhd.MhdBackgroundSpec()
        spec.enabled = True
        spec.bz0 = 1.0
        spec.profile = "uniform"
        spec.profile_scale = 3.0
        spec.params = {"bx0": 99.0}
        cfg.background = spec

        got = cfg.background
        self.assertIs(got.enabled, True)
        self.assertEqual(got.bz0, 1.0)
        self.assertEqual(got.profile, "uniform")
        self.assertEqual(got.profile_scale, 3.0)
        self.assertEqual(got.params, {"bx0": 99.0})
        # untouched components keep their defaults
        self.assertEqual(got.bx0, 0.0)
        self.assertEqual(got.by0, 0.0)


class MhdBackgroundProfileRegistryTests(unittest.TestCase):
    """``registered_mhd_background_profiles()`` returns a sequence of strings
    that includes the built-in analytic profiles."""

    def test_accessor_present(self):
        self.assertTrue(hasattr(_core.mhd, "registered_mhd_background_profiles"),
                        "_core.mhd.registered_mhd_background_profiles missing")

    def test_lists_uniform(self):
        names = _core.mhd.registered_mhd_background_profiles()
        self.assertTrue(all(isinstance(n, str) for n in names))
        self.assertIn("uniform", names)
        self.assertIn("linear_vacuum", names)

    def test_linear_profile_sampler_uses_parameters_and_array_shape(self):
        x = np.array([[0.0, 1.0], [-2.0, 0.5]])
        y = np.array([[3.0, -1.0], [0.25, 2.0]])
        params = {"gradient": 2.0, "shear": -0.5}
        bx = _core.mhd.sample_mhd_background_profile(
            "linear_vacuum", 0, x, y, params)
        by = _core.mhd.sample_mhd_background_profile(
            "linear_vacuum", 1, x, y, params)
        np.testing.assert_array_equal(bx, 2.0 * x - 0.5 * y)
        np.testing.assert_array_equal(by, -0.5 * x - 2.0 * y)

    def test_sampler_rejects_nonfinite_coordinates(self):
        with self.assertRaises(ValueError):
            _core.mhd.sample_mhd_background_profile(
                "uniform", 0, np.array([np.nan]), np.array([0.0]), {})


# State components seeded for a trivial valid (quiescent magnetized vacuum)
# state, exactly as in test_mhd_bindings.py, so the solver can be constructed
# and queried.
UNIFORM_STATE = {"rho": 1.0, "mx": 0.0, "my": 0.0, "mz": 0.0,
                 "energy": 1.0, "bx": 0.0, "by": 0.0, "bz": 0.0}


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class MhdSolverBackgroundTests(unittest.TestCase):
    """Construct an MhdConfig + MhdSolver2D on a tiny grid (the exact idiom from
    test_mhd_bindings.py) with the background field enabled, then exercise
    ``has_background()`` and ``seed_background``."""

    NX = 8
    NY = 8

    def _make_solver(self, *, background_enabled: bool):
        mhd = _core.mhd
        cfg = mhd.MhdConfig()
        # Grid2D is the shared core grid type used by both PIC and MHD configs.
        # nghost=0 lets the MhdSolver2D ctor auto-size the ghost halo to the
        # reconstruction scheme's requirement (mp7 needs 4); the actual halo is
        # read back from solver.grid().nghost.
        cfg.grid = _core.pic.Grid2D(nx=self.NX, ny=self.NY, lx=1.0, ly=1.0,
                                    nghost=0)
        cfg.gamma = 1.6666667
        cfg.reconstruction = "mp7"
        cfg.riemann = "hlld"
        cfg.integrator = "ssprk3"
        cfg.ct = "fd_ct_christlieb"
        cfg.positivity = "troubled_cell"

        spec = mhd.MhdBackgroundSpec()
        spec.enabled = background_enabled
        spec.profile = "uniform"
        spec.bz0 = 1.0 if background_enabled else 0.0
        cfg.background = spec

        return mhd.MhdSolver2D(cfg)

    def _build_or_skip(self, *, background_enabled: bool):
        try:
            return self._make_solver(background_enabled=background_enabled)
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"solver build unavailable (no _core/device): {exc}")

    @staticmethod
    def _storage_size(solver) -> int:
        # The ctor auto-sizes nghost from nghost=0, so read the actual halo back
        # from solver.grid() and compute the ghost-padded storage size -- the
        # same length seed_state / seed_background buffers must have.
        g = solver.grid()
        return (g.nx + 2 * g.nghost) * (g.ny + 2 * g.nghost)

    def _seed_uniform_state(self, solver, n: int) -> None:
        for comp, value in UNIFORM_STATE.items():
            solver.seed_state(comp, np.full(n, value, dtype=float))

    def test_has_background_true_when_enabled(self):
        solver = self._build_or_skip(background_enabled=True)
        self.assertIs(solver.has_background(), True)

    def test_has_background_false_when_disabled(self):
        solver = self._build_or_skip(background_enabled=False)
        self.assertIs(solver.has_background(), False)

    def test_seed_background_accepts_storage_sized_sequence(self):
        solver = self._build_or_skip(background_enabled=True)
        n = self._storage_size(solver)
        self._seed_uniform_state(solver, n)
        # A storage-sized list and a storage-sized np.array are both accepted
        # without error for an enabled-background solver.
        solver.seed_background("b0z", [1.0] * n)
        solver.seed_background("b0z", np.full(n, 1.0, dtype=float))

    def test_seed_background_accepts_each_component(self):
        solver = self._build_or_skip(background_enabled=True)
        n = self._storage_size(solver)
        self._seed_uniform_state(solver, n)
        for comp in ("b0x", "b0y", "b0z"):
            with self.subTest(component=comp):
                solver.seed_background(comp, np.zeros(n, dtype=float))


if __name__ == "__main__":
    unittest.main()
