"""pybind11 bindings smoke test for the high-order ideal-MHD module.

Mirrors ``tests/python/test_magnetostatics_bindings.py`` and the registry-
introspection contract of ``bindings/python/bind_pic.cpp``:

* uses stdlib ``unittest`` (pytest is not installed on every node),
* the seven ``_core.mhd.registered_*`` accessors return sorted ``list[str]``,
  exactly like ``_core.pic.registered_particle_boundaries()`` etc.,
* device-touching construction/step is guarded by the same ``QUASAR_HAS_HIP_RUNTIME``
  skip and an ``(ImportError, RuntimeError)`` fallback skip the PIC tests use.

Until ``_core.mhd`` exists (and the build tree is refreshed) the top-level
``from quasar import _core`` succeeds but ``_core.mhd`` is absent, so every test
fails by a clean AttributeError -- the intended RED state.
"""

import os
import unittest

import numpy as np

from quasar import _core


def has_hip_runtime() -> bool:
    """The CTest wrapper sets QUASAR_HAS_HIP_RUNTIME=1 when a device was detected
    at configure time; device-touching tests SKIP otherwise (mirrors PIC)."""
    return os.environ.get("QUASAR_HAS_HIP_RUNTIME", "0") == "1"


class MhdSubmodulePresenceTests(unittest.TestCase):

    def test_mhd_submodule_exists(self):
        self.assertTrue(hasattr(_core, "mhd"),
                        "_core.mhd submodule is not exposed")

    def test_solver_and_config_types_exposed(self):
        self.assertTrue(hasattr(_core.mhd, "MhdConfig"))
        self.assertTrue(hasattr(_core.mhd, "MhdSolver2D"))


class RegistryIntrospectionTests(unittest.TestCase):
    """The seven registry accessors each return a non-empty list[str]."""

    REGISTRY_ACCESSORS = (
        "registered_riemann_solvers",
        "registered_reconstructions",
        "registered_integrators",
        "registered_ct_schemes",
        "registered_positivity_limiters",
        "registered_mhd_fluid_boundaries",
        "registered_mhd_field_boundaries",
    )

    def test_all_accessors_present(self):
        for name in self.REGISTRY_ACCESSORS:
            with self.subTest(accessor=name):
                self.assertTrue(hasattr(_core.mhd, name),
                                f"_core.mhd.{name} missing")

    def test_accessors_return_nonempty_string_lists(self):
        for name in self.REGISTRY_ACCESSORS:
            with self.subTest(accessor=name):
                names = getattr(_core.mhd, name)()
                self.assertIsInstance(names, list)
                self.assertTrue(names, f"{name}() is empty")
                self.assertTrue(all(isinstance(n, str) for n in names))

    def test_expected_riemann_solver_registered(self):
        self.assertIn("hlld", _core.mhd.registered_riemann_solvers())

    def test_expected_reconstructions_registered(self):
        recon = set(_core.mhd.registered_reconstructions())
        for name in ("mp7", "mp5", "muscl_minmod"):
            with self.subTest(name=name):
                self.assertIn(name, recon)

    def test_expected_integrator_registered(self):
        self.assertIn("ssprk3", _core.mhd.registered_integrators())

    def test_expected_ct_scheme_registered(self):
        self.assertIn("fd_ct_christlieb", _core.mhd.registered_ct_schemes())

    def test_expected_positivity_limiter_registered(self):
        self.assertIn("troubled_cell", _core.mhd.registered_positivity_limiters())

    def test_expected_fluid_boundaries_registered(self):
        fluid = set(_core.mhd.registered_mhd_fluid_boundaries())
        for name in ("periodic", "outflow", "wall"):
            with self.subTest(name=name):
                self.assertIn(name, fluid)
        self.assertNotIn("reflecting", fluid,
                         "fluid boundary 'reflecting' was renamed to 'wall' "
                         "and must no longer be registered")

    def test_expected_field_boundaries_registered(self):
        field = set(_core.mhd.registered_mhd_field_boundaries())
        for name in ("periodic", "outflow", "wall"):
            with self.subTest(name=name):
                self.assertIn(name, field)
        self.assertNotIn("reflecting", field,
                         "field boundary 'reflecting' was renamed to 'wall' "
                         "and must no longer be registered")


# State components seeded/read through the per-component seed_state /
# state_component_to_host seam (mirrors PIC's per-component seed_field).
STATE_COMPONENTS = ("rho", "mx", "my", "mz", "energy", "bx", "by", "bz")
# Uniform values for a trivial valid state: unit density/energy, zero momentum
# and B (a quiescent magnetized vacuum the solver can advance).
UNIFORM_STATE = {"rho": 1.0, "mx": 0.0, "my": 0.0, "mz": 0.0,
                 "energy": 1.0, "bx": 0.0, "by": 0.0, "bz": 0.0}


@unittest.skipUnless(has_hip_runtime(), "no HIP runtime visible")
class MhdSolverConstructionTests(unittest.TestCase):
    """Construct an MhdConfig + MhdSolver2D on a tiny grid, seed a uniform state
    via the per-component seed_state seam, step once, and read a component back as
    a NumPy array. Mirrors the PIC EmPicConfig/EmPic2D3V construction style in
    bind_pic.cpp."""

    NX = 8
    NY = 8

    def _make_solver(self):
        mhd = _core.mhd
        cfg = mhd.MhdConfig()
        # Grid2D is the shared core grid type used by both PIC and MHD configs.
        # Pass nghost=0 so the MhdSolver2D ctor auto-sizes the ghost halo to the
        # reconstruction scheme's requirement (mp7 needs 4). A positive nghost
        # smaller than that halo is a hard error (the ctor throws), so leaving the
        # Grid2D binding default of 1 would be rejected; nghost=0 keeps this test
        # scheme-agnostic. Read solver.grid().nghost back for the actual halo.
        cfg.grid = _core.pic.Grid2D(nx=self.NX, ny=self.NY, lx=1.0, ly=1.0,
                                    nghost=0)
        cfg.gamma = 1.6666667
        cfg.reconstruction = "mp7"
        cfg.riemann = "hlld"
        cfg.integrator = "ssprk3"
        cfg.ct = "fd_ct_christlieb"
        cfg.positivity = "troubled_cell"
        return mhd.MhdSolver2D(cfg)

    def _build_or_skip(self):
        try:
            return self._make_solver()
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"solver build unavailable (no _core/device): {exc}")

    @staticmethod
    def _storage_size(solver) -> int:
        # The Grid2D binding exposes only nx/ny/nghost (no storage_size), and the
        # ctor auto-sizes nghost from nghost=0, so read the actual halo back from
        # solver.grid() and compute the ghost-padded storage size.
        g = solver.grid()
        return (g.nx + 2 * g.nghost) * (g.ny + 2 * g.nghost)

    def _seed_uniform(self, solver, n: int) -> None:
        for comp, value in UNIFORM_STATE.items():
            solver.seed_state(comp, np.full(n, value, dtype=float))

    def test_seed_uniform_then_step_and_read_component(self):
        solver = self._build_or_skip()
        n = self._storage_size(solver)
        # Seed a uniform initial state via the per-component seed_state seam.
        self._seed_uniform(solver, n)
        solver.step(1.0e-3)
        rho = solver.state_component_to_host("rho")
        self.assertIsInstance(rho, np.ndarray)
        self.assertEqual(rho.shape, (n,))
        self.assertFalse(np.isnan(rho).any())

    def test_state_component_matches_storage_size_for_all_fields(self):
        solver = self._build_or_skip()
        n = self._storage_size(solver)
        self._seed_uniform(solver, n)
        for comp in STATE_COMPONENTS:
            with self.subTest(component=comp):
                arr = solver.state_component_to_host(comp)
                self.assertEqual(arr.shape, (n,))


if __name__ == "__main__":
    unittest.main()
