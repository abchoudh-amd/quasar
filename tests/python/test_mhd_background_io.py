"""Deck I/O tests for the optional static background-field block (B = B0 + b).

Pins the OBSERVABLE contract of the ``background_field:`` top-level deck block and
the new ``quasar.mhd.io`` API it drives:

* ``BackgroundConfig`` dataclass (``enabled``/``profile``/``bx0``/``by0``/``bz0``/
  ``params``/``file``) with a disabled default,
* ``parse(data)`` sets ``deck.background`` from ``data["background_field"]`` (an
  absent block => default, disabled),
* ``build_background_field(deck, nghost)`` returns ``None`` when disabled, else
  ``{"b0x", "b0y", "b0z"}`` 1-D host buffers in the solver storage layout
  (length ``(nx+2g)*(ny+2g)``), and in ALL modes enforces the discrete face
  div-free contract on ``(b0x, b0y)`` by raising :class:`ValueError`.

Scheme-name validation and analytic sampling are driven by the live C++ registry
through ``_core.mhd.registered_mhd_background_profiles()`` and
``sample_mhd_background_profile()``.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from quasar import _core
from quasar.mhd.io import (
    BackgroundConfig,
    BoundaryConfig,
    Domain,
    Initial,
    MhdDeck,
    Numerics,
    Time,
    _padded_grids,
    build_background_field,
    parse,
)
from quasar.mhd.numerics import (
    DISCRETE_SOLENOIDAL_TOLERANCE,
    _scaled_quotient_sum,
    background_curl_linf,
    background_divergence_linf,
    background_divergence_relative_linf,
    validate_background_boundary_compatibility,
)

# Ghost-cell width used to size the storage buffers in these tests. The solver's
# real nghost is an implementation detail; the contract only requires that
# build_background_field's buffers match (nx + 2g) * (ny + 2g) for the g passed.
NGHOST = 3


def _domain(**overrides) -> Domain:
    base = dict(nx=16, ny=16, lx_m=1.0, ly_m=1.0)
    base.update(overrides)
    return Domain(**base)


def _numerics(**overrides) -> Numerics:
    base = dict(
        gamma=1.6666667,
        reconstruction="mp7",
        riemann="hlld",
        integrator="ssprk3",
        ct="fd_ct_christlieb",
        positivity="troubled_cell",
        rho_floor=1.0e-8,
        p_floor=1.0e-9,
        cfl=0.4,
    )
    base.update(overrides)
    return Numerics(**base)


def _deck(**overrides) -> MhdDeck:
    base = dict(
        domain=_domain(),
        numerics=_numerics(),
        initial=Initial(type="orszag_tang"),
        time=Time(dt_s="auto", steps=5),
    )
    base.update(overrides)
    return MhdDeck(**base)


def _storage_size(domain: Domain, nghost: int) -> int:
    return (domain.nx + 2 * nghost) * (domain.ny + 2 * nghost)


def _base_data(**extra) -> dict:
    """A minimal, valid MHD deck mapping (for ``parse``); extra keys merge in."""
    data = {
        "domain": {"nx": 16, "ny": 16, "lx_m": 1.0, "ly_m": 1.0},
        "numerics": {"gamma": 1.6666667},
        "initial": {"type": "orszag_tang"},
        "time": {"dt_s": "auto", "steps": 5},
    }
    data.update(extra)
    return data


class BackgroundConfigDefaultsTests(unittest.TestCase):
    """An absent / disabled block behaves exactly like today's no-B0 run."""

    def test_absent_block_disabled_and_builds_none(self):
        deck = parse(_base_data())  # no background_field key
        self.assertIsInstance(deck.background, BackgroundConfig)
        self.assertFalse(deck.background.enabled)
        self.assertIsNone(build_background_field(deck, NGHOST))

    def test_enabled_false_block_same_as_absent(self):
        deck = parse(_base_data(background_field={"enabled": False, "bz0": 5.0}))
        self.assertFalse(deck.background.enabled)
        self.assertIsNone(build_background_field(deck, NGHOST))

    def test_background_config_dataclass_defaults(self):
        cfg = BackgroundConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.profile, "uniform")
        self.assertEqual(cfg.bx0, 0.0)
        self.assertEqual(cfg.by0, 0.0)
        self.assertEqual(cfg.bz0, 0.0)
        self.assertEqual(cfg.params, {})
        self.assertIsNone(cfg.file)


class UniformBackgroundParseTests(unittest.TestCase):
    """A valid uniform block parses and builds constant, div-free buffers."""

    def test_uniform_block_parses_into_config(self):
        deck = parse(_base_data(background_field={
            "enabled": True,
            "profile": "uniform",
            "bz0": 1.0,
        }))
        bg = deck.background
        self.assertIsInstance(bg, BackgroundConfig)
        self.assertTrue(bg.enabled)
        self.assertEqual(bg.profile, "uniform")
        self.assertEqual(bg.bx0, 0.0)
        self.assertEqual(bg.by0, 0.0)
        self.assertEqual(bg.bz0, 1.0)

    def test_uniform_build_returns_three_storage_buffers(self):
        domain = _domain()
        deck = _deck(domain=domain, background=BackgroundConfig(
            enabled=True, profile="uniform", bz0=1.0))
        bufs = build_background_field(deck, NGHOST)
        self.assertIsNotNone(bufs)
        self.assertEqual(set(bufs), {"b0x", "b0y", "b0z"})
        n = _storage_size(domain, NGHOST)
        for comp in ("b0x", "b0y", "b0z"):
            with self.subTest(component=comp):
                buf = np.asarray(bufs[comp])
                self.assertEqual(buf.ndim, 1)
                self.assertEqual(buf.shape[0], n)

    def test_uniform_build_fills_constants(self):
        deck = _deck(background=BackgroundConfig(
            enabled=True, profile="uniform", bx0=0.0, by0=0.0, bz0=1.0))
        bufs = build_background_field(deck, NGHOST)
        self.assertTrue(np.all(np.asarray(bufs["b0z"]) == 1.0))
        self.assertTrue(np.all(np.asarray(bufs["b0x"]) == 0.0))
        self.assertTrue(np.all(np.asarray(bufs["b0y"]) == 0.0))


class BackgroundValidationTests(unittest.TestCase):
    """Malformed background blocks are rejected with ValueError."""

    def test_uniform_is_registered_profile(self):
        registered = set(_core.mhd.registered_mhd_background_profiles())
        self.assertIn("uniform", registered)

    def test_enabled_requires_a_real_boolean(self):
        for bad in ("false", 0, 1):
            with self.subTest(enabled=bad):
                with self.assertRaises(ValueError):
                    parse(_base_data(background_field={"enabled": bad}))

    def test_unknown_profile_rejected(self):
        registered = set(_core.mhd.registered_mhd_background_profiles())
        bogus = "definitely_not_a_background_profile"
        self.assertNotIn(bogus, registered)
        with self.assertRaises(ValueError):
            parse(_base_data(background_field={
                "enabled": True, "profile": bogus, "bz0": 1.0}))

    def test_nonfinite_uniform_component_rejected(self):
        for bad in (float("inf"), float("nan")):
            with self.subTest(bx0=bad):
                with self.assertRaises(ValueError):
                    parse(_base_data(background_field={
                        "enabled": True, "profile": "uniform", "bx0": bad}))

    def test_enabled_without_source_rejected(self):
        # enabled but neither a usable analytic spec nor a readable file: a
        # non-existent file path with an otherwise empty spec must be rejected.
        with self.assertRaises(ValueError):
            parse(_base_data(background_field={
                "enabled": True,
                "file": "/nonexistent/definitely/not/here/b0.npz",
            }))

    def test_cylindrical_toroidal_background_is_supported(self):
        deck = _deck(
            geometry="cylindrical",
            numerics=_numerics(reconstruction="muscl_minmod"),
            domain=_domain(origin_x_m=0.5),
            boundary=BoundaryConfig(
                fluid=("wall", "wall", "periodic", "periodic"),
                field=("wall", "wall", "periodic", "periodic")),
            background=BackgroundConfig(
                enabled=True, profile="uniform", bz0=0.1))
        deck.validate()
        bg = build_background_field(deck, NGHOST)
        self.assertTrue(np.all(np.asarray(bg["b0z"]) == 0.1))

    def test_linear_vacuum_profile_samples_staggered_mesh(self):
        domain = _domain(nx=8, ny=6, lx_m=2.0, ly_m=3.0,
                         origin_x_m=-0.25, origin_y_m=0.5)
        outflow = BoundaryConfig(
            fluid=("outflow",) * 4, field=("outflow",) * 4)
        deck = _deck(domain=domain, boundary=outflow,
                     background=BackgroundConfig(
                         enabled=True, profile="linear_vacuum",
                         params={"gradient": 1.25, "shear": -0.4}))
        deck.validate()
        bg = build_background_field(deck, NGHOST)
        shape = (domain.ny + 2 * NGHOST, domain.nx + 2 * NGHOST)
        bx = np.asarray(bg["b0x"]).reshape(shape)
        by = np.asarray(bg["b0y"]).reshape(shape)
        xc, yc, xf, yf, _dx, _dy = _padded_grids(domain, NGHOST)
        np.testing.assert_allclose(bx, 1.25 * xf - 0.4 * yc,
                                   rtol=0.0, atol=2.0e-15)
        np.testing.assert_allclose(by, -0.4 * xc - 1.25 * yf,
                                   rtol=0.0, atol=2.0e-15)

    def test_unknown_analytic_profile_parameter_is_rejected(self):
        deck = _deck(background=BackgroundConfig(
            enabled=True, profile="linear_vacuum", params={"bogus": 1.0}))
        deck.validate()
        with self.assertRaisesRegex(ValueError, "unknown parameter"):
            build_background_field(deck, NGHOST)

    def test_divergence_reduction_cancels_extreme_finite_terms(self):
        g = 1
        shape = (3, 3)
        huge = np.finfo(np.float64).max
        bx = np.zeros(shape, dtype=np.float64)
        by = np.zeros(shape, dtype=np.float64)
        bx[g, g] = -huge
        bx[g, g + 1] = huge
        by[g, g] = huge
        by[g + 1, g] = -huge

        # Each raw face difference overflows, but after division the two finite
        # directional derivatives cancel exactly. The diagnostic must evaluate
        # the mathematical expression rather than reject an intermediate inf.
        self.assertEqual(
            background_divergence_linf(
                bx, by, 1, 1, g, 2.0, 2.0, geometry="cartesian"),
            0.0)

    def test_scaled_reduction_pairs_opposite_dominant_terms(self):
        huge = np.finfo(np.float64).max
        terms = tuple(np.array([[value]], dtype=np.float64) for value in (
            huge, huge, -huge, -huge, 3.0))

        # A same-sign-first tree can leave an ulp(huge) residue after the four
        # dominant terms cancel. The common-exponent reducer must retain the
        # separately representable finite survivor.
        result = _scaled_quotient_sum(terms, 1.0)
        self.assertEqual(float(result[0, 0]), 3.0)

    def test_divergence_reduction_rejects_true_overflow(self):
        g = 1
        shape = (3, 3)
        huge = np.finfo(np.float64).max
        bx = np.zeros(shape, dtype=np.float64)
        by = np.zeros(shape, dtype=np.float64)
        bx[g, g] = -huge
        bx[g, g + 1] = huge

        with self.assertRaisesRegex(ValueError, "not representable"):
            background_divergence_linf(
                bx, by, 1, 1, g, 1.0, 1.0, geometry="cartesian")

    def test_staggered_diagnostics_require_a_face_and_corner_halo(self):
        bx = np.zeros(4, dtype=np.float64)
        by = np.zeros(4, dtype=np.float64)
        diagnostics = (background_divergence_linf, background_curl_linf)
        for diagnostic in diagnostics:
            with self.subTest(diagnostic=diagnostic.__name__):
                with self.assertRaisesRegex(ValueError, "nghost must be at least 1"):
                    diagnostic(bx, by, 2, 2, 0, 1.0, 1.0)

    def test_relative_divergence_rejects_resolved_defect(self):
        g = 1
        bx = np.zeros((3, 3), dtype=np.float64)
        by = np.zeros_like(bx)
        bx[g, g + 1] = 1.0
        by[g + 1, g] = -1.0 + 1.0e-10
        defect = background_divergence_relative_linf(
            bx, by, 1, 1, g, 1.0, 1.0)
        self.assertGreater(defect, DISCRETE_SOLENOIDAL_TOLERANCE)

    def test_relative_divergence_accepts_roundoff_cancellation(self):
        g = 1
        bx = np.zeros((3, 3), dtype=np.float64)
        by = np.zeros_like(bx)
        bx[g, g + 1] = 1.0
        by[g + 1, g] = -1.0 + np.finfo(np.float64).eps
        defect = background_divergence_relative_linf(
            bx, by, 1, 1, g, 1.0, 1.0)
        self.assertGreater(defect, 0.0)
        self.assertLessEqual(defect, DISCRETE_SOLENOIDAL_TOLERANCE)

    def test_periodic_background_seam_must_wrap(self):
        nx = ny = 2
        g = 1
        shape = (ny + 2 * g, nx + 2 * g)
        bx = np.zeros(shape)
        by = np.zeros(shape)
        bz = np.zeros(shape)
        # x=-1 is the low periodic target for layer one; x=nx-1 is its source.
        bz[g, g - 1] = 1.0
        with self.assertRaisesRegex(ValueError, "periodic x boundary"):
            validate_background_boundary_compatibility(
                bx, by, bz, nx, ny, g, ("periodic",) * 4)

    def test_wall_background_requires_zero_normal_and_parity(self):
        nx = ny = 2
        g = 1
        shape = (ny + 2 * g, nx + 2 * g)
        boundaries = ("wall", "wall", "outflow", "outflow")

        bx = np.zeros(shape)
        by = np.zeros(shape)
        bz = np.zeros(shape)
        bx[g, g] = 1.0  # Bx on the low x-wall face must be exactly zero.
        with self.assertRaisesRegex(ValueError, "normal constraint"):
            validate_background_boundary_compatibility(
                bx, by, bz, nx, ny, g, boundaries)

        bx.fill(0.0)
        by[g, g] = 1.0       # even tangential source at x=0
        by[g, g - 1] = -1.0  # invalid odd low-wall ghost
        with self.assertRaisesRegex(ValueError, "x-wall parity"):
            validate_background_boundary_compatibility(
                bx, by, bz, nx, ny, g, boundaries)

    def test_cylindrical_axis_requires_odd_toroidal_parity(self):
        nx = ny = 2
        g = 1
        shape = (ny + 2 * g, nx + 2 * g)
        bx = np.zeros(shape)
        by = np.zeros(shape)
        bz = np.zeros(shape)
        boundaries = ("axis", "outflow", "outflow", "outflow")
        bz[g, g] = 1.0
        bz[g, g - 1] = -1.0
        validate_background_boundary_compatibility(
            bx, by, bz, nx, ny, g, boundaries)

        bz[g, g - 1] = 1.0
        with self.assertRaisesRegex(ValueError, "axis parity"):
            validate_background_boundary_compatibility(
                bx, by, bz, nx, ny, g, boundaries)


class BackgroundFileModeTests(unittest.TestCase):
    """A B0 supplied via npz file round-trips against the analytic-uniform path
    and is subject to the same div-free contract."""

    def _write_npz(self, path: Path, b0x, b0y, b0z) -> None:
        np.savez(path, b0x=b0x, b0y=b0y, b0z=b0z)

    def test_file_matches_analytic_uniform(self):
        domain = _domain()
        n = _storage_size(domain, NGHOST)

        analytic = _deck(domain=domain, background=BackgroundConfig(
            enabled=True, profile="uniform", bx0=0.0, by0=0.0, bz0=1.0))
        ref = build_background_field(analytic, NGHOST)

        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "b0.npz"
            # Same uniform field, written as flat storage-sized arrays.
            self._write_npz(
                npz,
                np.zeros(n, dtype=np.float64),
                np.zeros(n, dtype=np.float64),
                np.ones(n, dtype=np.float64),
            )
            file_deck = _deck(domain=domain, background=BackgroundConfig(
                enabled=True, file=str(npz)))
            got = build_background_field(file_deck, NGHOST)

        self.assertIsNotNone(got)
        for comp in ("b0x", "b0y", "b0z"):
            with self.subTest(component=comp):
                self.assertTrue(np.allclose(
                    np.asarray(got[comp]), np.asarray(ref[comp]),
                    rtol=0.0, atol=1.0e-12))

    def test_file_2d_shaped_arrays_accepted(self):
        domain = _domain()
        g = NGHOST
        height = domain.ny + 2 * g
        pitch = domain.nx + 2 * g
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "b0.npz"
            self._write_npz(
                npz,
                np.zeros((height, pitch), dtype=np.float64),
                np.zeros((height, pitch), dtype=np.float64),
                np.full((height, pitch), 2.0, dtype=np.float64),
            )
            deck = _deck(domain=domain, background=BackgroundConfig(
                enabled=True, file=str(npz)))
            bufs = build_background_field(deck, g)
        self.assertEqual(np.asarray(bufs["b0z"]).shape[0], height * pitch)
        self.assertTrue(np.all(np.asarray(bufs["b0z"]) == 2.0))

    def test_file_wrong_shape_rejected(self):
        domain = _domain()
        n = _storage_size(domain, NGHOST)
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "b0.npz"
            # Arrays sized to the wrong storage (too small).
            self._write_npz(
                npz,
                np.zeros(n // 2, dtype=np.float64),
                np.zeros(n // 2, dtype=np.float64),
                np.zeros(n // 2, dtype=np.float64),
            )
            deck = _deck(domain=domain, background=BackgroundConfig(
                enabled=True, file=str(npz)))
            with self.assertRaises(ValueError):
                build_background_field(deck, NGHOST)

    def test_file_not_divergence_free_rejected(self):
        domain = _domain()
        g = NGHOST
        height = domain.ny + 2 * g
        pitch = domain.nx + 2 * g
        rng = np.random.default_rng(12345)
        # A spatially-random in-plane (b0x, b0y) field is generically NOT
        # discretely divergence-free, so build_background_field must raise.
        b0x = rng.standard_normal((height, pitch))
        b0y = rng.standard_normal((height, pitch))
        b0z = np.zeros((height, pitch), dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "b0.npz"
            self._write_npz(npz, b0x, b0y, b0z)
            deck = _deck(domain=domain, background=BackgroundConfig(
                enabled=True, file=str(npz)))
            with self.assertRaises(ValueError):
                build_background_field(deck, g)


class CylindricalVectorPotentialTests(unittest.TestCase):
    """The annular A_phi path is mimetic and vacuum projection is explicit."""

    @staticmethod
    def _boundary() -> BoundaryConfig:
        return BoundaryConfig(
            fluid=("outflow",) * 4,
            field=("outflow",) * 4)

    @staticmethod
    def _write_a(path: Path, domain: Domain, g: int, values) -> None:
        height = domain.ny + 2 * g
        pitch = domain.nx + 2 * g
        archive = np.zeros((height + 1, 1, pitch + 1, 3), dtype=np.float64)
        archive[:, 0, :, 1] = values
        np.savez(path, A_xyz_grid=archive)

    def _deck_for(self, domain: Domain, path: Path, *, project: bool) -> MhdDeck:
        return _deck(
            geometry="cylindrical", domain=domain, boundary=self._boundary(),
            numerics=_numerics(reconstruction="muscl_minmod"),
            background=BackgroundConfig(
                enabled=True, a_file=str(path),
                params={"vacuum_project": project}))

    def test_annular_a_file_divergence_telescopes(self):
        domain = _domain(nx=12, ny=10, lx_m=0.6, ly_m=0.5,
                         origin_x_m=0.7, origin_y_m=-0.25)
        g = NGHOST
        dx = domain.lx_m / domain.nx
        height = domain.ny + 2 * g
        pitch = domain.nx + 2 * g
        r = domain.origin_x_m + (np.arange(pitch + 1) - g) * dx
        # A_phi = C r gives a constant axial field and is exactly vacuum under
        # the annular operator, so no projection is needed.
        a_phi = np.broadcast_to(0.25 * r[None, :], (height + 1, pitch + 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.npz"
            self._write_a(path, domain, g, a_phi)
            deck = self._deck_for(domain, path, project=False)
            deck.validate()
            bg = build_background_field(deck, g)
        divb = background_divergence_linf(
            bg["b0x"], bg["b0y"], domain.nx, domain.ny, g, dx,
            domain.ly_m / domain.ny, geometry="cylindrical",
            origin_x=domain.origin_x_m)
        self.assertLess(divb, 1.0e-12)

    def test_current_carrying_a_file_is_supported_without_projection(self):
        domain = _domain(nx=12, ny=12, lx_m=0.6, ly_m=0.6,
                         origin_x_m=0.7, origin_y_m=-0.3)
        g = NGHOST
        height = domain.ny + 2 * g
        pitch = domain.nx + 2 * g
        a_phi = np.ones((height + 1, pitch + 1), dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.npz"
            self._write_a(path, domain, g, a_phi)
            deck = self._deck_for(domain, path, project=False)
            deck.validate()
            bg = build_background_field(deck, g)
        self.assertTrue(np.all(np.isfinite(bg["b0x"])))
        self.assertTrue(np.all(np.isfinite(bg["b0y"])))

    def test_vacuum_projection_meets_curl_and_divergence_tolerances(self):
        domain = _domain(nx=24, ny=20, lx_m=0.6, ly_m=0.5,
                         origin_x_m=0.7, origin_y_m=-0.25)
        g = NGHOST
        dx = domain.lx_m / domain.nx
        dy = domain.ly_m / domain.ny
        height = domain.ny + 2 * g
        pitch = domain.nx + 2 * g
        r = domain.origin_x_m + (np.arange(pitch + 1) - g) * dx
        z = domain.origin_y_m + (np.arange(height + 1) - g) * dy
        rr, zz = np.meshgrid(r, z)
        # Axisymmetric dipole A_phi is continuum-vacuum on this r>0 domain. Raw
        # sampling leaves an O(h^2) discrete curl; projection removes that defect.
        a_phi = rr / (rr * rr + zz * zz) ** 1.5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.npz"
            self._write_a(path, domain, g, a_phi)
            raw = self._deck_for(domain, path, project=False)
            raw.validate()
            raw_bg = build_background_field(raw, g)

            projected = self._deck_for(domain, path, project=True)
            projected.validate()
            bg = build_background_field(projected, g)

        divb = background_divergence_linf(
            bg["b0x"], bg["b0y"], domain.nx, domain.ny, g, dx, dy,
            geometry="cylindrical", origin_x=domain.origin_x_m)
        curlb = background_curl_linf(
            bg["b0x"], bg["b0y"], domain.nx, domain.ny, g, dx, dy)
        raw_curlb = background_curl_linf(
            raw_bg["b0x"], raw_bg["b0y"], domain.nx, domain.ny, g, dx, dy)
        scale = max(1.0, (np.max(np.abs(bg["b0x"])) +
                          np.max(np.abs(bg["b0y"]))) / min(dx, dy))
        self.assertLessEqual(divb, 1.0e-9 * scale)
        self.assertLessEqual(curlb, 1.0e-9 * scale)
        self.assertLess(curlb, raw_curlb)


if __name__ == "__main__":
    unittest.main()
