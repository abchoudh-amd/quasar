"""Deck I/O tests for the optional static background-field block (B = B0 + b).

Pins the OBSERVABLE contract of the ``background_field:`` top-level deck block and
the new ``quasar.mhd.io`` API it drives:

* ``BackgroundConfig`` dataclass (``enabled``/``profile``/``bx0``/``by0``/``bz0``/
  ``params``/``file``/``a_file``/``conductors``) with a disabled default,
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

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from quasar import _core
from quasar.coil import cli as coil_cli
from quasar.coil import io as coil_io
from quasar.mhd.io import (
    BackgroundConfig,
    BoundaryConfig,
    Domain,
    Initial,
    MhdDeck,
    Numerics,
    Time,
    build_background_field,
    load as load_mhd_deck,
    parse,
)

# Ghost-cell width used to size the storage buffers in these tests. The solver's
# real nghost is an implementation detail; the contract only requires that
# build_background_field's buffers match (nx + 2g) * (ny + 2g) for the g passed.
NGHOST = 3
REPO_ROOT = Path(__file__).resolve().parents[2]



# Test-local staggered diagnostics.
#
# The production divergence and curl sweeps now run on device inside
# build_background_field, which raises before returning if a field is not
# discretely solenoidal. These are the independent references those results are
# checked against, and they are deliberately plain float64: the scaled
# arithmetic the device sweeps carry exists to survive fields at the edge of the
# exponent range, and the accuracy of that machinery is proved against a
# long-double oracle in the C++ suite (tests/unit/physics/mhd/
# test_mhd_background_field.cpp). Reproducing it here would only re-test the
# same code twice.
def _interior_divergence(b0x, b0y, nx, ny, g, dx, dy,
                         geometry="cartesian", origin_x=0.0):
    shape = (ny + 2 * g, nx + 2 * g)
    bx = np.asarray(b0x, dtype=np.float64).reshape(shape)
    by = np.asarray(b0y, dtype=np.float64).reshape(shape)
    lo = np.s_[g:g + ny, g:g + nx]
    hi_x = np.s_[g:g + ny, g + 1:g + nx + 1]
    hi_y = np.s_[g + 1:g + ny + 1, g:g + nx]
    axial = (by[hi_y] - by[lo]) / dy
    if geometry == "cylindrical":
        # Ring-volume divergence: (r_hi Br_hi - r_lo Br_lo) / (r_c dr), written
        # in the factored form the curl construction telescopes against.
        i = np.arange(nx)
        r_lo = origin_x + i * dx
        r_hi = r_lo + dx
        r_c = r_lo + 0.5 * dx
        radial = ((bx[hi_x] - bx[lo]) / dx
                  + (0.5 * bx[hi_x] + 0.5 * bx[lo]) / r_c[None, :])
        _ = (r_lo, r_hi)
    else:
        radial = (bx[hi_x] - bx[lo]) / dx
    return radial + axial


def _divergence_linf(b0x, b0y, nx, ny, g, dx, dy,
                     geometry="cartesian", origin_x=0.0):
    return float(np.max(np.abs(_interior_divergence(
        b0x, b0y, nx, ny, g, dx, dy, geometry, origin_x))))


def _relative_divergence_linf(b0x, b0y, nx, ny, g, dx, dy,
                              geometry="cartesian", origin_x=0.0):
    """||d_r + d_z||_inf / || |d_r| + |d_z| ||_inf, the scale-free form."""
    shape = (ny + 2 * g, nx + 2 * g)
    bx = np.asarray(b0x, dtype=np.float64).reshape(shape)
    by = np.asarray(b0y, dtype=np.float64).reshape(shape)
    lo = np.s_[g:g + ny, g:g + nx]
    hi_x = np.s_[g:g + ny, g + 1:g + nx + 1]
    hi_y = np.s_[g + 1:g + ny + 1, g:g + nx]
    axial = (by[hi_y] - by[lo]) / dy
    if geometry == "cylindrical":
        r_c = origin_x + (np.arange(nx) + 0.5) * dx
        radial = ((bx[hi_x] - bx[lo]) / dx
                  + (0.5 * bx[hi_x] + 0.5 * bx[lo]) / r_c[None, :])
    else:
        radial = (bx[hi_x] - bx[lo]) / dx
    scale = float(np.max(np.abs(radial) + np.abs(axial)))
    if scale == 0.0:
        return 0.0
    return float(np.max(np.abs(radial + axial))) / scale


def _curl_linf(b0x, b0y, nx, ny, g, dx, dy):
    """Max |d By/dx - d Bx/dy| at interior cell corners.

    Bx is x-face/y-centred and By is y-face/x-centred, so backward differences
    collocate both derivatives at the corner, matching the CT staggering.
    """
    shape = (ny + 2 * g, nx + 2 * g)
    bx = np.asarray(b0x, dtype=np.float64).reshape(shape)
    by = np.asarray(b0y, dtype=np.float64).reshape(shape)
    d_by_dx = (by[g:g + ny + 1, g:g + nx + 1] -
               by[g:g + ny + 1, g - 1:g + nx]) / dx
    d_bx_dy = (bx[g:g + ny + 1, g:g + nx + 1] -
               bx[g - 1:g + ny, g:g + nx + 1]) / dy
    return float(np.max(np.abs(d_by_dx - d_bx_dy)))


# The acceptance bound the native sweeps enforce; mirrors
# numerics::kDiscreteSolenoidalTolerance.
DISCRETE_SOLENOIDAL_TOLERANCE = 1024.0 * np.finfo(np.float64).eps

def _padded_meshes(domain, nghost):
    """Test-local staggered coordinate meshes over the full padded storage.

    Deliberately independent of the native builder rather than imported from
    it: this is the reference an analytic profile's samples are checked
    against, so sharing the coordinate code would let one mistake satisfy both
    sides. The native path evaluates the same coordinates with an FMA, which
    differs in the last bit; the comparisons below carry an absolute tolerance
    that covers it.
    """
    g = int(nghost)
    dx = domain.lx_m / domain.nx
    dy = domain.ly_m / domain.ny
    i_int = np.arange(domain.nx + 2 * g) - g
    j_int = np.arange(domain.ny + 2 * g) - g
    ii, jj = np.meshgrid(i_int, j_int)
    return (domain.origin_x_m + (ii + 0.5) * dx,   # cell-centre x
            domain.origin_y_m + (jj + 0.5) * dy,   # cell-centre y
            domain.origin_x_m + ii * dx,           # left-face x
            domain.origin_y_m + jj * dy)           # bottom-face y


def _has_hip_runtime() -> bool:
    return os.environ.get("QUASAR_HAS_HIP_RUNTIME", "0") == "1"


def _circular_loop(
    name: str,
    *,
    center_z_m: float = 0.0,
    current_A: float = 1.0,
    radius_m: float = 0.1,
    n_segments: int = 64,
) -> dict:
    return {
        "name": name,
        "current_A": current_A,
        "geometry": {
            "type": "circular_loop",
            "center_xyz": [0.0, 0.0, center_z_m],
            "axis_xyz": [0.0, 0.0, 1.0],
            "radius_m": radius_m,
            "n_segments": n_segments,
        },
    }


def _helmholtz_pair() -> list[dict]:
    return [
        _circular_loop("lower_loop", center_z_m=-0.05, current_A=1000.0),
        _circular_loop("upper_loop", center_z_m=+0.05, current_A=1000.0),
    ]


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
        deck = parse(_base_data(background_field={
            "enabled": False,
            "profile": "not-registered",
            "bz0": "not-a-number",
            "params": ["not", "a", "mapping"],
        }))
        self.assertFalse(deck.background.enabled)
        self.assertEqual(deck.background, BackgroundConfig())
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
        self.assertIsNone(cfg.conductors)


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

    def test_explicit_file_does_not_validate_ignored_profile(self):
        deck = parse(_base_data(background_field={
            "enabled": True,
            "profile": "stale-profile-that-is-ignored",
            "file": "relative-background.npz",
        }))
        self.assertEqual(deck.background.file, "relative-background.npz")

    def test_nonuniform_profile_rejects_legacy_uniform_components(self):
        with self.assertRaisesRegex(ValueError, "valid only for profile"):
            parse(_base_data(background_field={
                "enabled": True,
                "profile": "linear_vacuum",
                "bx0": 1.0,
            }))

    def test_explicit_file_rejects_unused_params(self):
        with self.assertRaisesRegex(ValueError, "not used with"):
            parse(_base_data(background_field={
                "enabled": True,
                "file": "relative-background.npz",
                "params": {"b_scale": 2.0},
            }))

    def test_a_file_rejects_unknown_params(self):
        with self.assertRaisesRegex(ValueError, "unknown.*a_file"):
            parse(_base_data(background_field={
                "enabled": True,
                "a_file": "relative-vector-potential.npz",
                "params": {"b_sclae": 2.0},
            }))

    def test_conductors_and_file_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "at most one|mutually exclusive"):
            parse(_base_data(background_field={
                "enabled": True,
                "conductors": [_circular_loop("loop")],
                "file": "relative-background.npz",
            }))

    def test_conductors_and_a_file_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "at most one|mutually exclusive"):
            parse(_base_data(background_field={
                "enabled": True,
                "conductors": [_circular_loop("loop")],
                "a_file": "relative-vector-potential.npz",
            }))

    def test_conductors_must_be_nonempty(self):
        with self.assertRaisesRegex(ValueError, "conductors.*non-empty"):
            parse(_base_data(background_field={
                "enabled": True,
                "conductors": [],
            }))

    def test_conductors_reject_in_plane_uniform_components(self):
        for component in ("bx0", "by0"):
            with self.subTest(component=component):
                with self.assertRaisesRegex(ValueError, "not used with conductors"):
                    parse(_base_data(units="SI", background_field={
                        "enabled": True,
                        "conductors": [_circular_loop("loop")],
                        component: 1.0,
                    }))

    def test_conductors_reject_unknown_params(self):
        with self.assertRaisesRegex(ValueError, "unknown.*conductors"):
            parse(_base_data(units="SI", background_field={
                "enabled": True,
                "conductors": [_circular_loop("loop")],
                "params": {"b_sclae": 2.0},
            }))

    def test_conductors_require_si_units(self):
        with self.assertRaisesRegex(ValueError, "conductors requires units: SI"):
            parse(_base_data(background_field={
                "enabled": True,
                "conductors": [_circular_loop("loop")],
            }))

    def test_malformed_conductor_geometry_is_rejected_during_parse(self):
        malformed = _circular_loop("missing_radius")
        del malformed["geometry"]["radius_m"]
        with self.assertRaisesRegex(ValueError, "radius_m"):
            parse(_base_data(units="SI", background_field={
                "enabled": True,
                "conductors": [malformed],
            }))

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
        xc, yc, xf, yf = _padded_meshes(domain, NGHOST)
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

    # The boundary-compatibility sweep runs on device inside every
    # build_background_* entry point. These drive it through the explicit-array
    # source, which is the one that lets a test hand it a deliberately
    # incompatible field. Each field below is constructed to be discretely
    # solenoidal, because the divergence proof runs first and would otherwise
    # be the error reported.

    @staticmethod
    def _closure_spec(nx, ny, g, modes, cylindrical=0, origin_x=0.0):
        spec = _core.mhd.MhdBackgroundBuildSpec()
        spec.grid = _core.mhd.Grid2D(nx, ny, 1.0, 1.0, origin_x, 0.0, g)
        spec.cylindrical = cylindrical
        spec.magnetic_scale = 1.0
        spec.field_modes = modes
        return spec

    def test_periodic_background_seam_must_wrap(self):
        nx = ny = 2
        g = 1
        shape = (ny + 2 * g, nx + 2 * g)
        bx = np.zeros(shape)
        by = np.zeros(shape)
        bz = np.zeros(shape)
        # x=-1 is the low periodic target for layer one; x=nx-1 is its source.
        bz[g, g - 1] = 1.0
        spec = self._closure_spec(nx, ny, g, [1, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "not periodic across the x"):
            _core.mhd.build_background_from_arrays(
                spec, bx.reshape(-1), by.reshape(-1), bz.reshape(-1))

    def test_wall_background_requires_zero_normal_and_parity(self):
        nx = ny = 2
        g = 1
        shape = (ny + 2 * g, nx + 2 * g)
        modes = [2, 2, 0, 0]   # wall, wall, ignored, ignored

        bx = np.zeros(shape)
        by = np.zeros(shape)
        bz = np.zeros(shape)
        # Uniform across the interior faces so the divergence stays zero, with
        # a correctly odd low-wall ghost so the parity rule is satisfied and
        # only the normal-component rule can fire. Bx on the wall face is
        # nonzero, and a wall requires it to vanish exactly.
        bx[:, :] = 1.0
        bx[:, g - 1] = -1.0
        spec = self._closure_spec(nx, ny, g, modes)
        with self.assertRaisesRegex(ValueError, "normal component at an x wall"):
            _core.mhd.build_background_from_arrays(
                spec, bx.reshape(-1), by.reshape(-1), bz.reshape(-1))

        bx.fill(0.0)
        by[:, g] = 1.0       # even tangential source at x=0, constant in y
        by[:, g - 1] = -1.0  # invalid odd low-wall ghost
        with self.assertRaisesRegex(ValueError, "x-wall mirror parity"):
            _core.mhd.build_background_from_arrays(
                spec, bx.reshape(-1), by.reshape(-1), bz.reshape(-1))

    def test_cylindrical_axis_requires_odd_toroidal_parity(self):
        nx = ny = 2
        g = 1
        shape = (ny + 2 * g, nx + 2 * g)
        bx = np.zeros(shape)
        by = np.zeros(shape)
        bz = np.zeros(shape)
        spec = self._closure_spec(nx, ny, g, [3, 0, 0, 0], cylindrical=1)
        bz[g, g] = 1.0
        bz[g, g - 1] = -1.0
        _core.mhd.build_background_from_arrays(
            spec, bx.reshape(-1), by.reshape(-1), bz.reshape(-1))

        bz[g, g - 1] = 1.0
        with self.assertRaisesRegex(ValueError, "axis parity"):
            _core.mhd.build_background_from_arrays(
                spec, bx.reshape(-1), by.reshape(-1), bz.reshape(-1))


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


@unittest.skipUnless(_has_hip_runtime(), "no HIP runtime visible")
class InlineConductorBackgroundTests(unittest.TestCase):

    @staticmethod
    def _write_a_file_via_coil_cli(
            path: Path, deck: MhdDeck, nghost: int) -> None:
        domain = deck.domain
        dx = domain.lx_m / domain.nx
        dy = domain.ly_m / domain.ny
        pitch = domain.nx + 2 * nghost
        height = domain.ny + 2 * nghost
        x_lo = domain.origin_x_m - nghost * dx
        x_hi = domain.origin_x_m + domain.lx_m + nghost * dx
        z_lo = domain.origin_y_m - nghost * dy
        z_hi = domain.origin_y_m + domain.ly_m + nghost * dy

        # Build the independent reference through the public coil-deck parser
        # and the CLI payload shaper. In particular, the coil path derives its
        # ObservationGrid spacing from exact bounds rather than reusing the MHD
        # loader's origin/spacing construction.
        coil_deck = coil_io.parse({
            "units": "SI",
            "conductors": deck.background.conductors,
            "observation": {
                "type": "grid",
                "bounds_m": [[x_lo, x_hi], [0.0, 0.0], [z_lo, z_hi]],
                "resolution": [pitch + 1, 1, height + 1],
            },
            "output": {
                "format": "npz",
                "path": path.name,
                "fields": ["A_xyz_grid"],
            },
        })
        evaluator = _core.magnetostatics.create_field_evaluator("biot_savart")
        evaluator.configure({})
        magnetic_field = np.asarray(evaluator.evaluate_B(
            coil_deck.conductors, coil_deck.observation.points))
        vector_potential = np.asarray(
            evaluator.evaluate_A(
                coil_deck.conductors, coil_deck.observation.points))
        payload, _ = coil_cli._build_payload(
            coil_deck, magnetic_field, vector_potential)
        np.savez(path, **payload)

    @staticmethod
    def _cartesian_deck() -> MhdDeck:
        domain = _domain(
            nx=10,
            ny=8,
            lx_m=0.04,
            ly_m=0.04,
            origin_x_m=-0.02,
            origin_y_m=-0.02,
        )
        boundary = BoundaryConfig(
            fluid=("outflow",) * 4,
            field=("outflow",) * 4,
        )
        return _deck(
            units="SI",
            domain=domain,
            boundary=boundary,
            background=BackgroundConfig(
                enabled=True,
                conductors=_helmholtz_pair(),
                bz0=0.02,
                params={"b_scale": 1.25},
            ),
        )

    def test_square_toroid_inline_matches_exact_padded_a_file_bitwise(self):
        example = REPO_ROOT / "examples" / "square_toroid_mhd" / "input.yaml"
        deck = load_mhd_deck(example)
        self.assertIsNotNone(deck.background.conductors)

        background = replace(
            deck.background,
            params={"b_scale": 1.0, "vacuum_project": False},
        )
        inline_deck = replace(deck, background=background)
        nghost = 2

        dx = inline_deck.domain.lx_m / inline_deck.domain.nx
        dy = inline_deck.domain.ly_m / inline_deck.domain.ny
        # The removed hand-written coil deck rounded these bounds to six
        # decimals, which slightly changed its pitch. The bitwise reference
        # below deliberately uses the exact domain-derived bounds specified by
        # the feature contract and feeds them through the independent coil path.
        self.assertAlmostEqual(
            inline_deck.domain.origin_x_m - nghost * dx,
            0.86078125,
            places=15,
        )
        self.assertAlmostEqual(
            inline_deck.domain.origin_y_m - nghost * dy,
            -0.13921875,
            places=15,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            a_path = Path(temporary_directory) / "coil.npz"
            self._write_a_file_via_coil_cli(a_path, inline_deck, nghost)
            a_file_deck = replace(
                inline_deck,
                background=replace(
                    background,
                    conductors=None,
                    a_file=str(a_path),
                ),
            )
            from_inline = build_background_field(inline_deck, nghost)
            from_a_file = build_background_field(a_file_deck, nghost)

        for component in ("b0x", "b0y", "b0z"):
            with self.subTest(component=component):
                np.testing.assert_array_equal(
                    from_inline[component],
                    from_a_file[component],
                )

    def test_cartesian_conductor_background_is_discretely_solenoidal(self):
        deck = self._cartesian_deck()
        deck.validate()
        nghost = 2
        background = build_background_field(deck, nghost)
        domain = deck.domain

        defect = _relative_divergence_linf(
            background["b0x"],
            background["b0y"],
            domain.nx,
            domain.ny,
            nghost,
            domain.lx_m / domain.nx,
            domain.ly_m / domain.ny,
            geometry="cartesian",
        )
        self.assertLessEqual(defect, DISCRETE_SOLENOIDAL_TOLERANCE)
        self.assertGreater(float(np.max(np.abs(background["b0x"]))), 0.0)
        self.assertGreater(float(np.max(np.abs(background["b0y"]))), 0.0)

    def test_nghost_expands_halo_without_changing_overlapping_values(self):
        deck = self._cartesian_deck()
        deck.validate()
        backgrounds = {
            nghost: build_background_field(deck, nghost)
            for nghost in (2, 3)
        }
        domain = deck.domain

        for nghost, background in backgrounds.items():
            expected_size = ((domain.nx + 2 * nghost)
                             * (domain.ny + 2 * nghost))
            for component in ("b0x", "b0y", "b0z"):
                with self.subTest(nghost=nghost, component=component):
                    self.assertEqual(
                        np.asarray(background[component]).size,
                        expected_size,
                    )

        small_shape = (domain.ny + 4, domain.nx + 4)
        large_shape = (domain.ny + 6, domain.nx + 6)
        for component in ("b0x", "b0y", "b0z"):
            with self.subTest(component=component):
                smaller = np.asarray(backgrounds[2][component]).reshape(small_shape)
                larger = np.asarray(backgrounds[3][component]).reshape(large_shape)
                # The two ObservationGrids have different padded origins, so
                # shared coordinates can differ by a few evaluation ULPs; SI
                # conversion then scales that harmless roundoff uniformly.
                np.testing.assert_allclose(
                    larger[1:-1, 1:-1],
                    smaller,
                    rtol=5.0e-12,
                    atol=1.0e-14,
                )


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
        divb = _divergence_linf(
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

        divb = _divergence_linf(
            bg["b0x"], bg["b0y"], domain.nx, domain.ny, g, dx, dy,
            geometry="cylindrical", origin_x=domain.origin_x_m)
        curlb = _curl_linf(
            bg["b0x"], bg["b0y"], domain.nx, domain.ny, g, dx, dy)
        raw_curlb = _curl_linf(
            raw_bg["b0x"], raw_bg["b0y"], domain.nx, domain.ny, g, dx, dy)
        scale = max(1.0, (np.max(np.abs(bg["b0x"])) +
                          np.max(np.abs(bg["b0y"]))) / min(dx, dy))
        self.assertLessEqual(divb, 1.0e-9 * scale)
        self.assertLessEqual(curlb, 1.0e-9 * scale)
        self.assertLess(curlb, raw_curlb)


if __name__ == "__main__":
    unittest.main()
