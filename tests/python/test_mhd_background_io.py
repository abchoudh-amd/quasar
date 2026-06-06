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

Scheme-name validation is driven by the live C++ registry
``_core.mhd.registered_mhd_background_profiles()`` (mirrors how the rest of the
MHD deck validates against ``_core.mhd.registered_*``).

Until the ``background_field`` schema (and the ``registered_mhd_background_profiles``
binding) exist, these tests fail at import/attribute time -- the intended RED state.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from quasar import _core
from quasar.mhd.io import (
    BackgroundConfig,
    Domain,
    Initial,
    MhdDeck,
    Numerics,
    Time,
    build_background_field,
    parse,
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


if __name__ == "__main__":
    unittest.main()
