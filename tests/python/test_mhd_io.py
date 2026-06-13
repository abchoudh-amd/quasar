"""Deck I/O tests for the high-order ideal-MHD module.

Mirrors the PIC deck-I/O contract in ``tests/python/test_pic_io.py``:

* the loader is ``quasar.mhd.io.load(path)`` / ``parse(data)`` (same names the PIC
  layer uses),
* invalid decks raise :class:`ValueError` (the project's deck-error type, matching
  what :mod:`quasar.pic.io` raises),
* scheme-name validation is driven by the live C++ registries exposed on
  ``_core.mhd`` (so a name present in ``registered_reconstructions()`` validates
  and a bogus one fails) -- exactly as PIC validates boundary/filter names against
  ``_core.pic.registered_*``.

Until the ``quasar.mhd`` package and the ``_core.mhd`` bindings exist (and the
build tree is refreshed), these tests fail at import with a clean
ModuleNotFoundError/AttributeError -- the intended RED state.
"""

import unittest

from quasar import _core
from quasar.mhd.io import (
    BoundaryConfig,
    Diagnostics,
    Domain,
    Initial,
    MhdDeck,
    Numerics,
    Time,
    _parse_side_map,
    parse,
)


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


class MhdDeckLoadTests(unittest.TestCase):
    """A valid deck parses and exposes the parsed fields."""

    def _full_data(self) -> dict:
        return {
            "units": "normalized",
            "domain": {"nx": 32, "ny": 24, "lx_m": 2.0, "ly_m": 1.0},
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
            "time": {"dt_s": "auto", "steps": 5},
            "diagnostics": {"output_path": "out.npz", "cadence": 1,
                            "fields": ["rho", "energy"], "divb": True},
            "boundary": {
                "fluid": ["periodic", "periodic", "periodic", "periodic"],
                "field": ["periodic", "periodic", "periodic", "periodic"],
            },
        }

    def test_parse_full_deck_exposes_fields(self):
        deck = parse(self._full_data())
        self.assertEqual(deck.domain.nx, 32)
        self.assertEqual(deck.domain.ny, 24)
        self.assertAlmostEqual(deck.numerics.gamma, 1.6666667)
        self.assertEqual(deck.numerics.reconstruction, "mp7")
        self.assertEqual(deck.numerics.riemann, "hlld")
        self.assertEqual(deck.numerics.integrator, "ssprk3")
        self.assertEqual(deck.numerics.ct, "fd_ct_christlieb")
        self.assertEqual(deck.numerics.positivity, "troubled_cell")
        self.assertEqual(deck.initial.type, "orszag_tang")
        self.assertEqual(deck.time.steps, 5)
        self.assertEqual(deck.time.dt_s, "auto")
        self.assertEqual(deck.geometry, "cartesian")
        self.assertEqual(
            deck.boundary.fluid,
            ("periodic", "periodic", "periodic", "periodic"))
        self.assertEqual(
            deck.boundary.field,
            ("periodic", "periodic", "periodic", "periodic"))

    def test_load_from_yaml_file(self):
        import tempfile
        from pathlib import Path

        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deck.yaml"
            path.write_text(yaml.safe_dump(self._full_data()))
            from quasar.mhd.io import load
            deck = load(path)
            self.assertEqual(deck.domain.nx, 32)
            self.assertEqual(deck.numerics.riemann, "hlld")

    def test_minimal_deck_validates(self):
        _deck().validate()

    def test_geometry_defaults_to_cartesian(self):
        self.assertEqual(_deck().geometry, "cartesian")

    def test_cylindrical_geometry_accepted(self):
        _deck(geometry="cylindrical").validate()

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            _deck(geometry="spherical").validate()


class GammaValidationTests(unittest.TestCase):

    def test_gamma_at_or_below_one_rejected(self):
        for bad in (0.5, 1.0):
            with self.subTest(gamma=bad):
                with self.assertRaises(ValueError):
                    _deck(numerics=_numerics(gamma=bad)).validate()

    def test_gamma_above_one_accepted(self):
        _deck(numerics=_numerics(gamma=1.4)).validate()

    def test_nonfinite_gamma_rejected(self):
        with self.assertRaises(ValueError):
            _deck(numerics=_numerics(gamma=float("nan"))).validate()


class DomainValidationTests(unittest.TestCase):

    def test_zero_nx_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=_domain(nx=0)).validate()

    def test_negative_nx_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=_domain(nx=-8)).validate()

    def test_nonpositive_ny_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=_domain(ny=0)).validate()

    def test_nonpositive_length_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=_domain(lx_m=0.0)).validate()

    def test_nonfinite_length_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=_domain(ly_m=float("inf"))).validate()


class SchemeNameValidationTests(unittest.TestCase):
    """Scheme-name validation is driven by the live ``_core.mhd`` registries,
    exactly as PIC validates boundary/filter names against ``_core.pic``."""

    def test_unknown_reconstruction_rejected(self):
        with self.assertRaises(ValueError):
            _deck(numerics=_numerics(reconstruction="nope")).validate()

    def test_unknown_riemann_rejected(self):
        with self.assertRaises(ValueError):
            _deck(numerics=_numerics(riemann="nope")).validate()

    def test_unknown_integrator_rejected(self):
        with self.assertRaises(ValueError):
            _deck(numerics=_numerics(integrator="nope")).validate()

    def test_unknown_ct_rejected(self):
        with self.assertRaises(ValueError):
            _deck(numerics=_numerics(ct="nope")).validate()

    def test_unknown_positivity_rejected(self):
        with self.assertRaises(ValueError):
            _deck(numerics=_numerics(positivity="nope")).validate()

    def test_accepted_reconstructions_equal_registry(self):
        registered = set(_core.mhd.registered_reconstructions())
        self.assertTrue(registered)  # registry is non-empty
        # Every registered name validates...
        for name in registered:
            with self.subTest(name=name):
                _deck(numerics=_numerics(reconstruction=name)).validate()
        # ...and a name guaranteed absent fails.
        bogus = "definitely_not_a_reconstruction"
        self.assertNotIn(bogus, registered)
        with self.assertRaises(ValueError):
            _deck(numerics=_numerics(reconstruction=bogus)).validate()


class InitialConditionValidationTests(unittest.TestCase):

    def test_canonical_initial_types_accepted(self):
        for kind in ("brio_wu", "alfven_wave", "orszag_tang", "blast", "rotor"):
            with self.subTest(kind=kind):
                _deck(initial=Initial(type=kind)).validate()

    def test_unknown_initial_type_rejected(self):
        with self.assertRaises(ValueError):
            _deck(initial=Initial(type="not_a_real_ic")).validate()


class TimeAndCflValidationTests(unittest.TestCase):

    def test_nonpositive_steps_rejected(self):
        with self.assertRaises(ValueError):
            _deck(time=Time(dt_s="auto", steps=0)).validate()

    def test_bad_dt_string_rejected(self):
        with self.assertRaises(ValueError):
            _deck(time=Time(dt_s="fast", steps=5)).validate()

    def test_nonpositive_dt_rejected(self):
        with self.assertRaises(ValueError):
            _deck(time=Time(dt_s=-1.0e-3, steps=5)).validate()

    def test_explicit_positive_dt_passes_validate(self):
        # validate() only checks that dt_s is a positive float or "auto"; it has
        # no constructed solver/seeded state, so (like quasar.pic.io.validate())
        # it does NOT do a CFL check. An over-CFL explicit dt is a structurally
        # valid deck here and is rejected later, at the CLI/solver step layer
        # (see test_mhd_cli.py::MhdCliOverCflDtTests).
        _deck(time=Time(dt_s=1.0e6, steps=5)).validate()

    def test_auto_dt_accepted(self):
        _deck(time=Time(dt_s="auto", steps=5)).validate()


class BoundaryParseTests(unittest.TestCase):
    """The boundary block accepts a scalar, a 4-list, or a side-keyed map, like
    the PIC ``_parse_side_map`` flexibility. At minimum the 4-list form is tested."""

    def test_four_list_maps_per_side(self):
        sides = _parse_side_map(
            ["periodic", "outflow", "wall", "periodic"],
            "periodic", "boundary.fluid")
        self.assertEqual(
            sides, ("periodic", "outflow", "wall", "periodic"))

    def test_scalar_broadcasts_to_four_sides(self):
        sides = _parse_side_map("periodic", "periodic", "boundary.field")
        self.assertEqual(sides, ("periodic", "periodic", "periodic", "periodic"))

    def test_none_uses_default(self):
        sides = _parse_side_map(None, "periodic", "boundary.fluid")
        self.assertEqual(sides, ("periodic", "periodic", "periodic", "periodic"))

    def test_side_keyed_map_supported(self):
        sides = _parse_side_map(
            {"x_lo": "outflow", "x_hi": "outflow",
             "y_lo": "periodic", "y_hi": "periodic"},
            "periodic", "boundary.fluid")
        self.assertEqual(sides, ("outflow", "outflow", "periodic", "periodic"))

    def test_wrong_length_list_raises(self):
        with self.assertRaises(ValueError):
            _parse_side_map(["periodic", "outflow"], "periodic", "boundary.fluid")

    def test_wall_fluid_boundary_validates_and_round_trips(self):
        # "wall" is the renamed reflecting boundary; a deck selecting it on a
        # side validates against the live registry and round-trips per side.
        # Boundary order: [x_min, x_max, y_min, y_max].
        deck = _deck(boundary=BoundaryConfig(
            fluid=("periodic", "outflow", "wall", "periodic"),
            field=("periodic", "outflow", "wall", "periodic")))
        deck.validate()
        self.assertEqual(
            deck.boundary.fluid,
            ("periodic", "outflow", "wall", "periodic"))
        self.assertEqual(
            deck.boundary.field,
            ("periodic", "outflow", "wall", "periodic"))

    def test_wall_boundary_parsed_from_yaml(self):
        deck = parse({
            "domain": {"nx": 8, "ny": 8, "lx_m": 1.0, "ly_m": 1.0},
            "numerics": {"gamma": 1.6666667},
            "initial": {"type": "brio_wu"},
            "time": {"dt_s": "auto", "steps": 2},
            "boundary": {
                "fluid": ["wall", "wall", "periodic", "periodic"],
                "field": ["wall", "wall", "periodic", "periodic"],
            },
        })
        deck.validate()
        self.assertEqual(
            deck.boundary.fluid,
            ("wall", "wall", "periodic", "periodic"))
        self.assertEqual(
            deck.boundary.field,
            ("wall", "wall", "periodic", "periodic"))

    def test_reflecting_fluid_boundary_rejected(self):
        # "reflecting" was renamed to "wall" and removed from the registry; a
        # deck selecting it on a side must fail registry-driven validation.
        with self.assertRaises(ValueError):
            _deck(boundary=BoundaryConfig(
                fluid=("periodic", "reflecting", "periodic", "periodic"))
            ).validate()

    def test_reflecting_field_boundary_rejected(self):
        with self.assertRaises(ValueError):
            _deck(boundary=BoundaryConfig(
                field=("periodic", "reflecting", "periodic", "periodic"))
            ).validate()

    def test_unknown_fluid_boundary_rejected(self):
        with self.assertRaises(ValueError):
            _deck(boundary=BoundaryConfig(
                fluid=("periodic", "teleport", "periodic", "periodic"))).validate()

    def test_unknown_field_boundary_rejected(self):
        with self.assertRaises(ValueError):
            _deck(boundary=BoundaryConfig(
                field=("periodic", "warp", "periodic", "periodic"))).validate()

    def test_four_list_boundary_parsed_from_yaml(self):
        deck = parse({
            "domain": {"nx": 8, "ny": 8, "lx_m": 1.0, "ly_m": 1.0},
            "numerics": {"gamma": 1.6666667},
            "initial": {"type": "brio_wu"},
            "time": {"dt_s": "auto", "steps": 2},
            "boundary": {
                "fluid": ["outflow", "outflow", "periodic", "periodic"],
                "field": ["outflow", "outflow", "periodic", "periodic"],
            },
        })
        self.assertEqual(
            deck.boundary.fluid,
            ("outflow", "outflow", "periodic", "periodic"))


class DiagnosticsValidationTests(unittest.TestCase):

    def test_negative_cadence_rejected(self):
        with self.assertRaises(ValueError):
            _deck(diagnostics=Diagnostics(cadence=-1)).validate()

    def test_divb_flag_parsed(self):
        deck = parse({
            "domain": {"nx": 8, "ny": 8, "lx_m": 1.0, "ly_m": 1.0},
            "numerics": {"gamma": 1.6666667},
            "initial": {"type": "rotor"},
            "time": {"dt_s": "auto", "steps": 2},
            "diagnostics": {"output_path": "out.npz", "divb": True},
        })
        self.assertTrue(deck.diagnostics.divb)


if __name__ == "__main__":
    unittest.main()
