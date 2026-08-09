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

"""

import math
import unittest

import numpy as np

from quasar import _core
from quasar.mhd.io import (
    BackgroundConfig,
    BoundaryConfig,
    Diagnostics,
    Domain,
    Initial,
    MhdDeck,
    Numerics,
    Time,
    _parse_side_map,
    build_initial_state,
    parse,
)
from quasar.mhd import _units as mhd_units
from quasar.mhd import numerics as mhd_num


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


def _brio_params(**overrides) -> dict:
    params = {
        "interface": 0.5,
        "left": {"rho": 1.0, "p": 1.0, "vx": 0.0, "vy": 0.0,
                 "vz": 0.0, "bx": 0.75, "by": 1.0, "bz": 0.0},
        "right": {"rho": 0.125, "p": 0.1, "vx": 0.0, "vy": 0.0,
                  "vz": 0.0, "bx": 0.75, "by": -1.0, "bz": 0.0},
    }
    params.update(overrides)
    return params


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

    def test_loader_rejects_duplicate_yaml_keys(self):
        import tempfile
        from pathlib import Path

        from quasar.mhd.io import load

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.yaml"
            path.write_text("units: normalized\nunits: normalized\n")
            with self.assertRaisesRegex(
                    ValueError, r"duplicate YAML key 'units'.*line 2"):
                load(path)

    def test_minimal_deck_validates(self):
        _deck().validate()

    def test_internal_tile_boundary_is_not_public_deck_vocabulary(self):
        boundary = BoundaryConfig(
            fluid=("internal",) * 4, field=("internal",) * 4)
        with self.assertRaisesRegex(ValueError, "reserved.*distributed"):
            _deck(boundary=boundary).validate()

    def test_geometry_defaults_to_cartesian(self):
        self.assertEqual(_deck().geometry, "cartesian")

    def test_cylindrical_geometry_accepted(self):
        axis = BoundaryConfig(
            fluid=("axis", "outflow", "periodic", "periodic"),
            field=("axis", "outflow", "periodic", "periodic"))
        _deck(geometry="cylindrical", boundary=axis,
              numerics=_numerics(reconstruction="muscl_minmod")).validate()

    def test_cylindrical_high_order_reconstruction_rejected(self):
        axis = BoundaryConfig(
            fluid=("axis", "outflow", "periodic", "periodic"),
            field=("axis", "outflow", "periodic", "periodic"))
        for reconstruction in ("mp5", "mp7"):
            with self.subTest(reconstruction=reconstruction):
                with self.assertRaisesRegex(
                        ValueError, r"only.*muscl_minmod.*r-weighted"):
                    _deck(
                        geometry="cylindrical", boundary=axis,
                        numerics=_numerics(
                            reconstruction=reconstruction)).validate()

    def test_annular_cylindrical_geometry_accepted(self):
        annular = BoundaryConfig(
            fluid=("wall", "wall", "periodic", "periodic"),
            field=("wall", "wall", "periodic", "periodic"))
        _deck(geometry="cylindrical",
              domain=_domain(origin_x_m=0.5), boundary=annular,
              numerics=_numerics(reconstruction="muscl_minmod")).validate()

    def test_annular_reconstruction_halo_cannot_cross_axis(self):
        annular = BoundaryConfig(
            fluid=("wall", "wall", "periodic", "periodic"),
            field=("wall", "wall", "periodic", "periodic"))
        deck = _deck(
            geometry="cylindrical",
            domain=_domain(origin_x_m=0.1), boundary=annular,
            numerics=_numerics(reconstruction="muscl_minmod"))
        deck.validate()
        with self.assertRaisesRegex(ValueError, r"origin_x_m - nghost\*dr > 0"):
            build_initial_state(deck, nghost=2)

    def test_axis_reconstruction_halo_uses_parity_extension(self):
        axis = BoundaryConfig(
            fluid=("axis", "outflow", "periodic", "periodic"),
            field=("axis", "outflow", "periodic", "periodic"))
        deck = _deck(
            geometry="cylindrical", boundary=axis,
            numerics=_numerics(reconstruction="muscl_minmod"))
        deck.validate()
        state = build_initial_state(deck, nghost=4)
        self.assertEqual(
            np.asarray(state["rho"]).size,
            (deck.domain.nx + 8) * (deck.domain.ny + 8))

    def test_negative_cylindrical_radius_rejected(self):
        physical = BoundaryConfig(
            fluid=("wall", "wall", "periodic", "periodic"),
            field=("wall", "wall", "periodic", "periodic"))
        with self.assertRaises(ValueError):
            _deck(geometry="cylindrical",
                  domain=_domain(origin_x_m=-0.1), boundary=physical,
                  numerics=_numerics(
                      reconstruction="muscl_minmod")).validate()

    def test_axis_boundary_rejected_on_annulus(self):
        axis = BoundaryConfig(
            fluid=("axis", "outflow", "periodic", "periodic"),
            field=("axis", "outflow", "periodic", "periodic"))
        with self.assertRaises(ValueError):
            _deck(geometry="cylindrical",
                  domain=_domain(origin_x_m=0.5), boundary=axis,
                  numerics=_numerics(
                      reconstruction="muscl_minmod")).validate()

    def test_cylindrical_axial_background_accepted(self):
        annular = BoundaryConfig(
            fluid=("wall", "wall", "periodic", "periodic"),
            field=("wall", "wall", "periodic", "periodic"))
        _deck(
            geometry="cylindrical", domain=_domain(origin_x_m=0.5),
            boundary=annular,
            numerics=_numerics(reconstruction="muscl_minmod"),
            background=BackgroundConfig(
                enabled=True, profile="uniform", by0=0.25)).validate()

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            _deck(geometry="spherical").validate()

    def test_periodic_boundaries_must_be_paired_and_consistent(self):
        one_sided = BoundaryConfig(
            fluid=("periodic", "outflow", "periodic", "periodic"),
            field=("periodic", "outflow", "periodic", "periodic"))
        with self.assertRaisesRegex(ValueError, "both sides"):
            _deck(boundary=one_sided).validate()

        mismatched = BoundaryConfig(
            fluid=("periodic", "periodic", "periodic", "periodic"),
            field=("outflow", "outflow", "periodic", "periodic"))
        with self.assertRaisesRegex(ValueError, "periodicity must match"):
            _deck(boundary=mismatched).validate()


class UnknownKeyValidationTests(unittest.TestCase):

    @staticmethod
    def _data() -> dict:
        return {
            "units": "normalized",
            "domain": {"nx": 8, "ny": 8, "lx_m": 1.0, "ly_m": 1.0},
            "geometry": "cartesian",
            "numerics": {"gamma": 5.0 / 3.0},
            "initial": {"type": "orszag_tang", "params": {}},
            "time": {"dt_s": "auto", "steps": 2},
            "diagnostics": {"cadence": 0, "divb": True},
            "boundary": {"fluid": "periodic", "field": "periodic"},
            "background_field": {"enabled": False},
        }

    def test_unknown_keys_are_rejected_at_every_schema_level(self):
        cases = (
            ("top", lambda d: d.__setitem__("geometery", "cylindrical")),
            ("domain", lambda d: d["domain"].__setitem__("nxx", 16)),
            ("numerics", lambda d: d["numerics"].__setitem__("cfl_number", 0.2)),
            ("initial", lambda d: d["initial"].__setitem__("kind", "rotor")),
            ("time", lambda d: d["time"].__setitem__("step", 10)),
            ("diagnostics", lambda d: d["diagnostics"].__setitem__("div_b", True)),
            ("boundary", lambda d: d["boundary"].__setitem__("fields", "wall")),
            ("background", lambda d: d["background_field"].__setitem__("enable", True)),
        )
        for name, mutate in cases:
            with self.subTest(section=name):
                data = self._data()
                mutate(data)
                with self.assertRaisesRegex(ValueError, "unknown key"):
                    parse(data)

    def test_unknown_boundary_side_is_rejected(self):
        data = self._data()
        data["boundary"]["fluid"] = {"x_low": "wall"}
        with self.assertRaisesRegex(ValueError, "unknown key"):
            parse(data)


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

    def test_fast_magnetosonic_helper_matches_ideal_gas_gamma_domain(self):
        for bad in (0.5, 1.0, float("nan")):
            with self.subTest(gamma=bad):
                with self.assertRaisesRegex(ValueError, "greater than one"):
                    mhd_num.fast_magnetosonic_speed(
                        1.0, 1.0, 0.0, 0.0, 0.0, bad)

    def test_numerical_helpers_reject_non_real_array_dtypes(self):
        invalid = (
            np.array([1.0 + 2.0j], dtype=np.complex128),
            np.array([True], dtype=np.bool_),
            np.array([1.0], dtype=object),
            np.array(["1.0"]),
        )
        for rho in invalid:
            with self.subTest(dtype=rho.dtype):
                with self.assertRaisesRegex(
                        ValueError, "real floating-point or integer"):
                    mhd_num.fast_magnetosonic_speed(
                        rho, 1.0, 0.0, 0.0, 0.0, 5.0 / 3.0)

    def test_numerical_helpers_accept_real_integer_arrays(self):
        speed = mhd_num.fast_magnetosonic_speed(
            np.array([1], dtype=np.int64), np.array([1], dtype=np.uint8),
            0, 0, 0, 5.0 / 3.0)
        np.testing.assert_allclose(speed, np.sqrt(5.0 / 3.0))


class DomainValidationTests(unittest.TestCase):

    def test_zero_nx_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=_domain(nx=0)).validate()

    def test_negative_nx_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=_domain(nx=-8)).validate()

    def test_grid_dimensions_must_be_exact_integers(self):
        for bad in (8.0, 8.5, True):
            with self.subTest(nx=bad):
                with self.assertRaises(ValueError):
                    _deck(domain=_domain(nx=bad)).validate()

    def test_nonpositive_ny_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=_domain(ny=0)).validate()

    def test_nonpositive_length_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=_domain(lx_m=0.0)).validate()

    def test_nonfinite_length_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=_domain(ly_m=float("inf"))).validate()

    def test_nonrepresentable_upper_bound_and_spacing_rejected(self):
        largest = float.fromhex("0x1.fffffffffffffp+1023")
        smallest = float.fromhex("0x0.0000000000001p-1022")
        with self.assertRaisesRegex(ValueError, "upper x bound"):
            _deck(domain=_domain(origin_x_m=largest, lx_m=largest)).validate()
        with self.assertRaisesRegex(ValueError, "x spacing"):
            _deck(domain=_domain(nx=2, lx_m=smallest)).validate()

    def test_actual_high_cell_center_cannot_collapse_onto_high_face(self):
        # This exact binary64 case distinguishes the Grid2D accessor expression
        # x0 + (nx-0.5)*dx from the non-equivalent reassociation x_hi-dx/2.
        origin = float.fromhex("0x1.58be77ab34af0p-1002")
        length = float.fromhex("0x0.0000000373a5dp-1022")
        with self.assertRaisesRegex(ValueError, "coordinates collapse"):
            _deck(domain=_domain(
                nx=2, origin_x_m=origin, lx_m=length)).validate()


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
        initials = (
            Initial(type="brio_wu", params=_brio_params()),
            Initial(type="alfven_wave"),
            Initial(type="orszag_tang"),
            Initial(type="blast"),
            Initial(type="rotor"),
            Initial(type="confined_blob"),
        )
        for initial in initials:
            with self.subTest(kind=initial.type):
                _deck(initial=initial).validate()

    def test_unknown_initial_type_rejected(self):
        with self.assertRaises(ValueError):
            _deck(initial=Initial(type="not_a_real_ic")).validate()

    def test_alfven_wavenumber_must_be_an_exact_integer(self):
        for bad in (1.0, 1.5, True):
            with self.subTest(wavenumber=bad):
                with self.assertRaises(ValueError):
                    _deck(initial=Initial(
                        type="alfven_wave", params={"wavenumber": bad})).validate()

    def test_brio_wu_requires_complete_states_and_continuous_normal_field(self):
        with self.assertRaisesRegex(ValueError, "left"):
            _deck(initial=Initial(
                type="brio_wu", params={"interface": 0.5})).validate()
        with self.assertRaisesRegex(ValueError, "mapping"):
            _deck(initial=Initial(type="brio_wu", params=_brio_params(
                left=[1.0, 2.0]))).validate()
        bad_right = dict(_brio_params()["right"], bx=0.5)
        with self.assertRaisesRegex(ValueError, "continuous normal"):
            _deck(initial=Initial(type="brio_wu", params=_brio_params(
                right=bad_right))).validate()
        with self.assertRaisesRegex(ValueError, "interface"):
            _deck(initial=Initial(type="brio_wu", params=_brio_params(
                interface=float("nan")))).validate()

    def test_alfven_requires_nonzero_background_and_resolved_positive_mode(self):
        with self.assertRaisesRegex(ValueError, "nonzero"):
            _deck(initial=Initial(
                type="alfven_wave", params={"b0": 0.0})).validate()
        for bad in (0, -1, 8, 9):
            with self.subTest(wavenumber=bad):
                with self.assertRaisesRegex(ValueError, "wavenumber|Nyquist"):
                    _deck(initial=Initial(type="alfven_wave", params={
                        "b0": 1.0, "wavenumber": bad})).validate()

    def test_alfven_accepts_field_split_uniform_guide_field(self):
        deck = _deck(
            initial=Initial(type="alfven_wave", params={"b0": 0.0}),
            background=BackgroundConfig(enabled=True, profile="uniform",
                                        bx0=1.0))
        deck.validate()

    def test_alfven_seed_is_exact_finite_volume_projection(self):
        gamma = 1.4
        rho0 = 4.0
        pressure0 = 0.7
        b0 = -1.5
        amplitude = 0.2
        mode = 2
        nx = 12
        nghost = 4
        origin = -0.7
        length = 2.5
        deck = _deck(
            domain=_domain(nx=nx, ny=3, lx_m=length, ly_m=0.6,
                           origin_x_m=origin, origin_y_m=1.2),
            numerics=_numerics(gamma=gamma),
            initial=Initial(type="alfven_wave", params={
                "rho": rho0,
                "p": pressure0,
                "b0": b0,
                "amplitude": amplitude,
                "wavenumber": mode,
            }),
        )
        state = build_initial_state(deck, nghost=nghost)
        shape = (deck.domain.ny + 2 * nghost, nx + 2 * nghost)
        x = origin + (np.arange(-nghost, nx + nghost) + 0.5) * length / nx
        phase = 2.0 * np.pi * mode * (x - origin) / length
        half_cell_phase = math.pi * mode / nx
        average = math.sin(half_cell_phase) / half_cell_phase
        sin_average = average * np.sin(phase)
        cos_average = average * np.cos(phase)

        def rows(values):
            return np.broadcast_to(values, shape)

        # my/mz and Bz are cell averages.  By is a y-normal face average;
        # because the wave varies in x, that face spans the same averaging
        # interval and carries the same sinc factor.
        np.testing.assert_allclose(
            state["my"].reshape(shape),
            rows(math.sqrt(rho0) * amplitude * sin_average),
            rtol=0.0, atol=3.0e-15,
        )
        np.testing.assert_allclose(
            state["mz"].reshape(shape),
            rows(math.sqrt(rho0) * amplitude * cos_average),
            rtol=0.0, atol=3.0e-15,
        )
        np.testing.assert_allclose(
            state["by"].reshape(shape), rows(amplitude * sin_average),
            rtol=0.0, atol=3.0e-15,
        )
        np.testing.assert_allclose(
            state["bz"].reshape(shape), rows(amplitude * cos_average),
            rtol=0.0, atol=3.0e-15,
        )

        # Circular polarization makes kinetic + transverse magnetic energy
        # pointwise constant.  Its cell average is therefore A^2, not A^2 times
        # sinc^2 as would result from squaring the averaged primitives.
        exact_energy = (pressure0 / (gamma - 1.0)
                        + 0.5 * b0 * b0 + amplitude * amplitude)
        np.testing.assert_allclose(
            state["energy"].reshape(shape), exact_energy,
            rtol=0.0, atol=4.0e-14,
        )

    def test_blast_rotor_and_blob_reject_invalid_geometry_parameters(self):
        invalid = (
            Initial(type="blast", params={"center": [0.0], "r_in": 0.1}),
            Initial(type="blast", params={"center": [0.0, 0.0], "r_in": 0.0}),
            Initial(type="rotor", params={"r0": 0.0, "r1": 0.2}),
            Initial(type="rotor", params={"r0": 0.2, "r1": 0.2}),
            Initial(type="rotor", params={"center": [0.0, float("inf")]}),
            Initial(type="confined_blob", params={"blob_half": 0.0}),
        )
        for initial in invalid:
            with self.subTest(kind=initial.type, params=initial.params):
                with self.assertRaises(ValueError):
                    _deck(initial=initial).validate()

    def test_orszag_tang_is_domain_relative(self):
        reference = _deck(
            domain=_domain(nx=12, ny=10, lx_m=1.0, ly_m=1.0),
            initial=Initial(type="orszag_tang"))
        transformed = _deck(
            domain=_domain(nx=12, ny=10, lx_m=2.5, ly_m=3.0,
                           origin_x_m=-4.0, origin_y_m=7.0),
            initial=Initial(type="orszag_tang"))
        reference.validate()
        transformed.validate()
        state_a = build_initial_state(reference, nghost=4)
        state_b = build_initial_state(transformed, nghost=4)
        for name in state_a:
            with self.subTest(component=name):
                np.testing.assert_allclose(
                    state_b[name], state_a[name], rtol=0.0, atol=2.0e-14)

    def test_orszag_tang_default_uses_consistent_mu0_one_normalization(self):
        gamma = 5.0 / 3.0
        deck = _deck(
            domain=_domain(nx=16, ny=16),
            numerics=_numerics(gamma=gamma),
            initial=Initial(type="orszag_tang"),
        )
        state = build_initial_state(deck, nghost=4)
        self.assertTrue(np.allclose(state["rho"], gamma * gamma))
        x = (np.arange(-4, 16 + 4) + 0.5) / 16.0
        y = (np.arange(-4, 16 + 4) + 0.5) / 16.0
        shape = (24, 24)
        np.testing.assert_allclose(
            state["bx"].reshape(shape),
            np.broadcast_to(-np.sin(2.0 * np.pi * y)[:, None], shape),
            rtol=0.0, atol=2.0e-15,
        )
        np.testing.assert_allclose(
            state["by"].reshape(shape),
            np.broadcast_to(np.sin(4.0 * np.pi * x)[None, :], shape),
            rtol=0.0, atol=2.0e-15,
        )

    def test_rotor_taper_uses_canonical_tangential_speed(self):
        r0, r1, u0 = 0.1, 0.2, 2.0
        # Physical cell centres lie at r=0.05 (solid body), 0.15 (mid-taper),
        # and 0.25 (ambient) along +x from the origin.
        deck = _deck(
            domain=_domain(nx=3, ny=1, lx_m=0.3, ly_m=0.1,
                           origin_x_m=0.0, origin_y_m=-0.05),
            initial=Initial(type="rotor", params={
                "center": [0.0, 0.0], "r0": r0, "r1": r1, "u0": u0,
            }),
        )
        state = build_initial_state(deck, nghost=2)
        row = 2
        shape = (5, 7)
        rho = state["rho"].reshape(shape)
        vx = state["mx"].reshape(shape)[row, 2:5] / rho[row, 2:5]
        vy = state["my"].reshape(shape)[row, 2:5] / rho[row, 2:5]
        np.testing.assert_allclose(vx, 0.0, atol=1.0e-15)
        np.testing.assert_allclose(vy, [1.0, 1.0, 0.0], atol=2.0e-14)

    def test_initial_state_requires_strictly_positive_density(self):
        deck = _deck(initial=Initial(
            type="alfven_wave",
            params={"rho": 0.0, "p": 1.0, "wavenumber": 1}))
        with self.assertRaisesRegex(ValueError, r"rho must be .*positive|density"):
            deck.validate()

    def test_initial_state_requires_strictly_positive_pressure(self):
        deck = _deck(initial=Initial(
            type="alfven_wave",
            params={"rho": 1.0, "p": 0.0, "wavenumber": 1}))
        with self.assertRaisesRegex(ValueError, r"p must be .*positive|pressure"):
            deck.validate()


class TimeAndCflValidationTests(unittest.TestCase):

    def test_nonpositive_steps_rejected(self):
        with self.assertRaises(ValueError):
            _deck(time=Time(dt_s="auto", steps=0)).validate()

    def test_steps_must_be_an_exact_integer(self):
        for bad in (5.0, 5.5, True):
            with self.subTest(steps=bad):
                with self.assertRaises(ValueError):
                    _deck(time=Time(dt_s="auto", steps=bad)).validate()

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


class SiMagneticNormalizationTests(unittest.TestCase):

    def test_magnetic_pressure_uses_b_squared_over_two_mu0(self):
        gamma = 5.0 / 3.0
        b_tesla = 0.2
        deck = _deck(
            units="SI", numerics=_numerics(gamma=gamma),
            initial=Initial(type="confined_blob", params={
                "bz": b_tesla, "rho_in": 2.0, "rho_out": 2.0,
                "p_in": 3.0, "p_out": 3.0,
            }))
        state = build_initial_state(deck, nghost=2)
        expected_b = b_tesla / mhd_units.SQRT_MU0
        self.assertTrue(np.allclose(state["bz"], expected_b))
        expected_energy = 3.0 / (gamma - 1.0) + 0.5 * expected_b**2
        self.assertTrue(np.allclose(state["energy"], expected_energy))
        np.testing.assert_allclose(
            mhd_units.magnetic_to_output(state["bz"], "SI"), b_tesla,
            rtol=0.0, atol=2.0 * np.finfo(float).eps)

    def test_alfven_eigenrelation_has_si_mu0(self):
        rho0 = 4.0
        amp_tesla = 2.0e-3
        deck = _deck(
            units="SI", domain=_domain(nx=8, ny=4),
            initial=Initial(type="alfven_wave", params={
                "rho": rho0, "p": 1.0, "b0": 0.1,
                "amplitude": amp_tesla, "wavenumber": 1,
            }))
        state = build_initial_state(deck, nghost=2)
        vtrans = np.sqrt((state["my"] / state["rho"])**2 +
                         (state["mz"] / state["rho"])**2)
        half_cell_phase = math.pi / deck.domain.nx
        average = math.sin(half_cell_phase) / half_cell_phase
        expected = (amp_tesla / math.sqrt(mhd_units.MU0 * rho0) * average)
        np.testing.assert_allclose(vtrans, expected, rtol=2.0e-14, atol=0.0)
        expected_energy = (1.0 / (deck.numerics.gamma - 1.0)
                           + 0.5 * 0.1**2 / mhd_units.MU0
                           + amp_tesla**2 / mhd_units.MU0)
        np.testing.assert_allclose(
            state["energy"], expected_energy, rtol=2.0e-14, atol=0.0)


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
            fluid=("outflow", "outflow", "wall", "wall"),
            field=("outflow", "outflow", "wall", "wall")))
        deck.validate()
        self.assertEqual(
            deck.boundary.fluid,
            ("outflow", "outflow", "wall", "wall"))
        self.assertEqual(
            deck.boundary.field,
            ("outflow", "outflow", "wall", "wall"))

    def test_wall_boundary_parsed_from_yaml(self):
        deck = parse({
            "domain": {"nx": 8, "ny": 8, "lx_m": 1.0, "ly_m": 1.0},
            "numerics": {"gamma": 1.6666667},
            "initial": {"type": "orszag_tang"},
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
            "initial": {"type": "orszag_tang"},
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

    def test_output_path_must_be_a_nonempty_string(self):
        for bad in ("", "   ", 123, None):
            with self.subTest(output_path=bad):
                with self.assertRaisesRegex(ValueError, "output_path"):
                    _deck(diagnostics=Diagnostics(output_path=bad)).validate()

    def test_parser_does_not_stringify_numeric_output_paths(self):
        data = {
            "domain": {"nx": 8, "ny": 8, "lx_m": 1.0, "ly_m": 1.0},
            "initial": {"type": "orszag_tang"},
            "diagnostics": {"output_path": 123},
        }
        with self.assertRaisesRegex(ValueError, "output_path"):
            parse(data)

    def test_negative_cadence_rejected(self):
        with self.assertRaises(ValueError):
            _deck(diagnostics=Diagnostics(cadence=-1)).validate()

    def test_cadence_must_be_an_exact_integer(self):
        for bad in (1.0, 1.5, True):
            with self.subTest(cadence=bad):
                with self.assertRaises(ValueError):
                    _deck(diagnostics=Diagnostics(cadence=bad)).validate()

    def test_divb_flag_parsed(self):
        deck = parse({
            "domain": {"nx": 8, "ny": 8, "lx_m": 1.0, "ly_m": 1.0},
            "numerics": {"gamma": 1.6666667},
            "initial": {"type": "rotor"},
            "time": {"dt_s": "auto", "steps": 2},
            "diagnostics": {"output_path": "out.npz", "divb": True},
        })
        self.assertTrue(deck.diagnostics.divb)

    def test_divb_requires_a_real_boolean(self):
        with self.assertRaises(ValueError):
            _deck(diagnostics=Diagnostics(divb="false")).validate()


if __name__ == "__main__":
    unittest.main()
