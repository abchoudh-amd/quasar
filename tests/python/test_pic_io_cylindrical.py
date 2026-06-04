"""Unit tests for cylindrical (axisymmetric r-z) PIC deck validation.

Covers the ``geometry`` field on ``PicDeck`` and ``_validate_cylindrical``:
the geometry value whitelist, the start-at-r=0 inner-radius rule (origin_x_m
must be exactly 0; finite/annular inner radii are rejected), the 2nd-order
FDTD requirement, and the rejection of a periodic outer-radius (x_hi) wall for
both field and particle boundaries while leaving the inner-radius (x_lo)
boundary alone (the C++ solver auto-replaces it with the on-axis closure).

Also covers the cylindrical CFL guard in ``quasar.pic.cli.prepare_run``: an
explicit ``time.dt_s`` above the cylindrical (r-z) CFL limit is rejected, while
the ``auto`` path yields a CFL-safe dt. Those two tests build the C++ solver and
therefore require a ROCm device; they skip cleanly when ``_core`` /the device is
unavailable.

Finally, covers the cylindrical physical-component deck interface (parsed via
``parse``, no device needed): the external uniform ``B_rzphi`` physical-axis key
and its plane-dependent translation into the slot-ordered ``uniform_b``, the
mutual exclusion with the raw ``B_T`` key, the cylindrical-only restriction of
``B_rzphi``, and the ``fields.initial.component`` physical names (Er/Ez/Ephi ->
ex/ey/ez slots) reinterpreted only in cylindrical mode.
"""

import unittest

from quasar.pic.io import (
    BoundaryConfig,
    Domain,
    Numerics,
    PicDeck,
    Species,
    SpeciesInitial,
    Time,
    parse,
)


def _species() -> Species:
    return Species(name="electron", charge_C=-1.602176634e-19,
                   mass_kg=9.1093837015e-31, n_particles=128,
                   initial=SpeciesInitial())


def _cyl_boundary() -> BoundaryConfig:
    """A boundary config valid for a cylindrical deck: x_lo periodic (auto-axis),
    x_hi non-periodic (outer wall), y sides periodic."""
    return BoundaryConfig(
        particle=("periodic", "absorbing", "periodic", "periodic"),
        field=("periodic", "pec", "periodic", "periodic"),
    )


def _cyl_deck(**overrides) -> PicDeck:
    """A cylindrical deck that is otherwise valid (2nd-order FDTD, origin_x_m==0,
    non-periodic outer-radius walls)."""
    base = dict(
        domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0, origin_x_m=0.0),
        numerics=Numerics(fdtd_order=2, shape="cic"),
        species=[_species()],
        time=Time(dt_s=1.0e-12, steps=8),
        boundary=_cyl_boundary(),
        geometry="cylindrical",
    )
    base.update(overrides)
    return PicDeck(**base)


class GeometryDefaultTests(unittest.TestCase):

    def test_geometry_defaults_to_cartesian(self):
        deck = PicDeck(
            domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0),
            numerics=Numerics(fdtd_order=2, shape="cic"),
            species=[_species()],
            time=Time(dt_s=1.0e-12, steps=8),
        )
        self.assertEqual(deck.geometry, "cartesian")

    def test_parse_geometry_defaults_when_absent(self):
        deck = parse({
            "units": "SI",
            "domain": {"nx": 4, "ny": 4, "lx_m": 1.0, "ly_m": 1.0},
            "species": [],
            "external_field": {
                "evaluator": {"type": "uniform", "B_T": [0.0, 0.0, 1.0]}},
            "time": {"dt_s": "auto", "steps": 1},
        })
        self.assertEqual(deck.geometry, "cartesian")


class CylindricalValidTests(unittest.TestCase):

    def test_valid_cylindrical_deck_passes(self):
        _cyl_deck().validate()

    def test_parse_geometry_cylindrical_from_yaml(self):
        deck = parse({
            "units": "SI",
            "geometry": "cylindrical",
            "domain": {"nx": 8, "ny": 8, "lx_m": 1.0, "ly_m": 1.0,
                       "origin_x_m": 0.0},
            "numerics": {"fdtd_order": 2, "shape": "cic"},
            "species": [],
            "external_field": {
                "evaluator": {"type": "uniform", "B_T": [0.0, 0.0, 1.0]}},
            "time": {"dt_s": "auto", "steps": 1},
            "boundary": {"field": ["periodic", "pec", "periodic", "periodic"],
                         "particle": ["periodic", "absorbing", "periodic",
                                      "periodic"]},
        })
        self.assertEqual(deck.geometry, "cylindrical")


class GeometryValueTests(unittest.TestCase):

    def test_invalid_geometry_rz_rejected(self):
        with self.assertRaisesRegex(
                ValueError,
                r"geometry must be 'cartesian' or 'cylindrical'"):
            _cyl_deck(geometry="rz").validate()

    def test_invalid_geometry_spherical_rejected(self):
        with self.assertRaisesRegex(
                ValueError,
                r"geometry must be 'cartesian' or 'cylindrical'"):
            _cyl_deck(geometry="spherical").validate()


class CylindricalInnerRadiusTests(unittest.TestCase):

    def test_negative_origin_x_rejected(self):
        # A non-zero inner radius (negative included) is rejected with the
        # unified "must be 0" message: the m=0 on-axis scheme requires the
        # radial domain to start exactly at r=0.
        with self.assertRaisesRegex(
                ValueError,
                r"geometry 'cylindrical': domain\.origin_x_m must be 0 .*"
                r"radial domain to start at r=0"):
            _cyl_deck(
                domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0,
                              origin_x_m=-0.5)).validate()

    def test_zero_origin_x_accepted(self):
        _cyl_deck(
            domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0,
                          origin_x_m=0.0)).validate()

    def test_positive_origin_x_rejected(self):
        # A finite (annular) inner radius is no longer supported: the m=0 scheme
        # pins r=0, so a positive origin_x_m is rejected just like a negative one.
        with self.assertRaisesRegex(
                ValueError,
                r"geometry 'cylindrical': domain\.origin_x_m must be 0 .*"
                r"finite inner radius / annular domains are not supported"):
            _cyl_deck(
                domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0,
                              origin_x_m=0.25)).validate()


class CylindricalFdtdOrderTests(unittest.TestCase):

    def test_fdtd_order_4_rejected(self):
        with self.assertRaisesRegex(
                ValueError,
                r"geometry 'cylindrical': numerics\.fdtd_order must be 2 "
                r"\(order-4 cylindrical axis closure is not supported yet\)"):
            _cyl_deck(
                numerics=Numerics(fdtd_order=4, shape="cic")).validate()

    def test_fdtd_order_2_accepted(self):
        _cyl_deck(numerics=Numerics(fdtd_order=2, shape="cic")).validate()


class CylindricalOuterRadiusBoundaryTests(unittest.TestCase):

    def test_periodic_field_x_hi_rejected(self):
        with self.assertRaisesRegex(
                ValueError,
                r"geometry 'cylindrical': boundary\.field x_hi \(outer radius\) "
                r"must not be 'periodic'"):
            _cyl_deck(boundary=BoundaryConfig(
                particle=("periodic", "absorbing", "periodic", "periodic"),
                field=("periodic", "periodic", "periodic", "periodic"),
            )).validate()

    def test_periodic_particle_x_hi_rejected(self):
        with self.assertRaisesRegex(
                ValueError,
                r"geometry 'cylindrical': boundary\.particle x_hi "
                r"\(outer radius\) must not be 'periodic'"):
            _cyl_deck(boundary=BoundaryConfig(
                particle=("periodic", "periodic", "periodic", "periodic"),
                field=("periodic", "pec", "periodic", "periodic"),
            )).validate()

    def test_periodic_x_lo_accepted(self):
        # x_lo (inner radius, side index 0) periodic is the default and must be
        # accepted: the C++ solver auto-replaces it with the on-axis closure.
        _cyl_deck(boundary=BoundaryConfig(
            particle=("periodic", "absorbing", "periodic", "periodic"),
            field=("periodic", "pec", "periodic", "periodic"),
        )).validate()


class CartesianUnaffectedTests(unittest.TestCase):

    def test_cartesian_all_periodic_still_valid(self):
        # The new cylindrical rules must not leak into cartesian decks: an
        # all-periodic cartesian deck (also with order-4 FDTD) still validates.
        PicDeck(
            domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0, origin_x_m=-0.5),
            numerics=Numerics(fdtd_order=4, shape="tsc"),
            species=[_species()],
            time=Time(dt_s=1.0e-12, steps=8),
            boundary=BoundaryConfig(),  # all periodic
            geometry="cartesian",
        ).validate()


class CylindricalAxisOverrideTests(unittest.TestCase):

    def test_nondefault_field_x_lo_still_validates(self):
        # x_lo (inner radius) is don't-care at the Python/deck layer: a cylindrical
        # deck may set field x_lo to a non-default, non-'axis' value (here 'pec')
        # and it must still VALIDATE without error. The C++ solver overrides x_lo
        # with the on-axis closure at construction and emits a std::cerr warning
        # there -- that cerr warning is not capturable at this io/deck level, so
        # we only assert silent-at-validation here.
        _cyl_deck(boundary=BoundaryConfig(
            particle=("specular", "absorbing", "periodic", "periodic"),
            field=("pec", "pec", "periodic", "periodic"),
        )).validate()


class CylindricalCflGuardTests(unittest.TestCase):
    """The explicit-dt cylindrical CFL guard lives in cli.prepare_run, which
    builds the C++ solver. These tests need ``_core`` and a ROCm device; the CFL
    check itself runs before stepping, but solver construction still requires the
    device, so we skip cleanly when it is unavailable."""

    def _prepare(self, dt_s):
        from quasar.pic.cli import prepare_run  # imported as the cli does
        from quasar.pic._units import Units
        deck = _cyl_deck(time=Time(dt_s=dt_s, steps=4))
        deck.validate()
        return prepare_run(deck, Units(deck))

    def _cyl_limit(self):
        from quasar.pic.cli import _cyl_cfl_limit_internal
        from quasar.pic._units import Units
        deck = _cyl_deck(time=Time(dt_s="auto", steps=4))
        deck.validate()
        return _cyl_cfl_limit_internal(deck.domain, Units(deck))

    def test_explicit_dt_above_cyl_cfl_rejected(self):
        # An explicit dt_s far above the cylindrical (r-z) CFL limit (1 s vs a
        # ~ns grid limit) must raise ValueError naming the cylindrical CFL /
        # stability limit.
        try:
            with self.assertRaisesRegex(
                    ValueError,
                    r"cylindrical \(r-z\) CFL stability limit"):
                self._prepare(1.0)
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"solver build unavailable (no _core/device): {exc}")

    def test_auto_dt_within_cyl_cfl_limit(self):
        # The 'auto' path is CFL-safe by construction: the resolved internal dt
        # must not exceed the cylindrical CFL limit.
        try:
            _solver, _idx, dt, _dt_si = self._prepare("auto")
            limit = self._cyl_limit()
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"solver build unavailable (no _core/device): {exc}")
        self.assertLessEqual(dt, limit)


def _cyl_uniform_b_deck(plane="xy", **ev_extra) -> dict:
    """A minimal valid cylindrical deck dict with a uniform external B field,
    suitable for ``parse``. ``ev_extra`` is merged into the evaluator mapping
    (e.g. B_rzphi / B_T)."""
    evaluator = {"type": "uniform"}
    evaluator.update(ev_extra)
    return {
        "units": "SI",
        "geometry": "cylindrical",
        "plane": plane,
        "domain": {"nx": 8, "ny": 8, "lx_m": 1.0, "ly_m": 1.0,
                   "origin_x_m": 0.0},
        "numerics": {"fdtd_order": 2, "shape": "cic"},
        "species": [],
        "external_field": {"evaluator": evaluator},
        "time": {"dt_s": "auto", "steps": 1},
        "boundary": {"field": ["periodic", "pec", "periodic", "periodic"],
                     "particle": ["periodic", "absorbing", "periodic",
                                  "periodic"]},
    }


def _cyl_seed_deck(component, geometry="cylindrical") -> dict:
    """A minimal valid deck dict with a fields.initial seed of the given
    component, parametrized on geometry (cylindrical needs the axis-safe
    boundary/origin; cartesian uses defaults)."""
    base = {
        "units": "normalized",
        "geometry": geometry,
        "domain": {"nx": 8, "ny": 8, "lx_m": 1.0, "ly_m": 1.0,
                   "origin_x_m": 0.0},
        "numerics": {"fdtd_order": 2, "shape": "cic"},
        "species": [],
        "fields": {"initial": {"type": "seed_perturbation",
                               "component": component, "amplitude": 1.0e-4}},
        "time": {"dt_s": "auto", "steps": 1},
    }
    if geometry == "cylindrical":
        base["boundary"] = {
            "field": ["periodic", "pec", "periodic", "periodic"],
            "particle": ["periodic", "absorbing", "periodic", "periodic"]}
    return base


class CylindricalUniformBRzPhiTests(unittest.TestCase):

    def test_brzphi_xy_matches_b_t_axial(self):
        # plane xy: physical B_rzphi=[0, B_z, 0] (axial only) must resolve to the
        # SAME slot-ordered uniform_b as raw B_T=[0, B_z, 0], i.e. axial B_z in
        # the by slot (index 1). The xy lab->slot map is the identity.
        from_rzphi = parse(_cyl_uniform_b_deck(
            plane="xy", B_rzphi=[0.0, 1.0, 0.0]))
        from_bt = parse(_cyl_uniform_b_deck(
            plane="xy", B_T=[0.0, 1.0, 0.0]))
        self.assertEqual(from_rzphi.external_field.uniform_b, (0.0, 1.0, 0.0))
        self.assertEqual(from_rzphi.external_field.uniform_b,
                         from_bt.external_field.uniform_b)

    def test_brzphi_xz_permutation(self):
        # plane xz: physical B_rzphi=[B_r, B_z, B_phi] resolves to the lab/slot
        # vector [B_r, -B_phi, B_z] (the inverse of the xz lab->slot map). Use
        # three distinct nonzero values to catch any wrong permutation/sign.
        deck = parse(_cyl_uniform_b_deck(
            plane="xz", B_rzphi=[2.0, 3.0, 5.0]))  # [B_r, B_z, B_phi]
        self.assertEqual(deck.external_field.uniform_b, (2.0, -5.0, 3.0))

    def test_brzphi_and_b_t_together_rejected(self):
        with self.assertRaisesRegex(
                ValueError,
                r"external_field\.evaluator: give either B_rzphi \(physical "
                r"axes\) or B_T \(storage slots\), not both"):
            parse(_cyl_uniform_b_deck(
                plane="xy", B_rzphi=[0.0, 1.0, 0.0], B_T=[0.0, 1.0, 0.0]))

    def test_brzphi_on_cartesian_rejected(self):
        # B_rzphi is a cylindrical-only physical-axis key; on a cartesian deck it
        # must be rejected (the user should spell raw B_T instead).
        data = {
            "units": "SI",
            "geometry": "cartesian",
            "domain": {"nx": 8, "ny": 8, "lx_m": 1.0, "ly_m": 1.0},
            "numerics": {"fdtd_order": 2, "shape": "cic"},
            "species": [],
            "external_field": {
                "evaluator": {"type": "uniform", "B_rzphi": [0.0, 1.0, 0.0]}},
            "time": {"dt_s": "auto", "steps": 1},
        }
        with self.assertRaisesRegex(
                ValueError,
                r"external_field\.evaluator\.B_rzphi \(physical r,z,phi axes\) "
                r"is only valid for geometry 'cylindrical'; use B_T for "
                r"cartesian decks"):
            parse(data)

    def test_raw_b_t_still_loads_on_cylindrical(self):
        # Backward compat: a cylindrical deck spelling raw B_T still parses and
        # the slot vector passes through unchanged.
        deck = parse(_cyl_uniform_b_deck(plane="xy", B_T=[0.1, 0.2, 0.3]))
        self.assertEqual(deck.external_field.uniform_b, (0.1, 0.2, 0.3))


class CylindricalSeedComponentTests(unittest.TestCase):

    def test_ez_resolves_to_ey_slot(self):
        # Cylindrical: physical Ez is the AXIAL field and resolves to the ey slot
        # (NOT the literal ez storage slot).
        deck = parse(_cyl_seed_deck("Ez"))
        self.assertEqual(deck.fields.initial.component, "ey")

    def test_ephi_resolves_to_ez_slot(self):
        # Physical azimuthal Ephi resolves to the ez slot.
        deck = parse(_cyl_seed_deck("Ephi"))
        self.assertEqual(deck.fields.initial.component, "ez")

    def test_er_resolves_to_ex_slot(self):
        # Physical radial Er resolves to the ex slot.
        deck = parse(_cyl_seed_deck("Er"))
        self.assertEqual(deck.fields.initial.component, "ex")

    def test_raw_slot_name_passthrough_on_cylindrical(self):
        # Backward compat: a raw storage-slot seed name (ey) is accepted verbatim
        # in cylindrical mode (no reinterpretation of an already-slot name).
        deck = parse(_cyl_seed_deck("ey"))
        self.assertEqual(deck.fields.initial.component, "ey")

    def test_cartesian_slot_name_taken_literally(self):
        # Cartesian: no physical reinterpretation -- the literal slot ez stays ez
        # (it is NOT mapped to ephi/anything else).
        deck = parse(_cyl_seed_deck("ez", geometry="cartesian"))
        self.assertEqual(deck.fields.initial.component, "ez")


if __name__ == "__main__":
    unittest.main()
