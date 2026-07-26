"""Unit tests for cylindrical (axisymmetric r-z) PIC deck validation.

Covers the ``geometry`` field on ``PicDeck`` and ``_validate_cylindrical``:
the geometry value whitelist, axis and annular inner-radius rules, both supported
FDTD orders, and rejection of periodic radial walls where they are unphysical.

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
    Fields,
    FieldsInitial,
    Numerics,
    PicDeck,
    Species,
    SpeciesInitial,
    Time,
    _parse_fields,
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
                "evaluator": {"type": "uniform", "B_T": [0.0, 1.0, 0.0]}},
            "time": {"dt_s": "auto", "steps": 1},
        })
        self.assertEqual(deck.geometry, "cartesian")


class CylindricalValidTests(unittest.TestCase):

    def test_valid_cylindrical_deck_passes(self):
        _cyl_deck().validate()

    def test_radial_bessel_seed_is_bounded_by_the_mesh_spectrum(self):
        _cyl_deck(fields=Fields(initial=FieldsInitial(
            type="seed_perturbation", component="Ey", mode=(8, 0)))).validate()

        with self.assertRaisesRegex(ValueError, "radial Nyquist"):
            _cyl_deck(fields=Fields(initial=FieldsInitial(
                type="seed_perturbation", component="Ey",
                mode=(9, 0)))).validate()

    def test_radial_bessel_seed_requires_the_outer_pec_wall(self):
        boundary = BoundaryConfig(
            particle=("axis", "absorbing", "periodic", "periodic"),
            field=("axis", "outflow", "periodic", "periodic"))
        with self.assertRaisesRegex(ValueError, "PEC outer-radius"):
            _cyl_deck(
                boundary=boundary,
                fields=Fields(initial=FieldsInitial(
                    type="seed_perturbation", component="Ey",
                    mode=(1, 0)))).validate()

    def test_parse_geometry_cylindrical_from_yaml(self):
        deck = parse({
            "units": "SI",
            "geometry": "cylindrical",
            "domain": {"nx": 8, "ny": 8, "lx_m": 1.0, "ly_m": 1.0,
                       "origin_x_m": 0.0},
            "numerics": {"fdtd_order": 2, "shape": "cic"},
            "species": [],
            "external_field": {
                "evaluator": {"type": "uniform", "B_T": [0.0, 1.0, 0.0]}},
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
        with self.assertRaisesRegex(
                ValueError,
                r"geometry 'cylindrical': domain\.origin_x_m must be >= 0"):
            _cyl_deck(
                domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0,
                              origin_x_m=-0.5)).validate()

    def test_zero_origin_x_accepted(self):
        _cyl_deck(
            domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0,
                          origin_x_m=0.0)).validate()

    def test_positive_origin_x_annulus_accepted_with_walls(self):
        _cyl_deck(
            domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0,
                          origin_x_m=0.25),
            boundary=BoundaryConfig(
                particle=("specular", "absorbing", "periodic", "periodic"),
                field=("pec", "pec", "periodic", "periodic"))).validate()

    def test_annulus_periodic_inner_radius_rejected(self):
        with self.assertRaisesRegex(ValueError, r"annulus: boundary\.field x_lo"):
            _cyl_deck(
                domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0,
                              origin_x_m=0.25)).validate()


class CylindricalFdtdOrderTests(unittest.TestCase):

    def test_fdtd_order_4_accepted(self):
        _cyl_deck(numerics=Numerics(fdtd_order=4, shape="cic")).validate()

    def test_fdtd_order_2_accepted(self):
        _cyl_deck(numerics=Numerics(fdtd_order=2, shape="cic")).validate()

    def test_fdtd_order_4_rejects_one_cell_axis(self):
        for nx, ny in ((1, 8), (8, 1)):
            with self.subTest(nx=nx, ny=ny):
                with self.assertRaisesRegex(
                        ValueError, r"fdtd_order 4 requires.*at least 2"):
                    _cyl_deck(
                        domain=Domain(
                            nx=nx, ny=ny, lx_m=1.0, ly_m=1.0,
                            origin_x_m=0.0),
                        numerics=Numerics(fdtd_order=4, shape="cic"),
                    ).validate()


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

    def test_explicit_nonaxis_wall_at_r_zero_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"at cylindrical r=0"):
            _cyl_deck(boundary=BoundaryConfig(
                particle=("axis", "absorbing", "periodic", "periodic"),
                field=("pec", "pec", "periodic", "periodic"),
            )).validate()

    def test_explicit_axis_at_r_zero_is_accepted(self):
        _cyl_deck(boundary=BoundaryConfig(
            particle=("axis", "absorbing", "periodic", "periodic"),
            field=("axis", "pec", "periodic", "periodic"),
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


def _cyl_uniform_b_deck(plane="xy", origin_x_m=0.0, **ev_extra) -> dict:
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
                   "origin_x_m": origin_x_m},
        "numerics": {"fdtd_order": 2, "shape": "cic"},
        "species": [],
        "external_field": {"evaluator": evaluator},
        "time": {"dt_s": "auto", "steps": 1},
        "boundary": {"field": ["periodic" if origin_x_m == 0.0 else "pec",
                                 "pec", "periodic", "periodic"],
                     "particle": ["periodic" if origin_x_m == 0.0 else "specular",
                                  "absorbing", "periodic",
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
        # The parser still performs the plane map before validation.  A pure
        # axial physical field is the only globally axisymmetric vector that a
        # Cartesian lab-uniform evaluator can represent.
        deck = parse(_cyl_uniform_b_deck(
            plane="xz", origin_x_m=0.25,
            B_rzphi=[0.0, 3.0, 0.0]))
        self.assertEqual(deck.external_field.uniform_b, (0.0, 0.0, 3.0))

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
        # Backward compat: the raw spelling remains available for a regular axial
        # field; non-axial constants are rejected regardless of spelling.
        deck = parse(_cyl_uniform_b_deck(plane="xy", B_T=[0.0, 0.2, 0.0]))
        self.assertEqual(deck.external_field.uniform_b, (0.0, 0.2, 0.0))

    def test_uniform_radial_and_toroidal_b_rejected(self):
        for value in ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]):
            with self.subTest(B_rzphi=value), self.assertRaisesRegex(
                    ValueError, r"only axial Bz; Br/Bphi require"):
                parse(_cyl_uniform_b_deck(B_rzphi=value))

    def test_on_axis_raw_non_axial_b_rejected(self):
        with self.assertRaisesRegex(
                ValueError, r"only axial Bz; Br/Bphi require"):
            parse(_cyl_uniform_b_deck(plane="xy", B_T=[1.0, 0.0, 1.0]))

    def test_annular_lab_uniform_radial_and_toroidal_b_rejected(self):
        with self.assertRaisesRegex(ValueError, r"only axial Bz; Br/Bphi require"):
            parse(_cyl_uniform_b_deck(
                origin_x_m=0.25, B_rzphi=[1.0, 0.0, 0.0]))
        with self.assertRaisesRegex(ValueError, r"only axial Bz; Br/Bphi require"):
            parse(_cyl_uniform_b_deck(
                origin_x_m=0.25, B_rzphi=[0.0, 0.0, 1.0]))

    def test_on_axis_uniform_non_axial_e_rejected(self):
        with self.assertRaisesRegex(
                ValueError, r"only axial Ez; Er/Ephi require"):
            parse(_cyl_uniform_b_deck(
                plane="xz", B_rzphi=[0.0, 1.0, 0.0],
                E_V_per_m=[1.0, 1.0, 0.0]))


class CylindricalSeedComponentTests(unittest.TestCase):

    def test_ez_resolves_to_ey_slot(self):
        # Cylindrical: physical Ez is the AXIAL field and resolves to the ey slot
        # (NOT the literal ez storage slot).
        deck = parse(_cyl_seed_deck("Ez"))
        self.assertEqual(deck.fields.initial.component, "ey")

    def test_ephi_resolves_to_ez_slot(self):
        # The parser resolves the physical name before validation, while the
        # current cylindrical seed implementation deliberately rejects this
        # unsupported non-axial eigenmode.
        fields = _parse_fields(
            _cyl_seed_deck("Ephi")["fields"], geometry="cylindrical")
        self.assertEqual(fields.initial.component, "ez")
        with self.assertRaisesRegex(ValueError, "supports only physical axial Ez"):
            parse(_cyl_seed_deck("Ephi"))

    def test_er_resolves_to_ex_slot(self):
        fields = _parse_fields(
            _cyl_seed_deck("Er")["fields"], geometry="cylindrical")
        self.assertEqual(fields.initial.component, "ex")
        with self.assertRaisesRegex(ValueError, "supports only physical axial Ez"):
            parse(_cyl_seed_deck("Er"))

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
