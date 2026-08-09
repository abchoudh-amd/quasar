import math
import unittest

from quasar.pic.io import (
    BoundaryConfig,
    Diagnostics,
    Domain,
    ExternalField,
    Fields,
    FieldsInitial,
    Normalization,
    Numerics,
    PicDeck,
    Species,
    SpeciesInitial,
    Time,
    VelocityPerturbation,
    _parse_boundary,
    _parse_fields,
    _parse_time,
    parse,
)


def _species() -> Species:
    return Species(name="electron", charge_C=-1.602176634e-19,
                   mass_kg=9.1093837015e-31, n_particles=128,
                   initial=SpeciesInitial())


def _deck(**overrides) -> PicDeck:
    base = dict(
        domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0),
        numerics=Numerics(fdtd_order=4, shape="tsc"),
        species=[_species()],
        time=Time(dt_s=1.0e-12, steps=8),
    )
    base.update(overrides)
    return PicDeck(**base)


class PicIoTests(unittest.TestCase):

    def test_schema_validation_minimal(self):
        _deck().validate()

    def test_internal_tile_boundary_is_not_public_deck_vocabulary(self):
        boundary = BoundaryConfig(
            particle=("internal",) * 4, field=("internal",) * 4)
        with self.assertRaisesRegex(ValueError, "reserved.*distributed"):
            _deck(boundary=boundary).validate()

    def test_required_nghost_binding_rejects_unsupported_orders(self):
        from quasar import _core

        self.assertEqual(_core.pic.required_nghost(2), 1)
        self.assertEqual(_core.pic.required_nghost(4), 2)
        for order in (0, 1, 3, 6):
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    _core.pic.required_nghost(order)

    def test_loader_rejects_duplicate_yaml_keys(self):
        import tempfile
        from pathlib import Path

        from quasar.pic.io import load

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.yaml"
            path.write_text("units: SI\nunits: SI\n")
            with self.assertRaisesRegex(
                    ValueError, r"duplicate YAML key 'units'.*line 2"):
                load(path)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            _deck(numerics=Numerics(shape="bad")).validate()

    def test_plane_defaults_to_xy(self):
        self.assertEqual(_deck().plane, "xy")

    def test_plane_xz_accepted(self):
        _deck(plane="xz").validate()

    def test_invalid_plane_rejected(self):
        with self.assertRaises(ValueError):
            _deck(plane="yz").validate()

    def test_cylindrical_plane_wave_rejected_as_non_axisymmetric(self):
        deck = _deck(
            geometry="cylindrical",
            fields=Fields(initial=FieldsInitial(
                type="seed_em_wave", component="Ez", mode=(1, 0))))
        with self.assertRaisesRegex(ValueError, "not a regular axisymmetric mode"):
            deck.validate()

    def test_em_wave_requires_a_strictly_sub_nyquist_mode(self):
        for nx, mode in ((8, 4), (7, 4)):
            with self.subTest(nx=nx, mode=mode):
                deck = _deck(
                    domain=Domain(nx=nx, ny=8, lx_m=1.0, ly_m=1.0),
                    fields=Fields(initial=FieldsInitial(
                        type="seed_em_wave", component="Ez", mode=(mode, 0))))
                with self.assertRaisesRegex(ValueError, "strictly below"):
                    deck.validate()

        _deck(fields=Fields(initial=FieldsInitial(
            type="seed_em_wave", component="Ez", mode=(3, 0)))).validate()

    def test_tm_cavity_requires_resolved_positive_mode_and_four_pec_walls(self):
        walls = BoundaryConfig(
            particle=("specular",) * 4, field=("pec",) * 4)
        _deck(
            fields=Fields(initial=FieldsInitial(
                type="seed_tm_cavity", component="Ez", mode=(2, 3))),
            boundary=walls).validate()

        invalid = (
            (FieldsInitial(type="seed_tm_cavity", component="Ey", mode=(1, 1)),
             walls, "component"),
            (FieldsInitial(type="seed_tm_cavity", component="Ez", mode=(1, 0)),
             walls, "m,n"),
            (FieldsInitial(type="seed_tm_cavity", component="Ez", mode=(9, 1)),
             walls, "spectrum"),
            (FieldsInitial(type="seed_tm_cavity", component="Ez", mode=(1, 1)),
             BoundaryConfig(), "PEC"),
        )
        for initial, boundary, message in invalid:
            with self.subTest(initial=initial, boundary=boundary):
                with self.assertRaisesRegex(ValueError, message):
                    _deck(fields=Fields(initial=initial),
                          boundary=boundary).validate()

    def test_tm_cavity_rejects_cylindrical_geometry(self):
        walls = BoundaryConfig(
            particle=("specular",) * 4, field=("pec",) * 4)
        with self.assertRaisesRegex(ValueError, "Cartesian rectangular"):
            _deck(
                geometry="cylindrical",
                fields=Fields(initial=FieldsInitial(
                    type="seed_tm_cavity", component="Ez", mode=(1, 1))),
                boundary=walls).validate()

    def test_parse_plane_from_yaml(self):
        deck = parse({
            "units": "SI",
            "plane": "xz",
            "domain": {"nx": 4, "ny": 4, "lx_m": 1.0, "ly_m": 1.0},
            "species": [],
            "external_field": {
                "evaluator": {"type": "uniform", "B_T": [0.0, 0.0, 1.0]}},
            "time": {"dt_s": "auto", "steps": 1},
        })
        self.assertEqual(deck.plane, "xz")

    def test_parse_plane_defaults_when_absent(self):
        deck = parse({
            "units": "SI",
            "domain": {"nx": 4, "ny": 4, "lx_m": 1.0, "ly_m": 1.0},
            "species": [],
            "external_field": {
                "evaluator": {"type": "uniform", "B_T": [0.0, 0.0, 1.0]}},
            "time": {"dt_s": "auto", "steps": 1},
        })
        self.assertEqual(deck.plane, "xy")

    def test_parse_optional_t_end(self):
        time = _parse_time({"dt_s": 0.1, "steps": 8, "t_end_s": 0.25})
        self.assertEqual(time.t_end_s, 0.25)

    def test_parse_velocity_perturbation(self):
        deck = parse({
            "units": "normalized",
            "domain": {"nx": 8, "ny": 4, "lx_m": 1.0, "ly_m": 0.5},
            "species": [{
                "name": "beam", "charge_C": -1.0, "mass_kg": 1.0,
                "n_particles": 32,
                "initial": {
                    "temperature_eV": 0.0,
                    "velocity_perturbation": {
                        "amplitude_v": [1.0e-5, 0.0, 0.0],
                        "mode": [2, -1], "phase_rad": 0.25},
                },
            }],
            "time": {"dt_s": "auto", "steps": 1},
        })
        perturbation = deck.species[0].initial.velocity_perturbation
        self.assertEqual(perturbation.amplitude_v, (1.0e-5, 0.0, 0.0))
        self.assertEqual(perturbation.mode, (2, -1))
        self.assertEqual(perturbation.phase_rad, 0.25)

    def test_velocity_perturbation_requires_finite_nonzero_mode_and_amplitude(self):
        invalid = (
            VelocityPerturbation((0.0, 0.0, 0.0), (1, 0)),
            VelocityPerturbation((1.0e-5, 0.0, 0.0), (0, 0)),
            VelocityPerturbation((float("nan"), 0.0, 0.0), (1, 0)),
            VelocityPerturbation((1.0e-5, 0.0, 0.0), (1.5, 0)),
        )
        for perturbation in invalid:
            with self.subTest(perturbation=perturbation):
                species = Species(
                    name="beam", charge_C=-1.0, mass_kg=1.0,
                    n_particles=32,
                    initial=SpeciesInitial(
                        temperature_eV=0.0,
                        velocity_perturbation=perturbation))
                with self.assertRaises(ValueError):
                    _deck(species=[species], units="normalized").validate()

    def test_velocity_perturbation_must_be_strictly_below_grid_nyquist(self):
        for mode in ((4, 0), (-4, 0), (0, 4), (5, 0)):
            with self.subTest(mode=mode):
                species = Species(
                    name="beam", charge_C=-1.0, mass_kg=1.0,
                    n_particles=32,
                    initial=SpeciesInitial(
                        temperature_eV=0.0,
                        velocity_perturbation=VelocityPerturbation(
                            (1.0e-5, 0.0, 0.0), mode)))
                with self.assertRaisesRegex(ValueError, "Nyquist"):
                    _deck(species=[species], units="normalized").validate()

        # The highest mode below Nyquist on an odd-sized grid remains valid.
        species = Species(
            name="beam", charge_C=-1.0, mass_kg=1.0, n_particles=32,
            initial=SpeciesInitial(
                temperature_eV=0.0,
                velocity_perturbation=VelocityPerturbation(
                    (1.0e-5, 0.0, 0.0), (2, -2))))
        _deck(
            domain=Domain(nx=5, ny=5, lx_m=1.0, ly_m=1.0),
            species=[species], units="normalized").validate()

    def test_cylindrical_axis_velocity_perturbation_enforces_vector_parity(self):
        def cylindrical(perturbation):
            species = Species(
                name="beam", charge_C=0.0, mass_kg=1.0, n_particles=32,
                initial=SpeciesInitial(
                    temperature_eV=0.0,
                    velocity_perturbation=perturbation))
            return _deck(
                geometry="cylindrical",
                boundary=BoundaryConfig(
                    particle=("axis", "specular", "periodic", "periodic"),
                    field=("axis", "pec", "periodic", "periodic")),
                species=[species], units="normalized")

        # vr/vphi must be odd in r: a radial sine with zero phase is valid.
        cylindrical(VelocityPerturbation(
            (1.0e-5, 0.0, -2.0e-5), (1, 0), 2.0 * math.pi)).validate()
        # Axial velocity must be even: a radial cosine (sine phase pi/2) is valid.
        cylindrical(VelocityPerturbation(
            (0.0, 1.0e-5, 0.0), (1, 0), 0.5 * math.pi)).validate()
        # An axial-only z wave is independent of r and therefore regular.
        cylindrical(VelocityPerturbation(
            (0.0, 1.0e-5, 0.0), (0, 1), 0.3)).validate()

        invalid = (
            VelocityPerturbation((1.0e-5, 0.0, 0.0), (0, 1), 0.0),
            VelocityPerturbation((0.0, 1.0e-5, 0.0), (1, 0), 0.0),
            VelocityPerturbation((1.0e-5, 1.0e-5, 0.0), (1, 0), 0.0),
        )
        for perturbation in invalid:
            with self.subTest(perturbation=perturbation):
                with self.assertRaisesRegex(ValueError, "r=0 domain"):
                    cylindrical(perturbation).validate()

    def test_time_rejects_unknown_key(self):
        with self.assertRaisesRegex(ValueError, "unknown key"):
            _parse_time({"dt_s": "auto", "steps": 8, "t_end": 1.0})

    def test_requires_species(self):
        with self.assertRaises(ValueError):
            _deck(species=[]).validate()

    def test_external_field_unsupported_evaluator(self):
        with self.assertRaises(ValueError):
            _deck(external_field=ExternalField(
                evaluator_type="unsupported",
                conductors=[{"x": 1}])).validate()

    def test_external_field_requires_conductors(self):
        with self.assertRaises(ValueError):
            _deck(external_field=ExternalField(
                evaluator_type="biot_savart",
                conductors=[])).validate()

    def test_external_field_rejects_unknown_or_inapplicable_keys(self):
        base = {
            "units": "SI",
            "domain": {"nx": 4, "ny": 4, "lx_m": 1.0, "ly_m": 1.0},
            "external_field": {
                "evaluator": {"type": "uniform", "B_typo": [0, 0, 1]}},
            "time": {"dt_s": "auto", "steps": 1},
        }
        with self.assertRaises(ValueError):
            parse(base)

    def test_gradient_evaluator_requires_matrix(self):
        with self.assertRaises(ValueError):
            _deck(external_field=ExternalField(evaluator_type="gradient")).validate()

    def test_current_filter_passes_must_be_positive(self):
        with self.assertRaises(ValueError):
            _deck(numerics=Numerics(
                current_filter=[{"type": "compensated_binomial", "passes": 0}]
            )).validate()

    def test_rejects_ambiguous_current_filter_pass_aliases(self):
        with self.assertRaisesRegex(ValueError, "multiple aliases"):
            _deck(numerics=Numerics(current_filter=[{
                "type": "compensated_binomial", "passes": 1, "n_passes": 1,
            }])).validate()

    def test_rejects_ambiguous_external_field_aliases(self):
        cases = (
            {"type": "uniform", "B_T": [1, 0, 0], "B": [1, 0, 0]},
            {"type": "uniform", "E_V_per_m": [0, 1, 0], "E": [0, 1, 0]},
            {"type": "dipole", "moment_Am2": [0, 0, 1],
             "moment": [0, 0, 1]},
            {"type": "dipole", "moment_Am2": [0, 0, 1],
             "origin_xyz_m": [0, 0, 0], "origin": [0, 0, 0]},
            {"type": "gradient",
             "grad_T_per_m": [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
             "gradient": [[1, 0, 0], [0, -1, 0], [0, 0, 0]]},
            {"type": "gradient",
             "grad_T_per_m": [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
             "B0_T": [0, 0, 0], "b0": [0, 0, 0]},
            {"type": "gradient",
             "grad_T_per_m": [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
             "origin_xyz_m": [0, 0, 0], "origin": [0, 0, 0]},
            {"type": "file_grid", "path": "field.npz", "file": "field.npz"},
        )
        base = {
            "units": "SI",
            "domain": {"nx": 4, "ny": 4, "lx_m": 1.0, "ly_m": 1.0},
            "species": [],
            "time": {"dt_s": "auto", "steps": 1},
        }
        for evaluator in cases:
            with self.subTest(evaluator=evaluator):
                data = dict(base)
                data["external_field"] = {"evaluator": evaluator}
                with self.assertRaisesRegex(ValueError, "multiple aliases"):
                    parse(data)

    def test_parse_gradient_external_field_params(self):
        deck = parse({
            "units": "SI",
            "domain": {"nx": 4, "ny": 4, "lx_m": 1.0, "ly_m": 1.0},
            "external_field": {
                "evaluator": {
                    "type": "gradient",
                    "B0_T": [0.0, 0.0, 1.0],
                    "grad_T_per_m": [
                        [1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, -3.0],
                    ],
                    "origin_xyz_m": [0.1, 0.2, 0.3],
                },
            },
            "species": [],
            "time": {"dt_s": "auto", "steps": 1},
        })
        self.assertEqual(deck.external_field.gradient_b0, (0.0, 0.0, 1.0))
        self.assertEqual(deck.external_field.gradient_matrix[2], (0.0, 0.0, -3.0))
        self.assertEqual(deck.external_field.gradient_origin, (0.1, 0.2, 0.3))

    def test_parse_full_deck(self):
        data = {
            "units": "SI",
            "domain": {"nx": 16, "ny": 16, "lx_m": 0.1, "ly_m": 0.1},
            "numerics": {"fdtd_order": 2, "shape": "cic"},
            "external_field": {
                "evaluator": {
                    "type": "biot_savart",
                    "conductors": [{
                        "name": "loop",
                        "current_A": 1.0,
                        "geometry": {
                            "type": "circular_loop",
                            "center_xyz": [0, 0, 0],
                            "axis_xyz": [0, 0, 1],
                            "radius_m": 0.05,
                            "n_segments": 32,
                        },
                    }],
                },
            },
            "species": [{
                "name": "H+",
                "charge_C": 1.602176634e-19,
                "mass_kg": 1.67262192369e-27,
                "n_particles": 256,
                "initial": {
                    "distribution": "maxwellian_uniform",
                    "density_per_m3": 1.0e16,
                    "temperature_eV": 5.0,
                },
            }],
            "time": {"dt_s": "auto", "steps": 64},
            "diagnostics": {"output_path": "out.npz", "cadence": 16,
                             "fields": ["bz", "ex"], "per_species": True},
        }
        deck = parse(data)
        self.assertEqual(deck.domain.nx, 16)
        self.assertEqual(len(deck.species), 1)
        self.assertEqual(deck.species[0].name, "H+")
        self.assertEqual(deck.external_field.evaluator_type, "biot_savart")
        self.assertEqual(deck.time.dt_s, "auto")

    def test_parser_rejects_lossy_integer_coercions(self):
        base = {
            "units": "SI",
            "domain": {"nx": 4, "ny": 4, "lx_m": 1.0, "ly_m": 1.0},
            "external_field": {
                "evaluator": {"type": "uniform", "B_T": [0, 0, 1]}},
            "time": {"dt_s": "auto", "steps": 1},
        }
        for path, value in (("nx", 4.5), ("nx", True)):
            with self.subTest(path=path, value=value):
                data = {**base, "domain": {**base["domain"], path: value}}
                with self.assertRaises(ValueError):
                    parse(data)
        data = {**base, "time": {"dt_s": "auto", "steps": 1.5}}
        with self.assertRaises(ValueError):
            parse(data)


class DomainValidationTests(unittest.TestCase):

    def test_nonpositive_nx_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=Domain(nx=0, ny=8, lx_m=1.0, ly_m=1.0)).validate()

    def test_nonpositive_ny_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=Domain(nx=8, ny=-4, lx_m=1.0, ly_m=1.0)).validate()

    def test_nonpositive_length_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=Domain(nx=8, ny=8, lx_m=0.0, ly_m=1.0)).validate()

    def test_nonfinite_origin_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0,
                                origin_x_m=float("nan"))).validate()

    def test_actual_high_cell_center_cannot_collapse_onto_high_face(self):
        # Exact binary64 regression from Grid2D: x_hi-dx/2 is distinct from the
        # high face, while the accessor's x0+(nx-0.5)*dx rounds onto it.
        origin = float.fromhex("0x1.58be77ab34af0p-1002")
        length = float.fromhex("0x0.0000000373a5dp-1022")
        with self.assertRaisesRegex(ValueError, "coordinates collapse"):
            _deck(domain=Domain(
                nx=2, ny=8, lx_m=length, ly_m=1.0,
                origin_x_m=origin)).validate()


class ParseBoundaryTests(unittest.TestCase):

    def test_default_is_all_periodic(self):
        bc = _parse_boundary(None)
        self.assertEqual(bc.particle, ("periodic", "periodic", "periodic", "periodic"))

    def test_scalar_broadcasts_to_four_sides(self):
        bc = _parse_boundary({"particle": "specular"})
        self.assertEqual(bc.particle, ("specular", "specular", "specular", "specular"))

    def test_four_list_maps_per_side(self):
        bc = _parse_boundary(
            {"particle": ["periodic", "specular", "absorbing", "periodic"]})
        self.assertEqual(
            bc.particle, ("periodic", "specular", "absorbing", "periodic"))

    def test_wrong_length_list_raises(self):
        with self.assertRaises(ValueError):
            _parse_boundary({"particle": ["periodic", "specular"]})

    def test_non_string_non_list_raises(self):
        with self.assertRaises(ValueError):
            _parse_boundary({"particle": 42})


class BoundaryValidationTests(unittest.TestCase):

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            _deck(boundary=BoundaryConfig(
                particle=("periodic", "teleporting", "periodic", "periodic"))
            ).validate()

    def test_valid_mixed_kinds_pass(self):
        _deck(boundary=BoundaryConfig(
            particle=("specular", "absorbing", "periodic", "periodic"),
            field=("pec", "pec", "periodic", "periodic"))).validate()

    def test_unpaired_periodic_side_rejected(self):
        with self.assertRaisesRegex(ValueError, "periodic sides.*must be specified as a pair"):
            _deck(boundary=BoundaryConfig(
                particle=("periodic", "specular", "periodic", "periodic"),
                field=("pec", "pec", "periodic", "periodic"))).validate()


class MaxwellianBlockRegionTests(unittest.TestCase):

    def _block_species(self, **region) -> Species:
        init = SpeciesInitial(distribution="maxwellian_block", **region)
        return Species(name="e", charge_C=-1.0, mass_kg=1.0, n_particles=16,
                       initial=init)

    def test_missing_bounds_rejected(self):
        with self.assertRaises(ValueError):
            _deck(species=[self._block_species(
                region_x_min_m=0.0, region_x_max_m=1.0)]).validate()

    def test_min_ge_max_rejected(self):
        with self.assertRaises(ValueError):
            _deck(species=[self._block_species(
                region_x_min_m=1.0, region_x_max_m=1.0,
                region_y_min_m=0.0, region_y_max_m=1.0)]).validate()

    def test_valid_block_passes(self):
        _deck(species=[self._block_species(
            region_x_min_m=0.0, region_x_max_m=1.0,
            region_y_min_m=0.0, region_y_max_m=1.0)]).validate()

    def test_region_outside_domain_rejected(self):
        with self.assertRaises(ValueError):
            _deck(species=[self._block_species(
                region_x_min_m=-0.1, region_x_max_m=0.5,
                region_y_min_m=0.0, region_y_max_m=1.0)]).validate()


class ParseFieldsTests(unittest.TestCase):

    def test_none_yields_empty_fields(self):
        self.assertIsNone(_parse_fields(None).initial)
        self.assertIsNone(_parse_fields({}).initial)

    def test_scalar_mode_coerces_to_pair(self):
        f = _parse_fields({"initial": {"type": "seed_perturbation", "mode": 3}})
        self.assertEqual(f.initial.mode, (3, 0))

    def test_list_mode_maps_both_components(self):
        f = _parse_fields({"initial": {"type": "seed_em_wave", "mode": [2, 1]}})
        self.assertEqual(f.initial.mode, (2, 1))

    def test_one_element_list_mode_defaults_my(self):
        f = _parse_fields({"initial": {"type": "seed_em_wave", "mode": [2]}})
        self.assertEqual(f.initial.mode, (2, 0))

    def test_defaults_component_and_amplitude(self):
        f = _parse_fields({"initial": {"type": "seed_perturbation"}})
        self.assertEqual(f.initial.component, "Ey")
        self.assertEqual(f.initial.amplitude, 1.0e-4)
        self.assertEqual(f.initial.mode, (1, 0))

    def test_missing_type_raises(self):
        with self.assertRaises(ValueError):
            _parse_fields({"initial": {"component": "Ez"}})

    def test_modes_must_be_exact_integers_with_at_most_two_entries(self):
        for mode in (1.5, True, [1.5], [], [1, 2, 3]):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    _parse_fields({"initial": {
                        "type": "seed_perturbation", "mode": mode}})

    def test_deck_validation_rejects_ignored_or_invalid_seed_parameters(self):
        for seed in (
                FieldsInitial(type="unknown"),
                FieldsInitial(type="seed_perturbation", mode=(1, 1)),
                FieldsInitial(type="seed_perturbation", mode=(0, 0)),
                FieldsInitial(type="seed_perturbation", component="Ex"),
                FieldsInitial(type="seed_perturbation", component="Bx"),
                FieldsInitial(type="seed_em_wave", component="ex")):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    _deck(fields=Fields(initial=seed)).validate()


class ResourceCeilingTests(unittest.TestCase):

    def test_direct_dataclasses_reject_fractional_integer_fields(self):
        with self.assertRaises(ValueError):
            _deck(domain=Domain(nx=8.5, ny=8, lx_m=1.0, ly_m=1.0)).validate()
        with self.assertRaises(ValueError):
            _deck(time=Time(dt_s="auto", steps=2.5)).validate()
        with self.assertRaises(ValueError):
            _deck(diagnostics=Diagnostics(cadence=1.5)).validate()

    def test_oversized_grid_dim_rejected(self):
        from quasar.pic.io import MAX_GRID_DIM
        with self.assertRaises(ValueError):
            _deck(domain=Domain(nx=MAX_GRID_DIM + 1, ny=8,
                                lx_m=1.0, ly_m=1.0)).validate()

    def test_oversized_grid_cells_rejected(self):
        # nx and ny each pass the per-axis MAX_GRID_DIM check, but their product
        # exceeds MAX_GRID_CELLS, so the distinct cell-count branch must fire.
        from quasar.pic.io import MAX_GRID_CELLS, MAX_GRID_DIM
        self.assertLessEqual(MAX_GRID_DIM, MAX_GRID_DIM)  # per-axis ok by construction
        self.assertGreater(MAX_GRID_DIM * MAX_GRID_DIM, MAX_GRID_CELLS)
        with self.assertRaises(ValueError):
            _deck(domain=Domain(nx=MAX_GRID_DIM, ny=MAX_GRID_DIM,
                                lx_m=1.0, ly_m=1.0)).validate()

    def test_too_many_particles_rejected(self):
        from quasar.pic.io import MAX_PARTICLES
        sp = Species(name="e", charge_C=-1.0, mass_kg=1.0,
                     n_particles=MAX_PARTICLES + 1, initial=SpeciesInitial())
        with self.assertRaises(ValueError):
            _deck(species=[sp]).validate()

    def test_zero_particles_rejected(self):
        sp = Species(name="e", charge_C=-1.0, mass_kg=1.0, n_particles=0,
                     initial=SpeciesInitial())
        with self.assertRaises(ValueError):
            _deck(species=[sp]).validate()

    def test_nonpositive_mass_rejected(self):
        sp = Species(name="e", charge_C=-1.0, mass_kg=0.0, n_particles=8,
                     initial=SpeciesInitial())
        with self.assertRaises(ValueError):
            _deck(species=[sp]).validate()

    def test_nonfinite_mass_rejected(self):
        sp = Species(name="e", charge_C=-1.0, mass_kg=float("nan"), n_particles=8,
                     initial=SpeciesInitial())
        with self.assertRaises(ValueError):
            _deck(species=[sp]).validate()

    def test_negative_density_rejected(self):
        sp = Species(name="e", charge_C=-1.0, mass_kg=1.0, n_particles=8,
                     initial=SpeciesInitial(density_per_m3=-1.0))
        with self.assertRaises(ValueError):
            _deck(species=[sp]).validate()

    def test_invalid_fdtd_order_rejected(self):
        with self.assertRaises(ValueError):
            _deck(numerics=Numerics(fdtd_order=3)).validate()

    def test_fourth_order_requires_two_cells_per_dimension(self):
        for nx, ny in ((1, 8), (8, 1)):
            with self.subTest(nx=nx, ny=ny):
                with self.assertRaisesRegex(
                        ValueError, r"fdtd_order 4 requires.*at least 2"):
                    _deck(
                        domain=Domain(nx=nx, ny=ny, lx_m=1.0, ly_m=1.0),
                        numerics=Numerics(fdtd_order=4, shape="cic"),
                    ).validate()

    def test_negative_temperature_rejected(self):
        sp = Species(name="e", charge_C=-1.0, mass_kg=1.0, n_particles=8,
                     initial=SpeciesInitial(temperature_eV=-1.0))
        with self.assertRaises(ValueError):
            _deck(species=[sp]).validate()

    def test_nonpositive_steps_rejected(self):
        with self.assertRaises(ValueError):
            _deck(time=Time(dt_s="auto", steps=0)).validate()

    def test_bad_dt_string_rejected(self):
        with self.assertRaises(ValueError):
            _deck(time=Time(dt_s="fast", steps=8)).validate()

    def test_nonpositive_dt_rejected(self):
        with self.assertRaises(ValueError):
            _deck(time=Time(dt_s=-1.0e-12, steps=8)).validate()

    def test_nonfinite_dt_rejected(self):
        with self.assertRaises(ValueError):
            _deck(time=Time(dt_s=float("nan"), steps=8)).validate()

    def test_nonpositive_or_nonfinite_t_end_rejected(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _deck(time=Time(dt_s="auto", steps=8,
                                    t_end_s=value)).validate()

    def test_positive_t_end_accepted(self):
        _deck(time=Time(dt_s="auto", steps=8, t_end_s=1.0e-9)).validate()

    def test_nonpositive_reference_density_rejected(self):
        with self.assertRaises(ValueError):
            _deck(normalization=Normalization(reference_density_per_m3=0.0)).validate()

    def test_negative_diagnostic_cadence_rejected(self):
        with self.assertRaises(ValueError):
            _deck(diagnostics=Diagnostics(cadence=-1)).validate()

    def test_boolean_fields_reject_truthy_strings(self):
        with self.assertRaises(ValueError):
            _deck(diagnostics=Diagnostics(per_species="false")).validate()
        data = {
            "units": "SI",
            "domain": {"nx": 4, "ny": 4, "lx_m": 1.0, "ly_m": 1.0},
            "external_field": {
                "evaluator": {"type": "uniform", "B_T": [0, 0, 1]}},
            "neutralizing_background": "false",
            "time": {"dt_s": "auto", "steps": 1},
        }
        with self.assertRaises(ValueError):
            parse(data)

    def test_unknown_diagnostic_field_rejected(self):
        with self.assertRaises(ValueError):
            _deck(diagnostics=Diagnostics(fields=["bz", "pressure"])).validate()

    def test_nonfinite_lengths_rejected(self):
        for lx, ly in ((float("inf"), 1.0), (1.0, float("nan"))):
            with self.subTest(lx=lx, ly=ly):
                with self.assertRaises(ValueError):
                    _deck(domain=Domain(nx=8, ny=8, lx_m=lx, ly_m=ly)).validate()

    def test_nonfinite_origin_y_rejected(self):
        with self.assertRaises(ValueError):
            _deck(domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0,
                                origin_y_m=float("inf"))).validate()

    def test_nonfinite_charge_rejected(self):
        sp = Species(name="e", charge_C=float("nan"), mass_kg=1.0, n_particles=8,
                     initial=SpeciesInitial())
        with self.assertRaises(ValueError):
            _deck(species=[sp]).validate()

    def test_nonfinite_temperature_rejected(self):
        sp = Species(name="e", charge_C=-1.0, mass_kg=1.0, n_particles=8,
                     initial=SpeciesInitial(temperature_eV=float("inf")))
        with self.assertRaises(ValueError):
            _deck(species=[sp]).validate()

    def test_nonfinite_drift_rejected(self):
        sp = Species(name="e", charge_C=-1.0, mass_kg=1.0, n_particles=8,
                     initial=SpeciesInitial(drift_v=(float("nan"), 0.0, 0.0)))
        with self.assertRaises(ValueError):
            _deck(species=[sp]).validate()

    def test_nonfinite_block_region_bound_rejected(self):
        init = SpeciesInitial(distribution="maxwellian_block",
                              region_x_min_m=float("nan"), region_x_max_m=1.0,
                              region_y_min_m=0.0, region_y_max_m=1.0)
        sp = Species(name="e", charge_C=-1.0, mass_kg=1.0, n_particles=16,
                     initial=init)
        with self.assertRaises(ValueError):
            _deck(species=[sp]).validate()

    def test_nonfinite_fields_initial_amplitude_rejected(self):
        from quasar.pic.io import Fields, FieldsInitial
        seed = FieldsInitial(type="seed_perturbation", amplitude=float("inf"))
        with self.assertRaises(ValueError):
            _deck(fields=Fields(initial=seed)).validate()


class ExternalFieldFiniteTests(unittest.TestCase):

    def _ext(self, **overrides) -> ExternalField:
        base = dict(evaluator_type="uniform")
        base.update(overrides)
        return ExternalField(**base)

    def test_nonfinite_uniform_b_rejected(self):
        with self.assertRaises(ValueError):
            _deck(external_field=self._ext(
                uniform_b=(float("nan"), 0.0, 0.0))).validate()

    def test_nonfinite_uniform_e_rejected(self):
        with self.assertRaises(ValueError):
            _deck(external_field=self._ext(
                uniform_e=(0.0, float("inf"), 0.0))).validate()

    def test_nonfinite_dipole_origin_rejected(self):
        with self.assertRaises(ValueError):
            _deck(external_field=self._ext(
                dipole_origin=(0.0, 0.0, float("nan")))).validate()

    def test_nonfinite_gradient_b0_rejected(self):
        with self.assertRaises(ValueError):
            _deck(external_field=self._ext(
                gradient_b0=(float("inf"), 0.0, 0.0))).validate()

    def test_nonfinite_gradient_origin_rejected(self):
        with self.assertRaises(ValueError):
            _deck(external_field=self._ext(
                gradient_origin=(0.0, float("nan"), 0.0))).validate()

    def test_nonfinite_dipole_moment_rejected(self):
        with self.assertRaises(ValueError):
            _deck(external_field=self._ext(
                evaluator_type="dipole",
                dipole_moment=(float("nan"), 0.0, 0.0))).validate()

    def test_nonfinite_gradient_matrix_rejected(self):
        grad = ((0.0, 0.0, 0.0),
                (0.0, float("inf"), 0.0),
                (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            _deck(external_field=self._ext(
                evaluator_type="gradient", gradient_matrix=grad)).validate()

    def test_non_solenoidal_gradient_matrix_rejected(self):
        grad = ((1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            _deck(external_field=self._ext(
                evaluator_type="gradient", gradient_matrix=grad)).validate()

    def test_generic_plugin_params_flatten_scalars_and_vectors(self):
        field = ExternalField(
            evaluator_type="test_plugin",
            params={"gain": 2, "axis": (1.0, -2.0, 3.0), "empty": []})
        self.assertEqual(field.evaluator_params(), {
            "gain": [2.0], "axis": [1.0, -2.0, 3.0], "empty": []})

    def test_generic_plugin_params_reject_invalid_shapes_and_values(self):
        invalid = (
            {"flag": True},
            {"name": "value"},
            {"nested": [[1.0], [2.0]]},
            {"mapping": {"x": 1.0}},
            {"bad": float("inf")},
        )
        for params in invalid:
            with self.subTest(params=params):
                with self.assertRaises((TypeError, ValueError)):
                    ExternalField(
                        evaluator_type="test_plugin",
                        params=params).evaluator_params()

    def test_builtins_emit_only_the_selected_evaluator_parameters(self):
        field = ExternalField(
            evaluator_type="biot_savart",
            uniform_b=(1.0, 2.0, 3.0),
            dipole_moment=(4.0, 5.0, 6.0),
            params={"plugin_only": 7.0})
        self.assertEqual(field.evaluator_params(), {})
        field.evaluator_type = "uniform"
        self.assertEqual(field.evaluator_params(), {
            "b0": [1.0, 2.0, 3.0], "e0": [0.0, 0.0, 0.0]})


class DiagnosticsNormalizationTests(unittest.TestCase):

    def test_direct_construction_lowercases_fields(self):
        diag = Diagnostics(fields=["BZ", "Ex"])
        self.assertEqual(diag.fields, ["bz", "ex"])
        _deck(diagnostics=diag).validate()

    def test_parsed_deck_lowercases_fields(self):
        deck = parse({
            "domain": {"nx": 8, "ny": 8, "lx_m": 1.0, "ly_m": 1.0},
            "species": [{"name": "e", "charge_C": -1.0, "mass_kg": 1.0,
                         "n_particles": 8}],
            "time": {"dt_s": 1.0e-12, "steps": 8},
            "diagnostics": {"fields": ["Bz", "EX"]},
        })
        self.assertEqual(deck.diagnostics.fields, ["bz", "ex"])


class FieldOnlyDeckTests(unittest.TestCase):

    def test_external_field_only_deck_valid(self):
        # No species, but an external field drives the run.
        _deck(species=[], external_field=ExternalField(
            evaluator_type="uniform")).validate()

    def test_units_normalized_accepted(self):
        _deck(units="normalized").validate()

    def test_bad_units_rejected(self):
        with self.assertRaises(ValueError):
            _deck(units="CGS").validate()


if __name__ == "__main__":
    unittest.main()
