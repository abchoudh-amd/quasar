import unittest

from quasar.pic.io import (
    BoundaryConfig,
    Diagnostics,
    Domain,
    ExternalField,
    Normalization,
    Numerics,
    PicDeck,
    Species,
    SpeciesInitial,
    Time,
    _parse_boundary,
    _parse_fields,
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

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            _deck(numerics=Numerics(shape="bad")).validate()

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

    def test_gradient_evaluator_requires_matrix(self):
        with self.assertRaises(ValueError):
            _deck(external_field=ExternalField(evaluator_type="gradient")).validate()

    def test_current_filter_passes_must_be_positive(self):
        with self.assertRaises(ValueError):
            _deck(numerics=Numerics(
                current_filter=[{"type": "compensated_binomial", "passes": 0}]
            )).validate()

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
                        [0.0, 0.0, 3.0],
                    ],
                    "origin_xyz_m": [0.1, 0.2, 0.3],
                },
            },
            "species": [],
            "time": {"dt_s": "auto", "steps": 1},
        })
        self.assertEqual(deck.external_field.gradient_b0, (0.0, 0.0, 1.0))
        self.assertEqual(deck.external_field.gradient_matrix[2], (0.0, 0.0, 3.0))
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
            particle=("periodic", "specular", "absorbing", "periodic"))).validate()


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
        self.assertEqual(f.initial.component, "Ex")
        self.assertEqual(f.initial.amplitude, 1.0e-4)
        self.assertEqual(f.initial.mode, (1, 0))

    def test_missing_type_raises(self):
        with self.assertRaises(ValueError):
            _parse_fields({"initial": {"component": "Ez"}})


class ResourceCeilingTests(unittest.TestCase):

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

    def test_nonpositive_reference_density_rejected(self):
        with self.assertRaises(ValueError):
            _deck(normalization=Normalization(reference_density_per_m3=0.0)).validate()

    def test_negative_diagnostic_cadence_rejected(self):
        with self.assertRaises(ValueError):
            _deck(diagnostics=Diagnostics(cadence=-1)).validate()

    def test_unknown_diagnostic_field_rejected(self):
        with self.assertRaises(ValueError):
            _deck(diagnostics=Diagnostics(fields=["bz", "pressure"])).validate()


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
