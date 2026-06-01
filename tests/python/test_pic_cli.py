import unittest

import numpy as np

from quasar import _core
from quasar.pic._units import Units
from quasar.pic.cli import (
    _apply_external_field,
    _build_parser,
    _flatten_for_npz,
    _seed_fields,
    _seed_species,
    _snapshot,
)
from quasar.pic.io import (
    Diagnostics,
    Domain,
    ExternalField,
    Fields,
    FieldsInitial,
    Numerics,
    PicDeck,
    Species,
    SpeciesInitial,
)


class _RecordingSolver:
    """Captures seed_field(component, values) calls without a GPU."""

    def __init__(self):
        self.calls = []

    def seed_field(self, component, values):
        self.calls.append((component, np.asarray(values)))


def _fields_deck(initial: FieldsInitial) -> PicDeck:
    return PicDeck(
        domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0),
        numerics=Numerics(fdtd_order=2, shape="cic"),
        species=[Species(name="e", charge_C=-1.0, mass_kg=1.0, n_particles=8,
                         initial=SpeciesInitial())],
        fields=Fields(initial=initial),
        units="normalized",
    )


class SeedFieldsTests(unittest.TestCase):

    def test_seed_em_wave_rejects_nonzero_my(self):
        deck = _fields_deck(FieldsInitial(type="seed_em_wave", component="Ez",
                                          mode=(1, 1)))
        with self.assertRaises(ValueError):
            _seed_fields(_RecordingSolver(), deck)

    def test_seed_em_wave_rejects_unsupported_component(self):
        deck = _fields_deck(FieldsInitial(type="seed_em_wave", component="Bz",
                                          mode=(1, 0)))
        with self.assertRaises(ValueError):
            _seed_fields(_RecordingSolver(), deck)

    def test_unsupported_initial_type_raises(self):
        deck = _fields_deck(FieldsInitial(type="seed_blastwave", component="Ex"))
        with self.assertRaises(ValueError):
            _seed_fields(_RecordingSolver(), deck)

    def test_seed_em_wave_ez_seeds_ez_and_by(self):
        deck = _fields_deck(FieldsInitial(type="seed_em_wave", component="Ez",
                                          mode=(1, 0)))
        solver = _RecordingSolver()
        _seed_fields(solver, deck)
        seeded = {c for c, _ in solver.calls}
        self.assertEqual(seeded, {"ez", "by"})


class BuildParserTests(unittest.TestCase):

    def test_run_defaults(self):
        args = _build_parser().parse_args(["run", "deck.yaml"])
        self.assertEqual(args.input, "deck.yaml")
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.log_every, 0)
        self.assertEqual(args.write_every, 0)
        self.assertIsNone(args.steps_override)
        self.assertFalse(args.print_config)

    def test_flag_types(self):
        args = _build_parser().parse_args(
            ["run", "d.yaml", "--seed", "9", "--log-every", "5",
             "--write-every", "20", "--steps-override", "7", "--print-config"])
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.log_every, 5)
        self.assertEqual(args.write_every, 20)
        self.assertEqual(args.steps_override, 7)
        self.assertTrue(args.print_config)

    def test_missing_subcommand_exits_nonzero(self):
        with self.assertRaises(SystemExit) as ctx:
            _build_parser().parse_args([])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_steps_override_must_be_positive(self):
        for value in ("0", "-3"):
            with self.subTest(value=value):
                with self.assertRaises(SystemExit) as ctx:
                    _build_parser().parse_args(
                        ["run", "deck.yaml", "--steps-override", value])
                self.assertNotEqual(ctx.exception.code, 0)


class FlattenForNpzTests(unittest.TestCase):

    def _final(self):
        return {
            "step": 10,
            "time_s": 1.0e-9,
            "nx": 4,
            "ny": 4,
            "nghost": 1,
            "external_bx": np.zeros(4),
            "external_by": np.zeros(4),
            "external_bz": np.zeros(4),
            "fields": {"bz": np.arange(16, dtype=float)},
        }

    def test_scalar_series_keys_and_shapes(self):
        series = {"step": [5, 10], "time_s": [0.5e-9, 1.0e-9],
                  "alive_e": [100, 98]}
        flat = _flatten_for_npz([], self._final(), series)
        self.assertIn("series_step", flat)
        self.assertIn("series_alive_e", flat)
        np.testing.assert_array_equal(flat["series_alive_e"], np.array([100, 98]))
        self.assertEqual(flat["series_step"].shape, (2,))

    def test_final_scalars_and_field(self):
        flat = _flatten_for_npz([], self._final(), None)
        self.assertEqual(int(flat["final_step"][0]), 10)
        self.assertEqual(int(flat["nx"][0]), 4)
        self.assertEqual(int(flat["nghost"][0]), 1)
        self.assertIn("field_bz", flat)
        self.assertEqual(flat["field_bz"].shape, (16,))

    def test_snapshot_stacking(self):
        snaps = [
            {"step": 5, "time_s": 0.5e-9, "fields": {"bz": np.ones(16)}},
            {"step": 10, "time_s": 1.0e-9, "fields": {"bz": np.zeros(16)}},
        ]
        flat = _flatten_for_npz(snaps, self._final(), None)
        self.assertEqual(flat["snapshot_field_bz"].shape, (2, 16))
        np.testing.assert_array_equal(flat["snapshot_steps"], np.array([5, 10]))


class _ComponentRecordingSolver:
    def __init__(self):
        self.field_calls = []
        self.external_calls = []

    def field_component_to_host(self, component):
        self.field_calls.append(component)
        return np.full(4, len(self.field_calls), dtype=float)

    def external_field_component_to_host(self, component):
        self.external_calls.append(component)
        return np.full(4, len(self.external_calls), dtype=float)


class SnapshotTests(unittest.TestCase):

    def test_snapshot_reads_only_requested_field_components(self):
        deck = PicDeck(
            domain=Domain(nx=2, ny=2, lx_m=1.0, ly_m=1.0),
            numerics=Numerics(fdtd_order=2, shape="cic"),
            diagnostics=Diagnostics(fields=["bz"], per_species=False),
            units="normalized",
        )
        solver = _ComponentRecordingSolver()
        snap = _snapshot(solver, deck, [], step=3, sim_time=1.5, units=Units(deck))

        self.assertEqual(solver.field_calls, ["bz"])
        self.assertEqual(solver.external_calls, ["bx", "by", "bz"])
        self.assertEqual(snap["nx"], 2)
        self.assertEqual(snap["ny"], 2)
        self.assertEqual(set(snap["fields"]), {"bz"})


class _RecordingSpecies:
    """Captures set_host_particles(...) kwargs without a GPU."""

    def __init__(self):
        self.host = None

    def set_host_particles(self, **kwargs):
        self.host = {k: np.asarray(v) for k, v in kwargs.items()}


class _SpeciesRecordingSolver:
    """Stub solver capturing add_species/species_at for _seed_species."""

    def __init__(self):
        self.configs = []
        self.species = []

    def add_species(self, cfg):
        self.configs.append(cfg)
        self.species.append(_RecordingSpecies())
        return len(self.species) - 1

    def species_at(self, idx):
        return self.species[idx]


def _species_deck(initial: SpeciesInitial) -> PicDeck:
    return PicDeck(
        domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0),
        numerics=Numerics(fdtd_order=2, shape="cic"),
        species=[Species(name="e", charge_C=-1.0, mass_kg=1.0, n_particles=64,
                         initial=initial)],
        units="normalized",
    )


class SeedSpeciesTests(unittest.TestCase):
    """The maxwellian_block branch is only reachable on a GPU end-to-end run, so
    pin its CPU-computable contract (positions confined to the region; macro
    weight derived from the block area, not the full domain) with a stub solver."""

    def test_block_positions_confined_to_region_and_weight_uses_block_area(self):
        region = dict(region_x_min_m=0.25, region_x_max_m=0.75,
                      region_y_min_m=0.5, region_y_max_m=1.0)
        deck = _species_deck(SpeciesInitial(distribution="maxwellian_block",
                                            density_per_m3=1.0e18,
                                            temperature_eV=1.0, **region))
        units = Units(deck)
        solver = _SpeciesRecordingSolver()
        rng = np.random.default_rng(0)
        _seed_species(solver, deck, units, rng)

        host = solver.species[0].host
        self.assertIsNotNone(host)
        x_lo = units.length(region["region_x_min_m"])
        x_hi = units.length(region["region_x_max_m"])
        y_lo = units.length(region["region_y_min_m"])
        y_hi = units.length(region["region_y_max_m"])
        self.assertTrue(np.all(host["x"] >= x_lo - 1e-12))
        self.assertTrue(np.all(host["x"] <= x_hi + 1e-12))
        self.assertTrue(np.all(host["y"] >= y_lo - 1e-12))
        self.assertTrue(np.all(host["y"] <= y_hi + 1e-12))
        # All positions lie inside the block, never the full domain.
        self.assertLess(host["x"].max(), units.length(deck.domain.lx_m))

        # Macro weight = density * block_area / N (block area, not domain area).
        block_area = (x_hi - x_lo) * (y_hi - y_lo)
        expected_w = units.density(1.0e18) * block_area / 64
        np.testing.assert_allclose(host["weight"], expected_w, rtol=1e-12)

    def test_uniform_positions_span_full_domain(self):
        deck = _species_deck(SpeciesInitial(distribution="maxwellian_uniform",
                                            density_per_m3=1.0e18,
                                            temperature_eV=1.0))
        units = Units(deck)
        solver = _SpeciesRecordingSolver()
        _seed_species(solver, deck, units, np.random.default_rng(0))
        host = solver.species[0].host
        # Uniform weight uses the full domain area, so it differs from a sub-block.
        domain_area = units.length(deck.domain.lx_m) * units.length(deck.domain.ly_m)
        expected_w = units.density(1.0e18) * domain_area / 64
        np.testing.assert_allclose(host["weight"], expected_w, rtol=1e-12)


class _ExternalRecordingSolver:
    """Captures the evaluator + scales passed to sample_external_field."""

    def __init__(self):
        self.evaluator = None
        self.conductors = None
        self.scales = None

    def sample_external_field(self, evaluator, conductors, length_scale,
                              e_field_scale, b_field_scale):
        self.evaluator = evaluator
        self.conductors = conductors
        self.scales = (length_scale, e_field_scale, b_field_scale)


def _external_deck(ef: ExternalField) -> PicDeck:
    return PicDeck(
        domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0),
        numerics=Numerics(fdtd_order=2, shape="cic"),
        external_field=ef,
        units="SI",
    )


class ApplyExternalFieldTests(unittest.TestCase):
    """_apply_external_field builds the evaluator by registry name then pushes deck
    params through configure(); this is only reached end-to-end on a GPU run, so
    exercise the (CPU-only) build+configure seam here against the real _core."""

    def _eval_for(self, ef: ExternalField):
        deck = _external_deck(ef)
        solver = _ExternalRecordingSolver()
        _apply_external_field(solver, deck, Units(deck))
        self.assertIsNotNone(solver.evaluator)
        return solver.evaluator

    def test_uniform_builds_uniform_evaluator_and_configures_b(self):
        ev = self._eval_for(ExternalField(evaluator_type="uniform",
                                          uniform_b=(0.0, 0.0, 2.0)))
        self.assertEqual(type(ev).__name__, "UniformEvaluator")
        # configure() must have applied the deck B0: evaluate_B returns it.
        cs = _core.magnetostatics.ConductorSystem()
        pts = _core.magnetostatics.PointCloud()
        pts.add(_core.Vec3(0.1, 0.2, 0.3))
        b = ev.evaluate_B(cs, pts)
        np.testing.assert_allclose(b[0], [0.0, 0.0, 2.0], atol=1e-12)

    def test_dipole_builds_dipole_evaluator(self):
        ev = self._eval_for(ExternalField(evaluator_type="dipole",
                                          dipole_moment=(0.0, 0.0, 1.0)))
        self.assertEqual(type(ev).__name__, "DipoleEvaluator")

    def test_gradient_builds_gradient_evaluator(self):
        grad = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        ev = self._eval_for(ExternalField(evaluator_type="gradient",
                                          gradient_b0=(0.0, 0.0, 1.0),
                                          gradient_matrix=grad))
        self.assertEqual(type(ev).__name__, "GradientEvaluator")

    def test_biot_savart_builds_biot_savart_evaluator(self):
        ev = self._eval_for(ExternalField(evaluator_type="biot_savart",
                                          conductors=[]))
        self.assertEqual(type(ev).__name__, "BiotSavartEvaluator")


if __name__ == "__main__":
    unittest.main()
