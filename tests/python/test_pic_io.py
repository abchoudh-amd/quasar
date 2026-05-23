import unittest

from quasar.pic.io import (
    Domain,
    ExternalField,
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


if __name__ == "__main__":
    unittest.main()
