"""Unit tests for the SI<->internal converter (quasar.pic._units.Units).

Units bridges an SI PIC deck into the solver's natural units (c = eps0 = mu0 = 1);
getting it wrong makes the EM evolution physically wrong, yet only the underlying
C++ Normalization is covered elsewhere. These tests pin the Python wrapper's
branching: the normalized identity path, the SI round-trip, the unknown-species
error, and the external_scales tuple. Only _core.pic.Normalization is needed
(no HIP runtime), so the test runs in the standard suite.
"""

import types
import unittest

from quasar.pic._units import Units


def _deck(units, *, species="electron", density=1.0e18):
    """Minimal duck-typed stand-in carrying only what Units reads."""
    return types.SimpleNamespace(
        units=units,
        normalization=types.SimpleNamespace(
            reference_species=species,
            reference_density_per_m3=density,
        ),
    )


class NormalizedIdentityTests(unittest.TestCase):

    def setUp(self):
        self.u = Units(_deck("normalized"))

    def test_identity_flag(self):
        self.assertTrue(self.u.identity)
        self.assertIsNone(self.u.normalization)

    def test_all_conversions_are_identity(self):
        for v in (0.0, 1.0, -3.5, 1.0e6):
            self.assertEqual(self.u.length(v), v)
            self.assertEqual(self.u.time(v), v)
            self.assertEqual(self.u.velocity(v), v)
            self.assertEqual(self.u.charge(v), v)
            self.assertEqual(self.u.mass(v), v)
            self.assertEqual(self.u.density(v), v)
            self.assertEqual(self.u.length_to_si(v), v)
            self.assertEqual(self.u.velocity_to_si(v), v)

    def test_external_scales_all_unity(self):
        self.assertEqual(self.u.external_scales(), (1.0, 1.0, 1.0))


class SiConversionTests(unittest.TestCase):

    def setUp(self):
        self.u = Units(_deck("SI", species="electron", density=1.0e18))

    def test_not_identity(self):
        self.assertFalse(self.u.identity)
        self.assertIsNotNone(self.u.normalization)

    def test_length_round_trip(self):
        for v in (1.0, 0.05, 123.4):
            internal = self.u.length(v)
            self.assertAlmostEqual(self.u.length_to_si(internal), v, places=9)

    def test_time_round_trip(self):
        for v in (1.0e-9, 3.3e-6):
            internal = self.u.time(v)
            self.assertAlmostEqual(self.u.time_to_si(internal), v, places=18)

    def test_velocity_round_trip(self):
        for v in (1.0e5, 2.99e8):
            internal = self.u.velocity(v)
            self.assertAlmostEqual(self.u.velocity_to_si(internal), v, places=3)

    def test_non_identity_factors_differ_from_one(self):
        # A genuine SI normalization should scale length away from 1:1.
        self.assertNotAlmostEqual(self.u.length(1.0), 1.0)

    def test_external_scales_match_normalization(self):
        norm = self.u.normalization
        self.assertEqual(
            self.u.external_scales(),
            (norm.length_scale(), norm.e_field_scale(), norm.b_field_scale()),
        )


class ReferenceSpeciesTests(unittest.TestCase):

    def test_proton_and_ion_accepted(self):
        for sp in ("proton", "ion", "hydrogen", "electron"):
            u = Units(_deck("SI", species=sp))
            self.assertFalse(u.identity)

    def test_unknown_species_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Units(_deck("SI", species="unobtanium"))
        self.assertIn("unobtanium", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
