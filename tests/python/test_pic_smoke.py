import unittest

import quasar.pic as pic
from quasar import _core


class PicSmokeTests(unittest.TestCase):
    def test_binding_symbols(self):
        self.assertTrue(hasattr(pic, "Grid2D"))
        self.assertTrue(hasattr(pic, "EmPicConfig"))

    def test_default_normalization_is_identity_for_every_unit_tag(self):
        norm = pic.Normalization()
        self.assertTrue(norm.is_identity())
        self.assertTrue(pic.Normalization.identity().is_identity())
        self.assertTrue(pic.EmPicConfig().normalization.is_identity())
        tags = _core.pic.UnitTag
        for tag in (
            tags.time,
            tags.length,
            tags.velocity,
            tags.e_field,
            tags.b_field,
            tags.density,
            tags.charge,
            tags.mass,
            tags.temperature_eV,
        ):
            self.assertEqual(norm.to_internal(-3.25, tag), -3.25)
            self.assertEqual(norm.to_si(-3.25, tag), -3.25)

        plasma = pic.Normalization.plasma(1.0e18, 1.602176634e-19,
                                          9.1093837015e-31)
        self.assertFalse(plasma.is_identity())


if __name__ == "__main__":
    unittest.main()
