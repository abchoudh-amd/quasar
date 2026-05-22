import unittest

from quasar.pic.io import Domain, Numerics, PicDeck


class PicIoTests(unittest.TestCase):
    def test_schema_validation(self):
        deck = PicDeck(domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0),
                       numerics=Numerics(fdtd_order=4, shape="tsc"),
                       units="SI")
        deck.validate()

    def test_invalid_shape(self):
        deck = PicDeck(domain=Domain(nx=8, ny=8, lx_m=1.0, ly_m=1.0),
                       numerics=Numerics(shape="bad"))
        with self.assertRaises(ValueError):
            deck.validate()


if __name__ == "__main__":
    unittest.main()
