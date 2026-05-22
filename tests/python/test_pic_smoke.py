import unittest

import quasar.pic as pic


class PicSmokeTests(unittest.TestCase):
    def test_binding_symbols(self):
        self.assertTrue(hasattr(pic, "Grid2D"))
        self.assertTrue(hasattr(pic, "EmPicConfig"))


if __name__ == "__main__":
    unittest.main()
