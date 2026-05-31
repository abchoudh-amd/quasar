"""Unit tests for the pure-numpy PIC postprocess helpers.

reshape_with_ghost strips the Yee halo using the explicit ``nghost`` persisted in
the npz; species_names recovers species from the npz key schema; rms is a plain
reduction. None need matplotlib or a GPU, yet the existing suite did not cover
quasar.pic.postprocess at all. A ghost-width off-by-one would silently corrupt
every PIC field heatmap, so pin the contract here.
"""

import math
import types
import unittest

import numpy as np

from quasar.pic.postprocess import reshape_with_ghost, rms, species_names


class RmsTests(unittest.TestCase):

    def test_known_value(self):
        self.assertAlmostEqual(rms([3.0, 4.0]), math.sqrt(12.5), places=12)

    def test_zeros(self):
        self.assertEqual(rms([0.0, 0.0, 0.0]), 0.0)


class ReshapeWithGhostTests(unittest.TestCase):

    def _ghost_padded(self, nx, ny, g):
        """Flat (nx+2g)*(ny+2g) buffer whose value encodes interior vs ghost: an
        interior cell (i,j) holds 1000*j + i (>= 0); ghost cells hold -1."""
        pitch, height = nx + 2 * g, ny + 2 * g
        buf = np.full(pitch * height, -1.0)
        for j in range(ny):
            for i in range(nx):
                buf[(i + g) + pitch * (j + g)] = 1000 * j + i
        return buf

    def test_strips_ghosts_for_each_width(self):
        nx, ny = 5, 3
        for g in (1, 2):
            flat = self._ghost_padded(nx, ny, g)
            interior = reshape_with_ghost(flat, nx, ny, g)
            self.assertEqual(interior.shape, (ny, nx))
            for j in range(ny):
                for i in range(nx):
                    self.assertEqual(interior[j, i], 1000 * j + i)
            self.assertTrue((interior >= 0).all())  # no ghost (-1) leaked in

    def test_no_ghost_buffer_reshapes_directly(self):
        nx, ny = 4, 2
        flat = np.arange(nx * ny, dtype=float)
        interior = reshape_with_ghost(flat, nx, ny, 0)
        self.assertEqual(interior.shape, (ny, nx))
        np.testing.assert_array_equal(interior, flat.reshape(ny, nx))


class SpeciesNamesTests(unittest.TestCase):

    def test_recovers_sorted_names_ignoring_decoys(self):
        data = types.SimpleNamespace(files=[
            "species_e_x", "species_e_y", "species_e_vx",
            "species_H+_x", "species_H+_y",
            "field_ez", "nx", "ny", "snapshot_field_ez",
        ])
        self.assertEqual(species_names(data), ["H+", "e"])

    def test_empty_when_no_species_keys(self):
        data = types.SimpleNamespace(files=["field_ez", "nx", "ny"])
        self.assertEqual(species_names(data), [])


if __name__ == "__main__":
    unittest.main()
