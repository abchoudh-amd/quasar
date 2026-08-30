"""Invariants of the device-resident PIC initial sampler.

This replaces the tests for the NumPy helpers in ``quasar.pic.initial_conditions``,
which are gone: the quiet-start lattice, the Maxwellian draw and the speed check
are kernels now, so the properties they used to assert have to be asserted about
the code that actually runs. The properties themselves are unchanged --
stratification, exact block centre of mass, equal-volume cylindrical strata,
antithetic pairing, and the rejections -- because they are what makes a quiet
start quiet, not artefacts of the old implementation.

The one thing deliberately *not* asserted is the value of any particular draw.
The generator is Philox4x32-10 counted by particle index, and pinning its output
here would turn a change of generator into a test edit rather than the reference
re-baselining it actually is.
"""

import math
import unittest

import numpy as np

from quasar import _core

pic = _core.pic


def _sample(count, x_min, x_max, y_min, y_max, *, cylindrical=False,
            thermal_speed=0.0, drift=(0.0, 0.0, 0.0), seed=0,
            weight=1.0, domain=None, perturb=None):
    """Sample one species and return the host arrays."""
    if domain is None:
        # Default to a domain that just contains the block, so the sampler's
        # in-domain check passes for the geometric tests. The check itself is
        # exercised separately through the solver entry point.
        domain = (0.0 if cylindrical else x_min, y_min, x_max, y_max)
    ox, oy, hx, hy = domain
    config = pic.ParticleSampleConfig()
    config.count = count
    config.x_min, config.x_max = x_min, x_max
    config.y_min, config.y_max = y_min, y_max
    config.cylindrical = cylindrical
    config.thermal_speed = thermal_speed
    config.drift_x, config.drift_y, config.drift_z = drift
    config.seed = seed
    config.weight = weight
    config.domain_origin_x, config.domain_origin_y = ox, oy
    config.domain_lx, config.domain_ly = hx - ox, hy - oy
    if perturb is not None:
        config.perturb = True
        (config.mode_x, config.mode_y, config.phase,
         config.amplitude_x, config.amplitude_y, config.amplitude_z) = perturb
    return pic.sample_particles(config)


def _positions(snapshot):
    return np.column_stack((np.asarray(snapshot["x"]),
                            np.asarray(snapshot["y"])))


def _velocities(snapshot):
    return np.column_stack((np.asarray(snapshot["vx"]),
                            np.asarray(snapshot["vy"]),
                            np.asarray(snapshot["vz"])))


class QuietPositionsBlockTests(unittest.TestCase):

    def test_count_matches_request(self):
        snapshot = _sample(50, 0.0, 1.0, 0.0, 2.0)
        self.assertEqual(_positions(snapshot).shape, (50, 2))

    def test_points_inside_block(self):
        x_min, x_max, y_min, y_max = -1.0, 1.0, 2.0, 5.0
        pts = _positions(_sample(100, x_min, x_max, y_min, y_max,
                                 domain=(x_min, y_min, x_max, y_max)))
        self.assertTrue(np.all(pts[:, 0] >= x_min))
        self.assertTrue(np.all(pts[:, 0] <= x_max))
        self.assertTrue(np.all(pts[:, 1] >= y_min))
        self.assertTrue(np.all(pts[:, 1] <= y_max))

    def test_each_axis_is_exactly_stratified_and_centered(self):
        n = 50
        pts = _positions(_sample(n, -2.0, 4.0, 3.0, 9.0,
                                 domain=(-2.0, 3.0, 4.0, 9.0)))
        expected_center = np.asarray([1.0, 6.0])
        # Summing binary64 stratum midpoints incurs an absolute error that
        # scales with the coordinate magnitude.  Use a component-wise bound;
        # a fixed multiple of epsilon is only meaningful near unity.
        center_error = ((np.mean(pts, axis=0) - expected_center)
                        / np.maximum(1.0, np.abs(expected_center)))
        np.testing.assert_allclose(center_error, 0.0, rtol=0.0,
                                   atol=8 * np.finfo(float).eps)
        expected = (np.arange(n) + 0.5) / n
        np.testing.assert_allclose(
            np.sort((pts[:, 0] + 2.0) / 6.0), expected, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(
            np.sort((pts[:, 1] - 3.0) / 6.0), expected, rtol=0.0, atol=1e-15)

    def test_both_axes_use_every_stratum_exactly_once(self):
        # The lattice is rank-1: the second coordinate walks the strata in a
        # strided order, not the same order as the first. Both must still be a
        # permutation of the same midpoint set, which is what makes the start
        # Latin-hypercube rather than diagonal.
        n = 64
        pts = _positions(_sample(n, 0.0, 1.0, 0.0, 1.0))
        strata = np.floor(pts * n).astype(int)
        np.testing.assert_array_equal(np.sort(strata[:, 0]), np.arange(n))
        np.testing.assert_array_equal(np.sort(strata[:, 1]), np.arange(n))
        self.assertFalse(np.array_equal(strata[:, 0], strata[:, 1]))

    def test_degenerate_and_collapsed_layouts_are_rejected(self):
        for count in (0, -1):
            with self.subTest(count=count):
                with self.assertRaises(Exception):
                    _sample(count, 0.0, 1.0, 0.0, 1.0)
        with self.assertRaises(Exception):
            _sample(2, 1.0e308, math.nextafter(1.0e308, math.inf), 0.0, 1.0)

    def test_large_translated_origin_retains_local_strata(self):
        # The kernel adds a fraction of the extent to the lower bound rather
        # than forming a convex combination of two origin-sized products, so a
        # domain far from zero keeps its local separation.
        n = 8
        lower = math.ldexp(1.0, 900)
        width = math.ldexp(1.0, 858)
        upper = lower + width
        pts = _positions(_sample(n, lower, upper, lower, upper,
                                 domain=(lower, lower, upper, upper)))
        self.assertGreater(np.unique(pts[:, 0]).size, 1)
        self.assertGreater(np.unique(pts[:, 1]).size, 1)
        expected = (np.arange(n) + 0.5) / n
        np.testing.assert_allclose(
            np.sort((pts[:, 0] - lower) / width), expected,
            rtol=0.0, atol=1.0e-2)
        np.testing.assert_allclose(
            np.sort((pts[:, 1] - lower) / width), expected,
            rtol=0.0, atol=1.0e-2)


def _measure(count, x_min, x_max, y_min, y_max, cylindrical=False):
    config = pic.ParticleSampleConfig()
    config.count = count
    config.x_min, config.x_max = x_min, x_max
    config.y_min, config.y_max = y_min, y_max
    config.cylindrical = cylindrical
    return pic.quiet_block_measure(config)


class QuietBlockCellAreaTests(unittest.TestCase):

    def test_perfect_square_tiles_exactly(self):
        area = (1.0 - 0.0) * (2.0 - 0.0)
        self.assertAlmostEqual(_measure(25, 0.0, 1.0, 0.0, 2.0) * 25, area)

    def test_non_square_represents_exact_block_area(self):
        # The rank-1 lattice contains exactly N equal-weight particles.  There is
        # no partially populated square layout, so every particle represents
        # exactly block_area/N.
        x_min, x_max, y_min, y_max = 0.0, 4.0, 0.0, 2.0
        block_area = (x_max - x_min) * (y_max - y_min)
        cell = _measure(50, x_min, x_max, y_min, y_max)
        self.assertAlmostEqual(cell, block_area / 50)
        self.assertAlmostEqual(cell * 50, block_area)

    def test_scaled_area_avoids_false_underflow(self):
        self.assertAlmostEqual(_measure(2, 0.0, 1.0e-300, 0.0, 1.0e300), 0.5)


class QuietRingVolumeTests(unittest.TestCase):

    def test_equal_volume_radial_strata_and_total_volume(self):
        n = 37
        r_min, r_max, z_min, z_max = 2.0, 5.0, -1.0, 3.0
        pts = _positions(_sample(n, r_min, r_max, z_min, z_max,
                                 cylindrical=True,
                                 domain=(0.0, z_min, r_max, z_max)))
        u = (pts[:, 0] ** 2 - r_min ** 2) / (r_max ** 2 - r_min ** 2)
        np.testing.assert_allclose(np.sort(u), (np.arange(n) + 0.5) / n,
                                   rtol=0.0, atol=2e-15)
        represented = _measure(n, r_min, r_max, z_min, z_max,
                               cylindrical=True) * n
        self.assertAlmostEqual(
            represented, math.pi * (r_max**2 - r_min**2) * (z_max - z_min))

    def test_negative_radius_and_unrepresentable_volume_are_rejected(self):
        with self.assertRaises(Exception):
            _sample(4, -1.0, 1.0, 0.0, 1.0, cylindrical=True)
        with self.assertRaises(Exception):
            _measure(1, 0.0, 1.0e200, 0.0, 1.0e200, cylindrical=True)

    def test_large_radius_thin_annulus_keeps_equal_volume_strata(self):
        n = 8
        r_min = math.ldexp(1.0, 900)
        width = math.ldexp(1.0, 858)
        r_max = r_min + width
        pts = _positions(_sample(n, r_min, r_max, 0.0, 1.0, cylindrical=True,
                                 domain=(0.0, 0.0, r_max, 1.0)))
        # Evaluate the normalized r^2 coordinate in factored form so the oracle
        # itself neither squares the large radii nor subtracts adjacent squares.
        radial_offset = pts[:, 0] - r_min
        u = ((radial_offset / width)
             * ((pts[:, 0] + r_min) / (r_max + r_min)))
        np.testing.assert_allclose(
            np.sort(u), (np.arange(n) + 0.5) / n,
            rtol=0.0, atol=1.5e-2)


class MaxwellianTests(unittest.TestCase):

    def test_shape_and_drift(self):
        v = _velocities(_sample(10000, 0.0, 1.0, 0.0, 1.0,
                                thermal_speed=0.02,
                                drift=(0.01, 0.0, -0.03), seed=7))
        self.assertEqual(v.shape, (10000, 3))
        # Antithetic pairing makes the thermal mean exactly zero, so the sample
        # mean is the drift to rounding, not merely near it for a large sample.
        np.testing.assert_allclose(np.mean(v, axis=0), [0.01, 0.0, -0.03],
                                   rtol=0.0, atol=1e-15)

    def test_deterministic_with_seed(self):
        a = _velocities(_sample(100, 0.0, 1.0, 0.0, 1.0,
                                thermal_speed=0.1, seed=123))
        b = _velocities(_sample(100, 0.0, 1.0, 0.0, 1.0,
                                thermal_speed=0.1, seed=123))
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_give_different_draws(self):
        a = _velocities(_sample(64, 0.0, 1.0, 0.0, 1.0,
                                thermal_speed=0.1, seed=1))
        b = _velocities(_sample(64, 0.0, 1.0, 0.0, 1.0,
                                thermal_speed=0.1, seed=2))
        self.assertFalse(np.array_equal(a, b))

    def test_pairs_are_local_antithetic_pairs(self):
        drift = np.asarray([0.01, -0.02, 0.005])
        v = _velocities(_sample(9, 0.0, 1.0, 0.0, 1.0, thermal_speed=0.05,
                                drift=tuple(drift), seed=4))
        for i in range(0, 8, 2):
            np.testing.assert_allclose(v[i] + v[i + 1], 2.0 * drift,
                                       rtol=0.0, atol=4 * np.finfo(float).eps)
        # An odd population gives its last particle a zero thermal draw.
        np.testing.assert_array_equal(v[-1], drift)

    def test_sample_is_independent_of_population_size(self):
        # The counter-based generator's defining property: particle p's draw
        # depends on p, not on how many particles were requested. A stream
        # generator cannot do this, and it is what makes a partitioned sample
        # agree with an unpartitioned one.
        small = _velocities(_sample(16, 0.0, 1.0, 0.0, 1.0,
                                    thermal_speed=0.1, seed=99))
        large = _velocities(_sample(64, 0.0, 1.0, 0.0, 1.0,
                                    thermal_speed=0.1, seed=99))
        np.testing.assert_array_equal(small, large[:16])

    def test_thermal_distribution_has_the_requested_width(self):
        n = 20000
        thermal = 0.01
        v = _velocities(_sample(n, 0.0, 1.0, 0.0, 1.0,
                                thermal_speed=thermal, seed=11))
        # Three components of n draws; the standard error of the sample
        # standard deviation is thermal/sqrt(2*3n), so 5 sigma is ~2%.
        np.testing.assert_allclose(np.std(v, axis=0), thermal,
                                   rtol=0.02, atol=0.0)

    def test_superluminal_and_nonfinite_samples_are_rejected(self):
        with self.assertRaises(Exception):
            _sample(64, 0.0, 1.0, 0.0, 1.0, thermal_speed=10.0, seed=0)
        largest = np.finfo(float).max
        with self.assertRaises(Exception):
            _sample(2, 0.0, 1.0, 0.0, 1.0, thermal_speed=largest,
                    drift=(largest, largest, largest), seed=0)


class VelocityPerturbationTests(unittest.TestCase):

    def test_perturbation_adds_the_expected_sinusoid(self):
        n = 128
        amplitude = (0.003, -0.002, 0.001)
        phase = 0.25
        base = _sample(n, 0.0, 1.0, 0.0, 1.0, thermal_speed=0.01, seed=5)
        perturbed = _sample(n, 0.0, 1.0, 0.0, 1.0, thermal_speed=0.01, seed=5,
                            perturb=(2.0, 1.0, phase) + amplitude)
        pts = _positions(base)
        expected = np.sin(2.0 * np.pi * (2.0 * pts[:, 0] + 1.0 * pts[:, 1])
                          + phase)
        delta = _velocities(perturbed) - _velocities(base)
        np.testing.assert_allclose(
            delta, expected[:, None] * np.asarray(amplitude),
            rtol=0.0, atol=8 * np.finfo(float).eps)


class SolverDomainTests(unittest.TestCase):
    """The solver entry point overrides the domain with its own grid."""

    def _solver(self, lx=1.0, ly=1.0, origin_x=0.0, origin_y=0.0):
        cfg = pic.EmPicConfig()
        cfg.grid = pic.Grid2D(nx=16, ny=16, lx=lx, ly=ly,
                              origin_x=origin_x, origin_y=origin_y, nghost=2)
        return pic.EmPic2D3V(cfg)

    def _seed(self, solver, count, x_min, x_max, y_min, y_max):
        config = pic.ParticleSampleConfig()
        config.count = count
        config.x_min, config.x_max = x_min, x_max
        config.y_min, config.y_max = y_min, y_max
        config.weight = 1.0
        index = solver.add_species(
            pic.SpeciesConfig(name="s", charge=-1.0, mass=1.0,
                              capacity=count))
        solver.sample_species_particles(index, config)
        return index

    def test_block_inside_the_domain_is_accepted(self):
        solver = self._solver()
        index = self._seed(solver, 32, 0.25, 0.75, 0.25, 0.75)
        self.assertEqual(solver.species_at(index).size, 32)

    def test_block_outside_the_domain_is_rejected(self):
        # The kernel checks every coordinate it produces against the solver's
        # grid, which is the device counterpart of the host loop
        # set_species_particles ran over an uploaded array.
        solver = self._solver()
        with self.assertRaises(Exception):
            self._seed(solver, 32, 0.5, 1.5, 0.25, 0.75)

    def test_a_rejected_sample_leaves_the_species_empty(self):
        solver = self._solver()
        with self.assertRaises(Exception):
            self._seed(solver, 32, -1.0, 2.0, 0.25, 0.75)
        self.assertEqual(solver.species_at(0).size, 0)


if __name__ == "__main__":
    unittest.main()
