import unittest

import numpy as np

from quasar.pic.cli import _build_parser, _flatten_for_npz


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


class FlattenForNpzTests(unittest.TestCase):

    def _final(self):
        return {
            "step": 10,
            "time_s": 1.0e-9,
            "nx": 4,
            "ny": 4,
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


if __name__ == "__main__":
    unittest.main()
