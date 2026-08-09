"""CPU-only tests for the public distributed Python contract."""

import inspect
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np

import quasar.distributed as distributed
from quasar import _checkpoint_diagnostics as checkpoint_diagnostics
from quasar import _distributed_helpers as distributed_helpers
from quasar.mhd import cli as mhd_cli
from quasar.pic import cli as pic_cli


class AvailabilityTests(unittest.TestCase):

    def test_serial_only_module_remains_importable_and_reports_reason(self):
        missing = ModuleNotFoundError(
            "No module named 'quasar._distributed'",
            name="quasar._distributed")
        with (patch.object(distributed, "_native", None),
              patch.object(distributed, "_native_import_error", missing)):
            self.assertIs(distributed.is_available(), False)
            self.assertIn("does not include distributed support",
                          distributed.unavailable_reason())
            with self.assertRaisesRegex(
                    distributed.DistributedUnavailableError,
                    "distributed execution is unavailable"):
                distributed.require_available()

    def test_broken_native_probe_is_not_reported_as_available(self):
        class BrokenNative:
            @staticmethod
            def is_available():
                raise RuntimeError("probe failed")

        with patch.object(distributed, "_native", BrokenNative()):
            self.assertIs(distributed.is_available(), False)

    def test_runtime_session_placeholder_fails_clearly_without_native_foundation(self):
        if distributed.foundation_available():
            self.skipTest("native distributed foundation is present")
        with self.assertRaisesRegex(
                distributed.DistributedUnavailableError,
                "runtime foundation is unavailable"):
            distributed.RuntimeSession()


class RunOptionsTests(unittest.TestCase):

    def test_defaults_are_explicit_distributed_runtime_defaults(self):
        options = distributed.RunOptions()
        self.assertEqual(options.devices, "auto")
        self.assertEqual(options.decomposition, "auto")
        self.assertEqual(options.transport, "auto")
        self.assertEqual(options.diagnostics_layout, "gathered")
        self.assertIsNone(options.checkpoint)
        self.assertIsNone(options.restart)

    def test_normalizes_device_decomposition_and_checkpoint_values(self):
        options = distributed.RunOptions(
            devices="3, 1", decomposition="2X1", transport="STAGED",
            diagnostics_layout="SHARDED", checkpoint="state.h5",
            checkpoint_every=8, restart=Path("old.h5"))
        self.assertEqual(options.devices, (3, 1))
        self.assertEqual(options.decomposition, (2, 1))
        self.assertEqual(options.transport, "staged")
        self.assertEqual(options.diagnostics_layout, "sharded")
        self.assertEqual(options.checkpoint, Path("state.h5"))
        self.assertEqual(options.restart, Path("old.h5"))
        self.assertEqual(options.as_dict()["checkpoint"], "state.h5")

    def test_rejects_invalid_or_ambiguous_values(self):
        invalid = (
            {"devices": []},
            {"devices": "0,0"},
            {"devices": [-1]},
            {"decomposition": "2x0"},
            {"decomposition": (1, 2, 3)},
            {"transport": "maybe"},
            {"diagnostics_layout": "rank-zero"},
            {"checkpoint_every": 2},
            {"checkpoint": "state.h5", "checkpoint_every": 0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                distributed.RunOptions(**kwargs)


class RunResultTests(unittest.TestCase):

    def test_validates_and_exposes_compatibility_aliases(self):
        result = distributed.RunResult(
            final_step=7, final_time=1.25, diagnostics_path="out.npz",
            telemetry={"transport": "staged"})
        self.assertEqual(result.steps_completed, 7)
        self.assertEqual(result.final_time_s, 1.25)
        self.assertEqual(result.output_path, Path("out.npz"))
        self.assertEqual(result.telemetry["transport"], "staged")
        with self.assertRaises(TypeError):
            result.telemetry["new"] = 1

    def test_rejects_uncommitted_result_values(self):
        for kwargs in (
                {"final_step": -1, "final_time": 0.0},
                {"final_step": 0, "final_time": float("nan")},
                {"final_step": 0, "final_time": -1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                distributed.RunResult(**kwargs)


class CheckpointDiagnosticPayloadTests(unittest.TestCase):

    def test_non_pickle_npz_roundtrip_and_corruption_rejection(self):
        encoded = checkpoint_diagnostics.encode_fragment({
            "schema": np.array(["quasar-test/v1"]),
            "values": np.arange(12, dtype=np.float64).reshape(3, 4),
        })
        decoded = checkpoint_diagnostics.decode_fragment(encoded)
        self.assertEqual(str(decoded["schema"][0]), "quasar-test/v1")
        np.testing.assert_array_equal(
            decoded["values"], np.arange(12).reshape(3, 4))

        with self.assertRaisesRegex(ValueError, "object arrays"):
            checkpoint_diagnostics.encode_fragment({
                "unsafe": np.asarray([{"pickle": True}], dtype=object)})
        with self.assertRaises(ValueError):
            checkpoint_diagnostics.decode_fragment(encoded[:-7])


class AtomicPublicationTests(unittest.TestCase):

    def test_random_exclusive_temporaries_do_not_follow_legacy_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            victim = root / "victim.txt"
            victim.write_text("unchanged", encoding="utf-8")

            npz_path = root / "out.npz"
            npz_legacy_temporary = root / "out.npz.tmp"
            npz_legacy_temporary.symlink_to(victim)
            distributed_helpers.atomic_savez(
                npz_path, {"value": np.arange(3, dtype=np.int64)})

            json_path = root / "out.manifest.json"
            json_legacy_temporary = root / "out.manifest.json.tmp"
            json_legacy_temporary.symlink_to(victim)
            distributed_helpers.atomic_json(json_path, {"complete": True})

            self.assertEqual(victim.read_text(encoding="utf-8"), "unchanged")
            self.assertTrue(npz_legacy_temporary.is_symlink())
            self.assertTrue(json_legacy_temporary.is_symlink())
            with np.load(npz_path, allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["value"], [0, 1, 2])
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")),
                {"complete": True})
            self.assertEqual(list(root.glob(".out.npz.*.tmp")), [])
            self.assertEqual(list(root.glob(".out.manifest.json.*.tmp")), [])


class DiagnosticsManifestTests(unittest.TestCase):

    def _document(self):
        return {
            "schema": "quasar-diagnostics-shards/v1",
            "physics": "pic",
            "geometry": "cartesian",
            "global_shape": [3, 5],
            "step": 8,
            "time": 0.25,
            "decomposition": {"px": 2, "py": 1},
            "shards": [
                {
                    "rank": 0, "node_rank": 0, "local_device": 0,
                    "endpoint": 0, "device_identity": "uuid:a",
                    "tile": [0, 0], "offset": [0, 0],
                    "owned_shape": [3, 3], "path": "out.0.npz",
                },
                {
                    "rank": 1, "node_rank": 0, "local_device": 0,
                    "endpoint": 1, "device_identity": "uuid:b",
                    "tile": [1, 0], "offset": [0, 3],
                    "owned_shape": [3, 2], "path": "out.1.npz",
                },
            ],
        }

    def test_reads_completed_manifest_and_resolves_relative_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("out.0.npz", "out.1.npz"):
                (root / name).touch()
            path = root / "out.manifest.json"
            path.write_text(json.dumps(self._document()), encoding="utf-8")

            manifest = distributed.read_diagnostics_manifest(path)

            self.assertEqual(manifest.global_shape, (3, 5))
            self.assertEqual(manifest.decomposition, (2, 1))
            self.assertEqual([shard.endpoint for shard in manifest.shards], [0, 1])
            self.assertEqual(manifest.shards[1].path, (root / "out.1.npz").resolve())

    def test_registered_third_physics_manifest_is_forward_compatible(self):
        with (patch.dict(distributed._RUNNER_REGISTRY),
              tempfile.TemporaryDirectory() as directory):
            distributed._register_runner("radiation", lambda *_args, **_kwargs: None)
            root = Path(directory)
            document = self._document()
            document["physics"] = "radiation"
            path = root / "out.manifest.json"
            path.write_text(json.dumps(document), encoding="utf-8")

            manifest = distributed.read_diagnostics_manifest(
                path, verify_shards=False)

            self.assertEqual(manifest.physics, "radiation")

    def test_rejects_checkpoint_schema_overlap_and_missing_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "out.manifest.json"
            for invalid_physics in (None, "", 17):
                with self.subTest(physics=invalid_physics):
                    document = self._document()
                    document["physics"] = invalid_physics
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "non-empty string"):
                        distributed.read_diagnostics_manifest(
                            path, verify_shards=False)

            document = self._document()
            document["schema"] = "quasar-checkpoint/v1"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                distributed.read_diagnostics_manifest(path, verify_shards=False)

            document = self._document()
            document["shards"][1]["offset"] = [0, 2]
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                distributed.read_diagnostics_manifest(path, verify_shards=False)

            document = self._document()
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing"):
                distributed.read_diagnostics_manifest(path)


class CliContractTests(unittest.TestCase):

    def test_plain_pic_and_mhd_invocations_do_not_request_distribution(self):
        for parser in (pic_cli._build_parser(), mhd_cli._build_parser()):
            with self.subTest(prog=parser.prog):
                args = parser.parse_args(["run", "deck.yaml"])
                self.assertIsNone(args.devices)
                self.assertIsNone(args.decomposition)
                self.assertIsNone(args.transport)
                self.assertIsNone(args.diagnostics_layout)
                self.assertIsNone(args.checkpoint)
                self.assertIsNone(args.checkpoint_every)
                self.assertIsNone(args.restart)

    def test_all_distributed_flags_are_exposed_and_normalized(self):
        argv = [
            "run", "deck.yaml", "--devices", "2,0",
            "--decomposition", "1x2", "--transport", "staged",
            "--diagnostics-layout", "sharded", "--checkpoint", "new.h5",
            "--checkpoint-every", "4", "--restart", "old.h5",
        ]
        for cli in (pic_cli, mhd_cli):
            with self.subTest(module=cli.__name__):
                args = cli._build_parser().parse_args(argv)
                options = cli._distributed_options_from_args(args)
                self.assertEqual(options.devices, (2, 0))
                self.assertEqual(options.decomposition, (1, 2))
                self.assertEqual(options.transport, "staged")
                self.assertEqual(options.diagnostics_layout, "sharded")
                self.assertEqual(options.checkpoint_every, 4)

    def test_any_distributed_only_flag_activates_options(self):
        cases = (
            ("--devices", "auto"),
            ("--decomposition", "1x1"),
            ("--transport", "auto"),
            ("--diagnostics-layout", "gathered"),
            ("--checkpoint", "state.h5"),
            ("--restart", "state.h5"),
        )
        for cli in (pic_cli, mhd_cli):
            for flag, value in cases:
                with self.subTest(module=cli.__name__, flag=flag):
                    args = cli._build_parser().parse_args(
                        ["run", "deck.yaml", flag, value])
                    self.assertIsInstance(
                        cli._distributed_options_from_args(args),
                        distributed.RunOptions)

    def test_pic_parser_preserves_seed_default_and_tracks_explicit_seed(self):
        parser = pic_cli._build_parser()
        implicit = parser.parse_args(["run", "deck.yaml"])
        explicit = parser.parse_args(["run", "deck.yaml", "--seed", "0"])
        self.assertEqual(implicit.seed, 0)
        self.assertIs(implicit._seed_explicit, False)
        self.assertEqual(explicit.seed, 0)
        self.assertIs(explicit._seed_explicit, True)

    def test_checkpoint_cadence_requires_checkpoint_path(self):
        for cli in (pic_cli, mhd_cli):
            args = cli._build_parser().parse_args(
                ["run", "deck.yaml", "--checkpoint-every", "2"])
            with self.subTest(module=cli.__name__), self.assertRaisesRegex(
                    ValueError, "requires checkpoint"):
                cli._distributed_options_from_args(args)

    def test_invalid_combinations_fail_before_deck_or_solver_construction(self):
        cases = (
            (pic_cli, pic_cli.pic_io, ["--checkpoint-every", "2"]),
            (pic_cli, pic_cli.pic_io,
             ["--restart", "state.h5", "--seed", "0"]),
            (mhd_cli, mhd_cli.mhd_io, ["--devices", "0,0"]),
        )
        for cli, deck_io, flags in cases:
            with (self.subTest(module=cli.__name__, flags=flags),
                  patch.object(deck_io, "load") as load,
                  redirect_stderr(io.StringIO()),
                  self.assertRaises(SystemExit) as raised):
                cli.main(["run", "deck.yaml", *flags])
            self.assertEqual(raised.exception.code, 2)
            load.assert_not_called()

    def test_compiled_out_cli_request_fails_clearly_before_deck_load(self):
        missing = ModuleNotFoundError(
            "No module named 'quasar._distributed'",
            name="quasar._distributed")
        for cli, deck_io in ((pic_cli, pic_cli.pic_io),
                             (mhd_cli, mhd_cli.mhd_io)):
            stderr = io.StringIO()
            with (self.subTest(module=cli.__name__),
                  patch.object(distributed, "_native", None),
                  patch.object(distributed, "_native_import_error", missing),
                  patch.object(deck_io, "load") as load,
                  redirect_stderr(stderr),
                  self.assertRaises(SystemExit) as raised):
                cli.main(["run", "deck.yaml", "--devices", "auto"])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("distributed execution is unavailable",
                          stderr.getvalue())
            load.assert_not_called()


class HighLevelRunRoutingTests(unittest.TestCase):

    def test_pic_runtime_probe_routes_to_python_orchestrator(self):
        class Native:
            @staticmethod
            def pic_runtime_available():
                return True

        returned = {
            "final_step": 3,
            "final_time": 0.25,
            "diagnostics_path": "pic-out.npz",
            "distributed": True,
        }
        options = distributed.RunOptions(devices="0", decomposition="1x1")
        with (patch.object(distributed, "_native", Native()),
              patch("quasar.pic._distributed_runner.run",
                    return_value=returned) as orchestrate):
            result = pic_cli.run(
                "input.yaml", options=options, seed=17, write_every=2)

        self.assertEqual(result.final_step, 3)
        self.assertEqual(result.output_path, Path("pic-out.npz"))
        orchestrate.assert_called_once_with(
            "input.yaml", options, seed=17, steps_override=None,
            verbose=False, print_config=False, log_every=0, write_every=2)

    def test_available_native_runner_receives_normalized_options(self):
        calls = []

        class Native:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def run_mhd(input_deck, options, **kwargs):
                calls.append((input_deck, options, kwargs))
                return {
                    "final_step": 4,
                    "final_time": 0.5,
                    "output_path": "out.npz",
                }

        options = distributed.RunOptions(devices="1,0", decomposition="2x1")
        with patch.object(distributed, "_native", Native()):
            result = mhd_cli.run("input.yaml", options=options, log_every=3)
        self.assertIs(result.distributed, True)
        self.assertEqual(result.output_path, Path("out.npz"))
        self.assertEqual(calls[0][0], "input.yaml")
        self.assertEqual(calls[0][1]["devices"], (1, 0))
        self.assertEqual(calls[0][1]["decomposition"], (2, 1))
        self.assertEqual(calls[0][2]["log_every"], 3)

    def test_serial_calls_route_without_changing_prepare_run_signatures(self):
        pic_signature = inspect.signature(pic_cli.prepare_run)
        self.assertEqual(tuple(pic_signature.parameters), ("deck", "units", "seed"))
        self.assertEqual(pic_signature.parameters["seed"].default, 0)
        self.assertEqual(
            pic_signature.parameters["seed"].kind,
            inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(tuple(inspect.signature(mhd_cli.prepare_run).parameters),
                         ("deck",))
        sentinel = distributed.RunResult(0, 0.0)
        with patch.object(pic_cli, "_serial_run", return_value=sentinel) as serial:
            self.assertIs(pic_cli.run("pic.yaml"), sentinel)
            self.assertEqual(serial.call_args.kwargs["seed"], 0)
        with patch.object(mhd_cli, "_serial_run", return_value=sentinel) as serial:
            self.assertIs(mhd_cli.run("mhd.yaml"), sentinel)
            serial.assert_called_once()

    def test_compiled_out_request_fails_before_loading_a_missing_deck(self):
        missing = ModuleNotFoundError(
            "No module named 'quasar._distributed'",
            name="quasar._distributed")
        with (patch.object(distributed, "_native", None),
              patch.object(distributed, "_native_import_error", missing)):
            for runner, name in ((pic_cli.run, "pic"), (mhd_cli.run, "mhd")):
                with self.subTest(physics=name), self.assertRaisesRegex(
                        distributed.DistributedUnavailableError,
                        "does not include distributed support"):
                    runner("does-not-exist.yaml",
                           options=distributed.RunOptions())

    def test_pic_seed_is_rejected_with_restart_before_availability_probe(self):
        with self.assertRaisesRegex(ValueError, "seed cannot be supplied"):
            pic_cli.run(
                "does-not-exist.yaml", seed=7,
                options=distributed.RunOptions(restart="state.h5"))

    def test_public_package_exports_run_lazily(self):
        from quasar import mhd, pic
        self.assertIs(pic.run, pic_cli.run)
        self.assertIs(mhd.run, mhd_cli.run)


class SerialCompatibilityTests(unittest.TestCase):

    def test_new_native_particle_state_does_not_change_legacy_npz_keys(self):
        class Units:
            @staticmethod
            def length_to_si(values):
                return np.asarray(values) * 2.0

            @staticmethod
            def velocity_to_si(values):
                return np.asarray(values) * 3.0

        host = {
            "x": np.array([1.0]), "y": np.array([2.0]),
            "vx": np.array([3.0]), "vy": np.array([4.0]),
            "vz": np.array([5.0]), "weight": np.array([6.0]),
            "alive": np.array([True]),
            "x_prev": np.array([0.5]), "y_prev": np.array([1.5]),
            "vphi_deposit": np.array([7.0]),
            "id": np.array([42], dtype=np.uint64),
        }
        converted = pic_cli._species_to_si(host, Units())
        self.assertEqual(
            set(converted), {"x", "y", "vx", "vy", "vz", "weight", "alive"})
        np.testing.assert_array_equal(converted["x"], [2.0])
        np.testing.assert_array_equal(converted["vy"], [12.0])
        np.testing.assert_array_equal(converted["weight"], [6.0])


if __name__ == "__main__":
    unittest.main()
