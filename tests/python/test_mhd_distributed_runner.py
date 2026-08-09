"""CPU-focused tests for distributed MHD Python orchestration and I/O."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from quasar import distributed
from quasar.mhd import _distributed_runner as runner
from quasar.mhd import io as mhd_io


class _ConsensusSession:
    rank = 0

    def __init__(self) -> None:
        self.calls: list[tuple[bool, str, str]] = []
        self.agreements: list[tuple[str, str, str]] = []

    def collective_require(
            self, success: bool, phase: str, message: str = "") -> None:
        self.calls.append((success, phase, message))
        if not success:
            raise RuntimeError(f"collective failure in {phase}: {message}")

    def collective_agree(
            self, value: str, phase: str, message: str = "") -> None:
        self.agreements.append((value, phase, message))


class _RejectingPolicySession(_ConsensusSession):

    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.configured = False

    def collective_agree(
            self, value: str, phase: str, message: str = "") -> None:
        super().collective_agree(value, phase, message)
        raise RuntimeError(message)

    def configure_devices(self, _devices: object) -> list[dict]:
        self.configured = True
        raise AssertionError("device configuration followed failed agreement")

    def close(self) -> None:
        self.closed = True


def _runner_state(
        deck: mhd_io.MhdDeck, offset: float = 0.0,
        ) -> dict[str, np.ndarray]:
    size = deck.domain.ny * deck.domain.nx
    base = np.arange(size, dtype=np.float64).reshape(
        deck.domain.ny, deck.domain.nx)
    return {
        name: base + offset + 100.0 * index
        for index, name in enumerate(mhd_io.STATE_COMPONENTS)
    }


class _RunSession(_ConsensusSession):

    def __init__(self, deck: mhd_io.MhdDeck, *,
                 restart_fragment: bytes | None = None) -> None:
        super().__init__()
        self.deck = deck
        self.closed = False
        self.mapping = [{
            "index": 0, "rank": 0, "node_rank": 0,
            "rank_local_index": 0, "device_identity": "uuid:local",
        }]
        self.topology = {
            "decomposition": (1, 1),
            "tiles": [{
                "endpoint": 0, "tile": (0, 0), "offset": (0, 0),
                "owned_shape": (deck.domain.ny, deck.domain.nx),
            }],
        }
        self.state = _runner_state(deck)
        self.restart_fragment = restart_fragment
        self.started = False
        self.restart_path: str | None = None
        self.topology_request: tuple[object, ...] | None = None
        self.step_calls: list[tuple[float, bool]] = []
        self.checkpoints: list[tuple[str, int, float, str, bytes]] = []

    def configure_devices(self, _devices: object) -> list[dict]:
        if not self.agreements:
            raise AssertionError(
                "run policy must be agreed before device configuration")
        return self.mapping

    def select_topology(
            self, nx: int, ny: int, decomposition: object,
            minimum_tile_width: int) -> dict:
        self.topology_request = (
            nx, ny, decomposition, minimum_tile_width)
        return self.topology

    def start_mhd(
            self, _config: object, _state: dict,
            _background: object, *, transport: str = "auto") -> None:
        del transport
        self.started = True

    def restart_mhd(
            self, _config: object, path: str, _units: str,
            _background: object, *, transport: str = "auto") -> dict:
        del transport
        if self.restart_fragment is None:
            raise AssertionError("restart requested without a diagnostic fragment")
        self.restart_path = path
        self.state = _runner_state(self.deck, offset=2.0)
        return {
            "step": 2,
            "time": 0.05,
            "diagnostic_state": [self.restart_fragment],
        }

    def mhd_cfl_limit(self) -> float:
        return 0.1

    def mhd_step(self, dt: float, *, check_cfl: bool = True) -> None:
        self.step_calls.append((dt, check_cfl))
        for name in self.state:
            self.state[name] = self.state[name] + 1.0

    def mhd_gather_cell_component(self, name: str) -> np.ndarray:
        return np.asarray(self.state[name]).reshape(-1)

    def mhd_global_cell_sums(self) -> dict[str, float]:
        return {
            "rho": float(np.sum(self.state["rho"])),
            "energy": float(np.sum(self.state["energy"])),
        }

    def mhd_divergence_b_max(self) -> float:
        return 1.0e-12 * (len(self.step_calls) + 1)

    def mhd_write_checkpoint(
            self, path: str, step: int, time_s: float, units: str,
            diagnostic_state: bytes) -> None:
        self.checkpoints.append(
            (path, step, time_s, units, diagnostic_state))

    @property
    def telemetry(self) -> dict:
        return {
            "mhd": {
                "transport": {
                    "requested": "auto",
                    "interprocess": "staged",
                    "direct_query_recognized": False,
                    "direct_startup_probe": False,
                },
            },
        }

    def close(self) -> None:
        self.closed = True


def _deck() -> SimpleNamespace:
    return SimpleNamespace(
        units="normalized",
        geometry="cartesian",
        domain=SimpleNamespace(nx=5, ny=3, lx_m=2.0, ly_m=1.0),
        numerics=SimpleNamespace(gamma=5.0 / 3.0),
        diagnostics=SimpleNamespace(
            divb=True, fields=("rho", "bx"), cadence=1),
    )


def _runner_deck() -> mhd_io.MhdDeck:
    deck = mhd_io.MhdDeck(
        domain=mhd_io.Domain(nx=8, ny=8, lx_m=2.0, ly_m=1.0),
        numerics=mhd_io.Numerics(reconstruction="muscl_minmod"),
        initial=mhd_io.Initial(type="orszag_tang"),
        time=mhd_io.Time(dt_s=0.025, steps=5, t_end=0.1),
        diagnostics=mhd_io.Diagnostics(
            output_path="out.npz", cadence=2,
            fields=["rho", "bx"], divb=True),
    )
    deck.validate()
    return deck


def _state(offset: float = 0.0) -> dict[str, np.ndarray]:
    values = np.arange(15, dtype=np.float64).reshape(3, 5) + offset
    return {
        name: values + index * 100.0
        for index, name in enumerate(
            ("rho", "mx", "my", "mz", "energy", "bx", "by", "bz"))
    }


class CollectivePythonPhaseTests(unittest.TestCase):

    def test_success_returns_value_and_records_consensus(self):
        session = _ConsensusSession()

        value = runner._collective_local(session, "prepare", lambda: 42)

        self.assertEqual(value, 42)
        self.assertEqual(session.calls, [(True, "prepare", "")])

    def test_local_exception_becomes_collective_failure(self):
        session = _ConsensusSession()

        def fail():
            raise OSError("disk unavailable")

        with self.assertRaisesRegex(RuntimeError, "disk unavailable"):
            runner._collective_local(session, "write", fail)
        self.assertEqual(session.calls[0][0:2], (False, "write"))
        self.assertIn("OSError: disk unavailable", session.calls[0][2])


class RunPolicyTests(unittest.TestCase):

    def test_timestep_signature_excludes_absolute_termination_targets(self):
        original = _runner_deck()
        extended = _runner_deck()
        extended.time.steps = 500
        extended.time.t_end = 10.0

        self.assertEqual(
            runner._timestep_signature(original),
            runner._timestep_signature(extended))

    def test_policy_mismatch_stops_before_device_or_topology_collectives(self):
        deck = _runner_deck()
        session = _RejectingPolicySession()
        config = SimpleNamespace(timestep_signature=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "committed.h5"
            restart = root / "restart.h5"
            options = distributed.RunOptions(
                devices=(2,), decomposition=(1, 1), transport="staged",
                diagnostics_layout="sharded", checkpoint=checkpoint,
                checkpoint_every=4, restart=restart)
            with (mock.patch.object(
                      runner._distributed, "RuntimeSession",
                      return_value=session),
                  mock.patch.object(Path, "cwd", return_value=root),
                  mock.patch.object(
                      runner, "_canonical_state", return_value={}),
                  mock.patch(
                      "quasar.mhd.cli._make_config", return_value=config)):
                with self.assertRaisesRegex(
                        RuntimeError, "different distributed MHD run policies"):
                    runner.run(
                        deck, options, verbose=True, print_config=True,
                        log_every=11)

        self.assertFalse(session.configured)
        self.assertTrue(session.closed)
        self.assertEqual(len(session.agreements), 1)
        signature, phase, message = session.agreements[0]
        self.assertEqual(phase, "mhd-run-policy")
        self.assertIn("different distributed MHD run policies", message)
        policy = json.loads(signature)
        self.assertEqual(policy["physics"], "mhd")
        self.assertEqual(policy["mode"], "restart")
        self.assertEqual(policy["restart_path"], str(restart.resolve()))
        self.assertEqual(policy["checkpoint"], {
            "path": str(checkpoint.resolve()), "cadence": 4})
        self.assertEqual(policy["diagnostics"], {
            "output_path": str((root / "out.npz").resolve()),
            "layout": "sharded", "cadence": 2,
            "fields": ["rho", "bx"], "divb": True,
        })
        self.assertEqual(policy["termination"], {
            "steps": 5, "end_time": 0.1})
        self.assertEqual(policy["timestep"], 0.025)
        self.assertEqual(policy["placement"], {
            "devices": [2], "decomposition": [1, 1],
            "transport": "staged",
        })
        self.assertEqual(policy["log_cadence"], 11)
        self.assertEqual(
            config.timestep_signature,
            f"policy=fixed;dt={float(deck.time.dt_s).hex()}")

    def test_deck_path_run_steps_checkpoints_decodes_and_closes(self):
        deck = _runner_deck()
        session = _RunSession(deck)
        config = SimpleNamespace(timestep_signature=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deck_directory = root / "deck"
            deck_directory.mkdir()
            deck_path = deck_directory / "input.yaml"
            deck_path.touch()
            checkpoint = root / "state.h5"
            options = distributed.RunOptions(
                devices=(0,), decomposition=(1, 1), transport="auto",
                diagnostics_layout="gathered", checkpoint=checkpoint,
                checkpoint_every=2)
            with (mock.patch.object(
                      runner._distributed, "RuntimeSession",
                      return_value=session),
                  mock.patch.object(
                      runner.mhd_io, "load", return_value=deck) as load,
                  mock.patch.object(
                      runner, "_canonical_state", return_value={}),
                  mock.patch(
                      "quasar.mhd.cli._make_config", return_value=config)):
                result = runner.run(deck_path, options)

            load.assert_called_once_with(deck_path.resolve())
            self.assertEqual(result["final_step"], 4)
            self.assertAlmostEqual(result["final_time"], 0.1)
            self.assertEqual(
                result["diagnostics_path"], deck_directory / "out.npz")
            self.assertEqual(session.topology_request, (8, 8, (1, 1), 2))
            self.assertTrue(session.started)
            self.assertTrue(session.closed)
            self.assertEqual(len(session.step_calls), 4)
            self.assertEqual(
                [(item[1], item[2]) for item in session.checkpoints],
                [(2, 0.05), (4, 0.1)])
            self.assertTrue(all(
                item[0] == str(checkpoint.resolve())
                for item in session.checkpoints))
            initial, snapshots, divb, extras = (
                runner._decode_mhd_checkpoint_history(
                    [session.checkpoints[-1][4]], deck))
            self.assertEqual([item["step"] for item in snapshots], [2, 4])
            self.assertGreaterEqual(len(divb), 3)
            self.assertIn("mass_initial", extras)
            np.testing.assert_array_equal(
                initial["rho"], _runner_state(deck)["rho"])
            with np.load(
                    deck_directory / "out.npz", allow_pickle=False) as output:
                np.testing.assert_array_equal(output["snapshot_steps"], [2, 4])
                self.assertEqual(int(output["final_step"][0]), 4)

    def test_restart_reconstructs_history_before_continuing(self):
        deck = _runner_deck()
        initial = _runner_state(deck)
        checkpoint_state = _runner_state(deck, offset=2.0)
        fragment = runner._mhd_checkpoint_fragment(
            0, "gathered", deck, initial, {}, [{
                "step": 2,
                "time_s": 0.05,
                "fields": {
                    name: checkpoint_state[name]
                    for name in deck.diagnostics.fields
                },
                "checkpoint_fields": checkpoint_state,
                "divb": 2.0e-12,
            }], {}, [1.0e-12, 2.0e-12], {
                "mass_initial": float(np.sum(initial["rho"])),
                "energy_initial": float(np.sum(initial["energy"])),
            })
        session = _RunSession(deck, restart_fragment=fragment)
        config = SimpleNamespace(timestep_signature=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = distributed.RunOptions(
                devices=(0,), decomposition=(1, 1),
                diagnostics_layout="gathered", restart=root / "restart.h5")
            with (mock.patch.object(
                      runner._distributed, "RuntimeSession",
                      return_value=session),
                  mock.patch.object(Path, "cwd", return_value=root),
                  mock.patch.object(
                      runner, "_canonical_state", return_value={}),
                  mock.patch(
                      "quasar.mhd.cli._make_config", return_value=config)):
                result = runner.run(deck, options)

            self.assertEqual(result["final_step"], 4)
            self.assertEqual(len(session.step_calls), 2)
            self.assertEqual(session.restart_path, str((root / "restart.h5").resolve()))
            self.assertTrue(session.closed)
            with np.load(root / "out.npz", allow_pickle=False) as output:
                np.testing.assert_array_equal(output["snapshot_steps"], [2, 4])
                np.testing.assert_array_equal(
                    output["state_rho_initial"], initial["rho"])


class DiagnosticsWriterTests(unittest.TestCase):

    def test_restart_history_without_divb_survives_enabled_diagnostics(self):
        deck = _deck()
        deck.diagnostics.divb = False
        initial = _state()
        evolved = _state(1.0)
        checkpoint_snapshot = {
            "step": 2,
            "time_s": 0.25,
            "fields": {
                name: evolved[name] for name in deck.diagnostics.fields},
            "checkpoint_fields": evolved,
        }
        fragment = runner._mhd_checkpoint_fragment(
            0, "gathered", deck, initial, {}, [checkpoint_snapshot], {},
            [], {})

        deck.diagnostics.divb = True
        canonical_initial, restored, _, _ = (
            runner._decode_mhd_checkpoint_history([fragment], deck))
        self.assertNotIn("divb", restored[0])
        restored[0]["fields"] = {
            name: restored[0]["checkpoint_fields"][name]
            for name in deck.diagnostics.fields}

        gathered = runner._gathered_flat(
            deck, evolved, canonical_initial, 2, 0.25, [0.25], restored,
            {}, 4)
        self.assertTrue(np.isnan(gathered["snapshot_divb_linf"][0]))

        mapping = [{
            "index": 0, "rank": 0, "node_rank": 0,
            "rank_local_index": 0, "device_identity": "uuid:a",
        }]
        topology = {
            "decomposition": (1, 1),
            "tiles": [{
                "endpoint": 0, "tile": (0, 0), "offset": (0, 0),
                "owned_shape": (3, 5),
            }],
        }
        metadata = {
            "endpoint": 0, "tile": (0, 0), "offset": (0, 0),
            "owned_shape": (3, 5),
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.npz"
            runner._write_sharded(
                _ConsensusSession(), output, deck,
                [{**metadata, "state": evolved}],
                {0: {**metadata, "state": canonical_initial}},
                2, 0.25, [0.25], {0: restored}, {}, 4, mapping, topology)
            with np.load(
                    Path(directory) / "out.rank000000.gpu000.npz",
                    allow_pickle=False) as payload:
                self.assertTrue(np.isnan(payload["snapshot_divb_linf"][0]))

    def test_checkpoint_history_reassembles_shards_and_original_initial_state(self):
        deck = _deck()
        initial = _state()
        evolved = _state(7.0)
        snapshot = {
            "step": 2, "time_s": 0.25,
            "fields": {"rho": evolved["rho"], "bx": evolved["bx"]},
            "checkpoint_fields": evolved,
            "divb": 0.125,
        }
        gathered = runner._mhd_checkpoint_fragment(
            0, "gathered", deck, initial, {}, [snapshot], {},
            [0.5, 0.125], {"mass_initial": 12.0})
        restored = runner._decode_mhd_checkpoint_history([gathered], deck)
        for name in initial:
            np.testing.assert_array_equal(restored[0][name], initial[name])
            np.testing.assert_array_equal(
                restored[1][0]["checkpoint_fields"][name], evolved[name])
        self.assertEqual(restored[2], [0.5, 0.125])
        self.assertEqual(restored[3], {"mass_initial": 12.0})

        fragments = []
        for rank, (begin, width) in enumerate(((0, 3), (3, 2))):
            endpoint = rank
            shard = {
                "endpoint": endpoint,
                "offset": (0, begin),
                "owned_shape": (3, width),
                "state": {
                    name: values[:, begin:begin + width]
                    for name, values in initial.items()},
            }
            local_snapshot = {
                "step": 2, "time_s": 0.25, "divb": 0.125,
                "fields": {
                    "rho": evolved["rho"][:, begin:begin + width],
                    "bx": evolved["bx"][:, begin:begin + width],
                },
                "checkpoint_fields": {
                    name: values[:, begin:begin + width]
                    for name, values in evolved.items()},
            }
            fragments.append(runner._mhd_checkpoint_fragment(
                rank, "sharded", deck, None, {endpoint: shard}, [],
                {endpoint: [local_snapshot]}, [0.5, 0.125],
                {"mass_initial": 12.0}))
        repartitioned = runner._decode_mhd_checkpoint_history(fragments, deck)
        for name in initial:
            np.testing.assert_array_equal(repartitioned[0][name], initial[name])
            np.testing.assert_array_equal(
                repartitioned[1][0]["checkpoint_fields"][name], evolved[name])

    def test_gathered_payload_preserves_serial_keys_and_shapes(self):
        deck = _deck()
        final = _state(1.0)
        initial = _state()
        snapshot = {
            "step": 2,
            "time_s": 0.25,
            "fields": {"rho": final["rho"], "bx": final["bx"]},
            "divb": 0.125,
        }

        payload = runner._gathered_flat(
            deck, final, initial, 2, 0.25, [0.5, 0.125], [snapshot],
            {"mass_initial": 12.0}, 4)

        for name in final:
            self.assertIn(f"state_{name}", payload)
            self.assertIn(f"state_{name}_initial", payload)
            self.assertEqual(payload[f"state_{name}"].shape, (3, 5))
        self.assertEqual(payload["snapshot_state_rho"].shape, (1, 3, 5))
        np.testing.assert_array_equal(payload["snapshot_steps"], [2])
        np.testing.assert_allclose(payload["divb_linf"], [0.5, 0.125])

    def test_shards_reconstruct_owned_fields_and_manifest_is_published_last(self):
        deck = _deck()
        final = _state(1.0)
        initial = _state()
        snapshot = {
            "step": 2,
            "time_s": 0.25,
            "fields": {"rho": final["rho"], "bx": final["bx"]},
            "divb": 0.125,
        }
        mapping = [
            {"index": 0, "rank": 0, "node_rank": 0,
             "rank_local_index": 0, "device_identity": "uuid:a"},
            {"index": 1, "rank": 0, "node_rank": 0,
             "rank_local_index": 1, "device_identity": "uuid:b"},
        ]
        topology = {
            "decomposition": (2, 1),
            "tiles": [
                {"endpoint": 0, "tile": (0, 0), "offset": (0, 0),
                 "owned_shape": (3, 3)},
                {"endpoint": 1, "tile": (1, 0), "offset": (0, 3),
                 "owned_shape": (3, 2)},
            ],
        }
        session = _ConsensusSession()
        local_shards = []
        initial_shards = {}
        local_snapshots = {}
        for tile in topology["tiles"]:
            endpoint = int(tile["endpoint"])
            offset_y, offset_x = tile["offset"]
            owned_ny, owned_nx = tile["owned_shape"]
            rows = slice(offset_y, offset_y + owned_ny)
            columns = slice(offset_x, offset_x + owned_nx)
            metadata = {
                "endpoint": endpoint,
                "tile": tile["tile"],
                "offset": tile["offset"],
                "owned_shape": tile["owned_shape"],
            }
            local_shards.append({
                **metadata,
                "state": {
                    name: values[rows, columns]
                    for name, values in final.items()
                },
            })
            initial_shards[endpoint] = {
                **metadata,
                "state": {
                    name: values[rows, columns]
                    for name, values in initial.items()
                },
            }
            local_snapshots[endpoint] = [{
                "step": snapshot["step"],
                "time_s": snapshot["time_s"],
                "fields": {
                    name: values[rows, columns]
                    for name, values in snapshot["fields"].items()
                },
                "divb": snapshot["divb"],
            }]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.npz"
            manifest_path = runner._write_sharded(
                session, output, deck, local_shards, initial_shards,
                2, 0.25, [0.5, 0.125], local_snapshots,
                {"mass_initial": 12.0}, 4, mapping, topology)
            manifest = distributed.read_diagnostics_manifest(manifest_path)

            self.assertEqual(manifest.global_shape, (3, 5))
            self.assertEqual(manifest.decomposition, (2, 1))
            rebuilt = np.empty((3, 5), dtype=np.float64)
            for shard in manifest.shards:
                with np.load(shard.path, allow_pickle=False) as payload:
                    y, x = shard.offset
                    ny, nx = shard.owned_shape
                    rebuilt[y:y + ny, x:x + nx] = payload["state_rho"]
                    self.assertEqual(
                        payload["snapshot_state_rho"].shape, (1, ny, nx))
                    np.testing.assert_array_equal(
                        payload["state_rho_initial"],
                        initial["rho"][y:y + ny, x:x + nx])
            np.testing.assert_array_equal(rebuilt, final["rho"])
            self.assertFalse((Path(directory) / "out.manifest.json.tmp").exists())

        self.assertEqual(
            [call[1] for call in session.calls],
            ["mhd-diagnostics-sharded-prepare",
             "mhd-diagnostics-sharded-begin",
             "mhd-diagnostics-shard-write",
             "mhd-diagnostics-manifest-publish"])


class ManifestValidationTests(unittest.TestCase):

    def test_duplicate_normalized_shard_paths_are_rejected(self):
        document = {
            "schema": "quasar-diagnostics-shards/v1",
            "physics": "mhd",
            "geometry": "cartesian",
            "global_shape": [1, 2],
            "step": 1,
            "time": 0.1,
            "decomposition": {"px": 2, "py": 1},
            "shards": [
                {"rank": 0, "node_rank": 0, "local_device": 0,
                 "endpoint": 0, "device_identity": "uuid:a",
                 "tile": [0, 0], "offset": [0, 0],
                 "owned_shape": [1, 1], "path": "same.npz"},
                {"rank": 0, "node_rank": 0, "local_device": 1,
                 "endpoint": 1, "device_identity": "uuid:b",
                 "tile": [1, 0], "offset": [0, 1],
                 "owned_shape": [1, 1], "path": "./same.npz"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.manifest.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "paths must be unique"):
                distributed.read_diagnostics_manifest(
                    path, verify_shards=False)


if __name__ == "__main__":
    unittest.main()
