"""CPU-focused tests for distributed PIC orchestration and diagnostics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from quasar import _core
from quasar import distributed
from quasar.pic import _distributed_runner as runner
from quasar.pic import io as pic_io
from quasar.pic._units import Units


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


def _deck(*, particles: int = 7, field_seed: bool = False,
          external_field: bool = False) -> pic_io.PicDeck:
    initial_field = (pic_io.FieldsInitial(
        type="seed_perturbation", component="Ey", amplitude=0.125,
        mode=(1, 0)) if field_seed else None)
    external = (pic_io.ExternalField(
        evaluator_type="uniform", uniform_b=(0.25, -0.5, 0.75))
        if external_field else None)
    deck = pic_io.PicDeck(
        domain=pic_io.Domain(nx=5, ny=3, lx_m=2.0, ly_m=1.0),
        numerics=pic_io.Numerics(fdtd_order=2, shape="cic"),
        species=[pic_io.Species(
            name="electrons", charge_C=-1.0, mass_kg=1.0,
            n_particles=particles,
            initial=pic_io.SpeciesInitial(
                distribution="maxwellian_uniform",
                density_per_m3=2.0, temperature_eV=0.01,
                drift_v=(0.02, -0.01, 0.03)))],
        time=pic_io.Time(dt_s=0.05, steps=2),
        diagnostics=pic_io.Diagnostics(
            output_path="out.npz", cadence=1,
            fields=["ey", "bz"], per_species=True),
        fields=pic_io.Fields(initial=initial_field),
        external_field=external,
        units="normalized",
    )
    deck.validate()
    return deck


def _state(deck: pic_io.PicDeck, *, seed: int = 19) -> dict:
    fields = runner._empty_fields(deck)
    external = runner._empty_fields(deck)
    for component, values in fields.items():
        if component.startswith("global_"):
            continue
        array = np.asarray(values)
        fields[component] = np.arange(
            array.size, dtype=np.float64).reshape(array.shape)
        external[component] = fields[component] + 1000.0
    return {
        "fields": fields,
        "external_fields": external,
        "species": runner._species_states(deck, Units(deck), seed),
    }


def _local_shards(
        deck: pic_io.PicDeck, state: dict, mapping: list[dict],
        topology: dict, *, rank: int = 0) -> list[dict]:
    units = Units(deck)
    tiles = {int(item["endpoint"]): item for item in topology["tiles"]}
    endpoint_count = len(mapping)
    result: list[dict] = []
    for endpoint_info in mapping:
        if int(endpoint_info["rank"]) != rank:
            continue
        endpoint = int(endpoint_info["index"])
        tile = tiles[endpoint]

        def owned_fields(source: dict) -> dict:
            fields: dict[str, dict] = {}
            for name in ("ex", "ey", "ez", "bx", "by", "bz"):
                values, offset = runner._component_owned_slice(
                    source[name], name, deck, tile)
                fields[name] = {
                    "offset": offset,
                    "shape": values.shape,
                    "values": np.ascontiguousarray(values).reshape(-1),
                }
            return fields

        offset_y, offset_x = map(int, tile["offset"])
        owned_ny, owned_nx = map(int, tile["owned_shape"])
        local_species = []
        for item in state["species"]:
            particles = item["particles"]
            x = np.asarray(particles["x"], dtype=np.float64)
            y = np.asarray(particles["y"], dtype=np.float64)
            alive = np.asarray(particles["alive"], dtype=np.uint8) != 0
            identifiers = np.asarray(particles["id"], dtype=np.uint64)
            dx = units.length(deck.domain.lx_m) / deck.domain.nx
            dy = units.length(deck.domain.ly_m) / deck.domain.ny
            ix = np.floor((x - units.length(deck.domain.origin_x_m)) / dx)
            iy = np.floor((y - units.length(deck.domain.origin_y_m)) / dy)
            ix = np.minimum(ix.astype(np.int64), deck.domain.nx - 1)
            iy = np.minimum(iy.astype(np.int64), deck.domain.ny - 1)
            mask = (alive & (ix >= offset_x) & (ix < offset_x + owned_nx)
                    & (iy >= offset_y) & (iy < offset_y + owned_ny))
            mask |= (~alive
                     & ((identifiers % np.uint64(endpoint_count))
                        == np.uint64(endpoint)))
            local_species.append({
                "config": dict(item["config"]),
                "particles": {
                    key: np.asarray(values)[mask]
                    for key, values in particles.items()
                },
            })
        result.append({
            "endpoint": endpoint,
            "tile": tuple(tile["tile"]),
            "offset": tuple(tile["offset"]),
            "owned_shape": tuple(tile["owned_shape"]),
            "fields": owned_fields(state["fields"]),
            "external_fields": owned_fields(state["external_fields"]),
            "species": local_species,
        })
    return result


class _ShardedRunSession(_ConsensusSession):

    def __init__(self, deck: pic_io.PicDeck) -> None:
        super().__init__()
        self.deck = deck
        self.closed = False
        self.local_extractions: list[bool] = []
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
        self.state: dict | None = None
        self.transport_requests: list[str] = []
        self.sampled_external: list[tuple[object, ...]] = []
        self.checkpoints: list[tuple[str, int, float, str, bytes]] = []
        self.step_calls = 0

    def configure_devices(self, _devices: object) -> list[dict]:
        if not self.agreements:
            raise AssertionError(
                "run policy must be agreed before device configuration")
        return self.mapping

    def select_topology(
            self, _nx: int, _ny: int, _decomposition: object,
            _nghost: int) -> dict:
        return self.topology

    def start_pic(
            self, _config: object, fields: dict,
            _external_fields: object, species: list[dict], *,
            transport: str = "auto") -> None:
        self.transport_requests.append(transport)
        self.state = {
            "fields": fields,
            "external_fields": runner._empty_fields(self.deck),
            "species": species,
        }

    def pic_sample_external_fields(self, *args: object) -> None:
        self.sampled_external.append(args)

    def pic_cfl_limit(self) -> float:
        return 1.0

    def pic_step(self, _dt: float) -> None:
        self.step_calls += 1

    def pic_gather_state(self) -> dict:
        raise AssertionError("sharded diagnostics must not gather global state")

    def pic_local_owned_shards(
            self, include_particles: bool = True) -> list[dict]:
        assert self.state is not None
        self.local_extractions.append(bool(include_particles))
        shards = _local_shards(
            self.deck, self.state, self.mapping, self.topology, rank=0)
        if not include_particles:
            for shard in shards:
                shard["species"] = []
        return shards

    def pic_alive_counts(self) -> np.ndarray:
        assert self.state is not None
        return np.asarray([
            np.count_nonzero(item["particles"]["alive"])
            for item in self.state["species"]
        ], dtype=np.uint64)

    def pic_kinetic_energies(self) -> np.ndarray:
        assert self.state is not None
        return np.zeros(len(self.state["species"]), dtype=np.float64)

    def pic_total_em_energy(self) -> float:
        return 0.0

    def pic_gauss_residual(self) -> float:
        return 0.0

    def pic_write_checkpoint(
            self, path: str, step: int, time_s: float, unit_system: str,
            diagnostic_state: bytes) -> None:
        self.checkpoints.append(
            (path, step, time_s, unit_system, diagnostic_state))

    @property
    def telemetry(self) -> dict:
        return {
            "pic": {
                "global_state_gathers": 0,
                "local_shard_extractions": len(self.local_extractions),
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


class _RestartRunSession(_ShardedRunSession):

    def __init__(self, deck: pic_io.PicDeck) -> None:
        super().__init__(deck)
        self.restart_config_signature: str | None = None
        self.restart_timestep_signature: str | None = None
        self.restored_external: dict[str, np.ndarray] = {}
        self.restart_events: list[str] = []

    def restart_pic(
            self, config: object, _path: str, _unit_system: str,
            _expected_species: list[dict], *,
            transport: str = "auto") -> dict[str, object]:
        self.restart_events.append("restart")
        self.transport_requests.append(transport)
        self.restart_config_signature = getattr(
            config, "external_field_signature", None)
        self.restart_timestep_signature = getattr(
            config, "timestep_signature", None)
        self.state = _state(self.deck)
        self.restored_external = {
            name: np.array(values, copy=True)
            for name, values in self.state["external_fields"].items()
            if not name.startswith("global_")
        }
        series = {
            "step": [self.deck.time.steps],
            "time_s": [0.1],
            "alive_electrons": [self.deck.species[0].n_particles],
        }
        diagnostic_state = runner._pic_checkpoint_fragment(
            0, "sharded", self.deck, [], {}, series)
        return {
            "step": self.deck.time.steps,
            "time": 0.1,
            "diagnostic_state": [diagnostic_state],
        }


class InitializationTests(unittest.TestCase):

    def test_serial_and_distributed_seeding_draw_the_same_sample(self):
        """One deck seed, one sample, whether run on one rank or many.

        The runner used to carry its own host Philox sampler alongside the
        serial CLI's NumPy one, so the two paths drew different velocities for
        the same deck. Both now drive the sampling kernels through
        ``_core.pic``, and this pins that they agree exactly rather than
        approximately.
        """
        from quasar.pic.cli import _seed_species

        deck = _deck(particles=9)
        units = Units(deck)
        distributed = runner._species_states(deck, units, 42)

        class _Recorder:
            def __init__(self):
                self.host = None

            def add_species(self, cfg):
                return 0

            def sample_species_particles(self, index, config):
                self.host = _core.pic.sample_particles(config)

        recorder = _Recorder()
        _seed_species(recorder, deck, units, 42)

        particles = distributed[0]["particles"]
        for name in ("x", "y", "vx", "vy", "vz", "weight"):
            np.testing.assert_array_equal(
                np.asarray(particles[name]), np.asarray(recorder.host[name]),
                err_msg=f"{name} differs between the serial and distributed "
                        "seeding paths")

    def test_counter_sampling_is_reproducible_antithetic_and_species_keyed(self):
        def sample(count, thermal, drift, seed, species_index):
            config = _core.pic.ParticleSampleConfig()
            config.count = count
            config.x_min, config.x_max = 0.0, 1.0
            config.y_min, config.y_max = 0.0, 1.0
            config.domain_lx = config.domain_ly = 1.0
            config.thermal_speed = thermal
            config.drift_x, config.drift_y, config.drift_z = drift
            config.seed = seed
            config.species_key = species_index
            config.weight = 1.0
            out = _core.pic.sample_particles(config)
            return np.column_stack((out["vx"], out["vy"], out["vz"]))

        first = sample(7, 0.2, (0.1, -0.2, 0.3), 1234, 0)
        repeat = sample(7, 0.2, (0.1, -0.2, 0.3), 1234, 0)
        other_species = sample(7, 0.2, (0.1, -0.2, 0.3), 1234, 1)

        np.testing.assert_array_equal(first, repeat)
        self.assertFalse(np.array_equal(first, other_species))
        thermal = first - np.asarray((0.1, -0.2, 0.3))
        np.testing.assert_allclose(
            thermal[0:6:2] + thermal[1:6:2], 0.0,
            atol=np.finfo(np.float64).eps)
        np.testing.assert_allclose(
            thermal[-1], 0.0, atol=np.finfo(np.float64).eps)

    def test_global_ids_and_initial_state_do_not_depend_on_decomposition(self):
        deck = _deck(particles=9)
        units = Units(deck)

        first = runner._species_states(deck, units, 42)
        repeat = runner._species_states(deck, units, 42)

        np.testing.assert_array_equal(
            first[0]["particles"]["id"], np.arange(9, dtype=np.uint64))
        for name in ("x", "y", "x_prev", "y_prev", "vx", "vy", "vz",
                     "vphi_deposit", "weight", "alive", "id"):
            np.testing.assert_array_equal(
                first[0]["particles"][name], repeat[0]["particles"][name])

    def test_host_field_seed_uses_canonical_yee_shapes_and_periodic_duplicates(self):
        deck = _deck(field_seed=True)
        fields = runner._canonical_initial_fields(
            deck, Units(deck), first_dt=0.05, nghost=1)
        extents = runner._field_extents(
            deck.domain.nx, deck.domain.ny, deck.geometry)

        for name, (rows, columns, _, _) in extents.items():
            self.assertEqual(np.asarray(fields[name]).shape, (rows, columns))
        np.testing.assert_array_equal(fields["bz"][:, -1], fields["bz"][:, 0])
        np.testing.assert_array_equal(fields["bz"][-1, :], fields["bz"][0, :])
        self.assertGreater(float(np.max(np.abs(fields["ey"]))), 0.0)


class DiagnosticsTests(unittest.TestCase):

    def test_checkpoint_history_reassembles_gathered_and_sharded_yee_fields(self):
        deck = _deck(particles=5)
        state = _state(deck)
        series = {"step": [1], "time_s": [0.05], "alive_electrons": [5]}
        canonical_snapshot = {
            "step": 1, "time_s": 0.05,
            "canonical_fields": state["fields"],
        }
        gathered = runner._pic_checkpoint_fragment(
            0, "gathered", deck, [canonical_snapshot], {}, series)
        restored, restored_series = runner._decode_pic_checkpoint_history(
            [gathered], deck)
        self.assertEqual(restored_series, series)
        for name in runner._PIC_FIELD_COMPONENTS:
            rows, columns, _, _ = runner._field_extents(
                deck.domain.nx, deck.domain.ny, deck.geometry)[name]
            np.testing.assert_array_equal(
                restored[0]["canonical_fields"][name],
                np.asarray(state["fields"][name]).reshape(rows, columns))

        mapping = [
            {"index": 0, "rank": 0, "node_rank": 0,
             "rank_local_index": 0, "device_identity": "uuid:a"},
            {"index": 1, "rank": 1, "node_rank": 1,
             "rank_local_index": 0, "device_identity": "uuid:b"},
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
        fragments = []
        for rank in (0, 1):
            shard = _local_shards(
                deck, state, mapping, topology, rank=rank)[0]
            endpoint = int(shard["endpoint"])
            history = {endpoint: [{
                "step": 1, "time_s": 0.05,
                "fields": shard["fields"],
            }]}
            fragments.append(runner._pic_checkpoint_fragment(
                rank, "sharded", deck, [], history, series))
        repartitioned, _ = runner._decode_pic_checkpoint_history(
            fragments, deck)
        for name in runner._PIC_FIELD_COMPONENTS:
            rows, columns, _, _ = runner._field_extents(
                deck.domain.nx, deck.domain.ny, deck.geometry)[name]
            np.testing.assert_array_equal(
                repartitioned[0]["canonical_fields"][name],
                np.asarray(state["fields"][name]).reshape(rows, columns))

    def test_gathered_snapshot_preserves_legacy_keys_and_padded_shapes(self):
        deck = _deck()
        units = Units(deck)
        state = _state(deck)
        snapshot = runner._snapshot(state, deck, 2, 0.1, units, nghost=1)
        from quasar.pic.cli import _flatten_for_npz
        payload = _flatten_for_npz([], snapshot, {"step": [2]})

        padded_size = (deck.domain.nx + 2) * (deck.domain.ny + 2)
        self.assertEqual(payload["field_ey"].shape, (padded_size,))
        self.assertEqual(payload["external_bx"].shape, (padded_size,))
        self.assertIn("species_electrons_x", payload)
        self.assertNotIn("species_electrons_id", payload)
        self.assertNotIn("species_electrons_x_prev", payload)

    def test_shards_own_each_yee_degree_and_particle_exactly_once(self):
        deck = _deck(particles=11)
        units = Units(deck)
        state = _state(deck)
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
        local_shards = _local_shards(
            deck, state, mapping, topology, rank=session.rank)

        with tempfile.TemporaryDirectory() as directory:
            out_path = Path(directory) / "out.npz"
            manifest_path = runner._write_sharded(
                session, out_path, local_shards, {}, None, deck, units, 1,
                2, 0.1, mapping, topology)
            manifest = distributed.read_diagnostics_manifest(manifest_path)
            rows, columns, _, _ = runner._field_extents(
                deck.domain.nx, deck.domain.ny, deck.geometry)["bz"]
            rebuilt = np.empty((rows, columns), dtype=np.float64)
            coverage = np.zeros((rows, columns), dtype=np.int64)
            particle_ids: list[int] = []
            for shard in manifest.shards:
                with np.load(shard.path, allow_pickle=False) as payload:
                    begin_y, begin_x = map(int, payload["field_bz_offset"])
                    values = payload["field_bz"]
                    end_y = begin_y + values.shape[0]
                    end_x = begin_x + values.shape[1]
                    rebuilt[begin_y:end_y, begin_x:end_x] = values
                    coverage[begin_y:end_y, begin_x:end_x] += 1
                    particle_ids.extend(
                        map(int, payload["species_electrons_id"]))
            np.testing.assert_array_equal(coverage, 1)
            np.testing.assert_array_equal(rebuilt, state["fields"]["bz"])
            self.assertEqual(sorted(particle_ids), list(range(11)))

        self.assertEqual(
            [call[1] for call in session.calls],
            ["pic-diagnostics-sharded-prepare",
             "pic-diagnostics-sharded-begin",
             "pic-diagnostics-shard-write",
             "pic-diagnostics-manifest-publish"])

    def test_sharded_writer_accepts_only_the_calling_ranks_shards(self):
        deck = _deck(particles=8)
        units = Units(deck)
        state = _state(deck)
        mapping = [
            {"index": endpoint, "rank": endpoint // 2,
             "node_rank": endpoint // 2,
             "rank_local_index": endpoint % 2,
             "device_identity": f"uuid:{endpoint}"}
            for endpoint in range(4)
        ]
        topology = {
            "decomposition": (2, 2),
            "tiles": [
                {"endpoint": 0, "tile": (0, 0), "offset": (0, 0),
                 "owned_shape": (2, 3)},
                {"endpoint": 1, "tile": (1, 0), "offset": (0, 3),
                 "owned_shape": (2, 2)},
                {"endpoint": 2, "tile": (0, 1), "offset": (2, 0),
                 "owned_shape": (1, 3)},
                {"endpoint": 3, "tile": (1, 1), "offset": (2, 3),
                 "owned_shape": (1, 2)},
            ],
        }
        session = _ConsensusSession()
        local_shards = _local_shards(
            deck, state, mapping, topology, rank=session.rank)
        self.assertEqual(
            [int(item["endpoint"]) for item in local_shards], [0, 1])

        with tempfile.TemporaryDirectory() as directory:
            out_path = Path(directory) / "out.npz"
            runner._write_sharded(
                session, out_path, local_shards, {}, None, deck, units, 1,
                2, 0.1, mapping, topology)
            written = sorted(path.name for path in Path(directory).glob("*.npz"))
            self.assertEqual(written, [
                "out.rank000000.gpu000.npz",
                "out.rank000000.gpu001.npz",
            ])
            self.assertTrue(out_path.with_suffix(".manifest.json").exists())

    def test_sharded_run_never_calls_global_gather(self):
        deck = _deck(particles=5)
        session = _ShardedRunSession(deck)
        options = distributed.RunOptions(
            devices=(0,), decomposition=(1, 1),
            diagnostics_layout="sharded")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (mock.patch.object(
                      runner._distributed, "RuntimeSession",
                      return_value=session),
                  mock.patch.object(Path, "cwd", return_value=root)):
                result = runner.run(
                    deck, options, seed=17, write_every=2)

            self.assertEqual(result["final_step"], 2)
            self.assertEqual(
                result["telemetry"]["pic"]["global_state_gathers"], 0)
            self.assertGreater(
                result["telemetry"]["pic"]["local_shard_extractions"], 0)
            self.assertEqual(result["telemetry"]["transport_requested"], "auto")
            self.assertEqual(result["telemetry"]["transport"], "staged")
            self.assertIn(
                "ROCm-aware",
                result["telemetry"]["transport_fallback_reason"])
            self.assertEqual(result["telemetry"]["decomposition"], (1, 1))
            phases = result["telemetry"]["phase_seconds"]
            self.assertEqual(set(phases), {
                "prepare", "setup", "evolution", "checkpoint", "diagnostics",
            })
            self.assertTrue(all(value >= 0.0 for value in phases.values()))
            self.assertEqual(phases["checkpoint"], 0.0)
            self.assertAlmostEqual(
                result["telemetry"]["wall_seconds"],
                phases["prepare"] + phases["setup"]
                + phases["evolution"] + phases["diagnostics"])
            self.assertEqual(session.transport_requests, ["auto"])
            self.assertEqual(session.local_extractions, [False, True, True])
            self.assertTrue((root / "out.manifest.json").exists())
            self.assertTrue(
                (root / "out_0000000002.manifest.json").exists())

    def test_sharded_run_checkpoints_cadence_and_continuation_history(self):
        deck = _deck(particles=5)
        session = _ShardedRunSession(deck)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "state.h5"
            options = distributed.RunOptions(
                devices=(0,), decomposition=(1, 1),
                diagnostics_layout="sharded", checkpoint=checkpoint,
                checkpoint_every=1)
            with (mock.patch.object(
                      runner._distributed, "RuntimeSession",
                      return_value=session),
                  mock.patch.object(Path, "cwd", return_value=root),
                  mock.patch("builtins.print")):
                result = runner.run(
                    deck, options, seed=17, log_every=1)

        self.assertEqual(result["final_step"], 2)
        self.assertTrue(session.closed)
        self.assertEqual(
            [(item[1], item[2]) for item in session.checkpoints],
            [(1, 0.05), (2, 0.1)])
        self.assertTrue(all(
            item[0] == str(checkpoint.resolve())
            and item[3] == deck.units
            for item in session.checkpoints))
        snapshots, series = runner._decode_pic_checkpoint_history(
            [session.checkpoints[-1][4]], deck)
        self.assertEqual([item["step"] for item in snapshots], [1, 2])
        self.assertEqual(series["step"], [1, 2])
        self.assertEqual(series["time_s"], [0.05, 0.1])
        self.assertEqual(series["alive_electrons"], [5, 5])
        self.assertEqual(session.local_extractions, [False, False, True])


class RunPolicyTests(unittest.TestCase):

    def test_file_grid_signature_changes_when_same_path_content_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            field_path = root / "field.npz"
            deck_path = root / "pic.yaml"
            deck_path.write_text(
                """units: normalized
domain:
  nx: 4
  ny: 4
  lx_m: 1.0
  ly_m: 1.0
external_field:
  evaluator:
    type: file_grid
    path: field.npz
time:
  dt_s: auto
  steps: 1
""",
                encoding="utf-8")
            metadata = {
                "grid_origin": np.zeros(3),
                "grid_spacing": np.ones(3),
            }
            np.savez(
                field_path, B_xyz_grid=np.zeros((2, 2, 2, 3)),
                **metadata)
            before = runner._external_signature(pic_io.load(deck_path))

            np.savez(
                field_path, B_xyz_grid=np.ones((2, 2, 2, 3)),
                **metadata)
            after = runner._external_signature(pic_io.load(deck_path))

        self.assertNotEqual(before, after)

    def test_file_grid_signature_ignores_path_for_identical_resolved_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "grid_origin": np.zeros(3),
                "grid_spacing": np.ones(3),
            }
            values = np.arange(24, dtype=np.float64).reshape(2, 2, 2, 3)
            np.savez(root / "first.npz", B_xyz_grid=values, **metadata)
            np.savez(root / "second.npz", B_xyz_grid=values, **metadata)

            def load(name: str) -> pic_io.PicDeck:
                return pic_io.parse({
                    "units": "normalized",
                    "domain": {
                        "nx": 4, "ny": 4, "lx_m": 1.0, "ly_m": 1.0,
                    },
                    "external_field": {
                        "evaluator": {"type": "file_grid", "path": name},
                    },
                    "time": {"dt_s": "auto", "steps": 1},
                }, base_dir=root)

            first = runner._external_signature(load("first.npz"))
            second = runner._external_signature(load("second.npz"))

        self.assertEqual(first, second)

    def test_external_signature_ignores_inactive_evaluator_fields(self):
        deck = _deck(external_field=True)
        assert deck.external_field is not None
        before = runner._external_signature(deck)

        deck.external_field.dipole_origin = (9.0, 8.0, 7.0)
        deck.external_field.gradient_b0 = (6.0, 5.0, 4.0)
        deck.external_field.file_path = "unused.npz"
        deck.external_field.params = {"unused": [3.0, 2.0, 1.0]}
        self.assertEqual(before, runner._external_signature(deck))

        deck.external_field.uniform_b = (0.5, -0.5, 0.75)
        self.assertNotEqual(before, runner._external_signature(deck))

    def test_policy_mismatch_stops_before_device_or_topology_collectives(self):
        deck = _deck(external_field=True)
        session = _RejectingPolicySession()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "committed.h5"
            restart = root / "restart.h5"
            options = distributed.RunOptions(
                devices=(3,), decomposition=(1, 1), transport="direct",
                diagnostics_layout="sharded", checkpoint=checkpoint,
                checkpoint_every=3, restart=restart)
            with (mock.patch.object(
                      runner._distributed, "RuntimeSession",
                      return_value=session),
                  mock.patch.object(Path, "cwd", return_value=root),
                  mock.patch.object(
                      runner, "_make_external_evaluator",
                      return_value=(object(), object(), (1.0, 2.0, 3.0)))):
                with self.assertRaisesRegex(
                        RuntimeError, "different distributed PIC run policies"):
                    runner.run(
                        deck, options, verbose=True, print_config=True,
                        log_every=7, write_every=9)

        self.assertFalse(session.configured)
        self.assertTrue(session.closed)
        self.assertEqual(len(session.agreements), 1)
        signature, phase, message = session.agreements[0]
        self.assertEqual(phase, "pic-run-policy")
        self.assertIn("different distributed PIC run policies", message)
        policy = json.loads(signature)
        self.assertEqual(policy["physics"], "pic")
        self.assertEqual(policy["mode"], "restart")
        self.assertEqual(policy["restart_path"], str(restart.resolve()))
        self.assertEqual(policy["checkpoint"], {
            "path": str(checkpoint.resolve()), "cadence": 3})
        self.assertEqual(policy["diagnostics"], {
            "output_path": str((root / "out.npz").resolve()),
            "layout": "sharded", "cadence": 1,
            "fields": ["ey", "bz"], "per_species": True,
        })
        self.assertEqual(policy["termination"], {
            "steps": 2, "end_time": None})
        self.assertEqual(policy["placement"], {
            "devices": [3], "decomposition": [1, 1],
            "transport": "direct",
        })
        self.assertEqual(policy["log_cadence"], 7)
        self.assertEqual(policy["write_cadence"], 9)
        self.assertIsNone(policy["seed"])
        self.assertEqual(
            policy["external_field"], runner._external_signature(deck))

    def test_fresh_run_samples_declared_external_field(self):
        deck = _deck(external_field=True)
        session = _ShardedRunSession(deck)
        options = distributed.RunOptions(
            devices=(0,), decomposition=(1, 1),
            diagnostics_layout="sharded")
        evaluator = object()
        source = object()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (mock.patch.object(
                      runner._distributed, "RuntimeSession",
                      return_value=session),
                  mock.patch.object(Path, "cwd", return_value=root),
                  mock.patch.object(
                      runner, "_make_external_evaluator",
                      return_value=(evaluator, source, (1.0, 2.0, 3.0)))):
                runner.run(deck, options, seed=17)

        self.assertEqual(
            session.sampled_external,
            [(evaluator, source, 1.0, 2.0, 3.0)])
        policy = json.loads(session.agreements[0][0])
        self.assertEqual(policy["mode"], "start")
        self.assertEqual(policy["seed"], 17)

    def test_restart_validates_declaration_without_resampling_external_field(self):
        deck = _deck(external_field=True)
        session = _RestartRunSession(deck)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = distributed.RunOptions(
                devices=(0,), decomposition=(1, 1),
                diagnostics_layout="sharded", restart=root / "restart.h5")
            evaluator = mock.Mock()

            def prepare_external(
                    _deck: object, _units: object) -> tuple[object, ...]:
                session.restart_events.append("external-prepare")
                return evaluator, object(), (1.0, 2.0, 3.0)

            with (mock.patch.object(
                      runner._distributed, "RuntimeSession",
                      return_value=session),
                  mock.patch.object(Path, "cwd", return_value=root),
                  mock.patch.object(
                      runner, "_make_external_evaluator",
                      side_effect=prepare_external)
                  as prepare_external_mock):
                result = runner.run(deck, options)

        prepare_external_mock.assert_called_once()
        self.assertEqual(
            session.restart_events, ["external-prepare", "restart"])
        self.assertEqual(result["final_step"], deck.time.steps)
        self.assertEqual(session.sampled_external, [])
        self.assertEqual(
            session.restart_config_signature,
            runner._external_signature(deck))
        resolved_dt = runner._resolve_timestep(deck, Units(deck))[0]
        self.assertEqual(
            session.restart_timestep_signature,
            runner._timestep_signature(deck, resolved_dt))
        assert session.state is not None
        for name, restored in session.restored_external.items():
            np.testing.assert_array_equal(
                session.state["external_fields"][name], restored)


class FailureConsistencyTests(unittest.TestCase):

    def test_loop_snapshot_assembly_failure_is_drained_collectively(self):
        deck = _deck()
        session = _ShardedRunSession(deck)
        options = distributed.RunOptions(
            devices=(0,), decomposition=(1, 1),
            diagnostics_layout="sharded")
        original_extract = runner._local_owned_shards

        class BrokenEndpoint:
            def __int__(self) -> int:
                raise OSError("injected snapshot conversion failure")

        def inject_bad_endpoint(
                active_session: object, include_particles: bool,
                phase: str) -> list[dict]:
            shards = original_extract(
                active_session, include_particles, phase)
            if phase == "pic-shard-convert-step-1":
                shards[0]["endpoint"] = BrokenEndpoint()
            return shards

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (mock.patch.object(
                      runner._distributed, "RuntimeSession",
                      return_value=session),
                  mock.patch.object(Path, "cwd", return_value=root),
                  mock.patch.object(
                      runner, "_local_owned_shards",
                      side_effect=inject_bad_endpoint)):
                with self.assertRaisesRegex(
                        RuntimeError, "injected snapshot conversion failure"):
                    runner.run(deck, options, seed=17)

        self.assertEqual(session.step_calls, 1)
        failures = [
            call for call in session.calls
            if call[1] == "pic-sharded-snapshot-assemble-step-1"
        ]
        self.assertEqual(len(failures), 1)
        self.assertFalse(failures[0][0])
        self.assertTrue(session.closed)


class CollectivePhaseTests(unittest.TestCase):

    def test_local_write_failure_is_made_collective(self):
        session = _ConsensusSession()

        with self.assertRaisesRegex(RuntimeError, "disk failed"):
            runner._collective_local(
                session, "write", lambda: (_ for _ in ()).throw(
                    OSError("disk failed")))
        self.assertEqual(session.calls[0][0:2], (False, "write"))


if __name__ == "__main__":
    unittest.main()
