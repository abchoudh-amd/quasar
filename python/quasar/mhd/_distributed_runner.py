"""High-level orchestration for the native tile-decomposed MHD runtime."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import distributed as _distributed
from .._checkpoint_diagnostics import (
    decode_fragment as _decode_checkpoint_fragment,
    encode_fragment as _encode_checkpoint_fragment,
    scalar_int as _checkpoint_scalar_int,
    scalar_text as _checkpoint_scalar_text,
)
from .._distributed_helpers import (
    RunPhaseTimes as _RunPhaseTimes,
    atomic_json as _atomic_json,
    atomic_savez as _atomic_savez,
    canonical_policy_signature as _canonical_policy_signature,
    collective_local as _collective_local,
    finalize_run_telemetry as _finalize_run_telemetry,
    restart_target_error as _restart_target_error,
)
from .._paths import confine_output_path
from . import io as mhd_io
from . import _units as mhd_units


_RECONSTRUCTION_HALO = {"muscl_minmod": 2, "mp5": 3, "mp7": 4}
_CHECKPOINT_DIAGNOSTICS_SCHEMA = "quasar-mhd-checkpoint-diagnostics/v1"


def _timestep_signature(deck: mhd_io.MhdDeck) -> str:
    """Return the restart compatibility identity, excluding end targets."""

    if deck.time.dt_s == "auto":
        return "policy=auto"
    return f"policy=fixed;dt={float(deck.time.dt_s).hex()}"


def _deck_and_path(input_deck: Any) -> tuple[mhd_io.MhdDeck, Path | None]:
    if isinstance(input_deck, mhd_io.MhdDeck):
        return input_deck, None
    path = Path(input_deck).resolve()
    return mhd_io.load(path), path


def _canonical_state(deck: mhd_io.MhdDeck, nghost: int) -> dict[str, Any]:
    padded = mhd_io.build_initial_state(deck, nghost)
    nx, ny, g = deck.domain.nx, deck.domain.ny, nghost
    shape = (ny + 2 * g, nx + 2 * g)

    def array(name: str) -> np.ndarray:
        return np.asarray(padded[name], dtype=np.float64).reshape(shape)

    result: dict[str, Any] = {"global_nx": nx, "global_ny": ny}
    for name in ("rho", "mx", "my", "mz", "energy"):
        result[name] = np.ascontiguousarray(array(name)[g:g + ny, g:g + nx])
    result["bx_face"] = np.ascontiguousarray(
        array("bx")[g:g + ny, g:g + nx + 1])
    result["by_face"] = np.ascontiguousarray(
        array("by")[g:g + ny + 1, g:g + nx])
    result["bz_cell"] = np.ascontiguousarray(
        array("bz")[g:g + ny, g:g + nx])
    return result


def _canonical_explicit_background(
        deck: mhd_io.MhdDeck, nghost: int) -> dict[str, Any] | None:
    explicit = (deck.background.enabled
                and (deck.background.file is not None
                     or deck.background.a_file is not None))
    if not explicit:
        return None
    padded = mhd_io.build_background_field(deck, nghost)
    assert padded is not None
    nx, ny, g = deck.domain.nx, deck.domain.ny, nghost
    shape = (ny + 2 * g, nx + 2 * g)

    def array(name: str) -> np.ndarray:
        return np.asarray(padded[name], dtype=np.float64).reshape(shape)

    return {
        "global_nx": nx,
        "global_ny": ny,
        "b0x_face": np.ascontiguousarray(
            array("b0x")[g:g + ny, g:g + nx + 1]),
        "b0y_face": np.ascontiguousarray(
            array("b0y")[g:g + ny + 1, g:g + nx]),
        "b0z_cell": np.ascontiguousarray(
            array("b0z")[g:g + ny, g:g + nx]),
    }


def _cell_state(session: Any, deck: mhd_io.MhdDeck) -> dict[str, np.ndarray]:
    shape = (deck.domain.ny, deck.domain.nx)
    result: dict[str, np.ndarray] = {}
    for name in mhd_io.STATE_COMPONENTS:
        native_values = session.mhd_gather_cell_component(name)

        def convert_component() -> None:
            values = np.asarray(native_values, dtype=np.float64).reshape(shape)
            if name in ("bx", "by", "bz"):
                values = mhd_units.magnetic_to_output(values, deck.units)
            result[name] = values

        _collective_local(
            session, f"mhd-cell-component-{name}-convert",
            convert_component)
    return result


def _local_cell_shards(
        session: Any, deck: mhd_io.MhdDeck) -> list[dict[str, Any]]:
    """Extract only this rank's endpoint-owned diagnostic cells."""

    native_shards = session.mhd_local_owned_shards()

    def convert_shards() -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for raw in native_shards:
            shard = dict(raw)
            try:
                endpoint = int(shard["endpoint"])
                tile = tuple(map(int, shard["tile"]))
                offset = tuple(map(int, shard["offset"]))
                owned_shape = tuple(map(int, shard["owned_shape"]))
                native_state = dict(shard["state"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "rank-local MHD shard has invalid ownership metadata") from exc
            if (len(tile) != 2 or len(offset) != 2 or len(owned_shape) != 2
                    or any(value < 0
                           for value in (*tile, *offset, *owned_shape))):
                raise ValueError(
                    "rank-local MHD shard metadata must be non-negative "
                    "2D values")
            state: dict[str, np.ndarray] = {}
            for name in mhd_io.STATE_COMPONENTS:
                if name not in native_state:
                    raise ValueError(
                        "rank-local MHD shard is missing state component "
                        f"{name}")
                values = np.asarray(native_state[name], dtype=np.float64)
                if values.size != owned_shape[0] * owned_shape[1]:
                    raise ValueError(
                        f"rank-local MHD component {name} has the wrong size")
                values = values.reshape(owned_shape)
                if name in ("bx", "by", "bz"):
                    values = mhd_units.magnetic_to_output(
                        values, deck.units)
                state[name] = np.ascontiguousarray(values)
            result.append({
                "endpoint": endpoint,
                "tile": tile,
                "offset": offset,
                "owned_shape": owned_shape,
                "state": state,
            })
        return result

    return _collective_local(
        session, "mhd-local-shards-convert", convert_shards)


def _snapshot(session: Any, deck: mhd_io.MhdDeck, step: int,
              sim_time: float, divb: float | None) -> dict[str, Any]:
    fields = _cell_state(session, deck)

    def assemble_snapshot() -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "step": step,
            "time_s": sim_time,
            "fields": {
                name: fields[name] for name in deck.diagnostics.fields},
            # Kept out of public NPZ output; checkpoints retain every state
            # component so a restart may change the diagnostic field policy.
            "checkpoint_fields": fields,
        }
        if divb is not None:
            snapshot["divb"] = divb
        return snapshot

    return _collective_local(
        session, f"mhd-snapshot-assemble-step-{step}", assemble_snapshot)


def _mhd_checkpoint_fragment(
        rank: int, layout: str, deck: mhd_io.MhdDeck,
        initial_state: dict[str, np.ndarray] | None,
        initial_shards: dict[int, dict[str, Any]],
        snapshots: list[dict[str, Any]],
        sharded_snapshots: dict[int, list[dict[str, Any]]],
        divb_series: list[float], extra_scalars: dict[str, float],
        ) -> bytes:
    """Encode this rank's globally located history pieces without gathering."""

    pieces: list[tuple[tuple[int, int], tuple[int, int],
                       dict[str, np.ndarray], list[dict[str, Any]]]] = []
    if layout == "gathered":
        if rank != 0:
            return b""
        if initial_state is None:
            raise ValueError("gathered MHD checkpoint history has no initial state")
        pieces.append(((0, 0), (deck.domain.ny, deck.domain.nx),
                       initial_state, snapshots))
    else:
        for endpoint in sorted(initial_shards):
            shard = initial_shards[endpoint]
            pieces.append((
                tuple(map(int, shard["offset"])),
                tuple(map(int, shard["owned_shape"])),
                shard["state"], sharded_snapshots.get(endpoint, [])))

    snapshot_steps: list[int] = []
    snapshot_times: list[float] = []
    snapshot_divb: list[float] = []
    if pieces:
        reference = pieces[0][3]
        snapshot_steps = [int(item["step"]) for item in reference]
        snapshot_times = [float(item["time_s"]) for item in reference]
        snapshot_divb = [float(item.get("divb", float("nan")))
                         for item in reference]
        for _, _, _, history in pieces[1:]:
            if ([int(item["step"]) for item in history] != snapshot_steps
                    or [float(item["time_s"]) for item in history]
                    != snapshot_times):
                raise ValueError(
                    "rank-local MHD checkpoint histories disagree on snapshots")

    payload: dict[str, np.ndarray] = {
        "schema": np.array([_CHECKPOINT_DIAGNOSTICS_SCHEMA]),
        "physics": np.array(["mhd"]),
        "global_shape": np.asarray(
            [deck.domain.ny, deck.domain.nx], dtype=np.uint64),
        "piece_count": np.array([len(pieces)], dtype=np.uint64),
        "snapshot_steps": np.asarray(snapshot_steps, dtype=np.uint64),
        "snapshot_times": np.asarray(snapshot_times, dtype=np.float64),
        "snapshot_divb": np.asarray(snapshot_divb, dtype=np.float64),
        "common_present": np.array([rank == 0], dtype=np.uint8),
    }
    if rank == 0:
        payload["divb_series"] = np.asarray(divb_series, dtype=np.float64)
        names = sorted(extra_scalars)
        payload["extra_names"] = np.asarray(names, dtype=np.str_)
        payload["extra_values"] = np.asarray(
            [extra_scalars[name] for name in names], dtype=np.float64)
    for index, (offset, shape, initial, history) in enumerate(pieces):
        prefix = f"piece.{index}."
        payload[prefix + "offset"] = np.asarray(offset, dtype=np.uint64)
        payload[prefix + "shape"] = np.asarray(shape, dtype=np.uint64)
        for name in mhd_io.STATE_COMPONENTS:
            values = np.asarray(initial[name], dtype=np.float64)
            if values.shape != shape:
                raise ValueError(
                    f"MHD initial checkpoint piece {name} has the wrong shape")
            payload[prefix + "initial." + name] = values
            history_values = [
                np.asarray(item.get("checkpoint_fields", item["fields"])[name],
                           dtype=np.float64)
                for item in history
            ]
            if any(value.shape != shape for value in history_values):
                raise ValueError(
                    f"MHD snapshot checkpoint piece {name} has the wrong shape")
            payload[prefix + "snapshot." + name] = (
                np.stack(history_values) if history_values else
                np.empty((0, *shape), dtype=np.float64))
    return _encode_checkpoint_fragment(payload)


def _decode_mhd_checkpoint_history(
        parts: list[bytes], deck: mhd_io.MhdDeck,
        ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]],
                   list[float], dict[str, float]]:
    """Validate fragments and assemble topology-independent canonical history."""

    archives = [_decode_checkpoint_fragment(part) for part in parts if part]
    if not archives:
        raise ValueError("MHD checkpoint has no diagnostic continuation state")
    shape = (deck.domain.ny, deck.domain.nx)
    initial = {name: np.empty(shape, dtype=np.float64)
               for name in mhd_io.STATE_COMPONENTS}
    coverage = np.zeros(shape, dtype=bool)
    snapshot_steps: np.ndarray | None = None
    snapshot_times: np.ndarray | None = None
    snapshot_divb: np.ndarray | None = None
    snapshot_fields: dict[str, np.ndarray] | None = None
    common: tuple[list[float], dict[str, float]] | None = None

    for archive in archives:
        if (_checkpoint_scalar_text(archive, "schema")
                != _CHECKPOINT_DIAGNOSTICS_SCHEMA
                or _checkpoint_scalar_text(archive, "physics") != "mhd"):
            raise ValueError("MHD checkpoint diagnostic schema is incompatible")
        global_shape = np.asarray(archive.get("global_shape"), dtype=np.uint64)
        if global_shape.shape != (2,) or tuple(map(int, global_shape)) != shape:
            raise ValueError("MHD checkpoint diagnostic mesh is incompatible")
        steps = np.asarray(archive.get("snapshot_steps"), dtype=np.uint64)
        times = np.asarray(archive.get("snapshot_times"), dtype=np.float64)
        divb = np.asarray(archive.get("snapshot_divb"), dtype=np.float64)
        if (steps.ndim != 1 or times.shape != steps.shape
                or divb.shape != steps.shape or not np.all(np.isfinite(times))
                or np.any(times < 0.0)):
            raise ValueError("MHD checkpoint snapshot index is invalid")
        if snapshot_steps is None:
            snapshot_steps, snapshot_times, snapshot_divb = steps, times, divb
            snapshot_fields = {
                name: np.empty((steps.size, *shape), dtype=np.float64)
                for name in mhd_io.STATE_COMPONENTS}
        elif (not np.array_equal(snapshot_steps, steps)
              or not np.array_equal(snapshot_times, times)
              or not np.array_equal(snapshot_divb, divb, equal_nan=True)):
            raise ValueError("MHD checkpoint fragments disagree on snapshots")

        if _checkpoint_scalar_int(archive, "common_present", maximum=1):
            if common is not None:
                raise ValueError("MHD checkpoint has duplicate common history")
            series = np.asarray(archive.get("divb_series"), dtype=np.float64)
            names = np.asarray(archive.get("extra_names"))
            values = np.asarray(archive.get("extra_values"), dtype=np.float64)
            if (series.ndim != 1 or not np.all(np.isfinite(series))
                    or names.ndim != 1 or names.dtype.kind not in ("U", "S")
                    or values.shape != names.shape
                    or not np.all(np.isfinite(values))):
                raise ValueError("MHD checkpoint common history is invalid")
            extras = {str(name): float(value)
                      for name, value in zip(names, values, strict=True)}
            if set(extras) - {"mass_initial", "energy_initial"}:
                raise ValueError("MHD checkpoint has an unknown scalar history")
            common = (series.tolist(), extras)

        piece_count = _checkpoint_scalar_int(
            archive, "piece_count", maximum=4096)
        for index in range(piece_count):
            prefix = f"piece.{index}."
            offset = np.asarray(archive.get(prefix + "offset"), dtype=np.uint64)
            owned = np.asarray(archive.get(prefix + "shape"), dtype=np.uint64)
            if offset.shape != (2,) or owned.shape != (2,):
                raise ValueError("MHD checkpoint piece geometry is invalid")
            oy, ox = map(int, offset)
            ny, nx = map(int, owned)
            if (ny <= 0 or nx <= 0 or oy + ny > shape[0]
                    or ox + nx > shape[1]
                    or np.any(coverage[oy:oy + ny, ox:ox + nx])):
                raise ValueError("MHD checkpoint pieces overlap or leave the mesh")
            for name in mhd_io.STATE_COMPONENTS:
                values = np.asarray(
                    archive.get(prefix + "initial." + name), dtype=np.float64)
                history = np.asarray(
                    archive.get(prefix + "snapshot." + name), dtype=np.float64)
                if (values.shape != (ny, nx)
                        or history.shape != (steps.size, ny, nx)
                        or not np.all(np.isfinite(values))
                        or not np.all(np.isfinite(history))):
                    raise ValueError("MHD checkpoint state history is invalid")
                initial[name][oy:oy + ny, ox:ox + nx] = values
                assert snapshot_fields is not None
                snapshot_fields[name][:, oy:oy + ny, ox:ox + nx] = history
            coverage[oy:oy + ny, ox:ox + nx] = True
    if not np.all(coverage) or common is None or snapshot_steps is None:
        raise ValueError("MHD checkpoint diagnostic pieces do not cover the mesh")
    assert snapshot_times is not None and snapshot_divb is not None
    assert snapshot_fields is not None
    snapshots: list[dict[str, Any]] = []
    for index, step in enumerate(snapshot_steps):
        item: dict[str, Any] = {
            "step": int(step),
            "time_s": float(snapshot_times[index]),
            "checkpoint_fields": {
                name: np.ascontiguousarray(snapshot_fields[name][index])
                for name in mhd_io.STATE_COMPONENTS},
        }
        if math.isfinite(float(snapshot_divb[index])):
            item["divb"] = float(snapshot_divb[index])
        snapshots.append(item)
    return initial, snapshots, common[0], common[1]


def _snapshot_divb_values(
        snapshots: list[dict[str, Any]]) -> np.ndarray:
    """Preserve snapshots whose earlier diagnostics omitted divergence."""

    return np.asarray(
        [item.get("divb", float("nan")) for item in snapshots],
        dtype=np.float64)


def _gathered_flat(
        deck: mhd_io.MhdDeck, final_state: dict[str, np.ndarray],
        initial_state: dict[str, np.ndarray], final_step: int,
        final_time: float, divb_series: list[float],
        snapshots: list[dict[str, Any]], extra_scalars: dict[str, float],
        nghost: int) -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {
        "final_step": np.array([final_step]),
        "final_time_s": np.array([final_time]),
        "nx": np.array([deck.domain.nx]),
        "ny": np.array([deck.domain.ny]),
        "lx_m": np.array([deck.domain.lx_m]),
        "ly_m": np.array([deck.domain.ly_m]),
        "nghost": np.array([nghost]),
        "units": np.array([deck.units]),
        "geometry": np.array([deck.geometry]),
        "gamma": np.array([deck.numerics.gamma]),
    }
    for name, values in final_state.items():
        flat[f"state_{name}"] = values
        flat[f"state_{name}_initial"] = initial_state[name]
    if deck.diagnostics.divb:
        converted = mhd_units.magnetic_to_output(
            np.asarray(divb_series, dtype=np.float64), deck.units)
        flat["divb_linf"] = converted
        flat["divb_linf_final"] = np.array([converted[-1]])
    if snapshots:
        flat["snapshot_steps"] = np.asarray(
            [item["step"] for item in snapshots])
        flat["snapshot_times_s"] = np.asarray(
            [item["time_s"] for item in snapshots])
        for name in deck.diagnostics.fields:
            flat[f"snapshot_state_{name}"] = np.stack(
                [item["fields"][name] for item in snapshots])
        if deck.diagnostics.divb:
            flat["snapshot_divb_linf"] = mhd_units.magnetic_to_output(
                _snapshot_divb_values(snapshots), deck.units)
    for key, value in extra_scalars.items():
        flat[key] = np.array([value])
    return flat


def _write_sharded(
        session: Any, out_path: Path, deck: mhd_io.MhdDeck,
        local_shards: list[dict[str, Any]],
        initial_shards: dict[int, dict[str, Any]],
        final_step: int, final_time: float, divb_series: list[float],
        snapshots: dict[int, list[dict[str, Any]]],
        extra_scalars: dict[str, float], nghost: int,
        mapping: list[dict[str, Any]], topology: dict[str, Any]) -> Path:
    manifest_path = out_path.with_suffix(".manifest.json")

    def prepare_records() -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        tiles = {
            int(item["endpoint"]): item for item in topology["tiles"]}
        for endpoint_info in mapping:
            endpoint = int(endpoint_info["index"])
            tile = tiles[endpoint]
            offset_y, offset_x = map(int, tile["offset"])
            owned_ny, owned_nx = map(int, tile["owned_shape"])
            shard_path = out_path.with_name(
                f"{out_path.stem}.rank{int(endpoint_info['rank']):06d}."
                f"gpu{int(endpoint_info['rank_local_index']):03d}.npz")
            record = {
                "rank": int(endpoint_info["rank"]),
                "node_rank": int(endpoint_info["node_rank"]),
                "local_device": int(endpoint_info["rank_local_index"]),
                "endpoint": endpoint,
                "device_identity": str(endpoint_info["device_identity"]),
                "tile": list(map(int, tile["tile"])),
                "offset": [offset_y, offset_x],
                "owned_shape": [owned_ny, owned_nx],
                "path": shard_path.name,
            }
            records.append(record)
        return records

    records = _collective_local(
        session, "mhd-diagnostics-sharded-prepare", prepare_records)

    # A stale manifest must never describe shards while they are being
    # replaced.  Its absence marks an incomplete update until rank zero
    # publishes the new manifest last.
    _collective_local(
        session, "mhd-diagnostics-sharded-begin",
        lambda: manifest_path.unlink(missing_ok=True)
        if int(session.rank) == 0 else None)

    def write_local_shards() -> None:
        local_records = {
            int(record["endpoint"]): record for record in records
            if int(record["rank"]) == int(session.rank)
        }
        shards_by_endpoint: dict[int, dict[str, Any]] = {}
        for shard in local_shards:
            endpoint = int(shard["endpoint"])
            if endpoint in shards_by_endpoint:
                raise ValueError(
                    f"duplicate rank-local MHD shard for endpoint {endpoint}")
            shards_by_endpoint[endpoint] = shard
        if set(shards_by_endpoint) != set(local_records):
            raise ValueError(
                "rank-local MHD shard endpoints do not match the endpoint "
                f"mapping: expected {sorted(local_records)}, received "
                f"{sorted(shards_by_endpoint)}")
        if set(initial_shards) != set(local_records):
            raise ValueError(
                "rank-local initial MHD shards do not match the endpoint "
                f"mapping: expected {sorted(local_records)}, received "
                f"{sorted(initial_shards)}")
        if not set(snapshots).issubset(local_records):
            raise ValueError(
                "rank-local MHD snapshots contain a remote endpoint")

        for endpoint, record in local_records.items():
            shard = shards_by_endpoint[endpoint]
            initial = initial_shards[endpoint]
            for key in ("tile", "offset", "owned_shape"):
                expected = tuple(map(int, record[key]))
                if tuple(map(int, shard[key])) != expected:
                    raise ValueError(
                        f"rank-local MHD shard {endpoint} has inconsistent "
                        f"{key} metadata")
                if tuple(map(int, initial[key])) != expected:
                    raise ValueError(
                        f"rank-local initial MHD shard {endpoint} has "
                        f"inconsistent {key} metadata")
            offset_y, offset_x = record["offset"]
            owned_ny, owned_nx = record["owned_shape"]
            payload: dict[str, np.ndarray] = {
                # Keep the familiar gathered metadata names where meaningful,
                # and add ownership metadata needed to reconstruct the mesh.
                "final_step": np.array([final_step]),
                "final_time_s": np.array([final_time]),
                "nx": np.array([deck.domain.nx]),
                "ny": np.array([deck.domain.ny]),
                "lx_m": np.array([deck.domain.lx_m]),
                "ly_m": np.array([deck.domain.ly_m]),
                "nghost": np.array([nghost]),
                "units": np.array([deck.units]),
                "geometry": np.array([deck.geometry]),
                "gamma": np.array([deck.numerics.gamma]),
                "offset": np.array([offset_y, offset_x]),
                "owned_shape": np.array([owned_ny, owned_nx]),
            }
            for name in mhd_io.STATE_COMPONENTS:
                values = np.asarray(shard["state"][name], dtype=np.float64)
                initial_values = np.asarray(
                    initial["state"][name], dtype=np.float64)
                expected_shape = (owned_ny, owned_nx)
                if (values.shape != expected_shape
                        or initial_values.shape != expected_shape):
                    raise ValueError(
                        f"rank-local MHD component {name} does not match "
                        "the owned shape")
                payload[f"state_{name}"] = values
                payload[f"state_{name}_initial"] = initial_values
            if deck.diagnostics.divb:
                converted = mhd_units.magnetic_to_output(
                    np.asarray(divb_series, dtype=np.float64), deck.units)
                payload["divb_linf"] = converted
                payload["divb_linf_final"] = np.array([converted[-1]])
            endpoint_snapshots = snapshots.get(endpoint, [])
            if endpoint_snapshots:
                payload["snapshot_steps"] = np.asarray(
                    [item["step"] for item in endpoint_snapshots])
                payload["snapshot_times_s"] = np.asarray(
                    [item["time_s"] for item in endpoint_snapshots])
                for name in deck.diagnostics.fields:
                    values = [
                        np.asarray(item["fields"][name], dtype=np.float64)
                        for item in endpoint_snapshots]
                    if any(value.shape != (owned_ny, owned_nx)
                           for value in values):
                        raise ValueError(
                            f"rank-local MHD snapshot {name} does not match "
                            "the owned shape")
                    payload[f"snapshot_state_{name}"] = np.stack(values)
                if deck.diagnostics.divb:
                    payload["snapshot_divb_linf"] = (
                        mhd_units.magnetic_to_output(
                            _snapshot_divb_values(endpoint_snapshots),
                            deck.units))
            for key, value in extra_scalars.items():
                payload[key] = np.array([value])
            _atomic_savez(out_path.parent / record["path"], payload)

    _collective_local(
        session, "mhd-diagnostics-shard-write", write_local_shards)

    def publish_manifest() -> None:
        if int(session.rank) != 0:
            return
        document = {
            "schema": "quasar-diagnostics-shards/v1",
            "physics": "mhd",
            "geometry": deck.geometry,
            "global_shape": [deck.domain.ny, deck.domain.nx],
            "step": final_step,
            "time": final_time,
            "decomposition": {
                "px": int(topology["decomposition"][0]),
                "py": int(topology["decomposition"][1]),
            },
            "shards": sorted(records, key=lambda item: item["endpoint"]),
        }
        _atomic_json(manifest_path, document)

    _collective_local(
        session, "mhd-diagnostics-manifest-publish",
        publish_manifest)
    return manifest_path


@dataclass(frozen=True)
class _PreparedMhdRun:
    deck: mhd_io.MhdDeck
    deck_path: Path | None
    nghost: int
    state: dict[str, Any]
    background: dict[str, Any] | None
    config: Any
    out_path: Path
    checkpoint_path: Path | None
    restart_path: Path | None
    log_interval: int
    policy_signature: str


@dataclass(frozen=True)
class _MhdLaunch:
    mapping: list[dict[str, Any]]
    topology: dict[str, Any]
    restart_metadata: dict[str, Any] | None


@dataclass
class _MhdClock:
    step: int
    sim_time: float
    dt: float
    dt_is_auto: bool


@dataclass
class _MhdHistory:
    initial_state: dict[str, np.ndarray] | None
    initial_shards: dict[int, dict[str, Any]]
    snapshots: list[dict[str, Any]]
    sharded_snapshots: dict[int, list[dict[str, Any]]]
    divb_series: list[float]
    extra_scalars: dict[str, float]


def _mhd_policy_signature(
        deck: mhd_io.MhdDeck, options: _distributed.RunOptions, *,
        out_path: Path, checkpoint_path: Path | None,
        restart_path: Path | None, log_interval: int,
        verbose: bool, print_config: bool) -> str:
    return _canonical_policy_signature({
        "schema": "quasar-distributed-run-policy/v1",
        "physics": "mhd",
        "mode": "start" if restart_path is None else "restart",
        "restart_path": None if restart_path is None else str(restart_path),
        "checkpoint": {
            "path": (None if checkpoint_path is None else
                     str(checkpoint_path)),
            "cadence": options.checkpoint_every,
        },
        "diagnostics": {
            "output_path": str(out_path),
            "layout": options.diagnostics_layout,
            "cadence": int(deck.diagnostics.cadence),
            "fields": list(deck.diagnostics.fields),
            "divb": bool(deck.diagnostics.divb),
        },
        "termination": {
            "steps": int(deck.time.steps),
            "end_time": (None if deck.time.t_end is None else
                         float(deck.time.t_end)),
        },
        "timestep": (deck.time.dt_s if deck.time.dt_s == "auto" else
                     float(deck.time.dt_s)),
        "placement": {
            "devices": options.devices,
            "decomposition": options.decomposition,
            "transport": options.transport,
        },
        "log_cadence": log_interval,
        "verbose": bool(verbose),
        "print_config": bool(print_config),
    })


def _prepare_mhd_run(
        input_deck: Any, options: _distributed.RunOptions, *,
        steps_override: int | None, verbose: bool, print_config: bool,
        log_every: int) -> _PreparedMhdRun:
    deck, deck_path = _deck_and_path(input_deck)
    if steps_override is not None:
        deck = copy.deepcopy(deck)
        deck.time = mhd_io.Time(
            dt_s=deck.time.dt_s, steps=steps_override,
            t_end=deck.time.t_end)
        deck.validate()
    nghost = _RECONSTRUCTION_HALO[deck.numerics.reconstruction]
    state = _canonical_state(deck, nghost)
    background = _canonical_explicit_background(deck, nghost)

    # Import lazily to avoid the cli -> distributed -> runner cycle at
    # module load.
    from .cli import _make_config
    config = _make_config(deck)
    config.timestep_signature = _timestep_signature(deck)
    deck_directory = deck_path.parent if deck_path is not None else Path.cwd()
    out_path = confine_output_path(
        deck_directory, deck.diagnostics.output_path,
        label="diagnostics.output_path")
    checkpoint_path = (None if options.checkpoint is None else
                       Path(options.checkpoint).expanduser().resolve())
    restart_path = (None if options.restart is None else
                    Path(options.restart).expanduser().resolve())
    log_interval = max(0, int(log_every))
    policy_signature = _mhd_policy_signature(
        deck, options, out_path=out_path,
        checkpoint_path=checkpoint_path, restart_path=restart_path,
        log_interval=log_interval, verbose=verbose,
        print_config=print_config)
    return _PreparedMhdRun(
        deck, deck_path, nghost, state, background, config, out_path,
        checkpoint_path, restart_path, log_interval, policy_signature)


def _launch_mhd_runtime(
        session: Any, prepared: _PreparedMhdRun,
        options: _distributed.RunOptions) -> _MhdLaunch:
    session.collective_agree(
        prepared.policy_signature, "mhd-run-policy",
        "MPI ranks supplied different distributed MHD run policies")
    mapping = session.configure_devices(options.devices)
    topology = session.select_topology(
        prepared.deck.domain.nx, prepared.deck.domain.ny,
        options.decomposition, prepared.nghost)
    restart_metadata: dict[str, Any] | None = None
    if prepared.restart_path is None:
        session.start_mhd(
            prepared.config, prepared.state, prepared.background,
            transport=options.transport)
    else:
        native_restart_metadata = session.restart_mhd(
            prepared.config, str(prepared.restart_path), prepared.deck.units,
            prepared.background, transport=options.transport)
        restart_metadata = _collective_local(
            session, "mhd-restart-metadata-convert",
            lambda: dict(native_restart_metadata))
    return _MhdLaunch(mapping, topology, restart_metadata)


def _finish_mhd_startup(
        session: Any, prepared: _PreparedMhdRun,
        launch: _MhdLaunch) -> tuple[
            tuple[dict[str, np.ndarray], list[dict[str, Any]],
                  list[float], dict[str, float]] | None,
            _MhdClock]:
    restored_history = None
    if launch.restart_metadata is not None:
        restored_history = _collective_local(
            session, "mhd-restart-diagnostics-decode",
            lambda: _decode_mhd_checkpoint_history(
                list(launch.restart_metadata.get("diagnostic_state", [])),
                prepared.deck))

    step_done, sim_time = _collective_local(
        session, "mhd-start-state-convert",
        lambda: (
            0 if launch.restart_metadata is None else
            int(launch.restart_metadata["step"]),
            0.0 if launch.restart_metadata is None else
            float(launch.restart_metadata["time"])))
    time_limit = (None if prepared.deck.time.t_end is None else
                  float(prepared.deck.time.t_end))
    target_error = _restart_target_error(
        step_done, sim_time, prepared.deck.time.steps, time_limit)
    session.collective_require(
        not target_error, "mhd-restart-target", target_error)

    native_cfl = session.mhd_cfl_limit()
    cfl = _collective_local(
        session, "mhd-initial-cfl-convert", lambda: float(native_cfl))
    if prepared.deck.time.dt_s == "auto":
        dt = cfl
        dt_is_auto = True
    else:
        dt = float(prepared.deck.time.dt_s)
        dt_is_auto = False
        session.collective_require(
            dt <= cfl, "mhd-timestep-initial",
            "" if dt <= cfl else
            f"time.dt_s ({dt:.6e}) exceeds the distributed MHD CFL "
            f"limit ({cfl:.6e})")
    return restored_history, _MhdClock(
        step_done, sim_time, dt, dt_is_auto)


def _print_mhd_config(
        session: Any, prepared: _PreparedMhdRun, launch: _MhdLaunch,
        options: _distributed.RunOptions, print_config: bool,
        dt: float) -> None:
    def print_resolved_config() -> None:
        if not print_config or int(session.rank) != 0:
            return
        deck = prepared.deck
        deck_source = (prepared.deck_path if prepared.deck_path is not None
                       else "<in-memory>")
        print(f"deck    : {deck_source}")
        print(f"grid    : {deck.domain.nx}x{deck.domain.ny}  "
              f"({deck.domain.lx_m}x{deck.domain.ly_m})  "
              f"origin=({deck.domain.origin_x_m}, "
              f"{deck.domain.origin_y_m})  nghost={prepared.nghost}")
        print(f"geometry: {deck.geometry}")
        print(f"gamma   : {deck.numerics.gamma}")
        print(f"schemes : recon={deck.numerics.reconstruction} "
              f"riemann={deck.numerics.riemann} "
              f"integ={deck.numerics.integrator} "
              f"ct={deck.numerics.ct} "
              f"pos={deck.numerics.positivity}")
        print(f"initial : {deck.initial.type}")
        print(f"dt      : {dt:.6e}    steps: {deck.time.steps}")
        native_mhd = dict(session.telemetry.get("mhd", {}))
        native_transport = dict(native_mhd.get("transport", {}))
        resolved_transport = native_transport.get(
            "interprocess", options.transport)
        print(f"topology: {launch.topology['decomposition']}  "
              f"endpoints={len(launch.mapping)} "
              f"transport={resolved_transport}")

    _collective_local(session, "mhd-print-config", print_resolved_config)


def _restore_mhd_history(
        session: Any, prepared: _PreparedMhdRun,
        options: _distributed.RunOptions, clock: _MhdClock,
        restored_history: tuple[
            dict[str, np.ndarray], list[dict[str, Any]],
            list[float], dict[str, float]],
        history: _MhdHistory) -> None:
    (canonical_initial, canonical_snapshots,
     history.divb_series, history.extra_scalars) = restored_history
    if any(int(item["step"]) > clock.step
           or float(item["time_s"]) > clock.sim_time
           for item in canonical_snapshots):
        raise ValueError(
            "MHD checkpoint diagnostic history extends past the checkpoint")
    if options.diagnostics_layout == "gathered":
        history.initial_state = canonical_initial
        for item in canonical_snapshots:
            fields = item["checkpoint_fields"]
            restored = dict(item)
            restored["fields"] = {
                name: fields[name]
                for name in prepared.deck.diagnostics.fields}
            history.snapshots.append(restored)
        return

    local_templates = _local_cell_shards(session, prepared.deck)
    for template in local_templates:
        endpoint = int(template["endpoint"])
        oy, ox = map(int, template["offset"])
        ny, nx = map(int, template["owned_shape"])
        initial_piece = dict(template)
        initial_piece["state"] = {
            name: np.ascontiguousarray(
                canonical_initial[name][oy:oy + ny, ox:ox + nx])
            for name in mhd_io.STATE_COMPONENTS}
        history.initial_shards[endpoint] = initial_piece
        endpoint_history: list[dict[str, Any]] = []
        for item in canonical_snapshots:
            checkpoint_fields = {
                name: np.ascontiguousarray(
                    item["checkpoint_fields"][name][
                        oy:oy + ny, ox:ox + nx])
                for name in mhd_io.STATE_COMPONENTS}
            restored = {
                "step": item["step"],
                "time_s": item["time_s"],
                "fields": {
                    name: checkpoint_fields[name]
                    for name in prepared.deck.diagnostics.fields},
                "checkpoint_fields": checkpoint_fields,
            }
            if "divb" in item:
                restored["divb"] = item["divb"]
            endpoint_history.append(restored)
        history.sharded_snapshots[endpoint] = endpoint_history


def _initialize_mhd_history(
        session: Any, prepared: _PreparedMhdRun,
        options: _distributed.RunOptions, clock: _MhdClock,
        restored_history: tuple[
            dict[str, np.ndarray], list[dict[str, Any]],
            list[float], dict[str, float]] | None) -> _MhdHistory:
    history = _MhdHistory(None, {}, [], {}, [], {})
    if restored_history is None:
        if options.diagnostics_layout == "gathered":
            history.initial_state = _cell_state(session, prepared.deck)
        else:
            local_initial_shards = _local_cell_shards(
                session, prepared.deck)
            history.initial_shards = _collective_local(
                session, "mhd-initial-shards-index",
                lambda: {
                    int(item["endpoint"]): item
                    for item in local_initial_shards
                })
    else:
        _restore_mhd_history(
            session, prepared, options, clock, restored_history, history)

    if (restored_history is None
            and prepared.deck.initial.type == "orszag_tang"):
        native_sums = session.mhd_global_cell_sums()

        def convert_initial_sums() -> None:
            sums = dict(native_sums)
            history.extra_scalars["mass_initial"] = float(sums["rho"])
            history.extra_scalars["energy_initial"] = float(sums["energy"])

        _collective_local(
            session, "mhd-initial-sums-convert", convert_initial_sums)
    if restored_history is None and prepared.deck.diagnostics.divb:
        native_initial_divb = session.mhd_divergence_b_max()
        _collective_local(
            session, "mhd-initial-divb-convert",
            lambda: history.divb_series.append(float(native_initial_divb)))
    return history


def _advance_mhd_clock(
        session: Any, prepared: _PreparedMhdRun,
        clock: _MhdClock) -> None:
    if clock.dt_is_auto:
        native_dt_limit = session.mhd_cfl_limit()
        dt_limit = _collective_local(
            session, f"mhd-cfl-convert-step-{clock.step}",
            lambda: float(native_dt_limit))
    else:
        dt_limit = clock.dt
    dt_step = dt_limit
    clipped = False
    if prepared.deck.time.t_end is not None:
        remaining = float(prepared.deck.time.t_end) - clock.sim_time
        dt_step = min(dt_step, remaining)
        clipped = dt_step == remaining
    if not math.isfinite(dt_step) or dt_step <= 0.0:
        raise RuntimeError("distributed MHD timestep cannot make progress")
    next_time = (float(prepared.deck.time.t_end) if clipped else
                 clock.sim_time + dt_step)
    if not clipped and (not math.isfinite(next_time)
                        or next_time <= clock.sim_time):
        raise RuntimeError("distributed MHD time cannot make progress")
    session.mhd_step(dt_step, check_cfl=not clock.dt_is_auto)
    clock.sim_time = next_time
    clock.step += 1


def _measure_mhd_divb(
        session: Any, prepared: _PreparedMhdRun,
        history: _MhdHistory, clock: _MhdClock,
        on_cadence: bool, is_last: bool) -> float | None:
    need_divb = prepared.deck.diagnostics.divb and (
        on_cadence or is_last
        or (prepared.log_interval > 0
            and clock.step % prepared.log_interval == 0))
    if not need_divb:
        return None
    native_divb = session.mhd_divergence_b_max()
    divb = _collective_local(
        session, f"mhd-divb-convert-step-{clock.step}",
        lambda: float(native_divb))
    _collective_local(
        session, f"mhd-divb-record-step-{clock.step}",
        lambda: history.divb_series.append(divb))
    return divb


def _record_mhd_snapshot(
        session: Any, prepared: _PreparedMhdRun,
        options: _distributed.RunOptions, history: _MhdHistory,
        clock: _MhdClock, divb: float | None) -> None:
    if options.diagnostics_layout == "gathered":
        snapshot = _snapshot(
            session, prepared.deck, clock.step, clock.sim_time, divb)
        _collective_local(
            session, f"mhd-snapshot-record-step-{clock.step}",
            lambda: history.snapshots.append(snapshot))
        return

    local_snapshot_shards = _local_cell_shards(session, prepared.deck)

    def append_sharded_snapshots() -> None:
        for shard in local_snapshot_shards:
            endpoint = int(shard["endpoint"])
            snapshot: dict[str, Any] = {
                "step": clock.step,
                "time_s": clock.sim_time,
                "fields": {
                    name: shard["state"][name]
                    for name in prepared.deck.diagnostics.fields
                },
                "checkpoint_fields": shard["state"],
            }
            if divb is not None:
                snapshot["divb"] = divb
            history.sharded_snapshots.setdefault(endpoint, []).append(
                snapshot)

    _collective_local(
        session, f"mhd-sharded-snapshot-assemble-step-{clock.step}",
        append_sharded_snapshots)


def _log_mhd_progress(
        session: Any, prepared: _PreparedMhdRun,
        clock: _MhdClock, divb: float | None,
        start_wall: float) -> None:
    def print_progress() -> None:
        if int(session.rank) != 0:
            return
        elapsed = time.perf_counter() - start_wall
        rate = clock.step / elapsed if elapsed > 0.0 else 0.0
        remaining = ((prepared.deck.time.steps - clock.step) / rate
                     if rate > 0.0 else float("nan"))
        message = (
            f"step {clock.step}/{prepared.deck.time.steps}  "
            f"t={clock.sim_time:.6e}  rate={rate:.0f} step/s  "
            f"eta={remaining:.0f}s")
        if prepared.deck.diagnostics.divb:
            message += f"  |divB|inf={divb:.3e}"
        print(message, flush=True)

    _collective_local(
        session, f"mhd-progress-step-{clock.step}", print_progress)


def _write_mhd_checkpoint(
        session: Any, prepared: _PreparedMhdRun,
        options: _distributed.RunOptions, history: _MhdHistory,
        clock: _MhdClock) -> float:
    assert prepared.checkpoint_path is not None
    checkpoint_started = time.perf_counter()
    diagnostic_state = _collective_local(
        session, f"mhd-checkpoint-history-step-{clock.step}",
        lambda: _mhd_checkpoint_fragment(
            int(session.rank), options.diagnostics_layout, prepared.deck,
            history.initial_state, history.initial_shards, history.snapshots,
            history.sharded_snapshots, history.divb_series,
            history.extra_scalars))
    session.mhd_write_checkpoint(
        str(prepared.checkpoint_path), clock.step, clock.sim_time,
        prepared.deck.units, diagnostic_state)
    return time.perf_counter() - checkpoint_started


def _evolve_mhd(
        session: Any, prepared: _PreparedMhdRun, options: _distributed.RunOptions,
        history: _MhdHistory, clock: _MhdClock,
        start_wall: float) -> tuple[float, float]:
    last_checkpoint_step: int | None = None
    checkpoint_seconds = 0.0
    while clock.step < prepared.deck.time.steps and (
            prepared.deck.time.t_end is None
            or clock.sim_time < float(prepared.deck.time.t_end)):
        _advance_mhd_clock(session, prepared, clock)
        is_last = clock.step == prepared.deck.time.steps or (
            prepared.deck.time.t_end is not None
            and clock.sim_time >= float(prepared.deck.time.t_end))
        on_cadence = (prepared.deck.diagnostics.cadence > 0
                      and clock.step % prepared.deck.diagnostics.cadence == 0)
        divb = _measure_mhd_divb(
            session, prepared, history, clock, on_cadence, is_last)
        if on_cadence:
            _record_mhd_snapshot(
                session, prepared, options, history, clock, divb)
        if (prepared.log_interval > 0
                and clock.step % prepared.log_interval == 0):
            _log_mhd_progress(
                session, prepared, clock, divb, start_wall)
        if (prepared.checkpoint_path is not None
                and options.checkpoint_every is not None
                and clock.step % options.checkpoint_every == 0):
            checkpoint_seconds += _write_mhd_checkpoint(
                session, prepared, options, history, clock)
            last_checkpoint_step = clock.step

    if prepared.deck.diagnostics.divb and not history.divb_series:
        native_final_divb = session.mhd_divergence_b_max()
        _collective_local(
            session, "mhd-final-divb-convert",
            lambda: history.divb_series.append(float(native_final_divb)))
        if prepared.checkpoint_path is not None:
            last_checkpoint_step = None
    if (prepared.checkpoint_path is not None
            and last_checkpoint_step != clock.step):
        checkpoint_seconds += _write_mhd_checkpoint(
            session, prepared, options, history, clock)
    return checkpoint_seconds, time.perf_counter()


def _write_final_mhd_diagnostics(
        session: Any, prepared: _PreparedMhdRun, launch: _MhdLaunch,
        options: _distributed.RunOptions, history: _MhdHistory,
        clock: _MhdClock) -> Path:
    if options.diagnostics_layout == "gathered":
        final_state = _cell_state(session, prepared.deck)
        assert history.initial_state is not None

        def write_gathered() -> None:
            if int(session.rank) != 0:
                return
            flat = _gathered_flat(
                prepared.deck, final_state, history.initial_state,
                clock.step, clock.sim_time, history.divb_series,
                history.snapshots, history.extra_scalars, prepared.nghost)
            _atomic_savez(prepared.out_path, flat)

        _collective_local(
            session, "mhd-diagnostics-gathered-write", write_gathered)
        return prepared.out_path

    final_shards = _local_cell_shards(session, prepared.deck)
    return _write_sharded(
        session, prepared.out_path, prepared.deck, final_shards,
        history.initial_shards, clock.step, clock.sim_time,
        history.divb_series, history.sharded_snapshots,
        history.extra_scalars, prepared.nghost, launch.mapping,
        launch.topology)


def _collect_mhd_telemetry(
        session: Any, launch: _MhdLaunch,
        options: _distributed.RunOptions,
        phase_times: _RunPhaseTimes) -> dict[str, Any]:
    telemetry = _collective_local(
        session, "mhd-telemetry", lambda: dict(session.telemetry))

    def finalize_telemetry() -> None:
        _finalize_run_telemetry(
            telemetry, physics="mhd",
            requested_transport=options.transport,
            decomposition=launch.topology["decomposition"],
            phase_times=phase_times)

    _collective_local(
        session, "mhd-telemetry-assemble", finalize_telemetry)
    return telemetry


def run(input_deck: Any, options: _distributed.RunOptions, *,
        steps_override: int | None = None, verbose: bool = False,
        print_config: bool = False, log_every: int = 0) -> dict[str, Any]:
    """Execute one shared distributed MHD simulation."""

    session = _distributed.RuntimeSession()
    started = False
    start_wall = time.perf_counter()
    try:
        prepared = _collective_local(
            session, "mhd-python-prepare",
            lambda: _prepare_mhd_run(
                input_deck, options, steps_override=steps_override,
                verbose=verbose, print_config=print_config,
                log_every=log_every))
        prepare_done = time.perf_counter()
        launch = _launch_mhd_runtime(session, prepared, options)
        started = True
        restored_history, clock = _finish_mhd_startup(
            session, prepared, launch)
        _print_mhd_config(
            session, prepared, launch, options, print_config, clock.dt)
        history = _initialize_mhd_history(
            session, prepared, options, clock, restored_history)
        setup_done = time.perf_counter()
        checkpoint_seconds, evolution_done = _evolve_mhd(
            session, prepared, options, history, clock, start_wall)
        diagnostics_path = _write_final_mhd_diagnostics(
            session, prepared, launch, options, history, clock)
        diagnostics_done = time.perf_counter()
        telemetry = _collect_mhd_telemetry(
            session, launch, options,
            _RunPhaseTimes(
                start_wall, prepare_done, setup_done, evolution_done,
                diagnostics_done, checkpoint_seconds))
        _collective_local(
            session, "mhd-print-result",
            lambda: print(f"wrote   : {diagnostics_path}")
            if verbose and int(session.rank) == 0 else None)
        return {
            "final_step": clock.step,
            "final_time": clock.sim_time,
            "diagnostics_path": diagnostics_path,
            "checkpoint_path": prepared.checkpoint_path,
            "distributed": True,
            "telemetry": telemetry,
        }
    finally:
        # close() first closes an active physics runtime, then tears down MPI only
        # if this session initialized it.  Destruction itself is non-collective.
        if started or not session.closed:
            session.close()
