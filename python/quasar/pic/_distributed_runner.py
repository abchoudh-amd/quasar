"""High-level orchestration for the native tile-decomposed PIC runtime."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import operator
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import _core
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
from ..coil.io import build_conductor_system
from . import initial_conditions as ic
from . import io as pic_io
from ._units import QE as EV_TO_J
from ._units import Units


_CHECKPOINT_DIAGNOSTICS_SCHEMA = "quasar-pic-checkpoint-diagnostics/v1"
_PIC_FIELD_COMPONENTS = ("ex", "ey", "ez", "bx", "by", "bz")


def _deck_and_path(input_deck: Any) -> tuple[pic_io.PicDeck, Path | None]:
    if isinstance(input_deck, pic_io.PicDeck):
        return input_deck, None
    path = Path(input_deck).resolve()
    return pic_io.load(path), path


def _field_extents(
        nx: int, ny: int, geometry: str,
        ) -> dict[str, tuple[int, int, bool, bool]]:
    """Return ``(rows, columns, face_x, face_y)`` for every Yee component."""

    if geometry == "cartesian":
        return {
            "ex": (ny, nx + 1, True, False),
            "ey": (ny + 1, nx, False, True),
            "ez": (ny, nx, False, False),
            "bx": (ny + 1, nx, False, True),
            "by": (ny, nx + 1, True, False),
            "bz": (ny + 1, nx + 1, True, True),
        }
    if geometry == "cylindrical":
        return {
            "ex": (ny, nx + 1, True, False),
            "ey": (ny + 1, nx, False, True),
            "ez": (ny, nx + 1, True, False),
            "bx": (ny + 1, nx + 1, True, True),
            "by": (ny, nx, False, False),
            "bz": (ny + 1, nx + 1, True, True),
        }
    raise ValueError(f"unsupported PIC geometry {geometry!r}")


def _empty_fields(deck: pic_io.PicDeck) -> dict[str, Any]:
    result: dict[str, Any] = {
        "global_nx": deck.domain.nx,
        "global_ny": deck.domain.ny,
    }
    for name, (rows, columns, _, _) in _field_extents(
            deck.domain.nx, deck.domain.ny, deck.geometry).items():
        result[name] = np.zeros((rows, columns), dtype=np.float64)
    return result


def _periodic_axes(deck: pic_io.PicDeck) -> tuple[bool, bool]:
    periodic_x = (deck.boundary.field[0] == deck.boundary.field[1] == "periodic"
                  and deck.boundary.particle[0]
                  == deck.boundary.particle[1] == "periodic")
    periodic_y = (deck.boundary.field[2] == deck.boundary.field[3] == "periodic"
                  and deck.boundary.particle[2]
                  == deck.boundary.particle[3] == "periodic")
    return periodic_x, periodic_y


def _rebuild_periodic_duplicates(
        fields: dict[str, Any], deck: pic_io.PicDeck) -> None:
    periodic_x, periodic_y = _periodic_axes(deck)
    extents = _field_extents(deck.domain.nx, deck.domain.ny, deck.geometry)
    for name, (_, _, face_x, face_y) in extents.items():
        values = np.asarray(fields[name], dtype=np.float64)
        if face_x and periodic_x:
            values[:, -1] = values[:, 0]
        if face_y and periodic_y:
            values[-1, :] = values[0, :]
        fields[name] = np.ascontiguousarray(values)


class _CanonicalFieldSink:
    """Adapter that lets the established field seeder target host lattices."""

    def __init__(self, deck: pic_io.PicDeck, nghost: int) -> None:
        self.deck = deck
        self._nghost = nghost
        self.fields = _empty_fields(deck)

    def nghost(self) -> int:
        return self._nghost

    def seed_field(self, component: str, values: Any) -> None:
        nx, ny, g = self.deck.domain.nx, self.deck.domain.ny, self._nghost
        padded = np.asarray(values, dtype=np.float64).reshape(
            ny + 2 * g, nx + 2 * g)
        rows, columns, _, _ = _field_extents(
            nx, ny, self.deck.geometry)[component]
        self.fields[component] = np.ascontiguousarray(
            padded[g:g + rows, g:g + columns])


def _canonical_initial_fields(
        deck: pic_io.PicDeck, units: Units, first_dt: float,
        nghost: int) -> dict[str, Any]:
    from .cli import _seed_fields

    sink = _CanonicalFieldSink(deck, nghost)
    _seed_fields(sink, deck, first_dt, units)
    _rebuild_periodic_duplicates(sink.fields, deck)
    return sink.fields


def _counter_maxwellian(
        count: int, thermal_speed: float, drift: tuple[float, float, float],
        seed: int, species_index: int) -> np.ndarray:
    """Counter-based, species-keyed antithetic Maxwellian sampling."""

    seed_word = int(seed) % (1 << 64)
    sequence = np.random.SeedSequence([
        seed_word & 0xFFFFFFFF,
        seed_word >> 32,
        int(species_index) & 0xFFFFFFFF,
        (int(species_index) >> 32) & 0xFFFFFFFF,
    ])
    generator = np.random.Generator(np.random.Philox(sequence))
    pair_count = count // 2
    with np.errstate(over="ignore", invalid="ignore"):
        draws = generator.normal(
            0.0, thermal_speed, size=(pair_count, 3))
        result = np.empty((count, 3), dtype=np.float64)
        result[0:2 * pair_count:2] = draws
        result[1:2 * pair_count:2] = -draws
        if count % 2:
            result[-1] = 0.0
        result += np.asarray(drift, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(
            "thermal speed and drift produce non-finite Maxwellian velocities")
    return result


def _species_states(
        deck: pic_io.PicDeck, units: Units, seed: int,
        ) -> list[dict[str, Any]]:
    from .cli import _macro_weight

    result: list[dict[str, Any]] = []
    next_id = 0
    domain_lx = units.length(deck.domain.lx_m)
    domain_ly = units.length(deck.domain.ly_m)
    domain_ox = units.length(deck.domain.origin_x_m)
    domain_oy = units.length(deck.domain.origin_y_m)
    for species_index, species in enumerate(deck.species):
        if units.identity:
            thermal_speed = float(np.sqrt(
                species.initial.temperature_eV / species.mass_kg))
        else:
            thermal_speed_si = float(np.sqrt(
                species.initial.temperature_eV * EV_TO_J
                / species.mass_kg))
            thermal_speed = units.velocity(thermal_speed_si)
        drift = tuple(units.velocity(value)
                      for value in species.initial.drift_v)

        if species.initial.distribution == "maxwellian_block":
            x_min = units.length(species.initial.region_x_min_m)
            x_max = units.length(species.initial.region_x_max_m)
            y_min = units.length(species.initial.region_y_min_m)
            y_max = units.length(species.initial.region_y_max_m)
        else:
            x_min, x_max = domain_ox, domain_ox + domain_lx
            y_min, y_max = domain_oy, domain_oy + domain_ly
        if deck.geometry == "cylindrical":
            positions = ic.quiet_positions_rz_block(
                species.n_particles, x_min, x_max, y_min, y_max)
            particle_volume = ic.quiet_block_ring_volume(
                species.n_particles, x_min, x_max, y_min, y_max)
        else:
            positions = ic.quiet_positions_2d_block(
                species.n_particles, x_min, x_max, y_min, y_max)
            particle_volume = ic.quiet_block_cell_area(
                species.n_particles, x_min, x_max, y_min, y_max)

        velocities = _counter_maxwellian(
            species.n_particles, thermal_speed, drift, seed, species_index)
        perturbation = species.initial.velocity_perturbation
        if perturbation is not None:
            mx, my = perturbation.mode
            phase = (2.0 * np.pi
                     * (mx * (positions[:, 0] - domain_ox) / domain_lx
                        + my * (positions[:, 1] - domain_oy) / domain_ly)
                     + math.remainder(
                         perturbation.phase_rad, 2.0 * math.pi))
            amplitude = np.asarray([
                units.velocity(value)
                for value in perturbation.amplitude_v], dtype=np.float64)
            velocities += np.sin(phase)[:, np.newaxis] * amplitude
        speeds = np.linalg.norm(velocities, axis=1)
        if speeds.size and float(np.max(speeds)) >= 1.0:
            raise ValueError(
                f"species {species.name!r}: sampled |v|/c >= 1, outside the "
                "nonrelativistic Boris model; lower temperature/drift or use "
                "a relativistic pusher")

        macro_weight = _macro_weight(
            species.initial.density_per_m3,
            units.density(species.initial.density_per_m3),
            particle_volume, species.name)
        stop_id = next_id + species.n_particles
        if stop_id > 1 << 64:
            raise OverflowError("distributed PIC stable particle IDs overflow uint64")
        identifiers = np.arange(next_id, stop_id, dtype=np.uint64)
        next_id = stop_id
        x = np.ascontiguousarray(positions[:, 0], dtype=np.float64)
        y = np.ascontiguousarray(positions[:, 1], dtype=np.float64)
        vz = np.ascontiguousarray(velocities[:, 2], dtype=np.float64)
        particles = {
            "x": x,
            "y": y,
            "x_prev": x.copy(),
            "y_prev": y.copy(),
            "vx": np.ascontiguousarray(velocities[:, 0], dtype=np.float64),
            "vy": np.ascontiguousarray(velocities[:, 1], dtype=np.float64),
            "vz": vz,
            "vphi_deposit": vz.copy(),
            "weight": np.full(
                species.n_particles, macro_weight, dtype=np.float64),
            "alive": np.ones(species.n_particles, dtype=np.uint8),
            "id": identifiers,
        }
        result.append({
            "config": {
                "name": species.name,
                "charge": units.charge(species.charge_C),
                "mass": units.mass(species.mass_kg),
                "capacity": species.n_particles,
            },
            "particles": particles,
        })
    return result


def _species_configs(
        deck: pic_io.PicDeck, units: Units) -> list[dict[str, Any]]:
    return [{
        "name": species.name,
        "charge": units.charge(species.charge_C),
        "mass": units.mass(species.mass_kg),
        "capacity": species.n_particles,
    } for species in deck.species]


def _external_signature(deck: pic_io.PicDeck) -> str:
    if deck.external_field is None:
        return "none"
    field = deck.external_field
    # Hash only inputs consumed by the selected evaluator.  This excludes
    # inactive dataclass defaults and a file grid's source location while
    # retaining its resolved samples through evaluator_params().  Plane,
    # geometry, units, and normalization live in the surrounding numerical
    # checkpoint signature.
    document = {
        "evaluator_type": field.evaluator_type,
        "parameters": field.evaluator_params(),
        "conductors": field.conductors,
    }
    encoded = json.dumps(
        document, allow_nan=False, separators=(",", ":"),
        sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _timestep_signature(deck: pic_io.PicDeck, internal_dt: float) -> str:
    """Return the restart identity for policy and nominal internal dt."""

    policy = "auto" if deck.time.dt_s == "auto" else "fixed"
    return f"policy={policy};dt={float(internal_dt).hex()}"


def _make_external_evaluator(
        deck: pic_io.PicDeck, units: Units,
        ) -> tuple[Any, Any, tuple[float, float, float]] | None:
    if deck.external_field is None:
        return None
    field = deck.external_field
    evaluator = _core.magnetostatics.create_field_evaluator(
        field.evaluator_type)
    evaluator.configure(field.evaluator_params())
    source = build_conductor_system(field.conductors)
    return evaluator, source, units.external_scales()


def _resolve_timestep(
        deck: pic_io.PicDeck, units: Units,
        ) -> tuple[float, float, float | None]:
    from .cli import (_cfl_dt_internal, _cfl_limit_internal,
                      _cyl_cfl_dt_internal, _cyl_cfl_limit_internal,
                      _internal_end_time)

    cylindrical = deck.geometry == "cylindrical"
    if deck.time.dt_s == "auto":
        dt = ((_cyl_cfl_dt_internal if cylindrical else _cfl_dt_internal)(
            deck.domain, units, deck.numerics.fdtd_order))
    else:
        dt = units.time(float(deck.time.dt_s))
        limit = ((_cyl_cfl_limit_internal
                  if cylindrical else _cfl_limit_internal)(
                      deck.domain, units, deck.numerics.fdtd_order))
        if dt > limit:
            qualifier = "cylindrical (r-z) " if cylindrical else ""
            raise ValueError(
                f"time.dt_s ({units.time_to_si(dt):.6e} s) exceeds the "
                f"{qualifier}CFL stability limit "
                f"({units.time_to_si(limit):.6e} s) for this grid; reduce "
                "dt_s or use 'auto'.")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            "time.dt_s is not representable as a positive solver timestep")
    return dt, float(units.time_to_si(dt)), _internal_end_time(deck, units)


def _padded_component(
        values: Any, component: str, deck: pic_io.PicDeck,
        nghost: int) -> np.ndarray:
    nx, ny, g = deck.domain.nx, deck.domain.ny, nghost
    rows, columns, _, _ = _field_extents(nx, ny, deck.geometry)[component]
    canonical = np.asarray(values, dtype=np.float64).reshape(rows, columns)
    padded = np.zeros((ny + 2 * g, nx + 2 * g), dtype=np.float64)
    padded[g:g + rows, g:g + columns] = canonical
    return padded.reshape(-1)


def _snapshot(
        state: dict[str, Any], deck: pic_io.PicDeck, step: int,
        sim_time: float, units: Units, nghost: int) -> dict[str, Any]:
    from .cli import _species_to_si

    fields = state["fields"]
    external = state["external_fields"]
    snapshot: dict[str, Any] = {
        "step": step,
        "time_s": sim_time,
        "fields": {
            name: units.field_component_to_si(
                name, _padded_component(fields[name], name, deck, nghost))
            for name in deck.diagnostics.fields
        },
        "external_bx": units.field_component_to_si(
            "bx", _padded_component(external["bx"], "bx", deck, nghost)),
        "external_by": units.field_component_to_si(
            "by", _padded_component(external["by"], "by", deck, nghost)),
        "external_bz": units.field_component_to_si(
            "bz", _padded_component(external["bz"], "bz", deck, nghost)),
        "nx": deck.domain.nx,
        "ny": deck.domain.ny,
        "nghost": nghost,
        "plane": deck.plane,
        "geometry": deck.geometry,
        "boundary_field": tuple(deck.boundary.field),
        "origin_x": deck.domain.origin_x_m,
        "origin_y": deck.domain.origin_y_m,
        "lx": deck.domain.lx_m,
        "ly": deck.domain.ly_m,
        "unit_system": deck.units,
    }
    if deck.diagnostics.per_species:
        snapshot["species"] = {
            str(item["config"]["name"]): _species_to_si(
                item["particles"], units)
            for item in state["species"]
        }
    return snapshot


def _gather_state(
        session: Any, phase: str) -> dict[str, Any]:
    """Gather native PIC state, then collectively validate Python conversion."""

    native_state = session.pic_gather_state()
    return _collective_local(session, phase, lambda: dict(native_state))


def _local_owned_shards(
        session: Any, include_particles: bool, phase: str,
        ) -> list[dict[str, Any]]:
    """Extract native PIC shards, then collectively validate list conversion."""

    native_shards = session.pic_local_owned_shards(
        include_particles=include_particles)
    return _collective_local(
        session, phase,
        lambda: [dict(item) for item in native_shards])


def _component_owned_slice(
        values: Any, component: str, deck: pic_io.PicDeck,
        tile: dict[str, Any]) -> tuple[np.ndarray, tuple[int, int]]:
    rows, columns, face_x, face_y = _field_extents(
        deck.domain.nx, deck.domain.ny, deck.geometry)[component]
    values = np.asarray(values, dtype=np.float64).reshape(rows, columns)
    offset_y, offset_x = map(int, tile["offset"])
    owned_ny, owned_nx = map(int, tile["owned_shape"])
    tile_x, tile_y = map(int, tile["tile"])
    begin_x = offset_x + (1 if face_x and tile_x > 0 else 0)
    begin_y = offset_y + (1 if face_y and tile_y > 0 else 0)
    end_x = offset_x + owned_nx + (1 if face_x else 0)
    end_y = offset_y + owned_ny + (1 if face_y else 0)
    return (np.ascontiguousarray(values[begin_y:end_y, begin_x:end_x]),
            (begin_y, begin_x))


def _owned_component(
        component: dict[str, Any], label: str,
        ) -> tuple[np.ndarray, tuple[int, int]]:
    try:
        offset = tuple(map(int, component["offset"]))
        shape = tuple(map(int, component["shape"]))
        values = np.asarray(component["values"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} has invalid owned-array metadata") from exc
    if len(offset) != 2 or len(shape) != 2:
        raise ValueError(f"{label} offset and shape must be two-dimensional")
    if any(value < 0 for value in (*offset, *shape)):
        raise ValueError(f"{label} offset and shape must be non-negative")
    if values.size != shape[0] * shape[1]:
        raise ValueError(f"{label} values do not match its owned shape")
    return np.ascontiguousarray(values.reshape(shape)), offset


def _shard_payload(
        shard: dict[str, Any], snapshots: list[dict[str, Any]],
        series: dict[str, list[Any]] | None, deck: pic_io.PicDeck,
        units: Units, nghost: int, final_step: int, final_time: float,
        ) -> dict[str, np.ndarray]:
    offset_y, offset_x = map(int, shard["offset"])
    owned_ny, owned_nx = map(int, shard["owned_shape"])
    payload: dict[str, np.ndarray] = {
        "final_step": np.array([final_step]),
        "final_time_s": np.array([final_time]),
        "nx": np.array([deck.domain.nx]),
        "ny": np.array([deck.domain.ny]),
        "nghost": np.array([nghost]),
        "plane": np.array([deck.plane]),
        "geometry": np.array([deck.geometry]),
        "boundary_field": np.asarray(deck.boundary.field, dtype=np.str_),
        "origin_x": np.array([deck.domain.origin_x_m]),
        "origin_y": np.array([deck.domain.origin_y_m]),
        "lx": np.array([deck.domain.lx_m]),
        "ly": np.array([deck.domain.ly_m]),
        "unit_system": np.array([deck.units]),
        "offset": np.array([offset_y, offset_x]),
        "owned_shape": np.array([owned_ny, owned_nx]),
    }
    for name in deck.diagnostics.fields:
        values, component_offset = _owned_component(
            shard["fields"][name], f"field {name}")
        payload[f"field_{name}"] = units.field_component_to_si(name, values)
        payload[f"field_{name}_offset"] = np.asarray(component_offset)
    for name in ("bx", "by", "bz"):
        values, component_offset = _owned_component(
            shard["external_fields"][name], f"external field {name}")
        payload[f"external_{name}"] = units.field_component_to_si(name, values)
        payload[f"external_{name}_offset"] = np.asarray(component_offset)

    if deck.diagnostics.per_species:
        names: list[str] = []
        for item in shard["species"]:
            name = str(item["config"]["name"])
            names.append(name)
            particles = item["particles"]
            for key, raw in particles.items():
                values = np.asarray(raw)
                if key in ("x", "y", "x_prev", "y_prev"):
                    values = units.length_to_si(values)
                elif key in ("vx", "vy", "vz", "vphi_deposit"):
                    values = units.velocity_to_si(values)
                payload[f"species_{name}_{key}"] = np.asarray(values)
        payload["species_names"] = np.asarray(names, dtype=np.str_)

    if snapshots:
        payload["snapshot_steps"] = np.asarray(
            [item["step"] for item in snapshots])
        payload["snapshot_times_s"] = np.asarray(
            [item["time_s"] for item in snapshots])
        for name in deck.diagnostics.fields:
            values = []
            for item in snapshots:
                owned, _ = _owned_component(
                    item["fields"][name], f"snapshot field {name}")
                values.append(units.field_component_to_si(name, owned))
            payload[f"snapshot_field_{name}"] = np.stack(values)
    if series:
        for name, values in series.items():
            payload[f"series_{name}"] = np.asarray(values)
    return payload


def _pic_checkpoint_fragment(
        rank: int, layout: str, deck: pic_io.PicDeck,
        snapshots: list[dict[str, Any]],
        sharded_snapshots: dict[int, list[dict[str, Any]]],
        series: dict[str, list[Any]]) -> bytes:
    """Encode rank-local globally located PIC history without a solver gather."""

    pieces: list[dict[str, Any]] = []
    if layout == "gathered":
        if rank != 0:
            return b""
        pieces.append({"history": snapshots, "gathered": True})
    else:
        for endpoint in sorted(sharded_snapshots):
            pieces.append({
                "history": sharded_snapshots[endpoint], "gathered": False})

    reference = pieces[0]["history"] if pieces else []
    steps = [int(item["step"]) for item in reference]
    times = [float(item["time_s"]) for item in reference]
    for piece in pieces[1:]:
        history = piece["history"]
        if ([int(item["step"]) for item in history] != steps
                or [float(item["time_s"]) for item in history] != times):
            raise ValueError(
                "rank-local PIC checkpoint histories disagree on snapshots")

    payload: dict[str, np.ndarray] = {
        "schema": np.array([_CHECKPOINT_DIAGNOSTICS_SCHEMA]),
        "physics": np.array(["pic"]),
        "global_shape": np.asarray(
            [deck.domain.ny, deck.domain.nx], dtype=np.uint64),
        "piece_count": np.array([len(pieces)], dtype=np.uint64),
        "snapshot_steps": np.asarray(steps, dtype=np.uint64),
        "snapshot_times": np.asarray(times, dtype=np.float64),
        "common_present": np.array([rank == 0], dtype=np.uint8),
    }
    if rank == 0:
        names = sorted(series)
        payload["series_names"] = np.asarray(names, dtype=np.str_)
        for index, name in enumerate(names):
            values = np.asarray(series[name])
            if values.ndim != 1:
                raise ValueError(
                    "PIC checkpoint scalar series must be one-dimensional")
            payload[f"series.{index}"] = values

    extents = _field_extents(deck.domain.nx, deck.domain.ny, deck.geometry)
    for piece_index, piece in enumerate(pieces):
        prefix = f"piece.{piece_index}."
        history = piece["history"]
        for name in _PIC_FIELD_COMPONENTS:
            rows, columns, _, _ = extents[name]
            values: list[np.ndarray] = []
            component_offset: tuple[int, int] | None = None
            if piece["gathered"]:
                component_offset = (0, 0)
                for item in history:
                    canonical = item.get("canonical_fields")
                    if not isinstance(canonical, dict) or name not in canonical:
                        raise ValueError(
                            "gathered PIC checkpoint snapshot lacks canonical fields")
                    values.append(np.asarray(
                        canonical[name], dtype=np.float64).reshape(rows, columns))
                owned_shape = (rows, columns)
            else:
                for item in history:
                    owned, offset = _owned_component(
                        item["fields"][name], f"checkpoint field {name}")
                    if component_offset is None:
                        component_offset = offset
                    elif component_offset != offset:
                        raise ValueError(
                            "PIC checkpoint piece ownership changed within history")
                    values.append(owned)
                # No values means there are no snapshots and therefore no
                # field history to cover.  The decoder ignores empty pieces.
                owned_shape = values[0].shape if values else (0, 0)
            if component_offset is None:
                component_offset = (0, 0)
            if any(value.shape != owned_shape for value in values):
                raise ValueError("PIC checkpoint field ownership is inconsistent")
            payload[prefix + name + ".offset"] = np.asarray(
                component_offset, dtype=np.uint64)
            payload[prefix + name + ".values"] = (
                np.stack(values) if values else
                np.empty((0, *owned_shape), dtype=np.float64))
    return _encode_checkpoint_fragment(payload)


def _decode_pic_checkpoint_history(
        parts: list[bytes], deck: pic_io.PicDeck,
        ) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    """Validate PIC fragments and assemble canonical global snapshots."""

    archives = [_decode_checkpoint_fragment(part) for part in parts if part]
    if not archives:
        raise ValueError("PIC checkpoint has no diagnostic continuation state")
    mesh_shape = (deck.domain.ny, deck.domain.nx)
    extents = _field_extents(deck.domain.nx, deck.domain.ny, deck.geometry)
    steps: np.ndarray | None = None
    times: np.ndarray | None = None
    fields: dict[str, np.ndarray] = {}
    coverage: dict[str, np.ndarray] = {}
    common_series: dict[str, list[Any]] | None = None

    for archive in archives:
        if (_checkpoint_scalar_text(archive, "schema")
                != _CHECKPOINT_DIAGNOSTICS_SCHEMA
                or _checkpoint_scalar_text(archive, "physics") != "pic"):
            raise ValueError("PIC checkpoint diagnostic schema is incompatible")
        global_shape = np.asarray(archive.get("global_shape"), dtype=np.uint64)
        if (global_shape.shape != (2,)
                or tuple(map(int, global_shape)) != mesh_shape):
            raise ValueError("PIC checkpoint diagnostic mesh is incompatible")
        local_steps = np.asarray(archive.get("snapshot_steps"), dtype=np.uint64)
        local_times = np.asarray(archive.get("snapshot_times"), dtype=np.float64)
        if (local_steps.ndim != 1 or local_times.shape != local_steps.shape
                or not np.all(np.isfinite(local_times))
                or np.any(local_times < 0.0)):
            raise ValueError("PIC checkpoint snapshot index is invalid")
        if steps is None:
            steps, times = local_steps, local_times
            for name in _PIC_FIELD_COMPONENTS:
                rows, columns, _, _ = extents[name]
                fields[name] = np.empty(
                    (steps.size, rows, columns), dtype=np.float64)
                coverage[name] = np.zeros((rows, columns), dtype=bool)
        elif (not np.array_equal(steps, local_steps)
              or not np.array_equal(times, local_times)):
            raise ValueError("PIC checkpoint fragments disagree on snapshots")

        if _checkpoint_scalar_int(archive, "common_present", maximum=1):
            if common_series is not None:
                raise ValueError("PIC checkpoint has duplicate scalar history")
            names = np.asarray(archive.get("series_names"))
            if names.ndim != 1 or names.dtype.kind not in ("U", "S"):
                raise ValueError("PIC checkpoint scalar names are invalid")
            common_series = {}
            for index, raw_name in enumerate(names):
                name = str(raw_name)
                if (not name or len(name) > 200 or name in common_series
                        or (name not in ("step", "time_s")
                            and not name.startswith("alive_"))):
                    raise ValueError("PIC checkpoint scalar name is invalid")
                values = np.asarray(archive.get(f"series.{index}"))
                if values.ndim != 1 or values.dtype.kind not in "iuf":
                    raise ValueError("PIC checkpoint scalar series is invalid")
                if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
                    raise ValueError("PIC checkpoint scalar series is non-finite")
                common_series[name] = values.tolist()

        piece_count = _checkpoint_scalar_int(
            archive, "piece_count", maximum=4096)
        for index in range(piece_count):
            prefix = f"piece.{index}."
            for name in _PIC_FIELD_COMPONENTS:
                offset = np.asarray(
                    archive.get(prefix + name + ".offset"), dtype=np.uint64)
                values = np.asarray(
                    archive.get(prefix + name + ".values"), dtype=np.float64)
                if (offset.shape != (2,) or values.ndim != 3
                        or values.shape[0] != local_steps.size):
                    raise ValueError("PIC checkpoint field piece is invalid")
                ny, nx = values.shape[1:]
                if ny == 0 or nx == 0:
                    if local_steps.size != 0:
                        raise ValueError("PIC checkpoint has an empty field piece")
                    continue
                oy, ox = map(int, offset)
                target = coverage[name]
                if (oy + ny > target.shape[0] or ox + nx > target.shape[1]
                        or np.any(target[oy:oy + ny, ox:ox + nx])
                        or not np.all(np.isfinite(values))):
                    raise ValueError("PIC checkpoint field pieces overlap or overflow")
                fields[name][:, oy:oy + ny, ox:ox + nx] = values
                target[oy:oy + ny, ox:ox + nx] = True

    if steps is None or times is None or common_series is None:
        raise ValueError("PIC checkpoint diagnostic continuation state is incomplete")
    if steps.size and any(not np.all(mask) for mask in coverage.values()):
        raise ValueError("PIC checkpoint snapshot pieces do not cover every field")
    snapshots = [{
        "step": int(steps[index]),
        "time_s": float(times[index]),
        "canonical_fields": {
            name: np.ascontiguousarray(fields[name][index])
            for name in _PIC_FIELD_COMPONENTS},
    } for index in range(steps.size)]
    return snapshots, common_series


def _write_sharded(
        session: Any, out_path: Path, local_shards: list[dict[str, Any]],
        snapshots: dict[int, list[dict[str, Any]]],
        series: dict[str, list[Any]] | None,
        deck: pic_io.PicDeck, units: Units, nghost: int,
        final_step: int, final_time: float,
        mapping: list[dict[str, Any]], topology: dict[str, Any]) -> Path:
    manifest_path = out_path.with_suffix(".manifest.json")

    def prepare_records() -> list[dict[str, Any]]:
        tiles = {
            int(item["endpoint"]): item for item in topology["tiles"]}
        prepared: list[dict[str, Any]] = []
        for endpoint_info in mapping:
            endpoint = int(endpoint_info["index"])
            tile = tiles[endpoint]
            shard_path = out_path.with_name(
                f"{out_path.stem}.rank{int(endpoint_info['rank']):06d}."
                f"gpu{int(endpoint_info['rank_local_index']):03d}.npz")
            prepared.append({
                "rank": int(endpoint_info["rank"]),
                "node_rank": int(endpoint_info["node_rank"]),
                "local_device": int(endpoint_info["rank_local_index"]),
                "endpoint": endpoint,
                "device_identity": str(endpoint_info["device_identity"]),
                "tile": list(map(int, tile["tile"])),
                "offset": list(map(int, tile["offset"])),
                "owned_shape": list(map(int, tile["owned_shape"])),
                "path": shard_path.name,
            })
        return prepared

    records = _collective_local(
        session, "pic-diagnostics-sharded-prepare", prepare_records)

    _collective_local(
        session, "pic-diagnostics-sharded-begin",
        lambda: manifest_path.unlink(missing_ok=True)
        if int(session.rank) == 0 else None)

    def write_local_shards() -> None:
        local_records = {
            int(record["endpoint"]): record for record in records
            if record["rank"] == int(session.rank)
        }
        shards_by_endpoint: dict[int, dict[str, Any]] = {}
        for shard in local_shards:
            endpoint = int(shard["endpoint"])
            if endpoint in shards_by_endpoint:
                raise ValueError(
                    f"duplicate rank-local PIC shard for endpoint {endpoint}")
            shards_by_endpoint[endpoint] = shard
        if set(shards_by_endpoint) != set(local_records):
            raise ValueError(
                "rank-local PIC shard endpoints do not match the endpoint "
                f"mapping: expected {sorted(local_records)}, received "
                f"{sorted(shards_by_endpoint)}")
        if not set(snapshots).issubset(local_records):
            raise ValueError(
                "rank-local PIC snapshots contain a remote endpoint")

        for endpoint, record in local_records.items():
            shard = shards_by_endpoint[endpoint]
            for key in ("tile", "offset", "owned_shape"):
                if tuple(map(int, shard[key])) != tuple(record[key]):
                    raise ValueError(
                        f"rank-local PIC shard {endpoint} has inconsistent "
                        f"{key} metadata")
            endpoint = int(record["endpoint"])
            payload = _shard_payload(
                shard, snapshots.get(endpoint, []), series, deck, units,
                nghost, final_step, final_time)
            _atomic_savez(out_path.parent / record["path"], payload)

    _collective_local(
        session, "pic-diagnostics-shard-write", write_local_shards)

    def publish_manifest() -> None:
        if int(session.rank) != 0:
            return
        document = {
            "schema": "quasar-diagnostics-shards/v1",
            "physics": "pic",
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
        session, "pic-diagnostics-manifest-publish",
        publish_manifest)
    return manifest_path


def _seed_value(value: int | None) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError("seed must be an integer")
    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("seed must be an integer") from exc


@dataclass(frozen=True)
class _PreparedPicRun:
    deck: pic_io.PicDeck
    deck_path: Path | None
    units: Units
    config: Any
    nghost: int
    dt: float
    dt_si: float
    t_end_internal: float | None
    fields: dict[str, Any] | None
    species: list[dict[str, Any]] | None
    expected_species: list[dict[str, Any]]
    external: Any
    out_path: Path
    checkpoint_path: Path | None
    restart_path: Path | None
    log_interval: int
    write_interval: int
    policy_signature: str


@dataclass(frozen=True)
class _PicLaunch:
    mapping: list[dict[str, Any]]
    topology: dict[str, Any]
    restart_metadata: dict[str, Any] | None


@dataclass
class _PicClock:
    step: int
    sim_time: float
    solver_time: float


@dataclass
class _PicHistory:
    snapshots: list[dict[str, Any]]
    sharded_snapshots: dict[int, list[dict[str, Any]]]
    series: dict[str, list[Any]]


def _pic_policy_signature(
        deck: pic_io.PicDeck, options: _distributed.RunOptions, *,
        out_path: Path, checkpoint_path: Path | None,
        restart_path: Path | None, log_interval: int, write_interval: int,
        dt: float, dt_si: float, seed: int | None,
        external_signature: str, verbose: bool,
        print_config: bool) -> str:
    return _canonical_policy_signature({
        "schema": "quasar-distributed-run-policy/v1",
        "physics": "pic",
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
            "per_species": bool(deck.diagnostics.per_species),
        },
        "termination": {
            "steps": int(deck.time.steps),
            "end_time": (None if deck.time.t_end_s is None else
                         float(deck.time.t_end_s)),
        },
        "timestep": {
            "policy": (deck.time.dt_s if deck.time.dt_s == "auto" else
                       float(deck.time.dt_s)),
            "internal": float(dt),
            "seconds": float(dt_si),
        },
        "placement": {
            "devices": options.devices,
            "decomposition": options.decomposition,
            "transport": options.transport,
        },
        "log_cadence": log_interval,
        "write_cadence": write_interval,
        "verbose": bool(verbose),
        "print_config": bool(print_config),
        "seed": seed,
        "external_field": external_signature,
    })


def _prepare_pic_run(
        input_deck: Any, options: _distributed.RunOptions, *,
        seed: int | None, steps_override: int | None,
        verbose: bool, print_config: bool, log_every: int,
        write_every: int) -> _PreparedPicRun:
    if options.restart is not None and seed is not None:
        raise ValueError("PIC seed cannot be supplied with restart")
    deck, deck_path = _deck_and_path(input_deck)
    if steps_override is not None:
        deck = copy.deepcopy(deck)
        deck.time = pic_io.Time(
            dt_s=deck.time.dt_s, steps=steps_override,
            t_end_s=deck.time.t_end_s)
        deck.validate()
    units = Units(deck)
    from .cli import _make_config, _required_solver_nghost
    config = _make_config(deck, units)
    external_signature = _external_signature(deck)
    if hasattr(config, "external_field_signature"):
        config.external_field_signature = external_signature
    nghost = _required_solver_nghost(deck)
    dt, dt_si, t_end_internal = _resolve_timestep(deck, units)
    config.timestep_signature = _timestep_signature(deck, dt)
    first_dt = dt if t_end_internal is None else min(dt, t_end_internal)
    seed_value = None if options.restart is not None else _seed_value(seed)
    fields = (None if options.restart is not None else
              _canonical_initial_fields(deck, units, first_dt, nghost))
    species = (None if options.restart is not None else
               _species_states(deck, units, seed_value))
    expected_species = _species_configs(deck, units)
    external = _make_external_evaluator(deck, units)
    deck_directory = deck_path.parent if deck_path is not None else Path.cwd()
    out_path = confine_output_path(
        deck_directory, deck.diagnostics.output_path,
        label="diagnostics.output_path")
    checkpoint_path = (None if options.checkpoint is None else
                       Path(options.checkpoint).expanduser().resolve())
    restart_path = (None if options.restart is None else
                    Path(options.restart).expanduser().resolve())
    log_interval = max(0, int(log_every))
    write_interval = max(0, int(write_every))
    policy_signature = _pic_policy_signature(
        deck, options, out_path=out_path,
        checkpoint_path=checkpoint_path, restart_path=restart_path,
        log_interval=log_interval, write_interval=write_interval,
        dt=dt, dt_si=dt_si, seed=seed_value,
        external_signature=external_signature, verbose=verbose,
        print_config=print_config)
    return _PreparedPicRun(
        deck, deck_path, units, config, nghost, dt, dt_si,
        t_end_internal, fields, species, expected_species, external,
        out_path, checkpoint_path, restart_path, log_interval,
        write_interval, policy_signature)


def _launch_pic_runtime(
        session: Any, prepared: _PreparedPicRun,
        options: _distributed.RunOptions) -> _PicLaunch:
    session.collective_agree(
        prepared.policy_signature, "pic-run-policy",
        "MPI ranks supplied different distributed PIC run policies")
    mapping = session.configure_devices(options.devices)
    topology = session.select_topology(
        prepared.deck.domain.nx, prepared.deck.domain.ny,
        options.decomposition, prepared.nghost)
    restart_metadata: dict[str, Any] | None = None
    if prepared.restart_path is None:
        assert prepared.fields is not None and prepared.species is not None
        session.start_pic(
            prepared.config, prepared.fields, None, prepared.species,
            transport=options.transport)
    else:
        native_restart_metadata = session.restart_pic(
            prepared.config, str(prepared.restart_path), prepared.deck.units,
            prepared.expected_species, transport=options.transport)
        restart_metadata = _collective_local(
            session, "pic-restart-metadata-convert",
            lambda: dict(native_restart_metadata))
    return _PicLaunch(mapping, topology, restart_metadata)


def _finish_pic_startup(
        session: Any, prepared: _PreparedPicRun,
        launch: _PicLaunch) -> tuple[
            tuple[list[dict[str, Any]], dict[str, list[Any]]] | None,
            _PicClock]:
    restored_history = None
    if launch.restart_metadata is not None:
        restored_history = _collective_local(
            session, "pic-restart-diagnostics-decode",
            lambda: _decode_pic_checkpoint_history(
                list(launch.restart_metadata.get("diagnostic_state", [])),
                prepared.deck))

    if prepared.restart_path is None and prepared.external is not None:
        evaluator, source, scales = prepared.external
        sampler = getattr(session, "pic_sample_external_fields", None)
        if not callable(sampler):
            raise RuntimeError(
                "the native PIC runtime does not expose tile-local "
                "external-field sampling")
        sampler(evaluator, source, *scales)

    step_done, sim_time, solver_time = _collective_local(
        session, "pic-start-state-convert",
        lambda: (
            0 if launch.restart_metadata is None else
            int(launch.restart_metadata["step"]),
            0.0 if launch.restart_metadata is None else
            float(launch.restart_metadata["time"]),
            float(prepared.units.time(
                0.0 if launch.restart_metadata is None else
                float(launch.restart_metadata["time"])))))
    time_limit = (None if prepared.deck.time.t_end_s is None else
                  float(prepared.deck.time.t_end_s))
    target_error = _restart_target_error(
        step_done, sim_time, prepared.deck.time.steps, time_limit)
    session.collective_require(
        not target_error, "pic-restart-target", target_error)
    native_cfl_value = session.pic_cfl_limit()
    native_cfl = _collective_local(
        session, "pic-initial-cfl-convert", lambda: float(native_cfl_value))
    session.collective_require(
        prepared.dt <= native_cfl, "pic-timestep-initial",
        "" if prepared.dt <= native_cfl else
        f"PIC timestep ({prepared.dt:.6e}) exceeds the distributed CFL limit "
        f"({native_cfl:.6e})")
    return restored_history, _PicClock(step_done, sim_time, solver_time)


def _print_pic_config(
        session: Any, prepared: _PreparedPicRun, launch: _PicLaunch,
        options: _distributed.RunOptions, print_config: bool) -> None:
    def print_resolved_config() -> None:
        if not print_config or int(session.rank) != 0:
            return
        deck = prepared.deck
        deck_source = (prepared.deck_path if prepared.deck_path is not None
                       else "<in-memory>")
        print(f"deck   : {deck_source}")
        print(f"grid   : {deck.domain.nx}x{deck.domain.ny}  "
              f"({deck.domain.lx_m}x{deck.domain.ly_m})  "
              f"origin=({deck.domain.origin_x_m}, "
              f"{deck.domain.origin_y_m})  nghost={prepared.nghost}")
        print(f"units  : {deck.units}")
        print(f"plane  : {deck.plane}")
        print(f"geometry: {deck.geometry}")
        print(f"species: {[item.name for item in deck.species]}")
        print(f"dt     : {prepared.dt_si:.6e} s    steps: {deck.time.steps}")
        native_pic = dict(session.telemetry.get("pic", {}))
        native_transport = dict(native_pic.get("transport", {}))
        resolved_transport = native_transport.get(
            "interprocess", options.transport)
        print(f"topology: {launch.topology['decomposition']}  "
              f"endpoints={len(launch.mapping)} "
              f"transport={resolved_transport}")

    _collective_local(session, "pic-print-config", print_resolved_config)


def _initialize_pic_history(
        session: Any, prepared: _PreparedPicRun,
        options: _distributed.RunOptions,
        restored_history: tuple[
            list[dict[str, Any]], dict[str, list[Any]]] | None,
        clock: _PicClock) -> _PicHistory:
    deck = prepared.deck
    snapshots: list[dict[str, Any]] = []
    sharded_snapshots: dict[int, list[dict[str, Any]]] = {}
    series: dict[str, list[Any]] = {"step": [], "time_s": []}
    for item in deck.species:
        series[f"alive_{item.name}"] = []
    if restored_history is None:
        return _PicHistory(snapshots, sharded_snapshots, series)

    canonical_snapshots, restored_series = restored_history
    expected_series = set(series)
    if set(restored_series) != expected_series:
        raise ValueError(
            "PIC checkpoint scalar history is incompatible with species")
    lengths = {len(values) for values in restored_series.values()}
    if len(lengths) > 1:
        raise ValueError("PIC checkpoint scalar histories have different lengths")
    if any(int(item["step"]) > clock.step
           or float(item["time_s"]) > clock.sim_time
           for item in canonical_snapshots):
        raise ValueError(
            "PIC checkpoint diagnostic history extends past the checkpoint")
    series = restored_series
    if options.diagnostics_layout == "gathered":
        current_state = _gather_state(
            session, "pic-restart-history-template")
        for item in canonical_snapshots:
            restored = _snapshot(
                {"fields": item["canonical_fields"],
                 "external_fields": current_state["external_fields"],
                 "species": []},
                deck, int(item["step"]), float(item["time_s"]),
                prepared.units, prepared.nghost)
            restored["canonical_fields"] = item["canonical_fields"]
            snapshots.append(restored)
    else:
        templates = _local_owned_shards(
            session, False, "pic-restart-history-templates")
        for template in templates:
            endpoint = int(template["endpoint"])
            history: list[dict[str, Any]] = []
            for item in canonical_snapshots:
                owned_fields: dict[str, Any] = {}
                for name in _PIC_FIELD_COMPONENTS:
                    component = template["fields"][name]
                    offset = tuple(map(int, component["offset"]))
                    shape = tuple(map(int, component["shape"]))
                    oy, ox = offset
                    ny, nx = shape
                    rows, columns, _, _ = _field_extents(
                        deck.domain.nx, deck.domain.ny, deck.geometry)[name]
                    canonical = np.asarray(
                        item["canonical_fields"][name],
                        dtype=np.float64).reshape(rows, columns)
                    owned_fields[name] = {
                        "offset": offset,
                        "shape": shape,
                        "values": np.ascontiguousarray(
                            canonical[oy:oy + ny, ox:ox + nx]),
                    }
                history.append({
                    "step": item["step"],
                    "time_s": item["time_s"],
                    "fields": owned_fields,
                })
            sharded_snapshots[endpoint] = history
    return _PicHistory(snapshots, sharded_snapshots, series)


def _record_pic_scalars(
        session: Any, deck: pic_io.PicDeck, series: dict[str, list[Any]],
        current_step: int, current_time: float) -> None:
    native_counts = session.pic_alive_counts()

    def convert_and_record() -> None:
        counts = np.asarray(native_counts, dtype=np.uint64)
        if counts.size != len(deck.species):
            raise RuntimeError(
                "distributed PIC alive-count result has the wrong size")
        series["step"].append(current_step)
        series["time_s"].append(current_time)
        for index, item in enumerate(deck.species):
            series[f"alive_{item.name}"].append(int(counts[index]))

    _collective_local(
        session, f"pic-scalar-record-step-{current_step}",
        convert_and_record)


def _advance_pic_clock(
        session: Any, prepared: _PreparedPicRun, clock: _PicClock) -> None:
    dt_step = prepared.dt
    clipped = False
    if prepared.t_end_internal is not None:
        remaining = prepared.t_end_internal - clock.solver_time
        dt_step = min(dt_step, remaining)
        clipped = dt_step == remaining
    if not math.isfinite(dt_step) or dt_step <= 0.0:
        raise RuntimeError("distributed PIC timestep cannot make progress")
    next_solver_time = (prepared.t_end_internal if clipped else
                        clock.solver_time + dt_step)
    if (next_solver_time is None or not math.isfinite(next_solver_time)
            or next_solver_time <= clock.solver_time):
        raise RuntimeError("distributed PIC time cannot make progress")
    session.pic_step(dt_step)

    def advance_clock() -> tuple[float, float, int]:
        next_internal = float(next_solver_time)
        next_sim_time = (
            float(prepared.deck.time.t_end_s) if clipped else
            float(prepared.units.time_to_si(next_internal)))
        return next_internal, next_sim_time, clock.step + 1

    clock.solver_time, clock.sim_time, clock.step = _collective_local(
        session, f"pic-clock-convert-step-{clock.step + 1}", advance_clock)


def _record_pic_snapshot(
        session: Any, prepared: _PreparedPicRun, history: _PicHistory,
        clock: _PicClock, state: dict[str, Any] | None,
        local_shards: list[dict[str, Any]] | None,
        layout: str) -> None:
    if layout == "gathered":
        assert state is not None

        def append_snapshot() -> None:
            item = _snapshot(
                state, prepared.deck, clock.step, clock.sim_time,
                prepared.units, prepared.nghost)
            item["canonical_fields"] = state["fields"]
            history.snapshots.append(item)

        _collective_local(
            session, f"pic-snapshot-assemble-step-{clock.step}",
            append_snapshot)
        return

    assert local_shards is not None

    def append_sharded_snapshots() -> None:
        for shard in local_shards:
            endpoint = int(shard["endpoint"])
            history.sharded_snapshots.setdefault(endpoint, []).append({
                "step": clock.step,
                "time_s": clock.sim_time,
                "fields": shard["fields"],
            })

    _collective_local(
        session, f"pic-sharded-snapshot-assemble-step-{clock.step}",
        append_sharded_snapshots)


def _log_pic_progress(
        session: Any, prepared: _PreparedPicRun, history: _PicHistory,
        clock: _PicClock, start_wall: float) -> None:
    _record_pic_scalars(
        session, prepared.deck, history.series,
        clock.step, clock.sim_time)

    def print_progress() -> None:
        if int(session.rank) != 0:
            return
        elapsed = time.perf_counter() - start_wall
        rate = clock.step / elapsed if elapsed > 0.0 else 0.0
        remaining = ((prepared.deck.time.steps - clock.step) / rate
                     if rate > 0.0 else float("nan"))
        alive = " ".join(
            f"{item.name}={history.series[f'alive_{item.name}'][-1]}"
            for item in prepared.deck.species)
        print(f"step {clock.step}/{prepared.deck.time.steps}  "
              f"t={clock.sim_time:.6e}s  rate={rate:.0f} step/s  "
              f"eta={remaining:.0f}s  alive: {alive}", flush=True)

    _collective_local(
        session, f"pic-progress-step-{clock.step}", print_progress)


def _write_periodic_pic_output(
        session: Any, prepared: _PreparedPicRun, launch: _PicLaunch,
        clock: _PicClock, state: dict[str, Any] | None,
        local_shards: list[dict[str, Any]] | None,
        layout: str) -> None:
    per_path = prepared.out_path.with_name(
        f"{prepared.out_path.stem}_{clock.step:010d}"
        f"{prepared.out_path.suffix}")
    if layout == "gathered":
        assert state is not None
        final = _collective_local(
            session, f"pic-periodic-snapshot-assemble-step-{clock.step}",
            lambda: _snapshot(
                state, prepared.deck, clock.step, clock.sim_time,
                prepared.units, prepared.nghost))
        from .cli import _flatten_for_npz
        _collective_local(
            session, f"pic-periodic-write-step-{clock.step}",
            lambda: _atomic_savez(
                per_path, _flatten_for_npz([], final, None))
            if int(session.rank) == 0 else None)
        return

    assert local_shards is not None
    _write_sharded(
        session, per_path, local_shards, {}, None, prepared.deck,
        prepared.units, prepared.nghost, clock.step, clock.sim_time,
        launch.mapping, launch.topology)


def _process_pic_step_outputs(
        session: Any, prepared: _PreparedPicRun, launch: _PicLaunch,
        options: _distributed.RunOptions, history: _PicHistory,
        clock: _PicClock, start_wall: float) -> None:
    on_snapshot = (prepared.deck.diagnostics.cadence > 0
                   and clock.step % prepared.deck.diagnostics.cadence == 0)
    on_write = (prepared.write_interval > 0
                and clock.step % prepared.write_interval == 0)
    state: dict[str, Any] | None = None
    local_shards: list[dict[str, Any]] | None = None
    if on_snapshot or on_write:
        if options.diagnostics_layout == "gathered":
            state = _gather_state(
                session, f"pic-state-convert-step-{clock.step}")
        else:
            local_shards = _local_owned_shards(
                session, on_write,
                f"pic-shard-convert-step-{clock.step}")
    if on_snapshot:
        _record_pic_snapshot(
            session, prepared, history, clock, state, local_shards,
            options.diagnostics_layout)
    if (prepared.log_interval > 0
            and clock.step % prepared.log_interval == 0):
        _log_pic_progress(
            session, prepared, history, clock, start_wall)
    if on_write:
        _write_periodic_pic_output(
            session, prepared, launch, clock, state, local_shards,
            options.diagnostics_layout)


def _write_pic_checkpoint(
        session: Any, prepared: _PreparedPicRun,
        options: _distributed.RunOptions, history: _PicHistory,
        clock: _PicClock) -> float:
    assert prepared.checkpoint_path is not None
    checkpoint_started = time.perf_counter()
    diagnostic_state = _collective_local(
        session, f"pic-checkpoint-history-step-{clock.step}",
        lambda: _pic_checkpoint_fragment(
            int(session.rank), options.diagnostics_layout, prepared.deck,
            history.snapshots, history.sharded_snapshots, history.series))
    session.pic_write_checkpoint(
        str(prepared.checkpoint_path), clock.step, clock.sim_time,
        prepared.deck.units, diagnostic_state)
    return time.perf_counter() - checkpoint_started


def _evolve_pic(
        session: Any, prepared: _PreparedPicRun, launch: _PicLaunch,
        options: _distributed.RunOptions, history: _PicHistory,
        clock: _PicClock, start_wall: float) -> tuple[float, float]:
    last_checkpoint_step: int | None = None
    checkpoint_seconds = 0.0
    while clock.step < prepared.deck.time.steps and (
            prepared.t_end_internal is None
            or clock.solver_time < prepared.t_end_internal):
        _advance_pic_clock(session, prepared, clock)
        _process_pic_step_outputs(
            session, prepared, launch, options, history, clock, start_wall)
        if (prepared.checkpoint_path is not None
                and options.checkpoint_every is not None
                and clock.step % options.checkpoint_every == 0):
            checkpoint_seconds += _write_pic_checkpoint(
                session, prepared, options, history, clock)
            last_checkpoint_step = clock.step

    if (prepared.log_interval == 0
            or clock.step % prepared.log_interval != 0):
        _record_pic_scalars(
            session, prepared.deck, history.series,
            clock.step, clock.sim_time)
        # A cadence checkpoint at the same final step preceded this
        # end-of-run scalar sample; replace it once more so restart sees
        # exactly the diagnostic history that the run will publish.
        if prepared.checkpoint_path is not None:
            last_checkpoint_step = None
    if (prepared.checkpoint_path is not None
            and last_checkpoint_step != clock.step):
        checkpoint_seconds += _write_pic_checkpoint(
            session, prepared, options, history, clock)
    return checkpoint_seconds, time.perf_counter()


def _write_final_pic_diagnostics(
        session: Any, prepared: _PreparedPicRun, launch: _PicLaunch,
        options: _distributed.RunOptions, history: _PicHistory,
        clock: _PicClock) -> Path:
    if options.diagnostics_layout == "gathered":
        final_state = _gather_state(session, "pic-final-state-convert")

        def prepare_gathered_output() -> tuple[
                dict[str, Any], list[dict[str, Any]]]:
            final_snapshot = _snapshot(
                final_state, prepared.deck, clock.step, clock.sim_time,
                prepared.units, prepared.nghost)
            clean = [
                {key: value for key, value in item.items()
                 if key != "canonical_fields"}
                for item in history.snapshots
            ]
            return final_snapshot, clean

        final, clean_snapshots = _collective_local(
            session, "pic-final-snapshot-assemble",
            prepare_gathered_output)
        from .cli import _flatten_for_npz
        _collective_local(
            session, "pic-diagnostics-gathered-write",
            lambda: _atomic_savez(
                prepared.out_path,
                _flatten_for_npz(
                    clean_snapshots, final, history.series))
            if int(session.rank) == 0 else None)
        return prepared.out_path

    final_shards = _local_owned_shards(
        session, True, "pic-final-shard-convert")
    return _write_sharded(
        session, prepared.out_path, final_shards,
        history.sharded_snapshots, history.series, prepared.deck,
        prepared.units, prepared.nghost, clock.step, clock.sim_time,
        launch.mapping, launch.topology)


def _collect_pic_telemetry(
        session: Any, launch: _PicLaunch, options: _distributed.RunOptions,
        phase_times: _RunPhaseTimes) -> dict[str, Any]:
    native_final_counts = session.pic_alive_counts()
    final_counts = _collective_local(
        session, "pic-final-counts-convert",
        lambda: np.asarray(native_final_counts, dtype=np.uint64).tolist())
    native_final_energies = session.pic_kinetic_energies()
    final_energies = _collective_local(
        session, "pic-final-energies-convert",
        lambda: np.asarray(
            native_final_energies, dtype=np.float64).tolist())
    native_final_em_energy = session.pic_total_em_energy()
    final_em_energy = _collective_local(
        session, "pic-final-em-energy-convert",
        lambda: float(native_final_em_energy))
    native_final_gauss_residual = session.pic_gauss_residual()
    final_gauss_residual = _collective_local(
        session, "pic-final-gauss-convert",
        lambda: float(native_final_gauss_residual))
    telemetry = _collective_local(
        session, "pic-telemetry", lambda: dict(session.telemetry))

    def finalize_telemetry() -> None:
        _finalize_run_telemetry(
            telemetry, physics="pic",
            requested_transport=options.transport,
            decomposition=launch.topology["decomposition"],
            phase_times=phase_times)
        telemetry["final_alive_counts"] = final_counts
        telemetry["final_kinetic_energies"] = final_energies
        telemetry["final_em_energy"] = final_em_energy
        telemetry["final_gauss_residual"] = final_gauss_residual

    _collective_local(
        session, "pic-telemetry-assemble", finalize_telemetry)
    return telemetry


def run(input_deck: Any, options: _distributed.RunOptions, *,
        seed: int | None = None, steps_override: int | None = None,
        verbose: bool = False, print_config: bool = False,
        log_every: int = 0, write_every: int = 0) -> dict[str, Any]:
    """Execute one shared distributed PIC simulation."""

    session = _distributed.RuntimeSession()
    started = False
    start_wall = time.perf_counter()
    try:
        prepared = _collective_local(
            session, "pic-python-prepare",
            lambda: _prepare_pic_run(
                input_deck, options, seed=seed,
                steps_override=steps_override, verbose=verbose,
                print_config=print_config, log_every=log_every,
                write_every=write_every))
        prepare_done = time.perf_counter()
        launch = _launch_pic_runtime(session, prepared, options)
        started = True
        restored_history, clock = _finish_pic_startup(
            session, prepared, launch)
        _print_pic_config(
            session, prepared, launch, options, print_config)
        history = _initialize_pic_history(
            session, prepared, options, restored_history, clock)
        setup_done = time.perf_counter()
        checkpoint_seconds, evolution_done = _evolve_pic(
            session, prepared, launch, options, history, clock,
            start_wall)
        diagnostics_path = _write_final_pic_diagnostics(
            session, prepared, launch, options, history, clock)
        diagnostics_done = time.perf_counter()
        telemetry = _collect_pic_telemetry(
            session, launch, options,
            _RunPhaseTimes(
                start_wall, prepare_done, setup_done, evolution_done,
                diagnostics_done, checkpoint_seconds))
        _collective_local(
            session, "pic-print-result",
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
        if started or not session.closed:
            session.close()
