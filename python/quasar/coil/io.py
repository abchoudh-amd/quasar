"""YAML/dict input-deck schema for the Quasar coil-design workflow.

The schema is implemented with dataclasses + manual coercion (no pydantic
dependency); the validation surface is intentionally small so it can be
swapped for pydantic later without touching call sites in
``quasar.coil.cli`` or downstream postprocessing.

Top-level deck structure::

    units: SI                    # only SI is recognized
    conductors:
      - name: <str>
        current_A: <float>
        geometry: {type: <enum>, <params...>}
    observation:
      type: grid | plane | line | points
      <params...>
    output:
      format: npz
      path:   <relative or absolute path>
      fields: [B_xyz, B_magnitude]

See ``examples/single_loop/input.yaml`` (added in Phase 2.F) for a worked
example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence, Union

import yaml

from .._core import Vec3
from . import (
    ConductorSystem,
    Filament,
    LineProbe,
    ObservationGrid,
    PlaneSlice,
    PointCloud,
    circular_loop,
    generic_polyline,
    helix,
    polygon,
    racetrack,
    solenoid,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _vec3(xyz: Sequence[float]) -> Vec3:
    if len(xyz) != 3:
        raise ValueError(f"expected 3-element xyz triple, got {xyz!r}")
    return Vec3(float(xyz[0]), float(xyz[1]), float(xyz[2]))


def _require(d: dict, key: str, context: str) -> Any:
    if key not in d:
        raise ValueError(f"{context}: missing required field {key!r}")
    return d[key]


# ---------------------------------------------------------------------------
# geometry dispatch
# ---------------------------------------------------------------------------


def _build_geometry(spec: dict, current_A: float, name: str) -> Filament:
    gt = _require(spec, "type", f"conductor {name!r}.geometry")
    if gt == "circular_loop":
        return circular_loop(
            center=_vec3(_require(spec, "center_xyz", "circular_loop")),
            axis=_vec3(_require(spec, "axis_xyz", "circular_loop")),
            radius_m=float(_require(spec, "radius_m", "circular_loop")),
            n_segments=int(_require(spec, "n_segments", "circular_loop")),
            current_A=current_A,
            name=name,
        )
    if gt == "helix":
        return helix(
            center=_vec3(_require(spec, "center_xyz", "helix")),
            axis=_vec3(_require(spec, "axis_xyz", "helix")),
            radius_m=float(_require(spec, "radius_m", "helix")),
            pitch_m=float(_require(spec, "pitch_m", "helix")),
            n_turns=int(_require(spec, "n_turns", "helix")),
            n_segments_per_turn=int(_require(spec, "n_segments_per_turn", "helix")),
            current_A=current_A,
            name=name,
        )
    if gt == "solenoid":
        return solenoid(
            center=_vec3(_require(spec, "center_xyz", "solenoid")),
            axis=_vec3(_require(spec, "axis_xyz", "solenoid")),
            radius_m=float(_require(spec, "radius_m", "solenoid")),
            length_m=float(_require(spec, "length_m", "solenoid")),
            n_turns=int(_require(spec, "n_turns", "solenoid")),
            n_segments_per_turn=int(_require(spec, "n_segments_per_turn", "solenoid")),
            current_A=current_A,
            name=name,
        )
    if gt == "racetrack":
        return racetrack(
            center=_vec3(_require(spec, "center_xyz", "racetrack")),
            axis=_vec3(_require(spec, "axis_xyz", "racetrack")),
            straight_length_m=float(_require(spec, "straight_length_m", "racetrack")),
            arc_radius_m=float(_require(spec, "arc_radius_m", "racetrack")),
            n_arc_segments=int(_require(spec, "n_arc_segments", "racetrack")),
            current_A=current_A,
            name=name,
        )
    if gt == "polygon":
        return polygon(
            center=_vec3(_require(spec, "center_xyz", "polygon")),
            axis=_vec3(_require(spec, "axis_xyz", "polygon")),
            circumradius_m=float(_require(spec, "circumradius_m", "polygon")),
            n_sides=int(_require(spec, "n_sides", "polygon")),
            current_A=current_A,
            name=name,
        )
    if gt == "polyline":
        pts = [_vec3(p) for p in _require(spec, "points_xyz_m", "polyline")]
        return generic_polyline(points=pts, current_A=current_A, name=name)
    raise ValueError(f"conductor {name!r}.geometry.type {gt!r} is not recognized")


# ---------------------------------------------------------------------------
# observation dispatch
# ---------------------------------------------------------------------------


@dataclass
class _ObservationResult:
    """Internal: handles for the materialized observation set."""

    points: PointCloud
    dims: list[int]
    kind: str
    detail: Any  # the source object (ObservationGrid, PlaneSlice, LineProbe, ...)


def _build_observation(spec: dict) -> _ObservationResult:
    ot = _require(spec, "type", "observation")

    if ot == "grid":
        bounds = _require(spec, "bounds_m", "observation.grid")
        res = _require(spec, "resolution", "observation.grid")
        if len(bounds) != 3 or len(res) != 3:
            raise ValueError("observation.grid: bounds_m and resolution must be length-3")
        g = ObservationGrid()
        g.origin = Vec3(float(bounds[0][0]), float(bounds[1][0]), float(bounds[2][0]))
        nx, ny, nz = int(res[0]), int(res[1]), int(res[2])
        g.spacing = Vec3(
            (float(bounds[0][1]) - float(bounds[0][0])) / max(1, nx - 1),
            (float(bounds[1][1]) - float(bounds[1][0])) / max(1, ny - 1),
            (float(bounds[2][1]) - float(bounds[2][0])) / max(1, nz - 1),
        )
        g.dims = [nx, ny, nz]
        return _ObservationResult(points=g.to_point_cloud(),
                                   dims=[nx, ny, nz], kind="grid", detail=g)

    if ot == "plane":
        origin = _vec3(_require(spec, "origin_xyz", "observation.plane"))
        u = _vec3(_require(spec, "u_axis_xyz", "observation.plane"))
        v = _vec3(_require(spec, "v_axis_xyz", "observation.plane"))
        u_extent = float(_require(spec, "u_extent_m", "observation.plane"))
        v_extent = float(_require(spec, "v_extent_m", "observation.plane"))
        nu = int(_require(spec, "nu", "observation.plane"))
        nv = int(_require(spec, "nv", "observation.plane"))
        s = PlaneSlice()
        s.origin = origin
        s.u_step = Vec3(u.x * u_extent / max(1, nu - 1),
                        u.y * u_extent / max(1, nu - 1),
                        u.z * u_extent / max(1, nu - 1))
        s.v_step = Vec3(v.x * v_extent / max(1, nv - 1),
                        v.y * v_extent / max(1, nv - 1),
                        v.z * v_extent / max(1, nv - 1))
        s.nu, s.nv = nu, nv
        return _ObservationResult(points=s.to_point_cloud(),
                                   dims=[nu, nv], kind="plane", detail=s)

    if ot == "line":
        start = _vec3(_require(spec, "start_xyz", "observation.line"))
        end = _vec3(_require(spec, "end_xyz", "observation.line"))
        n = int(_require(spec, "n_points", "observation.line"))
        lp = LineProbe()
        lp.start, lp.end, lp.n_points = start, end, n
        return _ObservationResult(points=lp.to_point_cloud(),
                                   dims=[n], kind="line", detail=lp)

    if ot == "points":
        raw_pts = _require(spec, "points_xyz_m", "observation.points")
        pc = PointCloud()
        for p in raw_pts:
            pc.add(_vec3(p))
        return _ObservationResult(points=pc, dims=[len(raw_pts)],
                                   kind="points", detail=None)

    raise ValueError(f"observation.type {ot!r} is not recognized")


# ---------------------------------------------------------------------------
# top-level deck
# ---------------------------------------------------------------------------


@dataclass
class OutputSpec:
    format: str
    path: str
    fields: list[str] = field(default_factory=lambda: ["B_xyz"])


@dataclass
class CoilDeck:
    units: str
    conductors: ConductorSystem
    observation: _ObservationResult
    output: OutputSpec
    raw: dict


def _parse_output(spec: dict) -> OutputSpec:
    fmt = _require(spec, "format", "output")
    if fmt != "npz":
        raise ValueError(
            f"output.format {fmt!r} is not supported in this phase "
            "(only 'npz' is implemented; VTK lands in a later phase)"
        )
    return OutputSpec(
        format=fmt,
        path=str(_require(spec, "path", "output")),
        fields=list(spec.get("fields", ["B_xyz"])),
    )


def parse(data: dict) -> CoilDeck:
    units = data.get("units", "SI")
    if units != "SI":
        raise ValueError(f"only units: SI is supported, got {units!r}")

    raw_conductors = _require(data, "conductors", "deck")
    if not raw_conductors:
        raise ValueError("deck.conductors must be non-empty")

    cs = ConductorSystem()
    for c in raw_conductors:
        name = str(_require(c, "name", "conductor"))
        current_A = float(_require(c, "current_A", f"conductor {name!r}"))
        geom_spec = _require(c, "geometry", f"conductor {name!r}")
        cs.add(_build_geometry(geom_spec, current_A, name))

    obs = _build_observation(_require(data, "observation", "deck"))
    out = _parse_output(_require(data, "output", "deck"))

    return CoilDeck(units=units, conductors=cs, observation=obs,
                    output=out, raw=data)


def load(path: Union[str, Path]) -> CoilDeck:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return parse(data)
