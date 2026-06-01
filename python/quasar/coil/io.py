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

Note: the output block intentionally differs from the PIC deck. The coil deck
writes a single field snapshot, so it uses ``output.path``; the PIC deck writes a
time series and groups it under ``diagnostics.output_path`` alongside cadence /
per-species options. The two schemas are kept distinct rather than forced into a
shared key because their output semantics differ.

See ``examples/single_loop/input.yaml`` (added in Phase 2.F) for a worked
example.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, Union

import yaml

from .._core import Vec3
from .._deck import require as _require, triple as _triple
from .._deck import validate_evaluator_type as _validate_evaluator_type
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
    return Vec3(*_triple(xyz))


def _unit(v: Vec3, context: str) -> Vec3:
    norm = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if norm == 0.0:
        raise ValueError(f"{context}: axis vector must be non-zero")
    return Vec3(v.x / norm, v.y / norm, v.z / norm)


# ---------------------------------------------------------------------------
# geometry dispatch
# ---------------------------------------------------------------------------


def _build_geometry(spec: dict, current_A: float, name: str) -> Filament:
    gt = _require(spec, "type", f"conductor {name!r}.geometry")
    # The error context is the geometry type already in hand, so bind it once and
    # require fields through `req` instead of repeating the literal on every field.
    def req(key):
        return _require(spec, key, gt)

    if gt == "circular_loop":
        return circular_loop(
            center=_vec3(req("center_xyz")),
            axis=_vec3(req("axis_xyz")),
            radius_m=float(req("radius_m")),
            n_segments=int(req("n_segments")),
            current_A=current_A,
            name=name,
        )
    if gt == "helix":
        return helix(
            center=_vec3(req("center_xyz")),
            axis=_vec3(req("axis_xyz")),
            radius_m=float(req("radius_m")),
            pitch_m=float(req("pitch_m")),
            n_turns=int(req("n_turns")),
            n_segments_per_turn=int(req("n_segments_per_turn")),
            current_A=current_A,
            name=name,
        )
    if gt == "solenoid":
        return solenoid(
            center=_vec3(req("center_xyz")),
            axis=_vec3(req("axis_xyz")),
            radius_m=float(req("radius_m")),
            length_m=float(req("length_m")),
            n_turns=int(req("n_turns")),
            n_segments_per_turn=int(req("n_segments_per_turn")),
            current_A=current_A,
            name=name,
        )
    if gt == "racetrack":
        return racetrack(
            center=_vec3(req("center_xyz")),
            axis=_vec3(req("axis_xyz")),
            straight_length_m=float(req("straight_length_m")),
            arc_radius_m=float(req("arc_radius_m")),
            n_arc_segments=int(req("n_arc_segments")),
            current_A=current_A,
            name=name,
        )
    if gt == "polygon":
        return polygon(
            center=_vec3(req("center_xyz")),
            axis=_vec3(req("axis_xyz")),
            circumradius_m=float(req("circumradius_m")),
            n_sides=int(req("n_sides")),
            current_A=current_A,
            name=name,
        )
    if gt == "polyline":
        pts = [_vec3(p) for p in req("points_xyz_m")]
        return generic_polyline(points=pts, current_A=current_A, name=name)
    raise ValueError(f"conductor {name!r}.geometry.type {gt!r} is not recognized")


def build_conductor_system(conductors: Sequence[dict]) -> ConductorSystem:
    """Build a ConductorSystem from a list of conductor dicts.

    Shared by the coil deck parser and the PIC external-field loader so the
    conductor schema (name / current_A / geometry) has a single home.
    """
    cs = ConductorSystem()
    for c in conductors:
        name = str(_require(c, "name", "conductor"))
        current_A = float(_require(c, "current_A", f"conductor {name!r}"))
        geom_spec = _require(c, "geometry", f"conductor {name!r}")
        cs.add(_build_geometry(geom_spec, current_A, name))
    return cs


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


# Upper bound on the number of observation points materialized on the host (and
# uploaded to the device). Guards against a typo'd resolution requesting an
# absurd buffer; raise it if a real workload needs more.
MAX_OBSERVATION_POINTS = 1 << 26  # ~67M points


def _check_point_count(kind: str, n: int) -> None:
    if n < 0 or n > MAX_OBSERVATION_POINTS:
        raise ValueError(
            f"observation.{kind}: point count {n} exceeds the limit "
            f"{MAX_OBSERVATION_POINTS}")


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
        _check_point_count("grid", nx * ny * nz)
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
        u = _unit(_vec3(_require(spec, "u_axis_xyz", "observation.plane")),
                  "observation.plane.u_axis_xyz")
        v = _unit(_vec3(_require(spec, "v_axis_xyz", "observation.plane")),
                  "observation.plane.v_axis_xyz")
        u_extent = float(_require(spec, "u_extent_m", "observation.plane"))
        v_extent = float(_require(spec, "v_extent_m", "observation.plane"))
        nu = int(_require(spec, "nu", "observation.plane"))
        nv = int(_require(spec, "nv", "observation.plane"))
        _check_point_count("plane", nu * nv)
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
        _check_point_count("line", n)
        lp = LineProbe()
        lp.start, lp.end, lp.n_points = start, end, n
        return _ObservationResult(points=lp.to_point_cloud(),
                                   dims=[n], kind="line", detail=lp)

    if ot == "points":
        raw_pts = _require(spec, "points_xyz_m", "observation.points")
        _check_point_count("points", len(raw_pts))
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
    # Registry name of the field evaluator; coil design uses Biot-Savart.
    evaluator_type: str = "biot_savart"

    def validate(self) -> None:
        """Validate the parsed deck. Mirrors :meth:`PicDeck.validate` so the two
        loaders share one validation convention (most per-field checks happen
        inline during parsing; this is the consolidated cross-field pass)."""
        if self.units != "SI":
            raise ValueError(f"only units: SI is supported, got {self.units!r}")
        if self.conductors.empty():
            raise ValueError("deck.conductors must be non-empty")
        _validate_evaluator_type(self.evaluator_type, "evaluator.type")


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

    cs = build_conductor_system(raw_conductors)

    obs = _build_observation(_require(data, "observation", "deck"))
    out = _parse_output(_require(data, "output", "deck"))

    evaluator_type = str(data.get("evaluator", {}).get("type", "biot_savart"))

    deck = CoilDeck(units=units, conductors=cs, observation=obs,
                    output=out, raw=data, evaluator_type=evaluator_type)
    deck.validate()
    return deck


def load(path: Union[str, Path]) -> CoilDeck:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return parse(data)
