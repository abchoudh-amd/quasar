"""YAML/dict input-deck schema for the Quasar coil-design workflow.

The schema is implemented with dataclasses and explicit coercion, keeping the
runtime dependency surface small for ``quasar.coil.cli`` and downstream
postprocessing.

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
      path:   <path relative to the deck>
      fields: [B_xyz, B_magnitude]

Note: the output block intentionally differs from the PIC deck. The coil deck
writes a single field snapshot, so it uses ``output.path``; the PIC deck writes a
time series and groups it under ``diagnostics.output_path`` alongside cadence /
per-species options. The two schemas are kept distinct rather than forced into a
shared key because their output semantics differ.

See ``examples/single_loop/input.yaml`` for a worked example.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, Union

from .._core import Vec3, magnetostatics as _magnetostatics
from .._field_grid import load_file_grid_npz
from .._deck import as_integer as _as_integer
from .._deck import as_finite as _as_finite
from .._deck import flat_evaluator_params as _flat_evaluator_params
from .._deck import load_yaml as _load_yaml
from .._deck import require as _require, triple as _triple
from .._deck import unique_alias as _unique_alias
from .._deck import validate_evaluator_type as _validate_evaluator_type
from . import (
    ConductorSystem,
    DevicePointCloud,
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


def _mapping(value: Any, context: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _reject_unknown(spec: dict, allowed: set[str], context: str) -> None:
    unknown = sorted(repr(key) for key in spec if key not in allowed)
    if unknown:
        raise ValueError(f"{context} contains unsupported key(s) {unknown}")


def _unit(v: Vec3, context: str) -> Vec3:
    if not all(math.isfinite(x) for x in (v.x, v.y, v.z)):
        raise ValueError(f"{context}: axis vector must be non-zero and finite")
    scale = max(abs(v.x), abs(v.y), abs(v.z))
    if scale == 0.0:
        raise ValueError(f"{context}: axis vector must be non-zero and finite")
    scaled = (v.x / scale, v.y / scale, v.z / scale)
    norm = math.hypot(*scaled)
    return Vec3(scaled[0] / norm, scaled[1] / norm, scaled[2] / norm)


# ---------------------------------------------------------------------------
# geometry dispatch
# ---------------------------------------------------------------------------


def _build_geometry(spec: dict, current_A: float, name: str) -> Filament:
    spec = _mapping(spec, f"conductor {name!r}.geometry")
    gt = _require(spec, "type", f"conductor {name!r}.geometry")
    if not isinstance(gt, str):
        raise ValueError(f"conductor {name!r}.geometry.type must be a string")
    allowed_by_type = {
        "circular_loop": {
            "type", "center_xyz", "axis_xyz", "radius_m", "n_segments"},
        "helix": {
            "type", "center_xyz", "axis_xyz", "radius_m", "pitch_m",
            "n_turns", "n_segments_per_turn"},
        "solenoid": {
            "type", "center_xyz", "axis_xyz", "radius_m", "length_m",
            "n_turns", "n_segments_per_turn"},
        "racetrack": {
            "type", "center_xyz", "axis_xyz", "straight_length_m",
            "arc_radius_m", "n_arc_segments"},
        "polygon": {
            "type", "center_xyz", "axis_xyz", "circumradius_m", "n_sides"},
        "polyline": {"type", "points_xyz_m"},
    }
    if gt not in allowed_by_type:
        raise ValueError(
            f"conductor {name!r}.geometry.type {gt!r} is not recognized")
    _reject_unknown(
        spec, allowed_by_type[gt], f"conductor {name!r}.geometry")
    # The error context is the geometry type already in hand, so bind it once and
    # require fields through `req` instead of repeating the literal on every field.
    def req(key):
        return _require(spec, key, gt)

    if gt == "circular_loop":
        return circular_loop(
            center=_vec3(req("center_xyz")),
            axis=_vec3(req("axis_xyz")),
            radius_m=_finite_scalar(
                req("radius_m"),
                f"conductor {name!r}.geometry.radius_m"),
            n_segments=_as_integer(req("n_segments"),
                                   f"conductor {name!r}.geometry.n_segments"),
            current_A=current_A,
            name=name,
        )
    if gt == "helix":
        return helix(
            center=_vec3(req("center_xyz")),
            axis=_vec3(req("axis_xyz")),
            radius_m=_finite_scalar(
                req("radius_m"),
                f"conductor {name!r}.geometry.radius_m"),
            pitch_m=_finite_scalar(
                req("pitch_m"),
                f"conductor {name!r}.geometry.pitch_m"),
            n_turns=_as_integer(req("n_turns"),
                                f"conductor {name!r}.geometry.n_turns"),
            n_segments_per_turn=_as_integer(
                req("n_segments_per_turn"),
                f"conductor {name!r}.geometry.n_segments_per_turn"),
            current_A=current_A,
            name=name,
        )
    if gt == "solenoid":
        return solenoid(
            center=_vec3(req("center_xyz")),
            axis=_vec3(req("axis_xyz")),
            radius_m=_finite_scalar(
                req("radius_m"),
                f"conductor {name!r}.geometry.radius_m"),
            length_m=_finite_scalar(
                req("length_m"),
                f"conductor {name!r}.geometry.length_m"),
            n_turns=_as_integer(req("n_turns"),
                                f"conductor {name!r}.geometry.n_turns"),
            n_segments_per_turn=_as_integer(
                req("n_segments_per_turn"),
                f"conductor {name!r}.geometry.n_segments_per_turn"),
            current_A=current_A,
            name=name,
        )
    if gt == "racetrack":
        return racetrack(
            center=_vec3(req("center_xyz")),
            axis=_vec3(req("axis_xyz")),
            straight_length_m=_finite_scalar(
                req("straight_length_m"),
                f"conductor {name!r}.geometry.straight_length_m"),
            arc_radius_m=_finite_scalar(
                req("arc_radius_m"),
                f"conductor {name!r}.geometry.arc_radius_m"),
            n_arc_segments=_as_integer(
                req("n_arc_segments"),
                f"conductor {name!r}.geometry.n_arc_segments"),
            current_A=current_A,
            name=name,
        )
    if gt == "polygon":
        return polygon(
            center=_vec3(req("center_xyz")),
            axis=_vec3(req("axis_xyz")),
            circumradius_m=_finite_scalar(
                req("circumradius_m"),
                f"conductor {name!r}.geometry.circumradius_m"),
            n_sides=_as_integer(req("n_sides"),
                                f"conductor {name!r}.geometry.n_sides"),
            current_A=current_A,
            name=name,
        )
    if gt == "polyline":
        pts = [_vec3(p) for p in req("points_xyz_m")]
        return generic_polyline(points=pts, current_A=current_A, name=name)
    raise AssertionError("unreachable geometry dispatch")


def build_conductor_system(conductors: Sequence[dict]) -> ConductorSystem:
    """Build a ConductorSystem from a list of conductor dicts.

    Shared by the coil deck parser and the PIC external-field loader so the
    conductor schema (name / current_A / geometry) has a single home.
    """
    if not isinstance(conductors, (list, tuple)):
        raise ValueError("deck.conductors must be a list")
    cs = ConductorSystem()
    for index, raw_conductor in enumerate(conductors):
        c = _mapping(raw_conductor, f"deck.conductors[{index}]")
        _reject_unknown(
            c, {"name", "current_A", "geometry"},
            f"deck.conductors[{index}]")
        raw_name = _require(c, "name", f"deck.conductors[{index}]")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"deck.conductors[{index}].name must be non-empty")
        name = raw_name
        current_A = _finite_scalar(
            _require(c, "current_A", f"conductor {name!r}"),
            f"conductor {name!r}.current_A")
        geom_spec = _require(c, "geometry", f"conductor {name!r}")
        cs.add(_build_geometry(geom_spec, current_A, name))
    return cs


# ---------------------------------------------------------------------------
# observation dispatch
# ---------------------------------------------------------------------------


@dataclass
class _ObservationResult:
    """Internal: handles for the materialized observation set."""

    # Device-resident. The structured kinds expand their description with a
    # kernel; the explicit ``points`` kind uploads the coordinates the deck
    # literally listed, which are data rather than the result of a calculation.
    # Either way the evaluator receives points that are already on the device.
    points: DevicePointCloud
    dims: list[int]
    kind: str
    detail: Any  # the source object (ObservationGrid, PlaneSlice, LineProbe, ...)


# Upper bound on the number of observation points materialized on the host (and
# uploaded to the device). Guards against a typo'd resolution requesting an
# absurd buffer; raise it if a real workload needs more.
MAX_OBSERVATION_POINTS = 1 << 26  # ~67M points


def _check_point_count(kind: str, n: int) -> None:
    if n <= 0 or n > MAX_OBSERVATION_POINTS:
        raise ValueError(
            f"observation.{kind}: point count must be in "
            f"[1, {MAX_OBSERVATION_POINTS}], got {n}")


def _finite_scalar(value: Any, context: str) -> float:
    return _as_finite(value, context)


def _build_observation(spec: dict) -> _ObservationResult:
    spec = _mapping(spec, "observation")
    ot = _require(spec, "type", "observation")

    if ot == "grid":
        _reject_unknown(
            spec, {"type", "bounds_m", "resolution"}, "observation.grid")
        bounds = _require(spec, "bounds_m", "observation.grid")
        res = _require(spec, "resolution", "observation.grid")
        if (not isinstance(bounds, (list, tuple))
                or not isinstance(res, (list, tuple))
                or len(bounds) != 3 or len(res) != 3):
            raise ValueError("observation.grid: bounds_m and resolution must be length-3")
        parsed_bounds: list[tuple[float, float]] = []
        for axis, pair in enumerate(bounds):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    f"observation.grid.bounds_m[{axis}] must be [lower, upper]")
            parsed_bounds.append((
                _finite_scalar(pair[0], f"observation.grid.bounds_m[{axis}][0]"),
                _finite_scalar(pair[1], f"observation.grid.bounds_m[{axis}][1]")))
        nx, ny, nz = (
            _as_integer(res[0], "observation.grid.resolution[0]"),
            _as_integer(res[1], "observation.grid.resolution[1]"),
            _as_integer(res[2], "observation.grid.resolution[2]"))
        _check_point_count("grid", nx * ny * nz)
        for axis, ((lower, upper), count) in enumerate(
                zip(parsed_bounds, (nx, ny, nz))):
            if upper < lower or (count > 1 and upper == lower):
                relation = "greater than" if count > 1 else "at least"
                raise ValueError(
                    f"observation.grid.bounds_m[{axis}] upper bound must be "
                    f"{relation} its lower bound for resolution {count}")
        g = ObservationGrid()
        g.origin = Vec3(*(pair[0] for pair in parsed_bounds))
        def interval_spacing(lower: float, upper: float, count: int) -> float:
            if count == 1:
                return 0.0
            denominator = count - 1
            span = upper - lower
            if math.isfinite(span):
                return span / denominator
            # The interval may be representable even when forming its full span
            # first overflows (for example [-DBL_MAX, DBL_MAX] with 3 points).
            return upper / denominator - lower / denominator

        g.spacing = Vec3(*(
            interval_spacing(lower, upper, count)
            for (lower, upper), count in zip(
                parsed_bounds, (nx, ny, nz))))
        g.dims = [nx, ny, nz]
        return _ObservationResult(points=g.to_device_point_cloud(),
                                   dims=[nx, ny, nz], kind="grid", detail=g)

    if ot == "plane":
        _reject_unknown(
            spec,
            {"type", "origin_xyz", "u_axis_xyz", "v_axis_xyz",
             "u_extent_m", "v_extent_m", "nu", "nv"},
            "observation.plane")
        origin = _vec3(_require(spec, "origin_xyz", "observation.plane"))
        u = _unit(_vec3(_require(spec, "u_axis_xyz", "observation.plane")),
                  "observation.plane.u_axis_xyz")
        v = _unit(_vec3(_require(spec, "v_axis_xyz", "observation.plane")),
                  "observation.plane.v_axis_xyz")
        cross = (u.y * v.z - u.z * v.y,
                 u.z * v.x - u.x * v.z,
                 u.x * v.y - u.y * v.x)
        if math.hypot(*cross) <= 64.0 * math.ulp(1.0):
            raise ValueError(
                "observation.plane u_axis_xyz and v_axis_xyz must be linearly "
                "independent")
        u_extent = _finite_scalar(
            _require(spec, "u_extent_m", "observation.plane"),
            "observation.plane.u_extent_m")
        v_extent = _finite_scalar(
            _require(spec, "v_extent_m", "observation.plane"),
            "observation.plane.v_extent_m")
        if u_extent <= 0.0 or v_extent <= 0.0:
            raise ValueError("observation.plane extents must be positive")
        nu = _as_integer(_require(spec, "nu", "observation.plane"),
                         "observation.plane.nu")
        nv = _as_integer(_require(spec, "nv", "observation.plane"),
                         "observation.plane.nv")
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
        return _ObservationResult(points=s.to_device_point_cloud(),
                                   dims=[nu, nv], kind="plane", detail=s)

    if ot == "line":
        _reject_unknown(
            spec, {"type", "start_xyz", "end_xyz", "n_points"},
            "observation.line")
        start = _vec3(_require(spec, "start_xyz", "observation.line"))
        end = _vec3(_require(spec, "end_xyz", "observation.line"))
        n = _as_integer(_require(spec, "n_points", "observation.line"),
                        "observation.line.n_points")
        _check_point_count("line", n)
        lp = LineProbe()
        lp.start, lp.end, lp.n_points = start, end, n
        return _ObservationResult(points=lp.to_device_point_cloud(),
                                   dims=[n], kind="line", detail=lp)

    if ot == "points":
        _reject_unknown(
            spec, {"type", "points_xyz_m"}, "observation.points")
        raw_pts = _require(spec, "points_xyz_m", "observation.points")
        if not isinstance(raw_pts, (list, tuple)):
            raise ValueError("observation.points.points_xyz_m must be a list")
        _check_point_count("points", len(raw_pts))
        pc = PointCloud()
        for p in raw_pts:
            pc.add(_vec3(p))
        return _ObservationResult(points=DevicePointCloud.upload(pc),
                                   dims=[len(raw_pts)],
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
    # Flat Vec3/Mat3 parameter map consumed by IFieldEvaluator.configure().
    evaluator_params: dict[str, list[float]] = field(default_factory=dict)
    evaluator_file: str | None = None

    def validate(self) -> None:
        """Validate the parsed deck. Mirrors :meth:`PicDeck.validate` so the two
        loaders share one validation convention (most per-field checks happen
        inline during parsing; this is the consolidated cross-field pass)."""
        if self.units != "SI":
            raise ValueError(f"only units: SI is supported, got {self.units!r}")
        if self.evaluator_type == "biot_savart" and self.conductors.empty():
            raise ValueError("deck.conductors must be non-empty for biot_savart")
        _validate_evaluator_type(self.evaluator_type, "evaluator.type")
        fields = set(self.output.fields)
        if fields & {"B_xyz_grid", "A_xyz_grid"} and self.observation.kind != "grid":
            raise ValueError("*_grid output fields require observation.type == 'grid'")
        if (fields & {"A_xyz", "A_xyz_grid"}
                and not _magnetostatics.field_evaluator_provides_vector_potential(
                    self.evaluator_type)):
            raise ValueError(
                f"evaluator.type {self.evaluator_type!r} does not provide "
                "magnetic vector potential A")


def _parse_output(spec: dict) -> OutputSpec:
    spec = _mapping(spec, "output")
    _reject_unknown(spec, {"format", "path", "fields"}, "output")
    fmt = _require(spec, "format", "output")
    if fmt != "npz":
        raise ValueError(
            f"output.format {fmt!r} is not supported; only 'npz' is implemented"
        )
    raw_fields = spec.get("fields", ["B_xyz"])
    if not isinstance(raw_fields, (list, tuple)) or not raw_fields:
        raise ValueError("output.fields must be a non-empty list")
    fields = [str(name) for name in raw_fields]
    supported = {"B_xyz", "B_magnitude", "B_xyz_grid", "A_xyz", "A_xyz_grid"}
    unknown = sorted(set(fields) - supported)
    if unknown:
        raise ValueError(
            f"output.fields contains unsupported field(s) {unknown}; "
            f"expected a subset of {sorted(supported)}")
    path = _require(spec, "path", "output")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("output.path must be a non-empty string")
    return OutputSpec(
        format=fmt,
        path=path,
        fields=fields,
    )


def _finite_vec(value: Sequence[float], context: str) -> list[float]:
    values = list(_triple(value))
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{context} must contain only finite values")
    return values


def _matrix3(value: Any, context: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{context} must be a 3x3 matrix")
    return [v for r, row in enumerate(value)
            for v in _finite_vec(row, f"{context}[{r}]")]


def _parse_evaluator(
    spec: Any,
) -> tuple[str, dict[str, list[float]], str | None]:
    if spec is None:
        return "biot_savart", {}, None
    if not isinstance(spec, dict):
        raise ValueError("evaluator must be a mapping")
    ev_type = str(spec.get("type", "biot_savart"))
    _validate_evaluator_type(ev_type, "evaluator.type")

    if ev_type == "biot_savart":
        allowed = {"type"}
        params: dict[str, list[float]] = {}
    elif ev_type == "uniform":
        allowed = {"type", "B_T", "b_tesla", "B", "E_V_per_m", "e_v_per_m", "E"}
        b = _unique_alias(
            spec, ("B_T", "b_tesla", "B"), "evaluator uniform B",
            [0, 0, 0])
        e = _unique_alias(
            spec, ("E_V_per_m", "e_v_per_m", "E"), "evaluator uniform E",
            [0, 0, 0])
        params = {"b0": _finite_vec(b, "evaluator.B_T"),
                  "e0": _finite_vec(e, "evaluator.E_V_per_m")}
    elif ev_type == "dipole":
        allowed = {"type", "moment_Am2", "moment_A_m2", "moment",
                   "origin_xyz_m", "origin"}
        moment = _unique_alias(
            spec, ("moment_Am2", "moment_A_m2", "moment"),
            "evaluator dipole moment")
        if moment is None:
            raise ValueError("evaluator.moment_Am2 is required for type 'dipole'")
        origin = _unique_alias(
            spec, ("origin_xyz_m", "origin"), "evaluator dipole origin",
            [0, 0, 0])
        params = {"moment": _finite_vec(moment, "evaluator.moment_Am2"),
                  "origin": _finite_vec(origin, "evaluator.origin_xyz_m")}
    elif ev_type == "gradient":
        allowed = {"type", "B0_T", "b0_tesla", "b0", "grad_T_per_m",
                   "gradient_T_per_m", "gradient", "origin_xyz_m", "origin"}
        grad = _unique_alias(
            spec, ("grad_T_per_m", "gradient_T_per_m", "gradient"),
            "evaluator gradient matrix")
        if grad is None:
            raise ValueError("evaluator.grad_T_per_m is required for type 'gradient'")
        b0 = _unique_alias(
            spec, ("B0_T", "b0_tesla", "b0"), "evaluator gradient B0",
            [0, 0, 0])
        origin = _unique_alias(
            spec, ("origin_xyz_m", "origin"), "evaluator gradient origin",
            [0, 0, 0])
        params = {"b0": _finite_vec(b0, "evaluator.B0_T"),
                  "grad": _matrix3(grad, "evaluator.grad_T_per_m"),
                  "origin": _finite_vec(origin, "evaluator.origin_xyz_m")}
        trace = params["grad"][0] + params["grad"][4] + params["grad"][8]
        scale = max(abs(params["grad"][i]) for i in (0, 4, 8))
        if abs(trace) > 64.0 * math.ulp(1.0) * scale:
            raise ValueError(
                "evaluator.grad_T_per_m must be trace-free (Maxwell div(B)=0)")
    elif ev_type == "file_grid":
        allowed = {"type", "path", "file"}
        file_path = _unique_alias(
            spec, ("path", "file"), "evaluator file-grid path")
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("evaluator.path is required for type 'file_grid'")
        params = {}
    else:
        # Registry plugins use the C++ configure seam directly. Built-ins retain
        # their unit-explicit friendly schemas above; plugins declare already
        # resolved flat numeric parameters under one unambiguous key.
        allowed = {"type", "params"}
        params = _flat_evaluator_params(
            spec.get("params"), "evaluator.params")

    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValueError(f"evaluator contains unsupported key(s) {unknown} for {ev_type!r}")
    return ev_type, params, (str(file_path) if ev_type == "file_grid" else None)


def parse(data: dict, *, base_dir: Path | str | None = None) -> CoilDeck:
    """Parse a coil deck.

    ``file_grid`` evaluators need a filesystem context for their confined NPZ
    input.  Pass ``base_dir`` for an in-memory mapping; :func:`load` supplies the
    YAML file's directory automatically.
    """
    data = _mapping(data, "deck")
    _reject_unknown(
        data, {"units", "conductors", "observation", "output", "evaluator"},
        "deck")
    units = data.get("units", "SI")
    if units != "SI":
        raise ValueError(f"only units: SI is supported, got {units!r}")

    evaluator_type, evaluator_params, evaluator_file = _parse_evaluator(
        data.get("evaluator"))

    raw_conductors = data.get("conductors", [])
    if evaluator_type == "biot_savart" and not raw_conductors:
        raise ValueError("deck.conductors must be non-empty for biot_savart")
    cs = build_conductor_system(raw_conductors)

    obs = _build_observation(_require(data, "observation", "deck"))
    out = _parse_output(_require(data, "output", "deck"))

    deck = CoilDeck(units=units, conductors=cs, observation=obs,
                    output=out, raw=data, evaluator_type=evaluator_type,
                    evaluator_params=evaluator_params,
                    evaluator_file=evaluator_file)
    deck.validate()
    if deck.evaluator_type == "file_grid":
        if base_dir is None:
            raise ValueError(
                "file_grid decks parsed from a mapping require parse(..., "
                "base_dir=...); use load(path) for YAML decks")
        deck.evaluator_params = load_file_grid_npz(
            base_dir, deck.evaluator_file, label="evaluator.path")
    return deck


def load(path: Union[str, Path]) -> CoilDeck:
    deck_path = Path(path).resolve()
    with open(deck_path) as fh:
        data = _load_yaml(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return parse(data, base_dir=deck_path.parent)
