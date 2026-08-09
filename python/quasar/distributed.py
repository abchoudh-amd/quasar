"""Public configuration and result types for Quasar distributed runs.

This module is intentionally pure Python and is staged in every build.  The
optional native ``quasar._distributed`` module is probed when this module loads;
serial-only builds therefore retain a stable, importable API that can explain
why a distributed request cannot be executed.
"""

from __future__ import annotations

import math
import operator
import re
import json
from dataclasses import dataclass, field
from importlib import import_module
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


class DistributedUnavailableError(RuntimeError):
    """Raised when a distributed run is requested from a serial-only build."""


try:
    _native = import_module("quasar._distributed")
except (ImportError, OSError) as _error:
    # ImportError covers a serial-only build; OSError covers a present extension
    # whose MPI/HDF5 runtime dependencies cannot be loaded.  In both cases the
    # public module must remain importable so callers receive the exception above
    # at the actual distributed request boundary.
    _native = None
    _native_import_error: BaseException | None = _error
else:
    _native_import_error = None


def foundation_available() -> bool:
    """Return whether the optional native MPI/HDF5 foundation is loaded."""

    if _native is None:
        return False
    probe = getattr(_native, "foundation_available", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:
        return False


def is_available() -> bool:
    """Return whether the complete native distributed runner is available."""

    if _native is None:
        return False
    probe = getattr(_native, "is_available", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:
        # Availability is a query, and is commonly used for optional test skips.
        # A broken runtime probe is unavailable; the request path below provides
        # the actionable error rather than making this predicate surprising.
        return False


def unavailable_reason() -> str | None:
    """Return a human-readable reason, or ``None`` when support is available."""

    if is_available():
        return None
    if _native is None:
        if (isinstance(_native_import_error, ModuleNotFoundError)
                and _native_import_error.name == "quasar._distributed"):
            return "this Quasar build does not include distributed support"
        if _native_import_error is not None:
            return ("the native distributed runtime could not be loaded: "
                    f"{_native_import_error}")
        return "this Quasar build does not include distributed support"
    detail = getattr(_native, "unavailable_reason", None)
    if callable(detail):
        try:
            value = detail()
        except Exception as exc:
            return f"the native distributed availability probe failed: {exc}"
        if value:
            return str(value)
    return "the native distributed runtime is not available"


def require_available() -> None:
    """Raise :class:`DistributedUnavailableError` unless support is usable."""

    reason = unavailable_reason()
    if reason is not None:
        raise DistributedUnavailableError(
            f"distributed execution is unavailable: {reason}")


_NativeRuntimeSession = (
    getattr(_native, "RuntimeSession", None)
    if foundation_available() else None
)

if _NativeRuntimeSession is not None:
    RuntimeSession = _NativeRuntimeSession
else:
    class RuntimeSession:
        """Unavailable-build placeholder for the native collective session."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            reason = unavailable_reason()
            if reason is None:
                reason = "the native build does not expose RuntimeSession"
            raise DistributedUnavailableError(
                f"distributed runtime foundation is unavailable: {reason}")


def _devices(value: str | Sequence[int]) -> str | tuple[int, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "auto":
            return "auto"
        if not stripped or any(not item.strip() for item in stripped.split(",")):
            raise ValueError("devices must be 'auto' or a comma-separated list of IDs")
        raw_values: Sequence[Any] = tuple(item.strip() for item in stripped.split(","))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = value
    else:
        raise TypeError("devices must be 'auto' or a sequence of device IDs")

    normalized: list[int] = []
    for raw in raw_values:
        if isinstance(raw, bool):
            raise ValueError("device IDs must be non-negative integers")
        try:
            device = operator.index(raw) if not isinstance(raw, str) else int(raw, 10)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("device IDs must be non-negative integers") from exc
        if device < 0:
            raise ValueError("device IDs must be non-negative integers")
        normalized.append(int(device))
    if not normalized:
        raise ValueError("at least one device ID is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("device IDs must not contain duplicates")
    return tuple(normalized)


_DECOMPOSITION_RE = re.compile(r"^([1-9][0-9]*)[xX]([1-9][0-9]*)$")


def _decomposition(value: str | Sequence[int]) -> str | tuple[int, int]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "auto":
            return "auto"
        match = _DECOMPOSITION_RE.fullmatch(stripped)
        if match is None:
            raise ValueError("decomposition must be 'auto' or PXxPY with positive integers")
        return int(match.group(1)), int(match.group(2))
    if (not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray))
            or len(value) != 2):
        raise TypeError("decomposition must be 'auto' or a two-integer sequence")
    normalized: list[int] = []
    for raw in value:
        if isinstance(raw, bool):
            raise ValueError("decomposition dimensions must be positive integers")
        try:
            dimension = operator.index(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "decomposition dimensions must be positive integers") from exc
        if dimension <= 0:
            raise ValueError("decomposition dimensions must be positive integers")
        normalized.append(int(dimension))
    return normalized[0], normalized[1]


def _choice(value: str, choices: tuple[str, ...], label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    normalized = value.strip().lower()
    if normalized not in choices:
        rendered = ", ".join(repr(item) for item in choices)
        raise ValueError(f"{label} must be one of {rendered}")
    return normalized


def _optional_path(value: str | PathLike[str] | None, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, PathLike)):
        raise TypeError(f"{label} must be a path-like value")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{label} must not be empty")
    return Path(value)


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Placement, transport, diagnostics, and restart policy for one run.

    Passing a ``RunOptions`` instance to :func:`quasar.pic.run` or
    :func:`quasar.mhd.run` explicitly selects the distributed runtime.  Omit it
    (``options=None``) to use the existing serial implementation.
    """

    devices: str | Sequence[int] = "auto"
    decomposition: str | Sequence[int] = "auto"
    transport: str = "auto"
    diagnostics_layout: str = "gathered"
    checkpoint: str | PathLike[str] | None = None
    checkpoint_every: int | None = None
    restart: str | PathLike[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "devices", _devices(self.devices))
        object.__setattr__(self, "decomposition", _decomposition(self.decomposition))
        object.__setattr__(
            self, "transport",
            _choice(self.transport, ("auto", "staged", "direct"), "transport"))
        object.__setattr__(
            self, "diagnostics_layout",
            _choice(self.diagnostics_layout, ("gathered", "sharded"),
                    "diagnostics_layout"))
        object.__setattr__(
            self, "checkpoint", _optional_path(self.checkpoint, "checkpoint"))
        object.__setattr__(self, "restart", _optional_path(self.restart, "restart"))

        cadence = self.checkpoint_every
        if cadence is not None:
            if isinstance(cadence, bool):
                raise ValueError("checkpoint_every must be a positive integer")
            try:
                cadence = operator.index(cadence)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "checkpoint_every must be a positive integer") from exc
            if cadence <= 0:
                raise ValueError("checkpoint_every must be a positive integer")
            if self.checkpoint is None:
                raise ValueError("checkpoint_every requires checkpoint")
            object.__setattr__(self, "checkpoint_every", int(cadence))

    def as_dict(self) -> dict[str, Any]:
        """Return values in the stable, native-runner interchange form."""

        return {
            "devices": self.devices,
            "decomposition": self.decomposition,
            "transport": self.transport,
            "diagnostics_layout": self.diagnostics_layout,
            "checkpoint": (None if self.checkpoint is None
                           else str(self.checkpoint)),
            "checkpoint_every": self.checkpoint_every,
            "restart": None if self.restart is None else str(self.restart),
        }


@dataclass(frozen=True, slots=True)
class RunResult:
    """Summary returned after a successfully committed serial or distributed run."""

    final_step: int
    final_time: float
    diagnostics_path: str | PathLike[str] | None = None
    checkpoint_path: str | PathLike[str] | None = None
    distributed: bool = False
    telemetry: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.final_step, bool):
            raise ValueError("final_step must be a non-negative integer")
        try:
            final_step = operator.index(self.final_step)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("final_step must be a non-negative integer") from exc
        if final_step < 0:
            raise ValueError("final_step must be a non-negative integer")
        object.__setattr__(self, "final_step", int(final_step))

        if isinstance(self.final_time, bool):
            raise ValueError("final_time must be finite and non-negative")
        try:
            final_time = float(self.final_time)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("final_time must be finite and non-negative") from exc
        if not math.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and non-negative")
        object.__setattr__(self, "final_time", final_time)
        object.__setattr__(
            self, "diagnostics_path",
            _optional_path(self.diagnostics_path, "diagnostics_path"))
        object.__setattr__(
            self, "checkpoint_path",
            _optional_path(self.checkpoint_path, "checkpoint_path"))
        if not isinstance(self.distributed, bool):
            raise TypeError("distributed must be a bool")
        if not isinstance(self.telemetry, Mapping):
            raise TypeError("telemetry must be a mapping")
        copied: dict[str, Any] = {}
        for key, value in self.telemetry.items():
            if not isinstance(key, str):
                raise TypeError("telemetry keys must be strings")
            copied[key] = value
        object.__setattr__(self, "telemetry", MappingProxyType(copied))

    @property
    def steps_completed(self) -> int:
        """Alias for ``final_step`` for orchestration code."""

        return self.final_step

    @property
    def final_time_s(self) -> float:
        """Alias matching the existing NPZ metadata key."""

        return self.final_time

    @property
    def output_path(self) -> Path | None:
        """Alias for the diagnostics artifact path."""

        return self.diagnostics_path


@dataclass(frozen=True, slots=True)
class DiagnosticShardInfo:
    rank: int
    node_rank: int
    local_device: int
    endpoint: int
    device_identity: str
    tile: tuple[int, int]
    offset: tuple[int, int]
    owned_shape: tuple[int, int]
    path: Path


@dataclass(frozen=True, slots=True)
class DiagnosticsManifest:
    physics: str
    geometry: str
    global_shape: tuple[int, int]
    step: int
    time: float
    decomposition: tuple[int, int]
    shards: tuple[DiagnosticShardInfo, ...]
    path: Path

    @property
    def global_ny(self) -> int:
        return self.global_shape[0]

    @property
    def global_nx(self) -> int:
        return self.global_shape[1]


def _manifest_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return int(result)


def _manifest_pair(value: Any, label: str, *, minimum: int) -> tuple[int, int]:
    if (not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or len(value) != 2):
        raise ValueError(f"{label} must contain exactly two integers")
    return (_manifest_integer(value[0], f"{label}[0]", minimum=minimum),
            _manifest_integer(value[1], f"{label}[1]", minimum=minimum))


def read_diagnostics_manifest(
        path: str | PathLike[str], *, verify_shards: bool = True,
        ) -> DiagnosticsManifest:
    """Read and validate a completed sharded-diagnostics manifest."""

    manifest_path = _optional_path(path, "manifest path")
    assert manifest_path is not None
    manifest_path = manifest_path.resolve()
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read diagnostics manifest {manifest_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("diagnostics manifest root must be an object")
    if document.get("schema") != "quasar-diagnostics-shards/v1":
        raise ValueError(
            "diagnostics manifest schema must be quasar-diagnostics-shards/v1")
    physics = document.get("physics")
    if not isinstance(physics, str) or not physics:
        raise ValueError("diagnostics manifest physics must be a non-empty string")
    geometry = document.get("geometry")
    if not isinstance(geometry, str) or not geometry:
        raise ValueError("diagnostics manifest geometry must be non-empty")
    global_shape = _manifest_pair(
        document.get("global_shape"), "global_shape", minimum=1)
    step = _manifest_integer(document.get("step"), "step")
    try:
        time_value = float(document.get("time"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("diagnostics manifest time must be finite and non-negative") from exc
    if not math.isfinite(time_value) or time_value < 0.0:
        raise ValueError("diagnostics manifest time must be finite and non-negative")
    decomposition_value = document.get("decomposition")
    if not isinstance(decomposition_value, dict):
        raise ValueError("diagnostics manifest decomposition must be an object")
    decomposition = (
        _manifest_integer(decomposition_value.get("px"), "decomposition.px",
                          minimum=1),
        _manifest_integer(decomposition_value.get("py"), "decomposition.py",
                          minimum=1),
    )
    records = document.get("shards")
    if not isinstance(records, list) or len(records) != math.prod(decomposition):
        raise ValueError(
            "diagnostics shard count does not match the decomposition")

    shards: list[DiagnosticShardInfo] = []
    endpoints: set[int] = set()
    tiles: set[tuple[int, int]] = set()
    shard_paths: set[Path] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"shards[{index}] must be an object")
        tile = _manifest_pair(record.get("tile"), f"shards[{index}].tile",
                              minimum=0)
        offset = _manifest_pair(
            record.get("offset"), f"shards[{index}].offset", minimum=0)
        owned_shape = _manifest_pair(
            record.get("owned_shape"), f"shards[{index}].owned_shape",
            minimum=1)
        endpoint = _manifest_integer(
            record.get("endpoint"), f"shards[{index}].endpoint")
        if endpoint in endpoints or tile in tiles:
            raise ValueError("diagnostics endpoints and tile coordinates must be unique")
        endpoints.add(endpoint)
        tiles.add(tile)
        if tile[0] >= decomposition[0] or tile[1] >= decomposition[1]:
            raise ValueError("diagnostics tile lies outside the decomposition")
        if (offset[0] + owned_shape[0] > global_shape[0]
                or offset[1] + owned_shape[1] > global_shape[1]):
            raise ValueError("diagnostics owned extent lies outside the global mesh")
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("diagnostics shard path must be non-empty")
        shard_path = Path(raw_path)
        if not shard_path.is_absolute():
            shard_path = manifest_path.parent / shard_path
        shard_path = shard_path.resolve()
        if shard_path in shard_paths:
            raise ValueError("diagnostics shard paths must be unique")
        shard_paths.add(shard_path)
        if verify_shards and not shard_path.is_file():
            raise ValueError(f"diagnostics shard is missing: {shard_path}")
        identity = record.get("device_identity")
        if not isinstance(identity, str) or not identity:
            raise ValueError("diagnostics device identity must be non-empty")
        shards.append(DiagnosticShardInfo(
            rank=_manifest_integer(record.get("rank"), f"shards[{index}].rank"),
            node_rank=_manifest_integer(
                record.get("node_rank"), f"shards[{index}].node_rank"),
            local_device=_manifest_integer(
                record.get("local_device"), f"shards[{index}].local_device"),
            endpoint=endpoint,
            device_identity=identity,
            tile=tile,
            offset=offset,
            owned_shape=owned_shape,
            path=shard_path,
        ))

    if endpoints != set(range(len(shards))):
        raise ValueError("diagnostics endpoints must be contiguous from zero")
    cell_count = 0
    for index, shard in enumerate(shards):
        cell_count += math.prod(shard.owned_shape)
        for prior in shards[:index]:
            separated = (
                shard.offset[0] + shard.owned_shape[0] <= prior.offset[0]
                or prior.offset[0] + prior.owned_shape[0] <= shard.offset[0]
                or shard.offset[1] + shard.owned_shape[1] <= prior.offset[1]
                or prior.offset[1] + prior.owned_shape[1] <= shard.offset[1]
            )
            if not separated:
                raise ValueError("diagnostics owned extents overlap")
    if cell_count != math.prod(global_shape):
        raise ValueError("diagnostics owned extents do not cover the global mesh")
    return DiagnosticsManifest(
        physics=physics,
        geometry=geometry,
        global_shape=global_shape,
        step=step,
        time=time_value,
        decomposition=decomposition,
        shards=tuple(sorted(shards, key=lambda shard: shard.endpoint)),
        path=manifest_path,
    )


def _coerce_result(value: Any) -> RunResult:
    if isinstance(value, RunResult):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("native distributed runner returned an invalid result")
    fields = dict(value)
    aliases = {
        "steps_completed": "final_step",
        "final_time_s": "final_time",
        "output_path": "diagnostics_path",
    }
    for old, new in aliases.items():
        if new not in fields and old in fields:
            fields[new] = fields.pop(old)
    fields.setdefault("distributed", True)
    return RunResult(**fields)


_RUNNER_REGISTRY: dict[str, Callable[..., Any]] = {}


def _register_runner(physics: str, runner: Callable[..., Any]) -> None:
    """Register one physics-owned distributed orchestration entry point."""

    if not isinstance(physics, str) or not physics:
        raise ValueError("distributed runner physics name must be non-empty")
    if not callable(runner):
        raise TypeError("distributed runner must be callable")
    _RUNNER_REGISTRY[physics] = runner


def _execute(physics: str, input_deck: Any, options: RunOptions,
             **kwargs: Any) -> RunResult:
    """Invoke an optional native runner after a collective-safe availability check."""

    if not isinstance(options, RunOptions):
        raise TypeError("options must be a quasar.distributed.RunOptions instance")
    runtime_probe = (None if _native is None else
                     getattr(_native, f"{physics}_runtime_available", None))
    registered_runner = _RUNNER_REGISTRY.get(physics)
    if (registered_runner is not None and callable(runtime_probe)
            and runtime_probe()):
        return _coerce_result(registered_runner(input_deck, options, **kwargs))
    require_available()
    runner = getattr(_native, f"run_{physics}", None)
    if not callable(runner):
        raise DistributedUnavailableError(
            f"distributed {physics.upper()} execution is unavailable: "
            "the native runner is not installed")
    return _coerce_result(runner(input_deck, options.as_dict(), **kwargs))


__all__ = [
    "DiagnosticShardInfo",
    "DiagnosticsManifest",
    "DistributedUnavailableError",
    "RunOptions",
    "RunResult",
    "RuntimeSession",
    "foundation_available",
    "is_available",
    "read_diagnostics_manifest",
    "require_available",
    "unavailable_reason",
]
