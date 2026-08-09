"""Physics-neutral helpers shared by distributed Python runners."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np


_T = TypeVar("_T")


@dataclass(frozen=True)
class RunPhaseTimes:
    """Wall-clock boundaries shared by the distributed run frontends."""

    start_wall: float
    prepare_done: float
    setup_done: float
    evolution_done: float
    diagnostics_done: float
    checkpoint_seconds: float


def error_message(error: BaseException) -> str:
    """Render a bounded, non-empty message for collective consensus."""

    detail = str(error).strip() or type(error).__name__
    return f"{type(error).__name__}: {detail}"[:4096]


def collective_local(
        session: Any, phase: str, operation: Callable[[], _T]) -> _T:
    """Run rank-local Python work and agree before the next collective."""

    result: _T | None = None
    failure: BaseException | None = None
    try:
        result = operation()
    except BaseException as error:
        failure = error
    session.collective_require(
        failure is None, phase,
        "" if failure is None else error_message(failure))
    return result  # type: ignore[return-value]


def canonical_policy_signature(document: dict[str, Any]) -> str:
    """Serialize a distributed-run policy for exact cross-rank agreement."""

    return json.dumps(
        document, allow_nan=False, separators=(",", ":"), sort_keys=True)


def restart_target_error(
        step: int, sim_time: float, step_limit: int,
        time_limit: float | None) -> str:
    """Describe an absolute restart target violation, if any."""

    if step > step_limit:
        return (
            f"restart step {step} exceeds the absolute termination "
            f"step {step_limit}")
    if time_limit is not None and sim_time > time_limit:
        return (
            f"restart time {sim_time:.17g} exceeds the absolute "
            f"termination time {time_limit:.17g}")
    return ""


def finalize_run_telemetry(
        telemetry: dict[str, Any], *, physics: str,
        requested_transport: str, decomposition: Any,
        phase_times: RunPhaseTimes) -> None:
    """Add physics-neutral timing and transport fields to telemetry."""

    telemetry["wall_seconds"] = (
        phase_times.diagnostics_done - phase_times.start_wall)
    telemetry["phase_seconds"] = {
        "prepare": phase_times.prepare_done - phase_times.start_wall,
        "setup": phase_times.setup_done - phase_times.prepare_done,
        "evolution": phase_times.evolution_done - phase_times.setup_done,
        "checkpoint": phase_times.checkpoint_seconds,
        "diagnostics": (
            phase_times.diagnostics_done - phase_times.evolution_done),
    }
    native_physics = dict(telemetry.get(physics, {}))
    native_transport = dict(native_physics.get("transport", {}))
    telemetry["transport_requested"] = native_transport.get(
        "requested", requested_transport)
    telemetry["transport"] = native_transport.get(
        "interprocess", requested_transport)
    if (telemetry["transport_requested"] == "auto"
            and telemetry["transport"] == "staged"):
        if not native_transport.get("direct_query_recognized", False):
            reason = "MPI did not report ROCm-aware device-buffer support"
        else:
            reason = "the collective device-buffer startup probe failed"
        telemetry["transport_fallback_reason"] = reason
    telemetry["decomposition"] = tuple(decomposition)


def _temporary_file(path: Path) -> tuple[int, Path]:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return descriptor, Path(raw_path)


def atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Atomically publish an NPZ through an exclusive same-directory file."""

    descriptor, temporary = _temporary_file(path)
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            np.savez(stream, **payload)
            stream.flush()
        temporary.replace(path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically publish JSON through an exclusive same-directory file."""

    descriptor, temporary = _temporary_file(path)
    try:
        stream = os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n")
        descriptor = -1
        with stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
        temporary.replace(path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
