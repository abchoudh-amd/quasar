"""Shared YAML deck-parsing helpers for the coil, pic, and mhd input loaders.

``quasar.coil.io``, ``quasar.pic.io``, and ``quasar.mhd.io`` parse user-authored
YAML decks and need the same missing-key check, 3-element coercion, finite-value
validation, and side-map parsing. Keep that logic here so the loaders stay in
sync.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


def require(d: dict, key: str, context: str) -> Any:
    """Return ``d[key]`` or raise a ValueError naming the missing field."""
    if key not in d:
        raise ValueError(f"{context}: missing required field {key!r}")
    return d[key]


def triple(xyz: Sequence[float]) -> tuple[float, float, float]:
    """Coerce a 3-element sequence to a float tuple, validating its length."""
    if len(xyz) != 3:
        raise ValueError(f"expected 3-element xyz triple, got {xyz!r}")
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


# -- Finite-value validators (shared by the pic and mhd deck loaders) ----------

def as_finite(value: float, context: str) -> float:
    """Coerce ``value`` to a finite float or raise a ValueError naming it."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{context} must be a finite number") from None
    if not math.isfinite(v):
        raise ValueError(f"{context} must be finite")
    return v


def require_finite(value: float, context: str) -> None:
    as_finite(value, context)


def require_positive_finite(value: float, context: str) -> None:
    if as_finite(value, context) <= 0:
        raise ValueError(f"{context} must be positive")


def require_nonnegative_finite(value: float, context: str) -> None:
    if as_finite(value, context) < 0:
        raise ValueError(f"{context} must be >= 0")


def require_vec_finite(values: Sequence[float], context: str) -> None:
    for i, value in enumerate(values):
        require_finite(value, f"{context}[{i}]")


# -- Per-side boundary spec parsing (shared by the pic and mhd deck loaders) ----

# Canonical side ordering [x_lo, x_hi, y_lo, y_hi].
SIDE_KEYS = ("x_lo", "x_hi", "y_lo", "y_hi")


def parse_side_map(spec, default: str, what: str) -> tuple[str, str, str, str]:
    """Parse a per-side boundary spec into a 4-tuple in SIDE_KEYS order.

    Accepts a scalar (all sides), a 4-element list ([x_lo, x_hi, y_lo, y_hi]), or
    a dict keyed by side name.
    """
    if spec is None:
        return (default, default, default, default)
    if isinstance(spec, str):
        return (spec, spec, spec, spec)
    if isinstance(spec, (list, tuple)) and len(spec) == 4:
        return (str(spec[0]), str(spec[1]), str(spec[2]), str(spec[3]))
    if isinstance(spec, dict):
        return tuple(str(spec.get(k, default)) for k in SIDE_KEYS)  # type: ignore[return-value]
    raise ValueError(f"{what} must be a string, 4-element list, or side-keyed map")


# Field evaluators selectable from a deck (coil top-level ``evaluator.type`` or
# pic ``external_field.evaluator.type``). These are registered on the C++ side
# (QUASAR_REGISTER_FIELD_EVALUATOR) and built by name via create_field_evaluator.
# "file_grid" is registered but not yet implemented, so it is intentionally
# excluded here — a deck selecting it would otherwise hit a raw C++
# std::logic_error. Single source of truth so the coil and pic loaders cannot
# drift apart.
SUPPORTED_EVALUATORS = ("biot_savart", "uniform", "dipole", "gradient")


def validate_evaluator_type(name: str, context: str) -> None:
    """Raise ValueError if ``name`` is not a deck-selectable field evaluator."""
    if name not in SUPPORTED_EVALUATORS:
        raise ValueError(
            f"{context} {name!r} must be one of {list(SUPPORTED_EVALUATORS)}")
