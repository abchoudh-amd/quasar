"""Shared YAML deck-parsing helpers for the coil and pic input loaders.

Both ``quasar.coil.io`` and ``quasar.pic.io`` parse user-authored YAML decks and
need the same missing-key check and 3-element coercion. Keep that logic here so
the two loaders stay in sync.
"""

from __future__ import annotations

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
