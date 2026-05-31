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
