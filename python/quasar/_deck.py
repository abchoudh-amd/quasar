"""Shared YAML deck-parsing helpers for the coil, pic, and mhd input loaders.

``quasar.coil.io``, ``quasar.pic.io``, and ``quasar.mhd.io`` parse user-authored
YAML decks and need the same missing-key check, 3-element coercion, finite-value
validation, and side-map parsing. Keep that logic here so the loaders stay in
sync.
"""

from __future__ import annotations

import math
import operator
from typing import Any, Sequence

import yaml


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(loader: _UniqueKeySafeLoader,
                              node: yaml.MappingNode,
                              deep: bool = False) -> dict:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        # A YAML merge key is intentionally allowed to provide defaults that an
        # explicit key overrides.  Duplicate spelling in the authored mapping
        # itself remains an error.
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
            seen.add(key)
        except TypeError:
            # The underlying SafeLoader will report an unhashable mapping key.
            continue
        if duplicate:
            mark = key_node.start_mark
            raise ValueError(
                f"duplicate YAML key {key!r} at line {mark.line + 1}, "
                f"column {mark.column + 1}")
    return loader.construct_mapping(node, deep=deep)


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping)


def load_yaml(stream: Any) -> Any:
    """Safely load one YAML document and reject duplicate mapping keys."""
    return yaml.load(stream, Loader=_UniqueKeySafeLoader)


def require(d: dict, key: str, context: str) -> Any:
    """Return ``d[key]`` or raise a ValueError naming the missing field."""
    if key not in d:
        raise ValueError(f"{context}: missing required field {key!r}")
    return d[key]


def unique_alias(d: dict, aliases: Sequence[str], context: str,
                 default: Any = None) -> Any:
    """Return the sole supplied alias, rejecting ambiguous spellings.

    Deck schemas retain a few historical names for the same physical quantity.
    Accepting two of them and silently preferring one makes the result depend on
    parser precedence, even when the author intended the other value to win.
    """
    present = [name for name in aliases if name in d]
    if len(present) > 1:
        raise ValueError(
            f"{context} supplies multiple aliases {present}; use exactly one of "
            f"{list(aliases)}")
    return d[present[0]] if present else default


def as_integer(value: Any, context: str) -> int:
    """Require an exact integer, rejecting booleans and lossy ``int(...)`` casts."""
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer, not a boolean")
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{context} must be an integer") from None


def as_boolean(value: Any, context: str) -> bool:
    """Require a real YAML/Python boolean instead of truth-value coercion."""
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be true or false")
    return value


def triple(xyz: Sequence[float]) -> tuple[float, float, float]:
    """Coerce a 3-element sequence to a float tuple, validating its length."""
    if isinstance(xyz, (str, bytes)):
        raise ValueError("expected a 3-element xyz triple")
    try:
        length = len(xyz)
    except TypeError:
        raise ValueError("expected a 3-element xyz triple") from None
    if length != 3:
        raise ValueError(f"expected 3-element xyz triple, got {xyz!r}")
    if any(isinstance(xyz[index], bool) for index in range(3)):
        raise ValueError("xyz triple entries must be numbers, not booleans")
    try:
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("xyz triple entries must be representable numbers") from None


# -- Finite-value validators (shared by the pic and mhd deck loaders) ----------

def as_finite(value: float, context: str) -> float:
    """Coerce ``value`` to a finite float or raise a ValueError naming it."""
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a finite number, not a boolean")
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
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
        unknown = sorted(set(spec) - set(SIDE_KEYS))
        if unknown:
            raise ValueError(f"{what} contains unknown side key(s) {unknown}")
        return tuple(str(spec.get(k, default)) for k in SIDE_KEYS)  # type: ignore[return-value]
    raise ValueError(f"{what} must be a string, 4-element list, or side-keyed map")


def validate_evaluator_type(name: str, context: str) -> None:
    """Require a field evaluator currently registered by the C++ runtime.

    Querying at validation time keeps Python decks compatible with evaluator
    plugins loaded after :mod:`quasar` itself, instead of freezing the built-in
    names in a second hard-coded registry.
    """
    from . import _core

    supported = tuple(_core.magnetostatics.field_evaluator_names())
    if name not in supported:
        raise ValueError(
            f"{context} {name!r} must be one of {list(supported)}")


def flat_evaluator_params(value: Any, context: str) -> dict[str, list[float]]:
    """Parse a plugin evaluator's generic ``name -> flat numeric list`` map."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    result: dict[str, list[float]] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{context} keys must be non-empty strings")
        if isinstance(raw, bool) or isinstance(raw, (str, bytes, dict)):
            raise ValueError(f"{context}.{key} must be a number or flat numeric list")
        entries = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        converted: list[float] = []
        for index, entry in enumerate(entries):
            if isinstance(entry, bool):
                raise ValueError(
                    f"{context}.{key}[{index}] must be a finite number")
            converted.append(as_finite(entry, f"{context}.{key}[{index}]"))
        result[key] = converted
    return result
