"""Shared YAML-emitting helpers for the example deck generators.

The ``build_yaml.py`` generators run as standalone scripts (the package's
compiled ``_core`` extension is not needed just to emit a deck), so they load
this module by file path rather than importing the ``quasar`` package. Keeping
the float formatter and the circular-loop block here means the two square-toroid
generators share one definition of each.
"""

from __future__ import annotations


def fmt_float(x: float) -> str:
    """Fixed-width signed float formatting used throughout the example decks."""
    return f"{x:+.8f}"


def loop_block(*, indent: int, name: str, current_A: float,
               center_z_m: float, radius_m: float, n_segments: int) -> list[str]:
    """Emit a ``circular_loop`` conductor block indented by ``indent`` spaces.

    ``indent`` is the column of the leading ``- name:`` item (2 for a top-level
    ``conductors:`` list, 6 when nested under ``external_field.evaluator``)."""
    pad = " " * indent
    inner = " " * (indent + 2)
    return [
        f"{pad}- name: {name}",
        f"{pad}  current_A: {fmt_float(current_A)}",
        f"{pad}  geometry:",
        f"{inner}  type: circular_loop",
        f"{inner}  center_xyz: [0.0, 0.0, {fmt_float(center_z_m)}]",
        f"{inner}  axis_xyz:   [0.0, 0.0, 1.0]",
        f"{inner}  radius_m: {fmt_float(radius_m)}",
        f"{inner}  n_segments: {n_segments}",
        "",
    ]
