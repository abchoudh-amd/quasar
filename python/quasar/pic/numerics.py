"""Shared numerical helpers for the PIC front-end.

Single source of truth for the CFL timestep so the CLI and the example deck
generators cannot drift apart (they previously each hardcoded the formula).
"""

from __future__ import annotations

import math

C_LIGHT = 299792458.0


def cfl_limit(dx: float, dy: float, c: float = C_LIGHT, fdtd_order: int = 2) -> float:
    """The 2D Yee CFL stability limit for the given FDTD order.

    Mirrors C++ ``quasar::cfl_dt`` (include/quasar/core/grid.hpp): the 4th-order
    curl tightens the limit by a factor 7/6.
    """
    if fdtd_order == 2:
        factor = 1.0
    elif fdtd_order == 4:
        factor = 7.0 / 6.0
    else:
        raise ValueError("fdtd_order must be 2 or 4")
    return 1.0 / (c * factor * math.sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy)))


def cfl_dt(dx: float, dy: float, c: float = C_LIGHT, fdtd_order: int = 2,
           safety: float = 0.5) -> float:
    """A safe timestep: the CFL limit scaled by ``safety`` (default 0.5)."""
    return safety * cfl_limit(dx, dy, c, fdtd_order)
