"""Small host-side initial-condition helpers for PIC examples."""

from __future__ import annotations

import numpy as np


def maxwellian(n_particles: int, thermal_speed: float, drift=(0.0, 0.0, 0.0), seed: int = 0):
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, thermal_speed, size=(n_particles, 3))
    v += np.asarray(drift, dtype=float)
    return v


def _block_side(n_particles: int) -> int:
    return int(np.ceil(np.sqrt(n_particles)))


def quiet_positions_2d_block(n_particles: int, x_min: float, x_max: float,
                              y_min: float, y_max: float):
    side = _block_side(n_particles)
    x = x_min + (np.arange(side) + 0.5) * (x_max - x_min) / side
    y = y_min + (np.arange(side) + 0.5) * (y_max - y_min) / side
    xx, yy = np.meshgrid(x, y, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return pts[:n_particles]


def quiet_block_cell_area(n_particles: int, x_min: float, x_max: float,
                          y_min: float, y_max: float) -> float:
    """Area each quiet-start particle represents: one cell of the side x side
    layout (side = ceil(sqrt(n_particles))). Using density * cell_area for the
    macro-weight keeps the local number density exactly `density` even when
    n_particles is not a perfect square and the last layout row is truncated;
    dividing the whole block area by n_particles instead would over-count the
    density by side**2 / n_particles."""
    side = _block_side(n_particles)
    return (x_max - x_min) * (y_max - y_min) / (side * side)


def quiet_positions_2d(n_particles: int, lx: float, ly: float):
    """Quiet-start grid over the whole [0, lx) x [0, ly) domain.

    Special case of quiet_positions_2d_block."""
    return quiet_positions_2d_block(n_particles, 0.0, lx, 0.0, ly)
