"""Small host-side initial-condition helpers for PIC examples."""

from __future__ import annotations

import numpy as np


def maxwellian(n_particles: int, thermal_speed: float, drift=(0.0, 0.0, 0.0), seed: int = 0):
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, thermal_speed, size=(n_particles, 3))
    v += np.asarray(drift, dtype=float)
    return v


def quiet_positions_2d_block(n_particles: int, x_min: float, x_max: float,
                              y_min: float, y_max: float):
    side = int(np.ceil(np.sqrt(n_particles)))
    x = x_min + (np.arange(side) + 0.5) * (x_max - x_min) / side
    y = y_min + (np.arange(side) + 0.5) * (y_max - y_min) / side
    xx, yy = np.meshgrid(x, y, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return pts[:n_particles]


def quiet_positions_2d(n_particles: int, lx: float, ly: float):
    """Quiet-start grid over the whole [0, lx) x [0, ly) domain.

    Special case of quiet_positions_2d_block."""
    return quiet_positions_2d_block(n_particles, 0.0, lx, 0.0, ly)
