"""Shared numerical helpers for the MHD front-end.

The C++ ``MhdSolver2D::cfl_limit()`` is the authoritative stable-step bound (it
scans the seeded state for the max fast-magnetosonic signal speed). These Python
helpers cover the small primitive<->conserved conversions the deck IC generators
and the CLI need, kept in one place so they cannot drift between io.py and cli.py
(mirrors ``quasar.pic.numerics`` keeping the CFL formula in a single module).
"""

from __future__ import annotations

import numpy as np


def primitive_to_energy(rho, p, vx, vy, vz, bx, by, bz, gamma):
    """Total energy density from primitives, the ideal-MHD conserved energy::

        E = p/(gamma - 1) + 0.5*rho*(vx^2+vy^2+vz^2) + 0.5*(bx^2+by^2+bz^2)

    This is the single source of truth for the p->E conversion every IC builder
    uses; it must match the C++ EOS (E = rho*e + 0.5*rho*v^2 + 0.5*B^2 with
    rho*e = p/(gamma-1)). Works elementwise on NumPy arrays or scalars.
    """
    kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
    magnetic = 0.5 * (bx * bx + by * by + bz * bz)
    return p / (gamma - 1.0) + kinetic + magnetic


def momentum(rho, vx, vy, vz):
    """Conserved momentum density (rho*v) per component."""
    return rho * vx, rho * vy, rho * vz


def fast_magnetosonic_speed_split(rho, p, bx, by, bz, b0x, b0y, b0z, gamma):
    """Fast magnetosonic speed for the field-split form B = B0 + b.

    The signal speed in the field-split ideal-MHD formulation depends on the
    TOTAL field B = B0 + b (the static background B0 contributes to the Alfven
    speed exactly as the evolved perturbation b does), so this simply forms the
    total field and reuses the single-field estimate. Kept here for parity with
    the C++ ``cfl_limit()`` which scans the total field; works elementwise.
    """
    return fast_magnetosonic_speed(
        rho, p,
        np.asarray(bx) + np.asarray(b0x),
        np.asarray(by) + np.asarray(b0y),
        np.asarray(bz) + np.asarray(b0z),
        gamma,
    )


def background_divergence_linf(b0x, b0y, nx, ny, nghost, dx, dy):
    """Max abs interior face-divergence of a staggered background field B0.

    ``b0x``/``b0y`` are 1-D ghost-padded host buffers in the solver storage
    layout (pitch = nx + 2*nghost, row-major). With b0x on the left face and b0y
    on the bottom face, the discrete divergence in interior cell (i, j) is::

        (b0x[i+1, j] - b0x[i, j]) / dx + (b0y[i, j+1] - b0y[i, j]) / dy

    A constant/uniform field gives 0 to round-off. Returns the L-inf norm over
    all interior cells (0.0 when there are no interior cells). This is the single
    source of truth for ``io.build_background_field``'s divergence-free check, so
    the Python staging cannot seed a div-B != 0 background past the solver.
    """
    g = int(nghost)
    pitch = int(nx) + 2 * g
    height = int(ny) + 2 * g
    bx = np.asarray(b0x, dtype=np.float64).reshape(height, pitch)
    by = np.asarray(b0y, dtype=np.float64).reshape(height, pitch)
    if nx <= 0 or ny <= 0:
        return 0.0
    # Interior cells span [g, g+nx) in x and [g, g+ny) in y. The x-face flux uses
    # faces i and i+1 (both valid for interior i); the y-face flux uses faces
    # j and j+1.
    dbx = (bx[g:g + ny, g + 1:g + 1 + nx] - bx[g:g + ny, g:g + nx]) / dx
    dby = (by[g + 1:g + 1 + ny, g:g + nx] - by[g:g + ny, g:g + nx]) / dy
    div = dbx + dby
    return float(np.max(np.abs(div))) if div.size else 0.0


def fast_magnetosonic_speed(rho, p, bx, by, bz, gamma):
    """Maximum (degenerate-direction) fast magnetosonic speed.

    A conservative Python-side estimate used only for sanity/documentation; the
    CLI's CFL guard uses the C++ ``cfl_limit()`` which scans per direction. The
    fast speed satisfies c_f^2 <= a^2 + b^2 where a^2 = gamma*p/rho is the sound
    speed and b^2 = |B|^2/rho is the Alfven speed, so this upper bound is safe.
    """
    rho = np.asarray(rho, dtype=np.float64)
    a2 = gamma * np.asarray(p, dtype=np.float64) / rho
    b2 = (np.asarray(bx) ** 2 + np.asarray(by) ** 2 + np.asarray(bz) ** 2) / rho
    return np.sqrt(a2 + b2)
