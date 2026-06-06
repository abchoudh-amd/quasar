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
