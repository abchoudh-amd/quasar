"""Magnetic-unit conversion for the ideal-MHD solver.

The C++ equations use the common ``mu0 = 1`` form.  An SI mechanical state can
be used without choosing arbitrary reference scales by storing
``B_internal = B_SI / sqrt(mu0)``.  Then ``B_internal**2`` has pressure units,
the Maxwell stress and magnetic energy are in pascals/J m^-3, and the computed
Alfven speed is the SI value.  Length, time, density, velocity, pressure,
momentum, and energy need no conversion.
"""

from __future__ import annotations

import math

import numpy as np

MU0 = 4.0e-7 * math.pi
SQRT_MU0 = math.sqrt(MU0)


def magnetic_to_internal(value, units: str):
    """Tesla -> sqrt(Pa) for SI decks; identity for normalized decks."""
    arr = np.asarray(value, dtype=np.float64)
    return arr / SQRT_MU0 if units == "SI" else arr


def magnetic_to_output(value, units: str):
    """sqrt(Pa) -> Tesla for SI output; identity for normalized output."""
    arr = np.asarray(value, dtype=np.float64)
    return arr * SQRT_MU0 if units == "SI" else arr


def vector_potential_to_internal(value, units: str):
    """T m -> sqrt(Pa) m for SI vector-potential input."""
    return magnetic_to_internal(value, units)
