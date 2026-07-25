"""Shared numerical helpers for the PIC front-end.

Single source of truth for the CFL timestep so the CLI and the example deck
generators cannot drift apart (they previously each hardcoded the formula).
"""

from __future__ import annotations

import math

import numpy as np

C_LIGHT = 299792458.0

# First several positive zeros of the Bessel function J0, j_{0,n}.  These make
# the common low modes exact to binary64 precision.  Higher zeros are evaluated
# with the convergent-in-practice McMahon asymptotic expansion below, keeping the
# front-end dependency-free (numpy + yaml only, no scipy).
J0_ZEROS = (
    2.404825557695773,
    5.520078110286311,
    8.653727912911012,
    11.791534439014281,
    14.930917708487785,
)


def besselj0(x: np.ndarray | float) -> np.ndarray:
    """Bessel function of the first kind J0, dependency-free.

    Abramowitz & Stegun 9.4.1 (|x| <= 3) and 9.4.3 (|x| > 3) polynomial
    approximations; max abs error ~1.5e-8, ample for seeding an initial-field
    profile. Vectorized over a NumPy array (or scalar) input so it drops in for
    ``scipy.special.j0`` on the cylindrical seed path without pulling in scipy.
    """
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)

    # |x| <= 3: rational polynomial in (x/3)^2 (A&S 9.4.1).
    y_small = (x / 3.0) ** 2
    p_small = (
        1.0
        + y_small * (-2.2499997
        + y_small * (1.2656208
        + y_small * (-0.3163866
        + y_small * (0.0444479
        + y_small * (-0.0039444
        + y_small * 0.0002100)))))
    )

    # |x| > 3: amplitude * cos(phase) asymptotic form (A&S 9.4.3).
    z = 3.0 / np.where(ax == 0.0, 1.0, ax)  # avoid 0-division; masked out below
    f0 = (
        0.79788456
        + z * (-0.00000077
        + z * (-0.00552740
        + z * (-0.00009512
        + z * (0.00137237
        + z * (-0.00072805
        + z * 0.00014476)))))
    )
    theta = (
        ax - 0.78539816
        + z * (-0.04166397
        + z * (-0.00003954
        + z * (0.00262573
        + z * (-0.00054125
        + z * (-0.00029333
        + z * 0.00013558)))))
    )
    p_large = f0 * np.cos(theta) / np.sqrt(np.where(ax == 0.0, 1.0, ax))

    return np.where(ax <= 3.0, p_small, p_large)


def j0_zero(n: int) -> float:
    """The n-th positive zero of J0 (1-indexed): j0_zero(1) ~= 2.4048."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("J0 zero index must be a positive integer")
    if n <= len(J0_ZEROS):
        return J0_ZEROS[n - 1]

    # DLMF 10.21.19 (McMahon expansion) for nu=0, with
    # beta=(n-1/4)*pi.  Starting at n=6, the first omitted beta^-7 term is
    # below 3e-9 in absolute value and rapidly decreases; this is comfortably
    # below the ~1.5e-8 approximation error of besselj0 used to build the seed.
    beta = (float(n) - 0.25) * math.pi
    beta2 = beta * beta
    return beta + (1.0 / beta) * (
        1.0 / 8.0
        + (1.0 / beta2) * (
            -31.0 / 384.0
            + (1.0 / beta2) * (3779.0 / 15360.0)))


def _scaled_quotient3(numerator: float, denominator_a: float,
                      denominator_b: float) -> float:
    """Evaluate numerator/(denominator_a*denominator_b) by exponents."""
    mn, en = math.frexp(numerator)
    ma, ea = math.frexp(denominator_a)
    mb, eb = math.frexp(denominator_b)
    mantissa, adjustment = math.frexp((mn / ma) / mb)
    try:
        return math.ldexp(mantissa, en - ea - eb + adjustment)
    except OverflowError as exc:
        raise OverflowError("CFL timestep is not representable") from exc


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
    dx = float(dx)
    dy = float(dy)
    c = float(c)
    if not (math.isfinite(dx) and dx > 0.0
            and math.isfinite(dy) and dy > 0.0):
        raise ValueError("dx and dy must be finite and positive")
    if not (math.isfinite(c) and c > 0.0):
        raise ValueError("c must be finite and positive")
    # Stable form of 1/(c*factor*hypot(1/dx,1/dy)).  The direct expression
    # squares inverse spacings and can overflow even when the CFL step itself is
    # representable.
    h_min = min(dx, dy)
    h_max = max(dx, dy)
    ratio = h_min / h_max
    spectral_factor = factor * math.sqrt(1.0 + ratio * ratio)
    dt = _scaled_quotient3(h_min, spectral_factor, c)
    if not (math.isfinite(dt) and dt > 0.0):
        raise OverflowError("CFL timestep is not representable")
    return dt


def cfl_dt(dx: float, dy: float, c: float = C_LIGHT, fdtd_order: int = 2,
           safety: float = 0.5) -> float:
    """A safe timestep: the CFL limit scaled by ``safety`` (default 0.5)."""
    safety = float(safety)
    if not (math.isfinite(safety) and 0.0 < safety <= 1.0):
        raise ValueError("safety must be finite and in (0, 1]")
    dt = safety * cfl_limit(dx, dy, c, fdtd_order)
    if not (math.isfinite(dt) and dt > 0.0):
        raise OverflowError("safe CFL timestep is not representable")
    return dt


def cyl_cfl_limit(dr: float, dz: float, c: float = C_LIGHT,
                  fdtd_order: int = 2) -> float:
    """The axisymmetric (r-z, m=0) Yee CFL stability limit.

    For the m=0 azimuthal mode the conservative (volume-weighted) curl operator
    is mimetic: the small on-axis cell volume cancels the apparent 1/r
    amplification, so the spectral radius — and hence the stability bound — is
    exactly the planar Yee limit of the same order with (dx, dy) -> (dr, dz). This is the
    geometry-named selector (mirroring C++ ``quasar::cyl_cfl_dt``); it delegates
    to :func:`cfl_limit` so the formula lives in one place.
    """
    return cfl_limit(dr, dz, c, fdtd_order=fdtd_order)


def cyl_cfl_dt(dr: float, dz: float, c: float = C_LIGHT,
               safety: float = 0.5, fdtd_order: int = 2) -> float:
    """A safe axisymmetric timestep: ``cyl_cfl_limit`` scaled by ``safety``."""
    safety = float(safety)
    if not (math.isfinite(safety) and 0.0 < safety <= 1.0):
        raise ValueError("safety must be finite and in (0, 1]")
    dt = safety * cyl_cfl_limit(dr, dz, c, fdtd_order=fdtd_order)
    if not (math.isfinite(dt) and dt > 0.0):
        raise OverflowError("safe cylindrical CFL timestep is not representable")
    return dt
