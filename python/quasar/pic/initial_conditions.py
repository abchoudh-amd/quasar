"""Small host-side initial-condition helpers for PIC examples."""

from __future__ import annotations

import math
import operator

import numpy as np


def _particle_count(value: int, *, allow_zero: bool) -> int:
    """Return an exact integer particle count without silently truncating."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("n_particles must be an integer, not a boolean")
    try:
        count = operator.index(value)
    except TypeError as exc:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"n_particles must be a {qualifier} integer") from exc
    if count < 0 or (count == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"n_particles must be {qualifier}")
    if count > np.iinfo(np.intp).max:
        raise ValueError("n_particles exceeds the platform array-index limit")
    return count


def _extent(lower: float, upper: float, count: int, label: str) -> float:
    """Validate a finite, resolvable interval and return its width."""
    width = upper - lower
    if not (math.isfinite(width) and width > 0.0):
        raise ValueError(f"quiet-start {label} extent is not representable")
    try:
        half_cell = 0.5 * (width / count)
    except OverflowError as exc:
        raise ValueError(
            f"quiet-start {label} strata are not representable") from exc
    if not (math.isfinite(half_cell) and half_cell > 0.0):
        raise ValueError(f"quiet-start {label} strata are not representable")
    if lower + half_cell == lower or upper - half_cell == upper:
        raise ValueError(
            f"quiet-start {label} strata collapse in floating-point precision")
    return width


def _scaled_product_over(factors: tuple[float, ...], divisor: int,
                         label: str) -> float:
    """Evaluate a positive product/division without false intermediate range errors."""
    mantissa = 1.0
    exponent = 0
    for factor in factors:
        if not (math.isfinite(factor) and factor > 0.0):
            raise ValueError(f"{label} is not representable")
        part, part_exponent = math.frexp(factor)
        mantissa *= part
        exponent += part_exponent
        mantissa, adjustment = math.frexp(mantissa)
        exponent += adjustment
    try:
        divisor_mantissa, divisor_exponent = math.frexp(float(divisor))
    except OverflowError as exc:
        raise ValueError(f"{label} is not representable") from exc
    mantissa /= divisor_mantissa
    exponent -= divisor_exponent
    mantissa, adjustment = math.frexp(mantissa)
    exponent += adjustment
    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError as exc:
        raise ValueError(f"{label} is not representable") from exc
    if not (math.isfinite(result) and result > 0.0):
        raise ValueError(f"{label} is not representable")
    return result


def maxwellian(n_particles: int, thermal_speed: float, drift=(0.0, 0.0, 0.0), seed: int = 0):
    """Antithetic (quiet) Maxwellian velocity sample.

    Every random draw is paired with its negative, so the thermal sample has
    exactly zero bulk momentum; an odd population receives one zero-thermal
    particle. The requested drift is then added exactly.
    """
    n_particles = _particle_count(n_particles, allow_zero=True)
    thermal_speed = float(thermal_speed)
    drift_arr = np.asarray(drift, dtype=float)
    if not (math.isfinite(thermal_speed) and thermal_speed >= 0.0):
        raise ValueError("thermal_speed must be finite and non-negative")
    if drift_arr.shape != (3,) or not np.all(np.isfinite(drift_arr)):
        raise ValueError("drift must contain exactly three finite components")
    rng = np.random.default_rng(seed)
    n_pairs = n_particles // 2
    # NumPy accepts every finite scale/drift, but their sampled product or sum
    # can still overflow. Suppress the low-level warning and convert that case
    # into the same deterministic validation error as any other invalid sample.
    with np.errstate(over="ignore", invalid="ignore"):
        draws = rng.normal(0.0, thermal_speed, size=(n_pairs, 3))
        # Interleave pairs so the x-stratified position layout does not place all
        # positive draws in one half of the domain and all negatives in the other.
        v = np.empty((n_particles, 3), dtype=float)
        v[0:2 * n_pairs:2] = draws
        v[1:2 * n_pairs:2] = -draws
        if n_particles % 2:
            v[-1] = 0.0
        v += drift_arr
    if not np.all(np.isfinite(v)):
        raise ValueError(
            "thermal_speed and drift produce non-finite Maxwellian velocities")
    return v


def _block_side(n_particles: int) -> int:
    n = _particle_count(n_particles, allow_zero=False)
    return int(np.ceil(np.sqrt(n)))


def _validate_block(n_particles: int, x_min: float, x_max: float,
                    y_min: float, y_max: float) -> tuple[int, float, float, float, float]:
    n = _particle_count(n_particles, allow_zero=False)
    bounds = tuple(float(v) for v in (x_min, x_max, y_min, y_max))
    if not all(math.isfinite(v) for v in bounds):
        raise ValueError("quiet-start bounds must be finite")
    xmin, xmax, ymin, ymax = bounds
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("quiet-start lower bounds must be smaller than upper bounds")
    _extent(xmin, xmax, n, "x")
    _extent(ymin, ymax, n, "y")
    return n, xmin, xmax, ymin, ymax


def quiet_positions_2d_block(n_particles: int, x_min: float, x_max: float,
                              y_min: float, y_max: float):
    # Latin-hypercube rank-1 lattice.  Both coordinates use every midpoint
    # stratum exactly once, giving the exact block centre of mass for every N.
    n_particles, x_min, x_max, y_min, y_max = _validate_block(
        n_particles, x_min, x_max, y_min, y_max)
    k = np.arange(n_particles, dtype=float)
    u = (k + 0.5) / n_particles
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    stride = max(1, int(round(golden * n_particles)))
    while math.gcd(stride, n_particles) != 1:
        stride += 1
    ranks = (stride * np.arange(n_particles, dtype=np.int64)) % n_particles
    v = (ranks.astype(float) + 0.5) / n_particles
    # _validate_block has already required each difference to be finite. Adding
    # a fraction of that local extent to the lower bound retains strata on a
    # large translated domain; the algebraically equivalent convex combination
    # forms two origin-sized products and loses most of their separation.
    x = x_min + u * (x_max - x_min)
    y = y_min + v * (y_max - y_min)
    pts = np.column_stack((x, y))
    if not np.all(np.isfinite(pts)):
        raise ValueError("quiet-start bounds produce non-finite coordinates")
    return pts


def quiet_positions_rz_block(n_particles: int, r_min: float, r_max: float,
                             z_min: float, z_max: float):
    """Equal-volume quiet start for an axisymmetric annular block.

    Uniform volume density requires uniform sampling in r^2, not r.
    """
    n_particles, r_min, r_max, z_min, z_max = _validate_block(
        n_particles, r_min, r_max, z_min, z_max)
    if r_min < 0.0:
        raise ValueError("cylindrical quiet-start radius must be non-negative")
    uv = quiet_positions_2d_block(n_particles, 0.0, 1.0, 0.0, 1.0)
    # Uniformity in r^2 means r^2-r_min^2 = u*(r_max^2-r_min^2).
    # Evaluate the square root after scaling by r_max, then rationalize
    # r-r_min. This avoids r^2 overflow and, for a thin annulus at a large
    # radius, retains (r_max-r_min) even when r_min/r_max rounds to one.
    radial_extent = r_max - r_min
    ratio = r_min / r_max
    sum_factor = 1.0 + ratio
    scaled_increment = ((radial_extent / r_max) * sum_factor * uv[:, 0])
    scaled_radius = np.sqrt(ratio * ratio + scaled_increment)
    radial_offset = (radial_extent * sum_factor * uv[:, 0]
                     / (scaled_radius + ratio))
    r = r_min + radial_offset
    z = z_min + uv[:, 1] * (z_max - z_min)
    pts = np.column_stack((r, z))
    if not np.all(np.isfinite(pts)):
        raise ValueError("cylindrical quiet-start bounds produce non-finite coordinates")
    return pts


def quiet_block_cell_area(n_particles: int, x_min: float, x_max: float,
                          y_min: float, y_max: float) -> float:
    """Area represented by each equal-weight quiet-start particle."""
    n_particles, x_min, x_max, y_min, y_max = _validate_block(
        n_particles, x_min, x_max, y_min, y_max)
    return _scaled_product_over(
        (x_max - x_min, y_max - y_min), n_particles,
        "quiet-start area per particle")


def quiet_block_ring_volume(n_particles: int, r_min: float, r_max: float,
                            z_min: float, z_max: float) -> float:
    """Axisymmetric ring volume represented by each equal-volume particle."""
    n_particles, r_min, r_max, z_min, z_max = _validate_block(
        n_particles, r_min, r_max, z_min, z_max)
    if r_min < 0.0:
        raise ValueError("cylindrical quiet-start radius must be non-negative")
    # (r_max^2-r_min^2) is evaluated as a scaled product of the difference,
    # r_max, and (1+r_min/r_max).  This is the factored difference-of-squares
    # formula without either catastrophic cancellation or a false intermediate
    # overflow in r_max+r_min.
    radial_difference = r_max - r_min
    radial_sum_factor = 1.0 + r_min / r_max
    return _scaled_product_over(
        (math.pi, radial_difference, r_max, radial_sum_factor,
         z_max - z_min),
        n_particles, "quiet-start ring volume")


def quiet_positions_2d(n_particles: int, lx: float, ly: float):
    """Quiet-start grid over the whole [0, lx) x [0, ly) domain.

    Special case of quiet_positions_2d_block."""
    return quiet_positions_2d_block(n_particles, 0.0, lx, 0.0, ly)
