"""Shared numerical helpers for the MHD front-end.

The C++ ``MhdSolver2D::cfl_limit()`` is the authoritative stable-step bound (it
scans the seeded state for the max fast-magnetosonic signal speed). These Python
helpers cover the small primitive<->conserved conversions the deck IC generators
and the CLI need, kept in one place so they cannot drift between io.py and cli.py
(mirrors ``quasar.pic.numerics`` keeping the CFL formula in a single module).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np


# Scale-free acceptance envelope shared with the native live/background
# solenoidality preflights.  The factor covers a short compensated stencil plus
# profile sampling roundoff while remaining an O(machine-epsilon) criterion.
DISCRETE_SOLENOIDAL_TOLERANCE = 1024.0 * np.finfo(np.float64).eps


def _integrate_polynomial_on_centered_cell(coefficients, power=0):
    """Integrate ``t**power * p(t)`` exactly over ``[-1/2, 1/2]``."""
    lower = Fraction(-1, 2)
    upper = Fraction(1, 2)
    total = Fraction(0)
    for degree, coefficient in enumerate(coefficients):
        exponent = degree + power + 1
        total += coefficient * (upper ** exponent - lower ** exponent) / exponent
    return total


def _generate_face_to_cell_rows():
    """Generate centered finite-volume rows from one exact definition.

    For each supported width, construct the Lagrange basis through the centered
    face nodes and integrate it exactly. The Cartesian coefficient is
    ``integral L_k dt``. The cylindrical correction is ``integral t L_k dt``;
    a ring average centered at dimensionless radius ``rho`` is consequently
    ``cartesian + correction/rho``.

    This is the Python source of truth for both row families. It replaces
    transcribed rational tables while expressing the same polynomial-moment
    definition as native ``solve_radial_row``. Cross-language golden tests pin
    their binary64 agreement.
    """
    cartesian = {}
    radial_correction = {}
    for width in (2, 4, 6, 8):
        nodes = [
            Fraction(2 * index - width + 1, 2)
            for index in range(width)
        ]
        cartesian_row = []
        correction_row = []
        for index, node in enumerate(nodes):
            # Ascending coefficients of the Lagrange basis polynomial L_index.
            coefficients = [Fraction(1)]
            denominator = Fraction(1)
            for other_index, other_node in enumerate(nodes):
                if other_index == index:
                    continue
                expanded = [Fraction(0)] * (len(coefficients) + 1)
                for degree, coefficient in enumerate(coefficients):
                    expanded[degree] -= other_node * coefficient
                    expanded[degree + 1] += coefficient
                coefficients = expanded
                denominator *= node - other_node
            coefficients = [value / denominator for value in coefficients]
            cartesian_row.append(
                _integrate_polynomial_on_centered_cell(coefficients))
            correction_row.append(
                _integrate_polynomial_on_centered_cell(coefficients, power=1))

        cartesian[width] = np.asarray(cartesian_row, dtype=np.float64)
        radial_correction[width] = np.asarray(
            correction_row, dtype=np.float64)
    return cartesian, radial_correction


(_FACE_TO_CELL_CENTERED,
 _RADIAL_FACE_TO_CELL_CORRECTION) = _generate_face_to_cell_rows()


def _collocation_width(extent, nghost):
    requested = 8 if nghost >= 4 else 6 if nghost >= 3 else 4 if nghost >= 2 else 2
    for width in (8, 6, 4, 2):
        if requested >= width and extent >= width:
            return width
    return 1


def _one_sided_face_integral_weights(
        width, relative_cell, radial_edge_rho=None):
    """Weights for an outer-ghost polynomial integral over one logical cell.

    ``radial_edge_rho`` is the target cell's low-face radius in units of the
    radial spacing. When supplied, the Gauss weights include ``abs(r)`` and are
    renormalized to a ring average, matching the native one-sided closure.
    """
    xq = np.array([0.0694318442029737124, 0.3300094782075718676,
                   0.6699905217924281324, 0.9305681557970262876])
    wq = np.array([0.1739274225687269287, 0.3260725774312730713,
                   0.3260725774312730713, 0.1739274225687269287])
    barycentric = {
        2: np.array([1.0, -1.0]),
        4: np.array([1.0, -3.0, 3.0, -1.0]),
        6: np.array([1.0, -5.0, 10.0, -10.0, 5.0, -1.0]),
        8: np.array([1.0, -7.0, 21.0, -35.0, 35.0, -21.0, 7.0, -1.0]),
    }[width]
    nodes = np.arange(width, dtype=np.float64)
    if radial_edge_rho is not None:
        radial_edge_rho = float(radial_edge_rho)
        if not np.isfinite(radial_edge_rho):
            raise ValueError("radial face coordinate must be finite")
        wq = wq * np.abs(radial_edge_rho + xq)
        normalization = np.sum(wq)
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise ValueError("radial cell has invalid ring volume")
        wq = wq / normalization

    result = np.zeros(width, dtype=np.float64)
    for point, weight in zip(xq, wq):
        terms = barycentric / (float(relative_cell) + point - nodes)
        result += weight * terms / np.sum(terms)
    return result


def _radial_face_to_cell_centered_weights(width, rho):
    """Return the centered R4 row for a target cell-center radius ``rho``."""
    rho = float(rho)
    if not np.isfinite(rho) or rho == 0.0:
        raise ValueError("radial cell-center coordinate must be finite and nonzero")
    try:
        weights = (_FACE_TO_CELL_CENTERED[width]
                   + _RADIAL_FACE_TO_CELL_CORRECTION[width] / rho)
    except KeyError:
        raise ValueError("radial collocation width must be 2, 4, 6, or 8") \
            from None
    weights = np.asarray(weights, dtype=np.float64).copy()
    # Match the native coefficient engine's binary64 partition-of-unity
    # invariant under an in-order reduction.
    weights[-1] = 1.0 - np.sum(weights[:-1])
    return weights


def face_samples_to_cell_average(values, axis, nghost):
    """Collocate staggered face samples as finite-volume cell averages.

    ``axis=1`` integrates x-face samples across each array row; ``axis=0``
    integrates y-face samples down each column. The order follows the solver's
    halo convention (4th/6th/8th for two/three/four ghosts), including an
    order-matched one-sided polynomial closure for the outermost ghost cells.
    This mirrors ``quasar/physics/mhd/mhd_staggering.hpp``.
    """
    arr = _finite_array(values, "face samples")
    if arr.ndim != 2:
        raise ValueError("face samples must be a two-dimensional array")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    if isinstance(nghost, (bool, np.bool_)) or int(nghost) != nghost:
        raise ValueError("nghost must be an integer")
    g = int(nghost)
    if g < 0:
        raise ValueError("nghost must be non-negative")

    extent = arr.shape[axis]
    width = _collocation_width(extent, g)
    if width == 1:
        return arr.copy()
    centered = _FACE_TO_CELL_CENTERED[width]
    out = np.empty_like(arr)
    for q in range(extent):
        start = q - (width // 2 - 1)
        if 0 <= start and start + width <= extent:
            weights = centered
        else:
            start = min(max(start, 0), extent - width)
            weights = _one_sided_face_integral_weights(width, q - start)
        slab = np.take(arr, np.arange(start, start + width), axis=axis)
        out_index = [slice(None), slice(None)]
        out_index[axis] = q
        out[tuple(out_index)] = np.tensordot(
            weights, np.moveaxis(slab, axis, 0), axes=(0, 0))
    if not np.all(np.isfinite(out)):
        raise ValueError("face-to-cell magnetic collocation is not representable")
    return out


def radial_face_samples_to_cell_average(
        values, nghost, origin_x, dr, scheme_order=None):
    """Collocate radial-face samples into cylindrical ring averages.

    ``values`` uses the solver's padded two-dimensional storage layout, with
    radial faces along array axis 1. ``origin_x`` is the physical low radial
    face of interior cell ``i=0`` and ``dr`` is the uniform radial spacing.
    ``scheme_order`` may be 2, 5, or 7 and selects the native R4 width (4, 6,
    or 8) independently of an overpadded halo; when omitted, the width is
    inferred from ``nghost`` for compatibility. Centered rows use the
    radius-dependent R4 finite-volume moments; outer ghost cells use the same
    one-sided polynomial closure with ``abs(r)`` folded into its Gauss weights.
    An axis at ``origin_x == 0`` is supported because the padded negative-radius
    parity cells use the ``abs(r)`` measure.
    """
    arr = _finite_array(values, "radial face samples")
    if arr.ndim != 2:
        raise ValueError("radial face samples must be a two-dimensional array")
    if isinstance(nghost, (bool, np.bool_)) or int(nghost) != nghost:
        raise ValueError("nghost must be an integer")
    g = int(nghost)
    if g < 0:
        raise ValueError("nghost must be non-negative")
    origin_x = float(origin_x)
    dr = float(dr)
    if not np.isfinite(origin_x) or origin_x < 0.0:
        raise ValueError("origin_x must be finite and non-negative")
    if not np.isfinite(dr) or dr <= 0.0:
        raise ValueError("dr must be finite and positive")
    origin_rho = origin_x / dr
    if not np.isfinite(origin_rho):
        raise ValueError("dimensionless radial origin is not representable")

    extent = arr.shape[1]
    if scheme_order is None:
        width = _collocation_width(extent, g)
    else:
        if isinstance(scheme_order, (bool, np.bool_)):
            raise ValueError("scheme_order must be 2, 5, or 7")
        try:
            order = int(scheme_order)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("scheme_order must be 2, 5, or 7") from None
        if order != scheme_order or order not in (2, 5, 7):
            raise ValueError("scheme_order must be 2, 5, or 7")
        width = {2: 4, 5: 6, 7: 8}[order]
        if extent < width:
            raise ValueError(
                "radial sample extent is smaller than the scheme stencil")
    if width == 1:
        return arr.copy()
    out = np.empty_like(arr)
    for q in range(extent):
        radial_edge_rho = origin_rho + float(q - g)
        radial_high_rho = radial_edge_rho + 1.0
        if radial_edge_rho < 0.0 < radial_high_rho:
            raise ValueError(
                "a cylindrical cell may not straddle the radial axis")
        rho = radial_edge_rho + 0.5
        if rho == 0.0:
            raise ValueError(
                "a cylindrical cell may not be centered on the radial axis")

        start = q - (width // 2 - 1)
        if 0 <= start and start + width <= extent:
            weights = _radial_face_to_cell_centered_weights(width, rho)
        else:
            start = min(max(start, 0), extent - width)
            weights = _one_sided_face_integral_weights(
                width, q - start, radial_edge_rho)
        slab = arr[:, start:start + width]
        out[:, q] = np.tensordot(weights, slab, axes=(0, 1))
    if not np.all(np.isfinite(out)):
        raise ValueError("radial face-to-cell collocation is not representable")
    return out


def _finite_array(value, name):
    try:
        source = np.asarray(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"{name} must contain only real floating-point or integer values") \
            from None
    # Match the strict real-array contract at the native bindings: accept real
    # floats and signed/unsigned integers, but never silently reinterpret bool,
    # complex, object, text, void, datetime, or timedelta values as physics data.
    if source.dtype.kind not in "fiu":
        raise ValueError(
            f"{name} must contain only real floating-point or integer values")
    with np.errstate(over="ignore", invalid="ignore"):
        arr = np.asarray(source, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def half_squared_norm3(x, y, z):
    """Return ``0.5*(x*x+y*y+z*z)`` without premature overflow.

    Scaling by the largest component mirrors the C++ MHD implementation. The
    result can still be infinite when the mathematical energy is not
    representable in float64, but no intermediate square overflows first.
    """
    x, y, z = np.broadcast_arrays(
        _finite_array(x, "x"), _finite_array(y, "y"), _finite_array(z, "z"))
    scale = np.maximum(np.abs(x), np.maximum(np.abs(y), np.abs(z)))
    xs = np.divide(x, scale, out=np.zeros_like(x), where=scale != 0.0)
    ys = np.divide(y, scale, out=np.zeros_like(y), where=scale != 0.0)
    zs = np.divide(z, scale, out=np.zeros_like(z), where=scale != 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        return (0.5 * scale) * scale * (xs * xs + ys * ys + zs * zs)


def _kinetic_from_velocity(rho, vx, vy, vz):
    rho, vx, vy, vz = np.broadcast_arrays(
        _finite_array(rho, "rho"), _finite_array(vx, "vx"),
        _finite_array(vy, "vy"), _finite_array(vz, "vz"))
    if np.any(rho <= 0.0):
        raise ValueError("rho must be positive")
    scale = np.maximum(np.abs(vx), np.maximum(np.abs(vy), np.abs(vz)))
    xs = np.divide(vx, scale, out=np.zeros_like(vx), where=scale != 0.0)
    ys = np.divide(vy, scale, out=np.zeros_like(vy), where=scale != 0.0)
    zs = np.divide(vz, scale, out=np.zeros_like(vz), where=scale != 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        q = np.sqrt(rho) * scale
        return (0.5 * q) * q * (xs * xs + ys * ys + zs * zs)


def primitive_to_energy(rho, p, vx, vy, vz, bx, by, bz, gamma):
    """Total energy density from primitives, the ideal-MHD conserved energy::

        E = p/(gamma - 1) + 0.5*rho*(vx^2+vy^2+vz^2) + 0.5*(bx^2+by^2+bz^2)

    This is the single source of truth for the p->E conversion every IC builder
    uses; it must match the C++ EOS (E = rho*e + 0.5*rho*v^2 + 0.5*B^2 with
    rho*e = p/(gamma-1)). Works elementwise on NumPy arrays or scalars.
    """
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than one")
    pressure = _finite_array(p, "p")
    if np.any(pressure <= 0.0):
        raise ValueError("p must be positive")
    kinetic = _kinetic_from_velocity(rho, vx, vy, vz)
    magnetic = half_squared_norm3(bx, by, bz)
    with np.errstate(over="ignore", invalid="ignore"):
        energy = pressure / (gamma - 1.0) + kinetic + magnetic
    if not np.all(np.isfinite(energy)):
        raise ValueError("primitive state energy is not representable in float64")
    return energy


def momentum(rho, vx, vy, vz):
    """Conserved momentum density (rho*v) per component."""
    rho, vx, vy, vz = np.broadcast_arrays(
        _finite_array(rho, "rho"), _finite_array(vx, "vx"),
        _finite_array(vy, "vy"), _finite_array(vz, "vz"))
    if np.any(rho <= 0.0):
        raise ValueError("rho must be positive")
    with np.errstate(over="ignore", invalid="ignore"):
        result = rho * vx, rho * vy, rho * vz
    if any(not np.all(np.isfinite(comp)) for comp in result):
        raise ValueError("momentum is not representable in float64")
    return result


def conserved_to_pressure(rho, mx, my, mz, energy, bx, by, bz, gamma):
    """Recover gas pressure from a conserved, cell-collocated MHD state.

    This mirrors the native EOS exactly::

        p = (gamma - 1) * ((E - |m|^2/(2 rho)) - |B|^2/2)

    The sequential subtraction and scaled quadratic forms matter near the edge
    of float64's exponent range.  Inputs must be finite and density strictly
    positive; the returned pressure may be non-positive so callers can diagnose
    an inadmissible state explicitly.
    """
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than one")
    rho, mx, my, mz, energy, bx, by, bz = np.broadcast_arrays(
        _finite_array(rho, "rho"), _finite_array(mx, "mx"),
        _finite_array(my, "my"), _finite_array(mz, "mz"),
        _finite_array(energy, "energy"), _finite_array(bx, "bx"),
        _finite_array(by, "by"), _finite_array(bz, "bz"))
    if np.any(rho <= 0.0):
        raise ValueError("rho must be positive")

    momentum_scale = np.maximum(np.abs(mx), np.maximum(np.abs(my), np.abs(mz)))
    mxs = np.divide(mx, momentum_scale, out=np.zeros_like(mx),
                    where=momentum_scale != 0.0)
    mys = np.divide(my, momentum_scale, out=np.zeros_like(my),
                    where=momentum_scale != 0.0)
    mzs = np.divide(mz, momentum_scale, out=np.zeros_like(mz),
                    where=momentum_scale != 0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        q = momentum_scale / np.sqrt(rho)
        kinetic = (0.5 * q) * q * (mxs * mxs + mys * mys + mzs * mzs)
        magnetic = half_squared_norm3(bx, by, bz)
        pressure = (gamma - 1.0) * ((energy - kinetic) - magnetic)
    if not np.all(np.isfinite(pressure)):
        raise ValueError("conserved-state pressure is not representable in float64")
    return pressure


def _validated_staggered_inputs(b0x, b0y, nx, ny, nghost, dx, dy):
    for value, name in ((nx, "nx"), (ny, "ny"), (nghost, "nghost")):
        if isinstance(value, (bool, np.bool_)) or int(value) != value:
            raise ValueError(f"{name} must be an integer")
    nx, ny, g = int(nx), int(ny), int(nghost)
    if nx < 0 or ny < 0 or g < 0:
        raise ValueError("nx, ny, and nghost must be non-negative")
    # A nonempty staggered grid needs the high face (i+1/j+1), and the corner
    # curl additionally needs the low neighbour (i-1/j-1). One halo cell is the
    # minimum storage that contains both; g=0 would otherwise produce truncated
    # or wraparound NumPy slices rather than a physical diagnostic.
    if nx > 0 and ny > 0 and g < 1:
        raise ValueError(
            "nghost must be at least 1 for staggered face/corner diagnostics")
    dx, dy = float(dx), float(dy)
    if not np.isfinite(dx) or dx <= 0.0 or not np.isfinite(dy) or dy <= 0.0:
        raise ValueError("dx and dy must be finite and positive")
    pitch = nx + 2 * g
    height = ny + 2 * g
    expected = pitch * height
    bx = _finite_array(b0x, "b0x")
    by = _finite_array(b0y, "b0y")
    if bx.size != expected or by.size != expected:
        raise ValueError(
            f"staggered field size must be {expected} for the supplied grid")
    return bx.reshape(height, pitch), by.reshape(height, pitch), nx, ny, g, dx, dy


_SCALED_ZERO_EXPONENT = np.int64(-1_000_000)


def _scaled_two_sum(lhs_mantissa, lhs_exponent,
                    rhs_mantissa, rhs_exponent):
    """Exact two-sum for finite values stored as ``mantissa * 2**exponent``."""
    rhs_larger = (
        (rhs_exponent > lhs_exponent)
        | ((rhs_exponent == lhs_exponent)
           & (np.abs(rhs_mantissa) > np.abs(lhs_mantissa))))
    larger_mantissa = np.where(
        rhs_larger, rhs_mantissa, lhs_mantissa)
    larger_exponent = np.where(
        rhs_larger, rhs_exponent, lhs_exponent)
    smaller_mantissa = np.where(
        rhs_larger, lhs_mantissa, rhs_mantissa)
    smaller_exponent = np.where(
        rhs_larger, lhs_exponent, rhs_exponent)

    gap = larger_exponent - smaller_exponent
    nonoverlapping = gap > 54
    # A gap this wide is already an exact two-component expansion. Avoid
    # materializing its tiny aligned component merely to underflow it.
    aligned_gap = np.where(nonoverlapping, 0, gap)
    with np.errstate(under="ignore"):
        smaller = np.ldexp(smaller_mantissa, -aligned_gap)
    rounded = larger_mantissa + smaller
    virtual_smaller = rounded - larger_mantissa
    error = ((larger_mantissa - (rounded - virtual_smaller))
             + (smaller - virtual_smaller))

    high_mantissa, high_shift = np.frexp(rounded)
    high_exponent = larger_exponent + high_shift.astype(np.int64)
    low_mantissa, low_shift = np.frexp(error)
    low_exponent = larger_exponent + low_shift.astype(np.int64)

    high_mantissa = np.where(
        nonoverlapping, larger_mantissa, high_mantissa)
    high_exponent = np.where(
        nonoverlapping, larger_exponent, high_exponent)
    low_mantissa = np.where(
        nonoverlapping, smaller_mantissa, low_mantissa)
    low_exponent = np.where(
        nonoverlapping, smaller_exponent, low_exponent)

    smaller_is_zero = smaller_mantissa == 0.0
    high_mantissa = np.where(
        smaller_is_zero, larger_mantissa, high_mantissa)
    high_exponent = np.where(
        smaller_is_zero, larger_exponent, high_exponent)
    low_mantissa = np.where(smaller_is_zero, 0.0, low_mantissa)
    high_exponent = np.where(
        high_mantissa == 0.0, _SCALED_ZERO_EXPONENT, high_exponent)
    low_exponent = np.where(
        low_mantissa == 0.0, _SCALED_ZERO_EXPONENT, low_exponent)
    return high_mantissa, high_exponent, low_mantissa, low_exponent


def _reduce_scaled_terms_to_scaled(mantissa, exponent):
    """Correctly round a short scaled sum without losing cancellation bits.

    This is the vectorized NumPy port of native
    ``reduce_scaled_terms_to_value``. It grows an exact, non-overlapping
    floating-point expansion with two-sum, then applies the same half-even
    final collapse as the C++/HIP implementation.
    """
    mantissa = np.asarray(mantissa, dtype=np.float64).copy()
    exponent = np.asarray(exponent, dtype=np.int64).copy()
    count = mantissa.shape[0]
    exponent = np.where(
        mantissa == 0.0, _SCALED_ZERO_EXPONENT, exponent)

    # Stable ascending-magnitude sort, matching the native insertion sort.
    for end in range(count - 1, 0, -1):
        for k in range(end):
            swap = (
                (exponent[k] > exponent[k + 1])
                | ((exponent[k] == exponent[k + 1])
                   & (np.abs(mantissa[k]) > np.abs(mantissa[k + 1]))))
            old_mantissa = mantissa[k].copy()
            old_exponent = exponent[k].copy()
            mantissa[k] = np.where(swap, mantissa[k + 1], mantissa[k])
            exponent[k] = np.where(swap, exponent[k + 1], exponent[k])
            mantissa[k + 1] = np.where(
                swap, old_mantissa, mantissa[k + 1])
            exponent[k + 1] = np.where(
                swap, old_exponent, exponent[k + 1])

    expansion_mantissa = np.zeros_like(mantissa)
    expansion_exponent = np.full_like(exponent, _SCALED_ZERO_EXPONENT)
    lane_shape = mantissa.shape[1:]

    def scatter_where(target, index, value, mask):
        gather = index[None, ...]
        previous = np.take_along_axis(target, gather, axis=0)[0]
        np.put_along_axis(
            target, gather,
            np.where(mask, value, previous)[None, ...], axis=0)

    # Grow one exact expansion independently in every array lane.
    for source in range(count):
        carry_mantissa = mantissa[source]
        carry_exponent = exponent[source]
        next_mantissa = np.zeros_like(mantissa)
        next_exponent = np.full_like(exponent, _SCALED_ZERO_EXPONENT)
        write = np.zeros(lane_shape, dtype=np.int64)
        for k in range(count):
            (high_mantissa, high_exponent,
             low_mantissa, low_exponent) = _scaled_two_sum(
                 carry_mantissa, carry_exponent,
                 expansion_mantissa[k], expansion_exponent[k])
            retain_low = low_mantissa != 0.0
            scatter_where(next_mantissa, write, low_mantissa, retain_low)
            scatter_where(next_exponent, write, low_exponent, retain_low)
            write += retain_low.astype(np.int64)
            carry_mantissa = high_mantissa
            carry_exponent = high_exponent
        retain_carry = carry_mantissa != 0.0
        scatter_where(next_mantissa, write, carry_mantissa, retain_carry)
        scatter_where(next_exponent, write, carry_exponent, retain_carry)
        expansion_mantissa = next_mantissa
        expansion_exponent = next_exponent

    # Correctly rounded collapse of the non-overlapping expansion. The loop is
    # mask-driven because cancellation leaves a different component count in
    # each grid cell.
    expansion_count = np.sum(
        expansion_mantissa != 0.0, axis=0).astype(np.int64)
    component = np.maximum(expansion_count - 1, 0)
    gather = component[None, ...]
    scale = np.take_along_axis(
        expansion_exponent, gather, axis=0)[0]
    leading_mantissa = np.take_along_axis(
        expansion_mantissa, gather, axis=0)[0]
    leading_exponent = np.take_along_axis(
        expansion_exponent, gather, axis=0)[0]
    high = np.ldexp(leading_mantissa, leading_exponent - scale)
    low = np.zeros_like(high)
    active = component > 0
    for _ in range(count - 1):
        next_component = np.maximum(component - 1, 0)
        gather = next_component[None, ...]
        next_mantissa = np.take_along_axis(
            expansion_mantissa, gather, axis=0)[0]
        next_exponent = np.take_along_axis(
            expansion_exponent, gather, axis=0)[0]
        with np.errstate(under="ignore"):
            next_value = np.ldexp(next_mantissa, next_exponent - scale)
        updated = high + next_value
        virtual_next = updated - high
        roundoff = next_value - virtual_next
        high = np.where(active, updated, high)
        low = np.where(active, roundoff, low)
        component = np.where(active, next_component, component)
        active &= (roundoff == 0.0) & (component > 0)

    # Half-even correction used by robust fsum implementations and by native.
    correctable = (component > 0) & (low != 0.0)
    next_component = np.maximum(component - 1, 0)
    gather = next_component[None, ...]
    next_mantissa = np.take_along_axis(
        expansion_mantissa, gather, axis=0)[0]
    next_exponent = np.take_along_axis(
        expansion_exponent, gather, axis=0)[0]
    with np.errstate(under="ignore"):
        next_value = np.ldexp(next_mantissa, next_exponent - scale)
    same_sign = (((low < 0.0) & (next_value < 0.0))
                 | ((low > 0.0) & (next_value > 0.0)))
    doubled_low = 2.0 * low
    adjusted = high + doubled_low
    exact_adjustment = adjusted - high == doubled_low
    high = np.where(
        correctable & same_sign & exact_adjustment, adjusted, high)

    result_mantissa, result_shift = np.frexp(high)
    result_exponent = scale + result_shift.astype(np.int64)
    result_exponent = np.where(
        result_mantissa == 0.0, _SCALED_ZERO_EXPONENT, result_exponent)
    return result_mantissa, result_exponent


def _scaled_quotient_sum_to_scaled(numerators, denominator0, denominator1=1.0):
    """Return ``sum(numerator / denominator0 / denominator1)`` as (m, e).

    Terms stay in mantissa/exponent form until an exact floating-point expansion
    has retained every two-sum roundoff component. Thus matched enormous terms
    can cancel without erasing a smaller representable remainder.
    This is the NumPy counterpart of native ``scaled_product_quotient_sum``.
    All inputs to this private helper are already finite and denominators are
    nonzero by the public diagnostic's validation.
    """
    values = np.stack(np.broadcast_arrays(*numerators), axis=0)
    d0 = np.broadcast_to(np.asarray(denominator0, dtype=np.float64),
                         values.shape)
    d1 = np.broadcast_to(np.asarray(denominator1, dtype=np.float64),
                         values.shape)

    value_mantissa, value_exponent = np.frexp(values)
    d0_mantissa, d0_exponent = np.frexp(d0)
    d1_mantissa, d1_exponent = np.frexp(d1)
    quotient_mantissa = ((value_mantissa / d0_mantissa) / d1_mantissa)
    mantissa, shift = np.frexp(quotient_mantissa)
    exponent = (value_exponent.astype(np.int64)
                - d0_exponent.astype(np.int64)
                - d1_exponent.astype(np.int64)
                + shift.astype(np.int64))

    return _reduce_scaled_terms_to_scaled(mantissa, exponent)


def _scaled_quotient_sum(numerators, denominator0, denominator1=1.0):
    """Materialize the range-safe quotient sum when its result is representable."""
    mantissa, exponent = _scaled_quotient_sum_to_scaled(
        numerators, denominator0, denominator1)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.ldexp(mantissa, exponent)


def _scaled_value_quotient_to_scaled(value, denominator0, denominator1=1.0):
    """Divide a retained ``(mantissa, exponent)`` value without materializing."""
    value_mantissa, value_exponent = value
    d0 = np.broadcast_to(
        np.asarray(denominator0, dtype=np.float64), value_mantissa.shape)
    d1 = np.broadcast_to(
        np.asarray(denominator1, dtype=np.float64), value_mantissa.shape)
    d0_mantissa, d0_exponent = np.frexp(d0)
    d1_mantissa, d1_exponent = np.frexp(d1)
    quotient_mantissa = (
        (value_mantissa / d0_mantissa) / d1_mantissa)
    mantissa, shift = np.frexp(quotient_mantissa)
    exponent = (value_exponent.astype(np.int64)
                - d0_exponent.astype(np.int64)
                - d1_exponent.astype(np.int64)
                + shift.astype(np.int64))
    exponent = np.where(
        mantissa == 0.0, _SCALED_ZERO_EXPONENT, exponent)
    return mantissa, exponent


def _scaled_value_product_quotient_to_scaled(
        value, numerator, denominator0, denominator1=1.0):
    """Return ``value * numerator / denominator0 / denominator1`` scaled."""
    value_mantissa, value_exponent = value
    numerator = np.broadcast_to(
        np.asarray(numerator, dtype=np.float64), value_mantissa.shape)
    d0 = np.broadcast_to(
        np.asarray(denominator0, dtype=np.float64), value_mantissa.shape)
    d1 = np.broadcast_to(
        np.asarray(denominator1, dtype=np.float64), value_mantissa.shape)
    numerator_mantissa, numerator_exponent = np.frexp(numerator)
    d0_mantissa, d0_exponent = np.frexp(d0)
    d1_mantissa, d1_exponent = np.frexp(d1)
    quotient_mantissa = (
        (value_mantissa * numerator_mantissa)
        / d0_mantissa / d1_mantissa)
    mantissa, shift = np.frexp(quotient_mantissa)
    exponent = (value_exponent.astype(np.int64)
                + numerator_exponent.astype(np.int64)
                - d0_exponent.astype(np.int64)
                - d1_exponent.astype(np.int64)
                + shift.astype(np.int64))
    exponent = np.where(
        mantissa == 0.0, _SCALED_ZERO_EXPONENT, exponent)
    return mantissa, exponent


def _scaled_directional_derivative(upper, lower, spacing):
    """Cancel a local field offset before applying the derivative scale."""
    difference = _scaled_quotient_sum_to_scaled((upper, -lower), 1.0)
    return _scaled_value_quotient_to_scaled(difference, spacing)


def _scaled_ulp(value):
    """One binary64 storage ulp as a normalized scaled value."""
    magnitude = np.abs(np.asarray(value, dtype=np.float64))
    _, value_exponent = np.frexp(magnitude)
    denorm_mantissa, denorm_exponent = np.frexp(
        np.nextafter(np.float64(0.0), np.float64(1.0)))
    normal = magnitude >= np.finfo(np.float64).tiny
    mantissa = np.where(normal, 0.5, denorm_mantissa)
    exponent = np.where(
        normal,
        value_exponent.astype(np.int64) - np.finfo(np.float64).nmant,
        np.int64(denorm_exponent))
    return mantissa, exponent


def _scaled_directional_roundoff(upper, lower, spacing):
    """Metric-weighted uncertainty from independently rounded face storage."""
    upper_term = _scaled_value_quotient_to_scaled(_scaled_ulp(upper), spacing)
    lower_term = _scaled_value_quotient_to_scaled(_scaled_ulp(lower), spacing)
    return _reduce_scaled_terms_to_scaled(
        np.stack((upper_term[0], lower_term[0]), axis=0),
        np.stack((upper_term[1], lower_term[1]), axis=0))


def _scaled_annular_radial_roundoff(
        upper, lower, spacing, radius):
    """Face-storage uncertainty with the exact annular divergence weights."""
    q = np.minimum(0.5 * (spacing / radius), 1.0)
    upper_term = _scaled_value_product_quotient_to_scaled(
        _scaled_ulp(upper), 1.0 + q, spacing)
    lower_term = _scaled_value_product_quotient_to_scaled(
        _scaled_ulp(lower), 1.0 - q, spacing)
    return _reduce_scaled_terms_to_scaled(
        np.stack((upper_term[0], lower_term[0]), axis=0),
        np.stack((upper_term[1], lower_term[1]), axis=0))


def _scaled_abs_less_equal_power_of_two(lhs, rhs, rhs_exponent_shift):
    """Elementwise ``abs(lhs) <= abs(rhs) * 2**shift`` without materializing."""
    lhs_mantissa, lhs_exponent = lhs
    rhs_mantissa, rhs_exponent = rhs
    lhs_zero = lhs_mantissa == 0.0
    rhs_nonzero = rhs_mantissa != 0.0
    shifted_rhs_exponent = rhs_exponent + np.int64(rhs_exponent_shift)
    ordered = ((lhs_exponent < shifted_rhs_exponent)
               | ((lhs_exponent == shifted_rhs_exponent)
                  & (np.abs(lhs_mantissa) <= np.abs(rhs_mantissa))))
    return lhs_zero | (rhs_nonzero & ordered)


def _normalized_scaled_linf_defect(
        scaled_values, roundoff_uncertainty=None):
    """Return ``||sum(d)||_inf / ||sum(|d|)||_inf`` from scaled terms."""
    mantissa = np.stack([value[0] for value in scaled_values], axis=0)
    exponent = np.stack([value[1] for value in scaled_values], axis=0)
    residual_mantissa, residual_exponent = _reduce_scaled_terms_to_scaled(
        mantissa, exponent)
    scale_mantissa, scale_exponent = _reduce_scaled_terms_to_scaled(
        np.abs(mantissa), exponent)

    if roundoff_uncertainty is not None:
        # External/background samples get the strict native gate: only genuine
        # opposite-sign cross-direction cancellation can consume the 1024-face-
        # ulp storage-forward-error envelope. A lone or same-sign one-ulp slope
        # therefore remains a resolved defect even on a huge DC field.
        lhs_mantissa = scaled_values[0][0]
        rhs_mantissa = scaled_values[1][0]
        both_nonzero = (lhs_mantissa != 0.0) & (rhs_mantissa != 0.0)
        opposite_sign = np.signbit(lhs_mantissa) != np.signbit(rhs_mantissa)
        residual = (residual_mantissa, residual_exponent)
        scale = (scale_mantissa, scale_exponent)
        explained = (both_nonzero & opposite_sign
                     & _scaled_abs_less_equal_power_of_two(
                         residual, scale, -1)
                     & _scaled_abs_less_equal_power_of_two(
                         residual, roundoff_uncertainty, 10))
        residual_mantissa = np.where(explained, 0.0, residual_mantissa)
        residual_exponent = np.where(
            explained, _SCALED_ZERO_EXPONENT, residual_exponent)

    def scaled_abs_max(value_mantissa, value_exponent):
        absolute_mantissa = np.abs(value_mantissa)
        maximum_exponent = np.max(np.where(
            absolute_mantissa == 0.0,
            _SCALED_ZERO_EXPONENT, value_exponent))
        maximum_mantissa = np.max(np.where(
            (absolute_mantissa != 0.0)
            & (value_exponent == maximum_exponent),
            absolute_mantissa, 0.0))
        return float(maximum_mantissa), int(maximum_exponent)

    residual_mantissa, residual_exponent = scaled_abs_max(
        residual_mantissa, residual_exponent)
    scale_mantissa, scale_exponent = scaled_abs_max(
        scale_mantissa, scale_exponent)
    if residual_mantissa == 0.0:
        return 0.0
    if not scale_mantissa > 0.0:
        raise ValueError("background divergence defect is not representable")

    common_exponent = max(residual_exponent, scale_exponent)
    with np.errstate(under="ignore"):
        scaled_residual = np.ldexp(
            residual_mantissa, residual_exponent - common_exponent)
        scaled_scale = np.ldexp(
            scale_mantissa, scale_exponent - common_exponent)
    defect = scaled_residual / scaled_scale
    if not np.isfinite(defect):
        raise ValueError("background divergence defect is not representable")
    return float(defect)


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


def background_divergence_linf(b0x, b0y, nx, ny, nghost, dx, dy,
                               *, geometry="cartesian", origin_x=0.0):
    """Max abs interior face-divergence of a staggered background field B0.

    ``b0x``/``b0y`` are 1-D ghost-padded host buffers in the solver storage
    layout (pitch = nx + 2*nghost, row-major). With b0x on the left face and b0y
    on the bottom face, the discrete divergence in interior cell (i, j) is::

        (b0x[i+1, j] - b0x[i, j]) / dx + (b0y[i, j+1] - b0y[i, j]) / dy

    For cylindrical geometry the radial term instead uses the exact annular
    weights ``(r_hi B_hi-r_lo B_lo)/int(r dr)``. A compatible uniform/solenoidal
    field gives 0 to round-off. Returns the L-inf norm over all interior cells
    (0.0 when there are no interior cells).
    """
    bx, by, nx, ny, g, dx, dy = _validated_staggered_inputs(
        b0x, b0y, nx, ny, nghost, dx, dy)
    if nx <= 0 or ny <= 0:
        return 0.0
    # Interior cells span [g, g+nx) in x and [g, g+ny) in y. The x-face flux uses
    # faces i and i+1 (both valid for interior i); the y-face flux uses faces
    # j and j+1.
    if geometry == "cylindrical":
        origin_x = float(origin_x)
        if not np.isfinite(origin_x) or origin_x < 0.0:
            raise ValueError("origin_x must be finite and non-negative")
        i = np.arange(nx, dtype=np.float64)
        r_lo = origin_x + i * dx
        r_hi = r_lo + dx
        r_mid = 0.5 * r_hi + 0.5 * r_lo
        if not np.all(np.isfinite(r_mid)) or np.any(r_mid <= 0.0):
            raise ValueError("cylindrical cell-center radii must be finite and positive")
        blo = bx[g:g + ny, g:g + nx]
        bhi = bx[g:g + ny, g + 1:g + 1 + nx]
        by_lo = by[g:g + ny, g:g + nx]
        by_hi = by[g + 1:g + 1 + ny, g:g + nx]
        div = _scaled_quotient_sum(
            (bhi, -blo, bhi, blo, by_hi, -by_lo),
            np.stack(np.broadcast_arrays(
                dx, dx, r_mid[None, :], r_mid[None, :], dy, dy), axis=0),
            np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0])[:, None, None])
    elif geometry == "cartesian":
        bx_lo = bx[g:g + ny, g:g + nx]
        bx_hi = bx[g:g + ny, g + 1:g + 1 + nx]
        by_lo = by[g:g + ny, g:g + nx]
        by_hi = by[g + 1:g + 1 + ny, g:g + nx]
        div = _scaled_quotient_sum(
            (bx_hi, -bx_lo, by_hi, -by_lo),
            np.array([dx, dx, dy, dy])[:, None, None])
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    if not np.all(np.isfinite(div)):
        raise ValueError("background divergence is not representable in float64")
    return float(np.max(np.abs(div))) if div.size else 0.0


def background_divergence_relative_linf(
        b0x, b0y, nx, ny, nghost, dx, dy, *, geometry="cartesian",
        origin_x=0.0):
    """Maximum derivative-scaled defect in the staggered divergence stencil.

    The result is ``max(|d_x+d_y|)/max(|d_x|+|d_y|)`` after each Cartesian
    face difference has cancelled its local field offset. Cylindrical ``d_x``
    is the complete annular contribution
    ``(B_hi-B_lo)/dr + (B_hi+B_lo)/(2*r_c)``, retaining the physical ``B_r/r``
    curvature. Directional contributions stay in mantissa/exponent form, so a
    represented slope cannot be hidden by a large Cartesian DC field, harmless
    roundoff at a local derivative null uses the global directional scale, and
    true cross-direction cancellation remains safe near float64's exponent
    limits. Before the global maximum, a residual may be removed as face-storage
    forward error only when both directional terms are nonzero and opposite in
    sign, the remainder is at most half their magnitude sum, and it lies within
    1024 metric-weighted face ULPs. One-direction and same-sign slopes receive no
    such allowance. A zero stencil reports zero.
    """
    bx, by, nx, ny, g, dx, dy = _validated_staggered_inputs(
        b0x, b0y, nx, ny, nghost, dx, dy)
    if nx <= 0 or ny <= 0:
        return 0.0
    bx_lo = bx[g:g + ny, g:g + nx]
    bx_hi = bx[g:g + ny, g + 1:g + 1 + nx]
    by_lo = by[g:g + ny, g:g + nx]
    by_hi = by[g + 1:g + 1 + ny, g:g + nx]
    if geometry == "cartesian":
        radial = _scaled_directional_derivative(bx_hi, bx_lo, dx)
        radial_roundoff = _scaled_directional_roundoff(bx_hi, bx_lo, dx)
    elif geometry == "cylindrical":
        origin_x = float(origin_x)
        if not np.isfinite(origin_x) or origin_x < 0.0:
            raise ValueError("origin_x must be finite and non-negative")
        radii = origin_x + (np.arange(nx, dtype=np.float64) + 0.5) * dx
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError(
                "cylindrical cell-center radii must be finite and positive")
        unit_row = np.ones((1, nx), dtype=np.float64)
        radial_denominator0 = np.stack(
            (dx * unit_row, dx * unit_row,
             2.0 * unit_row, 2.0 * unit_row), axis=0)
        radial_denominator1 = np.stack(
            (unit_row, unit_row, radii[None, :], radii[None, :]), axis=0)
        radial = _scaled_quotient_sum_to_scaled(
            (bx_hi, -bx_lo, bx_hi, bx_lo),
            radial_denominator0, radial_denominator1)
        radial_roundoff = _scaled_annular_radial_roundoff(
            bx_hi, bx_lo, dx, radii[None, :])
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    axial = _scaled_directional_derivative(by_hi, by_lo, dy)
    axial_roundoff = _scaled_directional_roundoff(by_hi, by_lo, dy)
    roundoff = _reduce_scaled_terms_to_scaled(
        np.stack((radial_roundoff[0], axial_roundoff[0]), axis=0),
        np.stack((radial_roundoff[1], axial_roundoff[1]), axis=0))
    return _normalized_scaled_linf_defect(
        (radial, axial), roundoff_uncertainty=roundoff)


def validate_background_boundary_compatibility(
        b0x, b0y, b0z, nx, ny, nghost, field_boundaries):
    """Require padded ``B0`` samples to be fixed by the configured field BCs.

    Static backgrounds are not ghost-filled by the solver.  Periodic samples
    must therefore wrap exactly to roundoff, wall samples must have odd normal
    and even tangential parity with zero normal field on the wall face, and the
    cylindrical ``axis`` closure additionally makes cell-centred ``B_phi`` odd.
    """
    bx, by, nx, ny, g, _, _ = _validated_staggered_inputs(
        b0x, b0y, nx, ny, nghost, 1.0, 1.0)
    bz = _finite_array(b0z, "b0z")
    if bz.size != bx.size:
        raise ValueError(
            f"staggered field size must be {bx.size} for the supplied grid")
    bz = bz.reshape(bx.shape)
    if len(field_boundaries) != 4:
        raise ValueError("field_boundaries must contain four side names")

    def at(component, i, j):
        return float(component[j + g, i + g])

    def require_pair(actual, source, source_sign, rule):
        scale = max(abs(actual), abs(source))
        if scale == 0.0:
            return
        lhs = actual / scale
        rhs = source / scale
        defect = abs(lhs - source_sign * rhs) / (abs(lhs) + abs(rhs))
        if (not np.isfinite(defect)
                or defect > DISCRETE_SOLENOIDAL_TOLERANCE):
            raise ValueError(f"background_field is incompatible with {rule}")

    def require_zero(actual, rule):
        if actual != 0.0:
            raise ValueError(f"background_field is incompatible with {rule}")

    for side, mode in enumerate(field_boundaries):
        if mode not in ("periodic", "wall", "axis"):
            continue
        x_side = side < 2
        low = side in (0, 2)
        if x_side:
            for j in range(ny + 1):
                for layer in range(1, g + 1):
                    if mode == "periodic":
                        target = -layer if low else nx - 1 + layer
                        source = nx - layer if low else layer - 1
                        for component in (bx, by, bz):
                            require_pair(at(component, target, j),
                                         at(component, source, j), 1.0,
                                         "the periodic x boundary")
                        continue

                    target = -layer if low else nx - 1 + layer
                    source = layer - 1 if low else nx - layer
                    rule = ("the cylindrical axis parity" if mode == "axis"
                            else "the x-wall parity")
                    require_pair(at(by, target, j), at(by, source, j), 1.0,
                                 rule)
                    require_pair(at(bz, target, j), at(bz, source, j),
                                 -1.0 if mode == "axis" else 1.0, rule)
                    if low:
                        zero_rule = ("the cylindrical axis constraint"
                                     if mode == "axis"
                                     else "the x-wall normal constraint")
                        require_zero(at(bx, 0, j), zero_rule)
                        require_pair(at(bx, -layer, j), at(bx, layer, j),
                                     -1.0, rule)
                    else:
                        require_zero(at(bx, nx, j),
                                     "the x-wall normal constraint")
                        if layer > 1:
                            offset = layer - 1
                            require_pair(at(bx, nx + offset, j),
                                         at(bx, nx - offset, j), -1.0,
                                         "the x-wall parity")
        else:
            for i in range(-g, nx + g):
                for layer in range(1, g + 1):
                    if mode == "periodic":
                        target = -layer if low else ny - 1 + layer
                        source = ny - layer if low else layer - 1
                        for component in (bx, by, bz):
                            require_pair(at(component, i, target),
                                         at(component, i, source), 1.0,
                                         "the periodic y boundary")
                        continue

                    target = -layer if low else ny - 1 + layer
                    source = layer - 1 if low else ny - layer
                    require_pair(at(bx, i, target), at(bx, i, source), 1.0,
                                 "the y-wall parity")
                    require_pair(at(bz, i, target), at(bz, i, source), 1.0,
                                 "the y-wall parity")
                    if low:
                        require_zero(at(by, i, 0),
                                     "the y-wall normal constraint")
                        require_pair(at(by, i, -layer), at(by, i, layer),
                                     -1.0, "the y-wall parity")
                    else:
                        require_zero(at(by, i, ny),
                                     "the y-wall normal constraint")
                        if layer > 1:
                            offset = layer - 1
                            require_pair(at(by, i, ny + offset),
                                         at(by, i, ny - offset), -1.0,
                                         "the y-wall parity")


def background_curl_linf(b0x, b0y, nx, ny, nghost, dx, dy):
    """Max ``|d By/dx - d Bx/dy|`` at interior cell corners.

    ``Bx`` is x-face/y-centred and ``By`` is y-face/x-centred.  Backward
    differences collocate both derivatives at a corner, matching the staggering
    used by CT. The diagnostic distinguishes vacuum from current-carrying
    prescribed backgrounds; nonzero curl is permitted by the exact split-energy
    change of variables.
    """
    bx, by, nx, ny, g, dx, dy = _validated_staggered_inputs(
        b0x, b0y, nx, ny, nghost, dx, dy)
    if nx <= 0 or ny <= 0:
        return 0.0
    # Corners (i,j), i=0..nx and j=0..ny, are covered by the padded arrays.
    dby_dx = (by[g:g + ny + 1, g:g + nx + 1] -
              by[g:g + ny + 1, g - 1:g + nx]) / dx
    dbx_dy = (bx[g:g + ny + 1, g:g + nx + 1] -
              bx[g - 1:g + ny, g:g + nx + 1]) / dy
    curl = dby_dx - dbx_dy
    if not np.all(np.isfinite(curl)):
        raise ValueError("background curl is not representable in float64")
    return float(np.max(np.abs(curl))) if curl.size else 0.0


def fast_magnetosonic_speed(rho, p, bx, by, bz, gamma):
    """Maximum (degenerate-direction) fast magnetosonic speed.

    A conservative Python-side estimate used only for sanity/documentation; the
    CLI's CFL guard uses the C++ ``cfl_limit()`` which scans per direction. The
    fast speed satisfies c_f^2 <= a^2 + b^2 where a^2 = gamma*p/rho is the sound
    speed and b^2 = |B|^2/rho is the Alfven speed, so this upper bound is safe.
    """
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than one")
    rho, p, bx, by, bz = np.broadcast_arrays(
        _finite_array(rho, "rho"), _finite_array(p, "p"),
        _finite_array(bx, "bx"), _finite_array(by, "by"),
        _finite_array(bz, "bz"))
    if np.any(rho <= 0.0) or np.any(p <= 0.0):
        raise ValueError("rho and p must be positive")
    sqrt_rho = np.sqrt(rho)
    sound = np.sqrt(gamma) * np.sqrt(p) / sqrt_rho
    alfven = np.hypot(np.hypot(bx / sqrt_rho, by / sqrt_rho), bz / sqrt_rho)
    result = np.hypot(sound, alfven)
    if not np.all(np.isfinite(result)):
        raise ValueError("magnetosonic speed is not representable in float64")
    return result
