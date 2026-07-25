"""Shared numerical helpers for the MHD front-end.

The C++ ``MhdSolver2D::cfl_limit()`` is the authoritative stable-step bound (it
scans the seeded state for the max fast-magnetosonic signal speed). These Python
helpers cover the small primitive<->conserved conversions the deck IC generators
and the CLI need, kept in one place so they cannot drift between io.py and cli.py
(mirrors ``quasar.pic.numerics`` keeping the CFL formula in a single module).
"""

from __future__ import annotations

import numpy as np


# Scale-free acceptance envelope shared with the native live/background
# solenoidality preflights.  The factor covers a short compensated stencil plus
# profile sampling roundoff while remaining an O(machine-epsilon) criterion.
DISCRETE_SOLENOIDAL_TOLERANCE = 1024.0 * np.finfo(np.float64).eps


_FACE_TO_CELL_CENTERED = {
    2: np.array([0.5, 0.5], dtype=np.float64),
    4: np.array([-1.0, 13.0, 13.0, -1.0], dtype=np.float64) / 24.0,
    6: np.array([11.0, -93.0, 802.0, 802.0, -93.0, 11.0],
                dtype=np.float64) / 1440.0,
    # Polynomial-exact coefficient is -9531, not -9504. The latter fails even
    # the degree-zero moment (its weights sum to 2241/2240).
    8: np.array([-191.0, 1879.0, -9531.0, 68323.0,
                 68323.0, -9531.0, 1879.0, -191.0],
                dtype=np.float64) / 120960.0,
}


def _collocation_width(extent, nghost):
    requested = 8 if nghost >= 4 else 6 if nghost >= 3 else 4 if nghost >= 2 else 2
    for width in (8, 6, 4, 2):
        if requested >= width and extent >= width:
            return width
    return 1


def _one_sided_face_integral_weights(width, relative_cell):
    """Weights for an outer-ghost polynomial integral over one logical cell."""
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
    result = np.zeros(width, dtype=np.float64)
    for point, weight in zip(xq, wq):
        terms = barycentric / (float(relative_cell) + point - nodes)
        result += weight * terms / np.sum(terms)
    return result


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


def _scaled_quotient_sum(numerators, denominator0, denominator1=1.0):
    """Elementwise ``sum(numerator / denominator0 / denominator1)``.

    Terms stay in mantissa/exponent form until the final result. At each round
    the largest term is paired with the largest opposite-sign term when one is
    available, so matched enormous terms cancel before a smaller, representable
    remainder is rounded away.
    This is the NumPy counterpart of native ``scaled_product_quotient_sum``.
    All inputs to this private helper are already finite and denominators are
    nonzero by the public diagnostic's validation.
    """
    values = np.stack(np.broadcast_arrays(*numerators), axis=0)
    count = values.shape[0]
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

    # A zero has no meaningful exponent. Keep it below every possible binary64
    # quotient exponent so it sorts after all nonzero terms without relying on
    # platform-specific integer minima during exponent subtraction.
    zero_exponent = np.int64(-1_000_000)
    exponent = np.where(mantissa == 0.0, zero_exponent, exponent)

    # Match the native reducer: combine the largest term with the largest
    # opposite-sign term whenever one exists. Pairing the two largest terms
    # regardless of sign can first round A+B and leave an ulp(A) residue in
    # A+B-A-B+c that overwhelms the finite survivor c.
    term_axis = np.arange(count).reshape(
        (count,) + (1,) * (mantissa.ndim - 1))
    for _ in range(count - 1):
        nonzero = mantissa != 0.0
        largest_exponent = np.max(
            np.where(nonzero, exponent, zero_exponent), axis=0)
        largest_mask = nonzero & (exponent == largest_exponent[None, ...])
        largest_index = np.argmax(
            np.where(largest_mask, np.abs(mantissa), -1.0), axis=0)
        largest_gather = largest_index[None, ...]
        largest_mantissa = np.take_along_axis(
            mantissa, largest_gather, axis=0)[0]
        largest_term_exponent = np.take_along_axis(
            exponent, largest_gather, axis=0)[0]

        remaining = nonzero & (term_axis != largest_gather)
        opposite = remaining & (
            np.signbit(mantissa) != np.signbit(largest_mantissa)[None, ...])
        has_opposite = np.any(opposite, axis=0)
        candidates = np.where(has_opposite[None, ...], opposite, remaining)
        has_partner = np.any(candidates, axis=0)
        partner_exponent = np.max(
            np.where(candidates, exponent, zero_exponent), axis=0)
        partner_mask = candidates & (
            exponent == partner_exponent[None, ...])
        partner_index = np.argmax(
            np.where(partner_mask, np.abs(mantissa), -1.0), axis=0)
        partner_gather = partner_index[None, ...]
        partner_mantissa = np.take_along_axis(
            mantissa, partner_gather, axis=0)[0]
        partner_term_exponent = np.take_along_axis(
            exponent, partner_gather, axis=0)[0]

        with np.errstate(under="ignore"):
            combined = largest_mantissa + np.ldexp(
                partner_mantissa,
                partner_term_exponent - largest_term_exponent)
        combined_mantissa, combined_shift = np.frexp(combined)
        combined_exponent = (
            largest_term_exponent + combined_shift.astype(np.int64))
        combined_exponent = np.where(
            combined_mantissa == 0.0, zero_exponent, combined_exponent)

        # Some array elements may already contain only one nonzero term. Leave
        # those lanes untouched while other cells finish their reductions.
        old_largest_mantissa = np.take_along_axis(
            mantissa, largest_gather, axis=0)[0]
        old_largest_exponent = np.take_along_axis(
            exponent, largest_gather, axis=0)[0]
        np.put_along_axis(
            mantissa, largest_gather,
            np.where(has_partner, combined_mantissa,
                     old_largest_mantissa)[None, ...], axis=0)
        np.put_along_axis(
            exponent, largest_gather,
            np.where(has_partner, combined_exponent,
                     old_largest_exponent)[None, ...], axis=0)
        partner_old_mantissa = np.take_along_axis(
            mantissa, partner_gather, axis=0)[0]
        partner_old_exponent = np.take_along_axis(
            exponent, partner_gather, axis=0)[0]
        np.put_along_axis(
            mantissa, partner_gather,
            np.where(has_partner, 0.0,
                     partner_old_mantissa)[None, ...], axis=0)
        np.put_along_axis(
            exponent, partner_gather,
            np.where(has_partner, zero_exponent,
                     partner_old_exponent)[None, ...], axis=0)

    final_index = np.argmax(mantissa != 0.0, axis=0)[None, ...]
    final_mantissa = np.take_along_axis(mantissa, final_index, axis=0)[0]
    final_exponent = np.take_along_axis(exponent, final_index, axis=0)[0]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.ldexp(final_mantissa, final_exponent)


def _normalized_product_quotient_defect(numerators, coefficients,
                                        denominators):
    """Return ``|sum(t)|/sum(|t|)`` without materializing physical-scale t."""
    values = np.stack(np.broadcast_arrays(*numerators), axis=0)
    coefficient = np.broadcast_to(
        np.asarray(coefficients, dtype=np.float64), values.shape)
    denominator = np.broadcast_to(
        np.asarray(denominators, dtype=np.float64), values.shape)
    if (not np.all(np.isfinite(values))
            or not np.all(np.isfinite(coefficient))
            or not np.all(np.isfinite(denominator))
            or np.any(denominator <= 0.0)):
        raise ValueError("invalid discrete-divergence term")

    value_mantissa, value_exponent = np.frexp(values)
    coefficient_mantissa, coefficient_exponent = np.frexp(coefficient)
    denominator_mantissa, denominator_exponent = np.frexp(denominator)
    raw_mantissa = ((value_mantissa * coefficient_mantissa)
                    / denominator_mantissa)
    mantissa, shift = np.frexp(raw_mantissa)
    exponent = (value_exponent.astype(np.int64)
                + coefficient_exponent.astype(np.int64)
                - denominator_exponent.astype(np.int64)
                + shift.astype(np.int64))
    zero_exponent = np.int64(-1_000_000)
    exponent = np.where(mantissa == 0.0, zero_exponent, exponent)
    common_exponent = np.max(exponent, axis=0)
    with np.errstate(under="ignore"):
        scaled = np.ldexp(mantissa, exponent - common_exponent[None, ...])

    signed_sum = np.zeros(values.shape[1:], dtype=np.float64)
    compensation = np.zeros_like(signed_sum)
    absolute_sum = np.zeros_like(signed_sum)
    for term in scaled:
        corrected = term - compensation
        updated = signed_sum + corrected
        compensation = (updated - signed_sum) - corrected
        signed_sum = updated
        absolute_sum += np.abs(term)
    defect = np.divide(
        np.abs(signed_sum), absolute_sum,
        out=np.zeros_like(signed_sum), where=absolute_sum > 0.0)
    if not np.all(np.isfinite(defect)):
        raise ValueError("background divergence defect is not representable")
    return defect


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
    """Maximum scale-free defect in the exact staggered divergence stencil.

    Each interior cell contributes ``|sum(t_k)| / sum(|t_k|)``, where the
    signed terms are the Cartesian face differences or the exact annular radial
    weights plus the axial face differences.  Terms share a binary exponent
    before either sum is formed, so the diagnostic is invariant under field-unit
    rescaling and remains meaningful at subnormal and near-overflow magnitudes.
    A zero field reports zero.
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
        coefficients = np.array([1.0, -1.0, 1.0, -1.0])[:, None, None]
    elif geometry == "cylindrical":
        origin_x = float(origin_x)
        if not np.isfinite(origin_x) or origin_x < 0.0:
            raise ValueError("origin_x must be finite and non-negative")
        radii = origin_x + (np.arange(nx, dtype=np.float64) + 0.5) * dx
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError(
                "cylindrical cell-center radii must be finite and positive")
        q = np.minimum(1.0, 0.5 * dx / radii)
        coefficients = np.stack(np.broadcast_arrays(
            1.0 + q[None, :], -(1.0 - q[None, :]),
            np.ones((ny, nx)), -np.ones((ny, nx))), axis=0)
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    denominators = np.array([dx, dx, dy, dy])[:, None, None]
    defect = _normalized_product_quotient_defect(
        (bx_hi, bx_lo, by_hi, by_lo), coefficients, denominators)
    return float(np.max(defect)) if defect.size else 0.0


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
