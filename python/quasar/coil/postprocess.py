"""Postprocessing helpers for B-field outputs.

Data-shaping helpers (reshape, slicing, magnitude) are pure-numpy and have
no plotting dependency. Plotting helpers lazy-import matplotlib so the
rest of the module is usable on machines without it; an ``ImportError``
from a plot function then has a clear message indicating that matplotlib
is required for that specific call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .._deck import as_integer as _as_integer


# ---------------------------------------------------------------------------
# data shaping (no matplotlib needed)
# ---------------------------------------------------------------------------


def magnitude(B: np.ndarray) -> np.ndarray:
    """``|B|`` along the last axis of ``B`` (works for flat ``(M, 3)`` or
    reshaped ``(..., 3)``)."""
    arr = np.asarray(B)
    if arr.ndim == 0 or arr.shape[-1] != 3:
        raise ValueError(f"expected last axis of size 3, got shape {arr.shape}")
    try:
        finite = bool(np.all(np.isfinite(arr)))
    except TypeError:
        finite = False
    if np.iscomplexobj(arr) or not finite:
        raise ValueError("B must contain only finite real values")
    # hypot scales its operands and therefore avoids the false overflow from
    # squaring components near sqrt(DBL_MAX), while retaining subnormal terms.
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.hypot(np.hypot(arr[..., 0], arr[..., 1]), arr[..., 2])
    if not np.all(np.isfinite(result)):
        raise ValueError("B magnitude is not representable in output precision")
    return result


def reshape_to_grid(B_flat: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    """Reshape a flat ``(M, 3)`` B-array back into a 3D grid.

    ``dims = [nx, ny, nz]`` matches ``ObservationGrid.dims``. The returned
    array is shaped ``(nz, ny, nx, 3)`` so that ``arr[iz, iy, ix, :]``
    indexes the field at grid coordinate ``(ix, iy, iz)`` (consistent with
    the x-fastest linear layout used in ``observation.cpp``).
    """
    arr = np.asarray(B_flat)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"reshape_to_grid expects shape (M, 3); got {arr.shape}")
    if len(dims) != 3:
        raise ValueError(f"dims must be length 3, got {dims!r}")
    nx, ny, nz = (
        _as_integer(dims[0], "dims[0]"),
        _as_integer(dims[1], "dims[1]"),
        _as_integer(dims[2], "dims[2]"))
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("dims entries must be positive")
    if arr.shape[0] != nx * ny * nz:
        raise ValueError(
            f"M={arr.shape[0]} does not match nx*ny*nz={nx*ny*nz}")
    return arr.reshape(nz, ny, nx, 3)


def slice_xy(B_grid: np.ndarray, iz: int) -> np.ndarray:
    """Extract the constant-z slice indexed by ``iz`` from a ``(nz, ny, nx, 3)``
    grid; returns ``(ny, nx, 3)``."""
    return _slice_axis(B_grid, axis=0, index=iz)


def slice_xz(B_grid: np.ndarray, iy: int) -> np.ndarray:
    """Extract the constant-y slice; returns ``(nz, nx, 3)``."""
    return _slice_axis(B_grid, axis=1, index=iy)


def slice_yz(B_grid: np.ndarray, ix: int) -> np.ndarray:
    """Extract the constant-x slice; returns ``(nz, ny, 3)``."""
    return _slice_axis(B_grid, axis=2, index=ix)


def _slice_axis(B_grid: np.ndarray, axis: int, index: int) -> np.ndarray:
    arr = np.asarray(B_grid)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"expected grid shape (nz, ny, nx, 3); got {arr.shape}")
    exact_index = _as_integer(index, "slice index")
    return np.take(arr, exact_index, axis=axis)


# ---------------------------------------------------------------------------
# plotting helpers (matplotlib is imported lazily)
# ---------------------------------------------------------------------------


def _require_matplotlib():
    from .._plotting import require_pyplot
    return require_pyplot()


def plot_magnitude_slice(
    B_grid: np.ndarray,
    axis: str = "z",
    index: int | None = None,
    extent: Sequence[float] | None = None,
    title: str | None = None,
    cmap: str = "viridis",
):
    """Imshow of ``|B|`` on a constant-axis slice through a 3D grid.

    Parameters
    ----------
    B_grid : ndarray, shape (nz, ny, nx, 3)
        Reshaped output of ``reshape_to_grid``.
    axis : {"x", "y", "z"}
        Constant axis defining the slice.
    index : int, optional
        Slice index along ``axis``. Defaults to the mid-plane.
    extent, title, cmap
        Forwarded to matplotlib.

    Returns
    -------
    (fig, ax)
        The matplotlib figure and axes.
    """
    plt = _require_matplotlib()
    arr = np.asarray(B_grid)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"expected grid shape (nz, ny, nx, 3); got {arr.shape}")

    nz, ny, nx = arr.shape[0], arr.shape[1], arr.shape[2]
    if axis == "z":
        idx = nz // 2 if index is None else _as_integer(index, "slice index")
        slab = slice_xy(arr, idx)        # (ny, nx, 3)
        title_default = f"|B| at z-slice idx={idx}"
    elif axis == "y":
        idx = ny // 2 if index is None else _as_integer(index, "slice index")
        slab = slice_xz(arr, idx)        # (nz, nx, 3)
        title_default = f"|B| at y-slice idx={idx}"
    elif axis == "x":
        idx = nx // 2 if index is None else _as_integer(index, "slice index")
        slab = slice_yz(arr, idx)        # (nz, ny, 3)
        title_default = f"|B| at x-slice idx={idx}"
    else:
        raise ValueError(f"axis must be one of 'x','y','z'; got {axis!r}")

    mag = magnitude(slab)                # (..., ...)
    fig, ax = plt.subplots()
    img = ax.imshow(mag, origin="lower", aspect="auto",
                    cmap=cmap, extent=extent)
    fig.colorbar(img, ax=ax, label="|B| (T)")
    ax.set_title(title or title_default)
    return fig, ax


def plot_line_profile(
    positions: np.ndarray,
    B: np.ndarray,
    component: str = "magnitude",
    label: str | None = None,
    ax=None,
):
    """1D plot of B or |B| along a polyline of observation points.

    Parameters
    ----------
    positions : ndarray, shape (M, 3) or (M,)
        Sample positions; if shape ``(M, 3)``, arc-length is computed
        cumulatively. If shape ``(M,)``, used as the x-axis directly.
    B : ndarray, shape (M, 3)
        Field values at each position.
    component : {"x", "y", "z", "magnitude"}
        Which scalar to plot.
    label, ax
        Forwarded to matplotlib.

    Returns
    -------
    (fig, ax)
    """
    plt = _require_matplotlib()
    B = np.asarray(B)
    if B.ndim != 2 or B.shape[1] != 3:
        raise ValueError(f"B must be shape (M, 3); got {B.shape}")
    try:
        b_finite = bool(np.all(np.isfinite(B)))
    except TypeError:
        b_finite = False
    if np.iscomplexobj(B) or not b_finite:
        raise ValueError("B must contain only finite real values")

    pos = np.asarray(positions)
    try:
        positions_finite = bool(np.all(np.isfinite(pos)))
    except TypeError:
        positions_finite = False
    if np.iscomplexobj(pos) or not positions_finite:
        raise ValueError("positions must contain only finite real values")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            plot_positions = np.asarray(pos, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "positions must be representable as real plot coordinates") from None
    if not np.all(np.isfinite(plot_positions)):
        raise ValueError(
            "positions must be representable as real plot coordinates")
    if pos.ndim == 2 and pos.shape[1] == 3:
        with np.errstate(over="ignore", invalid="ignore"):
            # Convert before differencing: np.diff on signed/unsigned integer
            # coordinates wraps at the dtype boundary (INT64_MIN -> INT64_MAX
            # would otherwise appear to be a unit-length segment).
            diffs = np.diff(plot_positions, axis=0)
            segment_lengths = np.hypot(
                np.hypot(diffs[:, 0], diffs[:, 1]), diffs[:, 2])
            s = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        if not np.all(np.isfinite(s)):
            raise ValueError(
                "cumulative position arc length is not representable")
    elif pos.ndim == 1:
        s = plot_positions.ravel()
    else:
        raise ValueError("positions must have shape (M,) or (M, 3)")
    if s.shape[0] != B.shape[0]:
        raise ValueError("positions and B size mismatch")

    component_index = {"x": 0, "y": 1, "z": 2}
    if component in component_index:
        y = B[:, component_index[component]]
    elif component == "magnitude":
        y = magnitude(B)
    else:
        raise ValueError(
            "component must be one of ['x', 'y', 'z', 'magnitude']; "
            f"got {component!r}")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            y = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "selected B component is not representable for plotting") from None
    if not np.all(np.isfinite(y)):
        raise ValueError(
            "selected B component is not representable for plotting")

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(s, y, label=label or component)
    ax.set_xlabel("arc length (m)")
    ax.set_ylabel(f"B_{component}" if component != "magnitude" else "|B| (T)")
    if label:
        ax.legend()
    return fig, ax


# ---------------------------------------------------------------------------
# convenience: load + dispatch
# ---------------------------------------------------------------------------


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load an ``out.npz`` produced by ``quasar coil run`` into a plain dict."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
