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


# ---------------------------------------------------------------------------
# data shaping (no matplotlib needed)
# ---------------------------------------------------------------------------


def magnitude(B: np.ndarray) -> np.ndarray:
    """``|B|`` along the last axis of ``B`` (works for flat ``(M, 3)`` or
    reshaped ``(..., 3)``)."""
    arr = np.asarray(B)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected last axis of size 3, got shape {arr.shape}")
    return np.linalg.norm(arr, axis=-1)


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
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
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
    return np.take(arr, index, axis=axis)


# ---------------------------------------------------------------------------
# plotting helpers (matplotlib is imported lazily)
# ---------------------------------------------------------------------------


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting; install it via "
            "`pip install matplotlib`"
        ) from e
    return plt


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
        idx = nz // 2 if index is None else int(index)
        slab = slice_xy(arr, idx)        # (ny, nx, 3)
        title_default = f"|B| at z-slice idx={idx}"
    elif axis == "y":
        idx = ny // 2 if index is None else int(index)
        slab = slice_xz(arr, idx)        # (nz, nx, 3)
        title_default = f"|B| at y-slice idx={idx}"
    elif axis == "x":
        idx = nx // 2 if index is None else int(index)
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

    pos = np.asarray(positions)
    if pos.ndim == 2 and pos.shape[1] == 3:
        diffs = np.diff(pos, axis=0)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    else:
        s = pos.ravel()
    if s.shape[0] != B.shape[0]:
        raise ValueError("positions and B size mismatch")

    comp_map = {"x": B[:, 0], "y": B[:, 1], "z": B[:, 2],
                "magnitude": magnitude(B)}
    if component not in comp_map:
        raise ValueError(
            f"component must be one of {list(comp_map)}; got {component!r}")
    y = comp_map[component]

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
    archive = np.load(path, allow_pickle=False)
    return {name: archive[name] for name in archive.files}
