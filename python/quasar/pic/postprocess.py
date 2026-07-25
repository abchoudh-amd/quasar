"""Plot helpers for PIC ``out.npz`` snapshots written by ``quasar.pic.cli``.

Two entry points:

* ``plot(npz_path, out_dir)``   — write field heatmaps + per-species particle
  scatter PNGs alongside the input.
* ``main(argv)``                — ``python -m quasar.pic.postprocess
  examples/square_toroid_pic/out.npz``.

matplotlib is imported lazily so the rest of the package stays usable on
machines without it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np


# Offsets are measured in cell widths from the lower domain face.  Keep this
# table in lock-step with CartesianYeeLayout2D/CylindricalYeeLayout2D in
# include/quasar/core/yee_field.hpp and with the field samplers/seeds.
_YEE_OFFSETS = {
    "cartesian": {
        "ex": (0.0, 0.5), "ey": (0.5, 0.0), "ez": (0.5, 0.5),
        "bx": (0.5, 0.0), "by": (0.0, 0.5), "bz": (0.0, 0.0),
    },
    "cylindrical": {
        "ex": (0.0, 0.5), "ey": (0.5, 0.0), "ez": (0.0, 0.5),
        "bx": (0.0, 0.0), "by": (0.5, 0.5), "bz": (0.0, 0.0),
    },
}


def rms(values) -> float:
    raw = np.asarray(values)
    try:
        arr = np.asarray(
            raw, dtype=np.complex128 if np.iscomplexobj(raw) else np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("rms values must be numeric") from None
    if arr.size == 0:
        raise ValueError("rms requires at least one value")
    if np.any(np.isnan(arr)):
        return float("nan")
    magnitude = np.abs(arr)
    scale = float(np.max(magnitude))
    if math.isinf(scale):
        return float("inf")
    if scale == 0.0:
        return 0.0
    # RMS is sqrt(mean(|x|^2)); using magnitudes is essential for complex input
    # and scaling first avoids false overflow for values near DBL_MAX.
    return float(scale * np.sqrt(np.mean((magnitude / scale) ** 2)))


def _validated_grid_shape(nx: int, ny: int, nghost: int) -> tuple[int, int, int]:
    """Validate integer archive dimensions without accepting bool as an int."""
    if isinstance(nx, (bool, np.bool_)) or isinstance(ny, (bool, np.bool_)) \
            or isinstance(nghost, (bool, np.bool_)):
        raise ValueError("nx, ny, and nghost must be integers")
    try:
        nx = nx.__index__()
        ny = ny.__index__()
        nghost = nghost.__index__()
    except (AttributeError, TypeError):
        raise ValueError("nx, ny, and nghost must be integers") from None
    if nx <= 0 or ny <= 0 or nghost < 0:
        raise ValueError("nx and ny must be positive and nghost non-negative")
    return int(nx), int(ny), int(nghost)


def reshape_with_ghost(flat: np.ndarray, nx: int, ny: int,
                       nghost: int) -> np.ndarray:
    """Return the cell-sized ``(ny, nx)`` view of a padded grid buffer.

    This helper is appropriate for cell-centred data.  Electromagnetic
    components do not all have this extent: use :func:`yee_component_view` for
    a Yee field so non-periodic high faces are not discarded.
    """
    nx, ny, nghost = _validated_grid_shape(nx, ny, nghost)
    arr = np.asarray(flat)
    if nghost > 0:
        padded_size = (nx + 2 * nghost) * (ny + 2 * nghost)
        if arr.size != padded_size:
            raise ValueError(
                f"field buffer has {arr.size} values; expected padded size "
                f"{padded_size} for nghost={nghost}")
        return arr.reshape(ny + 2 * nghost, nx + 2 * nghost)[
            nghost:-nghost, nghost:-nghost]
    if arr.size != nx * ny:
        raise ValueError(
            f"field buffer has {arr.size} values; expected {nx * ny}")
    return arr.reshape(ny, nx)


def _yee_offsets(component: str, geometry: str) -> tuple[float, float]:
    geometry = str(geometry).lower()
    component = str(component).lower()
    if geometry not in _YEE_OFFSETS:
        raise ValueError(
            f"geometry must be one of {sorted(_YEE_OFFSETS)}; got {geometry!r}")
    try:
        return _YEE_OFFSETS[geometry][component]
    except KeyError:
        raise ValueError(
            f"Yee component must be one of "
            f"{sorted(_YEE_OFFSETS[geometry])}; got {component!r}") from None


def _yee_counts(component: str, geometry: str, nx: int, ny: int,
                periodic_x: bool, periodic_y: bool) -> tuple[int, int]:
    offset_x, offset_y = _yee_offsets(component, geometry)
    # A zero-offset lattice owns both physical boundary faces.  On a periodic
    # axis the high face is the same topological point as the low face and is
    # the sole endpoint that should be removed.
    x_count = nx + int(offset_x == 0.0 and not periodic_x)
    y_count = ny + int(offset_y == 0.0 and not periodic_y)
    return x_count, y_count


def yee_component_view(flat: np.ndarray, nx: int, ny: int, nghost: int,
                       component: str, geometry: str = "cartesian", *,
                       periodic_x: bool = False,
                       periodic_y: bool = False) -> np.ndarray:
    """Return a component-aware physical view of a padded Yee buffer.

    Zero-offset components include the independent high face of a
    non-periodic axis.  That face occupies the first nominal high-halo slot in
    the uniform C++ allocation.  When (and only when) an axis is periodic, its
    high endpoint duplicates the low endpoint and is omitted from the view.

    Archives predating ``nghost`` may contain only an ``(ny, nx)`` compact
    array.  Such data cannot recover omitted high faces, so that legacy shape is
    accepted as-is.  New padded archives always expose the complete layout.
    """
    nx, ny, nghost = _validated_grid_shape(nx, ny, nghost)
    x_count, y_count = _yee_counts(
        component, geometry, nx, ny, bool(periodic_x), bool(periodic_y))
    arr = np.asarray(flat)

    if nghost > 0:
        pitch = nx + 2 * nghost
        height = ny + 2 * nghost
        expected = pitch * height
        if arr.size != expected:
            raise ValueError(
                f"field buffer has {arr.size} values; expected padded size "
                f"{expected} for nghost={nghost}")
        padded = arr.reshape(height, pitch)
        return padded[nghost:nghost + y_count,
                      nghost:nghost + x_count]

    component_size = x_count * y_count
    if arr.size == component_size:
        return arr.reshape(y_count, x_count)
    if arr.size == nx * ny:
        return arr.reshape(ny, nx)
    raise ValueError(
        f"field buffer has {arr.size} values; expected component size "
        f"{component_size} or legacy cell size {nx * ny}")


def yee_component_coordinates(
        component: str, geometry: str, nx: int, ny: int,
        origin_x: float, origin_y: float, lx: float, ly: float, *,
        periodic_x: bool = False, periodic_y: bool = False,
        view_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical sample coordinates ``(x, y)`` for a Yee component.

    ``view_shape`` normally comes from :func:`yee_component_view`.  It permits
    the cell-sized fallback used by legacy archives while rejecting arbitrary
    shape/layout mismatches.
    """
    nx, ny, _ = _validated_grid_shape(nx, ny, 0)
    offset_x, offset_y = _yee_offsets(component, geometry)
    expected_x, expected_y = _yee_counts(
        component, geometry, nx, ny, bool(periodic_x), bool(periodic_y))

    if view_shape is None:
        y_count, x_count = expected_y, expected_x
    else:
        if len(view_shape) != 2:
            raise ValueError("view_shape must contain (ny, nx)")
        try:
            y_count = view_shape[0].__index__()
            x_count = view_shape[1].__index__()
        except (AttributeError, TypeError):
            raise ValueError("view_shape must contain integer extents") from None
        if ((y_count, x_count) != (expected_y, expected_x)
                and (y_count, x_count) != (ny, nx)):
            raise ValueError(
                f"view_shape {(y_count, x_count)} does not match the Yee "
                f"extent {(expected_y, expected_x)}")

    try:
        origin_x = float(origin_x)
        origin_y = float(origin_y)
        lx = float(lx)
        ly = float(ly)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("origins and domain lengths must be real numbers") from None
    if not (math.isfinite(origin_x) and math.isfinite(origin_y)
            and math.isfinite(lx) and math.isfinite(ly)
            and lx > 0.0 and ly > 0.0):
        raise ValueError("origins must be finite and domain lengths positive")

    dx = lx / nx
    dy = ly / ny
    if not (math.isfinite(dx) and dx > 0.0
            and math.isfinite(dy) and dy > 0.0):
        raise OverflowError("Yee cell spacing is not representable")
    x = origin_x + (np.arange(x_count, dtype=np.float64) + offset_x) * dx
    y = origin_y + (np.arange(y_count, dtype=np.float64) + offset_y) * dy
    if offset_x == 0.0 and x_count == nx + 1:
        x[-1] = origin_x + lx
    if offset_y == 0.0 and y_count == ny + 1:
        y[-1] = origin_y + ly
    if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or (x.size > 1 and not np.all(np.diff(x) > 0.0))
            or (y.size > 1 and not np.all(np.diff(y) > 0.0))):
        raise OverflowError("Yee sample coordinates are not representable")
    return x, y


def field_periodicity(data) -> tuple[bool, bool]:
    """Return periodic-axis flags from four-side ``boundary_field`` metadata.

    Both sides must be periodic for an axis endpoint to be a duplicate.  Older
    archives have no boundary metadata; they conservatively return ``False`` on
    both axes so a physically independent high face is never discarded.
    """
    if "boundary_field" not in data.files:
        return False, False
    sides = np.asarray(data["boundary_field"]).reshape(-1)
    if sides.size != 4:
        raise ValueError("boundary_field must contain [x_lo, x_hi, y_lo, y_hi]")

    def _name(value) -> str:
        value = value.item() if isinstance(value, np.generic) else value
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    names = tuple(_name(value).lower() for value in sides)
    return (names[0] == names[1] == "periodic",
            names[2] == names[3] == "periodic")


def species_names(data) -> list[str]:
    """Species names present in an ``out.npz``, derived from the ``species_<name>_x``
    keys. Single source of truth for the per-species key schema so callers do not
    re-implement the string surgery."""
    if "species_names" in data.files:
        return [str(name) for name in data["species_names"]]
    return sorted({
        k[len("species_"):-len("_x")]
        for k in data.files
        if k.startswith("species_") and k.endswith("_x")
    })


def field_names(data) -> list[str]:
    """Final diagnostic field keys present in an archive.

    The deck may request any subset of the six electromagnetic components. Keep
    plotting driven by the archive rather than a hard-coded bz/ex/ey subset;
    snapshot keys deliberately do not start with ``field_`` and are excluded.
    """
    return sorted(
        name for name in data.files
        if name.startswith("field_") and len(name) > len("field_"))


def _archive_plane(data) -> str:
    """Return the archived sampling plane, defaulting legacy data to ``xy``."""
    if "plane" not in data.files:
        return "xy"
    values = np.asarray(data["plane"]).reshape(-1)
    if values.size != 1:
        raise ValueError("plane metadata must contain exactly one value")
    value = values[0]
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    plane = str(value).lower()
    if plane not in ("xy", "xz"):
        raise ValueError(f"plane must be 'xy' or 'xz'; got {plane!r}")
    return plane


def _geometry_labels(
        geometry: str, plane: str = "xy",
) -> tuple[str, str, dict[str, str]]:
    if plane not in ("xy", "xz"):
        raise ValueError(f"plane must be 'xy' or 'xz'; got {plane!r}")
    if geometry == "cylindrical":
        return "r", "z", {
            "ex": "Er", "ey": "Ez", "ez": "Ephi",
            "bx": "Br", "by": "Bz", "bz": "Bphi",
        }
    if geometry != "cartesian":
        raise ValueError(
            f"geometry must be 'cartesian' or 'cylindrical'; got {geometry!r}")
    if plane == "xz":
        # Cartesian x-z storage uses the right-handed basis (x, z, -y).
        # The out-of-plane slot therefore stores the negative lab-y component.
        return "x", "z", {
            "ex": "Ex", "ey": "Ez", "ez": "-Ey",
            "bx": "Bx", "by": "Bz", "bz": "-By",
        }
    return "x", "y", {
        "ex": "Ex", "ey": "Ey", "ez": "Ez",
        "bx": "Bx", "by": "By", "bz": "Bz",
    }


def plot(npz_path: Path | str, out_dir: Path | str | None = None) -> list[Path]:
    """Render diagnostic PNGs for an ``out.npz``. Returns written paths."""
    from .._plotting import require_pyplot
    plt = require_pyplot(agg=True)

    npz_path = Path(npz_path)
    out_dir = Path(out_dir) if out_dir is not None else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=False) as data:
        return _plot_archive(data, npz_path, out_dir, plt)


def _plot_archive(data, npz_path: Path, out_dir: Path, plt) -> list[Path]:
    """Render an already-open PIC archive (kept separate for scoped closing)."""
    nx = int(data["nx"][0])
    ny = int(data["ny"][0])
    # nghost is persisted by quasar.pic.cli; default to 0 for older npz files that
    # predate it (their buffers were already interior-sized or order-2).
    nghost = int(data["nghost"][0]) if "nghost" in data.files else 0
    geometry = (str(data["geometry"][0])
                if "geometry" in data.files else "cartesian").lower()
    plane = _archive_plane(data)
    # Validate the geometry even for an archive containing no requested field;
    # silently treating an unknown coordinate system as Cartesian would attach
    # incorrect component names and sample locations.
    _yee_offsets("bz", geometry)
    periodic_x, periodic_y = field_periodicity(data)
    axis_x, axis_y, component_labels = _geometry_labels(geometry, plane)
    unit_system = (str(data["unit_system"][0])
                   if "unit_system" in data.files else "SI")
    has_extent = all(name in data.files
                     for name in ("origin_x", "origin_y", "lx", "ly"))
    if has_extent:
        x0 = float(data["origin_x"][0])
        y0 = float(data["origin_y"][0])
        lx = float(data["lx"][0])
        ly = float(data["ly"][0])
        coordinate_unit = "m" if unit_system == "SI" else "internal length"
        x_label = f"{axis_x} ({coordinate_unit})"
        y_label = f"{axis_y} ({coordinate_unit})"
    else:
        # Cell coordinates still retain each component's half/whole-cell Yee
        # offsets when older archives do not carry physical domain metadata.
        x0 = y0 = 0.0
        lx = float(nx)
        ly = float(ny)
        x_label = f"{axis_x} cell"
        y_label = f"{axis_y} cell"
    field_unit = "T" if unit_system == "SI" else "normalized"
    written: list[Path] = []

    ext_bz = yee_component_view(
        data["external_bz"], nx, ny, nghost, "bz", geometry,
        periodic_x=periodic_x, periodic_y=periodic_y)
    ext_x, ext_y = yee_component_coordinates(
        "bz", geometry, nx, ny, x0, y0, lx, ly,
        periodic_x=periodic_x, periodic_y=periodic_y,
        view_shape=ext_bz.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(ext_x, ext_y, ext_bz, shading="nearest",
                       cmap="RdBu_r")
    ax.set_aspect("equal")
    ax.set_title(f"external {component_labels['bz']} ({field_unit})")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(im, ax=ax)
    p = out_dir / f"{npz_path.stem}_external_bz.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    for fname in field_names(data):
        component = fname.removeprefix("field_")
        arr = yee_component_view(
            data[fname], nx, ny, nghost, component, geometry,
            periodic_x=periodic_x, periodic_y=periodic_y)
        component_x, component_y = yee_component_coordinates(
            component, geometry, nx, ny, x0, y0, lx, ly,
            periodic_x=periodic_x, periodic_y=periodic_y,
            view_shape=arr.shape)
        vmax = float(np.max(np.abs(arr))) or 1.0
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(
            component_x, component_y, arr, shading="nearest",
            cmap="RdBu_r", vmin=-vmax, vmax=+vmax)
        ax.set_aspect("equal")
        if unit_system == "SI":
            component_unit = "V/m" if component.startswith("e") else "T"
        else:
            component_unit = "normalized"
        ax.set_title(
            f"field_{component_labels.get(component, component)} "
            f"({component_unit})")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(im, ax=ax)
        p = out_dir / f"{npz_path.stem}_{fname}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    names = species_names(data)
    if names:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pcolormesh(ext_x, ext_y, np.abs(ext_bz), shading="nearest",
                      cmap="Greys", alpha=0.4)
        ax.set_aspect("equal")
        colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
        for i, sp in enumerate(names):
            x = data[f"species_{sp}_x"]
            y = data[f"species_{sp}_y"]
            alive = data[f"species_{sp}_alive"].astype(bool)
            ax.scatter(x[alive], y[alive], s=0.5,
                       c=colors[i % len(colors)], label=sp, alpha=0.4)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("particle positions")
        ax.legend(markerscale=10)
        p = out_dir / f"{npz_path.stem}_particles.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    return written


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="quasar.pic.postprocess",
                                description="Render PIC out.npz diagnostics.")
    p.add_argument("npz", help="Path to out.npz produced by quasar.pic.cli.")
    p.add_argument("--out-dir", default=None,
                   help="Directory to write PNGs into (default: alongside npz).")
    args = p.parse_args(argv)
    written = plot(args.npz, args.out_dir)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
