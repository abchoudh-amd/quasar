"""Load rectilinear magnetic-field maps for the ``file_grid`` evaluator."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
import zipfile

import numpy as np

from ._paths import confine_input_path


# The registry bridge expands every vector into Python floats before pybind11
# copies it into C++.  A 2^26-point map would therefore require several GiB of
# Python-object overhead in addition to the NumPy and C++ buffers.  Keep the
# public deck limit honest for this transport; native C++ text maps are not
# subject to this Python-list ceiling.
MAX_FIELD_GRID_POINTS = 1 << 20
# The evaluator stores float64, and the loader accepts real numeric inputs up to
# NumPy's 16-byte long-double width.  This bound therefore admits every supported
# maximum-size values array (3 * points * 16 bytes) plus metadata and container
# overhead, while stopping a compressed archive from inflating without bound.
MAX_FIELD_GRID_ARCHIVE_BYTES = 64 << 20
MAX_FIELD_GRID_ARCHIVE_MEMBERS = 8


def _real_array(array, label: str) -> np.ndarray:
    values = np.asarray(array)
    if values.dtype.kind not in "iuf" or values.dtype.itemsize > 16:
        raise ValueError(f"{label} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values")
    return values


def _triple(array, label: str) -> list[float]:
    values = _real_array(array, label)
    if values.shape != (3,):
        raise ValueError(f"{label} must have shape (3,), got {values.shape}")
    return values.tolist()


def _dims(array, label: str) -> list[int]:
    values = np.asarray(array)
    if values.shape != (3,) or values.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a three-element integer array")
    dims = [int(value) for value in values]
    if any(value <= 0 for value in dims):
        raise ValueError(f"{label} must be positive")
    return dims


def _only_alias(keys: set[str], aliases: tuple[str, str], label: str) -> str:
    present = [key for key in aliases if key in keys]
    if len(present) != 1:
        if present:
            raise ValueError(
                f"{label} contains ambiguous aliases {present}; provide exactly one")
        raise ValueError(
            f"{label} must contain exactly one of {' or '.join(aliases)}")
    return present[0]


def _archive_headers(
        path: Path, rel: str, label: str
) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    try:
        compressed_size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"{label} {rel!r} could not be inspected: {exc}") from exc
    if compressed_size > MAX_FIELD_GRID_ARCHIVE_BYTES:
        raise ValueError(
            f"{label} {rel!r} archive is {compressed_size} bytes; limit is "
            f"{MAX_FIELD_GRID_ARCHIVE_BYTES}")

    try:
        container = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError(f"{label} {rel!r} is not a valid npz archive: {exc}") from exc

    headers: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    try:
        members = container.infolist()
        if len(members) > MAX_FIELD_GRID_ARCHIVE_MEMBERS:
            raise ValueError(
                f"{label} {rel!r} has {len(members)} archive members; limit is "
                f"{MAX_FIELD_GRID_ARCHIVE_MEMBERS}")
        expanded_size = sum(member.file_size for member in members)
        if expanded_size > MAX_FIELD_GRID_ARCHIVE_BYTES:
            raise ValueError(
                f"{label} {rel!r} expands to {expanded_size} bytes; limit is "
                f"{MAX_FIELD_GRID_ARCHIVE_BYTES}")

        for member in members:
            name = member.filename
            if (member.is_dir() or member.flag_bits & 0x1 or "/" in name
                    or not name.endswith(".npy")):
                raise ValueError(
                    f"{label} {rel!r} contains unsupported archive member "
                    f"{name!r}")
            key = name[:-4]
            if key in headers:
                raise ValueError(
                    f"{label} {rel!r} contains duplicate archive key {key!r}")
            try:
                with container.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError(f"unsupported npy version {version}")
                    payload_bytes = member.file_size - stream.tell()
            except (EOFError, OSError, ValueError) as exc:
                raise ValueError(
                    f"{label} {rel!r} has an invalid {name!r} header: {exc}") from exc
            dtype = np.dtype(dtype)
            if dtype.hasobject:
                raise ValueError(
                    f"{label} {rel!r} contains object array {key!r}")
            expected_bytes = math.prod(shape) * dtype.itemsize
            if expected_bytes != payload_bytes:
                raise ValueError(
                    f"{label} {rel!r} has inconsistent array size for {key!r}")
            headers[key] = (tuple(shape), dtype)
    finally:
        container.close()
    return headers


def _rounded_affine(origin: float, spacing: float, index: int,
                    label: str) -> float:
    # Evaluate on the exact binary inputs before rounding once to float.  This
    # avoids a false overflow in index*spacing when the origin later cancels it.
    exact = Fraction.from_float(origin) + index * Fraction.from_float(spacing)
    try:
        rounded = float(exact)
    except OverflowError:
        raise ValueError(f"{label} upper grid coordinate is not finite") from None
    if not math.isfinite(rounded):
        raise ValueError(f"{label} upper grid coordinate is not finite")
    return rounded


def load_file_grid_npz(base: Path | str, rel: str, *, label: str) -> dict[str, list[float]]:
    """Return registry parameters from a confined NumPy ``.npz`` field map.

    Accepted payloads contain either ``B_xyz_grid`` with shape
    ``(nz, ny, nx, 3)`` or flat ``B_xyz`` plus ``dims=[nx,ny,nz]``.  Every map
    must also contain ``grid_origin``/``grid_spacing`` (the shorter aliases
    ``origin``/``spacing`` are accepted). Values are flattened in x-fastest
    order, matching ``ObservationGrid`` and ``FileGridEvaluator``.
    """
    path = confine_input_path(base, rel, label=label)
    headers = _archive_headers(path, rel, label)
    container_keys = set(headers)
    try:
        archive_context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"{label} {rel!r} could not be read as an npz: {exc}") from exc

    try:
        with archive_context as archive:
            keys = set(archive.files)
            if keys != container_keys:
                raise ValueError(
                    f"{label} {rel!r} has an inconsistent npz directory")

            origin_key = _only_alias(
                keys, ("grid_origin", "origin"), f"{label} {rel!r}")
            spacing_key = _only_alias(
                keys, ("grid_spacing", "spacing"), f"{label} {rel!r}")
            has_grid = "B_xyz_grid" in keys
            has_flat = "B_xyz" in keys
            if has_grid == has_flat:
                raise ValueError(
                    f"{label} {rel!r} must contain exactly one of "
                    "B_xyz_grid or B_xyz")
            field_key = "B_xyz_grid" if has_grid else "B_xyz"
            allowed = {
                origin_key, spacing_key, field_key, "dims", "observation_kind"}
            unknown = sorted(keys - allowed)
            if unknown:
                raise ValueError(
                    f"{label} {rel!r} contains unsupported key(s) {unknown}")

            for key in (origin_key, spacing_key):
                shape, dtype = headers[key]
                if (shape != (3,) or dtype.kind not in "iuf"
                        or dtype.itemsize > 16):
                    raise ValueError(
                        f"{label}.{key} must be a three-element real array")

            origin = _triple(archive[origin_key], f"{label}.grid_origin")
            spacing = _triple(archive[spacing_key], f"{label}.grid_spacing")
            if "observation_kind" in keys:
                kind_shape, kind_dtype = headers["observation_kind"]
                if kind_shape != () or kind_dtype.kind != "U":
                    raise ValueError(
                        f"{label}.observation_kind must be the scalar string 'grid'")
                kind = np.asarray(archive["observation_kind"])
                if kind.shape != () or kind.dtype.kind != "U" or kind.item() != "grid":
                    raise ValueError(
                        f"{label}.observation_kind must be the scalar string 'grid'")

            if has_grid:
                value_shape, _ = headers["B_xyz_grid"]
                if len(value_shape) != 4 or value_shape[-1] != 3:
                    raise ValueError(
                        f"{label}.B_xyz_grid must have shape (nz,ny,nx,3), "
                        f"got {value_shape}")
                nz, ny, nx, _ = value_shape
                dims = [int(nx), int(ny), int(nz)]
                count = math.prod(dims)
                if any(n <= 0 for n in dims):
                    raise ValueError(f"{label}.dims must be positive")
                if count > MAX_FIELD_GRID_POINTS:
                    raise ValueError(
                        f"{label} has {count} points; limit is "
                        f"{MAX_FIELD_GRID_POINTS}")
                if "dims" in keys:
                    dims_shape, dims_dtype = headers["dims"]
                    if dims_shape != (3,) or dims_dtype.kind not in "iu":
                        raise ValueError(
                            f"{label}.dims must be a three-element integer array")
                raw_values = np.asarray(archive["B_xyz_grid"])
                if raw_values.ndim != 4 or raw_values.shape[-1] != 3:
                    raise ValueError(
                        f"{label}.B_xyz_grid must have shape (nz,ny,nx,3), "
                        f"got {raw_values.shape}")
                nz, ny, nx, _ = raw_values.shape
                dims = [int(nx), int(ny), int(nz)]
                if "dims" in keys and _dims(
                        archive["dims"], f"{label}.dims") != dims:
                    raise ValueError(
                        f"{label}.dims does not match B_xyz_grid shape")
            else:
                if "dims" not in keys:
                    raise ValueError(
                        f"{label} {rel!r} requires dims with flat B_xyz")
                dims_shape, dims_dtype = headers["dims"]
                if dims_shape != (3,) or dims_dtype.kind not in "iu":
                    raise ValueError(
                        f"{label}.dims must be a three-element integer array")
                dims = _dims(archive["dims"], f"{label}.dims")
                nx, ny, nz = dims
                count = math.prod(dims)
                if count > MAX_FIELD_GRID_POINTS:
                    raise ValueError(
                        f"{label} has {count} points; limit is "
                        f"{MAX_FIELD_GRID_POINTS}")
                value_shape, _ = headers["B_xyz"]
                if value_shape != (count, 3):
                    raise ValueError(
                        f"{label}.B_xyz shape {value_shape} does not match "
                        f"dims {dims}")
                raw_values = np.asarray(archive["B_xyz"])
                if raw_values.shape != (count, 3):
                    raise ValueError(
                        f"{label}.B_xyz shape {raw_values.shape} does not match "
                        f"dims {dims}")

            if any(n <= 0 for n in dims):
                raise ValueError(f"{label}.dims must be positive")
            count = math.prod(dims)
            if count > MAX_FIELD_GRID_POINTS:
                raise ValueError(
                    f"{label} has {count} points; limit is {MAX_FIELD_GRID_POINTS}")
            values = _real_array(raw_values, f"{label}.{field_key}")
            if has_flat:
                values = values.reshape(nz, ny, nx, 3)

            for axis, dimension in enumerate(dims):
                if spacing[axis] < 0.0 or (
                        dimension > 1 and spacing[axis] <= 0.0):
                    raise ValueError(
                        f"{label}.grid_spacing must be positive on "
                        "non-singleton axes")
                # Coil archives naturally write zero spacing when a sampled
                # observation axis has one point.  The spacing is immaterial on
                # that axis; canonicalize it so the registry parameter remains
                # portable to consumers that require a positive placeholder.
                if dimension == 1 and spacing[axis] == 0.0:
                    spacing[axis] = 1.0

            for axis in range(3):
                if dims[axis] <= 1:
                    continue
                first = _rounded_affine(
                    origin[axis], spacing[axis], 1, label)
                upper = _rounded_affine(
                    origin[axis], spacing[axis], dims[axis] - 1, label)
                previous = _rounded_affine(
                    origin[axis], spacing[axis], dims[axis] - 2, label)
                if first == origin[axis] or upper == previous:
                    raise ValueError(
                        f"{label} adjacent grid coordinates collapse in "
                        "floating-point precision")

            return {
                "origin": origin,
                "spacing": spacing,
                "dims": [float(value) for value in dims],
                "values": values.reshape(-1).tolist(),
            }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{label} {rel!r} contains an invalid field map: {exc}") from exc
