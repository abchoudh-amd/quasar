"""Safe opaque diagnostic-continuation fragments for HDF5 checkpoints.

The native checkpoint layer treats each rank's fragment as bytes and supplies
the collective framing, checksum, size bound, and atomic file publication.
This module defines the application payload: an uncompressed NPZ containing
only explicitly validated, non-object NumPy arrays.  Loading never enables
pickle and inspects the ZIP directory before NumPy allocates an array.
"""

from __future__ import annotations

import io
import re
import zipfile
from typing import Any, Mapping

import numpy as np


MAX_FRAGMENT_BYTES = 512 * 1024 * 1024
MAX_ARRAYS = 4096
_KEY = re.compile(r"[A-Za-z0-9_.-]{1,200}\Z")


def encode_fragment(payload: Mapping[str, Any]) -> bytes:
    """Encode a bounded mapping of non-object arrays without pickle."""

    if not payload or len(payload) > MAX_ARRAYS:
        raise ValueError("checkpoint diagnostic fragment has an invalid array count")
    arrays: dict[str, np.ndarray] = {}
    total = 0
    for key, raw in payload.items():
        if not isinstance(key, str) or _KEY.fullmatch(key) is None:
            raise ValueError("checkpoint diagnostic fragment has an invalid key")
        array = np.asarray(raw)
        if array.dtype.hasobject:
            raise ValueError(
                "checkpoint diagnostic fragments cannot contain object arrays")
        if array.ndim > 4:
            raise ValueError(
                "checkpoint diagnostic fragment arrays may have at most four dimensions")
        total += int(array.nbytes)
        if total > MAX_FRAGMENT_BYTES:
            raise ValueError("checkpoint diagnostic fragment is too large")
        arrays[key] = np.ascontiguousarray(array)

    stream = io.BytesIO()
    np.savez(stream, **arrays)
    result = stream.getvalue()
    if len(result) > MAX_FRAGMENT_BYTES:
        raise ValueError("checkpoint diagnostic fragment is too large")
    return result


def decode_fragment(payload: bytes | bytearray | memoryview) -> dict[str, np.ndarray]:
    """Decode a fragment after bounding its compressed and expanded sizes."""

    raw = bytes(payload)
    if not raw or len(raw) > MAX_FRAGMENT_BYTES:
        raise ValueError("checkpoint diagnostic fragment has an invalid size")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            members = archive.infolist()
            if not members or len(members) > MAX_ARRAYS:
                raise ValueError(
                    "checkpoint diagnostic fragment has an invalid array count")
            expanded = 0
            for member in members:
                name = member.filename
                key = name[:-4] if name.endswith(".npy") else ""
                if (_KEY.fullmatch(key) is None or "/" in name or "\\" in name
                        or member.file_size < 0):
                    raise ValueError(
                        "checkpoint diagnostic fragment has an invalid member")
                expanded += int(member.file_size)
                if expanded > MAX_FRAGMENT_BYTES:
                    raise ValueError(
                        "checkpoint diagnostic fragment expands beyond its size bound")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("checkpoint diagnostic fragment is not a valid NPZ") from error

    result: dict[str, np.ndarray] = {}
    total = 0
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            for key in archive.files:
                if _KEY.fullmatch(key) is None or key in result:
                    raise ValueError(
                        "checkpoint diagnostic fragment has an invalid array key")
                array = np.asarray(archive[key])
                if array.dtype.hasobject or array.ndim > 4:
                    raise ValueError(
                        "checkpoint diagnostic fragment has an unsafe array")
                total += int(array.nbytes)
                if total > MAX_FRAGMENT_BYTES:
                    raise ValueError(
                        "checkpoint diagnostic fragment arrays exceed the size bound")
                result[key] = np.array(array, copy=True, order="C")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, ValueError) and str(error).startswith(
                "checkpoint diagnostic fragment"):
            raise
        raise ValueError(
            "checkpoint diagnostic fragment contains an invalid NumPy array") from error
    return result


def scalar_text(payload: Mapping[str, np.ndarray], key: str,
                *, maximum: int = 4096) -> str:
    """Read one bounded Unicode scalar from a decoded fragment."""

    if key not in payload:
        raise ValueError(f"checkpoint diagnostic fragment is missing {key}")
    array = np.asarray(payload[key])
    if array.shape not in ((), (1,)) or array.dtype.kind not in ("U", "S"):
        raise ValueError(f"checkpoint diagnostic fragment {key} is not text")
    value = str(array.reshape(-1)[0])
    if not value or len(value.encode("utf-8")) > maximum or "\0" in value:
        raise ValueError(f"checkpoint diagnostic fragment {key} is invalid")
    return value

def scalar_int(payload: Mapping[str, np.ndarray], key: str,
               *, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    """Read one exact bounded integer scalar from a decoded fragment."""

    if key not in payload:
        raise ValueError(f"checkpoint diagnostic fragment is missing {key}")
    array = np.asarray(payload[key])
    if array.shape not in ((), (1,)) or array.dtype.kind not in ("i", "u"):
        raise ValueError(f"checkpoint diagnostic fragment {key} is not an integer")
    value = int(array.reshape(-1)[0])
    if value < minimum or value > maximum:
        raise ValueError(f"checkpoint diagnostic fragment {key} is out of range")
    return value
