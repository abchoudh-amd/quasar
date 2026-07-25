"""Shared filesystem and argument helpers for the Quasar CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path


def positive_int(value: str) -> int:
    """argparse ``type=`` for a strictly-positive integer (e.g. --steps-override)."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def confine_output_path(base: Path | str, rel: str, *, label: str = "output.path") -> Path:
    """Resolve a deck-supplied output path inside ``base`` and reject escapes.

    A stray absolute path or ``..`` segment that resolves outside ``base`` (the
    deck's own directory) is rejected, so a deck cannot write arbitrary files.
    Returns the resolved, confined path with its parent directory created.
    """
    base_dir = Path(base).resolve()
    out_path = (base_dir / rel).resolve()
    if not out_path.is_relative_to(base_dir):
        raise ValueError(f"{label} {rel!r} escapes the deck directory {base_dir}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def confine_input_path(base: Path | str, rel: str, *, label: str) -> Path:
    """Resolve a deck-supplied input path inside ``base`` and require a file.

    This is the read-only counterpart of :func:`confine_output_path`: it rejects
    absolute/``..`` escapes but never creates directories.
    """
    base_dir = Path(base).resolve()
    in_path = (base_dir / rel).resolve()
    if not in_path.is_relative_to(base_dir):
        raise ValueError(f"{label} {rel!r} escapes the deck directory {base_dir}")
    if not in_path.is_file():
        raise ValueError(f"{label} {rel!r} not found at {in_path}")
    return in_path
