"""Shared filesystem helpers for the Quasar CLIs."""

from __future__ import annotations

from pathlib import Path


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
