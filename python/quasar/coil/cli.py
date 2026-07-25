"""Coil-design command-line entry point.

The ``run`` subcommand evaluates the requested fields for a YAML deck and
writes a ``.npz`` archive:

    python -m quasar.coil.cli run input.yaml

Invoke it as ``python -m quasar.coil.cli`` from the staged build-tree package.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .._core import magnetostatics as _ms
from .._paths import confine_output_path
from . import io as coil_io


def _needs_A(deck: coil_io.CoilDeck) -> bool:
    """True if any requested output field is derived from the vector potential A."""
    return bool({"A_xyz", "A_xyz_grid"} & set(deck.output.fields))


def _build_payload(
    deck: coil_io.CoilDeck, B: np.ndarray, A: np.ndarray | None = None
) -> tuple[dict[str, np.ndarray], set[str]]:
    """Assemble the .npz payload from the evaluated B field (shape (M, 3)), the
    optional vector potential A (shape (M, 3)), and the deck's requested output
    fields. Returns the payload plus the resolved field set (so callers don't
    recompute it). Pure (no I/O, no kernel) so the field-shaping and the
    *_grid/observation-kind guards are unit-testable without a GPU."""
    B = np.asarray(B)
    if B.ndim != 2 or B.shape[1] != 3:
        raise ValueError(f"B must have shape (M, 3), got {B.shape}")
    expected_points = math.prod(deck.observation.dims)
    if B.shape[0] != expected_points:
        raise ValueError(
            f"B has {B.shape[0]} rows, expected {expected_points} observations")
    try:
        b_is_finite = bool(np.all(np.isfinite(B)))
    except TypeError:
        b_is_finite = False
    if np.iscomplexobj(B) or not b_is_finite:
        raise ValueError("B must contain only finite real values")
    payload: dict[str, np.ndarray] = {}
    fields = set(deck.output.fields)

    if "B_xyz" in fields:
        payload["B_xyz"] = B
    if "B_magnitude" in fields:
        # Nested hypot is scale-safe; squaring first (as np.linalg.norm does)
        # can turn a finite magnitude into infinity for large components.
        with np.errstate(over="ignore", invalid="ignore"):
            magnitude = np.hypot(
                np.hypot(B[:, 0], B[:, 1]), B[:, 2])
        if not np.all(np.isfinite(magnitude)):
            raise ValueError(
                "B magnitude is not representable in output precision")
        payload["B_magnitude"] = magnitude
    if "B_xyz_grid" in fields:
        if deck.observation.kind != "grid":
            raise ValueError(
                "field B_xyz_grid requires observation.type == 'grid'")
        nx, ny, nz = deck.observation.dims
        payload["B_xyz_grid"] = B.reshape(nz, ny, nx, 3)

    if {"A_xyz", "A_xyz_grid"} & fields:
        if A is None:
            raise ValueError(
                "vector-potential output requested but A was not evaluated")
        A = np.asarray(A)
        if A.shape != B.shape:
            raise ValueError(
                f"A must have the same shape as B ({B.shape}), got {A.shape}")
        try:
            a_is_finite = bool(np.all(np.isfinite(A)))
        except TypeError:
            a_is_finite = False
        if np.iscomplexobj(A) or not a_is_finite:
            raise ValueError("A must contain only finite real values")
        if "A_xyz" in fields:
            payload["A_xyz"] = A
        if "A_xyz_grid" in fields:
            if deck.observation.kind != "grid":
                raise ValueError(
                    "field A_xyz_grid requires observation.type == 'grid'")
            nx, ny, nz = deck.observation.dims
            payload["A_xyz_grid"] = A.reshape(nz, ny, nx, 3)

    payload["dims"] = np.asarray(deck.observation.dims, dtype=np.int64)
    payload["observation_kind"] = np.asarray(
        deck.observation.kind, dtype="<U16")
    if deck.observation.kind == "grid":
        grid = deck.observation.detail
        payload["grid_origin"] = np.asarray(
            [grid.origin.x, grid.origin.y, grid.origin.z], dtype=np.float64)
        payload["grid_spacing"] = np.asarray(
            [grid.spacing.x, grid.spacing.y, grid.spacing.z], dtype=np.float64)
    return payload, fields


def _do_run(args: argparse.Namespace) -> int:
    deck = coil_io.load(args.input)

    if args.print_config:
        print(f"deck: {deck.raw}")

    # Select the evaluator by name through the registry (default Biot-Savart for
    # coil design); a deck overrides via the top-level `evaluator.type` key.
    evaluator = _ms.create_field_evaluator(deck.evaluator_type)
    evaluator.configure(deck.evaluator_params)
    B = evaluator.evaluate_B(deck.conductors, deck.observation.points)
    # B is shape (M, 3).
    # Vector potential A (B = curl A) only when an A_* field is requested; the
    # MHD coil-seeded IC differences A on a grid for a divergence-free seed.
    A = (evaluator.evaluate_A(deck.conductors, deck.observation.points)
         if _needs_A(deck) else None)

    payload, fields = _build_payload(deck, B, A)

    # Confine the deck-supplied output path to the input deck's directory so a
    # stray absolute path or "../" cannot write outside it.
    out_path = confine_output_path(Path(args.input).resolve().parent,
                                   deck.output.path, label="output.path")
    # Pass an already-open file instead of the path.  When given a path without
    # a ``.npz`` suffix, NumPy silently appends that suffix; in the degenerate
    # case ``output.path: .`` this would move the actual write to a sibling of
    # the confined deck directory.  A file object makes the validated path the
    # exact write target.
    with out_path.open("wb") as output_file:
        np.savez(output_file, **payload)

    if args.verbose:
        print(f"quasar coil: wrote {out_path} ({len(B)} points, "
              f"{', '.join(sorted(fields))})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quasar-coil",
        description="Quasar magnetostatics coil-design driver.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="evaluate a coil.yaml deck")
    run_p.add_argument("input", help="path to the input YAML deck")
    run_p.add_argument("--print-config", action="store_true",
                       help="echo the parsed deck before running")
    run_p.add_argument("--verbose", action="store_true",
                       help="print informational output (default: quiet)")
    run_p.set_defaults(func=_do_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
