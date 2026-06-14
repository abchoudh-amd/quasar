"""``quasar coil`` command-line entry point.

Phase 2.D supports a single subcommand, ``run``, which evaluates the B
field for a YAML deck and writes a ``.npz`` archive:

    python -m quasar.coil.cli run input.yaml

A top-level ``quasar`` console script can dispatch to this module in a
later phase; for now ``python -m quasar.coil.cli`` is the entry point.
"""

from __future__ import annotations

import argparse
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
    payload: dict[str, np.ndarray] = {}
    fields = set(deck.output.fields)

    if "B_xyz" in fields:
        payload["B_xyz"] = B
    if "B_magnitude" in fields:
        payload["B_magnitude"] = np.linalg.norm(B, axis=1)
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
    return payload, fields


def _do_run(args: argparse.Namespace) -> int:
    deck = coil_io.load(args.input)

    if args.print_config:
        print(f"deck: {deck.raw}")

    # Select the evaluator by name through the registry (default Biot-Savart for
    # coil design); a deck overrides via the top-level `evaluator.type` key.
    evaluator = _ms.create_field_evaluator(deck.evaluator_type)
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
    np.savez(out_path, **payload)

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
