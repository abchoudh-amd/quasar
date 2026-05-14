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

from . import BiotSavartEvaluator
from . import io as coil_io


def _do_run(args: argparse.Namespace) -> int:
    deck = coil_io.load(args.input)

    if args.print_config:
        print(f"deck: {deck.raw}")

    evaluator = BiotSavartEvaluator()
    B = evaluator.evaluate_B(deck.conductors, deck.observation.points)
    # B is shape (M, 3).

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

    payload["dims"] = np.asarray(deck.observation.dims, dtype=np.int64)
    payload["observation_kind"] = np.asarray(
        deck.observation.kind, dtype="<U16")

    out_path = Path(deck.output.path)
    if not out_path.is_absolute():
        out_path = Path(args.input).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)

    if not args.quiet:
        print(f"quasar coil: wrote {out_path} ({len(B)} points, "
              f"{', '.join(sorted(fields))})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quasar-coil",
        description="Quasar magnetostatics coil-design driver.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="evaluate a coil.yaml deck")
    run_p.add_argument("input", help="path to the input YAML deck")
    run_p.add_argument("--print-config", action="store_true",
                       help="echo the parsed deck before running")
    run_p.add_argument("--quiet", action="store_true",
                       help="suppress informational output")
    run_p.set_defaults(fn=_do_run)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
