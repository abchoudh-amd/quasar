"""Minimal PIC command-line entry point."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    _ = argv
    print("quasar pic: Python deck parsing is available; C++ driver runs simulations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
