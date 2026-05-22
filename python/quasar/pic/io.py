"""YAML schema helpers for PIC input decks.

The C++ solver owns the numerical kernels; this module keeps the user-facing
deck conventions in one place and deliberately stays lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Domain:
    nx: int
    ny: int
    lx_m: float
    ly_m: float


@dataclass
class Numerics:
    fdtd_order: int = 2
    shape: str = "cic"
    current_filter: list[dict] = field(default_factory=list)


@dataclass
class PicDeck:
    domain: Domain
    numerics: Numerics = field(default_factory=Numerics)
    units: str = "SI"

    def validate(self) -> None:
        if self.numerics.fdtd_order not in (2, 4):
            raise ValueError("fdtd_order must be 2 or 4")
        if self.numerics.shape not in ("cic", "tsc"):
            raise ValueError("shape must be 'cic' or 'tsc'")
        if self.units not in ("SI", "normalized"):
            raise ValueError("units must be 'SI' or 'normalized'")
