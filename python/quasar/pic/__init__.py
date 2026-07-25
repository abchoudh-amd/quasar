"""PIC front-end for Quasar."""

from typing import TYPE_CHECKING

from .._core.pic import (
    EmPic2D3V,
    EmPicConfig,
    Grid2D,
    Normalization,
    ParticleSpecies,
    SpeciesConfig,
    alive_count,
    gauss_residual,
    total_em_energy,
    total_kinetic_energy,
)
from ._units import Units

if TYPE_CHECKING:
    from .cli import prepare_run as prepare_run


def __getattr__(name: str):
    # Avoid pre-importing the module executed by `python -m quasar.pic.cli`.
    # The lazy attribute retains `from quasar.pic import prepare_run` without the
    # runpy warning about a module already present in sys.modules.
    if name == "prepare_run":
        from .cli import prepare_run
        globals()[name] = prepare_run
        return prepare_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EmPic2D3V",
    "EmPicConfig",
    "Grid2D",
    "Normalization",
    "ParticleSpecies",
    "SpeciesConfig",
    "Units",
    "alive_count",
    "gauss_residual",
    "prepare_run",
    "total_em_energy",
    "total_kinetic_energy",
]
