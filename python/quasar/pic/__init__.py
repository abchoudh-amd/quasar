"""PIC front-end for Quasar."""

from typing import TYPE_CHECKING

from .. import distributed as _distributed_contract
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


def _run_distributed(input_deck, options, **kwargs):
    from ._distributed_runner import run as distributed_run
    return distributed_run(input_deck, options, **kwargs)


_distributed_contract._register_runner("pic", _run_distributed)

if TYPE_CHECKING:
    from .cli import prepare_run as prepare_run
    from .cli import run as run


def __getattr__(name: str):
    # Avoid pre-importing the module executed by `python -m quasar.pic.cli`.
    # The lazy attribute retains `from quasar.pic import prepare_run` without the
    # runpy warning about a module already present in sys.modules.
    if name == "prepare_run":
        from .cli import prepare_run as value
        globals()[name] = value
        return value
    if name == "run":
        from .cli import run as value
        globals()[name] = value
        return value
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
    "run",
    "total_em_energy",
    "total_kinetic_energy",
]
