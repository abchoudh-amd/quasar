"""Ideal-MHD front-end for Quasar."""

from typing import TYPE_CHECKING

from .._core.mhd import (
    Grid2D,
    MhdBoundarySpec,
    MhdConfig,
    MhdSolver2D,
    registered_ct_schemes,
    registered_integrators,
    registered_mhd_field_boundaries,
    registered_mhd_fluid_boundaries,
    registered_positivity_limiters,
    registered_reconstructions,
    registered_riemann_solvers,
)
from .io import MhdDeck, build_initial_state, load, parse

if TYPE_CHECKING:
    from .cli import prepare_run as prepare_run


def __getattr__(name: str):
    # Keep `python -m quasar.mhd.cli` from importing its target module while
    # Python is still initializing the parent package (which otherwise emits a
    # runpy RuntimeWarning). Preserve the public convenience import lazily.
    if name == "prepare_run":
        from .cli import prepare_run
        globals()[name] = prepare_run
        return prepare_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Grid2D",
    "MhdBoundarySpec",
    "MhdConfig",
    "MhdDeck",
    "MhdSolver2D",
    "build_initial_state",
    "load",
    "parse",
    "prepare_run",
    "registered_ct_schemes",
    "registered_integrators",
    "registered_mhd_field_boundaries",
    "registered_mhd_fluid_boundaries",
    "registered_positivity_limiters",
    "registered_reconstructions",
    "registered_riemann_solvers",
]
