"""Quasar — HIP-accelerated numerical simulation framework.

The top-level ``quasar`` package re-exports a small core surface from the
compiled C++ extension (``quasar._core``). Physics submodules such as
``quasar.coil`` are the user-facing entry points.
"""

from ._core import Vec3, mu0, mu0_over_4pi, pi  # noqa: F401

__version__ = "0.1.0"
__all__ = ["Vec3", "mu0", "mu0_over_4pi", "pi", "__version__"]
