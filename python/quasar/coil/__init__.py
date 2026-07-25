"""Coil-design front-end for the Quasar magnetostatics module.

Wraps the C++ Biot-Savart evaluator and geometry generators with a
Pythonic surface. ``evaluate_B(...)`` returns a NumPy array of shape
``(M, 3)`` carrying magnetic flux density in tesla at each observation
point.

Example::

    from quasar.coil import (
        BiotSavartEvaluator, ConductorSystem, PointCloud, Vec3,
        circular_loop,
    )

    cs = ConductorSystem()
    cs.add(circular_loop(
        center=Vec3(0, 0, 0), axis=Vec3(0, 0, 1),
        radius_m=0.1, n_segments=64, current_A=1.0,
    ))

    obs = PointCloud()
    obs.add(Vec3(0, 0, 0.05))

    B = BiotSavartEvaluator().evaluate_B(cs, obs)  # numpy (1, 3)
"""

from .._core import Vec3
from .._core.magnetostatics import (
    BiotSavartConfig,
    BiotSavartEvaluator,
    ConductorSystem,
    FileGridEvaluator,
    Filament,
    LineProbe,
    ObservationGrid,
    PlaneSlice,
    PointCloud,
    circular_loop,
    generic_polyline,
    helix,
    polygon,
    racetrack,
    solenoid,
)

__all__ = [
    "Vec3",
    "Filament",
    "ConductorSystem",
    "PointCloud",
    "ObservationGrid",
    "PlaneSlice",
    "LineProbe",
    "BiotSavartConfig",
    "BiotSavartEvaluator",
    "FileGridEvaluator",
    "circular_loop",
    "helix",
    "solenoid",
    "racetrack",
    "polygon",
    "generic_polyline",
]
