"""YAML schema for ideal-MHD input decks.

The C++ solver owns the numerical kernels (flux reconstruction, HLLD Riemann
solver, constrained transport, SSP-RK integration, positivity limiting); this
module models the user-facing deck as a small set of dataclasses, parses YAML into
them, validates, and builds the seeded initial-condition arrays. The CLI
(``quasar.mhd.cli``) consumes the parsed deck.

Top-level structure::

    units: normalized
    domain: {nx, ny, lx_m, ly_m, origin_x_m?, origin_y_m?}
    geometry: cartesian | cylindrical
    numerics:
      gamma, reconstruction, riemann, integrator, ct, positivity,
      rho_floor, p_floor, cfl
    initial: {type: <token>, params: {...}}
    time: {dt_s: auto|<float>, steps, t_end?}
    diagnostics: {output_path, cadence, fields, divb}
    boundary: {fluid, field}          # scalar | 4-list | side-keyed map

The registry-name fields (reconstruction / riemann / integrator / ct /
positivity / boundary.*) are validated against the LIVE C++ registries exposed at
``_core.mhd.registered_*`` so a newly-registered scheme needs no Python edit
(mirrors ``quasar.pic.io``).

``initial.type`` must be one of the canonical tokens
``{brio_wu, alfven_wave, orszag_tang, blast, rotor, confined_blob}``;
``initial.params`` is validated for the selected generator below.

Initial-condition projection accuracy
-------------------------------------

The evolved state is finite-volume data: cell averages for the fluid variables
and ``bz``, face averages for ``bx``/``by``. The builders below differ in how
exactly they realize that projection, which matters only for formal convergence
studies:

* ``alfven_wave`` is an **exact finite-volume projection**. It applies the
  analytic ``sinc(k*dx/2)`` element averages and restores the unresolved
  sub-cell sin/cos energy variance, so it is suitable for measuring the design
  order of MP5/MP7.
* ``orszag_tang`` is a **midpoint (second-order) projection**. Its smooth
  trigonometric profile is evaluated at element centers and stored as if it were
  the element average, which differs from the true moment by ``O(h^2)``. This is
  a valid, standard, reproducible benchmark state, but it caps a continuum
  convergence study at second-order *initialization* accuracy regardless of the
  reconstruction selected. Use ``alfven_wave`` for formal order verification, or
  upgrade this builder to averaged moments first.
* ``brio_wu``, ``blast``, ``rotor``, and ``confined_blob`` are **discontinuous
  benchmarks**, evaluated at element centers. Away from the discontinuity the
  states are piecewise constant, so the midpoint value *is* the exact element
  average; only the cut elements differ from the true moment. These problems have
  no high-order continuum convergence rate to cap.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from .. import _core
from .._deck import load_yaml as _load_yaml
from .._deck import require as _require
from .._deck import (
    as_finite as _as_finite,
    require_finite as _require_finite,
    require_positive_finite as _require_positive_finite,
    parse_side_map as _parse_side_map,
)
from ..coil.io import build_conductor_system
from . import _units as mhd_units


# Sanity ceilings on deck-supplied sizes that flow into device allocations
# (mirrors quasar.pic.io). Guard against typos / hostile decks, not legitimate
# large runs.
MAX_GRID_DIM = 1 << 16        # 65536 cells per axis
MAX_GRID_CELLS = 1 << 30      # ~1.07e9 cells total

# Canonical initial-condition tokens (the SINGLE source of truth, mirrored by the
# example decks). Validation rejects any initial.type not in this set.
INITIAL_TYPES = ("brio_wu", "alfven_wave", "orszag_tang", "blast", "rotor",
                 "confined_blob")

# Conserved-state components written to / read from the solver and the .npz.
STATE_COMPONENTS = ("rho", "mx", "my", "mz", "energy", "bx", "by", "bz")

_TOP_LEVEL_KEYS = {
    "units", "domain", "geometry", "numerics", "initial", "time",
    "diagnostics", "boundary", "background_field",
}


def _as_exact_int(value, context: str) -> int:
    """Return an integer-valued object without silently truncating floats.

    YAML integers arrive as :class:`int`; ``operator.index`` also admits NumPy
    integer scalars used by programmatic callers. Booleans and floats (including
    integral-looking values such as ``16.0``) are rejected so a misspelled deck
    cannot quietly change its grid, run length, cadence, or mode number.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} must be an integer, not a boolean")
    try:
        return int(operator.index(value))
    except TypeError:
        raise ValueError(f"{context} must be an integer") from None


def _as_bool(value, context: str) -> bool:
    """Require a real boolean instead of applying Python truthiness."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} must be a boolean")
    return bool(value)


def _mapping(value, context: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _reject_unknown(d: dict, allowed, context: str) -> None:
    unknown = set(d) - set(allowed)
    if unknown:
        names = ", ".join(sorted(repr(name) for name in unknown))
        raise ValueError(f"{context}: unknown key(s): {names}")


def _finite_pair(value, context: str) -> tuple[float, float]:
    """Return an exactly two-element finite coordinate pair."""
    if isinstance(value, (str, bytes, dict)):
        raise ValueError(f"{context} must be a two-element sequence")
    try:
        if len(value) != 2:
            raise ValueError(f"{context} must contain exactly two values")
        return (_as_finite(value[0], f"{context}[0]"),
                _as_finite(value[1], f"{context}[1]"))
    except TypeError:
        raise ValueError(f"{context} must be a two-element sequence") from None


_PRIMITIVE_KEYS = ("rho", "p", "vx", "vy", "vz", "bx", "by", "bz")


def _validate_brio_wu_params(params: dict) -> None:
    _reject_unknown(params, {"interface", "left", "right"},
                    "initial.params")
    interface = _as_finite(
        _require(params, "interface", "initial.params"),
        "initial.params.interface")
    _ = interface
    states: dict[str, dict] = {}
    for side in ("left", "right"):
        state = _mapping(
            _require(params, side, "initial.params"),
            f"initial.params.{side}")
        _reject_unknown(state, _PRIMITIVE_KEYS, f"initial.params.{side}")
        missing = [name for name in _PRIMITIVE_KEYS if name not in state]
        if missing:
            raise ValueError(
                f"initial.params.{side}: missing required field(s) {missing}")
        for name in _PRIMITIVE_KEYS:
            _as_finite(state[name], f"initial.params.{side}.{name}")
        if _as_finite(state["rho"], f"initial.params.{side}.rho") <= 0.0:
            raise ValueError(f"initial.params.{side}.rho must be positive")
        if _as_finite(state["p"], f"initial.params.{side}.p") <= 0.0:
            raise ValueError(f"initial.params.{side}.p must be positive")
        states[side] = state
    bx_left = _as_finite(states["left"]["bx"], "initial.params.left.bx")
    bx_right = _as_finite(states["right"]["bx"], "initial.params.right.bx")
    if bx_left != bx_right:
        raise ValueError(
            "brio_wu requires a continuous normal magnetic field: "
            "initial.params.left.bx must equal initial.params.right.bx")


def _alfven_reference_bx(deck: "MhdDeck") -> float:
    """Axial field that fixes the Alfven eigenvector's propagation sign."""
    b0 = _as_finite(deck.initial.params.get("b0", 1.0),
                    "initial.params.b0")
    bg = deck.background
    # A uniform field-split guide field contributes to the same total axial
    # field as the evolving state's uniform bx. File/non-uniform profiles do not
    # have one global axial value, so the analytic IC can only use its own b0.
    if (isinstance(bg.enabled, (bool, np.bool_)) and bool(bg.enabled)
            and bg.file is None and bg.a_file is None
            and bg.conductors is None
            and bg.profile == "uniform"):
        bx0 = _as_finite(bg.bx0, "background_field.bx0")
        with np.errstate(over="ignore", invalid="ignore"):
            b0 += bx0
        if not math.isfinite(b0):
            raise ValueError(
                "alfven_wave total axial background field is not representable")
    return b0


def _validate_alfven_wave_params(deck: "MhdDeck", params: dict) -> None:
    _reject_unknown(
        params, {"rho", "p", "b0", "amplitude", "wavenumber",
                 "polarization"}, "initial.params")
    _require_positive_finite(params.get("rho", 1.0), "initial.params.rho")
    _require_positive_finite(params.get("p", 0.1), "initial.params.p")
    _require_finite(params.get("amplitude", 1.0e-6),
                    "initial.params.amplitude")
    if _alfven_reference_bx(deck) == 0.0:
        raise ValueError(
            "alfven_wave requires a nonzero total axial background field")
    n = _as_exact_int(params.get("wavenumber", 1),
                      "initial.params.wavenumber")
    if n <= 0:
        raise ValueError("initial.params.wavenumber must be positive")
    # The circular mode needs both sine and cosine quadratures. At or above the
    # x-grid Nyquist mode one quadrature aliases away, so it is not resolved.
    if 2 * n >= deck.domain.nx:
        raise ValueError(
            "initial.params.wavenumber must be below the x-grid Nyquist mode")
    polarization = params.get("polarization", "circular")
    if not isinstance(polarization, str) or polarization != "circular":
        raise ValueError(
            "alfven_wave currently supports only polarization='circular'")


def _validate_orszag_tang_params(params: dict) -> None:
    _reject_unknown(params, {"b0"}, "initial.params")
    _require_finite(params.get("b0", 1.0),
                    "initial.params.b0")


def _validate_blast_params(params: dict) -> None:
    _reject_unknown(
        params, {"rho_ambient", "p_ambient", "p_core", "r_in", "center",
                 "bx", "by", "bz"}, "initial.params")
    _require_positive_finite(params.get("rho_ambient", 1.0),
                             "initial.params.rho_ambient")
    _require_positive_finite(params.get("p_ambient", 0.1),
                             "initial.params.p_ambient")
    _require_positive_finite(params.get("p_core", 10.0),
                             "initial.params.p_core")
    _require_positive_finite(params.get("r_in", 0.1),
                             "initial.params.r_in")
    _finite_pair(params.get("center", [0.0, 0.0]),
                 "initial.params.center")
    for name in ("bx", "by", "bz"):
        _require_finite(params.get(name, 0.0), f"initial.params.{name}")


def _validate_rotor_params(params: dict) -> None:
    _reject_unknown(
        params, {"center", "r0", "r1", "rho_in", "rho_out", "u0", "p",
                 "bx", "by", "bz"}, "initial.params")
    _finite_pair(params.get("center", [0.5, 0.5]),
                 "initial.params.center")
    r0 = _as_finite(params.get("r0", 0.1), "initial.params.r0")
    r1 = _as_finite(params.get("r1", 0.115), "initial.params.r1")
    if r0 <= 0.0:
        raise ValueError("initial.params.r0 must be positive")
    if r1 <= r0:
        raise ValueError("initial.params.r1 must be greater than r0")
    for name, default in (("rho_in", 10.0), ("rho_out", 1.0), ("p", 1.0)):
        _require_positive_finite(params.get(name, default),
                                 f"initial.params.{name}")
    u0 = _as_finite(params.get("u0", 2.0), "initial.params.u0")
    if not math.isfinite(u0 / r0):
        raise ValueError("rotor angular velocity u0/r0 is not representable")
    for name in ("bx", "by", "bz"):
        _require_finite(params.get(name, 0.0), f"initial.params.{name}")


def _validate_confined_blob_params(params: dict, domain: "Domain") -> None:
    _reject_unknown(
        params, {"bz", "rho_in", "rho_out", "p_in", "p_out", "blob_half"},
        "initial.params")
    for name, default in (("rho_in", 10.0), ("rho_out", 1.0),
                          ("p_in", 1.0), ("p_out", 0.1)):
        _require_positive_finite(params.get(name, default),
                                 f"initial.params.{name}")
    _require_finite(params.get("bz", 0.1), "initial.params.bz")
    _require_positive_finite(
        params.get("blob_half", 0.25 * min(domain.lx_m, domain.ly_m)),
        "initial.params.blob_half")


@dataclass
class Domain:
    nx: int
    ny: int
    lx_m: float
    ly_m: float
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0


@dataclass
class Numerics:
    gamma: float = 5.0 / 3.0
    reconstruction: str = "mp7"
    riemann: str = "hlld"
    integrator: str = "ssprk3"
    ct: str = "fd_ct_christlieb"
    positivity: str = "troubled_cell"
    rho_floor: float = 1.0e-8
    p_floor: float = 1.0e-9
    cfl: float = 0.4


@dataclass
class Initial:
    type: str
    params: dict = field(default_factory=dict)


@dataclass
class Time:
    dt_s: Union[float, str] = "auto"
    steps: int = 100
    t_end: Union[float, None] = None


@dataclass
class Diagnostics:
    output_path: str = "out.npz"
    cadence: int = 0
    fields: list[str] = field(default_factory=lambda: list(STATE_COMPONENTS))
    divb: bool = True

    def __post_init__(self) -> None:
        self.fields = [str(name).lower() for name in self.fields]


@dataclass
class BoundaryConfig:
    """Per-side boundary kinds. Order: [x_min, x_max, y_min, y_max].

    ``fluid`` and ``field`` are independent (a wall imposes different
    symmetries on momentum than on the magnetic field). Names are validated
    against the live C++ registries.
    """
    fluid: tuple[str, str, str, str] = (
        "periodic", "periodic", "periodic", "periodic")
    field: tuple[str, str, str, str] = (
        "periodic", "periodic", "periodic", "periodic")


@dataclass
class BackgroundConfig:
    """Static background magnetic field B0 for the field-split form B = B0 + b.

    Disabled by default (``enabled=False`` => the solver runs the zero-B0 fast
    path, bit-identical to the no-background solver). ``profile`` is a registry
    name validated against the live C++ registry
    (``_core.mhd.registered_mhd_background_profiles()``); ``bx0/by0/bz0`` are the
    uniform-vector parameters consumed when ``profile == "uniform"`` and no
    ``file`` is given. ``params`` configures any other registered analytic
    profile.
    ``file`` (optional) names an npz holding the staggered B0 arrays directly.
    ``conductors`` (optional) holds the coil/PIC conductor schema and samples
    its vector potential directly on the padded MHD corner grid.
    """
    enabled: bool = False
    profile: str = "uniform"
    bx0: float = 0.0
    by0: float = 0.0
    bz0: float = 0.0
    params: dict = field(default_factory=dict)
    file: Union[str, None] = None
    # Coil vector-potential mode: an npz with the cell-corner 'A_xyz_grid' (from
    # the coil CLI). The in-plane B0 is the discrete curl of the out-of-plane A,
    # so its staggered divergence telescopes; the uniform out-of-plane bz0 is
    # added as the toroidal component. This is one supported non-uniform input
    # paths. Its curl may be nonzero: the exact split-energy rate transformation
    # supports current-carrying backgrounds as long as B0 is discretely solenoidal.
    a_file: Union[str, None] = None
    conductors: Union[list[dict], None] = None


@dataclass
class MhdDeck:
    domain: Domain
    numerics: Numerics = field(default_factory=Numerics)
    initial: Union[Initial, None] = None
    time: Time = field(default_factory=Time)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    units: str = "normalized"
    geometry: str = "cartesian"
    raw: dict = field(default_factory=dict)
    # Directory the deck was loaded from, set by ``load()``; used to resolve and
    # confine a ``background_field.file`` path. None when the deck was built by
    # ``parse()`` alone (no source file); a file-mode background then resolves
    # relative to the current working directory.
    source_dir: Union[Path, None] = None

    def validate(self) -> None:
        if self.units not in ("SI", "normalized"):
            raise ValueError("units must be 'SI' or 'normalized'")
        if self.geometry not in ("cartesian", "cylindrical"):
            raise ValueError("geometry must be 'cartesian' or 'cylindrical'")
        self._validate_domain()
        self._validate_numerics()
        self._validate_initial()
        self._validate_time()
        self._validate_diagnostics()
        self._validate_boundary()
        self._validate_background()
        if self.geometry == "cylindrical":
            self._validate_cylindrical()

    def _validate_domain(self) -> None:
        nx = _as_exact_int(self.domain.nx, "domain.nx")
        ny = _as_exact_int(self.domain.ny, "domain.ny")
        if nx <= 0 or ny <= 0:
            raise ValueError("domain.nx and domain.ny must be positive")
        if nx > MAX_GRID_DIM or ny > MAX_GRID_DIM:
            raise ValueError(
                f"domain.nx/ny must be <= {MAX_GRID_DIM} (got "
                f"{nx}x{ny})")
        if nx * ny > MAX_GRID_CELLS:
            raise ValueError(f"domain.nx*ny must be <= {MAX_GRID_CELLS} cells")
        lx = _as_finite(self.domain.lx_m, "domain.lx_m")
        ly = _as_finite(self.domain.ly_m, "domain.ly_m")
        if lx <= 0.0 or ly <= 0.0:
            raise ValueError("domain.lx_m and domain.ly_m must be positive")
        x0 = _as_finite(self.domain.origin_x_m, "domain.origin_x_m")
        y0 = _as_finite(self.domain.origin_y_m, "domain.origin_y_m")
        x1 = x0 + lx
        y1 = y0 + ly
        _require_finite(x1, "domain upper x bound")
        _require_finite(y1, "domain upper y bound")
        dx = lx / nx
        dy = ly / ny
        _require_positive_finite(dx, "domain x spacing")
        _require_positive_finite(dy, "domain y spacing")
        # A formally positive length is still unusable when translating it to a
        # large origin erases cell-center distinctions in binary64. Catch both
        # ends, matching the coordinate convention used by Grid2D.
        if (x0 + 0.5 * dx == x0 or x0 + (nx - 0.5) * dx == x1
                or y0 + 0.5 * dy == y0 or y0 + (ny - 0.5) * dy == y1):
            raise ValueError(
                "domain cell coordinates collapse in floating-point precision")

    def _validate_numerics(self) -> None:
        # gamma must exceed 1 (the ideal-gas adiabatic index; gamma<=1 makes the
        # energy/pressure conversion p = (gamma-1)*rho*e nonphysical).
        if _as_finite(self.numerics.gamma, "numerics.gamma") <= 1.0:
            raise ValueError("numerics.gamma must be > 1")
        # Each scheme axis is validated against the live C++ registry so a newly-
        # registered scheme needs no Python edit.
        mhd = _core.mhd
        _check_registered(self.numerics.reconstruction,
                          mhd.registered_reconstructions(),
                          "numerics.reconstruction")
        _check_registered(self.numerics.riemann,
                          mhd.registered_riemann_solvers(),
                          "numerics.riemann")
        _check_registered(self.numerics.integrator,
                          mhd.registered_integrators(),
                          "numerics.integrator")
        _check_registered(self.numerics.ct,
                          mhd.registered_ct_schemes(),
                          "numerics.ct")
        _check_registered(self.numerics.positivity,
                          mhd.registered_positivity_limiters(),
                          "numerics.positivity")
        _require_positive_finite(self.numerics.rho_floor, "numerics.rho_floor")
        _require_positive_finite(self.numerics.p_floor, "numerics.p_floor")
        cfl = _as_finite(self.numerics.cfl, "numerics.cfl")
        if not (0.0 < cfl <= 1.0):
            raise ValueError("numerics.cfl must be in (0, 1]")

    def _validate_initial(self) -> None:
        if self.initial is None:
            raise ValueError("deck must define an 'initial' section")
        if self.initial.type not in INITIAL_TYPES:
            raise ValueError(
                f"initial.type {self.initial.type!r} must be one of "
                f"{list(INITIAL_TYPES)}")
        if not isinstance(self.initial.params, dict):
            raise ValueError("initial.params must be a mapping")
        validators = {
            "brio_wu": lambda: _validate_brio_wu_params(self.initial.params),
            "alfven_wave": lambda: _validate_alfven_wave_params(
                self, self.initial.params),
            "orszag_tang": lambda: _validate_orszag_tang_params(
                self.initial.params),
            "blast": lambda: _validate_blast_params(self.initial.params),
            "rotor": lambda: _validate_rotor_params(self.initial.params),
            "confined_blob": lambda: _validate_confined_blob_params(
                self.initial.params, self.domain),
        }
        validators[self.initial.type]()

    def _validate_time(self) -> None:
        if isinstance(self.time.dt_s, str) and self.time.dt_s != "auto":
            raise ValueError("time.dt_s must be a float or the string 'auto'")
        if not isinstance(self.time.dt_s, str):
            _require_positive_finite(self.time.dt_s, "time.dt_s")
        steps = _as_exact_int(self.time.steps, "time.steps")
        if steps <= 0:
            raise ValueError("time.steps must be positive")
        if self.time.t_end is not None:
            _require_positive_finite(self.time.t_end, "time.t_end")

    def _validate_diagnostics(self) -> None:
        if (not isinstance(self.diagnostics.output_path, str)
                or not self.diagnostics.output_path.strip()):
            raise ValueError(
                "diagnostics.output_path must be a non-empty string")
        cadence = _as_exact_int(self.diagnostics.cadence, "diagnostics.cadence")
        if cadence < 0:
            raise ValueError("diagnostics.cadence must be >= 0")
        _as_bool(self.diagnostics.divb, "diagnostics.divb")
        for field_name in self.diagnostics.fields:
            if field_name not in STATE_COMPONENTS:
                raise ValueError(
                    f"diagnostics.fields entry {field_name!r} must be one of "
                    f"{list(STATE_COMPONENTS)}")

    def _validate_boundary(self) -> None:
        allowed_fluid = set(_core.mhd.registered_mhd_fluid_boundaries())
        for i, bc in enumerate(self.boundary.fluid):
            if bc == "internal":
                raise ValueError(
                    f"boundary.fluid[{i}] = 'internal' is reserved for "
                    "the distributed tile runtime")
            if bc not in allowed_fluid:
                raise ValueError(
                    f"boundary.fluid[{i}] = {bc!r} must be one of "
                    f"{sorted(allowed_fluid)}")
        allowed_field = set(_core.mhd.registered_mhd_field_boundaries())
        for i, bc in enumerate(self.boundary.field):
            if bc == "internal":
                raise ValueError(
                    f"boundary.field[{i}] = 'internal' is reserved for "
                    "the distributed tile runtime")
            if bc not in allowed_field:
                raise ValueError(
                    f"boundary.field[{i}] = {bc!r} must be one of "
                    f"{sorted(allowed_field)}")
        if self.geometry != "cylindrical" and (
                "axis" in self.boundary.fluid or "axis" in self.boundary.field):
            raise ValueError("boundary kind 'axis' requires geometry 'cylindrical'")
        for side in range(4):
            if ((self.boundary.fluid[side] == "periodic") !=
                    (self.boundary.field[side] == "periodic")):
                raise ValueError(
                    "fluid and field periodicity must match on every side")
        for axis in range(2):
            lo, hi = 2 * axis, 2 * axis + 1
            if ((self.boundary.fluid[lo] == "periodic") !=
                    (self.boundary.fluid[hi] == "periodic") or
                    (self.boundary.field[lo] == "periodic") !=
                    (self.boundary.field[hi] == "periodic")):
                raise ValueError(
                    "periodic boundaries must be selected on both sides of an axis")

    def _validate_background(self) -> None:
        bg = self.background
        _as_bool(bg.enabled, "background_field.enabled")
        if not bg.enabled:
            return
        analytic_mode = (bg.file is None and bg.a_file is None
                         and bg.conductors is None)
        # A profile is consumed only in analytic mode. Explicit source modes
        # replace all native placeholder samples, so a stale profile name must
        # not make otherwise valid file input fail.
        if analytic_mode:
            _check_registered(bg.profile,
                              _core.mhd.registered_mhd_background_profiles(),
                              "background_field.profile")
        if not isinstance(bg.params, dict):
            raise ValueError("background_field.params must be a mapping")
        for key, value in bg.params.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "background_field.params keys must be non-empty strings")
            if key == "vacuum_project":
                continue
            _require_finite(value, f"background_field.params.{key}")
        if "vacuum_project" in bg.params:
            project = bg.params["vacuum_project"]
            if not isinstance(project, (bool, np.bool_)):
                raise ValueError(
                    "background_field.params.vacuum_project must be a boolean")
            if bg.a_file is None and bg.conductors is None:
                raise ValueError(
                    "background_field.params.vacuum_project requires "
                    "background_field.a_file or background_field.conductors")
            if project and self.geometry != "cylindrical":
                raise ValueError(
                    "background_field.params.vacuum_project is currently "
                    "defined only for cylindrical A_phi data")
        # Legacy uniform-vector values are finite and mode-specific. Reject
        # ignored nonzero values rather than silently accepting a misspelled or
        # contradictory deck.
        _require_finite(bg.bx0, "background_field.bx0")
        _require_finite(bg.by0, "background_field.by0")
        _require_finite(bg.bz0, "background_field.bz0")
        if bg.file is not None and not str(bg.file).strip():
            raise ValueError("background_field.file must be a non-empty path")
        if bg.a_file is not None and not str(bg.a_file).strip():
            raise ValueError("background_field.a_file must be a non-empty path")
        if bg.conductors is not None:
            if not isinstance(bg.conductors, list):
                raise ValueError("background_field.conductors must be a list")
            if not bg.conductors:
                raise ValueError(
                    "background_field.conductors must be a non-empty list")
        source_count = sum(source is not None for source in (
            bg.file, bg.a_file, bg.conductors))
        if source_count > 1 and bg.conductors is None:
            # Preserve the established file/a_file diagnostic verbatim.
            raise ValueError(
                "background_field: set at most one of 'file' (staggered B0 arrays) "
                "or 'a_file' (coil vector-potential A); they are mutually exclusive")
        if source_count > 1:
            raise ValueError(
                "background_field: set at most one of 'conductors' (inline coil "
                "geometry), 'file' (staggered B0 arrays), or 'a_file' (coil "
                "vector-potential A); they are mutually exclusive")
        if bg.conductors is not None:
            if self.units != "SI":
                raise ValueError(
                    "background_field.conductors requires units: SI because "
                    "conductor currents and geometry are evaluated in amperes "
                    "and meters")
            # Reuse the coil/PIC schema implementation so malformed geometry
            # fails while loading the deck, before the MHD solver is built.
            build_conductor_system(bg.conductors)
        if bg.file is not None and bg.params:
            raise ValueError(
                "background_field.params is not used with background_field.file")
        if bg.a_file is not None or bg.conductors is not None:
            unknown = set(bg.params) - {"b_scale", "vacuum_project"}
            if unknown:
                names = ", ".join(sorted(unknown))
                if bg.a_file is not None:
                    raise ValueError(
                        "unknown background_field.params key(s) for a_file: "
                        + names)
                raise ValueError(
                    "unknown background_field.params key(s) for conductors: "
                    + names)
        if analytic_mode and bg.profile != "uniform" and any(
                value != 0.0 for value in (bg.bx0, bg.by0, bg.bz0)):
            raise ValueError(
                "background_field.bx0/by0/bz0 are valid only for profile "
                "'uniform'")
        if bg.file is not None and any(
                value != 0.0 for value in (bg.bx0, bg.by0, bg.bz0)):
            raise ValueError(
                "background_field.bx0/by0/bz0 are not used with file input")
        if bg.a_file is not None and (bg.bx0 != 0.0 or bg.by0 != 0.0):
            raise ValueError(
                "background_field.bx0/by0 are not used with a_file input; "
                "bz0 is the supported uniform out-of-plane component")
        if bg.conductors is not None and (bg.bx0 != 0.0 or bg.by0 != 0.0):
            raise ValueError(
                "background_field.bx0/by0 are not used with conductors input; "
                "bz0 is the supported uniform out-of-plane component")
        # An ABSOLUTE file path's existence is knowable now (independent of the
        # deck directory), so reject an enabled background whose only source is a
        # non-existent absolute file at parse time -- an enabled block must name a
        # usable source. A RELATIVE file is resolved against the deck directory,
        # which is only known after load(); its existence + array-shape + div-free
        # checks happen in build_background_field(deck, nghost) once the solver's
        # ghost width and storage layout are available.
        for _key, _val in (("file", bg.file), ("a_file", bg.a_file)):
            if _val is None:
                continue
            _file_path = Path(str(_val))
            if _file_path.is_absolute():
                try:
                    _exists = _file_path.is_file()
                except OSError:
                    _exists = False  # unreadable / inaccessible path == not usable
                if not _exists:
                    raise ValueError(
                        f"background_field.{_key} {_val!r} does not exist")

    def _validate_cylindrical(self) -> None:
        # A cylindrical radial coordinate cannot be negative. At r_min=0 the
        # low side is the coordinate axis and needs the parity closure. For a
        # finite-inner-radius annulus it is an ordinary physical boundary and
        # must not use the axis closure.
        if self.domain.origin_x_m < 0.0:
            raise ValueError(
                "geometry 'cylindrical': domain.origin_x_m must be non-negative")
        if "axis" in self.boundary.fluid[1:] or "axis" in self.boundary.field[1:]:
            raise ValueError("the MHD 'axis' boundary is valid only on x_lo")
        fluid_axis = self.boundary.fluid[0] == "axis"
        field_axis = self.boundary.field[0] == "axis"
        if self.domain.origin_x_m == 0.0:
            if not fluid_axis or not field_axis:
                raise ValueError(
                    "geometry 'cylindrical' starting at r=0 requires "
                    "boundary.fluid.x_lo and boundary.field.x_lo to be 'axis'")
        elif fluid_axis or field_axis:
            raise ValueError(
                "geometry 'cylindrical' with origin_x_m > 0 is an annular domain; "
                "the x_lo boundary must be physical, not 'axis'")
        if ("periodic" in self.boundary.fluid[:2] or
                "periodic" in self.boundary.field[:2]):
            raise ValueError(
                "geometry 'cylindrical': the radial axis cannot be periodic")


def _check_registered(name: str, allowed: Sequence[str], context: str) -> None:
    if name not in set(allowed):
        raise ValueError(
            f"{context} {name!r} must be one of {sorted(set(allowed))}")


def _parse_domain(d: dict) -> Domain:
    d = _mapping(d, "domain")
    _reject_unknown(
        d, {"nx", "ny", "lx_m", "ly_m", "origin_x_m", "origin_y_m"},
        "domain")
    return Domain(
        nx=_as_exact_int(_require(d, "nx", "domain"), "domain.nx"),
        ny=_as_exact_int(_require(d, "ny", "domain"), "domain.ny"),
        lx_m=float(_require(d, "lx_m", "domain")),
        ly_m=float(_require(d, "ly_m", "domain")),
        origin_x_m=float(d.get("origin_x_m", 0.0)),
        origin_y_m=float(d.get("origin_y_m", 0.0)),
    )


def _parse_numerics(d: dict | None) -> Numerics:
    if d is None:
        return Numerics()
    d = _mapping(d, "numerics")
    _reject_unknown(
        d, {"gamma", "reconstruction", "riemann", "integrator", "ct",
            "positivity", "rho_floor", "p_floor", "cfl"}, "numerics")
    return Numerics(
        gamma=float(d.get("gamma", 5.0 / 3.0)),
        reconstruction=str(d.get("reconstruction", "mp7")),
        riemann=str(d.get("riemann", "hlld")),
        integrator=str(d.get("integrator", "ssprk3")),
        ct=str(d.get("ct", "fd_ct_christlieb")),
        positivity=str(d.get("positivity", "troubled_cell")),
        rho_floor=float(d.get("rho_floor", 1.0e-8)),
        p_floor=float(d.get("p_floor", 1.0e-9)),
        cfl=float(d.get("cfl", 0.4)),
    )


def _parse_initial(d: dict | None) -> Union[Initial, None]:
    if d is None:
        return None
    d = _mapping(d, "initial")
    _reject_unknown(d, {"type", "params"}, "initial")
    params = d.get("params", {})
    params = _mapping(params, "initial.params")
    return Initial(
        type=str(_require(d, "type", "initial")),
        params=dict(params),
    )


def _parse_time(d: dict | None) -> Time:
    if d is None:
        return Time()
    d = _mapping(d, "time")
    _reject_unknown(d, {"dt_s", "steps", "t_end"}, "time")
    dt_raw = d.get("dt_s", "auto")
    dt_s: Union[float, str] = dt_raw if isinstance(dt_raw, str) else float(dt_raw)
    t_end = d.get("t_end")
    return Time(dt_s=dt_s,
                steps=_as_exact_int(d.get("steps", 100), "time.steps"),
                t_end=None if t_end is None else float(t_end))


def _parse_diagnostics(d: dict | None) -> Diagnostics:
    if d is None:
        return Diagnostics()
    d = _mapping(d, "diagnostics")
    _reject_unknown(d, {"output_path", "cadence", "fields", "divb"},
                    "diagnostics")
    output_path = d.get("output_path", "out.npz")
    if not isinstance(output_path, str):
        raise ValueError("diagnostics.output_path must be a string")
    return Diagnostics(
        output_path=output_path,
        cadence=_as_exact_int(d.get("cadence", 0), "diagnostics.cadence"),
        fields=list(d.get("fields", list(STATE_COMPONENTS))),
        divb=_as_bool(d.get("divb", True), "diagnostics.divb"),
    )


def _parse_boundary(d: dict | None) -> BoundaryConfig:
    if d is None:
        return BoundaryConfig()
    d = _mapping(d, "boundary")
    _reject_unknown(d, {"fluid", "field"}, "boundary")
    for name in ("fluid", "field"):
        side_map = d.get(name)
        if isinstance(side_map, dict):
            _reject_unknown(side_map, {"x_lo", "x_hi", "y_lo", "y_hi"},
                            f"boundary.{name}")
    return BoundaryConfig(
        fluid=_parse_side_map(d.get("fluid"), "periodic", "boundary.fluid"),
        field=_parse_side_map(d.get("field"), "periodic", "boundary.field"),
    )


def _parse_background(d: dict | None) -> BackgroundConfig:
    # An absent ``background_field`` block leaves the static B0 disabled (the
    # solver runs its zero-B0 fast path). A present block defaults profile to
    # "uniform" and every uniform-vector component to 0.
    if d is None:
        return BackgroundConfig()
    d = _mapping(d, "background_field")
    _reject_unknown(
        d, {"enabled", "profile", "bx0", "by0", "bz0", "params", "file",
            "a_file", "conductors"}, "background_field")
    enabled = _as_bool(d.get("enabled", False), "background_field.enabled")
    # `enabled` is a true master switch: ignored values in a disabled block do
    # not participate in parsing, registry lookup, unit conversion, or native
    # validation. Canonicalize it to the same defaults as an absent block.
    if not enabled:
        return BackgroundConfig()
    file_raw = d.get("file")
    a_file_raw = d.get("a_file")
    conductors_raw = d.get("conductors")
    if conductors_raw is not None and not isinstance(conductors_raw, list):
        raise ValueError("background_field.conductors must be a list")
    params = _mapping(d.get("params", {}), "background_field.params")
    return BackgroundConfig(
        enabled=enabled,
        profile=str(d.get("profile", "uniform")),
        bx0=float(d.get("bx0", 0.0)),
        by0=float(d.get("by0", 0.0)),
        bz0=float(d.get("bz0", 0.0)),
        params=dict(params),
        file=None if file_raw is None else str(file_raw),
        a_file=None if a_file_raw is None else str(a_file_raw),
        conductors=(None if conductors_raw is None
                    else list(conductors_raw)),
    )


def parse(data: dict) -> MhdDeck:
    data = _mapping(data, "deck")
    _reject_unknown(data, _TOP_LEVEL_KEYS, "deck")
    deck = MhdDeck(
        domain=_parse_domain(_require(data, "domain", "deck")),
        numerics=_parse_numerics(data.get("numerics")),
        initial=_parse_initial(data.get("initial")),
        time=_parse_time(data.get("time")),
        diagnostics=_parse_diagnostics(data.get("diagnostics")),
        boundary=_parse_boundary(data.get("boundary")),
        background=_parse_background(data.get("background_field")),
        units=str(data.get("units", "normalized")),
        geometry=str(data.get("geometry", "cartesian")),
        raw=data,
    )
    deck.validate()
    return deck


def load(path: Union[str, Path]) -> MhdDeck:
    with open(path) as fh:
        data = _load_yaml(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    deck = parse(data)
    # Record the deck's directory so a background_field.file path resolves and is
    # confined relative to the deck (mirrors the CLI output-path confinement).
    deck.source_dir = Path(path).resolve().parent
    return deck


# =============================================================================
# Initial-condition generators
# =============================================================================
#
# The generators themselves are device kernels; this layer parses and validates
# the deck, lowers it to the native parameter block, and returns the resulting
# full ghost-padded host buffers (one per STATE_COMPONENT) in the solver's
# storage layout (pitch = nx + 2*nghost, height = ny + 2*nghost, row-major
# reshape(-1)), ready for solver.seed_state(). See
# quasar/physics/mhd/initial_conditions.hpp for what each generator computes and
# for the exactness of each analytic profile as an element moment.
#
# Storage convention (mirrors C++ MhdField2D / the solver seed contract):
#   * rho, mx, my, mz, energy : cell-centered conserved quantities.
#   * bx, by                  : seeded into the FACE-staggered slots (bx_face,
#                               by_face); for a UNIFORM or x-only-varying field
#                               the cell-centered analytic value sampled at the
#                               left/bottom face equals the cell value to the
#                               accuracy these smooth ICs need, and div B stays
#                               ~0.
#   * bz                      : cell-centered (bz_cell), out-of-plane toroidal.


def _validate_padded_cylindrical_domain(deck: MhdDeck, nghost: int) -> None:
    """Reject an annular reconstruction halo that reaches non-positive radius."""
    g = _as_exact_int(nghost, "nghost")
    if g < 0:
        raise ValueError("nghost must be non-negative")
    if deck.geometry != "cylindrical" or deck.domain.origin_x_m == 0.0:
        return
    dx = deck.domain.lx_m / deck.domain.nx
    padded_r_lo = deck.domain.origin_x_m - g * dx
    if not math.isfinite(padded_r_lo) or padded_r_lo <= 0.0:
        raise ValueError(
            "annular cylindrical geometry requires "
            "domain.origin_x_m - nghost*dr > 0 so the full "
            "reconstruction halo stays at positive radius")


def _initial_condition_spec(deck: MhdDeck, nghost: int):
    """Lower ``deck.initial`` to the native per-cell generator's parameters.

    Every value written here is a deck scalar or an O(1) reduction of deck
    scalars. Nothing per-cell is computed on this side of the boundary: the
    profile itself is evaluated by a kernel over the padded grid.
    """
    domain = deck.domain
    p = deck.initial.params
    spec = _core.mhd.MhdInitialConditionSpec()
    spec.kind = deck.initial.type
    spec.grid = _core.mhd.Grid2D(
        domain.nx, domain.ny, domain.lx_m, domain.ly_m,
        domain.origin_x_m, domain.origin_y_m, nghost)
    spec.gamma = deck.numerics.gamma
    spec.cylindrical = 1 if deck.geometry == "cylindrical" else 0
    spec.scheme_order = _core.mhd.reconstruction_order(
        deck.numerics.reconstruction)
    # SI decks give B in tesla; the solver evolves B/sqrt(mu0).
    spec.magnetic_scale = (1.0 / mhd_units.SQRT_MU0
                           if deck.units == "SI" else 1.0)

    if deck.initial.type == "brio_wu":
        spec.interface = float(p.get("interface", 0.5))
        order = ("rho", "p", "vx", "vy", "vz", "bx", "by", "bz")
        spec.left = [float(p["left"][key]) for key in order]
        spec.right = [float(p["right"][key]) for key in order]
    elif deck.initial.type == "alfven_wave":
        spec.rho = float(p.get("rho", 1.0))
        spec.pressure = float(p.get("p", 0.1))
        spec.b0 = float(p.get("b0", 1.0))
        spec.total_b0 = _alfven_reference_bx(deck)
        spec.amplitude = float(p.get("amplitude", 1.0e-6))
        spec.wavenumber = _as_exact_int(p.get("wavenumber", 1),
                                        "initial.params.wavenumber")
        # In SI the eigen-relation uses dB/sqrt(mu0*rho); a normalized deck
        # already carries B in mu0=1 units. Distinct from magnetic_scale: this
        # one turns dB into a velocity and also sets the sub-cell energy
        # correction, so the kernel needs both.
        spec.magnetic_velocity_scale = (1.0 / mhd_units.SQRT_MU0
                                        if deck.units == "SI" else 1.0)
    elif deck.initial.type == "orszag_tang":
        spec.b_uniform = [float(p.get("b0", 1.0)), 0.0, 0.0]
    elif deck.initial.type == "blast":
        center = p.get("center", [0.0, 0.0])
        spec.center = [float(center[0]), float(center[1])]
        spec.rho_ambient = float(p.get("rho_ambient", 1.0))
        spec.p_ambient = float(p.get("p_ambient", 0.1))
        spec.p_core = float(p.get("p_core", 10.0))
        spec.r_in = float(p.get("r_in", 0.1))
        spec.b_uniform = [float(p.get("bx", 0.0)), float(p.get("by", 0.0)),
                          float(p.get("bz", 0.0))]
    elif deck.initial.type == "rotor":
        center = p.get("center", [0.5, 0.5])
        spec.center = [float(center[0]), float(center[1])]
        spec.r0 = float(p.get("r0", 0.1))
        spec.r1 = float(p.get("r1", 0.115))
        spec.rho_in = float(p.get("rho_in", 10.0))
        spec.rho_out = float(p.get("rho_out", 1.0))
        spec.u0 = float(p.get("u0", 2.0))
        spec.p_ambient = float(p.get("p", 1.0))
        spec.b_uniform = [float(p.get("bx", 0.0)), float(p.get("by", 0.0)),
                          float(p.get("bz", 0.0))]
    elif deck.initial.type == "confined_blob":
        spec.center = [domain.origin_x_m + 0.5 * domain.lx_m,
                       domain.origin_y_m + 0.5 * domain.ly_m]
        spec.rho_in = float(p.get("rho_in", 10.0))
        spec.rho_out = float(p.get("rho_out", 1.0))
        spec.p_in = float(p.get("p_in", 1.0))
        spec.p_out = float(p.get("p_out", 0.1))
        spec.blob_half = float(
            p.get("blob_half", 0.25 * min(domain.lx_m, domain.ly_m)))
        spec.b_uniform = [0.0, 0.0, float(p.get("bz", 0.1))]
    return spec


def build_initial_state(deck: MhdDeck, nghost: int) -> dict:
    """Build the seeded conserved state for ``deck.initial.type``.

    Returns ``{component: 1-D host buffer}`` for every STATE_COMPONENT, ready to
    hand to ``solver.seed_state(component, buf)``.

    The analytic profile, the primitive-to-conserved assembly, the face-to-cell
    magnetic recollocation and the positivity preflight all run in a kernel over
    the padded grid; see ``quasar/physics/mhd/initial_conditions.hpp``. The host
    arrays returned here are a download at the output boundary, because the
    distributed runner slices them per tile and the deck tests compare them to
    closed-form references.
    """
    _validate_padded_cylindrical_domain(deck, nghost)
    return _core.mhd.build_initial_state(_initial_condition_spec(deck, nghost))


# =============================================================================
# Static background magnetic field B0 (field-split form B = B0 + b)
# =============================================================================

# Per-side field-closure code consumed by the native boundary-compatibility
# sweep, in side order (x_lo, x_hi, y_lo, y_hi). Mirrors
# background_boundary_mode in src/physics/mhd/mhd_solver.cpp: anything that is
# not periodic/wall/axis leaves the static samples unconstrained.
_BACKGROUND_BOUNDARY_MODES = {"periodic": 1, "wall": 2, "axis": 3}


def _background_build_spec(deck: MhdDeck, nghost: int):
    """Lower ``deck.background`` to the native builder's parameter block."""
    domain = deck.domain
    bg = deck.background
    spec = _core.mhd.MhdBackgroundBuildSpec()
    spec.grid = _core.mhd.Grid2D(
        domain.nx, domain.ny, domain.lx_m, domain.ly_m,
        domain.origin_x_m, domain.origin_y_m, nghost)
    spec.cylindrical = 1 if deck.geometry == "cylindrical" else 0
    # SI decks provide B0 in tesla (and A in T m); the solver stores B/sqrt(mu0)
    # so magnetic pressure and Alfven speed use the normalized mu0=1 equations.
    spec.magnetic_scale = (1.0 / mhd_units.SQRT_MU0
                           if deck.units == "SI" else 1.0)
    spec.field_modes = [_BACKGROUND_BOUNDARY_MODES.get(side, 0)
                        for side in deck.boundary.field]
    spec.b_scale = float(bg.params.get("b_scale", 1.0))
    spec.bz0 = float(bg.bz0)
    spec.vacuum_project = 1 if bool(bg.params.get("vacuum_project", False)) else 0
    return spec


def build_background_field(deck: MhdDeck, nghost: int) -> Union[dict, None]:
    """Build the static background field B0 for ``deck.background``.

    Returns ``{"b0x": buf, "b0y": buf, "b0z": buf}`` of 1-D ghost-padded host
    buffers (length ``(nx+2g)*(ny+2g)``, row-major like the initial state),
    ready for ``solver.seed_background(component, buf)``; or ``None`` when the
    background is disabled (the solver then runs its zero-B0 fast path).

    Five construction modes (mutually selected by the deck):

    * **uniform** (``profile == "uniform"``, no explicit source): constant
      components ``b0x == bx0``, ``b0y == by0``, ``b0z == bz0`` everywhere.
    * **profile** (named ``profile``, no explicit source): sample the analytic
      profile over the padded staggered meshes (b0x at the left face, b0y at the
      bottom face, b0z at cell centres). Registered profile parameters are
      forwarded to the profile object before it is lowered for the device.
    * **file** (``file:`` given): ``np.load`` the npz and read arrays ``b0x``,
      ``b0y``, ``b0z`` each shaped ``(ny+2g, nx+2g)`` or flat ``(storage,)``.
    * **a_file** (``a_file:`` given): load a coil-CLI ``A_xyz_grid`` corner
      sample and take its discrete curl.
    * **conductors** (``conductors:`` given): evaluate the inline coil geometry
      on the derived padded corner grid and take the same discrete curl, with no
      host round trip between the evaluation and the curl.

    In ALL modes the assembled field's discrete face-divergence and its
    compatibility with the configured field closure are checked on device before
    anything is returned; a failure raises ``ValueError``. See
    ``quasar/physics/mhd/background_builder.hpp``.
    """
    _validate_padded_cylindrical_domain(deck, nghost)
    bg = deck.background
    if not bg.enabled:
        return None

    g = nghost
    pitch = deck.domain.nx + 2 * g
    height = deck.domain.ny + 2 * g
    shape = (height, pitch)
    spec = _background_build_spec(deck, nghost)

    if bg.conductors is not None:
        conductors = build_conductor_system(bg.conductors)
        evaluator = _core.magnetostatics.create_field_evaluator("biot_savart")
        evaluator.configure({})
        return _core.mhd.build_background_from_conductors(
            spec, conductors, evaluator)
    if bg.a_file is not None:
        # Reading the archive is file I/O, not calculation; everything done to
        # the loaded potential afterwards is a kernel.
        return _core.mhd.build_background_from_corner_potential(
            spec, _a_corners_from_file(deck, shape).reshape(-1))
    if bg.file is not None:
        b0x, b0y, b0z = _background_from_file(
            bg.file, deck.source_dir, shape, pitch * height)
        return _core.mhd.build_background_from_arrays(
            spec, b0x.reshape(-1), b0y.reshape(-1), b0z.reshape(-1))

    params = dict(bg.params)
    if bg.profile == "uniform":
        params.update(bx0=float(bg.bx0), by0=float(bg.by0), bz0=float(bg.bz0))
    try:
        spec.set_profile(bg.profile, params)
    except ValueError as exc:
        raise ValueError(
            f"background_field.profile {bg.profile!r} could not be sampled: "
            f"{exc}") from None
    return _core.mhd.build_background_from_profile(spec)


def _a_corners_from_file(deck: MhdDeck, shape):
    """Load the lab-Y corner potential from a coil-CLI ``A_xyz_grid`` npz."""
    bg = deck.background
    height, pitch = shape                     # (ny+2g, nx+2g)

    file_rel = str(bg.a_file)
    base = (Path(deck.source_dir).resolve()
            if deck.source_dir is not None else Path.cwd())
    file_path = Path(file_rel)
    path = file_path.resolve() if file_path.is_absolute() \
        else (base / file_rel).resolve()
    if not file_path.is_absolute() and not path.is_relative_to(base):
        raise ValueError(
            f"background_field.a_file {file_rel!r} escapes the deck directory {base}")
    if not path.is_file():
        raise ValueError(
            f"background_field.a_file {file_rel!r} not found at {path}")
    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            f"background_field.a_file {file_rel!r} could not be read as an npz: "
            f"{exc}") from None
    if not hasattr(loaded, "files") or not hasattr(loaded, "close"):
        raise ValueError(
            f"background_field.a_file {file_rel!r} is not an npz archive")
    with loaded:
        if "A_xyz_grid" not in loaded:
            raise ValueError(
                f"background_field.a_file {file_rel!r} is missing array "
                "'A_xyz_grid' (run the coil CLI with output.fields including "
                "'A_xyz_grid')")
        a = np.asarray(loaded["A_xyz_grid"], dtype=np.float64)
    expected = (height + 1, 1, pitch + 1, 3)
    if a.shape != expected:
        raise ValueError(
            f"background_field.a_file {file_rel!r} array 'A_xyz_grid' has shape "
            f"{a.shape}; expected {expected} (the (Nx+1)x(Ny+1) corner grid over "
            f"the PADDED domain Nx=nx+2g={pitch}, Ny=ny+2g={height} on the lab Y=0 "
            "slice, coil resolution [Nx+1, 1, Ny+1])")
    a_ji = a[:, 0, :, 1]                       # (height+1, pitch+1), [j, i]
    if not np.all(np.isfinite(a_ji)):
        raise ValueError(
            f"background_field.a_file {file_rel!r} A_xyz_grid has non-finite values")
    return a_ji


def _background_from_file(file_rel: str, source_dir, shape, storage):
    """Load b0x/b0y/b0z from an npz.

    Path resolution: a RELATIVE ``file`` is resolved against (and confined to) the
    deck's directory ``source_dir`` when the deck was loaded from disk -- this is
    the security confinement for decks that name a sibling B0 file. An ABSOLUTE
    ``file`` is honored as given (a deck/program that supplies a full path is
    trusted to point where it means to; confinement only constrains relative
    names). When ``source_dir`` is unknown (a deck constructed in memory rather
    than loaded), a relative path resolves against the current directory.

    Each array must be either the 2-D ghost-padded shape ``(ny+2g, nx+2g)`` or
    flat ``(storage,)``; anything else is a ``ValueError``. Returns three 2-D
    ``shape`` arrays.
    """
    file_path = Path(file_rel)
    if file_path.is_absolute():
        path = file_path.resolve()
    else:
        base = Path(source_dir).resolve() if source_dir is not None else Path.cwd()
        path = (base / file_rel).resolve()
        if not path.is_relative_to(base):
            raise ValueError(
                f"background_field.file {file_rel!r} escapes the deck "
                f"directory {base}")
    if not path.is_file():
        raise ValueError(f"background_field.file {file_rel!r} not found at {path}")
    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception as exc:  # malformed / non-npz file
        raise ValueError(
            f"background_field.file {file_rel!r} could not be read as an npz: "
            f"{exc}") from None
    if not hasattr(loaded, "files") or not hasattr(loaded, "close"):
        raise ValueError(
            f"background_field.file {file_rel!r} is not an npz archive")
    out = []
    with loaded:
        for name in ("b0x", "b0y", "b0z"):
            if name not in loaded:
                raise ValueError(
                    f"background_field.file {file_rel!r} is missing array "
                    f"{name!r} (needs b0x, b0y, b0z)")
            arr = np.asarray(loaded[name], dtype=np.float64)
            if arr.shape == shape:
                out.append(arr.copy())
            elif arr.shape == (storage,):
                out.append(arr.reshape(shape).copy())
            else:
                raise ValueError(
                    f"background_field.file {file_rel!r} array {name!r} has "
                    f"shape {arr.shape}; expected {shape} or ({storage},)")
    if not np.all(np.isfinite(out[0])) or not np.all(np.isfinite(out[1])) \
            or not np.all(np.isfinite(out[2])):
        raise ValueError(
            f"background_field.file {file_rel!r} contains non-finite values")
    return out[0], out[1], out[2]
